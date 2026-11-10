---
title: "25 Common Issues — Foundations, Computing and Problem Solving"
topic: "Diagnostic catalogue for evidence, arithmetic, scale and decision-authority failures"
collection: "01_foundations_computing_and_problem_solving"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — Foundations, Computing and Problem Solving

> **Evidence declaration.** The arithmetic in this sheet is DERIVED and was recomputed at the
> precision shown. The *incidents* it describes are constructed teaching scenarios: no organisation,
> deployment, or measured outage is described. Where a number appears it is labelled ILLUSTRATIVE
> and exists only so that a diagnostic can be checked against an expected magnitude. A reader must
> not cite any figure here as an observed benchmark.

## 1. File Header

### 1.1 Scope

This file catalogues twenty-five recurring defects that appear whenever a measured claim is turned
into an operating decision. The scope is deliberately narrow: it covers what goes wrong between an
observation and the action taken on it, not what goes wrong inside a specific algorithm. A defect in
a gradient-descent implementation is out of scope; a defect in how its convergence was reported is
in scope.

The four defect families are:

| Category | Definition | Typical cost |
|---|---|---|
| VIOLATED_ASSUMPTION | The analysis or system silently relies on a precondition that did not hold | wrong decision, correct arithmetic |
| NUMERICAL_EDGE | A quantity is computed at a boundary where its arithmetic is undefined or lossy | wrong number, correct logic |
| SCALE_FAILURE | A cost or a rate grows faster than the design assumed | late failure, after a green pilot |
| CONCEPT_CONFUSION | Two distinct quantities are treated as interchangeable | unfalsifiable claim |

The taxonomy matters because the four families fail in different places. A VIOLATED_ASSUMPTION
survives every test written by the person who holds the assumption. A NUMERICAL_EDGE survives every
test written at typical values. A SCALE_FAILURE survives every test run at pilot volume. A
CONCEPT_CONFUSION survives every test that measures the wrong quantity precisely.

### 1.2 Reader task

For any symptom, the reader must be able to: classify it into one of the four families; write down at
least two competing mechanisms that produce the identical symptom; select a check that separates
them without needing a fix first; contain harm with something reversible; correct the mechanism
rather than the symptom; and replay a specific record plus a boundary record to verify.

The sequence is fixed and skipping a step is what turns a diagnosis into a guess. Containment before
diagnosis is sometimes right — during an active incident it usually is — but containment is never
verification.

### 1.3 Conventions used in the catalog

Every issue carries eight fields, in this order: identifier, category, symptom, competing causes,
discriminating check, correction, prevention, verification. The fields are defined once here and
never re-explained in the entries:

- **Symptom** is what a person can observe without instrumentation of the cause. Symptoms are
  deliberately coarse: "the benefit did not materialise" is a symptom, "the verification term was
  omitted" is a cause.
- **Competing causes** are at least two mechanisms consistent with the symptom. An entry with a
  single listed cause is an entry where the author stopped thinking early.
- **Discriminating check** is an observation that comes out differently under the competing causes. It
  must be runnable before any correction, otherwise it cannot discriminate.
- **Correction** removes the mechanism. **Prevention** changes a contract so the mechanism cannot
  recur.
- **Verification** replays a specific triggering record *and* a nearby boundary record, because a fix
  that works on the trigger alone is often a fix that has simply moved the boundary.

Units appear in square brackets after every symbol at first use, in the form `symbol [unit]`. A
quantity with no unit in a calculation is a defect in its own right (see CI-06 and CI-17).

### 1.4 Severity scale

Severity is the consequence within the declared scope, and it is assigned from the decision the
defect corrupts, not from how alarming the symptom looks.

| Level | Meaning | Response window |
|---|---|---|
| S1 | produces an irreversible effect with no audit record | stop the path, correct before any further traffic |
| S2 | produces a wrong decision that is reversible and recorded | correct within the current window |
| S3 | produces a misleading number that a human still reviews | correct before the number is cited |
| S4 | produces friction or rework with no decision effect | schedule |

The scale is ordinal and the boundaries are judgement calls, which is why the response window rather
than the label is the operative part. A defect labelled S2 that nobody corrects inside the window has
become S1 by neglect.

### 1.5 How this file relates to the rest of the unit

The case study in this collection (`Case_Studies/01_Cross_Module_Evidence_to_Operation.md`) works one
transfer function in full: elapsed minutes to net reviewer-minutes. Several issues below are the
failure modes of that case stated generically, so a reader who has worked the case will recognise
CI-05, CI-12, CI-13 and CI-25 as variations on its Section 12. The remaining issues are drawn from
the same foundation topics — arithmetic boundaries, queueing, rate arithmetic, and audit
completeness — without depending on the case.

**Takeaway:** a catalogue is useful only if its entries are distinguishable; four families and a
fixed field set are what make them so.

## 2. Concept Map / Mental Model

### 2.1 The diagnosis loop

```text
        symptom observed
              |
              v
      classify family  --------------------------------+
              |                                        |
              v                                        |
   enumerate >=2 competing mechanisms                  |
              |                                        |
              v                                        |
   choose a check that differs under each  -------------+ (check not
              |                                          discriminating:
              v                                          return here)
   contain with a reversible action                    |
              |                                        |
              v                                        |
   correct the mechanism, not the symptom              |
              |                                        |
              v                                        |
   replay trigger + boundary record -------------------+
              |
              v
   change the contract so it cannot recur
```

The loop's arrow back from verification is the important one. A replay that passes the trigger record
but fails a boundary record means the correction is narrower than the mechanism, and the loop must
return to mechanism enumeration rather than to correction.

### 2.2 The mental model: four layers that fail independently

Every evidence-to-operation pipeline has four layers, and a defect in each produces a
characteristically different symptom shape:

```text
  [1] RECORD        what was written, when, by whom        -> completeness defects
  [2] JOIN          which version of each field applies    -> as-of defects
  [3] COMPUTE       arithmetic and boundaries              -> numeric defects
  [4] INTERPRET     which quantity the number represents   -> semantic defects
  [5] ACT           who is allowed to act on it            -> authority defects
```

A record defect produces missing rows or impossible values. A join defect produces values that are
individually correct and jointly wrong. A compute defect produces a number that is wrong at a
boundary but right at the centre. An interpret defect produces a number that is arithmetically
perfect and answers a different question. An authority defect produces a correct recommendation with
an unaccountable effect.

These five layers are independent, which is the property that makes diagnosis tractable: a defect at
layer 3 does not disturb layer 1, so the check for layer 1 (counts and timestamps) is cheap and can
be run first.

### 2.3 Triage order

The order below is not arbitrary; each step is cheap and each can eliminate a whole layer.

1. **Count rows.** Extract count versus load count. A mismatch makes every downstream rate undefined
   and stops the investigation at layer 1.
2. **Check monotonicity.** For each case, does `opened_at ≤ closed_at`? Any violation stops at
   layer 1 with a timestamp defect.
3. **Reconcile one record by hand.** Pick a single row and reconstruct its numbers from the raw
   source. If the hand calculation disagrees, the defect is at layer 3 and needs no statistics.
4. **Compare two versions.** Join the same record against the as-of view and the latest view. A
   difference localises layer 2.
5. **Name the quantity in words.** "This is the mean of per-window rates" versus "this is the ratio
   of summed events". If the sentence does not match the formula, stop at layer 4.
6. **Check the actor field.** For every final action, is there a named human or an approved policy
   version? A blank actor localises layer 5.

### 2.4 Why symptoms collide

Three different mechanisms routinely produce "the dashboard shows an improvement that operations does
not feel". They are: the metric improved but the population changed (layer 2); the metric improved in
mean but not in the tail that constrains the process (layer 3, percentile defect); and the metric
measures a quantity that operations does not pay for (layer 4). Each has a distinct check and none is
distinguishable from the others by staring at the dashboard.

That collision is the reason this file insists on two competing causes per entry: a single-cause
diagnosis is a diagnosis that has not excluded the most likely alternative.

**Takeaway:** triage from the cheapest layer upward, and never accept a symptom as an identification.

## 3. Foundational Definitions

Each definition below is the operational one used by the catalog. Where a word has a looser everyday
meaning, the operational meaning is stated and the divergence is noted.

**Observation.** A value that was recorded by a system or a person at a specific time. An observation
has a source, a write time, and a schema version. Anything without all three is a recollection, not
an observation.

**Claim.** A statement that some quantity has a value, or that some action caused a change. A claim
must name its unit, population, window, and comparator to be testable.

**Comparator.** The state against which a claim is evaluated. Three comparators exist and they are
not interchangeable: the previous period (time comparator), a held-out population (population
comparator), and a do-nothing path (counterfactual comparator). A claim evaluated against a time
comparator cannot establish causation.

**Cut-off (`decision_cutoff_at`).** The instant at which the decision must be made, in UTC with an
explicit offset. Fields written after the cut-off are unavailable at decision time even if they are
present in the extract.

**Event time.** When the thing happened in the world. **Processing time.** When the system handled
it. The two differ by pipeline latency and by clock error, and every duration question must say which
one it means (see CI-14 on window definition).

**Grain.** The entity one row represents. A table whose grain is "one row per case per attempt" will
double-count cases if aggregated without deduplication, which is a defect in the query rather than in
the data.

**Denominator.** The population over which a rate is computed. A rate with an unstated denominator
has no meaning, and a rate with a zero denominator does not exist (CI-02).

**Utilisation (`rho`).** Offered work divided by sustainable capacity: `rho = lambda / mu`, both in
items per second. Dimensionless, valid on `[0, 1)` for a stable queue. At `rho >= 1` the queue has no
stationary distribution and any wait estimate is an artefact.

**Service time (`T_s`).** Time a server spends actively on one item, in seconds or minutes. Distinct
from **response time** (which includes waiting) and from **elapsed time** (which includes both plus
any external delay).

**Touch time.** Time a human actively spends. A subset of service time when the server is a person.
Touch time is the quantity that constrains a roster; elapsed time is the quantity a case feels.

**Backlog (`B`).** Items accepted but not completed. Distinct from **queue length**, which counts only
items currently waiting for a server, excluding those in service.

**Idempotency key.** A value that identifies a logical operation so that a repeat submission is
recognised as the same operation. It must be derived from stable inputs, never from a timestamp
(CI-05).

**Effect.** A change outside the system: a payment, a disposition, a notification. An effect count is
the only meaningful check for duplicate work, because internal counters can be rolled back and
effects cannot.

**Invariant.** A property that must hold in every reachable state, stated so it can be tested. "At
most one final action per case key" is an invariant. "The system behaves correctly" is not.

**Guardrail.** A bound that an optimisation must not cross. A guardrail differs from a target: a
target is optimised toward, a guardrail is never traded away.

**Leakage.** Use of information at decision time that was not available at decision time. Two forms
exist and they are distinct defects: a field written after the cut-off (CI-01) and a join that
selects the current version of a slowly changing dimension (CI-09).

**As-of join.** A join that selects the dimension row whose validity interval contains the cut-off,
rather than the most recent row. Requires the dimension to carry `valid_from` and `valid_to`.

**Selection effect.** A difference between the analysed population and the target population,
introduced by how membership was chosen. A pilot staffed by volunteers has a selection effect on
tenure, motivation, and case familiarity simultaneously.

**Rounding half-even.** Ties round to the nearest even digit. Differs from half-up at exactly the
tie: 0.5 becomes 0, 1.5 becomes 2. Displayed precision is not the same as computed precision (CI-10).

**Numerical undefined.** A result that has no value: division by zero, `0/0`, a rate over an empty
window. It is not zero, and substituting zero converts an unknown into a reassuring number (CI-02).

**Not-measured.** A status meaning no observation exists. Distinct from **zero** (measured, and the
value was none) and from **unknown** (an observation may exist but is not in hand). Conflating these
three is CI-08's mechanism.

**Approximation contract.** The stated error bound of an approximate answer, plus the consumer's
tolerance. Without both, an estimate is indistinguishable from a measurement (CI-20).

**Audit sample.** A random subset of decisions re-adjudicated by a second party. Provides the only
unbiased estimate of agreement, and its size determines its own precision (CI-25).

**Rollback path.** A pre-tested sequence that returns the system to its prior state without a deploy.
A rollback that requires a release is a plan, not a path.

**Takeaway:** precision in these twenty-two terms removes most of the ambiguity that makes symptom
reports unactionable.

## 4. Core Formulas with Full Derivation

This section derives the five quantities the catalogue uses to predict an expected magnitude. Each
derivation follows the five-part anatomy. The purpose is not to teach queueing or statistics but to
make every entry's "expected signal" numerical, so a diagnostician can tell a small deviation from a
structural one.

### 4.1 Base case: the closed-evidence invariant

Before any arithmetic, the catalog's base case is a mechanism, not a formula.

**Preconditions.**
- (P1) Every field used in the computation has a recorded write time.
- (P2) The decision record carries `decision_cutoff_at` with an explicit UTC offset.
- (P3) For every field, `write_time <= decision_cutoff_at`.
- (P4) Every aggregate names its grain and its denominator.
- (P5) Every final action carries an actor identity or an approved policy version.

**State transition.** A decision attempt moves through `validated → computed → gated → acted`. At each
transition the record accumulates fields; the transition is only legal if the accumulated field set
still satisfies P3.

**Invariant and justification.** *No field written after the cut-off may influence a decision.* The
justification is causal rather than procedural: the decision could not have used a value that did not
yet exist, so any dependence on that value makes the decision unreproducible at the time it was
taken. The invariant is checkable without knowing the domain, because it is a comparison of two
timestamps and a dependency list.

**Worked trace.** A case closes at 14:20 UTC. Its `final_disposition` is written at 14:20. The
decision cut-off for the same case was 11:00. A feature set built at build time 15:00 includes
`final_disposition` and achieves 0.94 AUC on a held-out split. The trace shows the invariant failing
at the field level: `14:20 > 11:00`, so the field is unavailable, so the 0.94 measures the pipeline's
knowledge after the fact and not the decision's.

**Failure boundary.** The invariant says nothing about fields written *before* the cut-off that are
nonetheless revised later. A field whose value at cut-off was 3 and whose current value is 7 passes
P3 and fails the intent, which is why CI-09 exists as a separate entry.

**Validity condition.** The invariant is valid whenever a decision has a defined cut-off. It becomes
vacuous when no cut-off is recorded, and that absence is itself the defect.

### 4.2 General case: backlog under rate imbalance

```text
FORMULA [CI-F1]:  B(t) = max(0, B_0 + (lambda - mu) * t)
WHERE:  B(t)     backlog after elapsed time t                  [items]
        B_0      initial backlog, >= 0                          [items]
        lambda   arrival rate                                   [items/s]
        mu       sustainable service rate                       [items/s]
        t        overload duration                              [s]
CONSTRAINT: single service class; rates constant over t; no abandonment, no loss,
            no retry amplification, and mu is the sustainable rate rather than a peak.
NUMERICAL EXAMPLE: lambda = 2.0 items/s and mu = 1.5 items/s for t = 240 s from B_0 = 0
        B = (2.0 - 1.5) * 240 = 0.5 * 240 = 120 items
INTERPRETATION: the backlog grows linearly at the rate imbalance, not at the arrival
        rate. Doubling the imbalance doubles the growth; doubling the duration also
        doubles it, and the two effects compose multiplicatively.
```

The linearity is the diagnostic value: a measured backlog that grows faster than
`(lambda - mu) * t` implies a mechanism outside the model — retries (CI-23), fan-out (CI-15), or a
service rate that degrades as the queue lengthens.

The drain side needs its own statement because it is where the model is most often misapplied:

```text
FORMULA [CI-F2]:  T_drain = B / (mu_r - lambda_r)
WHERE:  T_drain  time to return to zero backlog                   [s]
        B        backlog to clear                                [items]
        mu_r     recovery service rate, > lambda_r               [items/s]
        lambda_r arrival rate during recovery                     [items/s]
CONSTRAINT: mu_r > lambda_r strictly. At equality the denominator is zero and the
            backlog does not drain; below equality it grows.
NUMERICAL EXAMPLE: B = 120 items, mu_r = 2.4 items/s, lambda_r = 2.0 items/s
        T_drain = 120 / (2.4 - 2.0) = 120 / 0.4 = 300 s
INDEPENDENT CHECK: surplus 0.4 items/s * 300 s = 120 items, matching B.
INTERPRETATION: the recovery surplus, not the recovery rate, sets the drain time.
        Raising mu_r from 2.4 to 2.8 halves T_drain to 150 s; raising it from 2.4 to
        2.44 raises T_drain tenfold to 2,727 s.
```

That last comparison is the useful one. A 1.7% increase in service rate changes the drain time by a
factor of 9.1, which is why operators who "add a little capacity" during a recovery are often
surprised by the outcome. The reciprocal relationship is invisible in any dashboard that plots
service rate.

**Validity condition for F1 and F2.** Both require constant rates over the interval. Neither is valid
when arrivals are non-stationary across the interval, which is exactly the case during an incident
with retry feedback.

### 4.3 Variants: four secondary quantities

#### 4.3.1 Variant A — retry amplification

```text
FORMULA [CI-F3]:  A = 1 / (1 - p)          and    L_eff = lambda * A * f
WHERE:  A        expected attempts per logical request           [attempts/request]
        p        per-attempt failure probability, 0 <= p < 1      [dimensionless]
        f        downstream calls per attempt                     [calls/attempt]
        L_eff    effective downstream load                        [calls/s]
        lambda   logical request rate                             [requests/s]
CONSTRAINT: attempts are independent; retries continue until success; no retry budget
            or circuit breaker; failures are not correlated in time.
NUMERICAL EXAMPLE: p = 0.5, f = 4, lambda = 2.0
        A = 1 / (1 - 0.5) = 2.0 attempts per request
        L_eff = 2.0 * 2.0 * 4 = 16 calls/s   against a nominal 2.0 * 4 = 8 calls/s
INTERPRETATION: at a 50% failure rate the downstream sees twice its nominal load,
        and the failure rate it observes is itself inflated by that load.
```

The feedback term is where this variant earns its place. If the downstream's failure probability rises
with load, `p` is a function of `L_eff`, so the equation is a fixed point rather than an evaluation.
At `p = 0.8` the multiplier is 5.0; at `p = 0.9` it is 10.0; at `p = 0.95` it is 20.0. The curve is
flat until roughly `p = 0.5` and then rises without bound, which is why retry storms appear sudden.

**Boundary.** At `p = 1` the expected attempt count is undefined rather than infinite, and any
implementation will exhaust a budget or a timeout. That budget, not the formula, is what bounds the
load in practice.

#### 4.3.2 Variant B — mean of ratios versus ratio of sums

```text
FORMULA [CI-F4]:  r_pooled = sum(x_i) / sum(n_i)      versus    r_mean = (1/k) * sum(x_i / n_i)
WHERE:  x_i      event count in window i                          [events]
        n_i      exposure count in window i                       [items]
        k        number of windows                                [windows]
        r_pooled ratio of sums, the correct pooled rate           [events/item]
        r_mean   unweighted mean of per-window rates              [events/item]
CONSTRAINT: r_mean equals r_pooled only when all n_i are equal. Otherwise r_mean
            weights a 10-item window exactly as heavily as a 10,000-item window.
NUMERICAL EXAMPLE: window 1 has 10 events in 100 items; window 2 has 2 events in 1,000
        r_pooled = (10 + 2) / (100 + 1,000) = 12 / 1,100 = 0.010909  (1.09%)
        r_mean   = (0.10 + 0.002) / 2 = 0.102 / 2 = 0.051          (5.10%)
        ratio    = 0.051 / 0.010909 = 4.676
INTERPRETATION: the mean-of-ratios overstates the pooled rate by 4.68x here because
        the small window carries equal weight and a tenfold higher rate.
```

**Boundary.** The two agree exactly when all `n_i` are equal, so a test built on equal-volume windows
will never expose this defect. The discriminating test is to construct two windows with volumes in a
1:10 ratio, which is why the check in this subsection specifies that construction rather than "review the
query".

#### 4.3.3 Variant C — unit conversion

```text
FORMULA [CI-F5]:  MiB = bytes / 1,048,576        MB = bytes / 1,000,000
                  overstatement = 1,048,576 / 1,000,000 = 1.048576
WHERE:  bytes    an exact byte count                              [B]
        MiB      binary mebibytes, 2^20 bytes                     [MiB]
        MB       decimal megabytes, 10^6 bytes                    [MB]
CONSTRAINT: the two are different units with different sizes; a value cannot be
            relabelled from one to the other without a factor of 1.048576.
NUMERICAL EXAMPLE: 1,000,000,000 bytes
        = 1,000,000,000 / 1,048,576 = 953.674 MiB
        = 1,000,000,000 / 1,000,000 = 1,000.000 MB
        labelling the MiB figure as MB understates capacity by 4.63% (1 - 953.674/1,000)
INTERPRETATION: the error is small enough to pass review and large enough to exceed a
        provisioned quota, since quotas are enforced in one unit and reported in the other.
```

**Boundary.** At small magnitudes the absolute error is negligible — 4.63% of 10 MB is 0.46 MB — so
the defect only becomes material when the quantity is near a hard limit. It therefore appears first
as a capacity incident, never as a units bug report.

#### 4.3.4 Variant D — sampling error of an audited rate

```text
FORMULA [CI-F6]:  SE(r_hat) = sqrt(r_hat * (1 - r_hat) / n)
                  half_width_95 = 1.96 * SE          (large-sample normal approximation)
WHERE:  r_hat    observed proportion in the sample               [dimensionless, 0..1]
        n        sample size                                     [items]
        SE       standard error of the proportion                [dimensionless]
CONSTRAINT: simple random sample from the target population; n * r_hat >= 5 and
            n * (1 - r_hat) >= 5; sampling fraction small relative to the population.
NUMERICAL EXAMPLE: r_hat = 0.12, n = 216
        SE = sqrt(0.12 * 0.88 / 216) = sqrt(0.1056 / 216) = sqrt(0.00048889) = 0.022111
        half_width_95 = 1.96 * 0.022111 = 0.043338
        interval = [0.0767, 0.1633]  i.e. 7.67% to 16.33%
INTERPRETATION: a 12% observed exception rate on 216 audited cases is consistent with a
        true rate anywhere from 7.7% to 16.3%. A 12% rollback threshold compared against
        the point estimate would therefore be crossed by noise roughly half the time.
```

**Boundary.** At `n = 216` and `r_hat = 0.12` the interval spans 8.7 percentage points. To halve that
width the sample must quadruple to 864 cases, because the width scales as `1/sqrt(n)`. This is the
arithmetic behind the statement that audit samples are almost always too small to support a threshold
comparison.

**Validity condition for §4.3.** Each variant holds only within the stated constraint; A requires
independence of attempts, B requires unequal exposure to be the failure mode under study, C requires an
unambiguous byte count, and D requires a simple random sample. None of the four is valid when the
sample is self-selected — CI-16 — and that is the most common way these otherwise-correct formulas
produce wrong conclusions.

### 4.4 Limits and counterexamples

Four limits bound the whole section. Each is a place where the formula still evaluates and the answer
still looks reasonable, which is why they are listed as counterexamples rather than caveats.

**Limit 1 — `rho → 1`.** In any formula containing `rho / (1 - rho)`, the value diverges as
utilisation approaches one. At `rho = 0.5` the factor is 1.0; at 0.9 it is 9.0; at 0.99 it is 99.0. A
queue reported at `rho = 0.95` with a wait of 4 minutes will report 12 minutes at `rho = 0.983` for
the same service time. No operational conclusion should be drawn from a factor computed at
`rho > 0.9`, because at that point the model is dominated by its own asymptote rather than by the
system.

**Limit 2 — the zero-denominator rate.** `x / 0` is undefined for every `x` including zero. The
counterexample is instructive: a service that received no requests and returned no errors has an error
rate of *undefined*, not 0%, and an alert written as `error_rate > 5%` will not fire. The alerting
logic is arithmetically correct and operationally blind, which is why CI-02's correction is to emit an
explicit undefined token rather than a sentinel value.

**Limit 3 — the NaN comparison.** Every ordered comparison with NaN returns false. So
`x > threshold` is false, `x <= threshold` is false, and `x == x` is false. A filter written as "keep
rows where `latency <= 250`" silently drops NaN rows, while a filter written as "keep rows where not
(`latency > 250`)" keeps them. Two logically equivalent predicates differ on exactly the NaN
population, so the row counts diverge and each reviewer believes the other's query is wrong (CI-22).

**Limit 4 — the saturated retry fixed point.** F3 has no finite solution when `p` depends positively
on load and `p(load) → 1` as load grows. The system then has two equilibria: one at low load with low
`p`, one at maximum load with `p` near 1. A retry budget removes the second equilibrium by making the
attempt count finite; nothing else does. The counterexample to "add more retries for resilience" is
therefore structural, not a matter of tuning the backoff.

**Takeaway:** every formula in this section is valid inside a stated constraint and produces a
plausible number outside it — which is why the constraint is part of the formula, not a footnote.

## 5. Worked Numerical Example

### 5.1 The scenario

An ILLUSTRATIVE intake pipeline processes 1,200 cases per business day inside a 10-hour operating
window with 60 review seats. Baseline service time is 15.4 minutes per case. Over a 20-business-day
window a change reduces service time to 11.1 minutes, but adds 2.5 minutes of verification per case
and a 6% audit sample costing 4.0 minutes per sampled case.

The task is to compute the net capacity release per business day, check it three independent ways,
and then determine how the answer moves under four perturbations.

### 5.2 Primary calculation

```text
STEP 1  Convert volume to a rate.
        lambda = 1,200 cases / (10 h * 3,600 s/h) = 1,200 / 36,000 = 0.033333 cases/s

STEP 2  Convert one seat's service time to a rate.
        T_s = 15.4 min = 924 s  ->  1/924 = 0.0010823 cases/s per seat
        60 seats -> mu = 60 * 0.0010823 = 0.0649351 cases/s

STEP 3  Utilisation.
        rho = lambda / mu = 0.033333 / 0.0649351 = 0.513333

STEP 4  Net release per case, in minutes.
        touch_before = 15.4 min
        touch_after  = decision 8.6 + verification 2.5 = 11.1 min
        g_net = 15.4 - 11.1 = 4.3 min/case
        (verification is subtracted exactly once, as part of touch_after)

STEP 5  Daily capacity release.
        minutes released = 1,200 * 4.3 = 5,160 min/day
        hours released   = 5,160 / 60 = 86.0 h/day

STEP 6  Money value at 72 USD/h loaded cost (only if the hours are reassigned).
        value = 86.0 * 72 = 6,192 USD/day
```

### 5.3 Independent re-check, by three routes

The purpose of a re-check is to fail differently. Three routes that share an assumption can agree and
all be wrong, so the routes below deliberately use different inputs.

**Route A — seat-minutes.** Total available seat-minutes per day are `60 * 600 = 36,000`. Before:
`1,200 * 15.4 = 18,480` used, leaving 17,520. After: `1,200 * 11.1 = 13,320` used, leaving 22,680.
Difference in unused capacity: `22,680 - 17,520 = 5,160` minutes `= 86.0` hours. Agrees with STEP 5.

**Route B — utilisation delta.** `rho` falls from 0.513333 to `13,320 / 36,000 = 0.370000`. The
change is `0.143333` of total capacity, and `0.143333 * 36,000 = 5,160` minutes. Agrees.

**Route C — headroom.** Solve for the volume at which the new service time reproduces the old demand:
`N * 11.1 = 18,480` gives `N = 1,664.865` cases/day. The extra volume is `464.865` cases at 11.1
minutes each, which is `5,160` minutes. Agrees to the minute, and gives the throughput headroom —
38.7% — as a by-product.

Three independent routes agreeing is necessary and not sufficient: all three assume the roster is 60
seats for the full 600 minutes, which is the assumption most likely to be false.

### 5.4 Sensitivity

Each perturbation is applied alone against the base case of 86.0 hours per day.

| Perturbation | Recomputed quantity | Result | Delta vs base |
|---|---|---|---|
| Verification doubles to 5.0 min | `15.4 - 13.6` | 1.8 min → 36.0 h | −50.0 h |
| Verification halves to 1.25 min | `15.4 - 9.85` | 5.55 min → 111.0 h | +25.0 h |
| Audit rate doubles to 12% | `0.12 * 4.0 = 0.48` charged | 3.82 min → 76.4 h | −9.6 h |
| Volume falls 20% to 960/day | `960 * 4.3 / 60` | 68.8 h | −17.2 h |
| Roster falls to 50 seats | `rho = 0.0333 / 0.0541` | capacity-bound; see below | not additive |

Recompute the first three explicitly to confirm the table. Verification doubling: touch time after is
`8.6 + 5.0 = 13.6`, so `g_net = 1.8` and hours are `1,200 * 1.8 / 60 = 36.0`. Verification halving:
`8.6 + 1.25 = 9.85`, `g_net = 5.55`, hours `1,200 * 5.55 / 60 = 111.0`. Audit doubling: the audit
charge is inside the 11.1 only if it is charged per case, so restating it as an explicit per-case
charge gives `8.6 + 2.5 + 0.48 = 11.58` and `g_net = 15.4 - 11.58 = 3.82`, hours
`1,200 * 3.82 / 60 = 76.4`.

The fifth row is the important one and is deliberately not additive. At 50 seats, `mu` becomes
`50 / 924 = 0.0541126` cases/s and `rho = 0.033333 / 0.0541126 = 0.616000` on the *baseline* service
time. The assisted service time gives `rho = 1,200 * 11.1 / (50 * 600) = 13,320 / 30,000 = 0.444000`.
So the assisted workflow remains stable at 50 seats while the baseline does not — the baseline would
need `18,480 / 30,000 = 0.616000` utilisation at the same volume, which is stable but with far less
headroom. The correct statement is therefore that the workflow buys stability, not that it buys hours,
once the roster is the binding constraint.

### 5.5 What the example establishes

The net release of 4.3 minutes per case, 86.0 hours per day, and 6,192 USD per day are all
arithmetically verified for the stated inputs. None is an observation. The sensitivity table shows
that the verification term dominates every other input: a ±1.25-minute change in verification moves
the estimate by ±25 hours per day, more than a 20% volume change does.

The example also establishes a limit on the whole approach. At 50 seats the release stops being an
hours figure at all, because capacity is no longer the constraint the calculation assumes. A number
that changes its own meaning when an input crosses a boundary must be reported with that boundary.

**Takeaway:** recompute by routes that fail differently, then state which input the answer is most
sensitive to — here, the verification term.

## 6. Visual / Geometric Intuition

### 6.1 The five-layer defect map as a ladder

```text
   layer 5  ACT        actor blank? .......... authority defect (CI-04, CI-25)
   layer 4  INTERPRET  quantity named wrong? . semantic defect  (CI-04, CI-12, CI-20, CI-24)
   layer 3  COMPUTE    wrong at a boundary? .. numeric defect   (CI-02, CI-06, CI-10, ...)
   layer 2  JOIN       as-of vs latest? ...... version defect   (CI-01, CI-09, CI-17)
   layer 1  RECORD     count reconciles? ..... completeness     (CI-01, CI-14)
```

Read the ladder upward, and note that a defect low on the ladder contaminates everything above it. A
count mismatch at layer 1 makes every rate at layer 3 undefined, which makes every interpretation at
layer 4 meaningless. That is why triage starts at the bottom: a cheap check at layer 1 can invalidate
an expensive analysis at layer 4.

### 6.2 The backlog triangle

```text
backlog
  ^
  |                        /  slope = lambda - mu
  |                      /
  |                    /
  |                  /
  |                /
  |              /
  |            /
  +------------+------------------> elapsed time
               t_overload
```

The area under the rising edge is the total excess work, `0.5 * (lambda - mu) * t^2` for a ramp. The
peak height is `(lambda - mu) * t`, which is the quantity F1 computes. The distinction matters because
two teams can report "the same incident" using the peak and the area and disagree by a factor of
`t / 2`, which at `t = 240` seconds is a factor of 120 if unit confusion is added.

### 6.3 The drain curve and its reciprocal shape

```text
T_drain
  ^
  |*
  | *
  |  *
  |    *
  |        *
  |              *  *
  +-----------------------> (mu_r - lambda_r) surplus
```

The curve is `B / surplus`, a hyperbola. Its shape is the geometric reason that small capacity changes
produce large drain-time changes near the left edge and almost no change at the right. An operator
reading a table of drain times sees three numbers (2,727 s, 300 s, 150 s) at three surpluses (0.04,
0.40, 0.80) and reads the middle one as typical; the function has no typical value.

### 6.4 The retry amplification curve

```text
A = 1/(1-p)
  ^
20|                                     *
  |                              *
10|                       *
  |             *
 5|      *
  |  *
 1+-----------------------------> p
   0   0.2  0.4  0.6  0.8  0.95
```

The curve is flat to the eye until `p ≈ 0.5` and then vertical. This shape explains why a retry storm
looks like a step function in monitoring rather than a gradual degradation: at `p = 0.4` the
multiplier is 1.67, at `p = 0.8` it is 5.0, and the system crosses from "absorbable" to
"self-sustaining" between those two points.

### 6.5 The interval-versus-threshold picture

```text
             break-even
                 |
   [=============|====]         -> interval straddles threshold: undecided
        [====|====]             -> interval below: failed sufficiency test
                 |  [======]    -> interval above: passed

   point estimate   *
```

The three cases are different decisions, not different degrees of the same decision. Straddling
requires a design change; entirely below is a reject; entirely above is an accept. A report that
gives only the point estimate collapses all three into a single number and forces the reader to
invent the interval.

**Takeaway:** each defect family has a characteristic shape in a chart, and the shape is faster to
recognise than the field list.

## 7. Comparison / Selection Table

### 7.1 Selecting a diagnostic by what differs

The table below answers one question: given a symptom, which single observation most cheaply
distinguishes the competing mechanisms? Each row is a family of symptoms, not a single issue.

| Symptom family | First discriminating observation | Cheap? | Eliminates | Leaves standing |
|---|---|---|---|---|
| Number moved but operations disagrees | Recompute the metric on the pilot population reweighted to intake | medium | population shift | unit mismatch, tail effect |
| Same number, two teams disagree | Compare query grain and denominator between the two | low | aggregation error | join-version error |
| Improves in mean, not in tail | Plot p50 and p99 on the same axis | low | mean/tail confusion | hot-key skew |
| Fails only at month boundary | Re-derive with all timestamps forced to UTC | low | timezone/DST | clock drift |
| Fails only above pilot volume | Recompute at 10x volume and compare slope | medium | constant-cost assumptions | quadratic fan-out |
| Effect disappears after re-run | Diff configuration hash between runs | low | nondeterminism | data drift |
| Two identical requests, two effects | Count effects per logical key | low | transport duplication | key regeneration |
| Alert never fires during quiet periods | Inspect denominator and sentinel policy | low | threshold logic | undefined-rate semantics |

"Cheap" here means: can be run by one person with read access, in under an hour, without a fix. Any
check that requires a change first is not diagnostic — it is an experiment, and experiments belong in
the correction phase.

### 7.2 Containment selection

Containment is chosen before cause is known, so it must be safe under all candidate causes.

| Containment | Safe under | Unsafe when | Reversal |
|---|---|---|---|
| Defer the item with a reason code | any input defect | deferral volume exceeds manual capacity | drain queue |
| Cap admission at a share | overload, duplicate work, bad config | the cap hides a growing backlog | raise cap |
| Serve last known good configuration | config drift, bad deploy | the last good config was also wrong | repin |
| Disable the automated effect, keep the record | effect duplication, authority ambiguity | the effect is legally required per case | re-enable |
| Widen the log, not the sampling | diagnosis of rare events | storage is the constraint being hit | drop the sink |
| Freeze the analysis plan, not the traffic | any measurement dispute | the defect causes active harm | unfreeze |

The fifth row is worth dwelling on. During an investigation the instinct is to log everything, which
is correct when storage is plentiful and self-defeating when the symptom is "storage exhausted"
(CI-11). Choosing the wrong containment converts a measurement problem into an availability problem.

### 7.3 Correction selection: symptom versus mechanism

| Correction style | Fixes | Does not fix | When to prefer |
|---|---|---|---|
| Widen the type or add a finite check | numeric edge | upstream bad value | arithmetic proven wrong at a boundary |
| Add a precondition rejection | violated assumption | silent bad values already loaded | contract can be enforced at ingest |
| Add admission control or a budget | scale failure | quadratic behaviour inside one item | cost grows with volume, not with item size |
| Rewrite the question and the metric | concept confusion | any arithmetic | the number answers a different question |
| Add version pinning and a hash | drift | a wrong pinned value | changes are frequent and untracked |

A correction that appears in the "does not fix" column for the identified mechanism is a symptom fix,
and it will be followed by a second incident.

### 7.4 Choosing between replay and re-measurement

| Situation | Verification method | Sample | Pass condition |
|---|---|---|---|
| Deterministic computation | replay the trigger and one boundary record | 2 | both produce expected values |
| Statistical claim, small effect | re-run the interval on held-out data | >= prior n | interval position unchanged |
| Rate claim with a threshold | re-run the audit on a fresh sample | n from CI-F6 | interval clears the threshold |
| Scale claim | re-run at 10x volume | one load profile | slope unchanged within tolerance |
| Authority claim | replay one override and one denial | 2 | both produce a ledger row with an actor |

The third row is the one teams skip. A rule of the form "roll back if the rate exceeds 12%" requires a
sample large enough for the interval to resolve 12% from 8%, and CI-F6 shows that 216 cases do not.

**Takeaway:** choose the check that eliminates the most mechanisms per hour of effort, not the check
that is most familiar.

## 8. Failure Modes & Edge Cases

### 8.0 Catalog summary

The twenty-five entries below are ordered by layer, from record completeness upward, because that is
the order in which they contaminate each other. Each identifier is stable: a later revision may
sharpen an entry but must not renumber it, since cases and exercises cite these IDs.

| ID | Category | Symptom in one line |
|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | A decision-time field was written after the decision cut-off |
| CI-02 | NUMERICAL_EDGE | A rate over an empty window is reported as zero |
| CI-03 | SCALE_FAILURE | A pilot result does not reproduce at full volume |
| CI-04 | CONCEPT_CONFUSION | A model score is treated as decision authority |
| CI-05 | VIOLATED_ASSUMPTION | A retry produces a second real-world effect |
| CI-06 | NUMERICAL_EDGE | Bytes are reported in the wrong unit near a quota |
| CI-07 | SCALE_FAILURE | One hot key saturates while the pool looks idle |
| CI-08 | CONCEPT_CONFUSION | An untested plan is reported as a result |
| CI-09 | VIOLATED_ASSUMPTION | A slowly changing dimension is joined at its latest version |
| CI-10 | NUMERICAL_EDGE | Rounding at displayed precision reverses a boundary decision |
| CI-11 | SCALE_FAILURE | Diagnostic logging exhausts storage during the incident |
| CI-12 | CONCEPT_CONFUSION | A proxy metric is reported as the business objective |
| CI-13 | VIOLATED_ASSUMPTION | Configuration changes leave no version trace in the record |
| CI-14 | NUMERICAL_EDGE | Extraction and load row counts disagree, silently dropping rows |
| CI-15 | SCALE_FAILURE | Fan-out multiplies one request into many downstream calls |
| CI-16 | CONCEPT_CONFUSION | A correlation from a self-selected sample is read as a cause |
| CI-17 | VIOLATED_ASSUMPTION | A reused field changes meaning without a schema version |
| CI-18 | NUMERICAL_EDGE | An integer accumulator overflows and corrupts a total |
| CI-19 | SCALE_FAILURE | Cache expiry sends a synchronized stampede to the origin |
| CI-20 | CONCEPT_CONFUSION | An approximate answer is consumed as if it were exact |
| CI-21 | VIOLATED_ASSUMPTION | Authorization is evaluated after data is loaded |
| CI-22 | NUMERICAL_EDGE | NaN enters an ordered comparison and vanishes from a filter |
| CI-23 | SCALE_FAILURE | A retry budget absent, a retry storm hides the original fault |
| CI-24 | CONCEPT_CONFUSION | A formula is applied outside the range where it was derived |
| CI-25 | VIOLATED_ASSUMPTION | A human override is recorded as an action without a reason |

Two conventions hold for every entry. The discriminating check is written so that it produces *a
different observable under each competing cause*; where that is impossible the entry says so and
recommends escalation rather than a check. And every verification names a boundary record alongside
the triggering record, because a correction validated only on the trigger has usually been narrowed
to fit it.

### 8.1 CI-01 — a decision-time field was written after the cut-off

| Field | Content |
|---|---|
| ID | CI-01 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A retrospective analysis shows an effect that nobody can reproduce in live operation |
| Competing causes | (a) outcome field joined into the feature set; (b) the effect was real but decayed; (c) the live path uses a different configuration |
| Discriminating check | For each field, compare its write timestamp against `decision_cutoff_at`; count fields with `write > cutoff` |
| Correction | Remove every such field and rebuild the decision-time record |
| Prevention | Reject at build time any feature set containing a field whose write time exceeds the cut-off |
| Verification | Replay the triggering case and a case that closed one minute before the cut-off |

The mechanism is a join that does not know about time. An analyst joins the outcome table because it
is the most predictive table available, and it *is* the most predictive table precisely because it
is written later. The analysis then achieves a number that describes the pipeline's hindsight rather
than the decision's information.

Two competing causes deserve names because they produce the same report. A real effect that decayed
(mechanism b) produces the same "cannot reproduce" symptom with no leakage at all, and it is
distinguished by re-running the analysis on the *earliest* slice of the evaluation window rather than
the latest. A configuration difference (mechanism c) is distinguished by comparing configuration
hashes between the retrospective run and the live path.

Correction note: removing the field is not always the right move. If the field is genuinely useful,
the correct correction is to record its value at cut-off time in the decision ledger, which both
preserves the signal and makes it available. That requires a schema change, which is why it is a
prevention item as well as a fix.

Prevention note, stated precisely: the build must fail, not warn. A warning at build time produces a
comment in a pull request; a failure produces a removed field. The threshold for a build failure is
a single field, not a proportion, because a feature set with one leaked field has no interpretation.

Verification boundary: the second record should be a case that closed one minute before the cut-off,
where any timestamp confusion produces visible disagreement. A case far from the boundary will pass
under both the wrong and the right implementation.

### 8.2 CI-02 — a rate over an empty window is reported as zero

| Field | Content |
|---|---|
| ID | CI-02 |
| Category | NUMERICAL_EDGE |
| Symptom | A quiet period shows a perfect error rate and no alert fires |
| Competing causes | (a) denominator-zero substituted with 0; (b) a counter that was never initialised; (c) errors genuinely absent |
| Discriminating check | Print numerator and denominator separately for the window; a denominator of exactly 0 confirms (a) or (b) |
| Correction | Emit an explicit undefined token and exclude the window from rate aggregates |
| Prevention | Make the rate type carry an `undefined` variant so zero-denominator cases cannot be coerced |
| Verification | Replay an empty window and a window with one event; both must differ from zero |

The alerting consequence is subtler than the arithmetic. An alert written as `rate > 0.05` is
false for both an undefined rate coerced to 0 and a genuine 0, so the alert is silent in both. The
distinction between "nothing happened" and "nothing was measured" is lost at exactly the moment it
matters: during a partial outage, the service that measures errors may itself be down.

Distinguishing (b) from (a): an uninitialised counter usually reports 0 for the numerator as well as
the denominator, giving `0/0`. A genuine empty window with a working counter gives `0/0` too, so the
pair alone is insufficient. The discriminator is to check whether the counter's own emission
timestamp exists for the window: no emission means (b), an emission with zero means (a) or (c).

Correction detail: the aggregate must propagate undefined rather than treating it as missing. Filling
undefined with the previous window's value is a third defect — it invents an observation. The correct
aggregate over a set of windows containing undefined values is to report the defined subset with its
own denominator plus a count of undefined windows.

Prevention in practice: define the rate function's return type as a union of measured and undefined,
and forbid arithmetic on the union without an explicit case split. Languages with sum types make this
cheap; languages without them make it a convention, and conventions decay.

Verification boundary: the one-event window is the important half. It confirms that the rate is
computed rather than short-circuited, because a one-event window has a defined rate that differs
from both 0 and undefined.

### 8.3 CI-03 — a pilot result does not reproduce at full volume

| Field | Content |
|---|---|
| ID | CI-03 |
| Category | SCALE_FAILURE |
| Symptom | The effect measured on the pilot share is absent or reversed at full intake |
| Competing causes | (a) pilot population differed in mix; (b) a per-item cost was not paid at pilot scale; (c) utilisation crossed a non-linear region |
| Discriminating check | Recompute the metric per stratum at both scales; a mix difference confines the gap to strata, a cost difference scales with volume |
| Correction | Restate the claim at the scale it was measured at, then measure at the target scale before promoting |
| Prevention | Express every claim with its scale attached, and require the promotion to be a measured arm |
| Verification | Replay a pilot-volume window and a target-volume window through the same computation |

Mechanism (c) is the one that catches experienced teams, because it is arithmetic rather than
politics. A service-time reduction of 27.9% at `rho = 0.513` reduces the wait by roughly 57% in a
single-server view; the same reduction at `rho = 0.85` reduces it by roughly 186% of the same base,
which is not a percentage that can be reported because the linearisation has already broken. The
effect is *larger* near saturation and *less stable*, and a pilot run at low volume systematically
measures the stable regime.

Distinguishing (a) from (c): stratify by case category and compute utilisation per stratum. If the
gap appears within strata at equal utilisation, it is (a). If the gap appears only in the high-
utilisation stratum, it is (c).

Correction detail: "restate at measured scale" is not a rhetorical move. The concrete artefact is a
sentence of the form "at 15% of intake over 20 business days, net release was X", which is a claim a
reader can falsify. The equivalent sentence at full intake is a different claim requiring its own
evidence.

Prevention: the promotion arm is the measurement, so the correct prevention is structural — every
scale-changing decision carries a measurement arm at the target scale. That is what makes a 15% arm
worth its fixed cost rather than a compromise.

Verification boundary: the pair of windows should be at the same intake volume but different
promoted shares, which isolates the share effect from the volume effect that a pilot-versus-
production comparison confounds.

### 8.4 CI-04 — a score is treated as decision authority

| Field | Content |
|---|---|
| ID | CI-04 |
| Category | CONCEPT_CONFUSION |
| Symptom | An irreversible action occurs and no person or approved policy can be named as its author |
| Competing causes | (a) automation wrote the effect without an actor; (b) an actor exists but the field is unpopulated; (c) a batch process attributed the action to a service account by design |
| Discriminating check | List distinct actor values on final actions; a service account on an irreversible effect confirms (a) or (c) |
| Correction | Route the effect through a gate that requires either a named reviewer or an approved policy version |
| Prevention | Make the actor field non-nullable at the schema level for any row that carries an effect |
| Verification | Replay one auto-approved case and one human-overridden case; both must record distinct actors |

A threshold and an authorisation are different objects. A threshold answers "how confident is the
recommendation"; authorisation answers "who accepted the consequence". A system can have a
well-calibrated threshold and no authorisation, and the resulting record is complete arithmetically
and absent legally.

Distinguishing (b) from (a): if the schema allows null and the field is null on every automated
action, the mechanism is (a) with a contributing schema defect. If the field is null on only some
automated actions, the defect is in the writer, which is (b), and the fix is local.

Distinguishing (c) from (a): a deliberate service-account attribution has a policy version recorded
somewhere — a configuration, a change record, a deployment note. The check is to look for it. If it
exists, the correction is to move it into the decision record so it is discoverable from the case
alone; if it does not, (a) holds.

Correction detail: the policy version must be the version *that was in force at the decision*, not
the current version. A record citing the current policy cannot be used to reconstruct what the rule
was, which defeats the purpose of naming it.

Prevention note: non-nullable actor fields have a cost — they require every writer to know who is
acting — and that cost is the point. A writer that cannot name an actor has not decided who is
accountable, and forcing the question at write time is cheaper than answering it during a review.

Verification boundary: the overridden case tests the other direction. A system that records a human
action but loses the override reason satisfies the actor requirement and fails the audit, which is
CI-25.

### 8.5 CI-05 — a retry produces a second real-world effect

| Field | Content |
|---|---|
| ID | CI-05 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A case is disposed twice, or a payment is made twice, with two plausible audit rows |
| Competing causes | (a) the retry regenerated its idempotency key; (b) the effect committed but the acknowledgement was lost; (c) two actors legitimately acted on the same key |
| Discriminating check | Group effect rows by logical key, then compare the key values of the duplicate pair: different keys confirm (a) |
| Correction | Derive the key from stable inputs and enforce uniqueness at the effect store |
| Prevention | Remove any code path that can construct a key from a clock, a UUID, or a request-scoped value |
| Verification | Submit one logical request twice with the same key and once with a regenerated key |

The three causes leave different fingerprints and require different owners. (a) is a caller defect
and lives in the retry wrapper. (b) is a transport defect and lives in the acknowledgement protocol.
(c) is an assignment defect and lives in the case-routing rules, not in the ledger at all.

Distinguishing (b) from (a) concretely: in (b) the ledger holds exactly one row for the original key,
and the *caller* holds a timeout record. The check is therefore to join the caller log to the ledger
by key and look for a caller-side timeout with no corresponding ledger-side error. That join is cheap
and is the fastest single check in this catalog.

Distinguishing (c) from (a): different actor values. Two actors on one case is a routing question
("why did two reviewers hold the same case") and no amount of idempotency work will fix it.

Prevention, stated as a rule with a testable form: a key may be a function of `(logical_entity_id,
attempt_ordinal)` and of nothing else. A unit test asserts that two calls constructed from the same
entity and the same ordinal produce byte-identical keys, and that a call constructed after a clock
advance produces the same key. That second assertion is the one usually missing.

Verification boundary: the third submission — the regenerated key — must be *rejected* by the effect
store, not merely deduplicated by the caller. If deduplication lives only in the caller, a second
caller instance reproduces the effect, and the invariant is accidental rather than structural.

### 8.6 CI-06 — bytes are reported in the wrong unit near a quota

| Field | Content |
|---|---|
| ID | CI-06 |
| Category | NUMERICAL_EDGE |
| Symptom | A storage or memory quota is exceeded while the reported usage is below the limit |
| Competing causes | (a) binary units reported as decimal; (b) a quota enforced on compressed size and reported on raw; (c) replication factor omitted from the reported total |
| Discriminating check | Compute `bytes / 1,048,576` and `bytes / 1,000,000`; a 4.86% gap between them confirms (a) |
| Correction | Report in the unit the quota enforces, with the unit in the field name |
| Prevention | Encode the unit in the metric name and reject any metric whose name lacks one |
| Verification | Replay a value within 5% of the quota; both readings must fall on the same side of the limit |

The overstatement factor is `1,048,576 / 1,000,000 = 1.048576`, so a value reported as 953.674 in
MiB is 1,000.000 in MB. Presenting the first with the second's label understates the true size by
4.63%, which is enough to cross a quota set at a round decimal number.

Distinguishing (b) and (c) requires the storage layer's own accounting rather than the application's.
A compressed-versus-raw mismatch (b) shows a ratio that changes with content type; a replication
omission (c) shows a ratio that is a small integer. Both are visible in a single comparison of
reported usage against the platform's invoice or bucket listing.

Prevention detail: metric names should read `storage_used_mebibytes` and `storage_quota_mebibytes`
rather than `storage_used`. The cost is verbosity; the benefit is that a graph legend carries the
unit and a reader cannot relabel it.

Verification boundary: the near-quota case is the only informative one. A value at 10% of quota
passes under every unit convention, so testing there proves nothing.

### 8.7 CI-07 — one hot key saturates while the pool looks idle

| Field | Content |
|---|---|
| ID | CI-07 |
| Category | SCALE_FAILURE |
| Symptom | Median latency is flat while the slowest requests grow without bound; pool utilisation stays low |
| Competing causes | (a) key skew concentrating work on one partition; (b) a single-threaded dependency in the path; (c) retry amplification on a small subset |
| Discriminating check | Compute p50 and p99 per key; a p99 driven by one key whose per-key rate exceeds one partition's capacity confirms (a) |
| Correction | Bound per-key work, or add a per-key admission limit before adding capacity |
| Prevention | Choose partition keys with a measured distribution, and record per-key rate in the run metrics |
| Verification | Replay the skewed key and a balanced key at the same total load |

The tell is the gap between median and tail with no corresponding rise in mean utilisation. A
resource that is genuinely saturated shows both. A hot key shows a tail because a small fraction of
requests queue behind one constrained lane while the rest of the pool is idle.

Distinguishing (a) from (b): (b) produces the same tail for *every* key, because the single-threaded
step is common to all paths. The per-key comparison separates them cleanly. Distinguishing (a) from
(c): retries increase the *rate* on the affected key, so a retry-driven tail shows attempt counts
above one in the request log, which is a different observation from a genuinely high logical rate.

Correction detail: sharding the key is the obvious move and is often wrong, because a sharded hot key
still has one shard receiving most of the load unless the shard function includes entropy. Per-key
admission control is safer because it bounds the damage without changing the data layout, and it is
reversible.

Prevention: measure the key distribution at design time. A key chosen for uniform *counts* can still
be skewed by *size*, and a key chosen for uniform size can be skewed by arrival burstiness. The
metric that matters is per-key work per unit time, not per-key row count.

Verification boundary: the balanced key at the same total load is the control. It confirms that the
correction targeted skew rather than reducing throughput generally.

### 8.8 CI-08 — an untested plan is reported as a result

| Field | Content |
|---|---|
| ID | CI-08 |
| Category | CONCEPT_CONFUSION |
| Symptom | A status report cites an outcome for work that was designed but never executed |
| Competing causes | (a) proposal language drifted into past tense; (b) an executed test exists but its window and comparator differ from the claim; (c) a small execution was generalised to the full population |
| Discriminating check | For the cited number, ask for the artefact, the window, and the comparator; any missing item confirms (a) or (b) |
| Correction | Restate as not-measured with the plan labelled as a plan |
| Prevention | Require every reported number to carry a status token from a fixed vocabulary |
| Verification | Audit three reported numbers and check each one's artefact resolves and matches |

The mechanism is linguistic before it is analytical. "The workflow reduces handling time by 22%" and
"the workflow is expected to reduce handling time by 22%" differ by one word and by an entire
evidence base, and the first is shorter to write.

Distinguishing (b) from (a): in (b) an artefact exists. Reading it will show either a different
window or a different comparator from the claim, and the correction is to restate the claim to match
the artefact rather than to withdraw it. In (a) no artefact exists and the number's origin may be a
projection, a target, or a rival's published figure.

Distinguishing (c) from (b): (c) has a correct artefact and an overreaching generalisation. The
discriminator is the population definition in the artefact versus in the claim.

Prevention detail: the fixed vocabulary matters more than it seems. Four tokens — measured, derived,
illustrative, not-measured — force a writer to choose, and the choice is reviewable. Free-form
prose allows the choice to be made implicitly and never reviewed. `derived` covers arithmetic on
stated inputs, which is what most teaching documents actually contain.

Verification boundary: pick the oldest reported number in the portfolio, not the newest. Old numbers
are the ones that have lost their artefact and are still being cited.

### 8.9 CI-09 — a slowly changing dimension is joined at its latest version

| Field | Content |
|---|---|
| ID | CI-09 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A historical analysis shifts whenever a dimension table is updated, with no change to the fact table |
| Competing causes | (a) join uses `valid_to IS NULL` instead of the as-of interval; (b) the dimension overwrites rather than versions rows; (c) the extract was rebuilt from a later snapshot |
| Discriminating check | Re-run the same query against an immutable snapshot; a stable result from the snapshot confirms (a) or (c) |
| Correction | Join on `valid_from <= cutoff < valid_to` and require both columns to exist |
| Prevention | Make the dimension append-only with non-overlapping validity intervals enforced by a constraint |
| Verification | Replay one case whose dimension row changed after the cut-off and one whose row never changed |

This defect is distinguishable from CI-01 by direction. CI-01 uses a field written after the
decision; CI-09 uses a field written before the decision but *revised* after it. A record written at
09:00 with value 3 and revised at 16:00 to value 7 passes every write-time check and still corrupts
a decision taken at 11:00.

Distinguishing (b) from (a): if the dimension table has no `valid_from` column, versioning was never
implemented and the mechanism is (b), which requires a schema change before any query fix is
possible. If the columns exist and the query ignores them, it is (a), a one-line fix.

Distinguishing (c) from (a): rebuild provenance. An extract rebuilt from a later snapshot reproduces
the current dimension values regardless of query correctness, so the discriminator is the extract's
own timestamp versus the snapshot's.

Prevention: an append-only dimension with a constraint that rejects overlapping intervals is the only
prevention that survives analyst turnover. Documentation does not, because the next analyst writes
the query that runs.

Verification boundary: the unchanged row is the control. It confirms that the corrected join has not
simply excluded all dimension attributes, which would also make the historical result stable.

### 8.10 CI-10 — rounding at displayed precision reverses a boundary decision

| Field | Content |
|---|---|
| ID | CI-10 |
| Category | NUMERICAL_EDGE |
| Symptom | A case is approved in the interface and rejected in the audit, or vice versa, for the same inputs |
| Competing causes | (a) the comparison uses rounded values; (b) the tie rule differs between two implementations; (c) the two paths use different thresholds |
| Discriminating check | Recompute both paths with unrounded values and with half-even ties; a verdict that persists confirms (c) |
| Correction | Compare unrounded values and round only for display, after the branch |
| Prevention | Keep the decision function free of formatting code, and test it at an exact tie |
| Verification | Replay a value exactly at the threshold and one at threshold plus the last displayed digit |

The concrete case is a threshold at 0.75 with values displayed to two decimals. A computed 0.745
displays as 0.75 due to half-up, and a reviewer comparing the display to the threshold approves,
while the code comparing the raw value rejects. Both paths are internally consistent and the
disagreement is invisible unless the tie is tested.

Distinguishing (b) from (a): both paths round, but one uses half-up and the other half-even. At an
exact tie of 1.5 the two produce 2 and 2 respectively, but at 0.5 they produce 1 and 0. A single
tie value reveals which convention is in force, which is why the test vector must include a tie.

Distinguishing (c) from (a) and (b): a threshold difference persists when both paths are given
unrounded inputs. The check is therefore to bypass the display layer entirely and call both decision
functions with the same raw value.

Prevention: a unit test at the exact threshold is the cheapest possible guard and is almost never
written, because the threshold is usually defined in a configuration rather than in code and the test
is written against the code's literal.

Verification boundary: threshold plus the last displayed digit — 0.7501 at two decimals — confirms
the branch flips where it should rather than staying on one side across the display resolution.

**Takeaway:** the first ten entries all fail while producing a plausible number, which is why none of
them announces itself.

### 8.11 CI-11 — diagnostic logging exhausts storage during the incident

| Field | Content |
|---|---|
| ID | CI-11 |
| Category | SCALE_FAILURE |
| Symptom | The logging backend stops accepting writes at the moment the incident begins |
| Competing causes | (a) per-request payloads were widened for diagnosis; (b) retries logged each attempt with full context; (c) retention was extended without a volume check |
| Discriminating check | Compute bytes per event and events per request; a product above the provisioned write rate confirms (a) or (b) |
| Correction | Cap payload size and sample the high-volume path, keeping full detail on a bounded ring buffer |
| Prevention | Budget logging in bytes and enforce the budget in the emitter, not in the backend |
| Verification | Replay a burst at the incident's peak rate and observe the write-rejection count |

The arithmetic is simple and worth doing before any investigation. At 1,200 cases per business day
over 20 days with 18 events per case and 4 KB per event, the volume is `1,200 × 20 × 18 = 432,000`
events, or `432,000 × 4 KB = 1,728,000 KB = 1,687.5 MB`, which is 1.65 GB. Widening the payload from
4 KB to 20 KB multiplies that to 8.25 GB for the same twenty days — a figure that will exceed a
default retention provision on a small cluster.

Distinguishing (b) from (a): (b) scales with failure rate, so the volume spike coincides with the
error rate rather than with the deploy. The discriminator is to plot event volume against error rate;
a proportional relationship identifies (b).

Correction detail: the ring buffer is the important half of the correction, because it preserves full
detail for the recent window at a bounded cost. Sampling alone loses the detail exactly where it is
needed; full-detail buffering alone has no bound.

Prevention: a byte budget in the emitter means the emitter must know the shape of its own messages,
which is a design constraint on the schema. The alternative — enforcing the budget in the backend —
produces dropped writes, which is worse than sampling because it is invisible to the application.

Verification boundary: the peak-rate replay must include the retry path, since a burst of retries is
what produces the incident's peak. A replay at nominal rate will pass under every design.

### 8.12 CI-12 — a proxy metric is reported as the business objective

| Field | Content |
|---|---|
| ID | CI-12 |
| Category | CONCEPT_CONFUSION |
| Symptom | A metric improves while the outcome the business cares about does not move |
| Competing causes | (a) the proxy is only loosely coupled to the objective; (b) the objective is measured too late to see the effect; (c) the effect is real but absorbed elsewhere in the process |
| Discriminating check | Measure both quantities on the same population and window; a proxy gain with no objective movement confirms (a) or (c) |
| Correction | Report the objective alongside the proxy and state the assumed coupling explicitly |
| Prevention | Pair every proxy with its objective in the metric catalogue, so one cannot be reported without the other |
| Verification | Replay a case where the proxy improved and a case where it did not, and compare objective movement |

The mechanism is a units substitution. Elapsed time is measurable at the moment of the case; reviewer
minutes are measurable only with instrumentation; the business objective (cost per case, or backlog
age) is measurable only after a full accounting period. Analysis drifts to the measurable quantity
because it is available, and the drift is not visible in any single report.

Distinguishing (b) from (a): (b) produces a delayed movement. The check is to measure at two horizons,
one immediately after the change and one after the accounting period closes; movement in the second
confirms (b). If neither moves, (a) or (c) holds.

Distinguishing (c) from (a): (c) means the proxy gain is real but consumed downstream — the case
finishes faster and then waits at a subsequent gate. The discriminator is end-to-end time versus step
time: an improvement in step time with flat end-to-end time identifies (c), and the correction is to
move the constraint rather than to change the metric.

Correction detail: stating the coupling means writing the sentence "we assume that a one-minute
reduction in X produces a Y reduction in Z, because ..." and then designing the measurement that
would falsify it. That sentence is the deliverable; a metric hierarchy diagram is not.

Prevention: pairing proxy with objective in the catalogue is a structural constraint that a
dashboard cannot satisfy on its own, because a dashboard shows whatever panels were built. The
catalogue is the artefact that makes the pairing reviewable before the dashboard exists.

Verification boundary: the non-improving case is the control. A report built only from cases where
the proxy moved cannot distinguish a coupling from a selection of favourable cases.

### 8.13 CI-13 — configuration changes leave no version trace in the record

| Field | Content |
|---|---|
| ID | CI-13 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | Two runs on identical inputs produce different outputs, and nobody can say what differed |
| Competing causes | (a) an unpinned configuration was read at run time; (b) a default changed in a library upgrade; (c) the inputs differed in a field not considered an input |
| Discriminating check | Compare configuration hashes between the two runs; a differing hash confirms (a) or (b) |
| Correction | Record the hash of every configuration artefact in the run record, and rebuild both runs from pinned values |
| Prevention | Resolve configuration once at start-up into an immutable object whose hash is logged |
| Verification | Replay one run with a changed threshold and one with no change; only the first may differ |

Distinguishing (a) from (b) requires the lockfile and the library version. If the configuration hash
is identical but the library version differs, (b) holds — a default inside the dependency changed.
That mechanism is easy to miss because the application's own configuration is untouched.

Distinguishing (c) from (a): (c) means the "input" set was incomplete. The discriminating check is to
hash the full input payload rather than the field list the pipeline declares, and to compare. A
difference there identifies (c) and implies the pipeline's input contract is under-specified.

Correction detail: "rebuild both runs from pinned values" is the only correction that answers the
question. Reproducing one run under the new configuration tells you what the new configuration does;
it does not tell you what changed.

Prevention: start-up resolution into an immutable object also prevents the subtler failure mode where
a configuration file is edited mid-run and different cases see different values. That failure is
rare and produces results that are individually explicable and collectively incoherent.

Verification boundary: the unchanged run must produce an identical hash and identical output. If it
does not, the system has a nondeterminism defect independent of configuration, and CI-13's correction
will not be sufficient.

### 8.14 CI-14 — extraction and load row counts disagree

| Field | Content |
|---|---|
| ID | CI-14 |
| Category | NUMERICAL_EDGE |
| Symptom | Every rate in the analysis is slightly different from the operational report, with no visible cause |
| Competing causes | (a) rows rejected at load for schema reasons; (b) a join fan-out duplicated rows; (c) the extract window boundaries differ by one interval |
| Discriminating check | Compare extract count, loaded count, and distinct-key count; the three separate (a), (b), and (c) |
| Correction | Reconcile counts at each stage and publish the exclusion count with the analysis |
| Prevention | Make the load pipeline emit counts for read, rejected, written, and distinct keys |
| Verification | Replay a batch containing one malformed row and one duplicated key |

The three counts are diagnostic because each mechanism changes a different one. Rejection reduces
the loaded count below the extract count while leaving distinct keys consistent with the loaded
count. Fan-out leaves the loaded count *above* the distinct-key count while matching the extract
count. Window disagreement changes both the extract and the loaded count together and shifts the
distinct-key population.

The third mechanism is the one that produces arguments between teams, because neither side is wrong.
One team counts from 00:00 inclusive to 23:59:59 inclusive; the other counts from 00:00 inclusive to
next-day 00:00 exclusive. On a day with 1,200 cases the difference is the cases arriving in the final
second, which is small but non-zero and shifts every rate by a fraction that is large enough to be
noticed and small enough to be dismissed.

Correction detail: the exclusion count is part of the analysis, not an appendix. A rate computed on
an undisclosed survivor population is a rate of unknown meaning (this is the same defect family as
CI-16, arriving through a pipeline rather than a sample design).

Prevention: emitting four counts per batch costs nothing and makes every subsequent dispute
resolvable in one query. The absence of those counts is why the dispute escalates to a meeting.

Verification boundary: the duplicated-key batch tests fan-out; the malformed-row batch tests
rejection. A batch containing both tests the interaction, which is where the counts become
ambiguous and the pipeline's ordering must be documented.

### 8.15 CI-15 — fan-out multiplies one request into many downstream calls

| Field | Content |
|---|---|
| ID | CI-15 |
| Category | SCALE_FAILURE |
| Symptom | Downstream load grows faster than request volume, and the ratio worsens under failure |
| Competing causes | (a) an unbounded per-item expansion such as per-row lookups; (b) retries multiplying attempts; (c) a cache that stopped absorbing repeat reads |
| Discriminating check | Count downstream calls per inbound request, with retries counted separately from logical calls |
| Correction | Bound fan-out explicitly, batch the expansion, and set a call budget per request |
| Prevention | Make the call budget a parameter of the request context so it cannot be exceeded silently |
| Verification | Replay a request with 4 expansion items and one with 40 at the same arrival rate |

Distinguishing (a) from (b): the retry count is recorded where retries are attempted, so it is
separable by construction if the instrumentation distinguishes attempts from logical calls. Systems
that log only at the client boundary cannot separate them, which is why the prevention argues for
budget propagation rather than after-the-fact counting.

Distinguishing (c) from (a): a cache failure changes the ratio without changing the expansion
structure, so the per-request call count rises while the per-item structure stays constant. The
discriminator is the cache hit rate plotted against the same axis.

Correction detail: batching the expansion is the structural fix, and it changes the call count from
`f` per request to `ceil(f / batch_size)`. With `f = 40` and a batch of 20, the count falls from 40
to 2, a twentyfold reduction that costs one extra serialisation step.

Prevention detail: a call budget is more useful than a rate limit because it bounds the *per-request*
cost, which is the quantity that interacts with retries. A rate limit bounds aggregate load and
leaves one pathological request able to consume the entire allowance.

Verification boundary: the 40-item request at the same arrival rate isolates fan-out from volume. A
test at higher volume would confound the two effects, which is the error that makes capacity
planning for fan-out unreliable.

### 8.16 CI-16 — a correlation from a self-selected sample is read as a cause

| Field | Content |
|---|---|
| ID | CI-16 |
| Category | CONCEPT_CONFUSION |
| Symptom | An association found in observational data disappears when the intervention is deployed |
| Competing causes | (a) confounding by a variable that drives both selection and outcome; (b) reverse causation; (c) a genuine effect that is smaller than the observed association |
| Discriminating check | Draw the causal graph and identify every variable with arrows into both treatment and outcome |
| Correction | Narrow the claim to an association, or design a comparison that removes the confounder |
| Prevention | Require a causal graph before any causal verb is used in a report |
| Verification | Replay the analysis on a stratum where the suspected confounder is constant |

The distinction between (a) and (c) is not visible in the data; it is a question about the world.
That is why the discriminating check is a drawing rather than a computation. A confounder is a
variable with edges into both the treatment and the outcome, and the check is to name every candidate
and ask how it was measured. If it was not measured, (a) cannot be excluded.

Distinguishing (b) from (a): reverse causation requires the outcome to precede the treatment in time.
A timestamp comparison settles it, and it is the first check to run because it needs no modelling.

Distinguishing (c) from (a): if the association survives stratification on every measured candidate
confounder, either there is no measured confounder or the effect is genuine. The honest report states
which of those it cannot distinguish, and that statement is more valuable than a stronger claim.

Correction detail: narrowing the claim is not a retreat. "Cases handled by senior reviewers closed
faster" is a factual statement; "seniority causes faster closure" is a causal claim that the
observational design cannot support. The first is actionable — it suggests pairing — and the second
invites an intervention that may not work.

Prevention: requiring a causal graph before causal language is cheap because the graph is usually
three nodes and two arrows. The cost is a habit, and the benefit is that the confounder question gets
asked before the data is collected rather than after the result disappoints.

Verification boundary: a stratum with constant confounder is the test. If the association persists
within the stratum, the confounder is not the whole story; if it vanishes, the association was the
confounder.

### 8.17 CI-17 — a reused field changes meaning without a schema version

| Field | Content |
|---|---|
| ID | CI-17 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A metric steps at a deploy boundary with no corresponding change in behaviour |
| Competing causes | (a) a field's unit changed while its name stayed; (b) an enum gained a value that the consumer maps to the default; (c) the nullable state changed and nulls are being coerced |
| Discriminating check | Compare the schema version and the field's declared unit at two points in time |
| Correction | Version the field and migrate consumers before the writer changes |
| Prevention | Treat unit and nullability as part of the field's name or type, not as documentation |
| Verification | Replay one row written before the change and one after, through the same consumer |

A unit change under a stable name is the most dangerous variant because every downstream calculation
continues to work. Milliseconds read as seconds produce durations 1,000 times too large, which is
absurd enough to be caught; seconds read as milliseconds produce durations 1,000 times too small,
which looks like a performance improvement and is often welcomed.

Distinguishing (b) from (a): an enum addition produces a step confined to the affected category. The
discriminator is to segment the metric by the enum field; a step in exactly one segment identifies (b).

Distinguishing (c) from (a): nullability changes step the count of rows rather than the value
distribution. Comparing row counts across the boundary separates them, and a coercion rule that maps
null to zero will also make a mean collapse toward zero, which is a third fingerprint.

Correction detail: migrating consumers before the writer changes is the only ordering that avoids a
window of misinterpretation. The reverse order creates a period during which the writer emits new
semantics and some consumers still assume the old, and that period's data is permanently unusable
for cross-boundary comparison.

Prevention: encoding the unit in the field name is unfashionable and effective. `duration_ms` cannot
be silently changed to seconds without a rename, and a rename is visible in every diff.

Verification boundary: the two-row replay must include a row at the boundary minute. Rows far from the
boundary will pass under both interpretations if the consumer is idempotent in its parsing, which
most are.

### 8.18 CI-18 — an integer accumulator overflows and corrupts a total

| Field | Content |
|---|---|
| ID | CI-18 |
| Category | NUMERICAL_EDGE |
| Symptom | A daily total is large and negative, or resets to a small number, at high volume |
| Competing causes | (a) a 32-bit signed accumulator wrapped; (b) a conversion to a narrower type in serialisation; (c) a counter reset by a deployment |
| Discriminating check | Compute the total in 64-bit arithmetic and compare; an exact 2^32 offset confirms (a) |
| Correction | Widen the accumulator and reject unsafe conversions at the boundary |
| Prevention | Use a type whose maximum is documented against the expected volume, with a test at that maximum |
| Verification | Replay a batch at the int32 maximum and one at maximum plus one |

The arithmetic: a signed 32-bit integer holds up to 2,147,483,647. A count accumulating an event per
case at 1,200 cases per day reaches that limit after `2,147,483,647 / 1,200 = 1,789,569.7` days, which
is 4,902 years — so counts do not overflow. The quantities that do overflow are byte counts,
millisecond durations, and products: at 4 KB per event, the byte total reaches the limit after
`2,147,483,647 / 4,096 = 524,288` events, which at 21,600 events per day is 24.3 days. A monthly
reporting job therefore overflows a 32-bit byte accumulator, and the symptom appears as a negative
total in the fourth week.

Distinguishing (b) from (a): a serialisation narrowing produces a correct in-process value and a
wrong stored value. The discriminator is to log the value before serialisation and compare against
the stored value for the same batch.

Distinguishing (c) from (a): a reset produces a plausible small number rather than a wrapped negative,
and it coincides with a deploy. The discriminator is the process start time.

Correction detail: rejecting unsafe conversions means the boundary checks magnitude before narrowing,
which requires the boundary to know the destination width. That is a schema property and should be
declared alongside the field.

Prevention: documenting the maximum against expected volume is the only check that catches the
product case, where two individually safe quantities multiply into an unsafe one. A test at the
maximum is written once and catches every future narrowing.

Verification boundary: maximum-plus-one is the exact test. A test at half the maximum passes under
every width and proves nothing, which is why overflow tests that use "a large number" usually fail to
find the defect.

### 8.19 CI-19 — cache expiry sends a synchronized stampede to the origin

| Field | Content |
|---|---|
| ID | CI-19 |
| Category | SCALE_FAILURE |
| Symptom | Origin load spikes on a regular period, then recovers, with no change in request volume |
| Competing causes | (a) identical TTLs expiring together; (b) a deploy flushing the cache; (c) a key change that orphaned the cached entries |
| Discriminating check | Plot origin concurrency against time and look for periodicity matching the TTL |
| Correction | Add jitter to TTLs and single-flight the misses for a key |
| Prevention | Make TTL a distribution, not a constant, and cap origin concurrency per key |
| Verification | Replay a synchronized expiry and a jittered expiry at the same hit-rate target |

The arithmetic of the spike: with 60 worker processes and a single popular key expiring, all 60 miss
simultaneously and issue origin calls within the same few milliseconds. The origin sees 60 concurrent
requests for one key where it expects one, a 60× amplification. With jitter distributed over 10% of a
300-second TTL, the same 60 misses spread over 30 seconds, which is 2 per second — absorbable.

Distinguishing (b) from (a): a deploy flush coincides with a deployment record. The discriminator is
the deployment timestamp against the spike timestamp, which takes seconds to check.

Distinguishing (c) from (a): a key change produces a *step* in miss rate that does not recover, while
TTL synchronization produces a *periodic* spike that recovers. The shape of the series separates them
without any further instrumentation.

Correction detail: single-flight is the more valuable half because it bounds the amplification to one
regardless of concurrency. Jitter spreads the load but leaves the per-key amplification at the
concurrency of the workers that happen to miss.

Prevention: TTL as a distribution is a one-line change in most caches and is skipped because the
constant is simpler to reason about. The simplification is real and its cost is a periodic outage.

Verification boundary: the jittered replay must confirm that the *hit rate* is unchanged. A jitter
that improves the spike by reducing the hit rate is not a fix, and the miss rate is the guardrail.

### 8.20 CI-20 — an approximate answer is consumed as if it were exact

| Field | Content |
|---|---|
| ID | CI-20 |
| Category | CONCEPT_CONFUSION |
| Symptom | Two systems disagree by a small amount on the same quantity and both are correct |
| Competing causes | (a) one system uses a sampled or sketching estimate; (b) one system rounds at a different precision; (c) the two compute different quantities that share a name |
| Discriminating check | Compare each value against an exact computation on a small population |
| Correction | Label the approximate value with its error bound and expose an exact path for decisions |
| Prevention | Encode approximation in the return type so the consumer must opt in to the bound |
| Verification | Replay a small population where exact and approximate must coincide |

Distinguishing (a) from (b): an approximation's error varies with population size, while rounding error
is bounded by half the last digit and independent of size. The discriminator is to plot the gap
against population size; growth identifies (a).

Distinguishing (c) from both: the two quantities differ systematically rather than randomly, and the
difference persists on any population. A single small population where the exact values can be
computed by hand resolves all three at once, which is why the check is specified as a small
population rather than a large one.

Correction detail: the exact path is not optional. A consumer that must make a decision — approve,
dispose, bill — requires an exact value or a documented tolerance, and an approximate value without a
bound cannot supply either.

Prevention: encoding approximation in the type is the only prevention that survives a new consumer.
An approximation carried as a plain number will be treated as exact by the next developer, who has no
way to know.

Verification boundary: a population small enough that the sketch is exact — typically under a few
hundred items for a cardinality sketch — is the only case where the two must agree, and disagreement
there indicates a defect rather than an error bound.

### 8.21 CI-21 — authorization is evaluated after data is loaded

| Field | Content |
|---|---|
| ID | CI-21 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A forbidden record appears in a response, a log, or a cache, even if the response is later filtered |
| Competing causes | (a) the query scope is broader than the authorization scope; (b) an intermediate cache stores unfiltered rows; (c) the filter operates on a field that is itself derived from the row |
| Discriminating check | Trace the requested scope and the returned scope through each layer and compare |
| Correction | Authorize before access and construct the query from the authorized scope |
| Prevention | Make the data-access interface require an authorization token as an argument |
| Verification | Request a forbidden descendant and a permitted sibling and inspect every layer's record |

The distinction from CI-01 is direction of harm. CI-01 produces a wrong decision. CI-21 produces a
correct decision plus a disclosure, and the disclosure persists in logs and caches even when the
response is filtered. Filtering after load is a correctness measure, not a confidentiality measure.

Distinguishing (b) from (a): a cache leak is visible by inspecting the cache keys and values
directly, independent of any request. If the cache holds rows the requester was never authorized for,
the mechanism is (b) and the query fix alone will not close it.

Distinguishing (c) from (a): a filter on a derived field fails when the derivation is stale. The
discriminator is to compare the derived field's value against its source at read time for a sample of
rows.

Correction detail: constructing the query from the authorized scope is different from adding a
predicate. A predicate can be forgotten in a second code path; an interface that takes the scope as
an argument cannot be called without one.

Prevention: requiring the token as an argument is enforceable by the type system in most languages,
which makes it the only prevention that does not depend on review attention.

Verification boundary: the permitted sibling confirms the filter is not simply denying everything.
A test with only a forbidden record passes under a query that returns nothing at all.

### 8.22 CI-22 — NaN enters an ordered comparison and vanishes from a filter

| Field | Content |
|---|---|
| ID | CI-22 |
| Category | NUMERICAL_EDGE |
| Symptom | Row counts differ between two logically equivalent queries, and the missing rows are all invalid |
| Competing causes | (a) NaN propagated from an invalid computation; (b) a null was coerced to NaN rather than to a sentinel; (c) the filter uses an ordered predicate that is false for NaN |
| Discriminating check | Count rows failing `x = x`; a non-zero count confirms NaN is present |
| Correction | Test the finite predicate explicitly before any ordering, and quarantine invalid values |
| Prevention | Reject non-finite values at the boundary where they can first appear |
| Verification | Replay a record with a valid value and one with an invalid value through both predicate forms |

The mechanism is exact and worth stating: every ordered comparison with NaN is false, including
`x <= 250`. So `keep rows where x <= 250` drops NaN, and `keep rows where not (x > 250)` keeps NaN,
because `x > 250` is also false and its negation is true. Two predicates that are equivalent for
every real number are not equivalent on NaN.

Distinguishing (b) from (a): a coercion to NaN produces NaN where the source had no value, so the NaN
population matches the null population exactly. Comparing the two counts separates a genuine
computation defect from a coercion policy.

Distinguishing (c) from (a) and (b): (c) is a defect in the filter, not the data. The check is to
count NaN rows before the filter and after it; a change in count identifies (c) even when (a) or (b)
is also present.

Correction detail: the finite test must come before the ordering, and the quarantine must record the
source. A quarantine that drops the row without recording where the NaN came from leaves the
upstream defect to recur, which is the difference between a correction and a filter.

Prevention: rejecting at the boundary where a non-finite value can first appear means the rejection
is close to the cause. Rejecting at the consumer is far from the cause and produces the same
investigation repeatedly.

Verification boundary: both predicate forms must be exercised. A test that only uses the natural form
confirms the filter works and hides the divergence.

### 8.23 CI-23 — a retry storm hides the original fault

| Field | Content |
|---|---|
| ID | CI-23 |
| Category | SCALE_FAILURE |
| Symptom | Every dependency reports timeouts and the original failing component is invisible in the graphs |
| Competing causes | (a) no retry budget so attempts grow with failure; (b) synchronized exponential backoff across clients; (c) a circuit breaker that never opens because its threshold is above the storm's success rate |
| Discriminating check | Plot attempt rate and success rate on the same axis; a rising attempt rate with a flat low success rate confirms (a) |
| Correction | Budget retries per request, jitter the backoff, and open the breaker on the failure that started it |
| Prevention | Make the retry budget part of the request context and count it in the same place as the deadline |
| Verification | Replay one injected failure with a budget of 2 and one with no budget, at the same arrival rate |

The amplification arithmetic is F3: at a 50% per-attempt failure rate the expected attempt count is
`1/(1-0.5) = 2`, doubling downstream load; at 80% it is 5.0; at 90% it is 10.0. A storm is the regime
where `p` is itself a function of load, so the equation has no stable solution and the system sits at
whichever equilibrium its budget permits.

Distinguishing (b) from (a): synchronized backoff produces a *periodic* attempt pattern with retry
waves. The attempt-rate series will show a sawtooth at the backoff interval, which is a shape that an
unbudgeted constant-rate retry does not produce.

Distinguishing (c) from (a): a breaker with too high a threshold shows a success rate above its
threshold throughout, so the breaker's own state log shows it never opened. That log is the fastest
check in the entry.

Correction detail: opening the breaker on the originating failure rather than on a rate means the
breaker must be keyed to the dependency that failed first. Rate-based breakers are simpler and they
fail exactly when a cascade makes every dependency fail at once.

Prevention: counting the budget in the same place as the deadline means one object carries both, and a
request cannot exceed either without the other noticing. Separating them is how a deadline gets
honoured while the budget is exceeded.

Verification boundary: the budgeted and unbudgeted replays must use the same arrival rate, which
isolates the amplification from the load.

### 8.24 CI-24 — a formula is applied outside the range where it was derived

| Field | Content |
|---|---|
| ID | CI-24 |
| Category | CONCEPT_CONFUSION |
| Symptom | A model produces a confident number in a regime where its assumptions are known to fail |
| Competing causes | (a) a stationarity assumption was violated by a trend; (b) an independence assumption was violated by correlation; (c) a linearity assumption was violated near a boundary |
| Discriminating check | Construct the boundary case explicitly and compare the formula's output with the observed behaviour |
| Correction | Restrict the application to the valid range, or switch to a method whose assumptions hold |
| Prevention | State the validity condition next to the formula, and test at the boundary in the same suite |
| Verification | Replay an interior case and the boundary case through the same code path |

The clearest instance in this catalogue is the queue factor `rho / (1 - rho)`. It is derived for a
stable queue with `rho < 1` and constant moments. At `rho = 0.9` the factor is 9.0; at 0.99 it is
99.0; at 1.0 it is undefined. A system reporting a wait estimate at `rho = 0.98` is reporting its own
asymptote, and the estimate will be wrong in a direction that depends on which assumption broke
first.

Distinguishing (a) from (b): a trend violation produces a systematic bias in one direction over time,
visible as a residual pattern. An independence violation produces underestimation of variance with no
directional bias. Plotting residuals against time separates them.

Distinguishing (c) from (a) and (b): a linearity violation is confined to a region, so the error
appears only for cases near the boundary while interior cases remain accurate. Segmenting by distance
from the boundary identifies it.

Correction detail: restricting the range is a real correction when the range is knowable in advance,
which it often is — utilisation is observable. When the range is not knowable, the correction is to
switch methods, and the entry should say so rather than implying a range restriction is always
available.

Prevention: stating the validity condition beside the formula is why this file writes constraints in
uppercase inside every derivation block. A constraint that lives in a paper's methods section does not
travel to the code.

Verification boundary: the boundary case must be an actual input at the edge, not an extrapolation.
Testing at `rho = 0.5` and calling it a boundary test is the error that lets the defect ship.

### 8.25 CI-25 — a human override is recorded as an action without a reason

| Field | Content |
|---|---|
| ID | CI-25 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | The agreement metric shows high concordance while reviewers report frequent disagreement |
| Competing causes | (a) overrides are recorded without a reason code; (b) overrides occur in a channel outside the ledger; (c) the recommendation is silently recomputed after the override |
| Discriminating check | Compare the final action field to the recommendation field for every case; disagreements without a reason confirm (a) |
| Correction | Require a reason code at the moment of override and persist it with the actor |
| Prevention | Make the reason field required by the override endpoint, so no override can be submitted without one |
| Verification | Replay one override with a reason and one attempt without; the second must be rejected |

The mechanism corrupts two analyses at once. The agreement metric counts an unreasoned override as
agreement if the comparison is naively made on the final action alone, so concordance is overstated.
And the net-release estimate attributes the reviewer's extra minutes to the workflow's decision time,
so the workflow looks slower than it is for exactly the cases where the reviewer intervened.

Distinguishing (b) from (a): an out-of-channel override leaves no ledger row at all, so the case's
final action and the ledger's final action differ. That comparison identifies (b) immediately,
whereas (a) leaves a ledger row with a null reason.

Distinguishing (c) from (a): a silent recomputation changes the recommendation value after the
override, so the stored recommendation no longer reflects what the reviewer saw. Comparing the stored
recommendation against the value in the reviewer's session record identifies (c), and the correction
is to make the recommendation immutable once displayed.

Correction detail: requiring the reason at the moment of override is the only point where the reason
is known. A follow-up survey collects a rationalisation, and a retrospective label from a supervisor
collects a judgement, neither of which is the reviewer's actual reason.

Prevention: a required field at the endpoint is enforceable and cannot be bypassed by a second client.
A convention that reviewers "should add a reason" is bypassed by the first urgent case.

Verification boundary: the rejected submission is the test that the requirement is enforced rather
than merely documented.

**Takeaway:** every entry above shares one property — the system keeps working, reports a plausible
number, and fails at the boundary a human is least likely to test.

## 9. Complexity & Cost Analysis

### 9.1 Cost of the checks themselves

A diagnostic that costs more than the defect is not worth running, and the catalog's checks differ in
cost by two orders of magnitude. The table below gives the cost model for each class.

| Check class | Cost per run | Grows as | Reusable | Typical use |
|---|---|---|---|---|
| Row-count reconciliation | one query | `O(rows)` scan | yes, per batch | layer 1 triage |
| Timestamp monotonicity | one pass | `O(rows)` | yes | layer 1 triage |
| Single-record hand reconstruction | 5–15 minutes of attention | `O(1)` | no | localise a numeric defect |
| Configuration hash diff | one comparison | `O(1)` | yes | rule out drift |
| Key-based effect count | one group-by | `O(rows log rows)` | yes | duplicate detection |
| Stratified re-computation | one query per stratum | `O(k * rows)` | partly | population questions |
| Boundary replay | one record plus one | `O(1)` data, `O(setup)` | yes | verification |
| Load replay at 10x | provisioned environment | `O(10 * rows)` | no | scale claims |
| Fresh audit sample | `n` adjudications | `O(n)` human minutes | no | rate claims |

The last row is the expensive one and the one most often underfunded. At 4 minutes per adjudication,
a 216-case sample costs 864 minutes, or 14.4 reviewer-hours. A sample large enough to resolve a 12%
threshold from 8% needs roughly the square of the ratio of the current width to the target width, so
halving the width from 8.7 to 4.35 percentage points requires `4 × 216 = 864` cases, which is 57.6
reviewer-hours. That is the real price of a defensible rollback threshold.

### 9.2 Per-item versus per-volume cost

The single most useful distinction in this section is whether a cost is per item or per volume,
because it determines which defects a pilot can detect.

| Cost type | Formula | Detected at pilot volume | Detected at 10x |
|---|---|---|---|
| Per item, constant | `c * n` | yes | yes |
| Per item with a bounded term | `c * n + k` | partially | yes |
| Per item with a product | `a * b * n` | only if either factor is large | yes |
| Queueing at high utilisation | `n * T_s / (1 - rho)` | no | yes |
| Retry amplification | `n * 1/(1-p)` | no, unless `p` is high at pilot | yes |
| Fan-out expansion | `n * f` | yes, since `f` is visible | yes |

The two "no" rows are the reason scale failures cluster: both are flat in `n` at low load and
divergent near a boundary. A pilot at 15% of volume has `rho = 0.37` in this catalogue's example,
which is deep in the benign region; a production run at the same service time would be at `rho =
0.51` and one with a 20% volume increase at `rho = 0.64`, where the `1/(1-rho)` factor has moved from
1.59 to 2.78.

### 9.3 The cost of not fixing

An S2 defect that produces a wrong reversible decision has a cost measured in rework. An S1 defect
that produces an irreversible effect has a cost measured in consequences plus audit work. The catalog
does not price consequences, but it does make one asymmetry explicit: the cost of a *check* is bounded
and known, while the cost of a *missed* defect is unbounded in principle and typically discovered at
the worst moment.

The practical rule that follows: run the cheap checks (row counts, timestamp monotonicity, config
hash) on every batch, because their cost is a query. Run the expensive checks (fresh audit, load
replay) on a schedule that matches the defect's severity, not on every batch.

### 9.4 Complexity of a diagnosis

A diagnosis that enumerates `c` competing causes and runs one check per cause costs `O(c)` in
attention, and the catalog's insistence on at least two causes doubles the cost of every entry.
That doubling is the price of not being wrong half the time. In practice `c = 2` or `3` is the limit
of useful enumeration for a single session; beyond that the mechanisms become variants of each other
and the checks stop discriminating.

The exception is a defect family where one check separates many causes at once. The key-based effect
count separates three causes of duplication in a single group-by, which is why CI-05's check is
described as the fastest in the catalog.

**Takeaway:** budget diagnostics by severity, and note that per-item checks cannot detect
utilisation-dependent failures no matter how carefully they are run.

## 10. Key Theorems / Results Reference

### 10.1 Conservation of work

For a closed system with no loss and no duplication, the number of items accepted equals the number
completed plus the number in process. This is the basis of F1 and F2, and its force in diagnosis is
negative: any observation that violates conservation implies loss or duplication, and those are the
two mechanisms that break the model. Applied to a queue, an observed completion count that exceeds
acceptances without a corresponding backlog change identifies duplicate processing rather than
efficiency.

### 10.2 Little's law

`L = lambda * W` where `L` is mean number in system [items], `lambda` the arrival rate [items/s], and
`W` the mean time in system [s]. It holds for any stable system under very weak assumptions — no
distributional requirement, only that the averages exist.

The illustrative check: at `lambda = 0.033333` cases/s and `W = 25.0` minutes `= 1,500` s, `L =
0.033333 * 1,500 = 50` cases in system on average. That is 50 of the 60 seats' worth of cases either
in service or waiting, which is consistent with a utilisation near 0.51 once service time is
separated from waiting. Little's law therefore gives a cheap consistency check on any three of
`L`, `lambda`, `W` when the fourth is measured.

Its limitation is equally important: it says nothing about variance. Two systems with identical `L`,
`lambda`, and `W` can have radically different tails, so Little's law cannot validate a latency claim.

### 10.3 The `1/(1-p)` amplification result

For independent attempts with success probability `1-p`, the expected number of attempts is `1/(1-p)`.
This is the geometric distribution's mean, and its consequence in distributed systems is that retry
load diverges as `p → 1`. The result is exact for independent attempts and is *wrong* for
correlated failures, which is why a retry storm (where `p` rises with load) has a different
qualitative behaviour than the formula suggests: it has two equilibria instead of one.

### 10.4 The standard error of a proportion

`SE = sqrt(p(1-p)/n)` with the normal approximation valid when `n*p >= 5` and `n*(1-p) >= 5`. This
is the basis of F6 and of every audit sample size decision. The practical consequence is the square-
root law: precision costs the square of the desired improvement, so halving a confidence interval
costs four times the sample.

### 10.5 The square-root law for measurement

Closely related and worth separating: the standard error of any mean scales as `sigma/sqrt(n)`, so
the *relative* precision depends on the coefficient of variation `sigma/mu`, not on the mean alone.
A measurement problem with a coefficient of variation above about 1 requires a sample larger than the
inverse square of the effect size, which is a way of saying that small effects in noisy systems are
expensive to detect.

Illustratively: a mean of 4.3 minutes with a standard deviation of 6.1 has `CV = 6.1/4.3 = 1.4186`,
so `CV^2 = 2.0124`. Detecting a 1-minute effect at conventional confidence with 80% power requires
roughly `n ≈ 16 * CV^2 / (effect/mean)^2`; with `effect/mean = 1/4.3 = 0.23256` the denominator is
`0.05408`, giving `n ≈ 32.198 / 0.05408 = 595.4`, so about 596 cases. The exact formula and the power
convention matter less than the order of magnitude: a one-minute effect needs hundreds of cases, not
tens.

### 10.6 The idempotency result

A transport-layer retry cannot guarantee a single effect. This is not a theorem but a consequence of
the two-generals problem: with an unreliable channel, no finite protocol distinguishes a lost message
from a lost acknowledgement. The only workable construction is a stable key plus a deduplicating
store, which moves the guarantee from the transport to the data layer. Every CI-05 correction follows
from this and no amount of retry tuning substitutes for it.

### 10.7 The rounding-tie result

Half-up and half-even differ only at exact ties, and the set of exact ties in a binary representation
is a set of measure zero in theory and a set of common values in practice — 0.5, 1.5, 2.5 and their
scaled variants appear constantly in thresholds. The practical result is that a tie test is a
one-line addition with an outsized detection rate.

### 10.8 What is deliberately absent

No theorem here establishes that a given defect is *likely*. The catalog is a diagnostic resource, not
a risk model, and a reader who wants frequency should measure their own defect distribution rather
than infer it from the ordering of these entries.

**Takeaway:** the theorems that matter are the conservation and amplification results, because they
turn an observed violation into a named mechanism.

## 11. Interview / Exam Questions

The questions below test three things in roughly equal proportion: whether a candidate can compute the
expected magnitude, whether they can select a discriminating check, and whether they can state a claim
at the precision the evidence supports. Each answer is specific to its question; a candidate who
reaches for a general diagnostic slogan will not score.

### Q1. `lambda = 2.5` items/s, `mu = 2.0` items/s for 300 s. What is the backlog, and what is the drain time at `mu_r = 2.8` with `lambda_r = 2.5`?

**Answer:** backlog is `(2.5 - 2.0) * 300 = 0.5 * 300 = 150` items. Drain is `150 / (2.8 - 2.5) =
150 / 0.3 = 500` s. The check: surplus `0.3 * 500 = 150` items, matching.
**Reasoning:** the backlog grows at the imbalance, not at the arrival rate, and the drain time is
governed by the surplus after recovery, not by the recovered service rate.
**Follow-up:** if the recovery rate were 2.55 instead, what happens and why does it surprise operators?
**Red flag:** computing `150 / 2.8` and getting 53.6 s, which uses the recovery rate as if arrivals
had stopped.
**Scoring:** full credit requires both numbers, the independent check, and the observation that the
drain time is reciprocal in the surplus.

### Q2. At `lambda = 2.5` and `mu = 2.55`, how long to drain 150 items, and what does that imply about adding capacity?

**Answer:** surplus is `0.05` items/s, so `150 / 0.05 = 3,000` s, which is 50 minutes. A 2% increase
in service rate over the base case changes the drain time by a factor of 6 against the 500 s answer.
**Reasoning:** the reciprocal relationship means small capacity additions produce wildly different
outcomes depending on the surplus, so "add a little capacity" is not a predictable intervention.
**Follow-up:** what would you rather measure than service rate, and why?
**Red flag:** answering with a percentage improvement in service rate as if it translated linearly.
**Scoring:** full credit gives 3,000 s and articulates the non-linearity as the operational point.

### Q3. A pipeline reports 1,200 cases per day for 20 days, 18 events per case, 4 KB per event. How much log volume, and what happens at 20 KB per event?

**Answer:** `1,200 * 20 * 18 = 432,000` events; at 4 KB that is `1,728,000 KB = 1,687.5 MB = 1.65 GB`.
At 20 KB it is 8.25 GB, a fivefold increase for identical traffic.
**Reasoning:** the product of three easily-changed quantities, none of which looks dangerous alone; a
payload widening is a five-parameter decision disguised as a one-line change.
**Follow-up:** which of the three factors would you bound first, and how?
**Red flag:** multiplying days by events and forgetting the per-case factor, or reporting GB without
converting.
**Scoring:** full credit gives both volumes and identifies the payload as the cheapest factor to bound.

### Q4. Two teams compute an error rate over the same 20 windows and get 1.09% and 5.10%. Both are arithmetically correct. What happened?

**Answer:** one computed `sum(events)/sum(items)` and the other averaged the per-window rates. With
window volumes in a 1:10 ratio, the unweighted mean gives the small window equal weight despite its
tenfold higher rate. `12/1,100 = 0.010909` versus `(0.10 + 0.002)/2 = 0.051`.
**Reasoning:** the mean of ratios is only the ratio of sums when exposure is equal, which is why a
test built on equal-volume windows never exposes the defect.
**Follow-up:** construct the test that would have caught this before the disagreement.
**Red flag:** asserting one team made an error, or blaming rounding.
**Scoring:** full credit identifies both aggregations, reproduces both values, and prescribes an
unequal-volume test window.

### Q5. A service reports 12% exception rate on a 216-case audit and the rollback threshold is 12%. Is the threshold crossed?

**Answer:** the interval is `[0.0767, 0.1633]`, so the observed rate is consistent with anything from
7.7% to 16.3%, and a point comparison against 12% is a coin flip. To halve the width you need 864
cases, not a better estimate.
**Reasoning:** `SE = sqrt(0.12 * 0.88 / 216) = 0.022111`, half-width `1.96 * 0.022111 = 0.0433`. The
threshold decision requires the interval position, not the point estimate.
**Follow-up:** what sample size would separate 12% from 8% at 95% confidence?
**Red flag:** saying "12% equals 12%, so no", or proposing to raise the sample to 300.
**Scoring:** full credit computes the interval, states the undecidability, and gives a sample-size
answer in the hundreds.

### Q6. A field named `handle_time` reports 4,200 where it used to report 4.2. Which issue is this, and what is the discriminating check?

**Answer:** CI-17, a unit change under a stable name — the value is now in milliseconds. The check is
to compare the schema version and the field's declared unit across the two periods; a version bump
with a unit change confirms it. A 1,000× step is the signature.
**Reasoning:** a stable name with changed semantics survives every downstream calculation because the
arithmetic keeps working. The dangerous direction is the reverse — seconds reported as milliseconds —
because a 1,000× reduction looks like an improvement.
**Follow-up:** why is CI-17 distinguishable from CI-09, which also changes values historically?
**Red flag:** treating it as a formatting bug, or silently dividing by 1,000 in the consumer.
**Scoring:** full credit names CI-17, gives the 1,000× signature, and notes the dangerous direction.

### Q7. Two runs on identical inputs differ by 3%. Configuration hashes match. What are your two remaining hypotheses?

**Answer:** a dependency default changed (CI-13, mechanism b), or the declared input set is incomplete
so the runs consumed different data (CI-13, mechanism c). The check for the first is the lockfile and
library version; for the second, hash the full input payload rather than the declared field list.
**Reasoning:** a matching configuration hash eliminates the application's own configuration, which
narrows the space sharply. Both remaining causes are outside the artefacts the team normally inspects.
**Follow-up:** what change to the pipeline would have made both causes impossible?
**Red flag:** "nondeterminism" with no attempt to localise, or re-running until the numbers agree.
**Scoring:** full credit names both hypotheses with their respective artefacts.

### Q8. A byte accumulator reports a negative total in the fourth week of a monthly report. Walk through the arithmetic.

**Answer:** at 4,096 bytes per event and 21,600 events per day, the accumulator reaches 2,147,483,647
after `2,147,483,647 / 4,096 = 524,288` events, which is `524,288 / 21,600 = 24.3` days. A 32-bit
signed accumulator wraps at that point.
**Reasoning:** counts do not overflow in realistic horizons, but products do, and a bytes-per-event
factor turns a four-hourly counter into a three-week one.
**Follow-up:** which test would have caught it, and at what value?
**Red flag:** "increase to 64-bit" without identifying which quantity overflowed, or blaming storage.
**Scoring:** full credit derives the 24.3-day horizon and prescribes a maximum-plus-one test.

### Q9. A query written `WHERE latency <= 250` returns 40,000 rows; `WHERE NOT (latency > 250)` returns 40,900. No data changed. Explain.

**Answer:** 900 rows have non-finite latency. Every ordered comparison with NaN is false, so the
negated predicate keeps them and the direct one drops them. The count of rows failing `latency =
latency` confirms it.
**Reasoning:** the two predicates are equivalent over the reals and inequivalent over the extended
values, which is a property of the comparison operator rather than of the SQL dialect.
**Follow-up:** what is the correction, and why is a filter not sufficient?
**Red flag:** calling it a database bug, or adding `OR latency IS NULL` without checking for NaN.
**Scoring:** full credit identifies non-finite values, the predicate asymmetry, and the need to
quarantine with provenance.

### Q10. A p99 latency rises while p50 is flat and pool utilisation is 38%. Which issue, and what is the single fastest check?

**Answer:** CI-07, key skew. The fastest check is p50 and p99 computed per key — one key whose p99
dominates while its rate exceeds a single partition's capacity identifies it.
**Reasoning:** genuine saturation raises both percentiles and utilisation together. A tail-only rise
with idle capacity is a lane problem, not a capacity problem.
**Follow-up:** why is sharding the hot key often the wrong first move?
**Red flag:** recommending more replicas, or attributing it to retries without checking attempt counts.
**Scoring:** full credit names key skew and explains why the utilisation evidence rules out saturation.

### Q11. A cache has 60 workers, a single hot key, and a 300-second TTL with no jitter. Estimate the origin spike and propose the fix.

**Answer:** all 60 workers miss within the same few milliseconds, so the origin sees 60 concurrent
requests for one key, a 60× amplification. Jitter of 10% of the TTL spreads them over 30 seconds, or
2 per second. Single-flight bounds it at one regardless.
**Reasoning:** the amplification equals the concurrency of workers that miss together, so it scales
with fleet size rather than with traffic.
**Follow-up:** what guardrail must the fix not break?
**Red flag:** proposing a longer TTL, which lengthens the cycle without reducing the spike.
**Scoring:** full credit gives 60×, the jitter arithmetic, and names hit rate as the guardrail.

### Q12. `p = 0.85` per attempt, retries unbounded, 4 downstream calls per attempt, 3 requests/s. What is the effective load, and what bounds it?

**Answer:** attempts per request `1/(1-0.85) = 6.667`, so calls per request `6.667 * 4 = 26.67`, and
`3 * 26.67 = 80` calls/s against a nominal `3 * 4 = 12` calls/s — a 6.67× amplification. A retry
budget, not a backoff, bounds it, because the formula has no finite value when `p` depends on load.
**Reasoning:** the geometric mean is exact for independent attempts and fails for correlated ones,
which is why the budget is structural rather than a tuning parameter.
**Follow-up:** at what `p` does a budget of 3 become binding?
**Red flag:** adding exponential backoff and declaring the problem solved.
**Scoring:** full credit gives the 6.67× amplification and identifies the budget as the bound.

### Q13. Your promotion report says "the workflow reduced handling time by 22%". Rewrite it so it is defensible given a 40-case paired pilot.

**Answer:** "In a 40-case paired pilot with estimated counterfactuals, median elapsed time fell from
25.0 to 19.5 minutes, a 22.0% reduction. Net reviewer-minute release was 4.3 minutes per case with a
95% lower bound of 2.35 minutes; no intake-wide effect has been measured."
**Reasoning:** the rewrite adds population, comparator, the unit conversion, and an explicit
not-measured statement. Each addition corresponds to a defect in the catalog.
**Follow-up:** which clause would a finance reviewer challenge first?
**Red flag:** keeping 22% and adding "approximately", or removing the number entirely.
**Scoring:** full credit names the population, the comparator, the converted unit, and the absence of
an intake-wide measurement.

### Q14. Which is worse: a leaked field in a feature set, or a missing exclusion count? Argue both directions.

**Answer:** the leak produces a confident wrong answer that survives review because it is flattering;
the missing exclusion count produces a rate of unknown population that survives review because nobody
asks. The leak is worse for a single decision because it can drive an irreversible action; the missing
count is worse for a portfolio because it contaminates every later comparison.
**Reasoning:** both are record defects that make downstream arithmetic unfalsifiable, and neither is
detectable from the reported number.
**Follow-up:** which one does a build-time test catch more reliably?
**Red flag:** choosing one without acknowledging the other's failure mode.
**Scoring:** full credit argues both directions and names the reversibility difference.

### Q15. Give a check that separates CI-05 from CI-25 when the symptom is "the case was handled twice".

**Answer:** group by actor. Two distinct actors on one case is CI-25 territory (routing or an
unrecorded override); one actor with two effect rows and different keys is CI-05. The ledger's actor
and key fields separate them in a single group-by.
**Reasoning:** duplication of effect is a key defect; duplication of attention is an authority or
routing defect. They produce the same surface count and different ledger shapes.
**Follow-up:** what would a third mechanism be that matches neither?
**Red flag:** assuming any double-handling is idempotency.
**Scoring:** full credit gives the group-by and names the two shapes.

## 12. Quick Reference Card

| Step | Rule |
|---|---|
| Triage order | row counts → timestamp monotonicity → hand reconstruction → version diff → name the quantity → actor field |
| Two causes minimum | never accept a single mechanism for a symptom; the first plausible cause is usually the second-most likely |
| Containment | prefer defer-with-reason and last-known-good config; both are safe under unknown cause |
| Numeric edges | zero denominators are undefined not zero; NaN fails every ordered comparison; int32 dies on products not counts |
| Scale claims | per-item checks cannot detect utilisation or retry failures; require a target-scale measurement arm |
| Unit discipline | put the unit in the field or metric name; `1,048,576 / 1,000,000 = 1.048576` is the byte trap |
| Evidence status | every reported number carries one of measured / derived / illustrative / not-measured |
| Authority | a threshold is not an authorisation; irreversible effects need a named actor or an approved policy version |
| Verification | always replay the trigger **and** a boundary record; a fix that passes only the trigger has been narrowed to it |
| Cost rule | run cheap checks per batch, expensive checks per severity, and price an audit sample before trusting a threshold |

