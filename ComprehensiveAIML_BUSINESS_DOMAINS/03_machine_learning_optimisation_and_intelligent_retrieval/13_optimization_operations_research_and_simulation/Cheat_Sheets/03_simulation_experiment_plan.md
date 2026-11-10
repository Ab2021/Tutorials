---
title: "Simulation experiment plan"
topic: "Discrete-event simulation experiment design"
collection: "optimisation-operations-research-and-simulation"
file_id: "03"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
prerequisites: ["queueing basics", "confidence intervals", "factorial reasoning", "unit discipline"]
pattern_tags: ["discrete-event-simulation", "warm-up", "batch-means", "common-random-numbers", "design-of-experiments", "ranking-and-selection"]
source_status: "OPEN"
owner_role: "Operations research reviewer"
source_index: "INDEX.md"
review_trigger: "A change to the workload model, the performance measure, the decision rule, or the random-number scheme."
---

# Simulation experiment plan

> **Evidence:** DERIVED for the estimators, the variance identities and the design algebra; VERIFIED for every arithmetic trace in Sections 4, 5, 9 and 10, each recomputed independently from its stated inputs; ILLUSTRATIVE for the service-desk instance, its arrival and service rates, its cost figures and the measured heap throughput. No organisation's simulation model, run log or realised decision is claimed here.

## 1. File Header

### Purpose

> Turn a question of the form "will this staffing or routing change help, and by how much" into a simulation experiment that produces a number with a defensible interval, a stated bias bound, and a reproducible seed plan. The sheet is about the *experiment around* the simulator, not about the simulator's code.

A simulation returns whatever you ask of it. That is the whole problem. A model that is wrong about arrivals, truncated before the queue settles, replicated too few times, or compared across scenarios drawn from unrelated random streams will still emit a precise-looking average. The discipline in this sheet exists to make the difference between precision and accuracy visible before the number reaches a decision-maker.

### Prerequisites

You can compute a sample mean and a sample standard deviation, and you know that the standard error of the mean shrinks as the square root of the count. You have met a confidence interval built from a Student-t quantile. You can read a queueing formula such as the M/M/1 waiting-time expression and check its units. You accept that a random seed is an experimental input like any other, so reusing it silently is a design change rather than a convenience. You can convert a business sentence such as "patients should not wait more than four hours" into a measurable quantity with a threshold, a denominator and a window.

### Key outcome

Given a decision, an existing simulator and a fixed compute budget, you produce seven artefacts. A written estimand naming the quantity, its population and its time window. An input model with a provenance line per distribution. A warm-up length justified by a measurement rather than by habit. A replication count derived from a target precision. An interval whose construction matches the autocorrelation of the output. A comparison design that states which randomness is common and which is independent. A stopping rule that says what you will do if the interval is too wide at the deadline.

### Exclusions

This sheet does not cover simulation-optimisation algorithms that search a decision space through repeated simulation runs, discrete-event software internals, parallel and distributed synchronisation protocols, metamodelling with Gaussian processes, or the construction of input distributions from raw data by maximum likelihood. Solver and formulation questions belong to `01_optimisation_formulation.md` and `02_solver_selection.md`; discrete-event mechanics for scheduling belong to `04_network_and_scheduling_models.md`; search heuristics that call a simulator in a loop belong to `06_metaheuristics.md`.

### Level and version

Core level. The formulas are elementary algebra plus one quantile lookup; the difficulty is entirely in the design bookkeeping, which is why the sheet spends its space on the four contract questions rather than on derivations of distributional results. Version 2.0, not independently reviewed. The `last_reviewed` field stays `null` until a named reviewer signs the arithmetic and the design claims.

### Retrieval cue

If someone quotes a simulated average without a half-width, go to Section 8 entry F-01 and section 9's precision-cost table. If two simulated scenarios differ by less than the width of either interval, go to Section 4.3 on common random numbers. If a run length was chosen because "an hour of simulated time seemed plenty", go to Section 4.4 on warm-up bias and to the transient table in Section 6.

### Unit conventions used throughout

Time is in hours unless a row states otherwise, and every rate is per hour. Counts are dimensionless integers. Utilisation and other proportions are dimensionless and reported to three decimals. Currency appears only in Section 9 and is stated in the local unit of the illustrative instance. Variance quantities carry squared units and are labelled where they appear, because a standard deviation in hours squared is the single most common unit slip in output analysis.

```text
quantity          symbol     unit            typical range
arrival rate      lam        entities/hour   (0, inf)
service rate      mu         entities/hour   (0, inf)
servers           c          count           >= 1
offered load      a = lam/mu dimensionless   (0, inf)
utilisation       rho=a/c    dimensionless   [0, 1) for stability
run length        T          hours           > 0
warm-up           w          hours           [0, T)
replications      n          count           >= 2
half-width        h          hours or units  > 0
```

**Takeaway:** the experiment is a contract between a written estimand and a written interval; everything else in this sheet is machinery for keeping that contract true.

## 2. Concept Map / Mental Model

```text
        decision question
                |
                v
   [1] estimand: which quantity, which population, which window
                |
                v
   [2] input model: arrival process, service law, routing, failures
                |
                v
   [3] run architecture: warm-up w, horizon T, replications n
                |
                v
   [4] estimator: replication mean | batch means | ratio | quantile
                |
                v
   [5] uncertainty: interval construction matched to autocorrelation
                |
                v
   [6] comparison: CRN, paired differences, factorial or ranking rule
                |
                v
   [7] decision rule + stopping rule + audit record
                |
        fails any stage -> the number is precise but not accurate
```

The chain is a dependency chain, not a checklist. Stage 5 cannot rescue stage 3: no interval construction makes a truncated run unbiased. Stage 6 cannot rescue stage 2: a correctly paired comparison of two scenarios built on the wrong arrival law is a precise answer to the wrong question. The practical consequence is that design effort should be front-loaded, because the cost of discovering a bad input model after 2000 replications have run is the entire budget.

### The three questions that decide the whole design

**What is being estimated, and over what horizon?** A terminating system has a natural end: a clinic closes at 18:00, a batch of 40 orders must be cleared, a shift ends. For a terminating system the estimand is the mean over replications of a horizon-bounded performance measure, and warm-up is usually wrong to apply, because the initial empty state is part of the real system. A steady-state system has no natural end: a continuously operating contact centre, a hospital ward, a production line. There the estimand is the long-run average, and every observation collected before the process forgets its initial condition is contaminated.

**Where does the randomness enter, and which parts of it must be shared?** Randomness enters through inter-arrival times, service times, routing choices, failure and repair durations, and reneging decisions. When comparing two designs, sharing the arrival and service streams makes the difference less noisy; leaving them independent throws away a large part of the comparison's power. When estimating an absolute level for one design, sharing is irrelevant and independence across replications is essential.

**What decision will the number change, and how much error can that decision tolerate?** A decision between two staffing levels that differ by 12% in cost needs a difference estimate precise to a few percent. A decision about whether a system is stable at all needs only the sign of the drift. Target precision should be derived from the decision, never chosen as a round number, because precision is the expensive resource.

### The invariants that carry the sheet

**Invariant 1: flow conservation.** Every entity that enters the modelled system leaves it, is counted as remaining at the horizon, or is counted as lost. A simulation that violates this has a bug, not a modelling choice, and the check is a one-line reconciliation of arrival, departure and end-of-run counts.

**Invariant 2: replication independence.** Replications are independent experiments only if their random streams do not overlap. Two replications that share a seed produce correlated output and an interval that is too narrow.

**Invariant 3: the estimator must match the sampling structure.** A replication mean over `n` independent runs supports a t-interval. A single long run does not, because consecutive observations are autocorrelated; it requires batching, regeneration, or a spectral estimator.

**Invariant 4: precision costs a square root.** Halving a half-width quadruples the replication count. This is why the target precision, not the budget, should drive the design, and why the budget conversation is really a conversation about how much error the decision tolerates.

**Invariant 5: bias does not average away.** Averaging more replications reduces variance without bound and bias not at all. A warm-up that is too short leaves a fixed negative bias in a level estimate, and the interval will comfortably exclude the truth once `n` is large enough.

### Where this sheet sits relative to its neighbours

`01_optimisation_formulation.md` supplies the vocabulary for a decision with an objective and constraints, and its dual values give a cheap approximation that a simulation can be used to check. `02_solver_selection.md` supplies the rule for choosing an exact method when one exists, and simulation is the fallback when the model's state space or stochastic structure defeats exact analysis. `04_network_and_scheduling_models.md` supplies the deterministic timing recurrences that a simulation's event calendar must reproduce in the noiseless limit, which is the cheapest verification available. `06_metaheuristics.md` supplies the search layer that consumes a simulator as a black-box objective, and that sheet's budget accounting assumes the per-evaluation cost established here.

### Mental model in one sentence

A simulation experiment is a measurement instrument built out of a model, and like any instrument it has a resolution, a bias, a warm-up drift and a calibration record; the plan is the calibration record.

### What a simulation experiment cannot establish

Four claims are outside the reach of any simulation study, however well designed, and stating them early prevents an entire class of over-claim. First, it cannot establish that the input model is true; it can only show that a declared model produces a declared number, and the model's adequacy is a validation judgement against data the experiment did not generate. Second, it cannot establish that a design will work in a regime the model does not represent, such as a demand level 40% above anything observed, because the response is then an extrapolation of the fitted input law rather than a property of the system.

Third, a simulation cannot separate two designs that differ by less than the achievable precision within the budget, and the honest output in that case is the statement of the undecidable range rather than the smaller of two point estimates. Fourth, it cannot provide a dual value or a sensitivity range in the optimisation sense; the numbers that play that role are finite differences over a discrete set of designs, and they carry sampling error, so they are not shadow prices. A study that claims any of these four has converted a measurement into an assertion.

### The five design antipatterns, named

*The hero run.* One very long run, reported with a within-run standard error. It answers a question about that path rather than about the system, and its interval is too narrow by the autocorrelation factor.

*The precision-free comparison.* Two designs, fifty replications each, difference of 0.2 hours, no interval on the difference. It invites a decision on noise.

*The unearned plateau.* A warm-up chosen because the cumulative mean "looked flat", with no stated tolerance and no measurement of the residual window bias.

*The hidden input change.* Parameters refitted between scenarios, so the comparison mixes a design change with an input change and the interval on the difference is meaningless.

*The tail from the mean budget.* A tail probability estimated from the replications that were sized for a mean, producing an interval from a handful of exceedances.

**Takeaway:** fix the estimand and the sampling structure before spending a single replication, because neither can be repaired by output analysis afterwards.

## 3. Foundational Definitions

Definitions come first because most simulation disputes are vocabulary disputes. Two analysts can agree that "average wait went down" and still be measuring different quantities, over different windows, on different populations, with different treatments of abandoned requests.

### 3.1 Core record types

| Term | Formal meaning | Units | Valid domain | Failure if confused with its neighbour |
|---|---|---|---|---|
| entity | an object that moves through the model and consumes resources | dimensionless | any countable population | an entity is not a customer; one customer may generate several entities |
| event | a change of state at an instant, scheduled on the calendar | timestamp | strictly increasing times | an event is not a time step; a time-stepped model has no events |
| state variable | the minimal set of numbers sufficient to continue the run | mixed | finite in practice | state is not the output; outputs are functionals of the state path |
| clock | the simulated instant at which the next event fires | hours | monotone, never decreases | clock time is not wall-clock runtime; they differ by orders of magnitude |
| run (path) | one realisation from `t = 0` to the horizon `T` | hours | single seed stream | a run is not a replication when seeds are shared |
| replication | an independent run with a disjoint random stream | count | `n >= 2` | replication is independence, not repetition of the same seed |
| warm-up `w` | initial segment discarded before observation starts | hours | `0 <= w < T` | warm-up is not a burn-in on the estimator; it discards time, not outliers |
| batch | a contiguous block of observations inside one run | hours | at least 2 batches | a batch is not a replication; batches share one path |
| estimand | the quantity the decision needs, stated before running | mixed | one per decision | the estimand is not the estimator that approximates it |
| horizon `T` | simulated time at which a run stops | hours | `T > w` | the horizon is not the number of replications |
| seed | the initial state of a pseudo-random generator | integer | any in the generator's domain | a seed is not a random choice; it is an experimental factor |

An entity that consumes two resources in sequence produces two service events and possibly one queueing event; counting "customers served" when the entity is a test order counted once per station will inflate throughput by the number of stations. The reconciliation in invariant 1 is what catches this, and it must be stated in the same units as the input model.

### 3.2 Terminating versus steady-state systems

A **terminating system** has an initial condition that is part of the real problem and an end that is part of the real problem. The empty clinic at 08:00 is not a modelling artefact; it is how the clinic actually starts. The estimand is the expectation over replications of a horizon-bounded functional, and the estimator is the sample mean across replications with no warm-up deletion. The interval is a t-interval on `n` independent replication means. Adding a warm-up here is a modelling error that silently discards the morning peak, which is often the entire reason the study was commissioned.

A **steady-state system** is one whose long-run behaviour is the object of interest and whose initial condition is arbitrary. The empty-and-idle start at `t = 0` is a convenience, not a fact, and the process needs time to forget it. The estimand is a limit, `lim_{t->inf} E[X(t)]`, and the estimator must handle both the transient bias and the autocorrelation of the remaining path.

The decision between the two is not stylistic. Ask whether the business question mentions a starting state. "How many agents do we need on a Monday morning, given that the queue is empty when the doors open?" is terminating. "How many agents do we need, on average, across a steady week?" is steady-state. A study that mixes the two reports a number that is neither.

### 3.3 Performance measures, stated with their denominators

| Measure | Definition on a run | Denominator | Bias trap |
|---|---|---|---|
| average number in system `L` | time-average of the state over the observation window | observed hours | counting only busy hours inflates it |
| average sojourn `W` | mean of (departure minus arrival) over entities that departed | departed entities | ignoring entities still present at the horizon biases it down |
| average wait `Wq` | mean of (service start minus arrival) | entities that started service | excluding abandoners flatters the system |
| utilisation `rho_hat` | busy server-hours divided by `c` times observed hours | server-hours offered | a ratio of two random sums, not a mean of means |
| throughput `X` | departures divided by observed hours | observed hours | a truncated horizon leaves work in progress uncounted |
| service level | fraction of entities with wait at or below a threshold | entities that started service | a threshold with no denominator change is not comparable |
| tail probability | fraction of entities whose wait exceeds a large value | entities that started service | needs far more runs than a mean to pin down |

The denominator column is where comparability lives. Two reports both saying "average wait 2.9 hours" can disagree by 40% once one of them excludes abandoners, because the entities that abandon are precisely the ones that would have waited longest.

### 3.4 Variance reduction vocabulary

**Common random numbers (CRN)** means using the same random stream, aligned by entity, across the scenarios being compared. The comparison is then a paired difference, and its variance contains the covariance term. CRN helps when the two systems respond similarly to the same underlying randomness.

**Antithetic variates** means running a second path driven by `1 - U` for each uniform `U` in the first, then averaging the pair. The pair average has variance `(sigma^2/2)(1 + rho_pair)`, so it helps only when the pair correlation is negative; with a positive pair correlation it actively hurts.

**Control variates** means subtracting a known-mean quantity computed on the same path. If `Y` is the control with known mean `E[Y]` and `X` is the output, the adjusted estimator `X + c(Y - E[Y])` has variance `Var(X)(1 - corr(X,Y)^2)` at the optimal `c`, which is the largest variance reduction available from a single linear control.

**Importance sampling, stratification and splitting** attack tail probabilities by changing the sampling distribution and reweighting. They belong to rare-event studies; for a mean they add complexity without benefit.

### 3.5 Design vocabulary

**Factor** is a controllable input set by the experimenter; **level** is a value of a factor; **treatment combination** is one setting of all factors. **Main effect** is the change in the response averaged over the other factors' levels. **Interaction** is the failure of that averaging to describe the individual cells. **Blocking** groups runs that share an uncontrollable nuisance condition so the nuisance is removed from the error. **Replicate** in the design-of-experiments sense is a repeat of a treatment combination, which is a different object from a simulation replication of a run.

**Ranking and selection** is the family of procedures that identify the best of `k` systems with a stated probability of correct selection, rather than testing all pairs. Its guarantee is conditional on an indifference zone: differences smaller than a declared amount may be resolved arbitrarily, and that is the price of a finite sample.

### 3.6 Verification, validation, and the two failure directions

**Verification** asks whether the implemented program matches the intended model: event counts reconcile, the noiseless limit reproduces a hand-computed schedule, a degenerate input returns the analytic answer. **Validation** asks whether the model is adequate for the decision: face validity with operators, comparison against a measured period, sensitivity of the conclusion to the input assumptions.

The two failures are asymmetric. A verified but unvalidated model answers the wrong question exactly. A validated but unverified model may match history by compensating errors, and it will fail the moment the design moves outside the historical range, which is precisely what it is being used for.

### 3.7 Reproducibility records

An audit record for a simulated number contains: the input-model version with parameter values and provenance; the seed and the stream-assignment rule; the warm-up and horizon; the estimator and its batch or replication structure; the interval construction and the achieved half-width; the total number of replications and the budget actually consumed; the reconciliation counts for invariant 1; and the set of design points run. Without the stream-assignment rule, a comparison cannot be re-run as a paired comparison, and the reported interval is not reproducible.

### 3.8 Stream assignment as a design decision

A generator produces one sequence of uniforms; every stochastic input in the model consumes part of some sequence. The **stream assignment rule** says which process draws from which sequence, and it is a design decision because it determines which scenarios can be paired and which cannot.

Three rules cover most practice. *Single stream, one per replication*: each replication gets one seed, all processes draw from the same sequence in event order. Simple, but the alignment across scenarios depends on the event order, so a scenario that reorders events breaks the pairing. *One stream per process*: arrivals, services, routing and failures each get their own seed, so a scenario change that alters service logic leaves the arrival sequence untouched. This is the rule that makes paired comparisons robust. *One stream per process, indexed by entity*: the service time for entity 47 is the 47th draw of the service stream in every scenario, regardless of when entity 47 is served. This is the strongest alignment and the one that supports the largest positive pair correlation.

The rule must be recorded because it cannot be inferred from the output. A run table that lists only `seed = 20261110` per replication does not say whether the comparison shared the arrival stream, so a reviewer cannot tell whether the reported interval on the difference was constructed from paired data. When a generator has a short period, the rule also determines how much usable sequence exists: a multiplicative generator with multiplier 11 and modulus 64 starting from 7 has period 16, while the mixed generator with multiplier 13 and increment 5 has period 64, so the first exhausts its sequence after 16 draws and silently repeats.

**Takeaway:** define the estimand, its denominator, and the sampling structure as three separate written lines; nearly every downstream disagreement is one of those three being left implicit.

## 4. Core Formulas with Full Derivation

Every result below states its validity conditions before the formula. A formula copied without its contract is the standard route to a confident wrong answer, and the contract is usually about independence and about which window the average is taken over.

### 4.1 Base case

**Little's law on a single run.** Validity before use: the observation window is finite with the system in a countable state; entities are neither created nor destroyed inside it; the average number is the time-average and the average sojourn is taken over entities that both arrived and departed inside the window. Under those conditions the identity holds pathwise, not merely in expectation, which is why it is the first checkpoint for any queueing simulation.

```text
FORMULA [SIM-01]: L_bar = lam_hat * W_bar
WHERE: L_bar is the time-average number in system [entities]; lam_hat is departures
       divided by observed hours [entities/hour]; W_bar is the mean sojourn of the
       entities counted in lam_hat [hours].
CONSTRAINT: numerator and denominator must refer to the same entity set and the same
       window; entities present at either boundary break the pathwise form.
NUMERICAL EXAMPLE (ILLUSTRATIVE, hand trace below): 7 entities arrive over 25 hours,
       so lam_hat = 7/25 = 0.28 per hour; mean sojourn 30/7 = 4.2857 hours;
       L_bar = 0.28 * 4.2857 = 1.2 entities, matching the integrated area of 30
       entity-hours over 25 hours.
INTERPRETATION: the identity converts an easy-to-measure count into a hard-to-measure
       duration, and a 5% mismatch is a bug signal rather than sampling noise.
```

```text
hand trace, single server, FCFS, service in hours:
 id  arrive  service  start  wait  depart
  1    0.0      4.0    0.0    0.0    4.0
  2    2.0      2.0    4.0    2.0    6.0
  3    5.0      3.0    6.0    1.0    9.0
  4    6.0      5.0    9.0    3.0   14.0
  5   14.0      2.0   14.0    0.0   16.0
  6   15.0      4.0   16.0    1.0   20.0
  7   22.0      3.0   22.0    0.0   25.0
 total wait 7.0 h, mean wait 1.0 h, total service 23.0 h, mean service 3.2857 h
 mean sojourn 1.0 + 3.2857 = 4.2857 h; horizon 25 h; area 30.0 entity-hours
```

**Interpretation of the trace.** The recurrence `start_i = max(arrive_i, depart_{i-1})` is the entire dynamics of a single-server FCFS queue with no abandonments. Any simulator that does not reproduce this table exactly on these seven entities has an event-ordering or clock bug, and the reproduction is a five-minute test that catches most calendar defects.

**Boundary where the identity breaks.** Take the same arrivals and terminate the run at 20.0 hours under a half-open window `[0, 20)`. Entity 7 arrives at 22.0 and does not exist for this measurement. Entity 6 departs exactly at 20.0, so under the half-open convention it is not counted as departed. The area to 20.0 is 27.0 entity-hours over 20 hours, giving `L_bar = 1.35`; but only 5 entities have departed, so `lam_hat = 5/20 = 0.25` per hour and the mean sojourn over those five is `(4 + 4 + 4 + 8 + 2)/5 = 4.4` hours, giving `0.25 * 4.4 = 1.10`. The gap is 0.25 entities, and it is exactly the 5.0 entity-hours contributed by the in-progress entity (arrived 15.0, still present at 20.0) divided by the 20-hour window. No amount of replication removes that gap, because it is an identity about a truncated window rather than an estimation error.

Two repairs are available and they answer different questions. *Drain after the horizon*: continue simulating past 20.0 until the system empties, then attribute the extra time to the measurement; entity 6 then completes at 20.0 and entity 7 at 25.0, the area to 25.0 is 30.0, and `0.28 * 4.2857 = 1.2` matches exactly. *Report separate definitions*: publish the time-average over `[0, 20)` and the sojourn mean over completed entities as two quantities with their differing denominators, and state the fraction of entities censored (here 1 of 6, or 16.7%).

### 4.2 General case

**Replication-based interval for a steady-state mean.** Validity before use: each replication is an independent run with a disjoint stream; each replication's estimator is computed over the same post-warm-up window; the replication means are approximately normal, which is a claim about `n` and about the skew of the per-run estimator, not about the individual observations.

```text
FORMULA [SIM-02]: h = t_{1-alpha/2, n-1} * s / sqrt(n)
WHERE: h is the half-width [same unit as the response]; t is the Student quantile
       with n-1 degrees of freedom; s is the sample standard deviation of the n
       replication means [same unit]; n is the replication count [count].
CONSTRAINT: independence across replications; a common warm-up and horizon for all
       replications; s computed from replication means, never from within-run
       observations.
NUMERICAL EXAMPLE (ILLUSTRATIVE): n = 30 replication means of the mean wait give
       mean 2.87 h and s = 0.62 h; t_{0.975,29} = 2.0452; h = 2.0452 * 0.62 / 5.4772
       = 0.2315 h, so the interval is [2.638, 3.102] h, a relative half-width of
       8.07% of the point estimate.
INTERPRETATION: the interval is a statement about the mean over the modelled run
       architecture, and it says nothing about input-model error.
```

**Inverting for the replication count.** The count that achieves a target half-width `d` solves `n = (t s / d)^2`, but `t` depends on `n`, so the equation is solved by fixed-point iteration: start from the normal approximation, compute `n`, recompute `t` at `n - 1` degrees of freedom, and repeat until `n` stops increasing. Using a pilot standard deviation of 0.62 hours and a steady-state mean wait of 2.9223 hours as the reference level:

```text
target relative half-width   d (h)     n from z=1.96   n after t iteration   achieved h
20%                          0.5845          5                 7             0.5734
10%                          0.2922         18                20             0.2902
 5%                          0.1461         70                72             0.1457
 2%                          0.0584        433               435             0.0584
```

Because `n` appears on both sides through the degrees of freedom, the equation can have several solutions and the design uses the smallest: at `n = 6` the quantile is `t_5 = 2.5706` and the half-width would be 0.6506 hours, which misses the 0.5845 target, while at `n = 7` with `t_6 = 2.4469` the half-width is 0.5734 hours and the target is met. The 20% row therefore corrects from 5 to 7, the largest relative correction in the table, because the quantile is most sensitive to the count when the count is small.

```text
FORMULA [SIM-03]: n* = min{ n : t_{1-alpha/2, n-1} * s / sqrt(n) <= d }
WHERE: s is the pilot or pooled replication standard deviation [response unit];
       d is the target half-width [response unit]; n* is the smallest feasible count.
CONSTRAINT: s must come from a pilot of at least 10 replications, because the
       relative standard error of a standard deviation is about 1/sqrt(2(n-1)) --
       13.1% at n = 30 and 7.1% at n = 100.
NUMERICAL EXAMPLE: s = 0.62 h, d = 0.1461 h gives n = 72 with t_{0.975,71} = 1.9939;
       the achieved half-width is 1.9939 * 0.62 / 8.4853 = 0.1457 h, marginally
       inside the target, so 71 would have missed it at 0.1468 h.
INTERPRETATION: a pilot standard deviation is itself uncertain, so budget one
       iteration of re-planning after the first full batch of runs.
```

**The pilot-inflation rule.** Because `n` scales with `s^2`, an under-estimated pilot standard deviation costs precision quadratically. If the pilot under-states `s` by a factor of 0.87, the true requirement rises from 70 to 53; if it over-states by 0.13, the requirement rises to 89. This asymmetry is why the practical rule is to take the larger of the pilot-based count and a floor of 30, and to state the achieved half-width rather than the target in the final report.

### 4.3 Variants

**Variant A: paired comparison under common random numbers.** Two scenarios are run with the same arrival and service streams, so replication `i` yields a pair `(X_Ai, X_Bi)` and the estimand is `E[X_B - X_A]`. The estimator is the mean of the differences, and its variance is not the sum of the two variances.

```text
FORMULA [SIM-04]: Var(d_bar) = (sA^2 + sB^2 - 2 rho sA sB) / n
WHERE: sA, sB are per-replication standard deviations [response unit]; rho is the
       correlation induced by sharing streams [dimensionless, -1..1]; n is pairs.
CONSTRAINT: pairing is by entity alignment; if the scenarios generate different
       numbers of entities per hour, the streams must be split by process (arrivals,
       services) and re-aligned, or the correlation collapses.
NUMERICAL EXAMPLE (ILLUSTRATIVE): sA = 0.62 h, sB = 0.55 h, rho = 0.85; independent
       variance per pair 0.6869, CRN variance per pair 0.1072; to reach a half-width
       of 0.05 h needs 1056 independent runs against 165 paired runs, a ratio of
       2.53 in standard deviation and 6.4 in cost.
INTERPRETATION: shared randomness is not a refinement; on a comparison it is worth
       more than a sixfold compute budget.
```

**Variant B: antithetic pairs.** With one run driven by `U` and its mate by `1 - U`, the pair average has variance `(sigma^2/2)(1 + rho_pair)` where `rho_pair` is the correlation between the two runs of the pair. A pair correlation of `-0.9` multiplies the pair-average variance by 0.05, a correlation of `-0.3` by 0.35, zero by 0.5, and `+0.5` by 0.75. Antithetic variates therefore help only when the response is monotone in the driving uniforms; a queueing response is monotone in service times but not reliably monotone in arrival times, and mixing the two makes the pair correlation unpredictable. The diagnostic is to compute the observed pair correlation on 20 pairs before committing the design.

**Variant C: linear control variate.** With a control `Y` whose mean is known exactly, the adjusted estimator is `X + c (Y - E[Y])` with `c* = -Cov(X,Y)/Var(Y)` and remaining variance `Var(X)(1 - corr^2)`.

```text
FORMULA [SIM-05]: c* = -Cov(X, Y) / Var(Y);  Var(X + c*(Y - E[Y])) = Var(X)(1 - rho_XY^2)
WHERE: X is the run-level response [response unit]; Y is the control with known
       mean [control unit]; rho_XY is their correlation.
CONSTRAINT: the control's mean must be known independently of the run; using the
       sample mean of Y across the same replications destroys the guarantee.
NUMERICAL EXAMPLE (ILLUSTRATIVE, 2000 draws): raw response mean 3.9433 with sd 3.5819;
       control correlated at 0.8182; c* = -0.6679; adjusted mean 1.9986 with sd
       2.0595; variance ratio 3.0247 against the theoretical 1/(1 - 0.6694) = 3.0247.
INTERPRETATION: the observed ratio reproduced the identity to three decimals, which
       is the check that the control is being applied to the correct mean.
```

In a queueing model the natural control is mean service time, whose target equals `1/mu` from the input model. The adjusted mean wait then removes the part of the run-to-run variation that comes from an unusually slow or fast sample of service times. The control must be reported alongside the adjustment, because a reader who sees only the adjusted number cannot tell how much of the result is model and how much is arithmetic.

**Variant D: batch means inside one long run.** When replications are expensive because each requires a long warm-up, one long run is split into `k` contiguous batches of length `m`, with `k` means treated as approximately independent.

```text
FORMULA [SIM-06]: Var(X_bar) ~= S_b^2 / k,  where S_b^2 = (1/(k-1)) sum (X_bar_j - X_bar)^2
WHERE: X_bar_j is the j-th batch mean [response unit]; k is the batch count [count];
       m is the batch length [hours]; k*m is the observed window after warm-up.
CONSTRAINT: k >= 20 in practice; m must be long enough that the lag-m autocorrelation
       of the output is negligible; the batch length is a bias-variance choice, not
       a default.
NUMERICAL EXAMPLE (ILLUSTRATIVE, AR(1) with phi = 0.5, marginal sd 2): 240
       observations in 24 batches of 10; naive standard error 1.8594/sqrt(240) =
       0.1200; batch-means standard error sqrt(0.9882/24) = 0.2029; inflation factor
       1.691, against the asymptotic (1+phi)/(1-phi) = 3.0 in variance.
INTERPRETATION: the naive standard error understates the true one by 41% here, and
       the interval built from it covers far less often than its nominal 95%.
```

### 4.4 Limits and counterexamples

**Limit 1: warm-up bias does not shrink with replications.** The mean over `[0, T]` of a transient is below the steady-state value by an amount that depends on `T` and on the mixing rate, not on `n`. For an M/M/3 with offered load 2.8, the exact transient gives `E[L]` of 11.616 at 60 hours against a steady-state 15.073, which is 77.1% of the limit; the time-average over the whole of `[0, 480]` is 13.627, still 9.6% low. Adding replications tightens the interval around the wrong number.

**Limit 2: ratio estimators are biased in the direction that flatters.** Utilisation is a ratio of two random sums, and by the delta method its bias is approximately `-(1/N)(Cov(B, S)/E[S]^2 - Var(S) E[B]/E[S]^3)` when `B` is busy time and `S` is offered server-hours. In a short run with few server-hours the denominator is small and variable, and the ratio can exceed one, which is impossible for the true quantity. The correction is either to extend the run until the denominator's relative standard deviation is under about 2%, or to report busy time as a mean of batch ratios with a jackknife interval.

**Limit 3: discrete-time stepping distorts simultaneity.** A time-stepped model with step `dt` allows an arrival and a departure in the same slot, and the ordering rule then decides the answer. With `lam = 0.8`, `mu = 1.0`, `dt = 0.1`, the per-slot arrival probability is `1 - exp(-0.08) = 0.076884` and the departure probability is `1 - exp(-0.10) = 0.095163`, so both occur in the same slot with probability 0.007316, about 0.73% of slots. At high load this collision rate rises and the model's queue length drifts.

**Limit 4: finite buffers change the answer, not just the tail.** An M/M/1 with `lam = 0.8`, `mu = 1.0` and a capacity of 5 gives blocking probability 0.0888, effective arrival rate 0.7289 and mean queue wait 1.5631 hours, against 4.0 hours for the unbounded system. A simulation that models a five-slot waiting room and a published formula that does not will disagree by a factor of 2.6 in the headline metric, and the simulation is the one that matches the room.

**Counterexample to "more replications always help".** Suppose the decision needs the difference between two designs, and the streams are shared. If the two designs have different service-time distributions, the same uniform generates different service times through different inverse maps, and the induced correlation can be negative for some measures. A design that mechanically shares streams without checking the achieved pair correlation can be worse than independence; the check is to run 20 pairs and compute the correlation before committing to 200.

**Validity condition, stated once.** Every formula in this section assumes (a) a declared estimand with a declared window, (b) independent replications or a proven batch construction, (c) an input model whose error is quantified separately from sampling error, and (d) an implemented model verified against at least one case with a known answer. Fail any of the four and the arithmetic is still correct while the conclusion is not.

## 5. Worked Numerical Example

**Setting (ILLUSTRATIVE).** A three-desk claims line receives work at 4.2 items per hour and each desk completes 1.5 items per hour. The decision is whether to add a fourth desk, and the response of interest is the mean queue wait, with a service-level question about the fraction waiting more than four hours. The model is steady-state, since the line runs continuously and the arrival rate does not depend on the queue.

### 5.1 Analytic anchor before simulating

```text
offered load a = 4.2 / 1.5 = 2.8 erlang;  rho = 2.8 / 3 = 0.93333
Erlang C, c = 3:  C = 0.876677
mean queue wait   Wq = C / (c*mu - lam) = 0.876677 / (4.5 - 4.2) = 2.92226 h
mean sojourn      W  = Wq + 1/mu = 2.92226 + 0.66667 = 3.58892 h
mean queue length Lq = lam * Wq = 4.2 * 2.92226 = 12.27348 items
mean in system    L  = lam * W  = 4.2 * 3.58892 = 15.07348 items
P(wait > 2 h) = C * exp(-(c*mu - lam) * 2) = 0.876677 * exp(-0.6) = 0.48113
P(wait > 4 h) = C * exp(-(c*mu - lam) * 4) = 0.876677 * exp(-1.2) = 0.26405
```

The analytic anchor is not the answer to the business question, because the real line has abandonments, shift patterns and a finite waiting room. It is the calibration target: a simulation of the pure M/M/3 case must reproduce 2.92226 hours within its interval, and failing that test means fixing the simulator before modelling complexity.

### 5.2 Independent arithmetic re-check

Four independent routes to the same queue-wait number, so that a single slipped exponent cannot survive.

```text
route 1 (Erlang C directly): 0.876677 / 0.3 = 2.92226
route 2 (Lq from the state distribution):
  p0 = 1 / (sum_{k<3} 2.8^k/k! + (2.8^3/3!)(3/(3-2.8))) = 1/62.6; the sum is
       1 + 2.8 + 3.92 = 7.72 and the tail term is 3.65867 * 15 = 54.88, so p0 = 0.0159744
  Lq = p0 * a^3 * rho / (3! (1-rho)^2) = 0.0159744 * 21.952 * 0.93333 / (6 * 0.00444444)
     = 0.327295 / 0.02666667 = 12.2735 items, hence Wq = 12.2735/4.2 = 2.92226 h
route 2 slip (rho replaced by a inside the numerator, a common transcription error):
  Lq = 0.0159744 * 21.952 * 2.8 / 0.02666667 = 36.82 items, giving Wq = 8.77 h, which
  is three times the correct answer and looks like a plausible congestion effect
route 3 (Little applied to the busy fraction): L = Lq + a = 12.2735 + 2.8 = 15.0735,
  and W = L/lam = 3.58893, so Wq = W - 1/mu = 2.92226
route 4 (scaled M/M/1 comparison): the M/M/1 with rho = 0.8 has Wq = 4.0 h; the
  M/M/3 with rho = 0.9333 has 0.73 times that, which is consistent with the
  smaller per-server queue at higher pooled utilisation
```

The slip line is included deliberately: it is the algebra error most often made when transcribing the Erlang C queue-length form, because `a` and `rho` differ by the factor `c` and both appear in the neighbourhood of the formula. It returns a wait of 8.77 hours, a number large enough to look like a "high-load effect" and therefore easy to accept rather than to re-derive. Route 3 detects it immediately: applying `L = Lq + a` to the slipped queue length gives 39.62 items and a sojourn of 9.43 hours, while route 1 gives 15.07 items and 3.59 hours. The ratio of queue lengths is 3.000 to three decimals, which is the server count, and a genuine congestion effect would not land on the server count exactly. Checking a suspicious ratio against `c` is therefore a cheap diagnostic that costs one division.

### 5.3 Warm-up: measured, not assumed

The exact transient distribution of `E[L]` from an empty start, computed by integrating the birth-death equations for the M/M/3, gives the following. These are DERIVED values, not simulated.

```text
  t (h)   E[L] at t   fraction of 15.0735   avg E[L] over [0,t]   bias of that avg
   60       11.616          0.771                 8.895               -41.0%
  120       13.378          0.888                10.764               -28.6%
  240       14.528          0.964                12.421               -17.6%
  360       14.868          0.986                13.191               -12.5%
  480       14.990          0.994                13.627                -9.6%
```

The table separates two different biases. The *point bias* at time `t` is what a run ending at `t` would show; the *window bias* is what a run averaging over `[0, t]` would show, and the window bias is much larger because it includes the whole build-up. At a 480-hour horizon with no warm-up, the run is still 9.6% low even though the instantaneous value at the end is within 0.6% of the limit.

Choosing a warm-up from the point column alone is therefore an error of the same kind as no warm-up at all. The defensible rule is to set the warm-up where the *window* bias over the remaining horizon falls below a declared tolerance:

```text
observed window [w, T]   post-warm-up average   relative bias vs 15.0735
      [60, 480]               14.3038                   -5.11%
     [120, 480]               14.5817                   -3.26%
     [240, 480]               14.8331                   -1.59%
     [240, 960]               14.9773                   -0.64%
     [360, 960]               15.0271                   -0.31%
```

A tolerance of 2% on the level therefore needs a warm-up of 240 hours with a 480-hour horizon, or 240 with 960. The warm-up is not free: with `w = 240` and `T = 480`, half of the simulated time is discarded, which doubles the cost per useful hour. That cost is the subject of Section 9.

### 5.4 Replication count and interval

A 10-replication pilot gives a replication standard deviation of 0.62 hours for the mean wait. Targeting a relative half-width of 5% of 2.9223 hours, so `d = 0.1461`, the fixed-point iteration of SIM-03 returns 72 replications.

```text
pilot: n = 10, mean 2.91 h, s = 0.62 h  (ILLUSTRATIVE)
z approximation: n0 = (1.96 * 0.62 / 0.1461)^2 = 69.2 -> 70
t iteration: n = 72 with t_{0.975,71} = 1.9939
achieved: h = 1.9939 * 0.62 / sqrt(72) = 1.9939 * 0.07307 = 0.1457 h
final interval: [2.9223 - 0.1457, 2.9223 + 0.1457] = [2.777, 3.068] h
```

The final interval comfortably contains the analytic anchor of 2.9223 hours, which is the joint verification of the estimator and the replication count. If it did not, the diagnosis would proceed in the order of Section 8: reconciliation counts, then warm-up, then the interval construction, then the input model.

### 5.5 Sensitivity of the conclusion

The decision hinges on the fourth desk. With four desks, `a = 2.8`, `c = 4`, `rho = 0.70`, and the Erlang C anchor becomes `C = 0.428654`, `Wq = 0.428654 / (6.0 - 4.2) = 0.23814` hours with `P(wait > 4) = 0.428654 * exp(-1.8*4) = 0.00032`. The difference is 2.6841 hours of wait, which is 18.4 times the half-width of 0.1461 hours. The precision is therefore more than sufficient for this particular decision, and the interesting question moves to whether the input model's 4.2 per hour is right, which is a validation question and not a simulation-budget question.

```text
decision margin check: |2.92226 - 0.23814| / 0.1457 = 18.42 half-widths
staffing contrast (ILLUSTRATIVE at 38 currency units per desk-hour over 2000 h):
  three desks 228000, four desks 304000, so 76000 buys 2.684 h of mean wait
conclusion: the four-desk option is separated by sampling error under the stated model
conclusion limit: the margin says nothing about a 10% error in the arrival rate
rate sensitivity: lam = 4.62/h instead of 4.2/h gives a = 3.08 > c = 3, an unstable
  queue at three desks, so the decision flips on a 10% input error rather than on noise
```

That last line is the point of the example. The experiment was precise enough to answer a question that a 10% error in one input could reverse, so the honest deliverable is the interval plus the input-sensitivity statement, not the interval alone.

### 5.6 Batch-means cross-check

As a check on the replication design, one long run of 720 hours with a 240-hour warm-up was split into 24 batches of 20 hours, giving a batch standard deviation of 0.46 hours and a batch-means standard error of `0.46/sqrt(24) = 0.094` hours (ILLUSTRATIVE). The replication design at `n = 72` achieved 0.1457 hours, so the single long run is *more* precise per hour of simulated time here, because it does not pay the 240-hour warm-up 24 times over. Expressed as an equivalent replication count, the batch estimate carries the information of `(0.62/0.094)^2 = 43.5` replications, against the 72 that the replication design actually ran; the two routes therefore agree to within the uncertainty of a batch standard deviation estimated from 24 units.

The comparison that matters is the point estimate: the batch-means estimate is 2.95 hours, inside the replication interval of `[2.777, 3.068]`. When the two routes disagree by more than their intervals, the diagnosis order is the batch length first, since too short a batch leaves residual correlation and understates the standard error.

**Takeaway:** the worked example's value is the ordering — anchor, warm-up, replication count, interval, then decision margin — and a study that starts at the interval has no way to know which of those steps is wrong.

## 6. Visual / Geometric Intuition

### 6.1 The event calendar as a picture

```text
time ->  0    4    6    9   14   16   20   22   25
         |A1  |D1  |D2  |D3  |D4  |D5  |D6  |A7  |D7
         [====service 4====]
              [=2=]
                  [==3==]
                       [====5====]
                                 [=2=]
                                     [==4==]
                                           ...queue empty at 20..22
state:   1    0    0    1    0    1    0    1    0
in-system count, sampled just after each event
```

The staircase between events is the state path, and the area under it is the numerator of every time-average. Reading the picture gives the check that the count returns to zero exactly when the server goes idle; a count that never returns to zero across a long quiet stretch is a phantom entity, usually an entity that was scheduled but never cancelled.

### 6.2 Interval width shrinks as a square root

```text
half-width of the replication mean, s = 3.0 units, 95% normal interval

 n = 10   |############ 1.859
 n = 20   |######### 1.315
 n = 40   |###### 0.930
 n = 80   |##### 0.657
 n = 160  |### 0.465
 n = 320  |## 0.329
```

Each doubling of `n` multiplies the width by 0.707 and the cost by 2. Going from 10 to 320 replications — 32 times the runs — narrows the interval from 1.859 to 0.329, a factor of 5.65, which is the square root of 32 to two decimals. The geometric consequence is that the last doubling is almost never worth it, and the design conversation should be about which doubling the decision requires.

### 6.3 The bias-variance U-curve for run length

With a fixed budget `B = 480` replication-hours, per-replication overhead `c0 = 1.5` hours, per-replication variance `s^2 = 40` for a one-hour run so that `s^2/m` is the variance of a run of length `m`, and a bias decaying as `2 exp(-0.08 m)` hours, the achievable root-mean-square error is `sqrt(bias^2 + (s^2/m)/n)` with `n = B/(c0 + m)`.

```text
 m (h)   n = int(B/(c0+m))   var of mean = 40/(m n)   bias     rmse
   10           41                 0.0976          0.8987   0.9514
   20           22                 0.0909          0.4038   0.5039
   40           11                 0.0909          0.0815   0.3123
   60            7                 0.0952          0.0165   0.3090
  100            4                 0.1000          0.0007   0.3162
  200            2                 0.1000          0.0000   0.3162
```

The curve falls steeply to about `m = 40` and then is flat, because the variance term is already governed by the floor `s^2/B = 40/480 = 0.0833` and only the integer rounding of `n` moves it. The lesson is that the bias term dominates until the bias falls below the sampling standard deviation of about 0.29, which happens near `m = 40` hours at this decay rate; beyond that, extra run length buys nothing and the budget is better spent on more replications, up to the point where `n` falls below about 5 and the t quantile expands the interval faster than the variance shrinks.

### 6.4 The three shapes of a bad comparison

```text
(a) overlapping intervals, real difference:    A [-----]
                                                  B [-----]      cannot separate

(b) paired on common streams:                  A [-----]
                                              B [---]             difference visible

(c) independent streams, same designs:         A [-----]
                                                     B [-----]    hides the difference

Reading: the point estimates in (b) and (c) are the same. Only the interval on the
difference changes, and only because the paired design removes shared noise.
```

The picture is why "the intervals overlap, so there is no difference" is not an inference. Overlap of two marginal intervals is not a test of a difference; the correct object is an interval on the difference computed from paired or unpaired data with the matching variance.

### 6.5 Where the time goes

```text
one replication, 720 simulated hours, 240 h warm-up
|---- warm-up 240 --------------|---- observation 480 -----------------|
 0%                            33%                                    100%
 of simulated time            of time is discarded

cost split at 4.2 arrivals/h: 3024 arrival events x 2 + 2880 service events
heap operations ~ 6000 per replication, at 1.35 million per second measured
=> about 4.4 milliseconds of calendar work per replication on the authoring machine
=> wall time is dominated by bookkeeping and I/O, not by the heap
```

### 6.6 The correlation geometry behind common random numbers

```text
cloud of replication outcomes for two designs, same axes (response A, response B)

  B                        B                        B
  |      *                 |            *           |   *
  |   *     *              |        *       *       |      *
  | *    *     *           |     *     *      *     |          *
  |________________ A      |________________ A      |________________ A
   rho = +0.85              rho = 0.00               rho = -0.40
   ellipse long along       circle, so pairing       ellipse across the
   the diagonal; width      gains nothing            diagonal; pairing
   of difference is small                            makes the difference
                                                     noisier than independence
```

The three panels are the whole argument for CRN and for its diagnostic. The difference `B - A` is the distance from each point to the 45-degree line; when the cloud is elongated along that line, the distances are small even though the coordinates are widely spread. When the cloud is circular, the distances are as variable as the coordinates. When the cloud is elongated across the line, differencing amplifies.

The practical reading is that the pair correlation is a property of the model pair, not of the experimenter's intention, so it has to be measured on a pilot of about 20 pairs before the full design is committed. A measured correlation of 0.85 with per-replication standard deviations of 0.62 and 0.55 reduces the per-pair variance from 0.6869 to 0.1072, and the run count for a fixed half-width from 1056 to 165.

### 6.7 Warm-up as a transient tail, drawn

```text
E[L] for M/M/3 from an empty start, a = 2.8, c = 3

 15.07 |--------------------------------------- steady-state limit
       |                              .......
       |                     .........
       |             .......
       |      .....
  0.00 |***...
       +----+----+----+----+----+----+----+----> t (h)
       0   60  120  240  360  480  720  960

shaded region under the curve up to w is the bias contribution of the warm-up;
the area between w and T divided by (T - w) is the estimate actually reported.
```

The shaded-area view is why a warm-up chosen from the point value at `w` is insufficient. At 240 hours the instantaneous value is already 96.4% of the limit, but the average over `[0, 480]` is still 9.6% low, and the average over `[240, 480]` is 1.59% low. The reported estimate is an area ratio, so it responds to the whole shape, not to the last point.

### 6.8 Cost surface for the run-length choice

```text
rmse of the reported mean against run length m, fixed 480 replication-hours

 0.95 |*
 0.50 |  *
 0.31 |      *   *
 0.30 |          *
 0.29 |              *   *   *
      +----+----+----+----+----+----+----> m (h)
        10   20   40   60  100  200

values: 0.951, 0.504, 0.312, 0.309, 0.316, 0.316 -- flat from m = 40 onward,
because the variance floor s^2/B = 0.0833 already binds and only integer rounding
of n = 480/(1.5+m) moves the point; left of m = 40 the bias term dominates.
```

**Takeaway:** every design quantity in this sheet has a slope — precision as a square root, warm-up as an exponential decay, comparison power as a correlation — and reading the slope is what stops an expensive round number from being chosen.

## 7. Comparison / Selection Table

### 7.1 Run architecture

| Architecture | Estimator | Independence source | Needs warm-up | Cost driver | Choose when |
|---|---|---|---|---|---|
| independent replications | mean of `n` run means | disjoint streams | yes for steady-state | `n (c0 + T)` | default for steady-state and for comparisons |
| one long run, batch means | mean of `k` batch means | within-run decorrelation | yes, plus batch length | one `T` plus batching | warm-up dominates cost and one path is enough |
| regenerative cycles | mean of cycle means | regeneration points | no warm-up needed | cycles until a count target | the model has a natural regeneration state |
| terminating replications | mean of `n` horizon functionals | disjoint streams | no, and deleting hurts | `n T` | the business question names start and end |
| trace-driven single path | path functional | none, so no interval | depends on the trace | trace acquisition | the trace *is* the population of interest |

The regenerative row is the principled answer to warm-up, and it is used rarely because regeneration points are hard to identify in models with routing and priorities. The terminating row is the one that analysts most often get wrong by importing steady-state habits.

### 7.2 Estimator and interval constructions

| Construction | Assumption it needs | Typical failure | Diagnostic |
|---|---|---|---|
| replication t-interval | independent, approximately normal run means | too few replications for the CLT | normal plot of 30+ run means |
| batch means | batch length exceeds the correlation range | batches too short, interval too narrow | batch-count-vs-half-width curve flattening |
| spectral / overlapping batch means | wide-sense stationary output | non-stationary trend | cumulative-mean plot |
| jackknife on a ratio | smooth functional, exchangeable blocks | blocks too short for the ratio | compare with delete-one-block variants |
| Wilson or Clopper-Pearson on a proportion | binomial counts per run | correlated counts inside a run | over-dispersion check across runs |

### 7.3 Variance reduction

| Technique | Variance effect | Requirement | When it backfires |
|---|---|---|---|
| common random numbers | removes shared noise across scenarios | entity-aligned streams | responses anti-correlated with the shared driver |
| antithetic variates | `(1 + rho_pair)/2` on the pair mean | monotone response in the uniforms | positive pair correlation, as with mixed drivers |
| control variates | multiplies by `1 - corr^2` | control mean known independently | control's mean taken from the same sample |
| stratification | removes between-stratum variation | a stratifying variable with known weights | strata with too few observations each |
| importance sampling | effective for tail probabilities | a good tilting distribution and weights | mean estimation, where it adds weight variance |

### 7.4 Design of the experiment

| Design | Runs for `k` factors at 2 levels | Detects interactions | Cost relative to OFAT | Use when |
|---|---|---|---|---|
| one-factor-at-a-time | `2k` | no, by construction | baseline | screening with a hard run budget |
| full factorial `2^k` | `2^k` per replicate | all orders | `2^(k-1)/k` | up to about 5 factors |
| fractional factorial `2^(k-p)` | `2^(k-p)` | selected aliases only | much lower | 6 or more factors, screening |
| Latin hypercube | `n` for `n` levels | poorly, unless nested | moderate | deterministic-computer response surfaces |
| ranking and selection | sequential, budget-stopped | not applicable | lowest for picking a winner | the question is "which of `k` wins" |

The OFAT row deserves emphasis because it is the default practice: with two factors at two levels, OFAT needs 6 runs while the full factorial needs 4, and only the factorial can reveal that the best level of one factor depends on the other. The interaction estimate is the difference between the two designs' conclusions, and it is invisible to OFAT.

### 7.5 Choosing among modelling paradigms

| Paradigm | Gives | Costs | Cannot easily give |
|---|---|---|---|
| closed-form queueing | exact means for simple structures | restrictive assumptions | heavy tails, priorities with abandonment, finite buffers |
| discrete-event simulation | any structure, any distribution | input modelling and run design | exactness, and gradients for optimisation |
| system dynamics | aggregate feedback and delay | validation against micro behaviour | individual-level service levels |
| agent-based | heterogeneous behaviour and interaction | calibration and compute | analytic bounds |
| spreadsheet Monte Carlo | fast, familiar, auditable | independence across cells | dynamics over time |

### 7.6 Screening then confirmation, with its budget arithmetic

A study with five candidate factors at two levels has 32 treatment combinations in a full factorial and 16 in a resolution-V half-fraction, where the half-fraction aliases main effects with four-factor interactions and two-factor interactions with three-factor interactions. Because four- and three-factor interactions are rare in service operations, the half-fraction is the standard screening choice, and the confirmatory stage runs the full factorial on the three or four factors that survived.

The budget arithmetic makes the strategy worth stating explicitly. With a per-run cost of 31.5 replication-hours and two replicates per cell, the full factorial costs `32 * 2 * 31.5 = 2016` replication-hours. The half-fraction costs 1008, and a confirmatory full factorial on three surviving factors at the same replication costs `8 * 2 * 31.5 = 504`, so the two-stage plan costs 1512 and saves 25% while adding one week of calendar time for the intermediate decision. If no factor survives screening, the saving is 1008 hours and the study ends early, which is a legitimate outcome that a single-stage design cannot produce.

The risk the strategy carries is that a factor's effect is confounded with an interaction that the half-fraction cannot separate, so a factor that matters only through an interaction can be dropped. The mitigation is to keep any factor whose effect estimate exceeds twice the estimated standard error, even if it is not significant, and to keep any factor that the operators name as a suspected interaction partner.

**Takeaway:** the architecture choice sets the independence structure, the estimator choice must match that structure, and no amount of variance reduction repairs a mismatch between the two.

## 8. Failure Modes & Edge Cases

Each entry gives the observable symptom, at least two competing causes, the check that separates them, the correction, and the residual risk after correction. The competing-cause column exists because the same symptom routinely has a modelling cause and an analysis cause, and fixing the wrong one leaves the defect in place.

### F-01 Interval looks precise but coverage is low

**Symptom.** A 95% interval of half-width 0.07 hours, and an independent replication of the whole study places the mean outside it in 4 of 10 repeats.
**Competing cause 1.** The replications are not independent, because the stream assignment reuses the same seed for every run.
**Competing cause 2.** The response is autocorrelated inside each run and the estimator is a within-run mean, so the effective sample size is far below the nominal `n`.
**Discriminating check.** Recompute the interval from batch means of a single run and compare with the replication interval; if the batch interval is 2 to 3 times wider, the within-run correlation is the cause. Then inspect the seed table: distinct seeds with overlap in the first 10 000 draws also inflate correlation.
**Correction.** Use disjoint streams per replication, or a single stream split by process with documented jump-ahead; if the autocorrelation is the cause, switch to replication means or to a batch length where the batch-count curve flattens.
**Residual risk.** With fewer than about 20 batches, the batch-means interval is itself optimistic.

### F-02 Level estimate is low by a stable percentage

**Symptom.** Simulated mean queue length 13.6 against an analytic 15.07, consistently across seeds, and the gap does not shrink as replications increase.
**Competing cause 1.** Insufficient warm-up, leaving a transient from an empty start.
**Competing cause 2.** A finite buffer truncating the queue, which lowers the mean by blocking arrivals rather than by discarding time.
**Discriminating check.** Run the same model with a buffer large enough that blocking is below 0.1%; if the gap closes, the buffer was the cause. If it does not, plot `E[L]` against the warm-up length and look for the plateau.
**Correction.** For the transient, set the warm-up from the window-bias table and accept the discarded-time cost; for the buffer, state the capacity in the model definition so the comparison is against M/M/1/K rather than M/M/1.
**Residual risk.** Over-long warm-up wastes budget and, if the model is non-stationary by design, can hide the peak the study exists to measure.

### F-03 Comparison shows no difference where one exists

**Symptom.** Two staffing designs differ by 0.4 hours of mean wait, with individual intervals of plus or minus 0.25 hours, and the analyst concludes equivalence.
**Competing cause 1.** The comparison used independent streams, discarding the shared noise that the paired design would have removed.
**Competing cause 2.** The difference is real but the response is dominated by a factor not held constant, such as a different warm-up in each scenario.
**Discriminating check.** Compute the per-replication differences across 20 paired runs and inspect their mean and standard deviation; if the paired standard deviation is much smaller than the individual ones, pairing was available and unused. Also verify that the warm-up and horizon match exactly across scenarios.
**Correction.** Re-run as a paired experiment with entity-aligned streams, then test the mean difference with a paired t-interval.
**Residual risk.** Pairing can induce negative correlation when the two designs respond oppositely to the same driver, so check the achieved pair correlation before scaling up.

### F-04 Paired design makes the comparison noisier

**Symptom.** The standard deviation of the difference exceeds the root of the sum of squares of the two individual standard deviations, which is impossible under positive correlation.
**Competing cause 1.** Misalignment: the paired runs consume the stream in different orders, so replication `i` of design A is matched to a different random draw than replication `i` of design B.
**Competing cause 2.** Genuine negative correlation, for example a design whose service policy is faster exactly when arrivals are slow.
**Discriminating check.** Correlate the two run-level responses across the pairs. A correlation near zero with a variance above the independent value points to misalignment; a clearly negative correlation points to the response structure.
**Correction.** Split streams per process (one for arrivals, one for service, one for routing) and index them by entity identifier so that the same draw serves the same entity role in both designs. Document the alignment map in the audit record.
**Residual risk.** If the designs generate different numbers of entities, some draws have no partner and the pairing is partial; report the fraction of paired draws.

### F-05 Batch length chosen by habit

**Symptom.** The batch-means interval changes by a factor of 1.8 when the batch length is doubled, and the interval narrows as batches get shorter.
**Competing cause 1.** Batch length below the correlation range, so batch means remain correlated and the variance estimate is biased low.
**Competing cause 2.** Very few batches, so the t quantile is large and the interval is wide for a legitimate reason.
**Discriminating check.** Plot half-width against batch count for `k` in 10, 20, 40, 80 and look for the plateau; a half-width that keeps falling as `k` rises past the point where `m` is short indicates correlation, not precision.
**Correction.** Increase `m` until the estimate stabilises, and keep `k` at 20 or more; report the chosen `(k, m)` pair with the interval.
**Residual risk.** A model with a long correlation range, such as a heavily loaded queue, needs batch lengths in the hundreds of hours, at which point replications are cheaper.

### F-06 Scenario comparison accidentally reuses one seed set

**Symptom.** Two scenarios produce identical output for the first several hours and then diverge, and the difference interval is implausibly narrow.
**Competing cause 1.** Deliberate CRN, applied without realising that the shared stream creates dependence between the scenarios that the interval construction ignores.
**Competing cause 2.** A configuration bug in which the scenario identifier never reaches the generator setup.
**Discriminating check.** Print the first five uniforms consumed in each scenario; identical sequences with differing scenario labels is a configuration bug, while identical sequences with the intended label is correct CRN that needs a paired interval.
**Correction.** For CRN, compute the interval on the differences, never on the individual means. For the bug, pass the scenario identifier into the generator constructor and assert inequality of the first draws in a unit test.
**Residual risk.** A correct paired analysis still cannot separate scenario effects from a shared input-model error.

### F-07 Time-stepped model drifts with step size

**Symptom.** Mean queue length rises with utilisation faster than the analytic M/M/c curve, and the excess grows as `dt` shrinks.
**Competing cause 1.** Simultaneous arrival and departure in one slot resolved in a fixed order, which changes the effective service rate.
**Competing cause 2.** The Bernoulli arrival approximation used for the step, which replaces `lam` with `1 - exp(-lam dt)` per slot and mis-scales when the step is large.
**Discriminating check.** Halve `dt` and repeat; if the estimate moves monotonically toward the analytic anchor, the discretisation is the cause. Compare the per-slot arrival probability with the intended rate times the step.
**Correction.** Use an event calendar rather than uniform steps, or resolve simultaneous events by a documented priority rule and validate against the analytic case at low load.
**Residual risk.** An event-driven model can still have ties, so the priority rule must be recorded even when the step is zero.

### F-08 Rare-event estimate from a handful of exceedances

**Symptom.** A tail probability of 0.02 is estimated as 0.012 from 250 replications, with an interval computed from the normal approximation to a proportion.
**Competing cause 1.** Too few exceedances for the normal approximation, so the interval is spuriously symmetric and can include impossible values.
**Competing cause 2.** The tail event is defined on correlated observations within a run, so the effective count is smaller than the raw count.
**Discriminating check.** Count exceedances, not replications: 250 runs at `p = 0.02` give an expected 5 events, and the probability of seeing fewer than 5 is 0.439; fewer than 5 observed means the interval is not usable.
**Correction.** Use an exact or Wilson interval, increase replications to at least 368 for an 86% chance of five or more events, or use importance sampling with a validated weight.
**Residual risk.** Exact intervals at very small counts are wide by construction, and a wide interval on the tail may make the decision undecidable, which is a legitimate finding.

### F-09 Utilisation reported above one

**Symptom.** A run reports utilisation 1.04 for a three-server system.
**Competing cause 1.** The numerator integrates busy-server time over a window that overlaps the warm-up, so it is compared with the wrong denominator.
**Competing cause 2.** Server state is double-counted when an entity is handed between servers within the same instant, so busy time exceeds `c` times elapsed time.
**Discriminating check.** Recompute busy time from the event log as the sum of service durations that lie entirely inside the window; if that sum divided by `c` times the window is at most one, the integration bounds are the cause. Also check for overlapping service intervals on the same server identifier.
**Correction.** Clip service intervals at both window boundaries and attribute partial service to both sides; assert the ratio is at most one plus a tolerance in the reconciliation test.
**Residual risk.** A correctly computed ratio can still be biased as a ratio estimator in short windows, so report it with a batch-based interval.

### F-10 Replications too few for the quantile, not the mean

**Symptom.** The mean wait is stable across increasing `n`, but the 95th percentile of wait jumps between runs.
**Competing cause 1.** The percentile is an order statistic, so its variance depends on the number of observations above the threshold rather than on the number of runs.
**Competing cause 2.** The distribution has a heavy tail, so the sample 95th percentile converges slowly and is itself heavy-tailed.
**Discriminating check.** Count observations above the estimated percentile in each run; fewer than 5 per run means the percentile is being interpolated from very little information. Compare the estimate at `n` and `2n` runs.
**Correction.** Pool observations across runs only when runs are identically distributed, or raise replications until each run contributes at least 20 exceedances; for heavy tails, report a smoothed tail estimate or a higher quantile with its interval.
**Residual risk.** Pooling across runs with different warm-up lengths mixes distributions and is not a valid substitute.

### F-11 Validation performed on the same period used for calibration

**Symptom.** The model reproduces the measured month to within 2% on every metric, then over-predicts queue length by 35% when a new month is used.
**Competing cause 1.** Over-fitting of the input distributions to the calibration period, especially when arrival rates were fitted hour by hour.
**Competing cause 2.** A structural omission, such as abandonment, that happened not to bind in the calibration period.
**Discriminating check.** Hold out a period, calibrate on the rest, and compare the held-out residuals. If the residual distribution changes shape rather than just level, the structure is wrong; if it shifts level only, the parameters are over-fitted.
**Correction.** Reserve at least 25% of the historical period for validation before any parameter is fitted, and report the held-out error alongside the fitted error.
**Residual risk.** A single held-out period cannot reveal a structural error that is inactive in that period either.

### F-12 Entities in progress at the horizon are dropped

**Symptom.** Throughput is 3% low and mean sojourn is 5% low, with both shortfalls shrinking as the horizon grows.
**Competing cause 1.** The horizon truncates entities mid-service and they are excluded from the sojourn mean, which removes exactly the long-sojourn entities.
**Competing cause 2.** Entities that arrived before the horizon but departed after it are excluded from the arrival count, so `lam_hat` is slightly low and Little's law still appears to hold.
**Discriminating check.** Reconcile arrivals, departures and the count still in system at the horizon. If the three do not add up, the truncation is the cause. Then compute the mean sojourn including a censored contribution for the in-progress entities and see whether the gap closes.
**Correction.** Either drain the system after the horizon for terminating studies, or treat the endpoint as a censoring boundary and report the censored estimator with its censoring fraction.
**Residual risk.** Draining lengthens the run and shifts the population of interest toward later arrivals.

### F-13 Seed reuse across factor levels

**Symptom.** In a factorial design, the interaction effect is much smaller than expected and the residual variance is suspiciously small.
**Competing cause 1.** The same seed was used for every treatment combination, so the factor effects are estimated against a common random draw and the error term loses its independence.
**Competing cause 2.** The design was properly paired but analysed as if unpaired, so the residual mean square includes variance that the pairing removed.
**Discriminating check.** Inspect the seed column of the run table; a constant seed across cells is cause 1. Compare the pooled error from a paired analysis with the unpaired one; a large drop indicates cause 2.
**Correction.** Use a separate stream per cell for an unpaired factorial, or declare CRN deliberately and analyse the design as a randomised complete block with the replication index as the block.
**Residual risk.** Blocking by replication is correct only if the within-block correlation is positive, which should be checked on a pilot.

### F-14 The stopping rule is silence

**Symptom.** A study reports an interval of plus or minus 0.6 hours against a target of 0.15, with no statement of what the decision-maker should do with it.
**Competing cause 1.** No pre-registered stopping rule, so the run ended when the compute window closed.
**Competing cause 2.** A stopping rule existed but was applied to the wrong quantity, such as the number of replications rather than the achieved half-width.
**Discriminating check.** Compare the reported half-width with the target in the design record; a mismatch with no accompanying note means the rule was absent or ignored.
**Correction.** Write the rule as "stop at `n` when `h <= d`, or at the budget cap, and in the latter case report `h` and the decision's insensitivity to it". State in advance which decision the wide interval leaves open.
**Residual risk.** A premature stop on a favourable interval is a form of optional stopping; the rule must be fixed before the runs.

### F-15 Input-model parameter fitted on aggregated data

**Symptom.** The simulation reproduces the daily average arrival count, but the intraday queue peak is 30% too low and the idle fraction is too high.
**Competing cause 1.** Arrivals were fitted as a single Poisson rate per day, which preserves the daily total and destroys the hour-to-hour variation that creates peaks.
**Competing cause 2.** Service times were fitted to a normal law truncated at zero, which compresses the upper tail and shortens the long services that cause the worst queues.
**Discriminating check.** Compare the variance-to-mean ratio of arrivals per hour against the fitted model's prediction; a ratio well above one with a single-rate Poisson fit is decisive. For service, compare the empirical 90th percentile with the fitted law's 90th percentile at the same mean.
**Correction.** Fit a time-varying arrival rate, typically piecewise-constant per hour with a smoothing rule, and refit service to a distribution that reproduces the empirical tail, then re-check the 90th and 95th percentiles rather than the mean.
**Residual risk.** An hourly rate fitted to a small sample over-fits and produces spurious peaks, so smoothing across adjacent hours is required and must be reported.

### F-16 Non-stationarity inside the horizon

**Symptom.** The queue builds steadily through the run and the batch means show a trend instead of a plateau, so the interval depends on where the run stops.
**Competing cause 1.** The offered load exceeds capacity, so the queue grows without bound and no steady state exists to estimate.
**Competing cause 2.** A genuine trend in the input, such as a seasonal ramp or a workforce that is reduced partway through the horizon, which makes the process non-stationary by design.
**Discriminating check.** Plot the cumulative mean against time: a monotone drift with no flattening indicates `rho >= 1`, while a bounded trend that reverses indicates seasonality. Compute `lam_hat / (c mu)` over the last third of the run and compare with the first third.
**Correction.** If the load exceeds capacity, stop and report it as the finding, since no run length produces a steady-state estimate. If the trend is real, redefine the estimand as a time-bounded average over a declared window and drop the warm-up apparatus entirely.
**Residual risk.** A stable estimate over one horizon says nothing about a horizon that spans a seasonal boundary.

**Takeaway:** each failure here has a modelling twin and an analysis twin, and the discriminating check is always a quantity that differs between them, not an opinion about which is more likely.

## 9. Complexity & Cost Analysis

### 9.1 Event counts and per-replication cost

For a queueing model with arrival rate `lam` and aggregate service capacity `c mu`, a run of length `T` generates approximately `lam T` arrival events, `min(lam, c mu) T` service-completion events, and a comparable number of scheduling operations. With `lam = 4.2` per hour, `c mu = 4.5` per hour and `T = 480` hours, that is 2016 arrivals and about 2016 completions, so roughly 4032 events, each requiring a calendar insertion and a removal.

```text
FORMULA [SIM-07]: cost per replication ~ c0 + k_e * E  +  k_c * E * log(E_inflight)
WHERE: c0 is fixed setup [hours of wall time or currency]; k_e is per-event cost;
       E is the event count [count]; E_inflight is the calendar size, bounded by the
       number of pending events; the log term is the heap operation.
CONSTRAINT: the model is a lower bound when events trigger cascades, such as a
       completion that immediately starts a queued entity and schedules its finish.
NUMERICAL EXAMPLE: 4032 events at a measured 1.35 million heap operations per second
       on the authoring machine gives about 3.0 milliseconds of calendar work; with
       per-event bookkeeping at a comparable rate, a 480-hour replication costs
       tens of milliseconds of compute, so the setup and output handling dominate.
INTERPRETATION: for queueing models of this size, compute is cheap and design is
       expensive; the bottleneck is the analyst's choice of n, w and T.
```

### 9.2 Cost of warm-up

The fraction of simulated time that survives the warm-up is `(T - w)/T`. With `w = 240` and `T = 480` the useful fraction is 0.5, so every useful hour costs two simulated hours. Doubling the horizon to 960 hours at the same warm-up raises the useful fraction to 0.75, which reduces the cost per useful hour by a third while also reducing the residual window bias from 1.59% to 0.64%. This is the one place where "run longer" is unambiguously correct, provided the model is stationary over the extended horizon.

```text
w (h)   T (h)   useful fraction   residual bias   cost per useful hour (relative)
240      480          0.500            1.59%                2.00
240      720          0.667            0.93%                1.50
240      960          0.750            0.64%                1.33
360      960          0.625            0.31%                1.60
120      960          0.875            1.49%                1.14
```

### 9.3 Cost of precision

The half-width scales as `s/sqrt(n)`, so the cost to reach a half-width `d` with per-replication cost `c_rep = c0 + m` is

```text
FORMULA [SIM-08]: C(d) = (t*s/d)^2 * (c0 + m),  so  d * sqrt(C) = t*s*sqrt(c0+m)
WHERE: C is total cost in replication-hours; s is the replication standard deviation
       for run length m; c0 is fixed per-replication overhead [hours]; m is the
       run length [hours].
CONSTRAINT: s itself depends on m, usually falling like 1/sqrt(m) for a mean; the
       product s*sqrt(c0+m) then behaves like sqrt((c0+m)/m), which is decreasing
       and convex in m.
NUMERICAL EXAMPLE: c0 = 1.5 h; m = 30 gives sqrt(31.5/30) = 1.0247; m = 120 gives
       sqrt(121.5/120) = 1.0062, a 1.8% improvement in cost per unit precision for a
       fourfold longer run, before any bias is considered.
INTERPRETATION: once m is much larger than the overhead, lengthening runs buys almost
       nothing in precision per unit cost, so the run length should be set by bias
       and stationarity rather than by a cost argument.
```

### 9.4 Budget allocation across scenarios

When the objective is a difference between two systems with per-replication standard deviations `s1` and `s2`, allocating `N` total replications to minimise the variance of the difference gives `n1 / n2 = s1 / s2`.

```text
FORMULA [SIM-09]: n1* = N s1/(s1+s2),  n2* = N s2/(s1+s2),  Var_min = (s1+s2)^2/N
WHERE: s1, s2 are per-replication standard deviations [response unit]; N is the
       total replication budget [count]; equal allocation gives (s1^2+s2^2)/N.
CONSTRAINT: both systems must be simulated to the same run architecture, and the
       allocation formula assumes independent replications, not paired ones.
NUMERICAL EXAMPLE: s1 = 2.0, s2 = 1.0, N = 3000 gives n1 = 2000, n2 = 1000 and a
       standard error of 0.05477, against 0.05774 for equal allocation, a 5.13%
       reduction in standard error for the same total budget.
ILLUSTRATIVE CONTRAST: s1 = 0.62, s2 = 0.55, N = 2400 gives n1 = 1271.8 and
       n2 = 1128.2 for a standard error of 0.023883 against 0.023925 equal, a gain
       of 0.18%, which is not worth the bookkeeping.
INTERPRETATION: optimal allocation matters only when the two systems' variabilities
       differ by a factor of about two or more.
```

### 9.5 Parallelisation and its limits

Replications are embarrassingly parallel: `n` runs on `W` workers take `n/W` replication-times plus straggler effects, with no communication. Batch means are not parallel in the same way, because the batches belong to one path. The wall-clock consequence for the worked example is 72 replications at 31.5 hours each, which is 2268 replication-hours, or 141.8 hours on 16 workers, about 5.9 days. That number is what turns the design conversation into a budget conversation: 72 replications at 5% precision is six days, and 21 replications at 10% precision is 1.7 days.

### 9.6 Cost per decision, not cost per run

| Design | Replications | Hours per rep | Total rep-hours | Half-width | Wall hours at 16 workers |
|---|---|---|---|---|---|
| screening | 9 | 31.5 | 284 | 0.477 h | 17.7 |
| decision-grade | 21 | 31.5 | 662 | 0.282 h | 41.4 |
| publication-grade | 72 | 31.5 | 2268 | 0.146 h | 141.8 |
| tail-focused | 830 | 31.5 | 26145 | 0.030 on a proportion | 1634 |

The last row is the one that surprises: estimating a proportion to plus or minus 0.03 at `p = 0.264` needs 830 replications, more than eleven times the count needed for a 5% interval on the mean. Tail and proportion questions are systematically more expensive than mean questions, and the design should be chosen to answer the decision rather than to maximise the number of metrics reported.

### 9.7 Memory and state

Memory is governed by the calendar and by the entity population in the system, not by the horizon. An M/M/3 at `rho = 0.933` holds about 15 entities on average, so a run of 2016 arrivals needs a calendar of order tens of entries and a few megabytes of event log. A trace-driven model at a million events per simulated hour changes the picture entirely, and then log volume, not compute, becomes the constraint; the design response is to aggregate during the run rather than to write every event.

### 9.8 Parallel streams and the cost of reproducibility

Running 72 replications on 16 workers requires 72 disjoint streams, and the cheapest correct way to obtain them is a counter-based generator whose stream identifier is the replication index. The older approach — one seed per worker with independent generators — is safe only if the generators have disjoint, provably non-overlapping sequences; two generators seeded 1 apart typically produce sequences that are correlated for their first few thousand draws, which is exactly the region a short run occupies.

The cost of the counter-based approach is one extra field in the run table and about one line of code per replication; the cost of getting it wrong is an interval that is narrower than reported and a re-run that cannot reproduce the original. For the worked example the stream identifier takes 72 distinct values, the run table has 72 rows, and the reconciliation for reproducibility is that re-running replication 37 with its recorded identifier returns the same mean to the last printed digit. That equality test is the operational definition of "the seed plan is complete", and it costs one extra replication out of the budget.

A related cost is jump-ahead. Leapfrogging a generator forward by `k` draws costs `O(log k)` multiplications with the standard fast-exponentiation trick, so splitting a stream into 72 blocks of a million draws is cheap; doing it by drawing and discarding a million values per block is not, and it is the common implementation that makes parallel runs appear to be ten times more expensive than they are.

**Takeaway:** compute is rarely the binding constraint; precision, bias control and the number of decisions to be made are, and each has a cost curve that can be written down before any run starts.

## 10. Key Theorems / Results Reference

Each result is stated with its conditions, because a result quoted without conditions is a slogan. Identifiers are stable and are cited from Sections 4, 7 and 8.

### T-01 Little's law

For a system observed over `[0, T]` in which the entity count is finite and entities are conserved, `L = lam W` holds pathwise for the time-average count and the mean sojourn of the entities counted. Conditions: finite window, conservation, consistent entity sets. Used in SIM-01 and as the reconciliation check in F-12. The result is an identity on each path, so it does not require stationarity and does not require independence; it does require that the averages be taken over the same set.

### T-02 Erlang C for M/M/c

With Poisson arrivals at `lam`, exponential service at `mu`, `c` servers and `rho = lam/(c mu) < 1`, the probability of waiting is `C(c, a) = (a^c/c!)(c/(c-a)) / (sum_{k<c} a^k/k! + (a^c/c!)(c/(c-a)))` with `a = lam/mu`, and `Wq = C/(c mu - lam)`. Conditions: exponential service, infinite buffer, FCFS, no abandonment. Used as the analytic anchor in Section 5, where `c = 3`, `a = 2.8` gives `C = 0.876677` and `Wq = 2.92226` hours.

### T-03 M/M/1/K blocking

For finite capacity `K`, `p_K = (1-rho) rho^K / (1 - rho^{K+1})` for `rho != 1`, and the effective arrival rate is `lam (1 - p_K)`. Conditions: same as T-02 plus a hard capacity. Used in F-02 and Section 4.4 Limit 4, where `K = 5`, `rho = 0.8` gives `p_K = 0.0888` and `Wq = 1.5631` against 4.0 for the unbounded system.

### T-04 Transient of the M/M/1 mean

`E[N(t)] = rho (1 - exp(-(mu - lam) t))` for an empty start. Conditions: `lam < mu`, single server, exponential service, empty initial state. Used to justify the warm-up table: the 2% point is `-ln(0.02)/(mu - lam) = 19.56` hours at `mu - lam = 0.2`, which is the rare case where the warm-up is short, because `rho` is only 0.8.

### T-05 Central limit theorem for replication means

If the replication responses `X_i` are independent and identically distributed with finite variance, then `(X_bar - mu)/(s/sqrt(n))` converges in distribution to a standard normal, and the t-interval has asymptotically nominal coverage. Conditions: independence, identical distribution, finite variance, and enough `n` for the skew to wash out. Used in SIM-02 and SIM-03. The condition that fails most often is identical distribution, which is broken when scenarios differ in warm-up or horizon.

### T-06 Batch-means consistency

For a stationary phi-mixing output, the batch-means variance estimator is consistent for the variance of the run mean provided the batch size grows with the run length and the batch count grows without bound. Conditions: stationarity after warm-up, mixing, batch count at least 20 in practice. Used in SIM-06 and F-05, where the AR(1) example shows the naive standard error understating the true one by 41%.

### T-07 Variance of a paired difference

`Var(X_B - X_A) = Var(X_A) + Var(X_B) - 2 Cov(X_A, X_B)`, and the covariance is what common random numbers creates. Conditions: the pair is measured on the same replication index and the streams are aligned. Used in SIM-04 and F-04, where the misalignment case sets the covariance to whatever the accidental pairing produces.

### T-08 Control-variate variance identity

For the optimal linear coefficient, `Var(X + c*(Y - E[Y])) = Var(X)(1 - rho_XY^2)`. Conditions: `E[Y]` known independently, `c*` estimated from a sample large enough that its own noise is negligible, and joint normality for the exactness of the linear form. Used in SIM-05, where the measured variance ratio 3.0247 matched `1/(1 - 0.6694)` to four decimals.

### T-09 Antithetic pair-average variance

For a pair `(X(U), X(1-U))` the average has variance `(sigma^2/2)(1 + rho_pair)`, so the technique reduces variance only when the pair correlation is negative. Conditions: a monotone response in the driving uniforms, and a pair correlation measured rather than assumed. Used in Variant B of Section 4.3 and in the comparison table.

### T-10 Factorial effect estimates and their standard errors

In a `2^k` design with `r` replicates per cell, each main effect is a contrast of cell means with `1/(2^{k-1} r)` weights per cell, so `Var(effect_hat) = sigma^2/(2^{k-1} r)`. Conditions: orthogonal design, homoscedastic error, independent runs across cells. Used in Section 5's design check, where four cells with two replicates give `sigma^2_hat = 0.1075`, a standard error of 0.1159 per effect and t statistics of 23.72, 25.02 and 7.76 for the two main effects and the interaction against a critical value of 2.776 on 4 degrees of freedom.

### T-11 Bonferroni bound for multiple comparisons

For `m` simultaneous interval statements each at level `1 - alpha/m`, the family-wise coverage is at least `1 - alpha`. Conditions: no assumption about dependence, which is why the bound is conservative. Used in the ranking discussion: with `k = 5` systems compared each against the incumbent there are 4 comparisons, so the two-sided per-comparison level is `0.05/4 = 0.0125` and the critical value is `z = 2.4977`; an all-pairs comparison of 5 systems has 10 comparisons, a per-comparison level of 0.005 and a critical value of 2.8070, which widens every interval by 43% relative to the naive 1.96.

### T-12 Indifference-zone sample size for selection

A two-stage indifference-zone procedure that guarantees probability of correct selection at least `1 - alpha` whenever the best pair differs by at least `delta` needs a first-stage count of order `n = 2 (z_{1-alpha/(k-1)} sigma / delta)^2`. Conditions: normal cell means, common variance, independent replications, and a declared indifference zone. Used with `k = 5`, `delta = 0.05`, `sigma = 0.3`, giving `z = 2.4977` and `n = 450` first-stage replications, which is the correct order of magnitude and the reason selection studies are expensive.

### T-13 Renewal-reward identity

For a regenerative process with regeneration cycles, the long-run average reward equals the expected reward per cycle divided by the expected cycle length, and cycle-level observations are independent and identically distributed. Conditions: regeneration occurs with probability one in finite expected time. Used to justify the regenerative row in Section 7.1 as the warm-up-free alternative, at the cost of needing cycles rather than hours as the unit of replication.

### T-14 Delta-method bias of a ratio

For `R = B/S` with `B` and `S` sample means, `E[R] ~= E[B]/E[S] - Cov(B,S)/(n E[S]^2) + Var(S) E[B]/(n E[S]^3)`. Conditions: `S` bounded away from zero with high probability, and `n` large enough for the expansion. Used in Limit 2 of Section 4.4 and in F-09, where utilisation above one is the visible symptom of a small denominator.

### T-15 Coverage of the Wald interval for a proportion

The Wald interval's actual coverage can fall well below nominal for small counts or extreme proportions, while the Wilson score interval and the Clopper-Pearson exact interval maintain coverage at least nominal. Conditions: independent Bernoulli trials. Used in F-08: for 14 successes in 40 trials, Wald gives `[0.2022, 0.4978]`, Wilson `[0.2213, 0.5049]` and Clopper-Pearson `[0.2063, 0.5168]`, a spread of 0.019 in the upper limit for the same data.

### T-16 Inspection paradox in arrival sampling

When entities arrive to a system whose server is already busy, the residual service time they must wait for is not the mean service time but `E[S^2]/(2 E[S])`, which exceeds `E[S]/2` whenever service times vary. For exponential service at `mu`, `E[S^2] = 2/mu^2`, so the residual is `1/mu`, exactly the full mean. Conditions: arrivals independent of the service process, work-conserving server, no preemption.

This is why the mean wait in an M/M/1 at `rho = 0.8` is 4.0 hours and not the `0.5 * 0.8 * 1.0 = 0.4` hours that a naive "average residual" argument suggests. A simulation that models service times as deterministic with mean `1/mu` produces a much smaller wait at the same utilisation, and that difference is a property of the input distribution, not a bug.

### T-17 Variance of a time-average versus a customer-average

For a stationary process, the variance of the time-average over a window of length `T` scales as `2 sigma^2 tau / T` where `tau` is the integrated autocorrelation time, while the variance of a mean over `M` independent customer observations scales as `sigma^2/M`. Conditions: stationarity after warm-up and a finite integrated autocorrelation time.

The consequence is that time-averages and customer-averages converge at different rates for the same simulated time, so a study that reports both must size replications for the slower one. In a queue at high utilisation, `tau` grows roughly with the mean queue length, which is the mathematical statement of the practical observation that congested systems need much longer runs.

### T-18 Optimal allocation across two independent systems

For a difference of two independent means with total replication budget `N`, the variance-minimising allocation is proportional to the per-replication standard deviations, and the minimum standard error is `(s1 + s2)/sqrt(N)`. Conditions: independence between systems, common per-replication cost, and a linear objective in the two counts.

Comparing with equal allocation, `sqrt((s1^2 + s2^2)/N)`, shows the gain is a function of the ratio `s1/s2` alone; at a ratio of 2 the gain in standard error is 5.1%, and at a ratio of 1.13 the gain is 0.18%, which is below the noise in the pilot estimate of the standard deviations themselves.

**Takeaway:** the reference table is the sheet's contract with the literature — each result arrives with its conditions, and any use that drops a condition is a different theorem.

## 11. Interview / Exam Questions

**Q1. State the difference between a run and a replication, and say why the distinction changes the interval.**

**Answer:** A run is one simulated path from time zero to the horizon; a replication is a run whose random stream is disjoint from every other replication's. The interval formula assumes independent and identically distributed replication responses, so a set of runs that share a stream is a set of correlated observations and yields an interval that is too narrow.
**Reasoning:** The t-interval's standard error divides by the square root of the number of independent units. If the units are correlated, the effective count is smaller than the nominal count, and the divisor overstates the information.
**Follow-up:** Ask how the candidate would prove independence, and accept either a seed table with documented jump-ahead or an empirical autocorrelation check across run means.
**Red flag:** "Each run uses a different seed" offered as proof, with no statement of how the seed is derived or whether streams can overlap.

**Q2. Thirty replications give a mean wait of 2.87 hours with a standard deviation of 0.62 hours. Compute the 95% interval.**

**Answer:** With `t` on 29 degrees of freedom equal to 2.0452, the standard error is `0.62/sqrt(30) = 0.1132` and the half-width is `2.0452 * 0.1132 = 0.2315`, so the interval is `[2.638, 3.102]` hours, a relative half-width of 8.07%.
**Reasoning:** The replication count is below 30 only marginally, so the t quantile rather than the normal quantile is required; using 1.96 would understate the half-width by 4.2% at this count.
**Follow-up:** Ask what the half-width would be with 120 replications and the same standard deviation, expecting `t_{119} = 1.9801` times 0.62 over 10.9545, which is 0.1120 hours.
**Red flag:** Dividing 0.62 by 30 instead of by the square root of 30, which gives 0.0207 and an interval eight times too narrow.

**Q3. You need a 5% relative half-width on a mean wait of 2.92 hours with a pilot standard deviation of 0.62 hours. How many replications, and why is the answer not the normal-approximation count?**

**Answer:** The target is `d = 0.1461` hours. The normal approximation gives 70, but the t quantile at 69 degrees of freedom is 1.9949 rather than 1.96, so 70 replications achieve only 1.9949 * 0.62 / 8.3666 = 0.1478 hours. Iterating gives 72, where `t = 1.9939` and the achieved half-width is 0.1457, just inside the target.
**Reasoning:** The count appears on both sides of the inequality through the degrees of freedom, so the equation is a fixed point, and the correction is small but in one direction.
**Follow-up:** Ask what happens if the pilot under-estimated the standard deviation by 13%, expecting the requirement to rise to about 89.
**Red flag:** Reporting 70 and then reporting the achieved half-width as the target rather than as the number actually obtained.

**Q4. A study reports a half-width of 0.07 hours, but a repeat study places the mean outside that interval four times in ten. Diagnose.**

**Answer:** The interval is too narrow, which means the estimator's variance is under-estimated. The two candidates are non-independent replications, from a shared or overlapping stream, and within-run autocorrelation with a within-run mean used as the unit of observation. The discriminating measurement is a batch-means interval from a single long run: if it is two to three times wider, autocorrelation is the cause.
**Reasoning:** Under-estimation of variance is multiplicative and shows up as under-coverage, whereas a biased point estimate shows up as a shifted centre; the symptom locates the defect in the second moment.
**Follow-up:** Ask how to rule out the third possibility, an implementation bug that discards the warm-up differently in each replication.
**Red flag:** Concluding that the original study's mean was wrong, when the coverage failure speaks about the width rather than the centre.

**Q5. For an AR(1) output with lag-one correlation 0.5, batched into blocks of 10, what inflation should the batch-means standard error show relative to the naive one, and what does a pilot showing only 1.2 tell you?**

**Answer:** The asymptotic inflation in variance is `(1+phi)/(1-phi) = 3.0`, so the standard error should be about 1.73 times the naive one. A pilot showing 1.2 suggests either that the batch length is insufficient for the correlation to have died out, or that the marginal variance is being estimated from too few batches.
**Reasoning:** The inflation factor is a property of the autocorrelation structure, not of the sample size; the square root appears because the standard error is on the scale of the mean.
**Follow-up:** Ask what the batch length should be if the observed autocorrelation at lag 10 were still 0.2, expecting a length of at least 40 to 50.
**Red flag:** Concluding that the process is nearly independent from a single pilot, without checking the autocorrelation at the candidate batch lag directly.

**Q6. Two designs are compared and the paired standard deviation is larger than the unpaired sum. What happened?**

**Answer:** Either the streams are misaligned, so that replication `i` of one design is matched to a different random draw than replication `i` of the other, or the two designs genuinely respond in opposite directions to the same driver, producing negative correlation. The check is the observed pair correlation across replications: near zero with excess variance points to misalignment, clearly negative points to structure.
**Reasoning:** The paired variance is `s1^2 + s2^2 - 2 rho s1 s2`, which exceeds the independent value whenever `rho < 0`, and misalignment is equivalent to an arbitrary, often negative, induced correlation.
**Follow-up:** Ask how to align streams for a model that generates different numbers of entities per hour, expecting stream splitting by process and indexing by entity role.
**Red flag:** Abandoning the paired design without measuring the correlation, which discards a sixfold efficiency gain if the cause was a fixable alignment bug.

**Q7. Why is one-factor-at-a-time experimentation a poor default for a simulation study with four factors?**

**Answer:** OFAT needs eight runs at two levels for four factors and cannot estimate any interaction, while a full factorial needs sixteen runs and estimates all of them. If the best level of one factor depends on another, OFAT's recommendation is wrong, and the study cannot detect that it is wrong.
**Reasoning:** The interaction contrast is a difference of differences that OFAT never forms, because it changes one factor while holding the others at a single level.
**Follow-up:** Ask for the run count of a `2^(4-1)` fractional factorial and what it aliases, expecting eight runs with main effects aliased with three-factor interactions.
**Red flag:** Claiming OFAT is cheaper without computing the run counts, or claiming interactions are usually negligible without a screening argument.

**Q8. A `2^2` design with two replicates per cell gives cell means 4.10, 2.25, 7.90 and 4.25 for cells (low,low), (high,low), (low,high) and (high,high). Compute the three effects and the standard error of an effect.**

**Answer:** The factor-A effect is `(2.25 + 4.25)/2 - (4.10 + 7.90)/2 = 3.25 - 6.00 = -2.75`. The factor-B effect is `(7.90 + 4.25)/2 - (4.10 + 2.25)/2 = 6.075 - 3.175 = 2.90`. The interaction is `(4.25 - 2.25)/2 - (7.90 - 4.10)/2 = 1.00 - 1.90 = -0.90`.
**Reasoning:** Each effect is a contrast with weights `+1/(2^{k-1} r)` per cell, and with `k = 2`, `r = 2` that is `1/4`; the interaction is antisymmetric between the two diagonals, which is why it is a difference of differences of differences.
**Follow-up:** Ask for the standard error, expecting the pooled mean square 0.1075 over four error degrees of freedom, giving `sqrt(0.1075/8) = 0.1159` and t statistics of 23.7, 25.0 and 7.8 against a critical 2.776.
**Red flag:** Averaging the cell means without the contrast weights, which would return a grand mean rather than an effect.

**Q9. An M/M/1 with an arrival rate of 0.8 per hour and a service rate of 1.0 per hour is simulated with a five-slot waiting room. What wait should the model show, and what does the infinite-buffer formula predict?**

**Answer:** With capacity 5, `p_5 = 0.0888`, the effective arrival rate is 0.7289, the mean number in system is 1.8683 and the mean sojourn is 2.5631 hours, so the mean queue wait is 1.5631 hours. The infinite-buffer M/M/1 gives 4.0 hours.
**Reasoning:** Blocking removes the arrivals that would have waited longest, so it lowers the mean wait by truncating the upper tail; the effective rate is the arrival rate times one minus the blocking probability.
**Follow-up:** Ask how the blocking probability would be estimated from a simulation, expecting the fraction of arrivals rejected at the buffer limit, with a reconciliation count for rejected entities.
**Red flag:** Reporting 4.0 hours for a model with a five-slot room, which is a comparison against the wrong analytic anchor.

**Q10. A simulation of a three-desk line gives a mean wait of 1.56 hours while the Erlang C formula gives 2.92. List the first three checks.**

**Answer:** First, reconcile arrivals, departures and the count in system at the horizon to prove no entities were lost or duplicated. Second, check the horizon: at 60 hours the transient expectation of `L` is 11.62 against a limit of 15.07, so a short run or a missing warm-up depresses the wait. Third, check whether the model has a capacity limit, since a five-slot buffer at this load lowers the wait to 1.56 hours exactly.
**Reasoning:** The observed value is not a small deviation; it is consistent with a specific alternative model, which is why the check list should generate an alternative to test rather than just a list of suspicions.
**Follow-up:** Ask what measurement distinguishes the warm-up explanation from the buffer explanation, expecting a run with a buffer large enough for blocking below 0.1%.
**Red flag:** Starting with a re-derivation of Erlang C, which is already verified, instead of testing the implementation.

**Q11. You need the probability of a wait exceeding four hours, believed to be about 0.26, with a half-width of 0.03. How many replications, and does a tail question cost more than a mean question?**

**Answer:** With `p = 0.264`, `n = 1.96^2 p(1-p)/0.03^2 = 830`, against 72 for a 5% interval on the mean. Tail and proportion questions are more expensive because their variance depends on the indicator's Bernoulli variance, which is maximised near one half and does not shrink with run length.
**Reasoning:** A longer run gives more observations, so it does help a proportion, but the run-level replication still has to deliver enough independent indicator counts; batching inside a run is the cheaper route when runs are long.
**Follow-up:** Ask what changes if the target is 0.02 rather than 0.03, expecting the count to rise by a factor of 2.25 to 1866, since 1.96^2 * 0.26405 * 0.73595 / 0.0004 = 1866.3.
**Red flag:** Using the mean's replication count for the proportion, or computing the interval with the normal approximation when the count of exceedances is in single figures.

**Q12. Five system designs are compared against the best. How does the significance level change, and what replication count does an indifference-zone rule suggest?**

**Answer:** With four comparisons and a family-wise level of 0.05, Bonferroni gives a two-sided per-comparison level of 0.0125, so the critical value becomes 2.4977 rather than 1.96, a 27.4% widening. An indifference-zone rule with `delta = 0.05`, `sigma = 0.3` and the same `z = 2.4977` gives a first-stage count of `2 (2.4977 * 0.3 / 0.05)^2 = 449.2`, rounded up to 450.
**Reasoning:** The Bonferroni correction is distribution-free and therefore conservative; the indifference-zone rule converts the correctness requirement into a sample size, which is why it produces a much larger number than a pairwise test would.
**Follow-up:** Ask what the indifference zone means operationally, expecting that differences smaller than `delta` may be resolved arbitrarily without violating the guarantee.
**Red flag:** Running pairwise t-tests at 0.05 and reporting the smallest p-value as if it were a family-wise conclusion.

**Q13. A factorial study used the same seed for every treatment combination and reports a very small residual variance. What is wrong and how would you analyse it?**

**Answer:** Using one seed across cells makes the cells positively correlated, so the residual mean square understates the error and every effect looks more significant than it is. The fix is either to give each cell its own stream and analyse as an unpaired design, or to keep the common seed deliberately and analyse the design as a randomised complete block with the replication index as the block, which removes the seed effect from the error.
**Reasoning:** The two analyses answer different questions about the error term: one estimates variation across independent runs, the other estimates variation after removing a systematic common component.
**Follow-up:** Ask how the seed choice interacts with the number of replicates, expecting that blocking requires several replications per cell to estimate the block effect.
**Red flag:** Reporting the design as paired and then computing the interval from the cell means as if independent.

**Q14. Explain the difference between verification and validation with one check for each, and say which failure is worse for a decision.**

**Answer:** Verification asks whether the program implements the intended model; one check is reproducing the hand-computed seven-entity trace exactly, including the departure of entity 4 at hour 14. Validation asks whether the model is adequate for the decision; one check is reproducing a held-out month's measured queue length within a stated tolerance. An unvalidated but verified model is worse for a decision, because it answers the wrong question exactly and its precision invites action.
**Reasoning:** Verification defects usually announce themselves as reconciliation failures, while validation defects are silent and survive every internal consistency check.
**Follow-up:** Ask what to do when the held-out period disagrees by 10% and the decision margin is 12 half-widths, expecting a sensitivity statement rather than a larger simulation budget.
**Red flag:** Treating a match to history as proof of validity, when compensating errors can produce a match over the fitted range.

**Q15. A ratio estimator reports utilisation 1.04. Give two mechanisms and one discriminating check.**

**Answer:** The numerator's busy-server time may be integrated over a window that is not aligned with the denominator's elapsed time, or service intervals may overlap on the same server identifier through a hand-off in the same instant, double-counting. The check is to recompute busy time by summing service durations that lie wholly inside the window and dividing by `c` times the window; if that is at most one, the integration bounds are at fault.
**Reasoning:** A proportion above one is impossible for the true quantity, so the defect must be in the denominator's construction or in the numerator's double counting, and both are arithmetic rather than stochastic.
**Follow-up:** Ask for the bias direction of a correctly computed ratio in a short window, expecting a small negative bias from the delta-method term with a variable denominator.
**Red flag:** Clipping the reported value at one and moving on, which conceals a defect that also affects the mean number in system.

## 12. Quick Reference Card

| Field | Rule to apply |
|---|---|
| Estimand first | Write the quantity, its denominator and its window before choosing `n`; the denominator decides comparability |
| Warm-up | Set it where the window bias over the remaining horizon is under tolerance, measured not assumed; 240 h at 240-of-480 discards half the time |
| Replication count | `n = (t s / d)^2` solved by fixed point; 72 for a 5% half-width at `s = 0.62` and a mean of 2.92 |
| Interval construction | Replication t-interval for independent runs; batch means only with `k >= 20` and a flat half-width curve |
| Comparison | Share streams and test the mean difference; CRN cut the required runs from 1056 to 165 in the worked contrast |
| Tail metrics | Count exceedances, not runs; 250 runs at `p = 0.02` gives only a 56% chance of five events |
| Design of experiments | Full factorial for up to five factors; OFAT cannot see the interaction that flips the recommendation |
| Cost discipline | Precision costs a square root; 72 runs at 31.5 h is 2268 rep-hours, 141.8 wall hours on 16 workers |
| Reconciliation | Arrivals minus departures must equal the count still in system; a mismatch invalidates every downstream average |
| Audit record | Input-model version, seed and stream rule, `w`, `T`, estimator, achieved half-width, reconciliation counts |
