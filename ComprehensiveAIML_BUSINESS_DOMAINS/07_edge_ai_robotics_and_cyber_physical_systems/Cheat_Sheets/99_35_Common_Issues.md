---
title: "35 Common Issues Across Edge AI and Robotics Boundaries"
topic: "Cross-boundary diagnostic issues spanning embedded systems, deployment runtimes, sensor fusion, robotics middleware, perception, simulation, and fleet operations"
collection: "Part 07 — Edge AI, Robotics and Cyber-Physical Systems"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 35 Common Issues Across Edge AI and Robotics Boundaries

## 1. File Header

**Purpose.** Give a reviewer a way to turn a diffuse symptom — "the robot behaved oddly after the update" — into one discriminating observation that separates the three or four mechanisms that could produce it.

**Outcome.** Given a symptom, name the contract that was violated, choose the cheapest observation that separates the competing causes, contain the physical risk before collecting volume telemetry, and specify a correction with a verification that could fail.

**Audience.** Embedded engineers, perception engineers, robotics integrators, simulation owners, and fleet operations staff who sit in integration reviews and are asked to sign a release.

**Prerequisites.** Ability to convert units and timestamps, read a timestamped trace, compute a stopping distance, and distinguish an estimate from a measurement. No prior exposure to any particular vendor stack is assumed.

**Scope.** Thirty distinct issues, each tied to one boundary in the acquisition-to-fleet chain, with at least two competing causes and one discriminating check for each. The count is thirty and not thirty-five because padding the catalogue with near-duplicates would make the discriminating checks useless: two entries whose checks are the same are one entry.

**Not covered.** Electrical design approval, functional-safety certification, medical interpretation, and legal determination of liability. Those require qualified people and, in most jurisdictions, an accredited body; nothing here substitutes.

**Evidence.** Every arithmetic result is DERIVED and was recomputed. Every scenario is ILLUSTRATIVE: no incident described here was observed, reproduced, or deployed while authoring. Where a standard or method is named, the name identifies a technique to verify, not a compliance conclusion.

**Version.** 1.0. Independent editorial and numerical review not yet recorded.

**Retrieval cue.** Section 8 holds the catalogue; section 12 holds the one-page decision card. A reader with a live symptom should go to Section 8's index table, find the symptom family, and read only the entry it points at.

**Boundary.** The catalogue diagnoses interfaces. It does not tell anyone whether a robot is safe to operate; that determination belongs to a competent person with site knowledge and, where required, to a notified body.

## 2. Concept Map / Mental Model

```text
physical process       position, temperature, vibration, current, waveform
   |  (channel, units, sample clock, hardware sequence)
acquisition            samples that are individually right and jointly wrong
   |  (filter state, feature window, normalisation revision)
preprocessing          values whose meaning depends on a version string
   |  (model, quantisation scheme, tensor layout, kernel selection)
inference              outputs whose semantics differ from their shape
   |  (frame, covariance, observation time, association)
estimation             plausible numbers with invalid uncertainty
   |  (map, footprint, feasibility, policy envelope)
planning               paths that are valid in a frame nobody checked
   |  (deadline, command age, saturation, watchdog)
control                commands that arrive after the world moved
   |
physical process       the only place the loop is actually closed

fleet plane  --signed manifest-->  versioned local boundaries
remote operator --bounded lease--> local action gate
diagnostic plane <--sampled records-- every stage
safety function --independent of--> cloud health, network, operator attention
```

Four properties of that diagram carry most of the diagnostic value.

**Arrows are contracts, not pipes.** Each arrow has a producer, a consumer, and a set of assumptions the consumer makes that the producer never promised. Almost every issue in this file is a consumer trusting a metadata field the producer left unset, unversioned, or unpopulated.

**The loop closes only at the bottom.** Offline accuracy, simulator success, and bench repeatability each evaluate a fragment. A component that is 99 percent correct in isolation can be 80 percent correct in the loop, because the loop feeds its own errors back.

**The safety function must not depend on the diagnostic plane.** A protective stop that requires a healthy network, an available cloud service, or an attentive remote operator is not a protective stop; it is a request.

**Sampling has a cost that lands on the control path.** Every diagnostic record costs CPU, memory bandwidth, and bus time. A logging change that seems free at 10 Hz can make a 100 Hz deadline miss, and the resulting failure will look like a control bug.

| Symptom family | First boundary to inspect | Discriminating question | Issue IDs |
|---|---|---|---|
| Missing, duplicated, or stale samples | acquisition to consumer | Did the hardware sequence advance without delivering a complete block? | CI-001 to CI-005 |
| Correct host output, wrong device behaviour | model to runtime | Did semantics, support, or memory change between host and target? | CI-006 to CI-010 |
| Plausible estimates with wrong uncertainty | sensor to estimator | Are the time, frame, and correlation assumptions still valid? | CI-011 to CI-015 |
| Robot valid in isolation, wrong in the world | estimator to planner | Is the transform current, or merely plausible? | CI-016 to CI-020 |
| Path feasible on paper, unsafe in practice | planner to robot | Does the swept volume match the loaded machine? | CI-021 to CI-025 |
| Simulation passes, field fails | simulator to reality | Which latent variable differs between the two? | CI-026 to CI-028 |
| Fleet healthy, local hazard rising | operations to robot | Who owns authority, evidence, and rollback? | CI-029 to CI-030 |

## 3. Foundational Definitions

**Contract.** A set of assumptions a consumer makes about a producer's output: units, range, timestamp meaning, validity window, and failure signalling. A contract is violated when any of those assumptions stops holding without the consumer noticing.

**Boundary.** An interface where two components with different owners exchange data or authority. Boundaries are where version skew, unit confusion, and responsibility gaps accumulate.

**Observation time versus processing time.** Observation time is when the physical event occurred. Processing time is when the record describing it reached a database. Mixing them produces counts that depend on filing behaviour rather than on the world.

**Sample age.** The interval between the observation time stamped on a sample and the moment a consumer acts on it. Age is not latency: latency is one contributor to age, along with queueing, transport, and scheduling.

**Deadline.** The maximum sample age at which a consumer's output remains valid for its purpose. A deadline is a property of the consumer's decision, not of the producer's rate.

**Estimate versus measurement.** A measurement is a value read from an instrument with a stated calibration. An estimate is a value inferred from other measurements plus a model. Confusing them is the single most common cause of overconfident fusion.

**Uncertainty budget.** The set of terms that contribute to an estimate's error, each with a magnitude and a sign where known. A covariance that omits a term is not conservative; it is wrong in a direction nobody has checked.

**Compensating control.** A measure that reduces risk without removing the cause: a speed cap over bad geometry, an escort over a perception gap. Every compensating control has a validity condition and a review date, and loses both if nobody writes them down.

**Safe state.** The condition the machine is brought to when a protective function activates. It is hazard-specific: for a carried load it may be a controlled stop with brakes engaged, not power removal, because power removal can drop the load.

**Authority lease.** A time-bounded, scope-bounded permission for a remote or higher-level agent to influence local motion. The lease must expire without a renewal message, so that a lost network fails safe rather than leaving an unexpiring command.

**Latent variable.** A property of the environment or the machine that influences outcomes and is not observed. Camera exposure, floor friction, pedestrian density, and battery age are all latent variables that a simulator will hold constant unless told otherwise.

**Drift.** Movement of a calibration quantity over time, usually expressed in parts per million. Drift matters in proportion to the interval over which it accumulates, not to its instantaneous rate.

**Ascertainment.** The process by which events enter a dataset. When ascertainment changes — a new form, a new reporting channel, a new incentive — counts change without the world changing.

**Enforcement path.** The code and configuration that actually constrain behaviour, as opposed to the policy that describes the intended constraint. A policy without an enforcement path is a suggestion; the distinction is testable by commanding a violation and observing whether it is refused.

**Idempotency key.** An identifier attached to a request such that a repeated request with the same key produces at most one physical effect. Without a key, a retry cannot be distinguished from a new request, and the two require opposite handling.

**Swept volume.** The set of points occupied by a machine over a trajectory, including the load and any tool. It is a volume, not a footprint, and its extent grows with path curvature and load overhang; a plan-view outline is a projection that discards exactly the dimension most collisions occur in.

**Ascertainment ratio.** The number of recorded events divided by the number of candidate events that an independent instrument observed. It measures the recording process rather than the world, which is why it can separate a reporting change from a hazard change when no other quantity can.

**Latency budget.** The allocation of a deadline across stages, with each stage's allowance stated in advance. A budget without an owner per line is a wish; a budget whose lines are never measured is a fiction that survives until the first incident.

**Null model.** A model with the same input and output interface as the real one but no computation, used to measure the fixed cost of the surrounding pipeline. Its latency is the floor below which no compression of the real model can go, and it should be measured before any optimisation is planned.

**Idle-versus-productive exposure.** The distinction between time a machine is powered and time it can produce the event being counted. Using idle-inclusive exposure understates every rate by the duty cycle and makes the rate incomparable with any external threshold, even though ratios between two equally-defined periods survive.

## 4. Core Formulas with Full Derivation

### 4.1 Base case — sample age and the deadline it violates

```text
FORMULA:
  age = t_act - t_obs = t_capture + t_queue + t_transport + t_sched
  valid  <=>  age <= D
  utilisation = age / D

WHERE:
  t_obs       observation timestamp written by the producer [s]
  t_act       time the consumer's output is actuated [s]
  t_capture   sensor integration and readout [s]
  t_queue     time in any producer or transport queue [s]
  t_transport bus, network, or shared-memory transfer [s]
  t_sched     time waiting for the consumer's executor [s]
  D           the consumer's deadline [s]

CONSTRAINT:
  All four components must be measured on the same clock domain, or the sum
  is meaningless. Clock offset between producer and consumer adds a term that
  is indistinguishable from queueing unless synchronisation is verified.

NUMERICAL EXAMPLE:
  t_capture = 5 ms, t_queue = 30 ms, t_transport = 4 ms, t_sched = 18 ms
  age = 5 + 30 + 4 + 18 = 57 ms
  D = 80 ms  ->  utilisation = 57/80 = 0.7125

INTERPRETATION:
  The system meets its deadline with 28 percent headroom, which sounds
  comfortable until the queue term is examined: at 30 ms it is more than half
  the total age, so the deadline is governed by queueing policy rather than by
  sensor or compute speed.
```

**Validity condition.** The relation holds only if the four terms are additive and independent. They are not when the queue is downstream of a scheduler that also delays the transport, because then one stall appears in two terms and the sum double-counts. Verify independence by perturbing one term and observing whether the others move.

### 4.2 General case — a queue under a slower consumer

```text
FORMULA:
  arrival_rate = f_in            [frames/s]
  service_rate = f_out           [frames/s]
  rho          = f_in / f_out    stability requires rho < 1
  backlog(t)   = (f_in - f_out) * t           while saturated
  age_worst    = backlog / f_out + service_time

WHERE:
  rho          utilisation of the consumer [dimensionless]
  backlog      frames waiting [count]
  service_time 1/f_out [s]

CONSTRAINT:
  Valid while the consumer is work-conserving and the producer does not drop.
  If the producer drops when the queue is full, the formula gives a latency
  bound that overstates the age of the frames that do arrive — and understates
  the number of frames that never arrive at all.

NUMERICAL EXAMPLE:
  f_in = 40 frames/s, f_out = 25 frames/s  -> rho = 1.6 (unstable)
  backlog growth = 15 frames/s
  after 2.667 s: backlog = 40 frames
  age_worst = 40/25 + 0.040 = 1.640 s
  A 40-frame queue at 40 Hz instead: 40 * 0.025 = 1.000 s of buffered video

INTERPRETATION:
  The consumer is 37.5 percent too slow, so no tuning of the consumer fixes
  this. The only sound options are to drop by policy (keep the newest) or to
  reduce the producer's rate. A bounded queue with a drop-oldest rule converts
  an unbounded age into a fixed 40 ms penalty, which is a different system.
```

**Validity condition.** The instability conclusion requires that both rates be sustained. A burst of 40 Hz into a 25 Hz consumer for 100 ms creates a 1.5-frame backlog and no meaningful age growth; the same rates sustained for seconds create a growing queue until memory or the drop policy intervenes.

### 4.3 Variants — correlation in fusion, and the variance you did not add

```text
FORMULA (naive fusion of two estimates):
  var_naive = 1 / (1/v1 + 1/v2)
FORMULA (with correlation rho between the two inputs):
  var_true  = (v1 + v2 - 2*rho*sqrt(v1*v2)) / (1 - rho^2)   [equal-weight case]
  inflation = var_true / var_naive

WHERE:
  v1, v2  individual variances [units^2]
  rho     correlation coefficient, -1 < rho < 1 [dimensionless]

CONSTRAINT:
  The closed form above applies to a two-input inverse-variance combination;
  for n inputs the general result is a matrix inversion and the ratio depends
  on the correlation structure, not on a single rho. Using the scalar formula
  for n > 2 is an approximation that underestimates the inflation.

NUMERICAL EXAMPLE:
  v1 = v2 = 0.01 (sd 0.10 m), rho = 0.90
  var_naive = 1/(100 + 100) = 0.005        -> sd 0.07071
  var_true  = (0.01 + 0.01 - 2*0.9*0.01)/(1 - 0.81)
            = (0.02 - 0.018)/0.19 = 0.002/0.19 = 0.010526
            -> sd 0.10259
  inflation = 0.010526 / 0.005 = 2.105

INTERPRETATION:
  Fusion improves on either input (0.1026 against 0.1000) but delivers almost
  none of the improvement the independence assumption promises. A filter that
  reports 0.07071 m while the truth is 0.1026 m is 45 percent overconfident,
  and that overconfidence propagates into every gate downstream.
```

**Validity condition.** Correlated inputs arise whenever two sensors observe the same physical quantity through overlapping processes: two cameras sharing exposure control, two IMUs on the same chassis, a lidar and a camera whose extrinsic calibration came from one survey. Independence is a claim that must be evidenced, not assumed, and the check is whether the innovation sequence's empirical variance matches the filter's predicted variance.

### 4.3a Variant — the same inflation expressed as a covariance floor

```text
FORMULA:
  var_floor = rho * sqrt(v1 * v2)          [shared component]
  var_reported = var_fused + var_floor
  confidence_overstatement = sqrt(var_floor / var_fused)

WHERE:
  rho            correlation between the shared error sources
  v1, v2         individual variances
  var_fused      the independence-based combination

CONSTRAINT:
  The floor is a heuristic, not an exact result: it holds when the shared
  component is additive and identical in both inputs, which is true for a
  shared calibration and false for a shared clock offset, where the error
  enters through the derivative rather than the value.

NUMERICAL EXAMPLE:
  v1 = v2 = 0.01, rho = 0.9
  var_fused = 0.005, var_floor = 0.9 * 0.01 = 0.009
  var_reported = 0.014  -> sd 0.1183 against a true 0.1026
  confidence_overstatement = sqrt(0.009/0.005) = 1.342

INTERPRETATION:
  The floor over-corrects here (0.014 against the exact 0.0105) because the
  shared term is not fully additive. As a conservative bound for a filter
  that cannot model correlation, over-correcting is the right direction: it
  makes downstream gates stricter instead of more permissive, and the cost of
  strictness is extra conservatism rather than a missed hazard.
```

**Validity condition.** Use the floor when the exact structure is unknown and when the consequence of overconfidence exceeds the consequence of conservatism. Do not use it as a substitute for modelling correlation in a system where the covariance is itself a control input — a tracking gate sized from an inflated covariance will reject good measurements and diverge.

### 4.3b Variant — budget allocation across stages

```text
FORMULA:
  deadline D, stages i = 1..n with measured ages a_i
  slack S = D - sum(a_i)
  proposed allowance for stage j: a_j' = a_j + S * (a_j / sum(a_i))
  resulting utilisation = sum(a_i') / D = (sum(a_i) + S) / D = 1.0

WHERE:
  a_i  measured age contribution of stage i [s]
  S    total slack against the deadline [s]

CONSTRAINT:
  Proportional allocation assumes each stage's variance scales with its mean,
  which is false for a queue: a queue's variance is dominated by the arrival
  process, not by its mean occupancy. Proportional allocation is therefore a
  starting point, not an answer, for any stage that contains a queue.

NUMERICAL EXAMPLE:
  measured ages 5, 30, 4, 18 ms against D = 80 ms
  sum = 57, S = 23
  proportional allowances: 5 + 23*5/57 = 7.02; 30 + 23*30/57 = 42.11;
                           4 + 23*4/57 = 5.61; 18 + 23*18/57 = 25.26
  sum = 80.00, utilisation 1.00

INTERPRETATION:
  Proportional allocation spends the entire slack, leaving no margin for the
  tail. A better rule reserves a fraction of the slack as an unallocated
  contingency — 20 percent, so 4.6 ms — and allocates 18.4 ms proportionally,
  which leaves a 76.6 ms budget against an 80 ms deadline and gives the queue
  stage a measurable headroom rather than an exact fit.
```

**Validity condition.** Allocation by proportion is only meaningful when the stages are independent. If two stages share a resource — the same core, the same memory bus, the same lock — reducing one's allowance can increase another's actual age, and the allocation must be validated by measurement rather than by arithmetic.

### 4.4 Limits and counterexamples

**Limit 1 — zero latency does not imply zero age.** A sensor that timestamps at readout but whose clock is 2 s ahead of the consumer's clock produces negative age. Negative age is not a fast system; it is a clock defect, and downstream code that clamps age at zero will hide it permanently.

**Limit 2 — the stopping formula is not the braking formula.** Braking distance alone is v²/(2a). Required stopping distance includes reaction, braking, and the distance the hazard travels during reaction. At 1.2 m/s with 0.15 s reaction, 1.0 m/s² deceleration, and a hazard approaching at 1.3 m/s, braking alone is 0.72 m while the requirement is 2.655 m. A design checked against 0.72 m has a margin that exists only on paper.

**Limit 3 — peak memory is not the sum of buffers.** A graph with three 1.2 MB activation buffers whose lifetimes do not overlap needs 1.2 MB, not 3.6 MB, which is 33.3 percent of the naive sum. Conversely an allocator that cannot prove non-overlap will reserve the sum and then fail at the peak of a different graph. Peak is a property of the lifetime schedule, not of the tensor list.

**Limit 4 — a stable rate can hide an unstable rate.** A pooled event rate that has not changed can conceal two strata moving in opposite directions. The general case is Simpson's paradox, and the defence is to stratify before pooling whenever the population is heterogeneous in a way that matters to the mechanism.

## 5. Worked Numerical Example

A production line cell runs a 6-axis arm with a wrist force sensor, a 2D lidar for presence detection, and an edge box that runs a 12-bit ADC sampling a strain bridge at 1 kHz. The complaint is that the arm occasionally performs a slow, tentative approach it was not programmed to make, once or twice per shift.

**Step 1 — establish the symptom's grain.** "Once or twice per shift" is a rate of about 0.19 per hour over an 8-hour shift. A defect with that rate cannot be diagnosed from a single occurrence; it needs a trace, and the trace has to be at the boundary most likely to be involved.

**Step 2 — enumerate the mechanisms before collecting anything.** Five candidates: (a) the strain bridge's reference voltage drifted and the force reading is offset, causing a real force to be interpreted as contact; (b) the ADC's quantisation step is coarse enough that a small force is indistinguishable from noise; (c) the lidar's timestamp jitter makes a person appear closer than they are, triggering a slow-approach policy; (d) the wrist force offset drifts thermally within a shift; (e) a queued command from the previous cycle is executed late.

**Step 3 — compute the ADC resolution.** With a 3.3 V reference and a 12-bit converter, one least-significant bit is 3.3/4096 = 0.8057 mV. A raw reading of 2810 counts with a 2048-count mid-scale offset corresponds to (2810 − 2048) × 3.3/4096 = 0.6139 V. At a bridge sensitivity of 2.0 mV/V per full-scale force of 50 N, that is 0.6139 V / 2.0 mV/V = 307 full-scale units, which is not a plausible force reading — so the reading is being interpreted through a calibration constant that assumes a different bridge. That mismatch is mechanism (a), and it is a configuration defect, not noise.

**Step 4 — check the noise floor against the quantisation step.** A 12-bit converter has an ideal signal-to-noise ratio of 6.02 × 12 + 1.76 = 74.0 dB. Measured on this cell, the bridge's own noise floor is 0.4 mV RMS, which is 0.4/0.8057 = 0.50 LSB. A noise floor below one LSB means the converter is dithering on its own quantisation: consecutive readings of a static load will alternate between two adjacent codes, and any filter that averages will produce a value that sits on a code boundary and jumps. An 8-bit converter on the same bridge would have a 12.89 mV step, which is 32 times the bridge's noise floor, and the force resolution would be unusable. So mechanism (b) is real but not the cause of the symptom: the resolution is adequate; the calibration is wrong.

**Step 5 — compute the timestamp jitter's effect.** The lidar runs at 25 Hz with 3 ms of jitter. At an approach speed of 1.5 m/s, 3 ms is 4.5 mm of position error, which cannot trigger a slow approach. Mechanism (c) is excluded quantitatively, not by argument.

**Step 6 — compute the thermal drift.** The strain bridge's zero drifts 40 ppm per °C, and the cell warms 6 °C over a shift. A 240 ppm drift across a 2048-count offset is 2048 × 240 × 10⁻⁶ = 0.49 counts, which is below one LSB and cannot explain a force excursion. Mechanism (d) is excluded.

**Step 7 — find the queue.** The arm controller consumes force at 250 Hz but the edge box publishes at 100 Hz and the bridge samples at 1 kHz with a 10-frame averaging window. Averaging 10 frames at 1 kHz adds 10 ms of group delay; the publication queue adds up to 10 ms; the controller's executor adds up to 4 ms. Total age up to 24 ms, and the controller's deadline for force-based compliance is 20 ms. Utilisation is 24/20 = 1.2: the loop is over its deadline on every cycle, and the slow approach is the controller's own conservative behaviour when it detects an over-age input.

**Step 8 — verify the diagnosis cheaply.** Reduce the averaging window from 10 to 4 frames and observe whether the symptom's rate falls from 0.19 to near zero per hour. That is a one-parameter change, reversible, and it distinguishes the queueing mechanism from every calibration mechanism, because calibration errors do not respond to latency.

**Step 9 — sensitivity.** If the cell's ambient temperature swing were 15 °C instead of 6 °C, the drift term would be 2048 × 600 × 10⁻⁶ = 1.23 counts, which is still under 2 LSB, so the conclusion is robust to that input. If the bridge sensitivity were 0.2 mV/V instead of 2.0, the voltage reading would correspond to 3070 full-scale units, still impossible, so the calibration conclusion is robust too. The one input the conclusion is sensitive to is the controller's deadline: if the real deadline were 30 ms rather than 20 ms, the loop would meet it and the queue would be innocent.

**Step 10 — independent re-check of the headline number.** The conclusion rests on the 24 ms age against a 20 ms deadline. Recompute from the components without reference to the earlier sum: 10 frames at 1 kHz averaged as a moving block contributes a group delay of (10 − 1)/2 = 4.5 sample periods if the filter is centred, or 9.5 if it is causal. The causal figure is the right one for an online system, so averaging contributes 9.5 ms, not 10. The publication queue at 100 Hz with one frame buffered contributes up to 10 ms, and the controller's executor contributes up to 4 ms. The total is 23.5 ms, which is still above 20 ms, so the conclusion survives the re-check while the number moves by 0.5 ms — a useful reminder that a diagnosis resting on a 4 ms margin should be stated as a direction, not a precise value.

**Step 11 — the check that would overturn the diagnosis.** If the averaging window were already 4 frames in the field firmware and the 10-frame figure came from a configuration file that was not the one deployed, the whole queue analysis collapses and the calibration defect would be the sole cause of the slow approach. The discriminating observation is the deployed configuration's effective window, read from the device rather than from the repository. That is the same class of error the catalogue records as CI-010: the tested artefact is not the deployed artefact, and no amount of correct reasoning about the tested artefact helps.

**Step 12 — what to write down.** A useful incident note states: the symptom rate (0.19 per hour), the mechanism selected (over-age force input causing the controller's conservative mode), the two mechanisms excluded quantitatively (timestamp jitter at 4.5 mm, thermal drift at 0.49 counts), the one mechanism left untested (calibration scaling, which is a separate defect and does not explain the intermittent symptom), the correction (averaging window 10 to 4, giving 3.5 ms of group delay and 17 ms total age), and the verification (symptom rate over the next 40 hours). The note is useful because it records what was excluded and how, which is the part a later reader cannot reconstruct.

## 6. Visual / Geometric Intuition

Picture the acquisition-to-control chain as a sequence of windows, each with a width equal to its own latency. The end-to-end age is the sum of widths, and the deadline is a ruler laid alongside the chain. A chain that fits inside the ruler is valid; a chain that does not is invalid regardless of how fast any single window is.

The same picture explains why "optimise the model" so rarely fixes a latency complaint. Model inference is one window. When it is 10 ms of a 57 ms chain, halving it buys 5 ms and changes nothing; when it is 45 ms of 57, halving it is the only change that matters. Always compute the widths before choosing what to optimise.

A second picture is the safety envelope drawn as a corridor. The robot's width is the swept volume, not the chassis footprint: for an arm-mounted tool the swept volume includes the tool's path, and for a loaded mobile robot it includes the load's overhang. The corridor's clearance is measured to the nearest obstacle at the height band the load occupies, not at lidar height. Most footprint errors are height errors, and a lidar mounted at 0.15 m cannot see the pallet overhang at 0.9 m that the load will hit.

A third picture is the uncertainty ellipse that is drawn too small. Correlation between inputs does not merely fail to reduce the ellipse; it leaves the ellipse roughly the size of a single input while the filter draws it as if two independent measurements had been averaged. On a plot, the failure looks like a tight ellipse around a position that is consistently 10 cm off in the same direction — and that same-direction bias is the visual signature of a shared, unmodelled error source.

```text
age chain (57 ms of an 80 ms deadline):
|capture 5|--queue 30-------|bus 4|--sched 18----|
|0        5                35   39             57|  deadline 80

envelope corridor at 1.2 m/s:
|<-- reaction 0.18 -->|<-- braking 0.72 -->|<-- hazard travel 1.755 -->|
 total 2.655 m   |  measured clearance 1.90 m  |  shortfall 0.755 m
```

The corridor drawing is the one to keep in mind during an incident review, because it makes the shortfall a length rather than an adjective. "Insufficient clearance" ends a conversation; "755 mm short at 1.2 m/s" starts a computation.

### 6.1 Why the envelope grows faster than speed

The envelope has three terms with three different scalings, and the mixture is what makes speed reductions so much more effective than they look.

The reaction term v·τ is linear. It is the distance covered before anything happens, and it is the only term that a faster processor directly reduces.

The braking term v²/(2a) is quadratic. Doubling speed quadruples the distance needed to stop, which is why a small overspeed on a slow robot matters more than intuition suggests.

The hazard-travel term v_p·(τ + v/a) is linear in the robot's speed through the braking time and independent of the robot's speed through the reaction time. This is the term that is most often omitted and most often decisive at low speeds: at v → 0 it does not vanish, so there is a floor of v_p·τ = 0.35 m below which no speed reduction can go.

The practical consequence is that the marginal benefit of slowing down is largest in the range where the quadratic term dominates, roughly above v = a·τ = 0.15 m/s for the parameters above. Below that, the envelope is governed by the floor and speed changes buy almost nothing.

### 6.2 The queue as a growing segment

Draw the age budget as a bar and the queue as a segment that grows to the right as time passes while the arrival rate exceeds the service rate. The bar's fixed segments are the sensor, the transport, and the scheduler; the queue segment is the only one that is a function of time rather than of configuration.

At ρ = 1.6, the queue segment grows by 15 frames per second, which at a 25 Hz service rate is 0.6 s of age per second of operation. A system in that state is not "a bit slow": it reaches eight seconds of age in the time it takes to read this paragraph, and the only reason it does not is that something — a full queue, an allocator, or a drop policy — intervenes first.

The visual is useful during diagnosis because it makes the distinction between "the pipeline is slow" and "the pipeline is accumulating" obvious. A slow pipeline has a long bar with fixed segments. An accumulating pipeline has a short bar and a growing one, and the fix for each is entirely different.

### 6.3 The covariance ellipse that does not shrink

Plot two uncertainty ellipses over the same position estimate: the one the filter reports and the one the residuals actually describe. Independent inputs produce the second ellipse visibly smaller than either input's ellipse. Correlated inputs produce a second ellipse almost the same size as a single input, with a consistent offset direction.

The consistent direction is the visual signature. An unmodelled shared error does not produce random scatter around the truth; it produces scatter around a displaced truth, so the residuals look small from one viewpoint and large only when compared against an external reference. That is why internal consistency checks pass while the estimate is wrong, and why the only reliable check is an external reference.

The same picture explains why adding sensors sometimes makes a system worse: each additional sensor with the same shared error adds a term to the covariance reduction without adding information, so the reported uncertainty falls while the actual error does not.


## 7. Comparison / Selection Table

| Diagnostic instrument | Cost | Latency to answer | What it can distinguish | What it cannot |
|---|---|---|---|---|
| Hardware sequence counter | One field in the record | Immediate | Torn or duplicated sample blocks versus genuine data | Whether a complete block was semantically current |
| Producer-side timestamp plus clock sync check | One field plus an NTP/PTP audit | Minutes | Queueing versus transport versus scheduling delay | Whether the producer's own stamp is honest |
| Commanded-versus-realised comparison | Telemetry join | Hours | Policy violation versus enforcement gap | Which code path ignored the constraint, unless segmented by controller ID |
| Candidate-event versus filed-report ratio | Two exports | Days | Ascertainment change versus physical change | The absolute physical rate, because both sides are unvalidated |
| Paired bench-versus-field episodes | Rig time | Days | Transfer gap magnitude and direction | Whether the field sample is representative |
| Stratified rate comparison | Existing records | Hours | Which sub-population carries the change | The mechanism inside the stratum |
| Exact Poisson bound | Arithmetic | Minutes | Whether a rare-event gate can pass at all | Anything about the rate below the bound |
| Fault injection on the safety path | Controlled test | Hours | Whether the protective function is independent | Whether it is sufficient for the hazard |
| Simulator with the real layout imported | Modelling time | Weeks | Whether geometry explains the observed hazard | Anything the simulator's latent variables omit |
| Signed release manifest audit | Configuration diff | Hours | Whether the deployed artefact matches the approved one | Whether the approved one was correct |

The selection rule is the one in the table's first column: choose the cheapest instrument whose "can distinguish" column covers your live hypothesis and whose "cannot distinguish" column does not cover it too. A team that starts with the simulator usually learns the geometry it already knew; a team that starts with a stratified rate comparison usually narrows the problem before spending modelling time.

Two rows deserve a warning. The commanded-versus-realised comparison is the one instrument that catches the most common containment failure, and it is also the one most often omitted, because it requires a join between two systems with different owners. The candidate-versus-filed ratio is the only instrument that can rescue a conclusion from an ascertainment change, and it is almost always a data request rather than a query, which is why incidents are decided on unvalidated counts.

### 7.1 The four-question selection procedure

A reviewer facing a symptom should answer four questions in order, and stop at the first one that produces a discriminating observation.

**Question one: what changed, and could the change be an ascertainment change?** If the only change is in how events are recorded — a new form, a new channel, a new shift pattern — then no physical explanation is needed and the investigation is an audit. This question costs an hour and eliminates the largest class of false alarms.

**Question two: is the symptom localised, and to what?** Stratify by every dimension that is cheap: site, robot variant, geometry class, shift, software version, sensor model. A symptom confined to one stratum is a local mechanism; a symptom present in all strata is a fleet or process mechanism. The stratification costs a query and determines the entire subsequent search.

**Question three: what quantity, in physical units, would have to be wrong to produce this symptom?** This is the step most reviews skip. It forces the investigation to name a number — a latency in milliseconds, a gain in counts per unit, a distance in millimetres, a rate per thousand hours — and a named number can be measured. "The perception is unreliable" cannot.

**Question four: what is the cheapest observation that separates the top two mechanisms?** Not the most complete observation: the cheapest. If a log field distinguishes them, the field costs minutes. If a bench experiment distinguishes them, it costs a day. If only a field trial distinguishes them, the mechanism pair is too expensive to separate and the decision must be made under acknowledged ambiguity with a containment that covers both.

### 7.2 The instrument-versus-decision matrix

The instruments above are diagnostic; the decision they inform has a different structure, and mixing the two is a common review error.

A diagnostic instrument answers "what is happening". A decision instrument answers "what should we do, and what evidence would change it". The bridge between them is a threshold, and a threshold without a decision attached is a metric nobody acts on.

Three failure patterns follow from confusing them. The first is the metric without a threshold: a dashboard reports a rate and nobody knows whether it is acceptable, so it is read as green until an incident retroactively defines it as red. The second is the threshold without a mechanism: a rule such as "stop if the rate exceeds twice baseline" fires on a reporting change and stops a system whose behaviour is unchanged. The third is the decision without an instrument: a containment is chosen because it feels right, and its effect is never measured, so it is either kept forever or removed by the next budget cycle for reasons unrelated to safety.

The cure for all three is to write the decision and its evidence in the same document: the threshold, the population, the window, the comparator, and the action taken if the threshold is crossed. If any of those five is missing, the metric will eventually be interpreted by whoever is most confident in the room.

## 8. Failure Modes & Edge Cases

All entries are ILLUSTRATIVE teaching scenarios. The category describes the failure's structure, independently of how mature the surrounding evidence is. "Competing causes" names mechanisms to test, not conclusions about any real incident.

The catalogue is ordered along the signal chain rather than by severity, because a reviewer's first problem is localisation, not triage. Once the entry is found, the "competing causes" list is the triage: it names the mechanisms in the order they are cheapest to test, which is not the same as the order they are most likely.

Each entry has the same six fields, and all six are load-bearing. The symptom is written as an observation a person could make without instruments, because that is how symptoms arrive. The competing causes are at least three, because two causes invite a coin flip. The discriminating check must be able to return "neither", which is what makes it a check rather than a confirmation. The correction changes the system rather than the symptom. The prevention names the artefact that would have caught it. The verification is a test that could fail.

Read the six fields as a single argument. An entry whose prevention would not have caught the issue, or whose verification cannot fail, is an entry that describes a belief about the system rather than a defect in it.

### 8.1 Escalation protocol when two entries fit

A real symptom often matches two entries. The rule is to prefer the entry whose check is cheaper and to run both checks if the first returns "neither", but there are two situations where the order must be inverted.

If one of the candidate entries has a physical-consequence path — it can hurt someone before the diagnosis completes — its check runs first regardless of cost, or the system is contained while the other check runs. Cost ordering is a productivity heuristic, and it is subordinate to containment.

If one candidate entry explains a symptom that has been seen before at another site and resolved, its check runs first because the resolution is already known. Cross-site history is the cheapest instrument available and is chronically underused; a defect that took three weeks to find at one site should take an afternoon at the second.

A third consideration applies when both checks are expensive: whether the two mechanisms predict different behaviour under an observable condition. If they do, one experiment can separate them. If they do not, the ambiguity is irreducible with the available instruments, and the honest action is to contain for both mechanisms and record the ambiguity rather than to spend the window resolving it.


| ID | Category | Symptom | First check | Related module |
|---|---|---|---|---|
| CI-001 | VIOLATED_ASSUMPTION | Repeated or torn sensor blocks | Sequence counter continuity | 35 |
| CI-002 | SCALE_FAILURE | Deadlines missed only while logging | Executor occupancy under logging load | 35 |
| CI-003 | NUMERICAL_EDGE | Sudden offset in converted units | Reference voltage and calibration version pair | 35 |
| CI-004 | CONCEPT_CONFUSION | Fast interface, missing payloads | Full wire-format time budget | 35 |
| CI-005 | VIOLATED_ASSUMPTION | Acceptable battery life only on the bench | Whole-cycle energy including wake and sleep transitions | 35 |
| CI-006 | VIOLATED_ASSUMPTION | Rare-class saturation in production | Clipping rate on a held-out extreme slice | 36 |
| CI-007 | CONCEPT_CONFUSION | Smaller model, no latency gain | Whether a sparse or 4-bit kernel exists for the target | 36 |
| CI-008 | NUMERICAL_EDGE | Correct shape, wrong class after conversion | First tensor at which boundary values diverge | 36 |
| CI-009 | SCALE_FAILURE | Memory crash after warm start | Actual peak over the lifetime schedule | 36 |
| CI-010 | VIOLATED_ASSUMPTION | Optimised build drifts from reference | First failing compiler pass, by bisection | 36 |
| CI-011 | VIOLATED_ASSUMPTION | Fusion error grows during rotation | Residual correlation with angular rate | 37 |
| CI-012 | CONCEPT_CONFUSION | Mirrored point cloud | Known-target transform through the full chain | 37 |
| CI-013 | VIOLATED_ASSUMPTION | Overconfident fused estimate | Empirical versus predicted innovation variance | 37 |
| CI-014 | NUMERICAL_EDGE | Implausible physiological trend | Edit and exclusion ledger for the raw beats | 37 |
| CI-015 | SCALE_FAILURE | Low compute latency, stale output | Age distribution, not mean latency | 37 |
| CI-016 | VIOLATED_ASSUMPTION | Publisher and subscriber that never connect | QoS compatibility of the actual endpoints | 38 |
| CI-017 | CONCEPT_CONFUSION | Controller jerk at relocalisation | Odometry continuity versus map discontinuity | 38 |
| CI-018 | NUMERICAL_EDGE | Arm joint rates explode | Smallest singular value of the Jacobian | 38 |
| CI-019 | VIOLATED_ASSUMPTION | Oscillation after a payload change | Phase lag and command saturation at the new mass | 38 |
| CI-020 | SCALE_FAILURE | Watchdog fires only under load | Timer delay as a function of callback occupancy | 38 |
| CI-021 | VIOLATED_ASSUMPTION | SLAM locks onto the wrong aisle | Ambiguity of the matched geometry | 39 |
| CI-022 | CONCEPT_CONFUSION | Feasible path collides in practice | Swept volume at the load's height band | 39 |
| CI-023 | NUMERICAL_EDGE | High grasp score, dropped part | Failure dependence on friction and approach angle | 39 |
| CI-024 | VIOLATED_ASSUMPTION | Language-conditioned policy takes a forbidden action | Whether the action gate is independent of the policy | 39 |
| CI-025 | SCALE_FAILURE | Side effect repeated after a retry | Action identity across behaviour-tree ticks | 39 |
| CI-026 | VIOLATED_ASSUMPTION | Perfect synthetic validation | Group holdout partitioned by latent source | 40 |
| CI-027 | NUMERICAL_EDGE | Simulator stable, hardware unstable | Response as a function of integration step | 40 |
| CI-028 | CONCEPT_CONFUSION | Synthetic labels better than the task requires | Label-error dependence on the task's actual decision | 40 |
| CI-029 | VIOLATED_ASSUMPTION | Fleet green, local hazard rising | Stratification of the rate by site and corner | 41 |
| CI-030 | SCALE_FAILURE | Rollback that cannot complete inside the window | Time to restore, measured rather than planned | 41 |

### CI-001 — sequence counter continuity

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A consumer periodically sees a sensor block whose fields are individually plausible but mutually inconsistent: two channels that should move together are one frame apart. It appears as a small, intermittent estimation error rather than a visible fault.

**Competing causes.** (1) The DMA transfers into a buffer the consumer is still reading, so a block is overwritten mid-read. (2) The driver drops a block when its ring is full and the consumer cannot tell a gap from a repetition. (3) A filter's state is retained across a re-initialisation, so the first output after a reset mixes old and new data.

**Discriminating check.** Record the hardware sequence counter and a monotonic timestamp with every block, then look for (a) a counter that does not advance by exactly one, and (b) a counter that advances while the payload's own embedded counter does not. The first identifies a drop, the second identifies a torn read — the payload is from an older block than the header claims. Neither is visible in the field values.

**Correction.** Make ownership transfer atomic: the producer writes into a buffer the consumer does not own, then swaps. Bind the sequence counter into the payload checksum so a torn block fails validation rather than passing it.

**Prevention.** Treat the sequence counter as part of the contract, not as debug data. A contract without a continuity field cannot detect its own violation.

**Verification.** Inject a deliberate dropped block on the bench and confirm the consumer reports a gap rather than silently interpolating. Then run for 24 hours at full rate and confirm zero gaps.

### CI-002 — deadlines that fail only while logging

**Category:** SCALE_FAILURE.

**Symptom.** A control loop meets its deadline in every bench test and misses it in the field, always after a logging configuration change. Removing the log restores behaviour, which makes the log look like the cause rather than the symptom.

**Competing causes.** (1) The logger shares the control executor, so log formatting occupies the same thread the callback needs. (2) The logger fills a queue whose consumer runs at a lower priority and starves the producer. (3) The log write is synchronous to storage and the storage's worst-case latency is 50 ms, not the 2 ms measured on an empty device.

**Discriminating check.** Measure callback start-to-start jitter with logging off, at 10 Hz, and at 100 Hz. If the jitter's 99th percentile scales with log volume while CPU utilisation stays flat, the mechanism is queue or lock contention, not compute. Then measure the logger's own blocked time; a producer blocked inside a write is unambiguous.

**Correction.** Move logging off the control path: a lock-free ring with a drop-oldest policy, drained by a separate thread on a separate core. Accept that the log is lossy under overload and record the loss count.

**Prevention.** Budget the diagnostic plane explicitly, as a percentage of the control deadline, and test with it enabled at full volume. A control loop validated with logging disabled is validated for a system nobody runs.

**Verification.** Run the loop at its worst-case rate with logging at maximum and confirm the deadline's 99th percentile is unchanged from the logging-off baseline.

### CI-003 — converted units that shift after a calibration change

**Category:** NUMERICAL_EDGE.

**Symptom.** A converted quantity acquires a step offset that appears at a software version boundary and does not correlate with any physical event. Raw counts are stable while engineering units move.

**Competing causes.** (1) The reference voltage constant and the calibration table were versioned separately, so a new table met an old reference. (2) The gain is applied before the offset instead of after, which is a 1-count error at mid-scale and a large one at the extremes. (3) The sensor and the converter are on different supplies, so the ratiometric assumption underlying the conversion silently fails.

**Discriminating check.** Feed a known input at three points: zero, mid-scale, and near full scale. A pure offset error moves all three equally; a gain error moves them proportionally to the reading; a ratiometric mismatch moves them proportionally to the supply difference. Three points separate all three mechanisms, and a single point separates none.

**Correction.** Version the conversion constants with the calibration table in one artefact, and refuse to boot on a mismatch rather than applying a default.

**Prevention.** Store the calibration's provenance — date, reference, gain, offset — alongside the values, and make the conversion read the provenance rather than a global.

**Verification.** Apply the new calibration on the bench and confirm that all three test points land within one LSB of their expected values, not merely that the output "looks reasonable".

### CI-004 — a fast interface that cannot carry the payload

**Category:** CONCEPT_CONFUSION.

**Symptom.** A link is configured at a high baud rate and still loses data. Throughput measurements on a logic analyser look correct, so the link is declared healthy and the loss is attributed to the application.

**Competing causes.** (1) Start, stop, and parity bits inflate each byte's wire time by 25 percent at 8N1 with two stop bits. (2) The protocol's inter-frame gap and turnaround delay consume a fixed time per transaction that does not scale with baud. (3) The receiver's FIFO is being drained by a polled loop whose period exceeds the FIFO depth divided by the byte rate.

**Discriminating check.** Compute the full wire time: at 921,600 baud with 8N1, one byte is 10 bit-times = 10.85 µs, so a 64-byte frame is 694 µs plus a 50 µs turnaround. At 400 frames/s that is 297 ms of every second, which is 30 percent utilisation — not the 15 percent a payload-only calculation suggests. Then compare with the polled loop's period: a 1 ms poll with a 4-byte FIFO at 92 kB/s overflows every 43 µs.

**Correction.** Budget the wire time including framing and turnaround, and size the FIFO from the actual poll interval rather than from the burst length.

**Prevention.** Every protocol specification should carry a table of frames per second against wire utilisation, computed at the slowest supported baud.

**Verification.** Run at the maximum frame rate for an hour with the poll loop at its worst-case period and confirm zero overrun flags.

### CI-005 — battery life measured only on the bench

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A device that runs for 40 hours on the bench runs for 9 hours in the field. The bench test was accurate for what it measured: continuous operation at a fixed duty cycle.

**Competing causes.** (1) Wake transitions dominate: a device that wakes 200 times per hour at 3 ms of 30 mA costs 200 × 0.003 × 30 = 18 mAh per hour, against a 40 mAh average. (2) The radio's association and retry behaviour depends on the RF environment, which the bench does not have. (3) The regulator's quiescent current at low load is 12 µA on the datasheet but 90 µA at the temperature the field reaches.

**Discriminating check.** Log a timestamped current sample at 100 Hz for a full field day and integrate the charge. Then compare the integral against the sum of the bench-measured components. A gap that scales with wake count identifies transitions; a gap that scales with retry count identifies RF; a gap that is constant at low load identifies quiescent current.

**Correction.** Measure whole-cycle energy including transitions, and size the battery against the field duty cycle with at least a 2× margin while the duty cycle is still being characterised.

**Prevention.** Make the duty-cycle histogram a deliverable, not a by-product, and re-measure it whenever firmware changes the sleep policy.

**Verification.** Field-deploy three units with coulomb counters and confirm that predicted and measured charge agree within 15 percent over a week.

**Boundary.** The transition term is linear in wake count only while each wake's cost is independent of the previous one. If a wake triggers a calibration that survives into the next wake, the cost is superlinear and the whole-cycle estimate understates.

### CI-006 — rare-class saturation outside the calibration distribution

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A model trained on representative data performs well on the validation set and degrades on a specific field condition: a bright window, a reflective floor, a saturated sensor. The degradation is concentrated in a small slice that the validation set contained but did not weight.

**Competing causes.** (1) The calibration set's extremes were under-represented, so the model never saw the clipping pattern. (2) The input pipeline clips after normalisation, so a legal raw value maps outside the model's trained range. (3) The field condition changes the sensor's noise distribution rather than its mean, and the model is sensitive to variance.

**Discriminating check.** Compute the clipping rate per slice on a held-out set that deliberately over-samples the extreme condition. If the clipping rate rises while the error also rises, mechanism (1) or (2) applies; then re-run the pipeline with clipping moved before normalisation and see whether the error changes. If the error changes with clipping but not with the clip's position, mechanism (1) applies.

**Correction.** Re-weight the calibration distribution toward the operational extremes, and move clipping to a position where its effect is visible in the model's input rather than hidden in a normalisation constant.

**Prevention.** Report accuracy per slice with the slice's prevalence, so that a slice representing 0.5 percent of traffic cannot hide behind a 99.5 percent aggregate.

**Verification.** Hold out the extreme slice entirely and confirm the aggregate accuracy drop matches the slice's prevalence times its error rate.

### CI-007 — a smaller model that is not faster

**Category:** CONCEPT_CONFUSION.

**Symptom.** A model with 40 percent fewer parameters runs at the same latency. The size reduction is real; the speed does not follow.

**Competing causes.** (1) The target runtime has no kernel for the pruned or 4-bit structure, so it falls back to a dense path that executes zeros. (2) The bottleneck is memory bandwidth, not arithmetic, so reducing multiplies changes nothing while the bytes moved stay constant. (3) Latency is dominated by fixed per-inference overhead — preprocessing, allocation, synchronisation — that is independent of the model.

**Discriminating check.** Measure with a null model of the same input and output shapes. If a null model costs 60 percent of the full model's latency, the bottleneck is overhead and no compression will help. Then measure achieved FLOPS and bytes moved; a memory-bound kernel shows high bandwidth utilisation and low arithmetic utilisation.

**Correction.** Choose the compression that matches the bottleneck: structure-preserving pruning for a kernel-limited target, weight-only quantisation for a bandwidth-limited one, and pipeline fusion for an overhead-limited one.

**Prevention.** State the bottleneck before choosing the compression, and measure the null-model cost as part of the baseline.

**Verification.** Deploy the compressed model and confirm the latency gain matches the bottleneck analysis within 10 percent. A gain that appears only on the host is not a gain.

### CI-008 — correct shape, wrong semantics after conversion

**Category:** NUMERICAL_EDGE.

**Symptom.** A converted model produces an output tensor with the right shape and the right dtype whose contents are permuted or scaled. Downstream code indexes it successfully, so no error is raised.

**Competing causes.** (1) The layout changed from NCHW to NHWC and the consumer still indexes channels first. (2) The zero-point and scale were applied in the wrong order, or a per-channel scale was collapsed to per-tensor. (3) An activation's clipping range was computed on a different calibration set than the weights were quantised with.

**Discriminating check.** Compare the first and last tensors layer by layer with a deterministic input. The first divergence identifies the pass. Then check whether the divergence is a permutation — identical value multiset, different positions — which is layout, versus a scaling — values proportional but not equal — which is quantisation parameters.

**Correction.** Make the layout part of the model's declared interface, and carry the quantisation parameters as versioned metadata that the consumer validates.

**Prevention.** A conversion is not complete until a boundary tensor has been compared elementwise against the reference, and the comparison is retained as an artefact.

**Verification.** Re-run the elementwise comparison on the deployed artefact, not on the intermediate file, because deployment tooling can re-apply a layout change.

### CI-009 — memory that fits only before the warm start

**Category:** SCALE_FAILURE.

**Symptom.** An inference graph runs on the first call and fails after the first re-initialisation. The static memory report shows a comfortable fit.

**Competing causes.** (1) The arena was sized from the sum of activation buffers, but the allocator reserved the sum again on the second graph instance. (2) A warm-start path retains a reference to the previous arena, so two arenas coexist. (3) Fragmentation from varying sequence lengths makes a nominally sufficient arena unusable after a few dozen calls.

**Discriminating check.** Measure the allocator's high-water mark across a hundred calls, not the static sum. A peak that equals the sum of all buffers indicates no reuse; a peak near the largest buffer indicates correct lifetime scheduling. Then check the arena count; two live arenas is unambiguous.

**Correction.** Derive the arena from the measured peak over the real call sequence, and free the previous arena before allocating the next.

**Prevention.** Treat peak memory as a property of the workload sequence, and include a re-initialisation in the memory test, because that is where the second arena appears.

**Verification.** Run 1,000 alternating initialisations and inferences on the target and confirm the high-water mark is stable after the tenth call.

### CI-010 — an optimised build that drifts from the reference

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A compiled graph produces outputs that differ from the reference by an amount just large enough to change a decision at a boundary. The difference is invisible in an average error metric.

**Competing causes.** (1) A pass reassociated a floating-point sum, changing rounding. (2) A pass fused an activation whose saturation semantics differ at the extremes. (3) A pass lowered an operation to a fast-math variant that assumes no NaN or infinity.

**Discriminating check.** Bisect the pass pipeline by disabling passes in halves. Then feed a deliberately extreme input — large magnitudes, zeros, denormals — because these are where the semantics differ; a random input from the training distribution will not show it.

**Correction.** Pin the pass pipeline in the release manifest and test the extremes explicitly. Where a pass is required for performance, document the semantics it changes.

**Prevention.** The correctness suite should include a boundary input set that is checked against the reference at every release, not a random sample.

**Verification.** Confirm that the extreme-input test passes on the deployed binary and that the manifest's pass list matches the one that was tested.

### CI-011 — fusion that grows error during rotation

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A fused heading is accurate when the platform is stationary and drifts when it turns. The drift correlates with turn rate rather than with time.

**Competing causes.** (1) A time offset between the gyroscope and the reference sensor is mapped as motion: a 10 ms offset at 90 °/s is 0.9° of false rotation. (2) Lever-arm effects: the IMU is 15 cm from the rotation centre, so a 1 rad/s turn produces 0.15 m/s of tangential velocity that the model attributes to translation. (3) The gyroscope's scale factor is wrong by 2 percent, which is invisible at rest.

**Discriminating check.** Rotate at two rates in both directions and plot residual against rate. A residual proportional to rate with a sign that flips with direction is a scale factor; a residual proportional to rate with the same sign in both directions is a time offset; a residual present only during acceleration is a lever arm.

**Correction.** Estimate the time offset and scale factor as states in the filter, or calibrate them offline with a known-rate turntable. Move the model's reference point to the IMU or transform the IMU's measurements to the body origin.

**Prevention.** A calibration that does not include a rotation test is not a calibration of a system that rotates.

**Verification.** Re-run the two-rate test and confirm the residual is flat in rate to within the noise floor.

### CI-012 — a mirrored point cloud

**Category:** CONCEPT_CONFUSION.

**Symptom.** A point cloud looks correct in isolation — walls are vertical, the floor is flat — but the scene is mirrored. Occupancy grids built from it are plausible and wrong, and the error is a reflection, not a rotation, so it does not show up in any consistency metric that assumes a rigid transform.

**Competing causes.** (1) Optical and body axes were mixed, with one axis negated. (2) A left-handed frame convention was combined with a right-handed quaternion. (3) The extrinsic calibration was estimated on a target whose own geometry was symmetric, so the mirrored solution was equally good.

**Discriminating check.** Transform a known asymmetric target — a corner with unequal legs, or a fiducial with an off-centre marker — through the full chain and compare its coordinates with the tape measurement. A reflection is identified by a negative determinant of the composed rotation, which no amount of axis-swapping within right-handed conventions can produce.

**Correction.** Compose the chain from sensor to body with explicit frame names, and assert the determinant of every rotation is +1 at each step.

**Prevention.** Calibration targets must be geometrically asymmetric in all three axes, so a mirrored solution is measurably worse.

**Verification.** Re-project the target and confirm the residual is below the sensor's noise floor in all three coordinates, not merely in the plane.

### CI-013 — overconfident fusion from double-counted inputs

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A fused estimate's covariance shrinks as sensors are added while the actual error does not. Downstream gates that use the covariance therefore become more permissive as the system becomes no more accurate.

**Competing causes.** (1) Two inputs share a common error source — the same calibration, the same clock, the same physical process — so their errors are correlated. (2) A filter re-uses a measurement that was already incorporated, which is correlation of a different kind. (3) The process noise is set low to make the filter "smooth", which suppresses the covariance without adding information.

**Discriminating check.** Compute the normalised innovation squared over a long run and compare its mean with its expected value, which equals the degrees of freedom. A mean far above the expectation means the covariance is too small; a mean far below means it is too large. Then perturb one input's calibration and see whether the other's residual moves: if it does, they share a source.

**Correction.** Model the correlation explicitly, or fuse at a lower rate so that the shared component is not counted repeatedly. Report a floor on covariance, derived from the shared error, rather than allowing the filter to claim unlimited confidence.

**Prevention.** Every covariance must name the terms it includes and the terms it omits, and the omitted terms set a floor.

**Verification.** Confirm that the innovation statistic's mean is within the expected confidence interval over a run that covers the full operating envelope, including the extremes.

### CI-014 — a physiological trend produced by editing the raw signal

**Category:** NUMERICAL_EDGE.

**Symptom.** A derived trend improves markedly after a preprocessing change. The improvement is in the cleaning, not in the physiology, and the cleaned series no longer represents what was recorded.

**Competing causes.** (1) Artifact rejection removed the intervals that were hardest to measure, which are also the intervals with the most informative variability. (2) Interpolation across gaps created beats that never occurred. (3) A changed peak-detection threshold shifted the distribution of accepted intervals rather than improving their accuracy.

**Discriminating check.** Preserve a per-record ledger of the number of beats detected, the number rejected, and the reason. If the accepted count falls while the trend improves, the improvement is selection. Then recompute the trend on the unedited series with the same estimator and compare: a large gap between edited and unedited is the bias.

**Correction.** Report the trend with its rejection fraction, and treat a record with more than a stated rejection fraction as unmeasured rather than as a value.

**Prevention.** Quality metrics belong beside the derived value, not in a separate report nobody joins.

**Verification.** Confirm the trend on the unedited series reproduces within the stated uncertainty; if it does not, the edited value is not a measurement of the same thing.

### CI-015 — low compute latency and a stale output

**Category:** SCALE_FAILURE.

**Symptom.** Profiling shows a 6 ms inference against a 33 ms budget, yet the consumer acts on data that is 400 ms old. The compute metric is correct and irrelevant.

**Competing causes.** (1) Frames queue ahead of inference and are processed in arrival order, so the newest frame waits behind older ones. (2) A callback chain passes the frame through several stages, each with its own bounded queue, and the bounds sum to the observed age. (3) The consumer's own executor is occupied by a lower-priority task, so a freshly computed result waits to be actuated.

**Discriminating check.** Measure age at ingestion and at actuation, and plot the distribution rather than the mean. An age distribution with a long right tail and a fast left edge is queueing; a shifted distribution with constant width is a fixed pipeline delay. Then instrument each stage's queue depth.

**Correction.** Keep the newest frame and drop stale ones at ingestion, so the pipeline's age bound is its own latency rather than the queue's depth.

**Prevention.** Define the deadline in terms of age at actuation, and monitor age, because latency monitoring cannot detect a queue.

**Verification.** Confirm that the age 99th percentile under full load is within the deadline, and that a deliberately injected stall at the producer does not accumulate.

### CI-016 — endpoints that exist but never connect

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A publisher and a subscriber are both running, both report healthy, and no message arrives. Neither side logs an error.

**Competing causes.** (1) Quality-of-service settings are incompatible — a reliable publisher against a best-effort subscriber, or a volatile publisher against a transient-local subscriber. (2) The two sides use different message types with the same name, which the middleware treats as different types. (3) A namespace or domain identifier differs, so the discovery traffic never meets.

**Discriminating check.** Enumerate the actual endpoint list from the middleware rather than from configuration files. An incompatible pair shows as two endpoints that exist and do not match; a namespace mismatch shows as each side seeing no peer at all. The distinction is decisive and takes one command.

**Correction.** Set QoS explicitly and compatibly on both sides, and derive both from one shared profile artefact.

**Prevention.** A launch test should assert that every required pair has matched, not merely that every process started.

**Verification.** Confirm the matched count equals the expected count in the deployed configuration, and that a deliberate publisher restart re-matches without operator action.

### CI-017 — a controller that jerks when the map relocalises

**Category:** CONCEPT_CONFUSION.

**Symptom.** The robot lurches at the moment the localisation system corrects its global pose. Between corrections it drives smoothly.

**Competing causes.** (1) The controller consumes the global pose directly, so a map correction enters the control loop as motion. (2) The local odometry frame and the map frame are conflated, so a correction is interpreted as a velocity. (3) A safety monitor compares map-frame position against a limit and reacts to the correction rather than to the motion.

**Discriminating check.** Log odometry-increment and map-pose-increment separately for the same interval. A lurch with continuous odometry and a step in map pose identifies the mechanism. Then check which frame each consumer subscribes to; the consumers that jerk are those reading the map frame.

**Correction.** Control against a continuous local frame and treat the map frame as a slowly varying offset applied at a bounded rate. Bound the correction's rate of change and let the local frame carry the motion.

**Prevention.** Frames should be named for their continuity property, and consumers should state which property they depend on.

**Verification.** Inject a 0.35 m pose correction at 1.2 m/s and confirm the commanded velocity changes smoothly and the robot's actual path deviation stays within the correction's magnitude.

### CI-018 — joint rates that explode near a singularity

**Category:** NUMERICAL_EDGE.

**Symptom.** The arm's commanded joint velocities spike when the end effector approaches a singular configuration. The Cartesian command is small and smooth.

**Competing causes.** (1) The pseudo-inverse is computed without damping, so a small singular value amplifies a small Cartesian error into a large joint rate. (2) The damping is fixed rather than adaptive, so it is too small near the singularity and unnecessarily large elsewhere. (3) The trajectory passes through the singularity rather than around it, which no controller can make comfortable.

**Discriminating check.** Log the Jacobian's smallest singular value alongside the commanded joint rates. A joint rate that grows as the reciprocal of that value — a 0.02 singular value giving 50× amplification, so 0.5 rad/s of Cartesian demand becomes 25 rad/s — is the classical signature.

**Correction.** Apply damped least squares with a damping term that rises as the smallest singular value falls, and cap joint rates independent of the solve.

**Prevention.** The planner should avoid configurations with a singular value below a threshold, and the controller should refuse rather than amplify when it cannot.

**Verification.** Drive the arm deliberately toward the singular configuration and confirm joint rates stay within limits while tracking error degrades gracefully rather than discontinuously.

### CI-019 — oscillation after a payload change

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A controller tuned on an empty machine oscillates when a payload is added. The gains are unchanged and were correct for the old plant.

**Competing causes.** (1) The payload lowers the plant's natural frequency and increases its phase lag, so the same gain sits closer to instability. (2) The actuator saturates at a lower acceleration, so the loop's effective gain drops in the large-signal regime and the integrator winds up. (3) The payload changes the sensor's mounting compliance, adding a resonance within the loop's bandwidth.

**Discriminating check.** Step the loop at small and large amplitude. Oscillation that appears only at large amplitude indicates saturation and windup; oscillation present at both, with a frequency that scales as the inverse square root of the added mass, indicates a plant frequency shift. A high-frequency oscillation that appears only with the payload attached indicates a mounting resonance.

**Correction.** Gain-schedule by payload class, or estimate the payload and adapt the gains. Add anti-windup so saturation does not accumulate.

**Prevention.** Validate the controller across the declared payload range, and make the declared range part of the release condition.

**Verification.** Repeat the step response at the minimum and maximum payload and confirm phase margin at both, not merely that neither oscillates.

### CI-020 — a watchdog that fires only when the system is busy

**Category:** SCALE_FAILURE.

**Symptom.** A safety watchdog triggers under load and not otherwise. The watchdog's own timer is correct; what it observes is delayed.

**Competing causes.** (1) The watchdog callback shares an executor with long-running work, so the timer's expiry is late by the occupancy of whatever is running. (2) The watchdog is fed from the same thread it monitors, so it cannot detect that thread's stall. (3) The watchdog's clock is a monotonic timer that is correct, but the kick arrives late because the producer is blocked behind a lock.

**Discriminating check.** Measure the timer callback's start delay as a function of executor occupancy. A delay that tracks occupancy is scheduling, not clock error. Then identify which thread kicks the watchdog; if it is the monitored thread, the watchdog is a self-report.

**Correction.** Run the watchdog on an independent execution context with its own timer and no shared lock with the monitored path, and feed it from the monitored thread only as an assertion.

**Prevention.** The monitoring path must be demonstrably independent: separate context, separate memory, no shared mutex. Independence is testable by blocking the monitored path and observing the watchdog trip.

**Verification.** Block the control callback deliberately for twice the timeout and confirm the watchdog trips within the specified latency.

### CI-021 — SLAM that locks onto the wrong aisle

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** After a recovery, the robot localises confidently into an aisle that is geometrically similar to the correct one. Position is self-consistent and wrong.

**Competing causes.** (1) Perceptual aliasing: repeating rack geometry makes two locations indistinguishable to a scan matcher. (2) The recovery path re-initialised from a saved pose that was itself wrong. (3) The map contains a duplicated region from a merge, so both locations are "correct" in map coordinates.

**Discriminating check.** Compare the top two match candidates' scores. A correct localisation has a dominant best match; aliasing shows two candidates within the matcher's noise. Then check the map for duplicated geometry by correlating the two regions: a high correlation with a consistent offset is a map duplication, not an aliasing event.

**Correction.** Require a disambiguating observation before accepting a relocalisation — a unique fiducial, a known asymmetric landmark, or a commanded motion whose observation distinguishes the candidates.

**Prevention.** Design maps and environments so that recovery locations are distinguishable, and audit maps for duplicated geometry at build time.

**Verification.** Force a recovery at each aliased location and confirm the system either resolves correctly or refuses to move, rather than resolving confidently and wrongly.

### CI-022 — a path that is feasible for a footprint nobody is carrying

**Category:** CONCEPT_CONFUSION.

**Symptom.** A planner produces a collision-free path that the loaded robot collides with. The path is genuinely collision-free against the model the planner used.

**Competing causes.** (1) The planner uses the chassis footprint and ignores the load's overhang. (2) The footprint is correct in plan view but the obstacle exists only at the load's height, and the costmap is built from a sensor mounted below that height. (3) The clearance margin is applied to the path centreline rather than to the swept volume at the maximum curvature.

**Discriminating check.** Compare the planner's clearance at the tightest point against the swept volume measured on the real machine at the relevant height band. A path whose plan-view clearance is 0.20 m but whose collision occurs at 0.9 m height is a sensor-coverage failure, not a footprint error. Measure both; they are different fixes.

**Correction.** Model the loaded swept volume including height, and build the costmap from a sensor configuration that observes the same height band as the load.

**Prevention.** Footprint and sensor coverage are one artefact: a footprint that is taller than the sensor's coverage is a blind volume by construction.

**Verification.** Drive the tightest planned path with the maximum declared load and a deliberately placed test obstacle at load height, and confirm the robot refuses before contact.

### CI-023 — a high grasp score and a dropped part

**Category:** NUMERICAL_EDGE.

**Symptom.** A learned grasp scorer predicts success with high confidence and the part slips during transport. The score was calibrated on static grasps.

**Competing causes.** (1) The score predicts lift success, not transport success, and transport adds accelerations the training distribution did not include. (2) Friction varies with surface contamination and the model's inputs cannot observe it. (3) The approach pose's uncertainty is larger than the model's implicit assumption, so the achieved contact differs from the scored one.

**Discriminating check.** Stratify outcomes by transport acceleration and by the pose error measured after the grasp. A score that predicts lift but not transport shows a sharp drop beyond a modelled acceleration; a score that ignores friction shows a drop correlated with the part's surface history rather than with the pose.

**Correction.** Train and calibrate on the task's actual success criterion, including transport, or gate the score with an independent physical check such as a load-cell reading.

**Prevention.** The label must be the outcome the system needs, and the training distribution must include the dynamic phase.

**Verification.** Confirm the score's calibration curve holds in the transport regime, not only at lift.

### CI-024 — a language-conditioned policy that takes a forbidden action

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A vision-language-action policy receives a text instruction that is syntactically valid and semantically plausible, and executes an action the site forbids. The policy did what it was asked.

**Competing causes.** (1) The action gate validates the instruction rather than the action, so a benign instruction with a hazardous consequence passes. (2) The gate is inside the policy's own process, so the policy can bypass it. (3) The forbidden action is not in the gate's enumeration, because the enumeration was written before the policy existed.

**Discriminating check.** Send a safe-sounding instruction whose only admissible action is forbidden, and observe whether the gate blocks it. Then check the gate's process boundary: a gate that shares memory with the policy is not a gate. Finally, diff the gate's action enumeration against the policy's action space; a size mismatch is the mechanism.

**Correction.** Gate actions, not instructions, at a boundary the policy cannot cross, and derive the gate's enumeration from the policy's declared action space.

**Prevention.** Any policy with an open instruction channel needs a closed action channel, and the closure must be enforced by a separate process.

**Verification.** Attempt every action in the declared space against the gate and confirm each is either permitted with a reason or refused, with no unclassified path.

### CI-025 — a retry that repeats a side effect

**Category:** SCALE_FAILURE.

**Symptom.** A task is retried after a transient failure and the physical action occurs twice. The behaviour tree reports one success and the world shows two.

**Competing causes.** (1) The action has no identity, so the retry cannot be recognised as a repeat of the same request. (2) The action completed but the acknowledgement was lost, so the retry is a duplicate rather than a fresh attempt. (3) A behaviour-tree tick re-enters a node whose state was reset, re-issuing an action the world already performed.

**Discriminating check.** Log an action identity with every request and every completion, and count distinct identities against physical occurrences. Two occurrences with one identity is duplication; two identities is a planning error. Then check whether the completion was recorded before the acknowledgement was lost, which distinguishes (2) from (1).

**Correction.** Give every action an idempotency key that the effector honours, and make the effector's completion record durable before the acknowledgement is sent.

**Prevention.** Retry policy and idempotency are one design; a retry without a key is a duplication with extra steps.

**Verification.** Inject an acknowledgement loss and confirm the physical occurrence count is one.

### CI-026 — synthetic validation that is perfect because it leaks

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** A model validated on synthetic data achieves near-perfect scores and degrades immediately in the field. The validation split was random, and the synthetic data has latent identities that cross the split.

**Competing causes.** (1) The same scene, trajectory, or asset instance appears in both train and validation with different noise. (2) The label generator and the model share a code path, so the model learns the generator's quirks. (3) Domain randomisation ranges were fitted to the field data that was then used for evaluation.

**Discriminating check.** Group the holdout by latent source — scene identifier, asset instance, trajectory seed — and recompute. A large drop identifies leakage; no drop makes the leakage explanation unlikely. Then check whether the label generator's code is reachable from the model's input pipeline.

**Correction.** Partition by latent identity, not by sample, and generate the holdout from an independent seed block.

**Prevention.** The split must be defined by a key that is not a feature, and the key must be recorded in every episode's metadata.

**Verification.** Re-run with an independent generator version and confirm the score's drop is consistent with the leakage estimate rather than larger.

### CI-027 — a simulator that is stable where the hardware is not

**Category:** NUMERICAL_EDGE.

**Symptom.** A controller tuned in simulation is unstable on hardware. The simulation is deterministic and reproducible; the hardware is not.

**Competing causes.** (1) The integration step is too large for the controller's bandwidth, so the simulation's discrete dynamics are more stable than the continuous plant. (2) The simulator models the actuator as an ideal torque source with no delay, while the hardware has 8 ms of command latency. (3) Contact is modelled as a stiff spring whose stiffness, not the controller, sets the observed behaviour.

**Discriminating check.** Halve the integration step and re-run the same controller. If the behaviour changes materially, the step is inside the dynamics and the result is not converged. Then add the measured actuator delay and re-run; if the instability appears, the delay was the missing term.

**Correction.** Choose the step from a convergence study rather than from a real-time budget, and model latency explicitly because it is the term that most often decides stability.

**Prevention.** A simulator result is a hypothesis about the plant until a convergence study and a latency measurement have both been done.

**Verification.** Confirm the step-halving changes the outcome by less than the tolerance, and that the added latency reproduces the hardware's behaviour.

### CI-028 — labels better than the task needs, and worse than it assumes

**Category:** CONCEPT_CONFUSION.

**Symptom.** A model trained on synthetic labels exceeds the accuracy of the labels' own specification, and its field errors concentrate where the label's definition of the task differs from the operator's.

**Competing causes.** (1) The synthetic label encodes a geometric definition that is precise but not the operational one — a bounding box around visible pixels versus the object's full extent. (2) The simulator's label is derived from the same state the model can observe, so it is more informative than any field label could be. (3) The label's class boundaries were defined for annotation convenience and do not match the decision boundary the application uses.

**Discriminating check.** Compare the model's disagreement with human labels specifically at cases where the two definitions differ — partial occlusion, truncation at the frame edge. Disagreement concentrated there is definitional, not a model error, and the fix is the label definition rather than the training.

**Correction.** Write the operational definition as a decision rule, generate synthetic labels from that rule, and evaluate against human labels generated from the same rule.

**Prevention.** A label specification is part of the model's interface; changing it changes the task.

**Verification.** Evaluate on a human-labelled set generated from the operational definition and confirm the error is not concentrated on definitional edge cases.

### CI-029 — a fleet dashboard that reports green while a local hazard rises

**Category:** VIOLATED_ASSUMPTION.

**Symptom.** Every fleet-level rate is at or below baseline while a specific site reports a rising number of near-misses. The aggregate is arithmetically correct.

**Competing causes.** (1) The hazard is confined to a sub-population that is a small share of the denominator, so its rate change is diluted below the dashboard's threshold. (2) The aggregate uses processing time, so recent local reports have not yet arrived. (3) The local population's exposure fell, so a constant count produces a rising local rate that the fleet total does not see.

**Discriminating check.** Stratify the rate by site and by geometry class before pooling. A hazard confined to 7.8 percent of the fleet can double while the pooled rate moves by 8 percent, which is inside most dashboards' noise. Then recompute on event time to see whether the local rise survives the arrival lag.

**Correction.** Gate dashboards on stratified rates with per-stratum thresholds, and publish the stratum sizes so that a small stratum's change is visible rather than diluted.

**Prevention.** Any aggregate that pools heterogeneous populations must be accompanied by the strata that compose it.

**Verification.** Inject a known doubling in one small stratum and confirm the dashboard alerts.

### CI-030 — a rollback that cannot finish inside its window

**Category:** SCALE_FAILURE.

**Symptom.** A release is rolled back and the fleet spends longer in an inconsistent state than it spent in the failing state. The rollback plan was correct in sequence and unmeasured in duration.

**Competing causes.** (1) The previous artefact was not retained on the device, so the rollback requires a download over a constrained link. (2) The rollback path itself changed in the release being rolled back. (3) A staged rollout left part of the fleet on the new version, so the rollback must reconcile a mixed state rather than restore a known one.

**Discriminating check.** Time an actual rollback on one device of each hardware variant, from initiation to verified old-version operation. A measured time longer than the release cadence means the fleet can never be fully consistent. Then check whether the rollback artefact is present locally; absence is the dominant cost in most plans.

**Correction.** Retain the previous artefact on device with a verified boot path, and treat the rollback's measured duration as a release gate.

**Prevention.** The rollback is part of the release, not a recovery measure, and it should be exercised on a schedule rather than discovered during an incident.

**Verification.** Perform a rollback on a deliberately mixed cohort and confirm every device reaches a verified old version within the stated window.

The catalogue's organising principle is that every entry names a check that could fail. An entry whose check cannot fail is a belief, not a diagnosis, and it belongs in a training slide rather than in an incident review.

Two cross-cutting observations follow from reading all thirty in sequence.

The first is that the majority of entries are metadata failures rather than algorithmic ones. Sequence counters, sequence numbers, timestamp meanings, calibration provenance, quantisation parameters, frame names, action identities, and version manifests are all metadata. A team that instruments metadata first finds most of these issues by inspection, before any of them reaches a customer.

The second is that the entries cluster at ownership boundaries. Every failing interface in this file has a producer and a consumer with different owners, and the failure mode is consistently that the consumer assumed something the producer never promised — and never documented. That is an organisational defect expressed as a technical one, and no amount of model improvement addresses it.

### 8.2 Five worked diagnoses, abbreviated

Each of these compresses one catalogue entry into the sequence a reviewer would actually follow, because the format of the entries above is deliberately uniform and real diagnosis is not.

**A robot that reverses into a load it did not see.** The symptom is a collision during a manoeuvre, and the first instinct is a perception failure. The check that settles it: does the reverse sensor's coverage overlap the chassis footprint at the load's height? If the reverse path is covered but the overhang is not, it is a coverage defect, not a detection defect, and the fix is a footprint declaration rather than a model. The abbreviated sequence is: compute the swept volume at load height, subtract the sensor's observed region, and see whether the difference is where the collision happened.

**A pick-and-place cell that misses one part in two hundred.** The cell's aggregate success is 99.5 percent and the customer's tolerance is 99.9. The check is to stratify by part supplier, because surface finish is a latent variable the model cannot observe. If one supplier's parts carry most of the failures, the mechanism is friction and the fix is either a supplier specification or a physical grip check — not a retrain on the same distribution.

**A fleet that reports normal battery health while service calls rise.** The health metric averages across the fleet, and a failing batch of packs is a small share. The check is to stratify by pack lot; a lot-level defect that affects 3 percent of packs is invisible in a fleet mean and obvious in a per-lot distribution. The general form of this error is CI-029 and it recurs in every fleet metric without exception.

**A simulation that passes every scenario and a robot that fails on the third shift.** The latent variable is thermal: the simulator does not model the enclosure's temperature rise, so the third shift runs 12 °C hotter than the first. The check is to log the compute module's temperature against the failure time; a failure rate that rises with shift number and tracks temperature identifies the mechanism in one week of data.

**A model that improved after a data pipeline change and then regressed.** The improvement was in the pipeline's missing-value handling, which had been imputing a value the model learned to rely on. The check is to ablate the imputation and see whether the model degrades to the old performance or to worse; worse means the model learned the imputation's artefact, and the fix is to make missingness explicit as an input rather than imputed.

The common structure across all five is that the diagnosis is a comparison between a quantity and its counterpart — observed region against swept volume, part strata against each other, lot strata against each other, temperature against failure time, ablated pipeline against the original. The reviewer who can name both sides of the comparison is most of the way to the answer.

### 8.3 Entries that are commonly confused with each other

Three pairs in the catalogue produce the same first symptom and are routinely misdiagnosed as each other.

CI-001 and CI-015 both present as bad timing behaviour. CI-001 is corruption within a block, so the values are wrong in a way the continuity counter catches. CI-015 is correct values that are old, so the values are right and the age is wrong. The check that separates them is whether the payload's own internal consistency holds; if it does, the block is intact and the problem is age.

CI-007 and CI-009 both present as a deployment that disappoints. CI-007 is a performance disappointment with correct output; CI-009 is a memory failure with correct output until it fails. The check is whether the failure is gradual or abrupt: performance that is merely insufficient is CI-007, and an allocation failure after a warm start is CI-009.

CI-013 and CI-023 both present as overconfident scores. CI-013 is overconfidence in a continuous estimate, which shows in the covariance against the residuals. CI-023 is overconfidence in a classification of an outcome the model never saw, which shows in the calibration curve against the transport regime rather than the lift regime. The check is whether the score's calibration holds within the task that actually matters.


## 9. Complexity & Cost Analysis

The cost of these issues is dominated by the diagnosis, not the correction. In the illustrative cases above, the corrections are one-parameter changes, artefact retentions, or metadata additions. What consumes engineering time is the search, and the search is expensive in exactly the proportion that the team lacks a discriminating check.

A rough accounting, at illustrative loaded rates of EUR 95 per engineering hour.

A symptom with a discriminating check costs about 6 hours: 2 to compute which observation separates the candidates, 2 to collect it, 1 to confirm, 1 to write the correction. A symptom without one costs an unbounded multiple, because the team iterates over hypotheses by changing things, and each change is a day of build, deploy, and observe. The observed ratio in the cases above is roughly one to eight, which is the strongest argument in this file for spending the two hours on the check.

The second cost term is containment. Every hour a zone runs at reduced throughput costs its contribution margin; the worked example's cell at EUR 25,446 per day costs EUR 1,060 per hour of containment. That cost is incurred whether or not the diagnosis is progressing, which gives a strong incentive to instrument before the incident rather than during it.

The third term is the cost of the wrong fix. A fix applied without a discriminating check has roughly a one-in-three chance of being irrelevant to the actual mechanism when three causes are plausible, and an irrelevant fix has a specific additional cost: it consumes the organisation's belief that the problem is solved, so the next occurrence is investigated less urgently than the first.

| Instrument | Build cost | Per-use cost | Break-even against |
|---|---|---|---|
| Sequence counter in the record | 1 h | 0 | One torn-read investigation (6 h) |
| Clock sync audit with offsets logged | 4 h | Minutes | One age-versus-latency confusion (8 h) |
| Commanded-versus-realised join | 12 h | Minutes | One enforcement-gap investigation (16 h) |
| Candidate-versus-filed ratio dashboard | 30 h | 0 | One ascertainment ambiguity (40 h plus decision delay) |
| Null-model latency baseline | 3 h | Minutes | One compression effort that does not pay (40 h) |
| Latent-identity partition key | 6 h | 0 | One leaked holdout (60 h plus a bad release) |
| Rollback timing exercise | 8 h per variant | Hours | One rollback that cannot finish (unbounded) |

The break-even column is the useful one, and it should be read as an argument to build the instrument before the incident rather than after. Every row breaks even on a single occurrence, and every row's cost is small relative to the engineering hours already committed to the system it protects.

There is a fourth cost that does not appear in any table and is usually the largest: the cost of the decision that was made on an unvalidated count. In the incident case in this unit, the gate's numerator depended on a filing rate that took five days to obtain, and the gate was designed before that rate was known. The decision was still defensible, but its defensibility rested on an assumption rather than an instrument, and the difference between those two is the entire subject of this sheet.

### 9.1 A worked containment comparison

Two containment options for the same hazard illustrate how the cost analysis should be structured, because the naive comparison misleads.

Option A is a speed reduction on the affected machines: no capital, a configuration change, and a throughput cost proportional to the time added per traversal. At 9.33 s added per affected traversal on 416 traversals per day, the robot-time cost is 3,881 robot-seconds per day, or 1.078 robot-hours, which against a zone capacity of 828 robot-hours per day is 0.13 percent.

Option B is a human observer at the hazard: 18 staffed hours per day at EUR 41 per hour, or EUR 738 per day, plus the training and turnover costs that observer pools carry.

The ratio is about 39 to 1 in favour of the speed reduction on direct cost. That ratio is not the decision, and a review that stops there has made the classic error. Three further questions decide it.

The first is whether the speed reduction actually addresses the mechanism. If the hazard is detection latency and the reduction does not change latency, the throughput cost buys nothing and the correct comparison is between doing nothing and the observer. This is why the mechanism must be computed before the control is chosen.

The second is what each option does to the evidence. The observer suppresses the very events the gate would count, because a human in the hazard zone changes both the robot's behaviour and the pedestrians'. The speed reduction preserves the measurement. An option that costs less and destroys the measurement is not cheaper; it is a different problem.

The third is the cost of being wrong. If the speed reduction is insufficient, the hazard continues at a reduced rate until the next gate reads out, which is a bounded exposure if the gate is well designed. If the observer is withdrawn early, the hazard returns immediately at full rate. The asymmetry favours the option with the gate, provided the gate has power to fail — which is a computation, not a hope.

### 9.2 The cost of the gate itself

Gates are not free, and the cost is often omitted from the comparison that selects the containment.

A gate costs the monitoring engineering, the analysis time at each readout, and the decision latency when it fires. For a 23-day gate with a single readout, that is perhaps 24 engineering hours, or EUR 2,280 — comparable to the containment it is monitoring. A gate with weekly interim looks costs four times that, and the interim looks also raise the false-alarm rate by a factor of about four if each look is tested at the same significance, which is the multiple-comparisons problem in its most practical form.

The design consequence is that interim looks should be non-binding unless they cross a threshold that is deliberately conservative. A look at day 10 that would stop the trial on any event is not an interim look; it is a different experiment with a different error rate.

### 9.3 What not to cost

Three quantities are commonly costed and should not be.

The first is the notional value of an avoided injury. Any figure assigned to it dominates every other term in the comparison, which makes the comparison uninformative: whichever option has the smaller injury probability wins by an arbitrary multiple. The honest treatment is to constrain the decision on a hazard threshold and compare cost only among options that pass it.

The second is engineering morale or reputational cost. These are real but they are not commensurable with the other terms and they are not improved by being given a number.

The third is the cost of the investigation itself when it is already sunk. An investigation that has consumed 200 hours does not become more valuable because of that; the remaining decision should be made on the remaining evidence.

## 10. Key Theorems / Results Reference

**Theorem 1 — age is not latency.** For any chain, end-to-end age equals the sum of stage ages, and the minimum achievable age under a non-empty queue is the queue's occupancy divided by the service rate plus the service time. A system can have arbitrarily small latency and arbitrarily large age; the two are only equal when no queue ever forms. Nothing in the statement requires the chain to be linear, but it does require every stage to be work-conserving for the equality.

**Theorem 2 — a queue is unstable exactly when the arrival rate exceeds the service rate.** For a work-conserving queue with independent arrivals of mean rate λ and service of mean rate μ, the backlog grows without bound for λ > μ and is stable for λ < μ. The result is distribution-free in its stability claim, which is why the conclusion survives the fact that real sensor traffic is bursty; what the distribution changes is the backlog's variance, not its divergence.

**Theorem 3 — fusing correlated estimates yields less than the independence assumption promises.** For two unbiased estimates with variances v1 and v2 and correlation ρ, the minimum-variance combination's variance is bounded below by the larger single-input contribution when ρ = 1 and equals the inverse-variance combination only when ρ = 0. At ρ = 0.9 with equal variances, the achieved variance is about 1.9 times the naive prediction. The practical corollary: a covariance that omits a shared error source is wrong by the ratio of that source's variance to the total.

**Theorem 4 — the exact Poisson upper bound at zero events is 3/T.** With zero observed events in exposure T, the one-sided 95 percent upper bound on the rate is 3.00/T, arising from χ²(0.95, 2)/2. At one event the factor is 4.744, at two 6.296. The corollary is the one that catches gate designers: to certify a rate below L you need at least 3/L of exposure if you expect zero events, and 4.744/L if you are willing to tolerate one. A gate that cannot reach 3/L cannot pass at all.

**Theorem 5 — a hash of b bits collides with probability above one-half once about 1.18 × 2^(b/2) items are stored.** For b = 16 that is 301 items; for b = 32 it is 77,163; for a 64-bit truncated digest the expected collision count at 10⁶ items is about 2.7 × 10⁻⁸, which is why short hashes are unsafe as release identifiers at fleet scale and long ones are fine.

**Theorem 6 — a swept volume is not a footprint.** The set of points occupied by a rigid body over a trajectory is the union of its transformed shapes, whose volume is at least that of the largest single pose and can exceed it without bound as curvature grows. A footprint-based clearance check is therefore an approximation whose error grows with path curvature and with any load overhang, and it is exact only for straight-line motion of a body whose shape encloses the load at every height.

**Theorem 7 — a control loop's stability degrades with added delay in a way that no gain reduction fully compensates.** For a first-order plant with proportional control, the phase margin falls by ωc × Δt radians for added delay Δt at crossover frequency ωc. Reducing gain lowers ωc and restores margin, at the cost of bandwidth; this is why latency reduction and gain reduction are not interchangeable and why a delay that appears only under load shows up as an oscillation that no tuning removes.

**Theorem 8 — an idempotency key converts a retry from duplication to repetition.** If an effector records each key's completion durably, then any number of retries of the same key produce at most one physical effect, provided the record is written before the acknowledgement. Without the key, no retry policy can distinguish a lost acknowledgement from a lost request, and the two require opposite actions.

### 10.1 How these results are used, and how they are misused

Each theorem in this section is a compression of a longer argument, and each has a validity condition that the compression drops. The conditions matter more than the statements in practice, because the statements are memorable and the conditions are not.

Theorem 2 is the most abused. Its statement — a queue is unstable exactly when arrival exceeds service — is distribution-free, which people read as "robust". It is robust in its stability claim and useless for its transient claim: bursty arrivals at a mean rate below service will still build queues large enough to break a deadline, and the theorem says nothing about that. Anyone using it to argue that ρ < 1 is safe is quoting the stability condition as if it were a latency guarantee.

Theorem 4 is the most often mis-stated in the other direction. The zero-event bound of 3/T is a one-sided 95 percent ceiling, and it is frequently quoted as if it were a two-sided interval or as if it applied to any number of events. At one event the factor is 4.744, not 3, and a gate designed on the zero-event factor will fail unexpectedly at the first event and be read as a regression rather than as the arithmetic it is.

Theorem 7 is the one engineers most want to believe is false. Adding delay to a loop does degrade phase margin by ωc·Δt, and reducing gain does restore it — so a tuned-down controller appears to solve a latency problem. What the appearance hides is the bandwidth loss: the loop still responds, more slowly, and the delay is still there, so the next latency increase crosses the margin again at a lower gain, and the sequence of retunings walks the system into a sluggish configuration nobody chose. The correct response to a latency that appears under load is to remove the load-dependent delay, not to retune.

Theorem 6 is misused as a reason to enlarge footprints. A swept volume that accounts for curvature can be large enough to make a route infeasible, so the temptation is to keep the plan-view footprint and add a margin. That works for straight paths and fails at the curvature where the swept volume actually diverges — which is the case the footprint was meant to cover.

Theorem 1 is misused least, which is a symptom of its being the one that is easiest to compute and hardest to argue with. If a team measures age and shows it exceeding the deadline, no amount of latency improvement changes the conversation.

### 10.2 A note on what is not covered

The results above are the ones that survive contact with the catalogue entries. Several results that would be natural to include are omitted because they require assumptions this sheet cannot state cleanly.

No result is given for multi-robot collision avoidance, because the correctness properties depend on assumptions about communication and synchronisation that vary by architecture and are not compressible into a theorem without a page of preamble.

No result is given for learning-based control stability, because the field's results depend on smoothness assumptions that deployed policies routinely violate, and quoting them would create a false impression of assurance.

No result is given for the safety-integrity-level arithmetic of a protective function, because that arithmetic depends on the applicable standard, the architecture category, and the diagnostic coverage claimed — all of which are determinations for a competent body, not for a cheat sheet.

The absence of these is deliberate and is part of the sheet's claim: it covers what can be derived and checked, and it names what cannot be.

A reader who wants a single rule for using this section should take the following. Quote a theorem only with its validity condition attached, because the condition is what a reviewer will test and the statement is what a vendor will quote. Where the condition cannot be established from available data — a correlation coefficient nobody measured, a delay nobody profiled — say so and fall back to the conservative bound the catalogue's checks provide, rather than to the theorem's optimistic form.

## 11. Interview / Exam Questions

Twelve questions, each answered. At least half require a computation, a selection, or a diagnosis rather than recall. The answers reference the catalogue where the underlying mechanism is developed.

**Q1. A trace shows a fused position covariance of 0.0707 m while repeated static trials scatter with a standard deviation of 0.10 m. Diagnose and quantify.**
Answer: The filter is treating two inputs with correlation about 0.9 as independent. With equal variances of 0.01 m², the naive combination gives 0.005 m², or 0.0707 m, while the true variance is (0.02 − 0.018)/0.19 = 0.0105 m², or 0.1026 m. The filter is 45 percent overconfident.
Reasoning: The empirical scatter is the measurement; the covariance is the estimate. When they disagree by a factor of two in variance, an omitted correlation term is the most common cause, and the discriminating check is the innovation statistic's mean, which will sit far above its expected value.
Follow-up: If you add a third sensor sharing the same calibration, does the inflation follow the same scalar formula? No — the general case is a matrix inversion, and the scalar formula understates the inflation for n > 2.
Red flag: Increasing process noise until the covariance matches the scatter. That hides the shared source and destroys the filter's transient response.
Scoring: 3 points for the mechanism and the numeric inflation; 2 for the mechanism alone; 1 for "the covariance is too small".

**Q2. Compute the age at actuation for a chain with 5 ms capture, 30 ms queue, 4 ms transport, and 18 ms scheduling against an 80 ms deadline. What would you optimise first, and why?**
Answer: 57 ms, 71.25 percent of the deadline. The queue is 30 ms, more than half the total, so the queue's policy is the first target — specifically, switching from a FIFO that drops newest to a bounded queue that drops oldest.
Reasoning: Optimising the compute stage, if it is not in the list, buys nothing; optimising the largest term buys up to 30 ms if the queue can be made near-zero, which is 37 percent of the deadline recovered with a policy change.
Follow-up: What does the age distribution look like if the queue is bounded at four frames at 25 Hz? A fixed 160 ms worst case, which exceeds the deadline — so the bound must be set from the deadline, not from memory.
Red flag: Optimising inference because it is the stage the team owns.
Scoring: 3 points for the arithmetic, the identification of the queue, and the policy recommendation; 2 for the arithmetic and the identification; 1 for the arithmetic.

**Q3. A robot's required stopping envelope at 1.2 m/s is 2.655 m and the measured clearance is 1.90 m. A colleague proposes reducing the speed to 0.9 m/s. Check whether that suffices.**
Answer: At 0.9 m/s the envelope is 0.135 + 0.405 + 1.365 = 1.905 m, which still marginally exceeds 1.90 m. The proposal is right in direction and insufficient in magnitude; the largest speed that fits 1.90 m is about 0.896 m/s, so 0.9 is over by roughly 5 mm of required distance.
Reasoning: The envelope is not proportional to speed: the braking term scales with v² and the hazard term includes v/a, so a 25 percent speed reduction yields a 28 percent envelope reduction, and the residual is decided by the third decimal.
Follow-up: Why is 1.90 m the wrong number to fit anyway? Because clearance must be measured to the nearest obstacle at the load's height band, and the sensor that measured 1.90 m is mounted below it.
Red flag: Halving the speed and declaring the problem solved without recomputing.
Scoring: 3 points for the envelope at 0.9, the comparison, and the height caveat; 2 for the envelope and comparison; 1 for a qualitative answer.

**Q4. A 12-bit converter with a 3.3 V reference reads 2810 counts against a 2048-count mid-scale, and the resulting 0.614 V implies an impossible force. Enumerate the causes and select the check.**
Answer: Three candidates: a wrong bridge sensitivity constant, a gain applied before the offset, and a reference-voltage mismatch between calibration and conversion. The discriminating check is a three-point calibration — zero, mid-scale, near full scale — because an offset error moves all three points equally, a gain error moves them proportionally, and a reference mismatch scales all three by the supply ratio.
Reasoning: The reading is not noise; it is wrong by a factor that a single-point check cannot characterise. One LSB is 3.3/4096 = 0.806 mV, so a 0.614 V reading is 762 LSB from mid-scale, which no plausible noise process produces.
Follow-up: What would a 0.4 mV RMS noise floor against a 0.806 mV step tell you? That the converter is dithering on its own quantisation, so consecutive static readings alternate between codes and any averaging filter lands on a boundary.
Red flag: Adding a digital filter to a grossly mis-scaled reading.
Scoring: 3 points for three causes, the three-point check, and the LSB arithmetic; 2 for two causes and the check; 1 for one cause.

**Q5. A 240-robot fleet runs a 4-minute release per robot. Compute the exposure and say whether a single 30-minute observation can detect a 0.5 percentage-point regression in mission success.**
Answer: 240 × 4 minutes is 960 robot-minutes, or 16 robot-hours — about 240 releases if each is 4 minutes, not 240 hours. To detect a 0.5 pp change at a base rate of 0.95 with 95 percent confidence and 80 percent power requires roughly 7,299 samples per arm, so a 240-sample observation cannot detect it; the minimum detectable difference is about 2.8 pp.
Reasoning: n ≈ 2 × (1.96 + 0.84)² × p(1−p)/δ², and at p = 0.95 the numerator is 15.68 × 0.0475 = 0.745, so δ² = 0.745/240 gives δ ≈ 5.6 pp for a one-arm comparison at that power.
Follow-up: What would you change to make the test informative? Either accumulate several release waves, or gate on a paired pre/post difference within the same robot to remove between-robot variance.
Red flag: Judging the release from one wave's aggregate.
Scoring: 3 points for the sample-size arithmetic and the honest conclusion; 2 for the sample-size formula without the arithmetic; 1 for "the sample is too small".

**Q6. A watchdog fires only under load. What are the candidate mechanisms, and which single observation separates them?**
Answer: Timer callback delayed by executor occupancy; watchdog fed by the thread it monitors; kick delayed behind a lock. The separating observation is the watchdog callback's start delay as a function of occupancy — if the delay tracks occupancy, the timer is being scheduled late rather than being wrong, and the fix is an independent execution context.
Reasoning: A clock error would fire at the wrong absolute time regardless of load. Load dependence identifies scheduling, and scheduling dependence identifies a shared context, which is exactly what a watchdog must not have.
Follow-up: How do you prove independence rather than assert it? Block the monitored path for twice the timeout and confirm the trip latency.
Red flag: Increasing the timeout so the false trips stop.
Scoring: 2 points for the mechanism and the observation; 1 for the mechanism alone.

**Q7. A model with 40 percent fewer parameters shows no latency improvement. Give three mechanisms and the measurement that identifies each.**
Answer: No kernel for the compressed structure (measure achieved FLOPS and check whether the sparse path executes); memory-bandwidth bound (measure bytes moved against peak bandwidth); fixed per-inference overhead (measure a null model of the same shapes).
Reasoning: Each mechanism predicts a different signature: a fallback predicts dense execution despite the smaller parameter count; bandwidth binding predicts high bandwidth utilisation with low arithmetic utilisation; overhead predicts a null model costing a large fraction of the full model.
Follow-up: Which is most common in practice at small batch sizes? Overhead, which is why fusing pre- and post-processing often beats compressing the model.
Red flag: Compressing further when the bottleneck was never arithmetic.
Scoring: 3 points for three mechanisms with matching measurements; 2 for two; 1 for one.

**Q8. You must certify a near-miss rate below 0.5 per 1,000 traversals. How much exposure do you need, and what does a 14-day window at 416 traversals per day certify?**
Answer: 3/L = 6,000 traversals for zero events, 4.744/L = 9,488 for one. A 14-day window gives 5,821 traversals, whose zero-event bound is 3/5,821 × 1,000 = 0.515 — above 0.5, so it certifies nothing even with a perfect record.
Reasoning: The bound is exact rather than approximate, and at zero and one events the normal approximation is meaningless, so the arithmetic must use the χ² factors directly.
Follow-up: What is the shortest window that can pass with one event? 9,488/416 = 22.8 days, which rounds to 23.
Red flag: Scheduling the window by convenience and calling the result evidence.
Scoring: 3 points for both exposure figures and the 14-day contradiction; 2 for one figure; 1 for the rule of three alone.

**Q9. A planner's path is collision-free against the chassis and the loaded robot hits a pallet at 0.9 m. Is this a footprint error or a sensor-coverage error, and how do you tell?**
Answer: It is both, and the check distinguishes them: measure the plan-view clearance at the tightest point. If plan-view clearance was adequate and the collision occurred above the lidar's height, it is sensor coverage; if plan-view clearance was inadequate against the load's overhang, it is footprint. The fixes differ: raise or add sensing versus model the loaded swept volume.
Reasoning: A footprint model that is correct in plan view but shorter than the load creates a blind volume between the sensor's plane and the load's top, and no planner can avoid what the costmap does not contain.
Follow-up: What single artefact prevents the recurrence? A footprint declaration that includes a height band, checked against the sensor configuration's coverage at build time.
Red flag: Adding a clearance margin to the planner, which cannot help with an obstacle that is absent from the map.
Scoring: 3 points for the distinction and the matching fixes; 2 for the distinction; 1 for one cause.

**Q10. A fused heading drifts during turns but is stable at rest. Discriminate between a time offset, a scale factor, and a lever arm.**
Answer: Rotate at two rates in both directions. A residual proportional to rate with a sign that changes with direction is a scale factor; a residual proportional to rate with the same sign in both directions is a time offset (10 ms at 90 °/s is 0.9°); a residual present only during angular acceleration is a lever arm (15 cm at 1 rad/s is 0.15 m/s of tangential velocity).
Reasoning: The three mechanisms have different parity under time reversal, which is why a bidirectional test separates all three with one experiment.
Follow-up: Which mechanism does a turntable calibration NOT fix? The lever arm, because it depends on the mounting geometry rather than on the sensor's scale.
Red flag: Adding a low-pass filter, which suppresses the symptom during turns and leaves the heading error.
Scoring: 3 points for all three signatures and the bidirectional test; 2 for two signatures; 1 for one.

**Q11. A dashboard is green while one site's near-miss rate has doubled. Give three mechanisms and say which stratification exposes each.**
Answer: Small-stratum dilution (stratify by site and geometry); processing-time lag (recompute on event time); falling local exposure (stratify by robot-hours rather than by count). The first is arithmetic: a stratum carrying 7.8 percent of exposure can double while the pooled rate moves 8 percent, inside most noise bands.
Reasoning: Aggregates are weighted averages, and a weighted average cannot show an unweighted change in a small weight. The dilution is not an error in the aggregate; it is a mismatch between the aggregate and the question.
Follow-up: What must accompany any pooled rate to prevent this? The composition table, so a reader can see the weights.
Red flag: Adding an alert on the pooled rate with a tighter threshold, which fires on fleet-wide noise and misses localised changes.
Scoring: 3 points for three mechanisms with matching stratifications; 2 for two; 1 for one.

**Q12. A rollback is planned and never timed. Why is that a release-blocking defect, and what measurement closes it?**
Answer: Because the rollback's duration determines whether the fleet can ever be consistent: if the measured rollback takes longer than the release cadence, a mixed state is permanent and the "rollback" is a second migration. The measurement is an actual rollback on one device of each hardware variant, from initiation to verified old-version operation, including the download if the artefact is not retained locally.
Reasoning: The dominant cost is usually the absent local artefact, which turns a rollback into a download over a constrained link; that cost is invisible in any plan that assumes the artefact is present.
Follow-up: What does a measured rollback time longer than the cadence forbid? Multi-stage rollouts without a retained previous artefact, because no stage can be reversed before the next begins.
Red flag: Treating the rollback as a recovery measure rather than as part of the release.
Scoring: 3 points for the consistency argument and the measurement; 2 for the measurement; 1 for "rollback is important".

**Q13. A 64-byte frame at 921,600 baud with 8N1 framing and a 50 µs turnaround: how many frames per second before the link is 50 percent utilised, and what does a 4-byte FIFO imply at that rate?**
Answer: One byte is 10 bit-times at 1.0851 µs per bit, so 10.851 µs; 64 bytes is 694.4 µs; plus 50 µs turnaround is 744.4 µs per frame. Fifty percent utilisation allows 1,000,000/(2 × 744.4) = 671.6 frames per second. At 672 frames per second the byte rate is 43,008 B/s, so a 4-byte FIFO fills every 93 µs — which means a polled drain must run faster than about 10.7 kHz to avoid overrun.
Reasoning: The wire-time budget must include framing and turnaround, and the FIFO sizing follows from the byte rate rather than from the burst length. A payload-only calculation would give 64 × 8 / 921,600 = 555 µs, a 34 percent understatement.
Follow-up: What changes if the framing is 8E1? Eleven bit-times per byte, so 763.9 µs of data plus 50 µs turnaround is 813.9 µs per frame, and the frame rate at 50 percent falls to 614.
Red flag: Measuring throughput on a logic analyser and concluding the link is healthy because bits are leaving at the configured rate.
Scoring: 3 points for the wire time, the frame rate, and the FIFO implication; 2 for two of the three; 1 for one.

**Q14. A release manifest identifies artefacts by a 16-bit hash. At what fleet size does that become unsafe, and what replaces it?**
Answer: A 16-bit hash collides with probability above one-half at about 1.1774 × 2⁸ ≈ 301 distinct artefacts, and reaches a 1 percent collision probability at about 36 items. A fleet with more than a few dozen versioned artefacts cannot use a 16-bit identifier. Replace it with at least 64 bits of a cryptographic digest, whose 50 percent collision point is about 5.1 × 10⁹ items.
Reasoning: Collision probability for n items in b bits is approximately n²/2^(b+1), so the safe n scales as 2^(b/2). Halving the hash length quadruples the collision probability at fixed n.
Follow-up: Does a longer hash solve the problem completely? No — it solves accidental collision, not substitution. Identifiers must be bound to content by signature, or a short identifier can be attached to the wrong artefact without any collision occurring.
Red flag: Using a monotonic build number as an identity and assuming it is unique across build systems.
Scoring: 3 points for the birthday arithmetic and the replacement; 2 for the arithmetic; 1 for "use a longer hash".

## 12. Quick Reference Card

| Field | Value |
|---|---|
| Purpose | Turn a diffuse symptom into one discriminating observation and a falsifiable correction |
| First move | Contain physical risk, then find the earliest violated contract — not the loudest component |
| Most common defect | Metadata weaker than the consumer assumes: units, timestamps, frames, versions, identities |
| Age rule | age = capture + queue + transport + scheduling; a fast compute stage says nothing about age |
| Queue rule | Unstable exactly when arrival rate exceeds service rate; bounded drop-oldest fixes age, not throughput |
| Fusion rule | Correlated inputs deliver about half the promised variance reduction at ρ ≈ 0.9; omitted shared error = overconfidence |
| Envelope rule | Required stopping distance includes reaction and hazard travel, not just braking; 2.655 m at 1.2 m/s versus 0.72 m braking alone |
| Rare-event rule | Zero-event 95 percent bound is 3/T; one-event is 4.744/T; a gate that cannot reach 3/L cannot pass |
| Ascertainment rule | Counts change when the recording instrument changes; the filed-to-candidate ratio separates the two |
| Enforcement rule | Test that the constraint is refused at the code path that acts, not that the policy is published |
| Catalogue | 30 issues, CI-001 to CI-030, each with competing causes and a discriminating check |
| Cost rule | Diagnosis dominates: a checked symptom costs about 6 hours, an unchecked one about eight times that |
