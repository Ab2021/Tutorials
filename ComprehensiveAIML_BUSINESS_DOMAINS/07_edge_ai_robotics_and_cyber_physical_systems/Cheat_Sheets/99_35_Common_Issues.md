---
collection: "part07-edge-robotics"
file_id: "P07-CS-99"
topic: "35 cross-boundary edge and robotics diagnostic issues"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-08"
reviewed: null
review_trigger: "Sensor, firmware, model, clock, frame, safety envelope or release contract changes"
notation_style: "plain-text"
prerequisites: ["../INDEX.md"]
pattern_tags: ["boundary-contract", "safe-containment", "causal-diagnosis"]
source_status: "DERIVED and ILLUSTRATIVE; primary references scoped below"
owner_role: "Edge robotics integration reviewer"
source_index: "INDEX.md"
---
# 35 Common Issues Across Edge AI and Robotics Boundaries

## 1. File Header

**Purpose:** Distinguish failures that cross acquisition, inference, estimation, control and fleet boundaries.
**Outcome:** Given a symptom, choose a discriminating observation, safe containment and falsifiable correction.
**Audience:** Embedded, perception, robotics and fleet engineers conducting integration reviews.
**Prerequisites:** Read module indexes 35–41 in [part order](../INDEX.md).
**Concrete skills:** Convert units, interpret timestamped traces and distinguish estimates from measurements.
**Scope:** Thirty-five distinct teaching issues, five associated with each module.
**Evidence:** All issue scenarios are HYPOTHETICAL unless an external source claim is explicitly identified.
**Not claimed:** No incident was observed, reproduced or deployed while authoring this reference.
**Not covered:** Electrical design approval, medical interpretation or certification of a robot's safety function.
**Version:** 1.0; independent editorial and numerical review not yet recorded.
**Retrieval cue:** Start at Section 2 for localization; Section 8 contains actionable issue entries.
**Source inventory:** [directory.txt](../../../Templates/directory.txt), especially lines 1027–1128.
**Source syllabus:** [edge focused plan](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md), groups signals, compression, runtimes, firmware and compilers.
**Coverage boundary:** The visible inventory establishes topic names, not access to original lesson bodies.
**Robotics boundary:** The inventory does not supply an original ROS 2, SLAM or simulator lesson body.
**Collection relation:** Module troubleshooting files teach local mechanisms; this file diagnoses cross-interface confusion.
**Navigation:** [Collection](INDEX.md); [cases](../Case_Studies/INDEX.md); [exact ledger](../COVERAGE_LEDGER.md).

A fresh inference can still use an old observation.
A valid frame transform can still belong to another robot.
A signed update can still be incompatible with the installed calibration.
A simulator can faithfully reproduce an incorrect model.
An emergency stop's existence does not establish a safe stopping distance.
The common thread is a contract whose metadata is weaker than its consumer assumes.
Use the earliest violated contract as the diagnostic starting point.
Contain the physical risk before collecting high-volume telemetry.
Prefer a known controlled stop to an unvalidated autonomous fallback.
The appropriate safe condition is hazard-specific; it may not be power removal.
A falling load, unstable platform or thermal process can require maintained control.
No universal timeout or safe-state threshold is prescribed here.
Illustrative thresholds below are constructed to make arithmetic inspectable.

**Takeaway:** Trace identity, time, geometry, state and authority together before blaming the model.

## 2. Concept Map / Mental Model

```text
Physical process [position, temperature, vibration, physiological waveform]
  -> acquisition [channel, units, sample clock, hardware sequence]
  -> preprocessing [filter state, feature window, normalization revision]
  -> inference [model, quantization, tensor layout, target kernel]
  -> estimation [frame, covariance, observation time, association]
  -> planning [map, footprint, feasibility, policy envelope]
  -> control [deadline, command age, saturation, watchdog]
  -> physical process
Fleet plane -> signed release manifest -> each versioned local boundary
Remote operator -> bounded authority lease -> local action gate
Diagnostic plane <- sampled boundary records <- all stages
Safety-related mechanism -> hazard-specific action independent of cloud health
```

The arrows represent data and control interfaces, not proof of causality.
The diagnostic plane must not become an unbounded blocking dependency of control.
The fleet plane may distribute policy but cannot make a disconnected robot observe fresh state.
The local action gate checks validity at use time, not only at receipt time.
The physical process closes the loop; offline accuracy evaluates only one component.

| Symptom family | First boundary | Distinguishing question | Issue IDs |
|---|---|---|---|
| Missing, duplicated or stale samples | acquisition to consumer | Did hardware sequence advance without a complete immutable block? | 001–005 |
| Correct host output, wrong target behavior | model to runtime | Did semantics, support or memory change? | 006–010 |
| Plausible but inconsistent estimates | sensor to estimator | Are time, frame and correlation assumptions valid? | 011–015 |
| Jitter, motion discontinuity or oscillation | graph to controller | Is state coherent and fresh at the actuator? | 016–020 |
| Navigation or action success fails physically | perception to policy | Is the claimed confidence grounded in current geometry? | 021–025 |
| Excellent synthetic evaluation, weak transfer | simulator to field | Which real mechanism is absent or misidentified? | 026–030 |
| Fleet looks healthy despite local danger | operations to robot | Who owns authority, evidence and rollback? | 031–035 |

Read across the chain when a local test passes but the system fails.
A changed frame may invalidate covariance and map association simultaneously.
A release can be byte-correct yet change preprocessing and brake timing together.
A symptom can therefore have several contributing mechanisms.
Do not force a single root cause when the evidence supports interacting faults.
A negative diagnostic eliminates only the hypothesis its observation can distinguish.
A clean network trace does not eliminate sensor timestamp error.
A correct checksum does not eliminate a duplicated command.

**Takeaway:** Select the boundary whose assumptions are stronger than the evidence crossing it.

## 3. Foundational Definitions

| Term | Definition | Units/type | Operational nonexample |
|---|---|---|---|
| Observation time | Time at which a physical quantity was sampled | seconds in named clock | Host receive timestamp |
| Decision time | Time at which an estimate or action is chosen | seconds in named clock | Last training date |
| Information age | Decision time minus observation time after clock mapping | seconds, nonnegative if mapping is valid | Computation duration alone |
| Sequence | Monotone identifier within a stated epoch | unsigned integer plus epoch | Globally unique identity without an epoch |
| Frame | Coordinate basis and origin with authority | named rigid coordinate system | A topic namespace alone |
| Calibration | Versioned relation between measured and physical quantities | mapping plus uncertainty | Device serial number alone |
| Deadline | Latest acceptable completion relative to a release event | seconds | Mean period |
| Jitter | Variation in an explicitly defined timing quantity | seconds | One high latency without a timing definition |
| Safe condition | Hazard-specific state reducing the identified risk | state predicate | Universal power-off command |
| Authority lease | Expiring permission for a named actor to command a resource | identity, epoch, expiry | TCP connectivity |
| Correctness oracle | Independent rule for expected output on defined input | function or invariant | Repeating the same implementation |
| Evidence maturity | What was actually observed or derived | categorical label | Documentation length |

Clock mapping requires both an offset estimate and a drift/error boundary.
Unknown clock alignment makes a precise-looking age potentially meaningless.
A frame's name is an identifier; its semantics require an owner and convention.
A covariance states uncertainty under a model, not an unconditional truth certificate.
A command acknowledgement may mean accepted, queued, started or completed.
Declare which state it denotes before computing task completion.
A retry is safe only when repeated effects are prevented or explicitly acceptable.
A task can be delivered once and executed twice after state loss.
An update's cryptographic validity and its physical compatibility are independent gates.
A sample partition is not independent merely because its row numbers differ.
Synthetic and real images can share scene identity despite different rendering seeds.
An observed maximum is the largest value in one dataset.
A justified worst-case bound requires assumptions covering all permitted executions.
A percentile is a property of a population/window and a chosen estimator.
Component percentiles are not additive end-to-end guarantees.
A zero covariance can be interpreted as certainty by an estimator.
A missing covariance is not evidence of zero error.
A binary quality flag cannot preserve every reason for rejection.
Store a bounded reason code together with the validity decision.
A mitigation restores a bounded function under constraints.
A correction changes the causal mechanism that produced the failure.
A prevention control reduces recurrence or detects it earlier.
Verification is a test with expected observations and explicit limitations.

**Takeaway:** Most integration errors begin with two teams using the same word for different contracts.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Consider straight-line travel at constant speed until braking begins.
Assume a stationary obstacle, level ground and constant available deceleration after the delay.
Assume the speed remains constant during the entire delay.
This model omits brake buildup, jerk, slope, tire slip and moving obstacles.
It is a teaching lower-detail model, not a certified stopping calculation.
During the delay, distance equals speed multiplied by elapsed time.
During braking, integrating constant acceleration gives the standard energy-distance relation.
Combining the two disjoint travel intervals gives the stopping estimate.

```text
FORMULA [P07-D1]: d_stop = v * tau + v^2 / (2 * a)
WHERE: d_stop stopping distance [m]; v initial speed [m/s];
       tau observation-to-braking delay [s]; a deceleration magnitude [m/s^2].
CONSTRAINT: v>=0, tau>=0, a>0; straight-line constant-speed delay;
            constant braking deceleration; no obstacle motion or added margin.
NUMERICAL EXAMPLE: v=1.2, tau=0.15, a=0.8;
                   d_stop=1.2*0.15+1.44/1.6=0.18+0.90=1.08 m.
INTERPRETATION: a 0.95 m clearance cannot satisfy even this simplified estimate.
```

The first term has units metres per second multiplied by seconds.
The second has units square metres per square second divided by metres per square second.
Both terms are distances; adding latency directly to braking distance would be dimensionally invalid.
The result tends to zero as speed tends to zero under the model.
The braking term diverges as deceleration tends to zero at fixed positive speed.
That limit exposes why a stationary test cannot establish stopping performance.
At twice the speed, the delay term doubles and the braking term quadruples.
A material speed change therefore cannot be handled by doubling every clearance blindly.

### 4.2 General Case

Information age can contain several sequential intervals.
Use timestamps from a shared monotonic domain or a bounded clock mapping.
For one complete trace, differences telescope across adjacent boundaries.
This establishes an accounting identity, not independence between stages.
An omitted wait time appears as unexplained age, not faster computation.

```text
FORMULA [P07-T1]: tau_age = t_action - t_observe
WHERE: tau_age information age [s]; t_action actuation decision [s];
       t_observe physical observation [s], both in one mapped clock domain.
CONSTRAINT: causally paired observation/action IDs and valid clock mapping.
NUMERICAL EXAMPLE: observation=10.000 s, action=10.150 s;
                   tau_age=0.150 s=150 ms.
INTERPRETATION: a 20 ms inference kernel does not imply a 20 ms-old decision.
```

For a single trace, acquisition, buffering, transport and scheduling can explain this age.
For a distribution, quantiles of those components cannot simply be summed as the end-to-end quantile.
Collect paired traces or use an explicitly justified probability bound instead.
Clock error adds uncertainty to the age estimate.
If mapped observation time has a ±5 ms error and action time is trusted, age lies within ±5 ms.
The interval must be propagated into any hard freshness decision.
Reject an input when its entire validity claim cannot be established within the allowed age.
A future timestamp is not automatically exceptionally fresh data.
It may signal drift, epoch reset, unit conversion or clock-domain mismatch.

### 4.3 Key Special Cases / Variants

Buffer occupancy follows conservation of admitted and consumed items.
Assume fixed-size immutable blocks and one explicitly tracked consumer.
Start with an empty buffer and count each enqueue and dequeue once.
Subtracting consumed blocks from admitted blocks gives remaining occupancy.

```text
FORMULA [P07-B1]: q = n_admit - n_consume
WHERE: q occupancy [blocks]; n_admit admitted blocks [count];
       n_consume consumed blocks [count], within one reset epoch.
CONSTRAINT: no uncounted overwrites, drops, duplicates or reset crossings.
NUMERICAL EXAMPLE: n_admit=120, n_consume=113 gives q=7 blocks.
INTERPRETATION: an eight-block ring has one free slot only if its full/empty convention agrees.
```

A ring that reserves one slot for distinguishing full from empty has a smaller usable capacity.
A generation counter can remove that ambiguity but requires a correct memory-ordering contract.
Two consumers require separate acknowledgements or explicit reference ownership.
One consumer advancing its pointer cannot authorize reuse while another still reads.
A DMA producer is a hardware participant, not necessarily synchronized by a CPU mutex.
A correct software queue cannot repair a buffer overwritten before admission.
Do not use occupancy to infer sample freshness without block observation times.

A useful service metric counts complete useful outcomes, not healthy processes.
Assume a fixed eligible population and one classification per eligible event.
Partition that population into good and not-good outcomes.
The fraction in the good partition gives the useful-outcome service indicator.

```text
FORMULA [P07-S1]: SLI = n_good / n_eligible
WHERE: SLI dimensionless proportion; n_good successful fresh valid outcomes [count];
       n_eligible all eligible attempted outcomes [count] in one stated window.
CONSTRAINT: denominator>0; numerator is a subset of denominator; exclusions declared.
NUMERICAL EXAMPLE: 970 good tasks among 1000 eligible tasks gives SLI=0.97=97%.
INTERPRETATION: 995 online heartbeats cannot replace the task-level denominator.
```

[SRE's SLO workbook](https://sre.google/workbook/implementing-slos/) motivates good-events/total-events measurement.
The robotics choice of what counts as good still requires a task and safety contract.
A safe refusal may be correct safety behavior while still failing the requested service task.
Report both facts instead of hiding either behind one success percentage.
An empty population is not 100% successful; its ratio is undefined here.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

A controller can receive every message and still act on a growing backlog.
A model can fit in flash and still exceed peak live SRAM.
A pose can be globally corrected and still be discontinuous at the local controller.
A signature can authenticate malicious software produced by a compromised trusted build.
A simulation can pass every seeded episode and miss an unmodeled brake fault.
Each counterexample violates a different premise.
Replacing all five with a generic availability monitor cannot distinguish them.

| Reused rule | Required premise | Violating condition | Consequence |
|---|---|---|---|
| D1 stopping estimate | bounded delay and deceleration | load-dependent brake fade | predicted clearance insufficient |
| T1 information age | mapped clocks and paired IDs | sensor reboot resets timestamp | age appears negative |
| B1 occupancy | one epoch, complete accounting | overwritten blocks never admitted | healthy queue hides data loss |
| S1 useful-task ratio | fixed eligible denominator | failed offline tasks excluded | reported reliability inflates |

**Takeaway:** An invariant is useful only when its observation includes every participant that can violate it.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE integration trace; no physical robot was operated.
**Decision:** Determine whether a nominally fast perception path can authorize travel at 1.2 m/s.
**Scenario:** A warehouse base has 0.95 m observed clear distance to a stationary obstacle.
**Assumed deceleration:** 0.8 m/s² after braking begins.
**Omitted factors:** Obstacle motion, spatial uncertainty, slope, brake buildup and additional safety margin.
**Consequence:** Failure under this simplified model is already sufficient to reject the proposed clearance.
**Nonconsequence:** Passing this model would not establish safe operation.

| Boundary | Timestamp in illustrative common clock | Increment |
|---|---:|---:|
| Physical exposure midpoint | 10.000 s | start |
| Driver releases image | 10.030 s | 30 ms |
| Preprocessor starts after queue | 10.090 s | 60 ms |
| Model starts | 10.100 s | 10 ms |
| Model completes | 10.120 s | 20 ms |
| Action gate runs | 10.135 s | 15 ms |
| Brake command accepted | 10.150 s | 15 ms |

The inference kernel accounts for only 20 of the 150 ms information age.
The queue accounts for 60 ms and is therefore the largest listed removable contributor.
Reducing kernel time to 10 ms saves 10 ms only if downstream scheduling does not absorb the gain.
Replacing the queue with a justified latest-valid-frame policy could save more age.
Such a policy must not drop events needed for an estimator's state propagation.
The valid optimization depends on the consumer contract, not solely on queue length.

Applying T1 gives 10.150−10.000=0.150 s.
Applying D1 gives delay travel 1.2×0.150=0.180 m.
The braking contribution is 1.2²/(2×0.8)=0.900 m.
The simplified total is 1.080 m.
Compared with 0.950 m clearance, the model has a 0.130 m deficit.
The unit check produces metres for each contribution.
The sign check produces positive delay and braking distances.
The independent inverse check multiplies braking distance by twice deceleration.
That gives 0.900×1.6=1.440 m²/s², equal to the assumed speed squared.
This algebraic check is not a target measurement.

| Sensitivity | Delay distance | Braking distance | Total | Decision under simplified clearance |
|---|---:|---:|---:|---|
| Base: 1.2 m/s, 150 ms, 0.8 m/s² | 0.180 m | 0.900 m | 1.080 m | reject |
| Delay reduced to 50 ms | 0.060 m | 0.900 m | 0.960 m | still reject |
| Speed reduced to 1.0 m/s | 0.150 m | 0.625 m | 0.775 m | model alone no longer rejects |
| Deceleration falls to 0.6 m/s² | 0.180 m | 1.200 m | 1.380 m | reject more strongly |

The wrong path uses only the 20 ms kernel latency.
It predicts 0.024+0.900=0.924 m and appears to pass the 0.950 m clearance.
That false pass arises from omitting 130 ms of information age.
At 1.2 m/s, the omitted delay contributes 0.156 m.
Adding it back turns 0.924 m into 1.080 m exactly.
The error is in the causal interval, not numerical rounding.
The reduced-speed example does not authorize travel because omitted uncertainties remain.
A real release gate needs qualified physical bounds and the relevant hazard analysis.
The control owner must define what happens when the information-age bound is unavailable.
A safe fallback cannot be inferred from successful inference execution.

**Takeaway:** Optimize and validate observation-to-action age, not merely model execution time.

## 6. Visual / Geometric Intuition

```text
Illustrative straight-line position x [m], schematic not to scale
x=0.000        x=0.180                    x=1.080
observation -> braking begins ----------> simplified stop
|<--delay--->|<------braking 0.900 m------>|
                              obstacle starts at x=0.950 m
                              stopping estimate exceeds it by 0.130 m
```

The delay and braking segments are sequential pieces of physical travel.
They are not competing estimates of the same distance.
The obstacle marker falls before the simplified stop marker.
A lower mean inference time cannot move the observation origin forward.
Only a later valid observation or less downstream delay changes information age.

```text
Authority state, schematic transitions
LOCAL_READY -> REMOTE_LEASED -> LOCAL_READY
      |              |
      | lease/freshness invalid
      v              v
  HAZARD-SPECIFIC CONTROLLED CONDITION
      |
      | explicit recovery evidence and authorized release
      v
  LOCAL_READY
```

A network reconnect is not shown as an automatic motion-resume transition.
The controlled condition must preserve whatever ongoing actuation the hazard analysis requires.
An expired remote lease invalidates commands even if their transport connection remains open.
The state diagram teaches authority semantics, not a certified controller design.

**Takeaway:** Draw the physical interval and the authority transition; dashboards alone hide both.

## 7. Comparison / Selection Table

| Diagnostic method | Use when | Expected discriminating observation | Avoid or limit when |
|---|---|---|---|
| Boundary timestamp trace | freshness or deadline is disputed | paired stage gaps explain one causal path | clocks are unmapped or tracing blocks control |
| Hardware sequence capture | loss may precede software queue | gaps or reused generations before enqueue | capture cannot identify reset epochs |
| Reference tensor comparison | host/target semantic drift suspected | first differing boundary localizes conversion | reference shares the suspect preprocessing |
| Static target memory map | model fits on disk but device crashes | stack/arena/live-buffer overlap | dynamic workload peak is omitted |
| Frame-chain replay | motion estimates disagree spatially | inverse/axis/time inconsistency | replay substitutes latest transform for historic transform |
| Independent physical stopping test | clearance assumption is material | bounded stopping response under stated load | unsafe environment or unqualified procedure |
| Episode-group holdout | synthetic evaluation may leak | performance falls on unseen scenes/trajectories | held-out variants still share source identity |
| Lease and command journal | duplicate or stale remote action | repeated effect ID or expired authority | journal acknowledgement is mistaken for actuation |

Prioritize risk containment over diagnostic convenience.
If a physical hazard is active, use the approved local containment path first.
If state identity is uncertain, quarantine evidence rather than merge incompatible epochs.
If timing is uncertain, collect a paired monotonic trace before optimizing kernels.
If target output differs, compare preprocessing, layout, scales and operator support in that order of observability.
If geometry is uncertain, resolve frame convention and timestamp before retuning estimator noise.
If a simulator fails to transfer, test model identifiability before expanding randomization ranges.
If rollback is needed, inspect bundle compatibility before selecting an old binary.
If evidence remains ambiguous, prefer a bounded observation that separates hypotheses.
Do not increase logging indiscriminately on an already overloaded control host.
A useful diagnostic has a predicted signal under each competing explanation.
A test that returns only “healthy” without an observation model is weak evidence.
A positive correlation between logs and failure is not sufficient causal attribution.
Use intervention or an independently derived oracle when safe and feasible.
A reproducible software defect can coexist with a second physical defect.
After a correction, test the originally failing regime and the nearest boundary regime.

**Takeaway:** Choose a diagnostic for the hypothesis it can falsify, not for how familiar its tool is.

## 8. Failure Modes & Edge Cases

### Issue index

All provenance entries below are HYPOTHETICAL teaching scenarios.
The category is independent of evidence maturity.
“Root cause” in the index names a mechanism to test, not an observed incident conclusion.

| ID | Category | Symptom | Candidate mechanism | Discriminating signal | Bounded fix | Related home |
|---|---|---|---|---|---|---|
| P07-001 | VIOLATED_ASSUMPTION | repeated signal blocks | DMA overwrites reader | generation changes mid-read | immutable ownership handoff | module 35 |
| P07-002 | SCALE_FAILURE | deadlines fail during logging | low-priority lock holder | blocking chain, not CPU saturation | bounded ownership separation | module 35 |
| P07-003 | NUMERICAL_EDGE | sudden sensor offset | reference-voltage mismatch | raw counts stable, converted units change | version calibration together | module 35 |
| P07-004 | CONCEPT_CONFUSION | fast UART, missing waveforms | framing/line bits ignored | wire-time budget exceeds assumption | budget full wire format | module 35 |
| P07-005 | VIOLATED_ASSUMPTION | acceptable bench battery life only | wake and sensor energy omitted | whole-cycle energy differs | measure duty cycle boundaries | module 35 |
| P07-006 | VIOLATED_ASSUMPTION | rare target saturation | calibration misses extremes | clipping rises on held-out slice | revise representative calibration | module 36 |
| P07-007 | CONCEPT_CONFUSION | smaller model no faster | no sparse/INT4 target kernel | dense kernel still executes | choose supported structure | module 36 |
| P07-008 | NUMERICAL_EDGE | correct converted shape, wrong class | layout/zero-point mismatch | boundary tensors first diverge | align semantic conversion | module 36 |
| P07-009 | SCALE_FAILURE | memory crash after warm start | activation liveness omitted | overlapping buffers exceed arena | derive actual peak lifetime | module 36 |
| P07-010 | VIOLATED_ASSUMPTION | optimized compiler output drifts | illegal reorder or lowering | first failing pass identified | constrain transformation | module 36 |
| P07-011 | VIOLATED_ASSUMPTION | fusion error grows during turns | clock drift mapped as geometry | residual tracks angular speed/time | recalibrate time model | module 37 |
| P07-012 | CONCEPT_CONFUSION | mirrored point cloud | optical/body axes mixed | known target transforms incorrectly | explicit frame conversion | module 37 |
| P07-013 | VIOLATED_ASSUMPTION | overconfident fused estimate | correlated inputs double-counted | innovation inconsistency despite low covariance | model cross-correlation | module 37 |
| P07-014 | NUMERICAL_EDGE | implausible HRV improvement | filtering masks bad peaks | RR edits dominate statistic | preserve quality/exclusion ledger | module 37 |
| P07-015 | SCALE_FAILURE | low compute latency, stale output | old frames queued | age rises while kernel stable | bounded freshness-aware queue | module 37 |
| P07-016 | VIOLATED_ASSUMPTION | publisher/subscriber never connect | QoS incompatibility | endpoints exist, match absent | compatible explicit profiles | module 38 |
| P07-017 | CONCEPT_CONFUSION | controller jerks at relocalization | map jump used as local motion | odom continuous, map discontinuous | separate local/global contracts | module 38 |
| P07-018 | NUMERICAL_EDGE | arm velocities explode | Jacobian near singular | small singular value, joint-rate growth | bounded damped solve | module 38 |
| P07-019 | VIOLATED_ASSUMPTION | oscillation after heavier payload | delay/saturation ignored | phase lag and clipped command | revalidate plant/control envelope | module 38 |
| P07-020 | SCALE_FAILURE | watchdog fires only under load | timer shares blocked executor | timer delay follows callback occupancy | independent bounded execution | module 38 |
| P07-021 | VIOLATED_ASSUMPTION | SLAM snaps to wrong aisle | perceptual aliasing | ambiguous match geometry | demand disambiguating evidence | module 39 |
| P07-022 | CONCEPT_CONFUSION | feasible path collides physically | point footprint used for loaded robot | swept volume exceeds planned clearance | version full footprint | module 39 |
| P07-023 | NUMERICAL_EDGE | high grasp score, dropped part | pose uncertainty/contact mismatch | failure depends on friction/approach | evaluate physical grasp envelope | module 39 |
| P07-024 | VIOLATED_ASSUMPTION | VLA follows unsafe instruction | semantic intent bypasses action gate | valid text yields forbidden action | independent bounded action contract | module 39 |
| P07-025 | SCALE_FAILURE | repeated robot action after retry | behavior tree reissues side effect | same effect duplicated across tick | explicit action identity/state | module 39 |
| P07-026 | VIOLATED_ASSUMPTION | perfect synthetic test | shared scene/trajectory leakage | group holdout degrades sharply | partition latent source identities | module 40 |
| P07-027 | NUMERICAL_EDGE | simulator stable, robot unstable | integration timestep hides dynamics | response changes with timestep | convergence and plant checks | module 40 |
| P07-028 | CONCEPT_CONFUSION | randomized policy brittle | impossible independent parameter combinations | synthetic joint distribution implausible | constrained randomization model | module 40 |
| P07-029 | VIOLATED_ASSUMPTION | twin agrees after every update | measurement and validation share fitting data | independent sensor residual diverges | separate fit and validation | module 40 |
| P07-030 | SCALE_FAILURE | many episodes, no rare failures | scenario coverage mistaken for trial count | untested hazard cell remains empty | stratified exposure ledger | module 40 |
| P07-031 | VIOLATED_ASSUMPTION | two operators control one robot | stale overlapping leases | concurrent valid-looking command epochs | exclusive local lease gate | module 41 |
| P07-032 | NUMERICAL_EDGE | rollback breaks perception | firmware/model/calibration mismatch | bundle hashes incompatible | atomic compatible release unit | module 41 |
| P07-033 | CONCEPT_CONFUSION | signed update still unsafe | trust mistaken for correctness | verified image violates interface test | separate authenticity and behavior gates | module 41 |
| P07-034 | SCALE_FAILURE | fleet SLO improves during outage | offline attempts disappear | eligible count drops with failures | reconstruct full denominator | module 41 |
| P07-035 | VIOLATED_ASSUMPTION | near-miss investigation contradicts logs | unsynchronized evidence and overwritten buffers | event order intervals overlap | preserve clock/epoch provenance | module 41 |

### P07-001 — DMA block ownership survives the software queue
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; source topic inventory lines 1097 and 1109.
Related home: [module 35](../35_edge_ai_embedded_systems_and_tinyml/INDEX.md).
Trigger: The acquisition DMA wraps while preprocessing still reads a previously released half-buffer.
Symptom: Individual blocks look plausible, but adjacent windows repeat or contain impossible discontinuities.
Competing cause A: Hardware overwrote memory before the consumer finished.
Competing cause B: A downstream message was retransmitted without changing its sample sequence.
Competing cause C: The physical input was genuinely periodic over the compared windows.
Discriminator: Capture block generation at read start and read end without copying from the suspect reader.
Expected under A: The generation changes during one consumer read.
Expected under B: The hardware generation is stable but the downstream event ID repeats.
Expected under C: Independent capture shows matching waveform with advancing sequence and stable ownership.
Invariant: A producer cannot reuse a block while any authorized consumer may still read it.
Containment: Mark the entire affected block invalid rather than splice its two generations together.
Mitigation risk: Dropping blocks can break filter state and must be surfaced as a gap.
Correction: Transfer immutable block ownership or copy into a separately budgeted stable buffer at handoff.
Prevention: Define the exact interrupt, task and DMA ownership transitions in the interface contract.
Verification: Delay the consumer beyond one ring wrap and require explicit invalidation, never silent mixed data.
Negative result: Stable generations do not eliminate a wrong DMA transfer length or cache-coherency defect.
Escalation: The firmware owner must inspect device-specific DMA and memory behavior if the software trace is inconclusive.
Boundary: A CPU mutex does not automatically serialize independent DMA writes.
Evidence to retain: Hardware sequence, DMA index, buffer generation, consumer ID and reset epoch.

### P07-002 — Logging changes the schedulability argument
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; FreeRTOS priority-inversion topic at inventory line 1098.
Related home: [module 35](../35_edge_ai_embedded_systems_and_tinyml/INDEX.md).
Trigger: A high-priority inference task needs a lock held by a low-priority diagnostic writer.
Symptom: Deadline misses occur while overall CPU utilization remains apparently moderate.
Competing cause A: A medium-priority task preempts the lock holder and extends blocking.
Competing cause B: The inference workload itself exceeds its assumed execution bound.
Competing cause C: Interrupt service consumes the missing interval outside task-level CPU accounting.
Discriminator: Record the lock owner and ready/running transitions around a missed deadline.
Expected under A: The high-priority task is blocked while the low-priority owner is ready but not running.
Expected under B: The high-priority task runs continuously with unusually long execution.
Expected under C: Interrupt occupancy explains the interval without the corresponding task blocking chain.
Invariant: The response-time model must include bounded blocking and interrupt interference.
Containment: Disable nonessential synchronous logging through the approved diagnostic control, not safety reporting.
Mitigation risk: Removing logs can erase the only evidence of the scheduling chain.
Correction: Separate control-path state from slow logging and use bounded, explicitly analyzed handoff.
Prevention: Review lock ownership, priority policy and worst permitted sink delay together.
Verification: Stall the diagnostic sink while the controller runs its maximum allowed workload.
Negative result: No lock contention in one trace does not establish a global blocking bound.
Escalation: Scheduling changes require the real-time owner to review starvation and watchdog consequences.
Boundary: Priority inheritance addresses specific inversion patterns, not arbitrary unbounded I/O inside a critical section.
Evidence to retain: Task priorities, lock events, interrupt intervals and the exact logging configuration.

### P07-003 — ADC counts are not calibrated physical units
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; ADC, register and HAL topics at inventory lines 1096–1097 and 1107.
Related home: [module 35](../35_edge_ai_embedded_systems_and_tinyml/INDEX.md).
Trigger: Firmware changes the ADC reference or gain assumption while the feature pipeline retains old conversion metadata.
Symptom: The model's feature scale shifts even though raw count histograms appear unchanged.
Competing cause A: The count-to-voltage conversion is wrong.
Competing cause B: The physical sensor sensitivity actually changed with temperature or damage.
Competing cause C: Training normalization and serving normalization use different revisions.
Discriminator: Apply a known safe reference signal and compare raw counts, converted units and normalized features separately.
Expected under A: Counts agree with the hardware configuration but the converted voltage disagrees.
Expected under B: An independent physical reference disagrees before software conversion.
Expected under C: Physical units match but the normalized tensor diverges.
Invariant: Every derived feature must identify the calibration and normalization revision used.
Containment: Abstain from decisions whose input scale is not established.
Mitigation risk: Substituting an average gain can hide channel-specific faults.
Correction: Version the complete count/reference/gain/offset mapping with the feature specification.
Prevention: Treat analog configuration changes as model-interface changes.
Verification: Check multiple reference levels, including near-range limits, with the actual channel configuration.
Negative result: A single midscale point cannot distinguish gain error from offset error.
Escalation: Saturation, electrical isolation and reference accuracy require qualified hardware review.
Boundary: A high ADC bit depth does not guarantee accurate or noise-free measurements.
Evidence to retain: Raw codes, reference source, input units, device revision and calibration uncertainty.

### P07-004 — UART throughput includes bits that are not payload
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; UART framing topic at inventory line 1100.
Related home: [module 35](../35_edge_ai_embedded_systems_and_tinyml/INDEX.md).
Trigger: A pipeline budgets 115200 baud as 115200 useful bytes per second.
Symptom: Sample transfer falls behind despite a seemingly generous serial-rate budget.
Competing cause A: Start/stop bits and protocol headers were omitted.
Competing cause B: Host parsing or storage blocks reads despite adequate wire capacity.
Competing cause C: Electrical errors cause discarded frames or retransmission.
Discriminator: Compare captured wire occupancy, receiver byte counts and accepted sample sequence.
Expected under A: Wire utilization approaches capacity with correct framing but insufficient payload rate.
Expected under B: Bytes arrive before long host-consumption gaps.
Expected under C: Framing/checksum errors correlate with missing accepted frames.
Invariant: Wire bits, delivered bytes and useful samples are separate counters.
Containment: Bound queued sample age and flag gaps rather than silently delay every waveform.
Mitigation risk: Raising baud may increase errors or violate supported electrical/device limits.
Correction: Include line encoding, frame headers, checksums and scheduling gaps in the budget.
Prevention: Maintain a full wire-format size example beside each protocol revision.
Verification: Send a deterministic sequence with maximum permitted frame size and worst permitted host pause.
Negative result: A clean checksum does not prove sample order or absence of duplicated valid frames.
Escalation: Persistent line errors need electrical and clock-tolerance investigation, not only parser changes.
Boundary: For 8N1, ten wire bits carry one eight-bit byte before protocol overhead.
Evidence to retain: Baud configuration, parity/stop settings, wire capture and frame sequence/epoch.

### P07-005 — Duty-cycle estimates omit wake and peripheral energy
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; sleep/stop/standby topic at inventory line 1101.
Related home: [module 35](../35_edge_ai_embedded_systems_and_tinyml/INDEX.md).
Trigger: Battery sizing multiplies inference power by inference duration but ignores sensor and wake behavior.
Symptom: A bench kernel estimate predicts much longer autonomy than whole-device testing.
Competing cause A: Sensor, radio or analog front-end remains powered between inferences.
Competing cause B: Wake, initialization and clock stabilization dominate short work periods.
Competing cause C: Battery capacity derates under the actual load or temperature.
Discriminator: Integrate current over complete repeated cycles while separately observing power-domain transitions.
Expected under A: A sustained baseline remains after the processor enters its intended low-power mode.
Expected under B: Repeated startup peaks dominate the cycle integral.
Expected under C: Device energy matches the budget but usable battery discharge differs.
Invariant: Energy accounting includes every powered domain over the full operational cycle.
Containment: Reduce optional duty cycle only within the sensing and safety requirements.
Mitigation risk: Lower sampling or longer sleep can invalidate detection latency and filter assumptions.
Correction: Use a measured state-duration ledger including wake, compute, communication and sleep.
Prevention: Review energy and information-age budgets together whenever duty cycle changes.
Verification: Repeat under low-battery voltage, realistic temperature and the maximum required reporting schedule.
Negative result: Nominal average current alone cannot explain transient brownouts.
Escalation: Battery protection and allowable operating conditions need hardware-owner review.
Boundary: Low-power modes differ by exact device; a Cortex family name is insufficient documentation.
Evidence to retain: Voltage/current trace, sensor state, wake reason, period and battery condition.

### P07-006 — Calibration coverage misses rare activation extremes
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; PTQ calibration topics at inventory lines 1062–1065.
Related home: [module 36](../36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md).
Trigger: Post-training calibration contains common clean inputs but deployment includes rare high-amplitude conditions.
Symptom: Aggregate accuracy changes little while an important operating slice saturates.
Competing cause A: Quantization range clips valid rare activations.
Competing cause B: The rare slice is already misclassified by the floating-point model.
Competing cause C: Target preprocessing produces an unintended amplitude shift.
Discriminator: Compare float and quantized boundary activations for the same frozen rare-slice inputs.
Expected under A: Float remains usable while quantized values pile up at range endpoints.
Expected under B: Both models fail before target conversion.
Expected under C: The target input tensor diverges from the reference before the first model operation.
Invariant: Calibration selection and final evaluation are separate, versioned procedures.
Containment: Reject the unvalidated slice or retain a supported higher-precision path if its full budget is established.
Mitigation risk: Expanding range can reduce resolution for common values and damage another slice.
Correction: Revisit calibration coverage, clipping policy, per-channel treatment or QAT under a fixed evaluation protocol.
Prevention: Record activation-range and quality checks by physically meaningful operating slice.
Verification: Evaluate the revised design on untouched rare-slice data and supported target kernels.
Negative result: Fewer clipped activations do not automatically imply better task decisions.
Escalation: A changed calibration dataset must not silently become a tuned test set.
Boundary: AWQ or SmoothQuant methods for larger models are not automatically MCU CNN solutions.
Evidence to retain: Calibration IDs, scales, zero points, clipping counts and float/target paired outputs.

### P07-007 — Compression does not imply executable acceleration
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; sparse and mixed-precision topics at inventory lines 1068–1072.
Related home: [module 36](../36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md).
Trigger: A model has many zero weights or packed low-bit weights, but target execution still uses dense higher-width kernels.
Symptom: Artifact size falls while end-to-end latency barely changes or worsens.
Competing cause A: The runtime has no effective sparse or low-bit kernel for the operator.
Competing cause B: Unpacking or layout conversion consumes the saved execution time.
Competing cause C: The workload is dominated by memory movement or non-model processing.
Discriminator: Inspect resolved kernel names, executed instructions and boundary timing, not just file format.
Expected under A: Dense multiply operations remain despite sparse metadata.
Expected under B: Conversion stages account for the extra time.
Expected under C: Model time improves but observation-to-action age remains dominated elsewhere.
Invariant: A compression claim must name both representation savings and actual executed path.
Containment: Keep the known-correct supported path until the new path's total budget is measured.
Mitigation risk: Enabling an unvalidated custom kernel can trade latency for silent numerical errors.
Correction: Select supported structure, precision or architecture based on the identified bottleneck.
Prevention: Require operator-level support and conversion accounting in compression review.
Verification: Compare identical workloads, batch sizes, thermal conditions and quality guardrails.
Negative result: A faster microkernel does not establish a faster complete application.
Escalation: Unsupported operators require an explicit implementation decision, not a hidden fallback.
Boundary: Zero-valued parameters do not reduce dense arithmetic automatically.
Evidence to retain: Model bytes, metadata bytes, resolved operators, timing breakdown and target configuration.

### P07-008 — Tensor shape equality hides semantic conversion drift
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; interoperability and operator-coverage topics at inventory lines 1089–1090.
Related home: [module 36](../36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md).
Trigger: Conversion preserves dimensions but changes channel order, scale, zero point or padding semantics.
Symptom: Target outputs differ sharply despite successful import and shape checks.
Competing cause A: Channels-last data is interpreted as channels-first.
Competing cause B: Affine quantization uses the wrong zero point or rounding rule.
Competing cause C: A genuinely different implementation chooses a permissible floating-point evaluation order.
Discriminator: Compare named intermediate tensors on a small asymmetric input with nonzero offsets.
Expected under A: Values are permuted consistently across channel/spatial positions.
Expected under B: Differences show systematic shifts or saturation after quantization.
Expected under C: Small bounded differences emerge around arithmetic operations without semantic permutation.
Invariant: Shape, dtype, layout, units, quantization and padding jointly define a tensor contract.
Containment: Block promotion when equivalence tolerance or semantic interpretation is unresolved.
Mitigation risk: Relaxing tolerances broadly can hide a channel permutation.
Correction: Fix the conversion boundary and use operation-appropriate reference tolerances.
Prevention: Include asymmetric channel and boundary-padding fixtures in the conversion specification.
Verification: Check intermediate values, edge shapes and downstream decision stability.
Negative result: Matching one all-zero tensor is weak evidence because many wrong layouts agree on it.
Escalation: If the first divergent operation is unsupported, resolve operator semantics before retraining.
Boundary: Bitwise equality is not always appropriate across valid floating-point implementations.
Evidence to retain: Exporter/runtime revisions, tensor contracts, reference inputs and first divergent operation.

### P07-009 — Flash fit says nothing about peak live activation memory
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; arena and linker topics at inventory lines 1079 and 1106.
Related home: [module 36](../36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md).
Trigger: A model's weights fit persistent storage but its concurrently live tensors, scratch and stacks exceed SRAM.
Symptom: Warm-start or larger-input execution corrupts data or fails allocation.
Competing cause A: Peak activation liveness was underestimated.
Competing cause B: Stack depth grows under interrupt nesting or diagnostic callbacks.
Competing cause C: An actual out-of-bounds write corrupts otherwise sufficient memory.
Discriminator: Reconcile linker sections, planned tensor lifetimes, stack watermark and guard-region failures.
Expected under A: Valid live intervals overlap beyond the allocated arena.
Expected under B: Stack high-water use grows under the failing interrupt scenario.
Expected under C: Guard corruption occurs before total legitimate memory reaches capacity.
Invariant: Every simultaneously live byte has nonoverlapping ownership unless aliasing is proven safe.
Containment: Reject the configuration and preserve corruption evidence before restarting into normal operation.
Mitigation risk: Enlarging the arena can steal stack headroom and move the failure.
Correction: Replan lifetimes, reduce supported input/model size or choose a justified memory layout.
Prevention: Budget weights, activations, scratch, protocol buffers, stacks and alignment separately.
Verification: Exercise peak shapes, permitted concurrency, cold/warm paths and interrupt nesting.
Negative result: A successful single inference does not establish the memory bound.
Escalation: Memory corruption requires root-cause analysis; allocation suppression is not a correction.
Boundary: SRAM capacity must come from the exact device variant, not a related MCU example.
Evidence to retain: Linker map, arena plan, guard captures, input shape and stack configuration.

### P07-010 — An optimization pass can preserve types but change meaning
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; graph, loop and pass-debugging topics at inventory lines 1115–1126.
Related home: [module 36](../36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md).
Trigger: Fusion, reordering or quantized lowering crosses an unproved dependence or rounding boundary.
Symptom: An optimized target is faster on common inputs but wrong on edge values.
Competing cause A: A compiler transformation is semantically illegal for the operation contract.
Competing cause B: Input preprocessing differs between optimized and reference runs.
Competing cause C: The comparison demands bitwise equality where only bounded numerical equivalence is promised.
Discriminator: Freeze input tensors and bisect the pass pipeline with the same operation-level oracle.
Expected under A: The first unacceptable divergence appears after one specific transformation.
Expected under B: Divergence already exists at the pipeline input.
Expected under C: Differences remain within a justified error model and preserve required decisions.
Invariant: Every transformation preserves the declared semantics or an explicitly accepted approximation contract.
Containment: Disable the identified transformation for the affected configuration, not the correctness check.
Mitigation risk: Disabling one pass can alter scheduling, memory and timing assumptions elsewhere.
Correction: Repair the legality condition, lowering rule or approximation specification.
Prevention: Keep version-pinned IR snapshots and edge-value reference fixtures.
Verification: Recheck the failing shape/dtype and neighboring supported configurations.
Negative result: A compiler's successful verification of IR structure does not prove task-level equivalence.
Escalation: Report the minimal failing input and pass sequence to the responsible compiler owner.
Boundary: Relay, TIR, MLIR dialects and compiler APIs have version-specific roles; names alone do not establish support.
Evidence to retain: Source IR, pass order, compiler revision, target flags and independent expected outputs.

### P07-011 — Clock drift masquerades as spatial miscalibration
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; timing/calibration course topic and IMU inventory line 1039.
Related home: [module 37](../37_sensor_fusion_iot_and_realtime_perception/INDEX.md).
Trigger: A camera and IMU begin synchronized but their relative clock rates diverge during operation.
Symptom: Extrinsic calibration seems correct when stationary and wrong during turns.
Competing cause A: Timestamp offset or drift is unmodeled.
Competing cause B: The rigid spatial transform is physically wrong.
Competing cause C: The sensor mount flexes under acceleration.
Discriminator: Compare residuals across controlled motion speeds and elapsed time since synchronization.
Expected under A: Residual magnitude follows motion rate and accumulated timing error.
Expected under B: A repeatable spatial bias remains across synchronized observations.
Expected under C: Residual depends on load/acceleration with mechanical hysteresis or vibration.
Invariant: Measurement fusion requires bounded time alignment as well as a valid spatial relation.
Containment: Down-weight or reject combinations whose alignment bound is invalid.
Mitigation risk: Increasing measurement noise can conceal timing error without correcting phase bias.
Correction: Estimate and monitor the supported clock mapping, including drift and reset behavior.
Prevention: Store observation-clock identity and synchronization quality in the sensor contract.
Verification: Repeat after long duration, clock reset, network delay and temperature changes.
Negative result: Stationary agreement cannot distinguish a timing error because motion provides little excitation.
Escalation: Hardware synchronization and timestamp placement require the acquisition owner's evidence.
Boundary: Receive timestamps cannot replace physical exposure times without a justified delay model.
Evidence to retain: Raw clock pairs, synchronization events, motion rate, residuals and calibration revision.

### P07-012 — Camera optical coordinates are not body coordinates
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; coordinate conventions sourced to REP 103, created 2010, post-history 2014.
Related home: [module 37](../37_sensor_fusion_iot_and_realtime_perception/INDEX.md).
Trigger: A camera ray is treated as body-forward x without converting the optical-frame convention.
Symptom: Obstacles appear mirrored, sideways or below the floor after otherwise plausible projection.
Competing cause A: Optical/body axis conventions are mixed.
Competing cause B: The transform is applied in the wrong direction.
Competing cause C: Camera intrinsics or image mirroring is wrong.
Discriminator: Transform three known noncollinear target directions and check expected handedness and location.
Expected under A: A systematic axis permutation/sign pattern explains all three.
Expected under B: Applying the mathematically correct inverse resolves the discrepancy.
Expected under C: Errors depend on pixel position or optical distortion rather than one rigid permutation.
Invariant: Every point carries a frame and every transform states source, destination and time.
Containment: Withhold geometric actions based on an unresolved frame chain.
Mitigation risk: A hand-tuned sign flip may fix one pose and break another.
Correction: Implement the documented optical-to-body transform and verify chirality.
Prevention: Use [REP 103](https://www.ros.org/reps/rep-0103.html) conventions or explicitly document exceptions.
Verification: Check known right/up/forward targets and transform/inverse round trips.
Negative result: A round trip using the same wrong transform and inverse does not establish physical correctness.
Escalation: Calibration owner confirms the physical mounting and authoritative frame publication.
Boundary: REP 103's camera optical convention is z forward, x right, y down; body convention differs.
Evidence to retain: Target geometry, intrinsic/extrinsic revisions, frame IDs and transform direction.

### P07-013 — Two correlated estimates are not two independent witnesses
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; Kalman/Bayesian filtering course commitment.
Related home: [module 37](../37_sensor_fusion_iot_and_realtime_perception/INDEX.md).
Trigger: A localization filter fuses wheel odometry with another estimate already containing that same wheel odometry.
Symptom: Reported covariance shrinks while actual tracking error remains large.
Competing cause A: Shared information is double-counted under an independence assumption.
Competing cause B: One measurement covariance is simply underestimated.
Competing cause C: Both estimates share an unmodeled physical bias such as wheel slip.
Discriminator: Trace input lineage and compare innovations against independent external reference segments.
Expected under A: Removing the duplicated contribution improves consistency without adding new physical information.
Expected under B: One source alone is inconsistent with its declared uncertainty.
Expected under C: Both sources shift together during the same physical condition.
Invariant: The uncertainty update must reflect actual information dependence or use a defensible conservative alternative.
Containment: Avoid presenting the fused covariance as established physical accuracy.
Mitigation risk: Arbitrary covariance inflation can weaken useful information while retaining the shared bias.
Correction: Model cross-correlation, choose nonoverlapping inputs or use an appropriate conservative fusion method.
Prevention: Maintain a sensor-estimate lineage graph alongside the computational graph.
Verification: Assess residual consistency across motion and failure slices, not only mean position error.
Negative result: Low residuals between correlated sources do not prove either source is correct.
Escalation: Estimator owner must review observability and bias states if no independent reference is available.
Boundary: Gaussian filter equations do not remove an invalid independence premise.
Evidence to retain: Source lineage, covariance definitions, reference data and innovation sequence.

### P07-014 — Clean-looking RR intervals can be the result of invalid editing
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; HRV and signal-quality topics at inventory lines 1033–1035 and 1040.
Related home: [module 37](../37_sensor_fusion_iot_and_realtime_perception/INDEX.md).
Trigger: A peak detector's false or missing beats are aggressively interpolated before HRV statistics are computed.
Symptom: A new filter reports smoother RR intervals and apparently improved stability without a better peak oracle.
Competing cause A: Editing removed physiological variability along with detection errors.
Competing cause B: Motion artifacts created false detections that the edit legitimately removed.
Competing cause C: Timestamp gaps or sample-rate conversion changed interval units.
Discriminator: Compare raw waveform, expert/reference peak positions, edited intervals and exclusion flags together.
Expected under A: The editing rule systematically suppresses valid beat-to-beat changes.
Expected under B: Removed events align with independent artifact or peak-label evidence.
Expected under C: Interval distortion follows a timing conversion or gap boundary.
Invariant: A reported HRV statistic must identify valid beat selection, edits, segment duration and time units.
Containment: Mark the segment unsuitable when validity cannot be established.
Mitigation risk: Interpolation can manufacture regularity and invalid spectral structure.
Correction: Separate detection evaluation from downstream HRV estimation and preserve every edit reason.
Prevention: Use patient-separated evaluations and versioned waveform/annotation provenance.
Verification: Recompute statistics with and without justified edits and report sensitivity.
Negative result: A lower RMSSD is not by itself an improvement or a clinical conclusion.
Escalation: Physiological and clinical interpretation requires qualified domain review.
Boundary: [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) is research data, not evidence of a deployed device's validity.
Evidence to retain: Peak indices, sample rate, gap flags, edits, subject partition and reference annotation revision.

### P07-015 — Queue throughput can hide unacceptable perception age
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; perception-latency and edge-telemetry course commitments.
Related home: [module 37](../37_sensor_fusion_iot_and_realtime_perception/INDEX.md).
Trigger: An image queue preserves every frame even though service intermittently falls below acquisition rate.
Symptom: Kernel latency is stable but the robot acts on increasingly old scenes.
Competing cause A: FIFO backlog accumulates valid but stale observations.
Competing cause B: Sensor clock offset makes fresh frames appear old.
Competing cause C: The model itself has occasional long executions hidden by coarse timing.
Discriminator: Pair observation time, enqueue/dequeue time and model start/end for each frame ID.
Expected under A: Queue residence grows while compute duration stays stable.
Expected under B: Apparent age shifts without corresponding queue occupancy.
Expected under C: Compute-duration spikes explain the backlog onset.
Invariant: The action gate checks information age of the actual observation used.
Containment: Reject stale outputs and invoke the approved local behavior for missing fresh perception.
Mitigation risk: Dropping intermediate frames can invalidate temporal models or estimator propagation.
Correction: Choose a bounded queue/latest-valid policy consistent with the consumer's temporal contract.
Prevention: Monitor age and queue residence separately from throughput and kernel latency.
Verification: Apply bursts and sustained overload; require bounded age or explicit refusal.
Negative result: An empty queue does not establish freshness if acquisition itself stalled.
Escalation: Control owner determines the valid degraded behavior and maximum permitted age.
Boundary: Faster processing cannot recover physical information that was never observed.
Evidence to retain: Frame IDs, observation clock, queue depth, stage times and action decision.

### P07-016 — Visible ROS endpoints can still be unable to communicate
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; ROS 2 graph course topic; middleware behavior must be checked against the chosen distribution.
Related home: [module 38](../38_robotics_ros2_kinematics_and_control/INDEX.md).
Trigger: Publisher and subscriber have incompatible quality-of-service expectations or type/interface revisions.
Symptom: Discovery lists both nodes but the subscriber receives no useful measurements.
Competing cause A: QoS compatibility prevents endpoint matching.
Competing cause B: Network/domain/security configuration blocks data communication.
Competing cause C: Messages arrive but the application rejects stale or malformed contents.
Discriminator: Inspect endpoint type/QoS matching, transport observations and application rejection counters separately.
Expected under A: Discovery succeeds but compatibility evidence identifies a policy mismatch.
Expected under B: Matching may exist while packets or authorization fail at a separate boundary.
Expected under C: Receive counters advance with explicit application-level invalidation.
Invariant: Discovery, transport delivery and semantic acceptance are different states.
Containment: Preserve the no-data safety response instead of fabricating default valid measurements.
Mitigation risk: Weakening reliability or durability blindly can alter history and loss behavior.
Correction: Set explicit compatible profiles for the actual freshness and delivery needs.
Prevention: Version QoS, type and semantic contracts in deployment manifests.
Verification: Exercise late join, restart, packet loss and stale-data rejection under the selected middleware.
Negative result: A successful topic echo in another process does not prove the controller's callback receives valid data.
Escalation: Middleware and security owners resolve distribution-specific interoperability.
Boundary: This issue does not prescribe one universal QoS profile for all sensor and command topics.
Evidence to retain: Distribution, middleware, type hash/version, endpoint policies and rejection reasons.

### P07-017 — Global relocalization is not local velocity
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; frame semantics sourced to [REP 105](https://www.ros.org/reps/rep-0105.html), created 2010.
Related home: [module 38](../38_robotics_ros2_kinematics_and_control/INDEX.md).
Trigger: A controller differentiates map-frame pose across a discrete localization correction.
Symptom: A stationary or smoothly moving robot appears to have an extreme velocity and receives a jerk command.
Competing cause A: A legitimate map-to-odom correction was interpreted as physical movement.
Competing cause B: Odometry itself reset or jumped unexpectedly.
Competing cause C: Timestamp order changed across the differentiation interval.
Discriminator: Compare map-to-odom, odom-to-base and timestamp sequences around the discontinuity.
Expected under A: Odom-to-base remains continuous while map-to-odom changes.
Expected under B: The local odometry relation jumps or changes epoch.
Expected under C: The apparent derivative uses nonmonotone or mismatched times.
Invariant: Local control must use a state representation whose continuity assumptions match its derivation.
Containment: Reject the invalid derivative and hold the approved local control condition.
Mitigation risk: Smoothing every global correction can create untracked map error and delayed replanning.
Correction: Separate global localization correction from continuous local motion estimation.
Prevention: Document frame authority and allowed discontinuities at every control input.
Verification: Inject a known relocalization correction while the physical trajectory remains smooth.
Negative result: A smooth plotted map pose can hide filtering delay and is not enough evidence.
Escalation: Estimation and control owners jointly define reset and reinitialization behavior.
Boundary: REP 105 permits map discontinuities while odom can drift but is intended to remain continuous.
Evidence to retain: Full transform chain, authority identity, timestamps and estimator reset events.

### P07-018 — Near-singular kinematics amplify small task requests
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; kinematics and motion-planning course commitments.
Related home: [module 38](../38_robotics_ros2_kinematics_and_control/INDEX.md).
Trigger: A Cartesian velocity request is mapped through a nearly singular manipulator Jacobian.
Symptom: Small end-effector motion demands very large joint rates.
Competing cause A: A small singular value amplifies the inverse problem.
Competing cause B: Joint units mix degrees and radians.
Competing cause C: The Jacobian uses the wrong frame or joint ordering.
Discriminator: Check singular values, dimensional conventions and a finite-difference forward-kinematics comparison.
Expected under A: The difficult motion direction aligns with a small singular value.
Expected under B: A systematic unit factor appears across otherwise well-conditioned configurations.
Expected under C: Finite-difference columns disagree with the implemented Jacobian.
Invariant: A bounded Cartesian request does not imply bounded joint rates without conditioning and constraints.
Containment: Enforce validated joint-rate/acceleration limits and reject infeasible task velocity.
Mitigation risk: Clipping joints independently can destroy the intended end-effector path.
Correction: Use a constrained or appropriately damped solve and adapt the task near the singular configuration.
Prevention: Plan configurations and task directions with conditioning awareness.
Verification: Sweep the problematic pose neighborhood and compare task error, rate limits and continuity.
Negative result: A damped inverse's finite output does not prove the requested task remains feasible.
Escalation: Manipulation owner resolves collision, null-space and actuator-limit tradeoffs.
Boundary: Damping trades task accuracy for numerical and rate behavior; it is not free singularity removal.
Evidence to retain: Joint configuration, Jacobian convention, singular values, limits and requested twist.

### P07-019 — Payload and delay can invalidate a previously tuned controller
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; control-fundamentals course topic.
Related home: [module 38](../38_robotics_ros2_kinematics_and_control/INDEX.md).
Trigger: A heavier load, slower actuator or additional perception delay changes the closed-loop plant.
Symptom: A controller that worked unloaded oscillates, saturates or responds sluggishly.
Competing cause A: Plant dynamics and available actuation differ from the tuning model.
Competing cause B: Added measurement/command delay reduces the usable stability margin.
Competing cause C: Sensor noise or derivative implementation creates high-frequency command chatter.
Discriminator: Compare commanded and achieved response, saturation intervals and synchronized signal phase.
Expected under A: The same command produces a load-dependent acceleration or lag.
Expected under B: A repeatable extra delay appears between observation and actuation.
Expected under C: High-frequency sensor variation drives commands without equivalent physical response.
Invariant: Controller validity applies to a stated plant, timing and saturation envelope.
Containment: Reduce operation to the approved envelope or stop under the hazard-specific procedure.
Mitigation risk: Raising gains can aggravate saturation and excite unmodeled dynamics.
Correction: Reidentify the relevant plant behavior and retune or redesign within explicit limits.
Prevention: Treat payload and pipeline-latency changes as control-interface changes.
Verification: Evaluate worst permitted load, delay, actuator limits and recovery from saturation.
Negative result: Good nominal tracking does not establish robustness at the operating envelope boundary.
Escalation: Control owner approves any changed tuning; software deployment alone is not approval.
Boundary: Offline prediction accuracy does not establish closed-loop stability.
Evidence to retain: Load, timing trace, command/response signals, gain revision and saturation flags.

### P07-020 — A watchdog cannot depend on the task it watches
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; ROS timing and robot-safety lifecycle commitments.
Related home: [module 38](../38_robotics_ros2_kinematics_and_control/INDEX.md).
Trigger: The watchdog timer runs on an executor or lock path blocked by the same long callback it should supervise.
Symptom: Fault detection arrives late or only after the blocked computation returns.
Competing cause A: Watchdog scheduling shares the failing execution dependency.
Competing cause B: The watchdog runs but observes a heartbeat unrelated to useful control progress.
Competing cause C: The timer clock pauses or jumps with simulation/system time.
Discriminator: Compare an independent monotonic timing observation with watchdog callback execution and progress sequence.
Expected under A: Both controller and watchdog callbacks stall together.
Expected under B: Heartbeats continue while fresh valid control cycles do not complete.
Expected under C: Timer behavior follows clock discontinuity rather than CPU occupancy.
Invariant: Detection and mitigation must have justified independence from the monitored failure mode.
Containment: Use the approved independent protection path when control progress is not established.
Mitigation risk: A very short timeout can create repeated nuisance stops and unsafe recovery pressure.
Correction: Provide bounded independent execution and monitor meaningful progress with a monotonic time base.
Prevention: Include watchdog dependency analysis in architecture review.
Verification: Stall each monitored dependency separately and measure detection-to-mitigation behavior.
Negative result: One forced process crash does not cover deadlock, stale progress or clock failures.
Escalation: Safety owner determines required independence; this sheet does not certify an implementation.
Boundary: [ROS Jazzy real-time guidance](https://docs.ros.org/en/jazzy/Tutorials/Demos/Real-Time-Programming.html) warns about nondeterministic blocking, not a universal safe watchdog design.
Evidence to retain: Executor layout, clock source, progress ID and independent action timing.

### P07-021 — Similar aisles create a convincing wrong SLAM match
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; SLAM/localisation course commitment.
Related home: [module 39](../39_robot_perception_navigation_manipulation_and_vla/INDEX.md).
Trigger: Repeated shelves or visually similar corridors produce a strong-looking but incorrect loop closure.
Symptom: The map or global robot pose jumps to another structurally similar location.
Competing cause A: Perceptual aliasing accepts an incorrect place association.
Competing cause B: A true loop closure reveals accumulated odometry drift.
Competing cause C: Frame identity or map revision mismatches create an apparent jump.
Discriminator: Check geometric consistency, independent landmarks and trajectory history across competing place hypotheses.
Expected under A: Local appearance matches but broader geometry or motion history contradicts the association.
Expected under B: Independent observations support the correction and global consistency improves.
Expected under C: The discrepancy tracks a namespace or map-version boundary.
Invariant: A high descriptor similarity is evidence for a candidate, not proof of physical identity.
Containment: Restrict motion or defer global correction while ambiguity remains consequential.
Mitigation risk: Rejecting every large correction can preserve substantial genuine drift.
Correction: Require appropriate geometric and temporal evidence before accepting a place association.
Prevention: Include repeated-structure environments and ambiguous routes in evaluation.
Verification: Replay competing loop candidates with independent ground truth or disambiguating observations.
Negative result: A lower optimization residual can occur for a confidently wrong correspondence set.
Escalation: Localization owner defines ambiguity handling and recovery route.
Boundary: A road-network shortest path is not a SLAM validation oracle.
Evidence to retain: Candidate matches, map revision, transform changes, trajectory and independent landmarks.

### P07-022 — A point path is not a swept-volume guarantee
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; navigation-stack and manipulation course commitments.
Related home: [module 39](../39_robot_perception_navigation_manipulation_and_vla/INDEX.md).
Trigger: Navigation uses an unloaded or point-like footprint while the robot carries an extended payload.
Symptom: The base center follows a collision-free path but a load or arm strikes the environment.
Competing cause A: The collision model omits the current footprint or swept geometry.
Competing cause B: Localization uncertainty places the actual robot outside the planned corridor.
Competing cause C: The environment changed after the last valid observation.
Discriminator: Overlay time-indexed actual geometry, pose uncertainty and obstacle observation age.
Expected under A: Collision is predicted when the correct footprint is substituted at the same pose.
Expected under B: Nominal geometry clears but plausible actual poses do not.
Expected under C: The colliding obstacle is absent or stale in the planner's world model.
Invariant: Feasibility refers to the complete configuration and uncertainty envelope used at execution.
Containment: Stop or restrict motion under the approved load/configuration rule.
Mitigation risk: Arbitrary global inflation can make the map infeasible without representing the actual articulated shape.
Correction: Version load, arm configuration and footprint with the relevant planning contract.
Prevention: Require explicit checks when payload or articulation changes.
Verification: Test narrow passages, turns and simultaneous articulation with physical clearance measurement.
Negative result: Static footprint clearance does not prove dynamic braking clearance.
Escalation: Planning and safety owners resolve geometry uncertainty and stopping margins.
Boundary: A path valid for one configuration is not automatically valid for another.
Evidence to retain: Footprint revision, payload dimensions, trajectory, observations and execution pose.

### P07-023 — Grasp confidence is not a contact-stability measurement
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; grasping and manipulation course commitment.
Related home: [module 39](../39_robot_perception_navigation_manipulation_and_vla/INDEX.md).
Trigger: A learned grasp score is used outside the object's pose, material or friction distribution.
Symptom: The selected grasp looks confident in images but slips during lift or transfer.
Competing cause A: Pose uncertainty moves contacts away from the predicted stable region.
Competing cause B: Material/friction or mass differs from the assumed contact model.
Competing cause C: Gripper calibration or force command does not produce the intended contact.
Discriminator: Compare independently measured pose, contact/force observations and failure phase.
Expected under A: Failures cluster near geometric tolerances or occluded pose directions.
Expected under B: Initial contact succeeds but lift/acceleration causes material-dependent slip.
Expected under C: Commanded and observed jaw position/force disagree.
Invariant: A grasp score is calibrated only to its defined outcome and operating distribution.
Containment: Use the approved low-risk regrasp or reject path rather than repeated aggressive squeezing.
Mitigation risk: More force can damage the object or create a different hazard.
Correction: Evaluate grasp selection with physical outcome, uncertainty and contact constraints.
Prevention: Include materials, masses, occlusions and transfer accelerations in the scenario ledger.
Verification: Measure successful retained transfers, not only initial closure or image alignment.
Negative result: A successful static lift does not establish stability during transport.
Escalation: Manipulation owner defines allowable force and object-damage limits.
Boundary: Offline ranking quality does not establish physical grasp reliability.
Evidence to retain: Object identity, pose estimate/uncertainty, gripper revision, contact trace and failure phase.

### P07-024 — Language-conditioned intent cannot authorize arbitrary action
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; VLA-policy and behavior-integration course commitments.
Related home: [module 39](../39_robot_perception_navigation_manipulation_and_vla/INDEX.md).
Trigger: A vision-language-action policy's output bypasses local validity, workspace or authority checks.
Symptom: A semantically plausible instruction leads to an action outside the approved envelope.
Competing cause A: The policy misunderstood the instruction or visual scene.
Competing cause B: The instruction itself requests an unauthorized or physically unsafe action.
Competing cause C: Action normalization or coordinate conversion changes a valid policy output into an invalid command.
Discriminator: Preserve instruction, observation, normalized action, denormalized action and gate decision as separate records.
Expected under A: The proposed action conflicts with the authorized task semantics before execution conversion.
Expected under B: The request is disallowed even if the policy understood it correctly.
Expected under C: The action crosses the envelope only after scaling or frame conversion.
Invariant: Semantic plausibility does not replace independent local action and authority constraints.
Containment: Reject the action and enter the task-appropriate approved response.
Mitigation risk: Automatically asking the same policy to judge its action is not independent enforcement.
Correction: Enforce bounded, versioned action contracts outside the learned policy.
Prevention: Evaluate adversarial instructions, unusual scenes and conversion boundaries separately.
Verification: Require every forbidden test action to be rejected without relying on the policy's self-report.
Negative result: A safe benchmark prompt set does not establish robustness to all instructions or scene changes.
Escalation: Task owner clarifies allowed intent; safety owner defines physical constraints.
Boundary: This is a design principle, not a claim that a particular VLA implementation is safe or unsafe.
Evidence to retain: Policy/action schema revisions, instruction provenance, observations and gate reasons.

### P07-025 — Repeated behavior-tree ticks can repeat a physical effect
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; behavior-tree integration course commitment.
Related home: [module 39](../39_robot_perception_navigation_manipulation_and_vla/INDEX.md).
Trigger: An asynchronous action times out and a later tree tick issues the same physical operation as a new request.
Symptom: A robot grips, dispenses, opens or moves twice despite one intended task.
Competing cause A: The client created a new effect identity after losing the acknowledgement.
Competing cause B: The action server forgot durable completion state after restart.
Competing cause C: Two branches concurrently control the same actuator.
Discriminator: Compare task/effect IDs, server state transitions and authority ownership across the retry.
Expected under A: One logical task maps to two distinct effect IDs.
Expected under B: The same effect ID is executed again after a state-loss epoch.
Expected under C: Overlapping commands originate from separate branches with no exclusive gate.
Invariant: Repeated delivery or polling must not imply repeated consequential execution.
Containment: Resolve physical state before replaying an ambiguous completed action.
Mitigation risk: Blind cancellation can interrupt a load-bearing or otherwise hazardous intermediate state.
Correction: Use explicit action identity, terminal-state semantics and exclusive actuator ownership.
Prevention: Define behavior-tree halt, restart, timeout and recovery contracts.
Verification: Lose acknowledgements and restart components at each execution boundary.
Negative result: One successful retry does not establish idempotence across process failure.
Escalation: The action owner determines whether reconciliation, compensation or manual inspection is required.
Boundary: Transport reliability does not itself guarantee exactly-once physical effects.
Evidence to retain: Tree state, effect ID, authority epoch, acknowledgement semantics and observed actuator state.

### P07-026 — Synthetic variants can leak the same underlying scene
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; synthetic-data pipeline and evaluation course commitments.
Related home: [module 40](../40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md).
Trigger: Images from the same scene or trajectory are split randomly after rendering different seeds.
Symptom: Held-out synthetic performance is excellent but new warehouse layouts perform poorly.
Competing cause A: Shared latent scene identity leaked across partitions.
Competing cause B: Real-world rendering characteristics differ despite genuinely unseen scenes.
Competing cause C: Annotation or task definitions differ between synthetic and real evaluation.
Discriminator: Compare a scene/trajectory-group holdout with the random-image split and audit label contracts.
Expected under A: Performance drops substantially on genuinely unseen scene identities even within simulation.
Expected under B: Synthetic group holdout remains strong but real data still differs.
Expected under C: Error concentrates where label semantics change, such as occluded-object treatment.
Invariant: Split independence is defined at the latent source level relevant to generalization.
Containment: Stop using the leaking score as evidence for field readiness.
Mitigation risk: Holding out too few scenes can yield high uncertainty even when frame count is large.
Correction: Partition scenes, assets, trajectories and relevant source identities before generating variants.
Prevention: Preserve generation lineage and prohibit cross-partition reuse of restricted identities.
Verification: Re-evaluate untouched groups with confidence accounting at the group level.
Negative result: A new random seed is not proof of an independent scene.
Escalation: Evaluation owner chooses the generalization unit that matches the deployment decision.
Boundary: Pixel dissimilarity alone does not establish statistical independence.
Evidence to retain: Scene graph IDs, asset lineage, trajectories, seed, labels and partition manifest.

### P07-027 — A stable simulator timestep can hide unstable physical behavior
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; simulation-environment and sim-to-real course commitments.
Related home: [module 40](../40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md).
Trigger: Contact, actuator or controller integration is evaluated at one convenient simulation timestep.
Symptom: The simulated robot is stable while hardware chatters or overshoots.
Competing cause A: Numerical integration or contact approximation changes the apparent dynamics.
Competing cause B: The simulator's actuator model omits real delay, saturation or compliance.
Competing cause C: Hardware timing differs from the controller update period assumed in simulation.
Discriminator: Run a timestep/refinement study and compare isolated real actuator responses under qualified procedures.
Expected under A: Important behavior changes as the integration timestep is refined.
Expected under B: Simulation converges numerically but still disagrees with the measured plant response.
Expected under C: Hardware timestamps reveal a different update/delay pattern.
Invariant: Numerical convergence and physical model validity are separate requirements.
Containment: Do not promote the policy based solely on the unstable comparison.
Mitigation risk: Smaller timesteps can increase compute load and alter simulated real-time scheduling.
Correction: Resolve numerical sensitivity and identify the missing physical mechanism separately.
Prevention: Document timestep, solver, contact settings and controller scheduling in every evaluation manifest.
Verification: Check convergence of the decision-relevant quantities, not only visual appearance.
Negative result: Agreement between two simulators sharing the same model is not independent physical evidence.
Escalation: Simulation and controls owners agree which measured plant behaviors must be reproduced.
Boundary: A more detailed simulator is not automatically a more accurate one.
Evidence to retain: Timestep, solver settings, response traces, control period and physical identification data.

### P07-028 — Independent randomization can generate impossible robots
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; domain-randomisation course commitment.
Related home: [module 40](../40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md).
Trigger: Mass, inertia, dimensions, friction and actuator limits are randomized independently without physical constraints.
Symptom: Training is difficult and the resulting policy is brittle despite broad parameter ranges.
Competing cause A: The synthetic joint distribution contains physically inconsistent combinations.
Competing cause B: The intended real uncertainty range is genuinely too broad for one policy.
Competing cause C: Optimization fails even on plausible narrow distributions.
Discriminator: Audit sampled parameter tuples against physical relationships and evaluate structured subranges.
Expected under A: Many sampled combinations violate the modeled object's geometry or feasible actuation.
Expected under B: Plausible subranges require materially different behavior even with sound optimization.
Expected under C: Training remains poor on a validated narrow distribution.
Invariant: Randomization should represent a justified joint uncertainty model, not arbitrary independent intervals.
Containment: Remove invalid samples without claiming that narrower training proves transfer.
Mitigation risk: Over-narrowing can exclude the actual deployment condition.
Correction: Use constrained or hierarchical randomization with traceable parameter relationships.
Prevention: Review parameter distributions with hardware and domain owners before large training runs.
Verification: Evaluate the actual identified real parameter envelope and deliberately held-out plausible combinations.
Negative result: A broad marginal histogram does not establish coverage of important joint combinations.
Escalation: If real uncertainty is unknown, collect identification evidence rather than label arbitrary ranges robust.
Boundary: Domain randomization is not a universal substitute for system identification.
Evidence to retain: Joint sampler, constraints, physical source data and per-scenario performance.

### P07-029 — A digital twin cannot validate itself on its fitting data
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; digital-twin course commitment.
Related home: [module 40](../40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md).
Trigger: The twin is continuously tuned to recent measurements and judged only on the same measurements.
Symptom: Reported agreement is excellent but predictions under a changed operating condition are wrong.
Competing cause A: Evaluation reuses fitting data and measures adaptation rather than prediction.
Competing cause B: Sensors share a calibration bias that both fitting and evaluation inherit.
Competing cause C: The new operating condition excites an unidentifiable parameter direction.
Discriminator: Freeze the twin and evaluate independent time/condition segments with independent reference where possible.
Expected under A: Agreement degrades after the fit/evaluation boundary is enforced.
Expected under B: Both twin and primary sensor disagree with the independent reference.
Expected under C: Different parameter settings fit old data but predict different new responses.
Invariant: Validation evidence must be independent of the tuning decision it evaluates.
Containment: Restrict the twin's operational authority to the conditions supported by independent evidence.
Mitigation risk: Freezing adaptation too long can allow genuine process drift to accumulate.
Correction: Separate identification, adaptation monitoring and prospective validation protocols.
Prevention: Maintain parameter histories and explicit data cutoffs for every claimed prediction.
Verification: Test held-out interventions or operating conditions chosen to challenge identifiability.
Negative result: Low in-sample residual is not evidence of causal or extrapolative validity.
Escalation: The domain owner decides whether the twin supports advisory analysis or consequential control.
Boundary: A visually accurate digital replica can have an inaccurate dynamics model.
Evidence to retain: Parameter snapshot, fitting window, withheld window, reference source and intervention history.

### P07-030 — Episode count is not hazard coverage
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; scenario-matrix and staged-validation course commitments.
Related home: [module 40](../40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md).
Trigger: Thousands of episodes sample common nominal routes but never exercise a rare consequential condition.
Symptom: Aggregate success is high while a single untested scenario invalidates field readiness.
Competing cause A: The scenario generator assigns negligible probability to the hazard cell.
Competing cause B: The hazard occurs but success aggregation hides its severity.
Competing cause C: The simulator omits the physical mechanism entirely.
Discriminator: Audit exposure and outcomes by hazard-relevant scenario cell, not only total episodes.
Expected under A: The important cell has zero or very low exposure.
Expected under B: The cell has failures but its aggregate weight is small.
Expected under C: No generator configuration can express the relevant mechanism.
Invariant: A readiness claim names the covered operational domain and excluded hazards.
Containment: Keep the untested condition outside the authorized trial envelope.
Mitigation risk: Oversampling rare cells improves diagnosis but changes the interpretation of the pooled rate.
Correction: Use stratified scenario coverage and separately report deployment-weighted and stress-test results.
Prevention: Review scenario gaps with operations and safety owners before collecting more nominal episodes.
Verification: Require explicit evidence for each material trial-envelope condition and its recovery path.
Negative result: Zero failures in a finite sample does not prove zero failure probability.
Escalation: Unmodeled hazards require physical or alternative evidence, not more identical simulations.
Boundary: Simulation exposure does not automatically equal independent field exposure.
Evidence to retain: Scenario taxonomy, sampling distribution, group IDs, counts, outcomes and excluded mechanisms.

### P07-031 — Remote connectivity is not exclusive command authority
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; remote-operations and fleet-orchestration course commitments.
Related home: [module 41](../41_edge_robotics_safety_fleet_management_and_operations/INDEX.md).
Trigger: Two operator sessions reconnect after network loss and both retain plausible authority state.
Symptom: The robot receives conflicting commands or resumes an old command stream.
Competing cause A: Authority leases overlap or are not checked locally at execution.
Competing cause B: A single owner is correct but old buffered commands replay after reconnect.
Competing cause C: Identity mapping assigns two sessions to the same indistinguishable operator ID.
Discriminator: Compare local lease epoch, actor identity, command issue time and execution time.
Expected under A: Commands from two authority epochs are accepted concurrently.
Expected under B: One authority epoch accepts commands whose age exceeds the contract.
Expected under C: Audit identities collapse distinct sessions despite distinct credentials or transport connections.
Invariant: A local gate accepts only the current authorized identity/epoch and valid command age.
Containment: Revoke ambiguous authority and use the approved local condition before reassignment.
Mitigation risk: Abrupt authority removal can interrupt a task requiring maintained control.
Correction: Define exclusive leases, bounded expiry, reconnect rules and explicit handover acknowledgements.
Prevention: Separate authentication, authorization, freshness and actuator ownership in the protocol.
Verification: Partition the network, reconnect sessions and attempt stale command replay.
Negative result: TLS authentication alone does not establish exclusive or current control permission.
Escalation: Operations owner resolves handover; security owner reviews identity compromise suspicion.
Boundary: The correct lease timeout depends on the hazard and network envelope.
Evidence to retain: Actor/session identity, lease epoch/expiry, local clock bound and command journal.

### P07-032 — Rollback must restore a compatible bundle, not one binary
Category: NUMERICAL_EDGE.
Provenance: HYPOTHETICAL; OTA and versioned-deployment course commitments.
Related home: [module 41](../41_edge_robotics_safety_fleet_management_and_operations/INDEX.md).
Trigger: Firmware rolls back while the model, preprocessing, calibration or protocol schema remains newer.
Symptom: Boot succeeds but perception, command interpretation or memory use is wrong.
Competing cause A: Components violate their compatibility constraints after partial rollback.
Competing cause B: Persistent state migration cannot be read by the old binary.
Competing cause C: The rollback image itself is corrupt or targeted to the wrong hardware.
Discriminator: Compare the installed full manifest with compatibility rules, state schema and verified target identity.
Expected under A: Individually valid components form a disallowed combination.
Expected under B: Binary/interface versions match but persistent-state decoding fails.
Expected under C: Hash, size, signature or hardware-target checks fail independently.
Invariant: Release and rollback units include every component whose semantics must agree.
Containment: Keep the robot out of normal motion until a supported complete state is established.
Mitigation risk: Erasing persistent state may destroy calibration, evidence or task reconciliation information.
Correction: Restore a tested compatible bundle with an explicit state-migration/recovery procedure.
Prevention: Maintain a machine-readable compatibility manifest and power-failure-safe update design.
Verification: Interrupt update and rollback at each persistent boundary on the actual supported hardware.
Negative result: A successful boot is not an end-to-end compatibility test.
Escalation: Release owner and local operator authorize recovery when migration is irreversible.
Boundary: Anti-rollback security policy and operational rollback need an explicitly reconciled design.
Evidence to retain: Installed component hashes, hardware ID, state schema, migration journal and gate results.

### P07-033 — Authentic software can still be behaviorally invalid
Category: CONCEPT_CONFUSION.
Provenance: HYPOTHETICAL; [Uptane 2.1.0](https://uptane.org/docs/latest/standard/uptane-standard) supports the scoped secure-update distinction.
Related home: [module 41](../41_edge_robotics_safety_fleet_management_and_operations/INDEX.md).
Trigger: A release is promoted solely because its package and metadata verify cryptographically.
Symptom: A properly signed image violates timing, operator support or physical-envelope requirements.
Competing cause A: The trusted build contains a functional defect.
Competing cause B: The correct image is assigned to an incompatible target or component bundle.
Competing cause C: Repository or key compromise produced apparently trusted metadata.
Discriminator: Separate signature/target/metadata validation from independent behavioral and compatibility gates.
Expected under A: Authenticity passes while the deterministic functional oracle fails.
Expected under B: Target or dependency constraints identify an otherwise valid image as inapplicable.
Expected under C: Trust-chain or publication audit evidence indicates unauthorized release activity.
Invariant: Authenticity, authorization, compatibility and operational correctness are separate predicates.
Containment: Stop promotion and preserve the signed artifact and trust evidence.
Mitigation risk: Disabling signature checks to recover functionality creates a new update-security failure.
Correction: Fix the failed predicate without bypassing the others.
Prevention: Require both secure delivery checks and application-specific release gates.
Verification: Test valid-but-incompatible and valid-but-functionally-wrong fixtures as separate negative cases.
Negative result: A trusted signature does not establish source-code correctness or supply-chain integrity.
Escalation: Security response owns compromise investigation; functional owners own behavior correction.
Boundary: Uptane's stated out-of-scope items include compromise of packaged software and build supply chain.
Evidence to retain: Signed metadata, verified target, build provenance, gate reports and release authority.

### P07-034 — Fleet reliability can improve when failed robots disappear
Category: SCALE_FAILURE.
Provenance: HYPOTHETICAL; SLI denominator reasoning sourced to the SRE workbook methodology.
Related home: [module 41](../41_edge_robotics_safety_fleet_management_and_operations/INDEX.md).
Trigger: Task success is computed only from reports received by a central service.
Symptom: The dashboard success ratio rises during a connectivity outage.
Competing cause A: Offline robots and failed attempts vanish from the denominator.
Competing cause B: Dispatch legitimately stops admitting new work under degraded conditions.
Competing cause C: Duplicate success reports inflate the numerator.
Discriminator: Reconcile dispatch eligibility, local attempt journals and central completion reports by task ID.
Expected under A: Known eligible attempts lack terminal central records and are silently excluded.
Expected under B: Admission logs explain fewer eligible tasks while previously admitted work remains accounted for.
Expected under C: Multiple reports map to one eligible task and are counted repeatedly.
Invariant: Good outcomes are a subset of a complete explicitly defined eligible population.
Containment: Mark the measurement incomplete instead of reporting an unsupported reliability increase.
Mitigation risk: Treating every missing report as failure may be conservative but must not masquerade as observed outcome.
Correction: Reconstruct denominator and unknown-outcome states with event-time reconciliation.
Prevention: Store admission evidence independently of successful completion telemetry.
Verification: Disconnect a known subset and reconcile every admitted task after recovery.
Negative result: A stable heartbeat ratio does not prove task-level service quality.
Escalation: Service owner determines operational policy while measurement owner reports uncertainty explicitly.
Boundary: Safe refusal and service success need separate metrics; neither should be hidden.
Evidence to retain: Admission ledger, task IDs, local terminal state, central ingestion time and unknown count.

### P07-035 — Near-miss chronology needs uncertainty, not just sorted logs
Category: VIOLATED_ASSUMPTION.
Provenance: HYPOTHETICAL; incident-response and robot-observability course commitments.
Related home: [module 41](../41_edge_robotics_safety_fleet_management_and_operations/INDEX.md).
Trigger: A near miss is reconstructed by sorting timestamps from unsynchronized robot, camera and fleet clocks.
Symptom: Different log exports imply contradictory command and obstacle orderings.
Competing cause A: Clock offset/drift reverses apparent event order.
Competing cause B: Buffering delays report events out of physical order.
Competing cause C: Reset epochs or overwritten ring buffers erase part of the causal chain.
Discriminator: Preserve raw clocks, synchronization bounds, sequence IDs and ingest times before normalizing chronology.
Expected under A: Event-time uncertainty intervals overlap or reverse after mapping.
Expected under B: Event sequence is stable while ingestion order differs.
Expected under C: Sequence gaps or epoch changes identify unavailable evidence.
Invariant: A causal assertion cannot be more precise than the evidence's clock and identity bounds.
Containment: Preserve volatile evidence and apply the approved physical restriction before restart overwrites it.
Mitigation risk: Excessive evidence collection can expose sensitive imagery or interfere with safe recovery.
Correction: Build a partial-order timeline with explicit unknown intervals and corroborating physical observations.
Prevention: Predefine bounded ring retention, incident freeze and access-controlled evidence export.
Verification: Conduct a qualified evidence-capture exercise with known event order and clock perturbation.
Negative result: Agreement between two logs sharing one bad clock is not independent confirmation.
Escalation: Incident owner separates established facts, contributors and unresolved hypotheses before release decisions.
Boundary: An illustrative timeline is not an observed incident and must never be presented as one.
Evidence to retain: Raw logs, hashes, clock mappings, access history, sequence epochs and declared gaps.

**Takeaway:** A complete issue distinguishes hypotheses and specifies what observation would change the action.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let n be the number of retained boundary events in a diagnostic window.
Let b be the encoded bytes per event, excluding storage-system overhead.
Let k be the number of distinct trace keys concurrently retained.
Let r be the number of robots included in a fleet reconciliation.
A single-pass invariant check takes O(n) time when each event is processed once.
A hash-based trace join takes expected O(n) time under ordinary hashing assumptions.
That join retains O(k) active state plus any retained raw input.
Adversarial collisions or an unbounded key lifetime invalidate the simple expected-cost model.
Sorting unsorted events takes O(n log n) comparison time and implementation-dependent auxiliary space.
Sorting alone does not solve missing clocks or causal ambiguity.
A bounded per-robot ring retains O(capacity) bytes regardless of runtime duration.
Its bounded storage necessarily discards older evidence unless another retention path exists.
A complete fleet denominator reconciliation must inspect admitted tasks, not only successful events.
For fixed records per robot, ingest work scales with r; correlated reconnection can create a burst.
A diagnostic's operational cost includes perturbation of the measured system.
High-volume synchronous logs can introduce the very deadline failure being investigated.
Prefer bounded event records and sampled detail under an explicitly stated capture policy.
Do not sample away rare terminal-state transitions needed for denominator or authority correctness.
Sampling raw sensor payloads and preserving every decision boundary are different policies.

### 9b. Back-of-Envelope Sizing

**Evidence:** ILLUSTRATIVE sizing, not a measured telemetry benchmark.
**Workload:** Twenty robots, each emitting fifty boundary records per second.
**Record size:** 160 encoded bytes per event before transport and storage overhead.
**Retention window:** Fifteen minutes of local bounded evidence.
**Concurrency assumption:** All twenty robots are active throughout the window.
**Computation:** One robot emits 50×160=8000 bytes/s.
**Local payload:** 8000×900=7200000 bytes per robot for fifteen minutes.
**Local conversion:** 7200000/1048576≈6.866455 MiB.
**Fleet payload:** 20×7200000=144000000 bytes.
**Fleet conversion:** 144000000/1048576≈137.329102 MiB.
**Independent check:** 20 robots×50 records/s×900 s=900000 records.
**Byte check:** 900000 records×160 bytes/record=144000000 bytes.
**Excluded:** Filesystem metadata, indexes, encryption, replication and raw camera payloads.
**Sensitivity:** Doubling record rate doubles payload to 288000000 bytes.
**Sensitivity:** Adding a 1024-byte image crop to every record changes the payload assumption fundamentally.
**Decision:** Do not size boundary-event storage as though it contains full raw observations.
**Bottleneck to measure:** Storage write latency and CPU perturbation at synchronized reconnect.
**Recovery condition:** Evidence upload must not monopolize the control network or actuator CPU.
**Retention risk:** Fifteen minutes can be insufficient if incident detection or response is slower.
**Privacy risk:** More retention can increase access and sensitive-data obligations.
**Currency cost:** Not estimated; no dated hardware, cloud or labor price source was obtained for this constructed workload.
**Useful outcome:** A diagnosis that separates a material hypothesis, not simply a larger log archive.

### Diagnostic collection budget

Capture raw observation time rather than reconstructing it from receive time.
Record reset epoch because monotone counters can restart.
Record schema and model revision because identical bytes may have changed meaning.
Record rejection reason because absence of output is otherwise ambiguous.
Record authoritative command acceptance separately from physical completion.
Use bounded reason vocabularies to avoid unbounded metric cardinality.
Keep sensitive payload behind access controls rather than in public metric labels.
Require an owner for retention changes and incident freezes.
A full device image can contain credentials and is not automatically an appropriate debug artifact.
Evidence hashes detect later byte changes but do not establish that the original capture was truthful.
An independent timing observer can help quantify trace perturbation.
That observer also has a clock, capture delay and resolution that must be disclosed.

**Takeaway:** Size diagnostic evidence and its perturbation, not only the inference workload.

## 10. Key Theorems / Results Reference

| Result or guarantee | Supported statement | Required conditions | Does not imply |
|---|---|---|---|
| Constant-deceleration identity | D1 follows from sequential constant-speed delay and constant deceleration | fixed positive deceleration and modeled timing | certified stop or moving-obstacle clearance |
| Conservation of blocks | B1 follows from complete admission/consumption accounting | one epoch, no uncounted overwrite or loss | physical acquisition completeness |
| Telescoping time differences | Adjacent paired stage intervals sum to one trace's total | shared mapped clock, same causal path | additivity of component percentiles |
| SLI ratio identity | good/eligible is a proportion in [0,1] | positive denominator and numerator subset | complete eligible-population measurement |
| REP 103 convention | ROS uses documented SI/right-handed and suffix-frame conventions | conventions adopted or exceptions stated | a particular transform is physically calibrated |
| REP 105 semantics | odom is continuous but may drift; map may correct discontinuously | implementation follows the specified semantics | bounded actual estimator error |
| Secure update verification | Uptane defines metadata/image verification procedures | conformant implementation and threat assumptions | functional correctness of signed software |
| Numerical convergence check | Refinement can reveal timestep dependence | meaningful quantities and justified convergence study | physical validity of the model |

No theorem here establishes an application's safety.
No source here supplies measured latency, energy or outcome for the constructed scenarios.
The constant-deceleration derivation is a model consequence, not an empirical braking characterization.
The conservation equation fails as a diagnostic if its counters omit the failing participant.
A source's documented example run cannot be relabeled as this collection's experiment.
The ROS real-time tutorial explicitly discusses nondeterministic operations in timing-critical paths.
It also limits its demonstration environment; its commands are not portable verification instructions for every host.
Use its guidance to ask the right timing questions, not to claim an application-specific bound.
The update standard's ground-vehicle framing must be evaluated before applying it to another device class.
A secure-delivery framework does not define the robot's physical trial approval process.
The dataset source describes recordings and annotations, not the performance of an authored algorithm.
Dataset-specific records can share subjects; row-level splits do not guarantee subject independence.

### Primary source registry

| ID | Primary source | Version/date | Exact supported scope | Accessed | Limitation and trigger |
|---|---|---|---|---|---|
| S-01 | [ROS 2 understanding real-time programming](https://docs.ros.org/en/jazzy/Tutorials/Demos/Real-Time-Programming.html) | Jazzy documentation; page publication date not stated | deadlines, jitter and nondeterministic operation examples | 2026-11-08 | not application timing proof; recheck distribution/execution changes |
| S-02 | [REP 103](https://www.ros.org/reps/rep-0103.html) | created 2010-10-07; post-history 2014-12-31 | SI units, handedness, body/optical axes and representation conventions | 2026-11-08 | informational conventions; recheck interface exceptions |
| S-03 | [REP 105](https://www.ros.org/reps/rep-0105.html) | created 2010-10-27 | map/odom/base frame semantics and authorities | 2026-11-08 | not accuracy evidence; recheck frame ownership changes |
| S-04 | [Uptane Standard](https://uptane.org/docs/latest/standard/uptane-standard) | 2.1.0; publication date not extracted | update threat model, metadata roles and stated out-of-scope items | 2026-11-08 | not build or behavioral correctness; recheck standard version |
| S-05 | [SRE workbook: Implementing SLOs](https://sre.google/workbook/implementing-slos/) | online chapter; page date not extracted | user-centered good-events/total-events SLI methodology | 2026-11-08 | not a robot safety standard; recheck metric contract |
| S-06 | [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) | 1.0.0, published 2005-02-24 | dataset provenance, sampling and usage limitations | 2026-11-08 | not clinical device validation; recheck dataset/annotation revision |
| S-07 | [Local inventory](../../../Templates/directory.txt) | supplied visible tree | exact visible topic names at lines 1027–1128 | 2026-11-08 | original bodies unavailable; recheck source additions |

Source access is not an independent substantive review of this entire file.
The unavailable original lessons remain an explicit evidence gap.
No hardware was available through the supplied inventory itself.
No physical stopping, energy, calibration or fault-injection run is claimed.
No API support matrix is inferred from a runtime's name.
Review each issue when its corresponding source or boundary contract changes.

**Takeaway:** A reference supports its stated scope; it cannot transfer evidence maturity to an untested system.

## 11. Interview / Exam Questions

### Q01 — Recall: What makes an observation fresh?
Prompt: A model completes in 12 ms; its input frame was captured 300 ms earlier.
Clarification: Both timestamps are mapped to one clock with an adequately small known error.
Answer: Freshness is evaluated against observation-to-use age, not execution duration.
Reasoning: The physical state was observed at capture, not when the model began.
Calculation: The information age is at least 300 ms at completion under the stated timing.
Decision: Compare that age with the task's justified freshness contract.
Alternative: A model predicting forward to use time may be acceptable only with a validated prediction/uncertainty contract.
Nonexample: Relabeling the receive time as capture time does not make the scene newer.
Follow-up: Add an unknown 50 ms clock offset; the age bound must include that uncertainty.
Red flag: Calling the output 12 ms old because the kernel is fast.
Scoring 0: Confuses inference latency with observation age.
Scoring 1: Names staleness without defining the causal interval.
Scoring 2: Uses the correct age and states the contract.
Scoring 3: Propagates clock uncertainty and explains the prediction alternative.
Answer support: T1, Section 5 and P07-015.

### Q02 — Recall: Why can map and odom disagree without a bug?
Prompt: Global pose jumps by 0.4 m while local odometry remains smooth.
Clarification: The estimator follows REP 105 and accepted a legitimate localization correction.
Answer: Map supplies a globally corrected reference that may be discontinuous; odom supplies a continuous local reference that may drift.
Reasoning: The two frames prioritize different properties rather than duplicate one perfect state.
Check: Inspect map-to-odom and odom-to-base separately around the correction.
Decision: Do not differentiate the map jump as physical velocity for local control.
Alternative: Another documented frame architecture can be defensible if every consumer's assumptions are explicitly migrated.
Nonexample: Two independent publishers fighting over one transform is not the intended disagreement.
Follow-up: If odom jumps too, investigate reset epochs or authority changes.
Red flag: Smoothing every jump without identifying its source and semantics.
Scoring 0: Treats every disagreement as sensor noise.
Scoring 1: Knows names but not continuity properties.
Scoring 2: Explains the distinct semantics and controller consequence.
Scoring 3: Proposes a discriminating transform/epoch trace.
Answer support: P07-017 and S-03.

### Q03 — Recall: What does a valid signature establish?
Prompt: An authentic signed model bundle causes target deadline misses.
Clarification: Signature and hash verification succeeded under the selected trust policy.
Answer: Verification establishes the relevant authenticated artifact/metadata relationship, not application timing correctness.
Reasoning: A trusted publisher can sign an incorrect or incompatible artifact.
Check: Keep authenticity, target authorization, compatibility and behavioral gates separate.
Decision: Block rollout on the failed timing gate without weakening signature checks.
Alternative: A fully validated older compatible bundle may be the recovery choice if rollback policy permits it.
Nonexample: Transport encryption alone does not establish the artifact's complete release provenance.
Follow-up: If the signing key may be compromised, initiate the security response as a separate hypothesis.
Red flag: Disabling verification to install a conveniently working unsigned image.
Scoring 0: Equates authenticity with functional safety.
Scoring 1: States the distinction but proposes no gate separation.
Scoring 2: Identifies the failed behavior gate and preserves trust checks.
Scoring 3: Handles compatible rollback and compromise investigation independently.
Answer support: P07-032, P07-033 and S-04.

### Q04 — Apply: Compute the simplified stopping distance.
Prompt: Speed is 1.2 m/s, delay is 150 ms and deceleration magnitude is 0.8 m/s².
Clarification: Use D1's stationary-obstacle constant-deceleration model; ignore additional margins only for this exercise.
Answer: Delay travel is 0.18 m and braking travel is 0.90 m, totaling 1.08 m.
Reasoning: The intervals are sequential, so the distances add.
Check: 0.90 m×1.6 m/s² equals 1.44 m²/s², matching speed squared.
Decision: A 0.95 m clearance fails even this simplified model.
Alternative: Numerical integration is appropriate for a documented nonconstant brake model, but no such input is given here.
Nonexample: Adding 150 directly to a distance mixes milliseconds and metres.
Follow-up: At 0.6 m/s² deceleration, total becomes 0.18+1.20=1.38 m.
Red flag: Claiming the passing condition would certify safe stopping.
Scoring 0: Wrong units or wrong interval.
Scoring 1: Correct structure with arithmetic error.
Scoring 2: Correct 1.08 m and scoped conclusion.
Scoring 3: Gives the sensitivity and physical-model limitations.
Answer support: D1 and Section 5.

### Q05 — Apply: Diagnose an apparently healthy eight-block ring.
Prompt: Admissions are 120 blocks, consumption is 113 blocks and hardware reports two overwritten blocks before admission.
Clarification: Counters cover one reset epoch; overwritten blocks never entered the software queue.
Answer: Software occupancy is seven blocks, but that does not establish loss-free acquisition.
Reasoning: B1 accounts for admitted items only; the hardware loss occurred outside those counters.
Check: Reconcile hardware sequence with admitted sequence to expose the two missing blocks.
Decision: Mark affected signal continuity invalid and investigate producer/consumer ownership or throughput.
Alternative: A deliberate documented drop policy may be valid for some consumers but must surface the gap.
Nonexample: Reporting seven occupied blocks as proof that no sample was lost.
Follow-up: Add a second reader; reuse now requires both readers' ownership release.
Red flag: Enlarging the queue without checking whether overwrite occurs before enqueue.
Scoring 0: Denies loss because occupancy is below eight.
Scoring 1: Suspects loss without locating the accounting boundary.
Scoring 2: Computes occupancy and distinguishes pre-admission loss.
Scoring 3: Explains epoch, full/empty convention and multi-reader ownership.
Answer support: B1 and P07-001.

### Q06 — Apply: Size diagnostic payload retention.
Prompt: Twenty robots emit fifty 160-byte events per second for fifteen minutes.
Clarification: Estimate payload only; assume constant rates and no duplication.
Answer: There are 900000 events totaling 144000000 bytes, approximately 137.329 MiB.
Reasoning: Multiply robots, event rate and seconds; then multiply by bytes per event.
Check: One robot produces 7.2 million bytes; multiplying by twenty reproduces the fleet total.
Decision: Provision additional measured transport/storage overhead and bounded upload behavior.
Alternative: Per-robot local rings may reduce central dependence but not total raw payload under these assumptions.
Nonexample: Treating 160 bytes as 160 bits understates payload eightfold.
Follow-up: Doubling event rate doubles payload, but may also increase control-path perturbation nonlinearly.
Red flag: Calling the payload estimate a measured storage requirement.
Scoring 0: Wrong units or time conversion.
Scoring 1: Correct order of magnitude without overhead scope.
Scoring 2: Correct bytes/MiB with assumptions.
Scoring 3: Explains retention, perturbation and reconnect burst consequences.
Answer support: Section 9b.

### Q07 — Apply: Identify a denominator failure.
Prompt: A fleet admits 1000 tasks, receives 970 good completions and loses terminal reports for 20 tasks.
Clarification: The remaining ten tasks have known bad outcomes; all 1000 are eligible.
Answer: Observed good/eligible is 970/1000=97%; twenty outcomes remain unknown rather than silently excluded.
Reasoning: Eligibility was determined at admission, not by later reporting success.
Check: 970 good+10 bad+20 unknown reconciles to 1000 admitted tasks.
Decision: Report outcome uncertainty alongside the fixed denominator.
Alternative: A conservative success bound can treat unknown tasks as not-good, but must be labeled a bound rather than observed failures.
Nonexample: 970/980≈98.98% hides missing tasks from the eligible population.
Follow-up: If some tasks were never admitted, distinguish demand loss from admitted-task success.
Red flag: Improving reliability by deleting disconnected robots from the report.
Scoring 0: Uses only reported successful tasks as denominator.
Scoring 1: Notices missingness without reconciling counts.
Scoring 2: Gives 97% and explicit unknown state.
Scoring 3: Separates service demand, admission and task-outcome metrics.
Answer support: S1 and P07-034.

### Q08 — Judgment: Why did a smaller model not run faster?
Prompt: Weight storage drops by 60%, but target latency changes less than 2% in an illustrative comparison.
Clarification: These figures are supplied exercise inputs, not a benchmark claim.
Answer: Inspect executed kernels, conversion overhead and non-model age before attributing the result.
Reasoning: Sparse or low-bit representation may not have a corresponding accelerated implementation.
Check: Compare resolved operator path and timing breakdown under identical workload and thermal conditions.
Decision: Choose the next optimization only after identifying the actual bottleneck.
Alternative: Storage reduction can still be valuable when flash or transfer bandwidth is the binding constraint.
Nonexample: Repeating that fewer parameters must imply proportionally fewer executed operations.
Follow-up: A custom kernel is proposed; require independent numerical and end-to-end budget evidence.
Red flag: Reporting theoretical operation savings as measured target acceleration.
Scoring 0: Assumes the benchmark is impossible.
Scoring 1: Names overhead generically.
Scoring 2: Distinguishes representation, execution and application bottlenecks.
Scoring 3: Defends a constraint-specific storage win despite no latency gain.
Answer support: P07-007 and P07-010.

### Q09 — Judgment: Resolve ambiguous sensor-fusion error.
Prompt: Camera/IMU residuals grow during fast turns and after thirty minutes of operation.
Clarification: Stationary spatial calibration appears plausible; no independent timing evidence is yet available.
Answer: Test clock offset/drift, rigid extrinsic error and mount flexibility as competing hypotheses.
Reasoning: Time misalignment grows with motion rate, while drift can accumulate with duration.
Check: Capture raw clock pairs and residuals across controlled motion and temperature conditions.
Decision: Avoid retuning measurement covariance as the first claimed correction.
Alternative: Mechanical inspection is appropriate when acceleration/load dependence suggests flex.
Nonexample: Stationary agreement alone cannot prove time alignment.
Follow-up: A synchronization reset removes the error temporarily; that strengthens but does not alone prove the clock hypothesis.
Red flag: Declaring one root cause from a single correlation.
Scoring 0: Picks a cause without a discriminating observation.
Scoring 1: Lists hypotheses without expected signals.
Scoring 2: Predicts distinct timing/spatial/mechanical signals.
Scoring 3: Proposes bounded containment and independent validation after correction.
Answer support: P07-011 and P07-013.

### Q10 — Judgment: Evaluate a synthetic-data success claim.
Prompt: A detector scores highly on different rendered images from scenes used in training.
Clarification: Different seeds alter lighting and camera pose but reuse scene/asset identities.
Answer: The evaluation measures variation within shared scenes, not necessarily generalization to unseen scenes.
Reasoning: Latent source identity can leak despite pixel differences.
Check: Partition scene/trajectory/asset groups before generation and evaluate untouched groups.
Decision: Narrow the claim until independent groups and real-domain evidence are available.
Alternative: Within-scene evaluation is valid for a clearly limited task if that is the actual deployment question.
Nonexample: New random seeds are not automatically independent test entities.
Follow-up: Group holdout passes; real-world transfer still needs sensor/physics/task-contract validation.
Red flag: Calling thousands of related rendered frames thousands of independent deployment trials.
Scoring 0: Accepts the generalization claim from seed differences alone.
Scoring 1: Says leakage without defining the grouping unit.
Scoring 2: Defines latent-group holdout and limited claim.
Scoring 3: Separates statistical independence, simulation fidelity and field readiness.
Answer support: P07-026 and P07-030.

### Q11 — Judgment: Handle an ambiguous action timeout.
Prompt: A remote grip command times out after the server may have completed the action.
Clarification: Repeating the grip can damage the object; current physical state is not yet known.
Answer: Reconcile effect identity and observed actuator state before issuing a new consequential action.
Reasoning: A missing acknowledgement is not evidence that execution did not happen.
Check: Inspect durable terminal state, authority epoch and independently observed jaw/load state.
Decision: Use the approved ambiguous-state procedure rather than blind retry.
Alternative: Retrying the same idempotent effect ID is defensible only when the server's persistence and semantics establish it.
Nonexample: A new request ID converts one ambiguous effect into a possible duplicate.
Follow-up: Server restart erased history; physical reconciliation or authorized intervention becomes necessary.
Red flag: Claiming reliable transport ensures exactly-once physical execution.
Scoring 0: Retries blindly.
Scoring 1: Avoids retry but lacks a recovery contract.
Scoring 2: Uses identity and physical reconciliation.
Scoring 3: Explains persistence failure, authority and safe interruption limits.
Answer support: P07-025 and P07-031.

### Q12 — Judgment: What would make this troubleshooting reference verified?
Prompt: Every issue has a plausible diagnostic and every file is long enough for the course requirement.
Clarification: No hardware runs or independent technical review were performed during authoring.
Answer: The scenarios remain hypothetical; length and plausibility do not establish reproduced behavior or publication readiness.
Reasoning: Evidence maturity follows actual checks and artifacts, not editorial completeness.
Check: For each claimed reproduction, require inputs, environment, operation, observed output and limitations.
Decision: Keep draft/review metadata honest and run the relevant numerical, navigation and domain checks during integration.
Alternative: Derived reasoning can be published as derived after appropriate review without pretending to be empirical.
Nonexample: Copying an external tutorial's output and labeling it a local run fabricates evidence.
Follow-up: One hardware test passes; retain its exact supported scope rather than certify all thirty-five issues.
Red flag: Converting missing evidence into N/A merely to close a checklist.
Scoring 0: Equates documentation with validation.
Scoring 1: Requests generic testing without scope.
Scoring 2: Distinguishes derived, hypothetical, reproduced and reviewed states.
Scoring 3: Names concrete artifacts and limits each conclusion to exercised evidence.
Answer support: Section 1, source registry and course authoring standards.

### Arithmetic and integration oracle inputs

The following commands are proposed checks for the integration owner, not executed verification results.
They are independent of any physical measurement or model benchmark.
The stopping oracle inputs are v=1.2 m/s, tau=0.15 s and a=0.8 m/s².
Expected decomposition is 0.18 m delay plus 0.90 m braking, total 1.08 m.
Expected wrong-path total using 20 ms is 0.924 m.
Expected missing delay distance is 0.156 m.
The telemetry oracle inputs are 20 robots, 50 events/s, 160 bytes/event and 900 seconds.
Expected event count is 900000 and expected payload is 144000000 bytes.
Expected MiB is approximately 137.3291015625.
The denominator oracle inputs are good=970, bad=10, unknown=20 and eligible=1000.
Expected conservation is 970+10+20=1000 and good/eligible=0.97.
The queue oracle inputs are admitted=120 and consumed=113 within one epoch.
Expected occupancy is seven admitted blocks; pre-admission loss remains outside that identity.
A structural integration check should confirm twelve numbered sections in order.
A content check should confirm thirty-five issue IDs with distinct mechanisms and complete diagnostics.
A navigation check should confirm actual module/collection links and final card placement.
A domain review should challenge safe-condition assumptions rather than only count rows.
An evidence review should ensure no hypothetical entry acquired an observed/reproduced label accidentally.
These checks do not authorize physical experiments outside a qualified controlled procedure.

**Takeaway:** Defensible diagnosis requires a predicted signal, a safe action and an evidence boundary.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Thirty-five hypothetical cross-boundary edge/robotics diagnostic issues |
| Core rule | Pair observation, state, frame, version and authority before interpreting an output |
| Key assumption | Boundary records truthfully identify one causal path and clock/reset epoch |
| Complexity | One-pass invariant checks O(n) events; bounded rings retain only their declared window |
| Use when | A local component passes but end-to-end physical or operational behavior fails |
| Avoid when | Used as safety certification, clinical guidance or evidence of an observed incident |
| Go deeper | [Module order and local collections](../INDEX.md), [case catalog](../Case_Studies/INDEX.md) |
| Diagnostic | Choose an observation with different expected signals under competing causes |
| Evidence | Derived/illustrative reasoning; no physical reproduction or independent review claimed |
