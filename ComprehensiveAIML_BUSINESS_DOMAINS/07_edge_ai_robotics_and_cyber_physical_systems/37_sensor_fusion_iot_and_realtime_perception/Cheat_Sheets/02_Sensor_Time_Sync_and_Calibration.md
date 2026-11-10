---
collection: "sensor-fusion-iot-and-realtime-perception"
file_id: "02"
topic: "Sensor time synchronization and calibration"
level: 2
status: "draft; unreviewed"
version: "1.1"
created: "2026-11-08"
reviewed: null
review_trigger: "Clock source, transport, mounting, firmware, unit convention, sensor or intended action changes"
notation_style: "plain-text"
prerequisites: ["01_ecg_sampling_and_aliasing.md", "basic linear algebra", "timestamped traces"]
pattern_tags: ["capture-time-contract", "frame-and-unit-gate", "quality-aware-fusion"]
source_status: "DERIVED, ILLUSTRATIVE and OPEN; visible directory inventory only"
owner_role: "Perception and integration engineer"
source_index: "INDEX.md"
---

# Sensor Time Synchronization and Calibration

> **Purpose:** Establish whether observations may be compared or fused by proving their clock, timestamp, unit, frame, calibration and uncertainty contracts.
> **Prerequisites:** Sampled-signal units; [sampling and aliasing](01_ecg_sampling_and_aliasing.md); vectors/matrices; a trace with sequence numbers.
> **Key outcome:** Given two timestamped measurements and a transform record, calculate alignment uncertainty, reject an invalid join, and specify the next bounded diagnostic.
> **This file does NOT cover:** A complete Kalman-filter derivation, clinical interpretation, safety certification, or a target-specific calibration procedure.
> **Level:** 2 — Core Methods.
> **Version / reviewed:** 1.1 / not independently reviewed.
> **Evidence:** DERIVED clock/transform reasoning; ILLUSTRATIVE numerical examples; OPEN target behavior. No device result is claimed.
> **Retrieval cue:** If a fusion output looks plausible but disagrees with reality, begin with Sections 3–5 before changing estimator gains.

## 1. File Header

A fusion algorithm consumes assertions about physical measurements.
Those assertions include more than a numeric value.
They include when the physical event occurred.
They include where the measurement is expressed.
They include how raw sensor coordinates became reported coordinates.
They include the uncertainty of each conversion.
They include the revision that made the conversion.
A packet receipt timestamp cannot silently stand in for all of those assertions.
A valid matrix cannot silently stand in for a current mounting.
A small residual cannot prove that the sampled population was representative.

The engineering decision is whether to admit an observation to a downstream estimator or action.
The decision owner is the subsystem that owns the estimator input gate.
The input unit is one observation with value, time, frame, unit, calibration revision and quality fields.
The output is accept, transform, defer, reject, or enter a declared degraded mode.
The baseline is a system that joins packets by arrival order or trusts an unversioned transform.
That baseline can appear to work under low load while failing under jitter, reboot, remounting or clock drift.

**Takeaway:** Treat observation metadata as part of the measurement, not as logging decoration.

## 2. Concept Map / Mental Model

```text
Physical event at time t_p
  -> sensor exposure/integration interval
  -> local capture timestamp t_s in clock domain S
  -> packetization and transport
  -> receive timestamp t_r in clock domain R
  -> parsing/queueing
  -> processing timestamp t_u in clock domain U
  -> frame and unit conversion using calibration revision k
  -> quality/freshness/frame gate
  -> estimator or controller input

Clock contract:      t_ref = a_S * t_s + b_S +/- sigma_t
Geometry contract:   p_A = R_A_B(k) * p_B + d_A_B(k)
Validity contract:   age, covariance, unit, sequence and revision are acceptable
Failure contract:    reject / defer / use single sensor / controlled degradation
```

The physical event may occur over an exposure interval rather than at one mathematical instant.
A rolling-shutter camera, scanning lidar and integrated inertial sample have different time semantics.
The capture timestamp must identify the chosen convention.
A receive timestamp is useful for transport diagnosis but usually not for geometry.
A processing timestamp is useful for deadline diagnosis but usually not for fusion alignment.
A clock model maps local time to a chosen reference time.
A transform maps coordinates from a named source frame to a named destination frame.
A calibration revision binds the transform to a mounting, method and validity interval.
A quality gate consumes evidence before a filter assigns excessive confidence.

**Takeaway:** Separate event time, transport time, processing time, geometry and quality before composing them.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Required declaration | Nonexample |
|---|---|---|---|---|
| capture time `t_s` | time assigned at acquisition under a stated sensor convention | ns, us or s in clock S | epoch, clock domain and event/exposure convention | host receive time |
| receive time `t_r` | time packet becomes observable at receiver | time in clock R | receiver clock and queue boundary | physical exposure time |
| processing time `t_u` | time algorithm begins or completes work | time in clock U | stage and clock | capture time |
| offset `b` | additive difference from sensor clock to reference at a chosen origin | s | sign convention and reference | a drift rate |
| skew `a-1` | relative rate error in affine clock model | dimensionless | estimation interval | one fixed offset |
| jitter | random or bounded time variation around a timing model | s | distribution/bound and source | deterministic skew |
| latency | elapsed time from named start boundary to named end boundary | s | both boundaries | freshness by itself |
| freshness | age at the consumer’s decision point | s | reference clock and action point | packet age in another clock |
| intrinsic calibration | mapping raw sensor coordinates to ideal sensor coordinates | parameter set | model, units, range, method | an inter-sensor transform |
| extrinsic calibration | rigid relationship between named frames | rotation/translation | source/destination direction | camera focal length |
| covariance | uncertainty and correlation representation | matrix with squared units | variable order and frame | an arbitrary confidence score |
| observability | ability to infer a state from measurements and motion | property | model and excitation | low residual alone |
| association | claim that two observations refer to one entity/event | discrete relation | gating features/time/frame | nearest arrival |

A clock domain is a counter plus its oscillator, reset behavior and interpretation.
Two counters with the same numeric unit are not necessarily in the same clock domain.
An epoch is the defined zero point of a time value.
A monotonic counter need not be wall-clock synchronized.
A wall clock may step because of synchronization or manual correction.
A sequence number orders outputs from one producer but does not synchronize producers.
A timestamp resolution is the smallest representable increment.
A timestamp accuracy is closeness to a reference event time.
Resolution does not imply accuracy.
A calibration target is an artifact or environment used to estimate parameters.
A calibration residual is a model-data discrepancy under a stated observation model.
A low residual on the target does not prove validity outside the target’s range.

**Takeaway:** Define the time and geometry convention before interpreting any number that carries them.

## 4. Core Formulas with Full Derivation

### 4.1 Affine clock alignment

**Validity before use:** Use an affine model over an interval where relative oscillator rate is approximately constant, timestamp wrap/reset is handled, and the reference observations identify both offset and rate. Do not use it across an undisclosed reboot, leap/step, or temperature regime change.

Let `t_s` be local sensor time and `t_ref` be time in the reference domain.
At a single instant, an offset-only model is `t_ref = t_s + b`.
If the counters progress at different rates, offset changes with elapsed time.
The smallest linear extension is `t_ref = a * t_s + b`.
Here `a` is a dimensionless rate conversion and `b` has seconds.
Subtracting two matched observations removes `b`.
Thus `a = (t_ref,2 - t_ref,1) / (t_s,2 - t_s,1)` when the denominator is nonzero.
Substitution into either matched pair gives `b`.

```text
FORMULA [TC-01]: t_ref = a * t_s + b
WHERE: t_ref [s] is reference time; t_s [s] is sensor-clock time;
       a [dimensionless] is relative rate; b [s] is offset.
CONSTRAINT: affine clock relation is valid over the stated interval; wraps,
            resets and timestamp convention are handled before fitting.
NUMERICAL EXAMPLE: matched points are (t_s,t_ref)=(10.000000,10.002000)
                   and (70.000000,70.010000) seconds.
                   a=(70.010000-10.002000)/(70-10)=1.000133333.
                   b=10.002000-a*10=-0.000? 10.002000-10.001333333
                    =0.000666667 s.
                   At t_s=100 s, t_ref=1.000133333*100+0.000666667
                    =100.014000 s.
INTERPRETATION: a 133.333 ppm relative rate produces 14 ms reference offset
                at 100 s under this illustrative fitted model.
```

The arithmetic check is `1.000133333 * 100 = 100.0133333` seconds.
Adding `0.000666667` seconds yields `100.0140000` seconds.
The point at 70 seconds is reconstructed as `70.0093333 + 0.0006667 = 70.0100000` seconds.
The model has zero residual at two points because two parameters fit two observations.
That fact is not validation.
With more observations, inspect residuals against time, temperature, queue depth and reset events.

**Boundary/counterexample:** If the sensor reboots at local time zero during the interval, the same `t_s` can refer to two physical instants. No single affine map is valid without an epoch or boot identifier.

### 4.2 Drift accumulation bound

**Validity before use:** Use the linear drift approximation only for a declared maximum relative rate error `epsilon`, over a bounded elapsed time, with no resynchronization correction included.

Rate error causes time error to grow approximately linearly.
If the magnitude of relative skew is at most `epsilon`, elapsed time `Delta t` contributes at most `epsilon * Delta t` timing error.
This follows because a rate difference has units seconds per second.
Multiplying by duration leaves seconds.

```text
FORMULA [TC-02]: |e_drift| <= epsilon * Delta_t
WHERE: e_drift [s] is accumulated timing error; epsilon [s/s] is relative
       clock-rate error; Delta_t [s] is time since synchronization.
CONSTRAINT: epsilon is a credible bound over the interval; offset estimation,
            packet delay and quantization are accounted for separately.
NUMERICAL EXAMPLE: epsilon=100 ppm=100*10^-6 s/s; Delta_t=60 s.
                   |e_drift| <= 100*10^-6*60=0.006 s=6 ms.
INTERPRETATION: a 6 ms timing budget can be consumed by drift alone after one
                minute; a fusion gate needs margin for other uncertainty.
```

A unit check gives `(s/s) * s = s`.
At 10 minutes the same bound is `0.060 s`.
At 10 seconds it is `0.001 s`.
If the acceptance budget is 4 ms, a 100 ppm bound allows at most 40 seconds before drift alone reaches the budget.
That 40 seconds comes from `0.004 / 0.0001`.
A resynchronization schedule must also allow measurement, transport and estimation error.

**Boundary/counterexample:** A nominal oscillator specification is not automatically an in-system bound across temperature, supply, aging and firmware sleep behavior.

### 4.3 Measurement-time uncertainty composition

**Validity before use:** The root-sum-square rule requires zero-mean independent uncertainty terms or a deliberate approximation that ignores covariance. If errors share a clock, queue or synchronization event, include covariance or use a conservative bound.

For independent terms, variance of a sum equals the sum of variances.
Let synchronization, capture quantization and transport-estimation errors have standard deviations `sigma_sync`, `sigma_cap` and `sigma_net`.
Then standard deviation of their sum is the square root of summed squared standard deviations.

```text
FORMULA [TC-03]: sigma_t = sqrt(sigma_sync^2 + sigma_cap^2 + sigma_net^2)
WHERE: each sigma [s] is a standard deviation for the named time-error source;
       sigma_t [s] is the combined standard deviation.
CONSTRAINT: terms are independent or covariance is explicitly negligible;
            this is not a hard worst-case bound.
NUMERICAL EXAMPLE: sigma_sync=1.0 ms, sigma_cap=0.5 ms,
                   sigma_net=2.0 ms.
                   sigma_t=sqrt(1^2+0.5^2+2^2) ms=sqrt(5.25) ms
                   =2.291 ms.
INTERPRETATION: treating the terms as a 3.5 ms arithmetic sum is conservative
                but changes the meaning from a standard-deviation estimate.
```

The squared-unit check is `ms^2` inside the square root and `ms` outside.
If the terms are perfectly positively correlated, the combined standard deviation can approach 3.5 ms.
If a freshness gate needs a hard bound, standard deviations are insufficient without a tail policy.
A gate may instead retain individual bounds and sum those bounds conservatively.

**Boundary/counterexample:** Network delay can be asymmetric and bursty; a Gaussian standard deviation does not prove a 99.9th percentile deadline.

### 4.4 Rigid transform and direction convention

**Validity before use:** Use a rigid transform only when the frame relationship is a rotation plus translation over the validity interval. Flexing mounts, rolling timing errors and lens distortion require additional modeling.

Let `p_B` be coordinates of a point expressed in frame B.
A transform named `T_A_B` maps coordinates from B into A.
Its rotation `R_A_B` is dimensionless and orthonormal.
Its translation `d_A_B` has the coordinate unit.
The mapping is `p_A = R_A_B p_B + d_A_B`.
The inverse rotation is the transpose only when `R` is orthonormal.
The inverse translation is not simply negation in the original frame; it is rotated as well.

```text
FORMULA [TC-04]: p_A = R_A_B * p_B + d_A_B
WHERE: p_A and p_B [m] are 3-vectors in named frames; R_A_B [3x3] is a
       rotation; d_A_B [m] is B-origin expressed in A.
CONSTRAINT: R_A_B^T R_A_B=I, det(R_A_B)=+1, units match and calibration k
            is valid for the physical mounting.
NUMERICAL EXAMPLE: R_A_B rotates x_B to y_A by +90 degrees:
                   R=[[0,-1,0],[1,0,0],[0,0,1]], p_B=[2,0,0] m,
                   d_A_B=[1,3,0] m.
                   R*p_B=[0,2,0] m; p_A=[1,5,0] m.
INTERPRETATION: the point is not at [3,3,0]; adding translation before rotation
                or reversing the frame direction changes the physical claim.
```

A direct check multiplies the first row by `[2,0,0]` to obtain zero.
The second row yields two.
Adding `[1,3,0]` yields `[1,5,0]` meters.
The rotation transpose times itself yields identity for this matrix.
A transform unit test should include source basis vectors, origin and inverse round-trip.

**Boundary/counterexample:** If `T_A_B` is applied as though it maps A to B, a symmetric scene may hide the error until motion or asymmetric geometry exposes it.

### 4.5 Freshness at point of use

**Validity before use:** Compare timestamps only after mapping them to the same clock domain and defining the action boundary. Freshness is not a substitute for measurement quality or a hazard analysis.

A consumer at reference time `t_use` calculates observation age from capture time mapped to reference clock.
Let the mapped capture time be `t_capture_ref`.
Age is `t_use - t_capture_ref`.
An observation is eligible only if age is nonnegative within allowed reordering tolerance and does not exceed `A_max`.

```text
FORMULA [TC-05]: age = t_use - (a*t_capture + b)
WHERE: age [s]; t_use [s] in reference clock; t_capture [s] in sensor clock;
       a and b are from TC-01.
CONSTRAINT: clock model is valid; processing boundary is named; negative age
            is handled as a clock/order anomaly rather than silently clipped.
NUMERICAL EXAMPLE: t_use=100.120 s, mapped capture time=100.014 s.
                   age=100.120-100.014=0.106 s=106 ms.
                   If A_max=80 ms, reject or use the declared degraded path.
INTERPRETATION: low transport latency observed elsewhere cannot make this
                specific observation fresh at the action boundary.
```

Changing `t_use` by 20 ms changes age by 20 ms.
Changing the drift correction by 6 ms changes age by 6 ms.
That sensitivity is why a threshold should include explicit uncertainty margin.
A consumer may choose a stricter limit for a fast actuator than for a dashboard.

**Boundary/counterexample:** Reusing a last-known value can be correct for a slowly changing inventory label and unsafe for a moving obstacle; the contract must name the action.

**Takeaway:** An alignment formula is useful only with an interval, uncertainty model, clock identity and rejection rule.

## 5. Worked Numerical Example

**Scenario:** An illustrative mobile perception stack receives an IMU sample in clock S and a camera-derived feature in reference clock R. This is a constructed exercise, not a robot measurement or safety claim.

The IMU clock has an estimated maximum relative skew of 80 ppm since the last synchronization point.
The elapsed period since that point is 45 seconds.
The capture quantization is modeled as 0.25 ms standard deviation.
The synchronization residual standard deviation is 0.80 ms.
The transport estimation standard deviation is 1.50 ms.
The consumer decides at `t_use=45.090 s` reference time.
The mapped IMU capture time is `45.015 s` reference time.
The freshness threshold is 60 ms.

First compute drift bound with TC-02.
`80 ppm = 80 * 10^-6 s/s`.
`80 * 10^-6 * 45 s = 0.0036 s`.
The drift contribution can reach 3.6 ms.
This is a bound under the stated oscillator assumption, not a standard deviation.

Next compute combined standard deviation with TC-03 for the independent illustrative terms.
`0.80^2 = 0.64 ms^2`.
`0.25^2 = 0.0625 ms^2`.
`1.50^2 = 2.25 ms^2`.
The sum is `2.9525 ms^2`.
The square root is approximately `1.7183 ms`.
A calculator-independent bracket is useful: `1.7^2=2.89` and `1.72^2=2.9584`.
Thus 1.718 ms is consistent with the sum.

Now calculate age with TC-05.
`age = 45.090 - 45.015 = 0.075 s`.
`0.075 s = 75 ms`.
The raw age exceeds the 60 ms threshold by 15 ms.
Even before adding uncertainty margin, the observation is ineligible for the stated action.
The correct response is not to tune the filter until it accepts the observation.
The correct response is to reject/defer it or invoke the predeclared degraded behavior.

Sensitivity one changes the resynchronization interval.
If resynchronization occurs every 20 seconds, the 80 ppm drift bound is `1.6 ms`.
That improvement does not change the raw 75 ms age in this particular packet.
It can reduce uncertainty margin, but cannot recover a stale observation.
Sensitivity two changes the action threshold to 100 ms.
The packet then meets age threshold but still requires its frame, calibration revision and quality checks.
That is a policy change with physical consequence, not a numerical repair.

A minimal gate trace is:
1. Parse packet and verify schema revision.
2. Verify boot/epoch and sequence monotonicity for the producer.
3. Map capture time into the consumer reference clock using a valid model revision.
4. Compute age at the named decision boundary.
5. Reject if age exceeds the action-specific limit.
6. Verify unit, frame direction and calibration revision.
7. Attach time and geometry uncertainty to the estimator input.
8. Record accept/reject reason without blocking the real-time action path.

**Takeaway:** The illustrative packet is rejected because it is stale for the chosen action; better clock precision does not change that fact.

## 6. Visual / Geometric Intuition

Imagine two rulers lying on the same table.
One ruler starts 2 mm to the right of the other.
That is analogous to clock offset.
Now imagine the markings on one ruler are fractionally farther apart.
The disagreement grows with distance.
That is analogous to clock skew.
A synchronization observation estimates where the rulers align at a particular region.
It does not guarantee alignment everywhere after temperature or reset changes.

Imagine frame B painted on a camera housing.
Imagine frame A painted on the vehicle body.
The extrinsic transform says how B sits relative to A at a given mounting state.
If the camera bracket bends, the old transform remains a well-formed matrix.
It is simply no longer a valid description of the physical relationship.
A mathematically valid transform can therefore produce physically wrong coordinates.

A transform direction can be checked using the origin.
The origin of B, represented as `[0,0,0]` in B, maps to `d_A_B` in A.
If a diagram says otherwise, the named transform or diagram direction is inconsistent.
A basis-vector check catches rotation sign mistakes.
A round-trip check catches many inverse mistakes but can miss paired compensating bugs.
An independent physical reference remains necessary.

**Takeaway:** Offset is a displacement, skew is a rate mismatch, and calibration is a claim about a physical relationship that can expire.

## 7. Comparison / Selection Table

| Choice | Use when | Benefit | Avoid when | Disqualifier | Verification |
|---|---|---|---|---|---|
| receive-time join | only transport monitoring needs ordering | simple and observable | physical simultaneity matters | variable network/queue delay | compare against capture-time trace |
| fixed offset | short stable interval and measured rate mismatch negligible | low computation | long runs, sleep or thermal variation | residual grows with elapsed time | residual-versus-time plot |
| affine clock model | stable skew can be estimated | corrects first-order drift | resets/wraps or nonlinear behavior unmodeled | boot epoch unknown | holdout synchronization points |
| hardware trigger/shared clock | strict simultaneity is required | reduces uncertain software alignment | wiring/clock distribution unavailable | trigger latency not characterized | oscilloscope or equivalent target evidence |
| offline batch calibration | static mount and controlled target available | can estimate detailed model | mount changes during operation | invalid validity interval | held-out poses/environment |
| online self-calibration | enough excitation and observability exist | adapts slowly changing relation | ambiguity can be resolved only by prior assumptions | unobservable motion/scene | monitor residual and parameter drift |
| conservative reject gate | wrong action is costly and alternative exists | limits bad fusion | missing data itself is dangerous without fallback | no declared degraded behavior | injected stale/frame/unit tests |
| covariance inflation | uncertainty is modeled and estimator can consume it | retains partial information | failure is a categorical semantic mismatch | wrong frame, unit or identity | compare to rejection baseline |

Use a fixed offset only after demonstrating that the omitted drift is negligible for the action window.
Use covariance inflation for bounded quantitative uncertainty, not for unknown frame direction.
Use rejection for invalid semantics, then select a domain-specific fallback.
A remote dashboard may accept older observations than a collision-avoidance computation.
The difference should be a documented action contract rather than an accidental queue property.

**Takeaway:** Choose the simplest mechanism whose assumptions are demonstrably true for the decision horizon.

## 8. Failure Modes & Edge Cases

### FM-01 — Capture time is replaced by receive time

**Category:** VIOLATED_ASSUMPTION.
**Trigger:** Variable network or queue delay exists between sensor and consumer.
**Symptom:** Two stationary sensors appear to move relative to each other as load changes.
**Competing causes:** clock skew; queue backpressure; wrong timestamp field; actual motion.
**Discriminating diagnostic:** Record capture, receive and processing times in their own clock domains; induce queue load while keeping the physical target stationary.
**Expected signal:** receive-minus-capture delay changes with load while capture-time alignment remains stable.
**Immediate mitigation:** stop using receive time for physical association; retain it for transport telemetry.
**Correction:** transmit capture-time semantics, clock identity and uncertainty; map before association.
**Prevention:** schema test rejects untyped `timestamp` fields.
**Mitigation risk:** dropping packets may reduce visibility; define degraded mode.
**Verification:** replay a timestamped trace with injected queue delay and confirm association is unchanged.

### FM-02 — Offset-only correction hides clock drift

**Category:** NUMERICAL_EDGE.
**Trigger:** Two free-running clocks operate longer than the assumed alignment interval.
**Symptom:** residual increases approximately linearly with elapsed time.
**Competing causes:** actual sensor lag; time-unit conversion; temperature-dependent oscillator behavior; packet batching.
**Discriminating diagnostic:** fit residual against time since synchronization and compare separate thermal/load intervals.
**Expected signal:** a slope consistent with relative skew, possibly changing by condition.
**Immediate mitigation:** tighten freshness limit or resynchronize more often.
**Correction:** estimate affine model with boot epochs and monitor holdout residuals.
**Prevention:** define maximum unsynchronized interval from timing budget.
**Mitigation risk:** frequent synchronization can consume bandwidth or wake energy.
**Verification:** use held-out sync observations; verify error remains inside declared budget.

### FM-03 — Clock reset or counter wrap is interpreted as time travel

**Category:** CONCEPT_CONFUSION.
**Trigger:** Sensor reboot, firmware update, counter wrap or epoch step occurs.
**Symptom:** negative age, enormous apparent velocity or out-of-order burst.
**Competing causes:** packet reorder; signed/unsigned conversion; reference-clock step; true duplicate sequence.
**Discriminating diagnostic:** inspect boot identifier, counter width, wrap policy and sequence continuity.
**Expected signal:** time discontinuity coincides with epoch change or expected modulo boundary.
**Immediate mitigation:** quarantine observations from the changed epoch until resynchronized.
**Correction:** include epoch/boot ID; make wrap and restart state explicit.
**Prevention:** state-machine test covers reset while messages are queued.
**Mitigation risk:** a broad quarantine can conceal an actual sensor outage.
**Verification:** inject restart and wrap traces; ensure no pre/post-reset association occurs.

### FM-04 — Transform direction or unit is inverted

**Category:** VIOLATED_ASSUMPTION.
**Trigger:** A developer applies `T_A_B` as though it were `T_B_A`, or mixes meters and millimeters.
**Symptom:** mirrored, translated or scale-discontinuous trajectory.
**Competing causes:** wrong axis convention; stale calibration; sensor extrinsic error; coordinate serialization bug.
**Discriminating diagnostic:** transform origin and basis vectors; check inverse round trip; compare known physical reference distances.
**Expected signal:** origin does not map to documented translation, or one-axis sign/scale fails.
**Immediate mitigation:** block downstream action requiring that frame.
**Correction:** encode source/destination frame names and typed units in API; publish tested transform fixtures.
**Prevention:** code review requires frame direction in identifiers and metadata.
**Mitigation risk:** automatic inversion can mask ambiguous naming.
**Verification:** pass origin, three basis vectors and an asymmetric landmark through target implementation.

### FM-05 — Stale calibration revision is accepted after remount

**Category:** VIOLATED_ASSUMPTION.
**Trigger:** Sensor is moved, bracket flexes, lens setting changes or firmware changes preprocessing.
**Symptom:** residual shifts by location, temperature or maintenance event despite stable clock alignment.
**Competing causes:** time error; target movement; intrinsic drift; unit mismatch; association error.
**Discriminating diagnostic:** compare residual before/after event using a physical reference and calibration revision ledger.
**Expected signal:** change correlates with revision/mount event rather than network load.
**Immediate mitigation:** disable affected fusion path or increase only the declared uncertainty pending verification.
**Correction:** recalibrate under a documented method and bind new revision to validity interval.
**Prevention:** maintenance workflow invalidates calibration on relevant configuration changes.
**Mitigation risk:** blind covariance inflation can permit a semantic error into control.
**Verification:** evaluate held-out reference poses and record residual distribution by revision.

### FM-06 — Correlated errors are combined as independent

**Category:** NUMERICAL_EDGE.
**Trigger:** Two sensors share a clock, power event, map or preprocessing defect.
**Symptom:** fused covariance is too small and the estimate is confidently wrong.
**Competing causes:** model mismatch; outlier rejection bug; incorrect covariance units; true independent noise.
**Discriminating diagnostic:** fault-inject shared error and compare empirical innovation correlation to assumed covariance.
**Expected signal:** residuals co-vary while independent model predicts less covariance.
**Immediate mitigation:** gate the shared failure source and avoid overconfident action.
**Correction:** model covariance/correlation or choose a robust fusion architecture.
**Prevention:** catalog shared dependencies in sensor design review.
**Mitigation risk:** excessive inflation can make a usable estimator overly conservative.
**Verification:** evaluate normalized innovation consistency on held-out fault scenarios.

**Takeaway:** Diagnose metadata, reset and correlation failures before interpreting a fused output as a model-quality failure.

## 9. Complexity & Cost Analysis

An affine time conversion is constant time and constant memory per observation.
A sequence/epoch check is constant time and keeps constant state per producer.
A 3D rigid transform requires one 3-by-3 matrix-vector multiply and one vector addition.
That is nine multiplications and nine additions if implemented directly.
A quaternion representation may reduce storage but still requires a carefully normalized conversion or rotation operation.
A covariance transform `P_A = R P_B R^T` is cubic in state dimension for dense matrices.
For a 3-by-3 covariance, dense arithmetic is small but should not hide frame/unit errors.

Illustrative telemetry cost matters at scale.
Suppose each observation adds 8 bytes capture time, 4 bytes sequence, 2 bytes clock/revision identifiers, 12 bytes frame/unit/quality fields and 4 bytes checksum.
The added metadata is 30 bytes per observation before transport framing.
At 100 observations per second, payload addition is 3000 bytes/s.
Over one hour this is `3000 * 3600 = 10,800,000` bytes, about 10.3 MiB using 1 MiB = 1,048,576 bytes.
This arithmetic excludes compression, envelopes, replication and index overhead.
Removing identity metadata to save 30 bytes can make a future physical failure impossible to reconstruct.
The appropriate decision compares that cost with the action consequence and storage policy.

A calibration solver can be much more expensive than applying a calibration.
Application belongs on a bounded online path once parameters are validated.
Estimation may belong offline or in a controlled background job unless observability and resource budget are demonstrated.
Do not transfer an offline optimizer’s runtime to a real-time action claim.

**Takeaway:** Runtime cost of checking contracts is usually small; cost of missing provenance can dominate incident diagnosis.

## 10. Key Theorems / Results Reference

**Affine clock result:** Two nonidentical matched clock observations determine an affine offset/rate map algebraically; additional observations are needed to assess residuals and model validity.

**Variance-of-sum result:** For random variables with finite variance, variance of a sum includes each variance and twice each covariance term. Root-sum-square omits the covariance terms only under the stated independence approximation.

**Rigid-transform result:** A proper 3D rotation satisfies `R^T R = I` and `det(R)=+1`. Its transpose is its inverse. A translation must be transformed when forming the inverse rigid transform.

**Observability caution:** Parameters cannot be uniquely inferred from measurements that do not excite or distinguish them. A low training/calibration residual can coexist with unobservable ambiguity.

**Causality/freshness result:** An estimator can be internally causal yet operationally stale if its input capture time predates the consumer’s action horizon. Causality is not a freshness guarantee.

**Takeaway:** Named results supply checks and assumptions, not permission to omit target validation.

## 11. Interview / Exam Questions

1. **Why is receive time insufficient for sensor fusion?** It includes variable transport and queue delay, so it can order packets without representing the physical event time. Use it for network diagnosis; use mapped capture time for physical alignment.

2. **Derive the drift error after 90 seconds at 50 ppm.** `50*10^-6*90=0.0045 s`, or 4.5 ms, under a constant relative-rate bound. This excludes offset-estimation and transport uncertainty.

3. **A packet has age 70 ms against a 60 ms action limit. Should increasing covariance admit it?** No. Age-limit violation is a decision contract violation. Covariance inflation models bounded quantitative uncertainty; it does not make a stale observation fresh.

4. **What first test catches a transform-direction mistake?** Transform the source-frame origin and basis vectors, then compare them with documented destination-frame positions. Include an asymmetric landmark so a symmetric scene cannot hide sign errors.

5. **When does an offset-only clock model win over an affine model?** When the action interval is short enough that bounded skew contributes negligibly relative to the timing budget, and this is shown with units. It is not selected merely because it is simpler.

6. **How do you distinguish clock skew from queue delay?** Compare capture-to-reference residual against elapsed time and receive-minus-capture delay against load. Skew tends to create time-since-sync slope; queue delay tends to correlate with load/batching.

7. **What calibration metadata must travel with an observation?** At minimum frame names/direction, units, calibration revision, validity interval or status, and uncertainty convention. Exact fields depend on the action and estimator.

8. **Why can two accurate sensors create an overconfident fused estimate?** Their errors may be correlated through a shared clock, map, power event or preprocessing defect. Treating correlated errors as independent makes covariance too small.

9. **A camera is remounted but residuals remain small on one planar target. Is calibration valid?** Not established. A planar or symmetric target may not expose all degrees of freedom. Use held-out poses, depth variation and physical references appropriate to the model.

10. **How would you handle a sensor reboot midstream?** Detect epoch/boot change, quarantine or reset association state, obtain valid time alignment, and record the transition. Do not interpret the counter reset as an extremely late measurement.

11. **What is the difference between timestamp resolution and accuracy?** Resolution is representable increment; accuracy is closeness to defined event time. Nanosecond resolution can carry milliseconds of capture or synchronization error.

12. **What changes if the consumer is a dashboard instead of a motion controller?** The allowed age, fallback and audit requirements may differ because action consequence differs. The timestamp semantics and frame/unit correctness do not become optional.

**Takeaway:** Strong answers name the assumption, discriminating evidence, bounded correction and consequence of being wrong.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Admission precondition | Require producer/epoch, capture-time semantics, common clock mapping, units, frame direction, calibration revision and quality state. |
| Clock equation | `t_ref=a*t_sensor+b`; fit only over a valid interval and inspect holdout residuals. |
| Drift bound | `|e| <= epsilon*Delta_t`; include offset, network and quantization uncertainty separately. |
| Geometry equation | `p_A=R_A_B*p_B+d_A_B`; test origin, basis vectors and inverse with named frames. |
| Freshness rule | Compute age at the action boundary from mapped capture time; reject/degrade when it exceeds action-specific policy. |
| Do not use | Receive time as physical event time; covariance inflation for wrong frame/unit/identity; old calibration after an invalidating change. |
| First diagnostics | Check epoch/sequence, capture-vs-receive trace, residual versus elapsed time, frame/unit fixtures and calibration ledger. |
| Evidence boundary | All numerical values here are illustrative; target timing, calibration validity and safety consequences require independent evidence. |
