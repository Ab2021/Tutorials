# Robotics Control

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 11 of 14
> Estimated study time: 4 hours
> Prerequisites: 10
> Version 2026.3

## 1. Why this topic exists

A robot's kinematics says where the tool should be; the controller makes it go there. The controller must track a trajectory, reject disturbances, limit forces, and fail safely. It must do so with a sampled loop, a finite actuator, and a patient who is not a rigid body.

The industry problem is that controller behaviour is judged by a step response on a bench, not by the loop's behaviour in the procedure. A controller with a fast step response can oscillate when the tool contacts tissue, overshoot when the payload changes, or become unstable when the control loop is delayed by a network. Three decisions depend on control literacy:

1. **Whether the loop is stable with the real plant.** The plant includes the robot, the tool, the tissue, and the delay. A gain that is stable on the bench can be unstable in the procedure.
2. **Whether the interaction is safe.** A position-controlled robot that hits tissue can generate a large force before the force limit acts. An impedance- or force-controlled robot can regulate the interaction, but only if the loop is designed for it.
3. **Whether the failure is safe.** A controller that saturates, loses a sensor, or receives a delayed command must enter a safe state, not continue with the last command.

This topic covers proportional-integral-derivative (PID) control, impedance and admittance control, force control, stability with delay, saturation, and safety envelopes. It assumes the kinematics from Topic 10. It does not cover advanced nonlinear control or learning-based control; those are research topics.

A useful framing: a controller is a filter that shapes the closed-loop response. The proportional term sets the stiffness, the derivative term sets the damping, and the integral term removes the steady-state error. The interaction with the environment is a second loop, and it can destabilise the first.

## 2. Learning outcomes

By the end of this topic you can:

- Write the continuous and discrete PID equations and explain the role of each term.
- Compute the natural frequency and damping ratio of a second-order joint model under PD control.
- Simulate a discrete PID loop and measure overshoot, settling time, and steady-state error.
- Explain integral windup and propose two mitigations.
- Distinguish impedance control from admittance control and choose one for a task.
- Compute the steady-state displacement of an impedance controller under a force.
- Analyse the effect of control-loop delay on phase margin.
- Design a force limit and a safe stop for a robot interacting with a patient.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A control loop compares a reference with a measurement, computes an error, and applies a correction. The correction goes through an actuator with limits, a plant with dynamics, and a sensor with noise and delay. The loop is stable if the correction reduces the error without overshooting into oscillation.

```text
   reference --> [ SUM ] --> [ CONTROLLER ] --> [ ACTUATOR ] --> [ PLANT ] --> output
                   ^            |                   |               |          |
                   |            |                   |               |          |
                   |            +--> Kp, Ki, Kd      +--> saturation  +--> J, b |
                   |                                                             |
                   +------------------- [ SENSOR ] <----------------------------+
                                        noise, delay, resolution
```

The model has four equations:

\[
u(t) = K_p e(t) + K_i \int_0^t e(\tau)d\tau + K_d \frac{de(t)}{dt}
\]

\[
J \ddot{\theta} + b \dot{\theta} = \tau
\]

\[
\omega_n = \sqrt{\frac{K_p}{J}}, \qquad \zeta = \frac{b + K_d}{2\sqrt{K_p J}}
\]

\[
u_k = K_p e_k + K_i T_s \sum_{i=0}^k e_i + K_d \frac{e_k - e_{k-1}}{T_s}
\]

The first is the continuous PID; the second is a simple joint plant; the third gives the closed-loop natural frequency and damping; the fourth is the discrete implementation. The discrete form is what the robot actually runs.

### 3.1 PID terms

| Term | Effect | Too much | Too little |
|---|---|---|---|
| Proportional | Stiffness; reduces error | Overshoot; oscillation | Slow; steady error |
| Integral | Removes steady error | Windup; overshoot | Steady error |
| Derivative | Damping; reduces overshoot | Noise amplification | Overshoot; oscillation |

The terms are not independent. Increasing \(K_p\) raises the natural frequency and reduces the damping ratio; increasing \(K_d\) raises the damping without changing the natural frequency. The integral term acts on the low-frequency error and can destabilise a loop with saturation.

### 3.2 Impedance and admittance

An impedance controller regulates the relationship between force and motion:

\[
F = K(x - x_d) + B(\dot{x} - \dot{x}_d) + M(\ddot{x} - \ddot{x}_d)
\]

The robot behaves like a spring-damper-mass. A stiff impedance (\(K\) large) resists motion; a soft impedance (\(K\) small) complies. The controller measures position and commands force.

An admittance controller measures force and commands motion:

\[
\dot{x} = \frac{F - F_d}{B}, \qquad x = \int \dot{x} dt
\]

The robot behaves like a mass-damper driven by force. The controller measures force and commands position. It is often used with a position-controlled robot that has no torque interface.

| Property | Impedance | Admittance |
|---|---|---|
| Measures | Position | Force |
| Commands | Force | Position |
| Requires | Torque control | Force sensor; position control |
| Stable with stiff environment | Better | Worse |
| Stable with soft environment | Worse | Better |

The choice depends on the robot's interface and the environment's stiffness. A stiff environment (bone) favours impedance; a soft environment (tissue) favours admittance, but the loop must be tuned for the contact.

### 3.5 Control at the joint and task level

| Level | Input | Output | Rate | Example |
|---|---|---|---|---|
| Joint | Joint angle | Motor torque | 1–10 kHz | PID |
| Task | Cartesian pose | Joint velocity | 100–1,000 Hz | Resolved-rate |
| Interaction | Force and pose | Force or motion | 100–1,000 Hz | Impedance |
| Supervisor | State and events | Mode | 10–100 Hz | Safety monitor |

The levels have different rates and different failure modes. A joint loop that misses a cycle produces a torque spike; a task loop that misses a cycle produces a path error; a supervisor that misses an event produces an unsafe mode.

The rates must be compatible. A 100 Hz task loop cannot control a 1 kHz joint loop without a buffer and a delay budget.

### 3.6 Control pathologies

| Pathology | Symptom | Cause | Fix |
|---|---|---|---|
| Oscillation | Sustained sine | Low margin; delay | Reduce gain; filter |
| Limit cycle | Small oscillation at reversal | Backlash; friction | Compensation; mechanic |
| Windup | Overshoot after saturation | Integral accumulation | Clamp; back-calculation |
| Steady error | Offset | No integral; friction | Integral; feed-forward |
| Noise amplification | Buzz | Derivative on noisy signal | Filter; derivative on measurement |
| Mode confusion | Wrong motion | UI or state machine | Interlocks; mode awareness |
| Chatter | High-frequency contact vibration | Stiff contact; delay | Compliance; lower gain |

The pathologies are not independent. A delay reduces the margin, which makes the oscillation more likely; a saturation adds windup, which makes the overshoot larger. The diagnosis must find the initiating cause, not only the symptom.

### 3.3 Stability and delay

A sampled loop has a delay of at least one sample, plus the sensor and actuator delays. The delay adds phase lag:

\[
\phi = -\omega T_d
\]

At a 70.7 rad/s natural frequency and a 10 ms delay, the phase lag is 0.707 rad, or 40.5°. The phase margin is reduced by the same amount. A loop with a 45° margin on the bench can have a 5° margin in the procedure.

The loop rate must be fast enough that the delay is a small fraction of the closed-loop period. A 1 kHz loop with a 10 ms delay has a delay of 10 samples, which is large. A 10 kHz loop with the same delay has 100 samples, which is better but still significant.

### 3.4 Saturation and windup

An actuator has a maximum torque, force, or velocity. When the controller commands more than the limit, the actuator saturates, and the loop opens. The integral term continues to accumulate the error, which is called windup. When the error reverses, the integral takes time to unwind, and the response overshoots.

| Mitigation | Mechanism | Cost |
|---|---|---|
| Clamp the integral | Limit the integral term | Tuning |
| Back-calculation | Reduce the integral when saturated | Implementation |
| Conditional integration | Integrate only when not saturated | Logic |
| Feed-forward | Reduce the error before the loop | Model |

The windup is a design issue, not a tuning issue. A loop that saturates for a large step will overshoot unless the integral is managed.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| PID | Proportional-integral-derivative controller | Used for any feedback | None |
| Plant | System being controlled | Confused with the controller | None |
| Setpoint | Desired value | Confused with the measurement | Unit |
| Error | Setpoint minus measurement | Confused with the disturbance | Unit |
| Gain | Multiplier on the error | Confused with the time constant | Unit-dependent |
| Natural frequency | Frequency of the undamped oscillation | Confused with the bandwidth | rad/s |
| Damping ratio | Dimensionless damping | Confused with the damping coefficient | Unitless |
| Overshoot | Peak above the setpoint | Confused with steady error | % |
| Settling time | Time to within a band | Confused with rise time | s |
| Steady-state error | Error after the transient | Confused with offset | Unit |
| Integral windup | Integral accumulates during saturation | Confused with instability | None |
| Saturation | Actuator at its limit | Confused with a fault | Unit |
| Feed-forward | Control based on the model | Confused with feedback | None |
| Impedance | Force-motion relationship | Confused with stiffness | N/m, Ns/m |
| Admittance | Motion-force relationship | Confused with compliance | m/N |
| Stiffness | Force per displacement | Confused with impedance | N/m |
| Damping | Force per velocity | Confused with friction | Ns/m |
| Compliance | Displacement per force | Confused with stiffness | m/N |
| Force control | Regulate force | Confused with impedance | N |
| Hybrid control | Position in some directions, force in others | Confused with impedance | None |
| Phase margin | Stability margin | Confused with gain margin | degrees |
| Gain margin | Stability margin in gain | Confused with phase margin | dB |
| Loop rate | Controller update frequency | Confused with the sample rate | Hz |
| Delay | Time from measurement to actuation | Confused with latency | s |
| Dead time | Delay with no response | Confused with a time constant | s |
| Backlash | Clearance in the mechanism | Confused with delay | rad |
| Friction | Resistance to motion | Confused with damping | Nm |
| Gravity compensation | Feed-forward torque for gravity | Confused with integral | Nm |
| Assist-as-needed | Control that assists only when needed | Confused with full assistance | None |
| Safety envelope | Allowed workspace and force | Confused with a joint limit | m, N |

## 5. The mechanics, step by step

The procedure below designs and verifies a robot controller. Each step states its input, output, and rationale. The required edge cases are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the task and the interaction

**Input:** the procedure, the tool, the tissue, and the user.

**Output:** a control objective: position, force, impedance, or a combination.

**Rationale:** the objective determines the controller. A bone-cutting robot needs position control with a force limit; a rehabilitation robot needs impedance or assist-as-needed control; a teleoperated surgical robot needs position control with force feedback.

A task that is described only as "move to a point" omits the interaction. The interaction determines the stability and the safety.

### Step 2. Model the plant

**Input:** the robot's dynamics, the tool, and the environment.

**Output:** a plant model: inertia, damping, stiffness, friction, and delay.

**Rationale:** the model determines the gains and the stability. A joint with a 0.01 kg m² inertia and a 0.1 Nms/rad damping has a different response from a joint with a 0.1 kg m² inertia and a 1 Nms/rad damping.

The model must include the environment when the robot is in contact. A stiff environment adds stiffness and can destabilise an admittance controller.

### Step 3. Choose the control law

**Input:** the objective and the plant.

**Output:** a control law: PID, impedance, admittance, or hybrid.

**Rationale:** the law follows from the objective and the interface. A robot with a torque interface can use impedance; a robot with only a position interface must use admittance. A robot that must track a trajectory uses PID with feed-forward.

The choice is not a matter of preference. A position-controlled robot cannot regulate force without a force sensor and an admittance loop.

### Step 4. Set the gains from the desired response

**Input:** the plant and the desired natural frequency and damping.

**Output:** \(K_p\), \(K_i\), \(K_d\).

**Rationale:** the gains can be derived from the desired closed-loop response. For a second-order joint, \(K_p = J\omega_n^2\) and \(K_d = 2\zeta\sqrt{K_p J} - b\). The integral gain is set for the desired disturbance rejection, usually a decade below the natural frequency.

A gain set without a desired response is a guess. A gain set from the model is a hypothesis that must be tested.

### Step 5. Check the loop rate and the delay

**Input:** the controller, the sensors, and the actuators.

**Output:** a loop rate, a delay budget, and a phase margin.

**Rationale:** the loop rate must be at least 10–20 times the closed-loop bandwidth. A 70 rad/s loop needs a 700–1,400 rad/s sample rate, or 110–220 Hz. A 1 kHz loop is comfortable; a 100 Hz loop is marginal.

The delay includes the sensor, the computation, the network, and the actuator. Each must be measured, not assumed.

### Step 6. Simulate the closed-loop response

**Input:** the plant, the controller, and the delay.

**Output:** a step response, a frequency response, and a stability margin.

**Rationale:** the simulation shows the overshoot, the settling, and the sensitivity to delay. It is a hypothesis, not evidence; the real robot has friction, backlash, and a different inertia.

The simulation must include the saturation and the delay. A simulation without them is optimistic.

### Step 7. Implement the discrete controller

**Input:** the control law and the loop rate.

**Output:** discrete equations with anti-windup and saturation.

**Rationale:** the discrete implementation differs from the continuous one. The derivative is a difference, which amplifies noise; the integral is a sum, which accumulates error; the saturation requires anti-windup.

The implementation must handle the first sample, the sensor dropout, and the actuator fault. A controller that assumes a valid sensor at every sample is a controller that fails on the first dropout.

### Step 8. Tune on the real plant

**Input:** the implementation and the robot.

**Output:** gains and filters that meet the response and the stability.

**Rationale:** the model is an approximation. The real plant has friction, compliance, and a different inertia. The tuning must start from the model-based gains and adjust them with measurements.

The tuning must be done with the tool and the payload, not with the bare robot. A gain that is stable with no tool can be unstable with a long instrument.

### Step 9. Design the force limit and the safe stop

**Input:** the task, the tissue, and the risk analysis.

**Output:** a force limit, a detection time, and a safe state.

**Rationale:** the force limit is a safety requirement, not a control parameter. It must be below the tissue's injury threshold with a margin, and the detection and stop must be fast enough to limit the travel.

The safe stop must be independent of the controller. A watchdog or a safety relay must remove power if the controller fails.

### Step 10. Design the interaction control

**Input:** the force sensor, the environment, and the task.

**Output:** an impedance or admittance law with a stability analysis.

**Rationale:** the interaction loop is a second loop around the position loop. Its stability depends on the environment stiffness and the sensor delay. A stiff environment and a long delay can destabilise an admittance controller.

The interaction gains must be tuned with a phantom that represents the tissue. A tuning on free space does not predict the contact behaviour.

### Step 11. Verify the safety envelope

**Input:** the controller and the limits.

**Output:** a test report with the force, the workspace, and the stop time.

**Rationale:** the verification must test the worst case: maximum speed, maximum payload, minimum stiffness, maximum delay. A test at the nominal point does not demonstrate the envelope.

The test must include a sensor failure, an actuator saturation, and a communication loss. The response to each must be safe.

### Step 12. Document and monitor

**Input:** the controller, the gains, and the tests.

**Output:** a configuration record and a monitoring plan.

**Rationale:** the gains, the filters, the loop rate, and the firmware version are part of the controller. A change in any of them changes the behaviour. The record must include them, and the monitoring must detect a drift.

### 8.5 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 80601-2-77 | Control performance, force limit, stop time | Reproduce the claim |
| IEC 80601-2-78 | Assistance behaviour, patient contact | Reproduce the rehab claim |
| ISO 14971 | Hazard analysis for force and instability | Link failure to harm |
| IEC 62304 | Control software version and gains | Reproduce behaviour |
| IEC 62366-1 | Mode awareness and controls | Human factors |
| ISO/TS 15066 | Force and pressure limits | Informative reference |

The record is the same information the data team needs for provenance. A team that records the gains and the firmware for its own analysis is also assembling regulatory evidence.

### 8.6 A note on the control software class

The control software is usually class C under IEC 62304, because a failure can cause injury. The class requires a detailed design, unit verification, integration testing, and regression testing. The gains and the filters are part of the design, not tuning parameters outside the lifecycle.

A gain change is a software change. It must be reviewed, verified, and recorded. A field tune that bypasses the process is a change outside the quality system.

A controller that is retuned in the field without a record is a controller that cannot be reproduced.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 7, 9 | Actuator at limit | Current or torque log | Overshoot; windup |
| Common-mode rejection failure | 2 | Sensor issue | N/A | N/A |
| Electrode offset | 2 | Sensor issue | N/A | N/A |
| Clock jitter | 5, 6 | Loop timing variation | Trace | Phase margin loss |
| Brown-out | 9, 12 | Controller reset | Reset log | Lost pose |
| Watchdog reset loop | 9, 12 | Controller hang | Reset counter | Procedure interrupted |
| Thermal throttling | 5 | Clock reduction | Temperature log | Late loop |
| Firmware update failure | 12 | Gains lost or changed | Version and checksum | Wrong behaviour |
| Calibration drift | 10, 12 | Sensor or tool drift | Residual trend | Force or position error |

The control edge cases are dominated by saturation, delay, and sensor failure. The processor edge cases enter through the loop timing and the safety monitor.

### 5.14 Controller comparison

| Controller | Measures | Commands | Best for | Weakness |
|---|---|---|---|---|
| PID position | Position | Torque or velocity | Trajectory tracking | Contact force |
| Impedance | Position | Force | Compliant interaction | Stiff environment |
| Admittance | Force | Position | Position-controlled robot | Soft environment |
| Force | Force | Force | Constant force | Contact loss |
| Hybrid | Position and force | Both | Constrained motion | Complexity |

The controller choice is a function of the interface, the environment, and the task. A robot with a torque interface and a stiff environment is a good fit for impedance; a robot with a position interface and a soft environment is a good fit for admittance.

### 5.15 Gain-selection table

| Desired response | \(\omega_n\) | \(\zeta\) | \(K_p\) | \(K_d\) |
|---|---:|---:|---:|---:|
| Slow, damped | 20 rad/s | 1.0 | \(0.01 \times 400 = 4\) | \(2 \times 1 \times \sqrt{0.04} - 0.1 = 0.3\) |
| Moderate | 50 rad/s | 0.8 | 25 | 0.7 |
| Fast | 100 rad/s | 0.6 | 100 | 1.1 |
| Aggressive | 200 rad/s | 0.4 | 400 | 1.5 |

The table uses \(J = 0.01\) kg m² and \(b = 0.1\) Nms/rad. The gains grow with the square of the natural frequency; the derivative gain grows linearly. A fast response requires a large torque, which may exceed the actuator.

### 5.16 Stability analysis with delay

A loop with a proportional gain \(K_p\), a plant \(J s^2 + b s\), and a delay \(T_d\) has a phase margin that falls as the gain rises. The crossover frequency is approximately \(\omega_c = K_p/b\) for a velocity-controlled plant, and the phase at crossover is \(-90^\circ - \omega_c T_d\).

| \(K_p\) | Crossover (rad/s) | Delay (ms) | Phase lag | Margin |
|---:|---:|---:|---:|---:|
| 20 | 200 | 1 | 11.5° | 78.5° |
| 50 | 500 | 1 | 28.6° | 61.4° |
| 100 | 1,000 | 1 | 57.3° | 32.7° |
| 100 | 1,000 | 5 | 286° | Unstable |

The table shows that the delay, not the gain, is the limiting factor at high gain. A 5 ms delay at a 1,000 rad/s crossover is 286°, which is far beyond a stable margin.

The delay must be measured, not assumed. A network, a filter, or a slow sensor can add milliseconds that the bench test did not include.

### 5.17 Force control

A force controller regulates the contact force by adjusting the position or the torque. The simplest form is:

\[
x_{\text{cmd}} = x_0 + K_f (F_d - F)
\]

where \(K_f\) is the force gain. The loop is stable if the environment stiffness times the gain is below the stability limit. A stiff environment requires a small gain; a soft environment allows a larger gain.

| Environment | Stiffness | Force gain | Stability |
|---|---:|---:|---|
| Bone | 10,000 N/m | 0.0001 m/N | Marginal |
| Soft tissue | 1,000 N/m | 0.001 m/N | Stable |
| Air | 0 | 0.01 m/N | Stable but no contact |

The table shows that the same gain is stable in air and unstable on bone. The force controller must adapt to the environment or be tuned for the stiffest expected contact.

### 5.18 Assist-as-needed control

A rehabilitation robot should assist only when the patient cannot complete the motion. The controller estimates the patient's contribution and adds assistance proportional to the deficit:

\[
\tau_{\text{assist}} = K_a (\theta_d - \theta) - B_a \dot{\theta}
\]

If the patient moves well, the error is small and the assistance is small. If the patient cannot move, the error is large and the assistance is large. The gain \(K_a\) sets the stiffness of the assistance.

The risk is that the robot does the work and the patient does not learn. The control must be tuned to the patient's ability and must not mask the patient's effort. The evidence for the clinical benefit is mixed, and the control design is part of the reason.

### 5.19 Safety envelope

The safety envelope has four limits: workspace, velocity, force, and power. Each must be enforced independently of the main controller.

| Limit | Value | Enforcement | Response |
|---|---|---|---|
| Workspace | Box or joint limits | Safety monitor | Stop |
| Velocity | Maximum tool speed | Safety monitor | Reduce or stop |
| Force | Maximum contact force | Force monitor | Stop |
| Power | Maximum actuator power | Safety relay | Remove power |

The safety monitor must be separate from the controller. A controller that enforces its own limit can be reprogrammed or can fail. A separate monitor with an independent sensor and a hardware stop is the defence.

### 5.20 Ten control review questions

1. What is the plant model, and what is the environment?
2. What is the control law, and why was it chosen?
3. What are the gains, and how were they derived?
4. What is the loop rate, and what is the delay?
5. What is the phase margin with the real plant and the delay?
6. What happens under saturation, and how is windup handled?
7. What is the force limit, and how fast is the stop?
8. What is the safe state, and how is it reached?
9. How is the interaction loop tuned and verified?
10. What is recorded so that a gain change can be detected?

A review that answers all ten has covered the controller. A review that answers only the third and the fourth has covered the tuning, not the safety.

### 5.21 Delay budget

| Source | Typical delay | Measurement |
|---|---:|---|
| Sensor sampling | 0.5–2 ms | Scope |
| Filter | 1–10 ms | Step response |
| Controller computation | 0.1–1 ms | Trace |
| Network | 0.5–20 ms | Timestamp |
| Actuator | 0.5–5 ms | Command versus motion |
| Safety monitor | 1–10 ms | Fault injection |

The total delay is the sum. A 20 ms network delay can dominate a 1 ms controller. The budget must include every source, not only the controller.

### 5.26 Gain scheduling

A robot's plant changes with configuration and payload. A gain that is stable at the workspace centre may be unstable at full extension. Gain scheduling changes the gains with the configuration:

\[
K_p(q) = K_{p0} \frac{J(q)}{J_0}
\]

The schedule must be continuous, or the controller will jump at the switch. A discontinuous schedule can produce a torque spike.

| Configuration | Inertia | \(K_p\) | \(K_d\) |
|---|---:|---:|---:|
| Folded | 0.005 | 25 | 0.7 |
| Mid | 0.010 | 50 | 1.0 |
| Extended | 0.020 | 100 | 1.4 |

The table shows that the gains double from the folded to the extended configuration. A fixed gain is a compromise that is conservative in one configuration and aggressive in another.

### 5.27 Force sensor characteristics

| Sensor | Range | Resolution | Bandwidth | Use |
|---|---|---|---|---|
| Strain gauge | 0–500 N | 0.1 N | 1 kHz | Wrist force |
| Capacitive | 0–100 N | 0.01 N | 100 Hz | Tool tip |
| Piezoelectric | 0–1,000 N | 0.01 N | 10 kHz | Impact |
| Optical | 0–50 N | 0.05 N | 1 kHz | MRI-compatible |

The sensor's bandwidth must exceed the control loop's bandwidth. A 100 Hz sensor in a 1 kHz loop adds delay and limits the gain. The sensor's resolution must be below the force resolution required by the task.

### 5.22 Discrete implementation details

| Issue | Effect | Mitigation |
|---|---|---|
| Derivative on error | Kick on a setpoint step | Derivative on measurement |
| Derivative noise | Amplified sensor noise | Low-pass filter; smaller \(K_d\) |
| Integral windup | Overshoot after saturation | Clamp; back-calculation |
| Sample jitter | Phase noise | Fixed-rate loop; timestamp |
| Quantisation | Limit cycle | Dither; higher resolution |
| First sample | Derivative spike | Initialize previous error |

The derivative on measurement avoids the setpoint kick: the derivative is computed on the measured value, not on the error. The change is small in code and large in the response.

### 5.23 Filters and noise

A derivative term amplifies high-frequency noise. A first-order low-pass at \(f_c\) reduces the noise but adds phase lag:

\[
\phi = -rctan\left(rac{f}{f_c}
ight)
\]

At 70 Hz and a 200 Hz filter, the lag is 19°. The lag reduces the phase margin. The filter corner must be high enough for the loop and low enough for the noise.

A second-order filter adds more lag. The choice is a trade between noise and margin, and it must be included in the stability analysis.

### 5.24 Gravity and friction compensation

A feed-forward term can compensate for gravity and friction without waiting for the error:

\[
	au_{	ext{ff}} = G(q) + F_c \operatorname{sign}(\dot{q}) + F_v \dot{q}
\]

The gravity term is model-based; the friction terms are identified from experiments. A good feed-forward reduces the integral gain needed and improves the tracking.

The feed-forward is not a substitute for feedback. A model error in the gravity term becomes a steady error that the integral must remove. The feed-forward reduces the error; the feedback corrects the residual.

### 5.25 Ten controller implementation checks

- The loop rate is fixed and measured.
- The derivative is on measurement, not error.
- The integral is clamped or back-calculated.
- The actuator command is saturated before output.
- The sensor is checked for validity.
- The delay is measured and budgeted.
- The safe state is reachable in one cycle.
- The gains are recorded with the firmware.
- The unit tests cover saturation and dropout.
- The fault injection covers sensor loss and communication loss.

## 6. Worked example

### 6.1 The joint

A robot joint has inertia \(J = 0.01\) kg m² and damping \(b = 0.1\) Nms/rad. The actuator can deliver 5 Nm. The controller runs at 1 kHz. The desired closed-loop response is a natural frequency of 70.7 rad/s and a damping ratio of about 0.78.

### 6.2 The gains

The proportional gain is:

\[
K_p = J \omega_n^2 = 0.01 \times 70.7^2 = 50
\]

The derivative gain is:

\[
K_d = 2\zeta\sqrt{K_p J} - b = 2 \times 0.78 \times \sqrt{0.5} - 0.1 = 1.0
\]

The integral gain is set to 100, about 1.4 times the natural frequency, for disturbance rejection. The gains are \(K_p = 50\), \(K_i = 100\), \(K_d = 1.0\).

### 6.3 The linear response

The closed-loop natural frequency is 70.7 rad/s and the damping ratio is 0.78. A second-order system with these parameters has about 2% overshoot and a 2% settling time of:

\[
t_s \approx \frac{4}{\zeta \omega_n} = \frac{4}{0.78 \times 70.7} = 0.073\ \text{s}
\]

The linear prediction is 73 ms. The actual simulation gives 75 ms, which is close. The overshoot, however, is 10.1%, not 2%.

### 6.4 The simulated response

The discrete simulation at 1 kHz with a 5 Nm saturation gives:

| Metric | Value |
|---|---:|
| Overshoot | 10.1% |
| Settling to 2% | 75 ms |
| Final value | 1.005 rad |
| Steady-state error | 0.5% |

The overshoot is larger than the linear prediction because the controller saturates at the start of the step and the integral winds up. The final error is 0.5% because the integral has not fully converged at 1.5 s.

The saturation is the cause. A 1 rad step with \(K_p = 50\) commands 50 Nm, ten times the 5 Nm limit. The actuator saturates for the first part of the response, and the integral accumulates the error.

### 6.5 Anti-windup

The controller is changed to clamp the integral term to ±0.2 rad·s. The overshoot falls to about 4%, and the settling time to about 85 ms. The clamp reduces the overshoot at the cost of a slower disturbance rejection.

A second option is back-calculation: when the actuator saturates, the integral is reduced by the difference between the commanded and the actual torque. The back-calculation is smoother but requires the actuator model.

The choice depends on the task. A surgical robot that must not overshoot uses the clamp; a robot that must reject a disturbance quickly uses a larger integral with back-calculation.

### 6.6 Impedance control

The robot is switched to impedance control with \(K = 1,000\) N/m and \(B = 50\) Ns/m. A 10 N force is applied. The steady-state displacement is:

\[
x = \frac{F}{K} = \frac{10}{1,000} = 0.010\ \text{m} = 10\ \text{mm}
\]

The time constant is:

\[
\tau = \frac{B}{K} = \frac{50}{1,000} = 0.05\ \text{s}
\]

The bandwidth is about \(1/(2\pi\tau) = 3.2\) Hz. The robot complies by 10 mm under 10 N, which is a soft impedance suitable for tissue interaction.

If the stiffness is raised to 10,000 N/m, the displacement falls to 1 mm, and the robot is stiff. The choice depends on the tissue: a stiff interaction for bone, a soft one for soft tissue.

### 6.7 Diagnostic 1: the loop oscillates in contact

The robot is stable in free space but oscillates at about 15 Hz when it touches tissue. The admittance controller has a 5 ms force-sensor delay and a 1,000 N/m environment stiffness.

The diagnostic is to measure the loop's phase margin with the environment attached. The delay adds \(2\pi \times 15 \times 0.005 = 0.47\) rad, or 27°, at 15 Hz. The margin is consumed by the delay and the environment stiffness.

The design flaw is the gain, not the sensor. The admittance gain is too high for the delay and the stiffness. The fix is to reduce the gain, add a filter, or increase the loop rate.

### 6.8 Diagnostic 2: the force limit is too slow

The robot is commanded into a rigid phantom at 0.5 m/s. The force limit is 20 N, but the measured peak is 45 N. The force sensor has a 2 ms filter, the controller has a 1 ms loop, and the stop takes 5 ms.

The diagnostic is to measure the time from the force threshold to the power removal. The total is 8 ms. At 0.5 m/s, the robot travels 4 mm before the stop, and the force overshoots by 25 N.

The design flaw is the stop time, not the threshold. The fix is a faster stop, a lower approach speed, or a compliant cover that limits the contact force.

### 6.9 Vary one assumption: the loop rate falls to 100 Hz

Change the loop rate from 1 kHz to 100 Hz, keeping the gains. The delay increases from 1 ms to 10 ms, and the phase lag at 70.7 rad/s becomes 0.707 rad, or 40.5°. The phase margin falls from about 70° to about 30°, and the response overshoots more.

The simulation at 100 Hz gives an overshoot above 30% and a settling time above 150 ms. The loop is still stable, but the response is poor. At 50 Hz, the loop may become unstable.

The consequence is that the loop rate is part of the controller design. A gain that is stable at 1 kHz is not necessarily stable at 100 Hz.

### 6.10 What the example demonstrates

The arithmetic closes: \(K_p = 50\), \(K_d = 1.0\), 10.1% overshoot, 75 ms settling, 0.5% steady error, 10 mm compliance at 1,000 N/m, and a 27° phase loss at 15 Hz with a 5 ms delay. The two diagnostics reveal different flaws: one is a gain that is too high for the delay, the other is a stop that is too slow. The sensitivity analysis shows that the loop rate changes the stability.

The general method is to model the plant, set the gains from the desired response, simulate with saturation and delay, tune on the real plant, and verify the force and the stop.

### 6.11 Force control numbers

A force controller with \(K_f = 0.001\) m/N is commanded to 10 N against a 1,000 N/m environment. The steady-state displacement is 10 mm, and the force is 10 N. If the environment stiffness rises to 10,000 N/m, the same gain produces a loop gain of 10, which is unstable.

The stability limit is \(K_f K_e < 1\). At \(K_e = 10,000\) N/m, the maximum gain is 0.0001 m/N. The controller must reduce the gain or use a different law for stiff contact.

### 6.12 Verification matrix

| Test | Condition | Metric | Acceptance |
|---|---|---|---|
| Step response | Free space | Overshoot | < 10% |
| Step response | With tool | Overshoot | < 15% |
| Settling | Free space | 2% settling | < 100 ms |
| Phase margin | With delay | Margin | > 45° |
| Force step | Soft phantom | Overshoot | < 20% |
| Force limit | Rigid phantom | Peak force | < 25 N |
| Stop time | Maximum speed | Time | < 10 ms |
| Sensor loss | Force sensor disconnected | Safe state | Within 10 ms |

The matrix covers the free-space and contact behaviour. A controller that passes only the free-space tests is not verified for the procedure.

### 6.13 Safety distance

At 0.5 m/s and an 8 ms stop, the robot travels:

\[
d = 0.5 \times 0.008 = 0.004\ \text{m} = 4\ \text{mm}
\]

The force overshoot depends on the environment stiffness. Against a 10,000 N/m phantom, a 4 mm travel produces 40 N, which exceeds the 20 N limit. The design must reduce the speed, reduce the stop time, or add compliance.

At 0.1 m/s, the travel is 0.8 mm, and the force overshoot is 8 N, within the limit. The speed and the stop time are a coupled safety requirement.

### 6.14 The 100 Hz loop

At 100 Hz, the sample period is 10 ms. The discrete derivative is computed over 10 ms, and the delay is at least 10 ms. The phase lag at 70.7 rad/s is 40.5°, and the margin falls from about 70° to about 30°.

The simulation at 100 Hz gives an overshoot above 30% and a settling time above 150 ms. The loop is stable but poorly damped. A gain reduction of 30% restores the margin at the cost of a slower response.

### 6.15 Impedance versus admittance for the same task

For a soft tissue contact, an impedance controller with \(K = 1,000\) N/m is stable and compliant. An admittance controller with the same stiffness and a 5 ms force delay has a lower phase margin because the force loop includes the delay.

If the environment is stiff (bone), the admittance controller is more stable because it commands position and the position loop is stiff. The choice depends on the environment, not on a universal rule.

The comparison is not absolute. A well-tuned admittance controller can be stable on soft tissue, and a well-tuned impedance controller can be stable on bone. The difference is the margin and the delay, not the label.

### 6.16 A note on the safety stop

The safety stop is not a control function. It removes power or engages a brake, independently of the controller. The stop time includes the brake engagement, the power removal, and the mechanical travel. The design must measure the total, not the controller's response time.

### 6.17 A second joint with a different inertia

A second joint has \(J = 0.02\) kg m² and \(b = 0.2\) Nms/rad. For the same \(\omega_n = 70.7\) rad/s and \(\zeta = 0.78\), the gains are \(K_p = 100\) and \(K_d = 1.9\). The torque for a 1 rad step is 100 Nm, which exceeds a 5 Nm actuator. The joint must use a lower natural frequency or a larger actuator.

At \(\omega_n = 35\) rad/s, \(K_p = 25\) and \(K_d = 0.9\), and the torque is 25 Nm, still above 5 Nm. The actuator limits the achievable bandwidth. The control design must start from the actuator, not from the desired response.

### 6.18 A note on units

The gains have units: \(K_p\) in Nm/rad, \(K_i\) in Nm/(rad·s), \(K_d\) in Nm·s/rad. A gain quoted without units is ambiguous. A \(K_p\) of 50 is 50 Nm/rad for a joint, but 50 N/m for a Cartesian impedance. The unit determines the interpretation.

### 6.19 Summary of the worked numbers

| Quantity | Value | Basis |
|---|---:|---|
| \(K_p\) | 50 Nm/rad | \(J\omega_n^2\) |
| \(K_d\) | 1.0 Nm·s/rad | \(2\zeta\sqrt{K_pJ}-b\) |
| \(K_i\) | 100 Nm/(rad·s) | Disturbance rejection |
| Overshoot | 10.1% | Simulation |
| Settling | 75 ms | Simulation |
| Steady error | 0.5% | Simulation at 1.5 s |
| Compliance | 10 mm | \(F/K\) at 1,000 N/m |
| Impedance time constant | 50 ms | \(B/K\) |
| Phase lag at 15 Hz | 27.0° | 5 ms delay |
| Stop travel | 4 mm | 0.5 m/s, 8 ms |

The table is the evidence for the design review. Each number has a basis, and each basis can be checked.

### 5.28 Controller timing budget

| Stage | Time | Notes |
|---|---:|---|
| Sensor sample | 0.1 ms | ADC and bus |
| Filter | 1.0 ms | First-order at 200 Hz |
| Control computation | 0.2 ms | Cortex-M4 at 100 MHz |
| Actuator command | 0.1 ms | DAC or PWM |
| Actuator response | 1.0 ms | Current loop |
| **Total** | **2.4 ms** | Plus network |

The total is 2.4 ms, or 2.4 samples at 1 kHz. The delay is small enough for a 70 rad/s loop, but not for a 700 rad/s loop. The budget must be recomputed when the loop rate changes.

- Measure the brake engagement time.
- Measure the power removal time.
- Measure the mechanical travel.
- Add the controller cycle.
- Add the sensor filter.
- Add the communication delay.
- Verify with a fault injection.
## 7. Code, query, or calculation

The script reproduces the PID simulation, the gain calculation, and the impedance response in Section 6. It uses a simple Euler integration at 1 kHz with a 5 Nm saturation.

**Assumptions.** \(J = 0.01\) kg m², \(b = 0.1\) Nms/rad, \(K_p = 50\), \(K_i = 100\), \(K_d = 1.0\), a 5 Nm torque limit, a 1 rad step, and a 1 ms sample period.

```python
import math

J = 0.01
b = 0.1
Kp, Ki, Kd = 50.0, 100.0, 1.0
tau_max = 5.0
dt = 1e-3
T = 1.5
theta, omega, integ = 0.0, 0.0, 0.0
prev_e = 1.0
target = 1.0
max_over = 0.0
settle = None

for k in range(int(T/dt)):
    t = k*dt
    e = target - theta
    integ += e*dt
    deriv = (e - prev_e)/dt
    tau = Kp*e + Ki*integ + Kd*deriv
    tau = max(-tau_max, min(tau_max, tau))
    alpha = (tau - b*omega)/J
    omega += alpha*dt
    theta += omega*dt
    prev_e = e
    if theta > 1:
        max_over = max(max_over, theta - 1)
    if settle is None and abs(theta - 1) < 0.02:
        settle = t

print(f"overshoot {100*max_over:.2f} %")
print(f"settling 2% {settle*1000:.1f} ms")
print(f"final theta {theta:.4f}")
wn = math.sqrt(Kp/J)
zeta = (b + Kd)/(2*math.sqrt(Kp*J))
print(f"wn {wn:.2f} rad/s zeta {zeta:.3f}")
print(f"impedance x {10/1000*1000:.1f} mm, tau {50/1000*1000:.1f} ms")
```

**Expected output.**

```text
overshoot 10.13 %
settling 2% 75.0 ms
final theta 1.0050
wn 70.71 rad/s zeta 0.778
impedance x 10.0 mm, tau 50.0 ms
```

The output shows the difference between the linear prediction and the simulated response. The linear model predicts about 2% overshoot and 73 ms settling; the simulation gives 10.1% and 75 ms. The settling time is close, but the overshoot is five times larger because of saturation and integral windup.

The final value of 1.005 rad is a 0.5% steady-state error at 1.5 s. The integral is still converging; a longer simulation would reduce the error. In a real robot, the integral would continue to reduce the error until the sensor noise or the friction limits it.

The impedance calculation gives a 10 mm displacement for a 10 N force at 1,000 N/m, with a 50 ms time constant. The response is slow enough to be stable with a 1 kHz loop and fast enough for tissue interaction.

**A note on the simulation.** The Euler integration at 1 ms is adequate for a 70 rad/s loop, but it is not exact. A smaller step or a higher-order integrator would change the overshoot slightly. The simulation is a design tool, not a verification.

### 7.1 Phase lag and stop travel

The script computes the phase lag from a 5 ms delay at three frequencies and the travel during an 8 ms stop at 0.5 m/s.

```python
import math

for f in (5, 12, 15):
    lag = math.degrees(2*math.pi*f*0.005)
    print(f"{f:2d} Hz: {lag:5.1f} deg")
print(f"travel {0.5*0.008*1000:.1f} mm")
```

**Expected output.**

```text
 5 Hz:   9.0 deg
12 Hz:  21.6 deg
15 Hz:  27.0 deg
travel 4.0 mm
```

The lag grows linearly with frequency. At 15 Hz, a 5 ms delay consumes 27° of phase margin, which is a large fraction of a 45° budget. The travel at 0.5 m/s and 8 ms is 4 mm, which is the distance available for the force to rise before the stop.

The two numbers explain the two diagnostics: the oscillation in contact and the force overshoot. The delay and the stop time are not tuning details; they are the safety margin.

## 8. Data, standards, and regulatory context

Robot control is constrained by the medical robot standards, the risk management standard, and the software lifecycle standard.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 80601-2-77:2019 | Robotically assisted surgical equipment | Particular standard | Control, accuracy, safety, emergency stop |
| IEC 80601-2-78:2021 | Rehabilitation and assistive robots | Particular standard | Control, patient contact, assist-as-needed |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment | Consensus standard | Electrical safety, isolation |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis, force limits |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Control software, verification |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | Mode awareness, controls |
| ISO/TS 15066:2016 | Collaborative robots | Informative; not a medical substitute | Force and pressure limits |

### 8.1 What the robot standards require

IEC 80601-2-77:2019 requires the manufacturer to specify the robot's control performance, the accuracy, the workspace, the emergency stop, and the interaction with the patient. The standard does not prescribe PID gains or an impedance law; it requires evidence that the control is safe and effective.

IEC 80601-2-78:2021 adds requirements for rehabilitation robots, including the patient contact, the assistance behaviour, and the safe stop. A rehabilitation robot that assists a patient's movement must limit the force and the speed, and must detect a patient's resistance.

### 8.2 Force limits

ISO/TS 15066:2016 provides force and pressure limits for collaborative industrial robots. It is not a medical standard, but it is often used as a reference for the force limits in a medical robot's risk analysis. The limits depend on the body region and the contact duration.

The medical manufacturer must derive its own limits from the clinical literature, the tissue properties, and the risk analysis. A force limit copied from an industrial standard may be too high for a patient.

### 8.3 What the standards do not cover

The standards do not specify:

- The control law.
- The gains.
- The loop rate.
- The filter.
- The anti-windup method.

Those are design decisions that the manufacturer must justify. A robot can meet every clause and still oscillate in contact because the gain was tuned in free space.

### 8.4 The data implications

The gains, the loop rate, the filter, the firmware version, and the force limit are part of the controller. A change in any of them changes the behaviour. The record must include them, and the monitoring must detect a drift.

## 9. Industry practice and case evidence

### 9.1 Surgical robot control

A teleoperated surgical robot uses a master-slave control loop. The surgeon's hand motion is measured, scaled, filtered, and sent to the slave. The slave's position is controlled by a joint-level loop, and the force feedback, if present, is sent back to the master.

The loop has a delay from the master to the slave and back. A 20 ms delay is noticeable; a 100 ms delay makes teleoperation difficult. The control design must compensate for the delay or limit the task's speed.

**Documented fact:** the architecture is described in the surgical robotics literature and in product documentation. **Inference:** the specific gains and delays are not public.

### 9.2 Orthopedic robot control

An orthopedic robot uses a position-controlled arm with a force limit. The surgeon guides the arm, or the robot follows a planned trajectory. The control loop is stiff to resist the cutting forces, and the force limit prevents a large force if the tool jams.

The robot's accuracy depends on the registration and the stiffness of the arm. A stiff position loop reduces the deflection under load but increases the force if the tool hits a hard surface.

**Documented fact:** the systems and their indications are public. **Inference:** the control parameters are not public.

### 9.3 Rehabilitation robot control

A rehabilitation robot uses impedance or assist-as-needed control. The robot must comply with the patient's movement and assist only when needed. The control must detect the patient's intent from force, EMG, or motion.

The evidence for the clinical benefit is mixed. A 2024 umbrella review of the Lokomat found small effects and no convincing evidence. The control design is one reason: an assist-as-needed controller that actually assists too much can reduce the patient's effort.

**Documented fact:** the review is public. **Inference:** the relationship between the control law and the outcome is device- and protocol-specific.

### 9.4 What good control practice looks like

Teams that avoid control surprises tend to:

- Model the plant with the tool and the environment.
- Set the gains from the desired response.
- Simulate with saturation and delay.
- Tune on the real plant with the real payload.
- Measure the phase margin with the delay.
- Verify the force limit and the stop time.
- Record the gains and the firmware.
- Re-verify after a tool or payload change.

The practices are ordinary. The failures they prevent are not.

### 9.6 A note on the control interface

The control interface determines what the controller can do. A robot with a torque interface can implement impedance; a robot with only a position interface must use admittance. A robot with a velocity interface can implement either, with a conversion.

The interface is often a vendor limitation, not a design choice. The control engineer must know the interface before choosing the law. A design that assumes a torque interface and receives a position interface will not work.

### 9.5 A note on controller tuning in the field

A controller tuned in the field without a record is a controller that cannot be reproduced. The tune may improve one procedure and worsen another. The gains must be locked, recorded, and verified after any change.

A service engineer who adjusts a gain to "make it feel better" is changing the validated configuration. The change must go through the change-control process, with a verification.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Instability in contact | Oscillation at contact | Gain too high for stiffness | Uncontrolled motion | Reduce gain; filter; impedance |
| Overshoot | Peak above setpoint | Low damping; windup | Tissue damage | Increase \(K_d\); anti-windup |
| Steady-state error | Offset after settling | No integral; friction | Position error | Integral; feed-forward |
| Integral windup | Overshoot after saturation | Integral accumulates | Large overshoot | Clamp; back-calculation |
| Derivative noise | Buzz in the output | Noise amplified | Wear; instability | Filter; derivative on measurement |
| Delay-induced instability | Margin falls with distance | Network or filter delay | Oscillation | Reduce gain; increase rate |
| Force overshoot | Peak above limit | Slow stop; high speed | Injury | Faster stop; lower speed |
| Sensor dropout | Controller uses stale value | No validity check | Wrong motion | Validity check; safe state |
| Actuator saturation | Command exceeds limit | Large step; high gain | Overshoot; delay | Limit; anti-windup |
| Backlash limit cycle | Small oscillation at reversal | Clearance | Poor accuracy | Compensation; mechanic fix |
| Gravity uncompensated | Sag under load | No feed-forward | Steady error | Gravity model |
| Mode confusion | Wrong control mode | UI or state machine | Wrong motion | Mode awareness; interlocks |
| Safety monitor failure | No stop on fault | Shared processor | Injury | Independent monitor |
| Gain change unrecorded | Behaviour differs between units | Field tuning | Non-reproducible | Record and lock |

The most dangerous failure is the safety monitor that shares the controller. A fault in the controller can disable the monitor. The monitor must be independent, with its own sensor and its own stop path.

### 10.1 Ten red flags in control data

1. An oscillation that appears only in contact.
2. An overshoot that grows with the payload.
3. A steady error that the integral does not remove.
4. A torque command that sits at the limit.
5. A phase margin that is not measured.
6. A force peak above the limit.
7. A stop time that is longer than the budget.
8. A gain that differs between units.
9. A sensor dropout that does not trigger a safe state.
10. A mode change that is not interlocked.

Each is a prompt to measure, not a conclusion.

### 10.2 What to do when the loop oscillates

1. Measure the frequency and the amplitude.
2. Measure the loop delay.
3. Compute the phase lag at the oscillation frequency.
4. Reduce the gain or add a filter.
5. Re-test in the same contact condition.
6. Record the new gain and the test.

An oscillation is a measurement problem before it is a tuning problem. The frequency and the delay usually identify the cause.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The plant model is accurate | Gain selection | Friction, compliance, payload | System identification |
| The environment stiffness is known | Interaction stability | Tissue variability | Phantom tests |
| The delay is known | Stability margin | Network, filter, sensor | Measure the loop |
| The loop rate is fixed | Discrete analysis | Scheduling, load | Trace |
| The sensor is valid | Feedback | Dropout, saturation | Validity check |
| The actuator is linear | Control design | Saturation, friction | Actuator test |
| The force limit is below injury | Safety | Wrong limit | Risk analysis |
| The stop is independent | Safety | Shared processor | Fault injection |
| The gains are constant | Reproducibility | Field tuning | Record and lock |
| The payload is known | Gain selection | Tool change | Payload test |
| The patient is passive | Interaction | Voluntary motion | Assist-as-needed |
| The controller is verified | Evidence | Missing test | Verification matrix |

The dominant assumption is that the environment stiffness is known. It varies from air to bone, and the controller must be stable across the range. A gain tuned for soft tissue may be unstable on bone.

A second limit is that the model is linear. Friction, backlash, and saturation are nonlinear, and they change the response. The linear analysis is a starting point, not a guarantee.

A third limit is that the controller is verified on a phantom, not a patient. The patient's motion, perfusion, and anatomy change the interaction. The clinical validation is separate from the bench verification.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a model?
- Which assumption was true in free space but false in contact?
- Which assumption depends on the tool, the payload, or the environment?
- Which assumption is encoded in a gain or a filter?
- Which assumption is specific to the validation phantom?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Compute gains (recall, 20 minutes)

For \(J = 0.02\) kg m², \(b = 0.2\) Nms/rad, \(\omega_n = 50\) rad/s, \(\zeta = 0.8\), compute \(K_p\) and \(K_d\).

**Deliverable:** two calculations.

**Acceptance criteria:** formulas; units; values.

**Time:** 20 minutes.

### Exercise 2. Predict the response (application, 20 minutes)

For the gains in Exercise 1, compute the settling time and the expected overshoot.

**Deliverable:** two values with the formulas.

**Acceptance criteria:** settling formula; overshoot estimate; assumptions.

**Time:** 20 minutes.

### Exercise 3. Simulate a saturated loop (application, 30 minutes)

Simulate the loop in Exercise 1 with a 2 Nm limit and a 1 rad step. Measure the overshoot and the settling. Compare with the linear prediction.

**Deliverable:** a script and a table.

**Acceptance criteria:** simulation runs; metrics; comparison.

**Time:** 30 minutes.

### Exercise 4. Compute an impedance (application, 20 minutes)

An impedance controller has \(K = 2,000\) N/m and \(B = 100\) Ns/m. A 20 N force is applied. Compute the displacement and the time constant.

**Deliverable:** two values.

**Acceptance criteria:** correct formulas; units.

**Time:** 20 minutes.

### Exercise 5. Diagnose an oscillation (critique, 30 minutes)

A robot oscillates at 12 Hz in contact. The delay is 8 ms and the environment stiffness is 5,000 N/m. Explain the mechanism and propose two fixes.

**Deliverable:** a diagnosis and two fixes.

**Acceptance criteria:** phase lag; gain; fixes.

**Time:** 30 minutes.

### Exercise 6. Design a force limit (design, 30 minutes)

A robot approaches tissue at 0.2 m/s. The force limit is 15 N, the stop time is 10 ms, and the tissue stiffness is 2,000 N/m. Compute the force overshoot and decide whether the limit is safe.

**Deliverable:** a calculation and a verdict.

**Acceptance criteria:** travel; force; comparison.

**Time:** 30 minutes.

### Exercise 7. Write a verification plan (design, 40 minutes)

Write a control verification plan for the joint in Section 6. Include the step response, the delay, the saturation, the force limit, and the sensor dropout.

**Deliverable:** a plan with at least 8 rows.

**Acceptance criteria:** free space and contact; quantitative criteria; fault injection.

**Time:** 40 minutes.

### Exercise 8. Evaluate a control claim (critique, 25 minutes)

A vendor claims "stable control in all conditions." Write five questions that would test the claim.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover environment, delay, payload, saturation, and verification; no question is answerable by the headline.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Gains | Calculations |
| 2 | Response | Settling and overshoot |
| 3 | Simulation | Script and table |
| 4 | Impedance | Displacement and time constant |
| 5 | Diagnosis | Mechanism and fixes |
| 6 | Force limit | Calculation |
| 7 | Verification | Test plan |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** \(K_p = J\omega_n^2 = 0.02 \times 2,500 = 50\). \(K_d = 2\zeta\sqrt{K_p J} - b = 2 \times 0.8 \times \sqrt{1.0} - 0.2 = 1.4\).

**Answer 2 (Exercise 2).** \(t_s \approx 4/(\zeta\omega_n) = 4/(0.8 \times 50) = 0.1\) s. The overshoot for \(\zeta = 0.8\) is about 1.5%.

**Answer 3 (Exercise 3).** With a 2 Nm limit, the initial command is 50 Nm, so the actuator saturates. The simulated overshoot is about 12%, and the settling is about 140 ms. The linear prediction underestimates both because it ignores saturation.

**Answer 4 (Exercise 4).** \(x = F/K = 20/2,000 = 0.010\) m = 10 mm. \(\tau = B/K = 100/2,000 = 0.05\) s.

**Answer 5 (Exercise 5).** At 12 Hz, the delay of 8 ms adds \(2\pi \times 12 \times 0.008 = 0.60\) rad, or 34.6°. The environment stiffness adds gain. The loop's phase margin is consumed. Fixes: reduce the gain, add a low-pass filter below 12 Hz, or increase the loop rate.

**Answer 6 (Exercise 6).** Travel = 0.2 m/s × 0.010 s = 2 mm. Force from travel = 2,000 N/m × 0.002 m = 4 N. The peak force is 15 + 4 = 19 N, above the 15 N limit but below a 20 N injury threshold. The limit is not met; the speed or the stop time must be reduced.

**Answer 7 (Exercise 7).** Tests: step response in free space; step with the tool; phase margin with 5 ms delay; saturation at 5 Nm; force step on a soft phantom; force limit on a rigid phantom; sensor dropout; communication loss. Acceptance: overshoot < 20%, settling < 100 ms, margin > 45°, peak force < 25 N, safe state < 10 ms.

**Answer 8 (Exercise 8).** Questions: (1) Which environments were tested? (2) What is the loop delay? (3) What payload range? (4) What happens under saturation? (5) What fault-injection evidence is provided? "All conditions" is not a testable claim.

**Rubric for the design exercises.** A strong design quantifies the plant, the gains, the delay, the saturation, the force limit, and the verification. A weak design quotes a gain and assumes stability.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: \(K_p\) | 50 | ±1 |
| Exercise 1: \(K_d\) | 1.4 | ±0.05 |
| Exercise 2: settling | 100 ms | ±5 ms |
| Exercise 2: overshoot | 1.5% | ±0.5% |
| Exercise 4: displacement | 10 mm | ±0.5 mm |
| Exercise 4: time constant | 50 ms | ±2 ms |
| Exercise 6: peak force | 19 N | ±1 N |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 10 | 5 | 20 |
| 2 | 5 | 10 | 5 | 20 |
| 3 | 5 | 20 | 5 | 30 |
| 4 | 5 | 10 | 5 | 20 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 15 | 10 | 30 |
| 7 | 5 | 25 | 10 | 40 |
| 8 | 5 | 10 | 10 | 25 |

## 14. Interview questions

**1. What does the proportional term do?**
A strong answer explains stiffness and error reduction, and the overshoot trade. A weak answer says it "reduces error".

**2. What does the derivative term do?**
A strong answer explains damping and noise amplification. A weak answer says it "predicts".

**3. What is integral windup?**
A strong answer explains accumulation during saturation and the overshoot. A weak answer has not heard of it.

**4. What is the difference between impedance and admittance control?**
A strong answer explains measure/command and the environment dependence. A weak answer uses the terms interchangeably.

**5. How does delay affect stability?**
A strong answer explains phase lag and margin loss. A weak answer says delay "slows the response".

**6. How do you choose a loop rate?**
A strong answer uses 10–20 times the bandwidth and the delay budget. A weak answer picks a round number.

**7. How do you set a force limit?**
A strong answer derives it from tissue tolerance, speed, and stop time. A weak answer copies a number.

**8. Why is the safety monitor independent?**
A strong answer explains that a controller fault can disable a shared monitor. A weak answer says redundancy.

**9. What is assist-as-needed?**
A strong answer explains the assistance proportional to the deficit and the risk of doing too much. A weak answer says "the robot helps".

**10. What would make you reject a controller?**
A strong answer names instability in contact, an unmeasured delay, a slow stop, or an unrecorded gain. A weak answer rejects on overshoot alone.

**11. Why is a force sensor's bandwidth important?**
A strong answer explains that the sensor delay adds phase lag and limits the gain. A weak answer says the sensor must be fast.

**12. What is gain scheduling?**
A strong answer explains that the plant changes with configuration and the gains must change with it. A weak answer says it is auto-tuning.

**13. How do you verify a force limit?**
A strong answer measures the peak force against a rigid phantom at the maximum speed and stop time. A weak answer checks the setting.

**14. Why is a control loop's phase margin not constant?**
A strong answer explains that the delay, the environment, and the payload change the plant. A weak answer assumes the bench margin holds.

## 15. Key terms

- **Admittance** — motion-force relationship.
- **Anti-windup** — mitigation of integral accumulation.
- **Back-calculation** — anti-windup using the saturation error.
- **Bandwidth** — frequency range of the closed loop.
- **Compliance** — displacement per force.
- **Damping ratio** — dimensionless damping.
- **Dead time** — delay with no response.
- **Derivative** — control term on the rate of error.
- **Feed-forward** — control based on a model.
- **Force control** — regulation of contact force.
- **Gain margin** — stability margin in gain.
- **Gravity compensation** — feed-forward torque for gravity.
- **Hybrid control** — position and force in different directions.
- **Impedance** — force-motion relationship.
- **Integral** — control term on accumulated error.
- **Loop rate** — controller update frequency.
- **Natural frequency** — undamped oscillation frequency.
- **Overshoot** — peak above the setpoint.
- **Phase margin** — stability margin in phase.
- **PID** — proportional-integral-derivative.
- **Plant** — system being controlled.
- **Proportional** — control term on the error.
- **Saturation** — actuator at its limit.
- **Settling time** — time to within a band.
- **Stiffness** — force per displacement.
- **Setpoint** — desired value.
- **Steady-state error** — error after the transient.
- **Torque limit** — maximum actuator torque.
- **Velocity limit** — maximum joint velocity.
- **Windup** — integral accumulation during saturation.
- **Backlash compensation** — correction for mechanical clearance.
- **Control mode** — position, force, or impedance.
- **Force sensor** — transducer for contact force.
- **Gain scheduling** — configuration-dependent gains.
- **Interaction control** — control of contact behaviour.
- **Loop gain** — product of controller and plant gain.
- **Mode awareness** — user knowledge of the active mode.
- **Safety relay** — hardware device that removes power.
- **Torque constant** — torque per unit current.
- **Velocity feed-forward** — feed-forward for tracking.
- **Chatter** — high-frequency contact vibration.
- **Control interface** — torque, velocity, or position.
- **Limit cycle** — sustained oscillation from nonlinearity.
- **Loop delay** — total time from sensor to actuator.
- **Mode confusion** — user uncertainty about the active mode.
- **Phase lag** — delay expressed as angle.
- **Stability margin** — distance from instability.
- **Torque interface** — direct torque command.
- **Velocity interface** — velocity command.
- **Back-calculation** — anti-windup using the saturation error.
- **Force gain** — gain on the force error.
- **Loop shaping** — design of the open-loop frequency response.
- **Plant inversion** — feed-forward from the inverse model.
- **Stiffness control** — control of the force-displacement slope.
- **Trade-off** — the balance between speed and stability.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards**

1. IEC 80601-2-77:2019, *Robotically assisted surgical equipment*. https://webstore.iec.ch/en/publication/29933
2. IEC 80601-2-78:2021, *Medical robots for rehabilitation, assessment, compensation or alleviation*. https://webstore.iec.ch/en/publication/69618
3. IEC 60601-1:2005+AMD1:2012+AMD2:2020. https://webstore.iec.ch/en/publication/26169
4. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
5. IEC 62304:2006+AMD1:2015, *Medical device software*. https://webstore.iec.ch/en/publication/22794
6. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
7. ISO/TS 15066:2016, *Collaborative robots*. https://www.iso.org/standard/62996.html

**Devices and evidence**

8. Intuitive Surgical, da Vinci 5, March 2024. https://isrg.intuitive.com/news-releases/news-release-details/intuitive-announces-fda-clearance-fifth-generation-robotic
9. Medtronic, Hugo, December 2025. https://news.medtronic.com/2025-12-03-Medtronic-announces-FDA-clearance-of-Hugo-TM-robotic-assisted-surgery-system-for-urologic-surgical-procedures
10. Stryker, Mako 4, 2025. https://www.stryker.com/us/en/about/news/2025/stryker-showcases-next-generation-of-mako-smartrobotics--at-aaos.html
11. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm

**Background**

12. J. J. Craig, *Introduction to Robotics*, 4th ed., Pearson, 2017.
13. B. Siciliano et al., *Robotics: Modelling, Planning and Control*, Springer, 2009.
14. K. Astrom and R. Murray, *Feedback Systems*, 2nd ed., Princeton University Press, 2021. https://fbswiki.org/
15. J. G. Webster (ed.), *Medical Instrumentation*, 4th ed., Wiley, 2009.
16. K. Astrom and R. Murray, *Feedback Systems*, 2nd ed., 2021.
17. G. F. Franklin et al., *Feedback Control of Dynamic Systems*, 8th ed., Pearson, 2019.
18. B. Siciliano et al., *Robotics: Modelling, Planning and Control*, 2009.
19. J. J. Craig, *Introduction to Robotics*, 4th ed., 2017.
20. ISO 14971:2019, risk management.
21. G. F. Franklin et al., *Feedback Control of Dynamic Systems*, 8th ed., 2019.
22. K. Astrom and R. Murray, *Feedback Systems*, 2nd ed., 2021.
23. B. Siciliano et al., *Robotics: Modelling, Planning and Control*, 2009.
24. IEC 80601-2-77:2019, control and safety requirements.
25. IEC 80601-2-78:2021, rehabilitation robot control.
26. ISO/TS 15066:2016, force and pressure limits.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`10_robotics-kinematics.md`](10_robotics-kinematics.md)
- Next topic: [`12_surgical-and-rehab-robotics.md`](12_surgical-and-rehab-robotics.md)
- Related: [`08_firmware-and-rtos.md`](08_firmware-and-rtos.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on safety monitor: see Lab 14A.5 in the survey file.
- Lab on robot kinematics: see Lab 14A.4 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
