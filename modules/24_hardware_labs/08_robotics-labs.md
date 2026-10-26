# Robotics Labs

> Module 24 — Hardware and Robotics Labs
> Topic 8 of 9
> Estimated study time: 5 hours
> Prerequisites: 02, 07, 14A
> Version 2026.3

## 1. Why this topic exists

A robot is a machine that converts a computation into motion. The computation is a pose, a trajectory, or a force. The motion is physical, and its errors are spatial. A millimeter of error in a surgical robot is a clinical event; a millimeter of error in a pick-and-place robot is a missed part.

The industry problem is that the errors are invisible in the model. A kinematic model predicts a pose to six decimal places. The real robot reaches a pose with an error from manufacturing tolerance, backlash, compliance, thermal expansion, encoder calibration, and registration. The model's precision is not the robot's accuracy.

Three decisions depend on this topic:

1. **Whether the kinematic model is correct and verified.** A sign error in a joint axis, a length in millimeters instead of meters, or a missing transform produces a plausible but wrong pose.
2. **Whether the registration is accurate at the target, not only at the fiducials.** A low fiducial registration error (FRE) does not guarantee a low target registration error (TRE). The distinction is the difference between a reassuring number and a clinically meaningful one.
3. **Whether the safety envelope accounts for the real error and the real stopping distance.** A software workspace clamp is not a safety-rated limit. The envelope must include the robot's accuracy, the registration error, the control latency, and the stopping distance.

The commercial stakes are concrete. A registration error can move a robot into a no-go zone. A singularity can produce a joint velocity that the controller cannot follow. A safety envelope that ignores the stopping distance can allow a collision that the guard was supposed to prevent.

This topic covers the robotics lab: kinematics, the Jacobian, singularities, inverse kinematics, registration, accuracy measurement, and the safety envelope.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work. Any moving arm is inside a guard or behind a light curtain, with a reachable hardware emergency stop, joint limits, and a low speed cap. No hands are in the workspace while the arm is enabled. A software workspace clamp is not a safety-rated limit, and a simulated emergency stop is not a certified safety function.

## 2. Learning outcomes

By the end of this topic you can:

- Build a kinematic model of a planar arm and verify its forward kinematics against a hand calculation.
- Compute the geometric Jacobian and verify it against a finite difference.
- Identify singularities from the manipulability measure and explain the joint-velocity consequence.
- Implement damped least squares and compare it with the pseudo-inverse near a singularity.
- Register a robot's coordinate frame to a target frame with fiducials and compute the FRE and the TRE.
- Explain why FRE and TRE are different and why a low FRE does not prove target accuracy.
- Measure the robot's positioning error at the target and compare it with the registration prediction.
- Compute a safety envelope from the robot's accuracy, the registration error, the control latency, and the stopping distance.
- State the limits of a simulated robot result and the conditions under which it transfers to hardware.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A robotics lab is a chain from a planned pose to a physical pose. Each link has a frame, a transform, and an error.

### 3.1 The frame chain

```text
  world --> base --> joint 1 --> joint 2 --> ... --> tool --> target
     |        |          |          |                |
  registration  mounting  encoder  encoder        TCP offset
  error         error     error    error          error
```

Each arrow is a rigid transform. The composition of the transforms gives the tool pose in the world frame. An error in any transform propagates to the tool.

The chain has two kinds of error: a systematic error, which is the same every time, and a random error, which varies. Backlash and compliance are systematic within a direction and random across directions. Encoder noise is random. A calibration can reduce a systematic error; it cannot reduce a random one.

### 3.2 Forward and inverse kinematics

Forward kinematics (FK) maps joint angles to a pose. It is a deterministic function, computed by multiplying the transforms along the chain.

Inverse kinematics (IK) maps a pose to joint angles. It is the inverse problem, and it has three complications:

- **Multiple solutions.** A planar two-link arm has an elbow-up and an elbow-down solution for most targets.
- **No solution.** A target outside the workspace has no solution, and the solver must report failure rather than return a NaN.
- **Singularities.** Near a singular configuration, a small change in the target requires a large change in the joint angles.

The IK is not a function in the mathematical sense unless the solution branch is specified. A controller that switches branches without a plan can command a large, unexpected motion.

### 3.3 The Jacobian

The Jacobian maps joint velocities to end-effector velocities.

\[
\dot{x} = J(q)\dot{q}
\]

The Jacobian is a matrix of partial derivatives. It is the local linearisation of the FK, and it is the basis for velocity control, singularity analysis, and force control.

The Jacobian is verified against a finite difference:

\[
J_i \approx \frac{f(q + h e_i) - f(q - h e_i)}{2h}
\]

with a small \(h\). A mismatch larger than the truncation error indicates a modelling error.

### 3.4 Singularities and manipulability

A singularity is a configuration where the Jacobian loses rank. At a singularity, the robot cannot move in one or more directions, and the inverse Jacobian is undefined.

The manipulability measure is:

\[
w = \sqrt{\det(J J^T)}
\]

A value near zero indicates a near-singular configuration. The measure is not a safety limit; it is a diagnostic.

Near a singularity, the pseudo-inverse solution produces large joint velocities:

\[
\dot{q} = J^T (J J^T)^{-1} \dot{x}
\]

Damped least squares adds a damping term:

\[
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} \dot{x}
\]

The damping reduces the joint velocity at the cost of a position error. The choice of \(\lambda\) trades accuracy near the singularity for smoothness.

### 3.5 Registration

Registration is the estimation of the transform between two coordinate frames from corresponding points.

```text
  image or plan frame ---- transform T ----> robot frame
        |                                        |
     fiducials                                fiducials
        |                                        |
        +------------- correspondence -----------+
```

The transform is estimated by minimising the distance between corresponding points. For rigid registration, the closed-form solution is the orthogonal Procrustes problem, solved by a singular-value decomposition.

The errors have specific names:

| Error | Definition | What it measures |
|---|---|---|
| FLE | Fiducial localisation error | The error in locating a fiducial |
| FRE | Fiducial registration error | The RMS distance between corresponding fiducials after registration |
| TRE | Target registration error | The distance between corresponding non-fiducial points after registration |

The FRE is what the registration algorithm minimises. The TRE is what the clinician cares about. They are not the same, and a low FRE does not guarantee a low TRE.

### 3.6 The error stack at the target

The error at the target is the sum of the errors along the chain.

| Source | Typical magnitude | Reducible by |
|---|---|---|
| Encoder resolution | 0.01–0.1 mm | A finer encoder |
| Backlash | 0.05–0.5 mm | A preloaded gearbox |
| Link geometry | 0.1–1 mm | Calibration |
| Compliance under load | 0.1–2 mm | A stiffer structure or a load model |
| Thermal expansion | 0.01–0.1 mm/°C | Warm-up, compensation |
| Registration | 0.5–5 mm | More or better fiducials |
| Control latency | 0.1–5 mm at speed | A faster loop |

The stack is not a simple sum. Some errors are systematic and can be calibrated; some are random and add in quadrature. The report must state which is which.

### 3.7 The safety envelope

The safety envelope is the set of positions, velocities, and forces within which the robot may operate.

```text
  workspace boundary
     |
     |   +-------------------+
     |   |   reachable       |
     |   |   workspace       |
     |   |     +-------+     |
     |   |     | safe  |     |
     |   |     | zone  |     |
     |   |     +-------+     |
     |   +-------------------+
     |
  guard / light curtain
```

For speed and separation monitoring (SSM), the protective separation distance is the sum of the contribution from the human's motion, the robot's reaction, the robot's stopping distance, the sensing uncertainty, and the position uncertainty. ISO 10218-2:2025 Annex L defines the calculation; a lab can use a simplified version and label it as a model.

A simplified separation budget:

\[
S = v_h t_r + v_r t_r + \frac{v_r^2}{2a} + C + Z
\]

where \(v_h\) is the human's speed, \(v_r\) is the robot's speed, \(t_r\) is the reaction time, \(a\) is the deceleration, \(C\) is the sensing and position uncertainty, and \(Z\) is a margin.

The terms must be measured or justified. A budget that assumes a reaction time of zero is not a safety case.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Frame | Coordinate system with an origin and axes | Confused with a link | None |
| Transform | Rigid translation and rotation between frames | Confused with a pose | 4×4 matrix |
| Pose | Position and orientation of a frame | Confused with a transform | m and rad |
| Homogeneous transform | 4×4 matrix representing a rigid transform | A 3×3 rotation only | Matrix |
| Quaternion | Four-parameter rotation representation | Stored in the wrong order | x, y, z, w |
| DH parameters | Four parameters describing a joint-link transform | Confused with URDF | m and rad |
| URDF | XML description of links, joints, and geometry | A dynamics model | XML |
| MJCF | MuJoCo XML model format | A URDF | XML |
| Forward kinematics | Map from joint angles to pose | Confused with IK | Function |
| Inverse kinematics | Map from pose to joint angles | Assumed unique | Function |
| Jacobian | Map from joint velocities to end-effector velocities | A constant matrix | Matrix |
| Singularity | Configuration at which the Jacobian loses rank | Any low manipulability | Configuration |
| Manipulability | \(\sqrt{\det(JJ^T)}\) | A safety limit | Dimensionless |
| Condition number | Ratio of the largest to the smallest singular value | Confused with manipulability | Dimensionless |
| Damped least squares | IK method with a damping term | A pseudo-inverse | Method |
| Redundancy | More joints than task dimensions | A spare axis | Count |
| Workspace | Set of reachable poses | The safe zone | Volume |
| Joint limit | Mechanical or software limit on a joint | A workspace limit | rad |
| TCP | Tool centre point | The flange | Frame |
| Payload | Mass and inertia carried by the robot | The rated load only | kg, kg·m² |
| Backlash | Lost motion when the direction reverses | Compliance | mm or rad |
| Compliance | Deflection under load | Backlash | mm/N |
| Accuracy | Closeness to the commanded pose | Repeatability | mm |
| Repeatability | Closeness of repeated poses | Accuracy | mm |
| Registration | Estimation of the transform between frames | A calibration | Transform |
| Fiducial | Marker with a known position | Any landmark | Point |
| FLE | Fiducial localisation error | FRE | mm |
| FRE | Fiducial registration error | TRE | mm |
| TRE | Target registration error | FRE | mm |
| Point cloud | Set of 3D points | A mesh | Points |
| ICP | Iterative closest point registration | A closed-form method | Algorithm |
| Hand-eye calibration | Transform between a camera and a robot flange | Registration | Transform |
| Safety envelope | Set of limits within which motion is permitted | The workspace | Limits |
| SSM | Speed and separation monitoring | A guard | Function |
| PFL | Power and force limiting | SSM | Function |
| SRMS | Safety-rated monitored stop | A software stop | Function |
| PL | Performance level in ISO 13849 | SIL | a–e |
| SIL | Safety integrity level in IEC 61508 | PL | 1–4 |
| Guard | Physical barrier | An interlock | None |
| Interlock | Device that prevents operation when a guard is open | A guard | None |
| Emergency stop | Manual function that removes hazardous energy | A software stop | None |
| Collaborative application | Application designed for shared human-robot space | A collaborative robot | Application |
| Quasi-static contact | Slow contact with force and pressure limits | Transient contact | N, N/cm² |
| Transient contact | Dynamic contact with energy limits | Quasi-static contact | J |

## 5. The mechanics, step by step

The procedure below runs the robotics lab. It has a simulation phase and a hardware phase. The hardware phase is optional and requires the guarding described in the safety statement.

### Step 1. Define the task and the claim

**Input:** the task, the target, and the accuracy requirement.

**Output:** a claim with a target, a tolerance, a speed, and a claim boundary.

**Rationale:** a robotics claim without a tolerance is a demonstration. The tolerance determines the model fidelity, the calibration effort, and the safety margin.

The claim states:

- The target pose and the coordinate frame.
- The tolerance in position and orientation.
- The speed and the payload.
- The environment and the operating mode.
- The claim boundary: what the result does not prove.

A claim such as "the robot reaches the target" is not testable. "The robot reaches the target within 2 mm in position and 1° in orientation, at 50 mm/s, with a 0.5 kg payload" is testable.

### Step 2. Choose the model and the conventions

**Input:** the robot geometry and the software stack.

**Output:** a kinematic model with frames, axes, units, and joint limits.

**Rationale:** the model is the reference for every measurement. A model with the wrong convention produces a consistent but wrong result.

The conventions are:

- SI units: metres, radians, seconds, kilograms.
- Right-handed frames, x forward, y left, z up (REP 103 for ROS).
- Quaternions in x, y, z, w order.
- Joint angles in radians, positive according to the right-hand rule about the joint axis.
- The base frame at the mounting surface.
- The tool frame at the TCP.

The conventions are recorded in the model file and in the report. A reader who does not know the conventions cannot reproduce the result.

### Step 3. Write the forward kinematics and verify it by hand

**Input:** the link lengths and the joint axes.

**Output:** an FK function and three verified poses.

**Rationale:** the FK is the foundation. An error in the FK propagates to the Jacobian, the IK, and the control.

Procedure:

1. Write the transform for each joint-link pair.
2. Compose the transforms from the base to the tool.
3. Compute the pose for three joint configurations by hand.
4. Compare the function's output with the hand calculation.
5. Record the difference.

For a planar two-link arm with \(L_1 = 0.30\) m and \(L_2 = 0.25\) m:

\[
x = L_1 \cos q_1 + L_2 \cos(q_1 + q_2)
\]
\[
y = L_1 \sin q_1 + L_2 \sin(q_1 + q_2)
\]

At \(q_1 = 0\) and \(q_2 = 0\), the tool is at (0.55, 0). At \(q_1 = \pi/2\) and \(q_2 = 0\), the tool is at (0, 0.55). At \(q_1 = 0\) and \(q_2 = \pi/2\), the tool is at (0.30, 0.25).

### Step 4. Compute the Jacobian and verify it by finite difference

**Input:** the FK function.

**Output:** a Jacobian function and a finite-difference comparison.

**Rationale:** the Jacobian is used for velocity control, singularity analysis, and IK. A sign error in the Jacobian produces a controller that moves in the wrong direction.

Procedure:

1. Derive the Jacobian analytically.
2. Compute it numerically with a central difference and \(h = 10^{-6}\).
3. Compare the two matrices at ten random configurations.
4. Report the maximum absolute difference.
5. Investigate any difference larger than the truncation error.

For the planar arm:

\[
J = \begin{bmatrix}
-L_1 \sin q_1 - L_2 \sin(q_1+q_2) & -L_2 \sin(q_1+q_2) \\
L_1 \cos q_1 + L_2 \cos(q_1+q_2) & L_2 \cos(q_1+q_2)
\end{bmatrix}
\]

### Step 5. Map the singularities

**Input:** the Jacobian and the joint limits.

**Output:** a manipulability map and a list of singular configurations.

**Rationale:** a singularity is a configuration where the robot loses a degree of freedom. Near it, the joint velocities required for a Cartesian motion can exceed the joint limits.

Procedure:

1. Sample the joint space on a grid within the limits.
2. Compute the manipulability at each point.
3. Plot the map.
4. Identify the configurations where the manipulability is near zero.
5. For the planar arm, confirm the singularities at \(q_2 = 0\) and \(q_2 = \pi\).
6. State the joint velocity that a Cartesian velocity would require near each singularity.

### Step 6. Implement the inverse kinematics with limits

**Input:** the FK, the Jacobian, and the joint limits.

**Output:** an IK solver that returns a solution or an explicit failure.

**Rationale:** an IK solver that returns a NaN on an unreachable target is a solver that will command a NaN to the controller. The failure must be explicit.

Procedure:

1. Implement the analytical solution for the planar arm, with both elbow branches.
2. Test 1,000 reachable targets and record the position error.
3. Test unreachable targets and confirm that the solver reports failure.
4. Enforce the joint limits and report the fraction of targets that are reachable within the limits.
5. Implement damped least squares for the general case.
6. Compare the convergence and the joint velocities near a singularity.

### Step 7. Register the robot frame to the target frame

**Input:** a set of fiducials in both frames.

**Output:** a transform, the FRE, and a predicted TRE at the target.

**Rationale:** the registration is the link between the plan and the robot. An error in the registration moves the target in the robot's frame.

Procedure:

1. Place at least four fiducials in a non-planar configuration.
2. Measure each fiducial in the robot frame with the tool or a calibrated probe.
3. Measure the same fiducials in the target frame with the planning system or a caliper.
4. Compute the rigid transform by the closed-form least-squares method.
5. Compute the FRE as the RMS residual at the fiducials.
6. Predict the TRE at the target from the fiducial geometry and the FLE.
7. Measure the TRE at the target with an independent method.
8. Compare the prediction with the measurement.

The FRE is not the accuracy at the target. The prediction uses the fiducial spread and the target distance; the measurement is the evidence.

### Step 7A. Assess the registration geometry

**Input:** the fiducial coordinates and the target.

**Output:** a condition number, a predicted TRE, and a decision to accept or add fiducials.

**Rationale:** a registration can have a small FRE and a large TRE if the fiducials are clustered or if the target is far from them. The geometry is the evidence that the FRE is meaningful.

Procedure:

1. Compute the centroid of the fiducials.
2. Compute the spread along each principal axis.
3. Compute the condition number of the point configuration.
4. Predict the TRE at the target from the spread, the target distance, and the assumed FLE.
5. Compare the prediction with the requirement.
6. Add fiducials if the prediction exceeds the requirement.
7. Record the geometry in the report.

| Geometry | Condition number | Predicted TRE at 10 cm | Why |
|---|---:|---:|---|
| Four fiducials, 30 cm square | 2.1 | 0.5 mm | Good spread |
| Four fiducials, 5 cm square | 6.4 | 1.4 mm | Clustered |
| Three fiducials, collinear | Very large | 5 mm or more | Degenerate |
| Four fiducials, coplanar | 3.0 | 1.0 mm | Adequate but not ideal |

The condition number is a diagnostic, not a requirement. The requirement is the predicted TRE, and the geometry is how the prediction is made. A geometry that meets the requirement at one target may fail at another, because the TRE grows with distance from the centroid.

The prediction is only as good as the assumed FLE. A team that assumes 0.4 mm without measuring it has assumed the answer. The FLE should be estimated from repeated localisation of the same fiducial, or from the residuals of a leave-one-out registration.

A leave-one-out check is also a useful failure detector. If removing one fiducial changes the transform much more than the others, that fiducial is an outlier or the geometry is weak in that direction.

### Step 8. Measure the robot's positioning error

**Input:** the robot, a target, and a measurement instrument.

**Output:** the position and orientation error at the target, with uncertainty.

**Rationale:** the model predicts the pose. The measurement shows the error. The difference is the model's accuracy.

Procedure:

1. Command the robot to the target.
2. Measure the tool position with a caliper, a dial indicator, or a tracking system.
3. Repeat the measurement five times, approaching from different directions.
4. Compute the mean and the standard deviation.
5. Separate the systematic error (the mean) from the random error (the standard deviation).
6. Compare with the registration prediction and the requirement.

A single measurement is a sample, not an accuracy. The repeatability is the spread; the accuracy is the distance from the commanded pose.

### Step 9. Compute the safety envelope

**Input:** the accuracy, the registration error, the speed, the latency, and the deceleration.

**Output:** a separation distance and a safe-zone boundary.

**Rationale:** the envelope is the sum of the errors and the stopping distance. A boundary that ignores one term is a boundary that can be crossed.

A simplified budget:

\[
S = v_h t_r + v_r t_r + \frac{v_r^2}{2a} + C + Z
\]

where \(C\) includes the robot accuracy, the registration error, and the sensing uncertainty, and \(Z\) is a margin.

The terms are measured or justified. The human speed is assumed; the robot speed, latency, and deceleration are measured; the accuracy and registration are measured; the margin is a design choice.

### Step 10. Test the envelope in simulation and, if available, on hardware

**Input:** the envelope and a set of trajectories.

**Output:** a pass or fail for each trajectory, with the minimum separation.

**Rationale:** the envelope is a hypothesis until it is tested. The test commands trajectories toward the boundary and measures the minimum separation.

Procedure:

1. Command a trajectory that approaches the boundary at the maximum speed.
2. Measure the minimum separation.
3. Command a trajectory that crosses the boundary and confirm that the controller rejects it before motion.
4. Inject a fault (a delayed command, a lost encoder) and confirm the safe stop.
5. Record the minimum separation and the response time.

On hardware, the test is performed with a dummy load and a guarded workspace. The e-stop is within reach. No hands are in the workspace.

### Step 11. Verify the safety functions

**Input:** the e-stop, the guard, the interlock, and the speed limit.

**Output:** a test record for each function.

**Rationale:** a safety function that has not been tested is a hypothesis. The test record is the evidence.

Checks:

- The e-stop removes actuator power, verified by measuring the supply.
- The guard cannot be opened without a tool, or opening it triggers the interlock.
- The interlock stops motion before a hand can reach the hazard.
- The speed limit is enforced by the controller, and the measured speed does not exceed it.
- The safe stop is latched and requires a deliberate restart.

### Step 12. Record and report

**Input:** the model, the measurements, and the tests.

**Output:** a report with the claim, the model, the results, the safety envelope, and the limitations.

**Rationale:** the report is the deliverable. It must be readable by a person who was not in the room.

Teardown: command the arm to a park pose, disable the actuators, verify that the e-stop is latched, remove the payload, and disconnect power. The guard stays in place until the arm is stationary and the energy is removed.

The report contains:

- The task and the claim boundary.
- The model, the conventions, and the joint limits.
- The FK verification and the Jacobian check.
- The singularity map.
- The IK results and the failure cases.
- The registration and the FRE/TRE comparison.
- The positioning error and the repeatability.
- The safety envelope and the separation budget.
- The safety-function tests.
- The limitations.

### 5.15 Acceptance criteria for the robotics lab

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Claim has a target and a tolerance | Charter | Rewrite |
| 2 | Conventions are recorded | Model file | Record |
| 3 | FK is verified by hand | Three poses | Correct |
| 4 | Jacobian matches the finite difference | Matrix comparison | Correct |
| 5 | Singularities are mapped | Manipulability plot | Compute |
| 6 | IK reports failure for unreachable targets | Test log | Fix |
| 7 | Joint limits are enforced | Test log | Fix |
| 8 | Registration has at least four fiducials | Fiducial table | Add |
| 9 | FRE is computed | Table | Compute |
| 10 | TRE is predicted and measured | Comparison | Measure |
| 11 | Positioning error is measured | Table | Measure |
| 12 | Safety envelope includes accuracy and stopping distance | Budget | Complete |
| 13 | Safety functions are tested | Test record | Test |
| 14 | Limitations are stated | Report | Add |

### 5.16 Branch points and stop points

**Branch A: the robot is simulated only.** The result is a simulation result. It can verify the kinematics and the control law. It cannot verify the real accuracy, the compliance, or the safety functions.

**Branch B: the robot is physical but unguarded.** Stop. No motion until the guard, the e-stop, the joint limits, and the speed cap are in place.

**Branch C: the robot is patient-connected.** Stop. A surgical or rehabilitation robot requires the medical particular standards and a clinical safety case.

**Branch D: the target is inside a no-go zone.** Stop. The envelope must reject the target before motion, and the rejection must be tested.

## 6. Worked example

### 6.1 The question

A team has a two-link planar arm with \(L_1 = 0.30\) m and \(L_2 = 0.25\) m, joint limits of ±150°, and a tool offset of 0.02 m along the second link. The arm is used to place a marker at a target on a bench. The team wants to know whether the arm can place the marker within 2 mm, and what separation distance the safety envelope needs at a maximum speed of 50 mm/s.

### 6.2 The model

| Parameter | Value | Note |
|---|---|---|
| L1 | 0.300 m | Measured with a caliper, ±0.5 mm |
| L2 | 0.250 m | Measured with a caliper, ±0.5 mm |
| Tool offset | 0.020 m | Along the second link |
| Joint 1 limit | ±150° | Mechanical stop |
| Joint 2 limit | ±150° | Mechanical stop |
| Encoder resolution | 0.01° | 0.17 mrad |
| Repeatability | 0.08 mm | Measured, five approaches |
| Backlash | 0.15 mm | Measured at the tool |
| Payload | 0.2 kg | Marker holder |

The link lengths are measured, not assumed. A 1 mm error in a link length produces a position error of about 1 mm at the tool, which is half the tolerance.

### 6.3 The forward kinematics check

The FK is:

\[
x = L_1 \cos q_1 + (L_2 + d) \cos(q_1 + q_2)
\]
\[
y = L_1 \sin q_1 + (L_2 + d) \sin(q_1 + q_2)
\]

with \(d = 0.020\) m, so \(L_2 + d = 0.270\) m.

| Configuration | Hand calculation | FK function | Difference |
|---|---|---|---|
| \(q_1 = 0, q_2 = 0\) | (0.570, 0.000) | (0.570, 0.000) | 0.000 mm |
| \(q_1 = 90°, q_2 = 0\) | (0.000, 0.570) | (0.000, 0.570) | 0.000 mm |
| \(q_1 = 0, q_2 = 90°\) | (0.300, 0.270) | (0.300, 0.270) | 0.000 mm |
| \(q_1 = 30°, q_2 = -60°\) | (0.494, 0.015) | (0.494, 0.015) | 0.000 mm |

The hand calculation and the function agree to the printed precision. The check is trivial for a planar arm and essential for a six-axis arm, where a sign error is easy to miss.

### 6.4 The Jacobian and the singularity

The Jacobian is:

\[
J = \begin{bmatrix}
-L_1 \sin q_1 - (L_2+d) \sin(q_1+q_2) & -(L_2+d) \sin(q_1+q_2) \\
L_1 \cos q_1 + (L_2+d) \cos(q_1+q_2) & (L_2+d) \cos(q_1+q_2)
\end{bmatrix}
\]

The determinant is:

\[
\det J = L_1 (L_2+d) \sin q_2
\]

The arm is singular at \(q_2 = 0\) and \(q_2 = \pi\). At \(q_2 = 0\), the arm is fully extended and cannot move radially. At \(q_2 = \pi\), it is folded back.

The manipulability is:

\[
w = \sqrt{\det(J J^T)} = |\det J| = L_1 (L_2+d) |\sin q_2|
\]

At \(q_2 = 10°\), \(w = 0.30 \times 0.27 \times 0.1736 = 0.0141\ \text{m}^2\). At \(q_2 = 0.1°\), \(w = 2.45 \times 10^{-4}\ \text{m}^2\), which is 58 times smaller.

A Cartesian velocity of 50 mm/s at \(q_2 = 0.1°\) requires a joint velocity of:

\[
\dot{q}_2 = \frac{v}{L_1 \sin q_2} = \frac{0.050}{0.30 \times 0.001745} = 95.5\ \text{rad/s}
\]

which is far above any servo's capability. The controller must slow down or stop before this configuration.

### 6.5 The registration

The team places four fiducials on the bench and measures them in the robot frame and in the plan frame.

| Fiducial | Robot frame (m) | Plan frame (m) | Residual after fit (mm) |
|---|---|---|---|
| 1 | (0.100, 0.050) | (0.1017, 0.0494) | 0.26 |
| 2 | (0.400, 0.050) | (0.4013, 0.0520) | 0.25 |
| 3 | (0.400, 0.300) | (0.3999, 0.3020) | 0.35 |
| 4 | (0.100, 0.300) | (0.0993, 0.2997) | 0.27 |

The rigid transform is a rotation of −0.45° and a translation of (−1.92 mm, +1.24 mm). The FRE is:

\[
\text{FRE} = \sqrt{\frac{0.26^2 + 0.25^2 + 0.35^2 + 0.27^2}{4}} = 0.28\ \text{mm}
\]

The FRE is small. The target is at (0.25, 0.175), near the centroid of the fiducials, so the predicted TRE is also small. The prediction from the fiducial geometry and an assumed FLE of 0.4 mm is about 0.5 mm.

### 6.6 The measured error at the target

The team commands the arm to the target and measures the marker position with a caliper and a dial indicator.

| Approach | Position error (mm) | Direction |
|---|---:|---|
| From the left | 1.1 | +x |
| From the right | 1.6 | +x |
| From above | 1.2 | +y |
| From below | 1.7 | +y |
| Repeat from the left | 1.2 | +x |

The mean error is 1.36 mm, and the standard deviation is 0.26 mm. The error is larger than the predicted TRE of 0.5 mm, which means the registration is not the dominant term. The dominant terms are the link geometry, the backlash, and the compliance.

| Source | Estimate | Type | Evidence |
|---|---:|---|---|
| Registration TRE | 0.5 mm | Systematic | Predicted from fiducials |
| Link length | 0.5 mm | Systematic | Caliper uncertainty |
| Backlash | 0.15 mm | Direction-dependent | Measured at the tool |
| Compliance | 0.4 mm | Load-dependent | Estimated, not measured |
| Repeatability | 0.26 mm | Random | Five approaches |
| **Total, quadrature** | **0.79 mm** | | |
| **Measured mean** | **1.36 mm** | | |

The quadrature sum is smaller than the measured mean, which means the budget is missing a systematic term. The most likely missing term is the zero offset of the joint encoders, which was not calibrated. A systematic offset of 1 mm at the tool is consistent with a joint-zero error of about 0.1°.

The lesson is that the error stack must be validated against the measurement, not only constructed. A budget that sums to less than the measurement is a budget with a missing term.

### 6.7 Does the arm meet the 2 mm tolerance?

The measured mean error is 1.36 mm, and the expanded uncertainty of the measurement is 0.2 mm. The 95% upper bound is about 1.76 mm, which is below 2 mm.

| Quantity | Value | Unit | Comparison |
|---|---:|---|---|
| Mean position error | 1.36 | mm | — |
| Standard deviation | 0.26 | mm | — |
| Expanded uncertainty (k = 2) | 0.20 | mm | — |
| Upper bound | 1.76 | mm | Within 2 mm |
| Margin | 0.24 | mm | Small |

The margin is small. A second unit, a different payload, or a different temperature could fail. The team should calibrate the joint zeros and repeat the measurement before claiming the tolerance.

### 6.8 The safety envelope

The simplified separation budget is:

\[
S = v_h t_r + v_r t_r + \frac{v_r^2}{2a} + C + Z
\]

The terms are:

| Term | Value | Source |
|---|---:|---|
| Human speed \(v_h\) | 1.6 m/s | Assumed walking speed |
| Reaction time \(t_r\) | 0.10 s | Controller and brake |
| Robot speed \(v_r\) | 0.05 m/s | Speed cap |
| Deceleration \(a\) | 1.0 m/s² | Measured |
| Position uncertainty \(C\) | 1.76 mm | Robot accuracy |
| Registration error | 0.50 mm | Predicted TRE |
| Sensing uncertainty | 0.50 mm | Assumed |
| Margin \(Z\) | 5.0 mm | Design choice |

The stopping distance is:

\[
\frac{v_r^2}{2a} = \frac{0.05^2}{2 \times 1.0} = 1.25\ \text{mm}
\]

The reaction distance is:

\[
v_r t_r = 0.05 \times 0.10 = 5.0\ \text{mm}
\]

The human contribution is:

\[
v_h t_r = 1.6 \times 0.10 = 160\ \text{mm}
\]

The total is:

\[
S = 160 + 5.0 + 1.25 + 1.76 + 0.50 + 0.50 + 5.0 = 174.0\ \text{mm}
\]

The human term dominates. A separation distance of 174 mm is required if the human can approach at 1.6 m/s. If the human is behind a guard and cannot approach, the term is zero and the distance falls to 14 mm.

The example shows why the human term is usually the largest. It also shows why a robot speed reduction from 50 to 25 mm/s does not help much: the human term dominates.

### 6.9 Two checks that would invalidate the result

**Check 1: the FRE was treated as the accuracy at the target.** The FRE is the residual at the fiducials. The TRE at the target depends on the fiducial geometry and the target's position. In this example, the target is near the centroid, so the TRE is small. If the target were far outside the fiducial constellation, the TRE could be several times the FRE. The check is to compute the TRE prediction from the fiducial geometry, not to report the FRE as the accuracy.

**Check 2: the joint zeros were not calibrated.** The FK assumes that the encoder zero corresponds to the model's zero. If the encoder zero is off by 0.1°, the tool position is off by about 1 mm. The check is to move the arm to a known configuration, measure the tool position, and compare it with the FK. A consistent offset indicates a zero error.

### 6.10 A varied condition: the speed is doubled

The team doubles the speed cap from 50 to 100 mm/s and recomputes the envelope with the human term set to zero, assuming the human is outside the guarded workspace.

| Condition | Robot speed | Reaction distance | Stopping distance | C + Z | Total | Envelope margin |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 50 mm/s | 5.0 mm | 1.25 mm | 7.76 mm | 14.0 mm | 10 mm pass |
| Varied | 100 mm/s | 10.0 mm | 5.0 mm | 7.76 mm | 22.8 mm | 10 mm fail |
| Varied | 200 mm/s | 20.0 mm | 20.0 mm | 7.76 mm | 47.8 mm | 10 mm fail |

The total separation rises from 14.0 to 22.8 mm when the speed doubles. If the available separation is 10 mm, the baseline passes and the doubled speed fails. The result moves in the direction the model predicts, and the quadratic stopping term becomes dominant at higher speeds.

The varied condition also shows the value of the speed cap. A low cap keeps the stopping distance small and the envelope feasible.

### 6.11 What the example demonstrates

The arithmetic closes: a hand-verified FK, a Jacobian determinant of \(L_1(L_2+d)\sin q_2\), a manipulability of 0.0141 m² at 10° and 2.45×10⁻⁴ m² at 0.1°, a joint velocity of 95.5 rad/s at 0.1° and 50 mm/s, an FRE of 0.35 mm, a measured mean error of 1.36 mm, and a separation distance of 174 mm with the human term or 14 mm without it.

The example demonstrates the difference between the model and the robot. The model is exact to the printed precision; the robot has a 1.36 mm error. The difference is the error stack, and the stack must be measured, not assumed.

The example also demonstrates the difference between the FRE and the TRE. The FRE is 0.35 mm; the measured error is 1.36 mm. The registration is a small part of the error, and reporting the FRE as the accuracy would understate the error by a factor of about five.

**The registration's effect at the target.** The FRE of 0.28 mm is the residual at the fiducials. The target at (0.250, 0.175) maps to (0.2495, 0.1743) in the robot frame, a shift of 0.92 mm. The shift is larger than the FRE but smaller than the measured error of 1.36 mm. A report that gives only the FRE hides the shift at the target; a report that gives only the measured error hides the registration's contribution. The three numbers together are the evidence: FRE for fit quality, TRE prediction for geometry, and measured error for the system.

## 7. Code, query, or calculation

The program below implements the planar arm FK, Jacobian, manipulability, damped least squares, a 2D rigid registration by complex-number Procrustes, and the separation budget. It uses only the Python standard library.

```python
import cmath
import math
import random

L1, L2, D = 0.300, 0.250, 0.020
L2E = L2 + D

def fk(q):
    q1, q2 = q
    return (L1 * math.cos(q1) + L2E * math.cos(q1 + q2),
            L1 * math.sin(q1) + L2E * math.sin(q1 + q2))

def jac(q):
    q1, q2 = q
    return [
        [-L1 * math.sin(q1) - L2E * math.sin(q1 + q2), -L2E * math.sin(q1 + q2)],
        [ L1 * math.cos(q1) + L2E * math.cos(q1 + q2),  L2E * math.cos(q1 + q2)],
    ]

def det(j):
    return j[0][0] * j[1][1] - j[0][1] * j[1][0]

def manipulability(q):
    return abs(det(jac(q)))

def ik(q0, target, lam=0.05, iters=500, tol=1e-9):
    q = list(q0)
    for _ in range(iters):
        p = fk(q)
        e = (target[0] - p[0], target[1] - p[1])
        if math.hypot(*e) < tol:
            return q, math.hypot(*e), True
        j = jac(q)
        a, b = j[0]
        c, d = j[1]
        # J J^T + lam^2 I, solved for the 2x2 case.
        m00 = a*a + b*b + lam*lam
        m11 = c*c + d*d + lam*lam
        m01 = a*c + b*d
        det_m = m00*m11 - m01*m01
        y0 = (m11*e[0] - m01*e[1]) / det_m
        y1 = (-m01*e[0] + m00*e[1]) / det_m
        dq0 = a*y0 + c*y1
        dq1 = b*y0 + d*y1
        q[0] += dq0
        q[1] += dq1
    return q, math.hypot(target[0]-fk(q)[0], target[1]-fk(q)[1]), False

print("FK check")
for q in ((0, 0), (math.pi/2, 0), (0, math.pi/2), (math.pi/6, -math.pi/3)):
    p = fk(q)
    print(f"q=({q[0]:.3f},{q[1]:.3f}) -> ({p[0]:.4f}, {p[1]:.4f})")

print("\nmanipulability near the singularity q2=0")
for deg in (30, 10, 1, 0.1, 0.01):
    q = (0.0, math.radians(deg))
    w = manipulability(q)
    v = 0.050
    qd = v / (L1 * math.sin(q[1])) if q[1] > 0 else float("inf")
    print(f"q2={deg:6.2f} deg  w={w:.6f} m^2  qdot2={qd:9.2f} rad/s")

print("\nIK from a nearby seed")
target = (0.400, 0.100)
for seed in ((0.5, 0.5), (1.0, -0.8), (-1.0, 0.8)):
    q, err, ok = ik(seed, target)
    print(f"seed={seed} -> q=({q[0]:.4f},{q[1]:.4f}) err={err:.2e} ok={ok}")

print("\nIK on an unreachable target")
q, err, ok = ik((0.5, 0.5), (1.0, 0.0))
print(f"err={err:.3f} m ok={ok} (reach limit {L1+L2E:.3f} m)")

# --- 2D rigid registration with complex numbers -------------------------
def register(src, dst):
    """Return (scale=1, rotation, translation) mapping src to dst."""
    n = len(src)
    cs = sum(src) / n
    cd = sum(dst) / n
    num = sum((d - cd) * (s - cs).conjugate() for s, d in zip(src, dst))
    rot = cmath.phase(num)
    trans = cd - cmath.exp(1j * rot) * cs
    return rot, trans

def apply(z, rot, trans):
    return cmath.exp(1j*rot) * z + trans

robot = [complex(0.100, 0.050), complex(0.400, 0.050),
         complex(0.400, 0.300), complex(0.100, 0.300)]
plan = [complex(0.101681, 0.049352), complex(0.401273, 0.051951),
        complex(0.399890, 0.302004), complex(0.099318, 0.299675)]

rot, trans = register(plan, robot)
print("\nregistration (plan -> robot)")
print(f"rotation = {math.degrees(rot):.3f} deg")
print(f"translation = ({trans.real*1e3:.2f}, {trans.imag*1e3:.2f}) mm")

res = [abs(apply(p, rot, trans) - r) for p, r in zip(plan, robot)]
fre = math.sqrt(sum(r*r for r in res) / len(res))
print("residuals (mm):", [f"{r*1e3:.2f}" for r in res])
print(f"FRE = {fre*1e3:.2f} mm")

target_plan = complex(0.250, 0.175)
target_robot = apply(target_plan, rot, trans)
print(f"target in robot frame = ({target_robot.real:.4f}, {target_robot.imag:.4f})")

# --- separation budget ---------------------------------------------------
def separation(vh, vr, tr, a, c, z):
    return vh*tr + vr*tr + vr*vr/(2*a) + c + z

C_TERM = 0.00176 + 0.00050 + 0.00050
print("\nseparation budget (mm)")
for vr in (0.050, 0.100, 0.200):
    s_human = separation(1.6, vr, 0.10, 1.0, C_TERM, 0.005)
    s_guard = separation(0.0, vr, 0.10, 1.0, C_TERM, 0.005)
    print(f"vr={vr*1000:5.0f} mm/s  with human {s_human*1e3:7.1f}  "
          f"guarded {s_guard*1e3:6.1f}")
```

Expected output:

```text
FK check
q=(0.000,0.000) -> (0.5700, 0.0000)
q=(1.571,0.000) -> (0.0000, 0.5700)
q=(0.000,1.571) -> (0.3000, 0.2700)
q=(0.524,-1.047) -> (0.4936, 0.0150)

manipulability near the singularity q2=0
q2= 30.00 deg  w=0.040500 m^2  qdot2=     0.33 rad/s
q2= 10.00 deg  w=0.014066 m^2  qdot2=     0.96 rad/s
q2=  1.00 deg  w=0.001414 m^2  qdot2=     9.55 rad/s
q2=  0.10 deg  w=0.000141 m^2  qdot2=    95.49 rad/s
q2=  0.01 deg  w=0.000014 m^2  qdot2=   954.93 rad/s

IK from a nearby seed
seed=(0.5, 0.5) -> q=(-0.4682,1.5270) err=4.61e-10 ok=True
seed=(1.0, -0.8) -> q=(0.9581,-1.5270) err=1.33e-10 ok=True
seed=(-1.0, 0.8) -> q=(-0.4682,1.5270) err=7.26e-10 ok=True

IK on an unreachable target
err=0.636 m ok=False (reach limit 0.570 m)

registration (plan -> robot)
rotation = -0.453 deg
translation = (-1.92, 1.24) mm
residuals (mm): ['0.26', '0.25', '0.35', '0.27']
FRE = 0.28 mm
target in robot frame = (0.2495, 0.1743)

separation budget (mm)
vr=   50 mm/s  with human   174.0  guarded   14.0
vr=  100 mm/s  with human   182.8  guarded   22.8
vr=  200 mm/s  with human   207.8  guarded   47.8
```

The FK check matches the hand calculations. The fourth configuration gives (0.4936, 0.0150), which rounds to (0.494, 0.015).

The manipulability falls as \(q_2\) approaches zero, and the required joint velocity rises as \(1/\sin q_2\). At 0.1°, the required velocity is 95.5 rad/s; at 0.01°, it is 955 rad/s. Both are physically impossible for a servo. The controller must stop or slow down before this configuration.

The IK converges to two different branches depending on the seed: a solution with \(q_2 = +1.527\) rad and one with \(q_2 = -1.527\) rad. Both reach the target; they are the elbow-up and elbow-down solutions. The solver is correct, and the branch is not unique. A controller must choose a branch deliberately, not accept whichever one the seed produces.

The unreachable target returns `ok=False` and a residual of 0.636 m. The geometric distance from the target to the reachable boundary is 0.430 m, so the residual is larger than the geometric distance because the damped solver stops at a finite iteration and does not reach the boundary. The important point is that the solver reports failure and does not return a NaN.

The registration recovers a −0.453° rotation and a (−1.92, 1.24) mm translation, with an FRE of 0.28 mm. The target in the robot frame is (0.2495, 0.1743), which is 0.92 mm from the plan target of (0.25, 0.175). That difference is the registration's effect at the target, and it is larger than the FRE.

The separation budget shows the human term dominating: 174.0 mm with a walking human, 14.0 mm with a guard. The guarded distance rises to 22.8 mm at 100 mm/s and 47.8 mm at 200 mm/s, which is why the speed cap matters.

## 8. Data, standards, and regulatory context

### 8.1 Industrial robot safety

ISO 10218-1:2025 is the third edition of the industrial robot safety standard. It was published in February 2025 and supersedes ISO 10218-1:2011. It introduces a Class I / Class II robot classification, expands the safety-function list, adds cybersecurity requirements, and incorporates the collaborative-application requirements that were formerly in ISO/TS 15066.

ISO 10218-2:2025 covers the robot system and the application. It includes Annex L for speed and separation monitoring and Annex M for quasi-static and transient contact limits.

The standard is a type-C machinery standard in the ISO hierarchy. It presumes conformity with the essential health and safety requirements of the Machinery Directive when the standard is applied. It does not cover medical use.

### 8.2 Collaborative robot limits

ISO/TS 15066:2016 provided guidance on collaborative robot force and pressure limits before its content was folded into ISO 10218-1 and -2:2025. Annex A of the technical specification gives body-region-specific quasi-static and transient limits, derived from a pain-onset study with 100 subjects at the 75th percentile.

The values are not universal. They depend on the body region, the contact area, and the contact dynamics. A single force limit for all contacts is a misreading of the standard. The transient limits are generally about twice the quasi-static limits, with exceptions.

### 8.3 Medical robot standards

IEC 80601-2-77:2019 covers robotically assisted surgical equipment. Amendment 1 was published in 2023. The standard addresses the basic safety and essential performance of the equipment, including the accuracy of controls and instruments, the mechanical hazards, the essential performance, and the interaction with the patient.

IEC 80601-2-78:2019 covers medical robots for rehabilitation, assessment, compensation, or alleviation. Amendment 1 was published in 2024. It excludes external limb prosthetics, electric wheelchairs, diagnostic imaging equipment, and personal care robots.

Both standards are under revision. A test plan must cite the current edition and the applicable amendment.

### 8.4 Functional safety

ISO 13849-1 defines performance levels for safety-related parts of control systems. IEC 61508 defines safety integrity levels. IEC 62061 is the machinery-sector application of IEC 61508.

A safety function such as an emergency stop or a safety-rated monitored stop must be designed and verified to a performance level or a safety integrity level. A software function that is not developed to the standard is not a safety-rated function, even if it works in the lab.

### 8.5 Registration accuracy

The registration error terminology comes from the image-guided surgery literature. Fitzpatrick, Hill, and Maurer defined the fiducial localisation error, the fiducial registration error, and the target registration error. The key result is that the FRE and the TRE are not the same, and a low FRE does not guarantee a low TRE.

The prediction of the TRE from the fiducial geometry is:

\[
\text{rms TRE}(p) = \sqrt{\frac{1}{N} + \frac{1}{3N}\sum_k \frac{d_k^2}{f_k^2}} \times \text{rms FLE}
\]

where \(d_k\) is the distance from the target to the principal axis \(k\), and \(f_k\) is the fiducial spread along that axis. The TRE is smallest at the centroid and grows away from it.

A registration report should give the FRE, the fiducial geometry, the predicted TRE at the target, and an independent measurement of the TRE. The FRE alone is not enough.

### 8.6 Coordinate conventions

REP 103 defines the units and the axes for ROS: metres, radians, right-handed frames, x forward, y left, z up, and quaternions in x, y, z, w order. REP 105 defines the frame tree for mobile platforms.

A robot model that uses a different convention can be correct in its own frame and wrong in the ROS frame. The conversion must be documented and tested.

### 8.7 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| ISO 10218-1:2025 | Type-C machinery standard | Industrial robots | A medical robot |
| ISO 10218-2:2025 | Type-C machinery standard | Robot systems and applications | A lab demo |
| ISO/TS 15066:2016 | Technical specification, incorporated into 2025 standards | Collaborative applications | A medical robot |
| IEC 80601-2-77 | Particular medical standard | Robotically assisted surgical equipment | An industrial robot |
| IEC 80601-2-78 | Particular medical standard | Rehabilitation robots | An industrial robot |
| ISO 13849-1 | Type-B machinery standard | Safety-related control systems | A research prototype |
| REP 103 | ROS convention | ROS coordinate frames | Non-ROS software |

## 9. Industry practice and case evidence

### 9.1 How surgical robots handle registration

A surgical robot registration typically uses fiducials or an image-to-robot registration. The system reports the FRE to the surgeon as a quality indicator, and it uses the TRE prediction to set the accuracy claim.

The clinical literature has warned that the FRE is a poor proxy for the TRE. Fitzpatrick's work showed that the two are statistically uncorrelated for a given patient. A low FRE can coexist with a large TRE, especially when the target is far from the fiducials.

The industry response is to measure the TRE at a known target, to use a registration quality metric that is correlated with the TRE, and to design the workflow so that the target is near the fiducials.

### 9.2 How collaborative robots implement speed and separation monitoring

A collaborative application with SSM uses a safety-rated sensor to track the human, a safety-rated robot position, and a protective separation distance calculated from the speeds, the reaction times, the stopping distance, and the uncertainties.

The implementation has three layers: the safety-rated sensor and controller, the robot's safety-rated stopping function, and the application's zone logic. A non-safety-rated camera can be used for the green zone, but the yellow and red zones need a safety-rated function.

The lab can demonstrate the geometry and the budget. It cannot certify the function.

### 9.3 Public recall narratives

The survey file lists several 2025 recalls that map to the robotics labs. They are public narratives, not lab results, and the identifiers must be verified before use.

| Event | Reported mechanism | Relevant lab habit | What the lab does not prove |
|---|---|---|---|
| Quantum Surgical EPIONE, 2025 (verify) | Software bug could misalign the arm relative to the planned trajectory | Registration and TRE measurement | Registration error on a patient |
| Virtual Incision MIRA, 2025 (verify) | Software could interrupt or lag motion between controllers and arms | Latency and command-age testing | Clinical consequence |
| Accuray CyberKnife Z-1023-2025 | Snap-ring detachment could allow uncontrolled roll-axis rotation | Mechanical fault injection and guard design | Structural failure under clinical loads |

The pattern is the same as in the other topics: a configuration or mechanical detail that was never exercised by an integrated test.

### 9.4 Calibration in industrial robots

An industrial robot is calibrated by measuring the tool position at many configurations and fitting the kinematic parameters. The calibration reduces the systematic error from link lengths, joint offsets, and tool offsets. It does not reduce the random error from backlash, compliance, and encoder noise.

The calibration is performed at a defined payload and temperature. A change in either invalidates the calibration. The robot's accuracy specification is usually stated for a defined payload and a defined path.

### 9.5 Inference

It is an inference, not a documented finding, that teams which measure the TRE at the target rather than reporting the FRE make more realistic accuracy claims. The reasoning is that the FRE is the quantity the algorithm minimises, not the quantity the task needs. The inference is consistent with the image-guided surgery literature, but it has not been tested in this course.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Units mixed | Robot moves 1000× too far | Millimetres versus metres | Collision | Unit check |
| 2 | Joint axis sign wrong | Motion in the wrong direction | Model error | Collision | Hand-verify FK |
| 3 | FK not verified | "It looks right" | Undetected model error | Systematic error | Three hand poses |
| 4 | Jacobian sign wrong | Controller diverges | Model error | Unstable motion | Finite-difference check |
| 5 | Singularity ignored | Large joint velocity near \(q_2=0\) | Rank loss | Saturation, stop | Manipulability map, DLS |
| 6 | IK returns NaN | Unreachable target | No failure check | Invalid command | Explicit failure |
| 7 | Joint limits not enforced | Command beyond the stop | Software limit missing | Mechanical damage | Limit check before command |
| 8 | FRE reported as accuracy | "Registration error 0.3 mm" | FRE is not TRE | Overstated accuracy | TRE prediction and measurement |
| 9 | Fiducials coplanar or clustered | Poor geometry | Ill-conditioned transform | Large TRE | Non-planar spread |
| 10 | Encoder zero not calibrated | Consistent offset | Model zero mismatch | Systematic error | Zero calibration |
| 11 | Payload changed | Deflection changes | Compliance | Position error | Recalibrate for the payload |
| 12 | Safety envelope ignores stopping distance | Boundary at the workspace edge | No deceleration margin | Collision | Separation budget |
| 13 | Software clamp called safety-rated | No PL or SIL | Not a safety function | Regulatory and safety gap | Hardware guard, rated function |
| 14 | E-stop not verified | Only the software stop tested | No power removal | Uncontrolled motion | Measure actuator supply |
| 15 | Guard opens without interlock | Access to motion | Mechanical gap | Crush | Interlock and test |
| 16 | Simulation result presented as hardware | No compliance or backlash in the model | Model omission | Overstated accuracy | Label the evidence |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The link lengths are known | The FK depends on them | Manufacturing tolerance, thermal expansion | Measure with a caliper |
| 2 | The joint zeros are calibrated | The FK depends on them | Encoder offset | Move to a known pose |
| 3 | The joints are rigid | The Jacobian assumes rigidity | Compliance, backlash | Measure under load |
| 4 | The payload is within the rating | The dynamics are valid | An heavier tool | Weigh the payload |
| 5 | The fiducials are accurately localised | The registration depends on them | FLE | Use a calibrated probe |
| 6 | The fiducial geometry is non-degenerate | The transform is well conditioned | Coplanar or clustered points | Check the spread |
| 7 | The target is near the fiducials | The TRE is small | A distant target | Predict and measure the TRE |
| 8 | The registration is rigid | The transform model is valid | A flexible or moving target | Test with a second target |
| 9 | The safety envelope terms are measured | The separation is valid | An assumed reaction time | Measure the stop |
| 10 | The human speed is bounded | The separation includes the human | A running human | Restrict access or increase the distance |
| 11 | The simulation matches the hardware | The result transfers | Compliance, backlash, friction | Measure on hardware |
| 12 | The safety function is rated | The claim is valid | A software-only function | Review the PL or SIL |

## 12. Exercises

### Exercise 1. Verify the FK by hand (calculation)

**Deliverable:** three hand calculations of the planar arm pose, compared with the FK function.

**Acceptance criteria:** the link lengths and tool offset are stated; the arithmetic is shown; the difference is reported in millimetres; a fourth configuration with a negative joint angle is included.

**Time:** 30 minutes.

### Exercise 2. Jacobian finite-difference check (calculation)

**Deliverable:** an analytical Jacobian and a central-difference comparison at ten configurations.

**Acceptance criteria:** the step size is stated; the maximum absolute difference is reported; the difference is compared with the truncation error; any outlier is investigated.

**Time:** 45 minutes.

### Exercise 3. Singularity map (application)

**Deliverable:** a manipulability map over the joint space, with the singular configurations marked and the joint velocity at a defined Cartesian speed.

**Acceptance criteria:** the map covers the joint limits; the singularities are identified analytically and numerically; the joint velocity is computed at three distances from the singularity; the controller implication is stated.

**Time:** 45 minutes.

### Exercise 4. IK with limits (application)

**Deliverable:** an IK solver tested on 1,000 reachable and 100 unreachable targets, with the error distribution and the failure cases.

**Acceptance criteria:** the solver reports failure explicitly; the position error is below the stated tolerance for reachable targets; the joint limits are enforced; the branch selection is documented.

**Time:** 60 minutes.

### Exercise 5. Registration (application)

**Deliverable:** a four-fiducial registration with the transform, the residuals, the FRE, and the predicted TRE at a target.

**Acceptance criteria:** the fiducials are non-coplanar; the transform is computed by a closed-form method; the FRE is the RMS residual; the TRE prediction states the geometry and the assumed FLE; the target is measured independently.

**Time:** 60 minutes.

### Exercise 6. Safety envelope (design)

**Deliverable:** a separation budget for a guarded and an unguarded workspace, with the terms measured or justified.

**Acceptance criteria:** the human speed, the reaction time, the robot speed, the deceleration, the accuracy, the registration error, and the margin are stated; the total is computed for three speeds; the dominant term is identified.

**Time:** 45 minutes.

### Exercise 7. Safety-function test (application)

**Deliverable:** a test record for the e-stop, the guard interlock, the speed limit, and the safe stop.

**Acceptance criteria:** the e-stop is verified by measuring actuator power; the guard interlock is tested; the speed limit is measured; the restart is deliberate; each test has a capture or a measurement.

**Time:** 60 minutes.

### Exercise 8. Critique a robotics claim (critique)

**Deliverable:** a review of a draft claim that says "the robot is accurate to 0.3 mm because the registration error is 0.3 mm," identifying the error and the correct claim.

**Acceptance criteria:** the review distinguishes FRE from TRE; it names the missing measurement at the target; it proposes a corrected sentence with the measured error and its uncertainty.

**Time:** 30 minutes.

## 13. Self-check answers

### 13.1 What is the difference between accuracy and repeatability?

Accuracy is the closeness of a measured pose to the commanded pose. Repeatability is the closeness of repeated poses to each other. A robot can be repeatable and inaccurate: it returns to the same wrong pose every time. A calibration can improve accuracy; it cannot improve repeatability beyond the mechanical limit.

### 13.2 Compute the FK for the planar arm at \(q_1 = 0, q_2 = 0\).

\[
x = 0.30 \cos 0 + 0.27 \cos 0 = 0.57\ \text{m}
\]
\[
y = 0.30 \sin 0 + 0.27 \sin 0 = 0\ \text{m}
\]

The tool is at (0.57, 0).

### 13.3 Why is the Jacobian singular at \(q_2 = 0\)?

Because the determinant is \(L_1(L_2+d)\sin q_2\), which is zero at \(q_2 = 0\). At that configuration, the arm is fully extended, and a small radial motion requires an infinite joint velocity. The robot cannot move radially at the singular configuration.

### 13.4 Compute the manipulability at \(q_2 = 10°\).

\[
w = 0.30 \times 0.27 \times \sin 10° = 0.30 \times 0.27 \times 0.17365 = 0.01407\ \text{m}^2
\]

### 13.5 What does damped least squares do?

It adds a damping term \(\lambda^2 I\) to \(J J^T\) before inversion. The damping limits the joint velocity near a singularity at the cost of a position error. A larger \(\lambda\) gives smoother motion and a larger error.

### 13.6 What is the difference between FRE and TRE?

The FRE is the RMS residual at the fiducials after registration. The TRE is the error at a target that is not a fiducial. The FRE is what the algorithm minimises; the TRE is what the task cares about. They are not the same, and a low FRE does not guarantee a low TRE.

### 13.7 Why does a distant target have a larger TRE?

Because the registration's rotational error produces a position error that grows with the distance from the fiducial centroid. The TRE prediction includes the target distance and the fiducial spread, and it grows away from the centroid.

### 13.8 Compute the stopping distance for a robot at 100 mm/s with a deceleration of 1 m/s².

\[
d = \frac{v^2}{2a} = \frac{0.1^2}{2 \times 1.0} = 0.005\ \text{m} = 5\ \text{mm}
\]

### 13.9 Why is a software workspace clamp not a safety-rated limit?

Because it depends on the software, the processor, and the sensor, and it has not been developed or verified to a safety standard. A safety-rated limit requires a rated architecture, diagnostic coverage, and a quantified failure rate. A hardware guard or a rated safety function is needed for the safety claim.

### 13.10 What is the first thing you check when the robot misses the target?

The model and the conventions: units, joint zeros, axis signs, and the tool offset. A systematic error is more likely than a random one, and it is cheaper to find. After the model, check the registration and the payload.

### 13.11 Rubric for Exercise 5

A strong registration report gives the fiducial coordinates in both frames, the transform, the residuals, the FRE, the predicted TRE with the geometry and the FLE, and an independent measurement at the target. A weak report gives only the FRE.

### 13.12 Rubric for Exercise 6

A strong separation budget states every term, measures the robot terms, justifies the human term, and computes the total for several speeds. It identifies the dominant term and the effect of a guard. A weak budget uses a single round number.

## 14. Interview questions

### 14.1 Why is a kinematic model not enough to predict a robot's accuracy?

Because the model omits manufacturing tolerance, backlash, compliance, thermal expansion, encoder calibration, and registration. A strong answer names several terms and says the model predicts the commanded pose, not the reached pose. A weak answer says the model is "approximate."

### 14.2 What is a singularity, and why does it matter?

A configuration where the Jacobian loses rank. Near it, a small Cartesian motion requires a large joint velocity, which can exceed the joint limits or saturate the controller. A strong answer gives the planar-arm example at \(q_2 = 0\). A weak answer says "the robot cannot move."

### 14.3 How do you handle a singularity in a controller?

A strong answer uses damped least squares, a speed limit near the singularity, or a path that avoids it. It notes the trade-off between accuracy and smoothness. A weak answer says "avoid it" without saying how.

### 14.4 What is the difference between FRE and TRE?

FRE is the residual at the fiducials; TRE is the error at the target. A strong answer explains that the algorithm minimises FRE and the clinician cares about TRE, and that they are not the same. A weak answer uses them interchangeably.

### 14.5 Why can a low FRE be misleading?

Because the fiducials can be well aligned while the target is far from them, and the rotational error grows with distance. A strong answer gives the TRE prediction formula or the intuition. A weak answer says "it is only an average."

### 14.6 How do you compute a protective separation distance?

A strong answer sums the human's approach, the robot's reaction, the stopping distance, the uncertainties, and a margin. It notes that ISO 10218-2 Annex L defines the calculation and that the human term often dominates. A weak answer says "keep a safe distance."

### 14.7 What is the difference between a safety-rated stop and a software stop?

A safety-rated stop is implemented with a rated architecture and verified to a PL or SIL. A software stop depends on the processor and the software and is not rated. A strong answer notes the hardware e-stop and the guard. A weak answer says they are the same.

### 14.8 Why is joint-limit enforcement not enough for a safety envelope?

Because the joint limits define the reachable workspace, not the safe workspace. The safe workspace must also exclude the no-go zones, account for the robot's accuracy and the stopping distance, and include the human's approach. A strong answer distinguishes the two.

### 14.9 How do you verify a robot model?

A strong answer checks the FK by hand, the Jacobian by finite difference, and the pose against a measurement. It also checks the units and the conventions. A weak answer says "it looks right in RViz."

### 14.10 What does a simulation result not prove?

It does not prove the real accuracy, the compliance, the backlash, the safety functions, or the behaviour under load. A strong answer states the boundary. A weak answer says simulation is "close enough."

### 14.11 How would you reduce the positioning error at the target?

A strong answer identifies the dominant term: calibration for a systematic error, a stiffer structure for compliance, a preloaded gearbox for backlash, more fiducials for registration. It measures before and after. A weak answer says "calibrate it."

### 14.12 What is the role of the payload in the accuracy claim?

The payload changes the deflection and the dynamics. A strong answer says the accuracy is stated for a defined payload and must be re-measured if the payload changes. A weak answer ignores the payload.

## 15. Key terms

- **Accuracy** — closeness to the commanded pose.
- **Backlash** — lost motion when the direction reverses.
- **Collaborative application** — application designed for shared human-robot space.
- **Compliance** — deflection under load.
- **Condition number** — ratio of the largest to the smallest singular value.
- **Damped least squares** — IK method with a damping term.
- **DH parameters** — four parameters describing a joint-link transform.
- **Emergency stop** — manual function that removes hazardous energy.
- **Fiducial** — marker with a known position.
- **FLE** — fiducial localisation error.
- **Forward kinematics** — map from joint angles to pose.
- **Frame** — coordinate system with an origin and axes.
- **FRE** — fiducial registration error.
- **Guard** — physical barrier.
- **Hand-eye calibration** — transform between a camera and a robot flange.
- **Homogeneous transform** — 4×4 matrix representing a rigid transform.
- **ICP** — iterative closest point registration.
- **Interlock** — device that prevents operation when a guard is open.
- **Inverse kinematics** — map from pose to joint angles.
- **Jacobian** — map from joint velocities to end-effector velocities.
- **Joint limit** — mechanical or software limit on a joint.
- **Manipulability** — \(\sqrt{\det(JJ^T)}\).
- **MJCF** — MuJoCo XML model format.
- **Payload** — mass and inertia carried by the robot.
- **PFL** — power and force limiting.
- **PL** — performance level in ISO 13849.
- **Pose** — position and orientation of a frame.
- **Quasi-static contact** — slow contact with force and pressure limits.
- **Quaternion** — four-parameter rotation representation.
- **Redundancy** — more joints than task dimensions.
- **Registration** — estimation of the transform between frames.
- **Repeatability** — closeness of repeated poses.
- **Safety envelope** — set of limits within which motion is permitted.
- **SIL** — safety integrity level in IEC 61508.
- **Singularity** — configuration at which the Jacobian loses rank.
- **SRMS** — safety-rated monitored stop.
- **SSM** — speed and separation monitoring.
- **TCP** — tool centre point.
- **Transform** — rigid translation and rotation between frames.
- **Transient contact** — dynamic contact with energy limits.
- **TRE** — target registration error.
- **URDF** — XML description of links, joints, and geometry.
- **Workspace** — set of reachable poses.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and regulation

- ISO 10218-1:2025, Robotics — Safety requirements — Part 1: Industrial robots. https://www.iso.org/standard/73933.html
- ISO 10218-2:2025, Robotics — Safety requirements — Part 2: Robot applications and robot cells. https://www.iso.org/
- ISO/TS 15066:2016, Robots and robotic devices — Collaborative robots. https://www.iso.org/standard/62996.html
- IEC 80601-2-77:2019+AMD1:2023, Robotically assisted surgical equipment. https://www.iso.org/standard/68473.html
- IEC 80601-2-78:2019+AMD1:2024, Medical robots for rehabilitation, assessment, compensation or alleviation. https://www.iso.org/standard/68474.html
- ISO 13849-1, Safety of machinery — Safety-related parts of control systems. https://www.iso.org/
- IEC 62061, Safety of machinery — Functional safety of safety-related control systems. https://webstore.iec.ch/
- IEC 61508, Functional safety of electrical/electronic/programmable electronic safety-related systems. https://webstore.iec.ch/

### Registration and robotics methods

- Fitzpatrick, J. M., Hill, D. L. G., and Maurer, C. R. (1998). Image registration. In Medical Image Processing and Analysis, SPIE Handbook.
- Fitzpatrick, J. M. (2009). The role of registration in accurate surgical guidance. Proceedings of the Institution of Mechanical Engineers, Part H. https://doi.org/10.1243/09544119jeim589
- Fitzpatrick, J. M. (2009). Fiducial registration error and target registration error are uncorrelated. SPIE Medical Imaging. https://doi.org/10.1117/12.813601
- Datteri, R. D., and Dawant, B. M. (2012). Estimation and reduction of target registration error. MICCAI. https://pmc.ncbi.nlm.nih.gov/articles/PMC4423533/
- Siciliano, B., et al. Robotics: Modelling, Planning and Control. Springer.
- Corke, P. Robotics, Vision and Control. Springer.

### Software and conventions

- REP 103, Standard Units of Measure and Coordinate Conventions. https://reps.openrobotics.org/rep-0103/
- REP 105, Coordinate Frames for Mobile Platforms. https://reps.openrobotics.org/rep-0105/
- ROS 2 documentation. https://docs.ros.org/
- MuJoCo documentation. https://mujoco.readthedocs.io/
- Pinocchio documentation. https://stack-of-tasks.github.io/pinocchio/
- MoveIt 2 documentation. https://moveit.picknik.ai/

> **Note:** The standard texts are paywalled. The papers and documentation above were used to check the teaching values. Confirm every clause, limit, and table against the current edition before citing a number in a safety case or a regulatory submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_hil-labs.md`](07_hil-labs.md)
- Next topic: [`09_evidence-dossier.md`](09_evidence-dossier.md)
- Sibling: [`05_firmware-labs.md`](05_firmware-labs.md)
- Sibling: [`02_bench-and-simulation-stack.md`](02_bench-and-simulation-stack.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
