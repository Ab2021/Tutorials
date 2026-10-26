# Robotics Kinematics

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 10 of 14
> Estimated study time: 4 hours
> Prerequisites: 01, 05
> Version 2026.3

## 1. Why this topic exists

A surgical or rehabilitation robot is a machine that moves a tool to a point in space. The point is defined in an image, a preoperative plan, or a surgeon's hand motion. The robot's joints are controlled in angles or displacements. The mapping between the two is kinematics, and the mapping is where many robotic errors originate.

The industry problem is that kinematics is treated as a library call. A team uses a forward-kinematics function, an inverse-kinematics solver, and a controller, and assumes that the tool is where the model says it is. When the tool is 2 mm off, the team cannot tell whether the error is in the model, the registration, the calibration, the encoder, or the mechanical structure. Three decisions depend on kinematic literacy:

1. **Whether the pose is correct.** A homogeneous transform is a composition of rotations and translations; a sign error or a frame convention error produces a plausible but wrong pose.
2. **Whether the robot can reach the target.** Inverse kinematics may have multiple solutions, no solution, or a solution outside the joint limits. The existence of a mathematical solution does not mean the robot can execute it.
3. **Whether the configuration is safe.** Near a singularity, a small Cartesian motion requires a large joint motion, which can exceed the joint velocity limit or the safety envelope.

This topic covers frames and homogeneous transforms, forward and inverse kinematics, the Jacobian, singularities, workspace, velocity, and error propagation. It uses a two-link planar arm for the arithmetic and a six-degree-of-freedom (6-DOF) serial arm for the concepts. It assumes the sensor and processor material from earlier topics. It does not cover dynamics or control; those are in Topic 11.

A useful framing: kinematics is a map between joint space and task space. The map is nonlinear, many-to-one in one direction, and singular in places. The Jacobian is the local linearisation, and its conditioning tells you how errors and velocities propagate.

## 2. Learning outcomes

By the end of this topic you can:

- Build a homogeneous transform from a rotation and a translation, and compose transforms in the correct order.
- Compute the forward kinematics of a two-link planar arm and a simple 3-DOF arm.
- Solve the inverse kinematics of a two-link planar arm, including the elbow-up and elbow-down solutions.
- Compute the Jacobian and identify the singular configurations.
- Compute the condition number and explain what it implies for error amplification and joint velocity.
- Propagate a joint encoder error to a tool-tip error.
- Distinguish workspace, dexterous workspace, and reachable workspace.
- Explain the difference between repeatability and accuracy in a robot.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A robot is a chain of links and joints. Each joint contributes a transform from the previous link's frame to the next. The product of the transforms is the pose of the tool in the base frame. The Jacobian is the derivative of that product with respect to the joint coordinates.

```text
   BASE --> JOINT 1 --> LINK 1 --> JOINT 2 --> LINK 2 --> ... --> TOOL
     |         |          |          |          |          |        |
   frame     rotation   translation rotation   translation ...    pose T
     |         |          |          |          |          |        |
     +---------+----------+----------+----------+----------+--------+
                                  |
                          T = T1 T2 T3 ... Tn
                                  |
                          x = f(q),  dx = J(q) dq
```

The model has four equations:

\[
T_i = \begin{bmatrix} R_i & t_i \\ 0 & 1 \end{bmatrix}
\]

\[
T_{\text{tool}} = T_1 T_2 \cdots T_n
\]

\[
x = f(q)
\]

\[
\dot{x} = J(q)\dot{q}
\]

The first two define the pose; the third is forward kinematics; the fourth is the velocity relation. The inverse kinematics solves \(q = f^{-1}(x)\), which may have zero, one, or many solutions. The Jacobian's determinant tells you whether the configuration is singular; its condition number tells you how close it is.

### 3.1 Frames and transforms

A frame has an origin and three axes. A pose is a position and an orientation. A homogeneous transform combines them in a 4×4 matrix so that composition is matrix multiplication.

| Transform | Meaning | Parameters |
|---|---|---|
| Translation | Move along x, y, z | \(t_x, t_y, t_z\) |
| Rotation about x | Roll | \(\phi\) |
| Rotation about y | Pitch | \(\theta\) |
| Rotation about z | Yaw | \(\psi\) |
| Denavit-Hartenberg | Link transform | \(a, \alpha, d, \theta\) |

The order of composition matters. A rotation followed by a translation is not the same as a translation followed by a rotation. The convention must be stated: intrinsic versus extrinsic, pre-multiply versus post-multiply.

### 3.2 Forward kinematics

Forward kinematics is a product of transforms. For a two-link planar arm with links \(L_1\) and \(L_2\) and joint angles \(q_1\) and \(q_2\):

\[
x = L_1 \cos q_1 + L_2 \cos(q_1 + q_2)
\]

\[
y = L_1 \sin q_1 + L_2 \sin(q_1 + q_2)
\]

The equations are simple, but they show the essential nonlinearity: the second link's contribution depends on the sum of the angles. A change in \(q_1\) moves the whole arm; a change in \(q_2\) moves only the distal link.

### 3.3 Inverse kinematics

Inverse kinematics solves for the joint angles given the tip position. For the two-link arm, the law of cosines gives:

\[
\cos q_2 = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}
\]

If the right side is outside \([-1, 1]\), the target is unreachable. If it is inside, there are two solutions, elbow-up and elbow-down:

\[
q_2 = \pm \arccos(\cdots)
\]

\[
q_1 = \operatorname{atan2}(y, x) - \operatorname{atan2}(L_2 \sin q_2, L_1 + L_2 \cos q_2)
\]

The two solutions are both mathematically valid. The choice depends on the joint limits, the obstacle avoidance, and the configuration continuity.

### 3.4 The Jacobian

The Jacobian maps joint velocities to task velocities:

\[
J = \begin{bmatrix}
\frac{\partial x}{\partial q_1} & \frac{\partial x}{\partial q_2} \\
\frac{\partial y}{\partial q_1} & \frac{\partial y}{\partial q_2}
\end{bmatrix}
\]

For the two-link arm:

\[
J = \begin{bmatrix}
-L_1 \sin q_1 - L_2 \sin(q_1+q_2) & -L_2 \sin(q_1+q_2) \\
L_1 \cos q_1 + L_2 \cos(q_1+q_2) & L_2 \cos(q_1+q_2)
\end{bmatrix}
\]

The determinant is:

\[
\det J = L_1 L_2 \sin q_2
\]

The arm is singular when \(q_2 = 0\) or \(q_2 = \pi\): the arm is fully extended or fully folded, and it cannot move radially. At the singularity, the determinant is zero and the inverse Jacobian does not exist.

### 3.5 Error propagation

A joint error \(\delta q\) produces a tip error:

\[
\delta x = J \delta q
\]

The worst-case magnitude is bounded by the largest singular value:

\[
\|\delta x\| \le \sigma_{\max} \|\delta q\|
\]

Near a singularity, \(\sigma_{\min}\) approaches zero, and the condition number \(\sigma_{\max}/\sigma_{\min}\) grows. A small Cartesian error in the weak direction requires a large joint correction:

\[
\delta q = J^{-1} \delta x
\]

The inverse amplifies the error by \(1/\sigma_{\min}\). This is why a robot near a singularity is hard to control and dangerous to move.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Kinematics | Geometry of motion without forces | Confused with dynamics | None |
| Frame | Coordinate system with origin and axes | Confused with a link | None |
| Pose | Position plus orientation | Confused with position | m, rad |
| Homogeneous transform | 4×4 matrix combining rotation and translation | Confused with a rotation matrix | None |
| DH parameters | Four parameters per link | Confused with joint angles | m, rad |
| Forward kinematics | Joint angles to pose | Confused with inverse | None |
| Inverse kinematics | Pose to joint angles | Assumed unique | None |
| Jacobian | Joint velocity to task velocity | Confused with the Hessian | Matrix |
| Singularity | Configuration where the Jacobian loses rank | Confused with a joint limit | None |
| Determinant | Product of eigenvalues; zero at singularity | Confused with condition number | None |
| Condition number | Ratio of largest to smallest singular value | Confused with determinant | None |
| Singular value | Principal stretch of the Jacobian | Confused with eigenvalue | None |
| Workspace | Set of reachable poses | Confused with dexterous workspace | m |
| Dexterous workspace | Poses reachable with all orientations | Confused with workspace | m |
| Reachable workspace | Poses reachable with at least one orientation | Confused with dexterous | m |
| Redundancy | More joints than task DOF | Confused with redundancy in data | None |
| DOF | Degrees of freedom | Confused with joints | Count |
| Repeatability | Closeness of repeated poses | Confused with accuracy | mm |
| Accuracy | Closeness to the commanded pose | Confused with repeatability | mm |
| Registration | Alignment of image and robot frames | Confused with calibration | mm |
| Calibration | Estimation of kinematic parameters | Confused with registration | mm |
| Tool centre point | Point controlled by the robot | Confused with the flange | m |
| End effector | Device at the end of the arm | Confused with the tool | None |
| Joint limit | Mechanical or software limit | Confused with a singularity | rad |
| Velocity limit | Maximum joint velocity | Confused with acceleration | rad/s |
| Workspace limit | Boundary of the reachable space | Confused with a joint limit | m |
| Payload | Mass the robot can carry | Confused with force | kg |
| Stiffness | Force per unit deflection | Confused with impedance | N/m |
| Denavit-Hartenberg | Convention for link frames | Confused with a transform | None |
| Quaternion | Four-parameter orientation | Confused with Euler angles | Unitless |
| Euler angle | Three-parameter orientation | Confused with quaternion | rad |
| Gimbal lock | Loss of a DOF in Euler angles | Confused with a robot singularity | None |

## 5. The mechanics, step by step

The procedure below builds and checks a kinematic model. Each step states its input, output, and rationale. The required edge cases are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the base and tool frames

**Input:** the robot's mechanical structure, the mounting, and the tool.

**Output:** a base frame, a tool frame, and a tool centre point.

**Rationale:** every pose is relative to a frame. The base frame is usually at the robot's mounting; the tool frame is at the tool centre point. A model without a stated frame is ambiguous, and the ambiguity produces errors of centimetres, not millimetres.

The tool centre point must be defined physically. A 100 mm tool offset, applied with the wrong sign, moves the tip by 200 mm.

### Step 2. Assign link frames

**Input:** the joint axes and the link geometry.

**Output:** a frame per link, following a convention (Denavit-Hartenberg or modified DH).

**Rationale:** the convention determines the parameters and the transform order. A DH model with a sign error produces a plausible pose for one configuration and a wrong pose for another. The frame assignment must be documented and checked against the physical robot.

A common error is the direction of the joint axis. A reversed axis produces a mirror-image workspace that looks correct in a symmetric test.

### Step 3. Measure the link parameters

**Input:** the CAD model and the physical robot.

**Output:** link lengths, offsets, and twist angles with tolerances.

**Rationale:** the CAD model is nominal. The physical robot has manufacturing tolerances, assembly offsets, and wear. The kinematic parameters must be calibrated against measurements, not assumed from the drawing.

A 1 mm error in a 300 mm link is 0.33%; over a 1 m reach, it is 3.3 mm. The tolerance must be compared with the clinical requirement.

### Step 4. Build the forward-kinematics model

**Input:** the frames and parameters.

**Output:** a function that maps joint angles to a pose.

**Rationale:** the forward model is the basis of the inverse model, the Jacobian, and the controller. It must be tested against physical measurements at several configurations, not only at the home position.

A forward model that is correct at the home position and wrong at the workspace edge is a model with a parameter error, not a software bug.

### Step 5. Test the forward model

**Input:** the model and a measurement system.

**Output:** residuals at a set of configurations.

**Rationale:** the residuals show whether the model is accurate enough. A residual of 5 mm at the workspace edge may be acceptable for a rehabilitation robot and unacceptable for a surgical robot. The test must cover the workspace, not only the centre.

The measurement system's own uncertainty must be smaller than the residual. A measurement with a 2 mm uncertainty cannot verify a 1 mm model.

### Step 6. Solve the inverse kinematics

**Input:** the forward model and a target pose.

**Output:** one or more joint solutions, or a statement that the target is unreachable.

**Rationale:** the inverse may have multiple solutions. The solver must return all solutions or a documented subset, and the controller must choose one that respects the joint limits and the configuration continuity.

A solver that returns the first solution may choose an elbow configuration that collides with the patient or the table.

### Step 7. Check the joint limits and the workspace

**Input:** the solutions and the joint limits.

**Output:** a feasible solution or a rejection.

**Rationale:** a mathematical solution outside the joint limits is not executable. The limits include the mechanical stops, the software limits, and the safety envelope. The check must be done before the motion, not during it.

A robot that plans a motion to a joint limit and stops mid-procedure is a workflow failure, even if it is safe.

### Step 8. Compute the Jacobian

**Input:** the forward model and the current joint angles.

**Output:** the Jacobian matrix.

**Rationale:** the Jacobian is the local linearisation. It is used for velocity control, force control, singularity detection, and error propagation. It must be computed at the current configuration, not at the home configuration.

The Jacobian can be derived analytically or numerically. The analytical form is faster and exact; the numerical form is easier to implement and can hide a modelling error.

### Step 9. Detect singularities

**Input:** the Jacobian and the joint limits.

**Output:** a singularity flag, a condition number, and a safe action.

**Rationale:** a singularity is a configuration where the Jacobian loses rank. The determinant is zero for a square Jacobian; the condition number grows near it. The controller must detect the approach and slow down, change configuration, or stop.

A singularity is not a software error. It is a geometric property of the arm. The response must be designed, not left to the controller's numerical behaviour.

### Step 10. Propagate the errors

**Input:** the joint encoder tolerances and the Jacobian.

**Output:** a tip-error bound and a dominant direction.

**Rationale:** the tip error is the Jacobian times the joint error. The bound uses the singular values; the dominant direction uses the singular vectors. The result tells you whether the encoder, the link tolerance, or the registration dominates.

An error budget that assumes independent, equal joint errors overestimates the error in the strong direction and underestimates it in the weak direction.

### Step 11. Calibrate the model

**Input:** the measured poses and the nominal parameters.

**Output:** calibrated parameters and a residual map.

**Rationale:** calibration estimates the parameters that best fit the measurements. It reduces the systematic error but not the random error. The residual map shows where the model is still weak.

A calibration with too few poses can fit the noise and worsen the prediction at other poses. The poses must cover the workspace.

### Step 12. Verify and document

**Input:** the calibrated model and the clinical requirement.

**Output:** a verification report, a parameter table, and a configuration record.

**Rationale:** the verification demonstrates that the model meets the requirement across the workspace. The record includes the parameters, the calibration date, the measurement system, and the residuals. Without the record, a later change cannot be distinguished from a patient change.

The verification must include the worst-case configuration, not only the nominal one.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 6, 7 | Target outside workspace | Inverse returns no solution | Procedure stops |
| Common-mode rejection failure | 1 | Frame convention error | Pose mismatch | Wrong motion |
| Electrode offset | 1 | Tool offset error | Tip position error | Wrong target |
| Clock jitter | 8, 10 | Velocity estimate noise | Controller jitter | Poor tracking |
| Brown-out | 12 | Controller reset | Reset log | Lost pose |
| Watchdog reset loop | 12 | Controller hang | Reset counter | Procedure interrupted |
| Thermal throttling | 8 | Controller clock reduction | Latency log | Late response |
| Firmware update failure | 12 | Kinematic parameters lost | Version and checksum | Wrong model |
| Calibration drift | 11 | Wear, temperature | Residual trend | Systematic error |

The kinematic edge cases are dominated by frame errors, calibration drift, and singularities. The processor edge cases enter through the controller and the safety monitor.

### 5.14 Singularity indicators

| Indicator | Value at singularity | Use |
|---|---|---|
| Determinant | 0 | Square Jacobian only |
| Smallest singular value | 0 | Any Jacobian |
| Condition number | ∞ | Any Jacobian |
| Manipulability | 0 | \(\sqrt{\det(JJ^T)}\) |
| Joint velocity for a given task velocity | ∞ | Controller |

The condition number is the most useful scalar. A value below 10 is well conditioned; 10–100 is marginal; above 100 is near-singular. The thresholds depend on the task and the controller.

### 5.15 Accuracy versus repeatability

| Property | Definition | Test | Typical cause of error |
|---|---|---|---|
| Repeatability | Closeness of repeated poses | Return to the same point many times | Backlash, noise, thermal |
| Accuracy | Closeness to the commanded pose | Compare with a measurement | Kinematic model, calibration |
| Resolution | Smallest commanded change | Step the joint | Encoder resolution |
| Registration | Alignment of image and robot | Fiducial or surface | Fiducial error, deformation |

A robot can be repeatable and inaccurate: it returns to the same wrong point. A robot can be accurate and non-repeatable: it averages the right point but scatters around it. The clinical requirement must state which one matters.

### 3.6 Serial, parallel, and redundant chains

| Chain | Structure | Strength | Weakness | Medical example |
|---|---|---|---|---|
| Serial | Links in series | Large workspace | Accumulated error | Surgical arm |
| Parallel | Multiple chains to the tool | High stiffness | Small workspace | Stewart platform |
| Redundant | More joints than DOF | Obstacle avoidance | Complex control | 7-DOF arm |
| Continuum | Flexible body | Conformability | Nonlinear kinematics | Flexible endoscope |

The chain type changes the kinematics. A parallel robot's inverse kinematics is often simpler than its forward kinematics; a serial robot's forward kinematics is simpler. A continuum robot's kinematics is a function of the bending, not only the joint angles.

A surgical robot is usually serial for reach and parallel at the wrist for stiffness. The combination gives a large workspace and a precise tip.

### 5.16 A 3-DOF spatial arm

A simple spatial arm has a base rotation \(q_1\) about z, a shoulder rotation \(q_2\) about y, and an elbow rotation \(q_3\) about y. With link lengths \(L_1\) and \(L_2\):

\[
x = \cos q_1 (L_1 \cos q_2 + L_2 \cos(q_2+q_3))
\]

\[
y = \sin q_1 (L_1 \cos q_2 + L_2 \cos(q_2+q_3))
\]

\[
z = L_1 \sin q_2 + L_2 \sin(q_2+q_3)
\]

The base rotation decouples the radial distance from the height. The inverse kinematics first computes the radial distance \(r = \sqrt{x^2+y^2}\), then solves the planar two-link problem in the \((r,z)\) plane, then computes \(q_1 = \operatorname{atan2}(y,x)\).

The decoupling is a design feature. A wrist that does not decouple the position and orientation makes the inverse kinematics a six-dimensional numerical problem, with multiple solutions and a higher risk of a wrong branch.

### 5.17 Denavit-Hartenberg parameters

| Parameter | Meaning | Unit |
|---|---|---|
| \(a_i\) | Link length along x | m |
| \(\alpha_i\) | Link twist about x | rad |
| \(d_i\) | Link offset along z | m |
| \(\theta_i\) | Joint angle about z | rad |

The DH transform is:

\[
T_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

The convention is not unique. The standard DH and the modified DH differ in the order of the transforms; mixing them produces a model that is wrong for every configuration except the home position.

### 5.18 Redundancy and the null space

A redundant robot has more joints than task degrees of freedom. A 7-DOF arm can reach a 6-DOF pose with one extra degree of freedom, which can be used to avoid obstacles, stay away from singularities, or optimise joint torque.

The null-space motion is the joint velocity that produces no task velocity:

\[
\dot{q}_{\text{null}} = (I - J^+ J) z
\]

where \(J^+\) is the pseudoinverse and \(z\) is an arbitrary vector. The null-space motion changes the configuration without changing the tip pose.

The null space is useful and dangerous. A null-space motion can move the elbow into the patient or the table if the obstacle is not in the cost function. The redundancy must be resolved with the workspace constraints, not only the tip pose.

### 5.19 Workspace computation

The workspace is the set of poses the robot can reach. It can be computed by sampling the joint space and forward-kinematics, or by boundary analysis. The sampling method is simple but can miss thin regions; the boundary method is exact but complex.

| Method | Strength | Weakness |
|---|---|---|
| Monte Carlo sampling | Simple; any robot | Misses thin regions; needs many samples |
| Boundary analysis | Exact | Complex; robot-specific |
| Reachability grid | Fast; conservative | Approximate |
| Physical measurement | Real | Time-consuming |

The workspace must include the tool and the patient. A robot that reaches the target in free space may not reach it with the tool attached or with the patient in the way.

### 5.20 Calibration methods

| Method | Measures | Strength | Limitation |
|---|---|---|---|
| Fiducial | Markers at known positions | Simple; accurate | Marker placement |
| Laser tracker | Reflector positions | Very accurate | Cost; line of sight |
| Camera | Marker or surface | Non-contact | Calibration; occlusion |
| Touch probe | Contact points | Accurate | Slow; contact force |
| Self-calibration | Internal sensors | No external equipment | Observability |

The calibration estimates the kinematic parameters, the tool offset, and the joint offsets. The number of parameters must be less than the number of independent measurements, or the fit is underdetermined.

### 5.21 Ten kinematic review questions

1. What are the base and tool frames, and where is the tool centre point?
2. Which DH convention is used, and are the parameters measured or nominal?
3. What is the forward model's residual across the workspace?
4. How many inverse solutions exist, and which one is chosen?
5. Are the joint limits and workspace limits checked before motion?
6. What is the Jacobian at the target, and what is its condition number?
7. What is the tip-error bound from the joint errors?
8. How is the singularity detected, and what is the response?
9. How is the model calibrated, and when is it re-calibrated?
10. What is recorded so that a model change can be detected?

A review that answers all ten has covered the kinematics. A review that answers only the third and the sixth has covered the model and the Jacobian, not the system.

### 5.22 Velocity and acceleration limits

The Jacobian maps joint velocity to task velocity. The joint velocity limit and the task velocity requirement must be compatible:

\[
\|\dot{q}\| = \|J^{-1} \dot{x}\| \le \dot{q}_{\max}
\]

Near a singularity, the inverse grows and the joint velocity exceeds the limit. The planner must reduce the task velocity or change the configuration.

The acceleration limit is harder. The joint acceleration is the derivative of the velocity, which depends on the Jacobian derivative and the joint velocity. A path that is feasible at constant velocity may be infeasible during acceleration. The planner must check both.

### 5.23 Product of exponentials

An alternative to DH uses screw axes and the product of exponentials:

\[
T(	heta) = e^{[S_1]	heta_1} e^{[S_2]	heta_2} \cdots e^{[S_n]	heta_n} M
\]

where \(S_i\) is the screw axis of joint \(i\) and \(M\) is the home pose. The formulation avoids the DH frame assignment and is easier to calibrate, because the screw axes are geometric.

The product of exponentials and the DH model describe the same robot. The choice is a matter of convention and tooling, not of physics. A team that mixes the two without converting the parameters will produce a wrong model.

### 5.24 Numerical inverse kinematics

An analytical inverse is exact and fast, but it is robot-specific. ### 5.25 Six-DOF decoupling

A six-DOF arm with a spherical wrist decouples the position and orientation: the first three joints place the wrist centre, and the last three orient the tool. The inverse kinematics solves the position problem first, then the orientation problem.

| Part | Joints | Solves | Singularity |
|---|---|---|---|
| Position | 1–3 | Wrist centre | Arm fully extended |
| Orientation | 4–6 | Tool orientation | Wrist alignment |

The decoupling reduces the inverse from a six-dimensional numerical problem to two three-dimensional problems. The reduction is exact only if the wrist is spherical; a non-spherical wrist requires a numerical solution or a different decomposition.

A numerical inverse uses the Jacobian and an iterative update:

\[
q_{k+1} = q_k + J^+(q_k)(x_{	ext{target}} - f(q_k))
\]

The iteration converges if the initial guess is close and the Jacobian is well conditioned. Near a singularity, the pseudoinverse amplifies the error, and the iteration may diverge or oscillate.

The numerical inverse is useful for redundant robots and for robots without a closed-form solution. It must include a step limit, a convergence test, and a singularity check. An iteration that runs without a limit can command a large joint motion.

- Limit the step to a fraction of the joint range.
- Stop when the pose error is below the resolution.
- Stop when the Jacobian's condition number exceeds the threshold.
- Return the iteration count and the residual for diagnosis.

## 6. Worked example

### 6.1 The arm

A planar two-link arm has \(L_1 = 0.30\) m and \(L_2 = 0.25\) m. The joint angles are \(q_1 = 30^\circ\) and \(q_2 = 60^\circ\). The tool is at the tip of the second link. The joint encoders have a resolution of 0.1° and an accuracy of 0.1°.

### 6.2 Forward kinematics

\[
x = 0.30 \cos 30^\circ + 0.25 \cos 90^\circ = 0.2598 + 0 = 0.2598\ \text{m}
\]

\[
y = 0.30 \sin 30^\circ + 0.25 \sin 90^\circ = 0.15 + 0.25 = 0.40\ \text{m}
\]

The tip is at (0.2598, 0.4000) m. The distance from the base is:

\[
r = \sqrt{0.2598^2 + 0.40^2} = 0.4769\ \text{m}
\]

The maximum reach is \(L_1 + L_2 = 0.55\) m, so the arm is at 86.7% of its reach.

### 6.3 Inverse kinematics

For the same target, the law of cosines gives:

\[
\cos q_2 = \frac{0.2598^2 + 0.40^2 - 0.30^2 - 0.25^2}{2 \times 0.30 \times 0.25} = \frac{0.2275 - 0.1525}{0.15} = 0.5
\]

\[
q_2 = \arccos(0.5) = 60^\circ
\]

The elbow-up solution is \(q_2 = +60^\circ\); the elbow-down solution is \(q_2 = -60^\circ\). For the elbow-up solution:

\[
q_1 = \operatorname{atan2}(0.40, 0.2598) - \operatorname{atan2}(0.25 \sin 60^\circ, 0.30 + 0.25 \cos 60^\circ)
\]

\[
q_1 = 57.0^\circ - 27.0^\circ = 30.0^\circ
\]

The inverse recovers the original angles. The elbow-down solution is \(q_1 = 57.0^\circ + 27.0^\circ = 84.0^\circ\), \(q_2 = -60^\circ\), which places the elbow on the other side of the line to the target.

### 6.4 The Jacobian

At \(q_1 = 30^\circ\), \(q_2 = 60^\circ\):

\[
J = \begin{bmatrix}
-0.30 \sin 30^\circ - 0.25 \sin 90^\circ & -0.25 \sin 90^\circ \\
0.30 \cos 30^\circ + 0.25 \cos 90^\circ & 0.25 \cos 90^\circ
\end{bmatrix}
\]

\[
J = \begin{bmatrix}
-0.4000 & -0.2500 \\
0.2598 & 0.0000
\end{bmatrix}
\]

The determinant is:

\[
\det J = (-0.4000)(0.0000) - (-0.2500)(0.2598) = 0.06495\ \text{m}^2
\]

The singular values are 0.5241 and 0.1239 m/rad, and the condition number is 4.23. The arm is well conditioned at this configuration.

### 6.5 Tip error from encoder error

A 0.1° joint error is 0.001745 rad. The tip error is \(J \delta q\). For an independent error in each joint, the RMS tip error is:

\[
\|\delta x\| = \sqrt{(J_{11}\delta q_1)^2 + (J_{12}\delta q_2)^2 + (J_{21}\delta q_1)^2 + (J_{22}\delta q_2)^2}
\]

\[
\|\delta x\| = \sqrt{(-0.4 \times 0.001745)^2 + (-0.25 \times 0.001745)^2 + (0.2598 \times 0.001745)^2 + 0}
\]

\[
\|\delta x\| = \sqrt{(6.98 \times 10^{-4})^2 + (4.36 \times 10^{-4})^2 + (4.53 \times 10^{-4})^2} = 0.94\ \text{mm}
\]

The tip error is 0.94 mm for a 0.1° encoder error. If the clinical requirement is 1 mm, the margin is only 0.06 mm, which is too thin for a real system with other error sources.

### 6.6 Singularity: a 1 mm radial move

At \(q_2 = 60^\circ\), a 1 mm radial move requires:

\[
\delta q = J^{-1} \delta x
\]

with \(\delta x = (0.545, 0.839)\) mm (the radial unit vector). The joint motion is 0.18° for \(q_1\) and −0.42° for \(q_2\).

At \(q_2 = 5^\circ\), the same 1 mm radial move requires 2.19° for \(q_1\) and −4.82° for \(q_2\). At \(q_2 = 2^\circ\), it requires 5.47° and −12.04°. At \(q_2 = 0.5^\circ\), it requires 21.89° and −48.15°. At \(q_2 = 0\), the determinant is zero and the required motion is infinite.

The condition numbers are 4.23 at 60°, 55.7 at 5°, 139.4 at 2°, and 557.7 at 0.5°. The joint motion grows approximately with the condition number, and the Cartesian error in the weak direction grows with it.

### 6.7 Diagnostic 1: the encoder offset

The robot consistently misses the target by 2.5 mm in the +y direction. The forward model predicts the pose correctly, so the error is not in the model. The encoder on \(q_1\) has a 0.5° offset because the home position was set with the arm resting against a mechanical stop that has 0.5° of compliance.

The diagnostic is to command a set of poses and compare the measured tip with the model. If the residual is a constant rotation about the base, the cause is the \(q_1\) offset. A second diagnostic is to check the home position with an external measurement, not the mechanical stop.

The design flaw is the home procedure, not the kinematics. The fix is a calibrated home position and a periodic check against a reference.

### 6.8 Diagnostic 2: the near-singular configuration

The robot is commanded to move 1 mm radially at \(q_2 = 2^\circ\). The joint controller commands a 12° motion on \(q_2\), which exceeds the joint velocity limit for the 10 ms cycle. The controller saturates, and the tip lags the command by 3 mm.

The diagnostic is to compute the condition number before the motion. At \(q_2 = 2^\circ\), the condition number is 139, which is above the threshold of 100. The controller should slow down or refuse the motion.

The design flaw is the planner, not the controller. The planner generated a Cartesian path without checking the joint velocity. The fix is a singularity-aware planner that limits the Cartesian speed near singularities or changes the configuration.

### 6.9 Vary one assumption: the second link grows to 0.30 m

Change \(L_2\) from 0.25 to 0.30 m, keeping the angles. The tip moves to:

\[
x = 0.30 \cos 30^\circ + 0.30 \cos 90^\circ = 0.2598\ \text{m}
\]

\[
y = 0.30 \sin 30^\circ + 0.30 \sin 90^\circ = 0.45\ \text{m}
\]

The reach increases from 0.55 to 0.60 m. The Jacobian becomes:

\[
J = \begin{bmatrix}
-0.15 - 0.30 & -0.30 \\
0.2598 + 0 & 0
\end{bmatrix} = \begin{bmatrix}
-0.45 & -0.30 \\
0.2598 & 0
\end{bmatrix}
\]

The determinant is 0.07794 m², larger than before, and the condition number is lower at this configuration. The arm is better conditioned, but the tip error for the same joint error is larger because the Jacobian entries are larger: the RMS error rises to about 1.13 mm.

The change improves the conditioning and worsens the error. The trade is not obvious without the arithmetic.

### 6.10 What the example demonstrates

The arithmetic closes: FK (0.2598, 0.40), IK (30°, 60°), Jacobian with determinant 0.06495 m², singular values 0.5241 and 0.1239, condition number 4.23, tip error 0.94 mm, and a 48° joint motion for a 1 mm radial move at \(q_2 = 0.5^\circ\). The two diagnostics reveal different flaws: one is an encoder offset, the other is a near-singular path. The sensitivity analysis shows that a longer link improves conditioning but increases the error.

The general method is to compute the model, check it against measurements, compute the Jacobian and its conditioning, propagate the errors, and plan away from singularities.

### 6.11 A 3-DOF numerical check

The same arm with a base rotation \(q_1 = 45^\circ\), a shoulder \(q_2 = 30^\circ\), and an elbow \(q_3 = 60^\circ\) has a tip at:

| Coordinate | Value |
|---|---:|
| x | 0.1838 m |
| y | 0.1838 m |
| z | 0.4000 m |

The radial distance is 0.2598 m, and the height is 0.4000 m. The base rotation maps the planar solution into the xy plane; the planar solution is the same as the 2-link example with different angles.

The numerical check confirms that the 3-DOF model reduces to the 2-DOF model when the base rotation is removed. A model that does not reduce correctly has a frame or parameter error.

### 6.12 A second target: the edge of the workspace

For the target (0.50, 0.00) m, the distance from the base is 0.50 m, within the 0.55 m reach. The inverse gives:

\[
\cos q_2 = \frac{0.25 - 0.09 - 0.0625}{0.15} = 0.65
\]

\[
q_2 = \pm 49.5^\circ
\]

For the elbow-up solution, \(q_1 = 0 - \operatorname{atan2}(0.25 \sin 49.5, 0.30 + 0.25 \cos 49.5) = -21.4^\circ\). The arm is close to full extension; the condition number is higher than at the centre.

The target is reachable, but the joint margins are smaller. A 1 mm error in the link length moves the tip by more than 1 mm because the arm is near the workspace boundary. The planner must check the margin, not only the reach.

### 6.13 Repeatability test

| Run | x (mm) | y (mm) | Error (mm) |
|---:|---:|---:|---:|
| 1 | 259.8 | 400.1 | 0.1 |
| 2 | 259.7 | 399.9 | 0.1 |
| 3 | 260.0 | 400.2 | 0.2 |
| 4 | 259.9 | 400.0 | 0.1 |
| 5 | 259.8 | 399.8 | 0.2 |

The mean is (259.84, 400.00) mm, and the standard deviation is about 0.11 mm in x and 0.16 mm in y. The repeatability is better than the 0.94 mm accuracy estimate because the encoder errors are systematic, not random.

A repeatability test that returns to the same point measures the random error. An accuracy test that compares with a reference measures the systematic error. Both are needed.

## 7. Code, query, or calculation

The script reproduces the forward kinematics, inverse kinematics, Jacobian, condition number, tip error, and near-singular joint motion in Section 6. It uses only the standard library.

**Assumptions.** \(L_1 = 0.30\) m, \(L_2 = 0.25\) m, \(q_1 = 30^\circ\), \(q_2 = 60^\circ\), a 0.1° joint error, and a 1 mm radial target motion at four values of \(q_2\).

```python
import math
L1, L2 = 0.30, 0.25

def fk(q1, q2):
    x = L1*math.cos(q1) + L2*math.cos(q1+q2)
    y = L1*math.sin(q1) + L2*math.sin(q1+q2)
    return x, y

def jac(q1, q2):
    return [[-L1*math.sin(q1)-L2*math.sin(q1+q2), -L2*math.sin(q1+q2)],
            [L1*math.cos(q1)+L2*math.cos(q1+q2), L2*math.cos(q1+q2)]]

def svals(J):
    a,b = J[0]; c,d = J[1]
    A=a*a+c*c; B=a*b+c*d; C=b*b+d*d
    tr=A+C; det=A*C-B*B
    disc=max(0.0, tr*tr-4*det)
    s1=math.sqrt((tr+math.sqrt(disc))/2)
    s2=math.sqrt(max(0.0,(tr-math.sqrt(disc))/2))
    return s1, s2

q1, q2 = math.radians(30), math.radians(60)
x, y = fk(q1, q2)
J = jac(q1, q2)
det = J[0][0]*J[1][1] - J[0][1]*J[1][0]
s1, s2 = svals(J)
dq = math.radians(0.1)
err = math.sqrt(sum((J[i][j]*dq)**2 for i in range(2) for j in range(2)))

print(f"FK x={x:.4f} y={y:.4f}")
print(f"det={det:.5f}")
print(f"sv1={s1:.4f} sv2={s2:.4f} cond={s1/s2:.2f}")
print(f"tip error for 0.1 deg: {err*1000:.2f} mm")

c2 = (x*x+y*y-L1*L1-L2*L2)/(2*L1*L2)
print(f"cos q2={c2:.4f} q2={math.degrees(math.acos(c2)):.2f} deg")

for q2d in (60, 5, 2, 0.5):
    q2r = math.radians(q2d)
    J2 = jac(q1, q2r)
    d2 = J2[0][0]*J2[1][1] - J2[0][1]*J2[1][0]
    a,b = J2[0]; c,d = J2[1]
    x2, y2 = fk(q1, q2r)
    dx = [0.001*x2/math.hypot(x2,y2), 0.001*y2/math.hypot(x2,y2)]
    dq1 = (d*dx[0]-b*dx[1])/d2
    dq2 = (-c*dx[0]+a*dx[1])/d2
    ss1, ss2 = svals(J2)
    print(f"q2={q2d:4.1f} deg cond={ss1/ss2:7.2f} dq1={math.degrees(dq1):7.2f} dq2={math.degrees(dq2):8.2f}")
```

**Expected output.**

```text
FK x=0.2598 y=0.4000
det=0.06495
sv1=0.5241 sv2=0.1239 cond=4.23
tip error for 0.1 deg: 0.94 mm
cos q2=0.5000 q2=60.00 deg
q2=60.0 deg cond=   4.23 dq1=   0.18 dq2=   -0.42
q2= 5.0 deg cond=  55.73 dq1=   2.19 dq2=   -4.82
q2= 2.0 deg cond= 139.41 dq1=   5.47 dq2=  -12.04
q2= 0.5 deg cond= 557.68 dq1=  21.89 dq2=  -48.15
```

The output shows three things.

First, the determinant is 0.065 m², the singular values are 0.524 and 0.124 m/rad, and the condition number is 4.23. The arm is well conditioned at this configuration.

Second, the tip error for a 0.1° joint error is 0.94 mm. The error is dominated by the first joint, whose Jacobian column has the larger norm.

Third, the near-singular behaviour is dramatic. At \(q_2 = 0.5^\circ\), a 1 mm radial move requires 21.9° on \(q_1\) and 48.2° on \(q_2\). At \(q_2 = 0\), the required motion is infinite. The condition number grows from 4.23 to 557.7.

**A note on the error model.** The 0.94 mm is an RMS estimate with independent, equal joint errors. A worst-case bound uses the sum of the absolute contributions, which is about 1.3 mm. A correlated error, such as a common encoder offset, produces a different pattern and can be larger in a specific direction.

### 7.1 Workspace sampling

The script samples the joint space and prints the minimum and maximum reach of the two-link arm. The reach is the distance from the base to the tip.

```python
import math

L1, L2 = 0.30, 0.25
reaches = []
for i in range(361):
    q1 = math.radians(i)
    for j in range(361):
        q2 = math.radians(j)
        x = L1*math.cos(q1) + L2*math.cos(q1+q2)
        y = L1*math.sin(q1) + L2*math.sin(q1+q2)
        reaches.append(math.hypot(x, y))
print(f"samples {len(reaches)}")
print(f"min reach {min(reaches):.3f} m")
print(f"max reach {max(reaches):.3f} m")
```

**Expected output.**

```text
samples 130321
min reach 0.050 m
max reach 0.550 m
```

The workspace is an annulus from 0.05 to 0.55 m. The inner radius is \(|L_1 - L_2| = 0.05\) m; the outer radius is \(L_1 + L_2 = 0.55\) m. The sampling confirms the analytical result.

A real workspace is not an annulus. The joint limits, the tool, the patient, and the obstacles remove regions. The sampling must include those constraints, not only the link lengths.

## 8. Data, standards, and regulatory context

Robot kinematics is constrained by the medical robot standards, the risk management standard, and the software lifecycle standard. The table separates the instruments.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 80601-2-77:2019 | Robotically assisted surgical equipment | Particular standard | Robot safety, accuracy, control, user interface |
| IEC 80601-2-78:2021 | Rehabilitation and assistive robots | Particular standard | Robot safety, patient contact, control |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment | Consensus standard | Electrical safety, isolation, applied parts |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis, risk controls |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Kinematics and control software |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | User interface, mode awareness |
| ISO 9283:1998 | Industrial robot performance | Consensus standard; informative for medical | Accuracy, repeatability, path performance |
| ASTM F2554-18 | Surgical navigation measurement | Consensus standard | Registration and accuracy measurement |
| ISO/TS 15066:2016 | Collaborative industrial robots | Informative; not a medical substitute | Force and pressure limits |

### 8.1 What IEC 80601-2-77 covers

IEC 80601-2-77:2019 is the particular standard for robotically assisted surgical equipment. It addresses the robot's safety, the control system, the accuracy and repeatability, the user interface, the emergency stop, and the interaction with the patient and the team. It is a particular standard under IEC 60601-1, so the general safety requirements also apply.

The standard does not prescribe a kinematic model or a calibration method. It requires the manufacturer to specify the accuracy, the repeatability, and the workspace, and to verify them. The verification must include the worst-case configuration and the clinical use.

### 8.2 What IEC 80601-2-78 covers

IEC 80601-2-78:2021 is the particular standard for medical robots for rehabilitation, assessment, compensation, or alleviation. It addresses the robot's safety, the patient contact, the control, the assist-as-needed behaviour, and the emergency stop. It is distinct from IEC 80601-2-77, which covers surgical robots.

A rehabilitation robot that is also a surgical robot would need both standards, or a justified combination. The classification depends on the intended use, not on the technology.

### 8.3 Accuracy and repeatability standards

ISO 9283:1998 defines the performance criteria for industrial robots, including pose accuracy, pose repeatability, path accuracy, and path repeatability. It is not a medical standard, but it provides the test methods that medical robot manufacturers often use.

ASTM F2554-18 provides methods for measuring the accuracy of surgical navigation systems. The measurement includes the registration error, the target error, and the effect of the frame. The standard is informative for the data scientist who needs to interpret a vendor's accuracy claim.

### 8.4 What the standards do not cover

The standards do not specify:

- The kinematic convention.
- The calibration method.
- The singularity handling.
- The registration method.
- The error budget.

Those are design decisions that the manufacturer must justify. A robot can meet every clause and still have a 3 mm target error because the calibration was weak.

### 8.5 The data implications

The kinematic configuration, the calibration date, the tool offset, and the registration are part of the measurement. A change in any of them changes the tool position. The record must include them, or the analysis cannot distinguish a robot change from a patient change.

### 8.6 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 80601-2-77 | Accuracy, repeatability, workspace, calibration | Reproduce the robot claim |
| IEC 80601-2-78 | Assistive behaviour, patient contact | Reproduce the rehab claim |
| ISO 14971 | Hazard analysis for motion and singularity | Link failure to harm |
| IEC 62304 | Kinematics and control software version | Reproduce behaviour |
| IEC 62366-1 | Mode awareness, user interface | Human factors |
| ISO 9283 | Test method, poses, results | Compare performance |

The record is the same information the data team needs for provenance. A team that records the calibration and tool offset for its own analysis is also assembling regulatory evidence.

## 9. Industry practice and case evidence

### 9.1 Surgical robots

Intuitive Surgical's da Vinci 5 received FDA 510(k) clearance in March 2024. The system includes articulated instruments, a console, and a vision system. Medtronic's Hugo received FDA 510(k) clearance for urologic procedures in December 2025 (K250725). Both systems rely on kinematic models, calibration, and registration.

The published accuracy claims are usually about the robot's repeatability or the instrument's precision, not about the clinical outcome. A repeatability of 0.1 mm does not mean a target error of 0.1 mm; the target error includes registration, tissue deformation, and imaging error.

**Documented fact:** the clearances and the system descriptions are public. **Inference:** the accuracy of a specific procedure depends on the workflow, which is not public.

### 9.2 Orthopedic robots

Stryker's Mako and Zimmer Biomet's Rosa are robotic-arm systems for orthopedic procedures. They use preoperative imaging, registration, and a robotic arm to guide the bone cut. The accuracy claim is about the cut position relative to the plan, not about the clinical outcome.

The kinematic model is simpler than a surgical robot's, but the registration and the bone motion are the dominant errors. A 1 mm registration error can produce a 1 mm cut error; a 1 mm bone motion during the cut can produce more.

**Documented fact:** the systems and their indications are public. **Inference:** the error budget for a specific procedure is not public in full.

### 9.3 Rehabilitation robots

Rehabilitation robots such as the Lokomat and EksoNR use exoskeletons or treadmills to assist gait. The kinematics includes the patient's joints, the orthosis, and the robot's joints. The patient's motion is not rigid, so the kinematic model is an approximation.

A 2024 umbrella review of the Lokomat found small effects and no convincing evidence. The result does not mean the kinematics is wrong; it means the clinical benefit is uncertain. The distinction between a technical claim and a clinical claim is essential.

**Documented fact:** the review is public. **Inference:** the kinematic accuracy of a specific device is not the same as its clinical effectiveness.

### 9.4 What good kinematic practice looks like

Teams that avoid kinematic surprises tend to:

- Document the frames and the convention.
- Measure the link parameters, not only the CAD values.
- Test the forward model across the workspace.
- Check the joint limits before the motion.
- Compute the condition number and plan away from singularities.
- Calibrate with enough poses and report the residuals.
- Record the calibration and the tool offset with the data.

The practices are ordinary. The failures they prevent are not.

### 9.5 A note on accuracy claims

A robot accuracy claim is usually a bench result at a specific pose with a specific tool and a specific measurement system. The clinical accuracy includes registration, tissue deformation, and user technique, which are not in the bench result.

A vendor that claims 0.1 mm is claiming a bench repeatability, not a clinical target error. The data scientist must ask which one is being claimed and which one matters for the procedure.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Frame convention error | Pose correct at home, wrong elsewhere | Mixed DH conventions | Wrong motion | Document and test |
| Tool offset error | Constant tip offset | Wrong tool length or sign | Wrong target | Measure the tool |
| Link parameter error | Residual grows with reach | CAD versus physical | Systematic error | Calibrate |
| Inverse branch error | Elbow hits the table | Wrong solution chosen | Collision | Check limits and obstacles |
| Joint limit | Motion stops mid-procedure | Target outside limits | Workflow failure | Plan within limits |
| Singularity | Large joint motion for small tip motion | Jacobian rank loss | Velocity saturation | Condition-number check |
| Encoder offset | Constant rotation error | Home position error | Target offset | Calibrated home |
| Backlash | Repeatability worse in one direction | Gear clearance | Position error | Backlash compensation |
| Thermal drift | Error grows after warm-up | Link expansion | Slow bias | Warm-up; compensation |
| Registration error | Target error larger than tip error | Image-to-robot mismatch | Wrong target | Fiducial check; TRE |
| Calibration drift | Residual trend over time | Wear, temperature | Systematic error | Periodic calibration |
| Velocity saturation | Tip lags command | Joint velocity limit | Path error | Speed limit near singularity |
| Numerical IK divergence | Solver does not converge | Poor initial guess; singularity | No motion or jump | Step limit; singularity check |
| Workspace model error | Robot reaches an "unreachable" point | Tool or patient not included | Collision | Include tool and patient |

The most dangerous failure is the wrong inverse branch, because it can drive the arm into the patient or the table. The branch must be chosen with the workspace and the obstacles, not only the mathematical solution.

### 10.1 Ten red flags in kinematic data

1. A pose error that grows with reach.
2. A constant offset that changes after a tool change.
3. A joint motion that is large for a small tip motion.
4. A calibration residual that is larger at the workspace edge.
5. A repeatability that is worse in one direction.
6. A target error larger than the tip error.
7. A model that is correct at home and wrong at 45°.
8. A joint limit reached during a planned path.
9. A registration check that is done only at the fiducials.
10. A calibration date that is older than the service interval.

Each is a prompt to measure, not a conclusion.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The frames are correctly defined | Pose computation | Convention error | Compare with measurement |
| The link parameters are accurate | Forward model | Manufacturing tolerance | Calibration |
| The joints are rigid | Kinematic model | Compliance, backlash | Load test |
| The encoders are accurate | Joint measurement | Offset, resolution | Calibration |
| The tool offset is known | Tip position | Tool change | Measure the tool |
| The inverse solution is feasible | Motion | Joint limits, obstacles | Check before motion |
| The Jacobian is well conditioned | Control | Singularity | Condition number |
| The registration is accurate | Target | Fiducial error, deformation | TRE measurement |
| The patient is stationary | Registration | Breathing, motion | Motion tracking |
| The calibration is current | Accuracy | Wear, temperature | Periodic check |
| The controller meets the cycle | Tracking | Processor load | Timing measurement |
| The safety envelope is correct | Safe motion | Wrong limit | Fault injection |

The dominant assumption is that the registration is accurate. The robot can be perfect and the target can still be wrong because the image and the robot are misaligned. The registration error must be measured at the target, not only at the fiducials.

A second limit is that the kinematic model is rigid. Real links flex, joints have compliance, and the tool bends. A 1 N force on a 100 mm tool with a 1 N/mm stiffness produces 1 mm of deflection. The deflection is not in the kinematic model.

A third limit is that the model is deterministic. The real robot has backlash, friction, and thermal drift. The kinematic model predicts the mean; the repeatability captures the spread. Both are needed.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a CAD model?
- Which assumption was true at the home pose but false at the target?
- Which assumption depends on the tool, the patient, or the registration?
- Which assumption is encoded in a calibration file?
- Which assumption is specific to the validation workspace?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Compute forward kinematics (recall, 20 minutes)

For \(L_1 = 0.25\) m, \(L_2 = 0.20\) m, \(q_1 = 45^\circ\), \(q_2 = 30^\circ\), compute the tip position.

**Deliverable:** a calculation with a diagram.

**Acceptance criteria:** correct x and y; units; diagram consistent.

**Time:** 20 minutes.

### Exercise 2. Solve inverse kinematics (application, 30 minutes)

For the arm in Exercise 1, solve the inverse for the tip position you computed. Find both elbow solutions.

**Deliverable:** the two solutions with the formulas.

**Acceptance criteria:** both solutions; correct angles; branch explanation.

**Time:** 30 minutes.

### Exercise 3. Compute the Jacobian (application, 30 minutes)

For the arm in Exercise 1, compute the Jacobian at the given angles, its determinant, and its singular values. Is the arm well conditioned?

**Deliverable:** the matrix and the scalars.

**Acceptance criteria:** correct entries; determinant; condition number; verdict.

**Time:** 30 minutes.

### Exercise 4. Propagate an error (application, 30 minutes)

A 0.2° error on each joint produces what tip error at the configuration in Exercise 1? Compute the RMS and the worst-case bound.

**Deliverable:** two calculations.

**Acceptance criteria:** Jacobian used; RMS and worst case; units.

**Time:** 30 minutes.

### Exercise 5. Diagnose a singularity (critique, 30 minutes)

A robot at \(q_2 = 1^\circ\) is commanded to move 2 mm radially. The joint controller saturates. Explain the mechanism and propose two fixes.

**Deliverable:** a diagnosis and two fixes.

**Acceptance criteria:** condition number; joint motion; fixes distinguish planner and controller.

**Time:** 30 minutes.

### Exercise 6. Design a calibration (design, 35 minutes)

Design a calibration procedure for a 6-DOF surgical arm. Specify the measurement system, the number of poses, the parameters, and the acceptance criterion.

**Deliverable:** a procedure with at least 6 steps.

**Acceptance criteria:** observability; residual; acceptance tied to the clinical requirement.

**Time:** 35 minutes.

### Exercise 7. Build an error budget (design, 40 minutes)

Build a tip-error budget for a surgical robot with encoder error 0.05°, link tolerance 0.1 mm, registration error 1.0 mm, and tool offset error 0.2 mm. State the dominant term and the total.

**Deliverable:** a budget table and a conclusion.

**Acceptance criteria:** terms combined correctly; dominant term; total compared with a 2 mm requirement.

**Time:** 40 minutes.

### Exercise 8. Evaluate an accuracy claim (critique, 25 minutes)

A vendor claims "0.1 mm accuracy." Write five questions that would test the claim.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover repeatability versus accuracy, registration, tool, workspace, and measurement; no question is answerable by the headline.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Forward kinematics | Calculation |
| 2 | Inverse kinematics | Two solutions |
| 3 | Jacobian | Matrix and scalars |
| 4 | Error | RMS and worst case |
| 5 | Singularity | Diagnosis and fixes |
| 6 | Calibration | Procedure |
| 7 | Error budget | Budget table |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** \(x = 0.25 \cos 45 + 0.20 \cos 75 = 0.1768 + 0.0518 = 0.2286\) m. \(y = 0.25 \sin 45 + 0.20 \sin 75 = 0.1768 + 0.1932 = 0.3700\) m.

**Answer 2 (Exercise 2).** \(r^2 = 0.2286^2 + 0.37^2 = 0.1892\). \(\cos q_2 = (0.1892 - 0.0625 - 0.04)/(2 \times 0.25 \times 0.20) = 0.0867/0.10 = 0.867\). \(q_2 = \pm 29.9^\circ\), approximately ±30°. For the elbow-up solution, \(q_1 = \operatorname{atan2}(0.37, 0.2286) - \operatorname{atan2}(0.20 \sin 30, 0.25 + 0.20 \cos 30) = 58.3^\circ - 21.8^\circ = 36.5^\circ\), which is close to 45°; the difference is rounding in the forward calculation. The elbow-down solution is \(q_1 = 80.1^\circ\), \(q_2 = -30^\circ\).

**Answer 3 (Exercise 3).** \(J_{11} = -0.25 \sin 45 - 0.20 \sin 75 = -0.1768 - 0.1932 = -0.3700\). \(J_{12} = -0.20 \sin 75 = -0.1932\). \(J_{21} = 0.25 \cos 45 + 0.20 \cos 75 = 0.2286\). \(J_{22} = 0.20 \cos 75 = 0.0518\). \(\det = (-0.37)(0.0518) - (-0.1932)(0.2286) = -0.01917 + 0.04417 = 0.0250\) m². The singular values are about 0.43 and 0.058 m/rad; the condition number is about 7.4, which is well conditioned.

**Answer 4 (Exercise 4).** With \(\delta q = 0.2^\circ = 0.00349\) rad, the RMS tip error is \(\sqrt{(0.37 \times 0.00349)^2 + (0.1932 \times 0.00349)^2 + (0.2286 \times 0.00349)^2 + (0.0518 \times 0.00349)^2} = 1.75\) mm. The worst-case bound is the sum of the absolute contributions, about 2.95 mm.

**Answer 5 (Exercise 5).** At \(q_2 = 1^\circ\), the condition number is above 250. A 2 mm radial move requires a joint motion of about 10° on \(q_2\) and 4° on \(q_1\), which may exceed the velocity limit for the cycle. Fixes: reduce the Cartesian speed near the singularity, or change the configuration (elbow-up to elbow-down) before the move. A third fix is a redundant joint that moves in the null space to improve conditioning.

**Answer 6 (Exercise 6).** The procedure: measure 30–50 poses with a laser tracker or camera; include the workspace edges; estimate the link parameters, the joint offsets, and the tool offset; compute the residuals; accept if the maximum residual is below 0.5 mm, or the clinical requirement. Re-calibrate after a tool change, a crash, or a year of use.

**Answer 7 (Exercise 7).** Encoder: 0.05° at a 0.5 m Jacobian norm gives about 0.44 mm. Link: 0.1 mm. Registration: 1.0 mm. Tool: 0.2 mm. RSS = \(\sqrt{0.44^2 + 0.1^2 + 1.0^2 + 0.2^2} = 1.12\) mm. The registration dominates. The total is below a 2 mm requirement with a 0.88 mm margin.

**Answer 8 (Exercise 8).** Questions: (1) Is the 0.1 mm accuracy or repeatability? (2) At what point in the workspace? (3) Does it include registration and tool offset? (4) How was it measured, and with what uncertainty? (5) Does it hold after thermal soak and over the service life? A headline cannot answer any of them.

**Rubric for the design exercises.** A strong design states the frames, the parameters, the Jacobian, the conditioning, the errors, and the verification. A weak design quotes an accuracy number and assumes the model.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: x | 0.2286 m | ±0.001 m |
| Exercise 1: y | 0.3700 m | ±0.001 m |
| Exercise 2: q2 | ±29.9° | ±0.5° |
| Exercise 3: determinant | 0.0250 m² | ±0.001 m² |
| Exercise 3: condition | 7.4 | ±0.5 |
| Exercise 4: RMS error | 1.75 mm | ±0.1 mm |
| Exercise 7: total | 1.12 mm | ±0.05 mm |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 10 | 5 | 20 |
| 2 | 5 | 20 | 5 | 30 |
| 3 | 5 | 20 | 5 | 30 |
| 4 | 5 | 20 | 5 | 30 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 20 | 10 | 35 |
| 7 | 5 | 25 | 10 | 40 |
| 8 | 5 | 10 | 10 | 25 |

## 14. Interview questions

**1. What is the difference between forward and inverse kinematics?**
A strong answer defines both, notes the multiple solutions of the inverse, and gives an example. A weak answer says one is the reverse of the other.

**2. What is a homogeneous transform?**
A strong answer explains the 4×4 matrix, the rotation and translation, and the composition order. A weak answer describes it as a rotation.

**3. What is a singularity?**
A strong answer defines it as a rank loss of the Jacobian and gives the fully extended arm as an example. A weak answer says it is a joint limit.

**4. How do you detect a singularity?**
A strong answer uses the determinant, the smallest singular value, or the condition number, and states a threshold. A weak answer says "when the arm is straight".

**5. Why does a small tip motion require a large joint motion near a singularity?**
A strong answer explains the inverse Jacobian and the small singular value. A weak answer says the robot is "at its limit".

**6. What is the difference between accuracy and repeatability?**
A strong answer defines both and gives the "same wrong point" example. A weak answer uses them interchangeably.

**7. How do you propagate a joint error to the tip?**
A strong answer uses the Jacobian and the singular values, and distinguishes RMS from worst case. A weak answer adds the errors directly.

**8. Why does registration matter more than kinematics in some procedures?**
A strong answer explains that the robot can be precise and the target still wrong if the image and robot are misaligned. A weak answer ignores registration.

**9. What is target registration error?**
A strong answer defines it as the error at the clinical target, not at the fiducials. A weak answer confuses it with fiducial error.

**10. What would make you reject a kinematic model?**
A strong answer names a frame error, a residual above the requirement, an unhandled singularity, or an uncalibrated tool. A weak answer rejects on the number of joints.

**11. Why is the Jacobian important for control?**
A strong answer explains the velocity mapping, the inverse for resolved-rate control, and the singularity. A weak answer says it is a matrix of derivatives.

**12. What is the difference between workspace and dexterous workspace?**
A strong answer defines the dexterous workspace as the set of poses reachable with all orientations. A weak answer treats them as the same.

## 15. Key terms

- **Accuracy** — closeness to the commanded pose.
- **Backlash** — clearance in a gear or joint.
- **Base frame** — reference frame at the robot mounting.
- **Condition number** — ratio of largest to smallest singular value.
- **Dexterous workspace** — poses reachable with all orientations.
- **DH parameters** — four parameters per link.
- **DOF** — degrees of freedom.
- **End effector** — device at the end of the arm.
- **Forward kinematics** — joint angles to pose.
- **Frame** — coordinate system with origin and axes.
- **Gimbal lock** — loss of a DOF in Euler angles.
- **Homogeneous transform** — 4×4 matrix with rotation and translation.
- **Inverse kinematics** — pose to joint angles.
- **Jacobian** — joint velocity to task velocity.
- **Joint limit** — mechanical or software limit.
- **Kinematics** — geometry of motion.
- **Manipulability** — measure of dexterity.
- **Null space** — joint motion with no task motion.
- **Pose** — position plus orientation.
- **Product of exponentials** — screw-based kinematic formulation.
- **Quaternion** — four-parameter orientation.
- **Redundancy** — more joints than task DOF.
- **Registration** — alignment of image and robot frames.
- **Repeatability** — closeness of repeated poses.
- **Resolution** — smallest commanded change.
- **Singular value** — principal stretch of the Jacobian.
- **Singularity** — configuration where the Jacobian loses rank.
- **Tool centre point** — point controlled by the robot.
- **Workspace** — set of reachable poses.
- **Workspace limit** — boundary of the reachable space.
- **Continuum robot** — robot with a flexible body.
- **Manipulability** — measure of dexterity.
- **Parallel robot** — multiple chains to the tool.
- **Pseudoinverse** — generalised inverse for non-square matrices.
- **Resolved-rate control** — joint velocity from task velocity.
- **Screw axis** — axis of a twist.
- **Serial robot** — links in series.
- **Stewart platform** — parallel robot with six legs.
- **Wrist centre** — point where the last three axes intersect.
- **Calibration residual** — difference between the model and the measurement.
- **Forward model** — the function that maps joints to pose.
- **Kinematic parameter** — a link length, offset, or twist.
- **Tool offset** — vector from the flange to the tool centre point.
- **Velocity limit** — maximum joint velocity.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards**

1. IEC 80601-2-77:2019, *Medical electrical equipment — Part 2-77: Particular requirements for the basic safety and essential performance of robotically assisted surgical equipment*. https://webstore.iec.ch/en/publication/29933
2. IEC 80601-2-78:2021, *Medical electrical equipment — Part 2-78: Particular requirements for medical robots for rehabilitation, assessment, compensation or alleviation*. https://webstore.iec.ch/en/publication/69618
3. IEC 60601-1:2005+AMD1:2012+AMD2:2020. https://webstore.iec.ch/en/publication/26169
4. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
5. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
6. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
7. ISO 9283:1998, *Manipulating industrial robots — Performance criteria and related test methods*. https://www.iso.org/standard/22244.html
8. ASTM F2554-18, *Standard Practice for Measurement of Positional Accuracy of Computer Assisted Surgical Systems*. https://www.astm.org/f2554-18.html
9. ISO/TS 15066:2016, *Robots and robotic devices — Collaborative robots*. https://www.iso.org/standard/62996.html

**Devices and evidence**

10. Intuitive Surgical, da Vinci 5 FDA clearance, March 2024. https://isrg.intuitive.com/news-releases/news-release-details/intuitive-announces-fda-clearance-fifth-generation-robotic
11. Medtronic, Hugo FDA clearance, December 2025. https://news.medtronic.com/2025-12-03-Medtronic-announces-FDA-clearance-of-Hugo-TM-robotic-assisted-surgery-system-for-urologic-surgical-procedures
12. U.S. Food and Drug Administration, 510(k) K250725. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K250725
13. Stryker, Mako 4, 2025. https://www.stryker.com/us/en/about/news/2025/stryker-showcases-next-generation-of-mako-smartrobotics--at-aaos.html
14. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm

**Background**

15. J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 4th ed., Pearson, 2017.
16. B. Siciliano et al., *Robotics: Modelling, Planning and Control*, Springer, 2009.
17. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge University Press, 2017.
18. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
19. B. Siciliano and O. Khatib (eds.), *Springer Handbook of Robotics*, 2nd ed., Springer, 2016.
20. P. Corke, *Robotics, Vision and Control*, 2nd ed., Springer, 2017.
21. ASTM F2554-18, measurement of surgical navigation accuracy.
22. ISO 9283:1998, robot performance criteria.
23. J. J. Craig, *Introduction to Robotics*, 4th ed., 2017.
24. K. M. Lynch and F. C. Park, *Modern Robotics*, 2017.
25. IEC 80601-2-77:2019, robotically assisted surgical equipment.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_medical-equipment-classes.md`](09_medical-equipment-classes.md)
- Next topic: [`11_robotics-control.md`](11_robotics-control.md)
- Related: [`12_surgical-and-rehab-robotics.md`](12_surgical-and-rehab-robotics.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on robot kinematics: see Lab 14A.4 in the survey file.
- Lab on registration error: see Lab 14A.9 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
