# Surgical and Rehabilitation Robotics

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 12 of 14
> Estimated study time: 4 hours
> Prerequisites: 10, 11
> Version 2026.3

## 1. Why this topic exists

Surgical and rehabilitation robots are the two largest clinical robot categories, and they fail in different ways. A surgical robot must place a tool accurately relative to a patient who is moving, through a registration that maps an image to a body. A rehabilitation robot must assist a patient who is trying to move, without doing the work for them. Both must be safe, but the safety problem is different: one is a precision problem, the other an interaction problem.

The industry problem is that the two categories are often discussed as one. A team applies a surgical-robot accuracy mindset to a rehabilitation device and over-constrains the patient; another applies a rehabilitation-robot compliance mindset to a surgical device and loses precision. Three decisions depend on category literacy:

1. **What the accuracy requirement is.** A surgical robot's target error is measured in millimetres at the anatomy; a rehabilitation robot's assistance is measured in torque or effort, not position.
2. **Where the error enters.** In surgery, registration and tissue motion dominate; in rehabilitation, the patient's intent and the robot's compliance dominate.
3. **What evidence the claim needs.** A surgical accuracy claim needs a registration and target-error study; a rehabilitation claim needs a clinical outcome or an effort study.

This topic covers teleoperation, master-slave architecture, registration, haptics, latency, assist-as-needed control, and the safety envelope. It assumes kinematics and control from Topics 10 and 11. It does not cover every surgical specialty or every rehabilitation protocol.

A useful framing: a surgical robot is a measurement and motion system with a human in the loop; a rehabilitation robot is an interaction system with a patient in the loop. The human is part of the control loop in both, but the human's role is different.

## 2. Learning outcomes

By the end of this topic you can:

- Diagram a master-slave teleoperation system and identify the latency sources.
- Compute the target registration error from fiducial error and geometry.
- Explain the difference between fiducial registration error and target registration error.
- Compute the effect of force scaling and latency on teleoperation stability.
- Design an assist-as-needed controller and measure the patient's effort.
- Identify the safety envelope for a surgical and a rehabilitation robot.
- Critique a claim that a robot "improves outcomes" without an outcome study.
- Write a verification plan for a surgical or rehabilitation robot.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A surgical robot has a master (the surgeon's input), a slave (the robot), a registration (image to patient), and a feedback path (vision, force). A rehabilitation robot has a patient, an orthosis, a controller, and an intent estimate. Both have a safety monitor.

```text
 SURGICAL                                  REHABILITATION
 master ---> controller ---> slave         patient ---> intent ---> controller ---> orthosis
   ^             |              |             ^                                      |
   |             |              |             |                                      |
   +--- haptics -+              |             +------------ effort ------------------+
                                 v
 image ---> registration ---> patient
                                 |
                                 v
                           safety monitor
```

The model has four equations:

\[
\text{TRE} \approx \text{FLE} \times f(\text{geometry})
\]

\[
F_s = k_f F_m
\]

\[
\phi = -2\pi f T_d
\]

\[
\tau_a = K_a(\theta_d - \theta) - B_a \dot{\theta}
\]

The first relates target error to fiducial error; the second is force scaling; the third is the phase lag from latency; the fourth is the assistance law. The equations are simple; the assumptions behind them are not.

### 3.1 Teleoperation

A master-slave system measures the master's pose, sends it to the slave, and returns force or video. The loop has a delay from the master to the slave and back. The delay is the sum of the sensing, the network, the controller, and the actuator.

| Architecture | Master | Slave | Force feedback | Typical use |
|---|---|---|---|---|
| Position-position | Position | Position | No | Simple teleoperation |
| Position-force | Position | Position | Yes | Surgical robot |
| Force-position | Force | Force | Yes | Haptic master |
| Bilateral | Position and force | Position and force | Yes | Advanced teleoperation |

The architecture determines the stability. A bilateral loop with a long delay can oscillate; a position-position loop is more robust but transmits less information.

### 3.2 Registration

Registration aligns the preoperative image with the patient. The alignment uses fiducials, surface points, or intraoperative imaging. The error at the fiducials is the fiducial registration error (FRE); the error at the target is the target registration error (TRE).

| Method | Input | Strength | Weakness |
|---|---|---|---|
| Point-pair | Fiducial positions | Simple; accurate | Fiducial placement |
| Surface | Point cloud | No fiducials | Soft-tissue deformation |
| Intraoperative | Imaging | Accounts for motion | Radiation; workflow |
| Manual | Anatomical landmarks | No markers | Operator dependence |

The TRE is usually larger than the FRE, because the target is not at the fiducials. The geometry matters: fiducials that surround the target give a smaller TRE than fiducials that are all on one side.

### 3.3 Haptics and force scaling

Haptics return force to the master. The force can be scaled up to improve the surgeon's perception of small forces, or scaled down to reduce fatigue. The scaling changes the loop gain and the stability.

\[
F_s = k_f F_m
\]

A scale of 0.5 means the slave feels half the master's force; a scale of 2 means the slave feels twice. The scale must be disclosed and must not exceed the tissue's tolerance.

### 3.4 Latency

Latency is the delay from the master's motion to the slave's motion and back. It adds phase lag:

\[
\phi = -2\pi f T_d
\]

At 2 Hz and 100 ms, the lag is 72°, which consumes most of a 90° margin. The teleoperation loop becomes oscillatory. The surgeon perceives the delay as a lag or a "swimming" feeling.

The latency budget includes the network, the video, the controller, and the actuator. A 200 ms delay is common in telesurgery over a distance; a 20 ms delay is typical in a local system.

### 3.5 Assist-as-needed

A rehabilitation robot should assist the patient's movement without replacing it. The assistance law uses the error between the desired and actual motion:

\[
\tau_a = K_a(\theta_d - \theta) - B_a \dot{\theta}
\]

A high \(K_a\) moves the patient's limb regardless of effort; a low \(K_a\) assists only when the patient cannot complete the motion. The choice depends on the patient's ability and the therapy goal.

The patient's effort is the work done by the patient's muscles. It can be estimated from the force sensor, EMG, or a model. The robot's assistance should be tuned so that the patient's effort stays above a threshold.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Teleoperation | Remote control of a robot | Confused with autonomous | None |
| Master | Surgeon's input device | Confused with the controller | None |
| Slave | Robot that performs the motion | Confused with the tool | None |
| Bilateral | Force and position in both directions | Confused with unilateral | None |
| Haptics | Force feedback | Confused with vibration | N |
| Force scaling | Ratio of slave to master force | Confused with force limit | Unitless |
| Latency | Delay in the loop | Confused with bandwidth | ms |
| Registration | Alignment of image and patient | Confused with calibration | mm |
| FRE | Fiducial registration error | Confused with TRE | mm |
| TRE | Target registration error | Confused with FRE | mm |
| Fiducial | Marker with a known position | Confused with a landmark | None |
| Surface registration | Alignment using a point cloud | Confused with fiducial | mm |
| Intraoperative imaging | Imaging during surgery | Confused with preoperative | None |
| Soft-tissue deformation | Tissue motion after imaging | Ignored in registration | mm |
| Assist-as-needed | Assistance proportional to deficit | Confused with full assistance | None |
| Impedance control | Force-motion relationship | Confused with force control | N/m |
| Admittance control | Motion-force relationship | Confused with impedance | m/N |
| Effort | Patient's muscular work | Confused with robot torque | J |
| Exoskeleton | Wearable robot | Confused with a prosthesis | None |
| Prosthesis | Replacement limb | Confused with an orthosis | None |
| Orthosis | External support | Confused with a prosthesis | None |
| Gait phase | Stage of the walking cycle | Confused with joint angle | % |
| Intent detection | Estimation of patient's desired motion | Confused with trajectory | None |
| EMG | Electromyography | Confused with force | V |
| Safety envelope | Allowed workspace and force | Confused with a joint limit | m, N |
| Remote centre of motion | Fixed point at the port | Confused with a joint | m |
| Workspace | Reachable poses | Confused with the safety envelope | m |
| Autonomy level | Degree of independence | Confused with intelligence | 0–5 |
| Shared control | Human and robot share control | Confused with teleoperation | None |
| Learning curve | Improvement with experience | Confused with competence | Cases |

## 5. The mechanics, step by step

The procedure below designs and verifies a surgical or rehabilitation robot. Each step states its input, output, and rationale. The required edge cases are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the clinical task

**Input:** the procedure, the anatomy, the user, and the outcome.

**Output:** a task description with the accuracy, force, and time requirements.

**Rationale:** the task determines the architecture. A suturing task needs dexterity and force feedback; a bone-cut task needs stiffness and accuracy; a gait-training task needs compliance and repetition.

The task must name the clinical outcome, not only the motion. A robot that places a tool accurately but lengthens the procedure may not improve the outcome.

### Step 2. Choose the architecture

**Input:** the task and the user.

**Output:** a master-slave, hands-on, or autonomous architecture.

**Rationale:** a master-slave system keeps the surgeon in control; a hands-on system shares control; an autonomous system executes a plan. Most clinical robots are master-slave or hands-on; autonomy is limited.

The architecture determines the safety case. An autonomous system needs a complete hazard analysis for every motion; a teleoperated system needs the master, the slave, and the communication.

### Step 3. Design the master

**Input:** the surgeon's ergonomics and the task.

**Output:** a master with the degrees of freedom, the workspace, and the force feedback.

**Rationale:** the master must map the surgeon's hand motion to the slave without introducing tremor or fatigue. The workspace must be comfortable; the force feedback must be transparent enough to feel the tissue.

A master with a poor ergonomic design causes fatigue, which affects precision. The human factors are part of the robot.

### Step 4. Design the slave

**Input:** the anatomy and the task.

**Output:** a slave with the reach, the dexterity, and the force capability.

**Rationale:** the slave must reach the anatomy, fit through the port, and apply the required force. The remote centre of motion constrains the motion at the port; the wrist provides the dexterity inside.

The slave's kinematics must be calibrated; the tool must be sterile and interchangeable. A tool change changes the tool offset and the calibration.

### Step 5. Design the registration

**Input:** the image, the fiducials, and the patient.

**Output:** a transform from image to patient, and an error estimate.

**Rationale:** the registration is the bridge between the plan and the patient. The error at the target (TRE) depends on the fiducial error (FRE), the number and geometry of the fiducials, and the tissue deformation.

The registration must be verified at the target, not only at the fiducials. A check at a fiducial confirms the fit; a check at a landmark confirms the target.

### Step 6. Design the haptics

**Input:** the force sensor, the master, and the task.

**Output:** a force feedback law with scaling, filtering, and limits.

**Rationale:** the force feedback must be stable and transparent. A high gain makes the master oscillate; a low gain makes the tissue feel soft. The scaling must be disclosed and limited.

The haptic loop is a bilateral loop with delay. Its stability depends on the delay and the environment stiffness, as in Topic 11.

### Step 7. Budget the latency

**Input:** the master, the network, the slave, and the video.

**Output:** a latency budget and a stability margin.

**Rationale:** the latency is the sum of the stages. A 100 ms delay at 2 Hz consumes 72° of phase, which is most of the margin. The design must reduce the delay or the loop gain.

The video latency is separate from the force latency. A surgeon can adapt to a video delay of 100 ms but not to a force delay of the same magnitude.

### Step 8. Design the safety envelope

**Input:** the anatomy, the tool, and the failure modes.

**Output:** a workspace, a force limit, a speed limit, and a stop.

**Rationale:** the safety envelope must prevent motion outside the anatomy and force above the tissue tolerance. The limits must be enforced independently of the main controller.

The envelope must include the tool change, the registration error, and the patient motion. A workspace that is safe with a short tool may be unsafe with a long one.

### Step 9. Design the rehabilitation control

**Input:** the patient's ability, the joint, and the therapy goal.

**Output:** an assistance law with a gain and a threshold.

**Rationale:** the assistance must be proportional to the deficit, not to the error alone. A patient who is trying but weak needs assistance; a patient who is not trying needs encouragement, not assistance.

The gain must be tuned to the patient's ability and adjusted over the session. A fixed gain may be too high for one patient and too low for another.

### Step 10. Estimate the patient's intent

**Input:** the force, the EMG, or the motion.

**Output:** an intent estimate and a confidence.

**Rationale:** the robot must know what the patient is trying to do. A force sensor measures the interaction; an EMG measures the muscle activation; a motion sensor measures the movement. Each has a delay and a noise.

The intent estimate must be validated against the patient's actual motion. A wrong intent can cause the robot to resist the patient.

### Step 11. Verify the system

**Input:** the robot, the phantom, and the patient.

**Output:** a verification report with the accuracy, the force, the latency, and the safety.

**Rationale:** the verification must cover the bench, the phantom, the animal or cadaver, and the clinical pilot. Each stage answers a different question.

The verification must include the worst case: the maximum registration error, the maximum force, the maximum latency, and the patient's variability.

### Step 12. Document and monitor

**Input:** the design, the verification, and the clinical use.

**Output:** a configuration record and a post-market plan.

**Rationale:** the robot's behaviour depends on the calibration, the tool, the registration, and the software. The record must include all of them, and the monitoring must detect a drift.

A robot that is recalibrated in the field without a record is a robot that cannot be reproduced.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 8, 9 | Force or torque limit | Log | Loss of assistance |
| Common-mode rejection failure | 1 | Sensor issue | N/A | N/A |
| Electrode offset | 1 | Sensor issue | N/A | N/A |
| Clock jitter | 7 | Network or controller jitter | Trace | Haptic instability |
| Brown-out | 8, 12 | Controller reset | Reset log | Lost pose |
| Watchdog reset loop | 8, 12 | Controller hang | Reset counter | Procedure interrupted |
| Thermal throttling | 7 | Controller clock reduction | Temperature log | Late response |
| Firmware update failure | 12 | Calibration or gains lost | Version and checksum | Wrong behaviour |
| Calibration drift | 5, 12 | Tool or registration drift | Residual trend | Target error |

The surgical edge cases are dominated by registration and latency; the rehabilitation edge cases by assistance and intent. The processor edge cases enter through the controller and the safety monitor.

### 5.14 Registration methods

| Method | Fiducials | FRE | TRE | Use |
|---|---|---|---|---|
| Point-pair | 4–6 | 0.5–1.0 mm | 1–3 mm | Bone, skull |
| Surface | 100+ points | 0.5–1.5 mm | 1–3 mm | Spine, soft tissue |
| Intraoperative | Imaging | 0.5–1.0 mm | 1–2 mm | Moving tissue |
| Manual | Landmarks | 2–5 mm | 3–8 mm | Emergency |

The table is illustrative. The actual error depends on the fiducial placement, the imaging, and the tissue. A surface registration on soft tissue can have a larger TRE than a point-pair registration on bone.

### 5.15 Assistance levels

| Level | \(K_a\) | Patient effort | Robot effort | Use |
|---|---:|---|---|---|
| None | 0 | High | Low | Assessment |
| Light | 10 | High | Low | Early training |
| Moderate | 30 | Moderate | Moderate | Active training |
| Strong | 100 | Low | High | Severe impairment |
| Full | 300 | Very low | Very high | Passive movement |

The level must be chosen from the patient's ability and the therapy goal. A strong assistance may be appropriate for a patient with severe impairment and inappropriate for a patient who can move.

### 5.16 Latency budget

| Stage | Typical | Effect |
|---|---:|---|
| Master sensor | 1 ms | Phase lag |
| Master filter | 5 ms | Phase lag |
| Network | 20–200 ms | Phase lag |
| Slave controller | 1 ms | Phase lag |
| Slave actuator | 5 ms | Phase lag |
| Video | 50–200 ms | Visual lag |
| Force feedback | 5 ms | Haptic lag |

The force loop's total is 32–212 ms; the video loop's is 50–200 ms. The force loop is the stability-critical one; the video loop is the usability-critical one.

### 5.17 Master-slave control architectures

| Architecture | Master command | Slave control | Force feedback | Stability |
|---|---|---|---|---|
| Position-position | Position | Position | No | High |
| Position-force | Position | Force | Yes | Moderate |
| Force-position | Force | Position | Yes | Moderate |
| Bilateral | Position and force | Position and force | Yes | Low with delay |

A position-position system is the most stable because the slave does not feed force back into the master. It is also the least transparent: the surgeon feels no tissue force.

A bilateral system is transparent but sensitive to delay. The loop gain must be reduced as the delay rises, which reduces transparency. The trade is inherent.

### 5.18 Remote centre of motion

A surgical instrument inserted through a port must rotate about the port, not about a point inside the body. The remote centre of motion is a mechanical or software constraint that keeps the instrument at the port.

| Method | Mechanism | Strength | Weakness |
|---|---|---|---|
| Mechanical | Remote centre mechanism | Passive; precise | Complex; limited workspace |
| Software | Constrained inverse kinematics | Flexible | Depends on model |
| Hybrid | Mechanical plus software | Robust | Complexity |

A software remote centre uses the inverse kinematics to keep the port point fixed. The constraint reduces the workspace and can create singularities near the boundary.

### 5.19 Registration validation

| Check | What it measures | Acceptance |
|---|---|---|
| Fiducial residual | FRE | < 1 mm |
| Landmark check | TRE at a known point | < 2 mm |
| Target check | TRE at the target | < 2 mm |
| Repeat registration | Variability | < 0.5 mm |
| Tissue motion | Change after registration | < 1 mm |

The fiducial residual is necessary but not sufficient. The landmark and target checks are the evidence that the registration is valid where it matters.

### 5.20 Rehabilitation intent sensing

| Sensor | Measures | Delay | Limitation |
|---|---|---|---|
| Force/torque | Interaction force | 1–10 ms | Includes robot dynamics |
| EMG | Muscle activation | 20–100 ms | Noise; electrode placement |
| Joint angle | Motion | 1–5 ms | Does not measure intent |
| EEG | Cortical activity | 100–300 ms | Noise; non-specific |

The intent estimate must be validated against the patient's actual movement. A force sensor measures what the patient is doing; an EMG measures what the patient is trying to do. The two can disagree.

### 5.21 Ten review questions

1. What is the clinical task and the outcome?
2. What is the architecture, and why?
3. What is the registration method and the TRE at the target?
4. What is the latency, and what is the phase margin?
5. What is the force scaling, and is it disclosed?
6. What is the safety envelope, and how is it enforced?
7. What is the assistance law, and how is the patient's effort measured?
8. What is the intent sensor, and what is its delay?
9. What is the verification ladder?
10. What is recorded so that a change can be detected?

A review that answers all ten has covered the system. A review that answers only the second and the third has covered the robot, not the patient.

### 5.22 Safety envelope

| Limit | Surgical | Rehabilitation | Enforcement |
|---|---|---|---|
| Workspace | Anatomy plus margin | Joint range plus margin | Independent monitor |
| Force | Tissue tolerance | Patient tolerance | Force limit; stop |
| Speed | Procedure limit | Gait or motion limit | Velocity limit |
| Power | Actuator limit | Actuator limit | Safety relay |
| Registration | Target tolerance | Not applicable | Check before use |

The envelope must be enforced independently of the main controller. A controller that enforces its own limit can fail; a separate monitor with a hardware stop is the defence.

### 5.23 Shared control

Shared control divides the motion between the surgeon and the robot. The robot can constrain the motion to a safe region, smooth the tremor, or scale the motion, while the surgeon retains the high-level control.

| Mode | Robot role | Surgeon role | Risk |
|---|---|---|---|
| Teleoperation | Follows master | Direct control | Delay |
| Shared control | Constrains or assists | Direct control | Mode confusion |
| Hands-on | Assists the surgeon's hand | Direct control | Force |
| Supervised autonomy | Executes a sub-task | supervises | Autonomy error |

The mode must be visible, and the transition must be explicit. A surgeon who does not know whether the robot is constraining or following cannot control it.

### 5.27 Shared-control transparency

A shared controller must make its intervention visible. The force feedback, the display, or an audio cue must indicate when the robot is constraining the motion. A silent constraint feels like a malfunction.

| Cue | Strength | Weakness |
|---|---|---|
| Force feedback | Direct | Requires haptics |
| Visual overlay | Clear | Requires attention |
| Audio | Attention-grabbing | Annoying |
| Vibration | Simple | Ambiguous |

The cue must be tested with users. A cue that the designer understands may be invisible to the surgeon.

### 5.28 Training scenarios

The training must include the normal case, the registration failure, the latency, the force limit, and the mode change. A surgeon who has only practiced the normal case cannot handle the failure.

| Scenario | Skill | Frequency |
|---|---|---|
| Normal procedure | Technique | Every case |
| Registration error | Recovery | Rare |
| Force limit | Stop and reassess | Occasional |
| Mode change | Awareness | Every case |
| Communication loss | Safe stop | Rare |

The scenarios must be practiced on a simulator or a phantom. A classroom lecture is not training.

### 5.29 What to record per case

| Field | Why |
|---|---|
| Procedure and site | Context |
| Robot and tool | Configuration |
| Registration method and TRE | Accuracy |
| Master and scaling | Control |
| Latency | Stability |
| Assistance level | Therapy |
| Safety events | Safety |
| Outcome | Evidence |

The record is the case's provenance. A case without it cannot be compared with another.

### 5.30 Verification ladder

| Stage | Question | Evidence |
|---|---|---|
| Bench | Does the robot move correctly? | Kinematics, force |
| Phantom | Does it hit the target? | TRE, force |
| Cadaver | Does it work in anatomy? | Workflow, accuracy |
| Human factors | Can the surgeon use it? | Usability |
| Pilot | Is it safe in patients? | Adverse events |
| Pivotal | Is it effective? | Outcomes |

Each stage answers a different question. A bench test cannot answer a clinical question.

### 5.24 Autonomy levels

| Level | Description | Example | Evidence |
|---|---|---|---|
| 0 | No autonomy | Teleoperation | Master-slave |
| 1 | Assistance | Tremor filtering | Bench |
| 2 | Task autonomy | Suturing sub-task | Preclinical |
| 3 | Conditional autonomy | Supervised task | Clinical |
| 4 | High autonomy | Unsupervised in a defined setting | Clinical |
| 5 | Full autonomy | Not established | None |

Most clinical robots are level 0–2. A claim of level 3 or above requires explicit scope, supervision, and fail-safe design.

### 5.25 Training and credentialing

A robot that is technically precise but poorly used is not safe. The training must cover the master, the slave, the registration, the modes, the safety stop, and the failure response.

The credentialing must be based on proficiency, not only case counts. A surgeon who has done 100 cases with poor technique may be less proficient than one who has done 20 with good technique. The learning curve varies by procedure and platform.

### 5.26 Ten safety questions for a clinical robot

1. What is the workspace, and how is it enforced?
2. What is the force limit, and how fast is the stop?
3. What happens if the registration is wrong?
4. What happens if the network delays?
5. What happens if the master loses power?
6. What happens if the slave loses power?
7. What happens if the tool is changed?
8. What happens if the patient moves?
9. What happens if the surgeon makes an error?
10. What is recorded, and who reviews it?

A review that cannot answer all ten has not reviewed the safety case.

## 6. Worked example

### 6.1 The registration case

A surgical robot uses four fiducials on a bone. The fiducials are at (0, 0), (100, 0), (0, 80), and (100, 80) mm in the image. The true transform is a 20° rotation and a (10, 5) mm translation. The target is at (60, 30) mm.

The measured fiducials have a 0.5 mm standard deviation in each coordinate. A point-pair registration estimates the rotation and translation by aligning the centroids and the principal directions.

The fiducial registration error is:

\[
\text{FRE} = \sqrt{\frac{1}{N}\sum_i \|R p_i + t - q_i\|^2}
\]

The target registration error is the error at the target, not at the fiducials:

\[
\text{TRE} = \|R p_{\text{target}} + t - q_{\text{target}}\|
\]

A Monte Carlo simulation with 1,000 trials gives a mean FRE of 0.574 mm and a mean TRE of 0.278 mm. The maximum FRE is 1.183 mm, and the maximum TRE is 0.777 mm.

The TRE is smaller than the FRE because the target is near the centroid of the fiducials. A target outside the fiducial cloud would have a larger TRE.

### 6.2 The TRE geometry

The TRE depends on the target's position relative to the fiducials. For a target at the centroid, the error is small; for a target outside the cloud, the error grows with the distance.

| Target | Distance from centroid | Mean TRE |
|---|---:|---:|
| (50, 40) | 0 mm | 0.21 mm |
| (60, 30) | 22 mm | 0.28 mm |
| (150, 100) | 125 mm | 1.10 mm |
| (200, 150) | 200 mm | 1.90 mm |

The table is illustrative; the values depend on the fiducial geometry. The principle is general: the TRE grows with the distance from the fiducial cloud, and the direction matters.

A surgeon who registers at the fiducials and operates 100 mm away has a larger error than the fiducial check suggests. The registration check must be at the target.

### 6.3 The latency case

A teleoperation loop has a 100 ms delay. At 2 Hz, the phase lag is:

\[
\phi = -2\pi \times 2 \times 0.1 = -1.257\ \text{rad} = -72^\circ
\]

A loop with a 90° phase margin on the bench has an 18° margin with the delay. The loop is close to instability, and the surgeon feels a "swimming" master.

The delay is reduced to 50 ms. The phase lag becomes 36°, and the margin is 54°. The master feels firmer. The improvement is not linear: halving the delay doubles the margin.

### 6.4 The rehabilitation case

A rehabilitation robot assists a patient's elbow flexion. The desired trajectory is a 0.5 Hz sine with a 0.5 rad amplitude. The patient's torque is proportional to the error with a gain of 20 Nm/rad and a damping of 1 Nm·s/rad. The robot's assistance is proportional to the error with \(K_a\) and a damping of 2 Nm·s/rad.

A simulation over 4 s gives:

| \(K_a\) (Nm/rad) | Patient work (J) | Robot work (J) | Patient share |
|---:|---:|---:|---:|
| 0 | 8.86 | 7.06 | 0.56 |
| 20 | 3.34 | 1.09 | 0.75 |
| 100 | 2.01 | 4.49 | 0.31 |

The patient's absolute work falls from 8.86 J to 2.01 J as the assistance rises. The patient's share rises at \(K_a = 20\) because the robot's damping does less work, then falls at \(K_a = 100\) because the robot takes over.

The result shows that the assistance level changes the patient's effort, but not monotonically in share. The absolute work is the better measure of the patient's effort; the share is a measure of the division of labour.

### 6.5 Diagnostic 1: the registration check passes at the fiducials but fails at the target

The registration check at the four fiducials shows a 0.6 mm residual, which is within the acceptance criterion. The first incision is 3 mm off the plan. The target is 100 mm from the fiducial cloud.

The diagnostic is to check the registration at an anatomical landmark near the target, not only at the fiducials. The landmark check shows a 2.5 mm error, consistent with the TRE at that distance.

The design flaw is the acceptance criterion, not the registration algorithm. The criterion was specified at the fiducials, where the error is small. The fix is to specify the criterion at the target and to add a landmark check.

### 6.6 Diagnostic 2: the assistance is too high

A patient who could move the arm before training shows less effort after the robot is set to \(K_a = 100\). The therapist notices that the patient is relaxing and letting the robot move the arm.

The diagnostic is to measure the patient's EMG or force during the session. The EMG amplitude falls when the assistance rises, which confirms that the patient is doing less.

The design flaw is the assistance level, not the robot. The fix is to reduce \(K_a\) and to use an assist-as-needed law that adjusts to the patient's effort. The therapy goal is the patient's learning, not the robot's motion.

### 6.7 Vary one assumption: the fiducial noise doubles

If the fiducial noise rises from 0.5 mm to 1.0 mm, the FRE and TRE roughly double. The mean FRE becomes about 1.15 mm, and the mean TRE about 0.56 mm. The maximum TRE becomes about 1.55 mm.

The consequence is that a 2 mm target tolerance is still met on average but not at the maximum. The registration must be repeated or the fiducials must be improved.

### 6.8 What the example demonstrates

The arithmetic closes: mean FRE 0.574 mm, mean TRE 0.278 mm, maximum TRE 0.777 mm, 72° phase lag at 100 ms and 2 Hz, and patient work falling from 8.86 J to 2.01 J as assistance rises. The two diagnostics reveal different flaws: one is an acceptance criterion at the wrong location, the other is an assistance level that is too high.

The general method is to measure the error at the target, budget the latency, and measure the patient's effort, not only the robot's motion.

### 6.9 Force scaling

A master force of 2 N with a scaling of 0.5 commands a slave force of 1 N. With a scaling of 2, it commands 4 N. The scaling changes the loop gain: a higher scaling makes the slave more sensitive but less stable.

| Master force | Scale 0.5 | Scale 1.0 | Scale 2.0 |
|---:|---:|---:|---:|
| 0.5 N | 0.25 N | 0.5 N | 1.0 N |
| 1.0 N | 0.5 N | 1.0 N | 2.0 N |
| 2.0 N | 1.0 N | 2.0 N | 4.0 N |
| 5.0 N | 2.5 N | 5.0 N | 10.0 N |

A scale of 2 with a 5 N master force commands 10 N, which may exceed the tissue tolerance. The scale must be limited by the safety envelope, not only by the surgeon's preference.

### 6.10 Verification matrix

| Test | Condition | Metric | Acceptance |
|---|---|---|---|
| Registration | Phantom | TRE at target | < 2 mm |
| Registration repeat | Five registrations | Variability | < 0.5 mm |
| Latency | Master to slave | Round trip | < 100 ms |
| Haptic stability | Contact | Oscillation | None |
| Force scaling | Known weight | Slave force | ±10% |
| Safety stop | Maximum speed | Stop time | < 100 ms |
| Workspace | Boundary | Stop | No overshoot |
| Assistance | Patient simulator | Patient work | Above threshold |

The matrix covers the surgical and rehabilitation requirements. A system that passes only the bench tests is not verified for the clinical use.

### 6.11 TRE versus distance

| Distance from centroid | Mean TRE | Maximum TRE |
|---:|---:|---:|
| 0 mm | 0.21 mm | 0.55 mm |
| 50 mm | 0.45 mm | 1.10 mm |
| 100 mm | 0.85 mm | 1.90 mm |
| 150 mm | 1.30 mm | 2.80 mm |

The table shows the growth of the TRE with distance. A 2 mm tolerance is met at 100 mm on average but not at the maximum. The registration must be checked at the target.

### 6.12 Latency budget

| Stage | Delay | Cumulative |
|---|---:|---:|
| Master sensor | 2 ms | 2 ms |
| Master filter | 5 ms | 7 ms |
| Network | 80 ms | 87 ms |
| Slave controller | 1 ms | 88 ms |
| Slave actuator | 5 ms | 93 ms |
| Force sensor | 2 ms | 95 ms |
| Force filter | 5 ms | 100 ms |

The round trip is 100 ms, consistent with the 72° phase lag at 2 Hz. A 50 ms network would halve the delay and double the margin.

### 6.13 Shared control example

A shared-control robot constrains the tool to a 2 mm corridor around the planned path. The surgeon's command is projected onto the corridor. If the surgeon commands a point 5 mm off the path, the robot moves to the nearest point in the corridor, 3 mm away from the command.

The projection reduces the surgeon's freedom but prevents a large deviation. The corridor width is a clinical decision: too narrow and the surgeon cannot adapt; too wide and the constraint is ineffective.

- The corridor must be visible.
- The force feedback must reflect the constraint.
- The surgeon must be able to override.
- The override must be logged.
- The corridor must adapt to the anatomy.

The projection must be transparent. A surgeon who feels the robot resisting without knowing why may fight it. The force feedback or the display must show the constraint.

### 6.14 Registration with a distant target

If the target is 150 mm from the centroid, the mean TRE rises to about 1.30 mm and the maximum to about 2.80 mm. The fiducial check still passes at 0.6 mm, but the target is outside the 2 mm tolerance at the maximum.

The example shows that the acceptance criterion must be specified at the target, with the target's distance and direction. A criterion at the fiducials is not a criterion at the anatomy.

### 6.15 Assistance adjustment

If the patient's work falls below 50% of the unassisted work, the assistance is too high. The controller reduces \(K_a\) by 10% and re-measures. The adjustment is repeated until the patient's work is above the threshold or the minimum assistance is reached.

The adjustment must be bounded. A controller that reduces the assistance without a limit can leave a weak patient without support.

### 6.16 A note on the simulation

The rehabilitation model uses a proportional patient torque and a viscous robot. A real patient has nonlinear dynamics, fatigue, and spasticity. The simulation shows the direction of the effect, not the clinical magnitude.
## 7. Code, query, or calculation

The script reproduces the registration Monte Carlo and the rehabilitation assistance simulation in Section 6. It uses only the standard library: a 2D point-pair registration, a Gaussian fiducial error, and a simple patient-robot model.

**Assumptions.** Four fiducials at the corners of a 100 mm × 80 mm rectangle plus a centre point, a true 20° rotation and (10, 5) mm translation, 0.5 mm fiducial noise, 1,000 trials, a 0.5 Hz sine target, a patient torque gain of 20 Nm/rad and damping of 1 Nm·s/rad, and robot damping of 2 Nm·s/rad.

```python
import math, random
random.seed(7)
P = [(0,0), (100,0), (0,80), (100,80), (50,40)]
ang = math.radians(20)
c, s = math.cos(ang), math.sin(ang)
T = (10, 5)
Q = [(c*x - s*y + T[0], s*x + c*y + T[1]) for x, y in P]
target = (60, 30)

def reg(P, Q):
    n = len(P)
    px = sum(p[0] for p in P)/n; py = sum(p[1] for p in P)/n
    qx = sum(q[0] for q in Q)/n; qy = sum(q[1] for q in Q)/n
    num = sum((p[0]-px)*(q[1]-qy) - (p[1]-py)*(q[0]-qx) for p, q in zip(P, Q))
    den = sum((p[0]-px)*(q[0]-qx) + (p[1]-py)*(q[1]-qy) for p, q in zip(P, Q))
    th = math.atan2(num, den)
    cc, ss = math.cos(th), math.sin(th)
    return th, qx - (cc*px - ss*py), qy - (ss*px + cc*py)

fres, tres = [], []
for _ in range(1000):
    Qn = [(q[0] + random.gauss(0, 0.5), q[1] + random.gauss(0, 0.5)) for q in Q]
    th, tx, ty = reg(P, Qn)
    cc, ss = math.cos(th), math.sin(th)
    res = [math.hypot(cc*p[0] - ss*p[1] + tx - q[0], ss*p[0] + cc*p[1] + ty - q[1])
           for p, q in zip(P, Qn)]
    fres.append(math.sqrt(sum(r*r for r in res)/len(res)))
    ex = cc*target[0] - ss*target[1] + tx - (c*target[0] - s*target[1] + T[0])
    ey = ss*target[0] + cc*target[1] + ty - (s*target[0] + c*target[1] + T[1])
    tres.append(math.hypot(ex, ey))
print(f"FRE mean {sum(fres)/len(fres):.3f} mm max {max(fres):.3f}")
print(f"TRE mean {sum(tres)/len(tres):.3f} mm max {max(tres):.3f}")

J, b, Gp, Bp, Ba, dt, T_end = 0.05, 0.5, 20.0, 1.0, 2.0, 1e-3, 4.0
for Ka in (0, 20, 100):
    th, om, wp, wa = 0.0, 0.0, 0.0, 0.0
    for k in range(int(T_end/dt)):
        t = k*dt
        td = 0.5*math.sin(2*math.pi*0.5*t)
        tp = Gp*(td - th) - Bp*om
        ta = Ka*(td - th) - Ba*om
        alpha = (tp + ta - b*om)/J
        om += alpha*dt; th += om*dt
        wp += abs(tp*om)*dt; wa += abs(ta*om)*dt
    print(f"Ka {Ka:3d}: patient {wp:.3f} J robot {wa:.3f} J share {wp/(wp+wa):.2f}")
```

**Expected output.**

```text
FRE mean 0.574 mm max 1.183
TRE mean 0.278 mm max 0.777
Ka   0: patient 8.864 J robot 7.060 J share 0.56
Ka  20: patient 3.338 J robot 1.092 J share 0.75
Ka 100: patient 2.008 J robot 4.492 J share 0.31
```

The output shows the registration error at the fiducials and at the target. The mean FRE is 0.574 mm, and the mean TRE is 0.278 mm, because the target is near the fiducial centroid. The maximum values are about twice the means, which is the tail that a single registration can experience.

The rehabilitation simulation shows the patient's work falling from 8.86 J to 2.01 J as the assistance rises. The patient's share rises at \(K_a = 20\) and falls at \(K_a = 100\). The absolute work is the better measure of effort; the share measures the division of labour.

**A note on the registration model.** The model uses a 2D rigid transform and a Gaussian fiducial error. A real registration is 3D, with anisotropic error and tissue deformation. The 2D model is adequate for the arithmetic and the geometry lesson, not for a clinical accuracy claim.

## 8. Data, standards, and regulatory context

Surgical and rehabilitation robots are governed by their particular standards, the general safety standard, and the risk management standard.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 80601-2-77:2019 | Robotically assisted surgical equipment | Particular standard | Accuracy, control, safety, master-slave |
| IEC 80601-2-78:2021 | Rehabilitation and assistive robots | Particular standard | Patient contact, assistance, safe stop |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment | Consensus standard | Electrical safety, isolation |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis, force limits |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Control and registration software |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | Master, modes, alarms |
| ASTM F2554-18 | Surgical navigation measurement | Consensus standard | Registration accuracy |
| ISO/TS 15066:2016 | Collaborative robots | Informative | Force and pressure limits |

### 8.1 What the particular standards require

IEC 80601-2-77 requires the manufacturer to specify the robot's accuracy, the workspace, the control, the emergency stop, and the interaction with the patient and the team. It does not prescribe a registration method or a haptic gain; it requires evidence that the system is safe and effective.

IEC 80601-2-78 adds the patient contact, the assistance behaviour, and the safe stop for rehabilitation. A rehabilitation robot that assists a patient's movement must limit the force and the speed and must detect the patient's resistance.

### 8.2 Registration standards

ASTM F2554-18 provides methods for measuring the positional accuracy of computer-assisted surgical systems. The measurement includes the registration error at the target, not only at the fiducials. The standard is informative for the data scientist who must interpret an accuracy claim.

A registration claim that reports only the FRE is incomplete. The TRE at the target is the clinically relevant number.

### 8.3 What the standards do not cover

The standards do not specify:

- The registration algorithm.
- The fiducial placement.
- The haptic gain.
- The assistance law.
- The training programme.

Those are design decisions that the manufacturer must justify. A robot can meet every clause and still have a 3 mm target error because the registration was poor.

### 8.4 The data implications

The registration, the tool, the latency, the assistance level, and the firmware version are part of the measurement. A change in any of them changes the behaviour. The record must include them.

### 8.5 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 80601-2-77 | Accuracy, registration, force, stop time | Reproduce the claim |
| IEC 80601-2-78 | Assistance, patient contact, stop | Reproduce the rehab claim |
| ISO 14971 | Hazard analysis for registration and force | Link failure to harm |
| IEC 62304 | Control and registration software version | Reproduce behaviour |
| IEC 62366-1 | Master, modes, cues | Human factors |
| ASTM F2554 | Registration measurement | Accuracy evidence |

The record is the same information the data team needs for provenance. A team that records the registration and tool for its own analysis is also assembling regulatory evidence.

### 8.6 A note on the clinical evaluation

A surgical robot's clinical evaluation must compare the robot with the standard technique, not only report a case series. A case series shows that the robot can be used; a comparative study shows whether it is better. The endpoint must be clinical, not only technical.

A rehabilitation robot's evaluation must measure the patient's function, not only the robot's assistance. The two are not the same.

## 9. Industry practice and case evidence

### 9.1 Surgical robots

Intuitive Surgical's da Vinci 5 received FDA 510(k) clearance in March 2024. The system includes articulated instruments, a console, and a vision system. Medtronic's Hugo received FDA 510(k) clearance for urologic procedures in December 2025 (K250725). Both rely on master-slave control, calibration, and a safety envelope.

The published claims are about the system's capabilities, not about clinical outcomes. A clearance is not an outcome study; the evidence for the outcome is separate.

**Documented fact:** the clearances are public. **Inference:** the registration and control details are not public.

### 9.2 Orthopedic robots

Stryker's Mako and Zimmer Biomet's Rosa use preoperative imaging, registration, and a robotic arm. The accuracy claim is about the cut position relative to the plan. The registration and the bone motion are the dominant errors.

**Documented fact:** the systems and indications are public. **Inference:** the error budget is not public in full.

### 9.3 Rehabilitation robots

ReWalk 7 received FDA 510(k) clearance in March 2025 (K241822) for spinal cord injury ambulation. EksoNR and the Lokomat are established rehabilitation robots. A 2024 umbrella review of the Lokomat found small effects and no convincing evidence.

The evidence for rehabilitation robotics is mixed. The technical performance can be good and the clinical benefit uncertain. The distinction between a technical claim and a clinical claim is essential.

**Documented fact:** the clearances and the review are public. **Inference:** the relationship between the control and the outcome is device- and protocol-specific.

### 9.4 What good practice looks like

Teams that avoid surgical and rehabilitation surprises tend to:

- Measure the registration at the target, not only at the fiducials.
- Budget the latency and measure the phase margin.
- Limit the force scaling and the assistance.
- Measure the patient's effort, not only the robot's motion.
- Design the safety envelope independently of the controller.
- Train and credential on proficiency, not only case counts.
- Record the registration, tool, gains, and firmware.

The practices are ordinary. The failures they prevent are not.

### 9.5 A note on the learning curve

A robot's clinical benefit depends on the surgeon's learning curve. A new platform may have worse outcomes during the first cases and better outcomes after the curve. A study that compares a new robot with an established one without accounting for the curve may attribute the learning effect to the technology.

The curve must be measured, not assumed. The proficiency criteria, the case mix, and the supervision are part of the evidence.

### 9.6 A note on the console

The console is part of the robot. The display, the seating, the armrests, and the pedals affect the surgeon's fatigue and precision. A console that is uncomfortable after 30 minutes changes the outcome.

The console must be evaluated with users in a realistic session. A short usability test does not reveal the fatigue.

### 9.7 A note on the team

A surgical robot is used by a team, not only a surgeon. The bedside assistant, the scrub nurse, and the anesthesiologist interact with the robot. A robot that disrupts the team's communication can be less safe than a simpler tool.

The team training and the communication protocol are part of the safety case.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Registration error | Fiducial check passes; target is off | TRE grows with distance | Wrong target | Target check; better fiducials |
| Tissue motion | Registration drifts during surgery | Breathing, deformation | Target error | Intraoperative imaging |
| Latency | Master feels "swimming" | Phase lag | Instability | Reduce delay; reduce gain |
| Haptic instability | Oscillation in contact | Loop gain too high | Uncontrolled force | Reduce gain; filter |
| Force scaling error | Slave force differs from command | Wrong scale | Tissue damage | Calibrate; limit |
| Assistance too high | Patient relaxes | Robot does the work | No learning | Assist-as-needed |
| Assistance too low | Patient cannot complete | Robot does not help | Frustration | Increase gain |
| Intent error | Robot resists the patient | Wrong estimate | Unsafe interaction | Multi-sensor; validation |
| Workspace breach | Tool outside the anatomy | Limit failure | Injury | Independent monitor |
| Stop failure | Force peak above limit | Slow stop | Injury | Faster stop; lower speed |
| Tool change error | Offset wrong | Tool not calibrated | Target error | Tool identification |
| Mode confusion | Surgeon unsure | UI or state machine | Wrong motion | Mode awareness |
| Training gap | Poor technique | Inexperience | Complication | Credentialing |
| Cybersecurity | Unauthorized update | Weak boot | Compromise | Secure boot; SBOM |

The most dangerous failure is the registration error that passes the fiducial check. The check confirms the fit, not the target. The target check is the defence.

### 10.1 Ten red flags in clinical robot data

1. A registration check only at the fiducials.
2. A target error that grows with distance.
3. A master that feels different between units.
4. A force peak above the limit.
5. A stop time longer than the budget.
6. A patient whose effort falls over the session.
7. A mode change without a cue.
8. A tool change without a calibration check.
9. A latency that varies with the network.
10. An outcome claim without a comparator.

Each is a prompt to measure, not a conclusion.

### 10.2 What to do when the target is off

1. Check the registration at a landmark near the target.
2. Re-register with fiducials closer to the target.
3. Check the tool offset.
4. Check the patient motion.
5. If the error persists, stop and reassess.

A target error is a measurement problem before it is a robot problem. The landmark check localises it.

### 10.3 What to do when the patient's effort falls

1. Measure the patient's work or EMG.
2. Compare with the unassisted baseline.
3. Reduce the assistance if the effort is below the threshold.
4. Re-measure after the adjustment.
5. Record the level and the response.

A fall in effort is a therapy signal, not a robot fault. The response is a change in assistance, not a repair.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The fiducials are accurately placed | Registration | Placement error | Imaging; check |
| The target is near the fiducials | TRE estimate | Target far away | Target check |
| The tissue is rigid | Registration | Deformation | Intraoperative imaging |
| The latency is known | Stability | Network variation | Measure |
| The force sensor is calibrated | Haptics | Drift | Calibration |
| The patient's intent is observable | Assistance | Weak or noisy signal | Multi-sensor |
| The assistance is appropriate | Therapy | Patient variability | Adjust per session |
| The safety envelope is correct | Safety | Wrong limit | Fault injection |
| The tool is calibrated | Accuracy | Tool change | Tool identification |
| The training is adequate | Usability | Inexperience | Credentialing |
| The registration is current | Accuracy | Time, motion | Re-check |
| The controller is verified | Evidence | Missing test | Verification matrix |

The dominant assumption is that the target is near the fiducials. It often is not: a surgeon registers at the skull and operates at the spine, or registers at the surface and operates deep. The TRE must be measured at the target.

A second limit is that the tissue is not rigid. The registration is a snapshot; the tissue moves. The error is not in the algorithm; it is in the model.

A third limit is that the clinical benefit is not the technical performance. A robot can be precise and not improve the outcome. The evidence must be clinical.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a model?
- Which assumption was true on the phantom but false in the patient?
- Which assumption depends on the fiducials, the tool, or the tissue?
- Which assumption is encoded in a registration or a gain?
- Which assumption is specific to the validation population?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

### 11.2 A note on the evidence ladder

The evidence ladder runs from bench to phantom to animal or cadaver to human factors to clinical pilot to pivotal study. A robot that passes the bench and the phantom has not been validated in a patient. The clinical evidence is a separate step.

## 12. Exercises

### Exercise 1. Compute FRE and TRE (recall, 25 minutes)

For four fiducials with a 0.5 mm error, compute the expected FRE and explain why the TRE can be smaller or larger.

**Deliverable:** a calculation and an explanation.

**Acceptance criteria:** FRE formula; geometry; TRE dependence.

**Time:** 25 minutes.

### Exercise 2. Compute a phase lag (application, 20 minutes)

A teleoperation loop has a 150 ms delay. Compute the phase lag at 1 Hz and at 3 Hz. Which is more likely to oscillate?

**Deliverable:** two calculations and a verdict.

**Acceptance criteria:** formula; units; verdict.

**Time:** 20 minutes.

### Exercise 3. Compute a force (application, 20 minutes)

A master force of 3 N is scaled by 1.5. What is the slave force? If the tissue limit is 5 N, is the scaling safe?

**Deliverable:** a calculation and a verdict.

**Acceptance criteria:** scaling; comparison; verdict.

**Time:** 20 minutes.

### Exercise 4. Compute patient work (application, 30 minutes)

A patient generates 10 J of work without assistance and 4 J with assistance. What is the patient's share if the robot does 6 J? Is the assistance appropriate for a therapy goal of 50% patient effort?

**Deliverable:** a calculation and a judgement.

**Acceptance criteria:** share; comparison with the goal; judgement.

**Time:** 30 minutes.

### Exercise 5. Diagnose a registration failure (critique, 30 minutes)

A registration check at the fiducials shows 0.5 mm, but the target is 3 mm off. Explain the mechanism and propose two fixes.

**Deliverable:** a diagnosis and two fixes.

**Acceptance criteria:** TRE geometry; target check; fixes.

**Time:** 30 minutes.

### Exercise 6. Design a safety envelope (design, 35 minutes)

Design a safety envelope for a surgical robot with a 2 mm target tolerance, a 5 N force limit, and a 100 mm workspace. Include the enforcement and the stop.

**Deliverable:** a table with limits and enforcement.

**Acceptance criteria:** independent monitor; stop time; fault injection.

**Time:** 35 minutes.

### Exercise 7. Design an assistance law (design, 35 minutes)

Design an assist-as-needed law for a patient with moderate impairment. Specify the gain, the intent sensor, the effort threshold, and the adjustment.

**Deliverable:** a control law and a tuning plan.

**Acceptance criteria:** gain; sensor; threshold; adjustment.

**Time:** 35 minutes.

### Exercise 8. Evaluate an outcome claim (critique, 25 minutes)

A vendor claims its robot "improves surgical outcomes." Write five questions that would test the claim.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover study design, comparator, endpoints, population, and follow-up; no question is answerable by the clearance.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Registration | Calculation |
| 2 | Latency | Phase lag |
| 3 | Force scaling | Calculation |
| 4 | Patient work | Share and judgement |
| 5 | Diagnosis | Mechanism and fixes |
| 6 | Safety envelope | Table |
| 7 | Assistance | Control law |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** For four fiducials with independent 0.5 mm errors, the expected FRE is about 0.5 mm, and the TRE depends on the geometry. If the target is at the centroid, the TRE is smaller than the FRE; if the target is outside the fiducial cloud, the TRE is larger. The simulation gives a mean FRE of 0.574 mm and a mean TRE of 0.278 mm for a target near the centroid.

**Answer 2 (Exercise 2).** At 1 Hz, the lag is \(2\pi \times 1 \times 0.15 = 0.94\) rad, or 54°. At 3 Hz, the lag is 2.83 rad, or 162°. The 3 Hz case is more likely to oscillate because the phase lag exceeds 90°.

**Answer 3 (Exercise 3).** The slave force is \(3 \times 1.5 = 4.5\) N, below the 5 N limit. The scaling is safe for this master force but would exceed the limit at a master force above 3.33 N. The limit must be enforced independently of the scale.

**Answer 4 (Exercise 4).** The total work is 4 + 6 = 10 J, and the patient's share is 40%. The goal is 50%, so the assistance is slightly too high. Reducing the robot's work to 4 J would give a 50% share.

**Answer 5 (Exercise 5).** The FRE is small at the fiducials, but the target is far from the cloud, so the TRE is larger. Fixes: add fiducials near the target, or check the registration at an anatomical landmark near the target. A third fix is intraoperative imaging.

**Answer 6 (Exercise 6).** The envelope: workspace 100 mm around the planned path, force 5 N, speed 20 mm/s, stop within 100 ms. The monitor is independent, with a force sensor and a hardware relay. The fault injection tests a sensor loss, a controller hang, and a power loss.

**Answer 7 (Exercise 7).** The law: \(\tau_a = K_a(\theta_d-\theta) - B_a\dot{\theta}\), with \(K_a = 30\) Nm/rad for moderate impairment, a force sensor for intent, an effort threshold of 50% of the unassisted work, and an adjustment of ±10% per session. The gain is reduced if the patient's effort falls below the threshold.

**Answer 8 (Exercise 8).** Questions: (1) What is the study design and comparator? (2) What is the primary endpoint? (3) What is the population and the follow-up? (4) Is the outcome surgeon- or patient-reported? (5) Is the result independent of the learning curve? A clearance does not answer any of them.

**Rubric for the design exercises.** A strong design quantifies the registration, the latency, the force, the assistance, and the safety. A weak design names a robot and assumes the outcome.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 2: lag at 1 Hz | 54° | ±2° |
| Exercise 2: lag at 3 Hz | 162° | ±5° |
| Exercise 3: slave force | 4.5 N | ±0.1 N |
| Exercise 4: patient share | 40% | ±2% |
| Section 6: mean FRE | 0.574 mm | ±0.02 mm |
| Section 6: mean TRE | 0.278 mm | ±0.02 mm |
| Section 6: patient work at Ka=100 | 2.01 J | ±0.05 J |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 15 | 5 | 25 |
| 2 | 5 | 10 | 5 | 20 |
| 3 | 5 | 10 | 5 | 20 |
| 4 | 5 | 15 | 10 | 30 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 20 | 10 | 35 |
| 7 | 5 | 20 | 10 | 35 |
| 8 | 5 | 10 | 10 | 25 |

## 14. Interview questions

**1. What is the difference between FRE and TRE?**
A strong answer defines both and explains the geometry. A weak answer uses them interchangeably.

**2. Why is the TRE larger than the FRE?**
A strong answer explains the distance from the fiducial cloud and the direction. A weak answer says "because the target is not a fiducial".

**3. How does latency affect teleoperation?**
A strong answer explains phase lag and instability. A weak answer says it feels laggy.

**4. What is force scaling?**
A strong answer explains the ratio and the safety limit. A weak answer says it makes the force stronger.

**5. What is assist-as-needed?**
A strong answer explains the assistance proportional to the deficit and the patient's effort. A weak answer says the robot helps.

**6. How do you measure the patient's effort?**
A strong answer uses force, EMG, or work, and validates against the motion. A weak answer uses the robot's torque alone.

**7. What is the safety envelope?**
A strong answer names the workspace, force, speed, and power, and the independent monitor. A weak answer says "the limits".

**8. Why is registration checked at the target?**
A strong answer explains that the fiducial check does not measure the target error. A weak answer says "to be safe".

**9. What is the evidence for a surgical robot's benefit?**
A strong answer distinguishes clearance, technical performance, and clinical outcome. A weak answer cites the clearance.

**10. What would make you reject a clinical robot?**
A strong answer names a registration that is not checked at the target, an unbounded force, an unstable haptic loop, or no outcome evidence. A weak answer rejects on price.

**11. How does tissue motion affect registration?**
A strong answer explains that the registration is a snapshot and the tissue moves, so the TRE grows. A weak answer says the registration is "updated".

**12. Why is patient effort a therapy variable?**
A strong answer explains that too much assistance reduces the patient's learning, so the effort must be measured and kept above a threshold. A weak answer says the robot should help as much as possible.

**13. What is the role of the target check?**
A strong answer explains that it measures the TRE where it matters, not the FRE. A weak answer says it is a second fiducial check.

**14. How does tool change affect accuracy?**
A strong answer explains that the tool offset changes and must be recalibrated. A weak answer says the tool is the same.

## 15. Key terms

- **Assist-as-needed** — assistance proportional to the deficit.
- **Bilateral** — force and position in both directions.
- **Effort** — patient's muscular work.
- **Exoskeleton** — wearable robot.
- **Fiducial** — marker with a known position.
- **Force scaling** — ratio of slave to master force.
- **FRE** — fiducial registration error.
- **Gait phase** — stage of the walking cycle.
- **Haptics** — force feedback.
- **Intent detection** — estimation of the patient's desired motion.
- **Latency** — delay in the loop.
- **Master** — surgeon's input device.
- **Orthosis** — external support.
- **Prosthesis** — replacement limb.
- **Registration** — alignment of image and patient.
- **Remote centre of motion** — fixed point at the port.
- **Shared control** — human and robot share control.
- **Slave** — robot that performs the motion.
- **Soft-tissue deformation** — tissue motion after imaging.
- **Surface registration** — alignment using a point cloud.
- **Teleoperation** — remote control of a robot.
- **TRE** — target registration error.
- **Autonomy level** — degree of independence.
- **Learning curve** — improvement with experience.
- **Safety envelope** — allowed workspace and force.
- **Sterile field** — the area around the patient that must remain sterile.
- **Tool offset** — vector from the flange to the tool tip.
- **Workspace** — reachable poses.
- **Credentialing** — authorization to use the robot.
- **Mode awareness** — user knowledge of the active mode.
- **Bilateral control** — force and position in both directions.
- **Fiducial cloud** — the set of fiducials used for registration.
- **Haptic transparency** — how faithfully the master reproduces the slave force.
- **Master-slave** — teleoperation architecture.
- **Patient share** — fraction of work done by the patient.
- **Proficiency** — demonstrated skill, not case count.
- **Remote centre** — fixed point at the port.
- **Sterile drape** — barrier that keeps the robot outside the sterile field.
- **Target check** — registration check at an anatomical landmark.
- **Case record** — per-case provenance.
- **Comparative study** — robot versus standard technique.
- **Console** — surgeon's workstation.
- **Fatigue** — performance loss over a session.
- **Landmark** — anatomical reference point.
- **Proficiency** — demonstrated skill.
- **Re-registration** — repeating the registration.
- **Sterile field** — area around the patient.
- **Tool change** — replacement of the instrument.
- **Work envelope** — allowed motion range.
- **Assistance level** — gain of the assistive torque.
- **Baseline** — unassisted performance.
- **Effort threshold** — minimum patient work.
- **Team training** — training of the whole OR team.
- **Usability** — ease of safe use.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards**

1. IEC 80601-2-77:2019, *Robotically assisted surgical equipment*. https://webstore.iec.ch/en/publication/29933
2. IEC 80601-2-78:2021, *Medical robots for rehabilitation, assessment, compensation or alleviation*. https://webstore.iec.ch/en/publication/69618
3. IEC 60601-1:2005+AMD1:2012+AMD2:2020. https://webstore.iec.ch/en/publication/26169
4. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
5. IEC 62304:2006+AMD1:2015, *Medical device software*. https://webstore.iec.ch/en/publication/22794
6. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
7. ASTM F2554-18, *Measurement of positional accuracy of computer assisted surgical systems*. https://www.astm.org/f2554-18.html
8. ISO/TS 15066:2016, *Collaborative robots*. https://www.iso.org/standard/62996.html

**Devices and evidence**

9. Intuitive Surgical, da Vinci 5, March 2024. https://isrg.intuitive.com/news-releases/news-release-details/intuitive-announces-fda-clearance-fifth-generation-robotic
10. Medtronic, Hugo, December 2025. https://news.medtronic.com/2025-12-03-Medtronic-announces-FDA-clearance-of-Hugo-TM-robotic-assisted-surgery-system-for-urologic-surgical-procedures
11. U.S. Food and Drug Administration, 510(k) K250725. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K250725
12. Lifeward/ReWalk 7, March 2025. https://ir.rewalk.com/news-releases/news-release-details/fda-issues-clearance-rewalk-7-exoskeleton
13. U.S. Food and Drug Administration, K241822. https://www.accessdata.fda.gov/cdrh_docs/pdf24/K241822.pdf
14. Lokomat umbrella review, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11546060/
15. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm

**Background**

16. J. J. Craig, *Introduction to Robotics*, 4th ed., Pearson, 2017.
17. B. Siciliano et al., *Robotics: Modelling, Planning and Control*, Springer, 2009.
18. J. G. Webster (ed.), *Medical Instrumentation*, 4th ed., Wiley, 2009.
19. IEEE, *Haptics* and bilateral teleoperation literature. https://ieeexplore.ieee.org/
20. ISO 14971:2019, risk management.
21. ASTM F2554-18, registration accuracy measurement.
22. IEC 80601-2-77:2019, surgical robot requirements.
23. IEC 80601-2-78:2021, rehabilitation robot requirements.
24. J. J. Craig, *Introduction to Robotics*, 4th ed.
25. B. Siciliano et al., *Robotics: Modelling, Planning and Control*.
26. U.S. Food and Drug Administration, *Medical Device Development Tools*.
27. IEC 80601-2-77:2019, surgical robot standard.
28. IEC 80601-2-78:2021, rehabilitation robot standard.
29. ASTM F2554-18, registration measurement.
30. ISO 14971:2019, risk management.
31. IEC 62304:2006+AMD1:2015, software lifecycle.
32. IEC 62366-1:2015+AMD1:2020, usability.
33. U.S. Food and Drug Administration, *Human Factors and Usability Engineering* guidance.
34. ISO 14971:2019, risk management.
35. IEC 62304:2006+AMD1:2015, software lifecycle.
36. B. Siciliano et al., *Robotics: Modelling, Planning and Control*.
37. J. J. Craig, *Introduction to Robotics*, 4th ed.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`11_robotics-control.md`](11_robotics-control.md)
- Next topic: [`13_standards-and-emc.md`](13_standards-and-emc.md)
- Related: [`10_robotics-kinematics.md`](10_robotics-kinematics.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on registration error: see Lab 14A.9 in the survey file.
- Lab on safety monitor: see Lab 14A.5 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
- **Adverse event** — harm during the procedure.
- **Comparator** — standard technique in a study.
- **Endpoint** — outcome measured in a study.
- **Follow-up** — observation after the procedure.
- **Learning phase** — early cases on a new platform.
- **Outcome** — clinical result.
- **Pilot study** — small clinical study.
- **Pivotal study** — definitive clinical study.
- **Proficiency criterion** — standard for competence.
- **Randomization** — assignment to comparator or robot.
- **Registry** — observational data collection.
- **Safety event** — near miss or adverse event.
- **Simulator** — training environment.
- **Supervision** — oversight by an experienced surgeon.
- **Team communication** — coordination during the procedure.
