# Module 14A — Medical Hardware, Sensors, Semiconductors, and Robotics

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 36 hours (14 lecture, 4 case clinic, 18 lab)
> **Track fit:** MedTech, Sensors & Imaging core; elective for Health Systems and Biopharma
> **Prerequisites:** Modules 01, 01A, 11, 14
> **Tools:** Python, NumPy/SciPy, a circuit simulator (LTspice or Falstad), ROS 2 and Gazebo or MuJoCo, an edge-AI toolkit, a 510(k) summary
> **Industry question:** "What physical, electronic, and control assumptions turn a sensor reading into a clinical measurement — and where does the data science actually sit?"

---

## 1. Why hardware matters to a data scientist

Most biomedical data science courses begin at a CSV file. Industry data begin at a sensor, pass through an analog front end, an ADC, firmware, a processor, a wireless link, a gateway, and a clinical workflow. Each stage can distort, delay, or fabricate the number you model.

A data scientist who does not understand hardware will:

- Treat a device reading as ground truth.
- Attribute a model failure to the algorithm when the sensor was saturated.
- Train on a signal that was filtered differently at deployment.
- Ignore latency that makes a “real-time” alert useless.
- Miss that a robot's coordinate frame was misregistered.
- Accept a vendor's “AI-enabled” claim without asking what the silicon actually does.
- Underestimate electrical-safety, cybersecurity, and sterilization constraints.

This module builds hardware literacy without turning you into an electrical engineer. It focuses on the concepts that change data decisions: sensor physics, signal chains, embedded constraints, semiconductor choices, robotics, safety, and regulation.

## 2. Learning outcomes

By the end of this module, you can:

- Diagram a medical device signal chain from sensor to clinical output.
- Explain the physics, assumptions, and failure modes of common medical sensors.
- Select and critique analog front ends, ADCs, processors, and wireless modules for a data product.
- Explain edge-AI constraints: latency, power, memory, quantization, and thermal limits.
- Describe firmware, secure boot, update, and supply-chain risks.
- Explain the kinematics, control, registration, and safety of medical robots.
- Distinguish surgical, rehabilitation, prosthetic, capsule, imaging, and service robots.
- Map a hardware product to IEC 60601, IEC 62304, ISO 14971, IEC 80601-2-77/-2-78, and FDA/EU requirements.
- Evaluate hardware-related assumptions in a dataset or model.
- Design a hardware-in-the-loop validation plan.
- Communicate device behavior to clinicians, engineers, and regulators.

## 3. The medical device signal chain

```
Patient → Sensor/electrode → Analog front end → ADC → Processor/firmware
        → Algorithm → Output/actuator → Clinician/patient → Outcome
```

### 3.1 Stage-by-stage questions

| Stage | Question | Failure mode |
|---|---|---|
| Patient | What is being measured, where, and under what conditions? | Wrong site, poor contact, movement |
| Sensor | What physical quantity is transduced, and with what sensitivity? | Drift, saturation, cross-sensitivity |
| AFE | Is the signal amplified and filtered without adding noise or instability? | Clipping, common-mode noise, impedance mismatch |
| ADC | Is the sampling rate and resolution adequate? | Aliasing, quantization, timing jitter |
| Processor | Can it run the algorithm within latency and power budgets? | Dropped frames, thermal throttling |
| Firmware | Is the code correct, versioned, and secure? | Bug, stale calibration, vulnerable update |
| Algorithm | What assumptions does it make about signal quality and population? | Distribution shift, false confidence |
| Output | Is the display/alarm/actuator safe and usable? | Alarm fatigue, ambiguous display, unsafe action |
| Workflow | Does the output reach the right person at the right time? | Delayed review, no owner, alert ignored |
| Outcome | Does the device improve the decision? | No clinical utility |

### 3.2 Data-science interface

The data scientist usually receives:

- Raw waveforms.
- Device-computed values.
- Feature streams.
- Alarms/events.
- Logs and metadata.
- Images.
- Robot kinematics and video.

Each has different provenance and error structure. Ask for the rawest data the device can legally provide, plus the device-computed values, and the firmware/algorithm version.

## 4. Sensors

### 4.1 Biopotential electrodes

| Type | How it works | Strengths | Weaknesses |
|---|---|---|---|
| Ag/AgCl hydrogel | Electrochemical half-cell | Stable, low noise, clinical standard | Dries out, skin prep, single-use |
| Dry electrode | Conductive polymer/metal | Reusable, wearable | Higher contact impedance, motion |
| Capacitive | Insulated electrode | No gel, long wear | Sensitive to motion and gap |
| Textile | Conductive fabric | Comfortable | Variable contact, washing |

Used for ECG, EEG, EMG, EOG, and bioimpedance.

**Assumptions:** stable contact, adequate skin preparation, low motion, known electrode placement, and unchanged hardware. **Failure modes:** baseline wander, motion artifact, electrode offset, 50/60 Hz interference, and poor common-mode rejection.

### 4.2 Optical sensors: PPG and SpO₂

- LEDs emit light at selected wavelengths.
- Photodiodes measure transmitted or reflected light.
- Pulsatile arterial blood modulates absorption.
- PPG waveform yields heart rate, rhythm, perfusion, and respiratory modulation.
- SpO₂ uses the ratio of red and infrared absorption, calibrated against arterial blood gas.

**Assumptions:** adequate perfusion, stable contact, low motion, known skin pigmentation, no abnormal hemoglobin, and a calibrated device.

**Failure modes:** motion artifact, ambient light, low perfusion, venous pulsation, skin-tone bias, nail polish, and dyshemoglobins (carboxyhemoglobin, methemoglobin).

### 4.3 Pressure and flow

| Sensor | Principle | Examples | Issues |
|---|---|---|---|
| Strain gauge | Resistance changes with strain | Arterial line, infusion pump | Temperature drift, hysteresis |
| Capacitive MEMS | Capacitance changes with diaphragm deflection | Blood pressure, respiratory pressure | Offset, humidity |
| Piezoresistive MEMS | Resistance change | Invasive pressure, ventilators | Nonlinearity, thermal drift |
| Differential pressure | Flow through a resistance | Ventilator flow | Condensation, obstruction |
| Ultrasonic | Transit-time or Doppler | Blood flow, ventilation | Coupling, alignment |
| Thermal | Heat transport | Anemometry, flow | Gas composition, humidity |

**Assumptions:** known tubing compliance, no leaks, correct zeroing, and stable temperature. A ventilator flow sensor can drift silently and alter tidal-volume delivery.

### 4.4 Temperature

- **Thermistor:** resistance changes with temperature; common NTC 10 kΩ.
- **RTD:** platinum resistance; stable and accurate.
- **Thermocouple:** voltage from dissimilar metals; wide range.
- **Infrared:** tympanic or forehead; emissivity and site dependent.

**Assumptions:** site, equilibration time, perfusion, and ambient conditions. **Failure modes:** probe displacement, poor contact, and fever-reducing drugs.

### 4.5 Motion and position

- Accelerometer: linear acceleration.
- Gyroscope: angular velocity.
- Magnetometer: magnetic field.
- IMU: sensor fusion.
- Encoders: joint angle.
- Force/torque sensors: interaction forces.

**Use cases:** gait, falls, sleep, tremor, rehabilitation, surgical instrument tracking.

**Assumptions:** known placement, calibration, and no magnetic disturbance. **Failure modes:** drift, saturation, non-wear, and orientation ambiguity.

### 4.6 Chemical and biochemical sensors

| Sensor | Analyte | Principle | Key issue |
|---|---|---|---|
| Electrochemical CGM | Glucose | Glucose oxidase + electrode | Lag, drift, interference, calibration |
| Lactate sensor | Lactate | Enzymatic | Fouling, temperature |
| ISFET | pH/ions | Field-effect | Drift, encapsulation |
| Blood gas analyzer | pH, pCO₂, pO₂ | Electrochemical/optical | Calibration, sample handling |
| Lab-on-chip | Nucleic acids, proteins | Microfluidics + optical/electrical | Contamination, assay variability |
| Sequencing | DNA/RNA | Synthesis/optical or nanopore | Base-calling error, cost, run variability |

**Assumptions:** sample integrity, calibration, temperature, and interference profile. CGM lag and compression lows are classic data traps.

### 4.7 Ultrasound and imaging detectors

- **Ultrasound:** piezoelectric transducer or CMUT; pulse-echo; beamforming.
- **CT:** scintillator + photodiode; X-ray attenuation; Hounsfield units.
- **PET:** scintillator + photomultiplier/SiPM; annihilation photons.
- **MRI:** RF coils and gradients; not a simple sensor; safety dominated by magnetic field and heating.
- **Endoscopy:** CMOS image sensor; illumination; sterilization.
- **Pathology:** whole-slide scanner; optics and stitching.

**Assumptions:** calibration, geometry, acquisition protocol, and image quality. **Failure modes:** motion, metal artifact, beam hardening, and scanner drift.

### 4.8 Sensor selection checklist

1. What physical quantity is being measured?
2. What is the required range, resolution, and accuracy?
3. What is the sampling rate and bandwidth?
4. What is the environment (temperature, humidity, motion, EMI, sterilization)?
5. What is the power budget?
6. What is the size and biocompatibility requirement?
7. What is the calibration and service model?
8. What is the failure mode, and is it safe?
9. What is the regulatory precedent?
10. What is the lifecycle and supply-chain risk?

### 4.9 Performance examples that change assumptions

| Measurement | Evidence | Why it matters |
|---|---|---|
| SpO₂ by skin tone | A 2024 meta-analysis reported pooled A<sub>rms</sub> of 3.96% (light), 4.71% (medium), and 4.15% (dark) skin, with mean bias 0.70%, 0.27%, and 1.27%; FDA's January 2025 draft guidance proposes stricter diverse-pigment validation and non-disparity criteria | A device can meet a headline accuracy spec and still fail a subgroup |
| Wrist PPG by posture | SNR fell from ~18.6 dB supine to ~13.7 dB sitting and ~9.0 dB standing in one 2025 study; skin-tone effects depend on adaptive LED current | Motion and posture are part of the measurement model |
| CGM accuracy | Published MARDs cluster around 9–14%; lag, compression lows, temperature, hematocrit, and interferents (e.g., acetaminophen) vary by sensor | A CGM value is a delayed, calibrated estimate, not blood glucose |
| Dry electrodes | Dry ECG/EEG contact impedance is much higher and more motion-sensitive than wet Ag/AgCl; validation cohorts remain small | Contact impedance is a first-order signal-quality variable |
| Ultrasound-on-chip | Butterfly iQ3, cleared January 2024, uses a CMUT-on-ASIC architecture; CMUTs offer bandwidth and integration but face output-pressure and bias-voltage challenges | Probe architecture determines image quality and portability |
| Photon-counting CT | The Siemens Naeotom Alpha.Peak/Pro/Prime class received FDA clearance in February 2025 (K243523); the original photon-counting Naeotom Alpha was cleared earlier, in 2021. CZT/CdTe detectors provide energy discrimination but face charge trapping and calibration issues | Detector physics changes noise, dose, and reconstruction assumptions |

**Rule:** ask for subgroup and condition-specific performance, not a single accuracy number.

## 5. Semiconductors and embedded systems

### 5.1 Chip classes

| Class | Role | Examples | Data-science relevance |
|---|---|---|---|
| AFE | Amplify/filter sensor signal | TI ADS129x, ADI ADAS1000, MAX30001 | Noise, gain, bandwidth, channel count |
| ADC | Convert analog to digital | SAR, delta-sigma | Resolution, sample rate, aliasing |
| MCU | Embedded control | STM32, NXP, Renesas, Microchip | Firmware, latency, power |
| DSP | Signal processing | TI C5000/C6000 | Filtering, feature extraction |
| FPGA | Reconfigurable logic | AMD/Xilinx, Intel/Altera | Low-latency pipelines, prototyping |
| SoC | CPU + GPU/NPU + peripherals | NXP i.MX, Qualcomm, NVIDIA Jetson | Edge AI, connectivity, OS |
| GPU/NPU | Parallel acceleration | NVIDIA, Google Coral, Arm Ethos | Inference throughput, power |
| ASIC | Custom silicon | Mixed-signal ASIC | Ultra-low power, high volume |
| PMIC | Power management | TI, ADI, Maxim | Battery life, noise, safety |
| Wireless | BLE, Wi-Fi, NFC, UWB | Nordic, TI, Qualcomm | Latency, pairing, security |
| Isolation | Patient isolation | Digital isolators | Electrical safety |
| Memory | Code/data storage | Flash, DRAM, MRAM | Retention, wear, security |

These are representative classes, not endorsements or proof of suitability for a specific regulated device.

### 5.2 Analog front end

Key specifications:

- Input-referred noise.
- Common-mode rejection ratio (CMRR).
- Input impedance.
- Gain and bandwidth.
- Channel count.
- Power consumption.
- Electrostatic discharge protection.
- Defibrillation protection.
- Lead-off detection.

**Assumption:** the AFE's input range and filter settings match the sensor and the clinical signal. A clipped ECG cannot be recovered by a better model.

### 5.3 ADC and sampling

- **Resolution:** bits; effective number of bits (ENOB) matters more than marketing bits.
- **Sample rate:** must exceed twice the highest frequency of interest.
- **Anti-alias filter:** must precede the ADC.
- **Jitter:** timing error degrades high-frequency signals.
- **Quantization noise:** lower with more resolution.

**Clinical example:** a 10-second ECG sampled at 250 Hz has 2,500 samples per lead. If the device downsamples to 100 Hz before storage, high-frequency QRS content is lost.

### 5.4 Edge AI and TinyML

For the full compression and target-hardware workflow — distillation, PTQ/QAT, pruning, runtimes, p99 latency, energy, and PCCP change control — see Module 14B.

Constraints:

- Model size and memory.
- Inference latency.
- Power and battery.
- Thermal limits.
- Numerical precision.
- Toolchain maturity.
- Regulatory traceability.

Techniques:

- Quantization (int8, mixed precision).
- Pruning.
- Distillation.
- Efficient architectures.
- Operator fusion.
- Hardware accelerators.

**Rule:** measure accuracy after quantization on the target device, not only in the training environment.

### 5.5 Firmware and lifecycle

- Bare-metal vs RTOS.
- Bootloader and secure boot.
- Signed firmware updates.
- Watchdog timers.
- Fail-safe states.
- Logging and audit.
- Configuration management.
- SBOM and vulnerability monitoring.

**Assumption:** the deployed firmware is the version that was validated. If the device auto-updates, that assumption can break.

### 5.6 Electrical safety and EMC

- IEC 60601-1: general medical electrical equipment safety.
- Patient leakage current.
- Means of patient protection (MOPP) and means of operator protection (MOOP).
- Defibrillation protection.
- Electromagnetic emissions and immunity.
- Battery safety.
- MRI compatibility.

**Data implication:** electrical noise and safety limits can force filtering or sampling choices that change the signal you analyze.

**EMC in practice (IEC 60601-1-2).** Test categories include conducted and radiated emissions, electrostatic discharge, radiated RF immunity, electrical fast transients, surges, conducted RF, magnetic fields, and voltage dips. The applicable limits depend on the environment (professional healthcare, home healthcare, special). Electrosurgery, MRI, diathermy, cellular, Wi-Fi, and nearby radios are not abstract risks: they can saturate an AFE, corrupt a waveform, or reset a processor. Pre-compliance testing is cheaper than a late redesign, and an EMC fix can change the filter or sampling choices that your model depends on.

### 5.7 Materials, sterilization, and packaging

- Biocompatibility: ISO 10993.
- Sterilization: autoclave, ethylene oxide, gamma, e-beam, hydrogen peroxide.
- Hermeticity and ingress protection.
- Corrosion and fatigue.
- Adhesive and coating stability.
- Shelf life.

**Assumption:** the sensor's performance survives sterilization and aging. A single-use sensor may drift after reprocessing.

### 5.8 Supply chain

- Fabless vs IDM vs foundry.
- Wafer node and specialty process.
- Packaging and test.
- Long lifecycle support (10+ years).
- Obsolescence and last-time buys.
- Counterfeit risk.
- Export controls and tariffs.
- Dual sourcing.

**Industry lesson:** a clinically excellent device can fail because a critical chip is end-of-life or unavailable in a jurisdiction.

### 5.9 Device lifecycle, reliability, and quality artifacts

A device company runs on artifacts, not slogans. Data scientists will encounter:

| Artifact | What it contains | Why data teams care |
|---|---|---|
| Design History File (DHF) | Design inputs, outputs, reviews, verification, validation, changes | The evidence trail for every model and sensor choice |
| Device Master Record (DMR) | Specifications, drawings, software, labeling, production procedures | The released configuration |
| Design transfer | Evidence that the design can be manufactured reliably | Scale-up can change sensor performance |
| IQ/OQ/PQ | Installation, operational, and performance qualification | Process validation before production |
| Supplier qualification | Audits, incoming inspection, change control | A chip or sensor change can alter the data |
| CAPA | Corrective and preventive action | Complaint trends and recurring model errors |
| UDI | Unique device identification | Traceability and recall execution |
| Complaint/recall file | Adverse events, corrections, removals | Post-market evidence and monitoring |
| Reliability plan | FMEA/FMECA, fault tree, derating, MTBF, HALT/HAST | Failure modes that become data gaps |
| Gauge R&R | Measurement-system repeatability and reproducibility | Quantifies sensor and operator variation |

**Reliability methods:**

- **FMEA/FMECA:** failure modes, effects, severity, occurrence, detection, risk priority.
- **Fault-tree analysis:** top event decomposed into contributing failures.
- **Derating:** operating components below rated stress.
- **MTBF:** mean time between failures; useful but assumes a failure model.
- **HALT/HAST:** accelerated stress testing to find limits.
- **Shelf-life and packaging validation:** performance after storage and sterilization.
- **Gauge R&R:** variance components for repeatability and reproducibility.

**Metrics:** complaint rate, recall rate, time-to-correction, service uptime, CAPA cycle time, and measurement-system variation.

**Rule:** a model cannot be more reliable than the measurement chain that feeds it.

## 6. Medical equipment classes

| Equipment | Core sensing/actuation | Data-science relevance |
|---|---|---|
| Patient monitor | ECG, SpO₂, NIBP, temperature, capnography | Signal quality, alarm logic, deterioration |
| Ventilator | Pressure/flow sensors, valves, turbine | Waveform analysis, asynchrony, weaning |
| Infusion pump | Pressure sensor, motor, drug library | Occlusion detection, dosing errors |
| Dialysis machine | Flow, pressure, conductivity, temperature | Treatment adequacy, hypotension |
| ECMO circuit | Flow, pressure, oxygen saturation | Thrombosis, circuit failure |
| Pacemaker/ICD | Intracardiac electrograms, leads | Arrhythmia detection, battery, recalls |
| Neurostimulator | Electrodes, pulse generator | Programming, seizure control, battery |
| Insulin pump/CGM | Reservoir, cannula, glucose sensor | Closed-loop control, hypoglycemia |
| Ultrasound | Transducer, beamformer | Image quality, workflow, automated measurement |
| CT/MRI/PET | Detectors, gradients, RF coils | Protocol, reconstruction, artifacts |
| Linear accelerator | Beam, MLC, imaging | Planning, delivery, quality assurance |
| Endoscope | CMOS sensor, illumination | Image quality, sterilization, AI |
| Lab analyzer | Optics, electrochemistry, fluidics | Calibration, QC, result validation |
| Surgical robot | Arms, instruments, cameras, force sensors | Kinematics, workflow, skill, safety |
| Rehabilitation robot | Actuators, sensors, orthoses | Gait, assistance, intent detection |

## 7. Medical robotics

### 7.1 Why robotics is different

A robot physically acts on the patient or clinician. Errors are not only wrong predictions; they can be wrong forces, wrong trajectories, or wrong timing.

Robotics combines:

- Mechanical design.
- Sensors and actuators.
- Control theory.
- Computer vision.
- Human factors.
- Safety engineering.
- Regulatory evidence.

### 7.2 Categories

| Category | Examples | Core function |
|---|---|---|
| Surgical teleoperation | da Vinci 5, Hugo, Versius, MIRA | Surgeon controls articulated instruments |
| Surgical navigation | Mako, Rosa, StealthStation | Registration and trajectory guidance |
| Radiosurgery | CyberKnife, Gamma Knife | Precise radiation delivery |
| Rehabilitation/exoskeleton | ReWalk 7, EksoNR, HAL, Atalante X | Assist or retrain movement |
| Prosthetics | Bionic limbs | Replace missing function |
| Capsule endoscopy | PillCam; investigational PillBot | Imaging inside the GI tract |
| Robotic ultrasound | Mendaera Focalist | Guided imaging/intervention |
| Lab automation | Liquid handlers, sequencers | High-throughput processing |
| Pharmacy automation | Dispensing systems | Medication selection |
| Hospital logistics | TUG, Moxi | Transport |
| Disinfection | UV/robot systems | Surface disinfection |

### 7.3 Current examples

- **da Vinci 5** received FDA 510(k) clearance in March 2024. Intuitive claims force feedback, tremor filtering, and more than 10,000× the computing power of da Vinci Xi; these are vendor claims, not independent clinical evidence. Force-feedback instruments have labeled contraindications.
- **Medtronic Hugo** received FDA 510(k) clearance for urologic procedures in December 2025 (K250725), supported by the Expand URO trial. It is modular and uses an open console.
- **J&J MedTech Ottava** received FDA market authorization (DEN250068) in July 2026 for upper-abdominal general surgery, as an integrated operating-table/electromechanical system. It is early in its clinical and commercial lifecycle.
- **ReWalk 7** received FDA 510(k) clearance in March 2025 (K241822) for spinal cord injury ambulation, with setting-dependent indications.
- **Mendaera Focalist** received FDA 510(k) clearance (K250524) on 2 July 2025 for a handheld robotic system for ultrasound-guided needle placement.
- **Medtronic PillCam Genius SB** received FDA 510(k) clearance (K240276) in May 2024. **Endiatx PillBot** remains investigational as of the source cutoff; do not describe it as FDA-cleared.
- **Mako** and **Rosa** are established robotic-arm systems for orthopedic procedures; **CyberKnife** is a robotic radiosurgery system. Product generations and indications vary by jurisdiction.

**Rule:** a clearance is not a clinical-outcome study. Check the indication, predicate, and evidence.

### 7.3a 2024–2026 device and robotics snapshot

| System | Status / date | What changed | Evidence caveat |
|---|---|---|---|
| Intuitive da Vinci 5 | FDA 510(k) March 2024; cardiac indications expanded January 2026 | Force feedback, redesigned console, much greater compute (vendor claim) | Force-feedback clinical benefit still needs outcome evidence; some instruments have labeled contraindications |
| Medtronic Hugo | FDA 510(k) K250725, December 2025, urologic procedures | Modular arms, open console; Expand URO trial | Clearance is indication-specific; general surgery and gynecology require separate evidence |
| J&J Ottava | FDA market authorization DEN250068, July 2026, upper-abdominal general surgery | Integrated operating-table/electromechanical system | New category; long-term outcomes and adoption are early |
| Stryker Mako 4 | Commercial March 2025; spine and shoulder clearances 2024 | Unified hip/knee/spine platform | Robotic-arm assistance is not autonomous surgery |
| Zimmer Biomet Rosa Knee with OptimiZe | FDA 510(k) K251314, November 2025 `[VERIFY]` | Updated knee application | U.S. availability and outcomes still maturing |
| Accuray CyberKnife | Class 2 recall Z-1023-2025, January 2025 | Snap-ring detachment could allow uncontrolled roll-axis rotation | Mechanical failures can have procedure-level consequences |
| Mendaera Focalist | FDA 510(k) K250524, July 2025 | Handheld robotic ultrasound-guided needle placement; PCCP for added ultrasound systems | Limited launch; long-term clinical and workflow evidence pending |
| Medtronic PillCam Genius SB | FDA 510(k) K240276, May 2024 | New capsule endoscopy kit | Capsule retention and bowel preparation remain clinical limits |
| Endiatx PillBot | Investigational; not FDA-cleared | Actively navigated capsule | Company target for De Novo submission is later; do not call it cleared |
| ReWalk 7 | FDA 510(k) K241822, March 2025 | Next-generation personal exoskeleton | Coverage remains a separate hurdle |
| Ekso Indego Personal | CMS HCPCS K1007 payment ~$91,032, April 2024 | Reimbursement pathway | Payment is not proof of comparative effectiveness |
| Lokomat | Established rehabilitation robot | Gait training | A 2024 umbrella review found no convincing/highly suggestive evidence and only small effects |
| Diligent Moxi / Aethon TUG | Hospital logistics; company-reported deployments | Transport, pharmacy delivery | Vendor deployment counts are not independent outcome studies |
| UV disinfection robots | 2024 LAMP randomized trial | Pulsed-xenon UV added to standard cleaning | No reduction in environmentally implicated HAIs (RR 1.10, 95% CI 0.94–1.29) |

**Rule:** separate clearance, adoption, reimbursement, and outcome evidence. They are four different claims.

### 7.4 Autonomy levels

| Level | Description | Example |
|---|---|---|
| 0 | No autonomy; direct teleoperation | Master-slave surgery |
| 1 | Assistance | Tremor filtering, motion scaling |
| 2 | Task autonomy | Suturing sub-task |
| 3 | Conditional autonomy | Supervised task execution |
| 4 | High autonomy | Unsupervised in defined setting |
| 5 | Full autonomy | Not clinically established |

Most clinical robots are level 0–2. Autonomy claims require explicit scope, supervision, and fail-safe design.

### 7.5 Kinematics

**Coordinate frames.** A pose has position and orientation. A homogeneous transform combines rotation \(R\) and translation \(t\):

\[
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}
\]

**Forward kinematics (FK):** joint angles → end-effector pose.

**Inverse kinematics (IK):** desired pose → joint angles; may have multiple or no solutions.

**Jacobian:** maps joint velocities to end-effector velocities:

\[
\dot{x} = J(q)\dot{q}
\]

**Singularity:** a configuration where the Jacobian loses rank; small end-effector motion requires large joint velocities.

**2D worked example.** A planar arm with links \(L_1, L_2\) and joints \(\theta_1, \theta_2\):

\[
x = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2)
\]
\[
y = L_1\sin\theta_1 + L_2\sin(\theta_1+\theta_2)
\]

The Jacobian is:

\[
J = \begin{bmatrix}
-L_1\sin\theta_1 - L_2\sin(\theta_1+\theta_2) & -L_2\sin(\theta_1+\theta_2) \\
L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2) & L_2\cos(\theta_1+\theta_2)
\end{bmatrix}
\]

The arm is singular when \(\theta_2 = 0\) or \(\pi\): the arm is fully extended or folded, and it cannot move radially.

### 7.6 Control

- **PID:** proportional-integral-derivative feedback.
- **Impedance control:** regulate dynamic relationship between force and motion.
- **Admittance control:** measure force, command motion.
- **Force control:** regulate interaction force.
- **Trajectory planning:** joint-space or Cartesian.
- **Haptics:** force feedback to the operator.
- **Teleoperation:** master-slave with latency and scaling.

**Assumptions:** accurate kinematics, calibrated sensors, known tool geometry, stable control loop, and bounded latency.

### 7.7 Registration and calibration

- **Registration:** align preoperative imaging to the patient or robot.
- **Fiducial registration:** markers with known positions.
- **Surface registration:** point clouds.
- **FRE:** fiducial registration error.
- **TRE:** target registration error.
- **Calibration:** camera, tool, robot, and sensor.

**Clinical consequence:** a 2 mm registration error may be acceptable for one procedure and unacceptable for another. Report the error at the target, not only at the fiducials.

### 7.8 Safety

- Emergency stop.
- Safety-rated monitored stop.
- Speed and separation monitoring.
- Power and force limiting.
- Redundant sensors.
- Watchdog and fail-safe state.
- Workspace limits.
- Collision detection.
- Cyber-secure update.

**Standards:**

- IEC 60601-1: general medical electrical safety.
- IEC 80601-2-77: robotically assisted surgical equipment.
- IEC 80601-2-78: medical robots for rehabilitation/assistance.
- ISO/TS 15066: collaborative industrial robots; informative for medical robotics, not a substitute for medical standards.
- ISO 14971: risk management.
- IEC 62304: software lifecycle.
- IEC 62366-1: usability.
- IEC 81001-5-1: health software security.

### 7.9 Learning-based robotics and surgical data science

Modern surgical robots generate synchronized video, kinematics, force, and event logs. Research uses these data for:

- **Surgical phase recognition:** classify operation stages from video (e.g., Cholec80).
- **Instrument and anatomy segmentation:** track tools and critical structures.
- **Skill assessment:** estimate gesture, economy of motion, and force use (e.g., JIGSAWS).
- **Workflow analysis:** room turnover, delay, and team coordination.
- **Imitation learning:** learn policies from expert demonstrations.
- **Reinforcement learning:** optimize sub-tasks in simulation.
- **Vision-language models:** link video to reports and instructions.

**Assumptions:** synchronized clocks, known camera calibration, complete video, accurate phase annotations, and representative surgeons and cases. Surgical video is highly identifiable; governance is strict.

**Evidence:** most learning-based surgical autonomy is research. A 2025 supervised-autonomy system (SRT-H) performed a multi-step cholecystectomy on ex vivo tissue, but this is not a cleared clinical product. Do not confuse a research demo with clinical evidence. Source: https://pubmed.ncbi.nlm.nih.gov/40632876/

**Datasets and tools:** Cholec80, CholecT50, JIGSAWS, MISAW, HeiChole, dVRK, SurRoL, Surgical Gym, MuJoCo, Isaac Lab, ROS 2.

### 7.10 Rehabilitation, prosthetics, and capsule robotics

- **Rehabilitation robots** assist or retrain movement. Control uses intent detection from EMG, force, or motion; gait phase estimation; and impedance/admittance control. Evidence is mixed: a 2024 umbrella review of Lokomat found small effects and no convincing evidence; reimbursement is separate.
- **Prosthetics** use myoelectric pattern recognition, targeted muscle reinnervation, and increasingly sensory feedback. Assumptions include stable electrode contact, sweat, fatigue, and user training.
- **Capsule robots** move passively (peristalsis) or actively (magnetic, leg, or pump-jet). Localization, retention, bowel preparation, and image quality are key limitations.
- **Lab automation** uses liquid handlers, plate readers, and sequencers. The data-science issues are plate/batch effects, calibration, and chain of custody.

### 7.11 Human-robot interaction

- Shared control vs full teleoperation.
- Situational awareness.
- Trust and over-trust.
- Workload.
- Training and credentialing.
- Team communication.
- Ergonomics.
- Alarm and feedback design.

**Methods:**

- **Trust calibration:** measure whether trust matches reliability, not whether users like the system.
- **Situation awareness:** SAGAT or frozen-probe recall of patient, robot, and environment state.
- **Workload:** NASA-TLX or SURG-TLX; pair with task time and error.
- **Usability:** use-related risk analysis, formative and summative evaluation under IEC 62366-1.
- **Team communication:** OR black-box analysis, communication audits, handover checks.
- **Training:** proficiency criteria, not only case counts; learning curves vary by procedure and platform.
- **Mode awareness:** make autonomy state, tool, and next action visible; avoid mode confusion.

**Metrics:** time on task, error rate, near-miss rate, workload, situation awareness, trust calibration, training time, and team communication quality.

**Rule:** a robot that is technically precise but disrupts the OR team's communication can be less safe than a simpler tool.

## 8. Standards and regulatory map

| Standard/regulation | Scope | What data/software teams must produce |
|---|---|---|
| IEC 60601-1 | Medical electrical equipment safety | Risk file, electrical safety evidence, EMC |
| IEC 62304 | Medical device software lifecycle | Requirements, architecture, tests, traceability |
| IEC 62366-1 | Usability engineering | Use specification, formative/summative evaluation |
| ISO 14971 | Risk management | Hazard analysis, controls, residual risk |
| ISO 13485 | Quality management system | Design controls, document control, CAPA |
| ISO 10993 | Biocompatibility | Material and biological evaluation |
| IEC 80601-2-77 | Robotically assisted surgical equipment | Robot-specific safety and performance |
| IEC 80601-2-78 | Rehabilitation/assistive medical robots | Robot-specific safety and performance |
| IEC 81001-5-1 | Health software security | Secure lifecycle, threat model, SBOM |
| FDA premarket cybersecurity | Device cybersecurity | Threat model, testing, update plan |
| EU MDR | CE marking | Clinical evaluation, PMS, technical file |
| EU AI Act | High-risk AI | Risk management, data governance, transparency, monitoring |

**Rule:** hardware and software are one regulated system. A perfect model on an unvalidated sensor chain is not a medical device.

## 9. Data science at the hardware interface

### 9.1 Data types

- Raw waveforms (ECG, EEG, PPG, pressure, flow).
- Device-computed values.
- Alarm/event logs.
- Images and video.
- Robot kinematics, forces, and tool data.
- Device metadata and firmware versions.
- Maintenance and calibration records.
- Adverse-event reports.

### 9.2 Device data interfaces and time synchronization

| Interface | Use | Data-science issue |
|---|---|---|
| IEEE 11073 | Point-of-care medical device communication | Profile-specific semantics; not all devices expose raw values |
| IHE PCD | Enterprise device integration | Mapping and vendor conformance |
| HL7 v2 / FHIR | EHR integration | `Device`, `DeviceMetric`, `Observation`; event timing |
| Bluetooth GATT | Wearable/consumer devices | Proprietary service definitions; pairing and dropout |
| Proprietary serial/USB | Device service and research | Undocumented formats; driver dependency |
| DICOM | Imaging | Metadata and pixel provenance |
| NTP/PTP | Clock synchronization | Drift, time zones, daylight saving, latency |

**Log with every signal:** device model, serial or pseudonymous ID, firmware version, sampling rate, gain, filter settings, calibration date, sensor/lot, battery state, signal-quality flags, and both device and EHR timestamps. Without these, apparent model drift may be a firmware update, a changed sensor lot, or a clock offset.

### 9.3 Assumptions to record

| Assumption | Why it matters | How to check |
|---|---|---|
| Sensor is calibrated | Measurement bias | Calibration logs, reference device |
| Signal quality is adequate | Artifact corrupts features | Quality index, manual review |
| Sampling is consistent | Aliasing and feature mismatch | Metadata, spectral check |
| Device model/firmware is stable | Behavior changes | Version field |
| Placement is consistent | Signal morphology changes | Site/protocol metadata |
| Patient population matches validation | Performance drift | Subgroup analysis |
| Clinician uses output as intended | Workflow effect | Observation, logs |
| Robot registration is accurate | Safety and efficacy | Phantom/TRE measurement |
| Force limits are respected | Tissue injury | Force logs, safety testing |
| Cybersecurity is maintained | Safety and privacy | SBOM, patching, audit |

### 9.4 Validation ladder

1. Bench test with a signal generator.
2. Phantom or simulator.
3. Animal or cadaver, where appropriate.
4. Human factors study.
5. Clinical pilot.
6. Pivotal study.
7. Post-market surveillance.

**Rule:** do not skip from a public dataset to a clinical claim.

## 10. Industry case studies

### 10.1 da Vinci 5 and force feedback

Intuitive received FDA 510(k) clearance for da Vinci 5 in March 2024. The system added force-feedback instruments and a redesigned console. The company reported up to 43% less force on tissue in preclinical trials; this is a vendor result, not a substitute for clinical outcome evidence. The force-feedback needle driver has labeled contraindications.

**Lesson:** a new sensor stream (force) creates new data-science opportunities and new safety evidence requirements.

Sources: Intuitive press release; FDA 510(k) database.

### 10.2 Hugo and the urology clearance

Medtronic's Hugo received FDA 510(k) clearance for urologic procedures in December 2025 (K250725), after the Expand URO trial. The system is modular and uses an open console.

**Lesson:** a robotic platform's clearance is indication-specific. General-surgery and gynecology claims require separate evidence.

Sources: Medtronic press release; FDA 510(k) K250725.

### 10.3 ReWalk 7 and rehabilitation robotics

ReWalk 7 received FDA 510(k) clearance in March 2025 (K241822) for spinal cord injury ambulation, with different indications for home/community and rehabilitation settings.

**Lesson:** rehabilitation robots must demonstrate safe use with a trained companion and in defined environments. Reimbursement is a separate hurdle.

Source: Lifeward/ReWalk press release; FDA K241822.

### 10.4 Surgical robot recalls

In 2025, Virtual Incision's MIRA system was recalled for a software issue that could interrupt or lag motion between the surgeon's controllers and the robot arms, and Quantum Surgical's EPIONE was recalled for a software bug that could misalign the robotic arm relative to the planned trajectory. A CMR Surgical Versius console recall involved Secure Boot not being enabled.

**Lesson:** software and cybersecurity are safety-critical in robotics. Monitor recalls and firmware updates as part of post-market surveillance.

Sources: FDA recall database; manufacturer notices.

### 10.5 Capsule endoscopy

Medtronic PillCam Genius SB is a cleared capsule endoscopy platform (K240276, May 2024). Endiatx PillBot remains investigational; the company itself states it is not FDA-cleared.

**Lesson:** distinguish cleared products from research prototypes and press coverage.

Sources: Medtronic; Endiatx; ClinicalTrials.gov NCT06212206.

### 10.6 CGM and sensor lag

Continuous glucose monitors use electrochemical sensors and estimate interstitial glucose, which lags blood glucose. Compression, calibration, and interfering substances can distort readings.

**Lesson:** a CGM value is a model output with lag and error, not a direct blood glucose measurement.

### 10.7 Wearable PPG and skin tone

PPG-based heart rate and SpO₂ are affected by skin pigmentation, perfusion, motion, and device fit. A 2026 multi-wavelength study reduced skin-tone gaps only after uncertainty-guided rejection.

**Lesson:** optical sensor bias is a hardware, algorithm, and equity problem.

## 11. Labs

### Lab 14A.1 — Signal chain simulation

**Objective:** Simulate an ECG from electrode to ADC.

**Steps:**

1. Generate an ECG waveform.
2. Add baseline wander, 50 Hz interference, and muscle noise.
3. Apply an instrumentation amplifier model with CMRR.
4. Apply an anti-alias filter.
5. Sample at 250 Hz and 100 Hz.
6. Compare reconstruction and QRS detection.

**Deliverable:** signal-chain notebook and memo.

**Acceptance criteria:** aliasing demonstrated; CMRR effect quantified; filter order justified.

### Lab 14A.2 — PPG and skin-tone simulation

**Objective:** Explore how melanin, perfusion, and motion affect PPG.

**Steps:**

1. Simulate a PPG waveform.
2. Attenuate the signal as a function of skin tone and perfusion.
3. Add motion artifact.
4. Estimate heart rate.
5. Stratify error by simulated group.
6. Propose a quality-rejection rule.

**Deliverable:** bias analysis.

**Acceptance criteria:** error stratified; rejection trade-off shown; no claim of clinical validity.

### Lab 14A.3 — TinyML quantization

**Objective:** Deploy a small arrhythmia classifier to an edge target.

**Steps:**

1. Train a small 1D CNN.
2. Quantize to int8.
3. Measure accuracy, latency, and memory.
4. Compare float vs quantized.
5. Write a deployment memo.

**Deliverable:** edge benchmark.

**Acceptance criteria:** target-device measurement; accuracy loss reported; power estimated.

### Lab 14A.4 — Robot kinematics

**Objective:** Implement forward/inverse kinematics for a 2D arm.

**Steps:**

1. Implement homogeneous transforms.
2. Compute FK for a 2-link arm.
3. Derive and implement IK.
4. Compute the Jacobian.
5. Visualize singularities.
6. Add a joint limit.

**Deliverable:** notebook and plots.

**Acceptance criteria:** correct transforms; singularity identified; limits enforced.

### Lab 14A.5 — Safety monitor

**Objective:** Design a force-limit and workspace monitor.

**Steps:**

1. Define safe force and workspace limits.
2. Simulate sensor noise and latency.
3. Implement a watchdog and stop.
4. Test false-positive and false-negative scenarios.
5. Write a safety requirements table.

**Deliverable:** monitor and test report.

**Acceptance criteria:** fail-safe behavior; latency budget; traceability.

### Lab 14A.6 — Device architecture teardown

**Objective:** Reconstruct the signal chain of a cleared device from public documents.

**Steps:**

1. Choose a 510(k) summary.
2. Identify sensors, processor class, outputs, and claims.
3. Map the signal chain.
4. List assumptions and failure modes.
5. Identify missing evidence.
6. Write a one-page review.

**Deliverable:** architecture review.

**Acceptance criteria:** sourced; no unsupported internal details; evidence gaps clear.

### Lab 14A.7 — Hardware-in-the-loop test plan

**Objective:** Write a HIL plan for a sensor algorithm.

**Steps:**

1. Define the device under test.
2. Specify signal sources and phantoms.
3. Define test cases, including edge cases.
4. Define acceptance criteria.
5. Define data capture and traceability.
6. Estimate cost and time.

**Deliverable:** HIL plan.

**Acceptance criteria:** covers power, latency, artifact, and safety; reproducible.

### Lab 14A.8 — Surgical phase recognition

**Objective:** Classify surgical phases from video.

**Steps:**

1. Use Cholec80 or a public teaching subset.
2. Split by patient/video, not frame.
3. Extract frame features or use a temporal model.
4. Evaluate accuracy, macro-F1, and segmental edit distance.
5. Inspect phase-transition errors.
6. Discuss privacy and consent.

**Deliverable:** model and evaluation.

**Acceptance criteria:** patient-level split; temporal metrics; error analysis.

### Lab 14A.9 — Registration error

**Objective:** Quantify target registration error.

**Steps:**

1. Create synthetic fiducials with noise.
2. Implement point-pair registration (Kabsch/Umeyama).
3. Compute fiducial and target errors.
4. Vary fiducial count and noise.
5. Compare with a surface/ICP method.
6. State the clinical implication.

**Deliverable:** simulation and memo.

**Acceptance criteria:** FRE vs TRE distinguished; error propagated to a target.

### Lab 14A.10 — FMEA and Gauge R&R

**Objective:** Quantify failure risk and measurement-system variation.

**Steps:**

1. Build an FMEA for a sensor module.
2. Score severity, occurrence, and detection.
3. Identify the top three risks and controls.
4. Run a Gauge R&R with simulated operators and repeats.
5. Compute repeatability, reproducibility, and percent variation.
6. Recommend a measurement improvement.

**Deliverable:** FMEA and Gauge R&R report.

**Acceptance criteria:** traceable risks; variance components interpreted; action linked to safety.

### Lab 14A.11 — Summative usability test

**Objective:** Evaluate a device or robot interface for use-related risk.

**Steps:**

1. Write the use specification and hazard-related use scenarios.
2. Recruit representative users.
3. Run formative tasks and record use errors.
4. Run a summative scenario with simulated clinical pressure.
5. Analyze errors by cause and severity.
6. Propose design changes and re-test.

**Deliverable:** usability report.

**Acceptance criteria:** representative users; use-error taxonomy; severity and mitigation.

## 12. Assignment: device data and evidence plan

Choose a medical device or robot and write a 2,500-word plan:

1. Intended use and decision.
2. Signal chain and sensor physics.
3. Semiconductor/embedded architecture.
4. Data capture and quality.
5. Algorithm and assumptions.
6. Robot kinematics/control, if applicable.
7. Safety and cybersecurity.
8. Standards and regulatory pathway.
9. Validation ladder and HIL plan.
10. Monitoring and lifecycle.

## 13. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Signal chain | 20% | Lists components | Explains function and failure | Quantifies noise, latency, and safety |
| Sensor physics | 20% | Names sensors | Explains principle and assumptions | Links bias to clinical harm |
| Embedded systems | 15% | Generic block diagram | Constraints and trade-offs | Hardware/software co-design |
| Robotics | 20% | Describes a robot | Kinematics/control/safety | Registration, latency, HRI |
| Regulation | 15% | Names standards | Maps standards to artifacts | Evidence and lifecycle plan |
| Communication | 10% | Technical | Clear | Cross-functional |

## 14. Regulatory, ethical, and commercial lens

- **FDA:** device classification, 510(k)/De Novo/PMA, cybersecurity, human factors.
- **EU:** MDR, IVDR, AI Act, clinical evaluation, PMS.
- **Standards:** IEC 60601-1, 62304, 62366-1, 80601-2-77, 80601-2-78, ISO 14971, 13485, 10993, 81001-5-1.
- **Reimbursement:** hardware plus software plus service; capital equipment procurement.
- **Equity:** sensor performance across skin tones, body sizes, and access.
- **Safety:** hardware failures can injure immediately; software failures can injure at scale.
- **Supply chain:** device shortages can be clinical crises.

## 15. Failure modes and red flags

- Treating a device output as ground truth.
- Ignoring sensor calibration and drift.
- Training on filtered data and deploying raw data.
- Assuming a wearable is accurate across skin tones.
- Ignoring latency in a “real-time” system.
- Claiming autonomy without evidence.
- Confusing a cleared robot with a proven outcome.
- Ignoring firmware updates and SBOM.
- No HIL or phantom testing.
- No plan for obsolescence.
- No cybersecurity monitoring.

## 16. Interview and discussion questions

1. **Walk me through a medical device signal chain.**
   Hint: sensor, AFE, ADC, processor, algorithm, output, workflow.
2. **What is CMRR, and why does it matter?**
   Hint: rejection of common-mode interference.
3. **Why can a PPG reading be biased?**
   Hint: skin tone, perfusion, motion, fit.
4. **What does an ADC's effective number of bits mean?**
   Hint: real resolution after noise and distortion.
5. **When would you choose an FPGA over an MCU?**
   Hint: low-latency parallel processing, prototyping, power trade-off.
6. **What is a singularity in robot kinematics?**
   Hint: Jacobian loses rank; large joint velocities.
7. **What is the difference between impedance and admittance control?**
   Hint: regulate force-motion relationship vs motion from force.
8. **What is target registration error?**
   Hint: error at the clinical target, not at fiducials.
9. **What standards apply to a surgical robot?**
   Hint: IEC 60601-1, 80601-2-77, 62304, ISO 14971.
10. **What standards apply to a rehabilitation robot?**
    Hint: IEC 80601-2-78, 60601-1, ISO 14971.
11. **How do you validate an edge-AI model?**
    Hint: target-device quantization, latency, power, clinical population.
12. **What would make you stop a device project?**
    Hint: unacceptable harm, unverifiable safety, no regulatory path, no maintenance.

## 17. Key terms

| Term | Definition |
|---|---|
| AFE | Analog front end |
| ADC | Analog-to-digital converter |
| CMRR | Common-mode rejection ratio |
| PPG | Photoplethysmography |
| SpO₂ | Peripheral oxygen saturation |
| MEMS | Micro-electromechanical system |
| SoC | System on chip |
| FPGA | Field-programmable gate array |
| ASIC | Application-specific integrated circuit |
| PMIC | Power management integrated circuit |
| TinyML | Machine learning on tiny embedded devices |
| RTOS | Real-time operating system |
| SBOM | Software bill of materials |
| Homogeneous transform | Rotation plus translation matrix |
| Forward kinematics | Joint angles to end-effector pose |
| Inverse kinematics | End-effector pose to joint angles |
| Jacobian | Joint velocity to end-effector velocity map |
| Singularity | Configuration where Jacobian loses rank |
| Impedance control | Control of force-motion relationship |
| Registration | Alignment of image/robot/patient coordinates |
| FRE / TRE | Fiducial / target registration error |
| HIL | Hardware-in-the-loop |
| IEC 60601-1 | Medical electrical equipment safety |
| IEC 80601-2-77 | Robotically assisted surgical equipment |
| IEC 80601-2-78 | Rehabilitation/assistive medical robots |
| ISO/TS 15066 | Collaborative robot safety |

## 18. Sources and further reading

### Standards and regulation
- IEC 60601-1. https://www.iec.ch/
- IEC 62304. https://www.iec.ch/
- IEC 62366-1. https://www.iec.ch/
- IEC 80601-2-77. https://webstore.iec.ch/en/publication/29933
- IEC 80601-2-78. https://webstore.iec.ch/en/publication/69618
- IEC 81001-5-1. https://www.iec.ch/
- ISO 14971. https://www.iso.org/standard/72704.html
- ISO 13485. https://www.iso.org/standard/59752.html
- ISO 10993. https://www.iso.org/
- ISO/TS 15066. https://www.iso.org/
- FDA, *Cybersecurity in Medical Devices*. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- EU MDR. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- EU AI Act. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

### Devices and robotics
- Intuitive, da Vinci 5 clearance. https://isrg.intuitive.com/news-releases/news-release-details/intuitive-announces-fda-clearance-fifth-generation-robotic
- Medtronic, Hugo FDA clearance. https://news.medtronic.com/2025-12-03-Medtronic-announces-FDA-clearance-of-Hugo-TM-robotic-assisted-surgery-system-for-urologic-surgical-procedures
- FDA 510(k) K250725. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K250725
- Lifeward/ReWalk 7. https://ir.rewalk.com/news-releases/news-release-details/fda-issues-clearance-rewalk-7-exoskeleton
- FDA K241822. https://www.accessdata.fda.gov/cdrh_docs/pdf24/K241822.pdf
- Mendaera Focalist. https://www.mendaera.com/press-releases/mendaera-receives-fda-510-k-clearance-for-handheld-robotic-system
- Endiatx PillBot. https://endiatx.com/technology/
- ClinicalTrials.gov NCT06212206. https://clinicaltrials.gov/study/NCT06212206
- FDA recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
- MIRA recall Z-1905-2025 `[VERIFY]`. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm?id=213740
- EPIONE recall Z-1927-2025 `[VERIFY]`. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm?id=214111

### Sensors, chips, and methods
- Webster, J. G., *Medical Instrumentation: Application and Design*.
- Electronic Design, "Inside the Mixed-Signal ASICs Transforming Healthcare Sensing," 2025. https://www.electronicdesign.com/technologies/embedded/article/55323248/ensilica-inside-the-mixed-signal-asics-transforming-healthcare-sensing
- GlobalFoundries, custom silicon for medical devices, 2025. https://gf.com/news-and-events/blog/how-globalfoundries-is-powering-the-rise-of-ultra-portable-ai-enabled-medical-devices-one-custom-chip-at-a-time/
- MDPI *Micromachines*, hardware accelerators for cardiovascular signal processing, 2026 (verify volume/year). https://www.mdpi.com/2072-666X/17/1/51
- ROS 2. https://docs.ros.org/
- Gazebo. https://gazebosim.org/
- MuJoCo. https://mujoco.org/
- OpenSim. https://opensim.stanford.edu/
- Cholec80. https://camma.unistra.fr/datasets/
- JIGSAWS. https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- dVRK. https://dvrk.readthedocs.io/
- SurRoL. https://github.com/med-air/SurRoL
- NeuroKit2. https://neurokit2.readthedocs.io/
- WFDB. https://physionet.org/content/wfdb/

### Current evidence and device examples
- Intuitive, da Vinci 5 Q4 2025 results. https://investor.intuitivesurgical.com/news-releases/news-release-details/intuitive-announces-fourth-quarter-earnings-5
- J&J MedTech, Ottava FDA authorization. https://www.jnjmedtech.com/en-US/news/press-releases/johnson-johnson-receives-fda-market-authorization-in-the-u-s-for-its-ottava-robotic-surgical-system/
- FDA, Ottava DEN250068. https://www.accessdata.fda.gov/cdrh_docs/pdf25/DEN250068.pdf
- Stryker, Mako 4. https://www.stryker.com/us/en/about/news/2025/stryker-showcases-next-generation-of-mako-smartrobotics--at-aaos.html
- Zimmer Biomet, Rosa Knee with OptimiZe. https://investor.zimmerbiomet.com/news-and-events/news/2025/11-14-2025-123022448
- FDA, CyberKnife recall Z-1023-2025. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfRES/res.cfm?id=211275
- Medtronic, PillCam Genius SB. https://news.medtronic.com/Medtronic-announces-successful-first-ingestion-of-new-PillCam-TM-Genius-SB-Kit
- Ekso Bionics, CMS reimbursement. https://ir.chronoscale.com/press-releases/detail/764/ekso-bionics-announces-receipt-of-initial-cms-reimbursement
- Lokomat umbrella review, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11546060/
- Diligent Robotics, Moxi. https://www.therobotreport.com/diligent-robotics-ceo-discusses-road-to-1m-hospital-deliveries-future-moxi-destinations/
- UV disinfection LAMP trial, 2024. https://doi.org/10.1093/cid/ciae237
- SpO2 skin-tone meta-analysis, 2024. https://www.jmir.org/2024/1/e62769
- FDA draft pulse oximeter guidance, 2025. https://www.fda.gov/media/184896/download
- PPG posture and skin-tone study, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12204546/
- CGM comparison. https://pmc.ncbi.nlm.nih.gov/articles/PMC11089878/
- CGM iCGM special-controls analysis. https://pmc.ncbi.nlm.nih.gov/articles/PMC11969480/
- Butterfly iQ3. https://www.butterflynetwork.com/press-releases/butterfly-network-launches-butterfly-iq3
- Siemens Naeotom Alpha. https://www.siemens-healthineers.com/en-us/press-room/press-releases/naeotom-family-fda-clearance
- Dry ECG review. https://doi.org/10.1002/jbm.a.37845
- Dry EEG review. https://doi.org/10.1109/jsen.2020.3012394
- ISO 10993-1:2025. https://www.iso.org/standard/10993-1
- IEC 60601-1-2. https://webstore.iec.ch/en/publication/67554
- IEC 60601-1-11. https://webstore.iec.ch/en/publication/67384
- ISO 10218-1:2025. https://www.iso.org/standard/73933.html
- IEC 81001-5-1. https://webstore.iec.ch/en/publication/63293
- FDA cybersecurity guidance, 2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket
- FDA device software guidance, 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions

> **Next:** [Module 14B](14B_efficient_ai_distillation_quantization_on_device_inference.md) compresses and deploys models on the hardware described here.
