# Module 24 — Hardware, Firmware, EMC, HIL, and Robotics Labs

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (3 orientation and safety briefings, 25 guided lab, including the capstone studio and review)
> **Track fit:** MedTech, Sensors & Robotics advanced lab elective; open to Health Systems engineers who completed 14A and 14B
> **Prerequisites:** Modules 14A (hardware, sensors, robotics concepts), 14B (on-device inference), 11 (physiological signals); Python and basic electronics
> **Tools:** Python, NumPy/SciPy, LTspice or ngspice, an STM32/nRF/Arduino board or Renode/Wokwi, a logic analyzer and power monitor (or simulators), ROS 2 with Gazebo or MuJoCo, Pinocchio, MoveIt 2, LiteRT for Microcontrollers/CMSIS-NN
> **Industry question:** "Can you produce bench evidence that the number, the deadline, and the motion are what you claim — and can you say precisely what your evidence does not prove?"

---

## 1. Why this lab elective exists
Modules 14A and 14B teach the concepts: signal chains, sensor physics, embedded constraints, compression, and the regulatory map. Module 11 teaches the signal science. None of them makes a learner the person who can walk to a bench, inject a known input, and defend the result under questioning.

That person is who MedTech employers hire. Device and robot manufacturers need engineers who can write a calibration record, run a Gauge R&R, prove a watchdog resets a hung task, find an EMC weakness before a test-house booking, and build a HIL rig that exercises the faults in the risk file. Test houses, hospital clinical engineering groups, regulators, and technical due-diligence teams need the same skill in reverse: the ability to inspect evidence and ask what it actually establishes.

The commercial stakes are concrete:

- A late EMC redesign can consume a quarter of engineering time and delay a launch.
- A HIL escape can become a field correction or a recall.
- A frame or registration error can move a robot into a no-go zone.
- A calibration that was never validated can make a good model look wrong, or a bad sensor look good.
- A "secure boot enabled" checkbox that was never tested can become a cybersecurity recall.

This elective is deliberately artifact-first. Every lab ends in a document, dataset, or test rig that another engineer could inspect. The rule for the module:

> **The lab is not the product. The evidence is.**

## 2. Learning outcomes
By the end of this module, you can:

- Design a signal chain and defend its noise, resolution, headroom, and anti-alias choices with measured or simulated evidence.
- Calibrate temperature, pressure, and PPG-derived measurements, and quantify measurement-system variation with a crossed Gauge R&R.
- Build RTOS firmware with a watchdog, fail-safe state, secure boot, and rollback, then inject faults that try to defeat each control.
- Explain EMC coupling mechanisms, run safe pre-compliance experiments, and write a test plan that distinguishes pre-compliance from compliance.
- Build a hardware-in-the-loop rig that injects the faults named in a risk file and produces traceable pass/fail evidence.
- Build ROS 2, Gazebo, and MuJoCo models with correct frames, timestamps, and sensor pipelines, and diagnose frame errors.
- Implement forward and inverse kinematics and the Jacobian, detect singularities, and apply damped least squares.
- Measure teleoperation latency at 0, 50, 100, and 200 ms and design a safety monitor with stated limits, detection latency, and false-stop behavior.
- Deploy a quantized model to an MCU and measure accuracy, p99 latency, energy, flash, and SRAM on the target.
- Assemble a capstone evidence package with requirements, risk controls, verification results, and an explicit statement of regulatory gaps.

## 3. Mental model: the evidence ladder
A claim about a device moves up a ladder. Each rung answers a different question, and no rung substitutes for the next.

```
Concept -> Simulation -> Bench -> HIL -> Phantom -> Human factors -> Clinical -> Post-market
```

| Rung | Question answered | Typical artifact | What it does not prove |
|---|---|---|---|
| Simulation | Does the math or control law work? | Notebook, model, scenario suite | Real sensor, real timing, real fault behavior |
| Bench | Does the circuit and firmware work with known inputs? | Calibration record, scope captures, unit tests | Performance in the clinical population |
| HIL | Does the integrated system meet timing and safety requirements under faults? | HIL report, fault-injection matrix, latency histogram | Usability, clinical benefit, EMC compliance |
| Phantom | Does it work on a physical surrogate? | Phantom study, imaging or signal agreement | Human anatomy and workflow |
| Human factors | Can representative users use it safely? | Formative and summative usability report | Clinical efficacy |
| Clinical | Does it change the clinical decision or outcome? | Trial or pivotal study | Long-term safety, rare events |
| Post-market | Does performance hold in the field? | PMS, complaint and recall monitoring | Nothing by itself; it is the ongoing obligation |

**Lab map.** Labs 1–5 form the measurement and firmware spine; Labs 6–8 form the robotics spine; Lab 9 connects firmware to model deployment; Lab 10 integrates at least three labs and produces the evidence package.

## 4. Safety, standards, and the compliance boundary
### 4.1 Non-negotiable lab rules
1. **No electrodes on a person.** Use a resistor network, a signal generator, a commercial simulator, or a recorded waveform. ECG and PPG labs do not require a human subject.
2. **SELV only.** No mains voltage on the bench. Use a current-limited bench supply or a USB supply with a fuse. A current limit is not a substitute for a fuse.
3. **ESD control.** Wear a wrist strap when handling bare boards. Use an ESD mat. Treat every exposed pin as sensitive.
4. **Robot guarding.** Any moving arm is inside a guard or behind a light curtain, with a reachable emergency stop, joint limits, and a low speed cap. No hands in the workspace while enabled.
5. **Eye and thermal protection.** Safety glasses for soldering, cutting, and pressure work. Hot water for temperature calibration stays below 60 °C and is handled with tongs.
6. **Stop-work authority.** Any learner can stop a lab without justification. Near misses are reported, not hidden.
7. **No PHI.** Use public, licensed, or synthetic data. Do not put identifiable clinical data on a lab machine or in a repository.
8. **Label prototypes.** Mark every prototype "Not for clinical use — educational prototype."
9. **Record everything.** Firmware version, model hash, calibration date, instrument ID, operator, environment, and raw data hash.
10. **Do not overclaim.** A lab result is not a design verification, a clinical validation, or a compliance certificate.

### 4.2 Electrical safety
IEC 60601-1 is the general standard for basic safety and essential performance of medical electrical equipment. The concepts a lab can illustrate are isolation, protective earthing, patient leakage current, means of patient protection (MOPP) and means of operator protection (MOOP), defibrillation protection, and fail-safe behavior.

IEC 60601-1 sets patient leakage limits on the order of 10 µA for type CF applied parts in normal condition and 100 µA for type B/BF applied parts, with higher limits under single-fault conditions. The exact values depend on the applied-part type, the condition, and the edition. **Confirm the current edition and the exact clauses before citing a number in a report.**

A multimeter reading of "no voltage" does not establish isolation. Leakage current requires a defined measurement network, a calibrated instrument, and the conditions in the standard. A teaching lab can demonstrate the concept with a isolated supply and a modeled leakage path; it cannot produce a compliance result.

Battery safety: use a protected cell, a charger designed for the chemistry, a fire-resistant bag, and a charging area away from flammable material. Do not puncture, crush, or overcharge a lithium cell.

### 4.3 Robot safety
IEC 80601-2-77 covers robotically assisted surgical equipment. IEC 80601-2-78 covers medical robots for rehabilitation, assessment, compensation, or alleviation. ISO/TS 15066 provides guidance on collaborative industrial robot force and pressure limits; it is informative for medical robotics but is **not** a substitute for the medical particular standards.

Safety functions to understand and, where possible, demonstrate in simulation:

- Emergency stop and safety-rated monitored stop.
- Speed and separation monitoring.
- Power and force limiting.
- Workspace and joint limits.
- Redundant sensing and plausibility checks.
- Watchdog, command-age limit, and fail-safe state.
- Collision detection and safe retraction.

A software workspace clamp is not a safety-rated limit. A simulated e-stop is not a certified safety function. In hardware labs, use a torque-limited servo, a physical guard, and a hardware e-stop that removes actuator power.

Force and pressure thresholds in ISO/TS 15066 are body-region dependent. Do not quote a single universal number. Use the annex values only after reading the current standard and the applicable body-region table.

### 4.4 Human data and human subjects
No lab in this module requires a human subject. If a learner wants to run a small usability study with colleagues, they need informed consent, a clear statement that the task is not clinical care, the right to withdraw, and institutional review if the institution requires it. Do not record identifiable video or audio without consent.

Public physiological datasets such as MIT-BIH are de-identified but still governed by license and data-use terms. Record the dataset version, the license, and the preprocessing applied. Do not redistribute restricted data.

### 4.5 Standards map and claim boundary
| Standard | What it governs | Lab that touches it | What the lab cannot establish |
|---|---|---|---|
| IEC 60601-1 | Basic safety and essential performance | 25.1, 25.3, 25.4 | Type testing, leakage compliance, accredited results |
| IEC 60601-1-2 | Electromagnetic compatibility | 25.4 | EMC compliance |
| IEC 62304 | Medical device software lifecycle | 25.3, 25.5, 25.9 | Full lifecycle, software safety class assignment |
| ISO 14971 | Risk management | 25.5, 25.8, 25.10 | Complete risk management file |
| IEC 80601-2-77 | Robotically assisted surgical equipment | 25.7, 25.8 | Particular-standard testing |
| IEC 80601-2-78 | Rehabilitation and assistive medical robots | 25.8 | Particular-standard testing |
| IEC 62366-1 | Usability engineering | 25.8, 25.10 | Summative validation with representative users |
| ISO/TS 15066 | Collaborative robot force and pressure guidance | 25.8 | Medical compliance; it is industrial guidance |
| ISO 10218-1:2025 | Industrial robot safety | 25.8 | Medical use or clinical safety |
| IEC 81001-5-1 | Health software security lifecycle | 25.3, 25.9 | A complete security program |
| ISO 13485 | Quality management system | 25.10 | Certification |

**No lab in this module makes a device compliant with any of these standards.** Compliance requires a quality management system, design controls, accredited testing where applicable, clinical evidence, and a conformity assessment body. The labs teach the mechanisms and the evidence habits that make those later activities possible.

### 4.6 What "pre-compliance" means
Pre-compliance testing is early, in-house, margin-seeking testing. It is used to find design weaknesses before an accredited test-house booking. It is not a compliance test, and a pass in a teaching lab has no regulatory standing. A useful pre-compliance report states the test setup, the instrument calibration, the limits it borrowed, the margin observed, and the conditions it did not cover.

### 4.7 Documentation and traceability
A result that cannot be reproduced is an anecdote. Every lab report records:

- The question and the claim being tested.
- Hardware revision, firmware commit, model hash, and toolchain versions.
- Instrument ID and calibration status.
- Raw data location and SHA-256 hash.
- Environmental conditions that matter (temperature, supply voltage, sample rate).
- The analysis method and the uncertainty.
- The acceptance criterion and the verdict.
- What the result does not prove.

## 5. Bench and simulation stack
### 5.1 Bill of materials
Every lab has a Tier 0 path that uses only a computer. Buy hardware only after the simulation path works, and share instruments across teams. Prices are approximate, region-dependent, and should be verified before purchase.

| Tier | Approximate spend | What it enables | Limit |
|---|---:|---|---|
| 0 — simulation only | $0 | All labs except target timing, energy, and real analog behavior | Cannot claim target measurements |
| 1 — minimum bench | $150–250 | One MCU, sensors, DAC, logic analyzer, DMM | No scope, no calibrated reference, no ESD tester |
| 2 — fuller bench | $600–900 | Adds scope, power monitor, ESD protection, servo arm, reference thermometer | Still not an accredited EMC or safety lab |

| Category | Item | Low-cost option (approx. USD) | Simulation-only substitute | Labs |
|---|---|---|---|---|
| MCU | Cortex-M board | STM32 Nucleo-F446RE (~18), nRF52840 DK (~45), Arduino Nano 33 BLE Sense (~35), ESP32 (~8), RP2040 Pico (~5) | Renode, Wokwi, QEMU | 3, 5, 9 |
| Debug | Probe | On-board ST-LINK or CMSIS-DAP (~15) | Simulator debugger | 3, 9 |
| Prototyping | Breadboard, jumpers, passives | ~20 | Ideal wires in LTspice | 1, 2, 4 |
| DMM | Handheld multimeter | ~25 | LTspice/ngspice | 1–4 |
| Logic analyzer | 8-channel, 24 MHz | ~12 clone with sigrok/PulseView | Synthetic VCD, scope | 3, 5, 9 |
| Oscilloscope | 2-channel USB or bench | ~150–400 | ngspice/LTspice | 1, 3, 4 |
| Signal source | Function generator or DAC | AD9833 (~10), MCP4725 (~5), R-2R (~2) | NumPy waveform generator | 1, 4, 5 |
| Power monitor | Current/energy | INA219 (~6), INA228 (~12), PPK2 (~100), Joulescope (~400) | Datasheet estimate (not a measurement) | 3, 9 |
| ECG AFE | Analog front end | AD8232 (~10), INA333 (~5) plus passives | Python ECG model | 1, 5 |
| Temperature | Sensor and reference | TMP117 (~10), NTC 10k (~1); ice bath plus boiling water (altitude-corrected) or calibrated thermometer (~25) | Synthetic sensor model | 2 |
| Pressure | Sensor and reference | MPRLS (~20), BMP390 (~10), syringe plus manometer (~20) | Synthetic sensor model | 2 |
| PPG | Optical sensor | MAX30102 (~8) | Synthetic PPG with motion and skin-tone attenuation | 2 |
| Force | Load cell and amplifier | 1 kg load cell plus HX711 (~10) | MuJoCo contact force | 8 |
| Robot | 2-link servo arm | Two MG996R servos plus bracket (~40) | Gazebo or MuJoCo | 6–8 |
| ESD | Mat and wrist strap | ~25 | IEC 61000-4-2 waveform model | 4 |
| EMC | TVS diodes, ferrites, common-mode choke, shielded cable | ~20 | ngspice filter and rectification model | 4 |
| Safety | Guard, hardware e-stop, fuse | ~30 | Simulated e-stop with logged state | 8 |
| Cables | USB, BNC, headers | ~15 | — | All |

**Do not buy an ESD gun, an anechoic chamber, or a calibrated leakage tester for this course.** Those are shared facilities or simulation topics. The point is to understand the mechanism and the evidence, not to reproduce an accredited test.

### 5.2 Software and versions
| Tool | Status and version note | Used in |
|---|---|---|
| Python 3.11 or 3.12, NumPy, SciPy, pandas, matplotlib | Pin in `requirements.txt` | All |
| NeuroKit2 | Current release; check API changes for `ecg_simulate` | 1, 5 |
| WFDB | Current release; PhysioNet data access | 5, 9 |
| ROS 2 Jazzy Jalisco (LTS, 2024) or Humble (LTS, 2022) | 2026 LTS release status `[VERIFY]` | 6–8 |
| Gazebo Harmonic or Jetty | Pairing with the chosen ROS 2 release `[VERIFY]` | 6 |
| MuJoCo 3.x | Python bindings; MJCF and URDF import path `[VERIFY]` | 6–8 |
| Pinocchio 3.x | Rigid-body dynamics and Jacobians | 7 |
| MoveIt 2 | Motion planning; configuration effort is significant | 6, 7 |
| ros_gz | ROS 2 and Gazebo bridge | 6 |
| TensorFlow 2.x and LiteRT for Microcontrollers | Formerly TensorFlow Lite for Microcontrollers; API names `[VERIFY]` | 9 |
| CMSIS-NN | Arm Cortex-M optimized kernels | 9 |
| STM32CubeIDE and STM32Cube.AI | ST Edge AI naming and toolchain `[VERIFY]` | 3, 9 |
| FreeRTOS or Zephyr | RTOS kernel and drivers | 3 |
| MCUboot | Secure boot and image management | 3 |
| Renode or Wokwi | MCU simulation without hardware | 3, 9 |
| sigrok/PulseView | Logic analyzer capture and protocol decode | 3, 5, 9 |
| LTspice or ngspice | Analog and EMC mechanism simulation | 1, 4 |
| openEMS | Open-source FDTD; learning curve is steep `[VERIFY]` | 4 |
| statsmodels, pytest, Docker | Regression, tests, reproducible environment | 2, 5, 9 |

Pin versions and record the container digest. A lab that worked last semester with a different toolchain is not the same lab.

### 5.3 Instrument practice
| Instrument | What it tells you | Common mistake |
|---|---|---|
| DMM | DC volts, current, continuity, resistance | Ignoring burden voltage; wrong range; floating input |
| Oscilloscope | Waveform shape, noise, timing, transient response | Long ground lead; insufficient bandwidth; probing a floating circuit |
| Logic analyzer | Digital timing and protocol content | Sample rate too low; wrong threshold; decoding the wrong clock edge |
| Power monitor | Current, charge, energy per inference | Burden voltage; range switching; averaging hides current peaks |
| Signal generator or DAC | A known input with known amplitude and frequency | No reconstruction filter; DC offset; slew-rate limit |
| Reference thermometer | Traceable calibration point | No equilibration; wrong immersion depth; ice bath not stirred |
| Calipers and fixture | Geometry and repeatability | Measuring a compliant part with inconsistent force |

For energy measurements, use a series shunt and account for its burden voltage. For latency, measure on the target with a cycle counter or a GPIO toggle captured by the logic analyzer, not with a desktop timer. For p99, run enough iterations that the tail is populated and report the number of runs.

## 6. Lab conventions and reproducibility
### 6.1 Repository layout
```
lab-25/
  25_01_signal_chain/
    config.yaml
    src/
    data/            # licensed data not committed; record source and hash
    results/
    report.md
  shared/
    instruments/     # calibration records and serial numbers
    scenarios/       # YAML signal and fault scenarios
```

Commit code and configuration. Do not commit licensed datasets or large raw captures; record the source, version, license, and SHA-256 hash instead.

### 6.2 Report template
Each lab report contains: question and claim; setup and versions; calibration status; raw data and hash; method; results with uncertainty; acceptance criterion and verdict; limitations and what the result does not prove; next step. A report without a limitations section is incomplete.

### 6.3 Schedule
Ten sessions of 2–3 hours: orientation and Lab 25.1; Labs 25.2–25.9 in order; Lab 25.10 and capstone review. Times assume a prepared starter repository and a partner. First-time installation of ROS 2, a secure-boot toolchain, or an MCU debug stack can add two to four hours; record setup time separately from lab time.

## 7. From lab to field: failure patterns
The following public recall narratives map to the mechanisms the labs exercise. They are not lab results, and the labs do not reproduce the devices. Recall numbers and dates should be checked against the FDA recall database before use in a report.

| Event | Mechanism | Lab that exercises it | What the lab still does not prove |
|---|---|---|---|
| CMR Surgical Versius console recall, 2025 `[VERIFY]` | Secure Boot reportedly not enabled on a console | 25.3 | That the same configuration or attack path applies |
| Virtual Incision MIRA recall, 2025 `[VERIFY]` | Software could interrupt or lag motion between controllers and arms | 25.8 | Clinical consequence in a real procedure |
| Quantum Surgical EPIONE recall, 2025 `[VERIFY]` | Software bug could misalign the arm relative to the planned trajectory | 25.7, 25.8 | Registration error on a patient |
| Accuray CyberKnife recall Z-1023-2025 | Snap-ring detachment could allow uncontrolled roll-axis rotation | 25.7, 25.8 | Structural failure under clinical loads |
| Medtronic MiniMed 600 series recall, 2019 | Incorrect reservoir retention ring could cause over- or under-delivery | 25.2, 25.5 | Dosing error in a patient |

The pattern is consistent: a mechanical or configuration detail that was never exercised by an integrated test. The lab habit that catches these is fault injection tied to the risk file, not more nominal testing.
---
## 8. The ten labs
### Lab 25.1 — Signal chain, ADC, and aliasing bench
**Objective.** Build a biopotential signal chain in simulation and, optionally, on the bench. Predict and measure aliasing, quantization error, clipping, and common-mode rejection, then name the dominant impairment.
**Prerequisites.** Module 11 §4.1–4.2; Module 14A §3 and §5.3; comfortable with NumPy and the FFT.
**Hardware / software.** Python 3.11+, NumPy, SciPy, NeuroKit2, matplotlib. Optional bench: AD8232 or INA333 breakout, function generator or AD9833 module, 2-channel scope, 12-bit ADC on an STM32 or Arduino, breadboard, resistors, capacitors. Simulation-only path: Python plus LTspice or ngspice.
**Setup.** Use fs = 250 Hz and 100 Hz; band of interest 0.5–40 Hz; ADC 12 bits, Vref 3.3 V, gain 100; ECG amplitude 1 mV; 60 Hz interferer at 2 mV input-referred; common-mode 1 V at 60 Hz; CMRR 80 dB and 60 dB. Put every parameter in `config.yaml`.

**Tasks.**

1. Generate a 10 s ECG at 1 kHz with NeuroKit2 and resample it to 250 Hz and 100 Hz. Verify the resampling with a spectrum, not by eye.
2. Compute Nyquist for each rate. List every component above Nyquist: 60 Hz interference, EMG content above 40 Hz, and any baseline wander below 0.5 Hz that you choose to remove.
3. Build the ideal chain and quantize to 12, 10, and 8 bits. Compute SNR and ENOB using a full-scale sine test and again using the ECG. Report both.
4. Add a Butterworth anti-alias filter with a 40 Hz cutoff. Sweep order 1, 2, and 4. Measure attenuation at 60 Hz and the QRS amplitude error after filtering.
5. Sample at 250, 100, and 50 Hz with and without the anti-alias filter. Compare each observed spectral peak with `abs(f - fs*round(f/fs))`.
6. Model common-mode interference as `Vcm / 10**(CMRR/20)` at the ADC input. Compare the result with the quantization step and state which dominates.
7. Raise the gain until the QRS clips. Show that no downstream filter recovers the clipped peak, then write the headroom rule you would use in a requirement.
8. Optional bench: inject a 2 Hz sine plus 60 Hz into the ADC, capture with the scope and logic analyzer, and compare the measured alias with the prediction. Save the raw capture and a photograph of the setup.

**Code.**

```python
import numpy as np
from scipy import signal
import neurokit2 as nk

fs, vref, bits, gain = 100.0, 3.3, 12, 100
ecg = nk.ecg_simulate(duration=10, sampling_rate=1000, heart_rate=72)
ecg = signal.resample_poly(ecg, 1, 10)                 # 1 kHz -> 100 Hz
t = np.arange(ecg.size) / fs
x = ecg + 0.3*np.sin(2*np.pi*0.33*t) + 2.0*np.sin(2*np.pi*60*t)
y = signal.sosfiltfilt(signal.butter(4, 40, "low", fs=fs, output="sos"), x)
y_v = y * 1e-3
code = np.clip(np.round((y_v*gain + vref/2)/vref*(2**bits - 1)), 0, 2**bits - 1)
y_hat = ((code/(2**bits - 1))*vref - vref/2)/gain * 1e3
snr = 10*np.log10(np.var(y) / np.var(y - y_hat))
print(f"SNR {snr:.1f} dB, ENOB {(snr - 1.76)/6.02:.2f} bits")
print("60 Hz alias at 100 Hz:", abs(60 - 100*round(60/100)), "Hz")
```
**Safety.** Low voltage only. Do not attach electrodes to a person. Keep the generator below 5 V peak and use a current-limited supply. Connect the scope ground only to circuit ground.

**Deliverable.** Notebook, `config.yaml`, raw captures, and a one-page chain budget naming the dominant error source and the required filter order.
**Acceptance criteria.** The measured alias is within one FFT bin of the prediction at two sample rates; ENOB is reported for at least two resolutions with the method stated; filter order is justified by a stopband requirement, not by preference; a clipping case is shown and a headroom rule is stated; parameters, seeds, and library versions are recorded.

**Stretch goal.** Add sample-clock jitter with a stated standard deviation and show its effect on a high-frequency component. Compare a SAR model with a delta-sigma model at the same output rate.

**Time.** 2 hours.
### Lab 25.2 — Sensor calibration and Gauge R&R
**Objective.** Calibrate a temperature, pressure, or PPG-derived measurement against a reference, then quantify repeatability and reproducibility with a crossed Gauge R&R and decide whether the measurement system is fit for its tolerance.
**Prerequisites.** Module 14A §4 and §5.9; basic regression and ANOVA.
**Hardware / software.** TMP117 or NTC thermistor; MPRLS or BMP390 pressure sensor; MAX30102 PPG board. Reference: ice bath plus boiling water (corrected for altitude) or a calibrated thermometer; syringe plus manometer or a calibrated pressure gauge; ECG-derived heart rate or a pulse simulator for PPG. Python, pandas, statsmodels, matplotlib.
**Setup.** Write the calibration protocol before taking data: range, five points, ascending and descending, three repeats, randomized order, equilibration time, environmental conditions, and the acceptance tolerance. For the Gauge R&R use 10 parts or levels, 3 operators, 3 trials, randomized and blind. For temperature, set points are a stand-in for production parts; state that limitation.

**Tasks.**

1. Collect calibration data at five points across the range, ascending and descending. Record the reference value, the sensor raw value, the temperature, and the time.
2. Fit `reference = a*raw + b`. Report slope, intercept, residual standard deviation, and the largest residual. Plot residuals against the reference; do not rely on R² alone.
3. Compute hysteresis as the maximum difference between the ascending and descending fits at the same point. Compare it with the tolerance.
4. Build an uncertainty budget: reference uncertainty, repeatability, resolution, drift since calibration, and fit residual. Combine in quadrature and report expanded uncertainty with k = 2.
5. Run the Gauge R&R. Compute the ANOVA table and the variance components using the mean-square relations below.
6. Report `%GRR = 6*sigma_GRR / tolerance * 100` and the number of distinct categories `ndc = 1.41 * sigma_part / sigma_GRR`. Use the AIAG heuristic (under 10% acceptable, 10–30% conditional, over 30% unacceptable) as a starting point, then justify your decision for the clinical or manufacturing context.
7. For PPG, compare heart rate against the reference and report bias and 95% limits of agreement. Do not report SpO2 accuracy from a bench calibration; SpO2 requires calibration against arterial blood gas and a clinical validation.
8. Write a calibration certificate template with range, conditions, uncertainty, validity period, and the operator.

**Variance components.**

| Source | Mean square | Variance component |
|---|---|---|
| Repeatability | MS_e | `sigma_e^2 = MS_e` |
| Operator × part | MS_op | `sigma_op^2 = (MS_op - MS_e)/n` |
| Operator | MS_o | `sigma_o^2 = (MS_o - MS_op)/(p*n)` |
| Part | MS_p | `sigma_p^2 = (MS_p - MS_op)/(o*n)` |

```python
import pandas as pd, statsmodels.formula.api as smf
m = smf.ols("y ~ C(part) + C(operator) + C(part):C(operator)", data=df).fit()
print(smf.stats.anova_lm(m, typ=2))
# Map the mean squares to the table above, then compute GRR and ndc.
```
**Safety.** Hot water stays below 60 °C and is handled with tongs. Keep pressure below 200 kPa and do not use body fluids. Clean the PPG sensor between users. Do not make a clinical claim from a bench calibration.

**Deliverable.** Calibration report with curve, residuals, hysteresis, uncertainty budget, Gauge R&R table, and a fitness-for-purpose decision.

**Acceptance criteria.** The calibration curve, residuals, and hysteresis are reported; the uncertainty budget combines at least four sources and states the method; `%GRR` and `ndc` are computed from variance components, not from a spreadsheet shortcut; the fitness decision is tied to a stated tolerance; the limitation of using set points instead of production parts is explicit.

**Stretch goal.** Repeat the study with two sensor lots and test for a lot-by-operator interaction. Add a 24-hour drift run and a guard-band recommendation.

**Time.** 2.5 hours.
### Lab 25.3 — Firmware, RTOS, watchdog, and secure boot
**Objective.** Build acquisition firmware with an RTOS, a watchdog that is fed only by a health monitor, a defined fail-safe state, and a signed update with rollback. Then inject faults that try to defeat each control.
**Prerequisites.** Module 14A §5.5; C programming; basic digital electronics.
**Hardware / software.** STM32 Nucleo-F446RE, nRF52840 DK, or Arduino Nano 33 BLE Sense; on-board debug probe; logic analyzer; current-limited supply. Sim-only: Renode or Wokwi. Toolchain: STM32CubeIDE with FreeRTOS, nRF Connect SDK with Zephyr, or Arduino FreeRTOS. Secure boot: MCUboot on nRF52840, or an STM32 with TrustZone-M (L5/U5) and a documented secure-boot path. If your board has no secure-boot support, simulate the verification step and document the gap.
**Setup.** Start from the provided template: a timer-triggered ADC, a DMA buffer, a queue, and a UART log. Define the safe state for every output before writing code: actuator off, alarm asserted, output clamped, and a fault record persisted.

**Tasks.**

1. Build a bare-metal superloop that samples at 500 Hz with a timer and DMA. Toggle a GPIO in the ADC callback and measure jitter with the logic analyzer. Replace the timer with `HAL_Delay` and show that the sampling interval collapses.
2. Port to FreeRTOS with three tasks: acquisition (priority 4), processing (priority 3), logging (priority 1). Use a queue of 32 samples and static allocation. Record stack high-water marks and the maximum queue depth.
3. Create a priority-inversion scenario with a mutex shared by processing and logging. Measure the worst-case processing latency with and without priority inheritance.
4. Add an independent watchdog with a 1 s timeout. Only the health task refreshes it, and only after receiving a heartbeat from every other task within 200 ms. Inject a hang with `while(1)` in each task in turn.
5. Define and test the fail-safe state. On a fault, clamp outputs, assert the alarm, write a fault record to nonvolatile memory, and reset. Verify the state with the scope and the logic analyzer.
6. Sign an image and verify that an unsigned or corrupted image is rejected. Enable anti-rollback. Cut power during an update and verify that the device boots a valid image. Test rollback to the previous signed image.
7. Build a fault-injection matrix: task hang, stack overflow, brown-out, corrupted image, wrong version, watchdog disabled, and clock failure. Record detection mechanism, detection time, recovery action, and residual state.
8. Measure boot time, watchdog reset latency, update time, and current consumption in run, idle, and fault states. Record firmware commit, toolchain version, and an SBOM.

**Code.**

```c
void vHealthTask(void *arg) {
  EventBits_t bits;
  for (;;) {
    bits = xEventGroupWaitBits(evt, ALIVE_ACQ | ALIVE_PROC | ALIVE_LOG,
                               pdTRUE, pdTRUE, pdMS_TO_TICKS(200));
    if (bits != (ALIVE_ACQ | ALIVE_PROC | ALIVE_LOG)) {
      log_fault(FAULT_HEARTBEAT);      /* persist before reset */
      enter_safe_state();
      NVIC_SystemReset();              /* IWDG also expires if this hangs */
    }
    HAL_IWDG_Refresh(&hiwdg);
  }
}
```
**Safety.** Use a current-limited supply and an ESD strap. Keep a spare board and a recovery path (BOOT0 or the vendor's recovery mode) before enabling secure boot. Do not lock debug access on the only board you have.

**Deliverable.** Firmware repository, fault-injection matrix, boot and reset logs, and a one-page secure-update description.

**Acceptance criteria.** Sampling jitter is measured and compared with the requirement; the watchdog resets a hung task within twice the configured timeout; an unsigned image is rejected in at least 20 of 20 attempts; power loss during update leaves a bootable device, and rollback is demonstrated; the fail-safe state is verified for every output; versions and the SBOM are recorded.

**Stretch goal.** Add an MPU stack guard and a fault handler that captures the stack frame. Add secure debug locking and verify that the unlock process is audited.

**Time.** 3 hours.
### Lab 25.4 — EMC pre-compliance: ESD, conducted immunity, and electrosurgery interference
**Objective.** Explain how interference couples into a medical signal chain, run safe tabletop experiments on ESD and conducted immunity, model electrosurgery interference, and write a pre-compliance plan that does not claim compliance.
**Prerequisites.** Module 14A §5.6; basic circuit theory; familiarity with the FFT.
**Hardware / software.** Breadboard victim with a 12-bit ADC input, 10 kΩ source impedance, and a 10 cm unshielded lead; TVS diode; ferrite bead; common-mode choke; shielded cable; scope; signal generator; near-field probe (optional). Simulation: LTspice or ngspice, Python, openEMS (optional, steep learning curve). An ESD gun is used only under supervision in a shared lab; otherwise simulate.
**Setup.** Build the victim so that the ADC input, the lead, and the ground return are visible and probeable. Measure the baseline noise floor for 10 s at 1 kS/s with the lead connected and disconnected. Define a pass/fail metric in ADC codes or input-referred microvolts before testing.

**Tasks.**

1. Read the scope of IEC 60601-1-2. Identify the intended environment (professional healthcare or home healthcare) and list the applicable test categories: ESD, radiated RF immunity, conducted RF, electrical fast transients, surges, magnetic fields, and voltage dips. Build a table of category, standard, typical level, and relevance to your device.
2. Characterize the victim: FFT of the baseline, 50/60 Hz and harmonics, noise with the lead open and shorted, and the transfer function from an injected sine to the ADC code.
3. Simulate an IEC 61000-4-2-like ESD pulse into a TVS-protected and an unprotected input. Compare clamp voltage, residual energy at the ADC, and the risk of latch-up. On the bench, use a commercial ESD tester only with supervision and a defined test plan.
4. Inject a 1 kHz–10 MHz sine at 1 Vrms through a coupling network into the lead. Measure the rectified DC offset and the ADC error. Repeat with a ferrite bead, a common-mode choke, and a shielded cable; report insertion loss and residual error for each.
5. Model electrosurgery interference as a 300 kHz–1 MHz burst, amplitude-modulated at 10–100 Hz, entering the AFE. Include a semiconductor junction to show rectification. Measure baseline shift and saturation, then explain blanking, input filtering, shielding, and physical separation as mitigations.
6. Compare mitigations by residual error, cost, board area, and effect on the signal band. Rank them and state which would change the filter or sampling choices your algorithm depends on.
7. Write the pre-compliance plan: limits borrowed from the standard, test setup, instrument calibration, expected margin, in-house versus accredited lab, and redesign triggers. State clearly that this is not a compliance test.

**Code.**

```python
import numpy as np
def esd_pulse(t, ipk=30.0, tr=0.8e-9, td=60e-9):
    """Double-exponential approximation of an IEC 61000-4-2-like current."""
    return ipk * (1 - np.exp(-t / tr)) * np.exp(-t / td)
t = np.linspace(0, 300e-9, 5000)
i = esd_pulse(t)
print(f"peak {i.max():.1f} A at {t[i.argmax()]*1e9:.1f} ns")
```
**Safety.** No mains voltage. Do not radiate RF outside a shielded enclosure or without authorization. ESD can destroy boards and is a hazard to eyes and electronics; use an ESD gun only under supervision. Keep the test area clear of flammable material.

**Deliverable.** Pre-compliance plan plus a measurement report with baseline, injection results, mitigation comparison, and a limitations section.

**Acceptance criteria.** The applicable environment and test categories are identified correctly; at least two coupling mechanisms are demonstrated or modeled with measured or simulated evidence; each mitigation has a quantified effect and a cost note; the report states that it is not a compliance result and names what an accredited lab would add.

**Stretch goal.** Build an openEMS model of the lead and compare the simulated coupling with the bench measurement. Add a near-field scan of a switching regulator.

**Time.** 2 hours.
### Lab 25.5 — Hardware-in-the-loop rig for an ECG or PPG algorithm with fault injection
**Objective.** Build a HIL rig that runs the real algorithm on the target, injects known physiological and fault scenarios through the analog front end, and produces traceable evidence of detection performance, latency, and safe behavior.
**Prerequisites.** Labs 25.1–25.3; a QRS detector or PPG heart-rate algorithm; basic pytest.
**Hardware / software.** DUT: STM32 or Arduino running the algorithm. Signal path: MCP4725 12-bit DAC or a hardware-timer PWM plus RC filter, an attenuator and buffer, and a mid-rail bias to the ADC. Logic analyzer for timing, power monitor for current. Simulation-only path: run the algorithm on a PC as software-in-the-loop and label it SIL, not HIL. A HIL claim requires the algorithm to run on its target in real time.
**Setup.** Write the requirements first: sensitivity, positive predictive value, false alarms per hour, detection latency, behavior on lead-off, and the safe state. Trace each requirement to a hazard. Calibrate the DAC-to-ADC path with DC levels and a sine before running any scenario.

**Tasks.**

1. Define at least 30 scenarios in YAML: normal sinus at 60 and 90 bpm, AF, PVCs, bradycardia, tachycardia, baseline step, 60 Hz interference, EMG noise, motion artifact, lead-off, saturation, dropout, stuck ADC bit, clock drift, and delayed sample. Each scenario states its expected algorithm behavior.
2. Build the signal path and calibrate it. Measure gain error, offset error, and noise floor. Verify that a 1 mV input at the ADC corresponds to the intended code.
3. Write a runner that renders each scenario at 500 Hz and streams it to the DAC using a hardware timer. A `sleep()` loop is not acceptable for timing evidence; use the timer and verify with the logic analyzer.
4. Run each scenario with three seeds. Log raw ADC samples, algorithm output, timestamps, fault flags, and device state. Use a monotonic clock and record the offset to wall-clock time.
5. Inject faults at the electrical and digital levels: open lead (series 1 MΩ), short to mid-rail, rail saturation, stuck bit, dropped sample, corrupted configuration, and delayed command. Prove each fault was applied with a capture, not by assumption.
6. Analyze per scenario: confusion matrix, sensitivity and PPV with confidence intervals, false alarms per hour, time to detect, p99 end-to-end latency, and whether the device entered the safe state.
7. Build a traceability matrix: requirement, risk control, test case, result, evidence file. List the risks that remain untested.
8. Write the HIL report and a short demo script that reproduces one normal and one fault case.

**Code.**

```python
import time, yaml, numpy as np

SCENARIOS = {
    "normal_72":  dict(hr=72, noise=0.02, faults=[]),
    "lead_off":   dict(hr=72, noise=0.02, faults=["open_lead"]),
    "saturation": dict(hr=90, noise=0.02, faults=["rail_saturate"]),
}
for name, cfg in SCENARIOS.items():
    x = render(cfg)                     # float volts at 500 Hz
    for v in x:
        dac.write(encode(v))            # calibrated 12-bit code
        wait_for_timer_tick()           # hardware timer, not sleep
    result = read_dut_log()
    save(name, result, x)
```
**Safety.** No human connection. The "patient" is a resistor network or phantom. Use an isolated, current-limited supply and label the rig "Not for clinical use." Keep the DUT enclosure closed during fault injection.

**Deliverable.** HIL rig, scenario suite, raw logs, traceability matrix, and report.

**Acceptance criteria.** At least 30 scenarios and 3 seeds are run, with raw evidence retained; each injected fault is demonstrated in a capture; p99 latency and false alarms per hour are reported with the number of runs; lead-off and saturation drive the device to the defined safe state; every test maps to a requirement and a risk control, and untested risks are listed.

**Stretch goal.** Close the loop with a simple physiological model that responds to the algorithm output, or add a second modality and test fusion under fault.

**Time.** 3 hours.
### Lab 25.6 — ROS 2, Gazebo, and MuJoCo: frames, URDF, topics, and sensor pipeline
**Objective.** Build a robot model with correct frames, timestamps, and sensor topics; visualize it in RViz2 and Gazebo; replay a rosbag; and diagnose deliberately injected frame errors.
**Prerequisites.** Linux or WSL2 basics; Python; Module 14A §7.5.
**Hardware / software.** No hardware required. Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic, or Ubuntu 22.04 with ROS 2 Humble; `ros_gz`, RViz2, xacro, `tf2_tools`, `joint_state_publisher_gui`; MuJoCo 3.x for comparison.
**Setup.** Create a workspace and a `medarm_description` package. Use REP 103 conventions: x forward, y left, z up, right-handed, quaternions in x, y, z, w order. Decide the parent frame (`base_link`) and the tool frame (`tool0`) before writing the URDF.

**Tasks.**

1. Write an xacro model with `base_link`, `shoulder_link`, `upper_arm`, `forearm`, and `tool0`. Include visual, collision, inertial, joint axis, and joint limits. Run `check_urdf` and fix every warning.
2. Launch `robot_state_publisher` and `joint_state_publisher_gui`. Open the TF tree with `view_frames` and explain which transforms are static and which are dynamic.
3. Add a camera and an IMU in Gazebo with the appropriate plugins. Bridge the topics with `ros_gz_bridge`. Verify message type, `frame_id`, and timestamp on every topic.
4. Write a Python node that listens to `/joint_states`, looks up `tool0` in `base_link` with tf2, and publishes a marker. Measure the delay between the joint state and the marker.
5. Inject three frame bugs: a joint axis with the wrong sign, an origin offset in centimeters instead of meters, and a missing static transform or a mismatched `frame_id`. Diagnose each with `tf2_echo`, `view_frames`, `ros2 topic hz`, and `ros2 topic echo`.
6. Record a rosbag, replay it with `use_sim_time`, and verify that the TF output is reproducible. Explain what changes when the clock is not synchronized.
7. Build a small MJCF model or convert the URDF, then compare joint trajectories under gravity in Gazebo and MuJoCo. Note differences in actuator model, contact, and integrator.
8. Write a one-page frame convention memo covering units, axes, quaternion order, timestamp source, and the parent frame of every sensor.

**Code.**

```xml
<link name="upper_arm">
  <visual><geometry><cylinder length="0.30" radius="0.03"/></geometry></visual>
  <collision><geometry><cylinder length="0.30" radius="0.03"/></geometry></collision>
  <inertial><mass value="0.8"/>
    <inertia ixx="0.006" iyy="0.006" izz="0.0002" ixy="0" ixz="0" iyz="0"/>
  </inertial>
</link>
<joint name="shoulder" type="revolute">
  <parent link="base_link"/><child link="upper_arm"/>
  <origin xyz="0 0 0.10" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
</joint>
```
**Safety.** Simulation only. If the model is later used to drive hardware, add a guard, a hardware e-stop, joint limits, and a low speed cap before enabling actuators.

**Deliverable.** ROS 2 package, launch files, rosbag, TF diagnosis memo, and a comparison note on Gazebo versus MuJoCo.

**Acceptance criteria.** The TF tree is connected and every sensor has a correct `frame_id` and timestamp; three injected frame bugs are found and fixed with evidence; a rosbag replays with reproducible TF output under `use_sim_time`; the Gazebo and MuJoCo comparison names at least two model differences that affect results.

**Stretch goal.** Configure MoveIt 2 for the arm and plan a collision-free pick-and-place. Add a perception node that publishes an object pose in the correct frame.

**Time.** 2.5 hours.
### Lab 25.7 — Robot arm FK, IK, Jacobian, and singularity
**Objective.** Implement forward kinematics, analytical and numerical inverse kinematics, and the geometric Jacobian for a planar arm and a 6-DOF arm. Detect singularities, compare pseudo-inverse and damped least squares, and enforce joint limits.
**Prerequisites.** Lab 25.6; linear algebra; basic calculus.
**Hardware / software.** No hardware required. Optional: a two-link servo arm for physical validation. Python, NumPy, matplotlib, Pinocchio 3.x; MoveIt 2 optional.
**Setup.** Use a planar arm with `L1 = 0.30 m`, `L2 = 0.25 m`, and joint limits of ±150°. For the 6-DOF arm, use a public URDF such as a UR5 or Franka Emika Panda description. State the link lengths, joint limits, and frame conventions in the report.

**Tasks.**

1. Derive the homogeneous transforms for the planar arm and implement forward kinematics. Verify three poses against a hand calculation and record the position error.
2. Implement analytical inverse kinematics with the elbow-up and elbow-down solutions:
   `q2 = ±acos((x^2 + y^2 - L1^2 - L2^2)/(2*L1*L2))`,
   `q1 = atan2(y, x) - atan2(L2*sin(q2), L1 + L2*cos(q2))`.
   Test 1,000 reachable targets and report the mean and maximum position error. Test unreachable targets and confirm that the solver reports failure rather than returning a NaN.
3. Compute the geometric Jacobian and verify it against a central finite difference with `h = 1e-6`. Report the maximum absolute difference.
4. Compute manipulability `w = sqrt(det(J @ J.T))` over a grid of joint angles. Plot the workspace and identify the singular configurations at `q2 = 0` and `q2 = pi`.
5. Implement damped least squares: `dq = J.T @ inv(J @ J.T + lambda^2 * I) @ e`. Sweep `lambda` and compare joint velocities and convergence near the singularity.
6. Extend to the 6-DOF model with Pinocchio: build the model from URDF, compute forward kinematics and the Jacobian, and solve IK numerically with joint limits. Record the end-effector pose error and the number of iterations.
7. Command a Cartesian velocity through `q2 = 0`. Compare the pseudo-inverse and DLS joint velocities, report the condition number, and state the speed-scaling or stop rule a controller should apply.
8. Define a workspace and joint-limit monitor. Show that a Cartesian target outside the workspace is rejected before it reaches the controller.

**Code.**

```python
import numpy as np

def fk(q, L=(0.30, 0.25)):
    return np.array([L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1]),
                     L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])])

def jac(q, L=(0.30, 0.25)):
    return np.array([
        [-L[0]*np.sin(q[0]) - L[1]*np.sin(q[0]+q[1]), -L[1]*np.sin(q[0]+q[1])],
        [ L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1]),  L[1]*np.cos(q[0]+q[1])]])

def dls_ik(q0, target, lam=0.05, iters=200, tol=1e-6):
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        e = np.asarray(target) - fk(q)
        if np.linalg.norm(e) < tol:
            break
        J = jac(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(2), e)
    return q, np.linalg.norm(np.asarray(target) - fk(q))
```
**Safety.** If a physical servo arm is used, keep torque low, guard the workspace, mount the e-stop within reach, and keep hands clear during motion. A software limit is not a substitute for a mechanical stop.

**Deliverable.** Notebook, singularity map, IK benchmark table, and a short memo on the control implication.

**Acceptance criteria.** Forward kinematics is verified against a hand calculation; IK position error is below `1e-6 m` for reachable targets, and unreachable targets fail explicitly; the Jacobian matches the finite-difference result within a stated tolerance; the singularity is demonstrated and mitigated with DLS; joint velocities are compared; joint and workspace limits are enforced before command execution.

**Stretch goal.** Add redundancy resolution for a 7-DOF arm, compare with the MoveIt 2 IK plugin, and add an obstacle to the planning scene.

**Time.** 2 hours.
### Lab 25.8 — Teleoperation latency and safety monitor
**Objective.** Build a master-slave teleoperation task with configurable one-way latency of 0, 50, 100, and 200 ms. Measure task performance and stability, then design and test a safety monitor for force, workspace, speed, and command age.
**Prerequisites.** Labs 25.6 and 25.7; basic statistics.
**Hardware / software.** Simulation only is sufficient: Python, MuJoCo or a 2D plant, pygame or a joystick, pandas, matplotlib. Optional hardware: two servos, potentiometers, an ESP32, and a load cell with HX711. Use a torque-limited supply and a guarded arm.
**Setup.** Define the task: move a tool along a marked path, avoid a no-go zone, and keep contact force below 2 N. Define metrics before running: completion rate, task time, path error, peak force, collisions, safety stops, and workload. Write requirements with IDs.

**Tasks.**

1. Implement a delay buffer in the command path with 0, 50, 100, and 200 ms one-way delay. Add optional jitter and 1% packet loss so that latency is not unrealistically clean.
2. Add a safety monitor in the slave loop: workspace clamp, velocity clamp, force clamp, command-age watchdog at 100 ms, and an e-stop. Log every intervention with a timestamp and the triggering value.
3. Run 10 repetitions per latency condition in randomized order. Record task time, path error, peak force, collisions, stops, and completion. If using multiple participants, record only anonymized task data and obtain consent.
4. Analyze medians and interquartile ranges. Plot performance against latency. Report the false-stop rate and the detection latency for an injected unsafe command (for example, a step to 10 N or a target outside the workspace).
5. Test stability at 200 ms. Look for oscillation, overshoot, and limit cycles. Compare a naive proportional controller with a passivity or impedance approach.
6. Tune lead compensation or motion scaling, and show the trade-off between speed and overshoot. Do not claim statistical significance from a single pilot; report the sample size.
7. Write a safety requirements table: hazard, limit, sensing, response, detection time, and verification. Map the concepts to IEC 80601-2-77, IEC 80601-2-78, and ISO/TS 15066 without claiming compliance.
8. Produce a five-minute demo that shows one successful task and one safety intervention.

**Code.**

```python
from collections import deque
import time

class DelayLine:
    def __init__(self, steps):
        self.q = deque(maxlen=max(1, steps))
    def step(self, x):
        self.q.append(x)
        return self.q[0] if len(self.q) == self.q.maxlen else self.q[-1]

last_cmd = time.monotonic()
if time.monotonic() - last_cmd > 0.10:
    robot.stop()
    state = "SAFE_STOP"          # log the trigger and the command age
```
**Safety.** Simulation is preferred. If hardware is used, the arm is torque-limited, guarded, and equipped with a hardware e-stop that removes actuator power. No hands in the workspace while enabled. A software watchdog is not a safety-rated function.

**Deliverable.** Teleoperation rig, experiment dataset, analysis notebook, safety requirements table, and demo video.

**Acceptance criteria.** Four latency levels are tested with at least 10 repetitions each; performance is reported with a measure of spread, and the sample size is stated; the safety monitor detects an injected unsafe command within a stated time; false stops are counted and the trade-off is discussed; the report distinguishes simulation results from hardware results and makes no clinical claim.

**Stretch goal.** Add bilateral force feedback with a passivity controller, emulate a network with `tc netem`, and collect NASA-TLX workload ratings from consenting participants.

**Time.** 2.5 hours.
### Lab 25.9 — TinyML deployment on an MCU: quantization, latency, energy, and memory
**Objective.** Train a small ECG classifier, quantize it to int8, deploy it to an MCU with LiteRT for Microcontrollers or CMSIS-NN, and measure accuracy, p99 latency, energy per inference, flash, and SRAM on the target.
**Prerequisites.** Module 14B; Lab 25.3; TensorFlow and C++ basics.
**Hardware / software.** nRF52840 DK (256 KB RAM, 1 MB flash), STM32 Nucleo-F446RE (128 KB RAM, 512 KB flash), or Arduino Nano 33 BLE Sense. Power monitor: PPK2 or INA219. Logic analyzer. Software: TensorFlow 2.x, LiteRT for Microcontrollers (formerly TensorFlow Lite for Microcontrollers), CMSIS-NN, STM32Cube.AI or Edge Impulse (optional), WFDB.
**Setup.** Use the MIT-BIH Arrhythmia Database for a normal/PVC/other classifier, or the MIT-BIH Atrial Fibrillation Database for an AF task. Resample to 100 Hz, use 5 s windows (500 samples), and split by record, never by beat or window. Record the license, the record IDs, and the preprocessing. Do not put identifiable data on the device.

**Tasks.**

1. Train a small 1D CNN: three convolution blocks (16, 32, 64 filters), kernel size 5, ReLU, max pooling, global average pooling, and a dense output. Keep it under roughly 50,000 parameters so that it fits the target. Report accuracy, per-class sensitivity, PPV, and calibration on the held-out records.
2. Convert to int8 with a representative dataset of 200 windows stratified by class and record. Inspect per-layer quantization error and the operator list. If Conv1D is unsupported, use Conv2D with a height of one and document the reshape.
3. Integrate the model into the firmware with a static tensor arena. Record the arena size and the high-water mark. Measure flash usage from the map file, not from the model file size.
4. Measure latency with the DWT cycle counter on Cortex-M, or toggle a GPIO and capture it with the logic analyzer. Run 1,000 inferences and report p50, p95, and p99 for cold and warm starts.
5. Measure energy with the PPK2 or INA219. Report joules per inference, average current, and an estimated battery life with the assumptions stated (duty cycle, cell capacity, regulator efficiency).
6. Compare the LiteRT reference kernels with CMSIS-NN on the same target. Report the measured speedup for your workload; do not quote a vendor speedup.
7. Run fault tests: NaN input, all-zero input, saturated input, sensor dropout, and adversarial noise. Verify that the output is valid or that the firmware falls back to a defined safe behavior.
8. Record the model hash, conversion flags, arena size, toolchain versions, board revision, and the exact command used to build the image.

**Code.**

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model.h"

constexpr int kArena = 64 * 1024;
alignas(16) uint8_t arena[kArena];

tflite::MicroMutableOpResolver<6> resolver;
// Add the operators used by the model; names depend on the LiteRT version.
resolver.AddConv2D(); resolver.AddMaxPool2D(); resolver.AddFullyConnected();
resolver.AddRelu();   resolver.AddReshape();  resolver.AddSoftmax();

tflite::MicroInterpreter interp(tflite::GetModel(g_model), resolver, arena, kArena);
interp.AllocateTensors();
// Copy input, interp.Invoke(), read output; check arena used and latency.
```
**Safety.** Low voltage only, with an ESD strap. Do not flash a board that is the only copy of a working image without a recovery path. Use de-identified or synthetic data and do not make a clinical claim from a lab classifier.

**Deliverable.** Reproducible build, deployment report, latency and energy dataset, and a model card with target measurements.

**Acceptance criteria.** Accuracy, sensitivity, and PPV are reported on held-out records; latency is measured on the target at p50, p95, and p99 with the number of runs; flash and SRAM are measured, not estimated from the model file; energy per inference is measured, with the instrument and method named; the accuracy drop after quantization is within a pre-specified margin (for example, 2 percentage points absolute); otherwise the model is rejected or reworked; fallback behavior is tested for at least three fault inputs.

**Stretch goal.** Apply quantization-aware training, compare with post-training quantization, and test whether subgroup performance changes. If an NPU target is available, compare operator support and latency.

**Time.** 2.5 hours.
### Lab 25.10 — Capstone: integrated device or robot evidence package
**Objective.** Integrate at least three labs into one coherent prototype and produce the evidence package a technical reviewer would expect: intended use, requirements, architecture, risk analysis, HIL verification, a fault demo, and an explicit regulatory gap assessment.
**Prerequisites.** Labs 25.1–25.9; at least one real-time target and one simulation.
**Hardware / software.** Team choice. A workable scope is a wearable arrhythmia monitor with a HIL rig, or a two-link robot with a teleoperation safety monitor. Use the same repository, version control, and report template as the earlier labs.
**Setup.** Form teams of two or three. Assign roles: system lead, verification lead, and safety lead. Choose a concept that can be built with the Tier 1 bench or simulation only. Write the claim boundary before building: "educational prototype, not a medical device, no clinical claim."

**Tasks.**

1. Write the intended use, users, environment, and claim boundary. State explicitly what the system is not.
2. Write requirements with IDs, verification method (test, analysis, inspection, demonstration), and acceptance criterion. Build the architecture and the end-to-end timing budget.
3. Build the risk analysis: at least 10 hazards, the hazardous situation, the harm, severity and probability, the risk control, and the verification evidence. Use the ISO 14971 structure.
4. Integrate the system: sensor or simulated sensor, firmware, algorithm, output or actuator. Reuse at least three lab artifacts; document what changed when they were integrated.
5. Run a verification suite of at least 20 tests covering nominal, boundary, fault, latency, power, and safe-state behavior. Link each test to a requirement and a risk control.
6. Demonstrate the system live for five minutes, including one injected fault and the recovery or safe stop. Record the demo.
7. Write the gap assessment: clinical evidence, quality management system, accredited EMC and safety testing, usability validation, cybersecurity lifecycle, biocompatibility, and regulatory pathway. Estimate effort and cost at a high level, with sources for any regulatory claim.
8. Conduct a peer review with another team. Record the findings and show how the final package addresses them.
9. Submit the report, repository, raw evidence, and demo video.

**Deliverable.** A 3,000–4,000 word evidence package plus a repository and a five-minute demo. Sections: intended use; requirements; architecture; risk; software lifecycle; HIL verification; EMC plan; robot safety, if applicable; cybersecurity; limitations and gaps; references.

**Acceptance criteria.** Every requirement has a verification method and a result; at least 20 verification tests are recorded with raw evidence; every risk control has a verification link, and residual risk is stated; the demo includes a fault and a safe recovery or stop; the package states that the prototype is not compliant and is not a medical device; limitations and untested risks are explicit.

**Stretch goal.** Add a second modality or a second robot configuration, run a small formative usability study with consenting colleagues, or submit the package to a student design competition.

**Time.** 3 hours.
---
## 9. Assignment: the graded evidence dossier
Lab 25.10 produces the working prototype and the raw evidence. The graded assignment is the polished dossier that a reviewer who was not in the room can follow. Submit a 3,000–4,000 word report (12 pages plus appendix), a tagged repository, the raw evidence files, and a five-minute demo video.

1. Intended use, users, environment, and claim boundary.
2. Requirements table with IDs, verification method, and acceptance criterion.
3. Architecture, signal chain, interfaces, and end-to-end timing budget.
4. Risk file with at least 10 hazards in ISO 14971 structure, controls, and residual risk.
5. Software lifecycle evidence: IEC 62304 class, SOUP list, unit and integration tests.
6. HIL verification: at least 20 tests, fault injection, and a traceability matrix.
7. EMC pre-compliance plan, and the robot safety case if the system moves.
8. Cybersecurity, usability findings, gap assessment, and references.

Each team member submits an individual contribution statement. No compliance claim may appear anywhere in the dossier.

## 10. Assessment rubric
| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Measurement and signal chain | 12% | Lists components | Predicts aliasing and ENOB | Quantifies the dominant error and headroom |
| Calibration and uncertainty | 10% | Reports a curve | Computes an uncertainty budget | Separates variance components and ties them to tolerance |
| Firmware and fault handling | 12% | Boots and samples | Watchdog, fail-safe, signed image | Rollback, fault matrix, security configuration |
| EMC reasoning | 8% | Names categories | Explains coupling | Quantifies mitigation and margin |
| HIL design and evidence | 14% | Runs nominal tests | Injects faults | Traces to risk, measures p99 and safe state |
| ROS 2 frames and timing | 8% | Model displays | Correct TF and topics | Diagnoses timestamp and frame bugs |
| Kinematics and singularities | 8% | FK works | IK and Jacobian | Singularity analysis and DLS |
| Teleoperation and safety | 10% | Latency demo | Metrics and limits | Detection latency, false stops, traceability |
| TinyML target measurement | 8% | Model runs | Latency and memory | Energy, p99, fallback, accuracy margin |
| Capstone integration | 10% | Coherent demo | Requirements and risk | Full traceability, gaps, peer review |

## 11. Failure modes and red flags
- Simulation-only results presented as target measurements.
- Calibration against an uncalibrated reference.
- Gauge R&R parts that do not span the tolerance.
- A watchdog fed by the task it is meant to supervise.
- Secure boot enabled but the debug port left open.
- An EMC test with no characterized victim or no injected level.
- HIL scenarios that avoid the faults named in the risk file.
- HIL using the training data as test data.
- A TF tree connected in RViz but with wrong timestamps.
- TinyML latency measured on a desktop and reported as target latency.
- A capstone that claims "compliant," "clinical-grade," or "safe" without evidence.
- No version pinning, no raw data hash, and no limitations section.
- A safety monitor that has never been tested with an unsafe command.

## 12. Interview and discussion questions
1. **How do you know a sensor is calibrated?** Hint: traceability, uncertainty, drift, fit residuals, and a reference.
2. **What does Gauge R&R separate?** Hint: repeatability from reproducibility; `%GRR` and `ndc`.
3. **Why can a watchdog fail to protect?** Hint: fed by the hung task, disabled, too long, or resets into an unsafe state.
4. **What does secure boot actually verify?** Hint: signature, root of trust, anti-rollback, debug lock.
5. **Conducted versus radiated immunity?** Hint: coupling path, CDN versus antenna, frequency range.
6. **Why does electrosurgery corrupt an ECG?** Hint: rectification at semiconductor junctions, baseline shift, blanking.
7. **What does HIL prove, and what does it not?** Hint: timing and fault behavior versus human use and clinical benefit.
8. **Why is a TF timestamp as important as `frame_id`?** Hint: extrapolation, latency, `use_sim_time`.
9. **Why does IK fail near a singularity?** Hint: Jacobian rank loss, large joint velocities, damped least squares.
10. **How do you measure p99 latency and energy?** Hint: target cycle counter, many runs, series shunt, burden voltage.

## 13. Key terms
| Term | Definition |
|---|---|
| AFE | Analog front end that amplifies and filters a sensor signal |
| ADC | Analog-to-digital converter |
| ENOB | Effective number of bits after noise and distortion |
| CMRR | Common-mode rejection ratio |
| Gauge R&R | Repeatability and reproducibility study of a measurement system |
| Repeatability | Variation under one operator, one setup, short time |
| Reproducibility | Variation across operators or setups |
| Watchdog | Timer that resets the system if not refreshed |
| Secure boot | Cryptographic verification of firmware before execution |
| Anti-rollback | Prevention of installing an older, vulnerable image |
| HIL | Hardware-in-the-loop: target runs in real time with simulated inputs |
| SIL | Software-in-the-loop: algorithm runs on a host, not the target |
| URDF | XML description of a robot's links, joints, and geometry |
| TF | ROS 2 coordinate-frame transform system |
| Jacobian | Map from joint velocities to end-effector velocities |
| Singularity | Configuration where the Jacobian loses rank |
| DLS | Damped least squares, a singularity-robust IK method |
| Pre-compliance | Early in-house EMC testing that is not a compliance test |

## 14. Sources and further reading
All URLs accessed 24 October 2026 unless otherwise stated. Standards are paywalled; use the official catalogue record and the current edition.

### Standards and regulation
- IEC 60601-1, IEC 60601-1-2, IEC 60601-2-2. https://webstore.iec.ch/
- IEC 62304, IEC 62366-1. https://webstore.iec.ch/
- IEC 80601-2-77. https://webstore.iec.ch/en/publication/29933
- IEC 80601-2-78. https://webstore.iec.ch/en/publication/69618
- IEC 81001-5-1. https://webstore.iec.ch/en/publication/63293
- ISO 14971. https://www.iso.org/standard/72704.html
- ISO 13485. https://www.iso.org/standard/59752.html
- ISO/TS 15066 and ISO 10218-1:2025. https://www.iso.org/
- FDA, cybersecurity. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- FDA, PCCP. https://www.fda.gov/media/166704/download
- FDA, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
- EU MDR. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- EU AI Act. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

### Robotics and simulation
- ROS 2 documentation. https://docs.ros.org/
- Gazebo documentation. https://gazebosim.org/docs
- MuJoCo documentation. https://mujoco.readthedocs.io/
- Pinocchio. https://stack-of-tasks.github.io/pinocchio/
- MoveIt 2. https://moveit.picknik.ai/
- ros_gz bridge. https://github.com/gazebosim/ros_gz

### Embedded, TinyML, and instruments
- LiteRT for Microcontrollers. https://developers.google.com/edge/litert/microcontrollers/overview
- CMSIS-NN. https://github.com/ARM-software/CMSIS-NN
- FreeRTOS. https://www.freertos.org/
- Zephyr. https://docs.zephyrproject.org/
- MCUboot. https://docs.mcuboot.com/
- Renode. https://renode.io/
- Wokwi. https://wokwi.com/
- sigrok/PulseView. https://sigrok.org/wiki/PulseView
- Nordic Power Profiler Kit II. https://www.nordicsemi.com/Products/Development-hardware/Power-Profiler-Kit-2
- Joulescope. https://www.joulescope.com/
- TI INA219. https://www.ti.com/product/INA219
- Analog Devices AD8232. https://www.analog.com/en/products/ad8232.html
- openEMS. https://openems.de/
- ngspice. https://ngspice.sourceforge.io/

### Methods and data
- NeuroKit2. https://neurokit2.readthedocs.io/
- WFDB. https://physionet.org/content/wfdb/
- MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/
- AIAG, Measurement Systems Analysis. https://www.aiag.org/
- JCGM 100:2008, Guide to the Expression of Uncertainty in Measurement. https://www.bipm.org/en/committees/jc/jcgm/publications

### Verification notes
- `[VERIFY]` ROS 2 2026 LTS release name, Gazebo pairing, and MuJoCo URDF import path: check the current ROS 2 and Gazebo release pages before installing.
- `[VERIFY]` LiteRT for Microcontrollers API names and STM32Cube.AI/ST Edge AI naming: check the current vendor documentation; the APIs have changed across releases.
- `[VERIFY]` Recall identifiers and dates for the Versius, MIRA, and EPIONE events: query the FDA recall database; the entries are cited as public narratives, not as lab findings.
- `[VERIFY]` openEMS maturity and version: confirm the current release and example models before assigning it.

> **Previous:** [`23_specialty_electives.md`](23_specialty_electives.md)
> **Next:** [`25_trial_operations_hta_global_regulatory.md`](25_trial_operations_hta_global_regulatory.md)
