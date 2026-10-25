# Module 14A — Medical Hardware, Sensors, Chips, and Robotics

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../14A_medical_hardware_sensors_chips_robotics.md`](../../14A_medical_hardware_sensors_chips_robotics.md)
> **Folder:** `modules/14A_hardware/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01A |
| Unlocks | 14B, 24 |
| Assessment | Device data and evidence plan (Week 24) |
| Survey file | `14A_medical_hardware_sensors_chips_robotics.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_the-signal-chain.md`](01_the-signal-chain.md) | The Signal Chain | Sensor to ADC to processor to display; where information is lost. | 1,200-1,800 | planned |
| 2 | [`02_sensors.md`](02_sensors.md) | Sensors | Electrodes, optical, pressure, flow, chemical, and MEMS sensing principles. | 1,200-1,800 | planned |
| 3 | [`03_analog-front-ends.md`](03_analog-front-ends.md) | Analog Front Ends | Amplification, filtering, common-mode rejection, and noise budgets. | 1,200-1,800 | planned |
| 4 | [`04_adcs-and-signal-conditioning.md`](04_adcs-and-signal-conditioning.md) | ADCs and Signal Conditioning | Resolution, sample rate, aliasing, and quantization error. | 1,200-1,800 | planned |
| 5 | [`05_mcus-dsps-and-fpgas.md`](05_mcus-dsps-and-fpgas.md) | MCUs, DSPs, and FPGAs | Architecture choices, determinism, and real-time constraints. | 1,200-1,800 | planned |
| 6 | [`06_socs-npus-and-gpus.md`](06_socs-npus-and-gpus.md) | SoCs, NPUs, and GPUs | Heterogeneous compute for on-device inference. | 1,200-1,800 | planned |
| 7 | [`07_power-and-thermal.md`](07_power-and-thermal.md) | Power and Thermal | Battery budget, leakage, thermal throttling, and patient safety. | 1,200-1,800 | planned |
| 8 | [`08_firmware-and-rtos.md`](08_firmware-and-rtos.md) | Firmware and RTOS | Scheduling, watchdogs, secure boot, and update mechanisms. | 1,200-1,800 | planned |
| 9 | [`09_medical-equipment-classes.md`](09_medical-equipment-classes.md) | Medical Equipment Classes | Monitoring, imaging, therapeutic, laboratory, and implantable systems. | 1,200-1,800 | planned |
| 10 | [`10_robotics-kinematics.md`](10_robotics-kinematics.md) | Robotics Kinematics | Frames, forward and inverse kinematics, Jacobians, and singularities. | 1,200-1,800 | planned |
| 11 | [`11_robotics-control.md`](11_robotics-control.md) | Robotics Control | PID, impedance, admittance, force control, and safety envelopes. | 1,200-1,800 | planned |
| 12 | [`12_surgical-and-rehab-robotics.md`](12_surgical-and-rehab-robotics.md) | Surgical and Rehabilitation Robotics | Teleoperation, haptics, registration, and assist-as-needed control. | 1,200-1,800 | planned |
| 13 | [`13_standards-and-emc.md`](13_standards-and-emc.md) | Standards and EMC | IEC 60601-1, collateral standards, 60601-1-2 EMC, and 80601-2-77/-2-78. | 1,200-1,800 | planned |
| 14 | [`14_reliability-and-sterilization.md`](14_reliability-and-sterilization.md) | Reliability and Sterilization | MTBF, accelerated life testing, biocompatibility, and reprocessing. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Signal Chain ->
02 Sensors ->
03 Analog Front Ends ->
04 ADCs and Signal Conditioning ->
05 MCUs, DSPs, and FPGAs ->
06 SoCs, NPUs, and GPUs ->
07 Power and Thermal ->
08 Firmware and RTOS ->
09 Medical Equipment Classes ->
10 Robotics Kinematics ->
11 Robotics Control ->
12 Surgical and Rehabilitation Robotics ->
13 Standards and EMC ->
14 Reliability and Sterilization
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 14 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

