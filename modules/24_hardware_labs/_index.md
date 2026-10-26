# Module 24 — Hardware and Robotics Labs

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
> **Folder:** `modules/24_hardware_labs/`
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
| Prerequisites | 14A, 14B |
| Unlocks | none |
| Assessment | Graded evidence dossier |
| Survey file | `24_hardware_robotics_labs.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_safety-and-standards.md`](01_safety-and-standards.md) | Safety and Standards | Lab safety, electrical safety, and the compliance boundary. | 1,200-1,800 | complete |
| 2 | [`02_bench-and-simulation-stack.md`](02_bench-and-simulation-stack.md) | Bench and Simulation Stack | Instruments, dev boards, simulators, and software. | 1,200-1,800 | complete |
| 3 | [`03_lab-conventions.md`](03_lab-conventions.md) | Lab Conventions | Reproducibility, logging, calibration, and reporting. | 1,200-1,800 | complete |
| 4 | [`04_sensor-and-signal-labs.md`](04_sensor-and-signal-labs.md) | Sensor and Signal Labs | ECG front end, PPG, temperature, and noise measurement. | 1,200-1,800 | complete |
| 5 | [`05_firmware-labs.md`](05_firmware-labs.md) | Firmware Labs | RTOS scheduling, watchdogs, secure boot, and update. | 1,200-1,800 | complete |
| 6 | [`06_emc-labs.md`](06_emc-labs.md) | EMC Labs | Emissions, immunity, and pre-compliance testing. | 1,200-1,800 | complete |
| 7 | [`07_hil-labs.md`](07_hil-labs.md) | Hardware-in-the-Loop Labs | Closed-loop testing, fault injection, and latency measurement. | 1,200-1,800 | complete |
| 8 | [`08_robotics-labs.md`](08_robotics-labs.md) | Robotics Labs | Kinematics, control, registration, and safety envelopes. | 1,200-1,800 | complete |
| 9 | `09_evidence-dossier.md` | Evidence Dossier | Turning lab results into a defensible evidence package. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Safety and Standards ->
02 Bench and Simulation Stack ->
03 Lab Conventions ->
04 Sensor and Signal Labs ->
05 Firmware Labs ->
06 EMC Labs ->
07 Hardware-in-the-Loop Labs ->
08 Robotics Labs ->
09 Evidence Dossier
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 1 |
| draft | 0 |
| complete | 8 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

