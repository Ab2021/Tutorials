# Module 37 â€” Sensor Fusion, IoT and Real-Time Perception

> **Part:** [Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems](../INDEX.md)  
> **Prerequisites:** 09,17,35  
> **Purpose:** Fuse cameras, IMU, lidar, radar, audio, and telemetry while handling synchronisation, calibration, and uncertainty.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | sensor characteristics | [authored](Cheat_Sheets/02_Sensor_Time_Sync_and_Calibration.md) |
| 2 | time sync and calibration | [authored](Cheat_Sheets/02_Sensor_Time_Sync_and_Calibration.md) |
| 3 | Kalman and Bayesian filters | planned |
| 4 | edge telemetry | planned |
| 5 | perception latency | planned |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [Sensor Time Synchronization and Calibration](Cheat_Sheets/02_Sensor_Time_Sync_and_Calibration.md) — clock, frame, uncertainty and recalibration contracts; visible source inventory only.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Sampling, ECG anatomy, filtering, QRS/HRV, PPG/SpO2, FFT/wavelets, IMU and biomedical datasets](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-signals) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.