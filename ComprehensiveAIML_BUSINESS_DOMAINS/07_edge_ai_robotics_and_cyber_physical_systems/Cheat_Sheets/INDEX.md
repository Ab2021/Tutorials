# Part 07 Cheat Sheets — Edge AI, Robotics and Cyber-Physical Systems

## 1. Overview

This is the cross-module navigation collection for resource-bounded sensing, inference, estimation, motion and fleet operation. It is useful for a learner who must identify the first violated interface contract rather than treat a physical-system failure as a model-only problem. Module collections own their focused teaching artifacts; this collection owns the cross-boundary diagnostic reference. Readiness means tracing a unit, timestamp, frame, authority and version from acquisition to action, while preserving the distinction between an illustrative calculation and measured target evidence.

**Scope boundary:** no artifact here certifies an electrical design, clinical use, robot safety function, hardware benchmark, or production deployment. Original-source bodies named by the source register were unavailable; visible names define planned coverage only.

## 2. Topic ladder

| Level | File or home | Testable outcome | Prerequisite | Key method | Editorial state |
|---|---|---|---|---|---|
| Foundation | [Module 35 collection](../35_edge_ai_embedded_systems_and_tinyml/Cheat_Sheets/INDEX.md) | Budget acquisition, firmware timing and TinyML memory. | rates, C ownership | buffer and deadline invariants | mixed draft/planned |
| Core | [Module 36 collection](../36_edge_deployment_compilers_quantisation_and_runtime/Cheat_Sheets/INDEX.md) | Select a supported conversion/runtime/compression path. | Module 35 | numerical equivalence and liveness | planned |
| Core | [Module 37 collection](../37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/INDEX.md) | Check sampling, time, frame and uncertainty before fusion. | signals and units | capture-time contract | mixed draft/planned |
| Core | [Module 38 collection](../38_robotics_ros2_kinematics_and_control/Cheat_Sheets/INDEX.md) | Connect graph, transforms, planning and control safely. | Modules 35, 37 | transform/control invariant | planned |
| Advanced | [Module 39 collection](../39_robot_perception_navigation_manipulation_and_vla/Cheat_Sheets/INDEX.md) | Select perception-to-action boundaries. | Module 38 | confidence-to-action gate | planned |
| Advanced | [Module 40 collection](../40_robot_simulation_synthetic_data_and_sim_to_real/Cheat_Sheets/INDEX.md) | Design a transfer validation ladder. | Modules 37, 39 | reality-gap test | planned |
| Advanced | [Module 41 collection](../41_edge_robotics_safety_fleet_management_and_operations/Cheat_Sheets/INDEX.md) | Operate releases, assistance and incident recovery. | Modules 36, 38, 39 | authority/rollback contract | planned |
| Final | [35 Common Issues](99_35_Common_Issues.md) | Diagnose an observed cross-boundary symptom. | all module concepts | discriminating evidence | draft; 35 issues |

## 3. Quick topic lookup

| Question | Artifact | Exact section or use condition | Failure warning |
|---|---|---|---|
| Can the target carry a sensor stream? | [M35 acquisition](../35_edge_ai_embedded_systems_and_tinyml/Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) | §§4–5; frame and serial assumptions stated | baud is not payload bytes/s |
| Can a sampled signal support a frequency claim? | [M37 sampling](../37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/01_ecg_sampling_and_aliasing.md) | §§4–5; analog filter and uniform sampling stated | post-ADC filtering cannot undo aliasing |
| Are two observations fusible? | [M37 time/calibration draft](../37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/02_Sensor_Time_Sync_and_Calibration.md) | clock, frame, unit and uncertainty declared | receive time is not capture time |
| Why does a target differ from host inference? | [Common Issues](99_35_Common_Issues.md#8-failure-modes--edge-cases) | issues 006–010 | checksum alone does not establish semantics |
| Which boundary should be diagnosed first? | [Common Issues](99_35_Common_Issues.md#2-concept-map--mental-model) | trace earliest violated contract | a clean later trace cannot repair stale input |

## 4. Reading paths — three options

**Beginner:** Module 35 collection → Module 37 sampling sheet → Common Issues Sections 1–4. Assumes dimensional analysis and basic sampled-signal vocabulary; stop before closed-loop claims.

**Practitioner:** Module 35 → Module 36 → Module 37 → Module 38 collection indexes → Common Issues Sections 5–9. The module 36 and 38 focused files remain planned; their indexes are navigation and scope, not substitutes for authored lessons.

**Advanced:** Follow the practitioner path, then Modules 39–41 and the full Common Issues artifact. Add a module case only after its collection has an authored case; currently the case catalogs are plans.

## 5. Prerequisite skills

- Convert seconds, hertz, bytes, joules, radians and frame transforms without conflating a rate with a count.
- State event, capture, receive and processing timestamps separately.
- Read a state-machine or queue trace; identify owner, authority and recovery transition.
- Distinguish an estimate, a simulated result, a target measurement and a safety claim.
- Use the actual device/runtime/framework documentation for a target-specific assertion.

## 6. Exclusions and handoffs

This collection does not replace a hardware manual, a control-theory text, a safety lifecycle analysis, a clinical signal interpretation protocol, a cybersecurity assessment or target hardware testing. Compression and runtimes hand off to Module 36; multi-sensor estimation to Module 37; transforms/control to Module 38; autonomy behavior to Module 39; simulator fidelity to Module 40; operational safety and rollout evidence to Module 41. Planned homes remain planned until a linked individual artifact exists.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units/type | Domain or warning | First appearance |
|---|---|---|---|---|
| `f_s` | sample rate | samples/s | positive; actual jitter separately bounded | [M35 sheet](../35_edge_ai_embedded_systems_and_tinyml/Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) |
| `t_capture` | physical acquisition timestamp | time | not interchangeable with receive time | [M37 calibration](../37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/02_Sensor_Time_Sync_and_Calibration.md) |
| `T_A_B` | transform mapping coordinates in frame B to frame A | rigid transform | direction and units must be explicit | planned Module 38 |
| `C` / `D` | execution demand / deadline | s | worst-case assumptions required | [M35 sheet](../35_edge_ai_embedded_systems_and_tinyml/Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) |
| `P` | power | W | state and duty cycle defined | [M35 sheet](../35_edge_ai_embedded_systems_and_tinyml/Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) |
| freshness | age at the consumer’s decision point | s | needs a clock-domain definition | [Common Issues](99_35_Common_Issues.md) |
| authority lease | bounded permission to issue an action | state/interval | never inferred from connectivity | [Common Issues](99_35_Common_Issues.md) |

## 8. Quick Reference Card directory

| File/card | Core rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| [M35 §12](../35_edge_ai_embedded_systems_and_tinyml/Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md#12-quick-reference-card) | publish complete immutable blocks | DMA completion semantics known | sequence/time are trustworthy | ownership is implicit |
| [M37 sampling §12](../37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/01_ecg_sampling_and_aliasing.md#12-quick-reference-card) | protect before sampling | analog bandwidth bounded | filter/model match target signal | using it as clinical evidence |
| [Part §12](99_35_Common_Issues.md#12-quick-reference-card) | contain physical risk, then discriminate causes | hazard response is specified | local gate can act without cloud | generic timeout as safe state |

## 9. Common Issues locator

The final local troubleshooting artifact is [99_35_Common_Issues.md](99_35_Common_Issues.md), with 35 complete, distinct cross-boundary teaching issues. It is **draft**, not independently reviewed. It uses `VIOLATED_ASSUMPTION`, `NUMERICAL_EDGE`, `SCALE_FAILURE`, and `CONCEPT_CONFUSION`; all scenarios are labeled hypothetical unless a specific source says otherwise. Module-local Common Issues artifacts are still required separately and are not replaced by this file.

## 10. Sources, versions and maintenance

| Artifact | Evidence/status | Source basis | Recheck trigger | Open limitation |
|---|---|---|---|---|
| M35 acquisition | draft; DERIVED/ILLUSTRATIVE/OPEN | visible source inventory and cited primary links in file | target/configuration change | no target execution recorded |
| M37 sampling | draft; DERIVED/ILLUSTRATIVE/SOURCED where cited | PhysioNet dataset reference plus visible inventory | sample rate/filter/intended use change | no clinical validation |
| M37 calibration | short draft; OPEN | visible inventory only | clock/frame/calibration change | incomplete individual-file contract |
| Part Common Issues | draft; DERIVED/ILLUSTRATIVE | visible inventory and source plan | boundary contract or release process change | no incident reproduction |

The visible source inventory and focused source plan are planning provenance, not proof that original lesson bodies were available. Record source version, precise claim and review date before promoting a target-specific fact.

## 11. Collection acceptance checklist

- [x] Eleven collection-index sections are present and every existing link targets a real artifact.
- [x] The cross-boundary troubleshooting artifact has 35 complete issue entries and retains all 12 required headings.
- [ ] Every planned module reference family is an authored individual sheet with 12 sections and 1,000 substantive lines.
- [ ] Every module collection has a distinct local Common Issues artifact with at least 25 complete issues.
- [ ] Target hardware, safety, source and empirical claims have independent review.
- [ ] Case artifacts and sheet cross-links are reconciled after actual case authorship.
