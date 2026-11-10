# Part 07 exact local coverage ledger

Established 2026-11-08 before authoring. This ledger preserves indexed commitments, not a claim of completion. Authority: [inventory](../../Templates/directory.txt), [crosswalk](../00_SOURCE_TOPIC_CROSSWALK.md), [focused source syllabus](../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md), [standards](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md).

## Baseline accounting

Seven module INDEX files contain 35 original topic rows. Seven module sheet collections contain 21 original reference-family rows. Seven module case collections contain 14 original case rows. Module 37 additionally has two existing short drafts: ECG sampling and aliasing, and sensor synchronization/calibration. Both require substantive expansion, not replacement or completion by relabeling. Eight sheet collections (seven modules and this part) require distinct final Common Issues artifacts with at least 25 complete issues each. Part-level reference families and candidate threads repeat the module commitments; retain every named row as navigation, not 35 extra completions.

## Ordered original topic ledger

| Home | Original topics in INDEX order | Original reference commitments | Original case commitments |
|---|---|---|---|
| [35](35_edge_ai_embedded_systems_and_tinyml/INDEX.md) | embedded compute; TinyML lifecycle; sensor data pipelines; RTOS and scheduling; power and memory budgets | TinyML deployment checklist; MCU model budget; sensor sampling guide | Wearable activity classifier; industrial vibration monitor |
| [36](36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md) | ONNX/TFLite/TensorRT runtimes; PTQ and QAT; compiler optimisation; benchmarking; OTA deployment | edge runtime selection; quantisation playbook; benchmark protocol | Quantise an inspection model; compare edge runtimes on a Pi-class device |
| [37](37_sensor_fusion_iot_and_realtime_perception/INDEX.md) | sensor characteristics; time sync and calibration; Kalman and Bayesian filters; edge telemetry; perception latency | sensor fusion patterns; calibration checklist; telemetry schema | Warehouse asset tracking; multisensor worker-safety system |
| [38](38_robotics_ros2_kinematics_and_control/INDEX.md) | ROS 2 graph; kinematics and transforms; motion planning; control fundamentals; robot safety lifecycle | ROS 2 concepts; coordinate-frame guide; robot safety checklist | Robot arm pick-and-place cell; mobile robot fleet control design |
| [39](39_robot_perception_navigation_manipulation_and_vla/INDEX.md) | SLAM and localisation; navigation stacks; grasping and manipulation; VLA policies; behaviour-tree integration | robot autonomy stack; navigation failure modes; VLA evaluation card | Autonomous inventory robot; bin-picking system evaluation |
| [40](40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md) | simulation environments; synthetic data pipelines; domain randomisation; digital twins; sim-to-real validation | sim-to-real checklist; synthetic-data quality rubric; scenario matrix | Synthetic warehouse dataset; validate a navigation policy before field trial |
| [41](41_edge_robotics_safety_fleet_management_and_operations/INDEX.md) | functional safety concepts; fleet orchestration; remote operations; robot observability; OTA and security | robot fleet SLOs; OTA rollout checklist; remote-assistance policy | Fleet rollout readiness review; robot near-miss incident response |

## Source additions, separate from original rows

| Source group | Home | Exact visible technical commitments | Supporting entries | Inventory range |
|---|---|---:|---:|---|
| edge-arm-firmware | 35 | 15 | container and README | 1094–1110 |
| edge-compression | 36 | 15 | container and README | 1061–1077 |
| edge-runtimes | 36 | 14 | container and README | 1078–1093 |
| edge-compilers | 36 | 16 | container and README | 1111–1128 |
| edge-signals | 37 | 15 | container and README | 1027–1043 |

There are 75 visible source-specific technical commitments and 10 supporting source entries across these five focused groups. The 35 original course topics and 75 source-specific commitments overlap substantively; do not sum them into a count of distinct completed lessons. Each module's exact source-to-artifact mapping must expose all named methods and limitations. The original inventoried source bodies are not supplied by directory.txt. The similarly named collapsed compression directory remains unresolved in its shared crosswalk home; no hidden content is inferred here. Modules 38–41 have course-topic commitments even though this inventory adds no exact original source-file bodies for their topics.

## Completion interpretation

An authored file remains editorial draft until independent review. File existence, line count, a source title, or a planning checklist does not establish publication. Case intervention stage and evidence labels remain independent. Hardware timing, target memory use, clinical validity, closed-loop safety and field outcomes require real evidence; no authoring operation supplies them. Module collection indexes carry exact local manifests and source mappings. Part-level indexes aggregate without double-counting.

## 2026-11-09 local reconciliation checkpoint

| Artifact | Observed state after navigation reconciliation | What this establishes | What remains open |
|---|---|---|---|
| `Cheat_Sheets/INDEX.md` | 11-section part collection index; links only to existing module indexes/files and the actual part issue artifact | Part-level ladder, lookup, paths, glossary, card directory, issue locator and source boundary are navigable | Planned module sheets and local issue files are not authored by an index |
| `Case_Studies/INDEX.md` | Seven-section part collection index; 14 original module case commitments retained as `PLAN-P07-*` rows | Every original case has a decision/pattern/evidence-stage limitation record without a false `CASE` ID | No individual part/module case is promoted by catalog work |
| `Cheat_Sheets/99_35_Common_Issues.md` | Existing 1,637-line draft declares 35 cross-boundary issues | A part-level draft troubleshooting artifact exists | Independent technical review and the seven distinct module-local issue artifacts remain open |
| M35 acquisition sheet | Existing 1,001-line draft | One authored individual sheet covers embedded acquisition/firmware/TinyML foundation | Remaining M35 reference families, cases and local issues remain open |
| M37 ECG sheet | Existing 1,272-line draft | One authored individual sheet covers sampling/aliasing source commitments | Time/calibration draft is short; other M37 sheets, cases and local issues remain open |

**Verification boundary:** line counts are inventory observations only, not quality, source, target or publication evidence. Arithmetic and navigation checks must be rerun after further authored-file changes; this checkpoint does not claim completion.
