# Cross-Domain Cheat Sheets

## 1. Overview

This collection teaches cross-domain gates for system sizing, release, agent authorization, edge timing, evaluation, and investment comparison. It assumes explicit units, rates, service boundaries, and basic probability. Readiness means a reader can reconstruct an accounting boundary, compute the stated illustrative oracle, identify a disqualifying assumption, and choose a bounded fallback. The source-tree names consulted through [directory.txt](../../Templates/directory.txt) are syllabus provenance, not evidence of any architecture or outcome. Individual files are draft teaching artifacts; their illustrative quantities are not measured deployments.

## 2. Topic ladder

| Level | ID | Sheet | Testable outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|---|
| 1 | 01 | [AI system design review](01_AI_System_Design_Review.md) | calculate capacity, backlog, storage, recovery | units/rates | conservation boundary | draft |
| 2 | 02 | [Model-to-production launch checklist](02_Model_to_Production_Launch_Checklist.md) | select compatible release/rollback gates | 01 | error budget | draft |
| 2 | 03 | [Agent risk and control matrix](03_Agent_Risk_and_Control_Matrix.md) | bind exact action approval | 01 | authority invariant | draft |
| 2 | 04 | [Edge and robotics deployment gate](04_Edge_and_Robotics_Deployment_Gate.md) | reject infeasible timing/power design | 01 | cycle budget | draft |
| 3 | 05 | [AI evaluation scorecard](05_AI_Evaluation_Scorecard.md) | choose denominator/comparator/guardrail | 01–04 | estimand | draft |
| 3 | 06 | [AI investment and vendor assessment](06_AI_Investment_and_Vendor_Assessment.md) | compare cost per useful outcome | 01–05 | included-work ledger | draft |
| Final | 07 | [25 Common Issues](07_25_Common_Issues.md) | distinguish and verify 25 failure patterns | 01–06 | diagnostic evidence | draft |

## 3. Quick topic lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| Can workload fit? | [01 §4–5](01_AI_System_Design_Review.md#4-core-formulas-with-full-derivation) | rates and boundary known | average hides burst |
| Can release recover? | [02 §7–8](02_Model_to_Production_Launch_Checklist.md#7-comparison--selection-table) | state compatibility known | code rollback is incomplete |
| May agent act? | [03 §4, §8](03_Agent_Risk_and_Control_Matrix.md#4-core-formulas-with-full-derivation) | exact approval exists | capability is not permission |
| Is device feasible? | [04 §5, §9](04_Edge_and_Robotics_Deployment_Gate.md#5-worked-numerical-example) | tail/deadline modeled | mean latency is insufficient |
| Does score justify decision? | [05 §4–5](05_AI_Evaluation_Scorecard.md#4-core-formulas-with-full-derivation) | denominator/comparator set | offline metric is not impact |
| Is vendor comparison valid? | [06 §7, §9](06_AI_Investment_and_Vendor_Assessment.md#7-comparison--selection-table) | included work/date defined | unit price is incomplete |
| What failure pattern applies? | [07 §8](07_25_Common_Issues.md#8-failure-modes--edge-cases) | observable signal available | symptom is not cause |

## 4. Reading paths — three options

- **Beginner:** 01 → 02 → 03. Objective: express a bounded decision and recovery path. Assumes unit arithmetic; 04 is an optional physical-systems extension.
- **Practitioner:** 01 → 02 → 03 → 04 → 05 → 06. Objective: connect sizing, release, authority, device, evaluation, and economics without collapsing boundaries.
- **Advanced:** 01 → 06 → 07, with all relevant earlier sheets. Objective: defend a decision under changed constraint and diagnose a contradiction. Optional extensions are the focused module indexes named below.

## 5. Prerequisite skills

- Unit conversion, rates, counts, percentages, and dimensional checks; see [foundations](../01_foundations_computing_and_problem_solving/INDEX.md).
- Model lifecycle, observability, and reliability vocabulary; see [platforms and production](../05_ai_platforms_data_systems_and_production_engineering/INDEX.md).
- Sensor/control timing before Sheet 04; see [edge modules](../07_edge_ai_robotics_and_cyber_physical_systems/INDEX.md).
- Agent tool contracts and approvals before Sheet 03; visible source navigation: [directory.txt](../../Templates/directory.txt#L63).

## 6. What this collection does NOT cover

It does not replace detailed GPU kernels, modality objectives, actuarial methods, sector workflows, legal applicability, vendor benchmark verification, or organization-specific architecture. Those handoffs remain in [source coverage plans](../00_AUDIT_COVERAGE_MATRIX.md). The visible sovereign-edge filename prefix is incomplete; no hidden contents are inferred.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units/type | Domain | First use |
|---|---|---|---|---|
| lambda | arrival rate | events/s scalar | nonnegative, stated window | [01](01_AI_System_Design_Review.md) |
| mu | completed service rate | events/s scalar | nonnegative, same boundary | [01](01_AI_System_Design_Review.md) |
| B | backlog | events count | nonnegative | [01](01_AI_System_Design_Review.md) |
| t | elapsed time | seconds scalar | nonnegative | [01](01_AI_System_Design_Review.md) |
| E | allowed bad events | events count | explicit eligibility/window | [02](02_Model_to_Production_Launch_Checklist.md) |
| A | authorized actions | action count/set | exact approval context | [03](03_Agent_Risk_and_Control_Matrix.md) |
| T | cycle time | milliseconds scalar | defined control loop | [04](04_Edge_and_Robotics_Deployment_Gate.md) |
| p | observed proportion | dimensionless scalar | declared population | [05](05_AI_Evaluation_Scorecard.md) |
| C_useful | included cost/useful outcome | currency/outcome | dated workload basis | [06](06_AI_Investment_and_Vendor_Assessment.md) |

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Critical assumption | Avoid |
|---|---|---|---|---|
| [01 §12](01_AI_System_Design_Review.md#12-quick-reference-card) | dimensionally consistent sizing | shared boundary | rates defined | tail claim from mean |
| [02 §12](02_Model_to_Production_Launch_Checklist.md#12-quick-reference-card) | compatible release unit | reversible state | migration tested | code-only rollback |
| [03 §12](03_Agent_Risk_and_Control_Matrix.md#12-quick-reference-card) | exact-action approval | signed context | expiry checked | capability equals permission |
| [04 §12](04_Edge_and_Robotics_Deployment_Gate.md#12-quick-reference-card) | enforce cycle budget | worst-case modeled | safe state defined | offline accuracy=safety |
| [05 §12](05_AI_Evaluation_Scorecard.md#12-quick-reference-card) | comparator plus guardrails | valid denominator | population fixed | causal claim from before/after |
| [06 §12](06_AI_Investment_and_Vendor_Assessment.md#12-quick-reference-card) | cost/useful outcome | included work known | price date stated | quote-only comparison |
| [07 §12](07_25_Common_Issues.md#12-quick-reference-card) | diagnose before mitigation | observable signal | competing causes listed | symptom as cause |

## 9. Common Issues locator

[07_25_Common_Issues.md](07_25_Common_Issues.md) indexes 25 HYPOTHETICAL teaching entries, ISSUE-001 through ISSUE-025, across all four categories. Their required distinct investigations have not been reviewed to completion, so the file is a draft locator rather than a claim of 25 complete issues.

## 10. Empirical sources, versions and collection maintenance

| Source | Precise use | Reviewed | Limitation / trigger |
|---|---|---|---|
| [Authoring standard](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | format, numerical/evidence rules | 2026-11-09 | recheck template revision |
| [Cheat-sheet template](../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) | 12 sections and collection navigation | 2026-11-09 | recheck template revision |
| [Google SRE Workbook](https://sre.google/workbook/implementing-slos/) | SLO method vocabulary | 2026-11-09 | not workload evidence |
| [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) | risk vocabulary | 2026-11-09 | not certification |
| [ROS 2 real-time tutorial](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Demos/Real-Time-Programming.rst) | timing examples | 2026-11-09 | recheck ROS/version/device |

Illustrative arithmetic is hand-derived and independently recomputable from per-sheet oracle inputs; no runtime, benchmark, vendor price, hardware, or policy claim has been executed or verified as deployment evidence. Recheck sources, dependencies, workload, price date, and policy before using a conclusion.

## 11. Collection acceptance checklist

- [x] Seven indexed artifacts and eleven navigation sections exist in intended order.
- [x] Symbol glossary, three reading paths, exclusions, card directory, and issue locator link real local files.
- [ ] Final troubleshooting artifact has 25 ISSUE IDs and categories; entry-specific investigations remain incomplete and require review.
- [ ] Independently review individual-depth requirements, sources, assumptions, links, and all arithmetic before publication.
- [ ] Execute any claimed deployment-specific benchmark, restore, load, or safety test before upgrading evidence labels.
