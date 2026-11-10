# Part 07 Case Studies — Edge AI, Robotics and Cyber-Physical Systems

## 1. Collection metadata

**Purpose:** cross-module decision cases for systems that perceive, estimate, act and operate under physical constraints. A case must identify an owner, decision, baseline, input/output grain, unit, time horizon, recovery boundary and evidence basis; a technology inventory is not a case. **Evidence policy:** all presently catalogued cases are proposed illustrative exercises unless an authored file says otherwise. Source-register titles are scope provenance, not evidence that a company, device or original source used the design. **Editorial state:** catalog only; no part-level case file is currently authored. **Prerequisites:** modules 35–41 in order appropriate to the chosen decision, plus units, timestamps, state-machine reasoning and honest evidence labels.

## 2. Case catalog

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-P07-01 | Wearable activity classifier | embedded TinyML | predictive-ml | choose local acquisition/model/alert boundary | bounded edge inference | data-quality gate; safe degradation | illustrative | proposed | null | no target, clinical meaning or field outcome measured | planned |
| PLAN-P07-02 | Industrial vibration monitor | industrial edge monitoring | streaming-systems | choose sampling, features and transmission budget | bounded acquisition | anomaly review; loss accounting | illustrative | proposed | null | waveform, threshold and maintenance outcomes unavailable | planned |
| PLAN-P07-03 | Quantise an inspection model | edge deployment | predictive-ml | select PTQ, QAT or smaller model under target limits | compression selection | equivalence gate; rollback | illustrative | proposed | null | target kernel support and quality not measured | planned |
| PLAN-P07-04 | Compare edge runtimes on a Pi-class device | edge runtime | streaming-systems | select runtime/conversion path for fixed workload | compatibility evidence | benchmark protocol; release gate | illustrative | proposed | null | device, version and workload must be pinned | planned |
| PLAN-P07-05 | Warehouse asset tracking | sensor fusion | streaming-systems | select time/frame/fusion and stale-data policy | freshness gate | association uncertainty; degradation | illustrative | proposed | null | location accuracy and operations impact not measured | planned |
| PLAN-P07-06 | Multisensor worker-safety system | industrial perception | hybrid | choose evidence gate and escalation boundary | safety-aware fusion | human review; degraded mode | illustrative | proposed | null | no safety function or regulatory conclusion | planned |
| PLAN-P07-07 | Robot arm pick-and-place cell | manipulation | hybrid | choose transform/planning/control boundary | constrained motion | workspace gate; controlled stop | illustrative | proposed | null | geometry, hazard analysis and cell results not measured | planned |
| PLAN-P07-08 | Mobile robot fleet control design | robotics operations | streaming-systems | choose local/central authority and recovery | bounded authority | command freshness; fleet orchestration | illustrative | proposed | null | no fleet behavior or safety result measured | planned |
| PLAN-P07-09 | Autonomous inventory robot | navigation | hybrid | select localization/perception/recovery gating | confidence-to-action gate | map freshness; human handoff | illustrative | proposed | null | no navigation outcome measured | planned |
| PLAN-P07-10 | Bin-picking system evaluation | manipulation perception | predictive-ml | select grasp policy evaluation and abstention | perception-to-grasp contract | calibration; failure taxonomy | illustrative | proposed | null | object set and physical success not measured | planned |
| PLAN-P07-11 | Synthetic warehouse dataset | simulation | analytical | decide whether synthetic labels are fit for intended evaluation | provenance-controlled synthesis | shift analysis; label audit | illustrative | proposed | null | fidelity and downstream benefit not measured | planned |
| PLAN-P07-12 | Validate a navigation policy before field trial | sim-to-real | hybrid | define staged promotion or reject field trial | transfer validation ladder | scenario coverage; controlled recovery | illustrative | proposed | null | simulation cannot establish field safety | planned |
| PLAN-P07-13 | Fleet rollout readiness review | robot operations | analytical | approve, hold or stage a version rollout | release evidence gate | SLOs; rollback | illustrative | proposed | null | no actual rollout evidence | planned |
| PLAN-P07-14 | Robot near-miss incident response | robot operations | incident | contain, diagnose and prevent a near-miss recurrence | hazard containment | evidence preservation; authority recovery | illustrative | proposed | null | not an observed incident report | planned |

## 3. Canonical reusable patterns

| Pattern | Mechanism | Use when | Disqualifier / non-inference |
|---|---|---|---|
| bounded edge inference | validate sensor window, model identity, deadline and output age locally | action cannot wait for cloud | does not establish prediction quality or safety |
| freshness gate | reject/degrade output whose capture-to-use age exceeds a declared bound | delayed queues or remote control exist | a timestamp without clock semantics is insufficient |
| compatibility gate | prove format/operator/config/calibration compatibility before release | models or firmware change | signature checks do not prove behavioral compatibility |
| confidence-to-action gate | combine uncertainty/quality with a bounded permitted action | perception drives a physical action | confidence is not a hazard analysis |
| transfer validation ladder | progress offline → simulation → controlled hardware → limited operation with evidence gates | moving from simulation to field | a simulator pass is not field evidence |
| bounded authority | local action gate accepts only current, authorized commands | fleet or remote assistance exists | connectivity does not confer authority |
| hazard containment | move to the hazard-specific controlled state before diagnosis | a physical consequence may continue | power removal is not universally safe |

## 4. Similarity / contrast map

| Group | Primary unit and output | Central uncertainty | Required recovery reasoning | Closest handoff |
|---|---|---|---|---|
| M35–36 device cases | sample window/model artifact → local score/runtime selection | target resource and semantic compatibility | buffer/rollback/known-good image | Modules 35–36 sheets |
| M37 fusion cases | timestamped observation → track/escalation input | alignment, association and observability | stale/missing sensor degradation | Module 37 sheets |
| M38–39 robot cases | world/robot state → motion/action | geometry, map/perception validity | controlled stop or abstention | Modules 38–39 sheets |
| M40 transfer cases | scenario/configuration → promotion decision | unmodeled real mechanism | hold promotion, instrument next test | Module 40 sheets |
| M41 operation cases | release/incident record → fleet action | authority, observability and rollback | isolate, preserve evidence, recover | Module 41 sheets |

## 5. Questions / future-case backlog

1. Which decision owner can accept or reject the proposed action, and at what cadence?
2. What input exists at the decision time rather than after the event?
3. What manual, rule-based or hold-state baseline is credible?
4. Which metric has a numerator, denominator, unit, population, window and guardrail?
5. What must be measured on target before an architecture claim is made?
6. Which failed request, stale command or sensor dropout path reaches the hazard-specific recovery state?
7. What is the smallest reproducible workload trace and independent oracle?
8. What evidence distinguishes a simulator mismatch from a software/configuration fault?
9. Who has authority during remote assistance, network partition and rollback?
10. What assumption would reverse the selected method or block promotion?

## 6. Review / retirement log

| Date | Item | Event | Evidence effect | Action |
|---|---|---|---|---|
| 2026-11-09 | catalog | Reconciled the fourteen original module case commitments into one part navigation table. | None; rows remain planned illustrative case briefs. | Retain each module home and do not assign `CASE` IDs until a 15-section file exists. |
| 2026-11-09 | source basis | Confirmed source plan is visible-name planning provenance. | No source body, deployment or outcome may be inferred. | Preserve uncertainty in authored cases and obtain primary sources where relevant. |

## 7. Pre-publish checks

- [ ] A real case file has metadata and all 15 sections in order, including 5a–5e and Section 15 last.
- [ ] A stable `CASE` ID replaces the corresponding plan only after authorship; catalog preserves case type, patterns, evidence, stage, review date and limitation.
- [ ] Each case states a bounded owner/action/unit/time/baseline and labels every scenario/result as observed, published, estimated, simulated, illustrative or not-measured.
- [ ] System cases trace successful input-to-output behavior and failed/recovery behavior with ownership and state boundaries.
- [ ] Checked numerical walkthrough, sensitivity, workload/capacity reasoning, four-or-more diagnostic failures and 12–18 answered interview prompts are present.
- [ ] Links to actual module sheets, source records and collection patterns resolve; target/safety/field claims have independent review.
