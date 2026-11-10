# Module 37 Case Studies — Sensor Fusion, IoT and Real-Time Perception

## 1. Collection metadata

**Domain:** timestamped multi-sensor perception and telemetry. **Purpose:** teach a bounded operational decision about observations whose time, frame, uncertainty or delivery quality may be wrong. **Evidence policy:** these are illustrative proposed cases; no workplace, worker, warehouse, device, safety function or outcome is asserted. **Source basis:** the visible edge-signals register is planning provenance only; its source bodies were not supplied. **Prerequisites:** sampled signals, timestamp semantics, coordinate frames, basic probability and the Module 37 sheet collection. **Collection state:** catalog expanded; no authored individual case exists.

## 2. Case catalog

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-M37-01 | Warehouse asset tracking | indoor asset tracking | streaming-systems | choose alignment, association, freshness and degraded-mode contract | freshness gate | association uncertainty; telemetry lineage | illustrative | proposed | null | no site trace, accuracy result or operations outcome measured | planned |
| PLAN-M37-02 | Multisensor worker-safety system | industrial perception | hybrid | choose signal-quality gate, escalation boundary and safe handoff | safety-aware fusion | human review; controlled degradation | illustrative | proposed | null | no safety claim, worker data or incident outcome measured | planned |

## 3. Canonical reusable patterns

| Pattern | Mechanism | Use when | Non-inference |
|---|---|---|---|
| capture-time contract | preserve clock domain, capture/receive/process time, uncertainty and sequence | asynchronous sensors feed one decision | receipt order is not physical order |
| frame-and-unit gate | reject data without compatible frame direction, units and calibration revision | geometry or derived velocity is used | a valid transform name does not prove current mounting |
| freshness gate | bound capture-to-use age and select a declared degraded action | queued/networked observations exist | a dropped observation alone does not specify the safe action |
| quality-aware fusion | propagate quality/uncertainty and reject invalid observations before estimator update | measurements have distinct failure modes | fused output confidence is not independent evidence |
| telemetry lineage | carry schema/version/source/quality metadata to diagnosis | a remote operator or audit needs reconstruction | telemetry availability is not real-time authority |

## 4. Similarity / contrast map

| Dimension | Warehouse asset tracking | Multisensor worker-safety system | Consequence |
|---|---|---|---|
| Unit | asset observation/track update | bounded zone-event candidate | association and action costs differ |
| Baseline | manual scan or simple beacon position | fixed threshold/manual process | baseline must be measured or explicitly illustrative |
| Binding uncertainty | time/frame/association | quality, occlusion and escalation timing | validation oracle differs |
| Allowed recovery | mark unknown, request scan, suppress stale track | hazard-specific controlled escalation/handoff | never prescribe generic power removal |
| Evidence needed | synchronized trace and ground-truth protocol | hazard analysis, sensor-quality trace and authority policy | neither follows from model accuracy alone |

## 5. Questions / future-case backlog

1. Which clock is authoritative and how is cross-clock uncertainty represented?
2. What location/zone error is acceptable for the particular decision, not merely for a chart?
3. What observation-time cutoff prevents future information from entering an evaluation?
4. How are duplicate, out-of-order and late messages distinguished from true motion?
5. Which physical reference validates each calibration transform and its direction?
6. Who owns the action when sensor quality crosses the rejection bound?
7. What safe/degraded behavior is appropriate to the specific hazard and why?
8. Which independent ground-truth source can evaluate association and freshness?

## 6. Review / retirement log

| Date | Item | Event | Evidence effect | Action |
|---|---|---|---|---|
| 2026-11-09 | catalog | Replaced a two-row planning list with the required seven-section case-index structure. | None; both rows remain proposed illustrative plans. | Do not allocate `CASE` IDs until 15-section files exist. |
| 2026-11-09 | source scope | Reconfirmed edge-signals source names are not source-body access. | No medical/device/workplace claim is supported. | Keep source uncertainty visible in any future case. |

## 7. Pre-publish checks

- [ ] Each case has the required metadata and Sections 1–15, including 5a–5e and a final Section 15 card.
- [ ] Owner, action, input/output unit, deadline, baseline, clock/frame/uncertainty contract and recovery behavior are bounded.
- [ ] Successful and failed input-to-output/recovery paths have an explicit state and authority owner.
- [ ] Numerical walkthrough and sensitivity have checked units; no illustrative workload is presented as measured.
- [ ] Section 12 supplies competing causes, diagnostic signal, bounded correction, mitigation risk, prevention and verification.
- [ ] Section 14 contains 12–18 answered prompts with follow-up, alternatives, red flags, scoring and support.
