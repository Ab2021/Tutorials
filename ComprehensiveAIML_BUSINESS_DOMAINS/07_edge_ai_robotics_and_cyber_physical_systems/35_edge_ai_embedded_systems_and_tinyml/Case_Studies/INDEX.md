# Module 35 Case Studies — Edge AI, Embedded Systems and TinyML

## 1. Collection metadata

**Domain:** resource-bounded embedded AI. **Purpose:** cases test a bounded decision owner, baseline, data/firmware contract, validation plan and recovery boundary rather than assert a deployment. **Evidence policy:** named source titles do not prove a company architecture or measurement. **Collection state:** one authored illustrative draft and one remaining planned case. **Source syllabus:** [edge-arm-firmware](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware); [directory inventory, lines 1094–1110](../../../../Templates/directory.txt). Source bodies and target evidence remain OPEN.

## 2. Case catalog

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CASE-M35-01 | [Wearable activity classifier](01_wearable_activity_classifier.md) | embedded TinyML | predictive-ml | choose a local classifier/acquisition/display boundary | bounded acquisition | freshness gate; degraded local behavior | illustrative / not-measured | proposed | null | resource arithmetic and validation plan only; no target result, clinical meaning, or deployment | draft |
| PLAN-M35-01 | Wearable activity classifier — original planning row retained | embedded TinyML | predictive-ml | choose local model/acquisition/alert boundary | bounded edge inference | data-quality gate; safe degradation | illustrative proposed | superseded by CASE-M35-01 brief | null | target performance and clinical meaning not measured | retained planning provenance |
| PLAN-M35-02 | Industrial vibration monitor | industrial edge monitoring | streaming-systems | choose sampling, feature and transmission budget | bounded acquisition | anomaly review; loss accounting | illustrative proposed | proposed | null | machinery data, threshold and field outcome unavailable | planned |

## 3. Canonical reusable patterns

| Pattern | Mechanism | Use when | Do not infer |
|---|---|---|---|
| bounded acquisition | publish immutable DMA blocks with sequence/time | continuous sensor input | that DMA proves consumer timeliness |
| freshness gate | reject output whose input age violates decision contract | delayed queues or remote handoff | the correct hazard response |
| target evidence ledger | retain exact part/config/input/trace | claiming feasibility | that illustrative arithmetic is a benchmark |
| degraded local behavior | contain a violated deadline or quality gate | physical consequence exists | that power removal is universally safe |

## 4. Similarity / contrast map

| Dimension | Wearable activity classifier | Industrial vibration monitor | Why the distinction matters |
|---|---|---|---|
| Unit | timestamped participant/session window | asset/sensor-window | labels, action and sampling differ |
| Main decision | local classifier/display-abstention boundary | feature/reporting threshold and route | neither is a clinical or maintenance conclusion |
| Baseline | upload/manual review or magnitude threshold | periodic inspection/simple threshold | baseline must be measured or declared illustrative |
| Critical failure | split leakage, stale/misaligned window, overflow | aliasing, loss or false anomaly | validation oracles differ |
| Shared contract | sample, buffer, time, model and transport identity | same | reusable firmware reasoning only |

## 5. Questions / future-case backlog

1. Which exact target device, battery discharge trace, sensor range, and participant population would bound the wearable decision beyond its illustrative model?
2. Can the proposed prompted/adjudicated label protocol and participant-held-out split be executed without consent, session, or transition exclusions changing the denominator?
3. What vibration waveform, mounting condition and maintenance label defines an anomaly?
4. Which missed/false alerts are operationally tolerable, and who may override them?
5. What local safe/degraded behavior is hazard-appropriate after a deadline miss?
6. Can a host retry duplicate a command or only duplicate telemetry?
7. Which hardware-in-the-loop input and reset traces are needed before any result claim?

## 6. Review / retirement log

| Date | Item | Event | Evidence effect | Action |
|---|---|---|---|---|
| 2026-11-09 | catalog | collection expanded from short planning list | none; cases remain un-authored | preserve plans and require case-master sections before promotion |
| 2026-11-10 | CASE-M35-01 | authored illustrative wearable case from PLAN-M35-01 | no measured model, device, or clinical result; source bodies remain OPEN | retain original plan and require target/participant evidence before any promotion |

## 7. Pre-publish checks

- [x] Assign stable `CASE` IDs only when a real case file exists (`CASE-M35-01` is an illustrative draft; `PLAN-M35-02` remains planned).
- [x] CASE-M35-01 preserves 15 case sections, 5a–5e, 12 answered interview questions and Section 15 last.
- [ ] Obtain primary evidence or label every scenario and numerical result illustrative/not-measured.
- [x] Recompute CASE-M35-01 workload, memory, latency, freshness, overflow, and energy arithmetic with units and meaningful failed-state boundaries; target measurements remain required.
- [ ] Test successful and failed request/recovery paths on stated target inputs.
- [x] Link the supporting Module 35 sheet and update catalog evidence/stage/editorial status independently; preserve PLAN-M35-02 and source limitations.
