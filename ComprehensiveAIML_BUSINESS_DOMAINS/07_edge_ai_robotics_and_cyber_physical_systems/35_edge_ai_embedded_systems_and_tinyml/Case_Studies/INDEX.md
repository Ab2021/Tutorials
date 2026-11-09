# Module 35 Case Studies — Edge AI, Embedded Systems and TinyML

## 1. Collection metadata

**Domain:** resource-bounded embedded AI. **Purpose:** cases will test a bounded decision owner, baseline, data/firmware contract, validation plan and recovery boundary rather than assert a deployment. **Evidence policy:** named source titles do not prove a company architecture or measurement. **Collection state:** planned; no authored module case exists. **Source syllabus:** [edge-arm-firmware](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware); [directory inventory, lines 1094–1110](../../../../Templates/directory.txt).

## 2. Case catalog

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-M35-01 | Wearable activity classifier | embedded TinyML | predictive-ml | choose local model/acquisition/alert boundary | bounded edge inference | data-quality gate; safe degradation | illustrative proposed | proposed | null | target performance and clinical meaning not measured | planned |
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
| Unit | timestamped sensor window | asset/sensor-window | labels, action and sampling differ |
| Main decision | local classification/alert handling | feature/reporting threshold and route | neither is a clinical or maintenance conclusion |
| Baseline | fixed-rate upload/manual review | periodic inspection/simple threshold | baseline must be measured or declared illustrative |
| Critical failure | stale/misaligned window | aliasing, loss or false anomaly | validation oracles differ |
| Shared contract | sample, buffer, time, model and transport identity | same | reusable firmware reasoning only |

## 5. Questions / future-case backlog

1. What exact target device, battery and sensor stream would bound the wearable decision?
2. Which activity label protocol and participant split would avoid leakage?
3. What vibration waveform, mounting condition and maintenance label defines an anomaly?
4. Which missed/false alerts are operationally tolerable, and who may override them?
5. What local safe/degraded behavior is hazard-appropriate after a deadline miss?
6. Can a host retry duplicate a command or only duplicate telemetry?
7. Which hardware-in-the-loop input and reset traces are needed before any result claim?

## 6. Review / retirement log

| Date | Item | Event | Evidence effect | Action |
|---|---|---|---|---|
| 2026-11-09 | catalog | collection expanded from short planning list | none; cases remain un-authored | preserve plans and require case-master sections before promotion |

## 7. Pre-publish checks

- [ ] Assign stable `CASE` IDs only when a real case file exists.
- [ ] Preserve all 15 case sections, 5a–5e, 12–18 answered interview questions and Section 15 last.
- [ ] Obtain primary evidence or label every scenario and numerical result illustrative/not-measured.
- [ ] Recompute workload, UART, memory, latency and energy calculations with units and sensitivity.
- [ ] Test successful and failed request/recovery paths on stated inputs.
- [ ] Link the supporting Module 35 sheets and update catalog evidence/stage/editorial status independently.
