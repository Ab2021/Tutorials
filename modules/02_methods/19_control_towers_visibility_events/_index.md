# Module 19 — Control Towers, Visibility, and Event Intelligence

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/19_control_towers_visibility_events.md`](../../../course/19_control_towers_visibility_events.md)
> **Folder:** `modules/02_methods/19_control_towers_visibility_events/`
> **Version:** 2026.5

---

## 1. Module purpose

This module turns event data into action: visibility layers, event time, ETA, delay propagation, anomaly detection, exception queues, and decision rights.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 03, 06, 12 |
| Unlocks | 20, 22 |
| Assessment | Control-tower design (Week 22) |
| Survey file | `19_control_towers_visibility_events.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_visibility-layers.md`](01_visibility-layers.md) | The Five Visibility Layers | Shipment, inventory, order, supplier, and network visibility and what each can answer. | 1,200–1,800 | planned |
| 2 | [`02_event-streams-epcis.md`](02_event-streams-epcis.md) | Event Streams, IoT, Telematics, and EPCIS 2.0 | The raw material of a control tower and its quality problems. | 1,200–1,800 | planned |
| 3 | [`03_event-time-processing-time.md`](03_event-time-processing-time.md) | Event Time versus Processing Time | Late, out-of-order, and duplicated events in a live queue. | 1,200–1,800 | planned |
| 4 | [`04_eta-prediction.md`](04_eta-prediction.md) | ETA Prediction | Prediction intervals, censoring, and the promise that follows. | 1,200–1,800 | planned |
| 5 | [`05_delay-propagation.md`](05_delay-propagation.md) | Delay Propagation | How one late leg becomes a missed slot and a lost sale. | 1,200–1,800 | planned |
| 6 | [`06_anomaly-detection.md`](06_anomaly-detection.md) | Anomaly Detection | Statistical, distance, and model-based detectors in an operational stream. | 1,200–1,800 | planned |
| 7 | [`07_exception-queue.md`](07_exception-queue.md) | Exception Management and Queue Design | Prioritisation, ownership, and the state machine of an exception. | 1,200–1,800 | planned |
| 8 | [`08_alert-economics.md`](08_alert-economics.md) | Alert Precision, Recall, and the Economics of Attention | The cost of a false alarm and the cost of a miss. | 1,200–1,800 | planned |
| 9 | [`09_decision-rights.md`](09_decision-rights.md) | Decision Rights and Escalation | Who can reroute, expedite, or promise, and at what threshold. | 1,200–1,800 | planned |
| 10 | [`10_multi-party-sharing.md`](10_multi-party-sharing.md) | Multi-Party Data Sharing and Trust | Partner data, incentives, and the limits of a shared tower. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Five Visibility Layers ->
02 Event Streams, IoT, Telematics, and EPCIS 2.0 ->
03 Event Time versus Processing Time ->
04 ETA Prediction ->
05 Delay Propagation ->
06 Anomaly Detection ->
07 Exception Management and Queue Design ->
08 Alert Precision, Recall, and the Economics of Attention ->
09 Decision Rights and Escalation ->
10 Multi-Party Data Sharing and Trust
```

## 5. What this module does not cover

- It does not build the underlying integration platform; Module 03 covers data engineering.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`19_control_towers_visibility_events.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
