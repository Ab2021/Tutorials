# Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/14A_resilience_disruption_analytics.md`](../../../course/14A_resilience_disruption_analytics.md)
> **Folder:** `modules/02_methods/14A_resilience_disruption_analytics/`
> **Version:** 2026.5

---

## 1. Module purpose

This module quantifies disruption: exposure, time-to-recover, time-to-survive, stress tests, scenario analysis, buffering versus flexibility, and resilience metrics.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 11, 14 |
| Unlocks | 19 |
| Assessment | Resilience stress test (Week 17) |
| Survey file | `14A_resilience_disruption_analytics.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_disruption-timeline.md`](01_disruption-timeline.md) | The Disruption Timeline | Detect, respond, recover, and learn, and which metrics belong at each stage. | 1,200–1,800 | planned |
| 2 | [`02_risk-taxonomy.md`](02_risk-taxonomy.md) | Risk Taxonomy | Supply, demand, operational, financial, cyber, geopolitical, and natural hazards. | 1,200–1,800 | planned |
| 3 | [`03_mapping-exposure.md`](03_mapping-exposure.md) | Mapping and Exposure | Nodes, lanes, tiers, and the difference between importance and vulnerability. | 1,200–1,800 | planned |
| 4 | [`04_ttr-tts.md`](04_ttr-tts.md) | Time-to-Recover and Time-to-Survive | The two clocks that decide whether a disruption becomes a failure. | 1,200–1,800 | planned |
| 5 | [`05_stress-testing.md`](05_stress-testing.md) | Stress Testing | Severity, duration, and correlated shocks applied to a real network. | 1,200–1,800 | planned |
| 6 | [`06_scenario-analysis.md`](06_scenario-analysis.md) | Scenario Analysis | Building scenarios that are plausible, distinct, and decision-relevant. | 1,200–1,800 | planned |
| 7 | [`07_buffering-flexibility.md`](07_buffering-flexibility.md) | Buffering versus Flexibility | Inventory, capacity, dual sourcing, and the cost of each hedge. | 1,200–1,800 | planned |
| 8 | [`08_dual-sourcing-nearshoring.md`](08_dual-sourcing-nearshoring.md) | Dual Sourcing and Nearshoring | Qualification cost, volume split, and the real lead-time benefit. | 1,200–1,800 | planned |
| 9 | [`09_disruption-detection.md`](09_disruption-detection.md) | Disruption Detection | Early warning from supplier, port, weather, and news signals. | 1,200–1,800 | planned |
| 10 | [`10_resilience-metrics-iso.md`](10_resilience-metrics-iso.md) | Resilience Metrics and ISO 22301 | Measuring readiness without turning it into a compliance score. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Disruption Timeline ->
02 Risk Taxonomy ->
03 Mapping and Exposure ->
04 Time-to-Recover and Time-to-Survive ->
05 Stress Testing ->
06 Scenario Analysis ->
07 Buffering versus Flexibility ->
08 Dual Sourcing and Nearshoring ->
09 Disruption Detection ->
10 Resilience Metrics and ISO 22301
```

## 5. What this module does not cover

- It does not produce a business-continuity certification; ISO 22301 alignment requires an audited programme.
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

See section 14 of the survey file (`14A_resilience_disruption_analytics.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
