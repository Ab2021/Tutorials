# Module 05 — Experimentation, Pilots, and Evidence

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
> **Folder:** `modules/01_foundations/05_experimentation_pilots_evidence/`
> **Version:** 2026.5

---

## 1. Module purpose

This module designs the experiment, pilot, or quasi-experiment that produces the counterfactual, and teaches how to read vendor and analyst evidence without being fooled.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 04 |
| Unlocks | 09, 16 |
| Assessment | Evidence plan (Week 7) |
| Survey file | `05_experimentation_pilots_evidence.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md) | From Claim to Hypothesis to Design | Turning a business claim into a testable estimand and a decision rule. | 1,200–1,800 | planned |
| 2 | [`02_randomisation-units.md`](02_randomisation-units.md) | Randomisation Units | Customer, store, SKU, lane, DC, and the unit that matches the decision. | 1,200–1,800 | planned |
| 3 | [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md) | A/B, A/B/n, and Cluster Randomisation | Design effects when the unit of assignment is not the unit of analysis. | 1,200–1,800 | planned |
| 4 | [`04_geo-experiments.md`](04_geo-experiments.md) | Geo Experiments | Market-level tests when customer-level randomisation is impossible. | 1,200–1,800 | planned |
| 5 | [`05_switchback-pilots.md`](05_switchback-pilots.md) | Switchback and Operational Pilot Designs | Time-based alternation in stores, DCs, and lanes. | 1,200–1,800 | planned |
| 6 | [`06_power-mde.md`](06_power-mde.md) | Power, MDE, and Sample Size | What the pilot can detect and what it must not pretend to detect. | 1,200–1,800 | planned |
| 7 | [`07_interference-spillover.md`](07_interference-spillover.md) | Interference and Spillover | Marketplace, network, and fulfillment effects that break independence. | 1,200–1,800 | planned |
| 8 | [`08_sequential-testing.md`](08_sequential-testing.md) | Sequential Testing, Peeking, and False Discovery | Always-valid inference and the cost of early stopping. | 1,200–1,800 | planned |
| 9 | [`09_mmm-incrementality.md`](09_mmm-incrementality.md) | Marketing Mix Modelling and Incrementality | What MMM can and cannot identify without an experiment. | 1,200–1,800 | planned |
| 10 | [`10_vendor-evidence-appraisal.md`](10_vendor-evidence-appraisal.md) | Vendor and Analyst Evidence Appraisal | Sample, comparator, endpoint, duration, and conflict of interest. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 From Claim to Hypothesis to Design ->
02 Randomisation Units ->
03 A/B, A/B/n, and Cluster Randomisation ->
04 Geo Experiments ->
05 Switchback and Operational Pilot Designs ->
06 Power, MDE, and Sample Size ->
07 Interference and Spillover ->
08 Sequential Testing, Peeking, and False Discovery ->
09 Marketing Mix Modelling and Incrementality ->
10 Vendor and Analyst Evidence Appraisal
```

## 5. What this module does not cover

- It does not replace legal or statistical review for a specific regulated experiment.
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

See section 14 of the survey file (`05_experimentation_pilots_evidence.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
