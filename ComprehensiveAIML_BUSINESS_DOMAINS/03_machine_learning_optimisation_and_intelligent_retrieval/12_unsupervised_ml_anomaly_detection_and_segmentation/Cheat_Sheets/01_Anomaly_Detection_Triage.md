---
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
file_id: "01"
topic: "Anomaly detection triage"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in event schema, prevalence, investigator capacity, or label maturity."
notation_style: "plain-text"
prerequisites: ["Probability, distances, validation contracts"]
source_status: "OPEN"
owner_role: "Risk analytics reviewer"
source_index: "INDEX.md"
---

# Anomaly Detection Triage

> **Purpose:** turn an unsupervised score into a capacity-bounded investigation policy without calling an alert a confirmed event.
> **Prerequisites:** probability, feature scaling, and the validation contract in Module 11.
> **Key outcome:** Given scores, review capacity, and delayed labels, you can select a triage policy and test whether an apparent anomaly is a data or operational failure.
> **This file does NOT cover:** proof of fraud, causal attribution, or a measured fraud deployment.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED score/ranking identities; VERIFIED arithmetic; ILLUSTRATIVE records; OPEN source bodies.
> **Retrieval cue:** If alerts rise abruptly, start at Section 8.

The visible inventory lists `08_Anomaly_Detection_Unsupervised.md` under fraud cheat sheets at [`directory.txt`](../../../../Templates/directory.txt#L584). That filename supplies scope only; it does not expose source content or establish a production result.

## 2. Concept Map / Mental Model

```text
[event at t0] -> [point-in-time features] -> [score / density / isolation]
      |                 |                         |
      |                 +-- missing/schema shift? -+--> [data incident]
      v
[rank eligible alerts] -> [top C review queue] -> [mature adjudication]
      |                                            |
      +-- capacity/priority policy                 +--> [precision-at-C and root-cause learning]
```

Anomaly detection ranks observations that depart from a declared reference population. It is not a label generator. The useful output is a review decision with an owner, capacity, evidence retention, and an abstention path.

**Takeaway:** A score is evidence for inspection, not an outcome label.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Boundary |
|---|---|---|---|
| `x` | point-in-time event feature vector | vector | features must exist at `t0` |
| `a(x)` | anomaly score, larger is more unusual | model-specific | not comparable across versions |
| `C` | available reviews per day | alerts/day | distinct from model prevalence |
| `k` | selected top alerts | count | `k<=C` |
| mature label | adjudicated outcome after delay | categorical | missing outcome is not negative |
| contamination | assumed outlier fraction | probability | a tuning assumption, not observed fraud rate |
| drift | reference distribution changes | statistical/operational | may be a schema change rather than behavior |

**Takeaway:** Define the score direction, decision interval, and label maturity before interpreting alert volume.

## 4. Core Formulas with Full Derivation

**Validity before use.** The following ranking quantities require a fixed evaluation window, deduplicated alerts, and labels mature for reviewed and sampled unreviewed records. They do not establish future prevalence.

```text
FORMULA [ANO-01]: precision_at_k = confirmed_in_top_k / k
WHERE: confirmed_in_top_k [alerts] have mature adjudication; k [alerts] is review count.
CONSTRAINT: k>0; review selection and label process are recorded.
NUMERICAL EXAMPLE: 18 confirmed alerts among k=60 reviews gives 18/60=0.30.
INTERPRETATION: 30% is investigator yield in this illustrative queue, not recall.
```

**Derivation.** Density methods rank low estimated `p(x)`; isolation methods rank short expected path length in random partitions. Both are reference-relative. If a score threshold admits `A(tau)` alerts, feasibility requires `A(tau)<=C`; otherwise the decision policy must rank or defer alerts.

```text
FORMULA [ANO-02]: alert_rate = selected_alerts / eligible_events
WHERE: selected_alerts [events/day]; eligible_events [events/day].
CONSTRAINT: same eligibility filter and day; duplicates removed.
NUMERICAL EXAMPLE: 60 selected from 12,000 eligible events: 60/12000=0.005=0.5%.
INTERPRETATION: a 0.5% rate still consumes the entire 60-alert capacity.
```

**Boundary:** a score distribution shift can reflect a new merchant code, changed scaling, delayed feed, or novel behavior. Do not call it behavioral anomaly until discriminating checks rule out data causes.

**Takeaway:** Capacity converts an unsupervised ordering into an explicit policy.

## 5. Worked Numerical Example

**ILLUSTRATIVE; hand arithmetic checked.** A team receives 12,000 eligible events/day and can review 60. The top 60 contain 18 mature confirmed events, 30 mature non-events, and 12 pending adjudications. Report confirmed precision only for mature decisions: `18/(18+30)=37.5%`; do **not** divide by 60 as if pending records were negative. Alert rate is `60/12000=0.5%`. If 24 of 48 mature reviews are confirmed next week, mature precision is 50%; this change may be population mix, reviewer policy, or true scoring improvement, so compare eligibility and label delay before changing the model.

**Takeaway:** Pending labels are an uncertainty statement, not a convenient denominator.

## 6. Visual / Geometric Intuition

A density method asks whether a point lies in a sparse neighborhood after scaling; one large-unit feature can dominate distance. An isolation tree asks how few random splits isolate a point; rare categories and missing-value routing can look isolated even when they are normal operations.

**Takeaway:** “Far away” depends on representation and reference population.

## 7. Comparison / Selection Table

| Method | Use when | Avoid when | Check |
|---|---|---|---|
| robust rules | known invariant exists | behavior is novel/variable | rule coverage and expiry |
| isolation forest | mixed nonlinear features, ranking needed | score calibration is assumed | contamination sensitivity |
| density/LOF | local density matters | high-dimensional distance is weak | scaling and neighbor stability |
| change point | ordered aggregate shifts matter | individual-event attribution needed | seasonality baseline |
| human review | consequences require judgment | queue has no owner | capacity and maturity ledger |

**Takeaway:** Select the method whose failure mode you can observe and contain.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Discriminating diagnostic | Bounded correction | Verify |
|---|---|---|---|---|---|
| ANO-F01 | alert spike | real shift; schema nulls; scaling change | null rates/version/time join | quarantine bad feed or refit reference | replay known-good window |
| ANO-F02 | high score on new category | novelty; rare legitimate rollout | category launch/change log | allowlist with expiry | sampled review |
| ANO-F03 | reviewers overloaded | threshold, duplicates, demand jump | unique-key and queue-age audit | top-k plus dedupe | next-day capacity replay |
| ANO-F04 | apparent precision collapses | immature labels; mix shift; score decay | maturity cohort report | delay conclusion/recalibrate | mature future cohort |
| ANO-F05 | cluster looks actionable | projection artifact; sampling bias | stability across seeds/features | treat as hypothesis | prospective test |

**Takeaway:** Diagnose data and workflow before tuning the detector.

## 9. Complexity & Cost Analysis

Nearest-neighbor density can require substantial index/search work; brute-force pairwise distance is `O(n^2 d)` for `n` records and `d` features. Tree ensembles trade exact local-density reasoning for bounded tree traversal. The operational cost often dominates: 60 reviews at 20 minutes each equal 1,200 minutes, or 20 staff-hours/day, excluding escalation.

**Takeaway:** A detector that exceeds review capacity is a queue generator, not a finished control.

## 10. Key Theorems / Results Reference

Isolation Forest: Liu, Ting, and Zhou (2008), [DOI](https://doi.org/10.1109/ICDM.2008.17), supports the isolation mechanism. LOF: Breunig et al. (2000), [DOI](https://doi.org/10.1145/342009.335388), supports local outlier factor. These sources do not validate the illustrative workload.

**Takeaway:** Method citations support mechanisms, not local effectiveness.

## 11. Interview / Exam Questions

1. Why is anomaly score not fraud probability? It is relative unusualness without a calibrated label model.
2. What denominator is needed for precision-at-k? Mature reviewed alerts; pending outcomes require separate reporting.
3. What is the first check after an alert spike? Schema, null, version, and eligibility changes before behavioral interpretation.
4. Why can normalization create anomalies? Changed scale or a dominant feature changes distance/order.
5. How does capacity affect thresholding? It defines maximum selected alerts and requires top-k or an escalation policy.
6. What validates an unsupervised detector? Time-safe mature labels, review outcomes, sampled negatives, and operational queue data.
7. When does LOF struggle? In high-dimensional or poorly scaled spaces where neighborhood distances lose meaning.
8. What does contamination mean? An assumed tuning fraction, not a verified incident rate.
9. How would you test drift? Freeze version, compare feature/score distributions and label maturity across comparable windows.
10. What is a safe response to ambiguous alerts? Preserve evidence, route to review, and avoid irreversible action until contract permits it.

**Takeaway:** Defend the score through its reference population, queue policy, and evidence maturity.

## 12. Quick Reference Card

| Trigger | Rule | Assumption | Avoid | Verification |
|---|---|---|---|---|
| alert spike | audit feed before model | version ledger exists | calling it new behavior | replay prior window |
| limited reviews | rank top `C` | capacity measured | fixed threshold overflow | queue count |
| yield report | exclude pending labels | maturity known | treating pending as negative | cohort table |
| local outlier | scale and test neighbors | meaningful distance | raw mixed units | stability test |
| new category | inspect rollout | category provenance exists | permanent allowlist | sampled review |
