---
collection: "supervised-ml-and-feature-engineering"
file_id: "01"
topic: "Model validation and error analysis"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-08"
reviewed: null
review_trigger: "A change to prediction time, label maturity, feature lineage, population, or decision policy."
notation_style: "plain-text"
prerequisites: ["Probability, binary confusion matrices, and the module prediction contract."]
pattern_tags: ["temporal-validation", "leakage-control", "thresholding"]
source_status: "OPEN"
owner_role: "ML reviewer"
source_index: "INDEX.md"
---

# Model Validation and Error Analysis

> **Purpose:** select a deployment-faithful evaluation protocol, threshold, and error investigation plan rather than maximizing a convenient offline score.
> **Prerequisites:** probability, binary classification, and a stated prediction-time contract.
> **Key outcome:** Given event timestamps, entity identifiers, scores, labels, and a review capacity, you can reject a leaky split, calculate decision metrics, and specify the next validation check.
> **This file does NOT cover:** fitting a particular classifier; causal effect estimation; or a claim that any illustrative workload is deployed.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED metric identities; VERIFIED hand arithmetic below; ILLUSTRATIVE records; SOURCED method guidance; OPEN source-body coverage.
> **Retrieval cue:** If an offline lift seems implausible, start at Section 8.

**Source inventory and evidence boundary.** The visible inventory at [`directory.txt`](../../../../Templates/directory.txt#L575) includes fraud feature engineering, imbalanced learning, supervised models, and evaluation metrics. Filenames are syllabus provenance, not accessible source bodies or evidence of a production system. The cited scikit-learn pitfalls guide is a method reference, not evidence for the illustrative results here.

## 2. Concept Map / Mental Model

```text
[Business action at prediction time t0]
        |
        v
[unit, label horizon, costs, capacity, protected/operating slices]
        |
        +-- feature available after t0? --> reject or rebuild lineage
        |
        v
[split on the dependency that survives deployment]
        |
        +-- time/entity/site correlation? --> forward/group/site holdout
        |
        v
[fit every learned transform on training partition only]
        |
        v
[scores + calibration + threshold] --> [actions limited by capacity]
        |
        v
[aggregate metrics + slice errors + examples]
        |
        +-- failure signal --> diagnosis, bounded change, untouched confirmation
```

Validation estimates performance of a **decision procedure**, not just a fitted score function. The procedure includes the information available at `t0`, preprocessing fit, model fitting, threshold, ranking/queue policy, and the population to which the result will be applied.

**Takeaway:** Match the evaluation boundary to the future decision boundary before comparing models.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Boundary or nonexample |
|---|---|---|---|
| `t0` | time an action is committed | timestamp | a later chargeback time is not available at `t0` |
| `Y` | mature binary outcome | `{0,1}` | an unresolved label is not a negative label |
| `s(x)` | model score for record `x` | dimensionless | not automatically a probability |
| `p(x)` | calibrated estimated probability | `[0,1]` | calibration is population and time dependent |
| `tau` | action threshold | score/probability units | may change with capacity or cost |
| `TP,FP,FN,TN` | confusion-matrix counts after policy | records | counts require a label-complete window |
| group split | all rows of an entity in one partition | partition rule | random rows from one customer are not group held out |
| forward split | train strictly before evaluation interval | time rule | a randomly shuffled future row violates it |
| leakage | use of information unavailable or contaminated for the evaluated decision | mechanism | a predictive feature is not necessarily available |
| slice | prespecified subgroup tied to a mechanism or harm | population subset | post-hoc cherry-picked rows are exploratory only |

A label has an **observation time** and a **maturity time**. For a 30-day default outcome, a record scored on 1 March cannot be treated as a confirmed non-default on 2 March merely because no event is recorded. Censoring and delayed adjudication must be explicit.

**Takeaway:** A correct feature value with an impossible availability time is still leakage.

## 4. Core Formulas with Full Derivation

### 4.1 Confusion-matrix rates

**Validity before use.** Metrics below describe one fixed population, horizon, label definition, and threshold. They are unreliable when labels are immature, action policy changed during collection, or counts mix incompatible populations.

**Derivation.** Among actual positives, the model either acts (`TP`) or misses (`FN`), so actual positives equal `TP + FN`. Among actions, the action is correct (`TP`) or incorrect (`FP`), so actions equal `TP + FP`.

```text
FORMULA [VAL-01]: precision = TP / (TP + FP)
WHERE: TP [records] are correct positive actions; FP [records] are incorrect positive actions.
CONSTRAINT: TP+FP > 0; labels are mature for the stated decision window.
NUMERICAL EXAMPLE: TP=36, FP=12; precision=36/(36+12)=0.75 = 75%.
INTERPRETATION: three of four reviewed alerts are confirmed positives in this illustrative window.
```

```text
FORMULA [VAL-02]: recall = TP / (TP + FN)
WHERE: FN [records] are actual positives receiving no positive action.
CONSTRAINT: TP+FN > 0 and the outcome definition is stable across compared systems.
NUMERICAL EXAMPLE: TP=36, FN=24; recall=36/(36+24)=0.60 = 60%.
INTERPRETATION: the policy finds 36 of 60 mature positives; it does not estimate prevented loss.
```

**Boundary/counterexample.** A system can attain precision 1.0 by taking one nearly certain action; that says nothing about missed positives. A system can attain high recall by acting on everyone; that may be infeasible or harmful.

### 4.2 Capacity-constrained threshold selection

**Validity before use.** Sort only scores generated before the action. Capacity is a count per stated operational interval, not a percentage detached from demand.

**Derivation.** Let `A(tau)` be records with `s(x) >= tau`. If reviewers can process `C` records/day, an admissible threshold must satisfy `|A(tau)| <= C`. Expected reviewed positives, when scores are calibrated on the applicable population, are the sum of probabilities among actions.

```text
FORMULA [VAL-03]: E[TP(tau)] = sum_{i:s_i>=tau} p_i
WHERE: p_i is calibrated probability [dimensionless]; result is expected positives [records].
CONSTRAINT: calibration must hold for this queue; capacity C applies to the same day and population.
NUMERICAL EXAMPLE: selected p=[0.90,0.70,0.60,0.40], C=3; choose first three, E[TP]=2.20 expected records.
INTERPRETATION: selecting all four violates capacity; a probability sum is not a count of observed positives.
```

**Boundary/counterexample.** If the score is not calibrated, the sum is only a ranking score total. When demand doubles, a fixed threshold can overflow the queue even if score quality is unchanged.

### 4.3 Calibration diagnostic

**Validity before use.** A reliability bin must have adequate support and use outcomes mature under the same label definition. This is a descriptive check, not proof that every individual probability is correct.

**Derivation.** For bin `b`, compare mean predicted probability to empirical frequency. Their difference is a signed calibration residual.

```text
FORMULA [VAL-04]: residual_b = mean(p_i : i in b) - mean(Y_i : i in b)
WHERE: Y_i in {0,1}; residual is dimensionless; b is a fixed score interval.
CONSTRAINT: no label censoring; report n_b and uncertainty, not residual alone.
NUMERICAL EXAMPLE: p=[0.70,0.80,0.90,0.80], Y=[1,1,0,1]; mean(p)=0.80, mean(Y)=0.75, residual=+0.05.
INTERPRETATION: this illustrative bin overpredicts by five percentage points; n=4 is far too small for a deployment conclusion.
```

**Takeaway:** Thresholds are policy parameters constrained by calibrated evidence, demand, and error costs.

## 5. Worked Numerical Example

**Evidence: ILLUSTRATIVE and VERIFIED hand arithmetic; not a measured deployment.** A daily review team has capacity `C=48` alerts. A frozen, mature evaluation window contains 60 actual positives. At a candidate threshold, `TP=36`, `FP=12`, and `FN=24`.

1. Actions are `TP+FP=48` records, exactly filling capacity.
2. By `VAL-01`, precision is `36/48=0.75`.
3. Actual positives are `TP+FN=60`; by `VAL-02`, recall is `36/60=0.60`.
4. False-positive share of the queue is `12/48=0.25`; this is not false-positive rate because its denominator is actions, not actual negatives.
5. If each review takes 15 minutes, required labor is `48*15=720` minutes = 12 staff-hours/day.
6. Independent checks: `36+12=48` actions; `36+24=60` positives; `0.75*48=36`; `0.60*60=36`.

**Sensitivity.** If capacity drops to 36 records/day and the top 36 include an illustrative 30 TP and 6 FP, precision becomes `30/36=83.33%` and recall becomes `30/60=50%`. The narrower queue improves yield but misses six additional positives. If missing a positive has a high cost, lower capacity changes the operational decision; it does not establish that the stricter threshold is better.

**Takeaway:** Quote the queue interval, mature-label population, and capacity whenever reporting a threshold metric.

## 6. Visual / Geometric Intuition

Imagine scores on a horizontal axis. Positives and negatives overlap; the threshold is a vertical cut. Moving the cut right shortens the action queue and usually increases observed precision, but the cut also leaves more positives on the left. Calibration asks whether a vertical slice centered near 0.8 contains roughly 80% positives over a sufficiently large, comparable population.

```text
low score                                             high score
|---- negatives ----|-- mixed --|---- mostly positives ----|
                         ^ tau
       missed positives (FN)      actions: TP plus FP
```

A random row split can make points from the same customer appear on both sides of a train/test boundary. A group split moves the whole customer cluster to one side; a forward split moves a future time region to one side. That geometry explains why a lower score under a stricter split may be more credible.

**Takeaway:** The threshold separates actions, while the split separates evidence; neither can repair the other.

## 7. Comparison / Selection Table

| Deployment condition | Preferred evaluation | Why | Do not use alone | Fallback/check |
|---|---|---|---|---|
| independent, stationary units | stratified random holdout | preserves class support | random split when duplicate entities exist | duplicate and entity audit |
| repeat customer/device | group split by persistent ID | tests unseen entities | row-level CV | compare group and random gap |
| evolving policy or demand | forward/rolling windows | preserves availability and drift | shuffled future labels | time lineage table |
| site-specific operations | site/geographic holdout | tests transfer | pooled score only | report each site support |
| limited reviews | top-k or capacity threshold | matches action budget | AUC alone | queue-volume replay |
| probability-priced action | calibration plus cost curve | score magnitude matters | ranking metric alone | reliability bins and recalibration plan |
| high-stakes minority slice | prespecified slice evaluation | exposes concentrated harm | aggregate only | support/uncertainty and escalation |

**Takeaway:** Choose the split according to the dependency the deployment must survive, then choose metrics according to the action it enables.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|---|
| VAL-F01 | exceptional offline lift | future feature, duplicate entity, genuine signal | build availability ledger; group/time replay | remove unavailable columns and resplit | compare frozen-contract score |
| VAL-F02 | high AUC, overloaded reviewers | threshold too low, demand shift, duplicate alerts | daily top-k count and dedupe rate | capacity threshold and queue policy | replay a held-out day |
| VAL-F03 | calibrated globally, bad decisions in one site | population shift, small sample, label process difference | site reliability bins with n and maturity | site gate or collect labels | future site window |
| VAL-F04 | validation improves every iteration | analyst adapts to validation noise, true progress | untouched final window and change log | freeze selection then test once | final-window report |
| VAL-F05 | high accuracy, no useful alerts | low prevalence, wrong objective, label mismatch | confusion matrix and prevalence baseline | precision/recall/cost objective | threshold replay |
| VAL-F06 | feature importance looks causal | proxy, post-outcome field, correlated feature | timestamp and intervention audit | rename as association; seek causal design | reviewer sign-off |

**Diagnostic trace for VAL-F01.** For each feature record `event_time`, `materialized_time`, and earliest availability to the serving decision. A feature calculated at `t0+7 days` cannot be included in a `t0` score even when its SQL join key is valid. Remove it, refit preprocessing within the training partition, and compare to the old protocol. A collapse is evidence that the old estimate was contaminated; it does not prove a new model is production ready.

**Takeaway:** Treat a surprising score as a hypothesis about the evaluation pipeline, not a reason to celebrate.

## 9. Complexity & Cost Analysis

For `n` records, `k` folds, and model-training cost `T_train(n)`, basic k-fold work is approximately `k*T_train((k-1)n/k)` plus scoring and feature-transform costs. This is an accounting approximation; it excludes data extraction, hyperparameter search, retries, and human review.

A group split can create unequal fold sizes, and a forward split intentionally trains on less history in early folds. That may increase variance but represents the information actually available. Store out-of-fold predictions once rather than repeatedly joining labels; point-in-time feature extraction is often the dominant cost.

**Illustrative sizing.** 2,000,000 records at 1.5 KiB serialized feature payload require about 3,000,000 KiB = 2.86 GiB payload-only storage (`2,000,000*1,536 / 2^30`). Indexes, replication, encryption, and metadata are excluded. If five folds each write predictions at 16 bytes/record, output is 160,000,000 bytes = 152.59 MiB, excluding filesystems and formats.

**Takeaway:** Budget for lineage and point-in-time reconstruction, not only model fitting.

## 10. Key Theorems / Results Reference

| Result | Useful statement | Assumptions/boundary | Source |
|---|---|---|---|
| held-out evaluation | an untouched test set estimates performance of a fixed selected procedure | no repeated adaptation to it; population alignment | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), accessed 2026-11-09 |
| proper scoring rule | log loss rewards truthful probabilities in expectation | expected score and correctly specified outcome framing; not a capacity policy | [Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437) |
| precision-recall dependence | precision changes with prevalence and threshold | same label/selection mechanism needed for comparison | [Davis & Goadrich 2006](https://doi.org/10.1145/1143844.1143874) |

These references support methods, not any illustrative numerical result in this sheet.

**Takeaway:** A named metric result has conditions; cite those conditions instead of treating the metric as a universal score.

## 11. Interview / Exam Questions

1. **Why can random splitting fail for repeated customers?** It can put the same customer pattern in train and test, estimating memorization rather than unseen-customer performance. Use a group split keyed to the persistent entity; if the ID is unreliable, document the residual linkage risk.
2. **A feature is populated nightly but the decision occurs at noon. Is it safe?** Only if the version used at noon was materialized before noon. Check actual availability timestamps, not the nominal source date.
3. **Compute precision for TP=18 and FP=6.** `18/(18+6)=0.75`. It says 75% of actions were positive in the mature evaluation population, not that 75% of future scores are probabilities.
4. **Why is AUC insufficient with 50 reviews/day?** AUC weights rank pairs across all thresholds; it does not state performance at the operating queue cutoff or operational load.
5. **What invalidates a calibration plot?** Immature/selected labels, changed population, too little support, or mixing scores from different model versions. Report these before interpreting bins.
6. **A score rose after adding a feature. What is first?** Audit feature event time, materialization time, joins, and whether preprocessing was fit only on training data before tuning hyperparameters.
7. **When is temporal validation required?** When future information, policy, demand, or label generation can differ from historical training and deployment will score later observations.
8. **How do you set a threshold with unequal error costs?** Define action, miss, and review costs plus capacity; evaluate expected utility over feasible thresholds, then stress uncertain cost inputs. Do not claim a universal threshold.
9. **What is the difference between a slice and fairness certification?** A slice is an error audit for a defined subgroup. It neither proves a legal conclusion nor settles fairness without relevant context, definitions, and evidence.
10. **What would you do when group-split performance collapses?** Verify grouping, quantify overlap in the old split, identify entity-specific features, choose the deployment-matching estimate, and redesign features or scope rather than reporting the convenient score.
11. **Can you use test data to choose a threshold?** Not if it is the final untouched test set. Select on validation/inner folds, freeze the policy, then evaluate once on final data.
12. **Why report support with a slice metric?** A rate from few records is unstable; support makes uncertainty and operational scale visible.

**Takeaway:** A strong answer names the decision time, denominator, dependency, and the check that could overturn the conclusion.

### 11.1 Leakage forensic worksheet

Use this worksheet before accepting a feature, not after a suspiciously high metric appears.

| Control | Required record | Failure interpretation | Evidence artifact |
|---|---|---|---|
| Decision timestamp | exact instant, timezone, and policy cut-off | “day” granularity can hide future data | request/decision log |
| Source event timestamp | when underlying event happened | event may occur after decision | immutable source record |
| Ingest timestamp | when pipeline received event | late arrival can change snapshots | ingestion audit |
| Materialization timestamp | when feature became queryable | batch completion is not event time | feature-store version |
| Join key | entity and cardinality | many-to-many join duplicates outcome evidence | join-profile report |
| Transform fit population | rows used for imputing/scaling/encoding | global fitting leaks validation distribution | fold artifact |
| Label maturity | elapsed delay and adjudication rule | unlabeled is not necessarily negative | cohort ledger |
| Backfill policy | whether corrected history is used | retrospective correction can leak | snapshot identifier |
| Eligibility rule | rows allowed into score policy | denominator drift changes rates | policy version |
| Action timestamp | when threshold was applied | scores may be ranked after the decision | action log |

For a historical feature snapshot, ask two separate questions.

First, could the value have existed by `t0`?

Second, could the serving system have retrieved that exact version by `t0`?

The first is a data-generating-process question.

The second is a systems and lineage question.

A feature can pass the first and fail the second.

For example, a transaction may occur at 09:00.

If the warehouse load completes at 02:00 the next day, it is unavailable to a 12:00 same-day decision.

Do not repair this by shifting an extraction query after the fact.

Build an as-of snapshot or exclude the feature.

### 11.2 Cut-off construction trace

For each decision row, create an explicit cut-off record.

```text
decision_id: D-1042
decision_time: 2026-01-31T12:00:00Z
prediction_horizon: 30 days
label_observable_after: 2026-03-02T12:00:00Z
feature_snapshot: feature-store/v17/asof-D-1042
model_revision: risk-model/3.4.1
policy_revision: review-queue/2.1
```

The label observation date must be later than the outcome horizon.

It is not the same as the extract date.

If an outcome is adjudicated late, record both event and adjudication time.

Use adjudication time to determine maturity.

Use event time to define the outcome window.

Those choices are different because the model must act before adjudication.

For rolling validation, choose cut-offs before examining model results.

Freeze all feature specifications at each cut-off.

Train only on decisions whose labels were mature at the train cut-off.

Tune only on a later validation interval.

Reserve a final interval that selection has not influenced.

Document exceptions rather than silently adding records with convenient labels.

### 11.3 Split selection decision record

| Dependency | Candidate split | Why it estimates deployment | Residual risk | Required stress |
|---|---|---|---|---|
| same customer repeats | group by customer | new-customer generalization | household/device linkage remains | linked-entity sensitivity |
| same merchant repeats | group by merchant | new-merchant behavior | geography may still correlate | location holdout |
| model acts later | forward windows | future deployment | policy changes alter labels | pre/post-policy comparison |
| site rollout | leave-site-out | unseen site transfer | site sample may be too small | confidence interval by site |
| periodic demand | blocked time folds | preserves seasonality | rare events may be absent | horizon coverage report |
| repeated claim | group by claim | avoids duplicate adjudications | claimant linkage can remain | claimant-group check |

Never select a split solely because its score is stable.

Stability can result from shared leakage.

Never select a split solely because it has the largest test sample.

A large but misaligned sample estimates the wrong future.

If two credible splits answer different deployment questions, report both.

Name the primary deployment target.

Do not average incompatible estimates.

### 11.4 Uncertainty without false precision

Precision is a sample proportion conditional on action.

Its observed value has sampling variation.

Report the numerator and denominator beside any rounded rate.

For `TP=36` and `TP+FP=48`, the estimate is `0.75`.

The approximate standard error under independent Bernoulli actions is `sqrt(0.75*0.25/48)`.

That is approximately `0.0625`.

This approximation is only a rough diagnostic.

Repeated entities, temporal dependence, and threshold selection invalidate naive independent-row intervals.

Use grouped or time-respecting resampling when it matches the evaluation unit.

Do not use a confidence interval to repair selection leakage.

Uncertainty quantifies sampling under a protocol.

It does not validate the protocol.

For a slice, report:

- slice definition;
- support;
- numerator and denominator;
- label maturity;
- model and policy version;
- comparator;
- uncertainty method;
- whether the slice was prespecified;
- action threshold;
- decision consequence.

### 11.5 Calibration intervention protocol

Calibration is useful only when a downstream policy uses probability magnitude.

Start with a frozen scoring model.

Partition calibration data by time, not arbitrary random rows, when deployment is temporal.

Fit a calibration mapping on a calibration partition separate from training.

Select the mapping without reading the final holdout.

Apply the mapping unchanged to the holdout.

Compare reliability by score band and meaningful operating slices.

Record whether action capacity alters the observed label process.

If reviewers inspect only high scores, low-score labels can be selectively missing.

That is a label-selection problem, not proof that low scores are calibrated.

Use sampled review or an explicit missing-label model when policy permits.

Recalibration can improve probability alignment.

It cannot restore a feature that was unavailable at decision time.

Recalibration can also degrade ranking.

Evaluate ranking and calibration separately.

### 11.6 Error-analysis investigation protocol

Start with an error table at the unit of action.

Include decision ID, score, threshold, action, mature outcome, key feature versions, and slice membership.

Separate false positives from false negatives before reading free-text notes.

Cluster errors by mechanism, not merely by label.

Examples of mechanisms include delayed data, entity ambiguity, boundary values, policy overrides, missing modality, and novel population.

For each candidate mechanism, write one falsifiable claim.

Example: “false negatives rise when a required source is late.”

Then compute the late-source rate among false negatives and a comparable denominator.

Inspect individual records only after the aggregate mechanism check.

Record redaction and access limits.

Do not export sensitive examples into a teaching artifact.

An error review may generate a feature idea.

That idea is a new hypothesis.

It must enter the next training and validation cycle without reusing the final holdout.

### 11.7 Deployment-shift register

| Shift | Leading signal | Competing explanation | Guardrail | Revalidation action |
|---|---|---|---|---|
| prevalence | score/label base-rate gap | label delay | mature cohort trend | retune policy after review |
| covariate | feature PSI-like change | pipeline encoding bug | raw feature null/range checks | point-in-time replay |
| concept | mature error increase | changed adjudication | reviewer guideline audit | new temporal evaluation |
| policy | action mix changes | model change | policy/version join | compare fixed-policy counterfactual |
| capacity | queue age rises | demand burst | backlog and dedupe metrics | top-k or escalation update |
| data quality | missingness spike | legitimate absence | source health checks | quarantine/rebuild |

**Takeaway:** Validation is a versioned decision protocol whose evidence can be invalidated by data, policy, or operating-context change.

### 11.8 Investigated diagnostic failure dossiers

#### VAL-D01 — A future aggregate entered a training row

**Trigger/environment.** A daily aggregate is joined to a decision table by calendar date.

**Symptom.** Offline validation increases sharply after the aggregate is introduced.

**Competing causes.** The aggregate may add legitimate signal.

It may include events later that day.

It may have been recomputed after labels arrived.

It may have a many-to-many join that duplicates positive rows.

**Discriminating diagnostic.** Reconstruct the aggregate using only events with timestamp strictly before each decision timestamp.

Compare row counts before and after join.

Compare the original score with the as-of score on the same frozen split.

**Root-cause evidence.** A row-level lineage table showing an included event after `t0` establishes temporal leakage.

An uplift collapse after removal is consistent with contamination but does not identify every affected feature.

**Immediate mitigation.** Disable the feature in the serving candidate and block promotion.

**Correction.** Implement as-of aggregation with explicit watermark and snapshot version.

**Prevention.** Require event-time and availability-time fields in feature registration.

**Verification.** Run a synthetic trace with an event exactly one second after `t0`; assert it cannot change the feature.

**Mitigation risk.** Removing the feature can reduce useful signal and increase manual workload.

**Escalation.** If prior actions used the feature, route the impact assessment to the model owner and risk owner.

#### VAL-D02 — Entity overlap masks poor generalization

**Trigger/environment.** Multiple transactions per customer are randomly split.

**Symptom.** Random-holdout recall is high while pilot performance on new customers is low.

**Competing causes.** New customers may truly differ.

The persistent identifier may be incomplete.

The model may memorize customer-specific behavior.

The pilot label process may differ.

**Discriminating diagnostic.** Compute entity overlap across partitions.

Repeat evaluation with all rows for each stable entity in one partition.

Audit residual linkage through household, device, or account mappings only where permitted.

**Root-cause evidence.** A large random-to-group performance gap with substantial overlap supports the memorization hypothesis.

It does not prove that every feature is invalid.

**Immediate mitigation.** Use group-based validation as the primary estimate.

**Correction.** Remove entity identifiers and redesign features toward transferable behavior.

**Prevention.** Make split keys part of the prediction contract.

**Verification.** Confirm zero entity overlap and report unresolved linkage rate.

**Mitigation risk.** A conservative group may shrink training data and increase estimate variance.

**Escalation.** If identity linkage itself is uncertain, involve data governance before expanding keys.

#### VAL-D03 — Label delay turns pending records into negatives

**Trigger/environment.** An outcome is finalized weeks after a prediction.

**Symptom.** Recent-window precision appears lower than older-window precision.

**Competing causes.** Real deterioration may exist.

Recent labels may simply be incomplete.

Reviewers may have changed adjudication criteria.

**Discriminating diagnostic.** Plot outcome maturity by decision age.

Recompute all metrics on cohorts older than the maximum agreed maturation delay.

Compare adjudication policy versions.

**Root-cause evidence.** A rate that stabilizes with cohort age supports censoring as a contributor.

**Immediate mitigation.** Mark recent metrics provisional and avoid threshold changes based on them alone.

**Correction.** Maintain a label-maturity watermark and delayed evaluation schedule.

**Prevention.** Make maturity a required denominator field.

**Verification.** Reproduce old metrics using only rows that were mature at the historical report date.

**Mitigation risk.** Delayed feedback slows detection of true drift.

**Escalation.** Add sampled expedited adjudication only if its selection bias is documented.

#### VAL-D04 — A global metric conceals an operational slice regression

**Trigger/environment.** Aggregate precision is unchanged after a release.

**Symptom.** One region’s queue age or false-negative complaint rate rises.

**Competing causes.** Population mix may have shifted.

The feature feed may be delayed locally.

The region may have different labels or capacity.

**Discriminating diagnostic.** Compare slice support, feature missingness, score distribution, threshold, queue age, and maturity by region.

Use the same model/policy version and time window for comparison.

**Root-cause evidence.** A localized missingness increase before the score shift supports a data-path cause.

**Immediate mitigation.** Gate the affected feature or policy for the slice if a safe fallback exists.

**Correction.** Repair source freshness and rerun point-in-time evaluation.

**Prevention.** Monitor source health and outcome metrics by operationally meaningful slice.

**Verification.** Re-evaluate the repaired slice on a later mature window without tuning on it repeatedly.

**Mitigation risk.** Slice-specific rules can introduce maintenance complexity and disparate treatment.

**Escalation.** Route consequential subgroup effects to the accountable policy owner.

### 11.9 Reproducibility manifest

Record a model result with sufficient detail to reconstruct its logic.

```text
run_id: VAL-2026-017
code_revision: immutable revision identifier
environment: dependency lockfile and execution image digest
data_snapshot: source extracts plus as-of timestamp
split_spec: forward windows, entity exclusion, random seed if applicable
feature_spec: names, availability rules, fit scope, transformations
label_spec: event definition, horizon, maturity watermark
model_spec: algorithm, parameters, stopping rule
policy_spec: threshold/top-k, capacity, dedupe, override policy
metrics_spec: numerator, denominator, slices, uncertainty procedure
artifacts: predictions, confusion matrices, calibration bins, error sample access path
limitations: unavailable sources, redactions, nondeterministic components
```

A random seed is not a complete reproducibility guarantee.

Parallel libraries can be nondeterministic.

Upstream backfills can change source snapshots.

Library versions can alter defaults.

An unavailable private source must remain unavailable in the manifest.

Do not substitute reconstructed data and call it the same run.

### 11.10 Additional interview defenses

13. **How would you validate a model after a policy change?** Separate model version from policy version, replay the same scores under fixed and new policies when possible, then use a post-change mature window. A raw outcome comparison can confound model quality with changed selection.

14. **Why is feature-store timestamping insufficient by itself?** It can state when a value was written but not whether its source event or correction was permissible for the earlier decision. Keep event, ingest, materialization, and availability semantics.

15. **A calibration residual is positive. What does that mean?** Predictions exceed observed frequency in the stated bin and cohort. Check sample size, label maturity, selection, and population before recalibrating.

16. **What is a valid negative for error analysis?** A record whose outcome has had sufficient opportunity and the same adjudication process to become positive. “No event yet” can be censored.

17. **Why preserve an untouched final window?** It bounds adaptation to validation feedback. It is not a guarantee of future performance if deployment differs from that window.

18. **What is the strongest baseline?** The current decision procedure under the same information, capacity, and outcome window, including manual/rule alternatives. A weaker baseline exaggerates apparent value.

19. **How do you test a suspected join explosion?** Profile join cardinality by key, compare pre/post row counts and outcome prevalence, and assert expected one-to-one or many-to-one semantics before model fitting.

20. **When should a model abstain?** When inputs violate the feature contract, uncertainty/eligibility policy requires review, or no safe action is available. Log abstention separately from a negative prediction.

**Takeaway:** A defensible validation result is reproducible, time-correct, capacity-aware, and explicit about what its evidence cannot establish.

### 11.11 Source-scope ledger

| Source ID | Source | Claim supported | Date/version | Limitation |
|---|---|---|---|---|
| INV-01 | [`directory.txt`](../../../../Templates/directory.txt#L575) | visible fraud feature/evaluation topic names | inventory consulted 2026-11-09 | names only; source bodies unavailable |
| MTH-01 | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | training-only preprocessing and leakage guidance | accessed 2026-11-09 | not evidence for this workload |
| MTH-02 | [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437) | proper scoring-rule context | 2007 | does not choose a business threshold |
| MTH-03 | [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874) | precision-recall comparison context | 2006 | does not establish calibration |

Every number in Sections 4, 5, 9, and 11 is constructed teaching arithmetic.

The arithmetic is hand-derived and internally checked in the stated traces.

No dataset, model run, benchmark, or deployment was executed.

The following oracle inputs are sufficient for an independent checker.

```text
metric_oracle:
  TP: 36
  FP: 12
  FN: 24
  expected_precision: 0.75
  expected_recall: 0.60
capacity_oracle:
  probabilities: [0.90, 0.70, 0.60, 0.40]
  capacity: 3
  expected_selected_probability_sum: 2.20
calibration_oracle:
  probabilities: [0.70, 0.80, 0.90, 0.80]
  labels: [1, 1, 0, 1]
  expected_residual: 0.05
storage_oracle:
  records: 2000000
  bytes_per_record: 1536
  expected_payload_bytes: 3072000000
```

### 11.12 Publication decision checklist

Before any publication decision, verify that every source link resolves.

Verify that the source supports the claim beside which it appears.

Verify all numerical examples with an independent calculator or script.

Verify that every feature has an as-of availability rule.

Verify that train, validation, and test decisions are time and entity disjoint where required.

Verify that preprocessing is fit only on training partitions.

Verify that threshold selection did not use the final holdout.

Verify metric denominators and label maturity.

Verify slice definitions, support, and action consequence.

Verify that capacity, deduplication, and overrides match the policy being evaluated.

Verify that source and policy versions appear in the reproducibility manifest.

Verify that the Quick Reference Card remains the final section.

**Takeaway:** A draft becomes reviewable only when its claims, calculations, lineage, and decision policy can be independently challenged.

## 12. Quick Reference Card

| Trigger | Rule | Use when | Critical assumption | Avoid when | Check | Next action |
|---|---|---|---|---|---|---|
| score improvement | audit availability before comparison | new feature/model | timestamps are trustworthy | source lineage is unknown | event/materialization ledger | rebuild point-in-time data |
| repeated entities | group split | entities recur | grouping key is stable | ID misses linkage | overlap rate | conservative linkage test |
| later deployment | forward split | time evolves | cutoff matches action | future label is used as feature | train/end dates | rolling replay |
| fixed review team | top-k/capacity policy | actions are scarce | demand interval known | capacity is unmeasured | queue count | set escalation rule |
| probability action | reliability analysis | magnitude drives value | labels mature | bin support is tiny | residual plus n | recalibrate/test later window |
| aggregate looks good | slice audit | credible subgroup harm | slice is prespecified | post-hoc fishing | support and uncertainty | investigate mechanism |
| repeated tuning | freeze final holdout | selection is complete | holdout untouched | analysts saw its score | change log | one final report |
