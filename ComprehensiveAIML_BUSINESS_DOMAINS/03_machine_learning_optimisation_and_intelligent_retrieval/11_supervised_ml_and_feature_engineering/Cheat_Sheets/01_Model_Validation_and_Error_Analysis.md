---
collection: "supervised-ml-and-feature-engineering"
file_id: "01"
topic: "Model validation and error analysis"
level: 2
status: "draft"
version: "3.0"
created: "2026-11-08"
reviewed: null
review_trigger: "A change to prediction time, label maturity, entity lineage, population, review capacity, or decision cost."
notation_style: "plain-text"
prerequisites: ["Probability and conditional rates", "Binary confusion matrix", "The prediction contract from 02_Problem_Formulation_and_Losses.md"]
pattern_tags: ["temporal-validation", "leakage-control", "capacity-thresholding", "calibration", "error-analysis"]
source_status: "OPEN"
owner_role: "ML reviewer"
source_index: "INDEX.md"
---

# Model Validation and Error Analysis

> **Purpose:** decide whether an offline estimate is credible for a stated deployment, then convert scores into a capacity-bounded action and a defensible error investigation.
> **Prerequisites:** conditional probability; confusion-matrix vocabulary; a written prediction contract naming the decision time, action, horizon, and population.
> **Key outcome:** Given event timestamps, entity identifiers, scores, maturity-stamped labels, a review capacity, and error costs, you can reject a leaky evaluation protocol, compute the operating-point metrics, derive a cost-consistent threshold, bound the uncertainty of the estimate, and name the next discriminating check.
> **This file does NOT cover:** choosing or fitting a classifier family (see the linear/tree/boosting sheet); estimating causal treatment effects; designing online experiments; certifying fairness or regulatory compliance.
> **Level:** 2 — Core Methods, with production diagnostics in Sections 8 and 9.
> **Version / reviewed:** 3.0 / not independently reviewed; the `reviewed` field stays null until a substantive review is recorded.
> **Evidence:** DERIVED metric identities; VERIFIED hand arithmetic in Sections 4, 5, and 9; ILLUSTRATIVE operating data; SOURCED method references; OPEN source-body coverage.
> **Retrieval cue:** If an offline number looks too good, start at Section 8 and run dossier VAL-D01 before touching the model.

## 1. File Header

### 1.1 What decision this sheet changes

Model validation is not a reporting step at the end of modelling. It is the discipline that decides whether a number is allowed to influence an action. The action in view throughout this sheet is concrete: a human or automated queue reviews a bounded number of scored records per day, and each review consumes labour while each miss consumes money. Every quantity below exists to answer one of four questions about that queue.

1. Does the evaluation measure the information the deployed system will actually have at decision time?
2. Does the evaluation unit match the unit that repeats in deployment, so the estimate is not inflated by memory of a familiar entity?
3. Does the reported metric correspond to the operating point the capacity allows, rather than to some threshold nobody will ever run?
4. Is the observed error pattern attributable to a mechanism that can be fixed, or is it noise being promoted into a story?

A sheet about validation that only lists metric formulas is a glossary. This one is organised around the sequence of commitments a reviewer must make and the order in which they can be invalidated.

### 1.2 Audience and assumed starting point

The reader is expected to have seen a confusion matrix before, to know that precision and recall trade off, and to have fitted at least one classifier. What is assumed missing is the habit of treating the split, the threshold, the label clock, and the cost model as one coupled object rather than four independent choices. Sections 3 and 4 build that object; Sections 5 through 9 exercise it; Section 11 tests whether the reader can defend it under challenge.

### 1.3 Scope boundary

| In scope | Out of scope | Where the excluded topic lives |
|---|---|---|
| Split construction under time, entity, and site dependence | Model family selection and regularisation | planned `03_Linear_Trees_and_Boosting.md` |
| Leakage detection through availability lineage | Feature transformation design itself | planned `04_Feature_Engineering_Patterns.md` |
| Operating-point metrics at a fixed review capacity | Loss function derivation from the decision | [02 Problem Formulation and Losses](02_Problem_Formulation_and_Losses.md) |
| Calibration diagnostics and residual interpretation | Full recalibrator comparison | planned `05_Imbalance_and_Calibration.md` |
| Uncertainty of a rate estimate under clustering | Experimental design and power analysis | module 02 experimentation material |
| Error clustering by mechanism | Causal attribution of harm | causal inference material |

### 1.4 Evidence policy used in this file

Every quantitative claim carries one of the course labels at the point of use. The distinction matters because a teaching example can be arithmetically exact and still be operationally meaningless. A precision of 0.40 computed from constructed counts is `VERIFIED` arithmetic attached to an `ILLUSTRATIVE` workload; it is not evidence that any fraud queue achieves 40% precision. Readers who quote such numbers without the workload label turn a teaching device into a fabricated benchmark, which is the most common way validation material becomes misleading.

Method references are cited for the specific claim they support. A citation for proper scoring rules does not endorse the cost ratio used later in this sheet, because the cost ratio is a business input rather than a statistical result. Where a source body was not available, the file says so instead of paraphrasing a title as if the contents had been read.

### 1.5 Notation convention

All display mathematics uses fenced `text` blocks with stable equation identifiers of the form `VAL-NN`. Symbols are defined at first use and repeated inside each formula block so that a copied block remains readable alone. Counts are integers of records; rates are dimensionless proportions in `[0,1]`; monetary amounts are stated in an unnamed currency unit `CU` so that no reader mistakes the illustration for a specific market.

### 1.6 Minimum viable validation record

Before reading further, a reader should be able to state the seven fields that make a validation result interpretable. If any is missing, the result is not yet a result.

| Field | Question it answers | Typical failure when omitted |
|---|---|---|
| Decision time `t0` | When is the action committed? | Features dated by calendar day silently include later events |
| Label horizon and maturity | When is the outcome knowable? | Recent rows are counted as negatives while pending |
| Evaluation unit | What repeats in deployment? | Row-level splits leak entity identity |
| Split rule | What dependence must the model survive? | Random split estimates memorisation |
| Operating policy | Which records actually get actioned? | AUC is reported for a queue that runs at a fixed count |
| Population and window | Who and when does the estimate describe? | A seasonal slice is generalised to the year |
| Cost or capacity basis | Why this threshold rather than another? | The threshold is inherited from a tutorial default |

**Takeaway:** An offline score is only a claim about a decision procedure; the seven fields above are the claim's terms and conditions.

## 2. Concept Map / Mental Model

### 2.1 The decision procedure as the unit of validation

The thing being validated is not a fitted function `f(x)`. It is the whole pipeline that turns a request at `t0` into an action: which sources may be read, how they are joined as of `t0`, which transformations are applied with parameters learned where, which model revision scores the row, which threshold or top-k rule converts the score into a queue position, and which override policy a human may apply. Two systems sharing identical coefficients but different thresholds have different error profiles and different expected cost.

```text
                 [request arrives at t0]
                             |
                             v
        +-------------------------------------------+
        | information boundary: sources readable    |
        | with event_time <= t0 and materialised    |
        | before the scoring call                   |
        +-------------------------------------------+
                             |
                    fails |            passes
                          v                v
              [rebuild lineage]   [transform with parameters
               or drop column]     fitted on training rows only]
                                           |
                                           v
                              [score s(x) under model revision]
                                           |
                                           v
                        +--------------------------------------+
                        | decision boundary: threshold tau or  |
                        | top-k against capacity C per interval|
                        +--------------------------------------+
                                           |
                             +-------------+-------------+
                             v                           v
                     [action taken]              [no action / abstain]
                             |                           |
                             v                           v
                    [outcome matures]  <--------- [outcome matures]
                             |
                             v
                 [evaluation on a dependence-matching split]
                             |
                             v
                 [metric] -> [slice audit] -> [error clustering]
                             |
                             v
                 [mechanism hypothesis] -> next cycle, never the holdout
```

The diagram makes the ordering non-negotiable. Threshold tuning cannot repair an information-boundary violation, because the scores it tunes are computed from data the deployment will not have. A perfect calibration map cannot repair entity leakage, because the probabilities being mapped were learned from memorised entities. Section 8 turns each arrow into a diagnostic.

### 2.2 Three boundaries that must agree

Validation has three boundaries, and the classic failure is that they are set independently.

The **information boundary** is temporal: what exists and is retrievable before the action. It is violated by future aggregates, by post-outcome status fields, and by late-arriving corrections applied retrospectively.

The **dependence boundary** is structural: which observations are not independent of each other. It is violated when the same customer, merchant, device, claimant, household, or site appears on both sides of a split, or when adjacent time periods share a shock.

The **decision boundary** is operational: which subset of scores becomes action. It is violated when metrics are computed at a threshold the queue never uses, or when deduplication and override rules are omitted from the replay.

A credible evaluation is the intersection of the three. Each boundary has its own failure vocabulary, and mixing them produces the most common confused diagnosis in practice: a reviewer sees a score collapse, assumes the model is weak, and retrains, when the actual cause was that the earlier number was measured across a broken information boundary.

### 2.3 Estimate, decision, and diagnosis as three separate outputs

A validation exercise produces three different artifacts, and conflating them wastes review effort.

The **estimate** is a number with a denominator: precision at the operating queue, recall over mature positives, expected cost per thousand decisions. It is descriptive.

The **decision** is a comparison with a baseline under the same information, capacity, and window. It is comparative, and it can be negative: a model that beats a random split but loses to the incumbent rule has no case.

The **diagnosis** is a mechanism claim about why errors concentrate where they do. It is explanatory and must be falsifiable.

Most weak validation reports contain an estimate and call it a decision, then invent a diagnosis from a handful of hand-inspected rows. Sections 5 and 8 separate these outputs explicitly.

### 2.4 The counterfactual baseline is part of the protocol

Every metric computed for a candidate model should also be computed for the procedure currently in force, including purely manual and rule-based procedures. The incumbent is not a formality. In the illustrative case of Section 5, the incumbent rule recovers 1140 of 1980 positives and generates 1360 false alerts at the same 2500-record capacity; the candidate recovers 1320 with 1180 false alerts. The 180-record swing is the entire measured benefit, and it is only visible because both procedures were replayed with the same capacity rather than each at its own preferred operating point. A baseline measured on a different window, or allowed unlimited capacity, will exaggerate the candidate's value in exactly the direction the reviewer wants to believe.

### 2.5 Why the mental model must include the label clock

Labels are not properties of a row. They are results of a process that takes time: a chargeback window, an adjudication queue, a 30-day readmission window, a warranty claim period. At any evaluation moment, some labels are mature, some are pending, and some will never be observed because the entity left the population. Treating pending as negative biases every rate in recent cohorts and creates a spurious "the model is degrading" narrative that reappears every time the reporting window rolls forward. The label clock is therefore part of the concept map, not an implementation detail of the metrics section.

**Takeaway:** Validate the procedure, align three boundaries, and keep the estimate, the comparison, and the mechanism claim separate.

## 3. Foundational Definitions

### 3.1 Core terms

| Term | Meaning | Units / type | Boundary or nonexample |
|---|---|---|---|
| `t0` | Instant the action is committed and must be justified | timestamp with timezone | A later adjudication time is not `t0` |
| prediction horizon `H` | Length of the outcome window starting at `t0` | duration | A 30-day outcome is not observable on day 3 |
| `Y` | Mature binary outcome over `H` | `{0,1}` | Pending or censored is not `0` |
| maturity time | Earliest instant `Y` is final for a decision at `t0` | timestamp | Extract date is not maturity time |
| `s(x)` | Model output before calibration | dimensionless, model-specific range | Not a probability and not comparable across revisions |
| `p(x)` | Calibrated estimate of the positive probability | `[0,1]` | Calibration holds for a population and period, not universally |
| `tau` | Action threshold on `p` or `s` | same units as the score | Changed by capacity, cost, or policy; not a model parameter |
| `C` | Review capacity per operational interval | records / interval | An upper bound, not a target |
| `TP, FP, FN, TN` | Post-policy counts against mature labels | records | Require a mature, deduplicated cohort |
| prevalence `pi` | Share of positives in the evaluated population | `[0,1]` | Population-specific; not a property of the model |
| entity | Persistent unit that repeats across rows | identifier | A row is not an entity |
| group split | All rows of an entity confined to one partition | partition rule | Row-level holdout of a repeated entity is not a group split |
| forward split | Training data strictly precedes the evaluation interval | time rule | Shuffled future rows violate it |
| leakage | Use of information unavailable or contaminated for the decision being evaluated | mechanism | A legitimate, available predictor is not leakage |
| slice | Prespecified subgroup tied to a mechanism or potential harm | population subset | Post-hoc selected rows are exploratory, not confirmatory |
| baseline | The procedure in force, replayed under identical conditions | procedure | A trivial constant predictor is a weak stand-in |
| holdout | Data untouched by any selection decision | dataset role | Reused repeatedly, it becomes a validation set |
| OOF prediction | Out-of-fold score for a training row | per-record value | Only leakage-free if preprocessing is refitted per fold |

### 3.2 Availability, event time, materialisation time, and correction time

A feature has at least four clocks, and validation depends on the slowest one.

The **event time** is when the underlying fact occurred. A card transaction at 09:14 has event time 09:14.

The **ingest time** is when the pipeline received it. A batch source may deliver it at 02:00 the next day.

The **materialisation time** is when the value became queryable in the serving store. A nightly job that finishes at 03:40 produces a materialisation time of 03:40.

The **correction time** is when a restatement replaced the original value, for example when a merchant reclassified a transaction three weeks later.

A feature is admissible for a decision at `t0` only if a version existed with materialisation time before `t0`. Using the corrected value in a backtest is not evidence of quality; it is evidence that the backtest used information the live system did not have. When the four clocks are not recorded separately, an apparent uplift after adding a "slow" feature is uninterpretable, and the only safe response is to exclude the feature until lineage is reconstructed.

### 3.3 Leakage taxonomy

Leakage is not one bug. The taxonomy below is the one used throughout Sections 8 and 11, and each type has a different discriminating check. The general formulation of leakage as a mismatch between the training-time and prediction-time data-generating process follows Kaufman, Rosset, and Perlich (2012); the operational checks here are derived for this course.

| Type | Mechanism | Typical symptom | First check |
|---|---|---|---|
| Target leakage | A column is derived from the outcome or from a downstream process that observed it | One feature dominates importance | Inspect the feature's producing job, not its correlation |
| Temporal leakage | Aggregates span the decision instant | Large uplift from "recent activity" features | Recompute with events strictly before `t0` |
| Entity leakage | The same unit appears in train and test | Random-split score far above group-split score | Overlap rate by persistent identifier |
| Preprocessing leakage | Imputation, scaling, or target encoding fitted on all rows | Score drops when pipeline is refitted per fold | Fold-scoped transform audit |
| Selection leakage | The evaluated population was filtered by a variable correlated with the outcome | Rate changes when filters are removed | Eligibility-rule replay |
| Label-selection leakage | Labels exist only for reviewed rows, and review depended on the score | Model appears excellent on reviewed rows | Compare with sampled unreviewed outcomes |
| Join leakage | A many-to-many join duplicates rows carrying positives | Row count and prevalence jump after a join | Join cardinality by key |
| Backfill leakage | Restated history used to reconstruct an as-of snapshot | Uplift only in backfilled periods | Snapshot version identifier per row |
| Provenance leakage | A feature is available for some entities but not others, and availability correlates with outcome | Missingness indicator ranks high | Missingness-versus-outcome audit |

### 3.4 Discrimination, calibration, and decision value

These three properties are often collapsed into the single word "performance". They answer different questions and can disagree sharply.

**Discrimination** is the ability to rank a positive above a negative. It is measured by AUC or by recall at a fixed false-positive budget. It is invariant to any strictly increasing transform of the score, so it cannot detect that scores are systematically shifted.

**Calibration** is the agreement between predicted probability and observed frequency. It is measured by reliability residuals, Brier score, or log loss. It is sensitive to shifts that leave ranking untouched.

**Decision value** is the expected cost or utility of the actions the policy produces at its actual operating point. It depends on calibration at the threshold and on capacity, which neither of the other two properties encodes.

A model can have good discrimination and still produce a Brier score worse than a constant predictor, as the constructed example in Section 4.4 shows. It can be well calibrated in aggregate and still be unprofitable at the top of the queue if the top band is miscalibrated. It can be profitable and still lose to the incumbent rule if the incumbent is strong in a sub-population the candidate ignores. Section 7 gives the selection logic for which property to require when.

### 3.5 Population, window, and the denominators that follow

Every rate in this sheet is a ratio. The numerator is a count of records satisfying a predicate; the denominator is a count of records in a stated population over a stated window. "Precision" without its denominator is a claim about a different quantity, and the most frequent reporting error in practice is substituting the wrong denominator: dividing false positives by all records rather than by actions, or calling the false-positive rate a false-positive share of the queue.

The population must also be stated. A model scored on all applications behaves differently on applications that pass an eligibility filter first. When the filter depends on a variable correlated with the outcome, the scored population's prevalence differs from the deployed population's, and precision computed on one does not transfer to the other. That is selection leakage in the taxonomy above, and it is invisible in aggregate confusion-matrix arithmetic.

### 3.6 Censoring, immaturity, and the definition of a negative

A negative is a record that had the full opportunity to become positive under the same adjudication process and did not. Three conditions must hold: the horizon elapsed, the adjudication process ran to completion for that record, and the observation channel was open.

A record whose horizon has not elapsed is **immature**. A record whose adjudication is still queued is **pending**. A record from an entity that left the observation channel, for example a closed account that cannot generate a future event, is **censored**. None of these is a negative.

Failing to distinguish them has a predictable consequence: the most recent cohort, which is also the cohort most similar to deployment, is systematically under-counted for positives, so its measured recall falls and its precision moves in whichever direction the action composition dictates. Teams then conclude the model has drifted and retrain on the same biased labels, which reproduces the bias. The fix is procedural, not statistical: publish a maturity watermark, evaluate only cohorts older than the maximum adjudication delay, and report the pending volume alongside every rate.

### 3.7 Capacity, marginal value, and the difference between a limit and a target

Capacity `C` is the number of records the review process can handle per interval. It enters the threshold decision as an inequality: the action set must satisfy a size bound.

Two errors follow from misreading capacity. Treating it as a target produces a policy that fills the queue with low-value items because the budget exists, which is exactly the behaviour that makes reviewers distrust the model. Treating it as irrelevant produces a cost-optimal threshold that drowns the queue and forces arbitrary triage, which destroys the measured precision because the effective action set is chosen by queue pressure rather than by score.

The correct reading is that capacity bounds the action set from above while the cost ratio determines the desirable action set from below; the operating point is the tighter of the two. Section 4.2 derives the cost side, and Section 5 shows a case where capacity binds, which is the normal state in a well-run queue rather than an exceptional one.

### 3.8 Slices, guardrails, and prespecification

A slice is a subgroup defined before results are inspected, tied to a mechanism or a plausible harm: a region with a different adjudication process, a device class with a different sensor, a customer-tenure band where behaviour differs. A guardrail is a slice metric with a threshold that blocks promotion if violated.

Prespecification matters because an unbounded search over subgroups guarantees finding a bad one. With enough slices, the worst observed rate will look alarming even when no slice differs in truth. Section 9.4 quantifies this and Section 11 asks the reader to defend a multiple-comparison position.

### 3.9 Two definitions that are commonly swapped

The **positive predictive value** is precision: among actioned records, the share positive. It answers "how much of my review effort is wasted". The **negative predictive value** is among un-actioned records, the share negative; it answers "how safe is my abstention". Both depend on prevalence, and both are column-conditional.

The **false omission rate** is the complement of the negative predictive value: among un-actioned records, the share positive. It is the operational quantity that matters for a queue that misses cases, and it is almost never reported. In the Section 5 cohort, the false omission rate is `720/17100 = 0.0421`, meaning 4.2% of everything the policy ignored was in fact a positive. A policy with high precision and a 4.2% false omission rate over 17,100 ignored records leaves 720 positives unaddressed, which is the number a risk owner should see.

**Takeaway:** Define the clock, the leakage type, the denominator, and the slice list before computing a single rate.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: confusion counts and the four rates that follow

**Validity before use.** The counts below describe one fixed population, one label definition, one horizon, and one threshold. They are arithmetic identities given those counts; they are not estimates of future performance. They are invalid as descriptions of a deployment when labels are immature, when the action policy changed during the collection window, or when the records were deduplicated differently from the live queue.

**Derivation.** Partition the mature cohort by the truth `Y` and the action `A = 1{s(x) >= tau}`. The four cells are exhaustive and mutually exclusive by construction, so `TP + FP + FN + TN = N` exactly. Among records that are truly positive (`TP + FN`), the action either fires or does not, giving recall as the fire share of that row. Among records where the action fires (`TP + FP`), the truth is either positive or not, giving precision as the true-positive share of that column. Because both are ratios of counts drawn from the same table, any statement comparing them must hold the row and column totals in view.

```text
FORMULA [VAL-01]: precision = TP / (TP + FP)
WHERE: TP [records] positive and actioned; FP [records] negative and actioned;
       denominator is the action count, not the cohort size.
CONSTRAINT: (TP+FP) > 0; labels mature for every actioned record; action set
            deduplicated exactly as in the live queue.
NUMERICAL EXAMPLE: TP=1320, FP=1180;
                   precision = 1320/(1320+1180) = 1320/2500 = 0.5280.
INTERPRETATION: nearly 53 reviews in 100 confirm a positive; the other 47 consume
                review capacity without recovering value.
```

```text
FORMULA [VAL-02]: recall = TP / (TP + FN)
WHERE: FN [records] positive with no action; denominator is the positive mass.
CONSTRAINT: the cohort contains every positive in the population and window,
            including those the action missed; no positive may be dropped by
            an eligibility filter that the model itself influenced.
NUMERICAL EXAMPLE: TP=1320, FN=660;
                   recall = 1320/(1320+660) = 1320/1980 = 0.666667.
INTERPRETATION: the policy recovers two thirds of mature positives; the remaining
                third are silent losses that produce no review ticket.
```

```text
FORMULA [VAL-03]: false_positive_rate = FP / (FP + TN)
WHERE: TN [records] negative with no action; this is a *row* rate over negatives.
CONSTRAINT: denominator must be the negative mass, not the action count; do not
            report this quantity under the name "false-positive share of queue".
NUMERICAL EXAMPLE: FP=1180, TN=16640;
                   FPR = 1180/(1180+16640) = 1180/17820 = 0.066218.
INTERPRETATION: 6.6% of all negatives are actioned. The queue-level equivalent is
                FP/(TP+FP) = 0.4720, a different number answering a different
                question with a different denominator.
```

```text
FORMULA [VAL-04]: F1 = 2 * precision * recall / (precision + recall)
WHERE: harmonic mean of the two rates above; no units; undefined if both are 0.
CONSTRAINT: only comparable between systems evaluated at the same prevalence,
            same threshold policy, and same maturity rule.
NUMERICAL EXAMPLE: precision=0.5280, recall=0.666667;
                   2*0.528*0.666667/(0.528+0.666667) = 0.704/1.194667 = 0.589286.
INTERPRETATION: F1 sits below the arithmetic mean 0.5973 because the harmonic mean
                penalises the weaker rate; it encodes an implicit equal weighting
                of precision and recall that this cost structure does not justify.
```

**Boundary/counterexample.** Take one action with `p = 0.999` and set `tau = 0.999`. Then `TP = 1`, `FP = 0`, `FN = 1979`, `TN = 17820`. Precision is `1.0000` and recall is `1/1980 = 0.000505`. A report that quotes only precision describes a policy that finds one case in nineteen hundred and eighty. Symmetrically, actioning every record gives recall `1.0000` and precision `1980/19800 = 0.1000`, which is exactly the prevalence — the model has added nothing to a random draw. Neither degenerate point is a validation success, and the pair of them shows why a single rate cannot summarise an operating point.

### 4.2 General case: capacity-constrained, cost-weighted operating point

**Validity before use.** The derivations in this subsection assume the score is calibrated on the evaluation population, that costs are stated per action and per miss in a common unit, and that capacity is a count per operational interval that matches the population window. If the score is uncalibrated, the capacity part still applies but the cost part must be re-expressed in score units, and the resulting threshold is a ranking cut rather than a probability cut.

**Derivation of the cost threshold.** Consider one record with calibrated probability `p`. Two actions are available: act, or abstain. Acting on a true positive yields value, but it is cleaner to work in costs. Write `c_FP > 0` for the cost incurred when the action is wrong, and `c_FN > 0` for the cost incurred when no action is taken on a true positive. Expected cost of acting is `(1 - p) * c_FP`; expected cost of abstaining is `p * c_FN`. The action is preferred exactly when the acting cost is smaller:

```text
(1 - p) * c_FP  <  p * c_FN
c_FP             <  p * (c_FP + c_FN)
p                >  c_FP / (c_FP + c_FN)
```

That inequality is the complete derivation; there is no approximation. It is the Bayes-optimal decision rule for binary cost-weighted classification, and it is the reason a threshold is a property of the cost structure rather than of the model.

```text
FORMULA [VAL-05]: tau_cost = c_FP / (c_FP + c_FN)
WHERE: c_FP [CU per false action]; c_FN [CU per missed positive]; tau_cost [probability].
CONSTRAINT: p is calibrated on the deployment population; costs are marginal and
            constant across records; no interaction between decisions; capacity
            is not binding.
NUMERICAL EXAMPLE: c_FP=3 CU, c_FN=27 CU;
                   tau_cost = 3/(3+27) = 3/30 = 0.100000.
INTERPRETATION: act on records whose calibrated probability exceeds 10.00%.
                Raising the miss cost to 60 CU moves the cut to 3/63 = 0.047619.
```

**Derivation of the capacity-constrained cut.** Capacity does not change the cost ordering; it truncates it. Let `s_(1) >= s_(2) >= ... >= s_(N)` be the sorted scores. The admissible action sets are the prefixes of this ordering, because any non-prefix set is dominated: swapping a lower score in for a higher score out cannot increase expected recovered value under a monotone calibration. The largest admissible prefix has length `min(|A(tau_cost)|, C)`. Hence the effective threshold is the score of the `C`-th record when the cost cut admits more than `C`, and the cost cut itself otherwise.

```text
FORMULA [VAL-06]: tau_eff = max(tau_cost, s_(C))
WHERE: s_(C) [score units] is the C-th largest score in the scored population;
       C [records/interval] is capacity; tau_eff is the operating cut.
CONSTRAINT: the population is scored with one model revision; capacity and the
            scored population use the same interval; ties at s_(C) are broken by
            a stated deterministic rule.
NUMERICAL EXAMPLE: tau_cost=0.1000 admits 4000 records while C=2500;
                   s_(2500)=0.32, so tau_eff=0.32 and 2500 records are actioned.
INTERPRETATION: capacity binds; the cost model says the queue should be larger,
                and the correct response is to report the unserved value, not to
                silently raise the threshold and call the result optimal.
```

**Derivation of expected cost at the operating point.** Given the confusion counts at `tau_eff`, total cost is additive over records because costs are assumed constant per error type:

```text
FORMULA [VAL-07]: Cost = c_FN * FN + c_FP * FP
WHERE: Cost [CU per evaluated cohort]; FN, FP [records].
CONSTRAINT: additive, constant marginal costs; excludes fixed review staffing and
            the cost of the evaluation itself; excludes benefit of true positives.
NUMERICAL EXAMPLE: c_FN=27, c_FP=3; FN=660, FP=1180;
                   Cost = 27*660 + 3*1180 = 17820 + 3540 = 21360 CU.
INTERPRETATION: 83.4% of the cost comes from missed positives, so effort spent
                improving recall is better rewarded than effort on precision.
```

**Derivation of expected positives at capacity.** Because expectation is linear regardless of dependence between records, the expected number of true positives among the actioned set is the sum of calibrated probabilities over that set:

```text
FORMULA [VAL-08]: E[TP | A] = sum over i in A of p_i
WHERE: p_i [probability] calibrated; A [set of actioned records]; result [records].
CONSTRAINT: calibration must hold on A itself, which is the highest-score region,
            and that is precisely where calibration is hardest to verify.
NUMERICAL EXAMPLE: 2500 selected probabilities averaging 0.5124 give
                   E[TP] = 2500 * 0.5124 = 1281.0 expected records, versus
                   TP = 1320 observed.
INTERPRETATION: the 39-record gap is 2.0% of the observed positives and 1.6% of
                the action count, consistent with band-level under-prediction
                plus sampling noise; a 200-record gap would indicate that
                calibration fails badly at the top of the queue.
```

**Boundary/counterexample.** The cost-optimal threshold and the accuracy-maximising threshold are different cuts whenever the costs are unequal, and the Section 5 cohort shows the gap. Under `c_FP = 3`, `c_FN = 27`, the cost-optimal cut is `0.1`, which actioned 4,000 records and reached an accuracy of `(1560 + 15380)/19800 = 0.855556`. The accuracy-maximising cut is `0.52`, which actioned 1,400 records and reached `(964 + 17384)/19800 = 0.926667`. Equal costs are the special case where the two coincide, because then minimising expected cost is exactly minimising 0-1 error. Choosing the 0.52 cut because its accuracy is higher would raise expected cost from 18,660 CU to 28,740 CU, a 54% increase, in exchange for a metric that does not appear in any decision rule. The objective, not the threshold, decides which cut is correct. This is the cleanest demonstration that "optimal threshold" has no meaning without naming the objective.

### 4.3 Variants: F-beta, precision-recall area, lift and gain, calibration residuals, proper scores

**Variants of the base.** Each variant below answers a question the base rates cannot, and each carries its own invalidation condition. The pattern to notice is that every variant is a re-weighting of the same two denominators.

```text
FORMULA [VAL-09]: F_beta = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
WHERE: beta > 0 [dimensionless] weights recall relative to precision; beta=1 gives
       VAL-04; beta=2 weights recall twice as heavily as precision.
CONSTRAINT: same prevalence and threshold across compared systems; a single
            aggregate hides the operating point that produced it.
NUMERICAL EXAMPLE: beta=2, precision=0.5280, recall=0.666667;
                   (1+4)*0.528*0.666667/(4*0.528+0.666667) = 1.76/2.778667 = 0.633397.
INTERPRETATION: the recall-leaning variant scores this policy above its F1 of
                0.589, which matches a cost structure where 83% of the cost is
                missed positives.
```

```text
FORMULA [VAL-10]: AP = sum over k of (R_k - R_{k-1}) * P_k
WHERE: records sorted by descending score; R_k [proportion] recall after the k-th
       record; P_k [proportion] precision at that prefix; AP is average precision.
CONSTRAINT: the label prevalence must match the deployment population; AP is not
            comparable across populations with different pi, unlike ROC AUC.
NUMERICAL EXAMPLE: 6 records with labels by descending score 1,1,0,1,0,0
                   (3 positives); prefixes give (P,R) = (1,0.3333), (1,0.6667),
                   (0.6667,0.6667), (0.75,1.0);
                   AP = 0.3333*1.0 + 0.3333*1.0 + 0*0.6667 + 0.3333*0.75
                      = 0.3333 + 0.3333 + 0 + 0.25 = 0.9167.
INTERPRETATION: AP summarises the whole ranking, but it weights the low-precision
                tail heavily; the queue only ever runs the first C records.
```

```text
FORMULA [VAL-11]: lift(k) = (positives in top-k / k) / pi
WHERE: k [records]; pi [proportion] overall prevalence; lift is dimensionless.
CONSTRAINT: k must be stated; lift at k=100 describes a different decision from
            lift at k=10% of the cohort; cohort must be mature.
NUMERICAL EXAMPLE: top decile k=1980 contains 540 positives; pi = 1980/19800 = 0.1;
                   rate = 540/1980 = 0.272727; lift = 0.272727/0.100000 = 2.727273.
INTERPRETATION: the top tenth is 2.73 times denser in positives than the cohort;
                gain at the same point is 540/1980 = 0.272727 of all positives.
```

```text
FORMULA [VAL-12]: residual_b = mean(p_i : i in b) - mean(Y_i : i in b)
WHERE: b [score interval] a fixed bin; Y_i in {0,1}; residual dimensionless,
       positive means over-prediction in that band.
CONSTRAINT: n_b large enough to bound sampling error; labels mature; the bin
            edges fixed before inspection; report n_b beside every residual.
NUMERICAL EXAMPLE: band [0.3,0.4) with n=1200, mean(p)=0.3520, mean(Y)=0.3100;
                   residual = +0.0420; standard error of the observed frequency
                   is sqrt(0.31*0.69/1200) = 0.01335, so the gap is 3.15 SE.
INTERPRETATION: a 4.2-point over-prediction that is unlikely to be noise; the
                band should be mapped downward before probabilities price anything.
```

```text
FORMULA [VAL-13]: Brier = (1/N) * sum over i of (p_i - Y_i)^2
WHERE: p_i [probability]; Y_i in {0,1}; Brier [squared probability units], lower
       is better; the constant predictor at pi scores pi*(1-pi).
CONSTRAINT: requires calibrated probabilities to be meaningful as a score; the
            decomposition below assumes fixed bins with adequate support.
NUMERICAL EXAMPLE: p=[0.90,0.80,0.60,0.30], Y=[1,1,0,0];
                   squared errors 0.01, 0.04, 0.36, 0.09; mean = 0.50/4 = 0.1250.
INTERPRETATION: the model beats the constant predictor at pi=0.5, which scores
                0.2500; Brier rewards both good ranking and honest magnitudes.
```

**Murphy decomposition.** For bins `b` with `n_b` records and observed frequency `o_b`, the Brier score splits exactly into three terms when the bin means are used as the forecast in each bin:

```text
FORMULA [VAL-14]: Brier = sum_b (n_b/N)(p_b - o_b)^2
                        - sum_b (n_b/N)(o_b - pi)^2
                        + pi(1 - pi)
WHERE: p_b [probability] bin mean forecast; o_b [proportion] bin observed rate;
       first term reliability (lower better), second resolution (higher better,
       entered with a minus), third uncertainty (irreducible).
CONSTRAINT: exact only for bin-constant forecasts; bin choice changes the split
            without changing the total; sparse bins inflate reliability spuriously.
NUMERICAL EXAMPLE: two bins of 9900 each, pi=1/11=0.090909, uncertainty=0.082645;
                   bin A p=0.30, o=0.32 -> reliability (0.02)^2=0.000400,
                   resolution (0.32-0.090909)^2=0.052483;
                   bin B p=0.05, o=0.04 -> reliability (0.01)^2=0.000100,
                   resolution (0.04-0.090909)^2=0.002593;
                   reliability=(0.0004+0.0001)/2=0.000250;
                   resolution=(0.052483+0.002593)/2=0.027538;
                   Brier = 0.000250 - 0.027538 + 0.082645 = 0.055357.
INTERPRETATION: the model is nearly perfectly reliable here and earns its score
                almost entirely from resolution; recalibration would change little,
                while a better ranking would lower Brier further.
```

**Takeaway:** Every variant is a re-weighting of the same two denominators, and every one of them becomes meaningless without its prevalence, threshold, and maturity.

### 4.4 Limits, counterexamples, and the failures of each variant

**Limit 1 — perfect ranking with terrible probabilities.** Take four records with scores `[0.99, 0.99, 0.98, 0.97]` and labels `[1, 1, 0, 0]`. Every positive outranks every negative, so the ROC AUC is `1.0000`. The Brier score is `(0.0001 + 0.0001 + 0.9604 + 0.9409)/4 = 1.9015/4 = 0.475375`, while the constant predictor at the base rate `pi = 0.5` scores `0.2500`. A model with flawless discrimination is nearly twice as bad as predicting the base rate for every row. Any downstream rule that multiplies probabilities by a value, sums them for an expected total, or compares them against a cost ratio will be badly wrong, even though every ranking-based metric reports perfection.

**Limit 2 — a single aggregate at an unstated threshold.** ROC AUC is threshold-free, and that is exactly its limitation. With `C = 2700` out of `N = 19800`, the queue operates at the 13.64th percentile of the score distribution. Two models with identical AUC can differ by ten points of precision at that percentile. The AUC's pairwise weighting also counts comparisons deep in the negative tail, which the queue never reaches. This is why AUC is a screening metric and not an operating-point metric.

**Limit 3 — AP is prevalence-bound.** Average precision integrates precision over recall, and precision depends on prevalence. The same ranking evaluated on a population with `pi = 0.30` and one with `pi = 0.03` reports very different AP values. ROC AUC does not have this property. Comparing AP across populations, or across time windows with different prevalence, is therefore invalid, and this is the most common way a PR curve misleads a migration review.

**Limit 4 — lift without its `k`.** `lift(100)` on a cohort of 19,800 and `lift(1980)` on the same cohort are different quantities, and only one of them corresponds to a real queue. A vendor quoting "lift of 6" without a depth is quoting an unidentifiable number.

**Limit 5 — calibration residual with a tiny bin.** A bin holding 30 records can show a residual of `0.15` purely from sampling. The standard error of an observed frequency near 0.3 with `n = 30` is `sqrt(0.3*0.7/30) = 0.0837`, so a 0.15 gap is under two standard errors. Recalibrating on that evidence can degrade a model that was already well calibrated.

**Limit 6 — the cost threshold with a non-calibrated score.** Applying `tau_cost = 0.2647` to an arbitrary score with range `[-3, 3]` selects nothing. The threshold formula is only defined on `p`, and the frequent substitution of a raw score for `p` here is why teams report "the cost-optimal threshold performs terribly". It does not perform terribly; it was never applied.

**Limit 7 — costs that are not constant.** `VAL-05` assumes `c_FP` and `c_FN` do not depend on the record. In practice a missed positive among high-value accounts costs more than one among low-value accounts, and a false alert on a long-standing customer costs more in friction than one on a new account. The derivation then becomes per-record: act when `p_i * c_FN,i > (1 - p_i) * c_FP,i`, and the single scalar threshold disappears. Reporting one threshold in that setting is a deliberate simplification that must be labelled.

**Takeaway:** Every metric in this section is a conditional statement; the condition, not the number, is what a reviewer must challenge first.

## 5. Worked Numerical Example

### 5.1 The setting

**Evidence: ILLUSTRATIVE cohort with VERIFIED arithmetic.** A payments team scores merchant-initiated transactions for a manual review queue. The positive class is a confirmed fraudulent transaction within 30 days of the decision. Review capacity is 2,500 records per day. A wasted review costs 3 CU in loaded analyst time; a missed positive costs 27 CU in average unrecovered loss. The decision is which records to send to review, evaluated on a mature cohort.

Every number below is constructed for teaching. No system was run, no dataset was accessed, and the values are not benchmarks. The arithmetic is exact at the stated precision and is independently checked in Section 5.13.

### 5.2 Step 1 — build the cohort and verify the totals

The evaluation cohort contains 19,800 scored decisions with mature labels. Of these, 1,980 are true positives, so the prevalence is

```text
pi = 1980 / 19800 = 0.100000
```

The cohort is partitioned into seven score bands. Within each band the model's predicted probability is constant at the stated value, and the observed positive count is recorded. The bands are sorted from highest to lowest score.

| Band | Predicted `p` | Records `n_b` | Observed positives `o_b n_b` | Observed rate `o_b` |
|---|---|---|---|---|
| A | 0.95 | 200 | 196 | 0.980000 |
| B | 0.75 | 500 | 390 | 0.780000 |
| C | 0.52 | 700 | 378 | 0.540000 |
| D | 0.32 | 1,100 | 356 | 0.323636 |
| E | 0.16 | 1,500 | 240 | 0.160000 |
| F | 0.06 | 2,600 | 156 | 0.060000 |
| G | 0.02 | 13,200 | 264 | 0.020000 |

Two totals must hold before any metric is computed, and both are checked here.

```text
sum of n_b      = 200 + 500 + 700 + 1100 + 1500 + 2600 + 13200 = 19800   [OK]
sum of positives = 196 + 390 + 378 + 356 + 240 + 156 + 264      = 1980    [OK]
```

The second check is the one that catches an accidental join explosion. If a many-to-many join had duplicated positive rows, the positive total would exceed 1,980 while the record total rose by the same duplicated count, and prevalence would drift upward. Here prevalence returns exactly 0.1, so no duplication is present in the constructed cohort.

The observed rates are deliberately not equal to the predicted probabilities. Bands A, B, C, and D all under-predict, with the gap largest at the top of the distribution; bands E, F, and G are exactly calibrated by construction. That arrangement gives the calibration analysis in Step 7 something real to find, and it reproduces a common production shape: models are often too timid at the top of the distribution, where labels are dense and the training signal is strongest.

### 5.3 Step 2 — locate the operating point from capacity

Sort the bands from highest score downward and accumulate. The queue fills from the top.

| Band | `n_b` | Cumulative records | Cumulative positives |
|---|---|---|---|
| A | 200 | 200 | 196 |
| B | 500 | 700 | 586 |
| C | 700 | 1,400 | 964 |
| D | 1,100 | 2,500 | 1,320 |
| E | 1,500 | 4,000 | 1,560 |

Capacity is exactly 2,500, which lands on the boundary between D and E. The operating threshold is therefore the bottom edge of band D, `tau_eff = 0.32`, and the action set is the 2,500 highest-scoring records. This is a deliberately convenient alignment: in practice capacity falls inside a band, and the tie-breaking rule at the boundary must be stated because records sharing a score are indistinguishable to the policy.

If capacity had been 2,700, the extra 200 records would come from somewhere inside band D. Their expected positive count would be `200 * 0.323636 = 64.727`, giving `TP = 1384.727` in expectation and a fractional count that cannot be observed. Reporting fractional confusion counts is a signal that the analyst has mixed expectation with observation. Section 5.12 handles this by reporting expected values explicitly labelled as such.

### 5.4 Step 3 — confusion arithmetic with independent checks

At `tau_eff = 0.32`:

```text
TP = 1320
FP = actions - TP          = 2500 - 1320 = 1180
FN = positives - TP        = 1980 - 1320 = 660
TN = negatives - FP        = 17820 - 1180 = 16640
```

Four checks must pass before these counts are used anywhere.

```text
check 1 (column):  TP + FP = 1320 + 1180 = 2500  = capacity          [OK]
check 2 (row):     TP + FN = 1320 +  660 = 1980  = positives         [OK]
check 3 (row):     FP + TN = 1180 +16640 = 17820 = negatives         [OK]
check 4 (total):   2500 + 17300 = 19800                              [OK]
```

Check 4 uses the ignored mass: `FN + TN = 660 + 16640 = 17300`, and `2500 + 17300 = 19800`. When all four checks pass, the table is internally consistent. They pass trivially here because the counts were derived from the totals, and that is precisely why they are cheap: they catch transcription errors, capacity that was silently changed, and cohorts that include records outside the stated population.

### 5.5 Step 4 — operating-point rates

Now apply the four rate formulas from Section 4.1 to this table.

```text
precision        = TP/(TP+FP) = 1320/2500   = 0.528000
recall           = TP/(TP+FN) = 1320/1980   = 0.666667
false-pos. rate  = FP/(FP+TN) = 1180/17820  = 0.066218
F1               = 2*0.528*0.666667/(0.528+0.666667)
                 = 0.704000/1.194667        = 0.589286
false omission   = FN/(FN+TN) = 660/17300   = 0.038150
```

Read them as a coherent story rather than five independent scores. Roughly half of every review is wasted (`1 - 0.528 = 0.472`). Two thirds of the positives are caught. Of everything the policy chose to ignore, 3.8% was positive, which over 17,300 ignored records amounts to 660 missed cases. That last number is the one that reaches a risk committee, because it is a count of things that happened and were not looked at.

The relationship between precision and the false-positive rate deserves emphasis because it is the most frequently confused pair. The FPR is `0.0662`, a small-looking number. The queue-level waste is `0.472`, a large-looking number. Both are correct. The FPR divides by 17,820 negatives and the waste divides by 2,500 actions, and neither denominator is available unless the analyst states it. A report that shows `FPR = 6.6%` without the queue waste is technically accurate and operationally misleading, because the reviewers experience the waste, not the rate.

### 5.6 Step 5 — cost model and the cost-optimal cut

The cost basis is `c_FP = 3` CU and `c_FN = 27` CU. By `VAL-05`:

```text
tau_cost = c_FP / (c_FP + c_FN) = 3 / (3 + 27) = 3/30 = 0.100000
```

Act on every record whose calibrated probability is at least 0.1. Scanning the bands, that includes A, B, C, D, and E, and excludes F and G. So the cost-optimal unconstrained action set contains `4000` records. Compute its confusion table and cost:

```text
TP* = 196 + 390 + 378 + 356 + 240 = 1560
FP* = 4000 - 1560                  = 2440
FN* = 1980 - 1560                  = 420
cost* = 27*420 + 3*2440 = 11340 + 7320 = 18660 CU
```

Now evaluate the full sweep to confirm the cut is the minimum rather than merely a good value. Each row applies one threshold at a band boundary.

| `tau` | Actions | TP | FP | Precision | Recall | Cost (CU) |
|---|---|---|---|---|---|---|
| 0.95 | 200 | 196 | 4 | 0.980000 | 0.098990 | 48,180 |
| 0.75 | 700 | 586 | 114 | 0.837143 | 0.295960 | 37,980 |
| 0.52 | 1,400 | 964 | 436 | 0.688571 | 0.486869 | 28,740 |
| 0.32 | 2,500 | 1,320 | 1,180 | 0.528000 | 0.666667 | 21,360 |
| 0.16 | 4,000 | 1,560 | 2,440 | 0.390000 | 0.787879 | **18,660** |
| 0.06 | 6,600 | 1,716 | 4,884 | 0.260000 | 0.866667 | 21,780 |
| 0.02 | 19,800 | 1,980 | 17,820 | 0.100000 | 1.000000 | 53,460 |

The minimum sits at `tau = 0.16`, which contains exactly the records with `p >= 0.1`. Two independent derivations therefore agree: the analytic threshold from `VAL-05` and the tabulated minimum of `VAL-07`. The last row is the informative one. Actioning everything achieves recall `1.0000` and cost `53,460`, nearly three times the optimum, because 17,820 false alerts at 3 CU each swamp the 21,600 CU of recovered loss.

Note also what the sweep shows about precision: it falls monotonically from 0.98 to 0.10 as the queue widens, which is exactly the expected geometry. Recall rises. Neither curve alone identifies the optimum; only the cost column does, and the cost column is not a property of the model.

### 5.7 Step 6 — capacity as a binding constraint

The cost-optimal set contains 4,000 records. Capacity is 2,500. Capacity binds, so by `VAL-06` the effective threshold is the capacity cut, `tau_eff = 0.32`, and the operating cost is the row computed in Step 5:

```text
cost at capacity = 21,360 CU
cost unconstrained optimum = 18,660 CU
capacity penalty  = 21,360 - 18,660 = 2,700 CU per cohort
```

Decompose the penalty to prove it is not an artefact. The records excluded by capacity are exactly band E: 1,500 records containing 240 positives.

```text
delta_cost = - (positives gained)*c_FN + (negatives gained)*c_FP
           = - 240*27 + 1260*3
           = - 6480 + 3780
           = - 2700 CU
```

The negative sign means that adding band E would reduce cost by 2,700. The decomposition matches the direct subtraction exactly, which is the check that the two computations describe the same set of records. Had they differed, some record would have been double-counted or omitted.

The operational reading matters more than the arithmetic. The queue is not misconfigured; it is capacity-limited, and the correct report says "we are forgoing 2,700 CU of net value per cohort because 1,500 high-value records cannot be reviewed". The wrong report says "the threshold is optimal", which hides a staffing decision inside a modelling artifact. A reviewer who cannot distinguish these two statements will keep asking for a better model when the binding constraint is headcount.

### 5.8 Step 7 — calibration band by band

The aggregate view hides the miscalibration that matters. Compute residuals by `VAL-12`:

| Band | `p_b` | `o_b` | Residual `p_b - o_b` | `n_b` | SE of `o_b` | Residual in SE |
|---|---|---|---|---|---|---|
| A | 0.95 | 0.980000 | -0.030000 | 200 | 0.009899 | -3.03 |
| B | 0.75 | 0.780000 | -0.030000 | 500 | 0.018529 | -1.62 |
| C | 0.52 | 0.540000 | -0.020000 | 700 | 0.018840 | -1.06 |
| D | 0.32 | 0.323636 | -0.003636 | 1,100 | 0.014107 | -0.26 |
| E | 0.16 | 0.160000 | 0.000000 | 1,500 | 0.009467 | 0.00 |
| F | 0.06 | 0.060000 | 0.000000 | 2,600 | 0.004657 | 0.00 |
| G | 0.02 | 0.020000 | 0.000000 | 13,200 | 0.001219 | 0.00 |

The standard errors use `sqrt(o_b(1-o_b)/n_b)`; for band A that is `sqrt(0.98*0.02/200) = sqrt(0.000098) = 0.009899`. Band A's residual is three standard errors below zero, so the top band under-predicts in a way that sampling noise does not explain. Bands B through G are within ordinary variation.

This finding has a specific decision consequence: the probability assigned to the highest-scoring records is too low, so any rule that prices expected value from `p` will under-value the top of the queue. It does **not** justify recalibrating the whole model, because six of seven bands are already aligned. A band-local correction, or a monotone recalibration fitted on a separate calibration partition, addresses the defect without disturbing bands that are fine. Section 7.3 compares the available approaches.

The aggregate residual is a different number and nearly zero:

```text
mean predicted = 1941/19800 = 0.098030
mean observed  = 1980/19800 = 0.100000
aggregate residual = -0.001970
```

Reporting only the aggregate would have declared the model calibrated and hidden a three-sigma defect exactly where decisions are made. This is the single most important lesson in the section: **aggregate calibration is not operating-point calibration**.

### 5.9 Step 8 — Brier score and its decomposition

The Brier score at the bin level, using `VAL-13` per record and the bin-mean forecast, yields per-band mean squared errors computed as `o_b(1-p_b)^2 + (1-o_b)p_b^2`:

| Band | `p_b` | `o_b` | Mean squared error | Weight `n_b/N` | Contribution |
|---|---|---|---|---|---|
| A | 0.95 | 0.980000 | 0.020500 | 0.010101 | 0.000207 |
| B | 0.75 | 0.780000 | 0.172500 | 0.025253 | 0.004356 |
| C | 0.52 | 0.540000 | 0.248800 | 0.035354 | 0.008796 |
| D | 0.32 | 0.323636 | 0.218909 | 0.055556 | 0.012162 |
| E | 0.16 | 0.160000 | 0.134400 | 0.075758 | 0.010182 |
| F | 0.06 | 0.060000 | 0.056400 | 0.131313 | 0.007406 |
| G | 0.02 | 0.020000 | 0.019600 | 0.666667 | 0.013067 |

Summing the last column gives `Brier = 0.056175`. The constant predictor at prevalence scores `0.1 * 0.9 = 0.090000`, so the model improves on it by `0.033825`, or 37.6% of the constant score. Work band A explicitly to confirm the per-band formula: `0.98*(0.05)^2 + 0.02*(0.95)^2 = 0.98*0.0025 + 0.02*0.9025 = 0.00245 + 0.01805 = 0.020500`. The first term is the error on the 98% of records that are positive (forecast 0.95, truth 1), the second on the 2% that are negative. A common error is to compute `(p-o)^2` instead, which gives `0.0009` for band A and understates the score by a factor of 22; `(p-o)^2` is the reliability term, not the Brier score.

Now decompose with `VAL-14`:

```text
reliability = (200/19800)(0.03)^2 + (500/19800)(0.03)^2
            + (700/19800)(0.02)^2 + (1100/19800)(0.003636)^2
            = 0.00000909 + 0.00002273 + 0.00001414 + 0.00000073
            = 0.00004669

resolution  = (200/19800)(0.88)^2 + (500/19800)(0.68)^2
            + (700/19800)(0.44)^2 + (1100/19800)(0.223636)^2
            + (1500/19800)(0.06)^2 + (2600/19800)(-0.04)^2
            + (13200/19800)(-0.08)^2
            = 0.0078223 + 0.0116769 + 0.0068446 + 0.0027785
            + 0.0002727 + 0.0002101 + 0.0042667
            = 0.0338718

uncertainty = pi(1 - pi) = 0.1 * 0.9 = 0.0900000

Brier = 0.00004669 - 0.0338718 + 0.0900000 = 0.05617489
```

The decomposition reproduces the direct sum to five decimal places, which is the exactness check `VAL-14` demands. It also states where the model's value comes from: reliability is almost zero (the model is nearly perfectly calibrated on average), resolution is `0.0339` (the model separates bands well), and uncertainty is fixed by prevalence. Two conclusions follow.

First, recalibrating would improve Brier by at most `0.00004669`, which is negligible against a resolution of `0.0339`. A proposal to spend a quarter on a recalibration project because "Brier is not great" is aimed at the wrong term. Second, a model with the same reliability and half the resolution would score `0.00004669 - 0.01694 + 0.09 = 0.076106`, and that is the change worth engineering for.

### 5.10 Step 9 — uncertainty with entity clustering

Precision is a proportion over 2,500 actions. If those actions were independent, its standard error would be

```text
SE_ind = sqrt(0.528 * 0.472 / 2500) = sqrt(0.0000996864) = 0.009984
95% interval: 0.528 +/- 1.96*0.009984 -> [0.5084, 0.5476]
```

But records are nested in merchants, and merchants repeat. With an average of `m = 4` records per merchant and an intraclass correlation of `ICC = 0.2667`, the design effect is

```text
deff = 1 + (m - 1)*ICC = 1 + 3*0.2667 = 1.8001
n_eff = 2500 / 1.8001 = 1388.8
SE_cluster = sqrt(0.528 * 0.472 / 1388.8) = sqrt(0.00017948) = 0.013396
95% interval: 0.528 +/- 1.96*0.013396 -> [0.5017, 0.5543]
```

The cluster-adjusted interval is 34% wider. This changes a comparison: if an incumbent rule scored 0.551 on the same cohort, the independent-sample interval `[0.5084, 0.5476]` would exclude it while the cluster-adjusted interval `[0.5017, 0.5543]` would contain it, and the honest conclusion would be "no demonstrated difference at this sample size". That is a materially different sentence from "the candidate is better", and the only thing separating them is a variance calculation nobody enjoys doing.

### 5.11 Step 10 — baseline comparison on identical terms

The incumbent rule is replayed on the same 2,500-record capacity, same cohort, same deduplication, same maturity watermark. It produces `TP = 1140` and `FP = 1360`.

| Quantity | Incumbent | Candidate | Difference |
|---|---|---|---|
| Actions | 2,500 | 2,500 | 0 |
| TP | 1,140 | 1,320 | +180 |
| FP | 1,360 | 1,180 | -180 |
| FN | 840 | 660 | -180 |
| Precision | 0.456000 | 0.528000 | +0.072000 |
| Recall | 0.575758 | 0.666667 | +0.090909 |
| Cost (CU) | 27*840 + 3*1360 = 26,760 | 21,360 | -5,400 |

The candidate costs 5,400 CU less per cohort, or 20.2% of the incumbent cost. Since a cohort is one day here, that annualises to `5400 * 365 = 1,971,000` CU under the strong assumption that the cohort is representative of every day, that costs are constant, and that the queue is never capacity-constrained elsewhere. The annualisation is arithmetic, not a forecast, and stating it without those assumptions is how illustrative numbers become promises.

The comparison is honest only because of four conditions: identical capacity, identical cohort, identical maturity rule, and identical deduplication. Change any one and the difference absorbs the change. If the incumbent had been allowed unlimited actions it would have lower cost by construction, and the comparison would be meaningless. If the candidate had been evaluated on a newer, less mature cohort its recall would be understated.

### 5.12 Step 11 — sensitivity and what would flip the decision

Three inputs could reverse the conclusion, and each is checked rather than asserted.

**Cost ratio.** Holding the bands and capacity fixed, the cost-optimal cut moves with `c_FN/c_FP`. At `c_FP = 3`, `c_FN = 27`, the cut is 0.1 and capacity binds. Raise `c_FN` to 60: `tau_cost = 3/63 = 0.047619`, which still includes band F at 0.06 and now admits 6,600 records. The unconstrained optimum then costs `60*264 + 3*4884 = 15840 + 14652 = 30492` CU, while the capacity-limited policy costs `60*660 + 3*1180 = 39600 + 3540 = 43140` CU, so the capacity penalty widens from 2,700 to `43140 - 30492 = 12648` CU. Lower `c_FN` to 9: `tau_cost = 3/12 = 0.25`, which admits bands A through D — exactly 2,500 records, the capacity set — so capacity stops binding and the penalty falls to zero. The same model, the same scores, and the same queue therefore produce opposite staffing conclusions from a cost input that no model can estimate.

**Prevalence.** If the deployed population has prevalence 0.03 rather than 0.10, precision at the same threshold falls sharply while recall is roughly preserved, because the negative mass beneath the threshold grows. Recomputing the band table at a lower prior is a scoring and calibration exercise, not a resampling one, and it must not be faked by dropping rows. The correct statement is that the operating point must be re-derived on a representative cohort.

**Capacity.** If capacity rose to 4,000, the policy would reach the unconstrained optimum and the capacity penalty would vanish. Cost would fall to 18,660 CU, a further 2,700 CU saving, so the marginal value of the 1,500 additional reviews is 1.8 CU per review on average. That figure is the number to take to a staffing discussion, and it exists only because the analysis separated the capacity penalty from the model quality.

### 5.13 Independent check block

The following oracle reproduces every computed quantity in this section. It is stated so that a second person can verify the arithmetic without reading the prose.

```text
cohort_oracle:
  records: 19800
  positives: 1980
  bands: [[0.95,200,196],[0.75,500,390],[0.52,700,378],[0.32,1100,356],
          [0.16,1500,240],[0.06,2600,156],[0.02,13200,264]]
  expected_prevalence: 0.1
  expected_band_record_sum: 19800
  expected_band_positive_sum: 1980

operating_point_oracle:
  capacity: 2500
  tp: 1320    fp: 1180    fn: 660    tn: 16640
  expected_precision: 0.528
  expected_recall: 0.6666667
  expected_fpr: 0.0662177
  expected_f1: 0.5892857
  expected_false_omission_rate: 0.0381503

cost_oracle:
  c_fp: 3     c_fn: 27
  expected_tau_cost: 0.1
  expected_actions_unconstrained: 4000
  expected_cost_unconstrained: 18660
  expected_cost_at_capacity: 21360
  expected_capacity_penalty: 2700

calibration_oracle:
  expected_mean_predicted: 0.0980303
  expected_aggregate_residual: -0.0019697
  expected_band_A_residual: -0.03
  expected_band_A_se: 0.0098995

proper_score_oracle:
  expected_brier: 0.0561749
  expected_constant_predictor_brier: 0.09
  expected_reliability: 0.0000467
  expected_resolution: 0.0338718
  expected_uncertainty: 0.09

uncertainty_oracle:
  design_effect: 1.8001
  effective_n: 1388.8
  expected_se_independent: 0.009984
  expected_se_clustered: 0.013396

expected_tp_at_capacity: 1281
observed_tp_at_capacity: 1320
```

An independent checker can confirm the prevalence by summing the third column, confirm precision by division, confirm that the two cost paths differ by exactly the band-E contribution, and confirm that the Murphy terms sum to the direct Brier score. Any mismatch identifies the step that is wrong.

**Takeaway:** A worked validation produces a table that four independent identities must satisfy, and the interesting numbers are the residuals, the capacity penalty, and the cluster-adjusted interval, not the headline precision.

## 6. Visual / Geometric Intuition

### 6.1 The score axis and the threshold cut

Picture the cohort as points on a horizontal score axis. Positives and negatives overlap; the threshold is a vertical cut. Everything to the right becomes an action. The four confusion counts are the four regions the cut defines.

```text
score 0.0                                                          1.0
 |--G-------F---------E------------D----------C------B----A---------|
 |  p=.02   p=.06    p=.16        p=.32      p=.52  p=.75  p=.95   |
 |<-------------- ignored: 17,300 records ----------->|<-- 2,500 -->|
 |     660 positives among the ignored                 | 1,320 pos  |
 |                                                      | 1,180 neg  |
 |                                                          ^       |
 |                                                        tau_eff   |
 |                                                      = 0.32      |

 queue waste  = 1,180 / 2,500 = 0.472
 silent misses =   660 / 17,300 = 0.038 of the ignored mass
```

The geometry makes two things obvious that prose often obscures. First, the cut is a single scalar, so it cannot be chosen to optimise two rates simultaneously; moving right trades recall for precision at a rate set by the local density. Second, the widths of the bands are grossly unequal. Band G holds two thirds of the records and almost none of the positives, so any statement about "the average record" is a statement about band G, and any statement about "the average positive" is a statement about bands A through D. Reporting one number for both is a category error.

### 6.2 ROC and precision-recall geometry, and why prevalence changes only one

Draw the ROC curve by sweeping the threshold and plotting recall (true-positive rate) against the false-positive rate. Draw the PR curve plotting precision against recall. Both are generated by the same sweep, and they are related by the identity

```text
precision = recall * pi / (recall * pi + FPR * (1 - pi))
```

which is algebraic: `TP/(TP+FP)` divided above and below by the positive mass gives `TPR*pi` over `TPR*pi + FPR*(1-pi)`. The formula contains `pi` explicitly. That is the whole explanation of why ROC is prevalence-invariant and PR is not. Rotating the curve's vertical compression without moving any point in ROC space is possible for one and impossible for the other.

```text
ROC space (pi-invariant)          PR space (pi-dependent)
 1 |        ****                   1 |*
   |     ***                        | **
   |   **                           |   ***
   | **                             |      ****
   |*                               |          ********
 0 +------------ 1               0 +-------------------- 1
    FPR                                recall
```

Two operational consequences follow. Comparing ROC AUC across populations with different prevalence is legitimate for discrimination only. Comparing PR AUC across such populations is not, and the difference is often large: at `pi = 0.10` the PR baseline is 0.10; at `pi = 0.01` the same ranking produces a curve whose random baseline is 0.01, so the visual contrast is misleading and average precision values are not comparable across the two.

### 6.3 Reliability geometry

Plot predicted probability on the horizontal axis and observed frequency on the vertical, one point per band, sized by `n_b`. Perfect calibration is the diagonal. A point above the diagonal means the band produced more positives than it promised (under-prediction); below means over-prediction.

```text
observed
 1.0 |                                  A(0.95,0.98)
     |                        B(0.75,0.78)
     |              C(0.52,0.54)
 0.5 |        D(0.32,0.324)
     |  E(0.16,0.16)
     | F(0.06,0.06)
 0.0 |G(0.02,0.02)
     +-------------------------------------------- predicted
      0.0      0.25      0.5      0.75      1.0

  points above the diagonal near the top: the model is too timid where the
  queue actually operates; the aggregate point sits almost exactly on it.
```

The visual encodes the section's central diagnostic. The aggregate residual is a weighted average of these vertical gaps, and because band G carries two thirds of the weight with a zero gap, the average is dragged to zero. A reader who looks only at the aggregate point sees a calibrated model. A reader who looks at the plot sees that the only bands that matter for decisions are all slightly above the diagonal. Aggregation hides exactly the region the action lives in.

### 6.4 Split geometry: random, group, and forward

The three splits differ in which axis they cut, and the picture explains the performance gap that motivates the whole file.

```text
random row split                group split by entity
 +------------------------+      +------------------------+
 | o x o | x o o x o x    |      | ENTITY 1..400          |
 | x o x | o x o o x o    |      |   all rows together    |
 | (every entity appears  |      +------------------------+
 |  on both sides)        |      | ENTITY 401..800        |
 +------------------------+      |   all rows together    |
  estimates: memorisation        +------------------------+
                                 estimates: unseen entity

forward split (time)
 |========= train =========|--gap--|== validation ==|--gap--|== test ==|
 2025-01                 2025-06  2025-07        2025-09  2025-10  2025-12
                           ^ labels must be mature before each boundary
```

The random panel is the one that inflates scores, and the mechanism is visible: the same entity's rows are adjacent in the picture and end up on both sides. The group panel keeps each entity whole, so the evaluation asks the question deployment asks. The forward panel adds a dimension the others lack: the gap exists because a label for a training decision at the end of June is not knowable until its horizon elapses, and training on rows whose labels are not final at the split date would leak the outcome of the very period being validated.

A useful rule of thumb from this geometry: if the random-split estimate exceeds the group-split estimate by more than the group-split uncertainty, entity identity is carrying signal, and the random number is not a performance estimate for anything.

### 6.5 Gain chart geometry and the shape of the queue

Plot, on the horizontal axis, the fraction of the cohort actioned from the top, and on the vertical axis, the fraction of all positives captured. A model that ranks at random produces the diagonal.

```text
positives
captured
 1.0 |            *****  model
     |        ****
     |     ***
 0.5 |   **          ______ random
     |  *     _____/
     | *  ___/
 0.0 +*_/______________________________
     0   0.126   0.5            1.0  fraction actioned

 the operating point sits at 2500/19800 = 0.1263 of the cohort,
 where the model has captured 1320/1980 = 0.6667 of the positives
```

At the operating point the vertical gap between the model curve and the diagonal is the concentration advantage, and it is only meaningful at that horizontal position. A vendor curve that is steep over the first 2% and flat afterwards says nothing about a queue that runs at 12.6%. Reading a gain chart at the wrong horizontal position is the same error as quoting lift without its depth.

### 6.6 The confusion table as an area argument

Draw the cohort as a rectangle of area 19,800. Divide it horizontally into positives (1980) and negatives (17,820). Divide it vertically into actioned (2,500) and ignored (17,300). The four cells are the intersections.

```text
+----------------------+---------------------------+
|  TP  1320            |  FP  1180                 |  actioned 2500
+----------------------+---------------------------+
|  FN   660            |  TN 16640                 |  ignored 17300
+----------------------+---------------------------+
   positives 1980         negatives 17820            total 19800
```

Precision is the share of the top row that lands in the left column. Recall is the share of the left column that lands in the top row. The false-positive rate is the share of the bottom-right region's column in the top row. Reading the denominators off the picture is faster and more reliable than remembering the formulas, and it makes the prevalence sensitivity immediate: shrinking the left column while holding the row total fixed must reduce precision, because the same top row now covers a smaller absolute number of positives.

### 6.7 The cost landscape

Plot total cost against the number of actions. The curve is U-shaped, and the shape is the whole argument for threshold selection being a business task.

```text
cost CU
 54000 |*
       |  *
 36000 |     *
       |        *
 21360 |              *  (capacity 2500)
 18660 |                  *  (unconstrained min, 4000)
       |                     *        *
       |                                *
       |                                     *
       +---------------------------------------- actions
        200   1400   2500   4000   6600   19800

 left of minimum: each added action has p > tau_cost, so it saves more
                  missed-loss than it spends on review
 right of minimum: each added action has p < tau_cost, so the reverse
 capacity truncates the curve before its minimum, which is why the
 achievable cost exceeds the optimum
```

The left branch is steep because the first actions are almost all positives. The right branch rises slowly at first and then accelerates as prevalence collapses toward the base rate. The minimum is shallow, which is fortunate: a threshold that is somewhat off costs little. That flatness is also a trap, because two very different policies can have nearly equal cost and very different recall, and choosing between them requires a value judgement that the cost column cannot supply.

**Takeaway:** Geometry makes the denominators visible, and the visible denominators are what prevent the wrong rate from being reported.

## 7. Comparison / Selection Table

### 7.1 Choosing the evaluation protocol from the deployment condition

The table below is the sheet's primary selection artifact. Read the deployment condition in the left column, follow the row, and treat the "do not use alone" column as a disqualifier rather than advice. Every row assumes the seven-field record from Section 1.6 is complete.

| Deployment condition | Primary protocol | Why it estimates the deployment | Do not use alone | Required companion check |
|---|---|---|---|---|
| Independent units, stationary population | Stratified random holdout | Preserves class support; independence assumption holds | Row-level CV when duplicates exist | Duplicate-key and near-duplicate audit |
| Repeat customers, accounts, devices | Group split on persistent entity | Tests unseen-entity generalisation | Random split; its score is memorisation | Overlap rate and random-versus-group gap |
| Identity linkage uncertain | Group split on the widest defensible key | Conservative bound on leakage | Narrow key chosen because it scores better | Household/device linkage rate, governance review |
| Deployment strictly later in time | Forward or rolling-origin split | Preserves information availability and drift | Shuffled future rows; k-fold on time series | Maturity watermark per fold |
| Site or region rollout | Leave-one-site-out | Tests transfer to an unseen site | Pooled score with site-mixed folds | Per-site support and confidence interval |
| Seasonal or promotion-driven demand | Blocked time folds by season | Preserves seasonality composition | Random month shuffling | Horizon coverage and event counts per fold |
| Fixed review headcount | Top-k at capacity | Matches the actual action set | AUC or any threshold-free metric | Queue volume replay per day |
| Probability prices the action | Calibration plus cost curve | Score magnitude drives money | Ranking metric alone | Reliability by band at the operating region |
| Small positive count, rare events | Repeated grouped split with variance reported | Shows estimate instability | Single split point estimate | Interval width and seed sensitivity |
| Prespecified high-harm slice | Slice-level evaluation with support | Exposes concentrated harm | Aggregate rate only | Guardrail threshold and escalation owner |
| Regulatory or audit context | Frozen, versioned, reproducible protocol | Enables independent reconstruction | Ad-hoc notebooks | Manifest completeness per Section 9.6 |
| Incumbent rule in production | Side-by-side replay at equal capacity | Isolates the model's contribution | Candidate-only report | Difference in cost and its interval |

### 7.2 Choosing the metric from the question being asked

Metric selection is downstream of protocol selection and must not be used to rescue a protocol. The table maps a question to the metric that answers it and names the metric's blind spot explicitly.

| Question | Metric | Denominator | Blind spot |
|---|---|---|---|
| How much review effort is wasted? | Precision | Actions | Says nothing about missed positives |
| How many positives are missed? | Recall, and `FN` as a count | Positives | Ignores the cost of false actions |
| How much of the negative mass is disturbed? | FPR | Negatives | Denominator is invisible to the reviewer |
| How safe is abstention? | False omission rate | Ignored records | Depends strongly on prevalence |
| Is the ranking usable at all? | ROC AUC | All pairs | Threshold-free; hides the operating region |
| Is the ranking usable at low prevalence? | Average precision | Recall steps | Prevalence-bound; not comparable across cohorts |
| Does the cut beat the base rate? | Lift at the operating depth | Prevalence | Undefined without `k` |
| Are probabilities honest? | Reliability residual by band | Records in band | Needs support; aggregate hides the tail |
| Are probabilities honest overall? | Brier or log loss | Records | Mixes calibration with discrimination |
| What does the policy cost? | Expected cost `VAL-07` | Cohort | Requires defensible costs |
| Will it survive a staffing change? | Cost as a function of `C` | Cohort | Needs demand distribution |
| Is the estimate stable? | Interval or bootstrap width | Effective `n` | A wide interval can still be biased |

### 7.3 Choosing a calibration remedy

Section 5.8 found band-local under-prediction at the top with an aggregate residual near zero. Five remedies are available, and the selection depends on the shape of the defect rather than on convention.

| Remedy | Fits when | Data requirement | Risk | When to avoid |
|---|---|---|---|---|
| Do nothing; document the bias | The bias is small relative to decision margins | None | Bias compounds if scores are summed across records | Probabilities drive pricing or aggregation |
| Single-parameter Platt scaling | Defect is a consistent slope error across the range | Separate calibration partition, ≥1000 records | Cannot fix band-local error | Top band is miscalibrated while middle is fine |
| Isotonic regression | Monotone but nonlinear distortion | Large calibration set, thousands per region | Overfits sparse bands; creates flat regions | Positive count per band is small |
| Beta calibration | Distortion concentrated near 0 or 1 | Moderate calibration set | Two parameters may still be too rigid | Defect is non-monotone |
| Band-local correction table | Defect is isolated, as in Section 5.8 | Enough support per band | Discontinuous at band edges; brittle to drift | Bands are thin or the model revision changes often |
| Rebuild with a better loss | Miscalibration originates in the objective | Retraining budget | Regresses ranking if applied carelessly | Time pressure and the defect is small |

The recommendation for the Section 5 case is the band-local table or Platt scaling fitted on a separate calibration partition, evaluated on a later window, with the top-band residual rechecked. Isotonic is excluded because the defect is confined to one band with 200 records, which is too thin to fit a flexible map without overfitting. Rebuilding is excluded because resolution is strong and the defect is small in magnitude; the Brier decomposition in Step 8 showed that the entire reliability term is 0.0000467.

### 7.4 Choosing between uncertainty methods

| Method | Assumption | Cost | Use when |
|---|---|---|---|
| Wald interval on the proportion | Independent records | Closed form | Baseline sanity check only |
| Cluster-robust / design-effect adjusted | Known ICC and cluster size | Cheap after ICC estimate | Repeated entities, as in Section 5.10 |
| Grouped bootstrap | Resampling at the entity level | Recomputation per replicate | ICC unknown; moderate data |
| Block bootstrap over time | Serial dependence within blocks | Block-length selection | Temporal cohorts with autocorrelated errors |
| Bayesian posterior on the rate | Prior specification defensible | Model fitting | Sparse slices where intervals must remain bounded |
| None, with an explicit warning | Small `n`, unstable | Zero | Never for a decision; acceptable only for exploration |

### 7.5 Choosing between a single number and a table

The last selection decision is presentational, and it changes behaviour more than most analysts expect. A single headline metric invites optimisation of that metric. A table with support, interval, and denominator invites questions. When the audience is a promotion committee, the table is the correct artifact even when it is harder to read, because the committee's job is to find the reason to say no.

**Takeaway:** Select the protocol from the dependence the deployment must survive, the metric from the question, and the remedy from the shape of the defect — never the reverse.

## 8. Failure Modes & Edge Cases

### 8.1 Failure index

Eight failure modes are documented here. Each has competing causes because in practice a symptom rarely names its own mechanism, and the discriminating check is the artifact of the section. The dossiers in 8.4 through 8.7 give four of them in full; the others are given in compressed form because their diagnostics follow the same pattern.

| ID | Symptom | Competing causes | Discriminating check | Bounded fix |
|---|---|---|---|---|
| VAL-F01 | Sudden large offline uplift | Genuine signal; target leakage; temporal leakage; join explosion | As-of recomputation and row-count reconciliation | Remove the field, rebuild lineage, refreeze protocol |
| VAL-F02 | Random-split score far above pilot | Entity leakage; population difference; pilot label process | Entity overlap rate; group-split replay | Group split as primary estimate |
| VAL-F03 | Recent-window precision falls | True drift; label immaturity; adjudication change | Maturity curve by cohort age | Publish watermark, evaluate mature cohorts only |
| VAL-F04 | Aggregate metric stable, complaints rise | Slice regression; mix shift; capacity change | Slice support, missingness, queue age by slice | Slice gate, source repair, escalation |
| VAL-F05 | Validation improves every iteration | Real progress; adaptation to validation noise | Untouched final window plus change log | Freeze selection, test once |
| VAL-F06 | High accuracy, no useful alerts | Prevalence; wrong objective; label mismatch | Confusion matrix and prevalence baseline | Reframe objective; rederive the threshold |
| VAL-F07 | Calibration fine overall, bad at the top | Sparse support; selection on review; model timidity | Reliability by band with `n` and SE | Band-local or Platt remedy per 7.3 |
| VAL-F08 | Threshold "optimal" but queue overflows | Cost cut exceeds capacity; demand shifted | Daily count above the cut versus `C` | Apply `VAL-06`; report unserved value |

### 8.2 The universal triage order

When a validation number is questioned, the checks below are cheaper earlier in the list and more expensive later. Do not skip ahead: retraining before finishing step 2 is the most common waste in the file.

1. **Arithmetic and total reconciliation.** Do the four identity checks from Section 5.4 pass?
2. **Population and filter replay.** Is the scored population the deployed population, with the same eligibility filter?
3. **Maturity.** Are all included labels final at the reporting date?
4. **Availability lineage.** Does every feature have a materialisation time before `t0`?
5. **Entity disjointness.** Is the overlap rate between partitions zero for the stated key?
6. **Preprocessing scope.** Were transform parameters fitted inside the fold?
7. **Operating point.** Was the reported metric computed at the capacity cut?
8. **Deduplication and overrides.** Does the replay include the live queue's dedupe and human override?
9. **Baseline parity.** Was the incumbent replayed under identical conditions?
10. **Model.** Only now consider the model, and only with a pre-registered change.

### 8.3 Edge case: the degenerate cohort

A cohort can pass every check and still be unusable because one class is absent. If a window contains zero positives, precision is undefined, recall is undefined, and any "100% precision" claim is an artefact of an empty denominator. The same happens with zero actions, where precision is undefined rather than zero. Implementations that return `0` or `1` for these cases silently convert an undefined quantity into a confident one, and that value then propagates into a dashboard where nobody can tell it apart from a computed rate.

The bounded fix is to return an explicit null with the reason and to render it as "no support" rather than as a number. Verification consists of constructing a zero-positive fixture and a zero-action fixture and asserting that both produce the null path rather than a numeric value.

### 8.4 Dossier VAL-D01 — a future aggregate entered a training row

**Trigger and environment.** A daily transaction-count aggregate is joined to the decision table on calendar date. The join key is correct; the date semantics are not. A decision at 14:20 on 12 March receives the aggregate for 12 March, which was computed from the full day's events.

**Symptom.** Offline precision rises from 0.41 to 0.72 within one retrain, with no other change. Recall rises less. The uplift appears in the same revision that introduced eleven aggregate features, so the effect cannot be attributed to a single column by inspection.

**Competing causes.** The aggregate may carry legitimate signal about recent merchant activity that is genuinely available via a real-time counter. It may include events after `t0` and therefore be temporal leakage. It may have been recomputed after chargebacks landed, making it target leakage as well. It may duplicate rows through a many-to-many join on merchant-day, inflating both counts and prevalence.

**Discriminating diagnostic.** Three checks separate these causes, in this order. First, reconcile row counts: if the join increased records, the uplift is partly a join explosion and no further analysis of the features is meaningful until that is fixed. In the constructed incidence, the pre-join table has 19,800 rows and the post-join table has 21,384, a 1,584-row increase, so join duplication is present and contaminating. Second, reconstruct the aggregate using only events with `event_time < t0` and re-score the frozen cohort; if precision returns to approximately 0.41, the remaining uplift was temporal. Third, inspect the producing job's history for restatements: if the aggregate was overwritten after the outcome window, a target component is present even when the as-of version is used.

**Root-cause evidence.** A row-level lineage table showing at least one contributing event timestamped after the decision instant establishes temporal leakage. That is necessary but not sufficient for the full 0.31 gap, because the join duplication also contributes. Reporting the gap as entirely temporal would be an overclaim, and the honest statement is that two defects were present and their individual contributions cannot be separated from a single before-and-after comparison.

**Immediate mitigation.** Disable all eleven aggregates in the serving candidate and block promotion of the revision. Keep the previous revision live. Do not attempt a partial disable based on feature importance, because correlated aggregates will substitute for each other and mask the removal.

**Correction.** Implement as-of aggregation with an explicit event-time watermark and a snapshot version on every row. Add a uniqueness constraint or an explicit aggregation step that collapses the merchant-day grain before the join. Rebuild the evaluation cohort from the corrected snapshot.

**Prevention.** Require `event_time`, `materialisation_time`, and `snapshot_id` as mandatory fields in feature registration, and reject a feature whose declared availability is later than the earliest decision that consumes it. Add a CI check that scores a fixture where an event arrives one second after `t0` and asserts that the feature value is unchanged.

**Verification.** Re-run the frozen protocol and confirm that precision returns to the pre-change level within the interval computed in Section 5.10. Confirm the row count is exactly 19,800. Confirm the watermark check fails the deliberately-late fixture.

**Mitigation risk.** Removing the aggregates may cost genuine signal, and the corrected precision may be lower than a legitimate real-time counter would achieve. That gap is an argument for building the real-time counter, not for restoring the leaky join.

**Escalation.** If the revision was already used for actions, route the affected decisions to the model owner and the risk owner for an impact assessment of decisions taken on contaminated scores.

### 8.5 Dossier VAL-D02 — entity overlap masks poor generalisation

**Trigger and environment.** Multiple transactions per merchant are split randomly into train and test. Merchants average four records, and some have hundreds.

**Symptom.** Random-holdout recall is 0.667 in the constructed replay, while a pilot on newly onboarded merchants shows 0.41. The gap is large enough to be operationally visible in complaint volume.

**Competing causes.** New merchants may genuinely behave differently, so the pilot gap may reflect a real distribution shift rather than leakage. The persistent merchant identifier may be incomplete, in which case even a group split leaves residual linkage. The model may memorise merchant-specific history through identifier-derived features. The pilot's label process may differ because new-merchant adjudication is faster.

**Discriminating diagnostic.** Compute the entity overlap rate: the share of test records whose merchant appears in training. In a random split with four records per merchant, expected overlap is high by construction; the measured value in the constructed case is 0.94. Replay with a group split and compare. Then audit residual linkage through parent-company, address, and device identifiers where governance permits, and report the unresolved linkage rate as a bound rather than assuming zero.

**Root-cause evidence.** A large random-to-group gap combined with high overlap supports the memorisation hypothesis. It does not establish that every feature is invalid, and it does not establish that the pilot population is representative. Both facts must be reported.

**Immediate mitigation.** Adopt the group-split estimate as the primary claim and restate the pilot expectation from 0.667 to the group value.

**Correction.** Remove or replace identifier-derived features whose value is merchant-specific and non-transferable, and re-evaluate. Where the merchant is legitimately part of the decision, state that the deployment population includes returning merchants and report both splits as answering different questions.

**Prevention.** Make the split key a required field of the prediction contract, and reject a validation report that omits the overlap rate.

**Verification.** Confirm zero overlap for the stated key and report the unresolved-linkage bound. Re-run the pilot on a later cohort and compare its interval against the group-split estimate.

**Mitigation risk.** A conservative group shrinks the effective training sample and widens intervals. A drop in measured score after the switch is expected and is not evidence of a regression.

**Escalation.** If identity linkage itself is uncertain, involve data governance before widening the key, because expanding linkage can cross consent or purpose boundaries.

### 8.6 Dossier VAL-D03 — label delay turns pending records into negatives

**Trigger and environment.** The outcome is a chargeback that can arrive up to 45 days after the transaction. Reports are generated nightly, and the reporting query treats any record without a chargeback as negative.

**Symptom.** Precision in the most recent 30-day window is 0.31 while the 90-day-old window shows 0.53. The team concludes the model has drifted and schedules a retrain.

**Competing causes.** Real drift may be present. Recent labels may be incomplete, which mechanically lowers the observed positive count and therefore lowers precision at a fixed action count. Reviewers may have changed adjudication criteria. The action policy may have changed, altering which records appear in the queue.

**Discriminating diagnostic.** Plot the observed positive rate by cohort age in days. Under maturing labels, the rate rises with age and flattens after the maximum adjudication delay. In the constructed case it rises from 0.041 at age 5 to 0.101 at age 50 and is flat thereafter, which is the signature of censoring rather than drift. Compare adjudication policy versions across the two windows, and confirm the action policy version is identical.

**Root-cause evidence.** A rate that stabilises with cohort age establishes censoring as at least a major contributor. It does not exclude drift, because both can be present, and the two are separable only by evaluating mature cohorts across both windows.

**Immediate mitigation.** Mark recent-window metrics as provisional, and stop threshold changes that are motivated by them.

**Correction.** Add a maturity watermark: a record enters the evaluation population only when `now - t0 > 45 days` and its adjudication is final. Publish the pending volume beside every rate. Recompute the historical series using only records that were mature at each historical report date, so the series is comparable across time.

**Prevention.** Make maturity a required field of the denominator, and add a unit test that a 10-day-old record with no event is excluded rather than counted as negative.

**Verification.** Reproduce a historical report using the watermark and confirm it matches the recomputed mature value. Confirm the pending count displayed equals the count excluded.

**Mitigation risk.** Delayed feedback slows genuine drift detection. Sampled expedited adjudication can shorten the lag, but only if its selection bias is documented, because expedited review is rarely random.

**Escalation.** If the reporting contract is externally mandated, the policy owner must approve the change from a naive to a mature denominator.

### 8.7 Dossier VAL-D04 — a global metric conceals a slice regression

**Trigger and environment.** A release changes the missing-value imputation for a sensor-derived feature. Aggregate precision is unchanged at 0.528. One region reports a rise in queue age and in complaints about unreviewed cases.

**Symptom.** Slice precision in the affected region falls from 0.49 to 0.36 with unchanged support. The global number does not move because the region is 6% of volume.

**Competing causes.** Population mix may have shifted within the region. The feature feed may be delayed or sparse for that region's devices. The region may have different adjudication criteria. Capacity may have been reduced locally, changing the effective cut. The imputation change may interact with a sensor that behaves differently there.

**Discriminating diagnostic.** Compare, for the same model and policy version and the same window: slice support, feature missingness rate, score distribution, threshold in force, queue age, and label maturity. In the constructed case missingness for the sensor feature rises from 0.04 to 0.61 in that region only, immediately before the score shift, which points at the data path rather than at the model.

**Root-cause evidence.** A localised missingness increase preceding the score shift supports a data-path cause. A simultaneous adjudication-rule change would produce a shift without a missingness change, and the check separates them.

**Immediate mitigation.** Gate the new imputation for the affected region, or fall back to the previous revision there, if a safe fallback exists and the fallback's own behaviour is documented.

**Correction.** Repair source freshness for the region, then re-run the point-in-time evaluation rather than only the live dashboard.

**Prevention.** Monitor source health and outcome metrics by operationally meaningful slice, and make slice guardrails blocking rather than advisory.

**Verification.** Re-evaluate the repaired slice on a later mature window without repeatedly tuning on it, and confirm missingness has returned to its baseline.

**Mitigation risk.** Slice-specific rules add maintenance burden and can create disparate treatment that must itself be reviewed. A guardrail that blocks on noise will halt releases for no reason, which is why Section 9.4 quantifies the multiple-comparison problem.

**Escalation.** Route consequential subgroup effects to the accountable policy owner, not to the modelling team alone.

### 8.8 Edge case: capacity and demand both move

A threshold fixed in score units silently changes its action volume when demand moves. Suppose the cut was set to fill 2,500 reviews at a demand of 19,800 records per day. If demand rises 40% to 27,720 with a stable score distribution, the same cut produces 3,500 actions and the queue overflows by 1,000. The model did not change, the threshold did not change, and the operation broke.

The discriminating check is to plot daily actions above the cut against capacity, not to inspect the model. The bounded fix is to convert the absolute threshold into a capacity-relative rule (top-k per day) or to keep the absolute threshold and accept a variable queue with an explicit overflow protocol. Both are defensible; leaving the choice implicit is not. The verification is a replay of a high-demand day against the rule, asserting that the action count either stays within capacity or triggers the overflow path.

### 8.9 Edge case: overrides that vanish in the replay

Human reviewers override the model. If 12% of queued records are dismissed by a reviewer before investigation, the effective action set differs from the scored set, and precision computed on the scored set overstates the yield of the reviewed set. Replaying without overrides evaluates a policy that does not exist.

The discriminating check is to join the override log to the scored queue and recompute both variants. The bounded fix is to model the override as part of the policy and to report precision on reviewed records separately from precision on queued records. The verification is that the two numbers differ by the expected override effect and that both appear in the report.

**Takeaway:** Every failure in this section is a mismatch between the evaluation's stated conditions and its actual conditions; the discriminating check is always the artifact that makes the mismatch visible.

## 9. Complexity & Cost Analysis

### 9.1 Algorithmic cost of the evaluation protocol

The protocol has five cost centres, and only one of them is model fitting. Sizing the others first is the practical lesson, because they are usually the ones that make a credible evaluation too slow to run, which is the real reason teams fall back on a single random split.

Let `n` be decisions, `d` features, `k` folds, `T` training time for the full data, `S` scoring time for the full data, `E` point-in-time extraction cost per feature per row, and `B` bootstrap replicates.

| Stage | Cost | Dominant when |
|---|---|---|
| Extraction | `n * d * E` | Point-in-time joins; `d` large and joins many-to-many |
| Transform fitting | `O(k * n * d)` for scaling/encoding | Re-fitted per fold; cheap per unit but multiplied by `k` |
| Training | `sum over folds of T((k-1)n/k)` ≈ `(k-1) * T` | Model capacity dominates |
| Scoring and OOF assembly | `O(k * n)` predictions | Storage-bound at large `n` |
| Metric and interval | `O(n log n)` for sorting plus `B * n_cluster` | Bootstrap with many replicates |
| Human review of the report | Fixed | Slice count and evidence assembly |

With grouped or forward splits, `T((k-1)n/k)` is not the right argument because folds are deliberately unequal. A forward split's first fold trains on less history, which lowers training cost and raises variance simultaneously; the cost model must use per-fold row counts rather than the equal-fold approximation, or it will systematically understate the early-fold variance and overstate the compute saving.

### 9.2 Worked sizing: storage and throughput

**Evidence: ILLUSTRATIVE sizing with DERIVED arithmetic.** Assume `n = 2,000,000` scored decisions, `d = 350` features stored at 4 bytes each, and a requirement to retain twelve monthly as-of snapshots for replay.

```text
FORMULA [VAL-15]: Bytes = n * d * b
WHERE: n [records]; d [features/record]; b [bytes/value]; result [bytes].
CONSTRAINT: dense storage with no compression, no null bitmap, no index;
            sparse encodings and columnar compression change the result.
NUMERICAL EXAMPLE: n=2000000, d=350, b=4;
                   Bytes = 2000000*350*4 = 2800000000 bytes;
                   in GiB: 2800000000 / 1073741824 = 2.60773 GiB.
INTERPRETATION: one snapshot is small; the snapshot count is what bites.
```

```text
FORMULA [VAL-16]: Bytes_total = Bytes * M * (1 + r)
WHERE: M [snapshots]; r [fractional replica/index overhead].
CONSTRAINT: uniform snapshot size; excludes compute scratch and backups.
NUMERICAL EXAMPLE: M=12, r=0.35;
                   Bytes = 2800000000;
                   Bytes_total = 2800000000*12*1.35 = 45360000000 bytes
                               = 42.245 GiB.
INTERPRETATION: 3.36 GiB payload per snapshot-month at 350 features,
                before replication; twelve snapshots multiply the obligation.
```

```text
FORMULA [VAL-17]: t_score = n / q
WHERE: q [records/s] scoring throughput; t_score [s].
CONSTRAINT: one model revision, no network hops per record; batching as measured.
NUMERICAL EXAMPLE: n=2000000, q=4000; t_score = 2000000/4000 = 500 s = 8.33 min.
INTERPRETATION: a full rescore is cheap; a hundred experiments per day is not.
```

```text
FORMULA [VAL-18]: t_extract = (n * d) / (v * w)
WHERE: v [feature-values/s per worker]; w [workers]; result [s].
CONSTRAINT: point-in-time join per value; no cross-decision caching; linear
            scaling with workers, which fails when the source warehouse saturates.
NUMERICAL EXAMPLE: n=19800 decisions, d=350, v=1200, w=8;
                   values = 19800*350 = 6930000;
                   t_extract = 6930000/(1200*8) = 721.875 s = 12.03 min.
INTERPRETATION: rebuilding the evaluation cohort once is cheap; doing it on
                every experiment iteration makes iteration latency extraction-bound.
```

```text
FORMULA [VAL-19]: t_protocol = k*(T + S) + t_extract + t_metrics
WHERE: T [s per full training]; S [s per full scoring]; k [folds].
CONSTRAINT: sequential folds; training dominates; ignores queueing and failures.
NUMERICAL EXAMPLE: k=6, T=3600, S=500, t_extract=721.875, t_metrics=120;
                   t_protocol = 6*(3600+500) + 721.875 + 120
                              = 24600 + 841.875 = 25441.875 s = 7.07 h.
INTERPRETATION: one credible six-fold forward evaluation is a working day of
                compute; this is the price of not fooling yourself.
```

The last number is the one to bring to a planning conversation. A team that can afford a single random split and nothing else should say so explicitly and label its estimate as provisional, rather than implying that the random split is a validation. The protocol cost is roughly seven hours per candidate; three candidates plus one final holdout evaluation is a week of compute, which is a staffing input rather than a modelling choice.

### 9.3 Worked sizing: operational and review cost

The evaluation is only half the cost. Running the policy has its own budget, and it dwarfs the compute.

```text
FORMULA [VAL-20]: Cost_op = C * r * D
WHERE: C [records/day capacity]; r [CU per review]; D [operating days];
       Cost_op [CU].
CONSTRAINT: constant review cost; excludes infrastructure, appeals, and
            remediation of false actions.
NUMERICAL EXAMPLE: C=2500, r=3, D=365;
                   Cost_op = 2500*3*365 = 2737500 CU/year.
INTERPRETATION: review labour is ~2.74M CU/year, against which a 2,700 CU/day
                capacity penalty (985,500 CU/year) is a material line item.
```

```text
FORMULA [VAL-21]: N_inflight = C * L
WHERE: C [decisions/day]; L [label lag in days]; N_inflight [decisions].
CONSTRAINT: steady-state arrival rate; excludes reprocessing and re-scores.
NUMERICAL EXAMPLE: C=2500, L=45; N_inflight = 112500 decisions.
INTERPRETATION: 112,500 decisions carry pending labels at any moment, each
                needing a stored score and feature snapshot for later replay.
```

Storage for those in-flight snapshots at 350 float32 values per decision plus a 64-byte record header:

```text
per decision = 350*4 + 64 = 1400 + 64 = 1464 bytes
total        = 112500 * 1464 = 164700000 bytes = 157.06 MiB
```

Small in absolute terms and easy to forget entirely, which is why replays often cannot reconstruct the exact feature state that produced a decision. The 1464-byte figure excludes the immutable event log that the snapshot was derived from, and that log is typically two orders of magnitude larger.

### 9.4 Worked sizing: the cost of uncertainty and the cost of slices

**Cost of a wide interval.** Section 5.10 showed that clustering inflates the variance by a design effect of 1.8001. To reach the same standard error as an independent sample of 2,500, a clustered sample needs

```text
FORMULA [VAL-22]: n_cluster = n_independent * (1 + (m - 1)*ICC)
WHERE: m [records per cluster]; ICC [intraclass correlation, dimensionless].
CONSTRAINT: equal cluster sizes and constant ICC; with variable sizes, use the
            mean cluster size and accept the approximation error.
NUMERICAL EXAMPLE: n=2500, m=4, ICC=0.2667;
                   n_cluster = 2500*1.8001 = 4500.25 -> 4501 records.
INTERPRETATION: measuring the same precision costs 2001 extra records, or 80%
                more review labour in the evaluated window.
```

**Cost of slice testing.** With `K` prespecified slices and no true differences, the probability that at least one shows a two-sided `|z| > 2` flag is bounded by `K * 0.0455`. For `K = 20` that bound is `0.91`, so a spurious flag is nearly certain. Controlling family-wise error at 5% requires a per-test level of `0.05/20 = 0.0025`, corresponding to `|z| > 3.023`. Applying the 3.023 cut rather than 2.0 to the same data raises the support each slice needs by roughly `(3.023/2.0)^2 = 2.285` times for the same power. That ratio is the real price of slicing: either accept more false alarms or collect more data per slice.

```text
FORMULA [VAL-23]: alpha_per = alpha_family / K
WHERE: K [slices]; alpha_family [target family-wise rate]; alpha_per [per-test].
CONSTRAINT: Bonferroni control, conservative under positive dependence between
            slices; slices overlap in practice, so this over-corrects somewhat.
NUMERICAL EXAMPLE: alpha_family=0.05, K=20; alpha_per = 0.0025, z_crit = 3.023.
INTERPRETATION: the honest slice threshold is 50% larger than the naive one,
                which is why most single-slice alarms do not replicate.
```

### 9.5 Cost of the mistakes this sheet prevents

The failure modes in Section 8 have quantifiable costs, and quoting them changes how much validation effort is rational.

| Failure | Cost mechanism | Constructed magnitude |
|---|---|---|
| `VAL-F01` leakage | Decisions taken on a contaminated score, then reversed when the feature is withdrawn | The 0.31 precision overstatement on 2,500 daily actions implies ~775 incorrect actions per day until detected |
| `VAL-F03` immature labels | Threshold retuned against censored evidence | A 0.22 apparent precision gap drives an unnecessary retrain and a threshold shift |
| `VAL-F05` metric drift over iterations | Selection on validation noise, inflated expectation | Each unlogged holdout peek consumes one unit of a scarce resource; the reported gain does not replicate |
| `VAL-F08` capacity overflow | Queue pressure forces ad-hoc triage | 1,000 unscheduled reviews per day at 2.5 minutes triage each is 41.7 analyst-hours, unpaid |
| `VAL-F04` slice regression | Local harm invisible in aggregate | A 6%-of-volume slice at precision 0.36 versus 0.49 leaves 19.5 extra wrong reviews per day in that slice |

The last row is worth deriving because the number looks small until it is annualised. If the slice holds `0.06 * 2500 = 150` actions per day and precision falls from 0.49 to 0.36, the extra wrong actions per day are `150 * (0.49 - 0.36) = 19.5`, which is 7,117.5 per year. The aggregate metric moves by `19.5/2500 = 0.0078`, a shift smaller than its own standard error, so the regression is statistically invisible at the global level and operationally real at the local one. This is the structural reason slice monitoring cannot be replaced by tighter global intervals.

### 9.6 Reproducibility manifest and its cost

A manifest that permits independent reconstruction must capture the code revision, environment lockfile, data snapshot with as-of time, split specification, feature specification with availability rules, label specification with maturity watermark, model parameters, policy specification, metric definitions, and artifact locations. Writing it takes roughly twenty minutes for a run whose compute cost is seven hours, and its absence means the run cannot be defended later at any price.

| Manifest field | Cheap to capture? | Consequence if missing |
|---|---|---|
| Code revision | Yes, automatic | Results cannot be reproduced after a branch moves |
| Environment lockfile | Yes, automatic | Silent library-default changes break comparability |
| Data snapshot identifier | Sometimes | As-of reconstruction becomes impossible after a backfill |
| Split specification | Yes, if generated by code | The dependence boundary is undocumented and unauditable |
| Feature availability rules | Expensive | Leakage cannot be ruled out retrospectively |
| Label watermark | Yes | Denominators are uninterpretable |
| Policy and override logs | Requires join | Replay evaluates a policy that never ran |
| Artifact paths | Yes | The report cannot be checked |

**Takeaway:** Compute is the cheap part of credible validation; extraction lineage, snapshot retention, and review labour are the budget lines that decide whether the protocol is affordable at all.

## 10. Key Theorems / Results Reference

### 10.1 Results table

| ID | Result | Statement | Conditions | Use in this sheet |
|---|---|---|---|---|
| `TH-01` | Cost-weighted Bayes rule | The expected-cost-minimising action is to act when `p > c_FP/(c_FP+c_FN)` | Calibrated `p`; constant marginal costs; independent decisions | Derives `VAL-05` |
| `TH-02` | Strict propriety of Brier and log loss | Both are strictly proper: expected score is minimised by reporting the true probability | Forecast is a probability; outcome distribution as framed | Justifies using them as calibration evidence |
| `TH-03` | Murphy decomposition | Brier splits exactly into reliability − resolution + uncertainty for bin-constant forecasts | Fixed bins with adequate support | Derives `VAL-14` and the Step 8 analysis |
| `TH-04` | AUC as a pairwise probability | `AUC = P(s(X+) > s(X-)) + 0.5 P(tie)` | Independent positive and negative draws from the same population | Explains ROC's threshold freedom |
| `TH-05` | Precision–recall prevalence identity | `precision = recall*pi / (recall*pi + FPR*(1-pi))` | Same threshold generating both curves | Explains why AP is not portable across populations |
| `TH-06` | Design effect for clustered proportions | `Var(p_hat) ≈ p(1-p)/n * (1 + (m-1)*ICC)` | Equal cluster sizes; exchangeable within-cluster correlation | Derives `VAL-22` |
| `TH-07` | Holdout unbiasedness for a fixed procedure | The holdout risk estimate is unbiased for that fixed procedure on that population | Procedure fixed before inspection; exchangeable with deployment | Explains why reuse destroys the guarantee |
| `TH-08` | F1 bounds | `min(P,R) <= F1 <= (P+R)/2`, with equality only when `P = R` | `P, R > 0` | Sets the interpretation limit of `VAL-04` |
| `TH-09` | Bootstrap consistency (sketch) | The bootstrap distribution of a smooth statistic converges to its sampling distribution | IID or block/group resampling matching dependence; sufficient moments | Justifies grouped resampling in 7.4 |
| `TH-10` | Multiplicity bound | `P(any of K independent flags at level a) <= K*a` | Union bound; conservative under dependence | Derives `VAL-23` |
| `TH-11` | Non-transitivity of aggregated metrics | No single scalar orders policies consistently for all cost ratios | Any two operating points with crossing cost curves | Justifies presenting the cost sweep, not one number |
| `TH-12` | Censoring bias direction | Counting immature positives as negatives biases the observed positive rate downward | Pending records have a positive probability of becoming positive | Derives the `VAL-F03` diagnostic |

### 10.2 `TH-01` — the cost-weighted Bayes rule, stated properly

For a record with true posterior probability `p` of being positive, action `a in {0,1}`, and costs `c_FP, c_FN > 0`, the expected cost of each action is `a=1: (1-p)c_FP` and `a=0: p c_FN`. The difference is `(1-p)c_FP - p c_FN`, which is negative exactly when `p > c_FP/(c_FP+c_FN)`. Because expectation is linear over records and the decision for one record does not constrain another, minimising per-record expected cost minimises the total.

Three conditions are load-bearing and each fails in practice. **Calibration**: if `p` is only a ranking score, the inequality selects an arbitrary quantile rather than an optimal set. **Constant marginal cost**: if `c_FN` depends on the record's value, the single threshold is wrong and the per-record rule from Section 4.4 applies. **Independence of decisions**: if acting on one record changes the cost of acting on another, for example when a shared merchant is closed after the first confirmed fraud, the additive decomposition fails and the optimum is a set-selection problem rather than a per-record cut.

### 10.3 `TH-02` and `TH-03` — propriety and the decomposition

A scoring rule is proper if the expected score is minimised (for losses) by reporting the true probability, and strictly proper if the minimiser is unique. For Brier, the expected loss at reported `q` when truth is `p` is `p(1-q)^2 + (1-p)q^2`; differentiating with respect to `q` gives `-2p(1-q) + 2(1-p)q = 2(q - p)`, which is zero only at `q = p` and increasing in `q`. Hence strict propriety, and the same argument applies to log loss with the negative log likelihood. The consequence for practice is that a model trained with a proper loss is being pushed toward honest probabilities, while a model trained with an improper surrogate, such as accuracy on a thresholded label, is not.

The decomposition `TH-03` is exact for bin-constant forecasts. Write the Brier score as `E[(p_b - Y)^2]` within bins. Expanding `(p_b - Y)^2 = (p_b - o_b)^2 + (o_b - Y)^2 + 2(p_b - o_b)(o_b - Y)` and averaging within a bin, the cross term vanishes because the mean of `(o_b - Y)` is zero. The middle term averages to `o_b(1-o_b)`, which decomposes as `(o_b - pi)^2 + pi(1-pi) - ...` when weighted across bins, producing the reliability, resolution, and uncertainty terms. Because the algebra is an identity, any violation in a computed decomposition indicates a binning or weighting error, which is exactly how it was used as a check in Step 8.

### 10.4 `TH-04` and `TH-05` — ranking interpretation and prevalence dependence

`TH-04` states that AUC equals the probability that a randomly drawn positive scores above a randomly drawn negative, with ties counting one half. The derivation: for a threshold sweep, the true-positive rate is the fraction of positives above the cut and the false-positive rate is the fraction of negatives above it, so the area under the curve is the average of `TPR` with respect to `FPR`, which by a standard identity equals the pairwise concordance probability. The practical consequence is that AUC is invariant to any strictly increasing transform of the score, which is why a monotone recalibration leaves it unchanged and why it can never detect a calibration defect.

`TH-05` is algebraic and it is the result that most often changes a decision. Divide the numerator and denominator of precision by the positive mass: `TP/(TP+FP) = (TP/P) / (TP/P + FP/P)`, and `FP/P = (FP/N)(N/P) = FPR * (1-pi)/pi`. Substituting gives the identity in the table. Since `pi` appears explicitly, precision at a fixed operating point must change when prevalence changes, and average precision inherits that dependence. ROC AUC does not, because both of its axes are conditioned on the true class. This is the precise sense in which one metric is portable across populations and the other is not, and it is a property of the estimators rather than a matter of preference.

### 10.5 `TH-06` and `TH-07` — dependence and the price of reuse

`TH-06` follows from the variance of a mean of clustered observations. With `m` observations per cluster, within-cluster correlation `rho`, and total `n`, the variance of the sample mean is `p(1-p)/n * (1 + (m-1)rho)`. The bracket is the design effect. It equals one when `rho = 0` and `m` when `rho = 1`. Section 5.10 evaluated it at 1.8001, and the practical instruction is that any interval on a rate computed over repeated entities without a design-effect adjustment is too narrow by a factor of the square root of that bracket, which was 34%.

`TH-07` is the statement that makes validation worth doing but also fragile. If a procedure is fixed before the holdout is touched, the holdout's expected error equals the procedure's expected error on any exchangeable population. The proof is one line: the holdout is a random sample from the same distribution, so the empirical risk is an unbiased estimator of the risk, and the procedure is a deterministic function that does not depend on the holdout. Two conditions can fail. **Exchangeability** fails when the deployment population or period differs, which is the entire subject of forward splits. **Fixed procedure** fails when the analyst uses the holdout to choose among candidates, because then the selected procedure is a function of the holdout and the estimator is biased optimistically. The second failure is invisible in the arithmetic and is the reason for the untouched final window.

### 10.6 `TH-09` through `TH-12` — what the remaining results buy

The bootstrap result justifies resampling at the level of dependence rather than at the row level. It is a sketch here, not a proof: consistency requires the resampling scheme to reproduce the dependence structure, which is why a grouped bootstrap resamples entities and a block bootstrap resamples contiguous time blocks. Resampling rows from a clustered dataset reproduces the wrong variance and will understate the interval by the design effect, silently repeating the error it was meant to fix.

`TH-10` is the union bound and it is included because slice alarms are the most common place where a team mistakes arithmetic for evidence. The bound is conservative under dependence, so the correct calibration of the correction is empirical: if slices overlap heavily, Bonferroni over-corrects and a permutation-based family-wise method is tighter. The honest practice is to state which correction was applied and why.

`TH-11` states that no single scalar orders policies consistently across all cost ratios. The proof is by construction: two policies whose cost curves cross have different rankings on either side of the crossing, and any scalar that is a monotone function of neither cost nor a fixed weighted combination will order them the same way everywhere. Section 5.6 exhibits a crossing: at `c_FN/c_FP = 9` the capacity policy is optimal, and at `c_FN/c_FP = 20` the wider policy wins. Reporting one threshold as "the" optimum is therefore a claim about the ratio, not about the model.

`TH-12` fixes the direction of the censoring bias, which matters because direction is often enough to prevent a wrong action. If pending records may become positive but are counted as negative, the observed positive rate is below the true rate, so recall is understated, precision is understated at a fixed action count when the pending positives are inside the action set and overstated when they are outside. Because the sign of the precision effect depends on where the pending mass sits relative to the threshold, the only safe rule is to exclude immature records rather than to reason about the sign.

**Takeaway:** Each theorem in this table is a conditional sentence; the value of citing it is to state the condition that the current evaluation might violate.

## 11. Interview / Exam Questions

Fourteen questions follow: five recall, six application and calculation, three judgement. Each answer is checked against the arithmetic in Sections 4, 5, and 9, states the reasoning, and names the red-flag answer that would fail a review.

### 11.1 Recall

**Q1. State the four confusion counts and the denominators that make precision, recall, and false-positive rate different quantities.**

Precision is `TP/(TP+FP)` with the action count as denominator. Recall is `TP/(TP+FN)` with the positive mass as denominator. The false-positive rate is `FP/(FP+TN)` with the negative mass as denominator. In the Section 5 cohort these are 0.528, 0.666667, and 0.066218 respectively. Reasoning: each is a conditional share within a different conditioning set, so none of them can be derived from the others without the full table.

*Red flag:* saying precision is "the fraction correct" or that a low FPR implies low queue waste. With `FPR = 0.0662` and queue waste `0.472`, those two statements conflict by a factor of seven.

**Q2. What distinguishes a negative from a censored record?**

A negative had the full horizon elapse, completed adjudication under the same process, and an open observation channel, and did not become positive. A censored record had the channel close before the outcome could be observed, for example an account that departed. *Red flag:* treating "no event recorded" as negative without checking horizon and channel, which is the mechanism behind `VAL-F03`.

**Q3. Why is ROC AUC invariant to a monotone transformation but Brier score is not?**

AUC is the pairwise concordance probability `P(s(X+) > s(X-))`. A strictly increasing transform preserves every pairwise ordering, so every pair contributes the same term and the value is unchanged. Brier averages `(p - Y)^2`, which depends on the numerical magnitude of `p`, so squashing scores compresses or expands every error. *Red flag:* claiming AUC measures "how correct the probabilities are". It cannot distinguish a well-ranked but badly shifted model from a well-calibrated one.

**Q4. What are the three terms of the Murphy decomposition and which should an improvement project target first?**

Reliability (lower better), resolution (higher better), and uncertainty (fixed by prevalence). In the Section 5 case these were 0.0000467, 0.0338718, and 0.09. Reliability is already near zero, so recalibration can improve Brier by at most 0.0000467 while resolution contributes 0.0339 of useful signal; improvement effort belongs with resolution. *Red flag:* proposing recalibration whenever Brier "looks high", which targets the term that is already minimal.

**Q5. Name the four clocks a feature carries and say which one gates admissibility.**

Event time, ingest time, materialisation time, and correction time. Admissibility is gated by materialisation time: a version must have been queryable before `t0`. Correction time matters because using a later restatement in a backtest is backfill leakage. *Red flag:* answering only "event time", which permits a feature whose value was computed at `t0 + 7 days` from events at `t0 - 1 day`.

### 11.2 Application and calculation

**Q6. A cohort has 19,800 records, 1,980 positives, capacity 2,500, and the top 2,500 contain 1,320 positives. Compute precision, recall, F1, and the cost at `c_FP = 3`, `c_FN = 27`.**

`FP = 2500 - 1320 = 1180`; `FN = 1980 - 1320 = 660`. Precision `1320/2500 = 0.528`. Recall `1320/1980 = 0.666667`. `F1 = 2*0.528*0.666667/(0.528+0.666667) = 0.704/1.194667 = 0.589286`. Cost `= 27*660 + 3*1180 = 17820 + 3540 = 21360` CU. Checking: `TP + FP = 2500` and `TP + FN = 1980`, both consistent with the stated cohort. *Red flag:* computing cost as `27*660 + 3*1180` but then reporting the result as a benefit; cost and recovered value are different quantities, and only their difference against a baseline is a decision input.

**Q7. With the same costs, what is the cost-optimal threshold and does capacity bind?**

`tau_cost = 3/(3+27) = 0.1`. Applying it to the band table admits bands A through E, which is 4,000 records, so the unconstrained optimum needs 4,000 reviews against a capacity of 2,500 and capacity binds. The effective cut is `max(0.1, s_(2500)) = max(0.1, 0.32) = 0.32`. The capacity penalty is `21360 - 18660 = 2700` CU per cohort, which decomposes exactly as `-240*27 + 1260*3 = -2700` from excluded band E. *Red flag:* reporting `0.32` as "the optimal threshold" without noting that it is capacity-imposed rather than cost-optimal, which hides a staffing decision inside a modelling result.

**Q8. Precision over 2,500 actions is 0.528, records cluster in merchants with mean size 4 and ICC 0.2667. Give the correct standard error and explain the direction of the error if you ignore clustering.**

Independent SE: `sqrt(0.528*0.472/2500) = 0.009984`. Design effect `1 + 3*0.2667 = 1.8001`; effective `n = 1388.8`; clustered SE `sqrt(0.528*0.472/1388.8) = 0.013396`. Ignoring clustering understates the interval width by a factor of `sqrt(1.8001) = 1.3416`, or 34%. The 95% interval widens from `[0.5084, 0.5476]` to `[0.5017, 0.5543]`, which can flip a comparison against a 0.551 incumbent from "candidate is better" to "difference not demonstrated". *Red flag:* widening the interval by an arbitrary factor without estimating ICC or cluster size, which is a cosmetic fix rather than a variance estimate.

**Q9. Twenty prespecified slices are tested at `alpha = 0.05`. What is the family-wise risk and what per-test level controls it?**

The union bound gives `20 * 0.05 = 1.0` as an upper bound, and for independent tests the realised value is `1 - 0.95^20 = 0.6415`. Both are large; the bound is loose because slices correlate. Bonferroni at a family rate of 0.05 sets `alpha_per = 0.05/20 = 0.0025`, corresponding to `|z| > 3.023` rather than 2.0. To retain power at the stricter level the per-slice support must grow by roughly `(3.023/2.0)^2 = 2.285` times. *Red flag:* answering "0.05 because each test is at 0.05", which confuses the per-test with the family rate.

**Q10. A model has AUC 1.0000 on four records with scores `[0.99, 0.99, 0.98, 0.97]` and labels `[1, 1, 0, 0]`. What is its Brier score and what does that reveal?**

`0.99 - 1 = -0.01` twice, `0.98 - 0 = 0.98`, `0.97 - 0 = 0.97`. Squares `0.0001, 0.0001, 0.9604, 0.9409`, sum `1.9015`, mean `0.475375`. The constant predictor at `pi = 0.5` scores `0.25`. Perfect ranking coexists with a proper score nearly twice as bad as predicting the base rate for everything. Any rule that sums probabilities, multiplies them by value, or compares them to a cost ratio will be badly wrong. *Red flag:* concluding the model is "excellent" from AUC alone, or concluding it is "useless" from Brier alone; the two facts describe different properties and both are real.

**Q11. A daily threshold is fixed at the score corresponding to 2,500 actions at 19,800 daily records. Demand rises 40% and the score distribution is unchanged. What happens and what are the two defensible policies?**

The same cut now admits `2500 * 1.4 = 3,500` actions, overflowing capacity by 1,000. Policy one: keep the absolute cut and operate an explicit overflow protocol with a documented triage rule. Policy two: convert to a capacity-relative top-k rule that re-derives the cut daily from the score distribution. Both are honest; leaving the behaviour unspecified means the effective policy is chosen by queue pressure. *Red flag:* saying "the model needs retraining", because nothing about the model changed and the failure is entirely in the policy layer.

### 11.3 Judgement

**Q12. A feature raises offline precision from 0.41 to 0.72 in one revision. Give your first three checks in order and say what would make you reject the result.**

First, reconcile totals: confirm the row count and positive count are unchanged, so no join explosion. Second, recompute the feature as of `t0` with events strictly before the decision instant and re-score the frozen cohort. Third, inspect the producing job for post-outcome restatements. I would reject the result if the as-of recomputation removes most of the uplift, because that establishes temporal leakage; I would reject it as *interpretable* if the row count changed, because then the metric describes a different population. Note that in the `VAL-D01` incidence both defects were present, so the honest conclusion is that two causes contributed and their shares cannot be separated from one before-and-after comparison. *Red flag:* accepting the uplift because the feature "makes business sense", which is how most leakage survives review; a plausible mechanism does not establish availability.

**Q13. Your holdout has been consulted by four analysts across three weeks. What is its evidentiary status and what do you do?**

It is no longer a holdout in the statistical sense. `TH-07` requires the procedure to be fixed before inspection; once selection depends on the holdout, the estimate is optimistically biased by an amount that cannot be recovered from the data alone. The remedy is prospective: freeze the current protocol, quarantine the consulted window as a validation set, and obtain a genuinely untouched later window for the final estimate. If no later window exists yet, report the number explicitly as a validation estimate with an unknown optimism gap rather than presenting it as an unbiased one. *Red flag:* "we only looked at aggregate metrics, so it doesn't count". Aggregates are sufficient to guide selection; any inspection that influences a choice breaks the guarantee.

**Q14. You must choose between a model with higher precision and one with lower cost per cohort. How do you decide?**

Precision is not the objective unless the cost ratio is 1:1 in a specific sense, and even then it is not the objective at a fixed capacity; cost is the objective, and precision is one of its inputs. Compare the cost curves over the feasible capacity range, not two points. In the Section 5 sweep, cost at `tau = 0.95` was 48,180 CU with precision 0.98, and cost at `tau = 0.16` was 18,660 CU with precision 0.39; the higher-precision policy costs 2.6 times more. The decision is therefore not "which model is more precise" but "which policy minimises cost subject to capacity, and does either policy violate a hard constraint such as a regulator-mandated review of every high-severity case". If a hard constraint exists, it dominates both numbers. *Red flag:* choosing the higher-precision model because precision is easier to explain to a committee. That is a communication choice masquerading as an analytical one, and it is the correct response only when the cost model is genuinely unspecified.

**Takeaway:** A defensible answer names the denominator, the dependence assumption, and the check that could overturn the conclusion, in that order.

## 12. Quick Reference Card

| Trigger | Rule | Use when | Critical assumption | Avoid when | Check | Next action |
|---|---|---|---|---|---|---|
| Offline score looks too good | Audit availability, then resplit | Any new feature or revision | Timestamps are trustworthy and separate | Source lineage is undocumented | Event vs materialisation ledger; row-count reconciliation | Rebuild as-of data, refreeze protocol |
| Repeated entities in the cohort | Group split on the persistent key | Customers, merchants, devices recur | Key captures the real dependence | Identity linkage is incomplete | Overlap rate; random-vs-group gap | Report group estimate; bound residual linkage |
| Deployment is later than training | Forward split with a maturity gap | Time, policy, or demand drift | Cut-off matches the action instant | Labels are immature at the boundary | Per-fold maturity watermark | Rolling-origin replay |
| Fixed review headcount `C` | Top-k at capacity, `VAL-06` | Actions are scarce | Capacity and demand share an interval | Capacity is unmeasured or informal | Daily actions above the cut | Set overflow protocol; report unserved value |
| Probabilities price the action | Cost threshold `VAL-05` then capacity | `c_FP`, `c_FN` are defensible | `p` is calibrated on the queue | Score is a raw ranking, not a probability | Calibration in the top bands, with `n` | Recalibrate on a separate partition; re-derive |
| Aggregate metric is stable | Slice audit with support and SE | Credible subgroup harm | Slices prespecified | Post-hoc search over subgroups | Slice `n`, missingness, queue age | Gate the slice; repair the source; escalate |
| Interval needed on a rate | Design-effect or grouped bootstrap | Records cluster | ICC estimable | Independent-sample assumption fails | `1 + (m-1)*ICC` | Widen the interval; recompute the comparison |
| Calibration residual is nonzero | Compare against band SE before acting | Magnitude drives a decision | Support per band is adequate | Band holds fewer than ~100 records | Residual in SE units | Platt or band-local remedy per 7.3 |
| Selection used the holdout | Quarantine and obtain a fresh window | Any repeated inspection | A later untouched window exists | No fresh data is available yet | Change log versus holdout access log | Report as validation estimate; label the optimism gap |

**Card reading note.** The nine rows are ordered by the sequence in which the corresponding problems arise: availability first, because it invalidates everything downstream; dependence second, because it changes the estimate; time third, because it changes the population; then policy, calibration, slices, uncertainty, and the integrity of the holdout itself. A reviewer who follows the rows top to bottom has a defensible protocol; one who starts at the bottom is trying to rescue a number that may already be wrong.
