---
collection: "ai-evaluation-monitoring"
file_id: "04"
topic: "Model evaluation and production monitoring"
level: 3
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Changed label definition, deployment, population, or source evidence"
notation_style: "plain-text"
prerequisites: ["Binary classification and probability", "Basic statistical inference", "Production service operations"]
source_status: "OPEN: inventoried source files unavailable"
owner_role: "ML platform and evaluation maintainers"
source_index: "INDEX.md"
---
# Model Evaluation and Production Monitoring
## 1. File Header
**Purpose:** Design an evidence-based monitoring and evaluation loop for predictive models and agentic services.
**Audience:** ML engineers, evaluators, service owners, and incident responders.
**Outcome:** Given predictions, event timestamps, delayed outcomes, and an operational constraint, define valid metrics and choose a proportionate response.
**Prerequisites:** Binary confusion matrices, conditional probability, sampling uncertainty, event time, and service operations.
**Exclusions:** Full causal identification, formal fairness policy, model training algorithms, and vendor-specific APIs.
**Evidence:** All numerical scenarios are constructed; no deployment result is claimed.
**Status:** Draft; not substantively reviewed; validation protocols are proposed rather than executed.
**Retrieval cue:** Start at section 8 for symptoms, section 5 for delayed labels, and section 11 for interview practice.
**Source boundary:** The inventory identifies `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/05_production_monitoring.md` and `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/04_Model_Monitoring_and_Drift.md`.
Those files were unavailable in the working tree; neither their text nor their specific claims was inspected or copied.
The two paths are topic provenance, not verified source citations.
**Primary references:** [Google SRE Workbook, Implementing SLOs](https://sre.google/workbook/implementing-slos/) describes user-centered SLIs and error budgets; consulted 2026-11-07.
[scikit-learn 1.9.1 common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) describes preprocessing consistency and leakage; consulted 2026-11-07.
These references do not validate the illustrative figures or a particular product's quality.
**Takeaway:** Treat operational health, predictive quality, and causal product value as different claims.
## 2. Concept Map / Mental Model
```text
Request + context -> feature contract -> model/version -> score or agent action
       |                 |                   |                 |
       v                 v                   v                 v
 volume, latency      freshness/schema    score mix       action/trace telemetry
       |                 |                   |                 |
       +-----------------+------- detect unusual change -------+
                                       |
                               triage and contain
                                       |
Prediction ID + event time -> eligibility and maturity -> outcome join -> quality estimate
                                       |                    |
                                selection audit       comparison and uncertainty
                                       |                    |
                         offline regression / human review / experiment
                                       |
                   decide: repair data, rollback, threshold, retrain, or observe
                                       |
                         independent validation and controlled rollout
```
Monitoring is continuous collection and interpretation of production signals, often before ground truth exists.
Evaluation applies an explicit task, population, outcome definition, and scoring protocol to estimate quality.
Both can operate online or offline; neither is synonymous with the other.
An agent can finish a run successfully while producing an incorrect or unsafe answer.
A model can have stable infrastructure latency while its ranking quality degrades.
A feature distribution change is a signal to investigate, not an automatic diagnosis of model failure.
A downstream action can change the outcome being measured and thereby change the meaning of a label.
**Takeaway:** Monitoring detects and localizes; evaluation determines what a model achieved under an explicit protocol.
## 3. Foundational Definitions
| Term | Definition and scope | Type / units | Common confusion |
|---|---|---|---|
| Prediction | Model output for one decision at a recorded time | Event, score or label | Not the eventual outcome |
| Feature | Input available to the model at decision time | Scalar/vector, feature-specific units | Not a future outcome |
| Outcome | Event defined over a specified future horizon | Binary, numeric or categorical | Not necessarily observed immediately |
| Cohort | Predictions selected by decision time and eligibility | Set of decision IDs | Not a label-arrival-time window |
| Maturity | Time when outcome ascertainment should be substantially complete | Timestamp/duration | Not proof all labels arrived |
| Coverage | Joined eligible labels divided by eligible predictions | Dimensionless fraction | Not model accuracy |
| Feature shift | Change in input distribution | Comparison of distributions | Not necessarily quality loss |
| Covariate shift | Input distribution changes while outcome conditional on inputs remains stable | Assumption about joint distributions | Not every feature shift |
| Concept drift | Conditional outcome distribution given inputs changes | Distributional change | Not established by inputs alone |
| Label shift | Outcome prevalence changes while input conditional on outcome stays stable | Specific distributional assumption | Not any observed class-mix change |
| Statistical significance | Incompatibility of data with a specified null under a test | Test result | Not practical significance or causal proof |
| Evaluation | Scored assessment under a declared protocol | Estimate and uncertainty | Not equivalent to uptime |
| SLI | Defined indicator of service experience | Often good events / eligible events | Not the target itself |
| SLO | Agreed target for an SLI over a period | Target fraction and window | Not an SLA contract |
| Selection bias | Observed labels differ systematically from target population | Sampling mechanism | Not fixed by a bigger biased sample |
| Calibration | Predicted probabilities agree with observed rates in defined groups | Conditional relationship | Not the same as ranking |
| Precision | Positive outcomes among predicted positives | Fraction | Not recall |
| Recall | Predicted positives among actual positives | Fraction | Not precision |
| Trace | Linked events representing a request or run | Structured event graph | Not proof the final answer is correct |
| Window | Explicit interval for input or event selection | Duration and clock | Not interchangeable with cohort maturity |
Each probability refers to a specified population and time; individual outcomes remain uncertain.
Define decision entity, version, treatment, horizon, label source, time zone, and excluded traffic before comparing rates.
A metric without a denominator and eligibility rule is not an operational contract.
**Takeaway:** Define the population, clock, and outcome before comparing numbers.
## 4. Core Formulas with Full Derivation
### 4.1 Base: service indicator
Validity: good and eligible events are counted once under the same definition and window; eligible count is positive.
Derivation: a fraction is the count in the good subset divided by the count in its eligible parent set.
```text
FORMULA: SLI = G / N.
WHERE: G is good eligible events; N is all eligible events; both are nonnegative integer counts.
CONSTRAINT: 0 <= G <= N and N > 0; define what good means before observing the result.
NUMERICAL EXAMPLE: G=9900 and N=10000 gives SLI=0.99 = 99%.
INTERPRETATION: 100 events did not satisfy this specified good-event definition.
```
Boundary: zero eligible events means undefined, not 100% success.
For a 99.9% target over 10,000 eligible requests the allowed bad-event count is 10, not 100.
The target, measurement point, window and exclusions must all be agreed, as emphasized by the SRE reference.
### 4.2 General: confusion matrix and delayed labels
Validity: predictions and adjudicated outcomes refer to the same eligible mature decision cohort and threshold.
Derivation: partition predicted positives into true and false positives; partition actual positives into true positives and false negatives.
```text
FORMULA: precision = TP/(TP+FP); recall = TP/(TP+FN); coverage = L/N.
WHERE: TP, FP and FN are integer decision counts; L is cohort decisions with usable labels; N is eligible decisions.
CONSTRAINT: Each denominator must be positive; coverage is not a quality metric and missing labels may be selective.
NUMERICAL EXAMPLE: TP=60, FP=140, FN=40, L=1000, N=1000 gives precision=.30, recall=.60, coverage=1.
INTERPRETATION: 30% of flagged cases are positive and 60% of all positives are flagged.
```
Boundary: if outcomes are available only for flagged cases, recall for all decisions is not identified from these data alone.
### 4.3 Variant: feature distribution comparison
Validity: reference and current observations use identical feature semantics and fixed bins, with positive bin proportions.
Derivation: summing signed differences weighted by log current/reference gives the symmetric divergence often called population stability index (PSI).
```text
FORMULA: PSI = sum_i (q_i - p_i) * ln(q_i/p_i).
WHERE: p_i and q_i are reference and current fractions in bin i; ln is the natural logarithm.
CONSTRAINT: Each distribution sums to one; every compared proportion is positive; compare like populations and feature definitions.
NUMERICAL EXAMPLE: p=(0.5,0.5), q=(0.6,0.4); PSI=0.1*ln(1.2) - 0.1*ln(0.8) approximately 0.04055.
INTERPRETATION: Binned marginal input composition changed; the value alone does not establish harm.
```
A separately specified smoothing rule may avoid log zero but alters the estimator and must be reported.
PSI depends on bins, sample sizes, cohort choice, and reference period; universal operational cutoffs are not justified here.
### 4.4 Limits and counterexamples: statistical change versus impact
Validity: the following standard error describes an independent Bernoulli sample with a fixed sample size; approximation is less reliable near boundaries or with small counts.
Derivation: a Bernoulli observation has variance p(1-p); averaging n independent observations divides variance by n.
```text
FORMULA: SE(p_hat) approximately sqrt(p_hat*(1-p_hat)/n).
WHERE: p_hat is the sample proportion, n is its independent observation count, and SE is dimensionless.
CONSTRAINT: Fixed sampling design, approximate independence, adequate counts; clustering requires an appropriate alternative.
NUMERICAL EXAMPLE: p_hat=.5 and n=10000 gives SE=sqrt(.25/10000)=.005.
INTERPRETATION: One standard error is 0.5 percentage points, not a causal effect or a 95% interval.
```
A large sample can make a tiny harmless shift statistically detectable; a small sample can miss a consequential change.
Repeated tests across many features and hours increase false-alert opportunities; predeclare a multiple-testing and sequential-monitoring strategy.
A sudden feature shift can be a unit-conversion bug, an intentional product launch, or real population change.
Stable feature marginals can coexist with changed feature interactions or changed outcomes conditional on the same features.
**Takeaway:** A quantified shift is evidence about a chosen comparison, not proof of root cause or degraded utility.
## 5. Worked Numerical Example
### 5.1 Mature cohort and operational counts
**ILLUSTRATIVE:** A fraud model scores 1,000 eligible transactions on day 1.
Each decision is identified by a stable decision ID and recorded model and feature versions.
A positive outcome means adjudicated fraud within a 30-day horizon, as known at the 45-day reporting cutoff.
The extra 15 days is an assumed reporting lag allowance, not a guarantee of complete adjudication.
At cutoff, 900 decisions have usable outcomes; 100 remain pending or unlinked.
Coverage is 900/1000 = 90%, using the formula in section 4.2.
The unlabeled 100 must remain unknown rather than being coded as nonfraud.
Among the 900 labeled decisions, 80 are positives and 820 are negatives.
Of the 80 positives, 48 were flagged and 32 were not.
Of the 820 negatives, 72 were flagged and 748 were not.
The labeled confusion matrix totals 48+32+72+748 = 900.
Flagged labeled cases total 48+72 = 120.
Precision on labeled cases is 48/120 = 40%.
Recall on labeled positives is 48/80 = 60%.
The remaining 100 could change both estimates; these are not automatically population-wide metrics.
If every missing decision was an unflagged fraud, cohort recall would be 48/180 = 26.67%.
That extreme is illustrative, not a prediction of the missing labels.
If every missing decision was a flagged nonfraud, precision would be 48/220 = 21.82%.
That second extreme uses a different missing-data allocation and is not simultaneously assumed with the first.
First inspect label join failures, adjudication queue, and coverage by score and segment.
Then obtain a representative audited sample or use a justified missingness model, with uncertainty.
### 5.2 Decision and independent checks
Review capacity is assumed to be 150 alerts per day, counting all eligible alerts, not just labeled ones.
The 120 labeled alerts do not establish whether the total alerts fit capacity because the 100 unlabeled cases may include alerts.
Measure the full prediction log before changing the threshold.
A threshold reduction may increase recall but also reviews and false positives; validate using representative outcomes.
A threshold increase may reduce reviews while increasing missed fraud and may affect subgroups differently.
Neither choice follows from precision alone.
Arithmetic cross-check: fraud prevalence in labeled cases is 80/900 = 8.89%; nonfraud count is 900-80 = 820.
Arithmetic cross-check: 48 detected plus 32 missed equals the 80 labeled positives.
Arithmetic cross-check: 120 alerts comprise 48 correct plus 72 incorrect.
### 5.3 Feature shift and significance
Assume a stable two-bin feature definition with reference proportions 50% low and 50% high.
Current proportions are 60% low and 40% high; the PSI calculation in section 4.3 is approximately 0.04055.
That shift does not tell whether false negatives rose, whether a new population arrived, or whether a schema broke.
Compare feature provenance, unit and missingness before considering model retraining.
Then compare mature-label quality by segment and against a suitable baseline.
If the 10-point change follows an intentional expansion into a new region, compare within region as well as overall.
If a unit change caused the shift, repair and replay the pipeline rather than learning from corrupted inputs.
If outcome relationships changed on representative labels, evaluate a new model against the fixed baseline and guardrails.
### 5.4 Agent run example
**ILLUSTRATIVE:** Of 200 agent runs, 160 complete and 140 satisfy a predefined independent quality rubric.
Completion is 160/200 = 80% of attempted runs.
If all 140 rubric successes are among completed runs, success per attempt is 140/200 = 70%.
Conditional success among completed runs is 140/160 = 87.5%.
If total included model and tool cost is 28 currency units, cost per successful run is 28/140 = 0.2 currency units.
These figures do not cover omitted human-review costs, harm, or a representative population.
Treat incomplete runs as non-successes only if the evaluation protocol explicitly specifies that rule.
A rise in completion without a rise in rubric success can be a proxy improvement, not a quality improvement.
**Takeaway:** Mature cohorts, label coverage, and correct denominators determine what each metric actually says.
## 6. Visual / Geometric Intuition
```text
Decision cohort: |---------------------------------- 1000 ----------------------------------|
Observed labels: |--------------------------- 900 ---------------------------|---- 100 ----|
                                                                    pending or unlinked
Observed positive outcomes: |---- flagged 48 ----|--- unflagged 32 ---|
Observed negative outcomes: |------- flagged 72 -------|--------------- unflagged 748 ---------------|
```
The diagram shows a partition, not a time series or a representative sample guarantee.
```text
Time of decision ---- 30-day outcome horizon ---- 15-day reporting buffer ---- evaluation cutoff
                     outcomes can occur            labels can still arrive       freeze an as-of view
```
Some labels can still be absent or revised after the cutoff.
```text
Reference marginal: low 50% | high 50%
Current marginal:   low 60% | high 40%
Question remaining: within each bin, did outcomes or decisions change?
```
A marginal chart conceals interactions and subgroup composition.
Plot label coverage next to quality, and mark version or traffic changes on the same time axis.
Plot numerator and denominator rather than a rate alone when volume can change.
Show uncertainty and actual sample count on each slice; do not draw an unsupported smooth trend.
**Takeaway:** Time, selection, and distribution geometry reveal why a single line chart can mislead.
## 7. Comparison / Selection Table
| Choice | Use when | Advantage | Disqualifier / caution | Next evidence |
|---|---|---|---|---|
| Request success SLI | Prompt operational detection matters | Available before outcome labels | Healthy requests can be wrong | Trace and quality sample |
| Offline holdout | Candidate comparisons require controlled inputs | Reproducible paired comparisons | Stale or contaminated set | Fresh held-out production sample |
| Mature online cohort | Outcomes arrive after a known horizon | Measures operational population | Selective or delayed labels | Coverage and adjudication audit |
| Feature drift | Labels are delayed | Early investigation cue | Feature shift need not cause harm | Pipeline and outcome diagnostics |
| Score drift | Output mix matters operationally | Can reveal threshold load changes | Stable scores can hide concept drift | Mature quality and slices |
| Human review | Outcomes are nuanced or safety-sensitive | Rubric captures qualitative defects | Rater disagreement and sampling bias | Double rating and adjudication |
| Randomized experiment | Intervention impact is the decision | Can support causal effect under design assumptions | Unsafe or infeasible allocation | Guardrails, adherence, and power |
| Shadow deployment | Candidate needs live inputs without action | Exercises serving path | No direct effect on outcomes | Paired outcomes where valid |
| Canary | Rollout risk must be bounded | Limits exposed population | Small canary may miss rare harms | Prespecified gates and rollback |
| Retraining | Representative evidence shows model inadequacy | Can adapt to changed relationships | Corrupt inputs or selective labels | Held-out and prospective validation |
| Pipeline repair | Broken units, schema or stale features confirmed | Corrects root cause | Cannot by itself fix concept drift | Replay and contract checks |
| Threshold adjustment | Review capacity or decision cost changed | Fast policy change | May redistribute missed harms | Mature cost and subgroup analysis |
For each choice, the owner records the decision, evidence, uncertainty, and reversal condition.
Severity of user harm overrides a routine statistical significance threshold.
When evidence is inadequate, use a conservative fallback appropriate to the product and affected users.
**Takeaway:** Choose interventions by diagnosed mechanism, not by the name of an alert.
## 8. Failure Modes & Edge Cases
All scenarios below are hypothetical diagnostic patterns, not observed incidents.
Each provides competing explanations and a way to discriminate them.
### 8.1 No labels after horizon
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** No labels after horizon.
- **Candidate mechanism:** Outcome feed has stalled.
- **Competing mechanism:** Genuinely slow adjudication.
- **Discriminator:** Compare feed watermark with upstream adjudication timestamps.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore or replay the outcome feed.
- **Prevention:** Track completeness and feed freshness separately.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Do not classify pending decisions as negatives.
### 8.2 Precision appears to rise immediately
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Precision appears to rise immediately.
- **Candidate mechanism:** Only fast positive outcomes arrived.
- **Competing mechanism:** True improved discrimination.
- **Discriminator:** Stratify arrival lag by outcome and score.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Wait for comparable maturity and audit remaining labels.
- **Prevention:** Publish coverage alongside precision.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Early precision is conditional on observed labels.
### 8.3 Recall looks perfect
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Recall looks perfect.
- **Candidate mechanism:** Unflagged cases lack outcomes.
- **Competing mechanism:** A genuinely perfect classifier.
- **Discriminator:** Measure label coverage separately for flagged and unflagged cases.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Sample and adjudicate unflagged cases.
- **Prevention:** Require a representative label acquisition design.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Recall cannot be recovered from positives-only review.
### 8.4 Quality graph changes on backfill
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Quality graph changes on backfill.
- **Candidate mechanism:** Labels were revised after initial reporting.
- **Competing mechanism:** Cohort definition changed.
- **Discriminator:** Recompute the same cohort under two as-of snapshots.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Version label snapshots and explain the revision.
- **Prevention:** Track revisions and historical restatement policy.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Do not silently overwrite previously published estimates.
### 8.5 Prediction joins multiply
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Prediction joins multiply.
- **Candidate mechanism:** Outcome table contains duplicate updates.
- **Competing mechanism:** Repeated legitimate decisions per entity.
- **Discriminator:** Count decision IDs before and after join and inspect effective dates.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Join to the correct adjudication version per decision.
- **Prevention:** Enforce explicit uniqueness and audit unmatched IDs.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** One entity can legitimately have multiple decision IDs.
### 8.6 Most labels fail to join
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Most labels fail to join.
- **Candidate mechanism:** Decision identifiers differ across systems.
- **Competing mechanism:** Outcome events truly do not exist.
- **Discriminator:** Trace a sampled ID through both pipelines.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Repair identifier mapping and replay safely.
- **Prevention:** Add end-to-end reconciliation and unmatched-ID alerts.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Never use a many-to-many fuzzy join without audit.
### 8.7 Fraud rate falls after blocking
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Fraud rate falls after blocking.
- **Candidate mechanism:** Blocking prevents some outcomes from occurring.
- **Competing mechanism:** The risk population has become safer.
- **Discriminator:** Compare outcome availability and action policy by risk band.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Record action and use a designed audit or experiment.
- **Prevention:** Separate observed outcomes from counterfactual untreated outcomes.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Observed labels do not identify prevented fraud automatically.
### 8.8 Feature values suddenly multiply
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Feature values suddenly multiply.
- **Candidate mechanism:** A source changed currency units.
- **Competing mechanism:** Customers actually spend more.
- **Discriminator:** Compare raw source units and transformation versions.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore the feature contract and replay affected decisions where feasible.
- **Prevention:** Add unit metadata and train-serving parity checks.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Retraining on corrupted units can institutionalize the bug.
### 8.9 Missing feature rate rises
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Missing feature rate rises.
- **Candidate mechanism:** An upstream join key changed.
- **Competing mechanism:** The product launched a population with missing data.
- **Discriminator:** Compare missingness by producer version and cohort.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Repair the join or apply an approved fallback.
- **Prevention:** Monitor upstream schema and missingness per slice.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** An imputed value is not proof the source value exists.
### 8.10 Feature age increases
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Feature age increases.
- **Candidate mechanism:** Scheduled update is late.
- **Competing mechanism:** Traffic is using legitimately old records.
- **Discriminator:** Compare event timestamps, processing timestamps and access-weighted freshness.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore the update pipeline and assess impacted decisions.
- **Prevention:** Alert on a user-relevant freshness SLI.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Processing time alone can hide old event data.
### 8.11 Input drift alert after launch
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Input drift alert after launch.
- **Candidate mechanism:** The customer mix changed intentionally.
- **Competing mechanism:** An extraction or unit defect.
- **Discriminator:** Compare within-segment distributions and deployment timeline.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Verify launch, quality and segment-specific impact.
- **Prevention:** Maintain segmented reference cohorts.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A changed marginal alone cannot establish concept drift.
### 8.12 No drift alert but quality declines
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** No drift alert but quality declines.
- **Candidate mechanism:** Feature interactions or outcome relationship changed.
- **Competing mechanism:** Ground-truth or joining error.
- **Discriminator:** Inspect multivariate slices and independently adjudicated mature outcomes.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Investigate concept and label shifts before model changes.
- **Prevention:** Monitor joint patterns and mature quality.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Univariate stability is not a quality guarantee.
### 8.13 Feature shift is statistically significant
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Feature shift is statistically significant.
- **Candidate mechanism:** Large sample detects a tiny change.
- **Competing mechanism:** A material population change.
- **Discriminator:** Compare effect size, uncertainty and downstream sensitivity.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Set investigation priority from impact and reproducibility.
- **Prevention:** Define practical thresholds and sample-size rules.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A p-value does not measure business loss.
### 8.14 Feature shift is not significant
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Feature shift is not significant.
- **Candidate mechanism:** Sample size is too small.
- **Competing mechanism:** There truly is little change.
- **Discriminator:** Inspect confidence intervals and minimum detectable effect.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Collect additional representative evidence if safe.
- **Prevention:** Design alert sensitivity around risk and volume.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Failure to reject is not evidence of equivalence.
### 8.15 Many simultaneous drift alerts
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Many simultaneous drift alerts.
- **Candidate mechanism:** Multiple comparisons generate noise.
- **Competing mechanism:** A shared upstream change affects many features.
- **Discriminator:** Group alerts by producer and inspect feature lineage.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Investigate the earliest common dependency.
- **Prevention:** Use a prespecified multiplicity and grouping policy.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Shared causation is not proven merely by co-occurrence.
### 8.16 PSI becomes infinite
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** PSI becomes infinite.
- **Candidate mechanism:** A compared bin has zero observations.
- **Competing mechanism:** The feature genuinely has new support.
- **Discriminator:** Inspect raw counts, bin construction and sample size.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Report support change and apply documented smoothing only if justified.
- **Prevention:** Preserve fixed bins and raw counts.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Smoothing changes the numerical result.
### 8.17 PSI drops after changing bins
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** PSI drops after changing bins.
- **Candidate mechanism:** The comparison discretization changed.
- **Competing mechanism:** The population returned to baseline.
- **Discriminator:** Recompute both periods with identical fixed bins.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore comparable binning and annotate the dashboard.
- **Prevention:** Version the reference and bin edges.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** PSI values across different binning schemes are not directly comparable.
### 8.18 Reference drift is constant
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Reference drift is constant.
- **Candidate mechanism:** The baseline captures a seasonal peak.
- **Competing mechanism:** The process truly shifted permanently.
- **Discriminator:** Compare with aligned historical periods and segment mix.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Select a justified seasonally matched comparator.
- **Prevention:** Review reference cohort periodically with approval.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Replacing a reference must not erase incident evidence.
### 8.19 Score distribution changes
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Score distribution changes.
- **Candidate mechanism:** A threshold or model version changed.
- **Competing mechanism:** Upstream inputs changed.
- **Discriminator:** Split scores by model version, policy and input source.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Roll back a defective deployment or investigate its inputs.
- **Prevention:** Tag score telemetry with immutable versions.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Score shift is not itself outcome quality.
### 8.20 Score distribution is stable but precision falls
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Score distribution is stable but precision falls.
- **Candidate mechanism:** Outcome prevalence changed.
- **Competing mechanism:** Conditional model discrimination worsened.
- **Discriminator:** Compare prevalence, score-conditional outcomes and ranking on mature labels.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Investigate case mix and recalibration needs.
- **Prevention:** Track both calibration and ranking.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Stable score mix cannot guarantee stable precision.
### 8.21 Calibration fails in a new region
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Calibration fails in a new region.
- **Candidate mechanism:** Outcome prevalence or conditional risk differs.
- **Competing mechanism:** The label process differs by region.
- **Discriminator:** Check coverage, outcome definitions and calibration bins by region.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Assess region-specific recalibration with holdout data.
- **Prevention:** Monitor calibration and data quality per region.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Small subgroup bins carry substantial uncertainty.
### 8.22 AUC rises while operational value falls
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** AUC rises while operational value falls.
- **Candidate mechanism:** Threshold costs and capacity changed.
- **Competing mechanism:** Evaluation populations differ.
- **Discriminator:** Examine recall, precision and cost at feasible thresholds.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Choose policy using declared costs and capacity.
- **Prevention:** Monitor utility and workload alongside ranking.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** AUC does not choose an operating threshold.
### 8.23 Overall quality improves while a slice worsens
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Overall quality improves while a slice worsens.
- **Candidate mechanism:** Population weights changed.
- **Competing mechanism:** One slice suffered a genuine regression.
- **Discriminator:** Compare within-slice metrics and cohort proportions.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Investigate the harmed slice and apply guardrails.
- **Prevention:** Maintain stable slice definitions and minimum counts.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Aggregated metrics can hide subgroup harm.
### 8.24 Latency improves while task success falls
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Latency improves while task success falls.
- **Candidate mechanism:** A faster fallback bypasses needed work.
- **Competing mechanism:** The quality evaluator changed.
- **Discriminator:** Compare path, fallback rate and rubric outcomes by version.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore critical work or route safely.
- **Prevention:** Gate latency gains on quality and safety.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Lower latency alone is not better service.
### 8.25 Completion rises but answers worsen
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Completion rises but answers worsen.
- **Candidate mechanism:** The agent exits early with incomplete work.
- **Competing mechanism:** The rubric became stricter.
- **Discriminator:** Audit traces and blinded rubric samples under both versions.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Rollback or fix stopping conditions.
- **Prevention:** Measure success per attempted run and critical failures.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A completed run is not necessarily a successful task.
### 8.26 Tool error rate stays flat while users fail
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Tool error rate stays flat while users fail.
- **Candidate mechanism:** Tool responses are syntactically valid but semantically wrong.
- **Competing mechanism:** Requests are failing before reaching the tool.
- **Discriminator:** Review end-to-end traces and user-visible outcomes.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Fix tool contracts or validation at the boundary.
- **Prevention:** Monitor semantic failures and coverage separately.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** HTTP success is not task correctness.
### 8.27 Run cost suddenly increases
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Run cost suddenly increases.
- **Candidate mechanism:** An agent loops on repeated tool calls.
- **Competing mechanism:** Input complexity genuinely increased.
- **Discriminator:** Compare call counts, token usage and task mix by version.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Apply bounded stopping and repair the loop.
- **Prevention:** Monitor cost per attempt and per successful task.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A cost cap alone may truncate valid tasks.
### 8.28 Agent answers cite irrelevant passages
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Agent answers cite irrelevant passages.
- **Candidate mechanism:** Retrieval returns weak or unauthorized context.
- **Competing mechanism:** Generation ignores good context.
- **Discriminator:** Inspect retrieved documents, permissions and citation support separately.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Fix retrieval or grounding according to the failing stage.
- **Prevention:** Evaluate retrieval relevance and answer support independently.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A citation string does not prove entailment.
### 8.29 Agent succeeds offline but fails online
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Agent succeeds offline but fails online.
- **Candidate mechanism:** Offline tasks omit live tool failure modes.
- **Competing mechanism:** Online evaluator or traffic definition changed.
- **Discriminator:** Replay representative traces with versioned tool contracts.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Expand evaluation to relevant online conditions.
- **Prevention:** Keep held-out scenarios and production sampling distinct.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Replay cannot reconstruct unrecorded external state.
### 8.30 Judge scores rise unexpectedly
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Judge scores rise unexpectedly.
- **Candidate mechanism:** Candidate learned to exploit the judge rubric.
- **Competing mechanism:** Real quality improved.
- **Discriminator:** Blind and randomize human review of a representative sample.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Revise judge and rubric using independently held-out examples.
- **Prevention:** Monitor judge-human agreement and disagreement slices.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A model judge is a measurement tool, not ground truth.
### 8.31 Raters disagree on safety
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Raters disagree on safety.
- **Candidate mechanism:** Rubric permits multiple interpretations.
- **Competing mechanism:** Examples genuinely lie near a policy boundary.
- **Discriminator:** Measure agreement and inspect adjudicated disagreements.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Clarify rubric and document escalation criteria.
- **Prevention:** Recalibrate raters using separate development examples.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Agreement does not prove the rubric is correct.
### 8.32 One subgroup has no reported errors
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** One subgroup has no reported errors.
- **Candidate mechanism:** The slice is too small.
- **Competing mechanism:** The subgroup really has no failures in this sample.
- **Discriminator:** Report denominators and an uncertainty interval.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Collect more evidence or aggregate only with justification.
- **Prevention:** Require minimum sample and suppression rules.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Zero observed errors does not prove zero risk.
### 8.33 Alert does not fire during outage
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Alert does not fire during outage.
- **Candidate mechanism:** Monitoring uses only successful server-side logs.
- **Competing mechanism:** No users actually reached the service.
- **Discriminator:** Compare client, load balancer and server request counts.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore telemetry and use an independent external probe.
- **Prevention:** Monitor the monitoring pipeline itself.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A missing denominator must not be treated as healthy.
### 8.34 Alert floods on low traffic
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Alert floods on low traffic.
- **Candidate mechanism:** Rates are unstable with tiny denominators.
- **Competing mechanism:** A severe true incident affects few users.
- **Discriminator:** Inspect event counts, error types and user severity.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Use a count-aware alert with an explicit urgent safety path.
- **Prevention:** Tune windows and route alerts by actionability.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Low volume does not make a critical harm unimportant.
### 8.35 Canary is healthy but rollout fails
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Canary is healthy but rollout fails.
- **Candidate mechanism:** The canary omitted a critical traffic segment.
- **Competing mechanism:** Rollout introduced another independent change.
- **Discriminator:** Compare routing, segment coverage and versions across stages.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Halt or rollback and recover affected traffic.
- **Prevention:** Stratify canaries and stage changes independently.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A small canary cannot exclude rare failures.
### 8.36 Rollback does not restore quality
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Rollback does not restore quality.
- **Candidate mechanism:** Feature schema or policy remained on the new version.
- **Competing mechanism:** The issue predates the release.
- **Discriminator:** Compare the complete model, data and policy bundle.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore a compatible bundle and replay validation.
- **Prevention:** Version and rehearse bundle rollback.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Rolling back weights alone can leave incompatibility.
### 8.37 Shadow candidate shows perfect recall
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Shadow candidate shows perfect recall.
- **Candidate mechanism:** Offline replay contains future-derived features.
- **Competing mechanism:** The new model genuinely improved.
- **Discriminator:** Audit feature availability timestamps for each decision.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Remove leaked information and rebuild evaluation.
- **Prevention:** Use point-in-time feature joins and held-out data.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Shadowing does not automatically eliminate leakage.
### 8.38 Test quality exceeds production
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Test quality exceeds production.
- **Candidate mechanism:** The test set was used for repeated tuning.
- **Competing mechanism:** Production traffic or labels differ.
- **Discriminator:** Inspect evaluation access history and cohort definitions.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Reserve a fresh untouched test cohort.
- **Prevention:** Separate development and final evaluation access.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Repeated adaptation invalidates naive holdout interpretation.
### 8.39 Agent safety incidents rise
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Agent safety incidents rise.
- **Candidate mechanism:** A permission or approval boundary changed.
- **Competing mechanism:** The detector became more sensitive.
- **Discriminator:** Compare policy versions, actual actions and adjudicated reports.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Disable the unsafe capability or restore the guardrail.
- **Prevention:** Test authorization and approval on every relevant action.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Safety monitoring cannot replace enforcement.
### 8.40 Retrained model worsens immediately
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Retrained model worsens immediately.
- **Candidate mechanism:** Recent labels are selectively observed.
- **Competing mechanism:** A serving or preprocessing mismatch.
- **Discriminator:** Compare training sample selection and train-serving features.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Rollback and rebuild using an appropriate sampling design.
- **Prevention:** Audit selection, lineage and temporal splits before promotion.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** More recent data is not automatically more representative.
### 8.41 Model appears overconfident
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Model appears overconfident.
- **Candidate mechanism:** Probability calibration deteriorated.
- **Competing mechanism:** The scoring scale or label definition changed.
- **Discriminator:** Compare reliability bins against mature outcomes.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Recalibrate on an independent representative sample.
- **Prevention:** Track calibration with adequate counts and uncertainty.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Calibration does not imply good ranking.
### 8.42 Outcomes arrive before decisions
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Outcomes arrive before decisions.
- **Candidate mechanism:** Clock skew or incorrect event-time semantics.
- **Competing mechanism:** The outcome legitimately predates this decision.
- **Discriminator:** Inspect raw timestamps, zones and entity identity.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Repair timestamp mapping or correct eligibility rules.
- **Prevention:** Document event and processing time contracts.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A future label must not leak into the feature vector.
### 8.43 A single entity dominates quality
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** A single entity dominates quality.
- **Candidate mechanism:** Many correlated decisions belong to one entity.
- **Competing mechanism:** The workload really is concentrated.
- **Discriminator:** Compare decision-weighted and entity-weighted summaries.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Choose the estimand matching the business decision.
- **Prevention:** Use entity-aware uncertainty or resampling.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Treating correlated decisions as independent understates uncertainty.
### 8.44 Error rate drops after excluding timeouts
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Error rate drops after excluding timeouts.
- **Candidate mechanism:** The denominator silently excludes failed requests.
- **Competing mechanism:** The service truly improved.
- **Discriminator:** Reconcile eligible client requests with outcome categories.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore the agreed eligibility and bad-event definitions.
- **Prevention:** Version SLI queries and review exclusions.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Filtering failures out of the denominator creates false health.
### 8.45 Model drift coincides with a promotion
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Model drift coincides with a promotion.
- **Candidate mechanism:** Traffic allocation changed with the release.
- **Competing mechanism:** The candidate itself caused the shift.
- **Discriminator:** Compare randomized or matched concurrent cohorts.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Use the planned rollback or controlled comparison.
- **Prevention:** Record assignment and version at decision time.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A before-after difference alone is not causal.
### 8.46 Review queue overflows
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Review queue overflows.
- **Candidate mechanism:** Threshold yields more alerts than staff can inspect.
- **Competing mechanism:** Case handling time increased.
- **Discriminator:** Compare alert arrival rate, staffing and service time.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Apply an approved prioritization or fallback policy.
- **Prevention:** Track backlog age, throughput and missed cases.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A lower alert rate may also lower fraud capture.
### 8.47 Segment metric fluctuates by day
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Segment metric fluctuates by day.
- **Candidate mechanism:** Small samples and correlated outcomes create noise.
- **Competing mechanism:** A recurring real incident affects the segment.
- **Discriminator:** Inspect counts, interval estimates and event clusters.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Aggregate over a justified window and investigate severe events.
- **Prevention:** Use minimum evidence rules without suppressing urgent harm.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Overlapping rolling windows are not independent experiments.
### 8.48 Feature drift disappears after backfill
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Feature drift disappears after backfill.
- **Candidate mechanism:** The initial ingestion was incomplete.
- **Competing mechanism:** The historical comparator was rewritten.
- **Discriminator:** Compare immutable snapshots and ingestion watermarks.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Correct ingestion and annotate revised analysis.
- **Prevention:** Keep data lineage and as-of snapshots.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** A recomputed history is not the original operational view.
### 8.49 Anomaly detector fires at midnight
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Anomaly detector fires at midnight.
- **Candidate mechanism:** Timezone or daylight-saving boundaries shifted.
- **Competing mechanism:** A true scheduled traffic change occurred.
- **Discriminator:** Compare timestamps in UTC and local business time.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Correct the window alignment.
- **Prevention:** Store timezone-aware timestamps and explicit calendar rules.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Daily windows can have unequal durations.
### 8.50 A newly launched category is all unknown
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** A newly launched category is all unknown.
- **Candidate mechanism:** The encoder vocabulary is stale.
- **Competing mechanism:** The producer is sending invalid values.
- **Discriminator:** Compare category catalog and producer version.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Update a compatible encoder or use a safe unknown path.
- **Prevention:** Monitor unknown-category rate by source.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Silently mapping all new categories to one value loses information.
### 8.51 Review precision improves after stricter triage
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Review precision improves after stricter triage.
- **Candidate mechanism:** Reviewers now inspect only easier cases.
- **Competing mechanism:** The model genuinely improved.
- **Discriminator:** Compare review selection probabilities and score strata.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Audit a representative sample across the full eligible population.
- **Prevention:** Log selection policy and label acquisition route.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Reviewed-case precision need not generalize to all alerts.
### 8.52 Metric changes after model migration
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Metric changes after model migration.
- **Candidate mechanism:** Model version labels were missing or incorrect.
- **Competing mechanism:** The candidate genuinely changes behavior.
- **Discriminator:** Reconcile deployment manifest against per-decision logs.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Repair provenance and recompute comparable cohorts.
- **Prevention:** Make model and feature versions mandatory.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Unattributed metrics cannot identify a responsible version.
### 8.53 Output schema failures rise
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Output schema failures rise.
- **Candidate mechanism:** A model or tool emits incompatible fields.
- **Competing mechanism:** The validator is rejecting legitimate new schema.
- **Discriminator:** Compare schema versions and actual payloads.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Restore compatibility or roll back the producer.
- **Prevention:** Use contract tests and staged schema changes.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Never silently coerce invalid high-risk actions.
### 8.54 Outcomes improve while product value falls
- **Provenance:** Hypothetical; no incident is claimed.
- **Observed symptom:** Outcomes improve while product value falls.
- **Candidate mechanism:** The model optimized a proxy instead of user value.
- **Competing mechanism:** External economics changed.
- **Discriminator:** Compare outcome metric with cost, user harm and long-term metrics.
- **First response:** Preserve the affected cohort, version, timestamps and raw counts.
- **Containment:** Use the product's approved safe fallback if user risk is material.
- **Correction:** Revisit objective and require relevant guardrails.
- **Prevention:** Maintain a metric tree from service to user impact.
- **Validation plan:** Compare the same eligible population and metric definitions before and after the correction.
- **Escalation:** Involve the data owner and service owner if the discriminator is inconclusive.
- **Boundary:** Proxy gains do not establish causal product value.
**Takeaway:** A symptom has multiple plausible causes; discriminate before choosing a fix.
### Operational metric contracts
The following distinct contracts are monitoring design prompts; none is a measured deployment result.
**Contract 1: Eligibility count**
- **Definition:** Count decisions satisfying the policy-defined target population before filtering on outcomes.
- **Population and timing:** Exclude test traffic only by a declared rule, not because it failed.
- **Diagnostic use:** Reconcile this count against the request ingress and decision log.
- **Interpretation boundary:** A missing eligibility feed makes downstream rates uninterpretable.
**Contract 2: Prediction coverage**
- **Definition:** Count eligible requests with a recorded decision and divide by eligible requests.
- **Population and timing:** Include timeouts and rejected scoring attempts in the denominator where policy requires.
- **Diagnostic use:** Split missing predictions by endpoint and deployment version.
- **Interpretation boundary:** A fallback decision should be tagged separately from a model prediction.
**Contract 3: Outcome maturity**
- **Definition:** Group decisions by decision timestamp and require the full predefined outcome horizon.
- **Population and timing:** Apply a reporting buffer based on the actual label acquisition process.
- **Diagnostic use:** Compare remaining label arrivals across prior cohorts at the same age.
- **Interpretation boundary:** Some cases can remain unresolved even after the buffer.
**Contract 4: Label completeness**
- **Definition:** Count usable linked labels among eligible mature decisions.
- **Population and timing:** Display pending, disputed, censored, and unlinked cases as separate categories.
- **Diagnostic use:** Compare completeness by score band and intervention.
- **Interpretation boundary:** High overall completeness can conceal systematic omission in one subgroup.
**Contract 5: Label revision**
- **Definition:** Count cases whose adjudicated value changes between two frozen snapshots.
- **Population and timing:** Record the reason and timestamp of each correction.
- **Diagnostic use:** Recalculate affected cohort metrics under a documented restatement policy.
- **Interpretation boundary:** A correction is not necessarily evidence that the original model changed.
**Contract 6: Score distribution**
- **Definition:** Track score quantiles and fixed-bin counts per model version.
- **Population and timing:** Include exact threshold and fallback-path rates to interpret operational impact.
- **Diagnostic use:** Compare equivalent cohorts and time-of-day patterns.
- **Interpretation boundary:** A changed distribution does not show whether predicted probabilities are calibrated.
**Contract 7: Threshold workload**
- **Definition:** Count decisions routed to review at the active threshold.
- **Population and timing:** Use the complete decision log, including decisions without labels.
- **Diagnostic use:** Compare arrivals with review capacity and backlog age.
- **Interpretation boundary:** A threshold can meet volume limits while missing an unacceptable number of positives.
**Contract 8: False-positive workload**
- **Definition:** Count nonpositive mature outcomes among decisions sent to review.
- **Population and timing:** Disclose that unreviewed or pending outcomes are not observed negatives.
- **Diagnostic use:** Express both absolute workload and fraction of reviewed cases.
- **Interpretation boundary:** Label changes or reviewer selection can alter the apparent count.
**Contract 9: False-negative burden**
- **Definition:** Count adjudicated positive outcomes among decisions not flagged.
- **Population and timing:** The outcome acquisition process must cover unflagged decisions.
- **Diagnostic use:** Compare counts and exposure across cohorts and relevant segments.
- **Interpretation boundary:** Blocked or prevented outcomes need a separate causal design.
**Contract 10: Recall at capacity**
- **Definition:** Rank eligible decisions and examine positives captured within the permitted review budget.
- **Population and timing:** Use a fixed and auditable tie-breaking rule at the cutoff.
- **Diagnostic use:** Compare candidates on the same representative mature cohort.
- **Interpretation boundary:** An offline ranking does not guarantee reviewers can process cases at the assumed rate.
**Contract 11: Probability calibration**
- **Definition:** Compare mean predicted probability with observed outcome rate in defined score bins.
- **Population and timing:** Use representative mature outcomes and show counts and uncertainty per bin.
- **Diagnostic use:** Inspect whether calibration differs by segment and calendar time.
- **Interpretation boundary:** Binning hides within-bin structure and rare-outcome uncertainty.
**Contract 12: Ranking discrimination**
- **Definition:** Compare whether positives tend to receive higher scores than negatives.
- **Population and timing:** Use a metric consistent with the decision and prevalence regime.
- **Diagnostic use:** Compare models on the same cohort with dependence-aware uncertainty.
- **Interpretation boundary:** Good ranking alone does not imply accurate probabilities or sufficient capacity.
**Contract 13: Feature freshness**
- **Definition:** Measure age from source event time to the moment the feature is consumed.
- **Population and timing:** Distinguish an old valid value from a failed refresh under the feature contract.
- **Diagnostic use:** Report access-weighted freshness if users experience different records.
- **Interpretation boundary:** A fresh ingestion timestamp can conceal stale source events.
**Contract 14: Feature missingness**
- **Definition:** Measure the fraction of eligible decisions lacking each required feature.
- **Population and timing:** Separate null, absent field, invalid value, and default-imputed value.
- **Diagnostic use:** Break down by producer, deployment, and input segment.
- **Interpretation boundary:** A missingness increase is not necessarily caused by model weights.
**Contract 15: Feature range**
- **Definition:** Measure out-of-contract values before and after preprocessing.
- **Population and timing:** Specify physical or business units and valid bounds.
- **Diagnostic use:** Inspect raw source values before clipping or normalization.
- **Interpretation boundary:** Clipping may hide an upstream unit conversion defect.
**Contract 16: Unknown categories**
- **Definition:** Count categories not present in the deployed encoder vocabulary.
- **Population and timing:** Include the vocabulary version and the unknown-value handling rule.
- **Diagnostic use:** Compare the source catalog and deployment timelines.
- **Interpretation boundary:** An allowed new category can be legitimate even when it is unfamiliar to the model.
**Contract 17: Training-serving skew**
- **Definition:** Compare the point-in-time feature value used in serving with a reconstructed training feature.
- **Population and timing:** Align entity key, event time, transformation version, and null handling.
- **Diagnostic use:** Sample discrepancies and trace each to a source or transformation.
- **Interpretation boundary:** A reconstruction using future data is not a valid parity check.
**Contract 18: Input drift**
- **Definition:** Compare current and reference feature distributions using fixed definitions.
- **Population and timing:** Choose a comparator appropriate to season, population, and product version.
- **Diagnostic use:** Report sample sizes, effect size and localized bins or quantiles.
- **Interpretation boundary:** Marginal tests cannot establish unchanged interactions or outcomes.
**Contract 19: Output drift**
- **Definition:** Compare decision or score distributions across comparable cohorts.
- **Population and timing:** Annotate threshold, model, policy, and traffic routing changes.
- **Diagnostic use:** Investigate input changes and pipeline version changes separately.
- **Interpretation boundary:** Identical outputs can still have different actual outcomes.
**Contract 20: Latency**
- **Definition:** Measure a user-relevant elapsed time from a declared start to completion or failure.
- **Population and timing:** Include timeouts according to the SLI contract rather than dropping them.
- **Diagnostic use:** Report tail thresholds and request counts by path.
- **Interpretation boundary:** An application-server timer may miss failures before the server receives a request.
**Contract 21: Availability**
- **Definition:** Divide good eligible requests by all eligible requests over the agreed window.
- **Population and timing:** Define which status codes, fallbacks, and timeouts satisfy the good-event rule.
- **Diagnostic use:** Reconcile client-side and server-side counts for coverage gaps.
- **Interpretation boundary:** A successful transport response may contain a wrong answer.
**Contract 22: Agent completion**
- **Definition:** Count attempts that reach the declared terminal completion state.
- **Population and timing:** Keep cancellations, timeouts, rejected requests, and aborted runs visible.
- **Diagnostic use:** Inspect completion separately by task type and tool dependency.
- **Interpretation boundary:** A terminal state is not a factual or safety evaluation.
**Contract 23: Agent task success**
- **Definition:** Apply a versioned rubric to eligible attempted tasks.
- **Population and timing:** Use an independent representative review sample and disclose judge limitations.
- **Diagnostic use:** Report success per attempt and, separately, success among completed runs.
- **Interpretation boundary:** Do not alter the rubric after observing results without recording the change.
**Contract 24: Tool reliability**
- **Definition:** Track attempted calls, transport failures, semantic errors, and response times.
- **Population and timing:** Distinguish tool execution failure from invalid arguments generated by an agent.
- **Diagnostic use:** Inspect failures by tool version and dependency.
- **Interpretation boundary:** A valid JSON response can still return incorrect data.
**Contract 25: Retrieval relevance**
- **Definition:** Measure whether retrieved passages are relevant to the task under a declared annotation rule.
- **Population and timing:** Ensure assessors can see the query and the retrieved item without outcome leakage.
- **Diagnostic use:** Break down failures by corpus freshness and access permissions.
- **Interpretation boundary:** Relevant retrieval alone does not guarantee a supported final answer.
**Contract 26: Citation support**
- **Definition:** Assess whether each cited source actually supports the associated claim.
- **Population and timing:** Separate citation presence from entailment and authorized access.
- **Diagnostic use:** Audit unsupported claims and missing citations in a representative sample.
- **Interpretation boundary:** A reference to an accessible document is not itself evidence for every statement.
**Contract 27: Review agreement**
- **Definition:** Compare independently assigned ratings on the same items under the same rubric.
- **Population and timing:** Record rater qualifications, item sample, and adjudication protocol.
- **Diagnostic use:** Inspect disagreement by failure category rather than relying only on one aggregate.
- **Interpretation boundary:** Agreement can be high even when both raters apply a defective rubric.
**Contract 28: Cost per success**
- **Definition:** Divide included total costs by successfully completed tasks.
- **Population and timing:** Include model, tool, infrastructure, and review categories if they are within scope.
- **Diagnostic use:** Report cost per attempt and success rate separately.
- **Interpretation boundary:** The ratio is undefined when the success count is zero.
**Contract 29: Safety incident rate**
- **Definition:** Count policy-defined harmful events among eligible exposures.
- **Population and timing:** Record severity, ascertainment method, and report delay.
- **Diagnostic use:** Escalate severe individual events even when population rates are uncertain.
- **Interpretation boundary:** An incident detector is not a preventive authorization boundary.
**Contract 30: Backlog age**
- **Definition:** Measure time from review enqueue to completed review for each case.
- **Population and timing:** Separate waiting time from handling time and stalled cases.
- **Diagnostic use:** Track oldest outstanding work and service capacity by priority.
- **Interpretation boundary:** A small queue can still contain a dangerously old high-priority case.
## 9. Complexity & Cost Analysis
### 9.1 Data and aggregation costs
Let N denote predictions, F monitored features, B fixed histogram bins, and S explicitly materialized segments.
A straightforward pass over every feature for every prediction performs order N times F feature visits.
Histogram state for all feature-segment-bin combinations can grow on the order of F times S times B counters.
Those bounds assume dense materialization; sparse representations change actual state and overhead.
An exact quantile generally requires storing observations or an external sort; approximate sketches trade accuracy for bounded state.
A hash join of prediction and outcome records can be expected linear in input size under suitable hashing and memory assumptions.
The actual join can spill, shuffle, or become skewed when memory or key distribution is unfavorable.
Entity-level ordering and as-of label selection can add sort or indexing costs.
Sampling reduces review cost but introduces sampling error and potentially selection bias.
Never hide sampling probabilities if they are needed to estimate a population quantity.
### 9.2 Illustrative capacity estimate
**ILLUSTRATIVE:** Suppose 100,000 predictions per day each carry 20 monitored scalar features.
A full daily feature pass therefore visits 100,000 times 20 = 2,000,000 feature values.
Suppose ten bins and five predefined segments are materialized for each feature.
The dense histogram requires 20 times 10 times 5 = 1,000 counters.
At an assumed eight bytes per raw counter, the counter payload is 8,000 bytes.
This excludes metadata, keys, indexing, replication, runtime overhead and serialization.
If the segments overlap, one prediction can update multiple segment counters; the pass cost also reflects that fanout.
If the feature count doubles while the other assumptions stay fixed, dense counters double to 2,000.
This is a storage arithmetic example, not a throughput benchmark or a cloud cost quotation.
### 9.3 Human review and operational economics
If 200 randomly selected cases need an average of five review minutes, planned review time is 1,000 minutes.
That is 16 hours and 40 minutes before adjudication, calibration, handoffs and breaks.
If 20% need a second five-minute review, add 40 times five = 200 minutes.
Total planned review effort is then 1,200 minutes, or 20 person-hours under the stated assumptions.
The additional second reviews must be budgeted independently of the sample selection design.
Report total cost per attempted task and cost per successful task with included cost categories.
When no successful tasks occur, cost per success is undefined; do not substitute zero.
Protect sensitive traces with access controls, retention limits, and redaction consistent with policy.
Instrument enough context to diagnose failures without logging unnecessary personal data or secrets.
**Takeaway:** Budget feature computation, retained state, label latency, human review, and privacy controls together.
## 10. Key Theorems / Results Reference
### 10.1 Conditional decomposition of overall performance
A population rate can be decomposed as a weighted average of its mutually exclusive, exhaustive segment rates.
For segment weights w_i and segment rates r_i, the overall rate equals the sum of w_i times r_i.
The identity follows by partitioning the numerator and denominator into the same segments.
**ILLUSTRATIVE:** Two segments have weights 0.8 and 0.2 and success rates 0.9 and 0.5.
The overall success rate is 0.8 times 0.9 plus 0.2 times 0.5, or 0.82.
If weights change to 0.5 and 0.5 with unchanged segment rates, the overall rate becomes 0.70.
The 12-percentage-point overall change is attributable to the assumed weight change in this constructed example.
In real data, both rates and weights may change; decomposition alone does not establish why.
### 10.2 Selection and identification
The labeled-case rate equals the target-population rate only under suitable assumptions about label observation.
For example, independent label observation within a fixed score stratum supports using that stratum's labeled rate for that stratum.
If labeling depends on the unobserved outcome even within the stratum, the observed rate may be biased.
Known nonzero sampling probabilities can support weighted estimates, subject to correct design and uncertainty accounting.
Zero sampling probability in a relevant stratum prevents design-based recovery of that stratum from the observed labels alone.
Randomized audits and independent adjudication can improve identification but have cost and ethical constraints.
### 10.3 Classification metrics and prevalence
Precision depends on the balance of true and false positives, not only on sensitivity.
**ILLUSTRATIVE:** In 10,000 cases, assume 100 positives, sensitivity 80%, and false-positive rate 2%.
There are 80 true positives and 0.02 times 9,900 = 198 false positives.
Precision is 80/(80+198) = 80/278, approximately 28.78%.
If prevalence doubles to 200 positives with the same sensitivity and false-positive rate, true positives become 160.
False positives become 0.02 times 9,800 = 196, so precision is 160/356, approximately 44.94%.
This assumes unchanged conditional error rates; it does not prove the cause of any observed change.
### 10.4 Statistical inference boundaries
Confidence intervals describe uncertainty under their sampling assumptions, not the probability that an already fixed parameter is random.
Repeated peeking and selective reporting can invalidate a nominal fixed-horizon test's error guarantees.
Time dependence, repeated decisions per customer and cluster assignment affect effective sample information.
Use a method matched to the actual sampling and assignment design rather than assuming all events are independent.
A correlation between a feature shift and quality decline does not isolate the cause of either.
Safety-critical failures can justify immediate containment without waiting for a conventional significance threshold.
The [illustrative drift incident](../../../11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/01_ml_system_design_production/Case_Studies/01_model_monitoring_drift_incident.md) applies these distinctions to containment and recovery; its outcomes are not measured.
**Takeaway:** Decomposition, selection assumptions, prevalence, and dependence govern what monitoring evidence can support.
## 11. Interview / Exam Questions
### 11.1 What is the difference between monitoring and evaluation?
**Answer:** Monitoring continuously observes production behavior; evaluation scores performance under an explicit task, population, and protocol.
**Reasoning:** A production evaluation can be scheduled from monitoring data, so online versus offline is not the distinction.
**Follow-up:** What if the service is available but wrong?
**Follow-up answer:** Use a task-quality evaluation and trace the failure; availability alone does not measure correctness.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: declaring uptime to be model accuracy.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.2 How would you evaluate a model with a 30-day outcome?
**Answer:** Construct a decision-time cohort, wait for the outcome horizon and a declared reporting buffer, then join outcomes as of a fixed cutoff.
**Reasoning:** Publish coverage and analyze missingness by score, action, and segment; late or revised labels remain possible.
**Follow-up:** Can pending outcomes be counted as negatives?
**Follow-up answer:** No; they are unknown unless the outcome definition and ascertainment justify an explicit negative.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: computing recall using only cases that were reviewed.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.3 A feature drift test rejects its null. Must you retrain?
**Answer:** No; quantify the shift, check feature contracts and traffic mix, and evaluate mature quality before selecting an intervention.
**Reasoning:** Statistical significance depends on sample size and does not establish harm or a causal mechanism.
**Follow-up:** What if units changed from dollars to cents?
**Follow-up answer:** Repair the units and confirm train-serving parity; do not retrain on corrupted data.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: treating a drift score as model accuracy.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.4 Calculate precision and recall for TP 60, FP 140, FN 40.
**Answer:** Precision is 60/200 = 30%; recall is 60/100 = 60%.
**Reasoning:** The two denominators are predicted positives and actual positives, respectively.
**Follow-up:** What if labels exist only for flagged items?
**Follow-up answer:** Precision on flagged labeled items may be estimable subject to selection; population recall is not identified.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: using the number of all predictions as the recall denominator.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.5 Why can precision drop even if sensitivity and false-positive rate stay fixed?
**Answer:** The prevalence of positives can fall, changing the ratio of true to false positives.
**Reasoning:** Section 10 gives a checked example in which the same conditional rates yield different precision.
**Follow-up:** Is this necessarily concept drift?
**Follow-up answer:** No; prevalence, traffic composition, labeling, and conditional relationships need separate investigation.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: inferring a changed model from precision alone.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.6 How can accuracy remain stable while a subgroup is harmed?
**Answer:** Aggregate accuracy is a weighted average; improvements elsewhere or a low subgroup weight can hide deterioration.
**Reasoning:** Compare consistent slice definitions, denominators, and uncertainty rather than only the aggregate.
**Follow-up:** What if the subgroup has three cases?
**Follow-up answer:** Report the count and uncertainty; do not claim reliable evidence of parity or safety.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: reporting a zero error rate without the denominator.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.7 How would you design a safe canary?
**Answer:** Assign a limited, representative eligible cohort, record versions, and prespecify guardrails, ownership, and rollback.
**Reasoning:** Monitor operational failures and quality separately, while acknowledging rare harms may be missed.
**Follow-up:** If only latency improves, promote?
**Follow-up answer:** Not on that evidence alone; verify task success and critical safety requirements.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: assuming an unrepresentative canary proves safety.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.8 How do you distinguish score drift from feature drift?
**Answer:** Feature drift changes the distribution of model inputs; score drift changes the distribution of outputs.
**Reasoning:** Either can occur without observed harm and either can be caused by version or traffic changes.
**Follow-up:** Can stable scores prove stable recall?
**Follow-up answer:** No; the relationship between scores and outcomes may have changed.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: equating a score histogram with a confusion matrix.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.9 How should an agent success metric handle unfinished runs?
**Answer:** Define the evaluation unit and rule in advance; if incomplete runs are failures, success per attempt includes them in the denominator.
**Reasoning:** Also report completion and conditional quality to distinguish the stages.
**Follow-up:** For 140 successes among 200 attempts and 160 completions, what are the rates?
**Follow-up answer:** Success per attempt is 70%; completion is 80%; conditional success is 87.5% if successes are completed runs.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: using only successes divided by completions without disclosing excluded attempts.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.10 What makes a good production alert?
**Answer:** It corresponds to an actionable user or system risk, has a defined window and denominator, and routes to an owner with a response.
**Reasoning:** Check low-volume behavior, telemetry failures, relevant baselines, and repeated testing.
**Follow-up:** Would you suppress all low-count safety alerts?
**Follow-up answer:** No; a single critical event may require immediate containment even when rate estimates are unstable.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: treating missing telemetry as a healthy zero.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.11 A high-risk action changes the observed outcome. What is the problem?
**Answer:** The observed outcome is conditional on the action taken and may not reveal the untreated counterfactual.
**Reasoning:** Selective review and prevented events can bias naive labels and comparisons.
**Follow-up:** Can an experiment always fix it?
**Follow-up answer:** No; randomization may be unsafe or infeasible, and identification still depends on design and adherence.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: claiming prevented fraud is directly observed for each blocked case.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
### 11.12 How would you compare a new model with a baseline?
**Answer:** Use the same point-in-time eligible inputs, protected holdout and task definitions; compare paired metrics and relevant slices.
**Reasoning:** Report uncertainty, label maturity, review cost, operational guardrails and a staged rollback plan.
**Follow-up:** What if the candidate improves AUC but overloads reviewers?
**Follow-up answer:** Evaluate feasible thresholds and workload; an AUC gain alone does not satisfy capacity.
**Acceptable alternative:** A more conservative response is acceptable when it states its assumptions and evidence needs.
**Red flag:** Red flag: selecting a model on a repeatedly tuned final test set.
**Scoring:** Credit requires a correct answer and the relevant population, timing, or validity limitation.
**Support:** See the worked examples and diagnostic cases in sections 4–8.
**Takeaway:** Strong answers identify the estimand and evidence boundary before choosing an operational action.
## 12. Quick Reference Card
| Item | Value |
|---|---|
| Topic | Production model evaluation, observability and response |
| Core formula | Precision = TP / (TP + FP); recall = TP / (TP + FN), when the corresponding denominators are positive |
| Key assumption | Labels, eligibility, maturity and selection represent the target population for the reported metric |
| Complexity | Monitoring cost scales with events, retained dimensions, model evaluation and label joins; Section 9 states the workload assumptions |
| Use when | Choosing signals and actions for deployed ML or agent systems |
| Avoid when | Feature shift, task completion or partial label coverage is presented as proof of outcome quality |
| Go deeper | [Collection index](INDEX.md); evaluate data and response contracts against the relevant deployed system |
| Diagnostic | Compare label coverage, feature provenance, slice quality, baseline and operational capacity before intervention |
| Evidence | Worked results are illustrative; real performance and deployment outcomes are not measured |
