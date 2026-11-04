# Benchmark Contamination and Leakage

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: none; forecasting and evaluation concepts are introduced here  
> Version 2026.5

## 1. Why this topic exists

A planning director is choosing whether to replace a reorder forecast used for replenishment.
A vendor presents a strong average score on a public forecasting benchmark.
The director must decide whether that result predicts improvement on the company's own future decisions.
The public series may have been downloaded before model training.
They may also have appeared in copies, derived datasets, research repositories, or synthetic mixtures.
A model that learned those series can score well without demonstrating the same advantage on an unseen business.

The issue is not confined to foundation models.
A conventional forecasting pipeline can leak future outcomes through features, preprocessing, or model selection.
A language model can encounter benchmark questions, answers, documents, or near-duplicates during training.
A retrieval system can expose the answer through a document that was created after the decision date.
These mechanisms are different, but all violate a common information boundary.

A genuine evaluation asks what would have happened if a specified system had made a specified decision with only information available at that time.
That question requires a prediction origin, a versioned model, and a record of what the system could access.
It also requires an untouched evaluation period and a comparison that gives alternatives the same information.
A score without those details is a descriptive statistic, not a reliable deployment case.

A buyer also needs to separate three questions.
First, was the evaluation information independent enough for the claim being made?
Second, does the measured advantage transfer to the buyer's products, locations, lead times, and decisions?
Third, does the advantage justify its full cost and operational risk?
A clean test answers the first question better than a public leaderboard does, but it does not automatically answer the other two.

An error can have asymmetric consequences.
A low forecast may cause lost sales or an emergency shipment.
A high forecast may create spoilage or working-capital cost.
A claimed reduction in average forecast error is therefore not necessarily a reduction in total inventory cost.
The correct business comparison includes the replenishment policy and service constraints.

The cost of an optimistic result is not merely an inaccurate chart.
It can lead to a multi-year contract, integration work, changed safety stock, and a loss of planner trust.
A slow, well-controlled decision may be preferable to a fast rollout based on a contaminated score.
This unit gives you a practical way to make that decision without pretending that every unknown is proof of cheating.

### Decision owners

| Owner | Decision | Evidence needed |
|---|---|---|
| Planning director | Whether to run or expand a pilot | Local uplift, uncertainty, and impact on service |
| Data steward | Which data may enter the test | Provenance, access rights, and retention rules |
| Model owner | Which version to evaluate | Immutable version, cutoffs, and update policy |
| Evaluator | Whether comparison is valid | Point-in-time inputs, untouched outcomes, and baseline parity |
| Procurement | Whether to contract | Transfer evidence, total cost, and audit rights |
| Operations | Whether to act on outputs | Exception process, human authority, and rollback |

> **Note:** A public benchmark can still be useful for engineering screening.
> Its value as evidence of performance on genuinely unseen data depends on its history and the claim.

## 2. Learning outcomes

After this unit you can:

1. Draw the information boundary at a forecast origin and identify which records cross it.
   Exercise 1 tests the boundary using actual availability times.
2. Distinguish confirmed exposure, suspected overlap, and unknown provenance.
   Exercise 2 asks you to classify evidence without overclaiming.
3. Design a point-in-time rolling-origin evaluation with a locked model and baseline.
   Exercise 3 asks for an executable evaluation protocol.
4. Compute forecast errors, a paired uplift, and a simplified business-cost comparison.
   Exercise 4 and the runnable example check the arithmetic.
5. Explain why an aggregate public score may not transfer across products, horizons, or regimes.
   Exercise 5 asks you to critique a transfer claim.
6. Specify controls for model updates, shared information, and repeated benchmark use.
   Exercise 6 asks you to design a governance response.
7. Decide whether evidence supports rejection, a bounded pilot, or deployment.
   Exercise 7 asks for a decision memo.
8. Audit a document-extraction evaluation for leakage and retrieval timing.
   Exercise 8 extends the same boundary to language models.

Observable success means another reviewer can reconstruct the dates, inputs, exclusions, metrics, and decision from your artifacts.
It does not require access to a vendor's proprietary model weights.
It does require that an unknown training corpus be recorded as unknown rather than certified clean.

## 3. The core mental model

### An information boundary, not a purity label

```text
                    model development time
 public corpora ──> pretraining ──> frozen model version ──────────┐
 benchmark copies ────────?───────────────┘                        │
                                                                  v
 known by origin ──> feature snapshot ──> forecast at origin ──> decision
                           ^                  |                       |
                           |                  v                       v
 archived source    point-in-time audit    predictions          observed outcome
                           |                  |                       |
                           └──── evidence ────┴──── paired scoring ───┘
                                                  ^
                                         locked evaluation rules

 Forbidden route: future outcomes / future revisions / evaluation answers
                  ─────────────────────> input, fitting, selection, or retrieval
```

The vertical boundary is the forecast origin: the time when a forecast is issued.
The system is allowed to use the information actually accessible by that origin.
It is not allowed to use information learned from the later outcome it is supposed to predict.
The phrase “actually accessible” matters because event time and data availability time are not always the same.
A sales record for Monday may be reconciled and posted on Wednesday.
A Tuesday forecast cannot use Wednesday's final revision merely because the event happened on Monday.

The model itself is an input to the decision.
Its pretraining corpus, tuning history, released version, and update date can transmit information across the boundary.
A model released after a historical origin may have seen the future relative to that origin.
An offline backtest using that model is therefore not a faithful historical deployment simulation unless its exposure is ruled out or the claim is qualified.
It can still be a useful retrospective experiment, but its evidence is weaker for a prospective deployment claim.

A baseline also has an information boundary.
If the proposed model sees a promotion schedule but the baseline does not, the observed difference combines model and information advantages.
If the baseline is refit using the test outcomes while the proposed model is not, the comparison is unfair in the opposite direction.
Both systems should have the same legitimate information or the experiment should explicitly attribute the difference to the information bundle.

### Four separate questions

| Question | Example | Appropriate conclusion |
|---|---|---|
| Exposure | Was the benchmark included in pretraining? | A provenance or overlap claim |
| Leakage | Was future or held-out information used across a prohibited boundary? | A claim about evaluation validity |
| Memorization | Did the system reproduce particular training content? | A behavioral claim requiring evidence |
| Transfer | Does performance persist on the target population and decision? | A transportability claim requiring local evidence |

Exposure does not prove a particular answer was memorized.
A model may have seen a series and still need to generalize to an unseen continuation.
Conversely, absence of an exact duplicate does not establish independence if a transformed copy, future aggregate, or repeated selection supplied the answer indirectly.
The boundary and the intended claim determine the severity of the finding.

### From forecast to decision

A forecast is an estimate of future demand, not an order.
A plan is a proposed operational response using the forecast and other constraints.
A target is a desired result, such as 95% fill rate.
A commitment is an authorized promise or order to a supplier.
The example later in this unit compares forecasts and a simplified cost proxy; it does not claim that either forecast itself is a purchase commitment.

## 4. Definitions and vocabulary

| Term | Meaning in this unit | Common misuse or unit |
|---|---|---|
| Forecast origin | Timestamp at which a prediction is issued | Confusing it with the target period |
| Target period | Future interval for the quantity predicted | Treating a weekly bucket as a timestamp |
| Horizon | Time between origin and target | Comparing one-week and twelve-week forecasts as equivalent |
| Event time | When an operational event happened | Assuming the record was already visible |
| Availability time | When the evaluator's system could first access a record | Replacing it with a later extract timestamp |
| Revision time | When a value was corrected or updated | Using final values as historical features |
| As-of join | Join selecting records available by the origin | Joining to the latest row without a cutoff |
| Snapshot | Preserved state of a dataset or system at a time | A recent export mislabeled as historical |
| Pretraining | Initial model fitting on a broad corpus | Assuming public data implies independent evaluation |
| Fine-tuning | Further parameter fitting on a specific dataset | Calling it zero-shot on that dataset |
| Zero-shot | Use without task-specific fitting under a stated protocol | Claiming no prior exposure or no prompts |
| Few-shot | Use with a small number of task examples, usually in context | Treating demonstrations as independent test items |
| Benchmark | Defined dataset and scoring protocol for comparison | Equating public rank with buyer value |
| Contamination | Evaluation information entering development or inputs in a way that weakens a claim | Treating every shared pattern as an exact duplicate |
| Leakage | Information unavailable under the intended decision protocol influencing prediction or selection | Assuming all correlation is leakage |
| Exact overlap | Identical content or sequence in two sources | Proof that a model memorized the item |
| Near-duplicate | Material substantially derived from another item | A simple equality match |
| Provenance | Recorded origin, transformations, and custody of data or model | A source name without versions or dates |
| Training cutoff | Claimed latest data date used in model development | Proof that all source timestamps are correct |
| Knowledge cutoff | Stated information boundary for a released model | A verified list of training examples |
| Frozen model | A version held constant through evaluation | A hosted alias that silently changes |
| Holdout | Data reserved from fitting and selection for final assessment | Data repeatedly inspected during development |
| Validation set | Data used to choose a model or setting | A final independent test |
| Rolling origin | Evaluation over several forecast issue times | Randomly shuffling time-series rows |
| Gap or embargo | Excluded interval around a split to limit overlap | A universal number of periods that cures all leakage |
| Training window | Past data available for fitting at a given origin | The whole dataset after future rows are known |
| Context window | Past observations supplied to a model at inference | Proof that no future information entered weights |
| Ground truth | Reference outcome used for scoring | A final revised number assumed available at origin |
| MAE | Mean absolute error, in the target's units | A percentage or inventory cost |
| WAPE | Sum of absolute errors divided by sum of absolute actuals | A stable metric when total actual demand is zero |
| Bias | Mean signed error under a specified sign convention | Absolute error |
| Paired difference | Difference between model losses on the same evaluation units | Comparing unpaired averages across different samples |
| Cluster | Group of dependent observations, such as a product or week | An independent forecast observation |
| Distribution shift | Change in the relationship or population between evaluation and use | Proof of contamination |
| Label | The observed outcome or correct answer for an evaluation item | A feature that is safe to provide at prediction time |
| Retrieval-augmented generation (RAG) | Model generation using retrieved documents as context | Automatic proof that cited content is correct |
| Data lineage | Trace of a record through sources and transformations | A substitute for contractual permission |
| Shadow mode | Producing predictions without using them for live decisions | Evidence that live intervention effects are known |
| Service level | A specified measure of customer or inventory service | A generic synonym for forecast accuracy |

A label can be used after the outcome is observed for scoring.
The same label is prohibited as an input before that outcome is observable.
A calendar holiday known years ahead can be a legitimate future feature.
The future realized sales quantity cannot.
The question is availability at the decision time, not simply whether a feature describes a future period.

## 5. The mechanics, step by step

### Step 1. Specify the business decision and the claim

**Input:** the proposed operational use, decision frequency, and consequential error types.

**Procedure:**

1. Name the actor and the action, such as a replenishment planner proposing weekly purchase orders.
2. Define the product-location population and the unit of forecast, such as cases per store per week.
3. Define the forecast issue time, horizon, and lead time separately.
4. Identify the business objective, including service and cost constraints.
5. State whether the claim is about accuracy, cost, service, implementation speed, or all four.
6. State what counts as a deployment decision versus permission to investigate.

**Output:** a one-page decision charter with a primary estimand.
An estimand is the precise quantity the experiment aims to estimate.
For example, it may be the mean difference in absolute error per eligible product-week across a specified future window.
That is not the same estimand as percentage reduction in total company inventory cost.

**Branch:** if outcomes are censored by stockouts, recorded sales may understate demand.
A demand forecast evaluated against sales alone can reward a forecast that mirrors constrained supply.
Record the limitation, use an agreed demand proxy when defensible, and report stockout segments separately.
Do not relabel sales as true unconstrained demand merely to complete a scorecard.

### Step 2. Define the information boundary

**Input:** the intended production workflow and its timestamps.

**Procedure:**

1. Fix the forecast origin to a date and time including time zone.
2. List every feature, its source, event time, publication time, and revision history.
3. Determine whether the feature was actually accessible to the production role by that origin.
4. Record the allowable data latency, such as a two-day reporting delay.
5. Define the treatment of missing values without looking at the later outcome.
6. Define what future-known variables are genuinely committed and available at that point.
7. Record the model artifact and any retrieval index or configuration as inputs.

A promotion calendar is not a known future covariate if the schedule was approved only after the forecast origin.
A later cancellation must not be inserted into an earlier snapshot unless the cancellation was already known then.
A supplier lead-time forecast is legitimate if available at the origin; the actual arrival date is not.

**Output:** an availability matrix with a yes/no/unknown determination for each field.

**Branch:** when a historical snapshot is missing, distinguish an auditable reconstruction from a convenient guess.
If a field's historical availability cannot be reconstructed, exclude it or label that part of the test retrospective and non-decisive.
Do not assert a clean backtest based on a present-day warehouse extract alone.

### Step 3. Map all development exposure

**Input:** model card, vendor statements, training sources, and internal development records.

**Procedure:**

1. List pretraining datasets and their versions when disclosed.
2. Record the collection dates, publication dates, and time span of each disclosed source.
3. List subsequent supervised fine-tuning, preference tuning, adapters, prompts, and retrieval sources.
4. Identify the benchmark and all public copies or derived versions that can be traced.
5. Compare benchmark identifiers, value sequences, timestamps, and metadata against accessible training inventories.
6. Ask whether data were deduplicated, at what granularity, and with which matching method.
7. Record the status as confirmed overlap, credible suspicion, no overlap found within stated search coverage, or unknown.
8. Preserve evidence and the scope of each search.

Exact hashes detect byte-identical files but miss a copied series with different formatting.
Sequence fingerprints, normalized timestamps, or similarity matching can find some transformed copies.
Near-duplicate thresholds create false positives and false negatives.
A match should be reviewed against the content and transformation, not accepted automatically as proof of model memorization.

**Output:** a provenance register and an exposure matrix.

**Branch:** a closed training inventory prevents comprehensive overlap verification.
The reviewer can still test behavior and secure a prospective holdout.
Neither a negative behavioral probe nor a confidentiality agreement establishes complete non-exposure.
Request an independently audited attestation if feasible, and qualify residual uncertainty.

### Step 4. Separate kinds of exposure

**Input:** the overlap register and the intended evaluation claim.

**Procedure:**

1. Identify whether shared material is the exact target series, a prefix, a future suffix, or a related series.
2. Determine whether the exact target labels were available during model development.
3. Determine whether only the evaluation instructions, question format, or public leaderboard were exposed.
4. Identify access by humans who selected prompts, hyperparameters, and model versions.
5. Identify external tools that can retrieve public benchmark answers at inference.
6. Assess whether the exposure was allowed by the declared protocol.

An observed historical prefix supplied as context is normally legitimate in forecasting.
A target-period suffix embedded in pretraining can undermine a historical out-of-sample claim.
A public description of the task can guide modeling without necessarily exposing any specific target.
Repeatedly optimizing to a benchmark's reported scores can nevertheless overfit the benchmark even without direct label access.

**Output:** an itemized mechanism and a claim-level severity, not a binary accusation.

**Branch:** if the same data are used to build a general representation but an unseen continuation is scored, disclose that distinction.
Do not describe the result as proof of performance on entirely unseen series.
Do not automatically discard it for questions where prior exposure is part of the intended operating environment.

### Step 5. Freeze the test protocol

**Input:** decision charter, provenance register, and candidate systems.

**Procedure:**

1. Register primary and secondary metrics before revealing test outcomes.
2. Fix eligible series, product-location mapping, forecast origins, horizons, and exclusion rules.
3. Fix the baseline, its tuning budget, and access to legitimate covariates.
4. Fix the model version, prompts, inference settings, and allowed context length.
5. Specify the outcome vintage and its reconciliation rule.
6. Define aggregation and treatment of missing predictions before scoring.
7. State how repeated or multiple comparisons will be handled.
8. Assign an evaluator who controls the holdout and an owner who approves deviations.

**Output:** a dated protocol, code version, and signed-off evaluation manifest.

A missing forecast cannot simply disappear from the score when it is difficult.
A reasonable protocol may assign an operational fallback and report both coverage and fallback-inclusive performance.
The fallback must be chosen in advance and available in the real workflow.

**Branch:** if the experiment is exploratory, allow iteration on a development or validation period.
Reserve a later untouched period to evaluate the selected procedure.
If that reserve has also been used for selection, create a new holdout and relabel the prior result as validation.

### Step 6. Build point-in-time training and feature snapshots

**Input:** source histories, availability timestamps, and the frozen protocol.

**Procedure:**

1. For each origin, select only source records with availability time at or before that origin.
2. Resolve revisions using the latest version available at that origin, not the latest version known today.
3. Fit scaling, imputation, feature selection, and encoders using the training portion only.
4. Generate lags and rolling statistics ending before the forecast target.
5. Validate join cardinality so later records are not duplicated into earlier observations.
6. Save immutable snapshots or reproducible queries and versioned source manifests.
7. Trace representative rows back to original system records.

A full-series mean used to normalize all rows leaks target-period information.
A rolling average including the target week leaks the outcome into its own feature.
An inventory status recorded after a store's weekly reconciliation can leak a stockout that was not known at the forecast origin.
Each defect can make a backtest appear more accurate than the production workflow can be.

**Output:** per-origin feature snapshots and a lineage report.

**Branch:** if an availability timestamp is unreliable, investigate the source system's publishing and revision behavior.
An event timestamp alone cannot establish that a record was usable.
Record unresolved fields as unknown and evaluate without them where possible.

### Step 7. Prevent overlap between windows

**Input:** training examples, forecast origins, and prediction horizon.

**Procedure:**

1. Draw the input and label interval for every training and test example.
2. Ensure each training label was observable by the relevant training cutoff.
3. Remove or delay examples whose labels extend into the test period.
4. Add a gap when overlapping label windows or publication delays require one.
5. Keep dependent copies of the same product-event or document family in the appropriate split.
6. Document why the gap is sufficient for the particular label and process.

A fixed gap is not a magical contamination detector.
If pretraining already included the test suffix, a gap in local fine-tuning cannot remove knowledge from the model weights.
If weekly observations overlap across a four-week target, an embargo shorter than the outcome window may be inadequate.
If labels are instantly observed and nonoverlapping, an arbitrary long gap may waste relevant data without addressing a real leakage path.

**Output:** a split diagram and a rule for overlapping labels.

**Branch:** for a global model trained across many series, inspect cross-series timing.
A related aggregate containing another series' future target may leak into the target series.
Split by time consistently where the business claim requires future prediction, and by entity where the claim requires new-product or new-customer generalization.

### Step 8. Run rolling-origin forecasts

**Input:** frozen candidates, per-origin snapshots, and a schedule of origins.

**Procedure:**

1. At each origin, fit or update only as the production process would have done.
2. Supply each candidate the permitted historical context and authorized future-known features.
3. Save predictions before outcome access.
4. Score each forecast against the predeclared outcome vintage after it becomes observable.
5. Repeat across origins, including ordinary periods and relevant disruptions when available.
6. Log inference failures, latency, fallback use, and version changes.
7. Preserve prediction-level outputs, not only an overall mean.

**Output:** a prediction ledger keyed by model, version, series, origin, horizon, and target period.

**Branch:** if the model is a hosted service that updates silently, request version pinning or snapshot attestations.
If neither is possible, the evaluated service is a moving process rather than a fixed artifact.
Require a prospective observation period and change notification; do not attribute later behavior to the earlier version.

### Step 9. Score on paired and decision-relevant quantities

**Input:** the prediction ledger and outcome table.

**Procedure:**

1. Pair candidate and baseline predictions on the same eligible series, origin, horizon, and target.
2. Calculate the predeclared primary metric, such as mean absolute error (MAE).
3. Report coverage and the number of eligible and scored observations.
4. Report signed bias and underforecast and overforecast counts or quantities.
5. Segment by horizon, demand volume, product group, site, and disruption regime.
6. Map predictions through an agreed decision rule to estimate business effects where data permit.
7. Calculate uncertainty using a method that respects dependence among related observations.

Absolute error is `|forecast - actual|` in the target unit.
MAE is the average of those errors over the scored observations.
Weighted absolute percentage error (WAPE) divides their sum by the sum of actual demand, provided that denominator is positive.
Neither metric alone measures service level, inventory holding cost, or lost sales.

A paired comparison uses the same cases for both models.
If a model fails on difficult cases, a complete-case comparison can favor it unfairly.
Report common-case performance, coverage, and the predeclared fallback-inclusive result.

**Output:** an auditable scorecard with denominators and segment sizes.

**Branch:** if the observed uplift is smaller than uncertainty or a business threshold, do not translate the point estimate into a guaranteed saving.
Use a larger or prospective evaluation, or reject the business case if further evidence is not worth its cost.

### Step 10. Examine uncertainty and dependence

**Input:** paired loss differences and a definition of independent decision clusters.

**Procedure:**

1. Define the paired loss difference as baseline loss minus candidate loss.
2. Identify shared shocks, repeated forecasts, common stores, and correlated products.
3. Choose a resampling or statistical procedure consistent with the dependence structure.
4. Report an interval for the comparison as well as its point estimate.
5. Describe the number and diversity of independent clusters.
6. Review the sensitivity of the conclusion to influential weeks and product groups.

A naive interval that treats thousands of forecasts from one disrupted week as independent can be too narrow.
A cluster bootstrap resamples an appropriate unit, but a small number of clusters or cross-classified dependence can still make its interval unreliable.
When weeks and product families both share shocks, consult an analyst on a suitable multiway or block design.
No resampling technique creates evidence about regimes absent from the test.

**Output:** an uncertainty statement tied to the actual sampled population and design.

**Branch:** if the interval includes both operationally meaningful harm and gain, the decision is unresolved.
A bounded pilot with safety controls can be appropriate if the cost of learning is justified.
It is not evidence that the model is already safe to automate broadly.

### Step 11. Investigate suspicious benchmark behavior

**Input:** overlap evidence, per-item errors, and authorized probes.

**Procedure:**

1. Compare disclosed training sources and benchmark release dates.
2. Search accessible source files for exact and near-duplicate sequences.
3. Compare performance on exposed, unknown, and demonstrably post-cutoff data where available.
4. Examine whether performance concentrates on particular public datasets or recognizable suffixes.
5. Use permitted perturbations to test robustness without treating a single probe as proof.
6. Ask the model provider for a documented response and evidence of data handling.
7. Retain contradictory evidence and negative search results with their scope.

A model that predicts a known public sequence unusually well may have learned a useful pattern, memorized a copy, or benefited from another data path.
A poor score after random perturbation can arise because the perturbation creates an unrealistic series.
A train/test overlap match demonstrates exposure to content, not the model's internal reasoning.
The strongest conclusion is the one actually supported by the available evidence.

**Output:** a finding, supporting artifacts, competing explanations, and remaining uncertainty.

**Branch:** confirmed future-label exposure makes an independence claim invalid for the affected evaluation.
It does not establish that the model is unusable on all other tasks.
Retest the deployment claim on data inaccessible to development and protect that test from subsequent tuning.

### Step 12. Test transfer to the target operation

**Input:** the cleanest available evaluation, target-population description, and operational constraints.

**Procedure:**

1. Compare public and local distributions for cadence, scale, intermittency, seasonality, and missingness.
2. Compare product life cycles, promotions, stockout behavior, and aggregation levels.
3. Check whether the benchmark horizon matches supplier lead time and review cadence.
4. Check whether training and inference can use the same future-known features in production.
5. Evaluate new products, disrupted supply, and sparse series separately when sample size permits.
6. Quantify cost, latency, uptime, human workload, and integration requirements.
7. State which target populations and periods are not represented.

**Output:** a transfer-risk register and a target-specific test plan.

**Branch:** if the benchmark is clean but differs materially from the local decision, treat it as screening evidence only.
If local history is also limited, start with a controlled prospective shadow evaluation rather than claiming that absence of evidence is proof of harm or benefit.

### Step 13. Protect the holdout and document the decision

**Input:** experiment results, deviations, and operational risk assessment.

**Procedure:**

1. Limit access to test labels and detailed per-item feedback before the final run.
2. Log each model, prompt, and feature change after any holdout result is revealed.
3. Separate exploratory analyses from the predeclared primary comparison.
4. Obtain a fresh holdout after material adaptation to an earlier one.
5. Record the selection rationale, uncertainty, and approved deployment boundaries.
6. Set monitoring triggers, an owner, and a rollback path.
7. Retain data and artifacts under the applicable access and retention rules.

**Output:** a decision memo that another reviewer can reproduce and challenge.

**Branch:** if a vendor refuses enough information to identify a version or preserve a holdout, do not silently fill the gap.
Procurement can seek alternative evidence, narrow the trial, or reject the claim.
The degree of evidence required should be proportional to the potential business harm.

## 6. Worked example

### The proposed procurement decision

A fictional regional distributor replenishes packaged maintenance supplies once each week.
The planning director is considering a vendor's time-series foundation model for a one-week demand forecast.
The vendor claims a 15% average error reduction on public data.
Its training corpus is proprietary, and it cannot confirm whether the public benchmark series were included.
This is an unknown exposure status, not confirmed contamination.

The director wants evidence before authorizing a paid, limited shadow pilot.
The current baseline is last week's observed demand for each product-location series.
The candidate is a fixed vendor model version with no local fine-tuning during the test.
The term zero-shot here means no task-specific local fitting under this protocol; it does not establish that pretraining was free of related data.

### Decision and data assumptions

| Item | Assumption |
|---|---|
| Unit | Cases per product-location per week |
| Forecast origin | Monday 08:00 local time, before the target week |
| Horizon | One week |
| Decision | Compare forecast methods; live orders remain planner-approved |
| Plan | Existing replenishment workflow with review, not a model-generated commitment |
| Service target | 95% fill rate, measured separately from forecast error |
| Test population | Six fictional product-week observations shown below |
| Candidate inputs | Demand available by origin and an approved promotion calendar |
| Baseline inputs | Same available history; baseline does not need a promotion feature |
| Outcome | Reconciled weekly cases, available after each target week |
| Cost proxy | USD 3 per case underforecast, USD 1 per case overforecast |
| Cost period | Sum across the six illustrated product-weeks |

The six observations are intentionally small enough for hand calculation.
They are not sufficient for a production estimate or a confidence interval.
The outcome table is fictional and is not a claim about any real distributor or model.

### Exposure and temporal evidence

| Evidence | Finding | Interpretation |
|---|---|---|
| Public benchmark | Released before vendor model development | Exposure was possible, not established |
| Corpus inventory | Proprietary and not independently inspected | Training overlap remains unknown |
| Local sales history | Private and contractually restricted | Lower exposure risk, not proof of non-exposure |
| Vendor pilot | Predictions saved at each origin before outcomes | Supports prospective temporal ordering |
| Vendor model | Version pinned for six weeks | No silent model update within this illustration |
| Promotion record | Approval timestamp stored | Only schedules approved by each origin are allowed |
| Sales reconciliation | Final actuals used after the week for scoring | Valid label use, not a pre-origin feature |
| Holdout access | Evaluator retains outcomes until all predictions are locked | Reduces adaptation to test labels |

A private dataset could still leak through an earlier vendor integration or a shared report.
The data steward must confirm transfer history and contractual access, rather than infer safety from the word private.
A future prospective period provides stronger evidence about future behavior than replaying a series whose suffix might already be in the model's weights.

### Prediction ledger

The baseline and candidate forecasts are in cases per week.
Actuals are reconciled cases per week.

| Observation | Actual | Baseline | Candidate | Baseline absolute error | Candidate absolute error |
|---|---:|---:|---:|---:|---:|
| A | 83 | 72 | 78 | 11 | 5 |
| B | 116 | 104 | 109 | 12 | 7 |
| C | 47 | 55 | 52 | 8 | 5 |
| D | 132 | 119 | 126 | 13 | 6 |
| E | 68 | 75 | 73 | 7 | 5 |
| F | 94 | 88 | 91 | 6 | 3 |
| Total | 540 | — | — | 57 | 31 |

The baseline's MAE is `57 / 6 = 9.5 cases per product-week`.
The candidate's MAE is `31 / 6 = 5.1667 cases per product-week`, or approximately 5.17.
The paired improvement is `(57 - 31) / 6 = 4.3333 cases per product-week`.
The relative reduction in MAE is `(57 - 31) / 57 = 45.614%`, or approximately 45.6%.
The baseline WAPE is `57 / 540 = 10.556%`.
The candidate WAPE is `31 / 540 = 5.741%`.
The WAPE difference is approximately 4.815 percentage points, not 4.815% relative reduction.
The relative reduction in WAPE is also 45.614% here because both methods share the same positive actual-demand denominator.

### Directional error and cost proxy

Define signed error as forecast minus actual.
A negative value is an underforecast and a positive value is an overforecast.

| Observation | Baseline signed error | Candidate signed error | Baseline proxy cost, USD | Candidate proxy cost, USD |
|---|---:|---:|---:|---:|
| A | -11 | -5 | 33 | 15 |
| B | -12 | -7 | 36 | 21 |
| C | +8 | +5 | 8 | 5 |
| D | -13 | -6 | 39 | 18 |
| E | +7 | +5 | 7 | 5 |
| F | -6 | -3 | 18 | 9 |
| Total | -27 | -11 | 141 | 73 |

The baseline has 42 underforecast cases and 15 overforecast cases.
Its proxy cost is `42 × USD 3 + 15 × USD 1 = USD 141`.
The candidate has 21 underforecast cases and 10 overforecast cases.
Its proxy cost is `21 × USD 3 + 10 × USD 1 = USD 73`.
The difference is USD 68 across the six product-weeks, before model fees and integration costs.
The baseline mean signed error is `-27 / 6 = -4.5 cases per product-week`.
The candidate mean signed error is `-11 / 6 = -1.8333 cases per product-week`.

This linear proxy is not an inventory simulation.
It assumes each forecast miss has a known, constant marginal cost and ignores inventory on hand, lead time, substitution, order constraints, and planner action.
Real business cost may be nonlinear and can depend on service penalties and lost margin.
No actual fill rate or realized cash saving follows from this table.

### Cost of the proposed pilot

Suppose the six-week shadow pilot costs USD 900 in inference and evaluation support.
The illustration's USD 68 proxy improvement does not cover that pilot cost.
The figures are not a fair annual business case because six observations do not represent all product-weeks or downstream costs.
Extrapolating USD 68 to an entire chain without a representative sample and uncertainty analysis would be misleading.
The pilot may still be worthwhile as a learning expense if the potential business value and risk justify it.

### Decision

The director does not accept the public 15% claim as proof of local benefit.
The director authorizes only a pre-budgeted shadow extension with a pinned version and a larger set of product-location series.
The existing baseline continues to drive planner review.
The next evaluation will use multiple weeks, report paired performance by product group and horizon, and estimate uncertainty with dependence-aware methods.
A deployment gate will require a predeclared improvement threshold, acceptable operational cost, no material service-risk segment, and an approved rollback plan.
The exact numerical threshold must be set by the business before looking at the extension's outcomes.

This is a decision under uncertainty, not a finding that the vendor contaminated a benchmark.
If future test outcomes are fed back into prompts or model selection, those outcomes become validation data and a later holdout is required.

## 7. Code, query, or calculation

The following standard-library Python example reproduces the worked table.
It audits availability timestamps for features and reports both a forecast comparison and the illustrative asymmetric proxy cost.
It does not train a model, detect pretraining overlap, simulate inventory, or establish statistical significance.
The datetimes are timezone-aware; all example times are UTC.

**Requirements:** Python 3.10 or later; no third-party packages.
Save the block as `contamination_audit.py` and run `python contamination_audit.py`.

```python
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Forecast:
    item: str
    actual: float
    baseline: float
    candidate: float


@dataclass(frozen=True)
class FeatureRecord:
    name: str
    origin: datetime
    available_at: datetime


def utc(value: str) -> datetime:
    """Parse an ISO timestamp and make the intended UTC zone explicit."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def audit_availability(records: list[FeatureRecord]) -> list[str]:
    """Return features unavailable at their corresponding origin."""
    return [
        record.name
        for record in records
        if record.available_at > record.origin
    ]


def absolute_error(prediction: float, actual: float) -> float:
    return abs(prediction - actual)


def proxy_cost(prediction: float, actual: float) -> float:
    under_cases = max(actual - prediction, 0.0)
    over_cases = max(prediction - actual, 0.0)
    return 3.0 * under_cases + 1.0 * over_cases


def summarize(rows: list[Forecast], field: str) -> tuple[float, float, float]:
    if not rows:
        raise ValueError("At least one paired observation is required")
    errors = [
        absolute_error(getattr(row, field), row.actual)
        for row in rows
    ]
    total_actual = sum(row.actual for row in rows)
    if total_actual <= 0:
        raise ValueError("WAPE requires a positive total actual")
    mae = sum(errors) / len(rows)
    wape = 100.0 * sum(errors) / total_actual
    cost = sum(proxy_cost(getattr(row, field), row.actual) for row in rows)
    return mae, wape, cost


def main() -> None:
    rows = [
        Forecast("A", 83, 72, 78),
        Forecast("B", 116, 104, 109),
        Forecast("C", 47, 55, 52),
        Forecast("D", 132, 119, 126),
        Forecast("E", 68, 75, 73),
        Forecast("F", 94, 88, 91),
    ]
    origin = utc("2026-01-05T08:00:00+00:00")
    features = [
        FeatureRecord(
            "prior_demand", origin, utc("2026-01-04T18:00:00+00:00")
        ),
        FeatureRecord(
            "late_promotion_revision", origin, utc("2026-01-06T09:00:00+00:00")
        ),
    ]
    flagged = audit_availability(features)
    baseline_mae, baseline_wape, baseline_cost = summarize(rows, "baseline")
    candidate_mae, candidate_wape, candidate_cost = summarize(rows, "candidate")
    relative_gain = 100.0 * (baseline_mae - candidate_mae) / baseline_mae
    print("Unavailable features:", ", ".join(flagged) or "none")
    print(f"Baseline MAE: {baseline_mae:.2f} cases")
    print(f"Candidate MAE: {candidate_mae:.2f} cases")
    print(f"Relative MAE reduction: {relative_gain:.2f}%")
    print(f"Baseline WAPE: {baseline_wape:.2f}%")
    print(f"Candidate WAPE: {candidate_wape:.2f}%")
    print(f"Baseline proxy cost: USD {baseline_cost:.2f}")
    print(f"Candidate proxy cost: USD {candidate_cost:.2f}")
    print(f"Proxy cost difference: USD {baseline_cost - candidate_cost:.2f}")


if __name__ == "__main__":
    main()
```

Expected output:

```text
Unavailable features: late_promotion_revision
Baseline MAE: 9.50 cases
Candidate MAE: 5.17 cases
Relative MAE reduction: 45.61%
Baseline WAPE: 10.56%
Candidate WAPE: 5.74%
Baseline proxy cost: USD 141.00
Candidate proxy cost: USD 73.00
Proxy cost difference: USD 68.00
```

The audit checks the supplied timestamps, not their truth.
A fabricated or incorrect availability timestamp can make an invalid record pass.
In production, obtain availability timestamps from trustworthy source-system logs and retain the underlying evidence.
A feature available exactly at the origin is allowed by this example's `<=` rule; define a more conservative cutoff if processing latency makes it unusable in practice.
The sample predictions are already paired and complete.
A real implementation must enforce uniqueness, eligibility, coverage, and a predeclared fallback policy before calculating scores.
The code also assumes a positive baseline MAE for the relative reduction; if the baseline is perfect, use an absolute difference and do not divide by zero.

## 8. Data, standards, and regulatory context

### Data records required for an auditable test

| Record | Minimum content | Control |
|---|---|---|
| Source inventory | Owner, source, version, collection period, permissions | Data steward approval |
| Feature history | Event, availability, revision, and origin timestamps | Point-in-time join and sample trace |
| Model manifest | Provider, artifact or endpoint version, release and update dates | Version pin or explicit moving-service status |
| Development record | Training and tuning data, prompts, and selected settings | Separation from holdout access |
| Benchmark register | Source URL, release date, copies, labels, and license | Overlap review |
| Prediction ledger | Series, origin, horizon, model, prediction, status | Immutable capture before outcome |
| Outcome ledger | Target, value, publication and revision dates | Agreed scoring vintage |
| Decision memo | Metrics, uncertainty, exceptions, owners, authorization | Independent review |

A spreadsheet with a final score cannot reconstruct this chain.
A repository commit can preserve code, but it does not itself prove that a source record existed before the forecast origin.
A signed vendor assurance may be useful evidence, but it is not the same as an independently verified training-data inventory.
A confidential test does not automatically have permission to include supplier or customer data.

### Standards and governance frameworks

The National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 is voluntary guidance.
Its Govern, Map, Measure, and Manage functions offer a structure for ownership, context, measurement, and mitigation.
It does not prescribe a specific contamination detector or certify that a benchmark is clean.
The NIST Generative AI Profile adds guidance for generative AI risks and is likewise not a substitute for applicable law.

ISO/IEC 23894:2023 provides AI risk-management guidance.
ISO/IEC 42001:2023 specifies requirements for an AI management system.
Organizations should consult the authoritative standards and their own certification scope before asserting conformance.
Neither standard turns a vendor score into evidence of local forecasting benefit.

The Organisation for Economic Co-operation and Development (OECD) AI Principles provide high-level policy recommendations.
They are not a universal directly binding technical checklist for a particular replenishment model.
A jurisdiction may impose separate obligations on data protection, confidentiality, consumer protection, product safety, or sector-specific decisions.
Determine applicability with counsel and compliance specialists instead of assuming a general forecasting example meets or violates a particular legal category.

### European Union context

The EU Artificial Intelligence Act, Regulation (EU) 2024/1689, contains obligations that depend on the role, system classification, applicable provisions, and commencement dates.
Some provisions concern providers of general-purpose AI models, while other requirements apply to specified high-risk AI systems and other uses.
A supply-chain forecast is not automatically high-risk solely because it uses a foundation model.
Check the enacted text, current implementation guidance, and the facts of the use case before drawing a legal conclusion.
The regulation does not create a general rule that a public benchmark score proves independence from training data.
This teaching unit is not legal advice.

### Data protection and commercial rights

When test data contain personal data, applicable privacy law can constrain collection, transfer, retention, and processing.
For EU personal-data processing, Regulation (EU) 2016/679, the General Data Protection Regulation (GDPR), sets requirements whose application depends on the actual processing and roles.
A model evaluation plan should identify controller and processor responsibilities where applicable and use an approved lawful basis.
Do not assume hashing alone anonymizes data or authorizes sharing.

Supplier contracts, dataset terms, trade-secret rules, and confidentiality agreements can restrict use even where personal data are absent.
A public benchmark may permit research use but restrict redistribution or commercial use under its specific license.
A vendor's access to a local holdout should be governed by a purpose, recipient, retention period, and permitted feedback.
Deletion and training-use terms should be verified contractually and operationally where possible.

### Time and identity conventions

Use timezone-aware timestamps and a stated business calendar.
A week ending Sunday and a week beginning Monday can refer to the same operational period, but inconsistent labels can create false mismatches.
Store event time, availability time, and revision time separately.
Use stable product and location identifiers, and preserve mappings through mergers, renumbering, and hierarchy changes.
A product code reused for a different item should not silently merge two demand histories.

For product identification, GS1 standards can help organizations identify trade items and locations consistently.
A Global Trade Item Number (GTIN) identifies a trade item under the relevant GS1 rules; a Global Location Number (GLN) identifies a location or party as defined by GS1.
Neither identifier proves data accuracy, availability at a historical origin, or permission to use the record.
Local item-location crosswalks and effective dates remain necessary.

### Source and outcome vintages

The version of the label used for scoring needs an explicit rule.
One protocol may use the reconciled outcome available 30 days after the target week.
Another may use the latest final financial revision at a fixed evaluation close date.
The two can produce different scores, and neither should be passed into the prediction at the earlier origin.
Report later revisions as sensitivity analysis if material.

If a point-in-time source is not retained, a backtest may be impossible to reproduce faithfully.
Do not repair the missing history by asserting that today's value was available earlier.
Use a restricted feature set, collect prospective snapshots, or qualify the result as retrospective.

## 9. Industry practice and case evidence

### Public forecasting competitions

The M4 competition released a large collection of time series and reported results across forecasting methods and frequencies.
Its published results established a useful comparison under that competition's protocol, not a guarantee for any particular distributor.
The associated paper describes the dataset and evaluation context.
Once a competition's series and answers are public, later model developers may encounter them through many channels.
Public availability creates an exposure pathway; it does not by itself show that any named later model used the data.

For a buyer, the practical lesson is to ask which part of a published benchmark was public when a model was developed.
A paper's score on the benchmark may still be reproducible.
It is a weaker estimate of performance on new, undisclosed series if the benchmark or answers were available to model selection.
A score on an unseen future period, with a frozen model and protected test, addresses a different claim.

### Foundation-model reporting

The Chronos paper describes pretrained probabilistic forecasting models and evaluations across datasets.
Its training and evaluation descriptions are evidence to inspect, rather than a blanket certification of all possible downstream uses.
The Chronos repository also documents model artifacts and usage.
A buyer should identify the precise model version, available training-data descriptions, context length, and benchmark protocol relevant to the comparison.

The TimesFM paper presents a time-series foundation model and zero-shot forecasting evaluations.
Its published methodology is a source for understanding the evaluated system and datasets.
Do not infer from the word zero-shot that no evaluation series, near-duplicate, or related public data appeared in any model development step.
Instead, compare the paper's explicit protocol and disclosure against the buyer's desired claim.

These are named research systems, not examples of proven misconduct.
Public reporting can be useful and transparent while still leaving a particular commercial buyer unable to verify every source in a pretraining corpus.
The local transfer question remains even when a benchmark protocol is strong.

### Language-model benchmark contamination

The paper “Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models” studies how training-data contamination affects code-generation evaluation.
It motivates checking both exact and related content when benchmarks are widely distributed.
A supplier-document extraction task differs from code generation, so the paper does not quantify the supplier task's error.
The transferable practice is to document exposure pathways and use protected evaluation material appropriate to the intended use.

For contract extraction, a public sample contract and its labeled answer can enter model development.
A private test contract may still be a near-duplicate of a public template with a few commercial fields changed.
Group document families together and check whether the answer, not just the document shell, was held out.
If a retrieval index includes the answer key or a later annotated version, a fluent response can score well through an invalid route.
The evaluator should restrict retrieval to documents available at the extraction time and independently verify cited spans.

### Model-selection pressure

The study “The Reusable Holdout: Preserving Validity in Adaptive Data Analysis” analyzes the problem of repeated adaptive use of a holdout.
A real project need not query test labels directly to adapt to them.
A team can repeatedly change prompts or model versions after seeing a leaderboard or aggregate score.
Over time, the selected system can fit peculiarities of the holdout rather than the target population.
The remedy is a controlled selection process and fresh evaluation data, not a claim that a single universal query limit guarantees independence.

### Hypothetical operating patterns

The following examples are decision illustrations, not documented company outcomes.

| Situation | Observed evidence | Defensible decision |
|---|---|---|
| Public score strong; training history unknown | Exposure cannot be ruled out | Require a protected local or prospective test |
| Private test beats baseline; test used for repeated prompt selection | Result is validation evidence | Reserve a fresh final test before deployment |
| Candidate wins overall but loses on high-value spare parts | Aggregate masks segment risk | Restrict pilot scope and investigate economics |
| Candidate wins accuracy but exceeds processing window | Forecasts arrive after orders are due | Reject or redesign the workflow despite lower MAE |
| Model changes mid-test without version log | Comparison mixes artifacts | Qualify result and retest controlled versions |
| Good shadow results with no live action | Forecast quality is observed | Do not claim measured live savings or service gains |

### Evidence grading

A peer-reviewed paper, a model card, a vendor slide, and an internal audit answer different questions.
Peer review does not independently verify every implementation or prove the system will transfer to a buyer.
A model card can identify limitations and intended uses but may depend on provider disclosure.
A vendor claim can motivate a test but should not replace locally controlled evidence.
An internal pilot can be highly relevant yet weak if its sample is unrepresentative or its holdout was repeatedly used.
Grade the claim, the evidence, and the intended decision together.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Public answer in pretraining | Benchmark predates model release and sources are undisclosed | Test suffix may enter weights | Public score overstates independent generalization | Audit provenance and use protected prospective data |
| Exact duplicate ignored | Same sequence appears under a different filename | File-name comparison misses copied data | False clean-data claim | Compare normalized content and investigate matches |
| Near-duplicate missed | Series rescaled or time-shifted in a corpus | Hashes do not match transformed content | Residual unknown exposure | Use similarity search with reviewed thresholds |
| Future feature | Feature publication date is after origin | Target-period information enters input | Unrealistic forecast accuracy | Enforce as-of access and exclude unavailable features |
| Final revision leakage | Historical input matches today's revised table | Later corrections used as if known earlier | Backtest cannot be replicated live | Preserve historical vintages |
| Full-data normalization | Scaler fit on all dates | Test distribution influences features | Optimistic evaluation | Fit transforms inside each training window |
| Overlapping labels | Training target extends into test interval | Future test outcomes appear in training | Inflated apparent skill | Purge overlap and justify an embargo |
| Cross-series leakage | Aggregate includes future sales of a target item | Related future outcome becomes a feature | Global model benefits from forbidden information | Audit aggregation timing and split all series consistently |
| Repeated test tuning | Many prompt variants chosen by holdout score | Adaptive selection overfits the test | Winner fails on a fresh period | Limit feedback and reserve a final holdout |
| Hidden model update | Endpoint alias changes behavior or version | Results combine different model artifacts | Irreproducible evaluation | Pin a version or treat service as moving and retest |
| Unequal inputs | Vendor sees promotion plan; baseline does not | Model effect is confounded with information access | Wrong attribution of value | Equalize information or compare bundles explicitly |
| Missing predictions omitted | Coverage falls on difficult series | Complete-case score is selected | Poor cases vanish from aggregate | Report coverage and use an agreed fallback |
| Unequal horizons | Candidate one-week results compared with baseline four-week results | Difficulty differs | Misleading model ranking | Pair horizon and origin |
| Stockout censoring | Sales fall when on-hand inventory is zero | Sales do not equal unconstrained demand | Model learns constrained demand | Flag stockouts and assess a justified demand proxy |
| Small correlated sample | Many rows from one promotion week | Shared shock reduces independent information | Overconfident interval | Use cluster-aware analysis and additional periods |
| Aggregate hides harm | High-volume items dominate a weighted metric | Smaller or critical segments are masked | Service deterioration for vulnerable products | Report segments and predeclare guardrails |
| Post-origin retrieval | Index includes a later annotated document | RAG supplies the answer after the fact | Extraction score is invalid | Version and time-bound the retrieval index |
| Template family leakage | Train and test contracts share near-identical templates | Familiar answer patterns carry across splits | New-template accuracy is overstated | Group by document family and test unseen families |
| Cost shortcut | Error reduction called profit without a policy model | Accuracy is not a financial outcome | Unsupported business case | Simulate or measure decisions and include full costs |
| Unsupported legal claim | A standard is described as universally binding | Guidance and law are conflated | Bad governance or procurement decision | Attribute requirements and confirm applicability |

A single warning does not automatically determine the size of the bias.
Investigate whether the route actually affected the evaluated outputs.
Where effect size cannot be measured, record the uncertainty and restrict the claim.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Availability timestamps are accurate | As-of filtering depends on them | Backfilled or fabricated timestamps | Trace samples to source-system logs |
| Model version is stable | Reproducible comparison needs a fixed candidate | Hosted alias updates silently | Pin artifact or record each endpoint version |
| Holdout outcomes remain protected | Final assessment should not select the model | Scores or answers repeatedly disclosed | Review access logs and use a fresh holdout |
| Source inventory is sufficiently complete | Overlap search needs coverage | Opaque corpus or undocumented mirrors | Record unknowns and use prospective tests |
| Baseline represents a credible alternative | Business case is relative | Outdated or deliberately weak comparator | Include current operational baseline and fair tuning |
| Both methods see legitimate comparable inputs | Attribution requires comparable information | Different promotions or revision vintages | Compare manifests and evaluate information bundles |
| Labels measure the target quantity | Scoring needs meaningful actuals | Stockouts censor demand or labels are revised | Segment censored cases and document outcome vintage |
| Test cases represent deployment | Transfer needs relevant data | Different products, geography, or regimes | Compare distributions and run local prospective tests |
| Dependence is addressed | Uncertainty depends on independent information | Shared weeks, stores, and shocks | Use appropriate cluster or block analysis |
| Costs correspond to real decisions | Financial interpretation requires an operational model | Linear proxy ignores inventory or lead time | Validate a policy simulation or controlled rollout |
| Evaluation period covers relevant conditions | Performance can change by regime | No promotions, disruptions, or launches in test | Collect more periods and state uncovered regimes |
| Permissions allow evaluation | Access does not equal authorization | Restricted supplier, customer, or personal data | Review contracts, privacy basis, and retention |
| Data identities persist over time | Pairing requires stable series definitions | Product code reuse or location remapping | Audit effective-dated master-data crosswalks |
| Forecasts are captured before outcomes | Temporal ordering is central | Predictions reconstructed after the fact | Preserve immutable timestamped prediction ledger |

No negative overlap search can prove absence from an inaccessible corpus.
No historical score guarantees future performance after a distribution shift.
No statistical interval fixes a systematically leaked evaluation.
No clean evaluation removes the need for operational controls.
These limits should be part of the decision record, not footnotes discovered after rollout.

## 12. Exercises

### Exercise 1 — Draw an availability boundary

**Type:** recall and application.  
**Time:** 15 minutes.  
**Scenario:** a forecast is issued on Monday at 08:00 UTC.
Prior-week sales are available Monday at 07:45 UTC.
A promotion schedule was approved Sunday at 16:00 UTC, revised Tuesday at 09:00 UTC, and both versions are retained.
The actual target-week sales are reconciled the following Monday.

**Deliverable:** list which versions may be used as inputs and which may be used only for later scoring.
**Acceptance criteria:** identify the Sunday promotion version and prior-week sales as available, reject the Tuesday revision at the origin, and distinguish the later outcome label from a feature.

### Exercise 2 — Classify exposure evidence

**Type:** critique.  
**Time:** 15 minutes.  
**Scenario:** a benchmark was published two years before a model release.
The vendor discloses no complete corpus inventory.
A search of one public training repository finds an exact copy of a benchmark file, but the model's use of that repository is not established.

**Deliverable:** write a three-sentence exposure finding.
**Acceptance criteria:** distinguish a confirmed file copy from confirmed model training exposure, state the unknown, and propose an additional evidentiary control.

### Exercise 3 — Design a rolling-origin test

**Type:** design.  
**Time:** 30 minutes.  
**Scenario:** evaluate weekly forecasts at one- and four-week horizons for 500 product-location series.
Some four-week outcome labels overlap and sales records are revised after two weeks.
The candidate is a hosted foundation model; the baseline is the current production method.

**Deliverable:** draw two forecast origins and write a protocol covering version, available data, training labels, overlap, scoring, and holdout access.
**Acceptance criteria:** use as-of snapshots, specify a justified purge or delay for unobserved labels, pair methods on the same cases, and reserve an untouched final period.
Explain how the scoring outcome vintage differs from the feature vintage.

### Exercise 4 — Calculate paired performance

**Type:** calculation.  
**Time:** 20 minutes.  
**Scenario:** actual demand is 40, 60, and 50 cases.
Baseline forecasts are 35, 70, and 45 cases.
Candidate forecasts are 38, 64, and 54 cases.
Underforecast cost is USD 3 per case and overforecast cost is USD 1 per case.

**Deliverable:** compute each model's total absolute error, MAE, WAPE, and proxy cost.
Compute the relative reduction in total absolute error and state whether this establishes realized savings.
**Acceptance criteria:** use the common actual-demand denominator, distinguish relative percent from percentage points, and label the cost calculation as a proxy.

### Exercise 5 — Critique a transfer claim

**Type:** application and critique.  
**Time:** 20 minutes.  
**Scenario:** a vendor reports strong monthly national-level forecasts for electricity demand.
A buyer needs weekly forecasts for intermittent spare parts by warehouse and has a nine-week supplier lead time.

**Deliverable:** write five transfer risks and one local evaluation decision.
**Acceptance criteria:** address cadence, aggregation, intermittency, horizon, and business consequences without assuming contamination occurred.

### Exercise 6 — Control repeated adaptation

**Type:** design.  
**Time:** 25 minutes.  
**Scenario:** a team tests 30 prompts against the same labeled set and chooses the one with the highest score.
It now wants to publish that score as a final independent result.

**Deliverable:** propose a short governance procedure covering the existing set, a new holdout, access, and version changes.
**Acceptance criteria:** relabel the old set as selection or validation data, protect a fresh final set, identify an owner, and define what happens after feedback-driven changes.

### Exercise 7 — Write a decision memo

**Type:** decision design.  
**Time:** 30 minutes.  
**Scenario:** a prospective paired test finds lower average error, but the uncertainty interval includes a material loss.
The candidate is slow for a subset of stores and the service fee is known.

**Deliverable:** one-page memo recommending rejection, a controlled pilot, or deployment.
**Acceptance criteria:** state the evidence, uncertainty, full operational costs, decision boundary, monitoring owner, and rollback condition.
Separate a pilot authorization from a production commitment.

### Exercise 8 — Audit retrieval and document families

**Type:** application and design.  
**Time:** 25 minutes.  
**Scenario:** an LLM extracts payment terms from supplier contracts.
The evaluation set contains variants of a public template, and the retrieval index includes documents updated after the original extraction date.
Some evaluation contracts contain confidential information.

**Deliverable:** identify three leakage paths and propose a protected evaluation design.
**Acceptance criteria:** address template-family grouping, as-of retrieval, answer-key exposure, and permitted handling of confidential documents.
Require a cited source span and independent answer verification.

## 13. Self-check answers

### Exercise 1

At Monday 08:00 UTC, prior-week sales available at 07:45 UTC are eligible under the stated timing rule.
The Sunday promotion schedule is also eligible, assuming it was accessible to the system.
The Tuesday revision is not available at the origin and must not overwrite the Sunday snapshot.
The reconciled target-week sales are the label used after the week for scoring.
They are not an eligible feature for the Monday forecast.
If the production process cannot ingest a record within 15 minutes, adjust the rule for actual operational latency.

### Exercise 2

The public benchmark existed before model release, so there was a possible exposure pathway.
An exact copy exists in the searched repository, but the evidence does not establish that the model trained on that repository.
The corpus is undisclosed, so model exposure remains unknown; request provenance or an independent attestation and run a protected prospective evaluation.
A negative search in the one repository would not prove non-exposure through other sources.

### Exercise 3 rubric

A strong answer defines both one- and four-week targets at each origin.
It restricts local fitting to labels observable by the training cutoff, taking the two-week publication delay into account.
It purges or delays overlapping outcome windows where necessary, rather than asserting that a generic gap is always sufficient.
It pins the hosted version or documents why it cannot, and saves predictions before labels are revealed.
It pairs baseline and candidate on identical product-location, origin, and horizon keys.
It uses the latest feature revision available at each origin and a separately fixed later vintage for scoring.
It reserves a final period that is not used to choose prompts, features, or hyperparameters.
A weak answer randomizes individual rows across the entire history or lets future labels enter training.

### Exercise 4

The baseline absolute errors are 5, 10, and 5 cases, totaling 20 cases.
The candidate absolute errors are 2, 4, and 4 cases, totaling 10 cases.
The baseline MAE is `20 / 3 = 6.67 cases per observation` after rounding.
The candidate MAE is `10 / 3 = 3.33 cases per observation` after rounding.
The actual total is 150 cases.
Baseline WAPE is `20 / 150 = 13.33%` and candidate WAPE is `10 / 150 = 6.67%` after rounding.
The relative reduction in total absolute error is `(20 - 10) / 20 = 50%`.
The WAPE difference is approximately 6.67 percentage points, while the relative reduction in WAPE is 50%.

The baseline underforecasts 10 cases and overforecasts 10 cases.
Its cost proxy is `10 × USD 3 + 10 × USD 1 = USD 40`.
The candidate underforecasts 2 cases and overforecasts 8 cases.
Its cost proxy is `2 × USD 3 + 8 × USD 1 = USD 14`.
The USD 26 difference is a simplified proxy, not demonstrated realized savings.
Inventory positions, orders, service, costs, and operational behavior have not been simulated or measured.

### Exercise 5

Monthly frequency and weekly frequency can expose different patterns and data latency.
National aggregation can smooth local volatility that remains important at warehouse level.
Electricity and intermittent spare parts have different zero-demand patterns and business constraints.
A one-period benchmark horizon may not cover a nine-week procurement lead time.
An average error gain may not translate into reduced emergency-shipment cost or acceptable part availability.
A strong answer proposes a local, paired, point-in-time test at decision-relevant horizons with service and cost measures.
None of these differences alone proves benchmark contamination.

### Exercise 6 rubric

Treat the 30-prompt selection set as validation data, not an untouched final test.
Assign a holdout custodian and restrict labels and detailed feedback.
Freeze the selected prompt, model version, and scoring protocol before the final test.
Log each access and change; if final feedback drives another selection, obtain a new final holdout.
A strong answer also records data permissions and the cost of collecting new examples.

### Exercise 7 rubric

A defensible memo states that the interval permits both gain and material harm.
It accounts for the service fee, inference latency, fallback, evaluation, and integration costs.
A pilot recommendation specifies its restricted population, duration, safety checks, owner, and funding.
A deployment recommendation requires a reasoned risk acceptance and operational guardrails; a low point estimate alone is insufficient.
A rejection can be justified if costs or downside exceed the value of further information.
Any choice must define rollback authority and a measurable trigger.

### Exercise 8 rubric

A strong answer groups related contract templates and tests on held-out families where the claim concerns new templates.
It restricts retrieval to document versions available at the relevant extraction time.
It excludes answer keys and annotations from the model's accessible context.
It verifies each extracted payment term against a cited source span independently of the model's explanation.
It identifies a lawful and contractually authorized processing path, access control, retention, and deletion rules.
It states that a template-family split can reduce similarity leakage but cannot prove the model never saw the template during pretraining.

## 14. Interview questions

### 1. What is benchmark contamination?

A strong answer describes evaluation information reaching model development or inference in a way that compromises a stated independence claim.
It gives exact-copy, near-duplicate, and repeated-selection examples.
A weak answer equates every exposure with deliberate misconduct or assumes zero-shot means unexposed.

### 2. What is the difference between event time and availability time?

A strong answer explains that a transaction can occur before an origin but be posted or revised afterward.
It uses the historically available value in the feature snapshot.
A weak answer joins on the transaction date and ignores when the record became accessible.

### 3. Can a model trained on public time series be evaluated on a public benchmark?

A strong answer says yes, provided the claim and exposure are disclosed and the result is interpreted accordingly.
It distinguishes a known prefix from a potentially exposed target suffix.
A weak answer treats the score as either universally invalid or automatically independent.

### 4. How would you audit an opaque vendor model?

A strong answer requests model and data provenance, version pinning, update history, and an independently protected prospective test.
It records an incomplete corpus inventory as residual uncertainty.
A weak answer says a single negative prompt probe proves that no training overlap exists.

### 5. Why might a random train-test split be misleading for weekly demand?

A strong answer identifies temporal dependence, future features, overlapping windows, and regime changes.
It proposes rolling origins and appropriate entity splits for the intended generalization claim.
A weak answer assumes a larger random test set always cures time leakage.

### 6. What makes a baseline fair?

A strong answer matches the population, origins, horizons, available information, operational constraints, and reasonable tuning opportunity.
It includes the existing production method and reports coverage.
A weak answer compares a fully tuned candidate to an unrelated or intentionally weak benchmark.

### 7. Why can a model reduce MAE but increase business cost?

A strong answer discusses asymmetry, inventory position, service constraints, segment mix, and implementation cost.
It evaluates a policy or measures live outcomes before claiming savings.
A weak answer assumes one unit of MAE reduction has a fixed universal dollar value.

### 8. What does a holdout become after repeated model selection?

A strong answer calls it validation data and protects a new final test.
It notes that adaptive use can occur through aggregate feedback, not only through direct labels.
A weak answer keeps calling the set independent because its file name still says test.

### 9. How would you choose an uncertainty method?

A strong answer starts with the sampling and dependence structure and identifies correlated weeks, products, or sites.
It uses a justified cluster or block method and explains limitations with few clusters.
A weak answer treats every forecast as independent merely because there are many rows.

### 10. How does leakage arise in retrieval-augmented extraction?

A strong answer checks answer keys, later document revisions, retrieval index timestamps, and template-family overlap.
It verifies extracted fields against an authorized, time-appropriate source span.
A weak answer assumes retrieval guarantees truth and independence.

### 11. When is a prospective shadow pilot useful, and what can it not establish?

A strong answer uses it to obtain predictions before outcomes and measure local forecast behavior without changing live decisions.
It explains that shadow performance does not directly measure realized inventory or service effects.
A weak answer treats a shadow score as a realized financial saving.

### 12. What would make you stop or constrain a rollout?

A strong answer names predeclared risk thresholds, poor coverage, latency failures, segment harm, unacceptable data access, or uncontrolled model changes.
It identifies an owner and fallback.
A weak answer waits for an aggregate accuracy metric to collapse after the model is already driving orders.

## 15. Key terms

Each term points to its definition or application in this unit.

- [Availability time](#4-definitions-and-vocabulary) — time a record became accessible.
- [As-of join](#step-6-build-point-in-time-training-and-feature-snapshots) — historical join respecting the origin.
- [Baseline](#step-9-score-on-paired-and-decision-relevant-quantities) — comparison method available to the business.
- [Benchmark](#4-definitions-and-vocabulary) — defined comparison dataset and protocol.
- [Bias](#directional-error-and-cost-proxy) — mean signed error under a stated convention.
- [Cluster](#step-10-examine-uncertainty-and-dependence) — group of dependent observations.
- [Contamination](#4-definitions-and-vocabulary) — problematic exposure of evaluation information.
- [Context window](#4-definitions-and-vocabulary) — observations supplied to the model at inference.
- [Cost proxy](#directional-error-and-cost-proxy) — simplified monetary mapping, not realized cost.
- [Distribution shift](#step-12-test-transfer-to-the-target-operation) — difference between evaluation and use conditions.
- [Embargo](#step-7-prevent-overlap-between-windows) — excluded interval motivated by the split design.
- [Event time](#4-definitions-and-vocabulary) — when an event occurred.
- [Exposure](#four-separate-questions) — access to relevant information during development or use.
- [Fine-tuning](#4-definitions-and-vocabulary) — task-specific additional parameter fitting.
- [Forecast origin](#an-information-boundary-not-a-purity-label) — timestamp of a prediction.
- [Frozen model](#step-5-freeze-the-test-protocol) — fixed version and settings through a test.
- [Ground truth](#source-and-outcome-vintages) — reference outcome with a specified vintage.
- [Holdout](#step-13-protect-the-holdout-and-document-the-decision) — reserved final evaluation data.
- [Horizon](#step-1-specify-the-business-decision-and-the-claim) — distance from origin to target.
- [Leakage](#4-definitions-and-vocabulary) — prohibited information crossing an evaluation boundary.
- [MAE](#step-9-score-on-paired-and-decision-relevant-quantities) — mean absolute forecast error.
- [Near-duplicate](#step-3-map-all-development-exposure) — substantially derived or similar content.
- [Paired difference](#step-9-score-on-paired-and-decision-relevant-quantities) — loss difference on matched cases.
- [Pretraining](#step-3-map-all-development-exposure) — initial fitting on a broad corpus.
- [Provenance](#step-3-map-all-development-exposure) — recorded source and custody history.
- [Retrieval-augmented generation](#language-model-benchmark-contamination) — generation informed by retrieved content.
- [Revision time](#source-and-outcome-vintages) — time a record changed.
- [Rolling origin](#step-8-run-rolling-origin-forecasts) — repeated time-respecting forecast evaluation.
- [Shadow mode](#decision) — predictions produced without driving live decisions.
- [Transfer](#step-12-test-transfer-to-the-target-operation) — relevance of evidence to the target operation.
- [WAPE](#step-9-score-on-paired-and-decision-relevant-quantities) — absolute-error sum divided by actual-demand sum.
- [Zero-shot](#the-proposed-procurement-decision) — task use without specified local fitting, not proof of non-exposure.

## 16. Sources

These sources support the named concepts and cases; none validates the fictional example.
Publication dates and legal applicability should be checked against the current authoritative source before operational use.

1. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M4 Competition: 100,000 time series and 61 forecasting methods.” *International Journal of Forecasting*, 2020. [DOI: 10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014).
   Competition data and reported evaluation; public availability should not be equated with verified exposure of a particular model.
2. Ansari, A. F., et al. “Chronos: Learning the Language of Time Series.” *Transactions on Machine Learning Research*, 2024. [Paper](https://arxiv.org/abs/2403.07815).
   Model and evaluation methodology; check the specific artifact and data disclosures.
3. Das, A., et al. “A Decoder-only Foundation Model for Time-Series Forecasting.” *Proceedings of the 41st International Conference on Machine Learning*, 2024. [Proceedings](https://proceedings.mlr.press/v235/das24c.html).
   TimesFM architecture and forecasting evaluation.
4. Dwork, C., et al. “The Reusable Holdout: Preserving Validity in Adaptive Data Analysis.” *Science*, 2015. [DOI: 10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375).
   Formal treatment of adaptive analysis and holdout reuse.
5. Magar, I., and Schwartz, R. “Data Contamination: From Memorization to Exploitation.” *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 2022. [ACL Anthology](https://aclanthology.org/2022.acl-short.18/).
   Discussion of contamination and evaluation of language models.
6. Deng, C., et al. “Investigating Data Contamination in Modern Benchmarks for Large Language Models.” *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics*, 2024. [ACL Anthology](https://aclanthology.org/2024.naacl-long.482/).
   Benchmark contamination methods and limitations.
7. U.S. National Institute of Standards and Technology. [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework), 2023.
   Voluntary risk-management framework, not a certification of benchmark independence.
8. U.S. National Institute of Standards and Technology. [Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence), NIST AI 600-1, 2024.
   Generative-AI risk guidance.
9. ISO. [ISO/IEC 23894:2023, Information technology — Artificial intelligence — Guidance on risk management](https://www.iso.org/standard/77304.html).
   AI risk-management guidance.
10. ISO. [ISO/IEC 42001:2023, Information technology — Artificial intelligence — Management system](https://www.iso.org/standard/81230.html).
    AI management-system requirements within the standard's scope.
11. European Union. [Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), official text, 2024.
    Role-, use-, and date-dependent legal requirements; obtain legal advice for applicability.
12. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), official text, 2016.
    Personal-data obligations depend on the processing and relevant roles.
13. GS1. [GS1 identification keys](https://www.gs1.org/standards/id-keys).
    Official descriptions of identifiers including GTIN and GLN.
14. Organisation for Economic Co-operation and Development. [OECD AI Principles](https://oecd.ai/en/ai-principles).
    High-level policy principles, not a substitute for binding local requirements.
15. Riddell, M., Ni, A., and Cohan, A. “Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models.” *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*, 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.761/).
    Empirical study of contamination in code-generation evaluations; its estimates do not quantify supplier-document extraction risk.

## 17. Links

- [Module 18 topic index](_index.md)
- [Module 18 survey and lab sections](../../../course/18_deep_learning_foundation_models_llm_agents.md)
- [Neural network fundamentals](01_neural-network-fundamentals.md)
- [Global versus local models](03_global-local-models.md)
- [Time-series foundation models](05_ts-foundation-models.md)
- [Simulation, optimization, and production labs](../../../course/30_simulation_optimization_production_labs.md)

Use the survey's lab exercises to apply the information-boundary and decision controls to a broader model evaluation.
