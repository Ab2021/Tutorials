# Vendor Model Evaluation and SR 11-7

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic forecasting and error measurement, decision costs, model documentation, and validation independence; Module 06 is deferred, so the minimum concepts needed are explained below.  
> Version 2026.5; regulatory sources checked 5 November 2026.

## 1. Why this topic exists

A procurement team can buy a forecast without buying the right to inspect its code.
A planning team can act on the forecast without seeing the population on which it was trained.
The organization still owns the decision to use the output and the consequences of that decision.

A vendor's test result answers a narrow question about a specific product, dataset, period, and calculation.
A buyer needs an answer for its own products, locations, lead times, decisions, and operating conditions.
Those questions overlap, but they are not interchangeable.

Consider a supplier promising a 20% accuracy improvement.
The figure could be a reduction in mean absolute error, an increase in a differently defined accuracy score, or an average over selected series.
Without the denominator, baseline, population, and evaluation protocol, the claim is not operational evidence.
Even a genuinely improved forecast can increase cost if its errors are concentrated in high-margin items or if the ordering policy is inappropriate.

A good review therefore separates four decisions:

1. Is the proposed system understandable enough for its intended use?
2. Does evidence support the claimed performance in the buyer's setting?
3. Are remaining limitations and operational dependencies acceptable?
4. Who can stop use, and what happens when the vendor or model changes?

These decisions involve procurement, planning, data engineering, security, legal, and a reviewer who can challenge the sponsor.
The vendor can provide important evidence but cannot independently accept the buyer's residual risk on the buyer's behalf.

A supply-chain company is not automatically subject to U.S. banking supervisory guidance merely because it uses a model.
SR 11-7 is historically important, but it was superseded in 2026 for Federal Reserve supervisory purposes.
This unit teaches vendor evaluation as a transferable practice, not as a claim that the old banking guidance legally governs supply-chain models.

> **Note:** This material is educational, not a legal opinion, certification, or substitute for checking the law, contract, regulator, and current guidance applicable to a particular entity and use.

## 2. Learning outcomes

After this topic you can:

1. Define the decision, intended use, owner, and materiality of a purchased model; exercise 1 checks this skill.
2. Request traceable evidence and distinguish a vendor assertion from an independently reproducible result; exercises 2 and 3 test this skill.
3. Design an out-of-time buyer-side comparison that protects the holdout and uses an appropriate benchmark; exercise 4 tests this skill.
4. Calculate weighted absolute percentage error, bias, and an illustrative decision cost without confusing forecast, plan, target, or commitment; exercise 5 tests this skill.
5. Identify what black-box testing can and cannot establish, including the effects of unavailable code and training data; exercise 6 tests this skill.
6. Assign an independent reviewer, record conflicts and findings, and define a conditional approval or rejection; exercise 7 tests this skill.
7. Explain the historical and current U.S. banking guidance accurately and avoid applying it automatically to a nonbank; exercise 8 tests this skill.
8. Design monitoring, update, incident, and exit controls proportionate to the risk; exercise 9 tests this skill.

### Minimum prerequisite bridge

A **forecast** estimates an uncertain future outcome.
A **plan** is a proposed course of action informed by that forecast and other constraints.
A **target** is a desired result, not a prediction.
A **commitment** is a promise or obligation to a customer, supplier, or internal party.
Do not score a target or order plan as if it were the vendor forecast.

A **baseline** is a simple, feasible alternative available at decision time.
A **holdout** is data withheld from model selection and tuning until evaluation.
An **out-of-time** test uses information available before the forecast origin and evaluates subsequent outcomes.
Leakage occurs when information unavailable at that origin reaches the model, feature construction, selection process, or benchmark.

An error is forecast minus actual in this unit.
A positive error therefore means overforecasting; other conventions reverse the sign.
Absolute error discards the direction, while bias retains it.
Good average performance does not guarantee safe performance for an important subgroup.

## 3. The core mental model

```text
Business decision and harm tolerance
              |
              v
Vendor claims ---> evidence provenance and contract rights
              |                   |
              v                   v
Conceptual review       buyer-side time-respecting test
              \                   /
               v                 v
              independent challenge
                       |
                       v
        findings, uncertainty, and residual risk
                       |
             approve / limit / reject
                       |
          monitor, change, incident, exit
```

The diagram is a chain of evidence, not a procurement checklist whose boxes can substitute for one another.
Vendor documents describe what the product is supposed to do and how it was developed.
Buyer-side tests establish how a specified deployment behaved on a specified sample.
Independent challenge evaluates both the relevance and the weaknesses of those sources.
The decision ties evidence and uncertainty to a bounded use rather than declaring the system universally safe.

### Evidence has a scope

A performance report for a national aggregate demand forecast does not establish item-by-location performance.
A security audit does not establish forecast quality.
A model card can describe intended use but does not prove its claims.
A successful demonstration establishes that one demonstration worked; it does not establish future performance.
A contractual right to audit is useful only if the buyer can exercise it and receive meaningful evidence.

### Review has three connected lenses

**Conceptual soundness:** Is the approach plausible for the decision and data, and are limitations understood?
The buyer may use documented methodology, feature definitions, sensitivity tests, and expert challenge.
If core details are withheld, the conceptual conclusion must explicitly remain limited.

**Outcomes:** Do forecasts compare favorably with actuals and a suitable alternative on a representative, time-respecting sample?
Compute aggregate and subgroup metrics, check the shape and direction of errors, and relate errors to decisions.
No finite test proves future performance or establishes how an opaque algorithm works.

**Operational fit:** Can inputs, versions, outputs, monitoring, overrides, incidents, and fallback be controlled in the real workflow?
An accurate model that cannot meet the decision deadline or silently changes its forecast definition may be unsuitable.

### Independence is practical authority

The reviewer needs relevant technical and business competence, access to evidence, objectivity, and the authority to challenge and escalate.
A vendor sales team cannot provide independent challenge of its own claims.
A buyer employee whose bonus depends on the purchase has a conflict even when technically capable.
An external specialist can assist, but the buyer must assess the specialist's conflicts and oversee the work.
Independence is not a job title or a claim that the reviewer knows nothing about the system.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse or boundary |
|---|---|---|
| Vendor model | Third-party quantitative system used to generate estimates or decisions | Treating the vendor's brand as evidence of suitability |
| Intended use | Specific decision, population, horizon, environment, and user | A broad label such as “supply chain AI” |
| Model owner | Buyer-side person accountable for use and controls | Assuming vendor support owns the buyer's decision |
| Materiality | Importance of exposure and decision purpose | Treating price paid as the only risk measure |
| Inherent risk | Risk before controls, influenced by complexity and assumptions | Equating it with residual risk |
| Residual risk | Risk remaining after controls, including unresolved uncertainty | Assuming unmeasured risk is zero |
| Black box | System whose internals are inaccessible or insufficiently understood | Assuming every proprietary model is inherently untestable |
| Conceptual soundness | Fitness of assumptions, design, and data for the use | Treating prediction accuracy alone as a complete conceptual review |
| Development evidence | Records of design, data, training, tuning, and internal tests | Treating it as independent validation |
| Independent validation | Objective challenge by competent people with sufficient separation and influence | Calling a sales demonstration independent validation |
| Effective challenge | Critical expert analysis able to influence changes | A review with no access, time, or escalation authority |
| Benchmark | Feasible comparator available at the same decision time | Comparing with a deliberately weak or leaked baseline |
| Forecast origin | Time at which a prediction is made | Using data revised after that time |
| Horizon | Time between origin and forecasted outcome, in days or weeks | Comparing different horizons as though equivalent |
| Holdout | Data kept out of selection until final evaluation | Repeatedly tuning to the holdout and still calling it untouched |
| Backtest | Historical replay using only data available at each origin | Randomly splitting time-dependent records without checking leakage |
| Actual | Observed outcome used as a reference | Assuming censored sales equal unconstrained demand |
| WAPE | Sum of absolute errors divided by sum of actuals, expressed as a percentage | Averaging item-level percentage errors and calling it WAPE |
| Bias | Signed error summarized over observations | Omitting the sign convention |
| Segment | Subpopulation such as item class or location | Reporting a small group's unstable metric without its sample size |
| Overlay | Documented adjustment to output | An unlogged override that destroys reproducibility |
| Drift | Change in inputs, relationships, outputs, or performance over time | Treating every change in input distribution as performance degradation |
| Version | Identified release of model and relevant configuration | Assuming an API endpoint identifies an immutable model |
| Audit right | Contractual access to agreed evidence and review processes | Assuming a clause guarantees code access or regulatory access |
| Data lineage | Traceable path from source to feature, prediction, and outcome | A data dictionary without timestamps or transformations |
| Reproducibility | Ability to recreate or explain a past result from recorded artifacts | Assuming a seed alone recreates a hosted service |
| Validation finding | Documented issue, severity, impact, owner, and due date | A concern with no decision or follow-up |
| Fallback | Tested alternative process when the service or model cannot be used | An untested promise to use spreadsheets |
| Service-level agreement (SLA) | Contractual service measures and remedies | Treating uptime as evidence of forecast validity |
| Subprocessor | Another party processing data for the vendor | Ignoring downstream data access and dependencies |

## 5. The mechanics, step by step

### Step 1 — Define the decision and its boundary

**Input:** business proposal, process map, and proposed users.
**Action:** write a one-paragraph intended-use statement with decision, product-location population, cadence, horizon, and permissible action.
**Output:** an owned use case and a boundary for the evidence request.

For example, a weekly forecast may inform replenishment for stable finished goods at two distribution centers.
The system does not itself authorize a purchase order, establish a customer commitment, or set the service-level target.
Name the unit of forecast, such as units per item-location-week, and define when the forecast is created.
Identify lead time, review cycle, planning constraints, and the decision deadline.

List excluded uses such as promotional items, new products, safety-critical stock, or automatic ordering until evidence supports them.
Quantify exposure where practical: annual spend influenced, potential stockout impact, affected customers, and replacement time.
If a model influences a larger portfolio or more consequential decision, increase the depth of review and the authority needed for acceptance.
Do not reduce review solely because the vendor product is widely used.

### Step 2 — Map the full system

**Input:** data-flow diagram, integration design, vendor description, and purchasing workflow.
**Action:** trace data from source through preprocessing, vendor endpoint, output transformation, planner override, and ordering decision.
**Output:** system boundary, dependencies, and ownership map.

Record buyer-owned inputs and vendor-owned transformations separately.
Identify whether a vendor or its subprocessor retains data, retrains on customer data, or shares a foundation model across customers.
Identify versions for preprocessing, the model, configuration, post-processing, and the downstream ordering policy.
A buyer-side custom feature or post-processing adjustment is part of the deployed system even if the vendor does not own it.

Describe failure paths: stale data, missing predictions, duplicate records, late responses, changed units, authentication failure, and inaccurate predictions.
Decide which failures block an order and which trigger a human review.
Name the existing process to use if the vendor is unavailable.

### Step 3 — Inventory claims and request evidence

**Input:** proposal, demonstrations, and draft contract.
**Action:** turn each material claim into a testable statement and an evidence request.
**Output:** a claim-to-evidence register, with missing information explicitly marked.

| Claim or question | Minimum useful request | Evaluation question |
|---|---|---|
| “Improves accuracy” | Metric formula, comparator, sample, horizon, results, and exclusions | Is the improvement relevant and reproducible? |
| “Works for promotions” | Definition, covered promotion types, sample sizes, and subgroup outcomes | Does it cover this buyer's promotions? |
| “Learns automatically” | Update triggers, retraining schedule, approval and rollback mechanism | Can an unannounced change affect decisions? |
| “Explainable” | Example explanation, method, scope, and known failure modes | Does explanation describe an input association or prove causation? |
| “Secure” | Independent security assurance scope and date, plus incident controls | Does it cover the deployed service and data? |
| “Validated” | Full report or reviewable summary, reviewer independence, period, methods, and findings | Was the current version and use actually assessed? |
| “High availability” | SLA exclusions, outage history where available, and recovery procedures | Can the planning process continue during an outage? |
| “No data leakage” | Point-in-time feature and split construction, training cutoff, and tuning history | Can the claim be independently checked? |

Request a model description, expected input schema, feature availability, missing-value behavior, output semantics, and version history.
Ask for development and training data provenance at the level the vendor can disclose: source types, time periods, populations, consent or rights basis as appropriate, and representativeness.
Request known limitations, unsupported use cases, historic incidents, change notices, and monitoring guidance.
Ask for sample prediction records with origin time, target period, unit, model version, and confidence information if provided.

Specify whether the vendor may provide source inspection, a secure review room, redacted documentation, a third-party attestation, or controlled test access.
A redacted report may be useful, but the reviewer must know which questions the redaction prevents answering.
If the vendor refuses a material evidence request, record it as a limitation and decide whether compensating controls are sufficient.
Do not silently replace missing information with a favorable assumption.

### Step 4 — Assess provenance and conflicts

**Input:** received artifacts, authors, dates, versions, and reviewer relationships.
**Action:** record source, independence, scope, and relevance for each artifact.
**Output:** an evidence matrix, with confidence and unresolved gaps.

A vendor-authored technical note is development evidence, not an independent opinion.
An independent report may still be stale, limited to security controls, based only on vendor-supplied data, or outside the buyer's use case.
Check the report's commissioning party, payment relationship, access, test scope, limitations, and unresolved findings.
Do not describe an unverified reviewer as independent merely because the vendor used that word.

Compare the evaluated release and configuration with the proposed deployment.
If the report covers a different product or a previous model, request a change-impact analysis.
Check whether claims concern a global aggregate or a subgroup matching the buyer's units, horizon, and demand pattern.
Distinguish measured facts from vendor assertions and from the reviewer's inference.

### Step 5 — Challenge the conceptual design

**Input:** model documentation, data description, intended use, and expert knowledge.
**Action:** assess whether the stated mechanism and assumptions plausibly support the decision.
**Output:** conceptual findings, limitations, and targeted tests.

Ask what the model predicts: observed sales, unconstrained demand, orders, or shipments.
Sales may be lower than demand when stockouts censor demand; a system trained on sales may learn availability rather than underlying need.
Determine whether prices, promotions, inventory availability, lead times, and calendar events are known at the forecast origin.
A feature containing realized future promotions or revised future sales leaks information, even if the forecast algorithm itself is sound.

Examine the intended response to product launches, delistings, intermittent demand, structural breaks, and data gaps.
Determine whether the model pools across items or locations and whether this may hide critical local failures.
For an opaque model, use controlled input perturbations to inspect output sensitivity, while acknowledging that finite probes do not reveal all mechanisms.
For example, change one plausible promotion flag while holding other inputs fixed and record whether the response is plausible.
Avoid claiming a sensitivity test proves causal understanding or absence of hidden dependencies.

### Step 6 — Design a buyer-side evaluation

**Input:** intended use, accessible history, actuals, and operational benchmark.
**Action:** preregister the evaluation population, forecast origins, horizons, metrics, exclusions, and decision thresholds.
**Output:** signed test protocol before the final holdout is opened.

Select historical origins and recreate the information available at each origin.
Record data snapshots and publication timestamps, not just transaction dates.
Use rolling origins when feasible so that performance is not dominated by one period.
If a vendor backfills predictions using a model trained on later information, label the result as retrospective and do not treat it as a clean out-of-time test.

Compare vendor forecasts with a feasible current process and a simple seasonal or last-observation baseline where appropriate.
Ensure both receive equivalent input availability, decision horizons, and evaluation populations.
If a benchmark depends on a planner's later judgment, avoid presenting it as available at the original forecast time.
Define a valid observation and report missing vendor forecasts and exclusions rather than evaluating only successful predictions.

Measure WAPE and signed bias where meaningful, but do not rely on a single number.
Report by item importance, location, horizon, promotion status, stockout status, and other relevant groups when sample sizes permit.
Report denominators, counts, and periods, and distinguish the number of rows from the number of independent forecast origins.
Consider uncertainty from correlated observations, seasonality, and a small number of origins.
A bootstrap or interval needs a resampling unit that respects temporal and cross-series dependence; otherwise its precision may be misleading.

Before the holdout is opened, write what would cause approval, conditional approval, retesting, or rejection.
Separate a numerical performance hurdle from non-negotiable controls such as a tested fallback and known update procedure.
The thresholds are buyer policy choices, not universal legal requirements.

### Step 7 — Evaluate actual decision consequences

**Input:** predictions, actuals, baseline, and a stated decision-cost model.
**Action:** translate errors into an illustrative operational consequence or replay the actual ordering policy.
**Output:** decision-focused comparison, including distribution of harms and assumptions.

The cost of a unit underforecast may differ from the cost of a unit overforecast.
Cost can depend on lead time, inventory already held, substitutions, capacity, backorders, and the ordering policy.
A simple asymmetric error-cost calculation can expose directionality, but it is not a full inventory simulation.
State whether costs are per unit per week and whether they include purchase price, margin, holding, or lost service.
Do not double-count consequences already embedded in another cost term.

A real policy replay should respect order placement, replenishment lead time, available inventory, constraints, and observed demand where available.
Use comparable initial inventory and decision rules for both vendor and baseline.
If observed sales are censored, do not claim the replay measures all lost demand without a justified estimate.
Report subgroup harms even if total cost improves.

### Step 8 — Challenge black-box limitations

**Input:** evidence matrix, test results, and withheld information.
**Action:** determine what can be independently checked and what remains unknown.
**Output:** a bounded conclusion rather than a false assurance of complete validation.

A buyer can verify input and output contracts, test data availability, compare observed outputs to outcomes, and inspect decision impacts.
The buyer may test stability under legitimate input perturbations, missing data, and known edge cases.
These checks do not prove that hidden training data were lawful, that the model has no leakage, that its code is correct, or that a future release will behave identically.
A third-party report can reduce uncertainty about its own scope but cannot automatically close every gap.

Ask whether the withheld element is material to the intended use.
If inspection of training provenance is essential because customer data rights are disputed, accuracy tests do not settle the legal issue.
If a high-impact system cannot be explained sufficiently, consider restricting the use, requiring reviewable alternatives, or rejecting it.
If a limited use has a strong fallback, low exposure, and well-observed performance, a documented conditional acceptance may be proportionate.

### Step 9 — Perform independent review

**Input:** protocol, evidence, analysis, documented system, and conflict disclosures.
**Action:** assign a reviewer with skills, access, objectivity, and escalation authority outside the commercial decision incentive.
**Output:** signed findings and a recommendation to the designated decision maker.

The reviewer challenges the sponsor's assumptions, exclusions, benchmark, and threshold choices.
The reviewer should be able to request additional evidence, reproduce calculations, and raise unresolved disagreements.
Document reporting lines, vendor relationships, financial interests, and participation in model development.
A reviewer who helped tune the specific model should not alone sign off on the adequacy of that same tuning.
In a small organization, separation may require an outside reviewer or a distinct reporting line with documented safeguards.

The reviewer is not the business risk owner.
The reviewer recommends a disposition and explains evidence limitations; the accountable owner decides within delegated authority.
Internal audit, where used, can evaluate whether the governance process is effective without duplicating every validation test.
Do not equate an audit of the process with an independent technical validation of the model.

### Step 10 — Resolve findings and make a bounded decision

**Input:** findings, cost comparison, missing evidence, risk appetite, and control capabilities.
**Action:** classify each finding and record acceptance, remediation, or rejection with an owner and deadline.
**Output:** approval record with permitted use, conditions, expiry, and escalation.

A finding should include evidence, affected use, severity rationale, consequence, owner, due date, and closure criteria.
Severity depends on impact and likelihood or uncertainty, not just the number of missing documents.
A material unknown may justify rejection even if observed accuracy looks good.
Do not close a finding merely because the vendor promises to investigate.

Possible dispositions include approval for a defined use, conditional approval with enforceable controls and a deadline, and rejection.
Conditional approval is not permission to operate without the agreed controls.
Set a review date and identify who can suspend use before that date.
Document alternative action if conditions are missed.

### Step 11 — Negotiate operational and evidence rights

**Input:** risk decision and procurement contract.
**Action:** translate evidence and control needs into specific contractual obligations and verify feasibility with counsel.
**Output:** operationally enforceable evidence, notice, incident, and exit provisions where negotiated.

Consider access to documentation, independent assurance, controlled tests, and audit evidence.
Specify release identification, change notification, material-change criteria, and the period available for buyer review.
Negotiate incident notification, severity, contacts, evidence retention, and cooperation on root-cause analysis.
Define data use, retention, deletion, subcontractors, data location where relevant, and permitted training uses.
Record service availability, recovery objectives, remedies, and limits of liability as contractual issues rather than guarantees of model quality.

Specify what is exported at termination: inputs, outputs, configuration, documentation, logs, and permitted historical artifacts.
Define return or deletion of buyer data and verification mechanisms.
A source-code escrow clause, if offered, does not by itself ensure the buyer can operate the model, has rights to training data, or can reproduce a hosted environment.
The actual agreement and applicable law control; the checklist does not replace legal review.

### Step 12 — Monitor, respond, and exit

**Input:** approved use, production data, actuals, vendor notices, and incident reports.
**Action:** monitor model performance and operational controls on a schedule matched to risk and data availability.
**Output:** versioned monitoring record, incident decisions, and periodic revalidation triggers.

Track forecast completeness, timeliness, input freshness, version, overrides, bias, WAPE, and significant subgroup performance.
Set alert thresholds and an owner, investigate sustained or material breaches, and avoid automatic claims that a distribution shift proves failure.
When actuals arrive late or are revised, report the maturity and version of outcomes used.
Watch for drift in product mix and missingness as well as forecast errors.

Trigger review when the vendor changes model logic, training population, data usage, hosting, or output semantics; when the buyer changes use or population; or when a material incident occurs.
A normal scheduled update is still a change whose effect depends on its scope.
Require testing and explicit authorization before moving an unapproved version into a consequential workflow when feasible.
If a hosted vendor cannot freeze versions, agree on advance notice, shadow comparison, rollback, and a fallback before relying on it.

Rehearse the fallback: identify the person who invokes it, data needed, output location, and the time to restore a planning cycle.
Retain the evidence needed to explain which model and inputs informed past decisions, subject to retention and privacy rules.
Close the lifecycle by revoking access and checking data disposition when the relationship ends.

### A. Evidence request: development and training

Ask the vendor to identify the model family and its intended prediction target.
The answer need not reveal proprietary parameters to establish whether the target is sales, demand, shipments, or another quantity.
Ask what the unit of observation is and how products, locations, and time buckets are represented.
Record whether a single global model is shared or separate models are fitted to individual series.
Ask how the approach handles series with limited histories.
Ask whether external data are used and how frequently those data are updated.
For each important feature, request its definition, source, expected latency, and availability at the forecast origin.
Identify any variables that are known only after the outcome occurs.
Ask how missing values, zeros, outliers, and discontinuations are handled.
Distinguish a missing record from a measured zero.
Ask how promotions are represented and whether future promotions must be entered manually.
Ask how planned prices become model inputs.
Ask whether historical prices are revised after the fact.
Ask how stockouts and censored demand are identified, if at all.
If stockout correction is used, request its assumptions and developmental tests.
Ask how new products inherit information from existing products.
Ask whether model parameters are estimated across customers or within the buyer's data alone.
If pooling occurs, ask what access controls and data-use permissions govern it.
Ask whether customer data can affect predictions for other customers.
Ask whether the vendor uses buyer data for model training, evaluation, support, or marketing.
Request the applicable training periods and the latest data cutoff.
Request the criteria for selecting the training population.
Ask how exceptional periods were handled and whether any were excluded.
Ask for the method of hyperparameter selection and its evaluation periods.
Ask whether any human judgment changed training labels or configuration.
Request a description of checks for duplicates and leakage.
Request known failure cases and conditions under which the vendor advises against use.
Record the evidence received, its version, and any unanswered request.
If the response is unavailable for proprietary reasons, record what conclusions depend on it.

### B. Evidence request: performance and validation

Request the complete metric definitions rather than only a dashboard screenshot.
Request the number of forecast origins and the length of the evaluation window.
Request the number of item-location series and the number of forecasts attempted.
Request the number of forecasts returned on time.
Request the number of predictions excluded from scoring and the reason for each category.
Request actual-volume denominators alongside aggregate percentage metrics.
Request error distributions, not just averages.
Request separate results for long-tail and high-volume products.
Request results for new products and short-history items.
Request results for promotions and non-promotion periods where relevant.
Request results for stockout-affected and unaffected periods.
Request results by forecast horizon.
Request signed bias using an explicitly stated sign convention.
Request a comparator that is credible for the relevant planning setting.
Request confirmation that the comparator had the same information and forecast cutoff.
Request the selection procedure for any customers included in a public case study.
Request the size and duration of any improvement claim.
Ask whether the reported period overlaps development or parameter tuning.
Ask whether a vendor expert changed configuration after seeing holdout results.
Request an explanation for material changes in performance between releases.
Request the validation report's scope, author, date, and conflict disclosures.
Request the report's findings, remediation status, and any scope exclusions.
Ask whether an independent party could reproduce the reported metric from supplied predictions and actuals.
If raw data cannot be shared, consider a controlled review environment or a reproducible aggregate.
Record whether the buyer can inspect evidence directly or relies on an attestation.
Avoid converting a single positive result into a claim of performance in every deployment.

### C. Evidence request: production operation

Request the service interface specification and a sample input and output schema.
Request a definition of the timestamp attached to each forecast.
Ask whether a response includes the model release and configuration identifiers.
Ask how the service reports missing, invalid, or stale inputs.
Ask whether it can silently substitute defaults when data are missing.
Ask what happens when a product or location is not recognized.
Ask what happens when an input exceeds the range seen during development.
Ask whether forecasts are bounded and how negative outputs are treated.
Ask how output intervals or probabilities, if offered, were calibrated.
Ask whether the same request can yield different results and why.
Ask how long logs are retained and whether the buyer can export them.
Ask how corrections to historical data affect previous and future forecasts.
Ask how model updates are identified, announced, tested, and deployed.
Ask which updates are automatic and which require approval.
Ask whether a previous release can be restored.
If rollback is impossible, ask how a buyer can compare old and new outputs.
Ask how planned maintenance and unexpected outages are communicated.
Ask for incident severity definitions and contact responsibilities.
Ask how the vendor investigates output errors that are not service outages.
Ask for recovery objectives, dependencies, and tested restoration procedures.
Ask how subprocessors are selected, monitored, and changed.
Ask how the vendor separates customer data and authorizes access.
Ask how the buyer can retrieve predictions and configuration at termination.
Ask how data deletion is verified and which records must be retained.
For each answer, identify the contract provision or operational evidence that supports it.

### D. Point-in-time test construction

Start with an inventory of forecast origins, not a table of final revised outcomes.
At each origin, identify the inputs available before the operational cutoff.
Create a snapshot identifier for the data actually available at that time.
Store the forecast creation time separately from the target period.
Record the period start and end using an explicit time zone.
For a weekly forecast, define whether the week begins on Sunday, Monday, or another day.
Align the vendor and baseline on the same item, location, origin, and horizon.
Check for duplicate keys before joining predictions to outcomes.
Keep a record of all requested predictions, even if the vendor returns none.
Join the eventual actuals only after the prediction records are frozen.
Preserve the date and version of each actual, especially if outcomes are revised.
Do not retroactively replace the origin-time feature snapshot with its corrected future value.
If the data pipeline cannot recreate historical availability, state that limitation.
An available transaction date is not necessarily the timestamp when a planner could see the transaction.
A revised promotion calendar may contain information that was unavailable at the original forecast origin.
Historical backfilled predictions are not automatically invalid, but their training cutoff must precede each scored outcome.
Where that cutoff cannot be demonstrated, label the result appropriately and avoid claiming a clean holdout.
If a single static model is tested across multiple origins, ensure its training period predates the earliest scored target.
If the model is retrained at each origin, document exactly which observations enter each retraining run.
Compare the benchmark under equivalent access to data and revisions.
Define exclusions before reviewing performance.
Report any post hoc exclusions separately with their effect on results.
Use a later, previously unseen period for the final comparison if development used earlier test results.

### E. Monitoring specification

A monitoring plan should say who receives each result and what action follows.
The owner should be able to locate the underlying output and actuals for every alert.
Forecast completeness is the number of usable outputs divided by the number requested.
Define “usable” before calculating this rate.
Forecast timeliness is the share received before the planning cutoff.
Input freshness measures age relative to the promised update schedule.
Schema checks detect changed field names, units, and invalid values.
Version checks detect a change in the model or its configuration.
Override rate measures the share of outputs changed by planners.
An increased override rate is a signal for investigation, not proof that either planner or model is correct.
Track override direction and reason where this can be done consistently.
Track WAPE and signed bias as outcomes mature.
Avoid comparing a preliminary period with a fully revised period without explaining the difference.
Track critical segments separately from the aggregate.
Show sample counts when a segment is small.
Set alerts against a benchmark or tolerances agreed before an incident.
Distinguish an alert from a decision to suspend use.
An alert should create a triage task with an owner and time limit.
Triage should check data quality, mapping changes, operational changes, and model behavior.
If input data are stale, restoring the feed may be the immediate action.
If a model release caused a performance break, quarantine the release where feasible.
If no root cause is known and exposure is material, invoke the approved fallback.
Record the decision and evidence, including a decision not to suspend.
Escalate repeated alerts and unresolved findings to the accountable risk owner.
Review alert thresholds periodically so that they remain useful as the population changes.
Keep incident and remediation evidence subject to retention and access controls.

### F. Example of a findings register

The following entries are illustrative and do not refer to a real vendor.
Severity labels are local governance choices, not regulatory classifications.

**Finding F1 — version mismatch.**
The available report covers release 2.1 while the proposed deployment uses release 2.4.
The release notes do not say whether the training data or model behavior changed.
The consequence is that the reported performance cannot be attributed confidently to the deployed release.
Request a version-specific change analysis and a buyer-side comparison.
The vendor supplies release documentation; the buyer's reviewer evaluates it.
The finding closes only when the reviewer records why the evidence covers the deployed release.
If evidence remains unavailable, the risk owner decides whether the use must be restricted or rejected.

**Finding F2 — outcome censoring.**
The buyer's actuals represent sales, and no inventory availability flag is supplied.
Observed zero sales may mean no demand or no stock.
The consequence is that a low forecast may appear accurate during an unobserved stockout.
The data owner should join availability records and flag uncertain periods.
The reviewer should report how much evaluation volume is affected.
Closure requires a documented treatment of affected observations and its limitations.
If availability cannot be recovered, do not present sales as fully observed demand.

**Finding F3 — untested fallback.**
The business intends to revert to the existing planning process during a service outage.
No one has checked whether that process still receives the required data.
The consequence is a missed planning deadline despite an apparently documented fallback.
The operations owner should perform a timed exercise with the current inputs and staff.
The exercise should record who invoked the process and whether a usable plan was produced.
Closure requires an approved runbook and evidence that the recovery time meets the business need.
If the exercise fails, correct the dependency or limit reliance on the service.

**Finding F4 — conflict in review.**
The proposed reviewer has a financial incentive tied to the vendor contract.
This incentive may affect the reviewer's ability to challenge evidence.
The sponsor should disclose the relationship rather than simply removing the name from the report.
Assign a suitably competent reviewer with a separate incentive and escalation route.
The replacement reviewer should independently assess the disputed evidence.
Closure requires documented conflict treatment and an independent conclusion.

### G. Procurement decision record

Record the date, decision maker, and model owner.
Record the product, release, configuration, and environment.
Name the business decision supported by the output.
List the approved users and excluded populations.
State whether the model may advise, recommend, or automatically execute an action.
List the evidence considered and link each artifact to a version and date.
Summarize conceptual review and unresolved access limitations.
Summarize the test population, origins, horizon, and benchmark.
Show performance, bias, coverage, and critical subgroup results.
State the business cost assumptions and whether the calculation is a proxy or policy simulation.
List findings and their severity rationale.
Name the owner and deadline for each unresolved finding.
Record the controls that must operate before use begins.
Record the fallback and who can invoke it.
Define the conditions that trigger retesting or suspension.
Name the authority accepting the residual risk and any dissenting reviewer.
Set the expiry date or next review trigger in the approval record.
Provide a distribution list so operators receive the limitations relevant to their work.
Retain the decision and supporting artifacts under the organization's evidence policy.

## 6. Worked example

### Case and assumptions

A fictional consumer-goods distributor evaluates a vendor's weekly forecasts for four item-location combinations.
The use is a one-week-ahead replenishment planning input; a planner approves all orders.
The forecast is not the purchase plan, service target, or customer commitment.
Assume the four observations are a clean illustrative out-of-time holdout with the same origins, units, and availability for both methods.
This small sample is for arithmetic, not sufficient evidence for procurement.

The benchmark is the existing feasible planning forecast recorded at the same origins.
All figures below are units per item-location-week.
The sign convention is forecast minus actual.

| Series | Actual | Vendor forecast | Benchmark forecast | Vendor error | Benchmark error |
|---|---:|---:|---:|---:|---:|
| A | 120 | 110 | 100 | -10 | -20 |
| B | 80 | 90 | 85 | +10 | +5 |
| C | 150 | 135 | 145 | -15 | -5 |
| D | 50 | 55 | 40 | +5 | -10 |
| Total | 400 | 390 | 370 | -10 | -30 |

All totals are units across these four observations.
Forecast errors are calculated separately for each row; aggregation does not erase individual errors.

### WAPE and bias

Vendor absolute error is `10 + 10 + 15 + 5 = 40 units`.
Benchmark absolute error is `20 + 5 + 5 + 10 = 40 units`.
Actual demand is `120 + 80 + 150 + 50 = 400 units`.
Vendor WAPE is `40 / 400 × 100 = 10.0%`.
Benchmark WAPE is also `40 / 400 × 100 = 10.0%`.
The relative improvement in WAPE is `(10.0 - 10.0) / 10.0 = 0%`.
The vendor has not demonstrated an aggregate WAPE improvement on this sample.

Vendor signed error is `-10 + 10 - 15 + 5 = -10 units`.
Its aggregate signed bias divided by actual is `-10 / 400 × 100 = -2.5%`.
Benchmark signed error is `-20 + 5 - 5 - 10 = -30 units`, or `-7.5%` of actual.
The vendor is less negatively biased in aggregate, but that does not guarantee lower cost for every item.
Signed errors cancel, so bias is not a substitute for absolute error.

### Illustrative asymmetric cost

Assume each underforecast unit costs USD 8 and each overforecast unit costs USD 2 over the evaluation week.
This stylized cost treats errors as immediately consequential, with no initial inventory, lead time, substitution, or dynamic policy.
It is not an estimate of actual realized inventory cost.

| Series | Vendor under/over units | Vendor cost USD | Benchmark under/over units | Benchmark cost USD |
|---|---|---:|---|---:|
| A | 10 under | 80 | 20 under | 160 |
| B | 10 over | 20 | 5 over | 10 |
| C | 15 under | 120 | 5 under | 40 |
| D | 5 over | 10 | 10 under | 80 |
| Total | 25 under, 15 over | 230 | 35 under, 5 over | 290 |

Vendor cost is `25 × USD 8 + 15 × USD 2 = USD 230`.
Benchmark cost is `35 × USD 8 + 5 × USD 2 = USD 290`.
The illustrative saving is `USD 290 - USD 230 = USD 60` across these four item-location-weeks.
The relative reduction is `60 / 290 × 100 = 20.69%`.
Series C is worse under the vendor: USD 120 versus USD 40, an increase of USD 80.
A buyer must investigate whether C is a critical product before treating the aggregate saving as acceptable.

### Evidence and decision

Suppose the vendor provides a recent general technical summary but will not disclose its training records or source code.
Its sales presentation reports 20% better accuracy but does not supply the baseline, sample, or metric.
The independent reviewer can reproduce the four-row arithmetic, but cannot determine whether the general summary describes this exact configuration or whether training data include the holdout.
The reviewer records an unverified claim, unavailable provenance, and inadequate sample size as distinct findings.

The appropriate decision is **not approved for production on this evidence alone**.
The sponsor may request a version-specific development and data summary, a clean larger holdout, missing-prediction accounting, and a critical-item analysis.
A controlled shadow evaluation could generate additional evidence if it does not affect orders and data permissions are satisfied.
The owner can revisit the decision once the findings are addressed; no invented vendor assurance closes them automatically.

### Alternative assumptions

If underforecast and overforecast each cost USD 1 per unit, both methods cost USD 40 in this simplified calculation.
If the cost of underforecast is USD 2 and the cost of overforecast is USD 8, vendor cost is `25 × 2 + 15 × 8 = USD 170`, while benchmark cost is `35 × 2 + 5 × 8 = USD 110`.
Thus the preferred method depends on a defensible cost model, not WAPE alone.
If the true demand for a stockout item exceeds recorded sales, the table's “actual” is not necessarily true demand.
A decision memo must keep that uncertainty visible.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library and reproduces the example.
Save it as `vendor_review.py` and run `python vendor_review.py`.
The embedded data are fictional, already aligned at the same origin and horizon, and use units per item-location-week.
The program does not fetch a vendor service, validate data provenance, estimate uncertainty, or simulate inventory.
Its cost is the stated asymmetric error proxy, not a production financial estimate.

```python
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    series: str
    actual: float
    vendor: float
    benchmark: float


ROWS = (
    Observation("A", 120, 110, 100),
    Observation("B", 80, 90, 85),
    Observation("C", 150, 135, 145),
    Observation("D", 50, 55, 40),
)


def summarize(
    rows: Iterable[Observation],
    forecast_field: str,
    under_cost: float,
    over_cost: float,
) -> dict[str, float]:
    observations = tuple(rows)
    if forecast_field not in {"vendor", "benchmark"}:
        raise ValueError("forecast_field must be vendor or benchmark")
    if not observations:
        raise ValueError("at least one observation is required")
    if under_cost < 0 or over_cost < 0:
        raise ValueError("unit costs must be nonnegative")
    if any(row.actual < 0 for row in observations):
        raise ValueError("actuals must be nonnegative")
    actual_total = sum(row.actual for row in observations)
    if actual_total <= 0:
        raise ValueError("WAPE requires a positive sum of actuals")

    errors = [getattr(row, forecast_field) - row.actual for row in observations]
    absolute_error = sum(abs(error) for error in errors)
    signed_error = sum(errors)
    under_units = sum(max(-error, 0) for error in errors)
    over_units = sum(max(error, 0) for error in errors)
    cost = under_cost * under_units + over_cost * over_units
    return {
        "actual": actual_total,
        "absolute_error": absolute_error,
        "signed_error": signed_error,
        "wape_pct": 100 * absolute_error / actual_total,
        "bias_pct": 100 * signed_error / actual_total,
        "under_units": under_units,
        "over_units": over_units,
        "cost_usd": cost,
    }


def main() -> None:
    under_cost = 8.0  # USD per underforecast unit
    over_cost = 2.0   # USD per overforecast unit
    vendor = summarize(ROWS, "vendor", under_cost, over_cost)
    benchmark = summarize(ROWS, "benchmark", under_cost, over_cost)
    for name, result in (("vendor", vendor), ("benchmark", benchmark)):
        print(
            f"{name}: WAPE={result['wape_pct']:.2f}%, "
            f"bias={result['bias_pct']:.2f}%, "
            f"under={result['under_units']:.0f} units, "
            f"over={result['over_units']:.0f} units, "
            f"cost=USD {result['cost_usd']:.2f}"
        )
    saving = benchmark["cost_usd"] - vendor["cost_usd"]
    print(f"illustrative saving: USD {saving:.2f}")
    if benchmark["cost_usd"] > 0:
        print(f"relative cost reduction: {100 * saving / benchmark['cost_usd']:.2f}%")
    else:
        print("relative cost reduction: undefined (zero benchmark cost)")
    for row in ROWS:
        vendor_error = row.vendor - row.actual
        benchmark_error = row.benchmark - row.actual
        print(
            f"{row.series}: vendor error={vendor_error:+.0f} units, "
            f"benchmark error={benchmark_error:+.0f} units"
        )


if __name__ == "__main__":
    main()
```

Expected output:

```text
vendor: WAPE=10.00%, bias=-2.50%, under=25 units, over=15 units, cost=USD 230.00
benchmark: WAPE=10.00%, bias=-7.50%, under=35 units, over=5 units, cost=USD 290.00
illustrative saving: USD 60.00
relative cost reduction: 20.69%
A: vendor error=-10 units, benchmark error=-20 units
B: vendor error=+10 units, benchmark error=+5 units
C: vendor error=-15 units, benchmark error=-5 units
D: vendor error=+5 units, benchmark error=-10 units
```

### Reading and extending the calculation

WAPE has a positive denominator here because actual demand totals 400 units.
For an all-zero actual population, WAPE is undefined; report another appropriate measure rather than assigning zero.
If actuals can be negative, such as signed returns, decide a defensible metric and denominator rather than silently applying this function.
This example does not validate nonnegative forecasts; the production schema and policy must define whether negative predictions are allowed.

The arithmetic treats rows equally as physical observations but WAPE weights them by actual volume through the aggregate denominator.
For a subgroup, subset both forecasts and actuals consistently and show its denominator and count.
The program's row-level differences reveal cancellation but do not establish independent samples or statistical significance.
In a real dossier, preserve input snapshot IDs, forecast origins, outcome versions, software version, evaluation protocol, and exact vendor version.
Avoid copying sensitive commercial data into a public repository.

## 8. Data, standards, and regulatory context

### Historical SR 11-7 and the 2026 replacement

The Federal Reserve issued SR 11-7, *Guidance on Model Risk Management*, on 4 April 2011; the Office of the Comptroller of the Currency (OCC) issued parallel guidance as OCC Bulletin 2011-12.
The Federal Deposit Insurance Corporation (FDIC) adopted the 2011 guidance with technical changes in 2017.
The historical guidance is important to the development of model risk practice, including discussion of vendor products, validation, and effective challenge.
It was supervisory guidance, not a statute automatically binding all organizations that buy models.

On 17 April 2026, the Federal Reserve, OCC, and FDIC issued revised interagency model risk management guidance.
Federal Reserve [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) expressly supersedes and replaces SR 11-7 and SR 21-8.
[OCC Bulletin 2026-13](https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html) rescinds OCC Bulletin 2011-12 and other identified materials.
The [joint attachment](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) is the source for the revised principles and their scope.
Do not cite SR 11-7 as the current Federal Reserve guidance as of this topic's source-check date.

The revised guidance says it is expected to be most relevant to banking organizations with **over USD 30 billion in total assets**.
It can also be relevant to some organizations at or below that amount with significant model risk exposure, such as exposure arising from complex or prevalent models or activities outside traditional community banking.
The threshold is a statement of expected relevance, not a universal exemption from every applicable law or from sound risk management.
The OCC bulletin separately notes applicability to community banks subject to the guidance's limitations.
Identify the responsible regulator and institution before making a supervisory applicability determination.

The revised guidance explicitly states that it does not set enforceable standards or prescriptive requirements and that non-compliance with the guidance will not result in supervisory criticism.
Its footnote nevertheless notes that supervisory action may result for violations of law or unsafe or unsound practices stemming from insufficient management of model risk.
That distinction matters: “not itself enforceable” does not mean all model-related conduct is legally unconstrained.
Actual obligations depend on applicable law, regulations, orders, contracts, and facts; seek qualified advice.

For purposes of the revised guidance, a model is a complex quantitative method, system, or approach that applies statistical, economic, or financial theories to inputs to produce quantitative estimates.
The guidance excludes simple arithmetic and deterministic rule-based processes without those theories.
Its footnote excludes generative and agentic AI from the guidance's scope while stating that principles apply to traditional statistical and quantitative models and non-generative, non-agentic AI models.
Exclusion from this particular guidance does not imply an absence of other controls or legal obligations.
A course's broader operational use of “model” need not reproduce the regulatory definition in every context.

### Vendor product provisions

Section VII of the revised attachment recognizes that code, data, or methodology may be proprietary and unavailable to the bank.
It discusses validation by internal or outside parties, understanding conceptual soundness, design, development data and performance, and ongoing monitoring and outcomes analysis.
It also discusses documenting, justifying, and evaluating customization.
These are risk-based principles in the cited guidance, not a blanket claim that a bank must obtain every vendor's source code.
Nor do they prove a black box is adequately validated without evidence relevant to the particular use.

For a nonbank supply-chain firm, this topic borrows the reasoning as a governance pattern.
It does not assert that SR 26-2 applies to that firm, that its thresholds are legal safe harbors, or that every model must be reviewed annually.
Other sector-specific law, privacy obligations, contractual rights, and customer requirements may matter independently.
The organization should consult its legal and compliance teams about its own obligations.

### Data and technical standards

A data contract should specify item and location identifiers, units, time zone, forecast origin, target period, horizon, frequency, missingness, and revision policy.
A consistent identifier system can prevent mismatched joins, but an identifier standard does not establish that an actual is an uncensored demand observation.
Data-quality checks cannot prove representativeness unless the missing and excluded population is understood.

The [NIST AI Risk Management Framework 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), published in January 2023, is a voluntary framework for managing AI risk.
Its Govern, Map, Measure, and Manage functions offer a useful organizing vocabulary; using the framework is not proof of regulatory compliance.
The NIST [AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/) provides suggested actions, not a model certification.
A security assurance report and a model validation report answer different questions; request both when the use warrants them.

## 9. Industry practice and case evidence

### What official banking sources actually document

The revised interagency attachment explicitly recognizes that vendor model components may be proprietary and that banking organizations may lack the underlying code, data, or methodology.
It recommends understanding and validating the product and monitoring outcomes, including customer-specific adjustments.
This is evidence of a documented supervisory concern, not evidence that a particular supplier failed a supply-chain deployment.
The attachment's examples and scope concern banking organizations; transfer to a distributor's forecasting system is an analogy, not legal applicability.

The OCC's 2026 bulletin says practices suitable for one bank may not be suitable for another with a different risk profile or use.
It identifies vendor and other third-party products as a topic of the revised guidance.
It also explains that the guidance itself is not prescriptive or enforceable.
Those statements support risk-based tailoring, not an assertion that any specific validation frequency is universally mandated.

### Documented AI evaluation practice

NIST's AI Risk Management Framework describes governance, context mapping, measurement, and management as interconnected functions.
The framework calls for attention to deployment context, testing, transparency, and ongoing monitoring.
It is a general voluntary framework; it does not provide a supply-chain forecast accuracy result or certify a vendor.
Applying its functions to the fictional distributor is an instructional adaptation.

### Evidence hierarchy in procurement

A credible independent technical report with relevant scope is stronger evidence of the work actually performed than an unattributed marketing statement.
A current buyer-side out-of-time evaluation is more directly relevant to the buyer's forecasting population than an unrelated aggregate demonstration.
Neither evidence type automatically resolves every concern: a clean backtest may not reveal unlawful data use, and an assurance report may not measure forecast quality.
The strongest decision combines complementary evidence and records what each piece cannot establish.

Do not infer that a named business adopted this particular procedure or achieved the worked example's savings.
The distributor and numbers in section 6 are invented solely for instruction.
A vendor case study, if used in an actual procurement, should be identified as a vendor-published claim and assessed for population, timeframe, baseline, and independent corroboration.

### Operational pattern: a controlled shadow period

A buyer can obtain forecast outputs without using them to place orders and compare them with actuals and the existing planning process.
A shadow period can expose late predictions, missing fields, and integration failures as well as forecast errors.
Its coverage may be limited by season, sample size, and the inability to observe counterfactual orders or unobserved demand.
If a vendor tunes to shadow results, preserve a later untouched holdout for final evaluation.
A shadow success does not ensure production reliability or certify unseen model internals.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Marketing claim without denominator | “20% better” without a formula | Incomparable baseline or selected sample | False business case | Require metric, baseline, period, and population |
| Future leakage | Backtest uses revised data | Future information enters training or inputs | Inflated measured accuracy | Point-in-time snapshots and cutoff audit |
| Weak comparator | Baseline is an obsolete manual method | Improvement is against an unrealistic alternative | Wrong purchase choice | Compare with feasible current and simple baselines |
| Hidden exclusions | Missing predictions disappear from report | Hard cases are removed | Overstated performance | Reconcile input and output counts, report exclusions |
| Stale validation | Report names an earlier release | Code or data changed after review | Evidence does not cover production | Version mapping and change-impact assessment |
| Conflicted reviewer | Sponsor controls reviewer bonus | Incentives discourage challenge | Unresolved flaws accepted | Disclose conflicts and independent escalation |
| Proprietary opacity | Vendor declines all design and data information | Conceptual assumptions cannot be assessed | Unknown model risk | Request controlled access; limit or reject if material |
| Stockout-censored actuals | Sales fall when availability is zero | Observed sales understate demand | Apparent accuracy masks lost demand | Flag censored periods and estimate cautiously |
| Aggregate masking | Overall metric improves but critical items worsen | Volume weighting hides concentrated harm | Critical stockouts | Report material subgroup results and counts |
| Unannounced update | Endpoint changes without version notice | Model behavior changes outside review | Prior evidence becomes stale | Contract notice, shadow test, and fallback |
| Misread explanation | A feature attribution is presented as causal proof | Correlation is confused with intervention effect | Wrong planning action | Explain method limits and test interventions separately |
| Override loss | Planner changes are not logged | Forecast and final plan are conflated | Unreproducible decision and biased attribution | Log raw forecast, override, reason, and final plan |
| Missing fallback | Contract says “high availability” only | No usable alternative during outage | Missed planning cycle | Rehearse invocation and restoration |
| Security report substitution | SOC report is offered as accuracy validation | Control assurance is confused with model evidence | Unassessed forecast risk | Match each assurance to its scope |
| Unchecked subprocessor | Vendor changes hosting or subcontractors | Data exposure and dependencies change | Privacy or resilience incident | Review notice, permissions, and data flow |
| Small holdout overclaim | Four rows presented as proof | Sampling variability is ignored | Premature production approval | Longer independent test and uncertainty analysis |
| Cost proxy overreach | Unit error is called realized inventory loss | Policy and inventory dynamics omitted | Misleading ROI | Label proxy and replay actual policy when feasible |
| Permanent exception | Deadline repeatedly extended | Accountability and risk acceptance decay | Uncontrolled ongoing use | Named owner, expiry, closure evidence, escalation |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage it |
|---|---|---|---|
| Forecasts and actuals share units | Error arithmetic needs comparable quantities | Cases and individual units are mixed | Validate unit conversion and reconcile totals |
| Forecasts share origin and horizon | Comparator must face the same information | Benchmark sees later data | Preserve timestamps and availability snapshots |
| Holdout is independent of tuning | Test estimates performance after selection | Repeated vendor tuning on test outcomes | Lock protocol and reserve a later holdout |
| Actuals represent the target | Error measures meaningful prediction | Stockouts censor demand or actuals are revised | Flag availability and version outcome data |
| Test population resembles use | Results should inform deployment | New markets or product mix differ | Compare coverage and test material segments |
| Model release is stable | Evidence attaches to an identified system | Vendor silently retrains | Record release, detect changes, retest |
| Baseline is feasible | Incremental value depends on a real alternative | Baseline uses future judgment | Reconstruct the operational process at origin |
| Costs are decision relevant | Cost proxy is intended to express consequences | Holding, service, and lead-time effects differ | Review rates with finance and simulate policy |
| Reviewer can challenge | Validation needs objective authority | Sponsor controls evidence or suppresses findings | Document access, conflict, and escalation |
| Vendor evidence is authentic and scoped | Claims must relate to the deployed product | Report is stale, partial, or unrelated | Verify issuer, dates, versions, scope, and findings |
| Fallback is usable | Limited operation needs continuity | Data, staff, or access are unavailable | Conduct a timed fallback exercise |
| Data rights permit the test | Sharing history creates legal and commercial risk | Contract or privacy rules prohibit sharing | Review data use and permissions before transfer |
| Performance is sufficiently stable | Historical evidence informs future use | Market shift or structural break | Monitor outcomes and set revalidation triggers |

These are working assumptions, not guarantees.
Failure of an assumption can narrow a conclusion, require additional evidence, or make the intended use unacceptable.
A finite sample cannot prove absence of all defects, particularly inside an opaque hosted system.

## 12. Exercises

### Exercise 1 — Define the use

A vendor offers a daily delivery-delay prediction to a manufacturer.
Dispatchers propose using it for expedited freight and customer messaging.
The vendor says the output is a probability of late arrival but cannot initially define “late.”

**Deliverable:** a 150-word use statement and a list of five unresolved questions.
**Acceptance:** identify decision owner, unit, horizon, forecast origin, permissible actions, late definition, exclusions, and distinction between prediction and customer commitment.
**Time:** 15 minutes.

### Exercise 2 — Decompose a claim

The vendor claims “30% more accurate across all customers.”
No equation, baseline, period, sample size, or exclusions appear in the sales deck.

**Deliverable:** a five-row claim-to-evidence register.
**Acceptance:** request the formula, comparator, evaluation population, time window, and treatment of missing predictions; distinguish vendor assertion from verified evidence.
**Time:** 15 minutes.

### Exercise 3 — Assess a report

An outside firm reviewed vendor security controls 18 months ago.
The report excludes the forecasting algorithm and the current hosting subprocessor.
The vendor offers it as proof of model validity.

**Deliverable:** a short evidence assessment and three follow-up requests.
**Acceptance:** distinguish security scope from forecast validity, identify age and subprocessor gaps, and avoid claiming the report proves insecurity.
**Time:** 15 minutes.

### Exercise 4 — Design the test

You have two years of weekly item-location history, revised sales, and a vendor able to produce historical forecasts.
Promotions and stockouts vary by season.

**Deliverable:** a one-page out-of-time test protocol.
**Acceptance:** specify rolling origins, point-in-time inputs, baseline parity, holdout protection, missing-prediction reconciliation, subgroup counts, and limitations of censored sales.
**Time:** 30 minutes.

### Exercise 5 — Compute and interpret

Use the four observations in section 6 with USD 8 per underforecast unit and USD 2 per overforecast unit.

**Deliverable:** WAPE, signed bias, and cost for both methods, plus one subgroup concern.
**Acceptance:** show the denominator and error sign, calculate 10.0% WAPE for both, -2.5% and -7.5% bias, and USD 230 and USD 290 cost; identify series C.
**Time:** 20 minutes.

### Exercise 6 — Bound the black box

A vendor permits repeated queries and a redacted model description but refuses code and training-data access.
The system is proposed for automatic replenishment of safety-critical stock.

**Deliverable:** two lists of at least four verifiable and four unverifiable propositions, followed by a decision recommendation.
**Acceptance:** distinguish output tests from hidden data, implementation, and future behavior; justify any restriction or rejection against the materiality and evidence gaps.
**Time:** 20 minutes.

### Exercise 7 — Build independence

The head of procurement selects the model and proposes signing the validation, with a bonus linked to deployment.
A data scientist who tuned vendor settings offers to review the same settings.

**Deliverable:** a role and escalation matrix plus two conflict mitigations.
**Acceptance:** separate sponsor, technical reviewer, accountable owner, and audit where present; ensure access and authority to challenge and record unresolved conflicts.
**Time:** 20 minutes.

### Exercise 8 — Explain applicability

A supply-chain executive says, “SR 11-7 is a law requiring every vendor model to be validated annually.”

**Deliverable:** a correction in five sentences with current source and date.
**Acceptance:** identify the 2026 supersession, the banking context, guidance versus enforceable obligations, relevance threshold nuance, and no blanket annual rule.
**Time:** 15 minutes.

### Exercise 9 — Design lifecycle controls

The vendor can update a hosted model weekly and cannot freeze a version.
The business uses predictions every Monday morning.

**Deliverable:** a change, monitoring, incident, and fallback plan.
**Acceptance:** name owners, notice and identification needs, test or shadow mechanism, decision thresholds, incident escalation, fallback invocation, and revalidation triggers.
**Time:** 25 minutes.

## 13. Self-check answers

### 1. Use statement

A strong statement identifies the manufacturer, shipment or order unit, daily origin, delivery horizon, and precise late-arrival definition.
It explains whether a dispatcher may expedite, communicate a risk, or both, and who approves each action.
The probability is a forecast; it is not itself the promised delivery date or an instruction to purchase freight.
Questions include label definition, event timestamps, data availability, calibration, and whether the model covers the manufacturer's carriers and lanes.
A weak answer assumes the vendor's generic use description already defines these details.

### 2. Claim register

Ask whether “30%” is a relative change in error or a percentage-point difference in an accuracy score.
Ask for the exact formula and unit, the baseline and its information availability, and the evaluation period.
Request the number and identity or characteristics of participating customers and whether they were selected after results were known.
Request all exclusions, missing outputs, segment results, and the version tested.
Until those items are checked, the statement is a vendor claim rather than an established buyer-specific gain.

### 3. Report scope

A security-controls report can be useful evidence within its defined scope and reporting period.
It does not assess forecast correctness when the algorithm is excluded, and it says nothing conclusive about a subprocessor excluded from scope.
Request current assurance covering the actual service and subprocessor, model-specific validation and findings, and version-specific documentation.
The exclusions are evidence gaps; they are not proof of a security incident or model defect.

### 4. Test rubric

A strong protocol records each forecast origin and only the features available at that instant.
It compares the vendor and feasible baseline over identical item-location-weeks, horizons, and treatment of missing outputs.
It locks tuning before the final holdout, lists exclusions, and compares total submitted versus returned records.
It reports seasonal, promotion, and critical-item segments with counts and denominators.
It flags stockout-censored sales and identifies where true demand is unknown.
It defines approval thresholds and acknowledges sampling and dependence before seeing the final results.

### 5. Arithmetic

Actual total is 400 units and absolute error is 40 units for each method, so both WAPEs are 10.0%.
Vendor signed error is -10 units, giving -2.5%; benchmark signed error is -30 units, giving -7.5%.
Vendor cost is `25 × 8 + 15 × 2 = USD 230`.
Benchmark cost is `35 × 8 + 5 × 2 = USD 290`.
The USD 60 difference is a 20.69% reduction relative to USD 290, under the stated proxy.
Series C is USD 80 worse for the vendor, which matters if it is a critical item.

### 6. Black-box rubric

The reviewer can verify returned outputs, observed error, missing predictions, response time, and behavior on chosen input perturbations.
The reviewer cannot infer complete training-data provenance, correctness of all hidden code, absence of all leakage, or future version behavior from those observations alone.
A redacted design document may supply useful partial evidence, depending on its scope and credibility.
Because safety-critical automatic action can have material consequences, the owner should not treat the given access as sufficient by default.
Request stronger assurance and access, impose effective independent controls, or reject the proposed use when the residual uncertainty is unacceptable.

### 7. Independence rubric

Procurement sponsors and negotiates; the model owner accepts the use within delegated authority.
A competent reviewer without the purchase-linked bonus challenges evidence and documents conflicts.
The person who tuned settings can explain decisions but should not alone provide independent sign-off on those same settings.
If internal audit is involved, it examines governance effectiveness rather than substituting for every validation test.
The reviewer needs access to records and a route to escalate unresolved disagreement beyond the sponsor.
An outside specialist may help, provided conflicts and the buyer's oversight are addressed.

### 8. Regulatory correction

SR 11-7 was U.S. banking supervisory guidance, not a law automatically applicable to every supply-chain firm.
On 17 April 2026, Federal Reserve SR 26-2 superseded and replaced SR 11-7 and SR 21-8.
The revised interagency guidance is expected to be most relevant to banking organizations over USD 30 billion in total assets, with stated circumstances where smaller organizations may also find it relevant.
It says it is not enforceable or prescriptive; separate legal requirements and unsafe or unsound practices may still matter.
It does not impose a blanket annual validation rule on every vendor model.
See the official SR 26-2 letter and attachment and OCC Bulletin 2026-13 in section 16.

### 9. Lifecycle rubric

Assign a buyer model owner and vendor contact for release notifications and incident response.
Capture release identifiers or, if unavailable, the vendor's change records and observed output snapshots.
Compare updates in shadow where possible, review performance and input quality, and define an escalation threshold appropriate to the business.
Specify the person who can pause use, the process to invoke the tested baseline before Monday's deadline, and a communication path to planners.
Revalidate when changes, material performance deterioration, incidents, or expanded uses make prior evidence insufficient.
A service that cannot freeze versions requires stronger operational controls; a promise to “monitor” without ownership or action is not enough.

## 14. Interview questions

1. **What does a vendor validation report establish?** A strong answer names its scope, date, reviewer, version, methods, and limitations; a weak answer treats the title as universal proof.
2. **Why test on buyer data?** A strong answer connects population, horizon, process, and outcome differences; a weak answer assumes every customer is identical.
3. **How do you prevent leakage?** A strong answer describes origin-time availability, locked tuning, and revised-data handling; a weak answer relies only on a random split.
4. **What is a fair benchmark?** A strong answer uses a feasible alternative at the same decision time and population; a weak answer picks any inferior historical number.
5. **Can a black box be evaluated?** A strong answer separates observable outcome and interface tests from hidden internals and residual uncertainty; a weak answer says either everything or nothing can be known.
6. **What makes a reviewer independent?** A strong answer includes competence, objectivity, access, conflicts, and authority; a weak answer cites a reporting-line label alone.
7. **Why can equal WAPE lead to different business cost?** A strong answer discusses signed errors, asymmetric costs, and decision policy; a weak answer assumes every error has the same consequence.
8. **How do you handle a vendor update?** A strong answer covers version identification, impact assessment, testing, approval, monitoring, and fallback; a weak answer assumes the old validation is permanent.
9. **What does SR 11-7 mean in 2026?** A strong answer explains supersession by SR 26-2, banking scope, and guidance status; a weak answer calls it current universal law.
10. **When is conditional approval inappropriate?** A strong answer considers material unknowns, control effectiveness, authority, and enforceable deadlines; a weak answer treats a promise of future documentation as a control.

## 15. Key terms

- **Actual:** observed target outcome, potentially censored or revised; see sections 4 and 11.
- **Audit:** independent assessment of a process or controls, distinct from model validation; see section 5.
- **Baseline:** feasible alternative available at the forecast origin; see section 5.
- **Bias:** signed error summary; see section 6.
- **Black box:** model with inaccessible or insufficiently understood internals; see sections 3 and 5.
- **Change trigger:** event that prompts review of prior evidence; see section 5.
- **Conceptual soundness:** suitability of assumptions, design, and data; see section 3.
- **Cost proxy:** simplified monetary expression of errors, not necessarily realized loss; see section 6.
- **Data lineage:** trace from source through prediction and outcome; see section 4.
- **Drift:** change in inputs, relationships, outputs, or performance; see section 5.
- **Effective challenge:** expert objective critique with influence; see section 3.
- **Fallback:** tested alternative when the model or service is unavailable; see section 5.
- **Forecast:** prediction of an uncertain outcome, not a plan or target; see section 2.
- **Forecast origin:** time at which the prediction is made; see section 4.
- **Holdout:** data withheld from selection and tuning for evaluation; see section 5.
- **Horizon:** interval between forecast origin and target; see section 4.
- **Materiality:** importance of exposure and purpose; see section 5.
- **Model owner:** buyer accountable for use and controls; see section 4.
- **Out-of-time test:** evaluation using information available before subsequent outcomes; see section 5.
- **Overlay:** documented adjustment to model output; see section 4.
- **Residual risk:** uncertainty and risk remaining after controls; see section 5.
- **Shadow evaluation:** comparison without using predictions to drive the live decision; see section 9.
- **SR 11-7:** superseded 2011 Federal Reserve supervisory guidance; see section 8.
- **SR 26-2:** 2026 Federal Reserve letter issuing revised interagency guidance; see section 8.
- **Subprocessor:** party processing data on the vendor's behalf; see section 4.
- **Validation:** assessment of reliability, limitations, and fitness for a defined use; see section 3.
- **Version:** identified model and configuration release; see section 5.
- **WAPE:** total absolute error divided by total actuals; see section 6.

## 16. Sources

1. Federal Reserve, [SR 26-2, Revised Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm), 17 April 2026. Official supersession and Federal Reserve applicability statement.
2. Federal Reserve, FDIC, and OCC, [Supervisory Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf), 17 April 2026, especially sections I–II, V, and VII. Official joint guidance, including scope and vendor products.
3. OCC, [Bulletin 2026-13, Model Risk Management: Revised Guidance](https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html), 17 April 2026. Rescissions, community-bank note, exclusions, and guidance status.
4. NIST, [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), NIST AI 100-1, January 2023. Voluntary risk management framework.
5. NIST, [AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/), online resource. Suggested actions organized by framework function.

The 2026 official sources are primary sources for the supersession and scope claims; the NIST material is a voluntary framework, not a banking rule.
The fictional calculation and purchasing decision are instructional examples, not reported industry results.
Recheck the current official publications and applicable jurisdiction before using this material for a real legal or supervisory decision.

## 17. Links

- [Module 26 index](_index.md).
- [Model inventory and tiering](01_model-inventory-tiering.md).
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md).
- [Model documentation](06_model-documentation.md).
- [Independence and validation](08_validation-independence.md).
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md).
- [Module 30 lab index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md), including the demand-forecast pipeline lab listed there; topic availability is tracked in that index.
- [Module 30 lab survey](../../../course/30_simulation_optimization_production_labs.md).
