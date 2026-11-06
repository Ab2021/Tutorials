---
case_id: CASE-03-01-005
case_title: "Triage a shifted delivery-risk model without confusing drift with harm"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: proposed
content_status: draft
case_type: incident
domain: production ML monitoring
primary_pattern: model-monitoring
secondary_patterns: [distribution-shift, label-delay]
evidence_level: illustrative
source_basis: illustrative
as_of_date: "2026-11-07"
owner_role: ML on-call lead
source_index: INDEX.md
review_trigger: "Change in outcome maturity, serving contract, or intervention policy"
---
# Triage a shifted delivery-risk model without confusing drift with harm

## 1. The One-Line Framing
Given each accepted delivery order and its point-in-time features, decide whether the operations lead should continue, contain, or roll back a late-delivery prioritization model, against the prior stable model, under limited intervention capacity and delayed outcomes.
The decision is hourly incident triage, not a decision to refuse an order.
A successful response preserves safe dispatch, determines whether a data contract failed, and quantifies mature-cohort quality before any permanent model change.
All entities, incidents, dates, counts, thresholds, interventions, and architecture are invented teaching assumptions; nothing was deployed, observed, measured, or tested.
A feature distribution shift is evidence of changed inputs, not proof of worse prediction or causal business harm.

## 2. Context & Background
A hypothetical delivery operations team ranks accepted orders for a limited proactive-assistance queue.
An agent can contact a carrier or customer; an order is not automatically cancelled or delayed by the score.
The order is scored once at acceptance and may be rescored only on an explicit versioned event.
The baseline is a previously approved model with the same queue capacity, input contract, and eligibility rules.
The model estimates the probability that an order arrives more than 30 minutes after its promised time.
A higher score gives an order higher priority, subject to deterministic eligibility and capacity rules.
The fictional incident affects one region and one channel over two consecutive hourly scoring windows.
A new upstream merchant campaign is a possible mix change; a bad distance feature release is a competing explanation.
Dispatch continues while the lead investigates, with manual escalation available for critical orders.
Operations owns queue decisions; the ML on-call owns diagnosis; data engineering owns feature feeds.
The product owner approves rollback, and the incident commander owns communication and closure.
No actual company, commercial benchmark, regulation, or contractual service level is asserted.

## 3. The Business Question
Illustrative stakeholder question: should the team revert the current model now, or continue it while investigating the distance shift?
Decision unit: current model and feature pipeline for the affected region and channel.
Decision deadline: within one hour of the second qualifying alert.
Options: maintain with observation; contain by routing eligible orders to the prior approved model; suspend scoring and use a documented manual queue.
Primary criterion: maintain reliable prioritization without increasing avoidable late deliveries.
Immediate safety guardrails include feature validity, queue availability, and capacity; mature labels are required for predictive-quality claims.
A single feature alert cannot identify whether intervention caused a better or worse delivery outcome.
The owner can choose a reversible containment before causal certainty when the feature contract is broken.
Retraining, changing eligibility, and rolling out a new business policy are outside the bounded incident decision.

## 4. Assumptions & Constraints
All times below are UTC and all counts are illustrative, not a historical event.
The reference model, preprocessing artifacts, schema, and threshold are assumed recoverable and compatible.
The reference period is the previous four complete comparable weeks, stratified by weekday and hour.
The incident cohort consists of 1,000 unique eligible orders in each of two complete hourly windows.
Each score is associated with an immutable order ID, scoring time, model version, feature version, and policy version.
The illustration uses a daily eventual label maturity horizon of seven days after promised delivery.
After that horizon a label is final only if the source feed is complete and the order has a resolved outcome.
Orders with unresolved or cancelled outcomes are reported separately; they are not silently called on time.
Capacity is 100 assistance slots per 1,000 eligible orders, with 10 slots reserved for manual escalation.
The remaining 90 ranked slots are allocated using the model and deterministic tie-breaking.
No automated action changes the delivery promise, and the model output is advisory.
The upstream distance is expected in kilometres, nonnegative, and as-of scoring time.
A hypothetical producer release occurred shortly before the alert; its existence does not establish causality.
The example assumes observations can be joined consistently across model, event, feature, and outcome stores.
Unknown queue effectiveness, selection bias, and unit economics limit any business-value estimate.
The assumed alert limits are starting operating proposals, not universal statistical constants.

## 5. Data, Tools & Method
### 5a. Data Specification
The prediction grain is one eligible order at first acceptance; duplicate ingestion is deduplicated by order ID and scoring event ID.
The outcome grain is the same order, resolved as late or on time after its promised-delivery cutoff.
The feature contract includes distance in kilometres, merchant region, order channel, acceptance timestamp, and feature availability timestamp.
Each prediction record includes score, model artifact digest, preprocessing digest, feature schema version, and decision policy version.
The queue event records rank, capacity, manual override, contact attempt, and contact outcome without assuming effectiveness.
The label event includes source ID, event time, ingestion time, resolution status, and correction version.
The cohort manifest freezes eligible IDs and joins by order ID; later corrections are versioned instead of overwriting history.
Excluded orders include invalid identifiers, those outside the region or channel, and orders never eligible for scoring.
Cancellation and unresolved labels are tracked as separate outcomes with counts and rates by cohort.
Data ownership and access are role-limited; only operationally necessary identifiers enter dashboards.
### 5b. Tools & Stack
A generic event collector writes prediction and queue records to an append-only log.
A schema registry or equivalent contract gate validates units, ranges, required fields, and versions.
A scheduled batch job aggregates fixed bins and label-matured cohorts into an auditable metric store.
A dashboard displays current, reference, counts, completeness, model version, and confidence context.
An alert router pages the on-call for sustained breaches and records acknowledgement and resolution.
The precise vendor, implementation, and performance are unspecified; no vendor capability is presumed.
### 5c. Method Summary
Freeze the reference bin boundaries and compare the current feature histogram with the comparable reference.
Compute the population stability index (PSI), a descriptive distribution-divergence statistic, on the same bins.
Inspect missingness, units, preprocessing, and cohort composition before interpreting a model score change.
Compare realized model quality only on sufficiently mature and complete labeled cohorts.
Compare current and previous models on the same cases where both scores are available, while noting policy-selection limits.
Keep business impact causal claims separate from observational model metrics.
### 5d. Provenance, Quality, Exclusions & Leakage
The directory inventory names `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/04_Model_Monitoring_and_Drift.md` as topic provenance.
The source title is a scope pointer, not evidence that any detailed source text or deployment was examined.
The [focused teaching plan](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-ml-design) supplies the mapped home and authoring context.
The [authoring standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) define evidence and numerical requirements.
The case contains constructed data rather than external operational evidence; there are no execution artifacts.
Feature timestamps must precede or equal scoring time; labels and post-delivery information must never enter features.
Cross-region records, duplicate scores, inaccessible outcomes, and incomplete label feeds are counted and excluded according to a frozen manifest.
Reference and current periods must use the same definition, window alignment, bin edges, and inclusion rules.
Late-arriving and corrected events must trigger a versioned recomputation of affected cohort metrics.
### 5e. Reproducibility & Method-Selection Ledger
Record reference cohort query, timezone, bins, feature transformation, current query, and dependency versions.
Persist counts before normalization, cohort manifest hash, and alert configuration with every metric run.
The PSI choice summarizes binned univariate shift; it does not diagnose joint shift or prediction quality.
A raw quantile plot and missingness rate complement PSI; a schema gate detects impossible inputs sooner.
A held-out mature-cohort quality comparison is preferred to training loss for operational decision support.
A randomized holdout can estimate incremental intervention value when safe and authorized; it is not assumed here.
Alternative drift tests, thresholds, and reference windows require calibration on past stable and known-change windows.
The reference window and planned tests have not been evaluated on real data.

## 6. Analysis Walkthrough
### Constructed timeline and competing hypotheses
At 09:00 the producer release is assumed to begin for the affected channel.
At 10:00 the first completed window exceeds the proposed distance-shift threshold.
At 11:00 the second window exceeds the threshold and creates a page under the proposed rule.
The on-call checks ingestion, eligibility counts, distance units, freshness, and feature transformation before changing the model.
One hypothesis is a legitimate campaign-driven shift in delivery distance.
A second hypothesis is a kilometres-to-metres mismatch introduced by the producer.
A third hypothesis is an incorrect join or default that changes the distribution without a unit conversion.
The hypothetical evidence available at 11:00 is the histogram below; it alone cannot select among these hypotheses.
Assume a later contract inspection reveals a metres value in a kilometres field for affected events.
That assumed inspection supports containment of the data-contract failure, not a measured predictive-quality change.
### Checked distribution example
Use fixed distance bins of 0–2 km, over 2–5 km, and over 5 km.
Reference counts among 1,000 orders are 500, 300, and 200.
Current counts among 1,000 orders are 300, 300, and 400.
Reference proportions p are 0.5, 0.3, and 0.2; current proportions q are 0.3, 0.3, and 0.4.
PSI is the sum over bins of (q − p) × natural logarithm(q / p).
First bin: (0.3 − 0.5) × ln(0.3 / 0.5) = 0.102165.
Second bin: (0.3 − 0.3) × ln(0.3 / 0.3) = 0.
Third bin: (0.4 − 0.2) × ln(0.4 / 0.2) = 0.138629.
The sum is approximately 0.240794, dimensionless, for this feature and window.
Independent check: 0.5 + 0.3 + 0.2 = 1 and 0.3 + 0.3 + 0.4 = 1.
Each contribution is nonnegative because the difference and log ratio have the same sign.
With the illustrative warning threshold 0.10 and page threshold 0.20, this window crosses both.
The page additionally requires a second consecutive complete hourly window above 0.20, assumed here.
Bins with zero counts need a predeclared smoothing rule and renormalization, not division by zero.
Smoothing changes the estimate and must be reported; no smoothing is needed for these counts.
The first window alone is a warning and does not meet the two-window page condition.
### Sensitivity and mature-label calculation
If current counts instead were 400, 300, 300, proportions would be 0.4, 0.3, 0.3.
The resulting PSI is 0.1 × ln(1.25) + 0.1 × ln(1.5) = 0.062861.
That alternative falls below 0.10, showing the operational decision depends on shift magnitude and fixed bins.
For a separate fully mature evaluation cohort of 1,000 scored orders, assume 200 late and 800 on time.
At a fixed illustrative score cutoff, suppose true positives are 120, false negatives 80, false positives 180, and true negatives 620.
The four cells sum to 1,000; 120 + 80 = 200 actual late; 180 + 620 = 800 actual on time.
Recall for late orders is 120 / 200 = 0.60; precision among predicted late is 120 / 300 = 0.40.
The flagged fraction is 300 / 1,000 = 0.30; this is not the 90-slot capacity-constrained queue.
If an assumed comparable previous model had recall 0.70 at the same cutoff, the difference is −0.10 or −10 percentage points.
A cutoff comparison alone is not a top-90 ranking evaluation and says nothing about incremental intervention benefit.
The example counts are constructed; no actual quality degradation has been established.
Uncertainty requires matched-order paired intervals and completeness checks; no confidence interval is asserted from invented comparisons.

## 7. The Finding
CLM-01: the constructed histogram yields PSI approximately 0.241 and crosses the proposed sustained-page limit when repeated.
CLM-01 is an arithmetic conclusion supported by Section 6, not an observation of a deployed system.
CLM-02: in the hypothetical branch where unit mismatch is confirmed, the feature contract is broken and containment is justified.
CLM-02 does not establish how much model performance or customer outcomes changed.
CLM-03: the separate constructed confusion matrix yields recall 0.60 and precision 0.40 under its stated cutoff.
CLM-03 is not a measured change and cannot be attributed to the unit error.
CLM-04: true business impact requires a defensible comparison of outcomes under alternative interventions.
Section 9 specifies the unexecuted evaluation and Section 11 limits the outcome claim.

## 8. System/Solution Design
### Input-to-output architecture
Order acceptance → point-in-time feature lookup → contract validation → versioned model scoring → eligibility and rank → bounded assistance queue.
In parallel, prediction and feature metadata → aggregation → baseline comparison → alert router → on-call decision.
Delivery outcome → maturity and completeness gate → versioned label join → quality and calibration dashboard.
Audit state stores cohort manifests, feature schema, model and policy versions, alert decisions, and rollback approvals.
The online path must not depend on the monitoring dashboard to produce a queue decision.
A contract violation routes to the approved fallback only if its inputs and preprocessing are independently valid.
If both current and prior model require the same broken distance, switch to the documented manual queue rather than blindly rolling back.
An invalid record is recorded with a reason code and no invented valid score.
An idempotent routing key prevents a retry from consuming a second assistance slot.
Successful requests carry model and feature lineage through to the queue and the later outcome join.
Failed requests trigger bounded retry only for transient errors, then the explicit fallback and an operator alert.
The incident commander records the effective change time, eligible population, and rollback target.
### Alert contract and ownership
Distance PSI warning: at least 1,000 eligible records in one complete hour and PSI ≥ 0.10.
Distance PSI page: at least 1,000 eligible records in each of two consecutive complete hours and PSI ≥ 0.20 in both.
A breach of the nonnegative kilometre schema can page immediately because it indicates contract failure, independently of PSI.
A missingness increase of at least 5 percentage points against comparable reference also pages after two complete windows.
An incomplete, late, or undersized monitoring window is shown as unknown and pages the data owner after a one-hour freshness breach.
Suppress duplicate alerts for the same incident fingerprint, while preserving all underlying windows and changes.
Acknowledge within 15 minutes and assign an incident lead; escalate to the operations lead after 30 minutes unacknowledged.
Recovery requires two complete compliant hours, valid source contract evidence, and a deliberate owner-approved restoration.
These are proposed operating limits requiring calibration, not statistical guarantees or measured service commitments.
### Capacity and operational readiness
For 1,000 orders per hour, monitoring ingests approximately 1,000 prediction records per hour before retries and metadata.
With 90 model-ranked slots and 10 reserved manual slots, capacity is 100 per 1,000 orders, or 10%.
At 2,000 orders per hour with fixed 100 hourly slots, coverage becomes 5%, and the queue policy must explicitly choose whom to exclude.
Average throughput does not establish burst tolerance, p95 latency, storage, staffing, or availability.
Capacity, retry amplification, queue backlog, and the fallback's own dependencies need load testing before use.
No price, latency, or uptime claim is made because no workload measurements or dated prices are available.

## 9. Evaluation & Validation
The primary predictive metric is recall at the actual top-90 model-ranked eligible orders per 1,000, computed against matured late labels across all eligible orders.
Its numerator is the number of truly late orders in the selected top 90; its denominator is the total number of truly late eligible orders in the same mature cohort.
Report precision at 90 as selected truly late orders divided by 90 when exactly 90 scored eligible orders exist.
Manual reserved slots are reported separately; do not include their outcomes in model-ranked precision.
The comparator is the prior model scored on the same point-in-time features and same eligible cases, when valid and available.
A matched-order bootstrap or other paired interval must reflect clustering by merchant or day when material.
Calibration compares score bins with observed late proportions, including bin counts and uncertainty.
Label completeness is resolved eligible orders divided by all eligible orders, reported by age, segment, and policy.
Require at least seven days of maturity and 95% resolved labels before a headline cohort metric; report the remaining unresolved fraction.
The 95% criterion is an illustrative minimum, not evidence that missing outcomes are unbiased.
A sensitivity analysis bounds missing labels as all late versus all on time and compares missingness by score and segment.
Use the same eligibility, promised-time definition, horizon, and event cutoff for current and reference cohorts.
For intervention impact, plan a safe randomized holdout of assistance among eligible orders where ethics and operations permit.
Compare intention-to-treat late-delivery rates, with assignment probabilities, spillovers, and capacity policy documented.
Without such design or adequate causal assumptions, observed assisted versus unassisted outcomes are confounded by priority.
If intervention changes the label, quality is policy-dependent; use randomized evidence or explicitly qualify the estimand.
Planned checks include unit injection, point-in-time joins, deduplication, delayed-event replay, cohort backfill, and fallback exercises.
Planned tests also include drift without degradation and degradation without univariate feature drift.
None of the proposed tests, intervals, experiments, or operational exercises has been executed.

## 10. Action Taken
No operational action was taken; this is an illustrative design exercise.
In the hypothetical unit-mismatch branch, the proposed action is to quarantine invalid feature records and invoke a verified compatible fallback.
The operations lead would maintain manual review for orders without a valid fallback score.
The data owner would revert or fix the producer contract, preserve affected events, and rebuild corrected feature records.
The ML lead would inspect model and preprocessing lineage before any rollback or restoration.
The incident commander would publish the affected window, uncertainty, mitigation, and next update time.
Retraining is deferred because a corrupted input contract cannot be repaired reliably by blindly fitting more corrupted data.
After correction, the owner would require the recovery gates in Section 8 and explicitly authorize restoration.
If only legitimate mix shift were found, the proposed action would instead be continued observation and mature-cohort evaluation.

## 11. Outcome & Results
There is no measured outcome, real incident record, deployment, or verified intervention benefit.
The numerical outputs in Section 6 are illustrative arithmetic, not results from a system or simulation run.
The hypothetical contract inspection demonstrates how the response would change under an assumed observation.
No claim is made that rollback improves late deliveries, recall, precision, cost, or fairness.
The next evidence would be contract audit artifacts, corrected cohort data, mature labels, and a controlled effectiveness evaluation.
Track both incident duration and affected eligible orders only after real event boundaries are established.
A restored valid feature is a technical recovery condition, not proof of restored business performance.

## 12. Failure Modes & What Went Wrong
Nothing actually went wrong: the entries below are hypothetical diagnostic drills, not observed failures.
For every drill the signal distinguishes candidate causes imperfectly and the prescribed checks remain unexecuted.
### Diagnostic drill 1: Unit conversion
- Candidate mechanism: distance producer sends metres as kilometres.
- Discriminating clue: recent producer version correlates with values roughly 1000 times expected.
- First investigation: compare raw payload, schema units, and producer release.
- Immediate bounded mitigation: quarantine and repair conversion.
- Preventive control: contract tests with unit metadata.
- Evidence to preserve: raw producer payload.
- Expected signature: distance transformed by exactly 1,000.
- Competing explanation: new parser rather than merchant geography.
- Discriminating comparison: compare 100 source and serving pairs.
- Decision invariant: metres must convert once to kilometres.
- Containment choice: reject producer version with missing unit.
- Containment side effect: manual handling increases queue workload.
- Proposed replay: replay fixed values through both versions.
- Accountable role: producer owner signs unit contract.
- Ongoing monitor: watch transformed-distance quantiles.
- Recovery criterion: two compliant complete windows.
- Interpretation limit: a compliant histogram cannot prove late recall recovered.
- Audit artifacts: versioned conversion examples.
- Diagnostic caution: inspect negative and extreme values separately.
### Diagnostic drill 2: Natural campaign mix
- Candidate mechanism: campaign expands service to distant merchants.
- Discriminating clue: shift persists in valid raw values but concentrates in campaign orders.
- First investigation: stratify by campaign and verify raw kilometres.
- Immediate bounded mitigation: continue with monitored cohort.
- Preventive control: seasonal and campaign-aware references.
- Evidence to preserve: campaign assignment.
- Expected signature: distance shift concentrated in newly promoted merchants.
- Competing explanation: seasonal change rather than unit mismatch.
- Discriminating comparison: compare raw units inside and outside campaign.
- Decision invariant: matched merchant strata should retain plausible ranges.
- Containment choice: keep model under enhanced observation.
- Containment side effect: new merchant samples may be too small.
- Proposed replay: compare performance after label maturity.
- Accountable role: product owner confirms campaign exposure.
- Ongoing monitor: watch cohort composition by merchant.
- Recovery criterion: two stable matched-reference windows.
- Interpretation limit: stable model quality does not imply campaign benefit.
- Audit artifacts: campaign definition and eligible IDs.
- Diagnostic caution: avoid choosing a reference after seeing the result.
### Diagnostic drill 3: Stale feature
- Candidate mechanism: feature cache returns old distances.
- Discriminating clue: feature age rises while raw producer is current.
- First investigation: compare event and availability timestamps.
- Immediate bounded mitigation: route affected orders to valid fallback.
- Preventive control: freshness SLO and expiration tests.
- Evidence to preserve: feature age.
- Expected signature: old timestamps on cache hits.
- Competing explanation: stale cache rather than changed merchants.
- Discriminating comparison: compare cache entries to authoritative producer.
- Decision invariant: source timestamp should precede scoring within allowed age.
- Containment choice: bypass stale cache where safe.
- Containment side effect: source traffic and latency can rise.
- Proposed replay: inject expired entries in replay.
- Accountable role: cache owner defines expiration.
- Ongoing monitor: watch cache hit age and source latency.
- Recovery criterion: freshness gate passes on both paths.
- Interpretation limit: fresh values may still have wrong units.
- Audit artifacts: cache key and feature timestamp.
- Diagnostic caution: never overwrite event time with ingestion time.
### Diagnostic drill 4: Default value spike
- Candidate mechanism: failed lookup fills distance with zero.
- Discriminating clue: zero count rises with lookup failures.
- First investigation: join lookup errors to zero-valued events.
- Immediate bounded mitigation: remove silent default and fallback.
- Preventive control: missing-value contract gate.
- Evidence to preserve: lookup response code.
- Expected signature: zeros correlate with failed lookups.
- Competing explanation: genuine zero-distance orders rather than defaulting.
- Discriminating comparison: inspect raw lookup failures and default settings.
- Decision invariant: missing distance must be distinct from numeric zero.
- Containment choice: stop silent defaulting for failed lookups.
- Containment side effect: more orders enter fallback queue.
- Proposed replay: replay successful zero and failed lookup separately.
- Accountable role: feature owner approves missing-value policy.
- Ongoing monitor: watch missing and zero rates separately.
- Recovery criterion: failed lookups never produce valid zero.
- Interpretation limit: reduced zero rate does not establish score quality.
- Audit artifacts: raw response status and null marker.
- Diagnostic caution: check whether training used the same missing convention.
### Diagnostic drill 5: Join fanout
- Candidate mechanism: merchant table has duplicate keys.
- Discriminating clue: prediction event count exceeds unique eligible orders.
- First investigation: count keys at each join and inspect duplicates.
- Immediate bounded mitigation: deduplicate and recompute windows.
- Preventive control: join cardinality assertions.
- Evidence to preserve: join key multiplicity.
- Expected signature: multiple merchant matches per order.
- Competing explanation: legitimate rescoring rather than join fanout.
- Discriminating comparison: group by order and scoring event before aggregation.
- Decision invariant: one scoring event has one feature row.
- Containment choice: restore a unique dimension key.
- Containment side effect: removing duplicates can conceal conflicting source values.
- Proposed replay: replay conflicting duplicate keys.
- Accountable role: data engineer owns cardinality contract.
- Ongoing monitor: watch rows per scoring event.
- Recovery criterion: no unexplained multiplicity in affected batch.
- Interpretation limit: deduplication alone cannot identify the correct row.
- Audit artifacts: dimension effective-time and version.
- Diagnostic caution: retain conflicting rows for investigation.
### Diagnostic drill 6: Join loss
- Candidate mechanism: feature join drops unmatched orders.
- Discriminating clue: eligible IDs lack prediction records.
- First investigation: anti-join order manifest to score records.
- Immediate bounded mitigation: restore missing records and use manual queue.
- Preventive control: completeness alert by cohort.
- Evidence to preserve: anti-join counts.
- Expected signature: eligible order IDs absent from score output.
- Competing explanation: ineligible orders incorrectly entering the manifest.
- Discriminating comparison: reconcile eligibility at the original decision time.
- Decision invariant: every eligible order has score or explicit fallback reason.
- Containment choice: restore dropped orders through approved route.
- Containment side effect: late restoration may miss assistance opportunity.
- Proposed replay: inject an unmatched merchant into the join.
- Accountable role: operations owner reviews affected order list.
- Ongoing monitor: watch unmatched IDs and queue coverage.
- Recovery criterion: all eligible IDs reconcile by disposition.
- Interpretation limit: reconciled counts do not recover missed past actions.
- Audit artifacts: eligibility manifest and fallback log.
- Diagnostic caution: separate no-score from low-score outcomes.
### Diagnostic drill 7: Wrong reference window
- Candidate mechanism: holiday baseline compared to ordinary weekday.
- Discriminating clue: shift disappears against a matched weekday reference.
- First investigation: compare fixed and season-matched references.
- Immediate bounded mitigation: annotate and avoid changing model.
- Preventive control: documented reference selection.
- Evidence to preserve: calendar metadata.
- Expected signature: reference includes holiday while current day does not.
- Competing explanation: campaign shift rather than calendar mismatch.
- Discriminating comparison: compare comparable historical weekday and holiday cohorts.
- Decision invariant: reference choice must be fixed before evaluation.
- Containment choice: annotate the comparison and preserve original alert.
- Containment side effect: changing baseline may hide an actual incident.
- Proposed replay: backtest both reference definitions on prior windows.
- Accountable role: monitoring owner approves comparison schedule.
- Ongoing monitor: watch reference sample count and season mix.
- Recovery criterion: scheduled reference reproduces from stored query.
- Interpretation limit: a matched calendar does not eliminate all confounding.
- Audit artifacts: baseline query and cohort hash.
- Diagnostic caution: never reset reference merely to silence alerts.
### Diagnostic drill 8: Current-window truncation
- Candidate mechanism: aggregator computes before late events arrive.
- Discriminating clue: sample size increases after watermark closes.
- First investigation: inspect watermark and event arrival histogram.
- Immediate bounded mitigation: mark window incomplete and recompute.
- Preventive control: watermark and freshness gates.
- Evidence to preserve: watermark lag.
- Expected signature: hourly count rises after aggregation.
- Competing explanation: upstream traffic genuinely arriving late.
- Discriminating comparison: compare event time with ingestion time.
- Decision invariant: complete windows close only after configured watermark.
- Containment choice: mark provisional windows unknown.
- Containment side effect: waiting longer delays incident response.
- Proposed replay: replay events on both sides of watermark.
- Accountable role: stream owner sets allowed lateness.
- Ongoing monitor: watch late-event fraction and watermark age.
- Recovery criterion: late corrections create a new metric version.
- Interpretation limit: a complete ingestion window can still contain bad values.
- Audit artifacts: event-time and ingestion-time histograms.
- Diagnostic caution: do not mix partial and finalized windows.
### Diagnostic drill 9: Label feed delay
- Candidate mechanism: outcomes fail to land on time.
- Discriminating clue: mature cohort completeness drops across scores.
- First investigation: compare source and sink counts by event age.
- Immediate bounded mitigation: suspend quality interpretation.
- Preventive control: feed-lag alert and replay.
- Evidence to preserve: source-to-sink lag.
- Expected signature: resolved source outcomes missing downstream.
- Competing explanation: orders genuinely not yet delivered.
- Discriminating comparison: reconcile IDs and outcome statuses at source.
- Decision invariant: mature metric requires resolved source and successful ingestion.
- Containment choice: pause quality alert interpretation.
- Containment side effect: quality incidents may remain undetected longer.
- Proposed replay: replay known delayed label messages.
- Accountable role: label owner coordinates feed restoration.
- Ongoing monitor: watch completeness by promised-date cohort.
- Recovery criterion: resolved source records reconcile at sink.
- Interpretation limit: feed recovery does not remove selective missingness.
- Audit artifacts: source count, sink count, and lag.
- Diagnostic caution: keep unknown labels out of the denominator.
### Diagnostic drill 10: Outcome correction
- Candidate mechanism: late adjudication revises delivered status.
- Discriminating clue: metric changes after label version update.
- First investigation: compare source correction IDs and prior snapshots.
- Immediate bounded mitigation: recompute affected cohort with version.
- Preventive control: immutable versioned label ledger.
- Evidence to preserve: label correction version.
- Expected signature: an order changes from on-time to late.
- Competing explanation: a late duplicate mistaken for a correction.
- Discriminating comparison: compare correction identifier and event sequence.
- Decision invariant: latest authorized correction wins as of a frozen cutoff.
- Containment choice: recompute affected cohort with new version.
- Containment side effect: previous dashboards and reports may disagree.
- Proposed replay: replay correction and duplicate in both orders.
- Accountable role: label owner authorizes correction semantics.
- Ongoing monitor: watch corrections per cohort and status.
- Recovery criterion: recomputed result reproduces from versioned inputs.
- Interpretation limit: corrected labels do not reconstruct counterfactual outcomes.
- Audit artifacts: original label, correction ID, and query cutoff.
- Diagnostic caution: publish changes to prior metric snapshots.
### Diagnostic drill 11: Selective outcome missingness
- Candidate mechanism: unresolved orders concentrate among high scores.
- Discriminating clue: label completeness differs by score band.
- First investigation: audit missingness by score and queue treatment.
- Immediate bounded mitigation: report bounds instead of point claim.
- Preventive control: cohort completeness monitoring.
- Evidence to preserve: resolution by score band.
- Expected signature: high-risk orders remain unresolved more often.
- Competing explanation: random feed latency rather than risk-dependent censoring.
- Discriminating comparison: compare completeness by score and assignment.
- Decision invariant: resolved fraction uses all eligible orders in each band.
- Containment choice: publish missing-label sensitivity bounds.
- Containment side effect: bounds may be too wide for release approval.
- Proposed replay: calculate all-late and all-on-time endpoints.
- Accountable role: evaluation owner reviews missingness model.
- Ongoing monitor: watch resolution curves by age and segment.
- Recovery criterion: bounds and exclusions are reproducible.
- Interpretation limit: a 95% overall completion rate does not guarantee unbiased recall.
- Audit artifacts: cohort manifest with unresolved reason.
- Diagnostic caution: avoid silently extrapolating from easy-to-label orders.
### Diagnostic drill 12: Intervention feedback
- Candidate mechanism: assistance changes the late outcome.
- Discriminating clue: selected high-risk orders get extra help.
- First investigation: compare randomized assignment where available.
- Immediate bounded mitigation: qualify quality as policy-dependent.
- Preventive control: safe holdout and assignment logging.
- Evidence to preserve: assistance assignment.
- Expected signature: prioritized orders receive extra carrier contact.
- Competing explanation: high baseline risk rather than assistance effect.
- Discriminating comparison: compare randomized assignment within eligibility strata.
- Decision invariant: assignment must precede intervention and be logged.
- Containment choice: avoid causal claims from observed selected outcomes.
- Containment side effect: a holdout may withhold useful assistance.
- Proposed replay: run approved safe randomization only.
- Accountable role: operations and ethics owners approve limits.
- Ongoing monitor: watch assignment balance and spillover.
- Recovery criterion: audit probabilities and intention-to-treat joins.
- Interpretation limit: a randomized effect is local to its policy and population.
- Audit artifacts: assignment seed, eligibility, and override.
- Diagnostic caution: never label assisted versus unassisted difference as causal by default.
### Diagnostic drill 13: Score threshold change
- Candidate mechanism: policy changes cutoff but model is identical.
- Discriminating clue: positive rate changes at same score distribution.
- First investigation: compare policy version and raw score histogram.
- Immediate bounded mitigation: restore authorized policy if unintended.
- Preventive control: policy change audit.
- Evidence to preserve: policy configuration.
- Expected signature: positive fraction jumps at deployment.
- Competing explanation: underlying score distribution actually shifted.
- Discriminating comparison: replay fixed scores through old and new thresholds.
- Decision invariant: model artifact and raw score should remain unchanged.
- Containment choice: revert unintended threshold configuration.
- Containment side effect: reversion can exceed current review capacity.
- Proposed replay: reproduce boundary scores including exact ties.
- Accountable role: policy owner signs threshold changes.
- Ongoing monitor: watch cutoff, positive fraction, and queue saturation.
- Recovery criterion: old configuration produces expected assignments.
- Interpretation limit: stable raw scores do not establish stable calibration.
- Audit artifacts: policy digest and effective timestamp.
- Diagnostic caution: report cutoff metrics separately from top-k metrics.
### Diagnostic drill 14: Rank tie handling
- Candidate mechanism: equal scores resolve differently after release.
- Discriminating clue: queue membership changes at tie boundary.
- First investigation: replay ordering with deterministic order IDs.
- Immediate bounded mitigation: restore deterministic tie-break.
- Preventive control: versioned ranking contract.
- Evidence to preserve: rank ordering keys.
- Expected signature: queue members change only at equal-score boundary.
- Competing explanation: model scores changed slightly after quantization.
- Discriminating comparison: compare full precision scores and stable tie keys.
- Decision invariant: same inputs and policy must produce same ranking.
- Containment choice: restore stable order-ID tie-break.
- Containment side effect: different affected individuals may lose a queue slot.
- Proposed replay: replay tie-heavy and boundary cases.
- Accountable role: queue owner approves tie policy.
- Ongoing monitor: watch number of boundary ties and ordering version.
- Recovery criterion: identical replay produces identical queue membership.
- Interpretation limit: determinism alone does not imply fair allocation.
- Audit artifacts: precision, tie key, and ranking digest.
- Diagnostic caution: inspect whether ties cluster in protected or operational segments.
### Diagnostic drill 15: Schema version mismatch
- Candidate mechanism: consumer parses old field layout.
- Discriminating clue: only one consumer version has missing fields.
- First investigation: inspect producer-consumer schema compatibility.
- Immediate bounded mitigation: stop incompatible consumer and fallback.
- Preventive control: compatibility tests.
- Evidence to preserve: producer and consumer schema IDs.
- Expected signature: one parser reads the wrong field layout.
- Competing explanation: upstream fields actually missing before transmission.
- Discriminating comparison: compare raw bytes and decoded records by schema.
- Decision invariant: consumer must accept only explicitly compatible schema.
- Containment choice: isolate incompatible consumer version.
- Containment side effect: fallback traffic rises while repair is deployed.
- Proposed replay: replay old and new schema fixtures.
- Accountable role: schema owner approves compatibility rule.
- Ongoing monitor: watch decode failures by producer version.
- Recovery criterion: supported fixtures decode as expected.
- Interpretation limit: schema compatibility cannot guarantee semantic unit correctness.
- Audit artifacts: schema fingerprint and decoder version.
- Diagnostic caution: test optional-field defaults as well as types.
### Diagnostic drill 16: Normalization skew
- Candidate mechanism: serving divides distance twice.
- Discriminating clue: raw feature is stable but transformed value shifts.
- First investigation: compare offline and online transformation traces.
- Immediate bounded mitigation: restore matching preprocessing artifact.
- Preventive control: golden-vector parity tests.
- Evidence to preserve: raw and transformed feature values.
- Expected signature: raw kilometres stable but normalized inputs collapse.
- Competing explanation: upstream units changed before logging.
- Discriminating comparison: compare raw log, transformation, and model input.
- Decision invariant: training and serving apply the same transformation exactly once.
- Containment choice: restore approved preprocessing artifact.
- Containment side effect: correcting scaling may change score distribution abruptly.
- Proposed replay: replay golden examples through both paths.
- Accountable role: model owner signs transformation digest.
- Ongoing monitor: watch raw-to-model-input ratios.
- Recovery criterion: golden vectors agree across offline and online paths.
- Interpretation limit: feature parity does not establish generalization.
- Audit artifacts: preprocessing code and artifact digest.
- Diagnostic caution: include edge cases and missing-value transformations.
### Diagnostic drill 17: Model artifact mismatch
- Candidate mechanism: wrong model version loaded.
- Discriminating clue: model digest differs from approved release.
- First investigation: compare artifact registry digest and serving log.
- Immediate bounded mitigation: route to signed approved artifact.
- Preventive control: digest enforcement at startup.
- Evidence to preserve: loaded artifact digest.
- Expected signature: serving digest differs from approved manifest.
- Competing explanation: approval manifest stale rather than wrong artifact.
- Discriminating comparison: compare registry signature, approval, and process startup log.
- Decision invariant: every serving replica loads an authorized immutable artifact.
- Containment choice: drain affected replicas and route to valid deployment.
- Containment side effect: capacity falls during drain.
- Proposed replay: restart isolated replica from signed manifest.
- Accountable role: release owner validates approval chain.
- Ongoing monitor: watch digest distribution across replicas.
- Recovery criterion: all active replicas report approved digest.
- Interpretation limit: approved artifact can still perform poorly on new populations.
- Audit artifacts: registry digest and signed approval.
- Diagnostic caution: verify preprocessing and policy artifacts too.
### Diagnostic drill 18: Population eligibility change
- Candidate mechanism: new channel enters the monitored cohort.
- Discriminating clue: shift aligns with eligibility-policy revision.
- First investigation: reconstruct eligibility on fixed manifest.
- Immediate bounded mitigation: compare like-for-like slices.
- Preventive control: policy-version segmentation.
- Evidence to preserve: eligibility policy version.
- Expected signature: new channel included in current population.
- Competing explanation: real distribution change within unchanged channel.
- Discriminating comparison: apply old eligibility to both windows.
- Decision invariant: reference and current cohorts use comparable inclusion rules.
- Containment choice: report both fixed-population and all-traffic views.
- Containment side effect: excluding new traffic can hide its genuine risk.
- Proposed replay: reconstruct membership using policy effective time.
- Accountable role: product owner confirms eligibility intent.
- Ongoing monitor: watch eligibility counts by channel and version.
- Recovery criterion: historical membership reproduces from versioned policy.
- Interpretation limit: fixed-cohort quality does not describe new users.
- Audit artifacts: eligibility rules and event-time version.
- Diagnostic caution: do not rewrite the original historical cohort.
### Diagnostic drill 19: Duplicate event replay
- Candidate mechanism: retries create repeated scores.
- Discriminating clue: record count rises but unique IDs stay stable.
- First investigation: compare event IDs and retry markers.
- Immediate bounded mitigation: deduplicate by scoring event.
- Preventive control: idempotent ingestion.
- Evidence to preserve: scoring event identifier.
- Expected signature: record count grows without unique-order growth.
- Competing explanation: legitimate distinct rescore events rather than retries.
- Discriminating comparison: inspect event IDs, attempt IDs, and rescore reasons.
- Decision invariant: one idempotency key maps to one scoring decision.
- Containment choice: deduplicate retry copies before histogramming.
- Containment side effect: over-aggressive deduplication could remove real rescoring.
- Proposed replay: replay same event and a legitimate new event.
- Accountable role: ingestion owner sets idempotency semantics.
- Ongoing monitor: watch retry rate and duplicate-key rate.
- Recovery criterion: event counts reconcile with valid unique decisions.
- Interpretation limit: clean counts do not imply representative data.
- Audit artifacts: order ID, scoring event ID, and attempt ID.
- Diagnostic caution: retain the raw append-only retry log.
### Diagnostic drill 20: Tiny slice volatility
- Candidate mechanism: a low-volume merchant slice pages repeatedly.
- Discriminating clue: one order changes the apparent percentage sharply.
- First investigation: inspect denominators and intervals.
- Immediate bounded mitigation: suppress unsupported inference and aggregate.
- Preventive control: minimum sample and hierarchical reporting.
- Evidence to preserve: slice denominator.
- Expected signature: merchant rate swings after one or two orders.
- Competing explanation: a real isolated incident in a small merchant.
- Discriminating comparison: inspect raw count and absolute number of failures.
- Decision invariant: suppress rate-based inference below the minimum count.
- Containment choice: aggregate compatible slices while preserving severe cases.
- Containment side effect: aggregation can hide harms to small groups.
- Proposed replay: evaluate a small-count example with exact counts.
- Accountable role: on-call owner defines severe-event escalation.
- Ongoing monitor: watch sample size and confidence context.
- Recovery criterion: alert includes denominator and aggregation level.
- Interpretation limit: non-significance is not proof of equal performance.
- Audit artifacts: slice definition and numerator.
- Diagnostic caution: avoid suppressing independently critical safety events.
### Diagnostic drill 21: Multiple alert noise
- Candidate mechanism: many independent feature tests page together.
- Discriminating clue: simultaneous alerts cluster in a shared release.
- First investigation: group alerts by feature lineage and incident.
- Immediate bounded mitigation: page once while retaining all evidence.
- Preventive control: alert correlation and calibration.
- Evidence to preserve: alert fingerprint.
- Expected signature: many features fire after one producer change.
- Competing explanation: multiple unrelated failures happening simultaneously.
- Discriminating comparison: compare lineage, onset time, and affected cohorts.
- Decision invariant: group notifications without deleting distinct evidence.
- Containment choice: open one parent incident with linked alerts.
- Containment side effect: over-grouping can conceal an independent incident.
- Proposed replay: inject two simultaneous unrelated fault signatures.
- Accountable role: incident commander reviews correlation rules.
- Ongoing monitor: watch alert volume and unacknowledged groups.
- Recovery criterion: all underlying breaches remain visible.
- Interpretation limit: one incident label does not prove one root cause.
- Audit artifacts: feature lineage and alert event IDs.
- Diagnostic caution: allow responders to split an alert group.
### Diagnostic drill 22: Joint distribution shift
- Candidate mechanism: feature marginals stay stable but relationships change.
- Discriminating clue: model score or error changes despite stable univariate PSI.
- First investigation: inspect joint slices and interaction diagnostics.
- Immediate bounded mitigation: investigate correlated inputs.
- Preventive control: multivariate and outcome monitoring.
- Evidence to preserve: joint feature contingency.
- Expected signature: distance and merchant group relationship changes.
- Competing explanation: univariate drift hidden by binning choices.
- Discriminating comparison: compare joint counts with fixed marginal bins.
- Decision invariant: same marginals can have different co-occurrence patterns.
- Containment choice: investigate interaction-sensitive slices.
- Containment side effect: large contingency tables can be sparse.
- Proposed replay: construct paired datasets with equal marginals.
- Accountable role: model owner chooses supported interaction tests.
- Ongoing monitor: watch score distribution within joint slices.
- Recovery criterion: joint queries reproduce from frozen cohort.
- Interpretation limit: a joint change may be irrelevant to model behavior.
- Audit artifacts: joint bin edges and cell counts.
- Diagnostic caution: avoid interpreting sparse cells without uncertainty.
### Diagnostic drill 23: Concept shift
- Candidate mechanism: same feature distribution maps to different lateness.
- Discriminating clue: mature error changes at stable feature distributions.
- First investigation: compare conditional outcomes in matched slices.
- Immediate bounded mitigation: evaluate model change after data validation.
- Preventive control: mature quality and calibration tracking.
- Evidence to preserve: conditional outcome rate.
- Expected signature: mature late rate changes within a score band.
- Competing explanation: label-definition change or intervention change.
- Discriminating comparison: compare stable policy and outcome definitions over time.
- Decision invariant: conditional quality needs mature comparable outcomes.
- Containment choice: hold model change until data and policy checks finish.
- Containment side effect: waiting can prolong poor decisions.
- Proposed replay: evaluate reliability by score and segment.
- Accountable role: evaluation owner signs cohort comparability.
- Ongoing monitor: watch calibration and recall under fixed eligibility.
- Recovery criterion: quality calculation reproduces on frozen labels.
- Interpretation limit: conditional association does not prove model caused lateness.
- Audit artifacts: label version, score band, and policy.
- Diagnostic caution: check prevalence and sampling changes before concept-shift claim.
### Diagnostic drill 24: Queue capacity reduction
- Candidate mechanism: fewer agents are available.
- Discriminating clue: recall at fixed capacity drops despite stable scores.
- First investigation: compare capacity policy and staffing logs.
- Immediate bounded mitigation: restore capacity or prioritize safety.
- Preventive control: capacity telemetry and policy controls.
- Evidence to preserve: available assistance slots.
- Expected signature: top-ranked coverage falls when staff capacity drops.
- Competing explanation: ranking quality worsened independently of staffing.
- Discriminating comparison: replay identical scores at old and new slot counts.
- Decision invariant: top-k recall depends on k and cohort late count.
- Containment choice: restore staffing or revise policy with approval.
- Containment side effect: prioritizing one segment can displace another.
- Proposed replay: calculate coverage at 90 and 45 model slots.
- Accountable role: operations owner controls staffing and overrides.
- Ongoing monitor: watch backlog, slot count, and wait time.
- Recovery criterion: queue never exceeds approved hourly capacity.
- Interpretation limit: restored capacity does not guarantee intervention efficacy.
- Audit artifacts: staffing roster and queue-policy version.
- Diagnostic caution: compare model performance at a fixed capacity as well.
### Diagnostic drill 25: Bad rollback
- Candidate mechanism: previous model depends on the same faulty feature.
- Discriminating clue: both artifacts receive identical corrupt distance.
- First investigation: trace feature dependency and replay prior model.
- Immediate bounded mitigation: use independent manual fallback.
- Preventive control: dependency-aware rollback rehearsal.
- Evidence to preserve: fallback feature lineage.
- Expected signature: both models use the same malformed distance.
- Competing explanation: prior model has an independent valid input.
- Discriminating comparison: trace actual feature retrieval and artifact dependencies.
- Decision invariant: fallback must have a tested valid input path.
- Containment choice: use manual queue if both paths are invalid.
- Containment side effect: manual staff can become overloaded.
- Proposed replay: inject the same broken feature into both model paths.
- Accountable role: release owner approves fallback dependencies.
- Ongoing monitor: watch fallback rate and manual backlog.
- Recovery criterion: rehearsal demonstrates independent safe routing.
- Interpretation limit: a successful switch does not establish prediction quality.
- Audit artifacts: dependency graph and fallback decision log.
- Diagnostic caution: do not count a version switch as feature repair.
### Diagnostic drill 26: Dashboard outage
- Candidate mechanism: metric store is unavailable.
- Discriminating clue: serving continues while telemetry heartbeat stops.
- First investigation: inspect independent heartbeat and collector logs.
- Immediate bounded mitigation: page telemetry owner and use manual checks.
- Preventive control: out-of-band heartbeat.
- Evidence to preserve: independent heartbeat.
- Expected signature: monitoring dashboard stops refreshing while inference continues.
- Competing explanation: serving outage occurring at the same time.
- Discriminating comparison: compare external heartbeat with serving request logs.
- Decision invariant: telemetry failure must not be reported as healthy drift.
- Containment choice: page telemetry owner through separate channel.
- Containment side effect: manual inspection has lower coverage and slower response.
- Proposed replay: disconnect metric store in a controlled exercise.
- Accountable role: observability owner restores data pipeline.
- Ongoing monitor: watch heartbeat age and buffered event count.
- Recovery criterion: heartbeat and metric freshness recover independently.
- Interpretation limit: dashboard recovery cannot prove no incident occurred during the gap.
- Audit artifacts: heartbeat timestamps and backfill record.
- Diagnostic caution: mark missing windows unknown rather than interpolating.
### Diagnostic drill 27: Calibration degradation
- Candidate mechanism: predicted probabilities no longer match outcomes.
- Discriminating clue: mature reliability curve shifts within score bins.
- First investigation: compare bin outcomes and uncertainty.
- Immediate bounded mitigation: recalibrate only after validation.
- Preventive control: calibration monitoring by segment.
- Evidence to preserve: mature reliability table.
- Expected signature: observed lateness differs from predicted score bins.
- Competing explanation: base-rate shift or selective missing labels.
- Discriminating comparison: compare calibration with complete fixed-cohort labels.
- Decision invariant: bin fraction late is late count divided by resolved count.
- Containment choice: consider recalibration only after independent validation.
- Containment side effect: recalibration can change thresholds and queue composition.
- Proposed replay: check calibration on held-out later cohorts.
- Accountable role: model owner approves recalibration artifact.
- Ongoing monitor: watch calibration by bin and sample count.
- Recovery criterion: held-out estimates and uncertainty meet chosen criteria.
- Interpretation limit: good calibration does not guarantee good ranking.
- Audit artifacts: bin edges, counts, and score version.
- Diagnostic caution: do not fit and evaluate calibration on the same cohort.
### Diagnostic drill 28: Late cohort contamination
- Candidate mechanism: immature outcomes are counted as on time.
- Discriminating clue: recent cohort looks artificially precise or inaccurate.
- First investigation: rebuild using maturity and resolution flags.
- Immediate bounded mitigation: withdraw premature quality claim.
- Preventive control: label maturity gate.
- Evidence to preserve: label age distribution.
- Expected signature: recent cohort has many assumed on-time orders.
- Competing explanation: genuine operational improvement on recent orders.
- Discriminating comparison: compare promised-date age with final resolution flags.
- Decision invariant: unresolved orders never become negative labels by default.
- Containment choice: withdraw immature headline quality report.
- Containment side effect: stakeholders lose a rapid but misleading metric.
- Proposed replay: recompute the cohort after the seven-day horizon.
- Accountable role: report owner issues corrected interpretation.
- Ongoing monitor: watch unresolved rate as cohort ages.
- Recovery criterion: resolved cohort totals reconcile to manifest.
- Interpretation limit: maturity alone cannot eliminate intervention bias.
- Audit artifacts: promised time, resolution time, and status.
- Diagnostic caution: show provisional counts without calling them final quality.
### Diagnostic drill 29: Train-serving leakage
- Candidate mechanism: training uses post-acceptance delivery information.
- Discriminating clue: offline metrics exceed point-in-time replay.
- First investigation: audit feature availability against scoring timestamps.
- Immediate bounded mitigation: remove leaked feature and retrain offline.
- Preventive control: point-in-time feature lineage.
- Evidence to preserve: feature availability timestamp.
- Expected signature: offline features were created after scoring time.
- Competing explanation: serving logs have clock skew rather than leakage.
- Discriminating comparison: compare source event, availability, and scoring clocks.
- Decision invariant: feature availability must not exceed decision time.
- Containment choice: exclude leaked feature and rebuild evaluation.
- Containment side effect: removal may lower offline quality substantially.
- Proposed replay: replay point-in-time joins around boundary timestamps.
- Accountable role: training data owner signs cutoff semantics.
- Ongoing monitor: watch invalid temporal joins by feature.
- Recovery criterion: all retained joins obey cutoff after clock correction.
- Interpretation limit: a clean join does not guarantee representative future data.
- Audit artifacts: event timestamp, availability timestamp, and clock source.
- Diagnostic caution: do not use outcome-derived corrections as historical features.
### Diagnostic drill 30: Manual override shift
- Candidate mechanism: operators override more scores during incident.
- Discriminating clue: queue treatment changes while model is stable.
- First investigation: compare override rates by reason and cohort.
- Immediate bounded mitigation: separate manual and model queues.
- Preventive control: override audit and staffing review.
- Evidence to preserve: override reason codes.
- Expected signature: manual overrides rise during incident.
- Competing explanation: staffing or eligibility policy changed independently.
- Discriminating comparison: compare override rates by operator shift and reason.
- Decision invariant: manual slots and model-ranked slots stay distinguishable.
- Containment choice: audit overrides without suppressing urgent decisions.
- Containment side effect: restricting overrides can delay critical assistance.
- Proposed replay: replay queue with and without documented overrides.
- Accountable role: operations lead reviews reason-code quality.
- Ongoing monitor: watch override count, reasons, and outcomes.
- Recovery criterion: all overrides have actor, time, and reason.
- Interpretation limit: observational override outcomes cannot prove operator benefit.
- Audit artifacts: assignment rank, override actor, and timestamp.
- Diagnostic caution: never treat undocumented override as a model prediction.
## 13. Transferable Lesson
Distribution monitoring is an early warning system; supervised quality and causal intervention value answer different questions.
Fast schema checks protect the decision path while delayed labels prevent premature quality claims.
Stable marginal distributions can coexist with degraded ranking, and shifted distributions can coexist with stable quality.
Choose a fallback by dependency and safety rather than assuming an old model has independent inputs.
Maintain cohort identity and policy history so that later labels answer the original decision question.
Transfer requires a known decision time, an explicit outcome horizon, sufficient volume, and a feasible response owner.
For small populations, longer windows and partial pooling may replace hourly per-segment alerts.
For high-stakes decisions, manual review and experimental holdouts require domain-specific approval.
Thresholds and cost-benefit choices must be calibrated locally; none here is universally valid.

## 14. Connections & Cross-References
The [local collection](INDEX.md) groups this case with monitoring, distribution shift, and delayed-label patterns.
The [focused plan](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-ml-design) situates monitoring among production ML disciplines.
The [case master](../../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) defines the required evidence and interview structure.
The named source file is topical provenance only; its actual content and any source-specific claims are not represented as verified.
Open questions include unknown label-delay distribution, intervention effects, unit economics, and actual alert false-positive rates.
### Interview pack
Each question is answered under the invented case assumptions; no example is an observed operational result.
#### Question 1: What decision is being made?
- Answer: Choose continue, contain, or verified fallback for the affected region and channel.
- Reasoning: The decision is bounded to the model and pipeline, not order cancellation.
- Follow-up: What if fallback shares the broken input?
- Follow-up answer: Use the independent manual queue and record capacity limits.
- Acceptable alternative: A safe manual policy may beat a model rollback.
- Red flag: Automatically roll back every drift alert.
- Scoring: full credit requires: State owner, deadline, baseline, and safety constraint.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 2: Does PSI prove quality degradation?
- Answer: No; it summarizes a change in binned feature proportions.
- Reasoning: Input changes may be benign, harmful, or offset by model robustness.
- Follow-up: Can quality degrade with stable PSI?
- Follow-up answer: Yes; conditional outcomes or feature relationships can change.
- Acceptable alternative: Compare joint shifts and mature model metrics.
- Red flag: Call any PSI breach model failure.
- Scoring: full credit requires: Separate covariate change from predictive loss.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 3: Compute the example PSI.
- Answer: Approximately 0.240794, dimensionless.
- Reasoning: Add 0.102165 and 0.138629; the middle contribution is zero.
- Follow-up: Does it page immediately?
- Follow-up answer: Only if both complete consecutive hourly windows satisfy the minimum volume and threshold.
- Acceptable alternative: A schema violation has a separate immediate-page rule.
- Red flag: Treat one sample as a sustained page.
- Scoring: full credit requires: Show bins, proportions, logarithm, and timing.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 4: What changes if current counts are 400, 300, 300?
- Answer: PSI is approximately 0.062861 and below the proposed warning.
- Reasoning: The changed bins contribute 0.1 ln(1.25) and 0.1 ln(1.5).
- Follow-up: What if a bin has zero count?
- Follow-up answer: Apply a predeclared smoothing and renormalization method, and disclose it.
- Acceptable alternative: Use another suitable distribution diagnostic.
- Red flag: Divide by zero or silently discard bins.
- Scoring: full credit requires: Compute and interpret the sensitivity.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 5: How is late recall calculated in the example?
- Answer: 120 divided by 200 equals 0.60.
- Reasoning: Recall uses all actual late orders as denominator.
- Follow-up: What is precision?
- Follow-up answer: 120 divided by 300 equals 0.40.
- Acceptable alternative: Also report the full confusion matrix.
- Red flag: Use all 1,000 orders as the recall denominator.
- Scoring: full credit requires: Check row and column sums.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 6: Why is the confusion matrix not queue performance?
- Answer: The fixed cutoff flags 300 orders, whereas only 90 model-ranked slots exist.
- Reasoning: The top-capacity policy and manual reservations define a different selection.
- Follow-up: What should replace it?
- Follow-up answer: Measure precision and recall at top 90 on the same eligible mature cohort.
- Acceptable alternative: Report both metrics with their decision context.
- Red flag: Claim cutoff precision equals precision at capacity.
- Scoring: full credit requires: Explain the distinct denominators.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 7: How do delayed labels affect the decision?
- Answer: Use fast contract and feature signals for containment, then evaluate quality on mature resolved cohorts.
- Reasoning: Treat unresolved labels as unknown rather than on time.
- Follow-up: Is 95% completeness enough to remove bias?
- Follow-up answer: No; missingness can be concentrated in high-risk or treated orders.
- Acceptable alternative: Provide worst-case bounds and segment completeness.
- Red flag: Evaluate only quickly arriving labels as representative.
- Scoring: full credit requires: Specify cohort age, completeness, and sensitivity.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 8: What evidence separates a unit bug from campaign mix?
- Answer: Compare raw producer values and schema units by version and campaign.
- Reasoning: A unit-scale mismatch supports a contract failure; valid stratified mix supports the other hypothesis.
- Follow-up: Can PSI alone distinguish them?
- Follow-up answer: No; the same aggregate histogram can arise from several mechanisms.
- Acceptable alternative: Replay known raw examples through both transformations.
- Red flag: Retrain before examining the contract.
- Scoring: full credit requires: State competing predictions and discriminating checks.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 9: Why might an old model be an unsafe fallback?
- Answer: It may use the same corrupted input or incompatible preprocessing.
- Reasoning: Artifact age does not imply independence from the data fault.
- Follow-up: What if no valid score exists?
- Follow-up answer: Use the documented manual queue and record the affected population.
- Acceptable alternative: An independent verified model can be considered if available.
- Red flag: Assume rollback fixes every upstream fault.
- Scoring: full credit requires: Trace dependencies and choose a safe route.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 10: What is the intervention counterfactual?
- Answer: The delivery outcome for the same order under a different assistance assignment.
- Reasoning: Only one outcome is observed for each order, so naive treated-versus-untreated comparisons are confounded.
- Follow-up: How could it be estimated?
- Follow-up answer: A safe randomized holdout with logged assignment and intention-to-treat analysis.
- Acceptable alternative: A defensible observational design may be considered with explicit assumptions.
- Red flag: Credit the model with all assisted-order outcomes.
- Scoring: full credit requires: Distinguish prediction from causal benefit.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 11: How do you handle a monitoring outage?
- Answer: Treat missing or stale telemetry as unknown and alert its owner through an independent heartbeat.
- Reasoning: A quiet dashboard is not evidence that features are stable.
- Follow-up: Should serving depend on the dashboard?
- Follow-up answer: No; retain a separate bounded serving and fallback path.
- Acceptable alternative: Use sampled manual operational checks during recovery.
- Red flag: Silence all alerts and infer health.
- Scoring: full credit requires: Name the heartbeat and failure isolation.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 12: How can marginal drift miss a real issue?
- Answer: Feature relationships or conditional label distributions can change while each marginal remains stable.
- Reasoning: Univariate PSI does not characterize joint distributions or P(Y|X).
- Follow-up: Which next checks help?
- Follow-up answer: Inspect joint slices, score distributions, and mature cohort errors.
- Acceptable alternative: Use a calibrated multivariate test where appropriate.
- Red flag: Declare no degradation because every feature PSI is small.
- Scoring: full credit requires: Give a concrete distinction between joint and marginal changes.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 13: What changes when volume doubles but slots do not?
- Answer: Coverage falls from 100/1,000 to 100/2,000, or 5%.
- Reasoning: The policy remains capacity constrained even if scoring throughput is adequate.
- Follow-up: What about model-ranked slots?
- Follow-up answer: If 10 slots remain manual, only 90 of 2,000 are model-ranked, or 4.5%.
- Acceptable alternative: Increase staffing only after operational approval and cost analysis.
- Red flag: Assume fixed percentage coverage with fixed absolute slots.
- Scoring: full credit requires: Compute both total and model-ranked coverage.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
#### Question 14: When is recovery complete?
- Answer: After verified contract repair, two complete compliant windows, and explicit owner approval.
- Reasoning: Silence or a single passing window is insufficient.
- Follow-up: Does that prove business performance recovered?
- Follow-up answer: No; mature labels and a defensible comparator are still needed.
- Acceptable alternative: Use stronger release gates if the domain requires them.
- Red flag: Claim customer benefit from a green telemetry panel.
- Scoring: full credit requires: Separate technical recovery and outcome evidence.
- Partial credit: identify the relevant concept but omit the decision or its limitation.
- No credit: assert an unsupported causal or operational result.
- Answer support: Sections 4–9 and the illustrative calculations in Section 6.
- Evidence status: constructed example; no deployed-system claim.
## 15. Quick Reference Card
| Field | Case-specific answer |
|---|---|
| Problem | Determine whether and how to contain a delivery-risk model after a sustained distance shift. |
| Context | Hypothetical one-region, one-channel hourly incident with delayed delivery outcomes. |
| Decision | Continue, use a verified compatible prior model, or use the manual queue. |
| Method | Contract checks, fixed-bin PSI, cohort-matched mature quality, and counterfactual-aware evaluation. |
| Finding | Constructed PSI is 0.240794; shift does not prove degradation. |
| Lesson | Diagnose data validity quickly while keeping prediction quality and causal impact separate. |
| Evidence | Illustrative assumptions and checked arithmetic; no real measurements or execution. |
| Stage | Proposed design; no deployed incident or intervention. |
| Editorial review | Draft, unreviewed; last_reviewed null. |
| Review trigger | Outcome maturity, feature contract, or intervention policy changes. |
