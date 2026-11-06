---
case_id: CASE-01-01-001
case_title: Real-time payment-card fraud detection under a friction budget
case_version: "1.0"
created: 2026-11-07
last_reviewed: null
status: proposed
content_status: draft
case_type: predictive-ml
domain: Issuer payment-card fraud
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, cost-latency-tradeoff]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-07
owner_role: Issuer fraud product owner
source_index: INDEX.md
review_trigger: New label source, authorization deadline, network rules, or material change to policy or customer mix
---
# Case study: Real-time payment-card fraud detection under a friction budget

## 1. The One-Line Framing
At authorization, an issuer must choose approve, challenge, or decline for each card transaction, using only information available before its response deadline, while protecting legitimate customers and comparing with its existing rules-only policy.
The proposed model supplies risk estimates to a separately controlled decision policy.
This is an invented teaching case, not a reconstruction of a named institution or an implemented deployment.
No production population, model result, experiment, deployment or financial impact has been observed.

## 2. Context & Background
The issuer receives an authorization request from a network after the merchant initiates a card transaction.
The issuer can approve, decline, or invoke an available authentication flow, depending on channel and network capabilities.
The customer expects a prompt response; an asynchronous analyst cannot generally decide an in-flight authorization.
An approved fraudulent purchase may lead to a dispute and financial loss.
A false decline may block a legitimate purchase and cause customer and merchant friction.
A challenge may deter fraud but may also fail, expire, or frustrate a legitimate customer.
The illustrative issuer owns the authorization policy, not the merchant's checkout or the network's protocols.
The fraud product owner is accountable for the decision, and an operations owner handles downstream investigations.
The model risk owner independently approves evidence and thresholds.
The security owner controls card-data access and incident response.
The existing comparator is a versioned rules-only authorization policy with its actual fallback and challenge behavior preserved.
No baseline performance is assumed or measured.
The analytical scope is card-present and card-not-present purchase authorizations in a single illustrative issuer portfolio.
Cash withdrawals, refunds, AML surveillance, account opening, and claims adjudication are out of scope.
Different payment rails or jurisdictions require separate review of available actions, privacy, and legal requirements.

## 3. The Business Question
Illustrative stakeholder question: Which purchase authorizations should we approve, challenge, or decline to reduce fraud loss without exceeding a defined customer-friction budget or the response deadline?
The unit of action is one authorization attempt; linked attempts remain distinct decisions.
The decision is whether to replace the rules-only policy with a calibrated model-assisted policy, not whether to optimize classification accuracy.
The primary proposed business estimand is net portfolio loss per 1,000 eligible authorization attempts over a mature cohort.
It includes issuer-borne fraud loss and explicitly monetized customer-friction and authentication costs, with the components reported separately.
The numerator is total adjudicated issuer loss plus pre-agreed friction and challenge costs, in currency units.
The denominator is all eligible randomized attempts, including declined attempts, not only approved transactions.
A transaction-level amount is not automatically the issuer's ultimate fraud loss.
Guardrails include legitimate decline rate, challenge rate, challenge abandonment, customer complaints, subgroup outcomes, timeout rate, and authorization availability.
A target such as 1% challenge rate is a planning assumption, not a service commitment.
Baseline and treatment must face the same eligibility definition, calendar period, and attribution window.
Because a challenge changes the subsequent path, a score trained on historical outcomes alone does not identify each action's causal effect.
The product owner must approve the cost schedule and risk tolerance before threshold selection.

## 4. Assumptions & Constraints
| Assumption | Type | Impact if wrong / action |
|---|---|---|
| Issuer receives a stable request identifier | Infrastructure | If absent, establish a network-supported idempotency key before retry-safe action. |
| A subset of channels supports challenge | Business | Restrict challenge to eligible requests and measure coverage; never silently promise it everywhere. |
| Fraud and dispute feeds can be joined to attempts | Data | Estimate unmatched rate and audit before using labels for selection. |
| An agreed maturity period captures most relevant labels | Data | Plot reporting delay by cohort and extend follow-up if needed. |
| Tokenization and retention are approved | Regulatory/security | Stop processing until security and legal approve a jurisdiction-specific design. |
| Analyst audits can sample low-score attempts | Operations | If not, selective-label bias remains and decision claims are limited. |
| Model inference and features fit a bounded deadline | Infrastructure | Use tested fallback; do not call external enrichment synchronously without a budget. |
| Issuer can randomize eligible traffic ethically | Business | If not, use shadow evaluation and a justified quasi-experiment without causal overclaim. |
These are hypotheses, not observed conditions or universal regulatory requirements.
The illustrative end-to-end internal risk-decision budget is 100 ms at p99, measured at the issuer decision boundary.
That number is a design target, not a network-wide deadline and not an observed percentile.
Available time must be negotiated against the real network timeout and all downstream processing.
Mandatory legal, network, sanctions, security, and customer protections override model policy when actually applicable.
No universal fail-open or fail-closed decision is safe; the owner must approve a bounded rules-only fallback for each channel.

## 5. Data, Tools & Method
### 5a. Data Specification
The request grain is one authorization attempt keyed by issuer and stable attempt ID.
Repeat attempts, reversals, and incremental authorizations are related events rather than automatic duplicates.
Data fields include event time, issuer receipt time, action time, availability time, tokenized card key, merchant key, amount, currency, channel, and decision.
A tokenized account linkage enables customer-level evaluation without placing a primary account number in model logs.
Labels are confirmed fraud, confirmed legitimate where evidence exists, or unresolved; unresolved is not automatically legitimate.
A dispute is an evidence source, not an unconditional truth label.
Record label source, adjudication state, event time, ingestion time, confidence, reversal, and version.
Fraud loss is a separately reconciled monetary measure; a fraud flag does not establish net loss.
The illustrative training period would precede validation and a final untouched forward test.
The split dates, time zone, source volumes, cohort counts, and sampled coverage are not available and must be fixed before a real run.
The outcome-maturity window is proposed as 90 days after authorization, contingent on measured delay distributions and legal retention.
An event whose outcome matures later remains censored at the snapshot, not a negative label.
Use availability-time point-in-time joins, not a retrospective join to the latest entity record.
Link one request to zero or one canonical decision, and retain multiple associated evidence events with explicit aggregation rules.
Report unmatched, ambiguous, duplicate, and late-arriving records by channel and decision.
A sampling audit must not force approval of risky transactions merely to obtain labels.

#### Illustrative feature contract and boundary checks
The entries below describe candidates, not features known to exist or approved for use.
All historical values require both a prior event and prior system availability.
Any feature whose source cannot provide reliable availability time is excluded until its lineage is established.
Each feature's default and stale handling must be represented in training and serving.

**Request amount.**
The value is the nonnegative purchase amount in the original currency's smallest unit.
The request is the source; the issuer receipt time establishes availability.
Treat a missing amount differently from a genuine zero amount.
Amounts do not directly encode issuer-borne loss because recovery and liability vary.
Verify decimal scale separately for each supported currency.
Reject or explicitly route malformed values before inference.

**Original currency.**
The category is the transaction's currency code, not its country.
It is available in the incoming authorization request when supplied.
An unknown currency uses an explicit unknown category rather than the most frequent currency.
Changes in a merchant's settlement currency must not silently rewrite old requests.
Monitoring should distinguish an unknown code from a missing field.
Validate mappings against an approved, versioned currency reference.

**Converted amount.**
This derived feature converts the request amount to a common comparison currency.
Its exchange-rate snapshot must have been available before the decision.
Historical reconstruction must not use the final rate for the end of the day.
An absent exchange rate produces a missing indicator and approved fallback.
Rounding must preserve the source amount and the rate version for audit.
It is not a substitute for settlement accounting.

**Merchant identity.**
The merchant key combines the issuer-visible identifier with its namespace.
The incoming identifier can be reused across acquirers unless the contract forbids it.
An entity-resolution change may combine previously distinct histories.
The current decision must use only the mapping known at receipt.
An unknown merchant is not itself evidence of fraud.
Measure identifier collisions and mapping coverage separately.

**Merchant category.**
This is the category supplied on the authorization message.
It is a merchant classification, not a diagnosis of criminal activity.
Codes can change over time or differ across transaction paths.
Preserve the message value alongside any approved mapping version.
Evaluate whether sparse categories need a pooled representation.
Monitor unknown and changed codes after producer releases.

**Channel.**
The input distinguishes supported card-present and card-not-present pathways.
Challenge eligibility is a policy capability, not a synonym for channel.
Message defaults must not silently relabel an unknown channel.
The field affects both model calibration and the permissible action set.
Report coverage and false declines separately by eligible channel.
Use the versioned authorization protocol mapping for this field.

**Authorization entry mode.**
This indicates how payment credentials were presented when the network supplies it.
Its encoding depends on the protocol version.
An unrecognized entry mode must remain identifiable as unknown.
Treat merchant or terminal configuration errors as a competing explanation.
Test changes in encoded values before deploying an upstream parser.
Avoid extrapolating from one entry mode to unsupported channels.

**Device identifier.**
Only an approved pseudonymous identifier may enter the feature contract.
Its existence and quality depend on channel and customer consent or other permitted basis.
A shared device can legitimately serve multiple cardholders.
Device absence is not equivalent to a new device.
Rotation, reset, and access restrictions affect longitudinal comparisons.
Evaluate privacy, false-link, and exclusion risk before use.

**Card token.**
The token is a protected join key, not a meaningful numeric feature.
The model should not interpret arbitrary token ordering as ordinal risk.
Key rotation can reset history unless an approved stable linkage is available.
Do not infer that two tokens belong to one person without verified linkage.
Keep the mapping behind separate access controls.
Measure history coverage across token age and rotation events.

**Card age.**
This is elapsed time between card activation and authorization receipt.
The activation record must have been ingested before the decision.
A replacement card may be new while the underlying account is established.
Use a separate missing indicator where activation is unavailable.
Do not assume young cards are fraudulent.
Check negative durations caused by clock or event-order errors.

**Account tenure.**
The measure is elapsed time since an approved account-opening event.
It differs from card age when cards are replaced or reissued.
An account that predates available records may be left-censored.
Avoid substituting the data warehouse's earliest record as the true opening date.
Clock skew and migrations can produce impossible negative values.
Apply an explicit cap only if the training and serving definitions match.

**One-minute attempt count.**
Count eligible prior attempts linked to the token in a defined trailing minute.
Exclude the current attempt if the model is trained on prior-only histories.
The lower bound and treatment of ties must be fixed in the feature specification.
Late records may update future windows but must not change a past online decision.
Duplicate source events must not increment the count twice.
Track watermark lag because a low stale count can look like normal activity.

**One-hour attempt count.**
This uses a longer trailing interval and the same token identity.
It captures a different burst scale from the one-minute count.
The count is an attempt count, not a count of approved purchases.
Specify whether prior declines and reversals contribute.
Use an independently tested reference implementation for boundary events.
Compare online and reconstructed counts for sampled requests.

**Daily spend.**
Sum prior eligible purchase amounts within a specified trailing 24-hour interval.
Convert currencies only with historical rates that were then available.
Define whether reversed authorizations reduce this sum and when.
A missing conversion must not silently contribute zero.
A calendar-day total is a different feature from a trailing-day total.
Reconcile totals against canonical source events after replay.

**Merchant novelty.**
The flag indicates that no prior eligible interaction with the merchant is visible.
It must also carry a history-coverage indicator.
A genuinely new merchant and an account with missing history are different cases.
Restrict the history to records available before the current request.
Entity remapping can change novelty unless versions are preserved.
Validate behavior at account opening and data migration boundaries.

**Device novelty.**
This records whether the pseudonymous device appears in prior eligible history.
Absence of a device identifier makes novelty unknown, not automatically true.
Normal device upgrades can yield legitimate new-device events.
Measure the field's missingness by channel and customer segment.
A future customer-device association cannot be backdated.
Test shared devices, rotations, and simultaneous first use.

**Prior decline count.**
Count earlier issuer declines over a stated window, irrespective of their eventual fraud label.
The value is available from the decision log rather than a future investigation.
Changing the issuer policy changes this feature's distribution.
Repeated declines can represent a compromised card or legitimate retry frustration.
Preserve policy version to investigate feedback effects.
Exclude the current decision from its own prior count.

**Prior challenge completion.**
Use only completed challenge events that have already arrived.
An abandoned or timed-out challenge is a separate state from failure.
Channel support determines whether the field can exist.
An authentication success does not prove that a subsequent transaction is legitimate.
Completion events arriving after the current authorization are not eligible.
Keep challenge identifiers to prevent duplicate completion updates.

**Time since last attempt.**
Calculate elapsed receipt time since the most recent available eligible attempt.
The value has units of seconds and must be nonnegative.
An account without visible prior events has missing history, not infinite tenure.
Concurrent attempts require a deterministic tie-handling rule.
Arrival ordering and the underlying event timestamp serve different purposes.
Test same-millisecond attempts and late delivery explicitly.

**Merchant velocity.**
Count eligible prior requests at the merchant within a fixed window.
Large merchants naturally have high volume, so raw count alone is not proof of abuse.
The aggregate may require careful key partitioning to avoid hot spots.
Record merchant mapping version and aggregation freshness.
Avoid using aggregate updates that include the current request twice.
Compare rates with an appropriately scaled historical reference.

**Card-to-merchant diversity.**
Count distinct merchants encountered by a card during a stated prior window.
An approximate distinct counter introduces quantifiable error and version dependencies.
Its error must be evaluated near any operating decision boundary.
Merchant identity collisions can decrease or inflate the count.
Retention and reset policies define the maximum supported lookback.
Do not compare values generated by incompatible counter implementations.

**Merchant-to-card diversity.**
Count distinct card tokens seen at a merchant during a prior window.
It can reflect normal merchant popularity as well as a fraud burst.
Rotating tokens can inflate distinct counts if linkage is not stable.
An upstream delay may produce an artificially low diversity value.
Maintain strict source-time and availability-time cutoffs.
Do not treat this aggregate as evidence that individual customers are linked.

**Country consistency.**
This feature compares approved request geography with a separately supplied account reference.
Country of merchant registration, device location, and customer residence are distinct concepts.
Missing or low-confidence geography must remain explicit.
Travel, virtual merchants, and cross-border commerce can make mismatch legitimate.
Do not infer a person's nationality or protected status from a payment location.
Assess channel coverage and customer impacts before policy use.

**Historical amount deviation.**
Compare the request amount with a prior-only distribution for the same eligible account.
The reference distribution must exclude the current transaction.
Sparse histories require a documented fallback, not a fabricated personal average.
Changing currencies or conversion conventions can create false deviations.
Large purchases may be legitimate and deserve proportionate treatment.
Validate the feature with known histories and cold-start examples.

**Feature freshness.**
This measures the age of the latest accepted source update at decision time.
It is an operational signal, not a customer risk characteristic by itself.
Age must be computed using documented clocks and source semantics.
Negative ages can indicate clock skew or inconsistent timestamps.
A feature may be fresh yet incorrect, duplicated, or incomplete.
Monitor age distributions alongside source lag and reconciliation failures.

### 5b. Tools & Stack
All components are proposals; no product, version, environment, or implementation is claimed.
An authorization gateway validates and tokenizes requests, then calls an issuer-owned policy service.
A keyed stream processor maintains short-lived velocity aggregates and publishes auditable events.
An online store serves timestamped feature snapshots with explicit freshness and default indicators.
A model server returns a versioned score and input contract, and a policy engine applies rules and action thresholds.
An immutable decision log records tokenized request, feature version, action, reason, fallback state, and timings.
An offline warehouse joins as-of features with mature labels and supports rolling temporal evaluation.
A registry stores model artifacts, preprocessing, calibrator, training snapshot, threshold configuration, approvals, and rollback target.
The particular database, streaming engine, model package, and serving language remain unselected pending procurement and compatibility review.
No system is inferred from the source filename.

### 5c. Method Summary
Canonicalize eligible authorization fields and read point-in-time aggregates.
Apply mandatory rules, obtain a score when the model path is healthy, calibrate against a representative maturity-adjusted population, and apply a versioned action policy.
Return one terminal authorization decision; write the decision and outcome lineage asynchronously.
A practical candidate is a gradient-boosted tree trained with a binary log-loss objective on adjudicated, mature training labels.
The class score estimates the risk of the specified observed label under historical selection, not automatically untreated fraud probability.
Tree complexity and regularization are selected on earlier temporal validation folds; preprocessing and calibration are fitted within each fold.
The final test is untouched until model, calibration, feature contract, and thresholds are frozen.
Missing features have explicit indicators and an approved bounded fallback; no silent zeros masquerade as observed values.
| Candidate | Evidence | Strength | Limitation | Status / revisit |
|---|---|---|---|---|
| Existing rules | No measured baseline supplied | Auditable and already the relevant comparator | Manual rule coverage may decay | Proposed comparator; characterize its traffic first |
| Regularized logistic model | No experiments | Transparent, cheap inference | Nonlinear interactions require feature work | Considered; choose if validation and cost support it |
| Boosted trees plus calibration | No experiments | Captures nonlinear tabular interactions | Calibration and temporal stability require evidence | Proposed candidate, not selected by observed results |
| Deep sequential model | No experiments | Could model histories | Higher complexity, latency and sparse-history risks | Not tested; revisit only if incremental evidence justifies it |
| Thresholded risk score alone | No experiments | Simple | Ignores action-specific effects and friction | Not sufficient as a decision policy |
No optimizer, hyperparameters, run ID, selected model, or stopping iteration exists yet.
The proposed stopping rule is early stopping on temporal validation loss, with a fixed candidate grid and no test-set tuning.
After selecting a candidate, freeze a policy only after checking operational and customer constraints on validation.

### 5d. Data provenance, quality, exclusions and leakage controls
The sole source-specific scope is the visible source-tree entry, not the inaccessible contents of an original case.
Authorization, disputes, challenge outcomes, customer support, and fraud investigation feeds are proposed internal sources, not accessed datasets.
Assign a source owner, permitted use, extraction watermark, schema version, and retention schedule before acquisition.
Exclude unjoinable or invalid events only with reason codes and denominator accounting.
Never use a chargeback, final dispute status, or future device association as an authorization-time feature.
For each aggregate require both event timestamp strictly before the decision and source availability timestamp no later than the decision.
A timestamp alone cannot prove that the serving system actually had the value.
Offline reconstruction must replay ingestion delays, late corrections, and online default behavior.
Fit imputers, encoders, and calibrators within training folds, and freeze their artifacts before validation or test.
Use distinct missing, stale, and actual-zero states.
Investigated, challenged, and approved populations can have different probabilities of label observation.
Track these selection mechanisms; audit samples and weighting require known nonzero inclusion probabilities and measured outcomes.
If unknown outcomes remain, state bounds or sensitivity, not false certainty.
Apply least privilege, encrypted transport and storage, tokenization, retention enforcement, and access audit.
Do not log raw card numbers, security codes, or authentication secrets in model features or observability systems.
Incident logs contain reason codes and versions, not cardholder secrets.

### 5e. Reproducibility and method-selection ledger
| ID | Source / artifact | Version and accessibility | Role and limitation |
|---|---|---|---|
| SRC-01 | [Course source coverage](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-01) | Current course plan, accessible | Establishes topic and separate-case scope; not production evidence. |
| SRC-02 | [Source directory inventory](../../../../../Templates/directory.txt) | Repository inventory, accessible | Names the original case; does not reproduce its contents. |
| SRC-03 | [Fraud evaluation reference](../../../02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) | Local draft | Method companion, not proof of performance here. |
| RUN-01 | No run exists | Unavailable | No training, replay, measurement, or reproducible runtime. |
| DATA-01 | No transaction extract exists | Unavailable | Cohort composition and label quality are unknown. |
Future runs need immutable data snapshot hashes, code commit, container digest, package and hardware versions, random seeds, training and policy configs, threshold and calibrator versions, and access-controlled artifact paths.
A seed alone cannot ensure deterministic distributed training or reproduce an inaccessible snapshot.
Method choice is a proposal; no candidate has been fitted or compared.

## 6. Analysis Walkthrough
The only supported finding is a conditional arithmetic comparison, not evidence that the proposed system improves a real portfolio.
Consider an illustrative, fully labeled cohort of 100,000 authorization attempts under one fixed policy.
Suppose 1,000 are truly fraudulent and 99,000 legitimate; prevalence is 1%.
Suppose a decline-only rule flags 600 fraud and 1,400 legitimate attempts, for 2,000 total declines.
The remaining 400 fraud and 97,600 legitimate attempts are approved.
No challenges are included in this toy confusion matrix; it does not estimate the effect of introducing challenge.
The truth labels are assumed known for every row solely to make arithmetic transparent; this assumption fails for actual declined transactions.
Precision is confirmed fraudulent declines divided by all declines when truth is fully observed.
Recall is confirmed fraudulent declines divided by all fraudulent attempts in the same cohort.
False-positive rate is legitimate declines divided by all legitimate attempts in the same cohort.
Derive each rate by partitioning the appropriate population, rather than dividing by only reviewed alerts.
```text
FORMULA [F-01]: precision = TP / (TP + FP)
WHERE: TP fraud declines [attempts]; FP legitimate declines [attempts].
CONSTRAINT: mutually exclusive fully labeled rows; TP+FP > 0.
NUMERICAL EXAMPLE: 600 / (600+1400) = 0.30 = 30%.
INTERPRETATION: 30% of declines are fraud in this constructed cohort.
FORMULA [F-02]: recall = TP / (TP + FN)
WHERE: FN approved fraudulent attempts [attempts]; TP as above.
CONSTRAINT: complete fraud labels; TP+FN > 0.
NUMERICAL EXAMPLE: 600 / (600+400) = 0.60 = 60%.
INTERPRETATION: 40% of fraudulent attempts pass this toy rule.
FORMULA [F-03]: false-positive rate = FP / (FP + TN)
WHERE: TN approved legitimate attempts [attempts]; FP as above.
CONSTRAINT: complete legitimate labels; FP+TN > 0.
NUMERICAL EXAMPLE: 1400 / (1400+97600) = 0.0141414 = 1.414%.
INTERPRETATION: roughly 14.14 of every 1,000 legitimate attempts are declined.
```
Independent partition check: 600 + 400 = 1,000 fraud; 1,400 + 97,600 = 99,000 legitimate.
The four cells sum to 100,000 and the decline count is 2,000, or 2% of requests.
Accuracy is 98,200 / 100,000 = 98.2%, yet 400 fraud attempts pass and 1,400 legitimate attempts fail.
An always-approve rule is 99% accurate in this artificial cohort but catches no fraud.
The toy policy's lower accuracy does not imply a worse business decision.
For a cost illustration only, assume $100 issuer loss per approved fraud, $5 friction cost per legitimate decline, and no other costs.
The action cost is 400 × $100 + 1,400 × $5 = $47,000 per 100,000 attempts.
Always approve costs 1,000 × $100 = $100,000 under the same simplified assumptions.
The difference of $53,000 is a counterfactual calculation with identical underlying fraud, not a measured effect.
The $5 is a stylized monetization, not observed customer harm or a justified price.
If missed-fraud loss is only $10, this rule costs $4,000 + $7,000 = $11,000 while always approve costs $10,000.
That reversal demonstrates why costs and action effects require validation rather than importing the first result.
Break-even in this restricted example occurs when 600 prevented fraud losses equal $7,000 in friction: $11.67 per prevented fraud, approximately.
This algebra excludes challenge costs, partial recoveries, transaction-value heterogeneity, and behavioral adaptation.
The 600 declined frauds are presumed to have caused identical loss if approved; that is an unobserved counterfactual in real data.
At a different base rate, precision changes even if sensitivity and false-positive rate stay fixed.
For 100,000 attempts and 0.1% fraud, fixed 60% recall gives 60 fraud declines.
Applying the original 1,400/99,000 false-positive rate to 99,900 legitimate attempts gives about 1,413 legitimate declines.
Precision is about 60/(60+1,413) = 4.1%; this is a prevalence thought experiment, not a forecast.

## 7. The Finding
The illustrative arithmetic shows that a decision policy can have lower accuracy than always-approve while producing lower assumed cost, and that the ranking can reverse when loss assumptions change.
It does not establish the performance, calibration, fairness, or causal impact of any actual model.
The appropriate next decision is to specify costs and maturity, audit labels and features, and compare the candidate with the existing policy under explicit safety constraints.
Only a properly governed prospective evaluation can support a rollout decision.

## 8. System/Solution Design
### Offline and online architecture
```text
OFFLINE: authorization + timestamped feature history + outcomes
  -> audited as-of joins -> matured temporal splits -> train/calibrate
  -> frozen evaluation -> registry + policy approval -> signed release
ONLINE: network request -> gateway/tokenize -> mandatory rules
  -> bounded feature read -> model/calibrator -> action policy
  -> approve / eligible challenge / decline -> network response
  -> durable decision audit -> outcomes and monitoring (asynchronous)
```
The feature store is populated asynchronously from canonical events, with availability timestamps and versioned definitions.
The gateway validates schema and checks a stable attempt ID before invoking the decision service.
The request carries an absolute deadline and a trace ID, not a raw-card log payload.
The policy response contains action, reason category, decision ID, policy version, model version if used, and fallback indicator.
The model cannot itself send a payment response or override mandatory controls.
The policy engine takes a channel eligibility flag and must not emit challenge where no supported challenge path exists.
Challenge completion is a separate event and may lead to a new authorized action according to the network contract.
A customer abandonment is not proof of fraud.
The same request retry receives the persisted original decision, not a second independent action.
Any failure to persist an idempotency record must have an explicitly approved safe response; do not claim exactly-once behavior merely from message delivery settings.
A reversal is tied to its original authorization but does not erase the original decision.
On an expired decision deadline, return the channel-specific approved fallback rather than waiting for inference.
The fallback rules are versioned and separately monitored, and their error budget is not hidden inside model availability.
A stale or missing critical feature causes a preconfigured bounded fallback or missing-aware model path, not fabricated data.
The exact selection must be approved per feature and channel before deployment.
Durable logs can arrive after the response, but the business action and idempotency record need a coherent failure contract.
The platform owner must test partial failures between action persistence, response, and logging.
A fallback decision records the failure reason and version to enable later reconstruction.

### Sizing and deadline budget
The following workload is illustrative and not a benchmark.
Assume 1,000 requests/s average and a 3,000 requests/s peak lasting 10 minutes.
At a 4,000 requests/s steady tested service capacity, the peak uses 75% of nominal capacity; tail latency still needs measurement.
Suppose a single feature partition can process 2,000 updates/s during the peak while arrivals are 3,000/s for 600 s.
By queue conservation, 1,000 updates/s × 600 s = 600,000 queued updates, assuming constant rates and no losses.
If arrivals then fall to 1,000/s and service remains 2,000/s, net drain is 1,000 updates/s and drain time is 600 s.
If arrivals remain at or above service capacity, the backlog does not drain; the arithmetic is not a tail-latency guarantee.
A backlog can make velocity features stale even while the scoring endpoint is healthy.
Proposed internal budget allocation: 10 ms gateway, 20 ms feature read, 20 ms inference, 10 ms policy, and 40 ms remaining for transport and variance.
These allocations sum to 100 ms, but component p99 values cannot generally be added to obtain an end-to-end p99.
Measure the end-to-end distribution under peak and failure conditions; tune or shed work when the deadline is threatened.
The operating team needs separate dashboards for total request rate, fallback rate, deadline misses, feature age, queue lag, and action distribution.

## 9. Evaluation & Validation
No evaluation has been executed.
First reproduce the baseline and candidate on point-in-time historical data; this tests implementation consistency, not causal benefit.
Use rolling temporal folds with frozen feature availability, label maturity, and channel inclusion rules.
Keep the final forward test unmodified by feature selection, calibration, and threshold tuning.
Report precision, recall, false-positive rate, calibration, score coverage, and loss components with cohort counts and uncertainty.
AUC alone cannot establish net value at an operating threshold.
Evaluate calibration by score band and salient channel and customer slices, including counts and uncertainty for sparse groups.
Measure total challenge rate, legitimate-decline rate, successful authentication, abandonment, and complaints per eligible request or customer as appropriate.
For timely loss, compare mature cohorts after the same 90-day observation window; report unresolved and late-label rates.
Study label-delay curves and run sensitivity analyses for unresolved and selectively observed outcomes.
Shadow mode checks features, availability, and latency but cannot observe counterfactual decisions or their financial effect.
Before live testing, agree assignment unit, risk exclusions, capacity, stop rules, effect estimator, and sample size with governance.
A randomized eligible-account assignment avoids splitting related attempts across policies when interference within account is plausible.
Retain treatment assignment across retries and report exclusions, noncompliance, and assignment imbalance.
Compare intent-to-treat outcomes over all assigned eligible attempts, including fallback and missing-score requests.
Account-level clustering and calendar effects require appropriate uncertainty intervals rather than independent-attempt assumptions.
An independent audit of declined and low-score cases can inform missing labels, but no audit can reveal every unobserved counterfactual.
No model is authorized to override mandatory protections for experimentation.
Prospective stop criteria include critical service failure, customer harm, statistically and practically material guardrail deterioration, or security exposure.
The thresholds and statistical procedure are deliberately not invented; governance must set them before launch.
Progress from offline replay to shadow, then a small guarded trial, then staged expansion only after each gate is met.
Pre-register rollback ownership and criteria; preserve the comparator policy and its data path.

## 10. Action Taken
No organizational action, deployment, approval, training, experiment, or customer intervention is evidenced.
This case recommends documenting the existing rules and data contracts before approving the proposed design.
The proposed owner is the issuer fraud product owner, supported by security, model risk, platform, and customer operations.
The first deliverable would be a jointly approved decision and cost specification, not an unreviewed production threshold.
A deployment decision remains conditional on audited data, tested fallback, legal review, and prospective results.

## 11. Outcome & Results
| Result type | Status | Evidence |
|---|---|---|
| Constructed confusion matrix | Arithmetic only | Section 6 assumptions; not sampled transactions |
| Offline comparison | Not executed | No data or model artifacts |
| Pilot | Not executed | No trial records |
| Production impact | Not measured | No implementation or outcome feed |
| Customer impact | Not measured | No complaints, false-decline, or appeal observations |
| System latency and throughput | Not measured | Sizing inputs and deadline allocations are illustrative |
| Return on investment | Not measured | No observed costs or attributable benefits |
The example's $53,000 difference must never be presented as realized savings.
Even an apparent offline gain would not establish the causal business result without a valid comparison and mature outcomes.

## 12. Failure Modes & What Went Wrong
There is no reported incident and no observed failed implementation.
The following are distinct hypothetical fault-injection or review scenarios, not historical events.
Each states a competing explanation, a discriminating check, a bounded response, and a way to verify it.
### Diagnostic 01: late velocity updates
Observed-if-simulated symptom: Fraud bursts pass with unexpectedly low velocity.
Competing explanations: Stream lag or a real quiet period.
Discriminating evidence: Compare source event time, ingestion watermark and per-card history.
Bounded response: Mark aggregates stale and use the approved fallback.
Residual risk: Fallback can increase friction on legitimate traffic.
Verification: Inject delayed events and verify stale flag plus versioned fallback.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 02: future feature leakage
Observed-if-simulated symptom: Offline performance is implausibly strong.
Competing explanations: Post-decision outcome joined as a feature or genuine predictive signal.
Discriminating evidence: Inspect feature availability timestamps and rebuild as-of joins.
Bounded response: Remove post-decision fields and retrain on frozen folds.
Residual risk: Loss of apparent performance may trigger pressure to relax controls.
Verification: Replay historical events in arrival order and compare serving values.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 03: label maturity drift
Observed-if-simulated symptom: Recent fraud rate falls sharply.
Competing explanations: Unmatured disputes or genuinely less fraud.
Discriminating evidence: Compare cohort-age curves with prior cohorts and ingestion health.
Bounded response: Censor immature outcomes and delay comparative decisions.
Residual risk: Delayed reporting can mask an actual attack.
Verification: Backfill mature cohorts and reconcile label-age distributions.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 04: decline label selection
Observed-if-simulated symptom: Declines appear much cleaner than approvals.
Competing explanations: Missing counterfactual labels or actual policy precision.
Discriminating evidence: Audit label ascertainment by action and source.
Bounded response: Report unknowns and use independent evidence where feasible.
Residual risk: Review labels are also selected and cannot establish all counterfactuals.
Verification: Check audit inclusion probabilities and unresolved bounds.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 05: inference timeout
Observed-if-simulated symptom: Model scores are absent during a traffic spike.
Competing explanations: Model saturation or feature-store latency.
Discriminating evidence: Trace per-stage deadline and queue time.
Bounded response: Return the preapproved channel fallback before deadline.
Residual risk: A fallback can be less discriminating than the model.
Verification: Load-test end-to-end timeouts and reconcile fallback counters.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 06: duplicate retry
Observed-if-simulated symptom: One authorization attempt produces two actions.
Competing explanations: Retry without a stable idempotency key or two genuinely separate attempts.
Discriminating evidence: Compare network attempt ID, retry headers and decision IDs.
Bounded response: Persist and reuse a terminal response under the agreed key.
Residual risk: Overly broad deduplication may suppress a real second purchase.
Verification: Replay retries and separate-attempt cases with expected action counts.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 07: currency mismatch
Observed-if-simulated symptom: Foreign transactions dominate high-risk alerts.
Competing explanations: Currency conversion units or real geographic attack.
Discriminating evidence: Compare raw currency, conversion timestamp and converted amounts.
Bounded response: Use versioned as-of currency conversion or explicit missing state.
Residual risk: Using a current exchange rate creates historical leakage.
Verification: Replay known amounts across currencies and rate boundaries.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 08: time-zone boundary
Observed-if-simulated symptom: Daily velocity jumps around midnight.
Competing explanations: Local-time conversion or real concentrated activity.
Discriminating evidence: Compare UTC event timestamps and configured window boundaries.
Bounded response: Use documented UTC windows and explicit customer-local derivations.
Residual risk: A global UTC day may not match customer behavior.
Verification: Test daylight-saving transitions and late events.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 09: out-of-order events
Observed-if-simulated symptom: A velocity count changes after an authorization.
Competing explanations: Late arrival or incorrect replay ordering.
Discriminating evidence: Inspect event and availability timestamps for the affected key.
Bounded response: Use only information actually available by decision time.
Residual risk: Dropping late events forever can corrupt later aggregates.
Verification: Replay late arrivals and verify subsequent but not earlier decisions.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 10: hot card key
Observed-if-simulated symptom: One partition backs up while fleet averages look healthy.
Competing explanations: A concentrated key or a generally underprovisioned cluster.
Discriminating evidence: Compare partition-level rates and key frequency.
Bounded response: Isolate or redesign hot-key aggregation while preserving semantics.
Residual risk: Naive sharding can double-count or miss cross-shard velocity.
Verification: Stress a hot key and compare with a single-key reference.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 11: calibration shift
Observed-if-simulated symptom: Observed fraud by score band differs from predictions.
Competing explanations: Population shift or delayed/selected labels.
Discriminating evidence: Compare matured calibration by channel and label coverage.
Bounded response: Recalibrate on representative prior cohorts only after approval.
Residual risk: Recalibration on biased observed outcomes can worsen decisions.
Verification: Check reliability bands on a frozen forward cohort.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 12: threshold misconfiguration
Observed-if-simulated symptom: Decline rate jumps at release.
Competing explanations: Changed threshold or a shifted score distribution.
Discriminating evidence: Compare policy hashes and score histograms before and after.
Bounded response: Roll back the policy independently of the model.
Residual risk: Rolling back a needed security rule could expose fraud.
Verification: Replay release requests against both signed policy versions.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 13: challenge unsupported
Observed-if-simulated symptom: Policy requests authentication on an ineligible channel.
Competing explanations: Bad capability mapping or changed network contract.
Discriminating evidence: Inspect channel capability version and response code.
Bounded response: Route unsupported traffic to its approved non-challenge policy.
Residual risk: Fallback action may disproportionately burden one channel.
Verification: Run contract tests for every channel and capability version.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 14: challenge abandonment
Observed-if-simulated symptom: Apparent fraud falls after more challenges.
Competing explanations: Attack deterrence or legitimate customer abandonment.
Discriminating evidence: Compare randomized outcomes including abandonment and complaints.
Bounded response: Adjust policy only after considering both loss and friction.
Residual risk: Treating abandonment as fraud biases calibration.
Verification: Trace all challenge terminal states by assigned cohort.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 15: merchant identifier collision
Observed-if-simulated symptom: One merchant looks like a fraud hotspot.
Competing explanations: Identifier collision or a genuinely risky merchant.
Discriminating evidence: Inspect namespace, acquirer and identifier mapping history.
Bounded response: Correct entity namespace and reconstruct affected features.
Residual risk: Overmerging or oversplitting changes velocity counts.
Verification: Test fixture with identical IDs from separate namespaces.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 16: token rotation
Observed-if-simulated symptom: Card velocity unexpectedly resets.
Competing explanations: Token rotation or genuinely new card activity.
Discriminating evidence: Compare token lifecycle events with authorized linkage records.
Bounded response: Use an approved stable pseudonymous linkage.
Residual risk: Persistent linkage increases privacy exposure.
Verification: Replay token rotation without crossing unrelated accounts.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 17: negative amounts
Observed-if-simulated symptom: Refund-like events inflate purchase features.
Competing explanations: Sign conversion or true reversal events.
Discriminating evidence: Reconcile message type and signed amount against ledger.
Bounded response: Exclude or separately represent reversals by contract.
Residual risk: Discarding valid correction events can overstate exposure.
Verification: Test purchase, partial reversal and full reversal sequences.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 18: missing amount
Observed-if-simulated symptom: High missingness affects one merchant channel.
Competing explanations: Schema migration or actual unavailable amount.
Discriminating evidence: Check schema version and field-presence telemetry.
Bounded response: Use explicit missingness and approved channel fallback.
Residual risk: Zero imputation may imply a free transaction.
Verification: Contract-test null, zero and absent values separately.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 19: feature skew
Observed-if-simulated symptom: Offline replay diverges from online scores.
Competing explanations: Different transformations or true unavailable online values.
Discriminating evidence: Compare versioned feature vectors for sampled decision IDs.
Bounded response: Unify the transformation contract and replay availability.
Residual risk: Centralization alone does not fix clock or source timing differences.
Verification: Verify identical input vectors produce identical scores.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 20: model artifact mismatch
Observed-if-simulated symptom: Score distribution changes without a planned model release.
Competing explanations: Wrong calibrator or changed population.
Discriminating evidence: Compare model, calibration and schema digests in audit events.
Bounded response: Restore a compatible signed bundle.
Residual risk: Mixing old model and new calibrator can silently corrupt probabilities.
Verification: Reject mismatched artifacts in a deployment rehearsal.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 21: poisoned merchant signal
Observed-if-simulated symptom: Attacker causes benign merchants to appear risky.
Competing explanations: Adversarial manipulation or organic fraud cluster.
Discriminating evidence: Inspect concentration, source provenance and independent signals.
Bounded response: Cap aggregate influence and investigate source integrity.
Residual risk: Overaggressive caps may suppress real attack signals.
Verification: Simulate an injected burst and measure bounded influence.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 22: training contamination
Observed-if-simulated symptom: Test scores are suspiciously good.
Competing explanations: Repeated accounts across time or future information.
Discriminating evidence: Trace entity links and feature timestamps across folds.
Bounded response: Use temporal holdout and appropriate account-aware sensitivity.
Residual risk: Holding out every returning account may change the target population.
Verification: Compare new-account and returning-account performance separately.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 23: appeal reversal
Observed-if-simulated symptom: Reported false declines rise after initially clean reports.
Competing explanations: Late appeal decisions or new policy harm.
Discriminating evidence: Track label version histories and cohort age.
Bounded response: Recompute mature metrics with revision lineage.
Residual risk: Overwriting old labels destroys the ability to reproduce reports.
Verification: Reconcile initial and revised outcome snapshots.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 24: score clipping
Observed-if-simulated symptom: Many scores concentrate at exactly zero or one.
Competing explanations: Numeric overflow or genuinely extreme predictions.
Discriminating evidence: Inspect raw logits, calibrator inputs and serialization.
Bounded response: Use numerically stable transformations and validated bounds.
Residual risk: Blind clipping can hide an upstream artifact mismatch.
Verification: Run boundary vectors and compare with a high-precision oracle.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 25: sensitive proxy
Observed-if-simulated symptom: One customer segment experiences disproportionate friction.
Competing explanations: Proxy feature or different underlying traffic mix.
Discriminating evidence: Examine eligible population, threshold outcomes and uncertainty by slice.
Bounded response: Escalate to governance and assess feasible mitigations.
Residual risk: Removing a feature alone does not guarantee fair outcomes.
Verification: Evaluate mature group outcomes and customer appeals after mitigation.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 26: feedback loop
Observed-if-simulated symptom: A merchant risk signal worsens after elevated declines.
Competing explanations: Selective investigation feedback or increased actual fraud.
Discriminating evidence: Separate exposure, investigations and confirmation rates.
Bounded response: Version labels and analyze action-specific ascertainment.
Residual risk: Reweighting without known selection probabilities is unreliable.
Verification: Compare audited samples with routine feedback by action.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 27: response lost
Observed-if-simulated symptom: Decision is committed but network response is lost.
Competing explanations: Network failure or policy-service crash.
Discriminating evidence: Correlate durable decision ID with retry and response logs.
Bounded response: Return the stored decision on retry within the valid contract.
Residual risk: A stale decision may be invalid after an expiry or changed request.
Verification: Inject loss between commit and response and reconcile actions.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 28: event log loss
Observed-if-simulated symptom: A customer receives a decision absent from analytics.
Competing explanations: Asynchronous logging loss or ingestion lag.
Discriminating evidence: Reconcile gateway decisions against durable event offsets.
Bounded response: Recover from durable source and page on reconciliation gaps.
Residual risk: Backfill without deduplication inflates counts.
Verification: Drop a consumer and confirm replay restores one canonical event.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 29: bad fallback
Observed-if-simulated symptom: Fallback rate rises and legitimate declines spike.
Competing explanations: Overly broad emergency rule or changing attack mix.
Discriminating evidence: Slice actions by fallback reason and compare comparable channels.
Bounded response: Restore the last approved fallback and limit blast radius.
Residual risk: Turning off mandatory controls would be unsafe.
Verification: Exercise fallback scenarios and measure action distributions.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 30: stale rule bundle
Observed-if-simulated symptom: Different instances return inconsistent actions.
Competing explanations: Partial rollout or nondeterministic inputs.
Discriminating evidence: Compare bundle hashes and inputs per instance.
Bounded response: Pin versions and stop the rollout until instances converge.
Residual risk: Forceful restart may discard in-flight state.
Verification: Use a canary and verify version convergence with trace IDs.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 31: sampling bias
Observed-if-simulated symptom: Manual review precision looks excellent.
Competing explanations: Only high-score alerts reviewed or real model improvement.
Discriminating evidence: Compare review selection probability and score coverage.
Bounded response: Sample within safe audited strata and disclose unresolved outcomes.
Residual risk: Manual labels may still be erroneous or delayed.
Verification: Reconcile sampling frame, weights and adjudication quality.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 32: training-serving schema
Observed-if-simulated symptom: A category becomes unknown after a producer release.
Competing explanations: Vocabulary drift or mapping regression.
Discriminating evidence: Compare producer schema version and unknown-category rate.
Bounded response: Route unknowns through an explicit safe representation.
Residual risk: Treating unknown as a common category hides new attacks.
Verification: Run forward-compatible schema contract tests.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 33: threshold overfit
Observed-if-simulated symptom: Validation gains disappear on a new month.
Competing explanations: Repeated threshold tuning or genuine seasonal shift.
Discriminating evidence: Count tuning trials and compare untouched forward cohorts.
Bounded response: Freeze selection and evaluate once on the final holdout.
Residual risk: Repeatedly peeking at the holdout makes it another validation set.
Verification: Reproduce the preregistered selection on independent time windows.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 34: base-rate change
Observed-if-simulated symptom: Precision collapses while recall seems stable.
Competing explanations: Lower prevalence or degraded ranking.
Discriminating evidence: Compare mature prevalence and class-conditional score distributions.
Bounded response: Reassess action thresholds with costs and capacity.
Residual risk: Threshold movement before label maturity can amplify harm.
Verification: Use the prevalence sensitivity calculation and mature holdout.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 35: manual override
Observed-if-simulated symptom: Analysts reverse many automated declines.
Competing explanations: A policy defect or different override eligibility.
Discriminating evidence: Compare reason codes and independently adjudicated outcomes.
Bounded response: Give operations a bounded override with audit trail.
Residual risk: Override results are selected and cannot alone prove policy quality.
Verification: Audit a sample of overrides and non-overrides.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 36: privacy leak
Observed-if-simulated symptom: Debug traces contain sensitive card data.
Competing explanations: Unsafe logging or a misconfigured upstream gateway.
Discriminating evidence: Inspect redacted samples and access logs under incident procedures.
Bounded response: Disable the leaking sink and rotate exposed credentials if required.
Residual risk: Deleting evidence without incident coordination impairs response.
Verification: Test redaction on representative nested error payloads.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 37: transaction splitting
Observed-if-simulated symptom: Many small purchases evade amount rules.
Competing explanations: Adversarial splitting or normal installment behavior.
Discriminating evidence: Examine short-window linked attempts and merchant context.
Bounded response: Add time-window velocity subject to false-positive review.
Residual risk: Customer linkage and aggregate features create privacy risks.
Verification: Evaluate on mature historical sequences and simulated split attempts.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 38: replay duplication
Observed-if-simulated symptom: Feature counts double after consumer recovery.
Competing explanations: At-least-once event delivery or legitimate repeated attempts.
Discriminating evidence: Compare event IDs, source offsets and distinct transaction keys.
Bounded response: Make state updates idempotent using event identity.
Residual risk: Deduplication retention shorter than retry horizon permits repeats.
Verification: Replay a checkpoint boundary and reconcile counts with a reference.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 39: false zero freshness
Observed-if-simulated symptom: A store returns zero for an expired feature.
Competing explanations: Expired value encoded as zero or true no-history customer.
Discriminating evidence: Inspect feature age, missingness and store TTL.
Bounded response: Return stale and absent indicators separately.
Residual risk: Aggressive TTL may force too many fallback decisions.
Verification: Test fresh zero, stale zero and absent feature cases.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 40: cohort interference
Observed-if-simulated symptom: Treatment seems effective per attempt but users retry.
Competing explanations: Retry displacement or measurement error.
Discriminating evidence: Link eligible attempts by randomized account over a fixed horizon.
Bounded response: Measure account-level outcomes including subsequent attempts.
Residual risk: Entity linkage can be incomplete or violate privacy rules.
Verification: Compare attempt and account estimands in the experiment.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 41: investigation overload
Observed-if-simulated symptom: Review queue grows despite better alert precision.
Competing explanations: Higher alert volume or reduced investigator throughput.
Discriminating evidence: Measure arrivals, completions, aging and staffing separately.
Bounded response: Throttle discretionary alerts under approved capacity rules.
Residual risk: A throttle may suppress emerging high-loss cases.
Verification: Run a surge exercise and reconcile queue aging and escalations.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 42: network timeout
Observed-if-simulated symptom: Issuer p99 looks healthy but customers time out.
Competing explanations: Unmeasured network path or issuer boundary definition.
Discriminating evidence: Compare correlated network and issuer timestamps.
Bounded response: Budget the complete contracted path and return before the external deadline.
Residual risk: Client clocks may not be synchronized.
Verification: Trace synthetic requests end to end with synchronized clocks.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 43: retrain regression
Observed-if-simulated symptom: New model improves average loss but harms a channel.
Competing explanations: Channel shift or unstable rare-category learning.
Discriminating evidence: Inspect channel-level matured outcomes and uncertainty.
Bounded response: Hold the release and examine support and feature differences.
Residual risk: Tiny slices can produce noisy apparent regressions.
Verification: Use a predefined slice gate and independent temporal test.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 44: risk score misuse
Observed-if-simulated symptom: A calibrated score is treated as an action effect.
Competing explanations: Confusion between fraud risk and intervention benefit.
Discriminating evidence: Inspect whether policy uses action-specific outcome estimates.
Bounded response: Evaluate policy causally rather than promising individual prevention.
Residual risk: Unethical exploration could cause avoidable fraud or customer harm.
Verification: Audit the decision specification and randomized policy comparison.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.
### Diagnostic 45: recovery overload
Observed-if-simulated symptom: Lag grows again immediately after recovery.
Competing explanations: Replay traffic competes with live ingestion or service capacity is insufficient.
Discriminating evidence: Separate live, replay, backlog and processing rates.
Bounded response: Reserve capacity for live deadlines and pace replay.
Residual risk: Throttled replay prolongs feature staleness.
Verification: Run recovery under concurrent peak load and confirm drain rate.
Evidence state: hypothetical; no incident or experiment is claimed.
Decision owner: platform and fraud owners jointly evaluate customer impact.
Escalation: preserve the decision trace and use the approved incident process.
Measurement: compare eligible denominators and versions before and after mitigation.
Boundary: a single alert is not sufficient to establish root cause.

## 13. Transferable Lesson
Initial hypothesis: a better fraud ranker can improve an authorization decision only if the policy, labels, latency, and customer constraints align.
The toy arithmetic supports the narrower statement that accuracy and assumed cost can rank policies differently.
It does not verify that any trained ranker is better in production.
Transfer the decision contract, time-aware data controls, and explicit fallback to other real-time risk settings.
Do not transfer the 1% prevalence, $5 friction cost, $100 loss, 90-day maturity, or 100 ms budget as domain facts.
A different rail may not offer challenge; a different issuer may bear a different fraction of loss.
A process with delayed manual decisions requires a distinct queue and capacity design.

## 14. Connections & Cross-References
### 14a. Methods and contrasting cases
The [fraud evaluation sheet](../../../02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) explains calibration and constrained selection; it is a method reference, not a result for this case.
The [collection index](INDEX.md) lists planned claims and account-takeover cases; neither is represented here as an authored comparison.
The [source coverage brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-01) maps the separate source topics and marks the wider collection as planned.
The visible source identifier is DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling/01_Real_Time_Payment_Card_Fraud_Detection.md.
Its original contents were not available in the course source inventory; this is an independently constructed illustrative case, not a summary of that file.
### 14b. Open questions
| Question | Why unknown | Evidence to obtain | Priority / owner |
|---|---|---|---|
| Does the policy improve net outcomes? | No policy trial or actual costs | Pre-registered randomized comparison with mature outcomes | High / fraud product |
| How biased are the labels? | No independent audit or maturity curves | Stratified audit, source reconciliation and delay analysis | High / model risk |
| Can the platform meet the deadline during recovery? | No traces, capacity tests or deployment | End-to-end load and failure exercise including hot keys | High / platform |
| Does intervention displace fraud or harm customers later? | No account-level follow-up | Longer-window linked randomized analysis with privacy approval | Medium / risk analytics |
### 14c. Interview pack
Candidate prompts precede interviewer notes; all values are illustrative and none describes a real deployment.
#### Q01 — Framing
Candidate prompt: What is the action, unit, comparator and owner?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: An authorization attempt is assigned approve, eligible challenge or decline by the issuer fraud policy; compare with the existing rules-only policy.
Reasoning: The owner must jointly constrain fraud loss, friction and the issuer response deadline.
Acceptable alternative: A separate analyst review is acceptable after the synchronous decision, not as an assumed instant response.
Follow-up and answer: What changes for an unsupported challenge channel? Use its approved non-challenge policy.
Red flag: Optimizing accuracy alone ignores both action cost and the rare-fraud base rate.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Sections 1–4.
#### Q02 — Label design
Candidate prompt: Is every transaction without a chargeback a negative?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: No; use confirmed fraud, confirmed legitimate when evidence exists, and unresolved, with label maturity and source version.
Reasoning: Chargebacks are delayed and selectively observed, especially for declined transactions.
Acceptable alternative: A probabilistic label model may be considered only with identified assumptions and validation.
Follow-up and answer: What if reporting takes longer than 90 days? Extend or censor and report maturity curves.
Red flag: Calling all unresolved attempts legitimate biases both training and metrics.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 5a.
#### Q03 — Point-in-time join
Candidate prompt: Can the latest merchant risk value be joined to past transactions?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: No; use the value both event-time eligible and actually available by the decision.
Reasoning: A later investigation may otherwise enter the feature set before the decision in offline replay.
Acceptable alternative: A versioned snapshot is acceptable if it preserves availability and historical correction semantics.
Follow-up and answer: What if updates arrive late? Later decisions may use them; earlier ones cannot.
Red flag: Using final fraud status as a feature creates direct leakage.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Sections 5d and 12.
#### Q04 — Confusion matrix
Candidate prompt: Calculate precision and recall for 600 true positives, 1,400 false positives and 400 false negatives.
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Precision is 600/2000 = 30%; recall is 600/1000 = 60%.
Reasoning: Precision conditions on declines; recall conditions on all fraudulent attempts.
Acceptable alternative: At zero denominator report undefined rather than zero without an agreed convention.
Follow-up and answer: What happens if 200 additional fraud attempts become declines without other changes? Recall becomes 800/1000 = 80%.
Red flag: Dividing 600 by 100,000 computes neither precision nor recall.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 6 F-01 and F-02.
#### Q05 — Decision cost
Candidate prompt: Under the toy assumptions, compare the decline rule with always approving.
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: The decline rule costs 400×$100 + 1400×$5 = $47,000; always approve costs $100,000.
Reasoning: The $53,000 difference assumes equal counterfactual fraud loss and omits other costs.
Acceptable alternative: Use heterogeneous loss and intervention effects if defensibly measured.
Follow-up and answer: If fraud loss is $10, the decline rule costs $11,000 versus $10,000.
Red flag: Calling the illustrative difference measured savings is unsupported.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 6.
#### Q06 — Prevalence
Candidate prompt: At 0.1% fraud in 100,000 attempts, what happens if recall and false-positive rate stay fixed?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: There are 100 fraud attempts and 60 fraud declines; about 1,413 of 99,900 legitimate attempts are declined.
Reasoning: Precision becomes about 60/1473 = 4.1%, far below the 1% prevalence example.
Acceptable alternative: An exact fractional expectation is acceptable because this is a rate calculation, not a realized count.
Follow-up and answer: What assumption can fail? Conditional score distributions and error rates may shift too.
Red flag: Assuming precision is invariant to prevalence confuses conditional denominators.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 6 F-01 to F-03.
#### Q07 — Serving and retries
Candidate prompt: How do you avoid two distinct actions on a retried request?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Persist one decision under a stable attempt key and return it for a valid retry.
Reasoning: The failure boundary between persistence, network response and logging must be designed explicitly.
Acceptable alternative: A transactional store or another equivalent durable protocol can work if tested.
Follow-up and answer: What about a new purchase with the same card? It needs a distinct attempt key.
Red flag: Claiming broker exactly-once alone guarantees end-to-end exactly-once is wrong.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 8.
#### Q08 — Capacity
Candidate prompt: A partition receives 3,000 updates/s and processes 2,000/s for 600 s; how large is its backlog?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Under constant rates and no loss, backlog is (3000−2000)×600 = 600,000 updates.
Reasoning: If arrival falls to 1,000/s with 2,000/s service, draining takes 600 seconds.
Acceptable alternative: A queueing simulation may be used when rates or service times vary.
Follow-up and answer: What if arrival remains 2,000/s? There is no positive net drain.
Red flag: Calling nominal capacity a measured p99 latency is unjustified.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 8 sizing.
#### Q09 — Ambiguous diagnosis
Candidate prompt: A recent fraud rate falls while challenge rate rises; what happened?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: It may reflect deterrence, immature labels, selective observation, or legitimate abandonment.
Reasoning: Compare mature randomized cohorts, label-delay curves, completion and abandonment.
Acceptable alternative: A carefully bounded observational analysis can generate hypotheses but not automatically causal effects.
Follow-up and answer: What if challenge eligibility changed? Stratify by the eligibility definition and assignment.
Red flag: Declaring success from the recent rate alone overlooks selection and customer harm.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Sections 9 and 12.
#### Q10 — Wrong method
Candidate prompt: Why not tune a threshold until the final test shows the highest savings?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Repeated test tuning leaks selection information and invalidates the final holdout as an independent check.
Reasoning: Choose model and threshold on earlier temporal folds, freeze them, then evaluate once on the forward test.
Acceptable alternative: Nested temporal selection is acceptable with a separate untouched final period.
Follow-up and answer: What if the threshold is changed after release? Treat it as a new policy with new evaluation.
Red flag: Calling the same repeatedly inspected dataset a final test is not protection.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Sections 5c and 9.
#### Q11 — Changed constraint
Candidate prompt: What if the 100 ms target becomes 50 ms?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Reallocate the budget using measured end-to-end traces; remove or precompute optional work and use the approved fallback on deadline.
Reasoning: A faster model is insufficient if the feature read or network path dominates.
Acceptable alternative: A simpler model or local cache is acceptable if availability, freshness and accuracy are validated.
Follow-up and answer: What if no safe model path fits? Continue the validated rules-only policy.
Red flag: Adding component p99s is not a proof of the combined end-to-end p99.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 8.
#### Q12 — Causal validation
Candidate prompt: Why is shadow mode insufficient to establish savings?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Shadow scores do not change decisions, so outcomes under its alternative actions are not observed.
Reasoning: Randomize eligible accounts to policies with guardrails and compare intent-to-treat mature outcomes.
Acceptable alternative: A justified quasi-experiment may be used where randomization is impossible, with explicit confounding limits.
Follow-up and answer: Why account-level assignment? Related retries can otherwise contaminate comparison.
Red flag: Ignoring fallback traffic or analyzing only scored attempts breaks the assigned-population comparison.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Section 9.
#### Q13 — Operational recovery
Candidate prompt: What should happen when critical features are stale?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Apply the approved per-channel fallback or explicitly validated missing-aware path before the deadline.
Reasoning: Log stale reason, feature age, policy version and the action, then inspect stream lag and partitions.
Acceptable alternative: A cached feature is acceptable only within its validated freshness window.
Follow-up and answer: What if the backlog cannot drain? Add sustained capacity or reduce arrivals; a buffer alone cannot solve it.
Red flag: Replacing missing values with zero without an indicator changes their meaning.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Sections 8 and 12.
#### Q14 — Fairness and uncertainty
Candidate prompt: How should a subgroup friction difference be evaluated?
Clarify: use the stated toy cohort or proposed system; do not infer observed results.
Strong answer: Use eligible subgroup denominators, mature labels, action outcomes, counts, confidence intervals and an approved comparison.
Reasoning: Investigate proxies and different channel mix without assuming that a difference alone proves a cause.
Acceptable alternative: A privacy-preserving aggregate analysis is acceptable when individual data access is restricted.
Follow-up and answer: What if sample size is small? Report uncertainty and seek additional evidence rather than declaring parity.
Red flag: Suppressing all small groups or removing one proxy does not establish fairness.
Score 0: incorrect or unsafe conclusion.
Score 1: partially correct but missing the governing assumption.
Score 2: correct with assumptions and a defensible check.
Score 3: additionally explains a tradeoff or failure boundary.
Answer support: Sections 3, 9 and 12.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Domain | Issuer payment-card authorization fraud |
| Problem type | Approve, challenge, or decline under friction and response-time constraints |
| Method | Proposed calibrated tabular score plus versioned action policy; existing rules are comparator and fallback |
| Primary result | Illustrative arithmetic only; production results not measured |
| Key lesson | Decision cost, label observation and intervention effects matter more than accuracy alone |
| Tags | evaluation-mismatch; label-delay; cost-latency-tradeoff |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-01-01-001 / 1.0 / 2026-11-07 |
| Evidence level | Not measured; all numerical inputs constructed |
| Implementation stage | Proposed, not deployed; editorial status draft, not reviewed |
| Baseline / primary metric | Rules-only; mature net cost per 1,000 eligible attempts, not measured |
| Key limitation | Selective delayed labels and unobserved action counterfactuals |
| Review trigger / owner | Material policy, label, deadline, or rule change / issuer fraud product owner |
