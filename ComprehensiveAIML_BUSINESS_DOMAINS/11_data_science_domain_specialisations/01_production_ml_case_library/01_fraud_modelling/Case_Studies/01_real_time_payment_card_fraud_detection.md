---
case_id: CASE-01-01-001
case_title: Real-time payment-card fraud detection under a friction budget
case_version: "1.0"
created: 2026-11-07
last_reviewed: null
status: proposed
content_status: draft
implementation_stage: proposed
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
**Evidence declaration:** every count, probability, cost and capacity below is constructed for teaching. It is neither an observed portfolio nor a claim about the issuer's loss allocation, customer behaviour, challenge effectiveness, or model calibration.

### 6a. Point-in-time decision record
For decision `D`, freeze the request at issuer receipt time `t_D`. A feature `x` is eligible only when its underlying event occurred before `t_D` and its source made that value available by `t_D`. The score is therefore a function of the request and the eligible feature snapshot, not of a later dispute, investigation, final merchant mapping, or retry outcome.

```text
INVARIANT [PT-01]
INPUT: request r_D, feature candidates (value, event_time, availability_time).
STATE TRANSITION: retain a candidate iff event_time < t_D AND availability_time <= t_D;
                  otherwise mark the feature unavailable and apply its documented default.
OUTPUT: feature_snapshot_id, missing/stale indicators, score version, policy action.
RECOVERY: a late event may affect a later decision, never recompute D's original action.
```

The distinction matters for identity. `card_token`, `merchant_namespace + merchant_id`, device pseudonym, and account linkage are distinct keys with different collision and consent boundaries. A new token is not necessarily a new customer; a shared device is not evidence of common control; and a merchant identifier without its namespace is not a merchant entity. The feature contract retains each key's version and linkage confidence. It uses an explicit “unlinked” state rather than merging identities optimistically.

### 6b. Illustrative action-cost ledger
Let `p_a` be a **constructed policy scenario** probability that an attempt is fraud if approved; it is not an observed label and must not be backfilled onto a declined attempt. Let `L` be issuer-borne loss if an approved fraud materializes. It is not purchase amount: recovery, network allocation, and liability can make either value higher or lower. Let `q` be challenge completion, `e` the assumed conditional prevention effect after completion, `c_c` the challenge operating cost, `b` challenge abandonment probability, `c_b` its friction valuation, `c_d` the expected legitimate-decline friction valuation, and `c_o` the decline operating cost.

```text
FORMULA [AC-01]: C_approve = n * p_a * L
FORMULA [AC-02]: C_challenge = n * [p_a * (1 - q*e) * L + c_c + (1-p_a)*b*c_b]
FORMULA [AC-03]: C_decline = n * [(1-p_a)*c_d + c_o]
WHERE: n [attempts]; p_a, q, e, b [probabilities in 0..1];
       L, c_c, c_b, c_d, c_o [currency per applicable attempt].
CONSTRAINT: AC-02 is a scenario model, not a causal estimate; AC-03 counts immediate
            modeled friction and operations only. It assigns no invented fraud label or
            prevented-loss counterfactual to a declined attempt.
```

Use one constructed portfolio of 100,000 eligible attempts. The policy sends 95,000 to approve, 4,000 to a supported challenge, and 1,000 to decline. Assume `L=$120`, `q=0.80`, `e=0.70`, `c_c=$0.45`, `b=0.12`, `c_b=$4`, `c_d=$6`, and `c_o=$0.20`. The scenario probabilities are 0.0025, 0.0400, and 0.1200 respectively. They are inputs for arithmetic, not labels, scores, or observed calibration.

| Action | Attempts `n` | Scenario `p_a` | Formula substitution | Scenario cost | What it does **not** establish |
|---|---:|---:|---|---:|---|
| Approve | 95,000 | 0.0025 | `95,000*0.0025*$120` | $28,500.00 | Actual fraud count or issuer loss |
| Challenge | 4,000 | 0.0400 | `4,000*[.04*(1-.80*.70)*120+.45+.96*.12*$4]` | $12,091.20 | Completion, prevention, or abandonment effect |
| Decline | 1,000 | 0.1200 | `1,000*[.88*$6+$.20]` | $5,480.00 | Fraud among declined attempts or fraud displaced later |
| **Total** | **100,000** | — | Sum of constructed rows | **$46,071.20** | Net causal portfolio value |

Independent arithmetic: challenge residual loss is `4,000*.04*.44*$120 = $8,448`; operating cost is `$1,800`; modeled legitimate abandonment friction is `4,000*.96*.12*$4 = $1,843.20`; their sum is `$12,091.20`. Total cost per 1,000 eligible attempts is `$46,071.20 / 100 = $460.712`, or `$460.71` rounded to cents. The action counts sum to 100,000. This ledger supports a requirement for a cost schedule before threshold selection; it does not prove that this policy beats rules-only.

### 6c. Friction and capacity sensitivity
The proposed friction budget is at most 5% challenged attempts and at most 500 challenge abandonments per 100,000 eligible attempts. At 4,000 challenges, the challenge rate is `4,000/100,000 = 4%`. With the scenario abandonment rate of 12%, expected abandoned challenges are `4,000*.12 = 480`, leaving only 20 events of budget headroom. That narrow headroom is a policy-calibration constraint, not a claim that 480 customers will abandon.

| Scenario | Challenge count | Challenge rate | Abandonment assumption | Expected abandonments | Contact rate / contacts | Capacity consequence at 120 contacts/day |
|---|---:|---:|---:|---:|---:|---|
| Base | 4,000 | 4.0% | 12% | 480 | 4% / 160 | 1.33 days of one-day intake |
| Lower friction | 4,000 | 4.0% | 8% | 320 | 2% / 80 | 0.67 day |
| Higher friction | 4,000 | 4.0% | 16% | 640 | 6% / 240 | 2.00 days; breaches abandonment budget |
| Higher volume | 5,000 | 5.0% | 12% | 600 | 4% / 200 | Budget breach; do not expand without a changed guardrail |

The support calculation is `contacts / 120 contacts per day`; it assumes all contacts arrive at once, equal handling effort, and available staff. It is a deliberately crude capacity boundary, not an operations forecast. “Challenge succeeds” and “customer does not contact support” are different outcomes. A completed challenge can still be fraudulent; an abandoned challenge can be legitimate or fraudulent; neither can be converted into a fraud label.

### 6d. Calibration and causal boundary
For score band `k`, calibration is assessed only on a fixed mature, action-defined population:

```text
FORMULA [CAL-01]: observed_rate_k = mature_confirmed_fraud_k / eligible_attempts_with_mature_outcome_k
WHERE: numerator [attempts]; denominator [attempts]; k is a frozen score band.
CONSTRAINT: report unresolved outcomes separately; do not substitute approved-only labels for
            all attempted authorizations, and do not interpret this association as action effect.
NUMERICAL EXAMPLE: illustrative only; no mature cohort or calibration measurement exists.
INTERPRETATION: a well-calibrated approval-risk estimate still does not tell us whether
                challenge or decline changes loss more favorably than approval.
```

Approved-only outcomes select on the historic action: approved attempts may generate disputes and issuer-loss records, while declined attempts lack the unobserved “would have been approved” outcome. Challenges select on capability and policy as well as score. Investigation confirms a subset chosen for work; it is a downstream evidence process, not the authorization decision and not a synchronous substitute for it. Therefore the approved-only fraud rate, investigator precision, and challenge completion rate are each insufficient to estimate `e` or prevented loss.

The safe experimental estimand is the difference in mature net cost per eligible **assigned** attempt between preapproved policies, using account-level random assignment where linked retries could interfere. Include score failures, fallbacks, unsupported challenge channels, and all terminal action paths in the assigned denominator. Exclude only predeclared ineligible requests before assignment. Never randomize mandatory controls away, never use an unbounded exploration policy, and stop/roll back on pre-agreed customer-harm, security, availability, or friction triggers. If randomization is unavailable, an observational comparison can prioritize questions but cannot identify the action-specific effect without additional defensible assumptions.

### 6e. Mature-cohort measurement protocol
The proposed comparison begins with one immutable assignment record per eligible attempt: assignment cohort, account randomization key, authorization attempt ID, policy version, challenge-eligibility flag, chosen action, fallback reason, and decision time. It is then linked to outcomes without changing the original population. A reversal, retry, dispute, customer contact, recovery, or investigation outcome is an event attached to that record, not a replacement decision.

For reporting date `T`, define an attempt as mature only when `T - authorization_time >= 90 days` in the declared reporting time zone. This is a proposed administrative window. The dashboard must show three mutually exclusive states for every assigned attempt: mature with reconciled outcome, mature but unresolved, and not yet mature. A metric that drops the latter two without reporting them creates an artificial denominator improvement.

```text
FORMULA [MC-01]: mature_coverage = reconciled_mature_attempts / assigned_attempts_old_enough
FORMULA [MC-02]: ITT_net_cost = sum(reconciled issuer loss + pre-agreed action costs)
                              / all eligible assigned attempts
WHERE: all terms are grouped by assigned policy; costs [currency]; denominator [attempts].
CONSTRAINT: report unresolved mature records and not-yet-mature records separately;
            no individual declined attempt receives an unobserved fraud label.
NUMERICAL EXAMPLE: illustrative reporting shape only; no cohort has been measured.
INTERPRETATION: coverage qualifies evidence completeness, while ITT retains every assignment.
```

Mature coverage should be reported by action, channel, fallback reason, and assignment cohort because a high aggregate rate can conceal one poorly observed pathway. For example, a decline path may have very sparse downstream loss evidence precisely because it stopped the initial purchase. That sparsity is a reason not to claim individual prevented loss, not a reason to remove declines from the policy denominator.

The cost schedule is calibrated before a trial by the accountable owner, with finance or loss-accounting review for `L` and customer-operation review for friction valuations. Calibration means selecting and versioning assumptions that map outcomes to a decision objective; it does not mean that the risk score has been proved correct. Each schedule needs an effective date, currency, channel scope, loss-recovery convention, and explicit treatment of disputes still pending. Changing any of these produces a new policy analysis.

Threshold selection proceeds in this order:

1. Freeze point-in-time feature and label definitions on historical temporal folds.
2. Select candidate model and calibration procedure without the final forward period.
3. Enumerate policy thresholds subject to hard challenge, abandonment, decline, latency, and investigation-capacity constraints.
4. Present surviving policies with cost components, action counts, channel eligibility, and uncertainty rather than a single “best” score cutoff.
5. Freeze the rules-only comparator and proposed policy before the guarded prospective comparison.

The historical stage may eliminate a policy that cannot meet friction or latency constraints. It cannot validate `e`, the action-specific prevention effect, when historic actions determined which outcomes were observable. A prospective experiment estimates an average policy effect for its eligible population; it does not license a claim that any particular decline “would have prevented” a known amount.

### 6f. Fallback boundary trace
Consider one illustrative request that reaches the gateway with 100 ms remaining. Gateway validation uses 8 ms, feature read waits 55 ms and reports a stale critical velocity feature, leaving 37 ms. The configured boundary says the model path requires at least 30 ms inference plus 10 ms policy/response reserve. Because 37 is less than 40 ms, the service must skip model inference and return the signed channel fallback; waiting for a perhaps-fast inference violates the reserve.

The trace is: `REQUEST_RECEIVED -> VALIDATED -> FEATURE_STALE -> FALLBACK_SELECTED -> DECISION_PERSISTED -> RESPONSE_SENT -> ASYNC_AUDIT`. The audit record includes remaining budget, feature age, fallback reason, policy version, and terminal action. It must not invent a model score. On a valid retry with the same attempt key and request digest, the persisted action is returned even if the feature store has since recovered.

This trace is a design oracle, not evidence of measured latency. Its boundary is intentional: if the fallback itself cannot be returned before the real contracted deadline, the design is not safe enough to operate; adding a faster model does not repair a failing authorization path.

## 7. The Finding
The constructed ledger demonstrates a design requirement, not a winning policy: issuer loss, purchase amount, challenge operating cost, abandonment friction, and decline friction must be separate terms with declared denominators.
The decline row intentionally has no claimed prevented-fraud benefit because that counterfactual is unobserved.
At the base scenario, the 4% challenge rate fits the 5% rate budget but its 480 expected abandonments leave only 20 events below the illustrative limit of 500; the 16% sensitivity breaches it.
The appropriate next decision is to approve a measurable cost specification, mature-cohort definition, feature/identity audit, and guarded comparison against the actual rules-only policy.
No performance, calibration, fairness, causal effect, or production benefit has been observed.

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
| Constructed action-cost and friction ledger | Arithmetic checked | Section 6 constructed inputs; not sampled transactions |
| Mature-cohort measurement | Not executed | No authorization, outcome, or assignment extract |
| Offline comparison | Not executed | No data or model artifacts |
| Pilot or prospective experiment | Not executed | No trial record |
| Production/customer impact | Not measured | No deployment, complaints, or reconciled loss feed |
| System latency, fallback, and throughput | Not measured | Deadline trace and sizing inputs are illustrative design checks |
| Return on investment | Not measured | No attributable costs or benefits |

The `$46,071.20` ledger total is a conditional sum of constructed assumptions, not realized savings and not a comparison with rules-only. The base scenario's 480 abandoned challenges and the fallback trace are likewise arithmetic/design consequences, not observed customer behaviour or availability. No action-specific effectiveness has been measured: declined attempts lack their unobserved approval outcome, and challenge outcomes mix capability, customer choice, and risk. A valid business result needs the mature, assigned-policy comparison described in Sections 6e and 9.

## 12. Failure Modes & What Went Wrong
No incident or failed deployment is reported. These are proposed fault-injection and review diagnostics; each is a distinct mechanism, not evidence that it occurred.

### Diagnostic 01 — Late velocity state
**Symptom:** burst attempts receive unexpectedly low velocity values.
**Competing causes:** stream lag, hot-key skew, or a genuinely quiet history.
**Discriminator:** compare source event time, availability watermark, partition lag, and the retained decision snapshot.
**Bounded response:** mark state stale, use the approved channel fallback, and retain the stale reason.
**Recovery:** repair/replay state only for future decisions; never overwrite the original decision snapshot.
**Verification:** inject late events and confirm that only later requests see the repaired aggregate.
**Owner:** platform owner; fraud product owner owns the friction consequence.
**Boundary:** a fresh timestamp does not prove correct or complete state.

### Diagnostic 02 — Future-information leakage
**Symptom:** offline ranking or calibration is implausibly strong.
**Competing causes:** a later dispute/mapping entered the feature join, or there is a real predictive signal.
**Discriminator:** replay source arrivals and compare each serving feature vector with its offline equivalent.
**Bounded response:** exclude values unavailable at decision time and refit preprocessing inside temporal folds.
**Recovery:** invalidate affected offline reports and repeat selection before any policy change.
**Verification:** a deliberately delayed source field must be absent from earlier reconstructed decisions.
**Owner:** model risk owner.
**Boundary:** event time without availability time is not sufficient proof against leakage.

### Diagnostic 03 — Immature labels mistaken for improvement
**Symptom:** a recent cohort shows lower confirmed fraud after a policy change.
**Competing causes:** reporting delay, failed label ingestion, or a real change.
**Discriminator:** plot label source and adjudication status by cohort age and assignment, including unresolved counts.
**Bounded response:** censor immature cohorts from comparative conclusions and retain their denominator.
**Recovery:** rerun the report only after its predeclared maturity window or an approved survival adjustment.
**Verification:** later snapshots reconcile earlier unresolved rows rather than silently changing denominators.
**Owner:** risk analytics owner.
**Boundary:** a 90-day window is proposed, not proven adequate for this issuer.

### Diagnostic 04 — Approved-only selection
**Symptom:** approved transactions appear to have a different fraud rate than declines.
**Competing causes:** policy selection, true risk separation, or different label ascertainment.
**Discriminator:** tabulate action, label-source coverage, maturity, audit sampling probability, and missing outcomes together.
**Bounded response:** report action-specific observed outcomes; do not label a decline as fraud/nonfraud from its score.
**Recovery:** use an approved independent audit where safe, with its selection mechanism documented.
**Verification:** every reported rate names its action-defined denominator.
**Owner:** model risk owner.
**Boundary:** an audit informs label gaps; it cannot reveal every declined counterfactual.

### Diagnostic 05 — Challenge abandonment treated as fraud
**Symptom:** challenge completion falls while apparent fraud falls.
**Competing causes:** attack deterrence, checkout/customer abandonment, capability outage, or label delay.
**Discriminator:** compare assigned cohorts' completion, abandonment, contact, mature loss, and capability rates.
**Bounded response:** count abandonment as friction until independently adjudicated; do not relabel it as fraud.
**Recovery:** freeze threshold expansion if the abandonment guardrail is breached.
**Verification:** terminal challenge states reconcile to one eligible-assignment record.
**Owner:** fraud product owner with customer operations.
**Boundary:** an authentication success also does not prove legitimacy.

### Diagnostic 06 — Identity collision or overlinking
**Symptom:** a merchant or device feature becomes a concentrated risk hotspot.
**Competing causes:** namespace collision, token rotation, shared device, or a real concentration.
**Discriminator:** inspect identifier namespace, mapping version, linkage confidence, and collision samples.
**Bounded response:** separate ambiguous identities and route them to explicit unknown/unlinked states.
**Recovery:** rebuild only future aggregates under the corrected mapping; preserve historical mapping versions.
**Verification:** fixtures cover equal local IDs across acquirers, shared devices, and rotated tokens.
**Owner:** data owner with privacy review.
**Boundary:** joining more identifiers can increase both signal and unjustified linkage harm.

### Diagnostic 07 — Model/policy calibration mismatch
**Symptom:** mature observed rates differ materially by score band or channel.
**Competing causes:** population shift, delayed labels, action selection, or an artifact mismatch.
**Discriminator:** use frozen mature cohorts, coverage counts, and model/calibrator/schema digests.
**Bounded response:** hold threshold changes until the source of mismatch is isolated.
**Recovery:** recalibrate only on approved representative historical data and revalidate forward.
**Verification:** reliability reporting carries unresolved outcome rate and confidence intervals.
**Owner:** model risk owner.
**Boundary:** probability calibration does not estimate the benefit of challenge or decline.

### Diagnostic 08 — Deadline breach and unsafe fallback
**Symptom:** missing scores or issuer-boundary deadline expirations rise.
**Competing causes:** model saturation, feature latency, transport delay, or queueing.
**Discriminator:** trace absolute deadline remaining at gateway, feature, inference, policy, and response.
**Bounded response:** return the preapproved per-channel rules fallback before expiry and record the reason.
**Recovery:** shed optional enrichment, restore capacity, and separately evaluate fallback action distribution.
**Verification:** end-to-end load fault injection proves fallback is returned within the contracted boundary.
**Owner:** platform owner.
**Boundary:** component p99 sums do not prove end-to-end p99.

### Diagnostic 09 — Retry produces a second action
**Symptom:** one network attempt has conflicting approve/challenge/decline records.
**Competing causes:** missing idempotency persistence or two genuine attempts.
**Discriminator:** compare stable attempt key, request digest, retry metadata, and durable decision ID.
**Bounded response:** return the stored terminal response for a valid retry; reject digest conflict by contract.
**Recovery:** reconcile action and response logs without retrospectively changing the authorization action.
**Verification:** inject response loss after commit and assert one action across retries.
**Owner:** authorization platform owner.
**Boundary:** broker delivery semantics alone do not provide end-to-end exactly-once action.

### Diagnostic 10 — Unsupported challenge emitted
**Symptom:** policy selects challenge on a channel without a usable authentication route.
**Competing causes:** stale capability map, parser regression, or policy configuration error.
**Discriminator:** inspect channel capability version and network response code for each action.
**Bounded response:** use that channel's approved non-challenge policy and alert on any forbidden action.
**Recovery:** pin the last valid capability mapping and retest contracts before restoration.
**Verification:** exhaustive capability fixtures reject challenge where eligibility is false.
**Owner:** payment integration owner.
**Boundary:** channel labels and challenge eligibility are related but not interchangeable.

### Diagnostic 11 — Cost schedule is silently changed
**Symptom:** an apparently optimal threshold moves without score-distribution change.
**Competing causes:** altered loss allocation, friction valuation, or accidental configuration drift.
**Discriminator:** compare signed cost schedule and policy hash alongside action counts.
**Bounded response:** block unapproved cost changes; treat an approved schedule change as a new policy.
**Recovery:** restore the last approved bundle while governance reviews the economic assumption.
**Verification:** replay fixed requests against versioned schedules and explain each action difference.
**Owner:** fraud product owner.
**Boundary:** a purchase amount cannot substitute for reconciled issuer loss.

### Diagnostic 12 — Decline friction exceeds budget
**Symptom:** legitimate-decline complaints or appeals rise after a release.
**Competing causes:** threshold change, traffic mix change, fallback expansion, or complaint-ingestion change.
**Discriminator:** compare eligible denominators, policy/fallback reasons, mature outcomes, and complaint coverage.
**Bounded response:** roll back the policy independently of the score artifact if a pre-agreed guardrail trips.
**Recovery:** preserve the affected cohort and decision logs for review rather than deleting “bad” events.
**Verification:** canary reporting separates primary-policy and fallback friction.
**Owner:** fraud product owner with customer operations.
**Boundary:** complaints are a guardrail signal, not a complete legitimate-decline label.

### Diagnostic 13 — Investigation is confused with authorization
**Symptom:** a proposal promises analysts will decide an in-flight authorization.
**Competing causes:** unclear workflow ownership or an attempt to avoid policy calibration.
**Discriminator:** measure the authorization deadline against actual investigation queue and handling times.
**Bounded response:** keep authorization terminal; send post-decision investigation only where its separate objective permits.
**Recovery:** define investigation referral, evidence, capacity, and feedback as a downstream workflow.
**Verification:** process trace shows no analyst dependency on the synchronous response path.
**Owner:** operations owner.
**Boundary:** investigator confirmation is selected evidence, not a real-time action or universal truth label.

### Diagnostic 14 — Investigation capacity overload
**Symptom:** referral backlog grows despite high alert precision.
**Competing causes:** high arrivals, reduced completions, duplicate referrals, or changed referral eligibility.
**Discriminator:** measure queue arrivals, completions, age, staffing, and unique decision IDs separately.
**Bounded response:** throttle discretionary referrals under a published capacity rule; do not alter authorization retrospectively.
**Recovery:** prioritize by a separately validated investigation value policy and preserve deferred rows.
**Verification:** surge exercise reconciles backlog conservation and escalation counts.
**Owner:** operations owner.
**Boundary:** referral capacity cannot be used to justify slower synchronous authorization.

### Diagnostic 15 — Feature default changes meaning
**Symptom:** a producer release raises unknown or zero-valued feature rates.
**Competing causes:** schema drift, missing source, or genuine change in customer history.
**Discriminator:** inspect producer schema version, field presence, stale flag, and default reason.
**Bounded response:** route unknown, absent, stale, and genuine zero through distinct contract states.
**Recovery:** pin or translate the schema only after replay against known fixtures.
**Verification:** contract tests distinguish null, absent, zero, and expired values.
**Owner:** data platform owner.
**Boundary:** a silent zero can fabricate low velocity or a free purchase.

### Diagnostic 16 — Currency conversion leakage or scale error
**Symptom:** foreign-currency traffic dominates risk actions.
**Competing causes:** decimal-scale defect, end-of-day rate leakage, or a real attack concentration.
**Discriminator:** reconcile original minor units, currency reference version, as-of rate time, and converted value.
**Bounded response:** retain original amount, make conversion missing explicitly, and route malformed values safely.
**Recovery:** correct the feature definition and rerun point-in-time reconstruction before threshold review.
**Verification:** boundary fixtures cover currencies, rate changes, zero, and malformed scale.
**Owner:** data owner.
**Boundary:** converted authorization amount is still not issuer-borne loss.

### Diagnostic 17 — Hot key makes features stale
**Symptom:** fleet throughput looks healthy while one token or merchant partition lags.
**Competing causes:** concentrated fraud, normal merchant popularity, or poor partitioning.
**Discriminator:** compare per-partition arrival/service rates and key concentration with freshness flags.
**Bounded response:** redesign state partitioning without double-counting cross-shard history.
**Recovery:** reserve recovery capacity for live requests and pace replay under the stale-feature policy.
**Verification:** hot-key simulation checks count conservation before, during, and after recovery.
**Owner:** platform owner.
**Boundary:** adding buffer cannot drain a queue when arrival remains at service capacity.

### Diagnostic 18 — Unsafe experimentation
**Symptom:** proposed exploration would vary mandatory controls or expose high-risk requests without a stop rule.
**Competing causes:** causal ambition outruns governance, or eligibility has not been bounded.
**Discriminator:** inspect preregistration for exclusions, assignment unit, owner, guardrails, stop/rollback rules, and mature estimand.
**Bounded response:** restrict randomization to approved eligible policies; retain mandatory protections and rules fallback.
**Recovery:** pause assignment on a trigger and analyze intent-to-treat including fallback paths.
**Verification:** dry-run assignment proves stable account assignment across retries and complete denominator logging.
**Owner:** fraud product owner and model risk owner.
**Boundary:** shadow mode validates mechanics, not action-specific effectiveness.

### Diagnostic 19 — Observational effect is overstated
**Symptom:** post-launch lower fraud is attributed to challenge or decline from before/after charts.
**Competing causes:** seasonality, attacker adaptation, mix shift, label delay, or a true policy effect.
**Discriminator:** compare predeclared randomized mature cohorts; if unavailable, expose confounders and no causal claim.
**Bounded response:** phrase observational output as association and use it to prioritize a safe experiment.
**Recovery:** retain raw cohorts, assignment, policy versions, and maturity snapshots for reanalysis.
**Verification:** review checks that every effect claim names a comparator and assigned denominator.
**Owner:** risk analytics owner.
**Boundary:** no amount of approved-only adjustment identifies declined counterfactual fraud by itself.

### Diagnostic 20 — Fallback masks policy availability
**Symptom:** overall action metrics look stable while model-path availability falls.
**Competing causes:** successful fallback, dashboard aggregation, or missing fallback telemetry.
**Discriminator:** split every metric by model-path versus fallback reason, channel, and policy version.
**Bounded response:** set a separate fallback-rate alert and review whether fallback action mix honors friction budgets.
**Recovery:** restore healthy dependencies before re-enabling the primary policy; do not quietly reclassify fallback as scored.
**Verification:** failure drill reconciles gateway requests, response actions, durable decisions, and analytics rows.
**Owner:** platform owner with fraud product owner.
**Boundary:** availability of a response is not availability of the intended risk policy.

## 13. Transferable Lesson
A real-time risk score is an input to a constrained action policy, not the policy itself. Transfer the decision contract: name the unit, owner, baseline, allowed actions, deadline, hard friction/capacity boundaries, evidence state, and outcome denominator before choosing a model.

Transfer point-in-time discipline as well. A feature must be available at the decision, identity joins must preserve namespace and uncertainty, and later investigations must not rewrite the facts used for an authorization. Use mature cohorts to separate “not yet known” from negative evidence. Keep approved-only selection visible rather than solving it by inventing declined labels.

Transfer action accounting, but not its values. Purchase amount, realized issuer loss, operating cost, abandoned challenge friction, and false-decline harm have different owners and denominators. An action table is useful for exposing assumptions; it is not evidence of action effectiveness until a safeguarded comparison with mature outcomes tests those assumptions.

Transfer the fallback as a first-class policy. A response returned on time may still be a different risk policy if it used stale features or rules-only routing. Measure fallback separately, preserve idempotent decisions, and test recovery under load.

Do not transfer the constructed `$120` loss, `$6` friction value, 90-day maturity, 100-ms budget, challenge capability, 5% challenge budget, or 500-abandonment guardrail as facts. A rail without challenge needs a different action set; a delayed manual workflow needs a separate capacity and causal design. This case does not verify that any trained ranker or intervention improves production outcomes.

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
#### Q04 — Action-cost arithmetic
Candidate prompt: Recompute the challenge row in the constructed ledger.
Clarify: the inputs are scenario assumptions, not observed labels or a measured policy result.
Strong answer: Residual issuer loss is `4,000*.04*(1-.80*.70)*$120=$8,448`; operating cost is `$1,800`; modeled abandonment friction is `4,000*.96*.12*$4=$1,843.20`; total is `$12,091.20`.
Reasoning: the challenge action has completion, conditional-effect, operating, and friction terms, unlike a purchase amount shortcut.
Acceptable alternative: An equivalent expansion of `AC-02` is correct if units and the legitimate fraction are explicit.
Follow-up and answer: What does this prove? Only arithmetic under constructed assumptions, not prevention effectiveness.
Red flag: Calling 160 scenario fraud attempts an observed challenge label count.
Score 0: incorrect arithmetic or causal claim.
Score 1: correct total without assumptions.
Score 2: correct components, units, and constructed-evidence boundary.
Score 3: also distinguishes issuer loss from purchase amount and action effect.
Answer support: Section 6b AC-02.

#### Q05 — Declined-attempt label boundary
Candidate prompt: Why does the decline row omit a prevented-fraud benefit?
Clarify: a decline prevents the modeled authorization path, but its approval counterfactual is not observed.
Strong answer: The row counts immediate modeled friction and operating cost only. It must not assign fraud/nonfraud or a dollar prevented loss to an individual decline from its score.
Reasoning: approved-only data are selected by the historic action, and declines lack the outcome under approval.
Acceptable alternative: Estimate an average policy effect in a safeguarded randomized comparison with mature outcomes.
Follow-up and answer: Can an investigator confirmation solve this? No; it is selected downstream evidence, not every declined counterfactual.
Red flag: Filling declined labels with model probability and treating them as truth.
Score 0: invents decline labels or claims realized savings.
Score 1: notes missing labels but gives no decision consequence.
Score 2: states the counterfactual boundary and the assigned-policy alternative.
Score 3: additionally names approved-only selection and mature denominators.
Answer support: Sections 5a, 6b, 6d, and 6e.

#### Q06 — Friction/capacity sensitivity
Candidate prompt: Does the base challenge policy meet the constructed friction budget?
Clarify: use 4,000 challenges, a 5% rate cap, 12% abandonment, and 500-abandonment limit per 100,000 eligible attempts.
Strong answer: The 4% challenge rate is under the 5% cap. Expected abandonment is `4,000*.12=480`, so it is under the limit by 20; at 16%, it becomes 640 and breaches the limit.
Reasoning: rate and abandonment are separate constraints, and a near-boundary plan deserves no automatic expansion.
Acceptable alternative: also compute support contacts: 160 at 4%, requiring `160/120=1.33` intake-days under the stated simplification.
Follow-up and answer: What should happen at 640? Hold or reduce the policy until the owner changes the guardrail with evidence.
Red flag: Treating an abandonment limit as proof that abandonments are fraud.
Score 0: conflates percentages with counts.
Score 1: computes one constraint only.
Score 2: computes both and identifies the breach.
Score 3: explains the capacity implication and evidence limitation.
Answer support: Section 6c.
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
