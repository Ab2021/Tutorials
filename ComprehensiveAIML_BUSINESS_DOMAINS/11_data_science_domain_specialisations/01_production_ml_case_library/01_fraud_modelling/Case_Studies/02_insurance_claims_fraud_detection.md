---
case_id: CASE-01-01-002
case_title: Claims investigation triage under delayed adjudication
case_version: "1.1"
created: 2026-11-09
last_reviewed: null
status: proposed
content_status: draft
case_type: predictive-ml
domain: Property and casualty insurance claims
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, human-in-the-loop]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-10
owner_role: Claims special-investigations-unit product owner
source_index: INDEX.md
review_trigger: Material change to claim intake, referral policy, adjudication definition, investigation capacity, or regulated-data approval
---
# Case study: Claims investigation triage under delayed adjudication

## 1. The One-Line Framing

Given one newly reported eligible claim and only information that exists at intake, rank it for a
special-investigations-unit (SIU) referral for the claims operations owner, so that confirmed
recoverable value per investigator hour rises without breaching claimant-service, due-process,
privacy, or capacity limits, compared with the existing rules-plus-adjuster referral queue.

Everything in this case is **ILLUSTRATIVE**. No insurer, claim population, trained model,
investigation, or measured saving is described. The numbers are constructed teaching inputs for
arithmetic and decision discipline; they are not estimates of fraud prevalence, recovery rates, or
programme impact.

The deliverable of the decision is a prioritized referral recommendation. It is not an accusation,
a coverage determination, a denial, or an automated adverse action. The naive framing — classify a
claim as fraudulent or legitimate — fails on three counts. Truth arrives months after the decision.
Investigation effort is not independent of the decision, because referred claims are examined more
closely. And an investigator can open only a fixed number of files per day, so the real object being
allocated is scarce human attention.

The worked arithmetic in this case produces 30% precision at a 200-referral weekly capacity and 60%
recall against a constructed mature-positive count, a mechanical value ledger of $65,000 per
constructed cohort, and a break-even yield of 27.5 confirmed positives. Each figure is derived and
independently re-checked, and each is conditional on the constructed inputs.

**Takeaway:** the case selects one bounded, reviewable queue decision and refuses to convert an
uncertain risk score into a fraud verdict.

## 2. Context & Background

A first notice of loss arrives through a channel: a web form, a call centre, a broker upload, or a
mobile application. An intake service validates mandatory fields and creates a claim record with a
tokenized claim identifier. An adjuster later gathers documents, photographs, repair estimates, and
recorded statements, then assesses coverage. A subset of claims is referred to the SIU. An SIU
investigator reviews the file, may request further evidence, may close the referral without action,
or may escalate through the organisation's approved process. The claim continues through its ordinary
handling path in parallel or afterwards.

Four accountability boundaries matter and are frequently blurred in practice.

First, the decision owner is the claims operations leader, who is accountable for the referral policy
and for claimant outcomes. Second, the SIU supervisor owns capacity allocation and evidence quality;
an investigator cannot be silently conscripted into a queue designed by a model team. Third, the
adjuster retains the approved manual referral path and must never be blocked by a low model score.
Fourth, legal, compliance, privacy, and model-risk roles approve permitted data, notices, retention,
and escalation rules; the model team supplies a score and an explanation surface but does not decide
coverage or investigation outcomes.

The baseline is a versioned set of intake rules plus voluntary adjuster referral. Rules might encode
known operational signals such as a prior claim within a window, an unverified provider, or a
mismatch between reported loss date and incident date. Its actual referral rate, lead time,
investigation yield, and recovery results are unknown in this case; they must be measured before any
replacement is contemplated. The baseline persists for three defensible reasons: it is
understandable to front-line staff, it can encode mandatory control logic that a learned model cannot
be trusted to reproduce, and it provides a documented fallback when scoring is unavailable.

The scope is one line of business and one eligible population: first-notice claims that pass
eligibility screening and are not already in an investigation. Excluded are reopened claims unless
policy defines them as new decisions, claims already referred, claims outside the line of business,
and records with no lawful processing basis. Later artefacts — investigation notes, settlement
amounts, litigation outcomes, subrogation recoveries, and adjuster narratives — are unavailable at
the initial decision unless a historical version provably existed before the cutoff.

A high score triggers a reviewed referral. A low score must not prevent an adjuster from escalating a
documented concern. Referral itself has costs: it delays ordinary handling, it can impose document
requests on a claimant, and it consumes an investigator hour that could have been spent on a
different file. Those costs are decision metrics, not optional reporting.

**Takeaway:** the workflow, not the model, establishes the data cutoff and the human owner of every
consequential action.

## 3. The Business Question

| Decision-contract field | Illustrative contract |
|---|---|
| Question | Which eligible new claims enter tomorrow's SIU queue when only 40 investigator starts per day are verified? |
| Decision owner | Claims operations leader; SIU supervisor approves queue policy within that capacity. |
| Trigger and deadline | Claim-intake completion; ranking produced within 15 minutes so ordinary handling is not blocked. |
| Available actions | Keep rules-only queue; refer top-ranked claims; request standard additional intake information; defer with reason when a critical intake field is missing. |
| Value and downside | Mature net recoverable value per investigator hour versus avoidable delay, claimant burden, and investigator opportunity cost. |
| Objective | Mature net recoverable value per 1,000 eligible claims and per investigator hour, at fixed capacity. |
| Hard constraints | No automated denial; 40 verified daily starts; approved data only; structured reason code on every decision; applicable notice and review controls. |
| Scope | Eligible first-notice claims in one proposed pilot cohort, one line of business. |
| Out of scope | Coverage determination, pricing, criminal guilt, provider-network investigations, claims already referred, and any decision about a person's insurability. |

Mature net recoverable value is defined as confirmed recovery plus avoided payment that the approved
outcome policy attributes to the investigation, minus incremental investigation cost, minus the
pre-agreed representation of claimant-service cost. Its denominator is every eligible claim assigned
to the policy, not merely the referred subset. It is not the model's probability output, and it is
not the reported claim amount.

Three denominators are easy to confuse and each answers a different question:

- Referral-rate denominator: eligible claims assigned to the policy. Used for capacity and disparity.
- Precision denominator: claims actually referred within capacity. Used for queue yield.
- Value denominator: eligible claims in the cohort. Used for programme economics.

A policy can look excellent on the second denominator while losing money on the third, because the
referral set is a selected subset whose base rate differs from the eligible population.

**Takeaway:** a queue objective must name its denominator and its harms; rank precision alone cannot
choose a policy.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback / owner |
|---|---|---|---|---|---|---|
| A-01 | Intake timestamp and claim identifier are stable and unique per decision. | data | replay duplicate and ordering checks on 30 days of intake | unverified | point-in-time features and idempotency fail | retain rules queue / platform owner |
| A-02 | Referral disposition has an auditable maturity state with a timestamp. | label | plot resolution delay by cohort month | unverified | labels are selected or censored | report maturity strata / analytics owner |
| A-03 | SIU can start 40 files per day across the pilot queue. | operations | time study plus schedule reconciliation | illustrative | top-K policy is infeasible and backlog grows | lower K or explicit backlog policy / SIU supervisor |
| A-04 | Every candidate feature has a documented permitted-use basis. | privacy | field-level approval register with retention class | unverified | feature use is disallowed and must be removed | remove feature / privacy owner |
| A-05 | Adjuster manual referral remains available at all times. | workflow | acceptance test including score outage | proposed | model suppresses documented human concern | manual escalation / claims leader |
| A-06 | Investigation cost per file is stable within a complexity band. | economic | sampled time-and-motion by claim type | illustrative | value ledger misranks claim types | cost bands per type / finance partner |
| C-01 | Ranking never autonomously changes coverage or payment. | governance | policy and user-interface test | required | adverse-action and due-process risk | disable ranked referral / product owner |
| C-02 | Every referral and non-referral carries a structured reason code. | auditability | sampled file review plus log completeness check | proposed | outcomes cannot be interpreted later | rules-only routing / SIU supervisor |
| C-03 | No protected or special-category data enters the score without explicit approval. | legal | data inventory review | required | unlawful processing | drop field / legal owner |

The two assumptions most likely to reverse the recommendation are label maturity (A-02) and real
capacity (A-03). Neither can be repaired by a better offline AUC. A third risk, A-06, changes the
ranking of claim types even when the model is unchanged: if complex commercial claims cost five
investigator hours rather than one, a per-file capacity policy systematically misallocates time.

A useful discipline is to write the reversal condition for each assumption before running anything.
For A-02 the reversal is: fewer than 70% of referrals reach a final disposition inside the evaluation
window. For A-03 the reversal is: measured median starts per day below 30 with no compensating
backlog policy. For A-06 the reversal is: cost per file varies by more than 2x across the top
score deciles.

**Takeaway:** list a testable escape route for each assumption that could invalidate the queue, and
state the observation that would trigger it.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is one claim at one intake decision, keyed by a tokenized claim identifier plus
an intake-version identifier. A reopened claim is a new decision only if policy says so, and it must
retain a link to the prior record so that duplicated exposure is visible rather than hidden.

Candidate inputs at intake, all subject to approval and availability verification:

| Field group | Example fields | Timing check | Risk if misused |
|---|---|---|---|
| Timing | intake event time, declared incident date, report delay in hours | event clock from intake service | report delay can be a proxy for claimant sophistication rather than fraud |
| Loss description | loss type, cause category, reported amount, currency | intake form version | free-text cause coding drifts between form versions |
| Policy context | tenure band, coverage type, prior claim count as of intake | as-of join on effective dates | current policy dimension overwrites historical ownership |
| Channel | intake channel identifier, document count at submission | intake event | channel is a proxy for population and for missingness |
| Party and provider | repair-provider identifier when supplied, missingness flag | only if present at intake | provider aggregate risk is not proof of wrongdoing |
| Geography | loss location at the level approved | intake form | geography can proxy for protected characteristics |

Each field records event time, source-ingestion time, availability time, source version, and
permitted-use status. The decision cutoff is the earliest moment at which the queue is generated,
typically a nightly or intraday batch boundary. Any adjuster note, final payment, SIU narrative,
litigation flag, or adjudication event recorded after that cutoff is prohibited from the feature set
unless a historical version provably existed at the cutoff.

The outcome ledger retains referral status, referral reason, investigation start and end timestamps,
disposition code, resolution time, established recovery value where applicable, avoided-payment
estimate where the approved policy defines one, outcome confidence, and appeal or reversal status.
"No confirmed issue" is not a synonym for "legitimate claim". A referral might never be
investigated, might remain unresolved, or might lack documentary support. Each of those states is
recorded separately.

Population exclusions must be enumerated with counts and reasons by cohort: incomplete records,
claims already under investigation, out-of-scope lines of business, duplicates, test records, and
records lacking a lawful processing basis. Splitting is temporal, never random across rows, because a
random split places later process changes into the apparent past. An untouched forward cohort is
retained after policy selection so that the chosen policy is tested once on data no one has tuned on.

### 5b. Tools & Stack

The proposed stack is: a versioned intake snapshot store, an approved feature builder with an
availability contract, a calibrated tabular ranker, a deterministic policy service, a durable
decision log, an SIU work-queue service, and a monitoring and replay store. Product names and
versions are deliberately unspecified because no implementation was examined for this case.

The durable decision log stores claim key, intake-version key, cutoff timestamp, feature-contract
version, model artifact hash, score, policy version, action, structured reason codes, capacity state,
and fallback reason. It is an audit record, not evidence about outcomes. A separate outcome table
records later dispositions with their own timestamps so that back-filling cannot launder future
information into decision-time features.

### 5c. Method Summary

Input → schema and availability validation → point-in-time feature construction → calibrated risk
ranking → top-K policy with mandatory rules guardrails and capacity reservation → referral
recommendation with audit record → human investigation and documented disposition → outcome ledger
update.

| Candidate | Evidence considered | Strength | Limitation | Status |
|---|---|---|---|---|
| Existing rules plus adjuster referral | operational baseline to measure | transparent, immediate, encodes mandatory controls | may miss combinations; prioritisation inconsistent across adjusters | comparator, not measured |
| Random low-score audit sample | identifies outcomes outside the queue | reduces selective-label bias | consumes scarce investigator capacity | proposed audit arm |
| Calibrated tabular ranker with top-K policy | can prioritise within capacity | tunable, auditable, explanation surface available | needs mature labels, calibration, and capacity honesty | proposed |
| Unsupervised anomaly score | may surface novel patterns | no confirmed-label requirement for fitting | anomaly is not fraud; unstable across seasons | considered, not tested |
| Investigation-cost-aware ranking | accounts for per-file time | aligns capacity with value per hour | needs reliable cost bands by type | considered, proposed as v2 |

The policy score estimates a defined historical outcome conditional on the data-generation process.
It does not estimate the causal benefit of referring a claim, because referral changes the intensity
of scrutiny. Threshold selection therefore uses review capacity, mature outcomes, claimant-service
guardrails, and a controlled evaluation design, never a default 0.5 cutoff.

Cost-aware ranking deserves separate treatment. If claim types have different investigator cost,
ranking by expected value alone over-allocates time to cheap, low-yield files. Ranking by expected
value per investigator hour is the correct objective under a hard hours budget, but it requires
cost bands that are themselves measured rather than assumed.

### 5d. Provenance, quality, exclusions and leakage controls

For every feature, record the owner, extraction-query revision, raw snapshot identifier, availability
rule, missingness definition, transformation code version, join cardinality, and retention basis.
Joins use the claim key plus a version or event cutoff. A current dimension table must never overwrite
what was known at intake; that pattern silently imports the future and is the most common cause of
implausible offline gains in insurance triage work.

Audit the following before any model run: duplicate claim identifiers, provider-key collisions,
late-arriving corrections, timezone conversion between intake channels, negative or absurd
durations, and changes in missingness by channel and month. Distinguish "missing because the field
was never collected for this channel" from "missing because an upstream feed failed"; the first is a
population property, the second is an operational incident that must route to fallback rather than to
learned imputation.

Fit imputers, encoders, and calibrators only on the training period inside each temporal fold. When a
field's availability time is unverified, exclude the field rather than back-filling it, even when its
offline contribution looks strong. Strength under leakage is not evidence of value.

### 5e. Reproducibility and method-selection ledger

| Artifact ID | Location | Version / date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | scikit-learn common pitfalls documentation | accessed 2026-11-10 | training-only preprocessing and pipeline discipline | public | not evidence about this case's performance |
| SRC-02 | internal approved data dictionary | to obtain | feature availability and permitted use | restricted | unavailable in this illustrative case |
| SRC-03 | claim disposition taxonomy | to obtain | outcome definition and maturity states | restricted | not reviewed here |
| RUN-01 | proposed temporal replay configuration | not executed | reproduce ranking and policy evaluation | unavailable | no run, data, or result exists |
| POL-01 | proposed referral policy specification | draft | top-K, guardrails, capacity reservation | illustrative | not approved or deployed |

A reproducible run requires snapshot hashes, code revision, package versions, random seed,
feature-contract version, policy configuration, score artifact hash, and a retained decision cohort.
A seed does not make changing claims data reproducible; only the frozen snapshot does.

**Takeaway:** reproduce the decision-time dataset and the policy, not merely the model binary.

## 6. Analysis Walkthrough

The decisive current finding is negative: no evidence supports replacing the baseline, because no
data, run, or outcome measurement has been supplied. What follows is a checked illustrative
computation showing exactly what a future evaluation must report and how its arithmetic must
reconcile.

### Step 1 — Fix the cohort and the capacity

Assume a closed evaluation cohort of 10,000 eligible claims. Assume 100 of them eventually reach the
defined confirmed outcome under the approved taxonomy after the maturity window. Assume the policy
refers the top 200 claims by score, and 60 of those referrals later reach the confirmed outcome.
These are constructed counts.

Validity conditions before use: every claim belongs to the same closed cohort; the outcome definition
and maturity rule are fixed in advance; "confirmed" is recorded consistently across investigators;
and no claim is counted twice through re-referral.

```text
FORMULA [C-01]: precision@K = TP_K / K
WHERE: TP_K = mature positive outcomes among the K referrals [claims]; K = referral capacity [claims].
CONSTRAINT: the referral set and maturity rule are fixed; precision is undefined when K = 0.
NUMERICAL EXAMPLE: TP_200 = 60 claims; K = 200 claims; precision@200 = 60/200 = 0.30 = 30%.
INTERPRETATION: 30 of every 100 referred claims mature positive in this constructed cohort; it does
                not measure causal savings and does not transfer to another insurer.
```

```text
FORMULA [C-02]: recall@K = TP_K / P
WHERE: P = all mature positive outcomes in the eligible cohort [claims].
CONSTRAINT: P uses the same cohort and the same maturity definition; unresolved outcomes are neither
            positives nor assumed negatives.
NUMERICAL EXAMPLE: TP_200 = 60; P = 100; recall@200 = 60/100 = 0.60 = 60%.
INTERPRETATION: 40 constructed positives remain outside the 200-referral queue and receive no
                discretionary investigation under this policy.
```

Independent reconciliation. The referral row contains 60 positives and 140 non-positives, totalling
200. The non-referral row contains 40 positives and 9,760 non-positives, totalling 9,800. The positive
column totals 100 and the non-positive column totals 9,900. The grand total is 10,000. Every
denominator above is one of those four counts, so the two percentages cannot disagree with the
matrix.

### Step 2 — Convert capacity into workload hours

Weekly start capacity is a flow, not a stock, and investigator hours are the real budget.

```text
FORMULA [C-03]: W = s * d
WHERE: W = weekly starts [claims/week]; s = verified staffed starts [claims/day];
       d = operating days [days/week].
CONSTRAINT: constant capacity, no absence, no rework, no carryover; otherwise use queue simulation.
NUMERICAL EXAMPLE: s = 40 claims/day; d = 5 days/week; W = 40 * 5 = 200 claims/week.
INTERPRETATION: K should not exceed verified usable starts, not analyst headcount.
```

```text
FORMULA [C-04]: H = K * h_f
WHERE: H = investigation hours per cohort [hours]; h_f = mean hours per file [hours/claim].
CONSTRAINT: mean is meaningful only within a complexity band; a mixture of 0.5 h and 5 h files needs
            a weighted calculation, not a single mean.
NUMERICAL EXAMPLE: h_f = 75 min = 1.25 h; K = 200; H = 200 * 1.25 = 250 hours.
INTERPRETATION: the 200-claim week consumes 250 investigator hours.
```

```text
FORMULA [C-05]: F = H / (h_day * d)
WHERE: F = full-time-equivalent investigators required [FTE]; h_day = productive hours per
       investigator per day [h/day].
CONSTRAINT: productive hours exclude training, leave, quality review, and administrative overhead.
NUMERICAL EXAMPLE: h_day = 7.5 h/day; F = 250 / (7.5 * 5) = 250 / 37.5 = 6.67 FTE.
INTERPRETATION: a 40-start-per-day queue needs about 6.7 productive investigator FTE per week.
```

This exposes an internal consistency check that many proposals miss: three investigators working
7.5 productive hours per day supply 112.5 hours per week, less than half of the 250 hours implied by
40 starts per day. Either the start capacity is wrong or the staffing model is wrong; the policy must
state which. This is exactly the kind of cross-check that prevents a mathematically correct ranking
from producing an operationally impossible queue.

### Step 3 — Build the value ledger from defensible components

The value ledger uses three components, each named with its unit and its source assumption.

```text
FORMULA [C-06]: V = TP_K * v - K * (c_i + c_s)
WHERE: V = illustrative net value [currency per cohort]; v = attributable value per mature positive
       [currency/claim]; c_i = investigation cost [currency/referred claim]; c_s = claimant-service
       cost [currency/referred claim].
CONSTRAINT: values are attributable, comparable across actions, and measured on the same cohort; none
            of that is established before a controlled study.
NUMERICAL EXAMPLE: 60 * 2000 - 200 * (250 + 25) = 120000 - 55000 = $65,000 per constructed cohort.
INTERPRETATION: this is a transparent scenario, not evidence that deployment saves $65,000.
```

```text
FORMULA [C-07]: V_1000 = V * (1000 / N)
WHERE: V_1000 = net value per 1,000 eligible claims [currency]; N = eligible cohort size [claims].
CONSTRAINT: linear scaling is valid only while the hit rate and cost structure stay stable.
NUMERICAL EXAMPLE: V = 65000; N = 10000; V_1000 = 65000 * 0.1 = $6,500 per 1,000 claims.
INTERPRETATION: scaling to a million claims assumes the same yield; that assumption is unverified.
```

```text
FORMULA [C-08]: V_hour = V / H
WHERE: V_hour = net value per investigator hour [currency/hour].
CONSTRAINT: H must be the hours actually consumed by the cohort, including abandoned reviews.
NUMERICAL EXAMPLE: V_hour = 65000 / 250 = $260/hour.
INTERPRETATION: the queue's economic value is per hour of investigator attention, which is the truly
                scarce input.
```

```text
FORMULA [C-09]: TP_be = K * (c_i + c_s) / v
WHERE: TP_be = break-even number of mature positives in the queue [claims].
CONSTRAINT: linear value per positive; constant cost per referral.
NUMERICAL EXAMPLE: TP_be = 200 * 275 / 2000 = 55,000 / 2000 = 27.5 claims.
INTERPRETATION: the queue needs at least 28 confirmed positives to beat its own cost; 60 is a
                comfortable margin, but 20 is not.
```

### Step 4 — Check capacity sensitivity rather than asserting one K

The choice of K is a policy decision, so evaluate it explicitly.

```text
FORMULA [C-10]: dV/dK ≈ (ΔTP * v - ΔK * (c_i + c_s)) / ΔK
WHERE: dV/dK = marginal net value per additional review slot [currency/claim].
CONSTRAINT: valid as a discrete approximation over the reported interval, not as a derivative claim.
NUMERICAL EXAMPLE: from K=200 (TP=60) to K=300 (TP=75): (15*2000 - 100*275)/100 = 2500/100 = $25.
INTERPRETATION: the 201st to 300th reviews each add about $25 net, so expansion is nearly break-even
                and extremely sensitive to value per positive.
```

With K = 100 and TP = 45, the ledger is 45*2000 - 100*275 = 90,000 - 27,500 = $62,500, slightly below
the K = 200 result. With K = 300 and TP = 75, the ledger is 75*2000 - 300*275 = 150,000 - 82,500 =
$67,500. The three points show a flat optimum, which is the realistic and useful finding: at this
yield structure the policy is not highly sensitive to queue size between 100 and 300, but it is very
sensitive to whether the marginal referrals keep the same hit rate.

### Step 5 — Quantify label maturity instead of assuming it

Outcome timing determines whether any of the above can be measured.

| Days after referral | Referrals with final disposition | Share resolved | Consequence if treated as final |
|---:|---:|---:|---|
| 30 | 110 of 200 | 55% | 90 unresolved referrals would be mislabelled negative |
| 60 | 156 of 200 | 78% | 44 unresolved referrals inflate apparent precision failure |
| 90 | 180 of 200 | 90% | 20 unresolved referrals still require an explicit state |
| 120 | 188 of 200 | 94% | revision risk from appeals remains |
| 180 | 192 of 200 | 96% | residual 4% should be reported, not coerced |

Read the 30-day row carefully. If precision were computed at 30 days with unresolved records coded
negative, apparent precision would be measured against a referral set whose positive outcomes have
not yet been recorded, and the resulting number would be biased downward by construction. If instead
unresolved records were dropped, precision would be computed on a subsample that is not the queue.
Both shortcuts are common and both are wrong; the fix is to report precision with an explicit
unresolved count and a maturity horizon.

### Step 6 — Check queue behaviour with Little's Law

Referral queues are flows, and a policy that ignores residency time produces stale priorities.

```text
FORMULA [C-11]: L = λ * W_q
WHERE: L = mean claims in the referral system [claims]; λ = referral arrival rate [claims/day];
       W_q = mean time in system [days].
CONSTRAINT: steady state, stable rates, no priority pre-emption; use simulation when arrivals burst.
NUMERICAL EXAMPLE: λ = 40 claims/day; W_q = 2.5 days; L = 40 * 2.5 = 100 claims.
INTERPRETATION: at any moment roughly 100 referrals are open; a 250-hour week must cover that
                standing work plus new arrivals.
```

If W_q grows to 5 days because completions lag starts, L becomes 200 claims and the queue carries a
full extra week of work. A ranking policy that re-scores daily without re-ranking the backlog will
leave low-priority older files stranded; the policy must state whether backlog items are re-scored,
expired, or retain position, and that choice changes the intervention itself.

### Step 7 — Compare against the baseline on the same cohort

The rules baseline must be replayed with the same eligibility, the same K, and the same maturity
window. If the baseline's measured precision at 200 were 24%, the candidate's advantage would be six
percentage points, or twelve additional confirmed positives per 200 referrals, worth 12*2000 - 0 =
$24,000 of gross value at unchanged cost. That number is illustrative framing, not a result.

Two comparison traps recur. First, comparing a candidate at K=200 against a baseline at its natural
referral rate of, say, 90 claims conflates policy strength with capacity. Second, comparing on
referred-only outcomes ignores that the baseline's referrals were selected by a different process and
therefore have different observation intensity.

### Step 8 — State what was not measured

No baseline replay was run, no model was trained, no capacity study exists, no maturity curve was
observed, and no recovery value was audited. The arithmetic above is a worksheet, not a result. The
honest statement is: "under these constructed inputs, the policy shape is feasible and the value
ledger is positive above 28 confirmed positives per 200 referrals; all inputs are unverified."

**Takeaway:** verify counts, capacity, hours, and counterfactual value separately; an attractive
illustrative result does not establish a rollout case.

## 7. The Finding

Do not replace the existing rules-and-adjuster referral process on the evidence available. A
calibrated top-K ranker may be evaluated only as a referral aid with fixed capacity, point-in-time
features, mature outcome definitions, and claimant-service guardrails.

The constructed evaluation yields 30% precision at 200 referrals, 60% recall, $65,000 illustrative
net value per 10,000 eligible claims, $260 net value per investigator hour, and a break-even yield of
27.5 confirmed positives. Those figures describe the arithmetic of the proposed policy under stated
inputs. They are not observed, not causal, and not portable to an insurer.

The substantive analytical finding is structural rather than numerical: the value of this decision
depends far more on label maturity, capacity honesty, and cost-per-file variation than on the
discrimination of the ranker. A one-point AUC improvement is worth less than correcting a 90-day
maturity horizon that is currently treated as 30 days.

The recommendation is therefore to collect an approved temporal replay, a measured capacity study,
a mature outcome extract, and a cost-per-file band decomposition before deciding whether a ranker is
worth building at all.

**Takeaway:** the defensible answer is an evaluation gate with named evidence, not an invented
deployment outcome.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

```text
[Immutable intake snapshot] --> [availability + permission validation]
                                        | fail
                                        v
                               [baseline routing + reason code]
        |
        v
[Point-in-time feature builder] --> [feature-contract checks: freshness, nulls, ranges]
        |                                                   | fail
        |                                                   v
        |                                        [fallback + alert]
        v
[Calibrated score artifact] --> [policy: mandatory rules + top-K + capacity reservation]
        |
        v
[durable decision log] --> [SIU queue assignment]
        |
        v
[replay store: snapshot hash + score + policy version]
```

The offline path materialises a versioned feature snapshot per cohort, fits preprocessing inside each
training fold, calibrates on a disjoint partition, and writes score artefacts with hashes. It also
replays the baseline policy on the identical cohort so that comparison is like-for-like. Replay
outputs are stored with the configuration that produced them, not as free-floating metrics.

### 8b. Online / Serving / Real-Time Path

The online path accepts an immutable intake snapshot and must return a recommendation before the
handling deadline of 15 minutes. Determinism is required: the same snapshot and the same policy
version must yield the same action and the same reason codes. Feature lookups are bounded by the
cutoff, and any lookup that cannot complete within budget degrades to the approved baseline routing
rather than waiting indefinitely.

Idempotency is keyed on claim identifier plus intake-version identifier. A retry with the same key
and the same snapshot hash returns the stored decision; it must not create a second queue entry. A
genuine amendment produces a new intake version and therefore a superseding decision with an explicit
link to the prior one. Both behaviours need tests, because deduplicating the first case incorrectly
hides a real amendment, and not deduplicating the second creates duplicate investigator work.

### 8c. Key Design Decisions

| Decision | Options considered | Chosen for this design | Why | Reversal condition |
|---|---|---|---|---|
| Ranking grain | claim, claimant, policy, provider | claim at intake | matches the action and the investigator's unit of work | action moves to a party-level review |
| Score use | autonomous decline, referral priority, advisory only | referral priority with human decision | preserves due process and legal position | regulator or policy change |
| Capacity handling | soft target, hard K, queue simulation | hard K with reserved mandatory slots | prevents invisible overflow | measured starts vary more than 30% week to week |
| Fallback | fail-open, fail-closed, baseline routing | baseline routing with logged reason | keeps processing while preserving controls | baseline itself unavailable |
| Explanation | score bucket only, reason codes, full attribution | structured reason codes plus factor summary | auditable and bounded | model-risk requires attribution detail |
| Backlog ageing | re-score daily, expire, retain position | retain position with explicit age alert | avoids thrash and preserves fairness to older files | age alert breaches service limit |

### 8d. Architecture artifacts and operational readiness

Readiness requires five artefacts before any pilot: a data-flow diagram with cutoff annotations; a
feature-contract document listing availability and permission per field; a policy specification with
the reason-code taxonomy; a queue-operations runbook covering overflow, outage, and staff absence;
and an evaluation plan naming cohorts, maturity, comparators, and guardrails.

Recovery behaviour must be specified per failure class. If the feature store is stale, route to
baseline. If the score service is unavailable, route to baseline and record a fallback code. If the
queue service rejects an assignment, retry with backoff and then surface the item in an overflow
list; never drop it silently. If the decision log write fails, the recommendation is withheld, since
an unlogged referral cannot be audited.

Reconciliation is daily: decision-log rows must equal queue entries plus explicit capacity-rejected
records. Any mismatch is investigated before the next replay. This is a data-integrity control, not a
model-quality control.

**Takeaway:** state ownership, idempotency, fallback, and reconciliation explicitly; a score service
is not an operable referral system.

## 9. Evaluation & Validation

| Metric | Numerator / denominator | Population / window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|
| precision@K | mature positives in K / K | eligible claims, fixed cutoff, 90-day maturity | rules queue at same K | unresolved share | proposed |
| recall@K | mature positives in K / all mature positives | same cohort and maturity | baseline coverage | unknown-label rate | proposed |
| net value | attributable recovery minus costs / eligible claims | assigned cohort, mature window | existing policy | claimant service and appeals | not measured |
| value per hour | net value / investigation hours | same cohort | baseline hours | rework hours | not measured |
| referral start time | referrals started by deadline / referrals | same period | baseline distribution | staffing variance | proposed |
| disparity review | action and outcome rates by approved slice | sufficient-count cohort | policy comparator | minimum-count privacy rule | proposed |
| fallback rate | fallback decisions / all decisions | daily | operational tolerance | outage correlation | proposed |

Validation design: a chronological development window, a later validation window for policy
selection, and a final forward holdout evaluated once after the policy is frozen. Both candidate and
baseline are backtested on the same eligibility and capacity. Because historical referrals were
selected by the old policy, outcomes are selectively observed; permitted randomized low-score audit
referrals are included where governance allows, and unresolved outcomes are retained as a state
rather than coerced to negative.

For a prospective evaluation, randomize eligible units between the existing policy and the proposed
policy only after governance approval. Analyse intent-to-treat by assignment, retain all fallback
traffic, pre-register the primary metric and the stopping rule, and monitor service and safety
guardrails with the same seriousness as the primary metric. Shadow scoring tests technical
compatibility but reveals nothing about outcomes under changed referral behaviour.

Sensitivity tests that must be reported: maturity window (30/60/90/180 days), exclusion of uncertain
dispositions, conservative versus central value attribution, K set to measured rather than planned
staffing, replay with feature delays injected, and stratification by intake channel. A conclusion
that reverses under a plausible maturity or capacity definition is not a conclusion.

**Takeaway:** validate the whole decision policy on mature comparable cohorts; validating a score in
isolation produces numbers that do not survive contact with the queue.

## 10. Action Taken

No production action was taken. The action in this case is a proposed evidence plan with five steps:
inventory permitted intake data and confirm availability timestamps; define outcome-taxonomy states
including unresolved; measure baseline referral behaviour and verified start capacity by claim type;
build a point-in-time replay with baseline and candidate policy; and complete privacy and model-risk
review before considering a bounded prospective evaluation.

A real action log must record approver identity and role, policy version, eligibility definition,
assignment mechanics, fallback handling, claimant-service measurement, and the stop decision. It must
distinguish three different negative states: not approved; approved but not technically ready; and
evaluated but not beneficial. Collapsing these into "paused" destroys the information needed for the
next review.

**Takeaway:** a proposal is not a pilot, and a pilot is not a production result.

## 11. Outcome & Results

Outcome is **not measured**. There is no deployed ranker, no eligible-population count, no
investigator-workload observation, no rules baseline, no calibration result, and no recovery estimate.
The $65,000 ledger and the $260 per-hour figure are conditional arithmetic on constructed inputs.

A future result statement must name the policy, cohort, maturity window, comparator, numerator,
denominator, uncertainty interval, guardrails, and whether each number was observed, estimated, or
simulated. It must never report referral count as proven fraud prevented, and it must never report
recovered value as programme benefit without subtracting the counterfactual recovery the baseline
would have achieved anyway.

**Takeaway:** keep the evidence boundary beside the conclusion, not only in the metadata block.

## 12. Failure Modes & What Went Wrong

### F-01 — Future-information leakage through current-value joins

**Symptom:** offline precision rises sharply after adding adjuster-note, payment, or
document-completeness fields.

**Competing causes:** a genuinely useful intake signal; the field was recorded after the referral; the
join used a current dimension table that overwrote the historical value.

**Discriminating check:** compare each field's availability timestamp with the queue cutoff and
replay from an immutable snapshot. A leaked field is frequently unavailable for many historical
decisions, so its coverage rate falls when the cutoff is enforced.

**Bounded fix:** remove or lag the field; rebuild features from cutoff snapshots; refit preprocessing
within temporal folds. **Risk:** apparent rank quality drops. **Verification:** sampled decision
records show every retained feature existed before the decision, with the timestamp recorded.

### F-02 — Selective and immature labels

**Symptom:** referred claims look far more positive than non-referred claims, while recent cohorts
have few recorded outcomes.

**Competing causes:** effective targeting; investigation intensity differences; unresolved claims;
different follow-up windows per channel.

**Discriminating check:** plot disposition and maturity by referral status and cohort age, then audit
a permitted random sample outside the queue. **Bounded fix:** retain unresolved labels, wait for
maturity, use audit sampling, and report observation propensity beside every outcome rate. **Risk:**
slower learning and consumption of scarce capacity. **Verification:** the maturity curve and audit
outcomes appear in the evaluation ledger.

### F-03 — Capacity overflow and stale priority

**Symptom:** recommended referrals wait days while new high-priority claims enter behind an old
backlog; queue age grows while daily referrals stay constant.

**Competing causes:** capacity reduction; arrival surge; duplicate queue messages; longer
investigation cycles; unmeasured rework.

**Discriminating check:** reconcile accepted recommendations, queue positions, starts, completions,
and duplicate keys by day. Backlog growth should equal arrivals minus starts after cancellations.
**Bounded fix:** set K from measured starts, define backlog ageing policy, deduplicate, expose
overflow. **Risk:** lower recall and delayed valuable referrals. **Verification:** replay shows no
silent drops and queue age stays within the approved limit.

### F-04 — Proxy harm or opaque adverse handling

**Symptom:** one approved review slice receives materially more referrals or delay without a
documented explanation.

**Competing causes:** channel mix; missing intake data; genuine exposure differences; a feature acting
as a proxy; staffing geography.

**Discriminating check:** compare eligibility, missingness, score distribution, action rate, mature
outcomes, and uncertainty by permitted slice with minimum-count rules. **Bounded fix:** pause the
affected segment, constrain or remove the feature, improve intake data quality, and obtain governance
review. **Risk:** overall queue yield declines. **Verification:** repeat the pre-specified slice
analysis on a later mature cohort.

### F-05 — Duplicate or non-idempotent referral

**Symptom:** the same claim appears twice in SIU, or decision-log and queue counts disagree.

**Competing causes:** client retry; intake amendment; lost queue acknowledgement; identifier
collision.

**Discriminating check:** match the immutable decision key, snapshot hash, policy version, and queue
receipt timestamp. **Bounded fix:** durable upsert with explicit supersession protocol. **Risk:**
over-deduplication hides a legitimate amended claim. **Verification:** injected retry and amendment
traces produce exactly one active queue record with a complete audit chain.

### F-06 — Cost-blind ranking starving expensive high-yield files

**Symptom:** value per investigator hour falls even though precision at K improves.

**Competing causes:** cost per file varies by claim type; precision ignores hours; complex claims are
ranked below cheap ones with similar hit rates.

**Discriminating check:** produce a precision-per-hour table by claim complexity band; compare
expected value per hour rather than per file. **Bounded fix:** rank on expected value per hour and
reserve capacity per band. **Risk:** simple claim types lose slots, which could raise claimant delay
for fast-paying claims. **Verification:** replay reports both per-file and per-hour yield.

### F-07 — Reason-code drift making the audit trail unreadable

**Symptom:** the distribution of reason codes changes without a policy release, or fallback codes
appear under normal operation.

**Competing causes:** uncoordinated code deployment; taxonomy change without versioning; upstream
field rename silently mapped to a default.

**Discriminating check:** reconcile code taxonomy version against deployment records and sample
records per code. **Bounded fix:** version the taxonomy, reject unknown codes at write time, and alert
on unexpected fallback growth. **Risk:** strict rejection can block processing during rollout; stage
the change with dual-write. **Verification:** every sampled record has a code present in the approved
taxonomy with a policy version.

### F-08 — Investigation effect mistaken for model skill

**Symptom:** referred claims show high confirmed-fraud rates and the result is presented as evidence
that the model detects fraud.

**Competing causes:** referral increases scrutiny and therefore findings; investigators focus on
high-scoring files; documentation standards differ by queue.

**Discriminating check:** compare audit outcomes for low-score sampled claims against referrals, and
check whether disposition definitions are applied uniformly. **Bounded fix:** randomize audit
referrals, report observation-conditioned rates explicitly, and never claim causal detection from
selected data. **Risk:** audit capacity is expensive and reduces discretionary yield. **Verification:**
the evaluation report separates observed disposition rates from estimated detection effects.

**Takeaway:** every failure diagnosis needs a competing explanation, a signal that separates it, a
limited repair, and a verification step.

## 13. Transferable Lesson

Risk-ranking systems in investigative settings are allocation policies, not truth machines. The
transferable sequence is: define the decision-time unit and the action; construct only data available
then; state who is observed and who is not; constrain the policy to measured operational capacity;
evaluate outcome and harm on the assigned population; preserve a human, auditable path for
exceptions; and keep a safe existing process as fallback.

The same design transfers to payment review queues, content moderation, clinical coding review,
document quality assurance, and maintenance triage. It does not transfer unchanged where referrals
themselves impose materially different harms, where labels carry different legal meanings, or where
human review cannot make the action safe within the required time.

Two quantitative habits transfer as well. Convert any "number of reviews" target into hours and FTE
before believing it. And separate value per file from value per hour whenever the cost of review
varies across the population.

**Takeaway:** changing the intervention, the label process, or the capacity changes the problem even
when the model family is identical.

## 14. Connections & Cross-References

Related methods: the fraud evaluation, calibration, and capacity playbook provides reusable
rank-metric and capacity reasoning; the local payment-card case contrasts synchronous authorization
with asynchronous investigation; the claims-reserving material provides adjacent actuarial context
that is not fraud-method evidence.

Contrasts worth teaching explicitly:

| Case | Decision clock | Label truth | Main constraint | Transferable difference |
|---|---|---|---|---|
| Payment card fraud (CASE-01-01-001) | milliseconds | delayed chargebacks | friction and latency | action must be automatic and reversible |
| Claims triage (this case) | hours | delayed adjudication | investigator capacity | action is a queue position, not a decline |
| Account takeover (CASE-01-01-003) | seconds | delayed confirmation | support load | action includes challenge and recovery |
| AML monitoring (CASE-01-01-004) | days | very delayed SAR outcomes | regulatory reporting | obligation exists regardless of yield |

### 14a. Interview pack

#### Q01 — Why not optimise accuracy?
**Answer:** prevalence is low and capacity is capped, so an all-negative classifier can be highly
accurate and useless. The decision requires precision at a fixed K, recall against mature positives,
value per investigator hour, and claimant-service guardrails.
**Reasoning:** accuracy weights the 9,800 non-referrals heavily and makes the referral decision
almost irrelevant to the score.
**Follow-up:** if K doubles to 400, replay both policy and staffing; do not assume value doubles.
**Acceptable alternative:** cost-weighted utility with defensible cost inputs.
**Red flag:** calling 99% accuracy deployment-ready.
**Scoring:** 0 wrong; 1 names imbalance; 2 connects metric to capacity; 3 adds label-selection limits.
**Support:** Sections 3 and 6.

#### Q02 — Compute precision and recall from the constructed counts.
**Answer:** 60/200 = 30% precision at 200; 60/100 = 60% recall at 200.
**Reasoning:** the first denominator is referrals; the second is all mature positives in the eligible
cohort.
**Follow-up:** false positives are 140 and false negatives 40; check that 60+140=200 and 60+40=100.
**Acceptable alternative:** report counts before percentages.
**Red flag:** dividing 60 by 10,000 for either metric.
**Scoring:** 0 incorrect; 1 one metric; 2 both with denominators; 3 states the maturity assumption.
**Support:** C-01, C-02.

#### Q03 — What is wrong with training on final payment amount?
**Answer:** it is generally unavailable at intake and leaks both future handling and the outcome of
the process being influenced.
**Reasoning:** a score using it would predict the investigator's conclusion rather than rank the
claim at decision time.
**Follow-up:** retain only a versioned estimate that provably existed at the cutoff and is approved.
**Acceptable alternative:** lagged historical amounts aggregated strictly before the cutoff.
**Red flag:** "the warehouse has it now."
**Scoring:** 0 denies the issue; 1 says leakage; 2 explains timing; 3 designs a replay test.
**Support:** Section 5d and F-01.

#### Q04 — How do selected labels bias the result?
**Answer:** referred claims receive more investigation, so their outcomes are more observable than
non-referred claims; a model can learn the prior referral policy rather than underlying risk.
**Reasoning:** the label is a function of both the claim and the observation process.
**Follow-up:** use permitted random audits, maturity strata, and report observation propensity.
**Acceptable alternative:** carefully bounded propensity analysis with explicit limits.
**Red flag:** treating non-referred claims as legitimate.
**Scoring:** 0 unsafe; 1 names bias; 2 gives mechanism and mitigation; 3 names residual uncertainty.
**Support:** Sections 5a, 9, and F-02.

#### Q05 — Show the weekly capacity calculation and its implication.
**Answer:** 40 starts/day * 5 days = 200 starts/week; at 1.25 hours per file that is 250 hours,
requiring 6.67 FTE at 7.5 productive hours per day.
**Reasoning:** starts are a flow, hours are the resource, and headcount is neither until availability
and service time are measured.
**Follow-up:** if only 3 investigators are staffed, either reduce K to about 90-112 files per week or
add staff; do not silently accumulate backlog.
**Acceptable alternative:** queue simulation using arrival and service distributions.
**Red flag:** multiplying investigator headcount by an assumed throughput.
**Scoring:** 0 wrong arithmetic; 1 arithmetic only; 2 converts to hours; 3 reconciles staffing.
**Support:** C-03, C-04, C-05.

#### Q06 — Is $65,000 a business case result?
**Answer:** no. It is a conditional ledger on constructed inputs where attributable value, referral
effect, and baseline counterfactual are all unobserved.
**Reasoning:** recovery value requires a causal attribution method, and the baseline would have
recovered some of the same amount.
**Follow-up:** at $1,000 per positive the ledger is $5,000; at 20 positives it is -$15,000.
**Acceptable alternative:** defer the value statement until controlled evidence exists.
**Red flag:** presenting the ledger as measured savings.
**Scoring:** 0 claims proof; 1 adds a caveat; 2 lists missing evidence; 3 recomputes sensitivity.
**Support:** C-06 to C-09.

#### Q07 — Why retain the rules baseline?
**Answer:** it is the operational comparator, the fallback, and the carrier of mandatory controls that
a learned model cannot be assumed to reproduce.
**Reasoning:** without replaying it on the same cohort and capacity, no comparison is meaningful.
**Follow-up:** version the rule set and run both policies on an identical eligibility list.
**Acceptable alternative:** an approved manual queue as baseline if rules are not stable.
**Red flag:** assuming the model supersedes regulatory or policy controls.
**Scoring:** 0 unsafe; 1 calls it baseline; 2 explains like-for-like comparison; 3 includes fallback.
**Support:** Sections 2, 5c, 8.

#### Q08 — A referral queue is delayed by three days; diagnose before acting.
**Answer:** distinguish arrival surge, reduced starts, longer service time, duplicate messages, and
stale priority using arrival/start/completion reconciliation and duplicate-key checks.
**Reasoning:** queue age alone has at least five causes with different remedies; increasing K fixes
none of them.
**Follow-up:** expose overflow and stop silent drops before any threshold change.
**Acceptable alternative:** queue simulation after data validation.
**Red flag:** raising K without adding capacity.
**Scoring:** 0 single-cause assertion; 1 names backlog; 2 gives discriminating signals; 3 bounded
remediation with verification.
**Support:** F-03 and C-11.

#### Q09 — What does calibration add, and what does it not add?
**Answer:** it maps score ranges to observed outcome rates in a named cohort, which supports
probability language and cost reasoning; it does not establish that referral causes recovery.
**Reasoning:** ranking quality and probability calibration are different properties, and neither is
causal.
**Follow-up:** assess calibration by time period and channel after maturity, with uncertainty.
**Acceptable alternative:** use a pure rank policy and avoid probability claims.
**Red flag:** "a 0.8 score means 80% fraud everywhere."
**Scoring:** 0 false certainty; 1 definition; 2 boundary; 3 validation plan.
**Support:** Sections 5c, 9.

#### Q10 — A new intake channel has missing provider identifiers. Deploy unchanged?
**Answer:** no. Missingness may be a population property or a feed failure; either way the score
distribution and the fallback rate change. Route that channel to approved baseline or a
missingness-aware policy only after temporal and channel validation.
**Reasoning:** the model was fit on a missingness pattern that no longer holds.
**Follow-up:** compare missingness and mature outcomes on a pilot cohort before enabling.
**Acceptable alternative:** pause channel expansion until validation completes.
**Red flag:** silently imputing a modal provider.
**Scoring:** 0 blind deployment; 1 notes drift; 2 gives a safe path; 3 names verification.
**Support:** Sections 5d and F-04.

#### Q11 — How would you make referral creation idempotent?
**Answer:** store the decision keyed by immutable claim-decision key plus snapshot hash and return it
on retry; treat a changed intake version as an explicit superseding decision.
**Reasoning:** a network retry is not a new claim, but an amendment is not a retry.
**Follow-up:** test identical retry and valid amendment traces and reconcile the active queue count.
**Acceptable alternative:** transactional outbox with equivalent durable semantics.
**Red flag:** claiming broker exactly-once delivery proves end-to-end uniqueness.
**Scoring:** 0 unsafe; 1 mentions dedupe; 2 distinguishes amendment; 3 specifies an oracle.
**Support:** Section 8 and F-05.

#### Q12 — What changed constraint would stop the pilot?
**Answer:** a policy change prohibiting a needed field, measured capacity below the audit minimum, a
guardrail breach in an approved slice, or an unresolved-outcome rate that invalidates the primary
metric.
**Reasoning:** each changes the data contract, the feasibility, or the estimand, so the previous
evaluation no longer applies.
**Follow-up:** continue existing routing, preserve decision logs, and scope revalidation precisely.
**Acceptable alternative:** restrict the pilot to a validated segment with its own metric.
**Red flag:** retuning the threshold to hide a guardrail breach.
**Scoring:** 0 ignores the constraint; 1 pauses; 2 explains fallback; 3 defines revalidation scope.
**Support:** Sections 4, 9, 10.

#### Q13 — What denominator belongs in a referral-rate guardrail?
**Answer:** eligible claims assigned to the policy, stratified by approved channel or cohort where
relevant, with exclusions reported beside the rate.
**Reasoning:** referred-only denominators hide differential eligibility and exclusion, which is
exactly the mechanism by which a queue can appear fair while access is unequal.
**Follow-up:** report exclusion counts and reasons alongside every rate.
**Acceptable alternative:** an intent-to-treat denominator in a randomized pilot.
**Red flag:** dividing by reviewed claims when discussing claimant exposure.
**Scoring:** 0 wrong denominator; 1 names eligible claims; 2 explains selection; 3 adds exclusion audit.
**Support:** Sections 3 and 9.

#### Q14 — Design the test for an amended claim record.
**Answer:** send an identical retry and then a documented later intake version with a changed snapshot
hash; expect one durable action for the retry and one auditable supersession for the amendment.
**Reasoning:** the invariant is exactly one active queue record per claim with a complete chain.
**Follow-up:** reconcile active queue count and decision-log rows after both events.
**Acceptable alternative:** transactional outbox preserving the same invariant.
**Red flag:** overwriting the original record and losing its action.
**Scoring:** 0 unsafe; 1 says dedupe; 2 distinguishes amendment; 3 gives an oracle.
**Support:** Section 8 and F-05.

#### Q15 — Why is a high provider-risk association insufficient for referral policy?
**Answer:** association may reflect provider mix, reporting practices, entity-resolution error, coding
conventions, or a genuine signal; a referral decision needs timing, permitted use, and evidence that
the action helps.
**Reasoning:** an aggregate statistic is not an individual finding, and provider identifiers are
frequently shared or stale.
**Follow-up:** compare provider-key quality, join cardinality, and missingness before any feature use.
**Acceptable alternative:** exclude unverified provider features and use claim-level evidence.
**Red flag:** inferring wrongdoing from an aggregate rate.
**Scoring:** 0 accusation; 1 correlation caveat; 2 competing causes; 3 validation plan.
**Support:** Sections 5a, 5d, and F-04.

#### Q16 — The model is well ranked but the programme loses money. What do you check first?
**Answer:** cost per file, hours consumed, and the value per positive — the three components of the
ledger — in that order, because ranking quality cannot rescue negative unit economics.
**Reasoning:** with a break-even yield of 27.5 positives per 200 referrals, a well-ranked queue whose
marginal files cost 3 hours and yield $300 each is still value-destroying.
**Follow-up:** recompute value per investigator hour by complexity band.
**Acceptable alternative:** switch the objective to expected value per hour.
**Red flag:** blaming the model when the policy economics are wrong.
**Scoring:** 0 retrains immediately; 1 checks cost; 2 checks cost and hours; 3 ties to break-even.
**Support:** Sections 6 and F-06.

### 14b. Decision-defense drills

#### Drill D-01 — Fix the policy unit before discussing features
The policy unit is an eligible claim-intake record at a fixed cutoff, not a claimant, a policy, a
provider, or an eventual claim file. The claim key identifies the decision record; the intake-version
key separates a valid amendment from a retry; the cutoff is the earliest time the policy can act; the
action is referral priority, not coverage. The owner is the claims operations leader, the reviewer is
an SIU investigator, and the outcome is a defined mature review result. The consequence is the
allocation of scarce investigation time, and the counterfactual is the baseline action on the same
claim. Changing any element changes the estimand.

#### Drill D-02 — Trace one record without future information
At 09:00 the intake service emits claim C-17, snapshot version 1. The policy records receipt time
before feature construction. Reported amount is present with its currency and source version.
Policy tenure comes from a dimension row whose availability time precedes 09:00. Prior-claim count
is computed over claims available before 09:00. The provider identifier is absent, so the missingness
indicator is true; absence is not mapped to a risky-provider category. No adjuster note exists yet,
and no payment, settlement, disposition, or recovery field is permitted. Schema validation passes
because required intake fields are present. Feature-contract and score-artifact versions are recorded,
along with the capacity state showing 40 starts available. The frozen rule decides selection. The
decision log stores key, snapshot hash, score bucket, action, reason codes, and policy version. A
retry with the same hash returns the stored action; version 2 of intake creates a superseding
decision. The investigator sees only the permitted view, and later dispositions are appended as
events rather than back-filled into features.

#### Drill D-03 — Reconcile the confusion matrix
Cohort: 10,000 eligible claims; mature positives: 100; capacity: 200; top-K true positives: 60.
Referral row: 60 true positives plus 140 without the defined outcome equals 200. Non-referral row:
40 positives plus 9,760 without equals 9,800. Positive column: 60 + 40 = 100. Non-positive column:
140 + 9,760 = 9,900. Grand total 10,000. Precision 60/200 = 30%; recall 60/100 = 60%; referral
fraction 200/10,000 = 2%. The 140 referrals are not proven harmful and the 40 missed claims are not
proven mishandled; both labels are shorthand for the stipulated historical outcome, and selective
investigation can invalidate the non-referral column entirely.

#### Drill D-04 — Defend the capacity boundary
Capacity is stated in starts because a referral opens before it resolves; completion capacity can
differ within a week. Headcount is not capacity without availability and case-time data. Training,
quality review, leave, escalations, and rework reduce usable starts. Urgent and rules-mandated
referrals consume capacity before discretionary ranking, so the policy must reserve mandatory slots
explicitly and expose overflow rather than hiding it. Queue age is a guardrail because stale
referrals lose practical value. The policy owner must choose whether unreviewed referrals expire, are
re-ranked, or retain position, and log that choice. If arrivals exceed starts continuously, buffer
growth does not solve the imbalance; if service time doubles, K is unsafe even when the score is
accurate.

#### Drill D-05 — Test a temporal split
Development uses an earlier closed period; validation uses a later period the developer cannot use to
fit transformations; the forward test stays untouched until model and policy are frozen. All claims
for a linked entity need an explicit boundary rule. An entity appearing in two periods is acceptable
only if its features use prior information exclusively. Random splits conceal time drift in intake
process and can place later process labels into the apparent past. Imputation, encoding, and
calibration are fit only on the development partition. Threshold selection uses validation capacity
and guardrails only, and the forward test is evaluated once. Every policy revision creates a new
evaluation version, and missingness, maturity, and eligibility rules are compared across periods.
Temporal evaluation alone cannot identify referral causality.

#### Drill D-06 — Distinguish four different values
The reported claim amount is an intake statement in currency. Estimated covered loss is a later
handling estimate that may not exist at intake. Established recovery is an outcome occurring after
investigation. Avoided payment is a counterfactual estimate requiring a defensible method.
Investigation cost is a resource cost of staff time and vendor work; claimant-service cost is a
pre-agreed representation of delay or burden, not a moral valuation. The score may use none of these
unless availability and permission are established. The ledger must not double count recovery and
avoided payment, must identify incremental versus sunk costs, must name currency and window, must use
the same cohort for both policies, and must disclose unresolved outcomes.

#### Drill D-07 — Write a claim ledger
Claim 1: a top-K policy can be replayed from a versioned intake snapshot — proposed mechanism; does
not establish predictive performance. Claim 2: precision@200 is 30% under the constructed counts —
derived and hand-checked; does not establish a production rate. Claim 3: 200 reviews at 75 minutes
consume 250 hours and about 6.7 FTE — derived illustrative arithmetic; does not establish staffing
need. Claim 4: a current-value join can leak future information — method principle supported by
preprocessing guidance; does not establish that leakage occurred at an insurer. Claim 5: randomized
audit review reduces selective-label uncertainty — proposed design; does not establish feasibility.

#### Drill D-08 — Specify reason codes
REFERRAL_RANK: selected by the discretionary rank policy. REFERRAL_RULE: routed by a mandatory
approved rule. REFERRAL_MANUAL: authorised adjuster or supervisor override. NO_REFERRAL_CAPACITY:
eligible but outside K. NO_REFERRAL_INELIGIBLE: outside the declared population. FALLBACK_STALE:
feature-freshness condition failed. FALLBACK_POLICY_UNAVAILABLE: scoring unavailable, baseline used.
SUPERSEDED_INTAKE: later intake version replaced the prior decision. DUPLICATE_RETRY: prior durable
decision returned. Each code carries a policy version and timestamp, is never an accusation, and
feeds monitoring. Override codes require an authorised actor and bounded free text. The taxonomy is
itself under change control.

#### Drill D-09 — Evaluate calibration honestly
Calibration compares predicted risk with observed outcomes in a specified cohort under a defined
outcome and maturity rule, after the feature and policy population are fixed. A reliability plot is
meaningless if unresolved records are coded negative. Calibration can differ by channel and period.
A well-calibrated model can still have poor top-K value, and a well-ranked model can be badly
calibrated. Recalibration changes score meaning and must be versioned; isotonic or logistic
calibrators must not be fit on the forward test. Small bins need intervals or pooling, sparse slices
need minimum-count privacy rules, and the claim should read "calibrated in this cohort" rather than
"calibrated universally".

#### Drill D-10 — Define rollout gates
Gate 1: permitted fields and availability lineage approved. Gate 2: baseline eligibility and capacity
measured. Gate 3: replay reproduces baseline decisions within tolerance. Gate 4: retry and amendment
traces satisfy the decision invariant. Gate 5: outcome taxonomy and maturity curve reviewed. Gate 6:
candidate evaluation uses a locked temporal split. Gate 7: service and slice guardrails specified.
Gate 8: audit-sampling feasibility decided. Gate 9: fallback routing and monitoring ownership tested.
Gate 10: an authorised owner chooses pilot, pause, or rejection. No gate passes without its artifact,
and passing every technical gate still does not pass governance.

### 14c. Open questions this case does not resolve

Which outcomes are permitted and auditable under the applicable regime; whether randomized low-score
audit referral is feasible and lawful; how much of an established recovery is attributable to the
investigation rather than to ordinary handling; whether investigator cost bands are stable enough for
per-hour ranking; and which jurisdiction-specific notices and appeal rights attach to a referral.
Each is a scoping question for a real engagement, and none is answered by the illustrative arithmetic.

## Appendix A — Evidence gate register

Each gate names the artefact that must exist, the property it protects, the signal that reveals a
breach, the bounded response, and the verification oracle. Status is proposed for all gates because
no implementation was examined.

| Gate | Required artefact | Protected invariant | Breach signal | Bounded response | Verification oracle |
|---|---|---|---|---|---|
| E-01 cohort freeze | hash of the eligible claim list at cutoff | eligibility cannot change after assignment | late additions appear in a closed cohort | exclude and log additions, re-freeze | list hash matches the replay input |
| E-02 cutoff timing | availability timestamp per field | every retained feature precedes the queue time | a field's median availability exceeds cutoff | drop field or lag to a prior snapshot | zero features with availability after cutoff |
| E-03 identity linkage | claim-to-policy cardinality report | one-to-many links keep both keys and dates | current policy owner appears on a historic claim | rejoin on effective dates | historical ownership reproduces from versioned dimension |
| E-04 missingness semantics | reason code for every null | source-absent differs from feed failure | fallback rate spikes with null rate | route feed failures to baseline | nulls resolved to explicit reason codes |
| E-05 outcome taxonomy | disposition, recovery, reversal, unresolved states | no disposition is silently negative | precision changes when maturity window shifts | retain unresolved state, report it | every referral has one terminal or open state |
| E-06 capacity measurement | starts, completions, reopenings, aged files | K uses verified starts for the window | queue age grows while K is constant | recompute K, publish overflow | daily K equals measured starts within tolerance |
| E-07 baseline replay | rules version and adjuster route | candidate and baseline see identical inputs | baseline referral counts cannot be reproduced | rebuild baseline from the rule archive | baseline decisions match historical record within tolerance |
| E-08 transformation scope | fit scope and timestamps | learned transforms fit only on training folds | category encodings contain future categories | refit inside fold | fold-level fit log shows no forward data |
| E-09 calibration | score bins, maturity, selection route | reported bins name cohort and final outcomes | reliability plot drifts after maturity update | recompute and relabel observation-conditioned | bin counts sum to cohort totals |
| E-10 reason codes | policy and operational reason per decision | every referral and non-referral has a reason | unknown or default codes appear | reject unknown codes at write time | sample audit finds no unexplained decision |
| E-11 assignment | assignment record with mechanism | assignment precedes observation | outcomes used to assign claims | rebuild assignment from policy log | assignment reproducible from frozen inputs |
| E-12 service guardrail | referral-to-contact interval distribution | claimant burden is measured, not assumed | contact interval tail grows beyond limit | pause segment, review capacity | interval distribution within approved bound |
| E-13 slice review | action and outcome rates by approved slice | disparities are examined with uncertainty | a slice's rate exceeds comparator beyond interval | constrain feature or pause slice | repeated analysis on later cohort confirms or clears |
| E-14 override analysis | counts and reasons for manual referral and suppression | human judgement remains visible | override rate changes without policy change | review workflow change, retrain only after | override reasons reconcile to workflow release notes |
| E-15 retry safety | retry and amendment traces | exactly one active queue record per claim | duplicate referrals for one claim | durable upsert and supersession | injected traces produce one active record |
| E-16 queue reconciliation | decision log versus queue entries | no silent drops or phantom referrals | row counts diverge daily | halt ranked queue, reconcile, restore baseline | counts equal including explicit capacity rejects |
| E-17 value attribution | approved attribution method and counterfactual | recovered value is not credited twice | value exceeds plausible loss reduction | restrict claim to incremental amount | ledger ties to approved method with sign-off |
| E-18 maturity curve | resolution share by days and channel | unresolved outcomes are reported | maturity differs sharply by referral status | stratify and extend window | curve reproduces from outcome ledger |
| E-19 release compatibility | feature-contract and policy version pair | deployable combination is explicit | new contract with old policy passes tests | block mismatched pairs at deploy | deployment manifest pins both versions |
| E-20 fallback test | injected outage scenarios | fallback yields a logged, valid decision | unlogged default actions appear | fail closed for logging, route to baseline | every injected outage yields a reason code |
| E-21 retention boundary | retention class per field | audit need and privacy limits both respected | features retained beyond approved period | purge and document | retention audit shows no expired field in use |
| E-22 appeal and correction | correction path and reversal tracking | outcomes can be revised without rewriting history | reversals overwrite original dispositions | append revision events | revision chain reconstructs both states |
| E-23 prospective stop rule | pre-registered metric and stopping boundary | evaluation cannot be stopped opportunistically | decision to continue changes with interim results | freeze the rule before exposure | signed protocol predates first assignment |
| E-24 decision record | frozen record per decision | the decision can be reconstructed later | record missing policy version or score hash | withhold unlogged referrals | sampled records reconstruct fully offline |

## Appendix B — Sensitivity ledger

Each scenario changes exactly one input and holds the rest at the constructed base. The interpretation
column states what decision consequence follows, which is the purpose of sensitivity analysis.

| ID | Changed input | Recomputed ledger | Interpretation |
|---|---|---|---|
| S-01 | value per positive falls from $2,000 to $1,000 | 60,000 - 55,000 = $5,000 | the programme is nearly break-even; a modest yield error flips the sign |
| S-02 | mature positives found fall from 60 to 40 | 80,000 - 55,000 = $25,000 | still positive, but value per hour falls to $100 |
| S-03 | mature positives fall to 20 | 40,000 - 55,000 = -$15,000 | queue destroys value; referral policy must not be justified on rank alone |
| S-04 | investigation cost rises to $400 per file | 120,000 - 85,000 = $35,000 | cost inflation erodes margin more than a rank decline of similar size |
| S-05 | service cost rises to $75 per referral | 120,000 - 65,000 = $55,000 | claimant-burden pricing materially changes the decision |
| S-06 | K rises to 300 with 75 positives | 150,000 - 82,500 = $67,500 | marginal reviews add about $25 each; expansion is nearly neutral |
| S-07 | K falls to 100 with 45 positives | 90,000 - 27,500 = $62,500 | shrinking the queue costs little value and frees hours |
| S-08 | 15% of referrals are overridden by investigators | yield falls roughly 15% to 51 positives | $102,000 - $55,000 = $47,000; override analysis is mandatory |
| S-09 | mean file time rises to 2.5 hours at $200 per hour | cost 200*(500+25) = $105,000; net $15,000; value per hour falls to $30 over 500 hours | time inflation almost erases the margin; measure service time before scaling |
| S-10 | maturity window shortens to 30 days | 110 of 200 resolved, positive count understated | precision cannot be computed as final; report maturity state |

Reading the ledger as a whole: the sign of the programme is driven by value per positive and hours
per file, not by K within a plausible range. That is a design conclusion about where measurement
effort belongs.

## Appendix C — Replay validation worksheet

| Check | Inputs | Expected relation | Failure meaning |
|---|---|---|---|
| Row reconciliation | referral and non-referral counts | sums equal cohort total | a claim is double counted or dropped |
| Denominator agreement | TP, K, P | 60/200 and 60/100 derive from the same matrix | inconsistent metric definitions |
| Capacity identity | starts per day, days per week | K equals product within tolerance | planned capacity exceeds measured starts |
| Hours identity | files, mean hours | H equals K times mean hours | cost per file inconsistent with time study |
| Staffing identity | hours, productive hours per day | FTE equals H divided by weekly productive hours | staffing model contradicts queue target |
| Ledger identity | value, cost components | net equals gross minus cost | a cost component is omitted or double counted |
| Break-even identity | costs, value per positive | TP_be times v equals total cost | pricing and yield assumptions disagree |
| Maturity identity | referrals, resolved share | resolved plus unresolved equals referrals | unresolved records silently dropped |
| Queue identity | arrival rate, time in system | open count equals product | queue measurement uses different clocks |
| Sensitivity identity | base and changed input | only the named component changes | hidden assumption change during sensitivity |

**Takeaway:** every numeric claim in this case is reproducible from the tables above, and every one of
them remains conditional on inputs that no one has measured here.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Decision | Rank eligible intake claims for human SIU referral; never automate coverage denial. |
| Baseline | Versioned intake rules plus adjuster referral, replayed on the same cohort and capacity. |
| Core quantities | precision@K = TP_K/K; recall@K = TP_K/P; H = K*h_f; FTE = H/(h_day*d); net V = TP*v - K*(c_i+c_s). |
| Critical condition | Features must exist at intake cutoff; outcome labels need an explicit maturity and observation state. |
| Primary failure | Selected or immature outcomes make rank metrics look better than decision value. |
| Safe fallback | Approved existing routing with logged reason code and daily queue reconciliation. |
| Evidence / stage | Illustrative, not measured; proposed design; draft and unreviewed. |
| Review trigger | Intake, outcome taxonomy, capacity, policy, or approved-data change. |
