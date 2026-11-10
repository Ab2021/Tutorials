---
case_id: "CASE-03-01-001"
case_title: "Fund a fraud ranking platform when investigator capacity, not model accuracy, binds"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Payments fraud operations and ML platform"
primary_pattern: "evaluation-mismatch"
secondary_patterns: [cost-latency-tradeoff, label-delay]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Fraud platform product owner"
source_index: "../INDEX.md"
review_trigger: "Change in fraud prevalence, review capacity, loss-per-case, liability allocation, or label maturity horizon"
---
# Fund a fraud ranking platform when investigator capacity, not model accuracy, binds

## 1. The One-Line Framing

Given each card-not-present authorization attempt and the features observable at the authorization instant, decide whether the payments organization should replace a rules-only alert queue with a ranked-scoring platform plus investigator workflow, for the fraud-operations lead, to reduce net fraud loss subject to a fixed investigator review capacity and a 120 ms authorization latency budget, compared with the current rules engine evaluated on the same eligible population.

The decision unit is one eligible authorization attempt at its authorization timestamp; the decision is about the platform and policy, not about an individual transaction.
The transaction-level decision (approve, decline, or queue for review) is executed by policy code inside the authorization path; this case bounds itself to the platform-investment decision and the capacity-neutral policy definition that decision depends on.
The task category is binary classification for ranking plus a capacity-constrained allocation; the platform decision recurs once per budget cycle, while policy execution recurs continuously at authorization rates.
The primary success condition is a capacity-neutral reduction in net fraud loss per eligible attempt, measured on mature cohorts with matched eligibility and a fixed review budget.
A feature is not a launch criterion; the platform is justified only if the policy it enables beats the rules policy under the same 4,800-review-per-day budget and the same latency envelope.
The two secondary patterns are the cost-latency tradeoff at authorization and delayed chargeback labels; both constrain what can be known in time to act.
All entities, volumes, rates, dollar amounts, latency figures, staff counts, and architecture elements are invented teaching values.
Nothing in this case was deployed, measured, procured, or reviewed; there is no vendor, employer, or benchmark claim.
The obvious simpler framing ("buy a better model for higher AUC") misses the binding constraint: the number of human reviews per day is fixed, so only the composition of the reviewed set can change.
That reframing is the decisive idea: a ranking model is a mechanism for choosing which 4,800 of roughly 18,000 daily alerts receive scarce human attention, and that choice is what has to be valued.

## 2. Context & Background

| Context field | Case-specific content (all hypothetical) |
|---|---|
| Domain and product | Card-not-present payments at a hypothetical issuer-processor; authorization scoring and a fraud-operations review console |
| Workflow before the decision | Rules engine scores each authorization in-path; a high band auto-declines; a middle band opens an alert; a low band auto-approves |
| Workflow after the decision | Analysts work a queue ordered by arrival time today, or by model score under the proposal; confirmed fraud feeds a case-management record |
| Stakeholders | Fraud-operations lead (owns queue and staffing), ML platform owner, payments risk officer, merchant relations, customer support |
| Baseline | Rules engine with 61 hand-maintained predicates, arrival-order queue, no calibrated probability output |
| Scale | 2,000,000 eligible authorization attempts per day; 7,000 fraud attempts per day at the assumed 0.35% rate |
| Timing | Rolling 24-hour windows; chargebacks mature over an assumed 120-day horizon; budget decision quarterly |
| Geography and boundaries | One region, one currency, card-not-present channel only; in-person and cross-border corridors excluded from scope |
| Prior attempts | Two rule-threshold increases in prior quarters; both raised alert volume faster than reviewed frauds, per the invented premise |
| Applicable requirements | No regulation is asserted; the case treats data-minimization and auditability as internal design choices, not legal conclusions |

The organization's stated pain is not "the model is inaccurate" but "the queue is 18,000 alerts deep and we can review 4,800."
Unreviewed middle-band alerts age out and the authorization stands, so unreviewed volume is effectively an approve decision made by the absence of capacity.
That aging-out mechanism is the central operational fact: it converts a staff shortage into an implicit policy that nobody chose.
The rules engine also produces a high band that auto-declines; its precision is assumed to be 0.60, meaning 40% of auto-declined attempts are legitimate.
Legitimate declines cost margin and customer goodwill, which the ledger must count or the comparison is biased toward aggressive blocking.
The queue is worked in arrival order, so a fraud attempt arriving at 08:00 receives more scrutiny than an identical one arriving at 22:00.
That time-of-day artifact is a policy accident, and any ranking model removes it — which is part of the value even at constant precision.
The fraud-operations lead has authority over staffing and queue policy; the ML platform owner has authority over model and feature artifacts; the risk officer can veto loss-increasing changes.
The decision is therefore jointly owned, and this case names the fraud-operations lead as accountable for the bounded decision with a risk-officer veto.
Analysts are assumed to be trained to a common rubric with sampled quality review; that assumption is marked unverified in Section 4.
Merchant relations has an interest in false-decline reduction, which is a guardrail rather than the objective.
Customer support absorbs friction from declines; no support-cost figure is asserted here, so that cost enters only as a qualitative guardrail.
The incumbent rules have no calibration, so a "0.72 score" today means a predicate count, not a probability; this matters for threshold portability.
The proposal is to add a gradient-boosted ranker with a calibrated probability head, a versioned feature pipeline, and a score-ordered queue with abstention.
The platform also adds shadow-mode replay and a per-decision audit record, which the rules engine lacks; these are operational capabilities, not accuracy claims.
The case is deliberately channel-narrow to keep one data contract; expanding to card-present or cross-border changes features and liability assumptions materially.
Prior quarters' experience is part of the invented premise; it is not a published finding about any organization.

## 3. The Business Question

Illustrative stakeholder question, paraphrased rather than quoted: "Should we fund the ranking platform for the middle band this quarter, or is a rule-threshold change enough?"

| Decision contract | Content |
|---|---|
| Question | Replace the arrival-order rules queue with a ranked, calibrated scoring platform for the card-not-present middle band, or extend rules and defer the platform? |
| Decision owner | Fraud-operations lead, with a loss guardrail veto held by the payments risk officer and artifact sign-off by the ML platform owner |
| Trigger and deadline | Quarterly budget cycle; the queue backlog is assumed to exceed capacity every day, and prior threshold increases did not close the gap |
| Available actions | (a) keep rules unchanged; (b) raise rules thresholds; (c) fund ranked scoring with a score-ordered queue; (d) fund staffing only; (e) fund a hybrid: ranker plus unchanged auto-decline rules |
| Value and downside | Value is avoided fraud loss net of review and platform cost; downside is misallocated reviews, added latency, false declines, and a platform the operations team cannot operate |
| Objective | Minimize net fraud loss per eligible attempt per day at fixed review capacity, with false-decline rate as a guardrail |
| Hard constraints | Reviewer capacity 4,800 reviews/day; p95 authorization budget 120 ms; no decline without an authorized policy band; auditability of every band decision |
| Scope | One region, one currency, card-not-present channel, 2,000,000 attempts/day, 120-day label maturity horizon |
| Out of scope | In-person channel, cross-border corridors, merchant underwriting, collections, chargeback representment strategy, staffing negotiations beyond the capacity assumption |

The objective is monetary but the binding constraint is counted in reviews, so the two must be joined by an explicit exchange rate.
That exchange rate is the review budget: each review consumes one of 4,800 daily slots regardless of how likely it is to be fraud.
A proxy conflict exists and must be named: raising recall by lowering the threshold increases alert volume, and volume beyond capacity converts into unreviewed approvals rather than into protection.
A second proxy conflict: raising precision by reviewing only the top few hundred alerts reduces cost per catch but forgoes prevented fraud, and the ledger must show which effect dominates.
The no-intervention alternative includes the current aging-out behaviour, which is a real policy even though no one approved it in writing.
An expensive false decision looks like: a platform that improves offline AUC, concentrates reviews on easy cases, and leaves costly high-value fraud untouched while adding latency to every authorization.
Another expensive false decision: declining a large legitimate segment through a miscalibrated band, which is visible to customers immediately and to the ledger only later.
The decision can be answered with an action (fund, defer, or pilot) or with a justified evidence request; it cannot be answered by "the model has better AUC."

## 4. Assumptions & Constraints

| ID | Assumption/constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | 2,000,000 eligible attempts/day with 0.35% fraud prevalence | Data | Daily eligibility reconciliation by channel and currency | Unverified | Value ledger scales and precision change | Recompute with observed counts | Volume shift >10% / ops analyst |
| A-02 | Fraud loss per caught case averages $320 | Business | Finance-provided loss attribution by band and channel | Unverified | Objective magnitude and break-even | Use band-specific loss table | Loss mix change / risk officer |
| A-03 | Loaded analyst cost is $63/hour, so 8-minute review costs $8.40 | Business | Workforce cost model with overhead | Unverified | Review-cost term and capacity tradeoff | Re-derive from payroll + overhead | Contract or staffing change / ops lead |
| A-04 | Legitimate declined ticket loses $9.60 margin | Business | Margin model for declined-and-not-retried attempts | Unverified | False-decline guardrail weighting | Add retention/retry model | Margin or retry behaviour change / finance |
| A-05 | Chargebacks resolve within 120 days for 97% of cases | Data | Maturity curve by cohort age | Unverified | Any quality claim made early | Restrict claims to mature cohorts | Maturity curve shift / data owner |
| A-06 | Ranking model achieves precision 0.44 in the top 4,800 alerts | Model | Shadow-mode evaluation on a frozen mature cohort | Unverified; the case's central number | Entire economic comparison | Fall back to rules if precision at capacity ≤ 0.44 | First mature evaluation / model owner |
| A-07 | Auto-decline band at probability ≥ 0.85 yields 1,000 alerts at precision 0.65 | Model | Band-level precision with confidence interval | Unverified | Decline volume and false-decline guardrail | Widen review band instead of declining | Calibration drift / model owner |
| A-08 | Reviewers apply a consistent rubric with sampled second review | Human | Inter-rater agreement on a sampled batch | Unverified | Precision measured through review is noisy | Add adjudication before claims | Agreement below threshold / QA lead |
| A-09 | p95 authorization budget of 120 ms end to end | Infrastructure | Load test at peak rate with real dependency latency | Unverified | In-path model feasibility | Move scoring to a bounded cache or reduce features | Latency SLO change / platform owner |
| A-10 | Feature values available before authorization, never after | Data | Point-in-time replay rejecting post-cutoff fields | Assumed true by contract | Leakage inflates offline precision | Remove offending features and re-evaluate | Any new feature / data owner |
| C-01 | Review capacity fixed at 4,800 reviews/day | Cost/staffing | Staffing roster and per-review duration | Confirmed as scenario constraint | Whole comparison changes | Capacity-neutral pilot re-run | Staffing change / ops lead |
| C-02 | No decline may occur outside an approved policy band | Safety/policy | Policy version in the decision record | Confirmed as design constraint | Unauthorized blocking of legitimate payments | Route to review instead | Policy change / risk officer |
| C-03 | Model artifact and feature versions must be pinned per decision | Reproducibility | Manifest in the audit log | Confirmed as design constraint | Unreproducible incident analysis | Rebuild manifests | Schema change / platform owner |
| A-11 | Review completes before the attempt settles, so a confirmed fraud can be blocked | Business/process | Compare review completion timestamps with settlement timestamps | Unverified; largest leverage in the ledger | Review prevents nothing and only declines prevent loss | Convert review into a settlement hold with a stated duration | Hold-policy change / ops lead |
| A-12 | The queue is worked work-conserving at a stable service rate | Operations | Compare arrivals and completions by hour | Unverified | Waiting time grows without bound during bursts | Add burst staffing or an overflow priority rule | Shift pattern change / ops lead |
| A-13 | Unreviewed attempts age out at a documented age rather than silently persisting | Operations | Inspect queue aging rule and aged-out counts | Unverified | The baseline's implicit policy differs from the stated one | Reconstruct the baseline with the actual aging rule | Queue policy change / ops lead |

The three assumptions most likely to reverse the recommendation are A-06 (precision at capacity), A-09 (latency), and A-05 (label maturity).
If A-06 is materially below 0.44, the ranker cannot match the rules engine's prevented fraud inside the same review budget, and the platform should be deferred.
If A-09 fails at peak, the ranker must leave the synchronous path and the decision latency advantage disappears, changing the architecture rather than only the model.
If A-05 is wrong in the optimistic direction, early quality claims will be made on a cohort whose most damaging cases are still unresolved.
A-03 and A-04 are financial parameters; they change magnitudes and break-even points but not the structural conclusion that capacity binds.
A-08 is the quietest assumption: if reviewers disagree about what counts as confirmed fraud, precision estimates inherit that disagreement.
C-01 is a scenario constraint rather than an empirical fact; the case therefore reports what happens under the constraint instead of assuming staffing will grow.
Verification of every model-related assumption is explicitly unexecuted; Section 9 lists the protocol and Section 14b lists the resulting open questions.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Content |
|---|---|
| Unit of analysis | One eligible authorization attempt; key `(attempt_id)`; a retry is a separate attempt with its own identifier |
| Volume and window | 2,000,000 attempts/day in scope; 24-hour evaluation windows; UTC throughout |
| Inputs | Attempt amount and currency; card and merchant identifiers (tokenized); merchant category; channel; device and session signals; velocity counts over trailing 1h/24h/7d; account tenure; prior dispute indicator as of cutoff |
| Outcome/label | Confirmed fraud: a chargeback with fraud reason code, or an analyst-confirmed fraud on a non-charged-back attempt; maturity 120 days |
| Population and exclusions | Card-not-present, in-scope region and currency, non-zero amount; duplicates collapsed by attempt_id; test and internal attempts excluded |
| Join semantics | Attempt to case record on attempt_id; one case per attempt; chargeback may arrive once per attempt; corrections versioned |
| Splits | Time-based: train 9 months, validation 2 months, test 2 months, with a 120-day label buffer between split boundaries and evaluation cutoffs |

The feature contract is temporal and must be stated per field, not per table.
Velocity counts use event time with an availability lag: an event is eligible only if its availability timestamp is no later than the authorization timestamp.
Prior dispute indicators are the riskiest field because the dispute itself may be initiated after the chargeback window opens; the contract therefore uses "dispute opened before this attempt's authorization" and rejects any later state.
Amount and currency are recorded in minor units with an explicit currency code; no cross-currency arithmetic occurs unless an exchange rate with a timestamp is joined.
Merchant identifiers are pseudonymized consistently across train and serve so that merchant-level patterns remain usable without exposing identities.
Device and session signals are optional; a missing signal must stay distinguishable from a negative signal.
The label has two sources with different reliability: chargebacks are authoritative but delayed and incomplete at the horizon; analyst confirmation is faster but reflects analyst judgement and queue exposure.
Confounding is visible immediately: attempts that were declined never produce a chargeback, so a decline-policy change alters the observed label distribution.
That mechanism is the classic "decline hides fraud" problem and it must be handled explicitly in evaluation rather than assumed away.
Excluded attempts are counted, not dropped silently: out-of-scope channels, zero amounts, duplicates, and missing required identifiers each get a reason code and a daily count.
The evaluation manifest freezes the eligible attempt identifiers, the model and feature versions, the policy bands, and the review capacity for the window.

### 5b. Tools & Stack

A generic stream processor evaluates velocity aggregates over keyed state with event-time windows and bounded lateness.
A model-serving component loads a pinned artifact and returns a calibrated probability plus a model version identifier.
A rules overlay remains in the path for hard blocks that must not depend on model availability, such as a confirmed-fraud list.
A feature store supplies point-in-time values with both event and availability timestamps; the sheet in the sibling collection defines the selection rule.
An append-only decision log records attempt_id, timestamps, feature values, model version, probability, band, and final action.
A case-management system holds review outcomes with analyst identifiers, review duration, and adjudication status.
A batch evaluation job reads the decision log and matured labels to compute precision at capacity and the economic ledger.
No vendor, framework version, hardware, or throughput benchmark is asserted; the architecture is described by role, not by product name.

### 5c. Method Summary

The pipeline is: authorization event → eligibility filter → point-in-time feature retrieval → calibrated model score → band assignment (auto-decline / review / auto-approve) → capacity-constrained queue selection → action → logged record → matured label join → evaluation.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Rules unchanged | Current policy and its queue behaviour | Stable, explainable, no new dependency | Arrival-order queue ignores risk; no calibration; backlog ages out | Baseline | Never; it is the comparator |
| Rule threshold increase | Prior quarter's alert growth | No new platform | Raises volume against a fixed capacity, so added alerts age out | Considered, rejected in premise | Capacity grows materially |
| Staffing only | Cost of analysts vs prevented loss | Directly attacks the binding constraint | Cost scales linearly; no prioritization gain | Considered, not selected alone | Loss per case rises sharply |
| Ranked model + review queue, rules auto-decline retained | Offline precision at capacity (assumed) | Reorders fixed capacity toward higher-risk attempts | Depends on A-06 and A-09; adds in-path dependency | Chosen for a staged pilot | Precision at capacity ≤ baseline |
| Ranked model with model-driven auto-decline | Calibration assumption A-07 | Larger prevented loss at same review count | Decline errors are immediate and customer-visible | Partially adopted (band only, high threshold) | Calibration drift beyond tolerance |
| Full automation without review | None available | Maximum throughput | No containment for novel fraud patterns; unauditable | Rejected | Not applicable |

Threshold selection is a policy procedure, not a model property: the band boundary is chosen on validation data under the capacity constraint, then frozen before the test window is read.
Ties at the capacity boundary are broken deterministically by attempt timestamp then attempt_id so that the selected set is reproducible.
The model family is a gradient-boosted tree ensemble over the numeric and categorical features with monotone constraints on velocity features where the direction is defensible.
Calibration is fitted on a held-out slice disjoint from the model's training rows, and calibration is re-checked on the later test window before any probability-threshold claim.
The stopping rule for training is early stopping on the validation log-loss with a fixed maximum number of boosting rounds; no hyperparameter search touches the test window.
The invariant that matters operationally is: the number of review-band attempts per day must not exceed the reviewed capacity, or the overflow rule must be explicit.

### 5d. Provenance, Quality, Exclusions & Leakage

The directory inventory names `DS_Domain/Practical_ML_Design` fraud-platform material as topic provenance; that name is a scope pointer, not examined source text.
The [source-coverage brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-practical-sources) maps fraud platform design to this home and supplies authoring context only.
The [authoring standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) define the evidence vocabulary and numerical obligations used here.
No operational data, chargeback file, model artifact, or evaluation run exists for this case.
Every number in Sections 6, 9, and 11 is a constructed teaching value; none is a measurement, estimate from a real portfolio, or vendor figure.
Data quality controls are stated as design requirements: row-count reconciliation by hour, duplicate attempt detection by attempt_id, and null-rate monitoring by feature.
Late-arriving labels trigger versioned recomputation rather than in-place mutation, so a matured cohort can be rebuilt identically.
The leakage audit is temporal and per-feature: for each field, the earliest availability time is compared with the authorization timestamp on sampled rows.
Label leakage is separately audited because the fraud flag itself can be joined too early; the join key must be the attempt, not the card, and the label state must be as-of the evaluation date.
Selection bias is structural: declined attempts have no observable chargeback, so the evaluation must distinguish "would have been fraud" from "was charged back."
The honest consequence is that no offline estimate can fully simulate a decline policy; only an assigned holdout or an explicit assumption can.

### 5e. Reproducibility & Method-Selection Ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | `DS_Domain/Practical_ML_Design` (inventory name only) | Not examined | Topic provenance for fraud platform design | Unavailable | Contents and claims unknown |
| SRC-02 | [ds-practical-sources brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-practical-sources) | 2026-11-06 | Scope mapping | Local | Planning text only |
| SRC-03 | [Case master](../../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) | 1.0 | Section structure and evidence rules | Local | Editorial standard, not data |
| SRC-04 | [Authoring standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | 2026-11-06 | Numerical conventions | Local | Editorial standard, not data |
| RUN-01 | None | Not executed | Precision-at-capacity evaluation | Absent | All model numbers are assumed |
| RUN-02 | None | Not executed | Latency load test | Absent | A-09 remains unverified |

Reproduction requires the decision log, label store, policy band definitions, and staffed review records; none are available, so this case is reproducible only as arithmetic.
The seed and library versions would be recorded in a real run; a seed alone would not guarantee cross-platform determinism for a tree ensemble with multithreaded histogram building.
The method-selection ledger records that the rules baseline was retained as comparator rather than replaced by an easier strawman such as random selection.
Random selection at capacity is a useful sanity comparator for one arithmetic check in Section 6, and it is explicitly labelled a floor rather than the business baseline.

## 6. Analysis Walkthrough

### 6.1 The decisive comparison: precision at capacity, not AUC

The decision turns on one quantity: among the 4,800 attempts an analyst can actually review per day, how many are truly fraud.
That quantity is precision at capacity, and it is denominated in the review budget rather than in a percentage of all attempts.
A model can improve a global ranking metric while leaving precision at capacity unchanged if its gains concentrate far below the cutoff.
Conversely a model can leave a global metric unchanged and still double precision at capacity by reordering only near the boundary.
Because the review budget is fixed at 4,800, the total number of potentially prevented frauds is exactly the sum of what the auto-decline band removes and what the reviewed set removes.
Everything else — AUC, calibration, feature count — matters only through those two channels.

The constructed rules baseline is defined as follows.
Of 2,000,000 eligible attempts per day, 7,000 are assumed to be true fraud at a 0.35% prevalence.
The rules auto-decline band fires on 1,200 attempts and is assumed 0.60 precise, preventing 720 fraud cases.
The rules middle band opens 18,000 alerts containing 3,300 fraud attempts, which is a band precision of 0.1833.
The queue is worked in arrival order, so the 4,800 reviewed alerts are an essentially random sample of the 18,000.
The expected reviewed fraud is therefore approximately 4,800 × 0.1833 = 880 cases.

The constructed proposed policy is defined as follows.
The calibrated model's auto-decline band at probability ≥ 0.85 is assumed to contain 1,000 attempts at 0.65 precision, preventing 650 fraud cases.
The review band is the top 4,800 non-declined attempts by score, assumed 0.44 precise, preventing 2,112 fraud cases.
The remaining attempted fraud is neither declined nor reviewed and produces loss.

| Quantity | Rules baseline | Ranked proposal | Unit |
|---|---:|---:|---|
| Eligible attempts per day | 2,000,000 | 2,000,000 | attempts/day |
| True fraud attempts per day | 7,000 | 7,000 | attempts/day |
| Auto-decline band size | 1,200 | 1,000 | attempts/day |
| Auto-decline band precision | 0.60 | 0.65 | dimensionless |
| Fraud prevented by decline | 720 | 650 | cases/day |
| Review band size | 4,800 | 4,800 | reviews/day |
| Review band precision | 0.1833 | 0.44 | dimensionless |
| Fraud prevented by review | 880 | 2,112 | cases/day |
| Total fraud prevented | 1,600 | 2,762 | cases/day |
| Fraud-case prevention rate | 0.2286 | 0.3946 | dimensionless |
| Unprevented fraud cases | 5,400 | 4,238 | cases/day |

Independent checks on the table: 720 + 880 = 1,600 and 1,600 + 5,400 = 7,000; 650 + 2,112 = 2,762 and 2,762 + 4,238 = 7,000.
Prevention rates check as 1,600 / 7,000 = 0.228571 and 2,762 / 7,000 = 0.394571.
Band precision checks as 880 / 4,800 = 0.183333 and 2,112 / 4,800 = 0.440000, both dimensionless because the denominator is the reviewed set.
Note that the rules review band contains 3,300 fraud in 18,000 alerts but only 4,800 alerts are reviewed; mixing those two denominators would produce a false precision of 0.1833 versus a reviewed precision that happens to coincide only because arrival order is assumed random with respect to risk.

### 6.2 Worked economic ledger

The ledger compares total daily cost of ownership under each policy at identical review capacity.
Let loss per unprevented fraud case be $320, loaded review cost be $8.40 per review, and legitimate-decline margin loss be $9.60 per declined legitimate attempt.
Platform cost is built up separately because it is the only term that is new under the proposal.

```text
FORMULA [FR-01]: C_total = L_u * ell + R * c_r + D_l * m + P
WHERE: C_total total daily cost [currency/day]; L_u unprevented fraud cases [cases/day];
       ell loss per case [currency/case]; R reviews performed [reviews/day];
       c_r loaded cost per review [currency/review]; D_l legitimate declines [declines/day];
       m margin lost per legitimate decline [currency/decline]; P platform cost [currency/day].
CONSTRAINT: all terms use the same day window and the same currency; review cost is counted
            under both policies at identical capacity so it cancels in the difference only after
            the equality of R is verified.
NUMERICAL EXAMPLE (baseline): 5,400*320 + 4,800*8.40 + 480*9.60 + 0
            = 1,728,000 + 40,320 + 4,608 + 0 = 1,772,928 currency/day.
INTERPRETATION: the loss term is 97.5% of baseline cost, so the ranking decision is dominated by
            prevented cases, and the review and friction terms act as tie-breakers.
```

Derivation: total cost is the sum of independent per-day components, each of which is a rate multiplied by a per-unit price.
The loss term follows because each unprevented fraud case produces one realized loss; prevented cases produce none under the model's assumption.
The review term follows because each review consumes one funded analyst slot regardless of outcome.
The friction term follows because each declined legitimate attempt forgoes the margin that attempt would have earned.
The platform term is an amortized fixed cost and is not multiplied by volume inside this expression.

Applying the same formula to the proposal with its computed platform cost:

```text
FORMULA [FR-02]: C_prop = 4,238*320 + 4,800*8.40 + 350*9.60 + 388
             = 1,356,160 + 40,320 + 3,360 + 388 = 1,400,228 currency/day.
```

The difference is 1,772,928 − 1,400,228 = 372,700 currency/day in favour of the ranked policy.
That difference decomposes into three auditable parts rather than one opaque number.

| Component of the daily difference | Amount | Sign for proposal | Derivation |
|---|---:|---|---|
| Review-channel fraud prevented | (2,112 − 880) × 320 = 394,240 | favourable | 1,232 additional cases caught inside the same 4,800 reviews |
| Decline-channel fraud foregone | (720 − 650) × 320 = −22,400 | adverse | 70 cases that the older, wider decline band caught |
| Decline friction avoided | (480 − 350) × 9.60 = +1,248 | favourable | 130 fewer legitimate declines |
| Platform cost | −388 | adverse | amortized build plus run |
| Review cost | 0 | neutral | 4,800 reviews under both policies |
| Net daily difference | 394,240 − 22,400 + 1,248 − 388 = 372,700 | favourable | sums to the total above |

The decomposition is channel-based, so the loss column is deliberately split between the review channel and the decline channel rather than reported as one number.
Summing the two loss rows gives 394,240 − 22,400 = 371,840, which independently equals (5,400 − 4,238) × 320, the change in unprevented loss.
That cross-check matters because the two routes partition the same total differently: one by policy channel, one by net unprevented cases.

The decomposition demonstrates why the review cost term must be held equal: had the proposal reviewed 6,000 attempts instead of 4,800, part of the apparent gain would be purchased with additional staff hours.
The decline-band change is a genuine trade: the model's higher-precision decline band prevents fewer total frauds but blocks fewer legitimate customers, and the ledger prices both.
A reviewer who reports only the 394,240 figure would overstate the benefit by roughly 5.5%, because the decline-band loss is real.

### 6.3 Platform cost build-up

The platform cost is small relative to the loss terms, and the case states it explicitly rather than asserting that infrastructure is negligible.
Scoring compute is 2,000,000 attempts × 3 ms of CPU time = 6,000 CPU-seconds = 1.667 CPU-hours per day.
Feature retrieval is 2,000,000 attempts × 25 ms = 50,000 CPU-seconds = 13.889 CPU-hours per day.
At an illustrative $0.05 per CPU-hour, inference and retrieval together cost 15.556 × 0.05 = $0.78 per day.
Velocity state memory is 5,000,000 active cards × 2 KB = 10,000,000 KB = 9.54 GiB of payload, doubled for a replica to 19.07 GiB.
At an illustrative $0.10 per GiB-month, that is 19.07 × 0.10 = $1.91 per month, which is $0.064 per day on a 30-day month.
Decision-log storage is 2,000,000 records × 1.5 KB = 3,000,000 KB = 2.86 GiB per day before compression.
At an assumed 4:1 compression the daily increment is 0.715 GiB; 400 days of retention gives 286 GiB, costing 286 × 0.023 = $6.58 per month, or $0.219 per day.
Summing these payload terms gives roughly $1.06 per day of infrastructure, which the case rounds to an illustrative $6 per day to include indexes, backups, and monitoring overhead.
The dominant platform cost is therefore people, not compute.

```text
FORMULA [FR-03]: P = B / A + M + I
WHERE: P platform cost [currency/day]; B one-time build cost [currency];
       A amortization horizon [days]; M ongoing maintenance [currency/day];
       I infrastructure [currency/day].
CONSTRAINT: linear amortization with no discounting; a real business case would use the
            organization's cost of capital and would separate capitalized from expensed work.
NUMERICAL EXAMPLE: B = 180 engineer-days * 900 = 162,000; A = 730 days -> 221.92;
            M = 0.25 FTE * 260 days * 900 / 365 = 160.27; I = 6.00;
            P = 221.92 + 160.27 + 6.00 = 388.19 currency/day.
INTERPRETATION: platform cost is roughly 0.03% of the fraud loss it influences, so the funding
            decision is not sensitive to infrastructure price, but it is sensitive to build scope.
```

The amortization arithmetic is 162,000 / 730 = 221.918, and the maintenance arithmetic is (0.25 × 260 × 900) / 365 = 58,500 / 365 = 160.274.
The sum 221.918 + 160.274 + 6.00 = 388.192, reported as 388 to avoid false precision.
If the build estimate were wrong by a factor of three, platform cost would rise to about 1,118 per day, which changes the break-even precision by 730/320/4,800 = 0.00048, i.e. less than one tenth of a percentage point.
That insensitivity is itself a finding: arguing about infrastructure price while leaving precision at capacity unvalidated would be a misallocation of analytical effort.

### 6.4 Break-even and the random-selection floor

The break-even precision is the review-band precision at which the proposal exactly matches the baseline cost.
Setting the proposal cost equal to 1,772,928 and solving for review precision p:

```text
FORMULA [FR-04]: (7,000 - 650 - 4,800p)*320 + 3,360 + 40,320 + 388 = 1,772,928
             => 2,076,068 - 1,536,000p = 1,772,928
             => p = 303,140 / 1,536,000 = 0.197356
CONSTRAINT: the decline band is held at its assumed 1,000 attempts and 0.65 precision;
            p must lie in [0,1] and the review set must contain enough fraud to achieve it.
INTERPRETATION: the ranker must exceed the rules review precision of 0.183333 by about
            1.40 percentage points, or 67.3 additional cases per day, to justify itself.
```

The 67.3 figure comes from 21,540 / 320 = 67.3125, and 21,540 is the platform cost plus the decline-band fraud foregone minus the friction saving.
The implied extra precision is 67.3125 / 4,800 = 0.014023, and 0.183333 + 0.014023 = 0.197356, which matches the direct solution.
This is a strong check because two independent routes — solving the full cost equation and accumulating the incremental items — produce the same number.

The floor comparison is instructive and easy to compute wrongly.
Under random review of 4,800 attempts drawn from the 1,998,800 non-declined attempts containing 6,280 fraud, expected precision is 6,280 / 1,998,800 = 0.003142.
That is 4,800 × 0.003142 = 15.08 fraud cases per day, which is a 58-fold shortfall against the rules engine's 880.
Random selection is therefore a floor for validating arithmetic, not a credible baseline, and a proposal that cannot beat random is not a proposal at all.
Using random selection as the comparator would make almost any model look transformative, which is precisely the evaluation-mismatch failure this case is about.

The ceiling is equally worth stating: the entire candidate pool contains only 6,280 fraud attempts, so even perfect ranking of 4,800 reviews cannot prevent more than 4,800 cases through the review channel.
Perfect review-channel precision would prevent 4,800 cases; the assumed 2,112 is 44% of that ceiling and 33.6% of the available fraud pool.
Combined with a perfect decline band the absolute ceiling would be 7,000 prevented, so 2,762 represents 39.5% of the maximum conceivable prevention.
No claim is made that 0.44 is near the achievable ceiling for a real system; it is a constructed assumption used to make the ledger concrete.

### 6.5 Sensitivity across the two assumptions that can reverse the decision

| Scenario | Review precision | Total fraud prevented | Daily difference vs baseline | Conclusion |
|---|---:|---:|---:|---|
| Assumed proposal | 0.44 | 2,762 | +372,700 | fund the pilot |
| Optimistic | 0.60 | 3,530 | +618,460 | fund, but verify calibration before decline changes |
| Pessimistic but above break-even | 0.25 | 1,850 | +80,860 | fund a narrow pilot |
| Near break-even | 0.20 | 1,610 | +4,060 | defer; evidence insufficient to justify organizational change |
| Below break-even | 0.18 | 1,514 | −26,660 | reject; rules remain better at equal capacity |
| Random floor | 0.003142 | 665.1 | −298,314 | rejected decisively |

Each row keeps the decline band at 1,000 attempts and 0.65 precision and the review count at 4,800, so only the review composition varies.
Total prevented equals 650 plus 4,800 times review precision: for example 650 + 4,800 × 0.25 = 650 + 1,200 = 1,850.
The arithmetic for the 0.25 row is (0.25 − 0.183333) × 4,800 = 320 cases, worth 320 × 320 = 102,400, less the 21,540 net fixed disadvantage, giving 80,860.
The arithmetic for the 0.20 row is (0.20 − 0.183333) × 4,800 = 80 cases, worth 25,600, less 21,540, giving 4,060.
The 0.18 row is (0.18 − 0.183333) × 4,800 = −16 cases, worth −5,120, less 21,540, giving −26,660.
The random row is (0.003142 − 0.183333) × 4,800 = −864.9 cases, worth −276,774, less 21,540, giving −298,314.

```text
FORMULA [FR-05]: recomputation check for p = 0.18
             C_prop = (7,000 - 650 - 864)*320 + 3,360 + 40,320 + 388
                    = 5,486*320 + 44,068 = 1,755,520 + 44,068 = 1,799,588
             difference = 1,772,928 - 1,799,588 = -26,660
CONSTRAINT: 4,800*0.18 = 864 prevented cases, exact in integer arithmetic.
INTERPRETATION: the incremental route and the full recomputation agree exactly at -26,660,
             which is the required reconciliation before any row is reported.
```

Other sensitivity dimensions that matter operationally:

| Stress | Baseline condition | Alternative | Effect on the decision | Limitation |
|---|---|---|---|---|
| Review duration | 8 min/review, 640 analyst-hours/day | 10 min/review at fixed staff | capacity falls to 3,840 reviews/day, so both policies lose fraud prevention and the model's reordering value rises | assumes fixed staffed hours |
| Volume | 2,000,000 attempts/day | 2,600,000 attempts/day (+30%) | coverage drops from 0.24% to 0.185% of attempts; absolute prevented cases rise only if the fraud pool grows proportionally | fraud rate treated as constant, which is unverified |
| Loss per case | $320 | $500 | break-even precision falls to about 0.19 → the case for the pilot strengthens | loss mix may be band dependent |
| Fraud rate | 0.35% | 0.20% | fraud pool falls to 4,000; the 0.44 review precision cannot be sustained on a smaller pool at the same review depth | assumed precision is not prevalence-invariant |
| Latency budget | 120 ms p95 | 60 ms p95 | the 25 ms feature retrieval must be cached or precomputed; model may leave the synchronous path | no measurement exists |

The volume row deserves explicit arithmetic: at 2,600,000 attempts and a constant 0.35% rate the fraud pool is 9,100 cases, and 18,000 alerts would presumably grow too, so the review band of 4,800 covers a smaller fraction of alerts.
The fraud-rate row is the most dangerous for the naive analyst: precision at capacity is not invariant to prevalence, so a portfolio with a lower fraud rate would show lower precision at the same review depth even with an identical model.
That is a base-rate effect, not model degradation, and it is why the evaluation manifest must record prevalence alongside precision.

### 6.6 Timeline the decision would follow

| Step | Action | Owner | Output | Status |
|---|---|---|---|---|
| 1 | Freeze eligible-population definition and 120-day maturity rule | Data owner | Signed manifest | Not executed |
| 2 | Reproduce rules baseline precision at capacity on a mature cohort | Risk analytics | Baseline ledger | Not executed |
| 3 | Shadow-score 4 weeks without changing any action | ML platform owner | Shadow decision log | Not executed |
| 4 | Compute precision at top 4,800 on mature shadow cohort | Evaluation owner | Interval estimate for p | Not executed |
| 5 | Compare against break-even 0.1974 with uncertainty | Fraud-operations lead | Fund/defer recommendation | Not executed |
| 6 | Load test the in-path scoring at peak with real dependencies | Platform owner | Latency distribution | Not executed |
| 7 | Run a capacity-neutral live pilot on a randomized merchant slice | Ops lead with risk veto | Live precision and friction | Not executed |
| 8 | Decide expansion after two mature weeks and a friction guardrail check | Ops lead | Expansion or revert | Not executed |

The shadow step is deliberately non-intervening: it produces scores without acting, so it cannot measure the intervention effect on fraud, but it can measure ranking and calibration on a population that is not affected by the model.
The live pilot is necessary for any claim about prevented loss, because prevention depends on action, and action changes label observability.
This ordering is the concrete form of the "evidence before action" principle, and none of it has occurred.

### 6.7 Value-weighted ranking when loss varies by case

Precision at capacity treats every caught case as worth the same amount, which is a modelling choice that must be stated because it is often wrong.
If loss per case varies with transaction amount, the correct ranking objective is expected loss prevented per review consumed, not probability of fraud.
Let `p_i` be the calibrated fraud probability, `x_i` the attempt amount, and `r` the expected recoverable fraction of a fraudulent amount, assumed 0.55 here because some fraud is recovered after chargeback.
The expected loss prevented by reviewing attempt i is then `p_i * x_i * r`, and ranking by that quantity maximizes expected prevented loss inside the review budget.

```text
FORMULA [FR-07]: e_i = p_i * x_i * r
WHERE: e_i expected preventable loss [currency]; p_i probability [dimensionless];
       x_i amount [currency]; r recovery-adjusted fraction [dimensionless, 0..1].
CONSTRAINT: loss recoverability is assumed uniform; the model's probability must be calibrated for
            the amount range in question, and amount must be known before the review decision.
NUMERICAL EXAMPLE: p=0.50, x=40, r=0.55 -> e=11.00; p=0.20, x=900, r=0.55 -> e=99.00.
INTERPRETATION: the second attempt is four times less likely to be fraud but nine times more
            valuable to review, so a probability-only ranking reviews the wrong case first.
```

That worked comparison is the case's most transferable arithmetic: probability ranking and loss ranking disagree whenever amount variance is large relative to probability variance.
In a portfolio where 80% of attempts are under $60 and 2% exceed $600, a probability-ordered queue will systematically review small-ticket fraud and leave the expensive tail to chance.
The counterargument is that high-probability small attempts are cheaper to confirm, so a review of them completes faster and the analyst returns to the queue sooner.
That counterargument is real and can be handled by ranking on expected prevented loss per expected review-minute, which requires a duration model that this case does not have.
The case therefore states its assumption plainly: the ledger in Section 6.2 uses a single average loss of $320 per case, which is valid only if the reviewed mix has stable amount composition.
If the ranker shifts composition toward larger amounts, the same precision produces a larger benefit than the ledger shows, which biases the estimate conservatively rather than optimistically.
If the ranker shifts toward smaller, higher-confidence cases — the likely outcome of probability ranking — the ledger overstates benefit, and that is the dangerous direction.
This is why Section 12's F-12 treats a band-averaged loss as a real failure mechanism rather than a simplification.

| Ranking objective | What the analyst sees first | Strength | Failure mode | Precondition |
|---|---|---|---|---|
| Probability of fraud | Most likely frauds, often small | Simple, calibrated, stable | Ignores amount; under-serves high-value tail | Loss roughly constant across amount |
| Expected prevented loss | High-value likely frauds | Aligns with the ledger | Requires amount at decision time and calibrations across amounts | Amount and probability jointly reliable |
| Expected loss per review-minute | Value per scarce analyst minute | Most faithful to the constraint | Needs a duration model with its own error | Duration predictable per case type |
| Probability times recoverability | Cases where chargeback is winnable | Captures recovery economics | Recoverability depends on issuer and evidence, rarely known at review time | Recovery data available per case |

**Takeaway:** the ledger's unit (currency) and the model's unit (probability) differ, and the ranking objective must convert between them or the optimization targets the wrong quantity.

### 6.8 Queue dynamics: why backlog depth is a policy variable

The review queue is an inventory system, and its depth determines the effective decision policy more than any threshold does.
Let arrivals of review-band attempts be `a` per hour, review completion be `s` per hour, and the queue start empty.
If `a` exceeds `s`, depth grows linearly at `a − s` per hour, and every queued attempt waits proportionally to depth divided by service rate.

```text
FORMULA [FR-08]: W = Q / s
WHERE: W waiting time [hours]; Q queue depth [attempts]; s service rate [attempts/hour].
CONSTRAINT: stable queue with s > 0 and a work-conserving server; arrivals need not be Poisson
            for this deterministic approximation, but variability widens the distribution.
NUMERICAL EXAMPLE: Q=3,000 attempts, s=600 reviews/hour -> W=5.0 hours.
INTERPRETATION: an attempt queued at 23:00 is reviewed five hours later, after the authorization
            has settled, so its practical value is close to zero for blocking.
```

The consequence is severe and often unstated: a review that happens after settlement cannot prevent the loss, it can only confirm it.
Therefore the effective prevention rate is the fraction of reviewed attempts that are reviewed before settlement, not the fraction that are reviewed at all.
If settlement occurs at authorization, no amount of later review prevents loss, and the entire "fraud prevented by review" term in the ledger would be zero.
That is why real fraud operations treat review as a pre-settlement hold: the attempt is held, not merely flagged, and the hold duration is part of the policy.
This case assumes a hold mechanism exists and that reviews complete within it; that is assumption A-11 below, and it is the assumption with the largest unreported leverage in the whole ledger.

| Assumption | Statement | Why it matters | If false |
|---|---|---|---|
| A-11 | Review completes before the attempt settles, so a confirmed fraud can be blocked | Without it, review prevents nothing and only declines prevent loss | The ledger's 880 and 2,112 review-channel cases collapse toward zero |
| A-12 | Queue is worked work-conserving with a stable service rate | Waiting-time arithmetic above | Bursts create unbounded waits and the depth arithmetic understates harm |
| A-13 | Attempts age out at a documented age rather than silently | The baseline's "aging out" is itself a policy | Unreviewed attempts would be reviewed eventually and the baseline would look different |

Little's law applies directly: with 600 reviews/hour and 5.0 hours of waiting, the queue holds 3,000 attempts, which matches the stated depth by construction.
If the arrival rate is 700/hour against 600/hour service, the queue grows by 100 per hour, reaching 3,000 in 30 hours, and waiting time grows without bound unless arrivals fall.
At a fixed 4,800 reviews/day the daily capacity is 600 per eight-hour shift equivalent, so an hourly comparison must convert the daily figure into the shift structure rather than mixing units.
That unit conversion is a common error: 4,800 per day is 200 per hour averaged over 24 hours, but 600 per hour over eight staffed hours, and the queue behaviour differs entirely between those two readings.

**Takeaway:** prevention requires action before settlement, so queue depth and hold policy belong in the ledger, not in an appendix.

## 7. The Finding

Under the constructed assumptions, the ranked policy prevents 2,762 of 7,000 daily fraud attempts against 1,600 for the rules baseline at identical review capacity, which is 1,162 additional cases per day or a 72.6% relative increase over the baseline's prevention count.
Priced at $320 per case and net of the platform's 388 per day, the daily advantage is 372,700 currency units, or approximately 136 million per operating year at 365 days.
That magnitude is conditional on an unverified review-band precision of 0.44; the same ledger turns negative at any precision at or below 0.1974, and no precision estimate for any model has been measured here.
The evidence in this case is entirely arithmetic on invented inputs, so the finding is a decision procedure with a stated break-even rather than a recommendation to fund a specific platform.
The main boundary is that capacity, not accuracy, binds: if review capacity doubled, the marginal value of reordering would fall and the case for a larger review band would need re-examination rather than the case for the ranker being strengthened.
The next justified action is a four-week shadow evaluation on a mature cohort to estimate precision at 4,800 with an interval, followed by a latency load test; without both, funding would rest on an assumption.
A secondary finding is that the platform's infrastructure cost is roughly three orders of magnitude smaller than the loss it influences, so procurement debates about compute price are analytically irrelevant compared with validating precision and latency.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

```text
attempt + label history (Parquet, partitioned by UTC day)
        |
        v
[point-in-time join: feature event & availability <= attempt time] --> [leakage audit report]
        |
        v
[time-based split: 9 train / 2 valid / 2 test, 120-day label buffer]
        |
        v
[gradient-boosted tree training, early stopping on validation log-loss]
        |
        v
[calibration fit on disjoint validation slice] --> [calibration curve artifact]
        |
        v
[artifact registry: model + feature manifest + calibration version]
        |
        v
[evaluation job: precision at 4,800, band precision, economic ledger]
```

The training path reads only frozen snapshots; each snapshot carries the extraction timestamp and the eligibility manifest hash.
The label buffer of 120 days prevents the last training month from containing attempts whose chargebacks have not resolved.
The calibration slice is disjoint from the rows used to fit the tree ensemble, so the probability head is not fitted and evaluated on the same data.
The registry stores the model, the ordered feature manifest, and the calibration artifact as one release unit, because a model with mismatched calibration is a different policy.
The evaluation job emits both the point estimate and a cluster-bootstrap interval resampled by card, since attempts from one card are not independent.
Retraining cadence is monthly by default; an out-of-cycle retrain is triggered by a documented drift or precision breach, not by an alert firing alone.

### 8b. Online / Serving / Real-Time Path

```text
authorization request (attempt_id, card token, merchant, amount, ts)
        |
        v
[eligibility filter] --ineligible--> [return to existing rules path + reason code]
        |
        v
[hard-block rules: confirmed fraud list, sanctions-style static blocks]
        |
        v
[feature retrieval: velocity cache + point-in-time store, 25 ms budget]
        |                                          |
        |  cache miss / stale                     |  fresh
        v                                          v
[typed missingness -> abstain to review]     [model score, 3 ms budget]
        |
        v
[band assignment from calibrated probability and pinned thresholds]
        |
        +--> p >= 0.85 --> [decline] --> [audit record]
        +--> 0.15 <= p < 0.85 --> [review queue insert] --> [audit record]
        +--> p < 0.15 --> [approve] --> [audit record]
```

The in-path budget is 120 ms at p95, allocated as 25 ms retrieval, 3 ms scoring, 10 ms policy and logging, and the remainder for network and the existing authorization stack.
The allocation is a design intent, not a measured decomposition; no latency measurement exists.
Cache miss behaviour is the critical failure path: a miss must produce typed missingness and route to review rather than silently defaulting a velocity to zero.
The decline band requires a calibration check to have passed on the current artifact; if calibration status is unknown, the band collapses to review-only, which is the safe degradation.
Every branch writes an audit record before the response leaves the service, so a crash after decision but before logging is treated as a service failure rather than a decision.
Idempotency is keyed on attempt_id, so a retried authorization does not consume two review slots or produce two audits.
The rules overlay runs before the model, so a model outage degrades to rules rather than to approval.
A model timeout is treated as abstention to review, subject to a circuit breaker that prevents a timeout storm from flooding the queue.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Ranked queue at fixed capacity rather than a threshold that ignores capacity | Fixed probability threshold | Capacity is the binding constraint (C-01) | Requires daily capacity telemetry | Capacity varies hourly rather than daily | Review staffing model changes |
| DEC-02 | Calibrated probability with a high decline band retained | Rank-only score with rules decline | Decline needs a stable probability semantics | Calibration must be maintained and monitored | Calibration drift without monitoring | Calibration error beyond tolerance |
| DEC-03 | Typed missingness with abstention to review | Impute defaults inline | Missing velocity is not zero velocity | Raises review volume on outage | Abstention volume exceeds reserved capacity | Feature availability improves |
| DEC-04 | Rules overlay before model for hard blocks | Model includes hard blocks as features | Static blocks must not depend on model availability | Duplicated logic to maintain | Overlay list becomes stale | List governance changes |
| DEC-05 | Monthly retrain with documented triggers | Continuous online learning | Simpler audit and rollback | Slower adaptation to new fraud patterns | Fast-moving fraud ring | Pattern velocity exceeds monthly cadence |
| DEC-06 | Cluster bootstrap by card for intervals | Naive attempt-level bootstrap | Attempts within a card are dependent | Wider intervals, more data needed | Very sparse cards | Card-level clustering changes |

### 8d. Architecture Artifacts and Operational Readiness

| Area | Case-specific answer |
|---|---|
| Interfaces | Attempt event schema v3 with required fields; model response carries probability, model_version, feature_manifest_hash, band, and reason_code |
| Data/state | Velocity state keyed by card token with 7-day event-time windows; decision log append-only with 400-day retention |
| Reliability | SLI: fraction of eligible attempts receiving a decision within 120 ms; target 99.5% over a rolling day; measured at the service boundary |
| Resilience | Retrieval timeout 40 ms, model timeout 30 ms, circuit breaker opens after 5% error rate over 30 s, abstention fallback |
| Recovery | Replay from decision log plus feature store; RPO 0 for the decision log, RTO 30 min for read-path restore, both unvalidated |
| Observability | Per-band counts, abstention rate by reason, queue depth, review throughput, precision on matured cohorts |
| Deployment | Model, feature manifest, and calibration deploy as one release unit; rollback restores the prior triple atomically |
| Security | Tokenized card identifiers; least-privilege access to decision log; no raw PAN in any store; audit access itself logged |
| Economics | 388/day platform cost; review cost 40,320/day; loss term dominant at over 97% of total |
| Human workflow | Ops lead sets capacity and queue policy; analysts confirm or clear; risk officer holds decline-band veto |

Capacity arithmetic for the online path at peak: if peak traffic is 2.6 times the daily average rate, then 2,000,000 attempts/day is a mean of 23.15 attempts/s, and peak is about 60.2 attempts/s.
At 60.2 attempts/s with a 120 ms budget, Little's law gives an in-flight concurrency of 60.2 × 0.120 = 7.22 requests, which is small and implies the concurrency limit is not the constraint.
CPU demand at peak is 60.2 × (3 + 25) ms = 60.2 × 0.028 = 1.686 CPU-seconds per second, i.e. about 1.69 cores of sustained work before headroom.
Provisioning at three times that figure gives roughly 5.1 cores, which is a small fleet; the real cost driver is the analyst hours at 40,320/day against 388/day of infrastructure.
This arithmetic supports the earlier claim that platform economics are dominated by people, and it is the kind of sizing that must be shown rather than asserted.

**Takeaway:** the online path's expensive resource is human attention; the design's job is to keep that attention aimed at the highest-value attempts and to fail safe when inputs are missing.

## 9. Evaluation & Validation

The primary metric is precision at capacity: among the top 4,800 scored non-declined attempts in a day, the fraction that are confirmed fraud, aggregated across mature days, with a cluster bootstrap interval resampled by card.
Its numerator is confirmed fraud attempts inside the selected set; its denominator is exactly 4,800 per day, provided at least 4,800 eligible non-declined attempts exist, otherwise the actual selected count.
Secondary metrics are: review-band recall (selected fraud divided by all fraud not caught by the decline band), decline-band precision, false-decline rate per 10,000 legitimate attempts, abstention rate by reason, and p95 authorization latency.
The comparator is the rules engine on the same eligible population, same capacity, and the same mature window; where a randomized pilot exists, the comparator is the concurrent control slice.
The pass/fail rule is precision at capacity ≥ 0.1974 with the lower bound of a 95% interval above that break-even; the threshold is derived in Section 6 and selected before any test window is read.
Guardrails are: false-decline rate not to exceed the baseline's rate, abstention rate not to exceed the reserved review headroom, and p95 latency within 120 ms.
The test design is a four-week non-intervening shadow followed by a capacity-neutral randomized pilot on a merchant slice large enough to estimate precision with usable width.

Sample-size reasoning for the pilot: with p near 0.44 and 4,800 reviews per day, a naive standard error for a day's precision is sqrt(0.44 × 0.56 / 4,800) = sqrt(0.2464 / 4,800) = sqrt(0.00005133) = 0.007165.
Two mature weeks give 67,200 reviews, and the naive standard error shrinks to sqrt(0.2464 / 67,200) = 0.001914.
That shrinkage overstates precision because reviews cluster by analyst and by card, so the effective sample size is smaller; assuming a design effect of 2.0, the two-week standard error becomes about 0.002707.
Even with that inflation the interval is comfortably narrow relative to the 0.0427 gap between 0.44 and the 0.1974 break-even, which means the pilot can distinguish "clear pass" from "clear fail" but cannot resolve small differences.
That distinction matters: the pilot is sized to answer the funding question, not to rank two candidate models against each other.

```text
FORMULA [FR-06]: SE(p) = sqrt(p(1-p)/n) * sqrt(deff)
WHERE: SE standard error [dimensionless]; p precision [dimensionless]; n reviews [count];
       deff design effect [dimensionless, >= 1 for clustering].
CONSTRAINT: large-sample normal approximation; deff is an assumption, not measured here.
NUMERICAL EXAMPLE: p=0.44, n=67,200, deff=2.0 ->
       sqrt(0.44*0.56/67,200)=0.001914; 0.001914*sqrt(2)=0.002707.
INTERPRETATION: a 95% interval of roughly +/-0.0053 is narrow enough to compare against 0.1974.
```

Uncertainty for the economic comparison compounds this precision interval with uncertainty in loss per case and in the decline-band assumptions; the case does not construct a joint interval because the input distributions are unknown.
Validity threats are named explicitly: label maturity, decline-hiding of fraud, analyst rubric disagreement, and seasonal fraud composition.
Each threat has a planned check: cohort maturity reconciliation, holdout-based decline analysis, rubric agreement sampling, and matched-weekday comparison across seasons.
Planned technical checks include point-in-time replay rejecting any post-authorization feature, duplicate-attempt suppression, and clock-skew injection around the cutoff boundary.
Planned failure exercises include feature-store unavailability, model timeout, and decision-log write failure, each with an expected fallback behaviour.
None of these tests has been executed; the entire evaluation section describes intent.

**Takeaway:** the pass rule is a number with an interval, and the pilot is sized to resolve the funding question rather than to certify a model.

## 10. Action Taken

No action has been taken; this is an illustrative design exercise with a proposed decision procedure.
The recommendation would be to fund a bounded four-week shadow evaluation and a latency load test, not a production rollout.
Approval would sit with the fraud-operations lead, with the risk officer's veto on any decline-band change and the platform owner's sign-off on artifacts.
Implementation would be handed to the platform team with a runbook covering cache-miss behaviour, circuit-breaker thresholds, and queue overflow.
Effective scope of any pilot would be one merchant slice within the in-scope channel, with an explicit list of included merchant identifiers.
The release strategy would be a randomized capacity-neutral pilot, with the control slice continuing on rules, and no change to the auto-decline band during the first two mature weeks.
Stop conditions would be: precision at capacity below break-even after two mature weeks, false-decline rate above baseline, or p95 latency above 120 ms sustained for one hour.
Rollback would restore the rules-only policy path, which remains deployed and tested throughout the pilot.
Excluded actions: changing the decline threshold, expanding to card-present, and altering analyst staffing; each is a separate decision with different evidence requirements.

## 11. Outcome & Results

Outcome not yet measured; every figure below is an illustrative projection, not a result.

| Metric | Baseline: value/window/N | Treatment/after: value/window/N | Absolute change | Relative change | Uncertainty | Evidence/source | Limitation |
|---|---|---|---|---|---|---|---|
| Fraud cases prevented per day | 1,600 / constructed day / 7,000 | 2,762 / constructed day / 7,000 | +1,162 | +72.6% over 1,600 | none; assumed inputs | Illustrative arithmetic, Section 6 | Not measured; depends on assumed 0.44 precision |
| Review-band precision | 0.183333 / 4,800 reviews | 0.44 / 4,800 reviews | +0.256667 | +140% relative | interval not constructed for a projection | Illustrative arithmetic, Section 6 | Assumption, not an estimate |
| Net daily cost | 1,772,928 / day | 1,400,228 / day | −372,700 | −21.0% | none constructed | Section 6.2 ledger | Priced with invented unit costs |
| False-decline count | 480 / day | 350 / day | −130 | −27.1% | none constructed | Section 6.2 | Assumed precision of both bands |
| Platform cost | 0 / day | 388 / day | +388 | undefined from zero baseline | build estimate uncertain by ±3× | Section 6.3 | Excludes organizational change cost |
| p95 authorization latency | unknown | unknown | not measured | not applicable | no measurement | Absent | The 120 ms budget is unverified |

Relative change against a zero baseline is reported as undefined rather than as a percentage, because dividing by zero produces no meaningful ratio.
The 21.0% cost reduction is computed as 372,700 / 1,772,928 = 0.210216, and it must not be read as a 21% improvement in fraud outcomes, since the prevention metric moves 72.6% and the two are different questions.
An operational adoption metric would be the fraction of analysts working the score-ordered queue without manual re-sorting; that is unmeasured and would be needed before claiming operational benefit.
A sustained-duration metric would be the number of consecutive weeks with precision at capacity above break-even on mature cohorts; also unmeasured.

**Takeaway:** the projection table is a decision instrument with explicit assumptions, not a results table, and the latency row is empty because nothing was run.

## 12. Failure Modes & What Went Wrong

Nothing went wrong in this case, because nothing was built. The entries below are hypothetical failure mechanisms with discriminating tests, labelled as constructed.

### F-01: Decline-hiding inflates measured precision
Symptom: after the decline band narrows, measured fraud prevalence rises in the review band even though the model is unchanged.
Mechanism: declined attempts never produce a chargeback, so removing them from the observable population changes which fraud remains visible.
Discriminating test: compare the same cohort's label counts before and after the band change on a held-out slice that is not declined by either policy.
Competing explanation: the model genuinely improved ranking; ruled out only if the model artifact and scores are identical between windows.
Containment: freeze band changes during any evaluation window and record the policy version with each decision.
Prevention: always report precision with the policy version and the decline counts that shaped the observable population.
Blast radius: any stakeholder reading a precision number across a band change; the funding decision itself.
Verification: replay one week with bands held fixed and confirm precision reproduces within its interval.
Evidence status: hypothetical.

### F-02: Label maturity produces a premature pass
Symptom: the first two weeks of the pilot show precision 0.52, well above break-even, and the team proposes immediate expansion.
Mechanism: fast-resolving chargebacks skew toward small-ticket, high-velocity fraud, so early-mature cohorts are not representative of the full loss mix.
Discriminating test: compare the loss-per-case distribution of resolved versus unresolved attempts at the same cohort age.
Competing explanation: the model really is better on fast frauds; that is compatible with the maturity bias and does not remove it.
Containment: restrict any early claim to the resolved subset and label it provisional.
Prevention: require 120-day maturity and 97% resolution before a headline precision claim.
Blast radius: an expansion approved on a biased subset, later reversed after large-ticket fraud surfaces.
Verification: recompute the same cohort at full maturity and compare against the provisional figure.
Evidence status: hypothetical.

### F-03: Cache miss silently defaults a velocity to zero
Symptom: score distribution shifts toward approvals during a cache deployment, with no error rate increase.
Mechanism: a lookup miss returns a default that the serving code treats as a legitimate zero count.
Discriminating test: join lookup response codes to served feature values and count zero-valued features whose lookup failed.
Competing explanation: genuine zero-velocity attempts, which are common for new cards; discriminated only by joining the response code.
Containment: disable the default, route misses to abstention, and expect review volume to rise.
Prevention: typed missingness in the response schema with a non-nullable reason code.
Blast radius: every attempt scored during the deployment window, particularly new-card attempts.
Verification: inject a forced miss in staging and confirm the response is an abstention, not a zero.
Evidence status: hypothetical.

### F-04: Fan-out join duplicates decisions
Symptom: the decision log contains 2,140,000 records for a day with 2,000,000 eligible attempts.
Mechanism: the merchant dimension table contains multiple effective-dated rows per merchant identifier, and the join multiplies attempts.
Discriminating test: count rows per attempt_id before and after each join, and inspect the effective-date ranges of duplicated merchants.
Competing explanation: legitimate retries, which produce distinct attempt_ids and must be counted separately.
Containment: deduplicate on the effective-dated row valid at authorization time and recompute the day's metrics.
Prevention: cardinality assertions on every dimension join with an explicit one-to-one contract.
Blast radius: the count of reviews consumed, since duplicate records would appear to justify more reviews than capacity allows.
Verification: a fan-out test fixture with three effective versions must yield exactly one row.
Evidence status: hypothetical.

### F-05: Capacity is modelled per day while arrivals are hourly
Symptom: the queue is empty at 06:00 and 3,000 deep at 23:00, with the daily total still within capacity.
Mechanism: fraud attempts cluster in hours when legitimate volume is high, so a daily capacity statement hides an hourly imbalance.
Discriminating test: plot queue arrival rate and review completion rate by hour and compute the maximum hourly deficit.
Competing explanation: analyst shift patterns rather than arrival skew; discriminated by comparing arrivals to logged analyst availability per hour.
Containment: add hourly capacity checks and an overflow rule that prioritizes by expected loss within the hour.
Prevention: define capacity per hour as well as per day, with a documented burst allowance.
Blast radius: late-arriving fraud attempts, which are systematically under-reviewed.
Verification: replay one day's arrivals against hourly staffing and confirm no hour exceeds capacity without invoking overflow.
Evidence status: hypothetical.

### F-06: Calibration drift changes the effective decline band
Symptom: the number of attempts crossing the 0.85 threshold rises 40% while the model version is unchanged.
Mechanism: the calibration artifact was fitted on a period with different prevalence, so the same score no longer maps to the same probability.
Discriminating test: build a reliability curve on a mature recent cohort and compare observed frequency to predicted probability by bin.
Competing explanation: a genuine fraud-rate increase that raises scores legitimately; distinguished by whether the ranking order also changed.
Containment: restrict the decline band to review-only until recalibration is validated.
Prevention: monitor calibration by bin on a rolling mature window and treat calibration as a versioned release artifact.
Blast radius: legitimate customers declined by an inflated band, with immediate and lasting friction.
Verification: after recalibration, confirm bin-level observed frequencies fall within their intervals.
Evidence status: hypothetical.

### F-07: Analyst rubric disagreement makes precision unmeasurable
Symptom: two analysts reviewing the same 200 attempts disagree on 46 of them, and precision estimates move by several points depending on who reviews.
Mechanism: the case-management rubric does not define what evidence counts as confirmation for a not-yet-charged-back attempt.
Discriminating test: double-review a random sample and compute agreement with an adjudication of disagreements.
Competing explanation: genuinely ambiguous individual cases rather than a rubric defect; distinguished by whether disagreement concentrates in a describable evidence class.
Containment: adjudicate all disagreeing cases and report precision on adjudicated outcomes only.
Prevention: a written rubric with worked examples, plus periodic agreement monitoring.
Blast radius: the entire precision metric that the funding decision depends on.
Verification: re-run agreement after rubric revision and confirm it improves above the pre-set floor.
Evidence status: hypothetical.

### F-08: Model timeout storm floods the review queue
Symptom: during a dependency slowdown, abstention count exceeds 4,800 within twenty minutes.
Mechanism: timeouts route to review by design, so a slow dependency converts into a queue flood that displaces genuine high-risk attempts.
Discriminating test: measure abstention rate and model p99 latency together, and check whether abstentions cluster on timeout reason codes.
Competing explanation: a genuine fraud wave producing many high scores; distinguished by the reason code and by score distribution among abstentions.
Containment: open the circuit breaker, fall back to the rules path for the duration, and record the fallback interval.
Prevention: a reserved review headroom plus a circuit breaker with a documented error threshold.
Blast radius: all reviewed attempts during the outage, including genuine fraud that is displaced.
Verification: fault-inject a 5-second model delay and confirm the breaker opens and the rules path serves.
Evidence status: hypothetical.

### F-09: Randomized pilot assignment leaks across merchants
Symptom: the treatment effect appears smaller than offline expectations and control-slice precision is higher than the rules baseline.
Mechanism: the assignment unit is the merchant, but fraud rings operate across merchants, so treating one merchant changes the fraud pressure on its neighbours.
Discriminating test: compare control-slice precision across merchants with high and low treatment-neighbour density.
Competing explanation: the ranker is simply weaker than assumed; distinguished by whether the effect varies with neighbour density.
Containment: switch to a coarser assignment unit or accept an interference-adjusted estimand.
Prevention: choose the assignment unit by the interference structure before the pilot begins.
Blast radius: the interpretation of the whole pilot result.
Verification: a pre-registered spillover check comparing boundary and interior control merchants.
Evidence status: hypothetical.

### F-10: Feature backfill rewrites history used for evaluation
Symptom: re-running the evaluation produces a different precision than the reported figure, with identical model and labels.
Mechanism: a corrected merchant attribute was backfilled in place, so the point-in-time value at authorization is no longer reproducible.
Discriminating test: compare the feature manifest hash recorded with the decision to the hash produced by today's backfill.
Competing explanation: label corrections, distinguished by whether label counts also changed.
Containment: freeze the evaluation snapshot and report the manifest hash alongside every metric.
Prevention: append-only feature revisions with as-of queries rather than in-place updates.
Blast radius: every historical evaluation, including the one supporting the funding decision.
Verification: reproduce a stored evaluation from the frozen snapshot and confirm exact equality.
Evidence status: hypothetical.

### F-11: Threshold copied across currencies
Symptom: a policy validated in one currency declines an implausible share of attempts after expanding to a second currency with a different typical ticket size.
Mechanism: the probability threshold is valid, but the loss-per-case and margin terms that justified it do not transfer, so the cost-optimal band is different.
Discriminating test: recompute the break-even band per currency using local loss and margin values.
Competing explanation: genuine fraud-rate differences between currencies; distinguished by comparing precision at the same band across currencies.
Containment: restrict the band to the validated currency and route the new currency to review.
Prevention: store band configuration per currency with its own break-even computation.
Blast radius: customers in the newly added currency.
Verification: replay the new currency through review-only for one mature window before enabling a decline band.
Evidence status: hypothetical.

### F-12: Loss-per-case is averaged across bands that differ by 4x
Symptom: the funded business case overstates benefit and the realized loss reduction is far below projection.
Mechanism: low-value, high-velocity fraud dominates the case count, while high-value fraud dominates the loss; a single average loss per case misprices the ranking.
Discriminating test: compute loss per case by amount band and compare which bands the ranker moves into the review set.
Competing explanation: lower-than-expected precision; distinguished by whether precision matched while loss did not.
Containment: rebuild the ledger with band-specific loss values and re-derive break-even precision.
Prevention: value-weighted evaluation as the primary metric where loss varies materially by case.
Blast radius: the funding decision's magnitude, though not necessarily its sign.
Verification: recompute projected benefit with a loss table and compare against the single-average figure.
Evidence status: hypothetical.

**Takeaway:** every failure above attacks the same weak point — the link between a measured score and a real decision — and each has a cheap discriminating check that costs far less than the failure.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `evaluation-mismatch`: a model metric improves while the capacity-constrained decision does not, because the metric omits the binding constraint |
| General principle | When a downstream human or system resource is the scarce input, evaluate the policy at that resource's capacity, not on the full population |
| Preconditions | The capacity is known and stable; the action is comparable across units; a matured outcome exists; the comparator is measured under the same capacity |
| Mechanism | Ranking changes which units receive scarce action; its value is the change in outcome among the acted-upon set, which is precisely what precision at capacity measures |
| Supporting evidence | The derived break-even of 0.1974 and the ledger decomposition are arithmetic on stated assumptions; no measured evidence exists |
| Avoid when | Capacity is elastic, the outcome is unaffected by the action, or the action has irreversible consequences that require a different decision rule |
| Portability test | Ask: if I held the budget fixed and doubled the population, would my metric change without any change in the policy's quality? If yes, the metric is not capacity-aligned |
| Adaptation | In content moderation the capacity is reviewer-hours; in collections it is calls per day; in dispatch it is vehicles; in each case the metric must be normalized by that resource |

The mechanism is structural rather than incidental: any system that acts on a strict subset of candidates has a selection boundary, and quality at that boundary is the only quality the business experiences.
The most common misapplication is to compute precision on all flagged units when the flag rate exceeds capacity, which produces a number that no operator can act on.
A second misapplication is to compare a capacity-constrained model to an unconstrained historical process, which flatters the model by granting it resources the baseline never had.
The lesson also has a negative form: when capacity is elastic and cheap, ranking gains matter less than absolute coverage, and the correct investment is throughput rather than ordering.
Distinguishing those two regimes is the first question to ask in any fraud, moderation, or triage proposal, and it is answerable before any model is trained.

**Takeaway:** the pattern transfers wherever a scarce human or operational resource selects among candidates; the arithmetic changes, the structural question does not.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Actual target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| `uses-method` | [Feature Store Design](../Cheat_Sheets/03_Feature_Store_Design.md) | Point-in-time feature eligibility, revision ordering, parity | Feature parity prevents leakage but says nothing about capacity value |
| `uses-method` | [ML System Problem Framing](../Cheat_Sheets/01_ML_System_Problem_Framing.md) | Decision contract, expected action value, coverage | Framing yields `coverage`; this case prices the composition inside that coverage |
| `contrasts-with` | [Drift triage case](01_model_monitoring_drift_incident.md) | Both are production decisions under uncertainty | Drift asks whether to contain; this case asks whether to fund, and neither answer implies the other |
| `extends` | Local case collection [index](INDEX.md) | Adds the capacity-constrained economics to the platform catalogue | The ledger is arithmetic, not an observed result |

The feature-store handoff is precise: this case's A-10 temporal contract is exactly the eligibility rule derived there, and a violation would invalidate every precision figure rather than merely reduce it.
The framing handoff is also precise: `coverage = min(K,N)/N` explains why reach is small, and this case adds the question of which units are inside the reach.
The drift case supplies the containment logic that would govern the pilot if precision degraded mid-flight; this case supplies the funding logic that determines whether the pilot should exist.

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data/test that would resolve it | Priority / trigger | Owner role |
|---|---|---|---|---|
| What is precision at 4,800 for a real model? | No model was trained; 0.44 is assumed | Four-week shadow on a matured cohort with cluster intervals | Highest; blocks funding | Evaluation owner |
| Does review actually precede settlement? | A-11 is unverified and has the largest leverage | Settlement and review completion timestamps joined per attempt | Highest; can zero the ledger | Fraud-operations lead |
| How variable is loss per case across the reviewed mix? | A single $320 average hides a likely 4x spread | Band-specific loss attribution over a mature quarter | High; changes the ranking objective | Risk analytics |
| What is the latency distribution at peak with real dependencies? | No load test exists | Peak-rate load test with production-shaped dependency latency | High; determines in-path feasibility | Platform owner |
| How much fraud does the decline band hide from measurement? | Declined attempts never charge back | Randomized holdout of a small decline-band slice, where authorized | Medium; affects all future precision readings | Risk officer |

The first two questions are ordered deliberately: evaluating precision on a system that cannot act before settlement would produce a correct number about a policy that cannot work.

### 14c. Interview pack

Fourteen questions follow, each answered under the invented assumptions. No answer should be read as an observed result; the arithmetic is checked and the inputs are constructed.

#### Question 1: What decision does this case actually make?
- Level: framing. Skill: decision-contract construction.
- Candidate prompt: "Here is a fraud platform proposal with a better model. What decision are you being asked to make?"
- What the candidate should clarify: who owns the queue, what the review budget is, whether the decline band changes, and what outcome is being optimized.
- Strong answer: the decision is whether to fund a ranking platform that reorders a fixed 4,800-review budget; it is not a decision about model accuracy. The candidate states the owner (fraud-operations lead), the comparator (rules at identical capacity), the constraint (capacity and 120 ms), and the objective (net loss per day).
- Acceptable alternatives: framing it as a policy-selection decision rather than a platform decision is equally strong, provided the capacity constraint is named.
- Follow-up: "What if capacity could be doubled instead?" Answer: then the investment question shifts toward staffing throughput, and the ranking gain must be re-derived at the larger K because precision at 9,600 will be lower than at 4,800.
- Red flags: describing the decision as "improve the model" or listing architecture components before naming an owner and a constraint.
- Scoring: 0 = no decision or owner; 1 = decision named but capacity omitted; 2 = decision, owner, comparator, constraint stated; 3 = also explains why capacity, not accuracy, binds and what would change that.
- Answer support: Sections 1, 3, and 6.1.

#### Question 2: Why is AUC an inadequate headline metric here?
- Level: framing/evaluation. Skill: metric-to-decision alignment.
- Candidate prompt: "The model improves AUC from 0.81 to 0.87. Is that enough to fund it?"
- What the candidate should clarify: what fraction of the population is acted upon, and whether the AUC gain comes from regions the policy never reaches.
- Strong answer: AUC integrates over all thresholds and all units, while the policy acts on the top 4,800 of roughly 2 million. A gain concentrated below the cutoff changes nothing operationally. The candidate should ask for precision at capacity and note that a model can improve AUC by fixing easy negatives while leaving the boundary untouched.
- Acceptable alternatives: quoting a partial AUC restricted to the top decile is acceptable if the candidate explains the restriction matches the operating point.
- Follow-up: "Could AUC fall while the policy improves?" Answer: yes; reordering within the top 4,800 can raise precision at that boundary while degrading discrimination in the deep tail where most pairs live.
- Red flags: treating AUC as a probability, or asserting that higher AUC implies better calibration.
- Scoring: 0 = accepts AUC as sufficient; 1 = says AUC is imperfect without locating the mismatch; 2 = identifies the population mismatch; 3 = also explains a mechanism by which AUC and precision at capacity diverge.
- Answer support: Sections 6.1 and 9.

#### Question 3: Compute precision at capacity for the baseline.
- Level: calculation. Skill: denominator discipline.
- Candidate prompt: "Rules produce 18,000 alerts containing 3,300 frauds; you can review 4,800 in arrival order. What precision do you expect in the reviewed set?"
- What the candidate should clarify: whether arrival order is independent of risk, and whether all 4,800 reviews complete.
- Strong answer: band precision is 3,300/18,000 = 0.1833. If arrival order is exchangeable with respect to risk, the expected reviewed fraud count is 4,800 × 0.1833 = 880, so precision in the reviewed set is 880/4,800 = 0.1833 — the same number, because random subsetting preserves proportion. The candidate should note the 3,300/18,000 ratio and the 880/4,800 ratio are different quantities that coincide only under exchangeability.
- Acceptable alternatives: reporting the reviewed-set precision directly as 0.1833 with the reasoning stated is equivalent.
- Follow-up: "What if fraud clusters at night and the queue is worked during the day?" Answer: exchangeability fails, reviewed precision is below 0.1833, and the true baseline is worse than reported — which strengthens the case only if the proposal is measured the same way.
- Red flags: computing 3,300/4,800, or dividing by the wrong population.
- Scoring: 0 = arithmetic error; 1 = correct number without justification; 2 = correct with the exchangeability caveat; 3 = also identifies the clustering counterexample.
- Answer support: Section 6.1.

#### Question 4: Derive the break-even review precision.
- Level: calculation. Skill: economic algebra.
- Candidate prompt: "Given the ledger terms, at what review precision does the proposal stop being better than the baseline?"
- What the candidate should clarify: whether the decline band is held fixed, whether review count is equal, and which costs cancel.
- Strong answer: with the decline band fixed at 650 prevented and 350 legitimate declines, the proposal's cost is (7,000 − 650 − 4,800p) × 320 + 3,360 + 40,320 + 388. Setting that equal to 1,772,928 gives 1,536,000p = 303,140, so p = 0.197356. The candidate should show that review cost cancels because both policies review 4,800.
- Acceptable alternatives: solving incrementally — the fixed disadvantage is 388 + 22,400 − 1,248 = 21,540, requiring 21,540/320 = 67.3125 extra cases, i.e. 0.014023 extra precision — is an equally strong route and cross-checks the first.
- Follow-up: "What if the platform cost tripled?" Answer: the extra 776 per day requires 2.4 more cases, i.e. 0.00051 precision, so the break-even moves to about 0.1979 — materially unchanged, which shows the decision is insensitive to infrastructure price.
- Red flags: forgetting the decline-band loss, or dividing the cost gap by 4,800 without converting currency to cases.
- Scoring: 0 = no valid equation; 1 = partial setup with an arithmetic error; 2 = correct 0.1974 with units; 3 = correct plus an independent cross-check and a sensitivity statement.
- Answer support: Sections 6.2 and 6.4.

#### Question 5: Is a random-selection baseline acceptable?
- Level: judgment. Skill: comparator selection.
- Candidate prompt: "We compared the model to random review and it looks excellent. Ship it?"
- What the candidate should clarify: what the actual incumbent policy is and what resources it had.
- Strong answer: no. Random review of 4,800 from 1,998,800 non-declined attempts with 6,280 frauds yields precision 0.00314, about 15 cases per day, roughly 58 times worse than the rules engine's 880. Beating random proves the model ranks better than chance; it says nothing about beating the incumbent. The correct comparator is the rules policy at the same capacity.
- Acceptable alternatives: using random as a floor alongside the real comparator is good practice, provided it is labelled a floor.
- Follow-up: "When is random actually the right comparator?" Answer: only when the incumbent is literally random allocation, such as an unprioritised queue that is genuinely first-come-first-served with risk-independent arrivals.
- Red flags: accepting any positive lift as justification, or treating the random baseline as conservative.
- Scoring: 0 = endorses shipping; 1 = doubts without quantifying; 2 = computes the floor and names the right comparator; 3 = also identifies when random would be legitimate.
- Answer support: Section 6.4.

#### Question 6: Where does the 0.44 assumption come from, and how would you test it?
- Level: data/method. Skill: assumption discipline.
- Candidate prompt: "Your ledger depends on a precision you invented. Why should anyone trust the conclusion?"
- What the candidate should clarify: what is being claimed — a decision procedure or a measured benefit.
- Strong answer: the 0.44 is an illustrative input, and the conclusion is conditional: fund only if measured precision exceeds 0.1974 with a defensible interval. The test is a four-week shadow run that scores without acting, followed by matured-cohort evaluation with a cluster bootstrap by card.
- Acceptable alternatives: a retrospective replay on historical attempts using point-in-time features is acceptable as a first pass, provided it is labelled retrospective and cannot capture intervention effects.
- Follow-up: "Why shadow rather than a live pilot first?" Answer: shadow isolates ranking and calibration from the confounding introduced by acting, and it is safe; the live pilot is needed later for prevention claims because acting changes label observability.
- Red flags: presenting the number as an estimate, or claiming the model "will" achieve it.
- Scoring: 0 = treats the assumption as fact; 1 = admits it is assumed but proposes no test; 2 = proposes a shadow evaluation with maturity; 3 = also distinguishes shadow evidence from intervention evidence.
- Answer support: Sections 4, 6.6, and 9.

#### Question 7: How would you size the pilot?
- Level: calculation. Skill: uncertainty and sample size.
- Candidate prompt: "How much data do you need before you can call the pilot?"
- What the candidate should clarify: the required resolution, the clustering structure, and the maturity horizon.
- Strong answer: the naive standard error at p = 0.44 with n = 4,800 per day is 0.00716; two mature weeks at n = 67,200 gives 0.00191, inflated to about 0.00271 under a design effect of 2.0 from card and analyst clustering. The resulting 95% half-width near 0.0053 is small relative to the 24-point gap to break-even, so two mature weeks sufficed to answer this question.
- Acceptable alternatives: sizing to detect a smaller effect, such as a 1-point precision difference, requires far more data and should be rejected as unnecessary for a funding decision.
- Follow-up: "What if precision came out at 0.205, just above break-even?" Answer: that is within noise of the threshold and should not be treated as a pass; the correct action is to extend the window or defer, since a decision that flips within its own interval is not evidence.
- Red flags: sizing by total attempts rather than reviews, or ignoring clustering.
- Scoring: 0 = no quantitative reasoning; 1 = standard error without clustering; 2 = correct with design effect; 3 = also states what a borderline result implies for the decision.
- Answer support: Section 9.

#### Question 8: Design the in-path architecture under a 120 ms budget.
- Level: design. Skill: latency budgeting and failure semantics.
- Candidate prompt: "Walk through one authorization from request to decision, and tell me what happens when the feature store is slow."
- What the candidate should clarify: the p95 target's measurement point, the dependency timeouts, and the acceptable degradation.
- Strong answer: eligibility filter, hard-block rules, feature retrieval within 25 ms, model within 3 ms, policy and audit within 10 ms, with the remainder for network and the authorization stack. On a retrieval miss or stale value the response is typed missingness and abstention to review, never a default. On repeated timeouts a circuit breaker opens and the rules path serves, with the interval recorded.
- Acceptable alternatives: precomputing velocity features on a streaming path and reading them from a bounded local cache is acceptable if staleness is bounded and monitored.
- Follow-up: "Why not just approve on timeout?" Answer: because a timeout carries no information about fraud, and approving converts an availability incident into a loss incident; abstention preserves the decision while degrading latency.
- Red flags: defaulting missing features to zero, or letting the model outage degrade to approval.
- Scoring: 0 = no failure path; 1 = names components without budgets; 2 = budgets and a failure route; 3 = also explains why the chosen degradation is safe and how it is bounded.
- Answer support: Sections 8b and 8d.

#### Question 9: What does a feature-store outage do to the ledger?
- Level: debug/design. Skill: coupling failure to economics.
- Candidate prompt: "During a 40-minute feature outage, the system abstains. What is the operational and economic consequence?"
- What the candidate should clarify: whether abstention was within reserved headroom, and whether the rules path was used.
- Strong answer: at 60.2 attempts/s peak, 40 minutes is 2,400 s, so 60.2 × 2,400 = 144,480 attempts. Under pure abstention those would swamp the 4,800 daily review budget about 30 times over, so the circuit breaker must divert to rules rather than to review. Economically, those attempts revert to baseline protection, so the loss is the difference between proposal and baseline protection over that window, not a total loss of protection.
- Acceptable alternatives: bounding abstention to the reserved headroom and diverting the rest to rules is the correct bounded design.
- Follow-up: "How would you quantify the cost of the outage?" Answer: prevented cases under proposal minus prevented under baseline, scaled by the fraction of the day affected and priced at the loss per case; with a 372,700 daily advantage, 40/1440 of a day is about 10,353 of foregone advantage, not a 372,700 loss.
- Red flags: claiming the outage costs a full day's benefit, or claiming no cost because rules still protected.
- Scoring: 0 = no consequence; 1 = qualitative only; 2 = identifies diversion to rules and the proportional loss; 3 = computes the proportional figure and notes the attempt-volume mismatch.
- Answer support: Sections 8b, 8d, and 6.2.

#### Question 10: How do delayed labels limit what you can conclude?
- Level: evaluation. Skill: label maturity and selection.
- Candidate prompt: "Two weeks into the pilot, precision looks great. Can you recommend expansion?"
- What the candidate should clarify: cohort maturity, resolution rate, and whether fast-resolving cases differ in loss.
- Strong answer: no. With a 120-day horizon and 97% resolution at maturity, a two-week cohort is dominated by fast-resolving cases, which skew to small-ticket, high-velocity fraud. The evaluation must either wait or report the partial result labelled provisional with the resolved fraction. Expansion on the provisional figure risks approving on a biased subset.
- Acceptable alternatives: reporting both provisional and mature figures on an older cohort is acceptable and often more informative.
- Follow-up: "How would you test whether the bias is material?" Answer: compare loss-per-case and amount distributions between resolved and unresolved attempts at the same cohort age; a large gap means the provisional estimate is not representative.
- Red flags: treating unresolved outcomes as negatives, or claiming precision from a cohort that is 40% resolved.
- Scoring: 0 = endorses expansion; 1 = mentions delay without consequence; 2 = explains the selection mechanism; 3 = proposes a specific bias test and a decision rule.
- Answer support: Sections 5a, 9, and 12 F-02.

#### Question 11: The decline band's precision is 0.65. Is that acceptable?
- Level: judgment. Skill: asymmetric-cost reasoning.
- Candidate prompt: "35% of declined attempts were legitimate. Should we keep the band?"
- What the candidate should clarify: the margin lost per legitimate decline, the loss prevented per caught fraud, and whether alternatives exist.
- Strong answer: it depends on the exchange rate. At $9.60 of margin per legitimate decline and $320 of loss per fraud, the band is favourable in expectation when 0.65 × 320 > 0.35 × 9.60, i.e. 208 > 3.36, which is overwhelmingly true. But the arithmetic ignores retention, reputational cost, and the fact that a declined legitimate customer may not return; the guardrail should be a customer-level friction metric, not only the immediate margin.
- Acceptable alternatives: arguing for review instead of decline on the same cases, which trades latency for certainty, is a legitimate alternative answer if the queue has headroom.
- Follow-up: "Why did the proposal's decline band prevent fewer frauds than the baseline's?" Answer: because it is smaller (1,000 vs 1,200 attempts) and higher-precision, so it forgoes 70 catches to avoid 130 legitimate declines — a deliberate trade priced in the ledger.
- Red flags: evaluating precision without any cost ratio, or ignoring the decline's irreversible effect on the customer.
- Scoring: 0 = no cost reasoning; 1 = notes tradeoff qualitatively; 2 = computes the expectation; 3 = also names the unmodelled customer-retention cost.
- Answer support: Sections 6.2 and 6.5.

#### Question 12: Probability ranking or value ranking?
- Level: method selection. Skill: objective-to-ranking alignment.
- Candidate prompt: "Should the queue be ordered by fraud probability or by expected dollar loss?"
- What the candidate should clarify: whether amount is known at decision time and whether loss varies materially by amount.
- Strong answer: the ledger is in currency, so ranking by expected preventable loss, `p_i × x_i × r`, is aligned with it. The worked comparison shows a 0.20-probability $900 attempt is worth 99 versus 11 for a 0.50-probability $40 attempt, so probability ranking reviews the wrong case first. Probability ranking is correct only when loss is roughly constant across the queue.
- Acceptable alternatives: ranking by expected loss per review-minute is better still if a duration model exists, since it prices the actual scarce resource.
- Follow-up: "What breaks the value ranking?" Answer: miscalibration that varies by amount band, since `p_i` fitted mostly on small attempts may be unreliable in the large-amount tail; also unknown recoverability `r`, and attempts whose amount changes after authorization.
- Red flags: asserting probability ranking is always correct because probabilities are calibrated.
- Scoring: 0 = no distinction; 1 = picks one without reasoning; 2 = aligns objective to ledger with an example; 3 = also identifies the calibration-by-amount failure.
- Answer support: Section 6.7.

#### Question 13: Diagnose a sudden precision drop.
- Level: debugging. Skill: competing hypotheses.
- Candidate prompt: "Precision at capacity fell from 0.44 to 0.31 overnight. What do you check, in what order?"
- What the candidate should clarify: whether the model, the population, or the measurement changed.
- Strong answer: separate the three causes first. Model: artifact or calibration version changed — check the release manifest and the decision log's model_version. Population: prevalence or channel mix shifted — check attempt volume by segment and the fraud rate. Measurement: label maturity or rubric changed — check resolution rate and analyst agreement. The cheapest discriminator is a version diff of the decision log's model and feature manifest, which rules in or out a release in minutes.
- Acceptable alternatives: beginning with a score-distribution comparison is equally efficient, since a shifted score distribution points to features or calibration while an unchanged distribution with worse precision points to labels or population.
- Follow-up: "Score distribution unchanged, precision down, resolution rate unchanged. Next?" Answer: population mix; a shift toward a segment where the model is weaker would leave scores plausible while precision falls, so stratify precision by segment and compare segment weights to the prior week.
- Red flags: retraining immediately, or blaming the model before checking whether the measurement changed.
- Scoring: 0 = single guess; 1 = lists causes without an order; 2 = ordered discriminator with a version check; 3 = also interprets an unchanged score distribution against label and population hypotheses.
- Answer support: Sections 9 and 12 F-01, F-06, F-07.

#### Question 14: Change one constraint — what if review capacity doubled?
- Level: transfer. Skill: re-deriving under changed constraints.
- Candidate prompt: "Assume the organization hires enough analysts for 9,600 reviews. Does the platform case get stronger or weaker?"
- What the candidate should clarify: whether the fraud pool can support the deeper review, and what precision at 9,600 would be.
- Strong answer: weaker per review but larger in absolute terms, and possibly still positive. The fraud pool outside the decline band is 6,280, so reviewing 9,600 means 3,320 reviews are non-fraud by construction; precision at 9,600 is at most 6,280/9,600 = 0.654 and in practice far lower. The baseline also improves at 9,600 because it reviews more of its own band, so the comparison must be recomputed on both sides rather than holding baseline precision fixed.
- Acceptable alternatives: noting that at very large capacity the two policies converge toward reviewing the whole band, so the ranking advantage shrinks toward zero, is an equally strong answer.
- Follow-up: "At what capacity do the policies converge?" Answer: when capacity approaches the entire non-declined fraud pool, both policies prevent essentially all fraud and the difference is dominated by decline-band choice and cost, not ranking.
- Red flags: assuming the proposal's absolute advantage doubles with capacity, or holding the baseline's precision at 0.1833 while changing only the proposal.
- Scoring: 0 = asserts an answer without re-derivation; 1 = notes diminishing returns; 2 = recomputes with the pool ceiling; 3 = also observes convergence and states what then dominates.
- Answer support: Sections 6.4 and 6.5.

**Takeaway:** every question above rewards the same discipline — bind a number to a decision, and state what evidence would change it.

## 15. Quick Reference Card

| Field | Case-specific answer |
|---|---|
| Domain | Payments fraud operations; card-not-present authorization scoring and analyst review |
| Problem type | Capacity-constrained binary ranking plus a platform-investment decision |
| Method | Calibrated gradient-boosted ranker with banded policy, evaluated as precision at 4,800 reviews against rules at identical capacity; ledger priced per prevented case, review, decline, and platform day |
| Primary result | Not measured. Illustrative arithmetic: 2,762 vs 1,600 prevented cases/day, +372,700 currency/day, break-even review precision 0.1974 |
| Key lesson | When a scarce resource selects among candidates, evaluate at that resource's capacity; the model metric is not the decision metric |
| Tags | `evaluation-mismatch`; `cost-latency-tradeoff`, `label-delay` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-03-01-001 / 1.0 / 2026-11-10 |
| Evidence level | Illustrative; every count, rate, price, and latency figure is constructed |
| Implementation stage | Proposed; nothing built, piloted, or procured |
| Baseline / primary metric | Rules engine at 4,800 reviews/day; precision at capacity with a card-clustered interval |
| Key limitation | Precision at capacity of 0.44 is assumed, not estimated; A-11 (review before settlement) can zero the review-channel benefit |
| Review trigger / owner | Changes in capacity, loss per case, prevalence, maturity horizon, or settlement-hold policy / fraud-operations lead |

The card's critical validity condition is that the review budget is genuinely fixed and that reviews complete before settlement; relax either and the entire comparison must be rebuilt rather than rescaled.

