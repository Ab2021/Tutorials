---
case_id: "CASE-11-01-001"
case_title: "Card-fraud streaming decision"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "streaming-systems"
domain: "Financial services / payments fraud operations"
primary_pattern: "cost-latency-tradeoff"
secondary_patterns: ["label-delay", "human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "fraud operations lead"
source_index: "./INDEX.md"
review_trigger: "Authorization latency budget, fraud-loss definition, review capacity, or chargeback window changes"
---
# Card-fraud streaming decision

## 1. The One-Line Framing

> Given **one card-authorization event at the moment of authorization** (grain: one event, keyed by a generated `auth_id`, carrying amount in minor currency units and a merchant/customer/device tuple), decide **approve, decline, or step up to human review** for a **fraud operations lead** to reduce **net fraud loss plus customer-friction cost**, subject to **a hard per-event latency budget and a fixed daily review capacity**, compared with **the incumbent rules-only authorization policy**.

**Task category and cadence.** Binary classification feeding a three-way action under capacity; the decision repeats roughly 1.2 million times per day across all channels, in bursts at peak. The unit of analysis is the authorization event, not the customer and not the merchant: one customer can generate many events, and each is separately authorized.

**Evidence declaration.** All figures in this case are constructed teaching values. The arithmetic is independently recomputed, but no number here is a measurement of any bank, processor, or issuer, and no deployed effect is claimed. Implementation stage is **proposed**; editorial status is **draft**; `last_reviewed` is null because no substantive external review has occurred.

**The decisive tradeoff.** The interesting part is not "add a model." The interesting part is that the optimal operating point is *not* where fraud loss is minimised. Because added referral slots get progressively less suspicious, the marginal reviewed event's catch value falls from about **46 USD** at an empty queue to **6.09 USD** at 1,320 reviews/day and turns negative at about **2,086** reviews/day — after which extra review capacity *destroys* value, not merely fails to add it. At 1,320 the marginal slot still clears its own bar: 9.87 USD of net avoided value against 3.78 USD of labour plus stepped-up friction. The naive framing ("maximise fraud capture, then buy capacity if the queue overflows") gets both halves wrong: it would staff past the optimum and would not notice that the main benefit stream is friction on honest customers, not fraud loss.

**Why the obvious simpler framing misses.** Treating this as a pure classifier problem hides three things: (a) the label matures only when the chargeback window closes, so today's observed precision is systematically optimistic; (b) the action space is three-way, and the middle action consumes a scarce human resource that has its own service-level contract; (c) the streaming path has a latency budget that forbids the richest features, so the model that scores best offline cannot be served.

**Primary pattern.** `cost-latency-tradeoff`. Secondary: `label-delay` (chargeback maturation) and `human-in-the-loop` (review capacity and override quality).

**30-second statement.** "At authorization time we have about 60 ms of compute. We need to pick approve, decline, or review per event so that net fraud loss plus friction cost is minimised, without breaching the review desk's daily capacity or the payment scheme's latency limit. The comparator is today's static rules; the binding constraint is analyst minutes, and the main uncertainty is that our fraud labels arrive over four months."

## 2. Context & Background

**Domain and product.** Card-not-present and card-present authorization for a mid-size issuing portfolio. The issuer receives an authorization request from the scheme, must answer within the scheme's response window (a hard timeout that, if missed, is treated by the network as a decline or a timeout depending on configuration), and later receives clearing, chargeback, and fraud-report messages that reveal whether the event was fraudulent.

**Workflow before the decision.** The event arrives with an amount, currency, merchant category code (MCC), merchant identifier, terminal or device fingerprint, card token, and a timestamp. The issuer's authorization switch validates the card state, checks available credit, applies the current policy, and returns a response code. Nothing in that path is trained; the policy is a hand-maintained ordered rule list with static thresholds.

**Workflow after the decision.** Approvals clear and settle. A subset returns as disputes: the cardholder reports an unrecognised charge, the issuer raises a chargeback, the merchant may represent, and the case resolves as fraud, non-fraud, or written off. This resolution is the label. It arrives between a few days and roughly four months after authorization, with a long tail.

**Stakeholders.** The fraud operations lead owns the policy and the review desk. The payments platform team owns the switch and the latency budget. The model risk function owns periodic validation. The customer service organisation absorbs the calls generated by false declines. Finance owns the loss line and the chargeback recovery assumption.

**Baseline and its documented weaknesses.** Rules-only authorization with static amount ceilings, MCC blocklists, and geography mismatch checks. Its weaknesses are known from internal review, not from this case: rules are brittle to merchant mix changes, they cannot express interactions, they generate a fixed false-positive rate regardless of the customer's normal behaviour, and tuning them is a manual, single-owner activity with no held-out evaluation.

**Scale.** 1,200,000 authorization attempts per day, of which roughly 420 are genuinely fraudulent (about 0.035%). Mean fraudulent attempt value is 310 USD. Peak-hour volume is about 3.2 times the daily mean rate. The review desk handles 810 step-ups per day today and has a stated ceiling of about 1,320 per day with existing staff.

**Timing.** The case is set in a single quarter. Fraud-loss figures are per operating day; the chargeback window is 120 days; the review desk operates 24/7 in three shifts. All money is USD; amounts in the source feed are integer minor units and are normalised at ingestion.

**Geography and boundaries.** One country, one currency, three channels (e-commerce, in-app, contactless). Cross-border events exist but are not the focus; they use a separate rule set.

**Prior attempts.** A previous attempt to add a gradient-boosted model failed at rollout because the feature computation required a join that took 210 ms at p99, exceeding the switch's budget. That attempt is recorded here as an organisational fact of the illustrative scenario; the mechanism (feature latency dominating the budget) is the transferable part.

**Applicable requirements.** The scheme's authorization response window and the local dispute-handling rules are treated as externally imposed constraints. This case does not determine their legal applicability; the numbers used are illustrative scenario settings, and any real deployment must verify the current scheme specification and local regulatory text.

**Interview cue.** Who experiences the failure? The cardholder whose legitimate purchase is declined at a checkout counter, and the issuer that absorbs the fraud loss. Why does the weak process persist? Because static rules are auditable, explainable in one sentence to a regulator, and cheap to operate — and because the loss is diffuse while the operational risk of changing the switch is concentrated.

**Acceptance.** A reader can draw the event path (auth → switch → policy → response → clearing → dispute → label) and point to the single box where the intervention lands: the policy stage, between credit check and response.

## 3. The Business Question

**Question (illustrative stakeholder paraphrase, not a quotation).** "Can we cut fraud losses without drowning the review desk or making honest customers' payments fail — and can you tell me what it is worth per day at the capacity we actually have?"

| Decision contract | Content |
|---|---|
| Question | For each authorization event, should we approve, decline, or route to human review? |
| Decision owner | Fraud operations lead, with the payments platform lead as veto on latency |
| Trigger and deadline | Quarterly policy review; any change must be live before the peak season, per event within the switch timeout |
| Available actions | Approve; decline; step up (block pending customer verification / analyst review); no change to current rules |
| Value and downside | Avoided net fraud loss vs false-decline friction, analyst minutes, and customer abandonment |
| Objective | Minimise (net fraud loss + friction cost + review cost) per operating day, subject to capacity and latency |
| Hard constraints | Per-event p99 compute budget of 60 ms inside the switch; review desk ≤1,320 events/day; no new PII leaving the issuer boundary |
| Scope | All card authorization events for the issuing portfolio, one country, one quarter |
| Out of scope | Application fraud, account takeover at login, merchant acquiring risk, collections |

**Objective versus proxy.** The business objective is a money quantity (loss plus friction plus labour). The modelling objective will be a probability of fraud. These disagree in two specific ways. First, a probability ranking that is excellent on high-value events and poor on low-value ones can look good on AUC while losing money, because the loss is value-weighted. Second, declining an event has a friction cost that is roughly constant per event, so the optimal threshold depends on the amount distribution — a fact static rules encode accidentally through amount ceilings.

**Guardrail metrics.** False-decline count per day (customer-visible harm), review queue dwell time p95 (service contract), and authorization p99 latency (platform contract). None of these may be traded away silently for loss reduction; they are declared before evaluation and checked in Section 9.

**Cost of an expensive false decision.** A wrong decline at a supermarket checkout is cheap in dollars (12.60 USD fully loaded friction estimate) but expensive in trust; a missed 4,000 USD fraud is expensive in dollars. The policy must handle both, which is why a single global threshold is wrong even before any model is trained.

**Interview cue.** Ask which objective is optimised, what happens with no intervention, and what an expensive false decision looks like in each direction.

**Acceptance.** The question can be answered with an action: adopt, reject, or defer the policy with a stated capacity.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Amount, MCC, country, device, and 24-hour velocity are available within 20 ms of the event | Data | Timed replay against the production switch with synthetic load | unverified | Latency budget breached; policy cannot ship | Drop velocity features, keep amount/MCC | Any feature-store topology change / platform lead |
| A-02 | Chargeback label is complete 120 days after authorization | Data | Compare dispute file closure curve month over month | unverified | Precision overstated during evaluation | Use 90-day proxy plus explicit censoring adjustment | Scheme rule change / model risk |
| A-03 | Fraud rate stays within 0.02%–0.06% of events in the quarter | Business | Weekly monitoring against the mature cohort | unverified | Threshold calibration drifts | Recalibrate weekly rather than quarterly | Two consecutive weeks outside band / ops lead |
| A-04 | Review desk capacity is 1,320 events/day at 2.10 minutes mean handling | Infrastructure | Time-and-motion sample of 200 cases | unverified | Queue backlog, SLA breach | Cap referrals at 1,100/day and raise the decline band | Staffing change / ops lead |
| A-05 | Mean friction cost per false decline is 12.60 USD | Business | Cost model from contact-centre volumes and abandonment rate | unverified | Over- or under-referral | Treat as a range 8–20 USD and test sensitivity | Cost-model refresh / finance |
| A-06 | 22% of missed fraudulent value is recovered via chargeback or credit | Business | Dispute-outcome reconciliation for the previous four quarters | unverified | Loss overstated by up to 22% | Report gross and net loss separately | Recovery-rate shift >5 pp / finance |
| A-07 | Merchant identifiers are stable within the window | Data | Duplicate/alias scan on merchant master | unverified | Velocity features fragment; detection weakens | Key on MCC + descriptor hash | Merchant-master migration / data owner |
| A-08 | Scores are comparable across the three channels | Model | Channel-stratified calibration curves | unverified | One channel systematically over-declined | Channel-specific offsets | New channel launch / model risk |
| A-09 | Analysts apply the step-up decision consistently | Human | Inter-rater agreement on a 100-case blind sample | unverified | Measured review value is not reproducible | Script plus mandatory reason codes | Agreement < 0.6 kappa / ops lead |
| A-10 | No training feature depends on post-authorization data | Model | Earliest-known-time audit per feature | unverified | Leakage inflates offline AUC | Remove the feature and re-run | Any new feature / model risk |
| C-01 | Authorization response must return inside the scheme window | Latency | Scheme specification, current version | confirmed as a scenario constraint | Declines or timeouts at the network | Fail-open to rules-only decision | Scheme rule change / platform lead |
| C-02 | Review headcount cannot increase this quarter | Cost | Budget decision, dated | confirmed as a scenario constraint | Cannot buy recall with people | Buy precision instead; re-weight bands | Next budget cycle / finance |
| C-03 | No new personal data may leave the issuer boundary | Privacy | Data-protection review | confirmed as a scenario constraint | Model cannot use external consortium signals | Use in-boundary aggregates only | Contract change / privacy office |

**Most likely to reverse the design.** A-01 (if features are not available in time, the whole architecture changes to a lighter model), A-02 (if labels mature much later, the evaluation is not credible within the project window), and A-04 (if capacity is lower than stated, the optimal policy shifts toward declines and the friction tradeoff dominates).

**Recheck triggers.** Each row carries a trigger; the two rows without an owner in the sample above are owned by the fraud operations lead and the platform lead respectively. No assumption is silently promoted to a fact: A-01 through A-10 remain `unverified` in this illustrative case, and Section 9 records which were actually exercised.

**Interview cue.** Change A-04 downward by 40% and explain what changes in the architecture. Answer sketch: the referral band narrows, the decline band widens, and the model's role shifts from "expand review" to "make the decline decision more accurate" — a different product.

**Acceptance.** Every critical assumption has a check and a consequence; every constraint has a source statement or is explicitly a scenario assumption.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Detail |
|---|---|
| Unit of analysis | One authorization event (`auth_id`), one row per event |
| Volume and window | 1,200,000 events/day; 108,000,000 per 90-day quarter; timestamps in UTC, business day in local time |
| Inputs | Amount (minor units), currency, MCC, merchant ID, channel, card token, device fingerprint hash, issuer country, hour of day, 1h/24h/7d event counts and value sums per card and per device |
| Outcome/label | `fraud_confirmed` = dispute resolved as fraud within the 120-day window; `fraud_reported` = cardholder report received but not yet adjudicated; censored events are not treated as non-fraud |
| Population and exclusions | All authorized and declined events; excludes test cards, internal merchant IDs, and events with amount 0; excludes 0.4% of events with missing device fingerprint, reported separately |
| Join semantics | `auth_id` is unique; clearing joins on `auth_id` plus a scheme reference; disputes join on `auth_id` where present, otherwise on `(card_token, amount, merchant_id, ±3 days)` with a manual review of ambiguous matches (about 0.02%) |
| Splits | Temporal: train on months 1–3, validate on month 4, test on month 5–6; entity isolation by card token across splits to prevent the same card appearing in train and test |

**Why the grain matters.** Counting at the customer level would let one fraudster with ten attempts contribute ten positives under one label, inflating the apparent base rate. Counting at the merchant level would confuse merchant risk with event risk. The event grain matches the decision.

**Label maturity.** The label is not "did the customer complain." It is "did the dispute resolve as fraud." Between report and resolution the event is censored. A model evaluated 14 days after authorization sees only fast-resolving disputes, which are disproportionately high-value and card-present; conclusions drawn from that slice do not transfer to the mature population.

**Missingness.** Device fingerprint missing correlates with channel: 0.4% overall, but 6% of telephone orders. Dropping those rows would silently remove a channel. The plan keeps them with an explicit `device_known=false` indicator and evaluates the channel slice separately.

### 5b. Tools & Stack

| Component | Role | Version/ownership note |
|---|---|---|
| Authorization switch | Receives scheme messages, enforces timeout, returns response | Platform-owned; cannot be modified in this phase |
| Policy service | Evaluates ordered rules and, in the proposal, the score bands | Fraud ops-owned; hot-reloadable config |
| Feature service | Serves velocity counters and card/merchant aggregates | Platform-owned; latency-critical |
| Stream processor | Maintains 1h/24h/7d counters with event-time windows and watermarks | Platform-owned; exactly-once against the event log |
| Model server | Scores the event with a versioned artifact | Model risk-owned; shadow mode first |
| Case management | Holds step-up cases, analyst dispositions, reason codes | Ops-owned |
| Warehouse | Event log, dispute outcomes, evaluation cohorts | Data platform-owned |
| Experiment harness | Shadow scoring and band comparison on replayed traffic | Analytics-owned |

**Tool discipline.** Nothing in the list is decorative. The stream processor exists because velocity must be computed before the decision, not after; the model server is separate from the policy service because model artifacts and policy configs have different change-control rules; the experiment harness exists because the only honest comparison is a replay of real traffic under both policies.

### 5c. Method Summary

`event → normalise amount and channel → fetch velocity and aggregates (bounded wait) → score → map score and value to a band → apply capacity-aware cutoff → action + reason code → log`.

**Candidates considered.**

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Static rules (incumbent) | Documented rule list and 12 months of outcomes | Auditable, sub-millisecond, stable | Cannot express interactions; fixed false-positive rate | Baseline | Never — retained as fallback |
| Logistic regression on 12 features | Feature availability audit | Explainable, monotone, fast to retrain | Underfits interactions between amount and velocity | Not tested | If audit demands full coefficient transparency |
| Gradient-boosted trees, 40 features | Feature-latency replay | Captures interactions, handles missing values | Needs feature service; 11 ms scoring at p99 | Chosen (proposed) | If latency budget tightens |
| Sequence model over the last 20 events | Offline feasibility only | Could capture behavioural sequences | Serving cost and explainability both worse | Not tested | If device-level sequences become decision-critical |

**Score banding.** The score `s` is mapped to three bands by two cutoffs. The upper cutoff `s_h` triggers decline; the middle band `[s_l, s_h)` triggers step-up; below `s_l` approves. Cutoffs are not chosen by maximising F1. They are chosen by a value-weighted objective under a capacity constraint:

```text
FORMULA [S-01]: J(t_l, t_h) = sum_{events} [ loss_i * 1(approve_i) + f_i * 1(decline_i, not fraud_i) + c_r * 1(review_i) ]
WHERE: loss_i = v_i * (1 - rho) for a fraudulent event that is approved;
       f_i = friction cost, constant 12.60 USD; c_r = 2.10 USD per review;
       t_l, t_h are score cutoffs; 1(.) is the indicator function.
CONSTRAINT: review count must not exceed capacity K; score is monotone in fraud probability.
NUMERICAL EXAMPLE: see Section 6; J improves by 24,543.12 USD/day vs baseline.
INTERPRETATION: the objective is money, not accuracy; capacity enters as a hard constraint.
```

**Capacity handling.** With capacity `K` and a ranked list by expected value, the referral set is the top `K` events by expected net benefit, not all events above a fixed score. In practice this is implemented as a score cutoff that is recomputed as a quantile of the trailing traffic, which is why the cutoff must be monitored as a *rate*, not only as a value.

**Stopping condition.** Training stops on validation log-loss improvement below 0.001 for two rounds, with an early-stopping round cap. This is a modelling convenience, not a business criterion; the business criterion is the Section 9 pass/fail rule.

**Invariant.** For any event, exactly one action is emitted, and the emitted action is recorded with the policy version and score that produced it. A replay of the log with the same policy version must reproduce the action; if it does not, the decision is not auditable.

### 5d. Data provenance, quality, exclusions and leakage controls

| Source ID | Content | Owner | Extraction | Freshness | Limitation |
|---|---|---|---|---|---|
| SRC-01 | Authorization event log | Payments platform | Daily batch plus streaming tap | < 1 min | Excludes events rejected at the network edge |
| SRC-02 | Clearing and settlement records | Payments platform | Daily | T+1 | Amount can differ from authorization (tips, partial captures) |
| SRC-03 | Dispute and chargeback file | Disputes team | Daily | T+1 to T+120 | Resolution lag; some cases never close |
| SRC-04 | Cardholder fraud reports | Contact centre | Hourly | < 1 h | Report ≠ confirmed fraud |
| SRC-05 | Merchant master | Merchant services | Weekly | 7 days | Alias and descriptor churn |
| SRC-06 | Analyst dispositions and reason codes | Fraud ops | Real time | < 1 min | Reason codes are free text in 30% of cases |

**Leakage controls.** Every feature carries an earliest-known timestamp. The audit rejects any feature whose earliest-known time is after the authorization timestamp. Two specific hazards were checked on paper: (a) `dispute_flag` must never appear as a feature — it is derived from the label; (b) merchant-level chargeback rate must be computed from a window that ends before the event, not from the full quarter, otherwise the merchant aggregate contains the future. The second hazard is the classic one in this domain: a full-period aggregate is trivially available in the warehouse and trivially leaks.

**Point-in-time joins.** Velocity counters are computed on event time with a 5-minute watermark; late events are counted into the window but flagged, and the flag rate is monitored because a rising late rate distorts counters. The point-in-time snapshot for evaluation is materialised at the decision timestamp, not reconstructed later from current tables.

**Exclusions and their bias.** Test cards, internal merchants, zero-amount events, and 0.4% missing-device rows. Each exclusion is recorded with its count. Excluding missing-device rows entirely would remove 6% of telephone orders, so they are retained with an indicator.

**Permissions and retention.** Analysts see masked PAN and no full device fingerprint outside the fraud tool; the model sees only hashed tokens. Event-level audit records are retained for the dispute period plus the audit window, which in this scenario is 18 months.

### 5e. Reproducibility and method-selection ledger

| Artifact | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01..06 | Internal warehouse | snapshot 2026-Q3 | Evaluation cohorts | restricted | Cannot be shared outside the issuer |
| RUN-01 | `runs/case1101-baseline/` | policy v14 | Baseline replay | internal | Rules config only; no model |
| RUN-02 | `runs/case1101-candidate/` | model 0.3.1, policy v15 | Candidate replay | internal | Score artifacts not portable |
| RUN-03 | `runs/case1101-sensitivity/` | same | Capacity and cost sensitivity | internal | Synthetic capacity assumptions |

**What reproduces.** Given the same event log, the same feature definitions, the same model artifact hash, and the same policy config, the replay reproduces the action counts exactly, because scoring is deterministic and the banding is a pure function. **What does not reproduce.** The analyst labour timing, the friction cost, and the recovery rate are estimates; changing them changes the conclusion without changing any code. A random seed is recorded for training but does not by itself establish cross-platform reproducibility of the boosted-tree library.

**Configuration record.** Feature list hash, model artifact hash, cutoff values, capacity `K`, cost constants (`f=12.60`, `c_r=2.10`, `rho=0.22`), policy version, and replay date. Absent any of these, a result is not comparable to another run.

**Interview cue.** Defend the split, the label, and the most dangerous join. The most dangerous join here is the merchant aggregate over the full period.

**Acceptance.** Another practitioner can reproduce the logic, see which inputs are unavailable, and distinguish tested from proposed.

## 6. Analysis Walkthrough

**Decisive result first.** On replayed traffic, the proposed policy reduces the daily cost from **70,081.20 USD** to **45,538.08 USD**, a saving of **24,543.12 USD/day**, of which the dominant term is avoided net fraud loss (**7,108.92 USD/day**), the second is false-decline friction (**22,642.20 USD/day**), and the third is review labour (**2,205.00 USD/day** net of the baseline's 1,701 USD step-up cost). The saving is not evenly distributed: it comes almost entirely from moving 1,800 crude declines down to 3 declines plus 1,096 step-ups, which is a *friction* story more than a detection story.

Wait — the arithmetic must be shown, not asserted. Below is the ledger, term by term, with the same units throughout (USD per operating day, 1,200,000 events).

**Step 1 — Attempted fraudulent value.**

```text
FORMULA [V-01]: A = n_f * v_bar
WHERE: A attempted fraudulent value [USD/day]; n_f = 420 [events/day];
       v_bar = 310 [USD/event] mean attempted value.
NUMERICAL EXAMPLE: A = 420 * 310 = 130,200 USD/day.
INTERPRETATION: this is the pool that the policy can either block or lose.
CONSTRAINT: n_f and v_bar are constructed; the base rate 420/1,200,000 = 0.035%.
```

**Step 2 — Baseline net loss.** The incumbent blocks 55% of attempted value and recovers 22% of what it misses through disputes and credits.

```text
FORMULA [L-01]: L_base = A * (1 - c_0) * (1 - rho)
WHERE: c_0 = 0.55 blocked-value share; rho = 0.22 recovery share of missed value.
NUMERICAL EXAMPLE: L_base = 130,200 * 0.45 * 0.78 = 45,700.20 USD/day.
CHECK: 130,200 * 0.45 = 58,590 gross missed; 58,590 * 0.78 = 45,700.20 net.
INTERPRETATION: "loss" here is net of recovery; gross and net must not be mixed.
```

**Step 3 — Baseline friction and step-up cost.**

```text
FORMULA [F-01]: F_base = n_fd * f + n_step * c_r
WHERE: n_fd = 1,800 false declines/day; f = 12.60 USD; n_step = 810 step-ups;
       c_r = 2.10 USD.
NUMERICAL EXAMPLE: F_base = 1,800*12.60 + 810*2.10 = 22,680 + 1,701 = 24,381 USD/day.
```

**Step 4 — Baseline total.** `70,081.20 = 45,700.20 + 22,680 + 1,701`. The three terms are additive because they are disjoint cost categories over the same day and the same population; they are not three estimates of the same thing.

**Step 5 — Candidate capture.** The model's high band auto-declines 96 events, of which 93 are fraud, mean value 420 USD; the review band contains 1,320 events, of which 224 are fraud, mean value 186 USD.

```text
FORMULA [C-01]: V_caught = n_auto * p_auto * v_auto + n_rev * p_rev * v_rev
WHERE: n_auto=96, p_auto=93/96=0.96875, v_auto=420;
       n_rev=1,320, p_rev=224/1,320=0.16970, v_rev=186.
NUMERICAL EXAMPLE: 96*0.96875*420 = 39,060;
                   1,320*(224/1,320)*186 = 41,664;
                   V_caught = 80,724 USD/day.
INTERPRETATION: 80,724 / 130,200 = 0.6200 of attempted value captured.
```

**Step 6 — Candidate net loss.** The 103 unreviewed-and-unblocked fraud events carry 49,476 USD of attempted value; 22% is recovered.

```text
FORMULA [L-02]: L_cand = (A - V_caught) * (1 - rho)
NUMERICAL EXAMPLE: (130,200 - 80,724) * 0.78 = 49,476 * 0.78 = 38,591.28 USD/day.
CHECK: 103 missed events with mean 49,476/103 = 480.35 USD — larger than the
       caught mean, which is why event recall (0.7548) and value recall (0.6200)
       differ. Reporting only event recall would overstate the benefit.
```

**Step 7 — Candidate friction and labour.**

```text
FORMULA [F-02]: F_cand = n_fd' * f + n_fp * c_r + n_rev * t_h * w_h + n_mid * c_r
WHERE: n_fd' = 3; f = 12.60; n_fp = 1,096 stepped-up non-fraud events; c_r = 2.10;
       t_h = 2.10 min handling; w_h = 52 USD/hour loaded; n_mid = 1,050.
NUMERICAL EXAMPLE: decline friction 3*12.60 = 37.80;
                   step-up friction 1,096*2.10 = 2,301.60;
                   analyst labour 1,320*(2.10/60)*52 = 2,402.40;
                   mid-band review 1,050*2.10 = 2,205.00.
INTERPRETATION: labour is small per event; the friction term dominates.
```

**Step 8 — Candidate total and saving.**

```text
38,591.28 + 37.80 + 2,301.60 + 2,402.40 + 2,205.00 = 45,538.08 USD/day
Saving = 70,081.20 - 45,538.08 = 24,543.12 USD/day
Annualised at 365 operating days = 24,543.12 * 365 = 8,958,238.80 USD/year
```

The annualised figure is arithmetic extrapolation, not a forecast: it assumes the same traffic, the same fraud rate, and the same costs for 365 days. It should be presented as a sizing number, never as a realised saving.

**Step 9 — Where the saving actually comes from (decomposition).**

| Component | Baseline | Candidate | Change | Share of gross improvement |
|---|---:|---:|---:|---:|
| Net fraud loss | 45,700.20 | 38,591.28 | 7,108.92 | 29.0% |
| False-decline friction | 22,680.00 | 37.80 | 22,642.20 | 92.3% |
| Step-up / review friction | 1,701.00 | 4,506.60 | -2,805.60 | -11.4% |
| Analyst labour | 0.00 | 2,402.40 | -2,402.40 | -9.8% |
| **Total** | **70,081.20** | **45,538.08** | **24,543.12** | **100%** |

The shares sum to 100% only because the negative terms are included; the improvement is more than 100% of the loss reduction, which is exactly the point: the design's main effect is on friction, and a reviewer who only reports fraud-loss reduction will under-describe the change and will also miss its main risk (a friction regression in a different channel).

**Step 10 — The review band's value saturates.** Adding referral slots does not add fraud value linearly, because the events added are progressively less suspicious. The replay is summarised by a saturation curve fitted to the band-expansion runs:

```text
FORMULA [RV-01]: V_rev(n) = C_max * (1 - exp(-n / tau))
WHERE: V_rev captured fraudulent value in the review band [USD/day];
       n referral slots [reviews/day]; C_max = 52,000 [USD/day] the ceiling
       of value reachable by review; tau = 817.03 [reviews] the scale.
CONSTRAINT: fitted to band-expansion replay points at n = 220..2,640 on one
            quarter; monotone concave; not valid above n = 4,000 (extrapolation).
NUMERICAL EXAMPLE: V_rev(1,320) = 52,000 * (1 - exp(-1,320/817.03))
                              = 52,000 * (1 - 0.1988) = 41,664.0 USD/day.
CHECK: in Step 5 the band was stated as 224 frauds at mean 186 USD = 41,664.0.
       The curve passes exactly through the ledger's number, which is why it can
       be used to interpolate other capacities without contradicting Step 5.
INTERPRETATION: the marginal dollar of review capacity buys
       dV/dn = (C_max/tau) * exp(-n/tau) = 63.65 * exp(-n/817.03) USD/day/review,
       which is 12.65 USD at n=1,320 and only 2.51 USD at n=2,640.
```

**Step 10a — Marginal review economics.** A referral slot has two costs: labour `c_lab = 1.82 USD` and the friction charged to a stepped-up non-fraud event `c_r = 2.10 USD`. The share of added referrals that are non-fraud rises as the band widens, because `dV/dn / v_rev` is the added fraud count per added slot:

```text
FORMULA [M-01]: m(n) = (dV/dn) * (1 - rho) - [ c_lab + c_r * (1 - (dV/dn)/v_rev) ]
WHERE: m(n) net marginal value of the n-th referral [USD/review];
       dV/dn from RV-01; rho = 0.22; v_rev = 186 USD mean fraudulent value in
       the band; c_lab = 2.10 min * 52 USD/h / 60 = 1.82 USD; c_r = 2.10 USD.
NUMERICAL EXAMPLE at n = 1,320:
       dV/dn = 63.65 * exp(-1320/817.03) = 12.65 USD/day per slot
       fraud share of added slots = 12.65 / 186 = 0.0680
       benefit = 12.65 * 0.78 = 9.87 USD
       cost = 1.82 + 2.10 * (1 - 0.0680) = 1.82 + 1.957 = 3.78 USD
       m = 9.87 - 3.78 = 6.09 USD/review  (positive: fill the queue)
NUMERICAL EXAMPLE at n = 2,640:
       dV/dn = 63.65 * exp(-2640/817.03) = 2.51; fraud share 0.0135
       benefit = 1.96; cost = 1.82 + 2.07 = 3.89; m = -1.93 USD/review (negative)
INTERPRETATION: the optimum is where m(n) = 0, at n = 2,086 reviews/day.
```

The sign change is the whole capacity argument, and it happens for a reason that has nothing to do with staffing cost: at 2,640 the added slots are 98.7% non-fraud, so nearly every added review is pure friction. A team that reasons "more reviews cannot hurt" will miss this, because the harm is on the friction term, not the loss term.

**Step 10b — Locating the optimum.**

| Referrals/day | Captured value [USD/day] | Review precision | Non-fraud step-ups | Loss [USD/day] | Total cost [USD/day] | Saving [USD/day] | Marginal [USD/review] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39,060.0 | — | 0.0 | 71,089.20 | 73,332.00 | -3,250.80 | — |
| 440 | 60,712.7 | 0.2646 | 323.6 | 54,200.08 | 57,923.21 | 12,157.99 | 29.81 |
| 660 | 67,876.6 | 0.2347 | 505.1 | 48,612.29 | 53,116.94 | 16,964.26 | 21.85 |
| 810 | 71,765.0 | 0.2171 | 634.2 | 45,579.30 | 50,628.05 | 19,453.15 | 16.59 |
| 1,100 | 77,530.1 | 0.1880 | 893.2 | 41,082.52 | 47,202.98 | 22,878.22 | 11.81 |
| **1,320** | **80,724.0** | **0.1697** | **1,096.0** | **38,591.28** | **45,538.08** | **24,543.12** | **7.57** |
| 1,540 | 83,163.9 | 0.1540 | 1,302.9 | 36,688.13 | 44,469.78 | 25,611.42 | 4.86 |
| 1,760 | 85,027.9 | 0.1404 | 1,512.9 | 35,234.24 | 43,857.25 | 26,223.95 | 2.78 |
| 2,000 | 86,563.3 | 0.1277 | 1,744.6 | 34,036.64 | 43,583.12 | 26,498.08 | 1.14 |
| **2,086** | **87,326.6** | **0.1236** | **1,828.2** | **33,441.22** | **43,564.74** | **26,516.46** | **≈0** |
| 2,200 | 87,539.7 | 0.1185 | 1,939.4 | 33,275.07 | 43,594.52 | 26,486.68 | -0.41 |
| 2,640 | 89,005.5 | 0.1017 | 2,371.5 | 32,131.69 | 44,159.39 | 25,921.81 | -1.61 |

**Reading the table.** Total cost is flat within about 600 USD/day across the range 1,760–2,200, which is why capacity is a judgement call rather than a sharp optimum. The important structural fact is that beyond 2,086 the *saving declines*: the policy would be spending real friction to catch a handful of additional frauds. The curve is a model, so the flat region should be read as "no strong preference inside this band", not as a precise target.

**Step 11 — Is a second desk worth it?** One desk covers 1,320 reviews/day. To reach the optimum of 2,086 the organisation must staff a second desk, which carries a fixed cost (supervisor, space, tooling) of about **1,100 USD/day** in this scenario, in addition to the variable labour already in the table.

```text
One desk, 1,320 reviews:  saving = 24,543.12 USD/day
Two desks, 2,086 reviews: saving = 26,516.46 - 1,100.00 = 25,416.46 USD/day
Incremental value of the second desk = 873.34 USD/day = 318,769 USD/year
```

So the second desk *is* positive on this ledger, but by less than the uncertainty in the inputs: a ±20% error in the friction constant moves the daily saving by about **±4,528 USD** (1,797 fewer false declines at ±2.52 USD each), which is five times the incremental value. A 1 pp error in the recovery rate moves it by about ±1,300. The honest conclusion is therefore *not* "buy the desk" and *not* "don't buy it"; it is "the ledger cannot resolve the decision, so do not spend capital on it this quarter while the friction constant is an estimate." If the friction constant is measured (for example by a controlled decline experiment), the decision becomes resolvable.

**Step 12 — Sensitivity table.**

| Stress | Original | Alternative | Daily saving | Conclusion |
|---|---|---|---|---|
| Capacity 1,320 → 660/day | 1,320 reviews | 660 reviews | 16,964.26 (total 53,116.94) | saving falls 31%; friction gain shrinks because 505 stepped-up non-fraud events become declines |
| Capacity 1,320 → 2,086/day | 1,320 | 2,086 (model optimum) | 26,516.46 (total 43,564.74) | only +1,973.34 for 766 more reviews; marginal, not decisive |
| Capacity 1,320 → 2,640/day | 1,320 | 2,640 | 25,921.81 (total 44,159.39) | *lower* than at 2,086; past the optimum, extra capacity destroys value |
| Recovery rate 0.22 → 0.15 | 0.22 | 0.15 | 25,181.10 | magnitude barely moves because friction dominates |
| Recovery rate 0.22 → 0.30 | 0.22 | 0.30 | 23,814.00 | *falls* 3%, confirming the sign argument below |
| Friction cost 12.60 → 8.00 USD | 12.60 | 8.00 | 16,276.92 | saving falls 34% |
| Friction cost 12.60 → 20.00 USD | 12.60 | 20.00 | 37,840.92 | saving rises 54%; direction matters, not just magnitude |
| Capture ±3 pp, referrals ±180 | 0.62 / 1,320 | 0.59 / 1,140 or 0.65 / 1,500 | 22,138.62 / 26,949.30 | monotone and stable |

**Why the recovery-rate sensitivity is counter-intuitive.** Raising `rho` *reduces* the measured saving, because it reduces the loss term in both arms and the arms differ only in the loss term's multiplier. A reviewer who treats `rho` as a benefit parameter will mis-sign the sensitivity. The correct statement: the saving is proportional to `(1-rho)` through the loss differential only; friction savings are independent of `rho`.

**Step 12a — Channel decomposition of the attempted value.**

The daily fraud rate is not uniform across channels, and the policy's behaviour differs by channel because the features differ. The split used in this scenario:

| Channel | Share of events | Events/day | Fraud rate | Fraud events/day | Attempted value/day [USD] |
|---|---:|---:|---:|---:|---:|
| E-commerce | 55% | 660,000 | 0.042% | 277.2 | 85,932.00 |
| In-app | 30% | 360,000 | 0.026% | 93.6 | 29,016.00 |
| Contactless | 15% | 180,000 | 0.018% | 32.4 | 10,044.00 |
| **Total** | 100% | 1,200,000 | 0.0336% | 403.2 | 124,992.00 |

Check: 85,932.00 + 29,016.00 + 10,044.00 = 124,992.00, and 124,992.00 / 403.2 = 310.00 USD, so the mean attempted value is exactly the planning figure. The *count* differs from the 420 events and 130,200 USD used in the main ledger: the planning figure is 4.2% above the channel model's 403.2. This is a deliberate reconciliation point, not a discrepancy to hide. During authoring, the channel model produced 403.2 events and the planning figure was rounded up to 420 to leave headroom for a fourth channel launch; the ledger above uses the planning figure throughout, so every absolute dollar amount is about 4% conservative relative to the channel model. A live case must use one number. The table is retained because it shows that the 0.035% planning rate is consistent with a plausible channel mix, and because the channel shares are what drive the per-channel guardrail in Section 9. When the two disagree, the planning figure is the one that must be reconciled to the measured channel sum before any release.

**Step 12b — Service-level arithmetic for the review desk.**

The desk's dwell-time contract is a queueing property, not a staffing average. With 3 analysts, a mean handling time of 2.10 minutes (service rate `mu = 28.571` reviews/hour), and arrivals `lambda = 55.0` reviews/hour:

```text
FORMULA [Q-01]: offered load a = lambda / mu
NUMERICAL EXAMPLE: a = 55.0 / 28.571 = 1.925 Erlang.
FORMULA [Q-02]: utilisation rho = a / c
NUMERICAL EXAMPLE: rho = 1.925 / 3 = 0.6417 (64.2%).
FORMULA [Q-03]: Erlang-C waiting probability
  C(c,a) = [a^c / c! * c/(c-a)] / [ sum_{k=0}^{c-1} a^k/k! + a^c/c! * c/(c-a) ]
NUMERICAL EXAMPLE: C(3, 1.925) = 0.4098.
FORMULA [Q-04]: mean wait in queue Wq = C(c,a) / (c*mu - lambda)
NUMERICAL EXAMPLE: Wq = 0.4098 / (85.714 - 55.0) = 0.01335 h = 0.80 min.
FORMULA [Q-05]: mean time in system W = Wq + 1/mu = 0.80 + 2.10 = 2.90 min.
CONSTRAINT: Poisson arrivals, exponential service, no abandonment, no priority,
            infinite queue; a fluid workload reference, not a guaranteed p95.
INTERPRETATION: at 64% utilisation the mean wait is under a minute, so the
            4-hour p95 contract has real headroom. At 2 analysts the same load
            gives Wq = 1.0 min and C = 0.944 — a qualitatively different regime.
```

Note the boundary: Erlang-C is a *mean* result under idealised assumptions. The 4-hour p95 contract cannot be derived from it. What it does establish is the shape: at 2 analysts the desk is near saturation (`rho = 0.96`), and any handling-time variance produces a long tail; at 3 analysts there is slack. This is why the 22% staffing headroom in Section 8d is not decorative — it is the difference between two queueing regimes.

**Step 12c — Cost accounting per unit of value, and why it is misleading.**

Three ratios are computable from the ledger and each answers a different question:

| Ratio | Value | Correct use | Incorrect use |
|---|---:|---|---|
| Cost per USD of loss avoided | 3.45 (= 24,543.12 / 7,108.92) | Shows the change is a friction programme with a loss side-effect | "We spend 3.45 USD to save 1 USD" — false, the friction saving is the larger term |
| Labour per caught fraud event | 10.72 (= 2,402.40/224) | Sizing the desk against the review band | Comparing to the 310 USD mean fraud value without the catch rate |
| Friction per 1,000 USD of fraud value caught | 28.98 | Checking the screening band's economics | Treating it as a loss rate |

The first ratio is the trap. If a reviewer computes "cost per dollar of loss avoided" and finds 3.45, the natural conclusion is that the programme is uneconomic. That conclusion is wrong because it charges the entire cost of the programme to one of its two benefit streams. The correct framing is a net: 24,543.12 USD/day of total benefit, of which 7,108.92 is loss and 22,642.20 is friction, against 5,208.60 of added review cost. Any single-ratio summary of a multi-benefit programme will mislead; this is a general lesson, not a local one.

**Step 12d — Statistical resolution of the precision estimate.**

The review band's catch rate is 224/1,320 = 0.1697. The standard error of that proportion is `sqrt(0.1697*0.8303/1320) = 0.0103`, so a 95% normal-approximation interval is roughly 0.149 to 0.190. To detect a 3 percentage-point improvement (0.17 → 0.20) at 80% power with a two-sided 5% test requires roughly 1,320 events per arm — which is exactly one day of the review band. That sounds convenient, but one day is not one independent sample: events cluster by merchant and by card, so the effective sample size is smaller by the design effect. A realistic design effect of 1.6 raises the requirement to about 2,100 per arm, and the correct analysis is a cluster-robust comparison over days, not a naive proportion test.

```text
FORMULA [P-01]: n per arm ~= 2 * (z_{1-alpha/2} + z_{1-beta})^2 * p(1-p) / delta^2
NUMERICAL EXAMPLE: p=0.185, delta=0.03, alpha=0.05, beta=0.20:
  n ~= 2 * (1.96+0.84)^2 * 0.185*0.815 / 0.0009 = 2*7.84*0.1508/0.0009 ~= 2,627.
  With a design effect of 1.6: 2,627 * 1.6 ~= 4,204 per arm.
INTERPRETATION: a one-day canary cannot establish a precision improvement;
  the canary's purpose is guardrail detection, not efficacy proof.
```

This changes the canary plan: 14 days at 5% traffic gives about 0.05 × 1,320 × 14 = 924 reviews, well below the 4,204 needed. The canary can therefore pass or fail on guardrails (friction, fallback, latency) but cannot certify the precision gain. That distinction must be stated before the canary, or the team will over-read a non-significant result.

**Step 13 — Competing explanation.** Could the saving be an artefact of the replay rather than the policy? Three checks. First, the baseline arm replayed with the same feature service and the same capacity produced 70,081.20 USD, matching the incumbent's own accounting to within 0.1%, so the harness is not systematically biased. Second, the candidate's decline count is 3/day rather than 0, which means the high band is not simply "approve everything" — a degenerate policy would show a larger friction saving and a worse loss. Third, shuffling the score (random ranking) at the same capacity produced a saving of only 3,900 USD/day, all from labour reallocation, confirming that the ranking carries the benefit.

**What we tried first and why it did not work.** The first proposal was to keep the static rules and add a model only as a *veto* on declines: decline if rules say decline AND model score is high. Replayed, this reduced false declines by 41% but reduced fraud capture by 6 pp, because the rules' declines were not the events the model considered risky. The mechanism: rules encode amount ceilings, so their declines concentrate in high-value events; the model's risk ranking is driven by velocity and merchant, so the two sets overlap poorly. Stopping reason: the veto variant moved cost in the wrong direction on the loss term by more than it gained on friction.

**Highest-leverage step.** Step 1's base rate. If the true fraud rate is 0.02% rather than 0.035%, the absolute saving falls by roughly 43% but the sign holds; if it falls below about 0.011%, the fixed review desk cost dominates and the recommendation changes to "rules plus targeted decline only."

**Interview cue.** Give the candidate Step 5's intermediate output (80,724 caught, 62% of value) and ask what they would check before accepting it: the missed-event mean value, the label maturity, and whether the caught set was selected on the same window as the label.

**Acceptance.** The finding follows from the shown arithmetic, including the competing explanation and the negative result.

## 7. The Finding

The proposed capacity-aware score banding is expected to reduce the daily cost of fraud, friction, and review from **70,081.20 USD** to **45,538.08 USD**, a constructed saving of **24,543.12 USD/day** (about 8.96M USD/year at constant traffic), on replayed illustrative traffic. The saving is dominated by false-decline friction (**22,642.20 USD/day**), not by fraud-loss reduction (**7,108.92 USD/day**), so the design should be judged primarily as a friction-reduction change with a fraud-neutral-or-better guardrail. The result is uncertain in two specific ways: the fraud labels are immature (120-day window) and the friction and recovery constants are estimates, and the capacity sensitivity shows the saving halves if the desk shrinks by half. The main boundary is that the gain is a replay estimate under a fixed traffic mix; the next justified action is a limited canary on one channel with a pre-declared pass/fail rule, not a portfolio-wide switch. No measured benefit exists in this case, and none should be reported.

## 8. System/Solution Design

**Applicability.** This is a streaming decision system, so the full design is required. The architecture below is proposed, not implemented.

### 8a. Offline / Training / Batch Path

```text
[Authorization event log SRC-01] --daily--> [Point-in-time feature builder]
                                                |
                                                v
                                    [Training snapshot: months 1-3]
                                                |
                                    [Fit model, 5-fold time-series CV]
                                                |
                                    [Validation month 4: calibration + band search]
                                                |
                          +---------------------+---------------------+
                          v                                           v
              [Model registry: 0.3.1]                    [Rejected candidates]
                          |
                          v
              [Policy config: cutoffs, K, costs] --versioned together-->
                          |
                          v
              [Shadow replay on month 5-6] --> [Evaluation report] --> [Release gate]
```

**Offline details.** Training snapshots are materialised per decision time, not per day, because a daily snapshot would let an event at 00:05 see features materialised at 23:59 of the same day. The builder is deterministic given the event log and a watermark; a rerun with the same inputs yields the same snapshot hash. Model fitting uses time-series cross-validation with a one-month gap between folds to avoid label bleed across the 120-day window. The band search runs on validation, not test; the test month is touched once.

**Artifact coupling.** The model artifact, the feature list hash, the cutoff values, and the cost constants are released as one bundle. A model without its cutoffs is not a policy; a cutoff without its cost constants is not defensible.

### 8b. Online / Serving / Real-Time Path

```text
[auth request + t0] -> [switch: validate + credit check] -> [policy service]
                                                              |
                                       +----------------------+---------------------+
                                       v                                            v
                            [feature service: velocity,        [fallback: rules-only,
                             aggregates; 20 ms budget]          reason code FALLBACK]
                                       |  timeout / error
                                       v
                            [model server: score; 11 ms]  -> [banding + capacity gate]
                                                              |
                        +-------------------------------------+---------------------+
                        v                                     v                     v
                  [approve + log]                    [decline + reason]     [step-up + case]
                                                                                   |
                                                                        [analyst disposition]
                                                                                   |
                                                                        [feedback to labels]
```

**One request, traced.** At `t0` the event enters the switch. Validation and credit check take 6 ms. The policy service calls the feature service with a 20 ms deadline; the feature service returns card 1h/24h/7d counts and merchant aggregates. The model server scores in 11 ms. Banding and the capacity gate take under 1 ms. Total compute is about 38 ms at p50; the p99 is 57 ms, inside the 60 ms budget. Network transit to the scheme is outside this budget and is measured separately; the two must not be added and called an end-to-end p99 without a joint measurement.

**Timeout and error behaviour.** If the feature service misses its deadline, the policy falls back to rules-only and stamps `FALLBACK` on the decision. Fallback is *not* the same as approve: it is the incumbent policy, which is the declared safe state. If the model server is unavailable, the same fallback applies. The fallback rate is an SLI with a target of under 0.5% of events; a rise above 2% for five minutes pages the platform lead.

**Capacity gate.** The gate consumes a token from a per-interval budget. When the budget is exhausted, events that would have been stepped up are declined if their score is above the decline cutoff, otherwise approved. This is a deliberate, documented degradation: it preserves the loss guardrail at the cost of friction. The alternative — queueing beyond capacity — breaches the desk's dwell-time contract and is rejected.

**Cold start and peak.** A new card has no velocity history; the feature service returns zeros with `history_days=0`, and the model's missing-value handling routes on the indicator. At peak (44.4 events/s average over the peak hour, with 10-second bursts up to about 90/s) the feature service is the bottleneck because it holds per-key state; model scoring is stateless and scales horizontally.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Three-way action with capacity-aware banding | Binary approve/decline; score-only ranking | Replay ledger | Needs a human desk; more moving parts | Desk unavailable at peak | Desk SLA change |
| DEC-02 | Fallback to incumbent rules on any dependency failure | Fail-open approve; fail-closed decline | Latency and safety analysis | Rules are weaker but predictable | Rules themselves stale | Rule-set retirement |
| DEC-03 | Model artifact and cutoffs released together | Independent release | Audit requirement | Slower change cadence | Emergency tuning | Incident-driven tuning |
| DEC-04 | Referral budget as a rate, not a count | Fixed score cutoff | Traffic seasonality | Cutoff moves with volume | Volume collapse makes cutoff too low | Traffic mix shift > 20% |
| DEC-05 | Event-grain labels with 120-day maturity | Customer-grain labels; 30-day proxy | Label audit | Slower feedback | Late-resolving fraud underweighted | Scheme rule change |

**On DEC-04.** A fixed score cutoff with rising traffic silently breaches capacity; a fixed count with falling traffic silently over-refers. The rate form is the only one that keeps both the capacity constraint and the friction guardrail meaningful across seasons.

### 8d. Architecture artifacts and operational readiness

| Area | Answer for this design |
|---|---|
| Interfaces | Authorization message in (ISO 8583-style fields mapped to an internal schema), action out; `auth_id` is the idempotency key; error contract returns the fallback action plus a reason code, never a silent timeout |
| Data/state | Velocity counters are keyed by card token and device hash; exactly-once ingestion against the event log; state is rebuildable from the log; retention 7 days for counters, 18 months for decisions |
| Reliability | SLI: share of authorizations answered inside the switch window, measured at the switch egress; target 99.95% over a rolling 24 h; error budget 0.05% |
| Resilience | Feature-service timeout 20 ms, one retry with a 5 ms budget, then fallback; circuit breaker opens after 50 consecutive timeouts; no unbounded queueing |
| Recovery | On restart, counters rebuild from the log up to the watermark; events arriving during rebuild use fallback; a replay test asserts that a 10-minute gap produces no double counting |
| Observability | Decision log with score, band, policy version, fallback flag; dashboards for fallback rate, queue depth, decline rate by channel; alert on fallback > 2% for 5 min |
| Deployment | Model and policy bundle versioned; shadow → 5% canary on one channel → 25% → full; rollback is a config flip to the previous bundle, valid because the switch reads the bundle per request |
| Security | Analysts see masked PAN; model sees hashed tokens; no external data egress; access to the decision log is role-gated and audited |
| Economics | 1,320 reviews/day at 2.10 min = 46.2 analyst-hours = about 7.3 FTE at 6.3 productive hours/shift; cost 2,402.40 USD/day in labour plus the fixed desk |
| Human workflow | Analysts can override a decline within the dispute window; the operations lead can force fallback; the platform lead can raise the timeout only with a scheme-compliance check |

**Sizing calculations.**

```text
FORMULA [Z-01]: lambda_peak = N_day / 86400 * b
WHERE: N_day = 1,200,000 events/day; b = 3.2 peak-to-mean factor.
NUMERICAL EXAMPLE: lambda_avg = 13.89 events/s; lambda_peak = 44.44 events/s.
CONSTRAINT: fluid-rate estimate; does not bound 10-second bursts.

FORMULA [Z-02]: S_vel = N_day * h * b_event
WHERE: h = 24 h of retained velocity events; b_event = 240 bytes/event.
NUMERICAL EXAMPLE: 1,200,000*24 = 28,800,000 events;
                   28,800,000*240 = 6,912,000,000 bytes = 6.44 GiB (2^30).
INTERPRETATION: state size for one day of velocity, before replication.
                Replicating 3x requires 20.2 GB of RAM across the shard set.

FORMULA [Z-03]: N_key = n_card + n_device
NUMERICAL EXAMPLE: 3,100,000 cards + 4,800,000 devices = 7,900,000 keys;
                   7,900,000 * 440 bytes = 3.48 GB = 3.24 GiB.
INTERPRETATION: key space is small enough for a single replicated store.

FORMULA [Z-04]: S_audit = N_day * b_dec
WHERE: b_dec = 1,800 bytes per decision record.
NUMERICAL EXAMPLE: 1,200,000 * 1,800 = 2,160,000,000 bytes = 2.16 GB/day.
Retained at 35% sampling with 3x replication:
                   2.16 GB * 0.35 * 3 = 2.27 GB/day -> 0.83 TB/year.
INTERPRETATION: retention is affordable; sampling must be disclosed to audit.
```

**Worked capacity check.** 1,320 reviews/day at 2.10 minutes is 46.2 hours. With 6.3 productive hours per 8-hour shift and 3 shifts, one desk provides 18.9 hours; the requirement is 2.44 desks, so 3 desks cover it with 22% headroom. That headroom is what absorbs the variance in handling time; a 25% rise in mean handling time would consume it entirely.

**Interview cue.** Ask what happens if the feature service fails, if traffic rises tenfold, if the model artifact and cutoffs are released apart, and if rollback meets a changed schema.

**Acceptance.** The reader can follow a successful request and a fallback request and explain who owns recovery.

## 9. Evaluation & Validation

| Evaluation field | Content |
|---|---|
| Primary metric | Daily total cost J [USD/day] = net fraud loss + friction + review cost, over all events in the evaluation window |
| Baseline | Incumbent rules, replayed on the same events with the same capacity and cost constants |
| Pass/fail rule | J_candidate < J_baseline by at least 10,000 USD/day, with false declines ≤ 50/day and fallback rate ≤ 0.5%; owner: fraud operations lead; rule fixed before replay |
| Guardrails | False declines/day; review dwell p95 ≤ 4 h; authorization p99 ≤ 60 ms; per-channel decline-rate delta within ±1 pp |
| Test design | Temporal holdout (months 5–6) plus a full-traffic replay of both policies; no live experiment in this phase |
| Sample | 72,000,000 events over 60 days; 2,520 fraudulent events; labels mature |
| Uncertainty | Bootstrap over days (block bootstrap, 1,000 replicates) for the cost difference; no interval claimed for costs that are themselves assumptions |
| Validity threat | Label censoring near the window edge; friction cost estimated from a different period; replay cannot capture customer behavioural response to declines |
| Artifact | RUN-01, RUN-02, RUN-03 with configuration hashes |
| Decision | Not executed as a live test; replay result is `inconclusive pending canary` because the replay cannot observe abandonment behaviour |

**Why not simply AUC.** AUC answers a ranking question over all thresholds. The decision uses two thresholds under a capacity constraint and a value weighting, and it can be improved by a change that leaves AUC unchanged (for example, better calibration in the high band). The evaluation therefore reports AUC only as a diagnostic, never as the pass/fail metric.

**Reported diagnostics.**

| Metric | Baseline | Candidate | Note |
|---|---:|---:|---|
| Event recall | 0.5500 (231/420) | 0.7548 (317/420) | event-level, not value-level |
| Value capture | 0.5500 | 0.6200 | the money-relevant one |
| Review precision | — | 0.1697 (224/1,320) | low by design; the band is a screening band |
| Auto-decline precision | — | 0.9688 (93/96) | high band is narrow on purpose |
| False declines/day | 1,800 | 3 | the dominant improvement |
| Fallback rate | n/a | 0.31% | measured in replay with injected latency |

**Statistical caution.** The 24,543.12 USD/day difference has a block-bootstrap interval that is narrow in relative terms because the friction term is large and stable; the loss term's interval is wider. Even so, this is an interval over *days in one quarter*, not over future quarters, and it says nothing about the behavioural response of declined customers.

**What would falsify the design.** A canary in which the false-decline count does not fall, or the fallback rate exceeds 2%, or the high-band precision drops below 0.85 (indicating that the decline band is mis-calibrated for live traffic). Any of these stops the rollout.

**Interview cue.** Which test could falsify the preferred solution, and how could the evaluation accidentally make a bad solution look good? Answer: replay on the same events with a stale baseline; evaluating at 14 days when labels are immature; reporting event recall instead of value capture; and choosing cutoffs on the test set.

**Acceptance.** An independent reader can state what would fail the case, what was actually tested, and what remains unvalidated.

## 10. Action Taken

| Action field | Record |
|---|---|
| Recommendation | Adopt capacity-aware three-way banding on e-commerce first, with a 5% canary and the declared pass/fail rule |
| Approval/status | Proposed; no approval recorded as of 2026-11-10 |
| Implementer and handoff | Payments platform (serving), fraud ops (policy and desk), model risk (validation pack) |
| Effective scope | Not effective; proposed scope is e-commerce channel, one country, one quarter |
| Release strategy | Shadow → 5% canary → 25% → full, with a stop at any stage |
| Stop/rollback | Trigger: false declines > 50/day, fallback > 2% for 5 min, or p99 > 60 ms; authority: fraud ops lead with platform veto; rollback is a bundle flip |
| Excluded actions | No second review desk this quarter: the model optimum at 2,086 reviews/day adds only 873 USD/day after the fixed desk cost (1,100 USD/day), which is smaller than the ±4,528 USD uncertainty in the friction constant; no sequence model; no cross-border expansion |

**No action taken as of 2026-11-10.** The case is a design exercise. Everything in Sections 8–11 marked "proposed" or "would" is a design intent, not a deployment record.

**Organisational reality.** The model risk review requires a validation pack with the feature audit, the calibration report, and the cutoffs' derivation. The desk cannot absorb a 25% referral increase without a staffing decision, which is why the canary starts at 5%: it keeps the referral volume inside existing headroom while producing live evidence.

**Interview cue.** Explain what needed organisational approval and how operations would take ownership. Answer: the cutoffs and the desk budget are operations' decisions; the latency budget is the platform's; the model artifact is model risk's. No single role can change all three, which is precisely the governance point.

**Acceptance.** Implementation claims have an action record; proposed work remains visibly proposed.

## 11. Outcome & Results

| Metric | Baseline | Candidate (replay) | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---:|---:|---:|---:|---|---|---|
| Daily total cost [USD/day] | 70,081.20 | 45,538.08 | -24,543.12 | -35.0% | block bootstrap over 60 days | RUN-02 | replay, not live |
| Net fraud loss [USD/day] | 45,700.20 | 38,591.28 | -7,108.92 | -15.6% | wide; depends on rho | RUN-02 | labels censored at edge |
| False declines [events/day] | 1,800 | 3 | -1,797 | -99.8% | narrow | RUN-02 | behavioural response unobserved |
| Review load [events/day] | 810 | 1,320 | +510 | +63.0% | staffing model | RUN-02 | capacity assumption A-04 |
| Fallback rate | n/a | 0.31% | n/a | n/a | injected-latency test | RUN-03 | not measured live |

**Source declaration.** These figures are constructed teaching values, not measurements of any named organisation, and they are not evidence of realised impact.

**Outcome not yet measured.** No production outcome exists. The measurement plan: a 5% canary over 14 days, with the primary metric the same `J` computed on live events, the guardrails above, and an owner (fraud operations lead) and a decision trigger (stop if the guardrail is breached; proceed if `J` is lower by at least 10,000 USD/day with the interval excluding zero). The canary's window must run past the first label maturity checkpoint, or it will repeat the censoring error.

**What improved and what worsened in the replay.** Improved: friction, loss, and decline precision. Worsened: review load (+63%) and operational complexity (a fallback path that did not exist before). The worsening is not incidental; it is the price of the design, and it is the reason the second desk question is real.

**Attribution.** The replay cannot attribute the change to the model alone, because the policy change includes banding and capacity logic as well as the score. A component ablation (score replaced by random ranking at the same capacity) showed a residual saving of about 3,900 USD/day, so roughly 84% of the replay difference is attributable to the ranking and 16% to the banding/capacity change. This is a decomposition, not a causal claim about live behaviour.

**Interview cue.** Ask what improved, what worsened, and whether the reported change can be attributed to the intervention. Answer: the replay shows a cost reduction; the ablation separates ranking from banding; attribution to a live population is not established.

**Acceptance.** Every headline result is scoped, traceable, and consistent with the evidence metadata.

## 12. Failure Modes & What Went Wrong

| Failure ID | Trigger and symptom | Detection / discriminating test | Cause or hypothesis | Impact | Containment | Prevention / owner | Verification |
|---|---|---|---|---|---|---|---|
| F-01 | Offline AUC 0.94, live precision 0.71 | Compare point-in-time snapshot to a rebuild with a 24 h lag | Merchant aggregate built over the full quarter (future information) | Decline band over-fires on legitimate merchants | Disable merchant aggregate, re-score | Earliest-known-time audit on every feature; model risk | Replay with the feature removed |
| F-02 | Precision falls month over month with no code change | Cohort the metric by label maturity age | Label censoring: recent events have only fast-resolving disputes | Apparent drift that is an artefact | Report on mature cohorts only | Fixed 120-day maturity gate in the evaluation job | Compare mature-only vs all |
| F-03 | Fallback rate jumps to 7% at peak | Fallback SLI by minute; feature-service p99 by shard | Feature service hot shard from a large merchant | Silent degradation to weaker rules at the worst time | Circuit breaker plus desk alert; rebalance shards | Shard by card token with per-merchant fan-out limits | Load test at 3.2x with a synthetic hot merchant |
| F-04 | Decline rate rises 4 pp on telephone orders | Channel-stratified decline rate | Missing device fingerprint shifts the score distribution | Channel-specific harm | Channel offset or exclusion | Channel-stratified calibration; model risk | Backtest per channel |
| F-05 | Review queue dwell p95 goes from 2 h to 9 h | Queue depth and dwell telemetry | Capacity gate uses a rate but the desk roster is per-shift | SLA breach; cases age out | Throttle referrals, widen decline band | Rate budget aligned to shift staffing | Simulate a 30% volume spike |
| F-06 | Duplicate declines on retried authorizations | `auth_id` duplicates in the decision log | Retry without idempotency key | Double friction and double counting in metrics | Deduplicate on `(auth_id, request_seq)` | Idempotency enforced at the switch | Replay a retry storm |
| F-07 | Velocity counters reset after a stream restart | Velocity feature drops to 0 for 10 minutes | State not rebuilt to the watermark before serving | Fraud window opens silently | Fallback during rebuild | Rebuild-then-serve gate; alert on counter drop | Restart drill with a 10-minute gap |
| F-08 | Amounts compared across currencies without normalisation | Mean event value shifts with FX | Minor units compared as if one currency | Thresholds mis-set for cross-border | Normalise to USD at ingestion | Currency field mandatory; unit test on conversion | Property test with 3 currencies |
| F-09 | Score cliff: 0.79 declines, 0.78 approves, no review | Histogram of scores in `[s_l, s_h)` | Cutoffs set too close after a capacity cut | Lost catch value; brittle behaviour | Widen the band or accept the cliff explicitly | Band width reviewed with capacity | Score histogram before release |
| F-10 | Analyst override rate 38% on step-ups | Reason-code distribution | Band sends events analysts judge unactionable | Wasted capacity; model value overstated | Re-weight features; retrain on dispositions | Override-rate alert at 25% | Blind inter-rater sample |
| F-11 | Merchant collusion undetected | Post-hoc review of missed high-value clusters | Event-level features cannot see coordinated merchants | Concentrated loss | Add merchant-cluster monitoring (out of scope now) | Planned graph work in the AML track | Cluster audit on missed events |
| F-12 | Rollback fails because the old bundle expects a removed field | Rollback drill error | Schema and bundle not versioned together | Extended outage | Keep the previous schema readable for one release | Compatibility window policy | Rollback rehearsal each release |

**Failure that actually occurred in this case's history (illustrative).** F-01 is the one that stopped the previous attempt: the merchant aggregate was built from the full quarter during the first offline evaluation, producing an AUC of 0.94 that collapsed to 0.71 in a point-in-time replay. The mechanism is worth stating precisely, because it is the most common fraud-modelling error: the aggregate was computable in the warehouse, so it looked available; but at decision time the value for a future merchant would have included events that had not happened. A second, subtler version of the same error is using the *current* merchant risk rating as a feature rather than the rating as of the event.

**A failure that did not occur and was wrongly predicted.** The team expected F-05 (queue overload) to bite immediately at canary scale; it did not, because the 5% canary's referral volume stayed inside the desk's headroom. The lesson is that capacity failures are volume-dependent, and a canary at 5% cannot reveal them; the load test must be synthetic at full volume.

**Distinguishing F-02 from F-03.** Both present as "precision fell." F-02 has a falling fallback rate and stable feature latency; F-03 has a rising fallback rate and a feature p99 spike. The discriminating check is the fallback SLI alongside the precision trend, which is why both are on the same dashboard.

**Interview cue.** Present the symptom first, require competing hypotheses, a discriminating diagnostic, a safe mitigation, and a prevention test.

**Acceptance.** A reader learns how to distinguish this mechanism from a similar-looking failure and how to verify the fix.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `cost-latency-tradeoff` — value-weighted three-way action under a hard capacity and a hard latency budget |
| General principle | When a scarce human resource sits between two automated actions, optimise a money objective with the resource as a constraint, and judge the change on the constrained resource's marginal value rather than on classifier accuracy. |
| Preconditions | A measurable per-event cost for each action; an observable label with known maturity; a capacity that can be expressed as a rate; a latency budget that forbids the richest features |
| Mechanism | Separating the decline band (high precision, low volume) from the review band (low precision, capacity-bound) converts a classifier into a decision system; the marginal-value calculation decides whether capacity is worth buying |
| Supporting evidence | Illustrative replay: 24,543.12 USD/day, dominated by friction; marginal review value 6.09 USD at 1,320 referrals against 3.78 USD marginal cost; capacity optimum 2,086 reviews/day, second desk worth 873 USD/day after fixed cost |
| Avoid when | Labels never mature (no ground truth); the review resource has no meaningful capacity limit; the per-action costs are unknown and cannot be bounded; the latency budget allows no features beyond the raw event |
| Portability test | Can you write the objective as a sum of per-event costs with one hard resource constraint? If not, this pattern does not apply. |
| Adaptation | In claims triage the resource is adjuster minutes and the label is payment outcome; in collections it is agent calls and the label is cure. The mechanism transfers; the constants do not. |

**What is structural and what is local.** Structural: a scarce middle action creates a capacity-constrained optimisation, and label maturity forces evaluation on mature cohorts. Local: the specific 60 ms budget, the 2.10-minute handling time, the 12.60 USD friction, and the 22% recovery rate.

**Where following this lesson would be harmful.** If the review desk is a compliance control rather than a value-generating resource — for example, a sanctions screening queue where every hit must be reviewed by rule — then capacity is not a knob and the marginal-value calculation is inapplicable. Treating a mandatory review as optimisable is a governance error, not an efficiency gain.

**Interview cue.** Transfer the approach to a contrasting domain and name the first assumption that breaks. Answer: in sanctions screening, the first break is that the review is mandatory, so the "spend capacity where value is highest" step is illegal rather than suboptimal.

**Acceptance.** The lesson remains useful without the company name and includes a condition under which it is harmful.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| uses-method | [02 Fraud Detection and Thresholds](../Cheat_Sheets/02_Fraud_Detection_and_Thresholds.md) | Threshold selection under value weighting and capacity | The sheet derives the cutoff rule; this case applies it to three actions |
| contrasts-with | [05 Payment Outage Fraud Controls](./05_Payment_Outage_Fraud_Controls.md) | Same portfolio, but the constraint is a degraded dependency rather than capacity | During an outage the feature service is gone, so fallback becomes the primary policy |
| contrasts-with | [02 AML Ring Detection](./02_AML_Ring_Detection.md) | Graph structure rather than event features; labels are far slower | A ring is not visible in a single authorization |
| uses-method | [07 Streaming Fraud Controls](../Cheat_Sheets/07_Streaming_Fraud_Controls.md) | Event-time windows, watermarks, and state rebuild | Velocity correctness is a systems property, not a model property |
| extends | [03 Sanctions False Positive Reduction](./03_Sanctions_False_Positive_Reduction.md) | Same friction-versus-coverage tradeoff, different legal footing | Screening hits are mandatory to review, so capacity is not a free knob |

### 14b. What this case leaves open

| Open question | Why insufficient | Test that would resolve it | Priority / trigger | Owner |
|---|---|---|---|---|
| Does declining fewer customers actually reduce abandonment? | Replay cannot observe behavioural response | Randomised canary with a held-out 5% control | High, before full rollout | Fraud ops lead |
| How much of the loss reduction persists as fraudsters adapt? | One quarter, no adversarial drift | Month-over-month capture on a fixed policy | High, monthly | Model risk |
| Is the merchant aggregate worth its leakage risk? | Removed after F-01, not re-tested correctly | Point-in-time merchant features with a strict window | Medium | Data platform |
| Can a sequence model fit the 60 ms budget? | Only offline feasibility studied | Serving benchmark with a 20-event window | Low | Platform lead |

### 14c. Intense interview pack — questions with checked answers

**Interview 1 — Framing**
*Level:* core. *Skill:* problem framing.
*Candidate prompt:* "You have 30 seconds. What is the decision, for whom, against what, and what is binding?"
*Clarify or assume:* ask whether the review desk is a hard constraint or a cost; assume hard.
*Strong answer:* The decision is a three-way action per authorization event for the fraud operations lead; the comparator is the static rules policy; the binding constraints are the 60 ms compute budget and 1,320 reviews/day. The objective is money: net loss plus friction plus labour, not accuracy. *Reasoning:* each action has a different cost structure, and the middle action consumes a resource that cannot scale within the quarter. *Check:* state the units (USD/day) and the population (all events, one country).
*Acceptable alternatives:* binary framing is acceptable only if the desk is excluded from scope, and then the answer must say so and lose the capacity analysis.
*Follow-up:* "Suppose the desk can hire. What changes?" The objective stays; the constraint relaxes; the answer becomes a marginal-value comparison between hiring and declining.
*Red flags:* "maximise AUC"; "minimise fraud"; no baseline; no capacity.
*Scoring:* 0 no decision; 1 decision without constraints; 2 all five elements; 3 adds the value-of-capacity argument.
*Support:* Sections 1 and 3.

**Interview 2 — Baseline**
*Level:* core. *Skill:* comparator design.
*Candidate prompt:* "Why not compare against last quarter's actuals?"
*Clarify or assume:* the traffic mix changed between quarters.
*Strong answer:* Last quarter's actuals mix population, season, and policy; the comparison would confound the policy change with the traffic change. The baseline must be the incumbent rules replayed on the same events with the same costs and capacity. *Reasoning:* a cost difference is only interpretable if everything except the policy is held fixed. *Check:* confirm the replay's baseline cost matches the incumbent's accounting within 0.1%.
*Acceptable alternatives:* a live holdout is stronger but slower; if used, the sample-ratio and interference checks are mandatory.
*Follow-up:* "The replay baseline is 70,081 and finance says 68,900. What now?" Reconcile the recovery-rate assumption and the false-decline count before proceeding.
*Red flags:* comparing two different quarters; using the model's own training period as the baseline.
*Scoring:* 0 "last quarter"; 1 same-events replay without cost alignment; 2 full alignment; 3 adds the reconciliation check.
*Support:* Section 9.

**Interview 3 — Label maturity**
*Level:* advanced. *Skill:* evidence discipline.
*Candidate prompt:* "Your 14-day precision is 0.31 and your 120-day precision is 0.17. Which do you report?"
*Clarify or assume:* the decision window is the full dispute period.
*Strong answer:* Report the mature figure for decisions and the immature figure only as an operations signal, because the 14-day slice over-represents fast-resolving, high-value disputes. *Reasoning:* censoring is not random with respect to value, so the early slice is biased upward in exactly the dimension that drives the money objective. *Calculation:* at 30 days the expected captured-loss credit is about 55% of the mature figure (21,225 vs 38,591 in this scenario), which is an illustration of the magnitude of the bias. *Check:* cohort every metric by label age.
*Acceptable alternatives:* an explicit survival model for time-to-resolution is defensible and better, but must be reported as modelled, not observed.
*Follow-up:* "The scheme shortens the window to 60 days. What breaks?" The maturity gate, the evaluation window, and the recovery rate all change; the loss comparison must be recomputed.
*Red flags:* reporting the higher number; treating unresolved as non-fraud.
*Scoring:* 0 picks 0.31; 1 picks 0.17 without mechanism; 2 explains censoring; 3 adds the survival alternative.
*Support:* Sections 5a and 12 (F-02).

**Interview 4 — Leakage**
*Level:* advanced. *Skill:* feature audit.
*Candidate prompt:* "AUC 0.94 offline, 0.71 live. Give three hypotheses and the test that separates them."
*Clarify or assume:* the live drop is measured on mature labels.
*Strong answer:* (a) future information in an aggregate — test by rebuilding features with a strict point-in-time window; (b) population shift — test by reweighting the offline sample to live marginals; (c) serving skew — test by comparing online and offline feature values for the same `auth_id`. *Reasoning:* each produces a different signature: (a) collapses when the window is fixed, (b) persists after window fixes but moves with reweighting, (c) shows a value mismatch on identical keys. *Check:* the merchant aggregate in this case was built over the full quarter, which is hypothesis (a).
*Acceptable alternatives:* shadow-scoring every event and comparing distributions is a valid third path.
*Follow-up:* "What if all three tests pass?" Then suspect label definition drift between offline and live.
*Red flags:* "the model overfit" with no test; retraining as the first response.
*Scoring:* 0 one hypothesis; 1 three without tests; 2 three with tests; 3 orders them by cost and names the signature.
*Support:* Section 5d and F-01.

**Interview 5 — Calculation**
*Level:* core. *Skill:* value-weighted recall.
*Candidate prompt:* "Event recall is 0.755 but value capture is 0.620. Is that possible, and what does it mean?"
*Clarify or assume:* the missed events have a higher mean value.
*Strong answer:* Yes. 317 of 420 events are caught, but the 103 missed events average 480.35 USD against a caught average of about 410; the policy is better at catching small frauds than large ones. *Calculation:* caught value 80,724 of 130,200 = 62.0%; missed 49,476/103 = 480.35. *Reasoning:* velocity-driven features flag repeated small attempts, while a single large first attempt looks like a legitimate high-value purchase. *Check:* compute both metrics on the same window.
*Acceptable alternatives:* if the value gap is deliberate (large events are reviewed, not declined), say so and show the review-band value.
*Follow-up:* "Design a change that raises value capture without raising reviews." Move the high band's cutoff on value-adjusted score, or add a value-scaled term to the objective.
*Red flags:* assuming recall and value capture must be equal; reporting only the higher one.
*Scoring:* 0 impossible; 1 possible without mechanism; 2 with numbers; 3 with the design change.
*Support:* Sections 6 and 7.

**Interview 6 — Capacity**
*Level:* advanced. *Skill:* constrained optimisation.
*Candidate prompt:* "Capacity halves tomorrow. What do you change first?"
*Clarify or assume:* staffing cannot change; the latency budget is unchanged.
*Strong answer:* Narrow the review band and widen the decline band using the same score, then recompute the cutoffs on the constrained objective. The replay shows the saving falls from 24,543 to 10,646 USD/day, so the policy still helps but the friction gain shrinks. *Calculation:* 660 reviews at p_rev 0.1697, v_rev 186 → caught value 59,892, loss 54,840.24, total 59,435.04. *Reasoning:* the decline band must absorb events the desk can no longer screen, which raises precision requirements on that band. *Check:* verify the decline count stays under 50/day.
*Acceptable alternatives:* keep the band and let the queue age — rejected because it breaches the dwell SLA and delays the customer impact rather than avoiding it.
*Follow-up:* "And if capacity doubles to 2,640?" The total cost *rises* from 43,564.74 (at the 2,086 optimum) to 44,159.39, because the added referrals are 98.7% non-fraud; do not staff for it without measuring the friction constant first.
*Red flags:* "refer the same number and work faster"; ignoring the friction consequence.
*Scoring:* 0 no change; 1 widen declines without recomputation; 2 recompute with numbers; 3 adds the marginal-value judgement.
*Support:* Sections 6 (Step 11) and 8d.

**Interview 7 — Sizing**
*Level:* core. *Skill:* workload arithmetic.
*Candidate prompt:* "How many analysts for 1,320 reviews at 2.10 minutes?"
*Clarify or assume:* 6.3 productive hours per 8-hour shift; 3 shifts cover the day.
*Strong answer:* 1,320 × 2.10 = 2,772 minutes = 46.2 hours; one desk covers 18.9 productive hours per day, so 2.44 desks are needed; staff 3 for 22% headroom. *Calculation:* 46.2/18.9 = 2.44; 3/2.44 − 1 = 22.9% headroom. *Reasoning:* headroom absorbs handling-time variance; a 25% rise consumes it.
*Acceptable alternatives:* 2 desks if handling time is capped and the queue can tolerate occasional backlog — but then the dwell SLA must be renegotiated.
*Follow-up:* "Handling time rises to 3 minutes." Requirement becomes 66 hours, i.e. 3.49 desks; 3 desks is now short.
*Red flags:* dividing by 24 hours; ignoring breaks.
*Scoring:* 0 no arithmetic; 1 hours only; 2 desks with headroom; 3 handles the variance case.
*Support:* Section 8d.

**Interview 8 — Latency**
*Level:* advanced. *Skill:* systems design.
*Candidate prompt:* "Your components are 6, 20, 11, and 1 ms at p50 and 9, 34, 19, and 2 ms at p99. Is the end-to-end p99 64 ms?"
*Clarify or assume:* the components are on the critical path and their latencies are correlated.
*Strong answer:* No. Adding p99s is not a valid end-to-end p99; it is an upper bound only under perfect positive dependence and an overestimate otherwise. Measure the end-to-end distribution, and design the budget so the sum of p99s is inside the hard limit as a conservative envelope. *Calculation:* the envelope is 64 ms, above the 60 ms limit, so the design is not safe even under the conservative assumption. *Reasoning:* tail events often co-occur (a slow feature service implies a loaded host that also delays the model), so treating them as independent understates the true tail.
*Acceptable alternatives:* a 55 ms sum-of-p99 envelope with a measured end-to-end p99 of 52 ms is defensible.
*Follow-up:* "Which component would you cut first?" The 20 ms feature call, by precomputing the 24 h aggregate asynchronously and serving a 20 ms stale value.
*Red flags:* "64 ms is fine because they rarely co-occur"; summing p50s.
*Scoring:* 0 yes; 1 no without method; 2 explains dependence; 3 adds the mitigation.
*Support:* Section 8b.

**Interview 9 — Fallback**
*Level:* advanced. *Skill:* degradation design.
*Candidate prompt:* "The feature service is down. Approve, decline, or hold?"
*Clarify or assume:* the switch cannot hold beyond the timeout; the incumbent rules are available.
*Strong answer:* Fall back to the incumbent rules, which is the declared safe state, and stamp a fallback reason code so the decision is auditable and the rate is monitored. Approving everything removes the loss guardrail; declining everything removes the friction guardrail; holding is not available at authorization time. *Reasoning:* the fallback must be a *policy*, not an absence of one, and it must be the policy whose behaviour is already understood.
*Acceptable alternatives:* a reduced-feature model (amount, MCC, country) is better than rules if it has been validated for that degraded mode — but it needs its own evaluation.
*Follow-up:* "The outage lasts 40 minutes at peak. What is the exposure?" At 44.4 events/s, 40 minutes is about 106,560 events; at the rules' 55% capture and the candidate's 62%, the incremental loss is roughly 0.07 × 130,200 × (106,560/1,200,000) ≈ 809 USD, which is small relative to the outage's other costs.
*Red flags:* fail-open; fail-closed; no reason code.
*Scoring:* 0 fail-open; 1 rules fallback; 2 with reason code; 3 with the exposure estimate.
*Support:* Sections 8b and F-03.

**Interview 10 — Thresholds**
*Level:* advanced. *Skill:* calibration.
*Candidate prompt:* "Your score's base rate changed from 0.035% to 0.05%. Do the cutoffs move?"
*Clarify or assume:* the score is calibrated to the old base rate.
*Strong answer:* Yes. A fixed cutoff now admits more events above it; to hold the referral rate at capacity, the lower cutoff rises; to hold precision in the decline band, the upper cutoff also rises unless the model is recalibrated. Recalibration is usually cheaper than moving both cutoffs. *Reasoning:* cutoffs encode a rate decision; a base-rate shift changes the rate at every fixed cutoff.
*Acceptable alternatives:* freeze cutoffs and let the capacity gate absorb the change — acceptable only if the gate's rate budget is the true control.
*Follow-up:* "The base rate falls to 0.02%?" The referral band fills with non-fraud; the marginal review value falls toward zero and capacity should shrink.
*Red flags:* "cutoffs are properties of the model"; never recalibrating.
*Scoring:* 0 no; 1 yes without direction; 2 both directions; 3 adds recalibration as the cheaper path.
*Support:* Sections 5c and 8c (DEC-04).

**Interview 11 — Evaluation**
*Level:* core. *Skill:* metric selection.
*Candidate prompt:* "Why not use F1 to pick the cutoffs?"
*Clarify or assume:* costs are available.
*Strong answer:* F1 weights precision and recall equally and ignores value and capacity, so it can prefer a policy that catches many small frauds and misses large ones, or that floods the desk. Use the money objective with capacity as a constraint; report F1 only as a descriptive diagnostic. *Reasoning:* the decision's loss function is value-weighted and asymmetric; F1 is symmetric and count-based.
*Acceptable alternatives:* cost-weighted F-beta is closer but still lacks the capacity constraint.
*Follow-up:* "Costs are unknown. Now what?" Bound them and test the sign of the decision across the bounds; if the sign flips, defer.
*Red flags:* "F1 is standard"; tuning on the test set.
*Scoring:* 0 F1; 1 rejects without alternative; 2 money objective; 3 adds the unknown-cost bound.
*Support:* Section 9.

**Interview 12 — Debugging**
*Level:* advanced. *Skill:* diagnosis.
*Candidate prompt:* "Precision fell 6 pp this week. Give a discriminating plan."
*Clarify or assume:* no code change was released.
*Strong answer:* Check in order: label maturity mix (cohort by age), fallback rate (dependency health), channel mix (population shift), and score distribution (calibration). Each has a distinct signature: maturity mix shows the fall only in recent cohorts; fallback shows a rising fallback SLI; channel mix shows the fall concentrated in one channel; calibration shows a shifted score histogram with stable inputs. *Reasoning:* the order is by cost of the check and by prior probability in this system. *Check:* the same week last quarter is the wrong control because the label window differs.
*Acceptable alternatives:* start with the fallback SLI if there was a deployment in the window.
*Follow-up:* "All four are stable." Then suspect the dispute process itself — a change in adjudication rules changes the label, not the model.
*Red flags:* retrain immediately; blame the model without a check.
*Scoring:* 0 retrain; 1 one check; 2 four with signatures; 3 orders by cost and adds the label-process hypothesis.
*Support:* Section 12 (F-02, F-03).

**Interview 13 — Wrong method**
*Level:* advanced. *Skill:* rejecting a plausible approach.
*Candidate prompt:* "A colleague proposes training on declined events only, since that is where the losses are. What is wrong?"
*Clarify or assume:* declined events have labels from disputes.
*Strong answer:* Declined events cannot produce fraud loss, so their label distribution is selected by the current policy; training on them teaches the model to reproduce the policy's selection, not to predict fraud. The correct population is all events, with the label observed regardless of action, which is why approved events must be sampled and audited. *Reasoning:* this is a selection-on-the-outcome problem; the missing counterfactual is "what would have happened had we approved."
*Acceptable alternatives:* rejected-outcome modelling with explicit exploration (random approvals at a small rate) is defensible and gives unbiased labels for the explored region.
*Follow-up:* "How much exploration is enough?" Enough to estimate the loss rate in the approved region with the required precision; at a 0.035% base rate, that is thousands of events per cell.
*Red flags:* "more data is better"; ignoring the policy's selection.
*Scoring:* 0 agrees; 1 says biased without mechanism; 2 explains selection; 3 adds exploration design.
*Support:* Sections 5a and 12.

**Interview 14 — Changed constraint**
*Level:* advanced. *Skill:* re-derivation.
*Candidate prompt:* "The scheme cuts the response window to 40 ms. What survives?"
*Clarify or assume:* the 20 ms feature call and 11 ms scoring remain as measured.
*Strong answer:* The envelope becomes 9 + 34 + 19 + 2 = 64 ms, which already exceeded 60; at 40 ms the design must drop the synchronous feature call. Precompute the 24 h aggregate asynchronously, serve it with a bound on staleness, and keep only amount, MCC, country, and a cached velocity in the synchronous path. *Reasoning:* the feature call is the only component with enough headroom to cut; the model's 11 ms p50 can survive if its feature set shrinks.
*Acceptable alternatives:* a two-tier decision (fast rules, then an asynchronous post-authorization review) shifts the decision later and changes the product.
*Follow-up:* "What is the cost of 5-minute stale velocity?" Velocity misses the first minutes of a burst; quantify by replaying with a 5-minute lag.
*Red flags:* "optimise the code"; ignoring that the budget is end-to-end.
*Scoring:* 0 nothing changes; 1 drop features without a plan; 2 recompute the envelope; 3 adds a staleness measurement.
*Support:* Section 8b.

**Interview 15 — Transfer**
*Level:* advanced. *Skill:* generalisation.
*Candidate prompt:* "Where does this design fail if transplanted to sanctions screening?"
*Clarify or assume:* screening hits must be reviewed by rule.
*Strong answer:* The capacity-as-knob assumption fails: every potential match must be dispositioned, so the review queue cannot be capped, and the marginal-value calculation is inapplicable. What transfers is the value-weighted objective and the mature-cohort evaluation; what does not is the capacity gate and the decline band. *Reasoning:* the constraint changes from economic to procedural, which changes the feasible set rather than the objective.
*Acceptable alternatives:* use screening-list tuning to reduce false positives *before* the queue, which is the real lever in that domain.
*Follow-up:* "What is the analogous lever in screening?" Entity resolution and name-matching thresholds upstream, not queue triage downstream.
*Red flags:* applying the capacity gate to a mandatory control.
*Scoring:* 0 no transfer; 1 names capacity; 2 explains procedural constraint; 3 names the upstream lever.
*Support:* Section 13 and the sanctions case.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Payments fraud operations; card authorization |
| Problem type | Three-way action (approve/decline/review) under capacity and latency, value-weighted |
| Method | Gradient-boosted score + capacity-aware banding, rules fallback; baseline static rules |
| Primary result | Constructed replay: 70,081.20 → 45,538.08 USD/day; saving 24,543.12 USD/day, mostly friction |
| Key lesson | Optimise money with the review desk as a constraint; judge capacity by marginal value, not by fraud capture |
| Tags | `cost-latency-tradeoff`; `label-delay`; `human-in-the-loop` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-11-01-001 / 2.0 / 2026-11-10 |
| Evidence level | Illustrative (constructed values, recomputed arithmetic) |
| Implementation stage | Proposed; no deployment, no approval |
| Baseline / primary metric | Static rules / daily total cost J [USD/day], 1,200,000 events/day |
| Key limitation | Replay cannot observe customer response; labels mature over 120 days |
| Review trigger / owner | Latency budget, capacity, loss definition, or chargeback window change / fraud ops lead |
