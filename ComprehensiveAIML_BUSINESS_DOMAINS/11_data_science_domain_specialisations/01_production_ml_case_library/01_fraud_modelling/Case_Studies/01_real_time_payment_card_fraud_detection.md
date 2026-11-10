---
case_id: CASE-01-01-001
case_title: Real-time payment-card fraud detection under a friction budget
case_version: "2.0"
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
as_of_date: 2026-11-10
owner_role: Issuer fraud product owner
source_index: INDEX.md
review_trigger: New label source, authorization deadline, network rule, or material change to policy or customer mix
---

# Case study: Real-time payment-card fraud detection under a friction budget

Every quantity in this case is **ILLUSTRATIVE**. No issuer, portfolio, authorization log, model,
deployment, or measured financial result is described. The arithmetic is constructed to be
internally consistent and independently re-derivable from the stated inputs, which is a different
claim from being true of any real book of business. Where a number would in practice require a
measurement, the case says so and marks the quantity `not-measured`.

The one sourced element is the shape of the decision: a card authorization is a synchronous,
irreversible three-way choice made under a deadline measured in hundreds of milliseconds, and the
cost of each branch is asymmetric. NIST SP 800-63B and the EMV 3-D Secure specification describe
authentication and challenge mechanics that constrain what an issuer may do in flight; nothing in
those documents supplies the economic inputs used here.

## 1. The One-Line Framing

At authorization time, choose approve, step-up challenge, or decline for each card-not-present
purchase using only data that exists before the issuer's response deadline, so that mature
issuer-borne fraud loss plus monetized customer friction falls per 1,000 eligible attempts at an
unchanged challenge-rate ceiling, compared with the incumbent rules engine.

The framing matters because it fixes three things that are usually left vague. The unit of decision
is one authorization attempt, not one card and not one fraud case. The objective is a monetized
sum, not a classification score. The constraint is a friction budget expressed as a challenge rate
per 1,000 attempts, which is what a customer-experience owner will actually defend in a steering
meeting.

The naive framing — "detect fraud" — fails. Truth arrives as a chargeback or a confirmed-fraud
record weeks after the authorization, and only for the attempts the issuer chose to act on. The
score can never be evaluated against a clean label at the moment it is used, and the label it is
trained on was produced by the previous policy. Both facts are consequences of the decision, not
data-quality accidents.

The incumbent in this case is a versioned rules engine that already performs well enough to have
survived. Its measured numbers are unknown here; the case constructs a plausible baseline so that
the replacement argument can be examined on arithmetic rather than assertion. Any real programme
must measure its own baseline before opening the replacement question.

**Takeaway:** the deliverable is a three-way action policy under a hard deadline, and its value is
a monetized portfolio quantity, not an AUC.

## 2. Context & Background

A card-not-present purchase begins when a cardholder enters credentials at a merchant checkout. The
merchant's acquirer submits an authorization request through the card network to the issuing bank.
The issuer has a fixed window — commonly under two seconds, and often closer to one — to return an
approval, a decline, or (where the merchant and channel support it) a request for additional
authentication. If the issuer does not answer inside the network's window, the attempt times out,
and a timeout is operationally worse than a considered decision because it degrades both merchant
relations and customer trust while providing no fraud control at all.

Four parties hold distinct accountabilities, and conflating them is the most common governance
defect in this domain.

The **fraud product owner** owns the action policy: which band of risk receives which action, and
what the friction ceiling is. This role is accountable for the loss number and for the customer
experience number simultaneously, which is why the objective must be a single monetized quantity.

The **model risk function** independently approves the evidence package, the calibration method,
the challenger comparison, and the monitoring thresholds. It does not own the threshold; it owns
the claim that the threshold was chosen on defensible evidence.

The **authorization platform owner** owns latency, availability, idempotency, and the fallback
path. A model that cannot answer within the deadline is not a component of this system at all,
regardless of its offline discrimination.

The **customer operations and disputes function** owns the downstream consequences: contact-centre
volume after a false decline, goodwill credits, complaint handling, and the manual review queue for
appeals. Their capacity is a hard constraint that the fraud policy can silently violate.

The incumbent policy is a rules engine. Rules in production card fraud typically encode velocity
counters, merchant-category risk tiers, geography mismatch against the cardholder's recent history,
repeated-amount patterns, and issuer-specific blocklists. Such an engine is explainable to
auditors, cheap to evaluate, easy to roll back, and legally well-understood. It is also brittle at
the edges: a threshold tuned in one season drifts as the merchant mix changes, and rules interact in
ways that are hard to reason about once there are more than a few dozen.

Scope is one issuer portfolio, card-not-present purchase authorizations, thirty consecutive days.
Excluded: cash advances, refunds and reversals, ATM transactions, card-present contactless
authorizations, account-opening fraud, and merchant-side fraud. Each of those has a different
action set, a different loss bearer, and a different label process; folding them in would make the
ledger uninterpretable.

A crucial structural fact: an approved fraudulent purchase does not always cost the issuer. Under
liability-shift arrangements and depending on whether the merchant performed authentication, part
of the loss may fall on the merchant or acquirer. The case models only the issuer-borne share and
says so explicitly, because a programme that reduces total ecosystem fraud while increasing
issuer-borne loss is a failure from the issuer's chair.

**Takeaway:** the authorization platform, not the modelling team, defines the deadline; the
disputes function, not the model, pays for false declines.

## 3. The Business Question

| Decision-contract field | Illustrative contract |
|---|---|
| Question | For each eligible authorization attempt, which of approve, challenge, or decline minimizes expected issuer cost inside a fixed friction ceiling? |
| Decision owner | Fraud product owner, with model risk approval of the evidence and platform approval of the latency envelope. |
| Trigger | Authorization request arrival; response due within the network deadline. |
| Deadline | End-to-end scoring and policy must complete inside the stated window with margin for retries. |
| Available actions | Approve; challenge where the channel supports it; decline; fall back to the incumbent rules engine. |
| Value | Mature issuer-borne fraud loss plus monetized false-decline and challenge friction, per 1,000 eligible attempts. |
| Denominator | All eligible authorization attempts assigned to the policy, including ones that were challenged or declined. |
| Hard constraints | Challenge rate ceiling; no decline without a documented reason code; approved data only; deadline compliance. |
| Guardrail metrics | False-decline rate, challenge abandonment, complaint rate, timeout rate, subgroup decline parity. |
| Comparator | The versioned incumbent rules engine evaluated on the same attempts in the same window. |
| Out of scope | Criminal attribution, merchant compliance judgement, credit-limit policy, collection of losses. |

Three denominators get confused in practice and each answers a different question. The
**attempt denominator** — all eligible attempts — is the one the business metric uses, because it
is the only one unaffected by the policy's own selection. The **actioned denominator** — attempts
that were challenged or declined — is the one precision uses, and it is a selected subset whose
base rate is far above the population's. The **confirmed-fraud denominator** — mature positives —
is the one recall uses, and it is incomplete for the most recent weeks because outcomes have not
landed.

A policy can look excellent on the actioned denominator while losing money on the attempt
denominator, because the actioned set is chosen precisely where the model is confident. This case
demonstrates exactly that failure shape in section 12, and the fix is not a better metric but a
correct one.

The estimand is stated as a difference, not a level: the change in mature cost per 1,000 eligible
attempts relative to the incumbent, computed on a common cohort with a common attribution window.
Levels are not comparable across issuers because ticket sizes, margin rates, and liability
arrangements differ; differences on a common cohort are.

One more contract detail that is often omitted and always regretted later: the decision must be
logged with the policy version, the score, the calibrated probability, the chosen action, the
reason code, and the availability timestamp of every feature used. Without that record the
evaluation in section 9 cannot be executed at all, because the replay cannot reconstruct what the
policy knew.

**Takeaway:** write the estimand as a paired difference on a common cohort, and make the decision
log a deliverable of the first sprint rather than a later refinement.

## 4. Assumptions & Constraints

| ID | Assumption | How it would be tested | Consequence if false |
|---|---|---|---|
| A-01 | Attempt, decision, and settlement timestamps are retained with millisecond resolution. | Replay a week of authorizations against stored logs. | Policy cannot be replayed; evaluation reverts to aggregate totals. |
| A-02 | A mature confirmed-fraud outcome exists for approved attempts and matures within 90 days. | Age the label table and plot maturity by cohort week. | Labels are biased toward early-reporting cardholders; recall is overstated. |
| A-03 | The challenge flow is available on the measured share of channels. | Channel capability matrix from the gateway. | Challenge arm cannot be priced; policy collapses to approve/decline. |
| A-04 | Challenge effectiveness can be estimated, even if only from a controlled experiment. | Randomize challenge on a small eligible band. | Prevention is unidentifiable; treat `e` as a sensitivity axis, not a number. |
| A-05 | Issuer-borne loss per caught attempt is approximately constant within a ticket band. | Stratify realized loss by ticket size decile. | The scalar `V` misprices small and large tickets; ledger must be per-band. |
| A-06 | Friction costs are agreed with the business before threshold selection. | Signed cost schedule in the decision record. | Threshold becomes a political negotiation after the fact. |
| C-01 | Scoring completes inside the deadline at p99, not just p50. | Load test with production replay at peak. | Timeouts create an unmodelled fourth action with its own cost. |
| C-02 | The same attempt retried returns the same action. | Inject duplicate authorization requests. | Retries produce contradictory decisions and double decline friction. |
| C-03 | No feature is used whose availability timestamp is later than the decision cutoff. | Point-in-time feature audit per column. | Offline evaluation is optimistic and the deployment underperforms. |

Two constraints deserve separate emphasis because they are usually stated too softly to be
enforceable. The first is the **deadline**, which is a property of the whole path: feature fetch,
model inference, policy evaluation, serialization, and network. A 12 ms model inside a 900 ms p99
budget is fine; the same model inside a 40 ms budget with a 30 ms feature fetch is not, and the
failure shows up as timeouts rather than as a bad score. The second is the **friction ceiling**,
which must be expressed as a rate over eligible attempts, not as a count per day, because the count
moves with volume and the rate does not.

The cost schedule is an assumption, not a fact, and it is the assumption that drives the answer
most strongly. Section 6 shows the decline break-even threshold moving from roughly 16% to roughly
37% when the monetized cost of a false decline rises by a factor of four. A programme that fixes
the threshold and then renegotiates the cost schedule has the causality backwards.

**Takeaway:** the two constraints that most often invalidate a finished model are the p99 latency
budget and a cost schedule that nobody signed.

## 5. Data, Tools & Method

### 5a. Data Specification

Grain: one row per authorization attempt. The primary key is the network authorization identifier
combined with the issuer's internal attempt sequence, because retries carry the same network
identifier and must be distinguishable. Cut-off: the decision timestamp recorded by the
authorization switch. Population: eligible card-not-present purchase attempts in the portfolio for
the thirty-day cohort.

Units: all monetary amounts in one currency, stated at authorization, not at settlement, because
settlement amounts can differ after tips, partial captures, and currency conversion. Rates are
expressed per 1,000 eligible attempts so that they remain comparable when volume changes.

Fields available at decision time, all candidate only:

| Field group | Examples | Availability caveat |
|---|---|---|
| Attempt attributes | amount, currency, merchant category, entry mode | Present, but merchant category is coarse and sometimes stale. |
| Card state | tenure band, prior attempt count in window, prior decline count | Velocity counters must be computed before the decision, not after. |
| Counterparty history | merchant's prior fraud rate band | Merchant-level aggregates are lagged by days and must carry their lag. |
| Device and channel | device token presence, channel, client type | Token absence is not evidence of a new device. |
| Geography | coarse network region, billing-country agreement | IP geolocation is unreliable on mobile networks and must not be treated as identity. |
| Policy state | incumbent rules hit flags, blocklist membership | These encode the incumbent's own history and can leak its conclusions. |

Fields **not** available at decision time and therefore excluded from any claimed feature set:
chargeback outcome, confirmed-fraud case identifier, dispute reason code, recovery amount, the
merchant's later refund, the customer's later complaint, and the analyst's review disposition. Each
of these is a downstream consequence. Including any of them produces an offline model whose
performance cannot survive deployment, and the failure mode is silent in cross-validation because
the leak is present in every fold.

The label is `confirmed_fraud` under a controlled definition: an attempt for which a chargeback was
raised and the issuer's recovery process concluded fraud, or an attempt the fraud operations team
independently classified as fraud. The definition excludes "customer claims they did not
authorize" without further evidence, because friendly-fraud and genuine third-party fraud require
different responses and a policy that treats them identically will over-decline repeat customers.

Negative labels are the harder problem. An approved attempt with no chargeback within the maturity
window is treated as legitimate, which is an assumption rather than a fact: some fraud is never
disputed (small amounts, cardholder inattention, issuer write-off below the dispute threshold).
Section 9 prices that assumption instead of hiding it.

Exclusions: test cards, internal merchant test traffic, attempts from blocked countries where the
block is legal rather than fraud-driven, and attempts that failed for technical reasons before any
risk decision. Excluding technical failures is legitimate; excluding declined attempts is not,
because declined attempts are precisely where the policy's false positives live.

### 5b. Tools & Stack

Proposed components, described by function rather than by product because no environment was
inspected: an authorization switch that emits an immutable attempt record; a point-in-time feature
service that serves precomputed aggregates keyed by card and merchant with explicit freshness
stamps; a gradient-boosted tree scorer for tabular signals; an isotonic or Platt calibration layer;
a versioned policy table mapping calibrated probability bands to actions; a durable decision store
written before the response is emitted; a network adapter for step-up authentication; and a replay
harness that can reconstruct any historical decision from stored snapshots.

Why a tree ensemble rather than a neural network is a decision that should be argued rather than
assumed. The table below records the argument so that a reviewer can attack it.

| Option | Strength | Weakness | Status here |
|---|---|---|---|
| Incumbent rules only | Explainable, auditable, instant rollback, no training dependence | Brittle thresholds, poor at interactions, manual retuning | Comparator |
| Logistic regression on engineered features | Transparent coefficients, stable, fast | Needs manual interactions, misses nonlinearities | Fallback candidate |
| Gradient-boosted trees | Handles interactions and missingness, strong tabular baseline | Opaque without attribution tooling, calibration needs work | Proposed scorer |
| Sequence model over card history | Can exploit ordering and velocity patterns | Latency budget, feature-serving complexity, harder audit | Considered, deferred |
| Graph model over shared entities | Strong on organized rings | Labelling and latency burden, cold-start on new entities | Out of scope |

The scorer is a *component*; the policy is the *decision*. That separation is what allows the
evaluation in section 9 to compare policies while holding the scorer fixed, which is the only way
to attribute a change in cost to a change in thresholds rather than to a change in model.

### 5c. Method Summary

The pipeline is: receive attempt; validate eligibility; fetch features with freshness checks; score;
calibrate; apply the versioned policy band; emit action with reason code; persist the decision
before responding; asynchronously join the mature outcome when it lands.

| Stage | Input | Output | Failure behaviour |
|---|---|---|---|
| Eligibility filter | Attempt record | Eligible / not eligible | Not eligible bypasses scoring and uses incumbent rules. |
| Feature fetch | Card and merchant keys | Feature vector plus freshness | Stale beyond tolerance triggers fallback, not zero-fill. |
| Scoring | Feature vector | Raw score | Scorer error triggers fallback to incumbent rules with a logged flag. |
| Calibration | Raw score | Probability of confirmed fraud | Calibration table version pinned per day. |
| Policy | Probability plus context | Action plus reason code | Unknown band is a hard error, never a default approve. |
| Persistence | Decision record | Durable row | Persistence failure blocks the response; no silent approve. |

The choice to make persistence synchronous is deliberate and expensive. An asynchronous write would
buy a few milliseconds and lose the guarantee that every action can be reconstructed, which would
make the evaluation in section 9 impossible. The cost is roughly 3 ms at p99 in a co-located store,
which fits the budget; in a cross-region store it would not, and the design would have to change.

### 5d. Provenance, quality, exclusions and leakage controls

Each column carries an owner, a source system, a version, a timestamp semantic (event time versus
processing time), and a permitted-use note. The distinction between event time and processing time
is the single most common source of silent leakage in payment data: a velocity counter updated at
settlement looks like it belongs to the attempt, but its value at decision time was different.

Leakage controls, in the order they are applied:

1. Every feature is rebuilt from a snapshot taken strictly before the decision timestamp.
2. Incumbent rule flags are retained only as *context*, never as features, because they encode the
   incumbent's own conclusions and would let the challenger inherit its errors.
3. Merchant aggregates are joined with their documented lag; a same-day merchant fraud rate is not
   available at authorization.
4. Calibration and band selection are fitted on development folds only, by time.
5. Any feature whose univariate relationship with the label exceeds a plausibility bound is
   quarantined and re-audited before use, because implausible strength is the signature of leakage.

Quality checks that are actually run before the model is trusted: duplicate attempt keys within a
day; attempts with decision timestamps before their event timestamps; cards with more than a
threshold of attempts in a minute; amount distributions with impossible values; and label joins
that map more than one outcome to one attempt.

### 5e. Reproducibility & Method-Selection Ledger

| ID | Artifact | Purpose | Evidence boundary |
|---|---|---|---|
| RUN-01 | Constructed thirty-day cohort | Arithmetic demonstration | No dataset exists; all values constructed. |
| POL-01 | Three-action band table | Action selection | Illustrative; requires signed cost schedule. |
| SRC-01 | EMV 3-D Secure specification | Challenge mechanics context | Does not supply any economic input. |
| SRC-02 | NIST SP 800-63B | Authentication assurance vocabulary | Not a claim about this portfolio. |
| CHK-01 | Independent arithmetic re-check | Verifies every printed product | Executed by hand and by script; see below. |

Reproducibility means that a reviewer with the same inputs can regenerate every number in section
6 without reading the code. To make that possible, each derivation states its formula, its symbol
meanings and units, its validity condition, one numerical substitution, and the resulting figure.
Where a figure is rounded, the rounding rule is stated once: monetary values are carried at full
precision and displayed to two decimals; probabilities are displayed to two decimals as
percentages; counts are integers, and any fractional count is rounded with the rule stated at that
point.

**Takeaway:** the ledger is a list of what would have to exist for this case to be evidence, and
none of it does.

## 6. Analysis Walkthrough

The walkthrough builds the answer in seven steps. Each step either produces a decision input or
invalidates one. Nothing here was executed against real data; the cohort is constructed so the
arithmetic closes exactly, which is what allows a reader to check the reasoning rather than the
data.

### Step 1 — Fix the cohort, the base rate, and the cost scalars

The constructed cohort is thirty days of eligible attempts in one portfolio.

```text
Cohort size            N  = 5,000,000 attempts
Confirmed fraud        P  = 10,000 attempts
Base rate              pi = P / N = 10,000 / 5,000,000 = 0.0020 = 0.20%
Average ticket         T  = $62.00
Issuer margin rate     m  = 18%
Recovery share         r  = 85%   (share of a fraudulent approved amount the issuer ultimately bears)
```

`V`, the issuer-borne loss when a fraudulent attempt is approved, is the quantity the whole ledger
is denominated in. It is not the ticket size and it is not the ticket size times the margin. It has
two components that behave differently and must be added rather than multiplied.

The first is the amount the issuer ultimately bears on the transaction after representment and
write-offs. The constructed convention assigns 85% to the issuer, the remainder falling on merchants
or acquirers under the applicable liability arrangement:

```text
Amount borne = T * r = 62.00 * 0.85 = 52.70
```

The second is the fixed handling cost per confirmed fraud: case investigation, chargeback filing,
and the representment work that is incurred whether or not the amount is recovered. This is a
per-event cost, not a fraction of the ticket, and it is where most derivations of `V` go wrong by
omitting it, because it does not appear in the transaction record at all.

```text
Dispute and investigation handling, per confirmed fraud = 20.40
V = 52.70 + 20.40 = 73.10
```

Adding a per-event cost to a per-ticket amount is legitimate only because the two are both
*expected values per approved fraud* and the cohort's ticket distribution is stationary within the
window. If ticket sizes were bimodal, the sum would still be correct in expectation but the variance
would be large enough that a single `V` would misprice both modes; section 4's assumption A-05 exists
to force that check.

The honest structure of this figure is that `V` is an input calibrated from loss experience, and the
decomposition above is a plausibility check on it rather than its derivation. A programme that
derives `V` from a formula instead of measuring it will misprice the threshold by however wrong the
formula is, and it will never find out, because no experiment tests `V` directly.

Friction scalars, each of which must be signed off by the business:

```text
Margin at risk on a declined legitimate attempt  mgn  = 62.00 * 0.18 = 11.16
Service cost per decline (contact, goodwill)      svc  = 2.40
c_dec = mgn + svc = 13.56
Challenge send cost (SMS or push)                 c_sms = 0.43
Challenge abandonment rate                        ab    = 4.5%
Support cost per abandoned challenge              c_ab  = mgn + 3.20 = 14.36
c_chal = c_sms + ab * c_ab = 0.43 + 0.045 * 14.36 = 0.43 + 0.6462 = 1.0762
Challenge effectiveness                           e     = 0.65
```

`e` is the probability that a fraudulent attempt routed to challenge does not complete as fraud.
It is the weakest number in the ledger, and section 9 explains why it cannot be estimated from
observational data alone.

```text
FORMULA [D-01]: c_chal = c_sms + ab * c_ab
WHERE: c_sms [currency/attempt] is the per-challenge delivery cost; ab [fraction] is the share of
       challenges that legitimate customers abandon; c_ab [currency/abandonment] is margin plus
       support cost triggered by an abandonment.
CONSTRAINT: ab measured on legitimate challenged attempts only; if ab is measured on all challenged
       attempts it is inflated by fraudsters abandoning, and c_chal is overstated.
NUMERICAL EXAMPLE: 0.43 + 0.045 * 14.36 = 0.43 + 0.6462 = 1.0762.
INTERPRETATION: a challenge costs about one thirteenth of a decline (1.08 versus 13.56), which is
       why the incumbent can afford to challenge 4.2% of volume while declining only 0.22%.
```

Independent re-check: 0.045 * 14.36 = 0.6462 exactly; 0.43 + 0.6462 = 1.0762. Scaling check: if
ab doubles to 9%, c_chal rises to 0.43 + 1.2924 = 1.7224, a 60% increase in challenge cost even
though the challenge send cost is unchanged. This is why abandonment, not SMS price, is the
commercial sensitivity in the challenge arm.

### Step 2 — Derive the break-even probability for each action

For a single attempt with calibrated fraud probability `p`, compare expected cost under each
action. Approving costs `p * V`: with probability `p` the attempt is fraud and the issuer bears
`V`.

Declining costs `(1 - p) * c_dec`: with probability `1 - p` the attempt is legitimate and the issuer
pays the friction cost. A decline prevents the fraud loss entirely, which is an assumption examined
at the end of this step.

Challenging costs `c_chal + p * (1 - e) * V`: the challenge is paid for regardless of outcome, and
the fraud loss is borne only when the attempt was fraud and the challenge failed.

```text
FORMULA [D-02]: decline break-even
q_dec = c_dec / (c_dec + V)
WHERE: q_dec [probability] is the fraud probability at which declining and approving cost the same;
       c_dec [currency] is the friction cost of a false decline; V [currency] is issuer loss per
       approved fraud.
CONSTRAINT: requires V and c_dec constant across the band and requires that a decline actually
       prevents the loss (no attacker substitution, no re-attempt at another merchant).
NUMERICAL EXAMPLE: 13.56 / (13.56 + 73.10) = 13.56 / 86.66 = 0.15648 = 15.65%.
INTERPRETATION: below 15.65% fraud probability, declining costs the issuer more than approving.
```

Derivation, so the formula is not taken on trust. Set approve equal to decline:

```text
p * V = (1 - p) * c_dec
p * V = c_dec - p * c_dec
p * (V + c_dec) = c_dec
p = c_dec / (V + c_dec)
```

The same construction for the challenge arm, against approving:

```text
c_chal + p * (1 - e) * V = p * V
c_chal = p * V - p * V + p * e * V = p * e * V
q_chal = c_chal / (e * V)
```

```text
FORMULA [D-03]: challenge break-even against approving
q_chal = c_chal / (e * V)
WHERE: e [fraction] is challenge effectiveness; all other symbols as above.
CONSTRAINT: e > 0 strictly. If e = 0 the denominator vanishes and q_chal is undefined, which is the
       boundary case: a challenge that never prevents anything is pure cost and must be removed
       from the action set rather than given a threshold.
NUMERICAL EXAMPLE: 1.0762 / (0.65 * 73.10) = 1.0762 / 47.515 = 0.022649 = 2.26%.
INTERPRETATION: challenging becomes worthwhile at a fraud probability of 2.26%, eight times lower
       than the decline threshold, because challenge is cheap and partially effective.
```

The third comparison is the one most often skipped, and it decides the shape of the whole policy:
when does declining beat challenging?

```text
Decline cost   = (1 - p) * c_dec
Challenge cost = c_chal + p * (1 - e) * V
Set equal:
(1 - p) * c_dec = c_chal + p * (1 - e) * V
c_dec - p * c_dec = c_chal + p * V - p * e * V
c_dec - c_chal = p * (c_dec + V - e * V)
p = (c_dec - c_chal) / (c_dec + (1 - e) * V)
```

```text
FORMULA [D-04]: decline-versus-challenge crossover
q_dc = (c_dec - c_chal) / (c_dec + (1 - e) * V)
WHERE: numerator [currency] is the cost saved by declining instead of challenging a legitimate
       attempt; denominator [currency] is the extra cost of declining when the attempt is fraud.
CONSTRAINT: valid only while c_dec > c_chal. If declining were cheaper than challenging the
       crossover would be negative and decline would weakly dominate.
NUMERICAL EXAMPLE: (13.56 - 1.0762) / (13.56 + 0.35 * 73.10) = 12.4838 / (13.56 + 25.585)
       = 12.4838 / 39.145 = 0.31892 = 31.89%.
INTERPRETATION: above 31.89% fraud probability, a hard decline beats a step-up challenge.
```

The three thresholds together define the action regions, and this is the entire policy:

```text
p < 2.26%            -> approve
2.26% <= p < 31.89%  -> challenge
p >= 31.89%          -> decline
```

Check the regions against direct cost evaluation at three probabilities, computing all three costs
and taking the minimum:

```text
p = 2.00%: approve 1.46 | challenge 1.59 | decline 13.29  -> approve wins
p = 10.00%: approve 7.31 | challenge 3.63 | decline 12.20 -> challenge wins
p = 50.00%: approve 36.55 | challenge 13.87 | decline 6.78 -> decline wins
```

Both closed forms above are first-order approximations, and it is worth knowing how good they are
rather than treating 2.26% and 31.89% as exact. The approximation drops one term: the abandonment
cost `ab * c_ab` is incurred on legitimate attempts only, so as `p` rises it is paid on a shrinking
share of the band, and the approximate form treats it as a fixed charge. Carrying the term through:

```text
Challenge minus approve = ab*(1 - p)*c_ab + c_sms - p*e*V
Set to zero and solve:
p*e*V + p*ab*c_ab = c_sms + ab*c_ab
p = (c_sms + ab*c_ab) / (e*V + ab*c_ab)
  = 1.0762 / (47.515 + 0.6462) = 1.0762 / 48.1612 = 0.022347 = 2.235%

Decline minus challenge, with the same term:
(1 - p)*c_dec - ab*(1 - p)*c_ab - c_sms - p*(1 - e)*V = 0
p = (c_dec - ab*c_ab - c_sms) / ((c_dec - ab*c_ab) + (1 - e)*V)
  = (13.56 - 0.6462 - 0.43) / (12.9138 + 25.585)
  = 12.4838 / 38.4988 = 0.324264 = 32.43%
```

So the exact crossover sits at 2.235% against the approximate 2.265%, and at 32.43% against 31.89%.
The gaps are 0.03 and 0.54 percentage points. Both are small enough that the action regions are
unaffected in this cohort, where band volumes are set by the score distribution rather than by the
edges, but the direction of the error is systematic and worth recording: the approximation
*understates* the decline crossover, so a policy built on it will decline slightly earlier than the
cost schedule justifies. On a band with dense mass near 32% that bias would be material. The regions
below use the approximate forms, because they are what the band solver implements, and the exact
forms are recorded here as the check on the solver.

Boundary where the linear model breaks. The formula assumes a decline prevents the loss with
certainty. In card-not-present fraud that is false in two directions. A fraudster whose attempt is
declined typically re-attempts with another card, another merchant, or the same card at a different
acquirer; the loss may be displaced rather than prevented, so the true benefit of a decline is less
than `V` and the crossover moves down. Conversely, under some liability arrangements the issuer is
not the party bearing the loss on an approved fraudulent transaction, in which case `V` is smaller
for that merchant band and the whole band should be approved. A single global `V` is therefore a
convenience, and the case flags it rather than hiding it.

### Step 3 — Convert the action policy into a band volume

A threshold is not a volume. To price the policy the case must know how many attempts fall in each
band, which requires the score distribution. The constructed distribution below is stated directly
as counts rather than as a density, because counts are what the ledger consumes and they can be
checked by addition.

Baseline, the incumbent rules engine over the thirty-day cohort:

```text
Attempts                  5,000,000
Declined                  11,000    (0.220% of attempts)   of which confirmed fraud 5,200
Challenged                210,000   (4.200% of attempts)   of which confirmed fraud 1,600
Approved without action   4,779,000                        of which confirmed fraud 3,200
Sum check: 11,000 + 210,000 + 4,779,000 = 5,000,000
Fraud check: 5,200 + 1,600 + 3,200 = 10,000
```

Candidate policy, same cohort, same scorer, different bands:

```text
Attempts                  5,000,000
Declined                  9,500     (0.190% of attempts)   of which confirmed fraud 4,900
Challenged                46,000    (0.920% of attempts)   of which confirmed fraud 2,600
Approved without action   4,944,500                        of which confirmed fraud 2,500
Sum check: 9,500 + 46,000 + 4,944,500 = 5,000,000
Fraud check: 4,900 + 2,600 + 2,500 = 10,000
```

Two features of this table carry the case's main finding. The candidate challenges 164,000 fewer
attempts, which is where most of the saving comes from, and it declines 1,500 fewer. Neither of
those is a modelling achievement; both are the result of a policy that prices actions instead of
using fixed rules. The scorer merely makes the priced band actionable at the required latency.

Precision in each action band, which is the metric that will be quoted in the review and which is
not the business metric:

```text
Baseline decline precision    = 5,200 / 11,000  = 47.27%
Baseline challenge precision  = 1,600 / 210,000 = 0.762%
Candidate decline precision   = 4,900 / 9,500   = 51.58%
Candidate challenge precision = 2,600 / 46,000  = 5.652%
```

A reviewer will read "0.762% challenge precision" and conclude the baseline is broken. It is not.
At a 0.20% base rate, challenging 4.2% of volume yields 210,000 challenges and can only find a
fraud if the challenge band contains fraud; 1,600 of 210,000 is a lift of 3.8x over the base rate.
Low precision in a high-volume, low-cost action band is normal and is the reason precision must be
paired with the cost of the action in the same sentence.

```text
FORMULA [D-05]: precision and lift in an action band
precision_band = TP_band / K_band
lift_band      = precision_band / pi
WHERE: TP_band [attempts] are confirmed frauds inside the band; K_band [attempts] is the band
       volume; pi [fraction] is the population base rate.
CONSTRAINT: both use the same window and the same label definition; lift is not comparable across
       base rates, so it must never be quoted without pi.
NUMERICAL EXAMPLE: baseline challenge band: 1,600 / 210,000 = 0.007619; lift = 0.007619 / 0.0020
       = 3.81x. Candidate challenge band: 2,600 / 46,000 = 0.056522; lift = 28.3x.
INTERPRETATION: the candidate concentrates the same control into one fifth of the volume at 7.4
       times the precision, which is the mechanical source of the cost reduction.
```

### Step 4 — Build the total cost ledger for both policies

Now the ledger. Four terms: uncaught fraud loss, false-decline friction, challenge delivery cost,
and abandonment cost. Every term is computed on the attempt denominator.

```text
FORMULA [D-06]: total policy cost over a cohort
C = (P - D_pos - e * K_pos) * V + (D - D_pos) * c_dec + K * c_sms + ab * (K - K_pos) * c_ab
WHERE: P [attempts] confirmed fraud in cohort; D [attempts] declined; D_pos [attempts] declined and
       confirmed fraud; K [attempts] challenged; K_pos [attempts] challenged and confirmed fraud;
       e [fraction] challenge effectiveness; ab [fraction] abandonment rate on legitimate
       challenges; V, c_dec, c_sms, c_ab [currency] as defined in step 1.
CONSTRAINT: K_pos * e and the abandonment count are rounded to whole attempts; the rounding is
       applied once, before multiplication, so the printed components add exactly to the total.
NUMERICAL EXAMPLE: baseline substitution below; candidate substitution below.
INTERPRETATION: the first term dominates by construction in any real portfolio, which is why a
       policy that trades fraud loss for friction must be checked, not assumed.
```

Baseline substitution:

```text
Prevented by challenge   = round(1,600 * 0.65) = round(1,040.0) = 1,040
Uncaught fraud           = 10,000 - 5,200 - 1,040 = 3,760
Fraud loss               = 3,760 * 73.10     = 274,856.00
False declines           = 11,000 - 5,200 = 5,800
False-decline friction   = 5,800 * 13.56     =  78,648.00
Challenge delivery       = 210,000 * 0.43    =  90,300.00
Abandonments             = round((210,000 - 1,600) * 0.045) = round(9,378.0) = 9,378
Abandonment cost         = 9,378 * 14.36     = 134,668.08
Total C_base             = 274,856.00 + 78,648.00 + 90,300.00 + 134,668.08 = 578,472.08
```

Note the abandonment base: legitimate challenges only, `K - K_pos = 208,400`, not all 210,000.
Including the fraud attempts would count abandonment of an attack as a customer cost, which
inflates the friction term and biases the policy toward fewer challenges for the wrong reason.

Candidate substitution:

```text
Prevented by challenge   = round(2,600 * 0.65) = round(1,690.0) = 1,690
Uncaught fraud           = 10,000 - 4,900 - 1,690 = 3,410
Fraud loss               = 3,410 * 73.10     = 249,271.00
False declines           = 9,500 - 4,900 = 4,600
False-decline friction   = 4,600 * 13.56     =  62,376.00
Challenge delivery       = 46,000 * 0.43     =  19,780.00
Abandonments             = round((46,000 - 2,600) * 0.045) = round(1,953.0) = 1,953
Abandonment cost         = 1,953 * 14.36     =  28,045.08
Total C_cand             = 249,271.00 + 62,376.00 + 19,780.00 + 28,045.08 = 359,472.08
```

The paired difference and its decomposition:

```text
Fraud loss     274,856.00 - 249,271.00 =  25,585.00   (11.7% of saving)
False declines  78,648.00 -  62,376.00 =  16,272.00   ( 7.4% of saving)
Challenge send  90,300.00 -  19,780.00 =  70,520.00   (32.2% of saving)
Abandonment    134,668.08 -  28,045.08 = 106,623.00   (48.7% of saving)
Total                                   = 219,000.00
```

Check the sum: 25,585 + 16,272 = 41,857; 41,857 + 70,520 = 112,377; 112,377 + 106,623 = 219,000.
Exact, because the inputs were chosen to make it exact; a real ledger would not close to the unit
and would require a residual line.

Expressed on the business denominator:

```text
Baseline  = 578,472.08 / 5,000 = 115.69 per 1,000 eligible attempts
Candidate = 359,472.08 / 5,000 =  71.89 per 1,000 eligible attempts
Improvement = (578,472.08 - 359,472.08) / 578,472.08 = 219,000 / 578,472.08 = 37.86%
```

The decomposition is the finding, not the headline. Roughly 81% of the saving comes from
challenging fewer attempts — 32.2% from delivery cost and 48.7% from abandonment. Only 11.7% comes
from catching more fraud. If the business case is written as "the model catches more fraud", it
mis-describes its own arithmetic and will be evaluated on the wrong quantity. The honest statement
is that the incumbent over-challenged because it had no way to price the challenge, and the
candidate's contribution is a calibrated probability that makes pricing possible.

### Step 5 — Test whether the improvement survives the cost assumptions

A single-point ledger is not evidence. The table perturbs one input at a time and recomputes both
policies through the same formula, holding everything else fixed. Baseline is recomputed under the
perturbed assumption too, because a sensitivity study that only perturbs the candidate answers a
different question.

| Perturbation | Baseline cost | Candidate cost | Saving | Saving % |
|---|---|---|---|---|
| Base case | 578,472 | 359,472 | 219,000 | 37.9% |
| `V` reduced 20% (to 58.48) | 523,501 | 309,618 | 213,883 | 40.9% |
| `V` increased 20% (to 87.72) | 633,443 | 409,326 | 224,117 | 35.4% |
| `e` reduced to 0.50 | 596,016 | 387,981 | 208,035 | 34.9% |
| `e` increased to 0.80 | 560,928 | 330,963 | 229,965 | 41.0% |
| Abandonment 2% | 503,656 | 343,891 | 159,765 | 31.7% |
| Abandonment 8% | 683,214 | 381,285 | 301,929 | 44.2% |

The savings are not stable in absolute terms — they range from 159,765 to 301,929 — but the sign
and rough magnitude are stable, and that is the claim a decision needs. Note the direction of the
abandonment sensitivity: a *higher* abandonment rate makes the candidate look better, because the
saving is dominated by avoided challenge volume. That is a warning sign as much as a comfort: the
improvement is largest exactly when the incumbent's behaviour is most expensive, so a baseline that
had already learned to challenge less would shrink this case's headline substantially.

The perturbation most likely to be wrong is `e`, and it moves the answer least, in relative terms.
That is a useful robustness property and it is worth stating why: the fraud-loss term is only 11.7%
of the total saving, so `e` cannot move the headline much. A reviewer who attacks `e` and ignores
the abandonment rate is attacking the load-bearing wall's wallpaper.

### Step 6 — Price the identification problem for `e`

`e = 0.65` is an assumption and it is not identified from observational data. The reason is
selection: challenge outcomes are observed only for challenged attempts, and an attempt that a
fraudster abandons at the challenge never becomes a confirmed fraud. So the observed "challenged
and confirmed fraud" count is a *subset* of the attempts that would have been fraud absent the
challenge, and the complement is unobservable.

```text
FORMULA [D-07]: apparent effectiveness and its bias
Let A = attempts challenged that would have been fraud had no challenge occurred.
Let O = attempts challenged that mature as confirmed fraud.
Then O = A * (1 - e), and the naive estimator is 1 - O/A.
WHERE: A [attempts] is counterfactual and unobservable in an observational design; O [attempts] is
       observed.
CONSTRAINT: the estimator is unbiased only if A is known, which requires randomization or a
       holdout band. Without it, A is replaced by the observed challenged fraud count K_pos, which
       already excludes the prevented cases. Since K_pos = O by construction, the naive estimator
       becomes 1 - O/K_pos = 1 - 1 = 0, so effectiveness is estimated as zero.
NUMERICAL EXAMPLE: suppose the band truly contains A = 4,000 would-be frauds and e = 0.65; then
       O = 4,000 * 0.35 = 1,400 and 2,600 frauds were prevented but never observed. A practitioner
       substitutes K_pos = 1,400 for A and computes 1 - 1,400/1,400 = 0, concluding that the
       challenge prevents nothing.
INTERPRETATION: the observational estimate is biased to zero effectiveness, and zero effectiveness
       sends q_chal = c_chal/(e*V) to infinity, which removes the challenge action and pushes the
       policy toward declining. That is the direction that harms customers rather than the issuer.
```

The practical consequence is that `e` must come from a randomized challenge on a band where it is
affordable to withhold the intervention, or from a prior with wide uncertainty carried through the
sensitivity table. It cannot come from the production ledger no matter how large that ledger is.

A second, subtler bias runs the other way. `K_pos` counts only frauds that were *later* recognized.
A fraudster who aborts at a challenge and re-attempts elsewhere may generate a confirmed fraud on a
different attempt, so the same underlying criminal event appears once as an aborted challenge and
once as a positive elsewhere. This inflates the apparent `K_pos` in other bands and deflates the
apparent effectiveness. Both biases are real and they do not cancel.

### Step 7 — Check the decision-time budget, not just the offline metric

The policy is only deployable if it fits the deadline. The constructed budget:

```text
Network and switch overhead          35 ms
Feature fetch at p99                 12 ms
Scoring at p99                        5 ms
Policy and reason-code assembly       3 ms
Serialization and return             22 ms
Total at p99                         77 ms
```

Against a 900 ms p99 allowance this leaves 823 ms of headroom, which is comfortable. Against a
40 ms allowance on a different rail it fails at every component and the design must move to
precomputed bands or in-switch rules. The budget must be expressed at p99 because the tail is what
produces timeouts, and a timeout is not a benign degradation: the authorization returns without a
decision and the issuer loses the opportunity to act on a potentially fraudulent attempt while
still paying the customer-relations cost of a failed purchase.

The queueing analogue matters for the asynchronous review path rather than the synchronous one. If
the policy generates 9,500 declines per thirty days and the disputes function reviews a sample of
them, the review queue is a `lambda * W` calculation: arrivals times handling time equals the number
of files in the system at steady state. At 25% sampling that is 2,375 files per month, and at
3.1 minutes per file with 6,300 staffed minutes per day, the implied utilization is
2,375 * 3.1 / (30 * 6,300) = 7,362.5 / 189,000 = 3.9%. Comfortable; the point is that it must be
computed rather than assumed, because a policy that triples declines triples this load.

**Takeaway:** the ledger closes to the unit and the improvement is 37.9%, but 81% of it comes from
challenging less, not from detecting more.

## 7. The Finding

The constructed candidate policy reduces mature cost from 115.69 to 71.89 per 1,000 eligible
attempts, a 37.86% paired reduction, driven mainly by removing 164,000 challenges that the
incumbent could not price. The fraud-loss term falls by 25,585, or 9.3%, which is the part
attributable to better ranking. The remaining 88.3% of the improvement is a policy and cost
artefact, available to any team willing to price the actions.

Three conclusions follow, and the third is the one that constrains the programme.

First, the decision object is the band table, not the model. Two teams with the same scorer and
different band tables will produce different cost outcomes, and the spread between band tables can
exceed the spread between competing models. Evaluation must therefore hold the score fixed and
vary the bands, then hold the bands fixed and vary the scores, and report both.

Second, the incumbent's weakness was not discrimination but unpriced action. A rules engine that
challenges 4.2% of volume because a rule fires is not optimising anything; it is executing a policy
whose costs were never written down. This is typical and it means the challenger's first deliverable
should be the cost schedule and the action pricing, which requires no model at all.

Third, the finding rests on `V`, `c_dec`, `ab`, and `e`, none of which is measured in this case.
Ranked by leverage on the answer: `V` first (it sets the entire loss term and the decline
threshold), `c_dec` second (it sets how much friction the policy may impose), `ab` third (it is the
largest single saving component and is also the most volatile cost), and `e` last (the fraud-loss
term is small). A programme that spends its measurement budget on `e` and estimates the other three
by assertion has inverted its own sensitivity table.

The honest status of this finding is `not-measured`. It is a constructed demonstration that the
arithmetic of the decision works and that the dominant term is not the one the business case will
lead with. It is not evidence that any portfolio would realize a 37.86% reduction.

**Takeaway:** price the actions before building the model; the pricing alone is worth more than the
ranking, and it is auditable by arithmetic rather than by experiment.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

The offline path has four jobs, each with a different cadence and a different failure consequence.

The **label assembly job** joins mature outcomes to attempt records. It runs daily, writes an
immutable snapshot per cohort day, and never mutates a historical label in place — corrections
arrive as new versions with a reason. Maturity is tracked explicitly: for each cohort week, the
fraction of attempts whose outcome has landed is plotted as a curve, and no week below 95% maturity
enters training.

The **snapshot builder** replays the point-in-time feature definitions against historical raw data
to produce training features. This is the only correct way to build features, because it applies
the availability rule mechanically instead of relying on an engineer's memory of which join is
safe. A feature that cannot be rebuilt from the snapshot is removed, not approximated.

The **training job** fits the scorer and the calibration layer on time-ordered folds, never random
folds. Random folds leak future information through card-level aggregates and through the fact that
the policy's own action history is stable across a random split but changing across a temporal one.

The **band calibration job** takes the calibrated probabilities and a signed cost schedule and
solves for the bands that minimize expected cost subject to the friction ceiling. This job is where
the policy version is produced, and it is deliberately separate from training so that a cost-schedule
change does not require a model retrain.

### 8b. Online / Serving / Real-Time Path

```text
[Authorization request]
        |
        v
[eligibility + schema validation] --ineligible--> [incumbent rules, logged]
        |
        v
[point-in-time feature fetch] --stale/error--> [incumbent rules, logged flag]
        |
        v
[scorer] -> [calibration] -> [band table] -> [action + reason code]
        |
        v
[durable decision write] --failure--> [return approved-with-review? no: block and log]
        |
        v
[respond: approve | challenge | decline]
        |
        +--> [async: outcome join when label lands]
```

Four design decisions in that path are worth defending explicitly, because each is a place where a
cheaper implementation would be silently wrong.

**Persistence before response.** The decision row is written before the action is returned. The
cost is latency; the benefit is that every action is reconstructable for the evaluation in section
9 and for dispute defence. An asynchronous write would allow a small window in which the issuer
acted but cannot prove what it acted on.

**Fallback is an action, not an exception.** When features are stale or the scorer errors, the
policy executes the incumbent rules and logs `fallback_reason`. A fallback is therefore a fourth
action with its own count and its own cost, and it must appear in the monitoring dashboard. A
policy that reports only approve/challenge/decline understates its own complexity.

**Unknown band is a hard error.** If a calibrated probability falls outside the band table — for
example because the calibration version changed mid-flight — the policy must fail rather than
default to approve. Defaulting to approve on a table error converts a configuration bug into
uncontrolled fraud exposure, and it will not be detected because approvals are not logged as
anomalies.

**Idempotency keyed on the attempt.** A retried authorization with the same network identifier and
the same attempt sequence returns the stored decision, not a fresh score. Without this, a merchant
retry after a slow response can receive approve then decline, which produces a customer-visible
contradiction and a dispute the issuer will lose.

### 8c. Key Design Decisions

| Decision | Alternative rejected | Reason | Residual risk |
|---|---|---|---|
| Three priced bands | Single fraud/no-fraud threshold | A challenge and a decline have different costs and different effectiveness | Band edges are unstable when the cost schedule changes |
| Calibrated probability, not raw score | Rank-only model | The bands are defined on probability, so scale must be meaningful | Calibration drifts with base rate |
| Synchronous decision persistence | Async write | Auditability and replay | Adds ~3 ms at p99 |
| Fail-closed on unknown band | Default approve | Prevents silent exposure | Availability risk during a bad deploy |
| Fallback to incumbent rules | Fallback to allow | Preserves fraud control | Inherits incumbent's weaknesses |
| Per-band `V` where measurable | Single global `V` | Ticket-size and liability heterogeneity | Requires more measurement |

### 8d. Architecture artifacts and operational readiness

The artifact set that must exist before a production cutover, stated as a checklist a reviewer can
audit: the signed cost schedule with an owner and date; the band table with a version identifier and
a change history; the feature register with availability semantics per column; the point-in-time
snapshot builder and its test suite; the replay harness and one successful full replay; the fallback
runbook with a named on-call owner; the rollback procedure and the time to execute it; the
monitoring dashboard with alert thresholds; and the model risk approval with its conditions listed
as testable statements rather than as prose.

The rollback procedure deserves a specific note. In an authorization path the fastest rollback is
not a model redeploy; it is a band-table swap or a switch-level flag that routes traffic to the
incumbent rules. The rollback must be rehearsed, and the rehearsal must include the case where the
new band table is corrupt, because that is the failure that a flag swap actually fixes.

**Takeaway:** the architecture is boring on purpose; every interesting choice in it is a place where
a cheap shortcut produces a silent, unlogged error.

## 9. Evaluation & Validation

The evaluation contract, stated so that it can be falsified:

| Element | Definition |
|---|---|
| Numerator | Sum of mature issuer-borne fraud loss, false-decline friction, challenge delivery, and abandonment cost |
| Denominator | All eligible authorization attempts assigned to the policy in the cohort |
| Population | Card-not-present purchase authorizations, one portfolio, one currency |
| Window | Thirty-day cohort, ninety-day outcome attribution |
| Comparator | Incumbent rules engine on the same attempts in the same window |
| Guardrails | False-decline rate, challenge rate, abandonment, complaint rate, timeout rate, subgroup decline parity |
| Uncertainty | Paired difference with a bootstrap interval over card-level clusters |
| Status | **Not executed.** No data, no model, no deployment. |

Guardrails are not reported as secondary metrics; each has a stop condition agreed in advance. The
challenge-rate ceiling is the clearest: if the candidate exceeds the signed ceiling it is rejected
regardless of its cost saving, because the ceiling encodes a customer-experience commitment that
the fraud programme does not own. The constructed candidate runs at 0.92% against a 4.26% baseline,
so the ceiling is not binding here, which is itself informative — the incumbent was far from the
ceiling, so the constraint that bound it was something else, probably rule authoring capacity.

Uncertainty on the primary comparison. Fraud counts are Poisson-like at the band level, so the
precision estimates carry binomial error:

```text
Candidate challenge precision p = 2,600 / 46,000 = 0.056522
Standard error  = sqrt(p * (1 - p) / n) = sqrt(0.056522 * 0.943478 / 46,000)
                = sqrt(0.0000011593) = 0.0010768
95% interval    = 0.056522 +/- 1.96 * 0.0010768 = [0.05441, 0.05863] i.e. [5.44%, 5.86%]
```

```text
Candidate decline precision p = 4,900 / 9,500 = 0.515789
Standard error = sqrt(0.515789 * 0.484211 / 9,500) = sqrt(0.000026295) = 0.005128
95% interval   = [0.50574, 0.52584] i.e. [50.57%, 52.58%]
```

These intervals describe sampling error in the constructed cohort and would exist in a real one.
They do not describe the error that matters most, which is model and label bias, and that error is
not quantifiable from this design.

Sample size for a prospective test. To detect a one-percentage-point difference in challenge-band
precision from a baseline of 5.65%, at 95% confidence and 80% power, the standard normal
approximation gives:

```text
n = 7.849 * p * (1 - p) / d^2 = 7.849 * 0.056522 * 0.943478 / 0.01^2
  = 7.849 * 0.053328 / 0.0001
  = 0.41857 / 0.0001 = 4,186 challenged attempts per arm
```

At the candidate's 46,000 challenges per thirty days, that is roughly 5.5 days per arm, which is
comfortable. But the test that matters is not precision; it is the paired cost difference, which
requires the full cohort and a mature outcome window, so the binding duration is the ninety-day
attribution, not the sample size.

Executed versus not executed, stated plainly:

| Check | Status | Consequence |
|---|---|---|
| Arithmetic closure of the ledger | Executed by hand and by script | Components add to the total exactly |
| Threshold derivation at three probabilities | Executed | Regions match direct cost minimization |
| Sensitivity to `V`, `e`, `ab` | Executed | Sign stable, magnitude varies 31.7% to 44.2% |
| Baseline measurement | Not executed | The 578,472 baseline is constructed, not observed |
| `e` estimation | Not executed | Identified only by randomization; assumption here |
| Latency at p99 | Not executed | Budget is arithmetic, not a load test |
| Subgroup decline parity | Not executed | Guardrail undefined without a protected-attribute policy |
| Causal effect of the policy | Not executed | Requires a holdout; observational ledger cannot supply it |

The last row is the one that a reviewer should press on. The ledger computes the cost of two
policies on the same cohort, which is a *counterfactual comparison*, not an experiment. The
candidate's band volumes were assigned by assumption; in a real test they would be realized only if
the policy were live, and the attempts it acted on would then behave differently from the attempts
the baseline acted on. That is why a shadow deployment establishes latency and data integrity but
never the cost difference, and why the only clean evidence is a randomized holdout with a
pre-registered analysis.

**Takeaway:** the arithmetic is verified, the evidence is absent, and the difference between those
two statements is the entire content of the evaluation section.

## 10. Action Taken

No action was taken. No model was trained, no policy was deployed, no customer was affected.

The proposed sequence, in an order chosen so that each step can kill the programme cheaply:

1. Write and sign the cost schedule: `V`, `c_dec`, `c_sms`, `c_ab`, `ab`, and the friction ceiling.
   This requires no model and it is the step most likely to reveal that the programme's assumed
   value is wrong.
2. Measure the incumbent: band volumes, band precisions, fallback rate, timeout rate, and complaint
   rate over the same thirty days. Without this there is no comparator and no baseline.
3. Measure `ab` on legitimate challenged attempts, segmented by channel, because abandonment is the
   largest single cost component and the one most likely to have changed.
4. Build the point-in-time snapshot builder and pass the availability audit for every candidate
   feature.
5. Run the offline band search against the signed schedule and compare candidate bands at *equal
   volume* as well as at equal cost, so that the volume effect and the ranking effect are separated.
6. Shadow the scorer to establish latency at p99 and data integrity. Do not interpret shadow results
   as a cost estimate.
7. If and only if steps 1–6 pass, run a randomized holdout on an eligible band with a
   pre-registered primary metric and a stop rule.

Step 5 contains the discipline that most programmes skip. Comparing bands at equal cost conflates
two effects; comparing at equal volume isolates ranking quality and shows whether the scorer earns
its place at all. In the constructed case the equal-volume comparison is uncomfortable: if the
candidate is forced to challenge 210,000 attempts, its precision advantage shrinks because it must
reach deep into the score distribution, and the incremental fraud caught is small relative to the
added delivery cost. The case does not compute that variant, and it should be computed before any
deployment decision.

**Takeaway:** the first three actions are measurement and paperwork; the model is step four.

## 11. Outcome & Results

`not-measured`. No production outcome exists. The constructed ledger's results are restated here as
arithmetic, not as findings about a business:

| Quantity | Baseline | Candidate | Difference |
|---|---|---|---|
| Total cost, cohort | 578,472.08 | 359,472.08 | -219,000.00 |
| Cost per 1,000 attempts | 115.69 | 71.89 | -43.80 |
| Uncaught fraud attempts | 3,760 | 3,410 | -350 |
| Prevented share of fraud | 62.4% | 65.9% | +3.5 pp |
| Declines | 11,000 | 9,500 | -1,500 |
| Challenges | 210,000 | 46,000 | -164,000 |
| Decline precision | 47.27% | 51.58% | +4.31 pp |
| Challenge precision | 0.762% | 5.652% | +4.89 pp |
| Challenge rate | 4.20% | 0.92% | -3.28 pp |

The prevented-share row is the only one that speaks to fraud control directly, and it moves by 3.5
percentage points while the cost moves by 37.9%. Those two numbers must be reported together.
Reporting only the cost improvement would imply the programme is a fraud-control success; reporting
only the 3.5 point gain would imply it is marginal. The truth is that most of the value is in
stopping an expensive habit, and habit changes are cheap to make and hard to justify.

A second honest limitation: the cohort is a single thirty days. Fraud is seasonal, and a policy
tuned on a peak season will over-challenge in a trough. Any real result must be shown on at least
four consecutive cohorts with the band table frozen, which is the cheapest possible out-of-time test
and the one most often omitted.

**Takeaway:** the cost improvement is real arithmetic and unreal evidence; the fraud-control
improvement is small and is the number that would actually be audited.

## 12. Failure Modes & What Went Wrong

Each failure below is stated as a mechanism with a discriminating check, because "monitor the model"
is not a control.

### F-01 — Post-decision outcome leakage through the label join

**Mechanism.** The label table is joined on card and date rather than on attempt identifier. A single
chargeback on a card then labels every attempt on that card in the window, including attempts made
after the chargeback was raised. The offline model learns "this card has a dispute in its future",
which is a real and available pattern in the training set and impossible at serving time.

**Symptom.** Offline AUC that does not reproduce in shadow; feature importance concentrated on
card-level dispute aggregates.

**Competing causes.** Genuine strong signal from card history; a serving-time bug producing different
features; a genuine population shift between training and shadow.

**Discriminating check.** Rebuild the label join keyed on attempt identifier only, and re-score. If
performance collapses, the leak was the join. Separately, for a sample of positive attempts, print
the decision timestamp next to the dispute-open timestamp; any positive whose dispute predates the
attempt is a labelling defect, not a signal.

**Correction.** Label at attempt grain; treat a card-level dispute as a property of the card's
history up to the decision time, not of the whole window.

**Verification.** A held-out cohort scored with the fixed join shows performance within the shadow
band, and no positive attempt has a dispute timestamp earlier than its authorization.

### F-02 — Challenge effectiveness estimated from the challenged set

**Mechanism.** `e` is estimated as one minus the fraction of challenged frauds that completed. But
frauds prevented by the challenge never appear as completed frauds, so they are missing from the
numerator by construction. The estimator is biased toward zero, which understates challenge value
and pushes the policy toward declining.

**Symptom.** The fitted band table places the challenge/decline crossover far lower than the cost
schedule implies, so the policy declines more than the arithmetic supports.

**Competing causes.** A genuinely low challenge effectiveness on this channel; a cost schedule that
mis-states `c_dec`; a channel where the challenge is often unavailable and silently degraded.

**Discriminating check.** Compare the estimated `e` against a randomized slice where challenge was
withheld. If the randomized slice shows materially higher completed-fraud rates for the same score
band, the observational estimate is biased low and the size of the gap is the bias.

**Correction.** Estimate `e` from the randomized slice, or carry it as a wide prior and report the
decision under the whole range rather than at a point.

**Verification.** The band table is stable across the plausible range of `e`; if it is not, the
decision is not identified and must not ship.

### F-03 — Attacker substitution makes the saved loss fictional

**Mechanism.** A decline on a fraudulent attempt is credited with `V` of prevented loss. The
fraudster re-attempts with a different card or merchant and succeeds. The ledger counts the saving
and the loss both, in different rows, and the net is zero or negative.

**Symptom.** Declining more does not reduce the portfolio's mature fraud rate; it reduces the
per-attempt rate and increases the attempt count.

**Competing causes.** Fraud migrating to a channel outside the policy's scope; a genuine reduction
hidden by volume growth; a labelling gap where the re-attempt lands after the attribution window.

**Discriminating check.** Measure at the *entity* level, not the attempt level: cluster attempts by
device, by merchant, and by known fraud-ring linkage, then compute mature fraud per entity. If
attempt-level loss falls while entity-level loss is flat, substitution is occurring.

**Correction.** Price the decline benefit at less than `V` where substitution is plausible, and
scope the policy to include the substitution channel or accept the reduced benefit honestly.

**Verification.** Entity-level mature loss shows a reduction consistent with the claimed saving.

### F-04 — Challenge band drifts as the base rate moves

**Mechanism.** Bands are defined on calibrated probability. Calibration is fitted on a cohort with a
0.20% base rate; a seasonal shift to 0.30% leaves the probabilities miscalibrated, so the same band
admits a different mix, and the challenge volume silently exceeds the friction ceiling.

**Symptom.** Challenge rate creeps above the signed ceiling while every individual score looks
normal.

**Competing causes.** A merchant-mix change; a new attack pattern the scorer has never seen; a
calibration-table deployment error; a change in the incumbent rules that altered the population.

**Discriminating check.** Plot observed fraud rate against predicted probability by score decile,
weekly. A calibration curve that tilts indicates base-rate drift and separates it from a ranking
failure, which would show as a flat or inverted decile ordering.

**Correction.** Recalibrate on the most recent mature cohort and re-derive the bands from the signed
schedule; do not hand-tune the band edges.

**Verification.** The weekly calibration curve returns inside its tolerance and the challenge rate
returns under the ceiling without a band change.

### F-05 — Fallback traffic becomes invisible

**Mechanism.** Feature staleness triggers the incumbent rules path. The dashboard reports only the
score-driven actions, so fallback volume is undercounted and the measured precision of the policy is
computed on a biased subset that excludes exactly the attempts where features were missing — which
correlate with new devices, new merchants, and unusual behaviour.

**Symptom.** Reported policy precision higher than the ledger implies; a persistent gap between the
number of authorizations and the number of scored decisions.

**Competing causes.** A genuine decline in fallback rate; a logging gap in the switch; duplicate
attempt records inflating the authorization count.

**Discriminating check.** Reconcile switch attempt count against decision-store row count by hour.
Any residual is fallback plus loss. Then compute precision separately for the fallback subset; if it
differs materially, the headline metric is mis-scoped.

**Correction.** Make fallback a first-class action in the reporting schema with its own reason code
and its own precision, and include it in the denominator.

**Verification.** Decision-store rows equal switch attempts for every hour, with fallback explicitly
counted rather than inferred.

### F-06 — Reason codes drift until the audit trail is unreadable

**Mechanism.** Reason codes are added by engineers as new rules appear, never retired when the
underlying rule is removed. Within a year the mapping from code to mechanism is undocumented, so a
dispute or a regulator cannot reconstruct why an attempt was declined.

**Symptom.** Growing number of codes, many with zero or near-zero volume, and a support team that
asks the model team what a code means.

**Competing causes.** A deliberate taxonomy expansion; a logging defect that mislabels codes; a
policy change that legitimately retired a mechanism.

**Discriminating check.** Produce a code-frequency table by month and a code-to-mechanism mapping.
A code that is retired but still appearing indicates a stale writer; a code with volume but no
documented mechanism is the audit failure itself.

**Correction.** Version the code taxonomy with the policy version, require a documented mechanism and
a retirement date for each code, and refuse a decline whose code has no mapping.

**Verification.** Every emitted code in a sampled day resolves to a documented mechanism and an
active policy version.

### F-07 — Latency tail produces an unmodelled fourth action

**Mechanism.** At p99 the feature fetch exceeds the budget. The switch returns without a decision,
the merchant sees a timeout, and the issuer both fails to apply fraud control and records the event
as a technical error rather than as an action. The cost of timeouts never enters the ledger.

**Symptom.** Timeout rate correlated with peak volume; fraud rate on timed-out attempts higher than
on approved attempts.

**Competing causes.** A network issue at the switch; a genuine traffic spike; a downstream dependency
that degraded without failing.

**Discriminating check.** Compute the realized fraud rate on timed-out attempts. If it exceeds the
population base rate materially, the timeout is a fraud channel, not a technical nuisance, and must
be priced as an approve.

**Correction.** Add a hard deadline inside the scoring path with a pre-warmed fallback, so that the
p99 path returns the incumbent rules decision instead of nothing.

**Verification.** Timeout rate below the agreed bound at peak load, and the fraud rate on any
remaining timeouts indistinguishable from the population.

**Takeaway:** every one of these failures is silent in the primary metric and visible only in a
deliberately constructed secondary check.

## 13. Transferable Lesson

The transferable structure is: a synchronous, irreversible, multi-way action under asymmetric costs,
where the label arrives late and only for the branch that was taken. That structure recurs
wherever a system must act before it can know.

Five claims transfer, and each is stated with its boundary.

**Price the actions before choosing the model.** The band table, not the scorer, determines most of
the cost. Boundary: this holds when the cost ratio between actions is large, which it is here (a
challenge costs about one thirteenth of a decline). Where two actions have similar costs, ranking
quality dominates and the band table is nearly irrelevant.

**Report the paired difference on a common cohort.** Levels are not comparable across portfolios;
differences are. Boundary: a paired difference computed on two different assignment mechanisms is a
counterfactual, not an experiment, and must be labelled as such.

**Precision in a low-cost, high-volume band will look terrible and is not a defect.** Boundary: it is
a defect when the action is expensive or irreversible, which is why decline precision is the one to
watch.

**Effectiveness of an intervention is not identifiable observationally.** Boundary: it is identifiable
where a holdout is cheap, which is true for a challenge and false for a decline.

**Fallback is an action.** Boundary: a fallback that is rare enough to be operationally irrelevant
still matters for the denominator, because its presence changes the population the metric describes.

The lesson does not transfer to batch decisions where the action can be revisited, because there the
irreversibility assumption — the one that drives the whole cost asymmetry — is absent. It also does
not transfer to a context where the loss bearer is not the decision maker, because then `V` is not
the decision maker's `V` and the pricing is wrong regardless of the model.

**Takeaway:** the reusable artefact is the cost-priced three-action policy, not the fraud model.

## 14. Connections & Cross-References

- [Claims investigation triage under delayed adjudication](./02_insurance_claims_fraud_detection.md)
  shares the evaluation-mismatch pattern under a much longer label delay and a human review
  capacity constraint rather than a millisecond deadline.
- [Account takeover challenge policy with incomplete identity signals](./03_account_takeover_application_fraud.md)
  shares the challenge-versus-decline structure but replaces the network deadline with an
  incomplete-signal problem and a recovery-path dependency.
- [AML alert prioritization at fixed investigator capacity](./04_aml_transaction_monitoring_system.md)
  is the capacity-bound variant: the binding constraint is analyst hours, not latency or friction.
- The unit cheat sheets define the fraud taxonomy and the decision-grain vocabulary this case
  assumes, and the cutoff-and-losses sheet carries the break-even derivations in general form.
- The local collection index records this case's stage and review trigger; the case does not assert
  any status beyond `proposed`.

The contrast that a reader should hold onto: this case's binding constraint is a *deadline*, the
claims case's is an *investigator hour*, the account-takeover case's is a *recovery path*, and the
AML case's is an *alert queue*. All four are the same arithmetic with a different scarce resource,
and the arithmetic transfers while the thresholds do not.

### 14a. Interview pack

**Q01. In one sentence, what decision does this case make, and what does it refuse to decide?**
Answer: it chooses approve, challenge, or decline per authorization attempt inside a friction
ceiling, and it refuses to decide whether a cardholder is a fraudster or whether a transaction is
criminal.
Reasoning: the action set is what the switch can execute within the deadline; anything outside that
set is not a decision this system can make, whatever the model outputs.
Follow-up: name the fourth action that exists in production but is missing from the action set.
Red flag: describing the output as a fraud probability to be thresholded at 0.5.
Score: 0 treats the task as classification; 1 names three actions; 2 adds the deadline and the
ceiling as constraints; 3 also identifies fallback as a fourth action.

**Q02. Derive the decline break-even probability and evaluate it at `c_dec = 13.56`, `V = 73.10`.**
Answer: 15.65%. Set `pV = (1-p)c_dec`, giving `p = c_dec/(c_dec + V) = 13.56/86.66`.
Reasoning: declining is certain friction against probabilistic loss, so it needs enough probability
to justify certain cost.
Follow-up: what happens to the threshold if `c_dec` quadruples? Answer: to 42.6%.
Red flag: computing `c_dec / V` = 18.6%, which ignores the friction incurred on the legitimate case.
Score: 0 wrong; 1 quotes 15.65% without derivation; 2 derives it; 3 also states the constant-`V`
validity condition.

**Q03. Why is the challenge threshold so much lower than the decline threshold?**
Answer: because a challenge costs 1.08 against a decline's 13.56 and still prevents 65% of the loss,
so it pays at 2.26% instead of 15.65%.
Reasoning: the break-even is cost divided by expected benefit, and the challenge has both a smaller
cost and a positive expected benefit.
Follow-up: what if `e` were 0.05 instead of 0.65? The threshold rises to 29.4%, above the decline
threshold, so challenge would be dominated and should be removed.
Red flag: saying the challenge is lower because "it is less disruptive" without the arithmetic.
Score: 0 qualitative only; 1 quotes 2.26%; 2 derives from `c_chal/(eV)`; 3 also finds the `e` at
which challenge becomes dominated.

**Q04. The baseline shows 0.762% challenge precision. Is the baseline broken?**
Answer: no. At a 0.20% base rate, challenging 4.2% of volume can only achieve low precision; the
band still carries a 3.8x lift over the population.
Reasoning: precision is bounded by base rate times lift; with a low base rate and a high-volume
cheap action, low precision is arithmetically unavoidable.
Follow-up: compute the precision you would need for the challenge band to break even on its own
cost using `c_chal/(e·V)`.
Red flag: concluding the incumbent is useless and skipping the lift computation.
Score: 0 declares it broken; 1 notes the base rate; 2 computes the lift; 3 also explains why
precision must be quoted with the action's cost.

**Q05. Decompose the 219,000 saving. Which component is the finding?**
Answer: 25,585 fraud loss (11.7%), 16,272 false declines (7.4%), 70,520 challenge delivery (32.2%),
106,623 abandonment (48.7%). The finding is that 81% comes from challenging less, not detecting
more.
Reasoning: the candidate's volume change dominates its ranking change.
Follow-up: what would the ranking effect be worth at equal challenge volume? Much less — the
fraud-loss term is the only one that reflects it.
Red flag: quoting 37.9% as evidence of model quality.
Score: 0 quotes the headline only; 1 gives the total; 2 decomposes; 3 identifies the volume/ranking
confound.

**Q06. Why can `e` not be estimated from the production ledger?**
Answer: because attempts prevented by the challenge never mature as fraud, so the observed
challenged-fraud count is `A(1-e)` and the naive estimator degenerates to zero effectiveness.
Reasoning: selection on the outcome.
Follow-up: what design identifies it? A randomized band where challenge is withheld.
Red flag: proposing to estimate `e` as one minus the observed completion rate.
Score: 0 accepts the observational estimate; 1 notes selection; 2 explains the algebra; 3 names the
randomized design and the bias direction.

**Q07. A colleague proposes a single global `V`. What breaks?**
Answer: ticket-size heterogeneity and liability shift. A small-ticket band has a low `V` and should
be approved more; a merchant where the issuer does not bear the loss has `V` near zero and should be
approved almost entirely.
Reasoning: the threshold is `c_dec/(c_dec+V)`, so `V` enters the denominator directly.
Follow-up: compute the decline threshold with `V` = 10: 57.6%, which means almost nothing should be
declined on that band.
Red flag: accepting the scalar without a stratification check.
Score: 0 accepts; 1 mentions variation; 2 quantifies the threshold effect; 3 also names liability
shift and substitution.

**Q08. What does the p99 latency budget actually constrain?**
Answer: the whole path, not the model. 12 ms of feature fetch inside a 40 ms allowance leaves no
room for a 5 ms score plus serialization, so the design must change even though the model is fast.
Reasoning: the deadline is a property of the composed path and the tail governs timeouts.
Follow-up: what is the fourth action a timeout creates, and how should it be priced?
Red flag: quoting model inference time alone as the latency answer.
Score: 0 ignores latency; 1 quotes p50; 2 reasons at p99; 3 prices the timeout as an approve.

**Q09. Why is synchronous decision persistence worth 3 ms?**
Answer: because without it the evaluation in section 9 cannot be replayed and disputes cannot be
defended. The 3 ms buys reconstructability.
Reasoning: the decision record is the only artefact proving what the policy knew and did.
Follow-up: what would you drop instead to recover the 3 ms?
Red flag: treating the audit record as a post-launch refinement.
Score: 0 skips it; 1 mentions logging; 2 connects to replay; 3 identifies a tradeable alternative.

**Q10. Design the falsification test for this case's headline claim.**
Answer: freeze the band table, run on four consecutive cohorts out of time, and require the paired
cost difference to hold with the guardrails intact; a single cohort is not a test.
Reasoning: seasonality and mix shift are the dominant confounders, and freezing the table isolates
them.
Follow-up: what would make you conclude the 37.9% is a baseline artefact? A baseline that had already
priced its actions.
Red flag: proposing a random cross-validation split.
Score: 0 no test; 1 one cohort; 2 out-of-time cohorts; 3 pre-registers the metric and the stop rule.

**Q11. The challenge rate must stay under a 1% ceiling. What changes?**
Answer: the challenge band is capped at 50,000 attempts, which forces the decline band to absorb the
overflow or accept uncaught fraud; the marginal attempt at the cap is more valuable as a decline than
as a challenge only above the 31.89% crossover.
Reasoning: a volume constraint binds the action mix, not the threshold alone.
Follow-up: compute the cost of capping challenges at 46,000 when the unconstrained optimum is 60,000.
Red flag: applying the band table and ignoring the ceiling.
Score: 0 ignores the ceiling; 1 reduces volume; 2 reasons at the margin; 3 computes the cap's cost.

**Q12. What does the ledger omit entirely?**
Answer: timeouts, appeal outcomes, chargeback representment recoveries, network fees on declines,
reputational effects, and any cost of the incumbent's manual rule maintenance.
Reasoning: an omission that is small in one direction and correlated with the decision is worse than
a large random omission.
Follow-up: which omission is most likely to reverse the sign? Network fees on declines, because they
scale with decline volume.
Red flag: claiming the ledger is complete because it closes arithmetically.
Score: 0 claims completeness; 1 names one omission; 2 names several; 3 ranks them by leverage.

**Q13. Two models have the same AUC. Can they produce different costs?**
Answer: yes, substantially, because the action bands consume probability mass in specific regions and
calibration differences move attempts across band edges without changing the rank ordering.
Reasoning: AUC is rank-based and threshold-free; the decision is neither.
Follow-up: name a metric that would capture the difference: cost at the chosen bands.
Red flag: using AUC to choose between policies.
Score: 0 no; 1 yes without mechanism; 2 explains calibration; 3 proposes the cost-based comparison.

**Q14. Why is the abandonment base `K - K_pos` rather than `K`?**
Answer: because a fraudster abandoning a challenge is not a customer-friction event; including those
attempts would inflate the friction term and bias the policy toward fewer challenges.
Reasoning: the cost term must be attached to the population that actually experiences the cost.
Follow-up: what happens to the candidate's abandonment count if the base is wrongly `K`? It rises
from 1,953 to 2,070 and the friction term rises by 1,680.
Red flag: using `K` because it is simpler.
Score: 0 wrong base; 1 right base; 2 explains the bias direction; 3 quantifies the error.

**Q15. A reviewer says the model is unnecessary because the cost schedule drives everything. Reply.**
Answer: partly right. In the constructed ledger 81% of the saving is volume and 11.7% is fraud loss,
so the schedule dominates; but the schedule can only be executed if a calibrated probability
identifies the marginal attempt, and the incumbent's rules cannot do that at this grain.
Reasoning: the pricing and the ranking are complements, with different leverage.
Follow-up: what is the cheapest way to test the reviewer's claim? Apply the priced bands to the
incumbent's own score, with no new model.
Red flag: either dismissing the reviewer or conceding the model is pointless.
Score: 0 takes a side without arithmetic; 1 concedes; 2 cites the decomposition; 3 proposes the
cheapest decisive test.

**Q16. What would you measure first if you had one week and a production dataset?**
Answer: band volumes, band precisions, and the realized abandonment rate on legitimate challenged
attempts, because those three turn the constructed ledger into a baseline and cost less than a day
of engineering.
Reasoning: the baseline and `ab` are the largest and most volatile inputs, and both are directly
observable.
Follow-up: what would you deliberately not measure in that week, and why? `e`, because it needs a
randomized design and a week cannot supply one.
Red flag: starting with model experimentation.
Score: 0 trains a model; 1 names a metric; 2 prioritizes by leverage; 3 justifies the deferral of `e`.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Decision | Approve, challenge, or decline each card-not-present authorization inside a friction ceiling. |
| Unit | One authorization attempt; retries are distinct attempts with the same decision. |
| Objective | Mature issuer-borne loss plus monetized friction per 1,000 eligible attempts. |
| Core thresholds | approve below 2.26%, challenge to 31.89%, decline above. |
| Ledger | `(P - D_pos - e·K_pos)V + (D - D_pos)c_dec + K·c_sms + ab(K - K_pos)c_ab`. |
| Constructed result | 578,472 → 359,472, a 37.86% reduction, 81% of it from lower challenge volume. |
| Weakest input | `e`, unidentified without randomization; least leverage on the answer. |
| Largest input | `V`, then `c_dec`, then `ab`. |
| Binding constraint | p99 latency on the path, and the signed friction ceiling. |
| Never infer | That a prevented challenge is a prevented fraud, or that a decline's saving is realized. |
| Evidence | Illustrative arithmetic only; `not-measured`; no deployment observed. |
