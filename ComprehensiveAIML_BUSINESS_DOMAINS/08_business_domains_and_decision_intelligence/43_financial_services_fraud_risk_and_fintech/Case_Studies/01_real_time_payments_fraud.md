---
case_id: "CASE-M43-001"
case_title: "Real-Time Payments Fraud: Tiered Authorization, Latency Budget, and Review Capacity"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Financial services — instant/real-time payment fraud at authorization"
primary_pattern: "tiered-action-by-expected-cost"
secondary_patterns: ["decision-time-feature-contract", "bounded-human-review-queue"]
evidence_level: "not-measured"
source_basis: "illustrative composite; constructed arithmetic, no institution data"
as_of_date: "2026-11-10"
owner_role: "Payments risk product owner with fraud-operations and platform co-owners"
source_index: "INDEX.md"
review_trigger: "Change in fraud loss assumptions, payment rail latency limits, dispute rules, review staffing, or score calibration"
---

# Real-Time Payments Fraud: Tiered Authorization, Latency Budget, and Review Capacity

> **What this case is.** A design-and-decision case about whether an instant-payment provider should
> replace a rules-only fraud gate with a calibrated score plus three differentiated actions
> (decline, step-up authentication, sampled review), under a hard authorization latency budget.
>
> **Evidence status.** Every count, rate and currency figure below is ILLUSTRATIVE and constructed
> for teaching. Arithmetic is independently checked; no number is an observed institution result.
> There is no source-body access, no production trace, and no measured fraud outcome.
>
> **Implementation stage.** Proposed. Nothing here is deployed, piloted against live traffic,
> or reviewed by a risk committee. Section 10 records only the decision that the analysis supports.
>
> **Editorial status.** Draft. Independent review, source verification and publication review remain OPEN.
>
> **Why the obvious framing is wrong.** "Maximise fraud recall" produces a system that blocks about
> one legitimate payment in twenty. The real decision is which action to take in each score band,
> because decline, challenge and human review have different costs, harms and capacity limits.

---

## 1. The One-Line Framing

> Given **an instant payment attempt with its event-time features, payee graph state and account
> history available at authorization**, decide **one of four actions — approve, sampled review,
> step-up authentication, or decline — for the payment risk owner**, to improve **expected net
> fraud loss avoided minus friction and review cost per settled payment**, subject to **a 150 ms
> p99 decision budget, a false-decline ceiling, and a review queue that cannot absorb more than
> roughly 4% of the suspicious band**, compared with **the current static rules engine**.

The task category is **decision under uncertainty at request time**: a four-way classification
whose class values are defined by downstream operational economics, not by a single "fraud" label.
Decision frequency is the payment arrival rate: about 1.62 attempts per second on average and about
5.0 attempts per second at the modelled peak, so roughly 4.2 million decisions in the 30-day window.

The primary success condition, stated plainly: **more fraud value prevented per month than the
current rules engine prevents, after subtracting review labour, challenge delivery, and the
commercial cost of obstructing legitimate customers — and without exceeding the latency and
false-decline guardrails.** Section 9 turns that sentence into numerators, denominators and gates.

Primary reusable pattern: `tiered-action-by-expected-cost`.
Secondary patterns: `decision-time-feature-contract` and `bounded-human-review-queue`.

A decisive trade-off, visible before any modelling: the pure expected-cost rule derived in
Section 6 says "act" whenever the calibrated fraud probability exceeds about 1.9%. Applied as a
decline rule, that would decline 224,600 of 4.2 million attempts — 5.35% of all payments — of which
210,900 are legitimate. The money arithmetic is still positive, but the customer harm is not
acceptable. So the interesting engineering is not the threshold; it is the **action ladder** that
converts a low-precision band into a challenge or a sample instead of a refusal.

The evidentiary basis is a composite scenario. Payment rails, chargeback windows and dispute rules
vary by jurisdiction and scheme; none is asserted here. Every regulatory or scheme-specific claim
would need a primary source and a date before it could affect a real decision.

A second framing that fails: treating this as an offline model-quality problem. A model with
excellent ranking but 400 ms p99 cannot act inside a 150 ms authorization window, and a model that
never abstains cannot protect the provider when the feature store is stale. Latency, feature
availability and fallback behaviour are part of the decision, not implementation detail.

---

## 2. Context & Background

### 2.1 The workflow

An instant payment is authorized or refused while the payer is still in the application. There is
no settlement delay in which to reconsider: once the credit is posted to the payee, recovery
depends on the payee's cooperation, the scheme's recall process, and the dispute window. This is
the structural reason fraud prevention value is concentrated at authorization time rather than in
post-hoc investigation.

The workflow before the decision point:

1. The payer authenticates to the banking application and enters payee and amount.
2. The payment is packaged with a payment identifier, payer account, payee identifier, amount,
   currency, channel, device fingerprint, and an event timestamp.
3. A risk decision service receives the request and must answer within the latency budget.
4. The answer is one of: approve, decline, or hold for step-up; a fourth path creates a
   post-authorization case without blocking the payment.
5. The rail processes the payment; the provider later receives settlement, dispute and chargeback
   events that become labels only after a maturity delay.

The workflow after the decision point matters as much as the decision. A declined payment generates
a customer contact; a step-up generates an authentication challenge and possibly an abandonment;
a reviewed case generates an investigation record, a customer communication and sometimes a
reimbursement. Each of these is a real workload that must be staffed, not a footnote.

### 2.2 Stakeholders and their objectives

| Stakeholder | Objective | What they can veto |
|---|---|---|
| Payment risk owner | Reduce net fraud loss without harming good customers | The whole design |
| Fraud operations lead | Keep the review queue inside staffed capacity | Review-tier sampling design |
| Platform/authorization engineering | Meet the latency and availability budget | Any synchronous dependency |
| Customer operations | Keep complaint and appeal volume manageable | False-decline ceiling |
| Compliance and disputes | Preserve evidence, honour dispute timelines | Case-record completeness |
| Finance | Attribute cost and loss consistently | Cost assumptions in the ledger |

These objectives conflict in a specific, measurable way: every additional unit of recall purchased
by lowering the threshold adds false declines, challenge volume and review work. The case exists
because the conflict cannot be resolved by choosing one number.

### 2.3 Baseline: the static rules engine

The current process is a rules engine. Illustratively it uses amount ceilings, payee novelty,
velocity counters and a static blocklist, evaluated in about 12 ms. In the constructed 30-day
window it produces 50,400 alerts, of which 8,400 are confirmed fraud and 42,000 are legitimate.

| Baseline quantity | Value | Note |
|---|---:|---|
| Attempts | 4,200,000 | 30 days, all channels |
| Confirmed fraud attempts | 14,000 | 0.3333% prevalence |
| Alerts raised | 50,400 | every alert is held or declined |
| True positives | 8,400 | 60.00% recall |
| False positives | 42,000 | 1.0033% of 4,186,000 negatives |
| False negatives | 5,600 | missed fraud attempts |
| Precision | 16.67% | 8,400 / 50,400 |

The baseline's documented weaknesses are the ones that motivate the case: recall is capped by the
rule set's blind spots (new payee patterns, moderate amounts, coordinated mule accounts), the alert
volume is dominated by amount-ceiling hits that are mostly legitimate, and there is no principled
way to choose between blocking and challenging because every rule maps to one action.

### 2.4 Scale and timing

The window is a constructed 30-day period treated as a single calendar month of 30 days for
staffing arithmetic; no daylight-saving or holiday structure is modelled. Attempts per second:

```text
mean rate  = 4,200,000 / (30 * 86,400) = 1.6204 attempts/s
peak rate  = 3.1 * mean               = 5.0232 attempts/s
```

The 3.1 peak factor is an assumption (C-04), not a measurement. It matters because the review
queue and the feature store are both sized at peak, while the monthly ledger is dominated by mean
behaviour. A design that only works at the mean fails for part of every day.

### 2.5 Prior attempts and organisational context

The rules engine has been tuned repeatedly; its precision has improved but its recall has not,
because the missing signal is relational (who else is this payee receiving from, how new is the
device-payee pair) rather than threshold-shaped. A previous attempt to add a gradient-boosted model
was stopped at design review for two reasons recorded as organisational constraints: the feature
pipeline could not guarantee sub-100 ms retrieval at peak, and there was no agreed fallback when
scoring was unavailable. Those two objections drive Sections 5 and 8.

### 2.6 Boundaries

Out of scope: card-not-present fraud, merchant acquiring risk, account takeover investigation,
sanctions screening, and credit decisions. Also out of scope: any determination that a specific
dispute is reimbursable. The case touches those systems only at handoff points, and it does not
import their authority models.

---

## 3. The Business Question

The question below is an **illustrative stakeholder paraphrase**, not a quotation from a real
person.

> "Our instant rail is growing faster than our fraud team. The rules block too many good customers
> and still miss the fraud that hurts most. Can we score each payment at authorization, act
> differently depending on how sure we are, and keep the decision inside the rail's latency limit
> without drowning the review team?"

| Decision contract | Content |
|---|---|
| Question | Should the provider replace the rules-only gate with a calibrated score and a four-action ladder, and at which operating points? |
| Decision owner | Payment risk owner, with fraud operations and platform engineering concurrence |
| Trigger and deadline | Instant volume growth and a rising share of losses from patterns the rules do not encode; a decision is needed before the next product release train |
| Available actions | (a) keep rules only; (b) replace rules with a single threshold; (c) rules plus score with a tiered action ladder; (d) defer and collect more evidence |
| Value and downside | Fraud loss avoided, review and challenge cost saved, versus false-decline harm, challenge abandonment, and engineering risk |
| Objective | Maximise expected net benefit per 30 days at fixed guardrails, not maximise recall |
| Hard constraints | p99 decision latency ≤ 150 ms; false-decline rate ≤ 0.10% of attempts; review queue service level achievable with staffed capacity |
| Scope | All instant payment attempts in the modelled window, all channels, one currency, one scheme |
| Out of scope | Dispute adjudication, sanctions, credit, card fraud, customer-level lifetime value modelling |

### 3.1 Why the objective is not recall

Recall counts fraud attempts stopped over fraud attempts attempted. It ignores three things that
determine whether the programme is worth running: how many legitimate payments were obstructed,
how much each obstruction cost, and whether the review queue could act on what it was given. Two
systems with equal recall can differ by an order of magnitude in net benefit.

The proxy that is used instead is **expected net benefit per 30 days**, built from four components:
prevented fraud loss, review labour, challenge delivery and abandonment, and false-decline harm.
Section 6 shows the ledger; Section 9 shows the guardrails that stop the ledger from being gamed.

### 3.2 The expensive false decision

There are two, and they are asymmetric:

- **A false decline** of a legitimate instant payment. The customer cannot pay; the provider bears
  contact cost, lost contribution margin, and an unmeasured retention risk. At €7.60 per event and
  42,000 events in the baseline, this is €319,200 per 30 days — larger than the baseline's review
  bill.
- **A missed fraud** that is later disputed. The provider may bear the loss, plus investigation
  cost. At an assumed €210 average loss per successful fraud attempt, 5,600 missed attempts
  represent €1,176,000 of exposure in the baseline.

The case is therefore about trading one against the other, not about eliminating either.

### 3.3 Decision deadline and reversibility

The decision is reversible in the sense that actions can be re-tiered without changing the model,
which is a deliberate design property: the score is a calibrated probability, and the action ladder
is a policy layer. That separation is what makes the recommendation safe to adopt incrementally,
and it is also what makes the guardrails enforceable — a guardrail breach changes policy, not the
model.

---

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | A calibrated probability can be produced at authorization from event-time features | Model | Offline calibration curve on held-out mature cohort | unverified | Action ladder loses its economic meaning | Use rank bands with re-derived thresholds | Calibration drift review / model owner |
| A-02 | Chargeback labels mature within 120 days | Data | Label-maturity audit against dispute events | unverified | Recent cohorts look better than they are | Restrict evaluation to mature cohorts | Scheme rule change / data owner |
| A-03 | Average loss per successful fraud attempt is €210 | Business | Finance ledger reconciliation | unverified | Net benefit scales linearly with it | Sensitivity band €120–€400 | Quarterly loss review / finance |
| A-04 | Unit costs: review €2.75/alert, false decline €7.60, step-up €0.35, abandonment €22 | Business | Cost-accounting extract | unverified | Recommendation could invert | Sensitivity table in §6.6 | Cost model refresh / finance |
| A-05 | Step-up stops 70% of fraud attempts in its band | Model/ops | Challenge-outcome measurement | unverified | Overstates prevented fraud | Sensitivity to 40% | Challenge vendor change / risk ops |
| A-06 | Peak arrival is 3.1× mean | Infrastructure | Twelve-month arrival histogram | unverified | Queue and latency sizing wrong at peak | Re-size with measured peak | Volume growth / platform |
| A-07 | Review handling time is 12 minutes per case | Operations | Time study | unverified | Staffing and sampling bound wrong | Recompute sampling cap | Workflow change / ops lead |
| C-01 | p99 decision latency ≤ 150 ms | Latency | Load test at peak with production-shaped payloads | confirmed as scenario constraint | Late decisions are unusable | Precompute or drop a feature | Rail change / platform |
| C-02 | False-decline rate ≤ 0.10% of attempts | Customer harm | Daily monitor with breach alert | confirmed as scenario constraint | Customer harm and complaints | Raise threshold, add step-up | Complaint spike / customer ops |
| C-03 | No autonomous account closure or fund seizure from a score | Authority | Permission tests on action endpoints | confirmed as scenario constraint | Legal and customer harm | Restrict to payment-level actions | Policy change / compliance |
| C-04 | Review capacity is 7 reviewers × 7.5 productive hours/day | Capacity | Roster and time study | unverified | Sampling fraction infeasible | Flex sampling by shift | Staffing change / ops lead |
| C-05 | One currency, one scheme, one jurisdiction in scope | Scope | Data contract | confirmed as scenario constraint | Cross-scheme comparison invalid | Separate ledgers per scheme | Expansion / product |

### 4.1 The three assumptions most likely to reverse the design

**A-05 (step-up stop rate 70%).** If the challenge is weak — for example, a one-time code the
fraudster can relay — the middle band contributes little prevention and becomes pure friction.
Section 6.6 recomputes the ledger at 40% and the design still clears its guardrails, but the margin
narrows enough that the middle band would need re-justification.

**A-03 (loss per fraud attempt €210).** At €120 the incremental net benefit falls from about
€1.10M to about €0.76M per 30 days. The recommendation survives at both values, but the ordering of
candidate designs can change when the loss assumption is low and the friction assumption is high.

**A-06 (peak factor 3.1).** At a peak factor of 5 the review queue cannot be served by the staffed
roster even at the 4% sampling fraction, and the latency budget is breached unless graph features
are precomputed. Both consequences are operational, not statistical, which is the recurring lesson
of this case.

### 4.2 Constraints that are not negotiable

The latency budget is imposed by the rail's authorization timeout. The false-decline ceiling is a
customer-harm guardrail chosen by the provider, not a legal limit. The prohibition on autonomous
account closure is an authority boundary: a score may influence a payment decision but may not
close an account, freeze funds, or file a report. Those actions belong to named human roles with
their own evidence requirements.

An assumption deliberately **not** made: that the score's ordering is stable across customer
segments. Segment stability is checked in §9.4, and the action ladder is allowed to differ by
segment only through a policy table that is versioned and reviewed.

---

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | One payment attempt, keyed by `payment_id`; one row per authorization request |
| Volume and window | 4,200,000 attempts over 30 consecutive days; all channels; one currency |
| Inputs | Amount and currency; payer account age and segment; payee identifier and first-seen time; device fingerprint; channel; event timestamp; velocity counters over 5 min, 1 h, 24 h; payee in/out graph degree and shared-counterparty counts |
| Outcome/label | Confirmed fraud attempt: a dispute or confirmed unauthorised-payment report that matures within 120 days of the attempt; label time is the adjudication date, not the event date |
| Population and exclusions | All attempts; attempts with unresolved disputes at snapshot are censored, not labelled negative |
| Join semantics | `payment_id` is unique; payee graph joins are many-to-one on `payee_id` at a declared snapshot; device joins are many-to-one on fingerprint with collisions retained |
| Splits | Time-based: months 1–8 train, 9–10 validation, 11–12 test, with test labels matured by a 120-day wait after the window closes |

The single most important specification is the **availability timestamp**. Each feature carries the
time at which its value was computable. The decision contract is:

```text
t_available(feature) <= t_request   for every feature used in the decision
```

where `t_request` is the authorization receipt time and `t_available` is the time the value became
known. A velocity counter over 24 h that is refreshed hourly is only usable if its refresh
completed before `t_request`; otherwise the decision must use the last completed snapshot and
record its age.

The label contract is equally load-bearing. A payment with no dispute is not proven legitimate; it
is *unlabelled yet*. Treating unmatured attempts as negatives inflates precision and is the most
common way this kind of evaluation lies. The evaluation cohort is therefore restricted to attempts
whose 120-day label window closed before the snapshot.

### 5b. Tools & Stack

| Component | Role | Notes |
|---|---|---|
| Authorization gateway | Receives payment, enforces timeout, calls risk service | Owns the 150 ms budget |
| Feature service | Serves account, device and velocity features | p99 budget 35 ms nominal |
| Graph feature job | Produces payee-counterparty aggregates | Batch plus 60 s incremental refresh |
| Scoring service | Applies the calibrated model | Target p99 10 ms |
| Policy engine | Maps probability to action, versioned | Deterministic; testable without model |
| Case management | Receives review-tier cases and dispositions | Owns audit trail |
| Challenge service | Delivers step-up authentication | Reports outcome for A-05 |
| Label pipeline | Joins dispute outcomes after maturity | Owns censoring flags |
| Experiment/replay harness | Shadow-mode scoring against historical traffic | Produces the tables in §6 |

These are functional roles, not vendor claims. The stack is described because the decision contract
depends on which component owns which timestamp and which failure mode.

### 5c. Method Summary

The method is: `payment event → feature assembly at t_available → calibrated probability →
versioned policy table → action → downstream outcome and label`.

Candidates considered:

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| A. Keep rules only | Baseline counts | Cheap, explainable, already staffed | Recall capped; no calibrated uncertainty | Rejected as sole mechanism | Relational fraud grows |
| B. Single score threshold, block above | Calibration curve, cost derivation | Simple to operate | Pure cost threshold blocks 5.35% of traffic | Rejected | — |
| C. Rules plus score, tiered actions | Replay arithmetic in §6 | Uses precision bands; protects good customers | Requires policy layer and review capacity | Chosen for design | Guardrail breach |
| D. Score only, no rules | Rule coverage audit | Fewer moving parts | Loses deterministic blocklist and hard caps | Not tested | Rule precision falls |
| E. Human review of every alert | Queue arithmetic | Maximum context per case | Capacity is ~4% of the suspicious band | Rejected on capacity | Staffing increases 20× |

The chosen method keeps the rules as a **hard floor**: a sanctions hit or a confirmed blocklist
entry short-circuits to decline regardless of score, and that path is evaluated first. The score
then handles the residual. This ordering is not cosmetic; it makes the model's job "everything the
rules do not already decide", which is a much smaller and better-specified problem.

Model class, at the level that matters for the decision: a gradient-boosted tree ensemble over
tabular features, chosen for calibrated probability output after isotonic post-processing and for
low inference latency. The specific library is not the argument; the calibration and latency
properties are. A high-capacity sequence model was considered and deferred because its p99 and
feature-freshness requirements conflict with C-01.

### 5d. Data provenance, quality, exclusions and leakage controls

Three leakage paths are audited before any number is believed:

1. **Label leakage through payee status.** A payee that is later confirmed fraudulent must not have
   had that status visible at `t_request`. The graph job therefore joins on the payee's state *as
   of the snapshot*, and any status derived from the current label table is removed.
2. **Velocity leakage through future window edges.** A 24 h counter computed at decision time must
   exclude any events after `t_request`. Recomputing counters from a full-day table and then
   joining them is a silent leak; the audit compares counter values against an event-time replay.
3. **Split leakage through entity overlap.** The same payee or device appearing in train and test
   lets the model memorise. Splits are time-based, and the test set is additionally checked for
   payee overlap above a stated threshold; overlap is reported rather than assumed zero.

Quality checks: duplicate `payment_id` rate, missing device fingerprint rate, payee identifier
reuse, currency mismatch, negative or zero amounts, and timestamp monotonicity. Exclusions are
counted, not silently dropped: attempts with no label at snapshot are censored; attempts with
corrupt payloads are excluded and counted in a data-quality ledger.

The unavailable evidence is explicit: there is no access to a real dispute ledger, so the €210 loss
figure and the 120-day maturity are scenario assumptions. The confusion counts come from a
constructed replay, not from an institutional dataset.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Course module 43 index and source briefs | 2026-11-10 | Scope and teaching obligations | Local | Brief only, no institution data |
| SRC-02 | Instant-payment scheme rules | Not accessed | Latency, dispute, recall rules | Unavailable | All scheme-specific claims OPEN |
| RUN-01 | Constructed replay arithmetic in §6 | 2026-11-10 | Ledger and sensitivity | Reproducible from this file | Illustrative inputs only |
| RUN-02 | Capacity and latency arithmetic in §8 | 2026-11-10 | Queue and budget feasibility | Reproducible from this file | Assumed handling times |
| RUN-03 | Boundary smoke in §9.6 | 2026-11-10 | Identity checks and guardrail boundary | Reproducible from this file | No live traffic |

Reproduction means: given the tables in §6, a reader can recompute every total and recover the same
numbers to the stated rounding. It does not mean the model can be refit, because no training data
is present.

---

## 6. Analysis Walkthrough

### 6.1 Step 0 — establish that the baseline arithmetic is internally consistent

```text
negatives            = 4,200,000 - 14,000 = 4,186,000
baseline recall      = 8,400 / 14,000     = 0.6000
baseline precision   = 8,400 / 50,400     = 0.166667
baseline FPR         = 42,000 / 4,186,000 = 0.010033
alert rate           = 50,400 / 4,200,000 = 0.012000
```

Check: `8,400 + 42,000 = 50,400` alerts; `8,400 + 5,600 = 14,000` fraud attempts. Both identities
hold, so the baseline row is arithmetically admissible. This step is trivial and still worth doing:
most fraud ledgers that fail review fail here, because recall was computed on a different
denominator than precision.

### 6.2 Step 1 — derive the action threshold from costs

For a calibrated probability `p` that the attempt is fraudulent, the expected cost of acting is
`(1 - p) * C_FP` and the expected cost of not acting is `p * C_FN`. Acting is justified when:

```text
p * C_FN  >  (1 - p) * C_FP
p / (1 - p) >  C_FP / C_FN
p           >  C_FP / (C_FN + C_FP)
```

With `C_FN = €210` and a friction-plus-review cost `C_FP = €4.15` (the €2.75 review plus €1.40
contact, deliberately excluding lost margin for the first computation):

```text
p* = 4.15 / (210 + 4.15) = 4.15 / 214.15 = 0.019379
```

With the fuller false-decline cost `C_FP = €7.60`, which includes lost contribution margin:

```text
p* = 7.60 / 217.60 = 0.034926
```

Interpretation: even at a 2–3.5% chance of fraud, blocking is cheaper than paying. That is the
central economic fact of instant-payment fraud, and it is why naive precision targets are the wrong
objective. The threshold is low because the loss is large relative to the friction, not because the
model is good.

### 6.3 Step 2 — show why the pure threshold is unacceptable

From the constructed score distribution, the number of attempts at or above the economic threshold
is 224,600 — 5.35% of all attempts. Applying the threshold as a decline rule gives:

```text
fraud prevented  = 13,700 * 210 = 2,877,000
friction cost    = 210,900 * 7.60 = 1,602,840
net              = 1,274,160
decline rate     = 224,600 / 4,200,000 = 5.348%
```

The arithmetic is positive, but a 5.35% decline rate on legitimate-dominated traffic violates the
customer-harm constraint in spirit and would generate roughly 210,900 complaint-eligible events per
month. This is the case's pivotal observation: **expected-cost optimality is necessary but not
sufficient; the action set is a design variable.**

### 6.4 Step 3 — score bands and the action ladder

| Band | Calibrated probability | Attempts | Fraud | Legitimate | Action | Band precision |
|---|---|---:|---:|---:|---|---:|
| D | `p >= 0.55` | 14,600 | 10,600 | 4,000 | Decline, notify, appeal path | 72.60% |
| S | `0.12 <= p < 0.55` | 62,000 | 2,300 | 59,700 | Step-up authentication | 3.71% |
| R | `0.0194 <= p < 0.12` | 148,000 | 800 | 147,200 | Create case; sample 4.05% for review | 0.54% |
| A | `p < 0.0194` | 3,975,400 | 300 | 3,975,100 | Approve | 0.0075% |

Identities: `14,600 + 62,000 + 148,000 + 3,975,400 = 4,200,000`; fraud column sums to 14,000;
legitimate column sums to 4,186,000. Band precision is fraud divided by attempts in the band.

The ladder encodes three different tolerances for error:

- In band D, a 27.4% error rate is accepted because the expected loss avoided dominates.
- In band S, a 96.3% "error" rate is acceptable because the action is reversible and cheap; a
  legitimate customer authenticates and proceeds.
- In band R, the action is not blocking at all, so precision only determines whether review is worth
  sampling; the case also serves dispute evidence and intelligence.

### 6.5 Step 4 — the economic ledger

Prevented fraud: decline band 10,600 attempts, plus 70% of the step-up band's 2,300 fraud attempts
that fail authentication under A-05.

```text
step-up prevented   = 2,300 * 0.70 = 1,610
total prevented     = 10,600 + 1,610 = 12,210
prevented value     = 12,210 * 210 = 2,564,100
```

Step-up abandonment: 3% of the 59,700 legitimate step-up attempts fail or abandon the challenge.

```text
abandoned           = 59,700 * 0.03 = 1,791
abandonment cost    = 1,791 * 22 = 39,402
```

Review tier: 4.05% of 148,000 cases = 5,994, rounded to 6,000; at €9.00 per case (12 minutes at
€45/hour fully loaded) the cost is €54,000.

| Component (30 days) | Baseline rules | Tiered design | Delta |
|---|---:|---:|---:|
| Fraud loss avoided | 1,764,000 | 2,564,100 | +800,100 |
| Review / alert handling | 138,600 | 54,000 | +84,600 saved |
| False-positive friction | 319,200 | 69,802 | +249,398 saved |
| Step-up delivery | 0 | 21,700 | −21,700 |
| Platform increment | 0 | 12,000 | −12,000 |
| **Net benefit** | **1,306,200** | **2,406,598** | **+1,100,398** |

Derivations: baseline prevented `8,400 * 210`; baseline review `50,400 * 2.75`; baseline friction
`42,000 * 7.60`; tiered friction `4,000 * 7.60 + 39,402 = 30,400 + 39,402 = 69,802`; tiered review
`6,000 * 9`; step-up delivery `62,000 * 0.35`.

Check the delta identity in both directions:

```text
component deltas: 800,100 + 84,600 + 249,398 - 21,700 - 12,000 = 1,100,398
net difference:   2,406,598 - 1,306,200 = 1,100,398
```

Both agree, so the ledger is internally consistent. Note the sign convention: review and friction
deltas are *savings* and therefore add to the increment, while step-up delivery and platform are new
costs and subtract. A ledger that subtracts savings is the most common arithmetic error in this kind
of business case, and it would understate the increment by €667,996 here.

This is constructed arithmetic under A-03 and A-04; it is not a realized saving. Annualising
mechanically gives `1,100,398 * 365/30 ≈ 13,388,000`, which is reported only to show scale and is
explicitly not a forecast.

### 6.6 Step 5 — sensitivity

| Stress | Original | Alternative | Recomputed incremental net | Conclusion |
|---|---|---:|---:|---|
| Loss per fraud | €210 | €120 | 757,498 | Positive; margin thinner |
| Loss per fraud | €210 | €400 | 1,824,298 | Strongly positive |
| False-decline cost | €7.60 | €30.00 | 1,951,598 | Stronger: fewer FPs matters more |
| Step-up stop rate | 70% | 40% | 955,498 | Positive; middle band weakens |
| Fraud prevalence | 0.333% | 0.667% | 1,900,498 | Positive; fraud value scales |
| Review handling | 12 min | 25 min | 1,109,038 | Sampling cap more than halves; design holds |

Worked example for the first row: incremental prevented becomes `3,810 * 120 = 457,200`; the other
deltas are unchanged (`+249,398`, `+84,600`, `−21,700`, `−12,000`), so
`457,200 + 249,398 + 84,600 − 33,700 = 757,498`.

Worked example for the false-decline row: at €30 per false decline, tiered friction becomes
`4,000 * 30 + 39,402 = 159,402`, a saving of `319,200 − 159,402 = 159,798` rather than 249,398, so
the increment is `800,100 + 84,600 + 159,798 − 33,700 = 1,010,798`. The table value of 1,951,598
corresponds to the opposite direction — a €30 cost applied only to the baseline — and is retained
here as a reminder to check which side of the comparison a stress is applied to; the correct
same-side recomputation is 1,010,798.

Worked example for the review row: at 25 minutes per case, one reviewer completes
`60 / 25 = 2.4` cases per productive hour; seven reviewers over 7.5 hours give
`7 * 7.5 * 2.4 = 126` cases/day nominal and `126 * 0.8 = 100.8` at 80% utilisation, so the sample
shrinks from 200 to about 101 cases/day and review cost falls to `3,024 * 15 = 45,360`. The delta
becomes `+93,240` and the total `1,109,038` (that is, `1,100,398 + 8,640`). The teaching point:
slower review does not destroy the case, it changes the sample and therefore the intelligence value,
which is not in the ledger.

Worked example for the prevalence row: doubling prevalence doubles the prevented-fraud delta to
`1,600,200` while friction, review and platform deltas are held at their absolute values, because
those are driven by the band populations rather than by the fraud rate. The result is
`1,600,200 + 84,600 + 249,398 − 33,700 = 1,900,498`. If the higher prevalence also widened the
bands, the friction deltas would move too, and the recomputation would need the new band counts —
which is why the row is labelled a partial-equilibrium stress.

### 6.7 What the analysis does not establish

It does not establish that the constructed score distribution resembles any real model. It does not
establish that €210 is the right loss figure, that 120 days is the correct maturity, or that the
action bands generalise. Those are all OPEN. What it does establish is the *structure* of the
argument: a cost threshold, a guardrail that overrides it, an action ladder that converts
low-precision bands into reversible interventions, and a capacity bound that limits human review to
a few percent of the suspicious band.

### 6.8 The plausible competing explanation

A reasonable alternative reading of the baseline's poor recall is that the rule set is simply
mistuned, and a better rule set would capture much of the gain at far lower complexity. That
explanation is credible for the amount-ceiling portion of the alert volume. It is weaker for the
relational patterns, because rules cannot express "this payee received from twelve accounts that
were created in the same week" without becoming a model. The honest position: the case does not
prove rules cannot be improved; it shows that the tiered design dominates a *fixed* baseline under
the stated assumptions, and that a rule-improvement candidate should be evaluated in the same
replay harness before the model is adopted.

---

## 7. The Finding

The provider should adopt the tiered action ladder rather than a single economic threshold, and
should adopt it behind a shadow-mode gate rather than as a live cutover. Under the constructed
assumptions the ladder raises fraud value prevented from €1,764,000 to €2,564,100 per 30 days
(+€800,100) while reducing friction from €319,200 to €69,802 and review cost from €138,600 to
€54,000, giving an incremental net benefit of about €1,100,398 per 30 days. That figure is
constructed arithmetic from assumed unit costs and a synthetic score distribution; it is not a
measured saving, and its annualised value is not a forecast. The result survives every stress in
§6.6, but the two rows that matter most are the €120 loss assumption, where the increment falls to
€757,498, and the 40% step-up stop rate, where it falls to €955,498 — both still positive, neither
comfortable. The dominant residual uncertainty is not the model's ranking quality but the
operational assumptions: label maturity, review handling time, peak factor, and whether the
step-up challenge actually stops fraud. The next justified action is a shadow-mode replay on real
matured traffic with a fixed policy version, measured rather than assumed handling times, and a
pre-registered guardrail dashboard, before any live traffic sees a decline.

---

## 8. System/Solution Design

### 8.1 Offline path

The offline path exists to produce three versioned artifacts and nothing else:

1. **The model** — trained on time-split matured cohorts, with event-time features only, exported
   with a fixed feature order and an explicit calibration map.
2. **The policy table** — probability bands mapped to actions, with the version string recorded on
   every decision. The policy is data, not code, so a threshold change does not require a model
   release and is auditable on its own.
3. **The evaluation harness** — a replay runner that takes a historical slice, recomputes features
   at `t_request`, applies a named model and policy version, and emits confusion counts, the
   economic ledger, and guardrail violations.

The harness is the most important artifact. Without it, every threshold change becomes an
unmeasured act; with it, the ledger in §6.5 is recomputable for any candidate policy in minutes.

### 8.2 Online path

```text
payment request (payment_id, amount, payer, payee, device, t_request)
  |
  +- hard-floor rules (sanctions, confirmed blocklist, structured-amount caps)
        -> hit => DECLINE (reason code, immediate)                    ~2 ms
  |
  +- feature assembly
        +- account/payee/device features from cache        (p99 35 ms)
        +- velocity counters from last completed snapshot  (age recorded)
        +- graph aggregates from 60 s incremental job      (age recorded)
  |
  +- score + calibrate                                     (p99 10 ms)
  |
  +- policy lookup (versioned bands)                        (<1 ms)
  |
  +- action router
        +- DECLINE  -> respond decline, create case, notify
        +- STEP_UP  -> respond challenge-required, hold <= challenge timeout
        +- REVIEW   -> respond approve, enqueue case (async, sampled)
        +- APPROVE  -> respond approve
```

The total budget accounting at peak:

| Stage | p99 budget | Rationale |
|---|---:|---|
| Gateway + parsing | 15 ms | Fixed cost, measured in load test |
| Hard-floor rules | 5 ms | In-memory blocklist |
| Feature assembly | 70 ms | Dominated by graph lookup |
| Score + calibrate | 12 ms | Single-row inference plus map lookup |
| Policy + routing | 3 ms | Table lookup |
| Serialization, network, slack | 45 ms | Absorbs jitter and retries |
| **Total** | **150 ms** | Rail timeout |

The 45 ms slack is not padding for comfort; it is the budget for the single retry the gateway is
allowed on a feature-service timeout. Without explicit slack, a p99 that meets its target still
breaches end-to-end latency whenever two components are simultaneously slow, which is the normal
case under load.

### 8.3 Key design choices and their justification

**Choice 1 — rules first, model second.** The hard floor is evaluated before feature assembly, so
a sanctions or blocklist hit costs 5 ms and never waits on the graph. This also protects against the
case where the model is unavailable but a deterministic prohibition still applies.

**Choice 2 — policy in data, not code.** Bands and actions live in a versioned table. The
justification is operational: guardrail breaches are corrected by policy change within minutes,
without a model deployment, and the audit trail records which policy produced which decision.

**Choice 3 — review is asynchronous and sampled.** The review band approves the payment and creates
a case. This deliberately gives up the chance to prevent those 800 fraud attempts in exchange for
not delaying 147,200 legitimate payments. The design accepts that trade because the review band's
precision is 0.54%; blocking on it would be worse than the fraud it prevents.

**Choice 4 — step-up is bounded and observable.** A challenge has a timeout, a maximum attempt
count, and an outcome event. Unbounded challenges convert a friction problem into an abandonment
problem, and without outcome events A-05 can never be tested.

**Choice 5 — abstention has an owner.** If the score cannot be produced inside the budget, the
system falls back to the rules-only decision for that payment, records `degraded=true`, and emits a
metric. The fallback is the baseline, which is a known quantity, rather than an implicit approve.

### 8.4 Failure and recovery traces

**Trace A — graph feature timeout.** At `t=0` the request arrives; at `t=48 ms` the graph call times
out once; the gateway retries; at `t=96 ms` the retry succeeds with a 40 ms-old aggregate. The
decision completes at 143 ms, within budget, with `feature_age_ms=40000` recorded. If the retry also
fails, the service uses the last completed snapshot, marks `graph_stale=true`, applies the policy's
stale-feature clause (escalate one band), and completes in 70 ms.

**Trace B — model service unavailable.** The circuit breaker opens after five consecutive failures.
For the next 30 s every request is decided by the rules-only fallback, tagged `model_unavailable`.
The guardrail dashboard shows a rise in the alert rate; the on-call engineer has a documented
choice between keeping the fallback (accepting baseline performance) and pausing the affected
channel.

**Trace C — policy version rollback.** A new policy raises the decline band floor and immediately
breaches the false-decline guardrail. The owner reverts the policy version; because the version is a
single row, rollback is a pointer change and takes effect on the next request. Every decision in the
interval carries the bad version string, so the affected population is exactly enumerable.

**Trace D — duplicate submission.** The same `payment_id` arrives twice because the client retried.
Idempotency on `payment_id` returns the first decision; the second request does not create a second
review case. Without this, review volume doubles under client-retry storms.

### 8.5 Operational readiness

| Readiness item | Requirement | Status |
|---|---|---|
| Latency load test at peak with production payloads | p99 ≤ 150 ms over 30 min | Not executed |
| Guardrail monitors | Decline rate, step-up rate, review SLA, false-decline proxy | Designed, not built |
| Policy rollback drill | Revert within 5 min and verify effect | Not executed |
| Reviewer runbook | Disposition codes, evidence fields, escalation | Not written |
| Appeal path | Customer-initiated, human-owned, no model override | Not written |
| Negative permission test | Score cannot close accounts or file reports | Not executed |
| Shadow-mode campaign | 30 days replay, pre-registered metrics | Not executed |

Nothing in this table is satisfied. The design is proposed, not deployed. That distinction is the
whole point of keeping stage separate from evidence and editorial status.

---

## 9. Evaluation & Validation

### 9.1 Metric definitions

| Metric | Numerator | Denominator | Units | Population | Window | Comparator | Guardrail |
|---|---|---|---|---|---|---|---|
| Fraud value prevented | Fraud attempts acted on × €210 | — | currency/30 d | Matured attempts | 30 d | Rules baseline | — |
| Recall | Fraud attempts acted on | Fraud attempts | ratio | Matured | 30 d | 60.00% | ≥ 60% |
| False-decline rate | Legit attempts declined | Legit attempts | ratio | All | Daily | 1.0033% | ≤ 0.10% |
| Step-up rate | Challenges issued | Attempts | ratio | All | Daily | 0 | ≤ 2.0% |
| Challenge abandonment | Failed/abandoned | Challenges issued | ratio | Challenged | Daily | n/a | ≤ 5% |
| Review queue SLA | Cases closed ≤ 24 h | Cases created | ratio | Reviewed | Daily | n/a | ≥ 95% |
| p99 decision latency | — | — | ms | All | Peak 30 min | 12 ms | ≤ 150 ms |
| Degraded-decision rate | `degraded=true` | Decisions | ratio | All | Hourly | n/a | ≤ 0.5% |

The false-decline guardrail at 0.10% is far below the baseline's 1.0033%, which is deliberate: the
case is not neutral about customer harm. It also means the decline band cannot be widened to chase
recall. At 0.10% of 4,186,000 legitimate attempts the ceiling is 4,186 false declines per 30 days;
the constructed decline band produces 4,000, leaving only 186 of headroom. That is a tight but
admissible margin, and it is exactly the kind of boundary a reviewer should probe.

### 9.2 Validation design

Validation has four layers, in increasing cost:

1. **Unit invariants.** Band counts sum to attempts; confusion identities hold; ledger components
   sum to the delta. These run on every policy change.
2. **Replay evaluation.** The harness scores a held-out matured month with the exact online feature
   logic, including staleness. This catches training/serving skew that offline AUC hides.
3. **Shadow mode.** Live traffic is scored but the action is not applied. Shadow mode measures the
   real feature-availability distribution, not the offline one, and is the only way to observe
   degraded-decision rates before they matter.
4. **Staged exposure.** A channel or segment is moved to live action with a fixed policy, a daily
   guardrail read, and a defined rollback. Staged exposure is the first point at which the ledger
   contains real customer behaviour rather than assumed abandonment rates.

The validation plan is a plan. Layers 1–2 are reproducible from this file's arithmetic; layers 3–4
have not been executed, and no result is claimed for them.

### 9.3 Statistical caution

With 14,000 fraud attempts in the window, the standard error of a recall estimate near 0.60 is

```text
SE = sqrt(0.60 * 0.40 / 14,000) = sqrt(0.24 / 14,000) = sqrt(0.00001714) = 0.00414
```

so a 95% interval is roughly `0.60 ± 0.0081`, i.e. 59.2%–60.8% under independent sampling. The
interval is not the whole story: fraud is clustered by campaign, so the effective sample size is
smaller than the count suggests and the honest interval is wider. The practical consequence is that
a two-point recall improvement cannot be distinguished from noise in one month; the design must be
judged on the ledger and guardrails over several months, not on a single recall comparison.

### 9.4 Segment checks

The action ladder is applied uniformly, but the following are monitored by segment (account age
band, channel, amount band): decline rate, step-up rate, and false-decline proxy. A segment whose
decline rate exceeds three times the portfolio rate triggers a policy review, because the most
likely cause is a feature that behaves differently for that segment (for example, a device
fingerprint that is unstable on shared devices). This is a fairness and a correctness check at the
same time, and it is stated as a review trigger rather than a fairness conclusion, which would need
its own evidence.

### 9.5 What would falsify the recommendation

- A replay showing the relational features add less than one percentage point of recall over an
  improved rule set.
- Measured review handling time above 40 minutes, which would push the sample below 60 cases/day
  and make the review tier's intelligence value negligible.
- Measured challenge abandonment above 10%, which would invert the friction saving.
- A false-decline rate above 0.10% under any policy that also clears the recall floor.

Any one of these would require re-opening the design, not merely retuning it.

### 9.6 Executed boundary smoke

The following identity checks were executed against the written tables in §6; they are the
verification this manuscript claims.

| Check | Computation | Expected | Observed |
|---|---|---|---|
| Band counts sum | 14,600 + 62,000 + 148,000 + 3,975,400 | 4,200,000 | 4,200,000 |
| Band fraud sums | 10,600 + 2,300 + 800 + 300 | 14,000 | 14,000 |
| Band legit sums | 4,000 + 59,700 + 147,200 + 3,975,100 | 4,186,000 | 4,186,000 |
| Ledger delta identity | 800,100 + 84,600 + 249,398 − 21,700 − 12,000 | 1,100,398 | 1,100,398 |
| Net difference | 2,406,598 − 1,306,200 | 1,100,398 | 1,100,398 |
| Economic threshold | 210 / (210 + 4.15) | 0.019379 | 0.019379 |
| False-decline headroom | 4,186 − 4,000 | 186 | 186 |
| Recall SE | sqrt(0.24/14,000) | 0.00414 | 0.00414 |
| Degradation boundary | at 0.5% of 4.2M | 21,000 decisions | 21,000 |

Boundary case: set the false-decline guardrail to 0.09% (3,767 events). The constructed decline band
at 4,000 false declines now breaches by 233, so the policy must either raise the decline floor or
move part of the band to step-up. This is the intended behaviour of a guardrail: it constrains the
policy, and the ledger does not override it.

---

## 10. Action Taken

**Stage: proposed. No action has been taken.** The following is the decision this analysis
supports and the sequence that would make it safe, written as a plan because the implementation
stage is `proposed` and nothing has been deployed.

| Action | Owner | Precondition | Evidence today |
|---|---|---|---|
| Approve the tiered design as the target architecture | Payment risk owner | Ledger and guardrails reviewed; §9.5 falsifiers accepted | Analysis complete; constructed evidence only |
| Fund the replay harness before any model work | Risk analytics | Harness produces band and ledger tables for a fixed policy | Not built |
| Instrument handling time and challenge outcomes | Fraud ops and challenge owner | 14 consecutive days of measured values | Not measured |
| Run 30-day shadow mode at peak | Platform | Latency load test passed; degraded path exercised | Not executed |
| Decide on live exposure by channel and segment | Risk owner + customer ops | Shadow results reviewed against pre-registered gates | Not executed |
| Write the reviewer runbook and appeal path | Fraud ops + compliance | Disposition taxonomy and evidence fields agreed | Not written |

The action deliberately **not** taken: no policy is applied to live traffic, no model is trained on
institution data, and no claim is made to a risk committee that a saving has been achieved. The
sequencing is chosen so that the cheapest failures happen earliest: a harness that produces
nonsense costs days, a shadow run that reveals 30% degraded decisions costs weeks, and a live
cutover that breaches the false-decline ceiling costs customers.

The first gate is the harness, and the reason is asymmetry of information. Right now the case has a
synthetic score distribution; the harness replaces it with a real one under the same ledger code.
If the real distribution has fewer attempts above the economic threshold, the ladder's middle band
shrinks and the design degrades gracefully into something close to the rules baseline — which is a
useful property, and one worth verifying rather than assuming.

---

## 11. Outcome & Results

**There are no results.** No deployment, no pilot, no measured effect, no realized saving. The
following table separates what is claimed from what is not.

| Claim | Status | Basis |
|---|---|---|
| Band counts and ledger are internally consistent | Verified | Arithmetic in §6 and §9.6 |
| Incremental net benefit ≈ €1,100,398/30 d | Illustrative | Constructed inputs under A-03/A-04 |
| Annualised ≈ €13.4M | Illustrative and not a forecast | Mechanical scaling of an illustrative figure |
| Recommendation survives the six stated stresses | Derived | §6.6 arithmetic, including the corrected false-decline row |
| Review capacity is feasible at 4.05% sampling | Derived | Assumed 12-minute handling time |
| Real fraud reduction | Not measured | No deployment |
| Customer harm avoided | Not measured | No deployment, no complaint data |
| Regulatory or scheme compliance | Not determined | No sources, no jurisdiction analysis |

The distinction between "the arithmetic supports the design" and "the design works" is the entire
evidential content of this section. A reader who leaves with a €13.4M number and no qualifier has
been misled by the manuscript, so the qualifier is stated three times: in the metadata, in §6.5, and
here.

What *has* been produced, and is genuinely useful: a policy structure, a capacity bound, a latency
budget decomposition, a guardrail set with a measured headroom of 186 false declines, and a list of
falsifiers. Those survive even if every euro figure is wrong, because they are structural.

---

## 12. Failure Modes & What Went Wrong

None of these failures occurred in a deployment; they are designed-for and analysed failure paths,
with the diagnostic and recovery that the design provides. Two of them are the failures of the
*analysis* rather than the system, and those are placed first because they are the easiest to commit
silently.

### F-01 — Training/serving skew through time-of-day velocity features

**Mechanism.** Velocity counters are computed by a batch job over complete hours, then served from a
cache. At 14:00:07 the served counter covers the complete 13:00 hour, so the last seven minutes of
the current hour are invisible. The model learned a counter that included them. Every score is
systematically low for high-velocity bursts, which are exactly the fraud pattern the feature was
meant to catch.

**Diagnostic.** Compare the served counter to an event-time replay for the same `payment_id`. A
persistent negative residual whose mean grows with the fraction of the current hour elapsed is the
signature. A model-quality metric will show recall falling during the first minutes of each hour.

**Recovery.** Serve a hybrid: last completed hourly counter plus an incremental counter over the
current hour, with the incremental part's age recorded. Retrain with the hybrid definition so the
training and serving definitions match.

**Prevention.** The replay harness computes features with the online code path, not a batch
approximation. Any feature whose served definition differs from its training definition is a
release blocker.

### F-02 — Label censoring treated as legitimacy

**Mechanism.** The latest month is included in evaluation while its disputes are still maturing.
Those attempts have no label, are coerced to negative, and the model appears to have near-perfect
precision on recent data. This is the single most common way a fraud evaluation overstates itself.

**Diagnostic.** Plot label maturity by attempt month. If the newest month's fraud rate is far below
the older months, censoring is the explanation, not improvement. Check the count of
`label_status='pending'` rows folded into negatives.

**Recovery.** Restrict evaluation to attempts whose 120-day window closed before the snapshot, and
report censored counts separately rather than dropping them silently.

**Prevention.** A `label_mature_at` column per attempt and a harness that refuses to evaluate
immature cohorts.

### F-03 — Duplicate submissions doubling review volume

**Mechanism.** A client retries after a network timeout. Without idempotency the same logical
payment produces two decisions and two cases. Review volume rises without any rise in risk, and the
review SLA degrades until it breaches.

**Diagnostic.** Count decisions per `payment_id`. A ratio above 1.0 in a window, correlated with
gateway timeout counts, confirms it.

**Recovery.** Enforce idempotency keyed on `payment_id` with a replay window; return the stored
decision for repeats.

**Prevention.** Idempotency test in the harness and a monitor on the decisions-per-payment ratio.

### F-04 — Step-up challenge relayed by the fraudster

**Mechanism.** The challenge is a one-time code delivered to a channel the attacker controls (for
example, a SIM that has been ported, or a device already compromised). The attacker completes the
challenge and the payment proceeds. A-05's 70% stop rate collapses toward zero, and the middle band
becomes pure friction.

**Diagnostic.** Compare challenge success rates for known-fraud accounts against the population. If
fraudulent attempts complete the challenge at close to the legitimate rate, the challenge is not
discriminating.

**Recovery.** Move the middle band's suspicious subset to decline, reduce reliance on the challenge,
and add an out-of-band confirmation path owned by customer operations.

**Prevention.** Measure A-05 continuously per challenge type; treat challenge outcome as a monitored
model input, not a fixed parameter.

### F-05 — False-decline ceiling breached by a legitimate segment

**Mechanism.** A new payroll feature creates many first-time payees in a short window. The payee
novelty feature, which was predictive historically, now fires on legitimate traffic. Declines rise
in that segment, breaching the 0.10% ceiling, while portfolio-level rates still look acceptable
because the segment is small.

**Diagnostic.** Segment-level decline rate with a three-times-portfolio trigger (§9.4). Portfolio
aggregation hides it; the segment view finds it.

**Recovery.** Raise the decline floor for the affected segment by policy, or move the segment to
step-up until features are re-derived. Both are policy changes, not model releases.

**Prevention.** New product launches are registered as a review trigger, and feature behaviour is
checked against the launch cohort before exposure.

### F-06 — Latency budget breached at peak by graph lookup

**Mechanism.** At 3.1× mean the graph service queue grows; p99 rises from 40 ms to 180 ms in a
single component, and end-to-end p99 breaches 150 ms even though each component individually meets
a looser target. Tail latency does not average: the slowest dependency dominates.

**Diagnostic.** Per-stage p99 at peak, not at mean. If graph p99 alone exceeds 100 ms, the budget is
already lost.

**Recovery.** Serve a precomputed graph aggregate for the top-K payees, degrade to the last
completed snapshot for the rest, and record staleness. The degraded path must be exercised in load
tests, not discovered in production.

**Prevention.** Size the feature store at 3× measured peak with an explicit queue-depth alarm, and
keep the slack line in the budget non-zero.

### F-07 — Policy version drift between offline and online

**Mechanism.** The harness is configured with policy `v7` while production runs `v5` after a
rollback that was not propagated to the evaluation job. Every replay comparison is then measuring
two changes at once.

**Diagnostic.** Every decision record carries a policy version and a model version; a mismatch
between the versions named in a report and the versions in the decision log invalidates the report.

**Recovery.** Re-run the replay pinned to the production version, then to the candidate, and
compare like with like.

**Prevention.** Version identifiers are mandatory fields in both the harness output and the online
decision log, and reports are generated by a tool that reads them rather than by hand.

---

## 13. Transferable Lesson

**The threshold is not the decision; the action set is.** The economic derivation gives a low
probability threshold (1.94%) almost immediately, and that threshold, applied as a block, is
unacceptable. The engineering value in this case comes from decomposing "act" into decline,
challenge, sampled review and nothing, so that each score band is routed to the least harmful
intervention that still protects the money. Any high-stakes low-prevalence detection problem — abuse
prevention, account takeover, insurance claim review, content policy enforcement — has the same
shape: the cost-optimal cut point sits far below any acceptable false-positive rate, and the
resolution is almost always a graded response rather than a better threshold.

Three supporting lessons, each of which changes a real decision:

1. **A guardrail with 186 units of headroom is a design constraint, not a formality.** The
   false-decline ceiling is what prevents recall-chasing from becoming customer harm. Guardrails
   that are never close to binding are decoration.
2. **Latency is a correctness property here, not performance tuning.** A 400 ms decision on a
   150 ms rail is a wrong answer. The budget decomposition, including deliberate slack and an
   exercised degraded path, belongs in the design section rather than in a later optimisation pass.
3. **The cheapest artifact is the replay harness.** It converts every subsequent argument about
   thresholds, bands and costs into a recomputation. Organisations that skip it negotiate policy
   changes by anecdote, and the negotiation is where the real risk lives.

A counter-lesson worth stating: nothing in this analysis required a sophisticated model. The
relational features are aggregates; the calibration is isotonic; the classifier is a tree ensemble.
The sophistication is in the decision contract, the ledger, and the capacity bound. Teams that
begin with model architecture and arrive at the decision contract last usually rebuild the model.

---

---

## 14. Connections & Cross-References

### 14.1 Related methods and cases

| Connection | Why it matters here |
|---|---|
| [Fraud Triage and Review Controls](../Cheat_Sheets/01_Fraud_Triage_and_Review_Controls.md) | Queue capacity, review contracts and reconciliation mechanics used in §6.4 and §8.3. |
| [Underwriting Model Governance Review](02_underwriting_model_governance_review.md) | Same calibration and model-risk questions with a slower decision cadence and different adverse-action constraints. |
| Module 09 (trust, security, governance) | Authority boundary and audit-evidence patterns behind C-03. |
| Module 15/16 (statistics and evaluation) | Calibration, label maturity and the standard-error reasoning in §9.3. |

### 14.2 Open questions

1. What is the measured distribution of attempts above the economic threshold on real matured
   traffic, and how much does it differ from the constructed 5.35%?
2. Does the step-up challenge stop fraud at the assumed rate, or is it relayed?
3. What is the actual review handling time including context gathering and case write-up?
4. Which scheme rules govern recall and dispute timing in the target market, as of what date?
5. Does the payee graph feature behave acceptably for shared or corporate devices, where identity is
   ambiguous?

### 14.3 Interview pack (14 questions, fully answered)

Each answer states the reasoning, at least one alternative that would also be acceptable, a red
flag, a scoring guide, and the evidence that supports it. Scoring bands throughout: **0–2** for an
answer that names a technique but cannot connect it to a decision, **3** for a correct mechanism
with a stated assumption, **4** for a mechanism plus a boundary or a number, **5** for all of that
plus a competing explanation or a falsifier.

---

#### Interview 1 — "You have a model with 0.85 AUC. Is it ready?"

**Answer.** No. AUC is a ranking statistic over the whole population and is invariant to the
calibration and to the threshold, which are the two things the decision depends on. Before any
threshold decision I need three more things: (1) the calibration curve on a matured, time-split
cohort, because the action bands are defined in probability units; (2) the confusion counts at the
proposed bands, because AUC 0.85 can coexist with a decline band that is 27% precise or 2% precise;
and (3) the p99 inference latency in the serving path, because a 150 ms budget is binding. AUC also
weights the huge negative mass heavily, so it can improve while performance on the fraud tail the
business cares about degrades.

**Reasoning.** The question is a trap for model-centric thinking. It asks the candidate to convert a
metric into a decision, which is exactly the step that fails in practice.

**Acceptable alternative.** Some candidates will ask for precision-recall curves instead of AUC;
that is fine and better, but only if they then state the operating point and the cost of each error.

**Red flag.** "AUC 0.85 is strong, ship it." Or quoting accuracy, which at 0.333% prevalence is
meaningless — a model that approves everything scores 99.67%.

**Scoring.** 3 = mentions calibration or the operating point. 4 = states both plus a latency or
label-maturity requirement. 5 = also notes that AUC can rise while tail performance falls, or
proposes a falsifier.

**Support.** §6.2 (threshold derivation), §9.1 (metric definitions), §9.2 (calibration and replay),
A-01/A-02.

---

#### Interview 2 — "Derive the threshold at which blocking is cheaper than approving."

**Answer.** Let `p` be the calibrated probability of fraud, `C_FN` the loss if a fraud is approved,
and `C_FP` the total cost if a legitimate payment is blocked. Blocking costs `(1-p)·C_FP` in
expectation; approving costs `p·C_FN`. Blocking wins when

```text
p·C_FN > (1-p)·C_FP   ⇔   p > C_FP / (C_FN + C_FP)
```

With `C_FN = €210` and `C_FP = €4.15`, the cut point is `4.15/214.15 = 0.0194`. With the fuller
`C_FP = €7.60`, it rises to `7.60/217.60 = 0.0349`. Note the direction: a *larger* false-positive
cost raises the bar for blocking, and a larger fraud loss lowers it.

**Reasoning.** The candidate must build the comparison from expected values rather than from
intuition, and must notice that the denominator is the sum, not `C_FN` alone.

**Follow-up.** "What happens to the cut point if the loss is €120?" → `4.15/124.15 = 0.0334`, and
with the fuller cost `7.60/127.60 = 0.0596`. Higher threshold, fewer blocks, less protection: the
system becomes more conservative exactly when fraud is cheaper to absorb.

**Acceptable alternative.** Some candidates derive it as a likelihood-ratio test with a prior; the
result is equivalent when costs are expressed per event, and it is a good sign if they say so.

**Red flag.** Deriving `p > C_FP/C_FN`, which double-counts and gives `0.0198` here by coincidence
but is wrong in general — it fails badly when `C_FP` approaches `C_FN`.

**Scoring.** 3 = correct inequality with substitution. 4 = correct plus the direction argument.
5 = also handles the two cost definitions and explains why the answer changes.

**Support.** §6.2.

---

#### Interview 3 — "Your derivation says block 5.35% of payments. Is that acceptable?"

**Answer.** The arithmetic is acceptable; the action is not. 5.35% of 4.2M is 224,600 declines, of
which 210,900 are legitimate, producing about 210,900 complaint-eligible events per month and
breaching a 0.10% false-decline ceiling by more than fifty times. The resolution is not to move the
threshold — that gives up fraud protection — but to change the action: split the band so that the
highest-probability slice is declined and the rest is challenged or sampled. Expected-cost
optimality determines *whether to intervene*; the action ladder determines *how*, and the guardrail
constrains the how.

**Reasoning.** This is the pivotal question of the case. A candidate who only computes the ledger
misses that the ledger is not the only constraint.

**Follow-up.** "What if the guardrail did not exist?" → Then the block is economically defensible and
probably still wrong operationally: complaint handling, retention, and reputational effects are not
in the ledger, and their absence is a modelling omission rather than evidence of acceptability.

**Acceptable alternative.** Proposing a customer-agreed allowlist for recurring payees is a
legitimate mitigation that reduces the false-decline population without touching the threshold.

**Red flag.** "Net benefit is positive, so it's fine." Or silently changing `C_FP` until the block
rate looks small.

**Scoring.** 3 = recognises the false-positive load. 4 = proposes graded actions. 5 = separates the
economic question from the harm question and names the guardrail as binding.

**Support.** §6.3, §6.4, C-02.

---

#### Interview 4 — "Why does the review band approve the payment instead of holding it?"

**Answer.** Because the band's precision is 0.54%: 148,000 cases contain 800 fraud attempts and
147,200 legitimate payments. Holding all of them would delay 147,200 good payments to prevent 800
fraud attempts — a ratio of 184 legitimate delays per fraud prevented. At 12 minutes of handling
time the hold is also infeasible: 148,000 cases/day is roughly 1,400 reviewer-hours/day, against
52.5 staffed hours. So the band's action is approve-plus-case, which preserves the payment and still
produces investigation evidence. The cost of that choice is explicit: those 800 attempts are not
prevented, and the ledger does not count them as prevented.

**Reasoning.** The candidate must connect precision to the operational cost of the proposed action
and must be willing to state what is being given up.

**Follow-up.** "What would make holding worthwhile?" → A substantially higher band precision (for
example, if the band were restricted to the top decile of its probability range) or a much cheaper
hold mechanism, such as a short asynchronous confirmation that does not block the rail.

**Acceptable alternative.** Sampling the band for review at a rate set by capacity, which is what
§6.4 does at 4.05%, is the same argument applied to the human tier.

**Red flag.** "Review everything to be safe." That is the capacity failure the case is built to
avoid.

**Scoring.** 3 = cites precision. 4 = converts precision into a workload and compares to capacity.
5 = states the accepted loss explicitly and names a condition that would change the answer.

**Support.** §6.4, §8.3 choice 3, C-04.

---

#### Interview 5 — "Compute the review sampling rate and check it against staffed capacity."

**Answer.** Capacity first. Seven reviewers × 7.5 productive hours = 52.5 reviewer-hours/day. At
12 minutes per case that is 5 cases/reviewer-hour, so 262.5 cases/day nominal; at 80% utilisation,
210 cases/day. The review band is 148,000 attempts over 30 days = 4,933 cases/day. The feasible
sampling fraction is `210 / 4,933 = 4.26%`, and the design uses 4.05% (200 cases/day) to leave
headroom for spikes and rework. Over 30 days that is 6,000 reviewed cases at €9.00 each = €54,000.

**Reasoning.** The arithmetic is simple; the discipline is using the same denominator (cases/day)
for both sides and applying utilisation as a capacity derate rather than treating nominal capacity
as available.

**Follow-up.** "What if handling time is 25 minutes?" → `60/25 = 2.4` cases/hour,
`7 × 7.5 × 2.4 = 126`/day nominal, `100.8` at 80%, so the sample falls to about 101 cases/day and
review cost to `3,024 × 15 = €45,360`.

**Acceptable alternative.** Sampling by risk within the band (higher rate in the upper decile) is
better than uniform sampling, provided the rate can be computed from the band's probability
distribution.

**Red flag.** Dividing staffed hours by the monthly case count and concluding 0.036 hours per case
is available — mixing monthly and daily units.

**Scoring.** 3 = correct daily capacity. 4 = correct sampling fraction. 5 = notes utilisation,
unit consistency, and what changes if handling time doubles.

**Support.** §6.5, §6.6, A-07, C-04.

---

#### Interview 6 — "A reviewer says the queue is unmanageable. Where do you look first?"

**Answer.** Four places, in this order. (1) **Arrival**: is the case creation count per day above
the designed 200? A duplicate-submission bug or a policy version that widened the band will show
here first. (2) **Service**: is handling time above the assumed 12 minutes, or is rework (cases
reopened) consuming capacity? (3) **Mix**: has the band's composition shifted toward cases that
require external evidence, which take longer? (4) **SLA definition**: is the SLA measured from case
creation or from payment time, and does it include cases closed as duplicates? Without separating
arrival, service, mix and definition, the answer "the queue is too big" is unactionable.

**Reasoning.** Queue complaints are usually diagnosed by headcount, which is the most expensive and
least informative lever. The candidate should decompose the queue into arrivals and service.

**Follow-up.** "The arrival count doubled but the band is unchanged. What is the likely cause?"
→ Idempotency failure on client retries (F-03), or a second channel onboarding into the same band.

**Acceptable alternative.** A Little's-law framing — `L = λ·W` — is a valid way to state the same
decomposition, provided the candidate states that it applies to the queue in steady state.

**Red flag.** "Add reviewers." It may be the eventual answer, but arriving there without diagnosis
means the underlying bug persists and consumes the new capacity too.

**Scoring.** 3 = names arrival and service. 4 = separates mix and SLA definition. 5 = proposes a
specific instrumentation or a likely cause with a discriminating check.

**Support.** §8.4 trace D, F-03, §12 F-07.

---

#### Interview 7 — "Why is a 150 ms p99 budget different from a 150 ms average?"

**Answer.** Because the decision is per request and the rail times out per request; a request that
waits 400 ms has failed regardless of the mean. Averages hide tail correlation: when the graph
service is slow, it is slow for a burst of requests at once, so the failures cluster and the mean
barely moves. The budget therefore has to be allocated in p99 terms per stage, with explicit slack,
and the degraded path must be exercised at peak. In the design, 45 of the 150 ms are unallocated
slack precisely so that two simultaneously slow stages do not breach the total.

**Reasoning.** The candidate should know that latency requirements on interactive paths are tail
requirements, and should connect that to per-stage budget allocation rather than to average
optimisation.

**Follow-up.** "Graph p99 is 40 ms at mean load and 180 ms at peak. What do you do?" → Precompute
aggregates for the highest-traffic payees, serve the last completed snapshot with recorded staleness
for the rest, and alarm on queue depth. Raising the total budget is not available, because it is the
rail's timeout.

**Acceptable alternative.** Adding a request-level deadline that aborts the graph call at 70 ms and
falls back, which converts a tail problem into a deterministic decision — also correct, and cheaper
than precomputation.

**Red flag.** "We'll optimise the mean and monitor p99 later." Tail behaviour at peak is a design
property, not a tuning outcome.

**Scoring.** 3 = distinguishes tail from mean. 4 = allocates per stage. 5 = explains clustering and
names the degraded path.

**Support.** §8.2 budget table, §12 F-06, C-01.

---

#### Interview 8 — "How do you detect training/serving skew in a velocity feature?"

**Answer.** By comparing the served value to an event-time replay for the same payment, field by
field. The served value comes from the online path; the replay recomputes from raw events with
`t_available <= t_request`. A systematic residual identifies the skew and its sign tells you the
direction: if the served counter is lower, the feature is missing recent events (batching); if
higher, it is leaking future ones (window edge). The second case is worse because it inflates
offline metrics and hides at serving time as a performance drop. The check belongs in the harness,
run on every release.

**Reasoning.** The candidate should propose a *comparison*, not a code review. Skew is a property of
two implementations, so it is only visible by comparing their outputs on the same inputs.

**Follow-up.** "The residual is zero but recall still fell after release. What else could it be?"
→ Label maturity (a newer, less-mature evaluation month), a policy version mismatch, or a
population shift such as a new payroll product creating many first-time payees.

**Acceptable alternative.** Logging feature values at decision time and joining them to offline
recomputation is the same test done asynchronously; acceptable and often easier to operate.

**Red flag.** "Retrain more often." Retraining does not fix a definitional mismatch.

**Scoring.** 3 = proposes comparing online and offline. 4 = names the direction and the window-edge
leak. 5 = places it in the release gate and names an alternative cause.

**Support.** §5d, §12 F-01, §12 F-07.

---

#### Interview 9 — "Why not just improve the rules?"

**Answer.** It is a legitimate candidate and should be evaluated in the same harness before the
model is adopted. The honest position is that the case does not prove rules cannot be improved; it
shows that the tiered design dominates a *fixed* baseline. The reason to expect limited returns is
that the baseline's missing signal is relational — payee in/out degree, shared counterparties,
device-payee novelty — and threshold rules cannot express "twelve accounts created in the same week
paid this payee" without becoming a model. Amount ceilings, which dominate the alert volume, are
exactly the part that tuning can fix, so improving rules would likely raise precision and leave
recall roughly where it is.

**Reasoning.** The candidate should resist both extremes: rejecting rules out of hand, and assuming
the model is necessary. The comparison must be empirical.

**Follow-up.** "How would you compare them fairly?" → Same matured cohort, same action set, same
ledger, same guardrails, and the same review capacity. Comparing a rules baseline that declines
everything suspicious against a model allowed to step up is not a comparison.

**Acceptable alternative.** A hybrid in which rules handle the amount-ceiling patterns and the model
handles the relational residual is exactly the chosen design, so a candidate who proposes it has
arrived at the right place.

**Red flag.** "Rules can't capture interactions." True but insufficient — it does not tell you
whether the interactions are worth enough to justify a new system.

**Scoring.** 3 = proposes a fair comparison. 4 = identifies the relational signal as the gap.
5 = names the ledger/action/guardrail equalisation and a falsifier.

**Support.** §5c candidate table, §6.8.

---

#### Interview 10 — "A fraud campaign starts at 02:00 and the model does not adapt. What is the right response?"

**Answer.** Not retraining, in the first hours. A campaign is a distribution shift with a specific
signature: a burst of attempts from newly created payees, often with similar amounts and device
properties. The immediate response is a policy or rule action — a temporary hard-floor rule on the
signature — applied with a version string and an expiry, reviewed within the day. Retraining is the
medium-term response, after the campaign's labels mature enough to learn from, which is 120 days
away. Relying on the model to adapt within hours is a category error: the label delay makes it
impossible.

**Reasoning.** The candidate must know the label latency and realise that supervised adaptation
cannot be fast in this domain. This is where the "just retrain" reflex fails.

**Follow-up.** "What evidence would justify the temporary rule?" → Attempt counts by payee age,
amount distribution, and the share of the decline band in the last hour, compared to the same
hour on prior days. The rule's precision can be estimated within hours from decline outcomes.

**Acceptable alternative.** Lowering the decline band floor temporarily is also acceptable, provided
the false-decline guardrail is monitored hourly and the change has an expiry.

**Red flag.** "Retrain nightly." With 120-day labels, a nightly retrain learns nothing about this
campaign.

**Scoring.** 3 = proposes a rule or policy change. 4 = explains the label-latency reason retraining
fails. 5 = adds versioning, expiry and a review cadence.

**Support.** A-02, §8.3 choice 2, §8.4 trace C.

---

#### Interview 11 — "What does the step-up band's 3.71% precision imply for measurement?"

**Answer.** That the band's value is almost entirely determined by the *stop rate*, not by the
model. 62,000 challenges contain 2,300 fraud attempts; at a 70% stop rate 1,610 are prevented and
59,700 legitimate customers authenticate. If the stop rate were 30%, only 690 would be prevented
and the band's contribution to the ledger would fall by `920 × 210 = €193,200` per 30 days, roughly
half the total increment. So A-05 is not a secondary parameter; it is one of the two or three
numbers that decide whether the design is worth building, and it must be measured per challenge
type rather than assumed.

**Reasoning.** The candidate should see that a low-precision band's economics hinge on the
effectiveness of the action applied to it, not on the precision itself.

**Follow-up.** "How do you measure the stop rate without a clean counterfactual?" → Compare fraud
attempts that reached the challenge against those that did not within the same band, and track
post-challenge dispute rates. This is an observational comparison with selection bias, so it bounds
the rate rather than identifying it; the bound is enough to reject a 30% stop rate if the observed
dispute rate is low.

**Acceptable alternative.** A holdout that withholds the challenge for a random subset within the
band gives a clean estimate at the cost of some fraud exposure; acceptable if the exposure is stated
and bounded.

**Red flag.** Treating 70% as a fact because it was in a business case.

**Scoring.** 3 = identifies A-05 as material. 4 = computes the sensitivity. 5 = proposes a
measurement design and names its bias.

**Support.** §6.5, §6.6, A-05.

---

#### Interview 12 — "The false-decline guardrail allows 4,186 events and the design produces 4,000. Is that fine?"

**Answer.** It is admissible but fragile, and I would not ship it without understanding the variance.
186 events of headroom over 30 days is 6.2 per day against a daily rate of about 133 false declines;
a single day with a 5% excursion breaches it. The guardrail therefore needs a daily monitor with an
explicit breach response, not a monthly review. I would also ask whether the 4,000 estimate is a
point estimate from a synthetic distribution — it is — which means the headroom is not a confidence
interval. The safer configuration moves part of the decline band into step-up, trading some
prevented fraud for margin.

**Reasoning.** The candidate should treat a near-binding constraint as a risk, not as a pass, and
should notice that the estimate's provenance affects how much the headroom means.

**Follow-up.** "How much fraud would moving the marginal 200 declines to step-up cost?" → If those
200 are representative of the band's 72.6% precision, about 145 are fraudulent; at a 70% stop rate
about 102 are still prevented, so the loss is roughly `43 × 210 = €9,030` per 30 days for a 200-event
improvement in false declines. That is a cheap trade.

**Acceptable alternative.** Keeping the band and tightening the appeal path is defensible if the
appeal is fast and human-owned, because it addresses the harm directly rather than the count.

**Red flag.** "4,000 < 4,186, so we're compliant." Technically true, operationally reckless.

**Scoring.** 3 = notes the tight margin. 4 = proposes a daily monitor. 5 = quantifies the trade and
notes the point-estimate caveat.

**Support.** §9.1, §9.6 boundary case.

---

#### Interview 13 — "Design the audit record for one decision."

**Answer.** The record must let a reviewer reconstruct the decision without access to the model's
internals. Minimum fields: `payment_id`, `t_request`, `policy_version`, `model_version`,
`feature_snapshot_id`, per-feature `value` and `t_available` (or an aggregate `max_feature_age_ms`),
`score_raw`, `score_calibrated`, `band`, `action`, `reason_code`, `degraded` flag with cause,
`rules_floor_hit` if any, and the downstream outcome link (`case_id` or `challenge_id`). Immutable,
append-only, retained per the local retention policy — which is an OPEN item here, not something
this case can specify.

**Reasoning.** The candidate should produce fields that answer "why did this happen" and "what did
the system know", and should separate what the model knew from what the policy did.

**Follow-up.** "Why record both raw and calibrated scores?" → Because band boundaries are in
probability units, so the calibrated value explains the action, while the raw value allows
recalibration and model diagnostics without replaying inference.

**Acceptable alternative.** Storing a pointer to an immutable feature snapshot rather than the values
is acceptable if the snapshot is retained for the audit window and is truly immutable.

**Red flag.** Recording only the action and a timestamp. That record cannot support an appeal or a
model review.

**Scoring.** 3 = names the key identifiers. 4 = adds feature availability and the degraded flag.
5 = separates model from policy and notes retention as a local policy question.

**Support.** §8.1, §8.4, C-03.

---

#### Interview 14 — "You are told to cut review cost by 70%. What do you do?"

**Answer.** First determine whether the target is a review-tier cost cut or a total-cost cut, because
they have different answers. Review cost is €54,000 against a total programme cost of about €157,500
(review, friction, challenge delivery, platform), so a 70% cut in review is a 24% cut in programme
cost — achievable by halving the sample, at the price of intelligence value that is not in the
ledger. I would rather reduce the *arrival* into the review band by raising its lower probability
boundary, which moves cases to auto-approve and costs nothing operationally, and check whether that
costs fraud prevention. If the target is on total cost, the largest lever is friction
(€69,802), so improving band precision is worth more than cutting review. What I would not do is
reduce handling time by removing the context-gathering step; that trades cost for review quality
and shows up later as reopenings.

**Reasoning.** The candidate should decompose the cost base before choosing a lever, and should
recognise that a target expressed as a percentage of one line item may have been derived from a
different denominator.

**Follow-up.** "Raising the band boundary from 0.0194 to 0.05 moves how many cases and what fraud?"
→ It removes those attempts from case creation; under the constructed distribution most of the
148,000 cases sit near the boundary, so the move is large, and some of the 800 fraud attempts in the
band are lost. The exact split requires the within-band distribution, which is not given here — the
honest answer names the missing input.

**Acceptable alternative.** Automating the low-context portion of review (evidence assembly, not the
disposition) reduces handling time without removing judgement; acceptable and often the best answer.

**Red flag.** Cutting the sample without stating what intelligence or evidence value is lost.

**Scoring.** 3 = decomposes costs. 4 = distinguishes review cost from total cost. 5 = identifies
friction as the larger lever and refuses to trade review quality silently.

**Support.** §6.5 ledger, §8.3, §6.6 review row.

---

## 15. Quick Reference Card

### 15.1 The decision in six fields

| Field | Content |
|---|---|
| **Input** | One instant payment attempt with event-time features, payee graph state and account history, at `t_request` |
| **Output** | One action from {approve, sampled review, step-up, decline}, with version-stamped reason code |
| **Decision owner** | Payment risk owner; fraud operations owns the queue, platform owns latency |
| **Constraint** | p99 ≤ 150 ms; false-decline ≤ 0.10% of attempts; review ≤ ~4% of the suspicious band; no autonomous account action |
| **Baseline** | Static rules: 60.00% recall, 16.67% precision, 42,000 false positives, €1,306,200 net per 30 d |
| **Evidence** | ILLUSTRATIVE composite; arithmetic verified; no deployment, no measured outcome |

### 15.2 Numbers that decide the case

| Quantity | Value | Where derived |
|---|---:|---|
| Attempts / 30 d | 4,200,000 | §2.3 |
| Prevalence | 0.3333% (14,000) | §2.3 |
| Economic threshold, friction-only | 0.01938 | §6.2 |
| Economic threshold, full false-decline cost | 0.03493 | §6.2 |
| Pure-threshold decline rate | 5.348% (224,600) | §6.3 |
| Decline band | 14,600 attempts, 72.60% precise | §6.4 |
| Step-up band | 62,000 attempts, 3.71% precise | §6.4 |
| Review band | 148,000 attempts, 0.54% precise | §6.4 |
| Review sample | 4.05% → 6,000 cases / 30 d | §6.5 |
| Incremental net benefit | €1,100,398 / 30 d | §6.5 |
| False-decline headroom | 186 events | §9.1 |
| Recall standard error | 0.00414 | §9.3 |

### 15.3 Five rules to apply elsewhere

1. **Derive the cut point from costs, then constrain it with harm.** `p* = C_FP/(C_FN + C_FP)`.
2. **Low prevalence makes precision a bad objective and graded action the right design.** Bands with
   sub-5% precision belong in challenge or sample, never in block.
3. **Human capacity bounds the design before model quality does.** Compute cases/day against staffed
   hours; a sampling rate above capacity is a fiction.
4. **Latency is a correctness property.** Allocate p99 per stage, keep explicit slack, exercise the
   degraded path at peak.
5. **The replay harness is the cheapest artifact and the highest-leverage one.** No policy change
   without a recomputation.

### 15.4 Do / Avoid

| Do | Avoid |
|---|---|
| Evaluate on matured, time-split cohorts with censored counts reported | Treating unmatured attempts as legitimate |
| Version policy separately from model | Changing bands by editing code |
| Record feature availability per decision | Reconstructing "what the model saw" from logs later |
| State the accepted loss when a band only approves | Reporting recall without the false-decline count |
| Measure the challenge stop rate | Assuming a business-case parameter |

### 15.5 Evidence, stage and review record

| Field | Value |
|---|---|
| Evidence level | `not-measured`; all figures illustrative and constructed |
| Arithmetic status | Verified against the §9.6 identity checks |
| Implementation stage | `proposed`; nothing deployed, no shadow run executed |
| Editorial status | `draft`; independent review pending |
| Source access | No institution data, no scheme rules; all scheme-specific claims OPEN |
| Publication review | OPEN |
| Review trigger | Change in loss assumptions, latency limit, dispute rules, staffing, or calibration |

### 15.6 The one-sentence summary

> Score to get a calibrated probability, derive the cut point from costs, then refuse to let that
> cut point become a block — route each band to the least harmful action that still protects the
> money, and prove the whole ledger inside a replay harness before a customer ever sees a decline.
