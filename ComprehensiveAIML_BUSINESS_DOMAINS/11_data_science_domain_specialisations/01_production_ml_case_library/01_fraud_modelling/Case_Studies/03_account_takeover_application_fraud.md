---
case_id: CASE-01-01-003
case_title: Account takeover challenge policy with incomplete identity signals
case_version: "2.0"
created: 2026-11-09
last_reviewed: null
status: proposed
content_status: draft
case_type: predictive-ml
domain: Digital account security
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, cost-latency-tradeoff]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-10
owner_role: Identity-risk product owner
source_index: INDEX.md
review_trigger: Authentication flow, identity-signal consent, attack pattern, response deadline, or recovery-policy change
---

# Case study: Account takeover challenge policy with incomplete identity signals

Everything numeric in this case is **ILLUSTRATIVE**. No service, account population, device
population, authentication log, model, or prevention outcome is observed. The arithmetic is
constructed so that a reader can reproduce each figure from the stated inputs; that is a claim about
internal consistency, not about any real account base. Quantities that a real programme would have
to measure are marked `not-measured` and carried into the sensitivity analysis rather than quietly
assumed.

The one externally sourced element is vocabulary and mechanism. NIST SP 800-63B defines
authentication assurance levels and treats recovery as part of the authenticator lifecycle, which is
why a protective hold is a security control rather than a customer-service convenience. Nothing in
that document supplies the cost or effectiveness inputs used here, and the case does not imply
compliance with it.

A note on scope relative to the file's position in the catalogue. This case is about an
**established account** whose credentials may have been compromised: the identity exists, has a
history, and can be verified by something the holder knows or carries. Application fraud — a
fraudulent *new* account opened with a synthetic or stolen identity — is the neighbouring decision
and is treated here only as a boundary, because it lacks the one resource this case depends on: a
prior behavioural baseline for the person being verified. Section 4 states that boundary precisely,
and section 13 explains why the transfer does not hold.

## 1. The One-Line Framing

For a login or sensitive-account action on an established account, choose allow, step-up challenge,
or protective hold using only signals that exist at the moment of the event, so that mature
account-takeover loss falls per 1,000 eligible events without taking legitimate customers offline
beyond an agreed hold budget, compared with the existing authentication rules.

The decision is not "is this person the account holder". It is "which of three actions, given a
probability I cannot verify and a recovery path I do not control, minimizes expected harm". Those
phrasing differ in what they permit: the first invites a verdict about identity, the second keeps the
system honest about the fact that it is choosing an action under uncertainty.

The naive framing — classify the login as fraudulent and block it — fails on four counts, and each
one is load-bearing.

A failed challenge is not evidence of an attacker. Legitimate users fail challenges constantly:
new phones, cleared cookies, travel, password managers, screen readers, and simple typos. Section 6
shows a failed challenge moving the posterior probability of compromise from 0.027% to 0.55% — a
large relative move and a tiny absolute one.

A successful challenge is not evidence of the account holder either. An attacker with a phished
password frequently also holds the second factor, which is the entire reason step-up authentication
against a phished session is weaker than against a credential-stuffing attempt.

The label is late and selective. A takeover that is never reported is never labelled, and reporting
rates differ systematically by customer segment, which means the positive class is not a random
sample of takeovers.

A protective hold is a *different kind of harm*, not a larger amount of the same harm. It removes
access to money, messages, or medical records, and its cost is dominated by how long it lasts and
whether the recovery path works, not by how many are issued.

**Takeaway:** the object of the decision is an action with a recovery dependency, and the score is
one input to it.

## 2. Context & Background

An established account is accessed through a login flow. The flow may include a password, a
one-time code over SMS, a push notification to a registered device, a passkey, or a federated
identity provider. After login, some actions are additionally protected: changing the password,
changing the registered email or phone number, adding a payee, viewing or exporting personal data,
or initiating a payment. Both the login and each protected action are *events*, and each is a
decision point.

An attacker reaches an account through one of a small number of routes, and the route determines
which signals are informative.

**Credential stuffing** uses username and password pairs leaked elsewhere and replayed at scale. It
is high-volume, geographically dispersed, and leaves device and network fingerprints that differ
from the holder's history. It is the easiest route to detect because it is not targeted.

**Phishing and adversary-in-the-middle proxying** obtains credentials and often the one-time code in
real time. The attacker may then hold a live session token and never trigger a fresh login at all,
which means login-time detection cannot see the compromise at the moment it matters.

**Malware and infostealers** lift session cookies from a browser, again bypassing the login event.
The first observable signal may be a protected action from an unfamiliar device with a valid token.

**Help-desk social engineering and recovery abuse** manipulates the recovery process rather than the
credential. It is the route that makes the recovery path itself an attack surface, and it is why
recovery is modelled as part of the policy rather than after it.

**Insider or shared-credential access** in household and small-business contexts produces events
that look exactly like takeover and are legitimate, which is a persistent source of false positives
that no model resolves.

Four accountabilities matter and they are routinely misassigned.

The **identity-risk product owner** owns the action policy and the hold budget. The **authentication
platform owner** owns availability, retry semantics, token lifetime, and the challenge providers'
contracts. The **customer support function** owns the recovery path and its capacity; a hold that
support cannot clear is an outage with a security justification. **Privacy and security review**
govern which signals may be collected, for what purpose, and for how long.

The incumbent is a versioned rules policy: challenge when the device is unrecognized, when the
network geolocates far from recent history, when the velocity of failed logins exceeds a threshold,
or when the account is flagged. Such a policy is understandable, auditable, and fast. It is also
coarse in a specific way that matters here: it applies the same action to events with wildly
different stakes. A login to a dormant social account and a password reset on a bank account are
not the same decision, but a rule that keys on "unrecognized device" treats them as one.

Scope is one consumer product with an established-account login and a protected credential-change
action, over thirty days. Out of scope: new-account application and onboarding, merchant or
business accounts with delegated access, API clients, and any physical-access control. Those have
different action sets, different loss bearers, and — in the application case — no behavioural
baseline at all.

A structural fact that shapes everything: the account holder and the attacker may both be able to
pass a challenge, and the account holder may be unable to. The signal is therefore not a test of
identity but a friction device whose value depends on the attacker's cost of passing it, which is a
property of the attack route, not of the customer.

**Takeaway:** the attack route, not the customer, determines how informative a challenge is, so a
single effectiveness number is wrong before it is estimated.

## 3. The Business Question

| Decision-contract field | Illustrative contract |
|---|---|
| Question | Which eligible authentication and credential-change events receive allow, step-up challenge, or protective hold? |
| Owner | Identity-risk product owner; support approves hold volume; platform approves latency and capability. |
| Trigger | Authentication attempt or protected-action request. |
| Deadline | Login must complete inside the interactive budget; holds need not be synchronous with a user gesture but must be reversible inside the support SLA. |
| Actions | Allow; challenge where the channel supports it; protective hold routed to recovery; fall back to incumbent rules. |
| Value | Mature unauthorized-access loss plus monetized friction, support load, and hold-duration harm, per 1,000 eligible events. |
| Denominator | All eligible events assigned to the policy, including challenged and held events. |
| Hard constraints | Hold budget; recovery SLA; no unreviewed automated adverse path; approved signals only; accessibility of every challenge. |
| Guardrails | Legitimate challenge pass rate, hold duration distribution, support queue age, complaint rate, subgroup hold parity. |
| Comparator | Incumbent authentication rules on the same events in the same window. |
| Out of scope | Determining who the attacker is, criminal referral decisions, new-account application decisions. |

Three quantities are routinely conflated and each answers a different question.

**Challenge pass rate** is the share of challenged events that complete the challenge. It rises when
attacks succeed and also rises when challenges are easy, so it is not a security metric.

**Legitimate challenge pass rate**, computed only on events later confirmed legitimate, is a
customer-experience metric, and it is the one that predicts abandonment and support contact.

**Hold clearance time** is the distribution of durations between a hold and the restoration of
access. Its mean is nearly meaningless; the 95th percentile is what generates complaints and
regulatory attention.

The estimand is a paired difference on a common cohort: change in mature cost per 1,000 eligible
events relative to the incumbent, where cost includes loss, friction, support, and a pre-agreed
monetization of hold duration. Levels are not comparable across products; differences on a common
cohort are.

One contractual detail deserves early emphasis because it is usually omitted. A hold is *not* a
decline. It must have a stated maximum duration, a named recovery route, and an escalation path for
a customer who cannot complete that route. Without those three, the policy is an outage generator,
and the security team will discover this only after the first bank-holiday weekend.

**Takeaway:** separate pass rate, legitimate pass rate, and hold clearance; only the last two are
decision-relevant.

## 4. Assumptions & Constraints

| ID | Assumption | Test | Consequence if false |
|---|---|---|---|
| A-01 | Event, receipt, and decision timestamps are retained at millisecond resolution. | Replay a day of events against stored logs. | Policy cannot be reconstructed; evaluation reverts to aggregates. |
| A-02 | A mature compromise outcome exists and matures within 60 days. | Plot outcome landing by cohort week. | Recent cohorts are biased toward loud cases; recall is overstated. |
| A-03 | The challenge is available and accessible on the measured channels. | Capability matrix plus assistive-technology test. | Challenge arm cannot be priced; policy collapses to allow/hold. |
| A-04 | Challenge effectiveness is estimable from a randomized slice. | Withhold challenge on a small eligible band. | Prevention unidentifiable; `e` becomes a sensitivity axis. |
| A-05 | Loss per successful compromise is approximately constant within each event class. | Stratify realized loss by account type and balance band. | A single `L` misprices high-value accounts; ledger must be per-band. |
| A-06 | The recovery route is staffed to the hold volume the policy creates. | Queue-age measurement at the target volume. | Holds become outages; policy must be capped by support capacity. |
| C-01 | A retried event returns the same action. | Inject duplicate events with the same identifier. | Customer sees allow then hold; disputes are unwinnable. |
| C-02 | A held account retains a recovery path that the genuine holder can complete. | Recovery-path walkthrough with assistive technology. | The control locks out the very customers it protects. |
| C-03 | No signal is used without an approved purpose and retention rule. | Signal register audit. | Compliance exposure independent of model quality. |

Two of these deserve expansion because they are the ones that reverse decisions rather than shift
numbers.

**C-02 is a hard gate, not a guardrail.** If the genuine holder of a held account cannot restore
access — because the registered phone is the compromised one, because the recovery email is defunct,
because they have no document that matches the record — then the hold converts a fraud loss into a
total loss of access for a legitimate customer. In the constructed policy this is not a rare edge:
the recovery route is the same route an attacker tries to abuse, so the two are in direct tension.
Any design that hardens recovery against attackers also hardens it against the holder.

**A-06 is a capacity constraint that behaves like a budget.** Support capacity is not elastic within
a month. Section 6 shows a candidate policy that adds 60 holds over thirty days and moves support
load from 1.03 to 1.29 hours per day on average, but concentrates 40% of that load into a few days,
pushing a single agent's utilization to 79% on the peak day. An average that looks trivial can hide a
peak that fails.

The application-fraud boundary belongs here explicitly. A new-account application has no prior
authentication history, so every device and network signal is a first observation, and the action
set includes documentary verification and manual review with turnaround measured in days rather than
seconds. The arithmetic in this case assumes a baseline exists; applying these thresholds to
applications would be a category error, because `q_hold` and `q_chal` are defined against a
*behavioural* deviation whose scale is meaningless without history.

**Takeaway:** the recovery path and the support peak are constraints of the same kind as the loss
target, and both can invalidate a policy that scores well.

## 5. Data, Tools & Method

### 5a. Data Specification

Grain: one row per authentication event or protected-action event. Key: an immutable event
identifier plus the account token plus the session token, because a single session may generate
several protected actions and each is a separate decision with a different loss exposure.

Cut-off: the decision timestamp recorded by the identity service, not the time the event was
written to the warehouse. Population: eligible events on established accounts for the thirty-day
cohort, split into two classes that are never pooled for thresholding:

| Class | Events in cohort | Description | Loss per compromise |
|---|---|---|---|
| A — login | 1,120,000 | Ordinary authentication into the account | 711.50 |
| P — protected action | 80,000 | Credential change, contact-detail change, payee addition | 1,240.00 |

Units: monetary amounts in one currency. Probabilities as fractions, displayed as percentages to
four decimals where they are small, because at these base rates two decimals of a percentage hides
the entire decision.

Candidate signals at decision time, each with its availability caveat:

| Signal group | Examples | Caveat |
|---|---|---|
| Event context | event type, channel, client version, timestamps | Client version can be spoofed by a modified app. |
| Credential state | password age band, MFA enrolment state, recovery-detail age | Password age is a weak proxy for compromise. |
| Device | device token presence, first-seen age, token rotation history | Token absence is not a new device; tokens rotate legitimately. |
| Network | coarse region, ASN class, hosting-flag, latency band | Mobile and corporate NAT collapse many users onto one address. |
| Behaviour | recent event count, time-since-last-success, prior failed challenges | Counters must be computed pre-decision, not at settlement. |
| Recovery state | whether recovery details are fresh, whether recovery was recently used | Recent recovery use is both a legitimate and an attack signature. |

Signals that are **not** available at decision time and must never enter the model: support
disposition, confirmed-compromise case identifier, password reset outcome, refund or chargeback
record, customer complaint text, and any device-to-account link discovered after the event. Each is
a downstream consequence of the decision being made.

The label is `confirmed_compromise` under a controlled definition: the account owner reported
unauthorized access and the investigation confirmed it, or the security team independently
confirmed account abuse through forensic evidence. It excludes "the customer says they did not
recognize a login" without further evidence, because misremembering one's own sessions is common and
would poison the label with recall failures rather than attacks.

Three label defects must be handled explicitly rather than assumed away:

1. **Unreported takeovers.** An attacker who changes nothing and steals nothing generates no report.
   The positive class under-counts quiet compromise, which biases the model toward detecting noisy
   attackers.
2. **Late reporting.** Reports arrive with a long tail. A sixty-day maturity window truncates it.
   Section 9 quantifies how much of the positive class a longer window would add under a constructed
   delay curve.
3. **Reporting propensity.** Customers with app access, younger customers, and higher-value accounts
   report more. The model may therefore learn reporting propensity as much as compromise, and the
   discriminating check for this is in section 12, F-03.

Exclusions: synthetic monitoring traffic, internal test accounts, events from accounts already under
an active hold, and events that failed schema validation. Excluding already-held accounts is
legitimate because the hold is already the action; excluding failed challenges is not, because the
failure is part of the evidence chain.

### 5b. Tools & Stack

Proposed components by function: an identity event bus emitting immutable event records; a signal
service serving device and behaviour features with freshness stamps; a calibrated scorer over
tabular signals; a policy engine holding two class-specific band tables; a durable decision store
written before the response; an adapter layer for challenge providers; a recovery workflow with
queue and SLA tracking; and a replay harness for temporal evaluation.

| Option | Strength | Weakness | Status here |
|---|---|---|---|
| Incumbent rules | Explainable, instant rollback, no training dependence | Same action for very different stakes; brittle thresholds | Comparator |
| Rules plus class-specific step-up | Cheap, reversible, easy to audit | Needs challenge coverage per channel | Baseline comparator variant |
| Calibrated score plus priced bands | Prices the action and the class separately | Needs mature labels and a signed cost schedule | Proposed |
| Score plus automatic hold above a threshold | Limits damage fast | Holds are expensive and recovery-dependent | Considered, rejected as a default |
| Sequence model over session behaviour | Exploits ordering within a session | Feature serving cost; harder audit | Deferred |
| Graph model over shared devices and payees | Strong on coordinated rings | Cold start; label scarcity; latency | Out of scope here |

The relevant architecture decision is not the model family. It is that the **class-specific band
tables** exist at all. A single table across login and credential change forces one of two errors:
either credential changes are under-protected, because the table is calibrated for cheap logins, or
logins are over-protected, because the table is calibrated for expensive credential changes. Both
errors are common and both are invisible in a pooled metric.

### 5c. Method Summary

Pipeline: receive event; resolve class; validate eligibility and challenge capability; fetch signals
with freshness checks; score; calibrate; select the class table; choose action with reason code;
persist before responding; join the mature outcome asynchronously.

| Stage | Input | Output | Failure behaviour |
|---|---|---|---|
| Class resolution | Event record | A or P | Unresolved class is a hard error, never a default |
| Capability check | Channel, client version | Challenge usable | Not usable routes to the class table's no-challenge variant |
| Signal fetch | Account and device keys | Features plus freshness | Stale beyond tolerance triggers incumbent rules |
| Scoring | Feature vector | Raw score | Scorer error triggers incumbent rules with a flag |
| Calibration | Raw score | Compromise probability | Table version pinned per day |
| Policy | Probability, class, capability | Action, reason code | Unknown band is a hard error |
| Persistence | Decision record | Durable row | Failure blocks the response |
| Recovery handoff | Held decision | Support ticket with SLA | Failure must alert, not queue silently |

The recovery handoff row is the part that most designs omit and every support team discovers. A hold
that is written to the decision store but never reaches the support queue is an account locked
indefinitely with no ticket, and it will be found by the customer rather than by monitoring.

### 5d. Provenance, quality, exclusions and leakage controls

Each signal carries an owner, source system, version, timestamp semantic, permitted purpose,
retention rule, and a note on what it is *not*. The last item is unusual and valuable: recording
"device token absence does not mean new device" next to the field prevents the misreading that the
field invites.

Leakage controls, in application order:

1. Features are rebuilt from snapshots taken strictly before the decision timestamp.
2. Device-to-account linkage discovered *after* the event is excluded; the linkage is often how the
   compromise was diagnosed, so including it produces a model that cannot exist in production.
3. Behavioural counters are computed from a pre-decision window, with the window boundary exclusive.
4. Calibration is fitted on development folds by time, never at random.
5. Any field whose univariate association with the label exceeds a plausibility bound is quarantined
   and re-audited, because implausible strength is the signature of leakage in identity data.

Quality checks run before the model is trusted: duplicate event identifiers; events whose decision
timestamp precedes their event timestamp; accounts with impossible event rates; device tokens
attached to implausible account counts; and recovery-detail ages that are negative.

### 5e. Reproducibility & Method-Selection Ledger

| ID | Artifact | Purpose | Evidence boundary |
|---|---|---|---|
| RUN-01 | Constructed thirty-day event cohort | Arithmetic demonstration | No dataset exists; all values constructed. |
| POL-01 | Two class band tables | Action selection per class | Illustrative; requires a signed cost schedule. |
| SRC-01 | NIST SP 800-63B | Authentication and recovery vocabulary | Supplies no cost or effectiveness input. |
| SRC-02 | Proposed signal register | Permitted-use and retention control | Not reviewed by privacy; not in force. |
| CHK-01 | Independent arithmetic re-check | Verifies every printed product | Executed by hand and by script; see section 6. |

Reproducibility convention, stated once and used throughout: monetary values are carried at full
precision and displayed to two decimals; probabilities under 1% are displayed to four decimals as
percentages; counts are integers, and any fractional count is rounded once, before multiplication, so
the printed components add exactly to the total.

**Takeaway:** the evidence boundary is that nothing here was observed; the arithmetic is reproducible
and the inputs are invented.

## 6. Analysis Walkthrough

The walkthrough derives the two threshold pairs, then prices a policy, then attacks its weakest
assumption.

### Step 1 — Fix the cohort, the two classes, and the loss scalars

```text
Class A (login)            n_A = 1,120,000 events   positives 300   base rate 0.0268%
Class P (protected action) n_P =    80,000 events   positives 180   base rate 0.2250%
Total                      N   = 1,200,000          positives 480   base rate 0.0400%
```

The two classes are not pooled because their loss per compromise differs by a factor of 1.74 and
their base rates differ by a factor of 8.4. Pooling would produce one threshold that is wrong in
both directions, and the wrongness would be invisible because the pooled metric would look
respectable.

```text
L_A = 711.50   expected loss per successful login compromise
L_P = 1,240.00 expected loss per successful protected-action compromise
```

`L` is not the account balance. It is the expected amount the institution and customer together
bear after a compromise: unauthorized payments not recovered, cost of restoring the account, cost of
the investigation, and a pre-agreed allowance for customer detriment that is not reimbursed. Using
the balance would overstate it catastrophically, because most compromises do not drain the account;
using zero because "the customer is reimbursed" would understate it, because reimbursement does not
cover the customer's time, the institution's investigation cost, or the fraud that is never claimed.

Friction scalars:

```text
c_chal   = 0.5090   per challenge (delivery 0.18 plus 0.035 * 9.40 for support contact)
c_hold   = 64.30    per hold (support handling, identity verification, customer detriment)
e_A      = 0.72     challenge effectiveness on login events
e_P      = 0.28     challenge effectiveness on protected-action events
h        = 0.97     hold effectiveness (a hold blocks the action with near certainty)
```

The effectiveness split is the central claim of this case and it is not arbitrary. A login challenge
asks the person to prove possession of a second factor. Against credential stuffing, that works well,
because the attacker has only the password. Against a phishing proxy or a stolen session, the
attacker already has the factor, so the challenge does not separate attacker from holder. Class A
mixes all attack routes and lands at 0.72. Class P is worse because the protected actions that
matter — contact-detail change, password reset, payee addition — are precisely the actions an
attacker attempts *after* establishing a session, and a challenge sent to the registered device
reaches the very token the attacker may already control. Hence 0.28.

A hold, by contrast, is close to certain: it stops the action regardless of who is holding the
credential, because it requires human verification to lift. That is why `h = 0.97` and not 0.72.

```text
FORMULA [T-01]: challenge break-even probability against allowing
q_chal = c_chal / (e * L)
WHERE: c_chal [currency/event] is challenge cost; e [fraction] is the share of attacks the challenge
       prevents; L [currency] is loss per successful compromise.
CONSTRAINT: e > 0 strictly, e measured per class and per attack-route mix. If the route mix shifts
       toward session theft, e falls and q_chal rises; a stale e therefore under-protects.
NUMERICAL EXAMPLE: class A: 0.5090 / (0.72 * 711.50) = 0.5090 / 512.28 = 0.0009936 = 0.0994%.
                  class P: 0.5090 / (0.28 * 1,240.00) = 0.5090 / 347.20 = 0.0014660 = 0.1466%.
INTERPRETATION: challenging pays off at a fraud probability of about one tenth of one percent on
       logins and one seventh of one percent on protected actions. Compared with the class base rates
       of 0.0268% and 0.2250%, this yields an asymmetric conclusion: class P's base rate exceeds its
       threshold, class A's does not.
```

That asymmetry is the first real finding and it deserves stating plainly. Class P's base rate of
0.225% exceeds its challenge threshold of 0.1466%, so challenging *every* protected-action event
would be marginally worth its cost. Class A's base rate of 0.0268% is a quarter of its threshold of
0.0994%, so challenging every login would lose money. The policy must therefore concentrate class A challenges on a small high-probability
region and spread class P challenges almost uniformly. That structural asymmetry is invisible in
any pooled formulation and it is the reason the band tables are separate.

```text
FORMULA [T-02]: hold break-even probability against allowing
q_hold = c_hold / (h * L)
WHERE: c_hold [currency/hold] is total hold cost including support handling and customer detriment;
       h [fraction] is hold effectiveness; L [currency] as above.
CONSTRAINT: valid only if c_hold is a complete cost. Omitting the customer-detriment allowance makes
       holds look cheap and drives a policy that locks accounts for trivial risk.
NUMERICAL EXAMPLE: class A: 64.30 / (0.97 * 711.50) = 64.30 / 690.16 = 0.09317 = 9.32%.
                  class P: 64.30 / (0.97 * 1,240.00) = 64.30 / 1,202.80 = 0.05346 = 5.35%.
INTERPRETATION: a hold needs roughly one hundred times the probability mass a challenge needs,
       because it costs roughly one hundred times as much.
```

```text
FORMULA [T-03]: hold-versus-challenge crossover
q_hc = (c_hold - c_chal) / (L * (h - e))
WHERE: numerator [currency] is the extra cost of a hold over a challenge; denominator [currency] is
       the extra loss a challenge allows relative to a hold, for one fraudulent event.
CONSTRAINT: requires h > e strictly. Where a challenge is as effective as a hold the denominator is
       zero and the hold is never justified on cost grounds alone.
NUMERICAL EXAMPLE: class A: (64.30 - 0.5090) / (711.50 * 0.25) = 63.791 / 177.875 = 0.35863 = 35.86%.
                  class P: (64.30 - 0.5090) / (1,240.00 * 0.69) = 63.791 / 855.60 = 0.07456 = 7.46%.
INTERPRETATION: on logins a hold only beats a challenge above 35.86% probability, which almost
       nothing reaches; on protected actions the crossover is 7.46%, low enough that the policy will
       actually use holds.
```

Verify the three regions by direct cost evaluation at selected probabilities rather than trusting the
crossovers. For class A at `p = 0.05`:

```text
Allow      = 0.05 * 711.50                       = 35.575
Challenge  = 0.5090 + 0.05 * 0.28 * 711.50       = 0.5090 + 9.961 = 10.470
Hold       = 64.30 + 0.05 * 0.03 * 711.50        = 64.30 + 1.067 = 65.367
Minimum is challenge, consistent with 0.0994% < 5% < 35.86%.
```

Note the hold term: the hold cost is paid whether or not the event was fraudulent, and the residual
loss is the 3% of fraudulent events a hold fails to block. Writing the hold as `(1 - p) * c_hold`
instead — 0.95 * 64.30 = 61.085 here — is wrong in a way that matters, because it makes a hold look
cheaper at high probability, which is backwards: a hold's cost does not shrink as the event becomes
more likely to be fraud.

For class P at `p = 0.05`, which is below its hold crossover of 7.46%:

```text
Allow      = 0.05 * 1,240.00                     = 62.000
Challenge  = 0.5090 + 0.05 * 0.72 * 1,240.00     = 0.5090 + 44.640 = 45.149
Hold       = 64.30 + 0.05 * 0.03 * 1,240.00      = 64.30 + 1.860 = 66.160
Minimum is challenge, with hold worse than allow by 4.160 and worse than challenge by 21.011.
```

For class P at `p = 0.075`, above its crossover:

```text
Allow      = 0.075 * 1,240.00                    = 93.000
Challenge  = 0.5090 + 0.075 * 0.72 * 1,240.00    = 0.5090 + 66.960 = 67.469
Hold       = 64.30 + 0.075 * 0.03 * 1,240.00     = 64.30 + 2.790 = 67.090
Minimum is hold, by a margin of only 0.379 over challenge - consistent with p > 7.46%, and a useful
       illustration of how narrow the hold region is in class P.
```

**Boundary where the formulas break.** All three assume the probability `p` is correctly calibrated
for the *event*, and that the action's effectiveness is a constant. Both fail in an important case.
When an attacker holds a live session token, there is no authentication event to score at all, so
`p` is undefined for the action that matters; the only observable is the protected action itself,
which is why class P exists as a separate class rather than as a subclass of login. And when a
challenge is sent to a device the attacker controls, `e` is not 0.28, it is approximately zero, and
`q_chal` is undefined rather than large. The formulas are therefore valid only where the
challenge channel is independent of the credential path — a condition that must be checked at the
channel level, not assumed at the class level.

### Step 2 — Update the probability when a challenge fails

The most consequential piece of reasoning in this case is what a failed challenge means. The
temptation is to treat it as near-proof of compromise and escalate to a hold. The arithmetic says
otherwise.

```text
FORMULA [T-04]: posterior compromise probability after a failed challenge
prior odds      = pi / (1 - pi)
likelihood ratio = P(fail | compromise) / P(fail | legitimate)
posterior odds  = prior odds * likelihood ratio
posterior       = odds / (1 + odds)
WHERE: pi [fraction] is the class prior; P(fail | compromise) is taken as the challenge
       effectiveness e for that class, because a challenge that blocks an attacker manifests as a
       failure; P(fail | legitimate) = 0.035, the constructed legitimate failure rate.
CONSTRAINT: requires the failure rate to be measured on legitimate events. Measuring it on all
       challenged events inflates it enormously, because attacker failures dominate failures.
NUMERICAL EXAMPLE: class A: prior 0.000268, odds 0.0002680.
                  LR = 0.72 / 0.035 = 20.571.
                  posterior odds = 0.0002680 * 20.571 = 0.005513.
                  posterior = 0.005513 / 1.005513 = 0.005483 = 0.548%.
INTERPRETATION: a failed login challenge raises the compromise probability from 0.027% to 0.55%, a
       twenty-fold relative increase and an absolute probability of about one in 180.
```

Now compare that posterior with the hold threshold of 9.32%. It is seventeen times too small. A
failed challenge, on its own, does not justify a hold on a login event. It justifies a *second*
challenge or a different one, which is a different action with a different cost.

Continue the update to see where a hold does become justified:

```text
Second evidence: a password reset request from a different device within the same session.
Suppose that signal's likelihood ratio for compromise is 8.
posterior odds = 0.005513 * 8 = 0.044104
posterior      = 0.044104 / 1.044104 = 0.042241 = 4.22%
Still below 9.32%, so this too justifies a challenge, not a hold.

Third signal: a payee addition to a new beneficiary on the same session, likelihood ratio 14.
posterior odds = 0.044104 * 14 = 0.617456
posterior      = 0.617456 / 1.617456 = 0.381744 = 38.17%
Now above 9.32% and above the 35.86% hold-versus-challenge crossover, so a hold is justified.
```

The pass direction matters too, and it is the one that prevents over-blocking:

```text
P(pass | legitimate)   = 1 - 0.035 = 0.965
P(pass | compromise)   = 1 - e = 0.28   for class A
LR(pass)               = 0.28 / 0.965 = 0.290
prior odds             = 0.0002680
posterior odds         = 0.0002680 * 0.290 = 0.0000777
posterior              = 0.00779% -> 0.0078%
```

A passed challenge reduces the compromise probability from 0.0268% to 0.0078%, a threefold
reduction. That is a real but modest gain, and it is why "the customer passed the challenge" is weak
grounds for treating an event as safe. The asymmetry is the point: challenges discriminate much more
strongly in the failure direction than in the pass direction, because legitimate users fail often.

### Step 3 — Turn the thresholds into volumes and price both policies

Baseline, incumbent rules, over the thirty-day cohort:

```text
Class A: 1,120,000 events
  challenged 6,000 (0.536%) of which 72 confirmed compromises
  held          40 (0.0036%) of which  9 confirmed compromises
  allowed  1,113,960 of which 219 confirmed compromises
  Sum check: 6,000 + 40 + 1,113,960 = 1,120,000
  Class A positives check: 72 + 9 + 219 = 300

Class P: 80,000 events
  challenged   800 (1.00%) of which  5 confirmed compromises
  held          80 (0.10%) of which 11 confirmed compromises
  allowed   79,120 of which 164 confirmed compromises
  Sum check: 800 + 80 + 79,120 = 80,000
  Class P positives check: 5 + 11 + 164 = 180
```

Candidate policy, same cohort and same scorer, different band tables:

```text
Class A: 1,120,000 events
  challenged 9,000 (0.804%) of which 140 confirmed compromises
  held          30 (0.0027%) of which 16 confirmed compromises
  allowed  1,111,970 of which 144 confirmed compromises
  Sum check: 9,000 + 30 + 1,111,970 = 1,120,000
  Class A positives check: 140 + 16 + 144 = 300

Class P: 80,000 events
  challenged 2,400 (3.00%) of which 26 confirmed compromises
  held         150 (0.1875%) of which 58 confirmed compromises
  allowed   77,450 of which 96 confirmed compromises
  Sum check: 2,400 + 150 + 77,450 = 80,000
  Class P positives check: 26 + 58 + 96 = 180
```

The candidate's shape is the finding before any arithmetic. It moves protection *out* of class A
holds (40 to 30) and *into* class P holds (80 to 150), it triples class P challenges, and it only
modestly increases class A challenges. That reallocation costs almost nothing and it is the
mechanical reason the candidate wins. It is also the thing an incumbent rules engine cannot do,
because its rules do not know which class an event belongs to in the sense that matters.

```text
FORMULA [T-05]: total policy cost over the cohort
C = sum over classes c of [ (P_c - D_pos,c - h*H_pos,c - e_c*K_pos,c) * L_c
                          + (H_c - H_pos,c) * c_hold
                          + K_c * c_chal ]
WHERE: P_c [events] class positives; D_pos,c is zero here because no class-A/P event is hard-declined,
       H_c [events] held, H_pos,c held and confirmed compromised, K_c [events] challenged,
       K_pos,c challenged and confirmed compromised, and the remaining symbols as defined.
CONSTRAINT: h*H_pos and e_c*K_pos are each rounded to whole events before multiplication, and the
       rounding happens once so the printed components sum exactly.
NUMERICAL EXAMPLE: substituted below for both policies.
INTERPRETATION: prevented events are counted once each; a compromised event cannot be prevented
       twice, which is why the classes are summed separately rather than pooled.
```

Baseline substitution. Note the care needed at one point: a held positive is prevented by the hold,
and a challenged positive is prevented only in proportion to the challenge's effectiveness. A held
event is never also a challenged event, so the two prevented counts are disjoint and may be summed.

```text
Class A prevented by hold:      round(9 * 0.97)  = round(8.73)  =  9
Class A prevented by challenge: round(72 * 0.72) = round(51.84) = 52
Class A prevented, total:        9 + 52 = 61
Class A uncaught:                300 - 61 = 239
Class A loss:                    239 * 711.50 = 170,048.50

Class P prevented by hold:      round(11 * 0.97) = round(10.67) = 11
Class P prevented by challenge: round(5 * 0.28)  = round(1.40)  =  1
Class P prevented, total:        11 + 1 = 12
Class P uncaught:                180 - 12 = 168
Class P loss:                    168 * 1,240.00 = 208,320.00

Total loss:       170,048.50 + 208,320.00 = 378,368.50
Holds:            40 + 80 = 120;  120 * 64.30 = 7,716.00
Challenges:       6,000 + 800 = 6,800;  6,800 * 0.5090 = 3,461.20
Total C_base:     378,368.50 + 7,716.00 + 3,461.20 = 389,545.70
Per 1,000 events: 389,545.70 / 1,200 = 324.62
```

Candidate substitution:

```text
Class A prevented by hold:      round(16 * 0.97)  = round(15.52)  = 16
Class A prevented by challenge: round(140 * 0.72) = round(100.80) = 101
Class A prevented, total:        16 + 101 = 117
Class A uncaught:                300 - 117 = 183
Class A loss:                    183 * 711.50 = 130,204.50

Class P prevented by hold:      round(58 * 0.97) = round(56.26) = 56
Class P prevented by challenge: round(26 * 0.28) = round(7.28)  =  7
Class P prevented, total:        56 + 7 = 63
Class P uncaught:                180 - 63 = 117
Class P loss:                    117 * 1,240.00 = 145,080.00

Total loss:       130,204.50 + 145,080.00 = 275,284.50
Holds:            30 + 150 = 180;  180 * 64.30 = 11,574.00
Challenges:       9,000 + 2,400 = 11,400;  11,400 * 0.5090 = 5,802.60
Total C_cand:     275,284.50 + 11,574.00 + 5,802.60 = 292,661.10
Per 1,000 events: 292,661.10 / 1,200 = 243.88
```

One arithmetic trap is worth flagging because it is easy to fall into and hard to see. The uncaught
count must be `positives - prevented`, where prevented is the *rounded sum*. Subtracting the raw held
and challenged counts instead — `180 - 58 - 7 = 115` — understates the uncaught count by 2 and
understates the loss by 2,480. The difference arises because `round(58 * 0.97) = 56` is not 58: two
held positives are assumed to survive the hold. The convention is stated in section 5e for exactly
this reason, and a reader recomputing from the printed counts will otherwise reach a different figure.

The paired difference and its decomposition:

```text
Prevented, baseline:  61 + 12 = 73        = 15.2% of 480
Prevented, candidate: 117 + 63 = 180      = 37.5% of 480
Change in loss:       378,368.50 - 275,284.50 = 103,084.00
Change in hold cost:  11,574.00 - 7,716.00    = +3,858.00
Change in challenge:  5,802.60 - 3,461.20     = +2,341.40
Total:                103,084.00 - 3,858.00 - 2,341.40 = 96,884.60
Saving as a share:    96,884.60 / 389,545.70 = 24.87%
```

Check the sum: 103,084.00 - 3,858.00 = 99,226.00; 99,226.00 - 2,341.40 = 96,884.60. Exact.

The decomposition is the argument. The candidate spends 6,199.40 more on friction — 50% more on
holds and 68% more on challenges — and buys 103,084.00 of prevented loss. The marginal return on
friction is 16.6 to 1, which is the entire case for a priced policy, and it is only visible because
the two cost terms are reported separately from the loss term.

```text
Hold cost increase:      11,574.00 - 7,716.00 = 3,858.00   = 3,858.00 / 7,716.00 = 50.0%
Challenge cost increase:  5,802.60 - 3,461.20 = 2,341.40   = 2,341.40 / 3,461.20 = 67.6%
Total friction increase:  3,858.00 + 2,341.40 = 6,199.40
Return on friction:     103,084.00 / 6,199.40 = 16.6 to 1
```

The per-event framing makes the same point at a scale a steering committee can hold:

```text
107 additional prevented compromises across 1,200,000 events = 8.9 per 100,000 events
Incremental cost 6,199.40 / 107 = 57.94 currency units per prevented compromise
Average loss per compromise = (300 * 711.50 + 180 * 1,240.00) / 480 = (213,450 + 223,200) / 480
                            = 436,650 / 480 = 909.69
```

### Step 4 — Price the support load the policy creates

A hold is not free at the point of issue; it is free at the point of issue and expensive at the point
of clearance. The support load is computed from the holds the policy creates and the customer
contacts they generate.

```text
FORMULA [T-06]: support minutes per hold
M = H * f * t
WHERE: M [minutes] is total support time; H [holds] is hold volume; f [fraction] is the share of
       holds that generate a support contact; t [minutes/contact] is average handling time.
CONSTRAINT: t is an average and the distribution is long-tailed; a peak-day calculation must use the
       peak-day contact volume, not the monthly average divided by days.
NUMERICAL EXAMPLE: candidate holds 180 total; 106 on legitimate events at f = 0.78 and
       t = 22 minutes; 74 on compromised events at f = 0.55 and t = 12 minutes.
       Legitimate: 106 * 0.78 = 82.68 -> 83 contacts; 83 * 22 = 1,826 minutes.
       Compromised: 74 * 0.55 = 40.70 -> 41 contacts; 41 * 12 = 492 minutes.
       Total M = 2,318 minutes.
INTERPRETATION: 38.6 hours of support work across the month, or about 1.29 hours per day on average.
```

The average is the number that will be presented and the peak is the number that will break the
service. If 40% of the load lands in three consecutive days — which is what a credential-stuffing
spike looks like — the peak-day figure is:

```text
Peak-day minutes = 0.40 * 2,318 = 927.2 minutes across 3 days = 309.1 minutes per day = 5.15 hours
Against a 6.5-hour productive day for one agent: utilization = 5.15 / 6.5 = 79.2%
```

Seventy-nine percent utilization on a peak day is not a comfortable margin. Queueing theory makes the
point sharper than any rule of thumb: at 79% utilization with variable service times, expected wait
grows roughly as `rho/(1-rho)`, which is 3.8 times the mean service time; at 90% it is 9.0 times.
The same hold volume that reads as 1.29 hours per day on a monthly average is a support incident on
the day it arrives.

For comparison, the baseline's load under the same contact rates:

```text
Baseline holds 120: 100 legitimate at 78% and 20 compromised at 55%
Legitimate: 78 contacts * 22 = 1,716 minutes
Compromised: 11 contacts * 12 = 132 minutes
Total: 1,848 minutes = 30.8 hours = 1.03 hours per day
```

So the candidate adds 470 minutes, or 25%, to support load while adding 60 holds. That ratio —
7.8 minutes of support per additional hold — is the number the support owner needs, because it lets
them answer the only question they will ask: how many heads does this cost at peak.

### Step 5 — Test whether the reallocation is really doing the work

The candidate's improvement could come from better ranking, from the class split, or from the
reallocation of holds between classes. The table prices each component by constructing a variant that
removes it, holding everything else at the candidate's values. Hold-band precision is transferred to
the resized band by scaling the positive count proportionally, which is itself an assumption and is
stated as one.

| Variant | Class A holds | Class P holds | Prevented | Total cost | Saving vs baseline |
|---|---|---|---|---|---|
| Baseline | 40 | 80 | 73 | 389,545.70 | — |
| Candidate | 30 | 150 | 180 | 292,661.10 | 96,884.60 (24.9%) |
| Candidate ranking, holds frozen at 40/80 | 40 | 80 | 158 | 318,197.10 | 71,348.60 (18.3%) |
| Candidate with class P judged by class A thresholds | 30 | 55 | 154 | 318,945.30 | 70,600.40 (18.1%) |
| Candidate with class A judged by class P thresholds | 70 | 150 | 170 | 301,330.10 | 88,215.60 (22.6%) |
| Baseline ranking, holds reallocated to 30/150 | 30 | 150 | 80 | 383,666.70 | 5,879.00 (1.5%) |

Read the table in two directions. Comparing the candidate with the frozen-hold variant isolates the
hold volume: holding 130 instead of 180 while keeping the candidate's ranking costs 25,536.00 more,
so the extra 50 holds earn roughly 511 per hold against a hold cost of 64.30 — a return of 7.9 to 1
on the hold action specifically. Comparing the candidate with the two mis-thresholded variants
isolates the class split: applying class A thresholds to class P events costs 26,284.20, and applying
class P thresholds to class A events costs 8,669.00. Both directions are wrong, and the asymmetry
shows which one is the bigger mistake — under-protecting the high-loss class is 3.0 times as costly
as over-protecting the low-loss class.

The bottom row is the cheapest available improvement and it is worth stating carefully, because the
figure is small. Reallocating the incumbent's own holds to 30/150 with no new model at all recovers
only 1.5%, not the fourfold larger figure that a naive reading of the hold volumes suggests. The reason is that the
baseline's hold *precision* travels with its holds: moving 50 holds into the class P band while
keeping the old score means those holds are placed by the old rules, so the reallocation cannot find
the class P positives that the new scoring identifies. The reallocation is a rule change worth 1.5%,
and it is not a substitute for the ranking — a point worth stating plainly because the hold volumes
alone make the reallocation look four times more valuable than it is.

The frozen-hold result deserves its caveat before it is quoted. It shows that hold volume matters more
than the hold band edge, not that the band edge is irrelevant. That distinction surfaces in class P,
where the crossover at 7.46% sits close to a region the score populates densely. Step 6 tests
exactly that.

### Step 6 — Sensitivity, and the one input that reverses the decision

| Perturbation | Baseline | Candidate | Saving | Saving % |
|---|---|---|---|---|
| Base case | 389,545.70 | 292,661.10 | 96,884.60 | 24.87% |
| `L_P` reduced 30% (to 868.00) | 327,049.70 | 249,137.10 | 77,912.60 | 23.82% |
| `L_P` increased 30% (to 1,612.00) | 452,041.70 | 336,185.10 | 115,856.60 | 25.63% |
| `e_A` reduced to 0.50 | 400,929.70 | 314,717.60 | 86,212.10 | 21.50% |
| `e_P` reduced to 0.15 | 389,545.70 | 296,381.10 | 93,164.60 | 23.92% |
| `h` reduced to 0.90 | 391,497.20 | 299,044.10 | 92,453.10 | 23.62% |
| `c_hold` reduced to 32.15 | 385,687.70 | 286,874.10 | 98,813.60 | 25.62% |
| `c_hold` increased to 128.60 | 397,261.70 | 304,235.10 | 93,026.60 | 23.42% |

Both policies are recomputed under each perturbation, because a sensitivity study that perturbs only
the candidate answers a different question. The saving ranges from 21.50% to 25.63% across the whole
set, which is the kind of stability a decision needs. The largest mover is `e_A`, not the loss scalars:
halving login challenge effectiveness costs 10,672.50 of saving, because class A carries 1.12 million
of the 1.2 million events and its challenge band is the policy's main lever on volume.

Note a direction that inverts. A *lower* `c_hold` increases the saving, because the candidate holds
more than the baseline and therefore benefits more from cheap holds. The same is true of `L_P`:
increasing it favours the candidate because the candidate protects class P more. These interactions
mean the sensitivity table cannot be read as a set of independent margins; each is conditional on the
policy mix, and a different mix would reverse several signs.

Robustness in the aggregate is not the same as correctness in the policy, and the table hides the
input that actually flips the action choice.

`c_hold` is the input that governs the class P hold crossover:

```text
q_hc for class P = (c_hold - c_chal) / (L_P * (h - e_P))
At c_hold = 32.15:  (32.15 - 0.5090) / (1,240 * 0.69) = 31.641 / 855.60 = 3.70%
At c_hold = 64.30:  63.791 / 855.60 = 7.46%
At c_hold = 128.60: 128.091 / 855.60 = 14.97%
```

If the true hold cost is half the assumed value, the crossover falls to 3.70% and the policy should
hold far more class P events; if it is double, the crossover rises to 14.97% and many of the 150
holds should become challenges instead. The aggregate saving barely moves because holds are a small
share of total cost, but the action mix changes substantially. This is the characteristic failure of
aggregate sensitivity analysis: a metric can be stable while the decision it summarises is not.

The second reversal is not in this table at all, and it is the one that matters most. If the
recovery path cannot clear holds within the agreed SLA, the hold action must be removed from the
policy regardless of its arithmetic. A hold whose cost `c_hold = 64.30` excludes the cost of a
customer locked out for four days is mispriced by an order of magnitude, and the correct response is
to re-derive `c_hold` from measured clearance times rather than to adjust the threshold.

### Step 7 — State what the walkthrough did not establish

Six things are missing and each would change the answer.

The effectiveness inputs `e_A` and `e_P` are constructed. Section 12, F-02 shows why they cannot be
estimated from an observational log, and section 9 specifies the randomized slice that would identify
them.

The loss scalars `L_A` and `L_P` are constructed from a loss taxonomy that has not been applied to
any real incident. In practice these are the hardest inputs to obtain because the institution's cost
accounting does not separate investigation cost from fraud loss from customer detriment.

The hold costs exclude the long tail of clearance duration. The constructed `c_hold` assumes a
typical clearance measured in hours; a clearance measured in days would dominate every other term.

The class split assumes each event belongs unambiguously to one class. In production, a password
reset often occurs mid-login, and the classification is a design choice that can be gamed by an
attacker who sequences actions to land in the cheaper class.

No account-level clustering is modelled. A determined attacker targeting one account generates
multiple events, and the policy treats each independently, which under-counts both the loss (a
successful compromise is one event) and the friction (one customer may receive several holds).

The candidate's band volumes are assigned, not realized. A real evaluation needs the policy to be
live or simulated against logged scores, and this case has no logged scores.

**Takeaway:** the arithmetic supports a 24.87% cost reduction, of which the ranking contributes
18.3% and the hold-volume decision the remainder; every input that produces it is constructed rather
than measured.

## 7. The Finding

The constructed candidate reduces mature cost from 324.62 to 243.88 per 1,000 eligible events, a
24.87% paired reduction, while preventing 180 of 480 constructed compromises instead of 73. The
mechanism is two things working together: the policy prices two event classes separately, and it
places holds where a hold is worth its cost.

Four conclusions follow, ordered by what a reader should act on.

**The class split is worth about 6.5 percentage points, and under-protecting the high-loss class is
the larger error.** Judging class P events by class A thresholds costs 26,284.20, while judging class
A events by class P thresholds costs 8,669.00 — a 3.0 to 1 asymmetry. Pooling the classes into one
table is the default in most implementations because it is simpler to operate, and it is wrong for a
reason that is arithmetic rather than aesthetic: the classes differ in loss per event by 1.74x and in
challenge effectiveness by 2.6x, so no single threshold can be right for both.

**Hold volume, not the hold band edge, carries most of the hold-side value.** Cutting holds from 180
to 130 while keeping the candidate's ranking costs 25,536.00, which prices the marginal hold at
about 511 against a hold cost of 64.30 — a return of 7.9 to 1. The band edge matters much less, and
that is a warning against spending effort tuning a threshold whose region the score barely populates.

**A failed challenge is weak evidence and must not trigger a hold.** The posterior after one failure
is 0.55% against a hold threshold of 9.32%. A policy that escalates on challenge failure will issue
holds at roughly seventeen times the rate the cost schedule justifies, and the harm will land
entirely on legitimate customers with forgotten passwords and new phones.

**The binding constraint is recovery capacity, not model quality.** Holds cost 64.30 each in the
constructed ledger and that figure assumes an hours-scale clearance. A four-day clearance would make
the hold arm strictly dominated and force the policy back to challenges regardless of how well the
scorer ranks.

Two things this finding does *not* support deserve equal prominence, because both are plausible
misreadings of the hold volumes alone. Reallocating the incumbent's own holds returns only 1.5%,
because the baseline's hold precision travels with its holds. And the improvement is not attributable
to better discrimination alone: the frozen-hold variant, which keeps the candidate's ranking but the
baseline's hold volume, recovers 18.3% of the 24.87%, so roughly a quarter of the benefit comes from
the hold-volume decision rather than from ranking.

The finding's status is `not-measured`. It is a demonstration that the arithmetic of the decision
supports these thresholds, not evidence that any product would realise a 24.87% reduction.

**Takeaway:** the case's answer is a class split plus a hold-volume decision, and its weakest joint is
recovery rather than ranking.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

Five jobs, each with a distinct cadence and failure consequence.

The **label assembly job** joins confirmed-compromise outcomes to event records and writes an
immutable per-cohort-week snapshot. Corrections arrive as new versions with a reason, never as
in-place edits. Maturity is tracked as a curve: for each cohort week, the share of events whose
outcome has landed, plotted against days since the event. No week below 90% maturity enters training,
and the threshold is stated because a cohort at 70% maturity is not merely noisier — it is biased
toward loud attacks.

The **signal snapshot builder** replays the signal definitions against raw history to produce
training features. A signal that cannot be rebuilt from a snapshot is removed, not approximated,
because the approximation is exactly the leakage the snapshot exists to prevent.

The **training job** fits the scorer and calibration on time-ordered folds. Random folds are
forbidden here for a specific reason: device and account aggregates are stable within a random split
but change across a temporal one, so a random split reports a performance the deployed model cannot
reach.

The **threshold solver** takes the calibrated probabilities, the two class tables, and the signed
cost schedule and solves for the band edges that minimize expected cost subject to the hold budget.
It runs per class and it refuses to run if the cost schedule is older than a stated age, because a
stale schedule silently re-prices every action.

The **recovery simulation** takes the hold volume the solver produces and simulates support load
against staffed capacity, including a peak-day profile. If the simulation fails the capacity check,
the solver's output is rejected and the hold budget is tightened. This closes the loop that most
programmes leave open until the support team escalates.

### 8b. Online / Serving / Real-Time Path

```text
[Authentication or protected-action event]
        |
        v
[class resolution + eligibility + capability check] --cannot challenge--> [class no-challenge table]
        |
        v
[signal fetch with freshness stamps] --stale/error--> [incumbent rules, logged flag]
        |
        v
[scorer] -> [calibration] -> [class band table] -> [action + reason code]
        |
        +--> allow: respond, log
        +--> challenge: invoke provider, await result, re-evaluate on failure
        +--> hold: persist, create recovery ticket, notify customer
                     |
                     v
              [recovery queue with SLA clock]
```

Five design decisions in that path are load-bearing.

**Re-evaluate after a challenge result, do not short-circuit.** A failed challenge is new evidence
with a known likelihood ratio, so the correct behaviour is to recompute the posterior and choose
again from the same three actions. A policy that escalates directly from failure to hold skips this
step and over-holds by construction.

**Persist the decision before the response.** The decision record is the only artefact that proves
what the policy knew. Without it, the challenge-failure posterior cannot be recomputed later and the
evaluation in section 9 cannot be run.

**A hold creates a ticket, not a status flag.** The hold must produce a support item with an owner,
an SLA clock, and a maximum duration after which it auto-escalates. A status flag with no ticket is
an indefinite lockout.

**Idempotency keyed on the event.** Retries with the same event identifier return the stored action. A
customer must never see allow followed by hold for one attempt.

**Unknown band is a hard error.** If the calibrated probability falls outside the table, fail rather
than default to allow. Defaulting to allow on a configuration error converts a deployment mistake
into an unlogged security hole.

### 8c. Key Design Decisions

| Decision | Alternative rejected | Reason | Residual risk |
|---|---|---|---|
| Two class tables | One pooled table | Loss and effectiveness differ by class | Class misassignment at the boundary |
| Re-evaluate on challenge failure | Escalate to hold on failure | Failure is weak evidence (0.55% posterior) | Extra logic and a second provider call |
| Hold requires a ticket with SLA | Hold sets a flag | Prevents indefinite lockout | Support integration complexity |
| Capability check before ranking | Rank then attempt challenge | Action must exist to be chosen | Excludes channels from protection |
| Fail-closed on unknown band | Default allow | Prevents silent exposure | Availability during a bad deploy |
| Recovery simulation gates hold budget | Ship and monitor | Capacity is not elastic within a month | Slower change cadence |

### 8d. Architecture artifacts and operational readiness

Before cutover the following must exist as auditable artefacts, not as intentions: the signed cost
schedule with owner and date; the two band tables with version history; the signal register with
permitted purpose and retention per field; the snapshot builder and its leakage test suite; one
successful full replay of a historical week; the recovery runbook with a named owner and a maximum
hold duration; the support capacity simulation output; the rollback procedure and its measured
execution time; the monitoring dashboard with alert thresholds; and the model risk approval with its
conditions written as testable statements.

The rollback deserves a specific note because the fastest rollback is not a model redeploy. It is a
band-table swap or a switch flag routing to the incumbent rules. The rehearsal must include the case
where the new table is corrupt, because that is the failure a flag swap actually fixes.

**Takeaway:** three of the five load-bearing choices are about evidence and recovery, not about the
model.

## 9. Evaluation & Validation

The evaluation contract, stated so it can be falsified:

| Element | Definition |
|---|---|
| Numerator | Mature unauthorized-access loss plus challenge cost, hold cost, and support minutes converted at an agreed rate |
| Denominator | All eligible events assigned to the policy in the cohort, per class and pooled |
| Population | Established-account authentication and protected-action events, one product, one currency |
| Window | Thirty-day cohort, sixty-day outcome attribution |
| Comparator | Incumbent authentication rules on the same events in the same window |
| Guardrails | Legitimate challenge pass rate, hold clearance p95, support queue age, complaint rate, subgroup hold parity |
| Uncertainty | Paired difference with a bootstrap over account-level clusters |
| Status | **Not executed.** No data, no model, no deployment. |

Guardrails carry pre-agreed stop conditions. Hold clearance p95 is the sharpest: if p95 exceeds the
stated bound, the hold action is suspended regardless of its cost contribution, because the bound
encodes a customer-harm commitment that the identity-risk team does not own. The constructed policy
is at 180 holds with an assumed hours-scale clearance; the clearance distribution is the one thing
that must be measured before, not after, launch.

Sampling error on the class-level precision estimates, which would exist in any real cohort:

```text
Class A challenge precision, candidate: p = 140 / 9,000 = 0.015556
Standard error = sqrt(0.015556 * 0.984444 / 9,000) = sqrt(0.0000017015) = 0.0013044
95% interval = [0.01300, 0.01811] i.e. [1.30%, 1.81%]

Class P hold precision, candidate: p = 58 / 150 = 0.386667
Standard error = sqrt(0.386667 * 0.613333 / 150) = sqrt(0.0015810) = 0.039762
95% interval = [0.30873, 0.46460] i.e. [30.87%, 46.46%]
```

The class P hold interval spans 15.6 percentage points. With 150 holds and 58 positives, the estimate
of hold precision is barely informative, and a policy tuned to a point estimate of 38.7% is tuned to
noise. This is a genuine problem in small-class identity decisions and it argues for priors rather
than point estimates in the band solver.

Sample size for a prospective test of class A challenge precision against a baseline of 1.20%, to
detect a 0.5 percentage point difference at 95% confidence and 80% power:

```text
n = 7.849 * p * (1 - p) / d^2 = 7.849 * 0.015556 * 0.984444 / 0.005^2
  = 7.849 * 0.015314 / 0.000025
  = 0.12020 / 0.000025 = 4,808 challenged events per arm
At 9,000 class A challenges per thirty days: 4,808 / 9,000 * 30 = 16.0 days per arm
```

Comfortable for class A. Not comfortable for class P, where the candidate issues only 2,400
challenges and 150 holds per month: a hold-arm test at the same precision would need multiples of the
monthly volume, which is why hold decisions are best validated retrospectively against a frozen
policy rather than prospectively.

Executed versus not executed:

| Check | Status | Consequence |
|---|---|---|
| Threshold derivations at three probabilities | Executed | Regions agree with direct cost minimization |
| Ledger closure | Executed | Components add exactly under the stated rounding convention |
| Challenge-failure posterior chain | Executed | 0.55% then 4.22% then 38.17% |
| Support load including peak day | Executed | Peak utilization 79.2% of one agent |
| Baseline measurement | Not executed | 389,545.70 is constructed |
| `e_A`, `e_P` identification | Not executed | Requires a randomized slice |
| Hold clearance distribution | Not executed | The binding guardrail is unmeasured |
| Subgroup hold parity | Not executed | Undefined without a protected-attribute policy |
| Causal effect of the policy | Not executed | Requires a holdout, not a ledger |

The causal row is the one to press. The ledger compares two policies on one constructed cohort. The
candidate's volumes are assigned; in a real test they would be realised only if the policy were live,
and the events it acted on would then behave differently. Shadow mode verifies signals and latency
and establishes nothing about the cost difference.

**Takeaway:** every arithmetic check passed and no evidence exists; the two statements are compatible
and only one of them belongs in a decision paper.

## 10. Action Taken

No action was taken. No model was trained, no policy deployed, no account held.

Proposed sequence, ordered so each step can kill the programme cheaply:

1. Measure the hold clearance distribution at current volume. This is the guardrail that can veto
   everything else and it requires no model.
2. Sign the cost schedule: `L_A`, `L_P`, `c_chal`, `c_hold`, and the hold budget. Without this the
   band solver has no objective.
3. Measure the incumbent's per-class band volumes and precisions. Class-split baselines usually do
   not exist and must be constructed from logs.
4. Reallocate holds between classes using the incumbent's own score. This is a rule change, worth
   1.5% in the constructed ledger, and it needs no new model.
5. Build the signal snapshot builder and pass the leakage audit, with the post-event device linkage
   explicitly excluded.
6. Run the threshold solver against the signed schedule and report the action mix, not just the cost.
7. Run a randomized challenge slice to identify `e_A` and `e_P`. Without it, the challenge thresholds
   are assumptions.
8. Only then consider a prospective holdout with a pre-registered primary metric and a stop rule.

Step 4 is the one that deserves emphasis, because it is unusual to recommend a rule change before a
model. The constructed decomposition supports it with a caveat: the reallocation recovers only 1.5%
of the 24.87%, because the incumbent's hold precision travels with its holds. It is cheap, it is
measurable inside a week, and it will either work immediately or reveal that the class distinction is
not as clean in production as the design assumes.

**Takeaway:** the first four actions involve no model, and the fourth is measurable within a week.

## 11. Outcome & Results

`not-measured`. No production outcome exists. The constructed ledger, restated as arithmetic:

| Quantity | Baseline | Candidate | Difference |
|---|---|---|---|
| Total cost, cohort | 389,545.70 | 292,661.10 | -96,884.60 |
| Cost per 1,000 events | 324.62 | 243.88 | -80.74 |
| Prevented compromises | 73 | 180 | +107 |
| Prevented share | 15.2% | 37.5% | +22.3 pp |
| Holds issued | 120 | 180 | +60 |
| Holds on legitimate events | 100 | 106 | +6 |
| Challenges issued | 6,800 | 11,400 | +4,600 |
| Support minutes | 1,848 | 2,318 | +470 |
| Peak-day support utilization | 63.2% | 79.2% | +16.0 pp |
| Cost per additional prevented compromise | — | 57.94 | — |

Two rows carry the case. Holds rise by 50% while holds on legitimate events rise by 6%, which is the
whole content of the reallocation: the additional holds land almost entirely on events that were in
fact compromised. That is what a correctly priced hold threshold looks like, and it is the opposite
of the pattern a threshold-driven policy produces, where hold volume and hold precision rise together
and the customer harm rises with them.

The second row is support. The candidate adds 470 minutes of support work, a 25% increase, and pushes
peak-day utilization from 63.2% to 79.2%. In a monthly report that reads as a small increase; in a
queueing model it is the difference between a wait of about 1.7 times service time and one of about
3.8 times. Any deployment must carry the peak figure, not the average.

The honest limitation is that a single thirty-day cohort cannot detect a change in the attack mix. If
credential stuffing falls and targeted phishing rises, `e_A` falls, the challenge thresholds rise, and
the policy under-protects exactly when protection matters. Four consecutive cohorts with a frozen
policy is the cheapest available test for this and it is the one most often skipped.

**Takeaway:** the arithmetic shows a 24.87% reduction and a near-zero increase in legitimate holds; the
support peak is the number that will decide whether it ships.

## 12. Failure Modes & What Went Wrong

Each mechanism below is stated with a discriminating check, because "monitor it" is not a control.

### F-01 — Post-event device linkage leaks the answer

**Mechanism.** The compromise investigation links the attacker's device to the account after the
fact, and that linkage is written back to the device table. A feature that reads "devices associated
with this account" then contains a device that was, at decision time, unknown. The model learns to
recognise the attacker's device because the future was written into the present.

**Symptom.** Offline performance far above shadow performance; device-related features dominant;
the strongest single feature has a suspiciously clean relationship with the label.

**Competing causes.** Genuinely strong device signal from prior confirmed events on the same device;
a serving-side feature bug; a population shift between training and shadow.

**Discriminating check.** For a sample of positive events, print the attacker device's first-seen
timestamp next to the decision timestamp. Any device first seen after the decision but present in the
feature vector is a leak, and the count is the size of the problem. Then rebuild the device table as
of each decision timestamp and re-score.

**Correction.** Build all device features from snapshots keyed on decision time; treat post-event
linkage as an evaluation-only artefact.

**Verification.** No feature vector in the replay contains a device whose first-seen timestamp is
later than the decision, verified by an automated assertion rather than a spot check.

### F-02 — Challenge effectiveness estimated from challenged events

**Mechanism.** `e` is computed as one minus the share of challenged compromises that completed. But
compromises prevented by the challenge never complete and therefore never appear, so the
observational estimator is biased toward zero. In the limit it reports zero effectiveness, which
makes every challenge look worthless and pushes the policy toward holds.

**Symptom.** Fitted effectiveness far below the plausible range; a band solver that eliminates the
challenge action entirely; a policy that holds on probability bands far below the hold crossover.

**Competing causes.** Genuinely low effectiveness on this route mix; a cost schedule that understates
hold cost so holds look cheap; a channel where the challenge silently degrades to a no-op.

**Discriminating check.** Run a randomized slice with challenge withheld and compare completed
compromise rates in the same score band. The gap between the withheld and challenged arms is the
effectiveness. If the observational estimate and the randomized estimate disagree by more than the
binomial interval, the observational one is biased.

**Correction.** Estimate per class from the randomized slice, or carry a wide prior and report the
decision across it.

**Verification.** The band table is stable across the plausible effectiveness range; if it is not,
the decision is not identified and must not ship.

### F-03 — Reporting propensity learned as compromise risk

**Mechanism.** Customers who use the app, are younger, or hold higher-value accounts report
compromise more often. If the model's positive class is "reported and confirmed", it learns the
determinants of reporting, not of compromise. It then over-scores precisely the segments that report
loudly and under-scores quiet segments, which is both a performance defect and a fairness defect.

**Symptom.** Score distribution differs sharply across segments whose compromise rates should be
similar; hold rate by segment diverges while confirmed-compromise rate by segment does not.

**Competing causes.** A genuine difference in attack targeting by segment; a genuine difference in
loss per compromise by segment; a labelling difference in investigation depth.

**Discriminating check.** Hold the outcome definition fixed and compute the hold rate and the
confirmed rate by segment. A segment with high hold rate and proportional confirmed rate is being
correctly targeted; a segment with high hold rate and *lower* confirmed rate is being penalised for
reporting or for a proxy.

**Correction.** Investigate quiet segments deliberately to equalise label ascertainment; add an
ascertainment-aware label definition; evaluate parity on the hold rate rather than on the score.

**Verification.** Segment hold rates converge once label ascertainment is equalised, and the parity
guardrail sits inside its agreed band.

### F-04 — Hold without a clearing path

**Mechanism.** A held account is marked in the identity store and the recovery ticket fails to
create, or creates in a queue nobody owns. The account remains locked until the customer calls, and
the call volume is invisible because there is no ticket to count.

**Symptom.** Hold count in the decision store exceeds ticket count in the support system; a rising
share of tickets categorised as "cannot log in" with no corresponding hold record; clearance time
with a heavy tail.

**Competing causes.** A genuine support backlog; a ticket-system outage; a policy that legitimately
holds for a long duration pending documentary verification.

**Discriminating check.** Reconcile decision-store hold rows against support tickets hourly by hold
identifier. Any hold without a ticket is the defect, and the reconciliation gap is its volume.

**Correction.** Make ticket creation transactional with the hold, with an alert on failure; give the
hold a maximum duration after which it self-escalates.

**Verification.** Hourly reconciliation shows zero holds without tickets, and p95 clearance sits
inside the agreed bound for four consecutive weeks.

### F-05 — Challenge failure escalated straight to hold

**Mechanism.** The policy treats a failed challenge as proof and issues a hold. The posterior after
one failure is 0.55% against a 9.32% hold threshold, so this over-holds by roughly a factor of
seventeen. The cost lands on legitimate users with a new phone, a cleared cookie, or a typo.

**Symptom.** Hold volume correlates with challenge volume rather than with the score; hold precision
falls as the challenge rate rises; support contacts concentrate on password-reset and new-device
themes.

**Competing causes.** A genuine rise in attacks that fail challenges; a challenge provider delivering
codes late so that legitimate users time out; an accessibility defect in the challenge UI.

**Discriminating check.** Compute hold precision separately for holds that followed a failed
challenge versus holds issued from the score band. If the former is materially lower, the escalation
rule is the defect. Then check the failed-challenge rate for legitimate users by device age, which
separates provider and accessibility causes from attack causes.

**Correction.** Re-evaluate the posterior after failure and choose from the same three actions; never
map failure directly to hold.

**Verification.** Hold precision for failure-triggered holds is statistically indistinguishable from
score-triggered holds.

### F-06 — Class misassignment as an attack strategy

**Mechanism.** A password reset performed mid-login can be classified as a login event or as a
protected action depending on where the classifier boundary sits. An attacker who sequences actions
to land in the cheaper class faces the login thresholds, which are 15 times more permissive, and
obtains the protected outcome anyway.

**Symptom.** A shift in the event-type mix over time; a rising share of compromises whose preceding
events were all classified as class A; a class P volume that falls while protected outcomes rise.

**Competing causes.** A genuine client-version change altering event emission; a legitimate product
change moving recovery into the login flow; a logging defect that mislabels a subset.

**Discriminating check.** Reconstruct the full session for a sample of confirmed compromises and
tabulate the class of every event in it. If compromises consistently arrive through a sequence of
cheap-class events, the boundary is being exploited. Compare the sequence against the pre-deployment
baseline to separate exploitation from a product change.

**Correction.** Classify by *outcome capability* rather than by event label — any session that can
change credentials is class P from its first event — and re-derive the tables on the new definition.

**Verification.** Session-level reconstruction shows compromises arriving through the expensive class,
and the class mix is stable across cohorts.

### F-07 — Idempotency loss produces contradictory actions

**Mechanism.** A retried event is scored twice with slightly different signal freshness, producing
allow on the first call and hold on the second. The customer experiences a hold they were told they
had passed, and the support conversation starts from a contradiction.

**Symptom.** Two decision rows for one event identifier with different actions; customer complaints
that describe an allow followed by a lockout.

**Competing causes.** A genuine second event with the same user-visible appearance; a policy version
deployed between the two calls; a client retry storm.

**Discriminating check.** Group decision rows by event identifier and count distinct actions.
Any group with more than one action is either an idempotency defect or an explicit supersession that
must carry a supersession record; the absence of that record is the discriminator.

**Correction.** Store the decision keyed on event identifier with a unique constraint; treat a changed
payload as an explicit supersession with a linking record.

**Verification.** Injected retry and amendment traces produce exactly one active action per event with
a complete audit chain.

**Takeaway:** five of these seven failures are invisible in the primary cost metric and visible only
in a deliberate secondary check.

## 13. Transferable Lesson

The transferable structure is: a decision whose evidence is a *signal about a person* rather than a
measurement of an object, where the intervention itself changes what can be observed, and where the
heaviest action depends on an operational path outside the decision system.

Six claims transfer, each with a boundary.

**Price the action, then place it by class.** Where event classes differ in stakes or in
effectiveness, one threshold is wrong for all of them. Boundary: this fails when the class boundary
is attacker-controllable, which is why classification must key on capability rather than on event
label.

**A failed test is weaker evidence than it feels.** At low base rates, even a strong likelihood ratio
leaves a small posterior. Boundary: the arithmetic assumes the failure rate is measured on
legitimate cases; measuring it on all cases destroys the estimate.

**Update after the intervention, do not short-circuit.** The correct response to a failed challenge
is a recomputed posterior and a fresh action choice. Boundary: this requires the intervening
observation to have a stable likelihood ratio, which is false if the attack route changes mid-flow.

**An action's cost includes the path that undoes it.** A hold is only as cheap as its clearance.
Boundary: where clearance is immediate and automated, the hold approaches a decline and the
distinction collapses.

**Report the peak, not the average, when capacity is the constraint.** Boundary: for a
throughput-limited system with a flat arrival profile, the average is fine.

**Recovery is an attack surface, not a customer-service detail.** Every hardening of recovery against
attackers hardens it against the holder. Boundary: where recovery is fully automated and
identity-provider-backed, the tension moves to the provider.

The lesson does not transfer to application fraud, and the reason is structural rather than a matter
of degree. An application has no behavioural baseline, so "deviation from this account's history" is
undefined; every device and network signal is a first observation, and the class of legitimate
explanation is wider because a genuinely new customer has no history by construction. The loss per
event is also concentrated in a way this case's constant-`L` assumption cannot represent: one
synthetic identity may generate losses across many products over months. Application decisions
therefore need documentary verification and manual review with turnaround in days, and their
arithmetic is a portfolio problem rather than a per-event one.

**Takeaway:** the reusable artefact is the class-split, path-priced action table; the specific
thresholds do not travel.

## 14. Connections & Cross-References

- [Real-time payment-card fraud detection under a friction budget](./01_real_time_payment_card_fraud_detection.md)
  shares the three-action structure but is bounded by a millisecond deadline rather than by a
  recovery path; its binding resource is latency, this case's is support capacity.
- [Claims investigation triage under delayed adjudication](./02_insurance_claims_fraud_detection.md)
  is the same evaluation-mismatch pattern with a label delay measured in months and a human review
  queue instead of automated actions.
- [AML alert prioritization at fixed investigator capacity](./04_aml_transaction_monitoring_system.md)
  is the capacity-bound variant where the only action is queue position.
- The unit cheat sheets carry the fraud taxonomy and the decision-grain vocabulary this case assumes;
  the cutoff-and-losses sheet generalises the break-even derivations used in section 6.
- New-account application fraud is the neighbouring decision this case deliberately excludes. It
  shares the identity-verification framing but not the availability assumption, and it is a
  portfolio problem rather than a per-event one.

The contrast worth holding: the card case's scarce resource is a *deadline*, the claims case's is an
*investigator hour*, this case's is a *recovery slot*, and the AML case's is a *queue position*. Four
different scarcities, one arithmetic.

### 14a. Interview pack

**Q01. Why are there two band tables instead of one?**
Answer: because loss per compromise differs by 1.74x between classes and challenge effectiveness
differs by 2.6x, so a single threshold cannot be correct for both.
Reasoning: the threshold is `c/(e·L)`, so both `e` and `L` move the answer; a pooled table silently
picks a compromise between two wrong answers.
Follow-up: what does pooling cost? About 4 percentage points of the available reduction in the
constructed ledger: mis-thresholding class P costs 26,284.20, about 6.7% of the baseline.
Red flag: "one table is simpler to operate" as the deciding argument.
Score: 0 no reason; 1 says classes differ; 2 quantifies the difference; 3 prices the pooling error.

**Q02. Compute the challenge threshold for class P at `c_chal = 0.5090`, `e = 0.28`, `L = 1,240`.**
Answer: 0.1466%, or about one event in 682.
Reasoning: `0.5090 / (0.28 * 1,240) = 0.5090 / 347.20`.
Follow-up: the class P base rate is 0.225%, which exceeds the threshold. What does that imply?
Answer: challenging protected-action events nearly at random is marginally worth its cost, so the
class P challenge band should be broad.
Red flag: dividing by `L` alone, giving 0.041%, which ignores that the challenge often fails.
Score: 0 wrong; 1 quotes the number; 2 notes the base rate exceeds it; 3 draws the breadth
conclusion.

**Q03. A challenge failed. Do you hold the account?**
Answer: no. The posterior after one failed class A challenge is 0.55%, against a hold threshold of
9.32%, so it is seventeen times too small.
Reasoning: prior odds 0.000268 times a likelihood ratio of 20.57 gives posterior odds 0.005513.
Follow-up: what would justify the hold? Two further signals — a cross-device password reset and a
new payee — take the posterior to 38.17%, above the 35.86% crossover.
Red flag: "failure means it is probably an attacker."
Score: 0 holds; 1 says no without numbers; 2 computes the posterior; 3 chains the update to a hold
decision.

**Q04. Derive the hold-versus-challenge crossover and evaluate it for both classes.**
Answer: `(c_hold - c_chal)/(L(h - e))`; class A 35.86%, class P 7.46%.
Reasoning: the numerator is the extra cost of holding, the denominator the extra loss a challenge
allows.
Follow-up: why is class P's crossover so much lower? Because `h - e` is 0.69 there versus 0.25 for
class A, so a challenge leaves far more loss on the table.
Red flag: comparing hold cost to challenge cost without the effectiveness gap.
Score: 0 no derivation; 1 one class; 2 both; 3 explains the class difference mechanistically.

**Q05. The candidate issues 60 more holds than the baseline. Is that a customer-harm regression?**
Answer: no. Legitimate holds rise from 100 to 106 while total holds rise from 120 to 180, so the
additional holds land on events that were in fact compromised.
Reasoning: hold precision rises from 16.7% to 41.1% while the volume grows, which is the signature of
a correctly priced threshold rather than a loosened one.
Follow-up: what pattern would indicate harm? Volume and legitimate count rising together.
Red flag: reading total volume as harm without the precision split.
Score: 0 yes; 1 notes precision; 2 computes both; 3 names the harmful pattern.

**Q06. What is the marginal return on the extra friction?**
Answer: the candidate spends 6,199.40 more on holds and challenges to prevent 107 additional
compromises, which is 57.94 per prevented compromise against an average loss per compromise of about
909.69.
Reasoning: prevented events rose from 73 to 180 while cost rose by the hold and challenge deltas.
Follow-up: why is the return so large? Because holds are cheap relative to the loss and the
reallocation targets the class with the higher loss.
Red flag: comparing total cost to total loss without differencing.
Score: 0 no number; 1 total saving; 2 marginal return; 3 also notes the average loss it is compared
to.

**Q07. Class P has 150 holds and 58 positives. How precise is that estimate?**
Answer: 38.67% with a 95% interval of 30.87% to 46.46%, a span of 15.6 points.
Reasoning: binomial standard error `sqrt(p(1-p)/n)` with n = 150.
Follow-up: what should the band solver do differently? Use a prior rather than the point estimate,
because the interval is too wide to tune against.
Red flag: quoting 38.7% as if it were known.
Score: 0 point estimate only; 1 recognizes uncertainty; 2 computes the interval; 3 changes the
solver's use of it.

**Q08. Why is a pass weaker evidence than a failure is?**
Answer: because legitimate failure rates are high (3.5%), so the pass likelihood ratio is 0.29 versus
a failure likelihood ratio of 20.57.
Reasoning: the pass probability under compromise is `1 - e = 0.28`, close to the legitimate pass rate
of 0.965 in ratio terms only after division; the absolute pass rate for legitimate users is what
dilutes it.
Follow-up: what does a pass reduce the posterior to? 0.0078%, a threefold reduction.
Red flag: treating a pass as clearance.
Score: 0 treats pass as proof; 1 says it is weak; 2 gives the likelihood ratios; 3 states the
posterior.

**Q09. Design the test that identifies challenge effectiveness.**
Answer: a randomized slice in which challenge is withheld from a small eligible band, comparing
completed-compromise rates against the challenged arm in the same score band.
Reasoning: withholding creates the counterfactual that the observational log cannot contain.
Follow-up: what is the ethics constraint? The withheld band must be chosen where a completed
compromise is bounded in loss, and the band must be small enough that the expected harm is below a
pre-agreed limit.
Red flag: estimating effectiveness from the challenged set.
Score: 0 observational; 1 mentions randomization; 2 designs the comparison; 3 addresses the harm
bound.

**Q10. The support team says a hold costs about four hours of queued customer time. What changes?**
Answer: `c_hold` rises far above 64.30, which raises the hold threshold and moves the class P
crossover from 7.46% toward 15% or beyond, eliminating most of the 150 holds.
Reasoning: `q_hold = c_hold/(hL)` and `q_hc = (c_hold - c_chal)/(L(h-e))` both scale with hold cost.
Follow-up: recompute at `c_hold = 128.60`: crossover 14.97%, so holds would drop sharply.
Red flag: keeping the threshold and absorbing the cost.
Score: 0 ignores; 1 raises the threshold; 2 recomputes; 3 notes the action mix changes while the
aggregate saving barely moves.

**Q11. What happens if credential stuffing falls and targeted phishing rises?**
Answer: `e_A` falls, because the challenge no longer separates attacker from holder, so the class A
challenge threshold rises and the policy under-protects unless it is re-derived.
Reasoning: the threshold is inversely proportional to effectiveness.
Follow-up: what monitors it? Completed-compromise rate per score band over time, plus the share of
compromises preceded by a successful challenge.
Red flag: assuming effectiveness is a property of the challenge rather than of the route mix.
Score: 0 no effect; 1 notes drift; 2 explains the mechanism; 3 names the monitor.

**Q12. An attacker sequences a password reset inside the login flow to land in class A. What is the
fix?**
Answer: classify by capability, not by event label: any session that can change credentials is class
P from its first event.
Reasoning: the class boundary must not be attacker-controllable, or the cheaper table becomes the
attack route.
Follow-up: how do you detect that it happened before you fix it? Reconstruct sessions for confirmed
compromises and tabulate the class of every event.
Red flag: adding a rule for the specific sequence, which invites the next sequence.
Score: 0 no fix; 1 adds a rule; 2 reclassifies by capability; 3 also names the detection.

**Q13. Why is peak-day support utilization reported instead of the monthly average?**
Answer: because capacity is not elastic within a month and queue wait grows as `rho/(1-rho)`; 79.2%
on the peak day means a wait of roughly 3.8 service times against 1.7 at the baseline's 63.2%.
Reasoning: the average across 30 days hides the concentration that a stuffing spike creates.
Follow-up: how many agents does the candidate need on a peak day? 309.1 minutes against 390 minutes
of capacity for one agent, so one, with almost no margin for a second incident.
Red flag: quoting 1.29 hours per day as the capacity answer.
Score: 0 monthly average; 1 peak; 2 peak with utilization; 3 reasons about the wait.

**Q14. What does the ledger omit that could reverse the sign?**
Answer: the cost of a hold that is never cleared, the customer's non-reimbursed time, the
investigation cost of each hold, and the revenue effect of lost access. The first is the one that can
reverse the sign, because it is unbounded.
Reasoning: a bounded omission shifts a number; an unbounded one changes the ordering.
Follow-up: which of these is measurable in a week? Investigation cost per hold, from ticket time
stamps.
Red flag: claiming the ledger is complete because it closes arithmetically.
Score: 0 claims completeness; 1 names one; 2 names several; 3 ranks by the ability to reverse the
sign.

**Q15. A reviewer proposes a single risk score used for both login and application fraud. Reply.**
Answer: refuse on grounds of identification, not preference. An application has no behavioural
baseline, so the deviation features that carry this case's signal are undefined, and the loss is a
portfolio quantity spread over months rather than a per-event quantity.
Reasoning: the arithmetic requires `L` per event and a prior over an established account's history.
Follow-up: what could be shared? The signal register, the calibration discipline, and the recovery
guardrails, but not the thresholds.
Red flag: agreeing that a score is a score.
Score: 0 agrees; 1 refuses without reason; 2 explains the identification failure; 3 states what
transfers.

**Q16. Give the one-sentence version of this case's finding, with its evidence status.**
Answer: pricing two event classes separately and placing holds where they pay recovers about a
quarter of the constructed cost, and the entire figure is illustrative arithmetic with no measured
input.
Reasoning: the mechanism is the class split and the hold reallocation, not a better scorer.
Follow-up: which single measurement would most change the estimate? The hold clearance distribution,
because it can invalidate the hold action entirely.
Red flag: quoting 24.87% as a result.
Score: 0 wrong mechanism; 1 right mechanism; 2 adds the magnitude; 3 states the evidence status and
the decisive measurement.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Decision | Allow, step-up challenge, or protective hold for an established-account event. |
| Unit | One authentication or protected-action event; class A login, class P protected action. |
| Objective | Mature unauthorized-access loss plus friction and support, per 1,000 eligible events. |
| Thresholds | Class A: challenge above 0.0994%, hold above 9.32%, hold beats challenge above 35.86%. |
| Thresholds | Class P: challenge above 0.1466%, hold above 5.35%, hold beats challenge above 7.46%. |
| Failed challenge | Posterior 0.55% — far below any hold threshold; never escalate directly. |
| Constructed result | 389,545.70 → 292,661.10, a 24.87% reduction; hold reallocation alone is 1.5%. |
| Binding constraint | Hold clearance p95 and peak-day support capacity, not ranking quality. |
| Never infer | That a failed challenge proves an attacker, or that a passed one proves the holder. |
| Evidence | Illustrative arithmetic only; `not-measured`; no deployment observed. |
