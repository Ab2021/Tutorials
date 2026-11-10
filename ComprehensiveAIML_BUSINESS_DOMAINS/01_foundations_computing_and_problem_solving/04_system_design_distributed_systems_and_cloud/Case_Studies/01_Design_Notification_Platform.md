---
case_id: "CASE-04-001"
case_title: "Design a Notification Platform"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "idempotent-delivery"
secondary_patterns: ["correctness-contract", "admission-control"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "messaging lead"
source_index: "04_system_design_distributed_systems_and_cloud/Case_Studies/INDEX.md"
review_trigger: "Any change to workload, decision authority, or acceptance metric"
---

# Design a Notification Platform

**Evidence declaration:** This is an ILLUSTRATIVE teaching case. The workload, the channel mix, the
provider prices, and every outcome figure below are constructed to make one design decision auditable.
Nothing here is a company deployment, a vendor benchmark, or a measured production result. Arithmetic
is recomputed in the text so a reader can check each step; where a value depends on a demand
distribution that no public source fixes, the value is marked ILLUSTRATIVE at the point of use.
**Implementation stage:** proposed — no component described here has been operated by the author.
**Editorial status:** draft; source verification and publication review remain OPEN.

## 1. The One-Line Framing

A notification platform is not a message router; it is a **permission machine** that must decide, for
each (recipient, event) pair, whether a send is still worth the recipient's attention at the moment the
send would land — and must make that decision reproducible from a stored record after the fact.
The consequential design choice in this case is therefore not "which queue" or "which provider", but
**what the unit of idempotency is**, because that single choice determines duplicate rate, cost,
incident blast radius, and whether the audit trail can answer "why did this person get three copies".

The case follows a messaging lead who must commit to a delivery contract before traffic exists, using
only capacity arithmetic, a cost model, and a complaint budget. The alternative — the incumbent
pattern — is a fan-out service that calls provider APIs synchronously from the request path with a
best-effort retry wrapper. That baseline is simple and reversible, and it is wrong for a specific,
nameable reason: it makes the number of sends a function of retry weather rather than of intent.

The one-line decision: **route every intent through a deterministic decision step keyed on a
content-and-recipient identity, then deliver at-least-once behind that key, and never let a retry
create a second user-visible notification.** Everything downstream of that sentence is engineering
detail; everything upstream is policy the product owner owns.

## 2. Context & Background

The product surface is a consumer application with three notification classes that behave nothing
alike. **Transactional** messages confirm something the recipient just did: a one-time passcode, a
receipt, a password reset, a delivery window change. They are expected within seconds, they are
useless when late, and a duplicate is mildly annoying but a missing one blocks the user entirely.
**Engagement** messages try to pull a lapsed user back: "your saved search has 4 new results",
"your cart is still open". Late is tolerable by hours; duplicates destroy trust quickly because the
recipient notices repetition, not lateness. **Account and compliance** messages carry legal weight:
policy changes, billing failures, data-export completion. They must arrive, must be retained as
evidence, and must not be silently suppressed by fatigue logic.

Those three classes share infrastructure and share almost no acceptance criteria, and the first
architectural mistake in this domain is a single queue with a single retry policy serving all three.
A passcode that waits behind forty thousand marketing pushes is an outage even though every component
reports healthy. A marketing push that retries eight times because it shares a retry budget with
passcodes turns a provider blip into a complaint spike. The platform therefore has to be designed
around **class-separated admission**, with a shared decision ledger underneath so that cross-channel
deduplication still works when the classes are physically separated.

Scale is deliberately modest so that the arithmetic stays checkable by hand. The planning scenario is
36,000,000 notification intents per day (ILLUSTRATIVE), a figure chosen because it produces a mean of
about 417 intents per second and a peak minute near 2,100 per second — small enough to reason about,
large enough that per-message costs and per-key storage costs are visible rather than negligible.
The user base is 3,200,000 daily active recipients, which yields 11.25 intents per active user per day
on average. That average is the most misleading number in the entire case, and §6 shows why.

The organisation has one on-call rotation for messaging, a product team that owns send policy, and a
separate trust-and-safety team that owns suppression lists. The messaging lead owns the delivery
contract and the capacity plan; the product team owns frequency caps and quiet hours; trust and safety
owns the "do not contact" list and the complaint response process. This split matters because the
three owners can each individually break the platform: product can raise caps past capacity, safety can
add a suppression rule that starves a class, and messaging can change the idempotency key and silently
convert a deduplicated stream into a duplicated one.

Existing evidence capture is poor, and that is part of the starting condition rather than a detail to
be fixed later. The incumbent fan-out service logs an HTTP status from the provider but does not log
the decision that preceded the call, so nobody can answer whether a missing notification was never
decided, decided-and-dropped, or decided-and-failed. When recipients complain, support escalates to
engineering with a user identifier and a timestamp, and engineering greps application logs that roll
over in seven days. Any redesign has to make that question answerable cheaply, because the complaint
process is the only feedback loop the platform has.

The business pressure is a cost line and a complaint line moving in opposite directions. Growth wants
more engagement sends because engagement sends correlate with session starts; support wants fewer
sends because complaint volume is the top driver of contact-centre load. Both are measuring the same
stream from different ends, and neither has a number that ties send volume to recipient tolerance.
The decision contract in §3 exists to create that tie: a complaint budget expressed per delivered
notification, with the platform required to stay under it while maximising the sends that survive it.

Two prior attempts are relevant context because they constrain what the lead is allowed to propose.
A frequency-cap pilot shipped as a per-user counter in the application database and failed under
contention: the counter was read-modify-written on a row that every notification for that user
touched, so a burst for one popular account serialised on a single row and stalled the shared
database connection pool. It was rolled back within a day. A second attempt moved provider calls
behind a queue with unlimited retries and produced a three-hour duplicate storm when a provider
returned 500s for requests it had actually accepted; the retry wrapper could not distinguish
"rejected" from "accepted but unacknowledged". Both failures point at the same missing idea — a
decision record that is separate from the delivery attempt — and the case is organised around
supplying it.

The competitive context is thin on purpose. Notification platforms are not differentiated by
transport; every provider sells the same push, email, and SMS primitives at commodity prices. They are
differentiated by whether the recipient trusts the stream, which is a property of the decision layer,
not the transport layer. That asymmetry is why this case spends most of its length on identity,
idempotency, and admission rather than on queue technology.

**Takeaway:** the context establishes three classes with incompatible acceptance criteria, a modest
but spiky workload, three owners with independent break-powers, and a complaint process that is the
only real feedback loop.

## 3. The Business Question

The question is stated as a decision with an owner, a deadline, and a set of admissible actions,
because a question without those three attributes cannot be closed and will be reopened by whoever
disagrees with the answer.

| Field | Contract |
|---|---|
| Question | Should the messaging team replace synchronous fan-out with a decision-ledger platform, and on what idempotency key? |
| Owner | messaging lead (accountable), product owner (frequency policy), trust and safety (suppression) |
| Trigger | The next provider contract renewal, plus any complaint-rate breach above 0.0005 per delivered message |
| Actions | (a) keep synchronous fan-out; (b) adopt ledger with key = (recipient, event_id); (c) adopt ledger with key = (recipient, event_id, channel); (d) adopt ledger but defer the dedup window change |
| Deadline | Four weeks before renewal, so that the decision can be priced into the contract |
| Objective | Minimise complaints per delivered notification subject to class-specific latency ceilings and a cost ceiling |
| Hard constraint | No user-visible duplicate for a single intent; no suppression of transactional class by fatigue logic |
| Soft constraint | Cost per delivered notification must not exceed the incumbent's by more than 8 percent |
| Scope | One planning day of 36,000,000 intents across three classes and three channels |
| Out of scope | Provider selection, template authoring, in-app inbox rendering |

The admissible actions are worth reading carefully, because options (b) and (c) differ by a single
field and produce a six-figure annual cost difference. Keying on (recipient, event_id) means one
intent can be delivered on exactly one channel: if push fails because the device token is stale, the
ledger records a terminal state and email never fires. Keying on (recipient, event_id, channel) allows
channel fallback, at the cost of an additional dedup surface per channel and a new failure mode where
push and email both land and the recipient sees the message twice through different media. The lead
must choose, and the complaint model in §6 is what makes the choice decidable rather than aesthetic.

The objective needs one more clause to be operational: **minimise complaints per delivered
notification**, not complaints per intent. Complaints per intent falls if the platform simply sends
less, which is not a business win. Complaints per delivered notification is a quality ratio that
punishes both duplication (which raises complaints at fixed delivery) and irrelevance (which raises
complaints at fixed delivery through a different mechanism). It has the property that a platform which
delivers nothing has an undefined ratio rather than a good one, so it must be paired with a delivery
floor for the transactional class: at least 99.5 percent of passcode intents accepted by a provider
within 10 seconds.

The decision must also survive contact with a metric that is not the objective. Delivery rate — the
fraction of intents that reach a provider-accepted state — is easy to compute and easy to game: a
platform that retries indefinitely will show a high delivery rate precisely because it is generating
duplicates. The lead explicitly refuses to accept delivery rate as the primary metric and instead uses
it as a guardrail that must stay above a floor while the complaint ratio is optimised. This is the
same distinction as precision versus recall: one number cannot carry both, and choosing one as the
objective while the other sits as a constraint is the only honest formulation.

Finally, the question has a falsification condition. If, after four weeks of shadow traffic, the
ledger-based path shows a complaint ratio indistinguishable from the synchronous path at equal
delivery, then the added machinery buys nothing and option (a) stands. Stating that in advance
prevents the common failure where a redesign is declared successful because it shipped.

**Takeaway:** the question is closeable because it names an owner, four actions including "do
nothing", a latency floor, a cost ceiling, and a pre-registered falsification condition.

## 4. Assumptions & Constraints

Each assumption below carries a verification method and a fallback that the platform can execute
without a human in the loop, because an assumption whose violation requires a meeting is not a control.

- **A-01 — Intent identity exists before delivery.** Every send originates from a producer that
  supplies a stable `event_id` for the underlying business fact (an order shipped, a passcode
  requested). Verification: sample 1,000 events per class and confirm the producer emits the same
  `event_id` on replay. If violated: route the class through a synthetic key derived from
  (recipient, template, 10-minute bucket) and record the substitution in the ledger.
- **A-02 — Recipient identity is resolvable at decision time.** The decision step needs a canonical
  recipient key, not a device token, because a recipient may have four devices and two email
  addresses. Verification: measure the share of intents whose recipient resolves to zero or more than
  one canonical key. If violated for more than 0.5 percent: hold the intent and alert, rather than
  guessing, because a wrong resolution sends a stranger's data to the wrong person.
- **A-03 — The provider accepts an idempotency hint, or the platform can deduplicate before the call.**
  Not all providers honour a request-level dedup token. Verification: read the provider contract for
  the retry semantics of a 5xx response, which is the case that matters. If the contract is silent:
  treat every 5xx as "possibly accepted" and suppress the retry for engagement class while allowing
  one retry for transactional.
- **A-04 — Clock skew between producers and the decision service is bounded.** Quiet hours depend on
  local time computed from a stored timezone, and "do not send after" depends on the event timestamp.
  Verification: compare producer clock to NTP-corrected time on a sample of hosts; require p99 skew
  under 2 seconds, p100 under 30 seconds. If violated: compute quiet hours from the decision service's
  clock and the stored timezone, never from the producer timestamp.
- **A-05 — Suppression state is readable with low latency and is authoritative.** Unsubscribe must
  take effect before the next send. Verification: inject an unsubscribe and measure the time until the
  decision step observes it. If above 60 seconds: add a per-recipient suppression cache with explicit
  invalidation and fail closed (suppress) when the cache is cold.
- **A-06 — The incumbent path remains deployable during the transition.** Rollback must not require
  data migration. Verification: rehearse rollback in staging with a full day of shadow traffic. If it
  requires migration: the rollout plan is wrong and must be re-cut, not the rollback.
- **A-07 — Provider rate limits are per-account, not per-request-type.** This is the assumption that
  most often fails in practice, because a provider's published limit usually describes a burst bucket
  while the enforced limit is a sliding window measured over a minute. Verification: read the contract
  and then probe with a controlled ramp during a maintenance window. If the enforced limit is lower:
  size admission from the probed value, not the published one.
- **A-08 — The planning workload is a scenario, not a forecast.** 36,000,000 intents per day is a
  number chosen for arithmetic legibility. Verification: none is possible before traffic exists;
  therefore every capacity conclusion is stated as a function of the rate, and the sensitivity of each
  conclusion to the rate is shown in §6 and §11.

The constraints are the boundaries the design may not cross, and they are stated as inequalities so
that a reviewer can check compliance numerically rather than by opinion.

| ID | Constraint | Value | Why this value |
|---|---|---|---|
| C-01 | Transactional acceptance latency | p99 ≤ 10 s from intent to provider-accepted | A passcode arriving after 10 s is usually re-requested, doubling load |
| C-02 | Engagement acceptance latency | p99 ≤ 15 min | Engagement has no user waiting; 15 min absorbs provider blips |
| C-03 | Complaint ratio ceiling | ≤ 0.0005 per delivered message | Baseline measured by support sampling, ILLUSTRATIVE |
| C-04 | Cost ceiling | ≤ 108 percent of incumbent per delivered message | The renewal negotiation tolerates single-digit premium |
| C-05 | Duplicate rate | ≤ 0.002 of intents produce a user-visible second copy | Set from the complaint model in §6, not from taste |
| C-06 | Quiet-hours leakage | 0 sends inside recipient local 21:00–08:00 except transactional | Regulatory exposure plus complaint driver |
| C-07 | Suppression propagation | unsubscribe effective within 60 s | Legal and trust floor |
| C-08 | Ledger retention | 30 days hot, 400 days cold | Complaint investigation window and audit |

C-05 deserves a note because it is the constraint that forces the idempotency design. At 36 million
intents per day, 0.002 is 72,000 duplicated intents per day, which sounds enormous until it is
compared with the volume — it is 0.2 percent. A platform with no deduplication at all and a
four-percent provider failure rate that retries once would sit near 0.04 duplicates per intent, which
is twenty times the ceiling. The ceiling is not a rounding target; it is the level at which the
complaint model still closes.

The cost ceiling interacts with the duplicate ceiling in a way that is easy to miss. Duplicates are
not free: they consume provider quota and, in the SMS channel, they consume real money at roughly
0.0075 USD per message in this scenario. So a design that suppresses duplicates saves cost twice, once
in provider fees and once in support load. That double saving is what makes the ledger affordable even
though it adds a durable write per intent — a point quantified in §6.

**Takeaway:** eight assumptions each have a probe and a fallback; eight constraints are numeric, and
the duplicate ceiling is derived from the complaint model rather than chosen.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is one **intent**, not one message and not one HTTP call. An intent is the
platform's record that a business fact implies a possible notification to a specific recipient. One
intent may produce zero deliveries (suppressed, deduplicated, expired) or one delivery on one channel;
it must never produce two user-visible deliveries.

Required fields at decision time, with types and units:

| Field | Type | Units / domain | Available at decision | Notes |
|---|---|---|---|---|
| `intent_id` | uuid | — | yes | Assigned by decision service, monotonic per shard for ordering |
| `event_id` | string | — | yes | Producer-supplied business fact identity |
| `recipient_key` | string | — | yes | Canonical person key, not device token |
| `class` | enum | transactional, engagement, compliance | yes | Drives queue, retry, and latency ceiling |
| `template_id` | string | — | yes | Versioned; render inputs separate |
| `event_time` | timestamp | UTC, µs | yes | When the business fact occurred |
| `decision_time` | timestamp | UTC, µs | yes | When the platform evaluated the intent |
| `recipient_local_hour` | int | 0–23 | yes | Derived from stored timezone, not from event_time offset |
| `caps_state` | struct | counters per class per window | yes | Read from a fast counter store |
| `suppression_state` | enum | clear, unsubscribed, bounced, legal-hold | yes | Authoritative at read time |
| `device_tokens` | list | 0–8 entries | yes | Each with last-seen and platform |
| `channel_eligibility` | bitset | push, email, sms, in-app | yes | Includes cost and consent per channel |

Fields that must **not** be available at decision time, because including them would leak the future
into the decision and make the platform untestable:

- Delivery receipts, open events, and click events. These arrive after the decision and cannot inform
  it; using them to train the send policy creates a model that learns from outcomes it caused.
- The recipient's subsequent session behaviour. A platform that sends because the user later opened
  the app is measuring its own effect, not predicting attention.
- Complaint records. Complaints arrive hours to days later; using them at decision time requires a
  causal channel that does not exist in this design (a complaint does change future suppression
  state, but that is a lagged input owned by trust and safety, not a same-intent feature).
- Provider health observed after the fact. The circuit breaker is allowed to use live provider
  lateness, but the decision record must store the breaker state it saw, not the state it later became.

The grain is one row per intent in the ledger, with a child table of attempts. Attempts carry
`attempt_no`, `channel`, `provider_request_id`, `provider_status`, `sent_time`, and `receipt_time`.
The join between them is `intent_id`; the join to the business fact is `event_id`; the join to a person
is `recipient_key`. Three different keys for three different questions, and conflating them is the
single most common schema error in this domain — `event_id` alone cannot deduplicate across recipients
because one event fans out to many people, and `recipient_key` alone cannot deduplicate because one
recipient receives many events.

The cut-off is defined on `decision_time`: a decision may only read state whose committed version is
older than the decision's start. Implementations approximate this with a snapshot read or a
read-your-writes session, but the specification must state the intent, because the practical failure
is a counter read that misses an in-flight increment from a concurrent intent for the same recipient.

Population: all intents in the planning day, with no exclusions except synthetic test recipients.
There are no labels in the supervised-learning sense. The closest thing to a label is a downstream
adjudication — a complaint, an unsubscribe, a provider bounce — and those are used only for
evaluation, never for the decision. That separation is what lets §9 report an honest numerator.

Units are explicit everywhere because unit drift is a real defect in this area: latency in
milliseconds for per-call measurements and seconds for end-to-end ceilings; cost in USD per message
with six decimal places because SMS costs 0.0075 and push costs 0.0000002, a ratio of 37,500; rates in
per-second for capacity and per-day for budgets; timestamps in UTC microseconds for storage and local
civil time for quiet hours.

### 5b. Tools & Stack

The stack is described by responsibility rather than by product name, because the teaching content is
the responsibility boundary, not a vendor selection.

- **Intent intake:** a durable log partitioned by `recipient_key` so that all intents for one person
  are ordered. Partitioning by `event_id` would spread one recipient's intents across partitions and
  make per-recipient ordering impossible without a distributed lock. Partitioning by `recipient_key`
  risks hot partitions, which §12 treats as a first-class failure.
- **Decision service:** stateless workers that read caps state, suppression state, and eligibility,
  then emit a decision record with `decision ∈ {send, hold, digest, drop}` and a reason code. Stateless
  matters because the decision must be recomputable from the ledger and the input versions.
- **Dedup store:** a key-value store with a conditional write (`put-if-absent`) over the idempotency
  key, holding the terminal state and a TTL. Conditional write is the primitive that makes
  effectively-once possible; a read-then-write leaves a window where two workers both observe absence.
- **Delivery queue:** per-class queues with independent concurrency limits and retry budgets, so that
  a provider slowdown for engagement cannot consume transactional capacity.
- **Provider adapters:** one per channel, each translating platform semantics into a provider call and
  classifying the response into `accepted`, `rejected-permanent`, `rejected-transient`, `unknown`. The
  `unknown` class is the important one and the incumbent service did not have it.
- **Receipt ingester:** consumes provider webhooks, normalises receipts into attempt rows, and feeds the
  circuit breaker. Receipts are advisory for correctness (the platform does not retry based on a
  missing receipt) but essential for measurement.
- **Suppression service:** owns the authoritative do-not-contact state and publishes invalidations.
- **Ledger:** append-only decision and attempt history, queryable by `recipient_key` and time, hot for
  30 days, cold for 400. This is the artifact that turns a complaint into a five-minute lookup.
- **Reviewer console:** a read view over the ledger for support, showing for a given recipient and
  window: intents decided, reasons, attempts, receipts. It is deliberately read-only; support cannot
  force a resend, because a resend is itself an intent and must go through the same decision.

Configuration travels with each decision: template version, cap policy version, suppression snapshot
version, and breaker state. Without those four, a decision is not reproducible and the ledger becomes
an anecdote log.

### 5c. Method Summary

The method has four steps, and each produces an artifact that the next step consumes.

1. **Decompose demand into rates.** Convert the daily volume into mean, peak-hour, and peak-minute
   rates using an explicit diurnal profile. The profile is ILLUSTRATIVE but stated numerically so that
   a reader can substitute their own and re-run the arithmetic.
2. **Size admission and concurrency from the peak minute, not the mean.** Use Little's law to convert
   a latency target into an in-flight count, then convert that into worker concurrency. The peak minute
   is the binding constraint for queues; the mean is the binding constraint for cost.
3. **Build a cost model over the channel mix and the retry multiplier.** The retry multiplier is
   computed as a geometric series from the per-attempt failure probability, because retries are the
   difference between the intent count and the attempt count.
4. **Derive the duplicate rate from the key design and the retry semantics**, then check it against the
   complaint model and the C-05 ceiling. This is the step that decides between option (b) and (c).

The method deliberately does not optimise a single objective. It computes four quantities — capacity,
cost, duplicate rate, complaint ratio — and treats three of them as constraints. A single-objective
formulation would hide the fact that the cheapest design is also the one that sends most, because
suppression costs a store write and a duplicate costs nothing to the sender in the push channel.

Sensitivity is computed by re-running the arithmetic at the boundary of each assumption rather than by
a Monte Carlo, because with four or five inputs an extreme-case table is more informative and easier
to audit than a distribution. §6.7 gives the extreme cases.

### 5d. Provenance, Quality, Exclusions, Leakage

Provenance: all inputs are constructed for this case. The channel mix (88 percent push, 10 percent
email, 2 percent SMS) is ILLUSTRATIVE and chosen so that the cost model has three distinct orders of
magnitude. The diurnal profile (9 percent of daily volume in the peak hour, 0.35 percent in the peak
minute) is ILLUSTRATIVE; it implies a peak-minute-to-mean ratio of 5.04, which is typical of consumer
applications but is not measured here. Provider prices are ILLUSTRATIVE round numbers in the ranges
commonly quoted for self-serve tiers: 0.0000002 USD per push, 0.0009 USD per email, 0.0075 USD per SMS.
No contract was read.

Quality threats, and how the method handles each:

- **The daily volume is a single scalar.** A scalar hides weekday/weekend structure and campaign
  spikes. Handling: all capacity conclusions are expressed per unit rate, and §11 shows the peak-minute
  requirement at 1.5× and 0.5× the planning volume so the reader can read off their own value.
- **The complaint rate is assumed constant across classes.** In reality a duplicate engagement push and
  a duplicate passcode generate different complaint probabilities. Handling: the model in §6.6 carries
  separate duplicate-complaint probabilities for the two classes, and the numbers are stated so they can
  be replaced.
- **The retry multiplier assumes independent attempts.** Provider failures are correlated during an
  outage, so the geometric series understates attempts precisely when it matters. Handling: §6.4 also
  computes the correlated case where every first attempt fails inside a 10-minute window, which is the
  bound the retry budget must survive.
- **The suppression propagation delay is assumed to be zero.** Handling: A-05 requires measurement, and
  §12.2 treats "unsubscribe not yet propagated" as a distinct failure with a fail-closed correction.

Exclusions: provider selection, template rendering and localisation, in-app inbox semantics,
attribution of engagement to revenue, and the ML ranking of engagement candidates. The last is excluded
deliberately: ranking changes which intents exist, and this case's decision is about what happens to an
intent once it exists. A ranking model would add a second decision surface and confound the evaluation.

Leakage controls are stated because a notification platform is unusually prone to them. Any feature
computed from post-decision events is a leak; any policy tuned on the same day it is evaluated is a
leak; any capacity test run against a synthetic workload that the admission controller itself shaped is
a leak. The ledger schema in §5a is the enforcement mechanism: fields are either present at
`decision_time` or absent from the record entirely, and post-hoc joins happen in the evaluation layer
where they cannot feed back.

### 5e. Reproducibility & Method-Selection Ledger

Every number in this manuscript is recomputed from the five base inputs below. A reader with a
calculator can reproduce §6 and §11 exactly, and should get the same digits.

| Input | Symbol | Value | Unit | Status |
|---|---|---|---|---|
| Daily business events | E | 12,000,000 | events/day | ILLUSTRATIVE |
| Recipients per event | F | 3.0 | recipients/event | ILLUSTRATIVE |
| Daily intents | N = E·F | 36,000,000 | intents/day | DERIVED |
| Peak hour share of daily | s_h | 0.09 | fraction | ILLUSTRATIVE |
| Peak minute share of daily | s_m | 0.0035 | fraction | ILLUSTRATIVE |
| Push / email / SMS shares | p, q, r | 0.88 / 0.10 / 0.02 | fraction | ILLUSTRATIVE |
| Push / email / SMS unit cost | c_p, c_e, c_s | 0.0000002 / 0.0009 / 0.0075 | USD/message | ILLUSTRATIVE |
| First-attempt failure probability | f | 0.04 | fraction | ILLUSTRATIVE |
| Deduplicable share of intents | d | 0.18 | fraction | ILLUSTRATIVE |
| Quiet-hours share of day | q_h | 11/24 | fraction | DERIVED (21:00–08:00 inclusive of both endpoints is 11 h) |
| Engagement share of intents | g | 0.60 | fraction | ILLUSTRATIVE |
| Morning release window | W_r | 120 | minutes | ILLUSTRATIVE policy |
| Daily active recipients | U | 3,200,000 | persons/day | ILLUSTRATIVE |

Method selection, with the alternative that was rejected and why:

| Choice made | Alternative rejected | Reason |
|---|---|---|
| Peak-minute sizing | Mean-rate sizing | Mean understates admission by 5.04× and fails at the only time it matters |
| Geometric retry series | Fixed retry count of 3 | The series shows why 3 is enough: 0.04³ = 6.4e-5 residual |
| Conditional-write dedup | Read-then-write with TTL | Race window produces duplicates exactly under load |
| Per-class queues | One queue with priorities | Priority does not bound head-of-line delay from a bulk backlog |
| Local civil hour from stored timezone | Hour offset from event timestamp | Offsets break across DST and for travellers |
| Complaint ratio as objective | Delivery rate as objective | Delivery rate is maximised by retrying, i.e. by duplicating |
| Ledger as source of truth | Provider logs as source of truth | Provider retention is days, and provider logs lack the decision |
| 30-day hot ledger | 7-day hot ledger | Support escalation arrives at day 3–14 in this scenario |

**Takeaway:** the grain is one intent with three distinct keys; the method computes capacity, cost,
duplicates, and complaints separately and constrains three of the four.

## 6. Analysis Walkthrough

Ten derivations follow. Each states the formula, where it applies, the constraint it must respect, a
numeric evaluation with the intermediate values visible, and what the result means for the decision.
All inputs are the §5e table; every result is ILLUSTRATIVE even where the arithmetic is exact.

### 6.1 From a daily volume to three rates

```
FORMULA   lambda_mean = N / 86,400
          lambda_hour = N * s_h / 3,600
          lambda_min  = N * s_m / 60
          k_spike     = lambda_min / lambda_mean

WHERE     N        = intents per day (intents/day)
          s_h      = share of daily volume falling in the peak hour (dimensionless)
          s_m      = share of daily volume falling in the peak minute (dimensionless)
          lambda_* = arrival rate in intents/second at the stated scale
          k_spike  = peak-minute-to-mean ratio (dimensionless)

CONSTRAINT  s_m <= s_h/60 is a sanity check: the peak minute cannot hold more than
            1/60 of the peak hour. Here s_h/60 = 0.0015 and s_m = 0.0035, which
            VIOLATES it; the profile is therefore "one sharp minute", not a flat hour,
            and the hour figure must be recomputed as a rolling maximum over minutes
            rather than assumed constant.

NUMERICAL EXAMPLE
          N = 36,000,000 intents/day
          lambda_mean = 36,000,000 / 86,400 = 416.67 intents/s
          lambda_hour = 36,000,000 * 0.09 / 3,600 = 3,240,000 / 3,600 = 900 intents/s
          lambda_min  = 36,000,000 * 0.0035 / 60 = 126,000 / 60 = 2,100 intents/s
          k_spike     = 2,100 / 416.67 = 5.04

INTERPRETATION  Capacity sized on the mean is short by 5.04x at the only moment
          that matters. Sizing on the peak hour (900/s) is short by 2.33x at the
          peak minute. The binding number for admission control is 2,100 intents/s,
          and every queue length, concurrency limit, and rate limiter below is
          derived from it, not from 416.67.
```

The violated sanity check is deliberate and worth dwelling on: a naive planner multiplies the peak
hour by the number of hours and reconciles, which silently assumes uniformity inside the hour. If the
profile is one sharp minute (a scheduled digest firing at 08:00 for every recipient in a timezone), the
hour is 9 percent of the day only because 60 seconds carry a third of it. The correct reading is that
the platform must survive a square wave, and the peak-hour figure is a consequence rather than a
design input. This matters for the circuit breaker: a breaker tuned to 900/s will trip on the digest.

### 6.2 The retry multiplier

```
FORMULA   A = 1 + f + f^2 + f^3
          attempts_total = N * A
          extra_attempts = N * (A - 1)

WHERE     f = probability that a single provider attempt fails transiently (fraction)
          A = expected attempts per intent given at most 3 retries (attempts/intent)
          The series truncates at 3 retries by policy; it is the sum of the
          probability of needing attempt k, which for independent failures is f^(k-1).

CONSTRAINT  f < 1, and the residual failure probability after the last attempt is
          f^4 = 2.56e-6 per intent, which must be below the class delivery floor:
          for transactional, 1 - 0.995 = 0.005. 2.56e-6 << 0.005, so three retries
          satisfy the floor with four orders of magnitude of headroom; the retry
          budget is set by cost and duplicate risk, not by the floor.

NUMERICAL EXAMPLE
          f = 0.04
          f^2 = 0.0016, f^3 = 0.000064
          A = 1 + 0.04 + 0.0016 + 0.000064 = 1.041664 attempts/intent
          attempts_total = 36,000,000 * 1.041664 = 37,499,904 attempts
          extra_attempts = 36,000,000 * 0.041664 = 1,499,904 attempts
          extra cost at c_bar = 0.000240176 USD/message:
            1,499,904 * 0.000240176 = 360.24 USD/day

INTERPRETATION  Retries cost 4.17 percent more attempts and about 360 USD/day at
          this volume, which is affordable. The number that is not affordable is the
          duplicate exposure: every retry is a candidate second delivery, and at
          f = 0.04 with no dedup the duplicate rate would be 0.041664, twenty-one
          times the C-05 ceiling of 0.002. The retry budget is therefore safe only
          behind an idempotency key; without one, the same 0.04 failure rate is a
          complaint incident.
```

### 6.3 Deduplication escape as a function of window width

```
FORMULA   P_escape(W) = d * (1 - P_catch(W))
          escapes_per_day = N * P_escape(W)

WHERE     W = width of the dedup window (minutes)
          d = share of intents that are genuinely re-emitted by the producer
              (duplicate intents, not duplicate sends) — fraction
          P_catch(W) = probability a re-emission lands inside W
          P_escape = share of intents that slip past dedup and may send twice

CONSTRAINT  escapes_per_day must be <= 0.002 * N = 72,000 intents/day (C-05).

NUMERICAL EXAMPLE
          d = 0.18, and P_catch measured on replays (ILLUSTRATIVE):
            W =  5 min -> P_catch 0.940 -> P_escape 0.0108 -> 388,800/day (5.4x over)
            W = 15 min -> P_catch 0.990 -> P_escape 0.0018 ->  64,800/day (0.9x, pass)
            W = 60 min -> P_catch 0.997 -> P_escape 0.00054 -> 19,440/day (0.27x, pass)
          Window storage: a 15-minute window at 2,100 intents/s peak holds
            2,100 * 900 s = 1,890,000 keys live, versus
            2,100 * 300 s =   630,000 keys for a 5-minute window.

INTERPRETATION  A five-minute window fails C-05 by 5.4x, and no amount of retry
          tuning fixes it because the escapes are producer re-emissions, not
          transport failures. Fifteen minutes passes with 10 percent margin, and
          sixty minutes triples the margin for four times the live key count.
          The decision is 15 minutes only if the producer re-emission delay
          distribution is measured; if it is unknown, 60 minutes is the honest
          choice, because the cost of the wider window is bounded (3.8 million live
          keys at an estimated 116 bytes each is 0.44 GB) while the cost of
          under-sizing is a complaint incident that cannot be undone after send.
```

The 15-minute row's 0.9 multiplier is the reason this table exists rather than a rule of thumb. A
window that passes a ceiling with only 10 percent margin is not passing; it is passing for this
sample. `P_catch` is estimated from replays, and the estimate has sampling error. If the true catch
probability is 0.985 rather than 0.990, escapes become 0.0027 per intent, 97,200 per day, and the
ceiling is breached. So the window is set at 60 minutes, and the 15-minute row is retained as the
argument for why 5 minutes is indefensible.

### 6.4 Where the geometric series breaks: a correlated outage

```
FORMULA   attempts_during_outage = N_window * R_max
          cost_of_outage_window  = N_window * R_max * c_bar
          duplicate_bound        = N_window * (R_max - 1) if no dedup

WHERE     N_window = intents falling inside the outage window (intents)
          R_max = the retry cap per intent (attempts), here 3
          The independent-failure assumption of 6.2 fails because failures are
          correlated: during a provider incident f approaches 1 for all intents.

CONSTRAINT  The retry budget must be exhausted without violating C-05 and without
          consuming another class's capacity, so retries are per-class queued with
          a per-class token budget, never a shared pool.

NUMERICAL EXAMPLE  a 10-minute provider outage at peak
          N_window = 2,100 intents/s * 600 s = 1,260,000 intents
          attempts  = 1,260,000 * 3 = 3,780,000 attempts (cap reached, not 1.04x)
          extra attempts = 2,520,000, versus 52,500 predicted by the independent
            model (1,260,000 * 0.041664)
          ratio = 2,520,000 / 52,500 = 48x understatement
          duplicate bound without dedup = 1,260,000 * 2 = 2,520,000 user-visible
            second copies, i.e. 70x the entire daily C-05 allowance in ten minutes

INTERPRETATION  This is the boundary where 6.2's formula breaks, and it breaks in the
          worst direction. Correlated failure turns a 4 percent retry overhead into
          a 200 percent attempt multiplier and a 70x duplicate excess. The design
          consequence is not "retry less"; it is that retries must consult the
          dedup store, so a retry after an ambiguous 5xx is suppressed when the
          key already holds a non-terminal state. The ledger turns a correlated
          outage from a duplicate storm into a delivery delay, which is a
          recoverable failure.
```

### 6.5 Cost per delivered notification, with and without dedup

```
FORMULA   c_delivered = (N * A * c_bar) / (N * (1 - P_drop))
          c_with_dedup = (N * (1 - d) * A * c_bar) / (N * (1 - d) * (1 - P_drop))

WHERE     c_bar   = p*c_p + q*c_e + r*c_s (USD/message), the mix-weighted unit cost
          P_drop  = share of intents suppressed or expired before any attempt
          d       = deduplicated share, which removes both numerator and
                    denominator; algebraically it cancels, so dedup does not change
                    cost per delivered message — it changes total cost and the
                    duplicate rate, not unit cost.

CONSTRAINT  c_delivered must be <= 1.08 * c_incumbent (C-04).

NUMERICAL EXAMPLE
          c_bar = 0.88*0.0000002 + 0.10*0.0009 + 0.02*0.0075
                = 0.000000176 + 0.00009 + 0.00015 = 0.000240176 USD/message
          With A = 1.041664 and P_drop = 0:
            daily attempts cost = 37,499,904 * 0.000240176 = 9,006.58 USD/day
            per delivered = 9,006.58 / 36,000,000 = 0.00025018 USD
          SMS alone: 0.02 * 36,000,000 = 720,000 messages/day at 0.0075 = 5,400 USD
            = 60 percent of total spend from 2 percent of volume.

INTERPRETATION  The algebra in FORMULA is the useful part: dedup cancels out of the
          per-delivered-message cost, so "dedup saves money per message" is false,
          while "dedup saves total money" is true only because it removes intents.
          Meanwhile the SMS channel, at 2 percent of volume, carries 60 percent of
          spend, so any cost conversation that opens with push is misdirected. A
          1 percent reduction in SMS share is worth 54 USD/day; a 10 percent
          reduction in push volume is worth about 6.3 USD/day.
```

### 6.6 Complaint ratio and the implied duplicate ceiling

```
FORMULA   r_complaint = (1 - dup) * b + dup * b_dup
          dup_ceiling = (r_max - b) / (b_dup - b)

WHERE     b      = complaints per delivered notification at zero duplication (fraction)
          b_dup  = complaints per delivered notification when the message is a
                   duplicate (fraction), b_dup > b
          dup    = share of delivered notifications that are user-visible repeats
          r_max  = ceiling 0.0005 per delivered message (C-03)

CONSTRAINT  b_dup > b; if the inequality fails, duplication is not the complaint
          driver and the ceiling derived here is meaningless. This is the
          falsification test to run before adoption.

NUMERICAL EXAMPLE  b = 0.0004, b_dup = 0.00072
          dup = 0.000 -> r = 0.0004 -> 14,400 complaints/day
          dup = 0.002 -> r = 0.00040064 -> 14,423 complaints/day (+23)
          dup = 0.040 -> r = 0.0004128 -> 14,861 complaints/day (+461)
          dup_ceiling = (0.0005 - 0.0004) / (0.00072 - 0.0004) = 0.0001 / 0.00032
                      = 0.3125

INTERPRETATION  The linear model says duplication could reach 31 percent before
          breaching C-03, which contradicts the C-05 ceiling of 0.002 that the case
          enforces. The contradiction is informative: C-03 is a lagging, averaged
          metric that cannot see a duplicate storm because 0.00032 x 0.04 is small
          relative to a 0.0004 baseline. C-05 exists precisely because the complaint
          ratio is insensitive to duplication at these volumes, and a platform that
          managed to C-03 alone would tolerate one duplicate in three sends. The
          lesson generalises: an averaged quality metric cannot substitute for a
          structural correctness ceiling.
```

This is the counterexample the contract asks for, and it is the most important finding in the case.
The arithmetic is trivially correct and the conclusion it licenses is wrong, because the model is
linear in a quantity whose harm is superlinear in perception. Recipients do not complain in proportion
to duplicate count; they complain when they notice, and noticing is a threshold phenomenon
concentrated in a subpopulation. Replacing the population average with a per-recipient distribution
(§6.7) reproduces the correct ordering: the duplicates that matter are concentrated on the recipients
who already receive the most, and those are the ones nearest their tolerance threshold.

### 6.7 Why the mean is the wrong per-user number

```
FORMULA   sum_b s_b * m_b = N / U
          excess_per_dau = sum_b s_b * max(0, m_b - k_cap)
          suppressed_total = U * excess_per_dau

WHERE     s_b = share of daily active recipients in bucket b (fraction, sums to 1)
          m_b = mean intents per recipient in bucket b (intents/person/day)
          k_cap = per-recipient daily cap for engagement class (intents/person/day)
          U = daily active recipients (persons/day)

CONSTRAINT  sum_b s_b = 1 and the weighted mean must equal N/U = 11.25; the last
          bucket's mean is solved from that identity rather than assumed, so the
          distribution cannot silently change total volume.

NUMERICAL EXAMPLE  buckets (share, mean): (0.05, 1.2) (0.08, 3.5) (0.10, 5.5)
          (0.17, 8.0) (0.30, 11.5) (0.18, 16.0), k_cap = 6
          partial = 0.05*1.2 + 0.08*3.5 + 0.10*5.5 + 0.17*8.0 + 0.30*11.5
                    + 0.18*16.0
                  = 0.06 + 0.28 + 0.55 + 1.36 + 3.45 + 2.88 = 8.58
          last bucket mean m_7 = (11.25 - 8.58) / 0.12 = 2.67 / 0.12 = 22.25
          excess terms, one per bucket, using max(0, m_b - 6):
            b1 0.05 * max(0, 1.2 - 6)  = 0.00
            b2 0.08 * max(0, 3.5 - 6)  = 0.00
            b3 0.10 * max(0, 5.5 - 6)  = 0.00
            b4 0.17 * (8.0 - 6)        = 0.34
            b5 0.30 * (11.5 - 6)       = 1.65
            b6 0.18 * (16.0 - 6)       = 1.80
            b7 0.12 * (22.25 - 6)      = 1.95
          excess_per_dau = 0.34 + 1.65 + 1.80 + 1.95 = 5.74 intents/person/day
          suppressed_total = 3,200,000 * 5.74 = 18,368,000 intents/day
          share of all intents = 18,368,000 / 36,000,000 = 0.510 (51.0 percent)
          accepted volume = 36,000,000 - 18,368,000 = 17,632,000 intents/day
          accepted mean rate = 17,632,000 / 86,400 = 204.1 intents/s
          accepted peak minute = 204.1 * 5.04 = 1,029 intents/s

INTERPRETATION  A cap of six engagement intents per day suppresses 51 percent of
          total volume, because 60 percent of recipients sit above the cap and the
          top bucket averages 22.25. Two design consequences follow. First, the
          platform's effective capacity is roughly half its intake, so admission
          must be sized on accepted intents (about 17.6 million/day, 205/s mean,
          1,033/s peak minute), not on intake. Second, a cap change from 6 to 8
          moves suppressed volume by millions of intents per day, which is a
          business decision masquerading as a config edit.
```

A reader should check the middle step: my first pass mixed the partial sum with the excess sum, which
is exactly the kind of error that survives review when both numbers are stated in prose. The corrected
term-by-term line is what the total depends on. This is a good illustration of why the contract
demands visible intermediates rather than only a final figure.

### 6.8 Quiet hours: held volume and the morning release

```
FORMULA   H = g * q_h * N
          lambda_release = H / (W_r * 60)
          ratio_to_mean  = lambda_release / (N / 86,400)

WHERE     g = engagement share of intents (fraction)
          q_h = quiet-hours share of the local day (11/24)
          H = intents held overnight (intents)
          W_r = release window length (minutes)
          Transactional and compliance classes are excluded from holding by C-06.

CONSTRAINT  Holding is only admissible if the release is spread; a release that
          arrives as a step is an admission violation by construction.

NUMERICAL EXAMPLE
          H = 0.60 * (11/24) * 36,000,000 = 0.60 * 16,500,000 = 9,900,000 intents
          release over 120 min: 9,900,000 / 7,200 = 1,375 intents/s
          ratio to mean rate = 1,375 / 416.67 = 3.30
          release over 180 min: 9,900,000 / 10,800 = 917 intents/s (2.20x mean)
          held share of engagement class = 9,900,000 / (0.60*36,000,000) = 0.458

INTERPRETATION  Overnight holding accumulates 9.9 million intents, and releasing them
          over two hours produces a sustained 1,375/s demand that is 3.3 times the
          mean and 65 percent of the peak-minute rate — a second, self-inflicted peak
          that the original diurnal profile did not contain. The release window must
          therefore be a designed rate limit, not a cron job that drains a queue.
          Widening to 180 minutes cuts the crest to 917/s at the cost of delivering
          stale engagement at mid-morning, which is a product trade rather than an
          engineering one.
```

### 6.9 Concurrency from a latency target

```
FORMULA   L = lambda_peak * W_sys
          workers = L / U_pod
          utilisation = lambda_peak / (workers_rounded * capacity_per_worker)

WHERE     L = mean number of intents in the system (intents)
          W_sys = mean end-to-end time in the decision-plus-enqueue path (s)
          U_pod = concurrent intents one worker handles at the target latency
          workers_rounded = ceil(workers / 5) * 5, so the fleet survives one pod loss

CONSTRAINT  utilisation must stay below 0.70 on the hot path, because queueing
          delay is convex near saturation: at 0.90 utilisation the expected wait is
          9x the service time, at 0.70 it is 2.3x (M/M/1 ratios 9 and 2.33).

NUMERICAL EXAMPLE  lambda_peak = 2,100 intents/s, W_sys = 2.0 s
          L = 2,100 * 2.0 = 4,200 intents in flight
          at 8 concurrent intents per worker: 4,200 / 8 = 525 workers
          rounded to 5s: 525 workers; utilisation at 2,100/s with 525 workers each
          sustaining 8 intents per 2 s = 4 intents/s: capacity = 525*4 = 2,100/s
          -> utilisation 1.00, which violates the constraint.
          Resizing for 0.70: capacity needed = 2,100/0.70 = 3,000 intents/s
          -> workers = 3,000/4 = 750 workers.

INTERPRETATION  Two errors are being avoided here. The first is sizing on the mean,
          which would suggest 105 workers and fail by 7x. The second is sizing on
          the peak at full utilisation, which produces 525 workers and a queue that
          diverges on any 60-second surge. 750 workers at 4 intents/s is the number
          that satisfies C-01 at the peak with headroom, and it is 1.43x the naive
          peak figure — a premium that is the price of not queueing at the exact
          moment the digest fires.
```

### 6.10 Ledger storage

```
FORMULA   bytes_hot = N * b_row * T_hot * f_overhead
          bytes_cold = N * b_cold * T_cold

WHERE     b_row = bytes per ledger row with indexes (bytes/row)
          b_cold = bytes per archived row after compression (bytes/row)
          T_hot, T_cold = retention in days (days)
          f_overhead = index and write-amplification factor (dimensionless)

CONSTRAINT  hot storage must fit the operational budget stated in C-08: 30 days hot,
          400 days cold, queryable by recipient within 5 s for the hot window.

NUMERICAL EXAMPLE  b_row = 116 B, f_overhead = 1.6, b_cold = 40 B
          bytes_hot/day = 36,000,000 * 116 = 4.176e9 B = 4.18 GB/day
          30 days = 125.3 GB; with overhead 125.3 * 1.6 = 200.4 GB
          cold: 36,000,000 * 40 * 400 = 5.76e11 B = 576 GB
          total ~776 GB, or about 723 GiB

INTERPRETATION  The ledger is affordable: under a terabyte for a year of retention at
          36 million intents per day. The number that should shape the schema is the
          cold volume, which is 2.9x the hot volume despite 3.5x compression per row,
          purely because 400 days is 13x 30 days. If cold retention were cut to 90
          days the archive falls to 130 GB, so the retention decision is worth more
          than any schema optimisation. Retention is a compliance and support
          question, which is why the messaging lead does not own it alone.
```

**Takeaway:** the peak rate is 5.04x the mean; three retries cost 4.2 percent extra attempts but
create a 21x duplicate excess without dedup; a 5-minute dedup window fails its ceiling while 60 minutes
passes; the complaint ratio is too insensitive to detect duplication, which is why the structural
ceiling exists; and a cap of six suppresses 51 percent of volume by concentrating on a heavy tail.

## 7. The Finding

Five findings, ordered by how much they constrain the design rather than by how interesting they are.

**Finding 1 — The duplicate rate is a property of the key, not of the transport.** With
`key = (recipient_key, event_id)` and a conditional-write store, a retry cannot produce a second
user-visible copy: the second worker loses the conditional write and observes the stored state. The
same platform with `key = (recipient_key, event_id, channel)` allows one copy per channel, which is
correct if channel fallback is intended and a defect if it is not, because the recipient experiences
two media as two notifications. The measured quantity that settles this is the channel-fallback rate:
if push failure is 4 percent and fallback to email fires on all of it, then 4 percent of recipients
receive both media whenever the push was accepted-but-unacknowledged, which is the `unknown` class
from §5b. That is 1,440,000 intents per day at the planning volume, and it is a design decision made
by a schema field.

**Finding 2 — A dedup window is a correctness parameter disguised as a retention setting.** §6.3 shows
five minutes fails C-05 by 5.4x and sixty minutes passes with 3.7x margin. The window is not an
infrastructure knob; it is the width of the interval over which the platform promises that a producer
re-emission is absorbed. Choosing it smaller than the producer's retry horizon is choosing a duplicate
incident. The retention cost difference — 630,000 live keys versus 1,890,000 — is under half a gigabyte
and is not a reason to accept the risk.

**Finding 3 — The complaint ratio cannot detect duplication, so it cannot be the only control.** §6.6
derives that duplication could reach 31 percent before breaching a 0.0005 complaint ceiling, because
the duplicate-complaint probability is only 1.8x the baseline. A metric that tolerates one duplicate
in three sends is not a quality metric for this purpose. The platform needs a structural ceiling (C-05)
measured directly on the ledger, and the complaint ratio as a lagging check that catches relevance
failures rather than duplication.

**Finding 4 — Per-recipient caps, not global volume, decide the effective workload.** §6.7 shows a cap
of six suppresses 51 percent of intents by concentrating on the 60 percent of recipients above it. That
inverts the capacity plan: intake is 2,100/s at peak while accepted volume is 1,029/s at peak. A plan
built on intake over-provisions by 2x; a plan built on accepted volume fails the moment the cap policy
is relaxed, because the cap is a product-owned number that can change without an engineering release.

**Finding 5 — Retry semantics with an `unknown` class are the difference between a delay and a storm.**
§6.4 shows a 10-minute outage multiplies attempts by 48x relative to the independent-failure model and
would produce 2,520,000 second copies without dedup. With dedup and an `unknown` classification, the
same outage produces a delivery delay bounded by the outage length plus queue drain, and zero duplicates.
The incumbent system's failure was not a missing queue; it was treating a 5xx as `rejected-transient`
when it might mean `accepted`.

The findings converge on one structural statement, which is §1's framing made concrete: **the decision
record must exist before the delivery attempt and must be the authority for whether a retry may
proceed.** Every finding above is a consequence of separating decision from attempt, and every failure
in §12 is a consequence of merging them.

What the findings do **not** establish is that this platform is cheaper than the incumbent. §6.5's
algebra shows dedup cancels out of unit cost, and the ledger adds a durable write per intent plus about
776 GB of retention. The cost case rests on avoided duplicates and avoided support contacts, neither of
which is measured here. A reader should treat the cost argument as a hypothesis with a stated test —
shadow traffic for four weeks with both paths — not as a conclusion.

## 8. System/Solution Design

### 8.1 Decision path

The path from intent to delivery has six stages, and each stage has exactly one job.

```
intent intake (durable log, partitioned by recipient_key)
   -> validate & normalise (schema, recipient resolution, clock correction)
   -> decide (caps, suppression, quiet hours, eligibility, dedup claim)
        decision in {send, hold, digest, drop} + reason_code
   -> claim idempotency key (conditional put; loser exits here)
   -> enqueue per-class queue (token-bucket admission)
   -> provider adapter call (classify: accepted | rejected-permanent |
                             rejected-transient | unknown)
   -> record attempt + terminal state; receipt ingester updates later
```

The ordering of "decide" before "claim" is deliberate. Claiming first would reserve the key for
intents that are then suppressed, which wastes the window: a suppressed intent would hold the key and
block a legitimate later emission of the same event after the cap resets. Deciding first means the key
is claimed only by intents that will actually be attempted, so the window measures delivery
de-duplication rather than decision de-duplication. The counter-argument — two workers deciding
concurrently for the same key both pass the decision and one loses the claim, which is fine — is why
the order is safe: the claim is still the serialisation point, it just carries less traffic.

### 8.2 The four decisions and their reason codes

| Decision | Meaning | Reason codes | Reversible? |
|---|---|---|---|
| send | Attempt delivery now on the selected channel | `ok`, `fallback_chosen` | no — a send is irreversible |
| hold | Defer to the next permitted window | `quiet_hours`, `cap_window`, `breaker_open` | yes — held intents expire |
| digest | Merge into a scheduled summary | `digest_eligible`, `burst_detected` | yes — digest membership is mutable |
| drop | Never deliver this intent | `suppressed`, `duplicate`, `expired`, `no_route` | no, but silent to recipient |

The reason code taxonomy is small on purpose: four decisions and eleven codes. A larger taxonomy
becomes an unread support interface. The codes are chosen so that each maps to exactly one corrective
action: `breaker_open` means wait, `no_route` means fix device registration, `suppressed` means the
recipient or the law said no and no engineering action applies. Support can therefore answer "why did
this not arrive" without reading code, which is the operational test of a taxonomy.

### 8.3 Idempotency store mechanics

The store holds, per key: `state`, `channel`, `attempt_count`, `first_seen`, `updated_at`,
`provider_request_id`. States are `claimed`, `in_flight`, `accepted`, `rejected_permanent`,
`unknown`, `expired`. The conditional write is `put_if_absent(key, payload, ttl)`; a worker that loses
it reads the existing record and takes one of three actions: exit if `accepted`, exit if
`rejected_permanent`, and if `unknown` or `in_flight` beyond the expected attempt duration, proceed
with a single retry only for the transactional class.

The lease on `in_flight` prevents the classic stuck-key failure: a worker crashes after claiming and
before recording a terminal state, and every later intent for that key exits as a duplicate forever.
A lease of 60 seconds with attempt timeout of 10 seconds gives a 6x safety ratio; on lease expiry the
record moves to `unknown`, not to `claimed`, because the attempt may have reached the provider.

TTL is 60 minutes on terminal states, matching the dedup window chosen in §6.3, plus a 24-hour
retention of `accepted` states for the channel-fallback decision (a fallback must know whether the
push was accepted, and push receipts can lag by minutes). This two-tier TTL is where most implementations
get into trouble: a single TTL of five minutes is enough for retry suppression but not for fallback, so
fallback implementations quietly extend the TTL without re-deriving the key count, and the store grows
past its memory plan.

### 8.4 Class-separated admission

Each class has its own queue, its own concurrency limit, and its own retry token budget. The numbers
below follow from §6.9 and the class shares, and they are ILLUSTRATIVE allocations rather than tuned
values.

| Class | Share of accepted volume | Peak accepted rate (intents/s) | Workers | Retry tokens/s | Breaker trip at |
|---|---|---|---|---|---|
| transactional | 0.35 | 360 | 260 | 60 | p99 lateness > 3 s for 30 s |
| engagement | 0.60 | 617 | 450 | 40 | p99 lateness > 90 s for 5 min |
| compliance | 0.05 | 52 | 40 | 10 | never trip; alert instead |

The 360/617/52 split sums to 1,029 intents/s, which is §6.7's accepted peak minute. Transactional
carries more workers per unit rate than engagement (260 workers for 360/s is 0.72 workers per intent/s,
versus 0.73 for engagement — nearly identical, which is a coincidence of the arithmetic and not a
design signal) because transactional must hold latency at 10 seconds with no queue tolerance, while
engagement can absorb a two-minute queue delay without violating C-02.

The breaker policy difference is the substantive part. Tripping the transactional breaker converts a
provider slowdown into a delivery failure, which for a passcode means the user is locked out — worse
than a slow delivery. So transactional never trips its breaker; instead it fails over to a secondary
provider after the lateness threshold, and only escalates to the on-call if the secondary also exceeds.
Engagement trips freely because a suppressed engagement send is a missed opportunity, not a blocked
user. Compliance never trips because a compliance message delayed is a legal exposure, which is exactly
the case where a human must be woken rather than an automated policy applied.

### 8.5 Receipt ingestion and the measurement boundary

Receipts arrive as webhooks with `provider_request_id`, `event` (delivered, bounced, opened), and a
provider timestamp. The ingester writes to the attempts table and updates the breaker's lateness
estimate. Two rules keep this honest:

1. **A missing receipt never triggers a retry.** Absence of evidence is not evidence of non-delivery,
   and the provider's webhook pipeline is less reliable than its send path. Retrying on a missing
   receipt is the most common cause of duplicate storms in this domain, because webhooks lag and drop.
2. **Receipts never feed the decision for the same intent.** They feed policy — suppression state,
   breaker state, engagement ranking inputs — with an explicit lag. The ledger records which lagged
   snapshot was used, so an evaluation can reconstruct what the decision saw.

The measurement boundary is the intent ledger. Everything about the platform's own behaviour is
measurable there; everything about the recipient's experience is measurable only through receipts and
complaints, which are both lossy. §9 states the denominators accordingly.

### 8.6 What is deliberately not built

- No ranking model for engagement. It would change which intents exist and confound the duplicate
  measurement; a separate case owns that decision.
- No user-facing preferences beyond unsubscribe and quiet hours. Preference centres accumulate states
  that the decision path must reconcile, and reconciliation bugs are indistinguishable from suppression
  bugs in the logs.
- No cross-platform dedup between in-app inbox and push. The inbox is a rendering surface; treating it
  as a channel means one intent produces two visible artifacts and the duplicate rate becomes
  unmeasurable.
- No automatic resend from support. A resend is a new intent with a new `event_id` and goes through
  the same path; a support override that bypasses the decision is a permanent hole in the audit trail.

**Takeaway:** decide before claiming, four decisions with eleven reason codes, a leased idempotency
store with two-tier TTL, per-class queues with deliberately different breaker policies, and receipts
that measure rather than decide.

## 9. Evaluation & Validation

### 9.1 Metric definitions

Every metric below is stated with numerator, denominator, population, window, comparator, and guardrail,
because a metric without those six attributes is a number that will be argued about rather than used.

| Metric | Numerator | Denominator | Population | Window | Comparator | Guardrail |
|---|---|---|---|---|---|---|
| Duplicate rate | intents with ≥2 `accepted` attempts on the same key | intents with ≥1 `accepted` attempt | all classes | rolling 1 h, reported daily | C-05 = 0.002 | If >0.002: page; if >0.004: halt engagement class |
| Complaint ratio | complaints attributed to a send | delivered notifications (accepted − bounced) | all classes | 7-day trailing | C-03 = 0.0005 | If >0.0005 for 2 consecutive days: freeze cap increases |
| Transactional latency p99 | intents with `accepted` within 10 s | transactional intents decided | transactional only | 1 h | C-01 = 10 s | If >10 s: failover to secondary provider |
| Delivery floor | transactional intents accepted within 10 s | transactional intents not suppressed | transactional only | 1 h | ≥0.995 | If <0.995: incident, page immediately |
| Suppression correctness | intents dropped where suppression state was stale at decision | intents dropped for `suppressed` | all classes | daily audit sample | 0 tolerated | Any occurrence: audit and fix propagation |
| Quiet-hours leakage | sends with `recipient_local_hour` in 21–08 and class ≠ transactional | all sends | all classes | daily | 0 | Any occurrence: page |
| Ledger completeness | intents with a terminal state | intents with any state | all classes | 1 h | 1.0 | If <1.0: stuck leases, inspect oldest `in_flight` |
| Cost per delivered | USD of attempts | delivered notifications | all classes | daily | ≤1.08 × incumbent | If >1.08: review channel mix, not retries |

The duplicate-rate denominator deserves care. Using intents-with-at-least-one-acceptance rather than
all intents removes suppressed and dropped intents, which cannot duplicate. Using all intents would
understate the rate by the suppression share, and §6.7 says that share is 51 percent — so the metric
would read 0.001 when the true duplicate rate among delivered messages is 0.002.

### 9.2 Validation design

Four validations, with an explicit statement of what was and was not executed. Nothing in this case was
executed; the case is proposed, and the table is the plan a reviewer would audit.

| Validation | Design | Executed? | Would falsify |
|---|---|---|---|
| Shadow traffic comparison | 4 weeks, 100 percent of intents decided by both paths, only the incumbent sends | no — not executed | Finding 1 and the cost hypothesis: if duplicate rate and complaint ratio are equal at equal delivery, the ledger buys nothing |
| Dedup window replay | Replay 30 days of historical producer events through the dedup store at W ∈ {5, 15, 60} min, count escapes | no — not executed | §6.3: if measured P_catch is 0.999 at 5 min, the wide window is unnecessary |
| Outage injection | Block provider responses for 10 min at peak in staging; count attempts, duplicates, and drain time | no — not executed | §6.4: if drain exceeds the class latency ceiling, the retry budget is too generous |
| Cap-change simulation | Re-run §6.7 distribution at k_cap ∈ {4, 6, 8, 12}; measure accepted volume and complaint ratio | no — not executed | Finding 4: if accepted volume is insensitive to the cap, the distribution assumption is wrong |

The shadow-traffic design has one trap worth naming: the two paths must share the recipient's actual
experience, which means the shadow path must not send. So the shadow path validates decision agreement
and duplicate detection, not complaint outcomes. Complaint outcomes can only be measured after cutover,
which is why the falsification window is stated at four weeks of *live* traffic with a pre-registered
rollback.

### 9.3 Uncertainty

Three sources of uncertainty dominate, and each is quantified rather than described.

**Sampling in `P_catch`.** The 60-minute window decision rests on `P_catch = 0.990` at 15 minutes. If
`P_catch` is estimated from 10,000 replays with 9,900 catches, the standard error of the proportion is
`sqrt(0.99*0.01/10000) = 0.000995`, so a 95 percent interval is 0.988 to 0.992. Escapes at the lower
bound become `0.18 * 0.012 = 0.00216`, which exceeds C-05. The measurement is therefore not precise
enough to license a 15-minute window, which independently supports the 60-minute choice.

**The diurnal profile.** The 5.04 spike ratio is one number standing in for a distribution. At 4.0 the
peak minute is 1,667/s and the worker requirement falls from 750 to 596; at 6.0 it is 2,500/s and the
requirement rises to 893. The capacity plan is linear in the spike ratio, so an error in the profile is
a proportional error in the fleet.

**The complaint baseline.** At `b = 0.0004` the derived duplicate ceiling is 0.3125; at `b = 0.0003` it
is 0.5; at `b = 0.00045` it is 0.185. All three are far above 0.002, so the qualitative conclusion of
§6.6 — the complaint ratio is a weak duplicate detector — is robust to the baseline, which is the part
that matters. The exact ceiling is not robust and should not be quoted as one.

### 9.4 Guardrail interactions

Guardrails that trip independently can deadlock the platform, so their interaction is designed rather
than discovered. If the transactional breaker fails over to the secondary provider and the secondary
provider costs 3x, the cost guardrail trips, and the cost guardrail's corrective action is "review
channel mix" — which requires a human. During that window the transactional path continues on the
expensive provider, because blocking it fails C-01 and the delivery floor. Priority order is therefore:
correctness ceilings (C-05, C-06, C-07) preempt latency floors, latency floors preempt cost ceilings.
Stating the order prevents the common incident shape where a cost alert causes an engineer to disable
failover at the worst possible moment.

**Takeaway:** the primary control is the ledger-measured duplicate rate; the complaint ratio is a lagging
check that lags too much to catch duplication; nothing in this design has been executed, and each
validation names the finding it would falsify.

## 10. Action Taken

The lead commits to a sequence of five actions across fourteen weeks, with a rollback gate between each.
The staging matters because every step is reversible up to step four and irreversible after, so the
irreversible step is placed where the evidence is strongest.

**Week 1–2: instrument before changing anything.** Add `event_id` emission at the three producers
(cart, saved search, order), add a decision-record write to the incumbent path that records the decision
it would have made without acting on it, and stand up the ledger schema. Cost: one engineer, two weeks.
Success criterion: 99.9 percent of intents carry a producer `event_id`; the incumbent's implicit
decisions are captured with the four configuration versions from §5b.

**Week 3–4: build the dedup store and replay history.** Stand up the conditional-write store, replay 30
days of historical events at three window widths, and produce the `P_catch` table that §9.3 says is
underpowered. Success criterion: a reproduced escape table with confidence intervals, and a decision on
`W` that is defensible at the lower bound rather than the point estimate.

**Week 5–8: shadow the decision path.** Both paths decide every intent; only the incumbent sends. Compare
decision agreement per class, and specifically count intents where the ledger path says `send` and the
incumbent also sent, but on a different channel — the channel-fallback disagreement that Finding 1 says
is a schema consequence. Success criterion: decision agreement above 99.5 percent per class after
excluding the deliberate differences (cap policy is new, so engagement disagreement is expected).

**Week 9–10: cut over engagement first.** Engagement is the class where a duplicate is most visible but a
delay is harmless, which makes it the right class to expose to a new decision path: the failure mode is
annoyance, not lockout. Rollback is a feature flag that restores the incumbent decision for that class
within 60 seconds, with the ledger still recording decisions so that the rollback itself is auditable.
Success criterion: duplicate rate under 0.002 for seven consecutive days, complaint ratio no worse than
the trailing baseline, and no growth in support contacts.

**Week 11–12: cut over compliance, then transactional.** Compliance first because volume is 5 percent
and the cost of a mistake is a legal exposure that a human will catch in review; transactional last
because it has the tightest latency ceiling and the highest blast radius. Transactional cutover is by
recipient cohort in five tranches, 20 percent each, with a 24-hour bake per tranche. Success criterion
per tranche: delivery floor at or above 0.995 and p99 latency at or below 10 seconds measured from
intent creation, not from queue entry.

**Week 13–14: retire the incumbent path and adjust policy.** Remove the synchronous fan-out, keep the
ledger, and only then revisit the cap. The ordering matters: changing the cap during cutover would make
the duplicate measurement unattributable, because a cap change alters the population. This is the
single most common sequencing error in platform migrations — a policy change shipped alongside an
architecture change, leaving no way to attribute a movement in the metrics.

The decision among §3's four options is therefore option (b) with one modification: the key is
`(recipient_key, event_id)` for the duplicate ceiling, and channel fallback is recorded as an *attribute
of the key's state* rather than as a new key. That gives fallback without a second dedup surface, and it
means the recipient sees one notification on one channel even when the first channel fails, which is the
behaviour the complaint model rewards. Option (d) — ledger with the window deferred — is rejected
because §9.3 shows the 15-minute window is not supported at the lower confidence bound, so deferring the
window decision means deferring the correctness argument.

**Takeaway:** instrument, measure the window, shadow, then cut over engagement → compliance →
transactional, and change no policy during cutover.

## 11. Outcome & Results

No operational outcome exists, because nothing was deployed. What exists is a set of computed
consequences of the committed design, and a set of results that would count as success or failure. The
distinction is stated explicitly so that no reader mistakes arithmetic for evidence.

### 11.1 Computed consequences of the design as committed

| Quantity | Value | Derivation | Status |
|---|---|---|---|
| Intake peak minute | 2,100 intents/s | §6.1 | ILLUSTRATIVE |
| Accepted peak minute | 1,029 intents/s | §6.7 (51 percent suppressed) | ILLUSTRATIVE |
| Workers required at 0.70 utilisation | 750 | §6.9 | ILLUSTRATIVE |
| Attempts per day | 37,499,904 | §6.2 | ILLUSTRATIVE |
| Attempt cost per day | 9,006.58 USD | §6.5 | ILLUSTRATIVE |
| Duplicates per day at W = 60 min | 19,440 intents | §6.3 | ILLUSTRATIVE |
| Duplicate rate at W = 60 min | 0.00054 | §6.3 | ILLUSTRATIVE |
| Held engagement overnight | 9,900,000 intents | §6.8 | ILLUSTRATIVE |
| Release crest at 120 min | 1,375 intents/s | §6.8 | ILLUSTRATIVE |
| Hot ledger at 30 days | 200.4 GB | §6.10 | ILLUSTRATIVE |
| Cold archive at 400 days | 576 GB | §6.10 | ILLUSTRATIVE |

The duplicate rate of 0.00054 at a 60-minute window sits at 27 percent of the C-05 ceiling, which is the
margin the lead was buying with the wider window. The 5-minute alternative would sit at 270 percent of
the ceiling. That ratio — 10x between the two windows — is the whole argument in one line.

### 11.2 What success looks like after cutover

Four numbers, measured on live traffic, with the time at which each becomes readable:

- **Duplicate rate ≤ 0.002 within 7 days.** Readable immediately; requires no lag. This is the
  correctness claim and it should be true on day one, not on day thirty.
- **Delivery floor ≥ 0.995 for transactional within 24 hours.** Readable from the ledger plus provider
  acceptance. A floor failure at 24 hours is a design defect, not a tuning issue.
- **Complaint ratio ≤ 0.0005 on a 7-day trailing window, from day 14.** Complaint attribution lags by
  days, so the earliest honest reading is two weeks. Any earlier claim is a claim about a partial window
  and should be labelled as such.
- **Cost per delivered ≤ 1.08 × incumbent at 30 days.** Readable daily but meaningful only after the
  channel mix stabilises, because the mix is what drives cost and the mix moves during migration.

### 11.3 What the numbers do not show

The cost argument is weaker than it looks, and honesty about that is part of the deliverable. §6.5 shows
dedup cancels from unit cost, so the saving must come from total volume reduction and from avoided
support contacts. The volume reduction is the 51 percent cap suppression — which is a product decision
that would happen with or without the ledger — and the support saving is unquantified because the case
has no contact-rate data. The ledger's own cost is about 776 GB of storage plus one durable write per
intent; at a plausible storage price the ledger is a rounding error, and at a plausible durable-write
price it is not free but is small. The honest summary: **the ledger is justified by correctness, and it
is affordable, but it is not proven to be cheaper.**

The latency claim is similarly conditional. Workers were sized for a 2.0-second system time at peak,
which is a target rather than a measurement. If real decision latency is 40 ms and enqueue adds 15 ms,
the 2.0-second assumption is generous by more than an order of magnitude and the fleet is
over-provisioned by the same factor. The correct response is not to shrink the fleet before cutover; it
is to measure `W_sys` in the shadow phase and resize at week 8, when the number is observable.

**Takeaway:** duplicate rate at the committed window is 0.00054, 27 percent of ceiling; success is four
readings at four horizons; the cost case is a hypothesis and the latency sizing is conservative by
construction.

## 12. Failure Modes & What Went Wrong

Eight distinct mechanisms, each with the competing causes it is confused with and a check that
discriminates. The checks are chosen so that they can be run from the ledger alone, without provider
access, because during an incident that is what is available.

### 12.1 Merged decision and attempt: the retry storm

**Mechanism.** When the decision and the provider call happen in one code path with a retry wrapper, a
provider that returns 5xx after accepting produces a duplicate on retry. The wrapper cannot distinguish
"not accepted" from "accepted, acknowledgement lost", so its retry logic is a guess with a duplicate
cost.

**Competing causes.** (a) A legitimate re-emission from the producer inflating intent count; (b) a
double subscription to the same event bus topic, so each intent is processed twice; (c) the retry storm
itself. All three look identical in a "sends per minute" chart.

**Discriminating check.** Group attempts by `intent_id` and look at `provider_request_id`. A retry storm
shows multiple distinct provider request IDs for one intent, issued within the retry interval, with the
first returning 5xx. A double subscription shows two `intent_id`s with identical `event_id` and
`recipient_key` created microseconds apart by different worker IDs. A producer re-emission shows one
`intent_id` pattern but separated by the producer's own retry interval, typically seconds to minutes,
with the same `event_id`. The three signatures differ in which identifier repeats.

**Correction and prevention.** Add the `unknown` response class, gate retries on the dedup claim, and
store `provider_request_id` on every attempt. Verification: a synthetic 5xx-after-accept test in staging
must produce exactly one accepted attempt and one suppressed retry.

### 12.2 Stale suppression state: the message that should never have been sent

**Mechanism.** A recipient unsubscribes; the decision service reads a cached suppression snapshot that is
older than the unsubscribe; the send proceeds. The recipient experiences this as a broken promise, which
is worse than never having offered the option, and in some jurisdictions it is a compliance breach.

**Competing causes.** (a) A downstream channel adapter ignoring the suppression flag; (b) an unsubscribe
that was never committed because the UI swallowed the write; (c) genuine cache staleness. In the logs,
all three appear as "suppression state clear at decision time".

**Discriminating check.** Compare the suppression snapshot version recorded in the decision record with
the commit timestamp of the unsubscribe. If the unsubscribe committed before the snapshot version was
created, the decision read correct data and the defect is downstream (cause a). If the unsubscribe
committed after the snapshot but before the send, it is staleness (cause c) — and the gap between
`decision_time` and `sent_time` is the exposure window. If no unsubscribe record exists at all, it is
cause b, and the check is the UI's write path, not the platform.

**Correction and prevention.** Fail closed on a cold cache, publish invalidations, and cap the
`decision_time`-to-`sent_time` gap so the staleness window is bounded by queue delay rather than by cache
TTL. Verification: inject an unsubscribe and assert the next decision observes it within 60 seconds, on
every channel, measured from the commit.

### 12.3 Hot partition on a popular recipient

**Mechanism.** Partitioning the intent log by `recipient_key` orders one recipient's intents, which is
required for cap correctness, but a recipient with a large audience — a shared account, a test device
used by many sessions, a bot — concentrates traffic on one partition. The partition's consumer falls
behind, and every intent for that recipient queues behind the backlog.

**Competing causes.** (a) A global consumer lag caused by a slow downstream; (b) uneven partition
assignment from a bad hash; (c) a genuinely hot key. The distinguishing feature is that (c) is local:
one partition lags while its neighbours are healthy.

**Discriminating check.** Plot per-partition lag against per-partition key cardinality. A hot key has low
cardinality and high volume on one partition. A bad hash has low volume but skewed assignment across
many keys. A slow downstream has uniform lag across all partitions. The check is a scatter plot, not a
single number, and it resolves the three causes in one view.

**Correction and prevention.** Salt the partition key with a bounded suffix, for example two shards, and
keep per-recipient ordering by processing both shards through a per-recipient serialiser inside the
consumer. That preserves cap ordering while halving the worst-case partition load. Verification: a
synthetic key emitting 10,000 intents/s must not push any partition's lag above 5 seconds.

### 12.4 Clock and timezone defects in quiet hours

**Mechanism.** Quiet hours are computed from a stored timezone. Three defects are possible: the offset is
applied to the event timestamp instead of the decision time, so a delayed event is evaluated in the wrong
hour; the stored timezone is stale after a user moves; or the local time is computed with a fixed offset,
which is wrong by an hour across a DST transition.

**Competing causes.** (a) A cap counter reset at the wrong boundary, which also produces sends at odd
hours; (b) the quiet-hours defect; (c) a recipient with multiple timezones across devices, where the
"local" hour is ambiguous. All appear as sends in the 21:00–08:00 window.

**Discriminating check.** Take the offending sends and recompute the local hour three ways: from the
stored timezone at `decision_time`, from the stored timezone at `event_time`, and with a fixed offset.
If the first is inside the window and the platform sent, the defect is in code (cause b). If only the
second is inside the window, the defect is the timestamp choice. If the recomputation disagrees across
the recipient's devices, it is cause c, and the fix is to define the canonical timezone as the
recipient's most recently active device rather than any device.

**Correction and prevention.** Compute from `decision_time` plus stored timezone, use a timezone
database rather than a fixed offset, and record both the computed hour and the timezone version in the
decision record so that a mismatch is visible after the fact. Verification: a DST transition test at
both boundaries, plus a test with an event delayed by 14 hours.

### 12.5 Cap counter read-modify-write contention

**Mechanism.** The per-recipient cap counter is read, incremented, and written back. Two concurrent
intents for the same recipient both read 5, both write 6, and the cap permits seven sends. Under
contention the counter also serialises, producing latency spikes for exactly the recipients with the most
traffic.

**Competing causes.** (a) The cap value itself being wrong; (b) a duplicated intent stream inflating the
count (so the cap fires early rather than late); (c) the lost-update race. Note (b) pushes the observed
error in the opposite direction from (c), so a mixed incident can look like a correct cap.

**Discriminating check.** Compare the counter's value with a count of accepted sends for the same
recipient and window. Counter lower than actual means lost updates (cause c). Counter higher than actual
means duplicated intents (cause b). Counter equal to actual but the cap set to the wrong number means
cause a, which is a policy bug and needs no code change.

**Correction and prevention.** Use an atomic increment or a conditional write with a version, never
read-modify-write; keep the counter in the same store as the dedup claim so the two update in one
logical operation; and record the counter value at decision time in the ledger. Verification: a
concurrency test with 100 workers incrementing one key must produce exactly 100 increments.

### 12.6 Digest coalescing that defeats dedup

**Mechanism.** Engagement intents are merged into a digest. If the digest's identity is derived from the
digest's scheduled send time rather than from its member intents, two digests covering overlapping
membership can send the same item twice. The dedup key is per intent, and the digest path bypasses it.

**Competing causes.** (a) The duplicate is a retry (see 12.1); (b) the digest membership is genuinely
overlapping because the cap window and the digest boundary disagree; (c) the digest path never claims
keys, so the same intent is legitimately attempted twice by two paths. Cause c is the most likely and
the least visible, because each path is individually correct.

**Discriminating check.** For a duplicate pair, check whether both attempts share an `intent_id`. Same
`intent_id` means a retry (a). Different `intent_id`s with the same `event_id` and overlapping digest
membership means path duplication (c). Same `intent_id` but different digest IDs means membership
overlap (b). The identifier that differs tells you which of the three it is.

**Correction and prevention.** Make the digest path claim the same per-intent key before adding a member,
so a digest cannot absorb an intent that another path already claimed; define digest membership as a
deterministic function of the intent's key and the window, not of arrival order. Verification: replay a
day of intents through both paths and assert zero intents appear in two digests.

### 12.7 Ledger write failure creating an unaccountable send

**Mechanism.** The ledger write fails after the provider accepted. The message is delivered but no
record exists, so a later intent for the same event cannot be deduplicated and a complaint cannot be
investigated. This is the inverse of the usual concern: the dangerous state is a send without a record,
not a record without a send.

**Competing causes.** (a) A lost write due to backpressure on the ledger store; (b) an intent that was
never recorded because intake dropped it; (c) a send from a legacy path that bypasses the ledger. Cause c
is common during migration and inflates the apparent failure count.

**Discriminating check.** Join provider receipts to the ledger by `provider_request_id`. A receipt with no
ledger row is a lost write (a) if the provider request ID follows the platform's ID format, or a legacy
send (c) if it does not. Intents with no ledger row and no receipt are intake drops (b), and the check is
the intake log rather than the provider.

**Correction and prevention.** Write the ledger record before the provider call with state `in_flight`,
so a send can never exist without a prior record; never buffer ledger writes with lossy semantics; and
make the provider request ID carry the `intent_id` so receipts are joinable without a lookup. Verification:
kill the ledger store mid-flight and assert that every subsequent send has a prior `in_flight` row.

### 12.8 Channel fallback amplifying an outage

**Mechanism.** Push fails; the platform falls back to SMS. During a push provider outage of ten minutes
at peak, 1,260,000 intents fail push (§6.4's window) and the fallback fires all of them to SMS. At 0.0075
USD per message that is 9,450 USD in ten minutes, and the recipients receive an SMS about a saved-search
result, which is a channel mismatch as well as a cost event.

**Competing causes.** (a) A genuine push-token invalidation wave, where fallback is correct; (b) the
provider outage, where fallback is wrong because the message is not urgent; (c) a cap misconfiguration
allowing fallback for engagement class at all. All three produce an SMS spike.

**Discriminating check.** Segment the SMS attempts by class and by push failure reason. Fallback from
`no_route` on a stale token is cause a and is correct behaviour. Fallback from a transient 5xx during a
provider-wide lateness spike is cause b. Fallback from an engagement intent, regardless of reason, is
cause c — and the check is simply the class field, which means the defect is policy, not infrastructure.

**Correction and prevention.** Restrict fallback to transactional and compliance classes, cap fallback
volume per minute independently of the main admission rate, and require a fallback to be a decision
(`fallback_chosen` reason code) rather than an adapter behaviour. Verification: simulate a push outage
and assert SMS volume stays under the fallback cap and that zero engagement intents reach SMS.

**Takeaway:** eight mechanisms with three competing causes each; the discriminating checks turn on which
identifier repeats (intent, event, provider request, digest) or which quantity disagrees (counter vs
actual, receipt vs ledger).

## 13. Transferable Lesson

The transferable structure is this: **whenever an effect is visible to a third party and cannot be
undone, the record authorising the effect must be written before the effect, and the retry of the effect
must be a function of that record rather than of the attempted call.** Notifications are one instance.
Payments, webhook deliveries, file provisioning, ticket creation in an external tracker, and
idempotent HTTP APIs are the same shape, and the same four artefacts transfer without modification:

1. An identity for the intended effect, derived from the business fact rather than from the attempt.
2. A durable claim on that identity, taken with a conditional write, holding a terminal state.
3. A response classification that includes "unknown" as a first-class outcome, distinct from both
   success and failure.
4. A ledger that records the decision inputs and their versions, queryable by the party who will
   complain.

The case also transfers a negative lesson about metrics. The complaint ratio looked like the right
objective and turned out to be nearly blind to the defect the platform exists to prevent, because the
defect's harm is concentrated in a subpopulation while the metric averages over everyone. The general
rule: **for a threshold-harm defect, measure the structure (how many keys were claimed twice) rather than
the sentiment (how many people complained), and use sentiment only as a lagging cross-check.** A team
that adopts only the sentiment metric will find that their platform passes every review while a small
share of recipients quietly receives every message three times.

A third transferable point concerns the relationship between product policy and capacity. The cap of six
was a product-owned number, and §6.7 showed it suppresses half of all volume. That means the product
owner was, without a capacity model, choosing the platform's effective scale. Any platform where a
policy number sits upstream of an admission decision needs that number's capacity consequence computed
in the same document as the policy, or the engineering plan will be built for a workload that the policy
removes — and will then fail when the policy is relaxed for a campaign.

Finally, the case transfers a sequencing rule. Migration order should follow "most visible failure is
most tolerable first", not "least volume first". Engagement went first despite being 60 percent of
volume, because its worst failure is annoyance; transactional went last despite being the smallest class
in risk terms per message, because its worst failure is a locked-out user. Volume is a bad proxy for
blast radius, and engineering plans that order by volume will always cut over the riskiest path early.

**Takeaway:** write the authorisation record before the irreversible effect, classify "unknown" explicitly,
measure structure not sentiment for threshold harms, compute capacity for policy numbers, and sequence
migrations by blast radius.

## 14. Connections & Cross-References

Within this unit, three sibling artefacts carry the theory this case applies. The design-interview flow
sheet supplies the question-order discipline — clarify the workload, state the invariant, then size — that
§5 and §6 follow. The common-issues catalogue supplies the defect taxonomy that §12's eight mechanisms
extend with domain-specific instances (stale suppression, digest coalescing, fallback amplification). The
distributed-systems material supplies the conditional-write and lease primitives that §8.3 depends on, and
a reader who has not met those should read that section before this case's §8.

Outside this unit, the case connects to four areas. **Queueing theory** supplies Little's law and the
M/M/1 wait curve used in §6.9; the 0.70 utilisation constraint is a direct application and the reason the
fleet is 1.43x the naive peak figure. **Statistical estimation** supplies the proportion confidence
interval used in §9.3 to reject the 15-minute dedup window; without that interval the window choice looks
like taste. **Product analytics** supplies the per-recipient distribution in §6.7; the identity that the
weighted mean must equal total volume over active users is what stops the distribution from silently
inventing or destroying traffic. **Compliance** supplies C-06 and C-07; those constraints are stated as
numbers because "respect quiet hours" is not testable while "zero sends in 21:00–08:00 local for
non-transactional" is.

Three contrasts are worth holding side by side. The **feature-serving API** case in this unit shares the
correctness-contract pattern and the "unknown" classification problem, but its unit of correctness is a
timestamp (what the model knew at training time) rather than an identity (which send already happened).
The **payment authorisation** pattern is the closest structural analogue — an idempotency key, a durable
claim, an irreversible effect — and differs in that a payment has a monetary reconciliation path that a
notification lacks, so notifications must rely on the ledger where payments can rely on the settlement
feed. The **rate limiter** pattern shares the counter-contention failure in §12.5 but solves it with
approximate counting, which is acceptable there because the harm of a slightly wrong limit is small;
here the counter is a promise to the recipient, so approximation is not acceptable.

A reader who wants to extend this case should add one thing the case deliberately lacks: a measured
contact rate. Every complaint argument in §6.6 and §11 rests on `b` and `b_dup`, which are ILLUSTRATIVE.
Substituting measured values would convert the case's central finding — that complaint ratios are blind
to duplication — from a constructed argument into evidence, and would either confirm or refute the
0.3125 duplicate ceiling that §6.6 derives.

### Interview pack

Fourteen questions. Each has its own answer, its own reasoning, a follow-up that tests depth, a red flag
that identifies a candidate who has memorised without understanding, and a scoring guide. Roughly half
require a computation or a diagnosis rather than recall.

**Q1. What is the unit of idempotency in a notification platform, and why not the message ID?**

- **Answer:** The unit is the (recipient, business fact) pair — `(recipient_key, event_id)`. A message ID
  is assigned at send time and therefore cannot suppress a send that has not happened yet; by the time a
  message ID exists, the duplicate is already on the wire. The business-fact key exists before delivery
  and is stable across retries and across producer re-emissions, which are the two sources of duplication.
- **Reasoning:** Idempotency requires an identity that is stable *before* the effect. Anything minted at
  attempt time is by construction attempt-scoped, so it can deduplicate at-least-once delivery but not
  at-least-once *decisions*.
- **Follow-up:** What breaks if the producer mints a new `event_id` on each replay? *(Answer: the key
  degrades to attempt-scoped; you need a content hash fallback over recipient, template, and a time
  bucket, and you must record that you substituted.)*
- **Red flag:** Answering "the message ID, with a dedup table" — this reproduces the incumbent's bug.
- **Scoring:** 3 points for `(recipient, event)` with the pre-effect argument; 2 for keying on event
  alone (fans out to many recipients, so it under-deduplicates); 1 for a content hash without
  acknowledging the collision risk; 0 for message ID.

**Q2. Compute the mean and peak-minute rates for 36 million intents/day with a peak minute holding 0.35
percent of daily volume. What does the ratio tell you?**

- **Answer:** Mean is 36,000,000 / 86,400 = 416.67 intents/s. Peak minute is 36,000,000 × 0.0035 / 60 =
  2,100 intents/s. The ratio is 5.04. It tells you that capacity sized on the mean is short by 5.04×, and
  that any queue, concurrency limit, or provider rate limit must be sized on 2,100/s, with utilisation
  kept below 0.70 so the fleet must sustain about 3,000/s.
- **Reasoning:** Two unit conversions in sequence: per day to per second by dividing by 86,400, and a
  share-of-day to a rate by dividing by 60. The ratio is unit-free and is the number that transfers to
  other volumes.
- **Follow-up:** If the peak minute is a scheduled digest that fires once per timezone, what does the
  profile look like across the day? *(Answer: a series of square waves, so the peak minute repeats hourly
  and the "peak hour" is not smooth — the breaker must not trip on a legitimate digest.)*
- **Red flag:** Reporting 900 intents/s (the peak hour) as the design rate.
- **Scoring:** 3 for both figures and the 5.04 ratio with the sizing consequence; 2 for correct
  arithmetic without consequence; 1 for one figure; 0 for dividing by 3,600 twice.

**Q3. With a 4 percent per-attempt failure rate and three retries, how many extra attempts per day do you
generate at 36 million intents? And what is the duplicate exposure?**

- **Answer:** The multiplier is 1 + 0.04 + 0.0016 + 0.000064 = 1.041664, so extra attempts are
  36,000,000 × 0.041664 = 1,499,904 per day, about 4.17 percent. Duplicate exposure without dedup is the
  same 0.041664 per intent, which is 21× the 0.002 ceiling. With a conditional-write key the exposure is
  zero for retries, because the retrying worker loses the claim.
- **Reasoning:** Retries are a geometric series whose terms are the probability of needing attempt *k*.
  The critical observation is that the *cost* of retries is 4 percent while the *risk* is 4 percent per
  intent, and the risk is what the ceiling measures.
- **Follow-up:** During a provider outage, what does the same formula predict, and why is it wrong?
  *(Answer: it predicts 4 percent overhead because it assumes independence; correlated failure pushes f
  toward 1, so the cap of 3 is hit on every intent and attempts triple.)*
- **Red flag:** "Retries are cheap, just set max_retries high."
- **Scoring:** 3 for the series, the extra-attempt figure, and the duplicate exposure; 2 for the series
  without the risk link; 1 for "4 percent of 36 million"; 0 for multiplying 0.04 by 3.

**Q4. You hold a 60-minute dedup window and see a 0.00054 duplicate rate. A colleague proposes five
minutes to save memory. Make the argument numerically.**

- **Answer:** Escapes are `d × (1 − P_catch(W))`. At 60 minutes, `P_catch = 0.997` gives 0.18 × 0.003 =
  0.00054, or 19,440 intents/day. At five minutes, `P_catch = 0.940` gives 0.18 × 0.06 = 0.0108, or
  388,800/day — 5.4× the ceiling. Memory saved is 1,890,000 live keys minus 630,000, so 1,260,000 keys at
  about 116 bytes, roughly 146 MB. Trading 369,360 extra daily duplicates for 146 MB is not a trade.
- **Reasoning:** The window's cost is linear in live key count and bounded; its benefit is nonlinear
  because it sits in the tail of the producer re-emission distribution. Comparing a bounded cost with an
  unbounded risk is the whole argument.
- **Follow-up:** What measurement would justify five minutes? *(Answer: a replay showing `P_catch ≥ 0.989`
  at the lower confidence bound for a 5-minute window, which requires about 10,000 samples and a much
  tighter producer retry horizon.)*
- **Red flag:** "Five minutes is plenty; duplicates are rare."
- **Scoring:** 3 for both escape figures and the memory comparison; 2 for the direction without numbers;
  1 for "wider is safer"; 0 for accepting the proposal.

**Q5. Your complaint ratio is flat at 0.0004 while the duplicate rate triples. Is the complaint metric
broken?**

- **Answer:** It is not broken, it is insensitive. With `b = 0.0004` and `b_dup = 0.00072`, a duplicate
  rate of 0.04 moves the ratio to 0.0004128, a 3.2 percent relative change that is inside daily noise.
  Solving for the ceiling gives `(0.0005 − 0.0004)/(0.00072 − 0.0004) = 0.3125`, i.e. the metric would
  tolerate one duplicate in three sends. The metric measures sentiment, which is a threshold phenomenon
  in a subpopulation; the ledger measures structure, which is what the ceiling must key on.
- **Reasoning:** A linear mixture of two rates separated by a factor of 1.8 cannot be sensitive to a small
  mixing weight. Detecting a 1 percent defect with a metric whose signal is 1.8× requires averaging over
  populations far larger than the affected one.
- **Follow-up:** If you had to defend a complaint-based ceiling, what would make it work? *(Answer: a
  per-recipient duplicate count joined to complaints, so the metric becomes "share of recipients with ≥1
  complaint and ≥2 copies", not a population average.)*
- **Red flag:** "Complaints are the ground truth, so duplication must not matter."
- **Scoring:** 3 for the arithmetic and the structure-versus-sentiment distinction; 2 for recognising
  insensitivity without the numbers; 1 for "add duplication to the dashboard"; 0 for dismissing the
  discrepancy.

**Q6. A cap of six engagement intents per day suppresses half your volume. Does that mean your platform
is over-built?**

- **Answer:** No, it means the capacity plan must be built on accepted volume, not intake. Intake peaks at
  2,100/s; accepted volume is 36,000,000 − 18,368,000 = 17,632,000/day, a mean of 204/s and a peak minute
  of about 1,029/s. Sizing on intake over-provisions by about 2×; sizing only on accepted volume fails
  when the cap is relaxed, and the cap is a product-owned number that can change without a release. So the
  plan uses accepted volume for steady state and intake for the headroom test.
- **Reasoning:** The suppression share is a function of the per-recipient distribution, and the mean is
  misleading: 11.25 intents per active user hides a top bucket averaging 22.25. Any capacity number
  derived from a per-user mean will be wrong by the shape of the distribution.
- **Follow-up:** The cap rises to 8. What happens to the peak? *(Answer: suppressed volume falls, accepted
  peak rises toward intake; the fleet must absorb the difference, so the headroom test is what protects
  you.)*
- **Red flag:** "Just size for the mean, caps will keep it flat."
- **Scoring:** 3 for both the 51 percent and the two-rate distinction; 2 for noticing the cap matters; 1
  for "size for peak"; 0 for sizing on the mean.

**Q7. Where does Little's law enter, and what does it get wrong?**

- **Answer:** It converts a latency target into in-flight work: `L = λW`. At 2,100/s and a 2-second system
  time, `L = 4,200` intents in flight; at 8 concurrent per worker that is 525 workers, and at 0.70
  utilisation 750. It gets two things wrong if applied naively: it assumes a stable system (so it is
  meaningless during an outage when λ exceeds capacity and the queue grows without bound), and it says
  nothing about the *distribution* of wait, so meeting the mean does not meet a p99.
- **Reasoning:** Little's law is an identity for any stable system, not a model, which is why it is
  trustworthy for the count and useless for the tail. The tail needs the service-time distribution or an
  empirical p99.
- **Follow-up:** If p99 latency must be 10 seconds, what does `L` tell you? *(Answer: nothing directly; you
  need the p99 service time or a queueing approximation, and the utilisation margin is how you buy tail
  headroom.)*
- **Red flag:** "Little's law gives the p99 if you plug in the p99 latency."
- **Scoring:** 3 for the computation and both limitations; 2 for the computation only; 1 for naming the
  law; 0 for confusing it with a queueing model.

**Q8. A push provider returns 500 for requests it has accepted. Your retry wrapper retries twice. What
happens at 2,100 intents/s over ten minutes?**

- **Answer:** 1,260,000 intents in the window; with two retries each, up to 3,780,000 attempts and up to
  2,520,000 user-visible second copies — 70× the daily C-05 allowance in ten minutes. The correct fix is a
  fourth response class, `unknown`, which is not retried when the dedup claim already exists; the retry
  becomes a suppressed attempt and the outcome is a delay bounded by the outage plus drain.
- **Reasoning:** The wrapper's bug is a classification error, not a retry-count error. No retry policy can
  be correct if 5xx is mapped to "not accepted", because the mapping is false.
- **Follow-up:** Which class may retry on `unknown`, and why? *(Answer: transactional, once, because a
  duplicate passcode is tolerable while a missing one blocks the user; engagement must not.)*
- **Red flag:** "Lower max_retries to 1" — reduces but does not fix.
- **Scoring:** 3 for the numbers and the `unknown` class; 2 for the class without numbers; 1 for lowering
  retries; 0 for "add exponential backoff".

**Q9. Two workers decide the same intent concurrently and both see a clear dedup state. What prevents a
double send, and what does not?**

- **Answer:** A conditional write (`put_if_absent`) prevents it: exactly one worker's write succeeds, the
  other observes the stored state and exits. A read-then-write with a TTL does not, because both reads
  happen before either write and both proceed. Adding a distributed lock helps only if the lock is
  acquired before the read; if the lock is taken after the read, the race is unchanged.
- **Reasoning:** The serialisation point must cover the read that justifies the decision. Locks,
  conditional writes, and compare-and-swap all work when placed correctly and all fail when placed after
  the observation.
- **Follow-up:** What if the winner crashes after the conditional write and before the provider call?
  *(Answer: the key sits `in_flight` until its lease expires, then moves to `unknown`, and a single
  conditional retry is permitted for transactional.)*
- **Red flag:** "Use Redis SETNX" without saying where in the sequence.
- **Scoring:** 3 for the conditional-write placement and the crash case; 2 for the primitive without
  placement; 1 for "use a lock"; 0 for "check then send".

**Q10. You see a spike of sends at 03:00 local. Give three causes and one check that separates them.**

- **Answer:** Causes: (a) quiet hours computed from the event timestamp rather than the decision time, so
  a delayed event is evaluated in the wrong hour; (b) a fixed UTC offset instead of a timezone database,
  wrong by an hour across DST; (c) a cap counter resetting at the wrong boundary, releasing held intents
  early. Check: recompute the local hour from the stored timezone at `decision_time` and compare with the
  recorded `recipient_local_hour`. If they differ, it is a code defect (a or b), and the DST test
  distinguishes them; if they agree and the send happened, it is the release path (c), which is visible in
  the `hold` reason codes.
- **Reasoning:** The three causes are distinguished by *which clock* and *which record* disagree. Recording
  both the computed hour and the timezone version turns an unanswerable question into a diff.
- **Follow-up:** A recipient has devices in two timezones. What is "local"? *(Answer: define it as the most
  recently active device's zone, record the choice, and treat the ambiguity as a stated assumption.)*
- **Red flag:** "Add a timezone check" without saying which timestamp.
- **Scoring:** 3 for three causes and a discriminating check; 2 for two causes; 1 for one; 0 for "fix the
  cron".

**Q11. Your per-recipient cap counter lives in the application database. What fails first under load, and
how do you detect it?**

- **Answer:** Read-modify-write contention fails first: concurrent intents for one recipient both read 5
  and both write 6, so the cap permits more sends than intended, and the row lock serialises the requests
  that carry the most traffic. Detect by comparing the counter to a count of accepted sends per recipient
  per window: counter lower than actual means lost updates; counter higher means duplicated intents
  inflated it. The fix is an atomic increment or a versioned conditional write, in a store that can absorb
  the write rate.
- **Reasoning:** Two failures with opposite signs (lost update versus double count) can cancel, producing
  a cap that looks correct on average and is wrong per recipient. Only a per-recipient comparison exposes
  it.
- **Follow-up:** Why not approximate the counter, as a rate limiter would? *(Answer: because the counter is
  a promise to the recipient, and approximation is a broken promise; a rate limiter's harm from a 1 percent
  error is small, a cap's is a complaint.)*
- **Red flag:** "Caps are approximate anyway."
- **Scoring:** 3 for the race, the detection, and the fix; 2 for the race and fix; 1 for "use Redis"; 0 for
  "it's just a counter".

**Q12. Design the metric that would have caught the defect the complaint ratio missed. State all six
attributes.**

- **Answer:** Numerator: intents whose key reached two `accepted` attempts. Denominator: intents whose key
  reached at least one `accepted` attempt. Population: all classes, split by class because the ceilings
  differ. Window: rolling one hour, reported daily. Comparator: the C-05 ceiling of 0.002, with an action
  threshold at 0.004. Guardrail: halt the engagement class at 0.004 while transactional continues, because
  halting transactional fails the delivery floor.
- **Reasoning:** The denominator must exclude suppressed and expired intents, which cannot duplicate;
  using all intents would understate the rate by the 51 percent suppression share, making 0.002 read as
  0.001.
- **Follow-up:** Why a rolling hour rather than a day? *(Answer: a duplicate storm is minutes long; a daily
  average dilutes a ten-minute storm by a factor of 144.)*
- **Red flag:** A metric with no denominator, e.g. "count of duplicates".
- **Scoring:** 3 for all six attributes plus the denominator argument; 2 for five attributes; 1 for a
  reasonable metric minus the denominator; 0 for "monitor duplicates".

**Q13. You must choose the cutover order for three classes. Volume is 60/35/5 percent engagement,
transactional, compliance. What order and why?**

- **Answer:** Engagement, then compliance, then transactional, by tranches of recipients. The ordering
  criterion is blast radius per failure, not volume: an engagement defect is annoyance, a compliance
  defect is caught in human review because volume is small, and a transactional defect locks a user out.
  Volume is a bad proxy for blast radius, and the largest class also happens to be the safest to expose.
- **Reasoning:** A migration is a sequence of experiments, and an experiment should be run where the cost
  of a wrong result is lowest while the signal is still representative. Engagement gives the most signal
  at the lowest cost.
- **Follow-up:** Why tranche the transactional cutover rather than doing it atomically? *(Answer: the
  delivery floor is a 0.995 commitment; tranching means a defect affects 20 percent of passcodes rather
  than 100 percent, and rollback is a flag rather than a data operation.)*
- **Red flag:** "Smallest class first to limit damage" — confuses volume with risk.
- **Scoring:** 3 for the order plus the blast-radius criterion; 2 for the order with a weaker reason; 1 for
  volume-based ordering with acknowledged risk; 0 for "big bang, it's just notifications".

**Q14. Estimate the storage for a 30-day hot ledger and a 400-day cold archive at 36 million intents/day.
What decision does the answer actually drive?**

- **Answer:** At 116 bytes per row, hot is 4.18 GB/day, 125.3 GB for 30 days, 200.4 GB with a 1.6 index
  and write-amplification factor. Cold at 40 bytes per row is 576 GB for 400 days. Total about 776 GB. The
  decision it drives is retention, not schema: cutting cold retention from 400 to 90 days takes the archive
  from 576 GB to 130 GB, a saving of 446 GB that dwarfs any row-size optimisation.
- **Reasoning:** The compressed per-row size is 2.9× smaller but the retention period is 13.3× longer, so
  the archive dominates. Multiplying rate by retention is the whole calculation, and the largest lever is
  always the retention exponent.
- **Follow-up:** What makes a wider hot window worth 200 GB? *(Answer: support escalation latency — if
  complaints arrive at day 3–14, a 7-day hot window sends most investigations to cold storage, which costs
  query latency rather than bytes.)*
- **Red flag:** Quoting a per-row size without multiplying by volume and retention.
- **Scoring:** 3 for both figures and the retention conclusion; 2 for the arithmetic without the
  conclusion; 1 for an order-of-magnitude guess; 0 for "storage is cheap".

**Takeaway:** the interview pack mixes four pure calculations, five diagnoses, and five design judgements;
each answer carries a number or a discriminating check, not a principle.

## 15. Quick Reference Card

| Item | Value or rule | Why it matters |
|---|---|---|
| Idempotency key | `(recipient_key, event_id)`, claimed with `put_if_absent` | Retries become suppressed attempts; channel fallback is a state attribute, not a new key |
| Rate anatomy | mean 416.67/s, peak hour 900/s, peak minute 2,100/s (spike 5.04) | Size admission on 2,100/s; the hour figure is not a design input |
| Retry budget | 3 attempts; multiplier 1.041664; residual failure 2.56e-6 | Costs 1,499,904 extra attempts/day, about 360 USD |
| Retry safety | Retry only behind an existing claim; `unknown` is never silently retried | A 10-minute outage otherwise produces 2,520,000 second copies |
| Dedup window | 60 min (5 min fails at 388,800 escapes/day; 15 min margin too thin) | 19,440 escapes/day = 0.00054, 27 percent of ceiling |
| Duplicate ceiling | 0.002 structurally, not via complaints | Complaint ratio would tolerate 0.3125 duplication |
| Suppression | Fail closed on cold cache; unsubscribe effective ≤ 60 s | Stale suppression is a trust and compliance breach |
| Quiet hours | 21:00–08:00 local from stored TZ at `decision_time`; transactional exempt | 9,900,000 intents held; release at a designed rate, not a cron drain |
| Release crest | 1,375/s over 120 min; 917/s over 180 min | A held queue is a self-inflicted second peak |
| Fleet sizing | 750 workers at `W_sys` = 2 s, 0.70 utilisation | 525 workers at full utilisation diverges on a surge |
| Admission order | decide → claim → enqueue → call → record | Claiming before deciding locks suppressed intents out of their window |
| Ledger retention | 30 d hot (200.4 GB), 400 d cold (576 GB) | Retention, not row size, is the dominant cost lever |
| Primary control | ledger duplicate rate, rolling 1 h, class-split | The only metric with no lag and no averaging blindness |
| Cutover order | engagement → compliance → transactional, by tranche | Order by blast radius, not by volume |

The card is a compression of §6, §8, and §9 only; where the card and a section disagree, the section
wins, because the section carries the validity condition. Two entries are the ones most often forgotten
in review: the admission order (decide before claim) and the `unknown` classification. Both are one-line
rules whose absence produces the two failure modes in §12.1 and §12.4 that are hardest to detect from
outside the platform.
