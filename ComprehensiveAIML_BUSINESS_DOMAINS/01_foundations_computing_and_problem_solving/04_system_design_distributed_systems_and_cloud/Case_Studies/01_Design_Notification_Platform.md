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
