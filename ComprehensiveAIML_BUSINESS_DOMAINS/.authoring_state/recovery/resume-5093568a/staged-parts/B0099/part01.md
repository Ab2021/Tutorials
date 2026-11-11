---
title: "25 Common Issues — Omnichannel Loyalty and Fulfilment"
topic: "Common Issues"
collection: "retail-02"
file_id: "04"
level: "core"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
notation_style: "plain-text"
prerequisites: ["collection sheets 01-03"]
pattern_tags: ["diagnostics", "retail-operations", "point-in-time"]
source_status: "OPEN"
owner_role: "retail operations analyst"
source_index: "INDEX.md"
---

# 25 Common Issues

> **Purpose:** Give an on-call retail analyst a bounded diagnostic loop for the loyalty and fulfilment stack — identity, availability, promise, pick, notification, and points — so that a symptom is converted into one testable cause before any number is repaired.
> **Evidence:** Every quantity in this file is a constructed teaching value and is labelled ILLUSTRATIVE. The arithmetic has been recomputed line by line; the mechanisms are DERIVED from how these systems are built. No company, deployment, or measured result is claimed.
> **Reader task:** for a stated symptom, name the category, separate the competing causes, choose a check that gives a different answer for each cause, apply the narrowest correction that is reversible, then verify with a pre-declared signal.
> **Level:** core — assumes the event taxonomy, identity controls, and promise metrics sheets in this collection.
> **Retrieval cue:** if a promise rate moved, start at Section 4.2 and Section 8 entry CI-11; if a points balance moved, start at CI-19.

## 1. File Header

This is the troubleshooting companion to the three teaching sheets in this collection. Those sheets
build the concepts: what an event is, how a customer is resolved to one identity, and what a
fulfilment promise is measured against. This file does the opposite job. It starts from the
symptom a person actually sees — a dashboard cell that moved, a queue that will not drain, a
customer complaint that repeats — and works backward to a mechanism.

**Scope.** Twenty-eight catalogued issues across six subsystems: customer identity, inventory
availability, delivery and pickup promise, store pick capacity, notification and preference,
and the loyalty points ledger. Each entry names a category, a symptom, at least two competing
causes, a check that discriminates between them, a correction, a prevention, and a verification.

**Out of scope.** Pricing and promotion elasticity, media attribution, legal interpretation of
consumer-protection rules, and any vendor-specific product behaviour. Where a diagnosis depends on
a contract or a statute, this file says so and stops: it does not guess at the clause.

**How to use the file under time pressure.** Read Section 3 for the definitions, then jump to the
entry whose symptom matches. Do not read the catalog front to back during an incident; the entries
are numbered for lookup, not for narrative sequence. Section 12 is the card to keep open while a
call is live.

### 1.1 Identifier grammar

Every entry has a stable identifier of the form `CI-NN`, two digits, zero-padded. Identifiers are
never reused and never renumbered, because runbooks, tickets, and post-incident notes cite them.
Adding an entry appends `CI-29`; retiring an entry leaves its number vacant and records the reason
in the collection backlog. The category tag is part of the entry, not of the identifier, so an entry
whose category is later reclassified keeps its number.

The four category tags are deliberately coarse. They answer one question: what kind of assumption
broke? If the answer is not obvious in ten seconds, the diagnosis is not yet ready to be written
down, and the analyst should return to the symptom.

| Tag | The assumption that broke | Typical repair cost | Typical blast radius |
|---|---|---|---|
| VIOLATED_ASSUMPTION | the pipeline was told something about the world that stopped being true | contract change, schema version, renegotiation with a source owner | one field, every downstream consumer of it |
| NUMERICAL_EDGE | the arithmetic was right in the middle of its range and wrong at a boundary or a rounding step | code fix plus a replay of the affected window | exactly the records that hit the boundary |
| SCALE_FAILURE | a design that works at ten units per minute was never tested at a thousand | capacity, partitioning, batching, or admission control | the whole service, then everything that queues behind it |
| CONCEPT_CONFUSION | two teams are computing different quantities under the same word | definition, documentation, and a shared test fixture | every comparison built on the ambiguous word |

### 1.2 Severity and first response

Severity is assigned by blast radius and reversibility, not by how loud the alert was. A quiet
metric that is silently wrong for a month is more expensive than a loud one that is wrong for an
hour, because the quiet one has already been used to make decisions.

```text
SEV-1  money or promise moving wrongly and still accruing      -> contain now, freeze the decision, page the owner
SEV-2  a decision input is wrong but a fallback rule is running -> freeze the input, keep the fallback, fix within a day
SEV-3  a metric is wrong and no live decision depends on it    -> annotate the chart, fix within the sprint
SEV-4  a definition is contested and both variants are stored  -> schedule a definition review, do not renumber
```

The first response to any severity is the same and is not optional: preserve the evidence. Snapshot
the source extract, the decision log, the model or rule version, and the operator actions taken
during the window. Containment that destroys the record is not containment; it is a second incident
layered on the first, and the second one is usually harder to reconstruct.

### 1.3 The four-step loop

Every entry in Section 8 assumes the same loop. Skipping a step is the most common reason a
correction is applied to the wrong subsystem and the symptom returns a week later.

1. **Bound the symptom.** State the key, the time window, the population, and the direction of the
   change. "Pickup delays are up" is not a bound; "pickup ready-late rate for store 214, orders
   placed 14:00–18:00, week 42, up from 3.1% to 7.4% of 820 orders" is a bound.
2. **Name two causes that predict different things.** A hypothesis that predicts the same
   observation as its rival is not a hypothesis. Write down what each cause would make visible and
   where.
3. **Run the cheapest discriminating check.** Usually a slice, a join against a second source, or a
   replay of one hour. Prefer a check that can only pass for one cause over a check that is merely
   consistent with it.
4. **Correct narrowly, then verify against a declared signal.** Write the expected post-correction
   value before applying the fix. A verification that is chosen after seeing the result is not a
   verification.

### 1.4 What this file will not do

It will not tell you that a number is wrong because a model drifted. Drift is a description of a
symptom in the feature distribution, not a mechanism; the mechanism is always one of the four
categories above. It will not authorise a backfill of historical decisions — replaying a corrected
fact into a past decision manufactures a result the business never had. And it will not convert a
teaching scenario into an incident claim: no entry here is a report of something that happened at a
named retailer.

## 2. Concept Map / Mental Model

### 2.1 The stack as a chain of promises

The loyalty and fulfilment stack is not one pipeline. It is six subsystems in series, and the
customer experiences only the product of their successes. That product structure is the single most
useful fact in this file, because it explains why a small degradation in each of four stages
produces a large move in the number the customer sees.

```text
[customer] --identity--> [one profile] --assortment--> [availability claim]
                                                              |
                                                     [promise: date/slot]
                                                              |
                                            [pick + pack at a node] --notification--> [customer]
                                                              |
                                                     [fulfilment outcome]
                                                              |
                                                     [points / tier ledger]
```

Each arrow can fail independently, and each failure has a different signature in the data. An
identity failure shows up as a profile with implausible breadth. An availability failure shows up as
a promise that could not be kept at the moment it was made. A promise failure shows up as a
completion timestamp later than the quoted timestamp, with nothing else wrong. A pick failure shows
up as a pick duration or a substitution, not as a promise. A notification failure shows up as a
customer who did not know, which is invisible in fulfilment data but loud in the contact centre. A
ledger failure shows up as a points balance that disagrees with the arithmetic of its own
transactions.

The practical consequence: never diagnose across two arrows at once. If the promise rate fell and
the pick duration rose, first establish whether the promise fell *because* of the pick or alongside
it, by holding the node and the hour fixed and comparing the two.

### 2.2 Three clocks

Every entry in the catalog is, at bottom, a disagreement between three clocks. Naming them makes
the diagnosis mechanical.

- **Event time** — when the thing happened in the world: the order was placed, the item was
  scanned, the parcel was handed over.
- **Processing time** — when the system learned about it: the batch landed, the message was
  consumed, the reconciliation job ran.
- **Decision time** — when a fact was allowed to influence a choice: the promise was quoted, the
  points were posted, the offer was targeted.

A late arrival is not automatically a defect. It is a defect only when a fact crosses a decision
boundary late, so that the decision was made without it and the record now implies it had it. That
distinction is why the first question in every entry is "which clock moved", and why the correction
for a late-arriving fact is an annotation on the decision, never a silent restatement.

```text
event time        t_e   ---- observed
processing time   t_p   ---- arrives at t_e + lag
decision time     t_d   ---- uses facts with t_p <= t_d
defect condition: exists f with t_e(f) <= t_d and t_p(f) > t_d and f changes the decision
```

Read the condition literally. If the fact would not have changed the decision even had it arrived,
the late arrival cost nothing and must not be counted as an error in a quality score. If it would
have changed the decision, the record needs a flag, not a rewrite.

### 2.3 Four categories, one question each

The category tags are not a taxonomy of causes; they are a taxonomy of the *check* you run next.
Each tag implies a question whose answer is either yes or no, and the questions are chosen so that a
"yes" excludes the other three tags for that symptom.

| Tag | The question that settles it | If yes | If no |
|---|---|---|---|
| VIOLATED_ASSUMPTION | did an input's meaning or availability change at a specific instant? | find the change record, version the contract | the inputs were stable; look at arithmetic or load |
| NUMERICAL_EDGE | does the error vanish when you exclude the boundary records? | enumerate the boundary condition, fix the comparator | the error is uniform; look at a source or a definition |
| SCALE_FAILURE | does the error grow with volume while per-unit latency stays flat? | look for a queue, a lock, or a batch window | the volume is not the driver |
| CONCEPT_CONFUSION | do two teams disagree on the denominator for the same headline? | write one fixture both teams must reproduce | the definitions agree; the data is wrong |

The discipline that makes this work is refusing to accept an answer that is compatible with two
tags. "Data quality" is compatible with all four and therefore settles nothing.

### 2.4 Containment versus correction

Two actions, deliberately kept separate, because conflating them is how a team loses the ability to
learn from an incident.

**Containment** is reversible, narrow, and fast. It changes what the system does *from now on*: cap
the queue, switch the promise to a conservative rule, route affected orders to manual review, pause
points posting for one programme. Containment never edits history.

**Correction** is durable and slow. It changes the mechanism: the contract, the comparator, the
partition key, the shared definition. Correction is verified by a replay on a frozen extract, not by
watching the live dashboard for an hour.

A useful heuristic: if you cannot name the condition under which you would roll the action back,
you have not designed a containment action — you have made a permanent change during an incident,
which is how permanent changes get made without review.

### 2.5 The observability triangle

For each promise you care about, three numbers must be visible at the same grain, or the diagnosis
will be guesswork:

1. the **rate** the customer experiences (promise kept, points correct, substitution accepted);
2. the **population** over which that rate was computed, with its eligibility rule;
3. the **decision record** that produced the outcome, including the version of the rule or model.

Most organisations have (1) richly and (2) partially, and (3) only in logs that expire in fourteen
days. When an issue is older than the log retention, the diagnosis degrades from measurement to
reconstruction, and reconstruction is where invented causes enter a post-incident review.

### 2.6 The mental model in one sentence

A retail fulfilment symptom is almost always the product of several individually acceptable stage
reliabilities crossing a threshold that someone promised a customer, and the fastest way to find
which stage moved is to convert the compound rate back into its factors and compare factor by
factor at matched grain.

## 3. Foundational Definitions

Definitions are given with units and grain because the majority of the entries in Section 8 are
caused, ultimately, by two teams using one word for two quantities.

| Term | Definition | Units / grain | Why it matters diagnostically |
|---|---|---|---|
| promise | the date, window, or slot quoted to the customer at order confirmation | timestamp or interval, per order | the only commitment that creates a contractual expectation |
| promise kept | fulfilment completed at or before the quoted interval under the stated tolerance | boolean, per order | numerator of every on-time rate; tolerance must be fixed before measurement |
| stage pass rate | fraction of units entering a stage that leave it without a defect attributable to that stage | fraction, per stage per window | the factor in the compound reliability product |
| compound reliability | product of stage pass rates for the stages an order must traverse | fraction, per cohort | the customer-visible rate, always below the arithmetic mean of stages |
| availability claim | the statement that a unit can be sold from a specific node | boolean with a timestamp, per unit per node | an availability claim is a promise about the future made from a stale past |
| oversell | accepted orders exceeding the node's feasible capacity for the quoted interval | count, per node per interval | produces a failure that looks like a pick defect but is a promise defect |
| substitution | a different unit supplied or offered in place of the ordered unit | event, per line | a substitution can be a success for the customer and a defect for the metric |
| node | the physical or logical location from which a unit is fulfilled | identifier | capacity and accuracy are properties of a node, never of a region |
| identity | the resolved association between observations and one customer | identifier | merging and splitting are the two error directions, with asymmetric costs |
| block | a group of candidate records considered together for identity resolution | set, per run | false merges scale with pairs, so the block size is a cost driver |
| effective time | the earliest instant a fact may lawfully or operationally change a decision | timestamp, per fact | separates a late fact from an early decision |
| maturity | elapsed time after which an outcome is treated as final | duration, per event type | measuring before maturity manufactures apparent regressions |
| breakage | the share of issued loyalty value never redeemed | fraction, per programme per cohort | converts a points liability into an expected cost |
| points liability | value of unredeemed issued points under a stated redemption assumption | currency, per programme | a liability that must be estimated, not a balance that can be observed |
| tier | a customer segment defined by a rolling qualifying measure | label with a qualifying window | window changes silently re-segment customers at the boundary |
| qualifying spend | spend counted toward a tier over the qualifying window after exclusions | currency, per customer per window | the exclusion list is the most contested object in a loyalty programme |
| offer eligibility | the set of customers a promotion may lawfully and economically target | set, per campaign per day | eligibility computed at a different time than targeting produces leakage |
| suppression | the removal of a customer from a targeting set for a policy reason | set | suppression applied after targeting is a defect, before it is a control |
| consent state | the recorded permission for a channel and purpose, with its own timestamp | state machine, per customer per purpose | consent has event time; a stale consent state is a legal exposure |
| pick | the physical retrieval of units at a node for one order | task, per order per node | pick capacity is the standard constraint behind slot oversell |
| pick rate | units or lines picked per labour-hour at a node under stated conditions | lines per hour | a planning input; must never be inferred from a completed-orders count |
| slot | a quoted collection or delivery interval with finite capacity | interval with capacity count | the unit at which oversell is possible and therefore measurable |
| queue depth | units waiting for a resource at an instant | count | the state variable that predicts latency; the rate does not |
| drain time | queue depth divided by net service rate, holding arrivals fixed | hours | the number that tells you whether containment is working |
| notification | the outbound message that tells the customer a state changed | event, per channel per attempt | the only subsystem whose failure is invisible in fulfilment data |
| delivery attempt | one physical attempt to hand over a parcel or order | event, per parcel | the unit for cost and for customer-effort metrics |
| contact reason | the categorised reason a customer contacted service | label, per contact | the closest thing to ground truth on notification failure |
| attribution window | the period after an exposure during which an outcome is credited to it | duration | a shorter window moves credit, not behaviour |
| cohort | a set defined by a shared starting event and evaluated over the same elapsed time | set with index time | the only valid basis for comparing retention or repeat rates |
| baseline | the rule or policy that would have applied without the candidate change | decision record | the comparator for every claimed improvement |
| guardrail | a secondary measure that must not degrade for a change to be acceptable | metric with threshold | protects against optimising the headline at a cost elsewhere |
| MDE | the smallest true effect a test can detect at declared error rates | absolute difference | determines whether a test can answer the question at all |
| replay | re-execution of a pipeline over a frozen extract to check reproducibility | procedure | the only sound verification for a correction |
| fixture | a frozen input set with its expected output, shared between teams | artefact | converts a definition dispute into a testable claim |
| tolerance | the permitted deviation before a promise is counted as missed | duration | part of the definition; changing it changes the rate retroactively |

Three of these deserve a paragraph more than a table cell.

**Availability is a claim about the future made from a stale past.** The inventory record says a unit
is at the node; the claim says the unit will still be there when a picker reaches it. Between those
two statements sit three failure modes: the record was already wrong, the unit was sold by another
channel in between, or the unit exists but is unreachable. All three present identically as a
short-pick. Only the third is a store-discipline problem; the first two are promise-design problems.

**Breakage is an assumption, not a measurement.** The share of points never redeemed is estimated
from cohort behaviour under a specific expiry policy. When the policy changes, the estimate changes
without any customer behaviour changing, and a liability that appears to move has in fact been
re-priced. Any comparison of liability across a policy change must say so explicitly.

**Consent has event time.** A consent record without a timestamp for when the permission changed is
not a record; it is a claim. The failure mode is a campaign that suppressed a customer who had
re-consented, or contacted one who had withdrawn — and both are found in the same place, the change
log, not the current state table.
