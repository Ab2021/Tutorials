---
title: "Omnichannel Event Taxonomy"
topic: "Event taxonomy, lineage and promise-state semantics for omnichannel retail"
collection: "retail-02"
file_id: "01"
level: "core"
level_index: 2
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
prerequisites: ["basic rates and denominators", "event timestamps", "one relational join or one streaming-join concept"]
pattern_tags: ["retail-operations", "event-taxonomy", "point-in-time", "identity-resolution"]
source_status: "OPEN"
owner_role: "retail analytics practitioner"
source_index: "INDEX.md"
review_trigger: "Any change to event names, subject keys, channel enumeration, promise-state vocabulary, or late-arrival policy"
---

# Omnichannel Event Taxonomy

> **Purpose:** Give a retail analytics team a naming and lineage discipline that survives contact with three order channels, two inventory systems, and a returns process that nobody owns end to end.
> **Prerequisites:** Comfort with ratios and their denominators, the ability to read a timestamped event table, and one worked example of a join that fanned out.
> **Key outcome:** Given a proposed funnel, promise metric, or loyalty trigger, name its grain, its event lineage, its dedupe rule, its clocks, and the denominator it may legitimately use — then say which of those the current warehouse cannot supply.
> **This file does NOT cover:** retailer-specific system inventories, legal or privacy determinations on customer data, model training, or measured impact from any named organisation.
> **Level:** 2 — Core Methods.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED relations and ILLUSTRATIVE counts. No prevalence, revenue, or deployment result here is measured.
> **Retrieval cue:** If two teams report different order counts, or if a funnel changed after a backfill, start at Section 8 and work back to Section 4.

## 1. File Header

**Why this file exists.** An omnichannel retailer does not have one record of a customer's behaviour. It has at least five: the web storefront's clickstream, the mobile app's event stream that lags the web one by a release cycle, the order management system's transaction log, the store point-of-sale journal, and the warehouse management system's task log. Each was designed for its own consumer — the storefront for rendering, the order system for fulfilment, the WMS for pick routing — and none was designed to be joined to the others at the grain an analyst needs. The taxonomy is the thin, versioned contract that makes those five systems speak about the same subject, at the same time, with the same verb.

**What goes wrong without it.** The most expensive failure is not a wrong dashboard number. It is a decision taken on a number whose denominator changed underneath it. A team reports that thirty-eight per cent of buy-online-pickup-in-store orders were collected within two hours, a director commits staffing to that level, and three weeks later the figure drops to thirty-one per cent — not because service worsened but because the mobile release started emitting a second `order_collected` event for the same collection. Nobody lied; the taxonomy had no uniqueness rule, so the count of collections became a count of collection events.

**What this sheet delivers.** Four things a practitioner can use on Monday. First, an event grammar that forces subject, verb, channel, and effective time into every name. Second, a lineage model that tells you which joins are one-to-one, which fan out, and which silently drop rows. Third, the arithmetic for coverage, path completeness, and promise-state accuracy, each with a validity condition. Fourth, a failure catalogue whose entries are distinguishable from one another by a specific query rather than by an opinion.

**What this sheet refuses to do.** It does not propose a vendor schema, because the schema is the least portable part of the problem. It does not claim that any retailer achieved a particular improvement. It does not treat the taxonomies published by web-analytics vendors as authoritative, because those vocabularies were built for marketing attribution and are systematically silent about inventory, substitution, and partial fulfilment — exactly the states that dominate omnichannel operations.

**Why "taxonomy" and not "data dictionary".** A data dictionary describes columns. A taxonomy describes the decisions those columns are allowed to support. The distinction matters because two columns with the same name can support opposite decisions. `ship_date` in the order system means "the parcel left the building"; `ship_date` in the carrier feed means "the carrier accepted the parcel"; they differ by hours to days, and the difference is precisely the quantity a delivery-promise model must not confuse with itself. A dictionary records both spellings; a taxonomy records which one the promise is measured against.

**Source status and honesty.** The unit's planning tree lists this sheet against an open source body that was not available for technical review. Every definition is therefore DERIVED from stated premises, every number is ILLUSTRATIVE, and every formula is a relation that can be checked with arithmetic alone. Nothing here should be quoted as an industry benchmark, and no figure should be placed in a board pack without substituting the retailer's own counts into the same blocks.

**Notation conventions used throughout.** `t_e` is the time the business event occurred at the point of origin, `t_i` is the time the platform ingested the record, and `t_v` is the business-effective time used for reporting. `S` is a subject key (customer, order, or fulfilment unit), `E` is an event instance, `T` is an event type. Counts are per stated calendar day unless a window is named. Currency is US dollars; the symbol `$` never changes unit inside one block. Ratios are dimensionless in [0,1] and percentages appear only when explicitly written with a per-cent sign. Subscript `u` marks unit grain (one physical parcel or one collected line), and subscript `o` marks order grain (one commercial order header).

**The three clocks and why all three are named.** Event time is when the customer did the thing. Ingest time is when the platform learned about it. Effective time is the time the reporting layer chooses to attribute it to. A store point-of-sale can go offline at 14:05 and flush its journal at 18:40; those collected orders have `t_e` inside the afternoon and `t_i` after the evening backlog. Any metric whose numerator uses ingest time and whose denominator uses event time will produce a saw-tooth series that looks like a demand pattern and is actually a network artefact. Section 4.3 turns that observation into an inequality, and Section 8 gives the query that separates the two.

**How the sections fit together.** Sections 2 and 3 build the vocabulary: the four planes an event must be placed in, then the definitions with their confusion boundaries. Section 4 derives the four relations this sheet exists to teach — coverage, path completeness, join fan-out, and promise-state accuracy — plus a mechanism block for the state machine that the last of those depends on. Section 5 traces one order through six systems and re-derives the load-bearing numbers by an independent route. Section 6 draws the shapes, Section 7 compares candidate designs, and Section 8 collects the failure mechanisms with the query that discriminates each from its look-alikes. Section 9 costs the scheme, Section 10 states the invariants compactly, Section 11 tests whether you can use it, and Section 12 is the card to pin above a desk.

**Reader contract.** After Section 3 you should be able to write, for any proposed metric, five clauses: subject, event verb and qualifier, channel and node, time basis, and uniqueness rule. If any clause is missing, the metric is not specified yet, and it does not matter how sophisticated the model behind it is. That five-clause sentence is the real deliverable of this sheet; the arithmetic exists to make violations visible.

**Maintenance triggers.** This sheet must be revised as a set — Sections 3, 4, 7 and 8 together — when any of the following changes: the enumeration of channels (a new marketplace, a new store format), the promise-state vocabulary (for example introducing "ready for partial collection"), the subject-key policy (a new loyalty identifier or a merge of guest records), the late-arrival window (a new offline store batch schedule), or the retention horizon used for replay. Changing one of these and leaving the others is how a taxonomy becomes folklore.

**A note on the word "canonical".** Canonical here means "single definition with a named owner and a version", not "correct". A canonical event name can still be wrong about the world; what it cannot be is ambiguous about which system produced it. The version number is the escape hatch: when the definition must change, the version changes, and every downstream metric that consumed the old version is re-computed or explicitly frozen. Retrofitting a version field after eighteen months of unversioned events is the single most expensive repair in this domain, because the historical rows cannot be classified retrospectively.

**Why the sheet leads with lineage rather than with schema.** A schema can be regenerated from a lineage graph in an afternoon, because the graph already states which identifiers exist and which facts are attached to them. The reverse is not true: a schema without lineage tells you the shape of a row and nothing about whether two rows describe the same parcel. Retail teams that start with the schema spend months discovering that `fulfilment_id` in the warehouse means "wave", not "parcel", and that discovery invalidates a quarter of published metrics.

**One sentence for the design review.** An event taxonomy is a promise that two systems describing the same moment will be joinable without a human deciding, case by case, what the words meant.

## 2. Concept Map / Mental Model

The model is four planes and one spine. Every event instance must be placed on all four planes, and every operational metric is a walk along the spine with a filter on one of the planes. Teams that skip a plane do not get a vaguer answer; they get a confidently wrong one, because the missing plane is usually the one that carries the denominator.

```text
  PLANE            QUESTION IT ANSWERS                       KEY
  subject          who or what is this about?                customer_id | order_id | unit_id
  time             when did it become true?                  t_e (event) | t_i (ingest) | t_v (report)
  channel/node     where did it happen?                      web | app | store_no | fc_no | carrier
  promise          what did we commit to, and is it still?   promise_state + promise_due_at

  SPINE (the order lifecycle, one direction only):

  order_placed -> order_accepted -> [allocated] -> [picked] -> [packed]
       -> shipped  -> in_transit -> delivered
       -> ready_for_pickup -> collected
       -> (any of the above) -> return_initiated -> return_received -> refunded
```

**Why four planes rather than one big event table.** Each plane has a different change frequency and a different owner. The subject plane changes when identity policy changes, which is quarterly at most. The channel plane changes when the estate changes, which is a store opening. The promise plane changes when commercial policy changes, which can be weekly during a promotion. The time plane never changes in definition but is violated constantly in practice. Separating them means a promotion can revise promise states without forcing a re-key of the identity table, and it means an identity merge does not silently rewrite the channel enumeration.

**The spine is a partial order, not a queue.** A click-and-collect order and a ship-from-store order traverse different subsets of the spine, and a return traverses a branch that rejoins at `refunded`. The spine is a directed acyclic graph with two entry modes (delivery and collection) and two exits (completion and refund). Modelling it as a linear funnel is the second most common taxonomy error after duplicate emission, because a linear funnel forces every order into the shipping path and then produces a large "stuck" bucket that is actually the collection path working correctly.

**Event naming grammar.** `subject.verb.object_qualifier@version`, lower snake case, verb in past tense, no channel inside the verb. For example `order.placed`, `unit.allocated`, `unit.substituted`, `unit.collected`, `return.received`, `refund.issued`. The channel lives in a field, never in the name, because a channel-in-name scheme forces a rename when the same action becomes possible in a new channel; `store_order_collected` cannot describe curbside collection without inventing a third name. The version suffix is not decoration: it is what lets a consumer pin behaviour while a producer evolves.

**The verb must denote a state change, not an intention.** `order_shipped` is a state change. `shipment_created` is a record creation, which may precede physical movement by a day and may never be followed by movement at all. Taxonomies that mix the two produce promise metrics that are simultaneously optimistic and unmeasurable. The test is mechanical: if the event can be emitted and then never followed by any physical consequence, it is not a spine event, and it belongs in a separate planning stream with a different owner.

**One event, one transition.** The strongest single rule in the sheet: an event instance is the unique witness of one legal transition of one unit's promise state. This gives three immediate consequences. Counts of events become counts of transitions, which makes them comparable across channels. Duplicate emission becomes detectable as two witnesses for one transition — the pair shares a transition identity. And replay becomes idempotent, because re-applying the same transition to an already-advanced state is a no-op rather than a second increment.

**Absence is ambiguous, and the taxonomy must say so.** In a five-system estate, "no `unit_collected` event" can mean the customer never came, the store's scanner was offline, the event failed validation on a null field, or the collection happened and was recorded as a generic `order_completed` by an older client. These are four different facts with four different operational responses, and a taxonomy that maps all four to "not collected" will report a service failure that did not occur. Section 8.2 gives the discriminating check; the design fix is an explicit `unit_collection_failed` and a heartbeat event per node.

**Identity is a plane, not a join.** Customer identity in retail is a graph: a guest checkout, a loyalty card, a device cookie, an email normalisation, and a store associate's manual merge. The taxonomy does not resolve identity; it declares which identifier each event carries, at which confidence, and which identifiers are allowed to be used for which metric. A loyalty metric may use only loyalty-linked orders; a fulfilment metric may use any order because the parcel does not care who bought it. Mixing the two produces a fulfilment number that quietly excludes guest orders — often a fifth of the estate.

**Two clocks per event are the minimum; three for promise events.** `t_e` and `t_i` are mandatory on every event. Promise events additionally carry `promise_due_at`, the commitment in force at the moment of the event, because the commitment changes when a carrier announces a delay. Storing only the current commitment on the order header destroys the ability to answer "was the promise kept?" for any order whose promise was revised. Section 4.3 formalises this as a per-event snapshot rule.

**The mental model in one sentence.** Place each event on all four planes, advance exactly one unit one legal step along the spine, and let every metric be a count of transitions divided by a population that was eligible to transition — anything else is a number with a hidden denominator.

**Where the model leaks.** Three places. First, split shipments: one order, three parcels, three independent promise clocks, so order-grain on-time rate is not the average of unit-grain rates. Second, substitutions: a picked unit that is not the ordered unit, which is a legal transition only if the customer consented, so consent becomes a required attribute. Third, cross-channel returns: an online order returned in a store, which joins two systems that share no transaction identifier and must be matched on receipt plus customer plus item, a join with fan-out. Sections 4.2, 4.3 and 8.4 handle the three respectively.

**How to read a stranger's taxonomy in ten minutes.** Count the verbs. If there are more verbs than state changes in the operations manual, the taxonomy is capturing intentions. Then check whether every event carries a subject key and both timestamps. Then check whether the promise vocabulary has a terminal state for failure, not only for success. A taxonomy with no failure states will report a completion rate that is a rate of recorded completions among recorded attempts, which is not the service level anyone is paid against.

**Why this sheet is upstream of the unit's other sheets.** Identity controls, promise metrics, and the case studies all consume the vocabulary defined here. When a case study reports a promise-accuracy figure, the numerator is a count of units whose terminal state was reached before `promise_due_at` as recorded on the event, and the denominator is a count of units eligible at `t_e`. If the taxonomy does not carry the snapshot, the case study cannot be reproduced, and reproducing it is the whole point of writing it down.
## 3. Foundational Definitions

Definitions are grouped by plane, and each row carries a boundary column naming the neighbouring concept it is routinely confused with. The boundary is the part that earns its space: a definition without a stated confusion is a definition nobody will apply.

### 3a Subject and grain

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Subject | The entity an event is about, keyed once and never re-keyed in place | `customer_id` \| `order_id` \| `unit_id` | Not the session and not the device |
| `order_id` | Commercial order header: one checkout commitment, one price, one customer | string key, 1 per checkout | Not a parcel; a 3-parcel order is still one order |
| `unit_id` | One physical or logical fulfilment unit: one parcel, or one collection tote line | string key, 1 per parcel | Not a line item; two identical lines can ship in one unit |
| Order grain | Counting once per `order_id` regardless of parcels | decision unit | Not the grain of the WMS task log |
| Unit grain | Counting once per `unit_id`, the grain the customer experiences | decision unit | Not the grain the finance ledger books revenue |
| Fan-out | One input row producing several output rows across a join | cardinality property | Not duplication; the rows are legitimately different |
| Duplication | The same fact recorded twice under two keys | data defect | Not fan-out; deduplication here loses real units |
| Channel | Originating commercial surface: `web`, `app`, `store`, `marketplace`, `call_centre` | enumerated attribute | Not the fulfilment node that ships it |
| Node | Physical location that acted: store number or fulfilment centre number | enumerated attribute | Not the channel; a web order can be shipped from a store |

### 3b Time and effective attribution

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| `t_e` event time | When the business event occurred at the origin system | timestamp, UTC | Not the ingest timestamp, and not the promise due time |
| `t_i` ingest time | When the platform first stored the record | timestamp, UTC | Not evidence of when the fact became true |
| `t_v` effective time | Time used for reporting attribution, chosen by policy | timestamp, UTC | Not a physical time; it is a declared convention |
| Late arrival | `t_i - t_e` exceeding the declared window | duration | Not a data-quality error by itself |
| Restatement | A previously reported count changing after a backfill | event | Not a correction of a mistake necessarily |
| Censoring | The outcome is not yet knowable because not enough time has passed | structural absence | Not missing data; nothing is lost |
| Promise snapshot | The commitment in force at the instant of the event | attribute on event | Not the current promise on the order header |
| `promise_due_at` | The committed deadline at the time it was made | timestamp | Not the modelled expected delivery date |

### 3c Promise and fulfilment state

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Promise state | The commitment status of one unit: `committed`, `at_risk`, `fulfilled`, `breached`, `cancelled` | enumerated state | Not the shipment status, which is physical |
| Terminal state | A state after which no further legal transition exists for the unit | state set | Not "the last event we happen to have recorded" |
| Partial fulfilment | Some units of an order reach terminal state and others do not | order-level description | Not a breach if the customer consented to a split |
| Substitution | A different item supplied in place of the ordered item | transition + consent attribute | Not a cancellation and not an error by default |
| Collection | Customer physically takes possession at a node | physical transition | Not "ready for collection", which precedes it by hours |
| Breach | Terminal state reached after `promise_due_at` | derived boolean | Not a complaint; most breaches are never complained about |
| Silent breach | Breach where no customer contact followed | derived boolean | Not a non-breach; the absence of a signal is not a signal |

### 3d Lineage and quality

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Coverage | Share of expected transitions that produced a recorded event | ratio in [0,1] | Not accuracy; a complete but wrong event is still wrong |
| Path completeness | Share of orders observed at every required spine node for their entry mode | ratio in [0,1] | Not coverage; this is per-order, that is per-transition |
| Reconcile key | The identifier two systems agree on for the same real thing | string | Not a foreign key; it may not exist yet and must be built |
| Heartbeat | A per-node periodic event proving the emitter is alive | event | Not a business event; it advances no state |
| Replay | Reconstructing state by re-applying ordered transitions | procedure | Not a backfill; replay does not invent rows |
| Quarantine | Holding records that fail validation without discarding them | store | Not deletion; deletion destroys the audit trail |
| Version pin | Consumers declaring which definition version they consume | attribute | Not a schema version; definitions can change without schemas changing |

### 3e Population and latency

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Matched population | Orders carrying a resolvable loyalty identifier at decision time | subset of orders | Not the customer population; loyal customers order more often |
| Match rate `mu` | Matched orders over all orders in the window | ratio in [0,1] | Not a data-quality score; it is a population exclusion |
| Guest cohort | Orders with no resolvable identifier, usually a quarter of the estate | subset | Not fraud and not anonymous; the parcel still has an address |
| Lag bucket | Interval of `t_i - t_e` used to summarise freshness | classification | Not a percentile; buckets do not order within themselves |
| Tail mass | Share of events beyond the declared freshness target | ratio | Not the failure rate; a late event may still be correct |
| Declared target | The freshness the consumer is entitled to expect, per node class | policy | Not the observed mean, which is a description rather than a promise |

Two distinctions carry most of the operational weight. First, fan-out versus duplication: a genuine three-parcel order produces three `unit` rows and that is correct, while a double-emitting mobile client produces two `unit_collected` rows for one collection and that is a defect. Deduplicating on `order_id` destroys the first case to fix the second, which is why the uniqueness rule must be stated at unit grain with an event-instance identity. Second, censoring versus missingness: an order placed forty minutes ago has no delivery outcome because time has not passed, not because the feed is broken. Treating censored rows as failures biases every recent cohort downward, a defect Section 8.5 discriminates from a real service decline.

A third distinction belongs here because it is the one that breaks loyalty work specifically: matched versus eligible. An order is eligible for a loyalty metric only if the customer was a loyalty member at the time of the order; it is matched only if the identifier resolved. These differ, because membership can exist without resolution (a member paid as a guest) and resolution can exist without membership (a shared device linking two people). A metric computed on matched orders therefore answers "how did we serve orders we could attribute" rather than "how did we serve members", and the two differ by the resolution defect rate in one direction and by guest-checkout behaviour in the other.

## 4. Core Formulas with Full Derivation

Six relations are derived here, each with the five-part anatomy: FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION, then an explicit validity condition. Subsection 4.1 handles a single transition class, 4.2 generalises to a branching spine, 4.3 covers joins and clocks, 4.4 is a mechanism block rather than algebra because promise-state correctness is a state-machine property before it is a rate, and 4.5 and 4.6 close on the two boundary questions that decide which population and which latency a published number actually describes.

### 4.1 Base case: event coverage per transition

Every downstream metric silently assumes that the events it counts are all the events that happened. Coverage makes that assumption explicit and measurable, using a control total that arrives by a different route.

```text
FORMULA [T-01]:  c = R / X
WHERE:  c  coverage, dimensionless in [0,1]
        R  recorded transitions of type T in window W, counted once per transition identity
        X  expected transitions of type T in window W, from an independent control total
CONSTRAINT:  R and X must share the same population, window and node set, and R must be a
             subset of X conceptually; X must not be derived from R or from the same feed.
NUMERICAL EXAMPLE:  412 units collected per the event stream; the store point-of-sale journal
             shows 458 collection receipts in the same window and node set. c = 412/458
             = 0.8996, i.e. 89.96% coverage, a shortfall of 46 transitions.
INTERPRETATION:  roughly one collection in ten is invisible to the event stream. Any rate whose
             numerator counts collections is understated by about that factor unless corrected.
```

Validity condition: `c` is interpretable only while `X` is genuinely independent. If the control total is computed from the same event stream through a different query, `c` measures query consistency, not coverage. A useful sanity test is that `c` should occasionally exceed 1 during a window boundary mismatch — if it never does, suspect that `X` was derived from `R`.

**Why 89.96% is dangerous rather than merely imperfect.** Coverage error is not random noise across a funnel. It concentrates in the node with the weakest emitter, and it concentrates by time of day, because offline store batches flush at shift change. A collection rate computed on the covered rows therefore overweights stores whose scanners are reliable — typically the newer, higher-volume stores — and reads as a service gap in exactly the stores that need investment. The correction is not to multiply by `1/c` blindly, because the uncovered rows are not a random sample; the correction is either to repair the emitter or to stratify by node.

### 4.2 General case: path completeness for a spine with branches

An order's journey is not a fixed-length sequence. It depends on the entry mode, on whether collection or delivery was chosen, and on whether a substitution or a split occurred. Path completeness is the per-order probability that the traversal reached its declared terminal state through legal transitions, and its denominator differs by branch.

```text
FORMULA [T-02]:  p_b = A_b / N_b
WHERE:  b    branch: delivery | collection | substitute_and_collect | delivery_split
        N_b  orders entered on branch b in window W (by t_e of the branch-determining event)
        A_b  those of N_b whose last observed state is the branch terminal state, with every
             intermediate required node present in order
CONSTRAINT:  the node sequence required for each branch is declared in the taxonomy version
             pinned by the consumer; a missing intermediate node makes the order NOT complete,
             even when the terminal state is present.
NUMERICAL EXAMPLE:  delivery branch 1,240 orders, 1,187 complete -> p_del = 0.9573
                    collection branch 610 orders, 561 complete   -> p_col = 0.9197
                    weighted p = (1187+561)/(1240+610) = 1748/1850 = 0.9449
                    unweighted mean of the two rates = (0.9573+0.9197)/2 = 0.9385
INTERPRETATION:  the two averaging rules differ by 0.64 percentage points, and each answers a
             different question. The weighted figure answers "what fraction of orders are fully
             traced"; the unweighted answers "how healthy is a typical branch".
```

Validity condition: branch membership must be determined at entry, not inferred from the events that happened, or the denominator adjusts itself toward completeness. If an order is assigned to the delivery branch because a shipping event exists, then every order missing that event silently leaves the denominator.

**The boundary where T-02 breaks.** If a branch is defined by an event that can itself fail to be emitted, the denominator is conditioned on data quality, and `p_b` rises when quality falls. Consider a retailer where the `substitute_and_collect` branch requires a `unit.substituted` event. A store that fails to emit substitutions routes its real substitutions into the plain collection branch, where they appear as complete. Completeness looks stable or better while the taxonomy is being violated. The fix is a branch assignment from an immutable order attribute — the commercial choice of pickup versus ship — never from an operational event.

### 4.3 Variant: join fan-out and the effective-time asymmetry

Two relations matter when a warehouse joins an event stream to a dimension. The first is the cardinality multiplier, the second is the asymmetry between when an event is true and when the reporting layer attributes it.

```text
FORMULA [T-03]:  m = |J| / |L|
WHERE:  |L|  rows in the left table (events, one per transition)
        |J|  rows in the joined output
        m    mean multiplicity, dimensionless, m >= 1
CONSTRAINT:  the join predicate must be stated with its key set. m > 1 means the right table is
             not unique on the join key; every downstream SUM over left-grained measures is then
             inflated by m on average, and by a per-row multiplier that can be very uneven.
NUMERICAL EXAMPLE:  1,850 collection events joined to a unit dimension carrying 2,310 unit rows,
             of which 460 are split-parent rows duplicated per parcel. If 300 events match a
             parent with 3 children and 1,550 match a unique child, then
             |J| = 300*3 + 1550*1 = 2450, m = 2450/1850 = 1.3243.
             A SUM of collected value of $18.50/unit over events becomes $24.50/event on average.
INTERPRETATION:  the revenue column is inflated by about a third with no error message, and the
             inflation is concentrated on split orders, so the error correlates with basket size.
```

Validity condition: `m` summarises a distribution. It is safe to quote only alongside the concentration — here 16.2% of events carried the excess. A mean multiplicity near 1.01 can still hide a tail where a single supplier record has hundreds of children, and that tail will appear as an outlier that a reviewer mistakes for fraud.

```text
FORMULA [T-03b]:  attribution error  a = |t_v - t_e| distribution
WHERE:  t_v  effective time used by reporting (policy), t_e  event time at origin
CONSTRAINT:  numerator counts must use one clock and the denominator another only if the
             mismatch is stated; a rate whose two clocks differ by more than the reporting
             window cannot be compared across periods with different latency.
NUMERICAL EXAMPLE:  store batch flush at 18:40 covers orders with t_e between 13:00 and 18:30.
             All 214 of them carry t_i ~= 18:40. If reporting uses t_i, the 13:00-17:00 bucket
             shows 0 collections and 18:00-19:00 shows 214. If reporting uses t_e, the same
             records distribute 41, 52, 47, 38, 36 across the five hours.
INTERPRETATION:  using ingest time manufactures a demand spike of 214 against a true peak of 52,
             a factor of 4.1, and manufactures a matching dead zone.
```

Validity condition: event-time reporting requires that `t_e` be trustworthy at the origin. Store point-of-sale clocks drift; a clock six minutes fast turns a 21:58 collection into a 22:04 next-day record and moves it across a period boundary. The mechanical guard is a per-node clock-offset monitor, treated as a data-quality metric with an alarm, not as a background curiosity.

### 4.4 Limits and counterexamples: the promise state machine

Promise correctness is not a ratio first. It is a property of a state machine: whether each transition recorded is legal, whether any required transition is absent, and whether the terminal state is consistent with the timestamps. The block below is the mechanism version of the five-part anatomy.

```text
MECHANISM [T-04]:  promise state machine for one fulfilment unit
PRECONDITIONS:     a unit exists with committed promise_due_at and a declared entry mode
STATE TRANSITION:  committed --allocate--> allocated --pick--> picked --pack--> packed
                   packed --dispatch--> in_transit --deliver--> fulfilled
                   packed --ready--> ready_for_pickup --collect--> fulfilled
                   any live state --cancel--> cancelled
                   any live state --deadline passed--> breached (terminal, absorbent)
INVARIANT:         exactly one state per unit per instant; states only move forward along the
                   declared edges; promise_due_at may be REVISED only by a recorded event that
                   carries the old and new value, never mutated in place.
JUSTIFICATION:     the invariant is what makes the count of terminal states equal the count of
                   units, which is what makes any completion rate's denominator stable. Without
                   it, a unit can sit in two states and be counted twice, or in none and vanish.
WORKED TRACE:      unit U at 09:12 committed due 18:00; 09:40 allocated; 11:02 picked;
                   11:30 packed; 12:05 ready; 18:47 collected. Terminal fulfilled at 18:47 >
                   18:00, so the unit is a breach by 47 minutes even though it was collected.
FAILURE BOUNDARY:  if the collection event is re-emitted at 18:49 (duplicate), the machine must
                   reject the second transition as illegal-from-terminal, not absorb it; a
                   machine that absorbs duplicates reports one collection and one 49-minute
                   breach, and the breach figure drifts with emitter behaviour.
```

The counterexample worth memorising: `breached` as an absorbent terminal state means a unit that is collected late can never later be recorded as `fulfilled`. If a retailer needs both facts — it was delivered and it was late — then the terminal state must be `fulfilled_late` rather than a rewrite of `fulfilled`. Systems that model lateness by overwriting the outcome state lose the ability to compute on-time rate at all, because the numerator and the denominator then describe different cohorts.

### 4.5 Variant: identity match rate and the population it silently removes

A loyalty metric legitimately uses only loyalty-linked orders. A fulfilment metric does not, because the parcel does not know who bought it. The two populations differ by the match rate, and the match rate is almost never quoted.

```text
FORMULA [T-05]:  mu = M / O
WHERE:  O   orders in the window (order grain, after dedupe)
        M   orders carrying a resolvable loyalty identifier at decision time
        mu  match rate, dimensionless in [0,1]
CONSTRAINT:  M must be measured at the same instant as O, using only identifiers that were
             available before the metric window closed. Retrospectively linked orders inflate mu.
NUMERICAL EXAMPLE:  O = 2,400, M = 1,824, so mu = 1,824/2,400 = 0.76 exactly; guest orders
             = 2,400 - 1,824 = 576, i.e. 24.0% of the estate.
             If on-time rate is 0.90 among matched orders and 0.82 among guest orders, the
             whole-population rate is 0.76*0.90 + 0.24*0.82 = 0.6840 + 0.1968 = 0.8808.
             Publishing 0.90 overstates by 1.92 percentage points.
INTERPRETATION:  a 24-point exclusion produces a two-point error only because the two cohorts are
             close. If guest orders were served at 0.61, the population rate would be
             0.76*0.90 + 0.24*0.61 = 0.6840 + 0.1464 = 0.8304, a 7-point overstatement.
```

Validity condition: `mu` must be measured on the order population, not on the customer population. A retailer with 2,400 orders from 1,910 customers has a customer match rate that is higher than its order match rate, because loyal customers order more often. Quoting the customer-level figure for an order-level metric understates the exclusion by exactly the correlation between loyalty status and order frequency.

**Why the correction cannot be a single multiplier.** The guest cohort is not a random sample: it over-represents first-time buyers, marketplace purchases, and gift orders, all of which have different service characteristics. The honest treatment is to report the metric on the loyalty population with `mu` printed beside it, and to report the guest cohort separately at whatever precision its volume supports. A weighted blend is acceptable only when both cohort rates are published.

### 4.6 Boundary: freshness bound and the mean-lag trap

The real-time view is only as fresh as the slowest contributing node. Freshness is best expressed as the distribution of `t_i - t_e`, not as its mean, and the reason is that the distribution is heavy-tailed by construction.

```text
FORMULA [T-06]:  L = mean(t_i - t_e) = sum_k p_k * l_k
WHERE:  p_k  share of events in lag bucket k, sum_k p_k = 1
        l_k  representative lag of bucket k, minutes
CONSTRAINT:  buckets must be exhaustive and non-overlapping, and l_k must be the bucket mean,
             not its lower bound; otherwise L is biased downward by construction.
NUMERICAL EXAMPLE:  within 5 min p=0.71, l=1.2 min; 5 min to 6 h p=0.23, l=90 min;
             beyond 6 h p=0.06, l=400 min.
             L = 0.71*1.2 + 0.23*90 + 0.06*400 = 0.852 + 20.700 + 24.000 = 45.552 min.
             Contribution of the beyond-5-minute mass = (20.7+24.0)/45.552 = 0.9811, i.e. 98.1%.
             Median lag (constructed) = 2.4 min, so L / median = 45.552/2.4 = 19.0x.
INTERPRETATION:  a mean lag of 45.6 minutes describes almost nobody: 71% of events are under five
             minutes old and 6% drag the mean by a factor of nineteen. Any service-level target
             stated on the mean is therefore satisfied by the fast majority while the tail is
             unmanaged.
```

Validity condition: the relation holds for any bucketing, but the conclusion about which bucket dominates depends on the estate. In an estate with no offline stores, the beyond-6-hour bucket may be 0.2% and `L` collapses to a few minutes. That is why the freshness target must be declared per channel and per node class, and the aggregate `L` reported only as a weighted summary with the tail mass beside it.

**Practical reading of the bound.** If the promise metric is published once per day, freshness hardly matters and the tail is a reconciliation task. If it drives a same-day intervention — reassigning a pick, notifying a customer of a late collection — then the tail is the product, and `L` is the wrong statistic entirely. The operative number becomes the 95th-percentile lag per node, because the intervention is triggered by the late cases rather than by the typical one.

Takeaway: the rate formulas (T-01 to T-03) quantify how well the taxonomy is being followed; the state machine (T-04) decides whether the numbers they produce can mean anything; and the two boundary relations (T-05, T-06) say which population and which latency the resulting number actually describes.
## 5. Worked Numerical Example

**ILLUSTRATIVE and VERIFIED arithmetic.** This trace follows one simulated trading day of an unnamed mid-size omnichannel retailer: 2,400 orders across three channels, 61 store nodes, two fulfilment centres. Every count is constructed to make the relations in Section 4 visible. The retail situation is plausible; the values are not measurements and must not be cited as benchmarks.

### 5.1 The day's raw counts, before any taxonomy rule

```text
orders placed                 web 1,520   app 640   store 240   total 2,400
units per order               mean 1.6        total units 3,840
entry mode                    delivery 1,660 orders | collection 740 orders
split orders                  96 orders produced 2 parcels; 12 produced 3 parcels
                             -> delivered parcels = 1,660 - 96 - 12 = 1,552 single
                                                     + 192 (96*2) + 36 (12*3) = 1,780 parcels
                             -> delivered orders 1,660, delivered units 1,780
substitutions                 54 units, 41 of them consented at the moment of pick
returns initiated             118 units; 74 returned to a store, 44 to a carrier
```

Checking the split arithmetic independently: 96 orders at two parcels contributes 192 parcels and 96 orders; 12 orders at three parcels contributes 36 parcels and 12 orders. Non-split delivered orders are therefore 1,660 − 96 − 12 = 1,552. Parcel total 1,552 + 192 + 36 = 1,780, and the parcel-to-order ratio is 1,780/1,660 = 1.0723. That ratio is the multiplier any order-grain metric must not accidentally apply to a unit-grain measure. A per-parcel cost of $6.40 gives $11,392 per day at parcel grain; applying it at order grain gives $10,624, an understatement of $768 (6.7%).

### 5.2 Derivation 1 — collection coverage against a control total

The store point-of-sale journal is the independent control: 740 collection orders, of which 703 were marked collected in the journal. The event stream reports 651 `unit.collected` events after dedupe on transition identity.

```text
FORMULA [T-01]:  c = R / X = 651 / 703
                651 / 703 = 0.92589...
INTERPRETATION:  coverage 92.59%, shortfall 52 collections (7.41%).
                 If the published collection rate is 0.71 (651/917 attempted pickups),
                the coverage-corrected rate is 0.71/0.9259 = 0.7668, or 76.68%.
                The correction assumes the 52 missing rows behave like the 651 observed ones.
```

Independent re-derivation of the corrected rate by a different route: 703 journal collections over 917 attempted pickups is 703/917 = 0.7666, agreeing to 0.02 percentage points with 0.7668. The tiny gap comes from rounding 651/703 to five decimals; carrying the fraction through gives exact agreement. That agreement is the point of the second route: it verifies the correction algebra without reusing the same numbers.

The assumption is the weak part. The 52 unrecorded collections are not a random sample; they cluster in the 14 nodes that still run the older client. If those nodes serve 18% of pickups and their true collection rate is 0.78 rather than 0.71, the corrected national rate is 0.82×0.71 + 0.18×0.78 = 0.5822 + 0.1404 = 0.7226, i.e. 72.26%, not 76.68%. The gap between 72.26% and 76.68% — 4.4 percentage points — is the cost of ignoring non-random coverage.

### 5.3 Derivation 2 — promise accuracy at unit grain with revised commitments

Promise accuracy is the share of delivered units that reached terminal state at or before the commitment in force at the moment of delivery. Nineteen orders had their promise revised during the day, and the revision is carried as an event with old and new values.

```text
FORMULA [T-02 variant]:  q = F / D
WHERE  F  units with terminal_ts <= promise_due_at (snapshot valid at terminal_ts)
       D  units that reached a terminal state in the window
NUMERICAL EXAMPLE:
       delivered units reaching terminal     1,780
       late against stated commitment         214
       F = 1,780 - 214 = 1,566
       q = 1,566 / 1,780 = 0.87978...
       Naive order-grain version: 1,660 - 163 = 1,497 orders on time; 1,497/1,660 = 0.90181
INTERPRETATION:  unit-grain promise accuracy 87.98%, order-grain 90.18%, a 2.2pp gap that is
       entirely an artefact of grain, because the 51 extra late parcels belong to orders whose
       other parcels arrived on time and which therefore count as on time at order grain.
```

Independent check on the gap: the difference in late counts is 214 − 163 = 51, and 51/1,780 = 0.02865, while the gap in rates is 0.90181 − 0.87978 = 0.02203. Those do not match, which is correct and instructive: the denominators differ (1,780 units versus 1,660 orders), so the arithmetic must be done on the rates rather than on the counts. Verifying the order-grain rate independently: 163/1,660 = 0.09819 late, so 0.90181 on time, consistent.

Sensitivity of the unit-grain figure to the 19 revised promises. If the revision is applied retrospectively to the original commitment instead of the snapshot, all 19 orders with a lengthened commitment become late: add the units in those orders, say 31 units, giving (1,566 − 31)/1,780 = 1,535/1,780 = 0.86236, i.e. 86.24%. A 1.7 percentage point swing from a single modelling choice about what "the promise" means. That is why the snapshot rule is in the invariant, not in a footnote.

### 5.4 Derivation 3 — path completeness by branch and the averaging trap

```text
Branch assignment (from the immutable commercial choice at checkout):
  delivery    1,660 orders   1,571 complete   p_del = 1,571/1,660 = 0.94639
  collection    740 orders     681 complete   p_col =   681/740 = 0.92027
Weighted:  (1,571+681)/2,400 = 2,252/2,400 = 0.93833
Unweighted mean: (0.94639 + 0.92027)/2 = 0.93333
Difference: 0.50pp. The weighted figure is the one to publish, because the business is paid per
  order and the branch mix is a fact about the day rather than a design choice.
```

Now the substitution effect. Of the 54 substituted units, 41 were consented before dispatch and 13 were substituted at the door, which is a different commercial situation. If substituted units are excluded from path completeness, the collection denominator falls to 740 − 54 = 686 and completeness becomes (681 − 13)/686 = 668/686 = 0.97376, a 5.3pp improvement produced entirely by deleting the hard cases. That is the most common way a path-completeness figure is improved without any operational change, and it is why the definition must state that substituted units stay in the denominator.

Counter-check that the exclusion is real: 681 − 13 = 668 complete among 686 non-substituted collections; the 13 unconsented substitutions are precisely the incomplete non-consented rows. If instead all 54 substitutions were counted incomplete, completeness would be (681 − 54 + 41)/740 = 668/740 = 0.90270, an underestimate, because consented substitutions are complete by definition. The correct figure counts the 41 as complete and the 13 as incomplete, which is the 681 already used: 681 includes those 41. The three numbers 0.9738, 0.9203 and 0.9027 differ only in how the substitution class is handled, and a reader given one without the rule cannot tell which.

### 5.5 Derivation 4 — the cost of duplicate emission

After the mobile client release, 88 of the 651 collection events are duplicate pairs: same unit, same transition, `t_i` differing by 4 to 90 seconds.

```text
FORMULA [T-01 with duplication]:  observed = true * (1 + d) where d is duplicates per true event
NUMERICAL EXAMPLE:  true 651, duplicates 88, observed 739.
        apparent collection rate = 739/917 = 0.8059, against true 0.710.
        error = +9.6pp on a base of 71.0pp, i.e. +13.5% relative.
INTERPRETATION:  duplicate emission alone produces a service improvement of ten points with no
        change in operations, which is why a taxonomy needs a uniqueness rule before it needs a
        dashboard.
```

The detection rule that discriminates duplication from a genuine second collection: a genuine second collection of the same unit is impossible, so any pair sharing `unit_id` and transition type within a 24-hour window is duplicate by construction. That is a stronger test than a time threshold on `t_i`, and it does not depend on the client fixing anything. The residual risk is a unit identifier reused after a return and re-ship, which Section 8.4 handles by including a fulfilment attempt counter in the transition identity.

Takeaway from Section 5: four different rates — coverage, promise accuracy, path completeness, and a duplicated collection rate — move in the same direction when data quality degrades, and all four look like operational improvement. Only the state-machine invariant and the control total distinguish them.

## 6. Visual / Geometric Intuition

### 6.1 The population rectangle

Draw two rectangles. The outer one is every order placed in the window. Inside it, the eligible set — orders that entered the branch being measured — and inside that, the numerator. Changing the eligibility border moves numerator and denominator together. Changing the numerator definition moves only the numerator. Every argument about a metric is one of those two moves, and naming which one is happening ends most of them in a sentence.

```text
 +----------------------- all orders placed 2,400 ------------------------+
 |                                                                       |
 |   +------------- delivery branch 1,660 -----------+  +-- coll 740 --+ |
 |   |                                             |  |               |  |
 |   |   +---- complete 1,571 ----+                |  | +- 681 -----+ |  |
 |   |   |                        |  incomplete 89 |  | |           | |  |
 |   |   +------------------------+                |  | +-----------+ |  |
 |   +---------------------------------------------+  +---------------+ |
 +-----------------------------------------------------------------------+
```
The area ratio of an inner box to its parent is the rate. A rate computed against the outer rectangle — complete orders over all orders placed — is a different quantity from one computed against the branch box, and both are legitimate: the first is "what fraction of my customers got a fully traced order", the second is "how well did the delivery branch work". Publishing one and labelling it the other is the most common reporting defect in omnichannel dashboards.

### 6.2 The two-clock ribbon

Time runs left to right. Each event is a point on the event axis and a point on the ingest axis; the horizontal gap is latency, the vertical gap is out-of-order arrival.

```text
 t_e axis  --------o-------o-------------o------o------------o------>   (true order)
                   |       |             |      |            |
 latency           | 2min  | 6h  (batch) | 40s  | 3min       | 11min
                   v       v             v      v            v
 t_i axis  ----------o---------o--------------o-----o-------------o-->  (arrival order)
                            ^
                    invert: arrived after a later event
```
The ribbon shows why a streaming aggregate over ingest order can emit a state sequence that no unit ever occupied: `packed` before `picked`, or a collection before its own readiness. A state machine fed in arrival order will either reject legitimate transitions or accept illegal ones, and both failure modes are invisible in the count. The fix is to key by unit, hold a bounded state per unit, and apply transitions in `t_e` order with an explicit allowed-lateness horizon. Section 9 costs that buffer.

### 6.3 The fan-out tree and where money inflates

```text
     event row (1)                    dimension rows (2,310)
     +----------+                     +-- parcel A --+
     | collected|----join on order --->| parcel B     |   one event, three output rows
     +----------+                     +-- parcel C --+
```
The visual point is that the tree is asymmetric: 96 orders have two children and 12 have three, so the average inflation is 1.32 but a single 12-parcel order inflates by 3. A sum over the joined output is not a sum over events scaled by a constant; it is a sum whose error is a function of basket composition. When a revenue-per-collection figure rises after a change to split-shipment policy, this tree is the first place to look.

### 6.4 The promise-state lattice

```text
   committed --> allocated --> picked --> packed --> in_transit --> fulfilled
        |            |           |         |            |
        |            |           |         +--> ready_for_pickup --> fulfilled
        |            |           |         |
        +------------+-----------+---------+--> cancelled   (terminal)
                     |
                     +--> breached  (terminal, absorbent, entered by clock not by event)
```
The lattice makes one thing obvious that prose hides: `breached` is entered by the passage of time, not by a message. Nothing emits a breach event in most estates, so a breach denominator must be computed by evaluating the terminal timestamp against the snapshot. A team that waits for a breach event will find that breach count is zero and conclude that service is perfect. The visual also shows that `cancelled` can be reached from any live state, which is why a cancellation rate computed only from `allocated` understates by the volume cancelled before allocation — a stage that in some estates is over half of all cancellations.

## 7. Comparison / Selection Table

The table compares four candidate designs for the same problem: how to represent an order's journey so that funnel, promise, and returns metrics all remain computable. They are ordered from cheapest to most faithful, and the last column states the condition under which the design should be refused.

| Design | What it stores | Metrics it supports | Cost / failure signature | Refuse when |
|---|---|---|---|---|
| D1 Status column on order header | one mutable `current_state` per order, overwritten | latest-state counts, coarse funnel | Cheapest. History destroyed; every revision rewrites the past. Funnel needs `state_changed_at` guesses | any promise, latency, or funnel metric is required |
| D2 Event log, no version, no transition identity | append-only rows with `order_id`, `event_name`, `ts` | funnel with dedupe by hand, ad-hoc | Low cost, high analyst cost. Duplicates indistinguishable from real repeat events; renames break history | two producers emit the same semantic event |
| D3 Versioned event log with transition identity | append-only, `(unit_id, transition, attempt)` unique, definition version pinned | funnel, promise, coverage, path completeness | Moderate. Requires a producer contract and a replay path | the state machine has branches requiring consent attributes absent from the log |
| D4 D3 plus promise snapshots and heartbeats | as D3 plus `promise_due_at` per event and per-node heartbeat | all of the above plus breach, silent breach, node health | Highest write volume, roughly 1.2× D3 rows; needs retention policy | node count is so large that heartbeat volume dominates storage without aggregation |

Reading the table as a decision rule rather than a ranking: choose the lowest-numbered design that supports the metric you must publish, and no lower. D1 is not obsolete — a store-ops console that only needs "how many orders are waiting right now" is correctly served by a status column, and rebuilding it as an event log adds cost with no decision benefit. The failure begins when a D1 store is asked a promise question, because the answer requires a state that no longer exists.

Two comparisons deserve separate treatment because they are frequently confused with the main axis. First, **push versus pull**: D3 with producers pushing events gives lower latency and a producer-side schema contract; D3 with a consumer polling source tables costs less to build and silently loses events when a source table is truncated during maintenance. Polling is acceptable for dimension tables and unacceptable for transitions, because a truncated transition table cannot be reconstructed. Second, **single log versus per-domain logs**: one log with a versioned definition is easier to replay and harder to govern, since every producer must be admitted by one gate; per-domain logs are easier to govern and require a reconciliation layer that duplicates the dedupe logic in every consumer. For estates above roughly ten producers the per-domain pattern with a thin canonical projection tends to win, because the governance cost of a single gate grows faster than the reconciliation cost.

| Selection question | D1 | D2 | D3 | D4 |
|---|---|---|---|---|
| Can I compute coverage (T-01)? | no — no transition identity | no — duplicates merge with real events | yes | yes |
| Can I compute promise accuracy with revisions? | no — snapshot lost | no | partially — current snapshot only | yes |
| Can I detect a dead emitter? | no | no | no | yes, via heartbeat |
| Can a new channel be added without renaming history? | only if channel is a field | only if channel is a field | yes, version-pinned | yes, version-pinned |
| Replay cost for one day (Section 9 sizing) | trivial | 1× | ~1.05× | ~1.25× |

Takeaway: select the design that supports the decision you must defend, and be explicit that every design below it cannot answer the questions above it — not that it answers them approximately.
## 8. Failure Modes & Edge Cases

Each entry states the symptom an analyst sees, the competing causes that produce that identical symptom, the query that separates them, the correction, and the residual risk the correction introduces. A failure catalogue that lists causes without a discriminating check is a list of fears.

### 8.1 Duplicate emission across a mobile release

**Symptom.** Collection volume steps up by about thirteen per cent on the day a client release ships, with no change in store traffic or pickup bookings. The step is sharp, same-day, and persists.

**Competing causes.** (a) The new client emits both the legacy `order_collected` and the new `unit.collected` for one physical collection. (b) A marketing push genuinely increased pickups. (c) The control total itself changed because the journal now includes a new store.

**Discriminating check.** Group by `client_version` and count transitions per `unit_id` within 24 hours. If the mode of that count is 2 only for the new version, cause (a). Cross-check against store footfall: a genuine pickup increase moves with bookings, which are recorded at `ready_for_pickup` time and are unaffected by the collection emitter. A control-total change shows as a step in node count, which is directly countable.

**Correction.** Deduplicate on `(unit_id, transition, attempt)` rather than on `order_id`, so genuine multi-parcel orders survive. Do not subtract the step as an "adjustment" — the duplicates are still present in the history and every future query will rediscover them.

**Residual risk.** If `unit_id` is generated per emission rather than per physical unit, the dedupe key cannot work, and the taxonomy defect is upstream in identifier assignment. Detect it by checking that `unit_id` appears in the allocation event and the collection event with the same value.

### 8.2 Absence misread as failure

**Symptom.** A store shows a collection rate of 0.62 while its neighbours sit at 0.78 to 0.81. The store's manager disputes the number and has receipts.

**Competing causes.** (a) The store's scanner was offline and the journal was later manually reconciled, so events exist only in the journal. (b) Orders were genuinely collected late or not at all. (c) The store's orders were mostly substituted, which routes them through a branch whose terminal event the old client does not emit. (d) The store clock is offset, pushing `t_e` outside the reported window.

**Discriminating check.** Three counts on the same node and window: events per the stream, receipts per the journal, and heartbeats received. Heartbeats present with low event volume indicates (a) or (c); heartbeats absent indicates the node was dark, which is a plumbing fault not a service fault. A clock offset shows as an implausible modal collection hour and a `t_e > t_i` fraction above zero. Cause (c) is confirmed by counting `unit.substituted` versus the branch terminal event for that node: if substitutions exist and terminal events do not, the client is behind.

**Correction.** Restore the node from the journal with a documented source priority, then re-run the rate with the node flagged. Never impute a collection; a receipt is evidence, an imputation is not.

**Residual risk.** Manual journal entry is keyed by an associate and may use the order barcode rather than the unit barcode, which reconstructs the event at order grain. The reconstructed rows must be marked as order-grain so a unit-grain promise computation does not silently treat them as complete.

### 8.3 Denominator drift from a policy change

**Symptom.** Promise accuracy moves from 0.88 to 0.91 over one month with no operational change and no emitter change.

**Competing causes.** (a) The eligibility rule changed: an order-exclusion for pre-orders stopped applying, so a well-served cohort entered the population. (b) Carrier performance genuinely improved. (c) The promise window was lengthened in the checkout configuration, which raises on-time rate without any speed change. (d) Censoring: recent cohorts with long promises are excluded because they have no terminal state yet, biasing the sample toward fast orders.

**Discriminating check.** Freeze the eligibility SQL to the previous version and re-run on the new data. If the movement survives, it is (b) or (c). Then compare the mean promise length in days before and after: a step in the mean with constant actual transit time is (c). Censoring is detected by plotting the rate against order age and checking whether the newest bucket is computed on a smaller denominator than the older ones at the same age.

**Correction.** Publish the rate against a frozen cohort definition, and publish the definition's version beside the number. If the promise window changed commercially, report the change as a level shift rather than allowing it to be read as improvement.

**Residual risk.** Freezing the definition hides a real change in the business's question. The mitigation is a paired series — frozen and current — so the reviewer can see both what happened to service and what happened to the definition.

### 8.4 Cross-channel return matched by a lossy key

**Symptom.** Return rate for online orders shows 6.1%, and the finance ledger shows returned goods value consistent with 8.4%. The gap is roughly one and a half points, stable for months.

**Competing causes.** (a) In-store returns of online orders are not matched because the store system keys on its own receipt number. (b) Some returns are recorded as exchanges, a transition the taxonomy maps to `return_received` only after the replacement ships, so recent exchanges are censored. (c) The ledger includes damaged-goods write-offs that are not customer returns at all.

**Discriminating check.** Match candidate pairs on (item, store, day, tender) and count the unmatched store-side receipts. If their value approximates the ledger gap, (a). Compare the gap's timing to the exchange lag distribution for (b): a censoring effect closes as the window matures, while a key defect does not. Cause (c) is separable by ledger account code.

**Correction.** Build a reconcile key at the point of return: the store application should capture the original order reference from the customer's barcode, and if it cannot, the pair `(loyalty_id, item, return_day)` with a documented ambiguity flag. Accept ambiguity explicitly rather than assuming one-to-one.

**Residual risk.** Any match constructed on `(loyalty_id, item, day)` will double-match a customer who returns two of the same item on one day. The fan-out relation T-03 quantifies the inflation; the mitigation is to match at most one-to-one and route the remainder to a manual reconciliation queue with its volume reported.

### 8.5 Censoring read as performance collapse

**Symptom.** The seven most recent days show promise accuracy falling monotonically, from 0.88 to 0.61 on the most recent day. Operations is asked to explain.

**Competing causes.** (a) Genuine deterioration on recent days. (b) Censoring: recent orders have not had time to reach terminal state, and the ones that have are disproportionately the fast ones. (c) A feed outage means recent terminal events have not been ingested, so the denominator is full and the numerator is not. (d) The window boundary places in-transit orders with long promises into the recent bucket at a higher rate.

**Discriminating check.** Recompute with a maturity window: exclude orders whose age is below the 95th percentile of observed time-to-terminal, then look again. Censoring produces a curve that flattens to the old level when the window is applied; genuine deterioration does not. Then compare ingest lag on terminal events: a feed outage shows a collapse in `t_i` volume for terminal transitions, not in their `t_e` distribution.

**Correction.** Report against a maturity window and publish the window length with the number. For operational alerting, use a cohort view keyed by order age rather than by calendar day — the age view is immune to censoring by construction.

**Residual risk.** A maturity window makes the metric lag reality by its own length, so a genuine defect is detected late. Pair it with a leading indicator that is not censored — for example the rate of units still unallocated at age 24 hours — and alert on that instead.

### 8.6 Substitution treated as an error by default

**Symptom.** A store's fulfilment quality score is the worst in the estate, driven by a substitution rate of 9.8% against an estate median of 2.1%, while its on-time rate is the best in the estate.

**Competing causes.** (a) The store substitutes aggressively because its own stock positions are wrong. (b) The store's catch-weight or size-variant catalogue makes substitution normal and necessary. (c) Substitutions are being recorded where a consented equivalent should be recorded as a distinct, acceptable outcome. (d) The estate median is computed only on stores whose clients emit substitution events at all, and most do not.

**Discriminating check.** Compare substitution rate with stock-accuracy audits for the same store. Cause (a) shows a negative correlation between recorded stock and physical stock. Cause (b) shows substitution concentrated in a few variant families, not across the assortment. Cause (c) is confirmed by the consent attribute: consented substitutions should not be counted as defects, and if the store's consented share is 0.86 while the score treats all substitutions as defects, the score is wrong. Cause (d) is confirmed by counting emitters: if 40 of 61 stores emit no substitution events, the median is not a median of the estate.

**Correction.** Split the measure into consented substitution rate (a service feature) and unconsented substitution rate (a defect), and compute both against a denominator of units picked. Exclude non-emitting stores from any comparative ranking until their clients catch up, and report the excluded share.

**Residual risk.** Consent captured at pick time is a proxy for consent at delivery; a customer may accept on screen and refuse at the door. That second refusal arrives as a return, which is a different metric, so the two must not be netted against each other into a single "satisfaction" number.

### 8.7 Heartbeat absence and the silent node

**Symptom.** A node contributes zero events for eleven hours, then a normal-looking day. Nobody noticed during the eleven hours, and the week's aggregate looks fine.

**Competing causes.** (a) The node was closed (planned maintenance, holiday). (b) The emitter was dark and the backlog flushed on recovery, so events exist with `t_e` inside the gap but `t_i` after it — no data is lost, but the real-time view was blind. (c) The node's traffic was genuinely zero because of a local disruption. (d) Event validation rejected all rows for a required field, and they sit in quarantine.

**Discriminating check.** Compare heartbeat presence with event presence. Heartbeats present plus zero events is (c) or (a), distinguished by the store's opening-hours calendar. Heartbeats absent is (b) or (d). Then count quarantined rows for the node in the window: (d) shows a quarantine spike whose row count matches the missing event estimate.

**Correction.** Add a node heartbeat with a 15-minute cadence, alert on absence, and always reconcile a gap by comparing `t_e` distribution before and after recovery rather than by comparing `t_i`. Backlog flush is visible as a fat `t_e` histogram in the gap and a fat `t_i` spike at recovery.

**Residual risk.** Heartbeats add roughly 96 events per node per day — for 61 nodes, 5,856 rows daily, more than double the business event volume at collection grain in a small estate. Aggregate them before retention: keep hourly rollups indefinitely, raw heartbeats for 30 days, and alert on the rollup gap.

### 8.8 Version pinning skipped, and the rename that followed

**Symptom.** A funnel that has been stable for a year shows a step change in a stage that no team changed. The step is a fall, roughly in proportion to a platform migration's share of traffic.

**Competing causes.** (a) A producer renamed an event (`unit_shipped` to `unit.dispatched`) and consumers silently matched nothing. (b) Consumers matched both names and double-counted, producing a rise rather than a fall — the opposite signature. (c) The stage was redefined in a dashboard revision. (d) An upstream join lost rows because the dimension table's key was retyped from string to integer with leading zeros dropped.

**Discriminating check.** Count distinct `event_name` values per day. A rename shows one name disappearing and another appearing with complementary volume. A key retype shows join match rate falling, computed as matched rows over left rows, while the left row count is stable.

**Correction.** Consumers must declare a definition version and the build must fail when an unreferenced event name appears at above a threshold share. Renames are additive: emit the new name, keep the old for the deprecation window, and record the mapping in the taxonomy version that introduces the new name.

**Residual risk.** Dual emission for a deprecation window creates a duplication window, so the dedupe rule must treat the alias pair as one transition. That alias table becomes a permanent artefact and needs an owner, or the next rename duplicates the previous mapping.

### 8.9 Boundary and edge conditions worth testing explicitly

- **Leap-day and DST transitions.** A store in a daylight-saving region loses or gains an hour; an hourly bucket with local-time keys will show 23 or 25 buckets. Use UTC for storage and local time only for display, and test the collection rate across the transition day. A promise measured in "same day" is ambiguous on that day and must be defined in absolute hours.
- **Order placed at 23:59, collected at 00:01.** Two calendar days, two periods, one order. Any "same-day collection" metric must use elapsed hours from `order_placed`, not calendar date equality, or it will report a boundary artefact every night.
- **Cancelled before allocation.** No unit exists, so a unit-grain promise denominator excludes it, while an order-grain denominator includes it as cancelled-before-promise. State which; the difference can reach double digits in categories with stock volatility.
- **Return before collection.** Happens with marketplace resale; a `return_initiated` may legally precede `collected` for a unit that was never physically handed over. The state machine must allow it as a cancellation path rather than rejecting the transition.
- **Re-ship after return.** The same `order_id` and possibly the same `unit_id` carry two fulfilment attempts. Without an `attempt` counter in the transition identity, the second attempt's collection is deduplicated against the first and disappears.

Takeaway for Section 8: every symptom above is compatible with at least two causes, and the discriminating check is always a second, independent count over the same window — never a second look at the same aggregate.
## 9. Complexity & Cost Analysis

Cost in this domain is dominated by three quantities that are easy to mismeasure: the dedupe sort, the out-of-order state buffer, and the heartbeat volume. Each is computed below rather than asserted, and each has a stated regime in which a cheaper structure is adequate.

### 9.1 Dedupe and transition application

Applying a transition log idempotently requires knowing, per `(unit_id, transition, attempt)`, whether the transition has already been applied. Two structures are viable.

```text
FORMULA [C-01]:  T_sort = k * N * log2(N)          T_hash = k * N
WHERE:  N     number of event rows applied in one run
        k     constant per comparison or per hash probe (implementation-specific, not measured here)
CONSTRAINT:  the hash structure must fit in memory; the sort does not. Break-even is therefore
             set by memory, not by the asymptote.
NUMERICAL EXAMPLE:  N = 3,840,000 rows/day (a 2,400-order day at 1,600x is not the case; this is
             a 1,600-order-per-hour estate over 24 hours). With 16 bytes per key and load factor
             0.5, the key set costs 32 B/row * 3.84e6 = 122.9 MB, plus value slots ~48 B/row
             = 184 MB, total about 307 MB. That fits one process; a month of rows (115e6) would
             need about 9.2 GB and no longer fits a single worker.
INTERPRETATION:  for a daily batch the hash path is fine and simpler; for a month-scale replay the
             sort path or a partitioned hash is required, at a cost of log2(3.84e6) = 21.9x
             comparisons per row relative to the linear path in the comparison model.
```

Validity condition: the comparison-model factor `log2(N)` describes sorting, not a specific engine. A columnar engine with dictionary-encoded keys can group far faster than the asymptote suggests, and a streaming engine with pre-sorted input is linear. Quote the factor, never a wall-clock time, unless you have measured on your own estate.

**The cheaper regime.** Below roughly one million transitions per run, keep the state in a single process and skip partitioning entirely. The engineering cost of distributed dedupe exceeds the compute saving until the key set stops fitting, and a distributed dedupe introduces its own failure mode: partial application on a worker crash, which requires the run to be idempotent anyway — at which point per-row idempotence has already been solved and the distribution is optional.

### 9.2 Out-of-order state buffer

To apply transitions in `t_e` order with a bounded horizon `H`, the processor must hold every unit whose state is awaiting a possibly-later event.

```text
FORMULA [C-02]:  B = r_late * H * u
WHERE:  r_late  late-arrival rate, events per second arriving with t_e older than the current
                watermark (dimensionless share times arrival rate)
        H       horizon in seconds (declared policy, e.g. 21,600 s for a six-hour store batch)
        u       bytes per retained unit state (id, current state, pending transitions, timestamps)
NUMERICAL EXAMPLE:  arrival 44.4 events/s (3.84e6/day), late share 0.06, H = 21,600 s,
                u = 480 B.
                B = 44.4 * 0.06 * 21,600 * 480 = 27,617,280 B ~ 26.3 MiB.
                With H doubled to 43,200 s: 52.7 MiB. Linear in H, as expected.
INTERPRETATION:  a six-hour horizon costs single-digit megabytes for this estate, which is why the
                horizon can be chosen by policy (long enough to cover the worst batch) rather than
                by memory pressure. The constraint that bites is not memory but the latency of the
                real-time view, which is H behind by construction.
```

Validity condition: `r_late * H` must be below the number of distinct units in the horizon, or the model double-counts. In a small estate with 740 collections and a six-hour horizon, the late arrivals are dominated by a handful of nodes, so the aggregate rate hides concentration; compute `B` per node, not estate-wide, and size for the worst node. That node's buffer is the one that blocks the watermark, and one blocked node stalls every consumer downstream of the merge.

**The cheaper regime.** For nodes that batch daily and never stream, route them to a separate daily path and exclude them from the streaming watermark. The streaming view then covers only real-time nodes and stays minutes behind; the batch nodes appear in the reconciled daily view. Mixing the two in one pipeline forces the horizon to the slowest node, which makes the real-time view useless for all of them.

### 9.3 Heartbeat volume and retained rollups

```text
FORMULA [C-03]:  V_hb = n_nodes * (86,400 / p)          (events per day)
WHERE:  n_nodes   emitting nodes
        p         heartbeat period in seconds
NUMERICAL EXAMPLE:  61 nodes, p = 900 s -> 61 * 96 = 5,856 events/day.
                At 120 B/row raw, that is 702,720 B/day ~ 0.67 MiB/day, 245 MiB/year.
                Hourly rollups at 24 rows/node/day: 61 * 24 = 1,464 rows/day, 176 KB/day at
                120 B, 64 MiB/year for ten years.
INTERPRETATION:  raw heartbeats at 15-minute cadence are cheap in bytes and expensive in row
                count: they exceed the daily collection-event count (651) by about nine times,
                which matters for any per-row cost, quota, or scan-priced query engine.
```

Validity condition: the rollup must retain the fields the alerting rule needs — last-seen timestamp, count of events in the interval, and client version. A rollup that keeps only a count cannot distinguish "alive and idle" from "alive and broken", which is the whole point of the heartbeat. Keep three fields minimum; drop the rest.

### 9.4 Replay and restatement cost

```text
FORMULA [C-04]:  C_replay = N * (1 + m_join) * c_row + s * N
WHERE:  N       rows replayed
        m_join  mean joined rows per event (Section 4.3)
        c_row   per-row compute cost
        s       per-row storage write cost for the restated period
NUMERICAL EXAMPLE:  90 days at 3.84e6 events/day = 345.6e6 rows; m_join = 1.32;
                (1+m_join) = 2.32 -> 802e6 joined-row operations. If the engine sustains 50e6
                operations/minute, that is 16.0 minutes of pure join work, before shuffle and
                skew, so plan on a multiple of it rather than the figure itself.
INTERPRETATION:  a nine-month replay is a batch job measured in hours in practice, and the
                practical constraint is that it must be idempotent, because it will be re-run.
```

Validity condition: replay is only meaningful if the definition version is pinned per row. Replaying 90 days under today's definition produces a series that is internally consistent and historically false; the series is legitimate only if labelled as restated-on-definition-vN.

### 9.5 Human cost of the taxonomy itself

The scheme has a staffing cost that is routinely omitted from the business case: the producer contract must be reviewed, the alias table owned, the control totals reconciled, and the quarantine drained. A defensible estimate for a 60-node estate is a quarter of one analyst's time for the first two quarters, then roughly two days per month for maintenance, plus the store-side work to bring 14 legacy clients current. The store-side work is the item that actually delays the programme, because it competes with trading priorities; sequencing it before the metrics are promised is the difference between a two-quarter delivery and an open-ended one.

The cost that is not worth paying: a bespoke taxonomy service with its own API, its own UI, and its own permission model, built to serve one dashboard. That pattern appears because the taxonomy is politically contentious, and the service becomes the venue for the argument. The cheaper resolution is a written definition with a named owner and a diffable version, reviewed in the same place the schema changes are reviewed.

Takeaway: the arithmetic in this section is dominated by row count and horizon, not by cleverness; choose the simplest structure whose key set fits, and make the run idempotent so that every later decision is reversible.

## 10. Key Theorems / Results Reference

The results below are stated as invariants and dominance relations, because each can be checked by arithmetic on a single example rather than requiring an experiment. Each carries the condition that makes it true and the observation that would falsify it.

### R1 — Conservation of transitions

If every event instance is the unique witness of one legal transition, then for any unit and any window, `sum over states of (entries) = sum over states of (exits) + 1` for a unit that started in `committed` and is now terminal. Equivalently, the count of terminal states equals the count of units that entered the machine minus those still live.

Falsified by: any unit with two terminal states, or a unit whose state count is zero. Both indicate duplicate emission or a state that was overwritten. This is the cheapest integrity check in the sheet: one query, one number, and it catches the majority of taxonomy violations.

### R2 — Grain dominance for on-time rate

For any population where some orders contain a late unit and an on-time unit, order-grain on-time rate is greater than or equal to unit-grain on-time rate, with equality only when lateness is perfectly correlated within orders. The Section 5.3 trace shows the gap at 2.2 percentage points with 1.0723 parcels per order; the gap grows with both the split rate and the independence of lateness across parcels.

Falsified by: a case where the order-grain rate is below the unit-grain rate, which can only occur if orders are weighted differently between the two computations — usually because the order-grain computation excludes cancelled orders that the unit-grain one includes. That is not a falsification of the relation but of the claim that the two use the same population.

### R3 — Coverage correction bias

Applying `1/c` to a rate computed on covered rows is unbiased only if coverage is independent of the outcome. When coverage and outcome are correlated — as they are whenever the unreliable emitter is also the poor performer — the corrected rate is biased toward the well-covered nodes. In the Section 5.2 example, the correlated case gives 72.26% against a naive correction of 76.68%; the bias is 4.4 points and it points in the direction of flattering the estate.

Falsified by: a demonstration that covered and uncovered subsets have equal outcome rates. That is testable where a control total gives the outcome for both subsets — journal receipts give the outcome for both, so the test is available, and it usually fails.

### R4 — Fan-out inflation is convex in the parcel distribution

For a sum over a joined output where the left table is events and the right is units, the ratio of the joined sum to the true sum equals the mean parcel count per matched order, weighted by event volume. Because the multiplier is a weighted mean, it is dominated by the tail: a single 12-parcel order contributes 3x to its own event and raises the aggregate more than eleven 2-parcel orders of the same value.

Falsified by: a distribution where every order has exactly one parcel, in which case the multiplier is 1 and the join is safe. State this condition explicitly rather than relying on it.

### R5 — Promise-snapshot sufficiency

Promise accuracy computed against per-event snapshots equals the accuracy computed against a full revision history, provided that every revision is recorded as an event carrying both old and new values, and that the evaluation uses the snapshot valid at the terminal timestamp. Storing only the current value destroys sufficiency: any order revised after its terminal event is misclassified, and the Section 5.3 sensitivity shows a 1.7-point swing from that single defect.

Falsified by: a revision path that mutates the header without an event — for example a customer-service agent editing the promise in an admin screen with no audit row. That path is common, which is why the invariant is stated at write time rather than at read time.

### R6 — Heartbeat non-substitution

Heartbeat presence proves the emitter is alive and proves nothing about business events; event presence proves something happened and proves nothing about coverage of what did not. Neither can substitute for the other, and a monitoring design that alerts only on one will miss the failure the other detects. The Section 8.7 matrix has four cells — heartbeat present or absent crossed with events present or absent — and each cell maps to a different owner.

Falsified by: a node whose emitter is alive and reports heartbeats while its business-event validation rejects every row. That is cell (heartbeat yes, events no, quarantine non-empty), and it is the cell most often misdiagnosed as "quiet day".

### R7 — Replay determinism requires total order per key

A replay reconstructs the true state only if transitions for each unit are applied in a deterministic order. Ordering by `t_e` is deterministic; ordering by `t_i` is not, because ingest order depends on network timing. Ties in `t_e` must be broken by a stable secondary key, typically the transition identity, or two replays of the same data produce different states — which is observed as a metric that changes when the job is re-run unchanged.

Falsified by: two replays of identical input producing different counts. That outcome is not a hypothesis; it is a defect with a specific repair, and it should be a scheduled test rather than an incident finding.

Takeaway for Section 10: these seven statements are what make the taxonomy auditable. Each has a one-query falsification test, and the tests are cheap enough to run on every release.
## 11. Interview / Exam Questions

Fourteen questions, ordered from recall through calculation to diagnosis and judgement. Each has its own answer; several require arithmetic that is shown so the reasoning can be followed rather than guessed.

### Q1. Name the four planes an event must be placed on, and say what breaks if one is omitted.

**Answer.** Subject, time, channel/node, promise. Omitting subject makes counts unjoinable at the grain the business acts on. Omitting time makes point-in-time correctness impossible, so backtests leak. Omitting channel/node makes a coverage gap indistinguishable from a service gap, because the gap is attributed to nobody. Omitting promise makes it impossible to say whether a delivery was late, because lateness is relative to a commitment rather than to a clock.

**Reasoning.** Each plane carries a different owner and changes at a different cadence, so a missing plane is a missing decision, not a missing column. The failure surfaces late: the metric computes fine and the interpretation is wrong.

**Follow-up.** Which plane is missing when a dashboard shows a saw-tooth series that tracks shift change? Time — specifically effective time versus ingest time.

**Red flag.** Answering "we have a schema, so the planes are covered". A schema records columns; it does not record which of two `ship_date` values the promise is measured against.

**Scoring.** Full credit for naming all four and one concrete failure per plane; partial for naming planes without the failure mechanism.

### Q2. A rate is reported as 0.8059 for collection, but the journal shows 0.710. Reconcile the two numbers.

**Answer.** 739 event rows over 917 attempted pickups gives 0.8059. The 739 includes 88 duplicates, so the deduplicated count is 651 and 651/917 = 0.70992 ≈ 0.710, matching the journal. The journal's own count of 703 collections implies the stream is still short by 52 rows, i.e. coverage 651/703 = 0.9259, and 0.710/0.9259 = 0.7668 would be the coverage-corrected estimate.

**Reasoning.** Three distinct effects are stacked: duplication (+9.6pp), true rate, and coverage shortfall. Reconciling requires separating them in that order, because fixing coverage first would correct the wrong base.

**Follow-up.** Which of the two published numbers should go to the operations director, and with what caveat? The deduplicated one, with the coverage correction labelled as an estimate and the 14 legacy nodes named.

**Red flag.** Averaging 0.8059 and 0.710 to "split the difference". The two describe different quantities: one is events per attempt, the other is collections per attempt.

**Scoring.** Credit requires the duplicate subtraction and the 917 denominator; the coverage correction is the differentiator.

### Q3. Compute the gap between order-grain and unit-grain on-time rate for the Section 5 example, and explain why the denominators differ.

**Answer.** Unit grain: (1,780 − 214)/1,780 = 1,566/1,780 = 0.87978. Order grain: (1,660 − 163)/1,660 = 1,497/1,660 = 0.90181. The gap is 2.20 percentage points. Denominators differ because 96 orders shipped two parcels and 12 shipped three, giving 1,780 parcels against 1,660 orders, a ratio of 1.0723; an order counts as on time if any parcel arrived on time under the order-grain rule, which raises the order rate.

**Reasoning.** The gap is not an error in either number; it is the difference between "the customer's parcel arrived on time" and "the customer's order arrived on time". Which is correct depends on what the customer was promised and what the compensation policy pays on.

**Follow-up.** If the compensation policy pays per late parcel, which rate should the service-level agreement use? Unit grain, because the payout is per parcel and the order-grain rate would understate liability.

**Red flag.** Reporting the order-grain rate as if it were unit grain, or claiming the difference is caused by missing data.

**Scoring.** Both rates plus the parcel ratio; a correct explanation of why the order rate is higher is the key step.

### Q4. Why can a rate be undefined rather than zero, and give an omnichannel instance.

**Answer.** A rate is a ratio whose denominator is a population eligible to experience the event. When the eligible population is empty, the rate has no value; reporting 0% asserts that every eligible case failed, which is a different and false claim. Instance: a store with no collection orders in a window, where a collection rate computed as collected/attempted is 0/0. Reporting 0% ranks the store worst in the estate and triggers an intervention in a store that had nothing to collect.

**Reasoning.** The distinction matters for aggregation: zeros enter a mean and drag it down, while undefined values must be excluded and their exclusion reported. The count of excluded units is itself a metric.

**Follow-up.** How do you aggregate stores with undefined rates into a district figure? Weighted by attempted pickups, excluding undefined, and publish the excluded share — otherwise the district number is not comparable to a district where every store had volume.

**Red flag.** Coalescing null to zero in SQL. It is one character and it manufactures a service failure.

**Scoring.** Requires both the undefined/zero distinction and an omnichannel example with a consequence.

### Q5. A promise metric improves by 3 points after a checkout configuration change. What must you check before publishing?

**Answer.** Four things in order. Whether the eligibility SQL changed (freeze and re-run). Whether the promise window length changed — a longer committed window raises on-time rate with constant transit time; compute the mean promised days before and after, since a step in that mean with constant actual transit identifies this cause. Whether the mix changed, for example a promotion shifting orders toward fast categories. Whether censoring changed, because a longer promise lengthens time-to-terminal and can exclude more recent orders from the mature sample.

**Reasoning.** Of the four, only the last two are visible in the rate itself; the first two require an external fact. That is why the taxonomy must version the definition beside the number.

**Follow-up.** The rate stays at the new level for a quarter. Does that prove improvement? No — a window change is permanent, so persistence is compatible with it. Compare actual transit days, which are window-independent.

**Red flag.** Accepting the change because "operations says nothing changed". Operations does not own the checkout configuration.

**Scoring.** Three of four checks named, with the window-length check as the differentiator.

### Q6. Quantify the fan-out inflation if 300 of 1,850 events match a three-child parent and the rest match unique children.

**Answer.** |J| = 300×3 + 1,550×1 = 900 + 1,550 = 2,450, so m = 2,450/1,850 = 1.3243. A measure of $18.50 per event sums to $33,925 unjoined and $45,325 joined — an inflation of $11,400, i.e. 33.6% on the true figure. The excess is concentrated: 300 events carry the entire inflation, so the average understates the effect on split orders, where the multiplier is 3.

**Reasoning.** The inflation is a weighted mean, so quoting it without the concentration is misleading. Check the multiplier distribution, not only its mean, before concluding a join is acceptable.

**Follow-up.** How do you repair it? Aggregate the right table to the left grain before joining, or deduplicate the right table to one row per join key with an explicit rule for which child wins.

**Red flag.** Deduplicating the left table to fix a right-table fan-out. That deletes real events.

**Scoring.** Correct arithmetic plus recognition that the mean hides the tail.

### Q7. A store shows a 0.62 collection rate; the manager has receipts. Diagnose.

**Answer.** Treat the number as unverified. Compare three independent counts on the same node and window: stream events, journal receipts, heartbeats. Heartbeats present with low stream volume points to a dark emitter that later reconciled manually, or to substituted orders routed through a branch the old client cannot emit. Heartbeats absent points to a plumbing gap whose backlog may have flushed with `t_e` inside the gap and `t_i` after it — in which case no data is lost and the rate is correct once computed on event time. A clock offset shows as an implausible modal hour and a nonzero share of `t_e > t_i`.

**Reasoning.** Four distinct causes produce one symptom, and only the receipt count distinguishes "was collected" from "was recorded". The manager's receipts are the control total.

**Follow-up.** After restoring from the journal, is the rate publishable? Only with the reconstruction flagged, because journal rows are order-grain and a unit-grain promise computation would treat them as complete.

**Red flag.** Adjusting the store's rate by a fudge factor until it matches neighbours.

**Scoring.** Requires at least two competing causes and the heartbeat matrix; naming only "data quality" scores zero.

### Q8. Why is the substitution consent attribute part of the schema rather than a policy detail?

**Answer.** Because it changes the state machine's legal transitions. A consented substitution is a completed fulfilment of a modified commitment; an unconsented substitution is a defect that often ends in a return and sometimes a complaint. Without the attribute, the two collapse into one event and the resulting rate improves or worsens purely with recording practice. In the Section 5.4 trace, handling the 54 substitutions three ways yields 0.9738, 0.9203 and 0.9027 — a 7.1-point spread driven entirely by the rule.

**Reasoning.** The attribute is a branch condition, and branch conditions belong in the taxonomy because they determine denominators. A policy note cannot be joined.

**Follow-up.** What if consent is captured in the app and the store client cannot read it? Then the store's substitutions must be flagged ambiguous and excluded from the consented numerator, with the excluded volume reported — not silently counted as either.

**Red flag.** Counting substitutions as free text in an operations comment field.

**Scoring.** Requires the state-machine argument and at least one of the three arithmetic outcomes.

### Q9. Compute the storage cost of heartbeats for 61 nodes at 15-minute cadence and say when it matters.

**Answer.** 86,400/900 = 96 heartbeats per node per day, so 61 × 96 = 5,856 rows/day. At 120 bytes per row that is 702,720 bytes, about 0.67 MiB/day or 245 MiB/year. It matters for row count rather than bytes: 5,856 rows/day is about nine times the 651 collection events in the same estate, so in a scan-priced engine or a per-row quota the heartbeat dominates. Keeping hourly rollups (61 × 24 = 1,464 rows/day) solves it while preserving last-seen, interval count, and client version.

**Reasoning.** The byte total is trivial and tempting to dismiss; the row total is not. Cost models priced per row or per scanned partition make the trivial number expensive.

**Follow-up.** What must the rollup retain? Those three fields; a count alone cannot distinguish alive-and-idle from alive-and-broken.

**Red flag.** "Storage is cheap, keep everything raw forever" — which also keeps a year of a schema the team may not want to maintain.

**Scoring.** Correct arithmetic plus the row-versus-byte distinction.

### Q10. Two replays of the same input produce different collection counts. What is the defect?

**Answer.** Non-determinism in transition application. The likely cause is ordering by `t_i` rather than by `t_e`, so network timing decides which transition is applied first and an illegal transition is accepted in one run and rejected in the other. A secondary cause is a tie in `t_e` broken by a non-stable key, or a duplicate key that the dedupe resolves differently when input order changes.

**Reasoning.** Replay determinism requires a total order per unit key, and ties must be broken by a key that is itself deterministic — the transition identity. Any source of nondeterminism in the sort becomes a metric that changes without data changing, which destroys every comparison across runs.

**Follow-up.** How do you prove the fix? Run the replay twice on the same input and assert equality of state per unit, as a scheduled test rather than an incident response. Falsification of R7 in Section 10 is exactly that test.

**Red flag.** Explaining it as "distributed systems are eventually consistent" — this is a batch job, not an eventually consistent store.

**Scoring.** Requires ordering by event time plus a stable tiebreak, and the double-run test.

### Q11. A funnel stage drops 30% on the day of a platform migration. Give the two candidate signatures and how to tell them apart.

**Answer.** Rename signature: one `event_name` disappears and a new one appears with complementary volume, so a consumer matching the old name sees a fall. Double-count signature: consumers that match both names see a rise. The discriminator is a daily count of distinct `event_name` values with volumes — complementary volumes confirm the rename; equal volumes confirm dual emission. A third signature, a key retype (string to integer, leading zeros dropped), shows as a stable left-row count with a falling join match rate.

**Reasoning.** The two signatures point in opposite directions, so the sign of the change is itself evidence. Confusing them leads teams to "fix" a rename by adding an alias when the true problem is duplication.

**Follow-up.** How should the rename have been shipped? Additively: emit the new name, retain the old through a deprecation window, record the alias in the taxonomy version that introduces it, and make the alias pair a single transition for dedupe.

**Red flag.** Backfilling the old name into the new one without recording the mapping, which makes the history unauditable.

**Scoring.** Both signatures plus at least one discriminating query.

### Q12. Why does a status column on the order header fail for promise work, even though it is accurate right now?

**Answer.** Because promise accuracy is a question about a past instant: was the unit's terminal timestamp at or before the commitment valid at that timestamp? A mutable status column retains neither the terminal timestamp as of the event nor the commitment as of the event. Once a promise is revised, the header holds the new value and every historical evaluation silently uses it. The Section 5.3 sensitivity shows a 1.7-point swing from that alone; if revisions are frequent during disruption, the swing is larger exactly when the metric matters most.

**Reasoning.** The failure is not accuracy of the current value but the destruction of the value at the time of decision. Read-committed state cannot answer a point-in-time question.

**Follow-up.** Is an audit table on the header sufficient? Only if every write path writes an audit row, including admin screens; the invariant at write time is harder to enforce than an append-only event, which is why the event log is preferred.

**Red flag.** "We can reconstruct it from backups" — backups are periodic, unqueryable at grain, and usually outside the retention of the metric window.

**Scoring.** Requires the point-in-time argument and the revision mechanism.

### Q13. Design the minimum schema that supports coverage, path completeness, and promise accuracy together.

**Answer.** An append-only transition table with: `unit_id`, `order_id`, `transition` (enumerated verb), `attempt` (integer, increments on re-ship), `t_e`, `t_i`, `node`, `channel`, `promise_due_at` (snapshot), `client_version`, `definition_version`, and a derived `transition_id` hashed from `(unit_id, transition, attempt)`. Plus a node heartbeat table and a dimension for branch assignment from the immutable commercial choice.

**Reasoning.** Coverage needs the control total comparison, which needs `transition` and `node` to align with the journal. Path completeness needs `attempt` and branch assignment. Promise accuracy needs `promise_due_at` as a per-event snapshot. `client_version` is what makes duplicate diagnosis possible after the fact; without it, the Section 8.1 check cannot be run.

**Follow-up.** What can be dropped for a small estate? `channel` if only one channel exists, and the heartbeat if fewer than five nodes — but add the heartbeat before the sixth node, not after.

**Red flag.** Adding `promise_due_at` as a foreign key to a current-order table, which reintroduces the mutable-value defect through the back door.

**Scoring.** Requires `attempt` and the per-event snapshot; those two are the discriminating details.

### Q14. The operations director wants one number for "how good is our fulfilment". What do you say?

**Answer.** That there is no single number, and offer three with their questions attached: coverage-corrected collection rate (are we seeing the work at all), unit-grain promise accuracy against the snapshot (are we keeping commitments per parcel), and unconsented substitution rate (are we changing the deal without agreement). Each has a different owner and a different intervention. If forced to one number, use unit-grain promise accuracy with the coverage and substitution figures printed adjacent as guardrails, because the guardrails are what prevent the headline from being improved by redefining the population.

**Reasoning.** A composite score would need weights, and the weights would be contested exactly when the numbers disagree — which is when the score is needed. Three transparent numbers survive that argument better than one opaque one.

**Follow-up.** How do you prevent the headline from being gamed? Freeze the eligibility SQL per version, publish the version beside the number, and report the guardrails even when they are unflattering.

**Red flag.** Proposing a weighted index of the three, or reporting only the metric that improved.

**Scoring.** Requires the refusal of a single number with a workable alternative, and the guardrail mechanism.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Event grammar | `subject.verb.object_qualifier@version`, channel as a field, past-tense verb, state change only |
| Four planes | subject, time, channel/node, promise — all four on every event |
| Three clocks | `t_e` event, `t_i` ingest, `t_v` effective; rates must not mix them |
| Uniqueness | one event instance per `(unit_id, transition, attempt)`; dedupe at unit grain, never at order grain |
| Coverage | `c = R/X` against an independent control total; correct per node, never estate-wide |
| Path completeness | branch assigned at entry from the immutable commercial choice, not from events that happened |
| Promise accuracy | terminal timestamp against the per-event `promise_due_at` snapshot; revisions are events, never mutations |
| Grain gap | unit-grain rate ≤ order-grain rate; the gap scales with split rate and lateness independence |
| Fan-out | `m = |J|/|L|`; quote the concentration beside the mean or the tail is hidden |
| Absence | heartbeat present/absent × events present/absent; four cells, four owners |
| First diagnostic | recount the same window from a second source — never re-read the same aggregate |
| Refusal | no promise metric from a status column; no unit-grain denominator from order-grain receipts |
