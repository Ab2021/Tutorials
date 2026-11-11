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
