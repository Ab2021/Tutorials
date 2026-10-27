# Source Selection and the Hierarchy of Trust

> Module 02 — The Supply Chain Data Landscape
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01, 02.03, 02.08
> Version 2026.5

---

## 1. Why this topic exists

Two numbers arrive at the same meeting. The demand planner reports that sell-out for a promoted
item was 6,950 units last week. The supply planner reports that the company shipped 8,400 units
into the same channel. Both were produced by a correct query against a governed table. The meeting
resolves the difference by seniority, and the order is placed on whichever number the more senior
person prefers.

That resolution is a decision with a cost, and it is almost never written down. This topic
replaces it with a procedure.

The previous nine topics mapped the landscape: what each system can answer, the grain of
transactions, quality as tests, and how master data drifts and rewrites history. All assumed you had
already chosen a source. This topic is that choice, and the closing move of the module.

Three claims organise what follows.

1. **Every source has a position.** A number comes from somewhere, and where it comes from tells
   you how close it sits to the event and what its owner gains from the value. Those two
   dimensions, not the prestige of the system, determine what a number can prove.
2. **Conflict between sources is normal, and it is resolvable only by a rule written before the
   conflict.** Two systems disagreeing is usually the correct outcome of two systems measuring
   different fields. Which one you believe is a policy, and a policy invented in the meeting is a
   policy invented by the loudest person present.
3. **Reproducibility requires knowing what you knew then, not only what is true now.** Sources
   restate history. A model trained on the current version of last quarter's data and evaluated
   against a decision taken last quarter was trained on information nobody had at the time.

For a data scientist the stakes are specific. A feature built from shipments trains on a series
that includes pipeline fill. A label built from point-of-sale (POS) data trains on a series
censored by shelf availability. A training window assembled after a restatement contains numbers
that did not exist when the decision was made. In each case the model is not wrong about the
world. It is answering a question about a different source than the business thinks it is using.

Two examples show the shape of the problem.

**The 21% overship.** A manufacturer replenishes a retailer from its own shipment history. During a
promotion, shipments exceed sell-out by 1,450 units on a base of 6,950. The planner reads the
shipment number as demand, orders to it, and the pipeline ends the promotion carrying six weeks of
stock. The shipment number was accurate. It measured outbound logistics, not consumer takeaway.

**The model that retrained on corrected history.** A demand model is retrained monthly on the POS
table. The retailer restates the last three weeks as late transactions settle, and one promotion
week is revised down by 6%. The model learns a lower level for that week than the planner saw at
the time. Offline backtests look stable, the production forecast drifts against the plan, and the
difference is a revision that arrived after the decision.

Both are source choices, not algorithm failures. They are usually made implicitly, by whoever
built the query, and then inherited by everyone downstream without a record.

## 2. Learning outcomes

By the end of this topic you can:

1. **Rank** candidate sources on a trust ladder defined by proximity to the event and incentive to
   misreport, and **justify** each placement with a mechanism rather than a preference.
2. **Distinguish** sell-in, sell-out, warehouse movement, inventory, and consumption, and
   **explain** why a shipment is not demand, a POS scan is not consumption, and a withdrawal is not
   a stock.
3. **Decompose** a decision into the fields it requires, and **assign** a source of record per
   field rather than per system.
4. **Reconcile** conflicting sources using the five bridges, and **attribute** the residual rather
   than plugging it.
5. **Score** candidates on grain, latency, completeness, revision behaviour, auditability, and
   cost, and **explain** why a weighted score cannot override a critical requirement.
6. **Design** a bitemporal capture that answers what you knew, when you knew it, and what is true
   now.
7. **Write** a source-selection record naming the decision, candidates, choice, known defects,
   failure signature, and owner, and **critique** a claim of triangulation by identifying the
   shared upstream cause that would make three agreeing sources one source.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures: a ladder of sources ordered by proximity to the event and by the owner's
incentive to misreport, and a flow that turns a disagreement into a documented choice.

### 3.1 The trust ladder

```text
   HIGH  ^
         |  L7  audited physical observation          (counted, weighed, witnessed)
         |  L6  first-party event capture             (scan at till, putaway scan, sensor)
         |  L5  counterparty-confirmed transaction    (proof of delivery, receipt confirmation)
         |  L4  internal system of record, closed     (ERP ledger, posted inventory, invoice)
         |  L3  counterparty claim, unconfirmed       (advance ship notice, order, carrier status)
         |  L2  plan, forecast, commitment            (intent, allocation, promotional plan)
         |  L1  modelled or syndicated estimate       (panel projection, modelled POS, index)
   LOW   |  L0  narrative and anecdote                (opinion, press, memory)
         +-------------------------------------------------------------->
            near the event                          far from the event
            weak incentive to misreport             strong incentive to misreport
```

The ladder is a teaching device, not a law. It encodes two judgements that are usually left
implicit.

**Proximity.** How many transformations and interventions sit between the event and the number. A
till scan is one from the sale; a category report built from a store sample, projected to a
universe, and adjusted for coverage is six or seven.

**Incentive.** What the owner gains if the value moves. An ASN affects a supplier's compliance
score and its allocation in a shortage, so it is an interested claim. A proof of delivery signed
by the receiver is confirmed by a party with the opposite interest, which is why it ranks higher.

The two dimensions usually point the same way. When they do not, the conflict is informative. A
retailer's own inventory report is a first-party record of its stock and also its negotiating
number. Read it as evidence about the retailer's belief, not as a census of cartons.

| Level | Typical source | What it proves | What it cannot prove | Why it sits here |
|---|---|---|---|---|
| L7 | Cycle count, weighbridge ticket, witnessed sample | State of the sampled units at that time | The population outside the sample | Closest to the physical fact; observer independent by design |
| L6 | POS scan, putaway scan, telematics ping, sensor | An event was recorded by a device at the point of action | That the event means what the field name claims | Machine-timestamped by-product; device configuration still has an owner |
| L5 | Signed proof of delivery, receiving confirmation, three-way match | Two parties agreed a quantity at a time | That the agreed quantity is the true quantity | One step removed, but the counterparty's interest offsets the originator's |
| L4 | ERP ledger, posted inventory, invoice | What the organisation has committed to in its books | Physical state; the ledger is derived and closed | Aggregated and corrected; good for obligation, weak for current fact |
| L3 | ASN, supplier ship confirmation, carrier status, order | What a party asserted | That the goods match the assertion | Before the event or at dispatch; strong incentive to report favourably |
| L2 | Plan, forecast, allocation, promotional plan | Intent | Outcome | About the future; incentives include sandbagging and target-setting |
| L1 | Syndicated panel, modelled POS, category index | A model's output given its inputs and sample | The census; anything outside the sample frame | Derived through someone else's model and coverage assumptions |
| L0 | Sales team's read, press, executive memory | That someone said it | Anything about the world | No method, no frame, no audit |

Three placements are worth arguing about, because arguing about them is the skill.

**Why POS is L6 and not L5.** A POS scan is captured at the point of action by a device, with a
timestamp, as a by-product of the transaction. There is no counterparty confirming the quantity;
the shopper does not sign for the scan. It is not L7 because nobody audits each scan.

**Why the ASN is L3 and not L4.** The ASN is created before the goods arrive by the party that
benefits from the shipment being recorded as complete and on time. Retailer receiving systems
validate it and return an application advice (EDI 824) when quantity, item, or pallet does not
match. The existence of that return message is the operational proof that the ASN is a claim.
Kroger's vendor requirements make the point bluntly: a supplier may bypass manual verification
only after demonstrating 100% accuracy on licence plates, case codes, quantities, and expiry
dates, and even then remains subject to random audit.

**Why the retailer inventory report is L4 and not L6.** It is derived from scans and adjustments,
closed at a period end, and owned by a party with a commercial interest in the number. It is a
good record of the retailer's book. It is not a count of cartons.

### 3.2 The reconciliation flow

```text
   decision  --->  fields required  --->  candidates (grain, coverage)
                          |                        |
                          v                        v
                     tolerance              place on ladder, per field
                          |                        |
                          +----------> bridge: timing / coverage /
                                       definition / unit / inventory
                                               |
                                               v
                                         residual = 0 ?
                                          /          \
                                        yes           no
                                         |             |
                                    accept pair   attribute or escalate
                                         |             |
                                         +----> precedence rule (written before)
                                                       |
                                                       v
                                          chosen value + known-defect note
                                                       |
                                                       v
                                          bitemporal store + selection record
```

Five properties follow.

**Property 1 — The ladder ranks a source for a field, not for a system.** Oracle, SAP, a warehouse
management system (WMS), and a retailer portal are not ranked once and for all. A WMS is L6 for
"quantity put away" and L4 for "inventory value". An ERP is L4 for "invoiced quantity" and L2 for
"forecast". Every system is a different rung for every field it holds.

**Property 2 — A higher rung does not automatically win.** The ladder ranks evidential quality
conditional on coverage and grain. A L1 panel covering 95% of the market can beat a L6 scan feed
covering 60%, for a decision about the total market.

**Property 3 — Disagreement is usually correct.** Two numbers that measure different fields should
differ. Sell-in and sell-out differ by inventory build and shrink. That difference is information
about the channel. The failure is not the gap; the failure is a gap nobody can explain.

**Property 4 — A source's defects have a signature.** A censored POS series under-reads exactly
when shelves are empty, so its error correlates with availability. A shipment series over-reads
when the pipeline is filling, so its error correlates with the change in channel inventory. Name
the signature and you can monitor for it. Otherwise you cannot tell a defect from a change in the
business.

**Property 5 — Selection is a decision with an owner, and decisions decay.** A source chosen for a
weekly decision in 2024 may be wrong in 2026 because the channel mix moved, the retailer changed
its feed, or the promotion calendar changed. A record without a review trigger becomes folklore.

### 3.3 Why "single source of truth" is a slogan

The phrase describes a destination that does not exist and should not. A company needs one number
for quantity invoiced, because finance closes on it and auditors test it, but it needs several
legitimate numbers for demand, because manufacturer, retailer, and shopper views are different
objects. Collapsing them creates a number whose provenance nobody can reconstruct.

The defensible form is narrower: **one source of record per field, per decision context**. A field
can have one owner. A system cannot, because it holds many fields at different rungs.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Data source | Any system, document, device, or person from which a value is obtained | Treated as a synonym for database table | Source identifier |
| System of record | The system that owns a master or transactional object and is accountable for it | Assumed to own every field derived from it | System name plus object |
| Source of record | The source designated authoritative for one field in one decision context | Applied at system level | Field plus context identifier |
| Primary data | Data captured at the point of the event by the party to the event | Confused with "our own data" | Collection method |
| Secondary data | Data derived from primary data by aggregation, modelling, or estimation | Treated as observation | Derivation method |
| First-party data | Data collected by the organisation using it | Assumed unbiased | Collector identity |
| Third-party data | Data collected by an organisation with no direct relationship to the subject | Assumed independent of the buyer | Provider plus licence |
| Sell-in | Units shipped by a supplier into a customer's distribution network | Used as demand | Units per period |
| Sell-out | Units sold by a retailer to shoppers, usually at POS | Used as consumption | Units per period |
| Point of sale (POS) | The system and transaction record at the retail checkout | Used as a census of demand | Transaction identifier |
| Syndicated data | Retail measurement sold to multiple manufacturers, typically from a store sample | Treated as a census | Provider plus universe definition |
| Panel | A sample of stores, households, or outlets projected to a population | Confused with the population | Sample size plus projection method |
| Census | Complete enumeration of a defined population | Claimed for any large dataset | Population definition |
| Coverage | Share of the decision's population represented by a source | Reported as accuracy | Proportion of volume or outlets |
| Channel inventory | Goods held between the supplier's dispatch and the shopper's basket | Omitted from demand reasoning | Units |
| Pipeline fill | Inventory added in the channel to support a higher run-rate | Counted as demand | Units |
| Forward buy | Purchases brought forward from a future period by a commercial incentive | Counted as growth | Units |
| Stockout censoring | Under-observation of demand because the item was unavailable | Ignored in demand estimation | Availability rate, percent of store-days |
| Restatement | Re-issuance of a previously published value with a corrected value | Confused with a new period | Value plus revision timestamp |
| Revision | Any change to a stored value after first publication | Assumed to be an error correction only | Revision sequence number |
| Valid time | The period in real time that a fact describes | Confused with load time | Time interval |
| Transaction time | The period during which a stored value was current in the database | Confused with event time | Time interval |
| Bitemporal | Storing both valid time and transaction time for each version of a fact | Used loosely for any history table | Two interval pairs |
| As-of query | A query returning values known at a specified past instant | Run against current data and called historical | Timestamp parameter |
| Snapshot | A copy of a dataset at a point in time, usually losing intra-period detail | Treated as a backup | Snapshot date plus scope |
| Provenance | The origin and history of a value, including who produced it and how | Confused with lineage | Provenance record identifier |
| Lineage | The technical chain of transformations from source to output | Used to mean business meaning | Graph node and edge identifiers |
| Data contract | A versioned, testable agreement between a data producer and its consumers | Treated as a dashboard or wiki page | Contract version |
| Service-level objective (SLO) | A target for a measurable service property such as freshness | Confused with a contractual service-level agreement (SLA) | Metric plus threshold plus window |
| Declared state | What a contract or SLA claims about a dataset | Reported as performance | Claim plus effective date |
| Observed state | What measurement shows about the same dataset | Omitted from governance reporting | Metric plus measurement window |
| Reconciliation | Explaining the difference between two representations of the same field | Forced agreement | Residual plus attribution |
| Bridge | A structured decomposition accounting for the difference between two sources | A plug figure | Line items in units or currency |
| Residual | The part of a bridge that remains unexplained after attribution | Hidden in an adjustment line | Units or percent of flow |
| Triangulation | Using three or more sources to corroborate a value | Treated as proof of truth | Number of independent causes |
| Common-cause correlation | Agreement between sources that share one upstream origin | Mistaken for independent confirmation | Shared ancestor identifier |
| Precedence rule | A written rule stating which source wins when two remain in conflict after bridging | Decided in the meeting | Rule identifier plus version |
| Source-selection record | A document naming the decision, candidates, choice, defects, signature, and owner | Written once and never reviewed | Document identifier |
| Failure signature | The observable pattern indicating that the chosen source is failing | Confused with a general alert | Metric plus threshold plus direction |

Five deserve expansion.

**Sell-in, sell-out, and consumption.** Sell-in is what the supplier shipped into the channel,
sell-out what the retailer sold at the till, consumption what the shopper used. They differ by
channel inventory change, returns, shrink, and resale. A pantry-filling promotion raises sell-out
and not consumption; a shipment series measures dispatch policy as much as demand.

**Restatement and revision.** A revision is any change after first publication; a restatement is
one to a value already used in a decision. Only restatements break reproducibility.

**Valid time and transaction time.** Valid time is when the fact was true: the week of sales, the
day of delivery. Transaction time is when the database believed it. A POS row for week 37 can have
valid time 7–13 September and transaction time from 14 September to 2 October, when a corrected row
takes over. A single-timestamp table collapses these and loses the ability to answer either
question.

**Provenance and lineage.** Provenance is the origin and custody history of a value: who produced
it, from what, and when. Lineage is the technical transformation graph inside the pipeline. A
value can have complete lineage and no provenance, as when a pipeline faithfully transforms a
column whose business origin nobody recorded.

## 5. The mechanics, step by step

The procedure has eight steps. The order matters: most source failures come from skipping step 1
or step 4.

### 5.1 Step 1 — Fix the decision, its cadence, and its tolerance

A source cannot be selected for a decision that has not been stated as an action. Write the
decision in one sentence with five elements.

| Element | Question | Example |
|---|---|---|
| Actor | Who acts | Regional replenishment planner, North |
| Action | What they do | Place a purchase order on the manufacturer's DC |
| Cadence | How often | Weekly |
| Cut-off | When the decision is taken | Wednesday 14:00 |
| Unit | The unit of the action | Cases of 6 x 500 ml, pallets of 60 |

Then state the tolerance: the smallest input error that would change the action, derived as in
Topic 08. An order near 4,200 units rounded to pallets of 60 has a granularity of 1.4%. If two
inputs contribute independently, an error below roughly 0.7% cannot move the rounded order. That
number, not a general preference for accurate data, sets the bar a source must clear.

Two consequences follow. A monthly decision can tolerate weekly latency; a daily decision cannot.
And a tolerance of 1.4% does not require a source accurate to 0.1%, only one whose bias is below
0.7% and whose failure signature is monitorable.

### 5.2 Step 2 — Decompose the decision into fields

A decision is not served by one number. It is served by several fields, each with its own grain,
tolerance, and freshness. Skipping the decomposition is what produces "which system is right"
arguments.

| Field | Question the field answers | Grain | Tolerance | Freshness |
|---|---|---|---|---|
| Consumer takeaway | How fast the item leaves shelves | SKU x region x week | 5% | 7 days |
| Channel inventory | How much is in the retailer's network now | SKU x DC x day | 3% | 1 day |
| In-transit | What we shipped and have not received | SKU x DC x shipment | 2% | 1 day |
| Item identity | Is this the same item as last month | GTIN | Exact | On change |
| Order policy | What the retailer will accept and when | SKU x DC x week | Exact | 30 days |
| Price and promotion | What the shopper is being offered | SKU x channel x week | Exact | 14 days |

Each row can have a different source of record, and usually does. Consumer takeaway generates the
argument because it is the field with no direct observation available to the manufacturer.

> **Note:** A decision that needs six fields has six source decisions, not one. A team that
> selects "the ERP" or "the data warehouse" has selected a building, not a source.

### 5.3 Step 3 — Enumerate candidates and their coverage

For each field, list every candidate source and record three properties before comparing values:
grain, coverage, and frame.

| Candidate | Grain | Coverage | Frame and exclusions |
|---|---|---|---|
| Manufacturer shipment (sell-in) | SKU x customer DC x shipment line | 100% of own shipments | Own dispatch only |
| Retailer POS (sell-out) | SKU x store x day | 82% of regional volume | Excludes club, discount, online; excludes stores off the feed |
| Retailer DC withdrawals | SKU x DC x day | 82% | Same retailer only; measures replenishment policy |
| Syndicated panel | SKU x category x week | 78% of measured channel | Modelled from a store sample; vendor category definitions |
| Retailer inventory report (EDI 852) | SKU x DC x day | 82% | Retailer's own book, not a count |

Coverage is a share of the decision's population, not of the dataset: a feed with ten million rows and
60% volume coverage is less complete than one with ten thousand rows and 95% coverage.

### 5.4 Step 4 — Bridge before you compare

Two numbers that differ should be bridged before either is judged. The bridge accounts for every
unit of the difference using identities that must hold. The five bridges are applied in this
order, because each changes the population the next operates on.

| Bridge | What it accounts for | Identity or test | Typical size |
|---|---|---|---|
| Timing | Cut-off and transit differences | Same period, time zone, and week definition | Hours to days |
| Coverage | Population differences | Volume covered by each, as a share of the decision population | 0–40% |
| Definition | Different constructs under one name | Item, category, channel, and status compared field by field | 1–15% |
| Unit | Cases, eaches, pallets, litres, weight | Convert to one unit with a stated factor | 0–5% if the factor is wrong |
| Inventory | Movement versus stock | Opening + inflows - outflows ± adjustments = closing | The whole gap |

The inventory bridge resolves most sell-in versus sell-out arguments, and it is an identity.

```text
   closing inventory = opening inventory
                     + sell-in
                     - sell-out
                     + returns and adjustments
```

Rearranged, the difference between shipments and sales is the change in channel inventory plus
adjustments:

```text
   sell-in - sell-out = (closing - opening) + adjustments
```

Every unit of the gap is either in the channel or has left the system through shrink, damage,
returns, or an unrecorded adjustment. If the bridge does not close, you have not finished
attributing. You have a residual, and a residual is a finding, not a plug.

> **Note:** The identity is exact only if inventory and flows cover the same population. A retailer
> inventory report covering 82% of stores, combined with a sell-in figure covering 100% of
> shipments, will not close. Bridge coverage first, then inventory.

### 5.5 Step 5 — Resolve what remains with a written precedence rule

Some residual usually remains: measurement error, a real adjustment, or a definitional difference
nobody has found. The rule for resolving it must be written before the conflict, at three levels.

**Level 1 — Physical and accounting constraints.** A rule that cannot be violated wins over any
source. Negative inventory, delivery before dispatch, and a shipment larger than vehicle capacity
are impossible regardless of which system reports them. These need no meeting.

**Level 2 — Field precedence.** For each field, an ordered list of sources and the condition under
which the order changes.

| Field | First | Second | Fallback | Condition that changes the order |
|---|---|---|---|---|
| Quantity received | Receiving scan at the DC | Signed proof of delivery | Carrier delivery confirmation | None: the scan is the record |
| Quantity shipped | WMS dispatch confirmation | ASN | Invoice quantity | If the WMS is unavailable, ASN with a defect flag |
| Consumer takeaway | Retailer POS, adjusted | Syndicated panel | Shipment series | If POS coverage falls below 70% |
| Channel inventory | Retailer EDI 852 | Retailer portal extract | Modelled from shipments | If the retailer stops publishing daily |
| Item identity | GTIN in the product master | Retailer item file | Manual mapping | On any pack-size change |

**Level 3 — Tie-break.** When two sources at the same level disagree and neither is excluded, the
tie-break is a stated rule: shorter latency, higher coverage, or the more conservative value for
the decision's cost asymmetry. The rule must name which, because they give different answers.

A precedence rule that says "use the most reliable source" is not a rule. It restates the question.

### 5.6 Step 6 — Capture revisions bitemporally

Every source that matters revises. POS restates as late transactions settle. Invoices are
corrected. Inventory is recounted. If the pipeline overwrites the old value, three questions
become unanswerable: what the planner saw, when they saw it, and what we believe now.

A bitemporal table stores each version with two intervals.

| Column | Meaning | Example |
|---|---|---|
| `valid_from`, `valid_to` | The period in the world the fact describes | 2026-09-07 to 2026-09-13 |
| `txn_from`, `txn_to` | The period the database held this version | 2026-09-14 02:00 to 2026-10-02 03:00 |
| `value` | The quantity or attribute | 6,950 units |
| `revision_reason` | Why the version changed | Late transaction settlement |

An as-of query then answers the reproducibility question directly: filter
`txn_from <= decision_time < txn_to`. Running it at two decision times returns two numbers; the
difference is the restatement, and it must be recorded because it invalidates a backtest.

The concept is old and well specified. Snodgrass defines valid time and transaction time as
orthogonal dimensions and treats transaction-time relations as append-only. TSQL2 proposed the
language extensions in 1994, and SQL:2011 added system-versioned tables. This is a solved
modelling problem that most warehouses still do not implement, because overwriting is easier.

### 5.7 Step 7 — Score fitness for purpose

Ladder position is necessary and not sufficient. Fitness is scored on six anchored axes, so that
two people scoring the same source land within one point.

| Axis | What it measures | Score 1 | Score 3 | Score 5 |
|---|---|---|---|---|
| Grain | Fineness relative to the decision | Two levels coarser | One level coarser | Matches the decision |
| Latency | Event to query availability | More than one decision cycle | Within one cycle | Within the cut-off |
| Completeness | Coverage of the decision population | Below 70% | 85–95% | Above 98% |
| Revision behaviour | Predictability and traceability of changes | Silent overwrites | Revisions known, not versioned | Bitemporal and documented |
| Auditability | Ability to trace a value to origin | No provenance | Lineage only | Provenance, lineage, contract |
| Cost | Licence, integration, processing | Prohibitive | Material but affordable | Marginal |

The weighted score is the sum of axis scores times weights that sum to one. Weights come from the
decision. A daily replenishment decision might weight grain 0.25, latency 0.20, completeness 0.20,
revision 0.15, auditability 0.10, and cost 0.10.

Three rules govern the score.

1. **A critical requirement is a gate, not a weight.** If the decision cannot be taken at a coarser
   grain, a source that fails the grain requirement is excluded before scoring, whatever its total.
2. **Do not score what you have not measured.** Completeness is a number from a test; latency is a
   p95 from event time, not a vendor's median.
3. **Record the weights and who set them.** A score with unstated weights is an opinion with
   arithmetic attached.

### 5.8 Step 8 — Write and review the source-selection record

The record is the deliverable. It is one page, it has a named owner, and it states what would cause
it to be revised. Section 6.6 gives a completed example. The mandatory contents are:

1. **Decision and context.** Action, actor, cadence, cut-off, unit, and tolerance.
2. **Fields required.** The decomposition from step 2, with grain and freshness.
3. **Candidates considered.** Every source evaluated, including the rejected ones, with coverage.
4. **Choice per field.** The selected source, its rung, and one sentence of justification.
5. **Rejected candidates and why.** The specific defect that excluded each.
6. **Known defects of the chosen source.** Each with the expected direction and size of error.
7. **Failure signature and monitor.** The observable pattern, metric, threshold, and frequency.
8. **Reconciliation rule.** The bridge, the residual tolerance, and the escalation path.
9. **Revision policy.** Which sources restate, over what window, and how versions are captured.
10. **Owner and review triggers.** A role, a review date, and the events that force an earlier
    review.

A record with all ten is a control. A record with the first four is a memo.

### 5.9 Triangulation and its limits

Triangulation is worth doing and is weaker than it looks. The useful version counts independent
causes, not series.

| Situation | Apparent sources | Independent causes | Evidential weight |
|---|---|---|---|
| POS, retailer inventory, retailer withdrawals | 3 | 1 (the retailer's systems) | One observation, three views |
| POS, manufacturer shipments, consumer panel | 3 | 3, two sharing our item master | Two and a half |
| POS, shipments, physical store audit | 3 | 3, with different failure modes | Three, with the audit bounding the others |
| POS, panel, category index from the panel | 3 | 1 (the panel) | One observation, two derivations |

The test is causal, not statistical. Ask of every pair: what single upstream event would make both
numbers move together? If the answer is "the same feed", they are one source. Agreement is then a
consistency check, which is useful and is not corroboration.

A second limit: sources sharing a bias agree, and the agreement is evidence of the bias, not of the
value. Consensus around an excluded channel is still wrong by the excluded share.

## 6. Worked example

A manufacturer of household goods supplies a national grocery retailer. The item is Aurora
6-pack, a promoted stock-keeping unit (SKU) in the North region. The decision is the weekly
replenishment order from the manufacturer's regional distribution centre (DC) to the retailer's
regional DC, placed every Wednesday at 14:00 for delivery the following week. All figures are
illustrative and internally consistent.

### 6.1 The decision

| Element | Value |
|---|---|
| Actor | Regional replenishment planner, North |
| Action | Place a purchase order on the retailer's DC |
| Cadence | Weekly, Wednesday 14:00 cut-off |
| Unit | Cases of 6 x 500 ml, ordered in pallets of 60 |
| Order rule | Forecast for the delivery week + safety stock - inventory position |
| Safety stock | 1,000 cases |
| Lead time | 1 week |
| Retailer DC inventory | 3,050 cases |
| In-transit | 0 |
| Unit cost | EUR 14.50 |
| Retail price | EUR 22.00 |
| Unit margin | EUR 7.50 |
| Annual holding cost | 22% of unit cost |

The order rule is deliberately simple so that the source choice is the only variable. In practice
a service-level calculation sets safety stock; here it is fixed.

### 6.2 The four candidates, same week

Week 37 of 2026 runs from Monday 7 September to Sunday 13 September. Four sources report on that
week. They are not four estimates of one number. They are four measurements of different things.

| Candidate | Week-37 value (cases) | What it actually measures | Grain | Coverage |
|---|---:|---|---|---|
| A. Manufacturer shipment (sell-in) | 8,400 | Our outbound, including pipeline fill | SKU x DC x shipment | 100% of our shipments |
| B. Retailer POS (sell-out) | 6,950 | Store-level scanner transactions | SKU x store x day | 82% |
| C. Retailer DC withdrawals | 7,300 | The retailer's store-replenishment response | SKU x DC x day | 82% |
| D. Syndicated panel estimate | 7,600 | A model output from a store sample | SKU x category x week | 78% |

The spread from B to A is 1,450 cases, 20.9% of the smallest value. Treating these as four estimates
of one quantity produces an argument about which is right; they are not the same quantity.

### 6.3 The inventory bridge

The retailer publishes daily DC inventory and weekly store inventory, both on the same 82% store
universe as the POS feed.

| Line | Retail DC (cases) | Store (cases) | Total (cases) |
|---|---:|---:|---:|
| Opening inventory | 2,100 | 1,400 | 3,500 |
| + Sell-in from manufacturer | 8,400 | 0 | 8,400 |
| - Withdrawals to stores | -7,300 | +7,300 | 0 |
| - Sell-out at POS | 0 | -6,950 | -6,950 |
| = Expected closing | 3,200 | 1,750 | 4,950 |
| Reported closing | 3,050 | 1,690 | 4,740 |
| Residual | 150 | 60 | 210 |

The identity closes to within 210 cases, 2.5% of sell-in and 3.0% of sell-out, both below the 5%
escalation threshold. The residual is attributed to shrink and unrecorded damage, and the
attribution is recorded.

The gap between shipments and sales is now fully decomposed:

```text
sell-in - sell-out        = 8,400 - 6,950            = 1,450 cases
channel inventory build    = 4,740 - 3,500            = 1,240 cases
residual (shrink, damage)  = 150 + 60                 =   210 cases
check                      = 1,240 + 210              = 1,450 cases
```

So 85.5% of the shipment-sell-out gap is inventory still in the channel, and 14.5% is product that
has left the system without a sale. Neither is an error. The shipment series records dispatch and
the POS series records scans; they differ by exactly the channel dynamics the bridge measures.

Two adjustments are needed before either can be read as demand.

**Stockout censoring.** During the promotion the item was available on 88% of store-days in the
panel. If demand during stockouts is proportional to demand during availability:

```text
latent demand = 6,950 / 0.88 = 7,897.7, rounded to 7,898 cases
```

That is an upper bound. Some shoppers substitute, some defer, and some do not buy. The bound
brackets the answer.

**Forward buy.** The following week's sell-out was 5,400 cases against a non-promotional baseline
of 6,200, a dip of 800. If 60% of the dip is pull-forward rather than shelf-reset loss, 480 cases
of week-37 volume were borrowed from week 38:

```text
sustainable week-37 level = 7,898 - 480 = 7,418, rounded to 7,400 cases
```

The 60% attribution is an assumption and the record must say so. A different attribution moves the
sustainable level by up to 320 cases.

### 6.4 What each candidate implies for the order

The order is placed in week 38 for delivery in week 39. The base-case forecast for week 39, from
the statistical baseline, is 6,300 cases.

```text
order = forecast + safety stock - inventory position
      = 6,300 + 1,000 - 3,050
      = 4,250 cases
```

If the promotion expanded the base rather than merely pulling demand forward, the requirement is
the sustainable level of 7,400:

```text
order = 7,400 + 1,000 - 3,050 = 5,350 cases
```

Take each candidate at face value as the week-39 run-rate and compute the implied order. A competent
planner would not, but an unexamined pipeline does when a source is wired into the order rule.

| Candidate | Value used as run-rate | Implied order (cases) | Difference from base |
|---|---:|---:|---:|
| A. Shipments | 8,400 | 6,350 | +2,100 |
| B. POS | 6,950 | 4,900 | +650 |
| C. Withdrawals | 7,300 | 5,250 | +1,000 |
| D. Panel | 7,600 | 5,550 | +1,300 |
| Reconciled baseline | 6,300 | 4,250 | 0 |
| Sustainability case | 7,400 | 5,350 | +1,100 |

### 6.5 The cost of each choice

Two costs are in tension. Ordering too much leaves stock in the channel, incurring holding cost
and markdown risk. Ordering too little loses margin on demand that cannot be served.

**Excess cost per case.** Assume the excess is carried for six weeks and that 35% of it is
eventually marked down at 30% off the retail price.

```text
holding cost  = 14.50 x 0.22 x (6 / 52) = 14.50 x 0.02538 = EUR 0.368 per case
markdown cost = 0.35 x (22.00 x 0.30)   = 0.35 x 6.60     = EUR 2.310 per case
total excess  = 0.368 + 2.310                            = EUR 2.678 per case
```

**Shortfall cost per case.** Lost margin is EUR 7.50. Assume 60% of unserved demand is genuinely
lost and 40% is recovered at the next order.

```text
shortfall cost = 7.50 x 0.60 = EUR 4.50 per case, assessed for one week
```

The asymmetry is the point. A case short costs 1.68 times a case carried for six weeks. That
ratio, not the ladder, should decide how much insurance the order carries.

| Candidate | Implied order | Cost if base true (excess) | Cost if upside true (short or excess) | Maximum regret |
|---|---:|---:|---:|---:|
| A. Shipments | 6,350 | 2,100 x 2.678 = EUR 5,624 | 1,000 x 2.678 = EUR 2,678 | EUR 5,624 |
| B. POS | 4,900 | 650 x 2.678 = EUR 1,741 | 450 x 4.50 = EUR 2,025 | EUR 2,025 |
| C. Withdrawals | 5,250 | 1,000 x 2.678 = EUR 2,678 | 100 x 4.50 = EUR 450 | EUR 2,678 |
| D. Panel | 5,550 | 1,300 x 2.678 = EUR 3,481 | 200 x 2.678 = EUR 536 | EUR 3,481 |
| Reconciled baseline | 4,250 | 0 | 1,100 x 4.50 = EUR 4,950 | EUR 4,950 |

> **Note:** This table measures the ordering rule's cost asymmetry more than source quality.
> Candidate B wins because it sits 650 cases above the base order, insurance the asymmetry makes
> cheap; a source reading 7,100 would win by the same logic. Do not read it as a source ranking.

The decision consequence is now explicit. Using shipments as the demand input costs EUR 5,624
under the base case and is the worst of the five options; using POS at face value costs EUR 1,741
under the base case and EUR 2,025 under the upside; the reconciled baseline costs nothing if the
promotion was a pure pull-forward and EUR 4,950 if it was not. That range, from EUR 0 to EUR 5,624,
is wider than any accuracy improvement the planning team will deliver this year, and source choice
produced all of it.

### 6.6 The completed source-selection record

```text
SOURCE-SELECTION RECORD                              SSR-2026-041
Version 1.2                                         Effective 2026-09-16
Owner: Regional demand planning lead, North

1. DECISION
   Weekly replenishment order, manufacturer North DC to retailer North DC.
   Planner, Wednesday 14:00 cut-off, delivery week +1. Pallets of 60 cases.
   Tolerance: 5% on takeaway, 3% on channel inventory, 2% on in-transit.

2. FIELDS AND SOURCE OF RECORD
   Consumer takeaway     retailer POS, adjusted for availability and forward buy
   Channel inventory     retailer EDI 852 daily inventory report
   In-transit            carrier EDI 214 plus our WMS dispatch confirmation
   Item identity         GTIN in the product master, versioned on pack change
   Price and promotion   contract price master plus retailer promotion calendar

3. CANDIDATES CONSIDERED
   A manufacturer shipment, B retailer POS, C retailer DC withdrawals,
   D syndicated panel, E retailer inventory report. All five were measured
   against the same week-37 bridge before any was rejected.

4. CHOICE AND JUSTIFICATION
   B is L6 for takeaway: a device records the scan at the point of action and the
   record is a by-product rather than a report. It is the closest observation of
   shopper behaviour available to us, and its defects are measurable.
   E is L4 for channel inventory: the retailer's own closed book, daily, DC grain.
   A is used only for in-transit, where it is L6 at dispatch and is reconciled
   at receipt.

5. REJECTED CANDIDATES
   A. Shipments: measures our dispatch and the retailer's order policy, and
      contains pipeline fill of 1,240 cases in week 37. Rejected as demand.
   C. Withdrawals: measures the retailer's replenishment rule, including a
      safety-stock rebuild of 290 cases. Retained only as the outflow term.
   D. Panel: modelled from a store sample, 78% coverage, vendor category
      definitions. Rejected where POS exists; fallback below 70% coverage.

6. KNOWN DEFECTS OF THE CHOSEN SOURCE
   B under-reads when availability falls below 95% of store-days; the bias is
      approximately the unavailability share, up to 12% in promotion weeks.
   B excludes club, discount, and online channels, 18% of regional volume.
   B is restated for up to three weeks; observed restatement in week 37 was
      -70 cases (-1.0%).
   E is the retailer's book, not a count; residuals of 2-3% of flow are normal
      and are attributed to shrink monthly.

7. FAILURE SIGNATURE AND MONITOR
   Signature: POS falls while channel inventory rises for two consecutive weeks.
   That is either a genuine demand fall or a POS feed failure; the bridge
   distinguishes them, because a feed failure leaves the inventory identity
   unexplained by more than 5%.
   Monitor: weekly POS-to-shipment ratio, availability rate, residual as % of
   sell-in. Escalate at residual > 5% or availability < 90%.

8. RECONCILIATION RULE
   Weekly bridge at SKU x DC x week. Residual tolerance 5% of sell-in. Residuals
   inside tolerance are attributed to shrink and logged. Residuals outside are
   escalated to the retailer's replenishment contact within 48 hours and the
   week is flagged in the demand history.

9. REVISION POLICY
   POS and inventory are captured bitemporally. Every model and report records
   the as-of timestamp used. Backtests use the value known at decision time.
   Restatements above 2% of a week's volume are logged with their cause.

10. REVIEW TRIGGERS
   Annual review, or earlier on: channel mix shift above 2 percentage points;
   POS coverage below 70%; retailer feed change; promotion calendar change; any
   restatement above 5% of a week's volume.
```

### 6.7 What the analyst should have asked

1. Is the 82% coverage measured on volume or on store count? A feed covering 82% of stores can
   cover 70% of volume or 90%, depending on which stores are missing.
2. Does the retailer's store inventory cover the same 82% as its POS feed? If it is broader, the
   store-level bridge compares different populations and the 60-case residual is not a residual.
3. Is the 88% availability measured in store-days or in volume-weighted store-days? The difference
   changes the censoring adjustment materially.
4. Was the promotion calendar known at the Wednesday cut-off, or did it change afterwards? A later
   change is a restatement of the decision context, not of the data.

## 7. Code, query, or calculation

One query, because the point is conceptual. It returns week-37 sell-out as it was known at the
week-38 order cut-off. The `pos_daily` table is bitemporal: `valid_date` is the day the row
describes, and `txn_from` and `txn_to` bound the period during which that version was current.

```sql
-- Week-37 sell-out as known at the 2026-09-16 14:00 order cut-off.
with as_known_then as (
    select units
    from retail.pos_daily
    where sku = 'AURORA-6PK'
      and region = 'NORTH'
      and valid_date between date '2026-09-07' and date '2026-09-13'
      and txn_from <= timestamp '2026-09-16 14:00:00'
      and (txn_to > timestamp '2026-09-16 14:00:00' or txn_to is null)
)
select sum(units) as week_37_units,
       count(*)   as days_reported
from as_known_then;
```

Expected output: `6950, 7`. Replace the literal timestamp with `now()` and the same query returns
6,880, because the retailer restated the week down by 70 cases. That difference is the restatement,
and it is invisible in a table that overwrites in place.

Three checks before trusting such a query.

- **Run it at two cut-offs.** If the answer is identical, either the source does not restate or the
  table is not bitemporal. Both are worth knowing.
- **Check the day count.** `days_reported` below seven means the week was incomplete at the
  cut-off, and the total is a partial week rather than a low week.
- **Check the valid-time boundary.** A daily table needs it; a weekly aggregate keyed on a week
  identifier does not. Applying transaction time without valid time returns every day ever
  recorded, filtered by version.

## 8. Data, standards, and regulatory context

Three families of standard matter: provenance, contracts, and temporal representation.

### 8.1 Provenance

| Standard | What it defines | Useful idea | Grade |
|---|---|---|---|
| W3C PROV-DM, *The PROV Data Model* (Recommendation, 2013) | Provenance as entities, activities, and agents, with relations such as `wasGeneratedBy`, `used`, and `wasDerivedFrom` | Provenance as a graph of who did what to which object | B |
| W3C PROV-O | An OWL ontology for the same model | Machine-readable provenance a catalogue can query | B |

PROV answers the question a source-selection record needs answered: where the value came from,
through what process, under whose responsibility. It does not answer whether the value is true,
and it does not rank sources. It makes the ranking auditable.

The research literature treats provenance as a prerequisite for computational reproducibility
rather than a compliance artefact. The End-to-End Provenance Project, reported in *Patterns* in
2020, describes tools for capturing and querying execution provenance, with the explicit goals of
reproducibility and time-travel debugging. Ursprung, described at SIGMOD 2019, addresses the same
problem at scale with event hierarchies and capture rules. Both make the same practical point:
provenance captured after the fact is reconstruction, and reconstruction is unreliable.

### 8.2 Data contracts and service levels

| Standard or framework | What it defines | Useful idea | Grade |
|---|---|---|---|
| Open Data Contract Standard (ODCS) v3.2.0, Bitol / LF AI and Data | Contract sections for schema, quality, service levels, ownership, support | A contract is a versioned, testable agreement between producer and consumers | C |
| DataHub data contract model | Contracts built on schema, freshness, and data-quality assertions | Enforcement is not automatic; it needs assertion evaluation plus alerting or CI | D |
| Open Data Product Operating Framework v0.2.0 (draft) | Operating model including commitments and service assurance | Declared state and observed state are separate governance signals | C |

ODCS lists service-level properties that map onto the axes in section 5.7: availability,
throughput, error rate, latency or freshness, frequency, retention, time to detect, time to
notify, and time to repair. It also allows a driver such as regulatory, analytics, or operational,
which is the contract's way of saying the target depends on the use.

The framework document states the principle most relevant here: an SLA is not evidence of service
performance. It gives the example of a declared completeness of at least 98% against an observed
94.7%, which is a failed commitment even though the declaration is intact. The same distinction
applies to a source-selection record. The record declares authority. Only measurement shows
whether the source behaves as described.

> **Note:** ODCS is a consortium standard with real adoption, but it is not regulated, and its
> grade reflects that. DataHub's documentation is product documentation. Neither is evidence that
> contracts improve data quality. They specify a format, not an outcome.

### 8.3 Temporal representation

| Specification | What it defines | Relevance | Grade |
|---|---|---|---|
| SQL:2011 system-versioned tables | Standard syntax for transaction-time tables with system-maintained periods | The portable way to keep history without application logic | B |
| TSQL2 specification (Snodgrass et al., 1994) | Valid-time, transaction-time, and bitemporal semantics | The canonical statement of the two dimensions | A |
| Snodgrass, *Developing Time-Oriented Database Applications in SQL* (1999) | Implementation patterns in plain SQL | Constraints, keys, and as-of queries | A |

Transaction time is append-only: a correction adds a version and closes the previous one, and the
previous one remains queryable. A table with a single `updated_at` column records when the row
last changed and destroys what it said before. That is enough for operational support and not
enough for a backtest.

### 8.4 What regulation requires, and what it does not

Financial reporting controls require that a reported number be traceable to supporting records
and that the records be retained. That is a requirement for provenance and retention, not for a
particular source, schema, or ranking.

Regulation constrains the evidence a decision must produce, not the choice of source. A company
can satisfy an audit by showing that an invoiced quantity traces to a dispatch record and a
receipt. It cannot satisfy an audit by showing that the demand forecast was accurate, because
accuracy is not the standard.

The record is therefore an audit artefact as well as an analytical one. Keep it with the version
of the data it describes, and keep the as-of timestamps with it; a record that cites a source but
not a snapshot cannot be reconstructed after the source restates.

### 8.5 What is convention rather than requirement

- "Single source of truth" is a management slogan with no standard behind it.
- "The data warehouse is the source of record" is a convention about architecture. A warehouse is
  L4 or lower for every field it holds, because it is derived.
- "POS is demand" is a convention from retail analytics, censored by availability and bounded by
  coverage.
- "Three sources agreeing means it is true" fails when the sources share an ancestor.
- Weekly planning cycles and Monday data freezes are habits. They set the latency requirement, and
  they can be changed.

An analyst who can separate a requirement from a convention can ask whether the convention still
serves the decision. One who cannot will treat the current architecture as a constraint of nature.

## 9. Industry practice and case evidence

### 9.1 POS versus shipments: what the empirical work says

A study in the *Journal of Operations Management* (2019) compared forecast accuracy using three
demand signals in a large consumer-products chain: POS data, retailer orders, and distribution
centre orders. Its finding is more specific than the consulting claim it tests. POS data improved
forecast accuracy for demand planning and did not improve it for order-fulfilment planning. The two
tasks have different targets, and the signal that is best for one is not best for the other.
(Grade A.)

A second study, in the *Journal of Business Logistics* (2014), shows that POS data is itself
influenced by non-demand factors including stockouts, shelf placement, and promotions. The
implication is not that POS is unreliable. It is that POS measures sales conditional on
availability and merchandising, which is what the censoring adjustment tries to undo. (Grade A.)

Together the two results support the field-level framing. "Which source is better" has no answer.
"Which source is better for this field and this decision" has an answer, and it can differ between
two decisions taken in the same week by the same team.

### 9.2 Why the numbers differ in practice

| Cause | Mechanism | What it looks like | Grade |
|---|---|---|---|
| Pipeline timing | Goods move manufacturer to retailer DC to store over days or weeks | Shipments lead sales by a lag that varies with policy | C |
| Coverage gaps | Panels and retailer feeds exclude channels | A stable ratio between two series that is not 1.0 | C |
| Definition mismatch | Category, item, or status definitions differ | Gaps concentrated in specific items or periods | C |
| Unit mismatch | Cases, eaches, pallets, litres | Gaps that scale with pack configuration | C |
| Inventory build | Channel stocks rise or fall | Gaps that reverse over the promotion cycle | C |

The practitioner literature states these consistently. A retail measurement note lists pipeline
timing, coverage gaps, definition mismatches, and unit mismatches as the four standard reasons a
shipment number does not match a syndicated sales number. A distributor guide states the channel
balance as closing equals opening plus sell-in minus sell-out plus adjustments, the identity used
in section 6.3. An IMD note (2024) adds that sell-out is expensive, arrives in differing week
definitions, contains item-coding errors, and requires reconstructing the retailer's network.

These are practitioner sources, graded C or D, useful for mechanism rather than magnitude.

### 9.3 Inventory record inaccuracy

If the channel inventory term in the bridge is wrong, the bridge closes to a false residual. The
evidence on inventory record accuracy is therefore directly relevant.

A peer-reviewed simulation study in *Production* (2023) models inventory record inaccuracy across
five distribution-centre categories, separating ghost from hidden inventory. It reports that
without cycle counting, inaccuracy compounds over time, and that cycle counting reduces it in
every category, with required effort varying substantially by warehouse type. It explicitly
discourages relying on an annual full count. (Grade A for method; the magnitudes are simulation
outputs.)

A 2026 master's thesis at the University of Twente reports a full count in a wholesale warehouse:
616 SKUs and 53,678 units, discrepancies in 69.16% of SKUs, aggregate magnitude 33.62%, and record
accuracy of 30.84%, with most errors being ghost inventory in fast-moving items. The root cause was
no verification at the point of execution combined with paper-mediated, delayed registration in
the enterprise resource planning (ERP) system. (Grade C: a single case, not peer-reviewed.)

The case is not a benchmark. It is a reminder that a retailer's inventory report, L4 on the
ladder, can be materially wrong in a specific warehouse, and that the bridge residual is the only
signal that reveals it. An ERP-to-WMS reconciliation paper in a low-profile journal reports the
usual drivers: batch versus real-time sync, incompatible unit and location definitions,
transaction timing, and manual adjustments. Its reference list appears unreliable, so use the
driver list only as a checklist.

### 9.4 The ASN as a claim

The advance ship notice is the clearest everyday example of an L3 source, and the operational
documentation makes the status explicit.

| Source | What it documents | Grade |
|---|---|---|
| Oracle Purchasing documentation on ASNs | The ASN is validated in the receiving open interface; discrepancies generate an EDI 824 to the supplier | D |
| Kroger vendor ASN requirements | Suppliers may bypass manual verification only after 100% accuracy on licence plates, case codes, quantities, and expiry dates, with periodic audits | C |
| Magna EDI 856 guideline (v3.0, 2012) | Corrections are not permitted; the whole ASN must be resent and its number must match the packing slip and invoice | C |
| GS1 US case study, Dot Foods | A supplier data-accuracy scorecard for ASN completeness and accuracy gated automated receiving | C |

Two things follow. The ASN is a claim that receiving is designed to test, with a standard error
message, and the claim is commercially consequential: it feeds a supplier compliance score, which
is an incentive to report complete and on time. That is why a supplier with a perfect ASN score is
not thereby a supplier whose shipments match their notices.

### 9.5 The single-source-of-truth critique

The critique is consistent across practitioner sources, all vendor or consultancy content and all
graded D.

Profisee's chief data officer argues in a May 2025 post that truth is contextually bound, that a
business necessarily has multiple contexts and that within one context there can be only one
truth. Sifflet's interviews with data leaders report that a single source is unattainable and that
the achievable goal is agreement on which definition is authoritative for a given use. Customer.io
traces the phrase to mid-1990s enterprise software marketing.

The convergent claim, stripped of marketing, is the one this topic adopts: authority is a property
of a field in a context, not of a system.

### 9.6 Evidence hierarchies in forecasting

Forecasting has an unusually explicit evidence hierarchy, and it is worth borrowing.

The M-competitions, from the 1980s through M4 in 2019, established a norm of evaluation against
specified benchmarks on large, disclosed samples with rolling origins. M3 (2000) reported that
relatively simple methods were competitive with complex ones. M4 (2019) reported that
machine-learning and hybrid methods, and combinations, outperformed simple benchmarks on a large
and heterogeneous set. A 2021 review of competition design argues that benchmarks should be fully
specified and used to measure forecast value added, and that replicability should be a design
requirement. (All Grade A; the design paper is a preprint, A-.)

The transferable discipline is threefold: specify the benchmark before the evaluation, disclose
the sample and period, and report what the method adds over the benchmark rather than only its
standalone error. A source-selection record applies the same discipline to data.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Source chosen per system, not per field | "We use the ERP" answers every question | One system holds fields at different rungs | Wrong source for the contested field | Field-level source map per decision |
| Shipments treated as demand | Order rule fed from dispatch history | Pipeline fill and order policy enter the signal | Over-order after every promotion | Bridge sell-in to sell-out before use |
| POS treated as clean demand | Forecast error rises during promotions | Censoring, forward buying, channel exclusions | Systematic under- or over-forecast | Availability and forward-buy adjustment |
| Single source of truth declared | One number, several silent workarounds | Collapses fields with different contexts | Shadow reconciliation, no provenance | One source of record per field |
| Conflict resolved by seniority | No written precedence rule | The meeting decides, not the method | Decision quality tracks hierarchy | Publish precedence rules with identifiers |
| Residual plugged | Adjustment line labelled "other" | Bridge does not close and nobody investigates | Real shrink or feed failure hidden | Residual tolerance and escalation path |
| Snapshot overwritten | Backtest cannot be reproduced | In-place updates destroy transaction history | Models trained on the future | Bitemporal capture and as-of queries |
| Triangulation mistaken for validation | "Three sources agree" | Shared upstream feed | Consensus around one bias | Count independent causes |
| Panel treated as census | Universe definition never read | Sample projection and coverage exclusions | Total market wrong by the excluded share | Read the methodology; state coverage |
| ASN treated as receipt | Receiving discrepancies found late | A claim is used as an observation | Inventory and compliance errors | Reconcile at receipt; track the 824 rate |
| Monthly source used for daily decision | Planners keep a private spreadsheet | Grain and latency mismatch | Late, coarse recommendations | Match grain and latency to cadence |
| Record written once, never reviewed | Review trigger blank or "annual" | Channel mix and feeds change | Silent obsolescence | Named triggers and a review owner |
| Restatement enters training silently | Backtest stable, production drifts | Current value differs from value known then | Reproducibility loss | As-of training sets and a revision register |
| Contract has no observed-state check | Declared complete, measured incomplete | Declaration is not measurement | Contract becomes decoration | Compare declared and observed monthly |

Four deserve a note.

**Source chosen per system.** The parent failure. It survives because system ownership is
organisational and field ownership is not: a team can name the ERP owner, and almost no team can
name the owner of "consumer takeaway for the North region".

**Residual plugged.** Worse than a large residual, because it removes the signal. The 210-case
residual is small and attributed; a 2,000-case residual labelled "timing" is a suppressed finding.

**Snapshot overwritten.** Invisible until someone tries to reproduce a decision, and by then the
data is gone. It is the only failure here that cannot be remediated retrospectively.

**Record written once.** A control document without a review trigger describes the organisation as
it was when written, and nobody knows when that stopped being true.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision can be stated as an action with a cadence | Tolerance, grain, and latency derive from it | Decision contested or unowned | Ask for the smallest change that alters the action |
| The chosen source's coverage is known | Coverage determines representativeness | Never measured; vendor figure taken on trust | Compute against an independent volume estimate |
| The bridge identity holds over one population | Reconciliation needs comparable populations | Feeds cover different store sets or periods | Compare store lists and cut-offs first |
| Ladder position is stable over the review period | The record describes the source's role | Feed change, new channel, new process | Re-run the placement at each review |
| Revisions are bounded and traceable | As-of queries need a version history | Silent overwrites; unannounced restatements | Inspect version counts; test an as-of query |
| One source can be designated per field | A record needs an authority | Field genuinely has two contexts | Split the field by context, not the source |
| Cost asymmetry is estimable | It sets the insurance the decision carries | Margin and markdown behaviour unknown | Bound it; use the ratio, not a point |
| The analyst can see the decision's cost | Source choice is justified economically | Cost falls on another function | State cost from the decision owner's view |
| Independent sources have independent failure modes | Triangulation requires it | Shared feed, mapping, or vendor | Draw each source's causal graph |
| The record's owner can enforce it | A rule without enforcement is advice | Owner is an analyst without mandate | Name a role with decision rights |

Two limits are worth stating. The ladder is a heuristic, not a measurement: two reasonable
analysts can place a source one rung apart, and its value is that it forces the disagreement into
the open. And the procedure assumes the decision is stable enough to document; for a genuinely
novel decision the first record will be written after the fact, which is acceptable if it is
written.

## 12. Exercises

**Exercise 1 — Build the ladder (analysis, 45 minutes).** Choose one recurring decision and list
five sources that could inform it. Deliverable: a ladder table with level, proximity mechanism,
incentive mechanism, and one field where each source would rank highest. Acceptance: at least one
source is placed differently for two fields; every placement cites a mechanism; one placement
argues against the obvious answer.

**Exercise 2 — Field-level source map (design, 50 minutes).** Decompose the decision into at least
six fields. Deliverable: field, grain, tolerance, freshness, chosen source, rung, and the condition
that would change the choice. Acceptance: no two fields share a source without a reason; each
tolerance is derived; at least one field has a fallback; each field names an owner role.

**Exercise 3 — Reconcile four numbers (calculation, 60 minutes).** For one SKU and week: sell-in
12,400 cases; sell-out 10,850; DC withdrawals 11,200; panel 11,600. Retail DC inventory 3,400
opening and 4,180 closing; store inventory 2,100 opening and 2,340 closing. Deliverable: the full
bridge, the residual, its attribution, and a source recommendation. Acceptance: the identity
closes; residual shown as a share of sell-in against a 5% tolerance; the gap decomposition is in
cases and percentages; the recommendation names a field and a decision, not a system.

**Exercise 4 — Fitness scoring (design, 50 minutes).** Score three candidates on the six axes.
Deliverable: the table, weights and justification, totals, and the critical requirements acting as
gates. Acceptance: each score has a measurement or documented judgement; weights sum to one and
derive from the decision; at least one source is excluded by a gate despite a competitive total;
arithmetic shown.

**Exercise 5 — Bitemporal design (design, 45 minutes).** Design the table that answers what was
known, when, and what is true now. Deliverable: schema with keys and two interval pairs; three
example rows including a restatement; the as-of query. Acceptance: valid and transaction time are
distinct; the restatement closes the previous version rather than deleting it; the query returns
one version per period for any timestamp.

**Exercise 6 — Write the record (design, 60 minutes).** Write a complete record for a decision you
know, using the ten contents in section 5.8. Deliverable: one page with identifier, version, owner,
and triggers. Acceptance: all ten contents present; three rejected candidates with specific
defects; the failure signature names a metric, threshold, and direction; one trigger is event-based;
a stranger could reproduce the choice.

**Exercise 7 — Find the shared ancestor (critique, 40 minutes).** Take a claim that three
independent sources agree, and draw each source's causal graph to its origin. Deliverable: the
graph, a count of independent causes, and a rewritten claim. Acceptance: every source traced
through sample, mapping, and aggregation; at least one shared ancestor identified or its absence
argued; the rewritten claim does not overstate independence.

**Exercise 8 — Restatement audit (analysis, 50 minutes).** Find a series revised after publication
and quantify a year of revisions. Deliverable: distribution of revision size, the count exceeding
the decision tolerance, and the affected models. Acceptance: revisions measured against the value
known at the cut-off, not the previous version; one revision traced to a cause; affected models
named rather than described generically.

## 13. Self-check answers

**Exercise 3 arithmetic.**

```text
DC:    3,400 + 12,400 - 11,200 = 4,600 expected ; 4,180 reported ; residual 420
Store: 2,100 + 11,200 - 10,850 = 2,450 expected ; 2,340 reported ; residual 110
Total: 3,500 + 12,400 - 10,850 = 5,050 expected ; 4,520 reported ; residual 530

sell-in - sell-out = 12,400 - 10,850 = 1,550
inventory build    = 780 + 240 = 1,020
residual           = 420 + 110 = 530
check              = 1,020 + 530 = 1,550

residual / sell-in  = 530 / 12,400 = 4.27%
residual / sell-out = 530 / 10,850 = 4.88%
```

The bridge closes inside tolerance, so the residual is attributed: 65.8% of the gap is channel
inventory and 34.2% is shrink and adjustment. That split is high for grocery; the first check is
whether the DC inventory covers the same stores as the flows. For forecasting, use POS for
takeaway with the availability and forward-buy adjustments, and the panel below 70% coverage.
Sell-in is the worst choice: it contains 1,020 cases of intentional channel build.

**Exercise 4 arithmetic.** With weights grain 0.25, latency 0.20, completeness 0.20, revision
0.15, auditability 0.10, cost 0.10, and scores P (5,4,3,3,4,2), Q (2,5,4,2,5,5), R (4,3,5,5,3,3):

```text
P = 1.25 + 0.80 + 0.60 + 0.45 + 0.40 + 0.20 = 3.70
Q = 0.50 + 1.00 + 0.80 + 0.30 + 0.50 + 0.50 = 3.60
R = 1.00 + 0.60 + 1.00 + 0.75 + 0.30 + 0.30 = 3.95
```

R scores highest and fails the latency gate: a daily 06:00 decision cannot use a source a full
cycle late. Q fails the grain gate. P is chosen at 3.70 because it clears both critical
requirements. Recommending R demonstrates the failure the exercise exists to catch: a weighted
score letting a disqualifying defect be compensated by axes the decision does not value.

**Rubrics.** Exercise 1: a strong answer places a retailer inventory report at L4 for book value
and L3 for a replenishment negotiation, because the same document is a closed record in one context
and an interested claim in the other. Exercise 5: a strong schema has four time columns plus a
revision reason, keys on the transaction start, and shows the first row closed at the restatement
time while remaining retrievable. Exercise 6: a strong record is executable by a stranger and names
a role. Exercise 7: a strong graph separates sample, mapping, and aggregation as ancestors. Exercise
8: a strong audit measures revisions against the cut-off value and separates late arrival from
correction.

## 14. Interview questions

**1. "Two systems report different numbers for the same week. What do you do?"** A strong answer
refuses the premise: establish whether they measure the same field at the same grain and
population, bridge with the inventory identity, then apply a written precedence rule to the
residual. It may conclude both are correct. A weak answer asks which system is more reliable.

**2. "Is POS data demand?"** A strong answer calls it a transaction record of sales conditional on
availability, merchandising, and coverage: the best available observation of shopper behaviour in
many categories, censored by stockouts and inflated by forward buying, and not consumption. A weak
answer says yes because it is closest to the customer.

**3. "Why is an ASN ranked below a goods receipt?"** A strong answer explains that the ASN is
created before arrival by a party whose compliance score and allocation depend on it, while the
receipt is confirmed at physical arrival and tested against the ASN with a standard discrepancy
message. A weak answer says the ASN is less accurate without naming the incentive.

**4. "What does single source of truth mean to you?"** A strong answer reframes it as one source of
record per field per context, explains why a system cannot be authoritative for fields at
different rungs, and gives an example of two legitimate demand numbers. A weak answer endorses or
dismisses the phrase without an alternative.

**5. "How would you reconcile shipments and sales?"** A strong answer writes the identity: sell-in
minus sell-out equals the change in channel inventory plus adjustments. It bridges timing,
coverage, definition, and unit first, and treats a residual outside tolerance as a finding. A weak
answer differences the series and calls the gap channel stuffing or timing.

**6. "Why does bitemporal capture matter for a forecast model?"** A strong answer explains that a
backtest must use values known at each decision time, and that overwriting destroys that ability.
It distinguishes valid from transaction time and gives a restatement example. A weak answer says
history is useful for auditing.

**7. "Complete but coarse, or fine but incomplete?"** A strong answer starts from the decision's
grain and tolerance, applies gates, then compares residual risk, noting that a fine source covering
60% may be ideal for a store-level decision and unusable for total volume. A weak answer picks
completeness because it sounds safer.

**8. "Three sources agree. Are you confident?"** A strong answer asks what shared ancestor would
make them move together and counts independent causes. A weak answer treats agreement as
confirmation.

**9. "What goes in a source-selection record?"** A strong answer names the decision and tolerance,
fields, candidates including rejected ones, choice per field with justification, known defects with
direction and size, failure signature and monitor, reconciliation and revision policy, owner, and
triggers. A weak answer describes a data dictionary.

**10. "How would you detect that a source has started to fail?"** A strong answer names a
directional signature, such as POS falling while channel inventory rises, or a bridge residual
crossing tolerance, with a metric, threshold, and frequency. A weak answer says it would monitor
data quality, which names no mechanism.

## 15. Key terms

- **Source of record** — designated authoritative source for one field in one decision context.
- **System of record** — system accountable for a master or transactional object.
- **Trust ladder** — ranking by proximity to the event and incentive to misreport.
- **Proximity** — transformations and interventions between an event and a value.
- **Sell-in / sell-out / consumption** — into the channel, out at the till, used by the shopper.
- **POS** — checkout system and its transaction record; censored by availability.
- **Syndicated data / panel** — retail measurement from a sample, projected to a universe.
- **Coverage** — share of the decision population represented by a source.
- **Channel inventory** — goods between dispatch and basket.
- **Pipeline fill** — inventory added to support a higher run-rate.
- **Forward buy** — demand pulled from a future period by a commercial incentive.
- **Stockout censoring** — under-observation of demand when the item is unavailable.
- **Bridge** — decomposition explaining the difference between two sources.
- **Residual** — unexplained part of a bridge after attribution.
- **Precedence rule** — written rule for which source wins after bridging.
- **Valid time / transaction time** — when a fact was true; when the database held it.
- **Bitemporal** — storing both dimensions for each version.
- **As-of query** — returns values known at a specified past instant.
- **Restatement** — revision that changes a value already used in a decision.
- **Provenance / lineage** — origin and custody history; technical transformation chain.
- **Data contract** — versioned, testable producer-consumer agreement.
- **Triangulation** — corroboration using sources with independent failure modes.
- **Common-cause correlation** — agreement caused by a shared upstream origin.
- **Failure signature** — observable pattern indicating that a source is failing.
- **Source-selection record** — the documented choice, defects, signature, and owner.

## 16. Sources

**Peer-reviewed and academic**

- "Demand and order-fulfillment planning: The impact of point-of-sale data, retailer orders and
  distribution center orders on forecast accuracy." *Journal of Operations Management*, 2019.
  DOI: 10.1002/joom.1026. POS improved demand-planning accuracy but not order-fulfilment accuracy.
  (A)
- Zondag, M. M., and Ferrin, B. G. (2014). "Finding the True Voice of the Customer in CPG Supply
  Chains." *Journal of Business Logistics*. DOI: 10.1111/jbl.12055. POS influenced by stockouts,
  shelf placement, and promotions. (A)
- Destro, I. R., Staudt, F. H., Somensi, K., and Taboada, C. (2023). "The impacts of inventory
  record inaccuracy and cycle counting on distribution center performance." *Production*, 33,
  e20220077. (A for method; magnitudes are simulation outputs)
- Snodgrass, R. T. (1999). *Developing Time-Oriented Database Applications in SQL*. Morgan
  Kaufmann. Valid-time, transaction-time, and bitemporal patterns. (A)
- Snodgrass, R. T., et al. (1994). "A TSQL2 Language Specification." Append-only transaction-time
  relations. (A)
- "The End-to-End Provenance Project." *Patterns*, 2020. Provenance capture for reproducibility.
  (A)
- Rupprecht, L., Davis, J. C., et al. (2019). "Ursprung: Provenance for Large-Scale Analytics
  Environments." SIGMOD 2019 demonstration. (A)
- Makridakis, S., and Hibon, M. (2000). "The M3-Competition." *International Journal of
  Forecasting*. DOI: 10.1016/S0169-2070(00)00057-1. (A)
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2019). "The M4 Competition: 100,000 time
  series and 61 forecasting methods." *International Journal of Forecasting*.
  DOI: 10.1016/j.ijforecast.2019.04.014. (A)
- Makridakis, S., et al. (2021). "The future of forecasting competitions." arXiv:2102.04879.
  Benchmark specification and forecast value added. (A-)

**Standards and specifications**

- W3C, *PROV-DM: The PROV Data Model*, Recommendation, 2013. https://www.w3.org/TR/prov-dm/ (B)
- W3C, *PROV-O: The PROV Ontology*. Machine-readable provenance. (B)
- SQL:2011, system-versioned tables. Transaction-time history in standard SQL. (B)
- Open Data Contract Standard (ODCS) v3.2.0, Bitol / LF AI and Data.
  https://github.com/bitol-io/open-data-contract-standard (C)
- Open Data Product Operating Framework v0.2.0 (draft). Declared versus observed state.
  https://opendataproducts.org (C)

**Vendor and product documentation**

- DataHub, "Data Contract" metamodel. Contracts as assertions; enforcement is not automatic.
  https://docs.datahub.com (D)
- Oracle, "Advance Shipment Notices (ASNs)," Oracle Purchasing Help. ASN validation and EDI 824
  discrepancy advice. https://docs.oracle.com (D)
- Magna Powertrain, EDI 856 implementation guideline, v3.0, 2012. Corrections require resending the
  full ASN. (C)
- Kroger, "Advance Ship Notice Vendor Requirements." ASN accuracy certification. https://edi.kroger.com (C)
- GS1 US, "Dot Foods Improves Receiving and Meeting Regulations with GS1 Standards." ASN accuracy
  scorecard gating automated receiving. https://www.gs1us.org (C)
- AWS Well-Architected, "Improve traceability with data provenance tracking" (AG.DLM.8). (D)

**Practitioner, consultancy, and press**

- CPG Data Insights, "The Case of the Mismatched Volume: Comparing Shipments and Retail Sales."
  Four standard causes of shipment-versus-sales gaps. Undated; accessed 2026. (C)
- Marksyte, "Why distributor sales do not match retailer sell-out." Channel balance identity and an
  eight-step checklist. Undated; accessed 2026. (D)
- IMD, "The challenges of using true demand through the supply chain," 2024. Cost and messiness of
  sell-out data. https://www.imd.org (C)
- Profisee, "The Myth — And Reality — Of a Single Version of the Truth," May 2025. Truth as
  contextually bound. https://profisee.com (D)
- Sifflet, "Single Source of Truth — A Modern Data Myth?" Interviews with data leaders. Undated;
  accessed 2026. (D)
- Customer.io, "Why the Single Source of Truth is Not the Answer." Multiple sources by use case.
  Undated; accessed 2026. (D)
- Soysal (2026). "Reducing Inventory Record Inaccuracy in a Wholesale Warehouse." MSc thesis,
  University of Twente. 616 SKUs, 30.84% record accuracy. https://essay.utwente.nl (C)

**Note on grading.** Grades follow Module 28: A peer-reviewed or audited, B regulatory or official,
C company case, framework, or government guidance, D vendor or analyst marketing, E press or
anecdote. Preprints are A-. Undated vendor posts are dated by access in 2026. Figures are reported
as the sources state them and have not been independently verified.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_master-data-drift.md`](09_master-data-drift.md)
- Related: [`08_data-quality-dimensions.md`](08_data-quality-dimensions.md)
- Related: [`01_systems-of-record.md`](01_systems-of-record.md)
- Related: [`03_transaction-data.md`](03_transaction-data.md)
- Related: [`../../01_supply_chain_foundations/10_archetypes-and-metric-conflicts.md`](../01_supply_chain_foundations/10_archetypes-and-metric-conflicts.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
