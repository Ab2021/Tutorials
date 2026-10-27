# Master Data

> Module 02 — The Supply Chain Data Landscape
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01
> Version 2026.5

---

## 1. Why this topic exists

Every transaction in a supply chain is a sentence whose nouns are master data. "12 cases of item X
moved from location Y to location Z on date D." The verb is the event. The nouns are the master
data. If a noun resolves to two entities, or to none, the sentence is ambiguous, and no amount of
modelling downstream recovers the meaning.

Master data is therefore not a supporting table. It is the schema of the business, written in
rows. An enterprise resource planning (ERP) system will happily store ten million order lines
against an item master that is wrong, and every one of those lines will be wrong in the same way,
consistently, which is what makes the error hard to see. Bad transaction data is noisy. Bad
master data is *coherently* wrong, and coherent error survives every validation rule you write.

The consequence for analytics is specific and severe:

| What the analyst sees | What actually happened | Why the check did not fire |
|---|---|---|
| Forecast accuracy fell 9 points in one month | A pack size changed from 12 to 9 and the code was kept | Units per case changed; the series is still continuous |
| Aggregate service at 96% | Two duplicate item codes each stocked out on the same week | Aggregation hides the item-level failure |
| Demand looks intermittent for a "slow mover" | Half the demand is booked to a duplicate code | Classification rule trained on a split series |
| Inventory value fell 6% | Obsolescence was written off, not sold | A lifecycle status field is absent |
| Two reports disagree on units shipped | One counts eaches, one counts cases | Unit of measure (UOM) is not stored with the fact |
| Supplier risk score covers 94% of spend | 6% of suppliers have no parent company | Party hierarchy is 40% populated |

None of these is a statistical failure. Each is a master-data failure that presents as a
statistical one, which is why analysts are usually the last people to diagnose it and the first to
be blamed for it.

There is a second reason this topic exists. Master data is where the political economy of a company
becomes visible. Each business unit wants to define its own items, each region its own locations,
each acquired company its own supplier codes. The result encodes the org chart rather than the
supply chain: an analyst who can read the item master can often infer which businesses were
acquired, which regions won autonomy, and which channel was bolted on later.

This topic sits second in Module 02 because it defines the module's vocabulary. Topic 01 established
which system holds which record; this topic establishes what the record *is*. Topic 03 describes the
transactions pointing at these entities, and Topic 09 returns to what happens when master data
changes underneath a forecast.

## 2. Learning outcomes

By the end of this topic you can:

1. **Name** the six master domains — item, location, party, bill of materials, pricing, and
   reference — and **state** what each owns and what it must not own.
2. **Decompose** an item master record into identity, hierarchy, physical attributes, and lifecycle
   fields, and **identify** which of those fields a planning model actually depends on.
3. **Convert** between base unit, inner pack, case, and pallet, and **detect** a conversion error
   from a symptom in the data.
4. **Explain** the three ways a key fails — it splits, it merges, or it dangles — and **classify** a
   described data defect into one of the three.
5. **Design** an entity-resolution pass for a duplicate item set: blocking keys, match rules, score
   thresholds, and a human review queue.
6. **Apply** survivorship rules to a conflict between an ERP and a product information management
   (PIM) system, and **justify** the precedence choice attribute by attribute.
7. **Merge** two duplicate item records without losing history, stock, open orders, or audit
   traceability, and **prove** the merge with control totals.
8. **Specify** a governance control — owner, steward, creation gate, change classification — that
   prevents the duplicate from being created again.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a set of stable noun tables at the top, a stream of event tables at the bottom,
and every event row pointing upward at four or five nouns.

```text
                    +-----------------------------------------------+
                    |               REFERENCE DATA                  |
                    |  calendar | currency | UOM codes | HS tariff  |
                    |  country  | GPC brick | reason codes | tax     |
                    +-----------------------+-----------------------+
                                            |
    +-------------+    +-------------+      |      +-------------+    +-------------+
    |    PARTY    |    |  LOCATION   |      |      |    ITEM     |    |   PRICING   |
    |  supplier   |<-->|  GLN, plant |<-----+----->|  GTIN, SKU, |<-->|  price list |
    |  customer   |    |  DC, store  |             |  hierarchy  |    |  conditions |
    |  hierarchy  |    |  aisle, bin |             |  attributes |    |  validity   |
    +------+------+    +------+------+             +------+------+    +------+------+
           |                  |                           |                  |
           |                  |      +-------------+      |                  |
           |                  +----->|     BOM     |<-----+                  |
           |                         | parent -->  |                        |
           |                         | component   |                        |
           |                         | qty, scrap  |                        |
           |                         +-------------+                        |
           v                        v                       v               v
    +---------------------------------------------------------------------------+
    |                            TRANSACTION DATA                               |
    |  purchase order | sales order | shipment | receipt | invoice | inventory   |
    |  Every row is a sentence. The nouns must resolve to exactly one entity.    |
    +---------------------------------------------------------------------------+
```

Four properties of this picture explain most of what follows.

**Property 1 — Master data is small, reused, and long-lived; transactions are large, single-use,
and short-lived.** A company with 40,000 items may write 400 million transaction rows a year. The
ratio matters: a defect in one master record contaminates every transaction that points at it, and
the contamination is perfectly correlated, so it does not average out.

**Property 2 — The identifier is not the entity.** The item is a physical thing with a weight, a
shelf life, and a temperature requirement. The stock keeping unit (SKU) is a row in a database that
represents the thing *for a purpose*. Two SKUs can represent one thing (duplication); one SKU can
represent several things over time (drift). The purpose is why the identifier exists, and purpose
changes.

**Property 3 — Every domain has a natural external key and a political internal key.** The item has
a Global Trade Item Number (GTIN) from GS1 and an internal SKU from the ERP. The location has a
Global Location Number (GLN) and an internal four-character code. The party has a registered legal
name and a vendor number. The external key is stable and shared; the internal key is convenient and
contested. Systems join on the internal one and report on the external one, and the mapping between
them is where errors live.

**Property 4 — Master data is a set of decisions, not a set of facts.** Whether a 6-pack and a
12-pack are one item or two is a commercial decision about how the product is sold, not a physical
fact about the product. Whether two ship-to addresses are one customer location or two is a
decision about how invoices and rebates are calculated. An analyst who treats these as facts to be
discovered will keep asking a question that has no answer.

### 3.1 The six domains and what each owns

| Domain | Owns | Must not own | Typical system of record | External key |
|---|---|---|---|---|
| Item | Identity, classification, physical attributes, UOM chain, lifecycle status | Price, stock, demand, cost | ERP material master, PIM | GTIN |
| Location | Identity, address, hierarchy, function, time zone, capacity | Inventory quantity, throughput actuals | ERP or dedicated location master | GLN |
| Party | Legal identity, hierarchy, roles, addresses, terms, bank details, certifications | Performance scores, risk ratings | ERP vendor/customer master, CRM | GLN, legal entity identifier |
| Bill of materials (BOM) | Parent–component structure, quantity per, scrap, effectivity, alternates | Routing, capacity, cost roll-up | ERP, PLM | No global standard |
| Pricing | Price lists, conditions, discounts, surcharges, currency, validity | Standard cost, margin | ERP condition tables, pricing engine | No global standard |
| Reference | Calendar, currency rates, UOM codes, country codes, tariff codes, classification codes | Any entity-specific attribute | Shared services, data warehouse | ISO codes, GS1 code lists |

Two of these are contested. **Pricing** is master data in retail and distribution, where a price
list applies to many items and many customers and has its own validity dates; in project-based
manufacturing it behaves more like a transaction. **BOM** is master data in discrete manufacturing
and close to irrelevant in process industries, where the "recipe" lives in a different system with
different change control. State which convention you are using before you build on it.

### 3.2 The three ways a key lies

This is the most useful single idea in the topic. Every master-data defect that damages analytics
is one of three shapes.

| Shape | Definition | Symptom in the data | Symptom in the business | Example |
|---|---|---|---|---|
| **Split** (duplication) | One entity, many keys | Two series each with plausible volume; neither matches the expected total | Planner orders twice; channel P&L wrong | Same bottle created by grocery and e-commerce teams |
| **Merge** (drift) | One key, many entities over time | Series has a step change with no demand explanation | Forecast permanently biased after the change | Pack size changed 12 to 9 under the same code |
| **Dangle** (orphan) | A key with no valid parent, owner, or status | Inner joins silently drop 3–8% of rows | Volume disappears between systems | Item with no product family; location with no region |

A split inflates counts. A merge corrupts time series. A dangle corrupts totals.

The three are not independent. A merge is often *created* by a split: because duplication made the
item master unusable, someone "fixed" it by reusing an old code for a new pack, converting a split
into a merge. A dangle is often created by a merge: the old code was retired and the new one was
never added to the hierarchy table, so all its historical transactions orphan.

> **Note:** the widely repeated claim that duplicate items "double your safety stock" is
> arithmetically false for statistical safety stock. Section 6 shows why: two buffers computed on
> two half-sized demand streams sum to approximately one buffer computed on the whole. The real
> cost of duplication is elsewhere — in fixed buffers, order quantities, substitution failures, and
> spurious trends. Repeating the safety-stock claim in a business case will get the business case
> rejected by the first planner who checks it.

### 3.3 Why the GTIN is not a primary key

Analysts expect the external standard identifier to be the reliable key. It is not: not every item
has one (components, packaging, service items often do not); one product carries several across
packaging levels, so a join without a level qualifier doubles counts; a malformed GTIN that passes
the check digit joins to another company's product; and retailer-specific item numbers coexist with
it, mapped by hand.

Use the GTIN as the *reconciliation* key across companies and a *deduplication* key within one. Use
the internal SKU as the operational key. Keep an explicit cross-reference table, and never assume it
is complete.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Master data | The relatively stable entities that transactions reference: items, locations, parties, BOMs, prices, reference codes | Used to mean any small lookup table | None |
| Transaction data | Records of events with a time, a quantity, and a set of entity references | Used to mean any fact table | None |
| Reference data | Small, shared code sets used to interpret other data: calendars, currencies, UOM codes, tariff codes | Treated as part of master data or ignored entirely | ISO, GS1, or national code lists |
| Golden record | The single agreed representation of an entity, assembled from multiple sources under stated rules | Assumed to be the most complete record | None |
| Survivorship | The rules deciding which source wins for each attribute when sources disagree | Implemented as "last write wins" | Rule per attribute |
| Master data management (MDM) | The discipline and tooling for creating, governing, and distributing master data | Used to mean a software purchase | None |
| Product information management (PIM) | The application that holds and enriches product content for commercial channels | Confused with MDM | None |
| Global Data Synchronisation Network (GDSN) | GS1's network of certified data pools for synchronising item and party data between trading partners | Treated as a single database | GS1 GDSN release |
| Data pool | A GDSN-certified intermediary that validates and exchanges item data | Confused with the GDSN itself | Certified by GS1 |
| GTIN | Global Trade Item Number: a globally unique identifier for a trade item at a specified packaging level | Used as a synonym for SKU | 8, 12, 13, or 14 digits |
| GS1 Company Prefix | The licensed prefix that makes a GTIN globally unique to its owner | Confused with the full GTIN | 6–12 digits |
| Check digit | The final digit of a GTIN, computed by a mod-10 algorithm | Assumed to prove authenticity | Single digit |
| SKU | Stock keeping unit: the internal identifier for an item as managed at a location or in a channel | Used as a synonym for product | Internal alphanumeric |
| Item master | The system record holding identity, attributes, UOM, and lifecycle for an item | Assumed to be one system | Record |
| Item hierarchy | The classification tree above the item: division, family, variant, pack | Confused with the packaging hierarchy | Internal levels |
| Base unit | The lowest unit in which an item is counted, usually the each | Confused with the unit of sale | Each |
| Inner pack / case / pallet | Packaging levels above the base unit, each with its own quantity and often its own GTIN | Collapsed into "case" loosely | Count per level |
| TI / HI | Tiers (layers) and high (units per layer) describing a pallet configuration | Treated as a physical constant | Integers |
| Unit of measure (UOM) | The unit in which a quantity is expressed | Stored implicitly rather than explicitly | Code list |
| Conversion factor | The multiplier between two UOMs, for example 24 eaches per case | Keyed once and never re-verified | Ratio |
| Catch weight | The practice of pricing by actual weight rather than nominal pack weight | Ignored in unit-based planning | Kilograms |
| Lot / batch | A production or receipt group that shares a manufacturing history | Confused with serial number | Alphanumeric |
| Serial number | A unique identifier for one physical unit | Used for lots | Alphanumeric |
| Shelf life | The period between production and the last usable date | Confused with remaining shelf life at receipt | Days |
| Temperature class | The storage and transport range an item requires | Encoded as free text | Degrees Celsius band |
| Hazardous material (hazmat) code | The UN number and class governing transport of a dangerous good | Stored only in a safety document | UN number plus class |
| GLN | Global Location Number: a globally unique identifier for a party or physical location | Used only for warehouses | 13 digits |
| SGLN | A GLN extended with an extension component to identify a sub-location such as a bin | Confused with the GLN | GLN plus extension |
| Location hierarchy | The tree above a stocking point: region, distribution centre, zone, aisle, bin | Confused with the party hierarchy | Internal levels |
| Virtual location | A location that holds no physical stock but represents a state, such as quality hold or scrap | Omitted from inventory models | Internal code |
| In-transit location | A virtual location representing goods owned but not yet received | Usually missing from the master | Internal code |
| Party master | The record for a legal or commercial entity: supplier, customer, carrier, or service provider | Assumed to be one record per company | Record |
| Sold-to / ship-to / bill-to / payer | Four roles a party can play in one transaction, each potentially a different GLN | Collapsed into "customer" | GLN per role |
| Parent–child hierarchy (party) | The ownership or control structure above a party, used for spend and risk roll-up | Populated only for the largest suppliers | GLN or legal entity identifier |
| Bill of materials (BOM) | The structured list of components and quantities required to make a parent item | Confused with a recipe or a routing | Record set |
| Phantom item | A BOM node with no stock and no orderability, exploded through during planning | Treated as a real item | Item flag |
| Effectivity dating | Start and end dates that determine when a master record version applies | Assumed to be immediate | Date range |
| Data owner | The role accountable for the correctness and definition of a data domain | Confused with the system owner | Role |
| Data steward | The person who operates the creation and change process day to day | Confused with the owner | Role |
| Change control | The defined path by which a master record may be altered, with approval and audit | Replaced by direct table edits | Process |
| Entity resolution | The process of deciding whether two records refer to the same real-world entity | Used to mean exact matching | Process |
| Blocking | Partitioning records into candidate groups so that not every pair must be compared | Omitted, causing quadratic cost | Key function |
| Deterministic match | A match declared when exact keys agree, such as GTIN plus pack count | Used alone on dirty data | Rule |
| Fuzzy match | A match declared when a similarity score exceeds a threshold | Used without human review | Score 0–1 |
| Orphan record | A record whose parent, owner, or status reference does not resolve | Ignored until a report breaks | Referential check |
| Drift | Change in what a key means over time, without a change in the key | Confused with data quality decay | Time series |
| Duplicate proliferation | The accumulation of redundant records through repeated local creation | Treated as a one-off cleanup | Count and rate |
| Cross-reference table | The mapping between external and internal identifiers, and between retired and current codes | Assumed to be complete | Table |
| Lifecycle status | The state of an entity: proposed, active, blocked, discontinued, replaced | Absent in many item masters | Status code |

Four of these deserve expansion, because each is routinely misused in a way that costs money.

**GTIN versus SKU.** The GTIN is a property of the product as sold. The SKU is a property of the
product as *managed*. A company selling the same bottle through retail and direct-to-consumer may
hold one GTIN and two SKUs, or two GTINs (if the pack differs) and two SKUs, or two GTINs and one
SKU. There is no universal answer. The question "how many items do we have" is therefore
unanswerable until you say whether you mean GTINs, SKUs, or physical products, and the three counts
can differ by 30% in a mid-sized consumer business.

**Packaging level versus product hierarchy.** A case of 12 and an each of the same bottle are the
same product at two packaging levels. A 500 ml bottle and a 750 ml bottle are two products in the
same family. These are orthogonal dimensions, and conflating them is the single most common cause
of doubled quantities in a warehouse: a case GTIN mapped to an each-level item receives as one unit.

**Virtual location versus physical location.** A quality-hold location, a scrap location, and an
in-transit location hold real inventory value but no physical space. Systems that model only
physical locations lose the goods during the period they are in one of these states, which is
exactly the period when someone is asking where they are.

**Owner versus steward.** The owner is a role with budget authority and definitional authority: the
person who decides whether a 6-pack is a separate item. The steward is the person who executes. An
organisation that appoints stewards without owners has created a queue with no decision rights, and
the queue will be bypassed by anyone in a hurry — which is how duplicates are born.

## 5. The mechanics, step by step

This section follows one record through its life, to show where control is exercised and lost.

### Phase A — Design

**Step 1. Define the domain model.**

- Output: which entity types exist, what each owns, and the system of record for each.
- Key branch: **product-centric or SKU-centric?** A product-centric model holds one record per
  physical product plus its commercial variants; a SKU-centric model holds one per sellable unit.
  Most ERPs are SKU-centric, most PIMs product-centric, and the mapping is a permanent cost.
- Failure if skipped: every later decision is local, and the model records who joined when.

**Step 2. Choose the identifier scheme.**

- Output: a numbering convention, a number range per item type, and a rule for when a new number
  is required.
- Key branch: **significant or non-significant numbers?** A significant number encodes meaning
  (family, size, colour), is readable by humans, becomes wrong when the meaning changes, and tempts
  people to guess. A sequential number is stable, opaque, and forces a lookup.
- Failure if skipped: intelligent numbers get reused for near-matches, a merge defect by design.

**Step 3. Specify the attribute set and its validation.**

- Output: a data dictionary giving, per attribute, a definition, a type, a code list, whether it is
  mandatory, and its owner.
- Key branch: **which attributes are planning-critical?** Weight, volume, base unit, conversion
  factors, shelf life, lot control, serialisation, temperature class, and hazmat flag. Everything
  else can be imperfect without breaking a model.
- Failure if skipped: attributes are added by whoever asks loudest.

### Phase B — Create

**Step 4. The request is raised.**

- Artifact: a creation request with requester, justification, and proposed attributes.
- Key branch: **is the requester also the approver?** In most companies, yes, for the categories
  nobody watches. This is the root cause of duplicate proliferation, and no tool fixes it.
- Failure if skipped: records appear with no provenance.

**Step 5. The duplicate check runs.**

- Artifact: a candidate list scored by similarity, with the blocking key shown.
- Key branch: **mandatory or advisory?** An advisory check is a warning dialogue that a user in a
  hurry dismisses. A mandatory check requiring a documented override reason is the only version
  that works.
- Failure if skipped: the duplicate rate is set by user patience, which is zero near quarter end.

**Step 6. Approval and activation.**

- Artifact: an approval record with the approver's role, timestamp, and effective date.
- Key branch: **approval by role or by person?** Person-based approval breaks when the person
  leaves; role-based approval survives, provided the role has a named holder.
- Failure if skipped: activation is a database insert with no accountability trail.

### Phase C — Publish

**Step 7. The record is distributed.**

- Artifact: a distribution log showing which systems received which version, when.
- Key branch: **push or pull?** Push propagates changes immediately and can break a downstream
  system mid-run; pull lets consumers choose and produces divergence. Most landscapes use both and
  document neither.
- Failure if skipped: the ERP, warehouse management system (WMS), and PIM hold three versions, and
  the golden record is whichever one the report read.

**Step 8. External syndication.**

- Artifact: a GDSN publication with a target market and a validation status.
- Key branch: **which market's local data model applies?** The GS1 Global Data Model (GDM) has a
  global core plus local layers, and a US attribute set is not a German one.
- Failure if skipped: the retailer rejects the listing for a missing local attribute, at launch.

### Phase D — Change

**Step 9. The change is classified.**

- Artifact: a change record with a class and a decision.
- Key branch: **does this require a new GTIN?** The GTIN Management Standard gives ten rules and
  three guiding principles. Declared net content, dimensional or gross weight change beyond 20% on
  any axis, pack or case quantity, and primary brand all require one. Marketing copy does not.
- Failure if skipped: a new pack inherits an old GTIN and every historical comparison silently
  becomes invalid.

**Step 10. The change is applied with effectivity.**

- Artifact: a dated supersession record with old and new values and the dates each applies.
- Key branch: **immediate or effective-dated?** Immediate changes break any process running across
  the boundary. Effective dating lets a forecast be restated, but only if history is kept.
- Failure if skipped: the series holds two different products under one label.

### Phase E — Retire

**Step 11. The record is retired, not deleted.**

- Artifact: a lifecycle status change to discontinued, plus a successor pointer.
- Key branch: **is deletion possible?** It must not be. Deletion breaks referential integrity in
  every historical fact table, breaks recall traceability, and makes the audit trail
  unreconstructable.
- Failure if skipped: the item stays forecastable and accumulates phantom demand.

**Step 12. The successor is linked.**

- Artifact: a cross-reference row mapping the retired code to its replacement.
- Key branch: **is the substitution one-to-one?** A pack change from 12 to 9 is not: one old case
  equals 0.75 new cases, so restatement must convert units, not relabel them.
- Failure if skipped: history is dropped, or appended unconverted as apparent demand growth.

### Phase F — Reconcile

**Step 13. The golden record is assembled.**

- Artifact: a survivorship matrix, a match log, and a list of unresolved conflicts.
- Key branch: **which system wins for which attribute?** Never "which system wins". Precedence is
  per attribute and is a business decision.
- Failure if skipped: survivorship defaults to the last system to write, which is the system with
  the most recent integration bug.

**Step 14. The control totals are checked.**

- Artifact: a reconciliation report of entity counts and quantity totals before and after a merge.
- Key branch: **what is the control total?** For an item merge, units in history and units on hand
  must be unchanged; for a party merge, open receivables and payables must be unchanged. A merge
  that changes a total is a data loss.
- Failure if skipped: merges destroy history and nobody notices until the audit.

### 5.1 The creation gate: five checks before any new item exists

A workable gate is short enough to be used and strict enough to matter.

| # | Check | Pass condition | Override requires |
|---:|---|---|---|
| 1 | Exact match on GTIN at the same packaging level | No existing active record | Category owner plus documented reason |
| 2 | Fuzzy match above threshold on normalised description plus pack count | No candidate above 0.90 | Steward review of top five candidates |
| 3 | Pack count and base unit consistent with the proposed UOM chain | Conversion factors multiply to the case and pallet counts | Supply chain owner |
| 4 | Physical attributes within tolerance of an existing sibling | Weight and volume within 5% or a stated reason | Engineering or quality |
| 5 | Classification path exists to a leaf in the product hierarchy | No orphan category | Category manager |

Check 2 is where most duplicates are caught, and check 5 is where most dangles are prevented.

### 5.2 Change classification

| Change | New GTIN required | History restatement | Planning impact |
|---|---|---|---|
| Marketing copy, images | No | None | None |
| Declared net content | Yes | Unit conversion required | Demand per unit changes |
| Gross weight change over 20% on any axis | Yes | None if units unchanged | Transport and storage cost |
| Pack or case quantity | Yes | Unit conversion required | Order multiples, MOQ, pallet counts |
| Primary brand | Yes | None | Pricing and listing |
| New certification mark | Yes | None | Compliance only |
| Supplier changed, same specification | No | None | Origin, lead time, duty |
| Shelf life extended | No | None | Write-off and rotation policy |
| BOM component substitution | No | None | Cost and supply risk |

The middle column is the one companies skip. A GTIN change without a history restatement produces
two short series that are arithmetically the same product, and the forecaster treats them as two
products, one of which is new and therefore has no history.

### 5.3 How entity resolution works in principle

Entity resolution is the general problem of deciding whether two records denote the same thing. For
item masters the pipeline has five stages, and each stage has a characteristic failure.

```text
  raw records
      |
      v
  [1] NORMALISE   strip punctuation, expand abbreviations, standardise UOM tokens
      |           failure: over-normalising merges genuinely different items
      v
  [2] BLOCK       group by cheap key: brand, pack count, size bucket, first token
      |           failure: wrong block key puts true duplicates in different blocks
      v
  [3] COMPARE     score pairs inside each block: string similarity + attribute agreement
      |           failure: threshold set by intuition rather than by a labelled sample
      v
  [4] DECIDE      auto-merge above high threshold, auto-reject below low, review between
      |           failure: no review band, so every decision is automated
      v
  [5] CLUSTER     build transitive groups and choose a surviving record
                  failure: chained matches merge A-B and B-C into a group where A and C differ
```

Stage 3 needs a score. The standard building blocks are:

| Method | Mechanism | Good for | Weakness |
|---|---|---|---|
| Exact match | Equality on a key | GTIN, pack count, brand | Useless when the key is missing or dirty |
| Levenshtein distance | Minimum single-character edits between strings | Typos, transpositions | Ignores word order and token semantics |
| Jaro–Winkler | Character-match score with a bonus for a shared prefix | Short strings, names, codes | Over-weights prefixes that are boilerplate |
| Token cosine | Cosine similarity over token sets | Long descriptions with reordered words | Insensitive to numeric differences |
| Soundex / phonetic | Codes words by consonant skeleton | Names heard and typed phonetically | Poor on numbers and product codes |
| Probabilistic (Fellegi–Sunter) | Likelihood ratio from per-field agreement and disagreement weights | Mixed-quality records with labelled pairs | Requires estimated match and non-match distributions |

The published AWS Entity Resolution service, in a July 2025 release, supports Levenshtein, cosine,
and Soundex with user-set thresholds, and does not offer Jaro–Winkler. That is a useful reminder
that "entity resolution" names a family of methods, not a single algorithm, and that the choice
matters: a Jaro–Winkler score of 0.92 between "PAUL" and "PUAL" is not the same evidence as a
Levenshtein distance of 2.

Blocking is the stage that determines whether the project is feasible. Comparing every pair of
40,000 items is 800 million comparisons; blocking on brand plus pack count reduces that by three
orders of magnitude. The cost of blocking is false negatives: a true duplicate whose pack count was
keyed as 12 on one record and 1 on the other will never be compared. Choose block keys that are
robust to the errors you expect, and run a second pass with a different block key if recall matters
more than cost.

### 5.4 Survivorship when the ERP and the PIM disagree

The golden record is assembled attribute by attribute. A defensible matrix looks like this.

| Attribute | System of record | Rationale | Conflict rule |
|---|---|---|---|
| Internal SKU | ERP | Allocations and history key on it | Never overridden |
| GTIN | PIM / GDSN publication | Only the brand owner may allocate | PIM wins; ERP mirrors |
| Planning description | ERP | Short text used by planners and reports | ERP wins |
| Channel description | PIM | Written for the market | PIM wins |
| Standard cost | ERP costing | Finance owns valuation | ERP wins |
| Gross weight and dimensions | PIM | Measured at packaging | PIM wins if measured after ERP entry |
| Shelf life | Quality system | Regulatory and stability data | Quality wins over both |
| Temperature class | Quality system | Compliance attribute | Quality wins |
| Lifecycle status | ERP | Blocking is an operational control | ERP wins; PIM mirrors |
| Country of origin | ERP customs data | Declarant liability | ERP wins |

The rule to teach is not the matrix. The rule is that **precedence is per attribute and is a
business decision with a named owner**. A program that says "the PIM is the golden record" will
overwrite standard costs with marketing data within a quarter.

## 6. Worked example

Halden Beverages is a mid-sized drinks manufacturer. Its hero product is a 500 ml sparkling water
sold in cases of 12. In the year the direct-to-consumer (DTC) channel launched, two item records
were created for the same physical product. All figures are illustrative and internally consistent.

### 6.1 The two records

| Field | Record A | Record B |
|---|---|---|
| SKU | 10048213 | 10051907 |
| Description | SPARKLING WATER 500ML 12PK | Spkl Water 500ml x12 (DTC) |
| GTIN | 5012345678900 | Blank (entered as 0000000000000) |
| Base unit | EA | EA |
| Cases per pallet | 60 | 60 |
| Created by | Grocery channel team | DTC operations team |
| Created | March 2024 | August 2024 |
| Lifecycle status | Active | Active |
| Standard cost per case | EUR 17.20 | EUR 19.80 (broken-case buying) |

Record B was created because the DTC team needed an item to configure a fulfilment rule and the
grocery team's record was locked while a price condition was under review. This is the ordinary
mechanism. Nobody made a bad decision; the process made the good decision expensive.

### 6.2 The split demand history

Twelve months of demand, in cases. The split ratio drifts because DTC is growing.

| Month | Total demand | Record A | Record B | A share |
|---|---:|---:|---:|---:|
| M1 | 5,200 | 3,900 | 1,300 | 75.0% |
| M2 | 4,800 | 3,504 | 1,296 | 73.0% |
| M3 | 5,000 | 3,500 | 1,500 | 70.0% |
| M4 | 5,400 | 3,672 | 1,728 | 68.0% |
| M5 | 5,600 | 3,640 | 1,960 | 65.0% |
| M6 | 6,100 | 3,782 | 2,318 | 62.0% |
| M7 | 6,400 | 3,840 | 2,560 | 60.0% |
| M8 | 5,900 | 3,422 | 2,478 | 58.0% |
| M9 | 5,200 | 2,860 | 2,340 | 55.0% |
| M10 | 4,900 | 2,548 | 2,352 | 52.0% |
| M11 | 5,100 | 2,550 | 2,550 | 50.0% |
| M12 | 5,400 | 2,592 | 2,808 | 48.0% |
| **Total** | **65,000** | **39,810** | **25,190** | |

The totals are the control total for the later merge: 39,810 + 25,190 = 65,000.

### 6.3 The spurious trends

Fit a simple linear trend to each series over the twelve months.

| Series | First value | Last value | Change over 11 months | Approximate slope | Annualised reading |
|---|---:|---:|---:|---:|---|
| Record A | 3,900 | 2,592 | −1,308 | −119 cases/month | −3.0% per month, declining |
| Record B | 1,300 | 2,808 | +1,508 | +137 cases/month | +10.5% per month, high growth |
| Merged | 5,200 | 5,400 | +200 | +18 cases/month | +0.35% per month, flat |

The merged series is essentially flat with seasonality. The two component series are not: one looks
like a declining product approaching end of life, the other like a high-growth launch. Neither
reading is a fact about demand. Both are artifacts of a shifting split ratio between two records
that describe one product.

The consequence is mechanical. The planning system applies a declining-item treatment to record A
(reduced forward cover, no promotional stock build) and a growth-item treatment to record B
(expanded cover, expedited supply). The plant receives two sets of requirements for one product,
and the aggregate is wrong in both directions at once.

### 6.4 The stock and order arithmetic

At the start of week 40, the east distribution centre (DC) holds 4,200 cases physically. The
system shows 2,600 against record A and 1,600 against record B. The stock is not double counted in
units — 2,600 + 1,600 = 4,200 — but the two records are replenished independently.

Reorder points, using `ROP = demand during lead time + safety stock`, with lead time 21 days
(0.7 months), a 95% cycle service level (CSL) so z = 1.65, and a monthly standard deviation:

| Quantity | Record A | Record B | Merged |
|---|---:|---:|---:|
| Mean monthly demand (cases) | 3,318 | 2,099 | 5,417 |
| Assumed monthly standard deviation | 727 | 490 | 1,200 |
| Safety stock `z × sigma × sqrt(0.7)` | 1,003 | 676 | 1,656 |
| Demand during lead time | 2,323 | 1,469 | 3,792 |
| Reorder point | 3,326 | 2,145 | 5,448 |
| Sum of the two reorder points | | | 5,471 |

Read the last two rows together. The two independent reorder points sum to 5,471 cases. The merged
reorder point is 5,448 cases. The difference is 23 cases, or 0.4%.

This is the arithmetic that refutes the folklore. **Statistical safety stock does not double when
an item is duplicated**, because each half-sized series carries roughly half the standard deviation
and the square-root-of-time scaling applies to both halves equally. Anyone who builds a business
case on a doubled safety-stock claim will be corrected by a competent planner.

Where the money actually is:

| Cost mechanism | Arithmetic | Annual cost |
|---|---|---:|
| Duplicate order placement | Both records breached reorder point in week 40; two orders of 1,500 cases placed against a net need of 1,600 | 1,400 excess cases held about 6 weeks |
| Fixed presentation minimum | 1 pallet (60 cases) per active SKU per stocking location; 3 locations | 180 extra cases at EUR 18.00 carrying value |
| Broken-case buying on record B | EUR 2.60 per case premium on 25,190 cases | EUR 65,494 |
| Obsolete finished goods at end of season | 2.4% of the duplicated volume, at cost | 605 cases, EUR 10,890 |
| Forecast error cost | Higher safety cover set manually after two stock-outs | 2 weeks of cover, 2,080 cases |
| Channel margin misstatement | DTC margin computed on EUR 19.80 instead of EUR 17.20 | EUR 2.60 × 25,190 = EUR 65,494 overstated cost |

Two of these are the real numbers: the broken-case purchasing premium and the margin misstatement,
which are the same EUR 65,494 seen from two directions. The duplication did not create that cost —
buying broken cases did — but the duplication concealed it, because the blended cost never appeared
on one line. This is the characteristic damage of a split: it does not create cost so much as make
cost unattributable.

### 6.5 The merge procedure

The merge is a project with a defined sequence. Each step has a control total.

**Step 1 — Confirm the match.** Deterministic evidence: identical brand, identical declared net
content (500 ml), identical pack count (12), identical case weight within 1%, and the DTC record's
description resolving to the same GS1 Global Product Classification brick. Score 0.97. Confirmed by
physical inspection of one case from each channel: same bottle, same label, different outer carton
print. Human sign-off by the category owner.

**Step 2 — Choose the surviving record.** Record A survives. It holds the licensed GTIN, it has the
longer history, and it is the record the retailer's electronic data interchange (EDI) catalogue
points at. Record B is retired with a successor pointer to A, not deleted.

**Step 3 — Restate the history.** Sum the two series by month into the surviving key.

```text
merged(M1..M12) = A(M1..M12) + B(M1..M12)
control total before = 39,810 + 25,190 = 65,000
control total after  = 65,000        (must be equal; it is)
```

No unit conversion is needed here because both records use the same base unit and the same pack of
12. If record B had been a 6-pack, the restatement would require `B_units × 6 / 12`, and the
control total would have to be expressed in a common unit.

**Step 4 — Transfer the open objects.** Open purchase orders (2), open sales orders (14), inventory
balances at 3 locations, 4 price conditions, 1 BOM reference used by the variety-pack line, and 2
transportation lanes. Each transfer is logged with a count before and after.

**Step 5 — Publish the cross-reference.** A row mapping `10051907 → 10048213` with an effective
date, a reason code `DUPLICATE_MERGE`, and a review date 24 months out. Every historical fact table
keeps the old key and joins through this table for reporting.

**Step 6 — Reconcile.** Three controls must pass:

| Control | Before | After | Tolerance |
|---|---:|---:|---:|
| Units in 12-month history | 65,000 | 65,000 | Exact |
| Units on hand at week 40 | 4,200 | 4,200 | Exact |
| Distinct active SKUs for this product | 2 | 1 | Exact |
| Open order lines | 16 | 16 | Exact |
| Order lines with unresolvable item key | 0 | 0 | Zero |

**Step 7 — Recompute the plan.** One forecast on the merged series, one safety stock of 1,656
cases, one reorder point of 5,448 cases, one presentation minimum per location instead of two.

**Step 8 — Prevent recurrence.** Add a mandatory duplicate check to the DTC item creation screen,
and give the DTC team read access to the grocery item master. The second is the control that
actually works; the first is the control that gets audited.

### 6.6 What the duplicate cost, stated honestly

| Claim | Verdict |
|---|---|
| "Duplication doubled our safety stock" | False. Effect was 23 cases, 0.4%. |
| "Duplication split our demand history" | True, and this is the dominant analytical harm. |
| "Duplication caused two orders in one week" | True, 1,400 excess cases held about 6 weeks. |
| "Duplication hid a EUR 65,000 purchasing premium" | True as a concealment, not as a cause. |
| "Duplication made the DTC margin wrong" | True, by the same EUR 65,494, in the opposite direction. |

An analyst who can write that table is more useful than one who can write a bigger number.

### 6.7 Scaling to a portfolio

Halden has 4,000 active SKUs. A sample audit of 400 finds 71 duplicates, a rate of 17.8%. Applying
that rate:

```text
excess SKUs                = 4,000 x 0.178 = 712
stocking locations         = 3
presentation minimum       = 60 cases per SKU per location
excess fixed buffer        = 712 x 3 x 60 = 128,160 cases
carrying value per case    = EUR 18.00
annual holding cost rate   = 22%
annual carrying cost       = 128,160 x 18.00 x 0.22 = EUR 507,514
```

Half a million euros a year in fixed buffer alone, before the purchasing premium, the obsolescence,
and the planner time. That is a defensible business case, and every term in it is auditable. The
safety-stock claim is not needed and would weaken it.

### 6.8 What the analyst should have asked

1. Was record B created before or after the GTIN was licensed? If after, the failure is a process
   gate, not a data entry mistake.
2. Does the DTC record appear in the retailer's catalogue? If it does, the retailer is holding a
   phantom item and the merge has an external dependency.
3. Is the EUR 19.80 cost a true broken-case premium, or does it include freight that belongs in
   landed cost? If the latter, the margin error is smaller and the freight allocation is the defect.
4. How many other DTC items were created in the same launch window? Duplicates cluster by project,
   so one confirmed case usually implies a cohort.

## 7. Code, query, or calculation

The core operations here are arithmetic and matching, not engineering. One small block is enough.

Normalise, block, then score. Blocking is what makes the comparison tractable at portfolio scale.

```python
# Illustrative. jaro_winkler is not in the Python standard library; any implementation will do.
def norm(s):
    s = s.upper()
    s = re.sub(r"\b(EA|EACH|PK|PACK|CS|CASE|BTL|BOTTLE|X)\b", " ", s)
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", s).split())

def block_key(r):
    return (r.brand, r.pack_count, (r.size_ml // 50) * 50)   # coarse size bucket

for key, recs in group_by(block_key, items).items():
    for a, b in combinations(recs, 2):
        score = (0.50 * jaro_winkler(norm(a.desc), norm(b.desc))
                 + 0.30 * float(a.pack_count == b.pack_count)
                 + 0.20 * float(a.gtin == b.gtin and a.gtin != ""))
        if score >= 0.85:
            queue_for_review(a, b, score)
```

Three properties of this block matter more than the code.

- **Normalisation removes packaging words.** `X12`, `12PK`, `12 PACK`, and `-12-` all become the
  same token sequence, so pack count is compared as an attribute rather than as a string.
- **Blocking is coarse on purpose.** The size bucket rounds to 50 ml, which keeps a 500 ml and a
  520 ml record in the same block. A tighter bucket gives fewer comparisons and more misses.
- **The threshold queues work; it does not merge.** Everything at or above 0.85 goes to a human.
  Auto-merge above 0.97 and auto-reject below 0.70 is defensible; auto-merge at 0.85 is not.

Hand arithmetic for the GTIN check digit, which is the cheapest validation you will ever write.
For a 12-digit payload `d1 … d12`, weight odd positions by 1 and even positions by 3, sum, and take
the check digit as `(10 − sum mod 10) mod 10`.

| Step | Values |
|---|---|
| Payload | 5 0 1 2 3 4 5 6 7 8 9 0 |
| Weights | 1 3 1 3 1 3 1 3 1 3 1 3 |
| Products | 5 0 1 6 3 12 5 18 7 24 9 0 |
| Sum | 5+0+1+6+3+12+5+18+7+24+9+0 = 90 |
| Check digit | (10 − (90 mod 10)) mod 10 = (10 − 0) mod 10 = 0 |
| GTIN | 5012345678900 |

Verification is `sum mod 10 == 0` over all thirteen digits with the same weighting. A file in which
2% of GTINs fail this test has an entry problem; a file in which 30% fail has a *definition*
problem, usually a check digit computed over the wrong digit count.

## 8. Data, standards, and regulatory context

Master data is the most standardised part of the supply chain data landscape, which is fortunate,
because it is also the part where standards are most often ignored in practice.

### 8.1 Item identification

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN Management Standard, Release 1.1 (ratified September 2023) | When a new GTIN is required | Ten rules plus three guiding principles; a unique GTIN at every packaging level | B |
| GS1 General Specifications | GTIN structure and allocation | Company prefix, item reference, check digit; GTINs shall not be reallocated | B |
| GS1 Healthcare GTIN Allocation Rules, Release 10.0 (June 2020) | Regulated healthcare items | Adds unit-of-use, kits, barrier packs, language changes | B |
| GS1 Global Data Model (GDM), releases 2.15 (August 2025) and 2.16 (December 2025) | Attribute definitions for traded goods | Global core plus national local layers; food, near-food, pet food, alcohol, tobacco | B |
| GDM Attribute Implementation Guideline, Release 1.15 (November 2025) | How attributes are populated | Attribute groups, hierarchy rules, TI/HI, category-specific sets | B |
| Verified by GS1 | Whether a GTIN is licensed and to whom | Free lookup, 30 queries per day; batch and API via member organisations | B |

The current GTIN Management Standard is Release 1.1 from 2023. No 2025 or 2026 revision surfaced in
the search for this topic, so any claim that a "2025 GTIN rule" changed the pack-quantity
requirement should be treated as unsourced until the GS1 archive confirms it.

### 8.2 Exchange and synchronisation

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GDSN | Item and party data synchronisation between trading partners | Data pool certification, Global Registry, subscription model | B |
| GS1 Global Registry | Which data pool holds the master copy of an item | One registration per GTIN | B |
| GS1 EDI / EANCOM | Order, invoice, and catalogue messages | Regional document variants | B |
| EPCIS 2.0 | Event data about objects | What, where, when, why; used with SGLNs for sub-locations | B |
| ISO 8000-116 | Legal entity name verification | Identifier for registered legal entities, used in supplier master cleansing | B |

The GDSN model is worth understanding because it inverts the usual assumption. The brand owner
publishes to a certified data pool; the pool registers the item with the Global Registry; trading
partners subscribe through their own pool and receive updates automatically. The GS1 certified data
pool list, as published, contains roughly 45 pools, many of them national member organisation
services built on a small number of shared platforms. The practical consequence for an analyst is
that **GDSN data is only as good as the brand owner's publication discipline**, and a retailer's
copy can lag the manufacturer's by days.

### 8.3 Location and party identification

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GLN | Party and location identity | 13 digits, globally unique, not reused in healthcare | B |
| GS1 US GLN Hierarchy Best Practice Guide, Release 3.0 (October 2023) | How to structure a GLN hierarchy | Corporate, bill-to, sold-to, ship-to, deliver-to levels | B |
| GS1 Healthcare GLN Implementation Guideline, Release 2.1 (October 2020) | Healthcare location identity | Application identifiers 410–416; do not encode hierarchy in the number | B |
| GS1 US Data Hub GLN User Guide (December 2025) | Operational GLN management | Top-level entity GLN cannot be deleted; published GLNs are set inactive, not removed | B |

The hierarchy guidance contains the single most useful distinction in party master design. A
**sold-to** is where the sales agreement is negotiated and where ownership transfers; a **ship-to**
is where goods are delivered; a **bill-to** is where the invoice goes; a **payer** is who pays. These
can be four different GLNs for one commercial relationship, and the US healthcare guidance treats
sold-to as a party-level identifier precisely because it governs change of ownership under the Drug
Supply Chain Security Act (DSCSA). A customer master that stores one address per customer cannot
produce a compliant transaction.

### 8.4 What is convention rather than requirement

- Item numbers that encode product family are a habit, not a standard; GS1 recommends a
  non-significant item reference allocated sequentially.
- "One item per GTIN" is a reasonable internal rule and is not required by GS1: a company may hold
  one GTIN and several SKUs, provided each SKU's relationship to the GTIN is documented.
- Three-level location hierarchies (region, DC, bin) are a warehouse-management convention; the GLN
  guidance supports four or more levels and warns against encoding them in the number.

## 9. Industry practice and case evidence

### 9.1 How much duplication is normal

ECCMA's *Procurement Master Data Quality Analysis* (January 2025) reports that 25–30% duplication
is common in item and material masters, and that supplier masters show roughly 20% unidentifiable
suppliers and around 30% duplication once legal names are verified against registries. The same
report attributes a 15% reduction in purchasing and inventory costs to standards-based master data
formatting. Grade D: it is an industry association publication, the figures are presented without a
sampling frame in the summary available, and the cost-reduction figure is not a controlled study.
Cite the direction, not the precision.

The 17.8% duplicate rate in the worked example is deliberately below that range: a well-governed
consumer business with one ERP should not be at 30%, which is a rate typical of companies that have
acquired others or run several ERPs.

### 9.2 Governance, as practised

| Finding | Source and date | Grade | What it means for an analyst |
|---|---|---|---|
| 65% of organisations have a dedicated data governance function; 86% agree governance is a necessity | ISM 2025 Data and Analytics survey, fielded May–July 2025 | D | The function exists more often than the discipline does; ask what it can veto |
| 63% name master data management their top data priority | Forrester and Dun & Bradstreet, *Mastering Data Management*, May 2025, 304 respondents | D | Priority is not the same as funding; ask for the steward headcount |
| Stewardship and ownership ranked the first governance priority | 2025 State of Enterprise Data Governance Report, 14 senior leaders | E | Small sample; directional only. Note the sample size before quoting |
| 30% have no structured approach to governing external data | Same report | E | Explains why supplier parent hierarchies are incomplete |
| Business ownership assignment and AI-enabled stewardship recommended to scale data quality | Gartner, *Prioritize Master Data Quality to Enable Your Supply Chain Strategy*, November 2025 | D | Consistent with practitioner experience; vendor framing |
| Poor data quality costs the average organisation about USD 12.9 million per year | Gartner, widely cited, research dating to 2020 | D | Dated. Do not present as a 2025 figure |

The consistent signal across these sources is that governance is a named priority and a thin
practice. The proportion of organisations with a governance function is high; the proportion with
the authority to block an item creation is low. An analytics team that assumes the second follows
from the first will build controls that nobody enforces.

### 9.3 A documented failure mode: UOM conversion

USAID's Global Health Supply Chain Programme (GHSC-PSM) published a master data lessons-learned
document in 2024 that is unusually concrete about unit-of-measure failure. It reports missing
packaging hierarchy data breaking calculated conversion fields, and code-set mismatches between
systems — for example a description of "GRAM" in one system against the GDSN standard code "GRM" in
another — producing systematic conversion drift. It also identifies the highest-severity error: a
case-level GTIN stored against an each-level item, so that scanning a case receives one each.
Grade B for the programme documentation; the specific figures are project-level and not a
statistical sample.

That single error illustrates the whole topic. It is silent, it is consistent, it produces a
quantity error of one to three orders of magnitude, and it survives every row-count check because
the row is valid. Only a reconciliation against physical count or against the packaging hierarchy
will find it.

### 9.4 Entity resolution in practice, 2025

The methods are established and the tooling is commodity. A 2025 survey of heterogeneous entity
matching (arXiv 2508.08076) benchmarks rule-based, deep-learning, graph, and language-model matchers,
and notes that the standard benchmark datasets are dominated by product catalogues — Abt-Buy,
Walmart–Amazon, iTunes–Amazon — so the item-master case is unusually well served by published
methods. A reference architecture paper (arXiv 2503.08087) lays out the standard pipeline of
extraction, blocking, matching, clustering, and profile assembly, and compares open tools including
Dedupe, Splink, Zingg, and JedAI.

What changed after 2020 is not the mathematics but the matcher: language models now handle
descriptions that differ semantically rather than lexically — "sparkling water 500ml dozen" against
"carbonated mineral water, 12 x 0.5L" — where edit distance fails. What did not change is that
blocking sets recall and the review queue decides whether the project is adopted.

### 9.5 The honest position on vendor claims

Vendors here sell remediation, so their published numbers describe the size of the problem they
solve. Treat every "X% of records are duplicates" as an upper bound drawn from the worst customers,
and every "Y% cost reduction" as a best case without a control group. Measure the rate in your own
master with a stratified sample, then price the consequences as in section 6.5.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Duplicate created to bypass a blocked field | Two records created within days by different teams | Creation is slower than circumvention | Split demand history, double ordering | Mandatory duplicate check with documented override |
| GTIN reused after a specification change | Item count static while pack size changed | No lifecycle gate on GTIN allocation | Undetectable step change in every series | Change classification table enforced at creation |
| Pack-size change without history restatement | Series shows a one-month jump then a new level | Units changed, key did not | Forecast permanently biased | Unit conversion in restatement, not relabelling |
| UOM factor keyed once and never verified | Case counts differ from physical by a round ratio | Factor entered from memory or a similar item | 2x to 1000x quantity errors | Derive factors from the packaging hierarchy; re-verify annually |
| Case GTIN mapped to an each-level item | Receiving quantities too low by the pack count | Packaging level not qualified in the join | Silent stock understatement | Store packaging level with every GTIN reference |
| Ship-to records duplicated per order | Customer count grows faster than revenue | Address typed per transaction | Distorted segmentation, wrong rebates | Match on GLN plus normalised address, not free text |
| No in-transit location | Stock "disappears" between shipment and receipt | Ownership in transit not modelled | Phantom shortages, duplicate purchase orders | Virtual in-transit location per lane, per Incoterm |
| No lifecycle status | Discontinued items still appear in the plan | Retirement is a deletion or nothing | Phantom demand, obsolete build | Status field with a blocking behaviour |
| Golden record chosen by last write | Attribute values flip between reports | Survivorship unspecified | Unexplainable metric changes | Per-attribute precedence matrix with named owners |
| Phantom BOM nodes not flagged | Component demand exceeds any plausible build | Phantom exploded as a stocked item | Excess component inventory | Set the phantom flag; test with a single-level explosion |
| Cross-reference table missing | Legacy codes unresolvable after a migration | Key replaced without mapping | History lost at every system change | Maintain the table as a first-class artefact |
| Steward accountable but not authorised | Queue grows, users route around it | Accountability without decision rights | Governance theatre | Pair every steward with a named owner who can say no |

One deserves a note. **Duplicate created to bypass a blocked field** is the mechanism that produces
most duplicates, and it is a process design failure rather than a user failure. If a legitimate
creation takes five days and an illegitimate one takes five minutes, the organisation has chosen its
duplicate rate. The fix is to make the fast path legitimate — an expedited lane with post-hoc review
— not to add another warning dialogue.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| One item record corresponds to one managed unit | Forecast and inventory work at item level | Channel-specific or location-specific SKUs for the same product | Compare item count to physical product count in a sample |
| The internal key is stable over the analysis window | Time series require a fixed key | Key reuse, mergers, migrations | Check for step changes with no demand explanation |
| UOM conversion factors are constant over the window | Aggregation assumes consistent units | Pack changes, supplier pack changes, catch weight | Recompute totals from base units independently |
| The unit stored with each fact is the unit intended | Joins assume semantic agreement | Mixed eaches and cases in one column | Profile the unit column by source system |
| Party records map one-to-one to commercial relationships | Spend and risk roll-up need it | Sold-to and ship-to collapsed; franchises; buying groups | Count GLNs per legal entity in a sample |
| Location hierarchy is complete to the level used | Aggregation by region requires it | New sites, divestitures, unassigned bins | Count records with a null parent at each level |
| Lifecycle status is current | Planning excludes retired items | Status updated monthly or never | Age the status change dates against the change log |
| BOM effectivity dates are maintained | Component demand depends on them | Engineering changes applied without dates | Compare BOM versions to change records |
| Reference codes are stable | Tariff, currency, and calendar joins depend on them | Code list revisions, country changes, currency redenomination | Version the reference tables and diff them |
| The golden record reflects the current truth | Reporting assumes it | Batch survivorship running on stale sources | Check the freshness of each source against the record |

Two limits are worth stating. **This topic describes the model, not the migration**: remediating
40,000 records in a live ERP is a programme with sequencing, cutover, and political constraints, and
the content here is an input to it. And **the framing assumes a manufacturing or distribution
business**: for a retailer the item master is largely a supplier catalogue, and for a marketplace
the "item" may be a listing owned by sellers. The domains persist; their ownership moves. Module 21
treats marketplace structures.

## 12. Exercises

### Exercise 1 — Domain ownership map (recall and application, 40 minutes)

Choose a company you know or can research. For each of the six master domains, name the system of
record, the external key if one exists, and one attribute the domain must not own.

**Deliverable:** a six-row table plus a one-paragraph note on which domain is weakest and why.
**Acceptance criteria:** at least four domains have a named system rather than "the ERP"; at least
two external keys are identified with their standard; the "must not own" column contains a real
conflict, not a truism.

### Exercise 2 — UOM chain arithmetic (calculation, 30 minutes)

A case contains 4 inner packs of 6 eaches. A pallet holds 60 cases with TI = 10 and HI = 6.

Compute: eaches per case, eaches per pallet, cases per pallet, and the error introduced over a
5,000-case order if the conversion factor is keyed as 1 case = 12 eaches.

**Deliverable:** four numbers with units, plus the error in eaches and in cases and as a percentage.
**Acceptance criteria:** TI × HI reconciles to cases per pallet; the error is stated as an
understatement or overstatement, not just a magnitude.

### Exercise 3 — Duplicate cost, honestly (calculation, 60 minutes)

Using the parameters in section 6.4, compute the two reorder points and the merged reorder point.
Then compute the annual fixed-buffer cost for 712 excess SKUs across 3 locations with a 60-case
presentation minimum at EUR 18.00 and a 22% holding rate.

**Deliverable:** a table of reorder points and a single currency figure for fixed-buffer cost.
**Acceptance criteria:** the safety-stock difference is stated explicitly as a percentage; the
currency figure shows every factor; no claim of doubled safety stock appears anywhere.

### Exercise 4 — GTIN validation (calculation, 25 minutes)

Compute the check digit for payload `871234567890`. Then judge these three: `5012345678900`,
`6291041500213`, `8712345678905`.

**Deliverable:** the computed check digit and a pass or fail verdict for each, with the weighted sum
shown.
**Acceptance criteria:** the weighting scheme is stated (odd × 1, even × 3); the computed digit is
6; exactly one of the three is marked invalid and the position of the failing digit is identified.

### Exercise 5 — Entity resolution design (design, 75 minutes)

Design a deduplication pass for a 40,000-item master in a consumer business with three channels.
Specify two blocking keys, the match score with weights, the auto-merge, review, and auto-reject
bands, and the size of the review queue you expect.

**Deliverable:** a two-page design with a worked example of two records scoring above and below the
threshold.
**Acceptance criteria:** both block keys are justified against a specific error type; the review
band is non-empty; the design states its expected false-negative mechanism; the queue size is
estimated, not asserted.

### Exercise 6 — Survivorship matrix (design, 50 minutes)

Build a survivorship matrix for ten attributes across an ERP, a PIM, and a quality system. For each,
name the winning system, the rationale, and the conflict rule.

**Deliverable:** a ten-row table.
**Acceptance criteria:** at least three attributes are won by a system other than the ERP; every
rationale names the business consequence of the wrong choice; no row says "most recent".

### Exercise 7 — Location audit (application, 45 minutes)

For a described network — one plant, two DCs, a 3PL, and direct-to-store delivery — list every
location that must exist in the master, including virtual locations, and identify which are
commonly missing.

**Deliverable:** a location list with hierarchy level and function for each.
**Acceptance criteria:** at least two virtual locations are included; in-transit is modelled per
lane and tied to the Incoterm; the missing-location list has a stated consequence for each.

### Exercise 8 — Party role mapping (application, 45 minutes)

Take one B2B customer with a parent company, three ship-to sites, a central payer, and a buying
group. Draw the party and location records required, with GLN roles for each.

**Deliverable:** a diagram plus a table of records with role, GLN, and parent.
**Acceptance criteria:** sold-to, ship-to, bill-to, and payer are distinguished; the buying group
relationship is modelled without duplicating the legal entity; the parent–child link is directional
and dated.

### Exercise 9 — BOM audit (critique, 40 minutes)

Given a BOM with 42 nodes, of which 9 are phantoms and 4 have no effectivity dates, describe what
goes wrong in a planning run and what you would check first.

**Deliverable:** a failure narrative plus a three-item check list in priority order.
**Acceptance criteria:** the phantom mechanism is explained in terms of exploded demand, not
"incorrect BOM"; the first check is the one with the largest quantity consequence; the effectivity
gap is tied to a specific planning symptom.

### Exercise 10 — Governance design (design, 60 minutes)

Design the item creation process for a company with three channels and one ERP. Specify the gate,
the approvers by role, the override path, and the metric that tells you whether it is working.

**Deliverable:** a process diagram, a responsibility table, and one leading and one lagging metric.
**Acceptance criteria:** the override path exists and is time-bounded; every approval is by role,
not by person; the leading metric measures duplicate creation attempts, not duplicate count.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
eaches per case    = 4 inner packs x 6 eaches = 24
cases per pallet   = TI x HI = 10 x 6 = 60
eaches per pallet  = 60 x 24 = 1,440

with the factor keyed as 12:
recorded eaches    = 5,000 x 12 = 60,000
correct eaches     = 5,000 x 24 = 120,000
error              = 60,000 eaches = 2,500 cases = 50% understatement
```

The direction matters. An understated conversion makes the system believe it has less stock than it
does, so it over-orders and the physical count shows a persistent surplus. The opposite keying
produces phantom stock-outs that nobody can explain by looking at the shelf.

**Exercise 3 arithmetic.**

```text
Record A:  sigma 727,  mean 3,318
Record B:  sigma 490,  mean 2,099
Merged:    sigma 1,200, mean 5,417

lead time = 21 days = 0.7 months;  sqrt(0.7) = 0.8367;  z = 1.65

SS_A = 1.65 x 727  x 0.8367 = 1,003
SS_B = 1.65 x 490  x 0.8367 =   676
SS_M = 1.65 x 1,200 x 0.8367 = 1,656

DDLT_A = 3,318 x 0.7 = 2,323;  ROP_A = 2,323 + 1,003 = 3,326
DDLT_B = 2,099 x 0.7 = 1,469;  ROP_B = 1,469 +   676 = 2,145
DDLT_M = 5,417 x 0.7 = 3,792;  ROP_M = 3,792 + 1,656 = 5,448

sum of split ROPs = 5,471;  merged ROP = 5,448;  difference = +23 cases = +0.4%
```

Fixed-buffer cost:

```text
excess SKUs          = 712
locations            = 3
presentation minimum = 60 cases
excess cases         = 712 x 3 x 60 = 128,160
carrying value       = EUR 18.00 per case
holding rate         = 22% per year
annual cost          = 128,160 x 18.00 x 0.22 = EUR 507,514
```

A submission that reports a doubled safety stock has failed the exercise regardless of its other
arithmetic, because the point of the exercise is to test whether you check a plausible claim.

**Exercise 4 arithmetic.**

```text
payload      8  7  1  2  3  4  5  6  7  8  9  0
weights      1  3  1  3  1  3  1  3  1  3  1  3
products     8 21  1  6  3 12  5 18  7 24  9  0
sum          8+21+1+6+3+12+5+18+7+24+9+0 = 114
check digit  (10 - (114 mod 10)) mod 10 = (10 - 4) mod 10 = 6
GTIN         8712345678906   valid
```

Of the three, `6291041500213` is valid (weighted sum 57, check 3) and `5012345678900` is valid
(weighted sum 90, check 0). `8712345678905` fails: the payload's correct check digit is 6, so the
final 5 is a keying error. The instructional point is that a check digit validates *structure*, not
*authenticity* — a well-formed GTIN from another company's prefix passes every time, which is why
Verified by GS1 exists and why the registry lookup, not the check digit, detects a wrong-owner GTIN.

**Exercise 5 rubric.**

A strong answer names two block keys addressing different error types — `(brand, pack_count)` for
transpositions, `(size_bucket, first_description_token)` for missing pack counts — states that
blocking produces false negatives by construction, and estimates the review queue from a
pair-generation rate. Bands such as auto-merge at 0.97, review from 0.70 to 0.97, and auto-reject
below 0.70 are defensible; narrower bands are usually unstaffed and therefore unused. A weak answer
gives one block key and a 0.85 threshold with no review step.

**Exercise 6 rubric.**

A strong matrix gives the PIM the GTIN and channel description, the ERP the internal SKU, planning
description, and standard cost, and quality the shelf life and temperature class. Each rationale
names a consequence: "if the ERP wins gross weight, transport under-uses the trailer" is a
rationale; "the PIM is more accurate" is not. A weak answer assigns everything to one system.

**Exercise 7 rubric.**

The minimum set is plant, plant finished-goods store, two DCs, the 3PL site, each store, plus
virtual locations for quality hold, scrap, and in-transit. The commonly missing ones are in-transit,
quality hold, and the 3PL's internal bins. A strong answer ties in-transit to the Incoterm: under
free carrier the buyer owns goods from origin and needs the location for the whole transit; under
delivered duty paid the seller does.

**Exercise 10 rubric.**

A strong design has one gate with five checks, approval by category owner and supply chain owner, an
expedited path completing in under four hours with post-hoc review within five days, and metrics of
"duplicate creation attempts blocked per month" (leading) and "active SKU count versus physical
product count" (lagging). A weak design adds approvals without an expedited path, which guarantees
circumvention.

## 14. Interview questions

**1. "What is master data, and how is it different from transaction data?"**

A strong answer defines it as the stable entities transactions reference, notes that one defective
master record contaminates every transaction pointing at it *coherently*, and gives an example of
that correlated-error property. A weak answer says "it's the reference data" and stops.

**2. "How many items does a company with 40,000 SKUs have?"**

A strong answer refuses the question as posed and asks whether you mean GTINs, SKUs, or physical
products, noting that the three counts differ and the difference is the duplicate rate. A weak
answer says 40,000.

**3. "Why does the GTIN not solve the duplicate problem?"**

A strong answer gives three reasons: not every item has one, one product carries several across
packaging levels, and wrong-owner or malformed GTINs pass structural checks. It adds that a GTIN
needs a packaging-level qualifier to serve as a join key. A weak answer says people ignore it.

**4. "A pack size changes from 12 to 9. What has to happen?"**

A strong answer requires a new GTIN under the pack-quantity rule, retires the old record with a
successor pointer, and restates history with unit conversion — one old case is 0.75 new cases — not
relabelling. A weak answer says "update the item master."

**5. "How would you find duplicates in an item master you have never seen?"**

A strong answer describes normalise, block on two keys, score on string similarity plus attribute
agreement, then a human review band, and notes that blocking determines recall and thresholds must
be set against a labelled sample. A weak answer says "use fuzzy matching."

**6. "Your two reports disagree on the gross weight of an item. How do you debug it?"**

A strong answer asks which system wrote last, then whether survivorship is specified, then whether
the reports read different sources or snapshots. It does not start in the reporting layer. A weak
answer starts by checking the SQL.

**7. "What is the difference between a sold-to, a ship-to, and a bill-to?"**

A strong answer distinguishes the commercial agreement and ownership transfer (sold-to), the
delivery point (ship-to), and the invoicing party (bill-to), notes they can be three GLNs, and
mentions that in healthcare the sold-to governs change of ownership. A weak answer calls them
addresses.

**8. "How do you merge two duplicate item records without losing data?"**

A strong answer gives the sequence: confirm the match, choose the survivor, restate history with
conversion, transfer open objects, publish a cross-reference, reconcile with control totals,
recompute the plan, fix the process. It states that the record is retired, not deleted. A weak
answer says "sum the history and delete the second one."

**9. "What is the business case for master data remediation? Give me a number."**

A strong answer builds from measured duplicate rate, locations, fixed buffers, and carrying cost,
and deliberately avoids the doubled-safety-stock claim. It separates cost caused by duplication from
cost concealed by it. A weak answer quotes a vendor percentage.

**10. "Who should own the item master?"**

A strong answer separates the data owner (a business role with definitional authority), the steward
(execution), and the system owner (IT), and says the owner must be able to refuse a creation
request. A weak answer says "IT, because they run the system."

## 15. Key terms

- **Master data** — the stable entities that transactions reference.
- **Transaction data** — records of events, each pointing at several master entities.
- **Reference data** — shared code sets that interpret other data.
- **Golden record** — the single agreed representation of an entity.
- **Survivorship** — per-attribute rules for resolving source conflicts.
- **MDM** — master data management, the discipline and its tooling.
- **PIM** — product information management, the commercial product content system.
- **GDSN** — GS1's network of certified **data pools**, the intermediaries that hold the master
  copy of an item.
- **GTIN** — Global Trade Item Number; **GS1 Company Prefix** — its licensed prefix;
  **indicator digit** — the GTIN-14 digit distinguishing packaging levels.
- **Check digit** — the mod-10 final digit; a structural test, not an authenticity test.
- **SKU** — stock keeping unit, the internal managed item identifier.
- **Item hierarchy** — division, family, variant, pack.
- **Base unit / UOM / conversion factor** — the lowest counted unit, its code, and the ratio
  between units.
- **TI / HI** — tiers and units per tier in a pallet configuration.
- **Catch weight** — pricing by actual rather than nominal weight.
- **Lot / serial** — a production group / a single unit identity.
- **Shelf life / temperature class** — time and condition constraints on an item.
- **GLN** — Global Location Number for parties and places.
- **SGLN** — a GLN plus extension for a sub-location such as a bin.
- **Virtual location** — a non-physical location holding real inventory value.
- **In-transit location** — the virtual location for owned but unreceived goods.
- **Sold-to / ship-to / bill-to / payer** — four party roles in one transaction.
- **Party hierarchy** — the ownership or control structure above a party.
- **BOM** — bill of materials; **phantom** — a non-stocked exploded node.
- **Effectivity dating** — dated validity of a master record version.
- **Data owner / data steward** — accountability / execution.
- **Entity resolution** — deciding whether two records denote one entity; **blocking** —
  partitioning records to avoid comparing all pairs; **deterministic / fuzzy match** — exact key
  agreement / score above a threshold.
- **Split, merge, dangle** — the three ways a key lies.
- **Cross-reference table** — the mapping between identifiers and across time.
- **Lifecycle status** — proposed, active, blocked, discontinued, replaced.

## 16. Sources

**Standards and official**

- GS1, *GTIN Management Standard*, Release 1.1, ratified September 2023.
  `https://ref.gs1.org/standards/gtin-management/` (Grade B)
- GS1, *General Specifications*, current edition, sections on GTIN allocation and non-reuse.
  (Grade B)
- GS1, *Healthcare GTIN Allocation Rules Standard*, Release 10.0, June 2020. (Grade B)
- GS1, *Global Data Model*, Releases 2.13 (February 2025), 2.14 (May 2025), 2.15 (August 2025),
  and 2.16 (December 2025). `https://ref.gs1.org/standards/gdm/archive` (Grade B)
- GS1, *GDM Attribute Implementation Guideline*, Release 1.15, November 2025. (Grade B)
- GS1, *GDSN certified data pools list* and *How GDSN works*.
  `https://www.gs1.org/services/gdsn/certified-data-pools-list` (Grade B)
- GS1 US, *Building a GLN Hierarchy: Best Practice Guide*, Release 3.0, October 2023; GS1,
  *Healthcare GLN Implementation Guideline*, Release 2.1, October 2020. (Grade B)
- GS1 US, *Data Hub Managed GLN User Guide*, December 2025; GS1, *Verified by GS1* service
  description and support articles, last modified August 2024. (Grade B)
- USAID Global Health Supply Chain Program, *Product Master Data Management: Lessons Learned*,
  August 2024. (Grade B)

**Peer-reviewed and academic**

- Papadakis, G. et al. (2025). "Heterogeneity in Entity Matching: A Survey and Experimental
  Analysis." arXiv:2508.08076. (Grade A for method taxonomy and benchmark results)
- "Resolvi: A Reference Architecture for Scalable and Interoperable Entity Resolution" (2025).
  arXiv:2503.08087. (Grade A for architecture)
- "Over 50 Years of Probabilistic Record Linkage: A Journey with Three Selected Stops."
  *The Survey Statistician*, July 2025. (Grade A for the Fellegi–Sunter review)
- University of Mannheim, *Web Data Integration: Identity Resolution*, lecture deck, winter
  semester 2025. (Grade A for the similarity-method formulas; teaching material, not evidence)
- Fellegi, I. P. and Sunter, A. B. (1969). "A Theory for Record Linkage." *Journal of the American
  Statistical Association*, 64(328), 1183–1210. (Grade A)

**Industry, analyst, and vendor**

- ECCMA, *Procurement Master Data Quality Analysis*, 2025.
  `https://eccma.org/` (Grade D — association publication; sampling frame not stated in the
  summary consulted; treat the 25–30% duplication figure as directional)
- ISM, *2025 Data and Analytics Research*, fielded May–July 2025, 196+ respondents. (Grade D)
- Forrester and Dun & Bradstreet, *Mastering Data Management*, May 2025, 304 respondents.
  (Grade D)
- Enterprise Data Strategy Board, *2025 State of Enterprise Data Governance Report*, 14 senior
  leaders, concluded April 2025. (Grade E — very small sample)
- Gartner, *Prioritize Master Data Quality to Enable Your Supply Chain Strategy*, 19 November 2025.
  (Grade D — vendor-analyst)
- Gartner, poor-data-quality cost estimate of approximately USD 12.9 million per year, widely cited;
  the underlying research dates to 2020. (Grade D, dated)
- AWS, "Resolve imperfect data with advanced rule-based fuzzy matching in AWS Entity Resolution,"
  30 July 2025. (Grade D for product claims; Grade C for the description of supported algorithms)
- Informatica, Bluestone PIM, Canto, Centric Software, and ISG Research, PIM versus MDM comparison
  material, 2025. (Grade D — vendor content; the PIM-as-golden-record-for-channel-content point is
  consistent across sources and is treated as framing, not evidence)

**Note on citations.** No 2025 or 2026 revision of the GS1 GTIN Management Standard was found; the
current version identified is Release 1.1 (September 2023) and the text says so rather than
implying a newer edition. Figures attributed to surveys are reported as the surveys state them and
are not independently verified. Vendor and analyst grades follow Module 28.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_systems-of-record.md`](01_systems-of-record.md)
- Next topic: [`03_transaction-data.md`](03_transaction-data.md)
- Related: [`09_master-data-drift.md`](09_master-data-drift.md)
- Related: [`../01_supply_chain_foundations/03_network-entities.md`](../01_supply_chain_foundations/03_network-entities.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
