# Network Entities and Product Flow

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 02
> Version 2026.5

---

## 1. Why this topic exists

Ask five people in the same company how many distribution centres the company has, and you will
get five numbers. The logistics director counts buildings. The finance director counts legal
entities that hold inventory. The planning lead counts planning locations in the advanced planning
system. The transport manager counts ship-from addresses. The sustainability team counts sites
that appear in the emissions inventory. Every number is defensible and no two agree.

This is not a governance failure, or not only one. It is what happens when five entity models are
built at different times for different purposes. An analyst who joins across them without knowing
which model a table came from will get an arithmetically correct, semantically meaningless result.

Three examples show the shape of the problem, and each is common enough to have a name in
practice.

1. **The item count that changes when you change systems.** A retailer's merchandising system
   holds 48,000 active SKUs. The same retailer's warehouse management system holds 51,200. The
   difference is not error. It is 3,200 pack configurations that merchandising rolls up to a
   single sellable unit and the warehouse must store separately, plus a few hundred display
   shippers. Any "number of products" metric computed from one system is a statement about that
   system's grain, not about the business.
2. **The supplier that is five suppliers.** A risk model scores vendors by vendor code. Five
   vendor codes belong to one corporate group, ship from two plants, and share one tier-2
   supplier of a critical component. The model reports the category as well diversified because
   it counts five. The category is in fact single-sourced at the tier that matters.
3. **The lane with a mean and no variance.** A safety-stock calculation uses a 4.0-day ocean
   transit time. The observed transit time has a mean of 4.0 days, a standard deviation of 1.1
   days, and a 95th percentile of 5.9 days. The calculation is correct for the question "what is
   the average journey" and wrong for the question "how much buffer do I need". Treating a lane
   as a scalar discards the only part of the lane that matters for service.

Each example is an entity problem, not a modelling problem. The model was fine. The nouns were
wrong.

**Why this matters now.** Until roughly 2020, entity data was an internal convenience. It has
since become a legal artifact. The EU Digital Product Passport (DPP), the EU Corporate
Sustainability Due Diligence Directive (CSDDD), the U.S. Food Traceability Rule under the Food
Safety Modernization Act (FSMA 204), and pharmaceutical serialisation under the Drug Supply Chain
Security Act (DSCSA) all require naming a product, a site, and a batch in a way another
organisation can verify. An invented identifier can now fail an obligation, not just a join.

This topic supplies the nouns that Topic 01's flows move through and that Topic 02's decisions
act on. Topic 07 explodes the bill of materials (BOM) into capacity and material requirements.
Topic 08 prices the lane. Topic 09 describes the warehouse. Everything downstream assumes you
know what a node, a lane, and an item actually are.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** party, location, asset, item, and logistics unit, and **explain** why a single
   "site" table cannot serve as all five.
2. **Design** an item master grain for a stated analytical question, including the hierarchy,
   the unit-of-measure (UOM) set, and the identifier strategy.
3. **Convert** quantities across a packaging ladder (each, inner pack, case, pallet, container)
   and **identify** the point at which weight, rather than cube, becomes the binding constraint.
4. **Explode** a single-level and a multi-level BOM with scrap factors, and **state** the
   rounding convention used.
5. **Explain** effectivity dating, phantom items, alternates, and where-used, and **diagnose**
   an incorrect explosion caused by using the current BOM for a historical period.
6. **Define** a lane as origin, destination, mode, and service level, and **represent** transit
   time as a distribution rather than a point estimate.
7. **Compute** basic network topology measures — degree, path length, articulation points, and
   the Herfindahl-Hirschman Index (HHI) — and **interpret** them as resilience indicators.
8. **Diagnose** the tier-n visibility problem and **explain** why tier-1 diversification can
   coexist with tier-3 concentration.
9. **Specify** slowly changing dimension (SCD) handling for items and locations, and **justify**
   the choice between Type 1 and Type 2 for a given analytic use.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: five layers of identity stacked on top of one another, with a network of lanes
running between locations at layer two.

```text
   LAYER 5   LOGISTICS UNIT   the physical object handled as one thing
             (case, pallet, container)                     key: SSCC
                 ^
                 | packed into
   LAYER 4   ITEM             what is bought, made, moved, sold
             (the trade item at a stated pack level)       key: GTIN + pack level
                 ^
                 | produced on
   LAYER 3   ASSET            the equipment that does the work
             (line, reactor, vehicle, cold room)           key: GIAI / asset number
                 ^
                 | located at
   LAYER 2   LOCATION         where material or capacity can be
             (plant, DC, cross-dock, virtual location)     key: GLN
                 ^
                 | operated by
   LAYER 1   PARTY            who is legally responsible
             (company, supplier, customer, 3PL)            key: GLN / company number
```

The layers are not a hierarchy of importance. They are a hierarchy of *questions*. "How much did
we sell" is a layer-4 question. "Where is it" is layer-2. "Who owes us money" is layer-1. "What
is on that pallet" is layer-5. Most failed joins in supply chain analytics are a jump between
layers with no mapping table in between.

Two properties of this stack explain most of what follows.

**Property 1 — A party is not a place, and neither is a node.** A supplier is a party. The
supplier's factory is a location. The model that you build to optimise flows contains nodes,
which are a *selected subset* of locations chosen because they matter to the question. Three
different things, three different keys, and the same English word ("site") is used for all three
in most meetings.

**Property 2 — Every entity has a natural key and a surrogate key, and they answer different
questions.** The natural key is what the outside world uses: a GTIN, a GLN, a company
registration number. The surrogate key is what your warehouse uses: an integer, a UUID. The
surrogate key is stable and fast. The natural key is meaningful and joinable. An entity model
that keeps only one of them will eventually fail — either it cannot join to a partner, or it
cannot survive a re-keying.

### 3.1 The anatomy of a lane

A lane is not a line on a map. It has six attributes, and an analytics model that omits any of
them will misprice or mistime the flow.

```text
        ORIGIN                    LANE                        DESTINATION
   +---------------+   +-----------------------------+   +---------------+
   | GLN 5412345   |   | mode       road / ocean     |   | GLN 5419876   |
   | Bologna plant |   | service    48 h, Mon-Fri     |   | Utrecht DC    |
   | ship-from cal.|-->| transit    mean 3.1 d       |-->| receive cal.  |
   +---------------+   |            sd 0.8 d         |   +---------------+
                       | capacity   33 pallets / wk  |
                       | cost       EUR 0.042 /      |
                       |            pallet-km        |
                       +-----------------------------+
```

Read the lane left to right and you get the physical flow. Read it right to left and you get the
information flow. Read the cost line and you get the financial flow. The same object carries all
three, which is why lane tables are the most reused table in supply chain analytics and the most
frequently mis-joined.

Four distinctions are worth fixing in memory now.

| Concept | What it is | What it is not |
|---|---|---|
| Lane | A directional origin-destination pair with a mode and a service commitment | A road, a route, or a carrier |
| Route | The physical path a vehicle or vessel takes, possibly across several lanes | A commercial relationship |
| Itinerary | A multi-leg sequence with transshipment points | A single lane |
| Service level | The committed transit time and frequency on a lane | The customer-facing fill rate |

A single lane can be served by several routes. A single route can carry traffic for several
lanes. A shipment can have an itinerary that traverses three lanes and two modes. Confusing these
is how a "lane" table ends up with one row per sailing, which makes every aggregation wrong.

### 3.2 The item is a tree, not a row

An item record feels like a row. It is actually a position in a tree, and the tree has at least
four common levels above the pack:

```text
   BRAND
     |
     +-- FAMILY            e.g. pasta sauce
           |
           +-- VARIANT   e.g. tomato basil, 400 g
                 |
                 +-- PACK        e.g. single jar, 6-pack, 24-case
                       |
                       +-- TRADE ITEM   the thing with a GTIN
```

Different functions cut this tree at different heights. Finance reports by brand. Sales reports
by variant. Manufacturing schedules by pack. The warehouse stores by trade item. A dashboard that
mixes levels will double-count on one screen and under-count on another, and the analyst will
usually blame the data.

### 3.3 The BOM is a graph with time on its edges

A bill of materials looks like a nested list. It is a directed acyclic graph whose edges carry a
quantity, a scrap factor, and a validity period.

```text
   FG-1042  jar of tomato basil sauce, 400 g
     |
     +--(1 EA)---- JAR-400      glass jar
     +--(1 EA)---- LID-063      lug lid
     +--(1 EA)---- LBL-1042     label
     +--(0.0417)-- CASE-24      shipping case
     |
     +--(0.400 kg)- SB-01       sauce base   [PHANTOM: never stocked]
                     |
                     +--(0.380 kg)-- TOM-001  tomato concentrate
                     +--(0.120 kg)-- ONI-014  diced onion
                     +--(0.080 kg)-- OIL-003  olive oil
                     +--(0.030 kg)-- SAL-001  salt
                     +--(0.010 kg)-- CIT-005  citric acid
                     +--(0.050 kg)-- BAS-002  basil
```

The phantom box is the important one. `SB-01` exists as an item number, appears on a work order,
and is never held in stock. Material requirements planning (MRP) explodes *through* it. An
analyst who treats it as a stocked item will create a fictitious inventory position and add a
lead time that does not exist.

The edges also carry time. Revision 7 of this BOM is valid from 1 March 2026. Before that date,
revision 6 applies, and revision 6 uses more tomato concentrate because it was written against a
lower-brix supplier. Exploding a February production order with the March BOM over-orders tomato
by roughly four percent — small enough to look like noise, large enough to matter at scale.

### 3.4 Topology explains behaviour that node attributes cannot

Two networks with identical nodes, identical lanes, and identical costs can behave very
differently under stress, because of how the lanes are arranged. Three measures capture most of
the difference:

- **Degree** — how many lanes touch a node. A degree-1 node is a leaf: a supplier with one
  customer, or a customer with one source.
- **Articulation point** — a node whose removal disconnects the graph. In supply chain terms, a
  site whose failure splits the network into parts that can no longer reach each other.
- **Concentration** — how unevenly flow or spend is distributed across the alternatives. The
  HHI is the standard measure.

Section 6.7 works these through numerically. Hold the intuition: a network is fragile in
proportion to how much of its flow passes through nodes and edges with no alternative path.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Entity | A thing the business must name, count, or join on | Used to mean "table" or "record" | None |
| Party | An organisation with legal or commercial identity | Confused with the site it occupies | GLN, company registration number |
| Legal entity | The contracting and tax-reporting unit | Assumed to equal the operating company | National company number |
| Location | Any place material or capacity can be | Assumed to be a warehouse | GLN |
| Site | A physical place with an address and a boundary | Used for virtual locations too | GLN plus address |
| Node | A location admitted into a particular network model | Treated as every location in the master | Model-specific identifier |
| Asset | Equipment or a vehicle that performs work | Confused with an item | GIAI (Global Individual Asset Identifier) |
| Item | Anything bought, made, stored, or sold | Used to mean finished goods only | GTIN, internal material number |
| Trade item | An item at a defined packaging level that is ordered or sold | Assumed to be the consumer unit | GTIN-8, GTIN-12, GTIN-13, GTIN-14 |
| SKU | The item-at-location combination a business plans and stocks | Used as a synonym for item | Internal alphanumeric |
| GTIN | Global Trade Item Number, a globally unique trade item identifier | Treated as unique across pack levels | 8, 12, 13, or 14 digits |
| Indicator digit | First digit of a GTIN-14 that distinguishes packaging level | Ignored, so case and each collide | 1–8; 9 for variable measure |
| GLN | Global Location Number, identifying a party or a place | Assumed to identify physical places only | 13 digits |
| SSCC | Serial Shipping Container Code, identifying one logistics unit | Confused with a GTIN | 18 digits |
| Item hierarchy | The tree from brand to pack | Rebuilt annually without versioning | Internal plus GPC (Global Product Classification) |
| Brand / family / variant / pack | Four common levels of the item hierarchy | Collapsed into one "category" field | Internal |
| UOM | Unit of measure in which a quantity is expressed | Assumed unique per item | EA, CS, KG, L, PLT |
| Base UOM | Unit in which inventory is held and counted | Confused with the order UOM | EA or KG |
| Order UOM | Unit in which a customer buys | Confused with base UOM | CS, PLT |
| Catch-weight | A product whose individual units vary in weight | Treated as fixed-weight | GTIN indicator 9 with AI 310n or 320n |
| Inner pack | A bundle of eaches inside a case | Counted as a case | GTIN-13 or GTIN-14 |
| Case | The standard shipping unit | Assumed to hold a fixed count for all items | GTIN-14 |
| Pallet / unit load | A set of cases handled as one object | Confused with a container | SSCC |
| Lane | A directional origin-destination connection with mode and service | Treated as a road route | Origin GLN, destination GLN, mode |
| Lane service level | The committed transit time and frequency | Confused with customer service level | Days; deliveries per week |
| Transit time | Elapsed time from dispatch to arrival | Treated as a constant | Days, as a distribution |
| BOM | Structured list of components for a parent item | Assumed to be a single document | Item, revision, effectivity |
| Single-level BOM | Components directly under one parent | Assumed to be the whole product | Parent plus component |
| Multi-level BOM | The full explosion down to purchased parts | Confused with a sum of single levels | Depth in levels |
| Phantom item | A logical assembly that is never stocked | Planned as inventory | Item with phantom flag |
| Effectivity date | The period in which a BOM or change applies | Ignored when exploding history | Valid-from, valid-to |
| Alternate | A component approved to replace another | Treated as a second BOM | Item plus priority |
| Where-used | The list of parents that consume an item | Not maintained | Item plus parent list |
| Routing | The sequence of operations and work centres | Confused with the BOM | Operation number |
| Degree | Number of lanes incident to a node | Not measured | Count |
| Articulation point | A node whose removal disconnects the graph | Not measured | Boolean |
| HHI | Sum of squared shares, measuring concentration | Applied without stating the share base | 0–1 or 0–10,000 |
| Surrogate key | A generated key with no business meaning | Used as the only key, losing traceability | Integer or UUID |
| SCD | A dimension that tracks attribute change over time | Rebuilt as a snapshot each month | Type 1, 2, or 3 |
| Cross-dock | A site where goods transfer with little or no storage | Treated as a distribution centre | Location type |
| 3PL | Third-party logistics provider operating a site | Treated as a carrier | Contract |
| Virtual location | A location with no physical footprint | Omitted from the master | Internal code |
| Consignment stock | Stock at a customer site owned by the supplier | Counted as sold | Agreement |

Five of these cause disproportionate damage and deserve expansion.

**Party versus location versus node.** A party answers "who". A location answers "where". A node
answers "what is in the model". A supplier group with three plants is one party, three locations,
and possibly three nodes, or one node if the model aggregates by country. The node set is a
modelling choice; the party and location sets are facts about the world. Analysts frequently
inherit a node table and mistake it for a location master, then discover that a plant which
closed two years ago is still in the network because nobody removed the node.

**GTIN versus SKU.** A GTIN identifies a trade item globally. A SKU identifies an item at a
location for planning purposes. One GTIN can map to several SKUs (same product in three
warehouses), and one SKU can occasionally map to several GTINs (a pack change mid-season where
the old and new packaging are planned together). The mapping is many-to-many and it changes over
time. Any join that assumes one-to-one will silently drop rows.

**Base UOM versus order UOM.** Inventory is held in a base unit, usually eaches or kilograms.
Customers order in cases or pallets. Production is scheduled in batches. Freight is priced in
kilograms and cubic metres. Procurement buys in drums, bags, or containers. Five UOM systems,
one item. The conversion factors between them are business data, not constants, and they change
when a supplier changes a pack.

**Catch-weight versus fixed-weight.** A fixed-weight item has a known net weight per unit. A
catch-weight item — a cheese wheel, a tray of meat, a bag of produce — does not. Its GTIN uses
indicator digit 9, and the actual weight travels as a GS1 Application Identifier (AI) on the
label rather than in the item master. Aggregating catch-weight inventory in eaches is valid.
Aggregating it in kilograms by multiplying count by a nominal weight is an estimate with a
systematic error that grows with the variance of the unit weights.

**Echelon versus tier.** Echelon counts position in the chain: supplier, plant, distribution
centre (DC), customer. Tier counts depth of supplier relationship: tier 1 supplies you, tier 2
supplies tier 1. A company's plant is at echelon 2 of the chain and is a tier-1 customer of its
direct suppliers. The words are not interchangeable, and mixing them makes a resilience
discussion incoherent within about four sentences.

## 5. The mechanics, step by step

This section builds an entity model in the order a competent analyst would, naming the decision
at each step and the trap that follows from getting it wrong.

### Phase A — Parties and locations

**Step 1. Separate the legal party from the physical site.**

- What it is: a party table keyed by a company identifier, holding legal name, registration
  number, country, and corporate parent. A location table keyed by a GLN, holding address,
  coordinates, time zone, and site type.
- Why it matters: contracts, tax, credit, and sanctions screening attach to the party. Capacity,
  lead time, and emissions attach to the location.
- Trap: many enterprise resource planning (ERP) systems hold a single "vendor" record that
  conflates the two. The vendor code identifies the invoicing entity, while the ship-from address
  is a free-text field. When the supplier opens a second plant, the free-text field changes and
  the history becomes unjoinable.
- Analytic consequence: supplier risk measured on the party understates exposure measured on the
  location, because one party can concentrate production in one building.

**Step 2. Admit virtual locations to the master.**

- What it is: locations with no physical footprint but real inventory consequences — in transit,
  customs bond, quarantine, quality hold, consignment at a customer, returns staging, scrap.
- Why it matters: an inventory position that omits in-transit stock is wrong by the entire
  transit time. A quarantine location that does not exist in the master means held stock is
  counted as available.
- Trap: virtual locations are usually created as a workaround by a planner, given an informal
  code, and never documented. Two planners create two codes for the same concept.
- Test: reconcile the sum of all location-level inventory against the financial inventory
  balance. A persistent gap is usually an unmodelled virtual location.

**Step 3. Decide the model granularity.**

- What it is: choosing which locations become nodes. Options range from every GLN to a
  country-level aggregate.
- Why it matters: granularity determines what questions the model can answer. A country-level
  model cannot answer a question about a specific DC, and a GLN-level model of a 4,000-site
  retail network may be computationally unnecessary for a strategic study.
- Trap: building one model at the finest grain "so it can answer anything", then discovering that
  the cost of maintaining it exceeds its value. The usual resolution is two models: a strategic
  one at coarse grain and an operational one at fine grain, with an explicit mapping.

**Step 4. Reconcile the location master to the external registry.**

- What it is: matching internal location identifiers to GLNs, and checking that the GLN is
  active.
- Why it matters: GS1's Global Data Synchronisation Network (GDSN) statistics show that data
  pools periodically purge inactive or expired GLNs, so a registry count is a count of active
  parties, not of all parties ever registered. An internal master that has never been reconciled
  will contain locations that no trading partner recognises.
- Trap: assuming a GLN is one building. A GLN can identify a legal party, a functional entity
  within a party, or a physical location. The same 13-digit space is used for all three, and the
  meaning is declared by the registrant.

### Phase B — Items

**Step 5. Choose the item grain.**

- What it is: the decision about what one row in the item table represents. The three common
  choices are: the trade item (one GTIN), the SKU (item at location), and the material (item at
  any level, including components).
- Why it matters: every downstream join inherits this choice. A GTIN-grain item table cannot
  hold location-specific safety stock. A SKU-grain table duplicates 40 attribute fields across
  12 warehouses.
- Recommended pattern: a trade-item table with global attributes, a SKU table with
  location-specific attributes, and a material table for components. Three tables, three grains,
  explicit relationships.

**Step 6. Build and version the hierarchy.**

- What it is: a parent-child tree from brand down to pack, plus cross-classifications such as
  the GS1 Global Product Classification (GPC) and internal category trees.
- Why it matters: every report aggregates at some level of this tree. If the tree changes
  without versioning, last year's report cannot be reproduced.
- Trap: category teams reorganise the tree annually, usually in the same month, and usually
  without telling analytics. The result is a step change in every category series that looks
  like a demand event.
- Mitigation: version the hierarchy with valid-from and valid-to dates, and store the hierarchy
  version on every fact row or in a bridge table.

**Step 7. Define the UOM set and the base UOM.**

- What it is: the enumeration of units in which the item is expressed, plus the conversion
  factors between them, plus the unit in which inventory is held.
- Why it matters: conversion errors are silent. A wrong factor does not throw an error; it
  produces a plausible number.
- Required attributes: base UOM, order UOM, purchase UOM, production UOM, freight UOM, and the
  conversion factors with their effective dates.
- Trap: double rounding. Converting each → case with a factor of 24, then case → pallet with a
  factor of 96, then rounding at each step, accumulates error. Convert once from the base unit to
  the reporting unit, then round once.

**Step 8. Attach identifiers.**

- What it is: GTIN per trade item per pack level, plus internal material numbers, plus supplier
  part numbers, plus customer item numbers.
- Why it matters: each external identifier is a join path to a partner. A missing GTIN means the
  product cannot be described in an advance ship notice (ASN) or an EPCIS event.
- Trap: reusing a GTIN across pack levels. The indicator digit exists precisely to prevent this,
  and ignoring it is the most common cause of a case being received as a single each.

### Phase C — Lanes

**Step 9. Define the lane as origin, destination, mode, service.**

- What it is: one row per directional origin-destination pair per mode per service level.
- Why it matters: cost, transit time, and capacity are properties of this combination, not of the
  two nodes separately.
- Trap: modelling lanes as undirected. A lane from plant to DC and a lane from DC to plant have
  different costs, different equipment, and different documentation.

**Step 10. Estimate transit time as a distribution.**

- What it is: at minimum a mean and a standard deviation; better, an empirical distribution by
  day of week and season, with the 5th and 95th percentiles.
- Why it matters: safety stock is a function of variability, not of the mean. A lane with a mean
  of 4 days and a standard deviation of 1.1 needs materially more buffer than a lane with a mean
  of 4 days and a standard deviation of 0.2.
- Test: plot the transit time distribution for your ten highest-volume lanes. If the distribution
  is bimodal, you have two service patterns sharing one lane code, and the model should split
  them.

**Step 11. Attach capacity and cost.**

- What it is: capacity in the lane's natural unit (pallets per week, TEU per sailing, kilograms
  per truck), and cost in a form that can be applied to a flow (cost per pallet-kilometre, cost
  per container, fixed plus variable).
- Why it matters: a network model without lane capacity will route unlimited volume through the
  cheapest lane and produce an infeasible plan.
- Trap: capacity stated as an annual average when it is seasonal. Ocean capacity in the peak
  season is not the annual average, and the model will under-price the peak.

### Phase D — The BOM

**Step 12. Choose the BOM view.**

- What it is: engineering BOM (EBOM, how the product is designed), manufacturing BOM (MBOM, how
  it is built, including consumables and packaging), and sales BOM (SBOM, how it is sold as a
  kit). The same product has all three.
- Why it matters: exploding an EBOM for material planning omits packaging and process
  consumables. Exploding an MBOM for engineering change analysis includes items that are not
  part of the design.
- Trap: assuming one BOM serves all purposes. It does not, and the differences are exactly where
  cost and availability questions live.

**Step 13. Set revision and effectivity.**

- What it is: a revision identifier plus valid-from and valid-to dates, ideally tied to an
  engineering change order.
- Why it matters: any historical explosion must use the BOM that was valid at the time, not the
  current one.
- Trap: effectivity stored as free text ("from March") rather than a date. The explosion engine
  cannot use it, so it uses the current revision for everything.

**Step 14. Mark phantoms.**

- What it is: a flag on assemblies that are never stocked and never planned.
- Why it matters: MRP explodes through phantoms, consuming their components into the parent.
  Treating a phantom as a stocked item creates a fictitious inventory and an extra lead time.
- Trap: a phantom whose components have different effectivities from the parent. The explosion
  must apply the component's own effectivity, not the parent's.

**Step 15. Register alternates and substitutes.**

- What it is: alternates are approved replacements, often with a priority order; substitutes are
  temporary replacements, often unapproved, used under duress.
- Why it matters: an alternate changes the conversion factor, not only the item number. One
  kilogram of basil extract replaces nine kilograms of fresh basil, so the BOM quantity changes
  with the alternate.
- Trap: modelling an alternate as a second BOM. You then have two structures to maintain, and
  they drift. Model it as a property of the component with a priority and an effective period.

### Phase E — Topology

**Step 16. Compute degree, paths, and articulation points.**

- What it is: for each node, count incident lanes; for each pair of nodes, compute the shortest
  path in days; identify nodes whose removal disconnects the graph.
- Why it matters: these are the cheapest resilience diagnostics available and require no
  simulation.
- Test: remove each node in turn and recompute connectivity. Any node whose removal splits the
  network is a candidate single point of failure, and should be checked against capacity.

**Step 17. Compute concentration.**

- What it is: HHI over spend, volume, or capacity shares, at the category and item level.
- Why it matters: a category with three suppliers can still be concentrated if one holds 80% of
  the volume, and a category with one supplier is concentrated by definition.
- Trap: computing HHI on the number of suppliers rather than on shares. Ten suppliers each with
  10% is an HHI of 0.10; two suppliers at 50% each is 0.50. The count and the concentration tell
  different stories.

### 5.1 Reading an entity diagram correctly

Given any data model or network diagram, ask these six questions in order.

1. **What is the grain of each table?** One row per what? If the answer is not a single sentence,
   the table has more than one grain and will duplicate on join.
2. **Which key is natural and which is surrogate?** If there is only one key, ask what happens
   when the business re-codes.
3. **Where is the party, and where is the location?** If they share a table, find the column that
   distinguishes them. If there is none, the model cannot separate credit risk from physical
   risk.
4. **Which locations are virtual?** If none, in-transit and quarantine stock are missing.
5. **What is the pack-level key on the item?** If the item table has no pack level, case and each
   will collide.
6. **Does the BOM carry effectivity?** If not, historical explosions will use the present.

Each question is cheap. Each one prevents a class of error that is expensive to find later.

## 6. Worked example

Ardent Foods is a mid-sized European food manufacturer. It makes one product — a 400 g jar of
tomato basil pasta sauce, item code FG-1042 — in Bologna, distributes from Utrecht, and serves
retail customers in the United Kingdom and France. All figures are illustrative but internally
consistent, and every conversion is shown.

### 6.1 The network

```text
   [ S1 Valle Verde ]                         [ S2 Anadolu Cam ]
     Foggia, IT                                 Izmir, TR
     tomato concentrate                         glass jars
     drums, road FTL                            containers, ocean + road
          | 2 d                                      | 9 d
          |                                          |
          +----------------> [ P1 Bologna plant ] <--+
                                   |
                                   | 3 d road FTL
                                   v
                            [ D1 Utrecht DC ]
                              |            |
                   4 d road   |            |  2 d road
                   + ferry    |            |
                              v            v
                    [ C1 Manchester ]   [ C2 Lyon ]
                       retail DC          retail DC
                              ^
                              | 1 d
                        [ X1 Lille cross-dock ]
                              ^
                              | 2 d
                              +---- from D1
```

Seven nodes, seven lanes. Note that `X1` gives `C1` a second path, which is the only reason
`C1` is not a pure leaf. This detail matters in section 6.7.

### 6.2 The item and its packaging hierarchy

| Level | Code | Contents | Net weight | Gross weight | GTIN form |
|---|---|---|---:|---:|---|
| Each | EA | 1 glass jar, 400 g sauce | 0.400 kg | 0.590 kg | GTIN-13 |
| Inner pack | IP | 6 eaches, shrink-wrapped | 2.400 kg | 3.560 kg | GTIN-13 |
| Case | CS | 4 inner packs = 24 eaches | 9.600 kg | 14.790 kg | GTIN-14, indicator 1 |
| Pallet | PLT | 96 cases, 8 layers of 12 | 921.600 kg | 1,445.440 kg | SSCC |
| Container | 40HC | 18 pallets, floor-loaded | 16,588.800 kg | 26,017.920 kg | SSCC per pallet |

The gross weights build as follows, and the build is worth doing once by hand because every
downstream freight calculation depends on it.

```text
jar gross        = 0.400 net + 0.180 glass + 0.010 lid = 0.590 kg
inner gross      = 6 x 0.590 + 0.020 film             = 3.560 kg
case gross       = 24 x 0.590 + 4 x 0.020 + 0.550 carton
                 = 14.160 + 0.080 + 0.550             = 14.790 kg
pallet gross     = 96 x 14.790 + 25.000 pallet + 0.600 wrap
                 = 1,419.840 + 25.600                 = 1,445.440 kg
container gross  = 18 x 1,445.440                     = 26,017.920 kg
```

### 6.3 The UOM ladder, computed

The customer orders in cases. The plant produces in eaches. Procurement buys tomato concentrate
in drums and glass in containers. Finance reports in kilograms of product. One order, five units.

Take an order of 1,200 cases from customer `C1`.

```text
eaches        = 1,200 CS x 24 EA/CS              = 28,800 EA
inner packs   = 1,200 CS x 4 IP/CS               = 4,800 IP
net product   = 28,800 EA x 0.400 kg/EA          = 11,520.0 kg
gross product = 1,200 CS x 14.790 kg/CS          = 17,748.0 kg
pallet positions = 1,200 CS / 96 CS/PLT          = 12.5 -> 13 PLT
```

Two things to notice.

**The rounding is not symmetric.** Twelve and a half pallets becomes thirteen positions, but the
thirteenth is half full: `48 x 14.790 + 25.600 = 735.520 kg`. Charging a full pallet weight
overstates it by 709.9 kg, which at EUR 0.042 per pallet-kilometre over 1,100 km is EUR 32.8 of
phantom freight. Small, but it recurs on every part-pallet shipment.

**The case count and the pallet count answer different questions.** The customer bought 1,200
cases. The warehouse must build 13 pallet positions. The transport planner must decide whether
13 positions justify a full truckload (FTL) at 33 positions or a less-than-truckload (LTL)
movement. The answer depends on the lane, not on the order.

### 6.4 Weight versus cube

A 40-foot high-cube container has roughly 76 cubic metres of internal volume and a payload limit
of about 26,500 kg on a European road leg. The pallet footprint is 1.2 m by 1.0 m, and the loaded
pallet height is 2.40 m.

```text
pallet volume      = 1.2 x 1.0 x 2.40            = 2.880 m3
pallet weight      = 1,445.440 kg
container by cube  = 76 / 2.880                  = 26.4 -> 26 pallets
container by weight= 26,500 / 1,445.440          = 18.3 -> 18 pallets
binding constraint = weight
```

The container is 68% full by volume (`18 x 2.880 = 51.84 m3` of 76) and 98% full by weight
(`26,017.9 / 26,500 = 98.2%`). This is normal for dense food products and it inverts the usual
intuition. For a container of potato chips, cube binds and weight is irrelevant. For a container
of sauce, weight binds and a third of the box is air.

**Why an analyst should care.** Freight is quoted per container, per kilogram, or per cubic
metre. Model cost per cubic metre for a weight-bound product and you will under-cost the lane and
over-estimate the volume a container carries. The binding constraint is a property of the product,
not of the mode.

### 6.5 The BOM explosion, with scrap

The BOM is stated per 1,000 good jars. An order for 28,800 jars is therefore 28.8 times the base
quantity. Scrap is expressed as a fraction of *input*, so the gross requirement is net divided by
(1 − scrap), not net multiplied by (1 + scrap).

| Component | Net qty per 1,000 jars | Scrap | Gross per 1,000 | Net for 28,800 | Gross for 28,800 | Purchase UOM | Purchase qty |
|---|---:|---:|---:|---:|---:|---|---:|
| TOM-001 tomato concentrate | 152.0 kg | 2.5% | 155.90 kg | 4,377.6 kg | 4,489.8 kg | 220 kg drum | 21 drums |
| ONI-014 diced onion | 48.0 kg | 4.0% | 50.00 kg | 1,382.4 kg | 1,440.0 kg | 20 kg bag | 72 bags |
| OIL-003 olive oil | 32.0 kg | 1.5% | 32.49 kg | 921.6 kg | 935.6 kg | 183 kg drum | 6 drums |
| BAS-002 fresh basil | 20.0 kg | 6.0% | 21.28 kg | 576.0 kg | 612.8 kg | 5 kg crate | 123 crates |
| SAL-001 salt | 12.0 kg | 0.5% | 12.06 kg | 345.6 kg | 347.3 kg | 25 kg bag | 14 bags |
| CIT-005 citric acid | 4.0 kg | 0.5% | 4.02 kg | 115.2 kg | 115.8 kg | 25 kg bag | 5 bags |
| JAR-400 glass jar | 1,000 EA | 3.0% | 1,030.93 EA | 28,800 EA | 29,690.7 EA | 2,000 EA pallet | 15 pallets |
| LID-063 lug lid | 1,000 EA | 2.0% | 1,020.41 EA | 28,800 EA | 29,387.8 EA | 5,000 EA carton | 6 cartons |
| LBL-1042 label | 1,000 EA | 1.5% | 1,015.23 EA | 28,800 EA | 29,238.6 EA | 10,000 EA roll | 3 rolls |
| CASE-24 shipping case | 41.667 EA | 0.0% | 41.667 EA | 1,200 EA | 1,200 EA | 100 EA bundle | 12 bundles |

Worked longhand for tomato concentrate, because the arithmetic is where the convention lives:

```text
net per 1,000      = 152.0 kg
gross per 1,000    = 152.0 / (1 - 0.025) = 155.897 kg
net for 28,800     = 152.0 x 28.8        = 4,377.60 kg
gross for 28,800   = 4,377.60 / 0.975    = 4,489.85 kg
drums              = 4,489.85 / 220      = 20.41 -> 21 drums
purchased quantity = 21 x 220            = 4,620 kg
excess over gross  = 4,620 - 4,489.85    = 130.15 kg
```

That 130 kg is real, and it is neither scrap nor error. It is the difference between a requirement
and a purchase, and the reason material cost from gross requirements differs from material cost
from purchase orders.

**The convention trap.** The common alternative convention multiplies by (1 + scrap). For jars:

```text
net / (1 - scrap)  = 28,800 / 0.9700 = 29,690.7 EA   (correct if scrap is a share of input)
net x (1 + scrap)  = 28,800 x 1.0300 = 29,664.0 EA   (wrong by 26.7 jars at 3% scrap)
```

At low scrap rates the difference looks negligible. At 6% (fresh basil) the two conventions
differ by 6.4%, which is the difference between a full crate and a partial one. Pick one
convention, document it on the BOM, and use it everywhere.

### 6.6 Effectivity and alternates

Revision 7 of the BOM is valid from 1 March 2026. Revision 6 used 158 kg of tomato concentrate per
1,000 jars because it was written against a lower-brix supplier. A production order dated 25
February 2026 must explode with revision 6.

```text
rev 6 tomato gross for 28,800 = (158.0 x 28.8) / 0.975 = 4,667.1 kg
rev 7 tomato gross for 28,800 = (152.0 x 28.8) / 0.975 = 4,489.8 kg
difference                    = 177.3 kg  (3.9% of the rev 7 requirement)
```

An analyst who explodes the February order with the current BOM under-orders tomato by 177 kg.
Nobody notices, because the number is plausible. The shortfall appears on the line, three weeks
later, as an unplanned stoppage.

The BOM also carries an alternate for basil. `BAS-EXB` basil extract replaces fresh basil at a
ratio of 1 kg extract to 9 kg fresh. When the alternate is active, the BOM quantity changes as
well as the item:

```text
fresh basil gross    = 612.8 kg
extract required     = 612.8 / 9          = 68.1 kg
extract with 0.5% scrap = 68.1 / 0.995    = 68.4 kg
```

If the planning system models the alternate as a separate item with its own BOM line, the two
lines can both be active and the requirement doubles. The correct model is one line with an
alternate group and a priority, plus an effective period, plus a conversion ratio on the
alternate relationship.

### 6.7 Lanes as distributions, and the topology of the network

`L3` (Utrecht to Manchester) is quoted as a four-day service. Observed over 24 months, transit
time has a mean of 4.0 days, a standard deviation of 1.1 days, a 5th percentile of 2.6 days, and
a 95th percentile of 5.9 days. The distribution is right-skewed: the lane is usually on time and
occasionally very late, which is what a ferry plus a border plus a customer booking slot
produces.

If safety stock is computed from the mean alone, the buffer covers an average journey. If it is
computed from the 95th percentile, it covers 19 journeys in 20. The difference for a product with
daily demand of 1,600 jars is:

```text
extra days of cover = 5.9 - 4.0       = 1.9 days
extra buffer        = 1.9 x 1,600     = 3,040 jars
```

3,040 jars is 127 cases, or 1.3 pallets, or about EUR 2,100 at a EUR 0.69 unit cost. The number
is small. The point is that a mean-only calculation produces a systematically short buffer, and
the failures concentrate on the lanes with the highest variance.

Now the topology. Degree by node:

| Node | Type | Degree | Articulation point | Note |
|---|---|---:|---|---|
| S1 Valle Verde | Supplier site | 1 | No | Leaf; one customer |
| S2 Anadolu Cam | Supplier site | 1 | No | Leaf; one customer, ocean-dependent |
| P1 Bologna | Plant | 3 | **Yes** | Removing it isolates both suppliers |
| D1 Utrecht | DC | 4 | **Yes** | Removing it isolates C1, C2, and X1 |
| X1 Lille | Cross-dock | 2 | No | Provides the second path to C1 |
| C1 Manchester | Customer site | 2 | No | Two inbound paths |
| C2 Lyon | Customer site | 1 | No | Leaf; one path |

Two articulation points, `P1` and `D1`. Removing either disconnects part of the network. `X1`
exists precisely so that `C1` is not dependent on the `D1`–`C1` lane alone, which is a design
choice worth noticing: the cross-dock is not there to reduce cost, it is there to remove an
articulation dependency for the largest customer.

Now concentration. Tomato concentrate spend is split 55% Valle Verde, 30% Campo Rosso, 15% Iberia.

```text
HHI = 0.55^2 + 0.30^2 + 0.15^2
    = 0.3025 + 0.0900 + 0.0225
    = 0.4150            (on the 0-1 scale)
    = 4,150             (on the 0-10,000 scale)
```

Three suppliers, and the index says the category is concentrated. The OECD's 2024 work on
product-level vulnerabilities treats an index above 0.5 as highly vulnerable when combined with a
second index measuring the ability to substitute suppliers; the European Commission's 2021
methodology used 0.4 as a threshold for import concentration. Different bodies, different
thresholds. Treat the threshold as a convention and the index as the fact.

The practical question is not the index value. It is: if Valle Verde fails, what happens in the
first four weeks? The answer is not "45% of supply remains", because Campo Rosso requires a
four-week qualification for a new specification. Effective coverage in weeks 1–4 is zero for the
Valle Verde share. That is the number to put in front of a decision-maker.

### 6.8 What an analyst should store

For this one product, a defensible minimum entity model is:

| Table | Grain | Key | Critical columns |
|---|---|---|---|
| `dim_party` | One row per legal entity | `party_id` | GLN, parent, country, registration |
| `dim_location` | One row per GLN | `location_id` | Type, address, time zone, virtual flag |
| `dim_item` | One row per trade item per pack level | `item_id` | GTIN, indicator, base UOM, hierarchy version |
| `dim_uom` | One row per item per UOM | `item_id`, `uom` | Factor to base, pack size, valid-from, valid-to |
| `dim_lane` | One row per O-D-mode-service | `lane_id` | Transit mean, sd, capacity, cost basis |
| `bridge_bom` | One row per parent-component-revision | `parent_id`, `component_id`, `revision` | Qty, scrap, effectivity, phantom flag |
| `fact_shipment` | One row per shipment | `shipment_id` | Lane, SSCC list, weight, cube, dates |

Every table carries a grain declaration. Every key has a natural and a surrogate component.
Every relationship that can change over time carries a valid-from date.

## 7. Code, query, or calculation

This section stays small on purpose. The arithmetic is the lesson; the query is only a way to
make the arithmetic repeatable.

### 7.1 Hand calculation template

Given a net requirement `N` in eaches, a packaging ladder with factors `f1` (eaches per inner
pack), `f2` (inner packs per case), `f3` (cases per pallet), and `f4` (pallets per container),
plus a scrap rate `s`:

```text
gross eaches        = N / (1 - s)
cases               = ceil(gross eaches / (f1 x f2))
pallets             = ceil(cases / f3)
containers          = ceil(pallets / f4)
net weight kg       = gross eaches x unit net weight
gross weight kg     = cases x case gross weight
```

Use `ceil` at the point of purchase, not before. Rounding to cases and then to pallets and then
to containers is correct because each is a purchase decision. Rounding the *conversion factors*
is not.

### 7.2 One small query

The following structured query language (SQL) fragment explodes one level of a BOM with scrap
and rounds to a purchase UOM. It assumes the schema sketched in section 6.8.

```sql
-- One row per component for a given parent, revision, and required quantity.
-- scrap is stored as a fraction of input, not of output.
select
    b.component_id,
    b.qty_per_parent * :required_qty                    as net_qty,
    b.qty_per_parent * :required_qty / (1 - b.scrap)    as gross_qty,
    ceil((b.qty_per_parent * :required_qty / (1 - b.scrap))
         / p.pack_size)                                 as purchase_units
from bridge_bom b
join dim_uom p
  on p.item_id = b.component_id
 and p.uom     = b.purchase_uom
where b.parent_id = :parent_id
  and b.revision  = :revision
  and :production_date between b.valid_from and b.valid_to;
```

Expected output for `FG-1042`, revision 7, 28,800 jars: ten rows, with tomato concentrate
returning `net_qty = 4377.6`, `gross_qty = 4489.85`, and `purchase_units = 21`.

Two sanity checks before trusting any explosion:

- Sum the gross quantities back up the tree and confirm that no phantom item appears as a
  requirement. A phantom in the output means the explosion stopped one level too early.
- Compare the total material cost at standard cost against the finance standard for the same
  order. A gap above 1% usually means a UOM factor changed without the standard being updated.

## 8. Data, standards, and regulatory context

### 8.1 Identification standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity | One globally unique number per trade item per pack level | B |
| GS1 GLN | Party and location identity | The identity used in electronic data interchange (EDI), EPCIS, and registries | B |
| GS1 SSCC | Logistics unit identity | The label on a pallet or case as a physical object | B |
| GS1 GIAI | Asset identity | Equipment and returnable transport items | B |
| GS1 GRAI | Returnable asset identity | Pallets, crates, and totes that cycle | B |
| GS1 Digital Link | Identifier resolvable by a URL | The route from a scanned code to product data | B |
| GS1 EPCIS 2.0 | Event data | What happened to which object, where, when, and why | B |
| GS1 GPC | Product classification | A common category taxonomy across trading partners | B |

Two facts about these standards are worth stating plainly, because both are widely misunderstood.

**They are voluntary in most jurisdictions.** GS1 standards become binding through contracts,
retailer requirements, and sector regulation, not through general law. A company that sells only
business-to-business industrial parts may never need a GTIN. A company that sells through a
grocery retailer will be required to have one by the retailer's supplier agreement.

**They identify, they do not describe.** A GTIN says which trade item this is. It does not say
what it weighs, what it contains, or where it was made. Those attributes live in master data and
in product data pools, and the identifier is only the key that lets you find them.

### 8.2 Packaging, measurement, and variable measure

The GS1 Package and Product Measurement Standard (Release 3.2, March 2024) is the reference for
how packaging levels are measured and how units convert. Three points from it matter for
analytics:

- Trading partners may use any valid UOM, and the receiving partner is responsible for
  conversion. That means the conversion factor is a shared assumption, and it must be stated
  rather than inferred.
- The standard fixes conversion constants: 1 lb = 0.454 kg, 1 kg = 2.205 lb, 1 inch = 25.4 mm.
  Convert first, then round. Rounding before conversion is a common and silent source of drift.
- Weight and dimension tolerances apply at case level (4% is typical for cases). Variable-measure
  items are excluded from gross-weight tolerances and carry their actual measure in attributes
  instead.

Catch-weight items use GTIN indicator digit 9, and the actual measure travels in GS1-128
Application Identifiers: AI (30) for count, AI (310n) for net weight in kilograms, AI (320n) for
net weight in pounds, where `n` gives the decimal position. A 22.75 lb item is encoded as
`3201` followed by `002275`. A variable-weight item does not need a new GTIN when its weight
changes, because the weight is an attribute of the instance, not of the trade item.

> **Note:** the GS1 rule that a variable-weight item keeps one GTIN is convenient for
> identification and inconvenient for analytics. It means the item key cannot be used to
> distinguish a 4.1 kg wheel from a 4.6 kg wheel. Any weight-based cost or yield analysis must
> read the application identifier from the label data, not the item master.

### 8.3 Traceability and product regulation

| Requirement | Jurisdiction | Status as of late 2026 | Grade |
|---|---|---|---|
| DSCSA serialisation and interoperable exchange | United States, pharmaceuticals | Dispenser compliance deadline 27 November 2025; exchange now routine | B |
| EU Falsified Medicines Directive | European Union, pharmaceuticals | In force; verification at dispensing | B |
| ESPR Digital Product Passport | European Union | Registry operational 20 July 2026; batteries mandatory 18 February 2027; textiles delegated act planned Q4 2027 | B |
| EU Batteries Regulation | European Union | DPP for EV, LMT, and industrial batteries above 2 kWh from February 2027 | B |
| EU CSDDD (Directive 2024/1760, as amended by Directive (EU) 2026/470) | European Union | Omnibus I in force 18 March 2026; transposition by 26 July 2028; application 26 July 2029; scope now above 5,000 employees and EUR 1.5bn turnover | B |
| EU Deforestation Regulation (EUDR) | European Union | Phased application; geolocation due diligence for covered commodities | B |
| U.S. FSMA 204 Food Traceability Rule | United States | Original date 20 January 2026; FDA proposed a 30-month extension to 20 July 2028 and Congress directed non-enforcement before that date | B |
| EU CBAM definitive regime | European Union | Began 1 January 2026 for covered goods | B |

The DPP timeline is the one most likely to change an entity model in the next three years. The
architecture is hybrid: a central EU registry holds unique product identifiers and metadata, while
the product data itself stays with the economic operator or an authorised service provider. Eight
harmonised standards are being developed by CEN-CENELEC JTC 24; six were in final approval as of
mid-2026, with authentication and access-rights standards lagging. For an analyst, the practical
consequence is that the item master must eventually carry a resolvable unique identifier per
product instance or batch, not only a GTIN per trade item.

> **Note:** the ESPR delegated acts are still being adopted. Any specific date in this table
> should be re-checked against the current Commission page before it is used in a compliance
> plan. The direction of travel is stable; the calendar is not.

### 8.4 Master data quality: what the evidence says

The ECCMA Procurement Master Data Quality Analysis (January 2025) reports that 25–30% of
item/material master records are duplicates is a common condition, and that roughly 20% of
supplier masters contain unidentifiable suppliers, with about 30% supplier duplication after name
verification. These are industry-association estimates rather than audited findings, so treat them
as an order of magnitude (grade D). They are consistent with company-reported cases: Bayer's
Berlin site reduced materials from 63,000 to 48,000 during an SAP S/4HANA migration, a 20%
reduction attributed to proven duplication, reported through a vendor case study (grade D).

The GS1 US "Powering Supply Chain Confidence" survey (March 2025, 505 professionals) reports that
52% of respondents had fully adopted GS1 standards, and that full adopters were more likely to
report high confidence in managing disruptions (77%) than partial adopters (50%) or those still
considering adoption (25%). It also reports that only 53% could trace beyond one-up/one-down and
46% could perform lot- or batch-level recall. The association is real in the sample; the causal
direction is not established by a cross-sectional survey (grade C).

### 8.5 What is convention rather than requirement

Several things that feel like rules are habits, and an analyst who cannot tell them apart will
treat inherited parameters as physics.

- HHI thresholds. "Above 0.25 is concentrated" is a convention that varies by institution (0.25,
  0.4, and 0.5 all appear in official methodologies). The index is a fact; the threshold is a
  policy.
- The 80/20 Pareto split in ABC classification. A convention from a particular historical
  context, not an optimum for any specific business.
- "One SKU per pack" as an item-master rule. Convenient for planning, wrong for warehouses that
  store the same product in several configurations.
- The four-level item hierarchy (brand, family, variant, pack). Common, not universal, and not
  required by any standard.
- Weekly planning buckets. A habit inherited from batch systems, not a constraint of the
  business.

## 9. Industry practice and case evidence

### 9.1 The tier-n problem: two public illustrations

The gap between what a company knows about its tier-1 suppliers and what it knows about tier 3 is
the most studied structural weakness in supply chain risk.

**Tōhoku, March 2011.** Fujimoto's analysis for the University of Tokyo describes a "diamond
structure": tiers 1 and 2 are diversified across many suppliers, while tier 3 is concentrated in
a few specialised firms making simple components. The tiers that look most diversified are the
ones that conceal the concentration. Matsuo's study of Toyota documents that Toyota tracked
suppliers to tier 2 but generally not to tier 3 and below, and that it took about a week to list
500 parts sourced from 200 locations. Toyota, Honda, and Nissan bought through different tier-1
suppliers — Denso, Keihin, Hitachi Automotive — that all sourced microcontrollers from the same
Renesas plant. A single building four or five tiers upstream was a single point of failure for
three competing OEMs at once. The U.S. Congressional Research Service report on the same event
records Renesas at roughly 40% of world automotive microcontroller supply, and Merck's Xirallic
pigment produced at a single Fukushima plant. Recovery times differed by OEM — roughly three
months for Toyota, two for Nissan, four for Honda — which is itself evidence that the visibility
and response mechanism, not only the physical damage, determined the outcome.

**Semiconductors, 2021–2023.** The pattern repeated with a different mechanism. Ramani, Ghosh,
and Sodhi's analysis in *Omega* maps the chain as OEM to tier 1 (Continental, Bosch) to tier 2
(NXP, Infineon, STMicroelectronics) to foundries (TSMC), and notes that General Motors sourced
parts from roughly 250 suppliers buying chips from 11 chipmakers. Automotive-grade qualification
means a fab cannot be switched in a quarter, so a capacity constraint at one node propagates
without a substitution path. Swiss Re's 2023 analysis makes the single-point-of-failure mechanism
explicit: tier-1 parts suppliers depend on a small number of tier-2 semiconductor providers, which
depend on a handful of tier-3 wafer manufacturers, and a disruption to any tier-3 supplier creates
a bottleneck where no replacement can be found (grade C for the insurer's synthesis; the
underlying concentration figures are corroborated by the peer-reviewed literature).

### 9.2 What visibility surveys actually show

| Finding | Source and date | Grade |
|---|---|---|
| Tier-3 visibility rated "above average" by 29% of respondents; "little or none" by 40% | Automotive News Chip Shortage Report, April 2021, n=475 | C |
| Only 17% of organisations have visibility beyond tier 1 | Vendor white papers citing Gartner, 2024–2025; original note not seen | D |
| About half of disruptions originate beyond tier 1 | Business Continuity Institute, cited in vendor analyses, 2021–2024 | D |
| 51% say most supply chain mapping is regulation-driven; 32% say all of it | Sphera N-Tier Transparency survey, March 2025, n=250 | D |
| About 70% struggle with data accuracy from tiers 2–4 | Sphera N-Tier Transparency survey, March 2025, n=250 | D |
| Only 53% can trace beyond one-up/one-down | GS1 US, March 2025, n=505 | C |

The pattern across sources is consistent: visibility degrades sharply after tier 1, and the
degradation is driven by commercial confidentiality as much as by technical difficulty. A mapping
programme that offers suppliers nothing — no qualification support, no demand visibility, no
financing — will receive incomplete data.

### 9.3 Digital Product Passport readiness

The DPP is the first regulation that makes item-level identity a legal requirement across a broad
class of products, rather than a sector-specific one. Three practical observations from the
public record as of late 2026:

1. **The standards are not finished.** Six of eight harmonised standards were in final approval
   in mid-2026, with authentication and access rights lagging. Companies building to a draft
   standard should expect rework.
2. **The data model is decentralised.** The registry holds identifiers and metadata; the payload
   lives with the operator. That makes the identifier strategy the load-bearing decision, and it
   makes identifier persistence across repurposing and remanufacturing a real design problem —
   the battery guidance states that a new passport is required on repurposing or remanufacturing,
   and the passport ceases on recycling.
3. **Sector pilots are ahead of the delegated acts.** CIRPASS-2 ran pilots across textiles,
   electrical and electronic equipment, construction, and tyres through April 2027. Catena-X
   published a case study on cross-border data aggregation, noting that around 80% of EU static
   data could be achieved through a whitelist pilot, with China's data-export rules as the binding
   constraint on the remainder. That is a specific and sobering number for anyone assuming global
   DPP coverage at launch.

### 9.4 Network modelling practice in 2025–2026

Network design has moved from spreadsheet-based centre-of-gravity models to simulation-optimisation
tools that carry explicit node and lane data. Two documented examples:

- Borouge (an ADNOC and Borealis joint venture) built a supply chain network optimisation and
  simulation digital twin in eight months using commercial tools, modelling packing plants,
  inventory holding points, and destination ports with explicit lane capacities in tonnes per
  month, transit times, and feeder-leg rail and sea splits. Reported savings were up to 9% of
  logistics cost (grade C, company conference paper).
- The academic literature in 2025 converges on graph-based representations — nodes as assets,
  edges as flows — combined with simulation for stress testing. Ivanov's *Omega* paper
  formalises object-driven and data-driven twin construction; a *Logistics* review of 179 papers
  finds digital twin the dominant cluster and identifies interoperability standards (ISO 23247)
  and black-swan prediction as open gaps.

The practical lesson for an analyst is unglamorous: the model is only as good as the lane table.
Most failed network studies fail on capacity and transit-time data, not on the optimiser.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Party conflated with site | Supplier count falls when a group consolidates legal entities | Vendor record carries a free-text ship-from address | Risk understated; capacity double-counted | Separate party and location tables; map both |
| Duplicate item master | Total item count jumps after a migration | Same product created per site or per legacy system | Inflated inventory, wrong ABC, split demand history | Match on GTIN plus attributes; merge with audit trail |
| Pack-level GTIN collision | Case received as a single each | Indicator digit ignored or reused | Inventory overstated by the pack factor | Enforce GTIN-14 with indicator per pack level |
| UOM conversion drift | Material cost differs from standard by 1–5% | Factor changed without versioning; double rounding | Silent margin error; wrong purchase quantities | Version conversion factors; convert once, round once |
| Catch-weight treated as fixed | Yield variance unexplained by process | Count multiplied by nominal weight | Systematic cost and yield bias | Use label application identifiers; model weight distribution |
| Lane transit treated as a scalar | Service failures concentrate on high-variance lanes | Planning system stores a mean | Safety stock systematically short | Store mean, standard deviation, and percentiles |
| Lane modelled as undirected | Return and outbound costs look identical | One row per node pair | Freight cost error; wrong carrier selection | Directional lane keys with mode and service |
| BOM effectivity ignored | Historical standard cost cannot be reproduced | Explosion uses current revision | Wrong historical margin; wrong component orders | Date-effective BOM; explode as of production date |
| Phantom planned as stock | A stored item appears with zero on-hand forever | Phantom not flagged | Fictitious inventory, extra lead time | Flag phantoms; explode through |
| Alternate modelled as second BOM | Two component lines both active | No alternate group or priority | Double requirement | One line, alternate group, conversion ratio |
| Surrogate key without natural key | Cannot join to partner data | Internal integer only | No external traceability; re-keying breaks history | Keep both keys; map to GTIN and GLN |
| Virtual location omitted | Inventory reconciliation gap equals in-transit value | In-transit and quarantine not in master | Phantom stock-outs; duplicate orders | Model virtual locations explicitly |
| Hierarchy re-versioned without dates | Category series shows a step change in month 1 | Tree rebuilt annually | False trend; unreproducible reports | Version hierarchy with valid-from and valid-to |
| Concentration measured by supplier count | "We have five suppliers" for a single-source category | Count instead of shares | False diversification | Compute HHI on shares; check tier 2 |

Two of these deserve a note.

**UOM conversion drift** is the most expensive silent failure in the list. It produces no error,
no null, and no outlier. It produces a number that is 2% wrong for three years. The only reliable
detection is a periodic reconciliation between the material cost computed from BOMs and the
material cost in the ledger, with a tolerance and an owner.

**Hierarchy re-versioning** is the most common cause of a false trend. Every January, category
teams across the industry redraw their trees. Every February, an analyst reports that the
"snacks" category grew 30% and the "crackers" category collapsed. Nothing happened in the market.
The tree moved.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The item master has one row per trade item per pack level | Joins and counts depend on it | Pack variants collapsed or split inconsistently | Count distinct GTIN per item and look for duplicates |
| UOM conversion factors are stable over the analysis window | Aggregation requires fixed factors | Pack change, supplier change, reformulation | Compare factor history against known pack changes |
| Locations have stable identifiers over time | Time series need a persistent key | Site closure, divestment, GLN reissue | Check for gaps and re-use of identifiers |
| The node set in the model equals the locations that matter | Network metrics depend on it | Model built for one purpose reused for another | Compare node list to the GLN master |
| Transit time is stationary within a season | Safety stock and lane cost depend on it | Route change, carrier change, border policy | Plot transit time by month and by lane |
| BOM revision is recorded on the transaction | Historical explosion depends on it | Effectivity not stored at transaction grain | Try to reproduce last year's standard cost |
| Scrap is a stationary fraction of input | Explosion requires one convention | Ramp-up, new line, supplier quality change | Compare scrap by month and by line |
| The hierarchy used for reporting is the hierarchy in force at the time | Comparability depends on it | Annual reorganisation | Check for step changes at known reorganisation dates |
| Supplier shares are measured on a consistent basis | HHI comparisons require it | Spend versus volume versus capacity | Recompute with a second basis and compare |
| Tier-1 data is complete enough to be a starting point | All multi-tier mapping starts there | Supplier group structure hidden behind vendor codes | Reconcile vendor codes to corporate parents |

Two limits of the entity framing are worth stating plainly.

**Entities are a model, not the world.** Every entity table is a simplification built for a
purpose. The same warehouse may be one node in a strategic model, twelve locations in an
operational model, and three legal entities in a financial model. None is wrong. The error is
using one where another was intended.

**The framing assumes stable identifiers.** In practice, identifiers are reissued, merged, and
retired. A supplier is acquired and its GLN changes; a product is reformulated and gets a new
GTIN; an item is renumbered in an ERP migration. The entity model is therefore always partly a
historical reconstruction, and the reconstruction is always incomplete. Topic 02's decision stack
is the reminder that the reconstruction is itself a decision with a cost.

## 12. Exercises

### Exercise 1 — Entity taxonomy sort (recall and application, 25 minutes)

Take twelve rows from a real or synthetic vendor file — company names, plant addresses, port
codes, item descriptions, pallet labels — and classify each as party, location, asset, item, or
logistics unit.

**Deliverable:** a twelve-row table with the classification and the natural key you would use.
**Acceptance criteria:** every row classified; at least two rows where the same real-world thing
appears at two layers (for example, a supplier and its plant); a one-sentence justification for
each ambiguous case.

### Exercise 2 — Packaging ladder (calculation, 30 minutes)

A product ships as: 12 eaches per inner pack, 8 inner packs per case, 60 cases per pallet, 22
pallets per 40-foot container. Each gross weight is 0.85 kg, case gross is 82.4 kg, pallet tare
is 24 kg, container payload limit is 26,500 kg.

**Deliverable:** eaches per case, eaches per pallet, eaches per container, gross weight per
pallet, gross weight per full container, and the binding constraint with the arithmetic shown.
**Acceptance criteria:** correct arithmetic at every level; the binding constraint identified
with the two comparisons (cube is not given, so state the assumption); a statement of how many
pallets actually load.

### Exercise 3 — BOM explosion with scrap (calculation, 40 minutes)

Using the BOM in section 6.5, explode a requirement of 50,000 good jars. Use gross = net / (1 −
scrap) and round purchase quantities up to the stated purchase UOM.

**Deliverable:** a table of net, gross, and purchase quantity for all ten components, with the
arithmetic for at least two components shown longhand.
**Acceptance criteria:** correct convention; correct rounding direction; a note on how much
excess is purchased versus required, in kilograms and in currency at a stated unit cost.

### Exercise 4 — Effectivity diagnosis (critique, 35 minutes)

A plant reports a material shortage on tomato concentrate for a production week in February 2026.
The BOM was revised on 1 March 2026. Write a two-page diagnosis.

**Deliverable:** the most likely mechanism, the calculation that would confirm it, and two
alternative explanations.
**Acceptance criteria:** the calculation uses the correct revision for the production date; at
least one alternative is a data-capture problem rather than a planning problem; the confirmatory
test is specific enough to run.

### Exercise 5 — Lane definition (design, 45 minutes)

For a network with one plant, two DCs, one cross-dock, and four customer sites, define the lane
table.

**Deliverable:** a lane list with origin, destination, mode, service level, capacity basis, and
cost basis for each lane.
**Acceptance criteria:** lanes are directional; at least one lane has two service levels and is
therefore two rows; the capacity basis is stated in a physical unit, not in currency; the
cross-dock's role is justified in one sentence.

### Exercise 6 — Transit time distribution (application, 45 minutes)

Using any shipment history with at least 200 observations on one lane, characterise transit time.

**Deliverable:** a histogram or percentile table, the mean and standard deviation, and the 5th,
50th, and 95th percentiles.
**Acceptance criteria:** the distribution shape is described in words; one operational
consequence of the shape is stated; the analysis states the time unit and the observation window.

### Exercise 7 — Topology audit (application, 50 minutes)

Draw a seven-node network of your own construction with at least eight lanes. Compute degree for
each node, identify every articulation point, and compute the shortest path in days between the
upstream-most supplier and the downstream-most customer.

**Deliverable:** the diagram, the degree table, the articulation point list with the disconnection
each causes, and the path length.
**Acceptance criteria:** articulation points verified by removing each node in turn; path length
in days with the lane times listed; at least one design change proposed that removes an
articulation dependency.

### Exercise 8 — Concentration and tier-n (analysis, 50 minutes)

For one purchased category, obtain supplier shares by spend and by volume. Compute HHI on both
bases. Then identify, from public sources, whether any of the suppliers share a plausible tier-2
dependency.

**Deliverable:** two HHI calculations, a comparison of the results, and a tier-2 hypothesis with
the evidence and its grade.
**Acceptance criteria:** both share bases stated; the threshold convention named; the tier-2
claim labelled as fact, inference, or unresolved.

### Exercise 9 — SCD design (design, 45 minutes)

Design the dimension handling for an item whose pack size changes mid-year and a location that is
divested and renumbered.

**Deliverable:** a Type 1 versus Type 2 decision for each attribute, with the column design
(valid-from, valid-to, current flag) and one worked example row.
**Acceptance criteria:** the choice is justified by a named analytic use, not by preference; the
design handles a late-arriving correction; the renumbering case preserves history.

### Exercise 10 — Entity model critique (critique, 55 minutes)

Take any supply chain schema you can obtain — an open dataset, a public data model, or the schema
in section 6.8 — and critique it against the six questions in section 5.1.

**Deliverable:** a two-page review with one finding per question and a prioritised remediation
list.
**Acceptance criteria:** each finding names the table and column affected; at least one finding is
about grain rather than about missing fields; the remediation list is ordered by risk, not by
effort.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
eaches per case    = 12 x 8                 = 96 EA
eaches per pallet  = 96 x 60                = 5,760 EA
eaches per container = 5,760 x 22           = 126,720 EA
case gross         = 96 x 0.85 + packaging  = 82.4 kg  (given)
pallet gross       = 60 x 82.4 + 24         = 4,968.0 kg
container gross    = 22 x 4,968.0           = 109,296 kg
```

The container is far over the 26,500 kg payload: `26,500 / 4,968 = 5.33`, so only five pallets
load by weight. That is 28,800 eaches, or 23% of the cube-based count. The product is extremely
dense, and the "22 pallets per container" figure is a cube assumption that the weight makes
irrelevant. A strong answer states this explicitly rather than reporting the cube number.

**Exercise 3 arithmetic, two components longhand.**

```text
required good jars = 50,000

tomato: net   = 152.0 x 50          = 7,600.0 kg
        gross = 7,600 / 0.975       = 7,794.9 kg
        drums = 7,794.9 / 220       = 35.43 -> 36 drums = 7,920 kg

jars:   gross = 50,000 / 0.97       = 51,546.4 -> 51,547 EA
        pallets = 51,547 / 2,000    = 25.77 -> 26 pallets = 52,000 EA
```

Excess purchased: tomato `7,920 − 7,794.9 = 125.1 kg`; jars `52,000 − 51,547 = 453 EA`. At an
illustrative EUR 1.90/kg for tomato and EUR 0.11/jar, that is EUR 237.7 and EUR 49.8. The excess
is a purchase-rounding cost, not scrap, and it belongs in the material cost variance rather than
in the scrap account.

**Exercise 4 model answer.**

The most likely mechanism: the February production order was exploded with BOM revision 7, valid
from 1 March, instead of revision 6. Revision 6 requires 158 kg per 1,000 jars against revision
7's 152 kg, a 3.9% difference. On a 50,000-jar week that is roughly 296 kg of tomato concentrate
short — enough to stop a line for part of a shift. The confirming calculation is to explode the
same order with both revisions and compare the gross requirement against the purchase order
quantity.

Alternative explanations to test: (a) receipt timing — the drums arrived but were not confirmed,
so the system shows a shortage that does not exist physically; (b) a yield loss on the line above
the standard 2.5%, which would produce the same symptom without any BOM error. The confirmatory
test for both is a physical count against the system position on the day of the shortage.

**Exercise 5 rubric.**

A strong answer produces directional lanes, splits at least one lane into two service levels,
states capacity in pallets per week or tonnes per month, and states cost as a rate with its basis.
It justifies the cross-dock as a resilience or consolidation device, not as a cost saving in
general. A weak answer produces one row per node pair with a single lead time and cost.

**Exercise 7 rubric.**

A strong answer verifies each articulation point by removal, not by inspection. It notices that a
cross-dock can convert an articulation dependency into a redundant path, and that redundancy
without capacity is not redundancy: a second path that cannot carry the volume is a diagram, not
a capability. A weak answer declares the highest-degree node the most critical, which is often
wrong — a degree-4 node with three redundant paths can matter less than a degree-2 bridge.

**Exercise 8 rubric.**

A strong answer reports both HHI values and explains why they differ: spend-based HHI is usually
more concentrated than volume-based when the expensive supplier is also the small-volume
specialist. It names the threshold convention and notes that thresholds differ across
institutions. It labels the tier-2 claim as fact, inference, or unresolved. A weak answer computes
one index and treats a plausible tier-2 link as established.

**Exercise 9 rubric.**

A strong answer uses Type 2 where history matters to the analytic question — pack size affects
demand comparability — and Type 1 for corrections that should overwrite, such as a misspelled
address. It retires the divested location with a valid-to date and creates a new key, preserving
history on the old one. A weak answer applies Type 2 to everything, producing a correct but
unusable dimension, or Type 1 to everything, making last year's report unreproducible.

## 14. Interview questions

**1. "What is the difference between a party, a location, and a node?"**

A strong answer separates legal responsibility, physical place, and model membership, and gives an
example where one party operates three locations of which only two are nodes. A weak answer treats
them as synonyms, which usually means the candidate has only seen one system's data model.

**2. "Why can two analysts get different product counts from the same company?"**

A strong answer names the grain: trade item versus SKU versus material, pack-level splits, and
the difference between active and total records. A weak answer says "data quality", which explains
nothing and predicts nothing.

**3. "Walk me through converting a customer order in cases into a production requirement in
eaches."**

A strong answer states the ladder, the conversion factors, where rounding happens, and why
rounding at each step is wrong. It mentions that the factors are business data with effective
dates. A weak answer multiplies by one number and does not mention rounding or factor versioning.

**4. "What is a phantom item and why does it matter to a data scientist?"**

A strong answer explains that a phantom is a logical assembly never stocked, that MRP explodes
through it, and that treating it as stocked creates fictitious inventory and an extra lead time.
A weak answer says it is "a BOM item that does not exist", which is close but does not explain
the analytical consequence.

**5. "How would you detect that a BOM explosion used the wrong revision?"**

A strong answer compares the explosion against the standard cost for the period, checks the
transaction's production date against the revision's valid-from date, and looks for a systematic
difference concentrated in components that changed. A weak answer says "check the BOM", without
saying which BOM or against what.

**6. "Why is a lane not a route?"**

A strong answer defines the lane as a commercial O-D-mode-service object and the route as the
physical path, and notes that one lane can be served by several routes and one route can carry
several lanes. A weak answer uses the words interchangeably, which will produce a lane table with
one row per sailing.

**7. "How would you represent transit time in a planning dataset?"**

A strong answer says a distribution: mean, standard deviation, and percentiles, by lane and by
season, with the observation window stated. It notes that planning systems store a scalar and
that the analyst must reconstruct the distribution from shipment history. A weak answer says
"average lead time", which is the number that caused the problem.

**8. "What does HHI measure, and what does it miss?"**

A strong answer gives the sum of squared shares, distinguishes the 0–1 and 0–10,000 scales, and
notes that it misses substitutability, qualification time, and tier-2 concentration. A weak
answer gives the formula and stops, or confuses it with a supplier count.

**9. "Why can a company with five suppliers still be single-sourced?"**

A strong answer explains shared tier-2 dependencies, qualification lock-in, and geographic
concentration, and gives the Tōhoku or semiconductor pattern as an illustration. A weak answer
says "because they all use the same raw material", which is a guess rather than a mechanism.

**10. "What is the tier-n problem in one sentence, and why is it hard to fix?"**

A strong answer says it is the loss of visibility beyond direct suppliers, and that it is hard
because the information is commercially sensitive, costly to obtain, and changes faster than it
can be mapped. A weak answer says "suppliers do not share data", which names a symptom rather
than the structural cause.

**11. "When would you use a Type 2 slowly changing dimension for an item?"**

A strong answer ties the choice to an analytic question that requires historical attribute values
— pack size for demand comparability, or category for like-for-like reporting — and contrasts it
with a Type 1 correction that should overwrite. A weak answer says "always use Type 2 for
history", which produces an unusable dimension.

**12. "A regulation requires a digital product passport. What changes in your data model?"**

A strong answer identifies the shift from item-level identity to instance- or batch-level
identity, the need for a resolvable unique identifier, the persistence rules across repurposing
and recycling, and the access-control layer. A weak answer says "add a QR code", which is the
carrier rather than the model.

## 15. Key terms

- **Party** — an organisation with legal or commercial identity.
- **Location** — any place material or capacity can be, physical or virtual.
- **Node** — a location admitted into a particular network model.
- **Asset** — equipment or a vehicle that performs work.
- **Item** — anything bought, made, stored, or sold.
- **Trade item** — an item at a defined packaging level that is ordered or sold.
- **SKU** — the item-at-location combination a business plans and stocks.
- **GTIN** — Global Trade Item Number, unique per trade item per pack level.
- **Indicator digit** — the GTIN-14 digit that distinguishes pack level; 9 means variable measure.
- **GLN** — Global Location Number, identifying a party or a place.
- **SSCC** — Serial Shipping Container Code, identifying one logistics unit.
- **Base UOM** — the unit in which inventory is held.
- **Order UOM** — the unit in which a customer buys.
- **Catch-weight** — a product whose units vary in weight; measured per instance.
- **Inner pack** — a bundle of eaches inside a case.
- **Pallet** — a unit load handled as one object.
- **Lane** — directional origin-destination connection with mode and service.
- **Route** — the physical path a vehicle takes.
- **Itinerary** — a multi-leg sequence with transshipment.
- **BOM** — structured component list for a parent item.
- **Phantom item** — a logical assembly never stocked, exploded through by MRP.
- **Effectivity** — the date range in which a BOM or change applies.
- **Alternate** — an approved replacement component.
- **Tier** — depth of supplier relationship.
- **Articulation point** — a node whose removal disconnects the graph.
- **HHI** — Herfindahl-Hirschman Index, the sum of squared shares.
- **Surrogate key** — a generated key with no business meaning.
- **SCD** — slowly changing dimension; Type 1 overwrites, Type 2 versions.
- **Virtual location** — a location with no physical footprint, such as in-transit.
- **DPP** — Digital Product Passport, the EU product data container under ESPR.

## 16. Sources

**Peer-reviewed and academic**

- Matsuo, H. (2014). "Implications of the Tohoku earthquake for Toyota's coordination mechanism:
  Supply chain disruption of automotive semiconductors." *International Journal of Production
  Economics*. DOI: 10.1016/j.ijpe.2014.07.010.
- Ramani, V., Ghosh, D., and Sodhi, M. S. (2022). "Understanding systemic disruption from the
  Covid-19-induced semiconductor shortage for the auto industry." *Omega*. Open access via
  PubMed Central PMC9363154.
- Fujimoto, T. (2011). "Supply Chain Competitiveness and Robustness: A Lesson from the 2011
  Tohoku Earthquake and Supply Chain 'Virtual Dualization'." MMRC Discussion Paper 362,
  University of Tokyo.
- Berthou, A., Haramboure, A., and Samek, L. (2024). "Mapping and testing product-level
  vulnerabilities in granular production networks." OECD STI Working Paper.
  DOI: 10.1787/9bcde495-en.
- Ivanov, D. (2025). "Conceptual and formal models for design, adaptation, and control of digital
  twins in supply chain ecosystems." *Omega*, 137.
- Roman et al. (2025). "State of the Art of Digital Twins in Improving Supply Chain Resilience."
  *Logistics*, 9(1), 22. MDPI.

**Standards and official**

- GS1 General Specifications (current edition), including GTIN, GLN, SSCC, GIAI, GRAI, and
  EPCIS 2.0 / CBV 2.0.
- GS1 Package and Product Measurement Standard, Release 3.2 (March 2024).
- GS1 GTIN Management Rule 138, "Trade item sold in variable measure varies in total weight."
- GS1 US, "Packaging Levels in Data Hub," and "Working with Variable Measure."
- GS1, "2024/25 Year in Review," and the GDSN statistics dashboard.
- GS1, "Accelerating Value Chain Digitalisation," position paper, June 2025.
- European Commission, Digital Product Passport pages and FAQ, including the battery and textile
  product-group pages, 2026.
- Regulation (EU) 2024/1781 (ESPR); Regulation (EU) 2023/1542 (Batteries); Implementing
  Regulation (EU) 2026/1778 (DPP Registry).
- Directive (EU) 2024/1760 (CSDDD); Directive (EU) 2025/794 (stop-the-clock); Directive
  (EU) 2026/470 (Omnibus I).
- U.S. FDA, "Requirements for Additional Traceability Records for Certain Foods," 87 FR 70910;
  proposed compliance date extension, 90 FR 38084 (7 August 2025).
- U.S. Congressional Research Service, "The Motor Vehicle Supply Chain: Effects of the Japanese
  Earthquake and Tsunami," R41831, May 2011.

**Industry and analyst**

- ECCMA, "Procurement Master Data Quality Analysis," January 2025. Industry association estimate;
  grade D.
- GS1 US, "Powering Supply Chain Confidence Research Report," March 2025, n=505. Grade C.
- Sphera, "N-Tier Transparency: 10 Key Findings," survey report, March 2025, n=250. Vendor
  survey; grade D.
- Automotive News Research and Data Center, "Chip Shortage Report," April 2021, n=475. Trade
  survey; grade C.
- Swiss Re, "De-risking supply chains: a deep dive into the automotive industry," February 2023.
  Insurer analysis; grade C/D.
- CIMdata / AD PLM Action Group, "Multiple View Bill of Materials"; PDS Vision / PTC, "BOM: A
  Foundation for PLM Best-Practices," January 2025; Oracle, JD Edwards EnterpriseOne 643145.1 on
  phantom BOMs, updated April 2025. Vendor and consortium documentation; grade D.
- Borouge, supply chain network digital twin case, SPE/ADIPEC 2024. Company conference paper;
  grade C.
- Foley & Lardner, "Best Practices for Supply Chain Mapping," October 2025. Legal practitioner
  guidance; grade D.
- Bayer AG MRO data transformation case, reported via SPARROW Parts, 2025. Vendor case study;
  grade D.

**Note on citations.** Survey figures are reported as the surveys state them and are not
independently verified. Where two GS1 sources give slightly different 2D barcode counts (51
countries in the year-in-review, 45–48 elsewhere), the text says "around 50". Where a date is
still moving, the text gives the current status and tells you to re-check.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_decision-stack.md`](02_decision-stack.md)
- Foundation: [`01_three-flows-one-network.md`](01_three-flows-one-network.md)
- Next in sequence: [`04_planning-processes.md`](04_planning-processes.md)
- BOM, routing, and capacity: [`07_manufacturing-basics.md`](07_manufacturing-basics.md)
- Warehousing and sourcing: [`09_warehousing-and-sourcing.md`](09_warehousing-and-sourcing.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Return to the part map: [`../_index.md`](../_index.md)
