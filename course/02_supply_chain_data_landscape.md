# Module 02 — The Supply Chain Data Landscape

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 24 hours (9 lecture, 9 lab, 6 project)
> **Track fit:** Common core (all three tracks)
> **Prerequisites:** Module 01 and 01A
> **Tools:** Python, DuckDB, a spreadsheet, access to at least one real transaction extract
> **Industry question:** "Where does this number come from, who owns it, at what grain and latency, and what is it not telling me?"

---

## 1. Why industry cares

Supply chain analytics does not start with a dataset. It starts with a **system landscape**: an ERP that believes it holds the truth, a warehouse system that holds the physical truth, a transport system that holds the movement truth, a retailer's EDI feed that holds the customer's truth, and a planning spreadsheet that holds the version everyone actually uses. The data landscape determines what is possible before any model is chosen. A demand model cannot be better than the grain and latency of the sales signal it consumes. An inventory model cannot be better than the snapshot discipline of the warehouse system. A supplier risk model cannot be better than the completeness of the supplier master. Most "model" problems are **source problems wearing a model's clothes**.

| Symptom | Underlying data cause | What the team usually does | What it should do |
|---|---|---|---|
| Forecast is accurate in aggregate, wrong by store | Sales data allocated, not observed, at store level | Adds features | Establish true store-item POS capture |
| Inventory keeps drifting from the system | Snapshot timing and cycle-count adjustments | Reconciles monthly | Fix snapshot cut-off and adjustment posting |
| Promotion lift looks implausible | Stockouts censor sales; no availability flag | Removes outliers | Model censored demand with availability data |
| Supplier lead time is always "about two weeks" | No receipt timestamps at line level | Uses the average | Capture ASN, gate-in, and put-away events |
| New items have no history | Item master recycles codes | Backfills manually | Enforce unique GTIN and item lifecycle status |
| Same customer counted twice | Duplicate ship-to records | Deduplicates in the report | Entity resolution in master data |

The 2026 context adds pressure from two directions. First, **data volume and variety** have grown faster than governance: POS, e-commerce clickstream, IoT telemetry, carrier events, supplier portals, and external signals all arrive with different keys, units, and clocks. Second, **regulation and commercial sensitivity** have tightened: personal data, competitively sensitive pricing, supplier confidentiality, and Scope 3 evidence all constrain what can be collected, joined, and shared. A data-source decision is therefore a legal and commercial decision, not only a technical one.

> **Industry interpretation.** The most valuable data scientist in a planning team is often the one who can draw the system landscape from memory and say which number is trustworthy for which decision.

## 2. Learning outcomes

By the end of this module, you can:

- Map the eleven core supply chain systems to their grain, latency, owner, and decision.
- Distinguish master, transaction, reference, telemetry, and external data, and explain the governance each requires.
- Audit master data for drift, duplication, orphan records, and lifecycle errors, and quantify the forecast impact.
- Evaluate a data source on fitness, grain, latency, history, quality, cost, licence, and stability using a weighted framework.
- Identify personal, confidential, and commercially sensitive data in a supply chain context and state the access constraint.
- Apply the standard data-quality dimensions to supply chain records with concrete tests.
- Trace a metric back to its source system and document lineage and version.
- Critique a public benchmark dataset for licence, provenance, known defects, and fitness for a stated decision.
- Build a data-source map for a retailer and a manufacturer.

## 3. Mental model

### 3.1 From event to decision

```
  PHYSICAL EVENT            SYSTEM RECORD              DATA PRODUCT           DECISION
  goods move          →     WMS transaction      →     inventory fact    →    replenish?
  order placed        →     OMS / EDI 850        →     demand fact       →    promise?
  invoice issued      →     ERP AP               →     cost fact         →    margin?
  sensor reads        →     IoT platform         →     condition event   →    release?
        ↓                         ↓                        ↓
   master data  ──────────► keys, units, calendars ──────► joins
        ↓
   governance: ownership, licence, quality, lineage
```

Every layer can break the chain. The mental model forces three questions at each arrow: **Is the event captured? Is it keyed to the same entity? Is it aligned to the same clock?**

### 3.2 Five questions for any source

| Question | Why it matters | Typical answer format |
|---|---|---|
| What decision does it serve? | Prevents data hoarding | Replenishment at store-item-week |
| What is the grain? | Determines joins and aggregation | One row per order line |
| What is the latency? | Determines whether the decision can wait | 24 hours after event |
| Who owns it? | Determines who fixes it | Supply planning systems team |
| What is it not? | Prevents overreach | Orders, not demand; excludes returns |

### 3.3 The source hierarchy of trust

```
 1. Audited financial records        (highest confidence, lowest frequency)
 2. Contractual partner messages     (EDI 850/856/810, defined semantics)
 3. Internal transactional systems   (ERP, WMS, TMS; operationally controlled)
 4. Operational telemetry            (sensors, telematics; high volume, noisy)
 5. Derived and modelled data        (forecasts, allocations, plans)
 6. External and purchased data      (weather, macro, indices; variable quality)
 7. Manual spreadsheets              (lowest confidence, highest influence)
```

The paradox of the last row: the least trustworthy source often drives the most important decision. Fixing that is a governance project, not a modeling project.

## 4. Core concepts

### 4.1 Systems of record

| System | Full name | Primary object | Grain | Typical latency | Owner | Decision served |
|---|---|---|---|---|---|---|
| ERP | Enterprise resource planning | Purchase orders, inventory, finance | Item–location–day / document | Minutes–hours | Finance / IT | Buy, cost, account |
| WMS | Warehouse management system | Warehouse tasks, stock, locations | Item–location–bin–task | Real time | Warehouse operations | Receive, put away, pick, ship |
| TMS | Transportation management system | Shipments, loads, rates | Shipment–leg–lane | Minutes–hours | Logistics | Mode, route, tender, pay freight |
| OMS | Order management system | Orders, allocations, promises | Order line | Real time | Customer operations | Promise, allocate, release |
| POS | Point of sale | Basket, line, tender | Transaction line | Seconds–minutes | Retail operations | Replenish, price, assort |
| PIM | Product information management | Item attributes, descriptions, images | Item–market–channel | Hours–days | Merchandising / data | Publish, sell, comply |
| PLM | Product lifecycle management | BOM, specifications, changes | Item–revision | Days | Engineering | Design, source, change |
| MES | Manufacturing execution system | Production orders, operations, quality | Work order–operation–shift | Seconds–minutes | Plant operations | Schedule, execute, release |
| APS | Advanced planning and scheduling | Plans, schedules, capacity | Item–resource–bucket | Daily–weekly | Planning | Plan, sequence, capacity |
| CRM | Customer relationship management | Accounts, contacts, opportunities, service | Account–interaction | Minutes | Sales / service | Sell, serve, retain |
| SRM | Supplier relationship management | Suppliers, contracts, performance | Supplier–item–period | Daily–monthly | Procurement | Source, score, develop |

No single system is the truth for everything. The ERP usually owns financial inventory value; the WMS owns physical location and quantity; the OMS owns the customer promise; the TMS owns the movement. "Which inventory number is right?" is therefore an ill-posed question until you name the decision.

### 4.2 Master data

| Domain | Core attributes | Standard or key | Lifecycle state | Drift risk |
|---|---|---|---|---|
| Item | GTIN, description, brand, category, UoM, case pack, weight, volume, shelf life, status | GS1 GTIN | Active, discontinued, replaced | Pack size change, unit change, duplicate item |
| Location | GLN, site type, address, time zone, capacity, calendar | GS1 GLN | Open, closed, seasonal | Store closure, DC conversion, time-zone error |
| Supplier | Legal entity, site, DUNS or GLN, payment terms, Incoterms, lead time | DUNS / GLN | Approved, on hold, blocked | Site change, ownership change, bank detail |
| Customer | Sold-to, ship-to, parent, channel, terms, credit | Customer ID / GLN | Active, blocked | Duplicate ship-to, channel reclassification |
| BOM | Component, quantity per, scrap, effective dates, alternates | Item + revision | Engineering change | Phantom items, alternate not reflected |
| Routing | Operation, work center, standard time, yield | Item + revision | Engineering change | Time standards stale after automation |
| Calendar | Working days, shifts, holidays, fiscal periods | Calendar ID | Annual refresh | Holiday and shift changes |
| Hierarchy | Parent–child relationships for product, location, customer, channel | Surrogate keys | Restated periodically | Reparenting destroys comparability |

Master data is the join layer. A forecast is only as good as the item and location keys that connect sales, inventory, and supply. Most "model" failures that appear as sudden accuracy drops are master-data events: a pack-size change, a new item reusing an old code, a store renumbered, a supplier site moved.

### 4.3 Transaction data

| Record | Typical fields | Source | Grain | Use | Known defect |
|---|---|---|---|---|---|
| Sales order | Order ID, line, customer, item, quantity, requested date, price | OMS, EDI 850 | Order line | Demand, service | Requested date overwritten by promise |
| Shipment | Shipment ID, origin, destination, ship/arrival, mode, weight | TMS, ASN 856 | Shipment / handling unit | Lead time, transport | Status events missing or late |
| Receipt | Receipt ID, PO, item, quantity, condition, date | WMS, ERP | Receipt line | Lead time, supply | Over-receipt not flagged |
| Invoice | Invoice ID, PO, amount, currency, tax, duty | ERP AP, EDI 810 | Invoice line | Landed cost, DPO | Price mismatch after promotion |
| Inventory snapshot | Item, location, on-hand, allocated, in-transit, status | WMS, ERP | Item–location–time | DIO, service, policy | Snapshot gaps, negative stock |
| Forecast | Item, location, bucket, version, value, method | APS, planning tool | Item–location–bucket–version | Planning | Overwritten versions, no history |
| Plan | Supply, production, deployment, allocation | APS, ERP | Item–resource–bucket | Execution | Plan not versioned |

Forecast and plan records deserve special scrutiny: many systems overwrite them in place, so the version a planner saw on the day of the decision is unrecoverable. That makes post-decision evaluation impossible and override analysis unreliable.

### 4.4 Reference data

Reference data is shared, slowly changing, and used to interpret transactions: currency codes (ISO 4217), units of measure (UN/CEFACT Rec 20), country and region codes (ISO 3166), HS commodity codes, UNSPSC categories, calendars, exchange rates, tax rates, emission factors, and carrier service codes. It is small, high-leverage, and frequently unowned. A stale exchange rate or a superseded HS code contaminates every downstream calculation silently.

### 4.5 Telemetry, IoT, and computer vision

| Source | Measures | Grain | Latency | Use | Constraint |
|---|---|---|---|---|---|
| Cold-chain sensor | Temperature, humidity, door events | Reading per device per interval | Minutes | Excursion, release, compliance | Calibration, battery, coverage gaps |
| Vehicle telematics | GPS, speed, idle, fuel, harsh braking | Ping per vehicle | Seconds–minutes | ETA, safety, fuel, emissions | Privacy, driver consent, signal loss |
| RFID | Tag reads at portals and handhelds | Read event | Seconds | Inventory accuracy, receiving | Read rate, metal/liquid interference |
| Computer vision | Shelf availability, damage, count | Frame / detection | Near real time | On-shelf availability, quality | Labelling, false positives, privacy |
| Equipment sensors | Vibration, temperature, cycles | Reading per asset | Seconds–minutes | Predictive maintenance | Asset master mapping |

Telemetry is high-volume and low-signal. The analytical task is usually **event extraction and state reconstruction**, not model building: turning a stream of pings into "shipment arrived", "trailer dwelled 4 hours", "temperature exceeded 8 °C for 35 minutes".

### 4.6 External data

| Category | Examples | Provider type | Use | Main risk |
|---|---|---|---|---|
| Weather | Forecast, observed temperature, precipitation, severe alerts | NOAA, national met services, commercial | Demand drivers, disruption, routing | Spatial resolution, forecast revision |
| Macro | GDP, inflation, unemployment, consumer confidence | World Bank, OECD, national statistics | Demand planning, scenario | Revision, lag, aggregation |
| Trade | Import/export values and volumes by HS code | UN Comtrade, national customs | Sourcing, tariff exposure | Reporting lag, re-exports |
| Freight indices | Container, air, truckload spot rates | Freightos, Drewry, exchange data | Cost forecasting, tendering | Index methodology, lane mismatch |
| Port congestion | Vessel waiting time, berth occupancy | Port authorities, AIS providers | Lead time, disruption | Coverage, definition |
| Search and social | Query volume, sentiment, mentions | Google Trends, social platforms | New product, demand sensing | Representativeness, API limits |
| Commodity prices | Metals, resin, agricultural, energy | LME, World Bank Pink Sheet, exchanges | Should-cost, inflation | Contract vs spot, currency |
| Geopolitical | Conflict, protest, sanctions events | GDELT, ACLED, official notices | Risk monitoring | Coding error, false positives |

External data is seductive because it is easy to acquire and hard to validate. The test is not correlation with the past; it is whether the signal arrives **before** the decision and covers the entities you actually have.

### 4.7 Data movement: EDI, API, flat file, event

| Pattern | Mechanism | Typical latency | Strengths | Weaknesses |
|---|---|---|---|---|
| EDI | X12 or EDIFACT messages over VAN or AS2 | Minutes–hours | Standard semantics, audit trail, partner ubiquity | Rigid, implementation-guide variance, batch |
| API | REST/GraphQL over HTTPS | Seconds–minutes | Flexible, queryable, near real time | Rate limits, versioning, partner uptime |
| Webhook / event | Partner pushes an event | Seconds | Timely, decoupled | Delivery guarantees, ordering, replay |
| Flat file | CSV/SFTP/Excel drop | Hours–days | Universal, simple | No schema, encoding issues, manual steps |
| Database extract | JDBC/ODBC, CDC | Minutes–hours | Complete, schema-known | Load on source, permission complexity |
| Message queue / stream | Kafka, cloud pub/sub | Milliseconds–seconds | High throughput, replay | Operational complexity, ordering |

The supply chain reality is hybrid: an EDI 850 for the order, an API for the tracking event, a flat file for the weekly forecast, and a spreadsheet for the promotion calendar. Module 03 covers the engineering; this module covers the **selection** of which pattern is fit for which decision.

### 4.8 Access pathways

| Pathway | Description | Typical use | Constraint |
|---|---|---|---|
| Internal system access | Direct read from corporate systems | ERP, WMS, TMS | Security review, load, schema volatility |
| Partner exchange | EDI, API, or portal with a supplier, retailer, or carrier | POS 852, ASN, rates | Contract, implementation guide, uptime |
| Commercial purchase | Paid dataset or index | Weather, freight, credit | Licence, cost, redistribution limits |
| Public / open | Government, statistical, open data | Trade, macro, weather | Coverage, licence, quality |
| Synthetic | Generated data for testing and teaching | Labs, method development | Not evidence about the real world |
| Derived | Forecasts, plans, allocations, scores | Planning, benchmarking | Inherits assumptions of the model |

### 4.9 Licensing, confidentiality, and personal data

| Data class | Examples | Risk | Control |
|---|---|---|---|
| Personal data | Loyalty IDs, names, addresses, device IDs | Privacy law, re-identification | Minimisation, pseudonymisation, lawful basis |
| Commercially sensitive | Prices, promotions, margins, volumes by customer | Contract breach, competition law | Need-to-know, aggregation, no competitor exchange |
| Supplier confidential | Quotes, capacity, cost breakdowns | Relationship damage, contract breach | Confidentiality terms, restricted access |
| Customer confidential | Order patterns, inventory levels | Contract breach | Aggregation, purpose limitation |
| Regulated product data | Serial numbers, batch, temperature | Product safety, traceability law | Retention, integrity, access control |
| Public | Published filings, standards | Low | Attribution, version control |

Three rules that prevent most problems. First, **purpose limitation**: data collected for billing is not automatically available for demand modeling. Second, **aggregation as a control**: store-level data from a single small store can identify a customer or reveal a competitor's volume; set minimum cell sizes. Third, **contract before code**: confirm the licence permits the intended use, including derived outputs and model training, before building the pipeline.

### 4.10 Data-quality dimensions

| Dimension | Definition | Supply chain test | Example failure |
|---|---|---|---|
| Completeness | Required fields populated | Share of order lines with a valid item and location | 3% of receipts lack a PO reference |
| Accuracy | Value matches reality | Cycle-count variance vs system quantity | System says 40, shelf has 32 |
| Timeliness | Available when the decision needs it | Lag between event and record | POS arrives 48 h late for daily replenishment |
| Consistency | Same fact agrees across systems | ERP vs WMS inventory reconciliation | 1.8% value difference unexplained |
| Uniqueness | No unintended duplicates | Duplicate order lines by key | Same shipment counted twice |
| Validity | Values in allowed domain | UoM in the approved code list | "CS", "Case", "case " all present |
| Conformity | Follows a defined format | GTIN check digit valid | 7% of GTINs fail check digit |
| Integrity | Relationships hold | Every item has a valid category parent | 120 orphan items |
| Accessibility | Authorized users can retrieve it | Query latency and permission | Planner waits a day for a report |

Fitness for planning is stricter than general data quality: a dataset can be 99% complete and still be unfit because the missing 1% is concentrated in the promoted, high-volume items that drive the decision.

### 4.11 Master-data drift, worked

An item changes case pack from 24 to 12 on 1 July, but the reporting layer still divides by 24.

| Period | Cases sold | True eaches (pack of period) | Reported eaches (pack 24) |
|---|---:|---:|---:|
| H1 (pack 24) | 1,000 | 24,000 | 24,000 |
| H2 (pack 12) | 1,300 | 15,600 | 31,200 |
| Change | +30.0% cases | −35.0% eaches | +30.0% reported |

The case count rises 30%; true unit demand falls 35%; the report shows a 30% increase in "eaches" because the conversion was never updated. A forecast trained on the reported series learns a growth trend that does not exist and over-orders by a wide margin. The root cause is not the model, the forecast horizon, or the promotion calendar; it is an effective-dated attribute that nobody versioned.

The general rule: **any attribute that changes the meaning of a quantity must be effective-dated and joined as of the event date**, not looked up from the current master.

### 4.12 Lineage, provenance, and versioning

Lineage answers "where did this number come from?" at three levels: the **system** (ERP extract, table, view), the **transformation** (join, filter, unit conversion, allocation), and the **model or assumption** (forecast method, override, scenario). Provenance adds who produced it, when, under what licence, and with what known defects. Versioning makes the answer reproducible.

| Artifact | Minimum metadata | Why |
|---|---|---|
| Raw extract | Source system, extract timestamp, query or file hash, row count | Reproduce and detect change |
| Transformation | Code commit, parameters, run ID, input versions | Audit and debug |
| Master-data snapshot | Effective date, version, change log | Correct historical joins |
| Forecast / plan | Version, method, horizon, override flag, author | Evaluate decisions, not just models |
| External dataset | Provider, vintage, retrieval date, licence | Defend against revision and rights claims |

### 4.13 Source-selection framework

Score each candidate source 1–5 on weighted criteria. The weights depend on the decision; the table below is a starting point for a demand-planning decision.

| Criterion | Weight | What a 5 looks like | What a 1 looks like |
|---|---:|---|---|
| Fitness for the decision | 30% | Measures the construct directly at the needed grain | Proxy with known bias |
| Grain | 15% | Matches the decision grain exactly | Requires allocation or aggregation |
| Latency | 15% | Available before the planning cut-off | Arrives after the decision |
| History | 10% | Multiple years, stable definitions | One year or restated history |
| Quality | 20% | Complete, accurate, consistent, validated | Known gaps in critical segments |
| Cost, licence, stability | 10% | Affordable, permitted, contractually stable | Expensive, restricted, may disappear |

| Source | Fitness | Grain | Latency | History | Quality | Cost/licence | Weighted score |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: ERP orders | 3 | 3 | 5 | 5 | 4 | 5 | 3.90 |
| B: Retail POS 852 | 5 | 5 | 3 | 4 | 3 | 3 | 4.00 |

Source A is internal ERP orders (item-week, one-day lag, three years, quality 0.92); Source B is retailer POS via EDI 852 (item-store-day, two-day lag, two years, quality 0.85). Scoring: A = 0.90 + 0.45 + 0.75 + 0.50 + 0.80 + 0.50 = 3.90; B = 1.50 + 0.75 + 0.45 + 0.40 + 0.60 + 0.30 = 4.00. POS wins narrowly — but only if the 852 feed covers the stores in scope and the two-day lag fits the cut-off. Weighting latency at 25% reverses the ranking, which is exactly the finding the framework should surface.

## 5. Data, standards, and workflow

| Standard | Applies to | What it standardizes |
|---|---|---|
| GS1 General Specifications | Items, locations, logistics units | GTIN, GLN, SSCC, barcode symbology |
| ANSI X12 | North American EDI | 850, 855, 856, 810, 846, 852, 940, 945, 997 |
| UN/EDIFACT | International EDI | ORDERS, DESADV, INVOIC |
| EPCIS 2.0 | Event traceability | Object, aggregation, transformation, association events |
| UNSPSC | Spend classification | Product and service category codes |
| Harmonized System (HS) | Customs | Commodity classification |
| ISO 3166 / 4217 / 8601 | Reference data | Country, currency, date and time |
| GS1 EDI implementation guides | Partner messages | Segment and qualifier usage per transaction |

### 5.1 Data-source map: a grocery retailer

| Source | System | Grain | Latency | Key fields | Decision served | Access | Quality risk |
|---|---|---|---|---|---|---|---|
| POS transactions | POS | Basket line | Minutes | Store, item, qty, price, promo | Replenishment, price | Internal | Scan errors, voids |
| E-commerce orders | OMS | Order line | Real time | Customer, item, promise, channel | Fulfilment, promise | Internal | Cancellations, substitutions |
| Store inventory | Store system / WMS | Store–item–day | Daily | On-hand, allocated, shrink | Replenishment | Internal | Count accuracy |
| DC inventory | WMS | DC–item–bin | Near real time | On-hand, in-transit, status | Deployment | Internal | Location master |
| Shipments | TMS | Shipment–leg | Minutes | Lane, mode, ETA, cost | Transport, ETA | Internal | Late status |
| Supplier POS (852) | EDI | Item–store–week | Weekly | Sales, inventory, returns | Vendor-managed inventory | Contract | Coverage, definitions |
| Item master | PIM / ERP | Item–market | Daily | GTIN, pack, size, status | All | Internal | Pack change, duplicates |
| Loyalty | CRM | Household–basket | Daily | Pseudonymous ID, basket | Basket affinity, CLV | Restricted | Consent, bias |
| Weather | External | Grid–day | Hourly | Temperature, precipitation | Demand sensing | Licence | Spatial mismatch |
| Promotions | Planning | Item–store–week | Weekly | Mechanic, discount, feature | Lift, forecasting | Internal | Late changes |

### 5.2 Data-source map: a discrete manufacturer

| Source | System | Grain | Latency | Key fields | Decision served | Access | Quality risk |
|---|---|---|---|---|---|---|---|
| Customer orders | ERP / EDI 850 | Order line | Minutes | Item, qty, date, priority | MPS, ATP | Internal | Requested date changes |
| Forecast | APS | Item–bucket–version | Weekly | Qty, version, method | MRP, capacity | Internal | Overwritten versions |
| Production orders | MES | Work order–operation | Real time | Resource, qty, yield, scrap | Scheduling, OEE | Internal | Manual backflush |
| BOM and routing | PLM / ERP | Item–revision | On change | Component, qty, operation, time | MRP, costing | Internal | Phantom items, alternates |
| Supplier schedules | SRM / EDI 830 | Supplier–item–week | Weekly | Release, firm horizon | Procurement | Partner | Acknowledgement lag |
| ASN and receipts | EDI 856 / WMS | Shipment–line | Hours | SSCC, qty, lot, arrival | Lead time, inventory | Partner | ASN vs physical mismatch |
| Quality results | QMS / LIMS | Lot–test | Hours | Characteristic, result, disposition | Release, SPC | Internal | Test latency |
| Machine telemetry | IoT / OPC UA | Asset–reading | Seconds | Vibration, temperature, state | Maintenance, OEE | Internal | Asset mapping |
| Freight and customs | TMS / broker | Shipment–entry | Daily | Lane, duty, HS code, cost | Landed cost | Partner | HS classification |
| Supplier financial risk | External | Supplier–quarter | Quarterly | Credit score, filings | Risk scorecard | Licence | Private-firm coverage |

### 5.3 Profiling a source before trusting it

The following DuckDB script is runnable as written and demonstrates the first pass any analyst should run: row count, nulls, unit inconsistency, duplicate keys, and date range.

```sql
CREATE OR REPLACE TABLE orders AS
SELECT * FROM (VALUES
  ('SO1', 'SKU1', 'ST01', DATE '2026-01-02', 10,  'EA'),
  ('SO2', 'SKU1', 'ST01', DATE '2026-01-02', 2,   'CS'),
  ('SO2', 'SKU2', 'ST02', DATE '2026-01-03', NULL,'EA'),
  ('SO3', 'SKU1', 'ST01', DATE '2026-01-03', 5,   'ea')
) AS t(order_id, sku, location, order_date, qty, uom);

-- 1. Coverage and null profile
SELECT count(*)                            AS rows,
       count(*) FILTER (WHERE qty IS NULL) AS null_qty,
       count(DISTINCT uom)                 AS uom_variants,
       min(order_date)                     AS first_date,
       max(order_date)                     AS last_date
FROM orders;

-- 2. Duplicate key candidates
SELECT order_id, sku, count(*) AS n
FROM orders
GROUP BY order_id, sku
HAVING count(*) > 1;

-- 3. Unit variants, normalised for review
SELECT lower(trim(uom)) AS uom_norm, count(*) AS n
FROM orders GROUP BY 1 ORDER BY n DESC;
```

The output shows four rows, one null quantity, four nominal unit variants that collapse to two after normalisation (`ea`, `cs`), and one duplicated order–SKU pair. None of these is a modeling problem yet; each is a source problem that will become one.

Workflow for the module project: (1) inventory the sources that touch the chosen decision; (2) record grain, latency, owner, and licence for each; (3) profile each extract for completeness, uniqueness, validity, and date coverage; (4) trace one metric end-to-end and write the lineage; (5) score candidate sources for the decision; (6) document known defects and the workaround.

## 6. Industry case studies

### 6.1 M5 (Walmart) retail dataset, 2020–2021

**Context and characteristics.** Walmart released daily unit sales for 42,840 hierarchical series across 3,049 products and 10 stores in three U.S. states, with 1,913 training days, calendar and price data, and a 28-day forecast horizon. The dataset includes intermittent series, promotions, holidays, and SNAP day effects, and it is genuinely censored: a zero can mean no demand or no stock.

**What it teaches.** Hierarchy and weighting matter (WRMSSE weights by revenue), exogenous variables such as price and events matter, and a benchmark can be useful without being representative of a full retail network.

**Limitation.** Three states, ten stores, and a fixed product set; results do not transfer automatically to fresh, fashion, or e-commerce. [VERIFY: exact product and store counts if quoted in a deliverable; the M5 methods repository is authoritative.]

### 6.2 Corporación Favorita grocery sales competition, 2018

**Context and characteristics.** A Kaggle competition using daily sales for an Ecuadorian grocery retailer, with store metadata, item metadata, promotion flags, holidays, and daily oil prices. It shows strong promotion effects, many intermittent item–store series, a small number of stores relative to items, and an exogenous macro variable (oil) correlated with regional economic activity.

**What it teaches.** Promotions create large, sparse spikes that dominate error metrics; external macro signals can help at aggregate level but are weak at item level; unit sales and transaction counts are different constructs.

**Limitation.** A single national retailer, anonymized items, and a competition metric that does not reflect holding or shortage cost. [VERIFY: exact item and store counts before quoting.]

### 6.3 Rossmann store sales competition, 2015

**Context and characteristics.** Daily sales for 1,115 German drugstores from 2013 to 2015, with store metadata, promotion flags, state and school holidays, and a six-week forecast horizon. The store-level daily series show strong weekly and annual seasonality, promotion effects, and closures; a relatively small number of series makes store-level modeling feasible.

**What it teaches.** Store metadata (type, assortment, competition distance) explains a large share of variation; holiday calendars must be local, not national; closed days are structural zeros, not demand zeros.

**Limitation.** Store-level rather than item-level, so it cannot teach assortment or substitution; the metric (RMSPE) is undefined on closed days, which forced explicit handling.

### 6.4 UCI Online Retail II, 2009–2011

**Context and characteristics.** Transaction-level data from a UK-based online retailer of giftware, covering 1 December 2009 to 9 December 2011, with invoices, stock codes, quantities, prices, customer IDs, and countries. Cancellations and returns appear as negative quantities, missing customer IDs are common, prices include zero and negative adjustments, and the item catalogue is non-standardized.

**What it teaches.** Real transaction data requires sign conventions, return handling, and customer identity resolution before any demand model. It is an excellent dataset for teaching data quality because every defect is visible and consequential.

**Limitation.** One retailer, business-to-business and business-to-consumer mixed, no inventory data, so stockouts cannot be identified. [VERIFY: exact row count and date range against the UCI dataset page before quoting.]

### 6.5 DataCo Smart Supply Chain, 2018–2019

**Context and characteristics.** A public dataset distributed through Mendeley Data describing orders, shipping, and profitability for a global e-commerce-like supply chain, widely used for teaching. Order-level records include shipping mode, delivery status, late-delivery risk, customer segment, and profit ratios; the data is synthetic or anonymized rather than a raw operational extract.

**What it teaches.** Late-delivery prediction and cost-to-serve analysis are tractable at order level; class imbalance and label definition dominate model quality.

**Limitation.** Because it is synthetic or heavily processed, it cannot support claims about real carrier performance or real demand distributions. [VERIFY: exact order and attribute counts and the generation method before using it as evidence.]

## 7. Labs

### Lab 2.1 — Build a data-source map

**Objective and data.** Map every source that could serve one planning decision, with grain, latency, owner, licence, and quality risk, using a provided system inventory and a one-page decision brief (for example, "replenish 400 stores weekly").

**Steps.**
1. List every candidate source, including spreadsheets and partner feeds.
2. Fill the map: grain, latency, key fields, decision served, access pathway, owner, licence.
3. Mark which sources are systems of record and which are derived.
4. Identify three gaps where the decision needs data no source provides.
5. Draw the flow from event to decision, mark where joins occur, and score the top three sources with the §4.13 framework.

**Deliverable.** A one-page map for a retailer and one for a manufacturer, plus a gap list.

**Acceptance criteria.** At least 10 sources per map; grain and latency for every row; licence and owner named; joins identified by key; at least three gaps with a proposed remedy.

**Stretch.** Add a latency budget showing whether each source arrives before the planning cut-off.

### Lab 2.2 — Master-data drift audit

**Objective.** Detect and quantify the effect of master-data change on a reported metric.

**Data.** Item master with effective dates and a two-year sales extract in cases.

**Steps.**
1. Identify every attribute change that affects quantity meaning: pack size, UoM, item replacement.
2. Reconstruct the true quantity using the attribute valid on the event date.
3. Compare reported and reconstructed series; quantify the error in units and percent.
4. Identify items with orphan category parents or duplicate GTINs.
5. Estimate the forecast impact by fitting a simple trend to both series.
6. Write the remediation: effective dating, change alert, reconciliation test.

**Deliverable.** A drift register, a corrected series, and a one-page impact memo.

**Acceptance criteria.** At least five drift events; arithmetic shown; forecast impact quantified; remediation specifies an owner and a control.

**Stretch.** Implement the as-of join in DuckDB and compare with the naive current-master join.

### Lab 2.3 — Dataset critique and licence audit

**Objective.** Assess a public dataset for fitness, provenance, and legal use.

**Data.** One of M5, Favorita, Rossmann, UCI Online Retail II, or DataCo.

**Steps.**
1. Record the licence, access date, collection period, and provider.
2. Identify the grain, the unit, and what a zero means.
3. List known defects: censoring, missing IDs, cancellations, synthetic records.
4. Test one quality dimension with a reproducible query.
5. State one decision the dataset can support and one it cannot.
6. Write the citation and the limitations paragraph for a report.

**Deliverable.** A two-page dataset card.

**Acceptance criteria.** Licence and access date stated; grain and zero semantics explicit; at least three defects with evidence; supported and unsupported uses distinguished; no claim beyond the data.

**Stretch.** Compare two datasets on the same quality dimensions and recommend which to use for a stated decision.

## 8. Assignment / project

**Data-source map and master-data audit (individual, 8 hours).**

Choose a product category and a decision. Produce:

1. **System landscape** — the systems, their owners, and the flows between them.
2. **Source map** — at least 12 sources with grain, latency, key fields, access, licence, and quality risk.
3. **Master-data audit** — five drift or quality findings with quantified impact.
4. **Source-selection memo** — a weighted comparison of three candidates for the decision, with a sensitivity test.
5. **Lineage record** — one metric traced from source table to reported value, including every transformation.
6. **Data-protection note** — personal, confidential, and commercially sensitive elements, and the controls applied.

The memo must be readable by a planning director who does not know the systems by name.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| System landscape | 20% | Lists systems without roles | Maps systems to objects and owners | Explains truth conflicts and decision implications |
| Source evaluation | 25% | Describes sources qualitatively | Uses grain, latency, quality, licence | Weighted, sensitivity-tested, evidence-based |
| Master-data analysis | 20% | Finds duplicates | Quantifies drift impact | Links drift to forecast and inventory consequences; proposes controls |
| Quality dimensions | 15% | Checks nulls only | Applies five dimensions with tests | Designs tests tied to decision fitness |
| Governance and licence | 10% | Mentions GDPR | Identifies data classes and constraints | Purpose limitation, aggregation, contractual analysis |
| Communication | 10% | Technical dump | Clear memo | Decision-ready; states what would change the recommendation |

## 10. Regulatory, ethical, and commercial lens

- **Privacy.** Loyalty, e-commerce, and telemetry data are personal data under GDPR and CCPA/CPRA. Minimise, pseudonymise, define a lawful basis, and set retention. Store-level data can be re-identifying in small markets.
- **Contract, licence, and competition.** Retail POS feeds, carrier data, and purchased indices carry redistribution limits, and derived outputs or model training may be restricted even when raw access is allowed. Exchanging forward-looking prices, promotions, or volumes with competitors through a shared platform can constitute information exchange; aggregation and neutrality are legal safeguards.
- **Traceability.** DSCSA, EU FMD, and food-safety rules require lot- and serial-level data with retention and integrity controls; EPCIS is the common event standard.
- **Sustainability reporting.** CSRD/ESRS and CBAM require supplier activity data and emission factors with documented boundaries; spend-based estimates must be labelled.
- **Trade compliance.** HS codes, origin, and denied-party screening depend on accurate master data. A misclassified item is a customs risk.
- **Equity.** Data coverage is uneven: small suppliers and independent stores are less instrumented, so models trained on well-instrumented partners can systematically under-serve the rest.

## 11. Failure modes and red flags

| Failure mode | Early signal | Consequence | Countermeasure |
|---|---|---|---|
| Treating orders as demand | Forecast error spikes when orders batch | Inventory amplification | Use POS or consumption where available |
| Ignoring censoring | Zero sales in promoted weeks | Understated demand, repeat stockout | Availability flags, censored-demand models |
| Current master joined to history | Step change in a metric with no event | False trend | As-of joins on effective dates |
| One system assumed to be truth | ERP and WMS inventory disagree | Wrong replenishment | Define truth by decision |
| Latency ignored | Model uses data that arrives after cut-off | Plan always one cycle late | Latency budget per source |
| Licence assumed | "It's public" | Legal exposure, takedown | Licence review before pipeline |
| Duplicate customer records | Same account twice in reports | Distorted segmentation | Entity resolution |
| Missing UoM | Mixed cases and eaches | 12–48× error | UoM validation at ingest |
| External data over-trusted | Strong backtest, weak live signal | False confidence | Test lead time and coverage |
| No lineage | Cannot reproduce a number | Audit failure | Metadata and versioning |
| Synthetic data treated as evidence | Clean results, no operational fit | Wrong conclusions | Label synthetic explicitly |
| Data hoarding | Every source requested "for later" | Cost, risk, no use | Purpose limitation |

## 12. Interview and discussion questions

1. **"ERP and WMS show different inventory. Which is right?"** *Hint:* Neither universally; ERP for financial value, WMS for physical position; reconcile and state the decision.
2. **"How would you evaluate a new POS data feed?"** *Hint:* Grain, latency, coverage, completeness, unit consistency, licence, and whether it arrives before the cut-off.
3. **"What does a zero in sales data mean?"** *Hint:* No demand, no stock, not ranged, store closed, or scan failure; you cannot model until you distinguish them.
4. **"How do you handle a case-pack change mid-year?"** *Hint:* Effective-dated master, as-of join, reconstruct true units, add a change alert.
5. **"What is master-data drift and why does it matter to a forecast?"** *Hint:* Attribute change alters the meaning of quantity or hierarchy; produces false trends and broken joins.
6. **"Which data-quality dimension matters most?"** *Hint:* Depends on the decision; fitness is about the segment that drives the outcome, not the overall percentage.
7. **"How would you assess an external freight index?"** *Hint:* Methodology, lane coverage, revision policy, correlation with your contracted rates, licence.
8. **"When would you refuse to use a dataset?"** *Hint:* Licence prohibits the use, grain cannot support the decision, or known bias invalidates inference.
9. **"How do you protect commercially sensitive data in a shared analysis?"** *Hint:* Aggregation thresholds, purpose limitation, contractual terms, access control, no competitor-level detail.
10. **"Why is the spreadsheet often the most influential source?"** *Hint:* It encodes the real planning rules and overrides; govern it rather than pretend it does not exist.

## 13. Key terms

| Term | Definition |
|---|---|
| APS | Advanced planning and scheduling system for constrained plans. |
| ASN | Advance ship notice (X12 856 or EDIFACT DESADV) describing a shipment. |
| Censoring | Unobserved true demand because supply was unavailable. |
| CRM | Customer relationship management system. |
| Data lineage | The documented path from source to reported value. |
| Data provenance | Origin, owner, time, licence, and processing history of data. |
| EDI | Electronic data interchange using standardized message formats. |
| EPCIS | GS1 standard for supply-chain event capture and sharing. |
| ERP | Enterprise resource planning system; financial and transactional core. |
| GTIN | GS1 global trade item number. |
| Master data | Slowly changing reference entities such as item, location, supplier. |
| MES | Manufacturing execution system. |
| OMS | Order management system. |
| PIM | Product information management system. |
| POS | Point of sale; the retail sales transaction record. |
| Reference data | Shared code sets such as currency, UoM, country, HS code. |
| RFID | Radio-frequency identification for tag-based tracking. |
| SRM | Supplier relationship management system. |
| Telemetry | Automated sensor and device measurements transmitted remotely. |
| TMS | Transportation management system. |
| UNSPSC | United Nations Standard Products and Services Code. |
| WMS | Warehouse management system. |

## 14. Sources and further reading

**Standards and regulation**
- GS1, *General Specifications*; GTIN, GLN, SSCC; EPCIS 2.0. GS1, current editions.
- ANSI X12, transaction sets 850, 855, 856, 810, 846, 852, 940, 945, 997. Accredited Standards Committee X12.
- UN/EDIFACT, ORDERS, DESADV, INVOIC. UNECE. UNSPSC and Harmonized System nomenclature. UNDP and World Customs Organization.
- Regulation (EU) 2016/679 (GDPR); California Consumer Privacy Act as amended by CPRA; FDA DSCSA requirements, including enhanced tracing from 27 November 2023.

**Peer-reviewed and foundational**
- Lee, H., Padmanabhan, V., Whang, S. "Information Distortion in a Supply Chain: The Bullwhip Effect." *Management Science* 43(4), 1997.
- Batini, C., Cappiello, C., Francalanci, C., Maurino, A. "Methodologies for Data Quality Assessment and Improvement." *ACM Computing Surveys* 41(3), 2009.
- Hazen, B., Boone, C., Ezell, J., Jones-Farmer, L. "Data Quality for Data Science, Predictive Analytics, and Big Data." *International Journal of Production Economics* 154, 2014.

**Industry and datasets**
- M5 (Walmart) dataset and methods, Kaggle and https://github.com/Mcompetitions/M5-methods (accessed 27 October 2026).
- Corporación Favorita Grocery Sales Forecasting (Kaggle, 2018); Rossmann Store Sales (Kaggle, 2015); UCI Online Retail II (2009–2011); DataCo Smart Supply Chain (Mendeley). [VERIFY DataCo provenance and counts before citing as evidence.]
- GS1 EPCIS 2.0 specification and implementation guidance, GS1, 2022 onward.

**Context**
- Course source pack: `_research/industry_brief.md` (2026 adoption, regulation, and trade context).

---

> **Next:** [Module 03](03_data_engineering_interoperability.md) — turning this landscape into reproducible pipelines: layered architecture, SQL patterns, EDI parsing, data contracts, quality gates, and a demand-and-inventory mart.
