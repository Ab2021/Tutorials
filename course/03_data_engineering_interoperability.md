# Module 03 — Data Engineering and Interoperability

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 32 hours (12 lecture, 14 lab, 6 project)
> **Track fit:** Common core (all three tracks)
> **Prerequisites:** Module 01, 01A, and 02
> **Tools:** Python 3.12, DuckDB 2.x, dbt, Git, an X12 or EDIFACT sample file
> **Industry question:** "Can I rebuild last Tuesday's inventory position exactly, and can I prove where every number in it came from?"

---

## 1. Why industry cares

A planning decision is only as good as the pipeline that feeds it. Forecasts, replenishment recommendations, and control-tower alerts are all downstream of joins between systems that were never designed to agree: the enterprise resource planning (ERP) system's item master, the warehouse's snapshot, the carrier's event feed, the retailer's EDI file, and the supplier's portal. When those joins are wrong, the model is not "slightly off". It is confidently answering a different question.

The economics are large because the errors are systematic, not random. A one-day misalignment between shipment arrival and inventory snapshot understates in-transit stock for every lane at once. A case-pack change applied from the wrong effective date misstates demand in cases for every order after the change. A duplicate ship-to record splits a customer's history and makes a stable account look volatile. None of these show up as an exception; they show up as a plausible forecast, a plausible plan, and a real stockout.

| Symptom in the plan | Engineering cause | Typical patch | Correct fix |
|---|---|---|---|
| Inventory is 8% below the warehouse count every month | Snapshot taken before put-away posting | Monthly manual adjustment | Snapshot at a defined cut-off, after posting completes |
| Demand per store is smooth but total is right | Sales allocated pro rata, not observed | Ignore store variation | Capture point-of-sale (POS) data at store-item-day or mark allocation explicitly |
| Lead time is 14 days in the plan, 21 days in reality | Goods-receipt date used instead of gate-in | Add a buffer | Model lead time from order release to receipt, with components |
| Promo demand spikes then crashes | Orders, not consumption, are recorded | Trim outliers | Store shipments and consumption separately; keep both |
| The same supplier appears three times | No canonical supplier identifier | Manual merge in the report | Entity resolution in MDM with survivorship rules |

Scale makes the engineering non-optional. A mid-size retailer generating 10 million order lines per day at roughly 500 bytes per raw record produces about 5 GB per day, 1.8 TB per year before compression; at 5:1 columnar compression that is still about 365 GB per year of retained raw history (10,000,000 lines/day × 500 bytes = 5,000,000,000 bytes/day = 5 GB/day; × 365 = 1,825 GB/year ≈ 1.8 TB/year; ÷ 5 = 365 GB/year). Analysts cannot reconcile that by hand, so the pipeline must carry the semantics: units, effective dates, keys, and quality gates.

> **Industry interpretation.** Interoperability is not "we have an API". It is the ability of two organisations' systems to exchange a message and have both sides mean the same thing by every field, at the same grain, in the same units, on the same clock.

## 2. Learning outcomes

By the end of this module you can:

1. Design a layered (bronze, silver, gold) supply chain data platform and justify each layer's grain, key, and retention rule.
2. Choose between warehouse, lakehouse, and federated query patterns for a given latency, governance, and cost profile.
3. Implement slowly changing dimensions (SCD), as-of joins, date spines, deduplication, gaps-and-islands, and temporal interval joins in SQL.
4. Structure a Python analytics project with typed records, validated configuration, and tests that fail loudly on unit errors.
5. Write dbt models with contracts, tests, and documentation, and schedule them idempotently with an orchestrator.
6. Read an X12 or EDIFACT interchange at segment level and map it to canonical order, shipment, and invoice tables.
7. Apply GS1 identifiers and EPCIS 2.0 event semantics to traceability problems.
8. Specify an application programming interface (API) and a data contract that make freshness, schema, and quality expectations enforceable.
9. Run master data management (MDM) entity resolution and operate a quarantine pattern for bad records.

## 3. Mental model

Think of the platform as **three layers plus two boundaries**. The layers refine data; the boundaries are where meaning is negotiated.

```
 Sources                Bronze                Silver                 Gold
 ───────                ──────                ──────                 ────
 ERP orders      ──►  raw, immutable   ──►  conformed, keyed  ──►  facts and dims
 WMS snapshots         append-only           deduplicated           business grain
 TMS events            source schema         typed, unit-normalised tested, documented
 EDI 850/856/810       partitioned by date   quality-flagged        served to models
 POS / IoT / APIs      replayable            quarantined rejects    access-controlled
        │                     │                     │                     │
        └── ingestion ────────┴── transformation ───┴── publication ──────┘
             contract              contract              contract
```

Layer grains and latency modes are specified in §4.1 and §4.3. The model's one invariant is that refinement is monotonic: no gold table may contain a fact that silver cannot trace, and no silver row may exist without a bronze record.

**Boundary one: the source contract.** What the source promises about schema, keys, timing, and meaning. **Boundary two: the consumption contract.** What the model or dashboard is allowed to assume about freshness, completeness, and units. Most failures live at a boundary that nobody wrote down.

> **Rule of thumb.** If a transformation cannot be re-run for an arbitrary past date and produce the same result, it is not a pipeline; it is an anecdote.

## 4. Core concepts

### 4.1 Layered architecture and the medallion pattern

The bronze, silver, gold convention separates **fidelity** from **usability**. Bronze preserves the source exactly, including its defects, so a later dispute can be settled by replay. Silver applies typing, unit conversion, deduplication, and key resolution once, centrally. Gold shapes data to the decision grain and carries the tests stakeholders rely on.

The discipline is **immutable bronze and deterministic silver**. Bronze is append-only and partitioned by ingestion date; a corrected source file creates a new partition rather than rewriting the old one. Silver models are pure functions of declared bronze partitions, and gold models are pure functions of silver. When a number looks wrong, walk backwards until a layer stops reproducing.

```
s3://sc-raw/orders/source=erp/ingest_date=2026-10-27/part-0001.parquet
s3://sc-raw/edi856/partner=retailer_a/ingest_date=2026-10-27/856_000123.edi
```

Partition on **when you learned the fact**, not only when it happened. Both are useful; conflating them destroys replayability.

| Layer | Grain | Key discipline | Retention | Who queries it |
|---|---|---|---|---|
| Bronze | One row per source record, as received | Source key plus ingest timestamp; never overwrite | 13–25 months | Data engineers |
| Silver | One row per event or entity version | Canonical keys, eaches and base currency | 25–37 months | Analysts, scientists |
| Gold | Decision grain: item-location-week, lane-day, supplier-month | Conformed dimensions, additive measures, tests | 3–7 years | Planners, models, BI |

### 4.2 Warehouse, lakehouse, and federated patterns

| Pattern | Storage and compute | Best fit | Main risk |
|---|---|---|---|
| Warehouse | Managed columnar tables, elastic SQL | Governed gold marts, BI, finance reporting | Raw variety and schema rigidity |
| Lakehouse | Object storage plus table format (Delta, Iceberg, Hudi), decoupled engines | Large raw history, mixed SQL and Python, features | Small files, compaction, catalog sprawl |
| Federated | Data stays at source, query federation | Supplier or 3PL data that cannot be copied | Latency, partial failure, inconsistent semantics |

The choice is rarely global. A common pattern lands raw EDI and telemetry in object storage, conforms it in a lakehouse table format, and publishes gold marts to a warehouse. Federated queries are reserved for data whose residency or commercial sensitivity forbids copying, and must expose the same canonical schema or analysts will fork logic.

### 4.3 Batch, micro-batch, and streaming

Batch is the default because planning cycles are batch: a weekly replenishment run does not need second-level latency, and batch is cheaper and easier to reconcile. Streaming earns its cost only where the decision is time-critical: a cold-chain excursion that must quarantine stock within minutes, a yard move that changes dock availability, an order promise that must see inventory now.

| Mode | Latency | Typical use | Cost |
|---|---|---|---|
| Batch, daily | 2–31 hours | Demand history, reconciliation, replenishment | Low |
| Streaming, seconds | 1–60 seconds | Yard, cold chain, control tower | High |
| Federated query | Query time | Partner-owned data that cannot be copied | Medium |

Streaming adds three supply chain problems. **Event time versus processing time**: a carrier event may arrive 40 minutes late and belongs at the time it happened. **Out-of-order arrival**: "delivered" may precede "out for delivery", so the state machine must be monotonic or explicitly reconciled. **Exactly-once semantics**: duplicates are normal in EDI and telematics, so consumers must be idempotent on a natural key such as `(shipment_id, event_type, event_ts)`.

### 4.4 Ingestion patterns

| Pattern | Direction | Strengths | Weaknesses | Use when |
|---|---|---|---|---|
| Flat file / secure file transfer | Pull | Universal, auditable, cheap | Late, partial, encoding issues | Legacy ERP, 3PL, EDI |
| REST API | Pull | Typed, paginated, incremental | Rate limits, schema drift | SaaS transport, order, and product systems |
| Webhook | Push | Low latency | Delivery guarantees vary | Status changes, order events |
| Change data capture (CDC) | Push/stream | Captures deletes and updates | Log access, ordering | Databases you own |

Every ingestion job needs a **watermark**, an **idempotency key**, a **schema check**, a **raw landing path**, and a **failure alarm**. Without a watermark, reruns duplicate. Without idempotency, retries corrupt. Without a schema check, a renamed column silently becomes null.

### 4.5 SQL patterns that carry supply chain semantics

The following runs on DuckDB 2.x as written.

```sql
-- SCD2 item master with a mid-year case-pack change, joined as-of the order date.
CREATE TABLE item_master (item_id VARCHAR, valid_from DATE, valid_to DATE, case_pack INT);
CREATE TABLE orders (order_id VARCHAR, item_id VARCHAR, order_date DATE, qty_eaches INT);
INSERT INTO item_master VALUES
  ('SKU1', DATE '2026-01-01', DATE '2026-06-30', 24),
  ('SKU1', DATE '2026-07-01', DATE '9999-12-31', 12);
INSERT INTO orders VALUES
  ('SO1', 'SKU1', DATE '2026-03-02', 240),
  ('SO2', 'SKU1', DATE '2026-08-03', 240);
SELECT o.order_id, o.order_date, m.case_pack,
       o.qty_eaches // m.case_pack AS full_cases
FROM orders o
ASOF LEFT JOIN item_master m
  ON o.item_id = m.item_id AND o.order_date >= m.valid_from
ORDER BY o.order_id;   -- SO1: 24, 10 cases; SO2: 12, 20 cases.
```

```sql
-- Gaps and islands: contiguous stockout runs in daily snapshots.
CREATE TABLE snapshots (snap_date DATE, item_id VARCHAR, location_id VARCHAR, on_hand INT);
INSERT INTO snapshots VALUES
 ('2026-01-01','SKU1','ST01',50), ('2026-01-02','SKU1','ST01',40),
 ('2026-01-03','SKU1','ST01',0),  ('2026-01-04','SKU1','ST01',0),
 ('2026-01-05','SKU1','ST01',0),  ('2026-01-06','SKU1','ST01',25),
 ('2026-01-07','SKU1','ST01',20);
WITH flagged AS (
  SELECT snap_date, item_id, location_id,
         CASE WHEN on_hand = 0 THEN 1 ELSE 0 END AS is_zero,
         snap_date - (row_number() OVER (
             PARTITION BY item_id, location_id,
                          CASE WHEN on_hand = 0 THEN 1 ELSE 0 END
             ORDER BY snap_date) * INTERVAL 1 DAY) AS grp
  FROM snapshots
)
SELECT item_id, location_id, min(snap_date) AS gap_start,
       max(snap_date) AS gap_end, count(*) AS stockout_days
FROM flagged WHERE is_zero = 1
GROUP BY item_id, location_id, grp;   -- 3–5 Jan 2026, 3 days.
```

```sql
-- Date spine across a 53-week year boundary.
SELECT d::DATE AS cal_date, date_trunc('week', d::DATE)::DATE AS week_start,
       date_part('week', d::DATE) AS iso_week
FROM generate_series(DATE '2026-12-28', DATE '2027-01-03', INTERVAL 1 DAY) AS t(d);
-- 28 Dec 2026 to 3 Jan 2027 all report ISO week 53.
-- In-transit inventory as a half-open temporal interval join.
CREATE TABLE shipments (shipment_id VARCHAR, ship_date DATE, arrival_date DATE, qty INT);
INSERT INTO shipments VALUES
 ('SH1', DATE '2026-01-01', DATE '2026-01-04', 500),
 ('SH2', DATE '2026-01-03', DATE '2026-01-06', 300);
SELECT d.cal_date, sum(s.qty) AS in_transit_units
FROM (SELECT d::DATE AS cal_date
      FROM generate_series(DATE '2026-01-01', DATE '2026-01-05', INTERVAL 1 DAY) AS t(d)) d
JOIN shipments s ON d.cal_date >= s.ship_date AND d.cal_date < s.arrival_date
GROUP BY d.cal_date ORDER BY d.cal_date;   -- 500; 500; 800; 300; 300 units.
```

Three rules follow. **SCD joins must be as-of, never equality on a mutable attribute.** **Snapshot and transaction facts must not be summed together**; a snapshot is a state, a transaction a flow. **Intervals are half-open**: a shipment arriving on 4 January is on hand that day and not in transit.

Deduplicate with a deterministic ranking so the surviving row is explainable:

```sql
CREATE TABLE raw_order_lines (order_id VARCHAR, line_number INT,
                              source_updated_at TIMESTAMP, ingest_ts TIMESTAMP);
SELECT * EXCLUDE (rn) FROM (
  SELECT *, row_number() OVER (
      PARTITION BY order_id, line_number
      ORDER BY source_updated_at DESC, ingest_ts DESC) AS rn
  FROM raw_order_lines
) WHERE rn = 1;
```

### 4.6 Python project structure, typed data, and tests

A pipeline is a small software product. Structure it so a new engineer finds the unit conversion, the configuration, and the test in under a minute.

```
demand_mart/{pyproject.toml, conf/base.yaml, src/demand_mart/*.py, tests/*.py, dbt_project/}
```

```python
# src/demand_mart/models.py
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True, slots=True)
class OrderLine:
    order_id: str
    item_id: str
    location_id: str
    order_date: date
    qty_eaches: int
```

```python
# src/demand_mart/units.py — one conversion module, no duplicates
from math import ceil

def to_eaches(cases: int, case_pack: int) -> int:
    if case_pack <= 0:
        raise ValueError("case_pack must be positive")
    return cases * case_pack

def cases_to_order(eaches: int, case_pack: int) -> int:
    if case_pack <= 0:
        raise ValueError("case_pack must be positive")
    return ceil(eaches / case_pack)
```

```python
# tests/test_units.py
import pytest
from demand_mart.units import cases_to_order, to_eaches

def test_conversions():
    assert to_eaches(3, 24) == 72
    assert cases_to_order(100, 24) == 5

def test_rejects_zero_case_pack():
    with pytest.raises(ValueError):
        to_eaches(1, 0)
```

Configuration is code, not a spreadsheet: load YAML into a frozen dataclass whose `__post_init__` rejects a fiscal month outside 1–12, so a bad path or month fails before the job runs.

### 4.7 dbt models, tests, and documentation

dbt turns SQL transformations into a dependency graph with tests. The value is not templating; it is that every model declares its grain, upstream dependencies, and quality expectations in version control.

```sql
-- models/marts/fct_demand_weekly.sql
{{ config(materialized='table', cluster_by=['week_start']) }}

select
    date_trunc('week', order_date)::date as week_start,
    item_id,
    location_id,
    sum(qty_eaches) as demand_eaches,
    count(distinct order_id) as order_count
from {{ ref('stg_orders') }}
group by 1, 2, 3
```

```yaml
# models/marts/schema.yml
version: 2
models:
  - name: fct_demand_weekly
    description: Weekly demand in eaches; one row per item-location-week.
    columns:
      - {name: week_start, tests: [not_null]}
      - name: item_id
        tests: [not_null, {relationships: {to: ref('dim_item'), field: item_id}}]
      - {name: demand_eaches, tests: [not_null]}
```

Use **model contracts** for gold tables other teams consume, so a column rename fails the build rather than a dashboard. Use **source freshness** checks to alarm on late feeds, and **exposures** to record which dashboards depend on which tables.

### 4.8 Orchestration

An orchestrator (Airflow, Dagster, Prefect, or managed equivalent) owns scheduling, retries, dependencies, and backfills. Three properties separate a working schedule from a fragile one.

**Idempotent partitions.** Each run replaces its date partition atomically; re-running 12 October must not duplicate 12 October.

**Explicit data dependencies.** Wait for inputs, not for a clock. "Run at 02:00" is a guess; "run after `stg_shipments` for the same logical date succeeds" is a contract.

**Backfill as a first-class operation.** When logic changes, the team must replay 24 months consistently; a manual script will not do.

| Failure | Detection | Recovery |
|---|---|---|
| Late source file | Freshness service-level agreement (SLA) breach | Hold downstream, alert owner, never publish stale gold |
| Schema drift | Column presence and type test | Quarantine batch, map or reject explicitly |

### 4.9 EDI: X12 and EDIFACT

EDI remains the backbone of retail and manufacturing exchange. An 856 that fails to parse becomes a receiving discrepancy, then a deduction, then a margin leak.

| X12 | Purpose | EDIFACT | Key segments | Canonical target |
|---|---|---|---|---|
| 850 | Purchase order | ORDERS | `BEG`, `PO1`, `CTT` | `order_header`, `order_line` |
| 855 | Order acknowledgement | ORDRSP | `BAK`, `PO1`, `ACK` | `order_ack` |
| 856 | Advance ship notice | DESADV | `BSN`, `HL`, `LIN`, `SN1` | `shipment`, `shipment_line` |
| 810 | Invoice | INVOIC | `BIG`, `IT1`, `TDS` | `invoice_header`, `invoice_line` |
| 846 | Inventory advice | INVRPT | `BIA`, `LIN`, `QTY` | `inventory_snapshot` |
| 852 | Product activity | SLSRPT | `XQ`, `LIN`, `ZA`, `SDQ` | `pos_activity` |
| 940 | Warehouse shipping order | — | `W05`, `W01` | `warehouse_order` |
| 945 | Warehouse shipping advice | — | `W06`, `W12` | `warehouse_shipment` |
| 997 | Functional acknowledgement | CONTRL | `AK1`, `AK2`, `AK5`, `AK9` | `interchange_ack` |

An X12 850 purchase order, reduced to essentials:

```
ISA*00*          *00*          *ZZ*BUYER          *ZZ*SUPPLIER       *261015*1200*U*00501*000000001*0*P*>~
GS*PO*BUYER*SUPPLIER*20261015*1200*1*X*005010~
ST*850*0001~
BEG*00*SA*PO12345**20261015~
N1*ST*DC 101*92*DC101~
PO1*1*120*EA*12.50**UP*012345678905~
CTT*1~
SE*6*0001~
GE*1*1~
IEA*1*000000001~
```

`PO1` reads: line 1, 120 units, unit of measure EA (each), unit price 12.50 USD, identifier qualifier UP (UPC), identifier 012345678905. At a case pack of 24, 120 eaches is 5 cases; the line is stored as 120 EA with a conversion record, never as 5 cases alone.

A 855 acknowledgement returns `ST*855*0002~ BAK*00*AD*PO12345*20261015~ PO1*1*120*EA*12.50**UP*012345678905~ ACK*IA*120*EA~ CTT*1~ SE*6*0002~`.

The 856 carries the physical shipment and its hierarchy:

```
ST*856*0001~
BSN*00*SHIP123*20261016*0800*0001~
HL*1**S~
HL*2*1*O~
PRF*PO12345~
HL*3*2*I~
LIN*1*UP*012345678905~
SN1**120*EA~
CTT*3~
SE*10*0001~
```

`HL` defines shipment (S), order (O), item (I); `CTT` counts the three hierarchical loops. The invoice totals 1,500.00 USD, expressed in `TDS` as 150000 because X12 monetary fields are in cents (120 × 12.50 = 1,500.00):

```
ST*810*0001~
BIG*20261020*INV9001*20261015*PO12345~
IT1*1*120*EA*12.50**UP*012345678905~
TDS*150000~
CTT*1~
SE*6*0001~
```

Other sets use these qualifiers: `QTY*33` is quantity available in the 846; `ZA*F` and `SDQ*EA*33*480` report 480 EA on hand in the 852; `W05`/`W01` place a warehouse order in the 940; `W06`/`W12` report the warehouse shipment in the 945; `AK5*A*A` accepts a functional group in the 997.

EDIFACT uses `'` as segment terminator and `+` as element separator. An ORDERS message:

```
UNB+UNOC:3+BUYER:14+SUPPLIER:14+261015:1200+1'
UNH+1+ORDERS:D:96A:UN'
BGM+220+PO12345+9'
DTM+137:20261015:102'
NAD+BY+BUYER::9'
NAD+SU+SUPPLIER::9'
LIN+1++012345678905:EN'
QTY+21:120:PCE'
PRI+AAA:12.50'
UNS+S'
MOA+86:1500.00'
UNT+11+1'
UNZ+1+1'
```

`UNT` counts the segments in the message including `UNH` and `UNT`. A DESADV with 120 PCE packed as 5 cases, and an INVOIC with 120 PCE at 1,500.00:

```
UNH+2+DESADV:D:96A:UN'
BGM+351+SHIP123+9'
DTM+137:20261016:102'
NAD+SH+SUPPLIER::9'
CPS+1'
PAC+5+CT::9'
LIN+1++012345678905:EN'
QTY+12:120:PCE'
UNT+9+2'

UNH+3+INVOIC:D:96A:UN'
BGM+380+INV9001+9'
NAD+BY+BUYER::9'
NAD+SU+SUPPLIER::9'
LIN+1++012345678905:EN'
QTY+47:120:PCE'
MOA+203:1500.00'
UNS+S'
MOA+86:1500.00'
UNT+10+3'
```

Parsing rules that prevent most defects: read delimiters from `ISA`; honour the release character; validate `SE`/`UNT` counts and reject on mismatch; retain interchange control numbers; and never treat a 997 as business acceptance. A 997 says the syntax parsed; an 855 says the order was accepted.

### 4.10 GS1 identifiers and EPCIS 2.0

| Identifier | Structure | Identifies | Typical use |
|---|---|---|---|
| GTIN | 8, 12, 13, or 14 digits, last is check digit | Trade item type | Item master, barcode, EDI `UP` |
| GLN | 13 digits, last is check digit | Legal entity or location | Ship-to, bill-to, plant |
| SSCC | 18 digits: extension, company prefix, serial reference, check digit | Logistic unit | Pallet or case label |

GTIN `012345678905` is valid: weighting the eleven-digit body `01234567890` alternately 3 and 1 from the right gives 0+9+24+7+18+5+12+3+6+1+0 = 85, so the check digit is (10 − 85 mod 10) mod 10 = 5. Store identifiers as text: leading zeros and fourteen-digit forms are meaningful.

EPCIS 2.0 is the GS1 standard for event-based traceability, with five event types: `ObjectEvent`, `AggregationEvent`, `TransactionEvent`, `TransformationEvent`, and `AssociationEvent`. An ObjectEvent for a shipment:

```json
{
  "eventType": "ObjectEvent",
  "eventTime": "2026-10-16T08:00:00Z",
  "action": "OBSERVE",
  "bizStep": "shipping",
  "disposition": "in_transit",
  "epcList": ["urn:epc:id:sgtin:0614141.107346.2026"],
  "bizLocation": {"id": "urn:epc:id:sgln:0614141.00001.0"}
}
```

The value is that **aggregation and transformation are explicit**. A pallet is not a quantity in a big box; it is an aggregation event linking an SSCC to child SSCCs or GTINs, with time and location. That is what makes a recall query answerable: which pallets held this lot, where did they go, and what was on them.

### 4.11 APIs and data contracts

An API should expose **resources, not reports**: `/orders`, `/shipments`, `/inventory-positions`, `/items`. Four properties matter. **Stable identifiers and idempotency**: `POST /orders` accepts an `Idempotency-Key`, and a retry returns the original result. **Cursor pagination with a watermark**: `?updated_since=...&cursor=...` resumes without gaps or duplicates, unlike offset pagination. **Explicit units and time zones**: every quantity names its unit, every timestamp is ISO 8601 with an offset. **Versioned schema with a deprecation policy**: additive changes are minor; renames and type changes are major with a migration window.

```yaml
# contracts/orders.v1.yaml (illustrative)
id: sc.orders
version: 1.3.0
owner: supply-chain-data@example.com
schema:
  - {name: order_id, type: string, required: true, unique: true}
  - {name: qty_eaches, type: integer, required: true, constraints: {minimum: 0}}
  - {name: order_ts, type: timestamp, required: true, timezone: UTC}
quality:
  - {rule: null_rate, column: item_id, max: 0.001}
  - {rule: row_count_between, min: 1000, max: 50000000}
sla: {freshness_minutes: 60, availability: "99.5%"}
```

The contract is tested in continuous integration and monitored in production. A breach is an incident, not a warning in a log file.

### 4.12 MDM and entity resolution

Master data management creates one trusted record per item, location, supplier, customer, and carrier. Entity resolution is the matching step. A typical programme starts with 12,000 ship-to records, resolves them to 9,400 canonical locations, and retires 2,600 duplicates — 21.7% of the original set (2,600 ÷ 12,000 = 0.2167). Those duplicates had been splitting demand histories and inflating apparent volatility.

Matching follows a hierarchy of evidence: exact identifier (GLN, or DUNS where licensed), then normalized name and address, then fuzzy similarity with human review. Survivorship rules decide which attribute wins, and every merge must be reversible through a cross-reference table from source key to canonical key.

### 4.13 Quality tests and quarantine

| Test | Example rule | Action on failure |
|---|---|---|
| Not null | `item_id` on every order line | Quarantine row |
| Accepted values | `uom` in {EA, CS, PL, KG, L} | Quarantine row, alert source |
| Uniqueness | `(order_id, line_number)` unique | Deduplicate, log count |
| Referential integrity | `item_id` exists in `dim_item` | Route to unresolved queue |
| Freshness | Latest snapshot within 26 hours | Hold publication, alert |
| Control total | Invoice total equals sum of lines | Reject file |
| Distribution shift | Daily orders within 40% of 28-day median | Warn, then investigate |

Quarantine is a **table, not a log line**: each rejected record carries source file, ingest timestamp, rule violated, and raw payload. Target a rate below 0.5%: at 2.4 million rows per day, 0.4% is 9,600 rows, reviewable daily (2,400,000 × 0.004 = 9,600). At 5% it is 120,000 rows and the queue is ignored, which is worse than no queue.

### 4.14 Versioning and lineage

Three versions must be distinguishable: the **data version** (which extract), the **code version** (which commit), and the **semantic version** (which definition of the metric). A dashboard reporting "fill rate 94.2%" is meaningless unless all three are pinned. Lineage answers "what produced this number" at column level where possible: source table, transformation, target column, code commit. Minimum viable lineage is a run log with input partitions, output partitions, code commit, row counts, and quality results, queryable by run ID.

### 4.15 Security and incident response

Supply chain data is commercially sensitive: pricing, volumes, supplier terms, customer identities, and sometimes personal data. Controls are least-privilege access with row-level security on partner and region, encryption in transit and at rest, secrets in a managed vault, egress restrictions, and audit logs on every export.

Incident response follows six steps: **detect, triage, contain, communicate, recover, review**. For data incidents, containment often means stopping publication rather than stopping the source: publish a labelled "last known good" and halt downstream jobs. The review must produce a reproducible timeline, the blast radius in records and decisions, and a test that would have caught it.

## 5. Data, standards, and workflow

| Standard or artifact | What it governs | Where it appears |
|---|---|---|
| ANSI ASC X12 | US EDI transaction sets and envelopes | 850, 855, 856, 810, 846, 852, 940, 945, 997 |
| UN/EDIFACT | International EDI messages | ORDERS, ORDRSP, DESADV, INVOIC, INVRPT, SLSRPT, CONTRL |
| GS1 General Specifications | GTIN, GLN, SSCC, barcode symbology | Product and location master, labels |
| EPCIS 2.0 | Event-based traceability data model | Serialization, aggregation, chain of custody |
| ISO 8601 | Date, time, duration, week numbering | All timestamps and planning buckets |
| ISO 4217 | Currency codes | Pricing, landed cost, FX |
| UN/CEFACT units | Unit of measure codes | Quantity fields across systems |

A workable delivery workflow, in order: (1) map the source system and its owner; (2) land raw data immutably; (3) profile nulls, keys, and units; (4) write the canonical schema and data contract; (5) build silver transforms with tests; (6) build gold marts at decision grain; (7) add freshness and volume monitors; (8) document lineage and ownership; (9) rehearse a backfill; (10) hand over with a runbook.

## 6. Industry case studies

**Target Canada, 2013–2015.** Target opened its first Canadian stores in March 2013 and announced on 15 January 2015 that it would wind down the Canadian business. Public post-mortems by Target's own management and contemporary reporting described inventory records that did not match physical shelves, pricing and item-master discrepancies, and a supply chain that could not replenish stores reliably. The data-engineering lesson is that **a new market is a master-data problem before it is a merchandising problem**: item, location, and price masters must be reconciled and validated before volume goes live, and a single canonical item identifier must survive translation between the US and Canadian systems.

**US DSCSA enhanced drug distribution security, 27 November 2023.** The Drug Supply Chain Security Act required interoperable, electronic, transaction-level tracing of prescription drugs, with enhanced requirements taking effect on 27 November 2023 and an FDA stabilization period that followed. [VERIFY the current stabilization-period end date and any exemptions before citing in a compliance context.] The lesson is that **serialization is a data-model requirement, not a label requirement**: product identifiers, lot, expiry, and transaction information must be exchanged in a standard, queryable form across manufacturers, wholesalers, and dispensers, and the analytics layer must be able to reconstruct chain of custody per package.

**Maersk and NotPetya, June 2017.** A malware attack on 27 June 2017 disabled a large share of Maersk's IT estate. Maersk publicly described re-installing roughly 4,000 servers, 45,000 PCs, and 2,500 applications, and reported a financial impact in the range of USD 200–300 million. The lesson is **recovery depends on data portability and offline runbooks**. Backups that require the domain controller, or a pipeline whose only copy of the master data sits on encrypted shared storage, turn a security incident into an operations incident. Test restoration, not just backup.

**GS1 Sunrise 2027.** GS1's Sunrise 2027 initiative aims for retail point-of-sale systems to accept 2D barcodes carrying a GTIN plus lot, date, and serial data by the end of 2027. [VERIFY partner-specific readiness and exact scope; this is an industry transition target, not a regulation.] The lesson is **a data-carrier change is a master-data and event-model change**. Moving from UPC-A to a 2D code adds batch and expiry to every scan, which is useful only if the item master, the receiving process, and the traceability store can hold and query those attributes.

## 7. Labs

### Lab 3.1 — Build bronze, silver, and gold in DuckDB
**Objective:** Turn three raw extracts into a tested, queryable layered model.
**Data:** `orders.csv`, `shipments.csv`, and `snapshots.csv`, or the synthetic tables from §4.5.
**Steps:** (1) Load each file into a `bronze_*` table with an `ingest_ts` column. (2) Profile row counts, null rates, distinct keys, and unit values. (3) Build `silver_orders` with deduplication and typed columns; build `silver_shipments` with `ship_date` and `arrival_date`. (4) Build `gold_inventory_position` at item-location-day with on-hand, in-transit, and available-to-promise. (5) Write one test per table and record the quarantine count.
**Deliverable:** A `.sql` file that runs from an empty database, plus a profile report.
**Acceptance criteria:** Two runs produce identical row counts; no null `item_id` in silver; daily in-transit equals the sum of open shipments; every test passes or is waived in writing.
**Stretch goal:** Add a `run_id` and a `quality_result` table so each run's test outcomes are queryable.

### Lab 3.2 — SCD2 item master and as-of conversion
**Objective:** Measure the cost of applying a case-pack change from the wrong date.
**Data:** An item master with two versions and 12 months of orders.
**Steps:** (1) Implement the SCD2 table with `valid_from`/`valid_to`. (2) Convert orders to cases with an as-of join. (3) Repeat using the current case pack for all dates. (4) Compare case quantities by month and compute the difference in cases and in value at a stated unit cost.
**Deliverable:** A monthly cases table under both methods, with the variance.
**Acceptance criteria:** The as-of join returns exactly one master row per order; the comparison reconciles to total eaches; the unit-cost assumption is stated.
**Stretch goal:** Add a third version with a mid-month change and define the convention for orders on the change date.

### Lab 3.3 — Parse an X12 850 and 856 into canonical tables
**Objective:** Convert two EDI interchanges into rows a planner can query.
**Data:** The §4.9 850 and 856 samples, or partner samples with identifiers removed.
**Steps:** (1) Split the file into segments using the delimiters declared in `ISA`. (2) Extract header, party, and line fields into `order_header` and `order_line`. (3) Extract the 856 hierarchy into `shipment` and `shipment_line`, preserving the HL parent-child relationships. (4) Validate `SE` counts and reconcile ordered versus shipped quantity. (5) Store the raw segment text alongside the parsed row.
**Deliverable:** Parser code, canonical tables, and a reconciliation report.
**Acceptance criteria:** 120 EA ordered reconciles to 120 EA shipped; a file with an incorrect `SE` count is rejected; each line retains source file and segment position; difference is zero.
**Stretch goal:** Emit a 997 acknowledgement with correct `AK1`/`AK2`/`AK5`/`AK9` segments.

### Lab 3.4 — Demand-and-inventory mart with tests and a schedule
**Objective:** Deliver the module's final artifact: a mart a forecasting team can consume.
**Data:** Raw orders, shipments, and at least 90 days of daily inventory snapshots.
**Steps:** (1) Build a date spine covering the full history, including a 53-week boundary. (2) Aggregate demand to item-location-week and compute fill rate and weeks of cover. (3) Join in-transit inventory at week end. (4) Add dbt tests for uniqueness, not-null, non-negative demand, and referential integrity to the item dimension. (5) Add a freshness check and an orchestration directed acyclic graph (DAG) with a documented backfill command. (6) Write the data contract for the mart.
**Deliverable:** A repository with models, tests, contract, DAG, and a README stating grain, keys, units, and limitations.
**Acceptance criteria:** A 90-day backfill reproduces the mart; all tests pass; the contract names the owner and freshness SLA; the README separates forecast, plan, and actual.
**Stretch goal:** Add a quarantine table for orders that fail the item-master join and a weekly report of unresolved records.

## 8. Assignment

Design and build a **reproducible supply chain data product** for a scenario of your choice (retail replenishment, pharma traceability, or manufacturing component supply). The submission has four parts.

1. **Architecture note (1,500 words):** source map, layer design, grain and keys per table, batch versus streaming decision with justification, and the failure modes you designed against.
2. **Working pipeline:** bronze, silver, and at least two gold models, with tests, a data contract, and an orchestration definition. It must run from a clean checkout with one documented command.
3. **Quality report:** row counts, null rates, duplicate rates, quarantine counts, and test results for a 90-day backfill.
4. **Incident rehearsal:** inject one realistic defect (duplicate deliveries, a case-pack change, or a late feed), show how the pipeline detected it, and write a one-page post-mortem with timeline and corrective test.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---|---|---|---|
| Layered design and grain | 20% | Layers exist but grains are mixed | Layers and grains stated, keys defined | Grain is provably one row per decision unit; retention and replay justified |
| SQL correctness | 20% | Joins duplicate or drop rows | SCD, dedup, and interval logic correct | As-of and temporal logic tested against edge cases, including year boundaries |
| Units and semantics | 15% | Quantities unlabelled or converted inconsistently | Units named and converted centrally | Conversion is single-sourced, tested, and reconciled to source totals |
| Testing and quality gates | 15% | No tests | Core tests present | Tests cover contracts, freshness, control totals, and distribution shift |
| Interoperability | 10% | EDI or API consumed as opaque blobs | Fields mapped to canonical schema | Envelope validation, acknowledgements, and reconciliation implemented |
| Operability | 10% | Manual steps undocumented | DAG and backfill documented | Idempotent backfill rehearsed; runbook and lineage complete |
| Security and governance | 5% | Access uncontrolled | Roles and secrets managed | Row-level controls, audit trail, and incident plan demonstrated |
| Communication | 5% | Undocumented assumptions | Clear README and contract | Limitations and decision impact stated for a non-technical reader |

## 10. Regulatory, ethical, and commercial lens

**Traceability and product safety.** Pharmaceutical serialization (DSCSA in the US), food traceability rules, and medical-device unique device identification all require lot- or serial-level records that can be produced on demand. A pipeline that aggregates away lot identity cannot answer a recall query, and retrofitting lot capture is expensive.

**Personal and commercial data.** Delivery addresses, driver identities, and customer-level sales can be personal data under regimes such as the General Data Protection Regulation (GDPR) and comparable laws, and partner contracts often restrict re-use of retail POS data. The engineering consequence is purpose limitation: separate the identifiable layer from the analytics layer, minimize, and document lawful basis and retention. **Competition and confidentiality.** Supplier pricing and retailer sell-through are competitively sensitive. Sharing them across partners, or pooling them into a joint model without contractual cover, can create legal exposure; data contracts should state permitted use, not only schema. **Financial reporting.** Inventory valuation, revenue recognition, and cost of goods sold depend on the same data, so a pipeline that silently reclassifies freight or excludes in-transit stock can misstate financials; controls, reconciliation to the general ledger, and change management are part of the data engineer's remit.

**AI-specific obligations.** Where forecasts or automated decisions affect individuals or regulated products, emerging AI rules may require documentation, human oversight, and logging. The pipeline is where that evidence is generated: input versions, model versions, and decision logs.

## 11. Failure modes and red flags

| Failure mode | Early signal | Consequence | Prevention |
|---|---|---|---|
| Snapshot and transaction facts summed together | Inventory grows monotonically | Overstated stock, under-ordering | Separate fact types; document grain |
| SCD joined on current attributes | History changes when master data changes | Backtests look better than reality | As-of joins with validity intervals |
| Case-pack conversion applied from today's master | Demand in cases jumps on change dates | Wrong replenishment quantities | Effective-dated conversions, unit tests |
| Time zones mixed | Daily totals shift by a day | Phantom demand spikes at month end | Store UTC, convert at presentation |
| Duplicate EDI delivery | Order counts exceed PO counts | Double shipments, invoice disputes | Idempotency keys, natural-key uniqueness |
| Silent schema drift | A column becomes null | Metrics degrade without alarm | Contracts and column-presence tests |
| No raw retention | Cannot replay or audit | Disputes unresolvable | Immutable bronze partitions |
| Non-idempotent jobs | Re-runs duplicate rows | Corrupted history | Partition replace, deterministic keys |
| Quarantine as a log line | Bad rows vanish | Undetected data loss | Quarantine table with review workflow |
| Freshness unmonitored | Stale dashboard looks current | Decisions on old data | Freshness SLA and publish gate |
| Credentials in code | Secrets in repositories | Security incident | Vault, rotation, secret scanning |

## 12. Interview and discussion questions

1. **"What is the difference between a snapshot fact and a transaction fact, and why can't you sum them?"** — A snapshot is a state at a point in time; a transaction is a flow. Summing daily snapshots multiplies inventory by the number of days.
2. **"How would you join an order to the item master when the case pack changed?"** — As-of join on `valid_from <= order_date`, with `valid_to` as a half-open bound, and a test that exactly one version matches.
3. **"An 856 says 120 EA and the invoice says 5 CS. What do you do?"** — Convert with the effective case pack, reconcile to eaches, and flag the mismatch if 5 CS × 24 ≠ 120.
4. **"How do you detect a late-arriving fact?"** — Compare event time to the watermark, monitor lateness distribution, and either hold the partition or restate it with a documented policy.
5. **"What belongs in bronze?"** — Exactly what arrived, plus ingest metadata; no cleaning, so the source can be replayed and disputed.
6. **"How do you make a pipeline idempotent?"** — Deterministic keys, partition-scoped writes, and replace-not-append semantics for each logical date.
7. **"What is a data contract and who owns it?"** — A versioned agreement on schema, semantics, quality, and freshness between producer and consumer; the producer owns delivery, the consumer owns fitness for use.
9. **"What tests would you put on a demand mart?"** — Uniqueness at the declared grain, non-negative demand, referential integrity, freshness, and volume within an expected band.
10. **"How do you handle units across an EDI 850, a WMS, and a forecast?"** — One conversion module, store the source unit and quantity, store the normalized eaches, and reconcile totals at every boundary.
11. **"What is your first action in a data incident?"** — Stop publishing bad data, preserve the raw evidence, and establish the blast radius before fixing.

## 13. Key terms

- **ASN** — Advance Ship Notice; the X12 856 or EDIFACT DESADV describing a shipment's contents.
- **As-of join** — A join that selects the record version valid at a given point in time.
- **Bronze layer** — Immutable raw landing zone preserving source fidelity.
- **Silver layer** — Conformed, typed, deduplicated, unit-normalized data.
- **Gold layer** — Decision-grain marts with tests and documentation.
- **CDC** — Change Data Capture; reading a database log to capture inserts, updates, and deletes.
- **Data contract** — A versioned, testable agreement on schema, semantics, quality, and freshness.
- **EDI** — Electronic Data Interchange; structured business documents exchanged between organizations.
- **Entity resolution** — Deciding when records from different sources refer to the same entity.
- **EPCIS** — Electronic Product Code Information Services; GS1's event-based traceability standard.
- **Gaps and islands** — A SQL technique for finding contiguous runs in a sequence.
- **GLN** — Global Location Number; a GS1 identifier for a party or location.
- **GTIN** — Global Trade Item Number; a GS1 identifier for a trade item.
- **Idempotency** — The property that repeating an operation produces the same result.
- **Lineage** — The documented path from source data through transformations to an output.
- **MDM** — Master Data Management; creating and governing one trusted record per entity.
- **Quarantine** — A holding table for records that fail validation, with review workflow.
- **SCD Type 2** — A dimension that preserves history as multiple rows with validity dates.
- **SSCC** — Serial Shipping Container Code; an 18-digit GS1 identifier for a logistic unit.
- **Watermark** — A stored position marking how much of a source has been processed.

## 14. Sources and further reading

**Standards and specifications**
- ANSI ASC X12, transaction sets 850, 855, 856, 810, 846, 852, 940, 945, and 997; version 005010 and later. Implementation guides are partner- and version-specific. [VERIFY segment details against the applicable guide before production use.]
- UN/EDIFACT, ORDERS, ORDRSP, DESADV, INVOIC, INVRPT, SLSRPT, and CONTRL message types. Syntax version and directory (for example D.96A) must be declared in `UNH`.
- GS1 General Specifications, current edition; GTIN, GLN, SSCC, and barcode symbologies.
- GS1 EPCIS 2.0 standard and implementation guidance, 2022. [VERIFY exact publication and revision dates.]
- ISO 8601 (date and time), ISO 3166 (country codes), ISO 4217 (currency codes); OpenAPI Specification, version 3.1 and later.

**Engineering practice**
- Kimball, R. and Ross, M., *The Data Warehouse Toolkit*, 3rd edition, Wiley, 2013 — dimensional modeling, SCDs, fact types.
- Kleppmann, M., *Designing Data-Intensive Applications*, O'Reilly, 2017 — batch, stream, replication, and consistency.
- dbt documentation on models, tests, contracts, and exposures, https://docs.getdbt.com (accessed 27 October 2026).

**Case evidence**
- Target Corporation, announcement of Canadian market exit, 15 January 2015; contemporary reporting on Canadian inventory and supply chain problems.
- US Food and Drug Administration, Drug Supply Chain Security Act guidance and stabilization-period materials. [VERIFY current dates and exemptions.]
- Maersk public statements and 2017 annual reporting on the NotPetya attack of 27 June 2017.
- GS1, Sunrise 2027 initiative materials. [VERIFY scope and partner readiness.]

**Context**
- Course source pack: `_research/industry_brief.md` (2026 adoption, regulation, and trade context).

---

> **Next:** [Module 04](04_statistics_causal_inference.md) — from trustworthy data to trustworthy inference: estimation, uncertainty, confounding, and causal designs for supply chain decisions.
