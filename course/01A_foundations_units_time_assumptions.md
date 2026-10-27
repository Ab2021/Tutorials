# Module 01A — Foundations: Units, Time, Measurement, and Assumptions

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 24 hours (9 lecture, 10 lab, 5 project)
> **Track fit:** Common core (all three tracks)
> **Prerequisites:** Module 01; basic algebra and Python
> **Tools:** Python 3.11+, NumPy, pandas, DuckDB, a calendar library (`exchange_calendars` or `pandas_market_calendars` optional)
> **Industry question:** "What exactly is one unit, over exactly what time period, measured how, and on which assumptions does the answer depend?"

---

## 1. Why industry cares

Most supply chain analytics failures are not model failures. They are **measurement failures**: the model answered a question that was not asked, in units that did not match, over a period that did not align, on an assumption nobody wrote down.

The economics are asymmetric. A unit error does not degrade a forecast by 2%; it multiplies inventory by the case pack. A calendar error does not blur a trend; it creates a phantom one. A lead-time definition error does not add noise; it guarantees a stockout at the reorder point, every cycle, because the replenishment arrives after the point at which it was assumed to be available.

| Error class | Typical trigger | Consequence | Detection difficulty |
|---|---|---|---|
| Unit of measure (UoM) | Forecast in eaches, order in cases | 12× to 48× inventory or shortage | Easy once examined, invisible in aggregate |
| Case pack / MOQ | Pack size changed, master data stale | Systematic over- or under-supply | Hard: no exception is raised |
| Rounding | Independent rounding at several levels | Compounding overshoot, 10–20% | Hard: each step looks correct |
| Calendar | Fiscal month vs calendar month | False trend, misaligned plan | Medium |
| ISO week | 53-week year, week 1 spillover | Duplicate or missing weeks | Hard |
| Time zone / DST | Cut-off, carrier SLA, store hours | One-hour miss on promised date | Medium |
| Lead time | Order-to-receipt vs order-to-available | Reorder point too low | Hard |
| Currency | Budget rate vs actual, unhedged | Margin variance | Easy in finance, invisible in ops |
| Metric | MAPE on intermittent demand | Wrong model selected | Medium |
| Assumption | Undocumented demand or capacity assumption | Plan infeasible | Very hard |

The 2026 context makes this worse, not better. Data now arrives from more systems, in more units, at more frequencies, and through more partners than in 2015. LLM-based extraction and agentic planning add a new failure mode: a plausible-looking number whose provenance, unit, and period are unstated. The analyst who can **audit a number back to its definition** is more valuable than the analyst who can add another model.

> **Industry interpretation.** A number without a unit, a period, a population, and a source is not data. It is an opinion with a decimal point.

## 2. Learning outcomes

By the end of this module, you can:

- Construct a unit-of-measure hierarchy from each to truckload and convert between any two levels without floating-point drift.
- Apply case pack, layer, pallet, MOQ, and integer rounding rules, and quantify the compounding overshoot they create.
- Convert between weight, volume, and container capacity, and identify the binding constraint on a shipment.
- Align data to fiscal, 4-5-4 retail, ISO, and calendar time, including 53-week years, time zones, and daylight saving transitions.
- Define lead time at the component level and compute the correct protection interval for a reorder point.
- Choose a planning bucket and explain the aggregation error it introduces.
- Convert and compare across currencies using budget, actual, hedge, and constant-currency conventions.
- Distinguish accuracy, precision, bias, and measurement error, and decompose mean squared error into bias and variance.
- Compute MAE, RMSE, MAPE, WAPE, MASE, bias, and pinball loss, and state when each one fails.
- Classify a demand series by intermittency and choose a distribution appropriate to it.
- Maintain an assumptions ledger that another analyst could audit and reproduce.

## 3. Mental model

### 3.1 The measurement stack

```
  ENTITY        What is counted?         item, location, customer, lane
     ↓
  UNIT          In what?                 each, case, pallet, kg, m³, TEU, currency
     ↓
  TIME          Over what period?        event time, bucket, fiscal calendar, time zone
     ↓
  MEASURE       How accurately?          accuracy, precision, bias, censoring
     ↓
  METRIC        Scored how?              MAE, WAPE, MASE, pinball, service
     ↓
  ASSUMPTION    What must be true?       demand, supply, cost, capacity, behaviour
     ↓
  DECISION      So what?
```

Every layer can silently invalidate the one below it. The stack is read **bottom-up for audit** ("which assumption produced this number?") and **top-down for design** ("which entity and unit does this decision need?").

### 3.2 The conversion chain

```
 Customer demand (eaches/day)
   → forecast (eaches/week)
     → replenishment order (cases)
       → warehouse pick (layers)
         → transport (pallets, kg, m³)
           → customs (kg, value, HS code)
             → finance (currency, COGS, inventory value)
```

Each arrow is a conversion with a rounding rule. A common bug is to convert at every arrow independently and round each time. The correct pattern is to **carry full precision through the chain and apply a single rounding rule at the point where the physical constraint binds** — usually the order or the pallet.

### 3.3 The two questions that catch most errors

1. **"Compared with what?"** — a number needs a baseline, a period, and a population.
2. **"In what unit, and at what grain?"** — a quantity without a unit is a count of something unspecified.

## 4. Core concepts

### 4.1 The unit-of-measure hierarchy

| Level | Unit | Contains | Eaches | Typical use | Recorded in |
|---|---|---|---:|---|---|
| 1 | Each (EA) | 1 | 1 | Selling unit, POS, e-commerce | POS, OMS |
| 2 | Inner pack | 12 eaches | 12 | Shelf-ready tray, retail display | PIM, WMS |
| 3 | Case | 4 inner packs | 48 | Purchasing, receiving, picking | ERP, WMS |
| 4 | Layer (tier) | 8 cases | 384 | Pallet build, partial-pallet picking | WMS |
| 5 | Pallet | 6 layers | 2,304 | DC handling, ASN, transport | WMS, TMS |
| 6 | Truckload | 26 pallets | 59,904 | FTL tender, freight rating | TMS |
| — | Kilogram (kg) | mass | — | Freight, customs, emissions | TMS, customs |
| — | Cubic metre (m³) | volume | — | Cube, container loading | TMS, WMS |
| — | Twenty-foot equivalent unit (TEU) | one 20 ft container | — | Ocean capacity, port statistics | TMS, carrier |

The hierarchy is not universal. A "case" is whatever the supplier and the customer agreed it is. Two suppliers of the same item can ship different case packs; the same supplier can change pack size without changing the item number. That is why the case pack belongs in **master data with an effective date**, not in a formula.

Worked build-up. An item has an each weight of 0.85 kg and a case dimension of 0.30 m × 0.20 m × 0.15 m.

| Level | Count | Weight | Volume |
|---|---:|---:|---:|
| Each | 1 | 0.85 kg | 0.009 m³ |
| Case (24 eaches) | 24 | 20.4 kg | 0.216 m³ |
| Pallet (96 cases) | 96 | 1,958.4 kg + 25 kg pallet = 1,983.4 kg | 0.864 m³ |
| Truckload (26 pallets) | 2,496 cases | 51,568.4 kg | 22.46 m³ |

Pallet cube utilization: the stack occupies 1.2 m × 0.8 m × 1.35 m = 1.296 m³, so 0.864 ÷ 1.296 = **66.7%**. The remaining third is air, pallet void, and overhang allowance. Cube utilization is a physical constraint, not a cosmetic metric.

### 4.2 Container loading: which constraint binds?

A 40 ft high-cube container has roughly 76 m³ of internal volume and a payload of about 26,500 kg. Usable cube at 85% is 64.6 m³. With the pallet above:

| Constraint | Calculation | Limit | Pallets allowed |
|---|---|---:|---:|
| Weight | 26,500 ÷ 1,983.4 | 26,500 kg | 13 |
| Floor positions | 10 along × 2 across (1.2 m × 0.8 m pallet) | 20 positions | 20 |
| Cube | 64.6 ÷ 1.296 | 64.6 m³ | 49 |

The shipment is **weight-limited at 13 pallets**, not volume-limited. A planner who loads by cube will overload the container by 7 pallets (13,884 kg over payload) and create a safety and compliance incident. The same item shipped as light, bulky goods would be volume-limited. The lesson: compute both constraints every time; never assume which binds.

TEU arithmetic: a 40 ft container is 2 TEU. A service advertised as 10,000 TEU per week is 5,000 forty-foot moves if all units are 40 ft. Mixing 20 ft and 40 ft without converting to TEU is a common capacity-reporting error.

### 4.3 Case pack, MOQ, and integer rounding

Three independent integers constrain an order: **case pack** (units per case), **layer/pallet configuration** (cases per handling unit), and **minimum order quantity (MOQ)** (the smallest order a supplier will accept). They rarely divide evenly into each other.

Worked example. Weekly demand is 4,320 eaches. Case pack is 24; a layer holds 12 cases; a pallet holds 8 layers (96 cases). MOQ is 20 cases.

- Demand in cases = 4,320 ÷ 24 = **180 cases** exactly.
- 180 cases ÷ 12 = **15 layers** exactly.
- 180 ÷ 96 = **1.875 pallets**. Shipping whole pallets means 2 pallets = 192 cases = 4,608 eaches, an overshoot of 288 eaches or **6.67%**.
- If the order is rounded to whole layers, 15 layers is exact and there is no overshoot. The rounding rule, not the demand, determined the inventory.

Worked compounding example. Demand is 3,000 eaches. Case pack 48; layer 12 cases; pallet 6 layers (72 cases).

| Step | Exact | Rounded | Resulting eaches | Error vs 3,000 |
|---|---:|---:|---:|---:|
| Demand to cases | 62.5 | 63 (up) | 3,024 | +0.8% |
| Cases to layers | 5.25 | 6 (up) | 3,456 | +15.2% |
| Cases to pallets | 0.875 | 1 (up) | 3,456 | +15.2% |

Two independent round-ups turned a 0.8% error into a 15.2% error. In a network with 40 SKUs and weekly cycles, that is a systematic inventory build that no forecast improvement will offset.

Rounding rules to state explicitly in any model:

| Rule | When to use | Effect |
|---|---|---|
| Round up to case | Order must be shippable | Never under-supplies; small overshoot |
| Round to nearest case | Demand planning, cost analysis | Unbiased on average, can under-supply |
| Round up to layer | Pallet-build constraint | Moderate overshoot |
| Round up to pallet | Full-pallet-only receiving | Large overshoot for slow movers |
| Round down | Capacity-constrained allocation | Under-supplies; requires rationing logic |
| Floor plus remainder | Mixed pallets allowed | Lowest cost, highest handling complexity |

MOQ interacts with shelf life. An MOQ of 500 cases against weekly demand of 180 cases creates 2.8 weeks of cover; for a product with a 21-day shelf life, that is a markdown decision disguised as a purchasing rule.

### 4.4 Time: calendars, weeks, zones, and buckets

#### Fiscal and retail calendars

| Calendar | Definition | Year length | Alignment | Where it appears |
|---|---|---|---|---|
| Gregorian | 12 months, 28–31 days | 365/366 days | 1 January | Most ERP defaults |
| Fiscal (arbitrary) | Company-chosen start month | 365/366 days | e.g., 1 April | Finance, statutory reporting |
| 4-5-4 retail | 13 weeks per quarter as 4+5+4 weeks | 364 days, plus 1–2 days | Fiscal year ends on a Saturday near 31 January | Retail merchandising, category reporting |
| 53-week | 4-5-4 year with an extra week | 371 days | Roughly every 5–6 years | Retail comparability |
| ISO 8601 week | Monday–Sunday, week 1 contains the first Thursday | 52 or 53 weeks | Global | Logistics, European operations |
| Julian day | Day number within year | 365/366 | 1 January | Legacy systems, batch jobs |

The 4-5-4 calendar makes every quarter 13 weeks, so like-for-like weekly comparisons work, but the year is 364 days and drifts one or two days against the Gregorian year. A 53-week year is added periodically to realign. The U.S. National Retail Federation (NRF) publishes the 4-5-4 retail calendar used by many retailers; the exact dates must be taken from the current NRF calendar, not calculated ad hoc.

Worked ISO example. 1 January 2026 is a Thursday, so ISO week 1 of 2026 begins on Monday 29 December 2025 and ISO year 2026 contains **53 weeks**. Week 53 runs from Monday 28 December 2026 to Sunday 3 January 2027. A pipeline that joins on `ISO year + week number` will place 1–3 January 2027 into 2026 week 53 and double-count it if the same days also appear in 2027 week 1. The fix is to join on the **week start date**, not on the week number.

Worked fiscal example. A retailer's fiscal year ends on the Saturday nearest 31 January. For FY2025, if the year ends Saturday 31 January 2026, then FY2025 begins Sunday 1 February 2025 and contains 52 weeks and one day. The following year may contain 53 weeks depending on the weekday alignment.

#### Planning buckets

| Bucket | Length | Best for | Aggregation risk |
|---|---|---|---|
| Shift | 8–12 h | Labour, production scheduling | Ignores within-shift variability |
| Day | 24 h | Replenishment, e-commerce | Weekday/weekend mix |
| Week | 7 days | Most operational planning | Holiday weeks, 53-week years |
| 4-5-4 period | 4 or 5 weeks | Retail merchandising | Period length varies within quarter |
| Calendar month | 28–31 days | Finance, S&OP | Trading-day distortion up to ±10% |
| Quarter | 13 weeks | Executive review | Hides intra-quarter swings |
| Year | 52/53 weeks | Strategy, contracts | Hides seasonality |

Worked trading-day distortion. A product sells 400 units per day. A 31-day month yields 12,400 units; a 28-day month yields 11,200 units. Month-over-month comparison shows a **9.7% decline** with no change in underlying demand. Any monthly forecast that does not model trading days will chase this phantom.

#### Time zones and daylight saving

Order cut-off is 17:00 Central European Time (CET, UTC+1) in Rotterdam. In New York, that is 11:00 Eastern Standard Time (EST, UTC−5) — a six-hour difference. During U.S. daylight saving time (EDT, UTC−4), it is 10:00, a five-hour difference. The U.S. and EU switch on different dates: in 2026 the U.S. moves forward on Sunday 8 March and back on Sunday 1 November; the EU moves forward on Sunday 29 March and back on Sunday 25 October. Between 8 and 29 March 2026, the transatlantic difference is five hours rather than the usual six.

A carrier service level of "next business day by 12:00 local" is therefore ambiguous across a DST boundary. Store the timestamp in UTC, the local time zone identifier (IANA tz database name, e.g., `Europe/Amsterdam`), and the offset actually applied. Never store a local time without its zone.

### 4.5 Lead-time definitions

Lead time is not one number. It is a chain of intervals, each owned by a different function and recorded in a different system.

| Component | Starts | Ends | Owner | Typical system | Common error |
|---|---|---|---|---|---|
| Order processing | Order placed | Order confirmed | Customer service / EDI | OMS, EDI | Confirmation treated as shipment |
| Manufacturing / picking | Order released | Ready to ship | Plant / warehouse | MES, WMS | Queue time omitted |
| Order-to-ship | Order placed | Goods shipped | Supply planning | ERP, WMS | Includes or excludes weekends inconsistently |
| Transit (ship-to-receive) | Shipped | Received at dock | Carrier / TMS | TMS, carrier EDI | Uses scheduled, not actual, arrival |
| Dock-to-stock | Received | Put away and available | Warehouse | WMS | Receipt recorded before QC |
| Total replenishment lead time | Order placed | Available to promise | End-to-end | Composite | Sums components measured on different bases |

Worked timeline. An order is placed Monday 5 October. The supplier confirms Tuesday 6 October (1 day), ships Thursday 8 October (3 days from order), transit takes 12 days, the container arrives Tuesday 20 October, quality checks on Wednesday 21 October, and stock is available Thursday 22 October.

| Component | Interval | Days |
|---|---|---:|
| Order-to-ship | 5 Oct → 8 Oct | 3 |
| Ship-to-receive | 8 Oct → 20 Oct | 12 |
| Dock-to-stock | 20 Oct → 22 Oct | 2 |
| **Total replenishment lead time** | 5 Oct → 22 Oct | **17** |

If demand is 250 units per day and daily demand standard deviation is 60, the protection interval is 17 days. Demand over lead time has mean 250 × 17 = 4,250 units and standard deviation 60 × √17 = 247.4 units. At 95% cycle service level, safety stock = 1.645 × 247.4 = **407 units** and the reorder point is **4,657 units**.

If the planner had used the 12-day transit only, the reorder point would be 250 × 12 + 1.645 × 60 × √12 = 3,000 + 342 = 3,342 units — **1,315 units too low**, a stockout of more than five days of demand on every cycle.

Planned versus actual lead time: planned lead time is the parameter in the planning system; actual lead time is the observed interval from transaction timestamps. The gap between them is a **bias**, and it must be measured, not assumed away. Report lead time as a distribution (median, 90th percentile, and standard deviation), not a single average.

### 4.6 Currency, FX, and constant currency

| Convention | Definition | Use | Risk |
|---|---|---|---|
| Transaction currency | Currency of the contract | PO, invoice | None if recorded |
| Functional currency | Currency of the entity's primary environment | Accounting | Translation rules |
| Reporting currency | Currency of consolidated statements | Group reporting | FX translation |
| Budget rate | Rate fixed at budget time | Variance analysis | Diverges from actual |
| Actual rate | Rate on transaction or period average | Finance actuals | Volatility |
| Hedge rate | Contracted rate for a portion of exposure | Treasury | Residual unhedged exposure |
| Constant currency | Restates prior period at current rates | Like-for-like growth | Can obscure real price effects |

Worked example. A European supplier prices at EUR 12.50 per unit. The budget rate is USD 1.08 per EUR; the period average actual rate is USD 1.12. Annual volume is 120,000 units. The company hedges 70% of exposure at USD 1.09.

- Budget cost per unit = 12.50 × 1.08 = **USD 13.50**
- Unhedged actual cost = 12.50 × 1.12 = **USD 14.00**, a variance of USD 0.50 per unit, or **USD 60,000** per year
- Blended hedged rate = 0.70 × 1.09 + 0.30 × 1.12 = 0.763 + 0.336 = **USD 1.099**
- Hedged cost per unit = 12.50 × 1.099 = **USD 13.7375**, variance USD 0.2375 per unit, or **USD 28,500** per year
- The hedge avoided USD 31,500 of variance, at the cost of the hedge instrument

Constant-currency growth. If USD revenue grows 5% in local terms while the EUR depreciates 8% against the USD, reported USD growth is (1.05 × 0.92) − 1 = **−3.4%**. A headline of "revenue declined" and a constant-currency headline of "+5%" are both true and describe different things. State which one is being used.

### 4.7 Hierarchy design

| Hierarchy | Levels (example) | Join key | Design rule | Common defect |
|---|---|---|---|---|
| Product | Item → subcategory → category → department | GTIN or item ID | Every item has exactly one parent at each level | Item recoded, history split |
| Location | Store/DC → region → country → cluster | GLN or location ID | Separate physical from planning location | Store closures leave orphans |
| Customer | Ship-to → sold-to → parent → channel | Customer ID, GLN | Model the billing and delivery parties separately | Duplicate customer records |
| Channel | POS, e-commerce, wholesale, marketplace | Channel code | One order belongs to one channel | Channel reclassification breaks history |
| Time | Day → week → period → quarter → year | Date or week start | Join on dates, not labels | 53-week and fiscal drift |
| Supplier | Vendor → parent → tier | Vendor ID, DUNS or GLN | Distinguish vendor site from legal entity | Tier-2 invisible |

Hierarchy design rule: **a dimension member must have exactly one parent at each level, and the parent must be valid as of the event date.** If history is restated when the hierarchy changes, every prior comparison becomes ambiguous. Slowly changing dimension type 2 (SCD2) preserves history; Module 03 implements it.

### 4.8 Accuracy, precision, bias, and measurement error

| Concept | Definition | Example |
|---|---|---|
| Accuracy | Closeness of the mean to the true value | Forecasts average 100 when truth is 100 |
| Precision | Closeness of repeated measurements to each other | Forecasts vary by ±1 unit |
| Bias | Systematic deviation from truth | Forecasts average 110 when truth is 100 |
| Measurement error | Difference between recorded and true value | Scan misses a unit; scale is miscalibrated |
| Censoring | True value unobserved because of a constraint | Stockout hides demand |
| Truncation | Values outside a range are excluded | Returns above a threshold dropped |

Worked decomposition. True demand is 100 units in each of four periods.

| Forecaster | Forecasts | Mean | Bias | SD | MSE (bias² + variance) |
|---|---|---:|---:|---:|---:|
| A | 98, 102, 99, 101 | 100 | 0 | 1.83 | 0 + 3.35 = 3.35 |
| B | 110, 109, 111, 110 | 110 | +10 | 0.82 | 100 + 0.67 = 100.67 |
| C | 90, 112, 95, 103 | 100 | 0 | 9.54 | 0 + 91.0 = 91.0 |

Forecaster B is precise and biased; C is unbiased and imprecise; A is both accurate and precise. MSE alone would rank A best and B worst, but it would not tell you that B's problem is a fixable constant and C's is irreducible noise. Always report **bias and dispersion separately**, then the combined metric.

Measurement error in the data is different from forecast error. If 2% of store scans fail during a promotion, recorded sales understate true sales by 2% precisely when demand is highest. That is not noise; it is a systematic, event-correlated bias, and it will teach the model that promotions are less effective than they are.

### 4.9 Forecast error metrics

Let $a_t$ be actual, $f_t$ forecast, $e_t = f_t - a_t$, and $n$ the number of periods.

| Metric | Formula | Unit | Scale-free | Robust to zeros | Penalizes bias | Best use |
|---|---|---|---|---|---|---|
| MAE | $\frac{1}{n}\sum\lvert e_t\rvert$ | units | No | Yes | No | Operational error in units |
| RMSE | $\sqrt{\frac{1}{n}\sum e_t^2}$ | units | No | Yes | No | Penalizing large errors |
| MAPE | $\frac{1}{n}\sum\lvert e_t/a_t\rvert$ | % | Yes | **No** | No | High-volume, strictly positive series |
| WAPE | $\sum\lvert e_t\rvert / \sum\lvert a_t\rvert$ | % | Yes | Yes | No | Portfolio-level weighted accuracy |
| MASE | MAE ÷ naive MAE | ratio | Yes | Yes | No | Comparing across series and scales |
| Bias | $\sum e_t / \sum a_t$ | % | Yes | Yes | **Yes (it is the metric)** | Detecting systematic over/under-forecast |
| Pinball | mean of $\max(q(a-f),(q-1)(a-f))$ | units | No | Yes | Asymmetric | Quantile and probabilistic forecasts |

Worked calculation. Actuals: 100, 120, 80, 140, 60, 100. Forecasts: 110, 110, 90, 120, 80, 95.

| Period | Actual | Forecast | Error (f−a) | Absolute | Squared | APE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 110 | +10 | 10 | 100 | 10.00% |
| 2 | 120 | 110 | −10 | 10 | 100 | 8.33% |
| 3 | 80 | 90 | +10 | 10 | 100 | 12.50% |
| 4 | 140 | 120 | −20 | 20 | 400 | 14.29% |
| 5 | 60 | 80 | +20 | 20 | 400 | 33.33% |
| 6 | 100 | 95 | −5 | 5 | 25 | 5.00% |
| **Sum** | **600** | **605** | **+5** | **75** | **1,125** | **83.45%** |

- MAE = 75 ÷ 6 = **12.50 units**
- MSE = 1,125 ÷ 6 = 187.50; RMSE = √187.50 = **13.69 units**
- MAPE = 83.45% ÷ 6 = **13.91%**
- WAPE = 75 ÷ 600 = **12.50%**
- Bias = +5 ÷ 600 = **+0.83%** (slight under-forecast)
- Naive forecast (previous actual) absolute errors: 20, 40, 60, 80, 40; naive MAE = 240 ÷ 5 = 48.00; MASE = 12.50 ÷ 48.00 = **0.260**
- Pinball at q = 0.90: under-forecasts (actual above forecast) sum to 10 + 10 + 20 + 5 = 45? No — recompute on $a-f$: −10, +10, −10, +20, −20, +5. Positive terms sum 10 + 20 + 5 = 35; negative absolute terms sum 10 + 10 + 20 = 40. Loss = 0.90 × 35 + 0.10 × 40 = 31.5 + 4.0 = 35.5; ÷ 6 = **5.92 units**
- Pinball at q = 0.10 = 0.10 × 35 + 0.90 × 40 = 3.5 + 36.0 = 39.5; ÷ 6 = **6.58 units**

When each metric fails:

| Metric | Failure mode | Concrete example |
|---|---|---|
| MAE | Not comparable across SKUs with different volume | 10 units error is 10% of 100 and 0.1% of 10,000 |
| RMSE | One outlier dominates the score | A single 500-unit error outweighs ten 20-unit errors |
| MAPE | Undefined at zero; unbounded over-forecast; punishes low volume | Actual 10, forecast 30 = 200%; actual 30, forecast 10 = 33% |
| WAPE | High-volume items dominate; hides small-item failure | One large SKU masks 40% errors on the tail |
| MASE | Depends on the naive definition and history length | Seasonal naive with one year of data is unstable |
| Bias | Cancels over and under errors | +20 and −20 gives zero bias and MAE 20 |
| Pinball | Not comparable across quantiles without averaging | q = 0.5 equals half of MAE |

### 4.10 Why MAPE fails on intermittent and low-volume series

MAPE divides by the actual. When the actual is small, the ratio explodes; when it is zero, the metric is undefined; when the actual is large, a material error looks small.

Worked portfolio example.

| SKU | Actual | Forecast | Absolute error | APE | Share of volume |
|---|---:|---:|---:|---:|---:|
| A | 1,000 | 1,100 | 100 | 10.00% | 99.6% |
| B | 4 | 8 | 4 | 100.00% | 0.4% |
| **Portfolio** | **1,004** | **1,108** | **104** | **10.36% WAPE** | 100% |

Average MAPE = (10.00% + 100.00%) ÷ 2 = **55.0%**, while WAPE = 104 ÷ 1,004 = **10.36%**. A model selected on MAPE would spend its capacity on SKU B and ignore the SKU that represents 99.6% of volume.

Asymmetry. MAPE is bounded at 100% for under-forecasting (error cannot exceed the actual) but unbounded for over-forecasting. Actual 10 with forecast 30 gives 200%; actual 30 with forecast 10 gives 33%. The metric therefore systematically prefers under-forecasting, which in supply chain terms means stockouts.

Intermittent demand classification. Average demand interval (ADI) = number of periods ÷ number of non-zero periods. Squared coefficient of variation (CV²) = (standard deviation ÷ mean)² computed on non-zero demands.

| Class | ADI | CV² | Example | Appropriate method |
|---|---|---|---|---|
| Smooth | < 1.32 | < 0.49 | Fast-moving grocery | Exponential smoothing, ARIMA |
| Erratic | < 1.32 | ≥ 0.49 | Volatile but frequent | Smoothing with wider intervals |
| Intermittent | ≥ 1.32 | < 0.49 | Spare parts, slow SKUs | Croston, SBA, TSB |
| Lumpy | ≥ 1.32 | ≥ 0.49 | Project spares, seasonal parts | Croston variants, aggregate-disaggregate |

Worked classification. Twelve periods contain five non-zero demands: 2, 5, 3, 8, 2. ADI = 12 ÷ 5 = 2.40. Non-zero mean = 4.0; standard deviation = 2.45; CV² = (2.45 ÷ 4.0)² = 0.375. The series is **intermittent** (ADI ≥ 1.32, CV² < 0.49). Seven of twelve actuals are zero, so MAPE is undefined without a zero-handling convention, and any convention that replaces zeros with a small number produces arbitrary results.

The correct treatment is to forecast the **intermittent process** (probability of occurrence × size when it occurs), not to apply MAPE to the raw series. Module 08 covers Croston, SBA, TSB, and hurdle models.

### 4.11 Distributions used in demand and lead time

| Distribution | Support | Parameters | Mean | Variance | Use when | Failure mode |
|---|---|---|---|---|---|---|
| Normal | (−∞, ∞) | μ, σ | μ | σ² | High-volume, symmetric demand; lead-time demand | Produces negative demand; thin tails |
| Poisson | 0, 1, 2, … | λ | λ | λ | Count demand, low volume, mean ≈ variance | Variance exceeds mean (overdispersion) |
| Negative binomial | 0, 1, 2, … | r, p | r(1−p)/p | r(1−p)/p² | Count demand with overdispersion; lumpy | Requires estimating two parameters |
| Lognormal | (0, ∞) | μ, σ (of log) | exp(μ + σ²/2) | (exp(σ²) − 1)exp(2μ + σ²) | Positive, right-skewed; multiplicative effects; lead time | Heavy tail can explode safety stock |
| Gamma | (0, ∞) | shape k, scale θ | kθ | kθ² | Positive lead-time demand; flexible shape | Parameter interpretation is error-prone |
| Empirical / bootstrap | Observed values | none | sample | sample | Enough history; no distributional assumption | Cannot extrapolate beyond observed range |

Worked normal. Lead-time demand is normal with mean 1,000 and SD 170. Probability demand exceeds 1,200 = P(Z > 200/170) = P(Z > 1.176) = **11.98%**. In a periodic-review system without safety stock, roughly one cycle in eight stocks out.

Worked Poisson. Daily demand is Poisson with λ = 4. P(0) = e^−4 = **1.83%**; P(≤3) = 0.0183 + 0.0733 + 0.1465 + 0.1954 = **43.35%**. A stock level of 3 units covers only 43% of days, which is why Poisson demand needs explicit service-level calculation rather than a rule of thumb.

Worked negative binomial. Mean 4, variance 8. Overdispersion is present (variance > mean), so Poisson is wrong. Size r = μ² ÷ (σ² − μ) = 16 ÷ 4 = 4; p = r ÷ (r + μ) = 4 ÷ 8 = 0.5. P(0) = p^r = 0.5^4 = **6.25%**, more than three times the Poisson probability of zero. Using Poisson here would understate the probability of a zero-demand day and overstate the case for holding stock.

Worked lognormal. If demand is lognormal with median 100 and log-scale SD 0.5, the mean is exp(ln 100 + 0.5²/2) = 100 × exp(0.125) = **113.3 units**. The mean exceeds the median by 13.3%; planning to the median under-supplies half the time and understates the average by 11.8%.

### 4.12 Math foundations

| Area | What it buys | Supply chain use | Minimum competence |
|---|---|---|---|
| Linear algebra | Representing vectors, matrices, transformations | Inventory vectors, network flow matrices, demand features | Matrix multiplication, rank, solving linear systems |
| Calculus | Marginal reasoning, optimization | EOQ derivative, cost trade-off, gradient descent | Partial derivatives, chain rule, convexity |
| Probability | Quantifying uncertainty | Lead-time demand, service levels, newsvendor | Distributions, expectation, variance, conditional probability |
| Statistics | Estimation and inference | Forecast evaluation, A/B tests, causal claims | Sampling, confidence intervals, hypothesis tests |
| Graph theory | Structure of networks | Supply network, BOM, routing, shortest path | Nodes, edges, paths, connectivity, centrality |
| Optimization | Choosing the best feasible action | LP, MILP, routing, scheduling, network design | Formulation, duality, integrality, solver behaviour |
| Numerical methods | Making it computable | Simulation, root-finding, approximation | Convergence, stability, floating-point limits |

Two habits matter more than depth. First, **dimensional analysis**: carry units through every equation and check they cancel. Second, **order-of-magnitude checking**: before trusting a solver, estimate the answer roughly and compare.

### 4.13 Accounting foundations

| Concept | Definition | Formula | Why the analyst needs it |
|---|---|---|---|
| COGS | Cost of goods sold in the period | Opening inventory + purchases − closing inventory | Denominator of turns and DIO |
| Gross margin | Revenue minus COGS | Revenue − COGS | Connects service to profit |
| Gross margin % | Margin as a share of revenue | (Revenue − COGS) ÷ revenue | Comparable across sizes |
| Inventory turns | How often inventory is sold | COGS ÷ average inventory | Speed of working capital |
| GMROI | Gross margin return on inventory investment | Gross margin ÷ average inventory at cost | Profit per unit of inventory |
| FIFO | First-in, first-out valuation | Oldest costs expensed first | Common under IFRS |
| Weighted average | Average cost of goods available | Total cost ÷ total units | Smooths price changes |
| LIFO | Last-in, first-out | Newest costs expensed first | Not permitted under IFRS; permitted in U.S. GAAP |

Worked example. Opening inventory is 100 units at USD 20 (USD 2,000). A purchase adds 200 units at USD 22 (USD 4,400). 250 units are sold at USD 40.

| Method | COGS | Ending inventory | Gross margin | Gross margin % |
|---|---:|---:|---:|---:|
| Weighted average | 250 × 21.333 = 5,333.33 | 50 × 21.333 = 1,066.67 | 10,000 − 5,333.33 = 4,666.67 | 46.67% |
| FIFO | 100 × 20 + 150 × 22 = 5,300.00 | 50 × 22 = 1,100.00 | 10,000 − 5,300 = 4,700.00 | 47.00% |

The valuation method changes reported COGS by USD 33.33 and gross margin by 0.33 percentage points on a single small transaction. At scale, a method change can move reported margin materially without any operational change, which is why finance and analytics must agree on the valuation basis before comparing inventory metrics.

Turns and DIO: average inventory at cost = (2,000 + 1,066.67) ÷ 2 = 1,533.33; turns = 5,333.33 ÷ 1,533.33 = **3.48**; DIO = 365 ÷ 3.48 = **104.9 days**.

### 4.14 The assumptions ledger

An assumptions ledger is a structured register of every quantity the analysis treats as given. It is the difference between a reproducible model and a personal opinion.

| Category | Example assumption | Typical value | Unit | Source | Owner | Sensitivity | Review date | If wrong |
|---|---|---|---|---|---|---|---|---|
| Demand | Baseline weekly demand | 4,320 | eaches/week | 52-week history | Demand planner | High | Monthly | Inventory and service both wrong |
| Demand variability | Coefficient of variation | 0.24 | ratio | History, detrended | Demand planner | High | Quarterly | Safety stock mis-set |
| Supply | Supplier lead time | 17 | days | PO-to-availability | Procurement | High | Quarterly | Reorder point too low |
| Supply variability | Lead-time SD | 2.5 | days | Receipt history | Procurement | Medium | Quarterly | Safety stock too low |
| Cost | Holding rate | 25% | % of value/year | Finance policy | Finance | Medium | Annual | Order quantity and mode wrong |
| Cost | Shortage cost | 35% of margin | % | Finance estimate | Finance | High | Annual | Service level too low |
| Capacity | Weekly plant capacity | 12,000 | units/week | MES, engineering | Operations | High | Monthly | Plan infeasible |
| Behaviour | Promoted weeks lift | 40% | % over baseline | Prior promotions | Category | High | Per event | Inventory mismatch |
| Model | Forecast horizon | 13 | weeks | Planning calendar | Planning lead | Medium | Annual | Method mismatch |
| Data | Stockout censoring rate | 3% | % of SKU-days | Audit | Analytics | Medium | Quarterly | Demand understated |
| Commercial | Payment terms | 60 | days | Contract | Procurement | Low | Annual | DSO and cash wrong |
| Regulatory | Duty rate | 6.5% | % of FOB | Customs tariff | Trade compliance | Medium | Per shipment | Landed cost wrong |

Ledger discipline: every assumption needs a **value, unit, source, owner, review date, and sensitivity**. An assumption without an owner is a rumour. A model without a ledger cannot be audited, and an unauditable model will not survive a planning cycle.

### 4.15 Unit and assumption error gallery

| # | Failure | Arithmetic | Consequence | Prevention |
|---|---|---|---|---|
| 1 | Ordered eaches as cases | 1,000 "units" ordered in cases of 48 = 48,000 eaches | 48× inventory, USD cost and space | UoM field mandatory on every order line |
| 2 | Case pack changed, master data not | From 24 to 12; system still divides by 24 | Half the required replenishment, chronic stockout | Effective-dated pack size, change alert |
| 3 | MOQ ignored | Plan orders 37 cases; supplier minimum 500 | 13.5× order, excess and expiry | MOQ as a hard constraint in the model |
| 4 | Cascading round-ups | 3,000 eaches → 63 cases → 6 layers = 3,456 | +15.2% inventory | One rounding point, explicit rule |
| 5 | Calendar month comparison | 31-day month 12,400 vs 28-day 11,200 | False 9.7% decline | Trading-day adjustment |
| 7 | DST cut-off | 17:00 CET = 11:00 EST, 10:00 EDT | One-hour SLA miss | Store UTC plus tz identifier |
| 8 | Lead time to receipt, not availability | 15 days used instead of 17 | ROP 500 units low at 250/day | Define lead time to available-to-promise |
| 9 | FX budget vs actual | 3.7% cost variance unhedged | USD 60,000 annual variance | State rate convention and hedge |
| 10 | Weight vs cube | 20 pallets by floor, 13 by weight | 13.9 t overload | Compute all constraints |
| 12 | Pipeline double count | In-transit added to on-hand and to on-order | Inventory overstated by 3 weeks | Single inventory position definition |

### 4.16 Python: conversions and error metrics

Integer-safe conversion. Use exact rational arithmetic, then round once at the physical constraint.

```python
from fractions import Fraction
from math import ceil

# One case = 4 inner packs x 12 eaches = 48 eaches; one layer = 12 cases.
UOM_IN_EACHES = {
    "each": Fraction(1),
    "inner": Fraction(12),
    "case": Fraction(48),
    "layer": Fraction(12 * 48),
    "pallet": Fraction(6 * 12 * 48),
}

def to_eaches(qty: int | float, uom: str) -> Fraction:
    """Convert a quantity in `uom` to eaches without floating-point drift."""
    return Fraction(qty) * UOM_IN_EACHES[uom]

def from_eaches(qty_eaches, uom: str, rounding: str = "exact"):
    """Convert eaches to `uom`, applying exactly one rounding rule."""
    raw = Fraction(qty_eaches) / UOM_IN_EACHES[uom]
    if rounding == "exact":
        return raw
    if rounding == "up":
        return ceil(raw)              # never under-supply
    if rounding == "down":
        return raw.numerator // raw.denominator
    raise ValueError(f"unknown rounding rule: {rounding}")

assert to_eaches(63, "case") == 3024
assert from_eaches(3000, "case", "up") == 63        # 62.5 -> 63
assert from_eaches(3000, "layer", "up") == 6        # 5.25 -> 6
assert float(to_eaches(1, "pallet")) == 2304.0
```

Physical constraints, computed together rather than sequentially:

```python
def pallets_by_constraint(pallet_kg, pallet_m3, payload_kg, usable_m3, floor_positions):
    """Return pallet counts allowed by each constraint and the binding one."""
    limits = {
        "weight": int(payload_kg // pallet_kg),
        "cube": int(usable_m3 // pallet_m3),
        "floor": floor_positions,
    }
    binding = min(limits, key=limits.get)
    return limits, binding

limits, binding = pallets_by_constraint(
    pallet_kg=1983.4, pallet_m3=1.296, payload_kg=26500,
    usable_m3=64.6, floor_positions=20,
)
assert limits == {"weight": 13, "cube": 49, "floor": 20}
assert binding == "weight"
```

Error metrics, with explicit guards for the cases where each is invalid:

```python
import numpy as np

def mae(a, f):  return float(np.mean(np.abs(np.asarray(f) - np.asarray(a))))
def rmse(a, f): return float(np.sqrt(np.mean((np.asarray(f) - np.asarray(a)) ** 2)))

def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    if np.sum(np.abs(a)) == 0:
        raise ValueError("WAPE undefined when total actual is zero")
    return float(np.sum(np.abs(f - a)) / np.sum(np.abs(a)))

def bias_pct(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sum(f - a) / np.sum(a))

def mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    if np.any(a == 0):
        raise ValueError("MAPE undefined at zero actuals; use WAPE or MASE")
    return float(np.mean(np.abs((f - a) / a)))

def mase(a, f, season=1):
    a, f = np.asarray(a, float), np.asarray(f, float)
    naive = np.mean(np.abs(a[season:] - a[:-season]))
    if naive == 0:
        raise ValueError("naive error is zero; MASE undefined")
    return float(np.mean(np.abs(f - a)) / naive)

def pinball(a, f, q):
    a, f = np.asarray(a, float), np.asarray(f, float)
    e = a - f
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))

actual    = [100, 120, 80, 140, 60, 100]
forecast  = [110, 110, 90, 120, 80, 95]
assert round(mae(actual, forecast), 4) == 12.5
assert round(rmse(actual, forecast), 4) == 13.6931
assert round(wape(actual, forecast), 4) == 0.125
assert round(bias_pct(actual, forecast), 4) == 0.0083
assert round(mape(actual, forecast), 4) == 0.1391
assert round(mase(actual, forecast), 4) == 0.2604
assert round(pinball(actual, forecast, 0.9), 4) == 5.9167
```

The MAPE guard is the point: the function refuses to produce a number rather than producing a misleading one. That is the behaviour a production pipeline needs.

## 5. Data, standards, and workflow

| Standard | Scope | What it fixes | Where it appears |
|---|---|---|---|
| ISO 8601 | Date and time representation | `YYYY-MM-DD`, week numbering, UTC offsets | Every timestamp |
| ISO 4217 | Currency codes | USD, EUR, GBP, three-letter codes | Prices, invoices |
| UN/CEFACT Recommendation 20 | Unit codes | `EA`, `KGM`, `MTR`, `LTR`, `C62` (each) | EDI, customs |
| GS1 General Specifications | GTIN, GLN, SSCC | Item, location, logistics unit identity | Barcode, EPCIS |
| NRF retail calendar | 4-5-4 fiscal periods | Retail week and period definitions | Merchandising |
| IANA tz database | Time zone identifiers | `Europe/Amsterdam`, DST rules | Timestamps, schedulers |
| Incoterms 2020 | Delivery terms | Cost and risk transfer point | Purchase contracts |
| GHG Protocol | Emissions accounting | Scope and allocation rules | Carbon reporting |

Data-quality checks before any analysis:

1. **Unit reconciliation** — does every quantity have a UoM, and does the UoM set match the master data?
2. **Calendar coverage** — one row per entity per bucket, with explicit handling of missing dates.
3. **Duplicate detection** — same order line, same timestamp, different source.
4. **Date logic** — ship date ≥ order date; receipt ≥ ship date; no future actuals.
5. **Sign conventions** — returns and cancellations negative or flagged, not silently positive.
6. **FX provenance** — rate source, rate date, and whether the amount is already converted.
7. **Rounding audit** — where in the chain rounding occurs and whether it is documented.
8. **Assumption register** — every parameter has an owner and a review date.

Workflow: (1) freeze the data extract and record its version; (2) reconcile units to the base unit and record every conversion; (3) align to the decision calendar, not the source calendar; (4) compute metrics with at least two of them; (5) record assumptions in the ledger; (6) have a planner review the units, calendar, and lead-time definitions before modeling.

## 6. Industry case studies

### 6.1 Mars Climate Orbiter unit mismatch (1999)

**Context and outcome.** NASA's Mars Climate Orbiter was lost during orbit insertion in September 1999. The mishap investigation found that one team supplied thruster impulse data in pound-force seconds while the navigation software expected newton-seconds, producing trajectory errors that went undetected until the spacecraft was lost. The NASA Mishap Investigation Board report (1999) documents the interface failure and the absence of an end-to-end unit check.

**What transferred.** Units are an **interface contract**, not a formatting detail. Every interface between teams or systems needs an explicit unit definition and an automated check. **What did not.** The specific software defect is irrelevant; the organizational cause — no shared unit registry and no cross-check — is what generalizes. Supply chain equivalents are ERP-to-WMS quantity interfaces and supplier EDI unit codes.

### 6.2 Ever Given grounding and the Suez Canal blockage (23–29 March 2021)

**Context, intervention, and outcome.** The container ship Ever Given grounded in the Suez Canal on 23 March 2021 and was refloated on 29 March, halting a large share of Europe–Asia container traffic for six days and creating a backlog that took weeks to clear. Carriers rerouted vessels around the Cape of Good Hope, adding roughly one to two weeks of transit depending on route. Lead times and ETAs computed from a fixed transit assumption failed: planners holding safety stock against a 30-day ocean leg discovered that the protection interval had doubled with no change in demand. The event is documented in carrier advisories, port authority statements, and trade press.

**What transferred.** Lead time is a **distribution with a regime**, not a constant. A planning system must support scenario lead times and must show the assumed transit date on every in-transit unit. **What did not.** The specific duration is not a planning parameter; the lesson is the sensitivity, not the number.

### 6.3 M5 accuracy competition and the metric problem (2020–2021)

**Context and intervention.** The M5 competition used Walmart daily sales data: 42,840 hierarchical series, 1,913 training days, a 28-day horizon, and 5,507 teams. The metric was the weighted root mean squared scaled error (WRMSSE), designed to weight series by revenue importance and to scale errors across series — a choice that forced competitors to care about high-value series rather than unweighted error.

**Evidence and outcome.** The results paper (Makridakis, Spiliotis, Assimakopoulos, 2021) documents the benchmark and the dominance of machine-learning methods; the winning approach combined many LightGBM models with a Tweedie objective.

**What transferred.** **Metric design determines model behaviour.** A metric that weights every SKU equally will produce a model that is good on the long tail and poor on the volume that pays for the inventory. **What did not.** A competition metric is not a business objective; WRMSSE does not encode holding cost, shelf life, or service level.

### 6.4 The 4-5-4 retail calendar and like-for-like comparability (ongoing standard)

**Context and intervention.** Retailers report weekly and periodic sales, but calendar months have 28–31 days, so month-over-month comparisons mix trading-day effects with real demand change. The NRF 4-5-4 calendar divides each quarter into 13 weeks (4, 5, and 4), producing comparable periods and a consistent weekly cadence, with periodic 53-week years to realign with the Gregorian calendar.

**Evidence and outcome.** The NRF publishes the retail calendar and its 53-week rules, and retailers using it disclose the convention in financial reporting. Like-for-like comparisons become possible, at the cost of a fiscal year that drifts against the calendar and an extra week that must be handled explicitly in every trend calculation.

**What transferred.** Choose the calendar that matches the decision, then make the convention explicit in every table. **What did not.** The 4-5-4 calendar is not appropriate for daily replenishment or for statutory reporting, which follow their own calendars.

## 7. Labs

### Lab 1A.1 — Build a unit-conversion module

**Objective.** Implement an exact, testable conversion layer from each to pallet and truckload, with explicit rounding at the physical constraint.

**Data.** A product master with item, case pack, layers per pallet, cases per layer, each weight (kg), case dimensions (m), MOQ, and supplier.

**Steps.**
1. Implement `to_eaches` and `from_eaches` using `Fraction`.
2. Add a `round_at` parameter so rounding occurs at exactly one level.
3. Build a function that returns the order quantity in cases, layers, and pallets for a demand input.
4. Add MOQ enforcement and report the overshoot versus demand.
5. Write property-based tests: round-trip conversion, monotonicity, and no negative quantities.
6. Run the module on 20 SKUs and rank them by rounding overshoot.

**Deliverable.** A Python module, a test file, and a one-page table of overshoot by SKU.

**Acceptance criteria.** No floating-point comparisons in assertions; all conversions exact; MOQ enforced; overshoot reported in units and percent; at least 10 tests pass; the largest overshoot is explained.

**Stretch.** Add a shelf-life constraint that caps order quantity and show the interaction with MOQ.

### Lab 1A.2 — Measurement-error and bias simulation

**Objective.** Separate the effects of bias, noise, and censoring on a replenishment decision.

**Data.** Synthetic daily demand with known parameters; a simulated scan-failure process and a stockout-censoring process.

**Steps.**
1. Generate 365 days of demand from a negative binomial distribution.
2. Apply three measurement regimes: unbiased noise, +5% multiplicative bias, and event-correlated scan failure during promotions.
3. Run a base-stock policy on recorded demand and measure stockouts against true demand.
4. Decompose MSE into bias² and variance for a simple forecast in each regime.
5. Quantify the cost of each regime in units short and inventory carried.

**Deliverable.** A notebook and a 500-word memo on which error matters most and why.

**Acceptance criteria.** True and recorded demand distinguished; seed fixed; bias and variance reported separately; the memo identifies the event-correlated error as the most damaging and supports it with numbers.

**Stretch.** Add a correction model for censored demand and measure the improvement.

### Lab 1A.3 — Forecast-metric shootout, including intermittent demand

**Objective.** Show how metric choice changes model selection, especially on low-volume and intermittent series.

**Data.** At least 50 series from M5 or Favorita, plus 20 intermittent series selected by ADI and CV².

**Steps.**
1. Compute MAE, RMSE, MAPE, WAPE, MASE, bias, and pinball at q = 0.9 for a naive, seasonal-naive, and exponential-smoothing forecast.
2. Rank the three methods by each metric and record where the rankings disagree.
3. Classify every series by ADI and CV².
4. Show the MAPE failure on series with zeros or values below 5 units.
5. Recommend a metric set for a retail portfolio and defend it.

**Deliverable.** A benchmark table, a disagreement matrix, and a two-page recommendation.

**Acceptance criteria.** At least three metrics computed correctly; zero handling explicit; rankings compared; MAPE failure demonstrated with a concrete series; recommendation states the decision the metric serves.

**Stretch.** Add a cost-weighted metric using holding and shortage costs and compare the ranking.

### Lab 1A.4 — Calendar and lead-time reconstruction

**Objective.** Rebuild a planning calendar and a lead-time distribution from raw timestamps.

**Data.** Twelve months of purchase orders, receipts, and inventory snapshots with UTC timestamps and location time zones.

**Steps.**
1. Convert all timestamps to UTC and store the local zone identifier.
2. Build a date spine at daily grain and a week spine keyed on week start date.
3. Reconstruct order-to-ship, ship-to-receive, dock-to-stock, and total lead time per PO.
4. Report the lead-time distribution: median, 90th percentile, SD, and trend.
5. Align demand to both calendar and 4-5-4 periods and compare the apparent trend.
6. Identify one DST boundary and one 53-week year and show what breaks without handling.

**Deliverable.** A calendar table, a lead-time distribution chart, and a reconciliation note.

**Acceptance criteria.** No naive local-time joins; week join on start date; all four lead-time components present; distribution reported, not only a mean; the calendar comparison shows a material difference.

**Stretch.** Add a supplier-level lead-time comparison and test whether the difference is statistically and commercially meaningful.

### Lab 1A.5 — Assumptions audit

**Objective.** Audit an existing analysis or model against an assumptions ledger and quantify the exposure.

**Data.** A provided forecasting or inventory model with documentation, plus the underlying data extract.

**Steps.**
1. Extract every implicit assumption from code, documentation, and parameter settings.
2. Classify each into the ledger categories.
3. Assign an owner, source, and review date; mark unknown owners as findings.
4. Test sensitivity of the output to the five highest-impact assumptions.
5. Rank assumptions by impact × uncertainty and propose a validation plan.

**Deliverable.** A completed ledger and a one-page audit memo.

**Acceptance criteria.** At least 15 assumptions identified; at least 8 categories represented; sensitivity quantified in output units; every high-impact assumption has an owner or an explicit finding; the memo names the assumption most likely to invalidate the result.

**Stretch.** Automate the ledger as a YAML file loaded by the model, with schema validation.

## 8. Assignment / project

**Assumptions ledger and measurement audit (individual, 8 hours).**

Choose a model, report, or planning process. Produce:

1. **Unit map** — every quantity in the analysis, its UoM, its source, and the conversion path to the decision unit.
2. **Calendar map** — the time buckets used, the fiscal calendar, and the handling of 53-week years, DST, and trading days.
3. **Lead-time definition** — component-level definitions with the system of record for each.
4. **Metric justification** — at least three error metrics, with a statement of what each one would miss.
5. **Assumptions ledger** — at least 15 assumptions across at least 8 categories, with value, unit, source, owner, sensitivity, and review date.
6. **Error-gallery entry** — one real or reconstructed failure, with arithmetic showing the consequence.

The deliverable must be reproducible: another analyst should be able to rerun the calculations from the ledger and the data extract alone.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Unit integrity | 25% | Some quantities lack UoM | All quantities have UoM and conversions shown | Exact arithmetic, single rounding point, constraint interaction analysed |
| Calendar and time | 20% | Uses source labels without checking | Correct buckets and time zones | Handles 53-week, DST, trading days, and fiscal drift explicitly |
| Lead-time definition | 15% | Single average lead time | Component breakdown | Distribution, trend, and bias against planning parameter |
| Metric literacy | 20% | Uses MAPE by default | Multiple metrics, correct formulas | Selects metric from decision and cost; demonstrates failure modes |
| Assumptions discipline | 10% | Assumptions implicit | Ledger complete | Sensitivity-tested, owned, versioned, machine-readable |
| Communication | 10% | Dense prose | Clear memo with units | Executive-ready; states what would change the conclusion |

## 10. Regulatory, ethical, and commercial lens

- **Units in trade.** Customs declarations, freight contracts, and product labels use regulated units. A conversion error can be a false declaration, not just an analytical mistake. Use UN/CEFACT unit codes and keep the conversion basis on file.
- **Financial reporting.** FX translation, inventory valuation, and revenue recognition follow accounting standards. Analytics that restate revenue or COGS must use the same conventions as finance or label the difference.
- **Measurement in regulated products.** Pharmaceutical dosages, food weights, and hazardous-goods quantities have legal tolerances. A model that rounds in the wrong direction can create a compliance breach.
- **Privacy.** Unit-level data can be re-identifying when combined with location and time. Store-level sales from a single small store can reveal customer behaviour; aggregation thresholds matter.
- **Fairness.** Time-zone and calendar conventions can disadvantage locations. A cut-off defined in headquarters time gives some regions less effective lead time than others; that is an equity issue, not only an operational one.
- **Commercial.** Exchange-rate assumptions, unit definitions, and MOQ terms belong in contracts. An analyst who changes a conversion convention without procurement and finance agreement is changing the commercial meaning of a number.

## 11. Failure modes and red flags

| Failure mode | Early signal | Consequence | Countermeasure |
|---|---|---|---|
| UoM missing on a field | Column named `qty` with no unit | Silent 12–48× error | Mandatory UoM, schema check |
| Unit conversion at every step | Different totals at each level | Compounding overshoot | Convert once, round once |
| Case pack hard-coded | Formula contains `* 24` | Breaks when pack changes | Effective-dated master data |
| MOQ not modelled | Plan below supplier minimum | Order rejected or upsized | MOQ as a hard constraint |
| Fiscal and calendar months mixed | Finance and ops disagree | False trend, audit issue | One calendar per decision, documented |
| Week number join | Duplicate week 53 | Double-counted demand | Join on week start date |
| Local time without zone | Ambiguous cut-off | SLA breach, missed batch | Store UTC plus tz identifier |
| Lead time to receipt | Reorder point chronically low | Repeated stockout | Define to available-to-promise |
| Budget FX used as actual | Margin variance unexplained | Wrong sourcing decision | State rate type and date |
| MAPE on intermittent demand | Huge percentage, small absolute | Wrong model, wrong focus | ADI/CV² classification, WAPE/MASE |
| Assumption without owner | "It's in the spreadsheet" | Cannot be audited | Assumptions ledger |
| Precision confused with accuracy | Tight variance around a biased mean | Confident wrong answer | Report bias and dispersion separately |

## 12. Interview and discussion questions

1. **"Your forecast is in eaches and the supplier ships in cases of 48. Walk me through the conversion."** *Hint:* Convert at full precision, apply MOQ, round once at the order constraint, report overshoot; never round at every level.
2. **"Why is MAPE a bad metric for spare parts?"** *Hint:* Zeros make it undefined; small actuals explode it; over-forecasting is unbounded; use WAPE, MASE, or pinball with ADI/CV².
3. **"How would you compare accuracy across 5,000 SKUs with different volumes?"** *Hint:* Scale-free metrics plus a weighted portfolio metric; report bias separately; segment by volume and intermittency.
4. **"What is the difference between accuracy and precision, and why does it matter?"** *Hint:* Bias versus variance; a precise biased forecast produces confident over- or under-stock; MSE decomposition.
5. **"A planner says lead time is 15 days. What do you ask?"** *Hint:* From which event to which event, in which system, planned or actual, and what is the distribution around it.
6. **"How do you handle a 53-week retail year?"** *Hint:* Explicit 53rd week, comparability adjustment, join on week start date, disclose the convention.
7. **"Why can a 2% unit error become a 15% inventory error?"** *Hint:* Compounding rounding at case, layer, and pallet levels; independent round-ups multiply.
8. **"What is the risk of a normal distribution for lead-time demand?"** *Hint:* Negative demand, thin tails, underestimated extreme service levels; consider gamma, lognormal, or empirical.
9. **"How would you test whether a demand series is intermittent?"** *Hint:* ADI ≥ 1.32 and CV² thresholds on non-zero demands; then choose Croston-family methods.
10. **"Which FX rate would you use to compare two years of margin?"** *Hint:* Constant currency for like-for-like; state budget, actual, and hedge rates; do not mix.
11. **"What belongs in an assumptions ledger?"** *Hint:* Value, unit, source, owner, review date, sensitivity, and consequence if wrong; every category from demand to regulation.
12. **"Tell me about a time a units error reached production. What would have caught it?"** *Hint:* An interface contract, a range check, a reconciliation total, or a unit test at the boundary.

## 13. Key terms

| Term | Definition |
|---|---|
| Accuracy | Closeness of a measurement or forecast to the true value. |
| ADI (average demand interval) | Periods divided by non-zero periods; intermittency measure. |
| Assumptions ledger | Register of every quantity treated as given, with owner and sensitivity. |
| Bias | Systematic over- or under-forecast, measured as mean error relative to actual. |
| Case pack | Number of eaches in a case; a master-data attribute with an effective date. |
| Censoring | True value unobserved because of a constraint such as a stockout. |
| Constant currency | Restating results at a fixed exchange rate to remove FX effects. |
| CV² | Squared coefficient of variation; variability measure used with ADI. |
| 4-5-4 calendar | Retail fiscal calendar of 13-week quarters split 4, 5, 4. |
| Fiscal calendar | Company-defined accounting year, distinct from the Gregorian year. |
| ISO 8601 | International standard for date, time, and week representation. |
| Lead time | Elapsed time between a triggering event and availability. |
| Lognormal | Right-skewed positive distribution used for lead time and demand. |
| MAPE | Mean absolute percentage error; undefined at zero actuals. |
| MASE | Mean absolute scaled error; MAE divided by naive MAE. |
| MOQ | Minimum order quantity a supplier will accept. |
| Negative binomial | Count distribution allowing variance greater than the mean. |
| Pinball loss | Asymmetric loss for quantile forecasts. |
| Poisson | Count distribution with variance equal to the mean. |
| Protection interval | Time over which safety stock must cover demand. |
| Rounding rule | Explicit policy for converting fractional quantities to integers. |
| TEU | Twenty-foot equivalent unit; standard ocean container measure. |
| WAPE | Weighted absolute percentage error; total absolute error over total actual. |

## 14. Sources and further reading

**Standards and regulation**
- ISO 8601, *Date and time — Representations for information interchange*; ISO 4217, *Codes for the representation of currencies*. ISO, current editions.
- UN/CEFACT Recommendation 20, *Codes for Units of Measure Used in International Trade*. UNECE.
- GS1, *General Specifications*, including GTIN, GLN, and SSCC. GS1, current edition.
- National Retail Federation, *4-5-4 Retail Calendar*. NRF, current year.
- International Chamber of Commerce, *Incoterms 2020*. ICC, 2019.
- GHG Protocol, *Corporate Accounting and Reporting Standard*. WRI/WBCSD.

**Peer-reviewed and foundational**
- Hyndman, R., Koehler, A. "Another Look at Measures of Forecast Accuracy." *International Journal of Forecasting* 22(4), 2006.
- Syntetos, A., Boylan, J. "The Accuracy of Intermittent Demand Estimates." *International Journal of Forecasting* 21(2), 2005.
- Croston, J. "Forecasting and Stock Control for Intermittent Demands." *Operational Research Quarterly* 23(3), 1972.
- Silver, E., Pyke, D., Peterson, R. *Inventory Management and Production Planning and Scheduling*. Wiley.
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. "The M5 Accuracy Competition: Results, Findings and Conclusions." *International Journal of Forecasting*, 2022 (results first published 2021).

**Industry and official**
- NASA, *Mars Climate Orbiter Mishap Investigation Board Phase I Report*, 10 November 1999.
- Suez Canal Authority and carrier advisories on the Ever Given refloating, 29 March 2021.
- NRF retail calendar documentation and retailer fiscal-year disclosures, current.
- M5 competition data and results, Kaggle and the M5 methods repository. https://github.com/Mcompetitions/M5-methods (accessed 27 October 2026).

**Tools**
- Python `fractions` documentation, https://docs.python.org/3/library/fractions.html; `pandas` time series documentation, https://pandas.pydata.org/docs/ (both accessed 27 October 2026).
- IANA time zone database. https://www.iana.org/time-zones (accessed 27 October 2026).

---

> **Next:** [Module 02](02_supply_chain_data_landscape.md) — where these units, timestamps, and assumptions actually live: the systems, master data, transaction records, telemetry, and external signals that feed a planning model.
