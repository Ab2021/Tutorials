# Module 21 — E-commerce Supply Chain Analytics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (14 lecture, 10 lab, 6 project)
> **Track fit:** Retail & E-commerce track; open to all
> **Prerequisites:** Modules 07, 09, 10, 12, 13
> **Tools:** Python, DuckDB, OR-Tools, a warehouse or marketplace extract
> **Industry question:** "How do we profitably promise delivery when every order is a tiny, uncertain, returnable shipment?"

---

## 1. Why industry cares

E-commerce breaks the assumptions of store-based supply chain:

- **The order is the demand signal.** A store sees shelf sales; an e-commerce network sees individual orders with addresses, promises, and return paths.
- **The long tail is the business.** A retailer may carry millions of SKUs, most selling a few units a week.
- **The customer sees the promise, not the plan.** Delivery date, shipping cost, and returns policy are part of the product.
- **The network is a set of choices.** Ship from store, from a DC, from a 3PL, or drop-ship from a supplier.
- **Peak is extreme.** Promotional events can multiply volume for a few days, then collapse.

The analytics question is not "what will we sell?" alone. It is "which unit, from which node, by which carrier, at what promise, with what return probability, at what margin?"

## 2. Learning outcomes

By the end of this module, you can:

- Map an e-commerce fulfillment network and its cost drivers.
- Forecast demand at SKU × node × day with promotions and long-tail behavior.
- Choose inventory placement across nodes under service and cost constraints.
- Design a delivery promise and measure its conversion and cost impact.
- Select carriers and modes with a cost-service trade-off.
- Model returns and reverse logistics.
- Plan for peak events with capacity and labor scenarios.
- Evaluate cross-border and marketplace models.
- Measure the end-to-end customer experience and its supply chain causes.
- Build a unit-economics model per order.

## 3. Mental model

```text
  DEMAND ──> PROMISE ──> SOURCE ──> SHIP ──> DELIVER ──> RETURN
     │          │           │          │         │          │
  clickstream  date       node       carrier   experience  disposition
  basket       cost       inventory  mode      NPS/CSAT    resale/scrap
     │          │           │          │         │          │
     └──────────┴───────────┴──────────┴─────────┴──────────┘
                    unit economics per order
```

Every promise is a constraint on the network. If marketing promises next-day delivery, the supply chain must have the item within next-day reach, or pay to expedite.

### 3.1 Fulfillment models

| Model | Inventory owner | Speed | Cost | Control |
|---|---|---|---|---|
| Own DC | Retailer | Medium | Medium | High |
| Ship from store | Retailer | Fast | Medium | Medium |
| 3PL | Retailer owns stock | Medium | Variable | Low |
| Drop-ship | Supplier | Slow–medium | Low holding | Low |
| Marketplace | Seller | Variable | Commission | Low |
| Dark store | Retailer | Very fast | High | High |

Most e-commerce networks are hybrids. The analytics job is to route each order to the cheapest node that can meet the promise.

## 4. Core concepts

### 4.1 Demand forecasting at order grain

E-commerce demand is intermittent, promotion-driven, and censored by stockouts.

| Feature | Why it matters |
|---|---|
| Day of week | Weekly rhythm |
| Promotion calendar | Large spikes |
| Price | Elasticity |
| Stockout flag | Censoring |
| Delivery promise | Demand shifts with speed |
| New item age | Launch curve |
| Return rate | Net demand |
| Geography | Node-level demand |

Forecast at the **fulfillment node**, not only nationally. A national forecast hides the placement problem.

### 4.2 Long-tail forecasting

Most SKUs sell few units. Options:

- Aggregate to product family and disaggregate by share.
- Use attributes and analogy for new items.
- Use intermittent-demand methods (Croston, TSB).
- Set policy by class, not by item.
- Carry only in a central node; ship slower.

The long tail is often unprofitable to stock everywhere. Placement is a portfolio decision.

### 4.3 Inventory placement

Given demand \(d_{i,n}\) for item \(i\) at node \(n\), holding cost \(h_n\), and a service target, choose stock \(S_{i,n}\) to minimize:

\[
\sum_{i,n} h_n S_{i,n} + \sum_{i,n} c_{n\to customer} \cdot \text{split shipments}
\]

subject to a probability of fulfilling from stock within the promise.

Placement heuristics:

| Rule | Use |
|---|---|
| ABC by value | High-value central, fast movers forward |
| XYZ by variability | Stable forward, volatile central |
| Promise-driven | Place where speed is sold |
| Pooling | Consolidate slow movers |
| Postponement | Delay differentiation |

### 4.4 Delivery promise

The promise is a **date and a price**. It affects conversion and cost.

| Promise | Conversion effect | Cost effect |
|---|---|---|
| Same day | Highest | Highest |
| Next day | High | High |
| 2–3 days | Medium | Medium |
| Standard | Lower | Lowest |

Measure the causal effect of promise changes with geo tests or switchbacks. Do not infer from correlation between fast delivery and sales; fast-delivery customers differ.

### 4.5 Shipping cost and carrier selection

Cost drivers:

- Zone and distance.
- Weight and dimensional weight.
- Residential surcharge.
- Fuel surcharge.
- Peak surcharge.
- Return handling.

Carrier selection is a constrained optimization: minimize cost subject to transit time, coverage, and capacity. Use a rate card and a service matrix, not a single average rate.

### 4.6 Returns

Return rate \(r\) turns a sale into a reverse shipment and possibly a write-down.

\[
\text{Net margin} = (1-r)(p - c) - r(c + h_{\text{return}} + d)
\]

where \(d\) is the depreciation or disposal loss. High return categories (apparel, footwear) need return-aware forecasting and pricing.

Return analytics:

- Reason codes and free-text.
- Size and fit issues.
- Serial returners.
- Fraud and wardrobing.
- Disposition: restock, refurbish, liquidate, destroy.

### 4.7 Peak planning

Peak events are capacity problems with a deadline.

| Lever | Lead time | Cost |
|---|---|---|
| Temporary labor | Weeks | Medium |
| Extra shifts | Days | Medium |
| Overflow 3PL | Weeks | High |
| Inventory pre-position | Months | High |
| Promise throttling | Days | Lost sales |
| Carrier capacity | Months | High |

Simulate peak with arrival profiles, not average volume. Plan the cutoff dates backward from the promise.

### 4.8 Unit economics per order

| Line | Example |
|---|---|
| Revenue | €45.00 |
| COGS | €18.00 |
| Pick and pack | €3.20 |
| Outbound shipping | €5.40 |
| Payment fee | €0.90 |
| Return provision | €2.10 |
| Customer acquisition | €6.00 |
| Contribution | €9.40 |

Most e-commerce analytics stops at gross margin. Contribution per order is the decision metric.

### 4.9 Marketplace and cross-border

| Model | Control | Data | Margin |
|---|---|---|---|
| 1P wholesale | High | High | Wholesale |
| 3P marketplace | Low | Partial | Commission |
| Drop-ship | Low | Partial | Margin |
| Cross-border | Medium | Customs data | Duty and freight |

Cross-border adds duty, tax, customs clearance, and longer transit. Model landed cost, not freight alone.

### 4.10 Customer experience analytics

| Metric | Supply chain cause |
|---|---|
| On-time delivery | Carrier, node, promise |
| Delivery accuracy | Pick and pack |
| Damage rate | Packaging, handling |
| Return rate | Fit, quality, description |
| Contact rate | Delay, exception |
| NPS / CSAT | End-to-end experience |

Link experience metrics to operational causes. A late delivery is not a customer-service problem; it is a promise or network problem.

### 4.11 Split-shipment economics, worked

A customer orders three items. The central DC has all three; the forward node has two.

| Option | Freight | Packaging | Speed | Customer effect |
|---|---:|---:|---|---|
| Ship all from central | €7.80 | €1.10 | 3 days | One parcel, slower |
| Ship two forward, one central | €9.60 | €2.20 | 1–3 days | Two parcels, faster |
| Ship all forward after transfer | €6.90 | €1.10 | 2 days | One parcel, transfer cost |

The forward option looks cheaper until the transfer cost and the second-parcel experience are counted. The decision rule is not "ship from the closest node"; it is "minimize total cost subject to the promise, including split-shipment penalty." A useful model charges a fixed penalty per additional parcel and a customer-experience penalty for a split delivery, then compares the full landed cost.

### 4.12 Promise feasibility and the cutoff clock

A promise is feasible only if:

\[
\text{order time} + \text{processing} + \text{transfer} + \text{carrier transit} \le \text{promised date}
\]

Each term has a distribution, not a point. The probability of meeting the promise is the probability that the sum of the terms stays under the deadline. Two consequences follow:

- A promise should be set at a quantile of the transit distribution, not the mean. Promising the mean means missing roughly half the time.
- The order cutoff is a supply chain decision. Moving the cutoff earlier improves on-time performance and reduces the sellable window; moving it later does the reverse.

### 4.13 Reference data model

| Table | Grain | Key fields | Use |
|---|---|---|---|
| order_line | order × line | order_id, sku, qty, promise_date, node | Demand and economics |
| shipment | parcel | shipment_id, order_id, carrier, zone, cost | Carrier and split analysis |
| tracking_event | parcel × event | event_code, event_time, location | On-time and exceptions |
| inventory_snapshot | sku × node × day | on_hand, reserved, inbound | Placement and availability |
| return | return line | reason_code, condition, disposition | Reverse logistics |
| rate_card | lane × weight band | base, fuel, surcharge | Cost model |

The join keys are the analytics contract. If `order_line` cannot be joined to `shipment` at the parcel level, split-shipment cost cannot be measured and the placement model is flying blind.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Orders | Order line | Demand, economics |
| Clickstream | Session | Conversion, funnel |
| Inventory | SKU × node | Placement |
| Shipments | Parcel | Carrier, transit |
| Tracking events | Event | On-time, exceptions |
| Returns | Return | Reverse logistics |
| Product catalog | SKU | Attributes |
| Carrier rate cards | Lane | Cost |
| Warehouse capacity | Node | Peak planning |

### 5.2 Standards

- GS1 GTIN and SSCC for product and pallet identification.
- EPCIS 2.0 for supply chain events.
- Incoterms 2020 for cross-border.
- Country-specific consumer distance-selling rules.
- Packaging and extended producer responsibility rules.

### 5.3 Workflow

1. Define the promise and the unit-economics target.
2. Forecast at node × SKU × day.
3. Allocate inventory to nodes.
4. Route orders to nodes.
5. Select carriers.
6. Measure on-time, cost, returns, and experience.
7. Replan weekly; re-forecast daily where possible.

## 6. Industry case studies

### 6.1 Amazon — regionalization and Prime

**Context:** Amazon built a network of fulfillment centers and Prime's delivery promise.
**Intervention:** Regional inventory placement, own logistics, and a membership model tied to speed.
**Evidence:** Company disclosures describe the network and Prime; specific cost and speed figures are company claims.
**Outcome:** A reference for speed-as-product.
**What transferred:** the promise drives network design, not the reverse.
**What did not:** the scale and capital intensity are not replicable for most retailers.

### 6.2 Zalando — returns and fashion e-commerce

**Context:** Fashion e-commerce has high return rates and size uncertainty.
**Intervention:** Fit analytics, size advice, and reverse logistics.
**Evidence:** Zalando publicly discusses return rates and fit technology; exact figures vary by year.
**Outcome:** Returns are treated as a product problem, not only a cost.
**What transferred:** reducing avoidable returns improves both cost and customer experience.
**What did not:** the assumption that all returns are bad; some are necessary for conversion.

### 6.3 Ocado — automated fulfillment

**Context:** Online grocery requires high-density, high-throughput fulfillment.
**Intervention:** Automated storage and retrieval, grid robots, and route planning.
**Evidence:** Ocado publicly describes its technology and partners; performance figures are company claims.
**Outcome:** A reference for automation in e-commerce grocery.
**What transferred:** the economics depend on volume density and utilization.
**What did not:** the model is capital-intensive and not suitable for low-density markets.

### 6.4 Shein — on-demand small-batch

**Context:** Fast fashion with thousands of new styles.
**Intervention:** Small initial orders, real-time demand signals, and a large supplier network.
**Evidence:** Public reporting describes the model; exact metrics are not audited.
**Outcome:** A test of responsive sourcing versus forecast accuracy.
**What transferred:** small batches and fast replenishment reduce forecast risk.
**What did not:** labor and sustainability concerns are material and must be assessed separately.

## 7. Labs

### Lab 21.1 — Node-level demand forecast

Forecast daily demand for 500 SKUs across three nodes. Compare national vs node-level forecasts and measure the placement error.

**Deliverable:** Forecast, metrics, and node-level error analysis.

### Lab 21.2 — Inventory placement

Given demand, holding costs, and a promise, allocate inventory across nodes. Compare central, forward, and optimized placement on cost and service.

**Deliverable:** Placement model and trade-off table.

### Lab 21.3 — Delivery promise test

Design a geo or switchback test for a promise change. Estimate the sample size and the conversion effect you can detect.

**Deliverable:** Test design, power calculation, and decision rule.

### Lab 21.4 — Unit economics

Build a per-order contribution model for three customer segments. Identify the unprofitable segment and the driver.

**Deliverable:** Unit-economics model and recommendation.

### Lab 21.5 — Peak simulation

Simulate a 4× peak with labor and capacity constraints. Find the bottleneck and the cheapest mitigation.

**Deliverable:** SimPy model, scenario table, and memo.

## 8. Assignment / project

Produce an e-commerce fulfillment strategy for a category.

**Required components:**

1. Promise and customer segment.
2. Demand forecast at node level.
3. Placement plan.
4. Carrier and mode selection.
5. Returns model.
6. Peak plan.
7. Unit economics.
8. Risks and trade-offs.
9. Measurement plan.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Demand model | 20% | Aggregate forecast | Node-level and promotions | Long tail and censoring handled |
| Placement | 20% | Uniform rule | ABC and service | Optimization with trade-offs |
| Promise and experience | 15% | Describes promise | Measures on-time | Causal test and conversion link |
| Unit economics | 20% | Gross margin | Contribution per order | Segment and return effects |
| Peak and capacity | 15% | Average volume | Scenario plan | Simulation and bottleneck |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready |

## 10. Regulatory, ethical, and commercial lens

- **Consumer rights:** distance-selling withdrawal periods and refund rules vary by jurisdiction.
- **Delivery promises:** advertising a speed the network cannot meet is a consumer-protection risk.
- **Returns:** policies must be clear; fraud controls must not punish legitimate customers.
- **Packaging:** extended producer responsibility and plastic rules apply in many markets.
- **Customs and tax:** cross-border sellers must collect and remit correctly.
- **Labor:** peak hiring and warehouse targets affect safety and fairness.
- **Data:** clickstream and address data are personal; minimize and secure.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| National forecast for local placement | One demand number | Ignores node variation | Wrong stock | Node-level forecast |
| Promise without capacity | Marketing sets date | No network check | Late orders | Promise feasibility gate |
| Free returns blindness | Return rate ignored | Margin overstated | Losses | Return-aware economics |
| Long-tail overstock | Every SKU everywhere | Uniform policy | Write-offs | Class-based policy |
| Carrier average rate | One blended cost | Zone and weight ignored | Cost overrun | Rate-card model |
| Peak by average | Plan on mean | Variability ignored | Meltdown | Peak simulation |
| Marketplace data gap | No inventory feed | Blind spots | Stockouts | Data contract |
| Delivery correlation | Fast customers buy more | Selection bias | Wrong causal claim | Experiment |
| Packaging cost creep | Dim weight ignored | Oversized boxes | Freight overrun | Packaging analytics |
| Returns fraud | No reason codes | Unchecked abuse | Losses | Reason and pattern analysis |

## 12. Interview and discussion questions

1. Why is e-commerce demand harder than store demand? *(Hint: order grain, long tail, censoring.)*
2. How do you place inventory across nodes? *(Hint: service, holding, split shipments.)*
3. Does faster delivery increase sales? *(Hint: test causally; selection bias.)*
4. How do returns change unit economics? *(Hint: reverse cost, depreciation, fraud.)*
5. What is the right forecast grain? *(Hint: node × SKU × day, aligned to decision.)*
6. How do you plan for peak? *(Hint: arrival profile, capacity, cutoff dates.)*
7. When should you drop-ship? *(Hint: slow movers, supplier reliability, margin.)*
8. How do you measure carrier performance? *(Hint: on-time by lane, cost per parcel, damage.)*
9. What is the role of a dark store? *(Hint: speed density, high cost.)*
10. How would you reduce avoidable returns? *(Hint: fit, content, size guidance.)*
11. What data would you need for cross-border? *(Hint: duty, tax, customs, transit.)*
12. What is contribution per order? *(Hint: revenue minus all variable costs.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Fulfillment node | A location that ships orders |
| Drop-ship | Supplier ships directly to customer |
| Dark store | Small urban facility for fast delivery |
| Delivery promise | Date and price shown at checkout |
| Split shipment | One order shipped from multiple nodes |
| Dim weight | Volume-based billing weight |
| Long tail | Many low-volume SKUs |
| Return rate | Returns as a share of sales |
| Contribution margin | Revenue minus variable costs |
| Reverse logistics | Returns, repair, resale, disposal |
| Geo test | Randomized market-level experiment |
| Switchback | Alternating treatment over time |
| Landed cost | Product plus freight, duty, and fees |
| 3PL | Third-party logistics provider |
| Order cutoff | Latest time to meet a promise |
| Stockout censoring | Unobserved demand when inventory is zero |
| Regionalization | Placing stock close to demand |
| Marketplace | Platform where third parties sell |
| Unit economics | Profitability per order or unit |
| NPS | Net promoter score |

## 14. Sources and further reading

### Standards and regulation

- GS1, GTIN and SSCC. https://www.gs1.org/standards
- GS1, EPCIS 2.0. https://www.gs1.org/standards/epcis
- ICC, Incoterms 2020. https://iccwbo.org/
- EU Consumer Rights Directive. https://eur-lex.europa.eu/

### Peer-reviewed and professional

- Chopra, S., Meindl, P., *Supply Chain Management* (e-commerce chapters).
- Simchi-Levi, D. et al., *Designing and Managing the Supply Chain*.
- Fisher, M., "What Is the Right Supply Chain for Your Product?" *Harvard Business Review*, 1997.
- Netessine, S., Rudi, N., "Supply Chain Structures on the Internet," in *Handbook of Quantitative Supply Chain Analysis*.

### Industry

- Amazon, annual reports and Prime disclosures. https://www.aboutamazon.com/
- Zalando, annual reports and fit technology materials. https://corporate.zalando.com/
- Ocado, technology and results. https://www.ocadogroup.com/
- Shopify, fulfillment network materials. https://www.shopify.com/

### Tools

- Python, DuckDB, pandas.
- OR-Tools for routing and placement.
- SimPy for peak simulation.

> **Next:** [Module 22](22_quick_commerce_on_demand_delivery.md) compresses the promise from days to minutes.
