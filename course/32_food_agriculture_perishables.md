# Module 32 — Food, Agriculture, and Perishable Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Retail & E-commerce and Logistics tracks
> **Prerequisites:** Modules 07, 08, 10, 21
> **Tools:** Python, DuckDB, SimPy, GeoPandas
> **Industry question:** "How do you sell food before it expires without throwing away the margin?"

---

## 1. Why industry cares

Food supply chains combine the hardest features of every other domain: short shelf life, biological variability, weather, food-safety regulation, thin margins, and a customer who can see the date on the pack. A forecast error is not only a lost sale; it is waste, a markdown, or a safety incident.

The domain is also economically large and politically visible. Food prices, shortages, and safety recalls are public events. Retailers, manufacturers, food service, and quick-commerce operators all compete on freshness and availability, and all of them lose money on the same three failures: too much stock at expiry, too little stock at the peak, and a cold chain that broke somewhere between the farm and the shelf.

The analytics problem is a **shelf-life-constrained inventory and traceability problem**, not a generic demand-planning problem. The unit of analysis is the batch, not the SKU; the clock starts at harvest or production, not at the order; and the objective includes waste and safety alongside service and cost.

## 2. Learning outcomes

By the end of this module, you can:

- Classify a food supply chain by shelf life, temperature regime, and channel.
- Model remaining shelf life at receipt and its effect on available inventory.
- Forecast perishable demand with seasonality, weather, promotions, and substitution.
- Set order quantities with salvage, markdown, and waste costs.
- Plan harvest, yield, and pre-season contracts under weather uncertainty.
- Design cold-chain monitoring and excursion disposition.
- Build lot-level traceability from farm to shelf.
- Apply the food-waste hierarchy and measure waste by cause.
- Evaluate food-safety risk and the analytics that support HACCP.
- Design a freshness promise that operations can keep.

## 3. Mental model

```text
   FARM / HARVEST
   yield, weather, grade, price
          │
          ▼
   PROCESS / PACK ──> lot code, production date, shelf life
          │
          ▼
   COLD CHAIN ──> temperature, time, handling
          │
          ▼
   DC / MARKET ──> remaining shelf life at receipt
          │
          ▼
   STORE / KITCHEN ──> FEFO, markdown, waste
          │
          ▼
   CONSUMER ──> safety, freshness, date code
          │
          └──────────> TRACEABILITY (one up, one down)
```

Two clocks run at once: the **demand clock** (when customers buy) and the **quality clock** (how long the product remains saleable). Most food waste is a mismatch between them. Analytics that only forecasts demand and ignores the quality clock will over-order and under-sell.

### 3.1 Product regimes

| Regime | Examples | Shelf life | Binding constraint |
|---|---|---|---|
| Fresh produce | Leaf, fruit, vegetables | Days | Harvest, temperature, handling |
| Chilled | Dairy, meat, prepared foods | Days–weeks | Cold chain, date code |
| Frozen | Frozen meals, ice cream | Months | Temperature integrity, energy |
| Ambient | Canned, dry, snacks | Months–years | Stock rotation, obsolescence |
| Food service | Restaurant, catering | Hours–days | Prep scheduling, safety |
| Quick commerce | Groceries in 10–30 minutes | Days | Dark-store assortment and waste |

## 4. Core concepts

### 4.1 Shelf life is a declining asset

A batch does not have a fixed value. Its value declines as the remaining shelf life falls, and it may become unsaleable before the physical product is unsafe.

| Age | Remaining life | Commercial state |
|---|---|---|
| 0–25% | Full | Full price, primary shelf |
| 25–50% | Good | Full price, normal rotation |
| 50–75% | Limited | Markdown or secondary channel |
| 75–90% | Short | Clearance, donation, food service |
| >90% | At expiry | Waste or animal feed, subject to law |

Track **remaining shelf life at receipt**, not just expiry date. A DC that receives a batch with 40% of life already gone has less usable inventory than the system's unit count suggests.

### 4.2 FEFO and inventory policy

**First-expired, first-out (FEFO)** replaces FIFO for perishables. The policy must be enforced in the warehouse management system, not only in training.

Policy families:

- **Order-up-to with shelf-life constraint:** order the quantity that covers demand until the next review, but reject or discount receipts with too little remaining life.
- **Newsvendor with salvage:** order the critical fractile when there is a single selling season and a salvage value.
- **Age-based markdown:** reduce price as remaining life falls, to clear stock before expiry.
- **Donation or diversion:** move short-dated stock to food banks, staff sales, or secondary channels before it becomes waste.

The minimum remaining shelf life on receipt is a contractual term. A supplier that ships 60% life when the contract says 75% is creating waste downstream, even if the unit price is the same.

### 4.3 Forecasting perishable demand

Demand for food is driven by:

- Day of week and season.
- Weather, especially temperature and rain.
- Holidays and local events.
- Promotions and price.
- Freshness and display quality.
- Substitution when the preferred item is unavailable.
- Basket context: a missing ingredient can suppress the whole meal.

Forecast at the **item-store-day** grain for fresh, and at the **batch-store-week** grain for longer-life products. Include weather and holiday features, but test whether they add value over a seasonal baseline. A weather feature that is not available at the forecast cutoff is leakage.

**Substitution matters.** If strawberries are out of stock, some customers buy blueberries, some buy nothing, and some go to another store. A forecast that ignores substitution will over-order the substitute and under-order the original.

### 4.4 Newsvendor with waste

For a single order with salvage:

\[
\text{Critical fractile} = \frac{p - c}{p - s}
\]

where \(p\) is selling price, \(c\) is unit cost, and \(s\) is salvage value per unit. The overage cost is \(c - s\); the underage cost is \(p - c\).

If the product can be marked down before expiry, the salvage value is the markdown price, not zero. If it can be donated, the salvage value may be a tax benefit or zero; the disposal cost may be positive.

### 4.5 Harvest, yield, and contracts

Upstream food supply is a planning problem under biological and weather uncertainty:

| Decision | Horizon | Uncertainty |
|---|---|---|
| Acreage and planting | 6–12 months | Price, weather, demand |
| Pre-season contract volume | 3–9 months | Yield, quality, market price |
| Harvest scheduling | Days–weeks | Weather window, labour, maturity |
| Grade and channel allocation | Days | Quality, size, appearance |
| Spot purchase | Days–weeks | Price, availability |

Yield is not a single number. Model a distribution by field, variety, and year, and separate **quantity risk** from **quality risk**. A bumper crop of a low grade can be worse than a smaller crop of a high grade.

Contract design matters: a fixed-volume contract transfers price risk to the buyer and yield risk to the grower; a revenue-sharing or cost-plus contract allocates risk differently. State the allocation explicitly.

### 4.6 Cold chain and quality

Temperature is a quality variable with a time dimension. A product may be safe but unacceptable after a short excursion, or unsafe after a long one. Monitor:

- Temperature at each handoff.
- Duration outside the specified range.
- Door openings and defrost cycles.
- Product temperature, not only air temperature.
- Data-logger coverage and gaps.

**HACCP** identifies critical control points (CCPs) and critical limits. Analytics supports HACCP by monitoring CCP data, detecting drift, and linking excursions to batches. It does not replace the food-safety plan or the qualified person who owns it.

**Excursion disposition** is a decision: release, quarantine, test, downgrade, or destroy. The rule must be agreed with quality and, where required, the customer or regulator. An algorithm that automatically releases product after an excursion is a safety risk.

### 4.7 Traceability

Food traceability is the ability to follow a lot forward and backward through the chain:

- **One-up, one-down:** each handler knows its immediate supplier and customer.
- **Lot-level:** a batch code links raw material, production, and distribution.
- **Key data elements (KDEs):** the fields regulators require for high-risk foods.
- **Critical tracking events (CTEs):** growing, receiving, transforming, shipping.

GS1 standards (GTIN, lot/batch, SSCC, EPCIS) provide the identifiers and event vocabulary. Regulation differs by jurisdiction; verify the current scope and compliance date rather than relying on a summary.

### 4.8 Food waste and the hierarchy

| Priority | Action | Analytics question |
|---|---|---|
| 1 | Prevent | Why was it ordered or produced? |
| 2 | Reuse / donate | What is safe and legal to donate? |
| 3 | Recycle / animal feed | What is the permitted route? |
| 4 | Recover energy | What volume and calorific value? |
| 5 | Dispose | What is the residual and its cost? |

Measure waste by **cause**: forecast error, over-ordering, cold-chain failure, date-code rejection, damage, or customer behaviour. A single waste percentage hides the fact that different causes have different fixes.

### 4.9 Food service and quick commerce

Food service adds a production schedule and a menu. Quick commerce adds a 10–30 minute promise and a dark-store assortment. Both shorten the quality clock:

- **Food service:** prep forecasting by daypart, ingredient yield, menu engineering, allergen control, and waste from over-production.
- **Quick commerce:** small-basket, high-frequency demand; limited shelf space; FEFO in a dark store; substitution and partial fulfilment; waste from low-turnover SKUs.

In both, the freshness promise is part of the product. A quick-commerce operator that delivers expiring yogurt at full price is creating returns and churn, not just a one-off sale.

### 4.10 Worked example — yogurt with a 21-day life

A dairy orders weekly. Unit cost is €0.60, regular price €1.20, and short-dated stock is marked down to €0.36. The overage cost is €0.24 and the underage cost is €0.60, so the critical fractile is:

\[
\frac{0.60}{0.60 + 0.24} = 0.714
\]

Weekly demand is approximately normal with mean 3,300 and standard deviation 400. The order quantity is:

\[
3{,}300 + 0.565 \times 400 \approx 3{,}526
\]

| Policy | Order | Expected waste | Expected stockout | Contribution |
|---|---:|---:|---:|---:|
| Current | 3,500 | 200 units | Low | €1,932 |
| Newsvendor | 3,526 | 210 units | Slightly lower | €1,940 |
| Freshness-priced | 3,526 | 120 units | Low | €1,986 |

The third option uses markdown earlier, converting some waste into lower-margin sales. The lesson is that the order quantity is only one lever; the **markdown timing** and **remaining shelf life at receipt** change the economics more than a small forecast improvement.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Sales | Item × store × day | Demand, censoring |
| Inventory | Batch × location × day | Remaining shelf life |
| Batch / lot | Batch | Traceability and expiry |
| Temperature logs | Logger × time | Cold-chain integrity |
| Supplier receipts | Delivery × batch | Shelf life at receipt |
| Harvest / yield | Field × variety × season | Upstream planning |
| Weather | Location × day | Demand and yield |
| Waste | Store × item × reason | Cause analysis |
| Recalls | Batch × market | Traceability test |

### 5.2 Standards and regulation

- **Codex Alimentarius**, General Principles of Food Hygiene (CXC 1-1969) and HACCP.
- **EU General Food Law** (Regulation (EC) No 178/2002) and related hygiene regulations.
- **US FSMA** preventive controls and traceability rules; verify the current compliance date and food list.
- **GS1** GTIN, lot/batch, SSCC, and EPCIS 2.0.
- **ISO 22000** food safety management systems.
- **Cold-chain guidance** from the relevant competent authority and customer specification.

### 5.3 Workflow

1. Receive with a batch code and remaining shelf life.
2. Put away under the correct temperature regime.
3. Rotate by FEFO.
4. Forecast demand at the decision grain.
5. Order with shelf-life and salvage economics.
6. Monitor cold chain and CCPs.
7. Mark down or divert short-dated stock.
8. Record waste by cause.
9. Trace forward and backward when a recall or incident occurs.

## 6. Industry case studies

### 6.1 Chipotle — food safety and traceability

**Context:** A series of foodborne-illness outbreaks in 2015 affected a fast-casual chain known for fresh ingredients and decentralized prep.
**Intervention:** Supplier testing, changes to food preparation and supply, and a food-safety programme.
**Evidence:** Public health investigations and company disclosures (grade B for the outbreak; grade C for the remediation).
**Outcome:** The events showed that fresh, complex supply chains need traceability and supplier controls, not only speed and cost.
**What transferred:** traceability is an operational capability that must be tested before an incident.
**What did not:** the specific remediation, which is company-specific and not a template for another chain.

### 6.2 Tesco — food waste reporting

**Context:** A large grocery retailer with a public food-waste commitment.
**Intervention:** Supplier and store waste measurement, donation programmes, and customer-facing reporting.
**Evidence:** Company sustainability reports (grade C); figures are company claims.
**Outcome:** Measurement made waste visible by category and store, which changed ordering and markdown practice.
**What transferred:** waste by cause is more actionable than a single percentage.
**What did not:** the reported numbers depend on definition and boundary choices.

### 6.3 Zespri — kiwifruit global supply

**Context:** A single-desk exporter selling perishable fruit into distant markets.
**Intervention:** Coordinated harvest, cold-chain, shipping, and market allocation.
**Evidence:** Company and industry disclosures (grade C).
**Outcome:** Quality at arrival depends on the whole chain, not only on the ship.
**What transferred:** for perishables, the quality clock and the transport plan are one system.
**What did not:** the single-desk structure and cultivar-specific physiology.

### 6.4 A composite failure — the promotion that spoiled

**Context:** A retailer ran a deep promotion on fresh berries without informing the DC or adjusting shelf-life allocation.
**Intervention:** None; the promotion was planned by marketing alone.
**Evidence:** Composite teaching case.
**Outcome:** The DC over-ordered, stores received short-dated stock, and waste rose in the week after the promotion.
**What transferred:** promotions must be integrated with perishable ordering and shelf-life allocation.
**What did not:** the specific volumes, which depend on the category and store.

## 7. Labs

### Lab 32.1 — Remaining shelf life and FEFO

Build a batch-level inventory model. Simulate a week of receipts and sales with FEFO. Compare waste and service under FIFO and FEFO.

**Deliverable:** Simulation, waste comparison, and policy note.

### Lab 32.2 — Newsvendor with salvage

Estimate the critical fractile for three products with different margins and salvage values. Compare the recommended order with the current policy.

**Deliverable:** Order model and sensitivity table.

### Lab 32.3 — Cold-chain excursion

Given temperature logger data, identify excursions, link them to batches, and apply a disposition rule agreed with quality. Report the volume and value affected.

**Deliverable:** Excursion report and disposition log.

### Lab 32.4 — Traceability drill

Take a synthetic lot and trace it forward to stores and backward to suppliers. Measure the time and completeness of the trace.

**Deliverable:** Trace report and gap list.

### Lab 32.5 — Harvest and contract plan

Build a yield distribution for a crop and compare fixed-volume, revenue-sharing, and spot purchase under three weather scenarios.

**Deliverable:** Contract comparison and risk memo.

## 8. Assignment / project

Design a perishable supply chain for one category across a region.

**Required components:**

1. Product regime and shelf-life profile.
2. Demand forecast with weather and promotion features.
3. Ordering policy with salvage and markdown.
4. Cold-chain monitoring and excursion rules.
5. Lot-level traceability design.
6. Waste measurement by cause.
7. Supplier and contract structure.
8. Freshness promise and service metrics.
9. Risks, regulation, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Shelf-life logic | 20% | Uses expiry only | Remaining life at receipt | Age-based value and FEFO |
| Demand model | 20% | Average demand | Seasonality and promotions | Weather, substitution, censoring |
| Ordering | 20% | Fixed order | Newsvendor | Salvage, markdown timing, service |
| Cold chain | 15% | Describes process | Monitors temperature | Excursion disposition and CCP linkage |
| Traceability | 15% | Lot number only | One-up, one-down | Tested forward and backward |
| Waste and communication | 10% | Total waste | Waste by cause | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Food safety:** HACCP and preventive controls are legal requirements; analytics supports but does not replace them.
- **Traceability:** high-risk foods may require lot-level records within a defined time; verify the current rule.
- **Date labelling:** "best before" and "use by" have different meanings and legal implications.
- **Donation:** liability, temperature, and date rules vary; do not donate unsafe food.
- **Labour:** harvest and food processing rely on seasonal and migrant labour; scheduling must respect legal and ethical standards.
- **Environment:** food waste, packaging, and cold-chain emissions are material; measure them rather than claiming them.
- **Data:** supplier and farm data may be commercially sensitive; share only what the trace requires.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| SKU-level shelf life | Expiry ignored in allocation | Batch treated as fungible | Waste and short-dated stock | Batch-level inventory |
| FIFO instead of FEFO | Old stock remains | Rotation rule wrong | Expiry write-off | WMS enforcement |
| Forecast ignores substitution | Substitute overstock | Demand displaced | Waste | Substitution model |
| Promotion without supply plan | Waste spikes after promo | Marketing and supply disconnected | Cost and availability | Integrated promotion planning |
| Cold chain gaps | Logger coverage holes | Handoff not monitored | Safety risk | End-to-end monitoring |
| Excursion auto-release | No quality sign-off | Rule too permissive | Safety and legal risk | Agreed disposition |
| Traceability untested | Trace takes days | Records incomplete | Recall scope too wide | Regular drills |
| Waste not measured by cause | One aggregate number | Causes invisible | Wrong fixes | Cause coding |
| Over-tight shelf-life receipt | Supplier ships old stock | Contract not enforced | Downstream waste | Receipt specification |
| Donation as waste dump | Unsafe or unsaleable product | Liability | Reputational harm | Eligibility rules |

## 12. Interview and discussion questions

1. Why is remaining shelf life more important than expiry date? *(Hint: it determines usable selling time.)*
2. What is FEFO, and why does FIFO fail for perishables? *(Hint: value declines with age.)*
3. How do you set an order quantity when unsold stock can be marked down? *(Hint: critical fractile with salvage.)*
4. What makes food demand different from fashion demand? *(Hint: perishability, safety, weather, substitution.)*
5. How would you measure food waste by cause? *(Hint: reason codes at store and DC.)*
6. What is a critical control point? *(Hint: a step where a hazard can be prevented, eliminated, or reduced.)*
7. How do you trace a lot forward and backward? *(Hint: lot codes, KDEs, CTEs, one-up-one-down.)*
8. What should happen after a cold-chain excursion? *(Hint: quarantine, assess, consult quality, dispose or release.)*
9. How does quick commerce change perishable planning? *(Hint: shorter promise, smaller baskets, dark-store space.)*
10. Why is a promotion a supply chain event, not only a marketing event? *(Hint: it changes volume, mix, and shelf-life allocation.)*
11. What is the food-waste hierarchy? *(Hint: prevent, reuse, recycle, recover, dispose.)*
12. Which metric would you show a store manager every morning? *(Hint: short-dated stock and waste risk, not total sales.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Shelf life | Time a product remains saleable under stated conditions |
| Remaining shelf life | Shelf life left at a point in the chain |
| FEFO | First-expired, first-out |
| Batch / lot | Identifiable production or harvest quantity |
| Cold chain | Temperature-controlled chain from production to consumption |
| HACCP | Hazard analysis and critical control points |
| CCP | Critical control point |
| Excursion | Temperature outside the specified range |
| Mean kinetic temperature | Time-temperature summary used in stability assessment |
| Traceability | Ability to follow a lot forward and backward |
| KDE | Key data element required for traceability |
| CTE | Critical tracking event |
| Salvage value | Recovery from unsold or short-dated stock |
| Markdown | Price reduction to clear stock |
| Food-waste hierarchy | Priority order for prevention, reuse, recycling, recovery, disposal |
| One-up, one-down | Knowing immediate supplier and customer for each lot |
| Substitution | Customer accepts a different item when the preferred one is unavailable |
| Date code | Printed date indicating quality or safety limit |

## 14. Sources and further reading

### Standards and regulation

- Codex Alimentarius, General Principles of Food Hygiene (CXC 1-1969).
- EU Regulation (EC) No 178/2002, general food law. https://eur-lex.europa.eu/
- US FDA, Food Safety Modernization Act and traceability rules. https://www.fda.gov/food
- ISO 22000, Food safety management systems.
- GS1, EPCIS 2.0 and lot/batch identification. https://www.gs1.org/standards/epcis
- FAO, food loss and waste measurement guidance. https://www.fao.org/

### Peer-reviewed and professional

- Chopra, S., Meindl, P., *Supply Chain Management*.
- Van der Vorst, J. G. A. J., et al., research on food supply chain management.
- Aiello, G., et al., "Food supply chain analytics," research on traceability and cold chain.
- Silver, E. A., Pyke, D. F., Thomas, D. J., *Inventory and Production Management in Supply Chains*.

### Industry

- Tesco sustainability and food-waste reporting. https://www.tesco.com/
- Zespri annual reports and industry disclosures. https://www.zespri.com/
- Chipotle public food-safety disclosures and public health reports.
- FAO and World Bank food loss and waste data.

> **Next:** [Module 33](33_healthcare_provider_life_sciences.md) moves from perishable food to perishable clinical supply, where a stockout can affect a patient rather than a sale.
