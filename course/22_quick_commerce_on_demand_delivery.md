# Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (14 lecture, 10 lab, 6 project)
> **Track fit:** Retail & E-commerce track; open to all
> **Prerequisites:** Modules 07, 09, 10, 12, 21
> **Tools:** Python, OR-Tools, SimPy, geospatial data, a public courier or order dataset
> **Industry question:** "What does it take to promise delivery in minutes, profitably, without burning out the people who deliver?"

---

## 1. Why industry cares

Quick commerce and food delivery compress every supply chain problem into a small geography and a short clock:

- A dark store carries 2,000–10,000 SKUs for a 1–3 km radius.
- A restaurant has a menu, a kitchen, and a 30-minute promise.
- A courier platform must match supply and demand in real time.
- A 15-minute promise leaves no room for a stockout, a wrong turn, or a slow pick.

The economics are brutal: low basket values, high delivery cost per order, perishable inventory, and a workforce that must be available exactly when demand spikes. Analytics is not a nice-to-have; it is the operating system.

## 2. Learning outcomes

By the end of this module, you can:

- Map a quick-commerce or food-delivery operating model and its unit economics.
- Forecast demand at short intervals and small geography.
- Design assortment and replenishment for dark stores.
- Model courier supply, demand, and matching.
- Predict ETA and explain its uncertainty.
- Optimize batching, routing, and dispatch under time windows.
- Analyze food waste and perishable inventory.
- Design incentives that balance service, cost, and courier welfare.
- Evaluate the regulatory and labor context.
- Build a store-level P&L and a service trade-off.

## 3. Mental model

```text
   DEMAND (15-min slots, 1-3 km)
        │
        v
   ASSORTMENT & STOCK ──> PICK ──> PACK ──> DISPATCH
        │                                      │
        v                                      v
   REPLENISHMENT <── STORE ──> COURIER POOL ──> ROUTE
        │                                      │
        v                                      v
   WASTE / EXPIRY <──────────────────── DELIVER / RETURN
```

The system has two coupled queues: orders waiting for couriers, and couriers waiting for orders. Both must be managed; optimizing one alone makes the other worse.

### 3.1 Promise tiers

| Model | Typical promise | Inventory | Cost driver |
|---|---|---|---|
| 10–15 minute | Minutes | Dark store | Density, picking |
| 30-minute | 30 min | Dark store or store | Couriers, batching |
| Same-hour | 1 hour | Store or micro-fulfillment | Dispatch |
| Food delivery | 25–45 min | Restaurant | Kitchen, couriers |
| Scheduled grocery | 2 h–2 days | DC or store | Slot capacity |

## 4. Core concepts

### 4.1 Demand forecasting at short horizons

Forecast at **15-minute to 1-hour buckets**, by store and category.

| Feature | Why |
|---|---|
| Time of day | Lunch/dinner peaks |
| Day of week | Weekend patterns |
| Weather | Rain and heat change demand |
| Payday | Cash-cycle effects |
| Promotions | Spikes |
| Holidays | Local and national |
| Store age | Ramp curve |
| Competitor activity | Local shifts |

Short-horizon demand is noisy. Use hierarchical pooling: forecast the store total, then split by category share, then by SKU. Use quantiles for staffing and stock.

### 4.2 Assortment for dark stores

A dark store cannot carry everything. Choose SKUs by:

- Local demand and basket affinity.
- Margin and shelf life.
- Pick efficiency (location, weight).
- Substitution availability.
- Supplier reliability.

Assortment is a constrained optimization:

\[
\max \sum_i \pi_i x_i \quad \text{s.t.} \sum_i s_i x_i \le S,\; \text{service constraints}
\]

where \(\pi_i\) is expected contribution, \(s_i\) space, and \(x_i\) inclusion.

### 4.3 Replenishment and waste

Perishables decay. The newsvendor with salvage:

\[
Q^* = F^{-1}\left(\frac{p - c}{p - s}\right)
\]

where \(s\) is salvage value. For short shelf life, order multiple times per day and use a **sell-by-aware** policy.

Waste drivers:

| Driver | Lever |
|---|---|
| Over-ordering | Better forecast, smaller batches |
| Slow SKUs | Assortment pruning |
| Poor rotation | FEFO discipline |
| Temperature abuse | Cold-chain monitoring |
| Promotion mismatch | Demand shaping |

### 4.4 Courier supply and demand

The platform controls price, incentives, and matching, but not the courier's schedule.

| Lever | Effect | Risk |
|---|---|---|
| Surge pricing | Attracts supply | Customer backlash |
| Bonuses | Supply in peak | Cost |
| Shift scheduling | Predictable supply | Rigidity |
| Batching | Lower cost per order | Longer delivery |
| Delivery radius | Faster | Smaller market |
| Courier tiers | Priority access | Fairness |

Model the courier pool as a queue with endogenous supply: higher pay attracts more couriers, but with a delay and a reservation wage.

### 4.5 ETA prediction

ETA = pick time + wait for courier + travel time + handoff.

| Component | Model |
|---|---|
| Pick time | Items, store layout, staffing |
| Courier wait | Queueing and dispatch |
| Travel | Road network, mode, traffic |
| Handoff | Building access, elevator |

Use quantile regression or gradient boosting with route features. Report an interval, not a point. An ETA that is wrong 30% of the time destroys trust.

### 4.6 Batching and routing

Batching two orders reduces cost per delivery but increases lateness.

\[
\text{Cost} = c_{\text{courier}} + \lambda \cdot \text{lateness} + \mu \cdot \text{waste}
\]

Constraints: capacity, hot/cold separation, time windows, food safety.

OR-Tools can solve small instances; production uses heuristics and rolling re-optimization.

### 4.7 Dispatch and matching

| Policy | Rule | Trade-off |
|---|---|---|
| Nearest | Closest courier | Ignores future demand |
| Greedy cost | Lowest immediate cost | Myopic |
| Batch | Group orders | Lateness |
| Predictive | Anticipate demand | Complexity |
| Priority | Tiered customers | Fairness |

Matching is a two-sided market. Measure courier utilization, order lateness, and earnings per hour together.

### 4.8 Unit economics

| Line | Example (€) |
|---|---|
| Basket revenue | 24.00 |
| COGS | 15.00 |
| Pick and pack | 1.80 |
| Courier cost | 4.20 |
| Payment and platform | 0.70 |
| Refunds and waste | 1.10 |
| Contribution | 1.20 |

A 5% contribution margin leaves no room for error. Small changes in basket size, courier cost, or waste decide profitability.

### 4.9 Food safety and cold chain

- Temperature control from kitchen/store to door.
- Allergen and cross-contamination controls.
- Time-temperature abuse during batching.
- Traceability of ingredients.
- Hygiene during handoff.

Analytics can monitor sensor data and flag excursions, but food safety is a legal and operational control, not a model output.

### 4.10 Courier welfare and labor

| Issue | Analytics role |
|---|---|
| Earnings volatility | Transparent pay models |
| Safety | Route and speed monitoring |
| Algorithmic management | Appeal and human review |
| Insurance | Coverage tracking |
| Worker classification | Legal review |

Efficiency metrics that reward unsafe speed are a design failure. Include safety constraints in dispatch.

### 4.11 Dispatch policy comparison, worked

Twenty orders arrive in a 30-minute window; eight couriers are available. Each order has a ready time, a drop point, and a 30-minute promise.

| Policy | Mean lateness | On-time | Courier cost | Notes |
|---|---:|---:|---:|---|
| Nearest courier | 6.4 min | 82% | €31 | Ignores the next wave |
| Greedy lowest cost | 5.1 min | 87% | €28 | Myopic but decent at this scale |
| Batch up to two | 8.9 min | 74% | €21 | Cheap, but food quality suffers |
| Predictive re-optimization | 4.2 min | 91% | €27 | Needs a demand forecast and a rolling solve |

The comparison only means something if all four are run on the same arrival and courier-supply trace. A policy that looks better because it received a quieter day is not better. Use common random numbers and report the distribution of lateness, not only the mean.

### 4.12 Dark-store break-even, worked

| Line | Per order |
|---|---:|
| Basket revenue | €24.00 |
| COGS (75% of revenue) | €18.00 |
| Pick and pack | €1.60 |
| Courier cost | €4.00 |
| Payment and platform | €0.60 |
| Refunds and waste | €0.90 |
| Contribution | −€1.10 |

At this basket and cost, every order loses money. Break-even requires one of: basket up by about €1.50 at constant COGS ratio, courier cost down by about €1.10 through batching, or waste and refunds halved. Fixed store cost (rent, labour, equipment) sits on top, so the store also needs a minimum order volume. The model should show the break-even orders per day and the sensitivity of each lever, not a single contribution number.

### 4.13 Reference event model

| Table | Grain | Key fields | Use |
|---|---|---|---|
| order | order | store_id, slot, promise_at, basket_value | Demand and economics |
| order_line | order × item | sku, qty, substituted_from | Assortment and substitution |
| pick_task | order × task | start, end, picker_id | Pick-time model |
| courier_ping | courier × time | location, status, order_id | Supply and dispatch |
| delivery_event | order × event | picked_up, delivered, lateness | ETA and service |
| waste_log | store × sku × day | reason, quantity, value | Waste model |

The critical join is `order.promise_at` to `delivery_event.delivered_at`. Without it, lateness is a guess; with it, every promise can be scored and every dispatch policy can be compared.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Orders | Order | Demand, economics |
| Store inventory | SKU × store | Replenishment |
| Courier locations | Ping | Dispatch |
| Delivery events | Event | ETA, lateness |
| Weather | Hour, location | Demand |
| Menu/catalog | Item | Assortment |
| Waste logs | Item, day | Perishables |
| Ratings | Order | Experience |

### 5.2 Standards

- GS1 GTIN for products.
- ISO 22000 / HACCP for food safety.
- Local gig-economy and employment rules.
- Consumer distance-selling and refund rules.
- Privacy rules for courier location data.

### 5.3 Workflow

1. Forecast demand by slot and store.
2. Replenish to shelf-life-aware targets.
3. Publish assortment and promise.
4. Match and dispatch orders.
5. Predict and communicate ETA.
6. Measure lateness, waste, and courier utilization.
7. Replan intraday.

## 6. Industry case studies

### 6.1 Domino's — delivery promise and technology

**Context:** Pizza delivery competes on speed and reliability.
**Intervention:** Order tracking, route optimization, and a digital ordering stack.
**Evidence:** Company disclosures describe the technology and delivery performance; specific gains are company claims.
**Outcome:** A reference for promise-driven operations.
**What transferred:** the promise is a product feature backed by operations.
**What did not:** pizza has a simpler menu and production process than grocery.

### 6.2 Instacart — marketplace fulfillment

**Context:** Grocery delivery via personal shoppers.
**Intervention:** Batch shopping, route planning, and substitution logic.
**Evidence:** Company filings and public reporting describe the model; metrics are company claims.
**Outcome:** A marketplace model with variable supply.
**What transferred:** substitution and communication are core to the experience.
**What did not:** the model depends on dense demand and shopper supply.

### 6.3 Getir / Gopuff — dark-store quick commerce

**Context:** 10–30 minute delivery from micro-fulfillment centers.
**Intervention:** Dense dark stores, small assortments, and courier fleets.
**Evidence:** Public reporting documents rapid expansion and subsequent contraction in several markets; figures are company and press reports.
**Outcome:** A test of whether speed can be profitable at scale.
**What transferred:** density and basket size determine viability.
**What did not:** the assumption that rapid expansion creates a durable moat.

### 6.4 Meituan — on-demand delivery at scale

**Context:** Food delivery and local services in China.
**Intervention:** Dispatch algorithms, batching, and a large courier network.
**Evidence:** Company disclosures and academic studies describe the dispatch system; exact performance is not public.
**Outcome:** A large-scale matching system.
**What transferred:** dispatch is a marketplace design problem.
**What did not:** regulatory and labor context differs by country.

## 7. Labs

### Lab 22.1 — Slot-level demand forecast

Forecast 15-minute demand for 20 stores over four weeks. Compare pooled and store-level models. Report pinball loss.

**Deliverable:** Forecast, metrics, and peak analysis.

### Lab 22.2 — Dark-store assortment

Choose 300 SKUs from 2,000 for a dark store under space and service constraints. Compare to the current assortment.

**Deliverable:** Optimization model and assortment plan.

### Lab 22.3 — Courier dispatch simulation

Simulate orders and couriers over a day. Compare nearest, greedy, and batching policies on lateness, cost, and utilization.

**Deliverable:** SimPy model and policy comparison.

### Lab 22.4 — ETA model

Build an ETA model with pick, wait, and travel components. Report MAE and interval coverage by distance band.

**Deliverable:** Model, calibration plot, and error analysis.

### Lab 22.5 — Store P&L

Build a unit-economics model for a dark store. Identify the break-even orders per day and the sensitivity to basket size.

**Deliverable:** P&L model and memo.

## 8. Assignment / project

Design a quick-commerce operation for one city or neighborhood.

**Required components:**

1. Demand estimate by slot.
2. Assortment and store size.
3. Replenishment and waste plan.
4. Courier supply and dispatch model.
5. ETA and promise design.
6. Unit economics and break-even.
7. Labor and safety plan.
8. Regulatory triage.
9. Measurement plan.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Demand model | 20% | Daily total | Slot and store | Pooled hierarchy and quantiles |
| Assortment | 15% | Intuitive list | Constrained selection | Basket and space trade-offs |
| Dispatch | 20% | Nearest rule | Simulation comparison | Market and welfare effects |
| ETA | 15% | Average time | Component model | Calibrated intervals |
| Economics | 20% | Revenue only | Contribution | Break-even and sensitivity |
| Safety and labor | 10% | Mentions rules | Controls identified | Design constraints and appeal |

## 10. Regulatory, ethical, and commercial lens

- **Worker classification:** employee versus contractor changes cost and obligations.
- **Working time and safety:** delivery targets must not incentivize unsafe riding or driving.
- **Food safety:** HACCP and temperature controls are legal requirements.
- **Consumer protection:** refunds, substitutions, and promise accuracy.
- **Privacy:** courier location and customer address data are sensitive.
- **Competition:** local market power and exclusionary practices may be scrutinized.
- **Waste:** food waste has environmental and social costs.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Promise inflation | Marketing leads ops | No feasibility check | Late orders | Capacity-aware promise |
| Courier oversupply | Couriers idle | Incentives too strong | Cost | Dynamic incentives |
| Courier undersupply | Orders wait | Poor forecasting | Lateness | Supply planning |
| Batching greed | Late orders | Cost-only objective | Churn | Lateness penalty |
| Assortment sprawl | Slow SKUs | No pruning | Waste | Space optimization |
| ETA optimism | Point estimate | No uncertainty | Broken trust | Quantile ETA |
| Waste blindness | No waste data | Over-ordering | Margin loss | FEFO and tracking |
| Safety trade-off | Speed incentives | Misaligned metrics | Harm | Safety constraints |
| Dark-store density | Low orders/km² | Expansion without demand | Losses | Density threshold |
| Substitution failure | Refund spike | Poor alternatives | Experience | Substitution logic |

## 12. Interview and discussion questions

1. What makes quick commerce different from e-commerce? *(Hint: time, density, perishables.)*
2. How do you forecast 15-minute demand? *(Hint: pool, slot, weather, quantiles.)*
3. How do you choose dark-store assortment? *(Hint: space, margin, basket, shelf life.)*
4. What is the courier supply-demand problem? *(Hint: two-sided queue, incentives.)*
5. How do you predict ETA? *(Hint: pick, wait, travel, handoff; intervals.)*
6. When is batching worth it? *(Hint: cost versus lateness and food quality.)*
7. How do you reduce food waste? *(Hint: forecast, FEFO, smaller batches, pruning.)*
8. What is the break-even order volume? *(Hint: contribution margin and fixed cost.)*
9. How do you protect courier safety? *(Hint: constraints, monitoring, appeal.)*
10. What promise would you make in a low-density city? *(Hint: longer promise or no service.)*
11. How do you handle substitutions? *(Hint: rules, customer choice, refunds.)*
12. What regulation matters most? *(Hint: labor classification and food safety.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Quick commerce | Delivery in minutes from dense micro-fulfillment |
| Dark store | Retail facility closed to customers |
| Micro-fulfillment | Small automated or manual fulfillment site |
| Slot | A short demand interval |
| Batching | Combining orders for one courier |
| Dispatch | Assigning orders to couriers |
| ETA | Estimated time of arrival |
| Surge pricing | Price increase to attract supply |
| Substitution | Replacing an unavailable item |
| FEFO | First expired, first out |
| Waste rate | Unsold or spoiled share |
| Courier utilization | Share of time on active deliveries |
| Contribution margin | Revenue minus variable cost |
| Density | Orders or drop points per area |
| Promise | Delivery time commitment |
| Handoff | Transfer from courier to customer |
| Two-sided market | Platform matching supply and demand |
| Reservation wage | Minimum pay to accept work |
| Cold chain | Temperature-controlled flow |
| HACCP | Food safety management system |

## 14. Sources and further reading

### Standards and regulation

- ISO 22000, Food safety management. https://www.iso.org/standard/65472.html
- Codex Alimentarius, HACCP. https://www.fao.org/fao-who-codexalimentarius/
- EU consumer protection rules. https://eur-lex.europa.eu/

### Peer-reviewed and professional

- Chopra, S., Meindl, P., *Supply Chain Management*.
- Simchi-Levi, D. et al., *Designing and Managing the Supply Chain*.
- Powell, W. B., *Approximate Dynamic Programming* (matching and dispatch).
- Boysen, N. et al., "Last-mile delivery concepts," *OR Spectrum*.
- Agatz, N. et al., "Challenges and opportunities in attended home delivery."

### Industry

- Domino's investor materials. https://ir.dominos.com/
- Instacart investor filings. https://investors.instacart.com/
- Getir and Gopuff public reporting (label press claims).
- Meituan annual reports. https://www.meituan.com/en-US/investor

### Tools

- OR-Tools routing.
- SimPy.
- OSRM or a routing API.
- GeoPandas for density analysis.

> **Next:** [Module 23](23_omnichannel_fulfillment_returns.md) connects the digital promise to stores, returns, and the customer experience.
