# Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 28 hours (12 lecture, 10 lab, 6 project)
> **Track fit:** Retail & E-commerce track; open to all
> **Prerequisites:** Modules 09, 10, 13, 21
> **Tools:** Python, DuckDB, OR-Tools, store and order data
> **Industry question:** "When a customer can buy anywhere and return anywhere, where should the order come from?"

---

## 1. Why industry cares

A customer does not see a channel. They see a product, a promise, and a return option. The retailer sees a network of stores, DCs, marketplaces, and carriers that must behave as one system.

Omnichannel is the set of decisions that make that possible:

- **Where does the order ship from?** DC, store, 3PL, or supplier.
- **Can the customer pick up in store?** And when?
- **Can they return anywhere?** And what happens to the unit?
- **Is inventory visible across nodes?** In real time?
- **Is the promise accurate?** Or does it change after checkout?

The analytics problem is **inventory visibility plus order routing plus reverse logistics**, measured on customer promise, cost, and margin.

## 2. Learning outcomes

By the end of this module, you can:

- Map an omnichannel network and its order-routing choices.
- Design inventory visibility and availability logic across nodes.
- Evaluate ship-from-store, BOPIS, curbside, and endless aisle.
- Build a distributed order management (DOM) routing rule.
- Measure promise accuracy and its effect on conversion and contacts.
- Model returns networks and disposition economics.
- Detect and mitigate return fraud.
- Evaluate the sustainability of returns and packaging.
- Design a customer promise that operations can keep.
- Build a channel-level P&L.

## 3. Mental model

```text
   DEMAND (web, app, store, marketplace)
        │
        v
   AVAILABILITY ──> PROMISE ──> ROUTING ──> FULFILL ──> DELIVER
        │              │           │           │           │
   inventory       date/cost    node choice  pick/pack   carrier
   visibility                   rules                     or pickup
        │              │           │           │           │
        └──────────────┴───────────┴───────────┴───────────┘
                              RETURNS
                                 │
                     receive → inspect → disposition
```

The promise is made at checkout, but it is determined by inventory visibility and routing rules upstream. Most broken promises are designed before the customer clicks.

### 3.1 Fulfillment options

| Option | Customer | Cost | Inventory risk |
|---|---|---|---|
| Ship from DC | Delivery | Low–medium | Low |
| Ship from store | Delivery | Medium | Store stock |
| BOPIS | Pickup | Low | Store stock |
| Curbside | Pickup | Low–medium | Store stock |
| Endless aisle | Order in store | Medium | Network stock |
| Drop-ship | Delivery | Low holding | Supplier stock |
| Locker | Pickup | Low | Carrier |

## 4. Core concepts

### 4.1 Inventory visibility

Availability logic must answer:

- Is the item in stock physically?
- Is it reserved for another order?
- Is it in a pickable location?
- Is it promised to a future order?
- Is it damaged or expired?

A common failure is **phantom inventory**: the system says yes, the shelf says no. Cycle counting and computer vision reduce it; neither eliminates it.

### 4.2 Available-to-promise (ATP)

\[
\text{ATP} = \text{on hand} + \text{inbound} - \text{reserved} - \text{safety stock}
\]

For omnichannel, ATP must be calculated per node and per promise horizon. A store may have the item but not the labor to pick it within the promise.

### 4.3 Order routing

Routing rules:

| Rule | Objective | Risk |
|---|---|---|
| Nearest node | Speed | Splits orders |
| Lowest cost | Margin | Slower |
| Fewest splits | Experience | Higher cost |
| Store priority | Clear store stock | Store disruption |
| Capacity-aware | Smooth load | Complexity |
| Margin-aware | Protect contribution | Customer experience |

A production rule is usually a weighted score with hard constraints:

\[
\text{score} = w_1 \cdot \text{ship cost} + w_2 \cdot \text{lateness} + w_3 \cdot \text{splits} + w_4 \cdot \text{store load}
\]

### 4.4 Ship-from-store economics

Store fulfillment can be fast and reduce markdowns, but it consumes store labor and space.

| Benefit | Cost |
|---|---|
| Faster delivery | Picking disrupts store tasks |
| Liquidates slow stock | Higher pick error |
| Avoids DC capacity | Packaging and carrier pickup |
| Improves availability | Inventory accuracy issues |

Measure store-level contribution, not only network cost.

### 4.5 BOPIS and curbside

BOPIS reduces delivery cost and increases conversion, but creates new processes:

- Pick and stage.
- Customer notification.
- Hold time and expiry.
- Substitution.
- Curbside check-in.
- Returns at pickup.

Measure **ready-on-time**, **wait time**, and **abandonment**, not only order volume.

### 4.6 Promise accuracy

\[
\text{Promise accuracy} = \frac{\text{orders delivered by the promised date}}{\text{orders promised}}
\]

Track by channel, node, carrier, and product. A promise that is accurate but slow may be better than a fast promise that fails.

### 4.7 Returns networks

| Return path | Cost | Customer effort |
|---|---|---|
| Mail | Medium | Medium |
| Drop-off | Low | Medium |
| Store | Low–medium | Low |
| Curbside | Low | Low |
| Pickup at home | High | Lowest |

Reverse logistics costs include transport, inspection, restocking, refurbishment, and disposal. The cheapest return is the one that never happens, but not all returns are avoidable.

### 4.8 Disposition economics

| Disposition | Condition | Recovery |
|---|---|---|
| Restock | New, unopened | High |
| Refurbish | Minor damage | Medium |
| Resell | Used | Low–medium |
| Liquidate | Excess | Low |
| Donate | Usable | Tax/social value |
| Recycle | Material | Low |
| Destroy | Unsafe | Negative |

Route by expected recovery minus processing cost, not by habit.

### 4.9 Return fraud

| Pattern | Signal |
|---|---|
| Wardrobing | Wear and return |
| Receipt fraud | Mismatched purchase |
| Empty box | Weight discrepancy |
| Serial returns | High frequency |
| Switch | Different item returned |

Controls: reason codes, weight checks, serial numbers, policy limits, and human review. Avoid blanket suspicion of legitimate customers.

### 4.10 Channel P&L

| Line | Web | Store | Marketplace |
|---|---|---|---|
| Revenue | Net | Net | Gross |
| COGS | Product | Product | Product |
| Fulfillment | Pick, ship | Store labor | Commission |
| Returns | Reverse | Store | Marketplace |
| Marketing | CAC | Local | Platform |
| Contribution | | | |

Channel P&L prevents cross-subsidy blindness.

### 4.11 Routing worked example

One order for three units, each available at three nodes:

| Node | Units available | Ship cost | Promise | Store load | Split risk |
|---|---|---:|---|---:|---|
| DC-A | 2 | €6.40 | 2 days | Low | 1 split |
| Store-17 | 3 | €8.10 | Next day | High | None |
| Store-42 | 1 | €5.60 | 3 days | Medium | 2 splits |

A nearest-node rule chooses Store-17 because it can ship all three units. A lowest-cost rule chooses DC-A plus Store-42, producing two parcels and a split. A score-based rule with a split penalty may still choose Store-17 because the extra €1.70 buys one fewer parcel and a faster promise.

The lesson: the routing decision is not a cost comparison. It is a constrained choice among service, cost, store labour, and customer experience. The weights must be set by the business, documented, and tested against outcomes.

### 4.12 Disposition worked example

A returned coat costs €12 to inspect and process.

| Disposition | Recovery | Processing | Net | Condition |
|---|---:|---:|---:|---|
| Restock | €70 | €4 | €66 | New, tags attached |
| Refurbish | €45 | €12 | €33 | Minor damage |
| Resell | €30 | €12 | €18 | Used, good |
| Liquidate | €12 | €2 | €10 | Excess |
| Donate | €0 | €2 | −€2 | Usable, no demand |
| Destroy | €0 | €5 | −€5 | Unsafe |

The rule is not "restock if possible". It is "choose the highest net recovery that the condition and the law allow." A unit that can be restocked but has no demand may recover more through resale than through a restock that sits in inventory.

### 4.13 Reference data model

| Table | Grain | Key fields | Use |
|---|---|---|---|
| inventory_position | sku × node × time | on_hand, reserved, inbound, damaged | ATP |
| order_line | order × line | node_assigned, promise_at, split_flag | Routing and promise |
| store_capacity | store × hour | pick_minutes_available, orders_picked | Capacity-aware routing |
| return | return × line | reason, condition, disposition, recovery | Reverse logistics |
| contact | contact | order_id, reason, promise_gap | Promise failure |
| channel_cost | channel × month | fulfilment, returns, marketing | Channel P&L |

`inventory_position` must be a time series, not a current snapshot. Most promise failures are explainable only by looking at what the system believed at the moment of checkout.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Inventory | SKU × node | Availability |
| Orders | Order line | Demand and routing |
| Reservations | Order line | ATP |
| Store labor | Store, hour | Capacity |
| Shipments | Parcel | Cost and promise |
| Returns | Return | Reverse logistics |
| Customer contacts | Contact | Promise failure |
| Product master | SKU | Attributes and substitutes |

### 5.2 Standards

- GS1 GTIN and SSCC.
- EPCIS 2.0 for event visibility.
- EDI 850/855/856/810 for partner orders.
- Consumer distance-selling and returns rules.
- Packaging and waste regulations.

### 5.3 Workflow

1. Publish availability and promise.
2. Route the order.
3. Pick, pack, and stage.
4. Ship or notify for pickup.
5. Measure promise accuracy and cost.
6. Process returns and disposition.
7. Replenish and rebalance inventory.

## 6. Industry case studies

### 6.1 Walmart — store as fulfillment node

**Context:** Walmart combined stores and e-commerce.
**Intervention:** Ship-from-store, curbside pickup, and store-based fulfillment.
**Evidence:** Company disclosures describe the network and pickup; specific metrics are company claims.
**Outcome:** Stores became assets in the e-commerce network.
**What transferred:** proximity can beat a dedicated DC for speed.
**What did not:** the model depends on store density and labor capacity.

### 6.2 Target — same-day and store fulfillment

**Context:** Target used stores to fulfill digital orders.
**Intervention:** Store fulfillment, drive-up, and same-day services.
**Evidence:** Company disclosures describe growth in same-day services; figures are company claims.
**Outcome:** A store-centric omnichannel model.
**What transferred:** the store network is a fulfillment network if the process supports it.
**What did not:** not every store is a good fulfillment node.

### 6.3 Nordstrom — integrated inventory

**Context:** Nordstrom sought a single view of inventory across stores and online.
**Intervention:** Inventory visibility and ship-from-store.
**Evidence:** Public case descriptions exist; performance figures are company claims.
**Outcome:** An early omnichannel reference.
**What transferred:** visibility is the prerequisite for routing.
**What did not:** the technology alone does not fix inaccurate store inventory.

### 6.4 REI — used gear and returns

**Context:** Outdoor retail with a strong returns culture.
**Intervention:** Used-gear resale and return policies.
**Evidence:** Company disclosures describe the program.
**Outcome:** A disposition strategy that recovers value.
**What transferred:** returns can be a product line, not only a cost.
**What did not:** resale requires authentication and demand.

## 7. Labs

### Lab 23.1 — ATP and promise

Build an ATP calculation across DCs and stores. Simulate a week of orders and measure promise accuracy.

**Deliverable:** ATP model, promise accuracy, and failure analysis.

### Lab 23.2 — Order routing

Build a routing rule with cost, speed, and splits. Compare nearest, lowest-cost, and score-based rules.

**Deliverable:** Routing model and trade-off table.

### Lab 23.3 — Ship-from-store economics

Model the incremental cost and benefit of fulfilling online orders from 20 stores. Identify the break-even store volume.

**Deliverable:** Store-level P&L and recommendation.

### Lab 23.4 — Returns disposition

Given return condition and resale prices, build a disposition rule. Compare recovery and cost to the current process.

**Deliverable:** Disposition model and policy.

### Lab 23.5 — Promise failure analysis

Link late deliveries to node, carrier, product, and order characteristics. Identify the top three controllable causes.

**Deliverable:** Failure analysis and action plan.

## 8. Assignment / project

Design an omnichannel fulfillment and returns model for a retailer.

**Required components:**

1. Network and channel map.
2. Inventory visibility and ATP logic.
3. Routing rules and constraints.
4. Promise design and accuracy target.
5. Returns network and disposition.
6. Channel P&L.
7. Fraud and abuse controls.
8. Sustainability assessment.
9. Measurement plan.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Visibility and ATP | 20% | Stock count | Node-level ATP | Reservations and promise horizon |
| Routing | 20% | Nearest rule | Cost and speed trade-off | Constraints and store capacity |
| Promise | 15% | States date | Measures accuracy | Links to conversion and contacts |
| Returns | 20% | Describes process | Cost model | Disposition and fraud controls |
| Channel economics | 15% | Revenue | Contribution | Cross-subsidy and segment effects |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready |

## 10. Regulatory, ethical, and commercial lens

- **Consumer returns:** withdrawal periods and refund rights vary by jurisdiction.
- **False promises:** advertising a delivery date the network cannot meet is a consumer-protection risk.
- **Fraud controls:** must be proportionate and avoid discriminatory profiling.
- **Labor:** store fulfillment adds tasks; workload and safety must be managed.
- **Environment:** returns transport and packaging have measurable emissions.
- **Data:** customer and location data are personal; minimize and secure.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Phantom inventory | Cancellations after order | Inaccurate stock | Broken promise | Cycle counts, vision |
| Store overload | Picks delay customers | No capacity check | Service loss | Capacity-aware routing |
| Promise inflation | Late rate rises | Marketing-led | Churn | Feasibility gate |
| Split explosion | Multiple parcels | Cost-only routing | Cost and experience | Split penalty |
| Returns black box | No disposition data | Habit | Value loss | Recovery model |
| Fraud overreach | Blanket blocks | Biased rules | Customer harm | Proportionate controls |
| Channel blindness | Web looks profitable | Costs elsewhere | Bad decisions | Channel P&L |
| BOPIS hold forever | Stale orders | No expiry | Inventory lock | Hold rules |
| Substitution surprise | Refund spike | No consent | Experience | Customer choice |
| Packaging waste | Oversized boxes | Dim weight | Cost and carbon | Packaging analytics |

## 12. Interview and discussion questions

1. What is ATP? *(Hint: on hand + inbound − reserved − safety.)*
2. How do you decide ship-from-store? *(Hint: cost, speed, labor, accuracy.)*
3. Why is store inventory inaccurate? *(Hint: process, shrinkage, receiving.)*
4. How do you measure promise accuracy? *(Hint: on-time by promise, channel, node.)*
5. What is the cheapest return? *(Hint: the one avoided, but not all are avoidable.)*
6. How do you route a return? *(Hint: recovery minus processing cost.)*
7. What is phantom inventory? *(Hint: system stock that is not physically available.)*
8. How do you detect return fraud? *(Hint: patterns, weight, serials, limits.)*
9. Why is channel P&L necessary? *(Hint: cross-subsidy.)*
10. How does BOPIS change store operations? *(Hint: pick, stage, hold, handoff.)*
11. What promise would you make in a low-density area? *(Hint: longer or pickup.)*
12. How do returns affect sustainability? *(Hint: transport, packaging, disposal.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Omnichannel | Seamless experience across channels |
| ATP | Available to promise |
| BOPIS | Buy online, pick up in store |
| Curbside | Pickup outside the store |
| Endless aisle | In-store ordering from network stock |
| DOM | Distributed order management |
| Ship-from-store | Fulfilling online orders from a store |
| Phantom inventory | System stock not physically available |
| Promise accuracy | Share of orders met by the promised date |
| Split shipment | One order shipped from multiple nodes |
| Reverse logistics | Returns, repair, resale, disposal |
| Disposition | What happens to a returned unit |
| Wardrobing | Wearing and returning an item |
| Channel P&L | Profitability by sales channel |
| Hold time | How long a pickup order is held |
| Cycle count | Periodic inventory verification |
| Substitution | Replacing an unavailable item |
| Drop-ship | Supplier ships directly |
| Endless aisle | Ordering beyond store stock |
| Recovery rate | Value recovered from a returned unit |

## 14. Sources and further reading

### Standards and regulation

- GS1, EPCIS 2.0. https://www.gs1.org/standards/epcis
- EU Consumer Rights Directive. https://eur-lex.europa.eu/
- FTC, mail order and delivery rules. https://www.ftc.gov/

### Peer-reviewed and professional

- Chopra, S., Meindl, P., *Supply Chain Management*.
- Hübner, A. et al., "Omnichannel retail operations," research on fulfillment.
- Fisher, M., "What Is the Right Supply Chain for Your Product?" *Harvard Business Review*, 1997.
- Netessine, S., Rudi, N., supply chain structures on the internet.

### Industry

- Walmart investor materials. https://stock.walmart.com/
- Target investor materials. https://investors.target.com/
- Nordstrom investor materials. https://investors.nordstrom.com/
- REI used gear program. https://www.rei.com/

### Tools

- Python, DuckDB.
- OR-Tools for routing.
- GeoPandas for catchment analysis.

> **Next:** [Module 24](24_recommendation_systems_personalization.md) connects the demand signal to what each customer is shown.
