# Module 10 — Inventory Optimization and Multi-Echelon Planning

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 34 hours (12 lecture, 12 lab, 10 project)
> **Track fit:** Common core; required for all three tracks
> **Prerequisites:** Module 07 (forecasting, backtesting); Module 08 (lead-time demand distributions, critical fractile); Module 01A (costs, units, working capital)
> **Tools:** Python 3.12, NumPy, pandas, SciPy, matplotlib; SimPy or OR-Tools optional; a spreadsheet for cross-checking
> **Industry question:** "How much of your inventory exists because somebody applied a flat 95% service target to a SKU whose stockout costs less than its holding costs?"

---

## 1. Why industry cares

Inventory is the largest controllable asset on most balance sheets outside fixed assets, and the policy that sets it is usually a small number of parameters applied to thousands of items: a service level, a review period, a safety factor, a rounding rule. Those parameters are frequently inherited, rarely re-derived, and almost never differentiated by item economics.

The decision has three faces, and they are not separable:

- **Service.** A stockout loses a sale, delays a patient, or stops a production line. The cost per unit short varies by two or three orders of magnitude across a portfolio.
- **Working capital.** Every unit of safety stock is cash. A five-day reduction in days inventory outstanding at a $200M COGS business releases roughly $2.7M — more than most analytics programmes cost.
- **Obsolescence and waste.** For perishables, fashion, and technology, the overage cost is not holding cost but write-off, and it can exceed the margin.

The 2026 context has two features. First, multi-echelon inventory optimization (MEIO) moved from a specialist add-on to a standard module in planning suites, which means the analyst is now asked to defend parameter choices rather than to compute them. Second, the post-2020 shift toward resilience raised buffer targets in many firms, often by executive instruction rather than by cost analysis; the analyst's job is to show what those targets cost and where they buy the most resilience per dollar.

The cost of a bad policy is easy to state and hard to see. A flat 95% cycle service level applied to a portfolio with stockout costs ranging from $0.50 to $5,000 will over-invest in cheap, fast-moving items and under-invest in the expensive, slow-moving ones that actually stop the line. The aggregate service number can look healthy while the items that matter are the ones that fail. That is the failure this module exists to prevent.

## 2. Learning outcomes

By the end of this module, you can:

- Build a cost model for an item or a portfolio covering holding, ordering, shortage, obsolescence, and markdown.
- Compute and critique an EOQ, and state the four conditions under which it is the wrong answer.
- Derive the newsvendor critical fractile and apply it to a portfolio with heterogeneous margins.
- Compute safety stock under demand uncertainty, lead-time uncertainty, and both together, and explain which term dominates.
- Convert between cycle service level, fill rate, and OTIF, and show why they are not interchangeable.
- Select and parameterise (s,Q), (s,S), (R,S), and base-stock policies for different demand and cost regimes.
- Explain Clark–Scarf, METRIC, guaranteed-service, and MEIO models, and when each is the appropriate level of ambition.
- Quantify the value of pooling and postponement, including the limits of the square-root law.
- Design a segmented policy using ABC/XYZ or DDMRP, and defend the segmentation to operations.
- Handle shelf life, MOQ, case pack, and capacity constraints inside a policy rather than after it.
- Simulate a policy, validate it against theory, and report service, cost, and working capital together.

## 3. Mental model

### 3.1 The policy stack

```
demand history ──► forecast distribution ──► lead-time demand distribution
                                                        │
                            service target or cost ─────┤
                                                        ▼
                                              safety stock / ROP
                                                        │
                                              policy (s,Q) or (R,S)
                                                        │
                                    constraints: MOQ, case pack, shelf life
                                                        ▼
                                             orders, service, cash
```

Each layer inherits the errors of the one above. A perfect policy on a bad lead-time demand distribution is still wrong, and a good distribution with a flat service target is still misallocated.

### 3.2 The cost–service frontier

```
 cost
  │                         ╭──────  total cost
  │                    ╭────╯
  │              ╭─────╯        holding + obsolescence
  │        ╭─────╯
  │  ╭─────╯
  │──┴────────────────────────────────────────► service level
      90%      95%      98%      99.5%
              ▲
        marginal cost per point rises sharply
```

The frontier is convex: the last point of service costs several times the first. The economically optimal point is where the marginal cost of service equals the marginal cost of a stockout — which is exactly what the critical fractile computes.

### 3.3 Which policy for which situation

| Situation | Policy | Why |
|---|---|---|
| Stable demand, cheap item, automated replenishment | (s,Q), continuous review | Simple, low review cost, fixed order size |
| Erratic demand, expensive item | (s,S) | Order-up-to absorbs variability without a fixed lot |
| Supplier ships on a fixed cycle, many items per order | (R,S) periodic review | One review, many items, shared transport |
| Very expensive, very slow, one-for-one | Base-stock (S−1,S) | Holding one unit is cheaper than a stockout; no batching |
| Perishable with short shelf life | (R,S) with FEFO and a shelf-life cap | Age matters as much as quantity |
| Capacity-constrained supply | Allocation policy plus priority rules | The constraint binds before the inventory model does |

## 4. Core concepts

### 4.1 Cost taxonomy

| Cost | Symbol | What it includes | Common error |
|---|---|---|---|
| Holding | `h` per unit per year | Capital, storage, insurance, shrinkage, obsolescence | Using only the warehouse rate and ignoring capital |
| Ordering / setup | `S` per order | Admin, transport fixed cost, receiving, setup | Ignoring the fixed transport cost per drop |
| Shortage | `C_u` per unit short | Lost margin, expedite premium, penalty, goodwill | Using margin only and ignoring the expedite cost |
| Overage | `C_o` per unit left | Holding until sale, markdown, disposal | Treating overage as zero for non-perishables |
| Obsolescence | part of `h` or a separate term | Write-off probability by age | Averaging across items with very different lives |

Holding cost is usually expressed as an annual rate on unit value, typically 18–30% including capital, storage, insurance, and shrinkage; the correct rate is a finance decision and should be stated explicitly, not defaulted.

### 4.2 EOQ and its limits

```
EOQ = √(2 D S / h)

Total annual cost = (D/Q) S + (Q/2) h
Average cycle stock = Q / 2
```

**Worked example.** Annual demand `D = 10,000` units, order cost `S = $250`, holding cost `h = $5` per unit per year (20% of a $25 unit cost).

```
EOQ = √(2 × 10,000 × 250 / 5) = √1,000,000 = 1,000 units
Orders per year = 10,000 / 1,000 = 10
Average cycle stock = 500 units
Total annual cost = 10 × 250 + 500 × 5 = $2,500 + $2,500 = $5,000
```

At the EOQ, ordering cost equals holding cost — the structural result that makes the formula robust: a 20% error in any input produces roughly a 10% error in `Q` and about a 0.5% error in total cost. That flatness is why EOQ survives despite its assumptions.

| EOQ assumption | Where it breaks | What to do instead |
|---|---|---|
| Demand is known and constant | Seasonal, lumpy, or promoted demand | Newsvendor or dynamic lot-sizing |
| Lead time is fixed | Ocean freight variability, supplier delays | Safety stock with lead-time variance |
| No quantity discounts | Supplier tiers, freight brackets | Discounted EOQ or a full cost comparison |
| No shelf life | Perishables, fashion, technology | Age-constrained policy; markdown model |
| No capacity limit | Allocation, truck capacity, MOQ | Constrained lot-sizing |

EOQ sets **cycle stock**, not safety stock. Confusing the two is the most common inventory-modelling error: EOQ answers "how much to order each time", safety stock answers "how low before ordering".

### 4.3 Newsvendor and the critical fractile

For a single period with underage cost `C_u` and overage cost `C_o`:

```
q* = C_u / (C_u + C_o)
Q* = F⁻¹(q*)
```

**Worked example.** A seasonal item sells at $10, costs $4, and can be salvaged for $2. Demand is normal with `μ = 500`, `σ = 100`.

```
C_u = 10 − 4 = 6
C_o = 4 − 2 = 2
q*  = 6 / (6 + 2) = 0.75
Q*  = 500 + 0.6745 × 100 = 567.4 → order 568
```

Expected cost at various quantities, computed as `C_u·E[(D−Q)⁺] + C_o·E[(Q−D)⁺]`:

| Order quantity | Expected shortage cost | Expected overage cost | Total expected cost |
|---:|---:|---:|---:|
| 400 | 619.50 | 47.15 | 666.65 |
| 500 | 300.00 | 19.15 | 319.15 |
| 567 | 235.10 | 19.12 | 254.22 |
| 600 | 247.10 | 19.55 | 266.65 |
| 700 | 387.85 | 18.94 | 406.79 |

The cost curve is flat near the optimum and steep in the tails, which is the practical argument for spending analytical effort on the **cost ratio** rather than on the demand distribution's third moment.

| Business | `C_u` | `C_o` | Fractile | Order at |
|---|---:|---:|---:|---|
| Fashion, 50% margin, 20% markdown | 10 | 8 | 0.56 | Just above the median |
| Critical spare, downtime cost dominates | 500 | 20 | 0.96 | Far tail |
| Grocery perishable, thin margin, disposal cost | 3 | 4 | 0.43 | Below the median |

### 4.4 Safety stock under demand and lead-time uncertainty

With constant lead time `L` and iid daily demand with standard deviation `σ_d`:

```
σ_LT = σ_d √L
SS    = z · σ_LT
ROP   = d·L + SS
```

With uncertain lead time as well:

```
σ_LT = √( L·σ_d² + d²·σ_L² )

SS = z · √(L σ_d² + d² σ_L²)
```

**Worked example.** `d = 100` units/day, `σ_d = 20`, `L = 9` days, `σ_L = 2` days, target CSL 95% (`z = 1.645`).

```
σ_LT = √(9 × 400 + 10,000 × 4) = √(3,600 + 40,000) = √43,600 = 208.8
SS   = 1.645 × 208.8 = 343.5 units
ROP  = 900 + 343.5 = 1,243.5 units
```

The lead-time variance term (40,000) dominates the demand term (3,600) by a factor of eleven. **This is the single most useful diagnostic in inventory analytics**: if lead-time variance dominates, no amount of demand-forecast improvement will fix the buffer, and the intervention should be supply-side — reduce lead-time variability, not forecast error.

| Source of uncertainty | Contribution to `σ_LT²` | Lever |
|---|---:|---|
| Demand variability | `L σ_d²` | Better forecast, shorter review period |
| Lead-time variability | `d² σ_L²` | Supplier reliability, modal choice, safety time |
| Both | Sum | Prioritise the larger term; measure both |

### 4.5 Service metrics: CSL, fill rate, OTIF

| Metric | Definition | Formula | What it measures |
|---|---|---|---|
| Cycle service level (CSL) | Probability of no stockout in a replenishment cycle | `F(ROP)` | Whether the cycle ended short at all |
| Fill rate `β` | Share of demand met from stock | `1 − E[short]/Q` | How much demand was served |
| OTIF | Orders delivered on time and in full | `on-time and complete / total` | Execution performance, not just stock |

Expected shortage per cycle, with the standard normal loss function:

```
L(z) = φ(z) − z(1 − Φ(z))
E[short] = σ_LT · L(z)
β = 1 − σ_LT · L(z) / Q
```

**Worked example, continuing §4.4.** `z = 1.645`, `σ_LT = 208.8`, `Q = 500`.

```
L(1.645) = 0.1031 − 1.645 × 0.05 = 0.0209
E[short] = 208.8 × 0.0209 = 4.36 units per cycle
β = 1 − 4.36 / 500 = 0.9913 = 99.1%
```

A 95% CSL produced a 99.1% fill rate because the order quantity is large relative to the shortage. The relationship depends on `Q`: the same CSL with `Q = 100` gives `β = 95.6%`. **Never convert a service target between metrics without the order quantity.**

| Target stated as | Typical equivalent CSL | Note |
|---|---:|---|
| Fill rate 95% with Q = 50, σ_LT = 200 | ~99.5% CSL | Small Q relative to σ forces a high z |
| Fill rate 99% with Q = 1,000 | ~93% CSL | Large Q hides frequent small shortages |
| OTIF 95% | Not derivable | Includes transport and documentation failures |

### 4.6 Policy families

| Policy | Review | Order rule | Parameters | Best for |
|---|---|---|---|---|
| (s,Q) | Continuous | Order Q when inventory position ≤ s | `s`, `Q` | Stable demand, cheap items, automated |
| (s,S) | Continuous | Order up to S when position ≤ s | `s`, `S` | Erratic demand, expensive items |
| (R,S) | Periodic | Order up to S every R periods | `R`, `S` | Shared ordering cycles, supplier schedules |
| Base-stock (S−1,S) | Continuous | Order one for each unit consumed | `S` | Expensive, slow, critical spares |
| (R,s,S) | Periodic | If position ≤ s, order up to S | `R`, `s`, `S` | Both a review cycle and a reorder trigger |

Protection interval matters more than the policy label:

```
(R,S): protection = L + R
(s,Q): protection = L
```

A periodic review system with `R = 7` and `L = 9` needs a buffer for 16 days of demand, not 9. Moving from continuous to weekly review commonly increases required safety stock by 10–25% for the same service — a cost that should be attributed to the review decision, not to the forecast.

### 4.7 Lead-time demand distributions

| Demand process | Lead-time demand distribution | When to use |
|---|---|---|
| iid normal demand, fixed lead time | `N(dL, σ_d² L)` | High-volume items, short lead times |
| iid normal, random lead time | Mean `dL`, variance `Lσ_d² + d²σ_L²` | Supplier variability |
| Poisson demand | Poisson(`λL`) | Low-volume eaches, unit sizes |
| Compound Poisson | Convolution of size distribution over `N_L` arrivals | Spares, wholesale orders |
| Empirical | Bootstrap of historical lead-time demand | Skewed, lumpy, or regime-changing demand |

The normal approximation fails badly at low mean demand: at `dL = 4`, a normal 95th percentile is 7.3 while the true Poisson value is 8, and the normal has mass below zero. Use the exact distribution when the mean lead-time demand is below about 20 units.

### 4.8 Multi-echelon inventory

| Model | Idea | Decision it supports | Limitation |
|---|---|---|---|
| Clark–Scarf (1960) | Echelon-stock decomposition; each stage's optimal policy depends on the next stage's | Optimal policy structure in a serial system | Serial or tree structures; computational cost |
| METRIC (Sherbrooke, 1968) | Poisson demand, backorders, marginal analysis for recoverable items | Spare-parts stock across repair and depot levels | Recoverable items only; no batching |
| Guaranteed service (Graves–Willems, 2000) | Bound the service time at each stage; optimise where to hold safety stock | Strategic buffer placement in a network | Service times must be bounded and respected |
| MEIO (commercial) | Optimise safety stock across echelons with cost and service targets | Network-wide buffer allocation | Depends on the demand and lead-time inputs; not a substitute for them |

**Worked multi-echelon illustration.** One DC serves four identical retailers. Daily demand per retailer is iid normal with `σ = 20`; retailer lead time `L_r = 2` days, DC lead time `L_dc = 3` days; target CSL 95%.

*Installation-stock (each node buffers for its own lead time):*

```
Each retailer: 1.645 × 20 × √2 = 46.5;  four retailers = 186.1
DC sees σ = 20√4 = 40: 1.645 × 40 × √3 = 114.0
Total safety stock = 300.1 units
```

*Echelon-stock (DC buffers for the whole downstream chain over `L_dc + L_r`):*

```
DC covers σ = 40 over 5 days: 1.645 × 40 × √5 = 147.1
Retailers hold only local protection: 0 in this idealisation
Total safety stock = 147.1 units
```

The echelon policy uses 51% less safety stock for the same target. This is an **upper bound on the benefit**, not a forecast of it: it assumes deterministic lead times, no batching, no MOQ, independent identical retailers, and a DC that can serve every retailer within the promised time. Real implementations typically realise 10–30%, and the benefit falls as lead times become more variable and as batching appears.

### 4.9 Pooling and postponement

**Risk pooling.** Combining `N` independent demands with equal standard deviation `σ` in one location gives `σ√N` instead of `Nσ`, so safety stock falls by `1 − 1/√N`:

| Locations consolidated | Safety stock relative to separate | Reduction |
|---:|---:|---:|
| 2 | 0.71 | 29% |
| 4 | 0.50 | 50% |
| 9 | 0.33 | 67% |
| 16 | 0.25 | 75% |

The law assumes independence and equal variance. Correlated demand — the same weather across a region, a national promotion — destroys much of the benefit, which is why pooling works best across geographically dispersed, weakly correlated markets.

**Postponement.** Delaying differentiation (labelling, power supply, final assembly, packaging) until demand is observed moves the point of variety downstream. It reduces the number of SKUs carrying safety stock and shortens the forecast horizon for the differentiated item. It costs money: postponed configuration is usually more expensive per unit, and it requires a distribution centre that can perform the final step.

### 4.10 Segmentation: ABC/XYZ and DDMRP

| Class | Definition | Typical policy |
|---|---|---|
| A | Top 80% of value or volume | Frequent review, tight control, item-level forecasting |
| B | Next 15% | Periodic review, standard policy |
| C | Last 5% | Simple rules, larger cycles, higher tolerance |
| X | Low demand variability (CV < 0.5) | Forecast-driven, lean buffer |
| Y | Moderate variability (CV 0.5–1.0) | Buffer plus review |
| Z | High variability (CV > 1.0) | Buffer by service target, not by forecast |

The 80/15/5 and CV cut-offs are conventions. The value is in the differentiation, not in the exact boundaries — a policy that treats all 5,000 SKUs identically is almost certainly wrong for both the top 50 and the bottom 2,000.

**DDMRP** (Demand Driven MRP, from the Demand Driven Institute) places decoupling points in the bill of materials and sizes buffers in red/yellow/green zones from average daily usage, lead time, and variability. Its contribution is strategic: where to hold stock, not just how much. Its risk is that buffer sizing can become a formula applied without validation against actual cost and service outcomes.

### 4.11 Shelf life and perishables

For a perishable, the overage cost includes disposal and the value lost when stock ages out:

```
C_o = unit cost + disposal cost − salvage value
q*  = C_u / (C_u + C_o)
```

Three additional controls are needed:

1. **FEFO issuing** (first-expired, first-out), enforced in the warehouse management system, not assumed.
2. **Shelf-life cap on receipt** — reject or discount stock with less than a stated remaining life.
3. **Age-based forecasting** — the demand for a batch is not the demand for the SKU; near-expiry stock sells at a different rate and price.

The inventory position must be measured in **remaining-life buckets**, not units, or the model will treat 100 units with two days left the same as 100 units with two weeks left.

### 4.12 Capacity, MOQ, and lot-sizing constraints

| Constraint | Effect on the policy | Handling |
|---|---|---|
| MOQ | Raises average inventory; forces over-ordering of slow items | Add MOQ/2 to average cycle stock; consider holding cost versus supplier terms |
| Case pack | Orders must be multiples of a pack | Round up after optimisation, then re-check service |
| Truck capacity | Order size capped by cube or weight | Constrained lot-sizing; consolidate items |
| Supplier allocation | Supply below demand | Priority rules; rationing by customer segment |
| Shelf life | Maximum days of stock | Cap `S`; shorten `R` |

These constraints break the clean formulas. The practical approach is to optimise unconstrained, apply constraints, and then measure the service or cost loss — and to report it, because the constraint is usually a decision someone can change.

### 4.13 Working capital

```
DIO = 365 × average inventory value / COGS
Cash conversion cycle = DIO + DSO − DPO
```

**Worked example.** Average inventory $40M, COGS $200M, DSO 35 days, DPO 40 days.

```
DIO = 365 × 40 / 200 = 73 days
CCC = 73 + 35 − 40 = 68 days
A 5-day DIO reduction frees = 200 / 365 × 5 = $2.74M of cash
```


### 4.14 Policy simulation

Theory gives a starting point; simulation gives the distribution of outcomes under a real policy, including batching, discrete demand, and lead-time variability.

```python
import numpy as np

def simulate_sQ(demand_fn, lead_time, s, Q, days=3650, seed=7):
    """Continuous-review (s,Q) simulation with stochastic daily demand.

    demand_fn(rng) -> units demanded that day. Orders arrive at the start of
    day t + lead_time. Returns fill rate, stockout-day share, average on-hand,
    and counts of orders placed.
    """
    rng = np.random.default_rng(seed)
    on_hand = float(s + Q)
    on_order = 0.0
    arrivals = np.zeros(days + lead_time + 1)
    demand_total = served = 0.0
    stockout_days = 0
    inv_area = 0.0
    orders = 0

    for t in range(days):
        if arrivals[t] > 0:
            on_hand += arrivals[t]
            on_order -= arrivals[t]
        d = float(demand_fn(rng))
        demand_total += d
        ship = min(on_hand, d)
        on_hand -= ship
        served += ship
        if ship < d:
            stockout_days += 1
        if on_hand + on_order <= s:
            arrivals[t + lead_time] += Q
            on_order += Q
            orders += 1
        inv_area += on_hand

    return {
        "fill_rate": served / demand_total,
        "stockout_day_share": stockout_days / days,
        "avg_on_hand": inv_area / days,
        "orders_placed": orders,
    }

if __name__ == "__main__":
    demand = lambda rng: rng.poisson(20)
    for s, Q in [(80, 200), (120, 200), (120, 400)]:
        r = simulate_sQ(demand, lead_time=5, s=s, Q=Q)
        print(s, Q, {k: round(v, 3) for k, v in r.items()})
```

Expected behaviour: raising `s` from 80 to 120 raises the fill rate and average on-hand together; doubling `Q` to 400 cuts the order count roughly in half and raises average cycle stock by about 100 units while barely moving the fill rate. Those three comparisons — service, cycle stock, order frequency — are the ones a planner will ask about, and they should be produced from one simulation run rather than three models.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| On-hand inventory | SKU × location × day | WMS / ERP | Include in-transit and allocated stock; distinguish physical from available |
| Demand | SKU × location × day | Orders / POS | Censored at stockout; distinguish independent from dependent demand |
| Lead time | PO × line | ERP / supplier portal | Actual receipt dates, not promised dates |
| Cost | SKU × year | Finance | Unit cost, holding rate, stockout penalty, salvage |
| Open orders | PO × line | ERP | Quantity, expected date, status |
| MOQ and pack | SKU | Item master | Case pack, pallet quantity, minimum order |
| Shelf life | SKU × batch | WMS / quality | Remaining life, expiry date, FEFO flag |
| Service target | SKU or segment | Commercial policy | Stated as CSL, fill rate, or OTIF — never ambiguous |

**Standards.** GS1 GTIN and GLN for item and location identity; SSCC for pallet and case identification where receiving accuracy matters. ASCM SCOR DS for the service metrics and for the plan/source/deliver definitions used in reporting. ISO 8601 for dates. For pharmaceuticals, EU GDP and DSCSA constrain storage, traceability, and expiry handling, and those constraints belong in the policy, not in a footnote.

**Workflow checks.** Confirm that on-hand excludes allocated and quality-held stock; confirm lead time is measured receipt-to-receipt rather than promised-to-promised; confirm demand is de-censored for stockout days; confirm the service metric in the policy matches the metric in the report; confirm that MOQ rounding is applied before the service calculation, not after.

## 6. Industry case studies

### 6.1 Hewlett-Packard and DeskJet postponement (1995)

**Context.** HP's DeskJet printer business served European markets from a manufacturing operation with country-specific power supply and packaging requirements. Finished goods were differentiated early, so each country variant carried its own inventory and its own forecast error.

**Intervention.** Lee and Billington described the supply chain and the move toward delaying localization — shipping a generic product and completing the country-specific configuration closer to the market.

**Evidence.** The *Interfaces* paper reports the modelling work and the inventory and service consequences of the redesign, with the quantification drawn from HP's own operational data. The results are a company case study, not an independent trial.

**Outcome.** Postponement reduced the number of forecast-error-bearing SKUs and improved service at lower inventory. The case became the standard reference for delayed differentiation.

**What transferred.** The mechanism: postponement does not improve the forecast, it moves the point at which the forecast is required to a place where the horizon is shorter and the aggregation is higher. **What did not.** The specific inventory numbers, which depend on HP's cost structure and European distribution, and the assumption that a DC can perform the final configuration at acceptable cost.

### 6.2 Dell and build-to-order (1998)

**Context.** Dell assembled personal computers to order rather than to stock, taking customer orders before committing components to finished goods.

**Intervention.** Build-to-order with a configured supply chain, direct sales, and component inventory managed against real orders rather than a finished-goods forecast.

**Evidence.** Magretta's *Harvard Business Review* interview with Michael Dell describes the model and its economics; the figures are company-reported.

**Outcome.** Dell operated with materially lower finished-goods inventory than its competitors and, in the period described, a negative cash conversion cycle — customers paid before suppliers were paid.

**What transferred.** The structural insight that inventory is a consequence of lead time and forecast error: shortening the order-to-delivery cycle removes the need to forecast the configured product. **What did not.** The full model in categories where configuration is not modular, where customers will not wait, or where channel partners hold the inventory.

### 6.3 Walmart and vendor-managed inventory

**Context.** Retail replenishment in the 1990s was characterised by the bullwhip: orders to suppliers amplified demand variability at each tier.

**Intervention.** Walmart shared point-of-sale data with suppliers and moved toward vendor-managed and continuous-replenishment programmes, so that replenishment was driven by sell-through rather than by distribution-centre orders.

**Evidence.** The bullwhip mechanism is documented in Lee, Padmanabhan, and Whang (1997); the Walmart programme is described in industry and case literature. Quantified inventory and service improvements are company claims and should be treated as such.

**Outcome.** The programme is widely cited as an early demonstration that sharing demand signals upstream reduces amplification and inventory across the chain.

**What transferred.** The principle that the inventory policy at an upstream node should see downstream demand, which is the practical argument for multi-echelon visibility. **What did not.** The assumption that data sharing alone fixes the problem: without aligned incentives and replenishment rules, shared data is ignored or gamed.

### 6.4 METRIC and recoverable items (1968)

**Context.** Repairable spare parts — aircraft components, electronics — move through echelons and return to a repair depot. Holding a spare at every base is expensive; holding too few grounds equipment.

**Intervention.** Sherbrooke's METRIC model represented demand as Poisson, modelled backorders at each echelon, and allocated stock by marginal analysis so that each additional unit went where it reduced expected backorders most.

**Evidence.** The *Operations Research* paper sets out the model and its application to recoverable-item inventory management in a military context; the method became the basis for later multi-echelon spare-parts systems.

**Outcome.** METRIC established that the value of a spare depends on its position in the network, not only on its local demand, and that marginal allocation across echelons outperforms independent local policies.

**What transferred.** Marginal analysis as the allocation logic, and the framing of service as expected backorders rather than as a local fill rate. **What did not.** The Poisson and unit-demand assumptions for repairable items, which do not transfer to consumable, batch-ordered, or lumpy demand without modification.

## 7. Labs

### Lab 10.1 — Newsvendor portfolio and the cost of a flat service target

**Objective.** Show the cost of applying one service level to a heterogeneous portfolio.

**Data.** 500 SKUs with unit cost, price, salvage, and demand mean and standard deviation from a real or realistic extract.

**Steps.** (1) Compute `C_u`, `C_o`, and the critical fractile per SKU. (2) Compute the optimal order quantity under a normal assumption. (3) Compare with a flat 95% service policy. (4) Estimate expected cost under both. (5) Identify the SKUs where the flat policy is most wrong. (6) Quantify the aggregate cost difference and its sensitivity to the cost inputs.

**Deliverable.** A portfolio comparison with an aggregate cost difference, a per-SKU divergence table, and a sensitivity analysis.

**Acceptance criteria.** Critical fractiles per SKU; negative quantities handled; the flat-policy baseline explicit; sensitivity reported. **Stretch.** Use an empirical distribution for the lumpy SKUs.

### Lab 10.2 — Safety stock decomposition

**Objective.** Determine whether demand or lead-time variability drives the buffer.

**Data.** 300 SKUs with daily demand history and receipt-level lead times.

**Steps.** (1) Compute `σ_d`, `σ_L`, and the two variance contributions. (2) Compute safety stock under constant and variable lead time. (3) Rank SKUs by the share of variance from lead time. (4) Estimate the buffer reduction from halving lead-time variance versus halving demand variance. (5) Recommend an intervention per segment.

**Deliverable.** A decomposition table and a one-page recommendation on where to spend improvement effort.

**Acceptance criteria.** Both variance terms computed; units consistent; the recommendation distinguishes supply-side from demand-side action. **Stretch.** Add supplier lead-time correlation.

### Lab 10.3 — Simulate and tune a two-echelon policy

**Objective.** Compare installation-stock and echelon-stock policies under realistic constraints.

**Data.** Synthetic two-echelon network (one DC, four retailers) calibrated to a real cost structure; demand and lead-time distributions from Lab 10.2.

**Steps.** (1) Implement a simulation with daily demand, stochastic lead times, and case-pack rounding. (2) Parameterise installation-stock policies to hit 95% CSL at each node. (3) Parameterise an echelon-stock policy to hit 95% at the retailer. (4) Compare total safety stock, fill rate, and cost. (5) Add MOQ at the DC and re-run. (6) Report the conditions under which echelon stock stops helping.

**Deliverable.** A simulation study with a policy comparison table and a conditions-for-benefit analysis.

**Acceptance criteria.** Seeds fixed; service measured at the customer-facing node; MOQ applied; cost and service reported together; the §4.8 bound compared with the simulation. **Stretch.** Add a retailer shelf-life constraint.

### Lab 10.4 — Working capital and service trade-off

**Objective.** Build the case for an inventory reduction without ignoring service.

**Data.** Company or simulated balance sheet plus the policy model from Lab 10.3.

**Steps.** (1) Compute DIO and the cash conversion cycle. (2) Identify a 10% inventory reduction target. (3) Simulate the service consequence under three policies: uniform cut, ABC-differentiated cut, and re-optimised buffers. (4) Estimate cash released and service change for each. (5) Write the trade-off curve. (6) Recommend a policy with an explicit service floor.

**Deliverable.** A two-page memo with a trade-off curve and a recommendation.

**Acceptance criteria.** Cash release from COGS; service simulated, not assumed; a stated service floor with an owner. **Stretch.** Add markdown cost for residual overstock.

## 8. Assignment — Inventory policy model

Produce a **policy model** for one category or network that a supply planner could run next month.

| Section | Content |
|---|---|
| 1. Scope | Items, locations, service metric, and the decision the policy controls |
| 2. Cost model | Holding, ordering, shortage, obsolescence, with sources for each rate |
| 3. Demand | Distribution per segment, with the stockout-censoring treatment |
| 4. Lead time | Distribution and measurement method |
| 5. Policy design | Policy family per segment, parameters, protection interval |
| 6. Constraints | MOQ, case pack, capacity, shelf life, and their cost |
| 7. Simulation | Validation against theory, seeds, scenarios |
| 8. Results | Service, average inventory, cost, and working capital |
| 9. Risks | Parameter sensitivity and the conditions that break the policy |
| 10. Reproduction | Code, data snapshot, parameter file, one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Cost model | 20% | Holding cost only | All major cost terms included | Rates sourced and defended; obsolescence modelled by age |
| Demand and lead-time distributions | 20% | Normal assumed everywhere | Distribution chosen per segment | Empirical or compound models where the normal fails; censoring handled |
| Policy design | 20% | One policy for all items | Segmented policies | Parameters derived from cost and service; protection interval correct; constraints integrated |
| Multi-echelon reasoning | 15% | Single-echelon only | Echelon effects discussed | Quantified comparison with assumptions and limits stated |
| Simulation and validation | 15% | No simulation | Simulation with fixed seeds | Theory-versus-simulation validation; sensitivity and stress scenarios |
| Commercial framing | 10% | Units of inventory only | Cost and service reported | Working capital, cash release, and a service floor with an owner |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Essential medicines and food | A cost-optimal service level may be below the socially required one | Set a service floor by regulation or policy; document who approved it |
| Pharmaceutical storage and traceability | EU GDP and DSCSA impose storage, expiry, and serialization requirements | Shelf-life and FEFO rules in the policy; EPCIS for traceability |
| Allocation fairness | Buffer allocation favours high-margin or high-volume locations | Report service by region and customer segment; publish the allocation rule |
| Supplier commitments | Order quantities and forecasts shared with suppliers can become expectations | State horizon, binding window, and liability explicitly |
| Working capital targets | A finance-driven inventory cut can silently transfer cost to service | Simulate the service consequence; require a stated service floor |
| Waste and sustainability | Overstock becomes waste; understock causes expediting and emissions | Include disposal and expedite emissions in the cost model where material |
| EU AI Act (Regulation 2024/1689) | Automated allocation of essential goods may be high-risk depending on use | Documentation, human oversight, and an exception path |

The ethical question in inventory is distributional: an optimisation that maximises total profit will under-serve low-margin rural or small-account demand unless a constraint prevents it. That is a policy choice, and it should be visible in the model rather than buried in a service parameter.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| EOQ used as a safety-stock model | Cycle stock grows while service does not | Confusing two different quantities | Excess inventory, unchanged stockouts | Separate cycle and safety stock; derive each |
| Flat service target | Same CSL for a $0.50 and a $5,000 shortage cost | No cost differentiation | Over-invest in cheap items, under-invest in critical ones | Critical fractile per segment |
| Normal demand at low volume | Negative or zero reorder points | Continuous approximation | Broken policy, manual fixes | Poisson or empirical distribution |
| Lead-time variance ignored | Buffer set from demand variance only | Incomplete variance | Chronic stockouts despite "95%" | Measure `σ_L`; include the `d²σ_L²` term |
| Promised lead time used | Buffer looks adequate, service is poor | Optimism in the master data | Systematic shortfall | Measure actual receipt dates |
| Review period omitted from protection | Periodic review sized like continuous | Protection interval wrong | 10–25% under-buffer | Protection = L + R |
| CSL reported as fill rate | Service looks worse than customers experience | Metric confusion | Wrong target set | Report both; state the conversion |
| Censored demand treated as true | Fast movers look slower than they are | Stockout days counted as low demand | Under-forecast, repeat stockouts | De-censor or exclude stockout days |
| MOQ ignored in optimisation | Slow movers show impossible order quantities | Constraint applied after the fact | Overstock and expiry | Constrain, then re-measure service |
| Shelf life ignored | Inventory is "available" but unsellable | Units, not age, modelled | Write-off and waste | Age-bucketed inventory; FEFO |
| MEIO run on bad inputs | Confident network answer, poor service | Model precision exceeds data quality | Expensive wrong answer | Validate demand and lead time first |
| Inventory cut without simulation | Finance target met, service falls later | Local optimisation | Lost sales, expediting | Trade-off curve with a service floor |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is the difference between cycle stock and safety stock? | Cycle stock comes from order quantity; safety stock from uncertainty over the protection interval. |
| 2 | When is EOQ the wrong model? | Seasonal or lumpy demand, quantity discounts, shelf life, capacity constraints, or when the ordering decision is really a newsvendor decision. |
| 3 | Derive the critical fractile. | Marginal unit: cost `C_o` if unsold, benefit `C_u` if sold; set `q C_u = (1−q) C_o`; `q = C_u/(C_u+C_o)`. |
| 4 | Your demand forecast improves 20%. How much does safety stock fall? | Only the demand-variance term falls; if lead-time variance dominates, the buffer barely moves. |
| 5 | Why does a 95% CSL not mean 95% of demand is served? | CSL is a cycle-level probability; fill rate depends on the expected shortage relative to Q. |
| 6 | What does the square-root law assume, and when does it fail? | Independent, identically distributed demand; correlation, unequal variance, and lead-time differences erode the benefit. |
| 7 | Explain echelon stock in one sentence. | Inventory measured as all stock at or downstream of a node, so the node buffers for the chain rather than for itself. |
| 8 | What is the difference between METRIC and guaranteed-service models? | METRIC models stochastic demand and backorders with marginal allocation; guaranteed service bounds stage service times and optimises buffer placement. |
| 9 | When would you use periodic review? | Shared transport or supplier cycles make continuous review impractical; accept the larger protection interval. |
| 10 | How would you cut inventory 10% without hurting service? | Re-optimise buffers by critical fractile, cut C-class overstock, shorten review periods, reduce lead-time variance; simulate before committing. |

## 13. Key terms

| Term | Definition |
|---|---|
| Cycle stock | Inventory held to satisfy demand between orders |
| Safety stock | Buffer held against uncertainty over the protection interval |
| EOQ | Economic order quantity; minimises ordering plus holding cost |
| Newsvendor model | Single-period order decision under uncertain demand |
| Critical fractile | The optimal service level `C_u/(C_u+C_o)` |
| Underage cost | Cost of being one unit short |
| Overage cost | Cost of being one unit long |
| CSL | Cycle service level; probability of no stockout in a cycle |
| Fill rate | Share of demand met from stock |
| OTIF | On-time, in-full delivery performance |
| Protection interval | Lead time plus review period |
| Reorder point | Inventory position that triggers an order |
| (s,Q) | Continuous-review policy with a fixed order quantity |
| (s,S) | Continuous-review policy ordering up to S |
| (R,S) | Periodic-review policy ordering up to S every R periods |
| Echelon stock | Inventory at and downstream of a node |
| METRIC | Multi-echelon technique for recoverable item control |
| Guaranteed service | Buffer placement with bounded stage service times |
| Risk pooling | Variance reduction from aggregating independent demand |
| Postponement | Delaying differentiation until demand is known |
| DIO | Days inventory outstanding |
| FEFO | First-expired, first-out issuing |
| ABC/XYZ | Value and variability segmentation |

## 14. Sources and further reading

**Standards and regulation**

- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). Accessed 27 October 2026.
- GS1, *General Specifications*; GTIN, GLN, SSCC. https://www.gs1.org/standards — accessed 27 October 2026.
- European Commission, Guidelines on Good Distribution Practice of Medicinal Products for Human Use (2013/C 343/01). Accessed 27 October 2026.
- U.S. FDA, Drug Supply Chain Security Act requirements. https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa — accessed 27 October 2026.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex. Accessed 27 October 2026.

**Peer-reviewed and foundational**

- Harris, F. W. "How Many Parts to Make at Once." *Factory, The Magazine of Management*, 1913; reprinted in *Operations Research*, 38(6), 1990.
- Arrow, K. J., Harris, T., and Marschak, J. "Optimal Inventory Policy." *Econometrica*, 19(3), 1951.
- Clark, A. J., and Scarf, H. "Optimal Policies for a Multi-Echelon Inventory Problem." *Management Science*, 6(4), 1960.
- Scarf, H. "The Optimality of (S,s) Policies in the Dynamic Inventory Problem." In *Mathematical Methods in the Social Sciences*, Stanford University Press, 1960.
- Sherbrooke, C. C. "METRIC: A Multi-Echelon Technique for Recoverable Item Control." *Operations Research*, 16(1), 1968.
- Eppen, G. D. "Effects of Centralization on Expected Costs in a Multi-Location Newsboy Problem." *Management Science*, 25(5), 1979.
- Graves, S. C., and Willems, S. P. "Optimizing Strategic Safety Stock Placement in Supply Chains." *Manufacturing & Service Operations Management*, 2(1), 2000.
- Lee, H. L., and Billington, C. "The Evolution of Supply-Chain-Management Models and Practice at Hewlett-Packard." *Interfaces*, 25(5), 1995.
- Lee, H. L., Padmanabhan, V., and Whang, S. "Information Distortion in a Supply Chain: The Bullwhip Effect." *Management Science*, 43(4), 1997.
- Magretta, J. "The Power of Virtual Integration: An Interview with Michael Dell." *Harvard Business Review*, 76(2), 1998.
- Nahmias, S. "Perishable Inventory Theory: A Review." *Operations Research*, 30(4), 1982.
- Silver, E. A., Pyke, D. F., and Peterson, R. *Inventory Management and Production Planning and Scheduling*, 3rd ed. Wiley, 1998.

**Industry**

- Gartner, "Gartner Predicts 70% of Large Organizations Will Adopt AI-Based Supply Chain Forecasting to Predict Future Demand by 2030," press release, 16 September 2025. Analyst prediction. https://www.gartner.com/en/newsroom/press-releases/2025-09-16-gartner-predicts-70-percent-of-large-orgs-will-adopt-ai-based-supply-chain-forecasting-to-predict-future-demand-by-2030 — accessed 27 October 2026.
- DP World, *Global Trade Observatory 2026*, 2026. Reported: 44% of executives cite higher inventories as a resilience response. Survey evidence; treat as a dated snapshot. [VERIFY: final sample and wording.]
- Demand Driven Institute, Demand Driven MRP materials and the DDMRP standard. Accessed 27 October 2026.

**Tools and datasets**

- `SimPy` discrete-event simulation documentation. https://simpy.readthedocs.io — accessed 27 October 2026.
- OR-Tools and Pyomo for constrained lot-sizing and allocation models. Accessed 27 October 2026.
- UCI Online Retail II and the M5 dataset for demand histories used in the labs. Accessed 27 October 2026.

> **Note:** Service level, fill rate, and OTIF are different quantities. A policy document that names only "service" has not specified a policy.

---

> **Next:** [Module 11](11_network_design_optimization.md) — from item-level buffers to the shape of the network: facility location, MILP formulations, and the cost-to-serve arithmetic behind a redesign.
