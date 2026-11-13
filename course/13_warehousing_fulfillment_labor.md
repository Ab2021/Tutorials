# Module 13 — Warehousing, Fulfillment, and Labor Analytics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (12 lecture, 10 lab, 8 project)
> **Track fit:** Logistics & Distribution core; recommended for Retail & E-commerce and Manufacturing & Industrial Operations
> **Prerequisites:** Module 01A (units and time); Module 02 (operational data); Module 12 recommended; basic probability and queueing intuition
> **Tools:** Python 3.12, pandas, NumPy, SimPy, matplotlib, OR-Tools optional; warehouse layout and travel-time models in plain Python
> **Industry question:** "Given this order profile, layout, and labor pool, how many orders can this building ship on its peak day — and where does the next hour of capacity actually come from?"

---

## 1. Why industry cares

A warehouse is a factory whose product is a shipped order, and like a factory it has a bottleneck, a cycle time, a yield, and a labor cost per unit. Unlike a factory, its demand arrives as a mixture: pallets for stores, eaches for online orders, returns coming back through the same doors, and a peak season that can be three times the average week.

The economics are unforgiving. Order picking alone is commonly cited as roughly 55% of warehouse operating cost (De Koster, Le-Duc, and Roodbergen, 2007), and travel is the largest component of picking time. Labor is typically 50–65% of a distribution center's operating cost, which makes staffing the single largest controllable lever and the single largest source of variance.

Three forces shape the 2026 picture.

1. **Automation is real but selective.** Amazon reported surpassing one million warehouse robots in July 2025 and introduced DeepFleet, a generative-AI fleet coordination model; Ocado reported its On-Grid Robotic Pick handled more than 30 million items in 2024 with fewer than 100 arms. These are company claims, and both companies operate at a scale and order profile most firms do not have.
2. **Micro-fulfillment consolidated.** Several venture-backed micro-fulfillment vendors did not survive the shift from pilot to unit economics; the model works where real estate is expensive, delivery windows are short, and order density is high — not as a universal answer.
3. **Labor and safety are board-level.** Injury rates, ergonomics, turnover, and scheduling fairness are now measured alongside cost per order, and regulators and legislators have taken an interest in warehouse quotas and injury reporting.

The cost of getting this wrong is a building that looks efficient on an average day and collapses in week 48, or an automation investment that pays back only if utilization stays above a level the demand never reaches.

## 2. Learning outcomes

By the end of this module, you can:

- Map the physical and information flow of a warehouse from receiving to returns, and identify where cost and time accumulate.
- Read and extend a WMS data model, and query it to reconstruct inventory position, task history, and order cycle time.
- Compute UPH, lines per hour, dock-to-stock, order cycle time, and perfect order rate, and explain what each does and does not measure.
- Apply Little's Law to a fulfillment process and use it to size work-in-process and cycle time.
- Model a pick, pack, or dock process as a queue and quantify the waiting cost of high utilization.
- Build a discrete-event simulation of a fulfillment area in SimPy and validate it against observed throughput.
- Design an ABC slotting policy and estimate the travel-time saving from a layout change.
- Set and audit engineered labor standards, including allowances and performance rating.
- Compare picking strategies — discrete, batch, zone, wave, cluster — on the order profile they fit.
- Build an automation business case with throughput, uptime, labor, and payback sensitivity.
- Size dock and yard capacity and explain the relationship between appointment adherence and detention.
- Build a cost-to-serve model by channel and order profile.

## 3. Mental model

### 3.1 The warehouse as a flow system

```
receiving ──► putaway ──► storage ──► picking ──► packing ──► shipping
    │            │           │           │           │           │
   ASN,         slotting   inventory   pick path   cartonize   manifest
   dock         rules      accuracy    strategy    materials   and load
    │            │           │           │           │           │
    └────────────┴───────────┴───── returns ◄────────┴───────────┘
```

Every box on that line is a queue with a service rate. The building's throughput is the throughput of its slowest station, not the average of its stations.

### 3.2 The capacity identity

```
throughput (orders/hour) = available labor hours × productivity (orders/hour) × utilization
```

Capacity is not a single number. It is a curve that degrades as utilization rises, because queues grow non-linearly. A building at 95% utilization is not 5% better than one at 90%; it is unstable.

### 3.3 Utilization and waiting

```
waiting time grows roughly as ρ / (1 − ρ)
ρ = arrival rate ÷ service capacity
```

| Utilization | Relative waiting | Practical reading |
|---:|---:|---|
| 50% | 1× | Comfortable; absorbs variability |
| 75% | 3× | Normal operating target for variable demand |
| 85% | 6× | Peak mode; needs active management |
| 90% | 9× | Fragile; small demand blip causes a backlog |
| 95%+ | 19×+ | Effectively unstable without overflow capacity |

## 4. Core concepts

### 4.1 Process flows and the WMS data model

| Process | What happens | Data created | Typical failure |
|---|---|---|---|
| Receiving | Truck arrives, unload, count, inspect | ASN match, receipt, discrepancy | Blind receiving; wrong pallet count |
| Putaway | Move stock to a storage location | Putaway task, location assignment | Slotting ignores velocity or cube |
| Storage | Hold inventory by LPN or location | Inventory record, lot, expiry | Location and system disagree |
| Picking | Retrieve items for an order or wave | Pick task, pick confirmation | Short picks, wrong item |
| Packing | Cartonize, label, document | Carton, weight, label, packing list | Wrong carton, DIM weight surprise |
| Shipping | Stage, load, manifest | Shipment, BOL, SSCC | Missed cutoff, wrong route |
| Returns | Receive, inspect, disposition | Return authorization, disposition | Unlogged returns inflate inventory |

A minimal WMS data model:

| Entity | Grain | Key fields |
|---|---|---|
| Item master | SKU | Dimensions, weight, case pack, shelf life, handling class |
| Location | Location ID | Aisle, bay, level, position, type, capacity |
| License plate (LPN) | Pallet or carton | LPN ID, item, quantity, lot, expiry, location |
| Inventory | LPN × location | On-hand, allocated, available, status |
| Order | Order header | Customer, channel, service level, cutoff |
| Order line | Order × SKU | Quantity, unit of measure, substitution rule |
| Wave | Wave | Release time, orders, priority, carrier cutoff |
| Task | Task | Type, from, to, assignee, start, end, standard time |
| Receipt / ASN | Shipment | Supplier, PO, expected quantity, arrival |
| Shipment | Carton or pallet | SSCC, carrier, service, weight, manifest |

The task table is the analytics goldmine: it carries the timestamps and the standard time against which every labor and cycle-time question is answered.

### 4.2 Productivity metrics

| Metric | Formula | What it measures | Trap |
|---|---|---|---|
| UPH | units picked ÷ labor hours | Raw handling speed | Ignores travel, order mix, and quality |
| Lines per hour | order lines ÷ labor hours | Picking productivity across SKUs | A line of one unit equals a line of 50 |
| Orders per hour | orders ÷ labor hours | End-to-end throughput | Mix-sensitive; a 1-line order is not a 20-line order |
| Dock-to-stock | receipt time to available-to-promise | Inbound responsiveness | Excludes putaway backlog sitting in staging |
| Order cycle time | order release to shipment | Customer-facing speed | Averages hide the p95 that customers feel |
| Perfect order rate | orders complete, on time, undamaged, documented ÷ orders | End-to-end quality | The definition must be published |
| Inventory accuracy | locations matching system ÷ locations counted | Data integrity | Cycle-count sampling can flatter |
| Cost per order / line / unit | total operating cost ÷ denominator | Unit economics | Allocation method changes the ranking |

A useful rule: **report at least one speed, one quality, and one cost metric together.** UPH alone rewards speed that creates errors; perfect order rate alone can be bought with unlimited labor.

### 4.3 Little's Law and queueing

Little's Law: for a stable system, `L = λ × W` — work in process equals throughput times cycle time.

Worked example. A fulfillment center has 1,200 orders in process and ships 400 orders per hour.

```
W = L / λ = 1,200 / 400 = 3.0 hours
```

If the service promise requires a 2-hour cycle time, the maximum WIP is `400 × 2 = 800` orders. The gap is a release-control problem, not a picking-speed problem: releasing more orders makes every order slower.

Queueing. A pick area with 24 pickers, arrivals of 300 lines per hour, and a mean pick time of 4.2 minutes per line:

```
service rate per picker μ = 60 / 4.2 = 14.29 lines/hour
offered load a = λ / μ = 300 / 14.29 = 21.0
utilization ρ = a / c = 21.0 / 24 = 0.875
```

The Erlang C formula gives a 42.5% probability that an arrival waits, a mean queue wait of 0.6 minutes, and a mean time in system of 4.8 minutes. Twenty-four pooled servers absorb variability well even at 87.5% utilization.

Contrast a pack station with 6 packers and 6-minute service:

| Arrivals | Utilization | Mean wait | Mean time in system |
|---:|---:|---:|---:|
| 30/hour | 50% | 0.2 min | 6.2 min |
| 54/hour | 90% | 7.4 min | 13.4 min |

An 80% increase in volume produces a 37-fold increase in waiting. This is why peak season fails suddenly rather than gradually.

### 4.4 Labor standards and engineered standards

An engineered standard is the time a qualified worker should take to perform a task at a defined performance level with defined allowances.

```
standard time = observed time × performance rating × (1 + allowance)
```

| Element | Typical value | Note |
|---|---|---|
| Performance rating | 0.85–1.15 | Trained observer; subjective and contestable |
| Personal and fatigue allowance | 10–15% | Contract and climate dependent |
| Delay allowance | 5–10% | Covers unavoidable waiting |
| Learning curve | 80–85% on new tasks | Time falls with cumulative volume; do not standardize day one |

Predetermined motion time systems — MTM-1, MTM-2, MOST — build standards from elemental motions rather than a stopwatch. They are defensible and expensive; stopwatch time study is faster and more disputed. Whichever is used, the standard must be **auditable**: the element breakdown, the rating, the allowance, and the effective date.

Three governance rules: never use standards as a disciplinary tool without an appeal process; never compare productivity across buildings with different order mixes without normalizing; and never let a standard drift for years without re-timing the task.

### 4.5 Slotting and ABC

ABC classification by pick frequency: typically about 20% of SKUs generate about 80% of picks. Slot A items in the golden zone (waist to shoulder height, closest to the dispatch point), B in the mid-zone, C in the least accessible locations.

Worked travel comparison. A 12-line order, travel speed 1.2 m/s, 8 seconds of handling per line.

| Policy | Mean round-trip travel per line | Travel time | Handling | Total per order |
|---|---:|---:|---:|---:|
| Random storage | 140 m | 23.3 min | 1.6 min | 24.9 min |
| ABC slotting | 90 m | 15.0 min | 1.6 min | 16.6 min |

The saving is 8.3 minutes per order, a 33% reduction. At 400 orders per shift, that is 55.6 labor-hours, worth about $1,556 at $28/hour fully loaded. Slotting is usually the cheapest large productivity win available, and it decays: re-slot quarterly or whenever the pick-frequency distribution shifts.

Two refinements matter. **Affinity slotting** places items frequently picked together in nearby locations, which helps batch and zone picking. **Cube-aware slotting** avoids placing bulky A items in positions that block access to other fast movers.

### 4.6 Picking strategies

| Strategy | How it works | Best for | Weakness |
|---|---|---|---|
| Discrete | One picker, one order | Large orders, high-touch | Low density; long travel |
| Batch | One picker, several orders at once | Many small orders, similar items | Sortation at the end |
| Zone | Picker owns a zone; order travels | Large catalogs, dense pick faces | Handoff delays and balance |
| Wave | Orders released in batches by cutoff | Carrier schedules, wave planning | Peaks within the wave |
| Cluster | Picker takes a cart with multiple order containers | E-commerce, small items | Cart capacity and congestion |

The strategy must match the order profile. A building with 1.4 lines per order and 200 orders per wave should batch; a building with 40-line pallet orders should pick discrete or pallet-pick.

### 4.7 Pick path and travel time

Travel is typically 40–60% of pick time. Path heuristics inside a rectangular layout:

| Policy | Rule | Quality | When |
|---|---|---|---|
| S-shape | Traverse every aisle containing a pick, alternating direction | Good when picks are dense | Dense pick lists |
| Return | Enter and exit each aisle from the same cross-aisle | Simple | Sparse picks in wide aisles |
| Largest gap | Skip the largest unpicked segment of each aisle | Better than S-shape on sparse lists | Sparse, narrow aisles |
| Optimal (TSP) | Solve the aisle-level routing problem | Best | High-value picking, few aisles |

The practical result: path optimisation buys 5–15%; layout and slotting buy 20–40%. Optimise the layout before the path.

### 4.8 Capacity and discrete-event simulation

Analytical queueing gives fast answers for a single station. A discrete-event simulation (DES) is needed when the process has routing, batching, shared resources, shift breaks, and time-varying arrivals.

The SimPy model below represents a pick station with three pickers, Poisson arrivals at 3 per hour, exponential service with a 45-minute mean, and an 8-hour shift.

```python
"""Pick-station capacity model. Install: pip install simpy"""
import random
import statistics
import simpy

SEED, ARRIVAL_PER_MIN, PICKERS, MEAN_PICK_MIN, SHIFT_MIN = 42, 0.05, 3, 45.0, 480

def order(env, station, rng, waits):
    arrived = env.now
    with station.request() as req:
        yield req
        waits.append(env.now - arrived)
        yield env.timeout(rng.expovariate(1.0 / MEAN_PICK_MIN))

def arrivals(env, station, rng, waits):
    while True:
        yield env.timeout(rng.expovariate(ARRIVAL_PER_MIN))
        env.process(order(env, station, rng, waits))

env = simpy.Environment()
station = simpy.Resource(env, capacity=PICKERS)
rng = random.Random(SEED)
waits = []
env.process(arrivals(env, station, rng, waits))
env.run(until=SHIFT_MIN)

waits.sort()
print(f"orders started {len(waits)}")
print(f"mean wait {statistics.mean(waits):.1f} min")
print(f"p95 wait  {waits[int(0.95 * len(waits)) - 1]:.1f} min")
print(f"max wait  {waits[-1]:.1f} min")
```

Check the model against theory before trusting the complications. The analytical M/M/c result for the same parameters is 75% utilization and a 34-minute mean wait, so a correct simulation should produce a similar mean over a long run. Run it for a full shift, repeat with ten seeds, and report the spread before adding breaks, peaks, and rework.

Add realism one layer at a time: a lunch break that removes a server for 30 minutes, a peak hour at triple the arrival rate, a 5% rework probability, and a second station downstream. Validate each addition against an observed metric, not against intuition.

| Use analytical queueing | Use simulation |
|---|---|
| One station, stable arrivals, staffing decision | Multiple stations, routing, batching, breaks |
| Fast sensitivity on utilization | Peak-day and shift-level behaviour |
| Rough capacity check | Investment case with equipment uptime |
| Teaching the intuition | Validating a design before construction |

### 4.9 Automation

| Technology | What it does | Throughput profile | Capex | Best fit |
|---|---|---|---|---|
| Unit-load AS/RS | Crane stores and retrieves pallets | High, steady | Very high | Dense pallet storage, low land |
| Mini-load AS/RS | Totes and cartons in high-bay | Medium–high | High | Small items, high SKU count |
| Shuttle system | Robotic shuttles in dense racking | High | High | Case and tote buffer |
| AMR goods-to-person | Robots bring pods or shelves to stations | Medium, scalable | Medium | E-commerce, variable demand |
| AGV | Fixed-path transport | Steady | Medium | Pallet movement, repeatable routes |
| Pick-to-light / put-to-light | Directs picker by light | Medium | Low–medium | Dense pick faces, error reduction |
| Voice picking | Directs picker by speech | Medium | Low | Hands-free, refrigerated |
| Robotic picking arm | Picks items into totes | Emerging | High | Predictable, graspable items |

Automation economics:

```
payback (years) = capex ÷ (labor savings + throughput margin − maintenance − software − energy)
```

Worked example. A goods-to-person system costs $8.5 million installed, displaces 40 full-time equivalents at $52,000 fully loaded ($2.08 million), costs $350,000 per year in maintenance and software, and enables $400,000 of incremental margin through higher throughput.

```
net annual benefit = 2.08M − 0.35M + 0.40M = $2.13M
payback = 8.5M / 2.13M = 4.0 years
```

Now stress it: if uptime is 92% instead of 98%, throughput falls and the labor saving is partly eaten by overtime; if volume is 20% below plan, the displaced labor is not fully realized; if the labor rate is $38,000 rather than $52,000, the payback stretches beyond six years. **Automation business cases fail on the denominator, not the capex.**

Three decision rules: automate a process that is stable and high-volume, not one that is about to be redesigned; keep a manual overflow path for peak; and require the vendor to commit to a throughput number with a defined SKU mix, because "up to 1,000 units per hour" is a marketing statement.

### 4.10 Dock and yard

| Constraint | Symptom | Analytics |
|---|---|---|
| Door capacity | Trucks wait at the gate | Appointment slotting against door-hours |
| Unload rate | Detention charges | Time study by load type |
| Trailer pool | Drop-and-hook shortage | Pool sizing from turn time |
| Yard space | Trailer dwell, rehandling | Yard slot model |
| Cutoff clustering | Everything arrives 06:00–09:00 | Slot spreading with carrier incentives |

A dock at 85% booked utilization behaves like any queue at 85%: waiting rises sharply. Appointment systems that book to the mean unload time, ignoring the variance between a live unload and a drop, systematically over-book and then pay detention.

### 4.11 Fulfillment network and micro-fulfillment

| Model | Description | Cost profile | Best for |
|---|---|---|---|
| Central DC + parcel | One or few large buildings | Low storage cost, high last-mile | Broad catalog, national reach |
| Regional DCs | Several mid-size buildings | Balanced | Dense regional demand |
| Ship-from-store | Store acts as a node | Low capex, high pick cost, disrupts store | Fast delivery in cities |
| BOPIS | Buy online, pick up in store | Low delivery cost, labor in store | Grocery, convenience |
| Micro-fulfillment center | Small automated facility in a city | High automation capex per m², low last mile | Grocery, same-day, dense demand |
| Dark store | Retail-like store closed to public | Moderate capex | Urban same-day |

Micro-fulfillment economics depend on three variables: order density (orders per km² per day), real estate cost, and the labor rate of the alternative (usually store picking). A MFC that costs $3 million and handles 1,000 orders a day needs roughly 300–400 orders per day just to cover its fixed cost before labor savings; below that, store picking is cheaper.

### 4.12 Cost-to-serve

| Order profile | Lines/order | Units/line | Pick cost | Pack cost | Ship cost | Return rate | Cost/order |
|---|---:|---:|---:|---:|---:|---:|---:|
| Store replenishment (pallet) | 40 | 60 | $16.80 | $2.00 | $180.00 | 1% | $198.86 |
| Wholesale case | 12 | 24 | $5.04 | $1.60 | $38.00 | 3% | $44.83 |
| E-commerce parcel | 2.1 | 1.3 | $0.88 | $1.10 | $4.80 | 8% | $7.28 |

Assumptions: pick $0.42 per line, pack as shown, ship as shown, return cost $6.20 per return. Arithmetic check for e-commerce: `2.1 × 0.42 = 0.882`; `+1.10 + 4.80 + 0.08 × 6.20 = 0.496`; total `$7.28`.

The lesson is not the absolute numbers but the shape: a parcel order costs about 1/27th of a pallet order to serve but carries a fraction of the revenue. Channel profitability requires revenue per order alongside cost per order; cost alone will always make small orders look cheap.

### 4.13 Safety and ergonomics

Safety is a leading indicator of operational quality, not a compliance afterthought. Two rates are standard:

```
recordable incident rate = (recordable cases × 200,000) ÷ hours worked
DART rate = (days away, restricted, or transferred cases × 200,000) ÷ hours worked
```

The 200,000 constant normalizes to 100 full-time-equivalent workers at 40 hours for 50 weeks. Rates are only comparable when the case definition, the hours denominator, and the facility mix are published.

| Mechanism | Where it concentrates | Control |
|---|---|---|
| Musculoskeletal strain | Picking, packing, unloading | Load limits, mechanical aids, rotation, workstation height |
| Slips and falls | Docks, mezzanines, wet areas | Housekeeping, guarding, footwear, lighting |
| Powered truck incidents | Dock, cross-aisles, blind corners | Pedestrian segregation, speed limits, training, telematics |
| Struck-by falling objects | High racking | Rack inspection, netting, stacking rules |
| Heat stress | Non-climate-controlled buildings | WBGT monitoring, hydration, adjusted standards |

Leading indicators — near-miss reporting rate, safety observation completion, ergonomic assessment coverage, overtime intensity, and turnover — predict the lagging rates weeks earlier. A building with rising overtime and falling tenure is a safety risk before it is an injury statistic.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Task transaction | Task | WMS | Type, from/to location, assignee, start, end, quantity |
| Inventory snapshot | LPN × location × time | WMS | Consistent snapshot time; status codes |
| Receipt | ASN / PO | WMS / ERP | Expected vs received; discrepancy reason |
| Order | Order × line | OMS / ERP | Channel, cutoff, service level |
| Shipment | Carton / pallet | WMS / TMS | SSCC, weight, dims, carrier |
| Labor | Employee × shift × task | Time and attendance, WMS | Hours by indirect and direct work |
| Slotting | SKU × location | WMS slotting module | Velocity class, cube, constraints |
| Safety | Incident | EHS system | Type, severity, hours worked, body part |

**Standards.** GS1 GTIN for items, SSCC for pallets and cartons, GLN for locations, EPCIS 2.0 for event capture; ANSI X12 940 (warehouse shipping order) and 945 (shipping advice); ISO 45001 for occupational health and safety management; OSHA 29 CFR 1910.176 for materials handling and storage; ANSI/ITSDF B56.1 for powered industrial trucks; ASCM SCOR DS for the order-fulfillment metrics.

**Data-quality checks.** Every task has a start and end timestamp; indirect work (meetings, cleaning, waiting) is coded separately from direct work; units of measure are explicit and case packs are versioned; location IDs are stable and never reused across layouts; returns are recorded as a distinct inventory status, not as negative receipts.

**Workflow.** Measure the baseline by process step, find the constraint, model the capacity, test the intervention on a peak day, then pilot. A slotting change can be piloted in one zone; an automation decision cannot.

## 6. Industry case studies

### 6.1 Amazon warehouse robotics (2025)

**Context.** Amazon operates one of the largest warehouse networks in the world and has automated storage, transport, and picking progressively since the 2012 acquisition of Kiva Systems.

**Intervention.** By July 2025 Amazon reported more than one million warehouse robots and introduced DeepFleet, a generative-AI model for coordinating robot fleets, plus Vulcan, a robot with force-feedback touch that the company says handles about 75% of its item types at human-comparable speed.

**Evidence.** Company announcements and press reporting; throughput and safety outcomes are not independently audited. The claimed 75% item coverage is a company figure.

**Outcome.** Higher storage density and shorter associate travel in equipped buildings, with a workforce redeployed toward roles the robots do not perform.

**What transferred.** The incremental deployment model — automate one process, measure, then expand — and the principle that robots change the shape of labor rather than eliminating it.

**What did not.** The scale economics. A 200,000-order-per-day network can justify custom robotics; a 5,000-order-per-day operation cannot, and vendor case studies built on Amazon-like volumes mislead.

### 6.2 Ocado On-Grid Robotic Pick (2024–2025)

**Context.** Ocado operates highly automated grocery fulfillment centers built around a grid of bots that retrieve totes to stations, where humans pick items into customer bags.

**Intervention.** On-Grid Robotic Pick places robotic arms at the grid to pick items directly from totes, using machine vision and suction or gripper end effectors.

**Evidence.** Ocado reported that its robotic arms picked more than 30 million items in 2024 with fewer than 100 arms, that about 40% of groceries were packed robotically, and that it targets about 80%. These are company figures; item-level success rates by category are not published in detail.

**Outcome.** A measurable shift of picking labor to machine work in a controlled, high-density environment, with humans retained for items the robots cannot handle.

**What transferred.** The 80/20 approach: automate the items that are graspable and predictable, keep manual stations for the rest, and design the handoff explicitly.

**What did not.** The generality. Grocery is a favorable case — regular packaging, high volume, dense grid — and the result does not transfer to a catalog with soft goods, odd shapes, and low velocity.

### 6.3 Takeoff Technologies and micro-fulfillment consolidation (2024)

**Context.** Takeoff Technologies built automated micro-fulfillment centers for grocers, typically sited inside or beside a store, with the promise of automated grocery picking close to the customer.

**Intervention.** Robotic grocery picking in small-footprint automated facilities, sold to grocery chains as a turnkey system.

**Evidence.** Takeoff filed for Chapter 11 bankruptcy protection in 2024 after raising substantial venture capital; grocery trade press reported that deployments were fewer and unit economics harder than planned. [VERIFY: filing date and jurisdiction before citing.]

**Outcome.** The assets were sold and the independent MFC model contracted, while some grocers continued with in-house or partner automation.

**What transferred.** The constraint is economics, not technology: an MFC needs high order density, expensive urban labor, and a narrow, high-velocity assortment to pay back.

**What did not.** The assumption that automation cost per order falls enough at moderate volume. At a few hundred orders a day, the fixed cost per order dominates.

### 6.4 Amazon warehouse injury reporting (2024–2025)

**Context.** Amazon's injury rates at US warehouses became a public policy issue, with allegations that productivity targets drove musculoskeletal injuries.

**Intervention.** A US Senate committee majority staff investigation published a report in December 2024 examining injury rates and productivity practices; Amazon disputed the methodology and published its own safety data. OSHA separately cited several Amazon warehouses in 2022–2024. [VERIFY: report title, exact date, and citation details.]

**Evidence.** Contested. The committee's analysis and Amazon's rebuttal use different denominators (recordable incidents per 200,000 hours vs cases per 100 workers) and different facility scopes.

**Outcome.** Public scrutiny, state-level warehouse quota legislation, and changes to some facility practices.

**What transferred.** The measurement lesson: injury rates are exquisitely sensitive to the denominator, the case definition, and the facility mix. Any safety analytics must publish all three.

**What did not.** A simple causal claim that quotas cause injuries. The evidence supports association and mechanism, not a clean causal estimate.

### 6.5 Order picking as the cost center (De Koster, Le-Duc, Roodbergen, 2007)

**Context.** Warehouse automation was being sold without a clear map of where manual cost actually sits.

**Intervention.** A systematic literature review of order picking design and control, covering layout, routing, storage assignment, and batching.

**Evidence.** The review reports that order picking accounts for roughly 55% of warehouse operating cost and that travel is the dominant component of pick time, with routing and storage assignment the main levers.

**Outcome.** A durable framework that still organizes warehouse analytics: layout, storage assignment, routing, and batching are the four decisions.

**What transferred.** The prioritization: fix layout and slotting before buying robots; measure travel before optimising the pick.

**What did not.** The exact 55% figure, which varies by facility type, automation level, and cost boundary. Use it as an order-of-magnitude anchor, not a benchmark.

## 7. Labs

### Lab 13.1 — Little's Law and queue-based capacity

**Objective.** Use Little's Law and M/M/c queueing to size staffing and show the nonlinear cost of utilization.

**Data.** Course-provided task timestamps from a pick and pack area over 8 weeks, plus a synthetic arrival generator for stress cases.

**Steps.** (1) Compute WIP, throughput, and cycle time by hour and verify `L = λW`. (2) Estimate arrival rate and service time by area. (3) Compute Erlang C waiting time for the current staffing. (4) Build a staffing curve for a 30% peak, showing wait by headcount. (5) Identify the headcount at which p95 wait exceeds the service target. (6) Compare the staffing cost of meeting the target with the overtime cost of missing it.

**Deliverable.** A capacity memo with the utilization-wait curve and a staffing recommendation.

**Acceptance criteria.** Little's Law verified numerically; arrival and service estimates documented; Erlang C implemented and checked against simulation; recommendation states the service target and its cost.

**Stretch.** Add a second, downstream station and show how the bottleneck moves when the first is staffed up.

### Lab 13.2 — SimPy model of a peak day

**Objective.** Build and validate a discrete-event model of a fulfillment area under peak conditions.

**Data.** Lab 13.1 timestamps plus a peak-week order arrival profile; shift calendar and break schedule.

**Steps.** (1) Reproduce the base day in SimPy and match observed throughput within 5%. (2) Add breaks, a 3× peak hour, and 5% rework. (3) Compare throughput and p95 cycle time with the analytical model. (4) Test three interventions: add a picker, add a pack station, reduce rework. (5) Report which intervention buys the most throughput per dollar.

**Deliverable.** A validated SimPy model, a validation table, and an intervention comparison.

**Acceptance criteria.** Base case validated against observed data; random seed and run length stated; confidence intervals or multiple replications reported; interventions compared on the same demand profile.

**Stretch.** Add a second building and model a transfer of orders between them when one backs up.

### Lab 13.3 — Slotting and travel time

**Objective.** Quantify the travel-time and labor value of an ABC slotting change.

**Data.** 12 months of pick transactions with SKU, location, and timestamp; warehouse layout with aisle and bay coordinates.

**Steps.** (1) Compute pick frequency and cube per SKU and assign ABC classes. (2) Compute current travel distance with an S-shape path model. (3) Re-slot A items to the golden zone and recompute travel. (4) Convert distance to time and labor cost. (5) Test an affinity rule for co-picked items. (6) Estimate the labor hours required to re-slot and the payback.

**Deliverable.** A slotting analysis with before/after travel, cost, and a re-slot plan.

**Acceptance criteria.** ABC cut-offs stated; distance model documented; time conversion uses observed speed; labor cost of the change included; re-slot frequency recommended.

**Stretch.** Add a cube constraint that prevents moving all A items into the golden zone and solve the constrained assignment.

### Lab 13.4 — Automation business case

**Objective.** Build an automation investment case that survives sensitivity analysis.

**Data.** Vendor throughput and cost quotes (course-provided), current labor rates, order profile, and uptime assumptions.

**Steps.** (1) Compute current cost per order and per unit. (2) Build the annual benefit from labor, throughput margin, and quality. (3) Subtract maintenance, software, energy, and floor-space change. (4) Compute payback and NPV at a stated discount rate. (5) Run a tornado sensitivity on labor rate, volume, uptime, and capex. (6) State the break-even volume below which the case fails.

**Deliverable.** A business case with a tornado chart and a break-even statement.

**Acceptance criteria.** Benefits and costs both itemized; uptime explicit; NPV with discount rate; break-even volume computed; vendor claims labelled as claims.

**Stretch.** Add a phased deployment option and compare its NPV with the single-phase case.

## 8. Assignment — Fulfillment capacity model

Produce a **capacity and labor model** for one building and one peak season. It must answer how many orders the building can ship on the peak day, where the bottleneck is, and what the cheapest capacity increment is.

| Section | Content |
|---|---|
| 1. Scope | Building, channels, order profile, peak definition |
| 2. Baseline | Throughput, UPH, lines/hour, dock-to-stock, cycle time, cost per order |
| 3. Process map | Flow, staffing, task times, and queue points |
| 4. Capacity model | Little's Law, queueing, and simulation results |
| 5. Bottleneck | Evidence for the constraint and its sensitivity |
| 6. Peak plan | Staffing curve, overtime, temporary labor, overflow options |
| 7. Slotting | ABC analysis and travel-time opportunity |
| 8. Automation option | Business case with payback, NPV, and break-even |
| 9. Recommendation | Capacity increment, cost, risk, and implementation timing |
| 10. Monitoring | Metrics, cadence, and trigger thresholds |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Process and data framing | 15% | Lists processes without data | Process map with task times | Task-level data model with quality caveats |
| Capacity analysis | 25% | Single average throughput | Little's Law and queueing applied | Validated simulation, bottleneck proven, sensitivity |
| Labor and standards | 20% | Headcount from rules of thumb | Standards with allowances | Auditable standards, mix-normalized comparison, governance |
| Slotting and layout | 15% | ABC list only | Travel-time calculation | Constrained re-slot with payback and re-slot cadence |
| Automation case | 15% | Vendor payback quoted | Itemized benefits and costs | Tornado sensitivity, break-even volume, phased option |
| Cost-to-serve and communication | 10% | Cost per unit only | Cost by order profile | Channel profitability with revenue context and decision memo |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Occupational safety | OSHA general duty and 29 CFR 1910.176; ISO 45001 management system | Report recordable rate and DART with denominators; investigate near-misses |
| Productivity quotas | Several jurisdictions regulate warehouse quotas and disclose them to workers | Publish standards; include allowances; provide appeal |
| Ergonomic injury | Repetitive lifting and reaching are the dominant injury mechanism | Job rotation, mechanical aids, ergonomic assessment |
| Temperature and conditions | Heat illness rules in some jurisdictions | Monitor WBGT; adjust standards for environment |
| Worker surveillance | Cameras, wearables, and location tracking raise privacy issues | Minimise data; consult works councils; set retention limits |
| Temporary labor | Agency workers may receive less training and face higher injury risk | Equal safety induction; track injury rate by employment type |
| Accessibility | Pack stations and equipment must accommodate the workforce | Universal design; reasonable accommodation |
| Automation and displacement | Workforce impact is a commercial and social decision | Redeployment plan; consultation; transparent communication |

Two obligations: **never report a productivity gain without the quality and safety consequence**, and **never compare buildings without normalizing the order mix**. Both omissions are common and both mislead the investment decision.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Average-day design | Peak plan is a multiplier of the mean | Variance ignored | Peak collapse | Model the peak day explicitly |
| Utilization too high | Overtime every week | Queueing nonlinearity | Backlog and errors | Staff to 75–85%; overflow plan |
| Standards without allowances | Fatigue and injury rise | Unrealistic standard | Grievances, turnover | Allowance study; appeal process |
| Slotting never refreshed | Travel time creeps up | Assortment drift | Silent productivity loss | Quarterly re-slot |
| Automation on unstable process | Vendor throughput never reached | Process redesign after install | Stranded capex | Stabilize, then automate |
| Vendor throughput claim accepted | "Up to" numbers in the model | No SKU mix or uptime commitment | Payback miss | Contractual throughput test |
| WIP unbounded | Cycle time rises as volume rises | Release control absent | Missed cutoffs | Cap WIP; pull release |
| Returns unmodelled | Inventory accuracy falls | Returns booked as negative receipts | Wrong ATP, wrong slotting | Distinct returns process and status |
| Dock over-booked | Detention climbs | Booking to mean service time | Cost and carrier friction | Slot by service-time distribution |
| Metrics gamed | UPH up, perfect order down | Single-metric incentive | Customer harm | Balanced scorecard |
| Safety rate misreported | Denominator changes between periods | Definition drift | False reassurance | Publish case definition and hours |
| Micro-fulfillment at low density | Fixed cost per order high | Capex spread over too few orders | Negative unit economics | Density test before investment |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | State Little's Law and use it on a warehouse. | L = λW; WIP equals throughput times cycle time; use it to cap order release. |
| 2 | Why does a 90%-utilized pack line fail at peak? | Waiting grows roughly as ρ/(1−ρ); variance compounds; 90% is fragile. |
| 3 | How do you set a labor standard? | Time study or predetermined motions, performance rating, allowances, audit trail, re-timing. |
| 4 | What is the fastest warehouse productivity win? | Slotting and layout, usually 20–40% travel reduction, before any automation. |
| 5 | Discrete, batch, or zone picking — how do you choose? | By lines per order, order count, item similarity, and cutoff structure. |
| 6 | When does automation pay back? | Stable high volume, expensive labor, tight land, high uptime; payback sensitive to volume and labor rate. |
| 7 | Why did micro-fulfillment disappoint? | Fixed cost per order at low density; technology worked, economics did not. |
| 8 | How do you measure warehouse safety honestly? | Recordable rate and DART with hours worked, case definitions, and facility mix published. |
| 9 | Dock utilization is 92%. What happens? | Waiting and detention rise sharply; appointment system is over-booked relative to service-time variance. |
| 10 | UPH improved 15% but cost per order rose. How? | Mix shift, rework, overtime, or quality loss; UPH alone is not unit economics. |
| 11 | How do you validate a SimPy model? | Reproduce an observed throughput and cycle time within tolerance; then add complexity one layer at a time. |
| 12 | What is cost-to-serve and why does it matter? | Cost by order profile and channel; small orders look cheap but carry little revenue. |

## 13. Key terms

| Term | Definition |
|---|---|
| WMS | Warehouse management system; controls inventory, tasks, and locations |
| WES / WCS | Warehouse execution and control systems; coordinate automation |
| LPN | License plate number; unique identifier for a pallet or carton |
| ASN | Advance ship notice; expected receipt details |
| Putaway | Moving received stock to a storage location |
| Slotting | Assigning SKUs to locations by velocity, cube, and affinity |
| ABC classification | Grouping SKUs by pick frequency or value |
| Golden zone | Waist-to-shoulder storage area with the least bending and reaching |
| Discrete picking | One picker completes one order at a time |
| Batch picking | One picker collects for several orders in one pass |
| Zone picking | Each picker works a defined area; orders move between zones |
| Wave picking | Orders released in batches aligned to cutoffs |
| Cluster picking | A picker uses a cart with multiple order containers |
| Travel time | Time spent moving between locations; often 40–60% of pick time |
| UPH | Units per hour |
| Lines per hour | Order lines picked per labor hour |
| Dock-to-stock | Time from receipt to available-to-promise |
| Order cycle time | Order release to shipment |
| Perfect order rate | Share of orders complete, on time, undamaged, documented |
| Little's Law | L = λW; WIP equals throughput times cycle time |
| Erlang C | Formula for waiting probability in a multi-server queue |
| DES | Discrete-event simulation |
| Engineered standard | Defined time for a task at a performance level with allowances |
| MOST / MTM | Predetermined motion time systems for setting standards |
| AS/RS | Automated storage and retrieval system |
| AMR | Autonomous mobile robot |
| Goods-to-person | Robots bring inventory to a stationary picker |
| Shuttle system | Robotic carriers inside dense racking |
| Micro-fulfillment center | Small automated urban facility for fast delivery |
| Cost-to-serve | Cost allocated to an order, customer, or channel |

## 14. Sources and further reading

**Standards and regulation**

- ISO 45001:2018, *Occupational health and safety management systems — Requirements with guidance for use*.
- OSHA 29 CFR 1910.176, *Materials handling and storage*. https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.176 — accessed 27 October 2026.
- ANSI/ITSDF B56.1, *Safety Standard for Low Lift and High Lift Trucks*.
- GS1, *GTIN, SSCC, GLN*, and *EPCIS 2.0*. https://www.gs1.org/standards — accessed 27 October 2026.
- ANSI ASC X12 940 (warehouse shipping order) and 945 (shipping advice). https://x12.org — accessed 27 October 2026.
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). https://www.ascm.org — accessed 27 October 2026.
- FEM 9.851, *Performance data of stacker cranes* — throughput calculation for AS/RS. [VERIFY: current edition and applicability before citing in a design document.]

**Peer-reviewed and methodological**

- De Koster, R., Le-Duc, T., and Roodbergen, K. J. "Design and Control of Warehouse Order Picking: A Literature Review." *European Journal of Operational Research* 182(2), 2007.
- Little, J. D. C. "A Proof for the Queuing Formula: L = λW." *Operations Research* 9(3), 1961.
- Hopp, W. J., and Spearman, M. L. *Factory Physics*, 3rd ed. Waveland Press, 2011.
- Bartholdi, J. J., and Hackman, S. T. *Warehouse and Distribution Science*, open-access textbook. https://www.warehouse-science.com — accessed 27 October 2026.
- Tompkins, J. A., et al. *Facilities Planning*, 4th ed. Wiley, 2010.
- Gu, J., Goetschalckx, M., and McGinnis, L. F. "Research on Warehouse Operation: A Comprehensive Review." *European Journal of Operational Research* 177(1), 2007.
- Boysen, N., de Koster, R., and Weidinger, F. "Warehousing in the E-Commerce Era: A Survey." *European Journal of Operational Research* 277(2), 2019.
- Grosse, E. H., Glock, C. H., and Neumann, W. P. "Human Factors in Order Picking: A Content Analysis of the Literature." *International Journal of Production Research* 55(5), 2017.

**Industry and company evidence**

- Amazon, "Amazon's New Vulcan Robot," 2025. https://www.aboutamazon.com/news/operations/amazon-vulcan-robot-pick-stow-touch — accessed 27 October 2026. Company claims.
- Amazon, European robotics investment announcement, 2025–2026. https://www.aboutamazon.com/news/operations/amazon-europe-robotics-delivery-investment — accessed 27 October 2026. Company claims.
- Ocado Group, robotic picking stories and results, 2024–2025. https://www.ocadogroup.com/newsroom/stories/ocado-robotic-arms — accessed 27 October 2026. Company claims: 30 million items, about 40% packed, target about 80%.
- U.S. Senate Committee on Health, Education, Labor and Pensions, majority staff report on Amazon warehouse injuries, December 2024. [VERIFY: exact title, date, and scope.]
- Grocery Dive and *Wall Street Journal* coverage of Takeoff Technologies' 2024 Chapter 11 filing. [VERIFY: filing date and court.]
- DHL and Boston Dynamics, Stretch robot deployment announcements, 2022–2024. Company claims on unloading productivity.

**Tools and datasets**

- SimPy documentation. https://simpy.readthedocs.io — accessed 27 October 2026.
- `simpy.Resource`, `simpy.Environment`, and `env.timeout` for queue models.
- OR-Tools for slotting and assignment problems. https://developers.google.com/optimization — accessed 27 October 2026.
- UCI Online Retail II for order-profile practice. https://archive.ics.uci.edu/dataset/502/online+retail+ii — accessed 27 October 2026.

> **Note:** Warehouse benchmarks are facility-specific. A UPH figure without order mix, automation level, and measurement definition is not comparable across buildings.

---

> **Next:** [Module 14](14_procurement_supplier_risk_spend.md) — the building ships what somebody bought, and the next cost lever sits upstream in the spend cube, the contract, and the supplier's balance sheet.
