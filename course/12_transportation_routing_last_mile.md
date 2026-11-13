# Module 12 — Transportation, Routing, and Last-Mile Analytics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (12 lecture, 10 lab, 8 project)
> **Track fit:** Logistics & Distribution core; recommended for Retail & E-commerce and Manufacturing & Industrial Operations
> **Prerequisites:** Module 01A (units and cost conventions); Module 02 (shipment data); Module 03 (event data and joins); Module 11 recommended; basic Python and linear programming
> **Tools:** Python 3.12, pandas, NumPy, OR-Tools, DuckDB, matplotlib, GeoPandas; optional OSRM or Valhalla for road-network distances, LightGBM for ETA models
> **Industry question:** "Which mode, route, and delivery promise buys the service the customer paid for at the lowest total cost per stop — and what happens to that answer when the route changes at 10 a.m.?"

---

## 1. Why industry cares

Transportation is where the invoice arrives every week and where the customer sees the failure directly. A forecast error becomes excess inventory; a routing error becomes a late delivery, a detention charge, a re-delivery, and a service call. For most distributors and e-commerce operators it is the largest controllable operating cost and the largest share of the carbon footprint.

Four facts define the 2026 context.

1. **Freight rates are volatile and dated.** As of October 2026, Freightos/FreightWaves reported Asia–USWC at about $8,126 per FEU, Asia–USEC about $9,700, Asia–North Europe about $3,442, and Asia–Med about $3,927, with carriers pushing general rate increases into 2027 contract tendering. A model that hard-codes a 2023 rate models the past.
2. **Routing is solved only in the textbook sense.** The vehicle routing problem is NP-hard, and the gap between a benchmark solution and a production dispatch decision is made of time windows, driver hours, access constraints, and re-optimization latency.
3. **Last mile is where the promise is made.** Delivery windows, route density, failed first attempts, and returns decide whether a delivery is profitable. Cost per stop falls sharply with density and rises steeply with window tightness.
4. **Emissions reporting became auditable.** ISO 14083:2023 and the GLEC Framework turned transport carbon into a calculation with a boundary, an allocation rule, and a factor dataset. A carrier's "green" claim without a methodology is not evidence.

The cost of getting this wrong shows up in the freight P&L (accessorial leakage, poor tender acceptance, spot exposure), the service P&L (failed deliveries, penalties, churn), and the carbon report (assumptions that do not survive assurance).

## 2. Learning outcomes

By the end of this module, you can:

- Compare freight modes on cost per tonne-kilometre, transit time, variability, and emissions, and state the break-even volume between them.
- Decompose a freight invoice into linehaul, fuel, accessorials, and minimum charges, and identify which components are controllable.
- Build a lane-level cost and service baseline from tender, shipment, and invoice data, and detect rate leakage.
- Formulate a capacitated vehicle routing problem with time windows (CVRPTW) as a mixed-integer program and explain each constraint.
- Implement and tune an OR-Tools routing model, including search strategy, time limit, and the difference between feasible and good.
- Decide when to re-optimize a route dynamically and when re-optimization costs more than it saves.
- Model last-mile cost per stop as a function of density, window width, and failure rate.
- Evaluate an ETA model at the promise (tail lateness, promise-keeping) rather than on average error.
- Design a carrier scorecard and a tender-acceptance analysis that separates carrier behaviour from lane structure.
- Schedule dock appointments against door capacity and show the queueing consequence of over-booking.
- Apply Incoterms 2020 correctly to landed-cost and risk-transfer questions.
- Calculate transport emissions under ISO 14083 and compare a mode shift on cost and carbon.

## 3. Mental model

### 3.1 The transport decision stack

```
network design ──► mode and service ──► consolidation ──► route ──► dispatch ──► delivery
 (Module 11)        contract or spot      cube, weight     VRP/VRPTW  re-optimize   ETA/POD
        └──────────── cost per stop / per tonne-km / per order ────────────────────┘
```

Each layer constrains the one below. A route optimizer cannot fix a network with the wrong number of DCs, and dynamic dispatch cannot fix a promise that was never feasible.

### 3.2 The three cost currencies

| Currency | Formula | Used for | Trap |
|---|---|---|---|
| Cost per stop | total route cost ÷ stops served | Last mile, route design | Ignores weight and cube; a 5 kg stop is not a 500 kg stop |
| Cost per tonne-km | cost ÷ (tonnes × km) | Mode comparison, linehaul | Punishes small shipments the customer demanded |
| Cost per order | total transport cost ÷ orders shipped | Customer profitability | Hides the density effect inside an average |

Never compare modes on cost per shipment without stating weight, distance, and service level. A parcel and a truckload are not substitutes at the same volume.

### 3.3 Service-cost frontier

```
 cost per stop
     │                         ● 2-hour window, 1 attempt
     │                    ●
     │              ● 4-hour window
     │         ●
     │    ● next-day, 8-hour window
     │ ● 2-day, no window
     └──────────────────────────────────► service level
```

The frontier is real but not fixed: density improvement shifts the whole curve down. "Improve service without cost" is usually a density project, not a routing project.

## 4. Core concepts

### 4.1 Freight modes and their economics

| Mode | Typical use | Cost driver | Strength | Weakness |
|---|---|---|---|---|
| Full truckload (TL/FTL) | > 10,000 lb or > 12 pallets, one O-D | Miles, equipment, driver time | Lowest road cost per tonne-km; low damage | Pay for the whole truck; empty backhauls |
| Less-than-truckload (LTL) | 150–10,000 lb, shared trailer | Weight, class, distance, accessorials | No need to fill a truck | Terminal handling, longer transit, reweigh risk |
| Parcel | < 150 lb, residential and small business | Zone, dimensional weight, surcharges | Reach, tracking | DIM weight and surcharges dominate |
| Intermodal | > 500 miles, dense lanes, port hinterland | Ramp rate + drayage + chassis | Cheaper on long hauls, lower carbon | Longer, less flexible, ramp congestion |
| Ocean (FCL/LCL) | International, non-urgent | TEU/FEU rate, BAF, THC, D&D | Lowest cost per tonne-km | 20–45 day transit, unreliable schedules |
| Air | High value, perishable, urgent | Chargeable weight, fuel, security | Speed and reliability | 10–40× ocean cost; high emissions |
| Rail (carload) | Bulk, heavy, long distance | Car hire, switching, mileage | Very low cost per tonne-km | Minimum carload, poor for packaged goods |

**Industry interpretation.** Mode selection is a constrained cost problem. The binding constraint is usually not cost but the customer's required date and the inventory carrying cost of the transit time.

### 4.2 Worked comparison: cost per tonne-km

One tonne (1,000 kg) moving 800 km by road, plus ocean and air legs of 11,000 km. Rates are illustrative 2026 snapshots, not quotes.

| Mode | Assumed rate | Cost for 1,000 kg | Cost per tonne-km | Typical CO2e (g/tonne-km) |
|---|---|---:|---:|---:|
| LTL | 1,000 kg = 22.05 cwt at $32/cwt, +18% fuel | $832.61 | $1.041 | 120–150 |
| TL (allocated) | $2.10/mile × 500 miles for a 20,000 lb truck | $115.74 | $0.145 | 80–100 |
| Intermodal | $1,800 per 20,000 kg box, 800 km | $90.00 | $0.113 | 20–35 |
| Ocean (FCL) | $8,126/FEU, 20,000 kg payload, 11,000 km | $406.30 | $0.037 | 10–20 |
| Air | $4.50/kg chargeable, 11,000 km | $4,500.00 | $0.409 | 500–700 |

Arithmetic check: $832.61 ÷ (1 t × 800 km) = $1.041/t-km. $115.74 ÷ 800 = $0.145. $406.30 ÷ 11,000 = $0.037.

Three rules follow. **LTL is 7× the TL unit cost** at this distance because the trailer is shared and terminal handling is charged back; consolidate into a TL whenever you can fill roughly 60% of a trailer on the same lane-day. **Air is 11× ocean per tonne-km** and about 35× on carbon, justified only by value density, perishability, or a stockout whose margin loss exceeds the premium. **Intermodal wins on long, dense lanes** but loses the advantage when drayage and ramp dwell consume the linehaul saving.

### 4.3 Rate structures and accessorials

| Mode | Core rate | Surcharges that change the invoice | Controllable by the shipper? |
|---|---|---|---|
| TL | Rate per mile, minimum charge | Fuel surcharge (diesel-index peg), detention, layover, stop-off, lumper, reefer fuel, tolls, hazmat | Detention and stop-offs: yes. Fuel: hedged, not controlled |
| LTL | Rate per cwt by NMFC class, minimum charge | Fuel, liftgate, residential, inside delivery, appointment, reweigh/reclass, cubic capacity | Reclass and appointment fees: yes, via packaging and data accuracy |
| Parcel | Zone rate by weight and zone | Dimensional weight, residential, additional handling, oversize, address correction, peak surcharge | DIM weight and address quality: yes |
| Ocean | FAK or commodity rate per container | BAF/EIS, THC, documentation, ISPS, peak season, demurrage, detention, chassis | D&D and documentation: yes |
| Air | Rate per kg chargeable weight | Fuel, security, screening, dangerous goods | Packaging density: yes |

Two distinctions cause most audit disputes. **Demurrage vs detention (ocean):** demurrage is charged while the container sits inside the terminal beyond free time; detention while it sits outside the terminal, typically at the consignee. Different clocks, different owners. **Chargeable vs actual weight:** chargeable weight is the greater of actual and volumetric; air uses a 1:6,000 m³/kg convention (some carriers 1:5,000), parcel uses a DIM divisor (commonly 139 or 166 in³/lb). A packaging change that cuts cube reduces the invoice without renegotiation.

### 4.4 Lane and tender analytics

A **lane** is an origin–destination–equipment combination with a volume and a service requirement. A **tender** is the offer of a specific load to a carrier under contract or spot.

| Metric | Definition | Healthy range | What a bad value means |
|---|---|---|---|
| Tender acceptance rate | Accepted ÷ tendered loads | 90–98% on contracted lanes | Carrier is underpriced or capacity is short |
| Primary tender share | Loads accepted by primary ÷ loads | > 85% | Too much spot fallback; contract mispriced |
| Spot share of spend | Spot spend ÷ total transport spend | 5–15% in a stable market | Reactive procurement; budget volatility |
| Cost per mile vs benchmark | Lane rate ÷ DAT/Cass or internal benchmark | Within ±8% | Leakage or a structurally difficult lane |
| Accessorial share | Accessorials ÷ total invoice | < 8% | Bad master data, dock delays, weak audit |
| Invoice accuracy | Invoices matching contract ÷ invoices | > 97% | Rate engine or contract data wrong |

**Lane clustering.** Do not analyse 40,000 lanes individually. Cluster by distance band, weight band, and origin market, then examine the residual. A lane with 12 shipments a year has no reliable rate; it needs an external benchmark, not its own history.

### 4.5 Consolidation and cube utilisation

Consolidation is the highest-return, lowest-technology transport lever. Three tests decide whether it is available: **time fusion** (do the shipments share a dispatch window?), **physical fit** (weight and cube both within limits?), and **cost fusion** (does the combined shipment cost less than the sum of parts, including stop-off charges?).

```
cube utilisation   = occupied m³ ÷ usable trailer m³
weight utilisation = payload kg ÷ permitted payload kg
```

A trailer at 95% cube and 40% weight is cube-constrained and should be built by volume. A trailer at 60% cube and 98% weight is weight-constrained and should be built by mass. Optimising the wrong dimension produces loads that look full and cost more.

### 4.6 The vehicle routing problem

Capacitated VRP: customers with demand, a depot, and identical vehicles of capacity Q. Minimise travel cost subject to each customer being served once and no route exceeding Q.

```
minimise   Σ_{i,j,k} c_ij · x_ijk
subject to Σ_{j,k} x_ijk = 1                 for every customer i
           Σ_i d_i · x_ijk ≤ Q              for every vehicle k
           flow conservation, subtour elimination
           x_ijk ∈ {0,1}
```

Add time windows and it becomes VRPTW: service at customer i must start within [e_i, l_i].

| Variant | Additional constraint | Typical use |
|---|---|---|
| CVRP | Capacity only | Store distribution |
| VRPTW | Time windows | Appointments, business deliveries |
| VRPPD | Pickup before delivery, same vehicle | Returns, backhauls, dial-a-ride |
| MDVRP | Multiple depots | Regional fleets |
| HFVRP | Vehicles differ in capacity and cost | Mixed van/truck fleet |
| DVRP | Orders arrive during execution | Same-day delivery |
| 2E-VRP | Two echelons via a satellite | Urban micro-hub delivery |

CVRP is NP-hard; exact solutions are practical only for small instances (roughly under 100 customers without decomposition). Production uses heuristics and metaheuristics — savings, insertion, local search, large neighbourhood search — and accepts a gap to optimal in exchange for runtime.

### 4.7 OR-Tools: a runnable CVRPTW

The model below uses Manhattan distances on a toy 7-customer instance, two vehicles of capacity 100, service times, and time windows. It is deliberately small so the structure is visible and the result can be verified.

```python
"""CVRPTW with OR-Tools. Install: pip install ortools==9.11.*"""
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

SPEED_KMH, SERVICE_MIN, HORIZON, DEPOT = 30.0, 8, 480, 0
CAPACITY = [100, 100]
COORDS = [(0, 0), (2, 3), (5, 1), (6, 6), (3, 8), (8, 4), (1, 7), (4, 2)]
DEMAND = [0, 20, 15, 25, 18, 12, 10, 10]
WINDOWS = [(0, HORIZON), (60, 300), (90, 330), (120, 360),
           (150, 390), (180, 420), (210, 450), (240, 480)]

N = len(COORDS)
DIST = [[abs(COORDS[i][0] - COORDS[j][0]) + abs(COORDS[i][1] - COORDS[j][1])
         for j in range(N)] for i in range(N)]
TIME = [[int(round(DIST[i][j] / SPEED_KMH * 60)) for j in range(N)] for i in range(N)]

def solve():
    manager = pywrapcp.RoutingIndexManager(N, len(CAPACITY), DEPOT)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(fi, ti):
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return TIME[f][t] + (SERVICE_MIN if f != DEPOT else 0)

    transit = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(
            lambda i: DEMAND[manager.IndexToNode(i)]),
        0, CAPACITY, True, "Capacity")

    routing.AddDimension(transit, 30, HORIZON, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, N):
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(*WINDOWS[node])
    for v in range(len(CAPACITY)):
        time_dim.CumulVar(routing.Start(v)).SetRange(*WINDOWS[DEPOT])
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(5)
    return manager, routing, routing.SolveWithParameters(params), time_dim

if __name__ == "__main__":
    manager, routing, solution, time_dim = solve()
    if solution is None:
        raise SystemExit("No solution: check capacity, windows, and horizon.")
    total = 0
    for v in range(len(CAPACITY)):
        idx, load, leg = routing.Start(v), 0, 0
        print(f"Vehicle {v}")
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            load += DEMAND[node]
            print(f"  stop {node}  arrival {solution.Value(time_dim.CumulVar(idx))} min")
            nxt = solution.Value(routing.NextVar(idx))
            leg += DIST[node][manager.IndexToNode(nxt)]
            idx = nxt
        total += leg
        print(f"  load {load}, distance {leg} km")
    print(f"Total {total} km")
```

On this instance the solver returns two routes with loads 20 and 90 and a total of 44 km; an exhaustive search over all 7! orderings confirms 44 km is optimal, which is the point of a toy instance — you can verify the solver rather than trust it. In production, replace the Manhattan matrix with a road-network matrix and never trust great-circle distance: it underestimates urban distance by 25–40%. Set the time limit to 5–30 seconds for dispatch and 5–30 minutes for planning, and treat every dropped stop under a disjunction penalty as a visible exception, never as a silent omission.

### 4.8 Route optimisation versus dispatch

| Question | Route optimisation | Dispatch |
|---|---|---|
| When | Day before, fixed orders | During the day, changing orders |
| Objective | Minimum cost for known demand | Feasible, stable, service-preserving |
| Latency budget | Minutes to hours | Seconds |
| Stability | Less important | Critical — a route that changes twice is executed badly |
| Method | Offline metaheuristic | Insertion heuristic with re-optimization triggers |

A plan that is 3% cheaper but re-sequences the driver at 09:00 will be executed worse than one that is 1% worse and stable. Measure **plan adherence** and **churn** (share of stops whose sequence changed after dispatch), not only distance.

### 4.9 Last mile: density, windows, and failure

```
cost per stop ≈ (driver cost per hour × stop time) + (vehicle cost per km ÷ stops per km)
```

Urban van route, driver $28/hour fully loaded, vehicle $0.72/km:

| Stops | Distance | Paid hours | Labour | Vehicle | Total | Cost/stop | Cost/km |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 96 km | 8.0 | $224.00 | $69.12 | $293.12 | $9.77 | $3.05 |
| 45 | 120 km | 9.0 | $252.00 | $86.40 | $338.40 | $7.52 | $2.82 |
| 60 | 132 km | 9.5 | $266.00 | $95.04 | $361.04 | $6.02 | $2.74 |

Arithmetic check: $338.40 ÷ 45 = $7.52; $338.40 ÷ 120 = $2.82. A 33% increase in stops on the same route cuts cost per stop by 20% and cost per km by only 3%. Density is the lever; distance is not.

| Event | Direct cost | Indirect cost |
|---|---|---|
| Failed first attempt | Re-delivery stop, roughly $12–20 | Customer contact, churn risk |
| Narrow window (2 h) | 15–30% more route time | Lower stops per hour |
| Return pickup | Extra stop, handling | Refund and resale loss |
| Address correction | Parcel surcharge | Delay |

A useful rule: **every failed delivery consumes the margin of roughly two successful ones** at typical e-commerce contribution margins. Model the failure rate explicitly; do not bury it in cost per stop.

### 4.10 ETA prediction

ETA is a regression problem with an asymmetric loss. Being 20 minutes early is cheap; being 60 minutes late is expensive.

1. **Predict a distribution, not a point.** Use quantile regression (for example LightGBM with `objective="quantile"`, alpha 0.5 and 0.9) and quote the 90th percentile as the promise.
2. **Separate pre-assignment from post-assignment.** Before a driver accepts you have zone and order features; after assignment you have driver history, live position, and route sequence. The second model is materially better and should not be blended with the first.
3. **Validate at the promise.** Report promise-keeping rate, p90 lateness, and the cost of the buffer, not MAE alone.

Feature families are route (stops remaining, sequence position), traffic (time-of-day profiles, incident feed), driver history, order attributes (weight, cube, floor, access type), and weather. Each carries a leakage risk: the realised route, post-hoc congestion, or a weather observation that was not available when the promise was made.

Censoring trap: delivery time is observed only for orders that were delivered. Cancelled and returned orders are missing not at random, and excluding them biases ETA optimistic.

### 4.11 Dock scheduling and yard

A dock door is a server; a truck is a customer. Appointment systems fail when they book to average service time and ignore variance.

| Metric | Definition | Target |
|---|---|---|
| Dock utilisation | Occupied door-hours ÷ available door-hours | 65–80% |
| Truck turnaround | Arrival to departure | Lane-dependent; measure p90, not mean |
| Appointment adherence | Trucks arriving within slot ÷ booked | > 85% |
| Detention incidence | Loads incurring detention ÷ loads | < 3% |
| Yard dwell | Gate-in to dock-in | < 30 min |

Over-booking a dock above roughly 85% utilisation produces exponential waiting, the same mathematics as any queue (Module 13). Detention charges are then a symptom of the schedule, not of the carrier.

### 4.12 Incoterms 2020

Incoterms define who arranges and pays for carriage, insurance, and customs, and where risk transfers. They do not define title, payment, or governing law.

| Term | Mode | Risk transfers | Freight paid by | Insurance |
|---|---|---|---|---|
| EXW | Any | At seller's premises | Buyer | Buyer's choice |
| FCA | Any | On delivery to carrier | Buyer | Buyer's choice |
| CPT | Any | On handover to first carrier | Seller | Buyer's choice |
| CIP | Any | On handover to first carrier | Seller | Seller, Institute Cargo Clauses (A) |
| DAP | Any | At named destination, not unloaded | Seller | Seller's choice |
| DPU | Any | At destination, unloaded | Seller | Seller's choice |
| DDP | Any | At destination, import cleared | Seller | Seller's choice |
| FAS / FOB | Sea | Alongside ship / on board | Buyer | Buyer's choice |
| CFR / CIF | Sea | On board vessel | Seller | Buyer / Seller, Clause (C) |

The 2020 revision renamed DAT to DPU, raised the CIP insurance requirement to Clause (A), and added an optional on-board bill of lading mechanism for FCA. For analytics, the Incoterm decides whose freight cost appears in whose P&L; a landed-cost model that ignores it double-counts or omits freight.

### 4.13 Freight audit and payment

Freight audit compares the invoice to the contract rate and the shipment record. Recoverable leakage is usually 2–6% of freight spend at a mid-size shipper.

| Check | Rule | Typical finding |
|---|---|---|
| Rate match | Invoice linehaul = contracted rate × weight/distance | 1–3% overbilling |
| Accessorial validity | Charge has a matching event (detention clock, liftgate request) | 20–40% of accessorials unsupported |
| Duplicate invoice | Same carrier, PRO number, amount, within 30 days | 0.5–2% duplicates |
| Weight or reclass | Billed weight vs observed or certified | LTL reweigh disputes |
| Fuel surcharge | Applied peg and index date | Stale or wrong index |
| Contract expiry | Rate effective date within contract term | Expired rates still billed |

Build the audit as a rate engine, not a spreadsheet: contract terms in a table, shipment facts in a table, a deterministic comparison, and an exception queue. LLM extraction is useful for unstructured accessorial descriptions, but the arithmetic must stay deterministic.

### 4.14 Carbon per tonne-km

ISO 14083:2023 defines the quantification and reporting of GHG emissions from transport chain operations; the GLEC Framework supplies the calculation structure and default factors.

```
emissions (kg CO2e) = activity (tonne-km) × emission factor (kg CO2e/tonne-km)
```

Four adjustments change the answer materially. **Load factor:** a half-empty truck has roughly double the emissions per tonne-km; report the vehicle-km factor and the loaded tonne-km factor. **Empty running:** positioning and backhaul legs belong in the boundary under ISO 14083; excluding them understates road emissions by 15–30%. **Allocation:** mass allocation penalises dense goods, volume allocation penalises light bulky goods — state the choice. **Well-to-wheel vs tank-to-wheel:** biofuels and electricity look very different under the two boundaries.

| Mode shift | Cost change | Carbon change | When it works |
|---|---|---|---|
| Air → ocean | −80 to −90% | −95% | Low value density, lead-time tolerant |
| Road TL → intermodal | −10 to −25% | −50 to −70% | > 500 km, dense lane, ramp access |
| Diesel → HVO or biofuel | +5 to +20% | −70 to −90% well-to-wheel, feedstock-dependent | Carrier offers mass-balance certification |
| Van → cargo bike | −20 to −40% in dense urban | −90% | < 5 km legs, low weight, micro-hub |

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Shipment | Shipment ID | TMS / OMS | Origin, destination, weight, cube, class, mode, service |
| Tender | Load × carrier × attempt | TMS | Timestamp, response, rejection reason |
| Route and stop | Stop ID | Routing engine, telematics | Planned vs actual sequence and time |
| Invoice | Invoice line | Carrier EDI 210 or PDF | Charge code, amount, PRO number |
| Rate contract | Lane × effective date | Procurement / TMS | Rate basis, minimums, accessorials, fuel peg |
| Delivery event | Order × event | Carrier API, EPCIS, POD | Timestamp, geo, exception code |
| Master data | Location, item | ERP / PIM | Dock constraints, appointment rules, cube |

**Standards.** ANSI X12 204 (load tender), 990 (response), 214 (status), 210 (invoice), 997 (acknowledgement); UN/EDIFACT IFTMIN and IFTSTA; GS1 SSCC for pallet and parcel identity; EPCIS 2.0 for chain of custody; NMFC for LTL class; Incoterms 2020; ISO 14083 for emissions.

**Data-quality checks before any model:** every shipment has an explicit mode and service level; distances are road-network with a stated provider and version; weight and cube units are explicit and converted once at ingestion; timestamps carry a time zone; cancelled shipments are excluded from cost per stop but included in capacity demand; accessorial codes map to a controlled list.

**Workflow.** Build the lane baseline (contract vs spot, cost per shipment, service), then diagnose (leakage, density, tender acceptance), then optimise (route, mode, consolidation), then monitor (cost per stop, OTIF, emissions). Optimising before diagnosing produces beautiful routes on a broken rate base.

## 6. Industry case studies

### 6.1 UPS ORION (2011–2016)

**Context.** UPS ran roughly 55,000 US package cars with drivers relying on route knowledge built over years. Fuel and labour dominate parcel cost.

**Intervention.** ORION (On-Road Integrated Optimization and Navigation) combined telematics, digital maps, and a heuristic optimiser to sequence stops and reduce left turns, cutting idling and crossing traffic. Rollout was phased because dispatchers and drivers had to accept it.

**Evidence.** UPS reported targets of about 100 million miles and 10 million gallons of fuel saved per year, with claimed annual savings in the $300–400 million range. These are company figures, not an independent audit.

**Outcome.** ORION became the reference case for route optimisation at scale, and the left-turn policy a widely copied heuristic.

**What transferred.** The integration model — optimisation embedded in the dispatcher workflow, with driver acceptance treated as a first-class constraint — and the insight that a simple explainable rule can capture much of a full optimiser's value.

**What did not.** The savings figures, which depend on UPS density, fuel price, and labour contract, and the timeline: five years, not a six-week pilot.

### 6.2 Uber DeepETA (2022)

**Context.** Uber needed arrival predictions both pre-assignment (before a driver accepts) and post-assignment (after the route is known). Errors compound into bad promises and driver incentives.

**Intervention.** Uber published an engineering account of DeepETA, a transformer-based architecture replacing gradient-boosted trees, with separate treatment of pre- and post-assignment prediction and explicit attention to tail latency.

**Evidence.** Uber reported reduced mean absolute error on both tasks relative to its prior production model. This is a company technical report, not a peer-reviewed benchmark.

**Outcome.** The system entered production, where ETA feeds pricing, matching, and the customer promise.

**What transferred.** Splitting one "ETA model" into two decision contexts, and optimising tail rather than mean error. Both apply directly to retail delivery promises.

**What did not.** The architecture. A transformer is not required for a fleet of 200 vans; a gradient-boosted model with good features usually wins at that scale and costs far less to operate.

### 6.3 Amazon's US fulfilment network regionalisation (2023–2024)

**Context.** Amazon's US network had grown into a national hub-and-spoke system in which packages often crossed the country. After pandemic-era expansion, cost to serve rose.

**Intervention.** Amazon reorganised fulfilment into regional zones so most orders are served from inventory closer to the customer, cutting long-haul moves and enabling faster delivery.

**Evidence.** Amazon described the redesign in earnings commentary and shareholder communications, reporting shorter distances per package and improved cost to serve. Specific percentages are company claims. [VERIFY: exact distance and cost figures against Amazon filings before reuse.]

**Outcome.** Faster delivery on a large share of Prime orders at lower transport cost per unit, paid for with more inventory in more places — an inventory-for-transport trade.

**What transferred.** The recognition that last-mile cost is set upstream by inventory placement. Routing analytics cannot compensate for inventory sitting in the wrong region.

**What did not.** The scale. Regionalisation needs demand density most retailers lack; a mid-size retailer replicating it adds inventory without enough transport saving.

### 6.4 Convoy (2015–2023)

**Context.** Convoy built a digital freight brokerage matching shippers to carriers with algorithms and automated pricing, positioned as a technology-first replacement for the traditional broker.

**Intervention.** Automated load matching, dynamic pricing, and a carrier app, on the claim that software would remove brokerage inefficiency.

**Evidence.** Convoy raised several hundred million dollars and reached a multi-billion-dollar valuation, then ceased operations in October 2023; its technology assets were acquired by Flexport. Freight market conditions, negative unit economics, and the difficulty of automating relationship-heavy capacity sourcing are the commonly cited causes. [VERIFY: total capital raised and peak valuation.]

**Outcome.** Shutdown. The 2023–2024 freight recession exposed brokerages with thin margins and high fixed technology costs.

**What transferred.** Digital matching works and is now standard, but it is a feature, not a business model. Pricing and matching analytics must be paired with capacity relationships and a cost structure that survives a downcycle.

**What did not.** The thesis that algorithmic matching alone creates durable margin in a commodity market with low barriers to entry.

### 6.5 Maersk ECO Delivery Ocean (2019–2026)

**Context.** Shippers facing Scope 3 reporting and customer pressure wanted a lower-carbon ocean option without changing lead time.

**Intervention.** Maersk offers ECO Delivery Ocean, where an energy surcharge funds green fuels (initially biodiesel, later methanol where available) on a mass-balance basis, with a certificate documenting avoided emissions.

**Evidence.** Maersk claims an 80%+ GHG reduction versus fossil fuel for the product. The claim is company-reported and depends on fuel and boundary (well-to-wake); independent assurance varies by product. [VERIFY: boundary and assurance for the specific product year.]

**Outcome.** A commercial product that lets a shipper buy carbon reduction as a freight attribute, priced as a surcharge per container.

**What transferred.** The product design — carbon as a line item with a certificate — is a workable model for auditable emissions data, and it shows why "carbon-neutral shipping" needs a stated methodology.

**What did not.** The headline percentage is not usable in the shipper's own inventory without confirming boundary, allocation, and assurance under ISO 14083.

## 7. Labs

### Lab 12.1 — CVRPTW on a real road network

**Objective.** Build, solve, and diagnose a capacitated routing model with time windows, and quantify the gap between a naive sequence and an optimised one.

**Data.** 150–400 delivery stops from a public or course-provided order file with coordinates, demand, and windows; road-network distance and duration matrices from OSRM or a provided matrix.

**Steps.** (1) Build the matrices and compare with great-circle distance. (2) Solve with OR-Tools using `PATH_CHEAPEST_ARC` + `GUIDED_LOCAL_SEARCH`, with and without time windows. (3) Vary the time limit (1 s, 10 s, 60 s) and plot quality against runtime. (4) Compare with nearest-neighbour. (5) Report cost per stop, distance, vehicles used, and dropped stops under each disjunction penalty. (6) Increase demand 10% and identify the binding constraint.

**Deliverable.** Notebook plus a two-page route analysis with the runtime-quality curve and constraint diagnosis.

**Acceptance criteria.** Road-network matrices documented; baseline and optimiser on identical data; vehicle count and dropped stops reported; binding constraint identified with evidence.

**Stretch.** Add a heterogeneous fleet and show whether the optimiser prefers more vans or fewer trucks at equal total capacity.

### Lab 12.2 — Mode selection and landed cost

**Objective.** Build a mode-selection model trading freight cost against transit time, inventory carrying cost, and emissions.

**Data.** Course rate card across five modes and three distance bands; product file with value, weight, cube, and shelf life.

**Steps.** (1) Compute cost per shipment and per tonne-km per mode. (2) Add carrying cost during transit at 22% per year. (3) Add safety stock implied by transit variability, using the standard deviation of transit days. (4) Compute emissions with ISO 14083 default factors. (5) Build a decision table by value density and required lead time. (6) Find the volume at which LTL becomes TL on a representative lane.

**Deliverable.** Decision table and break-even chart with arithmetic shown.

**Acceptance criteria.** Units explicit; carrying cost included; variability drives safety stock; emissions boundary stated; break-even computed, not asserted.

**Stretch.** Add a carbon price of $50 and $150 per tonne CO2e and show how the mode boundary moves.

### Lab 12.3 — Carrier scorecard and tender analysis

**Objective.** Separate carrier performance from lane structure and locate where a rate or service problem sits.

**Data.** 12 months of tender, shipment, and invoice records for 8 carriers and roughly 2,000 lanes.

**Steps.** (1) Compute tender acceptance, on-time pickup, on-time delivery, damage rate, and invoice accuracy per carrier. (2) Segment lanes by distance, weight, and origin market and compute the carrier effect within segment. (3) Compute spot versus contract share and cost per mile versus benchmark. (4) Quantify accessorial leakage by type. (5) Build a weighted scorecard and rank carriers. (6) Test rank stability when weights change.

**Deliverable.** Scorecard workbook plus a one-page recommendation on which lanes to re-tender.

**Acceptance criteria.** Within-segment comparison, not raw averages; six metrics with definitions; weight sensitivity shown; re-tender list justified by lane economics.

**Stretch.** Estimate the value of a 5-point tender-acceptance improvement on the top 20 lanes.

### Lab 12.4 — ETA model and promise evaluation

**Objective.** Build and evaluate a delivery ETA model at the promise, not the mean.

**Data.** 6–12 months of delivered orders with promise time, actual time, driver, route, weather, and address features; at least 50,000 orders.

**Steps.** (1) Define the prediction point and remove post-delivery features. (2) Fit median and 0.9-quantile models. (3) Evaluate promise-keeping rate, p90 lateness, and MAE by distance band. (4) Benchmark against the carrier's quoted ETA. (5) Compute the cost of buffer: how many minutes of padding buy how many points of promise accuracy. (6) Audit the effect of excluding cancelled orders.

**Deliverable.** ETA evaluation report with calibration plot and a promise-policy recommendation.

**Acceptance criteria.** No leakage; quantile calibration shown; benchmark included; buffer cost quantified; censoring addressed.

**Stretch.** Build a segment-level conformal interval and compare coverage with the quantile model.

### Lab 12.5 — Emissions and mode shift

**Objective.** Calculate shipment-level emissions and evaluate a mode-shift business case.

**Data.** Shipment records with mode, weight, distance, and load factor; GLEC default factors or a carrier factor file.

**Steps.** (1) Compute emissions per shipment with mass allocation. (2) Repeat with volume allocation and quantify the difference. (3) Include empty running and show the change. (4) Model a 20% shift from air to ocean and road to intermodal. (5) Compute cost change, carbon change, and service consequence. (6) State what must be true for the customer to accept the shift.

**Deliverable.** Emissions baseline and mode-shift case with a boundary statement.

**Acceptance criteria.** Activity data and factors sourced; allocation stated; empty running included or explicitly excluded with justification; cost and service consequences quantified.

**Stretch.** Add a carrier primary-factor scenario and compare with default-factor results.

## 8. Assignment — Routing and tender analysis

Produce a **transport decision pack** for one real or realistic lane portfolio: which mode and route structure should serve the demand, which carriers keep the business, and what the service and carbon consequences are.

| Section | Content |
|---|---|
| 1. Scope | Lanes, volume, service levels, contract dates, decision owner |
| 2. Baseline | Cost per shipment, per stop, per tonne-km; OTIF; accessorial share |
| 3. Mode analysis | Break-even volumes, transit-time cost, emissions comparison |
| 4. Route model | CVRPTW formulation, solver settings, runtime-quality curve, baseline |
| 5. Last mile | Density analysis, window trade-off, failure cost |
| 6. Carrier analysis | Scorecard, tender acceptance, concentration, re-tender list |
| 7. Leakage | Audit findings with recoverable dollars |
| 8. Carbon | ISO 14083 calculation and mode-shift option |
| 9. Recommendation | Mode, route policy, carrier award, service concession required |
| 10. Risks | What breaks it: rate moves, capacity, regulation |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing and baseline | 15% | Optimises without a baseline | Cost per stop and OTIF baselined | Baseline segmented by lane and mode with data-quality caveats |
| Mode and cost analysis | 20% | Compares sticker rates | Includes fuel and accessorials | Adds transit carrying cost, variability, break-even volumes |
| Routing model | 25% | Uses a library example unchanged | Correct CVRPTW with real matrix | Tuned, validated against baseline, runtime-quality trade-off, binding constraints diagnosed |
| Last mile and ETA | 15% | Averages only | Cost per stop and failure rate | Density elasticity, quantile ETA, promise policy with buffer cost |
| Carrier and tender analytics | 15% | Raw ranking | Six metrics, within-segment | Separates carrier effect from lane structure; award sensitivity |
| Carbon and communication | 10% | No emissions | ISO 14083 calculation | Boundary, allocation, empty running handled; decision-ready memo |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Driver hours | Hours-of-service rules (US FMCSA; EU driving-time rules) constrain feasibility | Encode legal driving time as a hard constraint |
| Customs and Incoterms | Wrong Incoterm shifts cost and risk and can invalidate insurance | Validate Incoterm against movement and customs declaration |
| Emissions claims | Unsupported "carbon neutral" claims attract advertising regulators | ISO 14083 boundary; state the methodology |
| Sanctions and denied parties | Carriers and routes may touch restricted parties or regions | Screen counterparties; retain evidence |
| Delivery equity | Dense areas get cheap routes; rural and low-income areas may be underserved or surcharged | Report service and cost by area type; set a service floor |
| Driver surveillance | Telematics and cameras raise privacy and works-council issues | Data minimisation, retention limits, worker consultation |
| Algorithmic dispatch | Automated assignment can disadvantage drivers or gig workers | Publish rules; allow appeal; audit allocation |
| Antitrust | Sharing lane rates or capacity with competitors can facilitate coordination | Benchmark through an intermediary; never exchange forward-looking rates |

Two professional obligations: **never present a modelled rate as a contracted rate**, and **never report a carbon saving without its boundary**. Both errors survive into board decks, and both are avoidable with one line of labelling.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Great-circle distances | Route looks 30% shorter than driver experience | Geometry instead of road network | Understated cost and time | Road-network matrix, versioned |
| Optimising before auditing | Beautiful routes, unchanged invoice | Rate leakage unaddressed | Savings never reach P&L | Audit first |
| Capacity modelled, not real | Solver fills 100% of the trailer | No cube or pallet constraint | Loads rejected at the dock | Model weight and cube; practical fill ceiling |
| Soft time windows | Plan feasible, execution late | Dispatcher overrides | Service failure | Hard windows for committed appointments |
| Unstable re-planning | Drivers get three route changes | Re-optimisation without churn cost | Adherence collapses | Freeze windows; penalise sequence change |
| Average ETA reported | MAE improves, complaints rise | Tail error ignored | Promise failures | Quantile evaluation, p90, promise rate |
| Empty running excluded | Carbon per delivered tonne looks excellent | Boundary error | Understated footprint | Include repositioning and backhaul |
| Accessorials unmodelled | Cost per stop drifts upward | Detention, liftgate, reclass | Budget miss | Charge-code taxonomy; event validation |
| Tender acceptance assumed | Plan assumes contracted capacity | Carrier rejects at peak | Spot cost spike | Model acceptance by lane and season |
| Lane overfitting | A 12-shipment lane gets its own rate curve | Too little data | Noise presented as insight | Cluster lanes; external benchmark |
| Single-carrier concentration | 70% of volume with one carrier | Price and capacity risk | Service collapse if it fails | Concentration metric; tested backup |
| Carbon double counting | Scope 1 and 3 include the same fuel | Boundary error | Invalid disclosure | One boundary document; reconciliation |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | When does LTL beat TL? | Below roughly half a trailer and when the lane-day cannot be consolidated; compare landed cost including accessorials, not rate per cwt. |
| 2 | Why is cost per tonne-km wrong for last mile? | It ignores stop density and service; last mile is a stop-cost business. |
| 3 | What makes VRP hard in practice? | Time windows, driver hours, access, dynamic orders, latency; the combinatorics is the easy part. |
| 4 | Your optimiser saves 8% distance but service drops. Why? | Plan adherence, tighter windows, or churn; distance is not the objective. |
| 5 | How do you evaluate an ETA model? | Promise-keeping rate, p90 lateness, quantile calibration, buffer cost; MAE hides the tail. |
| 6 | Demurrage versus detention? | Inside the terminal versus outside; different clocks and owners. |
| 7 | Why does tender acceptance matter more than rate? | Rejected tenders move to spot at higher cost; effective rate is contract rate weighted by acceptance. |
| 8 | Cut freight spend 10% in 90 days — how? | Audit leakage, consolidate LTL, improve tender compliance, re-bid tail lanes; routing last. |
| 9 | Carbon effect of a half-empty truck? | Roughly double emissions per tonne-km; load factor is the dominant lever. |
| 10 | When should you not re-optimise? | When churn, driver trust, or appointment stability outweigh the distance saving. |
| 11 | How do Incoterms affect landed cost? | They set who pays freight and insurance and where risk transfers; ignoring them double-counts. |
| 12 | What would you demand before trusting a carrier OTIF? | Definition of on-time, denominator, appointment vs arrival, exclusions, audit trail. |

## 13. Key terms

| Term | Definition |
|---|---|
| TL / FTL | Full truckload; one shipper uses the whole trailer |
| LTL | Less-than-truckload; multiple shippers share a trailer through terminals |
| Intermodal | Movement using more than one mode, typically rail plus drayage |
| Drayage | Short-haul truck move between a port or ramp and a facility |
| TEU / FEU | Twenty-foot / forty-foot equivalent unit, the container capacity measure |
| Accessorial | Charge beyond linehaul, such as detention, liftgate, or appointment |
| Fuel surcharge | Per-mile or percentage charge tied to a published fuel index |
| Detention | Charge for holding a carrier's equipment beyond free time |
| Demurrage | Charge for a container held inside a terminal beyond free time |
| NMFC | National Motor Freight Classification, the LTL class system |
| Chargeable weight | Greater of actual and volumetric weight |
| DIM weight | Dimensional weight from cube and a divisor |
| Lane | Origin–destination–equipment combination with volume and service |
| Tender acceptance rate | Accepted tenders divided by tendered loads |
| Spot rate | One-off market rate, not under contract |
| CVRP | Capacitated vehicle routing problem |
| VRPTW | VRP with time windows |
| VRPPD | VRP with pickup and delivery |
| Metaheuristic | General-purpose search method for hard combinatorial problems |
| Guided local search | Metaheuristic penalising frequently used solution features |
| Disjunction | OR-Tools mechanism allowing a stop to be dropped at a penalty |
| Plan adherence | Share of executed stops matching the planned sequence or time |
| Route density | Stops per kilometre or per hour on a route |
| First-attempt delivery | Successful delivery on the first visit |
| ETA | Estimated time of arrival |
| Quantile regression | Model predicting a conditional quantile, not the mean |
| Incoterms 2020 | ICC delivery terms defining cost and risk transfer |
| ISO 14083 | Standard for quantifying transport chain GHG emissions |
| GLEC Framework | Logistics emissions accounting framework and default factors |
| Tonne-km | One tonne moved one kilometre; the standard transport activity unit |

## 14. Sources and further reading

**Standards and regulation**

- International Chamber of Commerce, *Incoterms 2020*. ICC, 2019. https://iccwbo.org — accessed 27 October 2026.
- ISO 14083:2023, *Greenhouse gases — Quantification and reporting of greenhouse gas emissions arising from transport chain operations*. ISO, 2023.
- Smart Freight Centre, *GLEC Framework for Logistics Emissions Accounting and Reporting*, version 3.0. https://www.smartfreightcentre.org — accessed 27 October 2026.
- U.S. Federal Motor Carrier Safety Administration, Hours of Service, 49 CFR Part 395. https://www.fmcsa.dot.gov — accessed 27 October 2026.
- National Motor Freight Traffic Association, *National Motor Freight Classification*. https://www.nmfta.org — accessed 27 October 2026.
- ANSI ASC X12 transaction sets 204, 210, 214, 990, 997. https://x12.org — accessed 27 October 2026.
- GS1, *EPCIS 2.0* and SSCC standards. https://www.gs1.org/standards/epcis — accessed 27 October 2026.

**Peer-reviewed and methodological**

- Dantzig, G. B., and Ramser, J. H. "The Truck Dispatching Problem." *Management Science* 6(1), 1959.
- Clarke, G., and Wright, J. W. "Scheduling of Vehicles from a Central Depot to a Number of Delivery Points." *Operations Research* 12(4), 1964.
- Solomon, M. M. "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints." *Operations Research* 35(2), 1987.
- Toth, P., and Vigo, D., eds. *Vehicle Routing: Problems, Methods, and Applications*, 2nd ed. SIAM, 2014.
- Ropke, S., and Pisinger, D. "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows." *Transportation Science* 40(4), 2006.
- Boysen, N., de Koster, R., and Weidinger, F. "Warehousing in the E-Commerce Era: A Survey." *European Journal of Operational Research* 277(2), 2019.
- Demir, E., Bektaş, T., and Laporte, G. "A Review of Recent Research on Green Road Freight Transportation." *European Journal of Operational Research* 237(3), 2014.

**Industry and company evidence**

- Freightos, *Freightos Global Outlook*, October 2026. https://www.freightos.com/freight-resources/freightos-global-outlook-october-2026/ — accessed 27 October 2026. Dated rate snapshot.
- FreightWaves, "How New Capacity Is Raising Rates in a Chaotic Ocean Market," 20 October 2026. https://www.freightwaves.com/news/how-new-capacity-is-raising-rates-in-chaotic-ocean-market — accessed 27 October 2026.
- Uber Engineering, "DeepETA: How Uber Predicts Arrival Times Using Deep Learning," 2022. Company technical report; verify the current URL before citing.
- UPS, ORION programme materials and sustainability reports. Company claims; figures not independently audited. [VERIFY: exact mileage and savings against the latest UPS sustainability report.]
- Amazon, shareholder letters and earnings commentary on US fulfilment regionalisation, 2023–2024. Company claims.
- Convoy shutdown coverage, October 2023 (FreightWaves, *Wall Street Journal*, GeekWire). [VERIFY: capital raised and peak valuation.]
- Maersk, ECO Delivery Ocean product documentation. Company claim of 80%+ GHG reduction; boundary and assurance [VERIFY] per product year.

**Tools and datasets**

- Google OR-Tools routing library documentation and examples. https://developers.google.com/optimization/routing — accessed 27 October 2026.
- OSRM (Open Source Routing Machine). https://project-osrm.org — accessed 27 October 2026.
- Valhalla routing engine. https://valhalla.github.io/valhalla/ — accessed 27 October 2026.
- Solomon VRPTW benchmark instances. https://www.sintef.no/projectweb/top/vrptw/ — accessed 27 October 2026.
- DAT Freight & Analytics and Cass Transportation Index for lane benchmarking. Commercial data; licensing required.

> **Note:** Freight rates in this module are dated October 2026 snapshots. Re-baseline before any commercial decision; a rate is a market observation, not a constant.

---

> **Next:** [Module 13](13_warehousing_fulfillment_labor.md) — the truck arrives, the clock starts, and the cost moves from the road to the dock, the pick path, and the pack bench.
