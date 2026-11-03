# The Network as a Graph and Cost-to-Serve

> Module 11 — Network Design and Optimization
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11, topic 01
> Version 2026.5

*Nodes, arcs, flows, and the decomposition of delivered cost.*

---

## 1. Why this topic exists

Almost no network study fails because the solver was wrong. They fail earlier, in two places nobody
puts on a slide: the representation and the measurement. The representation is the graph — which
objects exist, what attributes they carry, and which of them the model may choose. The measurement
is cost-to-serve (C2S) — what it actually costs to put a unit in front of a specific customer,
through a specific lane, in a specific period.

Both fail quietly. A wrong graph produces a feasible solution to a problem nobody has. A wrong cost
produces a defensible-looking recommendation built on an average that no customer ever experienced.
The optimiser will not tell you either has happened. It reports an objective value and a gap, and
both are correct for the model it was given.

Three errors recur, and each has a price attached.

1. **The arc that carries two costs.** A distribution centre (DC) is charged a handling rate built
   from "landed cost per pallet in", which already contains the inbound freight. The inbound freight
   is also modelled on the plant-to-DC arc. Inbound freight is counted twice, every DC looks more
   expensive than it is, and the bias favours fewer and larger sites — systematically, not randomly.
2. **The centroid that eats the last fifty kilometres.** Demand for a metropolitan area is
   aggregated to one point. Every customer inside the area was served by a local delivery run,
   averaging perhaps eighteen kilometres. Those kilometres are deleted. On a 400 km lane the
   deletion is four percent; on a 40 km metro lane it is most of the cost, and the model then
   concludes that the near market is cheap and the far market expensive when part of the difference
   was manufactured by the aggregation rule.
3. **The average rate applied to a marginal decision.** Freight is reported at €0.084 per
   tonne-kilometre across the network. The decision on the table is one more weekly delivery to a
   customer at the edge of the territory, priced at the marginal rate on that lane — perhaps €0.31,
   because the extra drop is a part load with a minimum charge attached. The average is right for the
   annual report and wrong for the decision, and the gap between them is where most of the money is.

**Why this matters to a data scientist specifically.** The graph and the cost model arrive as
inherited artefacts. Somebody hands you `lanes.csv` with columns `from`, `to`, `km`,
`cost_per_unit`, and it looks like a graph. It is not. It is a table that has lost what made it a
graph: which node types may transship, which arcs are directional, whether capacity sits on the arc
or the node, and whether `cost_per_unit` is a rate, a rate at a volume band, or the residue of last
year's allocation. Reconstructing those decisions explicitly is the first professional act on any
network study.

This topic is second in the module because every later topic is a construction on this
representation. Topic 03 turns it into a mixed-integer linear program (MILP). Topic 04 adds the
binary opening decision. Topic 05 adds commodity and period indices. Topic 09 adds an emissions term
with the same shape as a cost term. If the representation is wrong, all of them are wrong in the same
way.

**What this topic does not cover.** Module 12 owns transport operations and routing: load building,
vehicle routing, appointment scheduling, last-mile economics. Here you get only the cost structures
a network model needs to see. Module 13 owns warehouse cost mechanics: labour standards, slotting,
pick-path productivity. Here the warehouse appears only as a cost pool with drivers.

> **Note:** A network model is an argument about the future built from data about the past. The graph
> is its vocabulary and the cost model is its evidence. Neither is neutral. Both encode choices made
> by whoever built them, usually years ago, usually in a spreadsheet, usually undocumented.

## 2. Learning outcomes

By the end of this topic you can:

1. **Represent** a supply chain as three relations — nodes, arcs, flows — and **state** for each
   attribute whether it belongs on the node, the arc, or the flow, with the double-counting
   consequence of getting it wrong.
2. **Classify** a problem as transportation, transshipment, or facility location, and **explain** what
   each classification permits and forbids.
3. **Choose** between a directed and undirected representation for a named lane and **justify** it
   from the rate structure, the capacity, or the physical asymmetry.
4. **Quantify** the cost error from aggregating demand to zones, including the within-zone distance
   aggregation deletes, and **set** an aggregation level from a stated error budget.
5. **Select** a distance metric and **estimate** the gap between great-circle, road network, and
   routed distances for a stated geography.
6. **Convert** between eaches, cases, pallets, kilograms, cubic metres, and load equivalents at a
   stated load factor.
7. **Decompose** delivered cost into eight components, **separate** traced from allocated cost, and
   **report** a C2S figure with its boundary, drivers, period, and traced share.
8. **Compute** a customer-decile C2S matrix by hand and **identify** the order-value threshold below
   which a customer becomes unprofitable at a stated gross margin.

Each outcome maps to an exercise in section 12 and an answer in section 13.

## 3. The core mental model

Hold two objects at once. The first is a **graph**: nodes with types, arcs with attributes, flows
with units. The second is a **cost surface** on the same graph, where each arc and node contributes
a pool with a driver attached. The graph says what is possible; the cost surface says what it costs.
A network model searches the first, scored by the second.

### 3.1 The network as a graph

```text
  NODE-ARC GRAPH      capacities in thousand tonnes per year; node type in brackets

   [S1 Zeebrugge]                          [S2 Genoa]
        |  sea 12                              |  sea 8
        v                                       v
   +-------------+                        +-------------+
   | P1 Venlo    |                        | P2 Lyon     |
   | plant  14   |                        | plant   9   |
   +-------------+                        +-------------+
      |       |                              |       |
  road|  8    | road 6                  road | 5     | road 4
      v       v                              v       v
  +--------+ +--------+               +--------+ +--------+
  | D1     | | D2     |               | D3     | | D4     |
  | DC  6  | | DC  5  |               | DC  4  | | DC  4  |
  +--------+ +--------+               +--------+ +--------+
      |  \      |   \                    |           |
      v   v     v    v                  v           v
    [Z1] [Z2] [Z3]  [Z4]             [Z5]         [Z6]
    4.2  5.1  3.8   2.3              2.9          0.9
    demand zones — fixed, not decided; figures are thousand tonnes per year
```

Read the drawing for what it does **not** say. It does not say which arcs are one-way, whether "DC 6"
is a capacity on the building or the inbound road leg, what the rate is, whether a minimum charge
applies, or whether the sea leg is priced per tonne or per container. A picture can hold a network;
it cannot hold a rate structure.

### 3.2 The same network as a lane table

```text
  SAME NETWORK AS THREE RELATIONS

  nodes(id, type, fixed_cost, throughput_cap, open_is_decision)
    S1  source   0      -     no        S2  source   0      -     no
    P1  plant    0      14    no        P2  plant    0      9     no
    D1  dc       820     6    yes       D2  dc       640     5    yes
    D3  dc       710     4    yes       D4  dc       900     4    yes
    Z1  demand   0      -     no        ... Z6   (EUR thousand/yr, kt/yr)

  arcs(from, to, mode, km, days, cap_kt, rate_structure)
    S1  P1  sea   310  3  12   EUR 28/t + EUR 240/shipment, 1 TEU minimum
    S2  P2  sea   420  4   8   EUR 31/t + EUR 260/shipment, 1 TEU minimum
    P1  D1  road   35  0.5  8  EUR 0.092/t-km, EUR 120 minimum per shipment
    P1  D2  road  790  1.0  6  EUR 0.088/t-km, EUR 120 minimum per shipment
    P2  D2  road   15  0.3  5  EUR 0.095/t-km, EUR 95 minimum per shipment
    P2  D4  road  465  0.8  4  EUR 0.090/t-km, EUR 120 minimum per shipment
    D1  Z1  road  175  0.5  6  EUR 0.30/t-km, EUR 60 minimum, EUR 45 fixed per drop
    D2  Z3  road  465  0.9  5  EUR 0.30/t-km, EUR 60 minimum, EUR 45 fixed per drop
    ... (14 further lanes, same schema)

  flows(commodity, period, origin, destination, quantity, unit)
    fasteners  2027-Q1  P1  D1  1850  KGM
    fasteners  2027-Q1  D1  Z1   410  KGM
```

The second drawing is longer and uglier and it is the one that earns its keep. Every column is a
decision somebody made, and every decision can be wrong in a way the picture hides. The
`rate_structure` column is the one that most often turns out to be a single number where a piecewise
function was needed.

### 3.3 Four properties worth memorising

**Property 1 — Every attribute belongs to exactly one object.** Apply the removal test: if the
object is removed and the cost disappears, the cost belongs there; if the cost survives but moves,
it belongs elsewhere. Close the DC and the lease disappears, so the lease is a node attribute. Close
the DC and the road leg still exists but goes somewhere else, so the road leg is an arc attribute.
When an attribute appears twice, the model double counts, and the double count nearly always
penalises the option under evaluation.

**Property 2 — The problem family follows from two structural questions.** May flow pass through an
intermediate node, and is opening a node a decision. No-no gives a transportation problem; yes-no
gives a transshipment problem; any-yes gives a facility-location problem. The first two are linear
programs with totally unimodular constraint matrices, so their relaxations have integral optima and
they solve in polynomial time. The third breaks that property, is NP-hard in general, and is why
topics 03 to 10 exist.

**Property 3 — An arc is a function, not a number.** The cost of moving goods is a function of
weight, distance, shipment size, mode, and time, and it is usually piecewise: a minimum charge below
a threshold, rates that step down at break weights, a fixed component per shipment, a fuel surcharge
that moves monthly, accessorials for waiting and special handling. Replacing that function with one
per-unit number is the most consequential simplification in the representation, and it is made
silently in most spreadsheets.

**Property 4 — Cost-to-serve is a second graph on the same nodes.** The physical graph says where
goods can go; the cost graph says who pays for what. Most reporting failures come from reading a
cost graph off a physical graph never designed to carry it — allocating all warehouse cost by
weight when the warehouse's own cost is driven by lines and cube.

### 3.4 The decomposition of delivered cost

| Component | Where it accrues | Typical driver | Traced or allocated |
|---|---|---|---|
| Product cost | Plant or supplier | Units × standard cost | Traced to item |
| Inbound freight | Plant-to-DC arc | Tonne-km, shipment count | Traced to receipt |
| Outbound freight | DC-to-customer arc | Tonne-km, drops, minimum charge | Traced to shipment |
| Handling | DC node | Lines, cases, pallets | Traced if scanned |
| Storage | DC node | Pallet-days, cube occupied | Allocated unless measured |
| Order processing | Order | Orders, lines, manual touches | Traced to order |
| Returns | Reverse arc | Return units, reason code | Traced if authorised |
| Overhead | Network | Various | Allocated |

The right-hand column is the honest one and the one most presentations omit: a figure that is 60%
allocated is a measurement of the allocation rule with the customer's name attached.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Node | A place where material stops, is transformed, or changes custody | Treated as a synonym for warehouse | Location identifier, e.g. a GLN |
| Arc | A directed connection between two nodes that flow may use | Drawn as an undirected line on a slide | Origin–destination pair plus mode |
| Lane | An arc as procured commercially, with a rate, transit time, and validity | Used for a geographic route | O–D plus mode plus rate reference |
| Flow | Quantity moved on an arc in a period | Confused with demand or capacity | Commodity × period × unit |
| Transshipment node | A node that receives and forwards flow without net consumption | Modelled as source and sink at once | Throughput tonnes per period |
| Cross-dock | A transshipment node with near-zero storage time | Costed as a DC with storage | Throughput per day, doors |
| DC | Distribution centre; a storage node between inbound and outbound legs | Used for any intermediate node | Pallet positions, throughput |
| Customer zone | An aggregation of ship-to points treated as one demand node | Treated as a real place | Count of ship-to points inside |
| Centroid | The demand-weighted mean location of a zone | Assumed to be the cost-minimising representative | Projected coordinates |
| Aggregation | Replacing many demand points with fewer representatives | Assumed cost-neutral | Ratio of original to aggregate nodes |
| Directed arc | An arc usable in one direction only | Ignored because "roads run both ways" | Ordered from–to pair |
| Arc capacity | Maximum flow the arc can carry in the period | Confused with vehicle capacity | Tonnes or TEU per period |
| Throughput capacity | Maximum flow a node can process in the period | Applied to inbound arcs only, or twice | Tonnes per period |
| Fixed charge | A cost incurred if a node is used, independent of volume | Applied per unit "for simplicity" | Currency per period |
| Rate structure | The function mapping shipment attributes to a price | Replaced by a single average rate | Piecewise function plus validity dates |
| Minimum charge | A floor price per shipment below which the computed charge does not fall | Omitted, understating small-shipment cost | Currency per shipment |
| Break weight | The weight at which the next band's lower rate gives a lower total charge | Confused with a discount threshold | Kilograms |
| CWT rate | A rate per hundredweight; the traditional less-than-truckload (LTL) quoting basis | Applied to metric tonnes without conversion | Currency per 100 lb |
| Fuel surcharge | A per-shipment or percentage adder tracking a published fuel index | Treated as a fixed percentage of the rate | Percent of base, or currency per mile |
| Accessorial | A charge for a service beyond dock-to-dock movement | Assumed to sit inside the line-haul rate | Currency per event |
| Transit time | Elapsed time from departure to availability | Reported as one number, not a distribution | Days, with a percentile |
| Distance metric | The rule used to compute distance between two nodes | Left unstated in the model documentation | Great-circle, road, routed |
| Circuity factor | Road distance divided by great-circle distance for a lane | Assumed constant at 1.2 everywhere | Dimensionless ratio |
| Load factor | Share of a vehicle's capacity actually used | Reported by weight only, ignoring cube | Percent of weight or volume |
| Load equivalent | A shipment expressed as a fraction of a standard vehicle load | Used without naming the vehicle | Truckloads, TEU, pallet-equivalents |
| TEU | Twenty-foot equivalent unit; the standard container measure | Used for non-containerised freight | Count of 20 ft containers |
| Cube | Volumetric measure of a shipment | Ignored when product is light and bulky | Cubic metres |
| Tonne-kilometre | One tonne moved one kilometre | Used as the sole driver when cube binds | t·km |
| Cost pool | A group of costs sharing a single driver | Pooled too coarsely to allocate | Currency per period |
| Cost driver | The measure of activity that causes a pool to be incurred | Chosen for convenience or availability | Units of the driver |
| ABC | Activity-based costing; cost to activities, then to objects via drivers | Used as a synonym for any allocation | Method, not a unit |
| C2S | Cost-to-serve; cost attributable to a specific order, customer, or channel | Reported as total logistics cost divided by units | Currency per order or customer-period |
| Marginal cost | The change in total cost from one more unit of activity | Used in reporting, where it understates | Currency per additional unit |
| Average cost | Total cost divided by total volume | Used for decisions, where it misleads | Currency per unit |
| Scope boundary | The set of costs included in a stated C2S figure | Left implicit, making figures incomparable | Enumerated cost list |
| Traced cost | Cost assigned from direct observation of the object's consumption | Called allocated when any rule was used | Percent of total cost |
| Allocated cost | Cost assigned by rule from a shared pool | Presented as if measured | Percent of total cost |
| Total landed cost | Product plus freight, duty, and fees to a named place | Reported excluding duty or inbound | Currency per unit at a location |
| Transportation problem | Balanced single-commodity flow, sources to sinks, no intermediate nodes, no fixed costs | Used for every network problem | Problem family |
| Transshipment problem | Minimum-cost flow permitting intermediate nodes | Confused with the transportation problem | Problem family |
| Facility-location problem | Choosing which nodes to open and assigning demand | Used loosely for any network model | Problem family |
| Total unimodularity | A matrix property guaranteeing integral extreme points for the relaxation | Assumed to survive fixed charges | Property of a matrix |

Four of these deserve expansion.

**Lane versus arc.** An arc is a structural possibility. A lane is a commercial object with a
contract, a validity period, and a rate structure. A model needs the arc to represent possibility
and the lane to represent price. When a company re-tenders its freight, the arcs do not change and
every lane's rate does. A data model that merges them cannot express "the same physical possibility,
priced differently next quarter", which is the most common scenario in network work.

**Minimum charge versus fixed charge.** A fixed charge is incurred whatever the volume, once a node
is used. A minimum charge is a per-shipment floor on a variable formula. They look similar in a
table and behave differently in a model. A minimum charge makes cost per kilogram fall as shipments
consolidate; a fixed charge makes it fall as volume grows at a site. Confusing them produces a
recommendation to close the wrong thing.

**Cube versus weight.** Freight is priced on the greater of actual and volumetric weight at a stated
density. For dense goods the weight binds and tonne-kilometres are natural. For light bulky goods
the cube binds, and a model built entirely in tonnes understates the bulky products and overstates
the dense ones. This is a representation error, not a costing error, and it surfaces as a product-mix
recommendation nobody can explain.

**Traced versus allocated.** A cost is traced when the object's own consumption is observed: the
order shows four lines picked, so four lines of picking cost are traced, not allocated. A cost is
allocated when a rule spreads a shared pool. Both appear in a report with the same formatting. Only
one is evidence.

## 5. The mechanics, step by step

Each step states its input, its output, and the branch point where the judgement is made. Steps 1
to 8 build the graph; steps 9 to 11 classify the problem; steps 12 to 14 build and reconcile the
cost surface.

### Phase A — Build the graph

**Step 1. Fix the decision, and therefore the boundary.**

- Input: a decision statement of the form "should we, by when, at what cost".
- Output: a scope note naming the decision, the horizon, and what is excluded.
- Reasoning: the boundary determines which costs are inside the objective and which are
  externalities the recommendation must mention. A study that excludes working capital recommends
  fewer, larger sites; one that includes it recommends more, smaller sites, because in-transit and
  safety inventory rise with distance.
- Branch point: is the decision reversible within the horizon? If not, the model needs a transition
  path, not a steady state, and the boundary must include closure and relocation costs.

**Step 2. Enumerate nodes by function, not by name.**

- Input: site master, customer master, contracts.
- Output: a node table with a `type` column and a rule for what each type may do.
- Reasoning: type determines the constraint set. A plant cannot receive finished goods; a
  cross-dock may receive and forward within a day but may not hold. Modelling a cross-dock as a
  small DC with a storage cost gives it a reason to hold stock it does not have, and the model will
  use it.
- Branch point: which node types are decision variables and which are fixed. Opening a DC is a
  binary; using a port is not.

**Step 3. Enumerate arcs and decide direction.**

- Input: node table and transport contracts.
- Output: an arc list with mode and direction.
- Reasoning: use directed arcs unless the rate is genuinely symmetric. Direction matters when the
  rate differs by direction (headhaul and backhaul pricing), the capacity differs, the transit time
  differs, or returns must be modelled on the reverse arc at a different cost. In European road
  freight the first and fourth almost always hold.
- Branch point: an undirected model can satisfy demand by shipping "backwards" along a lane whose
  real rate is asymmetric, and the objective will be optimistic by the difference.

**Step 4. Put attributes on the correct object.**

- Input: contracts, leases, tariff tables.
- Output: node and arc attribute tables with no duplicated financial fields.
- Reasoning: apply the removal test from section 3.3.
- Branch point one: delivered pricing. If a supplier quotes delivered duty paid (DDP) to the DC,
  inbound freight sits inside the product price and must not be charged again on the inbound arc.
  This is the most common double count in imported-goods networks.
- Branch point two: throughput capacity. "The DC can handle 6,000 tonnes a year" is a node
  attribute. "The inbound contract allows 40 loads a week" is an arc attribute. Putting both in
  caps throughput twice, and the tighter silently governs.

**Step 5. Declare the flow unit.**

- Input: product master with weight, cube, and unit-of-measure data.
- Output: a named flow unit, stated in the model documentation and used everywhere.
- Reasoning: the flow unit must be the one that binds the cost. Dense industrial goods: tonnes.
  Light consumer goods: cubic metres or pallet positions. Containerised imports: TEU. Parcel
  networks: parcels and stops. A model in tonnes for a furniture business misprices every lane,
  because the trailer cubes out at 12 tonnes of a 26-tonne payload.
- Branch point: if two units bind in different parts of the network, convert at a stated density,
  and make the density a parameter rather than a constant buried in a rate.

**Step 6. Declare the period and the time treatment.**

- Input: planning horizon, seasonality of demand and rates.
- Output: a period index and a decision on single-period versus multi-period.
- Reasoning: a single-period model uses annual averages and cannot represent a peak season, a
  phased build-out, or a contract expiring mid-horizon. A multi-period model multiplies variables by
  the number of periods, which is why many studies become unsolvable at the grain chosen.
- Branch point: if the recommendation is a steady-state footprint, single-period with a peak
  capacity check is usually enough. If it is a sequence of openings, it is not.

**Step 7. Aggregate demand, with a stated error budget.**

- Input: ship-to points, volumes, coordinates.
- Output: a zone list and a measured aggregation error.
- Reasoning and branch points: section 5.2. The short version is that aggregation compresses the
  demand relation, it is never free, and the loss must be measured rather than assumed.

**Step 8. Choose the distance metric.**

- Input: coordinates, a routing engine, or historic driven distances.
- Output: a distance matrix with the metric named in its metadata.
- Reasoning and branch points: section 5.3. The choice is worth 15 to 40% of line-haul distance in
  typical European road freight, which is larger than most effects being studied.

### 5.1 What belongs on an arc and what belongs on a node

| Item | Belongs on | Why | Symptom if misplaced |
|---|---|---|---|
| Building lease | Node | Disappears if the site closes | Every site looks expensive; too few open |
| Inbound freight on a DDP purchase | Neither — inside product cost | Paid by the supplier, not the network | DC cost inflated; distant DCs look attractive |
| Inbound freight on an FCA purchase | Arc | Survives closure but moves to another lane | Double count if also inside a handling rate |
| Picking labour | Node, driven by lines | Scales with activity at the site | Small-order customers look cheap |
| Delivery stop charge | Arc, per shipment | Scales with drops, not tonnes | Model consolidates drops it cannot consolidate |
| Safety stock | Node, derived | Scales with lead time in and out | Far sites look cheap; service fails later |
| Duty | Arc or node, depending on the customs regime | Incurred at the border crossing | Landed cost wrong by the tariff rate |
| Fuel surcharge | Arc, percentage of base | Varies by month and mode | Right on average, wrong in the month that matters |

**Picking labour is not a per-tonne cost.** It is driven by order lines, and the ratio of lines to
tonnes varies by an order of magnitude across customers. A customer buying one heavy item per order
has a low line count and a high weight; one buying twenty small items has the reverse. Allocating
picking cost by weight makes the second look cheap and the first expensive, and the resulting
ranking is worse than no ranking.

**Safety stock is a derived node cost.** If the model treats inventory carrying as a fixed
percentage of throughput, it never sees the mechanism that makes centralisation expensive: lead time
into and out of a node drives safety stock through roughly the square root of lead time, so a node
serving a long lane needs more buffer. Module 10 gives the policy; the network model inherits its
output rather than re-parameterising it.

### 5.2 The aggregation problem

Every network model aggregates. A national distributor may have 40,000 ship-to points; no
formulation carries them. The question is not whether to aggregate but how much, and the obligation
is to measure the resulting error rather than inherit a zone map.

Aggregation does two distinct things, and they behave differently.

**It replaces many line-haul distances with one.** The direction depends on the representative. If
the representative is the demand-weighted centroid, the modelled distance falls, always, by
Jensen's inequality: for a depot at `p` and demand locations `X`,

```text
E[ |p - X| ]  >=  |p - E[X]|
```

The average distance to the individual customers is at least the distance to their mean. The gap is
the within-zone dispersion, and it is largest when the zone spans the depot: a zone with points on
both sides of a DC has a centroid close to the DC and an actual average distance much larger.

**It deletes the within-zone delivery distance.** This term is unambiguously negative and is almost
always omitted. Aggregating a metropolitan area to one point removes the run from the local depot to
each customer, which for dense urban delivery is 10 to 30 km per drop and is often most of the drop
cost. No representative choice recovers it, because the model no longer knows the customers exist.

```text
  AGGREGATION ERROR, ONE CLUSTER, SERVED FROM D1

   Z1 Rotterdam  o-----------------------------> 203.5 km road
       4,200 t           \  66.5 km to centroid
                          x  centroid (-39, 36)
       5,100 t            /  135.2 km to largest node
   Z2 Essen      o--------+-------------------> 135.2 km road
                        D1 Venlo (0,0)

   true demand-weighted distance   166.1 km
   modelled with centroid           66.5 km   -60.0%
   modelled with largest node      135.2 km   -18.6%
```

**Why centroid aggregation persisted despite the known bias.** Four reasons, none technical.

1. **It was the only computable summary.** Through the 1970s and 1980s, models ran on machines holding
   a few hundred nodes. A centroid compresses any number of ship-to points into two coordinates, and
   two coordinates fit.
2. **It is exactly optimal for a different problem.** The centroid minimises the sum of weighted
   *squared* distances. If the cost model used squared distance, as gravity models of retail
   interaction did, the centroid would be right. The error is a category mistake inherited from a
   neighbouring literature.
3. **The error is invisible from inside.** Seeing it requires the finer model, which costs as much as
   the first study, and nobody funds the study twice.
4. **Models are inherited.** The zone map is in the legacy system, the legacy system is the baseline,
   and the baseline is what the new study must reconcile to.

The remedy is a two-level run: build at the inherited aggregation and at twice the resolution, and
report the difference. If it is below the gap between the leading designs, the aggregation is
adequate for this decision. Otherwise refine. That is a stopping rule tied to the decision, not to a
target node count. As starting ranges: 150 to 400 zones for a national network with dozens of
candidate sites; 30 to 80 for a continent-scale study whose decision is about countries; no
aggregation for a tactical lane-level sourcing study.

> **Note:** The literature is unambiguous that the error is real and can be large. Limbourg and
> Jourquin, studying rail-road terminal location on a 200-node network, report optimality errors of
> several percent even at modest reduction ratios, growing as the number of facilities falls and the
> aggregation coarsens. Gavriliouk, reviewing aggregation in hub location, notes that exact hub
> models handle on the order of 100 to 200 nodes while real networks run to thousands. Quote the
> direction and order of magnitude for your own network, not a borrowed percentage.

### 5.3 The distance metric

| Metric | What it measures | Ratio to great-circle | When to use |
|---|---|---|---|
| Great-circle | Shortest surface path over a sphere | 1.00 by definition | Screening only, or no routing data |
| Road network | Shortest path on a road graph, ignoring traffic | 1.15 to 1.40 in dense European regions | Strategic models with a calibrated factor |
| Practical routed | Fastest path with restrictions, ferries, congestion | 1.20 to 1.50 typical; much higher for short urban trips | Operational costing and service promises |

The ratio is the **circuity factor**, and it is not constant. It falls with trip length, because a
long trip spends proportionally less time on the local grid. It rises with physical barriers and
with origin density. It is systematically higher for short urban freight trips than for intercity
line-haul, which matters because short trips are where delivery cost concentrates.

Published estimates span a wide range, and the spread is itself the finding. Ballou, Rahardja and
Sakai report country-level factors from roughly 1.12 to 2.10, with European values averaging near 1.46.
Qureshi, Hwang and Chin find a constant near 1.22 gives acceptably unbiased aggregate estimates for
United States commodity flows, though individual origin-destination pairs deviate far more. Heinold
and Makowski report European road circuity of roughly 1.15 to 1.76, median near 1.25, and rail circuity
median near 1.42. Urban last-mile studies report much higher factors, because a two-kilometre delivery
is dominated by one-way streets and barriers rather than by geography.

Three rules follow. **Calibrate, do not assume** — if driven distances exist for the lanes carrying
most of the volume, fit the factor to those and report the residual. **Stratify by distance band** —
one factor fitted across 20 km drops and 900 km trunk hauls will fit neither. **Name the metric in
the deliverable** — the same physical lane has three different lengths depending on who computed it.

### 5.4 Units, load factors, and the conversions that decide the answer

| Conversion | Typical value | Depends on | Failure if assumed |
|---|---|---|---|
| Each to case | 6, 12, or 24 units | Pack configuration | Cost off by the pack multiple |
| Case to pallet | 40 to 120 cases | Case dimensions, pallet standard | Handling cost wrong by 2 to 3 times |
| Pallet to cubic metre | 1.2 to 1.8 m³ | Stack height and overhang | Cube-based pricing wrong |
| Pallet to tonne | 0.4 to 1.0 t | Product density | Mode and vehicle choice wrong |
| Tonne to load | 24 to 26 t payload for a 40 t articulated vehicle in Europe | Vehicle configuration and weight limits | Freight cost wrong by the load factor |
| Cubic metre to load | 60 to 100 m³ usable in a 13.6 m trailer | Trailer height and double-stacking | Cube-bound product undercosted |

The **load factor** is the bridge between shipment and vehicle and the number most often missing from
a rate table. A lane quoted at €1,150 per full truckload (FTL) is €46 per tonne at 25 tonnes and €96 at
12 tonnes; converting at the full-load assumption underprices every lane serving a small market.

> **Note:** Vehicle weight limits are regulated, not conventional. European international road
> freight is generally limited to 40 tonnes gross under Directive 96/53/EC, with 44 tonnes for
> certain intermodal movements and higher national limits in some member states. Payload is gross
> minus tare, so a 40-tonne vehicle carries roughly 24 to 26 tonnes. Verify the limit that applies to
> your lane.

### 5.5 Classifying the problem

| Family | Permits | Forbids | Matrix property | Guarantee |
|---|---|---|---|---|
| Transportation | Direct flow from sources to sinks | Intermediate nodes, fixed costs | Totally unimodular | LP optimum is integral; polynomial |
| Transshipment | Flow through intermediate nodes | Fixed costs | Totally unimodular | LP optimum is integral; polynomial |
| Facility location | Choosing which nodes to open | Nothing, but pays for it | Not totally unimodular | NP-hard; MILP with a gap |

The constraint matrix of a transportation or transshipment problem is the node-arc incidence matrix
of a directed graph. Every column has one +1 and one −1, so the matrix is totally unimodular, so
every extreme point is integral when the right-hand side is integral. Since an optimum is attained at
an extreme point, solving the relaxation solves the integer problem. Add a fixed charge multiplied by
a capacity and the property is lost. Everything in topics 03 to 10 is a consequence of that single
structural loss.

**Where the formulations came from.** Hitchcock formulated the transportation problem in 1941, in a
paper on distributing a product from several sources to numerous localities, sketching a pivoting
procedure and a complementary-slackness optimality condition. Kantorovich had formulated the same
problem in 1939 in the Soviet Union, in work unknown in the West for about a decade, and Koopmans
arrived at it independently while working on Allied shipping logistics — hence the
Hitchcock–Koopmans label. The transshipment extension is associated with Orden's 1955 paper and with
the network-flow methods of Ford and Fulkerson. The practical consequence is small; the intellectual
one is that the model you are writing is a seventy-year-old object with a well-understood structure,
and the interesting questions are all in the data.

### Phase B — Build the cost surface

**Step 9. Draw the cost boundary explicitly.** Input: the profit and loss account, freight ledger, and
site cost centres. Output: an enumerated list of included and excluded lines. The eight standard
components are product cost, inbound freight, outbound freight, handling, storage, order processing,
returns, and overhead. Branch point: is the figure for pricing, customer management, or network design.
Pricing needs the marginal cost of the next order; customer management needs the fully loaded cost;
network design needs the cost that varies with the design. Three boundaries, three numbers, one
customer.

**Step 10. Build pools at the activity level.** A pool is useful only if it has one dominant driver. A
pool called "warehouse cost" mixing labour, rent, and equipment cannot be allocated defensibly,
because no single driver explains it. Split until each pool has a driver that plausibly causes most of
its cost.

**Step 11. Choose drivers that are causally linked.** The test is causal, not statistical: a driver is
defensible if a change in it would cause a roughly proportional change in the pool. Order count drives
order processing. Line count drives picking. Pallet-days drive storage occupancy. Tonne-kilometres and
drop count drive delivery. Revenue drives nothing in a warehouse and is the most commonly used driver
in practice. Branch point: if two drivers are plausible and correlated, choose the one the operational
manager can change. An allocation the manager cannot influence is a tax, not a signal.

### 5.6 The C2S reporting standard

| Statement | Content | Why required |
|---|---|---|
| Scope boundary | Which of the eight components, and which entity's costs | Figures with different boundaries are not comparable |
| Drivers | Driver per pool, with measured explanatory power | A figure without drivers cannot be challenged |
| Period | Fiscal period, volumes, actual or normalised | Seasonal cost per order swings by 30% or more |
| Traced share | Percentage directly traced rather than allocated | Below roughly 50%, the figure is mostly a rule |

Report the traced share at the top of any C2S deliverable, where a financial statement reports its
currency. "62% traced to the order, 38% allocated by pallet-day" tells the reader how much weight the
number bears. "The cost to serve this customer is €84 per order" tells them nothing.

### 5.7 The decile matrix

Rank customers by C2S or net contribution, group into ten bands, plot cumulative profit. The shape
that emerges is consistent enough to be a stylised fact: a minority generate more than total profit,
a middle band is approximately break-even, and a bottom band consumes a disproportionate share. The
usually quoted form, from Kaplan's customer-profitability work, has the top 15 to 20% generating
100% or more of profits while the bottom 5 to 20% destroy a large share. Magnitudes vary widely by
study and industry and should be treated as that case's finding, not a benchmark.

Two cautions. Below about fifty customers, deciles are too small to be stable, and quartiles or a
ranked list are more honest. And the matrix measures the past under current prices and service
levels: a customer who looks unprofitable at list may be profitable at the negotiated price, and one
who looks profitable may be consuming unbilled service. It is a diagnostic, not a verdict.

## 6. Worked example

Verhoeven BV distributes industrial fasteners from two overseas suppliers through four DCs to six
customer zones. Figures are illustrative and internally consistent; distances come from a simplified
grid multiplied by a circuity factor, so the arithmetic is reproducible by hand.

```text
  COORDINATES (km on a projected grid)          DEMAND (thousand tonnes per year)
    D1 Venlo    (0, 0)        Z1 Rotterdam  (-160, -30)   4.2
    D2 Lyon     (620, -430)   Z2 Essen      (  60,  90)   5.1
    D3 Milan    (780, -180)   Z3 Paris      ( 170,-380)   3.8
                              Z4 Lyon       ( 610,-420)   2.3
                              Z5 Milan      ( 760,-190)   2.9
                              Z6 Barcelona  ( 180,-900)   0.9
```

### 6.1 Distances and the base case

Road distance is grid distance multiplied by a circuity factor of 1.25, close to the European median
in the literature cited in section 5.3.

| Lane | Straight-line km | Road km at 1.25 | Demand kt | Thousand t-km |
|---|---:|---:|---:|---:|
| D1 to Z1 | 162.8 | 203.5 | 4.2 | 854.7 |
| D1 to Z2 | 108.2 | 135.2 | 5.1 | 689.5 |
| D2 to Z3 | 452.8 | 566.0 | 3.8 | 2,150.8 |
| D2 to Z4 | 14.1 | 17.7 | 2.3 | 40.7 |
| D3 to Z5 | 22.4 | 28.0 | 2.9 | 81.2 |
| D3 to Z6 | 937.2 | 1,171.5 | 0.9 | 1,054.4 |
| **Total** | | | **19.2** | **4,871.3** |

At €0.088 per tonne-kilometre, base-case outbound line-haul is `4,871,300 × 0.088 = €428,674` per
year. This is the number the aggregated models are compared against.

### 6.2 Aggregating six zones to three

Clusters: North (Z1, Z2 from D1), South (Z3, Z4 from D2), Mediterranean (Z5, Z6 from D3). Two
representative rules are compared.

```text
Rule A, demand-weighted centroid
  North: x = (4,200 x -160 + 5,100 x 60) / 9,300   = -39.35
         y = (4,200 x  -30 + 5,100 x 90) / 9,300   =  35.80
  South: x = (3,800 x  170 + 2,300 x 610) / 6,100  = 335.90
         y = (3,800 x -380 + 2,300 x -420) / 6,100 = -395.10
  Med:   x = (2,900 x  760 +   900 x 180) / 3,800  = 622.60
         y = (2,900 x -190 +   900 x -900) / 3,800 = -358.20

Rule B, largest node
  North represented by Z2, South by Z3, Mediterranean by Z5
```

| Cluster | True demand-weighted distance | Rule A | Error A | Rule B | Error B |
|---|---:|---:|---:|---:|---:|
| North (from D1) | 166.1 | 66.5 | −60.0% | 135.2 | −18.6% |
| South (from D2) | 359.3 | 357.8 | −0.4% | 566.0 | +57.5% |
| Med (from D3) | 373.6 | 297.2 | −20.4% | 28.0 | −92.5% |

Read the South row first, because it is the least dramatic and most instructive. The centroid error
is 0.4%, because Z4 sits almost on top of D2, so the centroid lands near the demand-weighted average
of two points already close in cost terms. Aggregation error is not a constant tax; it depends on
where the depot sits relative to the zone's dispersion.

Now the Med row. Rule B represents a 3,800-tonne cluster by its largest member, Milan, 28 km from the
Milan DC. Barcelona, 1,171 km away carrying 900 tonnes, disappears. The modelled cost of the cluster
falls by 92.5%. No analyst would choose this rule deliberately, and yet it is what happens whenever a
zone is named after its biggest city and the model reads the city's coordinates.

### 6.3 The error in money

```text
  BASE CASE (six zones)
    4,871,300 t-km x 0.088                               = 428,674 EUR

  RULE A (three centroids)
    North   9,300 x   66.5 =   618,450
    South   6,100 x  357.8 = 2,182,580
    Med     3,800 x  297.2 = 1,129,360
    Total                   = 3,930,390 t-km
    x 0.088                                            = 345,874 EUR   -19.3%

  RULE B (three largest nodes)
    North   9,300 x  135.2 = 1,257,360
    South   6,100 x  566.0 = 3,452,600
    Med     3,800 x   28.0 =   106,400
    Total                   = 4,816,360 t-km
    x 0.088                                            = 423,840 EUR    -1.1%
```

Rule B's net error is only 1.1%, and that number is a coincidence. It understates the Mediterranean
cluster by €116,000 and overstates the South cluster by €110,000, and the two nearly cancel. A design
built on Rule B would conclude that Barcelona is cheap to serve from Milan and Paris expensive from
Lyon, and would act on both. Netting to zero is not the same as being right.

### 6.4 The within-zone term no representative recovers

Both rules still model one delivery per cluster. In reality each zone contains dozens of ship-to
points served by local runs. Suppose the average distance from the zone representative to the
individual customers is 18 km, travelled for the whole zone volume.

```text
  deleted within-zone tonne-km
    North   9,300 x 18 = 167,400
    South   6,100 x 18 = 109,800
    Med     3,800 x 18 =  68,400
    Total              = 345,600 t-km  ->  30,413 EUR per year   (7.1% of base)
```

Add that to Rule A's 19.3% and the centroid model understates outbound cost by roughly a quarter. On
a network where the recommendation turns on a 5% difference, an error that size decides the answer.

> **Note:** The 18 km figure is an assumption and should be measured from delivery records. The
> point is not the number; it is that the term exists, has a sign, and is normally absent.

The six-zone representation compares designs whose costs differ by more than about 20%, not those that
differ by 5%. The three-zone centroid model is unusable for any decision, because its error exceeds
any plausible design difference. Barcelona warrants a different service model — local cross-dock,
another mode, a longer accepted lead time — because its cost per tonne is 5.7 times the network
average, a ratio that survives any aggregation error here. No solver was needed: the representation
and the arithmetic usually locate the decision first.

## 7. Code, query, or calculation

A C2S model for ten customers from four pools, computed by hand. The difficulty is in driver choice
and pool boundaries, not in the arithmetic.

**Assumptions.** One shipment per order. Pools: (A) order processing €38.00 per order; (B) warehouse
handling €1.15 per line; (C) outbound freight €45.00 per shipment plus €0.30 per tonne-kilometre,
floored at a €60.00 minimum charge; (D) returns €16.00 each at 1.5% of orders. Gross margin is the
realised product margin before logistics.

```text
Customer Orders Lines    kg/yr   km   Pool A  Pool B  Pool C  Pool D     C2S   GM%      Net
C01         520  6240  210000   310   19760    7176   42931     128   69995  15.0  320005
C02         340  3400  128000   260   12920    3910   25283      80   42193  15.0  220307
C03         240  1920   78000   420    9120    2208   20628      64   32020  16.0  147180
C04         160   960   41000   180    6080    1104    9600      32   16816  17.0   88584
C05         120   480   19000   540    4560     552    8478      32   13622  14.0   33978
C06          96   288    9600   610    3648     331    6077      16   10072  13.0   17228
C07          84   168    4800   150    3192     193    5040      16    8441  12.0    7399
C08          72   108    2900   700    2736     124    4320      16    7196  11.0    2484
C09          96   115    1600    95    3648     132    5760      16    9556  10.0   -4156
C10          48    48     620   380    1824      55    2880      16    4775   9.0   -1985

Freight C01:  210000/520 = 403.8 kg/shipment ; 0.4038 x 310 = 125.2 t-km
              45 + 0.30 x 125.2 = 82.56 ; 82.56 x 520 = 42,931
Freight C09:  1600/96 = 16.7 kg ; 45 + 0.30 x 0.0167 x 95 = 45.48 < 60
              minimum applies ; 60 x 96 = 5,760
Cost/order:   C01 134.6 ; C05 113.5 ; C09 99.5 ; C10 99.5
Break-even order value = cost per order / gross margin rate
              C01: 134.6 / 0.15 =   897 vs actual 5,000  -> profitable
              C09:  99.5 / 0.10 =   995 vs actual   563  -> loss
              C10:  99.5 / 0.09 = 1,106 vs actual   646  -> loss
Total net contribution 831,024 EUR ; top two customers 540,312 = 65.0% of it
```

**Expected output and how to read it.** Two customers are unprofitable and one, C08, is marginal at
34% headroom. The interesting result is not the two losses but the flatness of cost per order across
a tenfold range of order value: C01's average order is worth €5,000 and C10's €646, and the cost to
process, pick, and deliver them is €135 and €100.

- **C09 is worse than C10 despite being larger.** C09 has €54,000 of revenue over 96 orders; C10 has
  €31,000 over 48. C09's order value is €563 against €646, so C09 is further below the
  order-processing break-even despite being the bigger account. Size does not predict profitability;
  order cadence does.
- **The absolute money is small.** Exiting or repricing C09 and C10 recovers €6,141, which is 0.7% of
  net contribution. The value is in the middle of the distribution, where customers sit just above
  break-even and a change in ordering behaviour or a modest price adjustment moves them materially.
  Presenting only the loss-making tail routinely understates the case for C2S.

**Where the calculation would break.** It assumes one shipment per order, so a customer taking four
part-deliveries is undercosted fourfold on freight. It assumes the minimum charge applies per
shipment, true for pallet freight and false for parcel, where it is per parcel. It assumes a flat
1.5% return rate, which hides the customers whose rate is 15%.

## 8. Data, standards, and regulatory context

The graph and the cost model are unregulated objects. Nobody audits the structure of a lane table.
What is regulated is everything the table refers to: the identity of places and products, the units,
the weight limits, the emissions factors, and the statements the model must reconcile to.

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GLN | Location identity | The party and place identifier across order, shipment, and event data | B |
| GS1 GTIN | Product identity | The trade item that demand, weight, and cube attach to | B |
| GS1 SSCC | Logistics unit identity | The pallet or case label, hence the handling unit | B |
| GS1 EPCIS 2.0 | Event data | What moved, where, when, why, and in whose custody | B |
| UN/CEFACT Recommendation 20 | Unit codes | Binding by agreement in electronic trade documents; `KGM`, `MTQ`, `C62` | B |
| ISO 80000 | Quantities and units | Reference definitions, not an exchange format | B |
| ISO 6780 | Pallet dimensions | Four regional sizes recognised; not one global standard | B |
| ISO 14083:2023 | Transport emissions | Quantification and allocation rules between consignments | B |
| Directive 96/53/EC | Vehicle weights and dimensions | 40 t general limit for EU international road transport | B |
| Incoterms 2020 | Delivery responsibility | Who pays which leg, hence which cost sits where | B |
| IFRS 15 / ASC 606, IFRS 8 | Revenue and segment reporting | When revenue exists; the granularity at which profit must be defensible | B |
| Regulation (EU) 2023/956 | CBAM carbon cost on covered imports | Definitive regime from 1 January 2026 | B |

Three consequences are worth spelling out.

**Identity defects come first.** If your node identifiers are internal site codes rather than GLNs,
demand will not join to the customer master, which will not join to delivery records, and the join
will be done by postcode matching with a loss nobody measures. The model's first data defect is
usually an identity defect, not a value defect.

**Unit codes belong in the parameter table.** A rate quoted per 100 lb (hundredweight, or CWT) and a
demand table in kilograms differ by 2.2046, and the resulting cost is wrong by that factor without
looking wrong. State the Recommendation 20 code, not just a column name.

**Cost-to-serve has no accounting standard.** There is no IFRS that says which driver to use or which
pool to define. The only binding constraints are that product cost reconciles to the ledger and that
segment disclosures under IFRS 8 are consistent with the total. Everything else is method, defensible
on causal grounds rather than by appeal to a standard. That absence is why the reporting standard in
section 5.6 matters: in an unaudited domain, the analyst's own disclosure is the only quality control.

> **Note:** Pallet standards are a real trap. EUR pallets are 1200 × 800 mm, ISO pallets 1200 ×
> 1000 mm, North American pallets 1219 × 1016 mm. A trailer holds 33 EUR or 30 ISO pallets in a
> single tier, so a model that converts "pallets" to "truckloads" without naming the pallet is wrong
> by a tenth on every cube-bound lane.

What is convention rather than requirement: "150 to 400 customer zones", "circuity factor 1.2",
"freight is 8% of revenue", "80/20 customer profitability", "cost per order is the C2S unit". Each is
a working default that describes no specific business. Treating any of them as a constraint prevents
challenge.

## 9. Industry practice and case evidence

**Procter & Gamble, North American product supply, 1997.** Camm and colleagues built a large-scale
mixed-integer model covering plants, distribution centres, and customer zones, combining optimisation
with managerial judgement and a geographic information system (GIS) so the answer could be inspected
by people who do not read linear programs. The reported savings are a modelling projection reviewed
by the company, not an audited post-implementation result. The transferable element is reconciliation:
the model was credible because it reproduced existing accounting data before it was used to argue for
change. Grade A for the method, C for the savings.

**Digital Equipment Corporation, global supply chain, 1995.** Arntzen and colleagues built a
multi-commodity, multi-period global model including production, distribution, duties, and time as
well as cost. Its lasting contribution to representation was the insistence that cost alone misranks
global options when lead time differs, which forced inventory and time into the objective rather than
into a post-hoc adjustment. Grade A.

**National Semiconductor, centralised distribution.** Regional warehouses were consolidated into one
facility, with air freight holding the delivery promise. Documented in the supply chain literature and
taught in standard texts; figures are case-reported, not audited. The structural lesson is the
exchange rate: centralisation converts inventory cost into freight cost, and the rate of exchange is
the freight rate on the affected lanes. When that rate moves, the answer moves. Grade C.

**How cost-to-serve entered practice.** Activity-based costing (ABC) reached a management audience
through Cooper and Kaplan in 1988, arguing that multi-product firms make pricing and mix decisions on
distorted cost information because they allocate overhead by volume. The argument transferred to
customers a decade later: Kaplan's 1997 note applied ABC to customer mix, linking cost-to-serve to the
net margin earned with individual customers, and the whale-curve presentation comes from the same
body of work. Kaplan's 2005 summary of a distributor case reports the top 40% of customers generating
around 130% of annual profits while the worst 5% lost around 30%. Those are case figures reported by
an academic summarising an engagement, not a benchmark. Grade C for the figures, A for the method.

The practice arrived through third-party logistics (3PL) contracts rather than management accounting.
Through the mid and late 1990s, as shippers outsourced distribution, contracts moved from unit-rate
schedules toward open-book and cost-plus arrangements, and gain-sharing clauses appeared to give the
provider a share of the savings it generated. Trade coverage documents both the enthusiasm and the
scepticism: gain-sharing was reported as catching on in 1996, and the pitfalls of cost-plus were being
debated in the trade press by 1998. By the mid-2000s the practitioner taxonomy had settled into
transactional unit rates, activity-based costing, and open-book cost-plus. A 2003 practitioner review
drawing on earlier council research reported gain-sharing as the sole remuneration mechanism in a
small single-digit percentage of United States contracts.

This history explains a persistent habit: many companies' C2S definitions were written to satisfy a
contract clause, not to support a decision. A definition designed to adjudicate a gain-share is
auditable and stable, which is not the same as decision-relevant. When you inherit a C2S model, ask
which contract it was built for. Grade E for the chronology, D for the taxonomy, and treat adoption
percentages as indicative.

| Finding | Grade |
|---|---|
| Aggregation error grows with the reduction ratio and as the number of facilities falls | A |
| Exact hub models handle roughly 100 to 200 nodes; larger networks need aggregation or heuristics | A |
| Country road circuity ranges roughly 1.12 to 2.10, European average near 1.46 | A |
| A constant near 1.22 is acceptably unbiased in aggregate for US flows, with large pair errors | A |
| European road circuity median near 1.25, falling with distance; rail median near 1.42 | A |
| Urban last-mile circuity far higher, reported means around 2.5 in one dense city | C |

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Cost on both node and arc | DC cost per pallet × volume exceeds its total cost centre | Duplicated inbound freight or handling | Sites look expensive; model consolidates too far | Apply the removal test to every field |
| Undirected arcs on asymmetric lanes | Return legs cost the same as outbound in the rate table | Direction collapsed at data load | Optimistic cost; phantom backhaul savings | Keep direction; model reverse arcs |
| Centroid aggregation, no measured error | Zone file has one row per city, not per ship-to cluster | Within-zone distance deleted; centroid closer than average | Understated outbound cost, worst in dense markets | Two-level run; report the delta |
| Within-zone delivery omitted | Cost per tonne identical for a metro and a rural zone | Last-leg distance never entered | Near market looks cheaper than it is | Add a measured local delivery term |
| Great-circle distance as road distance | Lane costs 20% below the freight ledger on long lanes | Circuity ignored | Network biased toward distant sites | Calibrate against driven distances |
| Weight allocation when cube binds | Light products consistently look cheaper to serve | Density difference ignored | Product-mix recommendation nobody believes | Model the binding dimension per family |
| Average rate on a marginal decision | One rate per lane; the rate card has five bands | Structure flattened at load time | Wrong answer on small shipments | Carry the rate function, or the band |
| Minimum charge dropped | Small orders cost less than any invoice on file | Floor price omitted | Small customers look viable when they are not | Apply a per-shipment floor |
| Driver chosen for availability | Revenue or units appears as driver for three unrelated pools | No causal link between driver and cost | Numbers move with revenue, for no operational reason | Justify causally and record it |
| C2S without a scope statement | Two departments report different C2S for one customer | Different boundaries, same label | Debate about numbers instead of the customer | Publish all four statements |
| Deciles on too few customers | Ten customers presented as ten deciles | Small-sample instability | Spurious rankings | Quartiles below about fifty customers |

Two deserve a note. **Centroid aggregation without a measured error** survives review because the
model looks entirely normal: zones, distances, rates, a solution. Nothing in the output says the last
fifty kilometres are missing; the only reliable test is a finer run. **Driver chosen for availability**
survives audit because the arithmetic is correct; the defect is that the rule has no causal content.
Diagnostic: if changing a customer's revenue without changing anything operational changes their C2S
rank, the driver is wrong.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The node and arc set is fixed over the horizon | The model represents a fixed structure | New lanes, new DCs, or a 3PL change invalidate the graph | Compare the arc list to last year's; count additions |
| Cost scales with the stated driver over the modelled range | Linear cost functions are required | Step changes, volume discounts, fixed steps are invisible | Plot cost against driver; look for steps |
| The rate structure is stable over the period | One rate per lane in the objective | A tender changes every lane | Check contract expiry against the horizon |
| Demand is deterministic at the modelled grain | The base model has no uncertainty | Forecast error moves the answer more than the design difference | Run scenarios; compare to the design gap |
| Distance is proportional to cost at a fixed rate | Distance is the arc cost driver | Congestion, tolls, terrain change the ratio by lane | Regress lane cost on distance; check residuals |
| The load factor is constant per lane | Converts per-vehicle rates to per-unit | Consolidation changes the factor, which changes the rate | Compare modelled to actual load factor |
| Aggregation error is small relative to the decision | Compresses demand to zones | Error exceeds the gap between designs | Two-level run; report the delta |
| Returns are a small stable fraction of flow | Simplifies reverse flow | High-return channels have reverse costs comparable to forward | Compute return rate by channel |
| The traced share supports the conclusion | Distinguishes measurement from rule | A 30% traced figure is mostly allocation | Compute and report the traced share |

**Two limits of the framing.** The graph simplifies a contractual reality: consignment stock owned by
the supplier but held at a customer site, customer-collected freight whose carrier is invisible, and
shared-user warehouses priced per pallet position rather than per activity. None fits the node-arc-flow
vocabulary cleanly; each needs a stated approximation. C2S measures the past at current prices and
service — it locates the conversation, it does not settle it.

## 12. Exercises

### Exercise 1 — Draw the graph and the lane table (recall and application, 45 minutes)

Take a distribution network you know. Draw it twice: as a node-arc picture with capacities, and as
the three relations in a table.

**Deliverable:** one page with both drawings and a list of five attributes you could not place
confidently.
**Acceptance criteria:** at least eight typed nodes; ten arcs with mode, distance, transit time,
capacity, and a rate structure in words; three flow rows with commodity, period, and unit; each
uncertain attribute carries a one-line reason.

### Exercise 2 — Classify three problems (application, 30 minutes)

Classify and state what each permits and forbids: (a) a brewer assigning weekly production from three
breweries to forty depots, all sites fixed, no intermediate storage; (b) adding two regional
cross-docks that receive and forward within a day; (c) deciding whether to close one brewery.

**Deliverable:** a three-row table with family, permitted structures, forbidden structures, guarantee.
**Acceptance criteria:** correct family for each; total unimodularity stated for (a) and (b) and its
loss explained for (c).

### Exercise 3 — Find the double count (critique, 40 minutes)

Given a model where product cost includes freight to the DC under DDP terms; the inbound arc is
charged at €0.07 per tonne-kilometre; the DC has a handling rate of €4.20 per pallet "including
inbound receipt"; and the lease is a fixed cost. Identify every duplication and omission.

**Deliverable:** a corrected attribute table, one row per cost item, with its home and a one-line
removal-test justification.
**Acceptance criteria:** at least two duplications found; one cost legitimately in two places
identified and justified; the direction of the resulting bias stated.

### Exercise 4 — Aggregation error budget (calculation, 60 minutes)

Add Z7 (Strasbourg, 1,400 tonnes, coordinates (240, 40)) served from D1 to the section 6 data.
Recompute the base case, then aggregate Z1, Z2, Z7 into one cluster under both rules.

**Deliverable:** base, centroid, and largest-node costs with percentage errors, plus a paragraph on
whether the three-zone model is adequate for a decision whose options differ by 6% of total cost.
**Acceptance criteria:** arithmetic to the nearest €100; within-zone term computed separately; a
clear yes or no with the error budget named.

### Exercise 5 — Distance metric audit (application, 45 minutes)

For five lanes with known driven distances, compute great-circle distance, routed distance, and the
ratio.

**Deliverable:** a five-row table and a fitted circuity factor with its residual range.
**Acceptance criteria:** every column names its metric; one lane outside the published range is
explained; the factor is offered per distance band, not as one number.

### Exercise 6 — Unit and load-factor conversion (calculation, 40 minutes)

A product ships 24 eaches per case, 60 cases per EUR pallet, 0.62 tonnes per pallet. A 13.6 m trailer
takes 33 pallets single-stacked. Annual volume is 480,000 eaches; the lane rate is €980 per trailer.

**Deliverable:** pallets per year, trailers at full load, cost per each, and cost per each at a 70%
load factor.
**Acceptance criteria:** all four numbers correct; units stated at each step; one sentence naming
which conversion drives the result.

### Exercise 7 — Ten-customer cost-to-serve (application, 75 minutes)

Rebuild section 7, then apply one change: C03 moves to two drops per order with order count unchanged.

**Deliverable:** the ten-row table, break-even order value per customer, and a ranked list with
cumulative profit share.
**Acceptance criteria:** every pool reconciled to a stated total; break-even formula applied
correctly; at least one customer crosses from profitable to unprofitable, and the crossing is
explained.

### Exercise 8 — Allocation dispute memo (critique and design, 50 minutes)

Two managers dispute a C2S figure. Operations allocates warehouse cost by pallet-days; sales wants
revenue. Write the memo that resolves it.

**Deliverable:** a one-page memo naming the decision, the causally linked driver, the treatment of the
other driver as a secondary view, and the measurement that would settle it.
**Acceptance criteria:** the choice is justified causally, not by preference; the memo states what
each driver does to the top three customers; no appeal to accounting standards, because none applies.

### Exercise 9 — Write the reporting standard (design, 45 minutes)

Turn the four-part standard into a template a team could apply to any C2S deliverable.

**Deliverable:** a one-page template, an example filled in for the Verhoeven case, and a worked
example of a figure that would fail it.
**Acceptance criteria:** each statement has a required format; traced share is a percentage with its
computation; the failing example is realistic and its defect is named.

## 13. Self-check answers

**Exercise 1.** This is assessed against the representation, not the drawing. A strong answer has
eight typed nodes, at least one of each of source, plant, DC, and demand; ten arcs whose rows carry
mode, distance, transit time, capacity, and a rate structure expressed as a function rather than a
number; and flow rows with commodity, period, and a unit code. It lists five attributes it could not
place, and each one is genuinely ambiguous — a throughput figure that could be a node or an arc
capacity, a freight rate that could be delivered or origin-priced. A weak answer has boxes and arrows,
one rate per lane, no capacities, and no attributes it is unsure about, which usually means the
ambiguity was resolved silently rather than absent.

**Exercise 2.** (a) Transportation problem: single commodity, no intermediate nodes, no fixed costs.
The matrix is the incidence matrix of a bipartite graph, hence totally unimodular, hence the
relaxation has an integral optimum and the problem is polynomial. It forbids transshipment and forbids
closing a brewery. (b) Transshipment: intermediate nodes are permitted, flow conservation applies at
them, and total unimodularity survives, so the integrality guarantee survives. It still forbids fixed
costs and the choice of whether a cross-dock exists. (c) Facility location: the binary decision enters
as a fixed charge times a capacity, total unimodularity is lost, and the problem becomes NP-hard. The
guarantee changes from "the relaxation is integral" to "optimal within a reported gap, or a heuristic
with no guarantee". The common wrong answer treats all three as "a transportation problem with extra
columns", which is right for (a) and misses the only structural break.

**Exercise 3.** Duplications: (1) freight to the DC sits inside the DDP product cost and again on the
inbound arc — the supplier pays it, so remove the arc charge; (2) inbound receipt sits inside the
€4.20 pallet rate and is also represented by the inbound arc's activity. Inbound receipt is either a
DC activity, in which case it belongs in the handling rate and not on the arc, or a transport
activity, in which case the reverse. It cannot be both. Legitimate double presence to check: the lease
is correctly a fixed node cost and correctly appears in the DC cost centre total, but must not also be
inside a per-pallet rate. Bias: every DC's operating cost is overstated by the inbound freight share,
and because that share scales with distance from the plant, distant DCs are penalised more, biasing
the answer toward production rather than demand. A cost genuinely in two places is duty: on the border
arc for modelling, inside product cost for accounting, carried exactly once in the objective.

**Exercise 4.**

```text
  Z7 Strasbourg: straight 243.3 km x 1.25 = 304.1 km ; 1,400 x 304.1 = 425,740 t-km
  New base:      5,297,020 t-km x 0.088 = 466,138 EUR

  North cluster, 10,700 t
    centroid  x = -2.80 , y = 36.36 ; 36.5 km straight = 45.6 km road
              modelled 10,700 x 45.6 = 487,920 t-km
    true demand-weighted distance = 184.1 km
    largest node (Z2)  10,700 x 135.2 = 1,446,640 t-km

  centroid      487,920 + 2,182,580 + 1,129,360 = 3,799,860 -> 334,388 EUR  (-28.3%)
  largest node 1,446,640 + 3,452,600 + 106,400 = 5,005,640 -> 440,496 EUR   (-5.5%)
  within-zone, 18 km x 20,600 t = 370,800 t-km -> 32,630 EUR (7.0% of base)
```

Verdict: no, the three-zone model is not adequate for a 6% decision. The centroid rule understates by
28.3% and the largest-node rule by 5.5% before the within-zone term; after it, the largest-node error
is roughly 12% in the same direction. Note that adding Z7 made the centroid error worse, not better:
Strasbourg pulls the centroid closer to D1 while adding a 304 km leg, exactly the mechanism in section
5.2. Refine and re-measure.

**Exercise 5.** No single fitted factor is correct. A defensible answer labels the great-circle model
(spherical and ellipsoidal differ by up to about 0.5%), reports the factor per distance band with the
residual range, and explains any lane outside the published range by a physical cause — an estuary, a
mountain range, a ferry, or a border with long queues. A fit of 1.24 with residuals from 1.09 to 1.41
is a different object from 1.24 with residuals from 1.20 to 1.28. A weak answer reports one factor for
all five lanes; if those lanes mix a 30 km urban drop with a 900 km trunk haul, the residuals will
show the fit is meaningless, and the answer should say so.

**Exercise 6.**

```text
480,000 / 24 per case        = 20,000 cases
20,000 / 60 per pallet       = 333.3 pallets
333.3 x 0.62 t               = 206.7 t
333.3 / 33 per trailer       = 10.10 trailers
10.10 x 980 EUR              = 9,899 EUR  ->  0.0206 EUR per each

at 70% load factor
33 x 0.70 = 23.1 pallets     ; 333.3 / 23.1 = 14.43 trailers
14.43 x 980 = 14,143 EUR     ->  0.0295 EUR per each   (+43%)
```

The load factor drives the result, not the pack configuration: it changes the answer by 1.43, whereas
an error of one case per pallet changes it by 1.7%. That ranking is typical, and it is why load factor
belongs in the model as a parameter rather than in a footnote.

**Exercise 7.** With two drops per order and the order count unchanged, C03 doubles the shipment
count at half the size:

```text
C03:  480 shipments of 162.5 kg over 420 km
      0.1625 x 420 = 68.25 t-km ; 45 + 0.30 x 68.25 = 65.48 (above the 60 minimum)
      480 x 65.48 = 31,430 ; new C2S = 42,822 ; net = 179,200 - 42,822 = 136,378  (profitable)

C08:  144 shipments of 20.1 kg over 700 km
      45 + 0.30 x 0.0201 x 700 = 49.23 -> 60 minimum applies
      144 x 60 = 8,640 ; new C2S = 11,516 ; net = 9,680 - 11,516 = -1,836  (crosses to a loss)
```

Splitting a delivery does not double cost when each part load still clears the minimum: C03's freight
rises 52%, not 100%. And the crossing happens at the customer nearest the boundary, which here is C08
rather than either of the two already-loss-making accounts. A customer with 34% headroom absorbs a 60%
freight increase and survives; one with no headroom does not. That is why the marginal customers, not
the loss-making tail, are where the analysis earns its keep.

**Exercise 8.** Begin by asking what the number will support. If it is deciding whether to renegotiate
the service agreement, the relevant driver is the one the customer can influence. Shared-facility
warehouse cost is driven by occupied pallet positions over time, because the building's capacity and
rent are consumed by occupancy, not by sales value. Pallet-days is therefore primary and causally
linked. Revenue allocation is not defensible as a primary driver: a customer's sales value causes no
warehouse activity. It can appear as a labelled secondary view, because it answers a legitimate
different question — what share of cost the customer's revenue would bear under a value-based pricing
convention — but it is a convention, not a measurement. State what each driver does to the top three
customers: if the largest customer is also the longest-dwell customer, both agree and there is no
dispute; if the largest holds fast-moving stock and a mid-sized customer holds slow-moving stock,
pallet-days lifts the mid-sized one and revenue keeps them down, and the driver choice decides the
conversation. The measurement that settles it is an occupancy extract by customer over twelve weeks
including a peak. If occupancy cannot be measured, neither driver is evidenced and the traced share is
low.

**Exercise 9.** A passing template has four labelled fields with required formats. Scope boundary: an
enumerated list of included components from the eight, plus the legal entity and volume basis — not
"full logistics cost". Drivers: one line per pool with pool name, annual value, driver, and share of
variance explained — not "allocated on activity". Period: fiscal period, volume, actual or normalised,
seasonality adjustment — not "annual". Traced share: a percentage computed as traced cost over total
cost in scope, with the tracing basis for each component — not omitted. A figure that fails: "our cost
to serve this account is €84 per order". It has no boundary, no drivers, no period, and no traced
share. It is not a wrong number; it is an uninterpretable one, and the correct response in a meeting is
to ask the four questions rather than argue about the value.

## 14. Interview questions

**1. "Transportation problem versus facility location?"** Strong: transportation permits only direct
flow, has no fixed costs, and its constraint matrix is totally unimodular so the relaxation is
integral; facility location adds a binary open decision, breaks total unimodularity, and is NP-hard.
Weak: one is "bigger", or facility location "adds capacity", which misses the structure.

**2. "Where does inbound freight belong, arc or node?"** Strong: apply the removal test, then add the
DDP exception where the supplier pays and the cost is inside product price — the most common double
count in imported-goods models. Weak: "on the arc, because it is transport", which is right half the
time.

**3. "Why does centroid aggregation understate transport cost?"** Strong: two mechanisms — the centroid
is closer than the average customer by Jensen's inequality, and the within-zone delivery distance is
deleted entirely, which is unambiguously negative and almost always omitted. Error is largest when a
zone spans the depot. Weak: "aggregation loses detail", true of every model and explanatory of nothing.

**4. "How many customer zones?"** Strong: refuses to answer with a number first; the count follows a
measured error budget, and the budget is set by the gap between the designs compared — build at two
resolutions, measure the delta, refine until it is well below that gap. Ranges offered only as a
starting point. Weak: "about 200", as though it were a property of the problem.

**5. "A lane costs €0.088 per tonne-kilometre. What do you ask?"** Strong: which distance metric
produced the kilometres, whether the rate is an average across a band or the rate at modelled volume,
whether the minimum charge was applied, and whether the load factor is actual. Weak: whether the number
is high or low, which cannot be answered without a comparison.

**6. "Average versus marginal cost?"** Strong: average is total over volume and belongs in reporting
and tariffs; marginal is the change from the next decision and belongs in decisions; they diverge most
where the rate structure is piecewise, which is where small-shipment decisions live. Weak: marginal is
"more accurate", which is not a property either number has.

**7. "Why is cost-to-serve hard to standardise?"** Strong: there is no accounting standard for it —
IFRS governs revenue recognition and segment disclosure, not cost attribution to customers — so the
analyst's own disclosure of boundary, drivers, period, and traced share is the only control. Weak: "data
is siloed", which is a symptom.

**8. "What is a break weight?"** Strong: the weight at which the next band's lower rate yields a lower
total charge, computed as the band weight times its rate divided by the current rate; it makes cost per
kilogram discontinuous, and shippers exploit the step. Weak: a discount threshold.

**9. "How do you test whether an allocation driver is defensible?"** Strong: the causal test — would a
change in the driver cause a roughly proportional change in the pool — plus the practical test, that
changing a customer's revenue without operational change should not move their C2S rank. Weak: "industry
practice" or "what the ERP does".

**10. "C2S shows the smallest 10% of customers are unprofitable. What now?"** Strong: check the absolute
money, often under 1% of contribution, then look at customers near break-even where the value is; treat
the finding as pricing and service-design input, not an exit list; ask whether cadence rather than size
causes it. Weak: recommend exiting the customers.

**11. "You inherit a model with 40 zones, straight-line distances, and one rate per lane. First move?"**
Strong: do not rebuild. Measure the three errors in likely order of materiality — rate structure first,
then distance metric, then aggregation — and report each against the decision it supports. Weak:
"rebuild it properly", which is correct and unfundable.

## 15. Key terms

- **Node** — a place where material stops, is transformed, or changes custody.
- **Arc** — a directed connection between two nodes that flow may use.
- **Lane** — an arc as procured commercially, with a rate, a transit time, and a validity period.
- **Flow** — quantity moved on an arc, indexed by commodity and period.
- **Transshipment node** — a node that receives and forwards flow without net consumption.
- **Cross-dock** — a transshipment node with near-zero storage time.
- **Customer zone** — an aggregation of ship-to points treated as one demand node.
- **Centroid** — the demand-weighted mean location of a zone.
- **Aggregation error** — the cost difference between a fine and a coarse demand representation.
- **Circuity factor** — road distance divided by great-circle distance.
- **Load factor** — the share of a vehicle's capacity actually used.
- **TEU** — twenty-foot equivalent unit, the standard container measure.
- **Tonne-kilometre** — one tonne moved one kilometre; the standard freight activity unit.
- **Rate structure** — the function mapping shipment attributes to a price.
- **Minimum charge** — a floor price per shipment.
- **Break weight** — the weight at which the next band's lower rate yields a lower total charge.
- **Accessorial** — a charge for a service beyond dock-to-dock movement.
- **Cost pool** — a group of costs sharing a single driver.
- **Cost driver** — the measure of activity that causes a pool to be incurred.
- **ABC** — activity-based costing; cost to activities, then to objects via drivers.
- **C2S** — cost-to-serve; cost attributable to a specific order, customer, or channel.
- **Break-even order value** — cost per order divided by the gross margin rate.
- **Decile matrix** — customers ranked and grouped by cost-to-serve or profitability.
- **Traced share** — the proportion assigned by direct observation rather than by rule.
- **Total unimodularity** — a matrix property guaranteeing integral extreme points.
- **Transportation problem** — balanced single-commodity flow, no intermediate nodes, no fixed costs.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Hitchcock, F. L. (1941). "The Distribution of a Product from Several Sources to Numerous
  Localities." *Journal of Mathematics and Physics*, 20, 224–230.
- Kantorovich, L. V. (1939). *Mathematical Methods in the Organization and Planning of Production*.
  Leningrad State University. English translation in *Management Science*, 6(4), 1960.
- Orden, A. (1956). "The Transhipment Problem." *Management Science*. Cited by title and venue; volume
  and page details should be checked before formal citation.
- Ford, L. R., and Fulkerson, D. R. (1962). *Flows in Networks*. Princeton University Press.
- Schrijver, A. (2002). "On the history of the transportation and maximum flow problems."
  *Mathematical Programming*. DOI: 10.1007/s101070100259.
- Cooper, R., and Kaplan, R. S. (1988). "Measure Costs Right: Make the Right Decisions." *Harvard
  Business Review*, September–October 1988.
- Kaplan, R. S. (1997). "Using ABC to Manage Customer Mix and Relationships." Harvard Business School
  Background Note 197-094.
- Kaplan, R. S., and Anderson, S. R. (2004). "Time-Driven Activity-Based Costing." *Harvard Business
  Review*, November 2004.
- Kaplan, R. S. (2005). "A Balanced Scorecard Approach to Measure Customer Profitability." *HBS Working Knowledge*, 8 August 2005.
- Camm, J. D., Chorman, T. E., Dill, F. A., Evans, J. R., Sweeney, D. J., and Wegryn, G. W. (1997).
  "Blending OR/MS, Judgment, and GIS: Restructuring P&G's Supply Chain." *Interfaces*, 27(1).
- Arntzen, B. C., Brown, G. G., Harrison, T. P., and Trafton, L. L. (1995). "Global Supply Chain
  Management at Digital Equipment Corporation." *Interfaces*, 25(1).
- Limbourg, S., and Jourquin, B. (2007). "Rail-Road terminal locations: aggregation errors and best
  potential locations on large networks." *European Journal of Transport and Infrastructure
  Research*, 7(4). DOI: 10.18757/ejtir.2007.7.4.3402.
- Gavriliouk, O. (2009). "Aggregation in hub location problems." *Computers and Operations Research*.
- Ballou, R. H., Rahardja, H., and Sakai, N. (2002). "Selected country circuity factors for road travel
  distance estimation." *Transportation Research Part A*.
- Qureshi, M. A., Hwang, H.-L., and Chin, S.-M. (2002). "Comparison of Distance Estimates for Commodity
  Flow Survey: Great Circle Distances Versus Network-Based Distances." *Transportation Research
  Record*, 1804. DOI: 10.3141/1804-28.
- Heinold, A., and Makowski, M. (2026). "Driving the Extra Mile: Comprehensive Analysis of Road and
  Rail Transportation Distances in Europe." *Networks and Spatial Economics*.
- van Niekerk, H. C., and Bean, W. L. (2019). "Evaluation of the cost drivers and allocation framework
  in outbound logistics of the fast-moving consumer goods industry." *South African Journal of
  Industrial Engineering*.

**Standards, regulation, and official — Grade B**

- GS1 General Specifications, including GLN, GTIN, SSCC, and EPCIS 2.0.
- UN/CEFACT Recommendation 20, Codes for Units of Measure Used in International Trade.
- ISO 80000, Quantities and units; ISO 6780, Flat pallets for intercontinental materials handling.
- ISO 14083:2023, Greenhouse gases — Quantification and reporting of greenhouse gas emissions arising
  from transport chain operations; and the Smart Freight Centre GLEC Framework.
- Directive 96/53/EC on weights and dimensions of road vehicles in the European Union.
- Incoterms 2020, International Chamber of Commerce.
- IFRS 15 and ASC 606 on revenue from contracts with customers; IFRS 8 on operating segments.
- Regulation (EU) 2023/956 establishing the carbon border adjustment mechanism; definitive regime from
  1 January 2026.

**Company cases and practitioner sources — Grade C**

- Miller, T. (2017). "The ABCs of Activity-Based Costing for Logistics." *Material Handling and
  Logistics*.
- Seifert, R. W., and Markoff, R. (2018). "The hidden cost of cost-to-serve." IMD Business School.
- Carrier less-than-truckload rate guides, including minimum charge, weight break, and hundredweight
  tables published by Purolator and comparable carriers, and explanatory guides published by carriers
  and freight brokers. Used for rate mechanics only; rates themselves are not evidence.
- National Semiconductor centralised distribution case, as reported in standard supply chain
  textbooks. Case-reported, not independently audited.
- Merchán, D. (2018). Urban logistics circuity, MIT thesis. Not peer-reviewed; order of magnitude only.

**Vendor, analyst, and trade — Grade D and E**

- Gartner, "Toolkit: Allocating Logistics Costs for a Cost-to-Serve Analysis," November 2024. Vendor
  methodology, content gated. Grade D.
- ILOS, "Forms of PSL remuneration — from price schedules to gain sharing," practitioner review drawing on council research. Grade D.
- "Logistics 'Gain-Sharing' Catches On." *American Shipper*, December 1996. Grade E.
- Spira, R. M. (1998). "Pitfalls of Cost-Plus Contracts." *Traffic World*, 16 November 1998. Grade E.

> **Note on citations.** Case figures are reported as their sources state them and are not
> independently verified. Where a source could not be checked in full — the Orden paper's volume
> details, the gain-sharing adoption percentages — the text says so rather than quoting precision it
> cannot support.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Part map: [`../_index.md`](../_index.md)
- Prerequisite and related: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Transport operations and routing: [`../12_transportation_routing_last_mile/_index.md`](../12_transportation_routing_last_mile/_index.md)
- Warehouse cost mechanics: [`../13_warehousing_fulfillment_labor/_index.md`](../13_warehousing_fulfillment_labor/_index.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)
- Sibling topic: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- Sibling topic: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- Sibling topic: [`04_facility-location.md`](04_facility-location.md)
- Sibling topic: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- Sibling topic: [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md)
- Sibling topic: [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md)
- Sibling topic: [`08_robust-stochastic.md`](08_robust-stochastic.md)
- Sibling topic: [`09_carbon-objective.md`](09_carbon-objective.md)
- Sibling topic: [`10_solvers-heuristics.md`](10_solvers-heuristics.md)
