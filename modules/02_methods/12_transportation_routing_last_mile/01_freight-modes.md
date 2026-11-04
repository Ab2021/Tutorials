# Freight Modes and Their Economics — cost architecture, unit economics, and mode choice

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topic 01 (units of measure); Module 10 topic 04 (safety stock)
> Version 2026.5

---

## 1. Why this topic exists

A shipper switches a lane from full truckload to less-than-truckload and books a freight saving of
eleven per cent. The saving appears in the transport P&L in the month it is taken. The inventory
that the change requires does not appear anywhere, because inventory is a balance-sheet number
and the carrying rate that would convert it into a cost is owned by finance, disputed, and applied
to a different denominator. Two years later the company is still running the lane, still reporting
the saving, and still short of stock on the days the less-than-truckload trailer misses its
connection. Nobody connects the two facts, because they live in different reports.

This is the ordinary failure of mode selection, and it is expensive in a way that does not show up
in any single line. Mode choice is not a purchasing decision about a rate. It is a decision about
which cost curve the company will operate on for the life of the contract, and the costs that
differ between modes are not all in the freight invoice.

Three specific errors recur.

1. **The comparison is made on a rate, not on a cost.** Two quotes are compared per kilogram. One
   is a linehaul rate with a fuel component and a minimum charge; the other is an all-in price for
   a vehicle. They are not the same object, and the difference is not a rounding error. On the lane
   worked in section 6, the per-kilogram view and the per-cubic-metre view rank the same two options
   in opposite order.
2. **Transit time is treated as a number.** A carrier quotes four days. The mean is four days and
   the ninety-fifth percentile is nearly seven. The mean is the sales story; the tail is what the
   safety stock pays for. The protection-period logic that converts the tail into inventory is
   derived in `../10_inventory_optimization_multi_echelon/04_safety-stock.md` and is used here, not
   re-derived.
3. **A mode shift is reported as a carbon result without an allocation rule.** "We moved thirty per
   cent of volume from air to ocean and cut transport emissions by forty per cent" is not evidence
   unless the boundary, the allocation basis, the load factor, and the empty-running treatment are
   stated. ISO 14083:2023 supplies the calculation structure; a percentage without it is a claim.

The topic sits at the front of the module for a reason. Everything later — rate structures,
tender acceptance, consolidation, routing, dispatch, last mile — is an optimisation inside a mode.
Choosing the mode chooses the feasible set. A brilliant route plan inside the wrong mode is a
well-executed mistake.

> **Note:** This file covers the six modes and their cost architecture. Rate structures, fuel
> surcharges, and accessorial mechanics are the subject of `02_rate-structures.md`; lane
> segmentation and tender acceptance are `03_lane-tender-analytics.md`; weight-and-cube
> consolidation is `04_consolidation-cube.md`; vehicle routing is `05_vrp-formulation.md`. Each is
> named here only where the mode comparison cannot proceed without it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a shipment by its three currencies — mass, cube, and time — and **state** which one
   binds for each candidate mode.
2. **Draw** the cost architecture of full truckload (FTL), less-than-truckload (LTL), parcel,
   intermodal rail, air freight, and ocean (full container load, FCL, and less than container load,
   LCL), naming the dominant component and the shape of the curve it produces.
3. **Compute** cost per tonne-kilometre and cost per cubic metre for the same shipment, and
   **explain** why light bulky freight inverts the ranking between them.
4. **Derive** the LTL-to-FTL break-even weight, **state** the five conditions under which the
   derivation holds, and **compute** the break-even for a stated lane.
5. **Represent** transit time as a distribution, **convert** its tail into a safety-stock
   requirement using the protection-period formula, and **quantify** the inventory consequence of a
   mode change.
6. **Build** a total-cost comparison across linehaul, accessorials, inventory in transit, service
   failure, and emissions, and **defend** the ranking.
7. **Apply** ISO 14083:2023 and the GLEC Framework well enough to **critique** a mode-shift
   emissions claim, including the allocation rule it does not state.
8. **Distinguish** dedicated from common carriage in cost structure, flexibility, and negotiating
   position, and **choose** between them for a stated volume pattern.
9. **Write** a mode-decision record that states the basis, the date, the assumptions, and the
   conditions that would reverse the decision.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

### 3.1 Every move has five segments

A freight movement, whatever the mode, is five segments stacked end to end. The segments are the
same; the modes differ in how much of the total each one carries.

| Segment | What happens | Cost behaviour |
|---|---|---|
| Access | Collection at origin, drayage, first mile to a terminal or port | Per stop, per shipment, or per container move |
| Linehaul | The long move between terminals, hubs, ports, or directly between origin and destination | Per kilometre, per container slot, or per kilogram |
| Handling | Sort, cross-dock, lift, transload, customs examination | Per shipment per touch, per container lift |
| Delivery | Final leg, appointment, inside delivery, proof of delivery | Per stop, per shipment |
| Administration | Documentation, billing, claims, customs brokerage, compliance | Per shipment |

The single most useful consequence: **every handling step is a place where cost is added and
capacity is shared**. FTL shares nothing — one shipper, one vehicle, door to door. LTL shares the
trailer and pays for the terminal twice. Parcel shares the trailer, the hub, the delivery van, and
the route. Intermodal shares a train and pays for two terminal lifts and two drayage legs. Ocean
shares a vessel and pays for terminal handling at both ends. Air shares a hold and pays for
screening, handling, and often a truck leg at each end.

Cost per unit falls as the shared asset fills. That sentence explains most of the economics in
this file.

### 3.2 The dominant component sets the shape of the curve

A rate is a sum of components, and one of them usually dominates. The dominant component decides
how the cost responds to weight, to volume, and to distance. You can predict the shape before you
see a single quote.

| Mode | Dominant component | Charged per | Shape against weight | Shape against distance |
|---|---|---|---|---|
| FTL | Vehicle and driver time | Trip | Flat until the payload limit, then unavailable | Near-linear |
| LTL | Terminal handling plus a share of linehaul | Shipment plus kilogram or cubic metre | Declining per kilogram, stepwise at handling breaks | Near-linear with a fixed terminal term |
| Parcel | Sortation and delivery stop | Piece | Flat per piece through weight bands, then step | Zone bands, not continuous |
| Intermodal rail | Train slot plus two lifts and two drayage legs | Container | Flat per container | Low marginal cost per kilometre, high fixed terminal cost |
| Air | Payload lifted | Chargeable kilogram | Proportional to chargeable weight | Proportional |
| Ocean FCL | Vessel slot | Container | Flat per container | Flat per voyage, near-zero marginal per kilometre |
| Ocean LCL | Consolidation and container freight station handling | Weight or measure, whichever is greater | Proportional above the weight-or-measure threshold | Flat per voyage plus station charges |

Read the table as a set of instructions about where to look. On an FTL lane, the question is
whether you can fill the vehicle and how many kilometres the driver must cover. On an LTL lane,
the question is how much terminal handling your shipment attracts and whether it is charged on
weight or on cube. On an ocean lane, distance barely matters within a trade lane; the rate is a
slot price set by capacity, and the marginal cost of an extra thousand kilometres is close to
zero.

### 3.3 Three currencies: mass, cube, and time

Every shipment can be described in three currencies, and every mode charges in one or more of
them.

- **Mass**, in kilograms or tonnes. The currency of air, and of dense freight in every mode.
- **Cube**, in cubic metres or pallet positions. The currency of parcel, of light LTL, and of LCL.
- **Time**, in days and in the variance of days. The currency of inventory, and the reason a slower
  mode can be more expensive than a faster one even when its rate is lower.

A rate quote always converts the shipment into one of these, and the conversion is where the money
is. A parcel carrier converts cube into a fictitious weight using a dimensional divisor. An air
carrier converts cube into chargeable kilograms using a 1:6,000 cubic-metre-per-kilogram
convention. An LCL carrier converts using the weight-or-measure rule, in which one cubic metre is
treated as one tonne. An LTL carrier converts using a class-based rate table. Each conversion is a
policy, not a physical law, and each one can be changed by a packaging decision.

### 3.4 Design density is the hinge

Every piece of equipment has a payload and a volume. The ratio is its **design density**: the cargo
density at which the vehicle fills both at once. Cargo denser than the design density is
weight-limited; cargo lighter is volume-limited.

| Equipment | Payload (kg) | Volume (m³) | Design density (kg/m³) |
|---|---:|---:|---:|
| 53-ft dry van, road | 19,900 | 90 | 221 |
| 53-ft domestic container, rail | 21,700 | 90 | 241 |
| 40-ft ISO container, ocean | 26,500 | 67 | 396 |
| 20-ft ISO container, ocean | 28,200 | 33 | 855 |
| Air, chargeable convention | 1,000 per 6 m³ | 6 | 167 |
| Parcel, dimensional divisor 139 in³/lb | — | — | 199 |
| LCL, weight-or-measure | 1,000 per m³ | 1 | 1,000 |

The figures are rounded equipment and convention values, not a specification; verify against the
actual unit and the carrier's tariff. The last three rows are pure convention: air freight is
charged as if cargo density were 167 kg/m³ whether or not it is, parcel as if it were 199 kg/m³,
and LCL as if one cubic metre weighed one tonne. Those conventions are the mechanism by which
light freight pays more per kilogram than it would if it were dense.

This is the hinge of the whole topic. A shipment of 1,900 kg in 6.3 m³ has a density of 302 kg/m³.
It exceeds the design density of a road trailer, so it is weight-limited in road modes and
perfectly matched to a 40-ft container. A shipment of 300 kg in 6.3 m³ has a density of 48 kg/m³.
It is volume-limited in every mode, and every convention above will charge it as though it were
several times heavier than it is.

### 3.5 The five-bucket total cost

Mode choice is a total-cost decision. Five buckets, and a comparison that omits any of them is a
partial answer that will be overturned by the bucket it omitted.

| Bucket | What is in it | Where it is usually wrong |
|---|---|---|
| Linehaul | The base charge for moving the shipment | Compared per kilogram across modes that charge per shipment |
| Accessorials | Fuel, liftgate, appointment, detention, reweigh, documentation | Omitted from the comparison, then discovered on the invoice |
| Inventory in transit | Pipeline stock plus the safety stock the mode's tail requires | Counted twice, or not at all |
| Service failure | Expediting, penalties, lost margin, churn from late delivery | Assumed equal across modes |
| Emissions | Greenhouse gas per unit of activity, under a stated allocation | Reported without a boundary or an allocation rule |

The arithmetic of the five buckets is in section 5. The discipline is in the ordering: build all
five before ranking anything, because the ranking frequently changes when the third and fourth
buckets are added.

### 3.6 Four kinds of number

Every service or cost number in this file is one of four kinds, and the kind determines what you
may do with it. The distinction is stated because a plan and a commitment are routinely confused
in mode decisions, and the confusion is expensive.

| Kind | Definition | Example in this file |
|---|---|---|
| Forecast | A model output with a distribution attached | Transit time: mean 4.2 days, standard deviation 1.6 days |
| Plan | A dated number the organisation will manage to | Freight budget of 2,475 USD per truckload, Q4 2026 basis |
| Target | A number the organisation wants to reach | 95 per cent on-time delivery |
| Commitment | A number promised to a customer, with a consequence if missed | A four-day delivery promise with a service credit |

A forecast is not a commitment, and a plan is not evidence that the cost will occur. When a mode
is chosen, the plan becomes a commitment to the carrier and the forecast becomes a commitment to
the customer. The gap between them is the risk the safety stock covers.

### 3.7 What a mode decision actually commits

Choosing a mode commits four things at once: a cost curve, a transit-time distribution, a handling
count (which drives damage and claims), and an emissions intensity. It does not commit a rate —
rates move, and the market evidence in section 9 shows how far and how fast. It commits the shape
of the curve, and shapes are much harder to change than prices.

That is the argument for spending the analytical effort here rather than in rate negotiation. A
two per cent rate improvement on a curve with the wrong shape is a rounding error against a
fifteen per cent structural difference in total cost.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Full truckload (FTL) | One shipper's freight occupying a dedicated vehicle, priced per trip | Used for any shipment large enough to seem big | Equipment class code, e.g. 53-ft dry van |
| Less-than-truckload (LTL) | Freight shared in a trailer with other shippers' freight, rehandled at terminals | Treated as a small FTL with a lower price | NMFC class plus weight break |
| Parcel | Individual packages moving through a hub-and-spoke sort network, priced per piece and zone | Applied to palletised freight | Service code plus zone |
| Intermodal | A container or trailer moving by rail for the long leg and by truck at each end | Used to mean any combined transport | Ramp pair plus equipment type |
| Drayage | The short truck move between a terminal, port, or ramp and a door | Confused with the rail or ocean linehaul | Per container move, per day |
| Twenty-foot equivalent unit (TEU) | Container capacity measure based on a 20-ft box | Used to compare vessels of different beam | TEU |
| Forty-foot equivalent unit (FEU) | Container capacity measure based on a 40-ft box | Used interchangeably with TEU | FEU |
| Full container load (FCL) | One shipper's cargo filling a container, priced per container | Assumed to mean the container is physically full | Container size plus weight limit |
| Less than container load (LCL) | Cargo consolidated with other shippers' cargo in one container | Treated as ocean LTL with no minimum | Weight-or-measure tonnes |
| Weight or measure (W/M) | LCL charging rule taking the greater of weight in tonnes and volume in cubic metres | Assumed to favour light cargo | Revenue tonne |
| Chargeable weight | The greater of actual and volumetric weight under the carrier's convention | Treated as the physical weight | Kilograms |
| Volumetric weight | Cube converted to a fictitious weight by a divisor or convention | Computed with the wrong divisor | Kilograms |
| Dimensional divisor | The cubic inches per pound, or cubic centimetres per kilogram, used to convert parcel cube to weight | Applied across carriers with different divisors | 139 or 166 in³/lb |
| Density | Mass divided by volume | Reported without units | kg/m³ |
| Design density | Payload divided by volume for a piece of equipment | Confused with the cargo density | kg/m³ |
| Pallet position | One pallet footprint on a trailer floor or in a rack | Used loosely for a pallet | Positions per trailer |
| Load factor | Actual load carried divided by capacity, by weight or by cube | Reported without saying which | Per cent, weight or cube |
| Empty running | Vehicle kilometres travelled without revenue cargo | Left outside the emissions boundary | Vehicle-kilometres |
| Linehaul | The long-distance movement charge, excluding terminal and accessorial charges | Used to mean the total freight cost | Currency per shipment |
| Accessorial | A charge for a service beyond the base move | Called a surcharge when it is a service | Charge code |
| Minimum charge | The floor charge for a shipment regardless of size | Forgotten in small-shipment comparisons | Currency per shipment |
| Freight class | The LTL classification that sets the rate basis, from density, handling, and liability | Assumed to be a quality grade | NMFC class number |
| Cubic capacity | Usable volume of the equipment | Confused with floor area | m³ |
| Payload | Maximum cargo mass the vehicle may carry | Confused with gross vehicle weight | kg |
| Tare | The empty weight of the equipment | Subtracted from cargo weight by mistake | kg |
| Deadhead | Movement of an empty vehicle to reach a load | Included in the rate without disclosure | Kilometres |
| Backhaul | A revenue trip on the return leg | Assumed to exist on every lane | Loads per week |
| Transit time | Elapsed time from collection to delivery, a random variable | Reported as a single promised number | Days, with a distribution |
| On-time performance | Share of shipments delivered within the promised window | Measured against the mean rather than the promise | Per cent |
| Tender | The offer of a specific load to a carrier under a contract or in the spot market | Used to mean a quote request | Load identifier |
| Common carriage | Capacity offered to any shipper under published terms | Assumed to be lower service | Tariff reference |
| Dedicated carriage | Capacity committed to one shipper for a period | Priced as if it were common carriage | Contract term in months |
| Spot rate | A one-off market price for a single move | Used as a budget basis | Currency per shipment, dated |
| Contract rate | A price agreed for a lane and period, usually with volume commitments | Assumed to be available at any volume | Currency per shipment, effective dates |
| Break-even weight | The shipment mass at which two modes cost the same | Quoted without the lane or the rate date | kg |
| Tonne-kilometre | One tonne moved one kilometre | Computed with kilograms and kilometres | t·km |
| Revenue tonne-kilometre | A sold unit of transport work, used in modal statistics | Confused with physical tonne-kilometres | Revenue t·km |
| Carbon dioxide equivalent (CO2e) | A common unit aggregating greenhouse gases by global warming potential | Reported without a boundary | kg CO2e |
| Well-to-wheel | Boundary covering fuel production as well as combustion | Used interchangeably with tank-to-wheel | Boundary label |
| Tank-to-wheel | Boundary covering combustion only | Compared against well-to-wheel figures | Boundary label |
| Allocation rule | The method assigning a shared vehicle's emissions to one shipment | Omitted from almost every claimed saving | Mass, volume, or slot basis |
| Container freight station (CFS) | The facility where LCL cargo is consolidated and deconsolidated | Treated as a terminal without cost | Facility code |
| Incoterm | The trade term allocating cost, risk, and customs responsibility between buyer and seller | Used as a payment term | Three-letter code, Incoterms 2020 |

## 5. The mechanics, step by step

The method has seven phases and twenty-two steps. Work them in order for the first few lanes you
analyse; after that the order becomes habit.

### Phase A — Characterise the shipment

**Step 1. Record mass, volume, and the unit of sale.** Mass in kilograms, volume in cubic metres,
and the unit in which the business sells — cases, eaches, pallets. Record all three even when only
one is used downstream, because the conversion factors between them are where errors enter.

**Step 2. Compute density and compare it with design density.** Density is mass divided by volume,
in kilograms per cubic metre. Compare it with the design density of each candidate mode from the
table in section 3.4. Cargo below the design density will be charged on cube; cargo above it will
be charged on mass.

**Step 3. Decide which currency binds.** For each candidate mode, state whether the binding
constraint is mass, cube, or time. Write it down. A comparison in which the two options are
constrained by different currencies is not a like-for-like comparison, and the arithmetic will
hide the fact.

**Step 4. Fix the service requirement as a distribution.** A promise of four days is a point on a
distribution. Record the mean, the standard deviation, and the ninety-fifth percentile the mode
actually delivers. The promise is a commitment; the distribution is a forecast; the gap is the
inventory.

### Phase B — Map each mode onto the five segments

**Step 5. List access, linehaul, handling, delivery, and administration for each candidate.**
Intermodal has two access legs, two terminal lifts, and a rail linehaul. LTL has one terminal at
each end and possibly a break-bulk in between. Air has a truck leg at each end, screening, and a
handling charge. Writing the list makes the handling count visible, and the handling count drives
damage, claims, and the tail of the transit distribution.

**Step 6. Identify the dominant cost component.** Use the table in section 3.2. If you cannot name
the dominant component, you do not yet understand the quote.

**Step 7. Predict the shape of the cost curve before looking at a rate.** For FTL, predict a flat
line against weight and a near-linear line against distance. For LTL, predict a declining
per-kilogram line with steps. If the quote you receive does not have the predicted shape, the quote
is not what you think it is.

### Phase C — Price the candidates

**Step 8. Collect rates on a stated date and record the basis.** Every rate carries a date, a
currency, a basis (per kilogram, per cubic metre, per shipment, per container, per stop), and a
validity period. A rate without those four attributes is an anecdote. The market evidence in
section 9 exists to make the point that a rate is a dated observation.

**Step 9. Convert every quote to a common basis.** Convert to chargeable weight and chargeable
volume under each carrier's own convention, then to cost per shipment for the actual shipment.
Never compare a per-kilogram quote with a per-trip quote without this step.

**Step 10. Add an accessorial allowance, and name it.** Fuel, liftgate, appointment, detention, and
documentation are real costs that vary by mode and by shipper behaviour. The mechanics belong to
`02_rate-structures.md`. Here, add a stated allowance per mode and label it as an allowance, not a
quote.

**Step 11. Price the actual shipment, not the rate-sheet shipment.** Rate sheets are built around
representative shipments. A four-pallet shipment priced at a full-trailer rate is a category error,
and so is a full trailer priced at a per-kilogram rate.

### Phase D — Unit economics

**Step 12. Compute the four unit views.** Cost per kilogram, per cubic metre, per tonne-kilometre,
and per pallet position. They answer different questions and at least two of them will disagree.

**Step 13. Compare at the actual load factor and at design load factor.** The first tells you what
you will pay. The second tells you what the mode costs when it is used well. The gap between them
is the cost of your own volume pattern, and it is the number to attack before negotiating.

**Step 14. Find the load factor at which the ranking changes.** This is the break-even, and it is
the most useful single number in a mode decision, because it survives a change in rates. Section
5.6 derives it.

### Phase E — Inventory and service

**Step 15. Compute pipeline inventory.** Units in transit equal the daily demand rate multiplied by
the transit time in days. Its carrying cost is that quantity multiplied by unit value and by the
annual carrying rate, divided by the number of days in the year when you need a daily figure.

**Step 16. Compute the safety stock the mode's tail requires.** Use the protection-period formula
from `../10_inventory_optimization_multi_echelon/04_safety-stock.md`:

```text
   SS = z * sqrt( (R + L) * sigma_d^2  +  mu_d^2 * sigma_L^2 )
```

Here `R` is the review period, `L` the mean transit time, `sigma_d` the standard deviation of daily
demand, `mu_d` the mean daily demand, and `sigma_L` the standard deviation of transit time. The
second term is the one that mode choice moves: a slower, less reliable mode raises `sigma_L`, and
because that term carries `mu_d` squared, the penalty scales with volume. That derivation is not
repeated here. What matters for mode choice is that `sigma_L` is a property of the mode, not of the
carrier relationship, and it belongs in the comparison.

**Step 17. Compute the expected cost of service failure.** Multiply the probability of delivery
outside the promise by the cost of one failure. The probability comes from the transit
distribution; the cost comes from expediting, penalties, and lost margin. State both. A mode
comparison that assumes equal service failure costs across modes has assumed away the main reason
the faster mode exists.

### Phase F — Emissions

**Step 18. Choose the activity denominator and state it.** Tonne-kilometres for weight-driven
cargo, cubic-metre-kilometres for volume-driven cargo. The choice changes the ranking, so it is a
disclosure item, not a detail.

**Step 19. Apply the factor set with a stated boundary and allocation.** Activity multiplied by an
emission factor gives emissions in kilograms of CO2e. The factor depends on the boundary
(tank-to-wheel or well-to-wheel), the load factor, the empty-running treatment, and the allocation
rule. ISO 14083:2023 defines the calculation; the GLEC Framework supplies the structure and the
default factors. A number without those four disclosures is not comparable with any other number.

### Phase G — Decide and record

**Step 20. Sum the five buckets and show the margin.** A ranking with a two per cent margin is not
a decision; it is a tie to be broken by the qualitative factors. A ranking with a twenty per cent
margin survives most of the assumptions you are about to test.

**Step 21. Test the three inputs that move the answer most.** In practice these are unit value,
transit-time standard deviation, and the accessorial allowance. Sweep each across a plausible range
and report the range of the answer, not a point.

**Step 22. Write the decision record.** Mode, lane, basis, rate date, currency, the five buckets,
the assumptions, and the conditions that would reverse the decision. The last item is what makes
the record useful in eighteen months, when the rates have moved and nobody remembers why the
choice was made.

### 5.6 The LTL-to-FTL break-even

Let `P` be the FTL price for the lane in a stated currency and period, `f` the LTL per-shipment
fixed charge, `r` the LTL rate per kilogram, `m` the LTL minimum charge, and `w` the shipment
weight. Then:

```text
   LTL charge   c_L(w) = max( m ,  f + r * w )
   FTL charge   c_F    = P
   break-even   w*     = ( P - f ) / r        when  f + r*w* >= m
```

The derivation is one line: set the two charges equal and solve for `w`. The condition `f + r*w* >=
m` matters because below the weight at which the minimum charge stops binding, the LTL charge is
constant and the effective per-kilogram rate falls with weight. That region is a cliff, not a
curve, and comparisons made inside it are usually wrong.

Five conditions must hold for the break-even to mean anything:

1. The shipment fits the FTL vehicle on both weight and cube, so `w* <= payload` and `w*/density <=
   volume`.
2. The LTL carrier will actually accept a shipment of that size. Many cap LTL at around twelve
   pallets or a stated weight, above which the shipment must be tendered as a partial or a
   truckload.
3. Both charges are on the same basis: same accessorial allowance, same fuel treatment, same
   currency, same date.
4. The comparison is on freight alone. The inventory and service differences are added afterwards,
   in Phase E, and they move the effective break-even.
5. Distance is held constant. The break-even weight rises with distance only if the LTL rate per
   kilogram rises faster than the FTL trip price, which is usually but not always true.

The arithmetic for a stated lane is in section 6.2. The result is typically far below the folklore
figure. The commonly quoted "ten thousand pounds" comes from class-rate LTL pricing in specific
markets and does not survive contact with competitive truckload pricing on a long lane.

### 5.7 Transit time as a distribution

| Mode, 1,450 km lane | Mean (days) | Standard deviation (days) | 95th percentile (days) | Share later than a 4-day promise |
|---|---:|---:|---:|---:|
| FTL | 2.0 | 0.35 | 2.6 | under 1 per cent |
| LTL | 4.2 | 1.6 | 6.8 | about 55 per cent |
| Intermodal | 7.5 | 2.2 | 11.1 | about 94 per cent |
| Air | 1.2 | 0.30 | 1.7 | under 1 per cent |

The figures are illustrative planning values for a single lane, not benchmarks. Their purpose is
to show the shape of the argument: the mean separates the modes by a factor of six, and the tail
separates them by more. The share later than the promise is computed from a normal approximation,
`P(L > 4) = 1 - Phi((4 - mean)/sd)`, which is a convenience, not a claim that transit times are
normal. Transit times are right-skewed in reality, so the normal approximation understates the
tail, and the tail is the part that costs money.

The link to inventory is direct. Pipeline stock is proportional to the mean. Safety stock is
proportional to the standard deviation, weighted by the square of the demand rate. A mode change
that moves the mean by two days and the standard deviation by 1.25 days therefore has two
inventory effects of different sizes, and on the lane in section 6 the second is the smaller of the
two only because the unit value is low.

### 5.8 Dedicated versus common carriage

The same physical move can be bought two ways, and the difference is a difference in who carries
the variability.

| Dimension | Common carriage | Dedicated carriage |
|---|---|---|
| Capacity | Shared, allocated per shipment | Committed for the contract term |
| Price basis | Per shipment, per kilogram, per class | Per vehicle, per period, or per kilometre |
| Who bears demand variability | The carrier, through acceptance decisions | The shipper, through the commitment |
| Who bears service variability | The shipper, through the tail of the distribution | Shared, defined by service levels in the contract |
| Flexibility to change volume | High, at a price | Low, until the contract expires |
| Leverage in negotiation | Volume across many lanes | Volume on the contracted lanes, and duration |
| Failure mode | Tender rejection at peak, spot exposure | Paying for capacity that is not used |

The economic content is in the third row. Common carriage is an option: you buy capacity when you
need it and pay a premium for the option, which appears as a higher rate per shipment and as
tender rejection when the market is tight. Dedicated carriage is a future: you commit to a volume
and pay whether or not you use it, and you receive a lower rate and a defined service level.

The negotiation consequence follows. On common carriage you negotiate a rate schedule and your
leverage is the volume you can steer. On dedicated carriage you negotiate a commitment and your
leverage is the duration and the certainty of the volume. A shipper that negotiates a dedicated
rate while retaining the right to walk away has negotiated neither.

## 6. Worked example

### 6.1 The lane and the shipment

A consumer-goods shipper moves one stock-keeping unit from a plant near Chicago, Illinois, to a
distribution centre near Dallas, Texas: 1,450 km by road, 1,290 km by rail.

| Attribute | Value | Note |
|---|---|---|
| Shipment | 4 pallets, weekly | Plan, not a commitment to the carrier |
| Mass | 1,900 kg | Actual |
| Volume | 6.3 m³ | Actual |
| Density | 302 kg/m³ | Derived: 1,900 / 6.3 |
| Units per shipment | 3,800 eaches | 0.5 kg per unit |
| Daily demand | 542.9 units per day | Derived: 3,800 / 7 |
| Unit value | 12.00 USD, Q4 2026 plan basis | Standard cost |
| Carrying rate | 22 per cent per year | Policy rate, applied to average inventory |

All money in this section is in US dollars on a Q4 2026 planning basis. The rates are illustrative
plan assumptions chosen to be internally consistent, not quotations. A real analysis replaces them
with dated quotes and states the validity period.

### 6.2 The freight comparison

| Option | Basis | Charge (USD, Q4 2026) | Arithmetic |
|---|---|---:|---|
| FTL, 53-ft dry van | Per trip | 2,475.00 | Lane price, about 2.75 USD per mile |
| LTL | 95.00 USD per shipment plus 1.10 USD/kg | 2,185.00 | 95.00 + 1.10 × 1,900 |
| Air | 4.50 USD per chargeable kg | 8,550.00 | 4.50 × 1,900; volumetric weight is 6.3 × 167 = 1,052 kg, so actual weight binds |

Two things are worth noting before any ranking. First, the FTL option is priced per vehicle and
the shipment uses about ten per cent of its payload: 1,900 kg against 19,900 kg. Second, the air
option is charged on actual weight because the cargo is denser than the air convention. For a
light shipment the same air rate would be charged on a weight more than twice the physical one.

The LTL-to-FTL break-even follows the formula in section 5.6:

```text
   w* = (P - f) / r = (2,475.00 - 95.00) / 1.10 = 2,163.6 kg
```

The break-even is 2,163.6 kg, which is 4,770 lb. The actual shipment is 1,900 kg, so LTL is cheaper
on freight by 290.00 USD per shipment, or 15,080.00 USD per year across 52 shipments. That is a
11.7 per cent freight saving against the FTL plan.

Check condition 1: 2,163.6 kg is below the 19,900 kg payload, and at 302 kg/m³ it occupies 7.2 m³,
well below 90 m³. Check condition 2: 2,163.6 kg is about five pallets at this density, comfortably
inside normal LTL acceptance. Conditions 3 to 5 are handled in section 6.5.

The break-even is sensitive to both prices, and the sensitivity is worth tabulating because it
shows how little a rate movement is needed to flip the decision.

| Break-even weight (kg) | FTL at 2,000 USD | FTL at 2,475 USD | FTL at 2,950 USD |
|---|---:|---:|---:|
| LTL at 0.85 USD/kg | 2,241 | 2,800 | 3,359 |
| LTL at 1.10 USD/kg | 1,732 | 2,164 | 2,596 |
| LTL at 1.35 USD/kg | 1,411 | 1,763 | 2,115 |

### 6.3 The two unit views, and the inversion

| Shipment | Mode | Charge (USD, Q4 2026) | Per kg (USD) | Per m³ (USD) | Per t·km (USD) |
|---|---|---:|---:|---:|---:|
| 1,900 kg / 6.3 m³ | FTL | 2,475.00 | 1.303 | 392.86 | 0.898 |
| 1,900 kg / 6.3 m³ | LTL | 2,185.00 | 1.150 | 346.83 | 0.793 |
| 1,900 kg / 6.3 m³ | Air | 8,550.00 | 4.500 | 1,357.14 | 3.104 |
| 300 kg / 6.3 m³ | LTL | 535.50 | 1.785 | 85.00 | 1.231 |
| 300 kg / 6.3 m³ | Air | 4,734.00 | 15.780 | 751.43 | 10.883 |

Arithmetic check: 2,475.00 / 6.3 = 392.86 USD/m³; 2,475.00 / (1.9 × 1,450) = 2,475.00 / 2,755 =
0.898 USD per tonne-kilometre. For the 300 kg case, cube binds: 85.00 × 6.3 = 535.50 USD, while
weight would give 95.00 + 1.10 × 300 = 425.00 USD. Air chargeable weight is 6.3 × 167 = 1,052 kg,
so 1,052 × 4.50 = 4,734.00 USD, and the charge per physical kilogram is 15.78 USD.

Read the last three rows as the inversion. The 300 kg shipment is cheaper per cubic metre than the
1,900 kg shipment — 85.00 against 346.83 USD/m³ — because cube is the binding currency and the
shipment uses the same cube for less weight. It is 55 per cent more expensive per tonne-kilometre,
1.231 against 0.793 USD, because the tonne-kilometre denominator has shrunk. Neither number is
wrong. They are answers to different questions, and the choice of denominator decides which mode
looks efficient. A benchmark that compares light freight per tonne-kilometre against a dense
benchmark will conclude that light freight is three times worse than it is, and will then choose a
mode that cannot fix the problem.

At full payload the shape reverses completely:

| Basis | FTL (USD per t·km) | LTL (USD per t·km) | Ratio |
|---|---:|---:|---:|
| 1,900 kg shipment | 0.898 | 0.793 | 0.88 |
| 19,900 kg full payload | 0.0858 | 0.7617 | 8.9 |

Arithmetic check: 2,475.00 / (19.9 × 1,450) = 2,475.00 / 28,855 = 0.0858 USD per tonne-kilometre.
The LTL charge at full payload is 95.00 + 1.10 × 19,900 = 21,985.00 USD, and 21,985.00 / 28,855 =
0.7617 USD per tonne-kilometre. The LTL carrier would not accept that shipment, which is the point:
the LTL rate is a rate for small shipments, and extrapolating it to full-trailer volumes produces a
number that is both arithmetically correct and commercially meaningless.

### 6.4 The inventory consequence

The transit distribution from section 5.7 is used. `R` is seven days, `z` is 1.645 for a 95 per cent
cycle service level, `mu_d` is 542.9 units per day, and `sigma_d` is 120 units per day.

| Item | FTL | LTL | Difference |
|---|---:|---:|---:|
| Mean transit (days) | 2.0 | 4.2 | +2.2 |
| Standard deviation of transit (days) | 0.35 | 1.60 | +1.25 |
| Units in pipeline | 1,085.7 | 2,280.0 | +1,194.3 |
| Pipeline carrying cost (USD per year, Q4 2026) | 2,866 | 6,019 | +3,153 |
| Protection period (days) | 9.0 | 11.2 | +2.2 |
| Safety stock at 95 per cent (units) | 670 | 1,574 | +904 |
| Safety stock carrying cost (USD per year) | 1,768 | 4,155 | +2,388 |
| Total inventory cost (USD per year) | 4,634 | 10,174 | +5,541 |

Arithmetic check for the LTL safety stock: the demand term is 11.2 × 120² = 161,280, the lead-time
term is 542.9² × 1.6² = 294,697 × 2.56 = 754,424, the sum is 915,704 and its square root is 956.9
units; multiplied by 1.645 this gives 1,574 units. For FTL the same calculation with 9.0 days and
a standard deviation of 0.35 gives 407.1 units of standard deviation and 670 units of buffer.

The inventory difference of 5,541 USD per year consumes 37 per cent of the 15,080 USD freight
saving. The net advantage of LTL is 9,539 USD per year before service failure and emissions.

### 6.5 The value threshold

Inventory cost scales with unit value; freight does not. At 12.00 USD per unit the inventory
penalty is 5,541 USD per year, which is 461.8 USD per year per dollar of unit value. The freight
saving of 15,080 USD per year is exhausted at a unit value of 15,080 / 461.8 = 32.66 USD.

The statement that follows is the useful output of the whole example, and it is a plan, not a
forecast: **on this lane, at this shipment size and these rates, LTL remains cheaper on total cost
only while the unit value stays below about 33 USD (Q4 2026 basis).** Above that value the FTL plan
wins, and it wins by an amount that grows linearly with value. A product-mix change that raises
average unit value past that threshold reverses the decision without any change in freight rates.

Service failure enters as a third term. If one failure costs 150.00 USD in expediting and service
credits, the net LTL advantage of 9,539 USD per year survives unless LTL causes more than 63.6
additional failures per year, which is 1.2 per week. That is a testable proposition: compare the
late-delivery counts of the two options on the same lane over a quarter.

### 6.6 Emissions

Activity is 1.9 t × 1,450 km = 2,755 t·km. The intensity values below are course planning ranges
taken from the module survey file, not values from the current GLEC factor file. They are adequate
to show the method and inadequate for a disclosure.

| Mode | Intensity (g CO2e per t·km) | Emissions per shipment (kg CO2e) | Per unit (g CO2e) |
|---|---:|---:|---:|
| FTL | 90 | 248 | 65 |
| LTL | 135 | 372 | 98 |
| Air | 600 | 1,653 | 435 |

The annual difference between the LTL and FTL plans is 52 × 124 = 6,448 kg CO2e, or 6.4 t CO2e.
That figure is only comparable with another figure computed under the same four disclosures:
boundary, load factor, empty running, and allocation rule. Under ISO 14083:2023 the empty and
positioning legs belong inside the boundary, and excluding them understates road emissions. Under
a mass allocation the LTL shipment carries the share of the trailer proportional to its mass;
under a volume allocation it carries the share proportional to its cube, which for this cargo is
close to the same answer because the density is near the trailer's design density. For light
cargo the two allocations diverge sharply, and the choice must be stated.

### 6.7 The full-container case

The same lane at full-container volume shows why intermodal is a volume business. At 19,000 kg and
90 m³:

| Option | Charge (USD, Q4 2026) | Per kg (USD) | Per t·km (USD) | Mean transit (days) |
|---|---:|---:|---:|---:|
| FTL | 2,475.00 | 0.130 | 0.0898 | 2.0 |
| Intermodal | 2,050.00 | 0.108 | 0.0744 | 7.5 |

The rate saving is 425.00 USD per shipment, or 22,100.00 USD per year. The inventory cost of 5.5
additional days at 12.00 USD per unit is 29,857 units in pipeline, carried at 22 per cent, or
78,823 USD per year, before the safety-stock effect, which adds roughly 37,900 USD per year at the
same value. The intermodal option therefore loses by a wide margin at this unit value, and the
threshold value at which it breaks even is about 2.27 USD per unit.

This result is not an argument against intermodal. It is an argument that intermodal is a mode for
low-value, dense, distance-long freight, and that the transit penalty is a property of the terminal
and drayage structure rather than of the rail speed. On this 1,450 km lane the terminal time
dominates the linehaul time saving. The same calculation on a 2,500 km lane raises the rate saving
and leaves the transit penalty roughly unchanged, which is exactly the condition under which
intermodal wins.

## 7. Code, query, or calculation

The script below reproduces the arithmetic of sections 6.2 and 6.3. It is deliberately small: the
point is that the break-even and the unit views are four lines of arithmetic, and that a
spreadsheet which does not show them is hiding a decision rather than supporting one.

```python
# freight_modes.py — LTL versus FTL break-even and the two unit views.
# All money in USD on a Q4 2026 planning basis. Illustrative rates, not quotes.
LANE_KM = 1450.0
PAYLOAD_KG = 19900.0
VOLUME_M3 = 90.0
FTL_PRICE = 2475.0        # per 53-ft dry van trip
LTL_FIXED = 95.0         # per-shipment handling
LTL_PER_KG = 1.10
LTL_MIN = 245.0

def ltl_charge(kg):
    return max(LTL_MIN, LTL_FIXED + LTL_PER_KG * kg)

def breakeven_weight(p_ftl=FTL_PRICE, fixed=LTL_FIXED, rate=LTL_PER_KG):
    return (p_ftl - fixed) / rate

def unit_costs(charge, kg, m3, km=LANE_KM):
    tonnes = kg / 1000.0
    return {"per_kg": charge / kg,
            "per_m3": charge / m3,
            "per_tkm": charge / (tonnes * km)}

kg, m3 = 1900.0, 6.3
for name, charge in (("FTL", FTL_PRICE), ("LTL", ltl_charge(kg))):
    u = unit_costs(charge, kg, m3)
    print(f"{name} charge {charge:8.2f} | per kg {u['per_kg']:.3f} | "
          f"per m3 {u['per_m3']:7.2f} | per t-km {u['per_tkm']:.3f}")

print(f"break-even weight {breakeven_weight():.1f} kg")
print(f"break-even density {breakeven_weight() / m3:.1f} kg/m3 at {m3} m3")

for name, charge in (("FTL", FTL_PRICE), ("LTL", ltl_charge(PAYLOAD_KG))):
    print(f"{name} at full payload: {unit_costs(charge, PAYLOAD_KG, VOLUME_M3)['per_tkm']:.4f} USD/t-km")
```

**Expected output.** Values are rounded to the printed precision; the LTL line at full payload is
the extrapolation discussed in section 6.3, not a purchasable rate.

```text
FTL charge  2475.00 | per kg 1.303 | per m3  392.86 | per t-km 0.898
LTL charge  2185.00 | per kg 1.150 | per m3  346.83 | per t-km 0.793
break-even weight 2163.6 kg
break-even density 343.4 kg/m3 at 6.3 m3
FTL at full payload: 0.0858 USD/t-km
LTL at full payload: 0.7617 USD/t-km
```

Three checks to run when you port this to a real dataset: the LTL charge must respect the minimum,
the break-even must be tested against the payload and the cube, and the unit views must be computed
on the same charge. The most common error in a production version is mixing a linehaul-only charge
with an all-in charge between the two modes, which biases the break-even by the size of the
accessorial allowance.

## 8. Data, standards, and regulatory context

### 8.1 The two official statistics that frame the topic

Two public datasets are used often enough to be described precisely, because they measure
different things and are routinely compared by mistake.

| Dataset | Edition and date | Unit of measurement | What it measures |
|---|---|---|---|
| US Commodity Flow Survey (CFS) | 2022 edition, released 26 June 2025 by the US Census Bureau with the Bureau of Transportation Statistics | Tons shipped and value, by mode and commodity | Domestic shipments originating in the United States, from a sample of about 165,000 establishments |
| Eurostat freight transport statistics, modal split | 2024 reference year, data extracted February 2026 | Tonne-kilometres, by mode | Freight performance on the territory of the reporting countries, including international movements under a territoriality principle |

The 2022 CFS reported 12.2 billion tons shipped, valued at 18.0 trillion USD (2022, nominal), with
trucks carrying 68.1 per cent of tonnage and 73.5 per cent of value. Dividing value by tonnage
gives an average of 18.0 × 10^12 / 12.2 × 10^9 = 1,475 USD per ton, which is a derived figure, not
a published one, and is useful only as an order of magnitude for value density. The same release
reported that 56.4 per cent of tonnage moved less than 50 miles, which is the single most important
fact in the dataset for last-mile analysis and the least quoted.

The Eurostat 2024 modal split reported maritime at 67.0 per cent of tonne-kilometres, road at 25.7
per cent, rail at 5.4 per cent, inland waterways at 1.7 per cent, and air at 0.2 per cent.

The comparison trap is this: the CFS counts tons, the Eurostat series counts tonne-kilometres, and
the CFS excludes imports, farms, government entities, and several industries. A slide that places a
CFS tonnage share next to a Eurostat tonne-kilometre share has compared two different quantities
and will produce a conclusion that neither dataset supports.

### 8.2 Emissions accounting

ISO 14083:2023, *Greenhouse gases — Quantification and reporting of greenhouse gas emissions
arising from transport chain operations*, was published on 20 March 2023. It establishes a common
methodology for quantifying and reporting emissions from the operation of passenger and freight
transport chains, and it supersedes IWA 16:2015. It is the standard against which a transport
emissions claim is now assessed.

The GLEC Framework, issued by the Smart Freight Centre, is the practical structure and default
factor set that sits on top of the standard. Version 3.3 was released on 27 October 2026; version
3.1, published in 2023, was the release aligned to ISO 14083. A calculation that cites "GLEC" must
say which version, because the default factors and the category structure change between them.

Three disclosures make an emissions figure comparable. The boundary: tank-to-wheel covers
combustion, well-to-wheel covers fuel production as well, and the two differ materially for
biofuels and electricity. The allocation rule: mass, volume, or slot basis, stated explicitly.
The load factor and empty-running treatment: a half-empty trailer has roughly double the emissions
per tonne-kilometre, and excluding positioning and empty legs understates road emissions.

### 8.3 Maritime carbon intensity regulation

The International Maritime Organization (IMO) amendments to MARPOL Annex VI entered into force on
1 November 2022, and the Energy Efficiency Existing Ship Index (EEXI) and Carbon Intensity
Indicator (CII) certification requirements took effect on 1 January 2023. EEXI applies to ships of
400 gross tonnage and above as a one-time design check. CII applies to ships of 5,000 gross tonnage
and above as an annual operational measure, rating each ship A to E. A ship rated D for three
consecutive years, or E once, must submit a corrective action plan. The commercial consequence is
not a fine but a chartering preference, and it is one reason a shipper may see a green premium on
a specific vessel or service rather than on a trade lane.

### 8.4 Commercial and operational standards

| Standard or code | Owner | What it fixes in this topic |
|---|---|---|
| National Motor Freight Classification (NMFC) | National Motor Freight Traffic Association | The LTL class that sets the rate basis |
| Incoterms 2020 | International Chamber of Commerce | Which party buys the move, and therefore whose mode decision it is |
| Hours of Service, 49 CFR Part 395 | US Federal Motor Carrier Safety Administration | Driver time, which is a large share of FTL cost and a constraint on transit time |
| ANSI X12 sets 204, 990, 214, 210, 997 | Accredited Standards Committee X12 | Tender, response, status, invoice, and acknowledgement messages |
| UN/EDIFACT IFTMIN and IFTSTA | UN/CEFACT | The international equivalents of the tender and status messages |
| GS1 SSCC | GS1 | Pallet and parcel identity, which is what makes a handling count auditable |

The data-quality checks that precede any mode analysis are the same ones that precede any
transport analysis: every shipment carries an explicit mode and service level, distances are
road-network distances with a stated provider and version, weight and cube have explicit units
converted once at ingestion, timestamps carry a time zone, and cancelled shipments are excluded
from cost per shipment but included in capacity demand.

## 9. Industry practice and case evidence

### 9.1 What the structure of the statistics shows

The modal split is not an accident of geography. Road carries the majority of tonnage in every
developed economy because it is the only mode that goes door to door, and because average haul
lengths are short: more than half of US tonnage moves less than 50 miles. Rail and water carry a
minority of tonnage and a majority of tonne-kilometres in the European Union because they carry
bulk and containers over long distances at low marginal cost per kilometre. This is a statement
about the cost architecture in section 3.2, read off a public dataset, and it is graded B because
the datasets are official statistics with published methodology.

### 9.2 What the rate market shows, and how fast it moves

Container rates are the clearest available demonstration that a rate is a dated observation.
Freightos reported the Asia–US West Coast lane at about 8,300 USD per FEU in the week of 3 November
2026, broadly flat week on week, against 8,126 USD per FEU in the module survey file's reading of
27 October 2026. The same weekly commentary reported the composite index near 3,500 USD per FEU.
These are volatile, dated observations from a commercial index, graded C, and they are included to
make a methodological point rather than to inform a budget: a model that stores a container rate
as a constant is modelling the week it was built, not the market.

The derived unit economics are worth recording because they show how the denominator changes the
story. At 8,126 USD per FEU and a 20,000 kg payload over 11,000 km, the cost is 0.406 USD per
kilogram and 0.037 USD per tonne-kilometre. That is roughly an order of magnitude below road
truckload per tonne-kilometre and roughly an order of magnitude below air. A shipper comparing
ocean and air per kilogram and a shipper comparing them per tonne-kilometre will reach the same
ranking but will disagree about the size of the gap, and the disagreement matters when the cargo is
light.

### 9.3 The inventory-theoretic evidence

The idea that mode choice is an inventory decision is not new, and the empirical record is
consistent. Baumol and Vinod (1970) formalised mode choice as a trade-off among freight rate,
speed, dependability, and loss and damage, with the value of speed arising from the inventory it
removes. Hummels and Schaur (2013) estimated the willingness to pay for time in US imports and
found that each additional day in transit is equivalent to an ad valorem tariff of roughly 0.6 to
2.1 per cent. That range is the empirical anchor for the inventory calculation in section 6.4: the
per-day cost of transit is a real, measurable, and often large fraction of cargo value, and it is
the number that decides close mode comparisons.

### 9.4 The emissions evidence

Craig, Blanco, and Sheffi (2013) analysed more than 400,000 intermodal shipments and reported an
average CO2 intensity of 67 grams per ton-mile, which converts to 41.6 grams per tonne-kilometre,
with a range from 29 to 220 grams per ton-mile, or 18.0 to 136.7 grams per tonne-kilometre,
depending on the lane. Two conclusions follow. First, the average supports the claim that
intermodal is materially less carbon-intensive than road for the linehaul. Second, the range is
wide enough that a lane-level claim cannot be made from the average, which is precisely the
allocation-rule problem stated in section 8.2. A mode-shift claim that quotes the average and
ignores the lane variance is a claim about a population, not about the shipment.

### 9.5 Dedicated and common carriage in practice

The observable pattern in freight procurement is a mix, not a choice. Shippers contract a primary
carrier set for the base volume, hold a spot channel for peaks, and use dedicated fleets where
service or handling requirements make shared capacity unsuitable — temperature control, high-value
electronics, hazardous materials, and routes with tight delivery windows. The mix changes with the
market: when capacity is tight, contract coverage falls and spot exposure rises, and the effective
rate rises even when the contracted rate has not moved. That is why tender acceptance is a better
cost indicator than the contract rate, a point developed in `03_lane-tender-analytics.md`.

### 9.6 What the case evidence does not show

None of the sources above establishes a universal break-even weight, a universal transit-time
penalty, or a universal modal split. Each is a measurement on a population, a lane set, or a
moment. The analytical work in sections 5 and 6 exists because the population averages are not
transferable to a specific lane without recomputation.

## 10. Failure modes and red flags

### 10.1 Comparing a per-kilogram quote with a per-trip quote

The two charges are on different bases and one of them includes the vehicle whether or not it is
full. The comparison produces a number that looks like a saving and is an artefact of the basis.
Red flag: the two options in the comparison are quoted in different units and nobody has converted
them to cost per shipment.

### 10.2 Ignoring the minimum charge

Below the weight at which the minimum stops binding, LTL and parcel charges are flat, so the
effective per-kilogram rate falls sharply with weight. A small-shipment comparison that uses the
per-kilogram rate overstates the cost of the small shipment and understates the case for
consolidation.
Red flag: the rate table has a minimum charge column and the model does not reference it.

### 10.3 Comparing on weight when cube binds

For cargo below the design density, the binding constraint is volume, and a weight-based
comparison will recommend the wrong mode. The error is largest for parcel and LCL, where the
conversion convention can charge a light shipment at several times its physical weight.
Red flag: the analysis has no density column.

### 10.4 Using the mean transit time

The mean is the sales story and the tail is the inventory cost. A comparison that uses the mean
implicitly assumes zero variance, which is the assumption that makes the slow mode look cheap.
Red flag: the mode comparison contains one number per mode for transit time.

### 10.5 Treating a spot rate as a budget

A spot rate is a market observation on a date. Using it as a planning basis produces a budget that
is wrong as soon as the market moves, and it hides the exposure that a contract would have removed.
Red flag: the rate in the model has no date and no validity period.

### 10.6 Comparing modes on freight alone

The four omitted buckets — accessorials, inventory in transit, service failure, and emissions —
are frequently larger than the difference being compared. On the lane in section 6 the inventory
bucket consumed 37 per cent of the freight saving on its own.
Red flag: the business case has one cost line and a percentage saving.

### 10.7 Reporting an emissions saving without an allocation rule

A percentage reduction computed under one allocation and compared with a baseline computed under
another is not a saving. The same shipment can show materially different intensities under mass
and volume allocation when the cargo density differs from the vehicle's design density.
Red flag: the carbon claim names a percentage and does not name a boundary, an allocation basis, or
a factor version.

### 10.8 Assuming LTL is available at any size

LTL carriers cap shipment size, and above the cap the shipment must be tendered as a partial or a
truckload. A break-even computed above that cap is a number for a service that will not be sold.
Red flag: the break-even weight exceeds the carrier's published LTL acceptance limit and the model
does not check it.

### 10.9 Forgetting the access legs in intermodal

The drayage at each end, the terminal lift, and the ramp dwell are the costs that most often erase
intermodal's linehaul advantage. A comparison that uses the rail rate alone compares a linehaul
with a door-to-door move.
Red flag: the intermodal cost line contains a rail rate and no drayage or lift charge.

### 10.10 Double-counting inventory

Pipeline stock and safety stock are different objects. Pipeline stock is the inventory in motion,
proportional to the mean transit time. Safety stock is the buffer against variability, proportional
to the standard deviation. Adding a pipeline charge and then applying a buffer to the same units
counts the same inventory twice.
Red flag: the inventory cost line is described as "inventory during transit" and is computed from
the safety-stock formula.

### 10.11 Extrapolating a small-shipment rate to a full load

The LTL rate in section 6.3 produces 0.7617 USD per tonne-kilometre at full payload, nine times the
FTL figure, for a shipment the LTL carrier would not accept. The number is arithmetically correct
and commercially meaningless, and it appears in benchmarks whenever the analyst forgets to check
the acceptance limit.
Red flag: a unit cost computed at a load factor the mode does not serve.

### 10.12 Treating a mode shift as a rate negotiation

A mode shift changes the cost curve, the transit distribution, the handling count, and the
emissions intensity at once. Negotiating a lower rate inside the new mode does not recover the
service or inventory difference, and negotiating harder inside the old mode does not capture the
structural difference.
Red flag: the mode decision is recorded as a procurement saving with no service or inventory
annex.

## 11. Assumptions and limits

### 11.1 Rates are dated and lane-specific

Every cost in sections 6 and 9 is a dated observation on a stated lane in a stated currency. Rates
move with capacity, fuel, and season, and a lane in another market can differ by a factor of two.
**What breaks if wrong:** the break-even weight moves with the rate ratio, and a decision taken on
a stale rate can be reversed by a market move that has nothing to do with the shipment.

### 11.2 The shipment is homogeneous and its density is stable

The analysis assumes one density for the shipment and one unit value for the cargo. Mixed pallets
and product-mix changes move both.
**What breaks if wrong:** density determines which currency binds, so a mix change can move the
shipment from weight-limited to volume-limited and invert the mode ranking without any change in
rates. The value threshold in section 6.5 moves linearly with unit value and can be crossed by a
product-mix shift alone.

### 11.3 Transit times are approximately normal

The tail probabilities in section 5.7 use a normal approximation. Real transit times are
right-skewed, with a longer upper tail than the normal implies.
**What breaks if wrong:** the normal approximation understates the probability of a very late
delivery and therefore understates the safety stock and the service-failure cost of the slower
mode. The error is in the direction that flatters the slow option.

### 11.4 Demand is stationary over the protection period

The safety-stock calculation assumes a stable mean and standard deviation of daily demand over the
review-plus-lead-time window. Promotions, seasonality, and trend violate this.
**What breaks if wrong:** the buffer is sized for an average period that does not occur, and the
inventory comparison between modes is computed on a demand regime that neither mode will
experience.

### 11.5 The carrying rate is a policy number, not a measurement

Twenty-two per cent per year is a common policy rate covering capital, storage, obsolescence, and
insurance. It is not derived from this company's cost of capital or its shrink record.
**What breaks if wrong:** the inventory bucket scales linearly with the carrying rate, so a company
whose true rate is 12 per cent will see the inventory penalty at roughly half the size computed
here, and the value threshold in section 6.5 will roughly double.

### 11.6 Emission factors are planning ranges, not a factor set

The intensities in section 6.6 come from the course survey file and are order-of-magnitude values.
They are not extracted from the current GLEC factor file and carry no version or boundary.
**What breaks if wrong:** a disclosure built on these values will not survive assurance, and a
mode-shift claim built on them may reverse once load factor, empty running, and allocation are
handled properly.

### 11.7 Accessorials are an allowance, not a quote

The comparison adds a stated allowance rather than modelling the accessorial mechanics, which are
the subject of `02_rate-structures.md`.
**What breaks if wrong:** accessorials differ by mode in both size and predictability. A mode with a
low linehaul and a high accessorial exposure can be cheaper on the rate sheet and more expensive on
the invoice, and the allowance is too coarse to catch that.

### 11.8 The lane is one-way and the backhaul is not priced

The FTL price in section 6 is a one-way trip price. It embeds whatever backhaul the carrier expects
to find, and it does not give the shipper any claim on it.
**What breaks if wrong:** a shipper with balanced two-way volume has negotiating leverage the
one-way analysis does not see, and a shipper with a strongly directional lane is paying for the
carrier's empty return whether or not the invoice says so.

### 11.9 Service failure cost is constant per event

The model prices a failure at 150.00 USD regardless of customer, product, or season.
**What breaks if wrong:** failure costs are heavy-tailed. A stockout at a key account can exceed
the annual saving from the mode change, and averaging it across events hides the exposure that
matters.

### 11.10 The comparison is at the shipment level, not the network level

The analysis prices one lane and one shipment size. It does not account for the effect of the
decision on other lanes, on carrier relationships, or on the consolidation opportunities that a
network view would reveal.
**What breaks if wrong:** a mode decision that is correct lane by lane can be wrong network wide,
because volume moved off a lane weakens the shipper's position on adjacent lanes. The consolidation
question is taken up in `04_consolidation-cube.md`.

## 12. Exercises

### Exercise 1 — Design density and the binding currency (application, 40 minutes)

Take five stock-keeping units from a real or described catalogue. For each, record mass, volume,
and the equipment it ships in, compute density, and compare it with the design density of that
equipment.

**Deliverable:** a five-row table with mass, volume, density, design density, the binding currency,
and the chargeable quantity under the applicable convention.
**Acceptance criteria:** every unit stated; the convention used for air, parcel, and LCL named; one
sentence per item on what a packaging change would do to the chargeable quantity.
**Time estimate:** 40 minutes.

### Exercise 2 — Derive and compute the LTL-to-FTL break-even (calculation, 45 minutes)

Derive the break-even weight from the two charge functions. Compute it for a lane of your choice
using rates you can date, and test it against the payload, the cube, and the carrier's LTL
acceptance limit.

**Deliverable:** the derivation in no more than six lines, the computed break-even in kilograms,
and a three-by-three sensitivity table across FTL price and LTL rate.
**Acceptance criteria:** the minimum-charge condition stated and checked; all five conditions from
section 5.6 addressed; every rate dated and in a stated currency.
**Time estimate:** 45 minutes.

### Exercise 3 — The cube-bound break-even (calculation, 40 minutes)

For the same lane, compute the break-even volume at which LTL's cube charge equals the FTL trip
price. Compare it with the weight break-even and state which binds first for a cargo of 120 kg/m³
and for a cargo of 400 kg/m³.

**Deliverable:** two break-evens, the crossover density, and a one-paragraph interpretation.
**Acceptance criteria:** the crossover density derived, not assumed; the interpretation names which
break-even is relevant to which cargo; the LTL acceptance limit checked in both cases.
**Time estimate:** 40 minutes.

### Exercise 4 — From transit distribution to inventory (calculation, 50 minutes)

Using `R` of seven days, `mu_d` of 542.9 units per day, `sigma_d` of 120 units per day, and the
transit parameters in section 5.7, compute pipeline stock and safety stock at a 95 per cent cycle
service level for FTL, LTL, intermodal, and air.

**Deliverable:** a four-row table with pipeline units, safety stock, inventory value, and annual
carrying cost.
**Acceptance criteria:** the protection-period formula cited to Module 10 topic 04 rather than
re-derived; the two terms shown separately for at least one mode; the carrying rate and unit value
stated with their basis.
**Time estimate:** 50 minutes.

### Exercise 5 — The five-bucket total cost (analysis, 60 minutes)

Build the full comparison for the lane in section 6 at a unit value of 12.00 USD and again at 120.00
USD (Q4 2026 basis). Include linehaul, an accessorial allowance, inventory, service failure, and
emissions.

**Deliverable:** two five-bucket tables, the ranking under each, and the unit value at which the
ranking reverses.
**Acceptance criteria:** every bucket populated with a number and a unit; the reversal value
derived and shown; the service-failure assumption stated as a cost per event and a frequency.
**Time estimate:** 60 minutes.

### Exercise 6 — Critique an emissions claim (critique, 45 minutes)

Find a published mode-shift emissions claim from a carrier, a shipper, or a trade body. Assess it
against ISO 14083:2023 and the GLEC Framework version it cites, or fails to cite.

**Deliverable:** a one-page assessment with the claim quoted, the boundary, the allocation rule,
the load factor and empty-running treatment, the factor version, and a verdict.
**Acceptance criteria:** the four disclosures from section 8.2 each addressed explicitly; the
verdict states what could be concluded if the disclosures were supplied; no claim that the number
is false, only that it is unsupported.
**Time estimate:** 45 minutes.

### Exercise 7 — Dedicated versus common carriage (design, 50 minutes)

A shipper moves 18 loads per week on a lane with a seasonal peak of 26 and a trough of 9. Compare a
dedicated contract for 18 loads per week with a common-carriage arrangement at spot plus contract
prices.

**Deliverable:** an annual cost for each option, the break-even peak volume, and a recommendation
with a stated condition under which it reverses.
**Acceptance criteria:** the cost of unused committed capacity priced explicitly; the spot premium
at peak stated as an assumption; the service difference named rather than assumed away.
**Time estimate:** 50 minutes.

### Exercise 8 — Audit the official statistics (application, 40 minutes)

Using the 2022 Commodity Flow Survey and the Eurostat 2024 modal split, construct a table that
compares road, rail, and air shares on a consistent basis. Identify every point at which the two
datasets cannot be compared.

**Deliverable:** the table plus a list of at least four incompatibilities with the reason for each.
**Acceptance criteria:** the unit of measurement stated for each column; the coverage differences
named; no combined share computed across the two datasets.
**Time estimate:** 40 minutes.

### Exercise 9 — Write the mode-decision record (design, 45 minutes)

Write the decision record from step 22 for a mode choice you have made or can reconstruct.

**Deliverable:** a one-page record with lane, basis, rate date, currency, the five buckets, the
assumptions, and the reversal conditions.
**Acceptance criteria:** every number carries a unit and every money figure a currency and date;
the reversal conditions are specific and testable; the record distinguishes forecast, plan, target,
and commitment.
**Time estimate:** 45 minutes.

## 13. Self-check answers

### Exercise 1

The worked answer depends on the catalogue chosen, but the structure is fixed. For each item, mass
in kilograms divided by volume in cubic metres gives density in kilograms per cubic metre. Compare
it with the design density of the equipment: about 221 kg/m³ for a 53-ft road trailer, about 396
kg/m³ for a 40-ft ocean container, 167 kg/m³ for air under the 1:6,000 convention, 199 kg/m³ for
parcel under a 139 in³/lb divisor, and 1,000 kg/m³ for LCL under weight-or-measure. Cargo below the
design density is volume-limited and will be charged on cube or on a volumetric weight; cargo above
it is weight-limited. The packaging sentence should identify whether reducing cube changes the
chargeable quantity: it does for volume-limited cargo and does not for weight-limited cargo, which
is the practical reason packaging projects so often fail to show up on the freight invoice.

### Exercise 2

Starting from `c_L(w) = max(m, f + r*w)` and `c_F = P`, setting them equal in the region where the
minimum does not bind gives `f + r*w* = P`, hence `w* = (P - f)/r`. The condition for that region is
`w* >= (m - f)/r`. Using the section 6 values, `w* = (2,475.00 - 95.00)/1.10 = 2,163.6 kg`, and the
minimum-charge threshold is `(245.00 - 95.00)/1.10 = 136.4 kg`, so the region is satisfied. The
payload check passes at 2,163.6 kg against 19,900 kg; the cube check passes at 7.2 m³ against 90 m³;
the acceptance check passes at about five pallets. The sensitivity table in section 6.2 shows the
range from 1,411 kg to 3,359 kg across a plausible movement in either price, which is a factor of
2.4 and is the reason a break-even quoted without a rate date is not a fact.

### Exercise 3

The cube break-even solves `f + s*v* = P` for a cube rate `s`, giving `v* = (P - f)/s`. With `s` at
85.00 USD/m³ (Q4 2026 basis), `v* = (2,475.00 - 95.00)/85.00 = 28.0 m³`, about 17 pallets. The
crossover density, at which the weight charge and the cube charge are equal, is `s/r = 85.00/1.10 =
77.3 kg/m³`. Below that density cube binds first; above it weight binds first. For a cargo of 120
kg/m³ the weight break-even of 2,163.6 kg occupies 18.0 m³, which is below 28.0 m³, so weight binds
first. For a cargo of 400 kg/m³ the same weight occupies 5.4 m³, so weight binds first by a wide
margin. The cube break-even becomes the binding one only for very light cargo, which is exactly the
cargo for which the LTL carrier's acceptance limit is least likely to bind.

### Exercise 4

Pipeline units are `mu_d` multiplied by mean transit days. Safety stock is `1.645` times the square
root of `(R + L) * sigma_d^2 + mu_d^2 * sigma_L^2`. The inputs are `R = 7`, `mu_d = 542.9`, `sigma_d
= 120`, so the demand term is `(7 + L) * 14,400` and the lead-time coefficient `mu_d^2` is 294,697.

| Mode | L (days) | sd (days) | Pipeline units | SS (units) | Inventory value at 12.00 USD (USD) | Carrying at 22 per cent (USD/yr) |
|---|---:|---:|---:|---:|---:|---:|
| FTL | 2.0 | 0.35 | 1,086 | 670 | 21,072 | 4,636 |
| LTL | 4.2 | 1.60 | 2,280 | 1,574 | 46,248 | 10,175 |
| Intermodal | 7.5 | 2.20 | 4,071 | 3,368 | 89,268 | 19,639 |
| Air | 1.2 | 0.30 | 651 | 653 | 15,648 | 3,443 |

The intermodal safety stock is the instructive cell: the lead-time term dominates, because
`294,697 × 2.2² = 1,426,334` exceeds the demand term of `14.5 × 14,400 = 208,800` by a factor of
nearly seven.

### Exercise 5

At 12.00 USD per unit the inventory penalty is 5,541 USD per year against a freight saving of
15,080 USD per year, so LTL leads by about 9,539 USD per year before service failure and emissions.
At 120.00 USD per unit the inventory penalty is ten times larger at 55,410 USD per year, and FTL
leads by about 40,330 USD per year. The reversal value is the unit value at which the inventory
penalty equals the freight saving: inventory cost scales linearly with value at 461.8 USD per year
per dollar of unit value, so the threshold is `15,080 / 461.8 = 32.66 USD`. Adding emissions moves
the threshold slightly against LTL, since LTL is the more carbon-intensive option at 135 against 90
grams per tonne-kilometre; pricing carbon at 100.00 USD per tonne CO2e (Q4 2026 policy assumption)
adds 644.80 USD per year to the LTL option and pushes the threshold to about 31.26 USD.

### Exercise 6

A defensible assessment has five parts. Quote the claim exactly, with its date and its author.
Identify the boundary the claim implies and state whether it is tank-to-wheel or well-to-wheel.
Identify the allocation rule; if none is stated, the claim cannot be compared with any other
figure. Check the load factor and the empty-running treatment against ISO 14083:2023, which places
positioning and empty legs inside the boundary. Check the factor version against the GLEC Framework
release cited; version 3.3 of 27 October 2026 is current, and a claim built on an earlier version
may use different default factors. A verdict that says "unsupported as stated, and would be
supportable if the boundary, allocation, load factor, and factor version were disclosed" is the
correct form. A verdict that says "the number is false" overclaims, because the number may be
correct under a methodology that was not disclosed.

### Exercise 7

The dedicated option costs 18 loads per week multiplied by 52 weeks multiplied by the dedicated
per-load rate, and the shipper pays for all 18 whether or not the trough materialises. The common
option costs the actual loads at the contract rate plus the excess over contract at the spot rate.
At a trough of 9 loads the dedicated option is paying for nine empty commitments per week; at a peak
of 26 it is buying eight loads on the spot market, which erodes the service advantage that justified
the contract. The break-even peak volume is the peak at which the spot premium on the excess equals
the cost of unused commitment in the trough. The recommendation should be a base commitment sized
near the trough plus a spot and contract overlay for the peak, with the condition that the overlay
survives a tight market — which is exactly the condition that fails when capacity is scarce.

### Exercise 8

The incompatibilities are these. Unit: the CFS reports tons and value, Eurostat reports
tonne-kilometres, and no conversion between them is possible without average haul length. Coverage:
the CFS covers domestic shipments originating in the United States and excludes imports, farms,
government entities, and several industries; Eurostat covers movements on the territory of the
reporting countries under a territoriality principle. Mode definitions: the CFS reports a customer
pickup category that has no Eurostat equivalent, and Eurostat separates inland waterways, which the
CFS does not. Reference period: 2022 against 2024, in different economic conditions. Geography:
national against supranational, with different treatment of international flows. The defensible
output is two separate tables, each internally consistent, with the differences stated — not a
combined share.

### Exercise 9

A complete record has eight lines. Lane and equipment. Basis of the comparison, including whether
it is freight only or total cost. Rate date, currency, and validity period, with the note that the
rate is an observation and not a constant. The five buckets with a number and a unit each. The
assumptions, at minimum the carrying rate, the unit value, the transit standard deviations, and the
accessorial allowance. The ranking and its margin. The reversal conditions, stated as testable
thresholds: a unit value above 33 USD, a failure frequency above 64 per year, or a diesel-driven
break-even weight above the carrier's acceptance limit. The classification of each number as
forecast, plan, target, or commitment.

## 14. Interview questions

### 1. Why can two analysts compare the same two modes and reach opposite conclusions?

**Strong answer:** Because the comparison depends on the denominator and on which cost buckets are
included. Cost per tonne-kilometre rewards the mode that fills the vehicle by weight; cost per
cubic metre rewards the mode that fills it by volume. On a light shipment those two measures rank
the options differently, and the difference is not an error in either calculation but a difference
in the question being asked. The second source of divergence is scope: a freight-only comparison
and a five-bucket total-cost comparison can rank the same options differently once inventory and
service failure are included.
**Weak answer:** One of them made a mistake, or the rates are different.

### 2. Derive the LTL-to-FTL break-even and tell me when it does not apply.

**Strong answer:** Set the LTL charge, fixed plus rate times weight, equal to the FTL trip price and
solve for weight: `w* = (P - f)/r`. It does not apply when the minimum charge is binding, when the
result exceeds the payload or the cube, when the carrier will not accept an LTL shipment of that
size, when the two charges are not on the same accessorial and fuel basis, or when the inventory and
service differences would move the effective break-even.
**Weak answer:** It is around ten thousand pounds, which is the industry rule of thumb.

### 3. Why is the tail of the transit distribution more important than the mean?

**Strong answer:** The mean sets the pipeline inventory, which is a real but usually modest cost.
The standard deviation sets the safety stock, and it enters the protection-period formula multiplied
by the square of the demand rate, so the penalty scales with volume. The tail also sets the
probability of missing a customer promise, which is a service cost and sometimes a contractual one.
Two modes with the same mean and different variances have materially different total costs, and the
mean-only comparison cannot see it.
**Weak answer:** Because customers notice late deliveries.

### 4. A carrier offers a fifteen per cent lower rate on a slower service. What do you need before
you answer?

**Strong answer:** The transit distribution, not the promised transit time, so that I can compute
the change in pipeline and safety stock. The unit value and the carrying rate, so that I can price
it. The service-failure cost and the current failure frequency, so that I can price the tail. The
validity period on the rate, because a fifteen per cent offer that lasts a quarter is not a fifteen
per cent annual saving. And the accessorial basis, because a lower linehaul with a higher
accessorial exposure can be a higher total.
**Weak answer:** Whether the customer will accept the slower delivery.

### 5. Explain why empty running belongs in the emissions boundary.

**Strong answer:** Under ISO 14083:2023 the boundary is the transport chain operation, and
positioning and empty legs are part of that operation. Excluding them allocates only the loaded
movement to the cargo and understates the emissions attributable to it. The practical consequence is
that a carrier with a high empty-running ratio has a worse intensity than one with a balanced
network, even at the same vehicle efficiency, and a shipper comparing carriers on a loaded
tonne-kilometre basis will miss that difference.
**Weak answer:** Empty trucks do not carry cargo, so they do not count.

### 6. What is the difference between pipeline inventory and safety stock, and why does the
distinction matter for mode choice?

**Strong answer:** Pipeline inventory is cargo in motion; it equals the demand rate multiplied by
the mean transit time and scales with the mean. Safety stock is the buffer against variability; it
scales with the standard deviation and with the square of the demand rate. A mode change moves both,
by different amounts, and adding them without distinction double counts the inventory that the
transit time already accounts for.
**Weak answer:** They are both inventory, so I add them together.

### 7. When would you choose dedicated over common carriage?

**Strong answer:** When the volume is dense enough and stable enough that the commitment is cheaper
than the option premium, when service variance is expensive — temperature control, tight windows,
high-value handling — or when the lane is strategic and capacity in a tight market is worth paying
for in advance. The condition that reverses it is a volume pattern with a trough far below the
commitment, which converts the dedicated rate into a payment for idle capacity.
**Weak answer:** Dedicated is cheaper at high volume.

### 8. How would you critique a claim that a company cut transport emissions by forty per cent by
shifting modes?

**Strong answer:** I would ask for the boundary, the allocation rule, the load factor and
empty-running treatment, and the factor version. Without those, the percentage is a comparison
between two numbers computed under unknown conventions. I would also ask whether the baseline was
recomputed under the current methodology or carried over from an earlier one, because a
methodology change can produce a saving without any operational change. Craig, Blanco, and Sheffi
found intermodal intensities ranging from 18 to 137 grams per tonne-kilometre by lane, which is a
range wide enough to swamp a forty per cent claim if the lanes changed at the same time.
**Weak answer:** Forty per cent sounds plausible for a mode shift, so I would accept it.

### 9. Why does the cost per tonne-kilometre of a light shipment look so bad?

**Strong answer:** Because the denominator is small. The shipment pays for the cube it occupies,
and the cube does not shrink when the mass does. A 300 kg shipment in 6.3 cubic metres pays for the
same space as a 1,900 kg shipment, so its cost per tonne-kilometre is higher by roughly the ratio
of the masses, while its cost per cubic metre is lower. Neither figure is wrong; the
tonne-kilometre figure is answering a question about mass efficiency that the shipment was never
going to satisfy.
**Weak answer:** Light freight is inefficient to move.

### 10. What does a container rate index tell you, and what does it not tell you?

**Strong answer:** It tells me the direction and magnitude of spot pricing on a specific trade lane
on a specific date, which is useful for budgeting a spot-exposed lane and for understanding market
direction. It does not tell me my contract rate, my all-in cost including terminal handling and
inland legs, or my emissions, and it is volatile enough that a figure more than a few weeks old
should be treated as history. The Freightos Asia–US West Coast reading of about 8,300 USD per FEU in
the week of 3 November 2026 is a dated observation, not a planning constant.
**Weak answer:** It tells me what shipping costs.

### 11. How do you decide whether a mode comparison should be per tonne-kilometre or per cubic
metre?

**Strong answer:** By which currency binds. Compute density, compare it with the design density of
the equipment, and use the denominator that matches the binding constraint. If the cargo is denser
than the equipment's design density, mass binds and tonne-kilometres are the right view. If it is
lighter, cube binds and cubic metres are the right view. Report both when the audience includes
people who will use the other one, and state which one drove the decision.
**Weak answer:** Tonne-kilometres are the industry standard, so use those.

### 12. What would make you reject a mode decision that shows a twenty per cent saving?

**Strong answer:** If the saving is computed on freight alone and the mode is slower, I would want
the inventory and service cost before accepting it. If the rate is a spot observation without a
validity period, I would want a contract basis. If the shipment size is near the LTL acceptance
limit, I would want the carrier's confirmation that the service is available. And if the emissions
change is part of the case, I would want the allocation rule. A twenty per cent freight saving can
be a net cost increase once the other four buckets are counted.
**Weak answer:** Nothing, a twenty per cent saving is a twenty per cent saving.

## 15. Key terms

- **FTL** — full truckload; one shipper's freight in a dedicated vehicle, priced per trip.
- **LTL** — less-than-truckload; shared trailer capacity with terminal rehandling, priced per shipment plus weight or class.
- **Parcel** — individual packages through a hub-and-spoke sort network, priced per piece by weight and zone.
- **Intermodal** — rail for the long leg with truck drayage at each end, priced per container.
- **Drayage** — the short truck move between a terminal, port, or ramp and a door.
- **TEU** — twenty-foot equivalent unit; the standard container capacity measure.
- **FEU** — forty-foot equivalent unit; the container measure used in most published ocean rates.
- **FCL** — full container load; one shipper's cargo in a dedicated container.
- **LCL** — less than container load; consolidated cargo, charged on weight or measure.
- **Weight or measure** — the LCL rule charging the greater of tonnes and cubic metres.
- **Chargeable weight** — the greater of actual and volumetric weight under a carrier convention.
- **Dimensional divisor** — the cube-to-weight conversion used by parcel carriers, commonly 139 or 166 cubic inches per pound.
- **Density** — mass divided by volume, in kilograms per cubic metre.
- **Design density** — payload divided by volume for a piece of equipment; the cargo density at which weight and cube fill together.
- **Load factor** — actual load divided by capacity, by weight or by cube.
- **Empty running** — vehicle kilometres travelled without revenue cargo, inside the ISO 14083 boundary.
- **Linehaul** — the long-distance movement charge, excluding terminal handling and accessorials.
- **Accessorial** — a charge for a service beyond the base move, such as liftgate, appointment, or detention.
- **Minimum charge** — the floor charge for a shipment regardless of size.
- **Freight class** — the NMFC classification that sets the LTL rate basis.
- **Break-even weight** — the shipment mass at which two modes cost the same on a stated lane and date.
- **Tonne-kilometre** — one tonne moved one kilometre; the standard unit of transport work.
- **Pipeline inventory** — cargo in motion, equal to the demand rate multiplied by mean transit time.
- **Safety stock** — the buffer held above expected demand over the protection period.
- **Allocation rule** — the method assigning a shared vehicle's emissions to one shipment, by mass, volume, or slot.
- **Dedicated carriage** — capacity committed to one shipper for a contract term.
- **Common carriage** — capacity offered to any shipper under published terms.
- **Spot rate** — a one-off market price for a single move, dated and volatile.

## 16. Sources

**Peer-reviewed and academic (A)**

- Baumol, W. J., and Vinod, H. D. (1970). "An Inventory Theoretic Model of Freight Transport
  Demand." *Management Science*, 16(7), 413–421. (A) — the canonical statement that mode choice is
  an inventory decision trading rate against speed and dependability; supports section 9.3.
- Hummels, D. L., and Schaur, G. (2013). "Time as a Trade Barrier." *American Economic Review*,
  103(7), 2935–2959. (A) — estimates each day in transit at roughly 0.6 to 2.1 per cent ad valorem;
  the empirical anchor for pricing transit time in section 6.4.
- Craig, A. J., Blanco, E. E., and Sheffi, Y. (2013). "Estimating the CO2 intensity of intermodal
  freight transportation." *Transportation Research Part D*, 22, 49–53. (A) — 67 g CO2 per ton-mile
  average over more than 400,000 intermodal shipments, with a lane range of 29 to 220 g; supports
  the allocation and lane-variance argument in section 9.4.
- Özkaya, E., Keskinocak, P., Joseph, V. R., and Weight, R. (2010). "Estimating and benchmarking
  Less-than-Truckload market rates." *Transportation Research Part E*, 46(5), 667–682. (A) —
  regression-based LTL rate benchmarking and the drivers of LTL pricing; supports the cost
  architecture in section 3.2.
- Piecyk, M. I., and McKinnon, A. C. (2010). "Forecasting the carbon footprint of road freight
  transport in 2020." *International Journal of Production Economics*, 128(1), 31–42. (A) —
  scenario modelling of road freight emissions with load factor and efficiency assumptions; supports
  the load-factor discussion in section 8.2.

**Standards and regulation (B)**

- International Organization for Standardization (2023). *ISO 14083:2023, Greenhouse gases —
  Quantification and reporting of greenhouse gas emissions arising from transport chain operations*.
  Published 20 March 2023. (B) — the calculation standard for section 8.2 and the basis for
  assessing any emissions claim.
- Smart Freight Centre (2026). *GLEC Framework for Logistics Emissions Accounting and Reporting*,
  version 3.3, released 27 October 2026. (B) — the practical structure and default factor set;
  version 3.1 (2023) was the release aligned to ISO 14083.
- US Census Bureau and Bureau of Transportation Statistics (2025). *2022 Commodity Flow Survey*,
  released 26 June 2025. (B) — 12.2 billion tons and 18.0 trillion USD of shipments, trucks at 68.1
  per cent of tonnage; the frame for section 8.1.
- Eurostat (2026). *Freight transport statistics — modal split*, 2024 reference year, data extracted
  February 2026. (B) — EU modal shares in tonne-kilometres, maritime 67.0 per cent and road 25.7 per
  cent; the comparison frame in section 8.1.
- International Maritime Organization (2022–2023). MARPOL Annex VI amendments; EEXI and CII
  certification effective 1 January 2023. (B) — the regulatory context for ocean carbon intensity
  in section 8.3.
- National Motor Freight Traffic Association. *National Motor Freight Classification*. (B) — the LTL
  class basis referenced in section 4.
- International Chamber of Commerce (2019). *Incoterms 2020*. (B) — allocates cost and risk between
  buyer and seller, and therefore whose mode decision the move is.

**Industry and company (C)**

- Freightos (2026). Weekly market update, 3 November 2026, reporting Asia–US West Coast container
  rates near 8,300 USD per FEU and a composite index near 3,500 USD per FEU. (C) — a commercial index
  publisher's dated observation; supports the volatility argument in section 9.2 and is not a
  planning constant.

**Press (E)**

- Hellenic Shipping News (2026). "Container Report — Week 44", 2 November 2026, reporting the
  Asia–US West Coast assessment at 8,342 USD per FEU. (E) — a press summary of a weekly index
  reading, used only to corroborate the direction of the Freightos observation.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Part II map: [`../_index.md`](../_index.md)
- Prerequisite, units and measurement: [`../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md`](../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md)
  — the hierarchy that decides whether a shipment is counted in eaches, cases, or kilograms before
  it is priced.
- Prerequisite, lead-time definitions: [`../../01_foundations/01A_foundations_units_time_assumptions/05_lead-time-definitions.md`](../../01_foundations/01A_foundations_units_time_assumptions/05_lead-time-definitions.md)
  — the definitions behind the transit-time distribution in section 5.7.
- Safety stock under demand and lead-time uncertainty:
  [`../10_inventory_optimization_multi_echelon/04_safety-stock.md`](../10_inventory_optimization_multi_echelon/04_safety-stock.md)
  — the protection-period formula used in step 16, not re-derived here.
- Module 10 map: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Cost to serve on a graph:
  [`../11_network_design_optimization/02_graph-cost-to-serve.md`](../11_network_design_optimization/02_graph-cost-to-serve.md)
  — the delivered-cost decomposition this file fills in at the mode layer.
- Survey file:
  [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
  — the narrative arc, the planning emission ranges quoted in section 6.6, and the rate snapshot of
  27 October 2026.
- Next in this module: `02_rate-structures.md` (planned, not yet on disk) — linehaul, fuel,
  minimums, demurrage, detention, and accessorial creep, which this file deliberately leaves as an
  allowance.
- Then: `03_lane-tender-analytics.md` (planned) — lane segmentation and acceptance rates.
- Then: `04_consolidation-cube.md` (planned) — the weight-and-cube consolidation question that
  section 6.3 opens and does not close.
- Then: `05_vrp-formulation.md` (planned) — vehicle routing inside the chosen mode.
