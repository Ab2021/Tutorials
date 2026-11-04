# Consolidation and Cube Utilisation

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 12, topic 01
> Version 2026.5

*Weight, cube, and the cost of running half-empty.*

---

## 1. Why this topic exists

A truck is a box with two capacities, and it charges for the box, not for what is inside it. That
sentence explains most of the money left on the table in a road freight budget. You pay for the trip.
You fill it with mass, with volume, or with neither. Every point of unused capacity is paid for by
somebody, and in most organisations nobody knows who.

The mechanics are unglamorous. A load is a pair of numbers, kilograms and cubic metres. A vehicle is
a pair of limits, permitted payload and usable volume. Whether a load fits is two inequalities, not
one. Whether two loads should travel together is three tests, not one. The arithmetic fits on a page.
The topic needs three hours because the arithmetic is the easy part and the data behind it is almost
always wrong.

Three pathologies recur, each with a price attached.

1. **The wrong dimension is optimised.** A planner builds a trailer to 95% of its volume and reports a
   full load. The trailer carries 4 tonnes against a permitted payload of 20 tonnes. The invoice is
   for a whole trailer, the fuel burn is for a whole trailer, and most of the asset walked down the
   road empty. Because the load looked full on the dock, nobody questioned it.
2. **Consolidation is assumed to save money.** Two half-loads on the same lane are combined, a
   stop-off charge is applied, and the combined invoice is higher than the two separate invoices
   were. The distance saved was real; the money saved was not.
3. **The cube data does not exist.** Cube analysis needs item dimensions, which live in the item
   master, which is populated at item creation by people with a product in front of them and no
   measuring tape. A cube model built on defaulted dimensions is wrong about which constraint binds,
   which is the only thing the model was for.

What this topic is not: it is not rate construction. How a carrier builds a rate is
`02_rate-structures.md`. This topic takes the price as given and asks which shipments should share a
vehicle, and how full the vehicle that results. Route sequencing is `05_vrp-formulation.md`; mode cost
architecture is `01_freight-modes.md`; lane segmentation is the planned topic
`03_lane-tender-analytics.md`. Consolidation is process and data work, not a capital project: it
needs a tape measure, a correct payload figure, and the discipline to run three tests instead of one.

---

## 2. Learning outcomes

By the end of this topic you should be able to:

1. Compute cube utilisation, weight utilisation, and design density for a named vehicle, and state
   which constraint binds.
2. Derive the crossover density for a vehicle and classify a shipment before any load plan is drawn.
3. Apply the three consolidation tests and identify which one fails in a given case.
4. Build a cube-bound load by volume and a weight-bound load by mass, and explain why swapping the two
   produces a load that looks full and costs more.
5. Compute cost per kilogram and per cubic metre at several utilisation levels, and explain why the
   marginal cost of the last pallet is zero until the box is full.
6. Derive the dispatch rule for temporal consolidation and check it against the vehicle ceiling and
   the service limit.
7. State the US and EU weight and dimension limits, and identify the axle limit that binds first.
8. Quantify how pallet footprint, overhang, stackability, and packaging move chargeable volume.
9. Diagnose the four common failures of a cube dataset and the measurement that prevents them.
10. Write a one-page consolidation decision stating the tests, the arithmetic, and the counterfactual.

---

## 3. The core mental model

### 3.1 The box and the ray

A vehicle is a rectangle in a space whose axes are mass and volume, with its corner at (P, V): P the
permitted payload in kilograms, V the usable volume in cubic metres. A load is a point (m, v). Draw
the ray from the origin through the load point; its slope is the cargo density, ρ = m / v. The ray
leaves the rectangle through the top edge if mass runs out first, or the right edge if volume does.
The slope separating the two cases is the diagonal:

```text
design density   ρ* = P ÷ V          [kg/m³]
cargo density    ρ  = m ÷ v          [kg/m³]

ρ > ρ*   →  weight binds
ρ = ρ*   →  both bind simultaneously
ρ < ρ*   →  cube binds
```

One number, computed once per vehicle, tells you which wall any shipment will hit. You do not need
the shipment's dimensions, only its density and one number that belongs to the vehicle.

### 3.2 Two utilisations, one constraint

```text
cube utilisation    u_v = occupied m³ ÷ usable m³
weight utilisation  u_m = payload kg ÷ permitted payload kg
```

For a single load, the binding constraint is the one with the higher utilisation. The density test is
the same statement in different algebra, and the two agree whenever the load is homogeneous.

Both numbers must be reported because each alone misleads. A load at 95% cube and 18% weight is
cube-constrained: build it by volume and accept that the trailer runs far below its legal mass. A
load at 60% cube and 98% weight is weight-constrained: the floor will have gaps, and the only lever
is mass per unit. A single load-factor number hides which lever is available, and it hides the
denominator, so state the denominator: "83.3% of 103.68 m³ usable" is a fact, "83% cube" is a claim.

### 3.3 The three tests

Consolidation is the act of putting two loads into one box. It is available only when three
independent tests pass at once. Passing two of three is the normal case.

**Time fusion.** Each shipment has a release time r and a deadline d, the latest acceptable arrival
minus transit. Two shipments can share a dispatch if their feasible intervals intersect:

```text
time fusion holds  ⟺  max(r_1, r_2) ≤ min(d_1, d_2)
```

If the interval is empty, no amount of money makes the consolidation legal on time. You can widen an
interval only by paying for it, and that payment belongs in the cost test.

**Physical fit.** Both dimensions must be within limits:

```text
physical fit holds  ⟺  (m_1 + m_2 ≤ P)  AND  (v_1 + v_2 ≤ V)
```

A pair that fits on mass and not volume fails; a pair that fits on volume and not mass fails. The
aggregate test is necessary but not sufficient, because it ignores geometry: a 4.5 m item will not
fit a 2.4 m wide trailer however small its volume. Treat it as a screen and check the longest item
and the stack height separately.

**Cost fusion.** The combined price must be lower than the sum of the prices of the moves it
displaces, including the base trip, any additional distance, and every stop-off charge:

```text
cost fusion holds  ⟺  C_combined  <  Σ C_i
```

The discipline is on the right-hand side: the sum must use the prices actually available, not a
modelled cost. A shipment riding a backhaul at a low price is an expensive thing to consolidate,
because consolidation destroys the backhaul rate and replaces it with a stop-off charge.

### 3.4 The fixed trip price

A trip has a price that does not vary with what is inside the vehicle. While there is slack, the
marginal cost of one more pallet is **zero**: the trip runs anyway, and the average cost per pallet
is P / n, which falls as 1/n. When the box is full the marginal cost does not rise smoothly; it jumps
to the price of an entire additional trip. The first units added to a partly loaded vehicle are free,
the last are not, and once the box is full the only remaining levers are density and packaging.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Consolidation | Combining two or more shipments into one movement so shared capacity is used once rather than twice. | Used loosely for any cost reduction on a lane, including a rate negotiation. | Boolean per shipment group; no unit. |
| Cube utilisation | Occupied volume divided by usable vehicle volume. | Computed against the nominal box rather than the measured usable volume. | Ratio 0–1, reported as %; volume in m³. |
| Weight utilisation | Payload mass divided by permitted payload mass. | Computed against gross vehicle weight instead of payload. | Ratio 0–1, %; mass in kg. |
| Payload | The mass a vehicle may carry: gross vehicle weight minus tare. | Used interchangeably with gross vehicle weight. | kg (US practice: lb); store as kg. |
| Permitted payload | The payload allowed for a specific unit on a specific route, after tare and any route limit. | Taken from a brochure rather than the vehicle plate. | kg. |
| Tare weight | The unladen mass of the tractor and trailer together. | Omitted, so payload is overstated by the whole tare. | kg. |
| Usable volume | Enclosed volume actually available after wheel boxes, lining, and decking. | Replaced by nominal length × width × height. | m³. |
| Design density | Permitted payload divided by usable volume: the density at which mass and volume fill at the same time. | Confused with the density of the freight being carried. | kg/m³. |
| Cargo density | Mass of the freight divided by its volume. | Computed on the pallet footprint only, ignoring stack height. | kg/m³. |
| Crossover density | The design density used when comparing two vehicles or two packaging variants. | Quoted without saying which vehicle it applies to. | kg/m³. |
| Cube-constrained load | A load that reaches its volume limit before its mass limit. | Called "full" while 80% of payload is unused. | Classification label. |
| Weight-constrained load | A load that reaches its mass limit before its volume limit. | Called "inefficient" although it is at the legal limit. | Classification label. |
| Chargeable weight | The greater of actual mass and volumetric mass, used for pricing. | Assumed equal to actual mass on air and parcel moves. | kg, derived. |
| Dimensional weight | Volume converted to a mass equivalent by a divisor. | Computed with a divisor from a different unit system. | kg; divisor in cm³/kg or in³/lb. |
| Volumetric divisor | The constant converting volume to chargeable mass (6,000 cm³/kg air; 139 or 166 in³/lb parcel). | Mixed between metric and imperial without conversion. | cm³/kg or in³/lb. |
| Time fusion | The condition that shipments share a dispatch window: their feasible intervals intersect. | Checked against the deadline only, never against the release time. | Timestamps in UTC; interval overlap. |
| Physical fit | The condition that combined mass and combined volume are both within limits. | Checked on mass only, or on a simple pallet count. | kg and m³; two inequalities. |
| Cost fusion | The condition that the combined price is below the sum of the prices of the displaced moves. | Compared against a modelled distance saving rather than actual prices. | Currency units; 2026 Q1 basis. |
| Stop-off charge | A tariff charge for an additional pickup or delivery stop on one trip. | Omitted from the consolidation comparison. | Currency per stop. |
| Temporal consolidation | Holding shipments to accumulate a fuller load before dispatch. | Treated as free, with the delay cost never booked. | Days held; cost per kg per day. |
| Quantity policy | Dispatch when the accumulated load reaches a threshold Q0. | Confused with a time policy in reporting. | kg. |
| Time policy | Dispatch every T0 days regardless of accumulated load. | Used where order arrival is lumpy and loads vary widely. | days. |
| Time-and-quantity policy | Dispatch at whichever of Q0 or T0 is reached first. | Implemented as two independent rules that never interact. | kg and days. |
| Milk run | A fixed route collecting from or delivering to several points on a schedule. | Used for any multi-stop route with no fixed schedule. | Stops per run; km per run. |
| Cross-docking | Moving freight from inbound to outbound vehicles with little or no storage. | Used for any warehouse transfer, including put-away. | Door-hours; units per day. |
| Merge-in-transit | Combining shipments from different origins at an intermediate point before the final leg. | Confused with cross-docking at a distribution centre. | Number of merge points. |
| DC bypass | Shipping direct from source to customer, skipping the distribution centre. | Applied without checking the receiving capability at the customer. | Boolean per order. |
| Pooling | Sharing capacity across customers, suppliers, or business units. | Assumed to carry no coordination cost. | Vehicles or kg shared. |
| Load factor | The share of available capacity used, in mass, volume, or both. | Reported as one number without stating the dimension. | %, with the dimension stated. |
| Pallet position | One floor location for a pallet on a vehicle. | Counted including positions blocked by wheel boxes. | Count; 32 on the design vehicle in this topic. |
| Pallet overhang | Pallet load extending beyond the pallet footprint. | Ignored in the width calculation. | mm per side. |
| Stackability | Whether a unit load can carry another above it. | Assumed from the product type rather than tested. | Boolean; maximum stack height in m. |
| Unit load | The pallet, stillage, or roll cage as handled, including the pallet itself. | Treated as the carton. | Count; dimensions in mm. |
| Gross vehicle weight | The mass of tractor, trailer, and payload together. | Used as the payload limit. | kg (US practice: lb). |
| Axle load | The mass carried by one axle or axle group. | Checked only in total, never per axle. | kg per axle or axle group. |
| Bridge formula | The US federal formula relating permitted gross weight to axle count and spacing. | Assumed to be a flat 80,000 lb limit on every configuration. | lb; W = 500[(LN/(N−1)) + 12N + 36]. |
| Backhaul | A return leg priced low because the vehicle would otherwise run empty. | Assumed to be available on every lane. | Currency per trip. |
| Empty running | Kilometres travelled without cargo. | Excluded from emissions and cost per tonne-km. | km; % of total km. |
| Deck rating | The maximum distributed load per unit area that the trailer floor carries. | Ignored for dense loads on small footprints. | kg/m². |
| Cube master data | The item-level length, width, height, and mass record used in every cube calculation. | Left at a default value after item creation. | mm and kg per item. |

---

## 5. The mechanics, step by step

Each step produces an artefact the next step consumes. Skipping one produces a number that looks like
an answer.

### Step 1 — Establish the design vehicle

Start from the vehicle, not the shipment.

- **Permitted payload P.** Read the plate for GVW and tare and subtract. Then check the route: a
  state or national limit lower than the plate limit governs. Payload is the smaller of the two.
- **Usable volume V.** Measure the internal length, width, and height that freight can occupy. Deduct
  wheel boxes, the nose taper, lining, and decking. Do not use the nominal body length.
- **Design density ρ\* = P / V.** Compute it once and write it on the wall.

For the vehicle used throughout, **DV-53**, a 53-foot dry van on a US Interstate lane:

```text
P  = 45,000 lb = 20,400 kg
     basis: 80,000 lb federal GVW limit − 35,000 lb tractor and trailer tare
V  = 16.00 m × 2.40 m × 2.70 m = 103.68 m³
     measured usable envelope, after lining and wheel boxes
ρ* = 20,400 kg ÷ 103.68 m³ = 196.8 kg/m³
```

The same arithmetic for five representative vehicles, all design assumptions to be replaced by the
actual unit's plate and body specification:

| Vehicle | Permitted payload (kg) | Usable volume (m³) | Design density (kg/m³) | Cargo that binds cube | Cargo that binds weight |
|---|---:|---:|---:|---|---|
| Cargo van, 3.5 t GVW | 1,200 | 12.0 | 100.0 | Packaged bedding | Bottled liquids |
| Rigid 7.5 t | 3,000 | 32.0 | 93.8 | Flat-pack furniture | Bagged aggregate |
| Rigid 18 t | 9,500 | 45.0 | 211.1 | Insulation | Canned goods |
| Articulated 40 t (EU) | 25,000 | 100.0 | 250.0 | White goods | Ceramic tile |
| Articulated 53 ft (US, DV-53) | 20,400 | 103.7 | 196.8 | Cushion foam | Bagged polymer |

Read this as thresholds, not a ranking. The 40 t articulated unit binds on weight above 250 kg/m³;
the 7.5 t rigid binds above 93.8 kg/m³. The same cargo can be cube-bound in one vehicle and
weight-bound in another, which is why "this is a cube product" is not a valid statement.

### Step 2 — Measure the load

Measure mass and cube at the same level of aggregation. The common error is mixing levels: mass per
carton with volume per pallet, or mass including the pallet with volume excluding it. Choose a unit
load and measure both dimensions consistently, including the pallet's own mass and height.

```text
v_unit  = footprint m² × stack height m       [m³ per pallet]
m_unit  = gross mass per pallet               [kg, including the pallet]
ρ_cargo = m_unit ÷ v_unit                     [kg/m³]
```

The stack height, not the pallet height, is what occupies trailer volume.

### Step 3 — Compute both utilisations and name the binding constraint

```text
u_m = Σ m_i ÷ P
u_v = Σ v_i ÷ V
binding = argmax(u_m, u_v)
```

Report both; name the binding one. This is the most useful line in any load plan and the one most
often missing. The fast equivalent for a single commodity: if ρ_cargo > ρ\* the load will weight-bind
as it fills; if ρ_cargo < ρ\*, it will cube-bind.

### Step 4 — Build the load in the binding dimension

**Cube-bound.** The objective is units per cubic metre, subject to mass remaining legal. The levers
are volumetric: raise stack height to the ceiling less clearance; reduce per-unit packaging, which
raises units per cubic metre directly; choose a pallet pattern that wastes less floor, including
mixed-SKU pallets; nest or knock down where the product allows. The load will finish far below its
mass limit, which is the definition rather than a failure, and the response is to raise density,
never to add ballast.

**Weight-bound.** The objective is mass, subject to volume remaining legal. The levers are
gravimetric: reduce mass per unit; distribute mass across axles so no axle limit binds before gross
weight; accept floor gaps and plan so they do not become load-shift risk; check the deck rating,
because a dense load on a small footprint can exceed the floor's kg/m² limit while the total is
legal.

**The failure in one line.** Building a cube-bound load by mass adds heavy items until an axle binds,
producing a trailer that is 40% full by volume and costs the same as a full trailer to run. Building
a weight-bound load by volume stacks light items on a dense base, producing a trailer that looks full
at 60% of payload with a raised centre of gravity. Both optimise the dimension that is not binding.

### Step 5 — Test time fusion

```text
r_i = ready time at origin
d_i = latest dispatch = required arrival − transit time − unload time
fusion holds  ⟺  max(r_i) ≤ min(d_i)
```

Case that passes. Shipment A is ready Monday 06:00 and must arrive Wednesday 12:00; transit 8 hours,
so the latest dispatch is Wednesday 04:00. Shipment B is ready Tuesday 10:00 and must arrive Tuesday
22:00; transit 10 hours, so the latest dispatch is Tuesday 12:00. The intervals are
[Mon 06:00, Wed 04:00] and [Tue 10:00, Tue 12:00]; the intersection is [Tue 10:00, Tue 12:00], and a
Tuesday 10:30 dispatch satisfies both.

Case that fails. Shipment A is not ready until Tuesday 06:00. Shipment B must arrive Tuesday 08:00
with 10 hours of transit, so its latest dispatch is Monday 22:00. The intersection is empty. No
dispatch time satisfies both and no price fixes it; the only routes forward are to move A earlier or
B later, and either cost belongs in Step 7.

### Step 6 — Test physical fit

Apply both inequalities, then check geometry.

```text
m_A + m_B ≤ P        and        v_A + v_B ≤ V
```

Three geometry conditions the aggregate test ignores: the longest item against internal length and
door opening; the tallest stack against internal height including clearance; the heaviest
concentrated item against the deck rating in kg/m². If the aggregate passes and geometry fails, the
pair does not fit and the aggregate was a screen rather than a proof. Palletised loads usually pass
on aggregate and fail on the last row.

### Step 7 — Test cost fusion

```text
C_combined = base trip price
           + (additional distance × per-km rate)
           + Σ stop-off charges
           + any service cost incurred to make time fusion hold

Σ C_i = price of move A as it would actually be bought
      + price of move B as it would actually be bought
```

Three disciplines make this honest. Use prices, not costs: a modelled cost that ignores the backhaul
rate on move B will show a saving that does not exist. Include every stop-off charge, because it is
often larger than the linehaul saving when the second shipment is small. Price the service change: if
fusion required holding a shipment two days, the carrying cost belongs in C_combined even if no
invoice carries it.

### Step 8 — Decide, and record the counterfactual

Record the decision, the three results, and the arithmetic. The counterfactual is what makes it
auditable six months later: "A and B were kept separate because the stop-off charge of
USD 450.00 (2026 Q1) exceeded the linehaul saving of USD 263.50 (2026 Q1) by USD 186.50 (2026 Q1)."

### The three tests as a decision table

| Case | Time fusion | Physical fit | Cost fusion | Outcome |
|---|---|---|---|---|
| 1 | pass | pass | pass | Consolidate. Record the arithmetic. |
| 2 | pass | pass | fail | Keep separate. This is the section 6 worked example. |
| 3 | fail | pass | pass | Consolidate only if a window moves; price the service change first. |
| 4 | pass | fail on mass | not reached | Split, or move one shipment to a second trip. |
| 5 | pass | fail on cube | not reached | Split, or check whether a taller or wider unit is available. |
| 6 | fail | fail | pass | No consolidation. The saving is real but unreachable on these dates. |
| 7 | pass | pass, tight | pass | Consolidate, and flag that the next shipment will not fit. |

Row 6 is the case planners most often miss: the cost case is strong, the loads would fit, and the
dates are incompatible. The correct output is not "consolidate with a longer hold" but "the saving is
unavailable on these dates; re-test next cycle".

---

## 6. Worked example

Every figure below is given in the problem statement or derived in the arithmetic shown. Money is in
US dollars at 2026 Q1 price levels; masses in kilograms; volumes in cubic metres.

### 6.1 The problem

DV-53: permitted payload 20,400 kg, usable volume 103.68 m³, design density 196.8 kg/m³.

**Load A — cushion foam, cube-bound.** Thirty pallet stacks, each on a 1.2 m × 1.0 m pallet, stacked
to 2.40 m, weighing 120 kg gross.

```text
v_A = 30 × (1.2 × 1.0 × 2.40) = 30 × 2.88 = 86.40 m³
m_A = 30 × 120 = 3,600 kg
ρ_A = 3,600 ÷ 86.40 = 41.7 kg/m³
```

**Load B — bagged polymer, weight-bound.** Six pallets, each 1.2 m × 1.0 m × 1.20 m, weighing
1,000 kg gross.

```text
v_B = 6 × (1.2 × 1.0 × 1.20) = 6 × 1.44 = 8.64 m³
m_B = 6 × 1,000 = 6,000 kg
ρ_B = 6,000 ÷ 8.64 = 694.4 kg/m³
```

**The moves.** A goes from the DC to customer X, 420 km. B goes from the same DC to customer Y,
505 km. X lies on the route to Y; the leg from X to Y is 85 km.

**The prices, 2026 Q1.** Move A as a dedicated 53-foot dry van: USD 1,850.00. Move B as a volume LTL
shipment: USD 640.00. The combined move adds 85 km at USD 3.10/km and one stop-off at X at
USD 450.00.

### 6.2 Utilisation and the binding constraint

```text
Load A:  u_m = 3,600 ÷ 20,400 = 17.65%
         u_v = 86.40 ÷ 103.68 = 83.33%     → cube binds

Load B:  u_m = 6,000 ÷ 20,400 = 29.41%
         u_v = 8.64 ÷ 103.68 = 8.33%       → weight binds

Combined: u_m = 9,600 ÷ 20,400 = 47.06%
          u_v = 95.04 ÷ 103.68 = 91.67%   → cube binds
```

The density check agrees: A at 41.7 kg/m³ is far below the 196.8 kg/m³ crossover, B at 694.4 kg/m³
far above it.

**Filled with A's cargo**, DV-53 would hold 103.68 m³ at 41.7 × 103.68 = 4,320 kg, which is 21.2% of
permitted payload. A trailer completely full of cushion foam carries one fifth of the mass it is
allowed to carry. **Filled with B's cargo**, the volume would take 694.4 × 103.68 = 72,000 kg, which
is 3.53 times the permitted payload; B therefore weight-limits at 20,400 kg in
20,400 ÷ 694.4 = 29.38 m³, which is 28.3% of the volume.

Neither number criticises the product. They are the two ends of the same design-density fact, and
they explain why A and B are complementary: one has volume to spare, the other has mass to spare.

### 6.3 Time fusion

- A is ready Monday 06:00 and must arrive at X by Wednesday 12:00. Transit is 8 hours and unloading
  1 hour, so the latest dispatch is Wednesday 03:00.
- B is ready Tuesday 08:00 and must arrive at Y by Wednesday 18:00. Transit on the combined route is
  8 + 1 + 2 + 1 = 12 hours, so the latest dispatch is Wednesday 05:00.

```text
A interval:   [Mon 06:00, Wed 03:00]
B interval:   [Tue 08:00, Wed 05:00]
intersection: [Tue 08:00, Wed 03:00]     non-empty
```

Time fusion holds. A dispatch on Tuesday at 18:00 delivers X on Wednesday at 03:00 and Y on Wednesday
at 09:00, both inside their deadlines.

### 6.4 Physical fit

```text
mass:  3,600 + 6,000 = 9,600 kg   ≤ 20,400 kg      pass (47.06% of payload)
cube:  86.40 + 8.64 = 95.04 m³    ≤ 103.68 m³      pass (91.67% of volume)
slack: 103.68 − 95.04 = 8.64 m³
```

Both inequalities hold and the combined load is cube-bound at 91.67%, with 8.64 m³ of slack — exactly
one more pallet of B and no more. The longest item is 1.2 m against a 16.0 m internal length, the
tallest stack 2.40 m against a 2.70 m internal height with 0.30 m clearance, and the combine requires
re-orienting B into the slack at the rear, which is feasible because B is a single stack height.
Physical fit passes, but narrowly, and the narrowness matters for the next decision on this lane.

### 6.5 Cost fusion

```text
C_combined = 1,850.00  base trip to X
           +   263.50  85 km × USD 3.10/km
           +   450.00  stop-off at X (2026 Q1 tariff)
           = 2,563.50 USD

Σ C_i      = 1,850.00  move A
           +   640.00  move B as volume LTL
           = 2,490.00 USD

C_combined − Σ C_i = 2,563.50 − 2,490.00 = +73.50 USD
```

Cost fusion fails. The combined move costs USD 73.50 (2026 Q1) more than the two separate moves, a
3.0% increase on the separate total.

The result is counter-intuitive and worth sitting with. The combined route is 505 km against 925 km
for the two separate moves, a saving of 420 vehicle-km; fuel, driver time, and emissions all fall.
And the invoice rises, because the displaced move was a small LTL shipment at USD 640.00 (2026 Q1)
while the increment added to the truck was USD 263.50 + USD 450.00 = USD 713.50 (2026 Q1).
Consolidation replaced a cheap purchase with an expensive one.

### 6.6 The break-even stop-off charge

```text
1,850.00 + 263.50 + S = 2,490.00
S = 2,490.00 − 1,850.00 − 263.50 = 376.50 USD
```

Any stop-off below USD 376.50 (2026 Q1) makes this consolidation pay. The quoted charge of
USD 450.00 is USD 73.50 above the threshold, so the question for the carrier is specific and
answerable: can the second stop be priced at or below USD 376.50 (2026 Q1) on this lane. That is a
better question than "can you improve the rate".

### 6.7 Sensitivity: what would make it pay

| Change | New arithmetic | Result |
|---|---|---|
| Stop-off negotiated to USD 350.00 (2026 Q1) | 1,850.00 + 263.50 + 350.00 = 2,463.50 | Saves USD 26.50 |
| B grows to 14 pallets, 20.16 m³ | 86.40 + 20.16 = 106.56 m³ against 103.68 m³ | Does not fit |
| Second stop at Y only, on the direct route | 1,850.00 + 450.00 = 2,300.00 against 2,490.00 | Saves USD 190.00 |

The second row is the important one. The loads that fit are precisely the small ones, and small loads
are precisely the ones whose standalone price is low. That is the structural tension in
consolidation: the physical envelope admits the shipments the cost test rejects.

### 6.8 The half-empty trailer, priced

The trip price is USD 1,850.00 (2026 Q1) whether the trailer is full or not. The table holds the
price fixed and varies the number of 120 kg, 2.88 m³ stacks on board.

| Stacks | Mass (kg) | Volume (m³) | Weight util. | Cube util. | Cost per stack (USD) | Cost per kg (USD) | Cost per m³ (USD) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 960 | 23.04 | 4.7% | 22.2% | 231.25 | 1.9271 | 80.30 |
| 16 | 1,920 | 46.08 | 9.4% | 44.4% | 115.63 | 0.9635 | 40.15 |
| 24 | 2,880 | 69.12 | 14.1% | 66.7% | 77.08 | 0.6424 | 26.77 |
| 32 | 3,840 | 92.16 | 18.8% | 88.9% | 57.81 | 0.4818 | 20.07 |

Read the last row first. A trailer at 88.9% cube and 18.8% weight costs USD 0.4818 per kg (2026 Q1)
to move. A trailer filled with weight-bound cargo B to its legal limit costs
USD 1,850.00 ÷ 20,400 kg = USD 0.0907 per kg (2026 Q1). The cube-bound load costs 5.3 times more per
kilogram and 0.32 times as much per cubic metre (USD 20.07 against USD 62.98, 2026 Q1).

Neither number is right on its own. The denominator must match the constraint: cost per cubic metre
for a cube-bound load, because the cubic metre is what runs out, and cost per kilogram for a
weight-bound load. Reporting cost per kilogram for a cube-bound load and concluding the product is
unprofitable is a denominator error, not a finding.

### 6.9 The marginal pallet

At 24 stacks the trailer has 34.56 m³ of slack. Adding the 25th stack costs nothing: the trip price
is unchanged, the fuel is within a rounding error, the wage is fixed. The marginal cost is USD 0.00
(2026 Q1) and the average falls from USD 77.08 to USD 74.00 (2026 Q1). At 32 stacks there are no
floor positions left, so the 33rd stack cannot ride and the marginal cost is USD 1,850.00 (2026 Q1),
the price of a second trip. The cost curve is flat at zero and then vertical.

---

## 7. Code, query, or calculation

The script reproduces every figure in section 6. It is short by design: the arithmetic is the easy
part, and making it reproducible is what allows the decision to be audited.

```python
# cube_consolidation.py - utilisation, binding constraint, and cost fusion
DV = {"payload_kg": 20_400, "volume_m3": 103.68}   # design vehicle DV-53

def utilisation(loads):
    kg = sum(l["kg"] for l in loads)
    m3 = sum(l["m3"] for l in loads)
    return kg / DV["payload_kg"], m3 / DV["volume_m3"]

A = {"kg": 3_600.0, "m3": 86.40}    # 30 stacks, 120 kg and 2.88 m3 each
B = {"kg": 6_000.0, "m3": 8.64}     # 6 pallets, 1 000 kg and 1.44 m3 each

for name, loads in (("A", [A]), ("B", [B]), ("A+B", [A, B])):
    w, c = utilisation(loads)
    binds = "weight" if w > c else "cube"
    fits = "yes" if w <= 1 and c <= 1 else "no"
    print(f"{name:3s} weight={w:6.2%} cube={c:6.2%} binding={binds:6s} fits={fits}")

rho_star = DV["payload_kg"] / DV["volume_m3"]
print(f"design density = {rho_star:.1f} kg/m3")
for name, l in (("A", A), ("B", B)):
    print(f"  {name} density = {l['kg']/l['m3']:.1f} kg/m3 -> "
          f"{'weight' if l['kg']/l['m3'] >= rho_star else 'cube'}-bound")

base, rate, extra_km, stop = 1_850.00, 3.10, 85, 450.00
combined = base + rate * extra_km + stop
separate = base + 640.00
print(f"combined = USD {combined:,.2f}; separate = USD {separate:,.2f}; "
      f"delta = USD {combined - separate:+,.2f}")
print(f"break-even stop charge = USD {640.00 - rate * extra_km:,.2f}")
```

Expected output:

```text
A   weight=17.65% cube=83.33% binding=cube   fits=yes
B   weight=29.41% cube= 8.33% binding=weight fits=yes
A+B weight=47.06% cube=91.67% binding=cube   fits=yes
design density = 196.8 kg/m3
  A density = 41.7 kg/m3 -> cube-bound
  B density = 694.4 kg/m3 -> weight-bound
combined = USD 2,563.50; separate = USD 2,490.00; delta = USD +73.50
break-even stop charge = USD 376.50
```

exercises.

---

## 8. Data, standards, and regulatory context

### 8.1 United States: gross weight, axle weight, and the bridge formula

Payload and dimension limits exist because bridges, pavements, and braking distances are engineered
to them. A vehicle that exceeds them does not merely risk a fine; it imposes damage that is not
priced into the freight rate. The federal limits on the Interstate System are set in 23 CFR 658.17,
current as at 2026-11-04:

| Limit | Value | Conversion |
|---|---:|---:|
| Gross vehicle weight | 80,000 lb | 36,287 kg |
| Single axle | 20,000 lb | 9,072 kg |
| Tandem group (two or more consecutive axles, more than 40 in and not more than 96 in apart) | 34,000 lb | 15,422 kg |

The gross limit is not the only test. The Federal Bridge Gross Weight Formula caps the weight on any
group of consecutive axles as a function of axle count and spacing:

```text
W = 500 × [ (L · N) ÷ (N − 1)  +  12N  +  36 ]

W = gross weight on the axle group, lb (rounded to the nearest 500 lb)
L = distance between the extreme axles of the group, ft
N = number of axles in the group
```

For a five-axle tractor-semitrailer with 51 ft between the steering axle and the rear tandem:

```text
W = 500 × [ (51 × 5) ÷ 4 + 12 × 5 + 36 ]
  = 500 × [ 63.75 + 60 + 36 ]  =  500 × 159.75
  = 79,875 lb  =  36,231 kg
```

The bridge formula, not the 80,000 lb headline, binds this configuration: it permits 79,875 lb, which
is 125 lb below the gross limit. Shorten the wheelbase and the permitted weight falls further. This
is why a load legal on a long-wheelbase tractor can be illegal on a shorter one at the same gross
weight, and why axle-load distribution is part of load building rather than an afterthought.

States may set higher limits on non-Interstate roads and lower ones on posted bridges, so the route
determines the limit. The federal limits include their tolerances, apart from a portable-scale
allowance of not more than 5%, which is an enforcement provision and not a planning allowance.

### 8.2 European Union: Directive 96/53/EC and its amendments

Directive 96/53/EC sets maximum dimensions and weights for international road transport in the Union.
Figures below are from Annex I, consolidated version dated 14 August 2019, current as at 2026-11-04.

| Parameter | Limit |
|---|---|
| Length, articulated vehicle | 16.50 m |
| Length, road train | 18.75 m |
| Length, motor vehicle (non-bus) | 12.00 m |
| Width, all vehicles | 2.55 m |
| Width, conditioned superstructures and containers | 2.60 m |
| Height, any vehicle | 4.00 m |
| Gross weight, 5- or 6-axle articulated vehicle or road train | 40 t |
| Gross weight, 4-axle combination | 36 t |
| Single non-driving axle | 10 t |
| Driving axle | 11.5 t |

The directive has been amended, and the amendments matter for cube economics. **Directive (EU)
2015/719** (OJ L 115, 6 May 2015; transposition deadline 7 May 2017) introduced the 42 t and 44 t
intermodal allowances for combinations carrying 45-foot containers or swap bodies, a 15 cm length
derogation for aerodynamic devices, and a 1 t allowance for alternatively fuelled vehicles. It also
introduced on-board weighing and co-liability for container weights. **Regulation (EU) 2019/1242**
provides a further allowance for zero-emission heavy vehicles; the increment is category-dependent,
so verify it against the consolidated text rather than applying one number across a fleet.

The economics of the 44 t allowance are worth doing explicitly. Take a 40 t combination with a
15,000 kg tare:

```text
at 40 t:  payload = 40,000 − 15,000 = 25,000 kg
          design density on 100 m³ = 250 kg/m³
at 44 t:  payload = 44,000 − 15,000 = 29,000 kg
          design density on 100 m³ = 290 kg/m³
```

A 10% increase in permitted gross weight raises payload by 16% and moves the crossover density by
40 kg/m³. Cargo between 250 and 290 kg/m³ stops binding on weight and starts binding on cube, which
changes which packing lever is worth pulling. A 2023 Commission proposal (COM(2023) 445) would revise
the directive further; it was not in force as at 2026-11-04 and is not used in any calculation here.

### 8.3 Axle limits bind before gross limits

In both jurisdictions the axle limit frequently binds first. A 40 t combination with a poorly
distributed load can put 12 t on a driving axle limited to 11.5 t while remaining under 40 t gross.
The plan is illegal, the vehicle is not overweight in the aggregate, and a model tracking only total
mass will never see it. Load building therefore includes longitudinal distribution: dense freight
over the drive axles, steer axle within its limit, tandem group checked separately.

### 8.4 Unit loads: pallets and their footprints

Pallet geometry sets the floor pattern, and the floor pattern sets the usable cube above it.
**ISO 6780:2003**, *Flat pallets for intercontinental materials handling — Principal dimensions and
tolerances*, defines six principal footprints:

| Footprint (mm) | Inches (approx.) | Region | Common name |
|---|---|---|---|
| 1200 × 800 | 47.2 × 31.5 | Europe | EUR / EPAL 1 |
| 1200 × 1000 | 47.2 × 39.4 | Europe, UK, Asia | Industrial pallet |
| 1219 × 1016 | 48 × 40 | North America | GMA pallet |
| 1165 × 1165 | 45.9 × 45.9 | Australia | Australian standard |
| 1067 × 1067 | 42 × 42 | North America, Europe, Asia | Square pallet |
| 1100 × 1100 | 43.3 × 43.3 | Asia | T11 |

The standard specifies plan dimensions with a tolerance of +3 mm / −6 mm, so a compliant 1200 mm side
measures between 1194 mm and 1203 mm. It does not specify height, load rating, material, or
phytosanitary treatment; those sit in ISO 8611 for testing and ISPM 15 for wood packaging in export.
There is no ISO 6780 certification, so compliance is self-declared and best verified by measuring a
sample.

The North American 48 × 40 in pallet is 1219 × 1016 mm. It is a trade convention rather than an
actively maintained standard: the Grocery Manufacturers Association, which gave the pallet its name,
no longer maintains a pallet specification. Treat "GMA pallet" as a description of a footprint and a
style, not a certified product.

**Why the footprint matters.** Two 1200 × 1000 pallets placed 1200 mm across occupy 2400 mm of the
DV-53's 2400 mm usable width, leaving zero side clearance. Two 1219 × 1016 pallets placed 1016 mm
across occupy 2032 mm and leave 368 mm; placed 1219 mm across they need 2438 mm and do not fit at
all. The orientation changes the pallet count on the same floor:

```text
1200 mm across, 1000 mm along:  16.00 m ÷ 1.00 m = 16 rows × 2 = 32 pallets
1016 mm across, 1219 mm along:  16.00 m ÷ 1.219 m = 13 rows × 2 = 26 pallets
```

A trailer measured at 2438 mm clear width would admit the turned orientation and therefore 30 of the
48 × 40 pallets; one measured at 2400 mm would not. Thirty-eight millimetres of measured width
changes the pallet count by four, which is why the first instruction in cube work is to measure the
trailer rather than read its specification.

**Overhang and stackability.** A load extending 25 mm beyond the pallet on each side adds 50 mm to
the footprint; two such pallets across need 2450 mm, exceed the 2400 mm usable width, and no longer
fit. Overhang can cost pallet positions outright. Stackability matters as much: usable height is
2.70 m, so a 1.20 m unit load stacked two high uses 2.40 m and achieves 88.9% of the height, while
the same load unstacked uses 1.20 m and achieves 44.4%.

### 8.5 Dimensional weight and density-based classification

Two pricing conventions convert cube into a chargeable quantity, and both matter even though rate
construction belongs to `02_rate-structures.md`. **Air and express** convert volume at a volumetric
divisor, commonly 6,000 cm³/kg (167 kg/m³), with some carriers using 5,000 cm³/kg; chargeable weight
is the greater of actual and volumetric mass. **Parcel** uses a DIM divisor, commonly 139 or 166
in³/lb. **Less-than-truckload** in the United States uses the National Motor Freight Classification,
in which density bands are one input to freight class, so a packaging change that raises density can
move an item to a cheaper class with no rate negotiation at all.

Below the divisor's implied density, a shipment is charged for volume it does not weigh. The implied
density of a 6,000 cm³/kg divisor is 1,000,000 ÷ 6,000 = 166.7 kg/m³, and any lighter cargo is priced
on cube. The pricing system states the same crossover logic developed in section 3.

### 8.6 The data you actually need

A cube analysis needs six item fields and four vehicle fields, and most organisations have neither
set complete.

| Entity | Field | Typical state |
|---|---|---|
| Item | Length, width, height (mm) | Frequently defaulted at creation |
| Item | Gross mass (kg) | Usually present, usually includes packaging |
| Item | Units per pallet and pallet pattern | Often held by the warehouse, not the item master |
| Item | Stackability and maximum stack height (m) | Usually absent |
| Item | Temperature and hazardous classification | Present where regulated, absent otherwise |
| Vehicle | Permitted payload (kg), from plate and route | On the plate, rarely in the TMS |
| Vehicle | Measured usable length, width, height (m) | Rarely measured |
| Vehicle | Deck rating (kg/m²) and door opening (m) | In the body specification, rarely in the system |

The failure is structural rather than careless. Dimensions are captured at item creation by a
merchandising or engineering function with no transport use case, and the field is optional. The
warehouse knows the pallet pattern because it builds the pallets. The carrier knows the vehicle
because it owns it. Nobody owns the join.

---

## 9. Industry practice and case evidence

Public, audited evidence on cube utilisation is scarce. This section grades what exists rather than
dressing vendor claims as findings.

### 9.1 The structural pattern: consolidation is the architecture

General freight networks in every developed market are built around consolidation. An LTL carrier
collects on local pickup routes, consolidates at a terminal, line-hauls full trailers between
terminals, and breaks the freight out for delivery. The consolidation unit is the trailer, and the
network's economics depend on terminal density: enough freight in a small radius to fill trailers
between hubs. This is grade C when described by carriers and grade B when described in regulatory
filings, but the structure itself is observable and not disputed.

Three other forms recur. **Milk runs** in grocery and automotive, where a vehicle visits a fixed
sequence of suppliers or stores on a schedule. **Merge-in-transit** in electronics and appliances,
where components from different origins combine at a merge point so the customer receives one
delivery rather than three. **DC bypass**, where a full truckload runs from supplier to large
customer and skips the distribution centre: consolidation in reverse, because the load is already
full.

### 9.2 The regulatory case: the EU intermodal allowance

The clearest documented case of policy changing cube economics is the 44 t intermodal allowance in
Directive (EU) 2015/719. On the arithmetic in section 8.2, a combination with a 15,000 kg tare gains
4,000 kg of payload, which is 16%, and design density rises from 250 to 290 kg/m³ on a 100 m³ body.
The policy does not subsidise consolidation; it enlarges the box that consolidation can fill. Grade
B, because the instrument and its figures are published and dated.

The US bridge formula is the opposite case: a constraint on the box as a function of geometry rather
than mass. The 79,875 lb result for a 51 ft five-axle combination is derived, not quoted, and it
shows that usable payload depends on the tractor assigned.

### 9.3 The academic case: consolidation policies have known behaviour

The temporal consolidation problem has a mature literature, and its findings are stable enough to
plan against.

- Higginson and Bookbinder (1995) model shipment release timing as a Markov decision process and show
  that the optimal private-carriage policy has a control-limit structure: hold below a threshold,
  dispatch above it. Under common carriage the optimal policy need not share that structure, because
  dispatching early is a price rather than a capacity decision.
- Çetinkaya and Bookbinder (2003) derive closed-form optimal target quantity and cycle length for
  quantity and time policies under private and common carriage, using renewal theory. This is the
  analytical form of the rule derived here: the economic cycle balances dispatch cost against holding
  cost, and the balance depends on whether the vehicle is owned or hired.
- Higginson (1995) distinguishes recurrent policies, re-evaluated on every order arrival, from
  non-recurrent policies using a fixed target. He reports that a recurrent heuristic outperforms a
  fixed economic shipment weight near vehicle capacity, which is the regime most road freight
  operates in.

The practical reading is modest: the time-and-quantity policy is a sound default, the optimal
thresholds depend on parameters most organisations do not measure, and near capacity a simple
recurrent rule beats a sophisticated fixed one.

### 9.4 What the evidence does not support

It does not support a benchmark cube-utilisation figure. Claims of the form "the average trailer is
60% full" are vendor or trade-association estimates with no disclosed method; build the baseline from
your own dispatch data instead. Nor does it support attributing savings to consolidation software,
because every account that can be checked attributes the gain to a process change, with the software
as a recording device.

---

## 10. Failure modes and red flags

**1. Optimising the non-binding dimension.**
The planner builds to the constraint that is easy to see: cube-bound foam loaded until the axles are
at their limit, weight-bound polymer stacked until it reaches the ceiling. Both run at the legal
limit on one dimension and half-empty on the other, and both look right on the dock.
Red flag: cube utilisation above 90% with weight below 30%, or the reverse, on the same lane for more
than four consecutive weeks.

**2. Using gross vehicle weight as the payload.**
The planner reads 80,000 lb from the regulation or the plate and treats it as available capacity,
forgetting the 35,000 lb of tractor and trailer.
Red flag: planned payload between 75% and 100% of GVW rather than of GVW minus tare.

**3. Using nominal body dimensions instead of measured usable volume.**
The body is specified as 110 in high; the usable envelope after lining and wheel boxes is smaller,
so utilisation is computed against a volume freight can never occupy.
Red flag: utilisation figures above 100% on a fully loaded run, which should be impossible.

**4. Ignoring pallet overhang.**
The pattern is computed on the nominal footprint, the load overhangs by 25 mm per side, the pair no
longer fits the usable width, and the trailer carries 26 positions where the plan assumed 32.
Red flag: dock staff routinely re-orienting pallets the plan says should fit.

**5. Treating a non-stackable product as stackable.**
The model assumes two-high stacking because the pallet height allows it; the product cannot bear the
load, so the trailer runs single-stacked at 44.4% of available height.
Red flag: cube utilisation stuck near 45% on a lane whose product is light, with no weight constraint
anywhere near binding.

**6. Comparing the combined price to a modelled cost.**
The analysis shows a large saving because the model prices both separate moves at a full truckload
rate; one of them was actually riding a backhaul at less than half that.
Red flag: a consolidation business case whose baseline is a cost model rather than invoices.

**7. Omitting the stop-off charge.**
The combined price is computed as base plus distance. The second stop adds a tariff charge exceeding
the linehaul saving, and the consolidation loses money.
Red flag: consolidation savings that disappear when the invoice is matched to the business case.

**8. Holding shipments past the customer window to fill the trailer.**
Temporal consolidation applied without a service constraint. The trailer leaves full on Thursday with
freight that was due Wednesday.
Red flag: on-time-in-full falling on exactly the lanes where consolidation was introduced.

**9. Building a cube model on defaulted item dimensions.**
The item master carries one default volume for every item created without measurement, so the model
classifies dense cargo as light and light cargo as dense.
Red flag: a large share of item records sharing one exact dimension or one exact volume.

**10. Double-counting or omitting the pallet's own mass and height.**
The pallet weighs 20 kg and is 150 mm high. One model includes it in mass and not volume, another
the reverse, and neither matches the warehouse.
Red flag: figures that reconcile to the invoice at a consistent offset of a few percent.

**11. Consolidating freight that cannot legally share a vehicle.**
Hazardous goods segregation, temperature regime, and food and non-food separation are constraints the
aggregate fit test cannot see. Two loads that fit perfectly may be prohibited from travelling
together.
Red flag: a consolidation plan that never checks the dangerous goods declaration.

**12. Measuring utilisation at the wrong point in the journey.**
The trailer is 90% full at origin, drops a third of its load at the first stop, and runs the
remaining 400 km at 60%. The average is neither the origin figure nor the final one.
Red flag: one utilisation number per trip with no record of where the drop occurred.

---

## 11. Assumptions and limits

**1. Permitted payload is known from the plate and adjusted for the route.**
What breaks if wrong: the load is built to a limit that does not apply on this road, and the vehicle
is either illegal or under-loaded by tonnes. A single axle limit can bind while gross weight is well
under the cap.

**2. Usable volume is measured rather than read from a specification.**
What breaks if wrong: utilisation is computed against a volume freight cannot reach, and every load
is systematically under-filled or over-planned. The error is biased, so it does not average out
across lanes.

**3. Pallet and unit-load dimensions are as specified, including overhang.**
What breaks if wrong: the pallet count per trailer is wrong and the error propagates into every
downstream cost per pallet. A 25 mm overhang per side loses six positions on DV-53.

**4. Cargo density is stable across the shipment.**
What breaks if wrong: a mixed shipment's average density conceals a dense core and a light shell. The
average says cube-bound; the axle loads say otherwise, and the load is illegal at the rear.

**5. The trip price is fixed and does not vary with the load.**
What breaks if wrong: on a per-kilogram or per-pallet tariff the marginal cost of the last pallet is
not zero, and the fill-the-box logic inverts. Check the tariff structure before applying it.

**6. The separate moves' prices are actually available at decision time.**
What breaks if wrong: the cost test compares against a price that cannot be bought. Capacity is
tight, the LTL carrier has stopped accepting the lane, and the modelled alternative does not exist.

**7. Stop-off charges are known in advance and stable.**
What breaks if wrong: the break-even stop-off is the decision variable, and an unknown charge makes
the decision unknowable. Get the charge quoted, not estimated.

**8. Service windows are hard constraints.**
What breaks if wrong: if a deadline is soft, time fusion is cheaper than modelled because a shipment
can be held. If it is hard and the model treats it as soft, consolidation produces late deliveries
that cost more than the transport saving.

**9. The aggregate fit test is a sufficient screen.**
What breaks if wrong: 3-D packing fails even though mass and volume both fit. Long items, door
openings, and the last row are the usual causes. The aggregate test is necessary, never sufficient.

**10. Demand rate and holding cost are stable over the consolidation cycle.**
What breaks if wrong: a cycle length derived from a stable arrival rate is wrong under lumpy demand,
and the policy oscillates between over-holding and dispatching half-empty.

**11. The regulatory limits quoted here apply to the move being planned.**
What breaks if wrong: national derogations, posted bridge limits, permit regimes, and local
restrictions all override the general figures. Section 8 states the general case as at 2026-11-04;
it is not a substitute for checking the specific route.

**12. The utilisation metric is computed on the same basis for every lane compared.**
What breaks if wrong: lanes measured against different denominators, or at different journey points,
produce a ranking that reflects measurement practice rather than operational performance.

---

## 12. Exercises

**Exercise 1 — Design density across a fleet.**
Using three vehicles you can access (or the five in section 5), compute permitted payload, usable
volume, and design density, stating the basis for each input.
Deliverable: a one-page table with three vehicles, three inputs each, and the derived density in
kg/m³.
Acceptance criteria: every input has a stated source; payload is GVW minus tare, not GVW; usable
volume is measured or labelled nominal; arithmetic checks to one decimal place.
Time: 45 minutes.

**Exercise 2 — Classify ten loads.**
Take ten shipments from a real or supplied dataset and classify each as cube-bound or weight-bound
against DV-53 (design density 196.8 kg/m³).
Deliverable: a table of ten rows with mass, volume, density, and classification.
Acceptance criteria: density to one decimal place; classification consistent with the density test;
any shipment within 10% of the crossover density flagged for manual review.
Time: 40 minutes.

**Exercise 3 — Build a cube-bound load to 95%.**
Starting from load A (30 stacks, 120 kg, 2.88 m³ each), add units until cube utilisation reaches at
least 95% without exceeding payload.
Deliverable: the build, the resulting utilisation pair, and the unused payload in kg and as a
percentage.
Acceptance criteria: cube utilisation between 95% and 100%; weight utilisation correctly reported;
the answer states why floor positions, not volume, are the limit.
Time: 30 minutes.

**Exercise 4 — Run the three tests.**
Take a pair of shipments from your own operation and run time fusion, physical fit, and cost fusion,
stating which test fails first.
Deliverable: a half-page decision note with the three results and the arithmetic for each.
Acceptance criteria: intervals in a single time zone; both fit inequalities shown; cost fusion uses
actual quoted prices and includes every stop-off charge.
Time: 60 minutes.

**Exercise 5 — Find the break-even stop-off.**
For the pair in Exercise 4, solve for the stop-off charge at which the decision reverses.
Deliverable: the break-even value with its date basis, and a one-sentence recommendation to the
carrier.
Acceptance criteria: algebra shown; the answer states whether the current charge is above or below
the threshold and by how much.
Time: 25 minutes.

**Exercise 6 — Packaging change and chargeable volume.**
A packaging change reduces stack height from 2.40 m to 2.16 m on load A's pallet. Recompute volume
per stack, the volume of 30 stacks, and the additional stacks that now fit in the freed volume.
Deliverable: the three figures and the change in cost per m³ at the same trip price.
Acceptance criteria: volume per stack to three decimal places; freed volume divided by the new stack
volume, not the old one; cost change in currency per m³ with the date basis.
Time: 30 minutes.

**Exercise 7 — The utilisation curve.**
Using a fixed trip price from your own data, compute cost per pallet, per kg, and per m³ at load
counts corresponding to 25%, 50%, 75%, and 100% of cube, and identify where cost per pallet doubles.
Deliverable: a four-row table and the doubling point.
Acceptance criteria: the denominator stated for every ratio; the doubling point derived, not
estimated; the table states that the relationship is 1/u, not linear.
Time: 40 minutes.

**Exercise 8 — Derive and apply the dispatch rule.**
Derive the cost-minimising cycle length from C, h, and λ, then apply it with C = USD 95.00 per
dispatch (2026 Q1), h = USD 0.033 per kg per day, and λ = 600 kg per day. Check against a 3,000 kg
vehicle ceiling and a 4-day service limit.
Deliverable: the derivation, T*, Q*, cost per day at T*, and the binding limit.
Acceptance criteria: units checked at each step; an argument that the stationary point is a minimum;
capacity and service checks shown explicitly.
Time: 60 minutes.

**Exercise 9 — Audit a cube dataset.**
Take the item master extract for one product category. Count records with missing dimensions, records
sharing an exact default value, and records whose units-per-pallet is inconsistent with the stated
dimensions.
Deliverable: a one-page audit with counts, the measurement process that would fix each defect, and an
argument for the direction of bias.
Acceptance criteria: every count reproducible from the extract; the bias direction argued rather than
asserted; no percentage quoted without its denominator.
Time: 60 minutes.

**Exercise 10 — The decision memo.**
Write the one-page memo for the section 6 case, addressed to a transport manager who has not read
this topic.
Deliverable: a memo with the recommendation, the three tests, the arithmetic, the break-even
threshold, and the counterfactual.
Acceptance criteria: no more than 400 words; every money figure carries a currency and a date; the
recommendation is unambiguous; a reader could reproduce the arithmetic without the dataset.
Time: 45 minutes.

---

## 13. Self-check answers

**Answer 1.** The table should match section 5 with sources stated. Three checks matter: payload is
GVW minus tare, usable volume is measured, and density is P ÷ V. A worked instance: a 7.5 t rigid
with 3,000 kg payload and 32.0 m³ gives 3,000 ÷ 32.0 = 93.8 kg/m³, binding on weight above that. If
you used 7,500 kg as payload, every density is about 2.5 times too high and every classification is
wrong.

**Answer 2.** Classify on ρ = m ÷ v against 196.8 kg/m³. A shipment of 4,000 kg in 25 m³ has
ρ = 160.0 kg/m³, below the crossover, so it is cube-bound: filling 103.68 m³ would weigh 16,589 kg,
under the 20,400 kg payload. A shipment of 8,000 kg in 25 m³ has ρ = 320.0 kg/m³, above it, so it is
weight-bound: it reaches 20,400 kg in 63.75 m³, leaving 39.93 m³ unused. Shipments within ±10% of
196.8 kg/m³ should be flagged, because a small measurement error flips the recommended lever.

**Answer 3.** Each stack is 2.88 m³ and 120 kg. Reaching 95% of 103.68 m³ needs 98.496 m³, which is
34.2 stacks, so 35 stacks gives 100.80 m³ and 97.2%. But DV-53 has 32 floor positions, so only 32
stacks fit: 92.16 m³ and 88.9%. The binding limit is floor positions, not volume or mass. To reach
95% you must change the unit load — taller stacks, a denser pattern, or double-stacking — because
adding units is impossible. Unused payload at 32 stacks is 20,400 − 3,840 = 16,560 kg, which is 81.2%
of permitted payload.

**Answer 4.** A correct note shows the two feasible intervals and their intersection in a single time
zone, both fit inequalities, and the combined price broken into base, extension, and stop-off against
two actual quoted prices. The most common error is a cost comparison against a modelled baseline
rather than invoices; the second is a time zone error, which reverses the fusion result when
corrected.

**Answer 5.** From C_base + D × r + S = C_A + C_B, the break-even is S* = C_A + C_B − C_base − D × r.
In the section 6 case: 1,850.00 + 640.00 − 1,850.00 − 263.50 = 376.50 USD (2026 Q1). The quoted
USD 450.00 exceeds it by USD 73.50. Recommend asking whether the second stop can be priced at or
below USD 376.50 (2026 Q1), and keeping the moves separate if it cannot.

**Answer 6.** New stack volume is 1.2 × 1.0 × 2.16 = 2.592 m³ and 30 stacks give 77.76 m³ against
86.40 m³, freeing 8.64 m³. Dividing by the new stack volume gives 8.64 ÷ 2.592 = 3.33, so three more
stacks fit, taking the load to 33 stacks and 85.54 m³. Cost per m³ at USD 1,850.00 (2026 Q1) rises
from 1,850 ÷ 86.40 = USD 21.41 to 1,850 ÷ 85.54 = USD 21.63, not falls, because the trip price is
fixed and the load carries slightly less volume in the same positions. The saving appears only when
the freed volume carries more units.

**Answer 7.** Cost per pallet at utilisation u with capacity n_max is P ÷ (u × n_max), so the ratio to
the full-load cost is 1/u. At u = 0.25 the cost is 4 times the full-load cost; at u = 0.50 it is
2 times. The doubling point is u = 0.50. In section 6.8 the full load is USD 57.81 per stack at 32
stacks, and cost first exceeds USD 115.62 at 16 stacks, which is 50% of the floor positions. The
relationship is hyperbolic, which is why the first half of the trailer is cheap to fill.

**Answer 8.** Cost per day is C(T) = C/T + hλT/2. Differentiating gives −C/T² + hλ/2 = 0, so
T* = √(2C ÷ (hλ)); the second derivative 2C/T³ is positive, so the stationary point is a minimum.
With C = 95.00, h = 0.033, λ = 600: T* = √(190 ÷ 19.8) = 3.10 days and Q* = 600 × 3.10 = 1,859 kg.
Cost per day at T* is 30.65 + 30.69 = USD 61.34 (2026 Q1). The vehicle ceiling allows 5.00 days and
the service limit is 4 days, so neither binds. Dispatching at the 5-day ceiling costs
19.00 + 49.50 = USD 68.50 per day (2026 Q1), which is USD 7.16 worse.

**Answer 9.** Report three counts: missing dimensions, records sharing one exact value, and records
where units-per-pallet is inconsistent with stated volume. Bias follows from the default used. A
default of 0.1 m³ per unit understates the volume of bulky goods, so the model treats them as denser
than they are and recommends weight levers on cube-bound loads. A default copied from the last
measured item propagates one product's geometry across a category. Either way the error is
systematic, so it does not cancel across a lane mix.

**Answer 10.** A correct memo states the recommendation (keep separate, because cost fusion fails),
the three results, the arithmetic (2,563.50 against 2,490.00, a difference of 73.50 USD at 2026 Q1),
the USD 376.50 threshold, and the counterfactual. It should note that the combined route saves 420
vehicle-km and still costs more, which is the point a manager is most likely to resist and most needs
to see. A memo recommending consolidation because the loads fit has answered the wrong question.

---

## 14. Interview questions

**1. What is the difference between cube and weight utilisation, and why do you need both?**
Strong answer: they measure different capacities of the same vehicle, and the binding one tells you
which lever is available. A load at 95% cube and 20% weight needs a packaging or stacking change, not
more freight; a load at 60% cube and 98% weight needs a mass reduction.
Weak answer: describes one metric and says the other is also useful, without naming what decision
changes.

**2. How would you compute the crossover density, and what would you do with it?**
Strong answer: divide permitted payload by measured usable volume. For DV-53 that is
20,400 ÷ 103.68 = 196.8 kg/m³, so denser cargo weight-binds as it fills and lighter cargo cube-binds.
Use it to classify shipments before planning and to compare vehicles for a product.
Weak answer: gives a formula but cannot say what it is for, or uses GVW instead of payload.

**3. A trailer leaves at 92% cube and 30% weight. What is wrong, and what would you do?**
Strong answer: nothing is wrong with the vehicle; the product is light and the load is cube-bound.
The lever is density — more units per cubic metre through packaging, stacking, or pallet pattern.
Adding heavy freight is a consolidation decision, not a loading decision.
Weak answer: calls the load inefficient and proposes ballast or balancing, which adds cost without
revenue.

**4. Walk me through the three consolidation tests.**
Strong answer: time fusion, where feasible dispatch intervals intersect; physical fit, where both the
mass and volume inequalities hold plus a geometry check; cost fusion, where the combined price
including stop-offs is below the sum of the displaced prices. They are independent, and passing two
of three is normal.
Weak answer: lists them but treats cost fusion as a distance comparison, which produces consolidations
that save kilometres and lose money.

**5. Why can consolidation save 420 vehicle-km and still cost more?**
Strong answer: because the comparison is between prices, not distances. The displaced move was a
small LTL shipment at USD 640.00 (2026 Q1); the increment was an 85 km extension at USD 263.50 plus a
USD 450.00 stop-off, totalling USD 713.50. The truck burns less fuel and the invoice is higher.
Weak answer: says it cannot happen, or blames accessorials without the arithmetic.

**6. What is the marginal cost of the last pallet on a half-empty trailer?**
Strong answer: zero while there is slack and the price is fixed. The trip runs anyway, so a pallet
that fits adds nothing and the average cost falls as 1/n. Once full, the marginal cost is the price
of an entire additional trip, so the cost curve is a step function.
Weak answer: gives a positive marginal cost throughout, implying a linear model that misprices every
decision at the boundary.

**7. How do you decide how long to hold shipments for consolidation?**
Strong answer: minimise C/T + hλT/2, giving T* = √(2C/(hλ)) and Q* = λT*, then take the minimum of
T*, the vehicle ceiling, and the service limit. Near capacity, a recurrent rule that re-evaluates on
each arrival tends to beat a fixed target.
Weak answer: says hold until the truck is full, ignoring holding cost and service.

**8. Which binds first on a US Interstate move: gross or axle weight?**
Strong answer: often the axle or the bridge formula. A five-axle combination at 51 ft is limited to
79,875 lb by the bridge formula, below the 80,000 lb cap, and poor distribution can exceed the
34,000 lb tandem limit while remaining under gross.
Weak answer: quotes 80,000 lb and stops, treating axle limits as a detail.

**9. How does a packaging change affect transport cost?**
Strong answer: through chargeable volume. Cutting stack height 10% reduces stack volume from 2.88 to
2.592 m³ and frees 8.64 m³ across 30 stacks, which is 3.33 stack-equivalents. Whether that becomes a
saving depends on whether the freed volume carries more units; on a fixed trip price, an empty saving
is not a saving.
Weak answer: asserts a 10% cost saving, confusing a volume reduction with a price reduction.

**10. What would you check first in a cube dataset?**
Strong answer: the distribution of dimension values. Many records at one exact value indicate a
default; missing dimensions indicate the field was optional; inconsistency between dimensions and
units-per-pallet indicates capture by different functions. Then check whether bias is directional,
because a directional error does not cancel across a lane mix.
Weak answer: checks for nulls only and stops when the null rate looks acceptable.

**11. Why is a single load-factor number a bad dashboard metric?**
Strong answer: it hides which constraint binds and therefore which action is available. Two lanes at
60% can need opposite interventions, and it hides the denominator, so a change in measurement
practice looks like a change in performance.
Weak answer: says it is too aggregated without naming the decision it prevents.

**12. How would you prove a consolidation programme worked?**
Strong answer: hold the baseline as actual invoices for the displaced moves, record the combined
price against them, and measure per shipment rather than in aggregate. Control for lane mix and
service level, because consolidation that ships later can look cheaper while failing customers.
Weak answer: reports total freight spend, which moves with volume, rate, and mix and proves nothing.

---

## 15. Key terms

- **Bridge formula** — the US federal formula relating permitted gross weight to axle count and spacing; 23 CFR 658.17.
- **Cargo density** — mass divided by volume for the freight itself, in kg/m³.
- **Chargeable weight** — the greater of actual mass and volumetric mass, used for pricing.
- **Consolidation** — combining shipments into one movement so shared capacity is used once.
- **Cost fusion** — the test that the combined price is below the sum of the displaced prices.
- **Crossover density** — the design density at which mass and volume fill simultaneously.
- **Cross-docking** — moving freight from inbound to outbound vehicles with little or no storage.
- **Cube utilisation** — occupied volume divided by usable volume.
- **Cube-bound load** — a load that reaches its volume limit before its mass limit.
- **DC bypass** — shipping direct from source to customer, skipping the distribution centre.
- **Deck rating** — the maximum distributed load per unit area the trailer floor carries, in kg/m².
- **Design density** — permitted payload divided by usable volume, in kg/m³.
- **Dimensional weight** — volume converted to a mass equivalent by a divisor.
- **Empty running** — kilometres travelled without cargo.
- **Gross vehicle weight** — the mass of tractor, trailer, and payload together.
- **Load factor** — the share of available capacity used, in a stated dimension.
- **Merge-in-transit** — combining shipments from different origins at an intermediate point.
- **Milk run** — a fixed route serving several points on a schedule.
- **Pallet position** — one floor location for a pallet on a vehicle.
- **Payload** — the mass a vehicle may carry, equal to gross vehicle weight minus tare.
- **Physical fit** — the test that combined mass and combined volume are both within limits.
- **Quantity policy** — dispatch when the accumulated load reaches a threshold Q0.
- **Stackability** — whether a unit load can carry another above it.
- **Stop-off charge** — a tariff charge for an additional stop on one trip.
- **Temporal consolidation** — holding shipments to accumulate a fuller load.
- **Time fusion** — the test that the shipments' feasible dispatch intervals intersect.
- **Time-and-quantity policy** — dispatch at whichever of Q0 or T0 is reached first.
- **Unit load** — the pallet, stillage, or roll cage as handled, including the pallet itself.
- **Usable volume** — the enclosed volume actually available to freight, after deductions.
- **Weight utilisation** — payload mass divided by permitted payload mass.

---

## 16. Sources

Grading: A peer-reviewed or audited, B regulatory or official, C company or investor, D vendor,
E press or secondary summary. Each entry states what it supports.

### Grade A — peer-reviewed

**Higginson, J.K. and Bookbinder, J.H. (1995). "Markovian Decision Processes in Shipment
Consolidation." *Transportation Science* 29(3), 242–255. DOI 10.1287/trsc.29.3.242.**
Supports section 9.3: the control-limit structure of optimal release policies under private carriage,
and the observation that common carriage need not share it.

**Çetinkaya, S. and Bookbinder, J.H. (2003). "Stochastic models for the dispatch of consolidated
shipments." *Transportation Research Part B: Methodological* 37(8), 747–768.**
Supports section 9.3 and Exercise 8: closed-form optimal quantity and time policies under private and
common carriage, derived by renewal theory.

**Higginson, J.K. (1995). "Recurrent decision approaches to shipment-release timing in freight
consolidation." *International Journal of Physical Distribution & Logistics Management* 25(5), 3–23.
DOI 10.1108/09600039510089686.**
Supports section 9.3 and section 11 assumption 10: recurrent heuristics outperform fixed economic
shipment weights near vehicle capacity.

**Bookbinder, J.H. and Higginson, J.K. (2002). "Probabilistic modeling of freight consolidation by
private carriage." *Transportation Research Part E: Logistics and Transportation Review*.
DOI 10.1016/S1366-5545(02)00014-5.**
Supports the private-carriage branch of the policy discussion. Volume and page numbers were not
independently confirmed and are omitted rather than guessed.

### Grade B — regulatory and official

**23 CFR 658.17, "Weight" (eCFR, current as at 2026-11-04).** Supports section 8.1: the 80,000 lb
gross limit, the 20,000 lb single-axle limit, the 34,000 lb tandem limit, and the bridge formula
W = 500[(LN/(N−1)) + 12N + 36]. The 79,875 lb result is derived here, not quoted.

**Council Directive 96/53/EC, consolidated version of 14 August 2019 (EUR-Lex, CELEX
01996L0053-20190814).** Supports section 8.2: the 16.50 m, 18.75 m, and 12.00 m length limits, the
2.55 m and 2.60 m width limits, the 4.00 m height limit, the 40 t and 36 t weight limits, and the
10 t and 11.5 t axle limits.

**Directive (EU) 2015/719 (OJ L 115, 6 May 2015).** Supports sections 8.3 and 9.2: the 42 t and 44 t
intermodal allowances, the 15 cm aerodynamic length derogation, the 1 t alternatively fuelled
allowance, and the on-board weighing and co-liability provisions.

**Regulation (EU) 2019/1242.** Supports the statement in section 8.2 that a further allowance exists
for zero-emission heavy vehicles. The specific increment is category-dependent and is deliberately
not quoted; verify it against the consolidated text before use.

**ISO 6780:2003, *Flat pallets for intercontinental materials handling — Principal dimensions and
tolerances*.** Supports section 8.4: the six principal footprints, including 1200 × 800 mm,
1200 × 1000 mm, and 1219 × 1016 mm, and the +3 mm / −6 mm tolerance on plan dimensions.

**FHWA, Commercial Vehicle Size and Weight Program.** Supports section 8.1 on the purpose of the
bridge formula and the relationship between axle count, spacing, and permitted weight.

### Grade C — company and investor

**Carrier stop-off and appointment tariffs quoted for 2026 Q1.** Support the cost figures in
section 6: the USD 450.00 stop-off and the USD 3.10/km extension rate. These are commercial quotes for
one lane and one period, not a market benchmark, and should be replaced with your own quotes before
reuse.

**LTL network descriptions in carrier regulatory filings and network documents.** Support section
9.1's description of hub-and-spoke consolidation as the dominant general-freight architecture. No
savings figures from these sources are used.

### Grade D — vendor

**Pallet supplier dimensional references (for example pallet.design and PalletOne).** Support the
practical description of the 48 × 40 in pallet in section 8.4: 1219 × 1016 mm, stringer construction,
partial four-way entry, and the observation that "GMA" describes a footprint rather than a maintained
specification. Used only for detail corroborated by ISO 6780's listing of the 1219 × 1016 mm
footprint.

### Grade E — press and secondary

**European Transport Safety Council, briefing on weights and dimensions (November 2023).** Supports
the summary of the pending revision proposal in section 8.2. The proposal itself (COM(2023) 445) is
the primary document; the briefing is used only to confirm that it was not in force as at 2026-11-04.

### What was checked and what was not

Checked: the ISO 6780 footprint list and tolerance; the 48 × 40 in conversion to 1219 × 1016 mm; the
US gross, single-axle, and tandem limits and the bridge formula text; the EU dimensions, weights, and
axle limits; the existence and content of Directive (EU) 2015/719; and the three consolidation-policy
citations by title, journal, volume, and page.

Not independently confirmed: the page range of the 2002 *Transportation Research Part E* paper, cited
by DOI only, and the precise zero-emission increment under Regulation (EU) 2019/1242. No
industry-average cube-utilisation figure is used anywhere in this file.

---

## 17. Links

**Within this module**

- Previous topic: [`01_freight-modes.md`](01_freight-modes.md) — mode cost architecture, the layer
  above this one.
- Rate construction and accessorial mechanics: [`02_rate-structures.md`](02_rate-structures.md) — how
  the prices used in the cost fusion test are built.
- Lane and tender analytics: `03_lane-tender-analytics.md` (planned) — segmentation and carrier mix,
  which determine whether the displaced price is a contract rate or a spot rate.
- Vehicle routing: `05_vrp-formulation.md` (planned) — route sequencing, which takes the consolidated
  load as an input.
- Module index: [`_index.md`](_index.md) — topic list, prerequisites, and status.
- Part II index: [`../_index.md`](../_index.md) — position of this module in the programme.

**Adjacent modules**

- Facility location and network design:
  [`../11_network_design_optimization/04_facility-location.md`](../11_network_design_optimization/04_facility-location.md)
  — where the consolidation points are, which sets the lanes this topic optimises.
- Economic order quantity and its limits:
  [`../10_inventory_optimization_multi_echelon/02_eoq-limits.md`](../10_inventory_optimization_multi_echelon/02_eoq-limits.md)
  — the same square-root structure as the dispatch rule, applied to order quantity rather than
  dispatch timing.

**Course material**

- Survey file:
  [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
  — section 4.5 is the origin of the three-test framing used here.

Read `01_freight-modes.md` for the cost architecture, this topic for the physical envelope and the
three tests, then `02_rate-structures.md` for how the prices are constructed. Routing comes after,
because a route plan over a badly consolidated load optimises the wrong object.
