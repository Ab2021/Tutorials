# Container Loading: Which Constraint Binds?

> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 02 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01
> Version 2026.5

---

## 1. Why this topic exists

A load plan answers a question with four parts, and most plans answer one. A planner looks up "a
40-foot takes 22 pallets", books the box, and discovers at the dock that the pallets are 1,200 by
1,000 millimetres rather than 1,200 by 800, or that the stack is 2.4 metres and the door is 2.28.
The arithmetic that would have prevented each surprise is three lines long. Nobody runs it,
because the number they needed was already in a spreadsheet, rounded, undated, and for a
different pallet.

Three facts organise what follows, and each is counter-intuitive enough that competent people get
it wrong.

**A 40-foot container does not carry twice the mass of a 20-foot container. It carries slightly
less.** Both are rated at the same nominal maximum gross mass, commonly 30,480 kg (67,200 lb),
because that figure is set by the standard and by handling equipment rather than by volume. The
longer box has more steel, so its tare is higher and the payload left for cargo is lower. Carrier
specifications returned in this research give about 28,200 kg of payload for a 20-foot
general-purpose box and 26,700 kg for a 40-foot box of the same rating (Grade C, carrier-published,
2026). Doubling the length roughly doubles the cube and slightly reduces the mass allowed in it.

**A container loaded to its rating is not legal on a road.** United States federal limits on the
Interstate System cap a five-axle tractor and chassis at 80,000 lb (36,287 kg) gross, with 20,000 lb
(9,072 kg) on a single axle and 34,000 lb (15,422 kg) on a tandem (23 U.S.C. §127; 23 CFR 658.17;
Grade B). A 40-foot box with a 3,900 kg tare on a 3,600 kg chassis behind an 8,200 kg tractor
leaves about 20,600 kg of cargo before the combination reaches the cap, against a container
payload of 26,460 kg. The container is not overloaded. The vehicle is.

**Filling the floor is not filling the box.** Twenty pallets of cartons in a 40-foot high cube
occupy about 68% of the internal volume and 60% of the payload. The binding constraint is neither.
It is pallet positions on the floor multiplied by the tiers the stacked height allows. The planner
who checked weight and volume checked the two constraints that were not binding.

> **Note:** payload figures are carrier-published and vary by manufacturer, age, and configuration.
> Treat them as the shape of the problem, not as constants. Section 5.1 shows how to derive them
> from internal dimensions so the conclusion survives a different box.

**The general rule.** The binding constraint is not a property of the container. It is a property
of the pair: this product, in this box. Change the product's density and a different constraint
binds. Change the box and a different constraint binds for the same product. The only way to know
which is to compute all four and take the minimum — and the only way to improve the answer is to
know which one you took the minimum of.

Topic 01A.01 established that a quantity is a number plus a unit, that conversions have integrity
classes, and that mass and volume are coupled by density. This topic takes the top of that
hierarchy, the container, and asks what the coupling does inside a steel box with a door. Topic
01A.03 takes the next question: once you know how many containers, what integer rounding to cases
and minimum order quantities does to the plan.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** the four capacities of a named container type — internal volume, floor positions,
   payload, and door clearance — from its internal dimensions, each in its correct unit.
2. **Explain** why a nominal capacity on a specification sheet must never enter arithmetic, and
   **demonstrate** the error against the product of the internal dimensions.
3. **Compute** the crossover density and crossover stowage factor for any box, and **use** them to
   classify cargo as weight-bound or cube-bound before any loading plan exists.
4. **Predict**, for a given product and box, which of the four constraints binds, and **justify**
   the prediction with arithmetic rather than experience.
5. **Lay out** a palletised floor plan for a 20-foot and a 40-foot box, **quantify** broken stowage
   and ullage separately, and **contrast** it with a slip-sheet or hand-stacked plan.
6. **Check** a load against floor line load, point load, centre of gravity, axle load, and gross
   vehicle weight, naming the jurisdiction whose limit applies.
7. **Build** a verified gross mass declaration under both permitted methods, and **explain** what
   happens to a container that reaches the terminal without one.
8. **Convert** a container decision into cost per unit and cost per cubic metre, and **identify**
   the point at which a packaging or mode change beats adding another container.

## 3. The core mental model

A container is a box with four numbers that matter: how much space it has, how much floor it has,
how much mass it may carry, and how large an opening you must pass through to reach the space. A
product is a set of units with a mass, a volume, and a shape. Load the box until one number runs
out. Which one runs out first is the answer to the question in the title.

### 3.1 Three continuous constraints and one gate

```text
   ONE 40 ft HIGH CUBE, PRODUCT B (320 kg/m3, 0.030 m3 per case, 9.6 kg gross)

   max cases = min(  2,756  ,  2,545  ,  1,600  )   subject to  door: pass
                           |         |         |
                           |         |         +---- FLOOR x TIERS
                           |         |                20 positions x 80 cases
                           |         |                = 1,600 cases   <-- BINDS
                           |         |
                           |         +-------------- CUBE
                           |                          76.35 m3 / 0.030 m3
                           |                          = 2,545 cases (nominal)
                           |
                           +------------------------ PAYLOAD
                                                      26,460 kg / 9.6 kg
                                                      = 2,756 cases

   door gate: case 0.300 x 0.400 m against opening 2.585 x 2.340 m -> passes.
   The case also passes the internal height of 2.698 m. Gate open.
```

Read it as an evaluation order, not a ranking. The gate is binary: if the unit does not pass, the
answer is zero however much capacity remains. The three continuous constraints are computed from
the same box and the same unit, and the minimum is the load. Here the answer is 1,600 cases, and
the number that produced it — 20 floor positions times 8 tiers — appears on no specification sheet
anywhere.

Stated formally for a unit with gross mass `m` (kg), envelope volume `v` (m³), and footprint `a`
(m²):

| Constraint | Maximum units | Property of |
|---|---|---|
| Payload | `floor(payload / m)` | Box rating and tare, unit mass |
| Cube | `floor(usable_volume / v)` | Box internal volume, unit envelope, stowage loss |
| Floor times tiers | `positions x tiers` | Box floor plan and height, unit footprint and stackability |
| Door and opening | pass or fail | Box opening, unit's largest dimension |

**The floor constraint is not floor area.** Dividing area by footprint gives positions per tier.
The tier count comes from internal height divided by stacked height, and from whether the cargo
bears the tiers above it. A 40-foot high cube has 28.3 m² of floor and 2.698 m of height; a pallet
build 2.15 m tall uses one tier and wastes the rest. That is why "floor space" is shorthand for a
constraint that couples floor to height, and why treating area alone as the limit overstates the
load whenever the stack does not divide evenly into the height.

**The door is a gate, not a constraint.** Door openings are smaller than internal sections, and
not uniformly: internal height is about 2.698 m and door height about 2.585 m, internal width
2.352 m and door about 2.340 m (Grade C, carrier specification, 2026). A unit 2.6 m tall fits inside
and does not fit through the door, and a load planned against the internal width is rejected by 12
millimetres. A box can also be loaded and not unloaded, because the destination dock is lower. The
gate is evaluated in both directions.

### 3.2 The crossover density

Density classifies cargo because it decides whether mass or space runs out first. Take density as
gross mass per unit of occupied volume in kilograms per cubic metre, and its reciprocal, the
stowage factor, as cubic metres per tonne.

```text
   crossover density        = payload / usable volume      [kg/m3]
   crossover stowage factor = usable volume / payload       [m3/t]
```

Cargo denser than the crossover is weight-bound: the box reaches payload before it fills. Cargo
lighter is cube-bound: the box fills first. The crossover is a property of the box and varies by
more than a factor of three across the common types.

The practical rule falls out: **dense cargo wants the smallest box, light cargo the largest.** The
stowage-factor ladder runs from 0.98 m³/t for a 20-foot reefer to 3.13 m³/t for a 45-foot high cube.
That is the opposite of "a bigger box is always better", and it is why a chemical shipper and a
snack shipper should not be negotiating the same equipment. The figures are derived in section 5.1
and are not portable between carriers: a different tare changes the payload and a different
internal height changes the volume.

> **Note:** a crossover computed on nominal internal volume is optimistic. Real loading leaves
> space around pallets, above a load that stopped short of the roof, and behind the door recess.
> Apply a stowage loss and state it. A 10% loss moves the 40-foot high cube crossover from 347 to
> 385 kg/m³, which is the difference between calling a 360 kg/m³ cargo light and calling it heavy.

### 3.3 Why the bigger box is the lighter box

The relationship is structural, and it is worth a sentence you can defend in a meeting: **payload
per cubic metre falls as the box grows, because the rating is capped and the tare grows.** The
rating is a round number fixed by the standard and by the equipment that lifts the box; it does
not scale with length. The tare does. So payload falls slightly while volume grows in proportion
to length, and the ratio falls by roughly the same factor that volume rises.

Two consequences follow, and both are operational.

- **A dense product in a large box pays for air.** Freight is priced per container, so cost per
  tonne of a weight-bound cargo rises with box size; the move is a smaller box.
- **A light product in a small box pays for steel.** Tare is a fixed cost and a small box carries
  little volume; the move is a larger box, limited by the road rather than the ship.

### 3.4 Broken stowage is geometry, not waste

Two quantities are usually lumped together as "waste". Keep them apart.

- **Broken stowage** is space that cannot be occupied because of cargo shape, packaging, or
  securing. A pallet footprint does not divide evenly into the floor; a round drum leaves a
  triangular gap; a lashing lane is unavailable even if you had more cargo.
- **Ullage** is space left unoccupied because the cargo ran out. The load stopped at 2.15 m in a
  2.698 m box. It is available, unused, and cheaper to fix, because filling it needs no change to
  the cargo's shape.

Both reduce cube utilisation. Only the first belongs in a stowage factor. Neither appears in a
nominal capacity figure.

```text
   40 ft HC floor, 1,200 x 1,000 mm pallets, straight pattern
   +------------------------------------------------------+  2.352 m internal width
   | +------+ +------+ +------+ +------+ +------+ +---   |
   | |      | |      | |      | |      | |      | |       |  10 positions along
   | +------+ +------+ +------+ +------+ +------+ +---    |  x 1.200 m = 12.000 m
   | +------+ +------+ +------+ +------+ +------+ +---    |
   | |      | |      | |      | |      | |      | |       |  2 positions across
   | +------+ +------+ +------+ +------+ +------+ +---    |  x 1.000 m =  2.000 m
   +------------------------------------------------------+
    12.032 m usable length                 0.352 m across, unusable

   20 pallets occupy 24.00 m2 of 28.30 m2 = 84.8% of the floor
   broken stowage on the floor alone = 15.2%
```

The same box from the side, for the same palletised product:

```text
   section through the height, 40 ft HC
   +------------------------------------------+ 2.698 m internal height
   |                                          |
   |            0.548 m ullage                |
   |                                          |
   +------------------------------------------+ 2.150 m top of load
   |  8 layers of cases, 8 x 0.250 m = 2.000  |
   |                                          |
   +------------------------------------------+ 0.150 m pallet
   |__________________________________________|

   cube utilisation = 20 x (1.200 x 1.000 x 2.150) / 76.35 = 51.6 / 76.35 = 67.6%
```

The floor plan wastes 15.2% and the height 20.3%; the two multiply into the 32.4% of the box that
is not cargo. A planner who computed "20 pallets fit, each 2.58 m³, so 51.6 m³ of 76.4" has the
volume right and no idea where the loss came from. The fix for height is a taller build or a
hand-stacked top layer; the fix for floor is a different pallet or a mixed-orientation pattern.
Different fixes, separated by the arithmetic.

> **Note:** trade stowage losses run from about 5% in textbook examples to about 18% for awkward
> cargo. The Container Handbook uses 10% and 15% in its own worked examples and notes that
> practitioners estimate the figure from experience and records rather than from a formula (Grade
> C, industry reference, undated on the retrieved page). There is no universal number: state yours
> and show the floor plan that produced it.

### 3.5 The road is a fifth constraint, outside the box

The four constraints above belong to the container. The fifth belongs to the vehicle under it and
the jurisdiction it crosses, and it can bind below all four.

Three caps apply at once: 20,000 lb on a steer axle, 34,000 lb on each tandem, and 80,000 lb
(36,287 kg) gross combination weight on the US Interstate System, against 30,480 kg on the CSC
plate. A container can be under its rating, under the gross vehicle limit, and still illegal
because one axle is over its limit or because the axle spacing is too short for the bridge formula. The load's centre of gravity decides how the gross mass splits
between tractor and trailer: 20 tonnes concentrated in the first six metres loads the axles
differently from the same 20 tonnes spread evenly, and both are legal on the water.

The instruction is therefore: compute the four container capacities, ask which vehicle and which
jurisdiction, then take the minimum again with the road numbers included. Section 6.6 does the
arithmetic.

## 4. Definitions and vocabulary

The fourth column states the unit or coding system in which the term is expressed, because a
capacity quoted without a unit is the defect this topic exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Container type code | Four-character ISO designation of size and type, such as 22G1 or 45G1 | Treated as the ISO 668 dimension code | ISO 6346 code |
| ISO 668 dimension code | Standard external-size designation: 1CC, 1AA, 1AAA, 1EEE | Used as a commercial product name | ISO 668 code |
| TEU | Twenty-foot equivalent unit, a capacity measure based on the 20 ft box | Counted as a number of physical containers | TEU |
| FEU | Forty-foot equivalent unit, defined as exactly two TEU | Assumed to carry twice the payload of a 20 ft box | TEU |
| Nominal capacity | The rounded volume or payload printed on a specification sheet | Used in arithmetic | m³ or kg |
| Internal volume | Product of internal length, width, and height | Assumed equal to the nominal capacity | m³ |
| Floor area | Internal length multiplied by internal width | Used as the load limit without a tier count | m² |
| Payload | Maximum cargo mass, equal to maximum gross mass minus tare | Confused with maximum gross mass | kg |
| Tare mass | Mass of the empty container | Omitted from the verified gross mass | kg |
| Maximum gross mass | Maximum permissible mass of container plus cargo, marked on the safety approval plate | Read as the payload | kg |
| VGM | Verified gross mass, the declared mass of the packed and sealed container | Used for any weight figure | kg |
| SOLAS | International Convention for the Safety of Life at Sea, whose Chapter VI governs VGM | Treated as a national rule | Convention |
| Method 1 | Weighing the packed and sealed container on calibrated equipment | Assumed to require prior approval | Procedure |
| Method 2 | Summing gross package masses plus dunnage and securing plus tare, under an approved method | Any sum of item-master weights | Procedure |
| VGM cut-off | Deadline by which the verified mass must reach carrier and terminal | Assumed to be set by SOLAS | Clock time |
| Stowage factor | Volume occupied per unit mass of cargo | Confused with density, its reciprocal | m³ per tonne |
| Cargo density | Gross mass per unit of occupied volume | Confused with stowage factor | kg/m³ |
| Crossover density | Payload divided by usable volume; where weight and cube bind together | Treated as a fixed property of the container type | kg/m³ |
| Binding constraint | The first of the four capacities to run out, and therefore the load | Reported as "the limit" without saying of what | Units, or a named capacity |
| Door gate | The binary test of whether a unit passes the opening | Assumed equivalent to internal dimensions | Pass or fail |
| Broken stowage | Space that cannot be occupied because of cargo shape, packaging, or securing | Called "waste" or "rounding" | Percentage of volume |
| Ullage | Space left unoccupied because the cargo ran out | Used as a synonym for broken stowage | m³ or percentage |
| Cube utilisation | Occupied cargo volume divided by internal volume | Computed on nominal capacity | Percentage |
| Weight utilisation | Cargo mass divided by payload | Computed on maximum gross mass | Percentage |
| Floor positions | Units per tier from per-axis division | Estimated by dividing floor area by footprint | Count |
| Tier count | Stacked layers the internal height and cargo strength allow | Taken as internal height divided by unit height | Count |
| Floor load limit | Permissible distributed load on the container floor | Ignored because the payload was checked | kg/m² |
| Line load | Cargo mass per running metre of cargo length | Confused with point load | t/m |
| Point load | Load through a small contact patch, such as a forklift wheel | Assumed covered by the payload rating | kg per contact area |
| Axle load | Mass carried by one axle | Ignored when gross vehicle weight is legal | kg |
| GVW / GCW | Gross vehicle weight; gross combination weight of tractor, chassis, and box | Conflated with container maximum gross mass | kg |
| Bridge formula | Rule relating axle spacing to permissible mass | Assumed obsolete | Formula, jurisdiction-specific |
| Chassis | Wheeled frame on which a container travels by road | Assumed to be part of the container | Equipment unit |
| Kingpin | Tractor–trailer coupling that sets the effective wheelbase | Ignored in load distribution | Position, mm |
| High cube | A box 2,896 mm high externally rather than 2,591 mm | Assumed to carry more payload | Container type |
| Open top / flat rack | Boxes with removable roof bows, or with no roof and no sides | Treated as boxes for cube arithmetic | Container type |
| Reefer | Temperature-controlled box with insulated walls and reduced internal dimensions | Assumed to match a dry box | Container type |
| Pallet footprint / overhang | Plan dimensions of a loaded pallet, including load past the edge | Taken as the nominal pallet size | mm by mm |
| Slip sheet | Thin sheet replacing a pallet for handling | Assumed equivalent in protection | Handling unit |
| Dunnage | Material filling gaps, spreading load, bracing cargo | Omitted from the verified gross mass | kg or m³ |
| Centre of gravity | Mass centre of the loaded container | Assumed to be the geometric centre | m from a datum |
| Dead freight | Freight charged for booked space not used | Confused with demurrage | Currency |
| Chargeable weight | Greater of actual and volumetric mass at a stated factor | Applied to full-container-load ocean freight | kg |

Four rows carry most of the damage.

**Nominal capacity versus internal volume.** A carrier sheet returned in this research states a
40-foot high cube reefer at 67.5 m³ while listing internal dimensions of 11.580 by 2.275 by 2.415
m, which multiply to 63.62 m³ — a 6% gap the sheet does not explain (Grade C, 2026). The dimensions
may be measured to a load line, or the volume may be computed from another set of figures, or one
number may be stale. Use the dimensions: they are reproducible, and the discrepancy can be logged
rather than averaged.

**Stowage factor versus density.** Reciprocals, with units a factor of a thousand apart. "Density
2.5" is meaningless: 2.5 m³/t is a light cargo and 2.5 t/m³ a dense one.

## 5. The mechanics, step by step

Fourteen steps in five phases. Each names the arithmetic, the inputs, and the failure when skipped.

### Phase A — Establish the four capacities

**Step 1. Take internal dimensions, not nominal ones.** Three numbers per type, from the
carrier's specification for the specific unit. ISO 668:2020 fixes external length at 6,058 mm for
a 20-foot box and 12,192 mm for a 40-foot, external height at 2,591 mm and 2,896 mm, a minimum
internal width of 2,330 mm, and minimum internal height as external height minus 241 mm (Grade B,
standard preview). Commercial figures are typical, not required, and the gap between the standard's
minimum and a carrier's actual is where a load plan quietly fails.

**Step 2. Compute the three continuous maxima.** `payload / unit mass`, `usable volume / unit
volume`, and `positions per tier x tiers`. Skipping this means estimating from one constraint, in
the direction that costs money.

**Step 3. Test the door gate in both directions.** Unit cross-section against the opening, and
unit height against the internal height at the door end. Include the forklift mast if the unit is
driven in. Skipping it rejects a load at the door, or loads one that cannot be unloaded.

**Step 4. Record which constraint binds, and the margin.** No arithmetic beyond a subtraction, but
the margin matters as much as the minimum. A load binding at 1,600 against a payload limit of 2,756
has 42% of its capacity idle. One binding at 1,600 against 1,640 has no tolerance for a pack-data
error. Same answer, opposite risk.

### Phase B — Classify the cargo

**Step 5. Compute density on a gross basis.** `gross unit mass / envelope volume`, in kg/m³;
stowage factor is the reciprocal. Gross means including packaging and pallet; envelope means the
space the unit occupies, not the space its contents occupy. A drum with 240 kg of liquid and 18 kg
of steel has a net density of 1,200 kg/m³ and an envelope density near 857 kg/m³, because the
square around its circle is larger than the circle.

**Step 6. Compare with the crossover.** One division or a lookup from 5.1. Skipping it means the
box is chosen by habit, and habit is organised around the trade lane rather than the product.

**Step 7. Choose the box family before building the load.** For each candidate, containers equal
`ceil(demand / units_per_container)`; multiply by the rate. Skipping it optimises a floor plan
inside a box that should never have been booked.

### 5.1 The derived capacity table

Derived, not copied, from carrier-published internal dimensions and payloads. Every figure can be
reproduced from the two columns to its left.

| Type | Internal L x W x H (m) | Cube (m³) | Floor (m²) | Payload (kg) | Crossover (kg/m³) | Crossover SF (m³/t) |
|---|---|---:|---:|---:|---:|---:|
| 20 ft general purpose | 5.898 x 2.352 x 2.393 | 33.20 | 13.87 | 28,200 | 849 | 1.18 |
| 40 ft general purpose | 12.032 x 2.352 x 2.395 | 67.78 | 28.30 | 26,700 | 394 | 2.54 |
| 40 ft high cube | 12.032 x 2.352 x 2.698 | 76.35 | 28.30 | 26,460 | 347 | 2.89 |
| 45 ft high cube | 13.556 x 2.352 x 2.698 | 86.02 | 31.88 | 27,450 | 319 | 3.13 |
| 20 ft refrigerated | 5.451 x 2.290 x 2.167 | 27.05 | 12.48 | 27,490 | 1,016 | 0.98 |
| 40 ft high cube refrigerated | 11.580 x 2.275 x 2.415 | 63.62 | 26.34 | 29,480 | 463 | 2.16 |

Two readings matter. **The 20-foot box is the densest carrier in the fleet and the 45-foot high
cube the lightest**, spanning 319 to 1,016 kg/m³. **The 40-foot reefer carries more payload and
less volume than the dry high cube**, so its crossover of 463 kg/m³ sits between the two dry
40-footers, which is not where a planner reasoning "reefers are smaller" would put it.

> **Note:** the payload column is the weakest number here. It is carrier-published, varies by
> manufacturer and age, and describes a type rather than a unit. Where a decision sits close to a
> crossover, obtain the tare of the actual container from the CSC plate.

### 5.2 The worked crossover

```text
40 ft HC:  12.032 x 2.352 x 2.698 = 76.35 m3
           26,460 / 76.35            = 346.6 -> 347 kg/m3   (SF 2.886 m3/t)
           26,460 / (76.35 x 0.90)   = 385.1 kg/m3          (10% loss)

20 ft GP:  5.898 x 2.352 x 2.393  = 33.20 m3
           28,200 / 33.20            = 849.4 -> 849 kg/m3   (SF 1.177 m3/t)
           28,200 / (33.20 x 0.90)   = 943.8 kg/m3          (10% loss)
```

The ratio between the two crossovers is 2.45. That is the quantitative form of "dense cargo wants
a small box": per cubic metre of cargo space, the 20-foot box makes available 2.45 times as much
payload as the 40-foot high cube.

### Phase C — Build the floor plan

**Step 8. Lay out positions per tier on a scaled drawing.** For each orientation,
`floor(length / unit length) x floor(width / unit width)`; test both, then any mixed pattern.
Area division is the classic error: a 1.2 m² pallet into 28.30 m² suggests 23, and the straight
grid fits 20. The 15% difference is all remainder — 0.032 m along the length, 0.352 m across.

**Step 9. Multiply by the achievable tier count.** `floor(internal height / stacked height)`, then
reduce by one if the top tier cannot be braced or the cargo cannot bear it. Skipping this assumes
the load goes to the roof; the last tier arrives crushed or is never built.

**Step 10. Quantify broken stowage and ullage separately.**
`broken stowage = 1 - (positions x footprint / floor area)`;
`ullage = 1 - (load height / internal height)`; cube utilisation is the product of the two
remainders. Without the split the remedy cannot be chosen.

### 5.3 The palletised plan against the alternatives

Product B: a case 400 by 300 by 250 mm, 9.6 kg, 0.030 m³, built 10 per layer and 8 layers on a
1,200 by 1,000 mm pallet with a 150 mm pallet, giving a 2.150 m load of 793 kg.

```text
   PALLETISED, 40 ft HC
     floor positions   10 along x 2 across = 20
     tiers             2.698 / 2.150 = 1.25 -> 1
     cases per box     20 x 80 = 1,600
     cube used         20 x (1.200 x 1.000 x 2.150) = 51.60 m3 of 76.35 = 67.6%
     weight used       15,860 kg of 26,460 = 59.9%

   SLIP SHEET, same footprint and pattern, 10 layers instead of 8
     cases per box     20 x 100 = 2,000
     cube used         20 x (1.200 x 1.000 x 2.500) = 60.00 m3 of 76.35 = 78.6%
     gain over pallet  +25.0%

   HAND STACKED, floor filled to the internal height
     cases per layer   30 along x 7 across = 210
     layers            2.698 / 0.250 = 10.79 -> 10
     cases per box     2,100
     cube used         2,100 x 0.030 = 63.00 m3 of 76.35 = 82.5%
     gain over pallet  +31.3%
```

Three plans, one box, a 31% spread. The palletised plan is limited by the pallet's shape, not by
the container. The hand-stacked plan recovers the 15.2% floor loss and most of the 20.3% height
loss, and pays for both in labour at origin and destination, in damage, and in the customer's
willingness to receive a floor-loaded box. Whether the 500 extra cases are worth that is a
commercial question; whether they exist is arithmetic, and the arithmetic takes ten minutes.

The slip-sheet number is the interesting one, because it is usually available without changing how
the customer receives: the load stays a unit, a clamp truck moves it, and the pallet's 150 mm and
its footprint waste both disappear. It is not free — push-pull attachments are needed at both ends.

### Phase D — Check the external constraints

**Step 11. Distribute the load within the box.** Line load as mass per running metre of cargo
length, point load at each contact patch, and the longitudinal centre of gravity. A 20,000 kg load
over four metres is 5.0 t/m against a 40-foot limit near 3.0 t/m, and fails even with 6.5 t of
payload margin. Spread over seven metres it is 2.86 t/m and passes.

**Step 12. Check the chassis and the road.** `GCW = tractor + chassis + tare + cargo`, against the
jurisdiction's gross and axle limits and, where it applies, the bridge formula. Skipping it makes
the shipment legal on the water and illegal on the last mile.

**Step 13. Verify and declare the mass.** Either weigh the packed box, or sum gross package masses
plus dunnage plus tare under an approved method. Section 8.3 gives the rule and the consequence.

### Phase E — Price the decision

**Step 14. Convert to cost per unit and per cubic metre.** `containers x rate / units shipped`,
and `containers x rate / cubic metres shipped`. The two rank the same options differently and both
are correct; which to use depends on whether mass or volume is binding.

## 6. Worked example

One contract manufacturer, three products, two box types. The products span the density range and
every figure is reproducible from the one before it. Demand is 300 tonnes per product per quarter,
so the three share a denominator and the comparison is about density rather than volume.

### 6.1 The three products

| Attribute | A: syrup in drums | B: cartons on pallets | C: light bulky cases |
|---|---|---|---|
| Unit | 200 L drum, 585 mm dia x 880 mm high | Case 400 x 300 x 250 mm | Case 600 x 400 x 300 mm |
| Net / gross mass | 240 kg / 258 kg | 9.0 kg / 9.6 kg | 4.0 kg / 4.2 kg |
| Envelope volume | 0.3011 m³ (square envelope) | 0.0300 m³ | 0.0720 m³ |
| Density, gross envelope | 857 kg/m³ | 320 kg/m³ | 58.3 kg/m³ |
| Stowage factor | 1.167 m³/t | 3.125 m³/t | 17.14 m³/t |
| Unit load | none, floor loaded | 1,200 x 1,000 mm pallet, 80 cases, 2.150 m, 793 kg | none, floor stacked |
| Classification | heavy | mid | light |

The drum's envelope is the square that circumscribes its circle, 0.585 m by 0.585 m by 0.880 m. A
round unit occupies the square around it in a grid, and pretending otherwise is the first error in
any drum plan. The envelope density of 857 kg/m³ is 29% below the liquid's own 1,200 kg/m³, and it
is the envelope density that decides what fits.

### 6.2 The four maxima for each product and box

| Product | Box | Payload limit | Cube limit | Floor x tiers | Binding | Units per box |
|---|---|---:|---:|---:|---|---:|
| A | 20 ft GP | 109 drums | 110 drums | 40 x 2 = 80 | Floor x tiers | 80 |
| A | 40 ft HC | 103 drums | 254 drums | 80 x 3 = 240 | Payload | 102 |
| B | 20 ft GP | 35 pallets | 111 pallets | 4 x 2 x 1 = 8 | Floor x tiers | 8 pallets (640 cases) |
| B | 40 ft HC | 33 pallets | 99 pallets | 10 x 2 x 1 = 20 | Floor x tiers | 20 pallets (1,600 cases) |
| C | 20 ft GP | 6,714 cases | 461 cases | 45 x 7 = 315 | Floor x tiers | 315 |
| C | 40 ft HC | 6,300 cases | 1,060 cases | 100 x 8 = 800 | Floor x tiers | 800 |

Read the A rows together, because they are the point of the topic. The same drum in a 20-foot box
is limited by geometry at 80 units, using 72.6% of the cube and 73.2% of the payload; in a 40-foot
high cube it is limited by payload at 102 units, using 40.2% of the cube. The smaller box is better
for this cargo because it wastes less air, and the larger is better only if the cargo is light
enough to fill it.

The B rows show the second lesson: both boxes are limited by pallet positions, and the pallet
build, not the container, sets the answer. The C rows show the third: the cube limits of 461 and
1,060 are never approached, because geometry runs out first, and a planner checking only the cube
would be 25% optimistic.

### 6.3 Containers required at 300 tonnes per quarter

| Product | Units at 300 t | 20 ft GP | 40 ft HC | Cheaper box | Why |
|---|---:|---:|---:|---|---|
| A | 1,163 drums | 15 | 12 | 20 ft | Dense: mass per box capped by payload, so the small box wastes less |
| B | 31,250 cases | 49 | 20 | 40 ft HC | Mid: pallet count per box dominates, and the big box holds 2.5 times as many |
| C | 71,429 cases | 227 | 90 | 40 ft HC | Light: volume is what is being bought, and only the big box has enough |

The counts are ceilings of the exact quotient: `1,163 / 80 = 14.54`, so 15; `31,250 / 1,600 =
19.53`, so 20. Rounding is the subject of Topic 01A.03; here it is only the mechanism that turns a
rate per container into a cost per unit.

### 6.4 Freight cost at €1,900 per 20 ft and €2,600 per 40 ft HC

Rates are illustrative and exclude terminal handling, inland transport, and duty.

| Product and box | Containers | Freight | Per tonne | Per unit | Per m³ of cargo |
|---|---:|---:|---:|---:|---:|
| A, 20 ft GP | 15 | €28,500 | €95.00 | €24.51 per drum | €81.4 |
| A, 40 ft HC | 12 | €31,200 | €104.00 | €26.84 per drum | €89.1 |
| B, 40 ft HC | 20 | €52,000 | €173.33 | €1.664 per case | €55.5 |
| B, 20 ft GP | 49 | €93,100 | €310.33 | €2.979 per case | €99.3 |
| C, 40 ft HC | 90 | €234,000 | €780.00 | €3.276 per case | €45.5 |
| C, 20 ft GP | 227 | €431,300 | €1,437.67 | €6.038 per case | €83.9 |

The two cost columns rank the products in opposite orders, and both are right. Per tonne, C is
eight times as expensive to move as A. Per cubic metre of cargo, C is the cheapest at €45.5 and A
the most expensive at €81.4.

The reason is in the rates. A 40-foot high cube at €2,600 and 76.35 m³ costs €34.05 per cubic metre
if filled completely; a 20-foot at €1,900 and 33.20 m³ costs €57.23. The small box is the more
expensive box per unit of space; the large box is the more expensive per unit of mass. A
weight-bound cargo is buying space it cannot use, so it should buy the cheap space in small
quantities. A volume-bound cargo is buying mass capacity it cannot use, so it should buy the cheap
space in large quantities.

C at €45.5 per cubic metre pays 34% above the theoretical floor of €34.05, which is the cost of
its 25% cube loss. A at €81.4 pays 139% above the floor, which is the cost of shipping air in a box
designed for light goods. Neither number appears in a rate negotiation, because both are
properties of the product-container pair rather than of the rate.

Add product value and the picture changes again. At €2.40, €6.50, and €9.00 per kilogram of goods
value, freight is 3.8%, 2.6%, and 8.0% of goods value: the product with the lowest freight per
tonne has the highest freight as a share of value, and a memo that makes only one of those points
is making half an argument.

### 6.5 Where the constraint actually sits

**A is geometry-bound in the small box and payload-bound in the large one.** In the 20-foot it
stops at 80 drums because two tiers of 0.88 m is 1.76 m and a third would need 2.64 m against 2.393
m of internal height. Payload would allow 109. The box is 27% empty by mass and 27% empty by
volume, and the missing 29 drums are a consequence of the drum's height rather than of the
container.

> **Note:** a drum 750 mm tall would allow three tiers and 120 positions, at which point the
> payload limit of 109 binds instead — a 36% gain from a packaging change, not a freight decision.

**B is floor-bound in both boxes, and the pallet is the constraint.** Twenty positions at 2.15 m of
build height leave 0.548 m of 2.698 m unused. The pallet holds 80 cases because 8 layers is the
conventional build, not because 10 is impossible: 10 layers of 0.250 m plus a 0.150 m pallet is
2.650 m, which fits with 48 mm to spare. The container is not the constraint; the pallet build is.

**C is geometry-bound and badly so.** One hundred cases across a 2.352 m width leaves 0.352 m
unused, 15% of the width, because five cases of 0.400 m fit and a sixth needs 2.400 m. Eight
tiers of 0.300 m is 2.400 m in a 2.698 m box, leaving a further 0.298 m. The two remainders
multiply: the box is 75% full and all of the 25% is geometry.

### 6.6 The road check, and the tolerance it removes

The 40-foot high cube loaded with 102 drums, on a five-axle combination in the United States:

```text
tare                      3,900 kg
chassis                   3,600 kg
tractor                   8,200 kg
cargo (102 x 258)        26,316 kg
------------------------------------
gross combination        42,016 kg  = 92,630 lb

US federal GCW cap      36,287 kg  = 80,000 lb
excess                    5,729 kg  = 12,630 lb
legal cargo on this rig  20,587 kg  (payload 26,460 kg, so 5,873 kg unusable)
```

The container is inside its rating, the cargo inside its payload, and the combination 5.7 tonnes
over the federal gross limit. This is the ordinary case for a 40-foot box loaded near its rating,
and it is why several states publish overweight container permits at 90,000 lb to 100,000 lb
(Grade B for the federal limits, Grade B for the state rules as summarised by the Federal Highway
Administration).

Gross weight does not settle the axle question. Drums concentrated in the forward half can put the
drive tandem over 34,000 lb while the gross stays under 80,000 lb, and the bridge formula adds a
third test that depends on axle spacing rather than mass. Three tests, three numbers, one load.

Now the tolerance. Against a maximum gross mass of 30,480 kg, with 3,900 kg of tare and 120 kg of
dunnage:

```text
VGM = 26,316 + 3,900 + 120 = 30,336 kg
margin to maximum gross     =    144 kg  = 0.47%
```

A 0.8% error in the declared drum mass — 2 kg on 258 kg — puts the box over its plate: at 260 kg
per drum the VGM is 30,540 kg, 60 kg over. The plan is not wrong about the container; it is wrong
about the product, and it has no room for the error. A plan binding at 95% of a limit is better
than one binding at 100%, even when both produce the same container count.

### 6.7 Where a packaging change beats another container

Three levers in ascending order of cost, on the same demand and rates.

**Lever one: fill the height you already paid for (product B).** Rebuild the pallet to 10 layers.
The build is 2.650 m, fitting 2.698 m with 48 mm to spare, and the pallet holds 100 cases.

```text
cases per box      20 x 100 = 2,000
containers         31,250 / 2,000 = 15.63 -> 16
freight            16 x 2,600 = EUR 41,600
per tonne          138.67   (from 173.33, a 20% reduction)
saving             10,400 per quarter
```

The cost is more compression on the bottom layer and a check that the cases bear nine above them —
a packaging test, not a freight decision, and the one usually skipped.

**Lever two: change the pack (product C).** Cut case height from 300 mm to 255 mm by removing
headspace. Ten tiers then fit: 10 x 0.255 = 2.55 m.

```text
cases per box      100 x 10 = 1,000
containers         71,429 / 1,000 = 71.43 -> 72
freight            72 x 2,600 = EUR 187,200
per tonne          624.00   (from 780.00, a 20% reduction)
saving             46,800 per quarter

a 5% ocean rate cut on the old plan = 234,000 x 0.05 = 11,700
ratio                                = 4.0
```

The packaging change is worth four years of rate negotiation, and it is invisible to a procurement
team whose objective is the rate per container.

**Lever three: change the box or the mode (product A).** The drum is geometry-bound in a 20-foot at
80 units. A shorter drum is Lever two. A 40-foot is a box change and is worse: 102 drums at €104
per tonne against 80 at €95. The 40-foot carries 28% more drums and costs 9% more per tonne, which
is the arithmetic of shipping air. For cargo too dense even for a 20-foot, the answer is a
different product form — bulk liquid in a tank container rather than drums — which is a mode
change and outside this topic.

The generalisable result: **once the container is not the binding constraint, buying another
container is the most expensive way to buy capacity.** Fill the height, then change the pack, then
change the box, then change the mode. Each step costs more per unit of capacity gained, and the
first is usually free.

## 7. Code, query, or calculation

One block, because the arithmetic in section 6 is the point and the tooling is not. It returns the
four maxima, the binding constraint, the margin to the next one, and the gate.

```text
function capacity(box, unit):                 -- box and unit are records
    cube_max    = floor(box.volume * (1 - box.stowage_loss) / unit.volume)
    payload_max = floor(box.payload / unit.mass)
    pos_per_tyr = floor(box.length / unit.length) * floor(box.width / unit.width)
    tiers       = floor(box.height / unit.stack_height)
    geometry_max= pos_per_tyr * tiers
    maxima      = {cube: cube_max, payload: payload_max, geometry: geometry_max}
    binding     = argmin(maxima)
    second      = min of the other two
    gate        = (unit.door_width  <= box.door_width)
                  and (unit.door_height <= box.door_height)
    return {maxima, binding, margin: second - maxima[binding],
            gate: gate, load: gate ? maxima[binding] : 0}
```

Four decisions are explicit that a spreadsheet leaves implicit: the stowage loss is a parameter of
the box and the cargo; the floor plan is computed per axis so the remainder is visible; the tier
count comes from stacked height rather than unit height; and the door zeroes the answer rather
than reducing it.

| Test | Input | Expected | What a failure indicates |
|---|---|---|---|
| Dense cargo, small box | A in 20 ft GP | 80, binding geometry | Payload or cube used as the limit |
| Dense cargo, large box | A in 40 ft HC | 102, binding payload | Volume divided by unit volume, no payload check |
| Palletised cargo | B in 40 ft HC | 1,600, binding geometry | Floor area divided by pallet area, giving 23 |
| Light cargo | C in 40 ft HC | 800, binding geometry | Cube limit of 1,060 reported |
| Door gate | C in 20 ft GP with a 2.4 m case | 0 | Gate not evaluated |
| Stowage loss | B with 10% loss | unchanged, 1,600 | Loss applied to a geometry-bound case |

The last row is the subtle one. A stowage loss reduces the cube limit and leaves the floor limit
untouched, so it changes the answer only when the cube was binding; applying it everywhere hides
which constraint the loss acts on.

## 8. Data, standards, and regulatory context

Four bodies of rule touch this topic: dimensional and structural standards for the box, the mass
verification regime, the road limits, and the commercial conventions that govern nothing but are
followed by everyone.

### 8.1 The container

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 668:2020 | Classification, external dimensions, ratings | 6,058 mm and 12,192 mm external length; 2,591 mm and 2,896 mm height; 30,480 kg nominal rating, higher to 36,000 kg; minimum internal width 2,330 mm and internal height equal to external height minus 241 mm | B |
| ISO 1496-1 | Specification and testing | Floor strength, stacking, racking, concentrated load test | B |
| ISO 6346 | Identification and marking | The four-character type code on every equipment list | B |
| CSC, International Convention for Safe Containers | Safety approval plate | The maximum gross mass marked on the plate, which the VGM may not exceed | B |

### 8.2 The dimensions that vary by type

| Type | Internal length (m) | Width (m) | Height (m) | Note |
|---|---:|---:|---:|---|
| 20 ft general purpose | 5.898 | 2.352 | 2.393 | Reference case |
| 40 ft high cube | 12.032 | 2.352 | 2.698 | 305 mm taller than a standard box |
| 20 ft refrigerated | 5.451 | 2.290 | 2.167 | Insulation and floor plenum cost about 450 mm of length |
| 40 ft high cube refrigerated | 11.580 | 2.275 | 2.415 | About 280 mm shorter than the dry high cube |
| 20 ft open top | 5.919 | 2.346 | 2.286 | Removable roof bows; height limited by tarpaulin and lashing |
| 40 ft high cube open top | 12.032 | 2.352 | 2.653 | Loads from above; the crane is the gate, not the door |

The reefer row catches planners. A reefer is not a dry box with a compressor: it loses length to
machinery, width and height to insulation, and gains a load line that is the real height limit. A
palletised load planned against a dry high cube's 2.698 m will not fit 2.415 m, and 283 mm is more
than one layer of most cartons.

> **Note:** the 40 ft high cube reefer is where the sources disagree with themselves — 67.5 m³
> stated alongside dimensions that multiply to 63.62 m³. Use the dimensions, and record the
> discrepancy in the data dictionary rather than averaging the two.

### 8.3 Mass verification

| Requirement | Where | Effect on the load | Grade |
|---|---|---|---|
| SOLAS Chapter VI, Regulation 2, as amended by MSC.380(94) | In force 1 July 2016 | Shipper must provide a verified gross mass before loading | B |
| Method 1 | IMO MSC.1/Circ.1475 | Weigh the packed and sealed container on calibrated equipment | B |
| Method 2 | IMO MSC.1/Circ.1475 | Sum gross package masses plus dunnage and securing plus tare, under a method approved where packing was completed | B |
| No VGM, no loading | SOLAS VI/2.6 | Container shall not be loaded if the document lacks a VGM and neither master nor terminal has otherwise obtained one | B |
| Cut-off | Carrier, after consulting the terminal | SOLAS sets no deadline; the carrier sets one and must inform the shipper | B |
| Tolerances | National guidance | United Kingdom ±5% or ±500 kg, whichever is greater; Hong Kong ±5% or ±0.5 t | B |

Three practical points, none of them in the convention text. The shipper named on the bill of lading
is responsible, not the forwarder, unless the forwarder is the named shipper. Method 2 is not a
spreadsheet sum: it needs a method approved by the competent authority of the state where packing
was completed, and it is unsuitable for cargo that cannot be individually weighed. And the
declared mass and the weighed mass are different objects, which is why tolerance guidance exists;
when they differ, the weighed mass governs and the container may be held.

### 8.4 Road limits

| Rule | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| 23 U.S.C. §127 and 23 CFR 658.17 | United States, Interstate System | Single axle 20,000 lb, tandem 34,000 lb, gross 80,000 lb, subject to the bridge formula | B |
| Bridge formula | United States | Permissible mass as a function of axle count and spacing | B |
| State overweight container permits | Georgia, Maryland, New York, California, Indiana, Arkansas, Pennsylvania, Louisiana and others | 90,000 lb to 100,000 lb gross for sealed containers, on stated routes | B |
| National weights and dimensions regimes | Everywhere else | Their own gross, axle, and dimensional limits, which do not translate | B |

The existence of the permit regimes is the finding. A limit that can be raised by permit, route,
and fee is not a physical constraint; it is a price. Modelling the road as hard overstates the cost
of a heavy lane; ignoring it understates the risk. Model it as a cost that varies by jurisdiction.

### 8.5 What is convention rather than requirement

- "A 40-foot takes 22 pallets" is true for one pallet and one pattern, and false for most others.
  Section 9.1 shows the sources disagreeing by 10%.
- "Pallet height is 2.2 metres" is a distribution convention. The limit is internal height minus
  the clearance needed to load.
- "A container holds 20 tonnes" is a planning heuristic that ignores payload, road limits, and
  density. Useful as a default, poor as a specification.
- "The pallet is the unit load" is a commercial choice. Slip sheets, hand stacking, and floor
  loading are all available, and each moves the answer by 20% to 30%.

An analyst who can separate a requirement from a convention can challenge the convention with
arithmetic. One who cannot will treat "22 pallets" as a property of the container and be wrong for
every pallet except the one the number came from.

## 9. Industry practice and case evidence

### 9.1 The pallet count is not a fact

Ask five sources how many 1,200 by 1,000 mm pallets fit on the floor of a 40-foot container and you
get five answers: 20 for a straight grid and 22 for a mixed pattern from a load calculator; 21 from
a pallet manufacturer, noting that naive division gives 20 and an interlocked pattern adds one; 20
to 21 from a carrier's stuffing plan; 21 from a loading-plan drawing; and 21 from a logistics
supplier's calculator (Grade D, vendor tools and loading plans, retrieved 2026).

The disagreement is not an error. It is the difference between a grid and a pattern, and between a
plan a forklift driver can build in a shift and one that needs a drawn diagram. What matters for
analytics is that a number pulled from any of these sources and stored as `pallets_per_40ft` will be
wrong for some pallet, some pattern, and some driver. The number belongs to a plan, not to a
container type. Store the plan, or store the dimensions and compute.

### 9.2 The road is the constraint nobody models

The federal limits are simple; the state permit regimes are not. Georgia allows 100,000 lb gross for
a 40-foot container on a five-axle unit under a containerised-freight permit. Maryland allows 90,000
lb for sealed seagoing containers, with 22,400 lb on a single axle and 44,000 lb on a tandem. New
York allows sealed ISO containers to 125% of legal gross or 100,000 lb, whichever is lower.
California permits 95,000 lb on specified port routes. Louisiana distinguishes a Class I permit at
80,000 lb from a Class II at 95,000 lb within 50 miles of a port (Grade B, state regulations as
summarised by the Federal Highway Administration, retrieved 2026).

Two conclusions. **There is no single legal payload for a 40-foot container**, and a model with one
is a model of one jurisdiction. The variation is large enough to change a sourcing decision: the
5,700 kg gap between the federal limit and a permit is 22% of the payload.

### 9.3 What the loading literature actually optimises

A state-of-the-art review of container loading problems reports constraint frequencies across
formulations: orientation about 70%, stability about 37%, stacking about 15%, weight limits about
14%, weight distribution about 12% (Grade A, review paper; the same review is cited in Topic
01A.01).

Read those frequencies as a description of what the field chose to solve: most formulations pack
rectangles into a rectangle, so an optimiser can be geometrically optimal and commercially illegal.

### 9.4 What the evidence does not show

No source located in this research measures the distribution of binding constraints across a
representative population of shipments. The claim that most container loads are cube-bound is
plausible, widely repeated, and unsupported by any base rate found here. Treat it as a hypothesis
to test against your own data using section 5.

The carrier specifications are Grade C: company-published, partly undated, and in one case
inconsistent with themselves. They are the best source for internal dimensions and the worst for a
precise payload on a specific unit. The trade stowage losses, 5% to 18%, are rules of thumb
illustrated by examples rather than measurements of a population; replace them with a value
measured from your own floor plans as soon as you have one.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Nominal capacity used in arithmetic | Plan volume exceeds the product of internal dimensions | A rounded marketing figure treated as a measurement | Every volume wrong by 2% to 6% | Compute from internal dimensions; record the source |
| Payload assumed to double with length | "Two 20s or one 40, same thing" | Rating is capped while tare grows | Underestimated containers for dense cargo | Compute payload as rating minus tare per type |
| Cube checked, floor ignored | Reported volume utilisation above 90% | Volume divided by unit volume, no floor plan | Optimistic by 20% to 40% on palletised cargo | Lay out positions per tier |
| Floor area divided by pallet area | Position count fractional, then rounded up | Remainder on each axis ignored | Overstated count by 15% for a 1,200 mm pallet | Compute `floor(L/l) x floor(W/w)` per axis |
| Tier count taken from height alone | Load arrives crushed or is built short | Stacking strength and bracing not checked | Damage, or an unbuildable plan | State the tier count and its basis |
| Door opening ignored | Cargo fits inside and not through the door | Door smaller than the internal section | Rejected load, rework at the port | Test the gate in both directions |
| Ullage called broken stowage | Proposed remedy is a different container | Two losses conflated | Wrong remedy, no improvement | Separate the two percentages |
| Payload respected, line load not | Heavy cargo concentrated in a few metres | Distributed and concentrated limits differ | Floor damage, refused loading | Compute mass per running metre |
| Gross weight legal, axle not | Permit fine and the scale still red | Centre of gravity sets axle distribution | Fines, transload, delay | Compute per-axle load |
| Container rating treated as the road limit | Plan meets the payload and not the law | Two caps, different values and jurisdictions | 5 to 6 tonnes unusable, found late | Add the vehicle calculation |
| VGM built from item-master weights | Declared mass tracks the plan exactly | Unverified weights presented as verified | Container held or not loaded | Weighing, or a certified Method 2 |
| Plan binds exactly at a limit | Margin to the next constraint under 1% | No allowance for pack-data or dunnage error | Overweight or short | Carry a stated 2% to 5% margin |

**Cube checked, floor ignored** is the most common analytical failure because the number looks
right: a 40-foot high cube holds 76.35 m³, so a 0.03 m³ case "fits" 2,545 times, yet the answer for
the cartons in section 6 is 1,600 because the pallet plan runs out first.

**Plan binds exactly at a limit** is invisible in every report because the plan is internally
consistent. The 102-drum load has a 144 kg margin on a 30,480 kg maximum, and the only way to find
it is to compute it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Internal dimensions are those of the unit booked | Every capacity derives from them | Mixed fleet, substitution, aged equipment | Request the specification at booking; spot-check on arrival |
| Payload is rating minus tare for the specific unit | The crossover depends on it | Tare varies by maker and age | Compare the carrier sheet with the CSC plate |
| Cargo is one product with uniform units | The minimum-of-four needs one unit | Mixed loads, mixed pallets, part loads | Compute per SKU; solve the mixed case separately |
| Unit mass and volume are gross and accurate | Density determines the classification | Pack-data error, moisture, fill tolerance | Weigh a sample against the item master |
| One stowage loss applies to the load | Converts ideal cube into usable cube | Loss varies by position and cargo | Derive it from the actual floor plan |
| Load is evenly distributed along the length | Line load and centre of gravity computed once | Concentrated loads, heavy ends | Compute the centre of gravity explicitly |
| The container travels on the vehicle assumed | The road check needs a vehicle | Rail, barge, short sea, direct truck | State the vehicle in the plan |
| One jurisdiction's limits apply throughout | The road check needs one rule set | Cross-border, port-to-inland transfer | Identify each jurisdiction on the route |
| Stacking strength supports the tier count | The tier count sets the geometry limit | Crushable cargo, long transit, humidity | Compression test, or drop one tier |

Three limits are worth stating.

**The computation is an upper bound, not a loading plan.** It assumes uniform units, a regular
grid, and a stowage loss capturing everything the grid cannot; a real plan with mixed sizes, braced
loads, and a door recess will be lower. The bound says which constraint to attack, not what can
be executed.

**Mixed loads are outside the model.** Three products with different densities and stackabilities
is a packing problem with simultaneous cube, floor, and payload constraints, and the minimum-of-four
logic does not survive. Compute each product separately, then allocate.

**The evidence on typical loads is thin.** The arithmetic is solid and the standards are clear, but
no base rate on which constraint usually binds was located. Measure it in your own data before
designing a policy around it.

## 12. Exercises

### Exercise 1 — Reconstruct the four capacities (application, 30 minutes)

Take one container from a carrier's specification sheet: internal length, width, height, tare,
maximum gross mass, and door opening. Derive volume, floor area, payload, and door clearance.

**Deliverable:** a table of source and derived figures with the arithmetic shown.
**Acceptance criteria:** payload computed as maximum gross minus tare; door clearance in both width
and height; one sentence naming the figure that would change most with another carrier.
**Time:** 30 minutes.

### Exercise 2 — Compute the crossover (calculation, 30 minutes)

For the six types in section 5.1, compute crossover density and crossover stowage factor on nominal
volume and with a 12% stowage loss. Rank the six by crossover density.

**Deliverable:** six rows by four computed columns, the ranking, and a one-paragraph reading.
**Acceptance criteria:** arithmetic shown; both nominal and adjusted figures; the ranking unchanged
by the adjustment, with an explanation; the reading naming which type suits a 500 kg/m³ cargo and
which a 150 kg/m³ cargo.
**Time:** 30 minutes.

### Exercise 3 — Lay out two floor plans (application, 45 minutes)

For a 1,200 by 1,000 mm pallet and a 400 by 300 by 250 mm case, lay out a 20-foot general-purpose
and a 40-foot high cube, palletised and hand stacked. Compute positions per tier, tiers, units per
box, floor utilisation, and cube utilisation.

**Deliverable:** four scaled sketches or text grids, plus a table of computed figures.
**Acceptance criteria:** positions per axis, not by area; tier count justified by stacked height;
floor loss and height loss reported separately; which plan you would execute, and its hidden cost.
**Time:** 45 minutes.

### Exercise 4 — The height audit (calculation, 30 minutes)

A pallet build is 2,150 mm tall including a 150 mm pallet, in a box of 2,698 mm internal height,
carrying 80 cases of 250 mm. Find the unused height, the additional layers that fit, the new build
height, and the gain in cases per container.

**Deliverable:** the arithmetic and a one-paragraph recommendation.
**Acceptance criteria:** loading clearance stated; the compression implication named; the gain in
cases per container and in containers per 1,000 tonnes; the recommendation conditional on a test.
**Time:** 30 minutes.

### Exercise 5 — Classify five products (application, 30 minutes)

Compute gross envelope density and stowage factor for bagged flour at 25 kg in 500 by 350 by 150 mm;
bottled water at 12 by 1.5 L in 400 by 300 by 250 mm, gross 19.2 kg; flat-pack furniture at 18 kg in
1,200 by 400 by 100 mm; crisps at 4.2 kg in 600 by 400 by 300 mm; and steel fasteners at 22 kg in
300 by 200 by 150 mm. Assign each a box and predict the binding constraint.

**Deliverable:** a five-row table with mass, envelope, density, stowage factor, box, and predicted
constraint.
**Acceptance criteria:** gross mass including packaging; envelope rather than product volume; the
assignment justified by Exercise 2; one product within 15% of a crossover, with its consequence.
**Time:** 30 minutes.

### Exercise 6 — The road check (calculation, 45 minutes)

A 40-foot high cube with a 3,850 kg tare carries 24,000 kg on a five-axle combination with a 3,600
kg chassis and an 8,100 kg tractor. Compute gross combination weight in kilograms and pounds,
compare with the United States federal limit, and find the maximum legal cargo. Then compute line
load if the cargo occupies 8.0 m of 12.0 m, against a 3.0 t/m limit.

**Deliverable:** both computations with units, the comparison, and a recommendation.
**Acceptance criteria:** the two limits treated separately with the binding one named; line load on
cargo length, not box length; two remedies, one changing the load and one changing the vehicle; the
jurisdiction stated.
**Time:** 45 minutes.

### Exercise 7 — Build a VGM declaration (design, 40 minutes)

For the load in Exercise 6 plus 180 kg of dunnage and securing, produce the verified gross mass under
both methods. State which you would use, what evidence each requires, and what happens if the
declared mass exceeds the plate.

**Deliverable:** two computation sheets, a method recommendation, and a 200-word process note.
**Acceptance criteria:** tare and dunnage in both; Method 2 as a sum of gross package masses, not net
product; the approval requirement stated; the consequence stated as a loading prohibition.
**Time:** 40 minutes.

### Exercise 8 — Pallet against slip sheet (design, 60 minutes)

A customer takes 20 pallets a week and will accept slip sheets if the supplier provides a push-pull
attachment at destination. Palletised gives 1,600 cases per 40-foot high cube, slip-sheeted 2,000.
Freight is €2,600, demand is 31,250 cases per quarter, and pallets cost €14 each, one-way.

**Deliverable:** a one-page business case with container counts, freight saving, pallet cost, and the
break-even equipment price.
**Acceptance criteria:** both plans on the same demand; pallets treated as consumable; equipment cost
annualised over a stated life; the recommendation conditional on customer acceptance.
**Time:** 60 minutes.

### Exercise 9 — Critique a loading plan (critique, 45 minutes)

A plan proposes 22 pallets of 1,200 by 1,000 mm in a 40-foot standard container, stacked 2.4 m high,
gross 26,900 kg, Shanghai to Rotterdam with inland delivery in the Netherlands. It states "volume
utilisation 96%, within payload".

**Deliverable:** a written critique of every numerical and logical defect, with the correct figure
for each.
**Acceptance criteria:** at least four defects; volume utilisation recomputed from pallet
dimensions; payload checked by type; internal height checked against the 2.4 m stack; the road leg
identified as a separate jurisdiction; one defect the plan's own numbers contradict.
**Time:** 45 minutes.

### Exercise 10 — The binding-constraint memo (design, 45 minutes)

For a product you know, write a one-page memo stating the binding constraint, the margin to the next
one, and the single change that would most improve cost per unit.

**Deliverable:** a memo with a headline, the four maxima, the binding constraint, the margin, the
change, and the arithmetic of the improvement.
**Acceptance criteria:** all four maxima computed; units on every figure; the change classified as
fill-the-height, change-the-pack, change-the-box, or change-the-mode; improvement quantified per
unit and per quarter; one condition under which the recommendation reverses.
**Time:** 45 minutes.

## 13. Self-check answers

### Exercise 2 — crossovers

| Type | Payload (kg) | Cube (m³) | Crossover (kg/m³) | At 12% loss | SF (m³/t) |
|---|---:|---:|---:|---:|---:|
| 20 ft general purpose | 28,200 | 33.20 | 849 | 965 | 1.18 |
| 20 ft refrigerated | 27,490 | 27.05 | 1,016 | 1,155 | 0.98 |
| 40 ft high cube refrigerated | 29,480 | 63.62 | 463 | 527 | 2.16 |
| 40 ft general purpose | 26,700 | 67.78 | 394 | 448 | 2.54 |
| 40 ft high cube | 26,460 | 76.35 | 347 | 394 | 2.89 |
| 45 ft high cube | 27,450 | 86.02 | 319 | 363 | 3.13 |

A uniform loss multiplies every crossover by `1 / 0.88 = 1.136`, so the ranking cannot change — a
general property. It does change individual classifications near a boundary, which is why the loss
must be stated. A 500 kg/m³ cargo is weight-bound in every 40-footer and cube-bound in both
20-footers; a 150 kg/m³ cargo is cube-bound in all six.

### Exercise 3 — the four floor plans

| Plan | Positions per tier | Tiers | Units | Floor use | Cube use |
|---|---:|---:|---:|---:|---:|
| 20 ft, palletised | 4 x 2 = 8 | 1 | 8 pallets (640 cases) | 69.2% | 62.2% |
| 20 ft, hand stacked | 14 x 7 = 98 | 9 | 882 cases | 84.8% | 79.7% |
| 40 ft HC, palletised | 10 x 2 = 20 | 1 | 20 pallets (1,600 cases) | 84.8% | 67.6% |
| 40 ft HC, hand stacked | 30 x 7 = 210 | 10 | 2,100 cases | 89.1% | 82.5% |

The 20-foot hand-stack uses 14 positions along, not 30, because `5.898 / 0.400 = 14.7`. An answer
with 30 in a 20-foot box is the 40-foot answer copied down, and it is the most common mistake here.
The 20-foot palletised plan is only 69% floor-efficient because four pallets of 1.200 m use 4.800 m
of 5.898 m, leaving 1.098 m — nearly a whole pallet — unusable at the door end.

### Exercise 4 — the height audit

```text
unused height        = 2,698 - 2,150 = 548 mm
additional layers    = floor(548 / 250) = 2
new build height     = 2,150 + 2 x 250 = 2,650 mm
clearance to ceiling = 48 mm
cases per pallet     = 80 -> 100, a gain of 25%
containers per 300 t = 20 -> 16 ; per 1,000 t = 66.7 -> 53.3
```

A 48 mm clearance is enough on a clean floor with a square load and not enough with a damaged
pallet or an uneven top layer. The recommendation adds a layer only after a compression test and a
trial load.

### Exercise 5 — the five products

| Product | Mass (kg) | Envelope (m³) | Density (kg/m³) | SF (m³/t) | Box | Predicted binding |
|---|---:|---:|---:|---:|---|---|
| Bagged flour | 25.0 | 0.02625 | 952 | 1.05 | 20 ft GP | Payload |
| Bottled water | 19.2 | 0.03000 | 640 | 1.56 | 20 ft GP | Cube in the 20 ft, payload in the 40 ft |
| Flat-pack furniture | 18.0 | 0.04800 | 375 | 2.67 | 40 ft HC | Near the crossover; cube after a real loss |
| Packaged crisps | 4.2 | 0.07200 | 58 | 17.14 | 40 ft HC | Cube |
| Steel fasteners | 22.0 | 0.00900 | 2,444 | 0.41 | 20 ft GP | Payload, and probably line load |

Bottled water is the instructive row. At 640 kg/m³ it sits between the 40-foot high cube crossover
of 347 and the 20-foot crossover of 849: cube-bound at about 1,107 cases in the small box, weight-
bound at 1,378 in the large one. The same product flips classification when the box changes, which
is why a "cube-bound" label is meaningless without the box. Flat-pack at 375 kg/m³ is within 8% of
the 40-foot crossover before any loss and 5% after one, so its classification needs its margin.

### Exercise 6 — the road check

```text
GCW = 3,850 + 24,000 + 3,600 + 8,100 = 39,550 kg = 87,193 lb
federal cap         = 80,000 lb = 36,287 kg
excess              = 3,263 kg = 7,193 lb
maximum legal cargo = 36,287 - 15,550 = 20,737 kg
line load           = 24,000 / 8.0 = 3.0 t/m, exactly at the limit
```

The load is 3.3 t over the gross limit and exactly at the line limit, which in practice means over
it, because a 3.0 t/m rating carries no tolerance for an unevenly built load. Remedies: a lighter
load, a longer load (24 t over 12 m gives 2.0 t/m), or a permitted vehicle. Changing the vehicle
fixes the gross problem and does nothing for the line load.

### Exercise 8 — pallet against slip sheet

```text
palletised     31,250 / 1,600 = 19.53 -> 20 containers
slip sheet     31,250 / 2,000 = 15.63 -> 16 containers
freight saved  4 x 2,600 = EUR 10,400 per quarter
pallets        20 per week x 13 = 260 x EUR 14 = EUR 3,640
net saving     6,760 per quarter = EUR 27,040 per year
break-even equipment over 3 years = EUR 81,120
```

The break-even price converts an operational argument into a capital one. The recommendation holds
if the customer accepts slip-sheeted loads and the damage rate does not rise; it reverses if the
pallets are returnable and already in the loop, leaving the €10,400 freight saving alone.

### Exercise 9 — the loading-plan critique

1. **The stack does not fit.** 2.4 m against 2.395 m of internal height is over before any loading
   clearance. The correct comparison is internal height, not the 2.591 m external height.
2. **The pallet count is optimistic.** Twenty-two requires a mixed pattern; the straight grid gives
   20. Twenty-two is achievable and needs a drawing and a driver who follows it.
3. **The payload is over.** 26,900 kg exceeds the typical 26,700 kg payload before tare is
   considered against the rating. The plan violates the constraint it claims to satisfy.
4. **The volume figure is not reproducible.** 22 pallets at 1.200 by 1.000 by 2.400 m is 63.36 m³
   against 67.78 m³, or 93.5% — and that counts the void between pallets and the pallet itself as
   cargo.
5. **The door is not tested.** A 2.4 m stack cannot pass a 2.28 m opening; it must be built inside
   in layers, and the road leg quotes no vehicle, axle limit, or gross limit for the Netherlands.

A strong critique leads with defect 1 because it is fatal, and separates defects that make the plan
impossible from those that make it merely optimistic. A weak critique lists the payload and stops,
leaving a plan that still cannot be loaded.

### Design rubrics

A strong Exercise 3 shows positions per axis for both orientations, states its clearance, and
reports floor and height loss separately. A weak one divides areas and reports one utilisation
figure. A strong Exercise 10 memo names the margin and what would reverse the recommendation; a
weak one asserts "volume-limited" without showing the three numbers it did not bind on.

Award no credit for a container count that cannot be reproduced from the stated dimensions. A right
answer reached by an unstated shortcut is not transferable to the next product.

## 14. Interview questions

**1. "How many pallets fit in a 40-foot container?"**

Strong: refuses the question as asked, asks for pallet dimensions and pattern, gives the per-axis
arithmetic, and offers the straight-grid and mixed-pattern answers with the difference. Weak: says
"twenty" with confidence, which is a memory of one pallet and one dock.

**2. "Why can't you divide the container volume by the case volume?"**

Strong: the floor runs out first, the door is a gate, and volume is not usable space, with the
arithmetic for a palletised case where the quotient is 2,545 and the answer is 1,600. Weak: "you
need to allow for packing inefficiency", without a number or a location.

**3. "A 40-foot carries twice as much as a 20-foot. True or false?"**

Strong: false on mass, roughly true on volume, because the rating is capped while the tare grows.
Weak: true, because a 40 is two TEU — a capacity convention being used to answer a mass question.

**4. "What is stowage factor and how do you use it?"**

Strong: volume per unit mass in cubic metres per tonne, the reciprocal of density, compared with
the container's crossover to classify the cargo, with a number per box type. Weak: defines it and
stops, or inverts it and gets the direction backwards.

**5. "How would you choose between a 20-foot and a 40-foot for a new product?"**

Strong: computes the crossover for both, containers required for both, then cost per tonne and per
cubic metre, and shows the rankings differ. Weak: picks the 40 because the rate per container looks
better, without asking per what.

**6. "A container is inside its payload and inside its gross vehicle weight. Can it still be illegal?"**

Strong: yes — on axle load, bridge formula, line load, or a permit condition — and names the
mechanism by which the centre of gravity moves mass between axles. Weak: no.

**7. "What is verified gross mass and why does it exist?"**

Strong: a SOLAS requirement in force since 1 July 2016, two methods, the shipper on the bill of
lading responsible, and no VGM meaning no loading. Explains that it exists because misdeclared mass
caused stack collapses and lost containers. Weak: "the weight of the container".

**8. "How accurate are the weights in your item master?"**

Strong: distinguishes nominal from measured, cites audit evidence that pack-data defects survive in
large organisations, and states the margin a plan should carry. Weak: "they come from the supplier,
so they are accurate" — the assumption the VGM regime exists to replace.

**9. "A plan uses 96% of the cube. What do you check first?"**

Strong: whether the 96% is on internal or nominal volume, whether pallet envelope or case volume was
used, and whether the height is buildable through the door; notes that above about 90% on
palletised cargo is usually an artifact. Weak: congratulates the planner.

**10. "What is the difference between broken stowage and ullage?"**

Strong: shape loss versus short load, the different remedies — a different pattern versus a taller
build — and that only the first belongs in a stowage factor. Weak: treats them as synonyms, which
leads to recommending a new container when the answer is a higher pallet.

## 15. Key terms

- **Container type code** — ISO 6346 four-character designation, such as 22G1 or 45G1.
- **ISO 668 code** — external-size designation: 1CC, 1AA, 1AAA, 1EEE.
- **TEU / FEU** — twenty-foot and forty-foot equivalent units; FEU is exactly two TEU.
- **Internal volume** — product of internal length, width, and height; the only volume to use.
- **Nominal capacity** — rounded figure on a specification sheet; never used in arithmetic.
- **Payload** — maximum cargo mass, equal to maximum gross mass minus tare.
- **Tare** — mass of the empty container.
- **Maximum gross mass** — figure on the CSC plate, which the VGM may not exceed.
- **VGM** — verified gross mass, declared under SOLAS before loading.
- **Method 1 / Method 2** — weighing the packed box; or summing gross packages, dunnage, and tare.
- **Stowage factor** — volume per unit mass in cubic metres per tonne; reciprocal of density.
- **Crossover density** — payload divided by usable volume; the weight-versus-cube threshold.
- **Binding constraint** — first of the four capacities to run out, and therefore the load.
- **Door gate** — binary test of whether a unit passes the opening, in both directions.
- **Broken stowage** — space that cannot be occupied because of shape, packaging, or securing.
- **Ullage** — space left unoccupied because the cargo ran out.
- **Cube and weight utilisation** — occupied volume over internal volume; cargo mass over payload.
- **Floor positions and tier count** — units per tier from per-axis division; layers the height allows.
- **Line load and point load** — mass per running metre; load through a small contact patch.
- **Axle load, GVW, GCW** — mass per axle; gross vehicle weight; gross combination weight.
- **Bridge formula** — axle-spacing rule that can bind below the gross limit.
- **Slip sheet** — thin sheet replacing a pallet, gaining height and floor but needing a push-pull.
- **Dead freight** — freight charged for booked space not used.

## 16. Sources

**Peer-reviewed and academic**

- Bortfeldt, A., and Wäscher, G. (2012). "Container Loading Problems — A State-of-the-Art Review."
  Working paper, Otto-von-Guericke-University Magdeburg. Constraint frequencies: orientation about
  70%, stability about 37%, stacking about 15%, weight limits about 14%, weight distribution about
  12%. Grade A.

**Standards, official, and regulatory**

- ISO 668:2020, *Series 1 freight containers — Classification, dimensions and ratings*, 7th edition.
  External dimensions; nominal rating 30,480 kg, higher values to 36,000 kg; minimum internal width
  2,330 mm and internal height as external height minus 241 mm. Consulted via preview. Grade B.
- ISO 1496-1, *Specification and testing*, Part 1. Floor strength: two loads of 2,730 kg on contact
  areas of 142 cm². Grade B.
- ISO 6346, *Coding, identification and marking*. Container type codes. Grade B.
- CSC, 1972, Annex II. Safety approval plate and maximum gross mass. Grade B.
- IMO, SOLAS VI/2 as amended by MSC.380(94), in force 1 July 2016. Shipper VGM obligation; VI/2.6
  prohibits loading without one. Grade B.
- IMO, *Guidelines Regarding the Verified Gross Mass*, MSC.1/Circ.1475. Method 1, Method 2, and the
  approval requirement. Grade B.
- UK Maritime and Coastguard Agency, *Verification of the Gross Mass of Packed Containers*.
  Tolerance ±5% or ±500 kg, whichever is greater. Grade B.
- Hong Kong Marine Department, *Requirement on the Verified Gross Mass of a Container*. Penalties to
  HK$20,000 and two years; cut-off set commercially. Grade B.
- 23 U.S.C. §127 and 23 CFR 658.17. Single axle 20,000 lb, tandem 34,000 lb, gross 80,000 lb,
  subject to the bridge formula. Grade B.
- Federal Highway Administration, *Questions and Answers About Vehicle Size and Weight*. Containers
  as nondivisible loads; state permit regimes. Grade B federal, C for the state summary.

**Industry, carrier, and trade**

- Hapag-Lloyd, *Container Specification*. Line loads; steel-floor axle load of 9,200 kg. Grade C.
- Maersk, *Reefer Equipment Specifications* and *Special Cargo Container Specifications*, 2026.
  Reefer and open-top internal dimensions and payloads; 40 ft high cube reefer stated at 67.5 m³
  against dimensions multiplying to 63.62 m³. Grade C.
- CMA CGM, *Reefer Technical Specifications*. 40 ft high cube reefer at 68 m³ nominal and 65.3 m³
  usable, payload 29,260 kg. Grade C.
- K Line, "Container Floor Load Limits", citing UK P&I Club. Line loads near 4.5 t/m for a 20 ft and
  3.0 t/m for a 40 ft; roof 300 kg over 600 by 300 mm; end wall 0.4 of payload; side wall 0.6;
  centre of gravity within 0.6 m or 0.9 m of mid-length. Grade C.
- TIS-GDV, "Permissible loading capacity". Floor and concentrated load guidance. Grade C.
- Container Handbook, section 4.2.1.3, "Stowage planning". Stowage losses of 10% and 15%; net and
  gross stowage factors. Grade C, undated on the retrieved page.
- World Shipping Council, *Glossary of Shipping Terms*; Statistics Netherlands (CBS), TEU
  definition. TEU and FEU definitions; 20 ft as 1 TEU, 40 ft as 2 TEU, over 40 ft as 2.25 TEU.
  Grade C and B respectively.
- OOCL, "Stuffing Plan"; Aquablox, *Container Loading Plans*; Mervia Group and Palletly
  calculators; pallet.design. Counts of 20, 21, and 22 for 1,200 by 1,000 mm pallets in a 40 ft;
  20 to 25 for 1,200 by 800 mm. Grade D, retrieved 2026.

**Press and practitioner**

- freightcourse, "Broken Stowage: Definition and Ways to Avoid It". A 40 ft high cube of 76 m³ with
  55 m³ of paper rolls described as about 18% broken stowage; formula stated inconsistently with its
  own examples. Grade E.
- "Stowage Factor and Broken Stowage", training extract. 250 t at 1.8 m³/t occupying 450 m³, raised
  to 472.5 m³ at 5% broken stowage. Grade E.

**Note on citations.** Carrier figures are reported as published and not independently verified; at
least one carrier's volume does not reconcile with its own dimensions, and the text says so rather
than choosing. Vendor pallet counts disagree by up to 10%, which is the finding, and none should be
stored as a container-type constant. No source located here measures the distribution of binding
constraints across a representative population; claims about what "most" loads do are marked as
hypotheses.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_unit-of-measure-hierarchy.md`](01_unit-of-measure-hierarchy.md)
- Next topic: [`03_case-pack-moq-rounding.md`](03_case-pack-moq-rounding.md)
- Related: [`../01_supply_chain_foundations/08_transport-and-incoterms.md`](../01_supply_chain_foundations/08_transport-and-incoterms.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)
