# The Unit-of-Measure Hierarchy

> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 01 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01.01, 01.02, 01.03
> Version 2026.5

---

## 1. Why this topic exists

Every quantity in a supply chain is a pair: a number and a unit. The number is what the
database stores. The unit lives in a column name, a header row, a master-data field, a
contract clause, or nowhere at all. When the unit is missing, the number still looks
plausible, still sums, still plots, and still trains a model. That is the whole problem in
one sentence.

Consider the ordinary act of reporting monthly volume for a product sold in cases of
twenty-four. One table records shipments in eaches, another in cases, and a third in
pallet-equivalents. Sum the column called `quantity` across the three and you get a number
that is wrong by a factor of twenty-four on some rows. Nobody sees an error, because no
system knows what the number meant. The report is not slightly off. It is meaningless, and
it is formatted correctly.

Forecasting, inventory policy, network design, and optimisation all assume a clean quantity.
The assumption is never stated and frequently false. Four specific failures motivate the
treatment.

**Failure one: dimension errors are not arithmetic errors.** On 23 September 1999 the Mars
Climate Orbiter was lost during orbit insertion. The mishap investigation found that ground
software called `SM_FORCES` emitted thruster impulse in pound-force seconds (lbf-s) where
the interface specification required newton-seconds (N-s). The navigation software consumed
the numbers as metric, understating the impulse by a factor of 4.45. The arithmetic inside
each program was correct. The interface had no unit. The spacecraft arrived roughly 170 km
lower than planned, with an estimated periapsis near 57 km against a survivable minimum near
80 km. Grade B, from the mishap report. The mechanism is identical to two warehouse systems
exchanging a quantity with no declared unit and no test at the boundary.

**Failure two: pack data is systematically wrong, and invisible until a physical count.** A
GS1 Australia case study involving Kimberly-Clark reports weights missing or incorrect in
the enterprise resource planning (ERP) system for nearly 100% of audited units, alongside
dimension and identifier errors across 1,526 packaging levels, and links them to freight
costing, data-pool compliance, and distribution-centre automation faults. Grade C, because
it is company-reported rather than an independent audit. The direction is consistent with
practitioner experience; the percentage should not be quoted as a population parameter.

**Failure three: some products have two quantities and most systems have one field.** A
catch-weight item — a bag of frozen fruit, a tray of cut salmon — is handled in cases and
priced in kilograms. Ordered weight is a plan; actual weight is a measurement; they differ on
every lot. An ERP that records only the case count cannot value the inventory, and one that
records only the nominal weight cannot reconcile the invoice. Several ERP vendors document a
dual-quantity design and deliberately refuse to enforce `stock quantity = order quantity ×
conversion factor` (Grade D, vendor documentation, 2026), which is itself evidence that the
one-field model fails.

**Failure four: rounding is a one-way function, performed four to six times per planning
cycle.** A forecast in eaches becomes a plan in cases, an order in pallets, a shipment in
containers, a receipt in cases again. Each step rounds up, and each round-up is irreversible.
The cumulative residue later in this topic is 15.2% of forecast volume, most of it created by
the last two steps. Nobody decided to build 15.2% extra. The arithmetic did.

The four failures share a structure: a competent person made a locally reasonable choice,
and the loss occurred at a boundary where two representations met. Units are that boundary,
which is why this is the first topic of Module 01A and not an appendix.

**Where it leads.** The next topic, `02_container-loading.md`, takes the top of the hierarchy
and asks which constraint binds: mass, volume, floor area, or pallet positions. Topic
`03_case-pack-moq-rounding.md` treats rounding and minimum order quantities as an integer
problem. Both depend on three distinctions established here: a level is not a unit, a
conversion is not automatically exact, and a remainder is a quantity with a cost.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a six-level packaging hierarchy for a named product, labelling each level with
   its containment quantity, its identifier, and whether it is orderable, shippable,
   storable, and priceable.
2. **Distinguish** a packaging level from a unit of measure, and **explain** why "case" is a
   multiplier on a count rather than a physical dimension.
3. **Classify** any conversion as exact, variable, or lossy, and **state** the evidence that
   supports the classification.
4. **Compute** a full rounding cascade from a demand forecast in base units to containers,
   recording the exact quotient, the rounded quantity, and the cumulative residue at each
   level.
5. **Quantify** the cost of a rounding residue in three currencies: units, cash, and profit
   and loss, and **explain** why the first two are not the same as the third.
6. **Detect** a double-conversion defect by constructing a round-trip test and interpreting
   whether a failure comes from integer rounding, a truncated reciprocal, or a genuinely
   variable factor.
7. **Design** a record structure that holds two quantities for a catch-weight item and
   **explain** which field drives inventory valuation, which drives invoicing, and which
   drives the physical count.
8. **Compute** a stowage factor from item dimensions and **decide** whether a container will
   be weight-limited or volume-limited before any loading software is involved.
9. **Audit** an unfamiliar dataset for unit ambiguity and **produce** a written statement of
   what the quantity column actually measures.

Section 12 provides exercises for outcomes 1 to 8; section 13 works through the quantitative
answers.

## 3. The core mental model

Hold two pictures at once. The first is physical: boxes inside boxes, stacked on a platform,
loaded into a steel box. The second is informational: a set of identifiers and multipliers
that claim to describe the first. Most unit errors are places where the second picture has
silently simplified the first.

### 3.1 The hierarchy as a physical fact

```text
   CONTAINER 40 ft HC ............ 20 pallets .... 38,400 EA
     |
     +-- PALLET 1200 x 1000 mm ... 8 layers ......  1,920 EA   (SSCC)
           |
           +-- LAYER ............. 10 cases ......    240 EA   (no identifier)
                 |
                 +-- CASE ........ 4 inner packs .     24 EA   (GTIN-14)
                       |
                       +-- INNER PACK  6 eaches ..      6 EA   (GTIN-13)
                             |
                             +-- EACH ................  1 EA   (GTIN-13)
```

Six levels. Read the diagram upward and every arrow is a multiplication. Read it downward
and every arrow is a division that may leave a remainder. The upward direction is safe. The
downward direction is where information dies.

Three properties of the physical picture matter for everything that follows.

**Property 1 — only some levels are artefacts you can pick up.** An each, an inner pack, a
case, and a pallet are physical objects with identity. A layer is a position; a container is
a vehicle. Identifiers attach to artefacts, which is why layer-level rounding is invisible in
most systems.

**Property 2 — the multiplier belongs to the pack specification, not the product.** The same
product sold in a six-pack and a twelve-pack has two case-level identifiers and two
conversion factors. A data model that attaches `case_pack` to the product rather than to the
trade item cannot represent this, and such models are extremely common.

**Property 3 — the hierarchy is a tree in the catalogue and a graph on the dock.** In master
data, each case contains one stock-keeping unit (SKU) and each pallet one case type. On the
dock, a mixed case holds several SKUs, a mixed pallet holds several case types, and a
display-ready shipper is simultaneously a case and a shelf unit. The GS1 US packaging-level
guidance enumerates eight levels for exactly this reason: each, inner pack, case, mixed case,
pallet, mixed pallet, display shipper, and case-as-each. A model with one parent pointer per
level cannot represent the last four. Grade B.

```text
   CATALOGUE VIEW (tree)                    DOCK VIEW (graph)

        CONTAINER                              CONTAINER
            |                                  /       \
         PALLET (one GTIN)              PALLET A       PALLET B
            |                          /   |   \           |
          LAYER                     CASE  CASE  CASE    MIXED PALLET
            |                       |      |      |      /       \
          CASE                    EA     EA     EA    CASE      CASE
            |                                        (GTIN X)  (GTIN Y)
        INNER PACK
            |
          EACH
```

The practical consequence: a strict parent-child foreign key is the wrong schema. You need
a containment relation with three fields — parent, child, quantity — plus a role attribute
saying what the level is for. A case can be a shipping unit, a storage unit, and an ordering
unit at once, or any subset of the three. Topic 07 of this module develops the general
hierarchy design; here the point is only that packaging is not a strict tree.

### 3.2 A unit is a dimension plus a magnitude

"Unit" alone is ambiguous because it conflates two things. A dimension is the kind of
quantity: count, mass, volume, length, area, time. A unit is a declared magnitude within a
dimension: each, kilogram, cubic metre, metre, square metre, hour. A number without a
dimension cannot be added to another number safely, and a number without a magnitude cannot
be converted.

This is why "case" is not a unit of measure in the physical sense. A case is a count of
eaches bundled by a physical artefact. Its dimension is count. Its magnitude is whatever the
pack specification says, usually an integer between two and several hundred. Two cases of
different products are both counts of two, and they are not comparable quantities. The trade
abbreviation "CA" conceals this: in the UN/CEFACT Recommendation 20 annexes returned in the
research for this topic, `EA` denotes each, `CA` denotes can, and `PL` denotes pail, with
packaging units maintained separately under Recommendation 21. Trade systems overwhelmingly
use `CA` for case and `PL` for pallet. The collision is real. Grade B.

The coded landscape has two families that solve different problems. Recommendation 20 is a
list of three-character codes organised by quantity, used by GS1's EDI guidance for data
element 6411. UCUM, the Unified Code for Units of Measure, is a syntax: case-sensitive
expressions built from atoms and prefixes, `.` for multiplication and `/` for division, used
by HL7 under the identifier `http://unitsofmeasure.org`. Its base units are metre, second,
gram, radian, kelvin, coulomb, and candela — gram rather than kilogram, radian rather than
mole — so it is deliberately not isomorphic to the International System of Units (SI), and
curly braces in UCUM are annotations with no semantic meaning. Grade B for both.

The contrast is the lesson. A code list tells you which strings are legal; a syntax tells you
which expressions are dimensionally coherent. Supply chain systems almost always implement
the first and almost never the second, which is why a warehouse management system (WMS) will
happily store a quantity of `24` against a unit of `KG` and never complain.

### 3.3 Conversions have integrity classes

Not all conversions are the same kind of object, and systems that treat them as one kind
produce three distinct families of defect.

```text
   EXACT                    VARIABLE                  LOSSY
   1 CA = 24 EA             1 CA = 9.4 to 10.6 kg     4,167 EA / 24
   invertible               needs the scale           = 173 CA remainder 15 EA
   integer ratio            a distribution per lot    remainder has no
                                                      representation in CA
```

- **Exact.** The factor is a fixed integer ratio and the conversion is reversible when the
  intermediate is not constrained to an integer. A case of twenty-four always holds
  twenty-four. Within the pack specification, this is a definition, not a measurement.
- **Variable.** The factor is a random variable with a distribution, known per lot or per
  unit only by measurement. Catch-weight and liquid-fill products live here. The conversion
  is exact only in one direction and only after the measurement exists.
- **Lossy.** The conversion is exact in principle but the target unit cannot represent the
  remainder. A partial case is a real physical object and an unrepresentable data value.
  Rounding is the only available operation, and it destroys the remainder.

Most ERP conversion tables encode a single number per pair and a single rounding rule. That
is adequate for the first class, wrong for the second, and structurally incapable of
representing the third. The classification is therefore not academic bookkeeping. It
determines whether a conversion may be applied silently in a query or must be surfaced as a
decision.

### 3.4 Rounding is a cascade, not an event

```text
   forecast      100,000 EA          exact
       |  / 6  (inner pack)
       v
   inner packs    16,666.67 PK  --> 16,667 PK   (+0.33 PK = +2 EA)
       |  / 4  (case)
       v
   cases           4,166.67 CA  -->  4,167 CA   (+0.33 CA = +8 EA)
       |  / 10 (layer)
       v
   layers            416.70 LY  -->    417 LY   (+0.30 LY = +80 EA)
       |  / 8  (pallet)
       v
   pallets            52.13 PL  -->     53 PL   (+0.87 PL = +1,760 EA)
       |  / 20 (container)
       v
   containers          2.65 CN  -->      3 CN   (+0.35 CN = +15,200 EA)
```

The staircase is the core mental model of this topic. Each step rounds up to the smallest
executable unit at that level. The residue is not an error; it is the price of executability,
and it is invisible because the planning system reports the rounded number and calls it the
plan.

Two properties deserve memorising. **The residue compounds, then jumps.** The first three
steps add 80 eaches in total, under 0.1% of forecast; the pallet step adds 1,680 and the
container step 13,440. A planner watching only case-level rounding concludes that rounding is
negligible and is wrong by two orders of magnitude at the container. **The rounding direction
is a policy nobody owns.** Ceiling is safe for service and expensive for inventory, floor is
the reverse, and round-half is defensible for one step but not across nested steps because
the errors do not cancel. In most companies the rule is a default set years ago by someone
who has left.

### 3.5 Mass and volume are coupled by density

The last piece of the model is the one that connects the hierarchy to the vehicle. Mass and
volume are different dimensions, and a container constrains both. The coupling constant is
density, mass per unit volume in kilograms per cubic metre, or its reciprocal, the stowage
factor, volume per unit mass in cubic metres per tonne.

A container has a volume-to-payload ratio: usable volume divided by maximum cargo mass. The
Container Handbook gives 33.5 m³ and 21.75 t for a 20-foot box, a ratio of 1.54 m³/t, and
67.7 m³ and 26.74 t for a 40-foot box, a ratio of 2.53 m³/t. Cargo whose stowage factor
exceeds the ratio is volume-limited, because the box fills before it reaches payload; cargo
below it is weight-limited. Grade C, industry technical reference, undated on the retrieved
page.

This is why a unit-of-measure topic must include density. A plan in eaches, then cases, then
pallets is a count plan, and whether the pallets fit is a question about cubic metres and
kilograms. A count plan that ignores the coupling passes every internal check and fails at
the load port.

## 4. Definitions and vocabulary

The table below is the working vocabulary for the topic. The fourth column names the unit or
coding system in which the term is normally expressed, because a definition without a unit
is the defect this topic is about.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Unit of measure (UoM) | The declared basis in which an amount is expressed | Used to mean the packaging level | UN/CEFACT Rec 20 code, or UCUM expression |
| Dimension | The physical kind of a quantity: count, mass, volume, length, area, time | Used as a synonym for unit | SI base dimensions |
| Base unit | The lowest level at which an item is stocked and counted | Chosen as the case, which makes broken cases unrepresentable | Item-level declaration, often `EA` |
| Each | One countable item | Used for items that are in fact sold by mass | Code `EA` in UN/CEFACT Rec 20 |
| Inner pack | A sealed group of eaches that is neither the consumer unit nor the case | Called a case in systems that have only two levels | GS1 inner pack level |
| Case | The standard shipping container of a single trade item, built from eaches or inner packs | Used interchangeably with carton, shipper, box, outercase | GS1 case level; trade code often `CA` |
| Mixed case | A case containing more than one trade item identifier | Given a single item code and treated as homogeneous | GS1 mixed case level |
| Pallet | A platform plus its stacked cases, moved as one unit | Used to mean any large quantity | GS1 pallet level; `SSCC` |
| Mixed pallet | A pallet carrying more than one trade item identifier | Represented as a single item number | GS1 mixed pallet level |
| Unit load | The pallet or container as moved by handling equipment | Used to mean any shipment | `SSCC` or container number |
| Container | An intermodal freight unit, commonly 20 ft or 40 ft, with 40 ft high cube variants | Used loosely to mean a lot or a batch | ISO container type; TEU/FEU |
| TEU | Twenty-foot equivalent unit, a capacity measure | Used as a count of physical boxes | TEU |
| GTIN | Global Trade Item Number, a GS1 identifier for a trade item at one packaging level | Used to mean the product, ignoring the level | GTIN-8, GTIN-12, GTIN-13, GTIN-14 |
| GTIN-14 | Fourteen-digit GTIN used for cases and pallets, carrying an indicator digit | Confused with the thirteen-digit base-unit GTIN | GS1, indicator digit 1–9 |
| SSCC | Serial Shipping Container Code, an eighteen-digit identifier for a logistics unit | Used as a product identifier | GS1 |
| Case pack quantity | The number of base units inside one case | Confused with cases per pallet | Integer |
| Pack configuration | The full specification of inner pack, case, layer, and pallet | Called "pack size" and reduced to one number | Structured record |
| TI | The number of cases in one pallet layer | Read as "tier", which reverses the meaning | Integer |
| HI | The number of layers stacked on a pallet | Read as a height index rather than a count | Integer |
| Conversion factor | The multiplier relating one unit to another | Stored without its direction or basis | Decimal, with direction |
| Exact conversion | A conversion with a fixed integer ratio, reversible when no integer constraint is applied | Assumed true for every conversion | Integer ratio |
| Variable conversion | A conversion whose factor varies by lot or by unit and requires measurement | Treated as exact because a nominal value exists | Distribution, in kg or lb |
| Lossy conversion | A conversion in which the remainder cannot be represented in the target unit | Overlooked because the software rounds silently | Quotient plus remainder |
| Rounding residue | The quantity added or removed by rounding to an executable unit | Called "waste", "safety", or "buffer" | Base units |
| Catch weight | The measured weight of a variable-weight item, recorded alongside its handling quantity | Used to mean the nominal or standard weight | Kilograms (kg) or pounds (lb) |
| Nominal weight | The declared or standard weight used for planning and specification | Used as the invoicing quantity | kg |
| Actual weight | The measured weight of a specific lot or unit | Used as a planning quantity | kg |
| Tare | The weight of packaging and pallet, excluding product | Confused with net weight | kg |
| Net weight | Product weight excluding all packaging | Confused with gross weight | kg |
| Gross weight | Net weight plus packaging plus pallet | Used as net weight in freight calculations | kg |
| Variable measure | A trade item whose quantity is not fixed by the pack specification | Used to mean "loose" or "unpackaged" | GS1 variable measure; GTIN-14 indicator 9 |
| Density | Mass per unit volume | Used as a synonym for stowage factor, which is its reciprocal | kg/m³ |
| Stowage factor (SF) | Volume occupied per unit mass of cargo | Quoted without stating net or gross basis | m³ per tonne (m³/t) |
| Payload | The maximum cargo mass a container may carry | Confused with gross vehicle or gross container weight | kg |
| Cube | The usable internal volume of a container or a case | Treated as an exact capacity | m³ |
| Chargeable weight | The greater of actual mass and volumetric mass at a stated factor | Ignored in full-container-load ocean freight, where it rarely applies | kg |
| MOQ | Minimum order quantity | Confused with order multiple | Units, stated in a named UoM |
| Order multiple | The increment in which an item may be ordered | Confused with MOQ | Units |
| UCUM | Unified Code for Units of Measure, a case-sensitive syntax for unit expressions | Treated as a finite code list | UCUM expression |
| Application identifier (AI) | A GS1 prefix in a barcode that defines the meaning of the following data | Read as part of the payload | GS1 AI, for example `310n` |
| Decimal dust | Residue accumulated from repeated low-precision conversions | Called "rounding error" without locating it | Base units or kg |

Four rows carry most of the confusion. **Case versus unit of measure:** a case is a count of
eaches bundled by an artefact, not a dimension; the test is whether you can convert it to a
kilogram without knowing the product. **TI and HI:** product-specific values with unstable
terminology — some sources read TI as "tier" and reverse the pair — and standards govern
recording, not the numbers. **Net, tare, and gross:** freight prices on gross, inventory is
often valued on net, and packaging can be a material share of gross weight for glass, cans,
and liquids. **Density and stowage factor:** reciprocals, in kg/m³ and m³/t, and quoting one
without its basis is a common source of factor-of-1,000 errors.

## 5. The mechanics, step by step

This section walks the hierarchy from physical fact to executable transaction in thirteen
steps, grouped into five phases. Each step names the physical fact, the data representation
that should carry it, and the specific failure that occurs when it does not.

### Phase A — Establish the physical hierarchy

**Step 1. Name the levels from the handling artefact, not from the ERP screen.**

- Physical fact: material is bundled into objects that can be picked up, stacked, and moved.
- Data representation: an ordered list of levels, each with a name that a warehouse operator
  would recognise, plus a flag for whether the level is a physical object or a position.
- If skipped: the model inherits whatever four levels the ERP shipped with, and any product
  whose packaging does not fit those four is forced into a wrong shape. A bottle sold
  singly, in six-packs, in cases of four six-packs, on pallets of eighty cases has five
  meaningful levels and an ERP with three.

**Step 2. Record the containment quantity and its basis at every level.**

- Physical fact: each parent contains an integer number of children, or a measured quantity
  of a continuous product.
- Data representation: a containment relation of parent, child, quantity, and basis, where
  basis is count, mass, or volume.
- If skipped: the quantity is stored somewhere convenient — a field on the item master, a
  constant in a report query, a value in a spreadsheet — and diverges from the physical
  truth at the next pack change.

**Step 3. Record which levels are orderable, shippable, storable, and priceable.**

- Physical fact: a level can play several commercial roles at once, and which roles it plays
  is a commercial decision rather than a physical property.
- Data representation: four boolean role flags on each level, not a single "is this the
  unit" flag.
- If skipped: the system assumes the base unit is the ordering unit, and the sales team
  keeps a parallel price list in cases, and the two disagree by the case pack.

The role table below is worth keeping in view, because most unit confusion is a role
collision rather than a conversion error.

| Role | The question it answers | Typical level | Consequence of conflating it with another role |
|---|---|---|---|
| Ordering unit | In what increment may a customer or planner commit? | Case, or pallet for club channels | Order quantity appears to violate MOQ when it does not |
| Handling unit | What does a person or robot pick up at once? | Case, inner pack, or pallet | Labour standards computed on the wrong count |
| Storage unit | What occupies a bin location? | Pallet, case, or each | Cube utilisation and slotting both wrong |
| Pricing unit | What does the price refer to? | Each, kg, or case | Invoice and purchase order disagree in value |
| Reporting unit | In what unit is performance reported? | Case-equivalents or eaches | Trends contaminated by pack changes |

**Step 4. Assign an identifier to every level that is an artefact.**

- Physical fact: supply chain partners need to refer to the same object without ambiguity.
- Data representation: a GTIN per level above the base unit, following the GS1 rule that a
  unique GTIN is required at each packaging level, with the base-unit GTIN unchanged when a
  higher level changes.
- If skipped: the case and the pallet share the product code, and receiving cannot tell
  whether a scanned label means one case or eighty.

The GTIN Management Standard is explicit about the trigger. Changing the number of items in
a case, or the case count in a predefined pallet configuration, requires a new GTIN at that
higher level. The base unit keeps its identifier. This is the standard's way of encoding the
insight that a pack change is a new trade item, not a correction to an old one. The rule is
reported as 2.8 in version 1.1.0 of the standard. Grade B.

### Phase B — Classify the conversions

**Step 5. Enumerate every conversion and give it an integrity class.**

- Physical fact: conversions are performed by people, by the ERP, by the WMS, by planning
  software, and by spreadsheets, and the enumerations differ; some factors are definitions,
  some are measurements, some produce a remainder.
- Data representation: a conversion register with source unit, target unit, factor, basis,
  precision, direction, and class — exact, variable, or lossy — plus a distribution and
  tolerance for variable factors.
- If skipped: the conversion exists in six places at four precisions, a catch-weight factor
  is applied as if it were exact, and inventory value drifts by the average deviation times
  volume every period.

The classification test asks three questions in order. Is the factor fixed by specification?
Then exact. If not, is it measured per lot or per unit? Then variable. If the factor is exact
but the target unit cannot hold the remainder, lossy. One conversion can be more than one:
eaches to cases is exact when the count divides and lossy otherwise, which is why the class
belongs to the operation and not only to the pair.

**Step 6. Declare the dimension and the precision for every quantity.**

- Physical fact: mass, volume, length, and area cannot be added, and a finite-precision store
  truncates a rational factor.
- Data representation: a dimension attribute alongside the unit code, validated at entry from
  a coded system rather than free text, and a stated decimal precision with a documented
  rationale.
- If skipped: a column labelled `weight` holds kilograms in some rows and pounds in others,
  and a stored reciprocal silently gains 0.08% per round trip.

### Phase C — Cascade the plan

**Step 7. Keep the forecast in base units and never round it.**

- Physical fact: a forecast is a belief about demand, and demand is not constrained to
  cases.
- Data representation: forecast stored in eaches with its error distribution intact.
- If skipped: the forecast is rounded at entry, the error is absorbed into the baseline, and
  forecast accuracy is measured against a target that was itself rounded.

**Step 8. Convert upward one level at a time, recording the residue at each level.**

- Physical fact: each level imposes a different granularity.
- Data representation: for each level, the exact quotient, the rounded quantity, the
  rounding rule applied, and the cumulative residue in base units.
- If skipped: only the final number survives, and the planner cannot tell whether the
  overage came from a pack size, a pallet configuration, or a container.

**Step 9. Treat the residue as a quantity with an owner and a cost.**

- Physical fact: overbuilt units are inventory, and inventory consumes cash and space.
- Data representation: a residue record linked to the plan version, with a reason code and a
  valuation.
- If skipped: the residue becomes "normal" overbuild, is absorbed into standard cost, and is
  never attributed to the rounding decision that created it.

### Phase D — Transact in the right unit at each step

**Step 10. Order in the ordering unit, receive in the receiving unit, store in the storage
unit, and price in the pricing unit.**

- Physical fact: these four units are frequently different, and each document is correct in
  its own frame.
- Data representation: every transaction line carries a quantity, a unit code, and a
  conversion path back to the base unit.
- If skipped: the purchase order says 100 cases, the advance ship notice says 2,400 eaches,
  the receipt says 100 cases, and the invoice says 2,400 eaches at a case price. Three of
  the four are right.

**Step 11. For variable-measure items, record both quantities and the basis for each.**

- Physical fact: the handling quantity and the value quantity are different measurements of
  the same lot.
- Data representation: a primary quantity in the handling unit and a secondary quantity in
  the value unit, with the measurement source and the tolerance.
- If skipped: inventory is valued on nominal weight, the invoice is priced on actual weight,
  and the difference is booked to a variance account that nobody reconciles.

**Step 12. Carry the unit code on the transaction, not only on the item master.**

- Physical fact: the item master describes the item today; the transaction describes what
  happened last March.
- Data representation: unit code persisted on the line, immutably.
- If skipped: a pack change in the item master retroactively changes the meaning of every
  historical line, and last year's volume report changes when this year's pack changes.

### Phase E — Reconcile and learn

**Step 13. Run a round-trip test on every conversion, and attribute the residue.**

- Physical fact: a conversion that cannot be inverted loses information, and the loss has a
  cause, usually a pack configuration or a default rounding rule.
- Data representation: an automated test over a range of quantities reporting the maximum
  absolute and relative residue, plus a monthly attribution by level in units and currency.
- If skipped: the defect surfaces at a physical count as an unexplained variance, and the
  organisation concludes that forecasting is inaccurate when the pallet holds eighty cases
  and demand arrives in seventies.

## 6. Worked example

This example is constructed to be internally consistent and is not drawn from a named company;
every conversion, weight, and cost is stated so that you can reproduce the arithmetic.

### 6.1 The product and its hierarchy

A mid-value consumer product, sold through retail and e-commerce, packed in a four-level
hierarchy with a fifth transport level.

| Level | Contents | Units at this level per parent | Identifier | Roles |
|---|---|---:|---|---|
| Each | 1 unit, 0.42 kg net | — | GTIN-13 | Pricing, e-commerce pick |
| Inner pack | 6 eaches | 6 | GTIN-13 | Handling in store |
| Case | 4 inner packs | 24 eaches | GTIN-14 | Ordering, receiving, storage |
| Layer | 10 cases | 10 cases | none | Position only |
| Pallet | 8 layers | 80 cases, 1,920 eaches | SSCC | Shipping, storage |
| Container | 20 pallets, 40 ft high cube | 20 pallets, 38,400 eaches | container number | Transport |

The pack specification is therefore: 6 eaches per inner pack, 4 inner packs per case, 10
cases per layer, 8 layers per pallet, 20 pallets per container.

### 6.2 The dimensional facts

| Fact | Value | Basis and arithmetic |
|---|---:|---|
| Net weight per each | 0.42 kg | Declared |
| Gross weight per case | 10.50 kg | 24 × 0.42 = 10.08 kg product + 0.42 kg packaging |
| Case dimensions | 400 × 300 × 280 mm | 0.400 m × 0.300 m × 0.280 m |
| Case cube | 0.0336 m³ | 0.400 × 0.300 × 0.280 = 0.0336 m³ |
| Case density | 312.5 kg/m³ | 10.50 kg / 0.0336 m³ |
| Pallet footprint | 1200 × 1000 mm | 1.20 m² |
| Loaded pallet height | 2.39 m | 8 × 0.280 = 2.24 m + 0.15 m pallet |
| Pallet gross weight | 865 kg | 80 × 10.50 = 840 kg + 25 kg pallet |
| Pallet gross volume | 2.868 m³ | 1.20 × 1.00 × 2.39 |
| Container occupancy | 20 pallets, 17,300 kg, 57.4 m³ | 20 × 865 = 17,300 kg; 20 × 2.868 = 57.4 m³ |
| Cargo stowage factor | 3.20 m³/t | 1 / 0.3125 = 3.20 m³ per tonne |
| Container volume-to-payload ratio | 2.53 m³/t | 67.7 m³ / 26.74 t, Container Handbook figure |

Two conclusions follow before any plan is built. The pallet fits: 2.39 m is below the internal
height of a 40-foot high cube, and 20 pallets of this footprint occupy the floor in a single
tier. And the cargo is volume-limited: its stowage factor of 3.20 m³/t exceeds the container
ratio of 2.53 m³/t, so the box fills at 57.4 m³ of 67.7 m³ before it approaches the 26.74 t
payload. Payload utilisation is 17,300 / 26,740 = 64.7%, and any plan that reasons in kilograms
alone overstates how much fits by roughly a third.

> **Note:** the 2.53 m³/t ratio uses the Container Handbook's nominal 40-foot figures. High
> cube boxes have more volume and a slightly higher ratio, so the conclusion is robust for
> this cargo but the exact ratio is not portable. Recompute with the container type actually
> booked.

### 6.3 The cascade from 100,000 eaches

Demand for the planning quarter is a forecast of 100,000 eaches. Convert upward one level at
a time, rounding up at each level to the smallest executable quantity.

| Step | Operation | Exact quotient | Rounded | Units at this level | Cumulative eaches | Residue vs forecast | Increment from previous step |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Forecast | — | 100,000 | 100,000 EA | 100,000 | 0 | — |
| 2 | / 6 eaches per inner pack | 16,666.667 PK | 16,667 PK | 16,667 PK | 100,002 | +2 | +2 |
| 3 | / 4 inner packs per case | 4,166.750 CA | 4,167 CA | 4,167 CA | 100,008 | +8 | +6 |
| 4 | / 10 cases per layer | 416.700 LY | 417 LY | 417 LY | 100,080 | +80 | +72 |
| 5 | / 8 layers per pallet | 52.125 PL | 53 PL | 53 PL | 101,760 | +1,760 | +1,680 |
| 6 | / 20 pallets per container | 2.650 CN | 3 CN | 3 CN | 115,200 | +15,200 | +13,440 |

Read the last two columns together. The first four steps together add 80 eaches, which is
0.08% of forecast and would be dismissed by any planner as noise. The pallet step adds
1,680, which is 1.7%. The container step adds 13,440, which is 13.4%. The total is 15,200
eaches, or 15.2% above forecast.

The container step deserves careful reading because it is not really a rounding of the
plan. Fifty-three pallets occupy two full containers and thirteen positions in a third.
Twenty positions remain empty in that third container. The plan is not "115,200 eaches"; it
is "53 pallets, of which 40 fill two boxes and 13 fill 65% of a third". Whether the empty
positions are waste depends on what else could travel, which is a network question. The
count of 115,200 is an artefact of multiplying pallets by their nominal content, not a
decision anybody made.

### 6.4 The residue in three currencies

The 15,200 excess eaches are not a cost in themselves. They are a quantity that becomes
cash, then possibly a cost. Keep the three views separate.

**In units.** 15,200 EA = 633.33 cases = 7.92 pallets. Of the excess, 1,760 eaches (1.7%)
come from the pallet step and 13,440 (13.4%) from the container step. The case-level and
inner-pack-level residue is 80 eaches, or 3.3 cases, which is immaterial.

**In cash.** At a standard cost of €18.50 per each, the excess inventory is:

```text
inventory value = 15,200 EA x EUR 18.50/EA = EUR 281,200
```

That is working capital, not a loss. It becomes a loss only if the units are not sold, or
are sold below cost, or are held long enough for the carrying cost to exceed the margin.

**In profit and loss.** Three components, each with its own trigger.

| Component | Arithmetic | Amount | When it becomes a P&L item |
|---|---|---:|---|
| Carrying cost, 12 weeks at 22% per year | 281,200 × 0.22 × 12/52 | €14,276 | Accrued continuously |
| Freight premium on the third container | 2,850 − (2,850 × 13/20) | €997.50 | On shipment |
| Markdown, if 60% of excess clears at 30% of cost | 0.60 × 281,200 × 0.70 | €118,104 | At end of season |

The carrying cost arithmetic is `281,200 × 0.22 = 61,864` per year, and `61,864 × 12/52 =
14,276` for twelve weeks. The freight premium is the cost of the empty slots: the third
container costs €2,850 whether it carries 13 pallets or 20, so the thirteen pallets carry
`2,850 / 13 = €219.23` each against a full-box rate of `2,850 / 20 = €142.50` each. The
difference per pallet is €76.73, and over thirteen pallets it is €997.50.

> **Note:** the markdown line is a scenario, not a forecast. It is included because it is
> usually larger than the other two combined, and because it is the line that a plan built
> on standard cost will never show. If the product has a twelve-month shelf life and the
> excess can be carried into the next season, the markdown exposure falls and the carrying
> cost rises.

The honest summary is: the rounding residue ties up €281,200 of cash, costs about €15,300 in
carrying and freight before any markdown, and carries a markdown exposure of roughly
€118,000 in the downside case. A planning meeting that discusses only the 15.2% volume
overage is discussing the least important of the three numbers.

### 6.5 The double-conversion check

The example has so far used exact integer arithmetic. Now reproduce what a system does when
it stores a reciprocal.

The forward factor is exactly 24 eaches per case. Suppose a developer stores the reverse
factor as `0.0417` to four decimal places, because that is what the field allows. One round
trip on the planned quantity:

```text
115,200 EA x 0.0417 = 4,803.84 CA
4,803.84 CA x 24     = 115,292.16 EA
residue              = +92.16 EA per round trip
relative residue     = 92.16 / 115,200 = 0.0008 = 0.08%
```

One round trip is harmless. The problem is that nightly reconciliation jobs perform the
round trip and write the result back. After ten passes:

```text
115,200 x (1.0008)^10 = 115,200 x 1.008032 = 116,125 EA
cumulative residue    = +925 EA
```

Nine hundred and twenty-five eaches is 38.5 cases, or about €17,100 at standard cost, created
by a stored constant rather than by any physical event. The correct design stores the
integer ratio and the direction, and computes the reciprocal at full precision at the moment
of use, or refuses to invert at all when the intermediate is constrained to an integer.

There is a second, different failure in the same test. If the intermediate is constrained to
a whole case, the round trip fails whenever the quantity is not divisible by 24, regardless
of factor precision. The residue is bounded by the rounding rule: up to 23 eaches for ceil,
up to 11.5 for round-half. The two failures have different causes and different fixes, and a
test that does not separate them will send you to the wrong one.

## 7. Code, query, or calculation

The section uses one short procedure, because the arithmetic in section 6 is the point and
the tooling is not. The procedure below is language-neutral pseudocode. It makes three
decisions explicit that a normal conversion function leaves implicit: the rounding rule,
the residue accumulator, and the comparison denominator.

```text
demand_ea     = 100000
pack_size     = 6      -- eaches per inner pack
case_pack     = 4      -- inner packs per case
cases_per_lyr = 10
lyrs_per_plt  = 8
plts_per_cntr = 20
steps         = [pack_size, case_pack, cases_per_lyr, lyrs_per_plt, plts_per_cntr]

qty = demand_ea
for divisor in steps:
    exact    = qty / divisor          -- keep full precision
    rounded  = ceil(exact)            -- one rule, applied once, stated
    residue  = rounded * divisor - qty
    qty      = rounded
    emit(divisor, exact, rounded, residue, qty)
```

Three notes. `residue` is computed against the quantity entering the step, not the original
forecast, so the residues in section 6.3 are increments rather than totals. The loop divides
the rounded quantity, because that is what a planner does; carrying the exact fraction
forward is a different policy with a smaller residue. And `ceil` is a parameter in any real
implementation, because ceiling, floor, and round-half are commercial decisions.

The table below converts the procedure into a test specification. Run each case against your
implementation before you trust it on a planning cycle.

| Test | Input | Expected behaviour | What a failure indicates |
|---|---|---|---|
| Exact division | 120,000 EA through pack size 6 | Residue 0 at every level below case | — |
| Single surplus unit | 100,001 EA through pack size 6 | 16,667 PK, residue +1 EA | Correct ceiling |
| Just below a boundary | 119,999 EA through 24 | 5,000 CA, residue +1 EA | Ceiling applied at the right level |
| Exact pack boundary | 100,008 EA through 24 | 4,167 CA, residue 0 | No spurious round-up |
| Round trip, divisible | 115,200 EA to CA and back | 115,200 EA, residue 0 | — |
| Round trip, indivisible | 100,000 EA to CA and back | 100,008 EA, residue +8 | Expected; not a defect |
| Reciprocal precision | 115,200 EA via stored 0.0417 | 115,292.16 EA, +0.08% | Stored reciprocal, not integer rounding |
| Zero demand | 0 EA | 0 at every level, no division error | Guard clause present |
| Fractional input | 100,000.5 EA | Rejected or explicitly floored | Silent truncation is a defect |

Two tests are usually missing from production systems: the reciprocal precision test,
because the stored factor and the integer ratio live in different places, and the zero-demand
test, because a division by zero in a nightly job is discovered by its absence.

A final validation needs no code block: for each row of the conversion table, compare the
stored decimal factor with the exact rational it claims to represent and sort by the absolute
difference. Where no rational form is stored, the audit cannot be run at all, which is itself
the finding.

## 8. Data, standards, and regulatory context

Four bodies of standard touch this topic: identification, unit coding, variable measurement,
and transport. Knowing which one governs which field tells you what the data will look like
before you inspect it, and tells you which fields are safe to validate automatically.

### 8.1 Identification and packaging levels

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 General Specifications, packaging levels | The enumeration of trade item levels | Each, inner pack, case, mixed case, pallet, mixed pallet, display shipper, case as each | B |
| GS1 GTIN Management Standard v1.1.0 | When a new identifier is required | A new GTIN at a level when the pack or case quantity, or the predefined pallet configuration, changes at that level | B |
| GS1 Package and Product Measurement Standard v3.3 | How dimensions and weights are recorded | Measurement convention by trade item type, rounding of dimensions, tolerances | B |
| GS1 SSCC | Logistics unit identity | The pallet or container label, distinct from the product label | B |

The GTIN rule is the most consequential for analytics. A pack change is not an update; it is
a new trade item at that level. The GS1 Package and Product Measurement Standard, version
3.3, sets measurement conventions and rounding rules — dimensions in millimetres rounded up
to whole millimetres, inches to two decimal places at 25.4 mm per inch, and a conversion
convention of 0.454 kg per pound. Grade B.

> **Note:** the 0.454 kg per pound convention is a rounding convention for recorded
> measurements, not a definition of the pound. The exact international avoirdupois pound is
> 0.45359237 kg. Using 0.454 in a conversion chain introduces an error of about 0.09% per
> conversion, which is the same order as the double-conversion residue in section 6.5 and
> compounds in the same way.

### 8.2 Unit coding

| System | Governs | What it constrains | Grade |
|---|---|---|---|
| UN/CEFACT Recommendation 20 | Codes for units of measure used in international trade | Three-character codes organised by quantity, by name, and by common code | B |
| UN/CEFACT Recommendation 21 | Codes for packaging types | Packaging units kept separate from units of measure, prefixed `X` when used as a unit | B |
| GS1 EANCOM data element 6411 | Unit of measure in EDI messages | Points implementers to Recommendation 20 | B |
| UCUM | Syntax for unit expressions | Case-sensitive atoms and prefixes, `.` multiplication, `/` division, `{}` annotation with no semantics | B |
| HL7 terminology and ISO 11240 | Health-sector unit representation | UCUM as the coded system, with the identifier `http://unitsofmeasure.org` | B |

The split between Recommendations 20 and 21 matters more than it appears. A case is a
packaging type, and packaging types live in Recommendation 21. A kilogram is a unit of
measure, and units live in Recommendation 20. Systems that keep one field for both are
mixing two standards. The practical symptom is a code list in which `KGM` and `CA` sit
side by side with no indication that they answer different questions.

The specific codes are worth checking rather than assuming. The Recommendation 20 annexes
returned in the research for this topic list `EA` as each, `CA` as can, and `PL` as pail,
with packaging units maintained under Recommendation 21. Trade systems use `CA` for case and
`PL` for pallet almost universally. The collision is a documented ambiguity, not a rumour,
and it is one more reason to store the code-list identifier next to the code.

### 8.3 Variable measure

| Mechanism | Governs | What it constrains | Grade |
|---|---|---|---|
| GTIN-14 with indicator digit 9 | Variable-measure trade items | Identifies the item as variable measure at the case level | B |
| GS1 Application Identifier 30 | Variable count | The count of items in a variable-measure trade item | B |
| GS1 Application Identifier 310n | Net weight in kilograms | `n` is the number of implied decimal places; the value is measured, not nominal | B |
| GS1 Application Identifier 320n | Net weight in pounds | The same quantity in the imperial convention | B |
| GS1 US Data Hub variable-measure guide | Implementation of the above | Which packaging levels may carry variable measure | B |

The application identifiers are the mechanism by which the second number travels with the
goods. A case label carrying `3103` and `009940` states a net weight of 9.940 kg. A receiving
system that reads only the case count and looks up a nominal weight in the item master has
discarded the measurement the supplier took the trouble to encode.

### 8.4 Transport and stowage

| Requirement | Where | Effect on the hierarchy | Grade |
|---|---|---|---|
| IMSBC Code, amendment 07-23 | International maritime solid bulk | Defines stowage factor and bulk density, requires shipper declaration, sets a high-density threshold at 0.56 m³/t | B |
| Container Handbook, stowage planning | Industry technical reference | Volume-to-payload ratios, net and gross stowage factor, line load limits | C |
| Carrier payload and gross mass limits | Commercial and statutory | Caps the top of the hierarchy regardless of cube | B |

The transport layer is where the hierarchy stops being a count structure and becomes a
physical one. A count plan says 53 pallets. The stowage calculation says those pallets
occupy 57.4 m³ of a 67.7 m³ box and weigh 17.3 t of a 26.7 t payload. Both statements are
true, and only the second determines whether the freight is efficient.

### 8.5 What is convention rather than requirement

Several rules that feel like physics are habits. "A pallet holds a thousand kilograms" is a
planning heuristic; actual capacity depends on pallet, load, racking, and carrier. "A case
holds twelve" is a retail convention with no physical necessity. TI and HI are product-specific,
and standards prescribe only how they are recorded. Rounding up rather than down is a policy
default that is rarely reviewed. "The base unit is the each" is true for discrete goods and
false for process goods, where the base unit is often the kilogram or the litre. An analyst who
can separate a requirement from a convention can ask why the plan rounds up at the pallet and
answer with a cost rather than a shrug.

## 9. Industry practice and case evidence

### 9.1 The best-documented unit failure is not commercial

The Mars Climate Orbiter loss is the canonical case because the investigation was public and
the mechanism was stated precisely. The Mishap Investigation Board's Phase I report,
published 10 November 1999, attributes the loss to the failure to use metric units in a
ground software file named `Small Forces`, which supplied thruster impulse in pound-force
seconds where the Software Interface Specification required newton-seconds. The navigation
model understated the effect by a factor of 4.45. The spacecraft's trajectory at insertion
was approximately 170 km lower than planned, with an estimated periapsis near 57 km against
a survivable minimum near 80 km. Grade B.

Three features of the case transfer directly to commercial analytics. The arithmetic in both
programs was correct, so no unit test on either program would have caught it. The interface
specification existed and was not followed, so the defect was a governance failure rather
than a documentation failure. And the error accumulated slowly enough to look like a
modelling discrepancy for months, so the investigation initially pursued trajectory models
rather than units. Every one of those features appears in a warehouse that has been converting
pounds to kilograms with a truncated factor for three years.

### 9.2 Pack data quality in retail and consumer goods

A GS1 Australia case study with Kimberly-Clark reports weights missing or wrong in the ERP
for nearly 100% of audited units and 1,526 packaging levels with identifier or dimension
errors (Grade C, company-reported). GS1 Canada's TI/HI guidance and an Australian Food and
Grocery Council supplier guide document the operational consequences — receiving rejections,
slot resets, automation faults — and confirm that no universal numeric TI/HI standard exists:
standards govern recording, not configuration (Grade B for the guidance, C for the stated
consequences). The Kimberly-Clark figure rests on a sample the summary does not describe, so
it establishes that pack-data defects survive in large organisations and are found by audit,
not a population defect rate.

### 9.3 Catch weight as a design pattern

Multiple ERP vendors document a dual-quantity design for variable-weight items, and the
specifics are instructive even though the sources are marketing or product documentation.

- SYSPRO 8 (2026 R2) documentation states that transactions carry both the order or
  manufacturing quantity and the actual stock quantity, and that the system deliberately
  does not validate that stock quantity equals order quantity multiplied by a conversion
  factor. Its worked example is a box with a nominal factor of 10 kg that actually weighs
  11 kg, recorded as 11 kg with a 110% yield against the 10 kg estimate. Grade D, vendor
  documentation, 2026.
- Infor CloudSuite Distribution documentation describes an actual total weight entered at
  receipt or shipping, with the conversion recalculated for valuation. Its example is two
  cases at 50 lb nominal, estimated at $40, actually 90 lb and $36.00 at the stated rate.
  Grade D, vendor documentation, 2026.
- Aptean's catch-weight documentation defines a base unit of measure and a separate catch
  weight unit of measure, with nominal weight used for tolerance checking. Grade D, vendor
  documentation, 2026.

The convergence across three independent products is the evidence. When several vendors
independently refuse to enforce a single conversion factor, the single-factor model is not
a simplification; it is a known error. The vendors' worked examples also show the magnitude:
a 10% weight deviation on a nominal 10 kg box, and a 10% value deviation on a nominal 50 lb
case, are both ordinary rather than exceptional.

### 9.4 Rounding residue in ERP systems

A practitioner blog on the SAP Community describes kilogram-to-pound residues of roughly
0.0004 kg per conversion, producing blocked bins and stock inconsistencies, and cites SAP
Note 3138922 with a posting threshold of 0.0005. Grade D, retrieved 2026. One pound is
0.45359237 kg; converting a million kilograms a year at a stored factor of 0.4536 instead
introduces an error of about 0.0017%, or 17 kg per million. That is small. The problem is the
residue crossing a posting threshold, after which the system creates a document, and the
documents accumulate faster than anyone reconciles them.

### 9.5 What the evidence does not show

None of these sources measures the frequency of unit errors in a representative population of
companies. Case studies are selected because they are dramatic, vendor documentation because
it describes a product, and standards because they state what should happen. Treat the
evidence as establishing mechanisms and plausible magnitudes, not base rates; the one
quantitative claim with a defensible denominator is the Mars case, and it is a sample of one.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed units in one quantity column | Totals change when a filter is applied; a category total looks implausible | Unit stored in the header or nowhere; rows carry different bases | Any aggregate is wrong by an unknown factor | Add a unit code to every row; validate on ingest |
| Base unit chosen as the case | Requests for partial cases cannot be represented | Granularity set at a level that is not physically divisible in use | Rounding forced at the wrong level; phantom stock-outs | Store the each as base, convert upward for ordering |
| Forecast rounded at entry | Forecast accuracy is suspiciously good on slow movers | Rounding absorbed into the baseline | Model learns the rounding rule, not the demand | Keep forecast in base units; round only at plan |
| Ceiling applied at every level | Plan consistently exceeds forecast by 2–20% | Independent round-ups that do not cancel | Chronic overbuild, working capital, markdown | Round once at the binding level; carry exact fractions |
| Double conversion with a stored reciprocal | Inventory drifts in one direction only | Truncated factor applied repeatedly and written back | Slow, unattributable inventory growth | Store integer ratios; test round trips |
| Nominal weight used as actual | Invoice variance recurs on the same item | One field for a two-number quantity | Over- or under-payment; bad inventory value | Dual quantity with measurement source |
| Catch-weight receipt with one number | Three-way match passes but value is wrong | Case count substituted for measured mass | Silent value error in inventory and cost | Require actual weight at receipt for variable items |
| Conversion factor truncated to few decimals | Discrepancies appear only at volume | Precision below the required tolerance | Threshold breaches, blocked documents | Declare precision; store rationals where possible |
| Pack change without a new identifier | Historical volume jumps in the month of the change | Same code reused for a different pack | False trend, broken comparability | Enforce GTIN rule at change control |
| TI/HI mismatch between master data and physical pallet | Receiving rejects pallets that "should" fit | Configuration recorded as a guess | Rejections, rework, automation faults | Audit a sample physically each quarter |
| Mixed pallet modelled as a single item | Inventory of components cannot be traced | Graph represented as a tree | Wrong allocation, wrong costing | Containment relation with quantity per child |
| Count plan without a density check | Load plan passes internally, fails at the port | Mass and volume treated as independent | Rolled cargo, rework, demurrage | Compute stowage factor; compare with the container ratio |

Two deserve expansion. **Mixed units in one column** is the most damaging, because it survives
every check that does not know the unit: a `qty` column ranging from 1 to 40,000 looks like a
long tail, not an error. The diagnostic is distributional — plot `qty` by item and look for
spikes at 6, 12, 24, and 48. **Ceiling at every level** is not a coding error; each round-up is
defensible, but the errors are multiplicative rather than offsetting. The mitigation is a
decision about which level binds: if the container is the constraint, round once there and let
the lower levels stay fractional in the plan.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The pack specification is stable over the planning horizon | Conversions are applied as constants | Pack changes, promotional packs, supplier substitution | Compare pack records across versions; flag changes in-period |
| The base unit is the finest granularity the business transacts | Conversions are exact upward from it | Broken-case sales, repacking, kitting | Check for transactions with fractional base quantities |
| Each level contains a single trade item | The hierarchy can be a tree | Mixed cases, mixed pallets, display shippers | Inspect a sample of physical pallets against the item master |
| Nominal weight equals actual weight for fixed-weight items | One field suffices | Fill tolerance, moisture loss, baking loss | Compare weighed receipts against nominal over a month |
| Conversion factors are exact rationals | Round trips are reversible | Variable measure, unit-of-measure redefinition | Run round-trip tests across a quantity range |
| Container capacity is the nominal volume and payload | The load plan can be computed | Pallet overhang, load-bearing limits, line load | Check against a loading plan or a physical load |
| Demand is expressed in the same unit as the forecast | The cascade starts from one number | Forecasts in cases, orders in pallets, sales in eaches | Trace one number end to end through the systems |
| The item master describes the item as it was when the transaction occurred | History is comparable | Master data updated in place | Look for pack changes with no version history |
| Identifier changes accompany packaging changes | Aggregation by GTIN is valid | Pack change with a reused code | Compare GTIN set across two years |
| The rounding rule is the same in every system | Residues reconcile | Planning, ERP, and WMS use different defaults | Compare outputs for the same input |

Three limits are worth stating. The topic treats the hierarchy as given and does not ask
whether the pack configuration is well designed; that question belongs to Topic 03. It
assumes discrete goods, whereas for liquids, powders, and bulk solids the base unit may be
the kilogram or the cubic metre and the variable class becomes the norm. And the evidence
establishes mechanisms and dramatic cases, not base rates, so a reader who needs a defect
rate in their own organisation must measure it.

## 12. Exercises

### Exercise 1 — Reconstruct a hierarchy (recall and application, 30 minutes)

Choose a packaged product in your home. Reconstruct its full hierarchy from each to the
largest unit in which you have seen it sold, and identify the role of each level.

**Deliverable:** a table with level, contents, count per parent, identifier (if any), and
the four role flags from section 5.
**Acceptance criteria:** at least four levels; the containment count stated as an integer at
every level; at least one level identified as a position rather than an object; one sentence
stating which level you would choose as the base unit and why.
**Time:** 30 minutes.

### Exercise 2 — Run the cascade (calculation, 45 minutes)

A product is packed 24 eaches per case, 12 cases per layer, 6 layers per pallet, and 24
pallets per 40-foot container. Forecast demand is 250,000 eaches. Compute the full cascade
to containers under a ceiling rule, recording the exact quotient, the rounded quantity, and
the cumulative residue at each level. Standard cost is €1.85 per each. Full-container freight
is €2,600 and LCL is €110 per cubic metre with a €250 minimum; one pallet occupies 2.22 m³.

**Deliverable:** the cascade table, the total residue in eaches, cases, and pallets, the
inventory value of the residue, and a comparison of 7 full containers against 6 full
containers plus one LCL pallet.
**Acceptance criteria:** every division shown; residues consistent with the quantities at
each level; the two freight options compared on total cost, transit time, and handling
events; the recommendation stated with the condition under which it reverses.
**Time:** 45 minutes.

### Exercise 3 — Build a conversion register (design, 40 minutes)

For the product in Exercise 1, enumerate every conversion the business would perform and
classify each as exact, variable, or lossy. Include at least one conversion in each class.

**Deliverable:** a register with source unit, target unit, factor, basis, precision, class,
and the evidence that supports the class.
**Acceptance criteria:** at least eight conversions; each class justified rather than
asserted; at least one conversion classified as lossy *and* exact depending on the input, with
the condition stated; the precision column filled for every row.
**Time:** 40 minutes.

### Exercise 4 — Design the round-trip test (application, 40 minutes)

Write a test specification for the conversions in Exercise 3 that distinguishes three causes
of round-trip failure: integer rounding at an intermediate step, a truncated reciprocal
factor, and a genuinely variable factor.

**Deliverable:** a table of test cases with input, expected output, expected residue, and the
cause each failure would indicate, plus a statement of the maximum acceptable residue per
conversion.
**Acceptance criteria:** at least eight cases; the three causes separated by distinct
signatures; one case at zero and one at a boundary; the acceptance threshold expressed as a
relative number, not an absolute one, for at least the count conversions.
**Time:** 40 minutes.

### Exercise 5 — Reconcile a catch-weight receipt (calculation, 40 minutes)

A supplier ships 480 cases of a variable-weight product with a nominal weight of 12.5 kg per
case, priced at €4.85 per kg. The purchase order value is calculated on nominal weight. The
actual received mass is 5,952 kg. The contract tolerance is ±1.0% on weight and ±0.5% on
value.

**Deliverable:** the purchase order value, the invoice value, the difference in kilograms,
euros, and percentage, a statement of whether the invoice should be paid as presented, and
the number of fixed 12.5 kg sellable units the receipt supports.
**Acceptance criteria:** all arithmetic shown; the weight and value tolerances evaluated
separately; the distinction between a purchase order estimate and a commitment stated
explicitly; the sellable-unit count derived from the actual mass, not the nominal.
**Time:** 40 minutes.

### Exercise 6 — Compute the binding constraint (calculation, 30 minutes)

A bulk cargo has a stowage factor of 1.25 m³ per tonne. A vessel hold offers 10,000 m³ of
which 90% is usable, and the cargo allocation for this port is 6,500 tonnes. Compute the
volume-limited mass, the mass actually loadable, and the governing constraint.

**Deliverable:** the arithmetic, the governing constraint named, the occupied volume as a
percentage of usable volume, and the bulk density in kg/m³.
**Acceptance criteria:** usable volume computed first; the comparison made in the correct
direction; bulk density shown as the reciprocal of stowage factor; a one-sentence statement
of what would change if the allocation rose to 8,000 tonnes.
**Time:** 30 minutes.

### Exercise 7 — Model a mixed pallet (design, 60 minutes)

A distribution centre builds pallets that carry up to four different trade items, and a
retail customer orders in pallets. Design the data structure that represents this, given that
the same physical pallet must be storable, shippable, and orderable.

**Deliverable:** a schema of at least five tables or entities with keys and cardinalities, a
worked example of one mixed pallet with three items, and a statement of how inventory value
is computed for a partially picked pallet.
**Acceptance criteria:** the tree-and-graph distinction handled explicitly; the ordering role
distinguished from the handling role; a quantity per child on the containment relation; the
valuation rule stated for a pallet that is broken mid-pick.
**Time:** 60 minutes.

### Exercise 8 — Audit a real quantity column (application, 60 minutes)

Obtain any dataset with a quantity column — a public retail dataset, an internal extract, or
a constructed one. Determine what the quantity actually measures. Do not ask the data owner
first; reconstruct the answer from the data, then confirm.

**Deliverable:** a one-page audit with the distribution of quantities by item, the modes
observed, the pack sizes inferred, the proportion of rows whose unit is unambiguous, and a
written definition of the column.
**Acceptance criteria:** the inference method stated and reproducible; at least one item
shown where the unit is ambiguous; the definition written as a data contract sentence of the
form "quantity is the number of X, measured in Y, at the time of Z"; a named follow-up
question for the data owner.
**Time:** 60 minutes.

## 13. Self-check answers

### Exercise 2 arithmetic

The cascade, with 250,000 eaches as the starting quantity.

| Step | Operation | Exact | Rounded | Cumulative eaches | Residue |
|---|---|---:|---:|---:|---:|
| 1 | Forecast | — | 250,000 EA | 250,000 | 0 |
| 2 | / 6 per inner pack | 41,666.6667 | 41,667 PK | 250,002 | +2 |
| 3 | / 4 per case | 10,416.75 | 10,417 CA | 250,008 | +8 |
| 4 | / 12 per layer | 868.0833 | 869 LY | 250,272 | +272 |
| 5 | / 6 per pallet | 144.8333 | 145 PL | 250,560 | +560 |
| 6 | / 24 per container | 6.0417 | 7 CN | 290,304 | +40,304 |

Verification: step 2, `41,667 × 6 = 250,002`. Step 3, `10,417 × 4 = 41,668` inner packs.
Step 4, `869 × 12 = 10,428` cases. Step 5, `145 × 6 = 870` layers. Step 6, `7 × 24 = 168`
pallets, or `12,096` cases, or `290,304` eaches.

The residue is 40,304 eaches, which is 16.1% of forecast, or 1,679.3 cases, or 23.3 pallets,
or €74,562 of inventory at €1.85 per each.

The freight comparison: 7 full containers cost `7 × 2,600 = €18,200`. Six full containers
plus one LCL pallet cost `6 × 2,600 = €15,600` plus a €250 minimum charge for 2.22 m³ at
€110 per m³, or €15,850 — a saving of €2,350, or 12.9%. The recommendation is to ship the
LCL pallet if the transit penalty of roughly 7–14 days, one consolidation, and one extra
handling event is worth less than €2,350, and to reverse the decision if the LCL rate exceeds
about €215 per pallet including minimums, if a single bill of lading is required, or if the
forecast turns out to be over-stated.

### Exercise 4 expected test outcomes

The three signatures are distinct. Integer rounding on a divisible input returns residue zero
(`115,200 / 24 = 4,800`); a failure there is a factor error, not a rounding error. On an
indivisible input it returns a residue bounded by 23 under ceiling (`250,000 / 24 =
10,416.6667`, rounded to 10,417, residue +8) and always non-negative. A truncated reciprocal
returns a residue even on divisible inputs and scales linearly with quantity (`115,200 ×
0.0417 = 4,803.84`, residue +92.16, or 0.08%). A variable factor changes sign across lots and
correlates with the measured moisture or fill; that is correct behaviour, not a defect.
Thresholds should be relative: one pack minus one for counts, the invoicing tolerance for mass.

### Exercise 5 arithmetic

```text
PO mass        = 480 CA x 12.5 kg/CA = 6,000 kg
PO value       = 6,000 kg x EUR 4.85/kg = EUR 29,100.00
Actual mass    = 5,952 kg
Actual average = 5,952 / 480 = 12.40 kg per case
Mass variance  = 5,952 - 6,000 = -48 kg  (-0.80%)
Invoice value  = 5,952 x 4.85 = EUR 28,867.20
Value variance = 29,100.00 - 28,867.20 = EUR 232.80  (-0.80%)
```

The weight tolerance is ±1.0% and the deviation is −0.80%, so the shipment is within
tolerance. The value tolerance is ±0.5% and the deviation is −0.80%, so a value-keyed rule
would raise a query. The value-keyed rule is the wrong test: the purchase order value was
computed from a nominal weight and was an estimate, while the contract prices actual
kilograms. Pay the invoice as presented and book the €232.80 as the difference between
estimated and actual value, not as a supplier error.

The sellable-unit consequence is the part that most analyses miss. If the product is
repacked into fixed 12.5 kg units, the receipt supports `5,952 / 12.5 = 476.16`, so 476
sellable units with 2 kg of residual material. That is a shortfall of four units against the
480 cases ordered and received. The case count says the order is complete; the mass says it
is four sellable units short. Both statements are correct, and a system with one field can
express only one of them.

### Exercise 6 arithmetic

```text
usable volume        = 10,000 m3 x 0.90 = 9,000 m3
volume-limited mass  = 9,000 m3 / 1.25 m3/t = 7,200 t
cargo allocation     = 6,500 t
governing constraint = weight (6,500 t < 7,200 t)
occupied volume      = 6,500 t x 1.25 m3/t = 8,125 m3
utilisation          = 8,125 / 9,000 = 90.3%
bulk density         = 1 / 1.25 = 0.80 t/m3 = 800 kg/m3
```

If the allocation rose to 8,000 t, the volume-limited mass of 7,200 t would bind and only
7,200 t could be loaded, occupying the full 9,000 m³. The constraint would flip from weight
to volume at an allocation of 7,200 t, and the 800 t above that would be shut out.

### Design rubrics, and a note on marking

A strong Exercise 1 submission names the inner pack and the case separately, identifies the
layer as a position with no identifier, and chooses the base unit by what is physically
divisible in use. A weak one collapses inner pack and case into one level, which makes
broken-case planning impossible. A strong Exercise 7 submission uses a containment relation
with quantity per child and a role attribute, and values a partially picked pallet by summing
its children; a weak one gives the mixed pallet a single item code and cannot say what is on
it. A strong Exercise 8 audit shows the quantity histogram by item and the proportion of rows
with an ambiguous unit; a weak one reports summary statistics only, which cannot distinguish
units from a long tail.

Award no credit for a cascade that reaches the right final number by an unstated rounding
rule. What matters is that the exact quotient, the rounded quantity, and the residue are
visible at every level, and that the residue is priced in all three currencies.

## 14. Interview questions

**1. "What is a unit of measure?"**

Strong: separates dimension from magnitude, gives examples in two dimensions, and notes that
a packaging level is not a unit. Weak: says "how you measure something" and stops.

**2. "Is a case a unit of measure?"**

Strong: says no, it is a count of eaches bundled by a physical artefact, and explains that
the case pack belongs in the pack specification while the dimension belongs in the unit
field. It may note that trade code lists conflate the two. Weak: says yes, because you can
count cases, which is true but does not answer the question.

**3. "A forecast is in eaches and the plan is in cases. Where do you round?"**

Strong: keeps the forecast unrounded, converts at the plan step, states the rounding rule,
records the residue, and names the binding level. Weak: divides by the case pack and rounds,
without distinguishing the two artefacts or noticing the residue.

**4. "What is a double-conversion bug, and how would you test for it?"**

Strong: describes a round trip through an intermediate representation, separates integer
rounding from truncated reciprocals, gives the test design, and states the expected residue
bound. Weak: says "rounding errors accumulate" without a mechanism or a test.

**5. "How would you store a catch-weight product?"**

Strong: two quantities, one in the handling unit and one in the value unit, with the
measurement source and tolerance; the base unit is the handling unit; the invoice prices the
measured mass. Weak: picks one unit and converts, which loses either the physical count or
the value.

**6. "Why not store everything in kilograms and convert at the end?"**

Strong: notes that mass is undefined for countable goods in a useful way, that kilogram
conversions are themselves variable for some products, that the base unit must be the finest
granularity transacted, and that converting at the end moves the residue rather than removing
it. Weak: says kilograms are more precise, which is a category error.

**7. "A pack changes from 12 to 10 per case. What has to change in the data?"**

Strong: a new GTIN at the case and pallet levels, a new conversion factor with an effective
date, transaction-level unit codes so history is not rewritten, and a break in any trend
series that crosses the change. Weak: updates the case pack field, which silently
reinterprets every historical row.

**8. "What is stowage factor, and when does it bind?"**

Strong: volume per unit mass, compared with the container's volume-to-payload ratio; cargo
above the ratio is volume-limited. Gives a number and a direction. Weak: defines density,
which is the reciprocal, and does not compare it to anything.

**9. "A supplier invoices 0.6% below the purchase order value on a variable-weight item. Do
you pay?"**

Strong: separates weight tolerance from value tolerance, recognises that a purchase order
value based on nominal weight is an estimate, and pays against measured mass. Weak: applies a
generic percentage tolerance without asking what the percentage is measuring.

**10. "You inherit a table with a column called `qty`. What do you do first?"**

Strong: reconstructs the unit from the distribution, looks for modes at pack sizes, checks
the row count affected by ambiguity, and writes a definition before using the column. Weak:
sums it to check for nulls, which tells you nothing about the unit.

## 15. Key terms

- **Unit of measure (UoM)** — the declared basis in which an amount is expressed.
- **Dimension** — the kind of quantity: count, mass, volume, length, area, time.
- **Base unit** — the lowest level at which an item is stocked and counted.
- **Each** — one countable item; the usual base unit for discrete goods.
- **Inner pack** — a sealed group of eaches between the consumer unit and the case.
- **Case** — the standard shipping container of a single trade item.
- **Pallet** — a platform plus its stacked cases, moved as a unit load.
- **Container** — an intermodal freight unit, commonly 20 ft or 40 ft high cube.
- **GTIN** — Global Trade Item Number, assigned per trade item per packaging level.
- **GTIN-14** — the fourteen-digit form used for cases and pallets.
- **SSCC** — Serial Shipping Container Code, identifying a logistics unit.
- **Case pack quantity** — the number of base units in a case.
- **Pack configuration** — the full specification of inner pack, case, layer, and pallet.
- **TI / HI** — cases per layer and layers per pallet; their product is cases per pallet.
- **Exact conversion** — a fixed integer ratio, reversible without loss when unconstrained.
- **Variable conversion** — a factor that varies per lot or unit and requires measurement.
- **Lossy conversion** — a conversion whose remainder cannot be represented in the target unit.
- **Rounding residue** — the quantity added or removed by rounding to an executable unit.
- **Catch weight** — the measured weight of a variable-weight item, recorded beside its count.
- **Nominal weight** — the declared weight used for planning and specification.
- **Tare / net / gross** — packaging weight, product weight, and their sum.
- **Variable measure** — a trade item whose quantity is not fixed by the pack specification.
- **Density** — mass per unit volume, in kilograms per cubic metre.
- **Stowage factor** — volume per unit mass, in cubic metres per tonne; the reciprocal of
  density.
- **Payload** — the maximum cargo mass a container may carry.
- **UCUM** — Unified Code for Units of Measure, a case-sensitive syntax for unit expressions.
- **Application identifier** — a GS1 barcode prefix that defines the meaning of the data
  following it.
- **Decimal dust** — residue accumulated from repeated low-precision conversions.

## 16. Sources

**Peer-reviewed and academic**

- Bortfeldt, A., and Wäscher, G. (2012). "Container Loading Problems — A State-of-the-Art
  Review." Working paper, Otto-von-Guericke-University Magdeburg. Reported constraint
  frequencies: orientation about 70%, stability about 37%, stacking about 15%, weight limits
  about 14%, weight distribution about 12%. Grade A.

**Standards, official, and regulatory**

- GS1, *GTIN Management Standard* v1.1.0. Pack and case quantity rule, reported as 2.8. Grade B.
- GS1, *General Specifications*, current edition, and GS1 US packaging-level guidance: each,
  inner pack, case, mixed case, pallet, mixed pallet, display shipper, case as each. Grade B.
- GS1, *Package and Product Measurement Standard* v3.3.0. Measurement conventions, dimensional
  rounding, tolerances. Grade B.
- GS1 US, *Data Hub: Working with Variable Measure*. GTIN-14 indicator 9; application
  identifiers 30, 310n, 320n. Grade B.
- GS1, EANCOM 2002, data element 6411, pointing to UN/CEFACT Recommendation 20. Grade B.
- UN/CEFACT, Recommendation 20, *Codes for Units of Measure Used in International Trade*,
  Rev. 14 (2020); Recommendation 21, *Codes for Types of Cargo, Packages and Packaging
  Materials*. Grade B.
- Schadow, G., and McDonald, C. J., *Unified Code for Units of Measure (UCUM)*, ucum.org;
  HL7 Terminology UCUM page; ISO 11240 on unit representation in health informatics. Case
  sensitivity, base units, annotation syntax. Grade B.
- IMO, *International Maritime Solid Bulk Cargoes (IMSBC) Code*, amendment 07-23. Stowage
  factor, bulk density, high-density threshold at 0.56 m³/t. Grade B.
- NASA, *Mars Climate Orbiter Mishap Investigation Board, Phase I Report*, 10 November 1999;
  JPL news release 99-083. lbf-s versus N-s; factor 4.45; periapsis estimates. Grade B.
- Container Handbook, section 4.2.1.3. Volume-to-payload ratios of 1.54 m³/t for 20 ft and
  2.53 m³/t for 40 ft. Grade C, industry reference, undated on the retrieved page.
- GS1 Canada, "Populating TI and HI," and Australian Food and Grocery Council TI/HI
  guidance. Grade B for the guidance, C for the stated consequences.

**Industry, vendor, and trade**

- GS1 Australia data-quality case study with Kimberly-Clark, undated. Weights missing or wrong
  for nearly 100% of audited units; 1,526 packaging levels with defects. Grade C.
- SAP Community, "Rounding Residuals — Decimal Dust," citing SAP Note 3138922 and a 0.0005
  posting threshold, retrieved 2026; SAPinsider, "Avoid These 11 Common Unit-of-Measure
  Errors," undated. Grade D.
- SYSPRO 8 2026 R2, "Catch Weight"; Infor CloudSuite Distribution and M3 catch-weight
  documentation; Aptean catch-weight documentation; inecta, "Catch Weight: Meaning,
  Calculation and Management." Dual quantities, 10 kg nominal versus 11 kg actual, base unit
  plus catch-weight unit. Grade D, vendor documentation, 2026.
- Maia.ai, "How Nature's Touch Caught a $500K Variance." A pounds-to-kilograms formula error
  in a 72-page spreadsheet feeding ERP and material requirements planning. Grade D, undated.

**Note on citations.** Figures attributed to case studies and vendor documentation are reported
as the sources state them and are not independently verified; where a number could not be
confirmed, the text gives the mechanism and omits the number. The Mars figures are the
best-evidenced quantitative claims in the topic.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_container-loading.md`](02_container-loading.md)
- Related: [`03_case-pack-moq-rounding.md`](03_case-pack-moq-rounding.md)
- Related: [`../01_supply_chain_foundations/03_network-entities.md`](../01_supply_chain_foundations/03_network-entities.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)

