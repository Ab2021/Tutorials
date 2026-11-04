# Rate Structures and Accessorials

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 12, topic 01
> Version 2026.5

*Linehaul, fuel, minimums, demurrage, detention, and accessorial creep.*

---

## 1. Why this topic exists

A freight quote is a number. A freight **price** is a function. The difference between the two is the
subject of this topic, and it is where most of the money in a transport budget is either controlled
or lost without anybody noticing.

The quote is what appears in the bid package: "USD 28.50 per hundredweight, Columbus to Nashville,
class 100, effective 2026-01-01." The price is what the invoice says after the base tariff moved, the
fuel peg was applied to one base rather than another, the pallet was reweighed at the terminal, the
receiver kept the driver for three hours, and the consignee turned out to be a residence. Those are
five different mechanisms, owned by four different functions, and none of them appears in the quote.

**The central pathology is accessorial creep.** A procurement team negotiates a two-point improvement
in the discount off the base tariff and reports a win. The landed cost per shipment rises. Nothing in
the negotiation was dishonest and nothing in the invoice was necessarily wrong; the rate improved and
the total did not. A rate model that stores only the negotiated rate cannot see this happen, and a
dashboard that reports cost per hundredweight will report an improvement while the accounts payable
(AP) ledger reports an increase.

Three failures recur, and each has a price attached.

1. **Rating treated as arithmetic rather than as a join.** A shipment is rated by looking up a rate.
   The lookup has keys — lane, mode, service level, freight class or density band, weight band, and
   date — and the table has versioned rows. When the join is wrong the query either returns nothing
   (the shipment is rated at a fallback tariff, usually higher) or returns several rows (the shipment
   is rated several times, and the invoice may carry the highest). Neither error raises an exception
   in most systems. Both are data problems before they are transport problems.
2. **The fuel surcharge treated as a percentage.** Two carriers can both quote "10.13 percent fuel"
   and charge different dollars, because the percentage is applied to a different base, pegged to a
   different index, measured over a different lag, or capped at a different level. A percentage
   without a peg is not a price.
3. **Accessorials treated as noise.** They arrive as a long tail of small lines and get aggregated
   into an "other" bucket. The bucket is where detention, liftgate, reclassification, and appointment
   fees accumulate. Individually small, collectively large, and — unusually for freight — mostly
   controllable by the shipper, because the shipper usually controls the behaviour that triggers them.

**Why this matters to a data scientist specifically.** Rating is the most join-shaped problem in
transport analytics. The rate table is a slowly changing dimension with effective dates. The shipment
is an event. The invoice is a claim about the join between them. Before anyone optimises a lane, the
join has to be correct, and correctness here means: exactly one rate row matches, its validity window
contains the shipment date, its basis is the one the contract actually uses, and every surcharge and
accessorial carries a rule and a date. Module 03 covers the mechanics of those joins and of event data
generally; this topic covers what the rows mean.

**What this topic does not cover.** Mode comparison and the cost architecture of truckload, less-than-
truckload, parcel, rail, air, and ocean is `01_freight-modes.md`. Tender acceptance, carrier mix, and
lane-level procurement are `03_lane-tender-analytics.md`. Cube and weight utilisation as a
consolidation decision is `04_consolidation-cube.md`. Here the shipment is given; the question is what
it costs and why.

> **Note:** Every money figure in this file is in US dollars (USD) and carries the rate date or
> invoice date at which it applies. Where a figure is a forecast, a plan, a target, or a contractual
> commitment, the text says which. Freight rates are commercial terms that change; treat every number
> here as an illustration of a mechanism, not as a market reference.

## 2. Learning outcomes

By the end of this topic you can:

1. **Decompose** a freight invoice into linehaul, fuel surcharge, minimum-charge effect, and
   accessorials, and **attribute** each component to the party that controls it.
2. **Design** an effective-dated rate table with the keys and fields rating actually requires, and
   **explain** why `effective_from` and `effective_to` are not optional metadata.
3. **Rate** a shipment under class-based, density-based, and dimensional-weight bases, and **state**
   which basis governs for a named shipment and why.
4. **Compute** a fuel surcharge from a published index, a peg base, and a stated lag, and **show** how
   two carriers quoting the same percentage charge different dollars.
5. **Quantify** the effect of a minimum charge on the effective rate per pound at low shipment
   weights, including the weight at which the minimum stops binding.
6. **Build** an accessorial catalogue with trigger, owner, and evidence requirement for each charge,
   and **classify** each charge as shipper-controlled, carrier-controlled, or shared.
7. **Diagnose** accessorial creep by decomposing a change in landed cost per shipment into rate,
   fuel, and behaviour components, and **state** which component moved.
8. **Specify** the data an invoice audit needs to be possible, including the tolerance rule and the
   dispute workflow, and **identify** the audit findings that are unverifiable in principle.

Each outcome maps to an exercise in section 12 and an answer in section 13.

## 3. The core mental model

Hold one object: a **price stack**. A shipment's charge is built in three layers, each with its own
determinant, its own owner, and its own failure mode. The layers are not additive in the sense of being
independent; they interact, which is why a change in one can be masked by a change in another.

### 3.1 The three layers

| Layer | What it prices | What determines it | Who controls the trigger | Behaviour in the invoice |
|---|---|---|---|---|
| Linehaul | Moving the shipment from origin to destination | Distance, weight, class or density, mode, and the negotiated rate basis | Procurement negotiates; operations executes | One line, or a base plus a discount line |
| Surcharges | A named external cost passed through by formula | A published index, a peg base, a lag, an application base, sometimes a cap | External market; the contract fixes the formula | Separate line, usually percentage or per-mile |
| Accessorials | A service beyond dock-to-dock movement | The shipment's behaviour: dwell, access, packaging, documentation, equipment | Often the shipper or the consignee, not the carrier | Many small lines, each tied to an event |

The linehaul is the only layer that a negotiation directly moves. The fuel surcharge is a pass-through
whose formula is negotiable but whose level is not. Accessorials are the layer where operational
behaviour meets contract language, and the layer most often left out of the bid comparison.

### 3.2 The price is a function with an effective date

Write the price as a function rather than a number:

`charge = linehaul(lane, mode, class_or_density, weight, date) + fuel(linehaul_base, index, peg, lag, date) + sum(accessorials(events, schedule, date))`

Every argument matters, and the date appears three times. A rate model that omits the date arguments
cannot answer the only question that matters in a dispute: what did the contract say on the day the
shipment moved. A rate model that omits `linehaul_base` from the fuel function cannot explain why two
carriers with the same percentage differ in dollars.

### 3.3 Rating is a join before it is an optimisation

Retrieving the price for a shipment is a lookup against a versioned table. The keys are, at minimum:

| Key | Why it is needed | What goes wrong if it is missing |
|---|---|---|
| Origin and destination | The lane is the unit of pricing | A national average rate is applied to a specific lane |
| Mode and service level | Ground, expedited, and air have different tables | The cheapest table matches and underprices, or the wrong one matches and overprices |
| Class or density band | The rate varies by class within the same lane | Every shipment is rated at class 100 because that is the modal class |
| Weight band or break | Rate per hundredweight falls as weight rises | A 40,000 lb shipment is priced at a 500 lb rate |
| Effective date | Contracts roll, tariffs are reissued, surcharges reset | Last year's rate is applied to this year's shipment, silently |
| Currency and unit basis | Rates are quoted per 100 lb or per 1,000 lb or per kg | A factor-of-ten error that passes every plausibility check |

Two failure shapes follow directly. A **null match** means no row satisfies the keys, and the system
falls back to a default — usually the published tariff, which is higher than the contract. A **fan-out**
means several rows satisfy the keys, typically because two contract versions overlap or because the
weight band boundaries are inclusive at both ends, and the system either errors or picks one by
accident. Both are detected by counting matches per shipment before summing any money: the count must
be exactly one for every shipment that should be rated.

### 3.4 Accessorials attach to events, not to shipments

A shipment has one linehaul. It may have zero, one, or five accessorials, and each is evidence that
something happened. The data structure is therefore an event table joined to the shipment, not a set
of optional columns on the shipment row.

| Accessorial | Triggering event | Owner of the behaviour | Evidence the invoice should carry |
|---|---|---|---|
| Detention (trucking) | Equipment held beyond free time at a stop | Consignee or shipper dock | Arrival time, departure time, free-time allowance |
| Demurrage (ocean) | Container held inside a terminal beyond free time | Consignee, or the party holding the bill of lading | Container number, gate-out availability, free days |
| Liftgate | No dock, or no forklift at the stop | Shipper (order data) or consignee (site data) | Site attribute or service request |
| Residential delivery | Delivery to a non-commercial address | Consignee address data quality | Address classification |
| Limited access | Site with restricted entry, such as a school or a prison | Consignee site data | Site attribute |
| Inside delivery | Goods carried beyond the threshold | Shipper instruction or customer request | Service request line |
| Appointment or notification | A scheduled window was required or used | Receiver policy; shipper scheduling | Appointment identifier |
| Reweigh | Terminal scale reads a different weight | Shipper (weight capture) | Scale reading, scale identifier, date |
| Reclassification | Terminal judges the class wrong | Shipper (item description and packaging) | Original and revised class, reason code |
| Address correction | Delivery address is wrong or incomplete | Shipper (master data) | Original and corrected address |
| Oversize or overdimension | Shipment exceeds length or girth limits | Shipper (packaging) | Measured dimensions |
| Hazardous materials | Regulated goods requiring handling | Shipper (product classification) | UN number, class, packaging group |
| Chassis or equipment | A chassis or special equipment is supplied | Contract term | Equipment identifier, rental days |
| Stop-off or extra stop | Route includes an additional stop | Shipper (routing instruction) | Stop sequence and address |

The ownership column is the analytically useful one. It says where a fix is possible. A detention
charge caused by a receiver that cannot take a delivery before 09:00 is not a carrier problem, a
pricing problem, or a data problem in the usual sense; it is a scheduling problem that appears in the
freight budget. Attributing it to "carrier charges" hides the lever.

### 3.5 The waterbed: why a better discount can cost more

Many less-than-truckload (LTL) contracts are written as a discount off a published base tariff, not
as an absolute rate. The shipper negotiates "62 percent off" and later "64 percent off". If the base
tariff rises between the two agreements, the effective rate can rise even though the discount
improved. The mechanism is simple arithmetic and it defeats most bid comparisons, because the bid
sheet shows the discount and not the base.

| | Rate year 1 | Rate year 2 |
|---|---|---|
| Base tariff | USD 75.00 per hundredweight (2026-01-01 tariff) | USD 82.00 per hundredweight (2027-01-01 tariff) |
| Discount off base | 62.0 percent (2026-01-01 contract) | 64.0 percent (2027-01-01 contract) |
| Effective rate | USD 28.50 per hundredweight (2026-01-01) | USD 29.52 per hundredweight (2027-01-01) |
| Direction | Baseline | 3.6 percent higher |

The discount improved by 2.0 percentage points and the effective rate rose by 3.6 percent. This is not
a trick by the carrier; it is what "discount off tariff" means. The lesson for a rate model is that
the base tariff is a first-class field, not a reference document, and that a bid comparison must
normalise to an effective rate per unit before any comparison is made.

### 3.6 Controllability, restated

| Component | Share of a typical LTL invoice | Controllable by the shipper | Lever |
|---|---|---|---|
| Linehaul | The majority | Indirectly | Procurement, consolidation, mode choice |
| Fuel | A material minority | No | Nothing, beyond the formula |
| Accessorials | A minority, highly variable | Often yes | Packaging, master data, dock scheduling, address quality |
| Minimum-charge effect | Small in total, large per small shipment | Yes | Order minimums, consolidation |

The percentages in the first column vary by shipper, mode, and lane and are not offered as a benchmark.
The point of the table is the ordering of the last two columns: the component with the least direct
control is the largest, and the component with the most control is the one usually left out of the
review.

> **Note:** A rate model is a claim about a contract. The contract is the evidence. When the model and
> the contract disagree, the model is wrong until proven otherwise, and the place to look first is the
> effective date.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Linehaul | The charge for moving freight from origin to destination, before surcharges and accessorials | Used to mean the total freight charge | Currency per shipment, derived from a rate basis |
| Rate basis | The unit and formula a rate is expressed in | Left unstated, making rates incomparable | Per 100 lb, per mile, per shipment, percent of tariff |
| Tariff | A published schedule of rates, rules, and charges | Confused with a contract | Document plus version plus effective date |
| Contract rate | A negotiated price that overrides or discounts the tariff for named lanes | Assumed to cover every service the shipment needs | Lane, class, weight band, effective window |
| Effective date | The date from which a rate or rule applies | Recorded as a load date instead of a validity date | ISO 8601 date |
| Expiry date | The last date on which a rate applies | Omitted, so superseded rates stay matchable | ISO 8601 date, nullable for open-ended |
| Class rating | Pricing by a freight class assigned to the commodity | Treated as a property of the shipment rather than the item | Class code, 50 to 500 |
| Density rating | Pricing by weight per unit volume, often through a class scale | Treated as identical to class rating | Pounds per cubic foot, or kilograms per cubic metre |
| Freight class | A number from 50 to 500 expressing transportability | Assumed to be a regulated safety rating | Integer or half-step code |
| NMFC item number | The identifier in the National Motor Freight Classification (NMFC) that governs a commodity's class and rules | Quoted without the sub-item or packaging code | Item and sub-item, e.g. a five-digit item plus suffix |
| NMFTA | National Motor Freight Traffic Association, the non-profit standards body that publishes the NMFC | Described as a government agency | Organisation, not a code |
| Hundredweight | A unit of 100 pounds, the traditional LTL quoting basis | Applied to metric tonnes without conversion | Abbreviated cwt; 1 cwt = 45.359 kg |
| Minimum charge | A floor on the charge for a shipment, however small | Omitted from models, understating small-shipment cost | Currency per shipment, with an effective date |
| Absolute minimum | A carrier-level floor that a discount cannot reduce below | Confused with the lane minimum | Currency per shipment |
| Break weight | The weight at which the next rate band gives a lower total charge | Confused with a discount threshold | Pounds or kilograms |
| Minimum-charge threshold | The weight below which the minimum charge binds and the rate per unit is set by the floor | Not computed at all | Pounds, derived from minimum divided by rate |
| Discount off tariff | A percentage reduction applied to a published base rate | Compared across carriers without comparing base tariffs | Percent, with the base tariff identified |
| Base tariff | The published rate a discount is applied to | Treated as a constant across years | Currency per hundredweight, with a version date |
| FAK | Freight all kinds; a single rate applied to a mixed shipment regardless of class | Assumed to remove classification obligations | Rate code plus eligibility rules |
| Spot rate | A one-off price for a single shipment or short validity | Treated as a contract rate in a budget | Currency per shipment or per mile, with a quote date |
| Contract rate (committed) | A rate with a stated validity window and volume expectation | Used interchangeably with spot | Currency, plus validity window |
| General rate increase | A published percentage increase applied to a tariff, usually on a stated date | Applied to contract rates that are not tariff-based | Percent, with an announcement and effective date |
| Fuel surcharge | A charge that passes through fuel price movement by formula | Treated as a fixed percentage of the total | Percent of a stated base, or currency per mile |
| Fuel index | The published price series a surcharge is pegged to | Assumed to be the carrier's own purchase cost | Named series, with publisher and frequency |
| Peg base | The index price at which the surcharge is zero | Reset silently at contract renewal | Currency per gallon or per litre, with a date |
| Surcharge lag | The delay between an index movement and its application to rates | Assumed to be zero | Days or weeks |
| Application base | The charge the fuel percentage is applied to | Assumed to be linehaul only | Currency amount, defined by clause |
| Fuel cap | A ceiling on the surcharge percentage or amount | Assumed absent | Percent or currency, with a period |
| Accessorial | A charge for a service beyond dock-to-dock movement | Assumed to be included in the linehaul | Currency per event, with a schedule |
| Detention | A charge for holding a carrier's equipment beyond free time, in trucking | Used for ocean, where the term is demurrage or detention with different meanings | Currency per hour after free time |
| Demurrage | A charge for a container held inside a terminal beyond free time | Used for any storage charge | Currency per container per day |
| Free time | The period before detention or demurrage begins | Assumed to be uniform across carriers and ports | Hours or days, stated in the contract or tariff |
| Liftgate | A charge for delivering without a loading dock | Assumed to be a carrier choice rather than a site attribute | Currency per stop |
| Residential delivery | A surcharge for delivery to a non-commercial address | Triggered by a wrong address classification, not by intent | Currency per shipment |
| Limited access | A surcharge for a site with restricted entry or hours | Applied without a defined site list | Currency per shipment, with a site attribute |
| Inside delivery | A charge for carrying goods beyond the threshold | Assumed to be part of residential delivery | Currency per shipment |
| Appointment fee | A charge for scheduling or missing a delivery window | Confused with detention | Currency per appointment |
| Reweigh | A charge and a weight revision when a terminal scale disagrees with the declared weight | Treated as a data error rather than a billable event | Pounds, plus currency per shipment |
| Reclassification | A charge and a class revision when the carrier disputes the declared class | Assumed to be arbitrary | Original class, new class, currency per shipment |
| Address correction | A charge for fixing an address that cannot be delivered as given | Absorbed silently | Currency per correction |
| Oversize charge | A charge for shipments exceeding dimension or weight limits | Confused with overweight permits | Currency per shipment, with the limit stated |
| Hazardous materials surcharge | A charge for regulated goods | Assumed to cover all compliance obligations | Currency per shipment, with the UN number recorded |
| Chassis charge | A charge for the wheeled frame used to move a container | Assumed to be part of the ocean freight | Currency per chassis per day |
| Stop-off charge | A charge for an additional stop on a route | Priced as a fraction of linehaul without a rule | Currency per stop |
| Billable event | The occurrence that makes an accessorial chargeable | Recorded without a timestamp | Event type plus timestamp plus location |
| Rated shipment | A shipment with a computed charge attached, before invoicing | Treated as equal to the invoiced amount | Record identifier plus rate reference |
| Chargeable weight | The weight used for pricing, being the greater of actual and dimensional weight where the rule applies | Equated with actual weight | Pounds or kilograms |
| Dimensional weight | Volume converted to a weight equivalent by a divisor | Computed with inconsistent units | Pounds, from cubic inches divided by a divisor |
| DIM divisor | The number of cubic inches per pound used in the dimensional weight formula | Assumed universal across carriers and services | Cubic inches per pound, e.g. 139 |
| Cubic capacity rule | A rule allowing a carrier to reprice a light shipment by the space it occupies | Confused with dimensional weight | Cubic feet or cubic metres, with a threshold |
| Freight audit and pay | The process of checking carrier invoices against contract before payment | Treated as a data-entry function | Process, with a tolerance rule and a dispute path |
| Tolerance | The variance within which an invoice line is paid without dispute | Set to zero, producing dispute volume nobody can process | Percent or currency, per line or per invoice |
| Accessorial creep | The rise in accessorial share of landed freight cost while quoted rates improve | Reported as a carrier pricing problem | Percent of total, with a matched shipment cohort |
| Landed freight cost | Linehaul plus fuel plus accessorials plus any internal handling charge | Reported as the quoted rate | Currency per shipment, with a scope statement |

Five of these deserve expansion.

**Minimum charge versus minimum-charge threshold.** The minimum charge is a currency amount. The
threshold is a weight, derived by dividing the minimum charge by the rate. They are different objects
and only the second one tells you which shipments are affected. A USD 185.00 minimum (2026-01-01) at
USD 28.50 per hundredweight (2026-01-01) binds below 6.49 hundredweight, or 649 lb; every shipment
lighter than that is priced by the floor, and its effective rate per pound rises as it gets lighter.

**Peg base versus index.** The index is the market price; the peg base is the price at which the
surcharge is defined to be zero. The same index movement produces a different surcharge for every
peg base. Renewing a contract and resetting the peg to the current index makes the surcharge disappear
without any change in the underlying cost, which is why the peg is a negotiable term and not a
technical detail.

**Detention versus demurrage.** In ocean shipping the two are distinct: demurrage is charged while a
container sits inside the terminal beyond free time, and detention while it sits outside, typically at
the consignee. They have different clocks, different owners, and in the United States a federal
billing rule for both. In trucking, detention is a purely contractual charge for holding a trailer or
a driver beyond free time; there is no general statutory rule for it. Treating the ocean rule as if it
governed trucking is a common and expensive category error.

**Class versus density.** Class rating prices an item; density rating prices a shipment's weight per
unit volume. Where an item is density-rated, the density scale supplies the class, and the invoice
should show both the density computed and the class applied. Where an item carries handling,
stowability, or liability exceptions, density alone will understate the class, and the item number
governs. A model that converts every shipment to a density class will systematically misprice exactly
the goods that are hard to handle.

**Audit versus accrual.** An accrual is an estimate of freight cost for a period, usually built from
rated shipments. An audit is a comparison of an invoice to a contract. They answer different questions
and diverge: an accrual can be internally consistent while every invoice is wrong, and an audit can
find invoice errors while the accrual remains a reasonable planning figure. Reporting one as if it were
the other is how cost-per-shipment series acquire unexplained steps.

## 5. The mechanics, step by step

The sequence below is the order in which the objects have to be built, not necessarily the order in
which a system executes them. Steps 1 to 4 build the rate record; steps 5 to 7 classify the shipment;
steps 8 to 12 compute the linehaul; step 13 and 14 apply the fuel mechanism; steps 15 and 16 apply
accessorials; steps 17 and 18 audit and pay.

### Phase A — Build the rate record

**Step 1. Separate the tariff from the contract.**

- Input: the carrier's published tariff or rate guide, and the negotiated contract document.
- Output: two distinct objects, one published and one negotiated, linked by reference.
- Branch point: decide whether the contract expresses an absolute rate or a discount off the tariff.
  This single decision determines whether the base tariff must be loaded as data. If the contract is
  discount-based, the base tariff is a live input to every rating calculation and its version history
  must be stored.

**Step 2. Record the rate basis explicitly.**

- Input: the contract clause that states how the charge is computed.
- Output: a rate basis code, one of per hundredweight, per mile, per shipment, per stop, per pallet,
  or percent of tariff.
- Branch point: mixed bases. A contract may price linehaul per hundredweight and a stop-off per stop.
  Each component needs its own basis; a single basis field per lane forces one of them to be
  approximated, and the approximation is usually wrong in the carrier's favour.

**Step 3. Version the rate with validity dates.**

- Input: the contract's effective and expiry dates, plus any mid-term amendments.
- Output: rows that are never updated in place, only superseded.
- Branch point: decide the cut-over rule when an amendment is signed after its effective date.
  The usual rule is that the amendment applies from its stated effective date, which means historical
  shipments must be re-rated when the amendment lands. A system that applies new rates only going
  forward will carry a permanent discrepancy against the carrier's own re-rate.

**Step 4. Record the fuel peg as a structured object.**

- Input: the fuel clause.
- Output: five fields — index series name, index publisher, peg base price and its date, lag in days
  or weeks, application base, and cap if any.
- Branch point: whether the application base is linehaul only or linehaul plus accessorials. This is
  worth a direct question in negotiation, because it changes the dollars at identical percentages and
  is rarely stated on the bid sheet.

### Phase B — Classify the shipment

**Step 5. Determine chargeable weight.**

- Input: declared weight, measured dimensions, and the applicable rule.
- Output: chargeable weight in pounds or kilograms, plus the basis used.
- Branch point: for LTL the rule may be actual weight, or a cubic capacity rule that reprices light
  shipments by the space occupied. For parcel the rule is dimensional weight, the greater of actual and
  volume divided by the DIM divisor. For air the convention is volumetric, commonly 1 cubic metre per
  167 kg, equivalent to 6,000 cubic centimetres per kilogram, with some carriers using 5,000. State
  the rule and the divisor; they are commercial terms and they differ.

**Step 6. Determine the class.**

- Input: the NMFC item number and sub-item, the commodity description, the packaging, and the density.
   (NMFC is the National Motor Freight Classification, published by the National Motor Freight Traffic
  Association, NMFTA.)
- Output: a class code between 50 and 500.
- Branch point: whether the item is density-rated, exception-rated, or fixed. If the item number is
  density-rated, compute density and read the class from the density scale. If the item carries a
  handling, stowability, or liability exception, the item's assigned class governs and density is only
  supporting evidence.

**Step 7. Apply the rounding rule.**

- Input: the tariff's measurement and rounding clause.
- Output: rounded dimensions, rounded weight, and the final chargeable weight.
- Branch point: whether dimensions round up before multiplying or the volume rounds after. The two
  produce different answers on the same carton. A carton measuring 20.4 by 16.2 by 14.1 inches gives
  4,650 cubic inches if rounded first (21 by 17 by 15) and 4,660 if multiplied first, and the difference
  can cross a pound boundary after division.

### Phase C — Compute the linehaul

**Step 8. Convert to the rate unit.**

- Input: chargeable weight and the rate basis.
- Output: the quantity in rate units, for example hundredweight.
- Branch point: rounding of the converted quantity. Some tariffs round up to the next whole
  hundredweight, some price the exact weight, some round to the next pound first. Each choice changes
  the answer at the margin.

**Step 9. Apply the rate.**

- Input: the rate for the matching class, weight band, and lane, with its effective date.
- Output: a gross linehaul amount.
- Branch point: weight breaks. Where a lower rate applies above a break weight, the correct charge is
  the minimum of the charge at the actual weight and the charge at the break weight. Taking the first
  only overcharges; taking the second when the shipment is below the break invents a discount.

**Step 10. Apply the discount if the structure is discount-based.**

- Input: the base tariff rate and the discount percentage, both effective-dated.
- Output: the net rate per unit.
- Branch point: whether the discount applies before or after the minimum charge. Discounts usually
  cannot reduce the charge below the absolute minimum, which means the effective discount at small
  weights is smaller than the headline figure.

**Step 11. Apply the minimum charge.**

- Input: the net computed charge and the applicable minimum, both with dates.
- Output: the linehaul charge actually billed.
- Branch point: whether the minimum is a lane minimum, a shipment minimum, or an absolute minimum that
  survives discounting. Three different floors can apply to one shipment; the billed charge is the
  highest of them.

**Step 12. Check the break weight and the threshold.**

- Input: the rate table for adjacent weight bands and the minimum charge.
- Output: two diagnostic numbers — the weight at which the next band becomes cheaper, and the weight
  below which the minimum binds.
- Branch point: whether to re-rate the shipment at the break weight. For a shipment just below a break,
  declaring the higher weight can produce a lower charge. This is legitimate where the tariff permits
  it and is a standard rating check.

### Phase D — Apply the fuel mechanism

**Step 13. Compute the surcharge percentage from the index and the peg.**

- Input: the index value for the applicable period, the peg base, and the lag rule.
- Output: a percentage, or a per-mile amount, depending on the clause.

The arithmetic is a ratio of the movement to the peg:

`percent = (index_value - peg_base) / peg_base`

With an index value of USD 4.10 per gallon (EIA week of 2026-06-02) and a peg base of USD 3.90 per
gallon (2026-01-01 contract), the surcharge is (4.10 - 3.90) / 3.90 = 0.051282, or 5.13 percent.

- Branch point: whether the index is the national average, a regional (PAD district) average, or a
  state average. A national peg applied to a West Coast lane under-recovers; a regional peg applied to
  a national contract creates a permanent argument. The clause must name the series and the geography.

**Step 14. Apply the surcharge to the defined base.**

- Input: the percentage and the application base named in the clause.
- Output: the fuel charge.
- Branch point: whether negative movement is passed back. Where the index falls below the peg, some
  clauses credit the shipper and some floor the surcharge at zero. The second is common and is worth
  knowing before assuming a symmetric formula.

### Phase E — Apply accessorials

**Step 15. Match events to the accessorial catalogue.**

- Input: the event stream — arrival, departure, gate-out, appointment, scale reading, dimension
  capture — and the accessorial schedule.
- Output: candidate chargeable events, each with a timestamp and a location.
- Branch point: free time. Detention begins after a stated free period, and the clock's start point
  differs: arrival at the gate, check-in at the guardhouse, or the appointment time. A clause that does
  not name the start point will be argued about on every invoice.

**Step 16. Test authorisation and duplication.**

- Input: the candidate events and the shipment record.
- Output: chargeable accessorials, each with evidence attached.
- Branch point: whether the charge was requested. A liftgate charge on a shipment with no service
  request and a dock-level delivery address is a candidate for dispute regardless of what happened at
  the site.

### Phase F — Audit and pay

**Step 17. Match invoice lines to rated lines.**

- Input: the carrier invoice, the rated shipment from the transport management system (TMS), and the
  contract.
- Output: matched lines, unmatched invoice lines, and rated-but-not-invoiced lines.
- Branch point: tolerance. A tolerance stated as a percentage of the line and one stated as a currency
  amount per invoice behave differently at low line values; the percentage tolerance is more permissive
  on large lines and less permissive on small ones.

**Step 18. Route discrepancies to dispute and record the outcome.**

- Input: exceptions from step 17.
- Output: a dispute record with a reason code, an amount, a status, and a resolution date.
- Branch point: whether to pay and pursue a credit, or hold payment. The choice affects the ageing of
  the payable, the relationship with the carrier, and — where a regulatory billing rule applies, as in
  ocean demurrage and detention in the United States — the legal position on whether an obligation to
  pay exists at all.

## 6. Worked example

This example follows one lane through two rate years. It is constructed to show accessorial creep, so
the shipment profile is held constant and only the commercial and behavioural terms change.

### 6.1 The shipment

| Attribute | Value | Source |
|---|---|---|
| Lane | Columbus, Ohio to Nashville, Tennessee | Shipment record |
| Road distance | 379 miles, approximately 610 kilometres | Routed distance, stated provider not material here |
| Actual weight | 3,200 lb, equal to 32.0 hundredweight | Declared, then confirmed at origin |
| Volume | 400 cubic feet | Measured |
| Density | 8.0 lb per cubic foot | Derived: 3,200 divided by 400 |
| Class | 100 | NMFC item number governs; density of 8.0 lb per cubic foot falls in the 8 to under 10 band on the density scale |
| Mode | LTL | Contract |

### 6.2 Year 1: the rate as quoted

| Component | Rule | Arithmetic | Amount |
|---|---|---|---|
| Base tariff | Published, 2026-01-01 version | Given | USD 75.00 per hundredweight |
| Discount | 62.0 percent, 2026-01-01 contract | 75.00 multiplied by 0.38 | USD 28.50 per hundredweight |
| Linehaul | 32.0 hundredweight at the net rate | 32.0 multiplied by 28.50 | USD 912.00 |
| Minimum charge | USD 185.00, 2026-01-01 | 185.00 is below 912.00, so it does not bind | Not applicable |
| Fuel index | USD 4.10 per gallon, EIA week of 2026-06-02 | Given | — |
| Fuel peg base | USD 3.90 per gallon, 2026-01-01 contract | Given | — |
| Fuel percentage | (4.10 minus 3.90) divided by 3.90 | 0.20 / 3.90 | 5.13 percent |
| Fuel charge | Percentage applied to linehaul only | 912.00 multiplied by 0.051282 | USD 46.77 |
| Accessorials | None incurred | — | USD 0.00 |
| **Total** | | 912.00 plus 46.77 | **USD 958.77** (invoice dated 2026-06-18) |

Effective rates from year 1: USD 29.96 per hundredweight (958.77 divided by 32.0) and USD 0.2996 per
pound, both at the 2026-06 invoice date.

### 6.3 Year 2: the rate as negotiated, with behaviour change

The negotiation improves the discount from 62.0 to 64.0 percent. The base tariff is reissued at
USD 82.00 per hundredweight with a 2027-01-01 effective date. The fuel index is held at USD 4.10 per
gallon (2027-06-01 observation) so that the fuel effect is isolated. Three accessorials appear because
the receiving site changed: a liftgate is now needed at a new drop point, an appointment is required,
and one hour of detention is incurred.

| Component | Rule | Arithmetic | Amount |
|---|---|---|---|
| Base tariff | Published, 2027-01-01 version | Given | USD 82.00 per hundredweight |
| Discount | 64.0 percent, 2027-01-01 contract | 82.00 multiplied by 0.36 | USD 29.52 per hundredweight |
| Linehaul | 32.0 hundredweight at the net rate | 32.0 multiplied by 29.52 | USD 944.64 |
| Fuel peg base | Unchanged at USD 3.90 per gallon, 2027-01-01 contract | Given | — |
| Fuel percentage | (4.10 minus 3.90) divided by 3.90 | 0.20 / 3.90 | 5.13 percent |
| Fuel charge | Percentage applied to linehaul only | 944.64 multiplied by 0.051282 | USD 48.44 |
| Liftgate | USD 48.00 per stop, 2027-01-01 accessorial schedule | One stop | USD 48.00 |
| Appointment | USD 38.00 per appointment, 2027-01-01 | One appointment | USD 38.00 |
| Detention | USD 80.00 per hour after 2 free hours, 2027-01-01 | 3.0 hours on site less 2.0 free | USD 80.00 |
| **Total** | | 944.64 plus 48.44 plus 166.00 | **USD 1,159.08** (invoice dated 2027-06-21) |

### 6.4 The decomposition

| Component | Year 1 | Year 2 | Change | Share of change |
|---|---|---|---|---|
| Linehaul | USD 912.00 (2026-01-01 rate) | USD 944.64 (2027-01-01 rate) | +USD 32.64 | 16.3 percent |
| Fuel | USD 46.77 (2026-06 invoice) | USD 48.44 (2027-06 invoice) | +USD 1.67 | 0.8 percent |
| Accessorials | USD 0.00 | USD 166.00 (2027-01-01 schedule) | +USD 166.00 | 82.9 percent |
| **Total** | **USD 958.77** | **USD 1,159.08** | **+USD 200.31** | 100 percent |

Check: 32.64 plus 1.67 plus 166.00 equals 200.31. The total rose by 20.9 percent, computed as
200.31 divided by 958.77.

Now separate the rate effect from the behaviour effect. Hold accessorials at zero in year 2: the total
would be 944.64 plus 48.44 equals 993.08, which is 3.6 percent above year 1. The quoted rate improved
by 2.0 discount points, the effective linehaul rate rose by 3.6 percent because the base tariff rose
by 9.3 percent, and the remaining 17.3 percentage points of the increase are accessorials that did not
exist in year 1. That 17.3 is also 166.00 divided by 958.77, which is the accessorial load expressed
against the year-1 total.

Two arithmetic notes on the 9.3 percent: 82.00 divided by 75.00 equals 1.0933, so the base tariff rose
9.3 percent, while the net rate rose 29.52 divided by 28.50 equals 1.0358, or 3.6 percent. The discount
absorbed most of the tariff increase and none of the behaviour change.

### 6.5 The minimum charge, on a different shipment

Take a 250 lb shipment on the same lane, class 85, at USD 28.50 per hundredweight (2026-01-01 rate)
with a USD 185.00 minimum (2026-01-01).

| Weight | Rate units | Computed linehaul | Billed linehaul | Effective rate per lb |
|---|---|---|---|---|
| 250 lb | 2.5 hundredweight | USD 71.25 (2.5 × 28.50) | USD 185.00 (minimum binds) | USD 0.74 |
| 400 lb | 4.0 hundredweight | USD 114.00 (4.0 × 28.50) | USD 185.00 (minimum binds) | USD 0.46 |
| 649 lb | 6.49 hundredweight | USD 184.97 (6.49 × 28.50) | USD 185.00 (at the threshold) | USD 0.285 |
| 900 lb | 9.0 hundredweight | USD 256.50 (9.0 × 28.50) | USD 256.50 (rate binds) | USD 0.285 |

The threshold is 185.00 divided by 28.50 equals 6.4912 hundredweight, or 649 lb. Below it, the
effective rate per pound is set entirely by the floor: a 250 lb shipment pays 2.6 times the per-pound
rate of a 900 lb shipment on the same lane at the same rate. This is the mechanism by which a minimum
charge silently sets the effective rate at low volume, and it is why order minimums and consolidation
are economic levers rather than operational preferences.

### 6.6 The same fuel percentage, different dollars

Two carriers both quote 5.13 percent, pegged to the same index at the same peg base. Carrier A applies
the percentage to linehaul only. Carrier B applies it to linehaul plus accessorials.

| Carrier | Application base | Arithmetic | Fuel charge |
|---|---|---|---|
| A | Linehaul only | 912.00 × 0.051282 | USD 46.77 |
| B | Linehaul plus accessorials | (912.00 + 166.00) × 0.051282 = 1,078.00 × 0.051282 | USD 55.28 |

Difference: USD 8.51 on this shipment, at an identical headline percentage. On a book of 20,000
shipments a year with a mean accessorial load of USD 60.00 (2027-01-01 schedule), the difference is
20,000 × 60.00 × 0.051282 = USD 61,538 per year (2027 money), which is a derived figure from stated
assumptions and not a benchmark.

A third variant exists: a per-mile fuel charge. At USD 0.42 per mile (2026-01-01 contract) on 379
miles, the charge is USD 159.18, which is not comparable to either percentage and must be normalised
before any bid comparison.

### 6.7 Dimensional weight, on a parcel

A carton measures 24 by 18 by 12 inches, giving 5,184 cubic inches, and weighs 22 lb actual.

| Divisor | Volume divided by divisor | Rounded up | Billable weight | Basis |
|---|---|---|---|---|
| 139 cubic inches per pound | 37.29 lb | 38 lb | 38 lb | Dimensional weight exceeds actual |
| 166 cubic inches per pound | 31.23 lb | 32 lb | 32 lb | Dimensional weight exceeds actual |

At the 139 divisor the shipment is billed at 38 lb against 22 lb actual, a 72.7 percent increase in
billed weight (38 divided by 22 minus 1). At 166 it is billed at 32 lb, a 45.5 percent increase. The
product is identical; the divisor is a commercial term. A packaging change that reduces one dimension
by 2 inches changes the volume to 24 by 18 by 10 equals 4,320 cubic inches, which at 139 gives 31.08,
rounded to 32 lb — a 6 lb reduction in billed weight without any change in the negotiated rate.

### 6.8 Status of each number

| Number | Status | Basis |
|---|---|---|
| Year 1 rate and discount | Commitment | Signed contract effective 2026-01-01 |
| Year 1 fuel index | Observation | Published index, week of 2026-06-02 |
| Year 2 rate and discount | Commitment | Signed contract effective 2027-01-01 |
| Year 2 accessorial schedule | Commitment | Signed schedule effective 2027-01-01 |
| Detention hours in year 2 | Observation | Dock timestamps for this shipment |
| The 20,000-shipment book | Plan | Stated assumption, not a measured volume |
| The USD 61,538 annual figure | Derived forecast | Arithmetic on the plan volume and mean accessorial load |

## 7. Code, query, or calculation

The query below is the rating join in its simplest correct form: one shipment, one matching rate row,
one fuel index value from the most recent publication on or before the shipment date. It is written in
SQL for portability.

```sql
-- Rate one shipment against the effective-dated contract.
-- Expected output on the worked example:
--   shipment_id = SHP-4471, linehaul = 912.00, fuel_chg = 46.77, total = 958.77 (USD)
WITH ship AS (
  SELECT 'SHP-4471' AS shipment_id, DATE '2026-06-02' AS ship_date,
         32.0 AS cwt, 'CMH-BNA' AS lane_id, 'CLS-100' AS class_code
),
r AS (
  SELECT c.base_tariff_per_cwt, c.discount_pct, c.min_charge, c.fuel_peg_base
  FROM   contract_rates c, ship s
  WHERE  c.lane_id = s.lane_id
    AND  c.class_code = s.class_code
    AND  c.effective_from <= s.ship_date
    AND  (c.effective_to IS NULL OR c.effective_to > s.ship_date)
),
f AS (
  SELECT i.price_per_gal
  FROM   fuel_index i, ship s
  WHERE  i.series = 'EIA_DSL_ONHW'
    AND  i.week_ending <= s.ship_date
  ORDER BY i.week_ending DESC
  LIMIT 1
),
lh AS (
  SELECT GREATEST(s.cwt * r.base_tariff_per_cwt * (1 - r.discount_pct),
                  r.min_charge) AS linehaul,
         r.fuel_peg_base
  FROM   ship s, r
)
SELECT s.shipment_id,
       ROUND(lh.linehaul, 2) AS linehaul,
       ROUND(lh.linehaul * (f.price_per_gal - lh.fuel_peg_base)
             / lh.fuel_peg_base, 2) AS fuel_chg,
       ROUND(lh.linehaul * (1 + (f.price_per_gal - lh.fuel_peg_base)
             / lh.fuel_peg_base), 2) AS total
FROM   ship s, lh, f;
```

The query is deliberately minimal and has three properties worth naming. It selects the rate row by
validity window rather than by load order, so a late amendment with an earlier effective date is
picked up correctly. It selects the fuel observation by most recent publication on or before the
shipment date, which encodes the lag as data rather than as an assumption. And it applies the minimum
charge with `GREATEST` before the fuel calculation, which matches the clause used in the example where
the surcharge applies to the billed linehaul.

What it does not do, and what a production rating engine must add: a cardinality check that exactly one
row matched (a fan-out here silently multiplies the charge), a currency and unit check on every rate
row, a branch for discount-based versus absolute pricing, a per-mile fuel variant, an accessorial join,
and a rounding rule per component. Those additions are the subject of exercise 7.

## 8. Data, standards, and regulatory context

### 8.1 The classification standard

The National Motor Freight Classification (NMFC) is published by the National Motor Freight Traffic
Association (NMFTA), a membership-based non-profit organisation, not a government body. Classes run
from 50 to 500 and are assigned on four transportation characteristics: density, handling,
stowability, and liability. The NMFC is a voluntary standard in the sense that it binds parties who
reference it in contracts, tariffs, or bills of lading, and NMFTA licenses participants accordingly.

NMFTA has been moving many commodity items onto a density-based scale. In Docket 2025-1, the Freight
Classification Development Council adopted a standard 13-subprovision density scale, published in an
NMFC supplement issued 2025-05-10 and effective 2025-07-19. The scale as published runs: under 1 lb per
cubic foot to class 400; 1 to under 2 to class 300; 2 to under 4 to class 250; 4 to under 6 to class
175; 6 to under 8 to class 125; 8 to under 10 to class 100; 10 to under 12 to class 92.5; 12 to under
15 to class 85; 15 to under 22.5 to class 70; 22.5 to under 30 to class 65; 30 to under 35 to class 60;
35 to under 50 to class 55; and 50 and above to class 50. Items carrying handling, stowability, or
liability exceptions do not use the full density scale. The worked example above uses the 8 to under
10 band, which is consistent with the published scale.

Two cautions for a data team. First, the scale is a standards document that changes; the version
effective on the shipment date governs, so the scale itself needs an effective date in the rate model,
not just the rate. Second, an item number is not a class. The same class can come from a density band
or from an exception item, and the reason matters when a reclassification is disputed.

### 8.2 Dimensional weight in parcel

Parcel carriers price on the greater of actual and dimensional weight. UPS publishes a divisor of 139
cubic inches per pound for its daily rates and 166 for its retail rates in its rate and service guide,
with each dimension rounded up to the next whole inch and the result rounded up to the next whole
pound. FedEx domestic services use 139. These are commercial terms, not regulated ones, and they have
changed over time; the divisor and the rounding rule should be stored as dated contract attributes
rather than hard-coded.

This is a carrier-published term and the evidence grade is C: it is not peer-reviewed, and it is not an
official statistic. Carrier portals and rate guides are the primary source and should be checked
against the current guide for the service actually used.

### 8.3 The fuel index

In the United States the standard peg for on-highway diesel is the US Department of Energy (DOE) Energy
Information Administration (EIA) weekly retail price series. The series is a volume-weighted average
cash, self-serve retail price for ultra-low-sulfur on-highway diesel, including taxes, collected every
Monday at 8:00 a.m. local time from a sample of 590 retail outlets in the contiguous United States
under Form EIA-888. Prices are published on Tuesday at approximately 10:00 a.m. Eastern Time, a
schedule that moved from Monday afternoons beginning the week of 2025-04-07; Monday federal holidays
shift release to Wednesday, still using Monday's prices. The series is an official statistic, grade B.

Two consequences for a surcharge model. The lag is at least one day and in practice one to two weeks
between the price observation and its use, and the contract must state which publication date is
referenced. And the series is national: PAD district and state sub-series exist, and a contract pegged
to a regional series will produce a different percentage from one pegged to the national series on the
same day.

In Europe the analogous reference is the European Commission's Weekly Oil Bulletin, which publishes
member-state retail prices for automotive gas oil with and without taxes. Member states submit data on
Wednesdays and the bulletin is distributed on Thursdays. National tax treatment differs substantially
across member states, so a Europe-wide peg needs to specify whether it uses prices with or without
taxes, and which geography. The publication is an official statistic, grade B.

### 8.4 Ocean demurrage and detention billing

The US Federal Maritime Commission (FMC) issued a final rule, Demurrage and Detention Billing
Requirements, at 46 CFR part 541, Docket No. FMC-2022-0066, RIN 3072-AC90, published at 89 FR 14330 on
2024-02-26 and effective 2024-05-28. It was issued under the Ocean Shipping Reform Act of 2022 (OSRA
2022) and applies to vessel-operating common carriers (VOCCs), non-vessel-operating common carriers
(NVOCCs), and marine terminal operators (MTOs).

What the rule requires, in outline: an invoice must contain specified minimum information, including
the container and bill of lading identification, the basis on which the billed party is liable, the
invoice date and due date, and the dates on which charges began to accrue; the invoice must be issued
within 30 calendar days after the charge accrued; the billed party must have at least 30 days from
issuance to request mitigation, refund, or waiver, and the biller must attempt resolution within 30
days of a request. Failure to include the required minimum information removes the billed party's
obligation to pay. The invoice-contents provision at 46 CFR 541.6 was initially deferred pending
Paperwork Reduction Act clearance and took effect on the same 2024-05-28 date after the Office of
Management and Budget approved the information collection.

There is a subsequent legal development that a current file must record. On 2025-09-23 the US Court of
Appeals for the District of Columbia Circuit, in litigation brought by the World Shipping Council,
set aside 46 CFR 541.4, the provision governing to whom an invoice may be sent. The FMC's position, as
published on its own site, is that apart from section 541.4 the rest of the rule remains in effect.
A compliance process built on part 541 should therefore treat the invoice-content, timing, and dispute
provisions as live and the recipient provision as vacated, and should confirm the current position
before relying on either.

**Ground detention is different, and this matters.** Detention charged by a trucking carrier for
holding a trailer or driver beyond free time is a contractual charge. There is no general statutory
rule in the United States prescribing its invoice contents, its free-time allowance, or its dispute
process. The terms come from the carrier's tariff and the shipper's contract, and they vary. Statements
that treat the ocean rule as governing trucking detention are wrong. Where a shipper wants invoice
discipline on ground detention, it has to write it into the contract.

### 8.5 Invoice and event standards

Freight invoices and shipment events move on electronic data interchange (EDI) standards. In the ANSI
Accredited Standards Committee X12 set, the 210 transaction is the freight invoice, the 214 is the
shipment status message, the 204 is the load tender, and the 997 is the functional acknowledgement.
These carry the fields an audit needs, but they do not carry the contract: the rate, the peg, and the
accessorial schedule live in the shipper's systems and must be joined to the transaction. That join is
the audit.

For parcel and LTL identity, the GS1 SSCC (Serial Shipping Container Code) on the label ties the
physical unit to the shipment record. Where a reweigh or a dimension capture occurs, the scale or
dimensioner reading is the evidence, and it needs an identifier and a timestamp to be usable in a
dispute.

## 9. Industry practice and case evidence

**Discount off tariff is the dominant LTL contracting structure in North America.** The consequence,
described in section 3.5, is that a shipper's effective rate depends on a document the shipper does not
control and may not track. The practical countermeasure used by mature procurement teams is to
normalise every bid to an effective rate per hundredweight at a stated weight and a stated fuel index,
and to require the carrier to state the base tariff version. This is a practice description, not a
measured finding.

**Carriers reset fuel pegs at contract renewal.** Because the surcharge is defined relative to a peg
base, resetting the peg to the current index level sets the surcharge to zero at signature. This is
visible in the arithmetic and is not a secret; it is the reason a fuel clause must be evaluated at a
stated index value rather than at signature, when the surcharge is by construction at or near zero.
Any bid comparison that reads the fuel line on the day of signing will show zero for every carrier.

**Accessorials have been moving from bundled to itemised.** Itemising is not in itself an increase, but
it changes who sees the charge and when. Where a charge was inside a linehaul rate, a dock delay was
invisible to the shipper's operations team. Where it appears as a line, it can be attributed to a
site and a shift. The evidence for the direction of travel is largely trade press and vendor
commentary, grade D and E, and no audited series is available.

**The 2025 NMFC density change is a genuine standards event.** The adoption of a 13-subprovision density
scale effective 2025-07-19 changed the class of many density-rated items. For a shipper, the exposure
is asymmetric: items whose density falls just below a band boundary move to a higher class, and the
change lands on a known date. The NMFTA disposition bulletin is the primary source; carrier customer
advisories reproduce the table but are secondary.

**Parcel divisor changes are a pricing lever that looks like a technicality.** Changing a divisor from
166 to 139, or adding a rule that rounds each dimension up before multiplying, raises billed weight on
light, bulky cartons without any change to the published rate. Both UPS and FedEx have used a 139
divisor for domestic daily services for years, with 166 retained for retail rates. A shipper with
light, bulky products should model the divisor as a contract term and test packaging changes against
it. The specific dates on which rounding rules changed are reported in vendor summaries, grade D, and
should be checked against the carrier guide.

**On freight audit benchmarks, a caution.** The most widely repeated figure in this area is that
roughly one in five freight invoices contains an error. The traceable origin is a shipper survey
published by the *Journal of Commerce* in 2020, reporting an average self-reported invoice inaccuracy
rate of 20.8 percent from a sample of 77 mostly large shippers. That is a self-reported survey figure
with an undefined error metric, not a measured error rate, and it should be cited as such or not at
all. No audited, peer-reviewed measurement of freight invoice error rates was located for this file.
The peer-reviewed literature on freight bill auditing and payment outsourcing exists — Min (2002) is
the standard reference — but it does not supply a leakage benchmark. Any specific leakage percentage
quoted in a vendor proposal should be treated as grade D until the vendor produces the audit population.

## 10. Failure modes and red flags

**1. Rate rows that overlap in time.** Two versions of a contract both satisfy the effective-date
predicate, and the rating join returns two rows. The system takes the first, the last, or the maximum,
depending on an implementation detail nobody chose.
Red flag: the count of matched rate rows per shipment is not checked, or the check reports a
distribution with a tail above one.

**2. The expired rate that still matches.** `effective_to` is null on a superseded row, so the row
remains valid forever. The shipment is rated at a rate from two years ago and the invoice disputes it.
Red flag: a rate table in which the count of rows with a null expiry grows every year.

**3. The discount compared without the base.** The bid sheet ranks carriers by discount percentage.
The carrier with the deepest discount has the highest base tariff and the highest effective rate.
Red flag: a bid comparison table with a discount column and no effective-rate column.

**4. The fuel peg reset to zero at signature.** The fuel clause is evaluated on the day the contract is
signed, when the peg equals the index by construction, and the surcharge reads zero for every carrier.
Red flag: a fuel comparison in which all options are zero, or in which the peg date equals the contract
signature date.

**5. The percentage applied to an unstated base.** Two carriers quote the same percentage against
different application bases. The difference is invisible until the accessorial load grows.
Red flag: a fuel clause that does not state whether accessorials are inside the application base.

**6. The lag treated as zero.** The surcharge is computed from the index on the invoice date rather than
the index publication the contract names. The error is small in stable markets and large at an
inflection.
Red flag: a fuel calculation that reads the index by invoice date with no lag field anywhere in the
schema.

**7. The minimum charge omitted.** The model prices every shipment at the rate, so small shipments are
understated and consolidation appears less valuable than it is.
Red flag: a cost-per-shipment series that falls smoothly as weight falls, with no floor and no visible
step.

**8. The minimum charge applied after discount in the wrong order.** The discount is applied to a charge
that has already been floored, or the floor is applied before discounting, producing a charge below the
absolute minimum.
Red flag: invoices priced below the stated absolute minimum, or a discount that appears to exceed the
stated maximum on small shipments.

**9. Accessorials aggregated into "other".** The long tail is summed into one line, and detention,
liftgate, and reclassification become indistinguishable.
Red flag: an invoice-coding scheme with fewer than about a dozen accessorial codes, or an "other"
bucket above a few percent of freight spend.

**10. The accessorial charged without an event.** A liftgate line appears on a shipment with no service
request and a dock-level address. The charge may still be correct — the driver may have found no dock
— but there is no evidence either way.
Red flag: accessorial lines that cannot be joined to a timestamped event within the shipment window.

**11. Detention clocked from the wrong start point.** The clause says free time begins at arrival; the
carrier clocks from the appointment; the receiver records from check-in. Three clocks, three answers.
Red flag: detention disputes concentrated at one site or one carrier, with amounts clustering just
above the free-time allowance.

**12. The audit set to zero tolerance.** Every variance is disputed, the dispute queue exceeds capacity,
disputes age past the contractual window, and the operation reverts to paying everything.
Red flag: a dispute queue whose median age exceeds the contractual dispute window, or a tolerance of
zero on lines whose mean value is below the cost of investigating them.

**13. Invoice accuracy measured on the wrong population.** Accuracy is computed over invoices that
were audited, not over all invoices, so the metric improves as audit coverage falls.
Red flag: an accuracy metric whose denominator is not stated, or that moves when audit staffing
changes.

**14. The ocean rule applied to trucking.** A ground detention charge is disputed on the basis of the
FMC invoice-content rule, which does not govern it.
Red flag: a contract clause, policy, or training deck that cites 46 CFR part 541 in a domestic
trucking context.

## 11. Assumptions and limits

**1. Every shipment can be rated deterministically from stored data.** The model assumes the contract
covers the lane and the service actually used, and that the shipment's attributes are captured
correctly.
What breaks if wrong: shipments outside contract fall to a tariff or a spot rate that is not in the
model, and the rated total understates cost precisely on the exceptional shipments that matter most.

**2. The rate table is complete and current.** The model assumes all amendments, including those signed
after their effective date, have been loaded.
What breaks if wrong: historical shipments carry a systematic error that is invisible in any single
month and shows up as an unexplained variance against the carrier's own re-rate.

**3. Class is stable across the shipment population.** The model assumes the declared class matches the
class the carrier will apply.
What breaks if wrong: reclassification charges appear as accessorials rather than as rate errors, and
the root cause — item description or packaging — is never investigated.

**4. The fuel index used is the index named in the contract.** The model assumes the series, geography,
and publication timing are the ones the clause names.
What breaks if wrong: a small persistent bias in the fuel line that is impossible to explain to a
carrier without reconstructing the clause from the signed document.

**5. Accessorial events are captured with timestamps.** The model assumes arrival and departure times
exist at the stop level.
What breaks if wrong: detention can be neither validated nor disputed, and the charge becomes a
pass-through by default.

**6. The accessorial schedule is effective-dated like the rate.** The model assumes accessorial prices
carry the same validity discipline as linehaul.
What breaks if wrong: a new schedule applied to old shipments, or the reverse, producing a variance that
looks like a pricing error and is a date error.

**7. Distance is stable for the rating period.** The model assumes the rated distance basis does not
change mid-contract.
What breaks if wrong: a carrier re-rates on a shorter household-goods mileage guide or a different
practical-mileage rule and every line on the lane moves by a few percent with no rate change.

**8. Weight and dimensions are captured once and consistently.** The model assumes declared weight is
the weight the terminal will measure.
What breaks if wrong: reweigh charges and retroactive re-rates, with the shipper's own capture process
as the root cause.

**9. The invoice population is complete.** The audit assumes every invoice the carrier issued arrives in
the audit feed.
What breaks if wrong: leakage is measured on the audited subset, and the shipments most likely to be
misbilled — those with exceptions — are the ones most likely to be missing.

**10. Currency and unit conversions are explicit.** The model assumes every rate row states its
currency, its unit basis, and its rounding rule.
What breaks if wrong: a factor-of-ten unit error, or a currency conversion applied at the wrong date,
that passes every plausibility check because the total is still the right order of magnitude.

**11. Comparisons are made at a stated index value and a stated volume.** The model assumes bids are
normalised before ranking.
What breaks if wrong: the winning bid is the one with the best headline and the worst landed cost, and
the error is only discovered after the volume has moved.

## 12. Exercises

**Exercise 1. Design the rate record.** Write the schema for one lane's contract rate, including every
field needed to rate a shipment without consulting the contract document. State the primary key and
the cardinality constraint that prevents fan-out.

- Deliverable: a table definition in data definition language, or an equivalent column list with types,
  plus a short note on the constraint.
- Acceptance criteria: contains a validity window, a rate basis code, a currency, a unit basis, a
  rounding rule, a minimum charge, and five fuel peg fields; the constraint enforcing at most one
  matching row per (lane, class, date) is stated explicitly.
- Time: 45 minutes.

**Exercise 2. Compute the minimum-charge threshold.** On a lane priced at USD 31.00 per hundredweight
(2026-01-01 rate) with a USD 195.00 minimum (2026-01-01), compute the billed charge and the effective
rate per pound at 150 lb, 400 lb, and 900 lb, all class 85. State the weight at which the minimum stops
binding.

- Deliverable: a four-row table with the arithmetic shown.
- Acceptance criteria: the threshold is derived as a weight, not asserted; the per-pound rate at each
  weight is shown; the ratio between the lightest and heaviest per-pound rate is stated.
- Time: 30 minutes.

**Exercise 3. Compare two fuel pegs.** A linehaul of USD 1,480.00 (2026-03-01 rate) is subject to a
surcharge. Index value is USD 4.24 per gallon (2026-03-03 publication). Peg A is USD 3.85 per gallon
(2025-12-01 contract); Peg B is USD 4.00 per gallon (2026-03-01 contract). Compute both surcharges and
the difference, and state which peg the carrier would prefer and why.

- Deliverable: the two percentages, the two charges, and the difference.
- Acceptance criteria: the arithmetic is shown to two decimal places; the answer explains that the
  lower peg produces the higher charge for the same index; the note identifies that Peg B is what a
  renewal signed at the current index would produce.
- Time: 30 minutes.

**Exercise 4. Decompose a landed cost change.** A shipment costs USD 812.00 in year 1 and USD 967.00 in
year 2 (both invoice totals, rate years 2026-01-01 and 2027-01-01). The linehaul component rose from
USD 760.00 to USD 792.00 and the fuel component from USD 52.00 to USD 55.00. Compute the accessorial
component in each year and the share of the total change attributable to each of the three components.

- Deliverable: a decomposition table with shares summing to 100 percent.
- Acceptance criteria: components are derived rather than assumed; shares sum correctly; the note states
  which component dominates and what operational lever would address it.
- Time: 30 minutes.

**Exercise 5. Apply the density scale.** Four shipments have densities of 3.5, 7.9, 12.0, and 48.0 lb
per cubic foot. Assign the class for each under the 13-subprovision density scale effective 2025-07-19,
and state for each whether an exception item could override the assignment.

- Deliverable: a four-row table with density, band, class, and override note.
- Acceptance criteria: band boundaries are handled correctly at the boundary values; the note states
  that items with handling, stowability, or liability exceptions do not use the full scale.
- Time: 25 minutes.

**Exercise 6. Compute billable weight two ways.** A carton measures 20 by 16 by 14 inches and weighs 31
lb actual. Compute billable weight at divisors of 139 and 166, showing the intermediate volume, the
unrounded quotient, and the rounded result. Then compute the billable weight if one dimension shrinks
by 2 inches, at both divisors.

- Deliverable: a table of four cases with arithmetic.
- Acceptance criteria: rounding is applied at the stated step and not earlier; the change in billed
  weight from the packaging change is stated in pounds and as a percentage.
- Time: 30 minutes.

**Exercise 7. Specify the audit tolerance rule.** Write the rule your audit would apply to a portfolio
of LTL invoices: what tolerance, applied per line or per invoice, with what treatment of lines below the
investigation cost, and with what escalation.

- Deliverable: a rule statement of no more than 250 words, plus a worked example on three invoice lines.
- Acceptance criteria: the tolerance has a stated basis (not "industry standard"); the cost of
  investigating a line is acknowledged; the rule states what happens when the dispute window closes
  before resolution.
- Time: 45 minutes.

**Exercise 8. Design detention event capture.** Specify the fields, sources, and validation rules needed
to make a ground detention charge auditable, starting from the moment the trailer arrives at the
consignee. Include the free-time clock and its start point.

- Deliverable: a field list with source system and validation rule for each, plus a decision on what to
  do when the receiver's records and the carrier's records disagree.
- Acceptance criteria: every field has a source and a type; the start point of the free-time clock is
  named; the disagreement rule is stated and is implementable without human judgement on every case.
- Time: 45 minutes.

**Exercise 9. Model a general rate increase.** A tariff-based lane carries an annual spend of
USD 480,000 (2026 plan, based on 2025 actuals). A general rate increase of 5.9 percent is announced with
an effective date of 2026-09-01. Compute the effect on calendar-year 2026 spend if volume is flat and
evenly distributed through the year, and state what the model needs in order to be correct.

- Deliverable: the revised annual figure, the four-month arithmetic, and a list of the assumptions the
  model depends on.
- Acceptance criteria: the increase is applied only to the portion of the year after the effective date;
  the answer distinguishes the plan from the commitment; the assumption list includes whether the
  increase applies to accessorials and to the fuel peg.
- Time: 30 minutes.

## 13. Self-check answers

**Answer 1.** A workable definition is: `contract_rate(lane_id, mode, service_level, class_code,
weight_band_low, weight_band_high, effective_from, effective_to, rate_basis_code, base_tariff_per_cwt,
discount_pct, min_charge, absolute_min, fuel_index_series, fuel_index_geography, fuel_peg_base,
fuel_peg_base_date, fuel_lag_days, fuel_application_base, fuel_cap_pct, currency, unit_basis,
rounding_rule)`. The primary key is a surrogate identifier plus a version number. The cardinality
constraint is a unique index on (lane_id, mode, service_level, class_code, weight_band_low,
effective_from), with a check that `effective_to` is either null or strictly greater than
`effective_from`, and a non-overlap check per key. Without the non-overlap check, two rate rows can
satisfy the same shipment date and the join fans out. `base_tariff_per_cwt` is required rather than
optional because a discount-based contract cannot be rated without it. `currency` and `unit_basis` are
required because a rate row without them is not a price.

**Answer 2.** At USD 31.00 per hundredweight (2026-01-01) and a USD 195.00 minimum (2026-01-01):

- 150 lb equals 1.5 hundredweight; 1.5 × 31.00 = USD 46.50; the minimum binds, so the charge is
  USD 195.00; the effective rate is 195.00 / 150 = USD 1.30 per lb.
- 400 lb equals 4.0 hundredweight; 4.0 × 31.00 = USD 124.00; the minimum binds; the charge is
  USD 195.00; the effective rate is 195.00 / 400 = USD 0.4875 per lb.
- 900 lb equals 9.0 hundredweight; 9.0 × 31.00 = USD 279.00; the rate binds; the effective rate is
  USD 0.31 per lb.

The threshold is 195.00 / 31.00 = 6.2903 hundredweight, or 629 lb. The lightest shipment pays 1.30 /
0.31 = 4.2 times the per-pound rate of the heaviest. The general lesson: the minimum charge is a
currency floor that becomes a per-unit penalty as weight falls, and the penalty grows without bound as
weight approaches zero.

**Answer 3.** Peg A: (4.24 - 3.85) / 3.85 = 0.39 / 3.85 = 0.101299, or 10.13 percent; charge is
1,480.00 × 0.101299 = USD 149.92 (2026-03-01 linehaul). Peg B: (4.24 - 4.00) / 4.00 = 0.24 / 4.00 =
0.06, or 6.00 percent; charge is 1,480.00 × 0.06 = USD 88.80. Difference: USD 61.12 per shipment in the
carrier's favour under Peg A. The lower peg always produces the higher charge at a given index, because
the surcharge measures movement above the peg. Peg B is what a renewal signed on 2026-03-01 would
produce, since the peg is set at the index value on that date; this is why a fuel clause must be
evaluated at a stated future index value and not at signature.

**Answer 4.** Year 1 accessorials: 812.00 - 760.00 - 52.00 = USD 0.00. Year 2 accessorials:
967.00 - 792.00 - 55.00 = USD 120.00. Total change: 967.00 - 812.00 = USD 155.00. Linehaul change:
+USD 32.00, or 20.6 percent of the change. Fuel change: +USD 3.00, or 1.9 percent. Accessorial change:
+USD 120.00, or 77.4 percent. The shares sum to 99.9 percent because of rounding to one decimal; at full
precision they sum to 100 percent. Accessorials dominate, so the lever is operational — site access,
appointment discipline, packaging — not procurement. A rate negotiation targeting the linehaul would
address one fifth of the problem.

**Answer 5.** At 3.5 lb per cubic foot the band is 2 to under 4, class 250. At 7.9 the band is 6 to
under 8, class 125. At 12.0 the band is 12 to under 15, class 85. At 48.0 the band is 35 to under 50,
class 55. Boundary handling is the trap: 12.0 belongs to the 12 to under 15 band, not the 10 to under 12
band, because the upper bound is exclusive. For each, an exception item could override: an item carrying
a handling, stowability, or liability exception does not use the full density scale, so a fragile item
at 48.0 lb per cubic foot could still carry a class above 55. The item number governs; density is the
default path, not the rule.

**Answer 6.** Volume is 20 × 16 × 14 = 4,480 cubic inches. At 139: 4,480 / 139 = 32.23, rounded up to
33 lb billable, against 31 lb actual. At 166: 4,480 / 166 = 26.99, rounded up to 27 lb billable, which
is below actual, so 31 lb actual is billed. With one dimension reduced by 2 inches, volume is
20 × 16 × 12 = 3,840 cubic inches. At 139: 3,840 / 139 = 27.63, rounded to 28 lb. At 166:
3,840 / 166 = 23.13, rounded to 24 lb, so 31 lb actual still bills. The packaging change reduces
billed weight by 5 lb at the 139 divisor (33 to 28, a 15.2 percent reduction) and by zero at 166,
because actual weight still binds. The lesson is that a packaging change only pays where dimensional
weight is the binding constraint, and which divisor applies determines whether it binds at all.

**Answer 7.** A workable rule: pay any line whose absolute variance from the rated amount is at or below
the greater of 1.0 percent of the rated amount or USD 5.00 (2026 policy), because the fully loaded cost
of investigating a line is above USD 5.00 at any realistic audit staffing level. Aggregate all
tolerated variances by carrier and month; if the aggregate exceeds 0.25 percent of that carrier's
monthly invoiced value, escalate the pattern rather than the lines, because a systematic small variance
is a rate-table error and not a keying error. Dispute everything above tolerance within 10 days of
receipt, which leaves margin inside a 30-day window. Where the window closes before resolution, pay
under protest and record the dispute as a receivable, with the protest documented on the remittance.
On three lines: a USD 0.40 variance on a USD 120.00 line is 0.33 percent and below USD 5.00, so it is
tolerated; a USD 42.00 variance on a USD 900.00 line is 4.7 percent and disputed; a USD 6.50 variance on
a USD 4,000.00 line is 0.16 percent and tolerated, but if the same USD 6.50 appears on 200 lines from
one carrier in one month, the pattern is escalated.

**Answer 8.** Required fields: trailer identifier and carrier SCAC (Standard Carrier Alpha Code);
shipment identifier; stop identifier; arrival timestamp at the consignee gate, from the carrier's
telematics or the guardhouse log; check-in timestamp, if distinct; dock-assignment timestamp, from the
warehouse management system (WMS); departure timestamp; free-time allowance in minutes, from the
contract; free-time start point, from the contract, one of arrival, check-in, or appointment; the
appointment timestamp if one exists; the receiver's receiving-hours record; and the invoice line
reference. Validation rules: arrival must precede check-in, which must precede departure; the free-time
start point must be one of the three enumerated values and must match the contract version in force on
the shipment date; the computed detention minutes must equal the elapsed time from the start point to
departure minus the free-time allowance, floored at zero. Where carrier and receiver records disagree by
more than 15 minutes, the rule should be to accept the earlier of the two arrival timestamps and the
later of the two departure timestamps, and to log the disagreement against the site for review, on the
reasoning that the charge is for the time the equipment was actually unavailable and the shipper
controls the site's record quality.

**Answer 9.** The increase applies for four months of twelve, from 2026-09-01 to 2026-12-31. The
post-increase portion is 480,000 × 4/12 = USD 160,000 (2026 plan, pre-increase). The increase on that
portion is 160,000 × 0.059 = USD 9,440 (2026 money). Revised annual spend is USD 489,440. The model
needs: confirmation that the increase applies to the tariff base and not to contract rates already
fixed; a decision on whether accessorials escalate with the same percentage, since many tariffs apply
the general increase to accessorial schedules as well; confirmation that the fuel peg is unchanged,
because a simultaneous peg reset can offset or amplify the headline; and the volume distribution, since
an evenly distributed assumption is a plan assumption and not a commitment. If the 5.9 percent applies
to accessorials at a 9 percent share of spend, the increase is 9,440 × 1.09 = USD 10,290, and the
revised figure is USD 490,290. The difference between the two answers is a contract-reading question,
not an arithmetic one.

## 14. Interview questions

**1. Why is a freight rate a function rather than a number?**

Strong answer: because the price depends on attributes the quote does not fix — the effective date,
the base tariff version, the fuel peg, and the accessorial schedule — and the same quoted rate can
produce different invoices. The candidate names at least three of those and connects them to the fact
that a bid comparison has to normalise to an effective rate at a stated index and a stated weight.
Weak answer: describes freight as negotiated and variable without naming a mechanism, or treats the
rate as the price and accessorials as an unrelated cost.

**2. A carrier offers 64 percent off tariff against a competitor's 62 percent. Is that better?**

Strong answer: not necessarily, and the question cannot be answered without the base tariff. Works the
arithmetic on a hypothetical base to show that a 2-point discount improvement with a 9 percent base
increase produces a 3.6 percent effective rate increase. States that the comparison must be made on
effective rate per hundredweight at a stated weight and date.
Weak answer: says yes, or says it depends on service, without identifying the base tariff as the
missing term.

**3. How does a fuel surcharge work, and why can two carriers quoting the same percentage charge
different amounts?**

Strong answer: the percentage is the movement of a published index above a peg base, applied to a
defined base. Carriers differ in the index series and geography, the peg value and its date, the lag
between observation and application, the application base — linehaul only or linehaul plus
accessorials — and any cap. Names at least three of these and gives the dollar consequence at a stated
accessorial load.
Weak answer: describes the surcharge as a percentage of the freight bill and stops there.

**4. What does a minimum charge do to the economics of a small shipment?**

Strong answer: it sets a currency floor, so the effective rate per pound rises as weight falls. Gives
the threshold as minimum divided by rate, and notes that below it the shipment pays a per-unit penalty
that grows without bound. Connects it to order minimums and consolidation as economic levers.
Weak answer: says small shipments cost more per pound because of handling, without the floor mechanism,
or asserts a fixed ratio without deriving it.

**5. Walk me through rating a shipment. What is the join?**

Strong answer: names the keys — lane, mode, service level, class or density band, weight band, date —
and states that the join must return exactly one row. Describes the two failure shapes, null match and
fan-out, and says both are detected by counting matches before summing money. Mentions that the rate
row carries currency, unit basis, and rounding rule.
Weak answer: describes rating as multiplying weight by rate, with no validity window and no discussion
of what happens when no row matches.

**6. What is accessorial creep and how would you detect it?**

Strong answer: the accessorial share of landed cost rises while quoted rates improve. Detection
requires a matched cohort — the same shipments or a stable lane mix across periods — decomposed into
linehaul, fuel, and accessorial components, with the accessorial component further split by code and
by site. Notes that an unmatched comparison confounds mix with behaviour.
Weak answer: says accessorials are increasing, quotes a percentage without a denominator, or attributes
it to carrier pricing without testing the rate component.

**7. Which accessorials can a shipper actually control?**

Strong answer: distinguishes by trigger. Reweigh, reclassification, address correction, oversize, and
hazmat are driven by the shipper's master data and packaging. Liftgate, residential, limited access,
and inside delivery are driven by site or order attributes. Detention and appointment compliance are
shared with the consignee. Fuel and most carrier equipment charges are not controllable. Connects each
to a specific fix.
Weak answer: lists accessorials without assigning an owner, or claims all of them are negotiable.

**8. What is the difference between demurrage and detention, and does a federal rule govern both?**

Strong answer: in ocean shipping, demurrage is charged for a container inside the terminal beyond free
time and detention for one outside it, typically at the consignee. Notes the FMC rule at 46 CFR part
541, effective 2024-05-28, covering invoice contents, the 30-day issuance deadline, and the dispute
process, and notes that 46 CFR 541.4 was set aside on appeal in September 2025 while the rest remains
in effect. States clearly that ground detention in trucking is contractual and not governed by that
rule.
Weak answer: uses the terms interchangeably, or claims a federal rule governs trucking detention.

**9. What data does a freight audit need to be possible?**

Strong answer: the effective-dated contract with its base tariff, the rated shipment, the invoice at
line level, the event stream for accessorials, and a tolerance rule. Notes that the audit is a join
between the invoice and the contract, that the contract is the part most often missing, and that
without event data an accessorial can be neither validated nor disputed.
Weak answer: says the audit needs the invoice and the rate, omitting effective dates, events, and the
base tariff.

**10. How would you set an audit tolerance?**

Strong answer: starts from the cost of investigating a line rather than from an industry number, sets a
floor in currency and a percentage, aggregates tolerated variances by carrier and month to catch
systematic small errors, and states what happens when the dispute window closes before resolution.
Notes that zero tolerance collapses the process.
Weak answer: quotes a percentage as standard practice without a basis, or sets zero tolerance without
acknowledging the queue.

**11. A vendor claims that a fifth of freight invoices contain errors. How do you respond?**

Strong answer: asks for the population, the definition of error, whether invoices were actually
audited, and the date. Notes that the widely repeated figure traces to a self-reported shipper survey
and is not a measured error rate, and that no audited benchmark was found. Proposes measuring the
shipper's own error rate on a sample before committing to a recovery target.
Weak answer: accepts the figure as fact and builds a business case on it.

## 15. Key terms

- **Linehaul** — the charge for moving freight from origin to destination before surcharges and accessorials.
- **Rate basis** — the unit and formula a rate is expressed in, such as per hundredweight or per mile.
- **Tariff** — a published schedule of rates, rules, and charges.
- **Contract rate** — a negotiated price for named lanes with a validity window.
- **Effective date** — the date from which a rate or rule applies.
- **Freight class** — a number from 50 to 500 expressing an item's transportability.
- **NMFC** — the National Motor Freight Classification, the North American LTL class standard.
- **NMFTA** — the National Motor Freight Traffic Association, the non-profit body that publishes the NMFC.
- **Hundredweight** — a unit of 100 pounds, the traditional LTL quoting basis.
- **Minimum charge** — a currency floor on a shipment's charge, however small the shipment.
- **Minimum-charge threshold** — the weight below which the minimum charge binds, equal to the minimum divided by the rate.
- **Break weight** — the weight at which the next rate band produces a lower total charge.
- **Discount off tariff** — a percentage reduction applied to a published base rate.
- **Base tariff** — the published rate a discount is applied to, which must be versioned by date.
- **FAK** — freight all kinds, a single rate applied to a mixed shipment regardless of class.
- **Spot rate** — a one-off price for a single shipment or a short validity period.
- **General rate increase** — a published percentage increase applied to a tariff on a stated date.
- **Fuel surcharge** — a charge that passes through fuel price movement by a formula tied to an index.
- **Fuel index** — the published price series a surcharge is pegged to.
- **Peg base** — the index price at which the fuel surcharge is defined to be zero.
- **Application base** — the charge amount the fuel percentage is applied to.
- **Surcharge lag** — the delay between an index observation and its application to rates.
- **Accessorial** — a charge for a service beyond dock-to-dock movement.
- **Detention** — a contractual charge for holding a carrier's equipment beyond free time in trucking.
- **Demurrage** — a charge for a container held inside a terminal beyond free time in ocean shipping.
- **Free time** — the period before detention or demurrage begins.
- **Dimensional weight** — volume converted to a weight equivalent by a divisor.
- **DIM divisor** — the cubic inches per pound used in the dimensional weight formula.
- **Freight audit and pay** — the process of checking carrier invoices against contract before payment.
- **Accessorial creep** — the rise in accessorial share of landed cost while quoted rates improve.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Min, H. (2002). "Outsourcing Freight Bill Auditing and Payment Services." *International Journal of
  Logistics Research and Applications*. DOI: 10.1080/13675560210148687. Survey-based empirical study of
  the freight bill audit and payment function; supports the description of the audit process. It does
  not supply a leakage or error-rate benchmark, and this file does not attribute one to it.

**Standards, regulation, and official statistics — Grade B**

- National Motor Freight Traffic Association. *National Motor Freight Classification*. Class range 50
  to 500, assigned on density, handling, stowability, and liability. Supports section 8.1. Accessed
  2026-11-03.
- NMFTA, Freight Classification Development Council. "Notice of Disposition — Docket 2025-1,"
  Disposition Bulletin 1371, dated 2025-03-21, amendments published 2025-05-10, effective 2025-07-19.
  Establishes the 13-subprovision density scale. The table was read from the published disposition as
  summarised in the bulletin and corroborated by carrier advisories; the PDF text could not be
  extracted directly in this session.
- US Energy Information Administration. "Methodology for EIA Weekly On-Highway Diesel Fuel Price
  Estimates," Form EIA-888. Weekly collection on Mondays, 590 retail outlets, ultra-low-sulfur
  on-highway diesel including taxes. Supports section 8.3. Accessed 2026-11-03.
- US Energy Information Administration. "Gasoline and Diesel Fuel Update," publication date change
  notice of 2025-03-18, moving release to Tuesday mornings from the week of 2025-04-07, and the holiday
  release schedule. Supports the publication timing in section 8.3. Accessed 2026-11-03.
- European Commission, Directorate-General for Energy. Weekly Oil Bulletin. Member-state retail prices
  for automotive gas oil with and without taxes; data submitted Wednesdays, bulletin distributed
  Thursdays. Supports the European counterpart in section 8.3. Accessed 2026-11-03.
- US Federal Maritime Commission. Demurrage and Detention Billing Requirements, 46 CFR part 541,
  Docket No. FMC-2022-0066, RIN 3072-AC90, 89 FR 14330, published 2024-02-26, effective 2024-05-28.
  Supports section 8.4.
- US Federal Maritime Commission. "U.S. Court of Appeals Issues Decision in Case on Demurrage and
  Detention Billing Practices," 2025-09-23, recording that the D.C. Circuit set aside only 46 CFR
  541.4 and that the rest of the rule remains in effect. Supports the current-status paragraph in
  section 8.4. Accessed 2026-11-03.

**Company case and investor claim — Grade C**

- UPS. *Rate and Service Guide*, daily rates, 2026 edition. Dimensional weight divisor of 139 cubic
  inches per pound for daily rates and 166 for retail rates, with dimensions rounded up to the next
  whole inch and the result rounded up to the next whole pound. Supports section 8.2. The carrier's
  published guide is the primary source but is a commercial term, not peer-reviewed evidence, and the
  current guide should be checked for the service actually used.

**Vendor, analyst, and trade — Grade D**

- Shipping-software vendor summaries of 2026 dimensional weight divisors across UPS, FedEx, and USPS,
  including the reported 2025-08-18 change to rounding each dimension up before multiplying. Used only
  to flag that the rounding rule has changed; vendor summaries are not evidence of the carrier's terms.
- Third-party logistics provider and freight broker guides to the 2025 NMFC density changes, reproducing
  the 13-tier table and the 2025-07-19 effective date. Corroborates the NMFTA bulletin; not evidence in
  its own right.
- *Journal of Commerce* / IHS Markit, "Driving Strategic Value Through Outsourcing," 2020 freight
  payment benchmark study. Reports a self-reported average invoice inaccuracy rate of 20.8 percent from
  77 mostly large shippers, with an undefined error metric and no invoice audit. Cited in section 9
  only to explain why the figure should not be used as a benchmark. The PDF could not be parsed in this
  session; the sample size and figure are as reported by secondary summaries and should be verified
  against the document before citation.

**Press summary and anecdote — Grade E**

- Trade press reporting of historical post-audit invoice error rates by mode, and law-firm client
  alerts summarising 46 CFR part 541. The law-firm summary was used to cross-check the invoice-content,
  timing, and dispute provisions; the regulation itself is the authority.

> **Note on verification.** The NMFTA docket, the EIA series, the European Commission bulletin, and the
> FMC rule and its appellate status were checked against their publishing bodies on 2026-11-03. The
> carrier dimensional weight terms could not be read directly: the carrier rate guide returned no
> extractable text and the carrier portal returned an access-denied response, so the divisor is stated
> as the carrier publishes it and graded C. No audited freight-invoice leakage benchmark was located,
> and none is asserted anywhere in this file.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Part map: [`../_index.md`](../_index.md)
- Previous topic, mode economics: [`01_freight-modes.md`](01_freight-modes.md)
- Next topic, lane and tender analytics: `03_lane-tender-analytics.md` (planned, written next in
  this module).
- Related topic, consolidation and cube: `04_consolidation-cube.md` (planned, written later in this
  module).
- Inventory economics and order quantity: [`../10_inventory_optimization_multi_echelon/02_eoq-limits.md`](../10_inventory_optimization_multi_echelon/02_eoq-limits.md)
- Network cost-to-serve: [`../11_network_design_optimization/02_graph-cost-to-serve.md`](../11_network_design_optimization/02_graph-cost-to-serve.md)
- Data joins and event models: [`../../01_foundations/03_data_engineering_interoperability/04_sql-supply-chain-patterns.md`](../../01_foundations/03_data_engineering_interoperability/04_sql-supply-chain-patterns.md)
- Module survey file: [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
