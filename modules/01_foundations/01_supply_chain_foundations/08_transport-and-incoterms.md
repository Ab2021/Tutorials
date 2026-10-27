# Transportation Modes and Incoterms 2020

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04, 05, 06, 07
> Version 2026.5

---

## 1. Why this topic exists

Two decisions sit between a purchase order and a delivered unit, and both are routinely made by
people who cannot see the arithmetic behind them:

1. **Which mode** carries the goods.
2. **Which contract term** decides who pays for, and who bears the risk for, the goods at each
   point along the way.

Neither decision appears in a forecast. Both determine the number that the forecast is supposed
to optimise against. A landed-cost spread of 1–3% between two otherwise identical sourcing
options is larger than most companies' net margin on the product, and it is routinely decided by
whoever last edited the purchase order template.

The reason this is a data science topic rather than a procurement topic is that **the invoice
price is the only cost most systems record as the cost of the good**. Freight sits in a separate
cost centre. Duty sits in a tax account. Demurrage sits in an exception report that nobody
joins. The working-capital cost of transit time sits nowhere at all, because it is not
transacted by anyone. Any model that ranks suppliers, chooses a source, or optimises an order
quantity on invoice price is optimising a number that appears in the ERP and appears nowhere in
the economics.

Three recurring failures show the shape of the problem:

1. **The supplier scorecard that rewards the wrong supplier.** A scorecard ranks suppliers on
   unit price and on-time delivery. The winning supplier ships free carrier (FCA) from a plant
   eight hours from the port, so the buyer absorbs origin haulage, export documentation, and a
   longer cash cycle. The runner-up quotes delivered at place (DAP) and is 2% dearer per unit.
   The scorecard selects the first. The landed cost is 1.5% higher. Nobody computes it, so
   nobody notices, and the error compounds across every purchase order for three years.
2. **The "cost per kg" comparison that inverts the mode decision.** An analyst compares air at
   USD 4.50/kg against ocean at USD 0.28/kg and concludes that air is sixteen times more
   expensive. That is true per kilogram. It is not true per unit of *decision value*, because
   the ocean option adds thirty days of pipeline inventory to a product whose carrying cost per
   unit per day is high. For high-value, low-weight goods the financing cost of the sea voyage
   can exceed the ocean freight itself — a result that is arithmetically trivial and almost
   never computed.
3. **The Incoterm that silently moves the inventory.** A company moves a lane from cost,
   insurance and freight (CIF) to DAP to "simplify" the process. Under CIF the buyer owns the
   goods from the ship's rail at origin; under DAP the seller owns them to the destination
   address. In-transit inventory vanishes from the buyer's balance sheet, the inventory model
   keeps forecasting the old pipeline, and the service level drops because of a three-letter code
   in a contract.

Each of these is an accounting-of-reality failure, not a modelling failure. The fix is a correct
model of what a transport contract actually buys.

**Why this matters before the modelling topics.** Module 12 builds routing and last-mile models.
Module 11 builds network design models. Module 36 treats freight markets and trade policy.
Module 41 treats maritime, rail, and air operations. All four assume you can read a rate
quotation, decompose it, and say which components are variable, which are fixed, which are
contractual, and which party bears each. That is this topic.

This topic follows manufacturing (Topic 07) because goods must exist before they can move, and
precedes warehousing and sourcing (Topic 09) because the Incoterm decides where the goods land and
who is responsible for them when they arrive.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compare** six transport modes on cost per tonne-kilometre, transit time, reliability,
   minimum economic shipment size, and the cargo characteristics each serves, and **explain**
   why a per-kilogram comparison misleads.
2. **Compute** chargeable weight and revenue tons for air, less-than-container-load (LCL) ocean,
   and road less-than-truckload (LTL) consignments, and **identify** from density alone whether
   a consignment will be weight-rated or measure-rated.
3. **Reproduce** the eleven Incoterms 2020 rules in a table showing carriage, risk transfer,
   export clearance, import clearance, and insurance obligation, and **state** which are
   restricted to sea and inland waterway carriage.
4. **Diagnose** the two most common term misapplications — free on board (FOB) used for
   containerised cargo, and delivered duty paid (DDP) used by a seller with no import presence —
   and **explain** the cost and compliance consequence of each.
5. **Separate** the three independent clocks in a trade transaction: risk transfer under the
   Incoterm, title transfer under the sale contract, and settlement under the payment terms.
6. **Build** a full landed-cost schedule for a named shipment, showing every component, its
   owner, its currency, and its accounting treatment.
7. **Evaluate** whether two offers on different Incoterms are comparable, and **compute** the
   landed-cost difference including duty base effects and working capital.
8. **Explain** the carrier liability limit that applies to a given mode and **justify** why cargo
   insurance is a required cost rather than an optional one.

## 3. The core mental model

Hold one picture in your head: a single line from the seller's loading dock to the buyer's
receiving dock, with **two ladders standing on it** — one for cost, one for risk — and the
Incoterm chooses where each ladder ends.

```text
   SELLER'S PREMISES            ORIGIN PORT/TERMINAL          DESTINATION           BUYER'S PREMISES
   (works, factory)             (export clearance)            (import clearance)    (named place)

   EXW   cost  S|                                             
         risk  S|
                                                                                    
   FCA   cost  S|=========>|T
         risk  S|=========>|T
                                                                                    
   FOB   cost  S|=========>|=====>|V
         risk  S|=========>|=====>|V
                                                                                    
   CFR   cost  S|=========>|=====>|V================>|T======>|
         risk  S|=========>|=====>|V
                                                                                    
   CIF   cost  S|=========>|=====>|V================>|T======>|
         risk  S|=========>|=====>|V
                                                                                    
   DAP   cost  S|=========>|=====>|V================>|T======>|=========>|B
         risk  S|=========>|=====>|V================>|T======>|=========>|B
                                                                                    
   DDP   cost  S|=========>|=====>|V================>|T======>|=========>|B=====>|
         risk  S|=========>|=====>|V================>|T======>|=========>|B=====>|

   S = seller   T = terminal/forwarder   V = vessel   B = buyer
   | marks the end of the seller's obligation on that ladder
```

Read the CFR and CIF rows twice. They are the only shape in the diagram where the two ladders
disagree. **The seller pays the freight all the way to the destination port, but the risk leaves
the seller when the goods are on board at origin.** For roughly four weeks, the seller is paying
for a voyage during which the buyer already owns the risk. That single mismatch — cost follows
one clock, risk follows another — explains most of the confusion in trade finance, and it is the
reason CIF and CFR require the buyer to insure goods that the seller is still paying to move.

The picture has five properties worth memorising.

**Property 1 — The Incoterm allocates cost and risk, and nothing else.** It does not transfer
title. It does not set payment terms. It does not determine when revenue is recognised. It does
not, by itself, make the seller the importer of record in most jurisdictions. Those are four
separate instruments, and meetings conflate all five.

**Property 2 — The invoice price moves with the term, so invoice prices on different terms are
not comparable.** A DAP price contains freight that an FCA price does not. Comparing them
directly is comparing a delivered cost against a factory cost and calling the difference a price
difference. Section 6 shows a case where the DAP invoice is 0.85% higher and the landed cost is
1.23% lower.

**Property 3 — Mode choice is driven by value density and time value, not by freight rate per
kilogram.** The correct comparison is cost per unit of cargo moved against the carrying cost of
the time saved, plus the service consequence of being slow or unreliable. Freight rate is one
input to that comparison and rarely the deciding one.

**Property 4 — Every mode has a liability regime, and every regime caps the carrier's
obligation far below the value of the cargo.** Ocean carriage under the Hague-Visby Rules caps
at 2 Special Drawing Rights (SDR) per kilogram or 666.67 SDR per package, whichever is higher.
Air under the Montreal Convention caps at 26 SDR per kilogram. Road under the Convention on the
Contract for the International Carriage of Goods by Road (CMR) caps at 8.33 SDR per kilogram.
For most manufactured goods these caps recover a small fraction of the loss. Cargo insurance is
therefore a cost of doing business, not a discretionary add-on.

**Property 5 — Reliability is a separate variable from speed, and it is usually worth more.**
A 28-day ocean transit with a 60% on-time probability produces more safety stock than a 35-day
transit with a 95% probability. A planner who chooses a mode on transit time alone has chosen on
the wrong variable.

### 3.1 The two questions a term answers

Every Incoterm answers two questions: who arranges and pays for each leg, and where risk
transfers. Everything else — export declaration, import declaration, security screening, the
insurance obligation — elaborates those two. If you can state the cost point and the risk point
for a rule, you understand it well enough to use it.

### 3.2 The three clocks

A trade transaction runs three clocks that are frequently assumed to be one:

```text
   RISK      |---- Incoterm says who bears loss from this point ----|
   TITLE     |---- Sale contract says when ownership passes --------|
   CASH      |---- Payment terms say when money moves --------------|

   time --->
```

In the simplest case all three align. In practice they diverge, and each divergence is a
nameable risk: **risk before title** (the buyer bears loss on goods it does not yet own, which
insurance usually covers but which complicates the claim); **title before risk** (a retention-of-
title seller that has delivered under DAP has given up possession and risk while keeping
ownership, which is contested territory in insolvency); and **cash before risk** (payment in
advance leaves cash gone before any Incoterm right attaches, which is why documentary credits
exist).

The Incoterm answers the risk question. Nothing in the Incoterm answers the other two.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Incoterms | International Commercial Terms: eleven ICC rules defining delivery, cost, and risk between buyer and seller | Treated as a payment term or a title-transfer mechanism | Three-letter rule plus named place, e.g. `FCA Yantian Incoterms 2020` |
| ICC | International Chamber of Commerce, the body that publishes and revises the rules | Assumed to be a regulator with legal force | Not a coding system |
| EXW | Ex Works: seller makes goods available at its own premises, not loaded | Used when the seller can actually export-clear, which EXW does not require | Any mode |
| FCA | Free Carrier: seller delivers cleared for export to a carrier nominated by the buyer at a named place | Used with an on-board bill of lading without the agreed optional mechanism | Any mode |
| FAS | Free Alongside Ship: seller delivers alongside the vessel at the named port | Used for containerised cargo, where "alongside" has no operational meaning | Sea and inland waterway only |
| FOB | Free On Board: risk passes when goods are on board the vessel at the named port of shipment | Used for containers, where delivery happens at the terminal before loading | Sea and inland waterway only |
| CFR | Cost and Freight: seller pays freight to the destination port, risk passes on board at origin | Assumed to mean the seller bears risk to destination | Sea and inland waterway only |
| CIF | Cost, Insurance and Freight: as CFR plus seller-provided insurance at Institute Cargo Clauses (C) minimum | Assumed to include all-risk cover | Sea and inland waterway only |
| CPT | Carriage Paid To: seller pays carriage to the named destination, risk passes on handing to the first carrier | Confused with DAP because both name a destination | Any mode |
| CIP | Carriage and Insurance Paid To: as CPT plus seller-provided insurance at Institute Cargo Clauses (A) | Assumed to be identical to CIF with a different name | Any mode |
| DAP | Delivered at Place: seller delivers at destination ready for unloading, buyer clears import | Assumed to include duty | Any mode |
| DPU | Delivered at Place Unloaded: as DAP but seller must unload at destination | Treated as a synonym for DAP | Any mode; the only rule requiring seller unloading |
| DDP | Delivered Duty Paid: seller clears import and pays duty; maximum seller obligation | Used by sellers with no import establishment or VAT registration | Any mode |
| Named place | The location that completes an Incoterm, e.g. a terminal, a warehouse, a door | Omitted, leaving the rule legally incomplete | Free text, but must be precise |
| TEU | Twenty-foot Equivalent Unit: one 20 ft container | Used as a measure of cargo rather than of slot capacity | 1 TEU = one 20 ft box; 1 FEU = 2 TEU |
| FEU | Forty-foot Equivalent Unit | Assumed to carry twice the payload of a TEU | 1 FEU = one 40 ft box |
| FCL | Full Container Load: one shipper's cargo occupies the whole container | Assumed to mean the container is full by weight or volume | Container count |
| LCL | Less than Container Load: cargo consolidated with other shippers' cargo | Treated as a small FCL | Revenue tons |
| FTL | Full Truckload | Confused with FCL | Vehicle count |
| LTL | Less than Truckload | Confused with parcel | Revenue weight or pallet positions |
| Chargeable weight | The greater of actual gross weight and volumetric weight, used as the rating basis | Assumed to equal actual weight | Kilograms, rounded per carrier rule |
| Volumetric weight | Volume converted to a notional weight by a carrier divisor | Computed with the wrong divisor | kg = (L x W x H in cm) / divisor |
| Revenue ton | Ocean LCL rating unit: the greater of one cubic metre and one metric tonne | Assumed to be a metric tonne | 1 RT = max(1 m3, 1,000 kg) |
| Stowage factor | Volume occupied per unit of weight for a commodity, including unavoidable empty space | Confused with density | m3 per tonne, or cubic feet per long ton |
| CBM | Cubic metre | Used loosely as "one unit of freight" | m3 |
| Breakbulk | Cargo shipped as individual pieces rather than in containers | Confused with bulk | Tonnes or pieces |
| Bulk | Loose cargo carried in a vessel's hold or a tank, such as grain, ore, or liquid | Confused with breakbulk | Tonnes or m3 |
| Intermodal | A movement using two or more modes under one contract and one unit of equipment | Used to mean any multi-leg journey | Contract and equipment identifier |
| Demurrage | Charge for cargo or a container remaining at the terminal beyond free time | Confused with detention | Currency per container per day |
| Detention | Charge for a container held outside the terminal beyond free time | Confused with demurrage | Currency per container per day |
| Free time | The period before demurrage or detention begins | Assumed to be standard across carriers and ports | Days, per carrier tariff |
| THC | Terminal Handling Charge, levied at origin and destination | Assumed to be included in the ocean freight | Currency per container |
| B/L | Bill of Lading: the transport document that is also a receipt and, when negotiable, a document of title | Treated as a mere waybill | Document number plus type |
| AWB | Air Waybill: the air transport document, non-negotiable | Treated as a document of title, which it is not | 11-digit airline prefix plus serial |
| House B/L | Forwarder-issued bill of lading, distinct from the carrier's master B/L | Assumed to carry the same rights as a master B/L | Document number |
| HS code | Harmonized System classification, six digits internationally, extended nationally | Assumed to be the same everywhere beyond six digits | HS6 plus national tariff suffix |
| Customs value | The value on which duty is assessed, normally transaction value adjusted to a named delivery basis | Assumed to equal the invoice price | Currency |
| CIF value | Cost, insurance, and freight value, the duty base used by the EU and most markets | Confused with the CIF Incoterm | Currency |
| Duty | The tariff payable on import, set by HS classification and origin | Assumed to be a single national rate | Percentage of customs value |
| Preferential origin | Origin status that qualifies goods for a reduced or zero rate under a trade agreement | Confused with non-preferential origin marking rules | Origin rule set per agreement |
| CBAM | Carbon Border Adjustment Mechanism: EU charge on embedded emissions in covered goods | Assumed to be a customs duty with the same mechanics | EUR per tonne of embedded CO2e |
| VAT | Value Added Tax on import, normally recoverable by a registered trader | Treated as a cost when it is a cash-flow item | Percentage of customs value plus duty |
| SDR | Special Drawing Right: the IMF unit of account used to express liability limits | Converted at an arbitrary exchange rate without stating the date | ISO 4217 code `XDR` |
| Hague-Visby | International regime limiting ocean carrier liability | Assumed to apply worldwide; several states apply Hamburg or national law | 666.67 SDR per package or 2 SDR per kg |
| Montreal Convention | International regime for air carrier liability | Quoted at the superseded 22 SDR/kg limit | 26 SDR per kg since 28 December 2024 |
| CMR | Convention governing international road carriage in Europe and neighbouring states | Assumed to apply to domestic road carriage | 8.33 SDR per kg |
| Landed cost | Total cost of a unit delivered to a named place, including freight, duty, fees, and financing | Reported excluding duty or excluding working capital | Currency per unit |
| Incoterms 2030 | The next revision of the rules, in preparation with a target launch in late 2029 | Assumed to be published and in force | Not yet issued |

Five of these deserve expansion, because they cause the most damage when confused.

**Risk transfer versus cost transfer.** Under the C rules and only the C rules, these separate.
The seller contracts and pays carriage to the destination, but risk passes at origin when the
goods are handed to the carrier (CPT, CIP) or loaded on board (CFR, CIF). A seller who thinks
"we pay the freight, so we carry the risk" has misread four of the eleven rules.

**Demurrage versus detention.** Demurrage is charged when the *cargo or container* overstays
inside the terminal. Detention is charged when the *container* overstays outside the terminal —
typically after the buyer has taken delivery and has not yet returned the empty. A shipment can
incur demurrage because the buyer's customs broker was slow, then incur detention because the
same delay pushed the empty return past free time. These are two separate invoices from two
separate clocks, and they frequently appear as a single line called "port charges" in a
dashboard.

**Chargeable weight versus actual weight.** Every mode that prices by weight also prices by
volume. Air uses a divisor, commonly 6,000 cubic centimetres per kilogram under IATA rules,
though integrators frequently use 5,000. LCL ocean uses a one-to-one equivalence between a cubic
metre and a metric tonne. Road LTL uses a mix of weight, pallet positions, and linear metre
rules. A consignment's rated weight can be two to three times its actual weight, and a
comparison of rates per kilogram across modes is meaningless unless both sides use the same
basis.

**Duty versus VAT.** Duty is a cost. Import VAT is normally recoverable by a VAT-registered
trader and is therefore a cash-flow item, not a cost — but it is a large cash-flow item,
calculated on customs value plus duty, and it can sit for a month or more before recovery. A
landed-cost model that treats VAT as a cost overstates unit cost by roughly the VAT rate. One
that ignores it entirely understates the working capital required by the same amount.

**Landed cost versus invoice price.** Price is the number on the commercial invoice. Landed cost
is what the unit costs once it is in your building and paid for. The gap is a function of the
Incoterm, the mode, the duty rate, the payment terms, and the reliability of the import process.
Section 6 builds one in full. Any model that optimises on price optimises the wrong number, and
Topic 06 already established why the error is systematic rather than random.

## 5. The mechanics, step by step

This section walks a shipment from the mode decision to the empty container return. The purpose is
to make each cost and each liability visible at the point it is incurred, so a landed-cost
schedule can be assembled from events rather than guessed from a rate card.

### Phase A — Choosing the mode

**Step 1. Characterise the cargo.** Two derived numbers drive everything: **density** (kg/m3)
determines whether the consignment is rated on weight or volume, and **value density** (currency
per kg) determines whether the time value can justify a faster mode. These are different tests and
both must be run.

| Cargo type | Density (kg/m3) | Value density | Mode that usually wins |
|---|---:|---|---|
| Iron ore, grain, aggregates | 800–2,500 | Very low | Bulk ocean, barge, rail |
| Consumer durables, furniture | 120–250 | Low | FCL ocean; rail for long inland legs |
| Apparel, footwear | 90–180 | Medium | FCL ocean; air for reorders |
| Consumer electronics | 150–350 | High | Ocean for planned volume, air for launch and shortage |
| Pharmaceuticals, biologics | 60–200 | Very high | Air, temperature-controlled; ocean for stable SKUs |
| Semiconductors, precision instruments | 200–600 | Extremely high | Air, expedited |
| Documents, samples, spare parts | Any | High per unit | Parcel, express |

**Step 2. Price the time.** Carrying cost per day is `value x cost of capital / 365`, plus
obsolescence where material. Compare the time saved by a faster mode against the extra freight.
The key branch is whether the relevant cost is capital or margin at risk: for a product with a
short selling window, a late arrival costs lost margin, not interest, and the two answers differ
by an order of magnitude.

**Step 3. Test the service requirement.** Some cargo cannot tolerate a 40-day transit: live
goods, short shelf life, fixed launch dates, and anything feeding a line with less than the
transit time in buffer. The branch that matters is whether speed or reliability is the
constraint. A mode that is fast on average but highly variable needs more safety stock than a
slower, steadier one, and that safety stock can cost more than the freight difference.

**Step 4. Book.** Rates are spot (published, volatile) or contract (fixed against a volume
commitment). The choice between them is a risk decision. Record the rate basis, the validity
window, and any surcharge structure, because a base rate plus surcharges is not comparable to an
all-in rate.

### Phase B — Choosing the term

**Step 5. Run the capability test.** The term follows from what each party can actually do, not
from preference.

| Capability | If the seller has it | If the seller lacks it |
|---|---|---|
| Export clearance at origin | Any rule is usable | EXW becomes unusable cross-border; every other rule requires the seller to clear export |
| Import clearance at destination | DDP becomes possible | Use DAP or DPU and let the buyer clear |
| Destination tax registration | DDP is workable | DDP creates a registration obligation or non-recoverable VAT |
| Competitive contract carriage | C and D rules are attractive | F rules let the buyer use its own contracts |

**Step 6. Choose the rule and the named place.** `FCA` alone is not a term; `FCA Yantian
Container Terminal Incoterms 2020` is. The branch that matters is whether the named place sits
inside or outside the terminal: FCA at the seller's premises includes loading onto the collecting
vehicle, FCA at the terminal does not.

**Step 7. Record it in three places.** The contract, the commercial invoice, and the customs
declaration must agree. The common failure is an ERP term at customer level, a contract term at
shipment level, and a third term in the letter of credit, reconciled only when a claim arises.

### Phase C — The physical movement

**Step 8. Export clearance.** An export declaration is lodged referencing the HS code, value,
destination, and delivery term. Any export duty, licence, or inspection is borne by the party
responsible for clearance — the buyer under EXW, which is that rule's central defect, since a
buyer with no presence in the origin country cannot easily lodge the declaration.

**Step 9. Stuffing and verification.** Cargo is loaded at the seller's premises, or at a container
freight station (CFS) for LCL. The verified gross mass (VGM) must be declared before loading under
regulation VI/2 of the International Convention for the Safety of Life at Sea (SOLAS), in force
since 1 July 2016, with no exception. The standard maximum rating for most container types is
30,480 kg gross, with tare of roughly 2,200 kg for a 20 ft general-purpose box rising to 5,300 kg
for a large flatrack, so payload is roughly 25 to 28 tonnes. The branch is weight versus volume: a
40 ft box holds about 67 m3 of practical stow and about 26 tonnes, so cargo denser than roughly
390 kg/m3 hits the weight limit first, and lighter cargo pays for space it cannot use.

**Step 10. Main carriage.** Ocean is the longest and least controllable block in the chain. Events
arrive from automatic identification system (AIS) data and carrier feeds. Freight is invoiced at
booking or shipment, often as a base rate plus bunker adjustment factor (BAF), currency adjustment
factor (CAF), peak season, and security surcharges.

**Step 11. Import clearance.** The importer of record or its broker lodges the declaration
referencing the HS code, the customs value, the origin, and any preference claim. Duty, import
VAT, excise, anti-dumping duty, and brokerage are assessed here. Since 1 January 2026, goods
covered by the EU Carbon Border Adjustment Mechanism (CBAM) require an authorised CBAM declarant.
The branch that matters is who the importer of record is: it is the party liable for the
declaration, and in several jurisdictions, including Germany, domestic law deems the buyer to be
the importer regardless of the contract.

**Step 12. Inland delivery, unloading, empty return.** Under DAP the seller delivers ready for
unloading; under DPU the seller unloads. Detention frequently accrues because the buyer's dock
could not take the container on the day it arrived, and stops only when the empty is returned and
accepted at a depot the carrier nominates, which may be 90 km away.

### Phase D — The documents and the money

**Step 13. Commercial invoice and customs value.** The invoice states the goods, value, Incoterm,
and named place. A missing named place makes it non-compliant for customs and for documentary
credits. The branch is whether the invoice value equals the declared customs value: under FCA the
invoice excludes freight, so freight and insurance must be added to build the CIF duty base.
Declaring the invoice value alone understates duty and is a common audit finding.

**Step 14. Transport document and insurance.** Ocean uses a bill of lading (B/L), negotiable and
a document of title when issued to order; air uses an air waybill (AWB), never a document of
title; road uses a CMR note; rail uses a CIM note under COTIF. Under CIF the seller provides
Institute Cargo Clauses (C) cover at 110% of contract value; under CIP, Clauses (A), a materially
broader cover, which was a 2020 change. Every other rule is silent on insurance, which does not
mean nobody should insure — it means the contract does not say and the risk sits with whoever
bears it. The branch is whether the cover fits the cargo: Clauses (C) is a named-perils cover
excluding theft and non-delivery, and it is the CIF default.

### Phase E — When it goes wrong

**Step 15. Loss, delay, and storage charges.** The carrier's liability is capped by convention:
ocean under Hague-Visby at 2 Special Drawing Rights (SDR) per kg or 666.67 SDR per package,
whichever is higher; air under Montreal at 26 SDR per kg since 28 December 2024; road under the
Convention on the Contract for the International Carriage of Goods by Road (CMR) at 8.33 SDR per
kg. A 500 kg pallet worth EUR 40,000 recovers about 1,000 SDR, roughly EUR 1,200, or 3% of the
loss. Delay damages under CMR are capped at the carriage charge. Demurrage and detention are
invoiced by the carrier against the party named in its tariff, and free time is set by tariff and
port, not by treaty. The practical test for disputing such a charge is whether the party billed
could have avoided it.

### 5.1 The eleven rules in one table

Seven rules apply to any mode, including multimodal. Four apply only to sea and inland waterway
carriage, because their delivery point is defined relative to a vessel.

**Any mode**

| Rule | Carriage arranged by | Risk passes | Export clearance | Import clearance | Insurance obligation |
|---|---|---|---|---|---|
| EXW | Buyer | When goods are placed at the seller's disposal at the named place, not loaded | Buyer | Buyer | None |
| FCA | Buyer, unless otherwise agreed | When goods are handed to the carrier at the named place; loaded if the place is the seller's premises | Seller | Buyer | None |
| CPT | Seller | When goods are handed to the first carrier | Seller | Buyer | None |
| CIP | Seller | When goods are handed to the first carrier | Seller | Buyer | Seller, Clauses (A), 110% of contract value |
| DAP | Seller | At destination, ready for unloading | Seller | Buyer | None |
| DPU | Seller | At destination, unloaded | Seller | Buyer | None |
| DDP | Seller | At destination, ready for unloading | Seller | Seller | None |

**Sea and inland waterway only**

| Rule | Carriage arranged by | Risk passes | Export clearance | Import clearance | Insurance obligation |
|---|---|---|---|---|---|
| FAS | Buyer | When goods are placed alongside the vessel at the named port | Seller | Buyer | None |
| FOB | Buyer | When goods are on board the vessel at the named port | Seller | Buyer | None |
| CFR | Seller | When goods are on board the vessel at the named port | Seller | Buyer | None |
| CIF | Seller | When goods are on board the vessel at the named port | Seller | Buyer | Seller, Clauses (C), 110% of contract value |

Four observations follow, and each is a source of real money.

**The C rules split cost from risk.** CPT, CIP, CFR, and CIF have the seller paying carriage to a
destination while risk transfers at the origin end. This is not careless drafting. It exists
because the seller can usually buy carriage more cheaply than the buyer can, and the buyer wanted
a price that included delivery without imposing on the seller an obligation to deliver in fact.

**Only two rules impose insurance, and they impose different covers.** CIF requires Clauses (C);
CIP requires Clauses (A). A buyer who assumes CIF gives all-risk protection is buying a cover that
excludes theft.

**DDP is the only rule where the seller clears import, and it is the one most often sold to
sellers who cannot perform it.** Section 10 examines why. **FAS and FOB require a vessel**: their
delivery point is defined by the ship, not the terminal, so for containerised cargo handed over
days before the vessel arrives that point does not exist at handover, which is why the ICC
recommends FCA for containers.

## 6. Worked example

A European lighting distributor — call it Lumina BV — sources a single product from a Chinese
manufacturer and has to choose between two offers. All figures are illustrative but internally
consistent, and every assumption is stated so that the arithmetic can be reproduced.

### 6.1 The shipment

| Attribute | Value |
|---|---|
| Product | LED panel luminaires, 40 W, 600 x 600 mm |
| HS classification | 9405 42, lamps and lighting fittings |
| Quantity | 14,400 pieces, packed 12 per carton |
| Cartons | 1,200 |
| Carton dimensions | 60 x 40 x 15 cm, therefore 0.036 m3 each |
| Total volume | 43.20 m3 |
| Gross weight | 9.5 kg per carton, therefore 11,400 kg |
| Density | 11,400 / 43.20 = 264 kg/m3 |
| Origin | Foshan, Guangdong, China |
| Port of loading | Yantian, Shenzhen |
| Port of discharge | Rotterdam, Netherlands |
| Final place | Venlo, Netherlands |
| Sea distance | approximately 19,500 km by the Suez route |
| Exchange rate | EUR 1.00 = USD 1.08 (illustrative, held constant) |
| Buyer's cost of capital | 6.0% per annum |
| EU duty rate on HS 9405 42 | 4.7% of the customs value, third-country rate |
| EU import VAT | 21%, recoverable, treated as cash flow and excluded from cost |

Two container-utilisation facts follow immediately, and both matter:

```text
volume utilisation  = 43.20 / 67.0  = 64.5%   (40 ft high cube, practical stow)
weight utilisation  = 11,400 / 26,000 = 43.8% (payload after tare)
```

The container is volume-limited at about two-thirds full. Luminaires are bulky and light, so the
buyer pays for a whole 40 ft box and uses 43.2 of roughly 67 usable cubic metres. This is the
first appearance of the volumetric trap: the freight is charged per container, so the effective
cost per cubic metre of cargo is 55% higher than the headline rate per container implies.

### 6.2 The two offers

| | Offer A | Offer B |
|---|---|---|
| Term | FCA Yantian Container Terminal, Incoterms 2020 | DAP Venlo, Incoterms 2020 |
| Unit price | EUR 41.20 | EUR 41.55 |
| Invoice total | EUR 593,280 | EUR 598,320 |
| Payment terms | 30 days from bill of lading date | 30 days from delivery at Venlo |
| Difference on invoice | — | +EUR 5,040, or +0.85% |

On the invoice alone, Offer A is 0.85% cheaper. That comparison is the one that appears in the
sourcing spreadsheet, and it is the wrong one.

### 6.3 The full landed-cost schedule

Every component, its owner under each term, and its treatment.

| Component | Basis | Offer A: FCA | Offer B: DAP | Owner A | Owner B |
|---|---|---:|---:|---|---|
| Goods | 14,400 units | 593,280 | 598,320 | Buyer | Buyer |
| Origin terminal handling and documentation | Per container | 285 | in price | Buyer | Seller |
| Ocean freight, 1 x 40 ft HC | Spot, USD 5,600 | 5,185 | in price | Buyer | Seller |
| Marine cargo insurance | 0.10% of 110% of CIF | 658 | in price | Buyer | Seller |
| Import customs brokerage | Per entry | 180 | 180 | Buyer | Buyer |
| Destination terminal handling | Per container | 240 | in price | Buyer | Seller |
| Inland haulage Rotterdam to Venlo | Per container | 480 | in price | Buyer | Seller |
| Demurrage | 8 days at EUR 210 | 1,680 | 0 | Buyer | Seller |
| Detention | 6 days at EUR 145 | 870 | 0 | Buyer | Seller |
| Duty | 4.7% of customs value | 28,172 | 28,098 | Buyer | Buyer |
| Unloading at Venlo | Per container | 0 | 180 | Buyer | Buyer |
| Working capital | 6.0% per annum | 8,149 | 4,630 | Buyer | Buyer |
| **Total landed cost** | | **639,179** | **631,408** | | |
| **Landed cost per unit** | / 14,400 | **44.39** | **43.85** | | |

The DAP invoice is 0.85% higher. The DAP landed cost is 1.23% lower. The swing is EUR 12,811 on a
shipment of roughly EUR 600,000, which for a distributor with a 3% net margin is more than two
months of profit on the consignment.

### 6.4 The arithmetic, line by line

**Offer A — FCA Yantian.**

```text
goods                          = 14,400 x 41.20            = 593,280
origin handling                                            =     285
ocean freight                  = 5,600 USD / 1.08          =   5,185
insurance                      = 0.0010 x 1.10 x 599,408   =     658
brokerage                                                  =     180
destination THC                                            =     240
inland Rotterdam - Venlo                                   =     480
demurrage                      = 8 x 210                   =   1,680
detention                      = 6 x 145                   =     870
                               --------------------------------
logistics subtotal (excl. duty and capital)                =   9,578

customs value (CIF) = 593,280 + 5,185 + 285 + 658          = 599,408
duty                = 0.047 x 599,408                      =  28,172
duty base for VAT   = 599,408 + 28,172                     = 627,580
working capital     = 627,580 x 0.06 x 79 / 365            =   8,149

total               = 593,280 + 9,578 + 28,172 + 8,149     = 639,179
per unit            = 639,179 / 14,400                     =   44.39
```

The 79 financing days need justification. The supplier invoices on the bill of lading date, day 0.
Lumina pays on day 30. The goods arrive at Venlo on day 34. Lumina holds them for 30 days and
sells on day 64, then collects from its own customers on 45-day terms, receiving cash on day 109.
Cash is therefore committed from day 30 to day 109, which is 79 days.

**Offer B — DAP Venlo.**

```text
goods (DAP price)                                          = 598,320
brokerage                                                  =     180
duty                                                       =  28,098
unloading                                                  =     180
working capital                                            =   4,630
                               --------------------------------
total                                                      = 631,408

customs value = 598,320 - 480 (post-importation inland)    = 597,840
duty          = 0.047 x 597,840                            =  28,098
duty base     = 597,840 + 28,098                           = 625,938
working capital = 625,938 x 0.06 x 45 / 365                =   4,630
per unit      = 631,408 / 14,400                           =   43.85
```

The 45 financing days are shorter because the seller invoices on delivery, day 34, not on
shipment. Lumina pays on day 64 and collects on day 109, so cash is committed for 45 days.

Two technical points in this arithmetic are worth pausing on, because both are routinely got
wrong.

> **Note:** The EU customs value is a CIF value at the place of introduction into the customs
> territory. Under FCA the invoice excludes freight and insurance, so both must be added to the
> declared value. Under DAP the price already includes carriage to Venlo, so the post-importation
> inland leg is deducted and the result is the CIF value. The two duty bases land within 0.3% of
> each other even though the invoice prices differ by 0.85%. Duty is nearly a wash here, which
> contradicts the common assumption that the term drives the duty.

> **Note:** Import VAT of 21% on 627,580 is EUR 131,792. It is recoverable, so it is not a cost,
> but it is EUR 132,000 of cash out for one to two months, and a model that omits it understates
> the working capital requirement by an order of magnitude relative to the freight.

### 6.5 Why the cheaper invoice is the more expensive landed cost

The bridge from the invoice comparison to the landed-cost comparison:

| Driver | Amount (EUR) | Favours |
|---|---:|---|
| Invoice premium paid under DAP | 5,040 | FCA |
| Ocean freight: spot at USD 5,600 against the seller's contract rate of USD 3,400 | 2,037 | DAP |
| Insurance: buyer's retail premium against the seller's master policy | 18 | DAP |
| Demurrage, 8 days | 1,680 | DAP |
| Detention, 6 days | 870 | DAP |
| Unloading at Venlo, borne by the buyer under DAP | -180 | FCA |
| Seller's margin embedded in the DAP logistics content | -247 | FCA |
| Duty: higher dutiable value under DAP | -74 | FCA |
| Working capital: 79 days against 45 days | 3,519 | DAP |
| **Net difference in favour of DAP** | **7,771** | |

Reading the bridge in order of size: **working capital at EUR 3,519** is the largest driver and
the one nobody models, because the whole difference comes from the payment trigger — FCA starts
the clock at the bill of lading, DAP at delivery, and that 34-day gap is worth more than the
entire ocean freight. **Demurrage and detention at EUR 2,550** is second and is a process failure,
not a price difference: the buyer's broker queried the declared value because the FCA invoice had
not been grossed up to CIF, the container sat eight days, and the same error produced six days of
detention on the empty. **Freight at EUR 2,037** is a pure scale effect, since the seller moves 40
containers a month against the buyer's two. **Duty, negative EUR 74**, favours FCA, because the
DAP price embeds a logistics margin that becomes part of the customs value.

The general lesson is not "use DAP". It is that **the term changes which party's cost structure
applies to five of the nine cost drivers, and the biggest of those drivers is the timing of cash,
not the price of freight.**

### 6.6 The reversal condition

Suppose the FCA supplier offers 90 days from bill of lading instead of 30. Cash leaves on day
90, the goods arrived on day 34, and Lumina sells on day 64. Cash is committed only from day 90
to day 109, which is 19 days.

```text
working capital = 627,580 x 0.06 x 19 / 365 = 1,959
Offer A total   = 593,280 + 9,578 + 28,172 + 1,959 = 632,989
Offer B total   = 631,408
FCA still dearer by 1,581
```

Now add the buyer's own contract freight at the seller's rate and remove the demurrage event,
which were both buyer-side execution problems rather than term effects:

```text
ocean freight    = 3,400 USD / 1.08        = 3,148
insurance        = 0.0010 x 1.10 x 596,713 =   656
logistics (excl. duty, capital)            = 4,989
customs value    = 596,713 + 656           = 597,369
duty             = 0.047 x 597,369         = 28,076
working capital  = (597,369 + 28,076) x 0.06 x 19 / 365 = 1,953
Offer A total    = 593,280 + 4,989 + 28,076 + 1,953 = 628,298
```

FCA now wins by EUR 3,110. Both conditions concern the buyer's commercial position, not the
freight rate: extended supplier terms and enough volume to earn a contract rate. The honest
conclusion is that a term has no intrinsic cost; it allocates costs between two parties whose
cost structures differ.

### 6.7 The same cargo by air, and why it does not fly

Chargeable weight is the greater of actual and volumetric weight: 43,200,000 cm3 / 6,000 = 7,200 kg
volumetric against 11,400 kg actual, so the cargo is weight-rated at 11,400 kg. At USD 4.50 per kg
that is USD 51,300, or EUR 47,500, against EUR 3,148 for ocean. Air saves 30 days, worth EUR 3,083
of capital, so it costs EUR 41,269 net more and is justified only by a launch, a stock-out, or a
line stoppage.

Note the asymmetry: 34 days of capital at 6% on EUR 625,445 is EUR 3,494, which exceeds the EUR
3,148 ocean freight. For this cargo the time at sea costs more than the voyage itself. That is why
high-value-density goods fly despite air being roughly sixteen times the ocean rate per kilogram,
and why a per-kilogram comparison is not a mode decision.

## 7. Code, query, or calculation

One query, because the arithmetic in section 6 is the deliverable. The query aggregates a
component table into a landed cost and a per-unit figure. The interesting part is not the
aggregation; it is the `cif_base` rule, which is where most implementations are wrong.

```sql
-- One row per cost component per shipment. `borne_by` is resolved from the Incoterm
-- before this query runs; that resolution is a contract problem, not a SQL problem.
with comp as (
    select shipment_id, component, borne_by, amount_eur
    from landed_cost.component
),
agg as (
    select
        shipment_id,
        sum(case when borne_by = 'buyer' then amount_eur else 0 end) as buyer_cost,
        sum(case when component in ('goods', 'freight', 'origin_charges', 'insurance')
                 then amount_eur else 0 end)                         as cif_base
    from comp
    group by shipment_id
)
select
    a.shipment_id,
    s.incoterm,
    round(a.buyer_cost, 2)                                    as landed_cost_eur,
    round(a.cif_base * s.duty_rate, 2)                        as duty_eur,
    round((a.buyer_cost + a.cif_base * s.duty_rate)
          / nullif(s.quantity, 0), 4)                         as landed_per_unit
from agg a
join landed_cost.shipment s using (shipment_id)
order by a.shipment_id;
```

Two checks before trusting any implementation of this:

- Run it for a DAP shipment and confirm that the inland leg after import is excluded from
  `cif_base`. If the duty is computed on the full DAP price, the company is overpaying duty on
  its own domestic haulage.
- Compare `cif_base` to the commercial invoice total. If they are equal on an FCA shipment, the
  freight and insurance have not been added and the duty is understated.

## 8. Data, standards, and regulatory context

Almost every number in a landed-cost model is defined by an instrument, and knowing which one
tells you what the data will look like before you go looking for it.

### 8.1 The contract and carriage framework

| Instrument | Governs | What it constrains | Grade |
|---|---|---|---|
| Incoterms 2020 (ICC) | Delivery, cost, and risk between buyer and seller | Which party performs each function and where risk passes | B |
| UCP 600 (ICC) | Documentary credits | Which documents, and in what form, the bank will accept | B |
| Hague-Visby Rules | Ocean carrier liability | 666.67 SDR per package or 2 SDR per kg, whichever is higher | B |
| Hamburg Rules | Ocean carrier liability in states that adopted them | 835 SDR per package or 2.5 SDR per kg | B |
| Rotterdam Rules | Multimodal carriage including a sea leg | Not in force among major trading states as of 2026 | B |
| Montreal Convention 1999 | Air carrier liability | 26 SDR per kg, revised from 22 SDR effective 28 December 2024 | B |
| CMR Convention 1956 | International road carriage | 8.33 SDR per kg; delay damages capped at the carriage charge | B |
| COTIF / CIM | International rail carriage | 17 SDR per kg | B |

The practical consequence for analytics is that **carrier liability data is not a measure of
loss.** A claims table showing EUR 1.2 million recovered against EUR 40 million of damaged cargo
does not mean the carrier performed well or badly. It means the convention caps applied. Any
model that treats recovered claims as a proxy for loss frequency will be fitting the liability
regime, not the operation.

> **Note:** The Montreal limit was revised from 22 to 26 SDR/kg on 28 December 2024. Many
> templates still quote 22, which understates recovery by roughly 18%.

### 8.2 Customs and trade regulation

| Requirement | Jurisdiction | Status as of 2026 | Grade |
|---|---|---|---|
| Customs valuation: transaction value, adjusted to CIF | WTO Customs Valuation Agreement, applied nationally | In force; the EU and most markets use a CIF duty base, the United States uses FOB | B |
| HS classification | World Customs Organization, six digits; national extensions beyond | In force; national tariff lines differ beyond HS6 | B |
| Preferential origin rules | Each trade agreement | In force; qualification requires a specific origin rule per product | B |
| CBAM definitive regime | European Union | Applies from 1 January 2026 under Regulation (EU) 2023/956, as amended by Regulation (EU) 2025/2083 | B |
| Union Customs Code declarant establishment | European Union | The declarant must be established in the EU; this is the constraint that breaks DDP | B |
| SOLAS VI/2 verified gross mass | Global, via IMO | In force since 1 July 2016 | B |
| EU low-value consignment relief | European Union | The EUR 150 de minimis exemption was reported removed from 1 July 2026 | E |

On CBAM specifically, the mechanics matter and are frequently misdescribed as a tariff. The
regime applies from 1 January 2026. Certificates are not sold during 2026: sales start on 1
February 2027 through a central platform, declarants must hold certificates covering at least 50%
of year-to-date embedded emissions at each quarterly checkpoint from 2027, and the first
declaration and surrender is due by 30 September 2027 for 2026 imports. There is a single
mass-based de minimis of 50 tonnes net mass per importer per calendar year, cumulative across
steel, aluminium, fertilisers, and cement, and it does not apply to electricity or hydrogen.
Exceeding it mid-year triggers full obligations for the whole year.

> **Note:** CBAM is priced per tonne of embedded carbon dioxide equivalent, not per unit of value,
> and the price tracks the Emissions Trading System auction average. Treating it as an ad valorem
> duty misallocates it across a product mix, because a heavy low-value item carries a far higher
> CBAM cost per euro of goods value than a light high-value one.

### 8.3 Transport documentation standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| IATA TACT Rules and Resolution 502 | Air cargo rating | Volumetric divisor of 6,000 cm3 per kg for general cargo | B |
| Integrator dimensional weight rules | Express and parcel | Divisors of 5,000 or 5,000-plus, varying by carrier and service | C |
| Incoterms 2020 article A9/B9 | Cost allocation | The consolidated list of costs each party bears under each rule | B |
| Institute Cargo Clauses (A), (B), (C) | Marine cargo insurance | The scope of cover; (A) is all risks, (C) is named perils | B |
| FMC 46 CFR 545.5 and 46 CFR Part 541 | US demurrage and detention | Reasonableness, invoice content, and dispute timelines | B |
| ISPS Code | Port and vessel security | Security surcharges and facility access | B |

### 8.4 Where the data lives, and why it does not join

The components of a landed cost are generated by at least five systems that have no common key:

| Component | Typical system of record | Join key available |
|---|---|---|
| Goods value | ERP purchase order | PO number |
| Freight and surcharges | Forwarder invoice, often PDF | House bill of lading number, free text |
| Duty and VAT | Customs declaration, broker system | Entry number, sometimes absent from the ERP |
| Demurrage and detention | Carrier invoice | Container number |
| Inland haulage | 3PL invoice | Delivery note, free text |
| Insurance | Broker policy and certificate | Certificate number |
| Working capital | Treasury model | None; derived |

The only key present in all of them is usually the container number or the shipment identifier,
and each party frequently re-creates the shipment identifier. Landed-cost analytics is therefore
more often a data-matching problem than a modelling problem, and the first deliverable of any such
project is a shipment identity resolution rule rather than a cost model.

## 9. Industry practice and case evidence

### 9.1 Mode economics: the measured numbers

The most useful published comparison of unit freight cost by mode is Krammer and Schäfer (2025)
in *npj Sustainable Mobility and Transport*, drawing on 1990–2019 data across ten countries plus
EU and global aggregates. Mean unit costs in 2021 US dollars per tonne-kilometre:

| Mode | Mean USD per tonne-km | 10th–90th percentile | Energy intensity (MJ per tonne-km) | Mean load |
|---|---:|---|---:|---:|
| Ocean vessel | 0.004 | 0.002–0.007 | 0.136 | 26,600 t |
| Pipeline (oil) | 0.021 | 0.008–0.053 | 0.408 | — |
| Rail | 0.026 | 0.017–0.033 | 0.241 | 979 t |
| Truck | 0.114 | 0.058–0.170 | 2.19 | 10.9 t |
| Freighter aircraft | 0.844 | 0.228–1.74 | 16.4 | 33.1 t |

Grade A for the data. The authors are explicit that the averages are meaningless on their own,
because unit cost depends heavily on average load: the elasticity of unit cost to load is about
-0.95 for rail, -0.57 to -0.75 for sea, -0.63 to -0.88 for road, and -0.91 for air. Each mode
occupies a load window in which it is competitive, and outside that window the ranking can
invert.

The ocean figure is the one most often abused. At USD 0.004 per tonne-km, a 1,000 kg pallet moved
19,500 km should cost USD 78. It does not, because that rate assumes a vessel carrying 26,600
tonnes. A single pallet moving LCL pays for consolidation, a container freight station at each
end, a documentation fee, and a share of a container that is priced as a container. The
per-tonne-km average is a property of the mode at scale, not a price available to a shipper.

Dutch national benchmarks (Panteia/KiM, 2021) show the same ordering with wider ranges: inland
waterway EUR 0.010–0.092 per tonne-km, rail EUR 0.011–0.047, road EUR 0.099–0.487, maritime EUR
0.0014–0.0051, air EUR 0.19. The ranges are not directly comparable to the means above because
they differ in year, currency, and whether handling is included. Grade B.

### 9.2 The ocean market in 2026

| Observation | Value | Date | Grade |
|---|---|---|---|
| Asia–North Europe spot | USD 3,698 per FEU, up 6.8% week on week | 26 October 2026 | D |
| Asia–Mediterranean spot | USD 4,442 per FEU | 26 October 2026 | D |
| Asia–US West Coast spot | USD 8,311 per FEU | 26 October 2026 | D |
| Asia–US East Coast spot | USD 11,525 per FEU | 26 October 2026 | D |
| Bab el-Mandeb transit capacity | 23% of the August 2023 level, rising to 48% by mid-October | September–October 2026 | D |
| Global schedule reliability | 56.4% in the first half of 2026 | September 2026 | D |
| Orderbook as share of active fleet | 41.7%, a record | September 2026 | D |

The picture is a market in transition and the sources disagree about where it lands. Returning
Suez routings add effective capacity and a record orderbook points to medium-term oversupply;
against that, Asia–Europe demand grew 12.2% in the first seven months of 2026 on one measurement,
port congestion reached a record 4.31 million TEU, and carriers are blanking sailings to defend
rates. One carrier alliance expects nearly all services back to Suez routings in the first quarter
of 2027, which is a company claim rather than an observation.

> **Note:** Every figure in this table is vendor or press data (Grade D or E) and several carry
> dates within weeks of writing. Rates move weekly and reliability moves monthly. Use the shape of
> the market, not the number: a model with a hard-coded rate from this table will be wrong within
> a quarter.

### 9.3 Air cargo in 2026

IATA reported 2025 as a record year: demand in cargo tonne-kilometres (CTK) up 3.4%, capacity up
3.7%, and a load factor of 45.7%, with yields down 1.5% but still 37.2% above 2019. Growth shifted
away from Asia–North America, which fell 0.8%, toward Europe–Asia at +10.3% and within Asia at
+10.0%. Grade C, since IATA is an industry association reporting its members' data.

For 2026 the forecast moved twice. The December 2025 outlook projected 2.6% CTK growth; in June
2026 IATA cut the volume forecast to +0.2%, to 71.7 million tonnes, while still projecting cargo
revenue up 7.2% to USD 162 billion on higher yields. By August 2026 demand had rebounded to +4.4%
year on year with load factor at 46.0%. Grade C for the statistics, Grade D for the forecast,
given the revision history.

The structural driver in both directions is e-commerce. It is also the segment most exposed to
policy: the removal of low-value consignment relief in the United States and, as reported, in the
EU from 1 July 2026 changes the economics of cross-border parcel flows directly, because duty and
VAT administration is levied per consignment rather than per container. That report is Grade E and
should be verified against the official text before it is used in a model.

### 9.4 Road and parcel economics

Road is the only mode with a continuous cost curve from a single envelope to a full truckload, and
the curve is steep at the bottom. A parcel carries dimensional weight, a residential surcharge, a
fuel surcharge indexed to published diesel prices, and often a peak surcharge. An LTL consignment
carries a rate base, a discount, a minimum charge, and accessorials for liftgate and appointment.
An FTL movement is priced per kilometre plus fuel and waiting time. The consequence is that at
small shipment sizes road cost is a function of shipment size, not distance, because the fixed
components dominate.

### 9.5 Demurrage and detention: the documented scale

The FMC reports approximately USD 15.4 billion in demurrage and detention collected between April
2020 and March 2025 across nine major carriers, though billings and collections have fallen
substantially from the 2022 peak. Grade B, since the FMC is a regulator collecting the data.

Two things follow. First, these charges are large enough to be a material landed-cost line and are
almost always treated as an exception rather than a cost category. Second, the regulatory trend is
toward shifting the burden of proof onto the carrier: a 2026 FMC rule effective 1 September 2026
requires the carrier to prove the reasonableness of a charge under the incentive principle in a
charge complaint. A company that disputes systematically recovers more than one that pays.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| FOB used for containerised cargo | Bills of lading show an on-board date days after the container entered the terminal | The delivery point is defined by the vessel, but the goods were handed over at the gate | Disputed cost of terminal handling; risk gap between gate-in and loading | Use FCA at the container terminal, or FCA with the article B6 on-board option |
| DDP sold by a seller with no import presence | Seller asks the buyer's broker to "clear it in your name" | The declarant must be established in the destination customs territory | Goods held, penalties, non-recoverable VAT, or an unusable sale | Use DAP and let the buyer clear, or establish a local entity |
| Invoice price compared across different terms | Sourcing spreadsheet has one price column and no term column | A DAP price contains freight an FCA price does not | Wrong supplier selected, systematically | Enforce a term field at shipment level and normalise to one basis before comparison |
| Duty declared on invoice value under FCA | Declared value equals the invoice to the cent | The customs value must be grossed up to CIF | Underpaid duty, penalties on audit, understated landed cost | Gross up to CIF in the declaration and in the model |
| DAP duty computed on the full delivered price | Duty base equals the DAP invoice | The post-importation inland leg is not dutiable | Overpaid duty on domestic haulage | Deduct post-importation transport from the customs value |
| Per-kilogram mode comparison | The mode analysis is one column of USD/kg | Chargeable weight, minimum shipment size, and time value are all ignored | Air rejected on a route where it is correct, or accepted where it is not | Compare fully loaded cost per shipment including capital and service |
| Volumetric trap on light goods | Container loaded to 60% of volume and 40% of payload | Freight is charged per container, not per tonne | Effective cost per unit 50% above the headline | Model cost per m3 and per unit alongside cost per kg |
| Demurrage treated as an exception | A single "port charges" line with no owner | Two clocks, two invoices, no attribution | Cost never allocated, so never managed | Split demurrage and detention, attribute by cause, accrue monthly |
| Insurance assumed to be all-risk under CIF | Policy cites Institute Cargo Clauses (C) | CIF mandates the narrowest cover | Theft and non-delivery uninsured | Buy difference-in-conditions cover or negotiate CIP |
| Liability cap mistaken for recovery | Claims recovery is 3% of loss value | Convention limits, not carrier performance | Loss budgeting understated by an order of magnitude | Model uninsured loss explicitly as a cost of the lane |
| Incoterm treated as title or payment term | Contract has one term and no separate title or payment clause | Three clocks, one instrument assumed | Goods owned by one party and at risk with another; disputes in insolvency | Document risk, title, and payment separately in every contract |

Two deserve a note. **FOB for containers** is the most common term error in world trade, and the
ICC has published guidance on it rather than leaving it to inference: a 2023 ICC Commission survey
of container ports, published as guidance in January 2024, documented frequent mismatch between
chosen terms and port reality, with an on-board delivery point that does not exist for a container
handed over days before loading, producing double-charged terminal handling and disputed cost
allocation. The ICC recommends FCA at the departure terminal and CPT at the arrival terminal,
describing FCA, CPT, and CIP as the updated maritime rules for containerised trade.

**DDP by a non-established seller** is the most expensive term error, because it stops the goods
rather than merely mispricing them. The Union Customs Code requires the declarant to be
established in the EU. A non-EU seller therefore needs an indirect representative, who becomes
jointly liable for duties and subject to audit, which is why such representatives are scarce and
expensive. Where the seller instead clears in the buyer's name without authority, the agent can
become the debtor for duties and VAT with no route to recovery. The usual mitigations are DAP with
the buyer as importer, an EU-established selling entity, or a local warehouse.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The Incoterm recorded in the system matches the contract | Cost allocation and inventory ownership depend on it | Side agreements, framework contracts, letter-of-credit terms | Sample 30 shipments and read the underlying contract |
| One shipment has one term | Simplifies allocation | Split shipments, partial deliveries, mixed terms on one PO | Check for multiple terms per shipment identifier |
| Customs value is transaction value plus freight and insurance | The duty base must be constructed | Related-party pricing, assists, royalties, transfer pricing adjustments | Reconcile declared value to invoice plus freight plus insurance |
| The HS code is stable across the shipment | Duty rate and CBAM scope depend on it | Reclassification, national tariff changes, binding rulings | Compare declared codes across shipments of the same SKU |
| Freight rates are known at shipment time | Cost must be accrued before the invoice arrives | Spot bookings, surcharges invoiced later, rate validity windows | Measure the lag between shipment and freight invoice |
| Demurrage and detention are attributable to a shipment | Otherwise the cost cannot be allocated | Consolidated carrier invoices covering many containers | Check invoice granularity; require per-container billing |
| Transit time is a property of the lane | Planning and capital cost depend on it | Seasonality, port congestion, blank sailings, route changes | Plot transit time distribution by month, not just the mean |
| Carrier liability is small relative to cargo value | Justifies treating insurance as a cost | High-value, low-weight cargo where the per-kg cap is trivial | Compute the cap as a percentage of cargo value per lane |
| VAT is fully recoverable | Justifies excluding it from cost | Partial exemption, non-established trader, blocked input tax | Confirm the entity's VAT recovery position per jurisdiction |
| The buyer's cost of capital is a single rate | Needed to price transit time | Different product lines, different funding, different currencies | Segment where the transit time is material |

Three limits are worth stating plainly. **The rules are a default contract, not law**: Incoterms
bind because the parties incorporate them, and a court applies the sale contract, the applicable
convention, and mandatory local law first, so an analyst reading only the term will sometimes be
wrong about who paid what. **Landed cost is a snapshot of a price, not a measure of performance**:
it contains no counterfactual, so a lane with high demurrage may be badly run or may pass through a
port where every competitor pays the same. **The market data here has a shelf life of weeks**, and
a model that hard-codes it is a model of October 2026 rather than a model of the lane.

## 12. Exercises

Eight exercises: four calculations, two designs, two critiques. Section 13 gives the arithmetic
for the first four and a rubric for the rest.

### Exercise 1 — Chargeable weight and container utilisation (calculation, 30 minutes)

A consignment of 480 cartons, each 55 x 45 x 40 cm and 14 kg gross, moves from Ningbo to Hamburg.

**Deliverable:** air chargeable weight, ocean LCL revenue tons, and utilisation of a 40 ft high
cube at 67 m3 practical stow and 26,000 kg payload. State which rating basis applies and why.
**Acceptance criteria:** both the volumetric and actual figures shown; the correct maximum chosen;
density stated in kg/m3; utilisation given as two percentages.

### Exercise 2 — Build the landed-cost schedule (calculation, 60 minutes)

A shipment moves EXW Stuttgart to a buyer's warehouse in Lyon. Goods EUR 88,000; origin haulage
EUR 400; export clearance EUR 120; road freight EUR 2,600; insurance 0.15% of 110% of CIF; duty
6.5%; brokerage EUR 160; destination terminal EUR 230; inland EUR 350; no demurrage. The buyer's
cost of capital is 6.0% and cash is committed for 60 days.

**Deliverable:** a thirteen-line schedule with a total and a cost per unit for 4,000 units.
**Acceptance criteria:** the CIF build-up shown explicitly; duty computed on the CIF value, not
the invoice; the working-capital base stated; every line labelled with its currency.

### Exercise 3 — Are the two offers comparable? (calculation, 45 minutes)

Offer A: FCA Shanghai, EUR 50.00 per unit, 10,000 units. The buyer pays freight EUR 4,000 and
other costs EUR 1,000, clears at 5% duty, and finances for 90 days at 5.0%.
Offer B: DAP Lyon, EUR 51.80 per unit, same quantity. The buyer clears at 5% duty on the DAP
price less EUR 400 of post-importation inland transport, and finances for 45 days at 5.0%.

**Deliverable:** landed cost for each, the percentage gap on invoice and on landed cost, and one
sentence on which offer to take and why.
**Acceptance criteria:** correct duty base for each; working capital computed on customs value
plus duty; the two percentages distinguished; the recommendation consistent with the arithmetic.

### Exercise 4 — The liability cap (calculation, 20 minutes)

A single pallet of 800 kg, insured value EUR 60,000, is lost. Use SDR 1 = EUR 1.22.

**Deliverable:** the carrier's maximum liability by ocean under Hague-Visby, by air under
Montreal, and by road under CMR, each as a currency amount and as a percentage of value.
**Acceptance criteria:** the higher of the two Hague-Visby alternatives used; the Montreal limit
at 26 SDR/kg, not 22; one sentence on what the results imply for insurance purchasing.

### Exercise 5 — Term selection memo (design, 45 minutes)

A UK manufacturer sells into the EU through a German distributor. The manufacturer has no EU
establishment, no EU VAT registration, and ships four containers a month. The distributor has
asked for DDP.

**Deliverable:** a one-page memo recommending a term, with the named place, the reason, and the
alternative if the distributor insists.
**Acceptance criteria:** the recommendation names the customs-establishment constraint; the
VAT consequence is stated; the fallback is a specific rule, not "negotiate"; the inventory
ownership consequence is mentioned.

### Exercise 6 — Mode selection (design, 60 minutes)

A medical device company moves 900 kg of consumables per week, value EUR 45,000 per shipment,
with a 120-day shelf life and a hospital customer that will not accept more than 10 days of
lead time. Ocean takes 32 days door to door and costs EUR 2,900 per shipment. Air takes 4 days
and costs EUR 9,800. Cost of capital is 7%.

**Deliverable:** a mode recommendation with the full cost comparison including capital, shelf
life, and service, plus the volume at which the answer changes.
**Acceptance criteria:** capital cost computed on the shipment value for the days saved; shelf
life treated as a constraint, not a cost; the break-even expressed as a shipment value or a
required service level.

### Exercise 7 — Diagnose the lane (critique, 45 minutes)

A lane reports landed cost 18% above budget for two consecutive quarters. Freight rates are flat.
Demurrage is 4% of landed cost. The declared customs value equals the FCA invoice value to the
cent. Duty is 6.5%.

**Deliverable:** a diagnosis naming at least three mechanisms, ranked by likely contribution,
with the measurement that would confirm each.
**Acceptance criteria:** at least one mechanism is a duty-base error; at least one is a
data-attribution error rather than a real cost increase; each is paired with a specific query or
reconciliation.

### Exercise 8 — Shipment identity resolution (design, 45 minutes)

You must join freight invoices, customs entries, carrier demurrage invoices, and 3PL delivery
notes for 20,000 shipments a year. No common key exists.

**Deliverable:** a matching design with candidate keys, fallback rules, a tolerance for fuzzy
matching, and an expected match rate.
**Acceptance criteria:** at least three key strategies in priority order; a stated handling rule
for one-to-many matches such as a consolidated carrier invoice; a measurable acceptance
threshold; the failure consequence of a false match stated.

## 13. Self-check answers

**Exercise 1.**

```text
volume per carton   = 55 x 45 x 40 = 99,000 cm3 = 0.099 m3
total volume        = 480 x 0.099 = 47.52 m3
total weight        = 480 x 14 = 6,720 kg
density             = 6,720 / 47.52 = 141.4 kg/m3

air volumetric      = 47,520,000 / 6,000 = 7,920 kg
air chargeable      = max(7,920, 6,720) = 7,920 kg   measure-rated
LCL revenue tons    = max(47.52, 6.72) = 47.52 RT    measure-rated

volume utilisation  = 47.52 / 67.0  = 70.9%
weight utilisation  = 6,720 / 26,000 = 25.8%
```

Both modes rate this cargo on volume: its density of 141 kg/m3 is below the air pivot of 166.7
kg/m3 and far below the ocean pivot of 1,000 kg/m3. Using actual weight would understate the air
charge by 18% and the ocean charge by a factor of seven.

**Exercise 2.**

```text
CIF base        = 88,000 + 400 + 120 + 2,600 = 91,120
insurance       = 0.0015 x 1.10 x 91,120     =    150
CIF value       = 91,270
duty            = 0.065 x 91,270             =  5,933
logistics       = 400 + 120 + 2,600 + 150 + 160 + 230 + 350 = 4,010
working capital = (91,270 + 5,933) x 0.06 x 60 / 365 = 959
total           = 88,000 + 4,010 + 5,933 + 959 = 98,902
per unit        = 98,902 / 4,000 = 24.73
```

The most common error is computing duty on 88,000, which gives 5,720 and understates the total by
EUR 213. The second most common is omitting the working capital line entirely, which understates
it by EUR 959, or about 1% of landed cost.

**Exercise 3.**

```text
Offer A
  CIF base    = 500,000 + 4,000 + 1,000 = 505,000
  duty        = 0.05 x 505,000 = 25,250
  capital     = (505,000 + 25,250) x 0.05 x 90 / 365 = 6,536
  total       = 500,000 + 5,000 + 25,250 + 6,536 = 536,786

Offer B
  customs val = 518,000 - 400 = 517,600
  duty        = 0.05 x 517,600 = 25,880
  capital     = (517,600 + 25,880) x 0.05 x 45 / 365 = 3,350
  total       = 518,000 + 25,880 + 3,350 = 547,230

invoice gap = 18,000 / 500,000 = 3.60%
landed gap  = 10,444 / 536,786 = 1.95%
```

Offer A wins. Offer B's longer inclusive price and shorter cash cycle claw back 1.65 percentage
points of the 3.60% invoice premium, so the true disadvantage is roughly half what the invoice
suggests. A buyer who rejected Offer B on the invoice number alone would have been right, but for
a reason half the size of the one they used.

**Exercise 4.**

```text
Hague-Visby  = max(666.67, 2 x 800) = 1,600 SDR = 1,952 EUR   = 3.3% of value
Montreal     = 26 x 800             = 20,800 SDR = 25,376 EUR = 42.3% of value
CMR          = 8.33 x 800           = 6,664 SDR = 8,130 EUR   = 13.6% of value
```

Ocean recovery is 3% of the loss. This is why marine cargo insurance is a required cost on ocean
lanes and why the choice between Institute Cargo Clauses (A) and (C) is not a formality.

**Exercises 5 to 8: rubrics.**

A strong answer to Exercise 5 recommends DAP, names the customs-establishment constraint under
the Union Customs Code as the reason DDP fails, states that the seller would need an EU-established
entity or an indirect representative who becomes jointly liable for duties, and notes that the
buyer becomes importer of record with its own VAT recovery. It mentions that under DAP the seller
retains ownership and risk to the distributor's door, increasing its in-transit inventory. The
fallback is DDP through an EU-established selling entity, not through the buyer's broker. A weak
answer recommends DDP because the customer asked, and mentions neither establishment nor VAT.

A strong answer to Exercise 6 rejects ocean on the shelf-life and lead-time constraints before
doing arithmetic, then shows that air saves 28 days worth EUR 242 of capital, so the decision
rests entirely on the service and shelf-life constraints. It notes that 32 days of transit against
a 10-day tolerance is infeasible at any freight rate, and that the real options are air or a
forward stocking location. A weak answer compares cost per kilogram, finds air twelve times
dearer, and recommends ocean, violating both stated constraints.

A strong answer to Exercise 7 names the duty-base error first — declared value equals the FCA
invoice, so freight and insurance are missing from the CIF base — quantifies it at roughly 4% of
landed cost where freight is around 8% of goods value, attributes the demurrage to a specific
process cause, then checks whether the variance is a data artifact such as inland haulage being
reclassified into the lane. A weak answer blames freight market conditions, which the question
states are flat.

A strong answer to Exercise 8 proposes the container number as primary key, the house bill of
lading number as fallback, and a shipment reference pattern-matched from free-text invoice fields
as a third, with a date-plus-amount tolerance for the residual. It states an explicit rule for
consolidated invoices covering many containers, an expected match rate of 90–97% with the rest
queued for manual review, and the consequence of a false match: cost attributed to the wrong lane,
corrupting every downstream lane comparison. A weak answer proposes matching on shipment ID
without acknowledging that each party re-creates it.

## 14. Interview questions

**1. "What does an Incoterm actually transfer?"**

Strong answer: cost and risk, at a named place, and nothing else — not title, not payment, not
revenue recognition. Weak answer describes it as "who pays for shipping."

**2. "Why is FOB wrong for containers?"**

Strong answer: FOB's delivery point is on board the vessel, but a container is handed to the
carrier at a terminal days before loading, so the risk point does not exist when the handover
happens; FCA at the terminal is the correct rule, and the ICC has published guidance saying so.
Weak answer says FOB is "for sea freight" and stops.

**3. "A supplier offers DDP and you have no EU establishment. What do you ask?"**

Strong answer: who is the declarant, whether the seller is established in the EU, how import VAT
will be recovered, and whether an indirect representative is being used and at what liability.
Weak answer asks only about price.

**4. "Why can a DAP price be higher and still cheaper?"**

Strong answer: the DAP price embeds freight the FCA price excludes, so the invoice comparison is
invalid; normalise both to landed cost, including duty base and the cash cycle. Weak answer says
"volume discounts."

**5. "How would you compare air and ocean for a specific product?"**

Strong answer: compute capital cost of the days saved against the freight difference, then apply
service and shelf-life constraints, then check chargeable weight. Weak answer compares USD per kg.

**6. "What is chargeable weight and why does it matter?"**

Strong answer: the greater of actual and volumetric weight, with the divisor varying by mode and
carrier; it can be two to three times actual weight for light cargo, so rate comparisons across
modes are meaningless without it. Weak answer says it is "the weight you pay for" without the
mechanism.

**7. "What limits a carrier's liability for lost cargo?"**

Strong answer: the applicable convention — Hague-Visby at 2 SDR/kg or 666.67 SDR per package,
Montreal at 26 SDR/kg since December 2024, CMR at 8.33 SDR/kg — and the fact that these caps are a
small fraction of most cargo value. Weak answer says "the carrier pays for the loss."

**8. "Demurrage and detention — what is the difference?"**

Strong answer: demurrage is cargo or container overstaying inside the terminal; detention is the
container overstaying outside it; two clocks, two invoices, usually two causes. Weak answer treats
them as one charge.

**9. "How would you attribute demurrage cost to a lane?"**

Strong answer: per-container billing, a cause code, and a rule for consolidated invoices; without
that the cost cannot be allocated and therefore cannot be managed. Weak answer says "spread it
across shipments."

**10. "What is CBAM and how does it enter a landed cost?"**

Strong answer: an EU charge on embedded emissions in cement, iron and steel, aluminium,
fertilisers, electricity, and hydrogen, applying from 1 January 2026, priced per tonne of CO2e
against the EU ETS average, with a 50-tonne annual de minimis per importer and first surrender due
30 September 2027. It is not ad valorem, so it does not scale with goods value. Weak answer calls
it a carbon tariff on all imports.

## 15. Key terms

- **Incoterms 2020** — eleven ICC rules allocating cost and risk, each requiring a named place.
- **Named place** — the location that completes the rule; without it the term is incomplete.
- **FCA** — seller delivers export-cleared to the buyer's carrier; the correct rule for containers.
- **FOB** — risk passes on board the vessel; sea and inland waterway only.
- **CIF / CFR** — seller pays freight to destination, risk passes at origin; the cost-risk split.
- **CIP** — as CPT with Institute Cargo Clauses (A) insurance, broader than CIF's Clauses (C).
- **DAP** — delivered at destination, duty unpaid, buyer clears import.
- **DPU** — as DAP but the seller unloads; the only rule requiring seller unloading.
- **DDP** — seller clears import and pays duty; requires an establishment the seller may not have.
- **TEU / FEU** — twenty-foot and forty-foot equivalent units of container capacity.
- **FCL / LCL** — full container load versus consolidated less-than-container load.
- **FTL / LTL** — full truckload versus less-than-truckload road carriage.
- **Chargeable weight** — the greater of actual and volumetric weight; the rating basis.
- **Volumetric divisor** — 6,000 cm3/kg under IATA rules; 5,000 common among integrators.
- **Revenue ton** — the greater of one cubic metre and one metric tonne, used for ocean LCL.
- **Stowage factor** — cubic metres occupied per tonne, including unavoidable void space.
- **Demurrage** — charge for cargo or container overstaying inside the terminal.
- **Detention** — charge for a container overstaying outside the terminal.
- **Free time** — days before demurrage or detention begins; set by tariff, not by treaty.
- **THC** — terminal handling charge, levied separately at origin and destination.
- **Bill of lading** — receipt, contract of carriage, and, when negotiable, document of title.
- **Customs value** — the duty base; transaction value adjusted to CIF in the EU.
- **HS code** — six-digit international classification, extended nationally.
- **CBAM** — EU charge on embedded emissions in six covered sectors, from 1 January 2026.
- **SDR** — IMF unit of account used to express carrier liability limits.
- **Hague-Visby / Montreal / CMR** — the ocean, air, and road liability regimes.
- **Landed cost** — delivered unit cost including freight, duty, fees, and financing.

## 16. Sources

**Peer-reviewed and academic**

- Krammer, P. and Schäfer, A. (2025). "Energy and economic benefits from economies of scale in
  intercity freight transportation." *npj Sustainable Mobility and Transport*. DOI:
  10.1038/s44333-025-00028-6. Source of the per-tonne-km cost and load-elasticity figures.
- Lee, C. and Lee, S. (2024). "Limitations of and proposals for L/C application of optional
  on-board B/L under the Incoterms 2020 FCA rules." *International Trade, Politics and
  Development*. DOI: 10.1108/ITPD-05-2024-0030.
- OECD International Transport Forum (2022). *Mode Choice in Freight Transport*. ITF Research
  Report. Qualitative mode-choice evidence and external-cost comparisons.
- Panteia / KiM (2021). *Cost Figures for Freight Transport*. Dutch national cost benchmarks by
  vehicle type, in EUR per tonne-km.

**Standards, treaties, and official**

- International Chamber of Commerce, *Incoterms 2020*. The eleven rules, with articles A1–A10 and
  B1–B10 and the introduction.
- ICC Germany (January 2024). Guidance note on FCA and CPT for containerised cargo through ports,
  based on a 2023 ICC Commission on Commercial Law and Practice port survey.
- ICC Digital Library, Incoterms 2020 Q&A and "Differences between Incoterms 2010 and 2020."
- ICC, *UCP 600* — Uniform Customs and Practice for Documentary Credits.
- WTO Customs Valuation Agreement; World Customs Organization HS nomenclature.
- Regulation (EU) 2023/956 establishing the Carbon Border Adjustment Mechanism, as amended by
  Regulation (EU) 2025/2083.
- Regulation (EU) 952/2013, the Union Customs Code, on declarant establishment.
- International Maritime Organization, SOLAS regulation VI/2 and MSC.1/Circ.1475 on verified gross
  mass.
- Hague-Visby Rules; Hamburg Rules; Rotterdam Rules; Montreal Convention 1999; CMR Convention 1956;
  COTIF/CIM.
- United States Federal Maritime Commission, 46 CFR 545.5 and 46 CFR Part 541.

**Industry and analyst**

- Xeneta weekly ocean container market updates, September and October 2026. Grade D.
- FreightWaves, "New conundrum: Asia-US, Europe ocean rates diverge," 26 October 2026, and "How
  new capacity is raising rates in chaotic ocean market," 20 October 2026. Grade E.
- Leschaco, Sea Freight Market Update, September 2026. Grade D.
- IATA, "Global Air Cargo Demand Achieved Record Volume in 2025," 29 January 2026; August 2026 air
  cargo market analysis, released 29 September 2026. Grade C.
- Air Cargo News, "IATA revises cargo growth forecast for 2026," June 2026. Grade E.
- Federal Maritime Commission, demurrage and detention collection data, April 2020 to March 2025.
  Grade B.

**Note on citations.** Market figures are reported as the sources state them, with dates, and are
not independently verified. Vendor, forwarder, and trade-press sources are graded D or E in the
text and describe a market at a point in time, not a fact about the lane. The reported removal of
the EU low-value consignment relief is Grade E and should be checked against the official text.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_manufacturing-basics.md`](07_manufacturing-basics.md)
- Next topic: [`09_warehousing-and-sourcing.md`](09_warehousing-and-sourcing.md)
- Related: [`06_cost-and-working-capital.md`](06_cost-and-working-capital.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Deeper treatment of freight markets and trade policy: [`../../../course/36_global_freight_ports_trade_policy.md`](../../../course/36_global_freight_ports_trade_policy.md)
- Deeper treatment of routing and last mile: [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
- Return to the part map: [`../_index.md`](../_index.md)
