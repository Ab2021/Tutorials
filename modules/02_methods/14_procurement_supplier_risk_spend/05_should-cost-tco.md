# Should-Cost and Total Cost of Ownership

> Module 14 — Procurement, Supplier Risk, and Spend Analytics
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 14, topics 01 and 02
> Version 2026.5

## 1. Why this topic exists

A quoted price is not a delivered outcome. A procurement
manager can negotiate a lower piece price and still increase
the cost of meeting demand if the new source adds freight,
defects, inventory, qualification expense, or exposure to an
interruption. Conversely, an apparently expensive supplier
may make the whole operation cheaper by supplying conforming
units close to the point of use.

Should-cost and total cost of ownership (TCO) answer
different questions. Should-cost asks what a reasonably
efficient process might cost to produce a defined product
under specified conditions. TCO asks what the buying
organization expects to spend, or economically sacrifice, to
acquire and use the same functional outcome over a stated
horizon. Neither is a supplier's actual accounting ledger,
and neither tells the buyer what price a supplier must
accept.

The operating decision is an award, renewal, redesign, or
negotiation, not the preparation of a spreadsheet.
Engineering defines acceptable specifications; operations
provides demand, logistics, and defect consequences; finance
agrees which cash flows and opportunity costs are decision
relevant; procurement owns the commercial conversation. A
cost model is a shared hypothesis that permits these groups
to challenge assumptions before commitments are made.

This lesson uses a fictional industrial component in 2026 US
dollars (USD). The figures illustrate an analytical method;
they are not industry benchmarks, a supplier audit, or
evidence of actual market conditions. Forecast demand,
proposed terms, negotiation targets, and contractual
commitments are distinct. Legal, tax, customs, and
accounting treatment requires qualified review.

## 2. Learning outcomes

After this topic you can:

1. Define should-cost, landed cost, and TCO and explain why their boundaries differ.
2. Translate a drawing, forecast, and quote into comparable functional units and time periods.
3. Build a transparent bottom-up manufacturing cost model with yield, overhead, and profit.
4. Reconcile price, freight, duty, quality, inventory, risk, and switching without double counting.
5. Calculate expected annual cost, discounted multi-year TCO, and a break-even threshold.
6. Test assumptions and uncertainty, including correlated disruption and nonfinancial constraints.
7. Prepare an ethical negotiation plan with evidence, concessions, and a walk-away alternative.
8. Review data provenance, update the model, and communicate an auditable award recommendation.

Exercises 1–8 in section 12 test these outcomes in the same
order. Solutions and rubrics appear in section 13.

## 3. The core mental model

```text
Specification + volume + process + market evidence
                     |
                     v
      SHOULD-COST: credible production economics
                     | compare, do not substitute
                     v
  SUPPLIER OFFER: price, obligations, lead time, capacity
                     |
                     v
  LANDED + QUALITY + INVENTORY + RISK + TRANSITION
                     |
                     v
      TCO over one common horizon and functional unit
                     |
                     v
  FEASIBILITY -> SENSITIVITY -> NEGOTIATION -> AWARD
                     |
                     v
       actual outcomes -> variance -> model update
```

The should-cost model is a cost hypothesis, not an invoice
and not a claim to see a supplier's confidential books. Its
components have physical drivers: mass times material price,
machine time times an appropriate rate, labor time times a
loaded rate, and a justified allocation of support costs. A
realistic offer also needs commercially sustainable profit
and capacity to satisfy the specification. The estimate is
conditional on process, geography, volume, and utilization.

The TCO model is a buyer decision model. It starts with the
offer rather than replacing it with should-cost. It adds
buyer-borne costs that change with the decision and
subtracts any genuine benefits only once. Do not add
supplier production cost to the supplier's quoted price: the
quote already recovers production cost and margin. A cost
element can be visible in both models but belongs only once
in the TCO cash-flow bridge.

The functional unit is the service delivered, not
necessarily an ordered unit. For example, compare
alternatives that meet the same annual requirement for
accepted components, drawing revision, delivery point, and
service standard. If the quote is per shipped unit and
defects require replacement purchases, divide the good-unit
requirement by the expected acceptance yield. If the
supplier replaces defects free of charge, model the
contract's actual remedy instead. The worked case
deliberately assumes annual ordered quantity is fixed and
replacement costs are included in the stated handling cost;
its demand is therefore an order forecast rather than a
guaranteed accepted-unit requirement.

Price is observable when an offer is firm, but many other
inputs are estimates. Point estimates and uncertainty bands
must be kept apart. An expected disruption loss is a
probability-weighted average, not a guaranteed annual
expense; a severe scenario or financial constraint may
matter even when expected cost is low. Separate financial
comparisons from minimum quality, safety, continuity, and
regulatory requirements that cannot be bought off by a low
bid.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Should-cost | Estimated efficient supplier cost under stated operating assumptions, USD/unit | Treating an estimate as the supplier's actual audited cost |
| Cost driver | Physical or commercial factor that changes cost, such as kg/unit | Using a cost pool without its activity basis |
| Bill of materials | Itemized material quantities and specifications, kg or pieces per unit | Assuming net design weight equals purchased input |
| Yield | Good output divided by input, proportion | Adding scrap percentage to output instead of dividing by yield |
| Cycle time | Time required per operation or unit, minutes/unit | Confusing cycle time with labor content or lead time |
| Setup | Changeover activity before a run, hours/run | Charging one setup to every unit |
| Capacity utilization | Actual productive use divided by available capacity, percent | Assuming 100% utilization without downtime |
| Overhead | Indirect cost allocated by a documented driver, USD/unit | Applying an overlapping factory rate and overhead markup |
| Markup | Profit divided by cost, percent | Treating markup as margin |
| Margin | Profit divided by selling price, percent | Dividing cost by one plus margin |
| Landed cost | Acquisition cost through a defined delivery point, USD/unit | Assuming every organization includes the same items |
| Incoterms | ICC rules allocating specified delivery obligations and risks | Assuming a rule transfers title or overrides a contract |
| Duty | Import charge computed using the applicable customs rules | Applying a rate to an invented customs value |
| Recoverable tax | Tax eligible for credit or refund under relevant law | Treating all invoiced tax as a permanent cost |
| Quality cost | Incremental prevention, appraisal, internal failure, or external failure cost | Counting the same replacement and defect cost twice |
| Carrying cost | Cost of capital and relevant inventory-related costs, USD/year | Multiplying the annual rate by annual purchases rather than average inventory |
| Expected loss | Probability times monetary consequence for a specified event and period | Describing an expected value as a certain payment |
| Switching cost | Incremental cost caused by moving supplier, USD at a stated time | Adding costs that would occur under either option |
| Sunk cost | Cost already incurred and unchanged by the decision | Letting historical spending determine the new award |
| Discount factor | Conversion from a future amount to present value | Discounting a current payment as if it occurs in year one |
| Net present value | Sum of discounted future net cash flows, USD at valuation date | Combining real cash flows with a nominal discount rate |
| Sensitivity | Change in the result as an assumption changes | Changing several parameters and attributing the difference to one |
| Scenario | Internally consistent joint set of assumptions | Combining incompatible best cases into a single forecast |
| Break-even | Input value at which the compared alternatives have equal cost | Treating an infeasible break-even value as a viable offer |
| Forecast | Best estimate of future demand, not a commitment | Treating a planning forecast as a minimum purchase obligation |
| Target | Desired negotiation outcome, not the agreed contract | Booking a target as realized savings |
| BATNA | Best alternative to a negotiated agreement | Confusing a desired price with a feasible outside option |
| Open-book | Voluntary disclosure of specified cost information under agreed protections | Assuming disclosure verifies the supplier's entire cost base |

## 5. The mechanics, step by step

### Step 1 — Define the decision and decision rights

Name the decision maker, alternatives, decision date,
horizon, and what happens if no agreement is reached.
Specify whether this is a new award, renewal, make-or-buy
comparison, design change, or supplier recovery. These
decisions have different counterfactuals. If the incumbent
can continue only six months, do not compare a three-year
new offer with an imaginary three-year incumbent commitment.

Identify mandatory constraints before optimizing cost:
approved drawings, safety, compliance, geographical
requirements, verified capacity, financial health, and
service coverage. Record evidence for each and an owner who
can accept or reject it. A failed mandatory requirement is
not a small cost penalty. If a constraint is temporarily
waived, record who has authority and when the waiver
expires.

### Step 2 — Fix the comparison basis

Describe the functional requirement, part revision, usage,
annual demand forecast, delivery location, service level,
and price currency. Confirm that the competing quotations
cover the same tooling, packaging, test certificates,
intellectual property rights, warranty, and freight
boundary. Use a conversion table for currencies and units.
Do not compare USD per kilogram with EUR per thousand pieces
until both quantities and exchange-rate dates are
reconciled.

State the period and timing of every cash flow. For an
annual model, document when costs occur and whether they
repeat. For a multi-year model, document the inflation
assumption, price escalators, volume changes, and discount
rate. Nominal cash flows need a nominal discount rate; real
cash flows need a real rate. The buyer's approved rate is a
decision input, not a universal market constant.

### Step 3 — Map the manufacturing route

Obtain the design drawing and bill of materials, then
describe plausible processes from raw material to packed
product. Identify input mass, yield, machine operations,
cycle times, number of cavities or simultaneous parts,
changeovers, inspection, packing, and shipment. A route made
for a different product family is at most a starting
hypothesis. Ask engineering to challenge the route and
supplier-facing staff to identify feasible alternative
methods.

At each operation show input, usable output, scrap, and
where rework occurs. When successive yields are independent
process fractions, multiply them to get overall yield. For a
net material requirement of 1.20 kg and a 96% material
yield, input is 1.20 / 0.96 = 1.25 kg per good unit. If
scrap is sold, subtract only the recoverable net proceeds;
do not subtract its full purchase price and also credit it
as a sale.

### Step 4 — Price direct materials

Multiply gross input quantities by evidence-based purchase
prices for the relevant grade, form, location, volume, and
date. A commodity index may be a proxy for an input, but it
is not necessarily the supplier's delivered purchase price:
conversion, alloy, freight, minimum quantities, rebates, and
timing can matter. Distinguish a published index from a
verified supplier invoice, and never represent one as the
other.

Separate production scrap from downstream buyer defects. If
the supplier's process yield is already reflected in its
input quantity, do not apply the same loss again to its
finished-unit price. Record who owns scrap proceeds and
whether the price is net of recovery. A high scrap-credit
assumption can create a misleadingly low estimate if
recovery is uncertain.

### Step 5 — Estimate conversion and support

Convert cycle time to machine hours per good unit and
multiply by a machine-hour rate that covers exactly the
costs named in the rate card. Add direct labor only when it
is not included in that rate. If one operator attends two
machines, allocate labor by actual attention and coverage,
including breaks and quality checks. Divide setup cost by
expected run size, not annual demand unless there is truly
one setup per year.

Use annual available capacity, normal utilization,
maintenance, energy, and depreciation assumptions that match
the supplier's scale. A rate based on an expensive machine
dedicated to one product is not comparable with a rate for
shared capacity unless allocation is explicit. Avoid
treating a financial depreciation charge as the cash
purchase of a machine in the same time period. Quote an
uncertainty range where the route, rate, or utilization is
unverified.

### Step 6 — Add overhead and profit consistently

Identify which indirect expenses are absent from direct
rates: factory support, quality administration, engineering,
sales, and general administration. Apply a documented
allocation base, such as conversion cost, with a range and
evidence. Separate one-time tooling or qualification from
recurrent unit economics. If overhead is already included in
the fully burdened machine rate, add only the remaining
non-overlapping pools.

A cost of USD 40 per unit with 10% markup gives a price of
USD 44 per unit. A 10% margin gives USD 40 / (1 - 0.10) =
USD 44.44 per unit. A model with no allowance for supplier
returns is not a neutral estimate of a sustainable offer.
Profit, investment, financing, warranty exposure, and
supplier bargaining power may warrant discussion, but the
model does not establish a fair or legally required margin.

### Step 7 — Construct the price-to-TCO bridge

Start from the contractual offer: unit price times volume,
with price tiers and minimum commitments. Add buyer-paid
inbound transport, insurance, customs brokerage, duty, and
nonrecoverable taxes according to actual contractual and
legal boundaries. State the customs value and rate evidence
rather than assuming that duty is always price times a fixed
rate. If a supplier quote already includes freight to the
buyer, do not add it again.

Add quality costs that are incremental to each choice:
inspection, disruption of production, sorting, rework,
warranty, or scrap not reimbursed. Apply the expected defect
rate to the appropriate population and make the cost per
defect consistent with included replacements. Keep
prevention spending visible if it reduces future defects,
and avoid adding a full warranty claim on top of the same
costs already included in the loss estimate.

Average inventory is the relevant base for a percentage
carrying charge, not annual purchases. Model cycle stock,
safety stock, in-transit inventory, ownership point, and
relevant payment terms consistently. Use capital cost on the
inventory value actually financed; storage or obsolescence
can be separate costs if not already included in the rate.
For service purchases, replace inventory with other relevant
drivers such as implementation support, integration,
downtime, and exit costs.

### Step 8 — Model uncertainty and disruption

For each discrete event, define the time period,
probability, exposure, mitigation, and loss conditional on
the event. For a 5% annual disruption probability and USD
100,000 loss conditional on disruption, expected loss is USD
5,000 per year. This assumes a single modeled event per
year; if the number of events is random, use an appropriate
frequency-severity model. Do not add overlapping event
losses that represent the same shutdown.

Scenario analysis preserves dependencies. For example, a
port closure can simultaneously raise freight and delay
inventory; a single coherent disruption scenario should
capture both once. Probability estimates may be weak,
nonstationary, or dependent on sourcing concentration.
Present the expected cost alongside tail exposure, recovery
time, and service constraints. Do not interpret a plausible
scenario as a calibrated probability.

### Step 9 — Treat transition and timing explicitly

Include new tooling, validation, supplier onboarding, data
integration, training, qualification samples, dual-running,
and termination charges only when incremental to the
alternative. Place each cost in the period in which it is
paid. Existing unrecoverable qualification spending is sunk.
Salvage value and contractual exit obligations belong in the
horizon when they differ between alternatives and are
credible.

Discount annual end-of-year costs by 1 / (1 + r)^t and add
time-zero costs without discounting. State whether costs are
cash flows, opportunity costs, or noncash allocations. For
comparing alternatives, consistently include all
decision-relevant economic costs even when an accounting
system reports them differently. A short horizon can
unfairly penalize an upfront investment whose benefits
continue after the horizon, so show the horizon sensitivity
and terminal assumptions.

### Step 10 — Test, negotiate, and implement

Reconcile each model element to a source, date, owner, and
confidence grade. Check units, formulas, double counting,
and whether the recommendation changes under reasonable
alternatives. Calculate the price or volume at which the
choice switches. Where assumptions are uncertain and
material, define an evidence-gathering action and deadline
before negotiating an irreversible commitment.

Convert the analysis into questions and tradeable terms
rather than asserting that an estimate proves a supplier is
overpriced. Discuss the operating drivers: yield, order
size, delivery frequency, lead time, specification, capacity
reservation, payment terms, and indexation. Verify a
concession's cost and benefit for both parties. Seek legal
review of confidentiality, competition, and contract terms.
After award, track actual price, volume, quality, freight,
lead time, and realized transition costs against the
approved baseline.

## 6. Worked example

### Case and decision

In this fictional 2026 example, a buyer forecasts orders of
100,000 machined housings per year for three years. The
forecast is not a purchase commitment. Each supplier quotes
the same drawing, packaging, and delivery performance; the
buyer will independently verify qualification and capacity
before an award. Prices and costs are USD. Costs recur
unchanged and are paid at each year end unless explicitly
stated otherwise. The buyer uses an 8% annual discount rate.

Supplier A is the incumbent. Its quoted price is USD 48.00
per ordered unit. Supplier B is a potential entrant at USD
45.00 per ordered unit. B requires USD 80,000 of incremental
validation and onboarding at time zero; A has no incremental
transition expense. Existing historic tooling is sunk in
both comparisons. The buyer's forecast is 100,000 ordered
units, not a guarantee of 100,000 accepted units. Both
suppliers replace rejected units free; the defect cost below
covers handling and lost capacity not otherwise reimbursed.
Confirm that delivery capacity covers replacements.

### Independent should-cost for the entrant

Engineering estimates a plausible process, with all values
per good shipped unit. Direct material is 1.25 kg at USD
8.00/kg, giving USD 10.00. Machining is 0.20 machine-hours
at USD 55/hour, giving USD 11.00. Direct labor not included
in that rate is 0.10 labor-hours at USD 30/hour, giving USD
3.00. Setup is USD 1,000 per run of 5,000 units, giving USD
0.20. Packing is USD 0.80. These direct costs total USD
25.00 per unit.

Factory support and administration are modeled as 40% of
direct cost, or USD 10.00 per unit, bringing cost to USD
35.00 per unit. A 20% markup on cost gives USD 7.00 and a
modeled selling price of USD 42.00 per unit. This is not a
negotiated price, not a measured supplier cost, and not
proof that USD 45.00 is excessive. Differences may reflect
capacity, actual material, cost of capital, warranty, sales
costs, or process assumptions. The buyer requests evidence
before relying on the USD 3.00 per unit gap.

### Annual buyer economics

| Annual element | A | B | Calculation and scope |
|---|---:|---:|---|
| Purchase | 4,800,000 | 4,500,000 | 100,000 orders times quoted USD/unit |
| Inbound freight and duty | 100,000 | 260,000 | Buyer-paid, not included in quotes |
| Defect handling | 40,000 | 100,000 | 2% versus 5% of orders, USD 20 per defect |
| Inventory carrying | 30,000 | 60,000 | Annual buyer-financed stock costs |
| Expected disruption | 20,000 | 60,000 | Annual probability times conditional net loss |
| Total per year | 4,990,000 | 4,980,000 | Sum of non-overlapping buyer costs |

The freight and duty amounts are consolidated estimates, not
a legal duty calculation. A's quality cost is 100,000 × 0.02
× USD 20 = USD 40,000 per year. B's is 100,000 × 0.05 × USD
20 = USD 100,000 per year. A's buyer-financed average
inventory is USD 200,000 at a 15% annual carrying rate; B's
is USD 400,000 at the same rate. The rate includes the
modeled storage, capital, and obsolescence costs; no
separate duplicate storage charge is added.

A has a 10% annual event probability and USD 200,000
conditional net loss, producing USD 20,000 annual expected
loss. B has a 20% annual event probability and USD 300,000
conditional net loss, producing USD 60,000. These are
illustrative expected economic losses, not certain cash
payments. Their event definitions exclude the routine
freight, quality, and inventory costs above. Tail and
financial resilience remain separate considerations.

B's annual expected saving is only USD 10,000, despite a USD
300,000 annual price saving. The bridge is USD 300,000 less
USD 160,000 additional freight and duty, USD 60,000
additional defect handling, USD 30,000 additional inventory
carrying, and USD 40,000 additional expected disruption. The
result is USD 10,000. This comparison uses B's quote, not
the hypothetical USD 42.00 should-cost price.

### Three-year present value and break-even

The three annual discount factors are approximately
0.925926, 0.857339, and 0.793832; their sum is 2.577097. A's
present value is USD 4,990,000 × 2.577097 = approximately
USD 12,859,714. B's is USD 80,000 + USD 4,980,000 × 2.577097
= approximately USD 12,913,943. A is lower by approximately
USD 54,229 in present value. Values may differ by a dollar
from rounded intermediate factors.

To break even, B needs an annual saving of USD 80,000 /
2.577097 = approximately USD 31,043, versus the current USD
10,000. The additional annual improvement needed is
approximately USD 21,043, or USD 0.21043 per ordered unit at
100,000 units per year. Holding all else fixed, B's
break-even unit quote is approximately USD 44.78957, not USD
42.00. A quote at that level is an analytical threshold, not
necessarily a feasible or acceptable negotiated agreement.

If B reduces its annual freight and duty by USD 40,000 with
no compensating costs or changed service, its annual saving
becomes USD 50,000 and its three-year net present value
advantage becomes about USD 48,855. This is a scenario, not
an achieved saving. If its disruption exposure increases,
the decision can reverse again. The recommendation is to
retain A on the stated assumptions, subject to its actual
three-year availability and all mandatory requirements, and
to investigate whether B can credibly change the economics
or resilience.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard
library. It reproduces the case using decimal arithmetic.
Annual totals are end-of-year expected economic costs, not
guaranteed invoices. The program fixes annual volume, price,
quality, inventory, and event probabilities; it does not
simulate tail events or tax effects.

```python
from decimal import Decimal, getcontext

getcontext().prec = 28
D = Decimal
volume = D('100000')
rate = D('0.08')
years = 3

suppliers = {
    'A': {
        'price': D('48'), 'freight_duty': D('100000'),
        'defect_rate': D('0.02'), 'cost_per_defect': D('20'),
        'inventory': D('200000'), 'carry_rate': D('0.15'),
        'event_probability': D('0.10'), 'event_loss': D('200000'),
        'transition': D('0'),
    },
    'B': {
        'price': D('45'), 'freight_duty': D('260000'),
        'defect_rate': D('0.05'), 'cost_per_defect': D('20'),
        'inventory': D('400000'), 'carry_rate': D('0.15'),
        'event_probability': D('0.20'), 'event_loss': D('300000'),
        'transition': D('80000'),
    },
}

def annual(s):
    return (volume * s['price'] + s['freight_duty']
            + volume * s['defect_rate'] * s['cost_per_defect']
            + s['inventory'] * s['carry_rate']
            + s['event_probability'] * s['event_loss'])

def pv_factor():
    return sum((D(1) / (D(1) + rate) ** t
                for t in range(1, years + 1)), D(0))

factor = pv_factor()
a, b = (annual(suppliers[name]) for name in ('A', 'B'))
pv_a = suppliers['A']['transition'] + a * factor
pv_b = suppliers['B']['transition'] + b * factor
needed = (pv_b - pv_a) / (volume * factor)
break_even = suppliers['B']['price'] - needed
print(f'Annual A: ${a:,.2f}; annual B: ${b:,.2f}')
print(f'PV A: ${pv_a:,.2f}; PV B: ${pv_b:,.2f}')
print(f'A advantage: ${pv_b - pv_a:,.2f}')
print(f'B break-even price: ${break_even:.5f}/unit')
```

Expected output, allowing normal rounding:

```text
Annual A: $4,990,000.00; annual B: $4,980,000.00
PV A: $12,859,713.97; PV B: $12,913,943.00
A advantage: $54,229.03
B break-even price: $44.78957/unit
```

> **Note:** The authoritative calculation is the code's full-precision arithmetic. Rounded factors should not be reused for a final award. Expected disruption is included as a decision cost rather than represented as certain payment.

## 8. Data, standards, and regulatory context

### Data lineage and evidence

The product drawing, revision, and approved supplier list
define the comparable item. The bill of materials and
routing support engineering quantities; purchase orders and
invoices show contractual price and actual spend; receipt
and inspection records show delivery and measured defects.
Freight invoices, customs entries, inventory snapshots, and
incident records supply other cost drivers. Their grains
differ. Join only after documenting supplier, part,
revision, location, date, currency, quantity, and unique
transaction identifiers.

An invoice date is not necessarily the receipt date or the
consumption date. A credit memo may reverse a previous
invoice rather than represent a negative purchase. Some
defect records count defects while others count defective
units, and a single unit can have more than one defect. A
warehouse balance observed at month end is not a daily
average if inventory cycles sharply. Reconcile transaction
totals to finance and quantify unmatched records before
presenting savings.

Provenance must distinguish supplier assertion, engineering
estimate, market index, contractual term, and observed
internal transaction. Store the source date, unit, owner,
transformations, confidence, and update trigger with each
assumption. A single point estimate can conceal a stale
source or an incompatible manufacturing route. Request
permission before using confidential supplier material
beyond its agreed purpose, and limit access to negotiation
teams with a legitimate need.

### Commercial and customs boundaries

The International Chamber of Commerce's Incoterms 2020 rules
specify defined delivery-related responsibilities, risk
transfer points, and allocation of specified transport and
clearance obligations when incorporated into a contract.
They do not themselves determine ownership transfer, payment
terms, product conformity, or the entire contract. The named
place matters. Check the actual agreement, insurance, and
customs arrangement before assigning freight or inventory
risk.

Customs valuation, classification, origin, preferential
treatment, and duty rates are jurisdiction- and
transaction-specific. The World Trade Organization customs
valuation agreement provides an international framework;
implementing laws and current local guidance govern actual
declarations. A model's simple duty assumption is not a
customs ruling. Use the relevant authority's current tariff
schedule and qualified trade advice; document entry date,
valuation basis, classification, and eligibility evidence.
Nonrecoverable taxes can be economic costs, while
recoverable taxes generally require cash-timing treatment
rather than permanent expensing.

### Accounting and environmental claims

IAS 2 addresses accounting for inventories under IFRS,
including purchase and conversion costs and specified
exclusions. That accounting boundary is not identical to
decision-relevant TCO: lost production, opportunity cost,
and some transition expenses may matter to a sourcing
decision even when they are not inventory costs. US GAAP or
another applicable framework may differ. Finance should
reconcile the decision model to the applicable reporting
treatment rather than relabeling an economic estimate as an
accounting balance.

ISO 20400 provides guidance on integrating sustainability
into procurement; it is not itself a substitute for
applicable legal obligations or a universal carbon price. If
monetizing emissions, specify organizational and product
boundaries, activity data, factors, price basis, period, and
whether any carbon cost is already embedded in energy or
supplier prices. Report physical emissions separately when
monetization is uncertain. Avoid claiming a reduction merely
because costs have shifted to an unobserved tier.

### Evidence quality controls

The model owner should define a refresh cadence: prices and
exchange rates at the negotiation date, engineering routes
when specifications change, freight when lanes or service
levels change, and quality performance after enough
comparable receipts. Explain whether historical observations
are representative of proposed volume and supplier operating
conditions. A proprietary benchmark with unknown composition
is not equivalent to verified transaction data.

## 9. Industry practice and case evidence

### What the literature actually supports

Lisa Ellram's 1995 article, “Total cost of ownership: an
analysis approach for purchasing,” describes TCO as a
purchasing approach that looks beyond purchase price at
costs associated with acquisition and use. The article
supports the conceptual separation of purchase price from
ownership costs; it does not validate any numerical
assumption in this lesson. Its context and available data
should be considered before transferring a particular
implementation to another category.

The US Government Accountability Office's Cost Estimating
and Assessment Guide describes principles for credible cost
estimates, including documentation, sensitivity analysis,
risk analysis, and updating estimates with actual costs. It
addresses a broader government acquisition context, not a
specific private-sector supplier negotiation. Its principles
are useful controls, but the guide does not prescribe the
exact model or parameters used here.

### Public case: Toyota and supplier collaboration

Toyota's published supplier sustainability guidance
describes expectations for supplier relationships and
supply-chain practices. It is evidence that Toyota publishes
supplier expectations, not evidence that a particular
supplier accepted an open-book estimate or achieved a stated
saving. [INFERENCE] A buyer seeking durable cost improvement
can use a shared process map and performance discussion to
identify operational changes rather than treating every
estimated cost gap as supplier profit. Whether a supplier
can disclose its costs depends on agreement and applicable
competition and confidentiality rules.

### Public case: public procurement lifecycle cost

European Commission guidance on life-cycle costing describes
how public purchasers may consider acquisition, use,
maintenance, end-of-life, and, under appropriate conditions,
environmental externalities. This is evidence that lifecycle
cost is a recognized procurement evaluation approach. It
does not mean every public tender may choose any scoring
formula: applicable procurement law and the tender documents
determine permitted criteria and disclosure. A private buyer
should still document comparable assumptions and avoid
inventing monetary benefits.

### Transfer to practice

High-volume manufactured parts often justify a process-level
should-cost model because small unit differences accumulate.
A one-off consulting engagement may require a
staffing-and-deliverables model instead: expected hours by
role, utilization, travel, management, intellectual
property, and service quality. Software ownership can be
driven by implementation, integration, licenses, data
migration, support, security, and exit rather than physical
freight. The principle is stable; the cost drivers and
evidence are not.

Teams commonly distinguish an internal negotiation model
from the contractual offer. A model can guide requests for
process improvement, but the agreed price and terms
determine the transaction. Actuals then test the
assumptions: if defects fall while inspection spend rises,
the net effect matters more than either metric in isolation.
Supplier relationship quality is not a license to skip
independent validation, and adversarial bargaining is not a
substitute for knowledge of the process.

### From estimate to an offer comparison

A useful cost-model review begins with the scope of the
offer, not a debate about the final number. Ask the supplier
whether the quoted item includes the specified coating,
inspection report, returnable packaging, delivery frequency,
and the same warranty period. If one quote excludes a
required certificate, obtain the cost of the certificate or
request an amended offer. A price cannot be compared fairly
by removing an obligation from only one side of the
comparison.

Separate fixed and variable charges. A monthly
capacity-reservation fee may be real even when no components
are ordered. A per-unit machining charge scales with volume,
while a minimum annual purchase obligation may create excess
stock when demand falls. Create a volume schedule covering
the contract's minimum, forecast, and plausible high demand,
and use the applicable price tier for each schedule. A
single average price based on forecast volume can conceal a
costly minimum commitment.

The supplier's quote may be denominated in a different
currency from the buyer's accounts. Record whether an
exchange rate is a forward contract, a spot rate, a planning
assumption, or an indexation clause. Convert amounts at a
consistent valuation date for comparison. If the buyer can
hedge, the hedge cost and timing are decision relevant; do
not assume that using a single exchange rate eliminates
exposure after award. Currency uncertainty can change the
ranking even when the supplier's local price stays constant.

### Negotiation is a design problem

Before the meeting, distinguish facts from hypotheses and
requests. A signed quote is a fact about the offered terms,
subject to its expiry. A projected supplier yield is a
hypothesis until supported. A request for a lower price is
an objective, not evidence that the supplier can produce at
that cost. Mark each line accordingly so that the
negotiation does not depend on false precision.

The buyer can prepare a range rather than a single target.
The upper end can be the cost at which the next-best
feasible option becomes preferable, subject to service and
risk constraints. The desired target can reflect operational
improvements with evidence, but it should not be represented
as the supplier's actual cost. The lower end of a cost range
may require assumptions about capacity, automation, or
material prices that the buyer cannot impose unilaterally.

Concessions should be conditional and measurable. For
example, a buyer may offer a longer forecast window in
exchange for a fixed freight charge and a documented service
commitment. A longer forecast is not necessarily a binding
order, and a commitment to buy a minimum quantity is a
different commercial obligation. Quantify the cost of the
concession under low and high demand before proposing it.
Document the person authorized to approve the commitment.

A supplier may resist disclosing direct costs for legitimate
competitive or contractual reasons. A productive alternative
is to validate observable drivers together: dimensional
tolerance, run size, order frequency, yield improvement, or
logistics plan. Sharing another supplier's confidential
quote or strategically sensitive information is not required
to challenge a model. Competition-law requirements depend on
jurisdiction and facts; obtain legal advice before
information exchanges or joint activities that could
restrict competition.

### Quality economics need a causal chain

Suppose an inspection record shows 500 nonconformances. That
number does not tell you whether there were 500 defective
units, whether some units had several defects, or whether
the buyer paid to repair them. Link inspection events to
lots, units, dispositions, supplier credits, and actual
labor. Classify detection at the supplier, at receipt,
during production, and after customer delivery; the cost of
the same defect can differ substantially by detection point.

The expected cost of a defect is conditional on what happens
next. If 90% of defects are detected at receipt and cost USD
5 each, while 10% cause USD 100 of downstream loss each, the
expected loss is 0.90 × USD 5 + 0.10 × USD 100 = USD 14.50
per defect. This example assumes the branches are mutually
exclusive and exhaustive. If one defect can trigger both
losses, use a model that includes the joint pathway rather
than forcing a false choice.

Supplier credits and warranty recoveries are not always
equal to the buyer's total loss. A credit for the purchase
price may not cover line stoppage, sorting, expedited
replacement, or customer compensation. Conversely, a
supplier may reimburse some of those costs. Record the
contractual remedy, actual collection rate, and delay in
recovery. Subtract expected recoveries only once and retain
a gross-to-net bridge so the model can be audited.

### Inventory, lead time, and payment terms

A useful stock bridge distinguishes cycle stock, safety
stock, and in-transit stock. If an order quantity is Q units
and consumption is steady, a basic sawtooth model has
average cycle stock Q/2 units, assuming immediate
replenishment and no backorders. Safety stock is additional
protection against demand and supply variation; it should be
calculated under the chosen service policy rather than
assumed to be zero. In-transit stock matters when ownership
and financing transfer before receipt.

For illustration, if Q is 2,000 units and unit value is USD
30, average cycle stock is 1,000 units, or USD 30,000. At a
15% annual carrying rate, the annual cycle-stock cost is USD
4,500. This excludes safety stock unless the stated value
includes it. If a proposal doubles order quantity, average
cycle stock doubles under the same assumptions, even if
fewer deliveries lower freight. Compare the incremental
carrying charge against the freight reduction.

Payment terms influence the cash conversion cycle, but avoid
treating the full invoice value as extra annual cost. A
simple approximation for financing an additional 30 days on
USD 1,200,000 annual purchases at a 12% annual financing
rate is USD 1,200,000 × 30/365 × 0.12, approximately USD
11,836 per year. This assumes purchases and payments are
evenly distributed, the rate applies to the relevant working
capital, and there are no other terms. If the stock
valuation or carrying rate already incorporates this
financing effect, do not add it again.

### Risk mitigation has its own economics

A backup supplier can reduce the severity or duration of a
disruption, but maintaining it may require qualification,
periodic orders, higher unit prices, duplicate tooling, and
engineering capacity. Compare the present value of those
incremental costs with the reduction in expected loss and
the effect on severe scenarios. A backup that shares the
same constrained sub-tier, power supply, or port may offer
less protection than the headline count of suppliers
suggests.

Suppose a backup program costs USD 35,000 per year and
reduces modeled expected annual interruption loss from USD
90,000 to USD 45,000. The expected net annual benefit is USD
10,000 before other effects. This result is contingent on
the estimated event distribution, program cost, and
successful recovery. It does not prove the program is
sufficient for a service obligation, and it does not mean a
USD 45,000 interruption expense will actually occur each
year.

A resilience review should ask how quickly the backup can
accept an order, ramp output, pass inspection, and deliver
to the required site. A signed framework with no available
capacity may not be an effective backup. Model mitigation
only after assigning capacity, recovery time, geographic
dependencies, and cost. If a disruption exceeds the buyer's
financial or operational tolerance, the decision may require
mitigation even when a simple expected-value calculation
does not justify it.

### Contract and performance follow-through

After negotiation, transform the agreed offer into an
executable baseline. Record unit price by part and currency,
effective dates, index base and reset formula, freight
boundary, service level, quality remedies, annual minimum
and maximum quantities, payment terms, ownership of tooling,
and exit arrangements. A spreadsheet assumption is not a
contract term until the authorized agreement makes it one.
Operations and accounts payable need the same terms in their
systems to prevent a negotiated rate from being lost in
execution.

Compare actuals with a baseline using a clear volume
convention. A price variance can be calculated as actual
quantity times actual price minus baseline price; it is
negative when actual price is lower under that sign
convention. A volume variance requires a separate baseline
quantity and consistent treatment of fixed costs. Do not
attribute a fall in total expenditure solely to procurement
savings when demand fell. Finance should approve the savings
methodology and prevent multiple teams from claiming the
same benefit.

For the fictional case, record B's actual freight and duty
by receipt, defect counts and paid handling hours by lot,
average financed stock by time bucket, and any disruption or
mitigation cost. Reconcile purchases and credits to accounts
payable and reconcile receipts to the warehouse. A monthly
dashboard can show both the original forecast and current
estimate; any revised estimate should retain version history
rather than overwrite the baseline. Disputes over a variance
should lead to a source-level investigation, not a silent
adjustment.

### What to communicate to decision makers

A concise decision memo should state the recommended option,
feasible alternatives, three-year economic difference,
one-time cash requirement, key uncertainty, mandatory
constraints, and next evidence gate. Show the price bridge
and a downside scenario next to the expected result. If the
conclusion depends on a supplier remaining available or
accepting a particular price, make that dependency explicit.
The memo should not state a hypothetical negotiation saving
as an approved budget reduction.

For the example, the preliminary recommendation favors A by
approximately USD 54,229 in present value. This is a narrow
result relative to the total modeled spend, and it is driven
by uncertain risk and transition inputs. A change in B's
freight, inventory, quality performance, or expected outage
loss could change the ranking. The decision maker should
therefore request credible supporting evidence and consider
the consequences of an outage, rather than interpreting the
point estimate as a guarantee.

A responsible owner also specifies what would change the
decision. B's price at approximately USD 44.78957 per unit
is one threshold if all other case assumptions remain
unchanged. Equivalent improvements could come from freight,
defect handling, inventory, or disruption loss, but a change
in one term can affect the others. A lower price achieved by
less inspection may increase quality cost. Recompute the
complete bridge whenever an operational term changes.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Price-only award | Buyer dashboard excludes operations costs | Costs migrate outside procurement | False saving | Approve a full boundary with finance |
| Double-counted production | Should-cost added to quote | Price already includes supplier costs | Inflated TCO | Start from quote and bridge incremental buyer costs |
| Inconsistent quality | Different defect denominators | Defects and defective units are mixed | Unfair comparison | Align units, period, and remedy |
| Unrealistic yield | Gross input equals net product mass | Production scrap ignored | Understated should-cost | Trace material through process |
| Overlapping overhead | Labor appears in machine rate and separately | Burden is included twice | Inflated estimate | Define rate composition |
| Wrong delivery point | Freight is quoted and added again | Incoterm boundary misunderstood | Overstated landed cost | Check named place and contract |
| Weak capacity evidence | Promised volume exceeds demonstrated output | Bottleneck and downtime ignored | Shortage | Validate realistic capacity and reserves |
| Unpriced transition | Qualification lacks an owner | Work is spread across departments | Missed upfront expense | Build a work breakdown with finance |
| Hidden tail exposure | Expected loss shown alone | Rare severe loss averages away | Unacceptable outage | Show scenario and recovery capacity |
| Arbitrary risk penalty | Score multiplied by dollars without evidence | Ordinal rank treated as money | Misleading optimization | Model explicit probability and severity |
| Unmatched horizon | One offer has three years, another one | Future options are implicitly assumed | Biased award | Align feasible horizon and renewal cases |
| Margin confusion | Profit percent applied inconsistently | Markup and margin differ | Wrong target | Show denominator and formula |
| Sunk-cost bias | Past tooling cited as future expense | Historical spend cannot change | Poor incremental decision | Identify cash flows affected by choice |
| Stale market proxy | Index date predates sourcing event | Input market or grade differs | Invalid negotiation anchor | Refresh and map index to item |
| False realized saving | Negotiation target entered as actual | Forecast volume treated as spend | Overstated results | Reconcile actual price and volume |
| Confidentiality breach | Supplier data circulated broadly | Access controls absent | Legal and relationship harm | Limit access and review agreements |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Volume is 100,000 orders per year | Converts rates to annual totals | Demand falls or changes mix | Recalculate with forecast bands and tier prices |
| Costs recur unchanged | Makes three-year illustration tractable | Inflation, indexation, or changing service | Model annual terms separately |
| Orders have comparable specification | Makes offers economically comparable | Revision or warranty differs | Engineering and legal compare scope |
| Replacements are free | Keeps purchases based on ordered volume | Contract excludes replacement or capacity | Model replacement purchases and delays |
| Defect loss excludes replacement price | Prevents double counting | Cost estimate already includes another charge | Trace a sample claim to its cash flows |
| Average stock values are financed by buyer | Supports carrying charge | Title and payment timing differ | Reconcile stock ownership and terms |
| Event loss excludes routine costs | Makes risk additive | Same freight or defect appears twice | Map event consequence to cost ledger |
| Event probability is stable | Produces expected annual loss | Climate, geopolitical, or supplier changes | Refresh evidence and show scenarios |
| 8% rate is appropriate | Values timing consistently | Different financing or approved hurdle rate | Finance approves rate and sensitivity |
| No common-mode outage | Keeps supplier event terms separate | Shared port, sub-tier, or utility | Map common dependencies |
| Supplier A remains available | Defines a feasible counterfactual | Contract expires or capacity changes | Obtain a firm offer and feasibility check |
| No terminal difference | Restricts the horizon to three years | Residual tooling or termination differs | Add defensible terminal values |

## 12. Exercises

Each exercise uses USD unless stated otherwise. Show
formulas, assumptions, units, and decision implications. Do
not infer that any fictional number is a market benchmark.

### Exercise 1 — Boundaries (10 minutes)

Deliverable: classify five items as a should-cost input,
buyer TCO item, both in different roles, or neither:
material yield; supplier quote; buyer-paid freight; already
sunk tooling; supplier markup.

Acceptance criteria: explain why the supplier's production
cost is not added on top of its quote, and distinguish an
input from a cost already incurred. See answer 1.

### Exercise 2 — Comparable unit (15 minutes)

A buyer requires 9,600 accepted units. A new supplier has a
96% first-pass acceptance rate, no free replacement, and
charges USD 12 per unit ordered. Assume defects cannot be
repaired and rates are deterministic for planning.

Deliverable: calculate planned ordered quantity and purchase
cost. Acceptance criteria: show the yield denominator and
explain why the result is an estimate rather than a
guaranteed realized yield. See answer 2.

### Exercise 3 — Should-cost (20 minutes)

A component needs 1.8 kg net material at 90% yield. Gross
material costs USD 5/kg. Conversion requires 0.25
machine-hours at USD 40/hour, with labor included. Other
direct cost is USD 2/unit. Overhead is 20% of direct cost
and a 25% markup applies to full cost.

Deliverable: show material, direct total, overhead, full
cost, and modeled selling price per good unit. Acceptance
criteria: do not add labor again and do not confuse markup
with margin. See answer 3.

### Exercise 4 — TCO bridge (20 minutes)

For 10,000 ordered units, quote A is USD 30/unit with
freight USD 15,000 and quality USD 5,000. Quote B is USD
28/unit with freight USD 32,000 and quality USD 14,000. No
other cost differs.

Deliverable: calculate annual TCO and the lower-cost choice.
Acceptance criteria: show the purchase-price saving and the
net effect of the non-price differences. See answer 4.

### Exercise 5 — Discounting and break-even (20 minutes)

A change costs USD 21,000 immediately and saves USD 10,000
at each year end for three years. The annual discount rate
is 10%.

Deliverable: calculate discounted savings, net benefit, and
the minimum equal annual saving required to break even.
Acceptance criteria: discount each year and place the
transition at time zero. See answer 5.

### Exercise 6 — Uncertainty critique (20 minutes)

An analyst compares independent event expected losses for
two suppliers and declares the lower expected loss
inherently safer. Both rely on one shared port, and the
expensive alternative has a faster recovery site.

Deliverable: write a short critique and a revised analysis
plan. Acceptance criteria: identify common cause, tail
impact, recovery, and a data action; distinguish expected
loss from a service constraint. See answer 6.

### Exercise 7 — Negotiation design (25 minutes)

Use the worked example. Supplier B says it can reduce
freight by USD 40,000 per year if orders are consolidated,
but the change increases average inventory and could
threaten service. The supplier does not guarantee the
freight saving.

Deliverable: draft a negotiation agenda, evidence requests,
tradeable terms, and walk-away logic. Acceptance criteria:
specify who pays each cost, avoid treating an unverified
saving as realized, and identify a feasible alternative. See
rubric 7.

### Exercise 8 — Evidence and monitoring (25 minutes)

Design a model governance sheet for a three-year supplier
award with a commodity-linked price. The buyer has invoice
lines, inspection events, monthly stock snapshots, a market
index, and incident logs.

Deliverable: propose source fields, owners, refresh
triggers, variance measures, and confidentiality controls.
Acceptance criteria: reconcile grains and contract terms,
define baseline and actual, and give a response when a key
assumption changes. See rubric 8.

## 13. Self-check answers

### Answer 1

Material yield is primarily a should-cost input; a different
buyer-side acceptance yield can also affect TCO when defined
separately. Supplier quote is the purchase-price starting
point for TCO and a comparison point for should-cost.
Buyer-paid freight is a TCO item and may inform a supplier
process estimate only if that estimate explicitly includes
the same boundary. Sunk tooling is neither an incremental
cost nor a reason to add a cost to the new decision.
Supplier markup is a should-cost pricing assumption and is
already embedded in the quoted purchase price.

### Answer 2

Required orders = 9,600 / 0.96 = 10,000 units. Purchase cost
= 10,000 × USD 12 = USD 120,000. Actual accepted output is
uncertain if yield varies, so a service-level or buffer
analysis may be needed. Do not charge a second defect
replacement cost on top of the already inflated order
quantity unless it captures a distinct loss.

### Answer 3

Gross input is 1.8 / 0.90 = 2 kg per good unit; material
cost is USD 10. Conversion cost is 0.25 × USD 40 = USD 10.
Other direct cost is USD 2, so direct cost is USD 22.
Overhead is 20% × USD 22 = USD 4.40, giving USD 26.40 full
cost. Markup is 25% × USD 26.40 = USD 6.60, so modeled
selling price is USD 33 per good unit. Labor is already
inside the machine rate.

### Answer 4

A costs 10,000 × USD 30 + USD 15,000 + USD 5,000 = USD
320,000 per year. B costs 10,000 × USD 28 + USD 32,000 + USD
14,000 = USD 326,000 per year. B saves USD 20,000 in
purchase price but has USD 17,000 more freight and USD 9,000
more quality cost. A is lower by USD 6,000 per year.

### Answer 5

The annuity factor is 1/1.10 + 1/1.10² + 1/1.10³ =
approximately 2.486852. Present value of savings is
approximately USD 24,868.52; subtract the USD 21,000
time-zero cost to obtain approximately USD 3,868.52 net
benefit. Break-even equal annual saving is USD 21,000 /
2.486852 = approximately USD 8,444.41. The arithmetic
assumes the savings are certain and occur at year end.

### Answer 6

Expected loss alone does not describe tail severity,
recovery time, or dependence. Model a shared-port event
affecting both suppliers and supplier-specific events
without counting the port twice. For each scenario estimate
duration, unmet demand, lost contribution or recovery
expense, and mitigation, and test the faster recovery site's
capacity and activation terms. Obtain port exposure and
incident or engineering evidence. Treat minimum service or
maximum tolerable outage as a separate constraint where
required.

### Rubric 7

A strong answer calculates B's preliminary improvement only
after adding any incremental carrying and service costs;
requests lane quotes, consolidation cadence, inventory
model, and performance evidence; negotiates measurable
commitments and remedies; and compares with a genuinely
available A offer. It identifies who owns stock and delivery
risk and specifies a trial or approval gate. A weak answer
books the USD 40,000 as certain or demands the USD 42
should-cost as if it were proven supplier cost.

### Rubric 8

A strong answer names unique line identifiers, supplier and
part mappings, revision, transaction dates, currencies,
units, source and owner; reconciles invoices, credits, and
receipts; and states whether stock snapshots approximate the
relevant average. It records the index formula and reset
dates, tracks price and volume separately, flags changed
assumptions, restricts supplier-confidential information,
and assigns a review owner. A weak answer aggregates raw
events without checking grain or treats forecast savings as
realized.

## 14. Interview questions

1. **How does should-cost differ from TCO?** Strong: supplier-process estimate versus buyer lifecycle decision costs, with no double count. Weak: they are two names for the quoted price.
2. **How would you estimate material usage?** Strong: net mass, process yield, recovery, unit and grade evidence. Weak: net mass alone is always purchase quantity.
3. **What is the difference between markup and margin?** Strong: profit divided by cost versus price, with a numerical example. Weak: the percentages are interchangeable.
4. **How do you compare defective parts?** Strong: align good-unit requirement, replacement terms, defect denominator, and distinct consequences. Weak: apply an arbitrary penalty per quoted unit.
5. **Why might a lower unit quote lose?** Strong: freight, quality, working capital, disruption, and transition can exceed purchase savings. Weak: claims that a cheaper quote always wins.
6. **How would you model a rare shutdown?** Strong: conditional loss, frequency, dependence, recovery, and tail constraints. Weak: multiply an uncalibrated score by spend and call it probability.
7. **How do you treat switching costs?** Strong: incremental timing, sunk costs, horizon, and discounting. Weak: include all historical spending.
8. **What evidence makes a model credible?** Strong: source, timestamp, units, triangulation, range, sensitivity, and reconciliation. Weak: a spreadsheet with precise decimals.
9. **How would you negotiate a cost gap?** Strong: present assumptions as hypotheses, seek evidence, trade terms, and respect confidentiality. Weak: claim the estimate proves actual profit.
10. **When is the lowest TCO not an acceptable award?** Strong: mandatory qualification, legal, capacity, service, and resilience constraints can exclude it. Weak: score every violation as a small cost.

## 15. Key terms

- [Should-cost](#4-definitions-and-vocabulary): conditional process-based estimate.
- [TCO](#3-the-core-mental-model): buyer's full decision boundary.
- [Functional unit](#step-2--fix-the-comparison-basis): comparable delivered outcome.
- [Yield](#step-3--map-the-manufacturing-route): good output over input.
- [Scrap recovery](#step-4--price-direct-materials): net value of recoverable material.
- [Cycle time](#step-5--estimate-conversion-and-support): processing time per operation.
- [Setup](#step-5--estimate-conversion-and-support): run-level changeover effort.
- [Machine rate](#step-5--estimate-conversion-and-support): defined cost per machine-hour.
- [Overhead](#step-6--add-overhead-and-profit-consistently): allocated indirect costs.
- [Markup](#step-6--add-overhead-and-profit-consistently): profit divided by cost.
- [Margin](#step-6--add-overhead-and-profit-consistently): profit divided by price.
- [Landed cost](#step-7--construct-the-price-to-tco-bridge): cost through a named delivery boundary.
- [Incoterms](#commercial-and-customs-boundaries): contractually incorporated delivery rules.
- [Quality cost](#step-7--construct-the-price-to-tco-bridge): incremental cost of quality activities or failures.
- [Carrying cost](#step-7--construct-the-price-to-tco-bridge): cost of holding buyer-financed stock.
- [Expected loss](#step-8--model-uncertainty-and-disruption): probability-weighted loss.
- [Common cause](#answer-6): shared event affecting alternatives.
- [Scenario](#step-8--model-uncertainty-and-disruption): coherent joint assumptions.
- [Sunk cost](#step-9--treat-transition-and-timing-explicitly): past cost unaffected by the decision.
- [Switching cost](#step-9--treat-transition-and-timing-explicitly): incremental transition expense.
- [Discount factor](#step-9--treat-transition-and-timing-explicitly): future-to-present multiplier.
- [Break-even](#three-year-present-value-and-break-even): value where alternatives tie.
- [Forecast](#case-and-decision): estimate rather than commitment.
- [Negotiation target](#step-10--test-negotiate-and-implement): desired outcome, not an agreement.
- [BATNA](#4-definitions-and-vocabulary): best feasible alternative to agreement.
- [Evidence lineage](#data-lineage-and-evidence): trace from assumption to source.

## 16. Sources

- Ellram, Lisa M. “Total cost of ownership: an analysis approach for purchasing.” *International Journal of Physical Distribution & Logistics Management*, 25(8), 1995, 4–23. [Publisher record](https://doi.org/10.1108/09600039510099928). Foundational TCO framing; not validation of this lesson's illustrative inputs.
- U.S. Government Accountability Office. [Cost Estimating and Assessment Guide, GAO-20-195G](https://www.gao.gov/products/gao-20-195g), 2020. Public-sector estimating, documentation, risk, and sensitivity practices.
- International Chamber of Commerce. [Incoterms rules](https://iccwbo.org/business-solutions/incoterms-rules/). Official overview; consult the incorporated rule and complete contract for a transaction.
- World Trade Organization. [Customs valuation](https://www.wto.org/english/tratop_e/cusval_e/cusval_e.htm). International framework; national implementation and current tariff rules remain relevant.
- IFRS Foundation. [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/). Financial reporting standard, not a complete economic decision model.
- International Organization for Standardization. [ISO 20400:2017 Sustainable procurement — Guidance](https://www.iso.org/standard/63026.html). Guidance, not a blanket statement of legal duties.
- European Commission. [Life-cycle costing](https://green-business.ec.europa.eu/green-public-procurement/life-cycle-costing_en). Public procurement guidance; actual legal requirements depend on jurisdiction and tender.
- Toyota Motor Corporation. [Sustainability](https://global.toyota/en/sustainability/). Corporate disclosure; not independent verification of the fictional case.

## 17. Links

- [Module 14 index](_index.md).
- [Source-to-pay](01_source-to-pay.md).
- [Spend cube construction](02_spend-cube.md).
- [Module 14 survey](../../../course/14_procurement_supplier_risk_spend.md).
- [Simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
