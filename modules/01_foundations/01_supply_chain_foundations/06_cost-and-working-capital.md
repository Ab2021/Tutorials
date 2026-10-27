# Cost Taxonomy and Working Capital

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04, 05
> Version 2026.5

---

## 1. Why this topic exists

A supply chain spends money at every handoff and ties up cash between handoffs. Those are two
different statements, and almost every expensive mistake in this area comes from treating them
as one. Cost is an accrual: it is recorded when a resource is consumed. Cash is a timing: it
leaves when a supplier is paid and returns when a customer pays. A company can reduce cost and
worsen cash, or improve cash and increase total cost, and both outcomes can be reported as
success by different functions in the same week.

Topic 01 introduced the three flows and computed days inventory outstanding (DIO), days sales
outstanding (DSO), days payable outstanding (DPO), and the cash conversion cycle (CCC) on a
single month of data. That was the arithmetic. This topic is the economics behind the arithmetic:
what the cost categories actually contain, which of them are marginal, why the denominators
matter more than the formulas, how growth consumes cash, and how supply chain finance changes who
carries the burden without changing the physical chain at all.

Three examples show the shape of the problem.

1. **The holding cost rate that decided the order quantity.** A planner uses the 25% rate in the
   planning system. The company's actual rate, built from its own capital, storage, insurance,
   tax, obsolescence, shrinkage, and damage, is 13%. Because EOQ scales as one over the square
   root of the holding rate, order quantities at 25% are about 28% too small. The company runs
   more orders, more freight, and more stock-outs than its economics justify, and every report
   shows the policy being followed correctly.
2. **The DIO computed on revenue.** An analyst divides inventory by daily revenue rather than
   daily cost of goods sold (COGS). At an 80% COGS ratio, the reported DIO is 20% too low. The
   dashboard is internally consistent, the trend is real, and the level is wrong. A target set on
   the wrong level is a target that cannot be met without destroying service.
3. **The cost-to-serve model that recommended firing a good customer.** A distribution centre's
   fixed cost is allocated to orders on a per-order basis. A customer placing many small,
   predictable, high-margin orders is reported as unprofitable and put on a remediation list. Its
   orders consume almost no incremental warehouse cost, because the shift is already staffed and
   the space already leased. The average cost said "exit"; the marginal cost said "grow".

Each example is a cost-classification error before it is a modelling error. The fix is not a
better algorithm. It is a disciplined taxonomy and a clear statement of the question.

**Why this matters to a data scientist specifically.** Cost and cash are the two places where an
analytics result becomes a financial decision. A forecast improves a service metric; a cost model
changes a price; a working capital model changes a funding requirement; a cost-to-serve model
changes which customers a company keeps. In each case the model output is only as good as the
cost object it was built on. If you inherit a holding cost rate, an allocation key, or a DPO
denominator without knowing where it came from, you have inherited someone else's assumption and
attached your name to it.

This topic bridges the module. Inventory policy in Topic 05 is a cost trade-off; manufacturing in
Topic 07 is a conversion cost structure; transport in Topic 08 is a landed-cost structure. Module 42
takes the treasury and credit machinery further.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** any supply chain cost as ordering, holding, shortage, quality, transport, duty,
   reverse, or administrative, and state whether it is traceable or allocated, marginal or
   average, inventoriable or period.
2. **Build** a holding cost rate from its components for a named product family, and **quantify**
   what happens to an order quantity and a holding cost estimate when the rate is wrong by half.
3. **Compute** DIO, DSO, and DPO with the correct denominators, and **explain** why DPO on
   purchases differs from DPO on COGS.
4. **Distinguish** the operating cycle from the cash cycle, and **convert** a net working capital
   balance into a financing cost with stated units and rate.
5. **Calculate** the incremental working capital required to grow revenue by a stated percentage,
   and **compare** it with the profit the growth produces.
6. **Distinguish** reverse factoring, dynamic discounting, confirming, and inventory finance, and
   **state** the accounting and disclosure consequence of each.
7. **Compute** total landed cost under stated allocation bases, and **explain** why two products
   on the same shipment can carry very different uplifts over invoice price.
8. **Critique** a claim that working capital improved, identifying whether value was created,
   transferred, or merely reclassified.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: a unit of product climbing a cost ladder while a parallel cash
timeline runs underneath it, offset in time. The ladder is the accrual view. The timeline is the
cash view. The distance between them is working capital.

```text
   COST LADDER (EUR per unit, cumulative)        CASH TIMELINE (same unit)

   supplier invoice          100.00              t = -90   order placed (commitment)
   + inbound freight           8.00              t = -60   cash out: supplier
   + duty                     12.00              t = -45   cash out: freight, duty
   + brokerage/handling        2.00              t =   0   invoice issued
   = LANDED COST             122.00              t = +55   cash in: customer pays
   + warehousing               3.50
   + obsolescence              4.00              cash is out for 60 days,
   + expected stock-out        2.50              back 55 days after invoice:
   + reverse logistics         1.00              a 115-day funding gap
   = COST TO SERVE           133.00
                                                 area under the gap = financing cost
```

The ladder and the timeline are the same physical journey read by two different departments. The
ladder is what the general ledger will eventually record. The timeline is what the bank account
will actually experience. Every number in the ladder has a counterpart in the timeline, and the
counterpart usually arrives in a different period.

The picture has five properties worth memorising, because they explain most of what follows.

**Property 1 — Cost is a stack with a boundary, and the boundary is an accounting choice, not a
physical fact.** The supplier's invoice is a fact. Whether inbound freight sits in inventory or in
the period's expenses, and whether plant depreciation sits in product cost or overhead, are
policies. Two companies with identical physical operations can report different unit costs and
profit timing because they drew the boundary differently. Before comparing cost numbers, ask where
the boundary is.

**Property 2 — Traceable costs are facts; allocated costs are arguments.** A freight invoice for a
specific container is traceable to that container. Warehouse rent is not traceable to a specific
order; it is allocated by a rule somebody chose. Allocation is necessary, but an allocated number
is a negotiated position, not a measurement. When an allocated cost drives a decision, the
allocation rule is doing the deciding.

**Property 3 — Average cost reports; marginal cost decides.** Average cost is total cost divided by
total volume, and it is what financial statements need. Marginal cost is the change in total cost
from one more unit, order, or customer, and it is the only cost that should drive a volume
decision. Activity-based costing (ABC) produces averages with better resolution, not marginals.
Confusing the two is the most consequential error in this topic.

**Property 4 — Working capital is time, priced at the cost of capital.** Net working capital is
inventory plus receivables minus payables. Each term is a quantity of money and a duration:
multiply the duration by the daily flow to get the balance, and the balance by a financing rate to
get the annual cost of waiting. The days describe the duration; they are not a separate phenomenon.

**Property 5 — Every DPO day is someone else's DSO day.** Extending payables improves the buyer's
cycle and lengthens the supplier's; the physical flow is unchanged. Value is created only if the
party now carrying the financing can carry it more cheaply, or if the change removes a real
inefficiency. Otherwise the improvement is a transfer, and transfers return as price increases,
quality failures, or supplier exit.

### 3.1 The three questions behind any cost number

Every cost number answers a question with three parts: the **cost object** (unit, order, customer,
lane, SKU–location, channel, business unit), the **horizon** (one unit, one season, one year,
product life), and the **traceability rule** (direct attribution, driver-based allocation, arbitrary
allocation, exclusion). Write all three down: "cost per order, one-year horizon, freight traced and
warehouse allocated by lines picked" is a specification; "cost per order" is not.

### 3.2 The holding cost rate is a portfolio, not a constant

The holding cost rate is the annual cost of keeping one euro of inventory for one year. It is
usually expressed as a percentage and treated as a single company-wide parameter. It is neither
single nor stable: it is the sum of eight costs with eight drivers, and the composition varies by
product, location, and year.

| Component | What it measures | Typical driver | Varies with |
|---|---|---|---|
| Capital charge | Opportunity cost of the cash invested in stock | Weighted average cost of capital (WACC) or short-term borrowing rate | Firm, business unit, interest rates |
| Storage | Rent, depreciation, utilities for the space occupied | Pallet positions, cubic volume, square metres | Product cube, location, season |
| Handling | Labour and equipment to receive, move, count, and pick | Touches per unit, lines per order | Product handling characteristics |
| Insurance | Premium on the value at risk | Insured value, peril, location | Product value, hazard class |
| Inventory tax | Ad valorem tax on stock held at a date | Assessed value, jurisdiction | Location, assessment date |
| Obsolescence | Expected write-down from age, fashion, or specification change | Shelf life, season length, engineering change rate | Product lifecycle |
| Shrinkage | Theft, loss, misplacement | Location, value density, controls | Product, site |
| Damage | Breakage and deterioration in storage and handling | Fragility, packaging, handling intensity | Product, handling |

The widely quoted 20–25% figure is a convention with a traceable history, not a law. A literature
review at Iowa State traces the 25% rule to a 1944 production handbook, where it was built from
interest, obsolescence, storage, taxes, and handling assumptions specific to that era. The same
review reports published estimates ranging from roughly 8% to 55%, with survey averages nearer
14–16%. A 2022 practitioner survey put most companies in a 20–30% band. The range is not noise.
It is the composition difference showing up.

> **Note:** The practical consequence is not "never use 20%." It is "never use 20% silently."
> Segment the rate by product family and location, document the components, and run the decision at
> half and double the rate. If the decision does not change, the rate does not matter for it. If it
> does change, you have found a parameter that needs evidence.

The sensitivity is larger than intuition suggests because several decisions scale non-linearly.
EOQ scales as the inverse square root of the holding rate. Halving the rate raises EOQ by about
41%. Safety stock set against a cost trade-off moves in the same direction. A holding cost rate
that is wrong by a factor of two is not a small calibration error; it is a different policy.

### 3.3 Cash is a stock, profit is a flow

The balance sheet identity that matters here is:

```text
net working capital (NWC) = inventory + trade receivables - trade payables
cash conversion      = NWC / daily revenue      (days of revenue tied up)
financing cost       = NWC x annual rate        (currency per year)
```

Profit is a flow over a period. Working capital is a stock at a point in time. A company earning
a healthy margin can still run out of cash, because growth requires more stock and more
receivables before it produces more cash. The mechanism is arithmetic, not mismanagement:

```text
each extra EUR 1.00 of revenue requires   NWC/Revenue   of cash
each extra EUR 1.00 of revenue produces   EBIT margin x (1 - tax rate)   of retained cash
if NWC/Revenue > retained margin, growth consumes cash
```

For a distributor with 16% of revenue in working capital and a 4.5% after-tax margin, every euro
of growth needs 16 cents of cash and generates 4.5 cents. The gap is funded by debt, equity, or
supplier terms. This is the arithmetic behind the observation that profitable companies run out
of cash. It is also why a growth plan without a working capital line is not a plan.

The **cash-to-cash curve** is the cumulative cash position of a unit or a cohort over its life.
Cash goes out before the invoice and comes back after it; the curve dips negative, stays negative
for the length of the cash cycle, then recovers. The area between the curve and zero is the
financing cost of that unit. Shortening the cycle narrows the dip; growing the volume deepens it.
A company can improve its cycle and still need more cash if volume grows faster than the cycle
shortens, and both facts belong in the same sentence.

## 4. Definitions and vocabulary

Cost vocabulary is unusually prone to synonym and near-synonym. The table states one definition
per term and names the misuse that does the most damage.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Ordering cost | Cost of placing one order, independent of quantity: buyer time, approvals, inbound admin | Called "setup" in a factory | Currency per order |
| Setup cost | Cost of changing a machine or line over to produce a different item | Ignored when it is mostly labour time | Currency per changeover |
| Holding (carrying) cost | Annual cost of keeping inventory, expressed as a rate or a total | Treated as a single company constant | Percentage of inventory value per year, or currency per unit-year |
| Capital charge | Opportunity cost of the cash tied up in inventory | Set equal to the bank overdraft rate for all decisions | Percentage per year |
| Storage cost | Rent, depreciation, utilities, and space cost of holding stock | Assumed proportional to units rather than to cube | Currency per pallet-day or per square metre-year |
| Handling cost | Labour and equipment to receive, move, count, and pick stock | Counted twice, in holding and again in cost-to-serve | Currency per touch, per line, or per unit |
| Obsolescence | Loss from stock that cannot be sold at cost because demand, fashion, or specification moved | Treated as a one-off write-down rather than an annual rate | Percentage of inventory value per year |
| Shrinkage | Loss from theft, fraud, misplacement, and unrecorded depletion | Conflated with damage | Percentage of inventory value per year |
| Stock-out cost | Total economic cost of not having stock when demand arrives | Reduced to lost margin only | Currency per stock-out event |
| Expedite premium | Extra cost of faster production, freight, or supply to recover a delay | Excluded from the decision that caused it | Currency per expedite |
| Line-haul freight | Movement cost between two nodes on a lane | Compared across lanes without density correction | Currency per shipment, per container, per tonne-km |
| Accessorial charge | Carrier charge beyond line-haul: waiting time, liftgate, reweigh, appointment, detention | Missed at quote stage | Currency per event |
| Demurrage | Charge for a container or cargo remaining inside a terminal beyond free time | Used interchangeably with detention | Currency per container-day |
| Detention | Charge for a container remaining outside a terminal beyond free time | Used interchangeably with demurrage | Currency per container-day |
| Duty | Customs charge on imported goods, ad valorem or specific | Excluded from landed cost because it is "a tax" | Percentage of customs value, or currency per unit |
| Total landed cost | Unit cost once goods are received at a named location, including freight, duty, and fees | Used as a synonym for invoice price plus freight | Currency per unit at a named Incoterm and place |
| Reverse logistics cost | Cost of returns, repair, refurbishment, recycling, and disposal | Treated as a customer service cost, not a chain cost | Currency per return, per unit recovered |
| Cost object | The thing a cost is being attributed to | Left unstated | Unit, order, customer, lane, SKU–location, project |
| Traceable (direct) cost | Cost that can be attributed to the cost object without an allocation rule | Assumed to be all costs that "feel" related | Currency per cost object |
| Allocated (indirect) cost | Cost assigned to a cost object by a chosen driver or rule | Presented as if it were measured | Currency per cost object, plus the driver used |
| Marginal cost | Change in total cost from one additional unit, order, or customer | Used as a synonym for variable cost | Currency per incremental unit |
| Average cost | Total cost divided by total volume | Used for incremental decisions | Currency per unit |
| Sunk cost | Cost already incurred and not recoverable by any future action | Included in forward-looking decisions | Currency, historical |
| Opportunity cost | Value of the best alternative forgone | Omitted because it is not an invoice | Currency per period |
| Inventoriable cost | Cost capitalised into inventory under the applicable standard | Treated as a cash outflow at the same moment | Currency, balance sheet |
| Period cost | Cost expensed in the period incurred | Capitalised to flatter profit | Currency, income statement |
| Contribution margin | Revenue minus variable cost | Confused with gross margin | Percentage of revenue, or currency per unit |
| Days inventory outstanding (DIO) | Average inventory divided by average daily COGS | Computed on revenue | Days |
| Days sales outstanding (DSO) | Average trade receivables divided by average daily revenue | Computed on COGS | Days |
| Days payable outstanding (DPO) | Average trade payables divided by average daily purchases | Computed on COGS when purchases differ materially | Days |
| Operating cycle | DIO plus DSO: the time from paying for stock to collecting from the customer, before supplier credit | Confused with the cash cycle | Days |
| Cash conversion cycle (CCC) | DIO plus DSO minus DPO | Compared across firms with different denominators | Days |
| Net working capital (NWC) | Inventory plus trade receivables minus trade payables | Confused with total working capital including cash | Currency |
| Reverse factoring | Buyer-approved payables finance in which a bank pays the supplier early and the buyer pays the bank at maturity | Called "supply chain finance" without naming the structure | Contract; currency and days |
| Dynamic discounting | Buyer pays a supplier early from its own cash in exchange for a discount that varies with the payment date | Confused with reverse factoring | Percentage discount by day |
| Confirming | Bank confirms or guarantees payment of a trade payable to the supplier | Confused with early payment | Contract; currency |
| Inventory finance | Lending secured against inventory or warehouse receipts | Treated as a form of reverse factoring | Currency, loan-to-value ratio |
| Cost-to-serve | Cost attributable to serving a defined customer, channel, order, or SKU | Reported only as an average rate | Currency per order or per customer per period |
| Activity-based costing (ABC) | Costing that assigns indirect cost through activities and their drivers | Assumed to yield marginal costs | Currency per activity unit |

Four of these deserve expansion.

**Demurrage versus detention.** Both are per-container-day charges, and both are frequently
labelled "port charges." Demurrage applies while the container is inside the terminal beyond
free time. Detention applies while it is outside the terminal — typically at the consignee's
yard — beyond free time. The first is usually caused by documentation, clearance, or terminal
congestion. The second is usually caused by slow unloading, insufficient yard space, or missed
empty-return appointments. They have different owners and different remedies, which is why
lumping them together destroys the analysis.

**DPO on purchases versus DPO on COGS.** Payables arise from purchases, not from COGS. In a pure
reseller, the two are close. In a manufacturer, COGS includes labour and overhead that generate
no trade payable, so COGS is larger than purchases and DPO computed on COGS is understated. If
purchases are 60% of COGS, DPO on COGS understates the true payment period by 40%. The
convention of using COGS survives because it is simple and because the data is easier to get. It
should not survive a decision about payment terms.

## 5. The mechanics, step by step

The sequence builds a cost model from the cost object outward, in the order a decision needs it,
not the order an accounting system produces it.

### Stage A — Define the cost object and the decision

**Step 1. Name the decision the cost model will change.** "Should we add a second weekly order?"
and "what did we spend last year?" need different models. Write the decision first.

**Step 2. Name the cost object.** Unit, order, customer, lane, SKU–location, channel, or business
unit. A lane model cannot see customer behaviour; a customer model cannot see lane economics.
Pick the level at which the decision is made.

**Step 3. Name the horizon.** One unit, one season, one year, or the product life. A warehouse
lease is fixed for a year and variable over a five-year network decision. Most "this customer is
unprofitable" arguments are disagreements about the horizon.

**Step 4. Separate inventoriable from period costs.** Under IAS 2, inventory cost includes
purchase, conversion, and costs to bring goods to location and condition. Inbound freight and duty
are normally inventoriable; abnormal waste, most finished-goods storage, and selling costs are
period. The boundary changes profit timing and DIO.

> **Note:** Absorption costing capitalises fixed production overhead into inventory, so building
> stock defers expense and raises reported profit while consuming cash. Read the cash flow
> statement alongside the income statement.

### Stage B — Build the holding cost rate

**Step 5. Choose the capital charge.** Use pre-tax WACC for a strategic inventory decision and the
marginal short-term borrowing rate for a financing decision. Using the overdraft rate for a
five-year decision understates the cost of equity; using WACC for a one-month decision overstates
it.

**Step 6. Measure storage from space, not units.** Rent per pallet position, or internal cost per
square metre, divided by units per position. A light, bulky item costs more to store per euro of
value than a dense one. This is the largest reason a company-wide rate misallocates across
families.

**Step 7. Measure handling per touch.** Receiving, put-away, replenishment, picking, and cycle
counting have separate drivers. Add them only if the cost object consumes all of them
proportionally.

**Step 8. Add insurance and inventory tax as value-based rates.** Both are quoted as a percentage
of insured or assessed value. Inventory tax is jurisdiction-specific; do not assume it away.

**Step 9. Estimate obsolescence from history.** Use write-down history by product family, adjusted
for shelf life and engineering-change exposure. A single average rate hides the difference between
a nine-month-shelf-life product and a commodity fastener.

**Step 10. Sum, divide, and stress-test.** Divide total components by average inventory value,
check against prior-year write-downs and storage invoices, then run the decision at half and
double the rate. A rate without a date is a rumour.

### Stage C — Compute the working capital days

**Step 11. Use average balances.** A balance-sheet date is a seasonal snapshot. Use opening and
closing averages, or monthly averages, and state which.

**Step 12. Compute DIO on COGS.** `DIO = average inventory / (COGS / days)`. Inventory is carried
at cost. Using revenue inflates the denominator by the gross margin.

**Step 13. Compute DSO on revenue.** `DSO = average trade receivables / (revenue / days)`. If
receivables include value-added tax (VAT) and revenue excludes it, DSO is inflated by roughly the
VAT fraction. Normalise before cross-country comparison.

**Step 14. Compute DPO on purchases.** `DPO = average trade payables / (purchases / days)`.
Purchases, not COGS, creates payables. Report DPO on COGS as a secondary convention and say which
you used. Include only trade payables; payroll, tax, and leases are not supplier credit.

**Step 15. Compute the operating and cash cycles.** Operating cycle is `DIO + DSO`; cash cycle is
`DIO + DSO - DPO`. A negative cash cycle means suppliers and customers together fund the
operation.

**Step 16. Convert days to money carefully.** The exact requirement is
`NWC = inventory + receivables - payables`. Do not multiply a mixed-denominator CCC by daily COGS
and call it exact. Use `NWC / daily revenue` or `NWC / daily COGS` and say which; the two can differ
by 20% or more. Days working capital is `NWC / revenue x 365`, while conventional CCC mixes COGS,
revenue, and purchases, so compare them only when the denominators match.

### Stage D — Price growth

**Step 18. Estimate working capital intensity.** `NWC / revenue` on average balances. For a
seasonal business, use peak-quarter intensity for the funding plan and the annual average for the
P&L.

**Step 19. Multiply by incremental revenue.** `incremental NWC = incremental revenue x intensity`.
This is a plan assumption, not a forecast, and it breaks with a new DC, a new supplier, or a step
change in inventory.

**Step 20. Compare with retained profit.** `retained cash = incremental EBIT x (1 - tax rate)`. If
the requirement exceeds it, state the funding source and its cost.

### Stage E — Recognise what supply chain finance does

**Step 21. Identify the structure.** Reverse factoring: the buyer approves, a bank pays the
supplier early, the buyer pays the bank at maturity. Dynamic discounting: the buyer pays early
from its own cash for a discount. Confirming: a bank guarantees payment without necessarily
paying early. Inventory finance: lending against stock.

**Step 22. Identify who bears the credit risk.** In reverse factoring the bank takes the buyer's
risk, which is usually better than the supplier's. The value is the spread between the two rates
minus the bank fee. If the buyer also extends terms, part or all of that value is transferred.

**Step 23. Identify the accounting treatment.** The question is whether the payable is
derecognised and where the obligation is presented. The IASB's 2023 amendments to IAS 7 and
IFRS 7 require supplier finance disclosure from periods beginning on or after 1 January 2024. US
GAAP's ASU 2022-04 created ASC 405-50 disclosure requirements without changing recognition.
Disclosure is not reclassification, and neither is economic substance.

**Step 24. Compute the network effect.** If the buyer saves 6% and the supplier pays 9% on the
same balance, the network cost rises by 3% while the buyer's metric improves. Put both rates in
the business case.

### Stage F — Build landed cost and cost-to-serve

**Step 25. Build landed cost from the Incoterm outward.** Start at the invoice price at the named
Incoterm and add every cost the buyer bears to the named place: freight, insurance, duty,
brokerage, terminal handling, inland delivery, and expected demurrage or detention. Stop there.

**Step 26. Match allocation bases to cost drivers.** Freight follows weight or volume; duty and
insurance follow value; brokerage follows entries; handling follows units, lines, or touches.
Allocating everything by value is systematically wrong for dense, low-value goods.

**Step 27. Build cost-to-serve from activities.** Define cost pools (order processing, picking,
delivery, returns) and drivers (orders, lines, units, drops, return rate). Assign only costs that
vary with the driver or are caused by the customer, and report residual fixed cost separately.

**Step 28. Compute average and marginal views.** Average cost supports pricing architecture;
marginal cost supports retention and capacity decisions. If the two disagree about a customer,
the disagreement is the finding.

## 6. Worked example

Meridian Components is an industrial distributor. All figures are FY2025 actuals in euro millions
unless labelled otherwise; FY2026 figures are a plan, not a forecast and not a commitment. The
example is internally consistent and small enough to check by hand.

### 6.1 Baseline

| Item | Value | Basis |
|---|---:|---|
| Revenue | 500.0 | FY2025 actual, net of VAT |
| COGS | 400.0 | 80% of revenue |
| Purchases of goods | 340.0 | 85% of COGS; the rest is inbound freight, handling, and conversion |
| Average inventory | 60.0 | Average of four quarters, at cost |
| Average trade receivables | 75.0 | Average of four quarters, excluding VAT for comparability |
| Average trade payables | 56.0 | Average of four quarters, trade only |
| Weighted average cost of capital | 9.0% | Pre-tax, used for the capital charge |
| Marginal short-term borrowing rate | 6.0% | Revolver, used for financing cost |
| Holding cost rate used by planning | 20.0% | Built in section 6.4 |

### 6.2 The three days, with two DPO denominators

```text
daily revenue    = 500.0 / 365 = 1.3699
daily COGS       = 400.0 / 365 = 1.0959
daily purchases  = 340.0 / 365 = 0.9315

DIO = 60.0 / 1.0959 = 54.75 days
DSO = 75.0 / 1.3699 = 54.75 days
DPO on purchases = 56.0 / 0.9315 = 60.12 days
DPO on COGS      = 56.0 / 1.0959 = 51.10 days

operating cycle = 54.75 + 54.75 = 109.50 days
cash cycle (purchases basis) = 109.50 - 60.12 = 49.38 days
cash cycle (COGS basis)      = 109.50 - 51.10 = 58.40 days
```

The two DPO figures are nine days apart, both from the same balance sheet. The purchases basis is
economically correct, because payables arise from purchases. The COGS basis is the common
convention and understates the payment period here, because 15% of COGS is conversion and freight
that generates no trade payable.

> **Note:** Meridian's supplier terms are net 60, and DPO on purchases of 60.1 days is consistent
> with them. DPO on COGS of 51.1 days looks like the company pays ten days early. An analyst using
> the COGS basis would open a payment-terms review to fix a problem that does not exist. The
> denominator chose the conclusion.

### 6.3 The cash number, three ways

```text
NWC = 60.0 + 75.0 - 56.0 = 79.0

as a percentage of revenue   = 79.0 / 500.0 = 15.8%
as days of revenue           = 79.0 / 1.3699 = 57.67 days
as days of COGS              = 79.0 / 1.0959 = 72.09 days
annual financing cost at 6%  = 79.0 x 0.06 = 4.74
```

Three days figures — 49.4 conventional CCC, 57.7 days of revenue, 72.1 days of COGS — describe the
same balance and are not interchangeable. The conventional CCC is a ratio sum with mixed
denominators; days of revenue is the exact number of days of sales the balance represents; days of
COGS answers a different question and is useful only against a cost-based target. Put the
definition in the chart title. The €4.74m financing cost is 0.95% of revenue and 1.19% of COGS —
that is the number for a chief financial officer, because the days figure is abstract.

### 6.4 Building the holding cost rate

Meridian's planning system uses 20%. The build behind that number, on €60.0m of average inventory:

| Component | Rate | Annual cost (€m) | Source |
|---|---:|---:|---|
| Capital charge at 9% WACC | 9.0% | 5.40 | Treasury |
| Storage: 4,800 pallet positions at €450 per position-year | 3.6% | 2.16 | Lease and internal space cost |
| Handling: labour and equipment | 2.0% | 1.20 | Warehouse cost centre |
| Insurance | 0.5% | 0.30 | Policy premium on average stock |
| Inventory tax | 0.3% | 0.18 | Local assessment |
| Obsolescence: three-year write-down average | 3.5% | 2.10 | Finance write-down history |
| Shrinkage | 0.5% | 0.30 | Cycle-count and security records |
| Damage | 0.6% | 0.36 | Claims and scrap records |
| **Total** | **20.0%** | **12.00** | |

At the 25% convention the annual holding cost would be €15.00m, €3.00m higher, and the error would
appear in no report. If the true rate were half — 10% — holding cost would be €6.00m, overstating
the cost of stock by €6.00m. Both errors change order quantities.

The EOQ sensitivity, for an item with annual demand of 120,000 units, an order cost of €450, and
a unit cost of €200:

| Holding rate | Holding cost per unit-year | EOQ (units) | Change vs 20% |
|---:|---:|---:|---:|
| 10% | €20 | 2,324 | +41.4% |
| 20% | €40 | 1,643 | baseline |
| 25% | €50 | 1,470 | −10.5% |

EOQ is `sqrt(2 x annual demand x order cost / holding cost per unit-year)`. The square root is why
the rate matters more than it looks: a 25% error in the rate produces a 12–14% error in order
quantity, and persists in every replenishment until the parameter changes.

### 6.5 Growth: 20% more revenue and the cash bill

The FY2026 plan is €600.0m of revenue, 20% growth, with the same cost ratios and working capital
intensity. This is a plan assumption, not a forecast.

```text
revenue        600.0   (+100.0)
COGS           480.0
purchases      408.0
inventory       72.0   (60.0 x 1.20)
receivables     90.0   (75.0 x 1.20)
payables        67.2   (56.0 x 1.20)

NWC = 72.0 + 90.0 - 67.2 = 94.8
incremental NWC = 94.8 - 79.0 = 15.8
check: 100.0 incremental revenue x 15.8% = 15.8
```

The growth requires €15.8m of additional cash before it produces any. What it produces, at a 6%
EBIT margin and a 25% tax rate:

```text
incremental EBIT          = 100.0 x 0.06 = 6.00
financing cost on new NWC = 15.8 x 0.06  = 0.95
profit before tax         = 5.05
retained profit at 25%    = 3.79

cash absorbed              = 15.8
cash generated             = 3.8
funding gap                = 12.0
payback of the WC investment = 15.8 / 3.79 = 4.2 years
```

The company is profitable on the plan and short of cash by €12.0m. That is the arithmetic behind
"profitable companies run out of cash." The gap is funded by the revolver, equity, supplier terms,
or slower growth. A plan showing the €6.0m of extra EBIT without the €15.8m of extra working
capital is half a plan.

The cash-to-cash curve dips to about −€15.8m in year one and recovers as profit accumulates. If
growth falls in year two, it recovers faster. If growth continues at 20%, it never recovers: each
cohort adds another €15.8m before the prior one pays back. That is why fast-growing distributors
fail while their income statements look healthy.

### 6.6 One term change: DPO from 60 to 75 days

Procurement proposes extending supplier terms from net 60 to net 75, on a €408.0m purchase base.

```text
payables at 75 days = 408.0 x 75 / 365 = 83.84
payables at 60 days = 67.20
cash released       = 83.84 - 67.20 = 16.64

new NWC = 72.0 + 90.0 - 83.84 = 78.16
new conventional CCC = 54.75 + 54.75 - 75.00 = 34.50 days

buyer financing saving at 6%   = 16.64 x 0.06 = 1.00
supplier financing cost at 9%  = 16.64 x 0.09 = 1.50
network cost change            = +0.50
```

The buyer's CCC falls by 15 days and €16.6m of cash is released. The supplier now carries €16.6m of
additional receivables for 15 more days. At 9%, that costs €1.50m a year against the buyer's
€1.00m saving, so the network is €0.50m worse off before any price effect. If the supplier passes
the cost through — over a two-to-three-year contract it usually does — the buyer's saving becomes a
€0.50m loss plus relationship cost.

If the supplier cannot borrow, the cost does not disappear: it appears as deferred maintenance,
lower inventory, slower response, or exit. Those costs are absent from the business case because
they are absent from the buyer's ledger.

The decision rule compares the two rates. If the supplier's cost of funds is genuinely lower than
the buyer's, extension can create value. It rarely is: small suppliers usually borrow at a higher
rate than large buyers, which is why reverse factoring exists and why term stretching usually
destroys value.

## 7. Code, query, or calculation

The calculation below is deliberately small. It returns every days metric with its denominator
named, so that a reader cannot accidentally compare two of them.

```python
def wc_profile(revenue, cogs, purchases, inventory,
               receivables, payables, days=365):
    """All flows and balances in the same currency and period."""
    dr, dc, dp = revenue / days, cogs / days, purchases / days
    dio, dso = inventory / dc, receivables / dr
    dpo_p, dpo_c = payables / dp, payables / dc
    nwc = inventory + receivables - payables
    return {
        "dio_days": round(dio, 1),
        "dso_days": round(dso, 1),
        "dpo_days_purchases": round(dpo_p, 1),
        "dpo_days_cogs": round(dpo_c, 1),
        "operating_cycle_days": round(dio + dso, 1),
        "ccc_days_conventional": round(dio + dso - dpo_p, 1),
        "nwc": round(nwc, 1),
        "nwc_pct_revenue": round(100 * nwc / revenue, 1),
        "cash_days_of_revenue": round(nwc / dr, 1),
    }

wc_profile(500, 400, 340, 60, 75, 56)
```

Expected output: DIO 54.8, DSO 54.8, DPO on purchases 60.1, DPO on COGS 51.1, operating cycle
109.5, conventional CCC 49.4, NWC 79.0, NWC 15.8% of revenue, cash days of revenue 57.7.

Three checks before trusting a variant: switch DIO to revenue and confirm it changes by the gross
margin; confirm payables exclude payroll, tax, and leases; and recompute on period-end balances. If
the last differs by more than five days, the business is seasonal.

## 8. Data, standards, and regulatory context

Cost and cash are governed by accounting standards, payment regulation, and trade data standards
at the same time. The three layers produce different data with different clocks, and the analyst
usually receives them already joined by somebody else.

### 8.1 Inventory valuation and cost recognition

| Instrument | Governs | What it constrains | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | Inventory measurement | Cost includes purchase, conversion, and costs to bring goods to location and condition; measured at the lower of cost and net realisable value | B |
| ASC 330 *Inventory* | US inventory measurement | Similar lower-of-cost-or-market logic with US-specific rules | B |
| IAS 16 / ASC 360 | Property, plant, equipment | Depreciation policy that flows into production cost | B |
| IFRS 15 / ASC 606 | Revenue recognition | When revenue and the related receivable exist, hence DSO | B |
| IAS 7 / ASC 230 | Cash flow statements | Whether a cash movement is operating, investing, or financing | B |
| IFRS 18 | Presentation and disclosure | From 2027, changes how finance income and expense, and some supplier finance liabilities, are presented | B |

The practical constraint for analytics is that "cost" in an ERP is a composite of standards, local
policy, and allocation. Two plants in the same group can capitalise different amounts of freight or
overhead into the same SKU, so a unit-cost comparison across sites compares accounting policies as
much as operations.

### 8.2 Supplier finance disclosure

| Instrument | Scope | Requirement | Effective | Grade |
|---|---|---|---|---|
| IAS 7 amendments (paras 44F–44H) | IFRS reporters | Disclose terms, carrying amounts, amounts already paid to suppliers, and payment due date ranges for supplier finance arrangements | Periods beginning on or after 1 Jan 2024 | B |
| IFRS 7 amendment | IFRS reporters | Supplier finance as a liquidity risk disclosure factor | Same date | B |
| Commission Regulation (EU) 2024/1317 | EU adoption of the IAS 7 and IFRS 7 amendments | Applies the amendments in the EU | Financial years from 1 Jan 2024 | B |
| ASU 2022-04 / ASC 405-50 | US GAAP reporters | Disclose key terms, period-end confirmed obligations, presentation location, and a rollforward; disclosure only, no recognition change | Fiscal years beginning after 15 Dec 2022 | B |
| IFRS Interpretations Committee staff paper, June 2025 | IFRS reporters | Considers how the 2020 reverse factoring agenda decision applies under IFRS 18 | For IFRS 18 application from 2027 | B |

The disclosure regime exists because the previous one did not work; Carillion is the standard
example, discussed in section 9.3. Disclosure changes what you can see in the notes, not what the
arrangement does to the economics. A company can disclose a supplier finance programme fully and
still be using it to present borrowings as trade payables, if the accounting analysis supports
that presentation.

### 8.3 Late payment regulation, 2024–2026

| Instrument or finding | Jurisdiction | Status and substance | Grade |
|---|---|---|---|
| Proposal COM(2023)533, 12 Sep 2023 | EU | Would replace Directive 2011/7/EU with a regulation: 30-calendar-day cap on B2B and G2B payment, automatic interest at ECB rate plus 8 points, €50 flat compensation per invoice, nullity of contrary terms | B for the text |
| European Parliament position, 23 Apr 2024 | EU | Softened the proposal, allowing up to 60 days in some cases and exemptions for slow-moving or seasonal goods | B |
| Council Working Party note WK 3150/2025, 7 Mar 2025 | EU | States that a uniform 30-day limit "has proved to be unacceptable for most Member States"; floats size-differentiated rules | B |
| ICISA update, July 2025 | EU | Reports that a Council compromise was rejected and that the Danish Presidency was not expected to schedule further discussion | D |
| EU Payment Observatory, 2024 data | EU | 47% of businesses report late-payment problems (up from 43%); average B2B payment 62 days, G2B 69 days; 21 of 27 Member States deteriorated in 2023 | B |
| Reporting on Payment Practices and Performance Regulations 2017 | UK | Half-yearly published reports on payment practices for large businesses | B |
| Reporting changes from 1 Jan and 1 Apr 2025 | UK | Add total payments made, percentage of late payments due to dispute, and retention-clause reporting for construction | B |
| Size thresholds from 6 Apr 2025 | UK | £54m turnover, £27m balance sheet, 250 employees (two of three) | B |
| Companies (Directors' Report) (Payment Reporting) Regulations 2025, SI 2025/1152 | UK | Adds payment-practice disclosure to the directors' report for large companies; made 30 Oct 2025, in force 1 Jan 2026 | B |
| Procurement Act 2023 and prompt payment policy | UK | 30-day terms in public contracts from 24 Feb 2025; from 1 Oct 2025, suppliers bidding for central government contracts above £5m per year must average 45 days or fewer and pay 95% within 60 days | B |

> **Note:** The EU proposal is frequently described in 2025 and 2026 commentary as though it were
> settled law. It is not. As of the last public reporting retrieved for this file (October 2025),
> the proposal had not been adopted, the Council process was stalled, and the 30-day cap remained
> the central point of disagreement. Any model that assumes a 30-day EU cap is modelling a
> proposal, not a requirement. Label it as a scenario.

The direction of travel is clear without the regulation: reporting is becoming mandatory, public
procurement is tightening, and large buyers are being asked to evidence payment performance. A DPO
strategy that depends on invisibility has a shortening life.

### 8.4 E-invoicing and DSO

| Finding | Source and date | What it supports | Grade |
|---|---|---|---|
| E-invoicing reduced average time to pay by 1.4 days (5.2%) across 1,720 B2B firms in six countries | Cebr for Avalara, June 2025 | A modest, self-reported DSO effect; country range −2.5 to −0.4 days | D |
| E-invoicing improves payment efficiency but "doesn't necessarily improve payment behaviour" | EU Payment Observatory (CEPS and EY for the European Commission), July 2024 | Processing speed is not payer intent; regulation is still needed | B |
| E-invoices settled 5–7 days earlier than paper invoices | Billentis, cited in the EU Payment Observatory report, 2024 | A wider range than the vendor survey | D |
| In Italy after the e-invoicing mandate, 14–19% of firms reported reduced payment terms and 12–15% reported longer payment times | Cited in the EU Payment Observatory report, 2024 | Mandates do not uniformly shorten DSO | B |
| Widespread SME e-invoicing could cut late payments by about 20% and speed settlement by about 4 days | Sage/ICC expert panel, 2025, 9,000 SMEs | A larger effect, but vendor-sponsored | D |

The honest summary: e-invoicing reliably removes days of *processing* delay, typically a small
number, and does not reliably remove days of *payment* delay. The effect is largest where the
baseline is paper-heavy and smallest where the buyer's payment behaviour binds. A business case
claiming a large DSO reduction from e-invoicing alone claims a behavioural effect from a processing
intervention.

### 8.5 Trade and cost data standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| HS nomenclature (World Customs Organization) | Product classification for duty | The duty rate and the statistical category; errors are costly and retroactive | B |
| WTO Customs Valuation Agreement | Customs value | The basis for ad valorem duty, usually transaction value adjusted to the relevant Incoterm | B |
| Incoterms 2020 | Delivery responsibility | Which party books freight, insurance, duty, and when risk transfers | B |
| GS1 GTIN, SSCC, EPCIS 2.0 | Product, pallet, and event identity | Joining cost to the item, the pallet, and the physical event | B |
| XBRL | Financial reporting | Machine-readable extraction of reported inventory, receivables, and payables | B |

The join problem is the practical one. Freight invoices are keyed by container or bill of lading,
customs entries by declaration number, purchase orders by PO number, and the general ledger by
account and cost centre. None is the same key, so landed-cost analytics is mostly a data-resolution
problem wearing a finance costume.

### 8.6 What is convention rather than requirement

Several parameters that behave like rules are habits:

- **The 20–25% holding cost rate.** A mid-twentieth-century convention, not an accounting or
  regulatory requirement. Section 3.2 gives the range and the history.
- **The 365-day year and net 30.** Both are defaults. A 4-4-5 calendar produces different day
  counts from the same balances, and 90–120 days has been normal in parts of southern Europe.
- **DPO on COGS.** A convention that survives because COGS is easy to obtain, not because it is
  the right denominator.

## 9. Industry practice and case evidence

### 9.1 Working capital benchmarks, 2024–2026

| Source | Sample and period | Headline figures | Grade |
|---|---|---|---|
| PwC Working Capital Study 25/26 | About 17,000 listed companies, 2015–2024 | €1.84tn of excess working capital; net working capital 69.2 days in 2024 against 68.3 in 2015; DSO 50.0 days; DPO up 11.5% since 2015; EU DIO 90.2 days; small firms deteriorated 13.5% and mid-sized 19.8%, large firms flat | D |
| PwC Working Capital Study 24/25 | About 19,000 listed companies | €1.56tn of excess working capital; net working capital days up 9.1 since 2019; DSO up 6.6% over five years; SME DPO down 4.4% from its 2021 peak | D |
| Hackett Group 2025 US survey | Top 1,000 US public nonfinancials, FY2024 | CCC 37 days, improved 4%; DPO 59 days; $1.7tn trapped; DSO and DIO both slightly worse | D |
| Hackett Group 2025 Europe survey | Top 1,000 European nonfinancials | CCC 44.8 days, worse 3%; DPO 72.6 days; DSO 48.5; DIO 68.9, the highest in a decade; €1.4tn excess | D |
| J.P. Morgan Retail Working Capital Index 2025 | Retail subsectors, 2023–2024 | More than $90bn trapped; consumer staples CCC 18.8 days, DSO 21.3, DPO 58.0, DIO 55.6; food retail CCC negative at −3.5 days | D |

These are analyst studies built from published accounts, so the underlying data is audited but the
adjustments and the "excess working capital" estimate are the analyst's. They are useful for order
of magnitude and direction, not for a precise target. The PwC and Hackett samples differ, the
periods differ, and the day counts use different conventions: a 69-day PwC figure and a 44.8-day
Hackett figure are not a contradiction.

### 9.2 The burden-transfer evidence

The most important pattern in the 2024–2026 data is distributional. PwC reports that since 2015,
net working capital days deteriorated 13.5% for small firms and 19.8% for mid-sized firms while
large firms held flat, and that large corporates held DPO above pre-pandemic levels while SME DPO
declined from its 2021 peak — the aggregate signature of large buyers extending terms downwards.

The consequences are documented outside the working capital literature:

- **UK late payment cost.** London Economics, for the Department for Business and Trade and the
  Office of the Small Business Commissioner (July 2025), estimates late payments cost the UK
  economy about £11bn a year, about 0.4% of GDP, with 1.5m businesses affected and £26bn owed at
  any time. Its econometric analysis associates late payment being a "big problem" with a
  roughly 14 percentage point higher business closure rate over 2015–2019, implying about 14,000
  extra closures and 40,000 jobs a year. Grade B.
- **Default probability.** Banque de France Bulletin 227 (2020) finds late customer payments
  raise a firm's probability of default by about 25%, and by about 40% when payments are more
  than a month late. Grade B.
- **Supplier risk feedback.** Esenduran, Gray, and Tan (2021) model extended payment terms as a
  supply chain risk decision and show that stretching terms raises supplier bankruptcy risk and
  can worsen the buyer's own outcomes, particularly when the supplier is already exposed to
  disruption. Grade A.

> **Note:** None of these studies isolates deliberate term stretching by large buyers from
> ordinary late payment. The causal evidence is strongest for late payment and supplier financial
> distress, weaker for the specific act of renegotiating terms. The mechanism is well established;
> the magnitude attributable to term stretching is not.

### 9.3 Carillion: reverse factoring as concealed borrowing

Carillion was a UK construction and facilities-management company that entered liquidation in
January 2018. The public record on its supply chain finance arrangements is unusually detailed.

- Carillion operated an early payment facility from 2013: suppliers could be paid early by a bank,
  at a discount, while Carillion repaid the bank later. Standard subcontractor terms were reported
  at around 120 days.
- Moody's estimated up to £498m was owed to banks under reverse factoring, against reported bank
  loans and overdrafts of about £148m. The obligation sat within "other creditors" rather than
  borrowings.
- The House of Commons Business, Energy and Industrial Strategy and Work and Pensions Committees
  report (HC 769, 16 May 2018) described the facility as borrowing "under the radar" and criticised
  the audit and disclosure regime around it.
- The audit committee's own papers reportedly showed £472m drawn against Moody's £498m estimate;
  the difference matters less than the order of magnitude.

The lesson is not that reverse factoring is fraudulent; it is a legitimate instrument used by many
companies. The lesson is that a payable approved by the buyer, funded by a bank, and paid at a
fixed future date has the economics of borrowing, and can be presented as a trade payable if the
accounting analysis permits it. An analyst reading only the trade payables line sees a supplier
relationship where a lender relationship exists.

### 9.4 Greensill: when the receivable itself is the risk

Greensill Capital was a UK supply chain finance provider that filed for insolvency in March 2021.
The public record from Reuters and Bloomberg establishes the outline:

- Greensill funded itself by selling notes backed by receivables, including future receivables it
  had originated. Credit insurance was central to the model.
- Its main insurer declined to renew cover on $4.1bn of debt, which removed the ability to fund
  and refinance the book.
- Credit Suisse had marketed about $10bn across four supply chain finance funds invested in
  Greensill-issued notes, then froze and wound them down.
- GFG Alliance, the metals group and Greensill's largest client, began defaulting on obligations,
  deepening the crisis. Greensill Bank in Bremen was also a focus of the German regulator.

The analytical lesson is about what is being financed. Reverse factoring on an approved invoice for
delivered goods is a short, self-liquidating exposure to a creditworthy buyer. Financing based on
expected future receivables, or on a concentrated client whose solvency is uncertain, is a
different instrument with a different risk profile, even when sold under the same label.

### 9.5 Cost-to-serve and landed cost in practice

Three findings recur in the practitioner literature:

- **ABC is descriptive, not normative.** Shapiro makes the point precisely: ABC produces average
  unit costs and cannot answer a resource-allocation question. The marginal value of a resource
  comes from the shadow price in an optimisation model. A customer can be expensive on average
  and cheap at the margin, or the reverse.
- **The "flaw of averages" is the default failure.** A single average cost per order hides the
  dispersion that matters. A customer with 40 small orders a year and one with four large orders
  can receive the same average cost and have very different economics.
- **Data fragmentation, not method, is the binding constraint.** Cost-to-serve models fail because
  warehouse, transport, CRM, and finance data do not share keys, and because functions dispute
  activity definitions. The modelling is the easy part.

For landed cost, practitioner sources converge on the same omitted items: brokerage, duty, terminal
handling, demurrage and detention, waiting time, reweighs, address corrections, and delivery
appointment charges. They converge too on the point that landed cost is a per-unit figure requiring
an allocation, and that the basis changes the answer for products of different density and value.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Holding cost rate treated as a universal constant | One rate in the system, no components documented | Rate inherited from a template or a textbook | Order quantities and stock targets systematically wrong for some families | Build the rate from components; segment; stress-test at half and double |
| DIO computed on revenue | DIO falls when prices rise | Revenue includes margin; inventory is at cost | Days understated by the gross margin; target set too low | Use COGS; check the ratio against the margin |
| DPO computed on COGS when purchases differ | DPO looks shorter than the negotiated terms | COGS includes conversion costs with no payable | Payment-terms reviews chase a non-problem | Use purchases; reconcile DPO to terms |
| Period-end balances used in a seasonal business | DIO moves sharply between months with no operational change | Snapshot at a seasonal high or low | Funding plan wrong by a large multiple | Use average balances; model the peak separately |
| CCC compared across business models | Retailer with negative CCC ranked as "best" | Metric mixes denominators and financing structures | Wrong conclusions about efficiency and risk | Compare within archetype; disclose definitions |
| Working capital "improved" by stretching terms | CCC falls while DPO rises sharply and supplier count falls | Burden transferred to suppliers | Supplier distress, price increases, supply risk | Track supplier DSO and concentration alongside own CCC |
| Reverse factoring reported as trade payables with no analysis | Trade payables rise while reported borrowings fall | Payable derecognised or presented as trade | Leverage understated; covenant and liquidity risk hidden | Read the notes; treat confirmed obligations as debt-like in analysis |
| Growth plan with no working capital line | Plan shows revenue, EBIT, and capex only | Working capital is a balance-sheet item | Cash shortfall discovered mid-year | Add incremental NWC and its financing cost to the plan |
| Average cost-to-serve used for a marginal decision | Customer exit list built from fully loaded cost | Fixed cost spread across volume | Profitable customers dropped; capacity cost remains | Compute avoidable cost and marginal cost separately |
| Landed cost excludes duty, demurrage, or brokerage | Landed cost close to invoice price for imported goods | Costs arrive on separate documents after delivery | Sourcing decisions favour the wrong lane or supplier | Build landed cost from customs entries and carrier invoices |
| Stock-out cost counted as lost margin only | Service investment always loses to cost reduction | Expedite, penalty, and goodwill excluded | Under-investment in availability | Include expedite, penalties, and an explicit goodwill estimate |
| Inventory build flatters profit | Earnings rise while operating cash flow falls | Absorption costing defers fixed overhead into stock | Cash strain and later write-down | Read cash flow and inventory days together |
| Sunk or allocated cost included in a forward decision | A discontinued product still "carries" plant overhead | Fixed cost allocated per unit | Wrong keep-or-drop and make-or-buy decisions | Use relevant cost: future, incremental, cash |

Two deserve a note. **Reverse factoring without analysis** matters most for public financial
statements: the obligation may be disclosed, presented within trade payables, and excluded from
reported borrowings, none of which is necessarily an error. Add confirmed supplier finance
obligations to debt-like items, recompute leverage, and state the adjustment. **Average
cost-to-serve used for a marginal decision** is the most common commercial error: a customer who
is unprofitable on a fully loaded basis may still be contribution-positive, and dropping them may
leave the fixed cost behind. The reverse can also hold at peak capacity.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Working capital scales proportionally with revenue | Converts a growth rate into a cash requirement | Step changes in DC capacity, supplier terms, or product mix | Compare NWC intensity across the last five years and across business units |
| Payment terms are stable over the planning horizon | DPO and DSO are treated as parameters | Renegotiation, regulation, or supplier distress | Track realised terms, not contractual terms, by quarter |
| A 365-day year is acceptable | Simplifies day counts | 4-4-5 calendars, 52-week years, fiscal years | Recompute on the actual calendar and compare |
| Average balances represent the period | Days metrics are ratios of stock to flow | Strong seasonality or a year-end clean-up | Compare average, peak, and period-end balances |
| Inventory at cost approximates cash invested | Links the balance sheet to funding need | Vendor financing, consignment, progress payments | Reconcile inventory to cash paid by lane and by Incoterm |
| The WACC is appropriate for the capital charge | Prices the opportunity cost of stock | Different risk by business unit or region | Segment the rate where the businesses differ materially |
| Cost pools are separable and drivers are measurable | ABC and cost-to-serve depend on it | Shared assets, joint products, unmeasured activities | Test whether the driver explains cost variation across sites |
| Cost behaviour is linear in the driver | Simplifies allocation and forecasting | Step costs, economies of scale, minimum charges | Plot cost against the driver and look for kinks |
| Receivables are collectible at face value | DSO is treated as a timing measure | Bad debt, disputes, deductions, credit notes | Compare DSO with days sales outstanding net of provisions |
| Foreign exchange rates are constant in the plan | Avoids compounding two uncertainties | Currency moves in the planning horizon | Model FX as a separate scenario, not inside the cost model |

Two limits are worth stating. **The taxonomy is a management tool, not a physical classification.**
The boundary between holding cost and cost-to-serve is a choice, and two analysts can classify the
same invoice differently and both be consistent. **Working capital describes a financing position,
not a performance outcome.** A company can improve CCC by paying suppliers later, pressing customers
who cannot afford it, or running inventory too low to serve demand — read it beside service and
supplier-health metrics.

## 12. Exercises

### Exercise 1 — Classify the cost lines (recall and application, 30 minutes)

Take these ten costs for a manufacturer: inbound ocean freight; plant changeover labour; warehouse
rent; a customer's chargeback for a late delivery; customs duty; a sales order clerk's salary;
obsolescence write-down; a consumer return; the annual SAP licence; and order-picking labour.

**Deliverable:** a table with columns for cost category, traceable or allocated, marginal or
average for a one-unit decision, and inventoriable or period.
**Acceptance criteria:** all ten classified; the treatment of the SAP licence and the order clerk
justified in one sentence each; at least one line identified where the classification depends on
the cost object.

### Exercise 2 — Build and stress the holding cost rate (calculation, 45 minutes)

A product family has €40.0m of average inventory. The components are: WACC 8.0%; storage €1.40m;
handling €0.70m; insurance €0.20m; inventory tax €0.12m; obsolescence €1.10m; shrinkage €0.18m;
damage €0.16m.

**Deliverable:** the rate to one decimal place, the annual holding cost in euro, a comparison
with a 25% convention, and a statement of what happens to EOQ if the true rate is half the
computed rate.
**Acceptance criteria:** every component shown as a percentage and as a currency amount; the
difference from 25% quantified; the EOQ direction and approximate magnitude stated.

### Exercise 3 — Compute the days with correct denominators (calculation, 40 minutes)

A company reports: revenue €800m, COGS €600m, purchases €420m, average inventory €90m, average
trade receivables €120m, average trade payables €70m.

**Deliverable:** DIO, DSO, DPO on purchases, DPO on COGS, operating cycle, conventional CCC, NWC,
NWC as a percentage of revenue, and cash days of revenue.
**Acceptance criteria:** every denominator stated; the DPO gap explained; cash days interpreted.

### Exercise 4 — Price a growth plan (calculation and judgement, 50 minutes)

Using the Exercise 3 company, revenue grows 25% with constant ratios. The marginal borrowing rate
is 5.5%, the incremental EBIT margin is 7%, and the tax rate is 25%.

**Deliverable:** incremental revenue, incremental NWC, financing cost, incremental retained
profit, the funding gap, and the payback of the working capital investment.
**Acceptance criteria:** proportionality stated; the gap reconciled to a funding source; the break condition named.

### Exercise 5 — Analyse a term extension (calculation and critique, 50 minutes)

The same company, now at the grown revenue level, extends payment terms from 60 to 75 days on
purchases of €525m. The buyer borrows at 5.5%; the supplier borrows at 8.5%.

**Deliverable:** cash released, buyer annual saving, supplier annual cost, network effect, and the
condition under which the extension creates rather than transfers value.
**Acceptance criteria:** all four amounts shown; the network effect signed; the condition stated as a rate comparison.

### Exercise 6 — Name the structure (recall and application, 40 minutes)

For each arrangement, name it, state who bears the credit risk, and state the main accounting
question: (a) the buyer approves an invoice, a bank pays the supplier at day 10, and the buyer
pays the bank at day 90; (b) the buyer pays the supplier at day 10 from its own cash for a 1.5%
discount; (c) a bank guarantees payment at day 90 for a fee but does not pay early; (d) a lender
advances 60% of the value of goods held in a warehouse.

**Deliverable:** a four-row table with structure, risk bearer, accounting question, and the
disclosure that would reveal it.
**Acceptance criteria:** the four structures distinguished; confirming separated from early payment; collateral identified.

### Exercise 7 — Allocate landed cost (calculation, 45 minutes)

A shipment contains 600 units of product A (invoice €100 each, 2 kg each) and 400 units of
product B (invoice €50 each, 5 kg each). Freight is €9,000, insurance €1,200, duty 6.5% of CIF
value, brokerage €450, and drayage €800. Allocate freight by weight, insurance and duty by value,
and brokerage plus drayage by unit count.

**Deliverable:** landed cost per unit for A and B, and the percentage uplift over invoice price
for each.
**Acceptance criteria:** bases stated; totals reconcile to €97,313; the uplift difference explained.

### Exercise 8 — Average versus marginal cost-to-serve (analysis and judgement, 60 minutes)

A customer places 40 orders a year, five lines per order, two units per line, for revenue of
€180,000. Product cost is 75% of revenue. Activity costs are €45 per order, €1.20 per line, €85
per delivery (one per order), returns at 3% of units and €30 per return, and a dedicated account
manager at €18,000 a year. The distribution centre's fixed cost is €2.7m across 4,500 orders.

**Deliverable:** fully loaded profit, avoidable contribution, the allocation used, and a
recommendation with its condition.
**Acceptance criteria:** fixed cost separated; contribution as a percentage of revenue; the recommendation distinguishes constrained from unconstrained capacity.

### Exercise 9 — Critique a working capital claim (critique, 45 minutes)

A company announces: "CCC reduced by 28 days, from 65 to 37; €40m of cash released; DSO 52 to 49;
DIO 61 to 58; DPO 48 to 70; supplier count 1,200 to 900; €25m of payables now funded through a
reverse factoring programme." Revenue is €520m.

**Deliverable:** a one-page critique with the arithmetic check, the decomposition of the
improvement, and three questions for management.
**Acceptance criteria:** €40m reconciled to days and revenue; the DPO share quantified; reverse factoring and supplier count used as evidence.

## 13. Self-check answers

**Exercise 1 model answer.** Inbound freight: transport, traceable, marginal, inventoriable.
Changeover labour: setup, traceable, marginal for batch size, period or inventoriable by policy.
Warehouse rent: holding, allocated, fixed, period. Chargeback: shortage, traceable, marginal, period.
Duty: duty, traceable, marginal, inventoriable. Order clerk: administrative, allocated, fixed,
period. Obsolescence: holding, traceable to family, lifecycle-marginal, period when recognised.
Returned parcel: reverse logistics, traceable, marginal, period. SAP licence: administrative,
allocated, fixed, period. Picking labour: handling, traceable, marginal, period.

**Exercise 2 arithmetic.**

```text
non-capital components = 1.40 + 0.70 + 0.20 + 0.12 + 1.10 + 0.18 + 0.16 = 3.86
non-capital rate       = 3.86 / 40.0 = 9.65%
capital charge         = 8.00%
total rate             = 17.65%, rounded to 17.7%
annual holding cost    = 40.0 x 0.1765 = 7.06
at 25%                 = 40.0 x 0.25   = 10.00, overstating by 2.94
at half the rate       = 8.825%        = 3.53, understating by 3.53
```

EOQ scales as `1 / sqrt(rate)`. Halving the rate multiplies EOQ by `sqrt(2)`, about 1.41, so
order quantities should be about 41% larger. If the company is using 25% against a true 17.7%,
its EOQs are about 16% too small (`1 - sqrt(17.7/25)`), not catastrophic but persistent.

**Exercise 3 arithmetic.**

```text
daily revenue = 800/365 = 2.1918;  daily COGS = 600/365 = 1.6438
daily purchases = 420/365 = 1.1507

DIO = 90/1.6438 = 54.75
DSO = 120/2.1918 = 54.75
DPO purchases = 70/1.1507 = 60.83
DPO COGS      = 70/1.6438 = 42.58

operating cycle = 109.50
CCC = 109.50 - 60.83 = 48.67
NWC = 90 + 120 - 70 = 140.0
NWC/revenue = 17.5%
cash days of revenue = 140/2.1918 = 63.88
```

The two DPO figures differ by 18.3 days because purchases are 70% of COGS; the COGS denominator adds
30% of conversion cost that generates no payable. The 63.9 cash days figure says the company funds
about two months of revenue: €7.7m a year at 5.5%.

**Exercise 4 arithmetic.**

```text
incremental revenue = 200.0
incremental NWC     = 200.0 x 17.5% = 35.0
financing cost      = 35.0 x 5.5% = 1.93

incremental EBIT    = 200.0 x 7% = 14.0
profit before tax   = 14.0 - 1.93 = 12.07
retained profit     = 12.07 x 0.75 = 9.06

funding gap         = 35.0 - 9.06 = 25.94
payback             = 35.0 / 9.06 = 3.9 years
```

The plan is profitable and needs €25.9m of external funding in year one. Proportionality breaks if
growth needs a new distribution centre, a supplier with different terms, or a shift toward
slower-moving stock.

**Exercise 5 arithmetic.**

```text
purchases = 525.0
payables at 60 days = 525 x 60/365 = 86.30
payables at 75 days = 525 x 75/365 = 107.88
cash released       = 21.58

buyer saving at 5.5%   = 21.58 x 0.055 = 1.19
supplier cost at 8.5%  = 21.58 x 0.085 = 1.83
network effect         = -0.65 (a cost)
```

The extension creates value only if the supplier's cost of funds is below the buyer's, or if it
values payment certainty enough to accept a lower effective rate. With these rates it transfers
€0.65m of annual cost into the network; if passed through in price, the buyer's €1.19m saving
becomes a €0.65m loss.

**Exercise 6 model answer.**

(a) Reverse factoring (approved payables finance). Credit risk: the bank takes the buyer's risk.
Accounting question: is the payable derecognised, and is the obligation presented as a trade
payable or as borrowing? Disclosure: IAS 7 paragraphs 44F–44H or ASC 405-50. (b) Dynamic
discounting. Credit risk: the buyer, using its own cash; no bank. Accounting question: none about
derecognition; the payable is settled early and the discount is recorded. Disclosure: usually
none beyond normal purchase discounts. (c) Confirming. Credit risk: the bank, against the buyer.
Accounting question: whether the guarantee is a contingent obligation or a financial liability.
(d) Inventory finance. Credit risk: the lender, against the inventory and the borrower. Accounting
question: whether the arrangement is a secured borrowing and whether the inventory remains on
the balance sheet.

**Exercise 7 arithmetic.**

```text
invoice total = 600 x 100 + 400 x 50 = 80,000
CIF           = 80,000 + 9,000 + 1,200 = 90,200
duty at 6.5%  = 5,863
brokerage + drayage = 1,250
total landed  = 80,000 + 9,000 + 1,200 + 5,863 + 1,250 = 97,313

weight: A 1,200 kg, B 2,000 kg, total 3,200 kg
freight: A 9,000 x 1,200/3,200 = 3,375; B 5,625
insurance: A 900; B 300
duty: A 4,397; B 1,466
brokerage/drayage: A 750; B 500

A landed = 60,000 + 3,375 + 900 + 4,397 + 750 = 69,422; per unit = 115.70
B landed = 20,000 + 5,625 + 300 + 1,466 + 500 = 27,891; per unit = 69.73
uplift: A 15.7%; B 39.5%
```

B is cheaper per unit at invoice but heavy for its value, so it carries a much larger freight share
per euro. A sourcing comparison on invoice price ranks B better than landed cost does.

**Exercise 8 outline.**

```text
revenue 180,000; product cost 135,000; gross margin 45,000
orders 40; lines 200; units 400; returns 12

order processing 40 x 45        = 1,800
picking 200 x 1.20              =   240
delivery 40 x 85                = 3,400
returns 12 x 30                 =   360
account manager                 = 18,000
avoidable total                 = 23,800
avoidable contribution          = 21,200 (11.8% of revenue)

fixed allocation 40 x (2.7m/4,500) = 40 x 600 = 24,000
fully loaded profit             = -2,800 (-1.6% of revenue)
```

The customer is unprofitable fully loaded yet contributes €21,200 toward fixed cost. If capacity is
unconstrained and the fixed cost does not disappear, dropping it reduces profit by €21,200. If the
capacity has a better alternative use, compare contribution per unit of the constraint.

**Exercise 9 rubric.** A strong answer checks 65 − 37 = 28 days and that 28 days at €1.425m of
daily revenue is €39.9m, so €40m reconciles. It decomposes DIO 3, DSO 3, DPO 22 days, so 79% of the
improvement is payables. It notes the 25% fall in supplier count and the €25m of bank-funded
payables, which is borrowing in substance, and asks about supplier prices, lead times, disclosure,
and single-source exposure among the remaining 900 suppliers. A weak answer accepts 28 days as
operational improvement.

## 14. Interview questions

**1. "Why can a profitable company run out of cash?"**

A strong answer separates accrual from timing: revenue is recognised before cash arrives, inventory
is bought before it is sold, and growth increases both. It gives the arithmetic — at 16% NWC
intensity and 4.5% net margin, each euro of growth needs 16 cents and produces 4.5 — and names the
funding sources. A weak answer says "because customers pay late" and stops.

**2. "Where does the 20–25% holding cost rate come from, and when is it wrong?"**

A strong answer names the components (capital, storage, handling, insurance, tax, obsolescence,
shrinkage, damage), notes the 25% figure has a mid-twentieth-century origin and that estimates
range widely, and says it is wrong when the product's profile differs from the company average. A
weak answer treats it as a standard.

**3. "Explain the difference between average and marginal cost with a supply chain example."**

A strong answer uses a customer or an order: the average includes allocated fixed cost, the marginal
is the change in total cost if the order is accepted or the customer leaves. It notes that ABC
produces better averages but not marginals, and that a customer can be unprofitable on average and
contribution-positive at the margin. A weak answer defines the terms with no example.

**4. "A supplier asks to be paid in 90 days instead of 60. What do you analyse?"**

A strong answer quantifies the cash released and the financing saving at the buyer's rate, then the
supplier's cost at the supplier's rate, and compares. It asks about supplier size, concentration,
borrowing ability, and price effects. A weak answer says yes because it improves the CCC.

**5. "Is a negative cash conversion cycle good?"**

A strong answer says it depends on the business model: retailers and subscription businesses run
negative cycles because customers pay before suppliers are paid, which is structurally favourable;
a manufacturer with a negative cycle may be squeezing suppliers unsustainably. Negative CCC is a
financing structure, not proof of efficiency. A weak answer says yes.

**6. "How would you forecast working capital for a three-year plan?"**

A strong answer starts from NWC intensity by business unit, adjusts for known term changes and
capacity step changes, models seasonality separately from the annual average, and states the
funding line. It distinguishes a plan assumption from a forecast. A weak answer applies one
percentage to revenue and stops.

**7. "Explain reverse factoring and its accounting issue."**

A strong answer describes the structure: buyer approves, bank pays the supplier early, buyer pays
the bank at maturity, bank takes the buyer's credit risk. It explains that the payable may be
derecognised or shown as a trade payable, making borrowing look like supplier credit, and cites the
disclosure regimes (IAS 7/IFRS 7 amendments; ASC 405-50). A weak answer describes only the
supplier benefit.

**8. "What do companies most often leave out of landed cost?"**

A strong answer names duty, brokerage, terminal handling, demurrage and detention, accessorials,
and inland delivery, noting these arrive on separate documents after the shipment. It adds that
allocation basis changes the per-unit answer for heavy or low-value goods. A weak answer says
freight.

**9. "Why is cost-to-serve hard?"**

A strong answer gives three reasons: data fragmentation across warehouse, transport, CRM, and
finance systems; disputed activity definitions and drivers; and the gap between average and
marginal cost. The modelling is easier than the data resolution. A weak answer says it requires a
lot of data.

**10. "A company announces it has released €50m from working capital. How do you verify it?"**

A strong answer reconciles the release to days and revenue, decomposes it into DIO, DSO, and DPO,
checks whether payables were stretched or reverse-factored, and asks what happened to service,
supplier health, and prices. It separates a one-off release from a recurring saving. A weak answer
accepts the press release.

**11. "You are pricing a one-off order that uses spare capacity. Which costs do you include?"**

A strong answer includes future incremental cash costs: materials, variable labour, incremental
freight, duty, and expedite or penalty risk. It excludes sunk and allocated fixed cost, and notes
that a constrained resource carries an opportunity cost that belongs in the price. A weak answer
uses fully loaded standard cost.

## 15. Key terms

- **Ordering cost** — fixed cost per replenishment order, independent of quantity.
- **Holding cost** — annual cost of keeping one euro of inventory, as a rate or a total.
- **Obsolescence** — loss from stock that can no longer be sold at cost.
- **Stock-out cost** — lost margin plus expedite, penalty, and goodwill effects.
- **Line-haul** — the main movement between two nodes on a lane.
- **Demurrage** — container charge inside a terminal beyond free time.
- **Detention** — container charge outside a terminal beyond free time.
- **Duty** — customs charge on imported goods.
- **Total landed cost** — unit cost received at a named place under a named Incoterm.
- **Reverse logistics** — cost of returns, repair, recovery, and disposal.
- **Cost object** — the unit, order, customer, lane, or site a cost is attributed to.
- **Traceable cost** — cost attributable without an allocation rule.
- **Allocated cost** — cost assigned by a chosen driver or rule.
- **Marginal cost** — change in total cost from one more unit or order.
- **Average cost** — total cost divided by volume; what financial statements need.
- **DIO / DSO / DPO** — days inventory, sales, and payables outstanding, each with its own denominator.
- **Operating cycle** — DIO plus DSO.
- **Cash conversion cycle** — DIO plus DSO minus DPO.
- **Net working capital** — inventory plus receivables minus payables.
- **Reverse factoring** — buyer-approved payables finance funded by a bank.
- **Dynamic discounting** — early payment from the buyer's own cash for a discount.
- **Confirming** — bank guarantee of payment to a supplier.
- **Cost-to-serve** — cost attributable to a defined customer, channel, or order.
- **Activity-based costing** — costing through activities and their drivers.

## 16. Sources

**Peer-reviewed and academic**

- Esenduran, G., Gray, J. V., and Tan, B. (2021). "A Dynamic Analysis of Supply Chain Risk Management and Extended Payment Terms." *Production and Operations Management*. DOI: 10.1111/poms.13620. Grade A.
- Shapiro, J. F. (1998). "On the Connections among Activity-Based Costing, Mathematical Programming Models for Analyzing Strategic Decisions, and the Resource-Based View of the Firm." MIT working paper, hdl.handle.net/1721.1/2711. Grade A for the argument; not journal-refereed.
- Boţoc, C., and Anton, S. G. (2017). "Is Profitability Driven by Working Capital Management? Evidence for High-Growth Firms from Emerging Europe." *Journal of Business Economics and Management*. DOI: 10.3846/16111699.2017.1402362. Grade A.
- Sawarni, K. S., Narayanasamy, S., and Ayyalusamy, K. (2022). "Working Capital Management, Financial Performance and Growth of Firms: Empirical Evidence from India." *Journal of Indian Business Research*. DOI: 10.1108/JIBR-12-2020-0382. Grade A.
- Harms, R. (2001). "A Comparison of Inventory Carrying Cost in Literature and in Practice." Iowa State University. Grade C: graduate thesis, not peer-reviewed; traces the 25% convention to Bangs and Alford's 1944 *Production Handbook*.

**Regulatory and official**

- European Commission, Proposal on Combating Late Payment in Commercial Transactions, COM(2023)533, 12 September 2023. EUR-Lex CELEX 52023PC0533. Grade B.
- European Parliament, Question P-003939/2025 on the state of COM(2023)0533, 8 October 2025; Council Working Party note WK 3150/2025, 7 March 2025. Grade B.
- European Commission, EU Payment Observatory, 2024 data (CEPS and EY). Grade B.
- UK, Companies (Directors' Report) (Payment Reporting) Regulations 2025, SI 2025/1152, made 30 October 2025, in force 1 January 2026. Grade B.
- UK Department for Business and Trade, "Duty to Report," updated 2 September 2025; UK Government, "Prompt Payment Policy," updated 24 April 2025. Grade B.
- IFRS Foundation, *Supplier Finance Arrangements (Amendments to IAS 7 and IFRS 7)*, 25 May 2023; Commission Regulation (EU) 2024/1317, 15 May 2024; IFRS Interpretations Committee Staff Paper AP5B, June 2025. Grade B.
- FASB, Accounting Standards Update 2022-04, *Supplier Finance Programs (Subtopic 405-50)*. Grade B.
- House of Commons BEIS and Work and Pensions Committees, *Carillion*, HC 769, 16 May 2018; Treasury Committee, *Lessons from Greensill Capital*, July 2021. Grade B.
- London Economics for the Department for Business and Trade and the Office of the Small Business Commissioner, "Estimating the Total Economic Cost of Late Payments," July 2025. Grade B.
- Banque de France, *Bulletin* No. 227 (2020), "Do Late Customer Payments Impact Companies' Probability of Default?"; European Commission Economic Paper 531 (2014). Grade B.

**Standards**

- IAS 2 *Inventories*; ASC 330 *Inventory*; IAS 16; IFRS 15 and ASC 606; IAS 7 and ASC 230; IFRS 18
  (effective 2027); Incoterms 2020, International Chamber of Commerce.
- HS Nomenclature, World Customs Organization; WTO Customs Valuation Agreement; UN/LOCODE, UNECE;
  ISO 4217 currency codes; GS1 General Specifications and EPCIS 2.0.
- Peppol BIS Billing and EN 16931 for European e-invoicing.

**Analyst, vendor, and practitioner**

- PwC, *Working Capital Study 25/26*, 2025, about 17,000 companies; *Working Capital Study 24/25*, 2024, about 19,000 companies. Grade D.
- The Hackett Group, "2025 Working Capital Survey," 18 August 2025; "2025 Europe Working Capital Survey," 10 November 2025. Grade D.
- J.P. Morgan Payments, *Retail Working Capital Index 2025*, September 2025. Grade D.
- Cebr for Avalara, "The Business and Economic Benefits of E-Invoicing," June 2025, 1,720 firms. Grade D: vendor-commissioned.
- ICISA, "Update on Late Payment Regulation," July 2025. Grade D: industry association.
- Institute for Supply Management, "The Monthly Metric: Inventory Carrying Cost," January 2022; IMD, "The Hidden Cost of Cost-to-Serve," 2018. Grade D.
- Deloitte Accounting Research Tool, "7.13 Supplier Finance Programs," current edition; Loop and Lazr landed-cost guides, accessed 2026. Grade D.

**Press**

- Reuters, "Greensill Capital Files for Insolvency," March 2021, and related Reuters and Bloomberg coverage of the Credit Suisse funds. Grade E.
- Wall Street Journal, "Carillion Collapse Highlights Accounting Shortcomings — Moody's," 14 March 2018; CFO.com, "Carillion Collapse Exposes Flaws in Trade Finance Disclosure," 14 March 2018. Grade E.

**Note on citations.** Figures attributed to surveys and analyst studies are reported as those
sources state them and are not independently verified; where a number is uncertain, the text says
so. The EU late payment proposal is described as a proposal, not as law, because that is its status
in the public record retrieved for this file.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_inventory-and-service.md`](05_inventory-and-service.md)
- Next topic: [`07_manufacturing-basics.md`](07_manufacturing-basics.md)
- Related: [`08_transport-and-incoterms.md`](08_transport-and-incoterms.md)
- Related: [`10_archetypes-and-metric-conflicts.md`](10_archetypes-and-metric-conflicts.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Deeper treatment of treasury and supplier finance: [`../../../course/42_treasury_credit_financial_supply_chain.md`](../../../course/42_treasury_credit_financial_supply_chain.md)
- Return to the part map: [`../_index.md`](../_index.md)

