# Multi-Commodity and Multi-Period Models — product families, inventories, and time-expanded networks

> Module 11 — Network Design and Optimization
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 topics 03 and 04
> Version 2026.5

---

## 1. Why this topic exists

A single-commodity, single-period facility-location model is a teaching device, and it is worth
being explicit about what it teaches: that a fixed charge is a step, that capacity couples markets
with nothing in common, and that integrality turns a cost allocation into a design. Those lessons
survive every extension in this module. What does not survive is the setting.

The setting is a network with one product and one year. Almost no decision that matters has
either. A distribution centre (DC) that handles ambient grocery, chilled produce, and frozen
goods has three commodities sharing one dock, one labour pool, one fire-suppression system, and
one lease. A plant that makes three product families has one reactor whose changeover time
depends on the sequence. A network decision is made against a lease that runs seven to ten years,
a build that takes eighteen months, and demand that grows, so the question "which sites in 2030"
is really "which sites in 2027, and which in 2030, and what do you do with the ones you have in
between".

The moment a second commodity and a second period enter the model, three things change
qualitatively rather than merely quantitatively.

**Capacity stops being a site property and becomes a shared resource.** In topic 04, capacity was
a number beside a site, and the only question was whether the site's load fit under it. With
several commodities, the site's capacity is consumed by a weighted sum of flows, and the weights
differ by commodity. A pallet of ambient goods and a pallet of frozen goods are not the same
pallet: the frozen one occupies a position in a freezer, needs a dock with a refrigerated
trailer, and cannot sit on the ambient pick face. The capacity constraint is now the only place
in the model where commodities meet, and it is the only reason the model has to solve them
together.

**Inventory becomes a decision variable, and it is the only thing that connects periods.**
Without inventory, a multi-period model is a set of independent single-period models that happen
to share a file, and it will produce a different design for every period with no transition
between them. With inventory, the model can buy early against a constrained period, and the
question "what does this design cost" becomes "what does this design cost, run over time". That
is a different question, and it is the one the business is asking.

**The model gets large in a way that is easy to underestimate and hard to debug.** A network with
200 lanes, three commodities, and twelve periods has 7,200 flow variables before any inventory,
binary, or capacity term. Add a second commodity dimension by mistake — say, an item-level
breakdown inside a family-level model — and the size grows again without anyone noticing, because
the input file still looks like a spreadsheet.

The industry problem this topic addresses is specific. Single-period studies produce "the network
for 2030", and then someone in the room asks what to do in 2027, and the answer is a second study
that contradicts the first. Multi-period studies that are written carelessly produce plans that
empty the network in the final period, because the objective has no term that values stock left
behind, and the model is doing exactly what it was told. Both failures are common, both are
expensive, and neither is a solver problem.

**What this topic is not.** Module 10 owns inventory theory: safety stock, service levels, and the
policy question of how much to hold against uncertainty. This topic uses inventory as a *model
variable*, a quantity the optimiser chooses because it links periods. Topic 07 owns scenario
analysis, topic 08 owns uncertainty, and topic 10 of this module owns solver limits, so the
discussion of size stops at the arithmetic of variable counts. Module 15 owns production
scheduling; a network model approximates changeovers with a capacity derate or a setup cost, and
that approximation is a limit, not a method.

**The honest claim.** A multi-commodity, multi-period model does not answer a harder question than
the single-period model. It answers the same question with two dimensions the business already
knows are there, and it makes the commodity definition, the period length, and the treatment of
the horizon end documented decisions rather than defaults. Section 11 is honest about where the
model still stops.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** why the single-commodity single-period model is a teaching device rather than a
   template, naming the two dimensions it omits and the three properties that change when they
   are added.
2. **Choose** a commodity grain — product family, stock-keeping unit (SKU), or handling class —
   for a stated decision, **document** the aggregation rule, and **estimate** what the
   aggregation hides.
3. **Build** a commodity-resource matrix with conversion factors, and **demonstrate** that a
   shared capacity row is a coupling constraint across commodities rather than a site property.
4. **Write** the inventory balance equation for any node, commodity, and period, and **diagnose**
   the four standard ways it is written wrong from the resulting plan.
5. **Compare** lead time modelled as a lag on an arc against lead time modelled as a transit
   inventory arc, and **state** why the two disagree at the boundary periods.
6. **Choose** a terminal condition for a stated horizon, **quantify** the end effect it removes,
   and **explain** why the choice biases the early-period decisions.
7. **Distinguish** discounting an operating-cost stream from discounting a cash-flow stream, and
   **identify** the two errors that make a multi-period objective mis-rank designs.
8. **Size** a multi-period multi-commodity instance arithmetically before building it, and
   **report** a study to the six-item standard in section 5.3.

Each outcome maps to an exercise in section 12 and to a self-check answer in section 13.

## 3. The core mental model

Hold one picture: the topic 04 graph, copied once per period and stacked vertically. The copies
are identical in structure — the same sites, the same lanes, the same markets. What connects them
is inventory, drawn as a vertical arc from each node in one copy to the same node in the copy
below it. Nothing else crosses between periods. That single fact — inventory is the only
inter-period link — explains most of the modelling choices in this topic.

```text
   PERIOD t = 1 ------------------------------------------------------------
     SUPPLIER --s1--> PLANT --f1--> DC --d1--> MARKET
                        |            |
                   I(P,1)            I(D,1)
                        |            |
                        v            v
   PERIOD t = 2 ------------------------------------------------------------
     SUPPLIER --s2--> PLANT --f2--> DC --d2--> MARKET
                        |            |
                   I(P,2)            I(D,2)
                        |            |
                        v            v
   PERIOD t = 3 ------------------------------------------------------------
     SUPPLIER --s3--> PLANT --f3--> DC --d3--> MARKET

   horizontal arcs  flow within a period, one set per commodity k
   vertical arcs    I(n,k,t) stock carried from period t to period t+1
   capacity row     sum over k of (flow through n in t) * (resource factor k)
                    <= capacity of n in t
```

Four properties follow, and they carry the rest of the topic.

**Property 1 — Time is a layer, not an axis.** The same physical site appears once per period.
That is a bookkeeping device, not a claim that the site is rebuilt each period. The binary that
says "site j is open" either repeats in every period (and is held together by a constraint that
prevents closing and reopening at will), or is defined once for the whole horizon. Choosing
between those two is a modelling decision about whether the network can change shape inside the
horizon, and it changes the answer more than the period length does.

**Property 2 — Commodity identity survives in the flow and disappears in the capacity row.**
Every flow variable carries a commodity index, so the model knows that 40 thousand units of
ambient are not 40 thousand units of frozen. Every capacity row sums over that index with
conversion factors, so at the constraint the model no longer knows or cares which commodity
consumed the resource. This is the exact point where substitutability is lost: two commodities
that a customer would accept interchangeably are still two variables, and the model will not
swap one for the other unless you give it an arc or a substitution constraint that says it may.

**Property 3 — Inventory is a state variable and the periods decouple without it.** Remove the
vertical arcs and the three stacked copies become three independent facility-location problems.
Each will have its own answer. Nothing forces the site opened in period 1 to be the site used in
period 2, nothing carries stock across a peak, and the total cost is the sum of three optima that
no single plan can achieve. The vertical arcs are what make the model a plan.

**Property 4 — Size grows multiplicatively, and the multiplier is the product of two numbers
people choose casually.** Flow variables scale with lanes times commodities times periods.
Nobody sets out to build a 76,000-variable model. They set out with 200 lanes and then add three
commodities and thirty-six months, and the number arrives on its own.

| Lanes | Commodities K | Periods T | Flow variables | Inventory variables | Binaries (open) | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1 | 1 | 200 | 10 | 10 | 220 |
| 200 | 1 | 12 | 2,400 | 120 | 120 | 2,640 |
| 200 | 3 | 12 | 7,200 | 360 | 120 | 7,680 |
| 200 | 10 | 12 | 24,000 | 1,200 | 120 | 25,320 |
| 200 | 3 | 36 | 21,600 | 1,080 | 360 | 23,040 |
| 200 | 10 | 36 | 72,000 | 3,600 | 360 | 75,960 |

Assumes 10 sites and 20 markets, so 200 lanes; one inventory variable per site, commodity, and
period; one open binary per site per period; no in-transit stock variables. Constraints grow on
the same order as flow plus inventory variables, because the balance equation is written once per
node, commodity, and period.

Read the table as a warning, not a specification. The jump from the first row to the fourth is a
factor of 115, and it happens when a study moves from "a product" to "our product range over three
years". Most of that growth is legitimate; some is an aggregation decision nobody made deliberately.

### 3.1 The commodity dimension is an aggregation decision, not a data fact

There is no such thing as "the product" in a network model. There is a choice about the grain at
which products are carried, and the choice is made whether or not anyone notices. Three grains
are available, and they answer different questions.

| Grain | What it is | Typical count | What it preserves | What it hides |
|---|---|---|---|---|
| Handling class | Ambient, chilled, frozen, hazardous, high-value | 3 to 6 | Shared resources, storage and transport constraints, cost per pallet | Everything about demand and margin within the class |
| Product family | A group of SKUs with similar demand, cost, and handling | 8 to 40 | Demand shape, economics, capacity consumption | Item-level mix, substitution, tail behaviour |
| SKU | The item-location unit at which inventory is managed | Hundreds to tens of thousands | Everything the business can observe | Nothing, but the model becomes unsolvable at network scale |

The professional default in a strategic study is the product family, with handling classes
imposed first because handling is a capacity conversion rather than a cost. The mistake that
matters is to mix grains inside one model: an item-level family alongside a family-level other
family double-counts demand, and the error is invisible in the total because both are "units".

**The aggregation test.** Two SKUs belong in one family if four conditions hold: they share a
handling requirement, their unit costs sit within a stated band, substitution is acceptable to the
customer or the business never substitutes them, and their demand series move together. Fail any
one and the family is hiding a decision the model should see. The third condition is the one most
often skipped and the one that matters most: a family containing two non-substitutable items has
created a commodity that cannot satisfy either item's demand, and disaggregation will discover the
plan is infeasible at item level.

**What aggregation costs.** Rogers and colleagues, in the canonical treatment of aggregation and
disaggregation in optimisation, established that aggregation error is bounded and that the bound
depends on how similar the aggregated items are in their coefficients. The practical reading is
narrower: the bound says the aggregate cost is close, and says nothing about whether the aggregate
*plan* can be disaggregated into an executable item-level plan.

### 3.2 The period dimension is an accounting convention that becomes a constraint

A period is not a natural unit. It is the interval over which the business closes its books, and
that interval is chosen by finance, not by the network. Once it is in the model, it becomes a
constraint on what the model can see. Too long, and peaks disappear: an annual period reports the
network at 71% of the capacity the peak month needs, which section 6 quantifies. Too short, and
the horizon explodes: weekly buckets over five years is 260 periods, and every flow variable is
multiplied by 260. Not aligned with lead times, and lead times round: a 21-day ocean lead time in
a monthly model is one period or two, and the error is systematic over a hundred lanes. Not
aligned with the fiscal calendar, and the balances do not tie: a 4-4-5 retail calendar produces
quarters of 91, 91, and 92 days, and an inventory balance written on calendar months will not
reconcile to the ledger.

The four conventions — period length, horizon, fiscal alignment, and the definition of a period's
stock — must be written down before the model is built. Section 5, Steps 3 and 4, give the
procedure.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Commodity | A distinct flow in the model that cannot be substituted for another flow | Used to mean a raw material only | Index `k` |
| Stock-keeping unit (SKU) | The item-location unit at which inventory is managed and counted | Used as a synonym for product | Identifier |
| Product family | A set of SKUs carried as one commodity in the model | Treated as a natural category rather than a choice | Set of SKUs |
| Handling class | A group of products sharing storage and transport requirements, such as ambient, chilled, frozen, or hazardous | Confused with product family | Code |
| Commodity grain | The level of product detail at which the model carries flow | Left unstated in the model file | Family, SKU, or class |
| Aggregation consistency | The property that an aggregate plan can be disaggregated without violating item-level constraints | Assumed to hold automatically | Condition |
| Substitutability | The property that one commodity can serve demand for another | Assumed present inside a family | Boolean per pair |
| Coupling constraint | A constraint containing variables of more than one commodity, so the commodities cannot be optimised separately | Called simply a capacity constraint | Constraint |
| Shared capacity | A resource whose consumption is summed across commodities | Entered as one number for all products | Units per period |
| Capacity conversion factor | Resource units consumed per unit of a given commodity | Assumed to be 1 for every commodity | Resource units per commodity unit |
| Time-expanded network | A graph in which each physical node appears once per period; the freight literature calls the same object a time-space network | Confused with a project schedule | Graph |
| Period | The time bucket over which flows and stocks are accounted | Left implicit in the objective | Week, month, quarter |
| Horizon | The number of periods modelled | Confused with the planning cycle or the forecast horizon | Count of periods |
| Inventory variable | Stock held at a node at a point in time, chosen by the model | Confused with safety stock, which is a policy | Units |
| Opening inventory | Stock at the start of a period, carried from the previous period | Set to zero without comment | Units |
| Closing inventory | Stock at the end of a period | Confused with the opening of the next period | Units |
| Balance equation | Opening stock plus receipts equals closing stock plus shipments | Written from memory | Equation |
| Flow conservation | The same statement at a transshipment node that holds no stock | Used loosely as a synonym for balance | Equation |
| In-transit inventory | Goods shipped but not yet received at the destination | Counted as available at the destination | Units |
| Pipeline stock | All stock in motion on a lane at a point in time | Confused with the in-transit stock of one shipment | Units |
| Transit arc | An arc whose traversal consumes one or more periods and carries an explicit stock variable | Drawn as instantaneous | Arc |
| Lag | A fixed delay applied to an arc without an explicit stock variable | Confused with a transit arc | Periods |
| Lead time | Elapsed time from release to availability | Used to mean transit time only | Days |
| Terminal inventory | The stock the model is required to hold at the end of the horizon | Left free, so the model empties the network | Units |
| Terminal condition | The constraint that fixes or bounds terminal inventory | Omitted because "the model ends there anyway" | Constraint |
| End effect | Distortion of decisions near the horizon end caused by how the horizon end is treated | Dismissed as a solver artefact | Bias, in units or currency |
| Zero-inventory-ordering property | The lot-sizing result that an optimal uncapacitated plan never carries stock into a period with a setup | Applied to capacitated models where it does not hold | Property |
| Holding cost rate | Cost of holding one unit for one period | Entered as an annual percentage into a monthly model | Currency per unit per period |
| Discount factor | Multiplier converting a future cost or cash flow to present value | Applied to physical units instead of money | Dimensionless |
| Net present value (NPV) | The sum of discounted cash flows over the horizon | Used to describe discounted operating cost | Currency |
| Rolling horizon | Re-solving a fixed-length model every period as the horizon advances | Confused with simply using a long horizon | Procedure |
| Frozen horizon | The initial periods whose decisions are fixed before a re-solve | Omitted, so every re-solve may reverse last month's plan | Count of periods |
| Plan nervousness | The tendency of repeated re-solving to change planned quantities | Diagnosed as forecast error | Measure of change |
| Seasonal capacity | Capacity that varies by period because of staffing, temporary space, or campaign running | Entered as a constant across all periods | Units per period |
| Staffing step | A fixed capacity increment from adding a shift, crew, or line module | Modelled as continuously variable | Units per period per step |
| Lot sizing | Deciding production quantity per period with setup costs and capacity | Confused with network design | Problem class |
| Bill of materials (BOM) | The recipe linking input commodities to output commodities | Ignored in network models, so capacity is mis-stated | Structure |
| Sales and operations planning (S&OP) | The recurring cross-functional planning cycle | Confused with the optimisation model that supports it | Process |

Four of these carry more weight than the rest. **Commodity versus product:** a commodity is
whatever the model cannot substitute, which is a modelling property rather than a physical one.
Two SKUs in the same family are one commodity in the model and two in the warehouse; two SKUs in
different families that a customer will swap are two in the model and one in the market. The
commodity count is therefore always wrong about something, and the question is whether it is
wrong about something that changes the decision.

**Inventory variable versus safety stock:** the inventory variable is a quantity the optimiser
chooses to move stock between periods at least cost; safety stock is a quantity a policy sets to
cover variability the model does not represent. A deterministic model will run a node to zero at
the end of a period if the next period's demand is certain, which is correct under its assumptions
and catastrophic under the business's. **Terminal condition versus end effect:** the condition is
a constraint you write; the end effect is the bias that results from writing it badly or not at
all, and confusing them produces a condition added to make the numbers look sensible with no
statement of what it cost. **Lag versus transit arc:** they agree in the interior of the horizon
and disagree at the boundary, because a transit model can start with goods already on the water
and a lag model cannot unless you give it an opening pipeline. Section 5, Step 7, gives the
comparison in full.

## 5. The mechanics, step by step

This section turns a decision that needs more than one product and more than one year into a model
that can be solved, checked, and reported: twelve steps, then three subsections on what inventory
costs in the model against the building, lot sizing, and the reporting standard.

### Step 1 — Fix the commodity grain

**Input:** the SKU list with handling requirements, unit cost, demand history, and any customer
substitution rules.

**Output:** a set of commodities `K`, a documented mapping from SKUs to commodities, and a
compression ratio — how many SKUs each commodity represents.

**Why it matters:** this is the decision that sets the size of the model and the fidelity of the
answer, and in most studies it is made by whoever built the demand file. Apply the four-condition
test from section 3.1 in this order: split by handling class first, because handling is a capacity
conversion; then group by economic similarity, so every member sits inside a stated unit-cost band;
then test substitutability, because a customer who will not swap two items stops the family pooling
demand; then test co-movement, because two items with opposite seasonality cancel inside one family
and the model sees a flat commodity that exists in neither item.

**Branch:** if the compression ratio exceeds about 50 to 1, the aggregate demand series is not a
description of anything. Split the family, or say the model answers a category-level question.

> **Note:** Sali, Ghrab, and Chatras (*International Journal of Production Economics*, 2023)
> formulate the aggregation scheme for S&OP as a multi-objective programme and describe
> disaggregation as error-prone when families lack supply-relevant features. The family definition
> is a decision with an objective, not a naming convention.

### Step 2 — Build the commodity-resource matrix

**Input:** the process description of each node, the storage and handling equipment, and the
engineering data on throughput.

**Output:** a matrix with commodities as rows and resources as columns, where each cell is the
amount of that resource consumed per unit of that commodity, in stated units.

**Why it matters:** this matrix is where the coupling lives. If every cell in a column is 1, the
model has a single-commodity capacity constraint wearing several commodity labels, and the
commodities will not interact. If the cells differ, the commodities compete, and the competition
is the reason to solve them together.

| Resource at a DC | Ambient | Chilled | Frozen | Hazardous | Unit |
|---|---:|---:|---:|---:|---|
| Dock minutes per pallet | 4 | 6 | 7 | 12 | minutes |
| Storage positions per pallet | 1.0 | 1.0 | 1.4 | 1.8 | positions |
| Pick minutes per order line | 1.2 | 1.5 | 1.6 | 3.0 | minutes |
| Outbound trailer slots | 1.0 | 1.0 | 1.0 | 1.0 | slots |
| Fire-compartment occupancy | 1.0 | 1.0 | 1.0 | 2.0 | units |

Three disciplines make the matrix usable. State the unit of the flow — pallets, cubic metres,
tonnes, or cases — and use the same one in demand, capacity, and the matrix. Derive each factor
from an observed operation, not a nominal allowance. And check that the capacity row you intend to
constrain is the one that binds: dock minutes and pick minutes may bind at different times of year,
and a model with one aggregated "capacity" number has chosen silently which one to believe.

### Step 3 — Choose the period length and the horizon

**Input:** the decision calendar, the lead times on the lanes, the lease and build durations, and
the seasonality of demand.

**Output:** a period length in days, a horizon as a count of periods, and a one-paragraph
justification of each.

**Why it matters:** these two numbers decide whether the model can see the problem. The period
must be short enough to contain the peak you are worried about; if the network constraint is a
four-week harvest or a ten-day promotion, a quarterly model averages it away, and section 6 shows
an annual model reporting 76% utilisation on a network that is at 117% in one quarter. The period
should be at least as long as the planning cadence the model informs, should divide the lead
times, or the lead times must be handled explicitly: a 21-day lead time in a monthly model is 0.7
periods, and rounding it to 1 understates the pipeline on every ocean lane. And the horizon must
cover the longest commitment the decision creates, because a seven-year lease is not covered by a
three-year model.

**Branch:** if the horizon is short relative to the asset life, the terminal condition (Step 8)
carries most of the economic content, and it must be justified rather than set to zero.

### Step 4 — Fix the inventory convention

**Input:** the finance calendar and the warehouse management system's stock definition.

**Output:** one sentence in the model file stating what `I(n,k,t)` means.

**Why it matters:** two legal conventions differ by one period. Under the **closing-stock**
convention, `I(n,k,t)` is the stock at the end of `t`, which is the opening stock of `t+1`. Under
the **opening-stock** convention it is the stock at the start of `t`, the closing stock of `t−1`.
Both are correct; mixing them is the fourth balance error in Step 5, and the hardest to see.
A plan reported as "ending inventory" against an opening-stock model is off by one period
everywhere, and the error gets attributed to the data.

### Step 5 — Write the balance equation for every node, commodity, and period

**Input:** the network, the commodity set, and the inventory convention.

**Output:** one equation per node, commodity, and period, plus one for the opening state.

The general form, for node `n`, commodity `k`, period `t`:

```text
I(n,k,t-1)  +  sum over inbound arcs a of x(a,k,t - lag(a))  +  p(n,k,t)
            =  I(n,k,t)  +  sum over outbound arcs a of x(a,k,t)  +  w(n,k,t)

I(n,k,t-1)   stock carried in from the previous period     units
x(a,k,t)     flow on arc a in period t                      units
p(n,k,t)     production or supply at n in period t          units
w(n,k,t)     withdrawal to demand at n in period t          units
lag(a)       transit lag on arc a                           periods
```

At a node with no storage, drop `I` and the equation becomes flow conservation. At a market node,
`w` is demand and is fixed. At a source node, `p` is supply and is bounded. Everything else is
bookkeeping.

**The four ways it is usually written wrong.** Each produces a plan that solves and a number that
is wrong, and each has a signature you can see in the output.

| Error | What it looks like in algebra | Signature in the plan |
|---|---|---|
| Sign error | `I(t-1) + r(t) + s(t) = I(t)` | Inventory grows by shipments; the plan can serve the same unit twice |
| Opening stock dropped | `r(t) = I(t) + s(t)` | Every period is self-sufficient; peaks are infeasible, or carry-over silently vanishes |
| In-transit treated as available | `r(t)` includes goods shipped in `t` with a two-period lead | The plan is short by exactly the pipeline, and the shortfall appears as a stock-out later |
| Period-end and period-start mixed | `I` defined as closing in the balance and as opening in the terminal condition | Stock is double-counted at the boundary, or one period of stock disappears |

The sign error is the most common in hand-built spreadsheets, because the natural way to write
"in plus production minus shipments equals out" is to put every term on one side, and its
signature is an inventory series that rises with throughput even when nothing is stored. The
opening-stock error is the most common in generated code, because the loop starts at `t = 1` and
the first equation has no predecessor. The in-transit error is the most common in practice,
because transport data is a shipment record and warehouse data is a receipt record, and the model
is usually built against the wrong one.

**A check that costs nothing.** After solving, recompute the balance residual for every node,
commodity, and period from the reported values, independently of the solver. The residual should
be zero to within the solver's feasibility tolerance. A residual of exactly one period's flow at
one node usually means the lag is off by one; a residual that grows along the horizon usually
means a sign error; a residual that is zero everywhere but the plan is infeasible in reality
means the balance is right and the capacity row is wrong.

### Step 6 — Decide where inventory is allowed to live

**Input:** the node types and their physical storage. **Output:** an availability map `allow(n,k)`
and, where storage is finite, a storage capacity per node.

**Why it matters:** the model will put stock anywhere the equation permits. A cross-dock has no
storage, a cold room has positions and a temperature class, a hazardous store has a limit set by
regulation. If chilled stock may sit at an ambient site, the model will put it there because it is
cheaper and the plan will be unimplementable. Storage capacity and throughput capacity are also
different constraints — what a node can hold at an instant against what it can move per period —
and a model with one "capacity" number cannot tell them apart.

### Step 7 — Model the lead times

**Input:** the transit time and its variance for each lane, and the period length from Step 3.

**Output:** either a lag on each arc, or an explicit transit inventory arc, chosen lane by lane
and documented.

**Why it matters:** the two treatments give different answers at the boundary periods, and the
boundary is where the study's first-year numbers live.

| Aspect | Lag on the arc | Transit inventory arc |
|---|---|---|
| Representation | Flow in `t` is subtracted in `t` and added in `t + L` | Flow enters a stock on lane `a` in `t`, leaves it in `t + L` |
| Variables | None beyond the flow | Adds one stock per lane, commodity, and period |
| Cost | Holding cost applied to the delayed quantity | Holding cost applied to the stock, which is visible |
| Opening state | Starts empty unless an opening pipeline is supplied | Can start with goods already moving, from data |
| Boundary behaviour | Understates availability in the first `L` periods | Represents the pipeline that exists on day one |
| Size cost | Cheap | Multiplies lanes by commodities by `L` |
| Best use | Tactical models on lanes with stable transit | Strategic models where working capital in transit matters, or transit varies |

The boundary difference is not a rounding detail. Suppose the study starts in January and 40% of
one ocean lane's annual volume is already on the water on 1 January. A lag model with no opening
pipeline sees an empty lane and will either build inventory it does not need or declare the first
quarter infeasible. A transit model with the opening pipeline from shipment data sees the goods
and plans around them. The two models have the same equations in the interior and different
answers in the first quarter, which is the quarter the business is about to execute.

**Rounding.** If the lead time is not an integer multiple of the period, you have three options:
shorten the period, use a lag of `floor(L)` and accept the understatement, or use a fractional lag
with a flow split. The third is exact in expectation and obscures the fact that a shipment
arrives on a day, not a fraction of a month. On a 21-day lane in a monthly model, a lag of one
understates the pipeline by 30% and a lag of two overstates it by 43%. State the choice, and do
not make it silently a hundred times.

### Step 8 — Choose the terminal condition

**Input:** the horizon, the demand beyond it, and the asset life.

**Output:** one terminal constraint per node and commodity, with a written justification.

**Why it matters:** the model has no reason to leave stock behind. Stock at the end of the
horizon is cost without benefit inside the model, so a model with free terminal inventory will
sell everything, including the safety stock, in the last period. That is not a solver defect; it
is the objective working correctly on a badly posed problem.

| Terminal condition | Statement | What it assumes | When it is right |
|---|---|---|---|
| Free, zero value | No constraint | The business ends at the horizon | Almost never |
| Cyclic | `I(T) = I(0)` per node and commodity | The horizon end looks like its start | Steady-state networks with flat demand |
| Continuity floor | `I(T) >= d(T+1)` per node and commodity | One period of cover is the minimum sensible state | Most strategic studies |
| Valued terminal stock | Add a term `V(I(T))` to the objective | Terminal stock has a salvage or replacement value | When the horizon is short relative to the asset life |

Fisher, Ramdas, and Zheng, in *Management Science* in 2001, named the problem and the fix for the
lot-sizing case: short horizons produce end effects, the most common being the requirement that
ending inventory be zero, and a terminal valuation term derived from economic order quantity
logic outperformed both Wagner-Whitin and Silver-Meal in their rolling-horizon tests. The lesson
transfers to network models with one change: the valuation should be the cost of rebuilding that
stock in period `T+1`, not an average holding value, because the network will in fact rebuild it.

**The test that matters.** Solve with two terminal conditions, the one you chose and the one you
rejected. If the open set changes, the terminal condition is deciding the design and the study
must say so. If only the last period's flows change, the condition is doing its job. Section 6
shows a case where the condition moves total cost by 6.4% and leaves the open set alone, which is
the benign outcome and still has to be reported.

### Step 9 — Model seasonal capacity

**Input:** the staffing plan, the shift pattern, the temporary space options, and the campaign
structure of the plant.

**Output:** either a capacity parameter that varies by period, or a capacity decision with a cost,
and a statement of which.

**Why it matters:** a network model with constant capacity and flat annual demand will recommend
the smallest footprint that covers the average, and the peak will be served by overtime that the
model never sees. Three levels of treatment are available and they give different answers.

| Treatment | Model element | What the model can do | What it assumes |
|---|---|---|---|
| Parameter | `cap(j,t)` given per period | Nothing; capacity is data | Staffing is fixed by policy |
| Linear decision | `cap(j,t)` variable, cost per unit per period | Staff up smoothly for the peak | Agency labour is available at a linear price |
| Step decision | `cap(j,t)` = base plus `m` steps, step cost and step size | Add a shift or a temporary module | Steps are the real unit of expansion |

The linear treatment is the most dangerous because it is the most flexible, and therefore the most
attractive to the optimiser: it rents exactly the peak week's capacity and releases it, at a price
no labour market offers. If staffing is genuinely flexible, use a step cost and a minimum
engagement. If not, make capacity a parameter and let the model carry inventory instead, which is
what the business does. Campaigns at a plant follow the same logic: a campaign is a step. Module
15 owns the scheduling machinery; a network model represents a campaign as a derate or a setup
cost, and the derate is a judgement.

### Step 10 — Write the objective, and decide what the discount rate means

**Input:** the cost coefficients, the capital charges, the price year, and the discount rate.

**Output:** one objective function with every term in the same currency, price year, and time
basis.

**Why it matters:** a multi-period objective invites two specific errors, and both change the
ranking of designs rather than the total. The first is mixing real and nominal: constant 2026
currency discounted at a nominal rate that includes inflation double-counts the time value of
money. Nominal with nominal, real with real, never real at nominal.

The second is discounting operating cost as though it were a cash flow. That is defensible for a
cost incurred smoothly. It is wrong for a facility's fixed charge, a lumpy outflow when the lease
is signed, and for inventory, a working capital commitment recovered when the stock is sold. A
model that discounts both as annual annuities has turned two financial objects into one shape, and
the error is largest for the design with the most capital and the least flow cost, which is
usually the one being recommended.

**The practical rule.** Discount money, not quantities. Keep the physical plan undiscounted so the
flows can be checked against a balance sheet, and discount the cost lines separately. If the
decision is "which network", discount the cost; if it is "how much capital, and when", build the
cash-flow profile and discount that. Al-Ashhab and Afia, in a 2017 closed-loop study, report that
moving the discount rate from 1% to 10% left the network structure unchanged while changing the
quantities shipped. That is a calibration from their instance, not a law.

### Step 11 — Size the model and choose a formulation

**Input:** the sets, the period length, the horizon, and the solver available.

**Output:** a variable and constraint count, a formulation choice, and a time limit.

**Why it matters:** the count tells you whether you have a problem you can solve to a useful gap,
and the formulation choice can matter more than the count. Two formulations of the same model can
differ by orders of magnitude in solve time.

Compute the counts before building anything:

```text
flow variables      = |lanes| x |K| x |T|
inventory variables = |nodes| x |K| x |T|          (plus transit arcs if modelled)
binary variables    = |sites| x |T|                (open) + modules if capacity is modular
balance rows        = |nodes| x |K| x |T|
capacity rows       = |resources| x |T|
```

A **compact** formulation indexes flow by period and lets the balance carry the lag; a
**time-expanded** formulation makes the period a node and lets flow move along time arcs. They
describe the same feasible set with different relaxations. Meisel, Rei, Gendreau, and Bierwirth, in
a 2011 working paper on supply network design under lead-time constraints, report that
reformulating as a fixed-charge flow on a time-space network removed the explicit lead-time
constraints and solved their first test set in under a second where the lead-time formulation took
about eight hours, with average gaps of 0.5% against 2.6%. The instance is theirs; the lesson is
that the formulation is a decision, not a formality. Topic 10 owns solver limits; here the point is
that this model reaches the limit sooner, and the first response is to examine the commodity grain
and the period length, not to buy a bigger machine.

### Step 12 — Solve, then validate four things

**Input:** the solved model.

**Output:** a plan, a gap, and four checks with pass or fail.

1. **Balance residual.** Recompute every node-commodity-period balance from the reported flows
   and stocks. It should be zero within tolerance; a non-zero residual is a modelling error, not
   a numerical one.
2. **Capacity utilisation by resource and period.** List every resource above 95% in any period.
   Those are the weeks the design is fragile, and they are what the committee will ask about.
3. **Terminal state.** Report terminal inventory against the condition, per node and commodity. A
   model that exactly meets a lower bound everywhere was constrained to it, and the cost belongs
   in the report.
4. **The single-period comparison.** Fix the model to one period at a time and solve. If the
   multi-period plan differs materially, the difference is the value of the time dimension and
   should be stated; if it does not, the extra dimension bought nothing.

### 5.1 What inventory costs in the model and what it costs in the building

The model charges a holding cost per unit per period. The building charges something else, and
the difference is a systematic bias rather than a rounding error.

| Cost | In the model | In the warehouse | Why they differ |
|---|---|---|---|
| Capital | Rate times unit value times periods held | Weighted average cost of capital on the balance-sheet value | The model uses one rate for all commodities |
| Space | Usually folded into the holding rate | A step: a rack bay, a freezer position, a licensed area | Space is bought in blocks, not per unit |
| Handling | Not charged to holding | Every move in and out costs labour | The model treats storage as static |
| Obsolescence | Rarely modelled | Real, and concentrated in short-life items | Requires shelf-life data the model does not have |
| Quality | Not modelled | Stock ages; the oldest ships first | The model's stock is fungible across time |

Two consequences follow. First, a model that folds space into a linear holding rate will
under-value pre-building when the real constraint is positions, because the marginal cost of the
last pallet in a full rack is much higher than the average. Second, the model's inventory is
perfectly fungible across periods, so it will hold the cheapest unit and ship it whenever, which
is exactly what a shelf-life constraint forbids. If shelf life matters, add a maximum dwell time
per commodity and accept that this makes the inventory variable age-indexed.

> **Note:** none of this argues for a more elaborate holding cost. It argues for stating which
> costs the rate contains, and for not using the model's inventory level as a forecast of the
> warehouse's space requirement.

### 5.2 The relationship to lot sizing, which is where this model came from

The balance equation in Step 5 is the central object of a literature older than network design:
dynamic lot sizing, from Wagner and Whitin in 1958. The single-item uncapacitated problem has the
zero-inventory-ordering property — an optimal plan never carries stock into a period with a setup
— which makes it solvable by dynamic programming, and is the reason the end-effect problem
exists. If optimal plans end periods at zero, a model that ends at zero looks optimal and is
truncated.

Add capacity and the property fails. Charles, Dauzère-Pérès, Kedad-Sidhoum, and Mazhoud (*European
Journal of Operational Research*, 2022) show that with setup times and capacity, forcing zero
ending inventory distorts early decisions even on long horizons, and that a global minimum ending
inventory alongside per-item maxima cut lost sales from about 6.5% to 0.8% in their rolling-horizon
experiments. The network model here is formally a multi-commodity capacitated lot-sizing problem
with opening costs and a distribution graph, which tells you which literature to read when it
misbehaves: the elegant properties of the single-item problem are exactly the ones that fail once
capacity is shared.

### 5.3 The reporting standard for a multi-period multi-commodity study

Six items, and a study that omits any of them cannot be reviewed. This is the topic 04 standard
with two items replaced by the ones that only exist when there is more than one product and more
than one period.

| # | Item | What must be stated | Why a reviewer asks |
|---:|---|---|---|
| 1 | Commodity definition | The grain, the mapping rule, the count of commodities, the compression ratio from SKUs, and what the aggregation hides | Because the answer lives at this grain and nowhere else |
| 2 | Period length and horizon | Length in days, count of periods, fiscal alignment, and why the length can see the peak | Because a model that cannot see the peak cannot size the network |
| 3 | Terminal condition | The constraint, its justification, the cost of the alternative, and whether the open set changes | Because the end effect is a bias, and an unreported bias is a wrong answer |
| 4 | Lead-time treatment | Lag or transit arc, per lane class; rounding rule; opening pipeline treatment | Because the two treatments disagree exactly where the study's first year is |
| 5 | Model size and gap | Variables, constraints, solver, time limit, gap in percent and in currency | Because "optimal" without a bound is a claim rather than a certificate |
| 6 | Baseline validation | The model run on the current network and period, reconciled to the ledger by period | Because a model that cannot reproduce last year cannot be trusted about next year |

The commodity definition and the terminal condition are the two items a reviewer should ask about
first: they change the answer most and are most often left to whoever built the input file.

## 6. Worked example

Meridian Health Supplies runs two distribution centres, North and South, serving two regional
markets. It carries two commodities: ambient (A) and cold-chain (C). The planning question is
whether the receiving capacity at North is sufficient for the next three quarters, given demand
growth into a fourth-quarter peak. All figures are illustrative and internally consistent.
Quantities are in thousand units and money is in thousand currency units, so a cost of 8.00 per
thousand units is written 8.00 throughout.

### 6.1 The instance

| Demand, thousand units | Q1 | Q2 | Q3 | Total |
|---|---:|---:|---:|---:|
| North, ambient A | 20 | 25 | 40 | 85 |
| North, cold C | 5 | 6 | 15 | 26 |
| South, ambient A | 15 | 18 | 30 | 63 |
| South, cold C | 8 | 10 | 25 | 43 |

| Node | Resource | Capacity per quarter | Conversion |
|---|---|---:|---|
| North | Receiving dock and put-away | 60 | A = 1.0, C = 2.0 pallet-equivalents per unit |
| South | Receiving dock and put-away | 80 | A = 1.0, C = 2.0 pallet-equivalents per unit |

| Cost | North | South |
|---|---:|---:|
| Supply, ambient, per unit | 8.00 | 7.60 |
| Supply, cold, per unit | 11.00 | 10.40 |
| Receiving and put-away, per pallet-equivalent | 0.50 | 0.50 |
| Holding, ambient, per unit per quarter | 0.40 | 0.40 |
| Holding, cold, per unit per quarter | 1.20 | 1.20 |

Capacity is receiving throughput, not storage. Storage is assumed ample at both sites, which is
itself a modelling choice: in reality a pallet held is a position occupied, and section 5.1
explains what that omission costs. Inter-DC transfer is available from South to North for ambient
only, at 2.00 per unit, and it consumes one pallet-equivalent of North receiving capacity on
arrival. Cold cannot transfer, because there is no refrigerated lane.

The terminal condition requires 15 thousand ambient units at North at the end of Q3, representing
the stock the business must hold to open Q4. Cold ends at zero, because cold stock is made to
order and has a short life.

### 6.2 The annual model, and what it cannot see

Collapse the three quarters into one period. North receives 85 ambient units, which is 85
pallet-equivalents, plus 26 cold units, which is 52, for a total of 137 against a capacity of 180.
Utilisation is 76%. South receives 63 plus 86, which is 149 against 240, or 62%. Both sites are
comfortably within capacity. The annual model reports that the network has spare room and no
action is required.

Now disaggregate. North's quarterly requirement is 30, 37, and 70 pallet-equivalents, so the third
quarter needs 70 against a capacity of 60. The annual model could not see it because the peak is
averaged against two slack quarters. The ratio of peak to average is 70 over 45.7, or 1.53, an
ordinary seasonal profile. South is tighter than it looks for the same reason: its requirements
are 31, 38, and 80, with the third quarter exactly at capacity, so one more unit of cold demand
would make the plan infeasible and one more unit of ambient would have to be pre-received.

### 6.3 The multi-period plan

The model has to cover North's Q3 deficit of 10 pallet-equivalents. Three routes exist: carry
ambient stock from Q2, transfer ambient from South in Q3, or transfer in Q2 and hold. The
cheapest is to build in Q2, because holding costs 0.40 per unit per quarter against a transfer
cost of 2.00 per unit.

Building 25 units in Q2 does two things at once: it covers the 10-unit Q3 gap and it creates the
15-unit terminal stock. North's Q2 receipts become 25 units of demand plus 25 units of build,
which is 50 ambient units, or 50 pallet-equivalents, plus 12 for cold, for 62. That exceeds the
capacity of 60. The intuitive plan is infeasible, and the reason is the coupling between the
commodities: the cold units take twice the dock resource, so 6 cold units consume the room that
12 ambient units would have used.

The repair is to move two ambient units from Q2 to Q1. North's Q1 receipts become 22 ambient plus
5 cold: 22 plus 10, or 32 pallet-equivalents, comfortably inside 60. Q2 becomes 48 plus 12, or 60
exactly. Q3 becomes 30 plus 30, or 60 exactly. The plan now holds 2 units at the end of Q1, 25 at
the end of Q2, and 15 at the end of Q3.

| Period | Receipts A | Receipts C | Pallet-eq | Capacity | Closing A | Closing C |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 22 | 5 | 32 | 60 | 2 | 0 |
| Q2 | 48 | 6 | 60 | 60 | 25 | 0 |
| Q3 | 30 | 15 | 60 | 60 | 15 | 0 |

The cost of the plan:

```text
North ambient supply    100 units x 8.00           =   800.00
North cold supply        26 units x 11.00           =   286.00
South ambient supply     63 units x 7.60            =   478.80
South cold supply        43 units x 10.40           =   447.20
North receiving          (32 + 60 + 60) x 0.50      =    76.00
South receiving          (31 + 38 + 80) x 0.50      =    74.50
North holding            (2 + 25 + 15) x 0.40       =    16.80
South holding            0                          =     0.00
Total                                                = 2,179.30
```

Both sites run at exactly 60 and 80 in Q3. The design has no slack in the quarter that matters,
and the objective does not say so.

### 6.4 The coupling across commodities

The interesting property of this instance is that the capacity row is the only place the two
commodities meet, and the meeting is where the plan is decided. Take South in Q3: 30 ambient units
are 30 pallet-equivalents, and 25 cold units are 50, for 80 against a capacity of 80. Now raise
South's Q3 cold demand by two units, a 8% increase in that line.

A cold-only model sees 27 units against an implied capacity of 80 pallet-equivalents, which is 54,
or 68% utilisation. It reports ample room. A multi-commodity model sees 30 plus 54, which is 84
against 80. It is infeasible at the current plan and requires four pallet-equivalents of ambient
receipts to move from Q3 to Q2, where they are held for one quarter at 0.40 each, for an
additional 1.60. The design does not change. The plan does, and the cost changes by an amount
that is small in currency and decisive in feasibility.

This is the general shape of the coupling result: a shared resource converts an apparently
comfortable single-commodity constraint into a binding multi-commodity one, and the increase in
utilisation is not proportional to the increase in demand. It is proportional to the increase
measured in resource units, which is why the conversion factors have to be real.

### 6.5 The terminal condition, and what it is worth

Remove the 15-unit terminal requirement and re-solve. The model now buys 85 ambient units at North
instead of 100, holds 10 units from Q2 into Q3 to cover the 10-unit capacity gap, and ends with
zero stock.

| Plan | Ambient purchased | North holding | Total cost | Difference |
|---|---:|---:|---:|---:|
| Terminal 15 | 100 | 42 unit-periods | 2,179.30 | — |
| Terminal 0 | 85 | 10 unit-periods | 2,039.00 | −140.30, or −6.4% |

The zero-terminal plan is 6.4% cheaper and is not a cheaper plan. It is a plan that sold 15
thousand units of stock that the business needs on the first day of Q4, and the cost of replacing
those units falls outside the horizon. The end effect here is not a rounding error at the margin;
it is a sixth of the annual distribution cost, and it appears as an improvement.

Two things make this benign rather than fatal. The open set does not change, so the design
recommendation survives. And the size of the bias is computable, because it is exactly the cost
of the stock the model refused to hold: 15 units at a supply cost of 8.00 is 120.00, plus 7.50 of
extra receiving and handling, plus 12.80 of extra holding, gives 140.30. In a study where the
open set does change with
the terminal condition, the design is being decided by a constraint nobody argued for, and that
is the outcome to test for.

A model does not settle the questions that decided this plan: whether the receiving capacity of 60
is real in a quarter that also handles returns, and whether 15 units of terminal cover matches what
Q4 opens with. Those are input disputes, and they belong in the report beside the cost.

## 7. Code, query, or calculation

One node, two commodities, three periods, and a proposed plan that has to be tested against the
balance equations before it is tested against anything else. This is the check to do by hand on
every study, on at least one node, before the solver output is believed.

```text
Node N, commodities A and C, periods t = 1,2,3.
  I(k,t) stock at end of t     r(k,t) receipts     s(k,t) shipments to demand
  (1)  I(k,t-1) + r(k,t) = I(k,t) + s(k,t)          for k in {A,C}, t = 1..3
  (2)  s(A) = (20,25,40)   s(C) = (5,6,15)          demand, fixed
  (3)  r(A,t) + 2*r(C,t) <= 60                     receiving capacity
  (4)  I(A,0) = 0, I(C,0) = 0, I(A,3) = 15, I(C,3) = 0

Proposed plan P:  r(A) = (20,48,30)   I(A) = (0,23,15)   r(C) = (5,6,15)
  t=1  A: 0 + 20  = 0  + 20   ok        C: 0 + 5  = 0  + 5   ok
  t=2  A: 0 + 48  = 23 + 25   ok        C: 0 + 6  = 0  + 6   ok
  t=3  A: 23 + 30 = 15 + 40   FAILS, 53 against 55
  t=3  C: 0 + 15  = 0  + 15   ok
  capacity t=1: 20 + 2*5  = 30 <= 60   ok
  capacity t=2: 48 + 2*6  = 60 <= 60   ok
  capacity t=3: 30 + 2*15 = 60 <= 60   ok

Residual is 2 units short in period 3 only, because the shortfall entered at the
start of period 2 and surfaced when the carry-out at t=3 was tested.
Repair: r(A) = (22,48,30), I(A) = (2,25,15). All balances and capacities hold.
```

Three readings are worth making explicit. **The balance is checked period by period.** Summed over
the horizon, plan P buys 98 ambient units and ships 85, leaving 13 where the terminal condition
requires 15: a 2-unit discrepancy with no location. The period check localises it to `t = 3` and
names the cause, the 2 units that should have been carried out of period 1. Report the first
period with a non-zero residual, because that is where the modelling error is. **Balance and
capacity feasibility are independent tests.** Plan P passes every capacity row and fails the
balance; the plan in section 6.3 passes every balance row and fails capacity at `t = 2`. A plan can
fail either, both, or neither. **The repair is a purchase-timing decision.** Moving 2 units from
period 2 to period 1 costs 0.80 in holding and is the only way to satisfy the terminal condition
without transferring from South; in a larger instance it appears as a build-ahead decision with a
real cost.

> **Note:** the same check applies to the in-transit treatment. If a lane has a one-period lag and
> the plan shows receipts in period 1, either the opening pipeline is non-zero and stated, or the
> lag has been dropped from the code. There is no third explanation.

## 8. Data, standards, and regulatory context

The model's inputs are governed by standards at the identification layer, the calendar layer, and
the accounting layer. Knowing which standard governs which input tells you which argument about
an input can be settled by a document and which is a judgement.

### 8.1 Identity, units, and calendars

| Standard or convention | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 Global Trade Item Number (GTIN) | Product identity | The SKU list, the demand history, and the cost file must key to the same item identifiers or the aggregation cannot be audited | B |
| GS1 Global Product Classification (GPC) | Product classification | Provides a four-level hierarchy of segments, families, classes, and bricks; it is a classification, not a modelling aggregation | B |
| GS1 Global Location Number (GLN) | Site identity | Nodes, lanes, and the freight ledger must join on one location key | B |
| ISO 8601 | Dates, times, and intervals | Period boundaries, horizon start and end, and the definition of a duration | B |
| ISO 80000 | Quantities and units | Demand, capacity, and conversion factors must share a unit before any capacity row means anything | B |
| ISO 3166 / ISO 4217 | Country and currency codes | Cross-border lanes, duty, and multi-currency cost bases | B |
| 4-4-5 retail calendar (NRF convention) | Retail period structure | Quarters of 91, 91, and 92 days; an inventory balance on calendar months will not reconcile | C |
| Company fiscal calendar | Reporting periods | The periods the ledger closes on, which need not be the periods the model uses | C |

The GPC point is worth stating plainly. A company that has classified its range into GPC bricks
has a hierarchy, but it does not have a modelling aggregation: a GPC brick groups products by what
they are, a product family groups them by how they consume capacity and how their demand behaves,
and where the two disagree the modelling aggregation wins, because the model is about resources.
The calendar point is the one that breaks reconciliations: if the model uses calendar months and
the ledger uses 4-4-5, every period-end inventory comparison is off by up to three days of flow,
and the variance gets attributed to the model when it is a definitional mismatch.

### 8.2 Inventory valuation and the accounting interface

| Requirement | Where | Effect on the model | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | Global accounting | Inventory is carried at the lower of cost and net realisable value; the model's unit value must be a cost, not a selling price | B |
| IFRS 16 / ASC 842 leases | Global accounting | A leased site's annual cost appears as depreciation and interest, not rent; the model needs the cash cost or a bridge to it | B |
| IAS 37 provisions | Global accounting | Closure and restoration obligations belong in a transition calculation, not in the annual fixed charge | B |
| Transfer pricing rules | Multinational | Internal transfers carry a price that is not an economic cost; using it as a flow coefficient biases the design | B |
| ISO 14083:2023 | Global | Defines transport emissions quantification if carbon enters the objective; topic 09 | B |
| Incoterms 2020 | Global contract convention | Defines where cost and risk transfer, and therefore who owns in-transit stock | B |

The in-transit ownership question is the one that interacts with this topic most directly. Under
free carrier (FCA) the buyer owns the goods from the moment they leave the seller; under delivered
duty paid (DDP) the seller owns them until delivery. If the model is built for the buyer and the
lane is DDP, the in-transit stock belongs to someone else and should not appear in the buyer's
inventory variable. Getting this wrong adds a pipeline the company does not own, which flatters
the design by exactly the carrying cost of that pipeline.

### 8.3 What is requirement and what is convention

- **Requirement:** every flow is attributable to a commodity that maps to identifiable items.
  **Convention:** that the mapping uses GPC bricks, or any other published classification.
- **Requirement:** demand, capacity, and conversion factors share a unit. **Convention:** that the
  unit is pallet-equivalents rather than cubic metres or cases.
- **Requirement:** opening stock plus receipts equals closing stock plus shipments, for every
  node, commodity, and period. **Convention:** that the stock is measured at period end.
- **Requirement:** the terminal condition is stated and its cost is reported. **Convention:** that
  it is one period of forward demand.
- **Requirement:** a shared resource is consumed by every commodity that uses it. **Convention:**
  that the consumption is linear in volume.

The last convention is the one to challenge in a study with a real peak: dock and labour
consumption are not linear above about 85% utilisation, where queues form and the marginal minute
costs more than the average, which is precisely where the interesting decisions live.

## 9. Industry practice and case evidence

### 9.1 What the documented cases show

The Digital Equipment Corporation (DEC) global supply chain model, reported by Arntzen and
colleagues in *Interfaces* in 1995, is the reference application of a multi-commodity,
multi-period network model. It combined production, distribution, duty, and lead time in one
formulation and was used for strategic questions about sourcing and facility configuration. What
transferred from it is the structure: commodities with different economics, periods that carry
inventory, and an objective that compares designs rather than flows. What did not transfer is the
scale, which was at the edge of what the solvers of the period could do. Grade A for the published
account.

The Procter and Gamble restructuring study reported by Camm and colleagues in *Interfaces* in 1997
is the other canonical case, and its lesson is about validation: the model was credible because it
was reconciled to existing accounting data, the same discipline as the baseline validation in
section 5.3. Grade A for the published account.

### 9.2 What the multi-period literature reports about the boundaries

| Study | Setting | Reported finding | Grade |
|---|---|---|---|
| Hinojosa and colleagues, *Computers and Operations Research*, 2008 | Dynamic two-echelon, multi-commodity location with inventory carryover and outsourcing | A compact period-indexed formulation with flow conservation linking plant flows, warehouse stock, and customer demand, solved by Lagrangian relaxation by echelon | A |
| Correia and Melo, logistics working paper 21, 2023 | Multi-period, two-echelon, multi-product design with modular capacity expansion and reduction | Instances up to 48 periods and 400 customers, above 1.2 million binaries, solved by an LP-driven matheuristic that beat the solver on most instances | C |
| Meisel, Rei, Gendreau, and Bierwirth, CIRRELT working paper, 2011 | Supply network design under maximum customer order lead times | Reformatting as a fixed-charge flow on a time-space network removed explicit lead-time constraints and cut solve time from about eight hours to under a second on their first test set, with better average gaps | C |
| Melo, Nickel, and Saldanha-da-Gama, *European Journal of Operational Research*, 2009 | Review of facility location in supply chain management | Identifies integration of location with inventory, capacity, and multi-period decisions as the central open area, and notes that most applied models simplify at least one of them | A |

The pattern across these is consistent: the formulation dominates the solve time more than the
instance size does. A study that adds commodities and periods without a structural choice will hit
the time limit and report a large gap.

### 9.3 What the evidence supports

It supports three claims. A multi-commodity multi-period formulation captures effects that a
single-period model cannot, and the effects are large enough to change a recommendation when
there is a genuine peak. The formulation choice, especially compact against time-expanded, can
change the solve time by orders of magnitude without changing the feasible set. And the treatment
of the horizon end is a modelling decision with a measurable cost, not a technical detail.

It does not support the claim that more periods produce a better decision, that item-level models
are more accurate than family-level models for a strategic question, or that a percentage saving
from one network transfers to another.

## 10. Failure modes and red flags

Each row names a failure, the red flag that is visible in a meeting or a dataset before anyone has
diagnosed it, the mechanism, the consequence, and the mitigation.

| Failure | Red flag you can see | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Network emptied at the horizon | Terminal inventory is zero at every node in the last period | No terminal condition, and stock left behind is pure cost in the objective | Cost understated; the next horizon starts by rebuilding stock at unbooked cost | State a terminal condition and report the cost of the alternative |
| Terminal condition decided the design | The open set changes when the terminal rule changes | The condition is carrying most of the economics | The recommendation is an artefact of an unargued constraint | Solve two terminal rules and report whether the open set moves |
| Annual model hides the peak | Utilisation is comfortable and the operations lead disagrees | One period per year averages a seasonal profile | Design is at or beyond capacity in the peak month | Model at a period length that can see the peak; report peak utilisation |
| Balance written with a sign error | Inventory rises with throughput at a node that stores nothing | Every term placed on one side of the equation | The same unit is counted twice; cost and service both wrong | Recompute the residual per period after solving |
| Opening stock dropped | Period 1 is feasible only by buying more than the site can receive | Loop starts at `t = 1` with no predecessor term | Carry-over silently vanishes; peaks infeasible or falsely met | Check the equation for `t = 1` explicitly against opening stock data |
| In-transit stock counted as available | Receipts appear in the same period as the shipment on an ocean lane | The model is built against the shipment record, not the receipt record | Plan is short by exactly the pipeline | Use a lag or a transit arc, and state the opening pipeline |
| Commodity grain copied from the item master | The model has 4,000 commodities and runs for a day | Aggregation was never decided, only inherited | Solver time limit with a large gap; a plan nobody can read | Choose the grain, document the compression ratio, test the aggregation |
| Capacity row sums unweighted units | Ambient and frozen count the same per pallet | Conversion factors all set to 1 | Frozen demand is understated in resource terms; the design is short of freezer | Build the commodity-resource matrix from observed operations |
| Capacity converted with an average | One capacity number covers dock, storage, and labour | Different resources bind in different periods | Design feasible on average, infeasible in the week that matters | Separate rows for each resource that can bind |
| Lead time rounded to whole periods | A 21-day lane is modelled as one month | Period length does not divide the transit time | Pipeline understated by 30% on that lane, systematically | Choose periods that divide the transit, or model the lag fractionally |
| Inventory costed as a linear rate only | The plan holds stock in a site with no positions left | Space is a step, not a rate | Plan requires storage that does not exist | Add a storage capacity row where positions bind |
| Plan nervousness read as forecast error | Every re-solve changes last month's released quantities | No frozen horizon; the model is re-optimised with no commitment | Suppliers and sites lose confidence in the plan | Freeze the near periods; measure and report plan change |

Two rows deserve expansion. **The end effect is the most consequential and the least discussed.**
It is not an error in the model; it is the model doing what it was told. A deterministic model with
a finite horizon has no reason to leave stock behind, so the last period looks unusually cheap and
productive, and readers who have seen many studies learn to discount it — a heuristic substitute
for a terminal condition. **Unweighted capacity rows are the quietest failure in the table.**
Nothing about the output looks wrong: the utilisation figures are percentages, the cost is a cost,
and the plan is feasible. The error appears only when someone compares the model's "pallet" to a
freezer position and finds that a pallet of frozen goods occupies 1.4 positions and needs a
different dock. In a mixed network that is the difference between one site and two.

## 11. Assumptions and limits

| Assumption | Why the model needs it | What breaks if it fails |
|---|---|---|
| Demand per commodity and period is known | It is a parameter, not a variable | Near a capacity threshold the plan flips; the peak-period design is the one most exposed | 
| Commodities within a family are interchangeable for planning | The family is one flow | Disaggregation produces an item plan that cannot meet item demand |
| Capacity consumption is linear in volume | The capacity row is linear | Queues and congestion above about 85% utilisation make the marginal unit dearer than the average |
| Resource consumption per unit is constant across periods | Conversion factors are parameters | Mix shifts and campaign structures change the factor without changing the number |
| Inventory is fungible across periods | One stock variable per commodity | Shelf life and batch minimums make old and new stock different goods |
| Storage capacity does not bind | Only throughput is constrained | The plan holds stock the building cannot store |
| Lead times are deterministic and integer in periods | The lag or transit arc uses one number | Transit variability changes the pipeline and the stock needed to cover it |
| The terminal condition represents the world after the horizon | The model ends at `T` | The design is decided by an arbitrary end rule |
| Costs are separable per unit and per period | The objective is linear | Volume discounts and step tariffs make the marginal cost fall with volume |
| The discount rate applies uniformly to all cost lines | One rate in the objective | Capital, inventory, and operating costs have different time profiles and different risks |

Two limits are worth stating plainly.

**The model has no theory of product substitution.** If two commodities can serve the same
demand, the model will not discover it. Substitution appears only if you write an arc, a
constraint, or a merged commodity, and each of those is a statement about customer behaviour that
the data rarely supports. A network study that assumes substitution it cannot evidence will
under-provision the network and discover the error in a promotion.

**The model's time is accounting time, not physical time.** A period is a bucket, and everything
inside it is simultaneous. A shipment that takes eleven days and a shipment that takes forty are
both "one period" if the period is a month. For a strategic decision that is acceptable, because
the decision is about capacity and structure rather than about sequences. It stops being
acceptable when the question is whether a site can turn a trailer in a day, which is a
scheduling question and belongs to Module 15.

## 12. Exercises

### Exercise 1 — Choose a commodity grain (design, 50 minutes)

Take a product range you know, or a published one with at least 60 items and three handling classes, and define a family structure three ways: by handling class only, by the four-condition test in section 3.1, and by the company's existing item hierarchy.
**Deliverable:** three commodity lists, the compression ratio for each, and a one-page defence of the one you would use for a network study.
**Acceptance criteria:** every family satisfies the four conditions or has a stated exception; the compression ratio is computed for each option; the defence names one decision that would change with a coarser grain and one that would not.

### Exercise 2 — Build the commodity-resource matrix (application, 45 minutes)

For one distribution centre, list the resources that can bind — dock, storage positions, pick labour, trailer slots — and the commodities that consume them.
**Deliverable:** a matrix of conversion factors with units, plus a note on which resource binds first at the current volume.
**Acceptance criteria:** every factor is derived from an observation or labelled an assumption; the units of flow are the same in every row and column; at least one factor is not 1.0, with the reason stated.

### Exercise 3 — Repair a balance (calculation, 40 minutes)

Using the instance in section 7, test the plan `r(A) = (20, 48, 30)`, `I(A) = (0, 23, 15)`, then a second plan with `r(A) = (20, 50, 30)`, `I(A) = (0, 25, 15)`.
**Deliverable:** a period-by-period residual for both plans, the capacity check for both, and the repair for each failure.
**Acceptance criteria:** the first non-zero residual is identified by period and node; the failure mechanisms are named from the four in section 5, Step 5; the repair is costed.

### Exercise 4 — Period length and horizon (critique, 40 minutes)

A study proposes annual periods over a five-year horizon for a network with a four-week harvest peak and seven-year leases.
**Deliverable:** a one-page critique with a recommended period structure and horizon, and the cost of the recommendation in model size.
**Acceptance criteria:** the peak-to-average ratio is used explicitly; the lease term is addressed; the variable-count consequence is estimated; one alternative is rejected with a reason.

### Exercise 5 — The lead-time boundary (application, 50 minutes)

Take a lane with a 21-day transit and a monthly period, and model it as a lag of one, a lag of two, and a transit arc with an opening pipeline equal to 40% of annual volume.
**Deliverable:** the first-period receipts under each of the three treatments, and a paragraph on which you would use and why.
**Acceptance criteria:** the arithmetic is shown for all three; the percentage error of each lag against the transit arc is stated; the choice is justified by the decision the model supports.

### Exercise 6 — Terminal condition (calculation, 55 minutes)

For the section 6 instance, re-solve with the terminal requirement at 20 thousand ambient units instead of 15, then with the requirement set to zero.
**Deliverable:** the receipts, closing stocks, and total cost for each case, and the difference against the base case of 15.
**Acceptance criteria:** every balance holds; every capacity row is checked; the cost difference is decomposed into supply, handling, and holding; the direction of the bias is stated in one sentence.

### Exercise 7 — Discounting (calculation, 45 minutes)

Take the base plan's cost by period from section 6.3 and compute its present value at 3% per quarter, then at 10%, and compare with the undiscounted total.
**Deliverable:** a table of period, cost, discount factor, and present value at each rate, plus a paragraph on what the rate does and does not change.
**Acceptance criteria:** the real-versus-nominal rule is stated; the factors are shown; the difference is expressed in currency and percent; no claim is made about the design changing unless the arithmetic shows it.

### Exercise 8 — Rolling horizon versus one plan (design, 60 minutes)

Design the re-planning procedure for a model that supports a monthly S&OP cycle: the horizon at each re-solve, the frozen period, what is fixed from the previous plan, and how plan nervousness is measured.
**Deliverable:** a two-page procedure with a worked month showing what changes and what does not.
**Acceptance criteria:** the frozen horizon is justified; the instability measure is defined with a formula; one consequence of not freezing is described; the procedure says who owns the re-solve.

## 13. Self-check answers

**Exercise 1.** A strong submission gives a compression ratio of roughly 10 to 30 SKUs per family
and shows that the handling-class split comes first, because mixing ambient and frozen in one
family makes the capacity row meaningless. The decision that changes with a coarser grain is
usually the site count, through loss of handling-specific capacity: a network sized on total
pallets under-provides freezer positions. The decision that does not change is the choice of
region, because geography dominates at any grain. A weak submission adopts the item hierarchy
unchanged and reports a ratio of 1.

**Exercise 2.** A plausible answer: dock minutes per pallet of 4, 6, 7, and 12 for ambient,
chilled, frozen, and hazardous; storage positions of 1.0, 1.0, 1.4, and 1.8; pick minutes of 1.2,
1.5, 1.6, and 3.0. The resource that binds first is usually pick labour, not dock, because labour
scales with order lines rather than pallets. A submission that sets every factor to 1.0 has not
done the exercise.

**Exercise 3.** First plan: residuals are zero at `t = 1` and `t = 2` and 2 units short at `t = 3`
(23 plus 30 gives 53 against 55). Capacity passes at 30, 60, and 60. The mechanism is a
purchase-timing error: the closing stock of period 1 is zero when it should be 2, so the carry into
period 3 is short. The repair is `r(A) = (22, 48, 30)` and `I(A) = (2, 25, 15)`, costing 0.80 in
extra holding. Second plan: balances hold at every period, but capacity fails at `t = 2` (50
ambient plus 12 for cold gives 62 against 60). The mechanism is the shared capacity row, not the
balance, and the repair is the one in section 6.3. Both plans fail, for different reasons, and the
two tests are independent.

**Exercise 4.** The peak-to-average ratio is the argument. If the four-week harvest carries 40% of
annual volume, the peak month runs at roughly 4.8 times the average monthly rate, and an annual
model reports a network feasible at 20% of the capacity that month needs. Recommend monthly periods
for the harvest window and quarterly elsewhere; a non-uniform period structure is legitimate,
because the balance equations require stated periods, not equal ones. On size, twelve monthly
periods per year multiply flow variables by three against quarters. The lease term argues for a
horizon of at least seven years, or for treating years six and seven as an annuity beyond the
model and saying so in the terminal condition.

**Exercise 5.** A 21-day transit in a monthly model is 0.7 periods. A lag of one assumes arrival at
the end of the first month, understating the pipeline by about 11 days, or 35% of the transit; a
lag of two overstates it by about 9 days, or 43%. A transit arc with a 40% opening pipeline has 40%
of annual volume available in the first period. Use the transit arc when the first period is one
the business will execute and the pipeline is material; use the lag for a five-year strategic
comparison where the first-period error sits inside the design gap.

**Exercise 6.** With a terminal requirement of 20: receipts `r(A) = (27, 48, 30)`, closing
`I(A) = (7, 30, 20)`, capacity 37, 60, and 60, and cost 2,227.80, which is 48.50 above the base
case — 40.00 of extra supply for five more units, 2.50 of receiving and handling, and 6.00 of
holding. With a requirement of zero: `r(A) = (20, 35, 30)`, `I(A) = (0, 10, 0)`, capacity 30, 47, and
60, and cost 2,039.00, or 140.30 below. The zero case is cheaper because it does not buy the 15
units it would leave behind, which is precisely the bias.

**Exercise 7.** Period costs are 460.50, 749.80, and 969.00, summing to 2,179.30. At 3% per quarter
the factors are 1.0000, 0.9709, and 0.9426, giving 460.50, 727.98, and 913.38, for 2,101.86, which
is 77.44 or 3.6% below the undiscounted total. At 10% the factors give 460.50, 681.64, and 800.79,
for 1,942.93, which is 236.37 or 10.8% below. The rate changes the level substantially and these
plans' ranking not at all, because their cost profiles have almost the same shape; it would change
the ranking if one plan front-loaded capital and the other spread it.

**Exercise 8.** A workable procedure: horizon of 18 months at monthly resolution, re-solved
monthly; the first month frozen entirely and the second frozen for quantities already released to
suppliers. At each re-solve, opening inventory comes from the warehouse system, not the previous
plan, which is the most common procedural error. Plan nervousness is the mean absolute change in
planned quantity per period divided by mean demand, reported by node and commodity; above about
0.15 the plan is churning. Not freezing means a supplier receives a different call-off every month
and stops treating the plan as a commitment. The owner is the planning lead, not the analytics
team, because the output is a commitment.

## 14. Interview questions

**1. "Why not just run the single-period model for each year separately?"**
Strong: the years are linked by inventory, leases, and the fact that a site opened in one year is
still open in the next, so separate models optimise each year against a network it cannot inherit.
Weak: says the multi-period model is more accurate, without naming the coupling.

**2. "What changes when you add a second commodity?"**
Strong: capacity becomes a shared resource consumed with commodity-specific factors, so the
commodities couple at exactly one place, and a constraint slack for each commodity alone can bind
for the pair. Weak: says the model gets bigger.

**3. "How would you choose between product families and SKUs?"**
Strong: by the decision the model supports, the capacity conversion differences inside the family,
whether substitution is acceptable, and the compression ratio; a family whose members need
different handling is not a family. Weak: says SKUs are more accurate, which is true and
irrelevant if the model does not finish.

**4. "Write the inventory balance for a node."**
Strong: opening plus receipts plus production equals closing plus shipments, per commodity and
period, with the lag on inbound arcs and the period-end convention stated. Weak: writes flow
conservation with no stock term and calls it a balance.

**5. "How can a balance equation be wrong if the model solves?"**
Strong: names the four errors — sign, dropped opening stock, in-transit treated as available,
period-end mixed with period-start — and notes that each yields a feasible-looking plan with a
residual only recomputation finds. Weak: says the solver would catch it.

**6. "What is an end effect and what would you do about it?"**
Strong: the model has no reason to leave stock at the horizon end, so it empties the network and
understates cost; the fix is a terminal condition, and the test is whether the open set moves.
Weak: calls it a solver artefact.

**7. "Lag or transit arc — does it matter?"**
Strong: they agree in the interior and differ at the boundary, because a transit arc can carry an
opening pipeline and a lag cannot; the difference lands in the first periods, which are the ones
being executed. Weak: says they are equivalent.

**8. "How do you discount a network study's costs?"**
Strong: money, not quantities; real with real and nominal with nominal; capital as a lumpy outflow
rather than an annuity; inventory as working capital recovered later. Weak: applies one rate to
every line and calls it NPV.

**9. "The model runs for six hours and returns a 12% gap. What do you do?"**
Strong: examine the commodity grain, the period length, and the formulation before buying hardware;
report the gap in currency and present the incumbent as a good design rather than an optimum.
Weak: raises the time limit.

**10. "Why re-solve monthly rather than solve once for the whole horizon?"**
Strong: the forecast changes, and an unrevised plan is one the business ignores; but re-solving
without a frozen horizon creates nervousness, so the procedure needs a cadence and a commitment.
Weak: says re-solving is more accurate.

**11. "What six things must a multi-period network study report?"**
Strong: commodity definition, period length and horizon, terminal condition, lead-time treatment,
model size and gap, baseline validation. Weak: reports the saving and the open set.

## 15. Key terms

- **Commodity** — a flow the model cannot substitute for another; a modelling choice, not a fact.
- **Commodity grain** — the level of product detail carried: handling class, family, or SKU.
- **Product family** — a group of SKUs modelled as one commodity, justified by four conditions.
- **Handling class** — ambient, chilled, frozen, hazardous, or high-value; a capacity distinction.
- **Aggregation consistency** — the property that an aggregate plan can be disaggregated without
  violating item-level constraints.
- **Substitutability** — whether one commodity can serve demand for another; never assumed.
- **Coupling constraint** — a row containing more than one commodity, so they cannot be solved
  separately.
- **Commodity-resource matrix** — conversion factors from commodity units to resource units.
- **Time-expanded network** — the physical graph replicated once per period, linked by inventory;
  the freight literature calls the same object a **time-space network**.
- **Period** — the accounting bucket; **horizon** — the number of buckets.
- **Opening inventory** — stock at the start of a period; **closing inventory** — at the end.
- **Balance equation** — opening plus receipts equals closing plus shipments; at a node holding
  no stock it reduces to **flow conservation**.
- **In-transit inventory** — goods shipped and not received; **pipeline stock** — all of it on a
  lane.
- **Transit arc** — an arc that carries an explicit stock variable over its traversal; a **lag** is
  a fixed delay with no stock variable.
- **Terminal inventory** — the stock required at the end of the horizon; the **terminal condition**
  is the constraint that fixes or bounds it.
- **End effect** — the bias in early decisions caused by the treatment of the horizon end.
- **Zero-inventory-ordering property** — the uncapacitated lot-sizing result that fails under
  capacity.
- **Holding cost rate** — currency per unit per period, not an annual percentage.
- **Discount factor** — the multiplier converting future money to present value; **NPV** is net
  present value, the sum of discounted cash flows.
- **Rolling horizon** — re-solving a fixed-length model as time advances; the **frozen horizon** is
  the near periods whose decisions are fixed before a re-solve.
- **Plan nervousness** — the change in planned quantities between successive re-solves.
- **Seasonal capacity** — capacity that varies by period through staffing or temporary space.
- **Lot sizing** — production quantity per period with setup costs and capacity; the parent
  problem of the balance equation.

## 16. Sources

**Peer-reviewed and academic**

- Melo, M. T., Nickel, S., and Saldanha-da-Gama, F. (2009). "Facility location and supply chain
  management — a review." *EJOR*, 196(2), 401–412. DOI: 10.1016/j.ejor.2008.05.007. Grade A.
- Hinojosa, Y., Kalcsics, J., Nickel, S., Puerto, J., and Velten, S. (2008). "Dynamic supply chain
  design with inventory." *Computers and Operations Research*, 35(2), 373–391. Grade A.
- Fisher, M. L., Ramdas, K., and Zheng, H. (2001). "Ending inventory valuation in multiperiod
  production scheduling." *Management Science*, 47(5), 679–692. Grade A.
- Charles, A., Dauzère-Pérès, S., Kedad-Sidhoum, S., and Mazhoud, I. (2022). "Motivations and
  analysis of the CLSP with setup times and minimum and maximum ending inventories." *EJOR*.
  DOI: 10.1016/j.ejor.2021.12.017. Grade A.
- Rogers, D. F., Plante, R. D., Wong, R. T., and Evans, J. R. (1991). "Aggregation and
  disaggregation techniques and methodology in optimization." *Operations Research*, 39(4),
  553–582. Grade A.
- Sali, M., Ghrab, S., and Chatras, C. (2023). "Optimal product aggregation for S&OP in mass
  customisation context." *IJPE*, 263, 108948. Grade A.
- Wagner, H. M., and Whitin, T. M. (1958). "Dynamic version of the economic lot size model."
  *Management Science*, 5(1), 89–96. Grade A; origin of the balance equation and the
  zero-inventory property.
- Arntzen, B. C., Brown, G. G., Harrison, T. P., and Trafton, L. L. (1995). "Global supply chain
  management at Digital Equipment Corporation." *Interfaces*, 25(1), 69–93. Grade A.
- Camm, J. D., Chorman, T. E., Dill, F. A., Evans, J. R., Sweeney, D. J., and Wegryn, G. W. (1997).
  "Blending OR/MS, judgment, and GIS: restructuring P&G's supply chain." *Interfaces*, 27(1).
  Grade A.
- Al-Ashhab, M. S., and Afia, N. (2017). "An NPV optimization model for closed-loop supply chain
  network design and planning." *American Journal of Engineering and Applied Sciences*, 10(1),
  114–125. Grade A, with the caveat that the journal is not a specialist OR outlet.

**Working papers and technical reports**

- Meisel, F., Rei, W., Gendreau, M., and Bierwirth, C. (2011). "The design of supply networks under
  maximum customer order lead times." CIRRELT working paper 2011-57. Grade C: not peer-reviewed at
  the version consulted, and the timing figures are specific to their test sets.
- Correia, I., and Melo, T. (2023). "Modeling and solving a dynamic logistics network design
  problem with temporary capacity expansion and reduction." HTW Saar logistics working paper 21;
  journal version in *Computers and Industrial Engineering*. Grade C for the working paper.

**Standards and official**

- GS1 General Specifications, current edition, including GTIN, GLN, and GPC. Grade B.
- ISO 8601:2019; ISO 80000; ISO 3166; ISO 4217; ISO 14083:2023. Grade B.
- IAS 2 *Inventories*; IFRS 16 and ASC 842 *Leases*; IAS 37 *Provisions*. Grade B.
- Incoterms 2020, International Chamber of Commerce. Grade B.

**Industry and analyst**

- ASCM supply chain dictionary, for demonstrated capacity and related definitions. Grade C.
- National Retail Federation 4-4-5 calendar convention. Grade C.
- Vendor and consultancy network-design case studies. Grade D, and not quotable as evidence about
  any other network.
- Practitioner descriptions of typical multi-period study parameters. Grade E, offered as a
  description of the norm rather than as evidence about any company.

**Note on citations.** Figures are reported as their sources state them and are not independently
verified. The timing and instance figures in section 9.2 belong to their authors' test sets. The
Correia and Melo and Meisel and colleagues reports were consulted as working papers; their journal
versions may differ and should be cited in a formal study.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Predecessor, the single-period model this topic extends: [`04_facility-location.md`](04_facility-location.md)
- The MILP (mixed-integer linear programming) anatomy this topic assumes: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- The graph and cost-to-serve work that supplies the lanes: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- The decision ladder that places this model: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- The part map: [`../_index.md`](../_index.md)
- Module 10, which owns inventory theory: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Module 10's multi-echelon topic, for safety stock as a policy: [`../10_inventory_optimization_multi_echelon/08_multi-echelon.md`](../10_inventory_optimization_multi_echelon/08_multi-echelon.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- This module's survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)
- Next in this module: [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md) — phasing and legacy assets
- Then: [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md) — which assumptions change the answer
- Then: [`08_robust-stochastic.md`](08_robust-stochastic.md) — demand and disruption uncertainty
- Then: [`09_carbon-objective.md`](09_carbon-objective.md) — emissions in the objective
- Closing topic: [`10_solvers-heuristics.md`](10_solvers-heuristics.md) — gaps, decomposition, and limits
