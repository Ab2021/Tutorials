# Planning Processes: S&OP, IBP, and Execution

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 02, 03
> Version 2026.5

---

## 1. Why this topic exists

A supply chain does not decide anything. People in meetings decide. Between the network (Topic 03)
and the inventory position (Topic 05) sits a piece of social technology: a calendar of meetings that
turns data into a committed plan, and a committed plan into a month of physical work.

That calendar is the subject of this topic, and it is not a soft subject. It is where your model
either enters the business or fails to.

**The meeting is the algorithm.** A planning process is a function from inputs to a plan of record,
executed by humans on a schedule. It has gates, data cut-offs, owners, permission sets, and an
output. Change the calendar and you change the plan more than any model change will. A forecasting
model that improves mean absolute percentage error (MAPE) by two points but arrives after the demand
review cut-off changes nothing this month, and next month someone has learned not to wait for it.

**The number that reaches the executive is not the number the analyst produced.** Between the
statistical baseline and the signed plan sit four transformations, each performed by a different
function with a different incentive. Without knowing the sequence you cannot explain why your
forecast was "wrong" when it was overridden.

**The process is a chain of custody for a number.** Every gate signs the number and passes it on.
When the plan fails, "whose number was it?" has an answer, and the answer is in a meeting minute
rather than a model log.

**Why this matters to a data scientist.** Your output has a delivery slot. Arrive after the cut-off
and you are next month's input. Arrive without an owner and you are ignored. Contradict the financial
plan and you are overruled by arithmetic nobody showed you. Most analytics teams learn this after a
successful pilot that changed no behaviour. The failure is almost never in the model. It is in the
fit between the model's output and the calendar's slots.

Two examples show the shape of the problem.

1. **The forecast that was right and never used.** A model beats the statistical baseline by 6% at
   item-week level and runs nightly at 02:00. The demand review cut-off is 17:00 on the ninth
   working day, and the planning system imports forecasts only on the seventh. The output is never
   imported. The team reports success; the planners use the old numbers. The calendar had no slot
   for a daily refresh.
2. **The plan that was approved and immediately abandoned.** An executive review approves 41,800
   units for April. On 3 April a customer orders 6,000 units, 900 above the remaining commitment.
   Sales pulls from May. By 14 April the April plan is fiction and May is short. No process owned
   the transition from plan to execution, so execution rewrote the plan silently.

Both are process failures, not forecasting failures. This topic teaches you to see the difference
and to design analytics that land inside the process rather than beside it.

A third failure is the one that costs most and leaves no trace. A company runs every gate on time,
with good data, and produces a plan everyone signs. The plan is never executed, because the
commitments it contains were never translated into orders, schedules, and supplier releases. The
month ends, the plan is measured as if it had been an instruction, and the variance is attributed to
forecast error. Nothing in the meeting record shows that the plan was advisory.

The topic sits fourth because it needs the flows (Topic 01), the decision layers and time fences
(Topic 02), and the entities and product hierarchy (Topic 03) before it can be precise about who
meets, when, with what number, and what they can actually change.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the five-step sales and operations planning (S&OP) cycle from memory, labelling each gate
   with the number it owns and the changes it may make.
2. **Distinguish** S&OP, integrated business planning (IBP), and sales and operations execution
   (S&OE) by horizon, cadence, granularity, owner, and decision rights.
3. **Construct** a planning calendar with cut-offs, gate dates, inputs, owners, and artefacts.
4. **Predict** the three fates of a decision that misses its gate and **cost** each one.
5. **Compute** forecast value added (FVA) for a series of overrides and **interpret** it honestly.
6. **Critique** a consensus demand process for correlated-error and shared-optimism failure.
7. **Separate** process maturity from decision quality, and **assess** a maturity claim.
8. **Define** decision rights for an artificial intelligence (AI) planning agent: who signs the plan,
   what it may change unsupervised, and how an override is logged.

Each outcome maps to an exercise in section 12 and a self-check in section 13.

## 3. The core mental model

Hold one picture: a monthly loop of five gates, each holding a different number, with a fast inner
loop for execution and a slower outer loop for strategy.

```text
              THE MONTHLY CYCLE IS A CHAIN OF CUSTODY FOR ONE NUMBER

   history ---+
   orders  ---+--> [0] DATA CUT-OFF ---> one frozen version
   shipments--+             |
                            v
                   [1] PRODUCT REVIEW     owns: which items exist
                            |             may change: launch, retire, substitute
                            v
                   [2] DEMAND REVIEW      owns: the demand number
                            |             may change: volume, mix, timing
                            v
                   [3] SUPPLY REVIEW      owns: the supply number
                            |             may change: build, buy, move,
                            |             but only within existing capacity
                            v
                   [4] FINANCIAL          owns: the valuation
                       RECONCILIATION     may change: price, margin, cash, phasing
                            |
                            v
                   [5] EXECUTIVE REVIEW   owns: the trade-off
                            |             may change: which constraint breaks
                            v
                      PLAN OF RECORD
                            |
             +--------------+---------------+
             |                              |
             v                              v
   [ S&OE  weekly / daily ]         next month's cycle
   executes the committed month
   re-sequences, expedites, allocates
   may NOT reset the month's volume
```

Five properties explain most of what follows.

**Property 1 — Each gate owns one number and may change only that number.** Demand review owns
demand; it may not change capacity or price. Supply review owns supply; it may not edit the demand
number, even when it believes the number is wrong. Its remedy is to state the gap. The separation
feels bureaucratic until you watch a company without it: demand gets quietly reduced to match
capacity, the shortfall vanishes from the record, and the customer still experiences it.

**Property 2 — Authority to discuss is wider than authority to change.** Everyone may comment on
capacity; only the supply review may commit it. Most meeting dysfunction is people exercising
discussion rights as decision rights.

**Property 3 — The calendar, not the model, sets the latency of response.** A company that meets
monthly cannot respond to a demand shift in under a month, however good its sensing. One-day
detection with thirty-day decision latency is an accurate observer of a problem it cannot fix.

**Property 4 — A decision that misses its gate has three fates: wait, jump, or leak.** It queues to
the next cycle (cost: latency), escalates out of sequence (cost: legitimacy and rework), or is
decided informally outside the process (cost: a shadow plan). Leak is the most common and most
damaging.

**Property 5 — The executive review chooses which constraint to break; it does not compute.** By the
time it meets, the arithmetic has been done four times. Its job is to decide whether to break the
service promise, the cost target, the cash target, or the revenue target, and to say which one in
public with a name attached.

### 3.1 The permission ladder

| Gate | Question it answers | Number it owns | May change | May not change | Typical owner |
|---|---|---|---|---|---|
| 0. Data cut-off | Which version of the world are we planning against? | The input set | Nothing; it freezes | Any number after the freeze | Process owner |
| 1. Product review | Which items exist in the horizon? | The portfolio | Launch, retire, substitute, phase | Total volume, capacity, price | Product management, R&D |
| 2. Demand review | How much will the market take? | The demand number | Volume, mix, timing, channel split | Capacity, price, supply plan | Sales, demand planning |
| 3. Supply review | How much can we make, buy, and move? | The supply number | Build, buy, move, outsource, expedite | The demand number, the portfolio | Manufacturing, procurement |
| 4. Financial reconciliation | What is it worth, and when? | The valuation | Price, margin, cash timing, phasing | Volume, supply plan, portfolio | Finance |
| 5. Executive review | Which constraint do we break? | The trade-off | Targets, priorities, allocation rules | The arithmetic, the definitions | General manager, chief supply chain officer (CSCO), chief financial officer (CFO) |

The ladder is a separation of powers: no function sets both demand and supply, which is the
structural defence against the commonest form of planning corruption — adjusting the forecast to fit
the factory. It is also fragile at the seams. Gate 3 cannot change demand, so a shortfall becomes a
gap for gate 5. Gate 4 cannot change volume, so a margin problem becomes a pricing decision for
gate 5. Every seam is a place where a decision can leak.

### 3.2 The outer loop and the inner loop

The monthly cycle sets commitments: monthly volumes, inventory targets, capacity reservations, and
the financial plan. Inside it runs S&OE, weekly and sometimes daily over zero to thirteen weeks, at
stock-keeping unit (SKU) level.

- The outer loop decides **how much** the month contains and **what it is worth**.
- The inner loop decides **which order, which line, which truck, which customer** gets served.
- The outer loop may change the month's volume. The inner loop may not.
- The inner loop may re-sequence, expedite, substitute, allocate, and re-route.

This is why S&OP cannot fix the current month. The frozen zone from Topic 02 typically covers two to
four weeks. An S&OP meeting on the tenth working day is looking at a month already 50–80% committed
in the factory, released to suppliers, and promised to customers. The levers it holds are next
month's levers.

### 3.3 Why the gates are in this order

The order is not administrative. Each gate removes an ambiguity the next gate would otherwise
inherit.

Product before demand, because demand for a product that will not exist is not a forecast, it is an
error. Demand before supply, because capacity cannot be committed against a number that is still
moving. Supply before finance, because revenue cannot be valued before the constrained volume is
known; valuing unconstrained demand produces a plan the company cannot deliver and a gap that
finance spends the meeting explaining. Finance before executive, because the executive's only job is
the trade-off, and a trade-off needs one number set with a price attached.

Reverse any pair and the failure is predictable. Supply before demand gives a capacity plan searching
for a demand number. Finance before supply gives a revenue forecast built on unbuildable volume.
Executive before finance gives a decision taken without knowing what it costs.

## 4. Definitions and vocabulary

The vocabulary is treacherous because the same words mean different things in different companies,
and because several are also ordinary English.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| S&OP | Monthly cross-functional process balancing demand and supply at family level over 3–18 months | Used to mean "the forecast meeting" | Monthly buckets; family × location |
| IBP | S&OP extended to strategy, portfolio, capital, and finance, typically 24–36 months | Used as a synonym for "better S&OP" | Monthly buckets; 24–36 months |
| S&OE | Weekly or daily execution loop inside the current 0–13 weeks | Confused with "daily stand-up" | Daily or weekly; SKU × order |
| Consensus demand plan | One demand number agreed by functions, with assumptions recorded | Used to mean the average of four opinions | Units per period, plus assumption log |
| Demand review | The gate that owns the demand number | Used to mean the sales submission | Units per period by family |
| Supply review | The gate that owns the constrained supply number | Used to mean the master production schedule | Units per period by family or plant |
| Product review | The gate that owns which items exist in the horizon | Skipped when "nothing is launching" | Portfolio, SKU status codes |
| Financial reconciliation | The gate converting volume into revenue, margin, and cash, stating the gap to budget | Used to mean "finance checking the numbers" | Currency per period; profit and loss lines |
| Executive review | The gate that chooses which constraint breaks and signs the plan | Used to mean "the presentation to management" | Decision record |
| Pre-S&OP | Working session where finance and operations reconcile and frame options | Confused with the executive review | Decision options |
| Plan of record | The single version executed and measured against | Used loosely for "the latest plan" | Version identifier and date |
| Planning calendar | Published schedule of gates, cut-offs, inputs, owners, outputs | Treated as administrative | Date offsets, D−n |
| Data cut-off | The moment after which inputs to a gate are frozen | Treated as advisory | Timestamp, usually 24 h before the gate |
| Gate | Decision point with defined owner, input set, and permitted change | Confused with a deadline | Calendar date plus permission set |
| Freeze date | The date a plan version becomes the committed plan | Confused with a time fence | Date plus version |
| Time fence | A date inside which changes require defined authority | Confused with a freeze date | Days from today |
| Demand time fence (DTF) | Point inside which only actual orders count in demand | Treated as a forecast setting | Days from today |
| Planning time fence (PTF) | Point inside which the system will not auto-reschedule | Treated as a system parameter only | Days from today |
| Constrained plan | Supply plan respecting capacity, material, and transport limits | Presented as unconstrained | Units, plus binding constraint |
| Unconstrained plan | Supply plan meeting demand regardless of capacity | Treated as achievable | Units, plus capacity gap |
| Supply gap | Difference between demand and constrained supply | Reported as a demand error | Units per period |
| Scenario | Coherent alternative assumption set and its plan output | Used to mean a one-variable sensitivity | Named scenario, dated |
| Assumption log | Record of what had to be true for the plan to be right | Omitted from the pack | Text with owner and review date |
| Decision log | Record of what was decided, by whom, with rationale | Replaced by minutes | Decision ID, owner, due date |
| Forecast value added (FVA) | Error change attributable to one process step | Reported without a stated baseline | Percentage points of a named metric |
| Naive forecast | Benchmark assuming next period equals the last comparable period | Treated as a straw man | Units per period |
| Override | A judgmental change to a system forecast | Treated as expertise by definition | Units, plus reason code |
| Bias | The systematic component of forecast error | Confused with accuracy | Signed percentage over a period |
| MAPE | Mean absolute percentage error | Used on intermittent or near-zero series | Percent, per period |
| WAPE | Weighted absolute percentage error: total absolute error over total actual | Confused with MAPE | Percent, aggregate |
| Consensus | One number agreed by functions, with disagreement recorded | Used to mean the mean of submissions | Units, plus dissent record |
| Maturity model | Staged description of process capability | Used as a performance scorecard | Stage number plus dimension |
| Decision quality | Whether a decision was framed, informed, and owned well | Confused with outcome quality | Rubric, not a score |
| Touchless planning | Steps executed without intervention inside defined limits | Used to mean "fully automated" | Percentage of decisions, by value band |
| Agentic planning | Software agents propose and sometimes execute planning actions | Used to mean "a better dashboard" | Decision-rights specification |
| Human-in-the-loop | A human approves each agent action before it takes effect | Confused with human-on-the-loop | Approval rate, by decision class |
| Human-on-the-loop | A human monitors and may intervene, without approving each action | Confused with no oversight | Escalation rate, audit coverage |
| Decision rights | Who may decide what, at which value, with which review | Treated as a responsibility chart | Threshold table, by decision class |
| Rolling forecast | Forecast extended each period so the horizon stays constant | Confused with an annual budget | Months of horizon, refresh cadence |
| Volume, mix, price | The three decompositions of a revenue gap | Reported as one variance | Units, percent, currency |
| Integrated reconciliation | Step bringing demand, supply, and finance to one number set | Treated as a report | One number set, signed |

Four of these deserve expansion.

**Consensus is not an average.** A consensus plan is one number the functions agree to plan against,
with disagreement recorded. An average is the arithmetic mean of submissions. An average can be
produced without anyone agreeing to anything, hides a 20% disagreement between sales and marketing,
and distributes responsibility until nobody owns the error.

**FVA needs a stated baseline and metric.** "Our overrides add value" is not a claim until someone
says: relative to which baseline, measured how, over what period. FVA against naive and FVA against
the statistical forecast are different questions, and a team can honestly report positive on one and
negative on the other.

**Constrained versus unconstrained names the binding constraint.** Unconstrained answers "what would
we need?" Constrained answers "what can we do?" The gap is the planning problem and must be shown,
not smoothed.

**Maturity is not quality.** A maturity model describes whether the process has defined stages,
roles, and measurements. Decision quality describes whether a particular decision was framed with
alternatives, informed by evidence, and owned by a person. A stage-four process can approve a plan
nobody believes.

## 5. The mechanics, step by step

### 5.1 The cycle at a glance

| Step | Duration | Input | Output | Decision | Failure if skipped |
|---|---|---|---|---|---|
| 0. Data cut-off | 1 day | History, orders, pipeline, capacity, prices | Frozen versioned set | Which data version is authoritative | Two teams plan from two worlds |
| 1. Product review | 1–2 h | Portfolio plan, launch status, end-of-life | Item status by month | What exists, enters, retires | Demand includes unshippable products |
| 2. Demand review | 2–3 h | Baseline, pipeline, intelligence, FVA record | Demand plan | The demand number | Plan is a sales wish or a statistical artefact |
| 3. Supply review | 3–4 h | Demand plan, capacity, material, transport | Constrained plan and gap | The supply number and the constraint | Plan is unachievable and nobody knows where |
| 4. Financial reconciliation | 2–3 h | Both plans, prices, costs, budget | Revenue, margin, cash, gap | Valuation and phasing | Executive sees volume, not money |
| 5. Executive review | 1–2 h | One number set, options, recommendation | Plan of record | Which constraint breaks | Plan has no owner and no mandate |

Each step makes exactly one kind of decision, and the process fails when a step is asked to make a
different one.

### 5.2 Step 0 — The data cut-off

**Owns:** the input set. **May change:** nothing; its function is to stop change. **May not change:**
anything after the freeze. **Artefact:** a versioned snapshot with a timestamp and a manifest of what
was included and excluded.

The characteristic failure is that the cut-off exists on the calendar but not in behaviour. A sales
director emails a revised number on the morning of the demand review, the planner accepts because
refusing is socially expensive, and the review now plans against a set nobody else has seen.

> **Note:** a cut-off is real only if a late input has a documented destination. "We cannot take it
> this cycle; it goes to the exception log for the next" is a complete answer. "We cannot take it"
> without a destination invites the input to leak into an informal channel.

### 5.3 S&OP versus IBP

The horizon is the first distinction and the easiest to state: S&OP looks 3–18 months, IBP looks
24–36. The horizon follows from what the process must decide. A process deciding next quarter's
volume needs to see past the current lead time, roughly 3–6 months. A process deciding whether to
build a plant, launch a platform, or commit capital needs to see the payback, which is 2–3 years.

| Dimension | S&OP | IBP | Consequence of confusing them |
|---|---|---|---|
| Horizon | 3–18 months | 24–36 months | Capital decisions taken in a volume meeting |
| Cadence | Monthly | Monthly, with quarterly strategy input | Strategy reviewed at operating speed |
| Product scope | Existing and near-launch items | Full portfolio pipeline and lifecycle | Launches appear without a business case |
| Financial scope | Revenue, margin, cash for the horizon | Plus capital, funding, and full profit and loss (P&L) | Capex approved without a demand view |
| Participants | Sales, operations, supply chain, finance | Plus strategy, product, HR, and the board cycle | The people who own the constraint are absent |
| Granularity | Family × plant × month | Family or platform × quarter | False precision over 36 months |
| Primary output | Plan of record | Plan of record plus the strategic plan link | Two plans that never meet |
| Failure if absent | Service and inventory problems | Wrong capacity, wrong portfolio, wrong capital | — |

Three things IBP adds, and none of them is "more S&OP". **Strategy:** the plan is tested against
strategic choices rather than only against the budget. **Portfolio:** products are planned across
their lifecycle, including retirement and substitution, over a horizon in which a launch and its
successor both matter. **Finance:** capital and funding enter the same conversation as volume, so a
demand assumption and an investment decision cannot sit in different documents.

The common failure is relabelling. A company renames its S&OP meeting "IBP", adds a strategy slide,
and changes nothing about horizons, participants, or decisions. The test is not the name. It is
whether a decision exists that could not be taken in an 18-month process. If there is none, the
company has S&OP with a new title, and it has spent a year of meeting time to learn that.

### 5.4 Step 1 — Product review

**Owns:** the portfolio. **May change:** introduction and retirement dates, substitution mappings,
item status codes. **May not change:** total volume, capacity, or price.

The step is where the item master meets the plan. Every launch changes the bill of materials, the
routing, the safety stock, and the forecastability of the family. Treating it as a marketing update
rather than a planning gate is the most common structural defect in S&OP implementations.

The characteristic failure is that the step is skipped because "we have no new products this
quarter", and a delayed launch surfaces in the demand review instead. The demand number then
contains units of a product that cannot be built, and the supply review spends its hour explaining
why. Product review is cheap; skipping it is expensive.

### 5.5 Step 2 — Demand review

**Owns:** the demand number. **May change:** volume, mix, timing, channel split, within the evidence
participants can produce. **May not change:** capacity, price, or the supply plan.

A well-run demand review has three properties. It starts from the statistical baseline and requires
a reason to depart. It reviews exceptions rather than every item, because reviewing everything
consumes the meeting and produces no decisions. It records overrides with reason codes so FVA can be
computed by reason later.

The characteristic failure is that the review becomes a negotiation between sales, who want a low
number to protect delivery performance, and supply chain, who want a high number to protect
inventory. The number reflects seniority rather than evidence, and the assumption log is empty, so
the error cannot be diagnosed afterwards.

### 5.6 Step 3 — Supply review

**Owns:** the supply number. **May change:** production quantities, sourcing allocation, overtime,
outsourcing, expediting, inter-plant transfers, and inventory build or drawdown, within existing
capacity. **May not change:** the demand number.

When supply cannot meet demand, the review produces a gap and names the binding constraint. Editing
demand to match supply is the cardinal sin, because it converts a visible problem into an invisible
one. The constraint also deserves a name. "Capacity" is not an answer. "Line 2 at 38,500 units in
April because the V2-Pro changeover consumes two days" is an answer. A named constraint can be
priced, negotiated, or accepted; an unnamed one becomes an argument.

The characteristic failure is presenting the unconstrained plan and calling it the plan. This
happens because constrained planning is harder and the unconstrained plan is more comfortable. The
consequence appears later as a service failure nobody predicted, on data that was available.

### 5.7 Step 4 — Financial reconciliation

**Owns:** the valuation. **May change:** price assumptions, margin, phasing, capital allocation, and
the currency of the gap. **May not change:** volume or the supply plan. Finance may not absorb a
volume gap by adjusting phasing and calling it closed.

The artefact is one number set — volume, revenue, margin, cash, inventory value — with the gap to
budget decomposed into volume, mix, price, cost, and phasing. The decomposition is the analytically
interesting part. A revenue gap of −€62,000 might be −€153,000 of volume at budget mix, +€68,400 of
mix from a launch moving out of the month, and +€22,600 of price. Those three imply completely
different responses; a single net figure implies none.

The characteristic failure is a parallel forecast. Finance runs its own model in a spreadsheet and
the executive review receives two numbers with no reconciliation. The value of one number set is not
accuracy; it is the elimination of a meeting about whose number is right.

### 5.8 Step 5 — Executive review

**Owns:** the trade-off. **May change:** targets, priorities, allocation rules, investment release,
the acceptable service level for a named customer, and the rule that produced the problem. **May
not change:** the arithmetic or the definitions. An executive review that re-opens the demand number
has destroyed the four gates before it, and the organisation will learn to bring unresolved numbers
to the top rather than resolve them below.

Three tests distinguish a decision meeting from a review meeting:

1. **Is there a written option that was rejected?** If every option was approved, there was no
   trade-off.
2. **Is there a named owner for the consequence?** "The business" is not an owner.
3. **Is there a trigger that would reopen the decision?** Without one, the decision will be reopened
   informally the first time it is inconvenient.

### 5.9 Consensus and forecast value added

The demand review is where judgment enters the plan, so it is where most value is won or lost.

**Consensus as governance.** A consensus process is not a vote. It is a structured disagreement
ending in one number, with three records: the number, the assumptions, and the dissent. The dissent
is the only way to learn afterwards whether the process or the market was wrong.

**The correlated-error problem.** The argument for averaging opinions is the wisdom of crowds:
independent estimates, averaged, beat individual ones. The condition is independence. In a demand
review the participants are not independent — same pipeline, same meetings, same bonus pool. Their
errors are correlated, so averaging removes little error while creating an impression of rigour.
Worse, a group converging on an optimistic reading reinforces it: each participant reads the others'
confidence as evidence.

> **Note:** the popular claim that "averaging opinions is worse than the naive forecast" is not
> supported as a general statement. The evidence is narrower: adjustment improves accuracy for only
> about half of items, small adjustments often add nothing, upward adjustments are more likely to
> hurt than downward ones, and group processes can entrench a shared bias. See sections 9 and 11.

**FVA as audit.** FVA measures whether a process step improved or degraded the forecast, relative to
a defined baseline and metric.

```text
FVA(step) = error(baseline) - error(candidate)
```

| Step | Forecast | MAPE vs actual | FVA vs previous |
|---|---|---:|---:|
| Naive | Last year same month | 9.4% | — |
| Statistical | Model output | 6.1% | +3.3 pp |
| Judgmental | After planner override | 6.8% | −0.7 pp |
| Consensus | After demand review | 6.5% | +0.3 pp |

Read it honestly: the model adds a great deal over naive, the planner's override subtracts value, and
consensus recovers a little of what the override lost. That is typical and uncomfortable. It does
not mean planners are incompetent. It means the override process has no feedback loop, so planners
cannot learn which adjustments help.

| Adjustment type | Typical effect on accuracy | Why |
|---|---|---|
| Small upward (<5%) | Often harmful | Within noise; adds optimism without information |
| Large upward | Mixed | Sometimes real information, often escalation pressure |
| Small downward | Mixed | Often a correction of a known bias |
| Large downward | Often helpful | Usually reflects information the model lacks |

FVA is also a design tool. Once it can be computed by planner, reason code, and item class, you can
find the adjustments that work and change the process rather than exhorting people to forecast
better.

### 5.10 The S&OE inner loop

S&OE runs weekly over zero to thirteen weeks at SKU level, keeping the committed
month on track. Its decision set is narrow by design.

| Dimension | S&OP (outer loop) | S&OE (inner loop) |
|---|---|---|
| Horizon | 3–18 months (IBP: 24–36) | 0–13 weeks |
| Cadence | Monthly | Weekly, sometimes daily |
| Granularity | Family × plant × region | SKU × customer × order × line |
| Owner | CSCO, CFO, general manager | Planning manager, scheduler |
| Decides | Volume, inventory target, capacity reservation | Sequence, allocation, expedite, substitution |
| May not decide | Order-level allocation | The month's volume or inventory target |
| Metric | Forecast accuracy, margin, inventory coverage | On-time in full (OTIF), schedule adherence, expedite cost |
| Typical failure | Becomes a presentation | Becomes a firefight with no escalation path |

The interface is the design problem. S&OE generates two kinds of exception. **Execution exceptions**
are fixable inside the week — a late inbound, a breakdown, a short pick — and never reach S&OP.
**Structural exceptions** require changing a commitment — capacity persistently short, an inventory
target structurally wrong, a customer systematically above plan — and escalate with evidence as a
pattern, not an anecdote.

The commonest failure is that everything escalates, and S&OP becomes a firefighting meeting that
stops doing the one thing only it can do. The second is that nothing escalates, and structural
problems are absorbed week after week by expediting until the cost becomes normal and invisible. A
useful rule: S&OE may spend up to a stated weekly threshold to protect the plan; above it, it must
escalate. The threshold turns an argument about authority into a parameter, and makes the cost of
protecting the plan visible.

### 5.11 The planning calendar as an artefact

| Offset | Gate | Cut-off | Required input | Owner | Output |
|---|---|---|---|---|---|
| D−14 | Cycle opens | — | Prior decision log, calendar | Process owner | Cycle brief |
| D−13 | Data cut-off | 17:00 D−13 | History, orders, pipeline, capacity | Planning analytics | Frozen snapshot |
| D−12 | Product review | 17:00 D−11 | Launch status, end-of-life, substitution | Product management | Item status |
| D−9 | Demand review | 17:00 D−8 | Baseline, pipeline, promotions, FVA | Demand planning | Demand plan, assumptions |
| D−6 | Supply review | 17:00 D−5 | Demand plan, capacity, material, transport | Manufacturing | Constrained plan, gap |
| D−3 | Pre-S&OP | 17:00 D−2 | Both plans, prices, costs, budget | Finance | One number set, decomposition |
| D0 | Executive review | Pre-read D−2 | One number set, options, recommendation | General manager | Plan of record, decisions |
| D+1 | Publish | — | Signed plan | Process owner | POR v(n), distributed |
| D+2 | S&OE handover | — | Plan of record, open orders | Planning manager | Weekly S&OE agenda |

Four properties separate a calendar from a habit. Every gate has a cut-off, not just a date, because
a gate without one accepts inputs during the meeting and the analysis presented is already stale.
Every cut-off has a destination for late data. Every output has a version identifier: "the plan" is
not a version, "POR-2026-04-v3, signed 12 March" is. And the calendar is published for the year,
because one negotiated monthly is a symptom of a process that is not owned.

### 5.12 What happens to a decision that misses its gate

| Fate | What happens | Cost | When acceptable |
|---|---|---|---|
| Wait | Queues to the next cycle | One cycle of latency; the opportunity may expire | The decision's half-life is longer than the cycle |
| Jump | Escalates out of sequence to a senior forum | Legitimacy of the gate; rework; precedent | The decision is material and irreversible |
| Leak | Decided informally, outside the process | Shadow plan; two truths; the official plan becomes decorative | Never |

The table converts "the process is not followed" into a specific question: which of the three is
happening, and to which decisions?

Leak is the dangerous one, and it usually has a benign origin. A plant manager and a sales director
agree to shift 2,000 units between months because each is measured on their own month. The agreement
is sensible locally, invisible centrally, and means the plan of record no longer describes the
business. Six months later forecast accuracy looks terrible and nobody can say why: the forecast was
accurate and the execution was elsewhere.

A fourth, subtler fate deserves a name: **the decision is taken correctly but recorded nowhere**. It
happened, it was legitimate, and it cannot be audited. This is what makes AI governance hard, because
an agent that acts without a decision record has leaked by design.

## 6. Worked example

One company, one product family, one planning month. All figures are illustrative but internally
consistent.

### 6.1 The company and the family

Halden Home is a European small-appliance manufacturer with revenue of about €480 million. We follow
the cordless vacuum family V2: a base SKU and a premium variant, V2-Pro. The cycle runs in March 2026
for April 2026; the executive review is 12 March.

| Parameter | Value | Unit |
|---|---:|---|
| Base SKU net price | 85 | EUR per unit |
| Base SKU standard cost | 61 | EUR per unit |
| Base SKU contribution | 24 | EUR per unit |
| V2-Pro net price | 110 | EUR per unit |
| V2-Pro standard cost | 72 | EUR per unit |
| V2-Pro contribution | 38 | EUR per unit |
| April capacity, two lines | 38,500 | units |
| Opening finished goods, 1 April | 2,600 | units |
| Retail contract fill-rate requirement | 98 | percent |
| Shortfall penalty | 2.50 | EUR per unit short |

### 6.2 The starting numbers

The planning system produces a statistical baseline of 42,000 units for April, of which 1,800 are
V2-Pro. The FVA record for the family is unflattering: over twelve months, judgmental overrides had
a mean FVA of −3.1 percentage points of MAPE, and 61% of overrides made accuracy worse. The worst
were upward adjustments by sales in the first week of the cycle.

### 6.3 Product review

Engineering reports that the V2-Pro tooling failed its 1,000-hour reliability trial. Launch slips
from 1 April to 3 June. V2-Pro is removed from April and added to June.

```text
April demand = 42,000 - 1,800 = 40,200 units
June demand  = 2,000 + 1,800 = 3,800 units
```

This gate could not decide to launch anyway, and could not compensate for the lost volume by raising
the base forecast. Both would be demand decisions at the wrong gate.

### 6.4 Demand review

Sales proposes a 12% uplift on the remaining base, citing a verbal listing commitment from a German
electronics retailer for 2,500 units per month. The planner applies three tests. Is there a purchase
order? No. What is the retailer's historical sell-through for a comparable SKU in its first quarter?
About 1,600 units per month. What does the FVA record say about upward overrides here? −3.1 points on
average.

**Decision:** accept 1,600, reject 900. Assumption recorded: "German listing, verbal, no PO,
sell-through benchmark 1,600; review at the May cycle."

```text
April demand plan = 40,200 + 1,600 = 41,800 units
```

The gate could not change capacity or price, and could not accept the full 2,500 because sales was
confident. Confidence is not evidence; the FVA record is the evidence about confidence.

### 6.5 Supply review

```text
available = capacity + opening finished goods
          = 38,500 + 2,600 = 41,100 units
gap       = 41,800 - 41,100 = 700 units short
```

The binding constraint is line 2, which loses two days to the V2-Pro changeover that is now cancelled
but not yet re-sequenced. Four options are costed.

| Option | Units covered | Cost | Consequence |
|---|---:|---:|---|
| Overtime, line 2 | 500 | €2,100 | Line 2 overtime is 60% committed to family V1; 500 is the residual |
| Air freight from contract plant | 700 | €4,760 | Nine days transit; €6.80 per unit premium |
| Pull forward from May build | 700 | €0 | May demand is 44,200 against 42,000 capacity; May is already 2,200 short, so this makes it 2,900 |
| Allocate the shortfall | 0 | €8,470 expected | Contract penalty €1,750 plus expected loss of 40% of 700 units at €24 |

**Decision:** 500 units of overtime plus 200 units of air freight.

```text
expedite cost = 500 x 4.20 + 200 x 6.80 = 2,100 + 1,360 = 3,460 EUR
```

The pull-forward rejection is the most instructive moment in the example. It looks free because no
invoice is attached. It is not free: it relocates the gap to May, where the gap is already larger.
Most "solutions" proposed in S&OP meetings are gap relocation, and the test is always the same —
compute the receiving month before accepting the transfer.

### 6.6 Financial reconciliation

```text
Budget April (42,000 units: 40,200 base + 1,800 V2-Pro)
  revenue      = 40,200 x 85 + 1,800 x 110 = 3,417,000 + 198,000 = 3,615,000
  contribution = 40,200 x 24 + 1,800 x 38  =   964,800 +  68,400 = 1,033,200

Plan April (41,800 units, all base)
  revenue      = 41,800 x 85 = 3,553,000
  contribution = 41,800 x 24 = 1,003,200
  expedite     = 500 x 4.20 + 200 x 6.80 = 3,460
  contribution after expedite = 1,003,200 - 3,460 = 999,740

Gaps to budget
  revenue gap      = 3,553,000 - 3,615,000 = -62,000   (-1.7%)
  contribution gap =   999,740 - 1,033,200 = -33,460   (-3.2%)
```

The decomposition explains why the contribution gap is smaller than the revenue gap. The 1,800
V2-Pro units carried €38 of contribution against €24 for the base. Moving them out costs revenue at
€110 per unit but contribution at only €38, and the base volume that partly replaced them earns the
lower margin. The revenue gap is a mix story as much as a volume story, and a single net number would
have hidden it. June is the mirror image: 1,800 units return at €198,000 revenue and €68,400
contribution, but only if the 3 June gate is met.

### 6.7 Executive review

| Option | April contribution | Service outcome | Cash effect | Risk |
|---|---:|---|---|---|
| A. Expedite 700 (recommended) | €999,740 | Retail fill at 100% of plan | €3,460 cost in April | Overtime capacity is tight |
| B. Allocate 700 | €1,003,200 minus €6,720 expected | Retail fill 97.1%, penalty €1,750 | No expedite cost | Shelf-space review at the retailer |
| C. Reduce the demand plan by 700 | €986,400 | Plan met by definition; customers unserved | None | Destroys the credibility of the demand gate |

Option C is editing demand to match supply. It is in the pack precisely so it can be rejected in
public. A process that never shows the forbidden option cannot demonstrate it has refused it.

**Decision:** Option A, with two governance decisions attached.

1. **Launch gate.** V2-Pro launches on 3 June only if engineering signs off the 1,000-hour trial by
   15 May. If the trial fails, the family plan is re-cut in the June cycle and the financial impact
   restated in the same cycle.
2. **Override rule.** A sales override above 5% of the statistical baseline requires a purchase order
   or a signed listing. Verbal commitments are accepted up to 3%. Effective in the April cycle.

### 6.8 What actually happened

April demand came in at 40,900 units, 900 below plan and 1,100 below the original baseline.

```text
baseline error = |42,000 - 40,900| / 40,900 = 2.69%
final error    = |41,800 - 40,900| / 40,900 = 2.20%
FVA            = 2.69 - 2.20 = +0.49 percentage points
```

The override helped this month, which is easy to misread. The twelve-month record is −3.1 points. One
month is an observation, not a distribution. The correct statement is: "the override added 0.49
points in April against a twelve-month mean of −3.1; the April result is within the normal variation
of a process that is, on average, value-destroying." Expedite cost came in at €3,564 rather than
€3,460, because a changeover overran and 240 units were flown instead of 200.

### 6.9 What changed because of the meeting

This is the test of whether the cycle did anything. Five things changed.

1. **The April build plan** moved from 38,500 to 39,000 units, plus 200 units flown in. Without the
   meeting, the plan would have been 38,500 and the shortfall allocated.
2. **The V2-Pro launch** acquired a hard gate date and a named engineering sign-off, replacing an
   open-ended "Q2".
3. **The override rule** changed, affecting every future cycle, not just this one.
4. **The financial reconciliation** moved into the same cycle as the product review. Previously a
   launch slip reached the financial forecast one month late.
5. **The demand number** did not change after the gate closed. The 41,800 stood despite two requests
   to revisit it before the executive review.

Item 5 is the quiet one and the most valuable. A gate that holds is worth more than a gate that is
right.

### 6.10 The same month under IBP

An IBP process reaches the same April decision and sees three things S&OP cannot.

- **The slip is not a one-month event.** The 1,800 V2-Pro units move to June and compress the
  platform launch planned for September, because both products share a tooling cell. An 18-month
  S&OP horizon ends before the collision; a 36-month IBP horizon contains it.
- **The constraint recurs.** The line 2 changeover capacity that binds in April is the same capacity
  that will bind in the September launch window. S&OP sees a monthly expedite; IBP sees a capacity
  investment decision that either prevents both or neither.
- **The portfolio case moves.** A second slip weakens the business case for the V3 platform, whose
  capital release is scheduled for the fourth quarter. That is a portfolio decision, not a volume
  decision, and it belongs in IBP.

None of this changes the April decision. It changes the decision's meaning: a 700-unit expedite is a
tactical fix inside a structural pattern that only the longer process surfaces. A company that runs
S&OP well and IBP badly will keep solving April in April.

### 6.11 What the analyst should have asked

1. Does the 38,500 capacity already assume the V2-Pro changeover is cancelled? If it does not,
   cancelling frees two days, roughly 600 units, and the gap is 100 rather than 700.
2. Is the €6.80 air premium fully loaded, including duty, handling, and inbound truck? A freight-only
   premium understates the option by 20–30%.
3. Is the 98% fill-rate penalty per unit short, per order short, or per period below target? The
   three readings differ by an order of magnitude.
4. Is the 1,600-unit acceptance based on sell-through or on the retailer's order pattern? The order
   pattern includes the retailer's own safety stock.

Each question is a process question with an arithmetic consequence. None requires a model.

## 7. Code, query, or calculation

Deliberately short. The useful artefacts in planning are the calendar and the arithmetic.

### 7.1 The calendar as a table you can build

| Offset | Gate | Cut-off | Input owner | Output | Permission |
|---|---|---|---|---|---|
| D−14 | Cycle open | — | Process owner | Cycle brief | None |
| D−13 | Data freeze | 17:00 | Planning analytics | Snapshot v(n) | Freeze only |
| D−12 | Product review | 17:00 D−11 | Product management | Item status | Portfolio |
| D−9 | Demand review | 17:00 D−8 | Demand planning | Demand plan | Demand number |
| D−6 | Supply review | 17:00 D−5 | Manufacturing | Constrained plan | Supply number |
| D−3 | Pre-S&OP | 17:00 D−2 | Finance | One number set | Valuation |
| D0 | Executive review | Pre-read D−2 | General manager | Plan of record | Trade-off |
| D+1 | Publish | — | Process owner | POR v(n) | None |
| D+2 | S&OE handover | — | Planning manager | Weekly agenda | Execution |

Build this table before building any model. The blank cells are your requirements document.

### 7.2 The arithmetic of one cycle, in eighteen lines

```text
budget units        = 42,000        (40,200 base + 1,800 V2-Pro)
budget revenue      = 40,200*85 + 1,800*110 = 3,615,000
budget contribution = 40,200*24 + 1,800*38  = 1,033,200

plan units          = 41,800        (all base, launch slipped)
plan revenue        = 41,800*85              = 3,553,000
plan contribution   = 41,800*24              = 1,003,200
expedite cost       = 500*4.20 + 200*6.80    =     3,460
plan contribution after expedite             =   999,740

revenue gap         = 3,553,000 - 3,615,000  =   -62,000
contribution gap    =   999,740 - 1,033,200  =   -33,460
revenue gap %       = -62,000 / 3,615,000    =     -1.72%
contribution gap %  = -33,460 / 1,033,200    =     -3.24%

actual units        = 40,900
MAPE(baseline)      = |42,000 - 40,900| / 40,900 = 2.69%
MAPE(final)         = |41,800 - 40,900| / 40,900 = 2.20%
FVA                 = 2.69 - 2.20 = +0.49 pp
```

Two checks before trusting any FVA number: confirm the sign convention, and confirm the metric. A
team reporting FVA in MAPE and a team reporting it in WAPE will disagree about the same overrides,
and both will be computing correctly.

## 8. Data, standards, and regulatory context

Planning has no equivalent of the GS1 standards that govern identification, or IFRS 15 that governs
revenue. That absence is the most important fact in this section.

### 8.1 What is standardised

| Standard or framework | Governs | What it constrains in planning | Grade |
|---|---|---|---|
| ASCM SCOR Digital Standard (Supply Chain Operations Reference) | Process taxonomy: plan, source, make, deliver, return | Vocabulary and decomposition; voluntary | B |
| APICS Dictionary | Professional terminology | Definitions of time fence, S&OP, master production schedule | B |
| ISA-95 / IEC 62264 | Integration levels between enterprise and control systems | Which planning layer may write to which execution layer | B |
| ISO 8000 | Data quality | Master and transaction data quality claims | B |
| IFRS 15 / ASC 606 | Revenue recognition | When planned revenue may be recognised, hence how the plan is valued | B |
| Sarbanes-Oxley Section 404 | Internal control over financial reporting | Controls over inputs to guidance, including forecast data, for US-listed firms | B |
| National Retail Federation (NRF) 4-4-5 retail calendar | Retail period definition | Period boundaries for retail planning and comparison | C |
| EU Corporate Sustainability Reporting Directive (CSRD) and ESRS | Sustainability disclosure | Adds emissions and resource data to the planning pack, phased | B |

### 8.2 What is not standardised, and why it matters

No ISO standard and no audited specification defines what an S&OP process must contain. SCOR gives a
taxonomy, not a specification. The APICS Dictionary gives definitions, not requirements. There is no
certification that lets one company's "S&OP" be compared with another's.

Three consequences follow. **Vendor definitions proliferate:** one vendor defines IBP as "S&OP with
finance in the room", another as "S&OP plus portfolio and strategy over 36 months", a third as "our
software module". All three are defensible because there is no authority to appeal to, so when you
read an adoption statistic, ask which definition the respondent used. **Maturity claims are
unfalsifiable:** a company can claim stage four on a self-assessment with no external check.
**Regulation enters through finance, not planning:** the binding constraints are usually
financial-reporting controls, contractual service levels, and sector rules. If your analytics product
feeds the plan, the audit question will come from finance and will be about reproducibility, not
model elegance.

### 8.3 Data requirements the process imposes

| Requirement | Why the process needs it | What breaks without it |
|---|---|---|
| Versioned snapshot per cycle | Two teams must plan from the same world | Irreconcilable meeting, rework |
| Item status by month | Demand must not include unlaunched items | Phantom demand, missed launches |
| Capacity by work centre and week | Supply must be constrained, not aspirational | Unachievable plan, late service failure |
| Standard cost by item and period | Finance must value the same volume | Two revenue numbers |
| Reason code on every override | FVA must be attributable | No learning, no accountability |
| Decision record per gate | The plan must be auditable | Decisions leak and cannot be reviewed |
| Assumption log per cycle | The plan must be diagnosable afterwards | Error cannot be attributed to a cause |

The last three are what analytics teams most often omit, and they are what separates a process that
learns from one that repeats.

The cadence of master data is a planning constraint in its own right. Item status changes weekly,
standard costs monthly, capacity with maintenance and labour, prices on contract dates. A calendar
that refreshes all of them on the same day will be wrong about most of them most of the time. Match
each input's refresh cadence to its rate of change, and state in the calendar which version of each
was used. When two functions disagree about a number, the disagreement is often about vintage rather
than value.

## 9. Industry practice and case evidence

Most quoted planning statistics are self-reported survey data with unstated sampling. This section
separates what is measured from what is claimed.

### 9.1 What companies actually do

| Survey | Fieldwork | Sample | Finding | Grade |
|---|---|---|---|---|
| ABC Supply Chain S&OP/IBP survey | Published 25 Nov 2025 | 164 practitioners, 54 countries, 23 industries | 81.1% run S&OP in Excel or Google Sheets; 6.6% in ERP; 5.1% in dedicated tools (Kinaxis, o9, Anaplan); 3.6% custom or hybrid; 2.1% business intelligence tools; 1.0% no tool | D |
| BearingPoint, "Next Gen Integrated Planning" | Published 27 Jan 2026 | 193 executives, 7 European countries, 8 industries | 8% fully integrated end-to-end across sales, operations, and finance; about 70% rely on informal coordination or disconnected processes; nearly half lean heavily on spreadsheets; fewer than one-third use rolling forecasts; over half do not integrate operational and financial outputs | D |
| ORI / SCPI / Gatepoint | Fielded Sept 2025–Jan 2026 | 100+ respondents, 66% food and beverage manufacturing | S&OP owned by supply chain in 52% of organisations and sales in 19%; only 23% report highly integrated planning; 54% cite volatile demand shifts and 53% cite market fluctuations outpacing forecasting; 45% cite fragmented or non-real-time data as the biggest obstacle | D |

Three cautions apply. **The samples are self-selected.** The ABC survey reports no sampling frame,
no response rate, and no margin of error; people who answer an S&OP survey are, by construction,
interested in S&OP. The 81.1% is best read as "Excel dominates among practitioners who engage with
S&OP content", not as a population estimate. **The definitions differ:** the BearingPoint 8% and the
ORI 23% are not measuring the same construct and should not be averaged. **Two of the three are
vendor-conducted or vendor-commissioned:** ORI sells planning software, ABC sells training. That does
not make the numbers false; it means the framing is chosen.

> **Note:** a consistent signal across three independent samples is more informative than any single
> number. The signal is that the fully integrated, tool-supported, finance-linked process described
> in most textbooks is a minority practice in 2026, and spreadsheet planning remains dominant.
> Design analytics for that reality.

### 9.2 What the research says about judgment and consensus

**Fildes, Goodwin, and De Baets (2025)** pooled approximately 147,000 forecasts from six studies.
Judgmental adjustment improved accuracy or bias for only just over half of the SKUs examined.
Positive (upward) adjustments were more likely to worsen performance; negative (downward) adjustments
generally helped, especially when large. Evidence that forecasters used information unavailable to
the algorithm was weak; they appeared instead to respond to irrelevant or low-diagnostic-value cues.
A debiasing procedure applied to adjusted forecasts improved performance.

This is Grade A evidence and supports a specific conclusion: override processes need measurement and
governance, not exhortation. It does not support the claim that human judgment is worthless, nor the
claim that averaging opinions is worse than naive.

**Oliva and Watson (2007)** documented the Leitax case, in which a consensus process managed
political and informational bias well but could also reinforce shared optimism. The case records two
major over-forecast write-offs and argues that a coordination system designed to fix existing biases
will generate new ones. Grade A for the case; the causal reading is interpretation.

**McNees (1992)** made the methodological point that still governs the literature: an ex-post
consensus average is a synthetic construct not available in real time, so its apparent accuracy is
partly tautological.

The defensible claims are therefore narrower and more useful than the popular ones: overrides
frequently destroy value, concentrated in small and upward adjustments; group consensus can entrench
shared bias because errors are correlated; FVA makes both visible; and "averaging is worse than
naive" is not supported.

The measurement design matters as much as the result. FVA must be computed on the forecast that was
actually used, at the granularity at which it was used, over a period long enough to average out
noise. Three design choices change the answer: whether the baseline is naive or statistical, whether
error is aggregated before or after taking absolute values (WAPE versus MAPE), and whether the
comparison includes bias or only dispersion. A team that reports a single FVA number without stating
all three has not reported a measurement. It has reported a direction of travel.

### 9.3 Maturity models and their limits

The most cited S&OP maturity model is Grimson and Pyke (2007), based on a literature review and 15
company interviews. It defines five stages — non-existent, reactive, standard, advanced, proactive —
across five dimensions: meetings and collaboration, organisation, measurement, information
technology, and plan integration.

Two findings matter more than the stages. **No clear relationship was found between firm size or
process type and degree of integration** — size does not buy maturity. **Information technology was
not clearly a primary driver of plan integration** — the organisational variables were the enablers.
The second finding directly challenges the technology-led view of planning transformation, and is
consistent with the survey picture in 9.1: companies bought tools and still run the process in
spreadsheets, because the constraint was never the tool.

The critique literature applies too. A systematic review of business process maturity models (Smajli,
Feldman, and Cox, 2024) reports recurring limitations: the models are descriptive rather than
prescriptive, they provide no decision calculus for choosing between improvements, and there is
limited empirical evidence that higher maturity improves performance. Work on S&OP measurement
(Hulthén, Naslund, and Norrman, 2017) finds cross-functional trade-off measures are the hardest
measurement problem at every maturity level, and that more mature levels face new problems including
standardisation to support unbiased decision making.

The practical conclusion: **maturity describes the process; decision quality describes the
decision.** A stage-four process can approve a plan nobody believes, and a stage-two process can make
one good decision because the right three people were in the room with the right number.

| Test | Question | Evidence |
|---|---|---|
| Framing | Were at least two genuine alternatives considered? | A rejected option in the record |
| Information | Were the material assumptions stated and tested? | Assumption log with owners |
| Trade-off | Was the constraint being broken named? | Decision record names service, cost, or cash |
| Ownership | Is a named person accountable for the consequence? | A role, not a department |
| Reversibility | Is there a trigger that would reopen the decision? | A stated threshold or date |

### 9.4 Where AI and agentic planning entered, as of 2026

| Claim | Source and date | Grade | Reading |
|---|---|---|---|
| SCM software with agentic AI grows from under $2bn in 2025 to $53bn by 2030; 60% of enterprises adopt agentic features by 2030, from 5% in 2025 | Gartner, 7 Apr 2026 | D | Market forecast, not adoption observation |
| Vendors promising end-to-end autonomous planning before 2027 are overstating; most current agentic capability improves user experience rather than decision quality | Gartner, "agent washing", 20 May 2026 | D | The most useful caution in the 2026 material |
| Only 13% of agentic processes fully autonomous through end-2027; most enterprises rely on human-in-the-loop, more so for consequential decisions | ISG Research, Sept 2026 | D | Consistent with the Gartner caution |
| About 30% of firms report measurable AI value in planning; about 44% adoption | BCG, May 2026 | D | Self-reported; the adoption-value gap is the story |
| An inventory agent for 60 long-tail products scaling to 200, with a separate oversight agent auditing actions and ensuring reversibility; a new "AI owner" role | Zero100 account of a Sanofi pilot, Feb 2026 | C | Company claim about a live pilot; not independently audited |
| An "80/20 touchless" model: about 80% of routine decisions autonomous, 20% escalated with recommendations; the business review becomes a governance forum | EY, Aug 2026 | D | A design pattern, not an outcome |

Three observations. **The governance question is the process question.** "Who signs the plan?" is
not new; it is what gate 5 has always answered. What changes is the volume and speed of decisions
needing a signature, and the fact that an agent can act without leaving a record unless the record
is designed in. **The credible 2026 pattern is bounded agency, not autonomy.** Every working
deployment described has the same structure: routine, reversible, low-materiality decisions executed
inside thresholds; consequential or irreversible decisions escalated to a named human; every action
logged with a rationale; a reversal path. **The failure mode is already visible.** An agent overrides
the statistical forecast, the override is not logged, and FVA cannot see it. The organisation reports
improving accuracy while the plan drifts — the same failure as a human leak, at higher rate.

The design question is therefore concrete: what is the decision-rights table for the agent, and where
is it recorded? Exercise 9 asks you to write one.

### 9.5 What the evidence does not support

- **"Companies with integrated planning outperform."** The surveys show correlation between
  self-reported integration and self-reported satisfaction. Neither is an outcome measure, and both
  are reported by the same person.
- **"Maturity predicts performance."** The maturity literature itself reports weak evidence, and the
  models are descriptive rather than prescriptive.
- **"AI planning delivers measurable value at scale."** The 2026 sources report pilots and
  projections. The BCG figure of about 30% is the most concrete, and it is a self-report.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Executive review re-opens demand | First 20 minutes argue about the forecast | Gate 5 exercises a permission it does not have | Gates 2–4 lose authority | Publish the permission ladder; chair refuses re-litigation |
| Finance runs a parallel forecast | Two revenue numbers in one meeting | Finance does not trust the operational plan | No plan of record; meeting debates whose number is right | One number set signed at pre-S&OP |
| Gate dates slip silently | The cycle "runs late" every month | No cut-off enforcement; late data accepted | Compressed analysis on partial data | Cut-off with a destination; refusal authority |
| Overrides without FVA | Nobody can say whether last year's overrides helped | No reason codes, no baseline, no measurement | Systematic value destruction, invisible | Reason code on every override; quarterly FVA |
| Consensus treated as an average | The number equals the mean of submissions | Averaging mistaken for agreement | Correlated errors survive; nobody owns it | One number plus a dissent record |
| S&OP used to fix the current month | The agenda is full of expedites | No S&OE loop, or S&OE escalates everything | The only forum for next quarter spends time on this week | Separate the loops; escalation threshold |
| Unconstrained plan presented as the plan | Service failures "nobody could have predicted" | Capacity not loaded, or loaded but not believed | Credibility loss; chronic late delivery | Name the binding constraint; show the gap |
| Product review skipped | Launch surprises appear in demand review | "Nothing is launching" assumed, not checked | Phantom demand, missed launches, obsolete stock | Mandatory gate with a null return |
| Maturity assessment as scorecard | Teams optimise the assessment, not the decisions | Descriptive model used as a target | Process theatre; decisions unchanged | Assess decision quality separately |
| Multiple plan-of-record versions | Two plants build to different numbers | No version control; informal distribution | Shadow plan; reconciliation cost; mistrust | Single version identifier, published at D+1 |
| Meeting measures presentation quality | Slide templates standardised; decisions not | Review substituted for decision | No accountability; recurring issues | Decision log; rejected-option count |
| AI agent overrides unlogged | Accuracy improves while service falls | Agent actions not recorded as overrides | FVA blind; drift invisible; audit failure | Log every action with rationale and reversal path |

Two deserve a note. **The parallel forecast** is the most expensive failure on the list, because it
consumes the scarcest resource: executive attention. Once two numbers reach the room, the meeting is
about reconciliation and every other decision waits. The fix is structural — finance owns valuation,
not volume. **The unlogged agent override** is the newest and least understood. It is the same
failure as a human override, but it scales with the agent's throughput and is invisible to FVA unless
logging is designed in. An organisation that deploys an agent without override logging has removed
its own ability to learn.

One pattern connects most rows in the table: the process degrades by absorbing work it was not
designed for. Demand review absorbs capacity concerns, S&OP absorbs S&OE, the executive review
absorbs reconciliation. Each absorption is locally reasonable and collectively fatal, because the
gate that absorbed the work stops doing its own. When you diagnose a planning process, look for the
gate that is busy with the previous gate's job.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A statistical baseline exists at planning granularity | FVA and demand review both need a starting number | New items, sparse data, family-level only | Check baseline coverage by item and month |
| Each gate has one owner with authority to sign | The permission ladder depends on it | Matrix organisations, shared ownership | Ask who signed last cycle's number |
| The calendar is stable across cycles | Comparisons and FVA need consistent periods | Reorganisations, system migrations | Compare gate dates across six cycles |
| Financial conversion is roughly linear in volume | Simplifies reconciliation | Volume discounts, step costs, FX | Test the conversion at two volume levels |
| Capacity is known at family level before supply review | The constrained plan needs it | New products, shared lines, labour shortage | Compare planned and actual utilisation |
| The horizon exceeds the longest lead time | Otherwise the plan cannot be executed | Long ocean lanes, custom components | Plot lead time against horizon by family |
| Data cut-offs are enforceable | Late data destroys the analysis | Senior escalation, cultural norms | Count late inputs per cycle |
| Decisions are recorded | Audit, learning, and AI governance need it | No decision log, minutes only | Check whether last cycle's decisions can be listed |
| S&OE can execute the plan as signed | The plan must be operable | Plan granularity execution cannot use | Compare plan and scheduling granularity |
| Survey respondents share definitions | Cross-survey comparison | Vendor-specific definitions | Read the instrument, not the headline |
| Maturity stage correlates with capability | Maturity models assume it | Descriptive models, self-assessment | Test decision quality independently |

Two limits of the framing are worth stating.

**It describes a governance structure, not a guarantee of outcomes.** A well-run process can produce
a bad plan because the inputs were wrong or the market moved. The process's value is that it makes
the error visible, attributable, and correctable. It does not prevent error.

**It assumes a single enterprise with authority over its own plan.** In a marketplace, a
fourth-party logistics provider (4PL), or a highly outsourced network, the gates exist but authority
is contractual rather than hierarchical.
"Who signs the plan" becomes a question about contract terms, and the process becomes a negotiation
protocol. Module 21 treats these structures.

## 12. Exercises

### Exercise 1 — Draw the cycle (recall, 30 minutes)

Draw the five-step S&OP cycle from memory, with the inner S&OE loop, labelling each gate with the
number it owns and the changes it may make.

**Deliverable:** one page, hand-drawn or text.
**Acceptance criteria:** five gates in order; owned number and permitted change for each; S&OE shown
as a separate loop with at least one thing it may not change; data cut-off before gate 1.

### Exercise 2 — Build a permission ladder (application, 45 minutes)

For a company you know, write the permission ladder: gate, owned number, may change, may not change,
owner.

**Deliverable:** a six-row table plus one sentence per gate explaining why the prohibition exists.
**Acceptance criteria:** each gate owns exactly one number; at least two prohibitions are justified by
a specific failure they prevent; the owner is a role, not a department.

### Exercise 3 — Design a planning calendar (design, 60 minutes)

Design a monthly calendar for a manufacturer with a six-week ocean inbound lead time and a two-week
frozen production zone.

**Deliverable:** a calendar table with D−n offsets and a one-page note on the two most fragile points.
**Acceptance criteria:** every gate has a cut-off; every cut-off has a destination for late data; the
calendar is consistent with the lead time and frozen zone; the two fragile points are named with a
mitigation.

### Exercise 4 — Compute FVA (calculation, 45 minutes)

A product family has eight months of history. Compute FVA per period and on average using MAPE, then
recompute using WAPE.

| Period | Actual | Baseline | Final |
|---|---:|---:|---:|
| 1 | 1,000 | 1,050 | 1,020 |
| 2 | 1,100 | 1,080 | 1,120 |
| 3 | 900 | 980 | 1,050 |
| 4 | 1,200 | 1,150 | 1,180 |
| 5 | 950 | 1,020 | 1,080 |
| 6 | 1,050 | 1,000 | 1,040 |
| 7 | 1,150 | 1,200 | 1,180 |
| 8 | 1,000 | 1,080 | 1,120 |

**Deliverable:** a table of per-period MAPE for baseline and final, FVA per period, mean FVA, and the
WAPE version.
**Acceptance criteria:** correct arithmetic to two decimals; sign convention stated; the difference
between the metrics explained; at least one pattern identified in which periods the override helped.

### Exercise 5 — Diagnose a missed gate (critique, 40 minutes)

A launch slips from April to June. The April demand review still contains the launch volume. The
supply review reports a 1,400-unit capacity gap. The executive review approves expediting, and in
June the launch slips again.

**Deliverable:** a causal chain from the first gate failure to the final outcome, with the cost of
each consequence estimated.
**Acceptance criteria:** the first failing gate is identified, not the last; the cost includes at
least one non-financial consequence; one process change is proposed that would have prevented the
chain.

### Exercise 6 — Design the S&OP–S&OE interface (design, 60 minutes)

Specify the interface for a business with high promotional volatility: what escalates, at what
threshold, in what format, and who decides.

**Deliverable:** an escalation specification with three tiers, a spend threshold, and a template for
a structural exception.
**Acceptance criteria:** the threshold is expressed in money and in service impact; the template
distinguishes a pattern from an anecdote; at least one category is explicitly not escalated.

### Exercise 7 — Audit a maturity claim (critique, 45 minutes)

Find a public claim that a company reached a high S&OP or IBP maturity stage. Assess it against the
five decision-quality tests in section 9.3.

**Deliverable:** a one-page assessment with a verdict on process maturity and a separate verdict on
decision quality, each with evidence.
**Acceptance criteria:** the verdicts are separated; at least one test cannot be assessed from public
information and that gap is stated; no claim accepted on the company's assertion alone.

### Exercise 8 — Write the executive one-pager (design, 50 minutes)

Using the Halden Home case, write the one-page pack for the executive review: one number set, the gap
decomposition, three options, a recommendation.

**Deliverable:** a single page with an options table and a clear recommendation.
**Acceptance criteria:** the forbidden option is included and marked; each option states service and
cash consequences; the recommendation names the constraint being broken; no more than 12 numbers.

### Exercise 9 — Agent decision rights (design, 60 minutes)

Write the decision-rights table for an AI planning agent proposing replenishment orders for a
1,200-SKU portfolio.

**Deliverable:** a table with decision class, value threshold, reversibility, approval requirement,
and log fields.
**Acceptance criteria:** at least three decision classes; every class has a named accountable role;
log fields are sufficient to compute FVA on agent overrides; at least one class is irreversible and
always human-approved.

### Exercise 10 — S&OP versus IBP (analysis, 45 minutes)

For a business with a 26-week product development cycle and a 12-month capital plan, argue whether it
needs IBP or whether S&OP is sufficient.

**Deliverable:** a one-page argument with a recommendation and a list of the decisions IBP would add.
**Acceptance criteria:** the horizon is justified by the development and capital cycles, not fashion;
additional participants and their decisions are named; the cost of the extra process is stated.

## 13. Self-check answers

**Exercise 1 model answer.** Gates in order: data cut-off (freezes the input set, changes nothing),
product review (owns the portfolio), demand review (owns the demand number, may change volume, mix,
timing), supply review (owns the supply number, may change build, buy, move within capacity),
financial reconciliation (owns the valuation, may change price, margin, phasing, cash), executive
review (owns the trade-off, may change which constraint breaks). S&OE runs weekly inside the month,
may re-sequence, expedite, substitute, and allocate, and may not reset the month's volume or
inventory target. A submission placing the cut-off after product review, or letting supply edit
demand, should be marked down regardless of presentation.

**Exercise 4 arithmetic.**

```text
P1  base  50/1000 = 5.000%   final  20/1000 = 2.000%   FVA  +3.000
P2  base  20/1100 = 1.818%   final  20/1100 = 1.818%   FVA   0.000
P3  base  80/900  = 8.889%   final 150/900  = 16.667%  FVA  -7.778
P4  base  50/1200 = 4.167%   final  20/1200 = 1.667%   FVA  +2.500
P5  base  70/950  = 7.368%   final 130/950  = 13.684%  FVA  -6.316
P6  base  50/1050 = 4.762%   final  10/1050 = 0.952%   FVA  +3.810
P7  base  50/1150 = 4.348%   final  30/1150 = 2.609%   FVA  +1.739
P8  base  80/1000 = 8.000%   final 120/1000 = 12.000%  FVA  -4.000

mean baseline MAPE = 44.352 / 8 = 5.544%
mean final MAPE    = 51.397 / 8 = 6.425%
mean FVA           = 5.544 - 6.425 = -0.881 pp
```

WAPE, which weights by volume rather than by period:

```text
sum |baseline - actual| = 50+20+80+50+70+50+50+80 = 450
sum |final    - actual| = 20+20+150+20+130+10+30+120 = 500
sum actual = 8,350

WAPE baseline = 450 / 8,350 = 5.389%
WAPE final    = 500 / 8,350 = 5.988%
WAPE FVA      = -0.599 pp
```

The override helped in four periods (1, 4, 6, 7), was neutral in one (2), and hurt in three (3, 5,
8). All three harmful periods are upward adjustments made when demand fell below baseline. The mean
FVA is −0.88 points simple and −0.60 volume-weighted, so the direction is robust and the magnitude
depends on the metric. State the metric before quoting the number.

**Exercise 2 rubric.** A strong ladder has exactly one number per gate, prohibitions mapped to
specific failures (for example, "supply may not edit demand because it converts a visible capacity
problem into an invisible service failure"), and roles rather than departments. A weak ladder assigns
two numbers to one gate, or writes "shared ownership", which is the absence of an owner.

**Exercise 3 rubric.** A strong calendar back-plans from the executive review, gives every gate a
cut-off at least 24 hours before the meeting, provides an exception log for late data, and reconciles
the six-week inbound lead time with the horizon (the first six weeks are inbound-constrained, not
capacity-constrained). The fragile points are usually finance reconciliation, which depends on a
closed month, and product review, because engineering dates are uncertain. A weak calendar gives
dates without cut-offs, or places supply review before demand review.

**Exercise 5 model answer outline.** The first failing gate is product review, not demand, supply, or
executive. It passed an item into the demand plan whose launch date was not engineering-confirmed.
Demand inherited phantom volume; supply reported a gap for a product that could not be built;
executive approved expediting against an already invalid plan. Costs: the expedite spend, the
opportunity cost of capacity reserved for an absent product, the credibility loss at the second slip,
and components bought for the original date. Process change: product review must return an
engineering-confirmed date or a "not confirmed" status, and an item with that status cannot enter the
demand plan.

**Exercise 6 rubric.** A strong specification has three tiers (routine, exception, structural), a
threshold in currency per week and in service impact, and a template requiring a pattern (at least
four weeks of evidence) rather than a single event. It explicitly does not escalate supplier lateness
inside the agreed lead time, because that would flood the monthly cycle. A weak specification
escalates everything, or sets a threshold with no service dimension.

**Exercise 7 rubric.** A strong assessment separates two verdicts. The maturity verdict rests on
observable artefacts: whether gates have defined inputs and owners, whether the calendar is published,
whether measurement exists at the trade-off level, and whether the plan of record is single and
versioned. The decision-quality verdict rests on a specific decision: alternatives, information,
trade-off, ownership, reversibility. It states at least one test that cannot be assessed from public
information — usually reversibility triggers, which are internal — and declines to score it. A weak
assessment accepts a press release as evidence of maturity, or treats the two verdicts as one claim,
which is the exact confusion the exercise is designed to expose.

**Exercise 8 rubric.** A strong page shows one demand number, one supply number, a decomposed gap,
three options including the forbidden one, and a recommendation naming the constraint being broken.
It contains no more than 12 numbers, because more is a report rather than a decision aid. A weak page
shows volume only, omits the forbidden option, or presents one recommendation with no alternative.

**Exercise 9 rubric.** A strong table distinguishes at least three classes: autonomous execution
(routine, reversible, below a threshold), proposal with approval (material or partly reversible), and
prohibited (irreversible, cross-enterprise, above a ceiling). Every class has a named accountable
role. Log fields include input snapshot, proposed action, accepted action, rationale, threshold
applied, and reversal path — enough to compute FVA on agent overrides after the fact. A weak table
has two classes ("automatic" and "manual"), no thresholds, and no log specification.

**Exercise 10 rubric.** A strong answer recommends IBP when the capital and development cycles exceed
the S&OP horizon. For a 26-week development cycle and a 12-month capital plan, a 24-month horizon
covers at least one full capital cycle and two development cycles. Additional participants: finance
(capital), product management (portfolio), strategy. The cost is about one extra day per month of
senior time plus a heavier data requirement. A weak answer recommends IBP because "it is more
mature", which is a label, not an argument.

## 14. Interview questions

**1. "Walk me through an S&OP cycle."** A strong answer names the gates in order, states the number
each owns, and states what each may not change, mentioning the cut-off and plan of record. A weak
answer lists "demand meeting, supply meeting, exec meeting" without permissions.

**2. "What is the difference between S&OP, IBP, and S&OE?"** A strong answer separates them on
horizon, cadence, granularity, owner, and decision rights (3–18 months, 24–36 months, 0–13 weeks). A
weak answer treats IBP as "S&OP but better" and S&OE as "the daily meeting".

**3. "Why can't S&OP fix the current month?"** A strong answer invokes the frozen zone and
commitments already released to suppliers and customers, and explains that the remaining levers
belong to S&OE. A weak answer says "because it only meets monthly".

**4. "A decision misses its gate. What happens?"** A strong answer gives wait, jump, and leak with
the cost of each, and says leak is most damaging because it creates a shadow plan. A weak answer says
"it waits until next month".

**5. "How would you measure whether forecast overrides add value?"** A strong answer defines FVA
against a named baseline and metric, describes the stairstep, and requires reason codes so the
analysis can be cut by adjustment type. It mentions MAPE versus WAPE. A weak answer says "compare
accuracy before and after".

**6. "Is averaging sales and marketing forecasts a good idea?"** A strong answer distinguishes
consensus from average, explains correlated error, and states that the evidence does not support
"worse than naive". A weak answer either endorses averaging as wisdom of crowds or dismisses it as
always worse.

**7. "What does a mature S&OP process look like, and does maturity matter?"** A strong answer gives
the Grimson and Pyke dimensions and stages, separates maturity from decision quality, and notes the
weak evidence that higher stages improve performance. A weak answer recites the stages as a goal.

**8. "Who should own S&OP?"** A strong answer notes supply chain ownership in about half of
organisations and argues for whoever can arbitrate the demand-supply-finance trade-off, usually the
general manager or CSCO with CFO involvement. It distinguishes process ownership from decision
ownership. A weak answer says "supply chain, because it is a supply chain process".

**9. "How do you know whether an executive S&OP meeting is working?"** A strong answer uses
behavioural tests: a rejected option in the record, a named owner, a reopen trigger, and no
re-litigation of demand. A weak answer says "attendance is good".

**10. "An AI agent proposes replenishment orders. Who signs the plan?"** A strong answer says the
accountable human signs, and the agent operates inside a decision-rights table by value,
reversibility, and materiality, with every action logged and reversible. A weak answer says "the AI
signs if it is accurate enough".

**11. "How would you design analytics so it actually changes the plan?"** A strong answer starts from
the calendar: gate, cut-off, granularity, owner, then designs the output to arrive before the cut-off
with an assumption log. A weak answer starts from the model and assumes adoption follows accuracy.

**12. "What if finance and operations bring different numbers to the executive review?"** A strong
answer treats it as a process defect: pre-S&OP failed to produce one signed number set. The fix is
structural — one set at D−3, finance owns valuation only — and the immediate action is to reconcile
in the room with definitions on the table. A weak answer picks the more credible number.

## 15. Key terms

- **S&OP** — monthly cross-functional demand-supply balancing at family level.
- **IBP** — S&OP extended to strategy, portfolio, capital, and finance over 24–36 months.
- **S&OE** — weekly execution loop inside the committed 0–13 weeks.
- **Gate** — a decision point with an owner, an input set, and one permitted change.
- **Data cut-off** — the moment inputs freeze; its only job is to stop change.
- **Permission ladder** — the table of which gate owns which number and may change what.
- **Plan of record** — the single signed, versioned plan the company executes.
- **Pre-S&OP** — the session that reconciles and frames options before the executive review.
- **Consensus demand plan** — one agreed number with assumptions and dissent recorded.
- **Assumption log** — what had to be true for the plan to be right.
- **Decision log** — what was decided, by whom, with rationale and reopen trigger.
- **Constrained plan** — supply plan respecting capacity, material, and transport.
- **Supply gap** — demand minus constrained supply; a gap, not an error.
- **Binding constraint** — the specific resource limiting supply, named not generic.
- **FVA** — forecast value added, the error change attributable to one process step.
- **Naive forecast** — benchmark assuming the next period equals the last comparable period.
- **Override** — a judgmental change to a system forecast, with a reason code.
- **MAPE / WAPE** — percentage error metrics; simple average versus volume-weighted.
- **Stairstep** — successive comparison of naive, statistical, adjusted, and consensus forecasts.
- **Maturity model** — staged description of process capability, descriptive not prescriptive.
- **Decision quality** — whether a decision was framed, informed, traded off, and owned.
- **Touchless planning** — steps executed without intervention inside defined limits.
- **Human-in-the-loop / on-the-loop** — approve each action versus monitor and intervene.
- **Decision rights** — who may decide what, at which value, with which review.
- **Gap relocation** — moving a shortfall to another period and calling it solved.
- **Shadow plan** — the informal plan execution actually follows.

## 16. Sources

**Peer-reviewed and academic**

- Fildes, R., Goodwin, P., and De Baets, S. (2025). "Forecast value added in demand planning."
  *International Journal of Forecasting*, 41(2), 649–669. DOI: 10.1016/j.ijforecast.2024.07.006.
  Open access. The pooled analysis of approximately 147,000 forecasts in sections 5.8 and 9.2.
- Grimson, J. A., and Pyke, D. F. (2007). "Sales and operations planning: an exploratory study and
  framework." *The International Journal of Logistics Management*, 18(3), 322–346.
  DOI: 10.1108/09574090710835093.
- Oliva, R., and Watson, N. (2007). "Managing Functional Biases in Organizational Forecasts: A Case
  Study of Consensus Forecasting in Supply Chain Planning." Harvard Business School Working Paper
  07-024.
- McNees, S. K. (1992). "The uses and abuses of 'consensus' forecasts." *Journal of Forecasting*,
  11(8), 703–710. DOI: 10.1002/for.3980110807.
- Smajli, A., Feldman, G., and Cox, K. (2024). "Exploring the Limitations of Business Process
  Maturity Models: A Systematic Literature Review." *Information Systems Management*.
  DOI: 10.1080/10580530.2024.2332210.
- Hulthén, H., Naslund, D., and Norrman, A. (2017). "Challenges of Measuring Performance of the
  Sales and Operations Planning Process." *Operations and Supply Chain Management*, 10(1).
- Fildes, R., Goodwin, P., Lawrence, M., and Nikolopoulos, K. (2009). "Effective forecasting and
  judgmental adjustments." *International Journal of Forecasting*, 25(1), 3–23.

**Standards and official**

- ASCM SCOR Digital Standard, Association for Supply Chain Management (current edition).
- APICS Dictionary, current edition. Source of the time-fence and S&OP definitions used here.
- ISA-95 / IEC 62264, Enterprise-control system integration.
- ISO 8000, Data quality.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (FASB).
- Sarbanes-Oxley Act (2002), Section 404.
- Directive (EU) 2022/2464 (Corporate Sustainability Reporting Directive) and the ESRS standards.

**Industry and analyst**

- Gartner, "Predicts 2026: Supply Chain Planning," 28 April 2026.
- Gartner, "Prepare for Agentic AI in Supply Chain Planning Without Betting on Full Autonomous,"
  6 February 2026.
- Gartner press release, supply chain management software with agentic AI to $53 billion by 2030,
  7 April 2026.
- Gartner press release on "agent washing" in supply chain planning technology, 20 May 2026.
- BearingPoint, "From silos to synergy: Next Gen Integrated Planning," press release, 27 January
  2026. 193 executives, seven European countries, eight industries.
- ORI / SCPI / Gatepoint Research, 2026 supply chain planning survey, fielded September 2025 to
  January 2026.
- BCG, "From Copilots to Enterprise Transformation: AI's New Mandate in Supply Chains," May 2026.
- ISG Research, "The State of AI and Agents in Supply Chain Planning," September 2026.
- EY, "Agentic AI in integrated business planning," August 2026.

**Practitioner and vendor, dated**

- ABC Supply Chain (E. Thieuleux), "Why 81% of Companies Still Run Their S&OP Process in Excel,"
  25 November 2025. 164 respondents, 54 countries, 23 industries. Self-selected sample; no sampling
  frame or margin of error reported.
- Zero100, "Agentic AI in Action with Sanofi," February 2026. Company account of a pilot.
- Umbrex, "S&OP Governance and Cadence Design," 2025. Practitioner guidance on cut-offs and gates.

**Note on citations.** Survey figures are reported as the surveys state them and are not
independently verified. Three of the four planning surveys cited are vendor-conducted or
vendor-commissioned and are graded D; the ABC survey reports no sampling frame, response rate, or
margin of error. The Fildes et al. and Grimson and Pyke findings are peer-reviewed and graded A.
Where a claim rests on a vendor framing, the text says so.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_network-entities.md`](03_network-entities.md)
- Next topic: [`05_inventory-and-service.md`](05_inventory-and-service.md)
- Decision layers and time fences: [`02_decision-stack.md`](02_decision-stack.md)
- Entities and product hierarchy: [`03_network-entities.md`](03_network-entities.md)
- Inventory targets set by the plan: [`05_inventory-and-service.md`](05_inventory-and-service.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Return to the part map: [`../_index.md`](../_index.md)
