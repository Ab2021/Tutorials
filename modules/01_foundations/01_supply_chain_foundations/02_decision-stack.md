# The Decision Stack

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01
> Version 2026.5

---

## 1. Why this topic exists

Topic 01 described a supply chain as three flows through one network. That description is
correct and incomplete. Flows do not decide anything. Decisions are made by named people, in
scheduled meetings, using numbers that were prepared to a deadline, under authority that was
granted somewhere else. A forecast does not lower a safety stock. A person does, on a Tuesday,
in a room, against a deadline.

The decision stack is the arrangement of those decisions by horizon, cadence, owner, and
reversibility. It is the organisational counterpart to the physical network. Every company has
one. Very few companies have drawn it.

Three failures recur, and all three are stack failures rather than model failures.

1. **The number that means different things in different rooms.** The annual plan assumes 61,000
   units in April, the monthly plan says 58,000, and the commitment to a retailer implies 59,950.
   Three numbers, three owners, three cadences, no reconciliation. The forecast is not wrong; the
   stack is incoherent.
2. **The replan that destroys the plan.** A planner reruns the schedule every morning because the
   system allows it. Suppliers receive changed signals four days running, stop trusting them, and
   build to their own forecast. Nothing in the model is broken; the cadence is.
3. **The recommendation with no owner.** A weekly stock-reallocation recommendation shows a 6%
   margin improvement in backtest and nobody acts. The reason is not accuracy. The decision it
   implies — moving stock between two regional directors' portfolios — belongs to neither.

Each failure has the same shape: a decision exists, but its horizon, cadence, or owner has not
been designed. Analytics improves a decision only after the decision has an address.

**Why this matters to a data scientist.** Most analytics value is lost in the last mile between
a correct output and a committed action, and the loss is usually attributed to "change
management", a label that stops inquiry. The stack gives a more precise diagnostic: which layer,
who owns it, when it is made, what number is used, and what would have to change for the output
to enter the meeting? If you cannot answer those five questions, you have a model, not a use case.

A model excellent at one layer can be useless or harmful at another. A network optimisation that
minimises five-year cost is right for a strategic review and wrong for a weekly supply meeting,
where the network is fixed. Layer fit is a design property, not a quality property.

This topic sits second because everything after it zooms into one layer or object: the entities
(topic 03), the planning processes (04), inventory and service (05), and cost (06).

> **Note:** The four-layer model in this topic is a teaching construct, not a standard. Real
> organisations have five, six, or an uncountable number of layers, and some decisions sit
> between them. The construct is useful because the questions it forces — horizon, cadence,
> owner, granularity, reversibility, cost of error — are the questions that survive any
> reorganisation.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the four-layer decision stack from memory, labelling for each layer its horizon,
   cadence, decision owner, data granularity, and reversibility.
2. **Assign** a described business decision to a layer, and **justify** the assignment using
   horizon, commitment, and reversibility rather than job title.
3. **Explain** planning time fences, the frozen, slushy, and liquid zones, and material
   requirements planning (MRP) nervousness, and **predict** the operational consequence of
   shortening or removing a fence.
4. **Diagnose** a layer disagreement from a set of conflicting numbers, and **name** the missing
   reconciliation and its owner.
5. **Construct** a RAPID (Recommend, Agree, Perform, Input, Decide) decision-rights map for one
   supply chain decision, and **identify** the failure mode of an unowned decision.
6. **Decompose** decision latency into detection, decision, and execution time, and **compute**
   median and tail latency from a decision log.
7. **Design** an autonomy threshold policy for an automated or agentic recommendation, stating
   the confidence rule, the guardrail, and the escalation path.
8. **Critique** an analytics use case for layer fit, and **state** which meeting, which number,
   and which owner would have to change for it to alter behaviour.

Each outcome maps to an exercise in section 12 and to a self-check item in section 13.

## 3. The core mental model

Hold one picture: four horizontal bands stacked vertically. Authority flows down the stack as
constraint. Information and exceptions flow up the stack as feedback. Each band has its own
clock.

```text
  LAYER        HORIZON        CADENCE        TYPICAL OWNER       CORE QUESTION
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
| STRATEGIC|  1-5 years     annual or      CEO, CSCO,         Where do we put capacity,
|          |  (some 10)     semi-annual    CFO, board         and what do we promise?
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
      |  constrains: nodes, capacity, contracts, service promise
      v
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
| TACTICAL |  3-18 months   monthly        VP planning,       How much do we make, buy,
|  (S&OP)  |                (weekly in     sales, finance,    and hold, month by month,
|          |                disruption)    plant leadership   within the network we have?
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
      |  constrains: monthly volumes, budgets, family mix, headcount
      v
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
|OPERATIONAL|  1-13 weeks    weekly/daily   Planning manager,  Which order, which line,
| (S&OE,   |                (daily MRP)    supply manager,    which truck, this week,
|  MRP)    |                               schedulers         given the month's plan?
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
      |  constrains: sequenced work, allocations, release dates, routes
      v
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+
| EXECUTION|  minutes-7 d   real time /    Dispatch, dock,    What do we do in the next
|          |                shift / daily  line, carrier,     hour, given what just
|          |                               service desk       happened?
+----------+  +-----------+  +-----------+  +----------------+  +---------------------------+

      ^                                                                     |
      |  feedback: actuals, exceptions, variance, capacity truth           |
      +---------------------------------------------------------------------+
```

Five properties of the picture explain most of what follows.

**Property 1 — Authority flows down, information flows up, and the two are never in phase.**
The strategic layer sets the network in September for a year that will be executed in daily
dispatches twelve months later. By the time execution sees the error, the commitment is sunk.
This is the decision-stack version of the flow mismatch in Topic 01: inventory stores an
information delay, and a plan stores a commitment delay.

**Property 2 — Each layer's cadence should be set by the half-life of its decisions.** The half-life
is the time after which acting on a decision is worth materially less: days for a promotion
response, years for a distribution-centre (DC) location. Cadence should be shorter than the
half-life and longer than the time needed to execute. Monthly sales and operations planning
(S&OP) is too slow when the demand signal changes weekly; daily replanning is too fast when the
supplier lead time is six weeks, because most new information cannot be acted on.

**Property 3 — Commitment rises as you move down; reversibility falls.** A strategic decision is
slow, expensive, and hard to reverse; an execution decision is fast, cheap, and easy to reverse.
The layers therefore cannot be governed by the same rules. The cost of an error is not simply
proportional to the layer either: a strategic error costs more per decision, but thousands of
execution decisions per planner per year can make the annual aggregate comparable.

**Property 4 — Disagreement between layers is normal; unresolved disagreement is the pathology.**
Every company has a strategic assumption, a tactical plan, and an operational schedule that
differ. The question is whether a named process reconciles them and which number is the plan of
record. Three numbers and one reconciliation is healthy; three numbers and none is three
businesses.

**Property 5 — Analytics value depends on layer fit, not model quality.** The test is not "is the
model accurate" but "does this output enter a decision with the right horizon, at the right
granularity, in time to change the commitment".

### 3.1 Time fences and the three zones

Inside the planning horizon, most manufacturing planning systems define fences — dates that
govern what may change without human approval.

```text
   past            today                                    planning horizon
     |               |                                             |
     |  FROZEN       |        SLUSHY            |      LIQUID      |
     |  committed    |   limited change,        |  system may      |
     |  to orders;   |   trade-offs negotiated  |  reschedule      |
     |  change needs |   between sales and      |  freely within   |
     |  senior       |   manufacturing          |  the plan        |
     |  approval     |                          |                  |
     |               |                          |                  |
     +---------------+--------------------------+------------------+
     ^               ^                          ^
   release        demand time              planning time
   fence          fence (DTF)              fence (PTF)
```

The terms come from the APICS dictionary, the standard reference vocabulary of the profession.

- **Demand time fence (DTF)** — inside this date, the forecast is no longer added to customer
  orders in demand calculations; only actual orders count.
- **Planning time fence (PTF)** — inside this date, the system will not automatically reschedule
  planned orders; changes require the master scheduler or higher authority.
- **Release time fence** — inside this date, orders are released to the shop floor and cannot be
  changed without a formal deviation.

The zones are the operational consequence. The **frozen zone** is committed to specific orders
and capacity. The **slushy zone** allows negotiated trade-offs. The **liquid zone** is where the
system is free to replan.

### 3.2 Nervousness

**Nervousness** is instability in the plan: small changes in input produce large, repeated
changes in the schedule. It is the characteristic failure of an un-fenced material requirements
planning (MRP) system. A forecast revision of 2% on one component can ripple into rescheduled
orders across hundreds of part numbers, each generating exception messages, each requiring a
planner decision, most of which are noise.

The classical treatment is Blackburn, Kropp, and Millen (1986), which compares strategies for
damping nervousness and finds that freezing part of the schedule and attaching a change cost to
rescheduling are generally effective. The practical reading for an analyst is blunt: **a plan
that changes constantly is not a better plan, it is an unowned one.** Replanning frequency is a
design parameter with a cost, and the cost is paid by the people downstream who must execute.

Nervousness has three common causes, and each has a different fix:

| Cause | Mechanism | Fix |
|---|---|---|
| Forecast churn | Every forecast revision triggers a full regeneration | Freeze inside the cumulative lead time; bucket forecast changes |
| Lot-sizing instability | Order quantities flip between two lot-size rules as demand crosses a threshold | Stabilise lot-sizing; add a change cost |
| Data noise | Small inventory or lead-time errors create phantom shortages | Fix data quality before tuning the planner |

### 3.3 Where AI enters the stack

Automation does not remove the stack. It moves work between layers and changes who approves.

At the execution layer, automation is already normal: routing, dispatch, exception triage. At
the operational layer, optimisation is normal but approval usually remains human. At the
tactical layer, machine-generated plans are common and machine-generated commitments are not.
At the strategic layer, models inform and humans decide, because the decision encodes the
company's promise and its risk appetite.

The 2026 discussion of agentic planning is largely a discussion about moving the approval
boundary, not about moving the computation boundary. A model that can compute a replenishment
plan has existed for decades. An agent that can commit a purchase order needs a decision right,
a guardrail, an audit trail, and an escalation path. Section 9 returns to the evidence on how
far companies have actually moved.

### 3.4 Reading the stack in an unfamiliar company

When you join a company or start a project, the stack is invisible until you ask for it. Six
questions recover most of it in a week, without a systems integration.

1. **Who can commit without escalation?** Ask three people at different levels what they can
   approve. The answers locate the real decision rights, which often differ from the org chart.
2. **Which meeting authorises the monthly plan?** If no meeting does, the plan is a forecast and
   the functions are each executing their own.
3. **What is the plan of record, and where is it published?** Ask for the document. If there are
   several, ask which one finance uses.
4. **How many plans exist?** Count the spreadsheets that planners maintain outside the system.
   Each one is a hedge against a broken handoff.
5. **Where are the fences?** Ask a scheduler what they may change without asking, and what
   happens when they do ask.
6. **What is the decision latency?** Ask how long passed between the last three major exceptions
   and the action taken. The answers are usually longer than anyone expects.

The output is a one-page map: layer, decision, meeting, owner, number, cadence, and fence. It is
the most useful onboarding document an analyst can produce, and almost no company has one.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Decision stack | The layered arrangement of supply chain decisions by horizon, cadence, and authority | Used to mean the software stack | Four-layer ordinal, course construct |
| Strategic decision | A decision that changes network structure, capacity, or the service promise | Used for any decision made by a senior person | Horizon in years |
| Tactical decision | A decision that allocates capacity and volume within the existing network | Confused with strategic because both are "planning" | Horizon in months |
| Operational decision | A decision that sequences, allocates, or releases work inside the plan | Confused with execution because both are short-term | Horizon in weeks or days |
| Execution decision | A real-time decision to act on an event | Used for any daily task | Horizon in hours or minutes |
| Horizon | How far forward a decision looks | Confused with cadence | Months or years |
| Cadence | How often a decision is revisited | Confused with horizon | Meetings per year |
| Planning bucket | The time unit a plan is built in | Treated as the same as the reporting period | Day, week, month |
| Planning cycle | One complete pass of a planning process | Used to mean the lead time of planning | Days |
| Time fence | A date inside which changes follow special rules | Used loosely for any deadline | Date |
| Demand time fence (DTF) | Date inside which only actual orders count as demand | Confused with planning time fence | Date |
| Planning time fence (PTF) | Date inside which the system will not auto-reschedule | Assumed to block all change | Date |
| Release time fence | Date inside which released orders cannot change | Omitted from planning discussions | Date |
| Frozen zone | Committed segment of the horizon | Used to mean "no changes allowed at all" | Date range |
| Slushy zone | Segment where trade-offs are negotiated | Treated as free to change | Date range |
| Liquid zone | Segment the system may replan freely | Treated as unimportant | Date range |
| Nervousness | Plan instability caused by small input changes | Used to mean forecast error | Exception messages per cycle |
| Replanning | Regenerating the plan | Assumed to be costless | Runs per period |
| Plan of record | The single plan the organisation has agreed to execute | Used loosely for "the latest plan" | Document version |
| Consensus forecast | The single demand number agreed across functions | Confused with the statistical forecast | Units per bucket |
| Demand review | Meeting that agrees the unconstrained demand view | Confused with the sales forecast | Meeting |
| Supply review | Meeting that tests demand against capacity and supply | Confused with scheduling | Meeting |
| Pre-S&OP | Meeting that reconciles demand, supply, and finance before executives see options | Skipped in immature processes | Meeting |
| Executive S&OP | Meeting where the business plan is authorised | Confused with a reporting review | Meeting |
| Sales and operations planning (S&OP) | Monthly tactical process aligning demand, supply, and finance | Used to mean any planning meeting | Monthly cycle |
| Sales and operations execution (S&OE) | Short-cycle process managing the near-term plan and exceptions | Treated as a subcommittee of S&OP | Weekly or daily cycle |
| Integrated business planning (IBP) | S&OP extended to the full financial plan and longer horizon | Used as a synonym for S&OP | Monthly or quarterly cycle |
| Annual operating plan (AOP) | The budget and volume target for the fiscal year | Confused with the forecast | Currency and units per year |
| Material requirements planning (MRP) | Explosion of demand through the bill of materials into component orders | Used to mean the whole planning process | Planned orders per run |
| Master production schedule (MPS) | The schedule of what will be built and when | Confused with the production plan | Units per bucket |
| Available to promise (ATP) | Uncommitted supply that can be promised to a customer | Confused with on-hand stock | Units by date |
| Decision right | The authority to commit the organisation to an action | Confused with responsibility for the outcome | Role, not department |
| RAPID | Recommend, Agree, Perform, Input, Decide — a decision-role framework | Treated as a sequence of steps | Five role codes |
| RACI | Responsible, Accountable, Consulted, Informed — a task-role framework | Used for decisions rather than tasks | Four role codes |
| Decision latency | Elapsed time from signal to committed action | Used to mean system response time | Hours |
| Decision quality | The expected value of a decision given the information available at the time | Judged by outcome, which is noisy | Expected value, currency |
| Autonomy level | The degree to which a system may act without human approval | Treated as binary | Ordinal 0–4 |
| Guardrail | A hard constraint that bounds an automated action | Confused with a target | Constraint, with units |
| Human in the loop | A person approves or reviews an automated decision | Used to mean any human involvement | Approval pattern |
| Decision governance | The rules for who may act, on what evidence, with what audit | Used to mean data governance | Policy |
| Bounded autonomy | Automation allowed to act inside a defined envelope | Confused with full autonomy | Envelope definition |
| Agent washing | Labelling ordinary automation as agentic AI | Used as an accusation rather than a test | Claim category |
| Rollover | Moving the planning horizon forward by one bucket | Confused with replanning | One bucket |

Four of these deserve expansion.

**Forecast, plan, target, and commitment.** These four words recur in every topic of this
program because they are the most commonly conflated objects in supply chain meetings. A
forecast is what you believe will happen. A plan is what you intend to do. A target is what you
would like to achieve. A commitment is what you have promised. Only the plan is under your
control. Only the commitment carries a penalty. In the stack, the forecast lives at the tactical
and operational layers, the plan is the output of each layer, the target lives in the annual
operating plan, and the commitment lives wherever a customer or supplier has a contract. A
meeting that debates "the number" without saying which of the four it is will not converge.

**Decision right versus responsibility.** A decision right is the authority to commit. A
responsibility is the duty to deliver. They are often separated on purpose: a category manager
may recommend a supplier award, a procurement director may decide it, and the plant manager may
be responsible for the resulting quality. Confusing the two produces the worst outcome in
organisational design — a person accountable for an outcome they cannot authorise.

**Decision latency versus system latency.** System latency is how long a query or a model takes.
Decision latency is how long the organisation takes from the moment a signal exists to the
moment an action is committed. A company can have sub-second dashboards and a nine-day decision
latency, and the second number is the one that determines service.

**Autonomy level versus automation level.** Automation asks whether a task is performed by a
machine. Autonomy asks whether the machine may commit without approval. A fully automated
replenishment calculation at autonomy level 1 still requires a planner to release the order.
The distinction matters because most vendor claims describe automation and imply autonomy.

## 5. The mechanics, step by step

This section walks one planning month through all four layers, naming at each step the input
number, the owner, the output artifact, and the fence that governs it. The purpose is to build
the reflex of asking, for any decision, "which layer is this, and what is the cadence?"

### Phase A — The strategic layer sets the frame

**Step 1. Refresh the assumptions.**
- Input: long-range demand outlook, input-cost curves, capacity costs, tariff and trade
  assumptions, service promise.
- Owner and cadence: strategy or network design with finance; annual, aligned to the budget
  cycle, revisited after a structural shock.
- Output and artifact: an assumption set with named owners and review dates, held in a network
  model and an assumption register.
- Key branch: **is the assumption owned or borrowed?** An assumption with no owner becomes a
  fact by default.

**Step 2. Decide the footprint and the promise.**
- Input: the assumption set, candidate sites, lane costs, tax and duty, labour availability.
- Owner and cadence: chief supply chain officer (CSCO) with the chief executive, board approval
  above a threshold; annual or multi-year, triggered early by a structural event.
- Output and artifact: nodes, capacity, contracts, and the service promise by market, with a
  business case carrying net present value (NPV) and a sensitivity range.
- Key branch: **is the decision reversible?** A leased DC is reversible in three years; an owned
  plant is not. Reversibility should determine how much analysis the decision deserves.

**Step 3. Set the constraints the rest of the stack will inherit.**
- Input: the approved network, capacity, headcount, and contracts.
- Owner and cadence: finance and operations jointly; annual, with a mid-year reforecast.
- Output and artifact: capacity by site and period, budget envelope, service and inventory
  targets, recorded in the annual operating plan (AOP) — a target and a budget, not a forecast.
- Key branch: **is the AOP being used as a forecast?** Using it as the demand expectation biases
  every downstream forecast toward the number the company wants.

### Phase B — The tactical layer reconciles demand and supply

The tactical layer is usually a monthly S&OP cycle with five steps, run over a 3–18 month
rolling horizon. The five-step form is a convention rather than a law, but it is durable
because each step has a distinct owner and a distinct number.

**Step 4. Product and portfolio review.**
- Input: new product introduction (NPI) dates, end-of-life plans, cannibalisation assumptions.
- Owner and cadence: product management or marketing; monthly, first week.
- Output and artifact: a product family plan with launch and retirement dates, mapped to the
  planning buckets.
- Key branch: **does the forecast include the new product?** New products are absent from
  statistical history and present in commercial ambition, which is the gap that produces a
  launch stock-out.

**Step 5. Demand review.**
- Input: statistical forecast, sales pipeline, promotions, customer commitments, market
  intelligence.
- Owner and cadence: sales or demand planning with regional input; monthly, second week.
- Output and artifact: the consensus forecast, unconstrained by supply, with an override log by
  family and bucket.
- Key branch: **is the consensus forecast a forecast or a negotiation?** If the number is set by
  the most senior person in the room, it is a target wearing a forecast's clothes.

**Step 6. Supply review.**
- Input: consensus forecast, capacity, material availability, labour, supplier constraints.
- Owner and cadence: supply planning or operations; monthly, third week.
- Output and artifact: a feasible supply plan and a statement of the gap, with capacity load and
  material exceptions by site and bucket.
- Key branch: **is the plan feasible or merely optimistic?** A supply plan that ignores a known
  capacity constraint is a request.

**Step 7. Pre-S&OP and financial reconciliation.**
- Input: demand plan, supply plan, cost and revenue assumptions.
- Owner and cadence: planning director with finance; monthly, fourth week.
- Output and artifact: two or three options with financial consequences and a recommendation,
  one decision sheet per option.
- Key branch: **are the options genuinely different?** One plan plus two variants has not done
  the reconciliation.

**Step 8. Executive S&OP.**
- Input: the options and the recommendation.
- Owner and cadence: general manager or CSCO with functional leaders; monthly, last week.
- Output and artifact: the authorised plan of record, with a decision log naming decisions,
  owners, and dates.
- Key branch: **is the meeting deciding or reporting?** If the output is a slide deck rather
  than commitments, the process is reporting theatre.

### Phase C — The operational layer turns volume into sequenced work

**Step 9. Disaggregate the plan.**
- Input: the monthly plan of record.
- Owner and cadence: planning manager; weekly, with daily regeneration in MRP.
- Output and artifact: a weekly or daily master production schedule (MPS) and distribution
  requirements, expressed as scheduled receipts and planned orders.
- Key branch: **does disaggregation preserve the monthly total?** A schedule that sums to a
  different total has quietly created a second plan.

**Step 10. Explode through the bill of materials.**
- Input: MPS, inventory position, lead times, lot sizes, safety stock.
- Owner and cadence: MRP system, reviewed by material planners; daily or several times daily.
- Output and artifact: component planned orders and exception messages — an exception list, not
  a plan reading.
- Key branch: **how many exceptions are actionable?** If planners receive 800 messages a day and
  can act on 40, the system is generating noise.

**Step 11. Allocate constrained supply.**
- Input: available supply, order book, priority rules.
- Owner and cadence: supply manager under a published rule; weekly or daily as constraints bind.
- Output and artifact: an allocation by customer or order, with the rule cited.
- Key branch: **is the rule published before the shortage?** A rule invented during a shortage
  will be read as favouritism.

**Step 12. Sequence and release.**
- Input: allocated orders, changeover times, labour, material availability.
- Owner and cadence: production scheduler; daily, with intra-day adjustments by exception.
- Output and artifact: a finite-capacity schedule inside the frozen zone, with released work
  orders and a release list.
- Key branch: **is the schedule finite or infinite?** An infinite-capacity schedule produces a
  plan that cannot be built.

### Phase D — The execution layer acts on events

**Step 13. Detect the exception.**
- Input: event streams — late arrival, machine stop, quality hold, carrier delay, order change.
- Owner and cadence: control tower, dock, line, or service desk; real time.
- Output and artifact: an alert with severity and a proposed action, recorded as an event with a
  timestamp.
- Key branch: **is the alert actionable within its half-life?** An alert that arrives after the
  truck has left is a report, not a signal.

**Step 14. Decide and act.**
- Input: the alert, the authority matrix, the available options.
- Owner and cadence: the role named in the escalation matrix by severity; minutes to hours.
- Output and artifact: a committed action — reroute, expedite, substitute, defer, escalate —
  with a decision record naming time, owner, and rationale.
- Key branch: **who may act without asking?** An execution layer where every action needs
  approval has a decision latency equal to the approver's calendar.

**Step 15. Feed back.**
- Input: actual outcome versus planned outcome.
- Owner and cadence: planning and operations jointly; daily for exceptions, weekly for patterns,
  monthly for the S&OP input.
- Output and artifact: variance attribution and a corrective action, in a log that distinguishes
  cause from noise.
- Key branch: **is the feedback attributed to a mechanism or a person?** Attribution to a person
  ends learning; attribution to a mechanism changes the system.

### 5.1 The handoff contract

Between any two layers there is a handoff, and every handoff needs four things: a number, a
version, an owner, and a validity window. Without all four, the receiving layer will re-derive
the number, and the organisation will have two plans.

| Handoff | Number transferred | Validity window | Required statement |
|---|---|---|---|
| Strategy to tactics | Capacity by site, service promise | Fiscal year | "These are fixed; escalate if infeasible" |
| Tactics to operations | Monthly volume by family and site | Frozen period | "This is the plan of record as of this date" |
| Operations to execution | Sequenced work and allocations | Release fence | "This is released; change requires deviation" |
| Execution to operations | Actual completion, exceptions | Daily | "This is what happened, with timestamps" |
| Operations to tactics | Feasibility, capacity truth, backlog | Weekly | "This is what the plan can actually deliver" |
| Tactics to strategy | Trend, structural constraint, risk | Quarterly | "This assumption no longer holds" |

### 5.2 What happens when layers disagree

Four disagreements account for most of the observed dysfunction. Each has a resolution protocol,
and the protocol matters more than the numbers.

| Disagreement | Symptom | Resolution |
|---|---|---|
| Strategy assumes capacity operations cannot deliver | Network designed for 14,000 units a week; plant has never exceeded 12,600 | Validate capacity with operations before the business case closes |
| Tactics commits volume operations cannot sequence | Monthly total is feasible; product mix needs more changeovers than the month has hours | Rough-cut capacity check by family; show the mix constraint to executives |
| Operations schedules work execution cannot complete | Schedule assumes material that is in transit and late | Base ATP on confirmed supply; add a material-availability check |
| Execution improvises around the plan | Dock substitutes, carrier reroutes, line re-sequences | Exception log with reason codes; explicit bounded authority to deviate |

> **Note:** In each case the resolution is a process, not a model. Analytics can make the
> disagreement visible earlier and cheaper, but it cannot decide which layer wins. That is a
> decision right, and it belongs in the governance document.

### 5.3 Decision latency versus decision quality

Latency and quality are usually presented as a trade-off: wait longer, decide better. The
relationship has three regimes.

- **Information still arriving.** Delay improves quality, because a shipment event, a competitor
  price, or a retailer confirmation changes the expected value.
- **Information exhausted.** Delay no longer improves quality and starts to reduce it, because
  the value of acting decays: the truck leaves, the promotion starts, the customer buys elsewhere.
- **Information stale.** Delay actively harms quality, because the decision now describes a world
  that has moved.

The optimum is where the marginal quality gain from one more hour equals the marginal cost of
acting one hour later. For most operational decisions that point arrives quickly, because the
stake per decision is small and the decay is fast. For strategic decisions the stake is large and
the decay is slow, so more analysis is rational.

Two reallocation decisions make the asymmetry clear. Each has €22,000 of contribution at stake.
The first is made in four hours with 92% confidence; the second in 72 hours with 94% confidence.
Assume an incorrect allocation loses the full stake.

```text
value captured = confidence x stake
four-hour decision:  0.92 x 22,000 = EUR 20,240
72-hour decision:    0.94 x 22,000 = EUR 20,680
```

The slow decision captures €440 more, before the cost of delay. If the three-day delay forces 200
units to be expedited at €12 each, the delay costs €2,400 and the fast decision wins by roughly
€1,960. The general rule: **at the operational layer, latency usually dominates accuracy; at the
strategic layer, accuracy usually dominates latency.** An analytics team that optimises accuracy
everywhere is optimising the wrong variable for most of its users.

**Decision quality is not outcome quality.** Quality is the expected value of the decision given
the information available at the time; an outcome is one draw from the distribution. A company
that reviews decisions by outcome rewards luck and punishes well-reasoned decisions that went
wrong, which reliably makes future decisions worse.

The **last mile of analytics** runs from insight to outcome. Analysis produces an insight, a
person with authority converts it into a decision, an execution process converts the decision
into an action, and measurement converts the action into an outcome. Each handoff loses value in
a different way.

| Handoff | What is lost | Symptom |
|---|---|---|
| Insight to decision | Authority, timing, framing | Recommendation is correct and unowned |
| Decision to action | Capacity, material, sequence | Decision is made and not executed |
| Action to outcome | Measurement, attribution | Action happens and the effect is invisible |

Most analytics programmes invest in the first arrow and measure the last, and never instrument
the two handoffs in between. That is why a model can be excellent and the programme still fail.

### 5.4 Decision rights and the unowned decision

RAPID is a decision-role framework developed at Bain & Company: **Recommend, Agree, Perform,
Input, Decide**. The letters are roles, not stages, and the acronym is not a sequence.

- **Recommend** drives the process, gathers input, and produces the proposal. Ideally one person.
- **Agree** holds a veto, and only for a genuine constraint such as law, contract, or safety. A
  veto without an alternative is an escalation.
- **Perform** executes the decision and should also be consulted, so that feasibility surfaces
  before commitment.
- **Input** supplies facts and perspective without a vote.
- **Decide** is the single point of accountability that commits the organisation.

Two design rules matter more than the labels. First, there is exactly one D. A decision with two
deciders is a decision with none, because each can assume the other is acting. Second, A is used
sparingly; a process where four functions hold vetoes has no decider, only a negotiation.

RACI — Responsible, Accountable, Consulted, Informed — is a task framework, not a decision
framework. It answers "who does the work", not "who may commit". Using RACI for decisions
produces accountability without authority, which is the organisational form of an unowned
decision.

An unowned decision cannot be improved by analytics, because the binding constraint is authority,
not information. If the recommended action requires a role that does not exist, a more accurate
recommendation changes nothing. The diagnostic is simple: name the role that would sign, the
meeting where it would be signed, and the date. If any of the three is missing, fix the decision
before building the model.

## 6. Worked example

Meridian Home is a mid-sized European manufacturer of small kitchen appliances, with revenue of
approximately €480 million a year. This example follows one product family — countertop ovens —
through the April 2026 planning month. All figures are illustrative and internally consistent.
Currency is euro (€); units are finished units; the average selling price is €70.

### 6.1 The setting

The family is made at one plant in Poland on a line with a demonstrated capacity of 14,000 units
per week, equivalent to 56,000 units in a four-week production month. One controller is imported
from a supplier in Vietnam with a 42-day ocean lead time. The plant runs a monthly S&OP, a weekly
S&OE, and a daily MRP regeneration. The planning time fence for imported controllers is two
weeks; the frozen zone for the oven line is one week.

The commercial context: a German retail chain has been signed for a spring listing. The listing
commitment is 22,000 units across April and May, of which 4,000 units are promised in week 14.

### 6.2 The month, meeting by meeting

**Annual operating plan (set the previous October).** The AOP target for April is 61,000 units.
It is a target and a budget assumption. It was built before the German listing was won.

**February — the commercial commitment.** The commercial director signs the listing for 22,000
units, with 4,000 in week 14. Supply learns after signature. This is a commitment made at the
commercial layer without a supply feasibility check, and it is the root of the month's problem.

**3 March — demand review.** The statistical forecast for April is 57,400 units. The listing adds
2,550 units net of cannibalisation of the existing German business. The consensus forecast
becomes 59,950 units. The override is logged with a reason code and an owner.

**5 March — supply review.** The plant confirms 56,000 units of build. Opening inventory
available for April is 2,000 units. Available to sell is therefore 58,000 units against a
forecast of 59,950 — a gap of 1,950 units. The controller supply supports the 56,000-unit build
but constrains any overtime beyond 900 units.

**9 March — pre-S&OP.** Finance prices three options.

| Option | Action | Quantity covered | Cost | Service consequence |
|---|---|---:|---:|---|
| A | Overtime only | 900 | €1,890 | 1,050 units short |
| B | Overtime plus air freight | 1,950 | €17,115 | Full coverage |
| C | Defer and allocate | 0 | €50,900 | 1,950 units short, penalty |

The arithmetic behind the table:

```text
gap                     = 59,950 - 58,000 = 1,950 units
overtime premium        = 900 x EUR 2.10  = EUR 1,890
air freight premium     = 1,050 x EUR 14.50 = EUR 15,225
option B cost           = 1,890 + 15,225  = EUR 17,115

lost contribution       = 1,950 x EUR 22.00 = EUR 42,900
retail penalty          = EUR 8,000
option C cost           = 42,900 + 8,000   = EUR 50,900

net benefit of B over C = 50,900 - 17,115  = EUR 33,785
```

**11 March — executive S&OP.** The meeting approves Option B, provided the air freight is booked
by 13 March and commercial accepts a 50% cut in May promotional display volume to protect
controller supply. The decision is recorded with an owner (supply director), a date, and a
number. The plan of record becomes 59,950 units of demand and 57,950 of supply.

**17 March — S&OE.** At 08:00 on Tuesday the supplier alerts that the week-14 controller shipment
will arrive three days late. At risk: 2,400 units. The S&OE lead applies the published rule:
contracted retail first (1,600), e-commerce second (500), distributor deferred (300). The
decision is taken at 10:00 Wednesday, 26 hours after the alert.

**24 March — the fence change.** The S&OE review finds that three of the last five controller
disruptions were detected inside the two-week planning time fence, when no rescheduling was
possible. The supply director extends it to four weeks and adds a weekly supplier confirmation to
the S&OE agenda. This changes the decision stack, not the plan.

### 6.3 The month in numbers

| Layer | Number | Value | Owner | Cadence |
|---|---|---|---:|---|
| Target (AOP) | April volume target | 61,000 units | General manager | Annual |
| Forecast | Consensus demand, April | 59,950 units | Demand planning | Monthly |
| Plan | Feasible supply, April | 58,000 units before expedite | Supply planning | Monthly |
| Commitment | German retail listing, week 14 | 4,000 units | Commercial director | Contractual |
| Execution | Week-14 build at risk | 2,400 units | S&OE lead | Daily |
| Outcome | April sales | 59,100 units | — | Actual |

April sales of 59,100 units were 850 units below the consensus forecast, a forecast error of
1.4%. On-time-in-full (OTIF) for the family was 91.4%, computed as 84 late order lines out of
980, against a 95% target. The expedite cost of €17,115 was 0.41% of April family revenue of
€4,196,500. The promotion shift moved €120,000 of display spend from April to May.

The average numbers look tolerable. The concentration is the story: the 850-unit shortfall was
not spread evenly. It fell almost entirely on the distributor, whose 300 deferred units became
a 38% shortfall on its April order, while the contracted retailer received 99.2% of its
commitment. A service metric averaged across customers would have hidden the commercial damage;
a service metric by customer segment would have shown it immediately.

### 6.4 What changed as a result

Three things changed, and only one of them was a number.

1. **The plan of record became singular.** Before March, the commercial team planned from the
   listing commitment, the plant planned from the S&OP number, and finance planned from the AOP.
   After the executive meeting, the plan of record was published with a version number and a
   validity window. The three numbers still existed, but their relationship was now stated.
2. **The fence moved.** The controller planning time fence moved from two weeks to four. The
   consequence is a slightly less responsive plan and a materially more stable one. The trade-off
   was made explicitly, which is the point.
3. **The allocation rule acquired an owner.** The rule existed before, in a spreadsheet and in
   people's heads. After the S&OE decision it was published, owned by the supply director, and
   reviewed quarterly. A rule with an owner can be challenged and improved. A rule without one
   can only be resented.

### 6.5 The counterfactual

Suppose the company had not expedited. The 1,950-unit gap would have been allocated by the
default rule — first come, first served — and the retailer's week-14 commitment would have been
roughly half met. The contribution lost would have been €42,900, the penalty €8,000, and the
commercial damage to a new listing would have been unquantified but plausibly larger than both.
The expedite looks expensive per unit and cheap per relationship. This is the kind of judgement
that no model makes, and that a decision stack makes possible by putting the options in front of
the person with the authority to choose.

### 6.6 What an analyst should have asked

1. Was the 57,400-unit statistical forecast built before or after the listing was announced? If
   after, the 2,550-unit override double-counts the listing.
2. Is the 14,000-unit weekly capacity a demonstrated rate or an engineering rate? The difference
   is usually 5–12%, which is larger than the gap being managed.
3. Is the 91.4% OTIF measured against the customer's requested date or the company's confirmed
   date? The two definitions can differ by several points on the same data.

Each question is a stack question. None requires a new model.

### 6.7 What the 26-hour latency cost

Suppose 400 units of a similar model were available at the plant. Within six hours, 250 of the
300 deferred units could have been substituted. The arithmetic looks compelling:

```text
contribution preserved    = 250 x EUR 22 = EUR 5,500
SLA credit avoided        = EUR 2,000
benefit per event         = EUR 7,500
eligible events per year  = 9 x 0.60 = 5.4
annual benefit            = 5.4 x 7,500 = EUR 40,500
cost of on-call approver  = 0.15 FTE x EUR 80,000 = EUR 12,000
net benefit               = EUR 28,500
```

Read the assumption. The €5,500 is incremental only if the substitute units would not have sold
otherwise. If they would have sold anyway, the substitution moves contribution rather than
creating it, and the only incremental benefit is the €2,000 credit. Recompute:

```text
annual benefit = 5.4 x 2,000 = EUR 10,800
net benefit    = 10,800 - 12,000 = EUR -1,200
```

The investment is now marginally negative. The difference between €28,500 and −€1,200 is one
assumption about cannibalisation, and nobody in the meeting would have stated it. That is the
standard trap of latency business cases: they count the value of the action, not the value of
acting sooner. Before funding a faster decision, establish which part of the benefit is
incremental and which is transferred.

## 7. Code, query, or calculation

This section is deliberately arithmetic-first. The useful artifact for a decision stack is a
calendar and a decision log, not a software pipeline.

### 7.1 The decision-cadence calendar

Build this table for any supply chain you are analysing. It is usually the fastest way to find
out that two decisions with the same horizon are being made on different clocks.

| Decision | Layer | Cadence | Meeting | Owner | Number used |
|---|---|---|---|---|---|
| Open or close a DC | Strategic | Annual | Network review | CSCO | 5-year NPV |
| Sign an annual volume contract | Strategic | Annual | Commercial review | Commercial director | Committed volume |
| Set the monthly volume plan | Tactical | Monthly | Executive S&OP | General manager | Consensus forecast |
| Set promotional display volume | Tactical | Monthly | Pre-S&OP | Sales director | Lift estimate |
| Release the weekly schedule | Operational | Weekly | S&OE | Planning manager | Plan of record |
| Allocate constrained supply | Operational | Weekly | S&OE | Supply manager | Allocation rule |
| Reroute a delayed truck | Execution | Real time | — | Dispatch lead | Event + SLA |

### 7.2 One small query: decision latency

If the organisation keeps a decision log, latency is computable. The query below assumes columns
`detected_at`, `decided_at`, and `acted_at`, all timestamps, and a `layer` label.

```sql
select
    decision_id,
    layer,
    round(extract(epoch from decided_at - detected_at) / 3600.0, 1) as detect_to_decide_h,
    round(extract(epoch from acted_at - decided_at) / 3600.0, 1)   as decide_to_act_h,
    round(extract(epoch from acted_at - detected_at) / 3600.0, 1)  as end_to_end_h
from planning.decision_log
where detected_at >= date '2026-03-01'
order by end_to_end_h desc;
```

With the seven March decisions in the worked example plus four routine ones, the end-to-end
latencies in hours are 2, 4, 9, 18, 26, 31, 44, 58, 72, 96, and 120. The median is 31 hours; the
mean is 43.6 hours. The median is the number to report, because one 120-hour strategic decision
would otherwise dominate the mean and hide the operational pattern. Report the 90th percentile
separately, and always report the count — a median over seven decisions is a conversation
starter, not a benchmark.

> **Note:** Latency measured from `detected_at` inherits every weakness of the detection
> timestamp. If the alert was generated by a nightly batch, the "signal" is already a day old,
> and the latency metric flatters the organisation. Measure from the physical event where you
> can, and state which timestamp you used.

## 8. Data, standards, and regulatory context

Each layer consumes a different data model. Most integration failures are attempts to use one
layer's data at another layer's granularity.

### 8.1 Granularity by layer

| Layer | Time bucket | Entity granularity | Typical record | History needed | Latency tolerance |
|---|---|---|---|---|---|
| Strategic | Year or quarter | Product family, region, node | Network model, contract | 3–5 years | Weeks |
| Tactical | Month or week | Family or SKU family, site | S&OP grid, budget | 2–3 years | Days |
| Operational | Week or day | Stock-keeping unit (SKU) at location | MRP planned order, allocation | 13–52 weeks | Hours |
| Execution | Minute or hour | Order, load, operation, serial | Event log, work order | 4–13 weeks | Seconds to minutes |

The common error is strategic data used operationally. A product-family forecast cannot size a
safety stock for a single SKU; it can only set a family target. The reverse error also occurs:
an item-level forecast aggregated to a family is usually more accurate than a family forecast,
because aggregation averages errors, but the family number will not tell you which item to build.

### 8.2 The manufacturing-level standard

ANSI/ISA-95, adopted internationally as IEC 62264, defines a hierarchy of enterprise and control
systems that maps closely onto the lower half of the decision stack.

| ISA-95 level | Function | Typical time frame | Stack layer |
|---|---|---|---|
| Level 4 | Business planning and logistics | Months to weeks | Tactical and operational |
| Level 3 | Manufacturing operations management | Days, shifts, hours | Operational |
| Level 2 | Supervisory control | Minutes, seconds | Execution |
| Level 1 | Sensing and manipulation | Sub-second | Execution |
| Level 0 | The physical process | Continuous | Execution |

The useful consequence for analytics: **the level at which data is created constrains the
latency you can achieve.** A work-order confirmation at Level 3 cannot produce a Level 2
signal. If the business requirement is a fifteen-minute response, the data model has to be
designed at the level that produces fifteen-minute events.

### 8.3 Planning and calendar standards

| Standard or convention | Governs | Why it matters to the stack | Grade |
|---|---|---|---|
| APICS Dictionary (ASCM) | Definitions of time fence, MPS, MRP, ATP | The reference vocabulary for layer mechanics | C |
| ANSI/ISA-95 / IEC 62264 | Enterprise-control integration levels | Ties data latency to system level | B |
| ISO 8601-1:2019 | Date and time representation, durations | Makes cross-system timestamps comparable | B |
| ISO 80000 | Quantities and units | Prevents unit errors across planning systems | B |
| NRF 4-4-5 retail calendar | Retail fiscal periods | Explains why retail and manufacturing months differ | C |
| SCOR Digital Standard (ASCM) | Process reference model | Common language for plan, source, make, deliver, return | C |

The 4-4-5 calendar deserves a note. Many retailers close periods on a 4-4-5 week pattern so that
comparisons align to day-of-week. A manufacturer planning on calendar months will therefore have
period boundaries that never quite match its largest customers'. Monthly S&OP numbers reconciled
to a 4-4-5 retail calendar can differ by several days of demand, which is enough to explain a
few points of apparent forecast error.

### 8.4 Financial and control requirements

| Requirement | Where | Effect on the decision stack | Grade |
|---|---|---|---|
| IFRS 15 / ASC 606 | Global accounting | Revenue recognition depends on control transfer, which depends on execution events | B |
| Sarbanes-Oxley Act 2002, §404 | US-listed companies | Internal control over financial reporting extends to forecast and planning inputs | B |
| Local statutory audit requirements | Most jurisdictions | Plan-to-actual reconciliations become audit evidence | B |
| EU Corporate Sustainability Reporting Directive (CSRD), Directive (EU) 2022/2464 | European Union | Strategic footprint decisions acquire a mandatory reporting dimension | B |
| Transfer pricing rules | Multinational | Constrains how cost and margin are attributed across nodes | B |

In a listed company the number entering the financial forecast is subject to internal control: it
has an owner, a review, and a retention requirement. A model that produces a better number but
cannot be reconciled, versioned, and audited will not be adopted, however accurate it is.

### 8.5 What is convention rather than requirement

- The five-step S&OP cycle is a convention popularised in the 1980s, not a mandate; many
  companies run four steps or seven.
- Monthly S&OP cadence is a habit inherited from the monthly financial close. A business with a
  two-week demand half-life may need a fortnightly cycle.
- A two-week planning time fence and a 95% service target are policy choices, not constants. The
  fence should be derived from lead-time variability and change cost.

## 9. Industry practice and case evidence

### 9.1 How integrated planning actually is

The gap between the textbook stack and the operating stack is large, and it is measurable.

| Source | Date | Sample | Finding | Grade |
|---|---|---|---|---|
| BearingPoint, integrated planning study | 2026 | 193 executives, 7 European countries, 8 industries | 8% report fully integrated end-to-end planning; 14% have a mostly or fully integrated tool landscape; 95% call planning critical; 8% are fully satisfied | D |
| The Hackett Group, reported in Procurement Magazine | June 2025 | Not stated in the secondary report | 25% have implemented a full and consistent S&OP approach across the organisation | D |
| AbcSupplyChain S&OP/IBP survey | November 2025 | 164 practitioners, 54 countries | 81.1% still run S&OP in spreadsheets; 5.1% use dedicated integrated planning tools | D |
| APQC benchmarking, measure 107397 | Accessed 2026 | 1,068 organisations | Median average monthly demand forecast accuracy 85.0% | C |
| BCG, advanced planning survey | February 2026 | 180+ planning leaders | Over 70% have invested in advanced planning; about 1 in 5 report meaningful value; 7% report value from agentic or generative AI | D |

Read these together and one conclusion is difficult to avoid: the integrated, single-plan stack
that the literature describes as normal is, in survey after survey, a minority capability. An
analytics design that assumes a monthly consensus forecast, a single plan of record, and a
functioning S&OE will under-deliver in most organisations. The design should instead ask which
of those three exists, and build for the one that does.

**Note on the APQC figure.** A median of 85% monthly forecast accuracy is an aggregate across
industries and hierarchies, and it is self-reported. It is useful as a calibration point — a
model at 85% at the wrong granularity is not obviously better than the median — but it is not a
target. Accuracy is only meaningful relative to the decision it informs.

### 9.2 Cadence as a design choice: Zara

The most cited case of cadence as design is Zara, the Spanish apparel retailer. Ghemawat and
Nueno (2003) document store managers ordering twice weekly, short production cycles, and
design-to-shelf time measured in weeks. The mechanism is not a better forecast; it is a shorter
decision cycle plus a deliberately limited initial commitment, so the second decision — replenish
what sold — is made with real information.

The lesson: **shortening the cadence substitutes for forecast accuracy and is usually cheaper.**
A company that can reorder weekly needs a less accurate seasonal forecast than one that commits
once. Grade A for the case; C for performance claims, which are company-reported.

### 9.3 Execution authority: the andon precedent

Toyota's andon cord is the canonical case of pushing a decision right to the execution layer: any
line worker may stop the line on a defect. It is not merely a permission; it is an escalation
protocol with a response obligation, so stopping the line triggers help rather than blame. The
supply chain parallel is the dock lead who holds a shipment on a quality signal.

The lesson is decision latency: a defect detected at 10:00 and escalated at 14:00 has a four-hour
latency no dashboard fixes. Grade C for the practice, E for performance claims.

### 9.4 When the planning system outruns the process: Nike 2001

Nike's 2001 deployment of advanced planning software is a frequently cited case of a system
installed faster than the process governing it. Reporting at the time described a demand-planning
error producing excess inventory in some styles and shortages in others, with a reported cost of
roughly $100 million in lost sales. The details are disputed; treat the figure as an
order-of-magnitude claim, not an audited number.

The stack reading matters more. The system improved tactical computation while the demand review,
the override process, and the financial link were not ready. A better engine in an unbuilt stack
produces faster wrong numbers. Grade C for the outcome, E for causal attribution.

### 9.5 The 2026 agentic planning context

The 2026 material is consistent on one point and inconsistent on another. It is consistent that
decision governance is the binding constraint. Gartner's 2026 work argues that deploying agents
requires decision maps, guardrails including decision rights, knowledge assets, and decision-grade
data, not a better model alone; its planning guidance recommends staged adoption and warns
against "agent washing". Grade D.

It is inconsistent on adoption. Blue Yonder's Compass 2026 reports 8% using agentic AI and 46%
with no plans; an ORTEC survey published in January 2026 reports 42% not exploring it and 23%
planning a pilot within twelve months; BCG's February 2026 survey reports 7% deriving value from
agentic or generative AI in planning. Different questions, different samples. The spread is the
finding: an experimental fringe and a large non-adopting core. Grade D for all three.

Two implications follow. First, autonomy is arriving at the execution and operational layers
first, where reversibility is high and guardrails are easy to state; it arrives last at the
tactical and strategic layers, where the commitment is financial. Second, the human-in-the-loop
pattern is becoming a design object: approve every action, approve exceptions, sample, or be on
call. Each has a different latency and error profile, and none is universally correct.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reporting theatre | Executive S&OP ends without decisions, owners, or dates | Meeting reviews slides rather than committing resources | Plan never authorised; functions follow their own | Require a decision log as the meeting output |
| Shadow planning | Planners keep a personal spreadsheet that differs from the system | System plan is not trusted or not usable | Two plans; reconciliation cost; silent hedging | Publish the plan of record; fix the system gap |
| Fence erosion | Release dates change daily without a deviation record | Change is easier to make informally than formally | Supplier distrust; schedule nervousness | Exception route with a named approver and a log |
| Over-replanning | MRP regenerated several times a day; exception volume exceeds capacity | Compute is cheap, so change is assumed free | Planner burnout; noise treated as signal | Bucket the plan; freeze inside cumulative lead time |
| Under-replanning | Plan unchanged through a structural demand shift | Cadence set by tradition, not half-life | Service failure; obsolete inventory | Review cadence against signal half-life |
| Unowned decision | "Someone should decide" appears in meeting notes repeatedly | No decision right assigned | Latency equals the escalation chain; nobody acts | RAPID map; name the D |
| Layer conflation | Strategic questions debated in the weekly ops meeting | No forum for the strategic question | Both decisions made badly | Separate agendas; escalate structurally |
| Forecast used as commitment | Sales is measured on forecast accuracy and on revenue | Two objectives on one number | Sandbagging or over-commitment | Separate forecast, plan, target, commitment |
| Autonomy without guardrails | Agent can act outside the tested envelope | Approval removed before constraints defined | Unbounded error, reputational damage | Bounded envelope, hard limits, audit trail |
| Alert volume above decision capacity | Planners ignore most alerts | Detection scaled without triage design | Real exceptions missed in noise | Design exception thresholds against capacity |
| Metric at the wrong layer | Weekly dashboard drives a five-year network choice | Granularity mismatch | Optimising the local, damaging the structural | Match metric horizon to decision horizon |
| No decision log | Nobody can reconstruct why a plan changed | Decisions recorded in chat and memory | No learning; repeated mistakes | Timestamped decision record with owner and rationale |

**Reporting theatre** is the most common and the hardest to see from inside: the meeting is well
attended, the deck is polished, the discussion is substantive, and nothing is committed. Count,
over three cycles, the decisions with a named owner and a date. If the count is zero, the process
is a review.

**Unowned decision** is the failure analytics cannot fix. If a recommendation requires an action
that no role is authorised to take, improving its accuracy changes nothing. Assign the right,
define the trigger, and publish the rule.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Each decision has an identifiable owner | The stack is a map of authority | Matrix organisations, shared services, joint ventures | Ask who can commit without escalation; observe who actually does |
| Cadence is stable enough to plan around | Calendar design assumes recurrence | Crisis mode, continuous replanning | Count meetings held versus scheduled over a quarter |
| The plan of record is singular and published | Reconciliation requires one reference | Multiple systems, multiple regions | Ask three functions for "the plan" and compare versions |
| Horizons are nested | Each layer must fit inside the one above | A tactical horizon longer than the strategic one | Compare horizon lengths across layers |
| Lead times are known with a distribution | Fence sizing depends on it | New lanes, congestion, supplier change | Plot lead time by lane over 12 months |
| Capacity is a demonstrated rate | Feasibility depends on it | Engineering rates, unvalidated nameplate | Compare planned and actual output by week |
| Demand signal is observable in time | Detection latency depends on it | Batch POS feeds, distributor-only visibility | Measure event-to-record latency |
| Decision latency can be measured | Improvement requires a baseline | No decision log, no timestamps | Start a log; even a manual one |
| Autonomy envelope is testable | Guardrails must be verifiable | Novel conditions outside the training distribution | Backtest the envelope; run shadow mode |
| Financial and operational calendars can be reconciled | Plan-to-budget linkage | 4-4-5 retail calendar, fiscal year mismatch | Map period boundaries explicitly |

Three limits of the stack model are worth stating plainly.

**The model is descriptive, not prescriptive.** It tells you where a decision sits and what it
must reconcile with. It does not tell you the right cadence for a specific business. That
requires the half-life calculation and the change-cost estimate, both of which are empirical.

**The model assumes decisions are made in meetings.** A growing share of execution decisions are
made by software, and some operational decisions are made by planning engines with human
approval. The stack still applies, but the "meeting" becomes an approval queue, and the
decision-rights question becomes an autonomy-level question.

**The model assumes the firm controls its own stack.** In a marketplace, a 4PL, or a
contract-manufacturing relationship, several layers belong to different legal entities with
different cadences. The stack then becomes a set of inter-organisational contracts, and the
reconciliation problem is a negotiation rather than a meeting. Module 21 treats these structures.

## 12. Exercises

### Exercise 1 — Map your own stack (design, 45 minutes)

Choose an organisation you know, or a publicly described one. Identify six decisions across at
least three layers.

**Deliverable:** a table of decision, layer, horizon, cadence, owner role, number used, and
reversibility.
**Acceptance criteria:** six decisions; three layers; owners as roles; one assignment justified
by reversibility against its job title.

### Exercise 2 — Diagnose a layer disagreement (critique, 40 minutes)

A company reports a board target of 61,000 units, an S&OP plan of 58,000, and a customer
commitment implying 59,950. Write a one-page diagnosis.

**Deliverable:** the layer of each number, the missing reconciliation, its owner, and your first
question.
**Acceptance criteria:** each number labelled forecast, plan, target, or commitment; the
reconciliation is a named process; the first question is answerable from data.

### Exercise 3 — Design a time fence (design, 45 minutes)

A component has a 42-day ocean lead time, a 6-day standard deviation in arrival, and a change
cost of €1,200 per reschedule. Propose a planning time fence and a frozen zone.

**Deliverable:** fence lengths in weeks, the reasoning, and the exception process.
**Acceptance criteria:** justified against lead-time variability, not convention; the change cost
is used; the exception process names an approver and a trigger.

### Exercise 4 — Build a RAPID map (design, 40 minutes)

For "allocate constrained supply between a contracted retailer and a distributor", assign RAPID
roles.

**Deliverable:** a one-page role map plus the escalation path if the decider is unavailable.
**Acceptance criteria:** exactly one D; A only for a genuine veto; P roles are the executers; a
named alternate.

### Exercise 5 — Compute decision latency (calculation, 50 minutes)

A decision log contains these end-to-end latencies in hours: 3, 5, 8, 12, 19, 24, 30, 36, 48,
60, 84, 132.

**Deliverable:** the median, mean, 75th percentile, and a one-paragraph interpretation.
**Acceptance criteria:** correct arithmetic; percentile method stated; interpretation separates
central tendency from tail behaviour.

### Exercise 6 — Redesign a cadence (design, 60 minutes)

A business runs monthly S&OP, weekly S&OE, and daily MRP. Demand half-life is four days because
a retailer decides promotions weekly. Propose a cadence redesign.

**Deliverable:** a revised calendar, the cost of each change, and the decision that moves.
**Acceptance criteria:** tied to the half-life, not fashion; at least one decision changes layer;
cost includes meeting time and system change.

### Exercise 7 — The cost-of-error ladder (calculation, 40 minutes)

Estimate the annual cost of a 5% error at each layer: strategic network decision, €40m NPV, one
decision per five years; tactical plan, €4.2m monthly revenue, 30% contribution margin, twelve
cycles; operational schedule, 14,000 units a week, €22 contribution, 52 weeks; execution
dispatch, 2,000 units a day, €22 contribution, 250 days.

**Deliverable:** cost per decision and annual aggregate cost by layer.
**Acceptance criteria:** correct arithmetic; both figures shown; a conclusion about where effort
should concentrate, with a caveat.

### Exercise 8 — Autonomy threshold policy (design, 50 minutes)

Design an autonomy policy for an agent proposing safety-stock changes at SKU level.

**Deliverable:** three autonomy levels, confidence thresholds, hard guardrails, the review
pattern, and the audit record.
**Acceptance criteria:** testable triggers; numeric guardrails; the review pattern named; an
audit record sufficient to reconstruct a decision.

### Exercise 9 — Decision memo (design, 45 minutes)

Write a one-page memo proposing one analytics intervention as a change to a specific decision.

**Deliverable:** the decision, layer, owner, cadence, changed number, baseline, and failure mode.
**Acceptance criteria:** owner is a role; cadence is a real meeting; the number is defined; the
failure mode is a stack failure.

## 13. Self-check answers

**Exercise 5 arithmetic.**

```text
sorted values: 3, 5, 8, 12, 19, 24, 30, 36, 48, 60, 84, 132   (n = 12)

median = (24 + 30) / 2 = 27.0 hours
mean   = 461 / 12      = 38.4 hours
P75: position = 0.75 x (12 - 1) = 8.25, interpolating between the 9th and 10th values
      = 48 + 0.25 x (60 - 48) = 51.0 hours
```

Interpretation: the typical decision takes about a day and a quarter, but the tail is long — the
slowest decision took five and a half days, and the top quarter of decisions take more than two
days. A programme that reduces the median from 27 to 20 hours improves the routine; a programme
that reduces the tail requires a different intervention, usually an escalation rule or a
pre-authorised action. The mean is 11.4 hours above the median, which is the signature of a
right-skewed distribution; report the median and the 75th or 90th percentile, not the mean alone.

A common wrong answer computes the median as the 6th value (24) or the 7th (30) without
averaging for an even count. State the method, because different percentile conventions give
slightly different answers and the convention matters more than the decimal.

**Exercise 7 arithmetic.**

```text
strategic: 5% of EUR 40m = EUR 2.0m per decision
           annual = 2.0m / 5 years = EUR 400,000 per year

tactical:  5% of EUR 4.2m = EUR 210,000 revenue at risk
           contribution at 30% = EUR 63,000 per cycle
           annual = 63,000 x 12 = EUR 756,000

operational: 5% of 14,000 = 700 units per week
             contribution = 700 x 22 = EUR 15,400 per week
             annual = 15,400 x 52 = EUR 800,800

execution: 5% of 2,000 = 100 units per day
           contribution = 100 x 22 = EUR 2,200 per day
           annual = 2,200 x 250 = EUR 550,000
```

The per-decision cost falls by three orders of magnitude from strategic to execution, but the
annual aggregate is remarkably flat: roughly €0.4m, €0.76m, €0.80m, and €0.55m. The conclusion
is not "focus on operations". It is that a 5% error at every layer costs about the same per year,
so improvement effort should be allocated by the cost of reducing the error, not by the prestige
of the layer. The caveat is that the layers are not independent: an operational error is often a
symptom of a tactical error, and fixing the symptom without the cause transfers cost rather than
removing it.

**Exercise 1 rubric.** A strong answer gives decisions that differ in horizon, not six variants of
the weekly meeting, and assigns at least one by reversibility rather than title. A weak answer
lists departments and calls them layers.

**Exercise 2 rubric.** A strong answer labels 61,000 a target, 58,000 a constrained plan, and
59,950 a demand forecast plus a commitment, and names the pre-S&OP financial bridge as the missing
reconciliation. A weak answer declares one number correct.

**Exercise 3 rubric.** A strong answer sets the fence where lead-time variability makes
rescheduling futile — commonly two standard deviations plus the minimum run length — and
justifies the frozen zone from change cost. A weak answer copies a two-week fence.

**Exercise 4 rubric.** A strong answer gives the D to the supply manager or planning director,
gives A only to a genuine contractual or regulatory veto, and includes commercial as I. A weak
answer assigns multiple Ds.

**Exercise 6 rubric.** A strong answer moves promotion response into a weekly S&OE cycle, keeps
monthly S&OP for volume and finance, and states the cost in planning hours, system change, and a
new decision right. A weak answer says "meet more often".

**Exercise 8 rubric.** A strong answer defines recommend-only, auto-apply with exception review,
and auto-apply with sampling and a hard cap; guardrails are numeric and the audit record
reconstructs inputs, rule version, confidence, and approver. A weak answer says "human in the
loop" without saying which loop.

**Exercise 9 rubric.** A strong answer names a real meeting and a real number and states what
would change in the plan of record. A weak answer describes a dashboard.

## 14. Interview questions

**1. "What is the difference between S&OP and S&OE?"**

A strong answer gives horizon (months versus weeks), cadence (monthly versus weekly or daily),
granularity (family versus item), and purpose (reconcile demand, supply, and finance versus
manage the near-term plan and exceptions). It notes that S&OE's horizon is set by the response
lead time. A weak answer says S&OP is strategic and S&OE is operational, which blurs two
different distinctions.

**2. "Why do companies freeze part of the schedule?"**

A strong answer explains nervousness, the cost of change downstream, and the difference between
stability and rigidity. It mentions that a fence with an exception process is stable while a
fence without one is bypassed. A weak answer says "to stop planners changing things", which
describes the symptom.

**3. "A model recommends reallocating stock between two regions and nobody acts. What do you
ask?"**

A strong answer asks who owns the decision, which meeting it belongs to, what number the
decision would change, and whether the recommendation arrives before the commitment is made. It
treats this as a decision-rights and latency problem before a model problem. A weak answer
proposes a better model or a dashboard.

**4. "How would you measure decision latency, and why does it matter?"**

A strong answer decomposes it into detection, decision, and execution, names the timestamps, and
notes that measuring from a batch-generated alert flatters the result. It connects latency to
service and to the half-life of the decision. A weak answer equates it with system response time.

**5. "What is a planning time fence, and how would you set one?"**

A strong answer defines the PTF and DTF, distinguishes the frozen, slushy, and liquid zones, and
sizes the fence from lead-time variability and change cost. A weak answer gives a conventional
number with no derivation.

**6. "What does 'the plan of record' mean and why does it matter?"**

A strong answer defines it as the single authorised plan with a version and validity window, and
explains that without it functions re-derive their own numbers. A weak answer calls it "the
latest forecast".

**7. "Where would you allow an agent to act without approval?"**

A strong answer names execution and operational decisions with high reversibility and clear
guardrails, gives an autonomy ladder, and specifies thresholds, sampling, and audit. It notes
that tactical and strategic commitments stay human. A weak answer says "low-risk decisions"
without defining risk or the envelope.

**8. "How would you tell whether an AI planning tool is genuinely agentic?"**

A strong answer applies a behavioural test: can it adapt objectives, negotiate trade-offs, and
change execution logic, or does it merely automate a fixed rule? It mentions agent washing and
asks for the decision right and the audit trail. A weak answer relies on the vendor's label.

**9. "Two functions report different service levels for the same month. What is your
diagnosis?"**

A strong answer checks definition (requested versus confirmed date), completeness (line versus
order), exclusions, and time window, and notes that both numbers can be correct. A weak answer
assumes one function is wrong.

**10. "What would you do first if a company asked you to 'improve planning with AI'?"**

A strong answer maps the decision stack before selecting a model: which decisions exist, who
owns them, what cadence, what data granularity, and where the latency is. It then picks a
decision with an owner and a measurable number. A weak answer starts with a forecasting model.

## 15. Key terms

- **Decision stack** — the layered arrangement of supply chain decisions by horizon, cadence, and authority.
- **Strategic layer** — network, capacity, and service-promise decisions over one to five years.
- **Tactical layer** — monthly volume and financial reconciliation inside the existing network.
- **Operational layer** — weekly scheduling, MRP, and allocation inside the monthly plan.
- **Execution layer** — real-time response to events, measured in minutes to days.
- **Horizon** — how far forward a decision looks; **cadence** — how often it is revisited.
- **Time fence** — a date inside which changes follow special rules.
- **Demand time fence (DTF)** — inside it, only actual orders count as demand.
- **Planning time fence (PTF)** — inside it, the system will not auto-reschedule planned orders.
- **Frozen, slushy, liquid zones** — committed, negotiated, and freely replanned segments.
- **Nervousness** — plan instability from small input changes producing large schedule changes.
- **Plan of record** — the single authorised plan, with a version and a validity window.
- **S&OP** — sales and operations planning, the monthly tactical reconciliation.
- **S&OE** — sales and operations execution, the short-cycle near-term process.
- **IBP** — integrated business planning, S&OP extended to the full financial plan.
- **AOP** — annual operating plan, a target and budget rather than a forecast.
- **MRP** — material requirements planning, exploding demand through the bill of materials.
- **MPS** — master production schedule, what will be built and when.
- **ATP** — available to promise, uncommitted supply by date.
- **Decision right** — the authority to commit the organisation to an action.
- **RAPID** — Recommend, Agree, Perform, Input, Decide, a decision-role framework.
- **Decision latency** — elapsed time from signal to committed action.
- **Guardrail** — a hard constraint bounding an automated action.
- **Bounded autonomy** — automation permitted to act inside a defined, audited envelope.
- **Agent washing** — labelling ordinary automation as agentic AI.
- **Decision governance** — the rules for who may act, on what evidence, with what audit.

## 16. Sources

**Peer-reviewed and academic**

- Blackburn, J. D., Kropp, D. H., and Millen, R. A. (1986). "A Comparison of Strategies to
  Dampen Nervousness in MRP Systems." *Management Science*, 32(4), 413–429.
  DOI: 10.1287/mnsc.32.4.413. Grade A.
- Rogers, P., and Blenko, M. (2006). "Who Has the D? How Clear Decision Roles Enhance
  Organizational Performance." *Harvard Business Review*, January 2006. Grade A for the
  framework; the associated adoption statistic is Grade D.
- Ghemawat, P., and Nueno, J. L. (2003). "Zara: Fast Fashion." Harvard Business School case.
  Grade A for case documentation; performance claims Grade C.
- Wallace, T. F., and Stahl, R. A. *Sales and Operations Planning: The How-To Handbook*.
  T. F. Wallace & Company; consult the current edition for the five-step cycle. Grade C.
- Hopp, W. J., and Spearman, M. L. *Factory Physics*. Waveland Press; current edition. Grade A
  for the variability-buffering treatment that underlies fence sizing.

**Standards and official**

- ANSI/ISA-95, adopted as IEC 62264, *Enterprise-Control System Integration*. Grade B.
- APICS Dictionary, ASCM, current edition. Definitions of demand time fence, planning time
  fence, MPS, MRP, and ATP. Grade C (industry reference, not regulation).
- ISO 8601-1:2019, *Date and time — Representations for information interchange*. Grade B.
- ISO 80000, *Quantities and units*. Grade B.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (FASB). Grade B.
- Sarbanes-Oxley Act of 2002, §404, internal control over financial reporting. Grade B.
- Directive (EU) 2022/2464, Corporate Sustainability Reporting Directive. Grade B.
- ASCM SCOR Digital Standard. Grade C.
- National Retail Federation 4-4-5 retail calendar. Grade C.

**Industry, analyst, and practitioner**

- BearingPoint, *From Silos to Synergy: Next Gen Integrated Planning Excellence in a Volatile
  World*, 2026. 193 executives, 7 European countries, 8 industries. Grade D.
- The Hackett Group S&OP maturity finding, as reported in *Procurement Magazine*, June 2025,
  p. 17. Grade D (secondary reporting of analyst research).
- AbcSupplyChain, S&OP/IBP practitioner survey, November 2025, 164 respondents, 54 countries.
  Grade D.
- APQC, Open Standards Benchmarking, measure 107397, "Average monthly demand forecast
  accuracy," median 85.0%, n = 1,068. Accessed 2026. Grade C (self-reported benchmark data).
- BCG, *Getting the Most from Advanced Supply Chain Planning*, February 2026, n = 180+ planning
  leaders. Grade D.
- Gartner, "Build Your Supply Chain Decision Stack to Enable Agentic AI," 1 May 2026; "Prepare
  for Agentic AI in Supply Chain Planning Without Betting on Full Autonomous," 6 February 2026;
  "Most Supply Chains Are Optimizing for the Wrong Future," 26 August 2026. Grade D
  (vendor-analyst framing; abstracts only, not independently verified).
- Blue Yonder, *Supply Chain Compass 2026*, B2B International, autumn 2025, n = 678. Grade D.
- ORTEC, agentic AI logistics survey, published January 2026, 2,000+ executives, North America.
  Grade D.
- Involvation, "Manage the Short Term with S&OE." Practitioner article; source of the
  lead-time-defined S&OE horizon. Grade E.
- GAINS, "How to Eliminate Decision Latency in Supply Chain Management," December 2025. Grade D.
- Haptiq, "Decision Latency: How to Measure the Time Between Signal and Action," October 2025.
  Grade D.
- *Logistics Viewpoints*, "The Decision Bottleneck Is Becoming a Logistics Capacity Constraint,"
  20 October 2026. Grade E.

**Note on citations.** Survey figures are reported as their publishers state them and have not
been independently verified. Where a figure is disputed in the public record — the Nike 2001
cost, for example — the text says so rather than quoting precision it cannot support. No
benchmark for median decision latency is offered, because no credible cross-industry benchmark
was found; treat latency targets as organisation-specific until you have your own baseline.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_three-flows-one-network.md`](01_three-flows-one-network.md)
- Next topic: [`03_network-entities.md`](03_network-entities.md)
- Related: [`04_planning-processes.md`](04_planning-processes.md)
- Related: [`05_inventory-and-service.md`](05_inventory-and-service.md)
- Related: [`10_archetypes-and-metric-conflicts.md`](10_archetypes-and-metric-conflicts.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Return to the part map: [`../_index.md`](../_index.md)
