# Greenfield and Brownfield Design — building from scratch versus adapting an existing footprint

> Module 11 — Network Design and Optimization
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 topics 04 and 05
> Version 2026.5

---

## 1. Why this topic exists

Every network study begins with a question that sounds innocent and is not: **are we choosing
sites, or are we choosing what to do with the sites we already have?**

Those are different problems. In the first, which this course calls **greenfield**, every
candidate is new, nothing is inherited, and the design is unconstrained by history. In the
second, which it calls **brownfield**, most of the network already exists, most of the cost is
sunk, and the decisions are not open-or-not-open but keep, expand, reconfigure, downsize, or
close. The formulation in topic 04 can express both. The interpretation, the constraint set,
the cost basis, and the output are different enough that treating them as one problem is the
most reliable way to produce a study that is arithmetically correct and commercially useless.

Three failures follow from getting the class wrong, and each is common enough to have a name in
practice.

**The site that cannot be closed because it was recently built.** A company opens a distribution
centre (DC) in 2022, demand shifts, and a 2026 study recommends consolidating elsewhere. The
recommendation is correct on the numbers and is rejected in the first meeting, because the capital
spend is treated as a reason to keep using the site. The spend is gone; what remains is the
question of which future costs are lower. The organisation has confused the accounting record with
the decision.

**The site that is assumed to be free.** The opposite error is subtler and more common in models
than in meetings. The analyst sets the fixed cost of existing sites to zero, on the reasoning that
the sites are already paid for. The model then never closes anything, because nothing is cheaper
than free, and the study reports that the current network is optimal. That is not a finding. It is
a modelling artefact produced by deleting the only coefficient that could have challenged the
status quo.

**The plan that cannot be executed between two dates.** A target network is chosen, the transition
is costed as a single line labelled "one-off costs", and the programme fails eighteen months later
because the receiving site could not absorb the volume until its racking was installed, the lease
on the closing site ran for another three years, and the consultation period pushed the first
saving into the second financial year. The destination was right. The path was never modelled.

The three failures share a structure. Each is a disagreement about **what the counterfactual
is**. Greenfield has an easy counterfactual: no network, build one. Brownfield's counterfactual
is the network you already have, running forward, with its leases, its wage inflation, its
known capacity shortfalls and its renewal dates. Pricing that counterfactual is the work, and it
is work that greenfield studies never have to do.

There is a reason the literature and the textbooks are dominated by the greenfield case, and it is
worth stating plainly because it explains what you have and have not been taught. The canonical
facility-location formulations carry no state: a binary variable `x_j` costs the same to switch on
whether or not a building was already standing there. Adding legacy requires either a second binary
for closure, or a differentiated fixed cost, or a time index — and the first published treatments of
closing and relocation appear in 1975 and 1982, more than sixty years after Weber's continuous
location problem. The greenfield case is not the important case; it is the case the mathematics was
built for, and therefore the case that teaching examples demonstrate.

**What this topic is not.** Topic 04 owns the facility-location formulation: fixed charge, capacity,
linking constraint, relaxation, gap, and this topic does not re-derive any of it. Topic 05 adds
commodities and periods, and a migration path is a multi-period object, so topic 05 is the machinery
you will reach for. Topic 07 owns sensitivity, topic 08 owns uncertainty, topic 13 owns what happens
inside a reconfigured building, and Module 14A owns disruption and resilience. This topic owns the
distinction, the cost of change, and the transition. It matters commercially because a greenfield
study answers a question a company asks once a decade, usually when it has no network or is entering
a new country, while a brownfield study answers a question it asks every year — and the second is
where the analytical hours are spent and where the credibility of the function is made.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a stated network problem as greenfield, brownfield, or hybrid, and **justify**
   the classification from the decision verbs rather than from the industry.
2. **Explain** why the facility-location literature and its teaching examples are
   greenfield-dominated, and **state** what that bias does to a practitioner's intuitions.
3. **Distinguish** the sunk-cost trap from the sunk-capacity trap, and **identify** both in a
   written business case within one reading.
4. **Represent** an existing facility in a model: the avoidable annual cost, the step function
   for partial retention, the closure cost as an explicit one-off term, and the treatment of
   write-down and disposal value.
5. **Build** a cost-of-change account with named line items, **separate** the items that belong
   in the annual fixed cost from those that belong in the transition, and **state** what is
   excluded.
6. **Encode** physical and legal reconfigurability as capacity ceilings and arc removals, and
   **distinguish** a constraint from a cost.
7. **Construct** a migration path with a critical path, a capacity check at every date, and a
   stated earliest-effective date per closure, and **explain** why a destination and a path are
   optimised under two different objectives.
8. **Report** a brownfield study to the standard in section 5: baseline priced, option set
   enumerated with the do-nothing case, cost-of-change items listed as included and excluded,
   and the migration path with its critical assumptions.

Each outcome maps to an exercise in section 12 and to a self-check answer in section 13.

## 3. The core mental model

Hold two pictures of the same demand, drawn side by side, and notice that the second has
something the first does not: a status quo.

```text
  SAME DEMAND, TWO PROBLEM CLASSES

  GREENFIELD                                        BROWNFIELD
  every site a free choice                          every site a status quo

     D1 o         o g1                               D1 o----[A] Ashfield   620k
     D2 o   [g2]      o g3                          D2 o----[B] Bramble    980k
     D3 o         o g4                              D3 o----[C] Calder     860k
     D4 o   [g5]                                    D4 o----[D] Dunmore    740k
           \    /
            \  /   model picks {g3, g5, g2}              options priced at each site:
             [D]   cost 100.0, all-in                     A keep + reconfigure  -1.52
                   change: 3 opens, 0 closes             A close               +1.53
                   no closure cost, no legacy,           A downsize            +1.47
                   no permission, no notice              C expand              -0.90
                                                          D do nothing           0.00

  the greenfield number is a TOTAL cost of a new network
  the brownfield numbers are DELTAS against a baseline that already runs
```

The pictures differ in four ways, and those four ways carry the topic.

**Property 1 — Greenfield has no incumbent; brownfield's incumbent is the competitor.** In the
greenfield problem the model compares designs against each other; in the brownfield problem it
compares each design against the network that will run if nobody decides anything. That baseline
is not zero cost and it is not "today's cost": it is today's cost rolled forward with the lease
renewals, the wage settlements, the known capacity shortfall, and the service trajectory already
committed. A study that omits it has no denominator, and its savings percentages are ratios
against an imaginary world.

**Property 2 — In brownfield, the dominant cost is already spent, so the objective is a
difference.** A greenfield facility's fixed cost is a forecast; a brownfield facility's fixed cost
is a decision — which of its existing costs will disappear if it closes. The first comes from an
engineering estimate, the second from a management accounting judgement, and the second is wrong
more often because it looks like it should be easy.

**Property 3 — In brownfield, the binding constraints are physical and legal, not economic.** A
greenfield site can be designed to any specification. An existing site is a building with a clear
height, a floor load, a column grid, a dock count, a lease with a break clause in 2031, a landlord who
must consent to alterations, a planning use class, and a workforce with contractual notice. Some of
these are prices; most are walls, and a model that prices a wall will recommend knocking it down —
correct in the model, impossible in the building.

**Property 4 — In brownfield, the output is a sequence, not a set.** The greenfield answer is an open
set: build these three, serve these markets. The brownfield answer is a dated sequence: close this one
after the lease break, expand that one in the same financial year, keep the third and re-rack it, and
do not touch the fourth until the transport contract is re-tendered. The set is the destination; the
sequence is the decision.

### 3.1 Why the two traps are mirror images

The sunk-cost trap and the sunk-capacity trap are usually taught as separate errors, one
behavioural and one technical. They are the same error seen from two sides: **the wrong
counterfactual, priced at the wrong moment.**

- The **sunk-cost trap** asks "what did we pay for this?" and answers a question about the past.
  It keeps a site because the capital is recent, because a closure would "waste" the investment,
  or because writing it down would hit this year's results. All three are real organisational
  forces. None changes which future is cheaper.
- The **sunk-capacity trap** asks "what does this cost us now?" and answers zero, because the
  asset is owned. It is the same mistake with the sign flipped: it prices the past at full value
  in one case and at no value in the other, and in both cases it fails to price the *future* cost
  of retaining the asset.

The correct quantity is neither. It is the **incremental future cost of keeping the site**, which
has three parts: the avoidable cash cost of operating it (lease or rent, site labour, utilities,
maintenance, insurance, site-specific information technology); the one-off cost avoided by not
closing it (severance, make-good, dual running); and the opportunity cost of not disposing of it
— the disposal value foregone, if a credible buyer exists.

The third is the one most often missing. A site with a book value of 2.3 million and a realistic
disposal value of 0.5 million is worth 0.5 million to the company if kept for another purpose and
0.5 million in cash if sold. That 0.5 million is a cost of retention, and a model that charges
closure costs but never charges foregone disposal is systematically biased toward keeping.

> **Note:** you cannot charge both the disposal proceeds as a benefit of closure and the
> foregone disposal as a cost of retention. That double counts the same 0.5 million twice, in
> the same direction, and makes closure look better than it is. Choose one convention and state
> it in the cost basis.

### 3.2 Why brownfield optima cluster near the status quo

There is a structural reason, not merely an organisational one, why brownfield studies so often
conclude "change less than you thought". The objective surface is flat near the status quo.

When fixed costs dominate flow costs — the normal condition in distribution networks, where the
fixed term can be eight times the flow term at realistic scale — many open sets cost almost the
same. Topic 04's worked example had a 250 thousand currency unit margin between the best and
second-best design on a 1,955 thousand cost base, or 13%. Add a closure cost that falls only on
the designs that close, a transition cost that scales with the number of sites changed, and a
service risk that rises with the number of simultaneous changes, and the ranking can invert. The
optimiser's preferred design and the low-change design are often within the width of the
transition-cost estimate, which means the tie-break is not cost at all. It is executability, and
executability favours fewer changes.

This is why the honest output of a brownfield study is frequently not one design but a frontier:
"these two options are indistinguishable on cost; one requires a lease exit and a 400 kilometre
transport penalty, the other a capital case and a landlord's consent. Choose on which risk you
can manage."

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Greenfield design | A network problem in which every candidate facility is new and no asset is inherited | Used for any study that considers a new site | Problem class |
| Brownfield design | A network problem in which most facilities already exist and the decisions are keep, expand, reconfigure, downsize, or close | Used loosely to mean "an optimisation that includes existing sites" | Problem class |
| Hybrid design | A study that runs a greenfield ideal to establish a bound, then designs a brownfield migration toward it | Treated as a compromise rather than a two-stage method | Problem class |
| Do-nothing baseline | The priced counterfactual against which every option is measured | Reported as zero, or omitted | Currency per period, discounted |
| Sunk cost | A cost already incurred that no future decision can recover | Treated as a reason to continue | Currency, historical |
| Sunk-cost fallacy | The tendency to continue an endeavour because of prior investment rather than future returns | Used as a personal accusation rather than a description of a decision | Behavioural, no unit |
| Sunk-capacity trap | Modelling an owned asset at zero fixed cost, so the model can never justify closing it | Confused with sunk-cost fallacy | Modelling error |
| Avoidable cost | A cost that disappears if the activity stops | Taken as the ledger total of the cost centre | Currency per year |
| Write-down | A reduction in the carrying value of an asset, recognised as a loss | Cash-flowed as if it were a payment | Currency, non-cash |
| Disposal proceeds | Cash received on sale of an asset | Ignored in the closure case | Currency, one-off |
| Stranded overhead | Overhead that continues after the activity it supported has stopped | Excluded because "overhead is allocated" | Currency per period, with a duration |
| Closure cost | The total one-off cost of ceasing operation at a site | Entered as the annual fixed cost | Currency, one-off |
| Transition cost | The cost of moving from one network configuration to another | Treated as a single line item | Currency, dated profile |
| Cost of change | The umbrella term for reconfiguration, transition, severance, dual running, and stranded overhead | Omitted from the objective, then discovered in delivery | Currency, one-off and recurring |
| Dual running | Operating two sites in parallel while volume migrates from one to the other | Assumed to be a few weeks of rent | Currency per month, plus productivity loss |
| Severance | Payment to an employee on termination, statutory or contractual | Estimated per head without checking the schedule | Currency per employee |
| Notice period | The contractual or statutory period between notification and termination | Treated as a decision variable | Days or weeks |
| Consultation period | The period of formal consultation with employee representatives before redundancies take effect | Confused with the notice period | Days |
| Dilapidations | The contractual obligation to restore leased premises at the end of the term | Discovered at handback | Currency, one-off |
| Lease surrender | Negotiated early termination of a lease by agreement with the landlord | Treated as a right rather than a negotiation | Currency, one-off |
| Break clause | A contractual date at which a lease may be terminated on stated notice | Confused with lease expiry | Date plus notice period |
| Landlord consent | The landlord's permission for assignment, subletting, or alterations | Assumed to be automatic | Condition precedent |
| Use class | The planning category that governs what a building may be used for, and what consent exists for | Ignored when a site changes role | Regulatory code |
| Clear height | The usable vertical distance under the lowest obstruction in a warehouse | Treated as a layout detail rather than a capacity ceiling | Metres |
| Floor loading | The safe distributed load a slab can carry | Ignored when automated storage is proposed | Kilonewtons per square metre |
| Step-fixed cost | A cost that is fixed within a volume band and rises at a threshold | Modelled as purely fixed or purely variable | Currency per period, per band |
| Churn limit | A constraint on the number of sites changed in a given period | Absent, so the model recommends everything at once | Sites per period |
| Capital envelope | The maximum transition spend available in a period | Ignored, so the path is unaffordable even when optimal | Currency per period |
| Migration path | The dated sequence of network states from today to the target | Confused with the target network | Sequence of dated states |
| Critical path | The longest chain of dependent activities that determines the earliest completion date | Confused with the shortest route | Days |
| Option set | Three to five named designs, each costed against the same baseline | Replaced by a single recommendation | Set of designs |
| Payback period | Time until cumulative undiscounted savings recover the one-off cost | Computed on gross saving rather than net | Years |
| Net present value (NPV) | The sum of discounted future cash flows, including the one-off cost | Computed without the transition cost | Currency, at a stated rate |
| Internal rate of return (IRR) | The discount rate at which NPV equals zero | Compared across projects with different horizons | Percent per year |
| Hurdle rate | The minimum return a project must earn to be approved | Treated as a property of the project rather than the company | Percent per year |

Two deserve expansion, because they are where the argument breaks.

**Avoidable versus ledger cost.** The ledger cost is what the cost centre reports; the avoidable
cost is what disappears if the site stops. The gap is allocated overhead, shared services, and
anything recharged elsewhere rather than eliminated. In the worked example it is 0.16 million on
a 6.40 million site, or 2.5%; in other networks it runs to 30 or 50%. The size of the gap is a
property of how much of the site's cost is genuinely local, not of accounting.

**Notice versus consultation.** These are different clocks and they run partly in sequence. In
the United States the Worker Adjustment and Retraining Notification (WARN) Act requires 60
calendar days' written notice for covered plant closings and mass layoffs. In the European Union,
Directive 98/59/EC requires consultation "in good time" and bars redundancies taking effect
earlier than 30 days after notification to the public authority, extendable to 60. Both are
floors and national law is often stricter. The consequence: **the closure date is a parameter,
not a decision variable.**

## 5. The mechanics, step by step

This section is the procedure for a brownfield study. It assumes you can already write the
facility-location model from topic 04 and the multi-period model from topic 05. What it adds is
the sequence of judgements that the greenfield procedure does not contain.

### Step 1 — Classify the problem, and write the classification down

**Input and output:** the decision sentence from topic 01 becomes a one-line classification —
greenfield, brownfield, or hybrid — with the decision verbs named.

**Why it matters:** the classification fixes the cost basis, the constraint set, the output format,
and the review standard. It is usually left implicit, which is how a brownfield decision gets
modelled as greenfield and then rejected.

**The test is the verb, not the industry.** "Which of six regions should host our first European
DC" is greenfield even though the company is old. "Close Ashfield, expand Calder, or keep both" is
brownfield even for a one-site startup, because the status quo is an option. For a hybrid, say what
each stage is for: the greenfield run supplies a bound and a direction; the brownfield run supplies
the decision.

### Step 2 — Price the do-nothing baseline before anything else

**Input and output:** the current footprint, lease register, contracted wage and rent escalation,
and known capacity shortfall become the status quo's total network cost, by year, discounted at the
hurdle rate, on the cost basis every option will use.

**Why it matters:** every number in the study is a difference from this one. Three errors recur.
**Assuming the baseline is flat** — leases renew at market rent, wages escalate, and a site at 94%
of planning capacity needs capital whether or not the network changes; ignoring both understates
the value of change, sometimes by the entire saving reported. **Assuming the baseline is
infeasible** — sometimes it is, and then the baseline is the *minimum-compliance* network, the
least-cost set of interventions that keeps serving demand, which must be built and costed.
Comparing an option against an infeasible present is comparing against a fantasy. **Assuming the
baseline is free** — the status quo consumes capital and management attention like any other
option. If the minimum-compliance baseline costs more than one of the change options, the study has
found its headline before solving anything.

### Step 3 — Build the existing-asset register

**Input and output:** the site list, lease register, fixed-asset register, headcount, and permits
become one row per site with the fields below, each with a source and a date.

| Field | Why the model needs it | Typical source | Where it goes wrong |
|---|---|---|---|
| Site identifier | Join to freight, cost, and payroll data | GS1 Global Location Number (GLN) or internal code | Codes differ across systems; the join silently drops sites |
| Throughput and capacity basis | The capacity constraint | Warehouse management system, engineering records | Nameplate entered as effective capacity |
| Annual ledger cost | The starting point for the avoidable test | Cost centre report | Includes allocations that survive closure |
| Avoidable annual cost | The fixed cost in the model | Management accounting judgement | Taken as the ledger total |
| Lease term and break clause | Constraint on closure date and on avoidable rent | Lease register, not the property team's memory | Break date confused with expiry |
| Disposal value | Opportunity cost of retention | Indicative valuation or broker opinion | Assumed zero, or assumed equal to book value |
| Headcount and severance schedule | Closure cost | HR system plus the applicable statutory and contractual schedule | Averaged per head across a wide distribution |
| Permits and use class | Constraint on reconfiguration | Environmental and planning records | Assumed to transfer with a change of role |
| Physical specification | Capacity ceiling for a change of role | Building survey | Absent from the model entirely |

**Why it matters:** this register is the brownfield equivalent of topic 04's candidate set — the
most consequential input, assembled most casually. If a lease break date is unknown, it is a data
gap that changes the answer: obtain it, or treat the site as unleasable for the horizon.

### Step 4 — Apply the avoidability test to every site's cost

**Input and output:** the cost centre report and the register become one avoidable annual cost per
site, reconciled from the ledger and signed off.

The test is a question asked line by line: **if this site ceased operation on the first day of
next year, would this cost disappear?** The answers fall into four classes.

| Class | Examples | Treatment |
|---|---|---|
| Avoidable, immediate | Site labour on the site's own payroll, utilities, variable maintenance, site-specific insurance | In the fixed cost |
| Avoidable, delayed | Rent under a lease with a tail, maintenance contracts with notice periods, information technology contracts with minimum terms | In the fixed cost for the unexpired period, then removed; the tail is also a closure cost |
| Step-fixed | Labour that steps at a shift threshold; space that steps at a racking bank | Piecewise, with the bands stated |
| Not avoidable | Allocated corporate overhead, shared services, regional management, group insurance | Excluded from the fixed cost, and treated as stranded overhead if it persists |

**Why it matters:** this classification decides whether consolidation looks profitable. Topic 04 put
the overstatement from including allocations at 30 to 50% in the cases that course uses; here the
gap is smaller and arises from a different cause, which is the point — the size of the error varies,
and you cannot assume it away in either direction. If the avoidable cost cannot be signed off by a
management accountant, the study is using the analyst's opinion as a financial input: say so, and
run both figures.

### Step 5 — Decide the treatment of sunk capital, and write it in the cost basis

**Input and output:** the fixed-asset register, the depreciation policy, and the disposal evidence
become a stated convention, applied consistently, covering four items: historic capital and
depreciation are **excluded** from the annual fixed cost, being a record of a past decision while
the model's fixed cost is a future cost; **write-down** is included in the transition account, not
the annual cost, and labelled non-cash; **disposal proceeds** reduce the net closure cost, which
means getting an indicative valuation rather than using book value; and **foregone disposal value
is a cost of retention, or disposal proceeds are a benefit of closure — not both.**

**Why it matters:** this is where the two traps of section 3.1 are made or avoided. Setting the
existing site's fixed cost to zero is the sunk-capacity trap; setting it to depreciation plus a
return on capital is the sunk-cost trap. Both are conventions, both are defensible-sounding, and both
produce a systematically biased answer. If the company's capital allocation process requires a return
on existing assets (a "capital charge"), that charge is a management device, not an economic cost:
show it separately rather than embedding it, or the model will prefer closure for a reason that
exists only inside the company's accounting.

### Step 6 — Encode the option set per site as discrete alternatives

**Input and output:** the register and the physical and legal constraints become, for each site, a
small set of mutually exclusive states, each with a fixed cost, a capacity, a one-off cost, and an
earliest effective date.

The five verbs, and what each means in model terms:

| Verb | Meaning | Model representation | Typical reason |
|---|---|---|---|
| Keep | Continue as today | Base state, avoidable cost, base capacity | Lowest change cost |
| Expand | Add capacity at the site | Additional capacity block with its own fixed cost and capex | Demand growth, consolidation into a good site |
| Reconfigure | Change what the site does, or how | Alternative capacity and cost profile for the same site; may remove arcs | Role change, automation, product mix shift |
| Downsize | Give back capacity | Reduced capacity and fixed cost, often in steps; sublet or surrender | Demand decline, right-sizing |
| Close | Cease operation | Zero capacity, closure cost, delayed effective date | Cost, redundancy, end of lease |

Represent these as a small set of binary choices per site rather than a continuous capacity
variable. A site cannot be 37% expanded; it can have a mezzanine or not. Discreteness is not a
simplification here, it is the physical truth.

**Why it matters:** the option set is the output of the study, so it has to be an input to the
model. A model that can only open and close will report open-or-close, and the business will
correctly observe that it was never offered the option it wanted. If two states at a site are not
mutually exclusive in reality — a site can be expanded and later downsized — you need the
multi-period form from topic 05, not a single-period choice set.

### Step 7 — Encode reconfigurability as capacity ceilings and arc removals

**Input and output:** the building survey, the lease, the planning status, and the labour agreements
become, per site and per product family, a capacity ceiling that reflects what the building can
physically and legally do, and a set of market–site arcs that are forbidden.

The constraints that matter, and how each enters the model:

| Constraint | Physical or legal | Model representation | What it prevents |
|---|---|---|---|
| Clear height | Physical | Capacity ceiling per cubic measure, hence per unit | Re-racking a low building into a high-throughput hub |
| Floor loading | Physical | Capacity ceiling for heavy or automated storage | Installing automation on an inadequate slab |
| Column grid | Physical | Capacity ceiling and layout cost penalty | Dense racking in a wide-bay building |
| Dock count | Physical | Throughput ceiling per period | A site that stores fine and cannot ship |
| Site boundary | Physical | Hard capacity ceiling; no expansion state offered | A phased expansion that has nowhere to go |
| Lease term and break clause | Legal | Earliest effective date for any change; rent tail as a fixed cost | A closure that cannot happen when the model says |
| Landlord consent | Legal | Constraint or a stated probability; a cost if obtainable at a price | Alterations and sublets assumed to be free |
| Planning use class | Legal | Arc removal or capacity removal for the new role | A change of use that will not be granted |
| Environmental permit | Legal | Constraint on what may be stored or processed | Hazardous or refrigerated volume at an unpermitted site |
| Labour agreement | Contractual | Minimum notice, consultation period, redeployment obligation, change cost | A closure date earlier than the agreement permits |

**Why it matters:** the distinction between a constraint and a cost is the difference between a
study that respects reality and one that buys its way past it. If landlord consent can be
obtained for a price, it is a cost, and that price belongs in the transition account. If it
cannot be obtained at all, it is a constraint, and the state should not be offered. When you are
unsure which it is, offer the state with a high cost and a long lead time and let the option set
show what it is worth; that is more honest than either excluding it or assuming it is free.

### Step 8 — Build the cost-of-change account as a separate workstream

**Input and output:** the transition plan, the HR schedule, the property advice, and the project
plan become a dated schedule of one-off and temporary costs, split into included and excluded items,
with a range on each.

| Item | Typical driver | Where it belongs | Commonly omitted because |
|---|---|---|---|
| Reconfiguration capital | Racking, material handling equipment, fit-out, information technology | Transition account, dated | It is treated as a capital case separate from the network study |
| Dual-running operating cost | Two sites live, split volume, lower productivity on both | Transition account, per month | "The buildings are already paid for" |
| Severance and notice | Headcount, schedule, tenure | Transition account, dated | Estimated per head without the schedule |
| Consultation and garden leave | Legal minimum plus company policy | Transition account, dated | Treated as part of severance |
| Recruitment and training at receiving sites | New headcount, new systems | Transition account, dated | Assumed to be absorbed in existing budgets |
| Inventory build ahead of cutover | Weeks of cover, working capital, obsolescence risk | Transition account, dated | Invisible in a cost model with no inventory |
| Stranded overhead | Regional management, shared services, contract minimums | Transition account, with a duration | "Overhead is allocated, so it is not real" |
| Customer disruption | Service penalties, lost sales, expedite premium | Transition account, estimated range | Not visible in the facility model |
| Dilapidations and make-good | Lease terms, condition at handback | Transition account, dated | Discovered at handback |
| Disposal, remediation, and site clearance | Environmental condition, asset removal | Transition account, net of proceeds | Assumed to net to zero |
| Programme cost | Project management, temporary resource, assurance | Transition account | Treated as overhead |

**Why it matters:** the cost of change is the term that decides most brownfield cases and is
present in the fewest models. It is also the term with the widest uncertainty, which is why it
needs a range rather than a number.

> **Note:** do not annualise a one-off cost by dividing it by the horizon. A 7.55 million
> one-off spread evenly over five years is not 1.51 million a year in any decision-relevant
> sense: the cash leaves early and the saving arrives late, and the whole point of discounting
> is that those are not the same. Keep the profile dated.

### Step 9 — Assemble the multi-period model, with change binaries and a churn limit

**Input and output:** the site states, the capacity ceilings, the demand by period, and the
cost-of-change schedule become a model that chooses, for each site and period, which state it is in,
and pays the transition cost whenever the state changes.

The structure is topic 05's multi-period model with three additions: **state binaries**
`w_jst` = 1 if site `j` is in state `s` in period `t`, with `sum_s w_jst = 1` for every site and
period; **transition binaries** `u_jst` = 1 if site `j` moves into state `s` at the start of
period `t`, linked by `u_jst >= w_jst - w_js,t-1`; and a **churn limit**
`sum_j sum_s u_jst <= K_t` alongside a **capital envelope** `sum_j sum_s capex_js u_jst <= B_t`.

**Why it matters:** without the churn limit, a model with several periods will reconfigure the
entire network in year one, because transition cost is small against discounted saving and the
model has no theory of management capacity. Two changes a year is a common figure for a mid-sized
network; yours comes from the programme office. This is the dynamic facility location problem —
Wesolowsky and Truscott on relocation cost in 1975, Van Roy and Erlenkotter on opening and closing
in 1982. If the model is infeasible with a realistic churn limit, the target cannot be reached on
the schedule the business wants, which is a finding about the path.

### Step 10 — Check the path, not only the destination

**Input and output:** the solved multi-period model, or a hand-built transition plan, becomes — for
every date in the horizon — total available capacity against demand, by product family and region.

A target can be feasible and its path infeasible. The classic failure is a capacity trough: the
closing site's volume must go somewhere before the receiving site's expansion is commissioned. A
14-month engineering, procurement, and construction (EPC) lead time against a 5-month
consultation leaves a nine-month trough and a dead plan. Three checks, in order: **aggregate
capacity by date**, with the utilisation ceiling applied; **regional capacity by date**, because a
national total hides a regional shortage and a service promise is regional; and **product-family
capacity by date**, because ambient capacity cannot absorb chilled volume. If a trough exists,
cost four remedies — delay the closure, accelerate the expansion, bridge with temporary capacity,
or change the destination. The cheapest is usually the one nobody proposed: delay the closure and
absorb the dual-running cost.

### Step 11 — Optimise the destination and the path under different objectives

**Input and output:** the same instance, solved twice, becomes a destination design, a migration
path, and a reconciliation of the two.

This is the part practitioners most often skip, and it is where the honest answer lives.

| | Destination | Path |
|---|---|---|
| Objective | Steady-state annual cost at the target year, discounted | NPV of incremental cash flow from today, including all transition cost |
| Constraint set | Capacity and demand at the target year | Capacity at every date, capital envelope, churn limit, notice periods |
| Time treatment | One representative year | Every year, with dated cash flows |
| Data quality | Costs at a future price level | Transition costs, which are the least reliable numbers in the study |
| Typical output | The best network | The best route to a good network |
| Typical disagreement | The destination ignores that the route destroys value | The path locks in a design that is never optimal |

The two can disagree in both directions. A destination that is best in 2032 may require a
transition that destroys value if executed in 2027, because the demand that justifies it has not
arrived; a path that phases change slowly to protect service may accumulate so much dual-running
cost that it never earns back.

**Why it matters:** running one model with both objectives produces a compromise that satisfies
neither. Run two, and report where they differ; the differences are the decisions. If the
destination and the path agree, the study is simple and you should say so. If they disagree, the
option set should contain both, costed on the same basis, and the committee should be told what
the disagreement is about.

### Step 12 — Build the option set and price every member against the baseline

**Input and output:** the destination, the path, the near-optima, and the business's own proposal
become three to five named options, each with an NPV against the do-nothing baseline, a one-off
cost, a payback, a key risk, and a list of conditions that must hold.

A workable set contains: **do nothing**, or minimum compliance if do-nothing is infeasible; **the
destination optimum**, transitioned as fast as the constraints permit; **a low-change option** —
keep and reconfigure, or downsize — which is what the business will compare everything else to; **a
phased version** of the destination, respecting the churn limit and the capital envelope; and **the
business's own proposal**, if it differs, costed honestly.

**Why it matters:** the option set is the product. It converts a yes-or-no argument into a
choice among costed futures, and it makes the constraints visible: the difference between
options 2 and 4 is the price of the churn limit, and the difference between 2 and 3 is the price
of the lease. If every option costs more than the baseline, the baseline is probably mispriced:
check the escalation assumptions and the capacity shortfall before concluding that the status quo
is optimal.

### Step 13 — Report to the brownfield standard

A greenfield study is reviewed on its candidate set, capacity basis, cost basis, and gap. A
brownfield study needs those and four more, because its claim is comparative rather than
absolute.

| # | Item | What must be stated | Why a reviewer asks |
|---:|---|---|---|
| 1 | Baseline | The do-nothing (or minimum-compliance) network, costed by year, discounted, with escalation and known capex | Every number in the study is a difference from this |
| 2 | Option set | Every option, with the do-nothing case first, costed on the same basis | Because the recommendation is only as good as the alternatives priced |
| 3 | Cost of change, included | Each line item in step 8 that is in the numbers, with its amount and date | Because the omitted items are where the business case fails |
| 4 | Cost of change, excluded | Each item deliberately left out, with the reason and, where possible, an order of magnitude | Because an unstated exclusion is indistinguishable from an oversight |
| 5 | Migration path | The dated sequence, the earliest effective date per closure, and the capacity check by date | Because a feasible destination can have an infeasible path |
| 6 | Critical assumptions | Lease exit, landlord consent, planning, consultation, EPC lead time, transport penalty — each with the value at which the answer flips | Because these, not the objective value, are what the committee must secure |
| 7 | Destination versus path | Where the two disagree, and by how much | Because the disagreement is the decision |

Six of these are in the greenfield standard from topic 04 plus one, and the addition that matters
most is item 4: a brownfield study that lists only what it included cannot be reviewed, because the
reviewer cannot tell whether the exclusion was a judgement or an omission.

**The last step, which is not a step.** Before the report leaves the building, ask the question the
committee will ask: **what would have to be true for us to do nothing?** If you cannot answer it in
one sentence with a number in it, the baseline has not been priced.

## 6. Worked example

Ashfield is one of four DCs in a regional grocery distribution network. A study is commissioned
to decide its future. The instance is small, the arithmetic is checkable, and the result is
deliberately conditional: the answer flips on a lease clause and on a transport estimate, which
is what a brownfield study usually does. All figures are illustrative and internally consistent.
Currency is GBP, price year 2026.

### 6.1 The site, the network, and the decision

| Item | Value |
|---|---|
| Ashfield throughput | 620,000 cases per year |
| Network throughput | 3,200,000 cases per year |
| Ashfield share of volume | 19% |
| Ashfield headcount | 160 |
| Ashfield planning capacity | 660,000 cases per year (94% utilised) |
| Remaining lease | To 2031, no break clause, no assignment right without consent |
| Disposal value, indicative | 0.50 million |

Demand is assumed flat over the five-year horizon to 2030. That simplification matters: with flat
demand the status quo is feasible, so do-nothing is a real option. At 4% growth Ashfield would
exceed planning capacity in 2028 and "do nothing" would become "do the minimum", which is a
different baseline and a different answer.

### 6.2 The ledger and the avoidable cost

| Cost line | Annual, GBP million | Avoidable on closure | Why |
|---|---:|---|---|
| Site labour, 160 staff | 4.16 | Yes | The site's own payroll |
| Rent | 1.10 | No, during the lease | Lease runs to 2031 with no break |
| Utilities | 0.38 | Yes | Site-specific |
| Maintenance | 0.34 | Yes | Site-specific contract |
| Insurance | 0.12 | Yes | Site-specific |
| Site information technology | 0.14 | Yes | Site-specific |
| Allocated corporate overhead | 0.16 | No | Reallocated, not eliminated |
| **Total** | **6.40** | **5.14** | |

The avoidable cost is 5.14 million against a ledger of 6.40 million. A business case built on the
ledger would overstate the annual benefit by 1.26 million, or 25% — smaller than the 30 to 50% range
topic 04 reports, and from a different cause: not a large allocation, but an unavoidable lease. Two
facts organise what follows: the rent of 1.10 million is 21% of the avoidable cost and is not
avoidable until 2031, and the site is at 94% of planning capacity, so it is not a candidate for
downsizing without losing throughput.

### 6.3 The cost to close

| Item | GBP million | Note |
|---|---:|---|
| Severance and notice, 160 staff at 15,000 average | 2.40 | Statutory plus contractual, per the schedule |
| Write-down, book value 2.30 less disposal 0.50 | 1.80 | Non-cash; included for covenant impact |
| Dual running, 6 months | 0.90 | Rent, split-volume inefficiency, shuttle, supervision |
| Stranded overhead, 18 months | 0.60 | Regional management and an information technology minimum term |
| Dilapidations and make-good | 0.50 | Per the lease schedule |
| Inventory build, one-off carrying cost | 0.15 | Six weeks of cover ahead of cutover |
| Receiving-site capital to absorb the volume | 1.20 | Racking and material handling equipment at two sites |
| **Total** | **7.55** | |

The write-down generates the most argument and affects the cash least. It is included because the
study feeds a capital committee that will also see the accounting impact; excluding it would
produce a different conversation two months later. The report says so explicitly.

### 6.4 The annual saving, and the transport penalty

Closing Ashfield does not save 5.14 million. The 620,000 cases have to be served from somewhere,
and the remaining three sites are on average 190 kilometres further from Ashfield's customer
base.

| Item | GBP million per year |
|---|---:|
| Avoidable annual cost at Ashfield | +5.14 |
| Extra line-haul and delivery to remaining sites | −1.85 |
| Extra variable handling at receiving sites | −0.70 |
| **Net annual saving** | **2.59** |

### 6.5 Payback, NPV, and the notice period

Undiscounted, payback is `7.55 / 2.59 = 2.92 years` — the number that would appear in a first
draft. It is wrong, because it assumes the saving starts on day one. It cannot. A closure
affecting 160 employees triggers collective consultation and then notice; assuming a 90-day
consultation followed by notice served within it, the earliest effective date is roughly five
months after announcement, so year one captures seven months of saving.

| Year | Cash flow, GBP million | Cumulative |
|---|---:|---:|
| 0 | −7.55 | −7.55 |
| 1 | +1.51 (2.59 × 7/12) | −6.04 |
| 2 | +2.59 | −3.45 |
| 3 | +2.59 | −0.86 |
| 4 | +2.59 | +1.73 |
| 5 | +2.59 | +4.32 |

Payback arrives at 3.33 years, not 2.92. The notice period costs 0.41 years of payback and about
1.0 million of NPV. It is the single largest "soft" factor in the case and it is a legal
requirement, not a preference.

At a 9% hurdle rate over five years, the annuity factor for years 1 to 5 is 3.8897; discounting
year one for the delay scales its saving by 0.5352 rather than 0.9174, so the effective factor is
`0.5352 + (3.8897 − 0.9174) = 3.5075`. NPV is therefore `2.59 × 3.5075 − 7.55 = 9.08 − 7.55 =
+1.53 million`. The internal rate of return (IRR) is approximately 16% per year on the same cash
flows. Both numbers are positive, and both are thinner than the 2.92-year payback suggested.

### 6.6 What must be true for the closure to pay

Four conditions, each with a threshold computed from the same arithmetic.

**1. The lease must be partly escaped.** If the landlord refuses surrender and refuses consent
to a sublet, the 1.10 million rent continues and the net annual saving falls to 1.49 million.
NPV becomes −2.32 million. The break-even is an escape of **0.44 million a year, 40% of the
passing rent** — a sublet of about 40% of the space, or a surrender at that price. This is the
negotiation target, and it is a far more useful output than "close the site".

**2. The transport penalty must not be much worse than estimated.** Break-even is a penalty of
2.99 million, against the 2.55 million assumed — a margin of 0.44 million, or 17%. A 17% error in
a transport estimate is entirely ordinary. This is the most fragile assumption in the case, and
it is not the one the committee will ask about.

**3. Severance must not exceed about 24,600 per head.** The one-off cost can rise to 9.08
million before NPV reaches zero, which allows severance of 3.93 million rather than 2.40
million. That is a comfortable margin, and it means the case is far more sensitive to transport
than to headcount.

**4. The receiving sites must be able to take the volume.** The 1.20 million of capital assumes
two sites can absorb 620,000 cases within twelve months. If one of them needs a building
extension, the capital triples and the case disappears.

### 6.7 The option set

Five options, all costed against the do-nothing baseline on the same basis, at 9% over five
years.

| Option | One-off, GBP m | Net annual effect, GBP m | NPV, GBP m | Key condition |
|---|---:|---:|---:|---|
| 0. Do nothing | 0.00 | 0.00 | 0.00 | Demand stays flat; capacity holds to 2030 |
| 1. Close Ashfield, absorb into three | 7.55 | +2.59 | **+1.53** | Lease escape of 0.44 m/yr |
| 2. Close Ashfield, expand Calder first | 10.25 | +3.24 | +1.11 | 3.90 m of capital approved in year 0 |
| 3. Keep and reconfigure Ashfield | 2.10 | +0.15 | −1.52 | A capacity or service reason, which flat demand does not supply |
| 4. Downsize Ashfield, sublet 30% | 0.55 | +0.52 | +1.47 | Sublet at 65% of passing rent; 12 roles removed |

Read the table in three ways.

**The ranking is not the finding.** Options 1 and 4 differ by 0.06 million on a network of
roughly 21 million annual cost — 0.3%. That is inside the error on both the transport estimate and
the sublet assumption. They are tied, and the tie-break is which risk the company can manage: a
lease negotiation and a volume migration, or a sublet and a smaller reconfiguration.

**Option 2 is worse on NPV but better on robustness.** Expanding Calder first costs 2.70 million
more and recovers 0.65 million a year of transport penalty. It does not pay back inside five
years, but it removes the capacity trough and the 17% transport margin. A company that expects
growth should prefer it; one that expects flat demand should not pay for it.

**Option 3 destroys value, and that is a result.** Reconfiguring a site already at 94% of
planning capacity, to serve flat demand, is capital spent on a problem the network does not have.
Brownfield studies are often commissioned by someone who wants permission to reconfigure; the
model's job is to price it, not to agree.

### 6.8 What the model could not tell us

1. **Will the landlord consent?** This single answer is worth 3.85 million of NPV — the
difference between +1.53 and −2.32. No model can price a negotiation.
2. **Is the 190-kilometre average distance right?** It came from a centroid calculation that
   topic 02 warned about. The 17% margin is narrower than that method's typical error.
3. **What happens to service during the migration?** The model has no service variable. A 620,000-case
   transfer over six months will produce late deliveries, and the penalty is not in the 7.55 million.
4. **What does the workforce do?** Consultation will surface alternatives — redeployment, reduced
   hours, a phased closure — that change the cost and the date, and that the model cannot generate.

Four questions, none answered by the arithmetic. That is the normal condition of a brownfield
study, and it is why the output is an option set with conditions rather than a recommendation.

## 7. Code, query, or calculation

The calculation below is the whole of section 6.3 and 6.4 in one page. It is the form to use on
a real site, because it can be checked by a management accountant line by line and it makes the
two contested items — the rent and the transport penalty — impossible to hide.

```text
ASHFIELD DC — cost to close against annual saving   (GBP m, 2026 prices)
annual site cost per ledger                                    6.40
  less allocated overhead, not avoidable                      -0.16
  less rent to 2031, no break clause, not avoidable           -1.10
  = avoidable annual cost if closed                            5.14
  less extra line-haul and delivery to remaining sites        -1.85
  less extra variable handling at receiving sites             -0.70
  = net annual saving                                          2.59

one-off cost to close
  severance and notice, 160 staff x 15,000                     2.40
  write-down, book value 2.30 less disposal 0.50               1.80
  dual running, 6 months                                       0.90
  stranded overhead, 18 months                                 0.60
  dilapidations and make-good                                  0.50
  inventory build, one-off carrying cost                       0.15
  receiving-site capex to absorb the volume                    1.20
  = total one-off                                              7.55

payback, no delay             7.55 / 2.59 = 2.92 years
payback, 5-month consultation  cumulative -7.55 -6.04 -3.45 -0.86 +1.73
NPV at 9%, 5 years, delayed   +1.53
rent escape for NPV = 0       0.44 per year, 40% of 1.10
```

Three readings matter.

**The two subtractions at the top are the whole brownfield argument.** Every number below them
depends on 5.14 rather than 6.40. Had the analyst used the ledger, the NPV would be
`(6.40 − 2.55) × 3.5075 − 7.55 = +5.95 million`, four times the true figure. The error is not in
the arithmetic; it is in the first line.

**The payback and the NPV disagree in tone.** A 2.92-year payback reads as a strong case; a
1.53 million NPV on a 7.55 million commitment reads as a marginal one. Both are correct: the
payback ignores discounting and the delay, the NPV ignores nothing except the items in section 6.8.
Report both, and explain the gap. **The last line is the output:** "Close Ashfield" is a decision the
company may not be able to take, while "secure 0.44 million a year of rent relief and close
Ashfield" is a decision it can take, with a named owner, a counterparty, and a date.

> **Note:** the single-period form above is a screening calculation. It answers "is this worth
> a business case", not "what should we do and when". For the latter, use the multi-period
> structure in step 9, which will also tell you whether the capital can be spent in the year the
> single-period model assumes.

## 8. Data, standards, and regulatory context

Brownfield work has a larger regulatory surface than greenfield work, because it touches employment
law, property law, and accounting standards that a new-build study never encounters. What follows is
the map, not advice; every item requires professional review in the relevant jurisdiction.

### 8.1 Employment and consultation

| Requirement | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| WARN Act, 29 U.S.C. §§2101–2109 | United States | 60 calendar days' written notice before a covered plant closing or mass layoff; covered employers generally 100 or more employees; a covered closing affects 50 or more at a single site in 30 days | B |
| Directive 98/59/EC on collective redundancies | European Union | Consultation "in good time" with worker representatives; redundancies cannot take effect earlier than 30 days after notification to the public authority, extendable to 60; national law is often stricter | B |
| National transfer-of-undertakings rules (for example, the UK Transfer of Undertakings (Protection of Employment) regulations) | Varies | Whether staff transfer with a contract to a new operator, and on what terms | B |
| Company redundancy policy | Individual employer | Payment above statutory minimum, enhanced notice, redeployment obligations | C |

The modelling consequence is the one already stated: **the earliest effective closure date is a
parameter derived from these rules and the company's own agreements, not a variable the optimiser
may choose.** In the worked example it cost 1.0 million of NPV; in a network with several closures
it can cost the entire first-year saving.

### 8.2 Property, planning, and building

| Requirement | Where | What it constrains | Grade |
|---|---|---|---|
| Lease terms: term, break clause, assignment, subletting, alterations | Contract | Whether a site can be exited, and when; whether space can be given back | C |
| Landlord's consent for alterations and subletting | Contract and case law | Whether a reconfiguration or sublet is possible at all | C |
| Planning permission and use class | National and local planning law | Whether a building may change role, for example from ambient storage to food processing | B |
| Building regulations, fire safety, and environmental permitting | National | Whether a change of use or a change of stored goods is permissible | B |
| Dilapidations and handback condition | Lease | The make-good cost at exit | C |

None of these appears in a standard facility-location formulation. All of them can remove a state
from the model, and each removal should be recorded with its source, because the first question at
review will be "why can we not do that with Ashfield".

### 8.3 Accounting and reporting

| Requirement | Where | Effect on the brownfield case | Grade |
|---|---|---|---|
| IAS 37 *Provisions, Contingent Liabilities and Contingent Assets* | Global, IFRS | A restructuring obligation may require a provision when a detailed plan exists and a valid expectation has been raised; the timing affects when the cost is recognised, not whether it is real | B |
| IAS 36 *Impairment of Assets* | Global, IFRS | Governs when and how a write-down is recognised | B |
| IFRS 5 *Non-current Assets Held for Sale* | Global, IFRS | A site designated for sale stops being depreciated, which changes the fixed cost basis | B |
| IFRS 16 *Leases* | Global, IFRS | Rent appears as depreciation and interest rather than rent; the model needs the cash cost, so a bridge is required | B |
| Segment and restructuring disclosure requirements | Listed companies | The reason a disclosed redundancy figure can be used as evidence, as in section 9 | B |

The accounting standards determine *when* a cost is recognised and *how* it is presented. They do
not determine whether it is a cost. A common error is to treat a provision as a cost that matters
and a write-down as one that does not, when the reverse is closer to the truth for a cash-based
decision.

### 8.4 Identity, units, and carbon

| Standard or convention | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 Global Location Number (GLN) | Site identity | The register, the freight ledger, and the payroll must key to the same site or the avoidable-cost calculation fails at the join | B |
| ISO 8601 | Dates and periods | Lease dates, notice dates, and effective dates must be unambiguous before a path can be built | B |
| ISO 4217 | Currency | Multi-country networks need a stated rate and rate date | B |
| ISO 80000 | Quantities and units | Capacity, throughput, and cubic measure must be comparable across a mixed portfolio | B |
| ISO 14083:2023 | Transport emissions quantification | If carbon is in the objective, the extra transport in a consolidation is a cost, and dual running emits twice | B |

The carbon point is worth one sentence because it catches brownfield studies specifically: a
consolidation that adds 190 kilometres of average line-haul reduces fixed cost and increases
emissions. Topic 09 owns the treatment; the brownfield obligation is to make sure the extra
transport appears in both the cost and the carbon account, not just the first.

### 8.5 What is requirement and what is convention

- **Requirement:** a closure cannot take effect before the statutory and contractual clocks permit,
  and rent under an unbroken lease does not disappear. **Convention:** that the clock is 90 days, and
  whether a sublet is available and at what fraction of passing rent.
- **Requirement:** an existing site's capacity is bounded by its building, a write-down is non-cash,
  and allocated overhead that survives closure is not a saving. **Convention:** which of clear
  height, floor loading, or dock count is the binding bound; whether the business case is stated on a
  cash or an accounting basis; and which ledger lines are allocations.

The pattern is the same as topic 04's: definitions are requirements, and the numbers you put into them
are judgements. A brownfield study contains more judgements than a greenfield one, which is why it
needs a longer cost basis rather than a shorter one.

## 9. Industry practice and case evidence

### 9.1 What the literature actually models

The facility-location literature is greenfield-dominated for a structural reason, and the chronology
is the evidence. Weber's continuous location problem (1909), Hakimi's vertex optimality (1964), and
Balinski's fixed-charge formulation (1965) all describe a choice among candidate sites with no
inherited state. The first treatments that give an existing facility a distinct status appear much
later: Wesolowsky and Truscott (1975) on the multi-period location-allocation problem with relocation,
and Van Roy and Erlenkotter (1982) on the dynamic problem with opening and closing. Grade A for both.
The review by Melo, Nickel, and Saldanha-da-Gama in the *European Journal of Operational Research*
(2009) makes the point implicitly: the model classes it surveys are distinguished by capacity,
commodities, periods, and uncertainty, and the legacy asset enters only through the multi-period
formulation. Grade A. The consequence for a practitioner is not that the literature is wrong, but that
the default formulation you will reach for, and the one you were taught, has no place to put a lease.

### 9.2 What the modelling literature says about transition

Martin and Altmann, in *CIRP Annals* (2026), model long-term reconfiguration and mid-term
sequencing in global production networks with dismantling, transport, and installation costs
explicit, and report that long-term optimal plans become infeasible when transition timing is
ignored. Abdoli, Fallahi, and Mokhtari, in *Transportation Research Part E* (2025), model
facilities changing role — production, storage, delivery — with explicit transformation time and
cost. Grade A for both; the first citation is taken from the repository record and should be
checked against the published version before being quoted. Both papers make the same argument
this topic makes: the destination and the path are different objects, and the second is where the
plans fail.

### 9.3 What companies disclose

| Case | What is disclosed | Grade | How to read it |
|---|---|---|---|
| Woolworths Group, ASX announcement, 23 June 2021 | Approval of a 76,000 square metre DC to replace a fragmented temperature-controlled network; approximately 44 million Australian dollars of redundancy recognised in the 2021 financial year, alongside a build cost of roughly 400 million | C | A rare quantified disclosure of the closure cost separate from the capital cost, from a listed company's statutory reporting |
| Marks & Spencer, Hardwick DC closure, 2018 | Closure of a DC affecting approximately 450 colleagues, with a third-party operator appointed at a new South East DC as part of a move to fewer, larger automated sites | C | Shows the pattern: a closure and a contract transfer in the same announcement, which is what the transition actually looks like |
| Marks & Spencer, distribution review, 2010 | Closure of 21 of 110 warehouses, with distribution costs reduced by approximately 15.9 million pounds, or 3.9% | C | Reported as a percentage of a distribution bill, not of a network cost, so it is not comparable with a study output |

### 9.4 What vendor and consultancy material reports

| Source | Claim | Grade | Reading |
|---|---|---|---|
| 4flow VIA documentation; KPI Solutions, OPSdesign, TREW, Fortna | Software distinguishes greenfield, brownfield, and hybrid modes, with brownfield restricting candidates to existing locations; brownfield projects typically take 6 to 12 months against 18 to 36 for greenfield, and clear height is the binding capacity constraint in older buildings | D | Evidence that practitioners distinguish the classes, and consistent ranges across vendors; not a measurement |
| University of Twente thesis, van Sijpveld | Formal brownfield design model with three scenarios — empty facility, predefined areas, existing layout — and reported annual cost reductions of about 10.9% and 2.2% against prior consultant designs | C | A master's thesis with a real case; the percentages are instance-specific |
| Umbrex facility move playbook; warehouses.solutions lease transition checklist; Regent CRE | Transition cost model as lease overlap plus exit and restoration plus move execution plus temporary storage plus disruption; a 2 to 8 week overlap is the usual balance point | D | The most operationally specific public descriptions of dual running; treat the ranges as practitioner convention |
| TriVista grocery case | Consolidation of overlapping DCs with one-time costs explicitly including facility exit, severance, inventory relocation, and upfit; reported 17.9 million dollars of annual savings | D | Unusually complete cost list for a vendor case; baseline not disclosed |
| Miebach fashion retailer case | Five Canadian DCs facing a lease expiry; a "sweat the asset" strategy avoided relocation, with projected savings and service improvement | D | The clearest published example of the low-change option winning |

### 9.5 The practitioner folklore, and what supports it

The claim you will hear in every brownfield study review is that the answer turns out to be
"change less than you thought". The evidence is indirect, and it is worth separating three
mechanisms that are often bundled. **The baseline was mispriced** — once committed escalation and
known capital are in the do-nothing case, part of the apparent saving was going to happen anyway.
**The avoidable cost was overstated** — the site's ledger includes costs that survive. **The cost
of change was omitted** — with it included, payback moves beyond the period over which anyone
believes the saving.

All three are arithmetical, not cultural. That matters, because it means the finding can be
tested rather than asserted, and because it means a study that produces a large saving in a
brownfield network should be asked which of the three mechanisms it has skipped. Grade E for
the folklore as a claim about frequency; the mechanisms are A-grade arithmetic.

## 10. Failure modes and red flags

| Failure | Red flag you can see | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Existing sites entered at zero fixed cost | Every existing site stays open in every scenario | Sunk-capacity trap: the model can never justify a closure | The study "confirms" the status quo and is worthless | Enter the avoidable annual cost, not zero |
| Existing sites entered at depreciation plus capital charge | The model wants to close sites that are cheap to run | Sunk-cost trap, inverted: past capital priced as a future cost | Capital destroyed for an accounting reason | Exclude historic capital from the fixed cost; show any capital charge separately |
| Ledger cost used as the avoidable cost | `f_j` matches the cost centre to the currency unit | Allocations and lease tails included | Saving overstated by 25 to 50% | Line-by-line avoidability test, signed off |
| Closure cost excluded | The one-off column is empty or single-line | The model is a steady-state model applied to a transition | A closure that never pays back is approved | Build the cost-of-change account before solving |
| Lease tail ignored | The closing site's rent is not in any option | The lease register was not consulted | The saving evaporates at the landlord's first letter | Register the break date and the consent requirement per site |
| Do-nothing baseline unpriced | The comparison is "option versus today's cost" | No escalation, no known capex in the baseline | Change is flattered; the "saving" includes inflation | Price the baseline first, by year, discounted |
| Notice periods treated as a decision variable | The saving starts in month one | The model has no employment-law constraint | NPV overstated by the length of the consultation | Earliest effective date per closure, derived from statute and contract |
| Path never checked for capacity | The destination is feasible and the plan is not | The closing site's volume has nowhere to go until the expansion is commissioned | A nine-month capacity trough discovered in delivery | Capacity check by date, region, and product family |
| Reconfigurability assumed universal | Every site can serve every product family at any volume | One capacity number per site, no building spec | A design the buildings cannot deliver | Capacity ceiling per site per family, from a survey |
| No churn limit | The model changes the whole network in year one | Transition cost is small against discounted saving | An unexecutable programme presented as optimal | Limit changes per period to what the programme office can run |
| Greenfield answer presented as a plan | The recommended sites do not exist and no transition is costed | The problem was misclassified at step 1 | A plan nobody can start | Classify the problem and state the classification |

Two of these deserve a paragraph.

**Zero fixed cost at existing sites** is the single most common structural error in brownfield
models, and it is invisible in the output: the model returns a sensible open set, a sensible
allocation, and a cost that reconciles to the ledger better than it should. The error is only
detectable by reading the input file. Assert it: after every build, check that no existing site has
a fixed cost of zero unless the cost is genuinely zero.

**Notice periods** are where brownfield and greenfield studies most sharply diverge, because they
are the one "soft" factor that is legally hard. A greenfield site opens when the EPC contract says;
a brownfield site closes when the consultation ends. The difference in the worked example was 0.41
years of payback and 1.0 million of NPV, and the model that omits it is not approximately right.

## 11. Assumptions and limits

| Assumption | Why the model needs it | What breaks it if it fails |
|---|---|---|
| Demand is known and flat over the horizon | The baseline must be a well-defined network | Growth makes do-nothing infeasible; decline makes closure cheaper than modelled; either changes the ranking |
| The avoidable cost is a single annual number per site | The fixed charge is a scalar | Step-fixed labour and partial retention make it a function of retained volume |
| The lease can be exited at a stated date and cost | The closure date is a parameter | A landlord refusal removes the option entirely, not merely raises its cost |
| One-off costs are dated at the start of the transition | The NPV needs a profile | Costs spread over 24 months discount to roughly 8% less, which can flip a marginal case |
| Transition costs can be estimated within a stated range | They enter the objective | A 20% error on 7.55 million is 1.5 million, comparable to the entire NPV |
| The receiving sites can absorb the volume within the stated time | Capacity is checked by date | A longer ramp creates a trough the model cannot see |
| Capacity is independent of product mix | One capacity number per site | A mix shift toward bulky or chilled product reduces usable capacity without changing the number |
| Reconfiguration is possible wherever the model permits it | The state must be offered | Building, planning, or landlord constraints make an offered state impossible |
| The discount rate is constant over the horizon | NPV requires one rate | Risk rises sharply during a transition, so a single rate understates the cost of the risky option |

**The model has no theory of the organisation.** It does not know that the site manager has twenty
years of relationships, that the union will propose an alternative, that the regional director's
bonus depends on service during the transition, or that a closure in an election year attracts
attention the company would rather avoid. These are not soft factors to be waved away: they are
constraints and costs that exist outside the model and must be handled deliberately — as a
constraint if they are absolute, as a scenario if they are uncertain, and as a paragraph in the
option narrative if they are real but unquantifiable.

**The model is a screening device, not a decision.** It prices options on stated assumptions; it
does not negotiate a lease, obtain consent, consult a workforce, or run a migration. Its value is
that it tells you which of those activities is worth the effort, and by how much — which is what
the 0.44 million rent-escape threshold in section 6.6 is.

## 12. Exercises

### Exercise 1 — Classify three studies (recall and application, 30 minutes)

You are given three briefs: (a) a retailer choosing its first three DCs in a new country;
(b) a manufacturer deciding whether to close one of six plants; (c) a distributor asking whether
to add a fourth DC to an existing three.

**Deliverable:** a one-line classification for each, with the decision verbs named.
**Acceptance criteria:** each classification is justified from the verbs, not the industry; at least
one brief is identified as hybrid; for each, one sentence on what the cost basis must contain that a
greenfield study would omit.

### Exercise 2 — Price a do-nothing baseline (calculation, 45 minutes)

A network costs 40.0 million per year today. Rent on 30% of the footprint renews in year 2 at +12%.
Wages escalate at 3.5% a year on 55% of the cost base. One site is at 96% of planning capacity and
will need 2.0 million of capital in year 3 to keep serving demand.

**Deliverable:** the baseline cost by year for five years, discounted at 9%.
**Acceptance criteria:** escalation applied to the right cost lines; the capital included in the year
it is spent; the total NPV stated; one sentence on why this baseline is not "today's cost".

### Exercise 3 — Audit the avoidable cost (application, 50 minutes)

Given a site cost centre of 4.80 million comprising site labour 2.90, rent 0.90, utilities 0.28,
maintenance 0.25, insurance 0.09, site IT 0.11, and allocated overhead 0.27.

**Deliverable:** a table classifying every line, and one avoidable annual cost reconciled from the
ledger.
**Acceptance criteria:** every line classified; the reconciliation arithmetic shown; one line
identified as step-fixed with its band; the question "if this site ceased operation on 1 January,
would this cost disappear" answered explicitly for the two largest lines.

### Exercise 4 — Build the cost-to-close account (calculation, 60 minutes)

For the same site: 140 staff at 16,500 average severance, book value 1.9 million, disposal
0.4 million, six months of dual running at 0.13 million a month, stranded overhead of 0.05
million a month for 15 months, dilapidations 0.45 million, inventory build 0.12 million,
receiving-site capital 0.95 million.

**Deliverable:** the total one-off cost, the payback against a stated net annual saving of 2.10
million, and the cumulative cash position by year.
**Acceptance criteria:** every item present and dated; payback in years to two decimals; the
write-down flagged as non-cash; a statement of what would have to be true for payback to exceed
four years.

### Exercise 5 — Encode reconfigurability (design, 50 minutes)

A site has 9 metres clear height, a 12 by 24 metre column grid, 14 docks, and a lease with
landlord consent required for alterations. It is proposed as a regional automated hub.

**Deliverable:** the constraints you would add, each as a capacity ceiling or an arc removal, with
the quantity affected.
**Acceptance criteria:** at least three constraints, each tied to a physical or legal fact; at
least one stated as a cost rather than a constraint, with the reason; a statement of what the
model would have wrongly recommended without them.

### Exercise 6 — Build the option set (design, 60 minutes)

Using the section 6 instance, add an option in which Ashfield is kept and a fifth site leased on a
five-year term at 0.75 million rent with 3.2 million of fit-out.

**Deliverable:** the new option costed on the same basis as the others, and a revised option table.
**Acceptance criteria:** NPV computed with the same discount rate and delay treatment; the option's
key condition stated; whether the ranking changes, and by how much relative to the error on the
transport estimate.

### Exercise 7 — Draft the migration path (design, 60 minutes)

For option 1 in section 6.7, build the dated sequence from announcement to steady state.

**Deliverable:** a timeline with at least eight milestones, the critical path, and a capacity check
by quarter.
**Acceptance criteria:** consultation and notice dated from the statutory minimum; the receiving
capital commissioned before the volume moves; no quarter shown below demand; the earliest date at
which the first saving is banked is stated.

### Exercise 8 — Write the page, then catch the misclassification (critique and design, 75 minutes)

Produce the one-page brownfield report for the section 6 study to the seven-item standard in
step 13. Then read a second brief: a consultancy recommends three new DCs, reports a 28% cost
reduction, and includes one line for "transition costs", for a client with four existing DCs, two
of them leased to 2031.

**Deliverable:** the one-page report, and a one-page critique of the second study.
**Acceptance criteria:** all seven items present; the excluded cost-of-change items listed with
reasons; the four critical assumptions each given a flip value; in the critique, the problem class
named, at least four specific omissions from section 5 named, the direction of the bias stated for
each, and one question that would establish whether the 28% is comparable to anything.

## 13. Self-check answers

**Exercise 1.** (a) Greenfield: the verbs are build and locate, and there is no status quo; the cost
basis must include land, EPC, and permitting, which a brownfield study treats as given.
(b) Brownfield: the verbs are close, keep, and reconfigure, and the status quo is one of the options;
the cost basis must include the avoidable cost, the lease tail, and the closure cost.
(c) Hybrid: the greenfield run establishes whether a fourth site is justified at all and where; the
brownfield run decides whether to add it to the existing three and how to transition, including the
interaction with the existing three that a greenfield study would not model.

**Exercise 2.** Non-escalating base 40.0, of which 55% (22.0) is wage-exposed and 30% of
footprint rent is 12.0 at renewal. Year 1: 40.0 × 1.019 (3.5% on 55%) = 40.77. Year 2: wages
40.77 × 1.019 = 41.55, plus rent uplift 12.0 × 0.12 = 1.44 → 42.99. Year 3: 42.99 × 1.019 =
43.81, plus capital 2.00 → 45.81. Year 4: 44.64. Year 5: 45.49. Discounted at 9%: 37.40 + 36.18
+ 35.38 + 31.62 + 29.57 = 170.15 million. The baseline is not today's cost because 12.0 million
of rent steps up, 22.0 million of wages compounds, and 2.0 million of capital is committed
whatever the network decision.

**Exercise 3.** Avoidable: labour 2.90, utilities 0.28, maintenance 0.25, insurance 0.09, site IT
0.11 = 3.63. Not avoidable: rent 0.90 (unexpired term), allocated overhead 0.27. Total 4.80,
reconciliation 4.80 − 0.90 − 0.27 = 3.63. Labour is step-fixed, so the band must be stated; for rent
the answer is "no, not until the term ends", for labour "yes, after consultation and notice".

**Exercise 4.** One-off: severance 140 × 16,500 = 2.31; write-down 1.90 − 0.40 = 1.50; dual
running 6 × 0.13 = 0.78; stranded overhead 15 × 0.05 = 0.75; dilapidations 0.45; inventory 0.12;
receiving capital 0.95. Total 6.86 million. Payback 6.86 / 2.10 = 3.27 years. Cumulative: −6.86,
−4.76, −2.66, −0.56, +1.54. To exceed four years the one-off must exceed 8.40 million, which
requires roughly 1.5 million more of severance or a receiving-capital estimate 60% higher.

**Exercise 5.** Clear height at 9 metres caps cubic storage, so the capacity ceiling for the proposed
high-bay automation is roughly a third of a 30-metre building — a capacity ceiling, not a cost. The
12 by 24 metre grid permits racking but penalises dense layouts, so it enters as a layout cost
penalty inside the handling rate. Fourteen docks cap throughput at roughly one door per 10,000
square feet of floor, which is a throughput ceiling per period. Landlord consent for alterations is a
cost with a lead time, since it can usually be obtained for a price. Without these, the model would
place automated volume in a building that cannot hold the racking and report a capacity the site
cannot ship.

**Exercise 6.** Rent 0.75 and fit-out 3.20 are the new items. Take option 1's saving structure but
with a lower transport penalty, because a fifth site closer to Ashfield's customers can absorb part
of the volume — say 1.20 rather than 1.85 — giving a net annual saving of 5.14 − 1.20 − 0.70 = 3.24,
less the new rent 0.75, so 2.49. One-offs 3.20 plus 3.30 of severance and closure (the write-down
remains) = 6.50, and the lease runs five years so no rent tail matters. NPV = 2.49 × 3.5075 − 6.50 =
8.73 − 6.50 = +2.23 million, which moves it to the top of the table. The ranking changes by about
0.7 million, larger than the error on the transport estimate and therefore material.

**Exercise 7.** Announcement day 0. Consultation opens day 0 and closes day 90; notice served within
it, effective day 150. Receiving capital approved day 0, commissioned month 10. Volume migration
months 10 to 16, dual running months 9 to 16, first full saving month 17. Critical path: capital
approval, then EPC and racking, then commissioning — 10 months, longer than the 5-month employment
clock, so capital approval binds, not consultation. The capacity check by quarter must show the
receiving sites above demand at every quarter; if the expansion lands in month 10 and the closure is
effective in month 5, there is a five-month trough and the closure must be delayed.

**Exercise 8.** A strong page states: baseline 40.0 million escalating to 45.5 by year 5 with
2.0 million of committed capital, discounted to 170.2 million; options 0 to 4 with NPVs 0.00,
+1.53, +1.11, −1.52, +1.47; included cost-of-change items with amounts; excluded items — service
failure during migration, reputational cost, any landlord negotiation fee — each with a reason;
the path with a 10-month capital critical path and a month-17 first full saving; and the four
flip values: rent escape 0.44, transport penalty 2.99, severance per head 24,600, receiving
capital 2.2 million. A weak page gives the recommended option and the NPV and omits the excluded
list. The critique should say the second study is greenfield presented as a recommendation: the
verbs are build and locate, and the four existing DCs appear only as costs to be eliminated. Four
omissions — no priced do-nothing baseline, so the 28% is measured against an undefined
counterfactual; no established avoidable cost, so the saving is against a ledger; the two leases
to 2031 absent from the transition line, so the rent tail is missing; and no notice or
consultation clocks, so the saving starts in month one — each bias the result in favour of
change. The comparability question: "what is the total discounted cost of the current network run
forward over the same horizon, on the same cost basis?"

## 14. Interview questions

**1. "What is the difference between a greenfield and a brownfield network study?"**

Strong: greenfield has no status quo and optimises a total cost; brownfield has a status quo
that is one of the options, optimises a difference against a priced baseline, and outputs a
sequence. Weak: "brownfield means existing buildings", which describes the input rather than the
problem.

**2. "What fixed cost would you put on an existing site?"**

Strong: the incremental future cost of retaining it — avoidable cash cost, less the one-off
cost avoided by not closing, plus the foregone disposal value. Not zero, not depreciation.
Weak: zero, "because it is already paid for".

**3. "Explain the sunk-cost trap and its mirror image."**

Strong: the first keeps an asset because of past spend; the second prices the asset at zero
because the spend is past. Both fail to price the future cost of retention, in opposite
directions. Weak: names only the first, and treats it as a personality flaw.

**4. "A model says close a site that was built three years ago. What do you ask?"**

Strong: what is the avoidable annual cost, what is the cost to close including the lease tail
and severance, what is the disposal value, and what does the baseline cost if we keep it. Weak:
asks about the write-down, or defends the closure on the grounds that the money is sunk, without
pricing either side.

**5. "What belongs in the cost of change?"**

Strong: reconfiguration capital, dual running, severance and notice, recruitment and training,
inventory build, stranded overhead, customer disruption, dilapidations, disposal net of
proceeds, and programme cost — each dated. Weak: "one-off costs", as a single line.

**6. "A model proposes expanding a site by 40%. What do you check first?"**

Strong: whether the building can take it — clear height, floor loading, column grid, dock count,
site boundary, planning use class, landlord consent, permit headroom — each encoded as a
constraint rather than assumed; and whether the labour agreement permits the shift pattern.
Weak: checks the capex line and the labour rate, and treats the building as infinitely elastic.

**7. "Should the model choose a destination or a path?"**

Strong: both, under different objectives — steady-state discounted cost for the destination, NPV
including transition cost for the path, with the path additionally constrained by capacity at every
date and by the capital envelope. A feasible target can have an infeasible path. Report where they
disagree. Weak: picks one and argues it is sufficient.

**8. "Why is the closure date not a decision variable?"**

Strong: because statute and contract fix a minimum elapsed time from announcement to effect —
60 days under WARN in the United States, a 30-day standstill under Directive 98/59/EC in the
European Union, plus consultation — so it is a parameter derived from law and the company's
agreements. Weak: "because HR decides it", which is an organisational answer to a legal
question.

**9. "A brownfield study concludes that nothing should change. How do you test that?"**

Strong: check the three mechanisms — was the baseline priced, was the avoidable cost correct,
was the cost of change included — and check the zero-fixed-cost error in the input file. Weak:
accepts the finding, or rejects it because brownfield studies "always" find savings.

**10. "What is the one number you would put in front of a capital committee?"**

Strong: the flip value of the binding assumption — the rent that must be escaped, the transport
penalty that must not be exceeded, or the severance per head that must not be surpassed — because
that is what the committee can act on. Weak: the NPV, on its own, without the condition it
depends on.

## 15. Key terms

- **Greenfield design** — every candidate is new; no asset, no lease, no incumbent.
- **Brownfield design** — most of the network exists; the decisions are keep, expand, reconfigure, downsize, close.
- **Hybrid study** — greenfield run for a bound and a direction, brownfield run for the decision.
- **Do-nothing baseline** — the priced, discounted counterfactual every option is measured against.
- **Sunk cost** — a cost already incurred that no future decision recovers.
- **Sunk-cost fallacy** — continuing because of prior investment rather than future return.
- **Sunk-capacity trap** — entering an owned asset at zero fixed cost, so closure can never be justified.
- **Avoidable cost** — the cost that disappears if the activity stops; not the ledger total.
- **Stranded overhead** — overhead that outlives the activity it supported.
- **Write-down** — a non-cash reduction in carrying value, recognised as a loss.
- **Closure cost** — the total one-off cost of ceasing operation at a site.
- **Transition cost** — the cost of moving from one configuration to another, dated.
- **Cost of change** — the umbrella term for reconfiguration, transition, severance, dual running, and stranded overhead.
- **Dual running** — two sites live at once; cost is rent plus split-volume inefficiency, not rent alone.
- **Severance** — payment on termination, statutory or contractual, per the applicable schedule.
- **Notice period** — the interval between notification and termination; a parameter, not a variable.
- **Consultation period** — formal consultation with employee representatives before redundancies take effect.
- **Dilapidations** — the obligation to restore leased premises at exit.
- **Break clause** — a contractual date at which a lease may end on stated notice.
- **Lease surrender** — negotiated early termination; a negotiation, not a right.
- **Landlord consent** — permission for assignment, sublet, or alteration; a condition precedent.
- **Convertible capacity** — capacity that can change role or product family without structural change.
- **Churn limit** — a constraint on sites changed per period, standing in for management capacity.
- **Migration path** — the dated sequence of network states from today to the target.
- **Option set** — three to five named designs costed against one baseline, including do-nothing.
- **Flip value** — the parameter value at which the ranking of two options reverses.
- **Payback period** — time until undiscounted net savings recover the one-off cost.
- **NPV** — net present value; discounted future cash flows, including the transition cost.

## 16. Sources

Grading: A peer-reviewed or audited; B regulatory, official, or standard-setting; C company
case, conference paper, or institutional analysis; D vendor or consultancy; E press, practitioner
observation, or anecdote.

**Peer-reviewed and foundational**

- Weber, A. (1909). *Über den Standort der Industrien*. The continuous single-facility problem,
  with no legacy state. **A**
- Hakimi, S. L. (1964). "Optimum locations of switching centers and the absolute centers and medians of a graph." *Operations Research*, 12(3). Vertex optimality behind the p-median. **A**
- Balinski, M. L. (1965). "Integer programming: methods, uses, computation." *Management Science*, 12(3), 253–313. DOI: 10.1287/mnsc.12.3.253. The fixed-charge formulation. **A**
- Wesolowsky, G. O., and Truscott, W. G. (1975). "The multiperiod location-allocation problem with relocation of facilities." *Management Science*, 22(1), 57–65. DOI: 10.1287/mnsc.22.1.57. First treatment of relocation cost. **A**
- Van Roy, T. J., and Erlenkotter, D. (1982). "A dual-based procedure for dynamic facility location." *Management Science*, 28(10), 1091–1105. DOI: 10.1287/mnsc.28.10.1091. Opening and closing in the dynamic problem. **A**
- Arkes, H. R., and Blumer, C. (1985). "The psychology of sunk cost." *Organizational Behavior and Human Decision Processes*, 35(1), 124–140. DOI: 10.1016/0749-5978(85)90049-4. **A**
- Melo, M. T., Nickel, S., and Saldanha-da-Gama, F. (2009). "Facility location and supply chain management — a review." *European Journal of Operational Research*, 196(2), 401–412. DOI: 10.1016/j.ejor.2008.05.007. **A**
- Abdoli, Fallahi, and Mokhtari (2025). "A supply chain redesign problem considering facility functionality transformation." *Transportation Research Part E*. **A**
- Martin and Altmann (2026). "An integrated multi-horizon approach for long-term reconfiguration and mid-term sequencing in global production networks." *CIRP Annals*. As indexed in the KIT repository; verify volume and pages before quoting. **A**

**Standards, regulation, and official guidance**

- Worker Adjustment and Retraining Notification Act, 29 U.S.C. §§2101–2109, in particular §2102 on notice before plant closings and mass layoffs. **B**
- Directive 98/59/EC on collective redundancies, consolidated text. Consultation and the 30-day standstill. **B**
- IAS 37 *Provisions*; IAS 36 *Impairment of Assets*; IFRS 5 *Non-current Assets Held for Sale*; IFRS 16 *Leases*. **B**
- ISO 14083:2023, greenhouse gas emissions from transport chain operations. **B**
- GS1 Global Location Number; ISO 8601; ISO 4217; ISO 80000. Identity, dates, currency, units. **B**
- National planning, building regulation, environmental permitting, and transfer-of-undertakings rules. Jurisdiction-specific; consult current text. **B**

**Company, official, and institutional**

- Woolworths Group, "F21 Significant Items and Supply Chain Update," ASX announcement, 23 June 2021. About AUD 44 million of redundancy recognised in F21 alongside a roughly AUD 400 million distribution centre. **C**
- Marks & Spencer, "M&S Clothing and Home Logistics Changes: Hardwick Distribution Centre to Close," press release, 2018. About 450 colleagues affected; DHL appointed at the new South East DC. **C**
- Marks & Spencer distribution review as reported in *Commercial Motor*, 2010. 21 of 110 warehouses closed; distribution costs down about GBP 15.9 million, or 3.9%. **C**
- MIT Center for Transportation and Logistics, "The New Competitive Edge: Analytics-Driven Supply Chain Design." Brownfield optimisation as adjusting an existing network while retaining part of the design. **C**
- van Sijpveld, University of Twente master's thesis on conceptual brownfield warehouse design. Three brownfield scenarios; annual cost reductions of about 10.9% and 2.2% against prior designs. **C**

**Vendor, consultancy, and software documentation**

- 4flow VIA help documentation: greenfield, brownfield, and hybrid strategies, with brownfield restricting candidates to existing locations. **D**
- KPI Solutions, OPSdesign, TREW Automation, and Fortna, brownfield versus greenfield warehouse design notes. Clear height, dock count, and floor loading as binding constraints; 6–12 months against 18–36 months. **D**
- Umbrex, "The Facility Move Playbook"; warehouses.solutions lease transition checklist; Regent CRE, "20 Hidden Warehouse Relocation Costs." Transition cost as overlap plus exit and restoration plus execution plus temporary storage plus disruption; a 2–8 week overlap as the usual balance point. **D**
- TriVista, grocery distribution network case study. One-time costs itemised to include facility exit, severance, inventory relocation, and upfit; reported USD 17.9 million annual savings. **D**
- Miebach, fashion retailer distribution network case study, Canada. Five DCs, lease expiry, "sweat the asset" strategy. **D**

**Press and practitioner observation**

- Practitioner observation in this file on how baseline pricing, avoidable cost, and the cost of change are typically handled in project work. A description of the norm, not a measurement. **E**

> **Note on citation discipline.** The 6–12 against 18–36 month build comparison, the 2–8 week dual-running overlap, and the cluster of vendor savings percentages are practitioner ranges repeated across sources, not findings. The Woolworths and Marks & Spencer figures are quoted as disclosed and are not adjusted for scope differences. No percentage here should be transferred to another network without rebuilding the baseline.

## 17. Links

**This module**

- Module map: [`_index.md`](_index.md)
- The decision ladder, which frames the problem class: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- Graph and cost-to-serve, the source of the transport penalty: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- MILP anatomy, assumed here: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- The formulation this topic extends: [`04_facility-location.md`](04_facility-location.md)
- Multi-commodity and multi-period models, the machinery for a migration path: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- Next, the analysis a transition plan must survive: [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md)
- Then, uncertainty in the transition itself: [`08_robust-stochastic.md`](08_robust-stochastic.md)
- Then, emissions as a transition objective: [`09_carbon-objective.md`](09_carbon-objective.md)
- Closing topic, solver behaviour for a long-horizon transition model: [`10_solvers-heuristics.md`](10_solvers-heuristics.md)

**Adjacent modules and programme navigation**

- Disruption and resilience, which owns the network's response to shocks: [`../14A_resilience_disruption_analytics/_index.md`](../14A_resilience_disruption_analytics/_index.md)
- Part II methods map: [`../_index.md`](../_index.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

> **Note:** all ten topic files in this module are now written. Every markdown link on this page
> resolves to a file that exists on disk.
