# The Network Decision Ladder — Strategic, Tactical, and Operational Network Choices and Their Time Scales

> Module 11 — Network Design and Optimization
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (inventory and service), Module 07 (demand forecasting)
> Version 2026.5

---

## 1. Why this topic exists

Every other topic in this program optimises inside a structure. This topic chooses the structure.

That difference is not a matter of scale. It is a difference in the kind of question being asked, the kind of evidence that can settle it, and the kind of person who is allowed to answer it. A planner who changes a replenishment quantity on Tuesday can change it back on Wednesday. An executive who signs a fifteen-year lease on a 40,000 square metre distribution centre (DC) has made a decision that will still be shaping the company's cost base when the analyst who recommended it has left. Treating those two decisions with the same analytical habits is the most expensive category error in supply chain analytics, and it is made routinely.

The error shows up in three recognisable forms.

**The budget meeting that closes a site.** An annual budgeting exercise ranks facilities by cost per unit dispatched. The site at the bottom of the ranking is put forward for closure. Nobody in the room has asked whether the site's cost per unit is high because it is badly run or because it serves a sparse region that no other site can reach economically. The ranking is a snapshot of an allocated cost base, not a statement about the network. The decision it produces is structural, taken with tactical evidence, in an operational meeting, on a twelve-month horizon.

**The transport tender that "proves" a DC should move.** A freight optimisation shows that rerouting volume through a different node reduces line-haul cost by 6%. The number is correct. It is also answering a question about which lanes to use this quarter, and it is being used to justify a capital decision that will take three years to implement and seven years to pay back. The 6% is real and irrelevant, because it excludes the lease exit, the severance, the inventory repositioning, and the service promise the old site was holding up.

**The five-year model used to decide tomorrow's expedite.** The opposite failure. A team builds a full multi-period, multi-commodity network model, and then asks it whether to expedite a container. The model's grain is months and millions of dollars; the decision's grain is hours and thousands. The answer arrives after the vessel has sailed.

Each of these is a failure to place a decision on a ladder. The ladder is the subject of this topic.

**Why the question is live again in 2026.** For roughly two decades, most large companies treated the physical network as fixed and spent their analytical effort on planning within it. Three changes have reopened the structural question. Cost geometry moved: freight, labour, energy, and warehouse rents repriced sharply after 2020 and have not returned to the earlier baseline, so networks optimised on 2015 ratios are being re-examined on their merits rather than by inertia. Policy moved: the European Union's carbon border adjustment mechanism (CBAM), established by Regulation (EU) 2023/956, entered its definitive regime on 1 January 2026, which converts the carbon content of covered imports from a disclosure into a landed-cost parameter. Service expectations moved: in several consumer categories, two-day delivery migrated from a premium offer to a baseline expectation, which pulls inventory and facilities closer to demand and raises the fixed-cost burden of doing so.

None of these changes makes the analytical problem easier. They make the framing problem more expensive to get wrong, because more companies are now running structural studies, and a larger share of those studies are being scoped by people who have never had to defend one to a capital committee.

**What the reader is blocked on.** The practical blockage is vocabulary. In most organisations there is no agreed way to say "this is a structural decision and it needs a different process." The words strategic, tactical, and operational are used loosely and interchangeably, usually to mean "important", "medium-important", and "urgent". A study is therefore scoped by whichever function shouts loudest, with whichever data is nearest to hand, on whichever horizon fits the fiscal year. The result is a document that answers a question nobody asked, delivered after the decision was taken.

**What the topic gives you.** A four-rung ladder, a reversibility test for placing any question on it, a reporting standard that forces the rung and the horizon to be stated, and a clear account of what a network model can and cannot settle. The rest of Module 11 assumes this ladder. Topics 02 to 05 build the model; Topics 06 to 10 extend it to legacy footprints, uncertainty, carbon, and solution methods. None of them will re-teach the framing, and every one of them will be misused if the framing is wrong.

**The honest size of the prize.** A network study is a high-leverage, low-frequency engagement. That combination is dangerous: the leverage justifies the effort, and the low frequency means the organisation has little practice at scrutinising it. The published numbers are large, and they are almost always projections or company claims rather than audited outcomes. Section 9 treats this properly. For now, hold one calibration: the documented Procter & Gamble restructuring associated with Camm and colleagues reported savings well over $250 million per year before tax, and it also involved more than $1 billion in write-offs and transition costs and affected around 6,000 people. A network decision is not a spreadsheet exercise with a big number at the bottom. It is an industrial act with a financial appendix.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** any described supply chain decision onto one of four rungs — structural, tactical, operational, execution — and **justify** the placement with the reversibility scorecard rather than with the job title of the person who owns it.
2. **State** the horizon, the capital at risk, and the decision owner for each rung, and **explain** why the four are not interchangeable.
3. **Apply** the reversibility test to a decision you have personally been involved in, and **identify** which of the five test dimensions made it hard to undo.
4. **Distinguish** a network design study from an annual budgeting exercise on six dimensions: cadence, horizon, objective, constraint set, data vintage, and decision rights.
5. **Write** an objective function for a network decision in one sentence, naming what is maximised or minimised, over what horizon, and at what discount rate.
6. **Separate** the constraints that are genuinely fixed from those that are merely inherited, and **state** what it would cost to relax each one.
7. **Explain** what a network model can answer and **name** at least four questions it cannot answer, including organisational executability and data-quality disputes.
8. **Produce** a decision memo that meets the five-element reporting standard: rung named, horizon stated, reversibility assessed, objective function written down, fixed constraints identified.

Each outcome maps to an exercise in section 12 and to a self-check item in section 13.

## 3. The core mental model

Hold one picture: a four-rung ladder. Each rung is a class of decision. The rungs are ordered by how long the consequences last and how expensive it is to climb back down. Everything above constrains everything below. Nothing below can undo what is above.

```text
        +-------------------------------------------------------------------+
  RUNG  |  STRUCTURAL        horizon  3-10 years      reversible:  no       |
   4    |  how many facilities, where, what role, how much capacity         |
        |  capital at risk: $10m - $1bn+      owner: CEO/COO + CFO + board   |
        |  sets: the site list, the lane set, the service geography         |
        +-------------------------------------------------------------------+
                                      |  constrains
                                      v
        +-------------------------------------------------------------------+
  RUNG  |  TACTICAL          horizon  6-24 months     reversible:  partly   |
   3    |  which facility serves which market, seasonal capacity, lane deals|
        |  capital at risk: $0.1m - $10m      owner: supply chain director   |
        |  sets: the assignment map, the contract set, the seasonal shape   |
        +-------------------------------------------------------------------+
                                      |  constrains
                                      v
        +-------------------------------------------------------------------+
  RUNG  |  OPERATIONAL       horizon  days-weeks      reversible:  mostly   |
   2    |  flow allocation, expediting, cross-docking, stock rebalancing    |
        |  capital at risk: ~$0 - $100k       owner: operations manager      |
        |  sets: which unit goes where today, at what premium               |
        +-------------------------------------------------------------------+
                                      |  constrains
                                      v
        +-------------------------------------------------------------------+
  RUNG  |  EXECUTION         horizon  hours           reversible:  fully    |
   1    |  which truck, which route, which pick sequence, which dock door   |
        |  capital at risk: ~ zero           owner: dispatch / planning      |
        |  handed to Module 12 — transport execution is not this module     |
        +-------------------------------------------------------------------+
```

Read the ladder from the top. The structural rung fixes the set of sites and lanes. The tactical rung assigns demand to those sites and contracts capacity on those lanes. The operational rung moves units within the assignment. The execution rung moves vehicles within the plan. Each lower rung takes the rung above as a parameter.

Five propositions follow, and they carry most of what this module has to say.

**Proposition 1 — The ladder is a statement about which variables are allowed to be parameters.** Every model has decision variables and parameters. The rung determines which is which. In a structural model, the number of facilities is a variable and the demand forecast is a parameter. In a tactical model, the number of facilities is a parameter and the assignment is a variable. In an operational model, the assignment is a parameter and the flow is a variable. Most modelling arguments that look like disagreements about method are actually disagreements about this boundary, and they cannot be resolved inside the model.

**Proposition 2 — A rung cannot solve the rung above it.** No amount of tactical sophistication fixes a structural error. If the network has one DC serving a market that needs three, the best possible assignment is still a bad service outcome. If it has three serving a market that needs one, the best possible assignment still carries three fixed-cost bases. This is the single most important sentence in the topic: **a wrong structural decision cannot be corrected by better planning downstream.** It can only be corrected structurally, at structural cost.

**Proposition 3 — Reversibility, not importance, defines the rung.** People classify decisions by how much they care about them. The ladder classifies them by what it costs to undo them. A €40 million warehouse in the wrong place is structural. A €40 million advertising campaign in the wrong quarter is not. The amount is similar; the reversibility is not. Use the reversibility scorecard in section 7, not the size of the number.

**Proposition 4 — The rungs have different clocks, and the clocks do not synchronise.** A structural study runs every three to five years and is built on data frozen at kickoff. Demand data changes monthly. Contract rates change at renewal. Service expectations change with the competitor's last announcement. The study is therefore always answering a slightly stale question, and the professional response is not to pretend otherwise but to date-stamp every input and state the refresh tolerance.

**Proposition 5 — Authority runs down the ladder; evidence runs up.** The board authorises the structural decision. The evidence for it comes from operational data — shipment records, pick counts, dwell times, freight invoices. The people who generate the evidence are four rungs below the people who make the decision, and they never meet. Every network study is an attempt to bridge that distance, and most of its failure modes are failures of that bridge rather than failures of mathematics.

### 3.1 The same asset, four different rungs

A common confusion is to think a facility belongs to a rung. It does not. A question about a facility belongs to a rung.

| Question about the same DC | Rung | Horizon | Who decides |
|---|---|---|---|
| Should we have a DC in this region at all | Structural | 3-10 years | Executive committee, board above threshold |
| Should this DC serve the northern region or the eastern one | Tactical | 6-24 months | Supply chain director |
| Should we expedite this order from the other DC today | Operational | Days | Operations manager |
| Which vehicle should carry the expedited order | Execution | Hours | Dispatch (Module 12) |

The physical asset is identical in all four rows. The decision, the evidence, and the owner are not.

### 3.2 Why the strategic/tactical/operational split exists at all

The three-level split is older than supply chain management and was not invented for it. Robert N. Anthony, at Harvard Business School, set it out in *Planning and Control Systems: A Framework for Analysis* (1965), with an earlier version in *Management Services* in 1964. Anthony's three levels were strategic planning, management control, and operational control, distinguished by time horizon, by whether the information was financial or physical, and by whether the decision was programmable. The framework reached information systems through Gorry and Scott Morton's 1971 *Sloan Management Review* paper, which gave it the triangular picture that later literature calls the Anthony triangle.

Two details are worth knowing because they explain why the supply chain version is slightly distorted. First, Anthony's middle rung was called management control, not tactical planning, and it was defined by a rhythmic calendar — budgets, reports, reviews — rather than by a fixed number of months. Supply chain authors relabelled it tactical and gave it a horizon, which is a useful simplification and a small loss of meaning. Second, Anthony's levels were about the *kind of information* appropriate to a decision, not about the size of the decision. The modern habit of using "strategic" to mean "expensive" is a drift, not the original idea.

The supply chain literature adopted the triad in the 1980s and 1990s and hardened it into a taxonomy of planning problems. The version most often cited in network design work is the horizon split reported in reviews of the field: strategic facility decisions holding for roughly three to five years, tactical decisions over roughly three months to three years, and operational decisions over hours to a trimester. The exact boundaries vary by author and by industry, which is precisely why this topic teaches a test rather than a table.

### 3.3 The inheritance principle

Each rung inherits a problem from the rung above, and the inherited problem is often larger than the decision that created it.

Module 10 owns inventory structure within a given network. That ownership is real and this module does not re-teach it. But the network decision sets the inventory problem each site inherits. Open a fourth DC and you have created four forecast-error pools where there was one, four safety-stock calculations, four sets of cycle stock, and four obsolescence exposures. The pooling benefit that Module 10 quantifies is exactly the quantity the network decision spends. A network model that ignores the inventory consequence will systematically prefer more sites than the business can afford to stock, because it sees the transport saving and not the inventory bill.

The same inheritance runs in the other direction. Module 07 owns demand forecasting, and the network model consumes its output as a parameter. If the forecast is a single number, the network is designed for a world that will not occur. If the forecast is a distribution, the network model can be run against scenarios or against a stochastic formulation, and the answer changes. The network study does not improve the forecast; it inherits it.

### 3.4 Rung confusion, and what it costs

| Symptom in the room | What is actually happening | Consequence |
|---|---|---|
| "The model says we should close it" | An operational or tactical model is being used to settle a structural question | Closure executed on incomplete cost base; service and transition costs appear later |
| "We'll optimise the network next quarter" | A structural question is being deferred to an operational cadence | Decision drifts; leases renew by default; the option expires unused |
| "Finance won't fund it because the payback is four years" | A structural decision is being judged by an annual capital-rationing rule | Good structural moves rejected; bad ones approved when the payback is massaged |
| "Commercial blocked the study" | The objective function omitted a commercial constraint that was never written down | Months of work discarded at the final gate |
| "The numbers keep changing" | Data vintage was not fixed at kickoff | Study cannot be defended; every reviewer recomputes |
| "We already knew that" | The study confirmed the incumbent design | Value is real but invisible; the next study is not funded |

The last row deserves a note. A study that confirms the current network is a successful study. It has bought the organisation the right to stop arguing and the evidence to resist an internal push for change. Organisations that only count studies which produce a change will systematically under-invest in the ones that produce certainty.

### 3.5 The five properties of the structural rung

The structural rung is not simply "the top one". Five properties distinguish it, and a decision that has all five belongs there regardless of how it is described.

**1. Capital intensity.** The commitment is large relative to the business unit's cash flow, and it is lumpy: you cannot build a third of a distribution centre. This matters analytically because lumpiness creates a step function in cost, which is why the problem is a mixed-integer programme rather than a linear one, and it matters organisationally because a lumpy commitment consumes a year's capital allocation.

**2. Commitment windows.** The obligation persists for a contractual period that is long relative to the planning cycle. A fifteen-year lease, a twenty-year building, a ten-year automation contract, and a five-year dedicated transport agreement all create windows during which the organisation is bound. The window, not the asset life, is what constrains the next decision.

**3. Lease and permit lead times.** The elapsed time from decision to operation is measured in quarters or years. A greenfield site may need planning consent, environmental review, utility connection, and a construction programme. A lease assignment may need landlord consent. These lead times are the reason a structural recommendation must be phased, and the reason a study that ignores them produces a plan with an impossible first year.

**4. Workforce and community consequences.** Closing a facility is an employment decision. It triggers consultation obligations, severance costs, retention risk among the people you need to keep during the transition, and consequences for a community that may have few alternative employers. Section 9.4 gives the evidence on how persistent those consequences are. This property is the reason a structural decision cannot be delegated to an analyst, and the reason the transition cost line in a business case is never just a number.

**5. Non-substitutability downstream.** A wrong structural decision cannot be corrected by better planning. This is the property that makes the rung a rung rather than a severity label. Every other property could, in principle, be managed by a sufficiently good operational response. This one cannot. If the site is in the wrong place, the best planner in the company will still be shipping from the wrong place.

A useful diagnostic: if a decision has properties 1 and 5 but not 2, 3, or 4 — a large, lumpy, hard-to-reverse investment with a short commitment window — it is structural in consequence and tactical in process. That combination is rare and worth naming when you see it, because the standard structural apparatus (consultation, phasing, community impact) may be unnecessary and the standard tactical apparatus (fast decision, delegated authority) may be appropriate.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Network design | The choice of which facilities exist, where, with what role and capacity, and which flows connect them | Used to mean any logistics improvement | None |
| Structural decision | A decision whose consequences persist for 3-10 years and whose reversal costs a large fraction of the original commitment | Used to mean "important" | Years; currency |
| Tactical decision | A decision that allocates demand, capacity, or contracts within a fixed facility set, over 6-24 months | Used to mean "medium-term planning" generally | Months |
| Operational decision | A decision that allocates flow or expedites within a fixed assignment, over days to weeks | Confused with execution | Days |
| Execution decision | A decision that selects a specific vehicle, route, or sequence within a plan, over hours | Treated as a synonym for operational | Hours |
| Rung | One of the four levels of the decision ladder | Used loosely as "level" without a definition | Ordinal 1-4 |
| Planning horizon | The period over which a decision's consequences are evaluated | Confused with the forecast horizon | Months or years |
| Reversibility | The cost and time required to undo a decision | Treated as a binary property of an asset | Score 0-10 |
| Reversibility scorecard | The five-dimension test in section 7 that places a decision on a rung | Replaced by intuition about importance | Score 0-10 |
| Capital at risk | The money committed that cannot be recovered if the decision is reversed | Confused with total project cost | Currency |
| Commitment window | The period during which the organisation is contractually or physically bound | Confused with payback period | Months or years |
| Sunk cost | Money already spent that cannot be recovered by any future action | Used to justify continuing a failing course | Currency |
| Option value | The value of retaining the ability to defer or change a decision later | Ignored in deterministic business cases | Currency |
| Hurdle rate | The minimum return a project must earn to be funded | Applied identically to reversible and irreversible projects | Percent per year |
| Net present value (NPV) | The sum of a project's future cash flows discounted at a stated rate | Computed without stating the discount rate | Currency |
| Discount rate / WACC | The rate used to convert future cash to present value; weighted average cost of capital | Treated as a physical constant | Percent per year |
| Payback period | Time until cumulative cash inflow equals the initial outflow | Used as the sole criterion for structural decisions | Years |
| Break-even point | The value of a parameter at which two options have equal NPV | Confused with accounting break-even | Units, currency, or years |
| Greenfield design | A design with no legacy assets, where every site is a candidate | Presented as a recommendation | None |
| Brownfield design | A design constrained by existing sites, leases, and sunk assets | Treated as inferior rather than implementable | None |
| Cost of legacy | The gap between the greenfield optimum and the best achievable brownfield design | Omitted from the business case | Currency per year |
| Candidate site | A location the model is permitted to open | Confused with an existing site | Location identifier |
| Facility role | The function a site performs: production, storage, cross-dock, postponement, returns | Treated as a single undifferentiated "DC" | Categorical |
| Distribution centre (DC) | A facility that stores and dispatches finished goods | Used for any node with a roof | Location identifier |
| Regional distribution centre (RDC) | A DC serving a defined region, typically downstream of a national DC | Confused with a local depot | Location identifier |
| Cross-dock | A facility that transfers goods between vehicles with little or no storage | Modelled as a DC with zero inventory, which misstates its cost | Location identifier |
| Forward stocking location | A small inventory point close to demand, replenished from a larger node | Confused with a DC | Location identifier |
| Throughput | Units or cubic metres handled by a facility per period | Reported without the period or the unit | Units or m³ per period |
| Capacity | The maximum throughput a facility can handle under stated operating assumptions | Treated as a single number independent of shift pattern | Units per period |
| Utilisation | Throughput divided by capacity | Reported above 100% without explanation | Percent |
| Service window | The maximum elapsed time from order to delivery promised to a customer | Expressed in kilometres instead of hours | Hours |
| Cost-to-serve (C2S) | The full cost of serving a defined customer, channel, order, or market, including freight, handling, inventory, and failure | Used as a synonym for logistics cost | Currency per order or per unit |
| Total landed cost | The delivered cost of a good including purchase price, freight, duty, fees, and inventory in transit | Reported excluding duty or carbon | Currency per unit |
| Objective function | The mathematical statement of what the model maximises or minimises | Left implicit and discovered at the review | Mathematical expression |
| Fixed constraint | A restriction the model treats as non-negotiable, such as a service floor or a lease that cannot be broken | Confused with a preference | Mathematical expression |
| Decision right | The authority to make a particular class of decision | Assumed to follow the org chart | Role name |
| RACI | Responsible, accountable, consulted, informed — a decision-rights matrix | Used as a documentation ritual rather than an allocation | Matrix |
| Network study | A periodic, structural analysis producing a design recommendation for a capital decision | Confused with the annual budget | Document |
| Budgeting cycle | The annual process that sets next year's cost and revenue plan within the existing structure | Confused with network planning | Annual |
| Sales and operations planning (S&OP) | The monthly or quarterly process that aligns demand, supply, and financial plans | Confused with network design | Monthly |
| Transition cost | The one-time cost of moving from the current network to a proposed one | Omitted from the NPV | Currency |
| Stranded cost | A cost that continues after an asset stops being useful, such as a lease with no exit | Confused with a write-off | Currency |
| Write-off | An accounting charge recognising that an asset's carrying value exceeds its recoverable amount | Treated as the same thing as a cash cost | Currency |
| Restructuring provision | An accounting liability recognised for a planned restructuring, subject to recognition criteria | Recognised before a plan exists | Currency |
| Lease term | The non-cancellable period of a property or equipment lease, plus extension options reasonably certain to be exercised | Read as the first expiry date only | Months |
| Permit lead time | The elapsed time to obtain planning, environmental, or operating permits for a site | Assumed to be zero in greenfield models | Months |
| Mixed-integer linear programme (MILP) | An optimisation model with both continuous and integer decision variables | Used to mean any optimisation model | Model class |
| Candidate design | A complete network configuration evaluated against the objective | Confused with a scenario | Configuration |
| Scenario | A coherent set of assumptions about the future used to test a design | Confused with a sensitivity | Named case |
| Sensitivity | The change in the answer when one parameter is varied | Confused with a scenario | Percent change |
| Regret | The difference between a design's cost in a scenario and the best design's cost in that scenario | Computed against the wrong baseline | Currency |
| Maturity level | A stage on the cost-to-serve progression from reporting through scenario comparison to optimisation | Used as a marketing label | Ordinal 0-4 |

Four of these deserve expansion.

**Reversibility versus importance.** Importance is a judgement about consequence. Reversibility is a judgement about escape. The ladder uses escape. A decision can be extremely important and fully reversible — a pricing move, a promotion, a routing change — and such decisions should be made quickly by the people closest to the information. A decision can be moderately important and almost irreversible — a five-year lease on a specialised cold-store — and such decisions deserve the full structural apparatus even if the money is modest. Classify by escape, then calibrate effort by consequence.

**Sunk cost versus option value.** These are often confused because both concern irreversibility, and they point in opposite directions. Sunk cost says: money already spent is irrelevant to the forward decision, so ignore it. Option value says: money not yet spent is a resource, so do not spend it until the information is worth the commitment. The first is about the past and prevents you from throwing good money after bad. The second is about the future and prevents you from closing a door you may want to walk through. A network decision requires both disciplines at once, applied to different parts of the same business case.

**Cost of legacy.** A greenfield model will usually produce a design the company cannot reach, because it ignores leases, permits, workforce agreements, and customer contracts. The gap between that design and the best achievable brownfield design is not a modelling error. It is a real quantity — the price of the company's history — and it is one of the most useful numbers a network study produces. It tells the board what the constraints are worth, which is the first step to deciding whether any of them should be bought out.

**Service window versus service radius.** A service window is expressed in hours from order to delivery. A service radius is expressed in kilometres. They are not interchangeable. A 300-kilometre radius in a dense metropolitan area might be a two-hour promise; the same radius in a rural area might be a six-hour promise with a low probability of meeting it. Models that encode service as a radius are easy to build and easy to defend in a meeting, and they are wrong in exactly the markets where service matters most. Encode the promise in the unit the customer measures it in.

## 5. The mechanics, step by step

This section is a procedure. It takes a network question from the sentence in which it is first uttered to the memo that a capital committee can act on. Thirteen steps, each with its input, its output, and the branch that most often sends the work back.

The procedure is written for a structural decision, because that is the hardest case and the one this module exists for. Tactical and operational decisions use the same steps with a shorter horizon, a smaller option space, and fewer people in the room.

### Step 1 — Write the decision as a verb

**Input:** a sentence in a meeting, usually of the form "we should look at our network".

**Output:** one sentence with an action verb, an object, and a scope boundary. For example: "Decide whether to close the Northfield DC, keep it, or replace it with a smaller cross-dock, for the period 2027-2033, serving the northern region only."

**Why it matters:** most network studies begin with a topic, not a decision. A topic cannot be scoped, cannot be costed, and cannot fail. A decision can be all three. The verb also tells you which rung you are on: open, close, relocate, and re-role are structural; assign, contract, and pre-build are tactical; allocate and expedite are operational.

**Branch:** if you cannot write the verb, you do not have a decision yet. The most common honest output of Step 1 is "we do not know what we are being asked", and the correct next action is a conversation with the person who owns the capital, not a data request.

### Step 2 — Classify the rung with the reversibility scorecard

**Input:** the decision sentence.

**Output:** a rung number and a score out of ten, with the five component scores written down.

**Why it matters:** the rung determines the process, the evidence standard, the owner, and the review cadence. Getting it wrong in either direction is expensive. Classifying a structural decision as tactical produces an under-scoped study that will be rejected. Classifying a tactical decision as structural produces a six-month study for a question that needed an answer in a week.

The scorecard has five dimensions, each scored 0, 1, or 2. Section 7 gives the worked scoring.

1. **Time to undo.** Under one month scores 0; one to twelve months scores 1; more than twelve months scores 2.
2. **Cost to undo**, as a fraction of the original commitment. Under 5% scores 0; 5-25% scores 1; above 25% scores 2.
3. **Third-party consent.** None required scores 0; a commercial counterparty scores 1; a regulator, works council, planning authority, or community scores 2.
4. **Asset redeployability.** Standard and liquid scores 0; movable but specialised scores 1; site-specific and immobile scores 2.
5. **External commitment created.** None scores 0; a customer promise scores 1; a public, regulatory, or community commitment scores 2.

Totals of 0-2 map to execution, 3-4 to operational, 5-6 to tactical, and 7-10 to structural.

**Branch:** if the total falls within one point of a boundary, score the decision at the higher rung. The cost of over-preparing a tactical decision is a few weeks. The cost of under-preparing a structural one is a decade.

### Step 3 — Fix the horizon and the discount rate

**Input:** the decision sentence and the finance department's current cost of capital.

**Output:** an evaluation horizon in years, a discount rate as a percentage per year, and a statement of what happens at the end of the horizon.

**Why it matters:** a structural decision is evaluated over the life of its longest-lived asset, not over the budgeting year and not over the payback period. For a leased DC, the horizon is typically the lease term plus one renewal. For a owned plant, it is the remaining economic life. The discount rate should be the company's weighted average cost of capital (WACC), adjusted if the business unit's risk differs materially.

Two errors recur. The first is using the capital-rationing hurdle rate — often 15-20% — as the discount rate for a strategic decision. A hurdle rate is a device for allocating scarce capital across competing proposals within a year. Using it to discount a fifteen-year asset biases the analysis toward the short term and toward decisions that defer spending, which is precisely the bias that leaves companies with an ageing network. The second is using a single discount rate for cash flows with very different risk. The transport savings in a closure case are relatively certain; the revenue effect of a service degradation is not. Discounting both at the same rate is a simplification that should be stated.

**Branch:** if the organisation refuses to state a discount rate, compute the answer at three rates — for example 6%, 9%, and 12% — and report the range. A recommendation that survives all three is more robust than one that depends on a single number nobody will defend.

### Step 4 — Write the objective function before the model

**Input:** the decision sentence, the horizon, and the discount rate.

**Output:** one sentence and one expression. For example: "Minimise the NPV over 2027-2033 of total network cost — fixed, variable, transport, inventory, service failure, carbon, and transition — subject to a 98% next-day service floor in the top 40 markets."

**Why it matters:** the objective function is where the organisation's disagreements actually live, and writing it down before modelling forces those disagreements into the open while there is still time to resolve them. Three functions typically want three different objectives, and the differences are not cosmetic.

| Function | What it optimises | What it treats as a constraint | What it is blind to |
|---|---|---|---|
| Finance | NPV, return on invested capital, capital at risk | Service levels as given | The operational feasibility of the design |
| Operations | Cost per unit, labour productivity, throughput stability | Capital budget as given | Revenue effects of service change |
| Commercial | Revenue, availability, market coverage, customer retention | Cost baseline as given | The fixed-cost consequences of breadth |

A network study that writes only the finance objective will produce a design that operations cannot run and commercial will not sell. A study that writes only the commercial objective will produce a network nobody can afford. The professional practice is to write one objective, put the other two functions' concerns into the constraint set explicitly, and then price the constraints — that is, report what each constraint costs.

**Branch:** if the three functions cannot agree on a single objective, do not proceed to modelling. Run the study as a set of named designs under a shared cost base instead, and let the disagreement be visible in the comparison. This is slower and more honest.

### Step 5 — Separate fixed constraints from decision variables

**Input:** the objective function and the current network.

**Output:** three lists. One: what the model may change. Two: what the model must treat as fixed, with a reason and an owner. Three: what is fixed only because nobody has asked, with the cost of relaxing it estimated.

**Why it matters:** the third list is where most of the value in a network study hides. Constraints that are genuinely fixed include a lease with four years to run, a permit that cannot be transferred, a customer contract with a named delivery point, and a regulatory obligation to serve a region. Constraints that are merely inherited include "we have always run three DCs", "the ERP cannot support a fourth", and "the regional director would not accept it". The first set belongs in the model. The second set belongs in a conversation.

A useful discipline is to label every fixed constraint with one of three tags: **legal**, **contractual**, or **chosen**. Chosen constraints are decisions the organisation has made and could unmake. Reporting them as if they were physical facts is how a network study ends up ratifying the status quo while appearing rigorous.

**Branch:** if more than about a third of the constraints are tagged chosen, the study is not really a network design study. It is a validation of the existing design, which may be a perfectly good thing to do, but it should be labelled as such.

### Step 6 — Establish and reconcile the baseline

**Input:** last year's actual flows, costs, and service performance.

**Output:** a model configuration that reproduces the current network's cost within a stated tolerance, typically 2%, with every difference explained.

**Why it matters:** the baseline is the study's credibility. If the model cannot reproduce what the company spent last year, no reviewer will believe its projection for 2033, and they will be right. Reconciliation is also the most reliable way to find data defects, because every mismatch has a cause: a rate that has not been updated since the contract renewal, a lane that is billed but not modelled, an allocated overhead that the model treats as variable.

The reconciliation should be done at the level the finance function recognises: cost centre, general ledger account, and period. A model that reconciles only in total can be wrong in offsetting ways.

**Branch:** if reconciliation cannot be closed within tolerance, document each residual with its cause and its size, and state the effect on the recommendation. Do not adjust the model to fit. An unexplained plug in the baseline becomes an unexplained error in the answer.

### Step 7 — Assemble demand inputs from Module 07 and date-stamp them

**Input:** the demand forecast by market, product family, and period.

**Output:** a demand table with a stated grain, a stated unit, a stated vintage, and a stated uncertainty representation.

**Why it matters:** the network model consumes Module 07's output as a parameter, and it inherits every defect in it. Three defects matter most. A point forecast forces a deterministic design, which is fragile to exactly the growth assumptions that make a new site worthwhile. A forecast at the wrong grain — national rather than by market — cannot support a facility location decision, because the whole question is where the demand is. A forecast that is censored by past stock-outs understates demand in the markets that were poorly served, which biases the design toward serving them poorly again.

Date-stamping is not bureaucracy. It is the only defence against the frequency mismatch described in section 5.2.

**Branch:** if the forecast has no uncertainty representation, construct three scenarios by hand — a base, a downside at roughly the historical forecast error, and an upside — and run all three. A three-point range is crude and vastly better than a single number.

### Step 8 — Carry the inventory term from Module 10

**Input:** the inventory policy that each candidate site would inherit.

**Output:** an inventory cost per open facility, derived rather than assumed, with its pooling assumption stated.

**Why it matters:** this is the step most often skipped and it systematically biases the answer. A network model that counts only fixed and transport cost will prefer fewer, larger facilities, because consolidation reduces fixed cost and, usually, transport cost. What it cannot see is that consolidation also changes the inventory the network must hold, in both directions: fewer sites means less safety stock through pooling, but longer replenishment lead times to the remaining sites can mean more. Module 10 owns the mechanics of that trade. Here, the requirement is only that the network objective carries the result.

A practical approach is to compute, for each candidate configuration, the safety stock each site would hold under the policy Module 10 specifies, multiply by the carrying rate, and add the total to the objective. Where that is too expensive, add a per-site inventory term calibrated on the current network, and state that it is a calibration rather than a derivation.

**Branch:** if the inventory term cannot be derived at all, run the study with and without a rough per-site term and report both designs. The difference between them is an estimate of the error you would otherwise be hiding.

### Step 9 — Build the cost base, including transition

**Input:** rate cards, freight audit, payroll, lease register, utility bills, and the transition plan.

**Output:** a cost table with seven lines: fixed, variable, inbound transport, outbound transport, inventory carrying, service failure, and one-time transition.

**Why it matters:** the first six lines determine the steady-state comparison. The seventh determines whether the decision is worth taking. Transition costs are systematically underestimated because they are owned by people who are not in the study team: severance by human resources, lease exit by legal, dual running by operations, IT reconfiguration by technology, make-good by facilities. Each is real, each is cash, and each belongs in the NPV in the year it occurs.

A second discipline: separate avoidable from allocated fixed cost. A site's cost centre usually includes an allocation of corporate overhead that does not disappear when the site closes. Including it overstates the saving. Excluding it entirely understates the cost of the corporate functions that remain. The correct treatment is to model the allocation rule explicitly and state whether the corporate function shrinks.

**Branch:** if the transition cost cannot be estimated within a factor of two, the decision is not ready. Run the study to identify what transition information is missing, and go and get it.

### Step 10 — Generate candidate designs, then optimise

**Input:** the cost base, the demand scenarios, the constraint set, and a candidate site list.

**Output:** a small number of complete designs with full cost decompositions, and, if an optimisation is used, an optimal design with a reported gap.

**Why it matters:** optimisation answers "which design is best within this formulation". It does not answer "which designs are worth considering". In practice the most useful output of a network study is a table of five or six named designs — the incumbent, a consolidation, a deconsolidation, a re-roled site, a greenfield optimum, and a phased hybrid — each costed on the same basis. The optimum is one row in that table, and it is often not the row that gets implemented.

Topic 03 builds the mixed-integer linear programme (MILP). Topic 04 gives the baseline facility-location formulation. For this topic, the discipline that matters is that every design in the table is costed on an identical basis, with identical assumptions, and with the same transition-cost treatment.

**Branch:** if the optimisation's answer is not one of the candidate designs and is materially better, find out why before believing it. The usual reason is that the optimisation has found a constraint the candidate list accidentally imposed, such as "every existing site must stay open at its current role".

### Step 11 — Test sensitivity, scenario, and break-even

**Input:** the design table.

**Output:** a ranked list of parameters by their effect on the answer, a design-by-scenario cost matrix, and a break-even value for the two leading designs.

**Why it matters:** a network recommendation is an argument about the future. The sensitivity analysis is where the argument becomes inspectable. The most useful single output is the break-even: "consolidation is better than the incumbent if outbound freight rates rise above $0.41 per unit-kilometre, or if northern volume falls below 310,000 units per year." That sentence converts a disagreement about the future into a disagreement about a number, which is a much better argument to have.

Topic 07 owns the full treatment. Here the requirement is that the break-even is computed and reported, not that the analysis is elaborate.

**Branch:** if the recommendation flips inside the plausible range of a parameter, the recommendation is conditional, and the memo must say so in its first paragraph rather than its appendix.

### Step 12 — Test executability independently of the model

**Input:** the recommended design.

**Output:** a list of execution dependencies, each with an owner, a lead time, and a failure consequence.

**Why it matters:** a model can tell you that a site should close. It cannot tell you whether the organisation can close it. Leases have terms, works councils have consultation rights, planning authorities have timetables, customers have contracts with named delivery points, and communities have expectations. Each of these is a gate, and a gate that opens in eighteen months turns a 2027 recommendation into a 2028 one.

The test is a structured conversation, not a calculation. For each element of the design, ask: who has to agree, what do they need to see, how long do they take, and what happens if they say no. Record the answers. They belong in the memo.

**Branch:** if any dependency has a lead time longer than the decision horizon's first phase, the design must be phased. A recommendation that cannot be executed inside the study's own planning window is not a recommendation.

### Step 13 — Write the memo to the reporting standard, and set a review trigger

**Input:** everything above.

**Output:** a decision memo meeting the five-element standard in section 5.4, plus a named review trigger.

**Why it matters:** the memo is the artefact that survives. The model will be archived, the analyst will move on, and the assumptions will be forgotten. What remains is a document that says what was decided, on what basis, and what would change the answer. Without the last of these, the decision becomes permanent by default, which is exactly the failure the next study will have to correct.

**Branch:** if the memo cannot name a review trigger, the study is not finished. "Review in three years" is a trigger only if it is attached to a calendar owner. "Review if northern volume falls below 310,000 units or if the Northfield lease is renegotiated" is a trigger.

### 5.1 The five-element reporting standard

Every network recommendation should be readable as five statements, in this order. If any is missing, the recommendation is not decision-grade.

| Element | The question it answers | Weak form | Strong form |
|---|---|---|---|
| Rung named | What kind of decision is this | "A strategic review" | "Structural: site count and role, six-year horizon" |
| Horizon stated | Over what period is it evaluated | "Long term" | "2027-2033, evaluated at a 9% discount rate, with residual value at 2033" |
| Reversibility assessed | What does it cost to undo | "Difficult to reverse" | "Score 9 of 10; fifteen-year lease, planning consent required, works council consultation" |
| Objective function written down | What is being optimised | "Lowest cost" | "Minimise NPV of fixed, variable, transport, inventory, service-failure, and carbon cost, subject to a 98% next-day floor in the top 40 markets" |
| Fixed constraints identified | What was assumed immovable | "Current leases" | "Three leases with terms to 2029, 2031, and 2034; two sites with transferable permits; one customer contract naming a delivery point" |

The standard is deliberately mechanical. Its purpose is to make the two most consequential assumptions — the objective and the constraint set — impossible to leave implicit. In review, the first and fourth statements generate most of the useful argument.

### 5.2 The network study and the annual budget are different instruments

The confusion between a network study and a budgeting exercise is the most consequential in this topic, because both produce numbers in the same currency and both are reviewed by the same people.

| Dimension | Network design study | Annual budgeting exercise |
|---|---|---|
| Cadence | Every 3-5 years, or on a trigger | Annual, fixed calendar |
| Horizon | 3-10 years | One year, occasionally three |
| Question | What should the structure be | What will we spend within the structure |
| Constraint set | Explicitly examined; relaxation costed | Inherited and treated as fixed |
| Objective | NPV of total network cost, often with a service floor | Annual cost and revenue against target |
| Data vintage | Frozen at kickoff, date-stamped | Prior-year actuals plus current forecast |
| Decision right | Executive committee, board above threshold | Functional budget owners |
| Failure mode | Answers a stale question | Cannot see structural options at all |
| Output | A design and a transition plan | A set of cost centre limits |

The frequency mismatch is structural, not a process defect. A study every three to five years is built on data that changes monthly, and the study takes three to nine months to complete. By the time it reaches the committee, the demand forecast has been revised twice, a competitor has changed its service promise, and one freight contract has renewed at a different rate. This is normal. The question is what to do about it.

Three practices help. First, **freeze and date-stamp**: every input table carries the date it was extracted and the period it describes, and the memo states the vintage. Second, **refresh the finalists**: before the recommendation is signed, re-run only the two or three leading designs on the latest demand and rate data, and report whether the ranking changed. Third, **state a tolerance**: for example, "the recommendation is stable to a 10% change in any single input; it reverses if northern volume falls below 310,000 units." A study that names its own reversal condition cannot be invalidated by the next month's data, because it already told you what would invalidate it.

> **Note:** the frequency mismatch also runs the other way. Organisations that model the network continuously often find that the decision cadence has not sped up to match. A quarterly model refresh is useful for tactical questions and does not make the structural decision more frequent, because the binding constraint on structural change is leases, capital, and organisational absorption, not analytical turnaround. Do not confuse a faster model with a faster decision.

### 5.3 Why the study is periodic rather than continuous

Four reasons, in ascending order of importance.

**Data assembly cost.** A defensible network model needs demand by market, rates by lane, fixed cost by site, capacity by site, inventory policy by node, and a reconciled baseline. Assembling that once is expensive. Maintaining it is more expensive, and the maintenance is invisible work that no function owns.

**The decision itself is periodic.** Leases expire on dates. Capital cycles run annually. Strategy reviews run on a multi-year rhythm. There is no operational benefit to re-solving a structural model monthly when the structural decision cannot be taken monthly.

**Organisational absorption.** A structural change consumes management attention, employee goodwill, and customer patience. An organisation that restructures its network every year cannot execute any of the changes properly. The cadence of study is partly a protection against the cadence of enthusiasm.

**The study is a political object.** A network study does not only produce an answer; it produces a mandate. It gathers the evidence in one place, gives the answer an owner, and creates a document that can be cited. That function requires a beginning and an end. A permanently open model never acquires the authority of a finished argument.

The current direction of travel is toward lighter, more frequent modelling — a maintained model refreshed quarterly, with the full structural study still periodic. That is a sensible division: continuous for tactical questions, periodic for structural ones. It is also a commercial claim by software vendors, and should be read as one. Section 9 grades the evidence.

### 5.4 Decision rights: who owns which rung

| Rung | Accountable owner | Consulted | Informed | Typical capital threshold for escalation |
|---|---|---|---|---|
| Structural | Chief executive or chief operating officer, jointly with the chief financial officer | Commercial, operations, human resources, legal, sustainability | Board, investors, works council | Board approval above a stated amount, often $25m-$100m |
| Tactical | Supply chain or network director | Commercial for service, finance for cost | Operations, site leadership | Functional budget |
| Operational | Operations manager | Planning, customer service | Site leadership | Delegated limit, often $10k-$100k |
| Execution | Dispatch supervisor | None in the decision | Planning | None |

Three observations about this table.

First, the structural row has two accountable names on purpose. A network decision that belongs only to operations will under-weight capital and service economics. One that belongs only to finance will under-weight executability. The joint ownership is uncomfortable and correct.

Second, the thresholds are the real decision-rights document. An organisation that says the board approves structural decisions but has a $25 million threshold, while a single lease commitment runs to $40 million, has in practice delegated structural decisions to whoever signs leases. Check the threshold against the size of a typical commitment, not against the size of a typical project.

Third, the consulted column is where studies die. Commercial is consulted on every structural decision and accountable for none of them, which means it can block without owning the consequence. The fix is not to remove commercial from the room; it is to give commercial an explicit, written constraint to defend — for example, "a 98% next-day service floor in the top 40 markets" — so that its objection becomes a costed constraint rather than a veto.

### 5.5 What the model can answer, and what it cannot

The list of things a network model can answer is long and the list of things it cannot answer is short and decisive.

**It can answer:**

- For a given demand, cost, and constraint set, which combination of facilities and flows minimises the stated objective.
- What a service promise costs: the difference between a 24-hour, 48-hour, and 72-hour floor, in currency and in site count.
- What the current footprint's constraints are worth: the gap between the greenfield optimum and the best achievable brownfield design.
- At what demand, rate, or cost the preferred design stops being preferred.
- How the ranking of designs changes across named futures, and which design has the lowest worst-case regret.
- Where the network is most exposed to a single node, a single lane, or a single supplier.
- What a carbon price does to the design at a stated emissions factor and a stated price.

**It cannot answer:**

- Whether the organisation can execute a closure. That is a question about consultation, notice periods, union agreements, local politics, and management credibility. It is answered by talking to the people who would have to do it.
- Whether the data are right. A dispute about whether a lane rate is $0.38 or $0.42 per unit is resolved by reading the contract and the freight audit, not by running the model at both values. The model can show you how much the dispute matters, which is useful, and it cannot settle it.
- Whether a lease can be exited, and at what price. That is a negotiation.
- Whether a customer will accept a longer lead time. That is a commercial conversation, and the answer is often no even when the model prices the risk correctly.
- Whether a new site can be permitted, and how long it takes.
- Whether the demand forecast is right. Module 07 owns that, and the network study inherits the answer.
- What the inventory policy at each node should be. Module 10 owns that; the network study consumes it.
- How the trucks will actually run. Module 12 owns that.
- Whether the management team will hold the decision for the three years required to realise it. This is the largest uncertainty in most business cases and it appears in no model.

> **Note:** the boundary between "the model can answer" and "the model cannot answer" is not a boundary between quantitative and qualitative work. Several items in the first list are highly quantitative and belong in the model. Several in the second list can be quantified — the cost of a lease exit is a number — but the number comes from a negotiation, not from the model. The distinction is the source of the number, not its type.

### 5.6 The maturity ladder

Before an organisation can run a structural study, it needs a cost base it believes. The maturity ladder describes how far it has got. The levels are sequential in practice, because each one requires the data discipline of the one below.

| Level | Name | What exists | What it can answer | Typical artefact |
|---:|---|---|---|---|
| 0 | Cost centre accounting | Cost reported by department and general ledger account | What did we spend | The annual cost report |
| 1 | Cost-to-serve reporting | Freight, handling, and inventory allocated to customer, channel, or order | Which customers and orders are expensive to serve | The cost-to-serve report; the whale curve |
| 2 | Scenario comparison | Named designs costed on a common basis against a reconciled baseline | Which of these three designs is better | The design comparison table |
| 3 | Optimisation | A model that searches the design space within a stated objective and constraint set | Which design is best, and how sensitive is it | The MILP, the optimal design, the break-even chart |
| 4 | Maintained model | A model refreshed on a stated cadence with a governance process that acts on it | What has changed, and should we act | The quarterly network review |

Three observations.

**Most organisations are at level 1 or 2 and believe they are at 3.** The test is simple: does the organisation compare designs that a person thought of, or does it search a defined space? If a workshop produced the shortlist, it is level 2, and that is a perfectly good place to be. Level 2 done with a reconciled baseline beats level 3 done on numbers nobody trusts.

**Level 1 is not a stepping stone that can be skipped.** A cost-to-serve model is where the allocation questions get settled: how freight is attributed to an order, how a shared warehouse cost is split, how a return is costed. An optimisation built without those answers will optimise a cost base that does not reconcile, and the first finance reviewer will find it.

**Level 4 is not automatically better.** A maintained model requires an owner, a refresh budget, and a governance forum that will act. Organisations that build one without those three get a dashboard that is consulted annually and a modelling team that is absorbed into reporting. The cadence of the model must match the cadence of the decisions it serves, or it becomes expensive theatre.

Cost-to-serve itself is a management accounting import rather than a supply chain invention. It descends from activity-based costing, set out by Cooper and Kaplan in *Harvard Business Review* in 1988 as a correction to cost systems that distorted product and customer profitability. Activity-based costing went through a well-documented boom and partial retreat, analysed by Jones and Dugdale in *Accounting, Organizations and Society* in 2002, largely because the data collection burden exceeded what most organisations would sustain. Cost-to-serve inherits both the insight and the burden: the insight is that averaging hides the customers who cost more to serve than they contribute, and the burden is that the allocation logic has to be maintained by someone. Time-driven activity-based costing, published by Kaplan and Anderson in 2004, was an attempt to reduce the burden by estimating time equations rather than surveying activity. Every cost-to-serve model built today is choosing, implicitly or explicitly, where on that trade-off to sit.

## 6. Worked example

Meridian Industrial Supply is a distributor of industrial components with six DCs in a single country. The question on the table is whether to close the Northfield DC and serve its markets from the Central DC, 340 kilometres away. All figures are illustrative but internally consistent, and every one is stated in the unit the decision uses.

### 6.1 The question and the rung

The decision sentence, written to the Step 1 standard: "Decide whether to close the Northfield DC in 2027 and serve its markets from Central, or to retain it, evaluated over 2027-2032 at a 9% discount rate."

Applied to the reversibility scorecard:

| Dimension | Assessment | Score |
|---|---|---:|
| Time to undo | Reopening a closed, released DC takes more than a year | 2 |
| Cost to undo | Re-leasing, re-hiring, and re-fitting would exceed 25% of the original commitment | 2 |
| Third-party consent | Landlord, works council, and a customer contract naming the site | 2 |
| Asset redeployability | Leasehold improvements, immobile, specialised racking | 2 |
| External commitment | A next-day service promise to 38% of the site's volume | 2 |
| **Total** | | **10** |

Score 10 of 10: structural. The study therefore runs to the structural process — full cost base, NPV over six years, transition costs, service consequence, and a commercial consultation — even though the annual saving turns out to be under $1 million. The scorecard, not the size of the number, set the process.

### 6.2 The site and the numbers

Northfield handles 420,000 units per year. Its markets are within a 200-kilometre radius. Central is 340 kilometres from the centre of that radius and currently runs at 71% utilisation, so it has room.

| Item | Northfield (stay) | Central (after closure) | Basis |
|---|---:|---:|---|
| Volume, units per year | 420,000 | 420,000 | Current actual, stable for three years |
| Revenue through the site, $m per year | 58.0 | 58.0 | Used only to size the service consequence |
| Avoidable fixed cost, $m per year | 1.700 | 0.350 | Lease, labour, utilities; excludes $0.700m of allocated corporate overhead that does not disappear |
| Variable handling, $ per unit | 1.85 | 1.70 | Pick, pack, and inbound handling |
| Outbound transport, $ per unit | 3.10 | 5.10 | Rate card plus 340 km of line-haul |
| Inventory held, $m | 6.20 | 2.90 | Module 10 pooling effect: fewer stocking points, less total safety stock |
| Carrying rate, % per year | 18 | 18 | Capital, storage, insurance, obsolescence |
| Service consequence | Next-day to 38% of volume | Two-day to that volume | Modelled as 1.8% of the revenue at risk |

Two entries in this table are traps and are worth naming.

**The allocated overhead.** The Northfield cost centre reports $2.400 million of fixed cost. Of that, $0.700 million is an allocation of corporate finance, information technology, and executive cost that will be reallocated to the remaining sites, not eliminated. The avoidable figure is $1.700 million. A study that uses the reported $2.400 million will overstate the saving by 41% before it has done any arithmetic. This is the single most common source of overstated closure benefits.

**The inventory effect.** Closing a site does not reduce inventory to zero. Central must carry more to cover the wider geography. The correct comparison is total network inventory before and after, computed under the policy Module 10 specifies, and it is $6.20 million against $2.90 million — a genuine pooling gain, not a bookkeeping entry.

### 6.3 Annual cost comparison

| Cost line | Stay, $m per year | Close, $m per year | Difference, $m per year |
|---|---:|---:|---:|
| Avoidable fixed cost | 1.700 | 0.350 | (1.350) |
| Variable handling | 0.777 | 0.714 | (0.063) |
| Outbound transport | 1.302 | 2.142 | +0.840 |
| Inventory carrying | 1.116 | 0.522 | (0.594) |
| Service failure | 0.000 | 0.400 | +0.400 |
| **Total** | **4.895** | **4.128** | **(0.767)** |

The closure saves $0.767 million per year. Note the shape of the number: the fixed-cost saving of $1.350 million is largely consumed by $0.840 million of additional outbound transport and $0.400 million of service consequence. Anyone quoting the $1.350 million as "the saving" is quoting one line of a five-line table.

Arithmetic, for reproducibility:

```text
variable  = 420,000 x 1.85 / 1,000,000 = 0.777
outbound  = 420,000 x 3.10 / 1,000,000 = 1.302
inventory = 6.20 x 0.18               = 1.116
stay      = 1.700 + 0.777 + 1.302 + 1.116 = 4.895

variable  = 420,000 x 1.70 / 1,000,000 = 0.714
outbound  = 420,000 x 5.10 / 1,000,000 = 2.142
inventory = 2.90 x 0.18               = 0.522
close     = 0.350 + 0.714 + 2.142 + 0.522 + 0.400 = 4.128
```

### 6.4 One-time transition costs

| Item | $m | Owner | Timing |
|---|---:|---|---|
| Lease exit: 14 months remaining at $91,500 per month | 1.28 | Legal | Year 0 |
| Severance: 140 people at $18,000 average package | 2.52 | Human resources | Year 0 |
| Make-good and reinstatement of the leased building | 0.35 | Facilities | Year 0-1 |
| Dual running: three months of overlapping fixed cost | 0.30 | Operations | Year 0 |
| IT reconfiguration and retraining | 0.15 | Technology | Year 0-1 |
| **Total** | **4.60** | | |

None of these appear in the annual cost comparison, and all of them are cash. The lease exit is the largest single item and the one most often omitted, because the person who knows the lease terms is not in the study team.

### 6.5 Two demand scenarios and the NPV

Two scenarios, both drawn from Module 07's forecast distribution rather than invented here.

- **Base:** 420,000 units per year, flat, consistent with the last three years.
- **Growth:** 500,000 units per year by year two, reflecting two contract wins that are probable but not signed.

Recomputing the annual difference under growth — the transport penalty scales with volume while the fixed saving does not:

| Scenario | Stay, $m | Close, $m | Annual saving, $m | Break-even year on $4.60m | NPV over 6 years at 9%, $m |
|---|---:|---:|---:|---:|---:|
| Base, 420,000 units | 4.895 | 4.128 | 0.767 | 6.0 | (1.16) |
| Growth, 500,000 units | 5.291 | 4.672 | 0.619 | 7.4 | (1.82) |

The closure does not pay back inside a six-year horizon in either scenario. Under growth it is worse, because the transport penalty is proportional to volume and the fixed saving is not. This is the opposite of the intuition most people bring to a consolidation: consolidation looks better when volume falls, not when it rises, because it trades a fixed cost for a variable one.

### 6.6 The lease date changes the answer

The Northfield lease has 14 months to run. If the company waits for natural expiry, the $1.28 million exit cost disappears and the one-time cost falls to $3.32 million.

| Option | One-time cost, $m | Annual saving, $m | NPV over 6 years at 9%, $m |
|---|---:|---:|---:|
| Close now | 4.60 | 0.767 | (1.16) |
| Close at lease expiry, 14 months out | 3.32 | 0.767 | +0.12 |

The recommendation flips sign on a contract date. The value of the option to wait is not a modelling subtlety; it is the entire decision. This is what option value means in practice: the difference between committing now and committing when a constraint expires on its own.

The honest recommendation is therefore: do not close Northfield in 2027; prepare the closure for the lease expiry in early 2028, subject to two conditions — that the service degradation is acceptable to the 38% of volume affected, and that the Central site's utilisation can absorb the volume without a capacity investment. Both conditions are outside the model.

### 6.7 What the model could not tell us

Four questions decided the recommendation and none of them was answered by the arithmetic.

1. **Will the works council agree?** The site has 140 employees and a consultation requirement of at least six months. The model treated severance as a cost. It could not treat the negotiation as a risk.
2. **Will the 38% of customers accept two-day delivery?** The model priced the risk at 1.8% of revenue. That figure came from a churn assumption supplied by commercial, not from data. If the true figure is 4%, the annual saving falls to $0.29 million and the closure is clearly negative even at lease expiry.
3. **Can Central absorb the volume without a fourth shift?** At 71% utilisation it can absorb 420,000 units, but the peak month is 1.4 times the average, and the model used annual averages. The peak is the constraint.
4. **Is the $5.10 outbound rate real?** It came from the rate card for the 340-kilometre lane at the current volume band. At the higher volume, a volume discount applies, and the rate should be lower. The model used the current band, which is conservative against closure.

Each of these is a gate, and each gate is held by a person. The study's job was to make the arithmetic honest so that those people could decide with a number in front of them.

## 7. Code, query, or calculation

Two hand calculations, no libraries. The first is the closure NPV restated as a reusable template. The second is the reversibility scorecard applied to three decisions of different sizes, to show that the score does not track the money.

### 7.1 Break-even year and NPV for two scenarios

The template takes an annual saving, a one-time cost, a discount rate, and a horizon, and returns the payback year, the NPV, and the annual saving required to break even inside the horizon. The numbers are the Meridian case from section 6.

```text
# All values in $ million unless stated. Rate and horizon as stated.
saving_base      = 0.767     # 420,000 units per year
saving_growth    = 0.619     # 500,000 units per year
capex_now        = 4.60      # includes 1.28 lease exit
capex_expiry     = 3.32      # lease allowed to expire
r                = 0.09      # discount rate, per year
n                = 6         # horizon, years

af = (1 - (1 + r)^(-n)) / r          # annuity factor

payback_base   = capex_now / saving_base
payback_growth = capex_now / saving_growth
npv_base       = -capex_now + saving_base   * af
npv_growth     = -capex_now + saving_growth * af
npv_expiry     = -capex_expiry + saving_base * af
breakeven      = capex_now / af       # saving needed for NPV = 0
```

Expected output: `af = 4.486`; `payback_base = 6.00` years; `payback_growth = 7.43` years; `npv_base = -1.16`; `npv_growth = -1.82`; `npv_expiry = +0.12`; `breakeven = 1.03` $ million per year.

Three readings of this output, in order of importance.

**The break-even saving exceeds the actual saving in both scenarios.** The closure would need to save $1.03 million per year to justify its one-time cost inside six years. It saves $0.77 million at base volume and $0.62 million at growth volume. The gap is not a rounding difference; it is the decision.

**The payback year is outside the horizon in both scenarios.** Payback is a blunt instrument and it is the instrument most capital committees actually use. Reporting it, and reporting that it fails, is more persuasive than reporting an NPV they may not trust.

**Waiting dominates committing, and only because of a contract date.** The $1.28 million lease exit is 28% of the one-time cost. Removing it moves the NPV from -1.16 to +0.12. No improvement in the demand forecast, the transport rate, or the service assumption comes close to that effect.

> **Note:** the annuity factor assumes the annual saving is constant and arrives at year end. Both assumptions are false in detail — savings ramp as volume transfers, and cash arrives monthly. For a decision this marginal, the timing detail matters. The template is adequate for screening and not adequate for the final business case, which should use a monthly cash flow model.

### 7.2 The reversibility scorecard, computed three times

The scorecard is more useful than a definition because it produces a number you can argue with. Three decisions, three very different sizes, and a result that will surprise anyone who classifies by cost.

| Dimension | (a) Decide today's expedite | (b) Reassign a customer for one quarter | (c) Sign a 3-year dedicated third-party logistics (3PL) contract | (d) Close a leased DC |
|---|---:|---:|---:|---:|
| Time to undo | 0 (same day) | 1 (two weeks) | 1 (renegotiation) | 2 (>12 months) |
| Cost to undo, % of commitment | 0 (<5%) | 0 (<5%) | 1 (5-25%) | 2 (>25%) |
| Third-party consent | 0 (none) | 0 (notification only) | 1 (counterparty) | 2 (landlord, works council) |
| Asset redeployability | 0 (n/a) | 0 (n/a) | 0 (standard vehicles) | 2 (immobile, specialised) |
| External commitment | 0 (none) | 1 (customer informed) | 1 (service schedule) | 2 (service promise, community) |
| **Total** | **0** | **2** | **4** | **10** |
| **Rung** | Execution | Operational | Tactical | Structural |
| **Typical value at stake** | $2,000 | $150,000 | $4,000,000 | $4,600,000 |

Read rows (c) and (d) together. The contract is nearly as large a commitment as the closure in one-time cash terms, and it scores four points lower because it is exit-able, the assets are redeployable, and no community is affected. The ladder is doing its job: it has separated two decisions of similar size that require completely different processes.

Read rows (a) and (d) together. A factor of two thousand in value and a factor of infinity in reversibility. Any organisation that routes both through the same approval process is wasting effort on one and under-protecting the other.

### 7.3 What to do with a score near a boundary

Scores of 5 or 6 are the awkward cases. A three-year lease on a standard building with no fit-out might score 5. That is a structural commitment in duration and a tactical one in escape. The practical rule is to run the higher-rung process whenever the score is within one point of the boundary, and to say so in the memo: "scored 5, treated as structural because the lease term exceeds the planning horizon."

## 8. Data, standards, and regulatory context

A network study touches more external standards than most analytical projects, because it combines physical assets, employment, accounting recognition, and increasingly carbon. This section separates what is binding from what is convention, because the difference determines whether an assumption can be changed by the analyst or only by a lawyer.

### 8.1 The data objects a network model needs

| Data object | Grain | Typical source | What must be stated |
|---|---|---|---|
| Demand | Market × product family × period | Enterprise resource planning (ERP), point of sale, Module 07 forecast | Unit, period, vintage, uncertainty representation |
| Facility master | Site | Property register, ERP | Role, capacity, lease term, permit status |
| Rate card and freight audit | Lane × mode × volume band | Carrier contracts, freight audit | Effective date, minimum charge, fuel surcharge |
| Distances and drive times | Lane | Routing engine, not straight-line | Road distance, time-of-day assumption |
| Fixed cost | Site × cost centre | General ledger | Avoidable versus allocated |
| Inventory policy | Node × product family | Module 10 output | Policy type, service target, lead time |
| Labour | Site × shift pattern | Human resources | Permanent versus agency, notice period |
| Emissions factors | Lane × mode | Carrier, ISO 14083 | Boundary, factor year |
| Current flows | Lane × period | Transport management system, freight audit | Reconciliation to the ledger |

Two of these are worth a note. **Drive times, not distances.** Straight-line distance understates road distance by 20-40% in most geographies and by more where water intervenes. A network model built on crow-flies distances will systematically prefer long thin lanes, which is exactly the error that looks most plausible on a map.

**Avoidable versus allocated fixed cost.** The general ledger does not distinguish them, and this is the single largest source of overstated closure savings in practice. The reconciliation in Step 6 is where the distinction is established, and it requires a conversation with a management accountant, not a query.

### 8.2 Identity, period, and process standards

| Standard | What it governs | What it constrains in a network model | Grade |
|---|---|---|---|
| GS1 Global Location Number (GLN) | Identification of parties and places | Every site in the model should carry a GLN so it joins to master data, contracts, and events | B |
| GS1 Global Trade Item Number (GTIN) | Identification of trade items | Product families must roll up to identifiable items, not free-text descriptions | B |
| ISO 8601 | Date and time representation | Period definitions must be unambiguous across systems and countries | B |
| ASCM SCOR Digital Standard (SCOR DS) | Process reference for plan, source, make, deliver, return | Provides consistent process language; it is not a regulation | C |
| ISO 14083:2023 | Quantification and reporting of transport chain greenhouse gas emissions | Fixes the emissions boundary and the calculation approach for the carbon term | B |

The GS1 standards are voluntary in most jurisdictions and effectively mandatory in retail and consumer goods, because trading partners require them contractually. Their relevance here is narrower than in a traceability project: a network model joins facility, product, and cost data from at least four systems, and identity standards are what make that join possible without hand-mapping.

SCOR DS is a reference framework, not a requirement. It is useful for consistent vocabulary across a study and for benchmarking, and no regulator will ask whether you followed it.

### 8.3 Carbon as a network parameter

| Instrument | Status as of late 2026 | Effect on a network decision | Grade |
|---|---|---|---|
| Regulation (EU) 2023/956, CBAM definitive regime | Applies from 1 January 2026 to covered goods | Adds a carbon cost to imports of covered goods; a landed-cost parameter, not a disclosure | B |
| ISO 14083:2023 | Published standard | Defines how transport emissions are quantified for the model's carbon term | B |
| GLEC Framework | Global Logistics Emissions Council (GLEC) methodology underlying ISO 14083 | Provides default factors where carrier data is unavailable | C |
| CSRD / ESRS E1 | Corporate Sustainability Reporting Directive (CSRD) and European Sustainability Reporting Standards (ESRS) E1 | Requires scope 3 transport emissions to be disclosed and assured | B |

The practical consequence for this topic is narrow and important: carbon belongs in the same cost table as freight, at a stated factor and a stated price, or it does not belong at all. A carbon term without a stated price is decoration, and a carbon factor computed on a different boundary from the disclosure will produce a number that contradicts the annual report. Module 11 Topic 09 owns the detailed treatment; here the point is that the network decision is one of the few places where a carbon price can change an answer, because it changes the relative cost of a lane rather than the cost of a report.

### 8.4 Accounting for closures

A network decision that closes or opens facilities interacts with accounting standards, and the interaction changes the timing and sometimes the sign of the numbers the finance function will recognise.

| Standard | What it governs | Consequence for a closure case | Grade |
|---|---|---|---|
| IFRS 16 *Leases* | Recognition of leases as right-of-use assets and lease liabilities | A lease is on the balance sheet; the profit and loss (P&L) account shows depreciation and interest, not rent. The model's "rent" line and the ledger's "fixed cost" line will not match without adjustment | B |
| IAS 37 *Provisions, Contingent Liabilities and Contingent Assets* | Recognition of restructuring provisions | A provision requires a detailed formal plan and a valid expectation in those affected; the recognition threshold is an organisational act, not a modelling choice | B |
| IFRS 5 *Non-current Assets Held for Sale and Discontinued Operations* | Presentation of assets held for sale and discontinued operations | A site designated for closure may be presented separately, which changes the comparability of year-on-year figures | B |
| ASC 420 *Exit or Disposal Cost Obligations* (US GAAP) | Recognition of one-time termination benefits and exit costs | Timing of recognition differs from IAS 37; a US-listed entity and an EU entity can book the same closure in different periods | B |
| ASC 360 *Property, Plant, and Equipment* (US GAAP) | Impairment of long-lived assets | An impairment test may be triggered before any closure decision is taken | B |

Two implications. First, **the model's transition cost and the accounts' transition cost are different objects.** The model needs cash timing; the accounts need recognition timing. Both are legitimate, and a business case that uses one while the capital committee expects the other will be challenged.

Second, **IFRS 16 changed the fixed-versus-variable question.** Before 2019, an operating lease was an off-balance-sheet commitment with rent in operating cost. Since IFRS 16, the same lease appears as a right-of-use asset and a liability, with depreciation and interest in the P&L. A network model that treats lease cost as a simple annual fixed charge is not wrong for decision purposes, but it will not reconcile to the ledger without a bridge. Build the bridge in Step 6 or spend the review meeting explaining it.

### 8.5 What is requirement and what is convention

Several things that feel like rules are habits, and an analyst who cannot tell them apart will treat every inherited parameter as a physical constant.

- **The three-to-five-year study cycle** is a convention. It reflects lease terms, capital cycles, and organisational capacity. It is not a regulation, and there is no correct frequency.
- **The strategic/tactical/operational triad** is a framework from 1960s management accounting, not a law. Its boundaries are drawn differently by different authors.
- **A 2% baseline reconciliation tolerance** is a professional convention. It is a reasonable default and it is not derived from anything.
- **A 10% discount rate** is a company-specific number, not a universal one.
- **A "normal" service level of 95% or 98%** is a commercial choice, often inherited from a competitor's promise rather than from customer research.
- **"Network design is a five-year decision"** is a useful simplification. The right statement is that the decision's consequences last as long as its longest commitment, which may be three years or thirty.

### 8.6 What the standards do not decide

No standard tells you how many DCs to have, where to put them, or what service level to promise. The standards constrain the data you must keep, the emissions you must count, and the way a closure is recognised in the accounts. They do not constrain the decision. An analyst who hopes the regulatory framework will settle the argument will wait a long time.

## 9. Industry practice and case evidence

The evidence on network design is uneven. There are a small number of well-documented studies in the operations research literature, a larger number of company claims in investor communications, and a very large number of vendor assertions. This section keeps them apart.

### 9.1 Procter & Gamble, 1993-1997: the best-documented structural study

P&G's North American product supply system had accumulated plants and distribution points through acquisition and product growth, and the flows had never been designed as a system. The company's 1993 "strengthening global effectiveness" initiative created the mandate for a North American product supply study. The team, working with the University of Cincinnati's Center for Productivity Improvement, built a set of linked models: a mixed-integer programme for the plant and distribution network, an uncapacitated facility-location model for DCs, a product-sourcing model, and a geographic information system (GIS) layer that let managers see the answer on a map.

The reported outcome, in Camm, Chorman, Dill, Evans, Sweeney, and Wegryn, "Blending OR/MS, Judgment, and GIS: Restructuring P&G's Supply Chain," *Interfaces* 27(1), 1997, is a reduction of almost 20% in the number of North American plants and savings of well over $250 million per year before tax by mid-1996. The same body of work describes 12 site closures, more than $1 billion in write-offs and transition costs, and around 6,000 people affected through early retirement, relocation, and retraining. The paper also reports that operations research was internally credited with roughly 10% of the total savings, with the remainder attributed to the business decisions the analysis enabled.

Grade A for the case documentation. Read the last two facts together and the case teaches something the headline does not: a structural study that produces a very large annual saving can also produce a very large one-time cost and a very large human consequence, and the analytical contribution is a fraction of the total result. Any network study presented as "the model will save $X" is mis-describing its own role.

### 9.2 Digital Equipment Corporation, early 1990s: the global formulation

DEC manufactured and distributed globally, with plants, DCs, and suppliers spread across regions and material variation in duty, transport, and lead time by market. Arntzen, Brown, Harrison, and Trafton, "Global Supply Chain Management at Digital Equipment Corporation," *Interfaces* 25(1), 1995, describes a multi-commodity, multi-period global model that included production, distribution, duties, and — unusually for the period — explicit treatment of time as well as cost.

Grade A for the formulation and its documentation. The transferable contribution is the insistence that cost alone misranks global options when lead time differs, and the treatment of duty drawbacks as a network parameter. The non-transferable part is the model scale, which at modern data volumes requires decomposition or a commercial solver.

### 9.3 National Semiconductor and Amazon: two consolidation trades

National Semiconductor consolidated from multiple regional warehouses to a single central distribution facility, replacing regional inventory with air freight. The case is taught widely and the figures are case-reported rather than independently audited. Grade C.

The mechanism is the one this topic cares about: **centralisation converts inventory cost into freight cost, and the exchange rate is the freight rate.** When air freight rates rise, the economics of the centralisation deteriorate, and the decision that was right at one rate is wrong at another. That is not a failure of the original analysis; it is the reason network decisions need review triggers.

Amazon's shift from a national fulfilment model toward regionalised fulfilment is described by the company in shareholder communications and earnings commentary as a driver of cost and delivery-speed improvement. Grade C: these are company claims, not independent measurements. The legible mechanism is that shorter lanes reduce transport cost and delivery time while raising the inventory requirement, because demand must be buffered in more places. The structural insight — that network design and inventory policy are one decision — is the same one Module 10 makes from the inventory side.

### 9.4 What happens to the people: the closure evidence

A network study that closes a facility is an employment decision, and the evidence on closures is more consistent than the evidence on network design.

Beer and colleagues, "The urban and regional impacts of plant closures: new methods and perspectives," *Regional Studies, Regional Science*, 2019, review the Australian automotive closures. As reported in that review, three years after displacement roughly one-third of affected workers had left the workforce, one-third had found full-time work, and one-third were unemployed or underemployed, with most of the re-employed earning less. The review also notes that remaining businesses in the supply chain often suffer more than the large original employer, because they have less capacity to absorb the shock. Grade A for the review.

The Michigan Economic Transition Playbook's Closure Impact Assessment tool separates direct impacts — jobs, tax revenue, utility revenue, environmental remediation — from indirect impacts including housing, school enrolment, supplier spillover, and demand for essential services. Grade B: a government practitioner framework, not a research finding.

A 2026 analysis in Iowa State University's *Agricultural Policy Review* uses difference-in-differences to estimate local effects of food manufacturing plant closures in the Midwest, and reports persistent employment losses that extend into non-manufacturing sectors after a lag. Grade C: institutional analysis, not peer-reviewed.

None of this settles whether a closure is right. It settles that the transition cost line in a network model is not a formality, and that the community and workforce consequences are part of the decision rather than an externality to be noted in an appendix.

### 9.5 How often organisations actually revisit the network

The evidence here is almost entirely vendor survey and trade press, and should be read as such.

| Source | Claim | Grade | How to read it |
|---|---|---|---|
| AIMMS network design survey, 2019 | Fewer than 10% of respondents assess their network weekly; about 52% annually or quarterly; about 23% every few years; 38% take weeks to answer a network question, 30% days, 18% months | D | Vendor survey, self-selected, no published methodology |
| CSCMP Edge panel, reported 2022 | Review the network every six months, with a full overhaul every three to five years | E | Practitioner opinion reported in trade press |
| BCI Global, "living network" position | One-off multi-year studies are outdated; recommend quarterly review plus event-driven triggers | D | Consultancy position paper |
| Freightos case account, 2025 | One company moved to a dedicated team running about 50 projects a year and re-modelling the network every six months | E | Single case, trade press, unaudited |

The consistent picture across these sources, weak as they are individually, is that the three-to-five-year full study remains the norm and that a minority of organisations has moved to a lighter, more frequent cadence. The two are not in conflict. A maintained model refreshed quarterly serves tactical questions; a periodic full study serves the structural decision. The mistake is to believe that a faster model makes the structural decision faster, when the binding constraints are leases, capital, and organisational absorption.

### 9.6 The size and shape of the prize

There is no honest single number. What can be said with care is the shape.

**Where the value comes from.** In a network that has not been reviewed in a decade, most of the identified value is in three places: fixed cost from sites that no longer earn their keep, transport cost from flows that cross the network unnecessarily, and inventory from stocking points that exist for historical rather than demand reasons. In a network reviewed within the last three years, the same analysis typically finds much less, because the easy consolidations have been taken.

**Why published percentages should be discounted.** Almost every published network-design saving is one of three things: a vendor case study (Grade D), a company claim in an investor document (Grade C), or a modelling projection in a paper that was not audited after implementation (Grade A for the method, weaker for the outcome). The P&G figure is unusually well documented and still required more than $1 billion of transition cost to realise.

**Why the headline overstates the recurring benefit.** The comparison is usually against the current network's cost, which includes allocated overhead that does not disappear, service costs that the new design also incurs, and transition costs that the headline excludes. The Meridian example in section 6 is a small illustration: a $1.35 million fixed-cost saving became a $0.77 million annual benefit, and a negative NPV.

**What a credible claim looks like.** "The recommended design reduces annual network cost by $X million against a reconciled baseline, after transition costs of $Y million incurred over two years, with a service consequence quantified at $Z million per year, at a 9% discount rate over six years." Every element is checkable. No percentage is offered because a percentage of an unreconciled base is not a fact.

### 9.7 What the evidence does and does not support

It supports the claim that structural studies can produce large, durable value, and that the mechanism is usually the removal of capacity the business no longer needs rather than the discovery of a clever new location.

It does not support the claim that optimisation finds the answer. In the P&G case the models were one input among several, and the business decisions carried most of the value. It does not support the claim that more frequent modelling produces better structural decisions. And it does not support any specific percentage saving, because the base against which a percentage would be computed is almost never disclosed.

## 10. Failure modes and red flags

Each row names a failure, the red flag that is visible in a meeting or a dataset before the failure is diagnosed, the mechanism, the consequence, and the mitigation.

| Failure | Red flag you can see | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Structural decision taken on tactical evidence | The closure case is built from a transport or warehouse cost report | The data nearest to hand is operational | Service and transition costs appear after implementation | Apply the reversibility scorecard at Step 2 and state the rung in the memo |
| Allocated overhead counted as saving | The fixed cost in the case equals the cost centre total | The general ledger does not distinguish avoidable from allocated | Saving overstated, often by 30-50% | Reconcile at Step 6; ask a management accountant which costs disappear |
| Transition cost omitted | The business case has no line for severance, lease exit, or dual running | Transition is owned by functions outside the study team | Payback understated by years; project approved and then overruns | Seven-line cost table with transition as a required line |
| Single-point demand | One forecast number, no range, no vintage date | Module 07 output consumed uncritically | Design fails on modest growth or decline | Three scenarios minimum; date-stamp every input |
| Service encoded as distance | The constraint reads "within 500 km" | Distance is easy to compute and easy to defend | The promise is violated in exactly the markets where it matters | Encode in hours from order to delivery |
| Inventory consequence excluded | The design prefers more sites than the business can stock | Pooling effect invisible to a transport-and-fixed-cost model | Post-implementation inventory blow-out | Carry Module 10's policy into the objective |
| Greenfield answer presented as a plan | The recommendation requires breaking three leases | Legacy constraints excluded from the model | Study rejected at the first review | Run greenfield for the value of freedom, brownfield for the plan |
| Objective function left implicit | Different reviewers are arguing about different questions | The objective was never written down | Months of work lost at the final gate | Write the objective before the model; publish it in the memo |
| Fixed constraints not enumerated | Nobody can say which sites were excluded or why | Constraints inherited rather than chosen | The model ratifies the status quo and calls it analysis | Tag every constraint legal, contractual, or chosen |
| Data vintage unfixed | Two reviewers have different volumes for the same market | Extracts taken on different dates without labels | The study cannot be defended and is re-run | Freeze and date-stamp; refresh only the finalists |
| Optimisation reported as the answer | The memo says "the model shows" and stops | Method mistaken for judgement | Executability untested; failure blamed on the model | Candidate design table; executability test at Step 12 |
| Review trigger absent | The recommendation has no stated reversal condition | The study is treated as a one-off deliverable | Decision becomes permanent by default | Name the parameter, the threshold, and the calendar owner |

Two rows deserve expansion.

**Service encoded as distance.** This is the most common technical error in otherwise competent studies, and it is common because it is convenient. A model with a distance constraint can be drawn on a map, explained in one sentence, and defended without data. A model with a time constraint needs drive times by lane and time of day, which are harder to obtain and harder to explain. The convenience is real and the error is systematic: distance and time diverge most in dense areas, which are the areas where the service promise is most valuable.

**Objective function left implicit.** This is the most common organisational error. Two functions can agree on every number in the model and still disagree on the answer, because one is minimising cost subject to service and the other is maximising service subject to a cost budget. Neither is wrong. The disagreement is about the objective, and it cannot be resolved by more analysis. Writing the objective down at Step 4 costs an hour and saves a quarter.

## 11. Assumptions and limits

The ladder and the procedure in this topic rest on assumptions. Each is stated with what breaks it, because a reader who cannot see the limits cannot judge the advice.

| Assumption | Why the topic needs it | What breaks if it fails | How to test it |
|---|---|---|---|
| The decision can be classified on a single rung | The whole framing depends on discrete levels | Real decisions bundle a structural and a tactical choice, such as "close this site and renegotiate these lanes" | Split the bundle into separate decisions, classify each, and sequence them |
| Reversibility is the right classifier | It separates decisions by escape cost rather than by importance | In a regulated utility, an operationally reversible decision can still be politically irreversible | Add a sixth dimension for political or regulatory exposure when the context requires it |
| The horizon of a structural decision is 3-10 years | It sets the evaluation period | Some assets last 30 years; some categories change structure in 18 months | Use the longest non-cancellable commitment as the anchor, not a fixed number |
| The organisational objective is stable over the horizon | A single objective function is being optimised | Ownership changes, a merger, or a strategy reset replaces the objective | State the objective's sponsor; a change of sponsor is a review trigger |
| Demand, cost, and rate data can be reconciled to the ledger | The baseline is the study's credibility | Source systems disagree and reconciliation cannot close | Run Step 6 before modelling; if it fails, the study is a data project first |
| Leases and permits are the binding constraints on timing | Explains why studies are periodic | In some jurisdictions labour agreements or environmental review dominate | List the gates in Step 12 and take the longest |
| Service can be represented as a constraint or a penalty | Needed to put commercial concerns in the model | Customer behaviour is non-linear: a small degradation causes a large loss of a specific account | Test the penalty value against observed churn where data exists |
| The model's grain can be coarser than the item | Needed for tractability | Product families have different handling, temperature, or value density | Disaggregate the recommended design at item level before implementation |
| Transition cost can be estimated within a factor of two | Needed for the NPV to mean anything | Novel sites, unusual labour markets, or contested closures | If the range is wider, fund a scoping phase before the decision |
| The organisation can execute the recommended change | Needed for the recommendation to be actionable | Works councils, customer contracts, or management capacity block it | Step 12, done as a conversation with named gatekeepers |

Three limits are worth stating beyond the table.

**The ladder is a classification, not a theory.** It does not predict which decisions a company will make or why. It gives a shared vocabulary for arguing about process. The evidence that classification improves outcomes is professional experience, not controlled study, and the topic should not pretend otherwise.

**The four rungs are a simplification of a continuum.** Some decisions sit between rungs, and the boundary between operational and execution is largely a matter of whether the decision allocates units or vehicles. Module 12 draws that line. The value of the simplification is that it forces a conversation about horizon and reversibility; the cost is that boundary cases need judgement.

**The topic is written from the perspective of a single firm.** In a marketplace, a fourth-party logistics provider, or a platform, the structural decision belongs to a different party than the one bearing its consequences. The ladder still applies — someone is making an irreversible commitment — but the decision rights table in section 5.4 does not transfer directly.

## 12. Exercises

### Exercise 1 — Classify eight decisions (recall and application, 30 minutes)

Place each decision on a rung and state the horizon you would use.

1. Renew the lease on the eastern DC for a further five years.
2. Move the Smithson account from the eastern DC to the central DC for the next two quarters.
3. Charter a dedicated vehicle to recover a late delivery to a key account.
4. Add a fourth shift at the central DC for the peak season.
5. Replace the eastern DC with a cross-dock and serve the region from central.
6. Select the carrier for the northern lane for the next twelve months.
7. Re-sequence tomorrow morning's outbound loads.
8. Build a new automated DC on a greenfield site.

**Deliverable:** a table with decision, rung, horizon, capital at risk, and owner role.
**Acceptance criteria:** every rung justified in one sentence; at least two decisions identified as boundary cases with the reason; horizons consistent with the rung.
**Estimated time:** 30 minutes.

### Exercise 2 — Score a decision you have lived through (application, 45 minutes)

Choose a supply chain decision you have personally been involved in — a system implementation, a supplier change, a warehouse move, a planning process redesign. Score it on the five-dimension reversibility scorecard.

**Deliverable:** a completed scorecard with a score per dimension, a total, an implied rung, and a comparison with the process the organisation actually used.
**Acceptance criteria:** each score has evidence, not assertion; the comparison names at least one way the process was over- or under-engineered; the boundary rule from section 7.3 is applied if the total is 5 or 6.
**Estimated time:** 45 minutes.

### Exercise 3 — Write an objective function (design, 40 minutes)

For a company you know or can research, write the objective function for a structural network decision in one sentence and one expression. Name the horizon, the discount rate, and the service treatment.

**Deliverable:** the sentence, the expression, a list of what is excluded, and one paragraph on why the exclusions are defensible.
**Acceptance criteria:** the expression has named terms for fixed, variable, transport, inventory, and transition cost; service appears either as a constraint or as a penalty, not both; the discount rate is stated with its source.
**Estimated time:** 40 minutes.

### Exercise 4 — Separate legal, contractual, and chosen constraints (application, 45 minutes)

Take the fixed constraints from a real or constructed network study and tag each one legal, contractual, or chosen. For each chosen constraint, estimate the cost of relaxing it.

**Deliverable:** a three-column table with the constraint, its tag, and, for chosen constraints, an order-of-magnitude relaxation cost with the basis for the estimate.
**Acceptance criteria:** at least ten constraints; at least three tagged chosen; every relaxation cost has a stated basis or is marked as unknown with a plan to find out.
**Estimated time:** 45 minutes.

### Exercise 5 — Compare a study with a budget (critique, 50 minutes)

Obtain a capital request, a network study summary, or a strategic plan from a public company filing, or construct one from a case. Compare it with the annual budget section of the same filing on the six dimensions in section 5.2.

**Deliverable:** a comparison table and a one-page assessment of whether the document is answering a structural question or a budgeting question.
**Acceptance criteria:** all six dimensions addressed; the assessment names the evidence that would distinguish the two; at least one dimension where the document is ambiguous.
**Estimated time:** 50 minutes.

### Exercise 6 — Compute a closure NPV by hand (calculation, 45 minutes)

A site handled 260,000 units last year. Avoidable fixed cost is $1.42 million per year; allocated overhead is a further $0.55 million. Variable handling is $2.10 per unit at the site and $1.95 at the receiving site. Outbound transport is $2.80 per unit from the site and $4.40 from the receiving site. Inventory falls from $4.1 million to $1.9 million at an 18% carrying rate. One-time costs are $3.9 million. The discount rate is 8% and the horizon is five years.

**Deliverable:** annual cost comparison, annual saving, payback year, NPV, and the break-even annual saving.
**Acceptance criteria:** allocated overhead excluded and the exclusion stated; all five cost lines present; arithmetic shown; NPV to two decimal places in $ million.
**Estimated time:** 45 minutes.

### Exercise 7 — Name what the model cannot answer (critique, 40 minutes)

Take a network recommendation from a public source or a case, and list five questions that decided whether it was implemented and that the model could not answer. For each, name who holds the answer and what evidence they would need.

**Deliverable:** a table of five questions with holder, evidence, and lead time.
**Acceptance criteria:** none of the five is answerable by re-running the model; at least two are organisational rather than technical; each holder is a role, not a department.
**Estimated time:** 40 minutes.

### Exercise 8 — Write the decision memo (design, 60 minutes)

Using the Meridian case in section 6, write a two-page decision memo to the five-element reporting standard. The recommendation is to defer closure to lease expiry.

**Deliverable:** a memo with the five elements, the design comparison, the reversal condition, and an explicit statement of what is not known.
**Acceptance criteria:** all five elements present and identifiable; the reversal condition is a number; the unknowns are specific and each has an owner; no claim exceeds the evidence in section 6.
**Estimated time:** 60 minutes.

### Exercise 9 — Place an organisation on the maturity ladder (application, 45 minutes)

Assess an organisation you know against the five maturity levels in section 13, answer 8.

**Deliverable:** a level assignment with evidence for each of the five levels, and the single next step required to move up one level.
**Acceptance criteria:** the assignment cites artefacts, not opinions — a report, a model, a process, a decision log; the next step is specific and achievable within a year; the assessment names what the organisation would have to stop doing.
**Estimated time:** 45 minutes.

## 13. Self-check answers

**Exercise 1 — classification.**

| # | Decision | Rung | Horizon | Capital at risk | Owner |
|---|---|---|---:|---|---|
| 1 | Renew a five-year lease | Structural | 5+ years | Multi-million | COO with CFO |
| 2 | Move an account between DCs for two quarters | Tactical | 6-24 months | Low | Supply chain director |
| 3 | Charter a vehicle for a late delivery | Execution or operational | Hours to days | Thousands | Dispatch or operations |
| 4 | Add a peak shift | Tactical | 6-24 months | $0.1-1m | Operations director |
| 5 | Replace a DC with a cross-dock | Structural | 3-10 years | Multi-million | COO with CFO |
| 6 | Select a carrier for twelve months | Tactical | 12 months | Contract value | Procurement with supply chain |
| 7 | Re-sequence tomorrow's loads | Execution | Hours | Near zero | Dispatch |
| 8 | Build a greenfield automated DC | Structural | 10-20 years | Largest in the set | Board |

The two boundary cases are 3 and 6. Decision 3 is operational if it allocates volume between existing carriers and execution if it selects a vehicle; classify by what is being chosen, and note that the cost is trivial either way, so the boundary does not change the process. Decision 6 is tactical by horizon and structural by contract if the carrier requires dedicated equipment or a minimum volume commitment; the reversibility scorecard is what settles it, not the calendar.

**Exercise 2 — scoring guidance.**

There is no single correct answer. A strong submission has three properties. First, each dimension is scored with evidence: "time to undo: eleven months, because the supplier qualification takes two quarters and the notice period is ninety days" rather than "difficult". Second, the implied rung is compared with the process actually used, and the comparison is specific: "scored 7, treated as tactical, which is why no transition plan was written and the first year ran at 40% over budget." Third, if the total lands on 5 or 6, the boundary rule is applied explicitly. A weak submission scores by feel, produces a total, and does not compare it with anything.

**Exercise 3 — model answer structure.**

A strong objective sentence: "Minimise the net present value of total network cost from 2027 to 2033 at a 9% discount rate, including fixed, variable, inbound and outbound transport, inventory carrying, service-failure, carbon, and transition cost, subject to a 98% next-day service floor in the top 40 markets and no closure of the two sites with permits that cannot be transferred."

The expression should have one term per cost line, each with a stated driver. Exclusions are defensible when they are either immaterial or genuinely outside the decision — for example, product cost is excluded because it does not vary by network design; corporate overhead is excluded because it does not change. The justification for each exclusion is one sentence and names the test that would reverse it.

**Exercise 4 — tagging guidance.**

A strong submission has at least three chosen constraints and is honest about the relaxation cost. Typical chosen constraints: "the ERP cannot support a fourth site" (relaxation cost: a system change, months, unknown until scoped); "the regional director will not accept it" (relaxation cost: a governance decision, no cash, potentially large career risk); "we have always run three DCs" (relaxation cost: the value of the study). Typical contractual constraints: a lease to 2031, a customer contract naming a delivery point, a carrier minimum-volume commitment. Typical legal constraints: a permit that cannot be transferred, a regulatory obligation to serve a region, a planning consent condition.

A weak submission tags everything contractual and thereby removes the most interesting question from the study.

**Exercise 5 — comparison guidance.**

A strong comparison will find that most corporate documents are hybrids. A capital request for a new DC is structural in content and budgeting in form: it uses an annual cost comparison, a payback calculation, and a hurdle rate, and it does not state a constraint set or an objective function. The useful observation is not that the document is wrong but that its form determines which questions get asked, and a payback framing cannot ask what the design is worth over fifteen years.

**Exercise 6 — arithmetic.**

```text
volume            = 260,000 units
variable, stay    = 260,000 x 2.10 / 1e6 = 0.546
outbound, stay    = 260,000 x 2.80 / 1e6 = 0.728
inventory, stay   = 4.1 x 0.18           = 0.738
stay total        = 1.420 + 0.546 + 0.728 + 0.738 = 3.432

variable, close   = 260,000 x 1.95 / 1e6 = 0.507
outbound, close   = 260,000 x 4.40 / 1e6 = 1.144
inventory, close  = 1.9 x 0.18           = 0.342
close total       = 0.000 + 0.507 + 1.144 + 0.342 = 1.993
```

Annual saving = 3.432 − 1.993 = $1.439 million. Allocated overhead of $0.55 million is excluded because it does not disappear.

Payback = 3.90 / 1.439 = 2.71 years.
Annuity factor at 8% over five years = (1 − 1.08^-5) / 0.08 = 3.993.
NPV = −3.90 + 1.439 × 3.993 = −3.90 + 5.746 = **+$1.85 million**.
Break-even annual saving = 3.90 / 3.993 = **$0.98 million**.

The case works. Note how much of the result depends on the outbound premium: $1.60 per unit of additional transport on 260,000 units is $0.416 million per year, which is 29% of the annual saving. If the receiving site is further away than assumed, the answer can reverse inside a plausible range.

A common wrong answer includes the $0.55 million allocated overhead, giving a saving of $1.989 million and an NPV of $4.04 million. The error is not arithmetic; it is a misunderstanding of what a cost centre reports.

**Exercise 7 — guidance.**

Strong answers draw on gates rather than on data. Examples: can the works council be consulted within the timetable (holder: human resources director; evidence: the consultation agreement; lead time: six months); will the largest customer accept a two-day lead time (holder: commercial director; evidence: the contract and a customer conversation; lead time: weeks); can the receiving site be permitted for the additional throughput (holder: facilities; evidence: the existing consent; lead time: months); will the board hold the decision through a year of dual running (holder: chief executive; evidence: the capital approval minutes; lead time: the decision itself).

A weak answer lists data-quality questions. Those are answerable by auditing data, which is work the model can inform but not replace.

**Exercise 8 — memo guidance.**

A strong memo opens with the recommendation and the five elements, in order. It then gives the annual comparison as a five-line table, the one-time costs as a five-line table, and the two scenarios with break-even years. It states the reversal condition as a number — for example, "the recommendation reverses if the service consequence exceeds 3.2% of the revenue at risk, or if the Central site cannot absorb the peak month without a fourth shift". It lists the four unknowns from section 6.7, each with an owner. It does not describe the model.

A weak memo leads with methodology, buries the recommendation on page two, reports the $1.35 million fixed-cost saving without the offsetting lines, and states no reversal condition.

**Exercise 9 — maturity guidance.**

The five levels are: level 0, no cost-to-serve, cost reported by cost centre; level 1, descriptive cost-to-serve by customer, channel, or order; level 2, scenario comparison of named designs on a common cost base; level 3, optimisation within a stated objective and constraint set, with sensitivity; level 4, a maintained model with a governance process that acts on its output on a stated cadence.

Most organisations that believe they are at level 3 are at level 2, because they compare designs that someone thought of rather than searching the design space. Most organisations at level 1 believe they are at level 2, because they produce a cost-to-serve report and call the comparison of two years a scenario. The assessment should cite artefacts. A strong submission names the artefact that proves the level and the artefact that is missing for the next one.

## 14. Interview questions

**1. "What makes a network decision different from a planning decision?"**

Strong answer: names irreversibility and horizon first, then the inheritance principle — the network sets the inventory, transport, and service problem that every downstream decision inherits, and no amount of downstream optimisation can correct a structural error. It gives an example of a structural error that tactical excellence could not fix.

Weak answer: says network decisions are "bigger" or "more strategic", which is a restatement of the question rather than an answer.

**2. "How do you decide which level a decision belongs to?"**

Strong answer: applies the reversibility scorecard — time to undo, cost to undo, third-party consent, asset redeployability, external commitment — and says the score, not the size of the money, sets the process. It notes that boundary scores are treated at the higher rung.

Weak answer: classifies by job title or by the size of the budget, which produces the failure modes the ladder exists to prevent.

**3. "Why can't a good transport optimisation fix a bad network?"**

Strong answer: because the transport model takes the site list as a parameter. It can find the best flow given the sites, and the best flow through the wrong sites is still the wrong network. It may even look better than it is, because it excludes the fixed cost and the inventory consequence of the site list.

Weak answer: says transport is "downstream" without explaining what that means for the model's parameters.

**4. "What is the difference between a network study and the annual budget?"**

Strong answer: six dimensions — cadence, horizon, objective, constraint set, data vintage, decision rights — and one consequence: a budget cannot see structural options because it inherits the structure. It notes that the two use the same currency and are often reviewed by the same people, which is why they are confused.

Weak answer: says the study is "longer term", which is true and not sufficient.

**5. "Why are network studies periodic rather than continuous?"**

Strong answer: four reasons — data assembly cost, the decision itself is periodic because leases and capital cycles are periodic, organisational absorption, and the study's function as a political object that creates a mandate. It distinguishes continuous tactical modelling from periodic structural study, and notes that a faster model does not make the structural decision faster.

Weak answer: says "because they are expensive", which is one reason and the least interesting.

**6. "Who should own the structural decision?"**

Strong answer: the chief executive or chief operating officer jointly with the chief financial officer, with board approval above a stated threshold. It explains why joint ownership is uncomfortable and correct: finance alone under-weights executability, operations alone under-weights capital and service economics. It checks the threshold against the size of a typical lease, not a typical project.

Weak answer: names a single function, usually supply chain, without addressing the objective conflict.

**7. "Commercial blocked a network study. What went wrong?"**

Strong answer: the objective function omitted a commercial constraint that was never written down, so commercial's only remaining move was a veto at the final gate. The fix is to give commercial an explicit written constraint to defend — a service floor, a coverage requirement — so its objection becomes a costed constraint rather than a block. It adds that this should be settled at Step 4, before modelling.

Weak answer: says commercial was being political, which may be true and is not actionable.

**8. "How would you size the prize from a network study honestly?"**

Strong answer: against a reconciled baseline, with all seven cost lines, after transition costs, with the service consequence quantified, at a stated discount rate, over a stated horizon. It refuses a percentage of an unreconciled base, and it notes that published percentages are mostly vendor material or company claims.

Weak answer: quotes a percentage from a vendor report or a textbook without a base.

**9. "What can a network model not tell you?"**

Strong answer: at least four things — whether the organisation can execute a closure, whether the data are right, whether a lease can be exited and at what price, and whether customers will accept a service change. It adds that the boundary is about the source of the number, not about quantitative versus qualitative: a lease exit cost is a number that comes from a negotiation, not from the model.

Weak answer: says the model cannot capture "soft factors", which is vague and usually means the candidate has not thought about which specific decisions sit outside.

**10. "A site has the highest cost per unit in the network. Should it close?"**

Strong answer: not on that evidence. Cost per unit is a ratio whose denominator is throughput; a site with low throughput and high fixed cost will always look expensive, and closing it may simply move the fixed cost and add transport. The question requires a before-and-after network comparison including transition cost and service. It may also be that the site's cost per unit is high because it serves the sparse region that no other site can reach.

Weak answer: says yes, or says it depends on the strategy, without naming the calculation.

**11. "What is option value and why does it matter here?"**

Strong answer: option value is the worth of retaining the ability to defer or change a decision. It matters because a structural commitment destroys options that have value, so the hurdle for an irreversible decision should be higher than for a reversible one, or the decision should be staged. The Meridian example is concrete: an exit penalty that expires in fourteen months is worth more than any plausible forecast improvement.

Weak answer: confuses option value with the sunk-cost fallacy, which is the opposite discipline.

**12. "How would you tell whether an organisation is ready for optimisation rather than scenario comparison?"**

Strong answer: check whether it has a reconciled cost-to-serve baseline, a single agreed objective, and a constraint set that is written down and owned. Without those, an optimisation produces a number nobody can defend. The maturity ladder is descriptive, not aspirational: level 2 done well beats level 3 done on an unreconciled base. It adds that optimisation is a search over a defined space, and the value comes from searching designs nobody thought of, which requires the space to be defined correctly.

Weak answer: says readiness depends on the software, which inverts the dependency.

## 15. Key terms

- **Structural decision** — a choice that sets the facility and lane set for 3-10 years and costs a large fraction of its value to reverse.
- **Tactical decision** — a choice that allocates demand, capacity, or contracts within a fixed structure over 6-24 months.
- **Operational decision** — a choice that allocates flow or expedites within a fixed assignment over days to weeks.
- **Execution decision** — a choice that selects a vehicle, route, or sequence within a plan over hours; owned by Module 12.
- **Rung** — one of the four levels of the decision ladder, ordered by horizon and reversibility.
- **Reversibility** — the time and cost required to undo a decision.
- **Reversibility scorecard** — the five-dimension, ten-point test that places a decision on a rung.
- **Inheritance principle** — each rung takes the rung above as a parameter and inherits the problem it created.
- **Capital at risk** — the money committed that cannot be recovered if the decision is reversed.
- **Commitment window** — the period during which the organisation is contractually or physically bound.
- **Sunk cost** — money already spent that no future action can recover; irrelevant to the forward decision.
- **Option value** — the worth of retaining the ability to defer or change a decision.
- **Hurdle rate** — the minimum return a project must earn to be funded; not the same as the discount rate.
- **Net present value (NPV)** — future cash flows discounted to today at a stated rate.
- **Weighted average cost of capital (WACC)** — the blended cost of a company's debt and equity, used as a discount rate.
- **Payback period** — the time until cumulative cash inflow equals the initial outflow.
- **Break-even point** — the parameter value at which two options have equal NPV.
- **Greenfield design** — a design with no legacy assets; establishes the value of freedom.
- **Brownfield design** — a design constrained by existing sites, leases, and sunk assets; produces the implementable plan.
- **Cost of legacy** — the gap between the greenfield optimum and the best achievable brownfield design.
- **Facility role** — the function a site performs: production, storage, cross-dock, postponement, returns.
- **Cost-to-serve (C2S)** — the full cost of serving a customer, channel, order, or market, including failure cost.
- **Objective function** — the mathematical statement of what is optimised; should be written before the model.
- **Fixed constraint** — a restriction the model treats as non-negotiable; tag it legal, contractual, or chosen.
- **Decision right** — the authority to make a class of decision; check it against the commitment thresholds.
- **Transition cost** — the one-time cost of moving from the current network to a proposed one.
- **Review trigger** — the named parameter and threshold at which a decision must be reopened.
- **Maturity ladder** — the progression from cost-to-serve reporting, through scenario comparison, to optimisation and a maintained model.

## 16. Sources

Every entry is graded A-E. A is peer-reviewed or audited; B is regulatory, official, or standard-setting; C is a company case, company claim, or institutional analysis; D is vendor or consultancy marketing; E is press summary or practitioner anecdote.

**Foundational and peer-reviewed**

- Anthony, R. N. (1965). *Planning and Control Systems: A Framework for Analysis*. Division of Research, Harvard Graduate School of Business Administration. The origin of the strategic, management control, and operational control split. **A**
- Anthony, R. N. (1964). "Framework for Analysis." *Management Services*, 1(1). Earlier statement of the same framework. **A**
- Govindan, K., Fattahi, M., and Keyvanshokooh, E. (2017). "Supply chain network design under uncertainty: A comprehensive review and future research directions." *European Journal of Operational Research*. DOI: 10.1016/j.ejor.2017.04.009. Source for the planning-horizon taxonomy and for the observation that most network models are single-period. **A**
- Ivanov, D. (2009). "An adaptive framework for aligning (re)planning decisions on supply chain strategy, design, tactics, and operations." *International Journal of Production Research*. DOI: 10.1080/00207540902893417. Argues the levels are usually planned in isolation and should be linked. **A**
- Camm, J. D., Chorman, T. E., Dill, F. A., Evans, J. R., Sweeney, D. J., and Wegryn, G. W. (1997). "Blending OR/MS, Judgment, and GIS: Restructuring P&G's Supply Chain." *Interfaces*, 27(1), 128-142. DOI: 10.1287/inte.27.1.128. **A**
- Arntzen, B. C., Brown, G. G., Harrison, T. P., and Trafton, L. L. (1995). "Global Supply Chain Management at Digital Equipment Corporation." *Interfaces*, 25(1). **A**
- Melo, M. T., Nickel, S., and Saldanha-da-Gama, F. (2009). "Facility location and supply chain management: A review." *European Journal of Operational Research*, 196(2). **A**
- Beer, A., and colleagues (2019). "The urban and regional impacts of plant closures: new methods and perspectives." *Regional Studies, Regional Science*. DOI: 10.1080/21681376.2019.1622440. **A**
- Cooper, R., and Kaplan, R. S. (1988). "Measure Costs Right: Make the Right Decisions." *Harvard Business Review*, September-October. The origin of activity-based costing, the accounting ancestor of cost-to-serve. **B**
- Jones, T. C., and Dugdale, D. (2002). "The ABC bandwagon and the juggernaut of modernity." *Accounting, Organizations and Society*. Traces the naming and diffusion of activity-based costing, including its later decline. **A**
- Kaplan, R. S., and Anderson, S. R. (2004). "Time-Driven Activity-Based Costing." *Harvard Business Review*. The simplification that made activity-based costing implementable at scale. **B**
- Daskin, M. S. (2013). *Network and Discrete Location: Models, Algorithms, and Applications*, 2nd edition. Wiley. **A**
- Chopra, S., and Meindl, P. (2019). *Supply Chain Management: Strategy, Planning, and Operation*, 7th edition. Pearson. **A**
- MIT OpenCourseWare, *Mathematical Programming in Practice*, chapter 5. Teaching treatment comparing the three planning levels on objective, horizon, information, and uncertainty. **C**

**Standards and regulation**

- IFRS 16 *Leases*, International Accounting Standards Board. Right-of-use assets and lease liabilities; the reason a model's rent line and the ledger's fixed cost do not match without a bridge. **B**
- IAS 37 *Provisions, Contingent Liabilities and Contingent Assets*. Recognition criteria for restructuring provisions. **B**
- IFRS 5 *Non-current Assets Held for Sale and Discontinued Operations*. **B**
- ASC 420 *Exit or Disposal Cost Obligations* and ASC 360 *Property, Plant, and Equipment*, Financial Accounting Standards Board. **B**
- ISO 14083:2023, *Greenhouse gases: Quantification and reporting of greenhouse gas emissions arising from transport chain operations*. **B**
- Regulation (EU) 2023/956 establishing a carbon border adjustment mechanism. Definitive regime from 1 January 2026. **B**
- GS1 Global Location Number and Global Trade Item Number standards. **B**
- ISO 8601, date and time representation. **B**
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). A process reference framework, not a regulation. **C**

**Government and institutional**

- Michigan Economic Transition Playbook, Tool 8: Closure Impact Assessment. Michigan Department of Labor and Economic Opportunity. A practitioner framework separating direct and indirect closure impacts. **B**
- *Agricultural Policy Review*, Iowa State University, Winter 2026. Difference-in-differences analysis of local impacts of food manufacturing plant closures in the Midwest. **C**

**Company cases and company claims**

- Amazon, annual shareholder letters and earnings commentary on regional fulfilment, 2023-2025. Company claims about cost and delivery speed; not independently measured. **C**
- National Semiconductor centralised distribution case, as reported in the supply chain literature and standard textbooks. Widely taught; figures case-reported rather than audited. **C**

**Vendor and consultancy**

- AIMMS, network design maturity survey, 2019. Reported cadence and response-time figures; self-selected vendor sample with no published methodology. **D**
- BCI Global, "From Static Network Design to a Living Network." Consultancy position paper recommending quarterly review. **D**
- Tompkins, "Network Optimization: When Is the Right Time to Reevaluate Your Network Design." Consultancy guidance; the three-to-five-year trigger is a heuristic, not a finding. **D**

**Press and practitioner anecdote**

- *Supply Chain Xchange*, "Redesign logistics networks every three to five years, panel says at CSCMP Edge." Practitioner panel opinion reported in trade press. **E**
- Freightos, "From Fixed to Fluid: How Dynamic Network Design is Transforming Global Supply Chains." Single-company account in trade press. **E**

> **Note on citation discipline.** The horizon bands used in this topic (3-10 years structural, 6-24 months tactical, days operational) are a synthesis of the sources above, not a quotation from any single one. Published taxonomies differ: some put the strategic boundary at three to five years, some at two years and above. The bands in this file are the ones the module uses consistently, and the reversibility test, not the band, is the operative rule.

## 17. Links

**This module**

- Module map: [`_index.md`](_index.md)
- Next topic [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- Later in this module: [`03_milp-anatomy.md`](03_milp-anatomy.md), [`04_facility-location.md`](04_facility-location.md), [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- Later in this module: [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md), [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md), [`08_robust-stochastic.md`](08_robust-stochastic.md), [`09_carbon-objective.md`](09_carbon-objective.md), [`10_solvers-heuristics.md`](10_solvers-heuristics.md)

**Prerequisite modules**

- Module 10, inventory structure within a given network: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Module 10, the multi-echelon model the network inherits: [`08_multi-echelon.md`](../10_inventory_optimization_multi_echelon/08_multi-echelon.md)
- Module 10, pooling, which is the inventory cost of consolidation: [`09_pooling-postponement.md`](../10_inventory_optimization_multi_echelon/09_pooling-postponement.md)
- Module 07, demand forecasting, which supplies the demand inputs: [`../07_time_series_forecasting/_index.md`](../07_time_series_forecasting/_index.md)
- Module 08, probabilistic forecasting, for uncertainty representations: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)

**Adjacent modules**

- Module 12, transport execution, which owns the execution rung: [`../12_transportation_routing_last_mile/_index.md`](../12_transportation_routing_last_mile/_index.md)
- Module 13, warehousing and fulfilment: [`../13_warehousing_fulfillment_labor/_index.md`](../13_warehousing_fulfillment_labor/_index.md)
- Module 15, production planning and scheduling: [`../15_manufacturing_production_planning/_index.md`](../15_manufacturing_production_planning/_index.md)

**Programme navigation**

- Part I foundations index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Part II methods index: [`../_index.md`](../_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

> **Note:** all ten topic files in this module are now written, and every link on this page
> resolves to a file that exists on disk.

