# Module 11 — Network Design and Optimization

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (10 lecture, 10 lab, 10 project)
> **Track fit:** Common core; deepest in the optimization and analytics-engineering track
> **Prerequisites:** Module 10 (inventory, service, cost-to-serve inputs); Module 01A (costs and units); basic linear programming
> **Tools:** Python 3.12, Pyomo, HiGHS (open source), OR-Tools; pandas; GeoPandas and QGIS optional for maps
> **Industry question:** "Which facilities should exist in 2030, and what does the answer cost the customer in lead time, carbon, and service?"

---

## 1. Why industry cares

A network design decision — how many facilities, where, and which markets each serves — is made rarely, costs tens to hundreds of millions, and is difficult to reverse once a lease is signed or a building is built. It sets the physical structure within which every other decision in this course operates: the inventory buffers of Module 10, the transport lanes of Module 12, the service promises made to customers.

Three forces have made these decisions live again after two decades in which most companies treated the network as fixed:

- **Cost geometry changed.** Freight rates, fuel, labour, and warehouse rents moved sharply after 2020 and have not returned to the old baseline. A network optimised on 2015 cost ratios is unlikely to be optimal now.
- **Policy changed.** Tariffs, rules-of-origin requirements, and carbon disclosure alter the landed cost of a lane. CBAM's definitive regime began on 1 January 2026 (Regulation (EU) 2023/956), which for the first time puts an explicit carbon cost on imports of covered goods into the EU — a network parameter, not a sustainability report.
- **Service expectations changed.** Two-day and next-day delivery moved from premium to default in many consumer categories, which pushes facilities closer to demand and raises the fixed-cost burden.

The analytical failure is rarely the solver. It is the model around it: a demand forecast that is a single number, a transport cost table that is a straight-line distance multiplied by a rate, a service constraint expressed as "within 500 km" when the customer measures days, and a baseline that cannot be reconciled to the finance ledger. A MILP will answer the question you asked with great precision. Most failed network studies asked the wrong question.

## 2. Learning outcomes

By the end of this module, you can:

- Frame a network decision at the right level: strategic, tactical, or operational.
- Distinguish uncapacitated, capacitated, p-median, and covering formulations, and select the right one.
- Write a complete MILP: sets, parameters, variables, objective, and constraints.
- Model multi-commodity and multi-period flows with shared capacity.
- Encode service constraints as distance, time, or fill-rate requirements and explain what each one costs.
- Build a cost-to-serve model that reconciles to the P&L.
- Run scenario, sensitivity, and break-even analysis on a network model.
- Explain two-stage stochastic and robust formulations, and when the extra machinery is justified.
- Include carbon in the objective using a defensible factor and a stated price.
- Choose between commercial and open-source solvers for a given problem size.
- Validate a solution for feasibility, unit consistency, and baseline fidelity before presenting it.

## 3. Mental model

### 3.1 The decision ladder

| Level | Horizon | Decisions | Typical model | Reversibility |
|---|---|---|---|---|
| Strategic | 3–10 years | Number, location, and size of facilities; make/buy; market entry | MILP facility location | Very low |
| Tactical | 3–18 months | Sourcing assignments, seasonal capacity, lane contracts | Multi-period LP/MILP | Medium |
| Operational | Days–weeks | Order routing, load building, appointment scheduling | VRP, heuristics | High |

Most analytical effort is spent at the operational level, where the value per decision is small, while most value is created at the strategic level, where the decisions are rare. A network study is a high-leverage, low-frequency engagement — which is exactly why its assumptions get less scrutiny than they deserve.

### 3.2 The network as a graph

```
suppliers ──► plants ──► DCs ──► markets
   │            │         │         │
 fixed +     capacity   fixed +   demand
 unit cost   + change-  unit cost  + service
             over                  window

decision variables:  which nodes open  x_j ∈ {0,1}
                     which lanes carry  y_ij ≥ 0
```

Every model in this module is a choice of which nodes to open and which arcs to use, subject to capacity, demand, and service.

### 3.3 Cost-to-serve decomposition

```
cost-to-serve per market =
    inbound to DC        (freight + handling)
  + fixed DC cost / volume
  + outbound to market   (freight + delivery + returns)
  + inventory carrying   (Module 10 policy at that node)
  + cost of service failure (expedites, lost sales)
```

A network model that includes only freight and fixed cost will systematically favour fewer, larger facilities, because it omits the inventory and service-failure costs that rise when facilities are far from demand. The correction is not rhetorical: carry the Module 10 safety-stock model into the network objective, or at minimum add an inventory term per open facility.

## 4. Core concepts

### 4.1 Problem families

| Family | Objective | Constraint | Typical use |
|---|---|---|---|
| Uncapacitated facility location (UFL) | Min fixed + transport | Demand met | Strategic screening; tends to open too few sites |
| Capacitated facility location (CFL) | Min fixed + transport | Demand met; capacity respected | Realistic strategic design |
| p-median | Min demand-weighted distance | Exactly p facilities | Public-sector siting; budget-constrained design |
| Set covering | Min fixed cost | Every demand node covered within radius | Emergency services, regulatory coverage |
| Maximal covering | Max demand covered | p facilities | Coverage with a fixed budget |
| Multi-commodity flow | Min transport + production | Flow conservation per commodity | Product-mix and capacity interactions |
| Two-stage stochastic | Min first-stage + expected second-stage | Scenario-dependent flows | Demand and cost uncertainty |

The p-median and covering families answer different questions from UFL: "best p sites" versus "cheapest set that meets a service standard". Choosing the wrong family is a common and expensive error, because the answers are not close to each other.

### 4.2 MILP anatomy

| Element | Content | Common mistake |
|---|---|---|
| Sets | Facilities `j ∈ J`, markets `i ∈ I`, commodities `k ∈ K`, periods `t ∈ T` | Using a set that changes between scenarios |
| Parameters | `f_j`, `c_ij`, `d_i`, `cap_j`, `e_ij` | Transport cost built from straight-line distance |
| Decision variables | `x_j ∈ {0,1}`, `y_ij ≥ 0`, `z_ijkt ≥ 0`, `inv_it ≥ 0` | Continuous where integrality is required |
| Objective | Fixed + variable + carbon + penalty | Omitting inventory and service-failure cost |
| Constraints | Demand, capacity, linking, service, balance | Missing the linking constraint, which lets flow through a closed site |
| Solution | Optimal within gap, or infeasible | Presenting a solution without checking the gap |

### 4.3 The baseline capacitated formulation

```
min   Σ_j f_j x_j  +  Σ_i Σ_j c_ij y_ij
s.t.  Σ_j y_ij = d_i                    ∀ i        (demand)
      Σ_i y_ij ≤ cap_j x_j              ∀ j        (capacity + linking)
      y_ij ≥ 0,  x_j ∈ {0,1}
```

The linking constraint `Σ_i y_ij ≤ cap_j x_j` does two jobs: it enforces capacity and it forbids flow through a closed facility. Without it, the model will happily ship through a site that was never opened and report an impossible cost — the single most common structural bug in hand-built network models.

Service is added as a restriction on the allowed lane set:

```
y_ij = 0   for all (i,j) with travel_time_ij > service_window_i
```

This is correct but blunt: it treats a 4-hour breach the same as a 40-hour one. A softer alternative allows violations at a penalty:

```
Σ_j t_ij y_ij / d_i ≤ W_i + s_i ,   penalty = π Σ_i d_i s_i
```

The penalty `π` should be the actual cost of a late delivery — expedite cost, lost margin, contractual charge — not a round number chosen to make the model behave.

### 4.4 Multi-commodity and multi-period

| Extension | Adds | Why it matters |
|---|---|---|
| Commodity index `k` | `z_ijkt` flows; per-commodity demand | Shared capacity across products with different economics |
| Period index `t` | `inv_it`, opening/closing binaries | Phased build-out; leases that expire |
| Changeover | `w_jt` binary for plant mode | Production switching cost |
| Modular capacity | `cap_j` in increments `m` | Real facilities come in sizes, not continuous capacity |
| Single-sourcing | `y_ij ≤ d_i u_ij`, `Σ_j u_ij = 1` | Supplier or DC exclusivity requirements |

Single-sourcing turns the problem from an LP into a MILP with a much weaker relaxation. If the business does not actually require exclusivity, do not impose it — the cost of the constraint can be large and should be measured, not assumed.

### 4.5 Fixed versus variable cost

| Cost | Behaviour | Treatment |
|---|---|---|
| Building lease or depreciation | Fixed per site per year | `f_j x_j` |
| Labour (permanent) | Step-fixed by shift pattern | Fixed with capacity step |
| Labour (temporary) | Variable per unit | Unit cost |
| Inbound and outbound freight | Variable per unit, often with a fixed drop charge | Unit cost plus a per-lane fixed term |
| Inventory carrying | Scales with safety stock, which scales with demand and lead time | Derived from Module 10, per site |
| Obsolescence | Higher with more sites and longer dwell | Per-site multiplier on carrying rate |
| Service failure | Higher with fewer, more distant sites | Penalty per unit late |

The fixed/variable split determines the shape of the answer. A model that treats labour as fully variable will prefer many small sites; one that treats it as fully fixed will prefer few large ones. Neither is right — labour is step-fixed, and the model should say so.

### 4.6 Greenfield and brownfield

| Mode | Decision space | Typical output |
|---|---|---|
| Greenfield | All sites candidates; no legacy | True optimum, often unrealisable in the planning horizon |
| Brownfield | Existing sites plus a small candidate list | Incremental design that respects leases and sunk cost |
| Hybrid | Existing sites may close with a penalty | Phased transition path |

Brownfield studies are usually the ones that get implemented, because a greenfield answer that requires closing a leased DC in a region with three years left on the contract is not a recommendation, it is a provocation. Run greenfield to establish the value of the constraint (the "cost of legacy"), then brownfield to produce the plan.

### 4.7 Scenario and sensitivity analysis

| Technique | Question answered | Output |
|---|---|---|
| Tornado | Which parameter moves total cost most? | Ranked bars per input |
| Break-even | At what demand or rate does the preferred design flip? | Threshold value per design pair |
| Shadow prices | What is one more unit of capacity worth? | Dual value, from the LP relaxation |
| Scenario matrix | How does the best design change across futures? | Design × scenario cost table |
| Regret matrix | Which design is least bad across futures? | Minimax regret choice |
| Stress test | Does the design fail under an extreme? | Feasibility and cost under stress |

The most useful single chart in a network study is the break-even chart: it shows the demand or freight-rate threshold at which the recommendation changes, and it converts an argument about the future into an argument about a number.

### 4.8 Robust and stochastic optimization

Two-stage stochastic programming chooses first-stage facilities now and second-stage flows after uncertainty resolves:

```
min  Σ_j f_j x_j + Σ_s p_s Q(x, s)
```

where `Q(x,s)` is the optimal flow cost in scenario `s`. The approach requires a scenario set with probabilities — usually the hard part. Robust optimization instead protects against the worst case within an uncertainty set:

```
min  max_{u ∈ U}  Σ_j f_j x_j + Σ c_ij(u) y_ij
```

Bertsimas and Sim's budget-of-uncertainty formulation lets the analyst control how many parameters may deviate simultaneously, which avoids both the over-conservatism of full worst-case and the optimism of a single point forecast. The practical recommendation: use scenarios when you can defend probabilities, robust when you cannot, and report both when the decision is large enough to justify the extra work.

### 4.9 Carbon in the objective

```
emissions (kg CO2e) = mass (t) × distance (km) × factor (kg CO2e per tonne-km)
carbon cost ($)     = emissions (t) × price ($ per t)
```

**Illustrative arithmetic.** 120,000 units/year, 0.5 kg per unit, 300 km lane, 0.08 kg CO2e per tonne-km.

```
mass      = 120,000 × 0.5 / 1,000 = 60 t
transport = 60 × 300 = 18,000 t-km
emissions = 18,000 × 0.08 = 1,440 kg = 1.44 t CO2e
cost at $100/t = $144 per year
```

Against a lane-cost difference of $480,000 per year, carbon at current prices does not move the facility decision. It can move modal choice on long lanes, where the emissions factor differs by an order of magnitude between air, road, and sea. State the price and the factor explicitly; a carbon term without a stated price is decoration.

Factors should come from the carrier or from ISO 14083:2023, which specifies quantification and reporting of greenhouse gas emissions from transport chain operations. The GLEC Framework provides the underlying methodology that ISO 14083 standardises.

### 4.10 Solvers

| Tool | Type | Strength | Limit |
|---|---|---|---|
| HiGHS | Open-source LP/MILP | Fast, free, good for 10⁵–10⁶ variables | No indicator constraints; basic support model |
| Gurobi | Commercial | Strongest MIP performance; free academic licence | Licence cost at commercial scale |
| CPLEX | Commercial | Mature, strong for large MIP | Licence cost |
| OR-Tools | Open-source CP-SAT and routing | Excellent for VRP and scheduling | Less natural for pure facility MILP |
| Pyomo | Modelling layer | Solver-agnostic, Pythonic, auditable | Overhead on very large models |
| PuLP | Modelling layer | Lightweight, simple | Fewer advanced features |

For a network model with a few hundred candidate sites and a few thousand lanes, HiGHS through Pyomo solves in seconds to minutes. Solver choice becomes material only when the problem has millions of binaries, which usually means the model has been formulated at the wrong grain.

### 4.11 Heuristics versus exact

| Method | When to use | Guarantee |
|---|---|---|
| Greedy add | Very large candidate sets; quick screen | None |
| Greedy drop | Reducing an existing network | None; can be far from optimal |
| Local search / swap | Refining a good starting solution | Local optimum only |
| Lagrangian relaxation | Capacity coupling; decomposable structure | Bound available |
| Benders decomposition | Large multi-commodity, multi-period | Exact with convergence proof |
| Full MILP | Solvable within the time budget | Optimal within gap |

Use a heuristic to generate candidates and an exact model to choose among them. Reporting a heuristic answer as "the optimum" is a factual error, and reviewers in a capital committee will ask.

### 4.12 Validation and feasibility checks

| Check | Failure it catches |
|---|---|
| Every market assigned exactly once | Double counting or dropped demand |
| Flow ≤ capacity × open | Flow through a closed site |
| Flow only on permitted lanes | Service constraint silently violated |
| Total flow equals total demand | Unit conversion error |
| Fixed cost charged only for open sites | Objective understated |
| Baseline reproduces the current cost within a stated tolerance | Wrong rates, wrong volumes, wrong network |
| Units consistent across demand, capacity, and rates | The classic thousand-versus-unit error |
| Solver status optimal or gap reported | Presenting a feasible-but-poor solution |
| Solution stable under ±10% parameter perturbation | Fragile recommendation |
| Infeasibility diagnosed to a named constraint | A model that silently relaxes service |

### 4.13 A complete formulation

```python
"""Capacitated facility location with a leased cross-dock option.
Run: pip install pyomo highspy ; python network.py
Costs in $ thousand; flows in thousand units per year.
"""
import pyomo.environ as pyo

SITES = ["P1", "P2", "P3"]
MARKETS = ["M1", "M2", "M3", "M4"]
DEMAND = {"M1": 120, "M2": 90, "M3": 150, "M4": 60}
FIXED = {"P1": 500, "P2": 420, "P3": 0}      # P3 is leased; pay per unit only
CAP = {"P1": 450, "P2": 420, "P3": 200}
UNIT = {("P1", "M1"): 12, ("P1", "M2"): 14, ("P1", "M3"): 16, ("P1", "M4"): 18,
        ("P2", "M1"): 18, ("P2", "M2"): 12, ("P2", "M3"): 10, ("P2", "M4"): 8,
        ("P3", "M1"): 16, ("P3", "M2"): 17, ("P3", "M3"): 20, ("P3", "M4"): 22}
LANES = list(UNIT)

m = pyo.ConcreteModel("network")
m.J = pyo.Set(initialize=SITES)
m.I = pyo.Set(initialize=MARKETS)
m.L = pyo.Set(initialize=LANES, dimen=2)
m.x = pyo.Var(m.J, domain=pyo.Binary)                    # 1 if site open
m.y = pyo.Var(m.L, domain=pyo.NonNegativeReals)          # flow on lane

m.cost = pyo.Objective(
    expr=sum(FIXED[p] * m.x[p] for p in m.J)
         + sum(UNIT[p, i] * m.y[p, i] for p, i in m.L),
    sense=pyo.minimize,
)
# every market's demand is met
m.serve = pyo.Constraint(m.I, rule=lambda m, i:
    sum(m.y[p, i] for p in m.J) == DEMAND[i])
# capacity and the linking constraint in one line
m.capacity = pyo.Constraint(m.J, rule=lambda m, p:
    sum(m.y[p, i] for i in m.I) <= CAP[p] * m.x[p])

solver = pyo.SolverFactory("appsi_highs")   # fallback: pyo.SolverFactory("highs")
res = solver.solve(m, tee=False)
print(res.solver.status, "objective =", round(pyo.value(m.cost), 1), "($ thousand)")
for p in m.J:
    flow = sum(m.y[p, i].value for i in m.I)
    print(f"{p}: {'open ' if m.x[p].value > 0.5 else 'closed'} "
          f"flow={flow:>5.0f}  cost={sum(UNIT[p, i] * m.y[p, i].value for i in m.I):>8.1f}")
```

The expected output is `ok objective = 5400.0`, with P1 closed, P2 carrying 300 thousand units and P3 carrying 120. The result is worth pausing on: the model closes the facility with the most attractive unit transport rates for two of the four markets, because its fixed cost is not recovered by the traffic it attracts.

### 4.14 Worked cost comparison

All figures are $ thousand per year; the network is the instance in §4.13.

| Configuration | Fixed | Transport | Total | Change vs current |
|---|---:|---:|---:|---:|
| Current: P1 and P2 open | 920 | 4,500 | 5,420 | — |
| Single DC, P2 only | 420 | 5,220 | 5,640 | +220 |
| Single DC, P1 only | 500 | 6,180 | 6,680 | +1,260 |
| All three open (P3 idle) | 920 | 4,500 | 5,420 | 0 |
| Optimum: P2 and P3 | 420 | 4,980 | 5,400 | −20 |

Read the last two rows together. Closing P1 saves $500k of fixed cost and adds $480k of transport, because M1 moves from a $12 to a $16 lane; the net gain is $20k, or 0.4%. A recommendation built on the fixed-cost headline would claim a half-million saving; the honest claim is twenty thousand, and it is contingent on the M1 cross-dock working.

Two checks make the table credible. First, the "single DC, P2 only" row is what a fixed-cost-only heuristic would choose — it has the lowest fixed cost and still loses $220k. Second, the optimum is not robust to service: if M1 requires same-day delivery and only P1 is within the required radius, the model must be re-run with that lane restriction, and the answer reverts to $5,420k. The optimisation found the cost minimum of the problem as stated; the analyst's job is to state the problem completely.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Demand | Market × product × period | ERP, POS, forecast | One consistent unit; de-censored for stockouts |
| Facility candidates | Site | Real estate, strategy | Fixed cost, capacity, lease term, constraints |
| Transport rates | Lane × mode × volume band | Carrier contracts, rate cards | Include fixed drop charges and minimums |
| Distances | Lane | Routing engine, not straight-line | Road distance and drive time |
| Service requirement | Market or customer | Commercial policy | Stated as days, not kilometres |
| Product master | SKU | ERP | Weight, cube, value density, temperature class |
| Emissions factors | Lane × mode | Carrier, ISO 14083 | Same boundary as the cost model |
| Current flows | Lane × period | TMS / freight audit | Reconciles to freight spend in the P&L |

**Standards.** GS1 GLN for facility identity and GTIN for product identity, so that the model's entities match the master data. ASCM SCOR DS for the network and delivery process definitions. ISO 8601 for periods. ISO 14083:2023 and the GLEC Framework for emissions accounting. Regulation (EU) 2023/956 (CBAM) for the carbon cost of covered imports, phased in from 1 January 2026.

**Workflow.** Freeze the demand and rate assumptions with a named owner and a date; build the baseline and reconcile it to the freight ledger before optimising; run greenfield for the value of freedom and brownfield for the implementable plan; test service constraints as scenarios rather than hard-coding them; and produce a break-even chart for every recommendation.

## 6. Industry case studies

### 6.1 Procter & Gamble's North American restructuring (1997)

**Context.** P&G's North American product supply system had accumulated a large number of plants and distribution points through acquisition and product growth, with product flows that had never been designed as a system.

**Intervention.** Camm and colleagues built a large-scale mixed-integer model combining optimisation, internal judgement, and GIS, covering plants, distribution centres, and customer zones, and used it to evaluate consolidating the distribution network.

**Evidence.** The *Interfaces* paper describes the model, the data work, and the organisational process of using it. The reported savings figure is a modelling projection reviewed by the company, not an audited post-implementation result.

**Outcome.** The study projected substantial annual savings and led to a smaller, better-utilised distribution network. The paper's more durable contribution is methodological: the model was credible because it was reconciled to existing accounting data and because managers could interrogate it.

**What transferred.** The reconciliation discipline and the use of GIS to make the answer inspectable by people who do not read LP files. **What did not.** The specific network, and the assumption that a single national model is the right grain — in categories with different service requirements, the model has to be segmented.

### 6.2 Digital Equipment Corporation's global supply chain model (1995)

**Context.** DEC manufactured and distributed globally, with plants, distribution centres, and suppliers spread across regions and substantial variation in duty, transport, and lead time by market.

**Intervention.** Arntzen and colleagues built a multi-commodity, multi-period global supply chain model that included production, distribution, duties, and — unusually for the period — an explicit treatment of time as well as cost.

**Evidence.** The *Interfaces* paper documents the model structure and its application to strategic questions about sourcing and facility configuration.

**Outcome.** The work became a reference formulation for global network design, particularly for its treatment of duty drawbacks and its insistence that cost alone misranks global options when lead time differs.

**What transferred.** The multi-commodity, multi-period structure and the inclusion of time in the objective. **What did not.** The scale of the model as implemented; the same structure at modern data volumes requires decomposition or a commercial solver.

### 6.3 National Semiconductor's centralised distribution

**Context.** The company distributed from multiple regional warehouses, each holding finished goods against a local forecast, with delivery lead times measured in weeks.

**Intervention.** The network was consolidated to a single central distribution facility with air freight to customers, replacing regional inventory with transport speed.

**Evidence.** The case is documented in the supply chain literature and taught in standard textbooks; the figures are case-reported rather than independently audited.

**Outcome.** Inventory and warehouse costs fell and delivery lead time shortened, because the company traded local stock for expedited transport. The trade is the case's lesson: centralisation converts inventory cost into freight cost, and the exchange rate is the freight rate.

**What transferred.** The explicit comparison of inventory carrying cost against expedite cost per lane, which is exactly the trade the model makes. **What did not.** The assumption that air freight remains affordable — when rates spike, the economics of centralisation deteriorate quickly, which is why the answer must be re-run rather than inherited.

### 6.4 Amazon's regional fulfilment network

**Context.** Amazon operated a national fulfilment model in which inventory could ship from any node to any customer, which produced long average distances and high outbound cost.

**Intervention.** Over several years the company shifted toward regionalised fulfilment, placing inventory closer to demand and serving most orders from within the region.

**Evidence.** Described by the company in shareholder communications and earnings calls as a driver of cost and delivery-speed improvement; these are company claims, not independent measurements.

**Outcome.** The company reports lower cost to serve and faster delivery. The mechanism is legible: shorter lanes reduce both transport cost and delivery time, but regionalisation raises the inventory requirement because demand must be buffered in more places.

**What transferred.** The structural insight that the network design and the inventory policy are one decision — regionalisation only pays if the inventory penalty is smaller than the transport and service gain. **What did not.** The scale effects, which depend on Amazon's density of demand and its own logistics assets.

## 7. Labs

### Lab 11.1 — Build and validate a facility location model

**Objective.** Produce a correct MILP and prove it is correct before trusting its answer.

**Data.** A synthetic 10-site, 40-market instance with fixed costs, capacities, and lane rates supplied in the lab pack; one deliberately inconsistent rate.

**Steps.** (1) Write the model in Pyomo. (2) Solve with HiGHS. (3) Verify all validation checks from §4.12. (4) Deliberately remove the linking constraint and observe the impossible solution. (5) Restore it and compare. (6) Report the objective decomposition: fixed, transport, total.

**Deliverable.** A model file, a validation log, and a short note on what the no-linking run produced.

**Acceptance criteria.** Model solves to optimality with a reported gap; every validation check passes or is explicitly waived with a reason; the no-linking failure is documented. **Stretch.** Add single-sourcing and measure the cost of the constraint.

### Lab 11.2 — Cost-to-serve and the fixed-cost illusion

**Objective.** Show how the fixed/variable split changes the recommended network.

**Data.** The Lab 11.1 instance plus inventory carrying costs derived from a Module 10 policy and a service-failure penalty per late unit.

**Steps.** (1) Optimise with freight and fixed cost only. (2) Add inventory carrying cost per open site. (3) Add the service-failure penalty. (4) Compare the recommended networks and total costs. (5) Identify which markets change assignment. (6) Write the decision rule for when the fuller model is worth the data effort.

**Deliverable.** Three network designs, a cost comparison, and a decision rule.

**Acceptance criteria.** Each cost term has a stated source; the inventory term is derived, not assumed; differences between designs are attributed to specific cost terms. **Stretch.** Add carbon at two price levels and show whether the design changes.

### Lab 11.3 — Service constraints and the price of a promise

**Objective.** Quantify what a service promise costs the network.

**Data.** Lab 11.2 instance with drive times; three candidate promises: 24-hour, 48-hour, and 72-hour delivery to every market.

**Steps.** (1) Encode each promise as a lane restriction. (2) Solve each. (3) Record cost, number of open sites, and markets that force a site open. (4) Relax to a penalty formulation and compare. (5) Build the cost-versus-promise curve. (6) Recommend a differentiated promise by market tier.

**Deliverable.** A three-point cost curve, a map of forced sites, and a recommendation with the marginal cost per hour of promise.

**Acceptance criteria.** Promises encoded as time, not distance; the penalty formulation's penalty value justified from data; the marginal cost per hour reported. **Stretch.** Allow a small percentage of late deliveries and compute the cost of that tolerance.

### Lab 11.4 — Scenario, robustness, and the break-even chart

**Objective.** Test whether the recommendation survives the future.

**Data.** Lab 11.3 instance plus three demand scenarios (base, upside, regional shift) and a freight-rate range.

**Steps.** (1) Build a design-by-scenario cost matrix for the top four designs. (2) Compute regret per design. (3) Identify the minimax-regret design. (4) Build the break-even chart for the two leading designs across demand and freight rate. (5) Run a ±10% tornado on all parameters. (6) Write a two-page recommendation that names the conditions under which it fails.

**Deliverable.** Scenario matrix, regret table, break-even chart, tornado, and recommendation.

**Acceptance criteria.** Probabilities stated or explicitly absent; regret computed correctly; break-even thresholds numeric; the recommendation names its failure conditions. **Stretch.** Add a two-stage stochastic model over the three demand scenarios and compare its first-stage choice with the deterministic optimum.

## 8. Assignment — Network design study

Produce a **network design study** for one company or realistic simulated case, of the quality that could go to a capital committee.

| Section | Content |
|---|---|
| 1. Question | The decision, the horizon, and what is out of scope |
| 2. Baseline | Current network, flows, and cost reconciled to the P&L |
| 3. Data | Sources, units, vintage, and known defects |
| 4. Model | Formulation, assumptions, and why this problem family |
| 5. Costs | Fixed, variable, inventory, service failure, carbon, each with a source |
| 6. Service | The promise, encoded as a constraint, with its cost |
| 7. Results | Recommended design, objective decomposition, map |
| 8. Uncertainty | Scenarios, regret, break-even, tornado |
| 9. Validation | Checklist results and reconciliation evidence |
| 10. Recommendation | What to do, in what order, and what would change the answer |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 15% | Solves the wrong family | Correct family chosen | Family choice justified against alternatives and their answers |
| Formulation correctness | 25% | Missing linking or capacity constraint | Complete and feasible | Tight formulation; symmetry and scaling considered; relaxation quality reported |
| Data and units | 20% | Inconsistent units; straight-line distances | Consistent, sourced, dated | Reconciled to the ledger; defects documented and bounded |
| Cost completeness | 15% | Fixed and transport only | Inventory and service included | Every term sourced; sensitivity to each reported |
| Uncertainty | 15% | Single point forecast | Scenarios run | Regret, break-even, and robustness; decision rule under uncertainty |
| Communication | 10% | Solver output pasted | Clear tables and a map | A capital-committee document that names its failure conditions |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| CBAM (Regulation (EU) 2023/956) | Carbon cost on covered imports from 1 January 2026 | Include the cost in the landed-cost model, not in a separate report |
| CSRD / ESRS E1 | Scope 3 transport emissions disclosed and assured | Same emissions factors in the model and the disclosure |
| ISO 14083:2023 | Quantification and reporting of transport emissions | Use the standard's boundary and factors |
| Tariffs and rules of origin | Sourcing and location change landed cost and compliance burden | Model duty per lane; test origin scenarios |
| Labour and community impact | Facility closure affects employment | Include transition cost and a stated closure policy |
| Service equity | Cost-optimal design under-serves remote or low-density markets | Report service by region; set a floor where required |
| Competition and antitrust | Shared network or benchmarking data can raise issues | Do not exchange competitor capacity or rate data |
| EU AI Act (Regulation 2024/1689) | Allocation and access decisions may be high-risk in some uses | Documentation, oversight, and an appeal path |

Two ethical points deserve naming. First, a network optimisation will close the facility with the weakest cost position, and that facility is often in a region with fewer alternative employers; the model can price this, and the decision to price it is a governance choice. Second, a design that serves dense markets cheaply and remote markets expensively is efficient and inequitable; if the organisation has a service obligation, it belongs in the constraint set, not in the appendix.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing linking constraint | Flow exceeds capacity of closed sites | Structure bug in the model | Impossible, understated cost | Validation check; inspect per-site flows |
| Straight-line distances | Rates are suspiciously low on long lanes | Road distance differs from crow-flies by 20–40% | Wrong design, optimistic cost | Routing engine distances and drive times |
| Single-point demand | No scenario range | Certainty assumed | Design fails on modest growth | Scenario and break-even analysis |
| Freight-only objective | Few sites recommended | Inventory and service costs omitted | Service collapse after implementation | Carry Module 10 costs into the objective |
| Unit mismatch | Costs off by orders of magnitude | Thousands versus units | Nonsense answer accepted | Unit table; reconciliation to ledger |
| Unreconciled baseline | Model cannot reproduce current cost | Wrong rates or volumes | No credibility with finance | Reproduce current cost within 2% before optimising |
| Over-tight service radius | Infeasible or absurdly expensive | Radius drawn from a map, not from the promise | Study rejected | Encode the actual delivery promise in days |
| Ignoring lease terms | Greenfield answer requires breaking contracts | No transition constraint | Unimplementable plan | Brownfield model with closure penalties |
| Heuristic reported as optimal | No gap reported | Method mislabelled | Credibility loss at review | Report method and guarantee |
| Full multi-commodity model at SKU grain | Solver runs for days | Wrong grain | Project stalls | Aggregate to product families; disaggregate after |
| Carbon factor from the wrong boundary | Emissions inconsistent with disclosure | Well-to-wheel versus tank-to-wheel mixed | Regulatory mismatch | One factor set, one boundary, stated |
| Design not stress-tested | No failure conditions named | Optimisation presented as truth | Surprise when conditions change | Name the thresholds at which the answer flips |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is the difference between UFL and CFL, and which do you use? | UFL has no capacity limits and tends to under-open; CFL adds capacity and is the realistic strategic model. |
| 2 | Why is the linking constraint essential? | It enforces capacity and forbids flow through closed sites; without it the model reports impossible costs. |
| 3 | How do you encode a 48-hour delivery promise? | Restrict lanes by drive time, not distance; or allow violations at a penalty equal to the real failure cost. |
| 4 | What does the p-median answer that UFL does not? | The best p sites for a fixed number of facilities, which is a different question and yields a different network. |
| 5 | How would you defend a recommendation that saves only 0.4%? | Show the break-even threshold and the service risk; a small saving with implementation cost is not worth doing. |
| 6 | When is a two-stage stochastic model worth it? | Large irreversible capital, defensible scenario probabilities, and a decision that changes materially between scenarios. |
| 7 | How do you build transport costs you can defend? | Carrier contracts and freight-audit actuals by lane and volume band, with fixed drop charges included. |
| 8 | Why does the cost-optimal network often under-serve rural markets? | Density drives cost per unit; without a service floor, low-density demand is served from far away. |
| 9 | What is the cost of legacy? | The gap between the greenfield optimum and the best brownfield design that respects leases and sunk assets. |
| 10 | How do you know your model is right? | It reproduces the current cost within a stated tolerance, every validation check passes, and the answer is stable under perturbation. |

## 13. Key terms

| Term | Definition |
|---|---|
| Facility location problem | Choosing which facilities to open and which demand to assign |
| UFL / CFL | Uncapacitated / capacitated facility location |
| p-median | Minimise demand-weighted distance with exactly p facilities |
| Set covering | Open facilities so every demand node is within a radius |
| Maximal covering | Maximise demand covered with p facilities |
| Linking constraint | Ties flow to facility opening and capacity |
| Fixed cost | Cost incurred if a facility is open, regardless of volume |
| Cost-to-serve | Full cost of serving a market, including freight, inventory, and failure |
| Greenfield | Design from scratch with no legacy assets |
| Brownfield | Design constrained by existing facilities and leases |
| Cost of legacy | Gap between the greenfield optimum and the achievable brownfield design |
| Shadow price | Marginal value of one more unit of a constrained resource |
| LP relaxation | The model with integrality removed; gives a lower bound |
| Integrality gap | Relative difference between the integer solution and the LP bound |
| Benders decomposition | Exact method splitting first-stage and scenario subproblems |
| Lagrangian relaxation | Relaxing coupling constraints into the objective with prices |
| Two-stage stochastic | First-stage design, second-stage recourse under scenarios |
| Robust optimisation | Protect against the worst case within an uncertainty set |
| Minimax regret | Choose the design with the smallest worst-case regret |
| Tonne-kilometre | One tonne moved one kilometre; the basis of freight emissions |
| CBAM | EU carbon border adjustment mechanism for covered imports |
| ISO 14083 | Standard for quantifying transport chain greenhouse gas emissions |

## 14. Sources and further reading

**Standards and regulation**

- ISO 14083:2023, *Greenhouse gases — Quantification and reporting of greenhouse gas emissions arising from transport chain operations*. ISO, 2023.
- Smart Freight Centre, *GLEC Framework for Logistics Emissions Accounting and Reporting*, latest edition. Accessed 27 October 2026.
- Regulation (EU) 2023/956 establishing a carbon border adjustment mechanism. EUR-Lex. Definitive regime from 1 January 2026.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex. Accessed 27 October 2026.
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). Accessed 27 October 2026.
- GS1, GLN and GTIN standards. https://www.gs1.org/standards — accessed 27 October 2026.

**Peer-reviewed and foundational**

- Hakimi, S. L. "Optimum Locations of Switching Centers and the Absolute Centers and Medians of a Graph." *Operations Research*, 12(3), 1964.
- Benders, J. F. "Partitioning Procedures for Solving Mixed-Variables Programming Problems." *Numerische Mathematik*, 4, 1962.
- Geoffrion, A. M., and Graves, G. W. "Multicommodity Distribution System Design by Benders Decomposition." *Management Science*, 20(5), 1974.
- Cornuejols, G., Fisher, M. L., and Nemhauser, G. L. "Location of Bank Accounts to Optimize Float: An Analytic Study of Exact and Approximate Algorithms." *Management Science*, 23(8), 1977.
- Erlenkotter, B. "A Dual-Based Procedure for Uncapacitated Facility Location." *Operations Research*, 26(6), 1978.
- Owen, S. H., and Daskin, M. S. "Strategic Facility Location: A Review." *European Journal of Operational Research*, 111(3), 1998.
- Bertsimas, D., and Sim, M. "The Price of Robustness." *Operations Research*, 52(1), 2004.
- Snyder, L. V. "Facility Location under Uncertainty: A Review." *IIE Transactions*, 38(7), 2006.
- Melo, M. T., Nickel, S., and Saldanha-da-Gama, F. "Facility Location and Supply Chain Management — A Review." *European Journal of Operational Research*, 196(2), 2009.
- Camm, J. D., Chorman, T. E., Dill, F. A., Evans, J. R., Sweeney, D. J., and Wegryn, G. W. "Blending OR/MS, Judgment, and GIS: Restructuring P&G's Supply Chain." *Interfaces*, 27(1), 1997.
- Arntzen, B. C., Brown, G. G., Harrison, T. P., and Trafton, L. L. "Global Supply Chain Management at Digital Equipment Corporation." *Interfaces*, 25(1), 1995.
- Daskin, M. S. *Network and Discrete Location: Models, Algorithms, and Applications*, 2nd ed. Wiley, 2013.
- Chopra, S., and Meindl, P. *Supply Chain Management: Strategy, Planning, and Operation*, 7th ed. Pearson, 2019.
- Simchi-Levi, D., Kaminsky, P., and Simchi-Levi, E. *Designing and Managing the Supply Chain*, 4th ed. McGraw-Hill, 2021.

**Industry**

- Amazon, annual shareholder letters and earnings commentary on regional fulfilment, 2023–2025. Company claims. Accessed 27 October 2026.
- Gartner, "Gartner Predicts 70% of Large Organizations Will Adopt AI-Based Supply Chain Forecasting to Predict Future Demand by 2030," press release, 16 September 2025. Analyst prediction about forecasting, not network design; cited for context only.
- Council of Supply Chain Management Professionals, *State of Logistics Report*, latest edition. Accessed 27 October 2026.

**Tools and datasets**

- HiGHS optimisation solver. https://highs.dev — accessed 27 October 2026.
- Pyomo documentation, including the `appsi` solver interfaces. https://pyomo.readthedocs.io — accessed 27 October 2026.
- Google OR-Tools. https://developers.google.com/optimization — accessed 27 October 2026.
- Gurobi academic licensing and documentation. https://www.gurobi.com — accessed 27 October 2026.

> **Note:** A network model is an argument about the future built from data about the past. Its credibility comes from the baseline reconciliation, not from the solver.

---

> **Next:** [Module 12](12_transportation_routing_last_mile.md) — from where the facilities are to how the goods actually move: routing, last-mile economics, and the optimisation models behind the delivery promise.
