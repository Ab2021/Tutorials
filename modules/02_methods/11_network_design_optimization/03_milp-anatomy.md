# MILP Anatomy — variables, constraints, objective, and the modelling choices that decide solvability

> Module 11 — Network Design and Optimization
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11, topics 01 and 02
> Version 2026.5

*The model-construction topic: from an argument about the network to a program a solver can close.*

---

## 1. Why this topic exists

A network recommendation reaches a capital committee in one of two forms. The first is an argument:
a spreadsheet of savings, a map with arrows, and a sentence that says "consolidating to three
distribution centres (DCs) saves eleven million". The second is a model: a set of declared
decisions, a set of declared requirements, and a number the solver returns with a bound attached.
The first can be wrong in ways nobody can see. The second can be wrong in ways everybody can check.

This topic is about the transition between them. It is not about optimisation theory and it is not
about solver selection, which Topic 10 covers. It is about the twenty or thirty decisions a modeller
makes while writing a mixed-integer linear program (MILP) — a linear program (LP) in which some
variables must take integer values — and about which of those decisions determine whether the
resulting file is a faithful description of the business and whether it can be solved at all.

The topic exists because of a specific and repeatable failure. A solver never rejects a wrong model.
Feed it a model in which flow may pass through a closed facility and it will return an optimal
solution, an objective value, and a tight optimality gap. Every number is correct for the program it
was given. The program describes a business that does not exist. The output is indistinguishable, on
the page, from a correct one.

Four errors account for most of the damage, and each of them is a modelling choice rather than a
computational one:

1. **The missing linkage.** Flow is allowed through a facility that was never opened, because the
   constraint that ties flow to the opening binary was omitted or written in a form that does not
   bind. The reported cost is too low by roughly the fixed cost of the phantom facility.
2. **The wrong big-M.** The constant that switches a link on and off is set from last year's volume
   instead of from capacity. The model becomes infeasible the first time demand grows, and the
   analyst reports that the business cannot serve its customers.
3. **The unit that does not cancel.** Fixed costs are in euros per year, flows in thousands of cases
   per year, and one transport rate in euros per kilogram from a carrier tariff. The objective comes
   out three orders of magnitude wrong and nobody notices because the model still reports
   "optimal".
4. **The unvalidated baseline.** The model cannot reproduce last year's cost on last year's network.
   Every scenario delta it produces is therefore a difference between two numbers that are both
   wrong, and the direction of the error is unknown.

None of these is caught by a solver, a code review, or a senior reviewer reading the objective
function. All of them are caught by the disciplines in this topic: dimensional checking, a bound
audit on every M, an LP relaxation run, a symmetry audit, an infeasibility triage procedure, and a
reconstruction run against a closed historical period.

**Why this matters to a data scientist specifically.** The modelling layer is where your data work
becomes a decision. Topic 02 gave you a graph and a cost decomposition. Those artefacts are necessary
and not sufficient: a graph does not say what may be chosen, and a cost table does not say what must
hold. This topic supplies the missing half — the declaration of variables, the four constraint
families, and the diagnostics that tell you whether the declaration is honest. Topic 04 takes the
result and writes the baseline capacitated facility-location instance in full. Topic 05 adds the
commodity and period indices. If the anatomy here is wrong, every extension inherits the error.

**What this topic does not cover.** It does not teach how to make a solver faster. It names what
makes a model hard — a weak relaxation, symmetry, degeneracy, bad scaling — because those are
properties of your formulation, and the professional response is to reformulate rather than to tune.
Solver behaviour, decomposition, and the choice between exact and heuristic methods belong to
Topic 10.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a complete MILP for a three-plant, four-DC, six-market network in the canonical order —
   sets, parameters, variables, objective, constraints — and **state** the count of each element.
2. **Name** the four constraint families that every baseline network model contains and **predict**
   the specific symptom produced when each is written wrongly.
3. **Choose** a value of M in a linking constraint from a stated bound on the variables, and
   **explain** in one sentence each what happens when M is too small and when it is too large.
4. **Explain** why a fixed charge makes the problem non-convex, and therefore why the opening
   variable must be binary rather than continuous.
5. **Diagnose** a model from two numbers — the LP relaxation bound and the integrality gap — and
   **say** whether the formulation is well formed before a solver has finished running.
6. **Identify** symmetry in a facility-location instance, **quantify** the number of equivalent
   solutions, and **write** a symmetry-breaking constraint where the sites are genuinely
   interchangeable.
7. **Distinguish** an infeasibility caused by the business from an infeasibility caused by a
   mistyped constraint, using a stated triage procedure.
8. **Validate** a model by reconstruction: force the historical footprint, reproduce the historical
   cost within a stated tolerance, and **interpret** the gap between the two.
9. **Report** a model to a standard that a reviewer can audit: sets, parameters and their sources,
   variable count, constraint count, gap at termination, and the validation run.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a five-layer stack, read from the bottom. The bottom two layers are what you know.
The middle layer is what you may choose. The top two layers are what you are trying to achieve and
what must remain true while you achieve it. A model is complete when every layer is declared and
every layer can be counted.

```text
   +-------------------------------------------------------------------+
   │ 5  CONSTRAINTS        17 rows    what must remain true            │
   │      F1 demand          6 rows   every market served exactly once │
   │      F2 balance         4 rows   what enters a DC leaves it       │
   │      F3 plant cap       3 rows   no plant above its capacity      │
   │      F4 DC cap + link   4 rows   flow <= capacity x open          │
   │      F5 lane link      12 rows   (optional) per-lane switch       │
   │      F6 domain          0 rows   non-negative, binary             │
   +-------------------------------------------------------------------+
   │ 4  OBJECTIVE       EUR thousand per year    what you pay         │
   +-------------------------------------------------------------------+
   │ 3  VARIABLES          40 items   what the solver may choose       │
   │      x[p,j] >= 0      12   plant to DC flow,  thousand cases/yr   │
   │      y[j,m] >= 0      24   DC to market flow,  thousand cases/yr  │
   │      z[j] in {0,1}     4   DC open                             │
   +-------------------------------------------------------------------+
   │ 2  PARAMETERS         56 items   what you know, each with a source│
   +-------------------------------------------------------------------+
   │ 1  SETS           3 + 4 + 6      what exists                      │
   +-------------------------------------------------------------------+
```

The counts are those of the worked instance in section 6: three plants, four candidate DCs, six
markets, one commodity, one period. They matter because they are the first thing an auditor asks
for. A model that cannot state its own size has not been read back by its author.

Three properties of the stack are worth memorising.

**Property 1 — Money enters in exactly two places.** The objective and the constraint right-hand
sides. There is no third. A cost that appears nowhere in either is not in the model, however
carefully it was computed; this is why an inventory carrying cost or a service penalty discussed in
a meeting but absent from the objective changes nothing about the answer.

**Property 2 — Every binary needs a constraint that gives it meaning.** A binary variable is a
switch. A switch that nothing reads is a free variable with two values, and the solver will set it
to whichever value is cheaper and report a network with a facility that is simultaneously open and
irrelevant. The linking constraint is the circuit the switch controls.

```text
                 z_j  =  0                         z_j  =  1
          (facility closed)                (facility open)
     +--------------------------+     +--------------------------+
     |   x[p,j]  <=  M  x  0    |     |   x[p,j]  <=  M  x  1    |
     |            =  0          |     |            =  M          |
     +--------------------------+     +--------------------------+
                │                                 │
        flow cannot exist                 flow up to M, no more

     In the LP relaxation the switch slides between 0 and 1:
          z = 0.32   ->   x <= 0.32 M     "a third of a facility"
          and the objective pays 32% of the fixed cost
```

The lower half of that diagram is the whole reason this topic exists. The integer program has two
positions. The relaxation has a continuum, and the objective is linear in the position, so the
relaxation will always prefer a fractional switch when a fractional switch is cheaper than a whole
one. Everything the model does with fixed cost — and therefore every network design answer —
depends on how tightly you can pull that continuum back toward its two endpoints.

**Property 3 — The gap is a forecast of the solve.** The integrality gap is the relative distance
between the best integer solution found and the best bound the relaxation can prove. It is a
property of the formulation, computed in seconds from the root relaxation, and it predicts how much
work remains better than any timing of a previous run. A model with a 2% root gap usually closes in
seconds. A model with a 40% root gap may close in an hour or may not close at all, on the same
instance, on the same machine.

This is the claim worth internalising: **you can tell whether a model will be usable before you
know how long it takes.** The diagnostic is structural, not computational, and it is available at
the moment the first LP is solved. Section 5 turns it into a procedure.

### 3.1 The model as an argument with a proof obligation

An argument owes plausibility; a model owes a proof obligation in four parts. Completeness: every
decision the answer depends on appears as a variable. Fidelity: every requirement appears as a
constraint or a cost. Consistency: units agree, and each coefficient traces to a source.
Reproducibility: a second person can rebuild the number. The first two fail invisibly, and both make
the model answer a different question — "which of these must stay open", or "the cheapest network if
customers do not matter" — which somebody will then act on.

## 4. Definitions and vocabulary

The vocabulary below is the working language of the rest of the module. The fourth column is not
decoration: in this subject a term without a unit or a coding convention is a term that will be
misapplied, and the misapplication is usually silent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| MILP | Mixed-integer linear program: a linear objective and linear constraints in which some variables are restricted to integer values | Used for any optimisation model, including nonlinear ones | Model class, no unit |
| LP relaxation | The same model with all integrality restrictions removed | Treated as an approximate answer rather than a bound | Bound on the objective, same currency |
| Bound | The best objective value the relaxation can prove; a floor for a minimisation | Reported as if it were an achievable cost | Currency per period |
| Incumbent | The best feasible integer solution found so far | Called "the answer" before termination | Currency per period |
| Integrality gap | Relative distance between incumbent and bound at termination | Confused with the gap to the true optimum, which is unknown | Dimensionless proportion |
| Root gap | The same ratio measured at the root relaxation, before any branching | Quoted from a different formulation and compared | Dimensionless proportion |
| Index set | A declared collection of objects the model ranges over | Left implicit in the code and stated only in the slide | Set notation, e.g. `J = {D1..D4}` |
| Parameter | A number the model is given and does not choose | Confused with a variable that the analyst has fixed by habit | Declared unit per parameter |
| Decision variable | A quantity the model may choose | Omitted when the analyst has already decided it | Declared unit per variable |
| Continuous variable | A variable allowed any real value in its bounds | Used where the business counts discrete objects | Flow units per period |
| Binary variable | A variable restricted to 0 or 1 | Used for a quantity that can be 0, 1, or 2 | 0 or 1, dimensionless |
| Objective function | The single scalar the model minimises or maximises | Described in prose without stating its unit | Currency per period |
| Constraint family | A group of rows with the same mathematical form and business meaning | Treated as one row when counting the model | Count of rows |
| Demand satisfaction | The family requiring each market's demand to be met | Written as an inequality, which silently drops demand | Flow units per period |
| Flow conservation | The family requiring inflow to equal outflow at each transshipment node | Omitted, so the model creates and destroys product | Flow units per period |
| Capacity constraint | The family bounding throughput at a node or on a lane | Written without the opening binary, which detaches it from the design | Flow units per period |
| Linking constraint | The family tying a flow to the binary that permits it | Written with a constant that does not bound the flow | Flow units per period |
| Big-M | The constant in a linking constraint, equal to the largest flow the binary must permit | Set to a round number, or to a historical volume | Flow units per period |
| Fixed charge | A cost incurred if a facility is used at all, independent of volume | Charged per unit by mistake, which removes the design question | Currency per period |
| Fixed-charge network design | The problem class with a fixed charge per used arc or node plus a linear flow cost | Assumed to be an LP because the flow part is linear | Problem class, no unit |
| Facility location | The problem class choosing which facilities to open and which demand each serves | Used for any network question, including routing | Problem class, no unit |
| UFL / CFL | Uncapacitated and capacitated facility location | Used interchangeably when capacity is what makes the problem realistic | Problem class, no unit |
| Non-convexity | A feasible region or objective that is not convex, so local and global optima differ | Asserted of any model with a minimum, which is wrong | Property, no unit |
| Symmetry | The property that permuting a group of identical sites maps the model onto itself | Ignored because the sites "look different" in the map | Count of equivalent solutions |
| Symmetry breaking | A constraint that removes equivalent solutions without removing any distinct one | Applied to sites that are not actually interchangeable, cutting off the optimum | Constraint, no unit |
| Degeneracy | A condition in which more constraints are active than the solution strictly needs, so the basis is not unique | Reported as instability in the model rather than in the LP | Property, no unit |
| IIS | Irreducible infeasible subsystem: a minimal set of constraints that cannot all hold | Confused with the full model, which is usually much larger | Set of row identifiers |
| Coefficient scaling | The spread of magnitudes among coefficients in one row or the objective | Left unchecked until the solver reports numerical trouble | Ratio, largest to smallest |
| Model validation | Demonstrating that the model reproduces a known historical outcome | Replaced by "the model solved" | Pass or fail against a tolerance |
| Reconstruction run | A validation run with the historical footprint forced and the historical cost compared | Omitted because the current network "is obviously a solution" | Currency per period, compared |
| Optimality tolerance | The relative gap at which the solver stops and calls the answer optimal | Reported as exact optimality | Dimensionless proportion, e.g. 0.0001 |

Four of these deserve expansion, because they are the ones most often used loosely in a meeting.

**Bound, incumbent, gap.** At any moment during a solve there are two numbers and one ratio. The
bound is what the relaxation proves: no solution can be better than this, for a minimisation. The
incumbent is the best feasible design found so far. The gap is how far apart they are, relative to
the incumbent. When the solver stops at a 0.01% gap it has proved that no design more than 0.01%
cheaper exists; it has not necessarily found the cheapest one, though at that tolerance the
difference is immaterial. When it stops at 5% after an hour, the honest sentence is "the best design
we found is within 5% of the best possible, and we do not know which end of that interval the truth
is at". Most commercial network studies run to a gap and report a point; the gap is the part the
committee needs.

**Big-M is a bound, not a penalty.** M is not a large number that "makes the constraint work". It is
the largest value the linked flow can take when the binary is 1, and it is derivable from the other
parameters: the lane capacity, the plant capacity, the market demand, or the DC throughput. When
someone says "I used a million", the correct response is "a million of what, and why is that the
maximum". Section 5.6 gives the folklore and says where it is right.

**Non-convexity is the reason for the binary.** A fixed charge means the cost of serving a market
jumps at zero: zero if the facility is unused, and a positive step if it is used at all. That jump
is a discontinuity, a discontinuous function is not convex, and a linear program cannot represent a
non-convex feasible set. The binary is the standard way to linearise the jump. This is why a network
design problem is genuinely harder than its flow subproblem, and why removing the binary "to make
it solve" changes the question rather than the method.

**Degeneracy is a property of the formulation, not a fault.** In facility location the LP relaxation
is often massively degenerate: many constraints are active at the optimum, the optimal basis is not
unique, and therefore the dual values are not unique either. A solver returns one of many valid dual
solutions. Reporting a shadow price from that solution as "the marginal value of capacity" is
reporting an artefact of the pivot order.

## 5. The mechanics, step by step

This section builds a model in the order a solver reads it, then checks it in the order a reviewer
should. Each step states its input, its output, and the branch that decides what to do next.

### Phase A — Build the model

**Step 1. State the decision, the horizon, and the boundary.**

- Input: the business question from Topic 01, the graph from Topic 02.
- Output: one sentence naming what may be chosen, over what period, and what is excluded. For the
  worked instance: choose which of four candidate DCs to operate for one representative year, and
  which plant supplies each DC and which DC serves each market, minimising annual fixed and
  transport cost with all demand served.
- Branch: two decisions at different time scales belong in two models. A siting decision and a
  weekly routing decision do not share an objective, because their costs are not commensurable.
- Failure if skipped: the model answers a question nobody asked, and is defensible against it.

**Step 2. Declare the index sets.**

- Input: the node and lane lists from the graph.
- Output: `P` plants, `J` candidate DCs, `M` markets, and the lane sets `A = P x J` and `B = J x M`.
- Branch: decide whether the sets are fixed for the study or vary by scenario. If they vary, rebuild
  the model per scenario rather than re-parameterising, because a set is not a parameter.
- Failure if skipped: lanes present in the data but absent from the model are silently unavailable,
  and lanes absent from the data appear as available. Both produce a cost no feasible plan can
  achieve, and both look like answers.

**Step 3. Declare parameters, each with a unit and a source.**

- Input: the cost decomposition and the demand table.
- Output: a parameter table with five columns — name, meaning, unit, value, source.
- Branch: for every parameter ask whether the business would recognise the number. A transport rate
  built by dividing total freight spend by total tonne-kilometres is an average, and using it on a
  lane decision is the error Topic 02 describes.
- Failure if skipped: the units are inconsistent somewhere, and the objective is only approximately
  wrong, which is worse than obviously wrong because it survives review.

**Step 4. Declare the variables and their domains.**

- Input: the decision sentence from Step 1.
- Output: continuous flows, binaries for openings, and integers only where the business counts
  discrete objects.
- Branch: ask of each variable whether it can be fractional in reality. Flow can. A dock door
  cannot. A truck cannot. A shift cannot. Fractional docks are the usual over-modelling error;
  continuous where integrality is required is the opposite error, and it changes the answer.
- Failure if skipped: a model reports 2.4 dock doors, and the operations team stops reading.

**Step 5. Write the objective in one currency and one period.**

- Input: the parameter table.
- Output: a sum of terms, every one of which is in the same currency per year.
- Branch: convert any term in a different period before it enters. Annualising capital requires a
  stated discount rate and horizon; the conversion is a modelling decision, not arithmetic.
- Failure if skipped: capital and operating terms are added as if commensurable, which biases the
  answer toward whichever term was left in its native units.

### Phase B — The four constraint families

Every baseline network model has these four. Write them in this order, because each depends on the
one before.

**Step 6. Demand satisfaction (F1).**

```text
   sum over j of y[j,m]  =  d[m]        for every market m
```

- What it does: forces the flow arriving at each market to equal that market's demand.
- Written wrongly, variant A: `<=` instead of `=`. The model serves only the demand it finds
  profitable and drops the rest. Total cost falls, service falls, and the objective looks like a
  saving. In a model with a fixed cost per market served, this variant also lets the model avoid
  expensive markets entirely, which is a decision the business did not authorise.
- Written wrongly, variant B: `>=`. The model over-supplies, which is feasible only if there is
  spare capacity, and it pays transport for product nobody buys. The objective rises and the answer
  is conservative rather than dangerous, which is why variant B survives longer in practice.
- Written wrongly, variant C: the row is written per DC rather than per market, so a market can be
  served twice by two DCs or not at all.
- Detection: sum the served quantity by market and compare to the demand table. If any market is
  short or long, the family is wrong.

**Step 7. Flow conservation (F2).**

```text
   sum over p of x[p,j]  -  sum over m of y[j,m]  =  0     for every DC j
```

- What it does: makes each DC a pass-through. What arrives leaves.
- Written wrongly, variant A: the row is omitted. The model now creates product at DCs and destroys
  it at plants. Every unit of demand can be met from the cheapest plant without any inbound
  transport, and the reported cost is arbitrarily low. This is the single most common structural
  bug in hand-built network models, because the row is easy to forget when the modeller is thinking
  in terms of "flow from plants to markets" and has already written demand and capacity.
- Written wrongly, variant B: `<=` rather than `=`. Inflow may exceed outflow, so the model can
  discard product at a DC. The cost understatement equals the avoided production and inbound
  transport on the discarded quantity, and it is modest enough to survive review.
- Written wrongly, variant C: `>=`. Outflow may exceed inflow, so the model creates product. The
  saving equals the avoided production and inbound cost on the created quantity.
- Detection: for each DC, compute inflow minus outflow in the solution. It must be zero to the
  precision of the units. A non-zero value is a bug, not a rounding artefact, unless the units are
  so large that the tolerance is material.

**Step 8. Capacity (F3 and F4).**

```text
   sum over j of x[p,j]  <=  s[p]              for every plant p
   sum over p of x[p,j]  <=  cap[j] * z[j]     for every DC j
```

- What it does: bounds throughput at each node, and — in the DC row — ties throughput to the opening
  binary.
- Written wrongly, variant A: the plant row is omitted. The cheapest plant supplies all demand
  regardless of its real capacity, and the model recommends a network that cannot be operated.
- Written wrongly, variant B: the DC row omits `z[j]`. Flow through closed facilities is now
  permitted, and the model reports a cost that is too low by exactly the fixed cost of every
  phantom-open site it exploits.
- Written wrongly, variant C: the DC row uses a capacity that is a nameplate figure rather than an
  achievable one. The model fills the facility to a throughput the operation cannot reach, which is
  a fidelity error rather than a structural one and is much harder to see.
- Detection: check flow against capacity for every open site, and check that every site with
  non-zero flow has `z = 1`.

**Step 9. Linking (F5), when it is separate from capacity.**

```text
   x[p,j]  <=  M[p,j] * z[j]      for every lane (p,j)
   with  M[p,j] = min( s[p], cap[j] )
```

- What it does: makes the binary a switch on each individual lane. Where F4 already contains the
  binary, F5 is implied and is added only to strengthen the relaxation; where the model has no
  aggregate capacity row, F5 is the only thing standing between the solver and a phantom facility.
- Written wrongly: `M` set from a historical volume, from a round number, or from total demand. The
  consequences are the subject of Step 12 and section 5.6.
- Detection: for every lane, compare M to the maximum flow that lane can physically carry, computed
  from the parameters, not from the solution.

### Phase C — Check the model

**Step 11. Dimensional check every row.**

- Input: the formulation.
- Output: for each row, the unit of the left side and of the right side, written out.
- Procedure: write the units under each term and cancel. In the DC capacity row, the left side is
  thousands of cases per year and the right side is thousands of cases per year times a
  dimensionless binary. They agree. In an objective built from fixed cost in euros per year and
  flow in thousands of cases per year, the transport coefficient must be euros per case, not euros
  per thousand cases; if it is the latter, every transport term is a thousand times too large and
  the model will recommend a network with no transport at all.
- Branch: if any row fails, fix the coefficient before doing anything else. A model with a unit
  error can be feasible, optimal, and worthless.

**Step 12. Audit every M against a derived bound.**

- For each linking row, compute the tightest valid M from the parameters and compare.
- Record the ratio of the M used to the M derived. A ratio above about 2 is a flag; above 10 it is a
  defect.
- Note that M must be an upper bound on the flow, so the derived value is a minimum permitted
  choice, not a maximum. Choosing M smaller than the derived bound is not "tighter", it is wrong.

**Step 13. Solve the relaxation and compare.**

- Solve the LP relaxation. Record the bound, the number of fractional binaries, and the gap to a
  quick feasible solution.
- A relaxation in which every binary is 0 or 1 at the optimum is a sign that the fixed costs are
  small relative to transport, or that the instance is easy. It is not a proof that the model is
  correct.
- A relaxation in which a large share of binaries sit strictly between 0 and 1 is a sign that the
  formulation is weak. That is a diagnosis, not a failure; section 6 quantifies one case.

**Step 14. Audit symmetry.**

- For each group of candidate sites, ask whether swapping two of them produces a solution of
  identical cost. If yes, the group of size k contributes k factorial equivalent solutions.
- Four identical candidates give 24 equivalent solutions; six give 720; eight give 40,320.
- Decide whether to add ordering constraints (which site of an identical group opens first) or to
  leave the symmetry and accept the search cost. The decision depends on whether the sites are
  genuinely interchangeable in the business, not only in the cost table.

**Step 15. Check for degeneracy symptoms.**

- Symptoms in the output: assignment maps that change substantially between runs with identical
  inputs and different solver seeds; dual values that change sign between runs; a solution in which
  many lanes are used at exactly zero flow with positive reduced cost.
- Consequence for reporting: do not present dual values from a degenerate LP as marginal values
  without a stability check across runs.

**Steps 16 to 18. Triage, validate, report.** The procedures for infeasibility triage and
reconstruction are sections 5.7 and 5.8. The last step is the report: status, objective, bound,
relative gap, and the time or node limit that ended the run. A run stopped by a time limit is not
"optimal", because the word is reserved for a run that closed the gap to the stated tolerance.

### 5.5 The families written wrongly, in one table

| Family | Wrong form | What the solver exploits | Visible symptom | Cost of the error |
|---|---|---|---|---|
| F1 demand | `<=` instead of `=` | Drops unprofitable demand | Served quantity below the demand table | The margin on the dropped markets |
| F2 balance | `<=` instead of `=` | Discards product at a DC | Inbound exceeds outbound | Avoided production plus inbound cost on the discarded units |
| F3 plant cap | Row omitted | Uses the cheapest plant without limit | One plant carries most of the flow | The cost difference to the constrained plant |
| F4 DC cap | `z` omitted | Flows through closed sites | Closed site shows throughput | The fixed cost of the phantom sites |
| F5 lane link | M from history | Nothing, until demand grows, then infeasibility | Infeasible in growth scenarios only | The whole scenario analysis |

The `<=` demand row is the most dangerous single character in the model: it converts a service
requirement into an opportunity, and the objective reports the saving.

### 5.6 The folklore of M, and where it is right

Practitioners carry three rules about big-M. Each is a compressed version of a real argument.

**Rule 1: "Use the smallest M that is still valid."** Correct, and the reason is the relaxation. In
the LP relaxation the binary can be fractional, and the constraint `x <= M z` becomes `z >= x/M`. A
smaller M forces z closer to 1 for the same flow, so the relaxation pays more of the fixed cost and
the bound is tighter. The limit is validity: M must bound the flow the binary permits, so the
smallest valid M is the maximum physical flow on that lane, which is the minimum of the source
capacity, the destination capacity, and the demand it could serve.

**Rule 2: "Never use a million."** Correct as far as it goes, and the reason is numerical as well as
structural. A very large M makes every coefficient in the row comparable to nothing else in the
matrix, and LP solvers work with tolerances relative to coefficient magnitude. The practical
consequence is that a variable worth one unit can become invisible next to a coefficient worth a
million, and the solver reports feasibility for a point that is not feasible. The literature on
numerical accuracy in mixed-integer solving is explicit that large coefficient ranges degrade both
performance and accuracy, and that the standard remedies operate on the model rather than on the
solver's tolerance settings.

**Rule 3: "A big M only costs time, not correctness."** Wrong, in both directions. Too large costs
time and, at the margin, accuracy. Too small costs correctness: it removes feasible solutions, and
if it removes all of them the model reports infeasible. The asymmetry matters because the second
failure looks like a business finding.

The folklore is right about the direction and silent about the derivation. The professional habit is
to compute M from the parameters and to record the derivation in the model documentation, so that a
reviewer can check the number without reading the code.

> **Note:** a redundant-looking linking row can still improve the bound. If the aggregate capacity
> row already implies `x[p,j] <= cap[j] z[j]`, adding the per-lane row changes nothing about the
> integer feasible set and can still cut fractional points off the relaxation when the plant
> capacity is smaller than the DC capacity. Redundancy in the integer program and redundancy in the
> relaxation are different properties, and only the second one affects the solve.

### 5.7 Infeasibility: the business or the typing

An infeasible model is a statement that no assignment satisfies all constraints. It is not a
statement about the business until you have eliminated the possibility that it is about your typing.
Work through these five tests in order; each eliminates a class of cause.

| Test | What you do | If it fails | What it means |
|---|---|---|---|
| 1. Aggregate feasibility | Compare total demand to total capacity of the candidate set | Demand exceeds capacity | Either the business cannot serve this demand from this candidate set, or a capacity parameter is wrong |
| 2. Single-market feasibility | For each market, check that at least one open-able DC can reach it with enough capacity | A market has no feasible supplier | A lane is missing from the data, or a service restriction excludes the only lane |
| 3. Unit sanity | Recompute total demand and total capacity in a single unit | The two totals differ from the reported totals | A unit conversion error, usually cases against eaches |
| 4. Constraint relaxation | Relax one family at a time and re-solve | The model becomes feasible when one family is relaxed | That family contains the conflict; read its rows |
| 5. Historical replay | Force the historical footprint and historical flows | The model cannot reproduce a plan that actually happened | The data or a parameter is wrong, because the plan was feasible in reality |

Test 5 is decisive and is the one most often skipped. If last year's network served last year's
demand, then a model that cannot represent that plan is wrong about the world, because the world
produced a counterexample. An infeasibility that survives test 5 is a data defect, with no
exception worth arguing about.

An infeasibility that passes all five tests is a business finding: demand exceeds the capacity the
candidate set can provide, or a service promise cannot be met from any admissible site. That is a
real result and should be reported as one, with the binding constraint named.

### 5.8 Validation by reconstruction

The purpose of reconstruction is to make the model prove it knows the past before it is allowed to
comment on the future. Run it in two modes.

**Mode A — fixed footprint, optimised flows.** Force `z[j] = 1` for the historically open sites and
`z[j] = 0` for the rest, then solve. The result is the cheapest flow pattern within last year's
footprint. It should be no more expensive than the actual cost. If it is more expensive, a
constraint is missing or a parameter is wrong: the model is forbidding something the business did.

**Mode B — fixed footprint, fixed flows.** Force the flows to the actual historical volumes and
evaluate the objective without optimising. The result is the model's reconstruction of last year's
cost and should equal the ledger within a tolerance agreed in advance. Vendors of network design
software commonly describe a target of within 5% to 10% of actual costs and volumes for a validated
base case; that is a reasonable starting tolerance for a first build and a loose one for a model
that will carry a capital decision. A tighter tolerance is achievable when the cost data reconciles
to the freight ledger line by line.

The two modes answer different questions, and the difference between them is itself diagnostic.

| Mode A versus Mode B | Reading |
|---|---|
| Mode B matches the ledger, Mode A is slightly cheaper | The model is well calibrated and there is a small operational opportunity within the current footprint. |
| Mode B matches, Mode A is much cheaper | Either a large quick win, or the model is missing a constraint the business operates under (a contract, a service promise, a lane restriction). Investigate before believing the win. |
| Mode B does not match | A data or parameter error. Do not proceed to scenarios. |
| Mode B matches, Mode A is more expensive | The model forbids something the business did. A missing lane, an over-tight capacity, or a service restriction applied too broadly. |

Mode A being much cheaper than Mode B is the most interesting outcome and the one most often
mislabelled. It is either the finding of the study or the bug of the study, and the only way to tell
is to open the solution, pick the largest single flow change, and ask the planner who made that
decision last year why they did not make it.

## 6. Worked example

The instance is deliberately small enough to check by hand: three plants, four candidate DCs, six
markets, one product family, one representative year. Costs are in euros; flows are in thousands of
cases per year, written kcases.

### 6.1 Sets and parameters

Plants `P = {P1, P2, P3}`. Candidate DCs `J = {D1, D2, D3, D4}`. Markets `M = {M1, ..., M6}`.
Lanes: 12 plant-to-DC and 24 DC-to-market, all permitted.

| Plant | Capacity (kcases/yr) | Production cost (EUR/case) |
|---|---:|---:|
| P1 | 150 | 8.00 |
| P2 | 120 | 9.50 |
| P3 | 100 | 7.20 |

| DC | Fixed cost (EUR k/yr) | Throughput capacity (kcases/yr) |
|---|---:|---:|
| D1 | 2,300 | 130 |
| D2 | 1,450 | 130 |
| D3 | 2,100 | 140 |
| D4 | 1,100 | 140 |

| Market | Demand (kcases/yr) |
|---|---:|
| M1 | 55 |
| M2 | 45 |
| M3 | 75 |
| M4 | 30 |
| M5 | 55 |
| M6 | 40 |
| **Total** | **300** |

Inbound unit costs, EUR per case, plant to DC:

| | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|
| P1 | 1.20 | 1.60 | 2.10 | 2.50 |
| P2 | 1.80 | 1.30 | 1.50 | 2.20 |
| P3 | 2.40 | 2.00 | 1.40 | 1.70 |

Outbound unit costs, EUR per case, DC to market:

| | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 1.00 | 1.40 | 2.20 | 2.60 | 3.00 | 3.40 |
| D2 | 1.60 | 1.00 | 1.20 | 1.80 | 2.40 | 2.80 |
| D3 | 2.40 | 1.60 | 1.00 | 1.20 | 1.60 | 2.00 |
| D4 | 3.00 | 2.40 | 1.40 | 1.00 | 1.20 | 1.60 |

> **Note:** the inbound costs look large relative to the outbound costs. That is a deliberate
> feature of the instance, not an error: the plant-to-DC leg is the long-haul leg, and the DC choice
> is decided by the sum of fixed cost, line-haul, and last-leg cost. In a real instance the ratio
> comes from the data, and the analyst should check it against the freight ledger before trusting it.

### 6.2 The current network, costed

The current footprint is D1, D2, and D4 open, D3 closed. Flows are the ones the operation actually
ran.

| Lane | Flow (kcases) | Unit cost (EUR/case) | Cost (EUR k) |
|---|---:|---:|---:|
| P1 → D1 | 55 | 1.20 | 66.0 |
| P1 → D2 | 70 | 1.60 | 112.0 |
| P1 → D4 | 25 | 2.50 | 62.5 |
| P2 → D2 | 50 | 1.30 | 65.0 |
| P3 → D4 | 100 | 1.70 | 170.0 |
| **Inbound subtotal** | **300** | | **475.5** |

| Lane | Flow (kcases) | Unit cost (EUR/case) | Cost (EUR k) |
|---|---:|---:|---:|
| D1 → M1 | 55 | 1.00 | 55.0 |
| D2 → M2 | 45 | 1.00 | 45.0 |
| D2 → M3 | 75 | 1.20 | 90.0 |
| D4 → M4 | 30 | 1.00 | 30.0 |
| D4 → M5 | 55 | 1.20 | 66.0 |
| D4 → M6 | 40 | 1.60 | 64.0 |
| **Outbound subtotal** | **300** | | **350.0** |

Production cost: P1 150 at 8.00 is 1,200.0; P2 50 at 9.50 is 475.0; P3 100 at 7.20 is 720.0. Total
2,395.0.

| Component | EUR k/yr |
|---|---:|
| Production | 2,395.0 |
| Inbound transport | 475.5 |
| Outbound transport | 350.0 |
| Fixed DC cost (D1 2,300 + D2 1,450 + D4 1,100) | 4,850.0 |
| **Total** | **8,070.5** |

The ledger for the year, reconciled line by line, says EUR 8.07 million. The instance is calibrated
to it.

### 6.3 A candidate alternative

Close D1 and open D3. The candidate flow pattern below is feasible; whether it is the cheapest
pattern within that footprint is a solver question, and nothing in this hand analysis depends on
the answer.

| Lane | Flow (kcases) | Cost (EUR k) |
|---|---:|---:|
| P1 → D2 | 80 | 128.0 |
| P2 → D2 | 50 | 65.0 |
| P3 → D3 | 45 | 63.0 |
| P3 → D4 | 55 | 93.5 |
| P1 → D4 | 70 | 175.0 |
| **Inbound subtotal** | **300** | **524.5** |

| Lane | Flow (kcases) | Cost (EUR k) |
|---|---:|---:|
| D2 → M1 | 10 | 16.0 |
| D2 → M2 | 45 | 45.0 |
| D2 → M3 | 75 | 90.0 |
| D3 → M1 | 45 | 108.0 |
| D4 → M4 | 30 | 30.0 |
| D4 → M5 | 55 | 66.0 |
| D4 → M6 | 40 | 64.0 |
| **Outbound subtotal** | **300** | **419.0** |

Production is unchanged at 2,395.0 because the plant totals are unchanged at 150, 50, and 100.

| Component | Current | Candidate | Change |
|---|---:|---:|---:|
| Production | 2,395.0 | 2,395.0 | 0.0 |
| Inbound transport | 475.5 | 524.5 | +49.0 |
| Outbound transport | 350.0 | 419.0 | +69.0 |
| Fixed DC cost | 4,850.0 | 4,650.0 | −200.0 |
| **Total** | **8,070.5** | **7,988.5** | **−82.0** |

The saving is EUR 82 thousand per year, 1.02% of the current cost. The fixed cost falls by 200 and
the transport rises by 118, because M1 loses its cheap lane from D1 and must be split between D2
and D3, with D2's capacity binding at 130.

Two readings of this table are the modelling lessons. First, the headline is not the fixed cost: a
slide that says "close D1, save EUR 2.3 million" is arithmetically true and decision-wrong, because
the honest headline is EUR 82 thousand and it is contingent on D2 filling to its limit. Second, the
binding constraint is what makes the answer fragile: D2 runs at 130 of 130, so a 5% demand increase
in M2 or M3 forces flow onto more expensive lanes and the saving disappears. The candidate is not
the optimum in any proven sense; it is one feasible alternative that beats the current footprint.

### 6.4 What the relaxation says about the three ways of writing the switch

Focus on D3 in the candidate. It carries 45 kcases and opens at a fixed cost of 2,100. Write the
linking logic three ways and solve the relaxation in each. The relaxation may set `z[D3]` below 1,
and the fixed cost it charges falls in proportion.

| Form of the switch | Constraint on D3 | Fractional z permitted | Fixed cost charged (EUR k) | Fixed cost avoided (EUR k) |
|---|---|---:|---:|---:|
| Capacity row with z | `45 <= 140 z` | 0.321 | 675.0 | 1,425.0 |
| Aggregate link on total demand | `45 <= 300 z` | 0.150 | 315.0 | 1,785.0 |
| Per-market link | `45 <= 55 z` (M1 only) | 0.818 | 1,718.2 | 381.8 |

The last row is the tightest of the three because M1's demand of 55 is the smallest valid upper
bound on the flow from D3 to M1. The first is the standard capacity row and is 2.5 times weaker
than necessary in this instance, because D3's capacity is far above its load. The second is the
weakest and is the one modellers write when they think "M is a large number, and total demand is
the largest number I have".

The gap between the first and third rows, EUR 1,043 thousand of fixed cost, is money the relaxation
is allowed to spend on a facility it has not opened. It is not a solution error; the integer
program still returns the right answer if it can be solved. It is an error in the bound, and the
bound is what determines how much search remains.

### 6.5 A hand-computable integrality gap

Take a single-facility subproblem to see the mechanism without a solver. One DC, one market, demand
55 kcases, capacity 140 kcases, fixed cost 1,100, outbound cost 1.00 per case, no inbound cost.

```text
   LP relaxation:  z = 55 / 140 = 0.3929
                   cost = 1,100 x 0.3929 + 55 x 1.00 = 432.1 + 55.0 = 487.1
   Integer:        z = 1
                   cost = 1,100 + 55.0 = 1,155.0
   gap = (1,155.0 - 487.1) / 1,155.0 = 0.578, or 57.8%
```

A 58% gap on a one-facility instance is normal, and it says something precise: for this demand and
this capacity, the fixed cost is mostly a step that the relaxation cannot see. It also says that
the gap is a property of the ratio of demand to capacity, not of the solver. Raise the demand to 140
and the gap collapses to zero, because the facility is full in the relaxation anyway. A model whose
facilities are lightly loaded will always be gap-expensive, and that is a fact about the business,
not about the code.

## 7. Code, query, or calculation

The formulation for the instance in section 6, written as algebra rather than code. The instance is
three plants, four DCs, six markets, one commodity, one period. Read it against the stack diagram in
section 3: the counts there appear here as index ranges.

```text
Sets   P = {P1,P2,P3}   J = {D1,D2,D3,D4}   M = {M1..M6}
       A = P x J  (12 lanes)   B = J x M  (24 lanes)

Vars   x[p,j] >= 0        plant to DC flow            kcases/yr
       y[j,m] >= 0        DC to market flow            kcases/yr
       z[j] in {0,1}      DC open                      0 or 1

min    SUM_p cprod[p] * SUM_j x[p,j]                   production cost
     + SUM_(p,j) cin[p,j] * x[p,j]                     inbound transport
     + SUM_(j,m) cout[j,m] * y[j,m]                    outbound transport
     + SUM_j fix[j] * z[j]                             fixed DC cost

s.t. F1  SUM_j y[j,m] = d[m]                     for all m
     F2  SUM_p x[p,j] - SUM_m y[j,m] = 0         for all j
     F3  SUM_j x[p,j] <= scap[p]                 for all p
     F4  SUM_p x[p,j] <= dcap[j] * z[j]          for all j
     F5  x[p,j] <= min(scap[p], dcap[j]) * z[j]  for all (p,j)
     F6  x >= 0, y >= 0, z binary
```

Now hand-check the candidate solution from section 6.3 against each family. The flows are:

| Variable | Value | Variable | Value |
|---|---:|---|---:|
| x[P1,D2] | 80 | y[D2,M1] | 10 |
| x[P2,D2] | 50 | y[D2,M2] | 45 |
| x[P3,D3] | 45 | y[D2,M3] | 75 |
| x[P3,D4] | 55 | y[D3,M1] | 45 |
| x[P1,D4] | 70 | y[D4,M4] | 30 |
| all other x | 0 | y[D4,M5] | 55 |
| z[D1] | 0 | y[D4,M6] | 40 |
| z[D2] | 1 | all other y | 0 |
| z[D3] | 1 | | |
| z[D4] | 1 | | |

**F1, demand.** M1: 10 plus 45 is 55, equal to demand. M2: 45. M3: 75. M4: 30. M5: 55. M6: 40.
All six satisfied as equalities. Total served 300, equal to total demand. Pass.

**F2, balance.** D1: 0 in, 0 out. D2: 80 plus 50 is 130 in, 10 plus 45 plus 75 is 130 out. D3: 45 in,
45 out. D4: 55 plus 70 is 125 in, 30 plus 55 plus 40 is 125 out. All zero. Pass.

**F3, plant capacity.** P1: 80 plus 70 is 150, against a capacity of 150. P2: 50 against 120. P3: 45
plus 55 is 100, against 100. Two plants are exactly at capacity, which is the observation from
section 6.3 about fragility. Pass, and note the binding rows.

**F4, DC capacity and link.** D1: 0 against 130 times 0, which is 0. D2: 130 against 130. D3: 45
against 140. D4: 125 against 140. Pass.

**F5, lane link.** The only tight case is x[P1,D4] at 70 against min(150, 140) times 1, which is 140.
Every other lane is well inside. Pass.

**F6, domain.** All flows non-negative, all z in {0,1}. Pass.

The objective, term by term, is fixed 4,650.0 plus inbound 524.5 plus outbound 419.0 plus production
2,395.0, which is 7,988.5. That agrees with section 6.3.

### 7.1 Three linkage errors this check catches

The hand check is not ceremonial. Each of the following is a real error that the check above
detects and a solver does not.

**Error 1 — the omitted F4.** Remove F4 and F5 and keep the same flows. The solver now has no reason
to pay for D1, so it reports the current flow pattern at a total of 2,395.0 plus 475.5 plus 350.0
plus 2,550.0 (D2 and D4 only), which is 5,770.5. The reported saving against the true current cost is
2,300.0, which is exactly D1's fixed cost. The error is not a subtle bias; it is the entire fixed
cost of the facility the model declined to open. The check that catches it is one line: every site
with positive throughput must have `z = 1`, and every site with `z = 0` must have zero throughput.

**Error 2 — F2 written with `<=`.** Inflow may exceed outflow. The model discards product at D2
rather than shipping it. Discarding 30 kcases at D2 avoids producing them at the marginal plant,
which is P1 at 8.00 per case, plus the inbound transport of 1.60 per case. The understatement is
30 times 9.60, or 288.0 thousand euros. The check that catches it is the balance row of the hand
check: 130 in against 100 out.

**Error 3 — an M taken from history.** Set M[P1,D4] to 25, the volume that lane carried last year,
and the candidate's 70 makes the model infeasible. The analyst reports that closing D1 is
impossible. Step 12 of section 5 catches it: M is derived from capacity, and the derived value here
is min(150, 140), or 140.

> **Note:** the three errors have three signatures — a cost too low by a round number, a cost too
> low by a traceable amount, and an infeasibility. Reading the signature shortens the diagnosis.

## 8. Data, standards, and regulatory context

A model's credibility rests on the identity of its objects and the provenance of its numbers.
Standards matter here for a narrow reason: they constrain what the identifiers and units are, so
that the model's entities join to the master data without a hand-built crosswalk.

| Standard or convention | Governs | What it constrains in the model | Grade |
|---|---|---|---|
| GS1 GLN | Location identity | Every plant, DC, and market node carries a stable identifier that joins to the site master | B |
| GS1 GTIN | Trade item identity | The product family in a single-commodity model is a defined aggregation of GTINs | B |
| ISO 8601 | Date and period representation | The period index `t` in a multi-period model, and the definition of a year | B |
| ISO 3166-1 | Country codes | Duty, tax, and emissions jurisdiction attributes on nodes | B |
| UN/LOCODE | Port and location codes | Inbound lanes that move through a port node | B |
| ISO 14083:2023 | Transport chain greenhouse gas quantification | The emissions factor per tonne-kilometre, if carbon is in the objective | B |
| Regulation (EU) 2023/956 (CBAM) | Carbon border adjustment for covered imports | The carbon cost of a covered import lane, definitive regime from 1 January 2026 | B |
| CSRD / ESRS E1 | Sustainability disclosure | The emissions boundary that the model's carbon term must match | B |

Three observations about this table.

**The identifiers are the easy part and the unit conventions are the hard part.** A GLN either
matches or it does not, and the failure is loud. A case-to-each conversion either matches or it does
not, and the failure is quiet: a model built on eaches with costs per case is wrong by the pack
factor, which is typically 6 to 24, and the error is inside the range where a reviewer assumes the
difference is a modelling choice.

**The bill of materials (BOM) is where a single-commodity model stops being enough.** The moment the
product family is defined by a BOM with shared components, two products compete for the same plant
capacity and the model needs a commodity index. Topic 05 adds it. Here the point is only that the
aggregation decision — what counts as one product — is a modelling choice with a data consequence,
and it should be recorded next to the parameter table.

**Model interchange is a de facto standard, not a regulated one.** Most modelling tools can write a
model in an LP or MPS text format, and archiving that file alongside the parameter tables is the
cheapest form of auditability available. It lets a reviewer read the rows without reading your code,
and it lets a future analyst re-solve the exact model that produced a number in a board paper. Treat
the exported file as part of the deliverable, not as a debugging artefact.

What these standards do **not** constrain is the part that decides the answer: the cost function,
the capacity figure, the grain of the product, and the treatment of demand. There is no standard for
"the right capacity of a distribution centre", and there will not be one, because it depends on the
operation.

## 9. Industry practice and case evidence

### 9.1 How the formulation reached practice

The facility-location problem is older than the computer. Alfred Weber's *Über den Standort der
Industrien*, published in 1909 and translated into English in 1929, posed the problem of locating a
production site to minimise transport cost, and Weber enlisted the mathematician Georg Pick for the
derivations behind what became the location triangle. The German location school — von Thünen in
1826, Weber in 1909, Lösch in 1940 — treated location as a continuous geometric question, and the
review literature is clear that the transition to the discrete, combinatorial form used in modern
practice came later, once the problem was posed over a finite set of candidate sites.

The discrete formulation that a modern solver reads is the one usually associated with Balinski's
1965 integer-programming formulation of the simple plant location problem, and the working
literature still calls the disaggregated LP form "the Balinski formulation" when contrasting it with
tighter or weaker variants. The capacitated version with a fixed charge per site and a linear
transport cost is the workhorse of the commercial network design tools that appeared from the 1970s
onward, and the multi-commodity, multi-period version was established in practice by the
distribution-system work of Geoffrion and Graves in 1974 and the global supply chain models that
followed in the 1990s.

> **Note:** the historical claims above are drawn from review articles and secondary sources found
> in the search for this topic (Barnes 2003; Bruno, Genovese and Improta 2014; Marianov and Eiselt
> 2024; Mehrez's review in *Geography Research Forum*). The primary papers of the 1960s were not
> retrieved directly, so the citation chain is reported at the level the reviews support and no
> specific claim is made about a paper that was not read.

### 9.2 What the formulation-strength literature reports

The claim that a tighter linking row improves the solve is not folklore; it is measurable, and the
measurements are consistent in direction.

| Formulation choice | Reported effect | Setting | Grade |
|---|---|---|---|
| Aggregated big-M instead of disaggregated per-lane rows in UFL | LP bound far weaker, integer solve roughly 2 to 4 times slower on a benchmark family (one reported pair: 14.15 s against 4.93 s) | Expert commentary with computed benchmark results | E |
| Big-M written on an expression rather than a declared variable bound, in a capacitated facility-location model | Root gap 32.5% with M equal to total demand, against 8.7% with M equal to the facility capacity; 7 branch-and-bound nodes against 1; identical optimum | Vendor tutorial with an exported LP | D |
| Big-M too small | Can remove feasible solutions; solver reports infeasible or returns a suboptimal answer | Teaching material and practitioner discussion | C |
| Large coefficient ranges in a mixed-integer model | Degrades LP solver performance and accuracy; remedies act on the model, not on tolerance settings | Preprint on exact mixed-integer solving | B |
| Fixed-charge network design relaxations | Typically weak and highly degenerate, which is why formulation and decomposition matter more than solver choice at this problem class | Working paper on exact algorithms for the multicommodity variant | B |

Read the grades before the numbers. The two quantitative rows in that table come from sources
graded D and E: a vendor tutorial and an expert blog, both reporting computations that are
reproducible in principle but neither peer-reviewed nor independently replicated here. They are
cited because the direction of the effect is corroborated by the peer-reviewed and preprint
literature, not because the specific percentages should be carried into a business case. What you
can say with confidence is that the choice of M changes the root gap by a factor of several, and
that the root gap predicts the search.

A note on a question that sounds paradoxical and is not. A practitioner will sometimes report that
tightening M made the model slower. The usual explanation is not that tightness hurts but that the
tightened constraint was invalid or interacted badly with a heuristic or a cut: an M that is too
small removes the optimal solution or deactivates a valid inequality, and the solver then works
harder to prove a worse answer. The rule is not "tighter is always faster"; it is "tightest valid is
the right target, and invalid is never tight".

### 9.3 What teams actually do

Baseline reconciliation is described as standard by every vendor and is skipped under time
pressure, because the current network "obviously" solves. The gap is often in the log and not in the
summary, which is how a time-limited run reaches a committee as an optimal one. M is set to a round
number because it is faster to write than to derive. Symmetry is audited rarely, because the sites
look different on the map. And dual values are reported as marginal values because the solver
provides them, which is the most common way a correct model produces an incorrect sentence: in a
degenerate LP the dual solution is not unique, and one valid answer is presented as the answer.

## 10. Failure modes and red flags

Each row states a failure, the red flag that is visible in a meeting or a dataset, the mechanism, the
consequence, and the mitigation.

| Failure | Red flag | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing linking constraint | Closed site shows throughput in the solution table | Binary is declared but no constraint reads it, or the capacity row omits `z` | Cost understated by the fixed cost of the phantom sites | Check every site with flow has `z = 1` |
| M taken from a historical volume | Model infeasible only in growth scenarios | M bounds the flow at last year's level | Infeasibility misread as a business constraint | Derive M from capacity and record the derivation |
| M taken from total demand | Root gap above 20% on a small instance | The switch is looser than the lane requires | Search cost multiplies; bound useless for sensitivity | Use the tightest valid bound per lane |
| Demand written as an inequality | Served quantity below the demand table | The model may decline unprofitable markets | Cost reduction that is a service reduction | Compare served quantity to demand market by market |
| Balance row omitted | Outbound flow exceeds inbound at some DC | Product is created at the cheapest node | Arbitrarily large understatement | Compute inflow minus outflow per DC |
| Unit mismatch between cost and flow | Objective off by a round factor such as 1,000 | Cost per thousand cases applied to cases | Nonsense answer that still reports optimal | Write the units under every term |
| Fixed cost charged per unit | No fixed cost line in the objective decomposition | Fixed and variable costs merged into one rate | The design question disappears; the answer is the pure transport optimum | Decompose the objective by cost type in the report |
| Symmetric candidate group | Solve time far above what the instance size suggests | k identical sites give k factorial equivalent solutions | Study stalls; analyst concludes the model is too big | Order interchangeable sites, or break the ties |
| Degenerate relaxation reported as insight | Dual values change sign between runs | Optimal basis is not unique | Unstable numbers presented as marginal values | Check stability across runs before reporting |
| Infeasibility attributed to the business | The model cannot reproduce a historical plan | A typo or a unit error | A real business finding is missed, or a false one is announced | Run the five-test triage, ending with historical replay |
| Baseline not reproduced | No reconstruction run in the pack | Validation replaced by "it solved" | Every scenario delta is a difference of two wrong numbers | Agree a tolerance, then run both modes |
| Gap unreported | "Optimal" in the deck, time limit in the log | Status field read as objective quality | Overstated certainty in a capital decision | Report status, bound, gap, and limit together |

Two of these survive review. The unit mismatch survives because it is plausible: an objective a
thousand times too large produces a network with no transport at all, which somebody notices, while
one that is 1.7 times too large produces a network that is slightly wrong, which nobody notices. The
unreported gap survives because the deck has one number and the log has four — status, objective,
bound, limit — and the certainty is manufactured by omitting three of them.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Costs are linear in flow | Keeps the objective linear, so a MILP rather than a nonlinear program | Volume discounts, step tariffs, minimum charges per drop | Plot cost against volume per lane from the freight audit |
| Flow is divisible | Continuous flow variables | Pallet, truckload, and container indivisibility at the decision grain | Check whether the solution's fractional flows are material at the lane level |
| Demand is known and deterministic | A single right-hand side per market | Forecast error, promotion, new product introduction | Scenario range; if the design flips inside the range, the deterministic answer is not an answer |
| Capacity is a single number per site | One row per DC | Product mix, temperature zones, labour shifts, congestion | Check the capacity figure against the site's own operating standard |
| Fixed cost is independent of volume | One binary per site | Step-fixed labour, modular capacity | Check whether the lease or labour contract steps at a volume threshold |
| The candidate set is complete | The model can only choose from listed sites | Sites the business would consider but did not list | Ask the strategy team to name sites absent from the list |
| All demand must be served | Equality in F1 | Contracts with penalty for non-delivery, or demand that can be refused | Compare the model's service requirement to the commercial terms |
| Parameters are stationary over the horizon | One number per parameter | Freight rate cycles, wage inflation, tariff change | Sensitivity on the parameters with the widest plausible range |

Two limits are worth stating plainly. The model answers the question it was given: no solver quality
compensates for an objective that omits inventory carrying cost or a constraint set that omits a
service promise. And the model cannot validate itself: every diagnostic in this topic compares it to
something outside it — a hand count, a derived bound, a historical cost, a planner's explanation.

## 12. Exercises

### Exercise 1 — Read a model back (recall, 30 minutes)

**Task.** Take the formulation in section 7 and produce the count table: sets, parameters by type,
variables by domain, constraints by family. Compare your counts to the stack diagram in section 3
and explain any difference.

**Deliverable:** a one-page count table with a row per element type.
**Acceptance criteria:** every count is derived from the index ranges rather than copied; the
constraint total of 17 is reproduced or the discrepancy is explained; the variable total of 40 is
reproduced.

### Exercise 2 — Break the model three ways (application, 45 minutes)

**Task.** Starting from the section 7 formulation, produce three modified versions: F1 written with
`<=`; F2 omitted; F4 written without `z`. For each, state what the model would report on the
section 6.3 flows and by how much the objective would fall.

**Deliverable:** three short paragraphs, each with a numeric estimate.
**Acceptance criteria:** the F4 case equals the fixed cost of the phantom site; the F2 case is
justified with the marginal production and inbound cost; the F1 case names which markets would be
dropped and why.

### Exercise 3 — Derive M (calculation, 30 minutes)

**Task.** For every lane in the instance, compute the tightest valid M from the parameters. Then
compute the fractional `z` each lane would permit in the LP relaxation if the candidate flows were
the only flow, and identify the three lanes where a loose M would do most damage.

**Deliverable:** a 12-row table with lane, derived M, candidate flow, and permitted fractional z.
**Acceptance criteria:** every M is derived from `min(scap, dcap)` or a tighter valid bound with a
stated reason; the ranking of damage is justified by the product of fixed cost and avoided
fraction.

### Exercise 4 — Compute an integrality gap (calculation, 25 minutes)

**Task.** For the single-facility subproblem in section 6.5, compute the gap at demand 55, 90, 120,
and 140 kcases. Plot the gap against the demand-to-capacity ratio and state the relationship in one
sentence.

**Deliverable:** a four-row table and one sentence.
**Acceptance criteria:** the arithmetic is shown for at least one row; the limiting case at full
capacity returns a gap of zero; the sentence states the relationship without hedging.

### Exercise 5 — Symmetry audit (critique, 40 minutes)

**Task.** You are given a candidate set of nine DCs. Six have distinct cost structures; three are
identical in every parameter. Compute the number of equivalent solutions contributed by the
identical group, write the ordering constraint that removes the duplication, and state one
condition under which the constraint would be invalid.

**Deliverable:** the count, the constraint, and the invalidity condition.
**Acceptance criteria:** the count is 3 factorial, or 6, stated with the reasoning; the constraint is
written in the model's own notation; the invalidity condition names a parameter that would differ.

### Exercise 6 — Infeasibility triage (critique, 45 minutes)

**Task.** A model is infeasible under the base case. Total demand is 300 kcases; total candidate
capacity is 540 kcases; one DC serves four of six markets; a service radius of 300 km is imposed on
outbound lanes. Work the five tests from section 5.7 and write the diagnosis.

**Deliverable:** a completed five-test table with a verdict.
**Acceptance criteria:** each test states its result; the service radius is tested as a hypothesis;
the verdict distinguishes a data defect from a business finding.

### Exercise 7 — Build the reconstruction run (design, 60 minutes)

**Task.** Specify both reconstruction modes for the section 6 instance: what is fixed, what is
optimised, what the expected result is, and what each of the four possible outcomes in the section
5.8 table would mean.

**Deliverable:** a two-column run specification plus the interpretation table.
**Acceptance criteria:** Mode A and Mode B are distinguished by what is fixed; the expected number
for Mode B is stated (8,070.5) with the tolerance you would accept; each outcome is mapped to a
next action.

### Exercise 8 — Write the model card (design, 45 minutes)

**Task.** Produce the one-page documentation standard for the section 6 model: sets, parameters
with source, variable count, constraint count, gap at termination, validation result.

**Deliverable:** a filled model card.
**Acceptance criteria:** every parameter has a named source; the counts match Exercise 1; the gap
field states a number and the status that produced it; the validation field states both modes and
their results.

## 13. Self-check answers

**Exercise 1 — counts.** Sets: 3 plants, 4 DCs, 6 markets, 12 plant-to-DC lanes, 24 DC-to-market
lanes. Parameters: 3 plant capacities, 3 production costs, 4 fixed costs, 4 DC capacities, 6
demands, 12 inbound rates, 24 outbound rates, which is 56. Variables: 12 `x`, 24 `y`, 4 `z`, which is
40, of which 36 are continuous and 4 binary. Constraints: F1 has one row per market, 6; F2 has one
per DC, 4; F3 has one per plant, 3; F4 has one per DC, 4; total 17. F5 is optional and would add 12,
making 29. Any count that differs is usually a fan-out error: a row written over `P x J` when it was
meant over `J`.

**Exercise 2 — the three breakages.** With F1 as `<=`, the model drops any market whose served cost
exceeds the value it assigns to serving, so it drops M1 first (55 kcases, served from D1 at 1.00 plus
inbound, the cheapest in the network) only if dropping is rewarded, which happens when a fixed cost
depends on the market. In the section 6 instance the fixed cost is at the DC, not the market, so the
`<=` variant drops nothing and the error is latent; this is worth knowing, because it means the
error can sit in a model for years and appear only when a market-level fixed cost is added. With F2
omitted, the model creates product at the cheapest node and the objective falls by the avoided
production and inbound cost, unbounded in principle. With F4 lacking `z`, the objective falls by
2,300.0, D1's fixed cost.

**Exercise 3 — derived M.** For every lane, `M = min(scap[p], dcap[j])`. The values are 130 for
every lane into D1 and D2 (plant capacities 150, 120, 100 against DC capacities 130, 130), and 140
capped by plant capacity for D3 and D4 (P1 150, P2 120, P3 100 against 140). The lanes where a loose
M does most damage are those into the site with the largest fixed cost relative to its load: D3 at
2,100 fixed and 45 loaded, D1 at 2,300 and 0, and D4 at 1,100 and 125. The first is the live case; the
second is irrelevant because the site is shut; the third is nearly tight because capacity and load
are close.

**Exercise 4 — gap against load.** The gap is `1 - demand/capacity` when the fixed cost dominates:
at 55 kcases, 1 minus 0.393 is 0.607 of the fixed-cost component, which produces the 57.8% total gap
computed in section 6.5. At 90: `z = 0.643`, LP cost is 707.1 plus 90, which is 797.1; integer cost is
1,190; gap is 33.0%. At 120: `z = 0.857`, LP is 942.9 plus 120, which is 1,062.9; integer 1,220; gap
14.5%. At 140: `z = 1`, gap 0. The relationship: the gap falls monotonically as the facility fills,
and reaches zero exactly when the relaxation cannot avoid opening it.

**Exercise 5 — symmetry.** Three identical sites give 3 factorial, which is 6, equivalent solutions
for every design that treats them the same way. The constraint is `z[a] >= z[b] >= z[c]` over the
three, which selects the representative in which the alphabetically first site is used first. It is
invalid if any parameter differs within the group: a different fixed cost, a different capacity, a
different distance to any market, or a different lease start date. In that case the sites are not
interchangeable and the constraint removes distinct designs.

**Exercise 6 — triage.** Test 1 passes: 300 against 540. Test 2 is where the problem is: if the
300 km radius excludes the only DC that can reach a market, that market is unservable. Test 3 passes
if the totals agree with the demand and capacity tables. Test 4 identifies F1 as the conflicting
family once the radius rows are relaxed. Test 5, historical replay, is decisive: if the current
network served that market last year, the radius is wrong, because the business demonstrably reached
it. The verdict is a data or modelling defect — a radius drawn on a map rather than a drive time —
not a business finding.

**Exercise 7 — reconstruction.** Mode A fixes `z` to the historical footprint (D1, D2, D4 open, D3
closed) and optimises flows; expected result is at or slightly below 8,070.5. Mode B fixes `z` and
the flows to the historical values and evaluates; expected result is 8,070.5 to the precision of the
cost data, and a tolerance of 2% is reasonable when the freight ledger reconciles, 5% when it does
not. If Mode B misses by more than the tolerance, stop and fix the data. If Mode A is far below Mode
B, open the largest flow change and ask the planner who owns that lane.

**Exercise 8 — model card.** The card has five blocks. Sets and counts. Parameters with source, one
row each, 56 rows. Variables by domain, 40. Constraints by family, 17. Termination: status,
objective, bound, relative gap, limit. Validation: Mode B result and tolerance, Mode A result, and
the interpretation. A card without the termination block is not a card.

## 14. Interview questions

**1. "Walk me through the anatomy of a network MILP."**

Strong answer: sets, parameters, variables by domain, objective, then the four constraint families —
demand, balance, capacity, linking — and the counts for a concrete instance. It mentions that the
linking constraint is what gives the binary its meaning. Weak answer: lists "objective and
constraints" and cannot say how many of either.

**2. "Why does a fixed cost make the problem an integer program?"**

Strong answer: a fixed charge is a jump at zero, the jump makes the cost function non-convex, and a
linear program cannot represent a non-convex feasible set; the binary linearises the jump by
creating two regimes and a switch. It notes that the flow subproblem alone is an LP. Weak answer:
says "because fixed costs are discrete", which is not the reason.

**3. "What is big-M and how do you choose it?"**

Strong answer: M is the largest flow the binary permits, derived from capacity and demand, not
chosen for convenience; too small removes feasible solutions, too large weakens the relaxation; the
tightest valid value is `min` of the relevant capacities. It mentions recording the derivation.
Weak answer: says "a large number that makes the constraint work", or quotes a round number.

**4. "Why does the integrality gap matter more than the solve time?"**

Strong answer: the gap is a property of the formulation, available from the root relaxation in
seconds, and it predicts the remaining search; two models of the same size can have gaps differing
by a factor of ten, and the one with the larger gap may never close. Time is a symptom, gap is the
cause. Weak answer: says "a smaller gap is better" without saying why it is diagnostic.

**5. "How do you detect that a model is creating product?"**

Strong answer: compute inflow minus outflow at every transshipment node; any non-zero value is a
bug in the balance family, not a rounding issue. It notes that the objective will be lower and the
solution will look plausible. Weak answer: says "check the objective", which does not reveal the
error.

**6. "What is symmetry in a facility-location model and why does it hurt?"**

Strong answer: identical candidate sites produce identical solutions under permutation, so a group
of k contributes k factorial equivalent designs; the solver explores all of them. It notes that
ordering constraints remove the duplication and that they are valid only when the sites are truly
interchangeable. Weak answer: says "the model has many solutions", which is true of every
underdetermined model and does not identify the mechanism.

**7. "Your model is infeasible. What do you do?"**

Strong answer: run a triage — aggregate feasibility, per-market feasibility, unit sanity, relax one
family at a time, and finally force the historical plan. If a plan that happened cannot be
represented, the model is wrong, not the business. Weak answer: relax constraints until it solves
and report the relaxed answer.

**8. "How do you validate a network model?"**

Strong answer: reconstruction in two modes, fixed footprint with optimised flows and fixed
footprint with fixed flows, against a tolerance agreed in advance; then interpret the difference.
It notes that Mode A much cheaper than Mode B is either the finding or the bug. Weak answer: says
"the model solved to optimality", which validates the solver and not the model.

**9. "What is degeneracy and what does it do to your reporting?"**

Strong answer: many constraints active, non-unique basis, therefore non-unique duals; the solver
returns one valid dual solution and it may not be stable across runs, so dual values should not be
presented as marginal values without a stability check. Weak answer: confuses degeneracy with
infeasibility or with numerical error.

**10. "How do you keep units consistent across a model with fixed cost, transport rate, and carbon?"**

Strong answer: choose one currency and one period for the objective; write the unit under every term
and cancel; convert carbon to the same currency using a stated price and a stated factor; and check
the spread of coefficient magnitudes, because a range above about six orders of magnitude starts to
cost numerical accuracy. Weak answer: says "I am careful with units", which is not a procedure.

## 15. Key terms

- **MILP** — mixed-integer linear program; linear objective and constraints, some variables integer.
- **LP relaxation** — the model with integrality removed; supplies the bound.
- **Bound** — the best objective the relaxation proves; a floor for a minimisation.
- **Incumbent** — the best feasible integer solution found so far.
- **Integrality gap** — relative distance between incumbent and bound.
- **Root gap** — the same ratio at the root relaxation, before branching.
- **Index set** — a declared collection the model ranges over.
- **Parameter** — a number given to the model, with a unit and a source.
- **Decision variable** — a quantity the model chooses.
- **Binary variable** — a 0 or 1 switch, meaningful only if a constraint reads it.
- **Fixed charge** — a cost incurred if a facility or lane is used at all.
- **Facility location** — choosing which facilities to open and which demand each serves.
- **UFL / CFL** — uncapacitated and capacitated facility location.
- **Demand satisfaction** — the family requiring each market's demand to be met.
- **Flow conservation** — the family requiring inflow to equal outflow at a transshipment node.
- **Capacity constraint** — the family bounding throughput at a node or lane.
- **Linking constraint** — the family tying flow to the binary that permits it.
- **Big-M** — the constant in a linking constraint; the largest flow the binary permits.
- **Non-convexity** — a feasible set or objective where local and global optima differ.
- **Symmetry** — permuting identical sites maps the model onto itself.
- **Symmetry breaking** — constraints that remove equivalent solutions.
- **Degeneracy** — more active constraints than the solution needs; the basis is not unique.
- **Coefficient scaling** — the spread of magnitudes among coefficients in a row or objective.
- **Model validation** — showing the model reproduces a known historical outcome.
- **Reconstruction run** — validation with the historical footprint forced.

## 16. Sources

**Peer-reviewed and academic** (all grade A)

- Marianov, V., and Eiselt, H. A. (2024). "Fifty Years of Location Theory — A Selective Review." *European Journal of Operational Research*. DOI: 10.1016/j.ejor.2024.01.036. Location theory's history.
- Barnes, T. J. (2003). "The Place of Locational Analysis: A Selective and Interpretive History." *Progress in Human Geography*, 27(1), 69–95. Weber 1909, the 1929 translation, the German location school.
- Bruno, G., Genovese, A., and Improta, G. (2014). "A Historical Perspective on Location Problems." *British Journal for the History of Mathematics*, 29(2), 83–97. Pre-history; Weber and Georg Pick.
- Ahrens, J. H., and Finke, G. (1975). "Degeneracy in Fixed Cost Transportation Problems." *Mathematical Programming*. DOI: 10.1007/BF01580452. Degeneracy in fixed-charge transport.
- Gauthier, J. B., Desrosiers, J., and Lübbecke, M. E. (2016). "Tools for Primal Degenerate Linear Programs: IPS, DCA, and PE." *EURO Journal on Transportation and Logistics*. Preprint at publications.polymtl.ca. Consequences of primal degeneracy.
- Jans, R. (2008). "Solving Lot-Sizing Problems on Parallel Identical Machines Using Symmetry-Breaking Constraints." *INFORMS Journal on Computing*. DOI: 10.1287/ijoc.1080.0283. Structural analogue for identical facilities; symmetry breaking can hurt easy instances.
- Geoffrion, A. M., and Graves, G. W. (1974). "Multicommodity Distribution System Design by Benders Decomposition." *Management Science*, 20(5). The multi-commodity formulation entering practice.
- Camm, J. D., Chorman, T. E., Dill, F. A., Evans, J. R., Sweeney, D. J., and Wegryn, G. W. (1997). "Blending OR/MS, Judgment, and GIS: Restructuring P&G's Supply Chain." *Interfaces*, 27(1). Reconciliation discipline.
- Arntzen, B. C., Brown, G. G., Harrison, T. P., and Trafton, L. L. (1995). "Global Supply Chain Management at Digital Equipment Corporation." *Interfaces*, 25(1). Multi-commodity, multi-period structure.

**Working papers and preprints** (all grade B)

- Dokka, T., Letchford, A. N., and Mansoor, H. (2019). "Anomalous Behaviour of Dual-Based Heuristics." Optimization Online. Massive primal degeneracy in the facility-location relaxation; simplex and barrier return different duals.
- van Doornmalen, J., and Hojny, C. (2022). "A Unified Framework for Symmetry Handling." Optimization Online. Mechanism of symmetry handling in branch and bound.
- Walsh, T. (2012). "Symmetry Breaking Constraints: Recent Results." arXiv:1204.3348. Taxonomy of symmetry-breaking approaches.
- Eifler, L., and Gleixner, A. (2023). "Safe and Verified Gomory Mixed Integer Cuts in a Rational MIP Framework." arXiv:2303.12365. Large coefficient ranges degrade solver performance and accuracy.
- Zetina, C. A., Contreras, I., and Cordeau, J.-F. (2017). "Exact Algorithms for the Multicommodity Uncapacitated Fixed-Charge Network Design Problem." CIRRELT-2017-69. Fixed-charge relaxations are weak and highly degenerate.
- Karamanov, M., and Cornuéjols, G. "Branching on General Disjunctions." Carnegie Mellon working paper. Definition of gap closed at the root.

**Teaching material and vendor documentation** (grade B to D)

- Belov, G. "Big-M Constants Come from Declared Bounds." AMPL Colab notebook. Grade D. The 32.5% against 8.7% root-gap comparison, 7 nodes against 1, identical optimum.
- AMPL. "Supply Chain Network." Colab notebook. Grade D. Cross-check on the standard constraint structure.
- Gurobi. "Mixed Integer Linear Programming Tutorial" and "Modeling 2" slides, Frankfurt 2017. Grade D. Warning on overlarge M and numerical trickle flow.
- Ralphs, T. "Computational Integer Programming," Lecture 9, Lehigh University. Grade C. Duality gap as a difficulty measure.
- MOSEK. "The Optimizer for Mixed-Integer Problems," section 8.4. Grade B for definitions, D for framing. Relative-gap definition; implementations differ in the denominator.
- Kitchin group, Carnegie Mellon. "Mixed-Integer Linear Programming (MILP)" tutorial, discopt. Grade C. `x_ij <= y_i` is already the tightest M in UFL.

**Practitioner commentary** (grade E)

- Buchanan, A. (2017). "I Dislike Big M's and I Cannot Lie." Farkas' Dilemma blog. Direction of the aggregated versus disaggregated effect; reported 2 to 4 times integer solve time. Specific LP-bound figures are not quoted because their orientation could not be verified.
- Nicholas, T. "On Network Modeling: Blaspheming the Baseline." Opex Analytics. Matching baseline cost is insufficient validation.

**Vendor guidance on validation** (all grade D)

- AIMMS. "Getting Started with Scenario Modeling in Supply Chain Network Design." The 5% to 10% of actual costs and volumes tolerance.
- ReliaSim. "Validate a Supply Chain Model: Reproduce History First." Two-mode reconstruction; tolerances agreed in advance.
- OpsFactor. "Run and Compare Results," distribution network design template. Reconcile P&L components separately.

**Standards and regulation** (all grade B)

- ISO 14083:2023, *Greenhouse gases — Quantification and reporting of greenhouse gas emissions arising from transport chain operations*.
- Regulation (EU) 2023/956 establishing a carbon border adjustment mechanism; definitive regime from 1 January 2026.
- GS1 General Specifications, including GLN and GTIN.
- ISO 8601 for date and time representation; ISO 3166-1 for country codes; UN/LOCODE for port and location codes.
- Directive (EU) 2022/2464 (CSRD) and ESRS E1. Boundary matching only; consult the consolidated text before relying on any implementation date.

> **Note:** three groups here are graded below A. The quantitative big-M comparisons and the
> validation tolerance come from vendor and practitioner sources, are reported as those sources
> state them, and are not independently verified. They are included because the direction of each
> effect is corroborated by the peer-reviewed and preprint literature.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Previous topic, the network as a graph and cost-to-serve: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- First topic, the decision ladder: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- Next topic, the baseline capacitated facility-location model: `04_facility-location.md` (planned)
- Then multi-commodity and multi-period models: `05_multi-commodity-period.md` (planned)
- Later, solvers and heuristics: `10_solvers-heuristics.md` (planned)
- This part's index: [`../_index.md`](../_index.md)
- Module 10, inventory and multi-echelon optimisation: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Part I of the program: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- This module's survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)
