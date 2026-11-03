# The Baseline Capacitated Facility-Location Model — fixed charge, capacity, and demand allocation

> Module 11 — Network Design and Optimization
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 topics 02 and 03
> Version 2026.5

---

## 1. Why this topic exists

Almost every network study reduces to one model. Strip away the carbon term, the multi-period
opening schedule, the stochastic scenarios, and the service penalties, and what remains is a
list of candidate sites, a list of markets, a capacity for each site, a cost to serve each
market from each site, and a fixed cost to open each site. Choose the sites. Assign the demand.
Pay the fixed cost and the flow cost. That is the capacitated facility-location problem (CFLP),
and it is the object that the rest of this module extends.

The model is old, well understood, and easy to write. That combination is why it is so often
written badly. Three failures account for a large share of the network studies that produce a
confident answer nobody acts on.

**The candidate list was inherited.** The model can only choose among the sites someone put on
the list, and in most studies the list is assembled in a workshop from sites people already
know, plus perhaps one greenfield location suggested by a map. The optimisation then does
exactly what it was asked: it finds the best subset of a set that already encodes the
organisation's assumptions about geography. A study that evaluates 12 familiar sites and
reports a 9% saving has measured the value of choosing among 12 familiar sites, not the value
of redesigning the network. Section 5 treats candidate generation as a step in its own right.

**The capacity number was nameplate.** A site is entered with the throughput its design permits,
not the throughput it achieves. The model then packs demand into it at 100% utilisation, which
no warehouse or plant sustains. The recommended design looks feasible on paper and fails in the
peak week. Capacity is not one number; it is at least three, and the choice changes the open set.

**The fixed cost was the cost centre total.** The site's reported fixed cost includes allocated
corporate overhead that does not disappear when the site closes. Using it overstates the saving
from consolidation by 30 to 50% in the cases this course uses, and makes the model prefer
closure for reasons that are accounting artefacts. Topic 01 named this failure; this topic gives
it a coefficient and a test.

A fourth, subtler failure belongs to the model rather than its inputs. The model is a
mixed-integer linear programme (MILP), and its linear programming (LP) relaxation is a different
problem with a different answer. The relaxation will happily serve one market from three
facilities at 60%, 30%, and 10% of its demand, paying 60% of each facility's fixed cost. That
answer is not a worse design; it is not a design at all. Understanding why the relaxation
splits, and what the resulting gap tells you, separates using the model from being used by it.

**What this topic is not.** Module 11 topic 03 owns the anatomy of a MILP: how variables,
constraints, objectives, and bounds fit together, and why the solver behaves as it does. This
topic does not re-teach that. It applies the anatomy to the one formulation a practitioner will
write most often. Topic 05 adds commodities and periods, topic 06 adds legacy assets and phased
transition, topic 07 adds scenarios and sensitivity, topic 08 adds uncertainty, and topic 09 adds
carbon; each is this model with more indices. Module 12 owns transport cost mechanics.

**Why the baseline matters even when it is not the final model.** A multi-commodity,
multi-period, stochastic model with carbon pricing is impressive and hard to debug. When it
produces a strange answer, the diagnostic is almost always to strip it back to the baseline and
see whether the strangeness survives. If it does, the cause is in the data or the candidate set;
if not, it is in the extension. Practitioners who cannot write the baseline from memory cannot
debug the extension, and end up trusting a black box they cannot interrogate.

**The honest claim for the model.** A CFLP does not find the best network. It finds the
least-cost way to serve a stated demand from a stated candidate set at stated costs, subject to
stated capacities. Each "stated" does more work than the optimisation. The model's value is that
it makes the four statements explicit, prices the trade-offs between them, and produces a
defensible comparison. That is worth a great deal, but it is not the same as an answer.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the CFLP in words, before any algebra, naming the choice, the assignment, the
   capacity restriction, and the two cost components.
2. **Write** the full formulation — objective, demand satisfaction, capacity and linking
   constraints, variable domains — and **explain** what each constraint does, including what
   goes wrong when the linking constraint is omitted.
3. **Explain** why the LP relaxation produces fractional open variables and split allocations,
   **construct** a small instance where the relaxation is strictly cheaper than any integer
   solution, and **interpret** the gap as a bound rather than an error.
4. **Distinguish** nameplate, effective, and demonstrated capacity, and **choose** a basis for a
   stated planning horizon with a written justification.
5. **Build** a candidate-site list deliberately, **document** what was excluded and why, and
   **estimate** the cost of the exclusion rather than assuming it away.
6. **Assemble** the four cost and demand inputs with units, price year, and aggregation basis,
   and **carry** the aggregation error inherited from topic 02 into the interpretation.
7. **Read** a solution as an open set, an allocation matrix, a cost breakdown, and a set of
   binding constraints, and **name** the two questions the output cannot answer.
8. **Compute** the break-even threshold at which a facility enters or leaves the open set, and
   **present** an option set rather than a single recommendation, to the standard in section 5.

Each outcome maps to an exercise in section 12 and to a self-check answer in section 13.

## 3. The core mental model

Hold one picture: two columns of nodes. On the left, four markets with demand. On the right,
three candidate sites with a capacity written beside each and a fixed cost printed underneath.
Every market can be served by every site, and each arc carries a unit cost. Choose a subset of
the right-hand column to switch on, then push all the demand across the arcs without exceeding
any capacity.

```text
       MARKETS (demand, thousand units/yr)        CANDIDATE SITES
                                                   (capacity, fixed cost)

         M1  30 o---4---\        +----------------+
                 \--7-----\-----> | A  cap 60      |
                  \-9-------\     |    f 1,200     |
         M2  25 o---6--------\--> +----------------+
                 \--4----------\
                  \-7-----------\  +----------------+
         M3  20 o---8-----------\>| B  cap 50      |
                 \--6------------\ |    f   900     |
                  \-3------------->+----------------+
         M4  15 o---9-----------/  +----------------+
                 \--7----------/   | C  cap 40      |
                  \-5---------/    |    f   600     |
                                   +----------------+

   arcs labelled with c_ij, the unit cost of serving market i from site j
   (transport plus variable handling, currency units per unit of demand)
   one arc per market-site pair is drawn; the full instance has 4 x 3 = 12

   THE LP RELAXATION OF THE SAME PICTURE, ONE MARKET

         M2  25 o--- 0.6 x 25 ---> [ A ]  x_A = 0.6
                 \-- 0.4 x 25 ---> [ B ]  x_B = 0.6

   the relaxation pays 0.6 f_A + 0.6 f_B and splits one market's demand
   the integer model must pay f_A or f_B in full, and ship 25 or 0
```

Four properties follow, and they carry most of the topic.

**Property 1 — The fixed charge is a step, and the step is the whole problem.** Flow cost is
linear: twice the volume, twice the cost. The fixed charge is not: a facility handling one unit
costs the same to open as one handling its full capacity. That single non-linearity turns a
transportation problem, solvable in polynomial time with an integral LP relaxation, into a
mixed-integer programme that is NP-hard. Everything difficult about network design follows.

**Property 2 — Capacity couples markets that have nothing to do with each other.** Without
capacity, each market independently picks its cheapest open site and the problem decomposes.
With capacity, a large market can crowd a small one out of a cheap site, so one market's
assignment depends on all the others. This coupling is what makes capacity worth modelling and
what makes the answer sensitive to forecasts in a way per-market analysis cannot see.

**Property 3 — Integrality is the only reason the answer is physical.** The demand and capacity
constraints are linear and, alone, describe a good flow problem. What makes the output a network
rather than a cost allocation is the requirement that each open variable be 0 or 1. Drop it and
"open 60% of site A" becomes a legal answer — not nonsense, a lower bound, but not a design.

**Property 4 — The model compares; it does not invent.** The output is the best member of the
set of designs the inputs permit. If the true best design uses a site that is not on the list,
no amount of solver time will find it, which is why candidate generation is an analytical step
rather than an administrative one.

### 3.1 Why the relaxation splits, in one paragraph and one number

Take two markets, each with demand 60 thousand units per year, and two candidate sites, A and
B, each with capacity 100 thousand units per year and a fixed cost of 100 thousand currency
units per year. Unit flow costs are zero, so the only question is how many sites to open. Total
demand is 120 and one site holds 100, so every integer solution opens both and pays 200. The LP
relaxation may set `x_A = 0.6` and `x_B = 0.6`, supplying exactly 120 units of capacity for a
fixed cost of 120, and split each market 36 thousand units to A and 24 thousand to B. It is 40%
cheaper. The 0.6 is not a recommendation to build 60% of a warehouse; it is the price of capacity
where capacity can be bought in fractions, and 120 against 200 is the price of indivisibility.

The general statement is stronger than this example suggests. For the natural formulation of
the capacitated problem the integrality gap is unbounded: An, Singh, and Svensson give an
instance with `n + 1` clients, two facilities of capacity `n`, and opening costs of 0 and 1,
where the relaxation costs `1/n` and every integer solution costs 1. The gap grows with `n`.
That is not a reason to distrust the model, but a reason to report the gap and treat the LP
bound as a bound.

### 3.2 Where the model came from

The lineage is worth knowing because each step added exactly one idea, and the ideas are still
the ones practitioners argue about.

**Weber, 1909.** Weber's *Über den Standort der Industrien* posed the continuous single-facility
problem: locate one plant to minimise weighted distance to raw material sources and the market.
The weights are tonnage. There is no fixed charge, no capacity, and no discrete choice, so it
answers "where in this region", not "which of these sites".

**Hakimi, 1964 and 1965.** Hakimi showed that an optimal median on a network sits at a vertex,
converting a continuous problem into a discrete one. The p-median problem followed: choose
exactly `p` facilities to minimise demand-weighted distance. It fixes the number of facilities
exogenously and drops the fixed charge, which makes it a different question, and the two answers
are often far apart. Reese's annotated bibliography is the standard guide.

**Balinski, 1965.** The fixed-charge formulation used here is attributed to Balinski's "Integer
Programming: Methods, Uses, Computation" in *Management Science*, which put the fixed cost and
the capacity into one integer programme — the object in section 5.

**Erlenkotter, 1978, and the solution tradition.** Erlenkotter's dual-ascent procedure made the
uncapacitated problem solvable at scale before general-purpose MIP solvers existed. The
capacitated problem attracted Lagrangian relaxation, surrogate relaxation, and later cutting
planes. For a 2026 practitioner the practical consequence is small: a general-purpose solver
handles a few thousand candidates. The intellectual consequence is larger — the structure that
made those methods work, the fixed charge as the only non-linearity, is what to look for when a
model is slow.

**The p-median detour, and why it persists.** The p-median is taught first because it is easier
and because public-sector siting often has a fixed budget. Madleňák and colleagues, on a 110-node
Italian postal network in 2016, report 11 centres under the p-median and 13 facilities under the
uncapacitated fixed-charge model, at partly different nodes. Neither is wrong: the p-median was
told to pick 11, the fixed-charge model was told to pay for what it used and chose 13. If you
cannot say which instruction the business gave, you are not ready to choose a formulation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Capacitated facility location problem (CFLP) | Facility location with a finite throughput at each open facility | Used for the uncapacitated case, which behaves differently | Model class |
| Uncapacitated facility location problem (UFLP) | Facility location with no capacity limit on an open facility | Treated as a simplification with no consequence | Model class |
| p-median | Choose exactly `p` facilities to minimise demand-weighted distance, with no fixed charge | Treated as equivalent to the fixed-charge model | Model class |
| Fixed charge | A cost incurred once if a facility is used at all, independent of volume | Treated as a cost per unit at low volume | Currency per period |
| Fixed operating cost `f_j` | The avoidable annual cost of keeping site `j` open | Taken from the cost centre total, including allocations | Currency per year |
| Variable flow cost `c_ij` | Cost of moving one unit from site `j` to market `i`, including transport and variable handling | Reported as freight only | Currency per unit |
| Market (demand node) `i` | An aggregation of customers treated as one demand point | Confused with a customer or a postcode | Node identifier |
| Candidate site `j` | A location the model is permitted to open | Confused with an existing site | Node identifier |
| Candidate set | The complete list of sites offered to the model | Assembled by whoever attends the workshop | Set of node identifiers |
| Open set | The subset of candidates the solution selects | Reported as "the answer" without the allocation | Subset of `J` |
| Allocation matrix | The quantity served to each market from each site | Confused with a routing plan | Units per period, per pair |
| Capacity `cap_j` | The maximum throughput a site can handle under the stated operating assumptions | Entered as nameplate, then used as if achievable | Units or tonnes per period |
| Nameplate capacity | The design or rated maximum under ideal conditions | Quoted as the operating limit | Units per period |
| Effective capacity | Nameplate adjusted for availability, planned downtime, changeover, and product mix | Treated as a synonym for nameplate | Units per period |
| Demonstrated capacity | The throughput actually achieved and sustained in operation | Ignored because it is lower than the design figure | Units per period |
| Utilisation | Throughput divided by the chosen capacity basis | Reported without stating the denominator | Percent |
| Utilisation ceiling | The utilisation above which congestion makes the site unreliable | Treated as 100% in the model | Percent |
| Peak-to-average ratio | Peak-period demand divided by average-period demand | Omitted when the model runs on annual averages | Ratio |
| Big-M | A large constant that switches a constraint on or off with a binary variable | Set larger than necessary, weakening the relaxation | Same unit as the constrained quantity |
| Linking constraint | The constraint that ties flow to the open variable, usually `sum_i y_ij <= cap_j x_j` | Omitted, allowing flow through closed sites | Mathematical expression |
| Demand satisfaction constraint | The constraint that every market's demand is met in full | Relaxed without a penalty, producing unserved demand | Mathematical expression |
| Binary variable `x_j` | A variable equal to 1 if site `j` is open and 0 otherwise | Declared continuous "to speed it up" | 0 or 1 |
| Flow variable `y_ij` | The quantity shipped from site `j` to market `i` | Declared integer, which is unnecessary and slow | Units per period |
| LP relaxation | The same model with the binary variables allowed to take any value between 0 and 1 | Treated as a solution rather than a bound | Model |
| Fractional solution | An LP solution in which some open variables lie strictly between 0 and 1 | Presented as a design with "partial" facilities | 0 to 1 |
| Split allocation | Serving one market's demand from more than one facility | Assumed impossible when it is merely expensive | Units per market |
| Integrality gap | The worst-case ratio of the integer optimum to the LP optimum for a formulation | Confused with the gap reported by a solver on one instance | Ratio |
| MIP gap (mixed-integer programme) | The difference between the best integer solution found and the solver's bound, on one instance | Reported without the bound or the time limit | Percent of incumbent |
| Lower bound | A value no feasible design can beat | Treated as an achievable cost | Currency per period |
| Single-sourcing | The requirement that each market be served by exactly one facility | Imposed without checking that the business needs it | Constraint |
| Service window | The maximum elapsed time from order to delivery promised to a market | Encoded as a radius in kilometres | Hours |
| Cost basis | The statement of which costs are included, at what price year, in what currency | Left implicit, so two designs are costed differently | Documented convention |
| Price year | The year whose price level the costs represent | Mixed across inputs without deflation | Calendar year |
| Avoidable fixed cost | A cost that disappears if the site closes | Confused with total site fixed cost | Currency per year |
| Allocated overhead | Corporate or shared cost assigned to a site by an accounting rule | Counted as a saving in a closure case | Currency per year |
| Step-fixed cost | A cost that is fixed within a volume band and jumps at a threshold | Modelled as purely variable or purely fixed | Currency per period, per band |
| Shadow price | The marginal value of relaxing a constraint by one unit | Read from a MIP as if it were an LP dual | Currency per unit |
| Break-even threshold | The value of a parameter at which two designs cost the same | Confused with the accounting break-even | Units, currency, or percent |
| Option set | Three to five distinct designs costed on a common basis | Replaced by one recommendation and a sensitivity table | Set of designs |
| Baseline validation | Running the model on the current network and reconciling to actual cost | Skipped because "the model is for the future" | Currency and percent variance |

Five of these carry more weight than the rest.

**Fixed charge versus fixed operating cost.** The fixed charge is the modelling device; the
fixed operating cost is the number you put into it. They are not the same object, and the
distinction matters because the number is where most of the error enters. A fixed charge of
1,200 thousand currency units per year is a statement that opening site A commits the company
to 1,200 thousand per year of cost that it would not otherwise incur. If 400 of that is
allocated corporate overhead, the true fixed charge is 800, and the model will over-value
closing A by a third.

**Capacity, three ways.** Nameplate is what the building and equipment are rated for. Effective
is nameplate after availability, planned downtime, changeover, and mix. Demonstrated is what
the site has actually sustained. A model fed nameplate will recommend a design that cannot run;
a model fed demonstrated will recommend more sites than the design needs, because it bakes in
today's operating inefficiency. The defensible choice is effective capacity at the planning
horizon, with the derivation written down, and a sensitivity run at demonstrated capacity.

**LP relaxation versus relaxation of the problem.** Relaxing integrality gives a lower bound on
the cost of the best design within the candidate set. It says nothing about designs outside the
set. Practitioners sometimes report the relaxation as "the theoretical minimum", which
overstates it: it is the minimum of a mathematical object, not of the business problem.

**MIP gap versus integrality gap.** The integrality gap is a property of the formulation and is
worst-case. The MIP gap is a property of one solver run: the difference between the best
solution found and the solver's own bound, usually expressed as a percentage of the incumbent.
A solver that stops at a 1% gap has proved that no design is more than 1% cheaper than the one
it returned, within the formulation. It has not proved that the formulation is tight.

**Open set versus decision.** The open set is a set of site identifiers. The decision is a
sequence of leases, closures, hires, transfers, and customer conversations. Between them sit
stability and executability, and neither is in the model.

## 5. The mechanics, step by step

This section is the procedure. It takes the sentence "we should look at our network" to a
reported result that a capital committee can act on. Fourteen steps, each with its input, its
output, and the branch that most often sends the work back.

### Step 1 — Fix the decision, the product scope, and the period

**Input and output:** the decision sentence from topic 01 becomes one sentence naming the sites
in scope, the markets in scope, the product family, the period, and the objective.

**Why it matters:** the CFLP answers a narrow question, and a drifting scope produces a model
that mixes periods and product families with different economics. "Which of these nine candidate
distribution centres (DCs) should serve our ambient grocery volume in 2030 at minimum annual
cost" is answerable; "What should our network look like" is not.

**Branch:** if the product family contains items with different handling requirements — frozen,
chilled, ambient, hazardous — you have a multi-commodity problem, and topic 05 is the right
tool. Do not average the handling cost and proceed.

### Step 2 — Build the candidate set deliberately

**Input and output:** the existing footprint, the demand geography, the service promise, and
land and labour constraints become a documented candidate list with a source and a reason for
every inclusion and exclusion.

**Why it matters:** this is the most consequential input in the model and the least examined.
Four rules make it defensible.

1. **Start from the existing sites.** Keeping them is always an option; excluding one is a
   decision to close it and should be labelled as one.
2. **Generate greenfield candidates from the demand surface.** A coarse centre-of-gravity over
   demand, a grid or freight clusters, and a land and labour screen will produce candidates
   nobody in the room thought of. Topic 02 supplies the surface.
3. **Include the awkward site.** If the model rejects it, you have evidence; if you exclude it,
   you have an opinion.
4. **Record why each exclusion happened.** "No land" and "we did not think of it" are different
   reasons, and only one is a constraint.

**Branch:** if the list came from a workshop, run the model twice — once on it, once on a
superset. A changed answer is the study's most important result, found before solving anything.
Section 9.3 quantifies the risk.

### Step 3 — Build the market set and the demand input

**Input and output:** the Module 07 forecast and the topic 02 aggregation become a demand
quantity `d_i` per market, in units per period, with a stated vintage, aggregation rule, and
error budget.

**Why it matters:** demand enters as a parameter that the model treats as certain. Three choices
are being made silently.

- **The aggregation.** A market is not a customer. Topic 02 showed that centroid aggregation
  understates line-haul distance and deletes the within-zone delivery distance entirely. The
  error is carried into `c_ij`, not `d_i`, but it belongs in the same input package.
- **The period.** Annual demand against annual capacity hides the peak. At a peak-to-average
  ratio of 1.4, a design at 90% average utilisation is over capacity for part of the year.
- **The vintage.** A study built in March on data frozen in January is answering a January
  question.

**Branch:** if the aggregation error exceeds the cost difference between the leading designs,
the decision sits inside the noise. Refine, or say so; do not report the ranking as significant.

### Step 4 — Choose the capacity basis

**Input and output:** engineering data, historical throughput, the shift pattern, and the
product mix become one capacity figure per candidate, in the same unit as demand, with a written
derivation and a stated utilisation ceiling.

**Why it matters:** the capacity constraint is what makes the model capacitated, and the number
in it decides how many sites the model will open. Four bases are available and they differ
systematically.

| Basis | Definition | Typical relation to nameplate | When to use |
|---|---|---|---|
| Nameplate | Rated design capacity under ideal conditions | 100% by definition | Almost never in a network model |
| Effective | Nameplate times availability, adjusted for changeover and mix | 85 to 95% of nameplate in continuous process plant | The default for a design study |
| Demonstrated | Best sustained throughput actually achieved | At or below effective | Diagnostic run, or a site known to be constrained |
| Planning capacity | Effective capacity times the utilisation ceiling | 75 to 90% of effective | The number that belongs in a strategic model |

The derivation is multiplicative and each factor needs a source. The American Institute of
Chemical Engineers' guidance on calculating plant capacity gives the form: availability times
design capacity, so a plant rated at 200,000 tonnes per year with 93.7% uptime has an effective
capacity of 187,400 tonnes per year. The same logic applies to a DC, with availability replaced
by usable operating hours and mix by slot and cube constraints.

**The utilisation ceiling.** No facility runs at 100%. At 90% slot occupancy a warehouse spends
its labour on putaway into scattered single slots and on forklift queuing, and sustainable
throughput falls. Practitioner guidance converges on 80 to 85% as the point at which congestion
bites, with order processing reported 20 to 30% slower at 90% than at 75%. Treat the exact figure
as a convention and the existence of a ceiling as a fact: the 85% number is repeated across
vendor material, so measure your own pick rate against occupancy before importing it.

**Branch:** if the capacity basis changes the open set, the study has found its most important
uncertainty. Report both designs and the cost difference; do not average the capacities.

### Step 5 — Build the unit cost coefficients

**Input:** the freight rate card or freight audit, the site's variable handling cost, and the
distance or time matrix from topic 02.

**Output:** one number `c_ij` per market-site pair, in currency per unit of demand, at a stated
price year, including a stated list of components.

**Why it matters:** `c_ij` is the only place the model sees geography. Its components are:

| Component | Typical unit | How it is built | Common error |
|---|---|---|---|
| Line-haul transport | Currency per unit or per tonne-km | Rate times distance, converted by load factor | Straight-line distance at a full-truckload rate |
| Delivery to market | Currency per unit | Route or drop cost divided by drop size | Omitted when markets are aggregated |
| Variable handling in | Currency per unit | Labour and consumables per unit received | Double-counted with transport |
| Variable handling out | Currency per unit | Pick, pack, and load per unit shipped | Allocated as a percentage of revenue |
| Duty and compliance | Currency per unit | Tariff rate times value, plus fees | Omitted on cross-border lanes |
| Inventory in transit | Currency per unit | Carrying rate times value times transit days | Included inconsistently across lanes |

Two disciplines make the coefficient defensible. Every conversion is written down: a rate per
tonne-km becomes a cost per unit only after a load factor and a product weight, and the load
factor is the number most often missing. And the price year is stated, with a deflator applied
rather than a hope if the study spans years.

**Branch:** if the cost matrix is built from a rate card rather than from an audit of paid
freight, the study is using list prices. That may be right for a strategic study, and it must be
said. List rates and paid rates can differ by 10 to 30% on the same lane.

### Step 6 — Build the fixed cost coefficients

**Input:** the site cost centres, the lease register, the headcount plan, and the maintenance
contracts.

**Output:** one number `f_j` per candidate, in currency per year, representing the cost that
disappears if the site does not exist.

**Why it matters:** `f_j` is the coefficient that the model trades against transport, and it is
the coefficient most often taken from a report that was never designed for this purpose.

The test for inclusion is a question, not a category: **if this site did not exist, would this
cost disappear?** Lease or depreciation, site labour, utilities, maintenance, insurance, and
site-specific information technology pass. Allocated corporate overhead, shared services, and
regional management usually fail. The classification is a management accounting judgement, and
it should be signed by a management accountant rather than inferred by the analyst.

Two refinements matter. **Step-fixed labour** does not jump at zero; it jumps at shift
boundaries. A site with two shifts has a fixed cost within a volume band that steps up when the
third shift is added, so treating it as purely fixed understates high-volume cost and treating
it as purely variable overstates low-volume cost. **One-time costs** — lease exit, severance,
make-good — do not belong in `f_j`, an annual recurring cost; they belong in a transition
calculation, as topic 01's worked example showed.

### Step 7 — Check feasibility before modelling

**Input:** total demand, total capacity, and the candidate set.

**Output:** a one-line feasibility statement.

**Why it matters:** the most common cause of an infeasible network model is not a modelling
error. It is that total demand exceeds total capacity in the candidate set, usually because a
site was excluded or a capacity was entered too low. Check `sum_i d_i <= sum_j cap_j` before
you write the model. Then check the tighter condition: for every subset of markets, the
capacity of the sites that can reach them must cover their demand. The second check is
combinatorial and is what the solver will discover for you, but a coarse version — the remote
region served by one site — catches most cases.

### Step 8 — Write the model

**Input and output:** the four coefficient sets and the index sets become the formulation below,
with every symbol defined and every unit stated.

```text
Sets
  I   markets (demand nodes), indexed i
  J   candidate sites, indexed j

Parameters
  d_i     demand at market i                         units per year
  f_j     fixed annual cost if site j is open        currency per year
  cap_j   capacity of site j on the chosen basis     units per year
  c_ij    unit cost of serving i from j              currency per unit

Decision variables
  x_j     = 1 if site j is open, 0 otherwise          binary
  y_ij    quantity shipped from j to i               units per year

Objective
  minimise   sum_j f_j x_j  +  sum_i sum_j c_ij y_ij

Subject to
  (1)  sum_j y_ij  =  d_i                    for every market i
  (2)  sum_i y_ij  <= cap_j x_j             for every site j
  (3)  y_ij >= 0                            for every pair (i, j)
  (4)  x_j  in {0, 1}                       for every site j
```

The fixed term is paid once per open site and makes the problem combinatorial; the flow term is
linear, so the model prefers the cheapest open site for each unit, subject to capacity.
Constraint (1) forces every market to be served in full: equality, because unserved demand has
no value here. If the business allows unserved demand at a penalty, add a slack variable priced
at the cost of a lost sale. Constraint (2) is capacity and linking in one: when `x_j = 0` its
right side is zero, so no flow enters or leaves a closed site. Constraints (3) and (4) allow
divisible demand and require the open decision to be a set.

**The linking constraint is the one to check.** Omitting it, or writing `sum_i y_ij <= cap_j`
with no `x_j`, lets the model ship through closed facilities and report an impossible cost. It is
the most common structural bug in hand-built network models. After every solve, assert that
`y_ij > 0` implies `x_j = 1`.

**About big-M.** The tightest valid value in (2) is `cap_j`, because no site can handle more than
its capacity. The loose form with `M = sum_i d_i` is valid and much weaker: it tells the solver
almost nothing and slows branch-and-bound. Set `M_j = min(cap_j, sum_i d_i)`, never a round number
like 1,000,000.

### Step 9 — Solve, and record what the solver tells you

**Input:** the model, an objective sense, a time limit, and a gap tolerance.

**Output:** an incumbent design, a lower bound, a MIP gap, and a solve time.

**Why it matters:** the solver's output is a certificate as well as an answer. Three numbers
matter.

- **The incumbent objective.** The cost of the best design found.
- **The best bound.** The cost below which no design in the candidate set can exist.
- **The gap.** `(incumbent - bound) / incumbent`, usually reported as a percentage. A 0.5% gap
  on a 40 million currency unit network is 200 thousand units of currency of possible
  improvement. Whether that is material is a decision for the reader of the study, not for the
  solver.

If the solver hits the time limit with a 15% gap, you do not have an optimal design. You have a
good design and a statement that a better one might exist. Report both.

### Step 10 — Validate on the current network

**Input and output:** the model with `x` fixed to the current footprint becomes a modelled cost
for the current network, the ledger cost, and a reconciliation.

**Why it matters:** this step buys the study its credibility, and it is the step most often
skipped. Fix `x_j = 1` for today's sites and 0 for the rest, solve for allocation, and compare
with the actual cost of the same scope and period. A variance within a few percent, explained
line by line, tells the committee the model understands today; 25% tells them it is a different
business. The reconciliation is where the findings live: the gap is either a cost the model omits
— inventory, service failure, reverse logistics — or an inefficiency the optimal allocation would
remove. The two point in opposite directions for the business case.

### Step 11 — Interpret the output in four parts

**Input and output:** the solved model becomes four artefacts, each with a stated interpretation.

1. **The open set.** Which sites, with utilisation and share of throughput. A site at 97% of the
   capacity basis is a risk, not a result.
2. **The allocation matrix.** Which markets each site serves and in what quantity. A market
   split across sites means either deliberate dual sourcing or a coincidence of costs; either
   is worth a sentence.
3. **The cost breakdown.** Fixed, flow, and total, with flow split by component if built that
   way. Saving 8% by adding 30% to transport and removing 40% of fixed cost is a different
   proposition from saving 8% evenly.
4. **The binding constraints.** Which capacities are tight. A tight capacity at a site in the
   recommended set is where the design is fragile.

### Step 12 — Run the sensitivity that matters

**Input and output:** the solved model and the uncertainty list become the open set as a function
of the parameters that could plausibly move, with the thresholds at which membership changes.

**Why it matters:** the committee forgets the cost of the recommended design and remembers its
stability. A site in the open set at every plausible demand and cost is a robust choice; a site
that enters if demand rises 3% is a bet. Three sensitivities carry most of the information.

- **Demand.** Scale by 0.8, 1.0, and 1.2, record the open set, and find the level at which a site
  enters or leaves.
- **Fixed cost.** For each site in or near the open set, compute the fixed cost at which it
  becomes or stops being justified. This is the threshold a lease negotiation can move.
- **Transport rate.** A uniform change in `c_ij` shifts the balance toward fixed or flow cost.
  Report the rate at which the ranking flips.

Topic 07 owns the full method; run these three before presenting, because the first question
from the room will be "what if demand is different".

### Step 13 — Build the option set and test executability

**Input and output:** the optimum, the near-optima, the current network, and the transition
constraints become three to five named designs costed on the same basis.

**Why it matters:** a single optimum invites a yes-or-no argument, and the answer is usually no,
because the design requires a closure nobody wants to own. An option set changes the question
from "do we accept the model's answer" to "which of these futures do we prefer". A workable set
contains the current network, the model optimum, a constrained optimum that respects the binding
legacy constraints, and one design the business proposed that the model did not find. If the
business's design is close to the constrained optimum, say so; it builds trust in both.

Executability is tested outside the model: leases, consultation, permits, customer contracts, and
the management capacity to run a transition. Topic 06 owns the brownfield machinery.

### 5.1 The reporting standard, and the last step

**Step 14 — report to the standard.** The output is a report that answers the six questions
below, in the order a sceptical reader would ask them.

A study that does not report these six items cannot be reviewed, and a reviewer who cannot
review it will either accept it on trust or reject it on instinct. Both are bad outcomes.

| # | Item | What must be stated | Why a reviewer asks |
|---:|---|---|---|
| 1 | Instance size | Number of markets, candidates, variables, constraints; solver and version; time limit and gap tolerance | To judge whether the answer is a solved problem or a truncated search |
| 2 | Candidate set | How it was built, what was excluded and why, whether a superset was tested | Because the answer lives inside this list |
| 3 | Capacity basis | Nameplate, effective, demonstrated, or planning; the derivation; the utilisation ceiling | Because this decides how many sites the model opens |
| 4 | Cost basis | Included and excluded lines, price year, currency, treatment of allocated overhead and one-time costs | Because the saving is a difference between two cost bases |
| 5 | Gap | Incumbent, best bound, gap as a percentage, and the currency value of the gap | Because "optimal" without a gap is a claim, not a certificate |
| 6 | Baseline validation | Modelled cost of the current network against the ledger, with the variance explained | Because a model that cannot reproduce today cannot be trusted about tomorrow |

Six items, one page. A study that reports them can be argued with. A study that does not cannot.

## 6. Worked example

Harbour Components distributes a single product family from three candidate DCs to four regional
markets. The instance is small enough to solve by hand and constructed so that the interesting
questions — capacity, integrality, and the fragility of a near-tight design — are visible. All
figures are illustrative and internally consistent.

### 6.1 The instance

| Market | Demand `d_i`, thousand units/yr | Share of total |
|---|---:|---:|
| M1 | 30 | 33% |
| M2 | 25 | 28% |
| M3 | 20 | 22% |
| M4 | 15 | 17% |
| **Total** | **90** | **100%** |

| Site | Capacity `cap_j`, thousand units/yr | Fixed cost `f_j`, thousand currency/yr | Fixed cost per unit of capacity |
|---|---:|---:|---:|
| A | 60 | 1,200 | 20.0 |
| B | 50 | 900 | 18.0 |
| C | 40 | 600 | 15.0 |
| **Total** | **150** | **2,700** | |

| Unit cost `c_ij`, currency per unit | A | B | C |
|---|---:|---:|---:|
| M1 | 4 | 7 | 9 |
| M2 | 6 | 4 | 7 |
| M3 | 8 | 6 | 3 |
| M4 | 9 | 7 | 5 |

Each `c_ij` already includes line-haul transport, delivery to the market, and variable handling
at the site. The fixed cost is the avoidable annual cost of the site and excludes allocated
corporate overhead. Costs are in constant 2026 currency. Demand is annual and flat, which is
itself an assumption worth flagging.

Two arithmetic facts organise what follows. Total capacity is 150 against demand of 90, so the
instance is feasible with room to spare. Total fixed cost if everything opens is 2,700 against a
flow cost of 355, so the fixed term is roughly eight times the flow term at this scale — typical
of distribution networks, and why the open decision dominates the allocation decision.

### 6.2 The seven non-empty open sets

Three candidate sites give `2^3 - 1 = 7` non-empty open sets. Four of them are feasible, because
three single-site sets cannot cover 90 thousand units.

| Open set | Fixed cost | Total capacity | Feasible | Flow cost | Total cost |
|---|---:|---:|---|---:|---:|
| {A} | 1,200 | 60 | No | — | — |
| {B} | 900 | 50 | No | — | — |
| {C} | 600 | 40 | No | — | — |
| {A, B} | 2,100 | 110 | Yes | 465 | 2,565 |
| {A, C} | 1,800 | 100 | Yes | 405 | 2,205 |
| {B, C} | 1,500 | 90 | Yes | 455 | 1,955 |
| {A, B, C} | 2,700 | 150 | Yes | 355 | 3,055 |

The cheapest flow cost belongs to the all-open set, at 355. The cheapest total cost belongs to
{B, C}, at 1,955, because it pays 1,200 less in fixed cost to save only 100 in flow cost. This is
the central trade-off of the model in one line: **a facility is justified when the flow cost it
removes exceeds the fixed cost it adds.**

### 6.3 Costing the three interesting sets

**{A, C}.** Unconstrained, each market takes its cheaper site: M1 and M2 to A, M3 and M4 to C.
Loads are A = 55 against capacity 60, and C = 35 against 40. Both fit, so no reallocation is
needed, and the cost is `30x4 + 25x6 + 20x3 + 15x5 = 405`, for a total of `1,800 + 405 = 2,205`.

**{A, B}.** Unconstrained, M1 goes to A, and M2, M3, M4 go to B. That loads B with 60 against a
capacity of 50. Ten thousand units must move from B to A, and the cheapest place to move them is
any market where the penalty `c_iA - c_iB` is 2: M2, M3, or M4. Moving 10 thousand units of M3
gives A = 40, B = 50, and a flow cost of `120 + 100 + 140 + 105 = 465`, for a total of
`2,100 + 465 = 2,565`.

**{B, C}.** Unconstrained, M1 and M2 go to B, and M3 and M4 go to C. That loads B with 55
against 50. Five thousand units must move to C, and the cheapest penalty is on M1, at
`9 - 7 = 2` per unit. The result is exactly at capacity on both sites, with a flow cost of
`(175 + 45) + 100 + 60 + 75 = 455` and a total of `1,500 + 455 = 1,955`.

Note the shape of the {B, C} answer: both sites are full. The design has no slack, and that fact
does not appear in the objective. It is the most important thing in the solution and the model
does not report it.

### 6.4 The winner, and by how much

{B, C} costs 1,955 thousand currency units per year. The runner-up, {A, C}, costs 2,205. The
margin is 250 thousand per year, or 11.3% of the runner-up. The all-open design costs 3,055, so
opening A as well destroys 1,100 thousand of value: A's fixed cost of 1,200 buys only 100 of
transport saving.

The solution in full:

| Output | Value |
|---|---|
| Open set | B, C |
| Allocation | M1: 25 to B, 5 to C. M2: 25 to B. M3: 20 to C. M4: 15 to C |
| Utilisation | B at 50/50 = 100%; C at 40/40 = 100% |
| Fixed cost | 1,500 thousand per year |
| Flow cost | 455 thousand per year |
| Total | 1,955 thousand per year |
| Binding constraints | B capacity, C capacity, M1 demand |

A utilisation of 100% on both sites is not a triumph. It is a warning, and section 6.6 shows
what a 1% demand increase does to it.

### 6.5 The relaxation of the same instance

Relax `x_j` to the interval [0, 1] and the model finds a cheaper answer. Because the fixed cost
can now be paid in proportion to the load, each site has an effective unit cost of
`c_ij + f_j / cap_j`: 20 for A, 18 for B, and 15 for C. Re-solving with those effective costs
gives the fractional design below.

| Quantity | Integer solution | LP relaxation |
|---|---:|---:|
| `x_A` | 0 | 0.4167 |
| `x_B` | 1 | 0.5 |
| `x_C` | 1 | 1 |
| Flow | 455 | 380 |
| Fixed | 1,500 | 1,550 |
| Total | 1,955 | 1,930 |

The relaxation serves M1 with 25 thousand units from A and 5 thousand from C, and serves M2
entirely from B. It pays 41.7% of A's fixed cost and 50% of B's. No such design exists. Its value,
1,930, is a valid lower bound on every design in the candidate set, and the solver's gap at the
integer solution is `(1,955 - 1,930) / 1,955 = 1.3%`.

A small gap and a nonsensical fractional design are compatible. The gap measures how much cost
the integrality costs on this instance, not whether the fractional answer is usable.

### 6.6 Two flips

**A cost flip.** Hold everything except B's fixed cost. {B, C} costs `1,500 + 455`. If `f_B` rises
by more than 250, {A, C} at 2,205 becomes cheaper. The threshold is `f_B = 1,150` thousand per
year, 28% above the base value. A lease negotiation that adds 250 thousand to B's annual cost
reverses the recommendation. That is the number to put in front of the person negotiating the
lease, and it is more useful than the total saving.

**A demand flip.** {B, C} has total capacity exactly equal to demand. Add one thousand units of
demand anywhere and it becomes infeasible, not merely expensive. The next feasible design is
{A, C} at about 2,209, a 13% increase in cost for a 1% increase in volume. This is a constructed
knife-edge, and real instances usually have more slack, but the phenomenon is not artificial:
networks that have been consolidated to the edge of capacity are common, and a model run on
average demand will not warn you.

Both flips belong in the report. Neither is a sensitivity in the sense of topic 07; they are
thresholds that make the recommendation conditional, and they cost nothing to compute.

### 6.7 What the model could not tell us

Four questions decided the recommendation and none of them was answered by the arithmetic.

1. **Is B's fixed cost really 900?** It came from the site cost centre. If it includes an
   allocation that survives closure, the real figure is lower and {B, C} is safer; if it
   excludes a maintenance contract that must be renewed, the flip at 1,150 is closer than it
   looks.
2. **Can C actually sustain 40 thousand units?** The capacity is effective capacity at a 90%
   ceiling, derived from 44 thousand of demonstrated throughput, so at 100% of this basis C is
   at 111% of its planning ceiling. The model allowed it because it has no congestion term.
3. **Will M1 accept a split?** The integer solution splits M1 between B and C. If the customer
   requires a single source, M1 moves entirely and the cost rises.
4. **Can the organisation close A?** Whether it can be closed, at what cost, and over what
   period is a separate decision with its own evidence.

The first two are input disputes the model cannot settle, the third a constraint it did not
have, the fourth the executability question from topic 01. A study that reports 1,955 and stops
has answered a narrower question than the committee asked.

## 7. Code, query, or calculation

The calculation below reproduces every number in section 6 with a pencil and a calculator. It is
the preferred form for a small instance, because it exposes the structure that a solver call
hides: which sets are feasible, which constraint binds, and how far the answer is from its
neighbours.

```text
Instance:  d = [30, 25, 20, 15]   sum = 90
           f = [1200, 900, 600]  cap = [60, 50, 40]

Open set   capacity   fixed   constrained allocation           flow   total
{A,B}        110       2100    M1->A 30; M2->B 25; M3->A 10,B 10; M4->B 15   465   2565
{A,C}        100       1800    M1->A 30; M2->A 25; M3->C 20; M4->C 15        405   2205
{B,C}         90       1500    M1->B 25,C 5; M2->B 25; M3->C 20; M4->C 15   455   1955
{A,B,C}      150       2700    M1->A 30; M2->B 25; M3->C 20; M4->C 15        355   3055

best integer = 1955   (open B and C, both at 100% of capacity)

LP bound:  fixed cost per unit of capacity f_j/cap_j = 20.0 (A), 18.0 (B), 15.0 (C)
           add to c_ij, then assign within capacity:
           M1 -> A 25, C 5; M2 -> B 25; M3 -> C 20; M4 -> C 15
           flow = 25x4 + 5x9 + 25x4 + 20x3 + 15x5 = 380
           fixed = 0.4167x1200 + 0.5x900 + 1.0x600 = 1550
           bound = 1930        gap = (1955 - 1930) / 1955 = 1.3%

flip 1:  f_B* = 2205 - 600 - 455 = 1150   (900 + 250)
flip 2:  demand 91 makes {B,C} infeasible; {A,C} costs 2209 at 91
```

Two readings matter. **Feasibility is a property of the set, not the demand.** The single-site
rows are absent because no site carries 90 thousand units; at larger scale the same condition is
what makes a model infeasible. **The constrained allocation is where the arithmetic lives.** For
both {A, B} and {B, C} the unconstrained allocation overloads B, and the cheapest correction
costs 2 per unit, small against the fixed-cost differences. That is why `f` decides the open set
and the rate table decides the allocation, at realistic fixed-to-variable ratios.

The LP bound is easy to compute here and worth computing always. The effective-unit-cost trick
works whenever a site can open in proportion to its load, which holds here because no site is
forced open. A bound far below the incumbent means either a hard instance or a weak formulation.

> **Note:** for a real instance you will not enumerate open sets, and should not. The hand
> method is a teaching and validation device: run it on a 3-by-4 sub-instance drawn from real
> data and check the solver returns the same answer. If it does not, you have found a modelling
> error or a misunderstanding, cheaply.

## 8. Data, standards, and regulatory context

The model has four inputs, and each is governed by something. Knowing what governs an input
tells you which argument about it can be settled by a document and which cannot.

### 8.1 Identity and units

| Standard or convention | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 Global Location Number (GLN) | Site identity | The candidate list, the existing footprint, and the freight ledger must key to the same location identifiers or the cost join fails | B |
| ISO 3166 country codes | Market geography | Cross-border lanes, duty, and carbon boundaries | B |
| ISO 4217 currency codes | Cost coefficients | Every `f_j` and `c_ij` carries a currency; mixed-currency models need an explicit rate and date | B |
| ISO 80000 quantities and units | Demand and capacity | Demand and capacity must be in the same unit before the capacity constraint means anything | B |
| ISO 8601 | Periods and dates | The planning period, the price year, and the data vintage | B |

The unit discipline is not pedantry. A model that carries demand in cases and capacity in pallet
positions is not a model; it is two models sharing a file. Convert once, at the input boundary,
and record the conversion factor and its source. Topic 02 gives the conversion table; this topic
assumes it has been applied.

### 8.2 Capacity definitions

| Source | What it supplies | Status | Grade |
|---|---|---|---|
| ASCM supply chain dictionary | Demonstrated capacity as capacity proven by actual performance | Professional reference, not regulation | C |
| AIChE guidance on calculating plant capacity | The availability-times-design form of effective capacity | Professional engineering guidance | B |
| Site engineering records | Design ratings, uptime, changeover, and mix limits | Company data; quality varies by site | C |
| Warehouse operating data | Pick rate against occupancy, dock dwell, overtime | Company data; the best evidence for a ceiling | C |

Nothing in this list tells you which capacity to use. The standards define the terms; the
choice among them is a modelling decision that must be written down. This is the clearest case
in the topic of a distinction the spec insists on: a definition is a requirement, a basis is a
judgement.

### 8.3 Cost and accounting interfaces

| Requirement | Where | Effect on the model | Grade |
|---|---|---|---|
| IFRS 16 / ASC 842 leases | Global accounting | A leased site's rent appears as depreciation and interest, not rent; the model still needs the annual cash cost, so a bridge is required | B |
| IAS 37 provisions | Global accounting | Closure obligations may be recognised as a provision; they belong in the transition calculation, not in `f_j` | B |
| IFRS 5 non-current assets held for sale | Global accounting | A site designated for sale stops being depreciated; the fixed cost basis changes | B |
| Transfer pricing rules | Multinational | Internal flows between entities carry a transfer price that is not an economic cost; do not use it as `c_ij` | B |
| ISO 14083:2023 | Global | Defines how transport emissions are quantified if carbon enters the objective; topic 09 | B |
| CBAM (carbon border adjustment mechanism), Regulation (EU) 2023/956 | European Union | From 1 January 2026, a carbon cost on covered imports, which is an additional `c_ij` term on affected lanes | B |

The lease standard is the one that most often derails a cost base. A site's profit and loss
account shows a depreciation charge and an interest charge; the lease register shows a rent.
Neither is automatically the avoidable annual cost the model needs. Ask for the cash cost, and
state in the report which figure you used.

### 8.4 What is requirement and what is convention

- **Requirement:** a site must be identified by a stable identifier that joins to the freight
  ledger. **Convention:** that the identifier is a GLN rather than an internal code.
- **Requirement:** demand and capacity are in the same unit. **Convention:** that the unit is
  tonnes rather than cases.
- **Requirement:** an open facility cannot handle more than its effective capacity.
  **Convention:** that effective capacity is 90% of nameplate, or that the ceiling is 85%.
- **Requirement:** the fixed cost in a closure case is the avoidable cost. **Convention:** which
  ledger lines are avoidable.

An analyst who can separate the two columns can defend a modelling choice. One who cannot will
either treat a convention as a law or a requirement as a preference.

## 9. Industry practice and case evidence

### 9.1 What the documented cases show

The best-documented application of a distribution-network model in the marketing literature is
the GlaxoSmithKline study reported by Shang and colleagues in the *Journal of Marketing* in
2009. It redesigned regional distribution centre location, capacity, and allocation, and the
published account reports distribution costs down by about 1.99 million US dollars per year, or
6%, transport down 17.9%, and one-day delivery rising from 61.4% to 86.2%. Grade A; the figures
are quoted as indexed and should be checked against the paper before a business case repeats them.

The P&G study described in topic 01 is the other well-documented case, and it is worth noting
what it did not do. It used an uncapacitated facility-location model for the DC layer alongside
a capacitated mixed-integer model for plants and a separate sourcing model. The uncapacitated
choice was a deliberate simplification at one layer of a linked model, not a claim that DCs have
infinite capacity. That is a defensible use of the UFLP; using it as the only model is not.

### 9.2 What vendor cases report

| Case | Reported result | Grade | How to read it |
|---|---|---|---|
| Apparel distribution, simulation-based greenfield analysis | 32 to 45% cost reduction across scenarios | D | Vendor case study; baseline and scope not disclosed |
| Healthcare distributor, greenfield analysis | Operating cost down over 10%, service up 25% | D | Vendor case study; service metric undefined |
| Diagnostics manufacturer, India | 27% additional saving from a fourth warehouse and a relocation | D | Consultancy case study; short-horizon comparison |
| Canadian agribusiness | Logistics cost down up to 15%, warehouses halved | D | Vendor case study; "up to" is doing the work |

The pattern matches topic 01: cost-focused redesigns report savings in the 6 to 15% range, and
restructuring cases with new or relocated sites report much higher figures because the baseline
includes an inherited, unreviewed network. The percentages are not comparable; the baselines are
not.

### 9.3 Candidate-set evidence

The most useful quantified evidence on candidate-set bias is the Guazzelli and Cunha study cited
in section 5: on 147 microregions, restricting candidates to the highest-demand decile produced
suboptimality of roughly 29%, and a candidate set of about 40% of nodes was optimal in 83% of
instances. Grade C: a conference paper, and the percentages are specific to the instance. The
direction is what transfers.

Madleňák and colleagues, comparing the p-median and the uncapacitated fixed-charge model on a
110-node Italian postal network, report 11 centres under the p-median and 13 under the
fixed-charge model, at partly different nodes. Grade A: the formulation, not the data, chose the
number of facilities.

### 9.4 What practitioners actually do

The observable practice, from survey material and the authors' own project experience, is less
rigorous than the literature. Candidate lists are assembled in a workshop and rarely exceed a
few dozen sites. Capacity is a single number supplied by operations, with the basis unstated.
The fixed cost comes from the cost centre and is rarely reconciled to avoidability. Baseline
validation is often skipped because the model is "for 2030". These are Grade E observations,
offered as a description of the norm rather than evidence about any company.

The consequence is not that the models are useless. It is that the reported optimum is the
optimum of a narrower problem than the one the committee believes it is deciding, and that gap
is where the implementation risk lives.

### 9.5 What the evidence supports

It supports three claims. A facility-location model can produce large savings when applied to a
network that has not been reviewed. The savings come mostly from removing capacity the business
no longer needs, not from discovering a clever location. And the model's contribution is
conditional on inputs — candidate set, capacity basis, cost basis — supplied by people, not by
data. It does not support the claim that a bigger model produces a better decision, that a
solver gap of 1% means the answer is right, or that a vendor's percentage will transfer.

## 10. Failure modes and red flags

Each row names a failure, the red flag that is visible in a meeting or a dataset before anyone
has diagnosed it, the mechanism, the consequence, and the mitigation.

| Failure | Red flag you can see | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Flow through a closed site | The allocation includes a site that is not in the open set | Linking constraint missing or the big-M exceeds capacity | Impossible cost; design cannot be built | Assert `y_ij > 0 implies x_j = 1` after every solve |
| Capacity omitted or non-binding | Every site in the answer is at 100% utilisation | The capacity row was dropped because "it made the model infeasible" | Over-consolidated design; peak-week failure | Check total capacity against demand, then per-region capacity |
| Nameplate capacity used | Capacity equals the design rating in the input file | The only number in the engineering document | Design needs more sites than recommended, or fails | Use effective or planning capacity with a written derivation |
| Candidate list from a workshop | Every candidate is a site someone already knew | Memory and politics, not geography | Optimum outside the list is never found | Generate sites from the demand surface; test a superset |
| Fixed cost equals the cost centre | `f_j` matches the ledger total to the currency unit | Allocated overhead included | Closure saving overstated by 30 to 50% | Ask a management accountant which costs disappear |
| Price year mixed | Freight at one year's rates, rent at another's | Inputs pulled from systems with different vintages | Design biased toward the deflated component | State the price year; deflate every input to it |
| Annual average demand | Peak month is 1.4 times the average and absent from the file | Model runs on annual totals | Design is over capacity for part of the year | Run a peak period, or derate capacity by the peak ratio |
| Splits where the business forbids them | The allocation serves one customer from two sites | The model allows divisible demand by default | Recommended design breaches a customer contract | Add single-sourcing where required and price the constraint |
| Gap not reported | "Optimal" appears with no bound, gap, or time limit | The solver output was summarised by its first line | A 12% gap is presented as an optimum | Report incumbent, bound, gap, and the gap in currency |
| No baseline validation | No run of the model on the current network | The study is framed as forward-looking only | Model cannot reproduce today; every number is suspect | Fix `x` to the current footprint and reconcile to the ledger |
| One design presented | The deck has one recommended network and no alternatives | The optimum was treated as the answer | Committee argues yes/no and defaults to no | Present three to five designs on a common cost base |
| Aggregation error not carried | The zone map is inherited and unexplained | Topic 02's error is invisible at the coarser resolution | Ranking inside the noise; false precision | Two-level run and an error budget against the design gap |

**Capacity omitted to restore feasibility.** When a model is infeasible the fastest fix is to
delete a constraint, and the capacity row is usually the one that made it infeasible. The
infeasibility was information: the candidate set cannot serve the demand, which is a finding
about the network rather than a defect in the model. Deleting the row converts a finding into a
wrong answer. Diagnose it by finding the smallest set of markets whose reachable capacity is
short, then report that.

**Splits that the business forbids.** The baseline model treats demand as divisible. That is a
modelling convenience and often a commercial falsehood: customers have single-source clauses,
service contracts name a site, and quality agreements restrict transfers. The cost of imposing
single-sourcing can be large, because it turns the problem into one containing bin packing, so
measure it rather than assume either way. Solve with and without, and report the difference as
the price of the constraint.

## 11. Assumptions and limits

| Assumption | Why the model needs it | What breaks if it fails |
|---|---|---|
| Demand is known and constant over the period | It is a parameter, not a variable | Near a capacity threshold, a small forecast error flips the open set — as section 6.6 showed |
| Capacity is independent of product mix | One capacity number per site | A mix shift toward bulky or slow-moving items reduces usable capacity without changing the number |
| Costs are linear in volume | The objective is linear | Economies of density make large sites cheaper per unit than the model believes, biasing toward consolidation |
| The network handles one commodity | One flow index | Products with different handling, temperature, or value need separate flows and shared capacity (topic 05) |
| All demand must be served | The demand constraint is an equality | If unserved demand is acceptable at a cost, the model overstates the need for capacity |
| The candidate set contains the best site | The model cannot search outside it | The true optimum is excluded, and the reported saving is the value of choosing among the sites offered |
| Any open site can serve any market | Every pair has an arc | Service windows, customs, and customer contracts remove arcs; the model will use them anyway |
| Fixed costs do not vary with volume or time | `f_j` is a single number | Step-fixed labour and multi-year leases make the true fixed cost a function of throughput and calendar |
| Transport cost is separable per unit | `c_ij` is constant per unit | Minimum drop charges and full-truckload steps make the marginal cost fall with volume |
| One period, no inventory | No inventory variables | Multi-year phasing and inventory build-up are invisible; topic 05 and topic 06 |

**The model optimises a snapshot.** It compares designs under one set of costs and one demand
level, while real networks face cost drift, demand growth, and technology change. The answer is
conditional on the snapshot; topic 07 converts the condition into a range.

**The model has no theory of the firm.** It does not know about management capacity, union
agreements, customer relationships, or the credibility cost of reversing a decision. Those
constrain which designs are executable, and no coefficient represents them. An analyst who
forgets this presents a design that is optimal and impossible, then is surprised by the
rejection.

## 12. Exercises

### Exercise 1 — State the model in words, then in algebra (recall, 30 minutes)

**Deliverable:** one paragraph in plain language followed by the full formulation, with every
symbol defined and every unit stated.
**Acceptance criteria:** the paragraph names the choice, the assignment, the capacity limit, and
both cost terms; the algebra has demand, capacity-linking, and domain constraints; omitting the
linking constraint produces a written explanation of what goes wrong.

### Exercise 2 — Enumerate a small instance (calculation, 60 minutes)

Using the instance in section 6, change `f_C` to 750 and re-evaluate all seven open sets.
**Deliverable:** a table of feasible sets with fixed, flow, and total cost, and the new winner.
**Acceptance criteria:** every feasible set present; the constrained allocation shown for each;
the margin over the runner-up stated in currency and percent; arithmetic reproducible from the
table.

### Exercise 3 — Bound the answer (application, 45 minutes)

For the same instance, compute the LP relaxation using the fixed-cost-per-unit-of-capacity
method.
**Deliverable:** the fractional solution, the bound, and the gap against the integer optimum.
**Acceptance criteria:** the fractional open variables and the split allocation are stated; the
bound is shown to be a lower bound; the gap is expressed both as a percentage and in currency.

### Exercise 4 — Audit a capacity basis (application, 45 minutes)

Take a facility you know, or a published case, and derive its effective and planning capacity
from nameplate, availability, changeover, and mix.
**Deliverable:** a one-page derivation with each factor sourced, plus the utilisation ceiling and
its justification.
**Acceptance criteria:** every factor has a source or is labelled an assumption; the difference
between nameplate and planning capacity is quantified; the effect on a site count is estimated.

### Exercise 5 — Build and test a candidate set (design, 60 minutes)

For a named region, construct a candidate set three ways: existing sites only, a workshop list,
and a demand-surface list.
**Deliverable:** the three lists, the inclusion and exclusion reasons, and a plan for testing
whether the answer changes.
**Acceptance criteria:** at least one candidate appears only in the demand-surface list; every
exclusion has a stated reason; the test is a concrete run, not an assertion.

### Exercise 6 — Reconcile a fixed cost (critique, 50 minutes)

Given a site cost centre totalling 2.4 million per year, classify each line as avoidable,
step-fixed, or allocated.
**Deliverable:** a three-column table and a single `f_j` with a stated basis.
**Acceptance criteria:** allocated overhead is identified and excluded; step-fixed labour is
flagged with its volume band; the reconciliation from ledger total to `f_j` is shown.

### Exercise 7 — Compute the flip (calculation, 40 minutes)

For the section 6 instance, compute the fixed cost of B at which {A, C} becomes cheaper, and the
demand level at which {B, C} becomes infeasible.
**Deliverable:** both thresholds with the algebra.
**Acceptance criteria:** the fixed-cost threshold is within 1 of 1,150; the demand threshold is
stated as a total and as a percentage; the report says which threshold is more likely to move.

### Exercise 8 — Write to the reporting standard (design, 60 minutes)

Produce a two-page report on the section 6 study covering the six items in section 5.1.
**Deliverable:** the report, with the six items as headings.
**Acceptance criteria:** instance size and gap stated; candidate set and capacity basis
justified; cost basis enumerated; baseline validation described even though it is hypothetical
here; no claim unsupported by the arithmetic.

## 13. Self-check answers

**Exercise 1.** In words: choose which candidate sites to open, assign every market's demand to
open sites, respect each site's capacity, and minimise fixed plus flow cost. The algebra is the
four-part formulation in section 5, Step 8. Omitting the linking constraint leaves
`sum_i y_ij <= cap_j`, which permits flow through a site with `x_j = 0`; the model then serves
demand from sites it never pays for, and the objective understates cost by the entire fixed
charge of every site used.

**Exercise 2.** With `f_C = 750`: {A, B} stays 2,565; {A, C} becomes 1,800 − 600 + 750 + 405 =
2,355; {B, C} becomes 1,500 − 600 + 750 + 455 = 2,105; {A, B, C} becomes 2,850. The winner is
still {B, C} at 2,105, and the margin over {A, C} falls from 250 to 250 — the same 250, because
both sets contain C and the change cancels. This is the point of the exercise: a cost change
that affects every feasible set equally does not change the ranking. Only changes that
differentiate the sets move the answer.

**Exercise 3.** Fixed cost per unit of capacity is 20, 18, and 15. Adding those to `c_ij` and
assigning within capacity gives 25 thousand units of M1 from A, 5 from C, 25 of M2 from B, 20 of
M3 from C, and 15 of M4 from C. Flow is 380 and fixed is 1,550, so the bound is 1,930. The gap
against 1,955 is 25, or 1.3%. It is a lower bound because every feasible fractional point,
including every integer point, costs at least the LP optimum.

**Exercise 4.** A site rated at 200,000 tonnes per year with 93.7% availability has an effective
capacity of 187,400 tonnes. Applying a 90% planning ceiling gives 168,660 tonnes. If the site is
modelled at nameplate, the model will load it with 200,000 tonnes, which is 18.6% above the
planning figure. On a network of five such sites, that difference is roughly one site's worth of
capacity, so the model will recommend opening one site fewer than the operation can sustain.

**Exercise 5.** A strong submission shows at least one demand-surface candidate that the
workshop missed, typically a location between two existing sites or at the edge of a growing
market. Exclusion reasons must distinguish "no land or labour" from "not considered". The test is
a run on both lists, with the difference in total cost and in open-set membership reported; if
the difference is below the solver gap, the list was adequate.

**Exercise 6.** Avoidable: site labour, lease or depreciation, utilities, maintenance, insurance,
site IT. Step-fixed: labour that steps at a shift threshold, with the band stated. Allocated:
corporate overhead, shared services, regional management. If 0.7 million of the 2.4 million is
allocated, `f_j` is 1.7 million plus any step-fixed component at the planned volume. The
reconciliation line is 2.4 − 0.7 = 1.7, and it must be signed off by a management accountant.

**Exercise 7.** Setting `f_B + 600 + 455 = 2,205` gives `f_B = 1,150`, so the threshold is 1,150
thousand per year, 250 above the base. Demand: {B, C} has capacity 90 exactly, so any total above
90 is infeasible; at 91, {A, C} costs 1,800 + 409 = 2,209. The demand threshold is more likely to
move, because a 1% forecast error is common and a 28% lease increase is not.

**Exercise 8.** A strong report states: 4 markets, 3 candidates, 12 flow variables, 3 binaries,
7 constraints; solver and gap (1.3%, 25 thousand); candidate set built from the existing
footprint plus a demand-surface screen, with the excluded sites named; capacity as effective at a
90% ceiling; cost basis constant 2026 currency, avoidable fixed cost excluding 0.7 million of
allocation; baseline validation as a fixed-`x` run to be reconciled within 5%. A weak report gives
the open set and the saving, and omits at least three of the six items.

## 14. Interview questions

**1. "Explain the capacitated facility-location problem to someone who has never seen it."**

Strong: choose sites, assign demand within capacity, pay fixed plus flow cost; the fixed charge
is what makes it combinatorial. Weak: recites the objective without naming the decision.

**2. "Why does the LP relaxation split a market's demand?"**

Strong: relaxing integrality lets the fixed cost be paid in proportion to load, so capacity
becomes divisible and one market can be served from several sites at a lower bound. Weak:
"because it is a relaxation", with no economics.

**3. "What does a 1% MIP gap tell you?"**

Strong: no design in the candidate set is more than 1% cheaper than the incumbent, given the
bound and time limit; it says nothing outside the candidate set or about the integrality gap.
Weak: that the answer is optimal.

**4. "Which capacity would you put in the model?"**

Strong: effective capacity at the planning horizon, derived from nameplate times availability
and adjusted for mix, with a ceiling below 100% and a sensitivity at demonstrated capacity.
Weak: nameplate, because it is in the specification.

**5. "Why is the uncapacitated model easier, and why is it dangerous?"**

Strong: dropping capacity removes the coupling between markets and gives a better-behaved
relaxation, but lets the model consolidate without limit. Weak: easier because it has fewer
constraints, with no account of what changes.

**6. "What belongs in the fixed charge?"**

Strong: the avoidable annual cost — lease or depreciation, site labour, utilities, maintenance,
insurance — excluding allocated overhead and one-time costs. Weak: the cost centre total.

**7. "How would you validate the model before presenting it?"**

Strong: fix the open variables to the current footprint, solve, reconcile to the ledger, and
explain the variance; check units, the linking constraint, and capacity against demand. Weak:
that the solver said "optimal".

**8. "What two questions can the output not answer?"**

Strong: whether the solution is stable under plausible change, and whether the organisation can
execute it. Weak: data quality, a third question, not one of the two that decide implementation.

**9. "Why present an option set rather than one design?"**

Strong: a single optimum invites a yes-or-no decision that defaults to no; an option set makes
it a choice among costed futures and prices each constraint. Weak: options are more persuasive,
a communication claim rather than a decision claim.

**10. "What is the difference between a solution and a decision?"**

Strong: the solution is an open set and an allocation; the decision is a sequence of leases,
closures, hires, and customer conversations, conditional on stability and executability. Weak:
treats the solution as the decision with a recommendation attached.

## 15. Key terms

- **Capacitated facility location problem (CFLP)** — choose sites and allocations, with finite capacity at each site.
- **Uncapacitated facility location problem (UFLP)** — the same without capacity; better behaved, and usually a lie about the plant.
- **p-median** — choose exactly `p` sites to minimise demand-weighted distance, with no fixed charge.
- **Fixed charge `f_j`** — the annual cost incurred if a site is open, independent of volume.
- **Flow cost `c_ij`** — the cost of serving one unit of market `i` from site `j`.
- **Candidate set** — the sites the model may open; the most consequential and least examined input.
- **Open set** — the sites the solution selects; one output, not the decision.
- **Allocation matrix** — the quantity served to each market from each site.
- **Capacity basis** — the definition used for `cap_j`: nameplate, effective, demonstrated, or planning.
- **Nameplate capacity** — rated maximum under ideal conditions.
- **Effective capacity** — nameplate adjusted for availability, changeover, and mix.
- **Demonstrated capacity** — throughput actually sustained in operation.
- **Planning capacity** — effective capacity times the utilisation ceiling.
- **Utilisation ceiling** — the level above which congestion reduces sustainable throughput; a convention, not a law.
- **Big-M** — a constant that switches a constraint with a binary; the tightest valid value is the capacity.
- **Linking constraint** — the row that forbids flow through a closed site.
- **LP relaxation** — the model with binary variables allowed between 0 and 1.
- **Fractional solution** — an LP answer that opens a site partially and splits demand; a bound, not a design.
- **Integrality gap** — worst-case ratio of the integer optimum to the LP optimum for a formulation.
- **MIP gap** — the distance between the incumbent and the solver's bound on one instance.
- **Shadow price** — the marginal value of a constraint; valid from an LP, not directly from a MIP.
- **Break-even threshold** — the parameter value at which two designs cost the same.
- **Avoidable fixed cost** — the part of a site's cost that disappears if it closes.
- **Allocated overhead** — shared cost assigned by rule; not a saving in a closure case.
- **Option set** — three to five designs costed on a common basis, presented instead of one answer.
- **Baseline validation** — running the model on the current network and reconciling to the ledger.

## 16. Sources

Grading: A peer-reviewed or audited; B regulatory, official, or standard-setting; C company
case, conference paper, or institutional analysis; D vendor or consultancy; E press or anecdote.

**Peer-reviewed and foundational**

- Weber, A. (1909). *Über den Standort der Industrien*. The continuous single-facility problem. **A**
- Hakimi, S. L. (1964). "Optimum locations of switching centers and the absolute centers and
  medians of a graph." *Operations Research*, 12(3). Vertex optimality behind the p-median. **A**
- Balinski, M. L. (1965). "Integer programming: methods, uses, computation." *Management
  Science*, 12(3), 253-313. DOI: 10.1287/mnsc.12.3.253. The fixed-charge formulation. **A**
- Erlenkotter, D. (1978). "A dual-based procedure for uncapacitated facility location."
  *Operations Research*, 26(6). Dual ascent for the UFLP. **A**
- Reese, J. (2006). "Solution methods for the p-median problem: an annotated bibliography." *Networks*. **A**
- An, H.-C., Singh, M., and Svensson, O. (2014). "LP-based algorithms for capacitated facility
  location." *IEEE FOCS*; arXiv:1407.3263. The unbounded gap example. **A**
- Li, S. (2013). "A 1.488 approximation algorithm for the uncapacitated facility location
  problem." *Information and Computation*. Best known ratio for metric UFLP. **A**
- Madleňák, V., and colleagues (2016). p-median against fixed-charge location on a 110-node
  Italian postal network. *Transport and Telecommunication*, 17(3). 11 centres against 13. **A**
- Shang, J., and colleagues (2009). GlaxoSmithKline distribution network redesign. *Journal of
  Marketing*. Cost down about 6% (1.99 million dollars per year), transport down 17.9%, one-day
  delivery 61.4% to 86.2%. Citation as indexed; verify before quoting. **A**
- Camm, J. D., and colleagues (1997). "Blending OR/MS, judgment, and GIS." *Interfaces*, 27(1).
  DOI: 10.1287/inte.27.1.128. The linked P&G models described in topic 01. **A**

**Standards, regulation, and official guidance**

- AIChE, "Calculating the Capacity of Chemical Plants," *Chemical Engineering Progress*, August
  2014. Availability times design capacity; the 93.7% and 187,400 tonne example. **B**
- ASCM supply chain dictionary, demonstrated capacity. Professional reference. **C**
- IFRS 16 *Leases*; IAS 37 *Provisions*; IFRS 5 *Non-current Assets Held for Sale*. **B**
- ISO 14083:2023, transport chain greenhouse gas quantification. **B**
- Regulation (EU) 2023/956, carbon border adjustment mechanism; definitive regime from 1 January
  2026. **B**
- GS1 Global Location Number; ISO 80000, ISO 4217, ISO 3166, ISO 8601. **B**

**Company, conference, and institutional**

- Guazzelli, C. S., and Cunha, C. B. (ANPET, Brazil). Candidate pre-screening; 147 microregions.
  About 29% suboptimality at a 10% candidate set, 83% optimality at about 40%. Conference
  paper. **C**
- P&G and National Semiconductor cases as described in topic 01. **C**

**Vendor and consultancy**

- AnyLogic, apparel distribution centre case study. 32 to 45% cost reduction; baseline not
  disclosed. **D**
- anyLogistix, healthcare distributor and Canadian agribusiness cases. Cost down 10% and up to
  15%; service up 25%. **D**
- Nexdigm, warehouse network optimisation, India. 27% additional saving from a fourth warehouse
  and a relocation. **D**
- Warehouse utilisation guidance placing the congestion threshold at 80 to 85% and reporting
  20 to 30% longer order processing at 90% than at 75%; repeated across vendor sources, not a
  standard. **D**

**Press and practitioner**

- Practitioner observation in this file on how candidate lists, capacities, and fixed costs are
  typically assembled. Description, not measurement. **E**

> **Note on citation discipline.** The 85% utilisation ceiling, the 10 to 30% list-versus-paid
> freight gap, and the 6 to 15% vendor savings cluster are practitioner ranges, not findings.
> The GSK figures are quoted as indexed. No percentage here should be transferred to another
> network without rebuilding the baseline.

## 17. Links

**This module**

- Module map: [`_index.md`](_index.md)
- Prerequisite, framing: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- Prerequisite, graph and aggregation: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- MILP anatomy, the topic this one applies: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- Next topic: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- Later, all planned: `06_greenfield-brownfield.md`, `07_scenario-sensitivity.md`,
  `08_robust-stochastic.md`, `09_carbon-objective.md`, `10_solvers-heuristics.md`

**Adjacent modules**

- Inventory policy the network inherits: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Transport cost mechanics: [`../12_transportation_routing_last_mile/_index.md`](../12_transportation_routing_last_mile/_index.md)
- Warehouse capacity and labour: [`../13_warehousing_fulfillment_labor/_index.md`](../13_warehousing_fulfillment_labor/_index.md)

**Programme navigation**

- Part II methods map: [`../_index.md`](../_index.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

> **Note:** all ten topic files in this module are now written. Every markdown link on this page
> resolves to a file that exists on disk.
