# Solvers, Heuristics, and Practical Limits — MIP gaps, decomposition, and when a heuristic is the professional answer

> Module 11 — Network Design and Optimization
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11, topics 03, 04, and 05
> Version 2026.5

*The closing topic: what happens when a finished model meets a solver, and how a professional decides that the answer is good enough.*

---

## 1. Why this topic exists

Nine topics have built one object. Topic 01 chose the decision and its time scale. Topic 02 turned
the network into a graph and the cost into a decomposition. Topic 03 built the model and named the
choices that decide whether it can be solved at all. Topic 04 wrote the capacitated
facility-location problem in full. Topic 05 added commodities and periods. Topic 06 handled
greenfield against brownfield. Topic 07 varied the assumptions, Topic 08 hedged them, and Topic 09
put carbon in the objective. At the end of that chain there is a file — a few thousand coefficients
and a few hundred binaries — and a person who has to press run.

This topic is about the hour after that press. It is the only topic in the module that is about
behaviour rather than structure, and it exists because the behaviour is routinely misread.

The misreading has a standard shape. A model is submitted, the solver runs for a while, and it
returns a design, an objective value, and the word `OPTIMAL` or a status code that somebody
translates into that word. The design goes into a capital paper. Nobody asks what the solver proved,
how long it was allowed to run, or whether the run was stopped by a limit. The answer may be
excellent. The claim made about it is almost always stronger than the evidence, and the gap between
the claim and the evidence is the subject of this topic.

Three facts organise everything that follows.

**Fact 1 — A mixed-integer program (MIP) solver is a proof machine, not a solution machine.** It
does produce solutions, often quickly. Its distinguishing behaviour is that it maintains a
certificate: a bound that no solution can beat. The bound is the product. A solver that returned a
design without a bound would be a heuristic, and there are cheaper ways to build one.

**Fact 2 — The gap is the only honest statement of answer quality.** The relative gap is the
distance between the best design found and the best the solver can prove exists. Every other
quality claim — run time, node count, "it converged", "it looked stable" — is a proxy. The gap is
the thing itself, and it costs one number to report.

**Fact 3 — Most solve difficulty is created before the solver is called.** A loose big-M, an
unbroken symmetry group, an aggregated formulation, a badly scaled objective: these are modelling
choices, and each one multiplies the work required to close the gap. The professional response to a
slow solve is therefore usually a reformulation, not a longer time limit and not a different solver.

### 1.1 The decision this topic is really about

Every network study reaches a moment when somebody asks whether the answer is good enough to act
on. There are only four honest positions, and the professional skill is knowing which one you are
in:

| Position | What you have | What you may say | What you may not say |
|---|---|---|---|
| Proven optimal | Incumbent and bound equal within tolerance | "This is the cheapest design in the model, to within 0.1 per cent" | "This is the right business decision" |
| Proven good | Incumbent, bound, and a gap you chose in advance | "No design is more than 1.5 per cent cheaper than this one" | "This is optimal" |
| Feasible only | An answer with no usable bound | "This design costs 41.9 million and satisfies every constraint" | Anything about how close it is to the best |
| Bounded only | A bound with no feasible answer | "No design can cost less than 33.1 million" | Anything about what to build |

The third and fourth rows are the ones that get misreported, in both directions. A heuristic answer
is routinely presented with the confidence of the first row, and a relaxation bound is routinely
presented as if it described a design. The discipline of this topic is to keep the row and the
sentence aligned.

### 1.2 Where the difficulty actually comes from

A network design model of realistic size has three properties that make it hard, and none is a
property of the solver. It is **combinatorial**: choosing 8 sites from 40 is 76,904,685
combinations before any flow is assigned, so the solver must reason about sets of designs rather
than designs. Its relaxation is **weak**: the linear programming (LP) relaxation allows fractional
facilities, and a fractional facility is much cheaper than a real one, so the bound sits far below
any achievable cost. It is **symmetric and degenerate**: interchangeable sites produce many designs
of identical cost, and the LP has many optimal bases, which inflates the search and makes
incidental outputs such as dual values unstable.

Topic 03 named these as formulation issues. This topic shows what they cost at run time.

### 1.3 What a reader should be able to do afterwards

The test of this topic is a conversation. Somebody presents a network recommendation and says the
model is optimal. You should be able to ask five questions in order — which solver and version,
which termination status, what gap, what wall-clock limit, and was the model the one that was
validated — and to say what each answer would change about how much weight the recommendation
carries. Section 5 turns those five questions into a procedure and section 9 shows what the answers
usually are in practice.

> **Note:** this topic does not teach solver tuning. Parameter settings, cut strategies, and
> branching rules are the solver vendor's domain, they change between versions, and advice about
> them dates quickly. The levers described here are model properties, which are stable, portable
> between solvers, and auditable by a reviewer who has never opened the solver's manual.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** branch and bound as a proof procedure: what the incumbent is, what the bound is, why
   a node is pruned, and why the tree terminates with a certificate rather than with a lucky find.
2. **Compute** a relative gap from an incumbent and a bound, **state** which denominator the
   convention uses, and **explain** why two solvers can report different gaps for the same run.
3. **Read** a solver log and extract the five reporting facts: solver and version, termination
   status, objective, bound, and wall-clock limit.
4. **Diagnose** a slow or unclosed solve into one of five structural causes — loose big-M,
   unbroken symmetry, aggregated formulation, weak relaxation from a missing valid inequality, or
   poor scaling — and **propose** the reformulation that addresses it.
5. **Choose** between direct solution, decomposition, and a heuristic for a described instance, and
   **justify** the choice with instance size, re-solve frequency, and the decision's tolerance for
   error.
6. **Distinguish** the three classical decompositions — Lagrangian relaxation, Benders
   decomposition, and Dantzig-Wolfe (DW) decomposition with column generation — by what they
   relax, what they iterate on, and what engineering effort each demands.
7. **State** what a heuristic answer can and cannot claim, and **identify** the conditions under
   which reporting one without a bound is professionally defensible.
8. **Design** a measurement run: time to first feasible, time to a stated gap, and both as a
   function of instance size, plotted rather than asserted.
9. **Write** a termination statement to a standard a reviewer can audit, and **critique** one that
   omits the gap or the limit.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is the solve as a pair of converging lines with a shaded region
between them. The second is a decision tree for choosing how hard to try.

### 3.1 The gap closing over time

At every moment of a solve there are two numbers. The **incumbent** is the cost of the best feasible
design found so far: an achievable answer, and therefore an upper bound on the optimum for a
minimisation. The **bound** is the cheapest cost the relaxation can still prove possible: a floor,
and therefore a lower bound. The optimum lies between them. The distance between them, relative to
the incumbent, is the **gap**.

```text
  cost
  EUR m
   47 |  I
   46 |  I
   45 |  I   I
   44 |  :   I   I
   43 |  :   :   I   I   I
   42 |  :   :   :   :   :   I   I   I   I
   41 |  :   :   :   B   B   B   B   B   B   B   B
   40 |  :   B   B
   39 |  B
   38 |  B
   37 |  B   B
   36 |  B
   35 |  B
   34 |  B
   33 |  B
      +------------------------------------------------->  time (log scale)
         0   6s  20s  1m   5m   20m  1h   2h   4h

     B = best proven bound (floor)      I = incumbent (best feasible design)
     : = the gap between them, which is what the solver is working to close
```

Read the picture from left to right. The bound starts at the value of the LP relaxation, which for
a fixed-charge model is far below anything achievable: 33.1 against a true optimum near 41.9 in the
illustration of section 6. The incumbent does not exist until the first feasible design is found,
which is why the `I` line starts to the right of the `B` line rather than at the axis. Then the
incumbent drops as better designs appear, and the bound climbs as branching rules out regions. The gap is the vertical
distance, and it narrows from both ends.

Three properties of this picture are worth memorising.

**Property 1 — The bound never falls and the incumbent never rises.** They are monotone by
construction. A chart in which either violates monotonicity is not a solve chart; it is a restart, a
change of formulation, or a plotting error. This makes the pair easy to audit: any non-monotone
sequence in a log deserves an explanation.

**Property 2 — Gap improvement arrives in jumps, not smoothly.** A single good incumbent can close
most of the gap in one step. A long stretch with no progress is normal and does not mean the solver
is stuck; it means it is proving, not searching. This is why a time limit is a poor proxy for
quality and why the gap must be read at the moment of termination rather than inferred from the
clock.

**Property 3 — The two lines meet only at the end, and the end is a choice.** For a large instance
they may never meet within any budget you are willing to spend. The professional act is to decide
in advance what gap is acceptable for the decision, and to stop when it is reached. That decision is
about the business, not about the computer.

### 3.2 Choosing how hard to try

```text
                    model built (Topic 03) and validated (Topic 03, section 5.8)
                                        |
                        is the root LP bound within your
                        tolerance of a good known design?
                            /                        \
                         yes                         no
                          |                           |
                   solve the MIP directly      is the structure separable?
                   (seconds to minutes)          /                  \
                                            yes                     no
                                             |                       |
                                     Benders / Dantzig-Wolfe /   reformulate first:
                                     Lagrangian relaxation       tight M, symmetry
                                     (days to weeks of work)     breaking, disaggregation
                                                                         |
                                                              still not closed in budget?
                                                                  /            \
                                                               yes              no
                                                                |                |
                                                        heuristic with a      exact solve,
                                                        reported bound, or    report the gap
                                                        a changed problem
```

The tree is not a flowchart of solver features. It is a flowchart of cost. Direct solution costs
minutes. Reformulation costs hours to days of analyst time. Decomposition costs days to weeks,
because each of the three classical methods requires a custom subproblem, a convergence argument,
and a fallback when the master problem misbehaves. A heuristic costs hours and buys an answer with
no proof. The right branch is the cheapest one that produces a claim strong enough for the decision.

A solver proves; a heuristic guesses; a decomposition splits the proof into pieces small enough to
finish; and the analyst's job is to know which of those three sentences is true of the run in front
of them.

## 4. Definitions and vocabulary

The vocabulary of this topic is small and heavily overloaded. Almost every term below has a precise
meaning in the solver's documentation and a looser meaning in a meeting, and the looser meaning is
usually the one that reaches the capital paper.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| MIP | Mixed-integer program: a mathematical program in which some variables are restricted to integer values | Used for any optimisation model, including nonlinear and constraint-programming ones | Model class, no unit |
| MILP | Mixed-integer linear program: a MIP with a linear objective and linear constraints | Used interchangeably with MIP when a nonlinear term is present | Model class, no unit |
| LP | Linear program: continuous variables, linear objective and constraints | Called "the model" when it is only the relaxation | Model class, no unit |
| LP relaxation | The MIP with all integrality restrictions removed | Treated as an approximate answer rather than as a bound | Objective value in the model currency |
| Bound | The best objective value the relaxation can still prove achievable; a floor for a minimisation | Reported as if it were a cost somebody could incur | Currency per period |
| Dual bound | The bound as computed from the dual side; in most solvers the same number as "bound" | Confused with a dual value or shadow price | Currency per period |
| Primal bound | The incumbent's objective, seen as an upper bound on the optimum in a minimisation | Used as a synonym for the optimum | Currency per period |
| Incumbent | The best feasible integer solution found so far | Called "the answer" before termination | Currency per period |
| Optimality gap | Absolute distance between incumbent and bound | Reported without saying whether it is absolute or relative | Currency per period |
| Relative gap | The gap divided by a denominator, usually the incumbent | Assumed to be defined identically by every solver | Dimensionless proportion, e.g. 0.015 |
| Absolute gap | Incumbent minus bound, with no division | Used when the objective is near zero, which makes the relative gap meaningless | Currency per period |
| Root gap | The relative gap at the root relaxation, before any branching | Compared across formulations without noting that it is formulation-specific | Dimensionless proportion |
| Branch and bound (B&B) | The tree search that partitions the feasible set and uses relaxation bounds to discard regions | Treated as a synonym for "solving" rather than one algorithm among several | Algorithm, no unit |
| Node | One subproblem in the B&B tree, defined by added bound restrictions | Confused with a node of the supply network, which is a different object entirely | Tree node identifier |
| Branching | Splitting a node into disjoint subproblems on a variable | Assumed to be the modeller's choice; it is usually the solver's | Variable plus bound change |
| Pruning / fathoming | Discarding a node because its bound cannot beat the incumbent, or because it is infeasible, or because its relaxation is integral | Described as "the solver giving up" | Node disposition |
| Presolve | The solver's pre-processing: removing redundant rows, fixing variables, tightening bounds | Assumed to be cosmetic; it frequently changes the model more than the analyst did | Rows and columns removed |
| Cutting plane | A valid inequality added to tighten the relaxation without removing any integer feasible point | Confused with a constraint of the business model | Row added |
| Valid inequality | Any inequality satisfied by every feasible integer solution | Assumed to be a modelling assumption rather than a proven consequence | Mathematical statement |
| Warm start | Supplying an initial feasible solution or basis to shorten the search | Assumed to guarantee a faster solve; it can slow one down | Initial solution or basis |
| MIP start | A specific warm start in which a complete integer solution is handed to the solver | Treated as a bound; it supplies an incumbent and nothing else | Vector of variable values |
| Symmetry | A permutation of variables that maps every feasible solution to a feasible solution of equal cost | Ignored because the sites "look different" on a map | Count of equivalent solutions |
| Symmetry breaking | Constraints that select one representative from each symmetric class | Applied to sites that are not truly interchangeable, cutting off the optimum | Constraint set |
| Orbital fixing | A solver-side symmetry technique that fixes variables known to be equivalent to an already fixed one | Confused with a modelling constraint; it is an algorithm | Variable fixings |
| Big-M | The constant in a linking constraint, equal to the largest flow the binary permits | Set to a round number or to a historical volume | Flow units per period |
| Disaggregation | Writing one aggregate constraint as several tighter ones, usually one per lane or per market | Assumed to change the integer feasible set; it does not | Constraint count |
| Coefficient scaling | The spread of magnitudes among coefficients in a row or the objective | Left unchecked until the solver reports numerical trouble | Ratio, largest to smallest |
| Numerical tolerance | The solver's threshold for calling a value zero or a constraint satisfied | Assumed to be exact arithmetic | Absolute value in model units |
| Wall-clock limit | The elapsed-time budget after which the solver stops | Confused with CPU time, which differs under parallel search | Seconds |
| Node limit | The budget in B&B nodes after which the solver stops | Used as a proxy for effort across instances of different size | Count of nodes |
| Termination status | The solver's statement of why it stopped | Read as a quality statement | Enumerated status |
| Time to first feasible | Elapsed time until the first incumbent is found | Not measured, so never reported | Seconds |
| Time to gap | Elapsed time until the gap first falls to a stated tolerance | Replaced by total run time, which includes the proof | Seconds |
| Decomposition | Solving a large problem by iterating between a master problem and one or more subproblems | Used loosely for "breaking the model into pieces" without an iteration | Method class, no unit |
| Lagrangian relaxation | Relaxing a set of constraints into the objective with multipliers, producing a bound from an easier subproblem | Described as an approximation; the relaxed problem is a bound, not a plan | Bound in model currency |
| Lagrangian multiplier | The price attached to a relaxed constraint, updated iteratively | Confused with a market price or a shadow price | Currency per unit of the relaxed constraint |
| Subgradient method | The standard procedure for updating Lagrangian multipliers | Assumed to converge in a fixed number of iterations; it does not | Update step size, dimensionless |
| Benders decomposition | Partitioning variables into complicating integer decisions and a continuous subproblem, iterating cuts between them | Described as "solving the MIP in pieces" without mentioning the cut loop | Method, no unit |
| Master problem | In Benders, the integer program over the complicating variables plus accumulated cuts | Confused with the original model | Integer program |
| Optimality cut | A Benders cut that bounds the subproblem value from below as a function of the master's decisions | Confused with a feasibility cut, which excludes an infeasible master solution | Row added |
| Feasibility cut | A Benders cut that excludes a master solution for which the subproblem is infeasible | Omitted, which makes the algorithm cycle | Row added |
| Dantzig-Wolfe (DW) decomposition | Reformulating the problem over the extreme points of a subproblem's feasible region and generating them lazily | Confused with Lagrangian relaxation, which bounds the same reformulation's dual | Method, no unit |
| Column generation | The procedure that generates DW columns by solving a pricing problem | Treated as exact when embedded in a heuristic rounding scheme | Iteration scheme |
| Pricing problem | The subproblem that finds a column with negative reduced cost | Assumed to be easy; it is often a knapsack or shortest-path problem | Optimisation subproblem |
| Branch and price | B&B in which the bound at each node comes from column generation | Described as "column generation", which is only its bounding step | Algorithm |
| Heuristic | A procedure that searches for a good feasible solution without a guarantee of optimality | Used to mean "approximate method", which includes approximation algorithms with guarantees | Method class, no unit |
| Greedy construction | Building a solution by repeatedly making the locally best choice | Assumed to produce a good design; it commonly opens too many sites | Algorithm, no unit |
| Local search | Improving a solution by repeated small modifications | Described as an optimisation, which overstates what it proves | Algorithm, no unit |
| Savings algorithm | A construction that starts from a degenerate solution and merges pairs by descending saving | Applied outside routing, where the saving is not well defined | Algorithm, no unit |
| Matheuristic | A heuristic that uses a MIP solver on subproblems inside a search framework | Called a heuristic when the subproblem solve is the expensive part | Method class, no unit |
| Rolling horizon | Re-solving a shorter window repeatedly as time advances | Assumed to be equivalent to solving the full horizon | Time window in periods |
| Aggregation | Combining products, markets, or periods into fewer modelled entities | Treated as a data-reduction step rather than a modelling decision | Count of aggregated entities |
| MIPLIB | The Mixed Integer Programming Library: a curated public set of real-world MIP instances used to benchmark solvers | Cited as evidence that a method works on supply chain models, which it does not show | Benchmark library, no unit |

Six of these deserve expansion, because they are the ones most often used to end a conversation
that should continue.

**Bound, incumbent, gap.** These three are the whole of solution quality. The bound answers "how
good could any design be". The incumbent answers "how good is the design I have". The gap answers
"how much is still unknown". A report that gives the incumbent and not the bound has given a cost,
not a result. A report that gives the bound and not the incumbent has given a floor, not a plan.

**Relative gap, and why two solvers differ.** The relative gap is a ratio, and the denominator is a
convention. One common definition divides by the incumbent; another divides by the smaller in
absolute value of the bound and the incumbent; another adds a small constant to the denominator to
avoid division by zero. For the same run, two solvers can print gaps that differ in the second
decimal place, and both are correct under their own definitions. When you compare gaps across runs,
compare runs from the same solver, or recompute the ratio yourself from the two raw numbers.

**Root gap.** This is the gap at the root node, using the best known incumbent rather than one the
search has found. It is a property of the formulation and it is available after one LP solve. A
model with a 5 per cent root gap usually closes; a model with a 30 per cent root gap may not close
at any budget. It is the single most useful diagnostic in this topic because it is cheap and it
predicts.

**Presolve.** Modern solvers spend a meaningful fraction of a short run removing redundant rows,
tightening bounds, and fixing variables. For a network model this can be dramatic: a slack capacity
row, a lane that cannot be used because its cost exceeds any alternative, a market that one site
must serve. Presolve is why a model that looks large in the file can solve in seconds, and it is
also why the row count in the log differs from the row count in the model card.

**Symmetry.** If two candidate sites have identical fixed cost, identical capacity, and identical
transport cost to every market, then any design using one is matched by a design using the other.
A group of `k` such sites contributes `k` factorial equivalent designs. Four identical sites give
24, six give 720, eight give 40,320. The solver must, in the worst case, examine each of them.

**Heuristic.** A heuristic is not "a bad method". It is a method with a different contract: it
promises a feasible answer within a budget, and it promises nothing about distance from the
optimum. That contract is often exactly what a planning cycle needs. It is not what a capital
committee needs, unless the heuristic is paired with a bound from somewhere else.

> **Note:** the word "optimal" is reserved, in this topic and in any serious report, for a run that
> terminated with the gap at or below a stated tolerance. A run stopped by a time limit has produced
> a feasible design and a bound, and the design may well be the best one; the word does not follow
> from that.

## 5. The mechanics, step by step

This section follows a single model from the moment it is handed to a solver to the moment a
sentence about it is written. Each phase states what the solver does, what the analyst should read,
and the branch that decides what to do next.

### Phase A — Before the tree: presolve and the root

**Step 1. Read the log before reading the answer.**

- What happens: the solver applies presolve, builds the matrix, and solves the root LP relaxation.
- What to read: the number of rows and columns after presolve against the model card; the root LP
  objective; the number of fractional binaries at the root; the time to the root.
- Branch: if presolve removed a large fraction of the model, find out why before trusting anything.
  A row that presolve deletes as redundant is a row that was not binding, which may be correct or
  may be a modelling error that made it non-binding.
- Failure if skipped: the analyst interprets a solve on a model materially different from the one
  documented.

**Step 2. Compute the root gap.**

- Input: the root LP bound and any good feasible design, which may be the current network costed by
  the model, a greedy construction, or last year's plan.
- Output: `root gap = (incumbent - root bound) / incumbent`.
- Interpretation, for fixed-charge network models: below 5 per cent, the instance is usually easy;
  5 to 15 per cent, expect minutes to hours; above 25 per cent, expect the tree to be the problem
  and the formulation to be the lever.
- Branch: if the root gap is above roughly 20 per cent, do not extend the time limit. Work through
  Phase C and reformulate.

**Step 3. Decide the target gap from the decision, not from the solver.**

- The question is: how much error can this decision tolerate? A capital project with a 3-year
  payback and a 5 per cent hurdle can absorb a 1 per cent gap without changing its sign. A
  lease-versus-build choice between two options 0.4 per cent apart cannot: a 1 per cent gap means
  the model cannot tell the options apart.
- Practical convention in network studies: 1 per cent for a design that will be built; 2 to 5 per
  cent for a directional comparison across scenarios; 10 per cent only for screening a long list.
- Write the number down before the run. Choosing it afterwards is how a 12 per cent gap becomes
  "essentially optimal".

### Phase B — The proof machine

**Step 4. The tree starts from the root relaxation.**

The root node is the LP relaxation. It is feasible in the fractional sense, so the solver branches:
it picks a fractional binary, creates two children with `z <= 0` and `z >= 1`, and re-solves. Every
node inherits its parent's constraints plus one, so every node's bound is at least its parent's.

**Step 5. Incumbents arrive from three sources.**

- A rounding heuristic applied to the root or to a node's LP solution.
- A dive: the solver follows one branch to a leaf, fixing variables as it goes, and evaluates the
  resulting design.
- A warm start or MIP start supplied by the analyst.

The first incumbent matters more than any other, because every prune is measured against it. A
study that supplies a greedy design as a MIP start often cuts time to a usable answer by an order of
magnitude, and this is the cheapest performance intervention available.

**Step 6. Pruning is the whole algorithm.**

A node is discarded for one of exactly three reasons, and each has a name in the log:

| Reason | Condition (minimisation) | What it proves |
|---|---|---|
| Bound | Node bound is at or above the incumbent | No design in this subtree can beat the one you have |
| Infeasibility | The node's relaxation has no feasible point | No design in this subtree exists at all |
| Integrality | The node's LP solution happens to be integral | This node's best design is its bound, and it is feasible |

The tree is finished when no nodes remain. At that moment the best remaining bound equals the
incumbent, the gap is zero, and the answer is proven optimal. The proof is not that the solver
looked everywhere; it is that every region it did not look at was shown to be no better.

**Step 7. Read the termination status, not the objective.**

Every solver returns a status, and the statuses fall into four families. The wording differs by
product; the families do not.

| Status family | Meaning | What may be claimed |
|---|---|---|
| Optimal | Gap closed to the solver's tolerance | The design is optimal to that tolerance, which should be stated |
| Gap or objective limit | Stopped because the gap fell to the tolerance the analyst set | The design is within that gap of the best possible |
| Time or node limit | Stopped by a budget | A feasible design exists and a bound exists; the gap is whatever it was at the stop |
| Infeasible or unbounded | No feasible point, or an unbounded objective | A model or data finding that must be triaged before it is a business finding |

The third row is the one that produces bad papers. A run stopped at a 2-hour limit with a 6 per cent
gap has produced a real result; a report that calls it optimal has produced a false one.

### Phase C — Why the tree is slow: five structural causes

Work through these in order. Each has a diagnostic available before the run finishes, and each is
fixed in the model rather than in the solver.

**Step 8. Cause 1 — a loose big-M.**

- Diagnostic: compare every M to the bound derived from capacity (Topic 03, Step 12). A ratio above
  2 is a flag; above 10 is a defect.
- Mechanism: in the relaxation, `x <= M z` gives `z >= x/M`. A large M lets `z` sit near zero while
  the flow is large, so the relaxation pays a small fraction of the fixed cost and the bound falls.
- Fix: replace M with the tightest valid value, and disaggregate the linking row per lane where the
  aggregate is much weaker.

**Step 9. Cause 2 — unbroken symmetry.**

- Diagnostic: count the groups of candidate sites that are identical in every parameter, compute
  `k` factorial per group, and multiply.
- Mechanism: the solver explores permutations of the same design, and its symmetry-blind branching
  rules make no progress on any of them.
- Fix: add ordering constraints over each genuinely identical group, for example
  `z[a] >= z[b] >= z[c]`. This is valid only when the sites are interchangeable in the business,
  not merely in the cost table.

**Step 10. Cause 3 — an aggregated formulation.**

- Diagnostic: the root bound is far below a design the analyst believes is near-optimal, and the
  number of fractional binaries at the root is large.
- Mechanism: one aggregate capacity row and one aggregate linking row per site leave the
  relaxation free to satisfy demand with a thin smear of many facilities.
- Fix: disaggregate. One row per lane, one row per market, one row per period. The integer feasible
  set is unchanged; the relaxation tightens, sometimes by a large factor.

**Step 11. Cause 4 — a missing valid inequality.**

- Diagnostic: the LP bound is far below the cost of the trivial "open everything" design, which is
  itself a valid upper bound.
- Mechanism: the relaxation does not know any of the structural facts the analyst knows, such as
  "total capacity opened must cover total demand".
- Fix: add the constraints that express those facts. A total-capacity row, a flow-cover row, or a
  single-period relaxation of the multi-period model all tighten the bound and none of them changes
  the set of integer designs.

**Step 12. Cause 5 — poor scaling.**

- Diagnostic: the ratio of the largest to the smallest nonzero coefficient in the objective, or in
  any row, spans many orders of magnitude.
- Mechanism: LP solvers work with tolerances relative to coefficient magnitude, so a variable worth
  one unit becomes invisible beside a coefficient worth a million.
- Fix: choose units so that flows, costs, and fixed charges are within a few orders of magnitude of
  one another. A model in euros and cases and a model in millions of euros and thousands of cases
  are different numerical objects, and the second is usually the better one.

### Phase D — Decomposition: three methods and their costs

Decomposition is the principled response when the model is too large or too weak to close directly.
Each method splits the problem along a different seam, and each carries an engineering cost that is
routinely underestimated.

**Step 13. Lagrangian relaxation.**

- What it does: moves a set of constraints into the objective, priced by multipliers. The remaining
  problem is easier — often separable into one small problem per facility or per market — and its
  optimal value is a valid bound for any multiplier vector.
- What you iterate: the multipliers, usually by subgradient steps, with a step size that shrinks as
  the bound improves.
- What it gives: a bound, and usually a good one. The bound from the Lagrangian dual is at least as
  strong as the LP bound and often much stronger for facility location.
- What it does not give: a design. The relaxed solution generally violates the relaxed constraints,
  and recovering a feasible design requires a separate heuristic step.
- Engineering cost: a custom subproblem solver, a step-size schedule that works on your instances,
  a feasibility-repair procedure, and a stopping rule for the multiplier iteration. Days to weeks.
- When it is the right seam: the constraints that couple facilities to markets are the difficult
  ones, and the per-facility subproblem is easy. This is the classical case for capacitated
  facility location.

**Step 14. Benders decomposition.**

- What it does: partitions the variables into complicating integer decisions and a continuous
  recourse problem. The master problem chooses the integer decisions with a surrogate for the
  recourse cost; the subproblem evaluates the true recourse for a given master solution and returns
  a cut.
- What you iterate: master and subproblem. Each subproblem solve adds one optimality cut, and, when
  the subproblem is infeasible, one feasibility cut. Omitting feasibility cuts is the standard way
  to make the method cycle.
- What it gives: an exact method when it converges, with a bound at every iteration.
- Engineering cost: a correct dual formulation of the subproblem, cut management, and attention to
  the tail, where cuts stop helping and the master becomes slow. Weeks.
- When it is the right seam: the integer decisions are few relative to the continuous flow, and the
  subproblem is a well-behaved LP, typically a multicommodity flow. This is the classical case for
  distribution system design with many commodities.

**Step 15. Dantzig-Wolfe decomposition and column generation.**

- What it does: rewrites the problem over the extreme points of a subproblem's feasible region, then
  generates only the columns that can improve the master, using a pricing problem.
- What you iterate: restricted master and pricing. When no column has negative reduced cost, the
  master is optimal for the relaxation.
- What it gives: a bound that is typically much stronger than the plain LP bound, because the
  subproblem's integrality is enforced inside the column definition.
- Engineering cost: a pricing algorithm, stabilisation (plain column generation converges slowly and
  erratically near the optimum), and, if you want an integer answer, branch and price. Weeks to
  months.
- When it is the right seam: the problem has a natural block structure, such as a set of facilities
  each with its own internal configuration, and the master only needs to choose among
  configurations.

**Step 16. The honest cost comparison.**

| Method | Typical implementation effort | What you must build yourself | Failure mode |
|---|---|---|---|
| Direct solve | Hours | Nothing beyond the model | Tree too large; gap never closes |
| Reformulate, then direct solve | Hours to days | Tighter rows, symmetry constraints | Still too large |
| Lagrangian relaxation | Days to weeks | Subproblem solver, step schedule, repair | Bound good, no design, or slow multiplier convergence |
| Benders | Weeks | Subproblem duals, cut loop, stabilisation | Tail convergence; master explodes |
| Dantzig-Wolfe and branch and price | Weeks to months | Pricing, stabilisation, branching scheme | Pricing too slow; branching breaks the structure |

The table is the argument for trying reformulation first. A day spent tightening M and breaking
symmetry routinely removes the need for a week spent on a decomposition, and it produces a model a
reviewer can still read.

> **Note:** decomposition methods are not merely faster solvers. They change what the model can
> represent, because the subproblem can carry structure that a flat MIP cannot express
> efficiently. That is their real advantage, and it is also why they are hard to retrofit onto a
> model that was written flat.

### Phase E — Heuristics, and what they are allowed to claim

**Step 17. The constructions that are actually used.**

Most heuristics in network studies are not general-purpose search algorithms. They are
problem-specific constructions with a search wrapped around them, and the construction carries most
of their quality.

- **Greedy facility opening with local search.** Open sites one at a time, each time choosing the
  site that most reduces total cost given the sites already open, until no addition helps. Then run
  a local search: try closing each open site, try swapping an open site for a closed one, accept
  improvements. The construction typically opens too many sites, and the closing pass is where the
  quality comes from.
- **Savings-based assignment.** Start from a degenerate solution in which every market is served by
  its own dedicated facility, then merge pairs of assignments in descending order of saving, subject
  to capacity. This is the logic of the classical savings algorithm, transplanted from vehicle
  routing to facility assignment, where the "saving" is the fixed and transport cost avoided by
  consolidating.
- **Lagrangian heuristics.** Solve the relaxed problem, then repair its solution into a feasible
  design by opening the facilities the relaxation used and assigning markets greedily. The repair is
  guided by the multipliers, and the bound from the relaxation is available at every iteration. This
  is the only family of heuristics that comes with a bound attached, which is why it dominates the
  facility-location literature.
- **Matheuristics.** Fix the design decisions found by a heuristic and re-solve the flow problem
  exactly, or fix most of the design and let a MIP solver choose the remainder. The solver is used
  as a subroutine inside a search rather than as the search itself.

**Step 18. What each can and cannot claim.**

| Procedure | Produces a feasible design | Produces a bound | May be called optimal |
|---|---|---|---|
| Greedy plus local search | Yes | No | No |
| Savings-based assignment | Yes | No | No |
| Lagrangian heuristic | Yes, after repair | Yes, from the relaxation | No, unless the bound meets the design |
| Matheuristic | Yes | Only if the subproblem bounds are combined correctly | No |
| Relaxation only | No | Yes | Not applicable |
| MIP solver to a stated gap | Yes | Yes | Yes, to that gap |

Two of these rows are asymmetric in a way that matters. A heuristic gives an answer with no
certificate. A relaxation gives a certificate with no answer. Neither is a solution to a capital
decision on its own, and the pair together is often as useful as an exact solve: "the best design we
found costs 41.9 million, and no design can cost less than 41.3 million" is a strong statement even
though it is not a proof of optimality.

**Step 19. When a heuristic is the professional answer.**

- **The instance is far beyond exact solvability** and the decision cannot wait for a decomposition
  to be built. A 4,000-market, 300-candidate instance with a monthly planning cycle is this case.
- **The model is re-solved repeatedly** under rolling conditions, where the question is the
  trajectory of decisions rather than any single answer. A heuristic that runs in 40 seconds and is
  re-run every night is more useful than an exact method that runs in nine hours.
- **The work is interactive.** Scenario exploration in a workshop, where a planner changes a
  capacity and expects an answer before the conversation moves on, is a heuristic setting by
  construction. The bound is unavailable and the planner knows it.
- **The decision has a wide tolerance.** Screening 40 candidate sites down to 8 does not require a
  0.1 per cent gap. A 3 per cent answer changes nothing about which 8 survive.

**Step 20. When a heuristic is a shortcut that hides something.**

- **It is used because the model does not solve**, and nobody has diagnosed why. The solver is
  telling you something about the formulation. Replacing it with a heuristic silences the message.
- **It is used to avoid a validation.** A heuristic cannot fail a reconstruction test the way a
  solver can, because it has no baseline to reproduce.
- **Its answer is reported with the vocabulary of proof.** "Optimised network" and "optimal
  solution" attached to a local search output is the single most common misreporting in this field.
- **It is used on a decision with a narrow tolerance.** If two options differ by 0.4 per cent and
  the heuristic has no bound, the heuristic cannot distinguish them, and its choice between them is
  noise.

### Phase F — Changing the problem

**Step 21. The fourth option, which is often the right one.**

When a model will not close, there is an option between "try harder" and "guess": change the
problem so that it is smaller, and say so. The three standard reductions are:

| Reduction | What you do | What it buys | What it costs |
|---|---|---|---|
| Aggregate markets | Combine demand zones into regions, typically 180 into 40 | Fewer assignment variables and a much tighter relaxation | Answers at region level; intra-region cost differences disappear |
| Restrict the candidate set | Screen 40 sites to 15 on pre-specified criteria before modelling | Search space falls combinatorially | The screen, not the model, decides what can be chosen |
| Restrict the horizon | Model 12 representative weeks instead of 52 | Fewer period indices and links | Seasonality and end-effects are approximated |

All three change the answer, and all three are legitimate when the change is disclosed. The
professional failure is not aggregation; it is aggregation reported as if the model had considered
what was aggregated away. A study that says "markets are aggregated to 40 regions, which is where
the transport rates are contracted" has made a defensible choice. A study that says "the model
considered 180 markets" when it considered 40 has made a false statement.

### Phase G — Measurement and reporting

**Step 22. Measure three numbers, not one.**

- **Time to first feasible.** How long until any design exists. This is what an interactive user
  experiences and what a rolling re-solve depends on.
- **Time to the stated gap.** How long until the design is provably within the tolerance the
  decision needs.
- **Time to close.** How long until the tree empties. Usually much longer than the second number
  and rarely worth paying for.

Plot both of the first two against instance size. The curves are the evidence that a method is
adequate; a single run time is an anecdote.

**Step 23. Report five facts.**

Every reported solve carries: the solver and its version; the termination status in the solver's
own words; the objective; the bound; and the wall-clock limit or gap tolerance that ended the run.
With those five, a reviewer can reconstruct the claim. Without them, the claim is unfalsifiable,
which is convenient and unprofessional.

**Step 24. State which of the three the answer is.** Optimal, proven-good to a stated gap, or
merely feasible. The third is common, the first is rare, and the middle is where most defensible
studies live.

## 6. Worked example

Meridian Health Systems distributes three product families through a national network. The study
has 4 plants, 24 candidate distribution centres, 180 demand zones, and one representative year. The
model is the Topic 04 formulation extended to three commodities as in Topic 05. All figures below
are constructed to be internally consistent and arithmetically checkable; they are not a benchmark
result and should not be cited as one.

### 6.1 The instance

| Element | Count | Note |
|---|---:|---|
| Plants | 4 | Fixed capacity, one product mix |
| Candidate DCs | 24 | 18 existing, 6 greenfield options |
| Demand zones | 180 | Aggregated from 1,140 ship-to addresses |
| Product families | 3 | Ambient, chilled, controlled |
| Continuous variables | 25,920 | Plant to DC and DC to zone, by family |
| Binary variables | 24 | One per candidate DC |
| Rows before presolve | 26,150 | Demand, balance, capacity, linking |

The model is small by commercial standards and large enough that the gap is the interesting object.

### 6.2 The solve, as logged

The run below used a 4-hour wall-clock limit and a 0.1 per cent gap tolerance. The best design it
found costs EUR 41.86 million per year.

| Time | Incumbent (EUR m) | Bound (EUR m) | Gap | Event |
|---:|---:|---:|---:|---|
| 0 s | none | 33.10 | undefined | Root LP solved; 17 of 24 binaries fractional |
| 6 s | 46.10 | 33.42 | 27.5% | First feasible design, from the rounding heuristic |
| 20 s | 44.02 | 35.80 | 18.7% | Second incumbent; first improvement |
| 47 s | 42.94 | 38.65 | 10.0% | A 10 per cent tolerance would stop here |
| 6 min | 42.31 | 40.95 | 3.2% | Bound climbing steadily |
| 48 min | 41.98 | 41.60 | 0.9% | Within 1 per cent |
| 4 h | 41.86 | 41.82 | 0.10% | Stopped by the gap tolerance |

Three things in this table are worth more than the final number.

**Time to first feasible is 6 seconds; time to a usable answer is 47 seconds; time to a defensible
answer is 48 minutes.** These differ by three orders of magnitude, and a report that quotes one
number — usually the last — describes a process nobody ran.

**The incumbent moves in three jumps.** 46.10 to 44.02 to 42.94, then a long slow tail. Most of the
value was found in the first minute. The remaining three hours and fifty-nine minutes bought a
proof, not a better design, and the proof is what the capital paper needs.

**The bound is doing the work after minute one.** From 20 seconds onward the incumbent barely moves
and the bound climbs from 35.80 to 41.82. A reader who watches only the incumbent concludes the
solver is idle. It is closing the gap from the other side.

### 6.3 What the early stop costs

The 10 per cent tolerance stopped the run at 47 seconds with a design costing 42.94 million. The
4-hour run found 41.86 million.

```text
  difference            = 42.94 - 41.86 = 1.08 EUR m per year
  relative to best      = 1.08 / 41.86  = 2.58%
  ten-year undiscounted = 10.8 EUR m
```

The early answer satisfies every constraint and costs 2.58 per cent more than the design the longer
run found. Both facts hold at once: the 10 per cent gap was honestly reported, and the design
happened to sit near the good end of that band. This is why the gap is the honest statement and the
objective is not. A 10 per cent gap means the truth lies somewhere in a 4.3 million band, and
nothing in the run tells you whether your design is near the top of it or the bottom.

### 6.4 Four formulations of the same model, plus a warm start

The same business problem, written four ways, then started from a good design. None of the four
changes the set of integer designs; they change only what the relaxation can prove.

| Formulation | Root bound (EUR m) | Root gap | Nodes to 1 per cent | Time to 1 per cent |
|---|---:|---:|---:|---:|
| Aggregate big-M, one linking row per DC | 33.10 | 20.9% | 41,000 | 6 h 20 m (not reached) |
| Tight M from `min(plant cap, DC cap, zone demand)` | 35.60 | 15.0% | 12,400 | 52 min |
| Plus one linking row per plant–DC lane | 37.20 | 11.1% | 5,900 | 21 min |
| Plus ordering constraints on 6 interchangeable sites | 37.20 | 11.1% | 1,100 | 4 min 40 s |
| Plus a greedy design as a MIP start | 37.20 | 11.1% | 1,100 | 2 min 10 s |

The first row is the formulation most hand-built models use. It is not wrong; it is weak, and the
weakness costs a day. The symmetry constraints in the fourth row do not improve the bound at all —
the bound is a property of the relaxation and the symmetry is a property of the tree — and they cut
the node count by a factor of five. The MIP start does not change the bound and halves the time
again, because it supplies a better incumbent for pruning.

> **Note:** the row labels "plus" are cumulative. Each change is applied on top of the previous
> one, which is why the bound stops improving after the third row and the time keeps falling. This
> is the normal pattern: bound improvements come from the relaxation, time improvements come from
> the search.

### 6.5 The same instance under rolling conditions

Meridian re-plans monthly, with updated demand and freight rates. Re-solving the 4-hour run twelve
times a year is not an option for a planning team that needs the answer on the second working day.

| Mode | Time per re-solve | Answer quality | Suitable for |
|---|---:|---|---|
| Full solve to 0.1 per cent | 4 h | Optimal to tolerance | Annual network study |
| Solve to 1 per cent | 48 min | Proven within 1 per cent | Quarterly review |
| Warm start from last month, 10 per cent tolerance | 40 s | Within 10 per cent, usually much closer | Monthly planning |
| Greedy plus local search | 3 s | 2 to 6 per cent above best, no bound | Interactive workshop |

The third row is the professional compromise, and it is not a heuristic: it is the same solver,
started from last month's design, stopped at a tolerance the planning decision can absorb. The
warm start is what makes 40 seconds possible, and it is legitimate because last month's design is a
feasible design for this month's slightly different data.

## 7. Code, query, or calculation

The instance is deliberately small enough to trace by hand: two candidate DCs, one market, four
variables. It shows the mechanism that every solver, at every scale, is running.

```text
min  40 z1 + 30 z2 + 3 y1 + 5 y2
s.t. y1 + y2 = 10        demand, kcases
     y1 <= 8 z1          capacity and link, DC1
     y2 <= 8 z2          capacity and link, DC2
     y >= 0,  z binary

node 1  root       y1=8, y2=2, z=(1, 0.25)   40+7.50+24+10 = 81.50   bound 81.50
                   branch on z1
node 2  z1 = 0     y1 = 0, y2 = 10 > 8                            INFEASIBLE
node 3  z1 = 1     y1=8, y2=2, z2=0.25       40+7.50+24+10 = 81.50   bound 81.50
                   branch on z2
node 4  z2 = 0     y2 = 0, y1 = 10 > 8                            INFEASIBLE
node 5  z2 = 1     y1=8, y2=2, z=(1,1)       40+30+24+10   = 104.00  incumbent
                   LP solution integral   ->  node bound 104.00, pruned
open nodes: none
global bound 104.00, incumbent 104.00
gap = (104.00 - 104.00) / 104.00 = 0.0000     proven optimal, 5 nodes
```

Five lines of arithmetic carry the whole of branch and bound. The root bound of 81.50 is what the
relaxation can prove when a facility may be 25 per cent open. The two infeasible children are
pruned for a reason that has nothing to do with cost. The last child is pruned because its
relaxation is already integral, so there is nothing left to branch on. When the open-node list
empties, the incumbent and the global bound coincide, and the gap closes at zero.

Now the same model with a 10 per cent gap tolerance and a 1-second limit. The tolerance never binds,
because the gap jumps from 21.6 per cent at the root to zero at node 5; there is no intermediate
state in which the solver could stop. On the tiny instance the tolerance is inert, and that is the
first lesson about early stopping: it is a switch that only matters once the tree has enough nodes
for a gap to decline gradually.

The second run is therefore on the section 6 instance, where a gradual decline exists:

| Run | Tolerance | Stop | Incumbent | Bound | Gap | Distance from best known |
|---|---:|---|---:|---:|---:|---:|
| A | 0.1% | 4 h | 41.86 | 41.82 | 0.10% | 0.00% |
| B | 10% | 47 s | 42.94 | 38.65 | 10.0% | 2.58% above |

Run B is a correct solve of the same model. It is not a worse model and not a worse solver. It is a
different contract with the reader: it promises a design within 10 per cent of optimal and delivers
one that happens to be within 2.6 per cent. The number 2.58 per cent was knowable only afterwards,
which is why the gap of 10.0 per cent is the number that belongs in the report.

> **Note:** the 2.58 per cent is not a general rule about what early stopping costs. On some
> instances an early stop lands on the optimum; on others it can land 8 per cent out. What is
> general is that the gap tells you the width of the band and not your position in it.

## 8. Data, standards, and regulatory context

No regulator tells you which solver to use or what gap to accept. What exists instead is a set of
conventions that make a solve auditable, plus a documentation discipline borrowed from model risk
management. The distinction matters: the first is voluntary and the second is often contractual or
supervisory.

| Item | Governs | What it constrains | Grade |
|---|---|---|---|
| MPS format | Model interchange, in use since the 1960s | The model file a reviewer can read without your code | B |
| LP format | A more readable text model format | Human review of the constraint rows | C |
| MIPLIB | A curated public benchmark library of real-world MIP instances | What "a hard instance" means in the solver literature | A |
| Solver log | The record of a run | Status, objective, bound, gap, node count, time | B |
| ISO/IEC/IEEE 60559 | Floating-point arithmetic | The tolerances inside every numerical claim | B |
| Model risk guidance | Supervisory expectations for model documentation | Documentation of data, assumptions, limitations, and validation | B |
| Open-source licences (Apache 2.0, EPL, BSD) | Redistribution of solver code | Whether a model built on a solver can be shipped inside a product | B |

**MPS is the audit artefact, not a debugging leftover.** A model written to MPS in a fixed version,
archived with its parameter tables, lets a reviewer read the rows, and lets a future analyst
re-solve the exact model that produced a number in a board paper. Solvers change, defaults change,
and hardware changes; the file does not.

**MIPLIB exists because random instances lie.** The library was created in 1992 by Bixby, Boyd, and
Indovina in response to a specific problem: randomly generated MIPs did not behave like industrial
ones, so algorithms tuned on them did not transfer. Its later editions are curated for diversity,
and its benchmark set is used to compare solvers under a common protocol. Two consequences follow
for a practitioner. First, "this method is fast on MIPLIB" is evidence about MIPLIB, not about your
network model, which has a fixed-charge structure that most benchmark instances do not. Second, the
benchmark culture shaped solver development in a specific direction: instances that close within a
time limit are rewarded, so progress is measured partly in instances solved rather than in gap
closed on instances that remain open.

**Floating point is the reason tolerances exist.** Every constraint, every integrality claim, and
every gap is compared to a tolerance. A model whose coefficients span ten orders of magnitude is
asking the arithmetic to resolve differences smaller than its own precision, and no setting repairs
that.

**Model risk guidance constrains the report, not the algorithm.** Frameworks of the kind issued by
banking supervisors require that a model's data, assumptions, limitations, and validation be
documented, and that the model's output not be used outside its validated domain. A network model
that has never been reconstructed against a historical period (Topic 03) has no validated domain,
and a solver status of `OPTIMAL` does not create one. The two obligations are separate: the solver
certifies the arithmetic, and the modeller certifies the model.

> **Note:** the licensing row is a practical constraint that surprises teams late. A solver that is
> free for research may not be redistributable inside a commercial planning product, and a model
> that is fine in a study can become a licence problem when it is embedded in a workflow. Check the
> licence before the model, not after.

## 9. Industry practice and case evidence

### 9.1 What the classical applications actually report

The published record of large network design studies is unusually specific about instance size and
unusually quiet about solve behaviour, which is itself informative.

| Study | Instance | Method | What the source reports | Grade |
|---|---|---|---|---|
| Geoffrion and Graves (1974), food firm distribution | 17 commodity classes, 14 plants, 45 candidate DCs, 121 customer zones | Benders decomposition | An essentially optimal solution proved with a small number of cuts | A |
| Arntzen, Brown, Harrison, and Trafton (1995), Digital Equipment | Global, multi-commodity, multi-period | Large-scale MILP | The model as the decision instrument for a global supply chain | A |
| Camm et al. (1997), Procter and Gamble | Product sourcing and distribution | MILP with a reconciliation discipline | A validated baseline as the precondition for scenario work | A |
| Klose and Drexl (2005), capacitated facility location | Instances up to 1,000 facilities and 500 customers | Column generation for bounds | Column-generation bounds stronger than LP-plus-cuts bounds | A |
| Görtz and Klose (2011), capacitated facility location | Large CFLP instances | Lagrangian relaxation inside B&B | Large instances solved to optimality with the Lagrangian bound | A |
| Parra Galvez and Borenstein (2023), large-scale network design | Large multi-echelon instance | Two-phase heuristic | Average gap of 0.63 per cent against a literature benchmark | A |

Read the fourth and fifth rows together. Two peer-reviewed papers on the same problem class report
that a decomposition bound is strong enough to solve instances a flat model cannot. That is the
strongest evidence for the decomposition branch of the decision tree, and it comes from the
facility-location literature rather than from a solver vendor.

The sixth row is the counterweight, and the honest reading is not "heuristics are as good as exact
methods". It is that a heuristic's quality is an empirical property of its construction and its
instance family, measured after the fact, carrying no guarantee on the next instance.

### 9.2 The benchmark culture and what it did to solver development

MIPLIB's history is worth knowing because it explains why solver progress is easy to overstate. The
library was created in 1992 with real industrial instances; MIPLIB 2017 was compiled from 5,721
submissions into a collection of 1,065 instances and a benchmark set of 240, selected by solving an
optimisation problem over instance features to achieve diversity. By the time that edition was
published, a substantial number of instances that had been open had been closed. Each closure is a
genuine algorithmic advance.

The overstatement happens when that record is read as "solvers got a thousand times faster, so your
model will now solve". Three qualifications apply. Benchmark instances are selected for diversity
and difficulty, not for resemblance to fixed-charge network design. Solver benchmarks are usually
run with a time limit and scored on instances solved, a metric that rewards closing easy instances
and says nothing about the gap on the ones that stay open. And vendor benchmark publications are
marketing documents, graded D here, whose configurations are chosen by the vendor.

> **Note:** the practical translation is simple. A solver upgrade is worth trying, because it is
> free and sometimes large. It is not a substitute for the formulation work in Phase C, and it will
> not close a model whose root gap is 25 per cent.

### 9.3 What teams actually do

Three patterns recur in review work, and none is a technology problem.

**The gap is in the log and not in the summary.** A run stopped at a two-hour limit with a 6 per
cent gap becomes "optimal" in the slide, because the slide has one number and the log has five.
This is the most common misreporting in the field and the easiest to fix, since the fix is a
template.

**The time limit is chosen before the target gap.** Teams set "two hours" because two hours fits a
meeting, then accept whatever gap results. The professional order is the reverse: choose the gap
the decision needs, then choose the budget that usually achieves it, then report both.

**The reformulation is skipped because the model solves.** A model that closes in four hours is
"working", so nobody spends the day that would make it close in four minutes. The cost recurs at
every re-solve.

## 10. Failure modes and red flags

Each row states a failure, the red flag visible in a meeting or a dataset, the mechanism, the
consequence, and the mitigation.

| Failure | Red flag | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Optimality claimed from a limited run | "Optimal" in the deck, a time limit in the log | The status field is read as a quality statement | Overstated certainty in a capital decision | Report status, objective, bound, gap, and limit together |
| Gap not reported at all | One number on the slide, no interval | The bound is treated as an implementation detail | The reader cannot tell a proof from a guess | Make the gap a required field in the report template |
| Gap compared across solvers | Two runs, two gaps, second decimal place different | Denominators differ by convention | A false precision argument about which run is better | Recompute the gap from the raw incumbent and bound |
| Target gap chosen after the run | The tolerance matches the achieved gap to three decimals | The stopping rule was reverse-engineered | A tolerance that flatters the answer | Fix the tolerance before the run and record it |
| Time limit used as a quality proxy | "It ran for four hours, so it is converged" | Effort is confused with proof | Long runs with poor gaps are accepted | Read time to gap, not total time |
| Formulation left aggregate | Root gap above 20 per cent on a modest instance | One linking row per site instead of per lane | Search multiplies; sensitivity becomes impossible | Disaggregate and re-measure the root gap |
| Symmetry never audited | Node count far above what instance size suggests | Identical sites generate factorial equivalent designs | Study stalls; the model is blamed | Count identical groups, add ordering constraints |
| Heuristic answer reported as optimised | "Optimised network" attached to a local search | The vocabulary of proof applied to a construction | A 3 to 6 per cent error presented as zero | Report the procedure and its measured gap on a test set |
| Decomposition built before reformulation | A custom Benders code for a model that would close with a tight M | Effort applied to the expensive branch first | Weeks spent, model still hard to maintain | Try reformulation first, measure, then decompose |
| Aggregation undisclosed | The report says 180 markets; the model used 40 | The reduction is treated as data preparation | The answer is described at a grain it never had | State every aggregation next to the result |
| Solver upgrade treated as the fix | "Version 12 will solve this" | Version gains are real but bounded by the formulation | Delay without a diagnosis | Measure the root gap before and after the upgrade |
| Warm start assumed to help | Time to first feasible unchanged after a MIP start | A poor starting design can mislead the search | Wasted effort and a false sense of optimisation | Measure with and without; keep it only if it helps |

Two of these deserve a note. **The undisclosed aggregation** survives peer review, because the model
is consistent and only the description is wrong. **The reverse-engineered tolerance** survives
audit, because every individual number is correct; the sequence is what is dishonest.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The bound is valid for the model as written | Every quality claim rests on it | Numerical trouble, an unbounded variable, a tolerance that admits an infeasible point | Check that the bound never exceeds the incumbent and that both are finite |
| The solver's gap uses a known denominator | Cross-run comparison of gaps | Differing conventions between products | Recompute the gap from the raw numbers |
| Termination status is reported honestly | The claim depends on why the run stopped | Status translated into "optimal" by a template | Read the log, not the slide |
| The model that was solved is the model that was validated | Validation transfers to the answer | A last-minute parameter change, a scenario rebuild | Hash the model file; archive it with the parameter tables |
| The instance is within the solver's numerical range | Arithmetic resolves the differences that matter | Coefficient ranges of many orders of magnitude | Report the largest and smallest nonzero coefficients |
| The target gap is tighter than the decision's tolerance | A proven-good answer is usable | Two options closer than the gap | Compare the gap to the difference between the leading options |
| A heuristic's quality measured on a test set carries over | Justifies a heuristic at all | A different instance family or data vintage | Re-measure the gap whenever the instance changes |
| Re-solve time fits the planning cadence | Makes the method operational | Monthly planning on a four-hour solve | Measure time to first feasible and time to gap, not time to close |
| Decomposition effort is affordable | Makes decomposition a real option | No engineer available for weeks | Estimate the build cost before choosing the branch |

Two limits are worth stating plainly.

**The solver can only certify the program it was given.** Every property in this topic — bound,
gap, optimality — is a property of the mathematical object, not of the business. A model with a
missing service constraint solves to a tight gap and certifies a network the business cannot run.

**No measurement in this topic is portable between instances.** A root gap, a time to first
feasible, and a node count are properties of one model on one machine on one day. They are evidence
about that model. The only portable statements are structural: tighter is better than looser,
disaggregated is better than aggregated, and a bound with no answer is worth less than an answer
with a bound.

## 12. Exercises

### Exercise 1 — Read a log (recall, 30 minutes)

**Task.** Given a solver log with a 7,200-second limit, an objective of 12,431,880, a best bound of
11,902,140, and the status "time limit reached", state the relative gap, the target the analyst
would have needed, and the strongest sentence the run supports.
**Deliverable:** one paragraph with the arithmetic shown.
**Acceptance criteria:** the gap is computed as `(12,431,880 - 11,902,140) / 12,431,880 = 4.26%`;
the sentence claims feasibility and a bound and does not use the word optimal.

### Exercise 2 — Gap arithmetic under two conventions (calculation, 25 minutes)

**Task.** For the same run, compute the relative gap with the incumbent as denominator and with the
minimum of `|bound|` and `|incumbent|` as denominator. Then compute it for an objective shifted by
a constant of 5,000,000 and explain the difference.
**Deliverable:** three numbers and a two-sentence explanation.
**Acceptance criteria:** the shift changes the first convention's gap and leaves the underlying
difference unchanged; the explanation names the denominator as the cause.

### Exercise 3 — Formulation audit (application, 60 minutes)

**Task.** Take a network model you have access to. Record its root LP bound, the number of
fractional binaries at the root, the ratio of each M to its derived bound, and the count of
identical candidate sites. Then apply one reformulation and re-measure.
**Deliverable:** a before-and-after table of four metrics.
**Acceptance criteria:** the reformulation is a model change, not a solver setting; the root gap is
reported before and after; the effect on the integer feasible set is stated (none, for all three
reformulations in Phase C).

### Exercise 4 — Choose the method (design, 45 minutes)

**Task.** Three instances: (a) 12 candidate DCs, 60 markets, annual study; (b) 300 candidates,
4,000 markets, nightly re-plan; (c) 40 candidates, 200 markets, interactive workshop. Choose direct
solve, decomposition, or heuristic for each, with the target gap and the reason.
**Deliverable:** a three-row decision table with method, target gap, and justification.
**Acceptance criteria:** the choice cites instance size, re-solve frequency, and decision tolerance;
decomposition is chosen only where the seam is named; the heuristic rows state what is lost.

### Exercise 5 — Decomposition design (design, 75 minutes)

**Task.** For a multicommodity distribution model, sketch a Benders formulation: name the
complicating variables, the subproblem, the cut, and the feasibility cut. State what happens in the
first three iterations on a small instance.
**Deliverable:** a one-page formulation plus a three-iteration trace.
**Acceptance criteria:** the subproblem is a well-defined LP with stated duals; both cut types are
present; the trace shows the master bound rising.

### Exercise 6 — Heuristic honesty (critique, 40 minutes)

**Task.** A colleague reports a local-search design at EUR 18.4 million and calls it the optimised
network. Write the three questions you ask and the sentence you would substitute.
**Deliverable:** three questions and one replacement sentence.
**Acceptance criteria:** one question establishes whether any bound exists; the replacement sentence
states the design cost and explicitly disclaims a bound.

### Exercise 7 — Measurement design (design, 60 minutes)

**Task.** Specify a run that measures time to first feasible and time to 2 per cent gap for instance
sizes of 5, 10, 20, and 40 candidate sites, with three seeds each. State what you plot and what
would falsify the claim that the method scales.
**Deliverable:** a run specification and a plot description.
**Acceptance criteria:** axes, units, and aggregation across seeds are stated; the falsification
condition is specific.

### Exercise 8 — Reporting template (design, 30 minutes)

**Task.** Write the five-field termination block for a report, with a filled example and a rule for
what to do when the status is a time limit.
**Deliverable:** a template with one worked example.
**Acceptance criteria:** all five facts present; the time-limit rule forbids the word optimal;
the template fits in one table row per run.

### Exercise 9 — Change the problem (critique, 45 minutes)

**Task.** A model will not close below a 14 per cent gap. Propose three problem reductions, state
the cost of each, and say which you would take and why.
**Deliverable:** a three-row table plus a recommendation.
**Acceptance criteria:** each reduction is quantified in entities removed; each states what the
answer can no longer see; the recommendation names the decision it serves.

## 13. Self-check answers

**Exercise 1 — reading the log.**

```text
gap = (12,431,880 - 11,902,140) / 12,431,880 = 529,740 / 12,431,880 = 0.04261
```

The relative gap is 4.26 per cent. If the design decision cannot tolerate more than 1 per cent,
the analyst needed a tighter target and did not set one. The strongest supported sentence is: "The
best design found costs EUR 12.43 million per year, every constraint is satisfied, and no design
can cost less than EUR 11.90 million; the run was stopped by its time limit with a 4.3 per cent
gap." The word optimal does not appear, and the two numbers that would justify it are absent.

**Exercise 2 — denominators.**

```text
  incumbent denominator:  529,740 / 12,431,880 = 4.261%
  min(|bound|,|inc|):      529,740 / 11,902,140 = 4.451%
  shift by +5,000,000:     difference unchanged at 529,740
  incumbent denominator:  529,740 / 17,431,880 = 3.039%
  min(|bound|,|inc|):      529,740 / 16,902,140 = 3.134%
```

The two conventions differ by about 0.19 percentage points on the unshifted objective, which is
enough to change a "within 4 per cent" claim into a "within 4.5 per cent" one. The shift lowers both
ratios without changing a single design or a single euro of difference: the relative gap is
scale-dependent and the absolute gap is not. This is why a model with a near-zero objective should
report the absolute gap.

**Exercise 3 — formulation audit, rubric.** A strong submission reports four numbers before and
after: root bound, root gap, fractional binaries at the root, and maximum M ratio. It states
explicitly that the three Phase C reformulations leave the integer feasible set unchanged, so any
change in the answer is a change in what the solver could prove in the budget, not in what the model
permits. A weak submission reports run time only, which confounds the reformulation with machine
load and with the solver's own variability.

**Exercise 4 — method choice.**

| Instance | Method | Target gap | Reason |
|---|---|---:|---|
| (a) 12 DCs, 60 markets, annual | Direct solve | 0.5% | Small enough to close; the decision is capital and the study runs once |
| (b) 300 candidates, 4,000 markets, nightly | Heuristic with a periodic relaxation bound | 3% measured | Exact is out of reach at this cadence; a monthly exact run validates the heuristic's measured gap |
| (c) 40 candidates, 200 markets, workshop | Heuristic, warm start where possible | none claimed | Seconds matter; the planner is screening, not committing capital |

**Exercise 5 — Benders sketch.** Complicating variables: the 24 opening binaries `z`. Subproblem:
the multicommodity flow problem with `z` fixed, a well-behaved LP with a dual in the flow balance
rows. Optimality cut: `theta >= sum over j of pi[j] * (capacity[j] * z[j])` plus a constant term,
where `pi` are the subproblem duals on the capacity rows. Feasibility cut: when the subproblem is
infeasible for a given `z`, add the Farkas-ray inequality that excludes it. Iteration trace: master
solves to a low surrogate cost with too few sites; the subproblem returns an infeasible or expensive
recourse; the cut raises the master's estimate; by iteration three the master's design has moved
toward the sites the flow subproblem actually needs.

**Exercise 6 — heuristic honesty.** Ask: (1) what bound do you have, and from what relaxation;
(2) what is the procedure, and what gap did it achieve on instances with known optima; (3) how many
runs, with what seeds, and what was the spread. Replacement sentence: "A local search over facility
openings produced a design costing EUR 18.4 million per year, which satisfies every constraint; no
bound was computed, so the distance to the best possible design is unknown."

**Exercise 7 — measurement.** Plot time to first feasible and time to a 2 per cent gap, both on a
logarithmic time axis against candidate count on a logarithmic x-axis, with the median of three
seeds and the range shown as a band. State the machine, the solver version, and the model
generator. The claim "the method scales" is falsified if the fitted exponent for time to gap
exceeds about 2, or if the largest instance fails to reach 2 per cent inside the stated budget in
two of three seeds.

**Exercise 8 — template.** Five fields: solver and version; status as printed; objective with units
and period; best bound, same units; and the limit that ended the run, either a wall-clock limit in
seconds or a gap tolerance. Worked example: "Solver X 11.2; status: time limit reached; objective
41.86 EUR m/yr; bound 41.82 EUR m/yr; limit 14,400 s." Rule: when the status is a time limit, the
word optimal is replaced by the gap, and the design is described as the best found.

**Exercise 9 — changing the problem.** Aggregate 180 zones to 40 (-78 per cent of assignment
variables, and the answer can no longer see intra-region cost differences). Restrict 24 candidates
to 12 (-50 per cent of binaries, and the screen decides what is choosable). Reduce 52 weeks to 12
representative weeks (-77 per cent of period links, and seasonality is approximated). For a 14 per
cent gap, take aggregation first: it usually recovers most of the gap, it is disclosed in one
sentence, and it does not require an engineer for a week.

## 14. Interview questions

**1. "What does a MIP solver actually guarantee?"**

Strong answer: a feasible design and a bound, and, at termination, the gap between them; if the gap
is zero to tolerance, the design is optimal to that tolerance. It notes that the guarantee is about
the program, not the business. Weak answer: "it finds the optimal solution", with no mention of the
bound or of termination.

**2. "What is the difference between the incumbent and the bound?"**

Strong answer: the incumbent is achievable and therefore an upper bound for a minimisation; the
bound is provable and therefore a floor; the gap is the distance. It gives a concrete pair of
numbers. Weak answer: treats them as two estimates of the same thing.

**3. "Why do two solvers report different gaps for the same run?"**

Strong answer: the relative gap is a ratio and the denominator is a convention; one divides by the
incumbent, another by the smaller of bound and incumbent, another adds an epsilon. It says to
recompute from the raw numbers. Weak answer: assumes one of them is wrong.

**4. "Your model has been running for three hours with a 9 per cent gap. What do you do?"**

Strong answer: read the root gap first; if it was above 20 per cent, the formulation is the problem
and the fix is tighter M, disaggregation, or symmetry breaking, not more time. It also checks
whether the target gap was ever achievable for this decision. Weak answer: asks for a bigger
machine or a different solver.

**5. "When would you use Lagrangian relaxation?"**

Strong answer: when the coupling constraints are the difficult ones and the remaining problem
separates, typically per facility; it gives a strong bound but no design, so a repair heuristic is
needed. It mentions the multiplier update as the engineering cost. Weak answer: describes it as an
approximation method.

**6. "When would you use Benders rather than branch and bound?"**

Strong answer: when the integer decisions are few and the continuous recourse is a well-behaved LP,
and when the flat model's relaxation is too weak to close; it names the master, the subproblem, and
both cut types. Weak answer: says Benders is "faster for large problems", which is not a criterion.

**7. "Can you report a heuristic answer without a bound?"**

Strong answer: yes, if the decision tolerance is wide, the procedure and its measured gap on
comparable instances are disclosed, and the word optimal is not used. It notes that a bound from a
relaxation can often be obtained cheaply even when no exact solve is possible. Weak answer: says
either "never" or "yes, heuristics are usually fine", both of which skip the conditions.

**8. "How do you decide the gap tolerance?"**

Strong answer: from the decision, not the solver; compare the tolerance to the difference between
the leading options, and set it tighter than that difference or the model cannot rank them. It
notes that the tolerance is fixed before the run. Weak answer: uses the solver default without
knowing what it is.

**9. "What five facts belong in a termination report?"**

Strong answer: solver and version, status, objective, bound, and the limit or tolerance that ended
the run. It notes that a reviewer can reconstruct the claim from those five and from nothing less.
Weak answer: gives the objective and the run time.

**10. "A colleague says the new solver version made the model ten times faster. What do you ask?"**

Strong answer: asks for the root gap and the time to a stated gap on the same model file, because
version gains are real but bounded by the formulation, and because vendor benchmarks are chosen by
the vendor. Weak answer: accepts the claim and schedules the upgrade.

## 15. Key terms

- **MIP / MILP** — a program with integer variables; linear objective and constraints in the MILP case.
- **LP relaxation** — the model with integrality removed; supplies the bound.
- **Bound** — the best cost the relaxation still proves possible; a floor for a minimisation.
- **Incumbent** — the best feasible design found so far; an achievable cost.
- **Gap** — incumbent minus bound, absolute or relative; the only honest quality statement.
- **Root gap** — the gap at the root relaxation; a formulation property, available after one LP.
- **Branch and bound** — tree search over bound restrictions, pruned by bound, infeasibility, or integrality.
- **Node** — one subproblem in the B&B tree.
- **Pruning** — discarding a node that cannot improve on the incumbent.
- **Presolve** — the solver's pre-processing; it can change the model more than the analyst did.
- **Valid inequality / cutting plane** — a row that tightens the relaxation without removing integer points.
- **Symmetry** — a permutation of identical entities that maps the model onto itself.
- **Symmetry breaking** — constraints that select one representative from each symmetric class.
- **Big-M** — the largest flow a linking binary permits; derived, never guessed.
- **Disaggregation** — one row per lane instead of one per site; tighter relaxation, same integer set.
- **Scaling** — the spread of coefficient magnitudes; a numerical property of your data.
- **Termination status** — why the solver stopped; the fact that decides what may be claimed.
- **Time to first feasible** — how long until any design exists.
- **Time to gap** — how long until the design is provably within tolerance.
- **Lagrangian relaxation** — moving coupling constraints into the objective with multipliers to obtain a bound.
- **Benders decomposition** — master over integer decisions, subproblem over continuous flow, cuts between them.
- **Dantzig-Wolfe decomposition** — reformulation over a subproblem's extreme points, solved by column generation.
- **Branch and price** — B&B whose node bounds come from column generation.
- **Heuristic** — a search for a good feasible design with no guarantee and no bound.
- **Matheuristic** — a heuristic that calls a MIP solver on subproblems.
- **Aggregation** — combining entities into fewer modelled entities; a modelling decision, not data cleaning.
- **MIPLIB** — the public benchmark library of real-world MIP instances.

## 16. Sources

**Peer-reviewed and academic (grade A)**

- Geoffrion, A. M., and Graves, G. W. (1974). "Multicommodity Distribution System Design by Benders Decomposition." *Management Science*, 20(5), 822–844. DOI: 10.1287/mnsc.20.5.822.
- Klose, A., and Drexl, A. (2005). "Lower Bounds for the Capacitated Facility Location Problem Based on Column Generation." *Management Science*. DOI: 10.1287/mnsc.1050.0410.
- Görtz, S., and Klose, A. (2011). "A Simple but Usually Fast Branch-and-Bound Algorithm for the Capacitated Facility Location Problem." *INFORMS Journal on Computing*. DOI: 10.1287/ijoc.1110.0468.
- Beasley, J. E. (1993). "Lagrangean Heuristics for Location Problems." *European Journal of Operational Research*, 65, 383–399.
- Costa, A. M. (2005). "A Survey on Benders Decomposition Applied to Fixed-Charge Network Design Problems." *Computers and Operations Research*, 32, 1429–1450. DOI: 10.1016/j.cor.2003.11.012.
- Gleixner, A., et al. (2021). "MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library." *Mathematical Programming Computation*. DOI: 10.1007/s12532-020-00194-3.
- Ostrowski, J., Linderoth, J., Rossi, F., and Smriglio, S. (2011). "Orbital Branching." *Mathematical Programming*. Mechanism of symmetry in B&B.
- Fisher, M. L. (1985). "An Applications Oriented Guide to Lagrangian Relaxation." *Interfaces*, 15(2), 10–21.
- Camm, J. D., et al. (1997). "Blending OR/MS, Judgment, and GIS: Restructuring P&G's Supply Chain." *Interfaces*, 27(1).
- Arntzen, B. C., Brown, G. G., Harrison, T. P., and Trafton, L. L. (1995). "Global Supply Chain Management at Digital Equipment Corporation." *Interfaces*, 25(1).
- Parra Galvez, C., and Borenstein, D. (2023). "Large Scale Supply Chain Network Design: An Effective Heuristic Approach." *Pesquisa Operacional*. DOI: 10.1590/0101-7438.2023.043.00269080.

**Official and standards (grade B)**

- MIPLIB, "History of MIPLIB," Zuse Institute Berlin, miplib.zib.de/history.html. Origin in 1992; edition history.
- ISO/IEC/IEEE 60559, floating-point arithmetic. Governs the tolerances behind every numerical claim.
- Federal Reserve Board and OCC, "Supervisory Guidance on Model Risk Management," SR Letter 11-7, April 2011. Documentation and validation expectations.

**Teaching material and implementation documentation (grade C)**

- Chinneck, J. W. "Integer/Discrete Programming via Branch and Bound," *Practical Optimization: a Gentle Introduction*, Carleton University. Incumbent, bound, pruning.
- Carnegie Mellon "discopt" MILP tutorial. LP bound as lower bound, gap definition, B&B tree.
- OR-Tools, `MPSolverCommonParameters` documentation. Notes that relative-gap denominators differ between implementations.

**Vendor material (grade D, treated with suspicion)**

- Gurobi Help Center, "What is the MIPGap?" Definition of the relative gap and its denominator.
- Gurobi, "Provable Optimality: What a Solver Guarantees, and What It Doesn't," blog. Status semantics.
- IBM, CPLEX documentation, "relative MIP gap tolerance." Default 1e-4 and denominator form.
- AIMMS, "MIP Relative Optimality Tolerance," user guide. Definition with an epsilon in the denominator.
- Vendor benchmark publications generally. Configurations chosen by the vendor; no performance claim in this topic rests on them.

**Practitioner commentary (grade E)**

- Kunigami, C. (2012). "Lagrangean Relaxation — Practice." Implementation walkthrough of a subgradient scheme applied to facility location.

> **Note:** the graded D rows that give gap definitions are used only to establish that the
> definitions differ, which the grade C implementation documentation corroborates independently. No
> number in this topic is taken from a vendor benchmark.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Previous topic, carbon in the objective: [`09_carbon-objective.md`](09_carbon-objective.md)
- Model construction, where solve difficulty is created: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- The baseline facility-location model: [`04_facility-location.md`](04_facility-location.md)
- Multi-commodity and multi-period models: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- Scenario and sensitivity analysis: [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md)
- Robust and stochastic optimisation: [`08_robust-stochastic.md`](08_robust-stochastic.md)
- This part's index: [`../_index.md`](../_index.md)
- Simulation, metaheuristics, and uncertainty in a broader setting: [`../20_digital_twins_simulation_optimization_uncertainty/_index.md`](../20_digital_twins_simulation_optimization_uncertainty/_index.md)
- Part I of the program: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- This module's survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

You now hold the complete chain: a decision ladder, a graph and its cost decomposition, a model and
its anatomy, four model families, the greenfield and brownfield variants, the scenario and robust
extensions, carbon in the objective, and the behaviour of the whole apparatus once it meets a
solver. What remains is the professional habit this closing topic has tried to install — asking, of
every answer, what was proved, over what model, within what gap, and in what time.
