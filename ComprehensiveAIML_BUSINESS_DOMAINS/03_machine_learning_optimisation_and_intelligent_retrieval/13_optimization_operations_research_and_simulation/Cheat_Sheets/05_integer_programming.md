---
title: "Integer programming"
topic: "Integer programming, branch-and-bound, cutting planes and formulation strength"
collection: "Module 13 — Optimisation, Operations Research and Simulation / Cheat Sheets"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Integer programming

> **Evidence:** every number below is derived in this file from stated inputs. Item weights,
> demands, costs, and timings are constructed teaching values and are labelled ILLUSTRATIVE.
> No benchmark, licence price, deployment, or vendor performance claim is made anywhere.

## 1. File Header

**Purpose.** Give a practitioner a decision-grade reference for mixed-integer linear programming
(MILP): how to write a formulation whose LP relaxation is worth solving, how to read the bound the
solver returns, how to tell a weak model from a hard instance, and how to prove or refuse the claim
"optimal".

**Position in the unit.** This sheet is the integer-programming card of the operations-research
ladder. `01_optimisation_formulation.md` supplies continuous modelling and duality;
`02_solver_selection.md` supplies the solver-choice heuristic; `04_network_and_scheduling_models.md`
supplies the network structures that are often totally unimodular; `06_metaheuristics.md` covers the
case where no proof is available; `07_interval_greedy_and_weighted_scheduling.md` covers interval
structures with a combinatorial exact algorithm. This card owns the boundary between a linear
relaxation and an integral decision.

**In scope.**

- Binary and general-integer variables inside otherwise linear models.
- LP relaxation bounds, integrality gaps, and their use as evidence.
- Branch-and-bound mechanics: incumbent, dual bound, pruning, node accounting.
- Branch-and-cut: Gomory and Chvátal–Gomory cuts, cut validity, and closure rank.
- Formulation strength: big-M sizing, fixed-charge linearisation, convex-hull projections.
- Integrality that is free: total unimodularity, assignment and network structures.
- Tolerances, scaling, symmetry, and the failure modes that follow from each.

**Out of scope.**

- Nonlinear integer models, quadratic assignment, and MINLP outer approximation.
- Constraint-programming propagation internals beyond the modelling interface.
- Decomposition algorithms (Benders, column generation, Lagrangian) except as handoffs.
- Any statement about a specific commercial solver's default parameter values; those change
  between releases and are treated as `OPEN` until read from the installed version.

**Prerequisites.** You must be able to (a) write a linear objective with declared units, (b) state a
feasible region as equalities and inequalities with the direction of each inequality explicit,
(c) compute a ratio and a percentage gap without a spreadsheet, and (d) separate an event timestamp
from the time at which the decision had to be made. Without (d) a "backtest" of an optimisation model
silently uses post-decision information.

**Outcomes.** After working this sheet a reader can: size a big-M from the model's own activity
bounds instead of a round number; compute the relative gap from an incumbent and a dual bound and
say whether "optimal" is proven; reproduce a fifteen-node branch-and-bound trace by hand; derive one
Chvátal–Gomory cut and verify it against the integer feasible set; and reject a formulation whose
relaxation is far from its convex hull.

**Notation discipline.** All display mathematics appears in fenced `text` blocks, one convention per
file. Objective sense is stated in words next to every model. Units are attached at first use:
capacity in kilograms, cost in the declared currency unit, time in seconds or milliseconds with the
unit spelled out. Symbols are defined at first use and collected in §3's table.

**Evidence grading used in this file.**

| Grade | Meaning here | Example in this sheet |
|---|---|---|
| DERIVED | Recomputable from the stated formula and inputs | the 6.944% integrality gap in §5 |
| ILLUSTRATIVE | Constructed inputs, arithmetic verified | item weights, demand 650, timings |
| HEURISTIC | Rule of thumb, no proof | big-M should equal max activity, not 10⁶ |
| OPEN | Not established here | default solver tolerances of a named release |

**How to read it.** §2 gives the mental model, §4 the derivations, §5 a fully traced instance, §8
the diagnostics, §11 the interview drill. A reader under time pressure should take §5 and §12 first
and return to §4 when a number needs defending.

## 2. Concept Map / Mental Model

### 2.1 The pipeline

```text
business decision
      |
      v
[1] variables + units + integrality  --- wrong here -> model answers a different question
      |
      v
[2] constraints as inequalities      --- wrong direction -> infeasible or unbounded relaxation
      |
      v
[3] LP relaxation  z_LP              --- bound; also the first diagnostic
      |
      +--> relaxation integral?  --yes--> done, certificate = LP optimal basis
      |
      no
      v
[4] branch on a fractional variable  -> child bounds z >= z_LP (max sense: z <= z_LP)
      |
[5] add valid inequalities (cuts)    -> raise lower bound without removing integer points
      |
[6] prune when bound <= incumbent    -> incumbent is a feasible integer solution
      |
      v
[7] stop on gap tolerance or budget  -> report incumbent, bound, gap, and node count
```

### 2.2 The one invariant that governs everything

For a maximisation model, at every moment the solver holds two numbers: the best integer solution
found (the incumbent, a lower bound `z_inc` on the optimum) and the best bound over all unexplored
nodes (an upper bound `z_ub`). The optimum `z*` lies in the closed interval `[z_inc, z_ub]`. Every
action the solver takes either raises `z_inc`, lowers `z_ub`, or removes a node from the frontier.
Nothing else changes the certificate. A run that stops with `z_ub - z_inc > 0` has proved a bracket,
not an optimum, regardless of the word printed in the log.

### 2.3 Two different questions that get confused

**Question A — is the model right?** A model is right when every integer-feasible point of the model
is operationally executable and every operationally executable plan has a model point no worse than
it. This is verified by reading constraints against the business rule and by testing boundary plans;
no solver helps.

**Question B — does the algorithm finish?** A model can be right and still unsolvable inside the
decision window. This is verified by the bound trajectory: how fast `z_ub` falls relative to node
count.

Conflating A and B produces the two most common project failures: a fast solve of a wrong model, and
a refusal to accept a bounded, honest 0.4% gap on a right one.

### 2.4 Taxonomy of integer structures

| Structure | Integrality source | Typical relaxation behaviour |
|---|---|---|
| Assignment (n×n, one per row/column) | bipartite incidence matrix is totally unimodular | LP optimum already integral |
| Transportation / min-cost flow | node-arc incidence matrix is totally unimodular | LP optimum already integral |
| 0/1 knapsack | single capacity row, ratios decide | at most one fractional item |
| Set covering | dense 0/1 columns | large gap, needs cuts |
| Fixed-charge facility location | big-M linking rows | gap controlled entirely by M |
| Scheduling with disjunctions | pairwise ordering binaries | weak unless tightened by time windows |
| Routing (VRP) | exponential family of subtour constraints | needs separation, not a compact model |

### 2.5 What "hard" means concretely

Hardness is not a property of the topic; it is a property of the pair (formulation, bound quality).
Two models of the same feasible integer set can differ by orders of magnitude in nodes explored while
sharing an identical optimal value. The reason is geometric: the LP relaxation optimises over a
polyhedron, and the integer points are a lattice inside it. A formulation whose relaxation polyhedron
hugs the convex hull of the lattice gives bounds that prune almost immediately; a loose formulation
leaves a wide slab in which fractional vertices attract the simplex.

```text
tight formulation                      loose formulation
  hull = relaxation                      hull inside relaxation
  bound at root often equals optimum     bound at root far from optimum
  branching is a formality               branching is the work
```

### 2.6 Reading order of evidence

When a solve finishes, read in this order: (1) did it report a primal solution at all, (2) what gap
was proven, (3) what node and LP count, (4) which presolve reductions fired, (5) what the root bound
was before any cut. Item (5) is the fastest formulation-quality signal available; if the root bound
is within 1% of the incumbent on a well-posed model, the formulation is probably fine and the
instance is easy. If the root bound is 50% away, no amount of extra compute fixes the model.

### 2.7 The mental model in one paragraph

Integer programming is linear programming plus a lattice condition. The linear program is easy and
supplies a bound; the lattice condition makes the problem NP-hard and forces search. Every technique
in this sheet is one of three things: a way to tighten the bound (better formulation, cuts), a way to
search fewer nodes (branching rules, symmetry breaking, presolve), or a way to certify when to stop
(gap arithmetic). If a proposed technique does none of those three, it is decoration.

## 3. Foundational Definitions

Each definition below is written so that it can be checked against a model file. Definitions are
ordered from the object being modelled toward the algorithm that solves it.

**Decision variable.** A symbol whose value the model chooses. Declare its unit (units shipped,
kilograms, currency), its type (continuous, integer, binary), and its valid range. A binary variable
is an integer variable with range `[0,1]`; using a binary where an integer count is meant forces the
modeller to invent one binary per unit and inflates the model by the count.

**Integrality constraint.** The requirement `x_j ∈ Z` for one or more indices. Only the variables
declared integer carry it; everything else is continuous even inside a MILP. A model can be 99%
continuous and still be a MILP, and it will be as hard as its integer part.

**Feasible region.** The set of points satisfying all constraints, equalities, bounds, and
integrality. Written `F` for the integer-feasible set and `P` for the LP relaxation's polyhedron, we
always have `F ⊆ P`, and `F` is generally not convex.

**LP relaxation.** The problem obtained by dropping integrality. It is a linear program, solvable in
polynomial time in theory and quickly in practice, and its optimal value is a bound: for a
maximisation, `z_LP ≥ z_IP`. The relaxation is the single most informative object in the whole
workflow because it bounds the best possible outcome of any amount of search.

**Convex hull.** The smallest convex set containing `F`, written `conv(F)`. Optimising a linear
objective over `conv(F)` yields exactly `z_IP`. A formulation is called ideal or tight when its LP
relaxation polyhedron equals `conv(F)`. No compact ideal formulation is known for most hard
combinatorial families, which is why search exists.

**Integrality gap.** Two distinct quantities share this name and must not be mixed. The *instance*
gap is `z_LP - z_IP` (absolute) or `(z_LP - z_IP)/|z_IP|` (relative) for a specific data set. The
*formulation* gap is the supremum of that ratio over all data sets of a family. Only the first is
computable from a solve; the second is a theorem. Reporting the instance gap while calling it the
formulation gap is a common and material error.

**Incumbent.** The best integer-feasible solution found so far, with its objective value `z_inc`. An
incumbent is a witness: it proves `z* ≥ z_inc` in a maximisation. Without an incumbent a solver can
still prove infeasibility or unboundedness, but it cannot terminate by gap.

**Dual bound.** The best bound over the unexplored frontier, `z_ub`. It is the maximum over open
nodes of their LP bounds, or the node's inherited bound before its LP is solved. It is monotone
non-increasing during a maximisation run.

**Relative gap.** `g = (z_ub - z_inc)/|z_inc|` with the convention `|z_inc| ≥ 1` guard to avoid
division blow-up near zero. A run with `g = 0.005` has proved the incumbent is within 0.5% of optimal
in the multiplicative sense; it has not proved optimality.

**Branching.** Splitting a node's feasible region into disjoint parts, canonically `x_j ≤ k` and
`x_j ≥ k+1` for an integer variable taking fractional value `k+f`. Branching is exhaustive because
the two parts cover the integer points of the parent and are disjoint; it is not a heuristic.

**Node.** One subproblem in the search tree: a copy of the model with inherited bound changes. The
root is node 0. Depth is bounded by the number of integer variables in the worst case.

**Pruning.** Discarding a node without exploring children, justified by exactly one of three reasons:
its bound is no better than the incumbent, it is infeasible, or it yielded an integral solution that
has been recorded. Pruning is what makes search terminate; "the solver gave up" is not pruning.

**Valid inequality (cut).** An inequality satisfied by every point of `F` but not necessarily by every
point of `P`. Adding it shrinks `P` without removing any integer point, so it cannot change `z_IP`
but can only raise `z_LP` (maximisation: lower the upper bound).

**Chvátal–Gomory cut.** A cut derived mechanically: for `u ≥ 0`, from `uᵀA x ≤ uᵀb` with `x ≥ 0`
integer, the inequality `floor(uᵀA) x ≤ floor(uᵀb)` is valid. The derivation is a one-line rounding
argument and is verified numerically in §4.3.

**big-M.** A constant used to switch a constraint on and off with a binary: `aᵀx ≤ b + M(1-y)`.
Its correct value is the maximum possible value of `aᵀx - b` over the model's own bounds, not a large
round number. Sizing this constant is the highest-leverage single edit in most MILP models.

**Indicator constraint.** A native construct `y = 1 ⇒ aᵀx ≤ b` that avoids an explicit M by letting
the solver linearise internally with its own activity bounds. Functionally equivalent to a
well-sized big-M, safer by construction, but less portable across solvers and file formats.

**SOS1 / SOS2.** Special ordered sets: at most one nonzero in a set (SOS1), or at most two adjacent
nonzeros in an ordered list (SOS2). They encode disjunctions and piecewise-linear functions with
branching decisions placed on the ordering rather than on individual binaries, which can cut tree
size substantially on some families and not at all on others.

**Total unimodularity (TU).** A matrix `A` is TU when every square submatrix has determinant in
`{0, +1, -1}`. If `A` is TU and `b` is integral, every basic feasible solution of `Ax ≤ b, x ≥ 0` is
integral, so the LP relaxation solves the integer problem. Node-arc incidence matrices of directed
graphs and bipartite incidence matrices are TU.

**Presolve.** A set of model-to-model reductions applied before search: bound tightening, singleton
row removal, coefficient strengthening, duplicate row and column detection, and integrality
propagation. Presolve changes the instance, so the reported node count refers to the reduced model
and is not comparable across solver settings.

**Symmetry.** A permutation of variables that maps every feasible point to a feasible point with the
same objective. Symmetric models have exponentially many equivalent optima, and a naive search
explores many of them; symmetry-breaking constraints or orbital branching remove the redundancy.

**Fixed-charge cost.** A cost `f` paid once if an activity is used at all, plus a per-unit cost `c`.
Modelled with `z = f y + c x` and `x ≤ U y`. The linking row is where the formulation's strength is
won or lost.

**Knapsack problem.** Maximise total value subject to one capacity row with binary items. Its LP
relaxation is solved by ratio ordering and has at most one fractional variable, which makes it the
cleanest vehicle for teaching bounds and gaps.

**Set covering.** Choose the cheapest collection of sets whose union covers a universe. Covering rows
are `≥ 1` with 0/1 columns. Gaps are typically large and cut generation matters.

**Facility location.** Choose which facilities to open (fixed charge) and which demand to serve from
each (transport cost). The canonical example of a model whose LP bound is controlled by big-M sizing.

**Assignment problem.** Match `n` agents to `n` jobs at minimum cost. The constraint matrix is
bipartite incidence, hence TU, hence the LP relaxation is integral; a `n=3` instance is solved by
enumerating `3! = 6` permutations, and any LP solver returns the same value.

**MILP / MIP.** A linear program with some integer variables. "MIP" and "MILP" are used
interchangeably in practice; "MIP" is also used for the class including mixed-binary models.

**Pseudo-polynomial algorithm.** An algorithm polynomial in the numeric magnitude of the input rather
than its encoding length, such as knapsack dynamic programming in `O(nW)`. It is exact but its cost
is driven by a capacity value, which is why `W = 10⁹` is out of reach even though the item count is
small.

**Reduced cost.** For a variable at a bound, the rate at which the objective changes if the bound is
relaxed by one unit. In a relaxation it prices the variable out; after branching it explains why a
child's bound moved in a specific direction.

**Feasibility and integrality tolerance.** Numbers `ε_f` (constraint violation accepted) and `ε_i`
(distance from an integer accepted) chosen by the solver, typically around `10⁻⁶`. A "binary" value
of `0.9999994` is accepted as 1. The arithmetic consequence is worked in §8.

**Symbol table.** Symbols used in the formulas of this sheet.

| Symbol | Meaning | Units | Type / range | First appears |
|---|---|---|---|---|
| `x_j` | activity of decision `j` | item-specific | integer or binary, `≥ 0` | §4.1 |
| `y_j` | binary switch for fixed charge | dimensionless | `{0,1}` | §4.3 |
| `w_j` | resource consumption per unit | kilograms | positive real | §4.1 |
| `v_j` | value per unit | currency | positive real | §4.1 |
| `W` | capacity | kilograms | positive real | §4.1 |
| `z_LP` | LP relaxation optimum | currency | real | §4.1 |
| `z_IP` | integer optimum | currency | real | §4.1 |
| `z_inc` | incumbent value | currency | real | §4.2 |
| `z_ub` | dual bound | currency | real | §4.2 |
| `g` | relative gap | dimensionless | `≥ 0` | §4.2 |
| `M` | big-M constant | same as row activity | real | §4.3 |
| `u` | CG multiplier vector | dimensionless | `≥ 0` | §4.3 |
| `D` | total demand | kilograms or units | positive real | §4.3 |
| `U` | capacity of one facility | same as `D` | positive real | §4.3 |
| `f` | fixed charge | currency | positive real | §4.3 |
| `c` | variable cost per unit | currency per unit | positive real | §4.3 |
| `N` | number of nodes explored | count | integer `≥ 1` | §9 |

## 4. Core Formulas with Full Derivation

### 4.1 Base case — the LP relaxation bound of a 0/1 knapsack

```text
FORMULA
  sort items by density d_j = v_j / w_j descending, j = 1..n
  let k = first index with w_1 + ... + w_k > W
  z_LP = sum_{j<k} v_j + ( W - sum_{j<k} w_j ) * ( v_k / w_k )

WHERE
  w_j  kilograms consumed by item j          (positive real)
  v_j  currency value of item j              (positive real)
  W    kilograms of capacity                 (positive real)
  k    the critical (split) item index, 1 <= k <= n

CONSTRAINT
  all items binary, sum_j w_j x_j <= W, x_j in {0,1}
  the formula is exact only for the LP relaxation, not for the integer problem

NUMERICAL EXAMPLE  (ILLUSTRATIVE data, DERIVED arithmetic)
  items  B(8 kg, 7), A(12 kg, 10), C(6 kg, 5), D(5 kg, 4),  W = 15 kg
  densities  B 7/8 = 0.8750, A 10/12 = 0.8333, C 5/6 = 0.8333, D 4/5 = 0.8000
  take B whole: load 8 kg, value 7, capacity left 15 - 8 = 7 kg
  next by density is A (0.8333, tied with C; either order gives the same total)
  A needs 12 kg for its full value, so only the fraction 7/12 of A fits
  value of that fraction = 7 kg * (10 currency / 12 kg) = 5.8333 currency
  z_LP = 7 + 5.8333 = 12.8333
INTERPRETATION
  the relaxation fills capacity exactly (8 + 7 = 15 kg) and pays a proportional value for the
  split item; the resulting 12.8333 exceeds the best integer plan of 12 and is therefore a bound
  but never a plan. Any reported plan must come from an integer-feasible point.
```

The bound formula is exact because the LP has one nontrivial row: an optimal basic solution uses
`n` basic variables, `n-1` of them at a bound (0 or 1) and at most one in between. Ratio ordering
is the greedy exchange argument — if two items `i, j` with `d_i > d_j` are present with `x_i < 1`
and `x_j > 0`, moving mass from `j` to `i` cannot decrease value and cannot violate capacity, so
some optimum has the density order respected.

**Why "at most one fractional" matters operationally.** If the relaxation has at most one fractional
variable, one branch on that variable splits the problem into two subproblems each of which may
already be integral. The depth of the tree is then governed by how many times the split item changes,
not by the item count. This is why knapsack-shaped models often solve in a handful of nodes while
covering models of similar size do not.

**Validity condition for this subsection.** The formula requires positive weights, a single capacity
row, and binary items. It does not extend to multi-dimensional knapsack (several capacity rows), where
the LP optimum can have many fractional variables and no closed-form greedy exists.

### 4.2 General case — gap, bound monotonicity, and the pruning test

```text
FORMULA
  relative gap      g = ( z_ub - z_inc ) / max( |z_inc|, 1 )
  absolute gap      a = z_ub - z_inc
  pruning test      node n is pruned when  z_LP(n) <= z_inc   (maximisation)
  frontier bound    z_ub = max over open nodes n of z_LP(n)

WHERE
  z_inc  best integer objective found              currency
  z_ub   best bound over the frontier              currency
  z_LP(n) the LP optimum at node n                 currency
  g      dimensionless fraction, 0 <= g <= 1 in a sane run

CONSTRAINT
  maximisation sense throughout; for minimisation every inequality above flips
  the guard max(|z_inc|,1) prevents a divide-by-near-zero when the objective passes through 0

NUMERICAL EXAMPLE  (data from 4.1, DERIVED)
  z_LP(root) = 12.8333, best integer plan found = 12
  a = 12.8333 - 12 = 0.8333
  g = 0.8333 / 12 = 0.069444  -> 6.9444%
  so even a 10% tolerance would permit stopping, but a 1% tolerance would not
  after the fifteenth node of the trace in 5.3 the frontier is empty and g is reported as 0

INTERPRETATION
  the gap is a statement about the bracket, not about solution quality. A plan can be 6.9% from
  the bound and still be the true optimum; conversely, an incumbent 0.2% from the bound is not
  proven optimal until the frontier is empty or the tolerance is met.
```

**Rounding a bound into a proof.** If every objective coefficient is integral, `z_IP` is an integer,
so `floor(z_LP)` (maximisation) is also a valid bound. In the worked instance `floor(12.8333) = 12`,
which equals the best integer value `12`, so the root relaxation alone closes the bracket once any
12-valued plan is in hand — no branching is needed. Two conditions must hold: the objective data must
be integral (otherwise `floor(z_LP)` can fall below the true optimum and prune the optimal plan — an
invalid bound, not merely a loose one), and the incumbent must be verified against the original rows.

**Monotonicity.** `z_ub` never rises during a maximisation run because the frontier only loses nodes
and each child's LP bound is at most its parent's (the child's feasible region is a subset). Any log
line showing `z_ub` increasing means either a different sense was selected, a cut was removed, or
the reported column is the incumbent rather than the bound.

**Validity condition for this subsection.** The pruning test assumes the child LP bound is a valid
bound for the child, which requires that no constraint was dropped between parent and child. If a
warm start or a heuristic solution is used as an incumbent, it must be verified integer-feasible
against the original rows, not against the presolved model.

### 4.3 Variants — big-M sizing, fixed charge, CG cuts, and TU integrality

**Variant A: sizing big-M from model activity.**

```text
FORMULA
  intended set:  x is allowed only when the switch is on
                 x <= U y,  y in {0,1},  x >= 0
  generic switch form with an offset row a^T z <= b:
                 a^T z <= b + M (1 - y)
      y = 1  ->  a^T z <= b          the row is enforced
      y = 0  ->  a^T z <= b + M      the row is disabled
  smallest valid M = max over the model's own bounds of ( a^T z - b )
                   = U - b   when the activity is capped at U
                   = min(U_capacity, D_demand)  for the plain row x <= U y  (b = 0)

WHERE
  y  binary switch, 1 = allowed                  dimensionless
  z  the vector of variables in the switched row  model units
  a, b  row coefficients and right-hand side      mixed
  M  the disabling slack                          same unit as a^T z
  U  tightest upper bound on the row activity     same unit as a^T z
  D  total demand that the row must be able to carry   same unit as a^T z

CONSTRAINT
  M must be at least the largest activity the row can take when the switch is on;
  an M below that value cuts off feasible solutions, an M above it only weakens the relaxation
  M is meaningless unless the model has finite bounds on z; without bounds no finite M is valid

NUMERICAL EXAMPLE  (ILLUSTRATIVE demand and cost, DERIVED arithmetic)
  f = 12,000 currency, c = 9 currency/unit, U = 800 units, D = 650 units, b = 0
  integer plan: y = 1, cost = 12,000 + 9*650 = 12,000 + 5,850 = 17,850
  M = 650 : relaxation y = 650/650 = 1.00 -> 12,000 + 5,850 = 17,850   gap 0.00%
  M = 800 : relaxation y = 650/800 = 0.8125 -> 9,750 + 5,850 = 15,600  gap 12.61%
  M = 5000: relaxation y = 650/5000 = 0.13  -> 1,560 + 5,850 = 7,410   gap 58.49%

INTERPRETATION
  M = min(U, D) makes the relaxation's linking row coincide with the convex hull of the
  single-facility set, so the bound is exact. Choosing M as a round large number costs the
  model 58% of its bound at the root and typically multiplies node count by a large factor.
```

**Variant B: fixed-charge linearisation and why the hull is reached.** The set
`{(x,y) : 0 ≤ x ≤ U y, y ∈ {0,1}}` has convex hull `{(x,y) : 0 ≤ x ≤ U, x ≤ U y, 0 ≤ y ≤ 1}` — the
triangle with vertices `(0,0)`, `(U,0)`, `(U,1)`. Writing `x ≤ U y` already describes that triangle
when `x ≥ 0` and `y ≤ 1` hold, so no extra row is needed; the strength comes from using `U = min(capacity,
demand)` rather than a slack constant. The cost formula `z = f y + c x` is linear in `(x,y)` and
therefore keeps the model an LP; the only nonlinearity, the on/off nature of `f`, is carried by `y`.

**Variant C: one Chvátal–Gomory cut, derived and verified.**

```text
FORMULA
  valid CG cut:  for any u >= 0,   floor(u^T A) x <= floor(u^T b)   is valid for x >= 0 integer

WHERE
  A, b  the constraint matrix and right-hand side of the current relaxation
  u     nonnegative multiplier, one scalar per row
  floor applied componentwise to the row coefficients and to the right-hand side

NUMERICAL EXAMPLE  (ILLUSTRATIVE data, DERIVED)
  model: max 8 x1 + 5 x2   s.t.  5 x1 + 4 x2 <= 9,  x >= 0 integer
  LP optimum x1 = 9/5 = 1.8, x2 = 0, z_LP = 14.4
  take u = 1/5 (one fifth of the single row):
       x1 + 0.8 x2 <= 1.8
       floor(1) x1 + floor(0.8) x2 <= floor(1.8)
       x1 + 0 x2 <= 1
  the integer optimum of the two-variable problem is (1,1) with value 8 + 5 = 13
  validity check by exhaustion: x1 >= 2 forces 10 + 4 x2 <= 9, impossible, so x1 <= 1 holds
  adding x1 <= 1 gives LP optimum x1 = 1, x2 = (9-5)/4 = 1, z_LP = 13 = z_IP
  a second cut from u = 1/4 gives x1 + x2 <= 2, valid (max sum on the integer set is 2) but it
  does not separate (1.8, 0) because 1.8 <= 2, so it changes nothing

INTERPRETATION
  one rounding cut closed a 14.4 to 13 gap in a single round; the second, valid cut was useless
  because it did not cut the current fractional vertex. Valid is necessary, separating is what
  helps. Cut strength is measured by how much of the gap it removes at the current point.
```

**Variant D: integrality for free via total unimodularity.** For the assignment problem with cost
matrix (ILLUSTRATIVE)

```text
      job1  job2  job3
agent1   9     2     7
agent2   6     4     3
agent3   5     8     1

permutation values: 1-2-3 = 14, 1-3-2 = 20, 2-1-3 = 9, 2-3-1 = 10, 3-1-2 = 21, 3-2-1 = 16
minimum = 9 at (agent1->job2, agent2->job1, agent3->job3)
LP relaxation value = 9 as well, because the doubly stochastic polytope has permutation
matrices as its vertices (Birkhoff-von Neumann) and a linear objective is optimised at a vertex
```

The incidence matrix of a bipartite graph is TU, so integrality of `b` gives integral basic
solutions. The practical consequence: use the assignment/transportation formulation instead of a
generic MILP whenever the structure fits, because the LP will already return an integer plan and the
solver will terminate at the root. Adding one side constraint that is not a network row generally
destroys TU, and fractional vertices reappear; that is the moment a modeller has silently converted
an easy problem into a hard one.

**Validity condition for §4.3.** Variant A requires a single linking row per switch and known bounds
on `x`. Variant C requires `x ≥ 0` and integer, and integer-valued `A` after multiplication by `u`
is not needed but the flooring must be applied to the actual numeric coefficients, not to rounded
ones. Variant D requires the constraint matrix to be exactly a network or bipartite incidence matrix;
a single extra row or a coefficient of 2 breaks the guarantee.

### 4.4 Limits and counterexamples

**Limit 1: rounding the LP solution is not a solution method.** In the two-variable example of §4.3
the LP point is `x1 = 1.8`; rounding down gives `x1 = 1, x2 = 0` with value 8, while the true optimum
is 13 at `(1,1)`. Rounding down lost `1 - 8/13 = 38.46%` of the optimum. In the knapsack of §5 the
root relaxation is `B = 1` with `A = 7/12`; rounding A down gives value 7, which is `1 - 7/12 = 41.67%`
below the optimum, and rounding A up gives 20 kg, which is infeasible. Both directions fail on one of
the two instances, so neither is a method — the repair step has to re-optimise, not re-round.

**Limit 2: M below max activity is a silent feasibility cut.** If the same facility model used
`M = 400` with demand 650, the integer plan `y = 1, x = 650` would be infeasible, and the solver's
"optimal" answer would be a plan serving at most 400 units — a model that under-serves and reports
success. The discriminating check is to compute `min(U, D)` and compare it with the M in the file.

**Limit 3: the LP bound can be attained or missed by a wide margin on adjacent data.** For the same
four items, capacity 15 gives a 6.944% gap while capacity 20 gives none, because at 20 kg the greedy
order fills capacity exactly with B and A. Gap is a property of the data, so quoting a gap without
the data is meaningless.

**Limit 4: integrality tolerance leaks a fractional plan.** With `ε_i = 10⁻⁶` and a variable that
represents 1,200,000 units, an accepted deviation of one part per million is 1.2 units. The solver
reports "integer feasible" while the plan is operationally short by 1.2 units; if a unit is a 500 kg
pallet, that is 600 kg of unshipped goods.

**Limit 5: unbounded relaxation does not imply an unbounded model, and vice versa.** Dropping one
capacity row makes the relaxation unbounded; the integer model may still be bounded by a row that
became redundant only in combination with the dropped one. The diagnosis is to solve with all rows,
then remove rows one at a time and observe which removal makes the bound unbounded.

**Validity condition for the whole of §4.** Every formula above assumes (i) linear objective and
constraints, (ii) known finite bounds on the activity of every switched row, (iii) exact arithmetic
for the validity arguments, and (iv) the stated objective sense. The formulas give bounds and
certificates; they never by themselves establish that the model matches the business rule.

## 5. Worked Numerical Example

**Instance (ILLUSTRATIVE).** A packing decision with four item types and 15 kg of capacity.

| Item | Weight kg | Value currency | Density value/kg | Order |
|---|---|---|---|---|
| B | 8 | 7 | 0.8750 | 1 |
| A | 12 | 10 | 0.8333 | 2 |
| C | 6 | 5 | 0.8333 | 3 |
| D | 5 | 4 | 0.8000 | 4 |

### 5.1 Integer enumeration — the ground truth

Sixteen subsets; only those at or under 15 kg count.

| Subset | Weight kg | Value | Feasible at W = 15 |
|---|---|---|---|
| none | 0 | 0 | yes |
| A | 12 | 10 | yes |
| B | 8 | 7 | yes |
| C | 6 | 5 | yes |
| D | 5 | 4 | yes |
| AB | 20 | 17 | no |
| AC | 18 | 15 | no |
| AD | 17 | 14 | no |
| BC | 14 | 12 | yes |
| BD | 13 | 11 | yes |
| CD | 11 | 9 | yes |
| ABC | 26 | 22 | no |
| ABD | 25 | 21 | no |
| ACD | 23 | 19 | no |
| BCD | 19 | 16 | no |
| ABCD | 31 | 26 | no |

Best feasible: `B + C` at 14 kg and value 12, so `z_IP = 12`, and that value is attained by exactly one
subset. Second best is 11 at `B + D` (13 kg), then 10 at `A` alone (12 kg). The 12-versus-11 margin is
one currency unit on 12, i.e. 8.33%, which is the practical reason this instance is a fair test: a
model that loses one item type answers 11 and looks almost right.

### 5.2 The relaxation

```text
capacity 15
B: 8 kg -> load 8, value 7, remaining 7
A: 12 kg does not fit; take fraction 7/12 of A
   contribution = (10/12) * 7 = 5.8333
z_LP = 7 + 5.8333 = 12.8333
check of capacity: 8 + (7/12)*12 = 8 + 7 = 15 kg exactly
check of value:    7 + (10/12)*7 = 7 + 5.8333 = 12.8333
absolute gap: 12.8333 - 12 = 0.8333
relative gap: 0.8333 / 12 = 0.069444 -> 6.9444%
```

Independent check by the dual: the marginal value of capacity at this LP optimum is the density of the
split item, `10/12 = 0.8333` currency per kg. Raising capacity from 15 to 16 kg must raise `z_LP` by
`0.8333` to `13.6667`. Solving the relaxed knapsack at 16 kg: B loads 8, remaining 8, A fraction
`8/12` gives `6.6667`, total `7 + 6.6667 = 13.6667`. Difference is exactly `0.8333`, so the dual price
and the re-solve agree.

### 5.3 Branch-and-bound trace, fifteen nodes

Root: `z_LP = 12.8333`, with one fractional variable, A, at `7/12 = 0.5833`. The root solution here is
unique, which matters for reproducibility: when two items share a density and both enter the greedy
prefix, the relaxation has a face of alternative optima, different solvers return different vertices
with the same value, and the trees below diverge while the optimum does not. Here the tie is between A
and C at `0.8333`, but C never enters the prefix. Two independent checks of uniqueness: a solution
using C whole gives B + C = 14 kg and 12 value with 1 kg left at D's density `0.8` for `0.8`, total
`12.8 < 12.8333`; a solution using A whole leaves 3 kg, worth at most `3 × 0.875 = 2.625` against A's
10, total `12.625`. Neither reaches the root bound, so only the A-fraction vertex is optimal.

Branching order below is the `x_j = 0` child first, then `x_j = 1`, branching on the fractional
variable with the largest fractionality.

| Node | Fixings | LP bound | Fractional | Action | Incumbent after |
|---|---|---|---|---|---|
| 0 | none | 12.8333 | A = 0.5833 | branch on A | none |
| 1 | A = 0 | 12.8000 | D = 0.2000 | branch on D | none |
| 2 | A = 0, D = 0 | 12.0000 | none | integral 12, record as incumbent | 12 |
| 3 | A = 0, D = 1 | 12.6667 | C = 0.3333 | bound > 12, branch on C | 12 |
| 4 | A = 0, D = 1, C = 0 | 11.0000 | none | integral 11, prune by bound | 12 |
| 5 | A = 0, D = 1, C = 1 | 12.5000 | B = 0.5000 | bound > 12, branch on B | 12 |
| 6 | also B = 0 | 9.0000 | none | integral 9, prune by bound | 12 |
| 7 | also B = 1 | infeasible | — | 11 + 8 = 19 kg > 15, prune | 12 |
| 8 | A = 1 | 12.6250 | B = 0.3750 | bound > 12, branch on B | 12 |
| 9 | A = 1, B = 0 | 12.5000 | C = 0.5000 | bound > 12, branch on C | 12 |
| 10 | A = 1, B = 0, C = 0 | 12.4000 | D = 0.6000 | bound > 12, branch on D | 12 |
| 11 | also D = 0 | 10.0000 | none | integral 10, prune by bound | 12 |
| 12 | also D = 1 | infeasible | — | 12 + 5 = 17 kg > 15, prune | 12 |
| 13 | A = 1, B = 0, C = 1 | infeasible | — | 12 + 6 = 18 kg > 15, prune | 12 |
| 14 | A = 1, B = 1 | infeasible | — | 12 + 8 = 20 kg > 15, prune | 12 |

Node-by-node derivations of the non-obvious bounds:

- **Node 1.** A out, capacity 15 kg for B, C, D. Greedy: B 8 kg (7), C 6 kg (5) — 14 kg, 12 value,
  1 kg left; D takes `1/5 = 0.2` for `0.8`. Bound `12.8`, fractional D.
- **Node 3.** D in (5 kg, 4), capacity 10 kg: B 8 kg (7), 2 kg left, C takes `2/6 = 0.3333` for
  `5 × 0.3333 = 1.6667`. Bound `4 + 7 + 1.6667 = 12.6667`.
- **Node 5.** D and C in (11 kg, 9), capacity 4 kg, B needs 8 kg: `B = 0.5` for `3.5`. Bound
  `9 + 3.5 = 12.5`.
- **Node 8.** A in (12 kg, 10), capacity 3 kg, B needs 8 kg: `B = 0.375` for `2.625`. Bound
  `10 + 2.625 = 12.625`.
- **Node 9.** A in, B out, capacity 3 kg, C needs 6 kg: `C = 0.5` for `2.5`. Bound `12.5`.
- **Node 10.** A in, B and C out, capacity 3 kg, D needs 5 kg: `D = 0.6` for `2.4`. Bound `12.4`.

Frontier empties after fifteen LP solves, zero cuts, and the bracket closes at `z_ub = z_inc = 12`.
**The instructive part is the A = 1 subtree (nodes 8–14).** Its bounds are 12.625, 12.5 and 12.4 —
all strictly above the incumbent 12, so none of them can be pruned by the bound test, and the search
must descend seven more nodes purely to discover that each of them ends integral-and-worse or
infeasible. Bounds that are close to but above the incumbent are the expensive case in practice: the
bound test cannot help, and only better formulations or integrality arguments can.

### 5.4 Flooring collapses the expensive subtree

Every objective coefficient here is an integer, so `z_IP` is an integer and `floor(z_LP)` is a valid
bound at every node. Applying it:

```text
node 0  floor(12.8333) = 12   no incumbent yet  -> must branch
node 1  floor(12.8000) = 12   no incumbent yet  -> must branch
node 2  integral 12                             -> incumbent = 12
node 3  floor(12.6667) = 12   12 <= 12          -> prune without descending
node 8  floor(12.6250) = 12   12 <= 12          -> prune without descending
```

Five node LPs instead of fifteen, with the same proof and the same answer. The general rule: when all
objective data are integral and the incumbent is integral, replace the pruning test `z_LP ≤ z_inc`
with `floor(z_LP) ≤ z_inc` in a maximisation. The saving is largest exactly where the plain bound is
uselessly weak — bounds in `(z_inc, z_inc + 1)`.

The same floor argument at the root gives `floor(12.8333) = 12`, which equals the value of the greedy
plan `B + C` obtained without any search. So a pipeline that (a) runs a density-greedy heuristic, (b)
verifies the plan against the capacity row, and (c) compares with the floored root bound, resolves
this instance in one LP and one feasibility check. That is not a heuristic result: it is a proof,
because the flooring argument is valid whenever the objective data are integral.

**When the shortcut is illegal.** If any objective coefficient is fractional (a cost of `9.37` per
unit), `z_IP` need not be an integer and `floor(z_LP)` may lie below the true optimum, which would
prune a better plan. The legal version then requires a common denominator: multiply through by the
least common multiple of the denominators, floor in the scaled space, and divide back. For a single
coefficient with denominator 100, that is equivalent to a `0.01`-granularity floor — safe, but much
weaker, which is why unit costs are usually converted to integer cents at model build.

### 5.5 Sensitivity of the gap to capacity

| Capacity kg | z_LP | z_IP | Absolute gap | Relative gap | Reason |
|---|---|---|---|---|---|
| 13 | 11.1667 | 11 | 0.1667 | 1.5152% | B + 5/12 of A; best integer B + D |
| 14 | 12.0000 | 12 | 0.0000 | 0.0000% | B + C fills exactly |
| 15 | 12.8333 | 12 | 0.8333 | 6.9444% | the traced instance |
| 16 | 13.6667 | 12 | 1.6667 | 13.8889% | B + 8/12 of A; no integer plan beats 12 |
| 20 | 17.0000 | 17 | 0.0000 | 0.0000% | B + A fills exactly |

Recompute one row independently, capacity 13: B 8 kg, remaining 5 kg; A takes `5/12` → `4.1667`;
`z_LP = 7 + 4.1667 = 11.1667`. Integer: B + D = 13 kg, value 11; A alone 10; C + D 9. So `z_IP = 11`
and the gap is `0.1667/11 = 1.5152%`. Correct.

Recompute capacity 16: B 8 kg (14 kg and 12 value would need C, which is not next in density order;
A is, at 0.8333), so remaining 8 kg goes to A as `8/12` → `6.6667`; `z_LP = 7 + 6.6667 = 13.6667`.
Integer plans at 16 kg: B + C = 14 kg value 12; B + D = 13 kg value 11; A + C = 18 kg infeasible;
A + D = 17 kg infeasible; A alone 10. So `z_IP = 12` and the gap is `1.6667/12 = 13.8889%`.

**Interpretation of the table.** The gap is not monotone in capacity: it vanishes at 14 and 20 where
the greedy order lands exactly on an integer plan, and peaks at 16. A procurement team that tunes
capacity to "reduce the gap" is tuning an artefact of the data, not model quality.

**Sensitivity note.** All values above scale linearly with the value column; doubling every `v_j`
doubles both `z_LP` and `z_IP` and leaves every relative gap unchanged, because the ratio is
homogeneous of degree zero in the objective. Changing a weight, by contrast, changes the ordering and
can move the gap discontinuously — `w_A = 11` would put A before C at density `0.9091`, changing both
the bound and the optimum. The gap is therefore sensitive to the *ordering*, and ordering is set by
the weights; that is the quantity to perturb in a robustness check.

## 6. Visual / Geometric Intuition

### 6.1 The lattice inside the polyhedron

```text
value
 14 |            *
 13 |         .  .        * integer point
 12 |      *  .LP.  *     . fractional (relaxation) point
 11 |   *  .      .  *    / boundary of the LP polyhedron
 10 |*  .          .  *
    +-------------------- capacity (kg)
      0  4   8   12  16  20
```

The relaxation slides to a vertex that can be fractional; the integer points sit at lattice
coordinates inside. The gap is the vertical distance between the highest lattice point and the
highest polyhedron point along the objective direction.

### 6.2 What a cut does, geometrically

```text
before cut                      after cut x1 <= 1
   x2                             x2
    |      /                       |      /
  2 |     /  <-- LP vertex (1.8,0) |     /  +---- x1 = 1
    |    /                         |    /  |
  1 |   /                          |   /   |  LP vertex now (1,1), integral
    |  /                           |  /    |
    +---------- x1                 +---------- x1
       1  2  3                       1  2  3
```

The cut is a vertical line at `x1 = 1`. It removes the fractional vertex and every point to its
right, none of which contain an integer point (because `x1 ≥ 2` already violated capacity). The new
polyhedron's best vertex sits on a lattice point, so the relaxation becomes exact — that is what
"closing the gap in one round" looks like geometrically.

### 6.3 The branch-and-bound tree of §5

```text
                       [0] 12.8333  A=0.5833
                        /                \
             [1] A=0  12.8000      [8] A=1  12.6250  B=0.375
              /            \             /             \
   [2] D=0 12.0     [3] D=1 12.6667   [9] B=0 12.5   [14] B=1 INF
      integral 12    /          \          /      \
                [4] C=0    [5] C=1     [10] C=0  [13] C=1 INF
                  11.0      12.5        12.4
                  prune      /   \        /    \
                      [6] B=0  [7] B=1 [11] D=0 [12] D=1 INF
                         9.0     INF     10.0
                         prune            prune
```

The left child of the root (`A = 0`) carries the good news: its first descendant is the integral plan
of value 12, which sets the incumbent that prunes nodes 4, 6, 11. The right child (`A = 1`) carries
the cost: fixing A consumes 12 of the 15 kg, so every remaining bound is a near-miss in `(12, 12.625)`
and the bound test cannot prune any of the internal nodes 8, 9, 10. Seven nodes are spent proving that
branching into A is second best, which is the general shape of a wasteful subtree: bounds just above
the incumbent.

### 6.4 The gap funnel

```text
gap %  frontier bound minus incumbent, over the incumbent
 6.94 |*
 6.67 |  *                       node 1, bound 12.8000
 5.56 |      *                   node 3, bound 12.6667
 5.21 |          *               node 8, bound 12.6250
 4.17 |              *           node 9, bound 12.5000
 3.33 |                  *       node 10, bound 12.4000
 0.00 |                      *   incumbents 11 and 12 are found on the other branch
      +------------------------ nodes (0,1,3,8,9,10, then the incumbent branch)
```

Two distinct endings look identical in a log: the frontier emptying (a proof) and the solver stopping
on a tolerance (an estimate). Here the bound falls by `6.94 - 3.33 = 3.61` percentage points across
six node solves, roughly 0.6 points per node, which is the signature of a weak subtree — a tight
formulation would drop from 6.94% to 0% in one or two nodes. A slow, evenly stepped funnel is
therefore diagnostic of loose linking rows, not of a large instance.

### 6.5 Symmetry as a geometric redundancy

Eight interchangeable machines give `8! = 40,320` label permutations of the same schedule. Each is a
distinct point in variable space and an identical point in operational space. The search cannot know
this from the constraints alone, so it may branch on which machine is "first" and duplicate an entire
subtree 40,320 times. Symmetry-breaking constraints pin an ordering (machine `i` may be used only if
machine `i-1` is), collapsing the orbit to one representative.

## 7. Comparison / Selection Table

**7.1 Which solution technology for which structure.**

| Structure | Technology | Why | Watch out |
|---|---|---|---|
| Assignment, transportation, min-cost flow | LP (network simplex) | TU guarantees integral vertices | any extra non-network row destroys TU |
| 0/1 knapsack, W small enough for a table | dynamic programming `O(nW)` | exact, no solver | cost grows with W, not with n |
| Fixed-charge location, 10³ binaries | MILP with tight big-M | bound quality drives nodes | M sizing decides runtime |
| Scheduling with disjunctive orderings | MILP or CP-SAT | global constraints propagate | MILP needs time-indexed or ordering binaries |
| Routing with many vehicles | MILP + subtour separation, or heuristic | compact model is exponentially large | a "compact" VRP model is usually weak |
| Feasibility-only rostering | CP-SAT | propagation beats linear bounds | objective quality may be poor |
| Large nonlinear cost | MILP piecewise linearisation | keeps linearity | each piece costs binaries or SOS2 |

**7.2 Ways to express an on/off condition.**

| Mechanism | Row written | Bound quality | Portability | Failure mode |
|---|---|---|---|---|
| tight big-M | `x <= M y`, `M = min(U,D)` | equals hull in single-row case | universal | M too small cuts plans |
| loose big-M | `x <= 10⁶ y` | very weak, wide slab | universal | node explosion, no error message |
| indicator constraint | `y = 1 ⇒ x <= U` | solver derives its own bound | needs native support | silent reformulation on export |
| SOS1 | set of at most one nonzero | strong on the disjunction | wide | ordering decisions may be unhelpful |
| convex combination | `x = Σ U_i λ_i`, `Σ λ_i = y` | exact hull for one row | universal | more columns; slower LPs |

**7.3 Search strategies.**

| Strategy | Mechanism | Good when | Cost |
|---|---|---|---|
| pure branch-and-bound | branch, bound, prune | small integer count, tight root bound | no cut library benefit |
| branch-and-cut | add CG/MIR cuts at nodes | dense covering/packing rows | cut management, numerics drift |
| branch-and-price | column generation inside the tree | set partitioning with huge column sets | implementation heavy, pricing must be exact |
| feasibility pump / heuristics | round and repair to find incumbents | proving feasibility is the bottleneck | no bound improvement |

**7.4 Reading a solver's status word.**

| Status | Meaning | Action |
|---|---|---|
| optimal | frontier empty, `z_ub = z_inc` | accept; still verify the plan against business rules |
| optimal, tolerance | gap ≤ tolerance, frontier not empty | quote the gap; do not say "proved optimal" |
| feasible | incumbent exists, bound far away | usable plan, unknown distance from optimum |
| infeasible | no integer point exists | check M sizing and row directions before trusting |
| unbounded | relaxation unbounded | look for a missing capacity row |
| time/node limit | stopped on budget | report both bound and incumbent, never only the plan |

**7.5 Model-choice decision rule.** If the constraint matrix is a network, use the network model and
expect an integral LP. If it is not, and the binary count is under a few thousand with a tight
formulation, use MILP. If the model is dominated by sequencing and logical constraints, use CP-SAT.
If none of these fits inside the decision window, use a heuristic from
`06_metaheuristics.md` and report the bound you could not close.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom as seen by a user, the competing causes that produce that same symptom,
and the check that discriminates between them.

### 8.1 "Optimal" plan violates the operational rule

**Symptom.** The solver reports optimal; the plan cannot be executed — a truck carries 20.000001
tonnes against a 20-tonne limit, or a shift count is 3.9999998.

**Competing causes.** (a) The constraint was written with tolerance headroom by design; (b) the
integrality tolerance accepted a fractional value; (c) the constraint was scaled so that the
violation is below the feasibility tolerance; (d) the rule was never encoded at all and the plan is
"optimal" for a different model.

**Discriminating check.** Evaluate every row at the returned solution in exact rational arithmetic
and print the maximum violation. If the largest violation is at the 1e-6 scale, it is tolerance (b)
or (c); if a whole rule is absent from the row list, it is (d). Then re-solve with tightened
tolerances and see whether the violation moves — tolerance effects move, missing-row effects do not.

**Correction and prevention.** Round the accepted integers explicitly and re-check feasibility before
handing the plan downstream; add a post-solve validator to the pipeline that uses integer arithmetic.
For unit counts above about 10⁵, tighten `ε_i` to 10⁻⁹ or scale the variable.

**Verification.** The validator passes on the historical plan set and fails on a deliberately injected
0.0000005 violation.

### 8.2 Model reports infeasible, but a feasible plan is known

**Symptom.** Status infeasible while operations has been running a workable schedule for months.

**Competing causes.** (a) big-M too small, cutting off the plan; (b) an inequality direction reversed;
(c) a bound typed from the wrong period (last year's capacity); (d) an equality that should be `≤`;
(e) presolve propagated an integer bound from a fractional value.

**Discriminating check.** Fix the variables to the known plan and solve the feasibility problem. If
it is infeasible, the conflict is in the rows or bounds, not the search; then delete rows one at a
time and re-test to find the blocking row. If fixing the plan is feasible but the free solve is
infeasible, the conflict is in bounds implied by integer propagation — look for a rounding of a
continuous bound into an integer one.

**Correction and prevention.** Replace every big-M with `min(capacity, demand)` computed in code from
the data, not typed as a literal. Assert row direction in a unit test with a hand-built feasible
point.

**Verification.** The unit test containing the historical plan passes and the infeasibility
disappears with the corrected M.

### 8.3 Solve time explodes with no change in the answer

**Symptom.** The optimum is the same as last month's, but the run moved from 40 seconds to four hours.

**Competing causes.** (a) big-M inflated somewhere in an edit; (b) symmetry introduced by adding
interchangeable resources; (c) a new but valid constraint weakened the relaxation's structure; (d)
data drifted so the root bound got worse (capacity near a break point as in §5.5); (e) solver version
or parameter default changed.

**Discriminating check.** Compare root bounds, not runtimes. Same root bound plus more nodes means
symmetry or branching; worse root bound means formulation or data. Then diff the model file's
coefficients against the previous version and check the largest coefficient magnitude.

**Correction and prevention.** Store the root bound and root gap alongside every run as a regression
metric. A model change that reduces the root bound materially is a formulation change and needs the
same review as a code change.

**Verification.** The stored root bound returns to its previous value.

### 8.4 Bound improves but the incumbent never appears

**Symptom.** The solver prunes for hours, gap falls to a few percent, no integer solution is found at
all.

**Competing causes.** (a) The feasible integer points are extremely sparse relative to the
polyhedron; (b) an integrality tolerance so tight that legitimate lattice points are rejected; (c)
the objective pushes the search toward the fractional region; (d) a linking row demands an exact
combination that only a rare assignment satisfies.

**Discriminating check.** Run a feasibility-only version (constant objective) with a short time
limit. If a feasible point appears quickly, sparsity in the objective direction (a)/(c) is the cause;
if not, the integer set itself is thin or contradictory (d), or tolerances are rejecting points (b).

**Correction and prevention.** Seed with a heuristic incumbent — greedy by density, rounding plus
repair, or a short local search — and set the solver to accept any feasible point early. An
incumbent, even a mediocre one, converts pruning from "cannot" to "can".

**Verification.** The seeded incumbent appears in the log at node 0 and the gap trajectory becomes
monotone in both directions.

### 8.5 Numerically unstable model returns contradictory results

**Symptom.** Two runs with a permuted row order give different optima; or a run declares infeasible
that was feasible yesterday.

**Competing causes.** (a) Coefficient range spanning many orders of magnitude; (b) near-parallel rows;
(c) big-M values of 10⁶ against right-hand sides of 1; (d) a tiny coefficient (1e-4) used as a
logical switch instead of a binary.

**Discriminating check.** Compute the ratio of the largest to smallest nonzero coefficient absolute
value. Above roughly 10⁹, double precision (about 16 significant digits) leaves under seven usable
digits and basis factorisations become unreliable. Then resolve with rows scaled to unit magnitude:
if the answer stabilises, scaling was the cause.

**Correction and prevention.** Choose consistent units — currency in thousands, mass in tonnes, time
in hours — so coefficients land within a few orders of magnitude. Replace tiny-coefficient switches
with binaries and explicit bounds.

**Verification.** The permuted-order re-solve returns the identical objective to the last cent.

### 8.6 The gap is reported but nobody reads it

**Symptom.** A dashboard says "optimal" and a decision is made on a plan that is 8% from the
achievable bound.

**Competing causes.** (a) Solver status word surfaced without the gap field; (b) the gap tolerance was
set to a loose default to make the job finish; (c) the bound is being read from a node LP rather than
the frontier maximum; (d) a zero gap is reported because the incumbent was seeded as the bound.

**Discriminating check.** Re-read the raw log for the "best bound" and "best integer" columns at
termination; compute `g` by hand as in §4.2. If the printed gap is 0 but the bound column differs
from the incumbent column, the pipeline is mis-mapping fields (c) or (a).

**Correction and prevention.** Make bound, incumbent, gap, and node count mandatory columns in the
result record, and refuse to publish a plan whose gap exceeds the decision's stated tolerance. State
the tolerance in the decision record, not in the code.

**Verification.** A run stopped at 5% tolerance produces a record whose gap field reads 0.05 and
whose status field reads "limit", not "optimal".

### 8.7 Units drift between model and data feed

**Symptom.** The plan is feasible in the model but overflows warehouse capacity by a factor of 1000.

**Competing causes.** (a) Kilograms in the data, tonnes in the model; (b) currency in units versus
cents; (c) a per-hour rate applied to a per-minute time grid; (d) a percentage stored as 0.15 in one
table and 15 in another.

**Discriminating check.** Compute total demand from the raw data in its native unit and compare with
the sum of right-hand sides of the capacity rows. A factor that is a power of ten is a unit error; a
factor near 60 or 3600 is a time-unit error.

**Correction and prevention.** Name units in variable names and in the data schema, and assert a
dimensional total per period at model build time. Refuse to build a model whose declared capacity is
more than an order of magnitude away from observed demand.

**Verification.** The build-time assertion fires on a deliberately mis-scaled fixture.

### 8.8 Symmetry consumes the whole budget

**Symptom.** Node count in the millions with many nodes having identical bounds.

**Competing causes.** (a) Genuine interchangeability of resources; (b) duplicated columns created by
an unmerged data join; (c) a time-indexed formulation with interchangeable idle blocks.

**Discriminating check.** Count distinct objective values among the first few hundred explored nodes:
many identical bounds at the same depth indicates an orbit. Then test whether the column count equals
the distinct entity count from the master data.

**Correction and prevention.** Add lexicographic ordering constraints (`y_i ≥ y_{i+1}` for identical
facilities), deduplicate at the data layer, and use solver symmetry detection where available.

**Verification.** The duplicate-bound count at depth `d` falls from thousands to a handful.

## 9. Complexity & Cost Analysis

### 9.1 Theoretical complexity

General MILP is NP-hard: 0/1 knapsack already is, and it is a single-row model. Therefore no
polynomial algorithm exists for the general case unless `P = NP`, and any claim of a general-purpose
polynomial MILP solver is wrong. What is polynomial: the LP relaxation (interior-point methods in
polynomial time), and MILP with a fixed number of integer variables (Lenstra's algorithm, polynomial
for fixed dimension but with a constant that is astronomical in practice).

**Tree size.** With `n` binary variables the complete binary tree has `2^(n+1) - 1` nodes. For
`n = 40`:

```text
nodes = 2^41 - 1 = 2,199,023,255,551
at 800 nodes/s (four cores, 5 ms per node on a light model):  2.199e12 / 800 = 2.749e9 s
2.749e9 s / 86,400 = 31,814 days
31,814 / 365.25 = 87.1 years

for contrast, at the heavier 10 ms per node priced in 9.2 (400 nodes/s): 2.199e12 / 400 = 5.5e9 s
5.5e9 / 86,400 = 63,628 days = 174.2 years
```

That number is the reason bound quality dominates hardware. Halving the tree per unit of bound
improvement beats any constant-factor speed-up.

### 9.2 Per-node cost model

```text
node cost = t_LP + t_cut + t_heuristic
ILLUSTRATIVE measurements: t_LP = 4 ms (dual simplex warm start), t_cut = 5 ms, t_heur = 1 ms
node cost = 10 ms -> 100 nodes/s per core
four cores -> 400 nodes/s
900 s budget -> 400 * 900 = 360,000 nodes

a 5,000,000-node instance:
5.0e6 / 400 = 12,500 s = 3.472 h wall clock
core-hours = 12,500 * 4 / 3,600 = 13.89 core-hours
at $0.05 per vCPU-hour (ILLUSTRATIVE rate): 13.89 * 0.05 = $0.694
```

### 9.3 Where the time actually goes

| Component | Typical share of wall clock | Lever |
|---|---|---|
| node LP re-solves | 40–70% | formulation strength, warm starts |
| cut generation and management | 10–30% | cut family selection, round limits |
| heuristics for incumbents | 5–20% | seeding with a greedy plan |
| presolve | one-off, seconds | catches most modelling errors |
| I/O and model build | 1–5% but unbounded if naive | build once, solve many |

The distribution explains a practical asymmetry: improving the formulation by tightening one M is
free at solve time and can cut nodes by an order of magnitude, while increasing cores by 4 only
divides wall clock by about 3.5 because cut management and node selection serialise.

### 9.4 Memory

Per open node the solver keeps a basis, bound changes, and inherited cuts. Take 120 KB per node
(ILLUSTRATIVE): `10⁵` open nodes cost `10⁵ × 120 KB = 1.2 × 10⁷ KB = 1.2 × 10¹⁰` bytes, which is
12 GB in decimal units or 11.44 GiB in binary units — the 5% difference matters when a machine is
provisioned in the other convention. Breadth-first strategies
therefore die on memory before they die on time; depth-first keeps the frontier near `n × 120 KB` but
finds incumbents later. This is the real reason incumbent seeding matters: it converts a memory-bound
search into a pruned one.

### 9.5 Cost of the alternatives

| Approach | Time scale | Certificate | Engineering cost |
|---|---|---|---|
| LP relaxation only | milliseconds | none (bound only) | lowest |
| MILP to proven optimality | seconds to hours | full bracket closed | low, if formulation is tight |
| MILP to 1% gap | often 10× faster than proof | bracket, tolerance stated | low |
| Dynamic programming `O(nW)` | with `n = 200, W = 10⁶`: `2 × 10⁸` cells, about 800 MB at 4 bytes/cell | exact | medium memory |
| Local search heuristic | milliseconds per restart | none | medium |
| Manual analyst plan | hours of labour | none | high, non-reproducible |

The dynamic-programming row is included because it shows the trap: `2 × 10⁸` cells is comfortable at
`W = 10⁶` and impossible at `W = 10⁹`, where the table alone would be 800 GB. Pseudo-polynomial cost
is a capacity cost, and capacity is a physical quantity that grows.

### 9.6 Break-even against analyst time

An analyst needs 3 hours at 60 currency/hour to hand-build a schedule: 180 currency. An 8-vCPU solve
at 0.05 per vCPU-hour costs `8 × 0.05 / 3600 = 1.111e-4` currency per second. Break-even:
`180 / 1.111e-4 = 1.62e6 s = 450 hours` of wall clock. The economics therefore favour solving almost
always — but only because the model is reused. A one-off model that takes two engineer-weeks to build
must be justified by the decisions it changes, not by the solve cost.

### 9.7 Scaling rules that survive contact

- Bound the coefficient range to about six orders of magnitude; beyond nine, expect numeric failure.
- Keep binaries under a few thousand unless the formulation is a network or the root bound is tight.
- Prefer an incumbent-plus-flooring proof (see §5.4) to a longer search.
- Price a run in node-seconds, not minutes: `nodes × node cost` is reproducible across machines.

## 10. Key Theorems / Results Reference

**T1. Weak duality (LP).** For a maximisation LP, any feasible dual solution gives an upper bound.
*Use:* every MILP bound is an LP dual bound at some node. *Applicability:* linear objective, linear
constraints, dual feasible. *Check:* `cᵀx ≤ bᵀy` at the returned primal/dual pair.

**T2. LP relaxation bound.** `z_IP ≤ z_LP` for maximisation, with equality iff the relaxation has an
integral optimum. *Use:* the entire pruning mechanism. *Check:* compare incumbent and root bound;
equality at the root means integrality is free and no search is needed.

**T3. Total unimodularity.** If `A` is TU and `b` integral, every vertex of `{x ≥ 0 : Ax ≤ b}` is
integral. *Use:* assignment, transportation, min-cost flow solve as LPs. *Check:* is the matrix a
node-arc or bipartite incidence matrix with no extra rows?

**T4. Birkhoff–von Neumann.** The doubly stochastic matrices form the convex hull of permutation
matrices. *Use:* the assignment LP's vertices are assignments, so the LP value equals the assignment
value, as computed in §4.3 (14, 20, 9, 10, 21, 16 over six permutations, minimum 9).

**T5. Knapsack LP structure.** A basic optimal solution of the 0/1 knapsack LP has at most one
fractional variable. *Use:* branching on that variable bounds the tree depth by the number of times
the split item changes. *Check:* print the relaxation solution; more than one fractional value means
the model has more than one capacity row.

**T6. Chvátal–Gomory validity.** For `u ≥ 0` and integer `x ≥ 0` feasible for `Ax ≤ b`, the inequality
`floor(uᵀA)x ≤ floor(uᵀb)` is valid. *Use:* mechanical cut generation without domain insight. *Check:*
evaluate the cut at every integer point of a small instance; in §4.3 the cut `x1 ≤ 1` holds at all
five integer points.

**T7. Gomory's finiteness.** For a pure-integer problem, repeated fractional cuts with a suitable
dual simplex rule terminate finitely. *Use:* justifies cutting as a complete method. *Caveat:* the
bound on rounds is enormous and floating-point drift can break the guarantee; practical solvers
combine cuts with branching.

**T8. Chvátal rank.** The number of rounding rounds needed to derive `conv(F)` from `P`. *Use:*
explains why one cut can be enough (§4.3, rank 1) or hopelessly insufficient. *Check:* if adding all
first-round cuts leaves a fractional vertex, the rank exceeds one.

**T9. Branch-and-bound completeness.** If every node is either pruned by a valid bound, pruned as
infeasible, or split into a partition of its integer points, then termination with an empty frontier
proves optimality. *Use:* the correctness argument for any branching scheme. *Check:* the branch
partition is exhaustive and disjoint — `x_j ≤ k` and `x_j ≥ k+1` with integer `x_j` covers all integers.

**T10. Formulation dominance.** If `P₁ ⊆ P₂` and both contain `F`, then `P₁` gives a bound at least as
strong everywhere. *Use:* prefer the tighter of two correct formulations even if it has more rows.
*Check:* compare root bounds on identical data; the tighter model never has the worse bound.

**T11. NP-hardness of 0/1 knapsack.** Knapsack is NP-hard (subset-sum reduces to it), so no
polynomial exact algorithm exists unless `P = NP`. *Use:* calibrates expectations; a solver that
"handles it" is doing search, not magic.

**T12. Pseudo-polynomial dynamic programming.** Knapsack is solvable in `O(nW)` time and `O(W)` space.
*Use:* exact for small integer capacities. *Check:* `W` must be a moderate integer; the cost is in the
capacity value, not the item count.

**T13. Lenstra's theorem.** Integer programming in fixed dimension is polynomial. *Use:* justifies
special-purpose exact algorithms for 1–3 integer variables. *Caveat:* the constant is impractical;
it does not make 40 binaries tractable.

**Applicability summary.** T1, T2, T9 require only linearity and the stated objective sense. T3, T4
require the exact matrix structure and integral right-hand sides. T5 requires a single capacity row.
T6 requires `x ≥ 0` integer and nonnegative multipliers. T7 requires pure integrality, not mixed.
T12 requires integral capacity. Misapplying T3 by adding one non-network row is the most frequent
error in practice, because the model still solves — just slowly and with a gap.

## 11. Interview / Exam Questions

### Q1. Define the relative gap and state what a run with a 0.4% gap proves.

**Answer.** `g = (z_ub - z_inc)/max(|z_inc|,1)`. A 0.4% gap proves the incumbent is within 0.4% of
the best possible objective in the multiplicative sense; it does not prove optimality.
**Reasoning.** The bracket `[z_inc, z_ub]` contains the optimum; the width relative to the incumbent
is the only honest statement available before the frontier empties.
**Follow-up.** What changes if the objective can be negative? The guard becomes material: with
`z_inc = -0.5` the unguarded ratio `(z_ub - z_inc)/z_inc` is negative and meaningless, while
`max(|z_inc|, 1)` gives a defined, conservative gap. **Red flag.** "It proves optimal to 0.4% of the
solver's tolerance." **Scoring.** 2 = bracket and multiplicative meaning plus the sign guard; 1 =
formula only; 0 = claims optimality.

### Q2. A 0/1 knapsack has items (8 kg, 7), (12 kg, 10), (6 kg, 5), (5 kg, 4) and capacity 15 kg. Give the LP bound and the integer optimum.

**Answer.** Densities are 0.875, 0.8333, 0.8333, 0.8. Greedy takes 8 kg (value 7), then 7/12 of the
12 kg item for 5.8333: `z_LP = 12.8333`. Integer optimum is the 8 kg + 6 kg pair at 14 kg, value 12.
**Reasoning.** Fractional fill of capacity is optimal for one-row knapsack; enumeration of 16 subsets
gives 12 as the best feasible value.
**Follow-up.** Why is `floor(12.8333) = 12` a proof here? Because every value coefficient is an
integer, `z_IP` must be an integer, so the floored relaxation bound is a valid upper bound on any
integer plan; since a 12-valued plan exists (`B + C`), nothing better can exist. With a value like
`10.5` the floor would be invalid and the bracket would stay open.
**Red flag.** Reporting 12.8333 as the answer to the business question. **Scoring.** 2 = both values
plus the flooring argument; 1 = LP only; 0 = rounds 12.8333 to 13.

### Q3. Compute the smallest defensible big-M for `x ≤ M y` when capacity is 800 units and demand is 650.

**Answer.** `M = min(800, 650) = 650`. With `M = 650` the relaxation's `y = 1` and the bound equals
17,850 currency on the illustrative cost `12,000 + 9 × 650`; with `M = 5000` the bound collapses to
7,410, a 58.5% gap.
**Reasoning.** `M` must cover the largest activity the row can take, which is bounded by demand as
well as capacity.
**Follow-up.** What breaks with `M = 400`? The integer plan serving 650 becomes infeasible.
**Red flag.** "Use 10⁶, it always works." **Scoring.** 2 = min of the two bounds plus the gap
consequence; 1 = min only; 0 = a round number.

### Q4. Derive one cut for `max 8x₁ + 5x₂` subject to `5x₁ + 4x₂ ≤ 9`, both integer and nonnegative, and show the result.

**Answer.** With `u = 1/5`: `x₁ + 0.8x₂ ≤ 1.8`; flooring gives `x₁ ≤ 1`. Adding it yields the LP
optimum `x₁ = 1, x₂ = 1`, value 13, equal to the integer optimum.
**Reasoning.** Nonnegative integer combinations of integer coefficients can be floored without losing
integer points.
**Follow-up.** Is `x₁ + x₂ ≤ 2` useful? It is valid (max sum is 2) but does not separate `(1.8, 0)`,
so it changes nothing. **Red flag.** "Cuts always help if valid." **Scoring.** 2 = derivation plus
the separation discussion; 1 = the cut only; 0 = an invalid cut.

### Q5. Why does the assignment problem need no branching, and what destroys that property?

**Answer.** Its constraint matrix is the bipartite incidence matrix, which is totally unimodular; with
integral right-hand sides every vertex is integral, so the LP optimum is an assignment (Birkhoff–von
Neumann). Adding a non-network row — a side constraint with non-±1 coefficients or a covering row
over both index sets — generally breaks TU and fractional vertices return.
**Reasoning.** Integrality comes from the polyhedron's vertex structure, not from the objective.
**Follow-up.** On the 3×3 instance of §4.3 the six permutation values are 14, 20, 9, 10, 21, 16; the
LP returns 9. **Red flag.** "Assignment is easy because it is small." **Scoring.** 2 = TU plus what
breaks it; 1 = "it is a special case"; 0 = "the solver detects it".

### Q6. A run ends with status "optimal" after 40 seconds but the log shows a best bound 3% above the incumbent. Diagnose.

**Answer.** The status reflects the gap tolerance, not a proof. Either the tolerance is 3% or the
pipeline is reading the wrong column. Check the tolerance parameter and the raw bound/incumbent
columns; if they differ, the run proved a 3% bracket.
**Reasoning.** Optimality requires an empty frontier or an explicitly met tolerance, and the latter
must be quoted as a bracket.
**Follow-up.** How would you close the last 3%? Tighten the formulation or seed an incumbent and use
the flooring argument when objective data are integral.
**Red flag.** "Optimal means proven." **Scoring.** 2 = tolerance distinction plus a check; 1 =
mentions tolerance; 0 = accepts the status.

### Q7. Estimate the cost of exploring 5,000,000 nodes at 10 ms per node on four cores.

**Answer.** 5e6 nodes / 400 nodes per second = 12,500 s ≈ 3.47 hours; core-hours = 12,500 × 4/3600 =
13.9; at 0.05 per vCPU-hour, about 0.69 currency.
**Reasoning.** Per-node cost sets throughput; cores divide wall clock but not total CPU unless the
search parallelises perfectly, which it does not.
**Follow-up.** Which is cheaper: halving nodes or doubling cores? Halving nodes, and it also halves
memory.
**Red flag.** Dividing 5e6 by 10 ms to get a time in milliseconds. **Scoring.** 2 = correct units and
cost; 1 = time only; 0 = unit error.

### Q8. The solver says infeasible; operations insists a valid schedule exists. List the checks in order.

**Answer.** (1) Fix variables to the known plan and re-solve — this separates row conflicts from
search issues; (2) compute `min(capacity, demand)` for every big-M and compare with the file; (3)
check inequality directions and period of each bound; (4) look for integer propagation from a
fractional bound.
**Reasoning.** Infeasibility claims are about rows and bounds, not about the branching strategy, so
testing a known point is the fastest discriminator.
**Follow-up.** What does it mean if the fixed-plan solve is feasible but the free solve is not? The
conflict is in bounds implied during presolve or integer propagation.
**Red flag.** "Infeasible means the data are wrong." **Scoring.** 2 = ordered checks including the
fixed-plan test; 1 = one check; 0 = reruns the model unchanged.

### Q9. Explain why the same feasible integer set can solve in 3 nodes under one formulation and 300,000 under another.

**Answer.** The formulations have different LP relaxation polyhedra. The tighter one's relaxation
lies inside or closer to the convex hull, so its node bounds prune early. The integer set and the
optimal value are identical; only the bound quality differs.
**Reasoning.** Branch-and-bound work is driven by how quickly `z_ub` falls to `z_inc`, and that rate
is a property of the relaxation, not of the integer points.
**Follow-up.** Give a concrete lever: `M = min(U, D)` versus `M = 10⁶`, which changes the bound from
exact to 58% away in the facility example.
**Red flag.** "More cuts always fix it." **Scoring.** 2 = polyhedron and bound rate; 1 = "better
model"; 0 = attributes it to solver speed.

### Q10. A binary comes back as 0.9999994 and the plan is executed as if it were 1. What is the risk and the fix?

**Answer.** With `ε_i ≈ 1e-6`, that value is accepted as integral. If the variable counts units at the
10⁶ scale, the accepted slack is about 1 unit — 600 kg when a unit is a 500 kg pallet. The fix is to
round to integers explicitly and validate the rounded plan against the rows, tightening `ε_i` or
rescaling when counts are large.
**Reasoning.** Tolerances are absolute in variable units, so their meaning depends on the magnitude of
the variable.
**Follow-up.** Why not set `ε_i = 0`? Basis arithmetic is floating point; zero tolerance makes the
solver loop or falsely report infeasibility.
**Red flag.** "0.9999994 is close enough to 1." **Scoring.** 2 = tolerance magnitude plus validation
step; 1 = rounding; 0 = no risk identified.

### Q11. When would you choose CP-SAT over MILP for the same scheduling problem?

**Answer.** When the model is dominated by sequencing and logical structure that linear constraints
encode weakly — disjunctive no-overlap, cumulative resource limits, table constraints — because
propagation on those global constraints prunes far better than the LP bound of a time-indexed
formulation. Choose MILP when the objective is a linear cost with tight coupling rows and the binary
count is moderate.
**Reasoning.** The two technologies carry different certificates: LP bounds versus propagation plus
search. Neither dominates.
**Follow-up.** What do you lose with CP-SAT? A dual bound in the LP sense; the reported bound comes
from the search itself and may be weak early. **Red flag.** "CP-SAT is always faster."
**Scoring.** 2 = structural criterion plus the certificate difference; 1 = names the tool only;
0 = a benchmark claim with no source.

### Q12. Compute the effect on the root bound of using `M = 800` instead of `M = 650` in the facility example, and say whether the optimal plan changes.

**Answer.** `M = 650`: `y = 1`, cost 17,850, gap 0. `M = 800`: `y = 0.8125`, cost 15,600, gap
`(17,850 - 15,600)/17,850 = 2,250/17,850 = 12.605%` (12.61% rounded). The integer optimum is
unchanged at 17,850; only the bound weakens, so the work grows while the answer stays the same.
**Reasoning.** big-M slack lets the relaxation pay a fraction of the fixed charge while carrying the
full flow; it cannot beat the true integer planning cost.
**Follow-up.** Which M value is tightest and legal? `min(U, D) = 650`. **Red flag.** "The optimum
changed, so M matters." **Scoring.** 2 = both computations plus the invariance of the optimum;
1 = one computation; 0 = claims the plan changes.

### Q13. Your model's coefficient range is `1e-3` to `1e6`. What is the numerical consequence and the remedy?

**Answer.** Range ratio 10⁹; double precision carries about 16 significant digits, so roughly seven
usable digits remain after scaling, and simplex basis updates can return contradictory optima under
row permutation. Remedy: harmonise units so the range is about 10⁶ or less, replace tiny-coefficient
switches with binaries, and verify by re-solving with permuted rows.
**Reasoning.** Conditioning of the basis matrix governs how much of the available precision survives
elimination steps.
**Follow-up.** How do you detect it without a numerical analysis? Permute row order and compare
objectives; instability shows as a changed answer.
**Red flag.** "Use a tighter feasibility tolerance." That often makes instability worse.
**Scoring.** 2 = ratio, digits, and the permutation test; 1 = scaling only; 0 = no consequence stated.

### Q14. Why is `x ≤ U y` with `y` binary described as already being the convex hull of the single-row disjunction, and when does that stop being true?

**Answer.** For `0 ≤ x ≤ U` and `y ∈ {0,1}`, the set's convex hull is the triangle with vertices
`(0,0)`, `(U,0)`, `(U,1)`; the rows `x ≤ U y`, `x ≥ 0`, `y ≤ 1` describe exactly that triangle, so the
relaxation is exact whenever `U` equals the tightest activity bound. It stops being true when a second
linking row or a second facility couples the flows, because then the hull has additional facets that
the one-row description misses.
**Reasoning.** Hull descriptions are per-disjunction; coupling constraints create combinations that a
single row cannot express.
**Follow-up.** What do you do for two facilities? Use flow conservation and per-facility linking rows,
and check the root bound; if it is slack, add the aggregate row `Σ x_j ≤ Σ U_j y_j`.
**Red flag.** "The linking row is always tight." **Scoring.** 2 = hull plus the coupling caveat;
1 = the row only; 0 = "big-M is always weak".

## 12. Quick Reference Card

| Item | Rule | Number / form |
|---|---|---|
| LP bound, knapsack | sort by `v/w`, fill, split the critical item | `z_LP = Σ_{j<k} v_j + (W − Σ_{j<k} w_j)·v_k/w_k` |
| Relative gap | bracket width over incumbent | `g = (z_ub − z_inc)/max(|z_inc|,1)` |
| Pruning test | maximisation | prune node when `z_LP(node) ≤ z_inc` |
| Proof shortcut | integral objective data | if `floor(z_LP)` equals the value of a feasible plan, the bracket is closed |
| Tight big-M | single linking row | `M = min(capacity, demand)`; never a round 10⁶ |
| Big-M cost of slack | illustrative facility case | `M=650` gap 0%; `M=800` 12.61%; `M=5000` 58.49% |
| CG cut | nonnegative multiplier then floor | `floor(uᵀA)x ≤ floor(uᵀb)` |
| Integrality for free | TU matrices | assignment, transportation, min-cost flow |
| Tree worst case | `n` binaries | `2^(n+1) − 1` nodes; `n=40` ⇒ 87.1 years at 800 nodes/s |
| Tolerances | absolute in variable units | `ε_i ≈ 1e-6`; at 1.2e6 units ⇒ 1.2 units of slack |
