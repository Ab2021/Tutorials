---
title: "Optimisation formulation and duality"
topic: "Optimisation formulation and duality"
collection: "optimisation-operations-research-and-simulation"
file_id: "01"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
prerequisites: ["linear algebra basics", "inequalities and units", "reading a table of numbers"]
pattern_tags: ["linear-programming", "duality", "shadow-price", "mip-formulation", "sensitivity"]
source_status: "OPEN"
owner_role: "Operations research reviewer"
source_index: "INDEX.md"
review_trigger: "A change to the objective, a constraint set, a unit convention, or a solver certificate claim."
---

# Optimisation formulation and duality

> **Evidence:** DERIVED for the duality statements, the vertex-optimality argument, and the sensitivity algebra; VERIFIED for the arithmetic traces in Sections 5, 8 and 9, which were recomputed independently; ILLUSTRATIVE for the production-mix instance, its coefficients, and every currency figure. No organisation's model, solver log, or realised saving is claimed here.

## 1. File Header

> **Purpose:** Turn a spoken operational complaint into a stated optimisation model — variables with units, one objective with a direction, hard constraints with owners — and then read the dual correctly enough to answer "what is one more unit of this capacity worth, and over what range".
> **Prerequisites:** You can solve two-variable linear systems by hand; you can convert a business sentence into an inequality without silently dropping a condition; you accept that a number without a unit is not yet evidence; you have seen a matrix written as rows of constraints.
> **Key outcome:** Given a decision, you produce (a) a model in a declared standard form, (b) the objective value at the optimum, (c) at least one dual value with its valid range, (d) one integrality or convexity caveat that limits what the model may be used to claim, and (e) the exact sentence the model does not license.
> **This file does NOT cover:** stochastic programming formulations, column generation, interior-point algorithm internals, semidefinite models, constraint-programming propagators, or simulation experiment design. Those belong to `03_simulation_experiment_plan.md`, `05_integer_programming.md`, and `06_metaheuristics.md`.
> **Level:** foundation. The algebra is elementary; the discipline is not.
> **Version / reviewed:** 2.0 / not independently reviewed. The `last_reviewed` field stays `null` until a named reviewer signs off.
> **Retrieval cue:** If someone presents a "cost per extra hour" figure without a range, jump to Section 4.4 on degeneracy and Section 8 row F-05.

### What decision this sheet supports

The sheet supports one decision per use: whether a proposed optimisation model is the right representation of the operational choice, and whether the numbers read off its solution may be acted on. That decision has three parts, and each has a cheap test.

1. **Is the objective the thing the business actually wants?** A model maximising margin while the plant is measured on on-time delivery will be optimised against the wrong quantity, and the solution will be correct and useless.
2. **Are the constraints the binding physics and policy, or a memory of last year's plan?** Constraints that encode habits rather than limits quietly forbid the improvement the model was built to find.
3. **Does the claimed interpretation survive the model class?** A dual value from a linear program is a rate of change under continuous, small, one-at-a-time perturbation. Under integrality, that interpretation degrades to a bound, and sometimes to nothing.

### What "correct" means here

A formulation is correct for this sheet when all of the following hold, and it is not correct merely because a solver returned the word `optimal`.

- every decision that the action plan varies appears as a variable, and every variable has a declared unit and a declared range;
- every parameter has a source, a vintage, and a stated unit, and the source is distinguishable from the variable it feeds;
- every constraint is either a physical limit, a contractual limit, or a policy limit, and the owner of that limit is named;
- the objective is a single scalar expression, and any additional desire is either a constraint or an explicitly declared weight;
- the optimal value has been recomputed from the reported variable values, not read from the solver's objective field alone;
- the model's claim has been matched to its class: exact optimum for a linear program, a bound plus a gap for a mixed-integer program, and a bound only for a relaxation.

The last item is the one that separates a formulation note from a decision memo. A mixed-integer model with a 4% gap has not told you the optimal plan; it has told you that no plan is better than the upper bound and that one plan achieves the lower bound.

### Source boundary

The unit source inventory is a topic index, not access to any primary text. Everything here is derived from the definitions in Section 3, with arithmetic shown so a reader can re-derive it. Coefficients in Section 5 are invented for teaching and marked ILLUSTRATIVE. Textbook attributions in Section 10 are named but not quoted; check the edition before citing a theorem number. Nothing in this file describes a real plant, a real solver log, or a measured saving.

**Takeaway:** The header's job is to make it impossible to run a model without first naming its units, its objective, its constraint owners, and the class of claim it can support.

## 2. Concept Map / Mental Model

The left column is what the business says. The middle column is what must be decided before any solver is opened. The right column is what may be concluded afterwards. Most formulation failures are a jump from the left column straight to the right column.

```text
  spoken problem                 formulation decisions                 licensed conclusion
  ---------------                ---------------------                 -------------------
  "we cannot make            ->  which quantities vary?           ->   variables + units
   enough of both"               continuous or integer?                 model class
                             ->  what is one unit worth?          ->   objective, single scalar
                             ->  what physically stops us?        ->   hard constraints + owners
                             ->  which limits are soft?           ->   penalty or epsilon constraint
                             ->  what is uncertain?               ->   scenario, robust, or excluded
                                                                        |
                                                                        v
                                                             solve, then check:
                                                             objective recomputed
                                                             constraint slacks replayed
                                                             dual read with its range
                                                             integrality gap reported
```

### The invariant that carries the whole sheet

One sentence covers the entire theory below: **for a linear program, the best value of the primal equals the best value of a dual problem whose variables are the constraint prices, and at the optimum every constraint that is not tight has price zero.**

That single identity does three jobs at once. It certifies optimality without trusting the solver (you can exhibit a dual feasible point of equal value). It prices capacity (a dual variable is the marginal value of one unit of a right-hand side). And it tells you when a price is meaningless (when the constraint is slack, or when the optimum is degenerate and the price is one point on a face).

### What changes between the variants

| Variant | What is added | What survives | What breaks |
|---|---|---|---|
| Pure linear program | nothing | exact vertex optimum, duality, prices | integrality of people, trucks, orders |
| Mixed-integer program | some variables are integer | LP bound, branch-and-bound certificate | duals are not prices; only bounds remain |
| Fixed-charge model | a binary switch multiplies a cost | correct economics of setup | linearity of the objective; the LP bound weakens |
| Piecewise-linear cost | several linear segments | linearity preserved with extra variables | a wrongly ordered segment set creates a nonconvex model |
| Chance-constrained model | a probability statement on feasibility | a linear reformulation in special cases | the dual no longer has a per-unit price meaning |

### Tracing the decision flow on one complaint

Take the sentence a plant manager actually says: "we cannot keep up with both orders, and the second one is losing us money". The flow forces seven answers before a solver is opened, and each answer changes the model rather than the algorithm.

1. **Which quantities vary?** If the plan chooses only a weekly mix, there are two structural variables; if it may also request overtime, a third appears with its own cost and its own contractual cap.
2. **Continuous or integer?** Bulk chemical is continuous, pallets are integer, whole shifts are integer, hours inside a shift are continuous. Declaring this per variable is the difference between a plan that can be executed and one that must be re-rounded by hand.
3. **What is one unit worth?** The contribution margin, not the selling price, because the price contains costs that do not move with the decision; using the price inflates every activity that consumes a fixed-cost resource.
4. **What stops us physically?** Machine hours, labour hours, storage volume, and a packaging line are the usual four; each needs one number, one unit, and one source.
5. **Which limits are negotiable?** A customer's minimum order quantity is contractual, an internal utilisation target is not, and that classification decides hard row versus penalty.
6. **What is uncertain?** Demand, yield, and absenteeism are the standard three; each is either modelled as a scenario or excluded in writing with the reason.
7. **What claim will be made?** If the number will justify capital spending, the sensitivity range belongs in the deliverable rather than in an appendix nobody opens.

The residual risk is organisational: steps 1 to 6 are answered by the analyst and step 7 is claimed by someone else. A mismatch between what was modelled and what is claimed is the most common reason a technically correct model produces a wrong decision.

| Clause in the complaint | Modelling decision it forces | Artefact that must exist afterwards |
|---|---|---|
| "cannot keep up" | a binding capacity row with an owner | slack report showing zero on that row |
| "both orders" | two structural variables with margins | objective row with two named terms |
| "losing money" | a cost comparison against a stated baseline | baseline definition and its vintage |
| "we could work extra hours" | an overtime variable with cost and cap | third column in the variable register |
| "the customer insists" | a contract row with an expiry date | row class `contract` and renewal date |
| "sometimes the machine breaks" | scenario, robust form, or exclusion | stated treatment in the exclusions list |
| "we need a number by Friday" | acceptable gap and stopping rule | gap target and the class of claim |

**Takeaway:** A model class is chosen by the claim you intend to make, not by the solver that happens to be installed.

## 3. Foundational Definitions

### 3.1 The five objects of a model

Every optimisation model in this collection contains exactly these five kinds of object, and mixing two of them is the most common structural error.

- **Decision variable** — a quantity the action plan sets. Notation `x_j`, unit declared, domain declared (`x_j >= 0`, `x_j` binary, `x_j` integer). If the plan cannot change it, it is not a variable.
- **Parameter** — a quantity fixed before the decision, with a source and a vintage. Notation `a_ij`, `b_i`, `c_j`. A parameter that is actually a management target belongs in the objective or in a constraint, not in a table nobody audits.
- **Objective** — one scalar function of the variables, with a direction (maximise or minimise) and a unit (currency, hours, kilograms of carbon). Two objectives mean two models or a declared weight.
- **Constraint** — a relation that must hold for the plan to be admissible, with a right-hand side and a named owner.
- **Index sets** — the products, machines, periods, or routes over which the above are repeated. Set definitions carry the grain: daily versus weekly buckets change every number downstream.

### 3.2 Feasible set, binding constraints, and slack

The **feasible set** `F` is the set of variable values satisfying all constraints and domains. It may be empty (infeasible), unbounded, a single point, a polytope, or a union of disconnected pieces.

A constraint `a_i'x <= b_i` is **binding** or **active** at a point when it holds with equality. Its **slack** is `s_i = b_i - a_i'x`, always non-negative for a feasible point. Slack is the single most informative output of a linear solve, and it is the field most often omitted from a report.

- Binding constraint with zero slack: the plan is pressed against a limit. Ask who owns the limit.
- Slack constraint: capacity is idle. Do not quote its price as a saving opportunity.
- Many constraints binding at once in low dimension: the optimum is degenerate, and prices are not unique.

### 3.3 Standard form and why it is declared

A **standard form** fixes one convention so that algebra, solver input, and reporting cannot drift apart. The convention used in this file is:

```text
maximise    c'x
subject to  A x <= b
            x >= 0
```

where `c` is the objective row (unit: objective unit per variable unit), `A` is the technology matrix (unit: resource unit per variable unit), and `b` is the right-hand side (resource unit). Every inequality direction, every free variable, and every equality is converted explicitly:

- a `>=` constraint is multiplied by `-1`, which flips its row in `A` and its sign in `b`;
- an equality `a'x = b` becomes two inequalities, or is kept as an equality in a solver that accepts them — but the reporting convention must say which;
- a free variable `x` is written `x = x_plus - x_minus` with both parts non-negative;
- a slack variable `s_i >= 0` turns `a_i'x <= b_i` into `a_i'x + s_i = b_i`.

### 3.4 Dual objects and what each one means

For the standard form above, associate one **dual variable** `y_i >= 0` with each constraint row `i`.

- `y_i` is a price per unit of resource `i`, in objective units per resource unit. If the objective is currency and the resource is machine-hours, `y_i` is currency per machine-hour.
- The **dual objective** is `b'y`, the total value of the capacity endowment at those prices. It is an upper bound on any feasible primal value and equals the optimum when strong duality holds.
- The **reduced cost** of a variable is `c_j - a_j'y`; it is the rate at which the objective would improve if that variable were forced up from zero by one unit. A variable at zero with positive reduced cost is being held down by something specific, and the reduced cost says how much.
- The **shadow price** is the same number as `y_i`, quoted with the *range* of `b_i` over which it stays constant. Without the range it is not a decision input.

### 3.5 Integrality, relaxation, and the gap

An **integer program** restricts some variables to a lattice. Its **LP relaxation** drops that restriction. Three numbers then matter, and confusing them is a classic error:

- `z_LP` — the relaxation's optimum, a bound in the optimistic direction (upper bound when maximising);
- `z_inc` — the best integer feasible plan found so far, the **incumbent**;
- `z_IP` — the unknown true integer optimum, satisfying `z_inc <= z_IP <= z_LP`.

The **absolute gap** is `z_LP - z_inc`; the **relative gap** is `(z_LP - z_inc) / |z_inc|` for maximisation. The **integrality gap** of the *model family* is the worst-case ratio over instances between the relaxation and the integer optimum; it is a property of the formulation, not of one run.

### 3.6 Formulation idioms and their costs

| Business phrase | Idiom | Extra objects | Main risk |
|---|---|---|---|
| "either do A or do B, not both" | binary `z` with `x_A <= M z`, `x_B <= M(1-z)` | 1 binary, 2 big-M rows | `M` too small cuts off valid plans |
| "if we run the line, it costs a setup" | fixed charge with `x <= M z` | binary per line-period | weak LP bound, slow proof |
| "at least 40 or not at all" | semi-continuous: `40 z <= x <= M z` | binary per item | forgetting the lower link |
| "the cost falls in bands" | piecewise-linear with ordered segment variables | one variable and one row per segment | segments entered out of order unless chained |
| "we would rather not, but may" | soft constraint with penalty `p * u` | violation `u >= 0` | penalty too small makes the constraint decorative |
| "must hold with probability 0.95" | chance constraint | distributional assumption | the assumption is rarely audited |

### 3.7 Objective vocabulary

- **Weighted sum** `w1 f1 + w2 f2`: one solve, but the weights are a policy statement, and the achievable trade-off curve is not visible from one point.
- **Lexicographic**: solve for `f1`, fix its optimum as a constraint, then optimise `f2`. Two solves, no arbitrary weights, but the priority order must be defended.
- **Epsilon-constraint**: constrain `f2 >= e`, maximise `f1`, and sweep `e`. This is the only variant that shows the trade-off curve, at the cost of one solve per point.

### 3.8 Scaling, coefficient magnitude, and numerical conditioning

A model can be structurally perfect and numerically unusable. Two quantities diagnose this before any solver runs.

- **Coefficient spread** — the ratio of the largest to the smallest absolute non-zero entry in `A`, `b`, and `c`. A spread of `10^6` or more means the solver is working with numbers that cannot all be represented to the same relative precision, and residuals will look like constraint violations.
- **Right-hand-side to coefficient ratio** — for each row, `b_i / max_j |a_ij|`. When this is below about `10^-3` or above `10^6`, the row is either scaled inconsistently with its neighbours or encodes a unit error (Section 8, row F-02).

Scaling is a legitimate preprocessing step: divide row `i` by `max_j |a_ij|` and record the division factor in the row metadata, so reported slacks can be converted back. What is not legitimate is scaling silently, because then the reported dual prices are in rescaled units and every downstream currency figure is wrong by that factor.

### 3.9 The baseline, and why it is part of the formulation

An optimisation result is a difference from something. The **baseline** is the plan the business would execute without the model: last period's plan, the current heuristic, or the manual spreadsheet. Three properties of the baseline must be written down, because a model cannot be compared against an undefined alternative.

- **Same grain and window.** A weekly model compared against a monthly actual is not a comparison.
- **Same accounting basis.** Contribution margin against gross margin produces a difference that is entirely definitional.
- **Same feasibility standard.** If the baseline violates a rule the model enforces, the difference includes the cost of compliance and must be reported as such, not as optimisation benefit.

The commonest overstatement in a model memo is not a wrong optimum; it is a baseline that was quietly made infeasible while the optimised plan was not.

### 3.10 The exclusions register

Every model excludes something, and the exclusions are the boundary of the claim. The register is a short list, written at formulation time, with one row per excluded effect:

| Excluded effect | Why excluded | Direction of bias | Consequence for the claim |
|---|---|---|---|
| Demand uncertainty | demand treated as a fixed cap | optimistic on volume | the plan is a capacity plan, not a sales forecast |
| Yield variation | nominal yield used | optimistic on output | realised output will be below planned |
| Setup times | ignored inside a period | optimistic on capacity | the plan needs a feasibility re-check at shift level |
| Labour skills | hours treated as fungible | optimistic on flexibility | a plan may be infeasible if skills are restrictive |
| Changeover sequence | not modelled | optimistic on cost | sequence-dependent cost must be added before quoting savings |
| Storage constraints | aggregated | optimistic on space | check against warehouse capacity separately |

Two rules keep the register honest. First, each row names the direction of the bias, because a one-sided error changes how the result should be used: an optimistic capacity plan can size equipment but not commit delivery dates. Second, no excluded effect may be silently re-included at reporting time — if a reviewer mentions setup times in the memo, the register must show they were modelled or the claim must be withdrawn.

### 3.11 Units algebra and the dimensional audit

Units can be treated as an algebra: every entry of `A` carries `resource unit per activity unit`, every `b_i` carries `resource unit`, every `c_j` carries `objective unit per activity unit`, and every `x_j` carries `activity unit`. Then `Ax <= b` is dimensionally consistent by construction, `c'x` is in objective units, and `y_i` must be in `objective unit per resource unit` for `b'y` to land in objective units.

The audit is mechanical and catches a specific class of error that review does not. For each row, multiply the unit string of every coefficient by the unit string of its column, and assert that all terms agree and that the result equals the unit of the right-hand side. Four entries fail in practice:

| Entry | Common unit string | Failure it produces |
|---|---|---|
| `a_ij` for a time-based resource | `hours per unit` written as `minutes per unit` | plan overstates feasible volume by 60x |
| `b_i` for a period budget | `hours per day` against a weekly model | capacity appears 7x too small or large |
| `c_j` for a margin | `currency per unit` mixed with `currency per 1000 units` | objective off by 1000x, plan unchanged |
| a penalty coefficient | `currency per unit` mixed with `currency per percent` | soft constraint becomes hard, or disappears |

The last entry is the subtle one because it does not change the plan's shape, only the ranking of violations: a penalty of 500 currency per unit against a violation measured in percent is either overwhelming or negligible depending on the scale of the percentage, and the two readings are indistinguishable from the solver log alone.

### 3.12 What "solving the model" is not

Three claims are routinely confused with a solve, and each is a different deliverable.

- **Feasibility** — a point exists. Returned as a plan with no objective claim. Useful when the question is "can we physically do this at all", and it is the correct output of an elastic run at model build time.
- **Optimality** — no feasible point is better, within the model. Returned with a certificate: matching dual values for a linear program, or a zero gap with a proof for an integer program.
- **Sensitivity** — how the optimum and the value move as parameters change. Requires additional solves and is not part of the solve output unless the solver exposes a basis.

A memo that says "we solved the model" has said none of these. The foundation discipline is to name which one is being claimed before the number is written down.

**Takeaway:** Defining the objects is the whole job; the algebra afterwards is arithmetic a machine can do.

## 4. Core Formulas with Full Derivation

Each result below is stated as a five-part block: formula, where it applies, its constraint or validity condition, a numerical example, and the interpretation it licenses. The blocks use the standard form of Section 3.3 and are labelled `FOR-01` onward so later sections can cite them.

### 4.1 Base case: a two-variable linear program

**Validity before use.** Two decision variables, linear objective, linear constraints, continuous variables, non-empty bounded feasible region. Under these conditions the optimum, if it exists, is attained at a vertex of the feasible polygon, and checking all vertices proves optimality.

**Mechanism.** Each constraint is a half-plane. Their intersection is a convex polygon. The objective is a linear functional whose level sets are parallel lines. Because the polygon is convex, translating a level line in the improving direction must last touch the polygon at a vertex or along a whole edge; if it touches along an edge, both endpoints are optimal, so a vertex is still optimal.

```text
FORMULA [FOR-01]: z* = max { c'x : Ax <= b, x >= 0 }  evaluated over V = vertices(F)
WHERE: c in R^2 (objective units per unit of x), A in R^{m x 2}, b in R^m;
       V is the finite set of points where two constraints meet with all others satisfied.
CONSTRAINT: F bounded and non-empty; otherwise the maximum may be unbounded or missed.
NUMERICAL EXAMPLE: max 40x1 + 30x2 s.t. 2x1 + x2 <= 100, x1 + x2 <= 80, x1 <= 40.
                  Vertices (0,0), (40,0), (40,20), (20,60), (0,80)
                  give z = 0, 1600, 2200, 2600, 2400, so z* = 2600 at (20,60).
INTERPRETATION: evaluating a finite list replaces a search over infinitely many points;
                the list is finite because each vertex is the intersection of two rows.
```

**Conversion example.** Introducing slacks `s1, s2, s3 >= 0` turns the three rows into `2x1 + x2 + s1 = 100`, `x1 + x2 + s2 = 80`, `x1 + s3 = 40`. At `(20, 60)` the slacks are `s1 = 0`, `s2 = 0`, `s3 = 20`: machine hours and labour are exhausted, while the product-1 volume cap has 20 units of headroom. That sentence is the whole diagnostic value of the model, and it is available only because slacks were reported.

**Validity condition for 4.1:** the vertex-enumeration argument requires linearity of both objective and constraints. Add one `x1 * x2` term, or one "either-or" condition, and the polygon becomes a non-convex feasible set on which the best vertex need not be the best point.

### 4.2 General case: duality and complementary slackness

**Validity before use.** Pure linear program in the standard form of Section 3.3, with a non-empty feasible set in both primal and dual. Under these conditions weak duality holds for every pair of feasible points, and strong duality holds at the optimum.

**Derivation of weak duality in three lines.** Let `x` be primal feasible and `y >= 0` dual feasible. Since `Ax <= b` and `y >= 0`, we get `y'Ax <= y'b`. Since `A'y >= c` and `x >= 0`, we get `c'x <= x'A'y = y'Ax`. Concatenating: `c'x <= y'Ax <= y'b`. The middle term is the same scalar, so any primal value is at most any dual value. No assumption beyond feasibility and sign was used, which is why the bound is unconditional and why it survives rounding errors.

```text
FORMULA [FOR-02]: max c'x = min b'y  subject to  A'y >= c, y >= 0
WHERE: y in R^m, unit = objective unit per unit of the corresponding b_i;
       A'y is the vector of resource content of each activity at prices y.
CONSTRAINT: both problems feasible; primal bounded. If the primal is unbounded,
            the dual is infeasible, and no price vector exists.
NUMERICAL EXAMPLE: primal optimum 2600 at (20,60). Dual candidate y = (10, 20, 0):
                  A'y gives activity 1: 2(10) + 1(20) + 1(0) = 40 >= 40 (tight)
                  A'y gives activity 2: 1(10) + 1(20) + 0(0) = 30 >= 30 (tight)
                  b'y = 100(10) + 80(20) + 40(0) = 1000 + 1600 = 2600 = primal value.
INTERPRETATION: the dual point is a certificate. Anyone can verify 2600 is optimal by
                checking two products and one sum, without re-running the solver.
```

**Complementary slackness.** At an optimal pair `(x, y)`:

- `y_i (b_i - a_i'x) = 0` for every row: a constraint with slack has zero price;
- `x_j (c_j - a_j'y) = 0` for every column: a variable at positive level has zero reduced cost.

In the example, rows 1 and 2 are tight and carry prices 10 and 20; row 3 has slack 20 and price 0; both variables are positive and both reduced costs are 0. The pattern is consistent, which is a cheap self-check on any reported solution pair.

```text
FORMULA [FOR-03]: y_i = dz*/db_i  over the range where the optimal basis is unchanged
WHERE: z* is the optimal objective value; b_i is one right-hand side, unit of resource i.
CONSTRAINT: the basis must stay optimal, so the formula holds on an interval, not globally;
            at a degenerate optimum the one-sided derivatives may differ.
NUMERICAL EXAMPLE: raising machine capacity from 100 to 101 moves the optimum to
                   x1 = 21, x2 = 59, z = 840 + 1770 = 2610, a change of exactly +10.
                  Raising labour from 80 to 81 gives x1 = 19, x2 = 62,
                  z = 760 + 1860 = 2620, a change of exactly +20.
INTERPRETATION: the price is a rate over an interval; outside the interval the rate changes,
                and at a kink it may not exist as a single two-sided number.
```

**Validity condition for 4.2:** strong duality and the price interpretation require a pure linear program with continuous variables. Add integrality and the equality of primal and dual objectives fails; the dual of the relaxation remains a bound, and `y_i` becomes the rate of change of a *bound*, not of the achievable plan.

### 4.3 Variants: integrality, fixed charge, and piecewise cost

**Variant A — mixed-integer bound.** Keep the linear constraints, restrict `x_j` to integers. The relaxation value `z_LP` is still an upper bound, and the certificate becomes a pair `(z_inc, z_LP)` with a gap. No dual price may be quoted as a plan-level marginal, because the achievable objective is a step function of `b_i`: one extra machine-hour either changes nothing or enables a whole additional unit of a discrete activity.

```text
FORMULA [FOR-04]: relgap = (z_ub - z_inc) / |z_inc|,  z_inc <= z_IP <= z_ub
WHERE: z_ub is the best bound from any relaxation or cut; z_inc is the best feasible plan;
       both in objective units, both from the same run and the same model version.
CONSTRAINT: z_inc must be feasible for the original model, not for the relaxation.
NUMERICAL EXAMPLE: a binary model with 2x + 2y <= 3 has LP optimum 1.5 at (0.75, 0.75)
                  and integer optimum 1 at (1,0). Relative gap at the root = 0.5 / 1 = 50%.
INTERPRETATION: the gap measures ignorance, not error; a 50% root gap is a statement that
                the relaxation is weak, and the next move is a stronger formulation.
```

**Variant B — fixed charge.** A setup cost `F_j` applies whenever `x_j > 0`. Introduce binary `z_j` and the pair `x_j <= M_j z_j`, `x_j >= 0`. The `M_j` must be a valid upper bound on `x_j` from the other constraints, not an arbitrary large number, because a large `M` flattens the LP bound and a small `M` deletes feasible plans.

```text
FORMULA [FOR-05]: max sum_j (p_j x_j - F_j z_j)  s.t.  x_j <= M_j z_j,  z_j in {0,1}
WHERE: M_j = min(capacity of j, demand cap of j) computed from the other rows;
       F_j in currency per period; p_j in currency per unit.
CONSTRAINT: M_j must be tight enough that the LP bound is informative, and large enough
            that no feasible plan is cut off; both properties are checkable by inspection.
NUMERICAL EXAMPLE: with demand x2 <= 15, opening both products yields
                  40(40) + 30(15) = 1600 + 450 = 2050 before fixed costs.
                  Opening only product 1 yields 1600 - 200 = 1400.
                  Break-even: 2050 - 200 - F2 = 1400 gives F2 = 450 currency units.
INTERPRETATION: a small order stops paying for its own setup above a computable threshold;
                the threshold, not the solver's choice, is the reusable management fact.
```

**Variant C — piecewise-linear cost.** Model a cost that falls into bands with segment variables `d_k >= 0` and a linking row `x = sum_k d_k`, `0 <= d_k <= width_k`, with the objective coefficient of each segment set to that band's unit cost. If the objective is convex (increasing marginal cost), the solver fills cheap segments first automatically. If the cost is concave (volume discount), it does not, and the model is nonconvex unless the segments are chained with binaries.

**Validity condition for 4.3:** each variant changes what may be claimed. Variant A licenses bounds and a gap. Variant B licenses a break-even threshold. Variant C licenses a cost figure only when the direction of the marginal cost matches the objective direction, unless binaries are added.

### 4.4 Limits and counterexamples

**Counterexample 1 — the vertex argument fails without convexity.** Let the feasible set be the union of two disjoint triangles, one containing the best local vertex and one containing a slightly worse vertex. Enumerating "vertices" and taking the best is still valid here only because the set is a union of convex pieces; a point on the segment joining the two triangles is infeasible, and a local improvement method started in the wrong triangle never leaves it. The general lesson: convexity is what converts local checks into global proofs.

**Counterexample 2 — the price is not unique at a degenerate optimum.** Take the Section 5 instance and raise machine capacity to 120 while holding labour at 80 and the product-1 cap at 40. The optimum moves to `(40, 40)`, value 2800, and *three* constraints are now tight in two dimensions. The dual solution set becomes a segment: `y1` anywhere in `[0, 10]` with `y2 = 30 - y1` and `y3 = 10 - y1`, every point giving `120 y1 + 80 y2 + 40 y3 = 2800`. Quoting "the shadow price of machine capacity is 5" is then an arbitrary choice from a continuum; the honest report is "between 0 and 10, depending on which of the three tight limits you propose to relax first".

**Counterexample 3 — a relaxation gap is not an error.** The binary instance `2x + 2y <= 3` has LP value 1.5 and integer value 1. No solver has failed; the model class simply cannot achieve the fractional point. Reporting "the solver is 50% off" misdiagnoses a formulation weakness as a numerical fault and sends the team to the wrong fix.

**Counterexample 4 — an unbounded model is a formulation bug, not a big answer.** Maximise `40x1 + 30x2` with only `x1 + x2 >= 1` and `x >= 0`. The objective can be driven arbitrarily high by increasing both variables, and the solver will report unboundedness. The missing constraint is almost always an upper limit that everyone knew and nobody wrote: a demand cap, a machine hour budget, a storage limit.

**Validity condition for 4.4:** every result in 4.1 to 4.3 is conditional on linearity, feasibility, and (for prices) non-degeneracy. When one of those fails, the correct output is a bound plus a named caveat, not a point estimate.

**Takeaway:** Duality is a certificate and a price, but only for a linear program, only on an interval, and only when the optimum is not degenerate; each of those three restrictions has a concrete counterexample above.

## 5. Worked Numerical Example

**Instance (ILLUSTRATIVE).** A workshop makes two products. Product 1 earns 40 currency units per unit, product 2 earns 30. Machine time is 2 hours per unit of product 1 and 1 hour per unit of product 2, with 100 machine-hours available. Labour is 1 hour per unit of either product, with 80 hours available. Product 1 cannot exceed 40 units because of a packaging limit. All quantities are per week.

```text
maximise    40 x1 + 30 x2
subject to  2 x1 +  x2 <= 100     (machine hours, per week)
             x1 +  x2 <=  80      (labour hours, per week)
             x1       <=  40      (packaging cap, units per week)
            x1, x2 >= 0            (units per week)
```

### Step 1 — enumerate vertices

| Vertex | How obtained | Machine use | Labour use | Objective |
|---|---|---|---|---|
| (0, 0) | origin | 0 | 0 | 0 |
| (40, 0) | packaging cap with x2 = 0 | 80 | 40 | 1600 |
| (40, 20) | packaging cap ∩ machine row | 100 | 60 | 2200 |
| (20, 60) | machine row ∩ labour row | 100 | 80 | 2600 |
| (0, 80) | labour row with x1 = 0 | 80 | 80 | 2400 |

Check the infeasible candidate: `(40, 40)` satisfies labour and packaging but needs `2(40) + 40 = 120` machine-hours, which exceeds 100, so it is not a vertex. The maximum over the five vertices is 2600 at `(20, 60)`.

### Step 2 — read the slacks and name the binding limits

At `(20, 60)`: machine slack `100 - 100 = 0`; labour slack `80 - 80 = 0`; packaging slack `40 - 20 = 20`. Machine hours and labour are exhausted; packaging has headroom. The plan is therefore limited by hours, and the first question to the plant is which hour pool is cheaper to expand, which is exactly what the dual answers.

### Step 3 — obtain the dual and verify by complementary slackness

```text
minimise   100 y1 + 80 y2 + 40 y3
subject to   2 y1 +  y2 + y3 >= 40     (value of the resources product 1 consumes)
             1 y1 +  y2      >= 30     (value of the resources product 2 consumes)
             y1, y2, y3 >= 0
```

Two tight primal rows and two positive variables give two tight dual rows, so solve `2y1 + y2 + y3 = 40` and `y1 + y2 = 30`. Packaging has slack, so `y3 = 0`. Then `y1 + y2 = 30` and `2y1 + y2 = 40`, giving `y1 = 10`, `y2 = 20`. Dual objective: `100(10) + 80(20) + 40(0) = 1000 + 1600 = 2600`, equal to the primal value. Optimality is certified.

### Step 4 — independent arithmetic re-check

Three separate checks, each using different numbers than the derivation above.

1. **Resource-content check.** At prices `(10, 20, 0)`, product 1 consumes `2(10) + 1(20) = 40` of value, exactly its profit of 40; product 2 consumes `1(10) + 1(20) = 30`, exactly its profit of 30. Neither product earns excess profit at these prices, which is the definition of a competitive equilibrium and is independent of how the prices were found.
2. **Perturbation check.** Solve the variant with 101 machine-hours: `2x1 + x2 <= 101`, `x1 + x2 <= 80`, `x1 <= 40` gives `x1 = 21`, `x2 = 59`, objective `840 + 1770 = 2610`. Difference from 2600 is 10, matching `y1`. With 81 labour hours: `x1 = 19`, `x2 = 62`, objective `760 + 1860 = 2620`, difference 20, matching `y2`.
3. **Dimensional check.** `y1` is currency per machine-hour; multiplying by 100 machine-hours gives 1000 currency, which adds correctly to 1600 currency from labour. Adding `y1` to `y2` would be dimensionally invalid, and the dual constraints avoid that by weighting each price by the resource content of an activity.

### Step 5 — sensitivity ranges

**Right-hand-side range for machine hours.** The basis `{machine row, labour row}` stays optimal while `x1 = b1 - 80` stays within `[0, 40]`, that is `b1` in `[80, 120]`. At `b1 = 80` the optimum is `(0, 80)` with value 2400; at `b1 = 120` it is `(40, 40)` with value 2800. Both endpoints reproduce `2600 + 10(b1 - 100)`, so the price 10 is constant across the interval.

**Right-hand-side range for labour.** The same basis keeps `x1 = b2 - 60` in `[0, 40]`, that is `b2` in `[60, 100]`. At `b2 = 60`, `(40, 20)` gives 2200; at `b2 = 100`, `(0, 100)` gives 3000. Both match `2600 + 20(b2 - 80)`.

**Objective-coefficient range for product 1.** The vertex `(20, 60)` stays optimal while the iso-profit slope lies between the slopes of its two incident edges, that is while `c1 / 30` lies in `[1, 2]`, so `c1` in `[30, 60]`. At `c1 = 60`, `(20,60)` and `(40,20)` tie at 3000; at `c1 = 30`, `(20,60)` and `(0,80)` tie at 2400. Outside that window the optimal plan changes product mix rather than volume.

### Step 6 — sensitivity note and what the numbers do not say

- The price of a machine-hour is 10 currency units **only** while capacity stays within `[80, 120]`. Beyond 120 the packaging cap becomes the limit and the marginal value of a machine-hour drops to 0, because `(40, 40)` already leaves machine time idle relative to a further increase.
- The price of a labour-hour is 20 **only** within `[60, 100]`. Below 60 the machine row is the sole limit; above 100 the labour constraint stops binding in the region where `x1 <= 40`.
- The exact optimum is a property of the coefficients, which are ILLUSTRATIVE. A 5% error in the profit of product 2 moves `c1/c2` from 1.333 to 1.404, still inside `[1, 2]`, so the plan survives; a 30% error moves it to 1.905, still inside. Only errors large enough to cross 2.0 change the product mix, which tells you which coefficient to audit first.
- Nothing here says the workshop can sell 20 and 60 units. The demand side is absent from the model, and that omission is the first thing to check before acting.

The value 2600 is the least interesting number in this section; the three ranges and the zero price on packaging are what a manager can actually use. The second instance exists to show that the same machinery answers a ratio-shaped question.

### Step 7 — a second instance with a different constraint shape (ILLUSTRATIVE)

The first instance is a resource-allocation model: two scarce pools, one cap. The second is a **blend or specification** model, where the binding condition is a ratio rather than a pool. It exists because the same duality machinery answers a completely different management question, and because a ratio constraint is the most common place where a modeller writes a non-linear condition and calls it linear.

A feed mill must produce at least 100 tonnes per week of a blend. Ingredient P costs 5 currency units per tonne, ingredient S costs 3. The customer specification requires at least 40% of the blend to be P. P supply is capped at 60 tonnes per week.

```text
minimise    5 p + 3 s
subject to   p + s >= 100                 (demand, tonnes per week)
             0.6 p - 0.4 s >= 0           (specification: p >= 0.4(p + s), rearranged)
             p <= 60                      (supply cap, tonnes per week)
             p, s >= 0
```

The specification row deserves attention. The constraint "at least 40% P" is `p >= 0.4(p + s)`, which is linear once rearranged as `0.6p - 0.4s >= 0`. Writing `p >= 0.4D` instead — using the demand quantity rather than the produced quantity — is a different and usually wrong model, because it fixes the P quantity independently of how much is actually blended.

**Solution by vertex enumeration.** Two variables, three rows, so the candidate corners are the intersections of row pairs projected onto the feasible region.

| Candidate | p | s | Demand met | Specification met | Cap met | Cost |
|---|---|---|---|---|---|---|
| demand ∩ specification | 40 | 60 | yes (100) | yes (40%) | yes | 380 |
| cap ∩ specification | 60 | 90 | yes (150) | yes (40%) | yes | 570 |
| cap ∩ demand | 60 | 40 | yes (100) | yes (60%) | yes | 420 |
| demand only, s only | 0 | 100 | yes | no (0%) | yes | 300, infeasible |
| cap only | 60 | 0 | no | yes | yes | 300, infeasible |

Minimum feasible cost is 380 at `(p, s) = (40, 60)`. The binding rows are demand and specification; the supply cap has slack `60 - 40 = 20` tonnes.

**Closed form and its check.** With demand and specification both tight, `s = 0.6p/0.4 = 1.5p` and `p + 1.5p = D`, so `p = 0.4D` and `s = 0.6D`. Cost is `5(0.4D) + 3(0.6D) = 2D + 1.8D = 3.8D`. At `D = 100` that gives 380, matching the enumeration. As a fraction `f = 0.4`, the general form is `C(D, f) = D(3 + 2f)`: at `f = 0.4`, `3 + 0.8 = 3.8`; at `f = 0.45`, `3.9` and cost 390, which the enumeration confirms at `(45, 55)`.

**Dual and its interpretation.** The dual of this minimisation has non-negative variables `y1` (demand), `y2` (specification) and `y3` (cap), with one constraint per primal column:

```text
maximise    100 y1 - 60 y3
subject to   y1 + 0.6 y2 - y3 <= 5      (per tonne of P)
             y1 - 0.4 y2      <= 3      (per tonne of S)
             y1, y2, y3 >= 0
```

Both primal variables are positive, so both dual rows are tight. The cap is slack, so `y3 = 0`. Solving `y1 - 0.4y2 = 3` and `y1 + 0.6y2 = 5` gives `y2 = 2` and `y1 = 3.8`. Dual objective `100(3.8) - 60(0) = 380`, equal to the primal cost. The interpretation is the useful part: `y1 = 3.8` is the marginal cost of one more tonne of demand while the specification and the cap are unchanged, and `y2 = 2` is the marginal cost of tightening the specification by one percentage point expressed as a fraction — the specification is quoted in percentage points, so one point is `0.01` in `f`, worth `200 x 0.01 = 2` currency per week.

**Validity boundaries for the blend instance.** Three boundaries, each with a number.

- The demand price 3.8 holds while the cap stays slack, that is while `0.4D <= 60`, so `D <= 150`. At `D = 150` the plan is `(60, 90)`, cost 570, and all three rows are tight: the price is then a point on a degenerate face, and any non-negative combination of the three row prices summing correctly is admissible.
- Above `D = 150` the model is infeasible. The maximum feasible demand is `60 / 0.4 = 150` tonnes, so a demand of 151 is not a costing question but a capacity statement — the supply cap and the specification cannot both hold.
- The specification price 2 per percentage point holds while the specification is binding and the cap stays slack. If the specification loosens below the point where `s` alone can satisfy demand — that is below `f = 0` here, unreachable — the row stops binding. In a model where `S` also has a cap, the binding switch happens earlier, and the correct move is to re-solve rather than extrapolate.

The two instances together cover the two shapes a foundation sheet needs: a maximisation with pooled resources and a minimisation with a ratio specification. In both, the dual is a price per unit of a right-hand side in the same units as the objective, and in both the price is only reported with an interval.

## 6. Visual / Geometric Intuition

### 6.1 The polygon and the direction of improvement

Plot `x1` on the horizontal axis in units per week and `x2` on the vertical axis. The machine row `2x1 + x2 = 100` cuts the axes at `(50, 0)` and `(0, 100)`. The labour row cuts at `(80, 0)` and `(0, 80)`. The packaging cap is the vertical line `x1 = 40`. The feasible region is the pentagon with vertices `(0,0)`, `(40,0)`, `(40,20)`, `(20,60)`, `(0,80)`.

The objective gradient is `(40, 30)`, pointing up and to the right at a slope of `30/40 = 0.75`. Iso-profit lines are perpendicular to it. Pushing a line of slope `-0.75` in that direction, the last contact is the corner where the machine and labour rows cross, because that corner is the furthest point of the polygon in the direction `(40, 30)`.

### 6.2 Why the corner that wins is the "balanced" one

Each corner can be described by what it exhausts. `(40, 20)` exhausts machine time and packaging, leaving labour half idle relative to the mix. `(0, 80)` exhausts labour with machine time 20% idle. `(20, 60)` exhausts both hour pools and leaves only packaging slack. In a two-resource model, the optimum typically sits where the two scarce resources are simultaneously exhausted, because otherwise one of them is being wasted and some rebalancing improves the objective. That geometric fact is the intuition behind the dual: the two exhausted resources carry positive prices, and the abundant one carries zero.

### 6.3 What the dual looks like geometrically

In the dual plane, the two constraints `2y1 + y2 >= 40` and `y1 + y2 >= 30` bound a region in the first quadrant. The dual objective `100y1 + 80y2` has gradient `(100, 80)`, and minimisation pushes a level line toward the origin until it touches the feasible region at `(10, 20)`. The primal's binding rows become the dual's positive variables; the primal's positive variables become the dual's binding rows. This mirroring is worth seeing once by hand, because it explains why a price can only be non-zero for a resource that is exhausted.

### 6.4 Degeneracy as a flat face

Move machine capacity to 120 and the polygon changes: now `(40, 40)` is feasible, and it is where the machine row meets both the labour row and the packaging line. Three constraints pass through one point, which is one more than the two needed to pin a corner in the plane. Geometrically the dual feasible region has a whole edge that is optimal: the level line of the dual objective lies flat against that edge instead of touching a single vertex. Any point on the edge is a valid price vector, so a report that quotes one point is reporting a choice, not a fact.

### 6.5 The integer picture

For a binary model, feasibility is a set of isolated points rather than a polygon. The relaxation's polygon has corners in the gaps between them. The gap between the best corner and the best point is the integrality gap visualised: it is the width of the empty region the relaxation is allowed to use. Strengthening a formulation means adding rows that cut off those empty corners without cutting off any lattice point — the geometric definition of a valid inequality.

**Takeaway:** Every algebraic statement in Section 4 has a picture: a corner for optimality, an exhausted pair for positive prices, a flat face for degeneracy, and an empty corner for an integrality gap.

## 7. Comparison / Selection Table

### 7.1 Model classes against the claim they license

| Class | Variables | Guarantee available | Useful output | Typical formulation error |
|---|---|---|---|---|
| Linear program | continuous | exact optimum, dual certificate | plan, slacks, prices with ranges | quoting a price outside its range |
| Mixed-integer linear | some integer | bound plus gap, or proof at zero gap | plan, bound, gap, no prices | reading duals as plan marginals |
| Fixed-charge MIP | integer switches | bound plus gap | plan, break-even thresholds | `M` set to an arbitrary large value |
| Piecewise-linear | continuous per segment | exact if convex | plan, marginal band in force | concave cost modelled without binaries |
| Quadratic or convex nonlinear | continuous | local optimum; global only if convex | plan plus active set | assuming convexity without checking |
| Constraint model | finite domains | feasibility or optimum over domains | plan, propagation explanations | confusing a propagator's pruning with a proof |

### 7.2 Choosing the objective form

| Situation | Objective form | Cost of the choice | When to reject it |
|---|---|---|---|
| All terms convert to one currency | weighted sum | weights are a policy stance | a term is not monetisable without controversy |
| One concern dominates by policy | lexicographic | two or more solves | the priority order is not authorised |
| Trade-off must be shown | epsilon-constraint | one solve per point | the sweep is too slow for the decision cadence |
| A limit is contractual | hard constraint | none, if the limit is real | the limit is a target rather than a rule |
| A limit is negotiable | penalty term | penalty calibration is arbitrary | the penalty is large enough to hide infeasibility |

### 7.3 Diagnostic sequence when a formulation is challenged

| Challenge | First check | Second check | What settles it |
|---|---|---|---|
| "The plan is infeasible in reality" | replay slacks at the reported point | confirm every real limit appears as a row | an unmodelled limit, or a wrong unit |
| "The saving is overstated" | recompute objective from variables | check whether the baseline is the current plan | a baseline mismatch |
| "The marginal value is wrong" | check the constraint's slack | check the basis range for that right-hand side | a degenerate or out-of-range price |
| "The optimum is obviously wrong" | enumerate vertices or a small brute force | check constraint directions after sign flips | a flipped inequality |
| "The model takes hours" | count rows and columns, check `M` sizes | check whether the LP bound is near the integer optimum | a weak formulation, not a slow solver |

**Takeaway:** The selection tables are ordered by the claim being made, not by solver brand; every row names a formulation error that produces a plausible-looking wrong answer.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom, at least two competing causes, a check that separates them, and the correction. The check is always something observable in the model file or the solution report.

### F-01 — The optimal plan violates a rule everyone knows

**Symptom.** The solver returns a plan that ships below the minimum order quantity, or mixes products that cannot share a cleaning cycle.
**Competing causes.** (a) an unmodelled hard rule; (b) a rule modelled with the wrong direction after a sign flip; (c) a rule modelled as a large penalty that the objective outbids.
**Discriminating check.** Ask the solver for the plan's row-by-row slack report. If no row corresponds to the violated rule, cause (a). If a row exists and its slack is negative, cause (b). If a row exists as a penalty term in the objective, cause (c).
**Correction.** Add the rule as a row with an owner, or convert the penalty into a hard constraint where policy requires it.
**Prevention.** Maintain a rule register that maps each operational rule to one row identifier; review the register whenever the plan is rejected by operations.

### F-02 — Units drift between the model and the data feed

**Symptom.** Capacity looks ample on paper and the plan is physically impossible; or the model is infeasible although the plant clearly can produce something.
**Competing causes.** (a) hours versus minutes in `A` but not in `b`; (b) per-week demand entered against per-day capacity; (c) currency in thousands against currency in units.
**Discriminating check.** Compute the ratio `b_i / max_j a_ij` for each row and compare it to the plausible number of units of the most demanding activity the plant can run. A factor of 60, 24, 7, or 1000 is a unit error, not a capacity finding.
**Correction.** Declare units in the row name itself, for example `machine_hours_per_week`, and convert at ingestion.
**Prevention.** A single conversion layer with a units table; assert each row's unit string against an allow-list before solving.

### F-03 — A negotiable target is coded as a hard constraint

**Symptom.** The model is infeasible in a week when the business expected a strained but workable plan, and someone proposes deleting a constraint to "make it run".
**Competing causes.** (a) a genuine capacity shortfall; (b) a target that should have been soft; (c) a data error in one right-hand side.
**Discriminating check.** Solve the model with each single row removed in turn and record which removal restores feasibility. A single row that unlocks feasibility is either the real bottleneck or the mis-coded target; the owner of that limit distinguishes them.
**Correction.** Soften the target with a violation variable and a declared penalty, and report the violation quantity as an output.
**Prevention.** Classify every row as `physics`, `contract`, or `target` at formulation time, and require an owner signature for `contract` rows.

### F-04 — The dual is quoted as a marginal benefit of the decision

**Symptom.** A memo claims "one more machine-hour is worth 10", and the capital request built on it overstates the benefit.
**Competing causes.** (a) the price was read outside its basis range; (b) the optimum is degenerate and the price chosen was the most flattering point on the face; (c) integrality makes the objective a step function so no rate exists.
**Discriminating check.** Re-solve at `b_i + 1` and at `b_i + 5`. If the value changes by `y_i` and `5 y_i`, the price is locally valid. If it changes by less than `5 y_i`, the range ended. If the model has integers, compare the two solves: unequal steps prove the absence of a rate.
**Correction.** Report the price with its interval, and report the value at the interval endpoints so the reader sees the kink.
**Prevention.** Any price in a deliverable must be accompanied by the range and by the two endpoint values that bound it.

### F-05 — Degenerate optimum, single price reported

**Symptom.** Two analysts compute different shadow prices for the same capacity and both are right.
**Competing causes.** (a) degeneracy, a face of optimal dual solutions; (b) different bases reported by different solver versions; (c) different right-hand sides in the two models.
**Discriminating check.** Count binding rows. If more rows are tight than there are decision variables, degeneracy is present. Then vary `b_i` in both directions: if the two-sided derivatives differ, the price is not unique.
**Correction.** Report the price interval spanned by the optimal dual face, and state which other tight limit would have to be relaxed first for the higher price to apply.
**Prevention.** Flag any solution with more tight rows than variables in the report header, before anyone quotes a number.

### F-06 — Big-M chosen by habit

**Symptom.** The solver proves optimality quickly but the plan it proves is worse than an obvious manual plan; or a feasible manual plan is declared infeasible.
**Competing causes.** (a) `M` too small, cutting off feasible plans; (b) `M` too large, destroying the LP bound and producing numerical noise; (c) the linking row is missing so the binary does nothing.
**Discriminating check.** Fix the binaries to the manual plan's pattern and solve the remaining continuous model. If it is infeasible, `M` is too small. If the LP relaxation value is far above the integer optimum, `M` is too large. If the binary pattern has no effect, the link is missing.
**Correction.** Compute `M` from the other rows as `min(activity capacity, demand cap)` and record the derivation next to the row.
**Prevention.** A model-lint rule that rejects any `M` above ten times the tightest valid bound computed from the row set.

### F-07 — Objective and constraints disagree about direction

**Symptom.** The plan maximises revenue by producing the maximum of everything and ignoring a cost that should have limited it.
**Competing causes.** (a) a sign error in one objective coefficient; (b) a missing cost term; (c) a constraint written as `>=` after conversion that should be `<=`.
**Discriminating check.** Evaluate the objective at the all-zero plan and at the reported plan. If the zero plan is feasible and scores better than reported, a sign is wrong. Then test each coefficient by perturbing it by 10% and checking the plan moves in the expected direction.
**Correction.** Write the objective as a sum of named terms with explicit signs, and keep revenue and cost in separate term lists.
**Prevention.** Compare the reported plan against the naive plan (all capacity used, cheapest activity only) as a smoke test; the naive plan should never beat the model.

### F-08 — Overlapping conditions double-counted

**Symptom.** The model reports a resource as exhausted when the plant can run the two activities in parallel, or reports infeasibility because two constraints both consume the same hour pool.
**Competing causes.** (a) the same physical limit modelled in two rows with different units; (b) a shared-resource row applied per product family as well as total; (c) a genuine shared resource that the plant actually schedules separately.
**Discriminating check.** Sum the resource consumption implied by the reported plan and compare it to the right-hand side of each row. If two rows both reach equality while the physical total is below both limits, the rows overlap.
**Correction.** Keep one row per physical limit and express family-specific limits as additional rows only when they are additive, not duplicative.
**Prevention.** Tag rows with the resource identifier they consume; reject duplicate resource tags unless the model explicitly sums sub-limits.

### F-09 — Infeasibility reported as a data problem

**Symptom.** The solver returns infeasible, and the team spends a week re-checking data that is in fact correct.
**Competing causes.** (a) a hard rule that cannot hold simultaneously with another; (b) a target coded as hard (F-03); (c) a unit error (F-02); (d) a genuine shortfall.
**Discriminating check.** Use an elastic formulation: add a non-negative violation to every row with a unit penalty and minimise total weighted violation. The rows that take violation at the optimum are the conflict set, in priority order by their penalties.
**Correction.** Either relax the conflicting rule with the owner's authority, or report the shortfall explicitly with the quantity that cannot be met.
**Prevention.** Run the elastic version once at model build time, before the first operational use, so the conflict set is known in advance.

### F-10 — Piecewise cost entered with the wrong convexity

**Symptom.** A volume-discount model returns a plan that sits entirely inside the most expensive band, or jumps between bands arbitrarily.
**Competing causes.** (a) concave cost modelled with independent segment variables and no chaining binaries; (b) segment widths that overlap; (c) objective direction reversed relative to the segment ordering.
**Discriminating check.** Solve the model with the segments forced in order and compare. If the objective improves when segments are allowed out of order, the model is concave and needs binaries.
**Correction.** Add chaining binaries `d_k <= width_k z_k`, `z_{k+1} <= z_k`, or replace the cost with its convex envelope and report the approximation.
**Prevention.** State the direction of marginal cost in the row comment; if it decreases with volume, require the chained form.

### F-11 — Free variables forced non-negative

**Symptom.** A planning model that should allow a negative net position (for example, borrowing stock from a later period) returns a plan that is feasible but oddly expensive.
**Competing causes.** (a) a free variable declared `>= 0`; (b) a balance row written as a one-sided inequality; (c) a missing carry-over variable.
**Discriminating check.** Inspect every variable that can conceptually be negative. If its domain is `x >= 0` and its optimal value sits exactly at 0, the domain is binding and the split form `x_plus - x_minus` should be tested.
**Correction.** Split the variable into two non-negative parts, or allow the solver to carry a free variable explicitly and confirm the sign convention in reporting.
**Prevention.** A domain column in the variable register, reviewed against the physical meaning of each quantity.

### F-12 — Unboundedness treated as a solver failure

**Symptom.** The solver reports unbounded and the team switches solver.
**Competing causes.** (a) a genuinely missing upper limit; (b) a sign flip that turns a `<=` into an unbounded direction; (c) a free variable in the objective.
**Discriminating check.** Solve the model with every variable capped at a large finite value. If it becomes feasible and finite, the model was unbounded because of a missing cap; the reported plan will show which variable ran to the cap.
**Correction.** Add the real limit that everyone knew. Never deliver a plan produced under an artificial cap without saying the cap is artificial.
**Prevention.** A build-time check that every variable has an upper bound implied by some row; report any variable whose only limit is non-negativity.

### F-13 — Objective measured in the wrong accounting basis

**Symptom.** The model's preferred plan is rejected by finance as loss-making; the plan is defensible under contribution margin but the company reports gross margin.
**Competing causes.** (a) an accounting basis mismatch, contribution versus gross; (b) allocation of fixed overhead into the model's unit costs; (c) a currency or period conversion applied to one term only.
**Discriminating check.** Recompute the objective using only cash-varying costs per unit and compare to the reported plan's value at the same plan. If the ranking of two candidate plans flips between the two bases, the basis is the issue, not the plan.
**Correction.** Declare the basis in the objective row, strip allocated overhead from unit costs, and reconcile the resulting figure to the finance report as a separate note.
**Prevention.** A single margin definition reused across every model in the unit, with the definition's source document referenced.

### F-14 — A balance row written as an inequality

**Symptom.** Stock or cash accumulates without bound, or a pipeline empties in the plan although the real system conserves.
**Competing causes.** (a) an equality converted to `<=` during standard-form conversion and never restored; (b) a missing carry-over variable; (c) a demand row that only requires delivery of at least something rather than exactly.
**Discriminating check.** Sum the flow into every node over the horizon and compare with the sum of flow out plus the ending stock. A mismatch in the plan's implied totals that the model does not report is an inequality where an equality belonged.
**Correction.** Restore the balance as an equality, or write it as two inequalities and check that both are binding.
**Prevention.** Tag balance rows with a `balance` class and assert in the model-lint pass that both slack directions are zero at the reported solution.

### F-15 — Horizon truncation creates a spurious plan

**Symptom.** The plan front-loads every activity in the last period, or defers all maintenance outside the horizon.
**Competing causes.** (a) a genuine end-of-horizon effect caused by missing terminal conditions; (b) a demand row at the final period that only requires partial fulfilment; (c) a cost term for work in progress that is absent.
**Discriminating check.** Extend the horizon by one period with the same demand shape and re-solve. If the plan's early periods change materially, the original answer depended on the truncation.
**Correction.** Add a terminal stock target, a salvage value, or a terminal demand row, and report the chosen condition.
**Prevention.** State the horizon convention in the exclusions register, with the terminal condition named.

### F-16 — The model optimises a quantity nobody measures

**Symptom.** The plan is optimal on the stated objective and the operating team reports the month as a failure.
**Competing causes.** (a) the objective is a proxy, such as units produced rather than on-time delivery; (b) the objective omits a downstream cost the plant bears; (c) the metric used for assessment is computed on a different population or window.
**Discriminating check.** Compute the assessment metric on the model's plan and on the current plan. If the model's plan is worse on the assessment metric, the objective is a proxy of the wrong thing, and the gap between the two is the size of the misalignment.
**Correction.** Replace the proxy with the assessed quantity, or add the omitted cost with a sourced coefficient, or make the trade-off explicit as a weighted sum with the weights authorised by the metric's owner.
**Prevention.** Require the objective row to name the report it is intended to improve, and check that the plan improves that report.

### F-17 — Coefficient provenance decays between versions

**Symptom.** The same model, re-run after six months with re-sourced data, recommends a materially different plan and nobody can say whether the world changed or the data did.
**Competing causes.** (a) genuine parameter drift; (b) a source system change in unit or definition; (c) a silent edit to a coefficient during an unrelated update.
**Discriminating check.** Version the parameter table and diff it between runs. A coefficient that moved without a corresponding source-document version change is cause (b) or (c); one whose document version also moved is cause (a).
**Correction.** Attach a source and a vintage to every parameter, and treat a change as a versioned event with a date.
**Prevention.** A parameter register that the model reads from, with a checksum recorded in the run log; any run whose checksum differs from the approved one is flagged.

### F-18 — A solver tolerance is mistaken for a modelling fact

**Symptom.** Two runs disagree in the reported plan by a small amount, and a debate starts about whether the business changed.
**Competing causes.** (a) an alternative optimum reached by a different pivot path; (b) a tolerance setting that accepts a slightly infeasible point; (c) a genuine data change between runs. 
**Discriminating check.** Compare objective values: if they match to within tolerance while the variable vectors differ, (a). If a reported constraint is violated by an amount just under the tolerance, (b). If the parameter checksum differs, (c).
**Correction.** For (a), declare a secondary objective or a deterministic tie-break rule so the reported plan is stable. For (b), tighten the tolerance and re-check feasibility against the raw rows rather than the scaled ones.
**Prevention.** Replay the reported plan through an independent feasibility check with a strict tolerance before the plan is issued.

**Takeaway:** Each failure has a cheap observable check; the expensive mistake is not the failure but the attribution of the failure to the wrong layer.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity

| Object | Size measure | Growth | Practical consequence |
|---|---|---|---|
| Vertex enumeration | `m` rows, `n` columns | `C(m, n)` candidate bases | usable to about `n = 3` for teaching |
| Simplex method | basis size `n` | exponential worst case, fast in practice | never justify a model by its practical speed alone |
| Interior-point method | rows and columns | polynomial in problem size | iteration count nearly flat as size grows |
| Branch-and-bound | tree size | exponential worst case | bound quality decides runtime, not instance size |
| Fixed-charge model | binary count `k` | `2^k` leaves in the worst case | formulation strength is the only lever you control |

The combinatorics matter as a sanity check on "just enumerate the options". For five binary choices there are 32 subsets, which is fine by hand. For ten there are 1024. For twenty there are 1,048,576. For forty, `C(40, 10) = 847,660,528` subsets of size ten alone, which is why enumeration stops being a plan somewhere between twelve and twenty binaries.

### 9b. Back-of-envelope sizing

```text
DENSITY  rho = (m + 1) / (n + 1)
WHERE: m = number of rows including the objective, n = number of structural columns,
       both counted after presolve removes fixed variables.
CONSTRAINT: counts must come from the same model build; presolve changes them materially.
EXAMPLE: m = 401 rows, n = 600 columns gives rho = 402 / 601 = 0.669 non-zeros per column.
INTERPRETATION: rho near 1 means a sparse, well-scaled model; rho above 20 means a dense
                block that will dominate both memory and simplex iteration cost.
```

```text
BUILD COST  T_build ~ (#rows x #columns) x c_row
WHERE: c_row is the marginal cost of adding one matrix entry, in analyst minutes;
       T_build is in analyst hours for one model version.
CONSTRAINT: applies to hand-built spreadsheets; a generated model scales with generator cost.
EXAMPLE: 400 rows x 600 columns with c_row = 0.05 minutes gives
         240,000 x 0.05 = 12,000 minutes = 200 analyst hours, which is 5 working weeks.
INTERPRETATION: at this size a hand-edited matrix is not maintainable; the model must be
                generated from a parameter table, and that generator is the real deliverable.
```

```text
PRICE-SENSITIVITY COST  K = (n + 1) solves for a full basis-range report
WHERE: n is the number of rows whose right-hand sides matter to the decision.
CONSTRAINT: each solve must reuse the same model build, or the differences are not comparable.
EXAMPLE: 12 capacity limits, two directions each, gives 24 re-solves; at 8 s per solve that is
         192 s of compute, which is trivial next to the 6 hours of analysis it replaces.
INTERPRETATION: the ranges are cheap; the reason they are missing from reports is process,
                not cost.
```

### 9c. Where the money actually goes

- **Model build dominates** in small problems: 200 analyst hours against 0.2 seconds of solver time. Effort spent on solver tuning at this scale is misallocated.
- **Data reconciliation dominates** in medium problems: the same limit appears in three systems with three numbers, and choosing among them is a governance decision, not a modelling one.
- **Proof dominates** in large integer problems: the plan is found quickly and the last 1% of the gap takes most of the wall clock. The decision is usually to accept the gap and act, with the gap stated.
- **Maintenance dominates** in every long-lived model: coefficients that never get re-sourced silently decay, and the model keeps returning confident answers from stale numbers.

### 9d. Cost of the four formulation errors, quantified

The point of this subsection is that each failure mode in Section 8 has a price, and the prices are of very different sizes. That ordering is what justifies spending review effort on some gates and not others.

```text
ERROR COST 1 — unit error, factor 60 in one row's coefficients
WHERE: a time resource entered in minutes inside an hours-based model.
CONSTRAINT: the factor propagates through every activity that consumes that resource.
EXAMPLE: 100 machine-hours per week with a_ij entered as 120 minutes per unit instead of
         2 hours per unit. The row now allows 100/120 = 0.83 units, where truth is 50 units.
INTERPRETATION: the model is infeasible or absurdly pessimistic; the cost is not a wrong
                number but a wrong conclusion about whether the plant can produce at all.
```

```text
ERROR COST 2 — price extrapolated beyond its basis range
WHERE: b1 raised from 120 to 160 while quoting the price 10 measured at b1 = 100.
CONSTRAINT: the packaging cap binds at 120, so beyond that the marginal value is zero.
EXAMPLE: predicted value 2600 + 10(40) = 3000; actual value 40(40) + 30(40) = 2800.
         Overstatement 200 currency per week, which is 7.1% of the predicted value.
INTERPRETATION: linear extrapolation of a piecewise-linear value function overstates benefit
                exactly when a second limit binds, and the error grows with the distance.
```

```text
ERROR COST 3 — infeasible baseline, benefit computed against it
WHERE: the "current plan" is the plan the model would produce under a looser rule.
CONSTRAINT: the baseline must be executable under the same rules as the model's plan.
EXAMPLE: with 120 machine-hours allowed, the optimal value is 2800; the model at 100 hours
         gives 2600. Compared against the 120-hour baseline the claimed benefit is zero,
         while compared against a 100-hour baseline of 2400 it is 200.
INTERPRETATION: the same model produces a 0% or an 8.3% benefit depending only on which
                baseline is chosen, so the baseline is part of the claim, not context.
```

```text
ERROR COST 4 — sensitivity not run because it "costs too much"
WHERE: 12 capacity limits, two directions, 8 seconds per solve.
CONSTRAINT: the solve count is 12 x 2 x 2 = 48 when the base point is included twice.
EXAMPLE: 48 solves x 8 s = 384 s = 6.4 minutes of compute, against the 6 hours of analyst
         time needed to defend a single unsupported price in review.
INTERPRETATION: the omission is a process choice; its compute cost is under 2% of the
                analysis it replaces, so the barrier is habit rather than capacity.
```

The four costs are not comparable in kind. A unit error can invalidate the feasibility conclusion; an extrapolated price overstates a capital case by a modest percentage; a baseline error changes a benefit by an order of magnitude relative to the plausible range; and a missing sensitivity costs review time rather than being wrong. Effort should therefore follow the ordering: units first (structure), baseline second (claim), prices third (magnitude), sensitivity fourth (defensibility).

### 9e. Scenario-count arithmetic for the sweep

A price sweep is itself a small capacity-planning problem, and the arithmetic is worth doing before promising a deliverable.

| Sweep design | Solves | Minutes at 8 s per solve | When it is appropriate |
|---|---|---|---|
| One right-hand side, two directions, one step | 3 | 0.4 | a single contested number |
| Twelve limits, two directions, one step each | 24 | 3.2 | the default for a decision memo |
| Twelve limits, three steps each direction | 72 | 9.6 | when the kink locations are unknown |
| Forty limits, three values, two directions | 240 | 32.0 | a capacity-planning review |
| Two hundred coefficients, two values each | 800 | 106.7 | a coefficient audit, not a memo |

The last row is the useful caution: a full coefficient sweep is over an hour of compute and produces more numbers than anyone will read. The right design is to sweep the coefficients whose plausible error range crosses a basis boundary, which is determined from the Section 5 style ratio analysis, not from a complete grid.

**Takeaway:** Cost is dominated by model construction and data reconciliation until the instance is large, at which point proof cost takes over; neither of those is addressed by changing solver, and the cheap gates carry the largest error costs.

## 10. Key Theorems / Results Reference

| Result | Statement in one line | Precondition | What it licenses you to say |
|---|---|---|---|
| Weak duality | any primal feasible value is at most any dual feasible value | both feasible, `y >= 0` | a bound without solving to optimality |
| Strong duality | primal optimum equals dual optimum | linear, both feasible | a certificate from a matching pair |
| Complementary slackness | slack in a row or zero in a column, never both | optimal pair | which limits bind and which prices are zero |
| Dual price as derivative | `y_i = dz*/db_i` on the basis interval | non-degenerate basis | a marginal value with a range |
| Farkas lemma | exactly one of a feasible `x` and a separating certificate exists | linear system | a proof of infeasibility, not just a failure report |
| LP relaxation bound | for maximisation, `z_LP >= z_IP` | integer model, same rows | the upper number in a gap statement |
| Total unimodularity | if `A` is TU and `b` integral, LP vertices are integral | TU matrix | an integer answer from a linear solve, with proof |
| Integrality gap | worst-case `z_LP / z_IP` over a family | family definition fixed | why one instance cannot characterise a formulation |
| Convexity and local optimality | for convex objectives over convex sets, local is global | convexity | permission to trust a gradient-based solver |
| KKT conditions | stationarity, primal feasibility, dual feasibility, complementary slackness | differentiable, constraint qualification | the nonlinear analogue of the dual certificate |

Two cautions about this table. First, the theorem numbers in textbooks vary by edition and by whether the author states the primal in equality or inequality form; cite the statement, not the number, unless you have the volume in hand. Second, every row is a conditional: the "what it licenses" column is the operative content, and quoting the statement without its precondition is how a correct theorem becomes a wrong memo.

**Takeaway:** Each theorem licenses exactly one sentence; the discipline is to say that sentence and no more.

## 11. Interview / Exam Questions

### Level 1 — recall and definition

#### Q01 — Name the five objects a formulation must contain, and classify overtime hours using them.

**Answer.** Decision variables, parameters, an objective, constraints, and index sets. Overtime hours are a **variable** if the plan can request them inside the decision window: they have a cost coefficient, a contractual cap, and possibly a minimum-block constraint. They are a **parameter** only if the schedule is frozen before the plan is built, in which case they belong in `b_i` of the labour row. The classification matters because treating a variable as a parameter hides the most expensive lever in the model and makes the plan look cheaper than it is; treating a parameter as a variable produces a plan that requests overtime nobody ever approved. Missing the other objects has its own signature: no index set means the grain is ambiguous and a weekly model can be fed daily data; no objective means there is no notion of better; no constraint means the plan is unbounded.
**Reasoning.** Each object maps to a column or a caveat in the output report, so an absent object is an absent field — and the overtime case shows the classification is a statement about authority, not about arithmetic.
**Follow-up.** Which object is most often implicit in a spreadsheet?
**Red flag.** A candidate who lists variables and constraints and then classifies overtime by intuition rather than by the decision window.
**Scoring.** 3 = all five plus the overtime test and both misclassification directions; 2 = five named plus the test; 1 = variables and objective only.

#### Q02 — Why must every variable carry a unit, and what does a unit error look like numerically?

**Answer.** Because the objective is a sum, and summing currency per hour with currency per unit is meaningless; the solver will still return a number, which is the danger. Numerically a unit error shows up as a suspicious ratio between a right-hand side and the largest matrix coefficient in that row: 60, 24, 7, or 1000 are the giveaway factors. A capacity row whose `b_i / max_j a_ij` equals 60 when the plant runs at most a few hundred units per week is a minute-versus-hour error, not a discovery about capacity.
**Reasoning.** The ratio has a physical meaning — the maximum number of the most demanding activity the row allows — so an implausible value localises the error to one row.
**Follow-up.** How would you catch it automatically before solving?
**Red flag.** "The solver handles units" — it does not; it handles numbers.
**Scoring.** 3 = ratio test plus the physical meaning; 2 = ratio test only; 1 = "units are important".

#### Q03 — State complementary slackness and give the two concrete checks it enables.

**Answer.** At an optimal pair, `y_i` times the slack of row `i` is zero for every row, and `x_j` times the reduced cost of column `j` is zero for every column. The two checks are: (a) any row with positive slack must have zero price, so a price quoted for an idle resource is an error; (b) any variable at a positive level must have zero reduced cost, so a plan that produces a positive quantity with a positive reduced cost is not optimal. On the Section 5 instance, packaging has slack 20 and price 0, and both products are positive with reduced cost 0, so the pair is consistent.
**Reasoning.** The condition is a pairwise test that needs no re-solve, which makes it the cheapest available audit of a reported solution.
**Follow-up.** What does a violated complementary-slackness pair usually indicate?
**Red flag.** Reciting the equation without being able to name one check it performs.
**Scoring.** 3 = both checks plus the instance; 2 = both checks abstractly; 1 = the equation only.

### Level 2 — derivation and application

#### Q04 — Derive weak duality and explain why it needs no optimality assumption.

**Answer.** For primal feasible `x` and dual feasible `y >= 0`: from `Ax <= b` and `y >= 0`, `y'Ax <= y'b`; from `A'y >= c` and `x >= 0`, `c'x <= x'A'y`; and `x'A'y = y'Ax` is the same scalar. Therefore `c'x <= y'b`. Nothing assumed optimality — only feasibility and signs — so the inequality holds for every pair, which is exactly what makes a dual point a usable bound while the primal is still being solved.
**Reasoning.** The two steps use different sign conditions, and both are needed; dropping `x >= 0` or `y >= 0` breaks the chain.
**Follow-up.** What does the inequality become if the primal is unbounded?
**Red flag.** "Strong duality says they are equal", which is a different theorem with a feasibility precondition.
**Scoring.** 3 = both steps plus the role of the sign conditions; 2 = the chain without discussing signs; 1 = a statement with no derivation.

#### Q05 — Convert `x1 + x2 >= 80` and a free variable `x3` into the standard form of Section 3.3, and check the result.

**Answer.** Multiply the first row by `-1`: `-x1 - x2 <= -80`. Split the free variable as `x3 = x3p - x3m` with both parts non-negative, substituting into every row and into the objective. Check by substituting a feasible numeric point from the original model, for example `x1 = 30, x2 = 60, x3 = -5`, giving `x3p = 0, x3m = 5`; the converted row evaluates to `-30 - 60 = -90 <= -80`, true, and the objective's `x3` coefficient contributes `c3(0) - c3(5) = -5 c3`, which matches `c3(-5)`.
**Reasoning.** The numeric replay is what catches a sign error introduced in conversion; symbolic conversion alone is where most mistakes survive review.
**Follow-up.** Why is splitting preferred to trusting a solver's free-variable support?
**Red flag.** Converting the `>=` by moving terms without flipping the direction.
**Scoring.** 3 = conversion plus replay; 2 = conversion only; 1 = partial conversion.

#### Q06 — Using Section 5, compute the value of 10 extra labour hours and state the range over which your answer holds.

**Answer.** The labour price is 20 currency per hour, valid while `b2` lies in `[60, 100]`. Ten extra hours from the base of 80 gives `b2 = 90`, inside the range, so the value is `10 x 20 = 200`, taking the optimum from 2600 to 2800. Verify by re-solving: at `b2 = 90` the optimum is `x1 = 30, x2 = 50` with objective `1200 + 1500 = 2700`... which does not match. Recheck the intersection: the basis rows are `2x1 + x2 = 100` and `x1 + x2 = b2`, giving `x1 = 100 - b2 = 10` and `x2 = b2 - (100 - b2) = 2b2 - 100 = 80`. That gives `x1 = 10, x2 = 80`, objective `400 + 2400 = 2800`, consistent with 200. The first attempted solution used the wrong intersection and is discarded; the arithmetic check that exposed it was substituting the candidate into both rows.
**Reasoning.** The value must be confirmed by an independent solve, and the first candidate failing both rows is the diagnostic that the algebra was misapplied.
**Follow-up.** At what `b2` does the answer stop being linear?
**Red flag.** Quoting 200 without a range, or accepting a candidate point without substituting it into the rows.
**Scoring.** 3 = value, range, and a verified point; 2 = value and range; 1 = value only.

#### Q07 — A model has a root LP value of 50.8333 and an incumbent of 49. Compute the relative gap and say what it does and does not tell you.

**Answer.** `(50.8333 - 49) / 49 = 1.8333 / 49 = 0.0374`, so 3.74%. It tells you that no plan beats 50.8333 and that a plan worth 49 exists, so the true optimum lies in `[49, 50.8333]`. It does not tell you how long the proof will take, whether a better plan will be found, or whether the formulation is weak in general — that requires an integrality-gap analysis across instances. It also does not tell you the plan is good enough: whether 3.74% matters is a decision about the value of a marginal unit, not a numerical fact.
**Reasoning.** Three distinct claims are being separated: a bound, an incumbent, and a stopping decision; the gap supports only the first two.
**Follow-up.** If the gap formula used the upper bound in the denominator, what would change?
**Red flag.** "The solver is 3.74% wrong" — that misreads a two-sided interval as an error estimate.
**Scoring.** 3 = the arithmetic, the interval, and the separation of claims; 2 = arithmetic and interval; 1 = arithmetic only.

#### Q08 — Derive the break-even fixed cost at which product 2 disappears from the Section 5 plan when demand is capped at 15 units.

**Answer.** With `x2 <= 15`, both open gives `40(40) + 30(15) = 1600 + 450 = 2050` before fixed costs; product 1 alone gives `1600 - 200 = 1400`. Set `2050 - 200 - F2 = 1400` and solve: `F2 = 450`. Above 450 currency per period the small order stops justifying its setup; below it, it stays. The number is a threshold in the model's own units and is reusable in negotiation, unlike the solver's binary choice.
**Reasoning.** The threshold is what survives a change in the demand cap; the plan does not. Note the check: at `F2 = 450` both options tie at 1400, which is a useful sanity value.
**Follow-up.** What if the setup cost is shared between two products?
**Red flag.** Reporting "the solver chose not to make product 2" with no threshold.
**Scoring.** 3 = threshold with the tie check; 2 = threshold only; 1 = restating the solver output.

### Level 3 — judgement, diagnosis, design

#### Q09 — Operations rejects an optimal plan because it leaves a machine idle for two hours. Diagnose and respond.

**Answer.** First establish whether idleness is a defect or a consequence. Compute the plan's value and the best plan that eliminates the gap by adding a constraint `utilisation >= u`; the difference is the price of the packing requirement. On instances of this shape it is often a few percent, and the right move is to offer the manager a choice between two declared objectives with the cost of each quantified. If utilisation is a contractual requirement, it becomes a row and the model is re-solved; if it is a preference, it is a weight or a secondary objective. What is not acceptable is defending the solver: the objective was value, and the objection is about a different objective.
**Reasoning.** The disagreement is about which objective is authoritative, so the productive move is to price the difference, not to re-argue optimality.
**Follow-up.** The manager insists the plant looks inefficient to visitors. Now what?
**Red flag.** "The maths says this is optimal" — true about the model, unhelpful about the decision.
**Scoring.** 3 = quantify the trade-off and reframe; 2 = quantify only; 1 = defend the model.

#### Q10 — Two analysts report shadow prices of 5 and 10 for the same machine constraint. Who is right, and what do you do?

**Answer.** Possibly both. At a degenerate optimum the dual solution set has more than one point, so any point on the face is a valid price vector. The procedure is: count binding rows against the number of variables — three tight rows in two dimensions in the Section 5 variant signals degeneracy; then vary the right-hand side in both directions and record the one-sided derivatives; then report the price as an interval with the identity of the competing limits. The report should say "between 0 and 10 depending on which tight limit is relaxed first", which is more useful than either number because it tells the reader that relaxing one limit alone may not deliver the gain.
**Reasoning.** Degeneracy is a structural property of the solution, not a bug in either analyst's tool; the interval is the honest object.
**Follow-up.** How would you phrase this for a capital request?
**Red flag.** Picking the larger number because it supports the request.
**Scoring.** 3 = both-right explanation plus the procedure; 2 = degeneracy identified plus one check; 1 = "trust the solver log".

#### Q11 — Design a formulation review that would catch the failures in Section 8 before a plan is issued.

**Answer.** Four gates, each cheap. Gate 1, structure: every variable has a unit and a domain, every row has an owner and a class (`physics`, `contract`, `target`), every `M` has a written derivation. Gate 2, arithmetic: replay the plan, recompute the objective from the variables, print all slacks, and assert that the naive plan does not beat the reported one. Gate 3, duality: for every price in the deliverable, run the two endpoint solves and print the three values; flag any row with slack and a non-zero price. Gate 4, class: state whether the claim is an exact optimum, a bound plus gap, or a bound only, and verify that the wording matches. Each gate maps to specific Section 8 rows, so a gate failure localises the defect.
**Reasoning.** The gates are ordered by cost, and each one produces an artefact that a reviewer can check without re-running the model.
**Follow-up.** Which gate would you drop if the decision cadence were daily?
**Red flag.** "Have a senior person review it" — a control without a checkable artefact.
**Scoring.** 3 = four gates with artefacts and defect mapping; 2 = three gates; 1 = general review advice.

#### Q12 — The business wants one number: "how much is an extra hour of capacity worth". What do you deliver?

**Answer.** A number with three attachments: the range over which it holds, the value at both ends of that range, and the identity of the constraint that becomes binding beyond it. Concretely, "10 currency per machine-hour while capacity is between 80 and 120 hours; 0 beyond 120 because packaging caps output at 40 units; below 80 the machine row is the only limit and the plan is `(0, b1)` with value `30 b1`". Then add the class caveat: if any part of the plan is integer, the true relationship is a step function and the single number is a local slope at best, so the deliverable should be a short table of `b1` values against achievable values, not a coefficient.
**Reasoning.** The single number is only defensible as a local derivative; supplying the range and the kink converts it from a slogan into a decision input.
**Follow-up.** What if the capacity expansion comes in indivisible blocks?
**Red flag.** A single number with no qualifier, which is the exact input that produces an over-investment.
**Scoring.** 3 = number, range, endpoint values, and the integrality caveat; 2 = number and range; 1 = number.

#### Q13 — How do you decide whether to keep a constraint that no one can justify?

**Answer.** Run the model with and without it and price its cost: the difference in optimal value is what the rule costs per period. Then ask the owner whether the rule is `physics`, `contract`, or `target`. A `physics` rule stays regardless of cost, though the cost figure is still worth publishing. A `contract` rule stays while the contract runs, and the figure feeds the renewal negotiation. A `target` rule that costs a material amount is a candidate for softening into a penalty, with the violation quantity reported. Finally, check whether the rule is a proxy for something else — a cleaning rule often encodes a contamination risk that can be modelled more precisely, and replacing the proxy with the real mechanism usually recovers value.
**Reasoning.** The value of the difference is the only defensible way to argue about a rule; without it the discussion is about authority rather than about cost.
**Follow-up.** The owner refuses to classify the rule. What then?
**Red flag.** Deleting the constraint to make the model run, which is the F-03 failure.
**Scoring.** 3 = priced difference plus the three-way classification; 2 = priced difference only; 1 = "ask the business".

#### Q14 — A colleague says "the model is fine, it is just the data". How do you test that claim?

**Answer.** Separate the two claims empirically. Take the current data and perturb one coefficient at a time within its plausible range; if the plan changes materially, the model is sensitive to that coefficient and the data question is the binding one. Then take a synthetic instance with known structure and check that the model recovers the known optimum; if it does not, the model is wrong on that instance regardless of data. The pair of tests distinguishes "the model is a faithful representation of a decision that is genuinely sensitive" from "the model has a structural defect that any data will expose". Finally, check the failure modes with observable signatures — unit ratios, `M` sizes, tight-row counts — because those are structural and data-independent.
**Reasoning.** Both claims can be true at once, so the test must be able to separate them rather than arbitrate between two assertions.
**Follow-up.** The synthetic test passes and the real plan is still rejected. What does that imply?
**Red flag.** Re-running the same model on the same data and calling it a test.
**Scoring.** 3 = both tests plus the structural-signature check; 2 = one test well designed; 1 = an opinion.

### Calculation clinic: audit trails before a recommendation

The clinics below are ILLUSTRATIVE construction exercises, not solver results or operating observations.

They deliberately change one modelling assumption at a time so that an audit can distinguish arithmetic trouble from a different model.

#### Clinic A — dimension ledger catches a hidden time conversion

A packaging line consumes 18 minutes per carton, and the planning horizon supplies 72 machine-hours.

Let `x` be cartons per week, a non-negative integer if partial cartons cannot be shipped.

The coefficient must be converted to `0.30 machine-hours/carton` because `18 / 60 = 0.30`.

The capacity row is therefore `0.30 x <= 72`, whose two sides are both machine-hours per week.

Dividing by `0.30 machine-hours/carton` gives `x <= 240 cartons/week`.

The independent multiplication check is `240 cartons/week * 0.30 machine-hours/carton = 72 machine-hours/week`.

Writing `18x <= 72` without changing the right-hand side would falsely cap output at four cartons.

Writing `18x <= 4320` is equivalent only if both sides are explicitly minutes per week.

A unit ledger should record coefficient unit, right-hand-side unit, conversion source, and horizon.

The discriminating check is to divide available time by time per carton and compare the implied output with a physical shift plan.

A result near four cartons for three staffed shifts signals a minute-hour mismatch rather than scarce capacity.

The correction is not to rescale the solution after solve; convert the row before solve and rerun the feasibility replay.

If cartons must be whole, `x <= 240` remains exact after declaring `x` integer because the bound is integral.

If setup consumes 0.25 hours once per active week, that is a fixed term gated by a binary, not an extra 0.25 in every carton coefficient.

Validity condition: the 18-minute time is a stable per-carton requirement over the stated horizon and excludes setup deliberately.

**Audit conclusion:** dimensions can reject a plausible-looking feasible plan before any optimisation theorem is invoked.

#### Clinic B — lower bounds can create infeasibility without a solver defect

Consider two contractual minimums, `x1 >= 35 units/week` and `x2 >= 50 units/week`.

Each unit requires one labour-hour, while the available labour capacity is `x1 + x2 <= 80 hours/week`.

Adding the minimums gives `x1 + x2 >= 35 + 50 = 85 units/week`.

Because one unit consumes one hour, the minimum plan needs at least 85 labour-hours per week.

The capacity row allows at most 80 labour-hours per week.

The contradiction is `85 <= x1 + x2 <= 80`, which cannot be true.

No objective coefficient can repair that contradiction because feasibility is assessed before the objective is used.

The shortfall is `85 - 80 = 5 labour-hours/week` under the one-hour-per-unit assumption.

An IIS routine may return all three rows, but the hand certificate already explains their conflict.

Competing cause one is an outdated contractual minimum; compare the contract effective date to the model horizon.

Competing cause two is a labour capacity that excludes approved overtime; reconcile the calendar against payroll authorisation.

Competing cause three is a coefficient stated per batch rather than per unit; inspect the production routing grain.

Relaxing either minimum by five units, or adding five hours, restores feasibility only for this three-row core.

Adding an artificial slack with penalty can locate the shortage, but an unboundedly cheap penalty hides the decision that must be authorised.

The report should name the contradictory rows and their owners rather than saying merely that the solver found no solution.

Validity condition: all quantities use the same weekly horizon and each unit really consumes exactly one labour-hour in the simplified certificate.

**Audit conclusion:** an infeasibility explanation is a smaller proof, not a solver status translated into prose.

#### Clinic C — choose a soft target by comparing units and slope

A service target asks for at least 90 fulfilled orders, but demand may make that target costly.

Let `s` be unmet target orders measured in orders per day, with `s >= 0`.

The softened row is `fulfilled + s >= 90`, so a shortfall of three orders sets `s = 3 orders/day`.

Suppose the objective maximises contribution in currency per day and subtracts `p s`.

The penalty coefficient `p` must have units currency/order so that `p s` has currency per day.

With `p = 14 currency/order`, three missing orders cost `14 * 3 = 42 currency/day` in the objective.

That 42 is not a measured customer loss; it is an ILLUSTRATIVE revealed-preference weight chosen by a decision owner.

If an alternative plan earns 55 additional currency/day while increasing shortfall by three orders/day, its net change is `55 - 42 = 13 currency/day`.

The alternative is preferred under this chosen penalty because 13 is positive.

At `p = 20 currency/order`, the same comparison becomes `55 - 20 * 3 = -5 currency/day`.

The preference reverses between 14 and 20, so publishing a single plan without its penalty conceals a policy choice.

The break-even penalty is `55 / 3 = 18.333... currency/order`.

Multiplying `18.333... currency/order * 3 orders/day` returns 55 currency/day, independently checking the threshold.

A hard service requirement is inappropriate to soften when legislation or safety makes any violation unacceptable.

A lower penalty is appropriate only when the owner accepts the corresponding shortfall as an exchangeable service loss.

Validity condition: the penalty is linear over the three-order range and does not double-count an existing lost-margin term.

**Audit conclusion:** a slack variable makes a preference visible; it does not make the preference objective fact.

#### Clinic D — a nonlinear boundary changes the certificate, not the need for checks

Suppose energy use is `E(x) = 2x + 0.01x^2` kilowatt-hours per day for throughput `x` in units per day.

The quadratic term has units kilowatt-hours per unit-squared so that `0.01x^2` has kilowatt-hours per day.

At `x = 40 units/day`, energy is `2(40) + 0.01(40)^2 = 80 + 16 = 96 kilowatt-hours/day`.

The marginal energy use is the derivative `E'(x) = 2 + 0.02x` kilowatt-hours per unit.

At 40 units/day, `E'(40) = 2 + 0.8 = 2.8 kilowatt-hours/unit`.

A one-unit finite difference gives `E(41) - E(40) = [82 + 16.81] - 96 = 2.81 kilowatt-hours`.

The finite difference exceeds 2.8 because the convex quadratic slope increases over the increment.

Treating 2.8 as a constant LP coefficient is a local approximation, not an exact formulation over a wide range.

The second derivative is `0.02 kilowatt-hours per unit-squared`, which is positive and establishes convex energy use.

A convex constraint `E(x) <= 100` has a convex feasible interval in this one-variable example.

Maximising a linear contribution subject to that convex constraint may be handled by convex optimisation, with KKT conditions under a suitable constraint qualification.

Maximising a concave reward is different from maximising a convex reward; the curvature of both objective and feasible set matters.

A piecewise-linear outer approximation must state whether it overestimates or underestimates energy before it is used as a safety limit.

For a capacity limit, an underestimate of energy can create a physically infeasible dispatch even when the linear model reports feasibility.

The discriminating check is to replay the chosen `x` in the original nonlinear expression, not merely in its linearisation.

Validity condition: the energy curve is deterministic on the stated range and the derivative units match the throughput definition.

**Audit conclusion:** nonlinear structure changes the admissible solver claim, but never waives dimensional or replay checks.

#### Clinic E — LP relaxation is a bound, and rounding can violate a shared row

A delivery choice has binary variables `z1` and `z2`, each equal to one when a route is opened.

Route costs are 7 and 5 currency units, while their capacities are 6 and 5 parcels.

Demand is 8 parcels, giving the covering row `6z1 + 5z2 >= 8`.

The integer choices that cover demand are `(1,1)` with cost 12, because neither route alone reaches eight parcels.

The LP relaxation permits `z1 = 0.5` and `z2 = 1`, which provides `3 + 5 = 8 parcels`.

Its fractional cost is `7(0.5) + 5(1) = 3.5 + 5 = 8.5 currency units`.

For this minimisation model, 8.5 is a lower bound on the integer optimum 12, not an achievable opening plan.

The absolute bound gap is `12 - 8.5 = 3.5 currency units`.

The relative gap against the incumbent is `3.5 / 12 = 0.291666...`, or 29.17%.

Rounding `z1 = 0.5` down to zero leaves capacity five, violating demand by three parcels.

Rounding it up yields `(1,1)`, which is feasible but may be unnecessarily expensive on larger models.

A valid MIP solve searches or cuts until it proves a suitable incumbent-bound relation; it does not assume rounding preserves rows.

A stronger formulation might add cover inequalities, but their validity depends on the actual route capacity logic.

The discriminating check is to evaluate every original integer row after any heuristic rounding, including linked setup and capacity rows.

Validity condition: routes are indivisible and capacity is available only when the corresponding binary equals one.

**Audit conclusion:** relaxation arithmetic gives a decision bound; integer feasibility must be checked on the original discrete model.

#### Clinic F — a dual value changes only while its basis remains feasible

Use the Section 5 continuous model with machine capacity 100 and labour capacity 80.

Its stated machine shadow price is 10 currency per machine-hour over its local basis interval.

Reducing machine capacity from 100 to 95 predicts an objective change of `-5 * 10 = -50 currency/week`.

The predicted value is `2600 - 50 = 2550 currency/week` if the basis is unchanged.

The binding equations become `2x1 + x2 = 95` and `x1 + x2 = 80`.

Subtracting the second equation from the first gives `x1 = 15 units/week`.

Substitution gives `x2 = 65 units/week`.

The independently recomputed objective is `40(15) + 30(65) = 600 + 1950 = 2550 currency/week`.

Both products stay non-negative, so this particular five-hour movement is internally consistent.

Increasing machine capacity to 130 while labour stays 80 predicts 2900 only if machine remains scarce.

Yet labour alone limits total output, and the product-one packaging cap may alter the active basis before that point.

The test is to solve or derive the adjacent basis at the proposed right-hand side and compare the active-row set.

If the machine row has positive slack, its price must become zero by complementary slackness.

If two new rows become tight simultaneously, one scalar shadow price may be non-unique because of degeneracy.

The report therefore states the capacity perturbation, the holding assumptions, and the first expected kink.

Validity condition: coefficients and objective remain fixed, the variable domains are continuous, and the reported basis interval contains the perturbation.

**Audit conclusion:** a shadow price is a local derivative with a certificate, not an exchange rate that survives arbitrary expansion.

#### Clinic G — lexicographic objectives avoid an arbitrary scale conversion

A dispatcher first wants to minimise late orders and only then minimise travel distance.

Let `L` count late orders per shift and let `D` measure kilometres per shift.

The primary solve minimises `L` and obtains `L* = 2 late orders/shift` in an ILLUSTRATIVE instance.

The second solve adds `L = 2` and minimises `D` among plans that retain that best lateness count.

This is lexicographic optimisation: distance cannot buy an extra late order because the first-stage result is fixed.

A weighted objective such as `1000L + D` instead assumes 1000 kilometres is the value of one late order.

That conversion is rarely a physical fact; it is a policy decision that requires an owner.

If any feasible plan has distance differences above 1000 kilometres, the weight may fail to enforce the intended priority.

A safe dominating weight needs a proven upper bound on possible distance variation, not a merely large-looking number.

For example, if `D` lies between 0 and 620 kilometres per shift, weight 621 km per late order ensures one fewer late order dominates distance.

Check: reducing `L` by one changes the weighted objective by 621, while any adverse distance change is at most 620.

The unit of the weight is kilometres per late order, so `621L` can be added to `D` only after that conversion is named.

Lexicographic solving avoids that conversion but requires two solves and an exact primary-objective constraint.

The discriminating check is to find whether a proposed weighted solution has more late orders than the lexicographic solution.

Validity condition: the primary objective has a well-defined attainable optimum and the second solve preserves it exactly within chosen numerical tolerance.

**Audit conclusion:** priority is a modelling relation; a large coefficient is not automatically a proof of that relation.

#### Clinic H — scaling changes numerical conditioning, not the underlying plan

A planning model contains revenue coefficients near 1,200,000 currency units and penalty coefficients near 0.003 currency units.

The coefficient ratio is `1,200,000 / 0.003 = 400,000,000`.

That ratio signals a potential numerical scaling concern because solver tolerances operate on floating-point magnitudes.

It does not prove the model is wrong; a real business may combine large totals and small marginal penalties.

One remedy expresses currency in thousands, turning 1,200,000 into 1,200 and 0.003 into 0.000003 thousands.

That conversion alone still leaves a ratio of 400,000,000, so merely changing the currency label is not sufficient here.

A better formulation separates a tiny penalty into a policy constraint if it represents a threshold rather than a true trade-off.

Another remedy normalises an objective component by a declared reference scale, then records the induced weight explicitly.

Rows should also be scaled so typical nonzero coefficients and right-hand sides are comparable without changing their feasible set.

Multiplying a constraint `0.002x + 0.004y <= 0.8` by 500 gives `x + 2y <= 400`.

The two rows are algebraically equivalent because positive multiplication preserves the inequality direction.

Check with `x = 100, y = 100`: original left side is `0.2 + 0.4 = 0.6 <= 0.8`; scaled left side is `100 + 200 = 300 <= 400`.

Multiplying by a negative number would require flipping the inequality, so automated scaling must retain the multiplier sign.

A scale audit reports original units, numerical multiplier, transformed row, and a replay in original units.

Do not compare dual values from differently scaled rows without transforming them back to the original right-hand-side unit.

Validity condition: scaling is exact algebraic multiplication and solver tolerances are reviewed in both numeric and business units.

**Audit conclusion:** conditioning work is representation management; it must leave the original feasibility and interpretation recoverable.

#### Q15 — What would make you refuse to deliver a model's recommendation?

**Answer.** Four conditions. (a) A hard rule is missing or wrong and the owner will not confirm its class — the plan may be infeasible in the plant. (b) The claim exceeds the class: an integer model with a material gap presented as optimal, or duals presented as achievable marginals. (c) The objective is not the one the business is measured on and no one will authorise a change. (d) The plan's feasibility depends on a number that cannot be sourced, such as a capacity figure that exists in three systems with three values. In each case the deliverable is a bound plus the specific missing decision, which is useful, rather than a plan that will be acted on and fail.
**Reasoning.** The common thread is that the deliverable would be acted on as more certain than it is, which is the one failure mode a model author is fully responsible for.
**Follow-up.** Which of the four is most often overridden under schedule pressure?
**Red flag.** "I would deliver with caveats" without naming what the caveats would say.
**Scoring.** 3 = four conditions plus the substitute deliverable; 2 = three conditions; 1 = one condition.

**Takeaway:** The strong answers all name a number, its range, and the sentence the model does not license; that triad is the whole discipline of formulation.

## 12. Quick Reference Card

| Item | Value or rule | Why it matters |
|---|---|---|
| Standard form used here | `max c'x` s.t. `Ax <= b`, `x >= 0` | one convention prevents sign drift |
| Vertex rule | linear objective over a bounded polygon attains its max at a vertex | finite check replaces infinite search |
| Weak duality | `c'x <= y'b` for any feasible pair | a bound with no optimality assumption |
| Strong duality | optima coincide for a feasible linear pair | matching values certify optimality |
| Complementary slackness | row slack or row price, never both non-zero | cheapest audit of a reported solution |
| Price meaning | `y_i = dz*/db_i` on the basis interval only | a price without a range is a slogan |
| Degeneracy test | tight rows > decision variables | prices become an interval, not a number |
| Integrality rule | LP value is a bound; report `(z_inc, z_ub, gap)` | an integer model cannot quote duals as marginals |
| Big-M rule | `M` = tightest valid bound from the other rows | loose `M` destroys the bound, tight `M` cuts plans |
| Section 5 anchors | optimum `(20, 60)`, value 2600, `y = (10, 20, 0)` | a hand-checkable instance for every claim above |

