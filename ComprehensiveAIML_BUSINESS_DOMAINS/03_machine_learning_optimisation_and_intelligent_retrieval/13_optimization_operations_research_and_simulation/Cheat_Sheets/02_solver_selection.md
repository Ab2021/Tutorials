---
title: "Solver Selection"
topic: "Choosing, benchmarking and governing optimisation solvers"
collection: "Part 03 — Machine Learning, Optimisation and Intelligent Retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Solver Selection

All numeric examples in this sheet are constructed teaching values (ILLUSTRATIVE). Every
tolerance, gap, node count and running-time multiple below is derived from a stated model, not
measured on a named commercial solver run; the arithmetic is reproducible by hand and is checked
twice inside §5. Where a statement depends on a solver's internal algorithm it is marked as a
mechanism claim that a reader must confirm against that solver's documentation before relying on it
operationally.

## 1. File Header

**Purpose.** Give a practitioner a repeatable procedure for choosing an optimisation solver for a
stated model class, and for defending that choice with numbers that survive review.

**Prerequisites.** Ability to write a model in a standard form (variables, objective sense,
constraints), familiarity with LP duality, comfort reading a solver log line (gap, nodes, cuts),
and the discipline to record wall-clock time together with the machine and thread count.

**Outcome.** After this sheet the reader can (a) classify a model into a solver family, (b) build a
benchmark that is honest about cut-off time and incumbent quality, (c) compute the tolerance
budget implied by their parameterisation, and (d) state a fallback rule that fires automatically
when the primary solver misses its deadline.

**Exclusions.** This sheet does not cover modelling itself (see the formulation sheet), does not
compare commercial licences on price, does not cover constraint-programming propagators in depth,
and does not claim any vendor's relative performance. Solver performance is machine-, version- and
model-dependent; the ranking in §7 is a procedure for producing your own ranking, not a
substitute for it.

**Reading rule.** Every quantitative claim in this sheet carries its units. A gap without a
denominator, a running time without a time limit, and a node count without a branching rule are
all meaningless as comparison statistics. When you copy a number out of this sheet into a
procurement document, copy the units and the time limit with it.

**Position in the unit.** This sheet sits between the formulation sheet (which produces the model)
and the integer-programming and metaheuristic sheets (which consume the answer this sheet
produces). The decision it supports is narrow: *which engine do we run, with which settings, under
which deadline, and what happens when it fails.*

## 2. Concept Map / Mental Model

The mental model is a funnel with three gates. A model enters at the top described only by its
algebra. Gate one asks whether the feasible region is a polyhedron (continuous, linear) or a
lattice (integrality). Gate two asks whether the objective and constraints are convex — this
decides whether a local method's certificate is a global certificate. Gate three asks whether you
need a proof of optimality or merely a good feasible point by a deadline. Solvers fail selection
most often at gate three, not gate one: teams pick an exact solver for a problem whose business
value is entirely in the first hour.

```text
model algebra
   |
   +-- all continuous, linear ---------------> LP  (simplex / interior point)
   |        |
   |        +-- objective quadratic, convex --> convex QP (active set / IPM)
   |
   +-- integer or binary variables
            |
            +-- linear relaxations tight, structure known --> MILP branch-and-bound + cuts
            |        |
            |        +-- huge symmetric formulations --> column generation / decomposition
            |
            +-- nonlinear or nonconvex --> MINLP (global) or NLP local + multi-start
                     |
                     +-- no exploitable structure, black-box evaluation --> metaheuristic
```

Three consequences follow directly from the funnel and are worth memorising.

First, moving a model one gate down is usually cheap and buys a large speed-up: linearising a
product of a binary and a bounded continuous variable adds one variable and two constraints and
converts a nonconvex problem into a MILP. Second, the certificate you receive degrades as you move
down: an LP solver returns a proven optimum and a dual certificate, a MILP solver returns a proven
optimum for the *model as written* (not your business problem), a local NLP solver returns a point
satisfying first-order conditions only, and a metaheuristic returns a point with no certificate at
all. Third, the deadline is an input to the choice, not a post-hoc complaint: an exact solver given
thirty seconds behaves like a heuristic with an unpredictable incumbent quality, which is often
worse than a well-tuned heuristic given the same thirty seconds.

**The funnel is not a ladder.** Moving down a gate is not a one-way commitment. A common and
effective pattern is to run a heuristic gate-three method first to obtain a good incumbent, then
hand that incumbent to an exact solver as a warm start. The exact solver now spends its time
proving rather than searching, and the combined run usually beats either method alone under the
same deadline. The heuristic's answer is not a certificate, but it is a valid starting region, and
starting region quality is the one thing that reliably shortens the proof phase.

**Where each gate misleads.** Gate one misleads when a model looks continuous but encodes an
integrality decision through a big-M: the relaxation is then continuous but pathologically weak,
and the solver chosen at gate one will produce a fast answer that is far from any integer reality.
Gate two misleads because "convex" is a property of the *formulation*, not the business problem:
$x_1 x_2$ is nonconvex, but for binary $x_1$ it linearises exactly; the same economics can be
convex or nonconvex depending on how you write it. Gate three misleads most often, because teams
state the deadline in the calendar (we need the plan by Friday) rather than in compute (we have
600 core-seconds per instance), and the two imply very different configurations.

```text
solver landscape by certificate strength
  strongest  LP / convex QP      optimal value + dual prices + sensitivity ranges
             MILP                optimal value for the written model + bound history
             convex MINLP        value with a duality gap you can bound
             local NLP           KKT point only; global claim requires multi-start
  weakest    metaheuristic       feasible point, no statement about the rest of the space
```

**The two-question test before any benchmark.** Question one: *what certificate does the decision
need?* A capacity plan reviewed monthly needs a bound; a routing plan dispatched hourly needs a
feasible point and a deadline. Question two: *what is the smallest objective difference anyone will
act on?* That number, divided by the objective scale, is the relative tolerance; everything about
solver configuration follows from it. Teams that cannot answer question two cannot select a solver
rationally, and their benchmarks tend to optimise a metric nobody uses.

**The structural test that saves the most money, run first.** Before any benchmark, write the
constraint matrix to a file and check three properties in order: are all entries integers with
magnitudes small enough to avoid round-off; is the matrix an incidence matrix of a bipartite
structure (each column has exactly two non-zeros of opposite sign); and does adding the identity
columns for slacks leave every square subdeterminant in $\{-1,0,1\}$. The second check is the
cheap proxy for total unimodularity and catches the majority of genuine network models in
practice. When it passes, the integer requirement is decorative and an LP solve is provably
sufficient. When it fails by exactly one row — a cardinality limit, a budget cap — the model is a
MILP with a near-network relaxation, which is the easiest possible MILP class and will close at the
root or within a handful of nodes.

**Takeaway.** Choose the solver whose *certificate* matches the decision risk, then verify the
choice with a benchmark that honours the real deadline.

## 3. Foundational Definitions

**Model class.** The triple (variable domains, objective functional form, constraint functional
form). Two instances belong to the same class only if a transformation maps one to the other
without changing the feasible set's geometry. A binary variable relaxed to [0,1] is a different
class even though the syntax is identical.

**Instance.** A specific numerical realisation of a model class: the coefficient matrices, right
hand sides, and bounds for one planning period or one customer.

**Solver.** An algorithm implementation that maps an instance to either an optimal solution with a
certificate, a feasible solution with a bound, or an infeasibility certificate.

**Exact versus heuristic.** Exact means the solver terminates with a proof of optimality (or of
infeasibility) for the given tolerances. Heuristic means termination with a feasible point and no
guarantee. A solver can be exact in theory and behave heuristically in practice when a time limit
interrupts it; record both properties separately.

**MIP gap.** For a minimisation problem with incumbent objective value $z$ and the best remaining
bound $\underline{z}$, the relative gap is $g = (z - \underline{z})/|z|$. This is the number the
solver prints and the number your business tolerance must be expressed in.

**Absolute gap.** $z - \underline{z}$ in objective units (currency, minutes, kilograms). Business
tolerances are naturally absolute ("we will not chase savings below €200") but solvers stop on
relative gap, so a translation step is always needed.

**Tolerance stack.** The set of numerical tolerances that interact in one decision: integrality
tolerance, feasibility tolerance, optimality (reduced-cost) tolerance, and the gap target. They
are not independent; §4.4 shows a case where tightening one loosens the effective result.

**Warm start.** A previously computed basis, solution, or set of columns supplied to the solver so
that it begins from a known-good region rather than from scratch. It changes running time, never
the optimum.

**Deterministic parallel mode.** A solver configuration in which the result is bit-reproducible
regardless of thread scheduling, typically at the cost of some speed. Required whenever an audit
must reproduce a number months later.

**Certified bound.** A dual bound the solver can prove from relaxation and cuts. It is the only
legitimate basis for saying "no solution better than X exists".

**Presolve.** The solver's preprocessing pass: bound tightening, coefficient reduction, singleton
row removal, duplicate row detection. Presolve can make an infeasible model look feasible by
removing constraints, so always re-run a suspicious infeasibility with presolve disabled.

**Solver log fidelity.** The property that every reported number (gap, nodes, time) is defined on
the *presolved* model unless the solver says otherwise. Node counts and row counts read off a log
usually refer to the presolved model and will not match your input file's dimensions.

**Benchmark instance set.** A fixed list of instances with fixed random seeds, stored with a hash,
used to compare solver configurations. Changing the set invalidates every prior comparison.

## 4. Core Formulas with Full Derivation

Each result below uses the same anatomy: the formula, the symbols and units, the conditions under
which it is valid, a worked numerical substitution, and the reading a practitioner should take.
The section closes with the tolerance budget, which is the formula most often skipped and most
often the cause of a wrong solver choice.

### 4.1 Base case — relative MIP gap and the stopping rule

**Formula.** For a minimisation problem, with incumbent value $z$ (objective units) and dual bound
$\underline{z}$ (objective units, $\underline{z} \le z$):

$$g_{\text{rel}} = \frac{z - \underline{z}}{|z|}, \qquad g_{\text{abs}} = z - \underline{z}$$

**Where.** $z$ is the best integer-feasible objective found so far; $\underline{z}$ is the best
bound from the LP relaxation plus cuts; both are computed on the presolved model but expressed in
original objective units.

**Constraint.** The denominator must be non-zero and of consistent sign. For maximisation problems
the printed gap is $(z - \underline{z})/|z|$ with $\underline{z}$ the *upper* bound. If $z$ can be
zero or change sign, relative gap is undefined at $z=0$ and misleading for $z$ near zero — use the
absolute gap and state the objective offset.

**Numerical example.** A distribution instance minimises weekly transport cost. After 90 seconds:
$z = 48{,}600$ (currency units/week) and $\underline{z} = 47{,}200$. Then
$g_{\text{abs}} = 48{,}600 - 47{,}200 = 1{,}400$ currency units and
$g_{\text{rel}} = 1{,}400/48{,}600 = 0.02881$, i.e. 2.88 %. If the team's stopping rule is a
relative gap of 1 %, the run must continue; if it is "absolute gap below 2,000", the run may stop
immediately.

**Interpretation.** The gap is a promise about the *remaining* distance to the best possible
value, not the distance to the unknown true optimum of the business problem. A 2.88 % gap on a
model that omits a labour constraint can coexist with a 10 % error against reality. Gap protects
you against solver error only.

### 4.2 General case — expected work of branch-and-bound and why instance size misleads

**Formula.** For a binary program with $n$ binary variables solved by branch-and-bound where every
node is fathomed (pruned) with probability $q$ independently, the expected number of explored
nodes in the full binary tree of depth $n$ is the geometric series

$$N(n) = \sum_{k=0}^{n}\bigl(2(1-q)\bigr)^{k} =
\begin{cases}
\dfrac{1-\bigl(2(1-q)\bigr)^{n+1}}{1-2(1-q)}, & q \neq 0.5\\[4pt]
n+1, & q = 0.5
\end{cases}$$

**Where.** $n$ is the number of binary variables still undecided at the root after presolve; $q
\in (0,1]$ is the probability that the bound fathoms a node (a proxy for relaxation strength);
$2(1-q)$ is the expected number of surviving children per node; $N$ is a node count
(dimensionless).

**Constraint.** The independent-fathoming assumption is false on real instances, where bounds
tighten near the leaves, so $q$ is not constant with depth. The model also ignores node cost: it
counts nodes, not time, so it is only valid while the LP re-solve cost per node is roughly constant
across depths. It assumes a pure binary tree, so it does not apply to general-integer branching
where a variable can spawn many children.

**Numerical example.** Take $n = 40$ and $q = 0.6$, so $2(1-q) = 0.8$:

$$N(40) = \frac{1 - 0.8^{41}}{0.2}$$

Compute $0.8^{41}$: $\ln 0.8 = -0.22314$, times 41 is $-9.1489$, so $0.8^{41} = e^{-9.1489} =
1.07\times10^{-4}$. Then $N(40) = (1 - 0.000107)/0.2 = 4.9995 \approx 5$ nodes.

Now drop the pruning probability to $q = 0.4$, so $2(1-q) = 1.2$:

$$N(40) = \frac{1.2^{41} - 1}{0.2}$$

$\ln 1.2 = 0.18232$, times 41 is $7.4751$, so $1.2^{41} = e^{7.4751} = 1{,}763.7$. Then
$N(40) = (1763.7 - 1)/0.2 = 8{,}813.6$ nodes.

**Interpretation.** The critical value is $q = 0.5$, because each node has two children: above it
the tree's expected size is bounded by a geometric series, below it the series diverges and the
expected node count grows exponentially in $n$. Moving $q$ from 0.6 to 0.4 — a change no solver log
would make directly visible — multiplies expected nodes by $8{,}813.6/5.0 = 1{,}763$ on a
40-variable instance, and at $q = 0.3$ the count reaches $2.45\times10^{6}$. This is why variable
count is a poor predictor of difficulty: a 200-variable instance with $q = 0.7$ costs
$(1-0.6^{200})/0.4 \approx 2.5$ nodes, while a 40-variable instance with $q = 0.3$ costs millions.
The quantity to estimate before choosing a solver is $q$, which you can observe in one afternoon by
solving the LP relaxation, rounding heuristically, and comparing the two objective values: if the
rounded solution is within a few percent of the relaxation bound, $q$ is large and the solver
choice barely matters.

### 4.3 Variants — time-to-target and the correct way to compare two solvers

**Formula.** Given a set of instances $i = 1..m$, a time limit $T$ (seconds), and for each solver
$s$ the time $t_{s,i}$ at which it first reaches a target objective $f_i^{\text{target}}$, define
the censored time-to-target statistics and the shifted geometric mean of running times:

$$\text{SGM}(s) = \exp\!\left(\frac{1}{m}\sum_{i=1}^{m}\ln\big(t_{s,i} + \tau\big)\right) - \tau$$

**Where.** $t_{s,i}$ is either the observed hit time or the limit $T$ if the target was never hit
(censoring); $\tau$ is a shift, conventionally 1 second, that keeps $\ln$ finite when a solver
finishes in microseconds; $m$ is the instance count. SGM is in seconds.

**Constraint.** Shifting by $\tau = 1$ s makes the statistic insensitive to differences below one
second, which is deliberate: differences of milliseconds are noise across machines. Censoring at
$T$ biases in favour of fast-failing solvers — a solver that hits the target on 6 of 10 instances
at 2 s each and times out on 4 will show a better SGM than a solver that hits all 10 at 5 s each,
despite the second being more trustworthy. Always report the hit *count* alongside the SGM.

**Numerical example.** Two configurations on $m = 10$ instances with $T = 300$ s, $\tau = 1$.
Solver A hits the target at 4, 6, 8, 10, 12, 15, 20, 25, 30, 40 s — all ten. Solver B hits at 2,
2, 3, 3, 4, 5, 6, 8 s and times out on the last two.

For A: $\ln$ values are 1.6094, 1.9459, 2.1972, 2.3979, 2.5649, 2.7726, 3.0445, 3.2581, 3.4340,
3.7136. Sum $= 26.9382$, mean $= 2.69382$, $\exp = 14.788$, minus 1 gives SGM $= 13.79$ s.

For B: $\ln$ of (3, 3, 4, 4, 5, 6, 7, 9, 301, 301) $= 1.0986, 1.0986, 1.3863, 1.3863, 1.6094,
1.7918, 1.9459, 2.1972, 5.7071, 5.7071$. Sum $= 23.9284$, mean $= 2.39284$, $\exp = 10.9445$,
minus 1 gives SGM $= 9.94$ s.

Solver B wins on SGM (9.94 s versus 13.79 s) while hitting the target on 8/10 instances versus
10/10 for A.

**Interpretation.** Report the pair (SGM, hits). A configuration that wins SGM but loses hits is
usually the wrong production choice for a deadline-critical job, because its failure mode is
"no answer at deadline" rather than "slightly worse answer". If instead the target is loose and
every configuration hits it, SGM is the right primary statistic and the hit count is uninformative.

### 4.4 Limits and counterexamples — the tolerance budget and where it breaks

**Formula.** Let $\varepsilon_{\text{int}}$ be the integrality tolerance (variable units), let
$a_j$ be column $j$ of the constraint matrix, and let $\varepsilon_{\text{feas}}$ be the constraint
feasibility tolerance (resource units). A variable declared integer at value $x_j$ with fractional
part $\delta_j = x_j - \lfloor x_j \rfloor \le \varepsilon_{\text{int}}$ contaminates a constraint
row by at most

$$\text{leak} = \sum_{j \in J} |a_{ij}|\, \delta_j \le \varepsilon_{\text{int}} \sum_{j \in J} |a_{ij}|$$

**Where.** $J$ is the set of integer variables with non-zero coefficient in row $i$; $a_{ij}$ is
the coefficient (units of row $i$ per unit of variable $j$); leak is in row $i$'s units.

**Constraint.** Valid only for rows whose activity is near their bound. A row with slack far
larger than the leak is unaffected in practice. The formula is a worst case; in practice leaks
partially cancel. It also assumes the solver reports the violated value, which some do not.

**Numerical example (counterexample to "tolerance is harmless").** A vehicle-loading model has a
capacity row $\sum_j w_j x_j \le 12{,}000$ kg, with $w_j = 340$ kg for each of 60 identical items
and a default integrality tolerance of $10^{-5}$. Then
$\text{leak} \le 10^{-5} \times (60 \times 340) = 10^{-5} \times 20{,}400 = 0.204$ kg. Negligible.

Now change the units to tonnes: the same row reads $\sum_j 0.34 x_j \le 12$ and the leak becomes
$10^{-5} \times 20.4 = 2.04\times10^{-4}$ t, still negligible. But change the coefficients instead:
a financial row mixing currency in millions with a coefficient of $10^{6}$ gives
$\text{leak} \le 10^{-5} \times 10^{6} = 10$ currency units per variable. With 50 such variables
the row can be violated by 500 currency units while the solver reports "optimal, feasible".

**Where the formula breaks.** It assumes the only error is integrality rounding. It says nothing
about the feasibility tolerance itself: a solver declaring $\varepsilon_{\text{feas}} = 10^{-6}$ on
a row scaled to $10^{7}$ admits absolute violation of 10 units before it complains. Scaling rows
and columns so that all coefficients lie in $[10^{-2}, 10^{2}]$ keeps both tolerances meaningful;
this is the single highest-value preparation step before any solver comparison.

**Validity condition for the whole section.** All gap, node and tolerance statements assume the
model has been scaled, that presolve has not silently removed the binding rows, and that the
reported objective is in original (not presolved) units. If any of the three fails, the numbers
are internally consistent and externally wrong.

**Tolerance budget, stated as a rule.** Write down the largest acceptable absolute violation in
business units ($V$), the largest coefficient magnitude after scaling ($A$), and the number of
terms in the tightest row ($k$). Then the integrality tolerance you may request is
$\varepsilon_{\text{int}} \le V/(kA)$. Example: $V = 1$ kg, $k = 60$, $A = 340$ kg gives
$\varepsilon_{\text{int}} \le 1/20{,}400 = 4.9\times10^{-5}$, comfortably above the default
$10^{-5}$, so the default is acceptable. Set $V = 0.01$ kg on the same row and the bound falls to
$4.9\times10^{-7}$, below the default — you must now either rescale the row (divide by 340 so
$A = 1$ and the bound becomes $1.7\times10^{-4}$) or accept a violation you cannot detect.

## 5. Worked Numerical Example

The scenario is constructed (ILLUSTRATIVE) but internally consistent: a regional parcel carrier
must choose a solver for a weekly route-assignment MILP with 850 binary variables, 1,900
continuous variables and 2,400 rows. Three candidate configurations exist. The question is not
"which is fastest" but "which is safe under a 600-second deadline with an audit requirement".

### 5.1 Inputs and their provenance

| Item | Value | Units | Provenance |
|---|---|---|---|
| Decision deadline | 600 | s wall clock | operational SLA, stated |
| Acceptable cost error | 250 | currency units/week | business tolerance, stated |
| Incumbent typical value | 45,000 | currency units/week | ILLUSTRATIVE |
| Binaries after presolve | 610 | count | ILLUSTRATIVE |
| Rows after presolve | 1,780 | count | ILLUSTRATIVE |
| Integrality tolerance | 1e-5 | variable units | solver default |
| Max row coefficient | 340 | kg per unit | instance data |

The absolute-gap translation is the first computation: a 250 currency-unit tolerance on a 45,000
incumbent corresponds to a relative gap of $250/45{,}000 = 0.00556$, i.e. 0.56 %. If the solver's
default stopping gap is 1 %, the default will let the run terminate at up to 450 currency units of
error — 80 % above tolerance. The correct configuration sets the relative gap target to 0.5 % and
an absolute gap target of 225 currency units, and relies on whichever fires first. Setting only
one of the two is the most common configuration defect in production MILP.

### 5.2 Arithmetic re-check 1 — gap translation

```text
FORMULA: g_rel = delta / z,  delta_target = g_rel * z
WHERE:   delta in currency units/week; z in currency units/week
GIVEN:   z = 45000, tolerance = 250
STEP 1:  g = 250 / 45000 = 0.0055556
STEP 2:  round down for safety: 0.005
STEP 3:  check absolute: 0.005 * 45000 = 225 <= 250  OK
STEP 4:  default 0.01 would allow 0.01 * 45000 = 450 > 250  REJECT default
```

The re-check is independent: $0.005 \times 45{,}000 = 225$ and $225 + 250 = 475$, so a 0.005
target sits 10 % inside tolerance. Rounding down (never up) is required because the gap is an
upper bound on distance to the best possible value, not on distance to your unknown true optimum.

### 5.3 Arithmetic re-check 2 — tolerance budget for the capacity row

```text
FORMULA: eps_int <= V / (k * A)
GIVEN:   V = 1 kg, k = 60 terms, A = 340 kg/unit
STEP 1:  k * A = 60 * 340 = 20400 kg
STEP 2:  eps_int <= 1 / 20400 = 4.902e-5
STEP 3:  default 1e-5 <= 4.902e-5  ACCEPT default
STEP 4:  worst-case leak = 1e-5 * 20400 = 0.204 kg
```

Independent check: a 0.204 kg leak on a 12,000 kg vehicle capacity is $1.7\times10^{-5}$ of
capacity, far below any dispatch tolerance. The default is safe *for this row*. The check must be
repeated for every tight row; the row with the largest $kA$ product governs the configuration.

### 5.4 Arithmetic re-check 3 — which configuration to deploy

Three configurations were benchmarked on 10 historical weekly instances with a 600 s limit and
target = best-known objective plus 0.5 %. Results (ILLUSTRATIVE):

| Config | Hits (of 10) | SGM time (s) | Mean final gap (%) | Deterministic | Licence |
|---|---|---|---|---|---|
| A: default, 4 threads | 10 | 143 | 0.49 | no | included |
| B: aggressive cuts, 8 threads | 8 | 96 | 0.83 | no | included |
| C: single thread, deterministic | 10 | 233 | 0.50 | yes | included |

Compute the SGM check for A from its raw times 92, 104, 118, 131, 140, 149, 161, 172, 186, 227 s:
$\ln(t+1)$ values are 4.5218, 4.6444, 4.7707, 4.8752, 4.9416, 5.0039, 5.0814, 5.1475, 5.2257,
5.4250. Sum $= 49.6372$, mean $= 4.96372$, $\exp = 143.16$, minus 1 gives 142.16 s. The table
reports 143 s, so the two agree to within 1 s; the residual is rounding in the table, not
disagreement in the statistic. Note what the arithmetic does *not* say: the plain mean of the same
ten times is $(92+104+118+131+140+149+161+172+186+227)/10 = 1{,}480/10 = 148.0$ s, which is 6 s
above the SGM. The SGM is lower because the logarithm compresses the slow tail — the 227 s instance
contributes $\ln 228 = 5.43$ rather than a share weighted by its magnitude. Reporting the plain
mean therefore systematically penalises configurations that occasionally run long, which is exactly
the configuration a deadline-critical operation should not be over-penalised for. Report the SGM
and the hit count; state the plain mean only if a reader asks for total compute.

Decision: choose A for routine weeks. B's 96 s SGM is attractive but its 8/10 hit rate means two
weeks in ten would deliver an answer 0.83 % off, exceeding the 250-unit tolerance by
$0.83\% \times 45{,}000 = 374$ units. C is chosen for any run whose output enters a regulatory
filing, because determinism is worth $233 - 143 = 90$ s. The fallback rule: if A has not reached a
2 % gap by 420 s (70 % of deadline), the job is re-queued on C with a warm start; if C also misses,
dispatch the best incumbent and flag the week in the exception report.

### 5.5 Sensitivity

Two parameters dominate. First, the deadline. From A's raw series, 6 of 10 runs finish under 150 s
and 8 of 10 under 172 s, so tightening the deadline from 600 s to 150 s drops A's hit rate against
a 0.5 % target to roughly 6/10 — at which point B's 8/10 hit rate, despite its worse final gap,
becomes competitive and the recommendation flips. The crossover deadline is the third-largest
order statistic of A's times, 118 s, because below that A hits on at most 2 of 10.

Second, the cost tolerance. Loosening 250 to 500 units makes A's default 1 % gap acceptable
($0.01 \times 45{,}000 = 450 \le 500$) and removes the need for any custom gap configuration at
all. Before tuning solvers, check whether the tolerance is negotiable: a factor-of-two relaxation
in tolerance is usually cheaper than a factor-of-two speed-up, and unlike the speed-up it does not
have to be re-earned at every solver upgrade.

## 6. Visual / Geometric Intuition

Picture the branch-and-bound tree as a landscape of nested boxes. The root box is the LP
relaxation's feasible polytope; each branching decision slices it. The dual bound is the best
objective value on any box still open. The incumbent is the best vertex actually found inside a
box with integral coordinates. The gap is the vertical distance between the incumbent's height and
the highest remaining open box's floor — not the distance to the bottom of the landscape.

```text
objective
   ^
   |            +------------------+  <- dual bound (best open box)
   |            |                  |
   |   gap ---> |                  |
   |            +------------------+
   |                 *  <- incumbent (best integer point found)
   |
   +------------------------------------> time
```

Two geometric facts follow and both are counter-intuitive to newcomers.

**Fact one: the gap can widen.** After a branching decision, the solver may *improve* the
incumbent (lowering $z$) while the bound also moves down; if the bound moves faster, the relative
gap grows even though the answer got better. A log showing the gap rising from 1.2 % to 1.9 % while
the objective improves by 300 units is normal, not a regression. Judge progress by the incumbent's
trajectory and the bound separately.

**Fact two: symmetry creates identical boxes.** If a model has $k$ interchangeable vehicles, the
tree contains $k!$ subtrees that are isomorphic. The solver explores them as distinct because it
cannot see the symmetry. For $k = 8$ that is 40,320 equivalent leaves. The geometric fix is to cut
the polytope with symmetry-breaking constraints (order the vehicles' indices by their first
assigned job), which collapses the box count without changing the set of distinct solutions. On a
symmetric instance this is usually a larger effect than any solver parameter tuning.

**The relaxation-strength picture.** For a MILP, the LP relaxation's polytope contains the convex
hull of integer points. The further the relaxation's optimum sits from the hull, the more branching
is needed. Cutting planes shave the relaxation toward the hull; the *integrality gap at the root*
is the shaved-off distance expressed in objective units. Compute it once per instance family: solve
the LP relaxation exactly, solve the MILP to a tight gap, and take the difference. If the root gap
is under 1 % of the objective, parameter tuning is a waste of effort; if it is over 5 %, cuts and
formulation changes pay.

**The time-limit cliff.** Wall time versus quality is not linear: solvers typically find a
near-optimal incumbent in the first 10–20 % of the time and spend the rest proving it. Plot
objective against log time on one instance before setting a deadline; the knee of that curve is the
only defensible deadline, and it is usually far earlier than the default the team proposes.

**Reading the log as a shape, not a number.** A solver log is a time series with three traces: the
incumbent, the bound, and the node count. Their relative slopes tell you which lever to pull, and
the pattern repeats across models because it follows from the algorithm rather than the data.

```text
pattern 1  incumbent flat, bound falling   -> search is productive, let it run
pattern 2  incumbent flat, bound flat      -> stalled: cuts exhausted, branch differently
pattern 3  incumbent falling, bound flat   -> heuristics improving, proof is hopeless here
pattern 4  both falling, gap constant      -> steady progress, do not interrupt
pattern 5  bound rising                    -> numerical trouble or an invalid cut
```

Pattern 2 is the one that wastes the most compute in practice: the solver is exploring a subtree
whose bound will never improve because the formulation's relaxation is weak there. The
intervention is not more time but a reformulation — tighter big-M, a disaggregated formulation, or
a symmetry-breaking constraint. Pattern 5, a bound that rises, is the only unambiguous alarm: a
valid bound cannot increase, so either the solver is repairing numerics after a tolerance
violation or a user-supplied cut is invalid.

## 7. Comparison / Selection Table

The table below is a decision aid keyed on model properties, not on vendor names. Use it to
shortlist, then benchmark the shortlist on your own instances — the benchmark, not the table,
decides.

| Property | Primary choice | Why | Fails when | Verification signal |
|---|---|---|---|---|
| Continuous, linear, ≤10^6 rows | Simplex or interior point LP | proven optimum plus duals in seconds | coefficients span >10^9, numerical breakdown | dual infeasibility below tolerance; basis condition |
| Convex quadratic objective | Convex QP (IPM) | polynomial-time convergence, duality gap certificate | Hessian near-singular, degenerate active set | duality gap and KKT residual both below target |
| Binary/integer, tight relaxation | MILP branch-and-bound with cuts | certified global optimum | root gap >5 %, symmetry, big-M constants | root gap, nodes, cut effectiveness |
| Very many columns, few rows | Column generation / Dantzig-Wolfe | avoids enumerating implicit columns | pricing subproblem as hard as master | pricing gap per iteration; tailing-off |
| Two-stage with linking resources | Benders decomposition | separates integer master from continuous recourse | many similar scenarios, slow master | master bound movement per iteration |
| Nonconvex continuous, small n | Global MINLP (spatial branch-and-bound) | certified bounds on nonconvex terms | n>100, loose relaxations | relative gap over boxes; bound tightness |
| Nonconvex continuous, medium n, local answer adequate | Local NLP + multi-start | fast, cheap, finds good local optima | need for global certificate | spread of local optima across starts |
| Black-box, expensive evaluation | Metaheuristic (local search, GA) with budget | no derivative or algebra needed | need for proof; constraint handling poor | improvement per 1,000 evaluations; feasible fraction |
| Sequence-dependent scheduling, small n | Constraint programming / CP-SAT | propagation handles disjunctions natively | objective-heavy, weak propagation | propagator fixpoint counts, backtracks |
| Stochastic outcomes required | Discrete-event simulation + optimiser wrapper | represents variance explicitly | common random numbers not used | confidence interval width at fixed replications |

Reading rules for the table. Each "fails when" cell is the actual selection criterion: check those
properties *before* benchmarking, because a solver chosen against a property it cannot handle will
lose to a simpler method regardless of tuning. Each "verification signal" cell is what you look at
after the run to decide whether to trust it; a run with a good objective and a bad verification
signal is a coincidence, not a result.

Two selection errors deserve names. **Overspecification** is choosing an exact solver for a problem
whose tolerance exceeds the gap you can reach anyway: if the business will accept 2 % and the
solver reaches 2 % in a tenth of the time it needs for proof, the extra time buys nothing.
**Underspecification** is choosing a heuristic where a certificate is needed for audit or safety:
a local optimum reported as "the optimum" in a regulatory filing is a misstatement, even when the
two coincide numerically.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom an operator sees, two competing causes that produce that symptom, the
check that discriminates between them, and the correction. The discrimination column matters more
than the correction: most solver incidents are misdiagnosed as "we need a faster solver" when the
real cause is a formulation artefact.

**F1. Time limit hit with a large gap, on an instance that used to solve easily.**
Competing causes: (a) the instance grew — more vehicles or time buckets — so the same formulation
is now harder; (b) the data changed qualitatively — a capacity value dropped and the model became
nearly infeasible, which inflates node counts massively before the solver proves infeasibility.
Discriminating check: run the LP relaxation only and compare its objective with the previous
week's. If the relaxation is far worse (or infeasible), the cause is data, not size; if it is
similar, the cause is size. Correction: for (a) add symmetry breaking and tighten big-M; for (b)
push validation upstream, because no solver tuning fixes an infeasible instance.

**F2. Solver reports "optimal" but the plan violates a physical constraint.**
Competing causes: (a) a tolerance leak — integer variables at $10^{-5}$ fractional parts scale up
through large coefficients; (b) an omitted constraint — the model never encoded the limit, so the
answer is genuinely optimal for the wrong model. Discriminating check: evaluate the returned
solution in an independent verifier that uses exact rational arithmetic on the *original*
coefficients. If the violation disappears, it was tolerance; if it persists, the constraint is
missing from the model. Correction: rescale rows to coefficient magnitude near 1 and re-check the
budget in §4.4; separately, add a verifier to the deployment pipeline so this class of error cannot
ship silently.

**F3. Two runs of the same model give different objectives.**
Competing causes: (a) nondeterministic parallel search — thread count or scheduling changes the
tree; (b) the model is degenerate, so multiple optima share the same objective and the solver's
tie-breaking (which depends on variable order) returns different *solutions* with equal value.
Discriminating check: compare the *objective values*, not the solutions. Equal values with
different solutions is degeneracy; different values is nondeterminism. Correction: for (a) set a
deterministic mode and a fixed seed, and pin the solver version; for (b) add a secondary objective
(smallest index sum, or a tiny cost on the tie-breaking variable) to select one canonical optimum.

**F4. Warm start makes the second run slower than a cold start.**
Competing causes: (a) the previous solution is infeasible for the new data, so the solver spends
time repairing it; (b) the previous basis is valid but pathological for the new cost vector,
sending the simplex down a long path. Discriminating check: check feasibility of the warm-start
solution against the new right-hand sides *before* handing it over; if infeasible, the cause is
(a). If feasible, time a cold run on the same instance — if cold is faster, the cause is (b).
Correction: warm start only bases, not full solutions, and only when the data delta is small;
otherwise cold start with presolve.

**F5. Benchmark says solver X is twice as fast, production says the opposite.**
Competing causes: (a) the benchmark used instances from one week, so it measured one data regime;
(b) the benchmark machine had different memory bandwidth or thread count than production, and the
winning configuration is the one that exploits the benchmark's parallelism. Discriminating check:
stratify the benchmark by week and by instance size; if X's advantage is concentrated in one
stratum, cause (a) is live. Then re-run the top two on production-like hardware with the production
thread count; if the ranking flips, cause (b). Correction: benchmark on a stratified set that
includes the worst week, on production hardware, with the production thread limit.

**F6. Presolve removes a constraint and the "infeasible" verdict is wrong.**
Competing causes: (a) the model really is infeasible and presolve found it early; (b) presolve
applied a reduction that is invalid under a tolerance, e.g. a bound tightening that assumed
coefficients were exact when they were rounded inputs. Discriminating check: re-run with presolve
off. If the model becomes feasible, the verdict came from presolve; if it stays infeasible, the
solver will usually emit a conflicting row set you can inspect. Correction: keep presolve on for
production speed, but always re-run a disputed infeasibility with presolve off and an
irreducible-infeasible-set report before telling a business owner their plan is impossible.

**F7. The gap target is met but the plan is unusable for operational reasons.**
Competing causes: (a) the objective omits a soft cost the planners care about (driver fatigue,
customer preference), so a gap-optimal plan is not the plan anyone wants; (b) the solution is
symmetric-degenerate and the chosen representative is operationally ugly (one vehicle does eleven
stops, its twin does none). Discriminating check: ask whether all plans within the gap band are
acceptable. If not, the cause is (a) — the model, not the solver. If several equal-value plans
differ wildly in structure, the cause is (b). Correction: add the soft costs with explicit weights
and re-derive the tolerance from the new objective scale; add balance constraints to remove the
degenerate family.

**F8. Objective improves continuously but the reported gap stalls.**
Competing causes: (a) the dual bound is stuck because the LP relaxation cannot see the integer
structure — cuts are being generated but are ineffective; (b) numerical trouble — the solver is
discarding cuts as violated below tolerance or cycling on a degenerate vertex. Discriminating
check: read the cut counts and the "cuts applied" ratio in the log. Many generated, few applied
indicates (b); few generated at all indicates (a). Correction: for (a) reformulate (tighten big-M,
add valid inequalities from the problem's combinatorics); for (b) rescale and loosen the cut
tolerance slightly, then re-tighten.

**F9. Licence or thread limit forces a configuration change that invalidates history.**
Competing causes: (a) fewer threads genuinely slow the run; (b) the reduced configuration changes
the search path, so solutions differ in structure even at equal objective, and downstream systems
keyed on the old structure break. Discriminating check: compare objective values across the change
(equal) and solution structure (differs). If only structure differs, the incident is downstream
coupling, not performance. Correction: introduce a canonical tie-break before the configuration
change, and version the solver configuration in the output record so consumers can detect it.

**F10. An instance solves in milliseconds in testing and hours in production.**
Competing causes: (a) presolve behaves differently because production data has more zero rows or
duplicate rows, changing the presolved size; (b) the test harness silently applied a time limit
and returned the first incumbent, which happened to be optimal on the test data. Discriminating
check: log the presolved row and column counts, not the input counts, and compare them between
environments; then confirm the test harness reports "optimal" rather than "feasible". Correction:
record presolved dimensions in the run manifest and assert they match expectations; never accept
a test result without the termination status string.

## 9. Complexity & Cost Analysis

Cost of a solver choice has three components: licence, compute time, and the human cost of the
workflow around it. The third usually dominates and is almost never measured.

**Compute.** For MILP, worst-case complexity is exponential in the number of integer variables; the
practical cost driver is nodes multiplied by time per node, where time per node grows roughly with
the cube of the constraint matrix's column count for dense LP solves and near-linearly for sparse
ones. A model with 1,780 presolved rows and 610 binaries at 0.4 s per node and 12,000 nodes costs
$12{,}000 \times 0.4 = 4{,}800$ s of CPU. On 4 threads with 60 % parallel efficiency the wall time
is $4{,}800/(4 \times 0.6) = 2{,}000$ s, which blows a 600 s deadline — the arithmetic alone tells
you the configuration cannot meet SLA, before any benchmark.

**Parallel efficiency.** Amdahl's law applied to the tree search: if a fraction $s$ of the work is
inherently serial (root cuts, presolve, final proof), speed-up at $p$ threads is
$1/(s + (1-s)/p)$. With $s = 0.15$ and $p = 8$: $1/(0.15 + 0.85/8) = 1/(0.15 + 0.10625) =
3.91$. Eight threads buy 3.9×, not 8×. This is why the 8-thread configuration in §5.4 saved only
$143 - 96 = 47$ s of 143 s, i.e. 32.9 %, on workloads where the serial fraction is larger than the
model above: an observed 1.49× speed-up from 8 threads implies
$1/1.49 = 0.671 = s + (1-s)/8$, so $7s/8 = 0.671 - 0.125 = 0.546$ and $s = 0.624$ — 62 % of that
workload's time is serial, which is why extra cores returned so little.

**Licence.** Commercial MIP licences are typically priced per concurrent job, sometimes per core.
A configuration that needs 8 cores to hit the deadline costs, on a per-core basis, twice one that
needs 4 — and if the workload is 200 jobs per week, the relevant number is core-hours per week,
not seconds per job. Compute it: 200 jobs, mean 143 s, 4 cores, 55 % average utilisation gives
$200 \times 143 \times 4/3600 = 114{,}400/3600 = 31.8$ core-hours per week of busy time, and
$31.8/0.55 = 57.8$ reserved core-hours. On 8 cores at 96 s and 40 % utilisation:
$200 \times 96 \times 8/3600 = 153{,}600/3600 = 42.7$ busy core-hours, i.e. $42.7/0.40 = 106.7$
reserved. The "faster" configuration costs 34 % more busy core-hours and 85 % more reserved
capacity, while simultaneously being less reliable on hits.

**Human and change cost.** Every solver configuration is a piece of undocumented state. Changing
the cut strategy invalidates historical run comparisons, retraining material, and any audit trail
that assumed reproducible output. Price this at the number of person-days to re-baseline the
exception thresholds, and require it in the change proposal — a "free" 20 % speed-up that costs
three days of threshold recalibration is usually not worth taking.

**Decomposition economics, computed rather than asserted.** Suppose a formulation has 60,000
columns and 400 rows because each of 400 customers has 150 possible service patterns. Solved
monolithically the LP has $60{,}000 \times 400$ coefficient entries, most zero; the solver handles
it, but the branch-and-bound tree on the integer master is large because the relaxation is weak.
Column generation instead solves a master with 400 rows over a small subset of columns plus a
pricing subproblem per customer. If the master holds 900 columns on average and pricing costs
0.02 s per customer, one iteration costs $400 \times 0.02 = 8$ s of pricing plus a master LP solve
of perhaps 0.05 s. Reaching convergence in 120 iterations costs $120 \times 8.05 \approx 966$ s.
Compare with a monolithic run that closes in 12,000 nodes at 0.4 s per node, i.e. 4,800 s: the
decomposition wins by roughly 5×, and the margin grows with the number of patterns per customer
because the monolithic cost scales with columns while the pricing cost does not.

**The tailing-off trap.** Column generation's last iterations improve the bound by fractions of a
percent while costing full price. Stopping at a 0.5 % master gap after 90 iterations costs
$90 \times 8.05 = 725$ s instead of 966 s, and then branching on the generated columns recovers the
remaining gap. The decision rule is: never pay for generation beyond the point where the marginal
bound improvement per second falls below the marginal improvement per second of branching.

**Cost of reproducibility.** Deterministic single-threaded mode typically costs 1.3–2.0× the wall
time of a tuned parallel run on the same instance. Whether that is acceptable is an arithmetic
question: if 200 weekly jobs at 143 s each on 4 threads cost 31.8 core-hours, the deterministic
configuration at 233 s single-threaded costs $200 \times 233/3600 = 12.9$ core-hours while using
one core — 59 % cheaper in licence capacity, 63 % more expensive in latency
($233/143 = 1.63$). Where the deadline has slack, determinism is the cheaper choice; where it does
not, pin the version and accept nondeterminism with a documented seed and a recorded thread count.

**Budget allocation rule.** Given a fixed engineering budget for making a model faster, spend in
this order: (1) formulation and symmetry, typically 2–10× for days of work; (2) presolve-friendly
data hygiene and scaling, 1.2–2× for hours; (3) solver parameter tuning, 1.05–1.5× for days and
which must be re-tuned each version; (4) hardware and threads, up to 4× but with ongoing licence
cost; (5) switching solver vendors, highly variable and carries re-validation cost. Teams invert
this order constantly because tuning is the most visible activity.

## 10. Key Theorems / Results Reference

**LP strong duality.** If an LP is feasible and bounded, primal and dual optimal values coincide.
Consequence for selection: any LP solver can certify optimality by exhibiting feasible primal and
dual solutions with equal objective; this is why LP "optimal" is trustworthy and why the dual
prices are usable for sensitivity analysis.

**LP optimal basis at a vertex.** An optimal solution exists at a vertex of the feasible polyhedron
when the optimum is finite. Consequence: simplex methods move vertex to vertex and terminate; and
degenerate vertices (more active constraints than variables) cause stalling, which is why
degeneracy shows up as non-improving iterations rather than as wrong answers.

**LP polynomial solvability.** Interior-point methods solve LPs in polynomial time in the input
length for a given accuracy. Consequence: LP size alone is not a reason to fear the run; numerical
conditioning is.

**Integrality of totally unimodular matrices.** If the constraint matrix is totally unimodular and
the right-hand side integral, every vertex of the LP relaxation is integral, so the LP solves the
integer problem. Consequence: network flow and assignment models need no branch-and-bound at all.
This is the single most valuable structural test before choosing a MILP solver — run it first.

**MILP NP-hardness.** Binary programming contains satisfiability as a special case, so no
polynomial algorithm is known and none is expected. Consequence: worst-case guarantees do not
exist; the only defensible promise is a gap at a deadline.

**Branch-and-bound correctness.** If the bounding function is a valid relaxation (never
underestimates a minimisation objective within a subtree) and branching partitions the feasible
set, the algorithm terminates with a proven optimum. Consequence: any "wrong" MILP answer traces
to either an invalid bound (bad formulation, invalid cut) or an early stop misreported as optimal.

**Cut validity.** A cut removes only points that violate the integrality requirement. Consequence:
adding an invalid cut produces a fast, confidently wrong answer — the most dangerous failure in the
list, because the solver still reports a certificate for the cut-augmented model.

**Convexity and local-global equivalence.** For a convex objective over a convex feasible set, any
local minimum is global. Consequence: a local NLP solver's certificate is meaningful only after
convexity is established; otherwise multi-start and interval reasoning are required.

**Duality gap as a stopping certificate.** For convex problems, a non-negative duality gap bounds
the suboptimality of the primal point. Consequence: the gap is a *guarantee*, not an estimate, and
it is why convex problems can be stopped early with confidence while nonconvex ones cannot.

**No-free-lunch for search.** Averaged over all problem instances, all search algorithms perform
identically. Consequence: solver selection must be justified on the structure of *your* instance
family; general claims that one metaheuristic dominates are unsupported.

**Amdahl's law.** Speed-up is bounded by the reciprocal of the serial fraction. Consequence: thread
counts above roughly $1/s$ return almost nothing; identify the serial fraction before buying cores.

## 11. Interview / Exam Questions

**Q1.** A minimisation MILP returns incumbent 82,400 and bound 79,900. What is the relative gap,
and may the run stop under a 3 % tolerance?
*Answer:* $(82{,}400 - 79{,}900)/82{,}400 = 2{,}500/82{,}400 = 0.03034$, i.e. 3.03 %. It may not
stop under a 3 % rule; it misses by 0.03 percentage points, which is 25 currency units. Rounding
the gap to "3 %" and stopping is the error the question is testing — compute the unrounded value
and compare.
*Reasoning:* The denominator is the incumbent, not the bound; using the bound gives 3.13 % and
overstates the gap.
*Follow-up:* If the business tolerance were stated absolutely as 3,000 units, would the answer
change? Yes — absolute gap is 2,500 < 3,000, so it stops.
*Red flag:* Dividing by the bound, or reporting "about 3 %" without the direction of the error.
*Scoring:* 1 point unrounded quotient, 1 point correct comparison, 1 point recognising the
absolute/relative distinction.

**Q2.** Your MILP has 610 binaries and solves in 3 s on historical data but 40 minutes on one new
instance. Name the first two checks, in order.
*Answer:* First, compare the LP relaxation objective against the historical range — a large
degradation or infeasibility indicates a data-driven near-infeasibility, not a harder search.
Second, compare presolved dimensions (rows, columns) with the historical distribution; a jump in
rows means duplicated or contradictory data got through validation.
*Reasoning:* Both checks separate "the problem got harder" from "the model got different", and the
second is far more common than the first.
*Follow-up:* The relaxation is fine and dimensions match. What next? Inspect symmetry: a new
customer with interchangeable vehicles identical to an existing one can multiply the tree.
*Red flag:* Answering "add more time" or "switch solver" without a discriminating measurement.
*Scoring:* 1 point relaxation check, 1 point dimension check, 1 point ordering rationale.

**Q3.** Explain why a solver can report "optimal" for a model whose solution violates a physical
limit, with the mechanism and the fix.
*Answer:* Integrality tolerance permits variables to sit at $10^{-5}$ from an integer; multiplied
by large coefficients this leaks into row activity. With coefficient 340 and 60 terms, the leak is
$10^{-5} \times 20{,}400 = 0.204$ kg — harmless; with coefficients near $10^{6}$ and 50 terms it is
10 units per variable, up to 500 units of violation that the solver still calls feasible. The fix
is row/column scaling so coefficients lie in $[10^{-2},10^{2}]$, plus an independent verifier.
*Reasoning:* The optimality claim is about the model under its stated tolerances, not about physics.
*Follow-up:* Which tolerance governs — integrality or feasibility? Both interact; the feasibility
tolerance is absolute on the row, the integrality tolerance is per variable, so the row's
coefficient magnitudes convert one into the other.
*Red flag:* Claiming the solver is buggy, or that the constraint is missing, without checking the
scaled leak.
*Scoring:* 1 point mechanism, 1 point numeric example, 1 point fix.

**Q4.** You must choose between a configuration with SGM 96 s and 8/10 hits and one with SGM 143 s
and 10/10 hits, under a 600 s deadline and a hard 0.5 % tolerance. Which, and what would change it?
*Answer:* The 10/10 configuration. With a hard tolerance, a miss is a process failure, not a
degraded result; two weeks in ten producing 0.83 % error violates the tolerance contract. It would
change if the tolerance were soft (report the plan with a flag) or if the missing 2 instances were
a small, predictable subset that could be routed to a different configuration.
*Reasoning:* SGM weights time; the business weights reliability under a hard constraint.
*Follow-up:* What if the 8/10 configuration's misses were all instances above 900 binaries? Then a
size-based router is defensible: use it below the threshold and the slow configuration above.
*Red flag:* Choosing on SGM alone without mentioning the hit count.
*Scoring:* 1 point choice, 1 point justification, 1 point condition that flips it.

**Q5.** Derive the expected number of explored nodes for a 30-binary model in which the bound
fathoms 60 % of nodes, and state the condition under which the count would explode.
*Answer:* With $q = 0.6$ the expected surviving children per node is $2(1-q) = 0.8$, so
$N(30) = (1 - 0.8^{31})/0.2$. Compute $0.8^{31}$: $\ln 0.8 = -0.22314$, times 31 is $-6.9174$, so
$0.8^{31} = e^{-6.9174} = 9.904\times10^{-4}$. Then $N = (1 - 0.000990)/0.2 = 4.995$, about
5 nodes. The count explodes when $q$ falls below 0.5, because then $2(1-q) > 1$ and the geometric
series diverges: at $q = 0.45$ the ratio is 1.1 and $N(30) = (1.1^{31}-1)/0.1 = (19.19-1)/0.1
\approx 182$; at $q = 0.4$ it is $8{,}813/2^{0} $ scaled to 30 levels, namely
$(1.2^{31}-1)/0.2 = (288.1-1)/0.2 \approx 1{,}436$.
*Reasoning:* Each node has two children and each child survives with probability $1-q$, so the
expected node count at depth $k$ is $\bigl(2(1-q)\bigr)^{k}$, and summing over depths gives the
geometric series.
*Follow-up:* With 200 binaries at $q = 0.6$, what is $N$? $(1-0.8^{201})/0.2 \approx 5$ — the
series saturates, which is why strong bounds make variable count almost irrelevant; the same 200
binaries at $q = 0.45$ give $(1.1^{201}-1)/0.1$, and $1.1^{201} = e^{201 \times 0.09531} =
e^{19.16} = 2.1\times10^{8}$, i.e. hundreds of millions of nodes.
*Red flag:* Answering $2^{30}$ (worst case presented as an expectation), or using $(1-q)^{k}$
instead of $\bigl(2(1-q)\bigr)^{k}$ and thereby dropping the branching factor.
*Scoring:* 1 point geometric series with the factor 2, 1 point arithmetic, 1 point critical
threshold $q = 0.5$ identified.

**Q6.** When is an LP relaxation guaranteed to give an integer optimum, and why does that matter
for solver selection?
*Answer:* When the constraint matrix is totally unimodular and the right-hand side is integral,
every vertex is integral, so the LP optimum is integral. For assignment, transportation and network
flow models this removes the need for a MILP solver entirely — a 100,000-arc flow problem becomes
a polynomial-time LP or a specialised network algorithm.
*Reasoning:* Total unimodularity is preserved by transposition and by adding identity columns,
which is why standard forms of these models qualify.
*Follow-up:* Adding one side constraint (e.g. a cardinality limit) usually destroys total
unimodularity. Then what? You are back to MILP, but with a strong LP bound, so branch-and-bound
will usually close quickly.
*Red flag:* Claiming any integer model with integer data is integral.
*Scoring:* 1 point condition, 1 point example, 1 point consequence.

**Q7.** Two runs with identical inputs and identical configuration produce equal objectives but
different route structures. Diagnose and state the one-line fix.
*Answer:* Degeneracy: multiple optimal solutions with equal value, and the solver's tie-breaking
depends on variable ordering or thread scheduling. The fix is a canonical secondary objective — a
lexicographic or tiny-weight term on index order — so exactly one optimum is selected.
*Reasoning:* Equal objective rules out nondeterminism in the *value*; differing structure at equal
value is the definition of alternative optima.
*Follow-up:* Would setting a random seed help? No — a seed fixes one path, but the underlying
multiset of optima remains, and a solver upgrade will change the path again. The secondary
objective is version-stable.
*Red flag:* "Set the seed" as the whole answer.
*Scoring:* 1 point diagnosis, 1 point distinction from nondeterminism, 1 point fix.

**Q8.** Compute the weekly core-hours for 150 jobs at 210 s each on 6 cores with 50 % utilisation,
and compare with 150 jobs at 150 s on 12 cores at 35 % utilisation.
*Answer:* First: $150 \times 210 \times 6/3600 = 189{,}000/3600 = 52.5$ core-hours, but utilisation
50 % means the cores are reserved half the time, so reserved core-hours are
$52.5/0.5 = 105$. Second: $150 \times 150 \times 12/3600 = 270{,}000/3600 = 75$ core-hours, and
reserved $75/0.35 = 214.3$. The faster wall-clock configuration costs roughly twice the licence
capacity.
*Reasoning:* Utilisation divides, because low utilisation means you pay for idle reservations.
*Follow-up:* If jobs can share a pool, utilisation rises and the comparison narrows; pool
scheduling is the lever, not the solver.
*Red flag:* Comparing wall time only, or multiplying by utilisation instead of dividing.
*Scoring:* 1 point core-hours, 1 point utilisation direction, 1 point conclusion.

**Q9.** A colleague proposes switching to a global MINLP solver because "local solvers might miss
the optimum". What two facts decide whether that is sensible?
*Answer:* First, the size: spatial branch-and-bound on 150 nonconvex variables typically cannot
close a meaningful gap in operational time, so the global solver may return a worse point at the
deadline than a well-started local search. Second, the consequence of a local optimum: if the
objective is a cost estimate reviewed by a human, a 2 % local gap is acceptable; if it is a safety
margin, it is not.
*Reasoning:* The certificate's value depends on the decision's risk, and the method's cost depends
on dimension.
*Follow-up:* What cheap middle ground exists? Multi-start from structured seeds plus an interval
bound on the objective to detect gross nonconvexity.
*Red flag:* "Global is always better."
*Scoring:* 1 point size argument, 1 point risk argument, 1 point middle ground.

**Q10.** Interpret a log line showing gap rising from 1.4 % to 2.1 % while the objective improves
from 51,200 to 50,900.
*Answer:* The incumbent improved by 300 units, but the dual bound improved by more in absolute
terms. Before: with gap 1.4 %, bound was $51{,}200 \times 0.986 = 50{,}483$. After: bound is
$50{,}900 \times 0.979 = 49{,}832$. The bound fell 651 units while the incumbent fell 300, so the
gap widened. This is normal: cuts tightened the relaxation faster than the heuristic found better
solutions.
*Reasoning:* The gap is a ratio of a difference to the incumbent; both move.
*Follow-up:* Should you stop? Only if the absolute gap is within tolerance: before 717, after
1,068 — it grew, so no.
*Red flag:* Concluding the solver is malfunctioning.
*Scoring:* 1 point reconstruction of both bounds, 1 point explanation, 1 point decision.

**Q11.** What does "warm start" change, and what does it never change?
*Answer:* It changes the search path and therefore running time; it never changes the optimal
value or the validity of the certificate, because the algorithm still proves optimality from the
relaxation. It can, however, change *which* optimal solution is returned when there are ties.
*Reasoning:* A warm start supplies an initial basis or incumbent, not a shortcut in the proof.
*Follow-up:* When does a warm start hurt? When the supplied solution is infeasible under new data,
or when its basis is pathological for the new costs — both add repair work.
*Red flag:* Claiming warm starts improve solution quality.
*Scoring:* 1 point time only, 1 point tie-breaking caveat, 1 point harm case.

**Q12.** State the selection procedure you would defend in a design review, in five steps.
*Answer:* (1) Classify model class and check structural shortcuts (total unimodularity, convexity,
pure network). (2) Compute the root relaxation gap on three representative instances; if under
1 %, stop tuning and pick any exact solver. (3) Translate the business tolerance into absolute and
relative gap targets and set both. (4) Benchmark two configurations on a stratified instance set
with the real deadline and production thread count, reporting SGM plus hit count. (5) Define the
fallback: trigger, alternate configuration, and the exception report field.
*Reasoning:* Steps 1–3 cost hours and eliminate most of the search space; step 4 is the only one
that needs compute.
*Follow-up:* What would make you skip step 4? A model that closes at the root or is totally
unimodular — there is nothing to benchmark.
*Red flag:* A procedure that begins with benchmarking or with vendor comparison.
*Scoring:* 1 point classification, 1 point tolerance translation, 1 point benchmark design,
1 point fallback.

**Q13.** Your benchmark used 10 instances from one week and shows a 2× speed-up. Give the cheapest
experiment that could overturn it.
*Answer:* Stratify by instance size and re-run on the quartile with the largest binaries. A 2×
average driven by small instances is common; production pain lives in the largest. Cost: one
afternoon of compute, no new licences.
*Reasoning:* Aggregate means hide stratum effects, and solver behaviour is strongly size-dependent.
*Follow-up:* If the advantage survives stratification, what next? Re-run on production hardware
with the production thread limit, because thread scaling is the second hidden variable.
*Red flag:* "Collect more instances" without stratification — more of the same week adds nothing.
*Scoring:* 1 point stratification, 1 point cost awareness, 1 point next step.

**Q14.** A model with 40 binaries solves in 2 seconds; the same model with 41 binaries takes 20
minutes. What is the most likely single explanation?
*Answer:* The 41st binary broke a structure the solver was exploiting — most often total
unimodularity or a network substructure that the extra column destroyed, or a big-M that became
binding and loosened the relaxation. It is a formulation event, not a scale event: one variable
cannot change exponential behaviour unless it changes the structure.
*Reasoning:* Exponential growth is continuous in $n$; a discontinuous jump implicates structure.
*Follow-up:* How do you confirm? Remove the 41st variable and check the root gap; if the gap jumps
from near zero to several percent, the relaxation lost its integrality property.
*Red flag:* "MILP is exponential, so that is expected."
*Scoring:* 1 point structural diagnosis, 1 point reasoning against pure scaling, 1 point test.

**Q15.** Where does the "certified bound" come from in a MILP, and why can a heuristic not produce
one?
*Answer:* From the LP relaxation plus valid cuts over the open nodes; because every integer solution
in an open subtree has objective at least the subtree's relaxation bound, the minimum over open
subtrees is a valid lower bound. A heuristic explores points without any relaxation, so it has no
statement about unexplored regions.
*Reasoning:* The bound is an argument about the *complement* of the searched set, not about the
searched set.
*Follow-up:* Can a heuristic be given a bound? Yes, by pairing it with a relaxation — that is
exactly what a matheuristic with an LP-based bound does.
*Red flag:* "The best found value is a bound."
*Scoring:* 1 point relaxation origin, 1 point complement argument, 1 point matheuristic note.

## 12. Quick Reference Card

| Question | Answer |
|---|---|
| Stop when? | Both a relative and an absolute gap target, whichever fires first; derive the relative target as tolerance/incumbent, rounding down |
| Which gap formula? | $g = (z - \underline{z})/\lvert z \rvert$ for minimisation; denominator is the incumbent, never the bound |
| Instance hard or model broken? | Compare LP relaxation objective and presolved row/column counts with history before touching solver settings |
| Safe integrality tolerance? | $\varepsilon \le V/(kA)$: acceptable violation over terms times coefficient magnitude; rescale rows into $[10^{-2},10^{2}]$ |
| Compare two configs how? | Shifted geometric mean of censored times with $\tau = 1$ s, reported together with the hit count |
| Reproducible output? | Deterministic mode, pinned version, fixed thread count, plus a canonical secondary objective against degeneracy |
| Certificate value? | LP/QP: proof and duals. MILP: proof for the model as written. Local NLP/metaheuristic: none |
| Deadline rule? | Read the objective-versus-log-time knee; set the deadline at the knee, not at the wall-clock budget |
| Fallback rule? | Named trigger (gap and elapsed fraction), named alternate configuration, warm start only for small data deltas |
| Effort order? | Formulation and symmetry, then scaling and data hygiene, then parameters, then hardware, then vendor |
