---
title: "Metaheuristics"
topic: "Neighbourhood search, acceptance criteria and population methods for problems with no exploitable exact structure"
collection: "Part 03 — Machine Learning, Optimisation and Intelligent Retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Metaheuristics

Every number in this sheet is either derived in place from a stated model or is a constructed
teaching value, and both are labelled. The running example is an invented twelve-point routing
instance inside a 16.1 km by 16.5 km box, marked ILLUSTRATIVE throughout; its coordinates, its
optimum and its search traces were all computed here so that every printed figure can be re-derived
by hand from the printed inputs. No vendor benchmark, published solver ranking or named company
result is claimed. Where a statement rests on a theorem from the literature, the theorem is named as
a mechanism claim about the method's behaviour and the reader is expected to consult the primary
source before relying on it operationally.

## 1. File Header

**Purpose.** Turn "use a metaheuristic" from a slogan into an auditable engineering decision:
which representation, which neighbourhood, which acceptance rule, which stopping budget, and what
evidence lets a reviewer believe the answer is good enough to ship.

**Prerequisites.** Comfort with finite state spaces and counting, the ability to read a transition
probability, working knowledge of the difference between a heuristic, a bound and a certificate,
and the habit of recording a random seed with every number. A reader who cannot say what the
neighbourhood of a candidate solution is cannot evaluate any method in this sheet.

**Outcome.** After this sheet the reader can (a) decide in ten minutes whether a metaheuristic is
the right tool at all, (b) construct a neighbourhood and an acceptance rule that provably target a
stated distribution, (c) calibrate a starting temperature and a cooling rate from measured move
deltas instead of copying defaults, (d) size a replication study so that a reported mean has a
stated standard error, and (e) state honestly what the result does not prove.

**Exclusions.** This sheet does not cover gradient-based continuous optimisation, does not cover
constraint-programming propagation, does not cover the design of the objective function itself (a
modelling question owned by the formulation sheet), and does not cover experiment design for
stochastic simulators beyond the replication arithmetic in §9. It also does not rank commercial
software. Distribution-fitting for randomised search belongs with the simulation experiment sheet.

**Reading rule.** Three units are mandatory whenever a number from this sheet is quoted onwards:
the objective's unit (kilometres here), the evaluation budget (objective calls, not seconds), and
the seed policy (one seed or a stated ensemble). A mean objective without a standard error, and a
"good solution" without a bound or a comparator, are both unreadable claims and should be sent
back to the author rather than into a decision memo.

**Position in the unit.** This sheet consumes the model produced by `01_optimisation_formulation.md`
and the gate decision produced by `02_solver_selection.md`: it is the layer that runs only after
the exact-solver gate has been closed with a reason. It supplies the search machinery that the
routing case study exercises and the contrast that the scheduling case study uses to explain why a
proof of optimality was worth paying for there.

**Evidence grades used.** DERIVED means the result follows from arithmetic or from a stated
transition rule reproduced in this file. ILLUSTRATIVE means the input data are constructed for
teaching and no real system was measured. HEURISTIC means a rule of thumb that survives practice
but has no proof attached; those are marked where they appear so a reviewer can discount them.

## 2. Concept Map / Mental Model

A metaheuristic is four decisions stacked, not one algorithm selected from a menu. The decisions are
the *representation* (what a candidate solution is), the *neighbourhood* (which nearby candidates a
move can reach), the *acceptance* (when a worse candidate is kept anyway), and the *budget* (when
the search stops and what is reported). Almost every disappointing result attributed to "the
algorithm" is a defect in one of these four, and the defect is usually diagnosable by instrumenting
the search rather than by swapping the algorithm name.

```text
representation  ->  neighbourhood  ->  acceptance  ->  budget  ->  report
     |                   |                 |             |
  feasibility        connectivity      escape power   evidence
  encoding cost      move cost         stationarity   error bars
     |                   |                 |             |
  wrong encoding -> disconnected -> quenches or  -> no comparator,
  hides good points   graph            wanders      no bound
```

The mental picture that carries the most explanatory weight is a *landscape*: each candidate is a
point, each move is an edge, and the objective is height. Local search walks downhill only. A
metaheuristic either (i) sometimes walks uphill on purpose — simulated annealing, threshold
accepting, tabu with aspiration — or (ii) maintains several walkers and recombines them — genetic
algorithms, particle swarms, ant colonies — or (iii) restarts from perturbed incumbents — iterated
local search, variable neighbourhood search, large neighbourhood search.

Three structural facts organise the whole field and each has a numeric consequence computed later.

*First, the neighbourhood graph determines what is reachable at all, and whether a stopping point is
an optimum of anything you care about.* The 62.1 km tour of the running instance is optimal for the
2-opt move set: all 54 moves have a positive delta, the smallest being 0.10 km. It is not optimal
for a move set that also relocates a single stop, because moving stop 10 from position five to the
end lowers the tour to 53.67 km in one step. The same candidate is a local optimum and a mediocre
solution depending only on the declared move set, which is why no local-optimality claim is
meaningful without naming the moves.

*Second, the acceptance rule determines the stationary distribution, not just the trajectory.* The
Metropolis rule is not a heuristic flourish: with a symmetric proposal it makes the chain's
long-run distribution exactly proportional to exp(-f/T), which is checkable numerically to machine
precision (§5c). That is the strongest guarantee any method in this sheet offers, and it is a
guarantee about the *distribution of visited solutions*, never about the single best solution seen.

*Third, the budget determines the achievable quality, and the relationship is steeply nonlinear.*
On the twelve-point instance, 500 proposals per run reached the optimum in 28 of 30 seeds; 2,000
proposals reached it in 30 of 30. On a sixty-point instance the same annealing design at 200,000
proposals averaged 2.24 percent above the best value a competing method reached with 1.59 million
evaluations. Budget, and where the budget is spent inside the schedule, explains most of that
difference — not algorithm branding.

**What the map forbids.** It forbids the two most common shortcuts. You may not claim optimality
from a metaheuristic: the best visited point carries no certificate, and the gap to the true optimum
is unknown unless a bound is computed independently. You also may not compare two methods at
different budgets: an annealing run allowed 200,000 evaluations and an iterated local search allowed
1.59 million are not in the same experiment, and reporting them side by side without saying so is a
misrepresentation even when both numbers are correct.

## 3. Foundational Definitions

Each definition below is written so that it can be dropped into a design review sentence without
further explanation. Units are given where a quantity has one.

- **Candidate solution (state).** An element of the finite set the search moves through; here a
  cyclic ordering of twelve stops, written `0-3-6-1-…`, with the depot fixed at position one to
  remove the twelve rotational duplicates.
- **Objective function f.** A real-valued cost on candidates, in kilometres for the routing case.
  Minimisation is assumed; all inequalities flip for maximisation.
- **Move.** A function mapping a candidate to a neighbouring candidate. The 2-opt move deletes two
  edges of the cycle and reconnects the four exposed endpoints the other way, reversing the segment
  between the cuts.
- **Neighbourhood N(x).** The set of candidates reachable from x by one move. Its size for the
  2-opt move on a cycle of n stops is n(n-3)/2, which is 54 at n=12 and 1,710 at n=60.
- **Delta (move gain).** Δ = f(x') - f(x), in the objective's unit. A negative delta is improving.
  The entire arithmetic of local search, tabu and annealing is expressed in deltas because a delta
  costs four distance lookups instead of a full re-evaluation of a length-n tour.
- **Local optimum.** A candidate with no improving neighbour, so every Δ ≥ 0. Local optimality is
  relative to the neighbourhood, and the qualifier is not optional: the 62.1 km tour is 2-opt
  optimal at Δ_min = 0.10 km while a single-stop relocation still gains 8.5 km on it.
- **Basin of attraction.** The set of starting points from which a given descent rule terminates at
  a given local optimum. Basin size, not average cost, governs how often a restart finds the good
  attractor.
- **Escape barrier.** The smallest uphill delta that must be paid to leave a basin, in the objective
  unit. From the 62.1 km tour the cheapest uphill 2-opt move costs 0.10 km; the median uphill move
  costs 10.6 km, so the barrier distribution is wide and its minimum is uninformative on its own.
- **Temperature T.** A positive scalar with the objective's unit that scales how much uphill travel
  is tolerated; acceptance of a move with delta Δ occurs with probability exp(-Δ/T).
- **Acceptance criterion.** The rule converting a delta into a keep/reject decision. Metropolis is
  the canonical probabilistic rule; threshold accepting is the deterministic cousin that accepts any
  Δ below a falling threshold; late acceptance compares against a value recorded a fixed number of
  steps ago.
- **Cooling schedule.** The map from iteration index to temperature, given either as a geometric
  factor α per round or as a reciprocal-log law T(k) = c / log(1+k). The geometric form is the one
  used for budgets here because its iteration count is available in closed form.
- **Quenching.** Cooling fast enough that the chain stops exploring before it reaches a good basin.
  A quench is not a bug in the code; it is a schedule whose final temperature arrives too early.
- **Intensification.** Effort spent improving the current region, typically by running a descent to a
  local optimum before the next perturbation.
- **Diversification.** Effort spent leaving the current region, achieved by restarts, by a
  perturbation (kick), or by penalising revisited structure.
- **Restart.** Reinitialising the current state from a fresh random or constructive start, keeping
  the incumbent archive. Restarts parallelise trivially, which is their main practical advantage.
- **Kick / perturbation.** A randomised move applied to a local optimum before re-descending. The
  double-bridge kick cuts the cycle in three places and reorders the three segments, and it cannot
  be undone by a single 2-opt move, which is why iterated local search uses it.
- **Population.** A maintained set of candidates updated in rounds, as in genetic algorithms and
  particle swarms, in contrast with the single-trajectory methods above.
- **Selection pressure.** The degree to which a population method favours fitter parents. For
  tournament selection of size k over N ranked individuals, the probability that the individual at
  rank r wins a tournament is 1 - ((r-1)/N)^k, which at N=10 and k=3 gives 0.936 for rank 5.
- **Crossover.** A recombination operator on two parents. Order crossover copies a contiguous slice
  from one parent and fills the remainder in the other parent's visiting order, which preserves
  permutation validity without repair.
- **Mutation.** A small random modification applied after crossover, whose role is to restore
  diversity the selection has removed; its probability is a diversity dial, not a quality dial.
- **Pheromone.** A positive weight on graph edges, evaporated each round and reinforced by
  constructed solutions, used by ant-colony methods to bias the probabilistic construction rule.
- **Tabu tenure.** The number of iterations a recently reversed attribute stays forbidden, which
  prevents immediate cycling. A tenure range of ⌈0.1n⌉ to ⌈0.3n⌉, widened on stagnation, is the
  standard starting point; at n=12 that is two to four iterations.
- **Swarm.** A population of particles whose update combines inertia, an attraction to the particle's
  own best position, and an attraction to the swarm's best position.
- **Anytime behaviour.** The property that the method can be interrupted at any moment and return the
  best point found so far. All methods here are anytime; exact branch-and-bound is anytime only in
  the weaker sense that it also reports a bound.
- **Optimality gap.** (f_found - LB)/LB for a lower bound LB on the optimum, or, when no bound
  exists, the difference to a stated incumbent of known quality. A gap whose denominator is
  unstated is not a gap, it is a number.
- **Budget.** A declared limit on objective evaluations, moves, or wall-clock, fixed before the run
  and identical across compared methods. Evaluations are preferred because they are machine-neutral.
- **Stagnation.** A run of iterations with no improvement to the incumbent. Its length is the input
  to restart and tenure-widening rules, so it must be counted in the run's telemetry.
- **Premature convergence.** Loss of diversity in a population method such that the population's
  distinct candidate count collapses while the incumbent is still far from good.
- **Elite archive.** The set of best distinct candidates retained across restarts, so that a restart
  cannot lose a good solution the search has already paid for.
- **Seed.** The integer that fixes a pseudo-random stream. Reporting a result without its seed makes
  the run unreproducible, which is a reproducibility defect, not a stylistic one.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**FORMULA.** For a symmetric proposal q and objective f with temperature T, the Metropolis
acceptance probability is

```text
a(x, y) = 1                            if f(y) <= f(x)
a(x, y) = exp( -(f(y)-f(x)) / T )      otherwise
```

and the implied transition kernel is P(x,y) = q(x,y) a(x,y) for y ≠ x, with the rejected mass
returned to P(x,x).

**WHERE.** On a finite state space with a neighbourhood relation, one proposal per iteration, and a
temperature held fixed. T carries the objective's unit so that Δ/T is dimensionless; a temperature
quoted without its unit is meaningless because it is only interpretable relative to the deltas it is
compared with.

**CONSTRAINT.** The proposal must be symmetric, q(x,y) = q(y,x), and the neighbourhood must be
connected enough that every state can reach every other in finitely many steps with positive
probability. If the proposal is asymmetric — for example "pick a uniformly random neighbour of the
current state" on a graph whose degrees differ — the bare Metropolis rule gives the wrong stationary
distribution and the Hastings correction q(y,x)/q(x,y) is required (§4.2).

**NUMERICAL EXAMPLE.** Take f(x) = 3.0, f(y) = 0.5, T = 1.7, so y is the better state. The move
x → y has Δ = -2.5 and is accepted with probability 1. The reverse move y → x has Δ = +2.5 and is
accepted with probability exp(-2.5/1.7) = exp(-1.4706) = 0.2298. Dividing, the stationary
probabilities must satisfy π(y)/π(x) = 1/0.2298 = 4.3518 = exp(2.5/1.7), so the better state is
occupied 4.35 times as often. Note the direction: the ratio that the acceptance probabilities produce
is π(better)/π(worse), not the reverse, and inverting it is the single most common arithmetic slip in
this derivation.

**INTERPRETATION.** Uphill moves are not noise and are not a failure to converge: they are the
mechanism that produces the target distribution. A run with zero accepted uphill moves is a hill
climber, and a hill climber's stationary distribution is a point mass on whichever local optimum it
fell into.

**Detailed balance, stated as the reason the formula works.** For f(y) ≤ f(x) the forward
probability is q(x,y) and the reverse is q(y,x)exp(-(f(x)-f(y))/T). With π(z) ∝ exp(-f(z)/T) the
two quantities π(x)P(x,y) and π(y)P(y,x) differ by the factor exp(-(f(x)-f(y))/T) applied to both, so
equality holds. Because the chain is finite, irreducible and aperiodic, detailed balance implies
that this π is the unique stationary distribution. Detailed balance is a *sufficient* condition, not
a necessary one, which matters in §4.2 where a correction restores it.

**Validity condition for 4.1.** Symmetric proposal, fixed temperature, finite irreducible
neighbourhood graph, and an acceptance probability evaluated with the same f and the same units on
both sides. Violating the last clause is the usual silent failure: a delta computed from rounded
distances compared against a temperature calibrated on exact distances biases every acceptance.

### 4.2 General case

**FORMULA.** With an asymmetric proposal q(x,y), use the Metropolis–Hastings ratio

```text
a(x, y) = min( 1,  exp(-(f(y)-f(x))/T) * q(y,x)/q(x,y) )
```

and for a geometric cooling schedule with start temperature T0, final temperature Tend and per-round
factor α, the number of rounds is

```text
K = ln(Tend / T0) / ln(α)
```

while a start temperature calibrated to accept a reference uphill move with probability p0 is

```text
T0 = -Δ_ref / ln(p0)
```

**WHERE.** The first expression is used whenever the move distribution is not uniform over the
neighbourhood — for instance when proposing to reverse the last move, or when sampling neighbours in
proportion to a heuristic score. The second is used to convert a wall-clock or evaluation budget
into a schedule factor, and the third to convert a measured delta distribution into a defensible
start.

**CONSTRAINT.** α must satisfy 0 < α < 1, Tend must be small relative to the smallest delta of
interest but never zero (the logarithm diverges and the schedule stalls), and the calibration delta
Δ_ref must be a real quantile of the observed uphill deltas, not a guess. The three formulas are
coupled: changing T0 changes K at fixed α, and changing the budget changes α at fixed endpoints. The
Hastings ratio must be applied on *both* sides of the comparison, including the branch where the
exponential would be 1; applying it only to uphill moves leaves the chain biased, and the bias is
large enough to see (§5c).

**NUMERICAL EXAMPLE.** Using the twelve-point instance evaluated on distances rounded to 0.1 km, the
54 2-opt moves at the nearest-neighbour tour contain 3 improving moves and 51 uphill moves, and the
uphill deltas have minimum 0.1 km, lower quartile 5.7 km, median 10.1 km, upper quartile 17.7 km and
maximum 26.5 km. Calibrating to accept a 5.7 km uphill move 80 percent of the time gives
T0 = -5.7/ln(0.8) = -5.7/(-0.22314) = 25.544 km. Calibrating instead to accept the median move half
the time gives T0 = -10.1/ln(0.5) = 14.571 km. With T0 = 25.544, Tend = 0.05 and α = 0.995 the
budget is K = ln(0.05/25.544)/ln(0.995) = ln(0.0019574)/(-0.0050125) = (-6.2362)/(-0.0050125) =
1244.1 rounds, which at 50 proposals per round is 62,205 proposals. Choosing α = 0.999 instead needs
K = 6233 rounds, five times the work at the same endpoints, and choosing α = 0.99 needs only 621
rounds but arrives at the final temperature five times sooner.

**INTERPRETATION.** The schedule factor is a *time rescaling*, not a quality knob: at fixed endpoints
α only decides how long the search spends in each temperature band. The quality knob is the product
of T0 and the number of proposals spent below the scale of the barriers, which is why calibrating T0
from data and then fixing the budget is more reproducible than copying a factor from another paper.

**Validity condition for 4.2.** The schedule must remain above the smallest escape barrier long
enough for at least one accepted escape per candidate basin, and Tend must be below the smallest
delta that separates two competitive optima, otherwise the final phase cannot discriminate between
them. Quantitatively: at T = 0.05 km the acceptance probability for a 5.7 km uphill move is
exp(-114) ≈ 0, so the last few hundred rounds are pure descent, which is the intended behaviour.

### 4.3 Variants

**Threshold accepting.** Replace the probabilistic test with a deterministic one: accept any move
with Δ ≤ τ(k), where τ falls from τ0 to zero. Fewer random numbers per iteration, same qualitative
escape behaviour, no stationary-distribution guarantee. Choose it when the random-number budget is
the bottleneck, which happens in hardware or embedded settings rather than in ordinary analysis.

**Late acceptance.** Keep a circular buffer of the last L objective values and accept a move if it
improves on the value L steps ago. The buffer length L replaces temperature entirely and is easier to
defend to a non-specialist, at the cost of having no closed-form cooling arithmetic.

**Iterated local search.** From a local optimum x*, apply a kick to obtain x', descend to a local
optimum x'', and accept x'' if it beats x* (or under a small probabilistic rule). With a double-bridge
kick the method is a stochastic descent over the *set of local optima*, which is a much smaller space
than the set of tours. On the sixty-point instance, starting from the nearest-neighbour 2-opt optimum,
200 kicks at about 7,950 move evaluations each reached 612.1 km in 20 of 20 seeds at a total cost of
1.59 million evaluations. Starting instead from a random tour, the same 200-kick budget averaged
623.9 km with standard deviation 9.8 km and reached 612.1 km in only 2 of 20 seeds — the start, not
the kick, decided that outcome.

**Genetic algorithm with order crossover.** Maintain a population, select parents by tournament,
recombine by copying a slice from one parent and filling the rest in the other's order, then mutate
with small probability. The crossover preserves permutation validity by construction, so no repair
operator is needed. In the twelve-point instance, a population of 40 over 60 generations with
mutation probability 0.3 reached the optimum by generation 22 at roughly 5,340 evaluations, whereas
the same design with population 20 was still at 62.1 km after 60 generations: the population size,
not the operator set, decided the outcome.

**Particle swarm.** Each particle updates velocity as

```text
v_i <- w*v_i + c1*r1*(p_i - x_i) + c2*r2*(g - x_i)
x_i <- x_i + v_i
```

with inertia w, cognitive coefficient c1, social coefficient c2, personal best p_i and global best g.
The recursion's behaviour is governed by the roots of λ² - (1 + w - φ)λ + w = 0 with φ = c1 + c2.
At w = 0.9, c1 = c2 = 1.49445 the roots are complex with modulus sqrt(w) = 0.9487, so a deviation
decays by 0.9487 per iteration and to 2.7e-5 of its size after 200 iterations. At w = 0.729 the
modulus is 0.8538 and 200 iterations leave 1.9e-14. At w = 1.2 the modulus is 1.0954 and the swarm
diverges geometrically, which is observed as particles leaving the feasible box and returning
clamped.

**Penalty-based constraint handling.** For a knapsack with capacity C, minimise
`-value + λ*max(0, weight - C)`. The method is only as good as λ. With items of weight
[7, 5, 9, 4, 11, 6, 3, 8] and value [14, 11, 19, 9, 22, 13, 7, 16] at C = 26, a λ of 2.0 drives all
three tested starts to infeasible sets of weight 34, 35 and 45; λ of 2.5 recovers feasible values of
54, 56 and 55 depending on the start; λ of 3.0 still prefers an infeasible weight-28 set with value
60 from the empty start, because its penalised objective (-60 + 3×2 = -54) beats the feasible 56 at
-56. The crossover point is not a matter of taste: it is where λ times the violation exceeds the
value gained.

**Validity condition for 4.3.** Each variant's guarantee must be read from the variant, not from the
family. Threshold accepting and late acceptance have no stationarity guarantee at all; the genetic
algorithm has no convergence guarantee to the optimum under any finite budget; the swarm's stability
condition must be checked numerically for the chosen coefficients before the run, not diagnosed from
particles flying away.

### 4.4 Limits and counterexamples

**Limit 1: T → 0.** Acceptance of any uphill move tends to zero, so the chain becomes steepest
descent and converges to a local optimum of the neighbourhood, not of the problem. Concretely, the
62.1 km tour in the running instance is the limit of the 2-opt chain at zero temperature, and the
true optimum is 53.4 km, so the gap is 8.7 km or 16.3 percent. No amount of extra time at T = 0
closes that gap, because the barrier is structural.

**Limit 2: barrier above the temperature range.** If the escape barrier from the incumbent basin
exceeds the highest temperature ever used, the search behaves as if it were at zero temperature
regardless of the schedule. The clean counterexample is the 4-state chain in §5c: at T = 0.5 the
expected hitting time from the worst state to the best is 598 iterations, while at T = 8 it is 12.9.
A budget of 200 iterations at T = 0.5 will usually report the wrong answer with complete confidence.

**Limit 3: penalty weight below the violation price.** Any λ that does not dominate the marginal
value of violation returns infeasible incumbents, and it returns them *silently*, because the
penalised objective is a legitimate number. The knapsack λ = 2.0 case above produced a weight-45 set
against a capacity of 26 and reported a better "objective" than the feasible optimum.

**Limit 4: no-free-lunch reasoning misapplied.** Wolpert and Macready's result says that averaged
over all possible objective functions, all search algorithms have equal performance. It does not say
that any algorithm is as good as any other on your problem, and it does not license refusing to
encode problem structure. It does say that a claim of general superiority is unsupportable, so
benchmarks must be problem-family specific and the family must be declared.

**Counterexample to "a local optimum is a good solution".** The 62.1 km tour admits no improving
2-opt move, which is a complete and correct certificate of 2-opt local optimality: the minimum delta
over all 54 moves is +0.10 km. Relocating the single stop 10 from its position to the end of the
cycle produces 0-3-6-1-4-5-11-7-2-8-9-10 at 53.67 km, a gain of 8.5 km in one move, and that tour is
itself 2-opt optimal with a best delta of +0.10 km. Two different move sets therefore certify the
same tour as excellent and as 16 percent too long. The lesson is procedural: a local-optimality
certificate is a statement about a neighbourhood, so it must be reported together with the move set
and never as "the optimum".

**Counterexample to "more iterations always help".** On the twelve-point instance the annealing
result is stable from 2,000 proposals upward, and 16,000 proposals gives the identical 53.4 km at
eight times the cost. On the sixty-point instance, raising the budget of a badly calibrated schedule
from 200,000 to 800,000 proposals moved the mean from 625.8 km to 628.4 km — no improvement, because
the deficiency was the schedule's temperature range, not its length. Extra budget buys nothing when
the acceptance rule cannot express the needed escape.

**Validity condition for 4.4.** Each limit is a boundary on the *model of the search*, so the
diagnostic is to instrument the run and check which boundary was crossed: acceptance rate near zero
means quench, acceptance rate near one means the temperature is above every delta and the search is
a random walk, and a zero-variance result across seeds means the method is deterministic in disguise
or the seeds are not being used.

## 5. Worked Numerical Example

### 5a The instance and its baselines

The instance is twelve points in kilometres on a plane, with point 0 as the depot, marked
ILLUSTRATIVE. The coordinates are: 0 at (0.0, 0.0); 1 at (7.8, -2.8); 2 at (2.1, 11.2); 3 at
(-0.5, -2.2); 4 at (13.1, 3.6); 5 at (10.0, 5.9); 6 at (4.7, -3.3); 7 at (2.4, 11.3); 8 at
(2.5, 13.2); 9 at (-3.0, 10.2); 10 at (2.2, 2.7); 11 at (7.3, 9.1). Distances are Euclidean, rounded
to 0.1 km for the search and recomputed exactly for the sensitivity check in §5e.

The nearest-neighbour construction from the depot visits 0-3-6-1-10-11-5-4-7-2-8-9 and returns a
length of 67.2 km. Its edge list is 2.3 + 5.3 + 3.1 + 7.8 + 8.2 + 4.2 + 3.9 + 13.2 + 0.3 + 2.0 +
6.3 + 10.6. The single worst edge is 4-7 at 13.2 km, which is the signature of a greedy method: it
committed to point 4 early, when point 4 was cheap to reach, and paid for the return later.

Steepest-descent 2-opt over the 54 possible moves finds exactly one improving move from that tour:
cutting edges 10-11 (8.184 km) and 4-7 (13.183 km) and reconnecting as 10-4 (10.937 km) and 11-7
(5.371 km), for a delta of 10.937 + 5.371 - 8.184 - 13.183 = -5.059 km. The new tour
0-3-6-1-10-4-5-11-7-2-8-9 has length 62.1 km, and after the move every one of the 54 remaining moves
is non-improving, so the method stops. That stop is a local optimum, not the optimum.

The optimum, computed by Held–Karp dynamic programming in 2^11 × 11 × 11 = 247,808 transition
evaluations, is 53.4 km on the tour 0-3-6-1-4-5-11-8-7-2-9-10. The exact search is feasible here
because 19,958,400 distinct undirected tours is small enough to avoid, and 479,001,600 labelled
permutations is not the relevant count. The gap structure of the three baselines is therefore known
exactly: nearest neighbour is 25.8 percent above the optimum, 2-opt is 16.3 percent above it, and a
uniformly random tour averages 119.5 km with a standard deviation of 13.5 km over 200,000 draws,
whose best observed value was 75.3 km — worse than the nearest-neighbour tour, which is the cleanest
available demonstration that random sampling is not a baseline worth defending.

### 5b Annealing on the instance, with the delta distribution measured first

Before setting a temperature, the uphill deltas at the starting tour were measured: of the 54 2-opt
moves, three are improving, none is neutral, and 51 are uphill, with minimum 0.1 km, lower quartile
5.7 km, median 10.1 km, upper quartile 17.7 km and maximum 26.5 km. This distribution, not a
borrowed default, is what calibrates the schedule.

At T0 = 25.544 km the acceptance probability is 0.9961 for the smallest uphill delta of 0.1 km,
0.8000 for 5.7 km, 0.6734 for 10.1 km, 0.5001 for 17.7 km and 0.3544 for 26.5 km. Summing
min(1, exp(-Δ/T0)) over the 51 uphill moves predicts 33.9 accepted uphill moves per sweep, or 66.4
percent of the uphill neighbourhood, so the first sweep behaves as intended: heavy but not blind
exploration.

The run used α = 0.995 with 50 proposals per round and 1,245 rounds, which is 62,250 proposals and a
final temperature of 0.0498 km. Its trajectory on one seed, sampled every 50 rounds, is the useful
artefact because it shows where the budget went:

| Round | Temperature (km) | Current tour (km) | Incumbent (km) |
|---|---|---|---|
| 100 | 15.47 | 125.4 | 67.2 |
| 200 | 9.37 | 106.1 | 62.4 |
| 300 | 5.68 | 77.3 | 57.2 |
| 400 | 3.44 | 65.2 | 54.2 |
| 600 | 1.26 | 53.7 | 53.4 |
| 800 | 0.46 | 53.4 | 53.4 |
| 1245 | 0.05 | 53.4 | 53.4 |

The incumbent first beat the construction at round 125 and fell to 62.4 km by round 155, to 57.2 km
by round 277 and to 53.4 km by round 401 — one third of the way into the schedule. The remaining 844
rounds, two thirds of the budget, produced no improvement, and by round 800 the current tour had
collapsed onto the incumbent, which is the signature of a chain that is no longer exploring. The
acceptance rate over the whole run measured 0.247 averaged over 30 seeds, with standard deviation
0.0066 and a range of 0.213 to 0.251 across seeds.

That trace is the single most transferable observation in this sheet: calibration decided the first
third, and the last two thirds were pure descent that added nothing. A schedule truncated to 450
rounds would have produced the same answer at 36 percent of the cost, and a schedule that spent the
saved budget on four independent restarts would have reduced the failure probability rather than
polishing a converged chain. The distinction between "current" and "incumbent" is the whole of
annealing's reporting discipline: the current tour is deliberately bad for most of the run, and the
number you ship is the best one seen.

Thirty independent seeds at that design reached 53.4 km in 30 of 30 runs with zero observed variance.
That result is a property of a 12-point instance, not of annealing: with 54 moves per sweep and
62,250 proposals a run performs about 1,153 sweeps, so the instance is solved by saturation of the
neighbourhood rather than by any subtle escape mechanism. The honest conclusion is that this instance
cannot discriminate between good and mediocre schedules, which is exactly why the budget study below
is reported as a hit rate rather than as evidence of method quality.

Reducing the budget changes the picture in the way a practitioner needs to know: at 500 proposals, 28
of 30 seeds reached the optimum with mean 53.56 km and standard deviation 0.84 km; at 1,000 proposals,
29 of 30 with mean 53.68 km and standard deviation 1.53 km; at 2,000 and above, 30 of 30 with zero
variance. The single failures at 500 and at 1,000 proposals are a reminder that a hit-rate claim needs
the seed count attached: "reaches the optimum" is a statement about a distribution, and one run is a
sample of size one from it.

A restart-based alternative was also checked, because it is cheap to do and often forgotten. Four
independent 500-proposal runs taking the best of four reached the optimum in 40 of 40 trials, which
matches the arithmetic of independent failure: if one run of 500 proposals succeeds with probability
0.933, four independent runs fail with probability (0.067)^4 = 2.0e-5. Restarting is therefore the
first thing to try before lengthening a chain, since it parallelises perfectly and cannot be trapped
by an unlucky seed shared across the whole budget.

### 5c The acceptance rule verified against its own stationary distribution

The Metropolis rule is checkable, and checking it is the cheapest way to catch an implementation bug
before it reaches a production forecast. Take a four-state chain with costs 0.0, 3.0, 0.5 and 4.2 —
deliberately non-monotone in index order, so that "state 3 is best" is false — a reflecting random
walk proposal that offers each of the two adjacent indices with probability one half, and T = 1.7.

The transition matrix built from the rule has four rows. Row 0 proposes state 1 with probability 0.5
and is rejected with probability exp(-3.0/1.7) = 0.1712, so it holds 0.9144 and moves 0.0856. Row 1
proposes states 0 and 2, both cheaper, so both are accepted and the row is 0.5, 0.0, 0.5, 0.0. Row 2
proposes state 1 (cost 3.0, more expensive) with acceptance 0.2298 and state 3 (cost 4.2) with
acceptance 0.1134, giving 0.0, 0.1149, 0.8284, 0.0567. Row 3 proposes state 2 at lower cost, accepts,
and holds 0.5. Every row sums to 1.000000.

Power iteration on this matrix converges to (0.4998, 0.0856, 0.3724, 0.0422), and the Boltzmann
distribution exp(-f/T)/Z is (0.4998, 0.0856, 0.3724, 0.0422) to four decimals. The maximum residual
of detailed balance across all sixteen entries is 1.4e-17, which is floating-point zero.

The value of that check is best seen by breaking it deliberately. Keeping the same proposal but
applying the acceptance ratio only on uphill moves, and omitting the Hastings factor q(y,x)/q(x,y)
that unequal degrees demand, gives stationary weights (0.3428, 0.1174, 0.5109, 0.0290). State 2 is
then occupied 51.1 percent of the time instead of the correct 37.2 percent, a 37 percent relative
inflation, and the best state 0 drops from 50.0 to 34.3 percent. Applying the correction on both
branches — including the branch where the exponential is 1 — restores (0.4998, 0.0856, 0.3724,
0.0422) with a detailed-balance residual of 6.9e-18. A search with the uncorrected kernel still
"works" in the sense of returning tours, which is precisely why the defect survives casual testing.

The same chain answers a question annealing users ask constantly: how long must the search run before
it can plausibly find the best state? Solving the linear system for expected hitting times gives
h(state 1 → state 0) = 11.69 and h(state 3 → state 0) = 23.38 iterations at T = 1.7. At T = 0.5 the
same quantities are 299.0 and 598.0; at T = 3.0 they are 7.94 and 15.89; at T = 8.0 they are 6.46 and
12.91. Expected cost at stationarity, by contrast, falls monotonically as temperature falls: 1.5586
at T = 8, 1.0413 at T = 3, 0.6204 at T = 1.7, 0.3085 at T = 1.0 and 0.1403 at T = 0.5.

Those two facts point in opposite directions, and that tension is the whole design problem. A cold
chain has the better stationary distribution and needs the longest time to reach the good part of it:
598 iterations at T = 0.5 against 12.9 at T = 8. A schedule that ends at T = 0.05 has the best
distribution of all and, on a landscape of this shape, no realistic chance of exploring it. This is
the arithmetic behind "anneal slowly, then quench": the cooling schedule should spend its iterations
where the hitting times are small relative to the remaining budget, and the final low temperature is
a *reporting* phase rather than an exploration phase.

### 5d A harder instance where the design, not the instance, decides

The twelve-point instance saturates. A sixty-point instance (ILLUSTRATIVE, generated from a stated
seeded uniform distribution on a 100 km square) separates the methods. Its nearest-neighbour tour is
741.5 km; steepest-descent 2-opt improves it to 612.1 km in eleven improving moves after evaluating
20,520 moves, a 17.5 percent gain. Twenty random starts followed by 2-opt average 656.4 km with
standard deviation 18.9 km, so construction quality is worth about 44 km on this instance and the
variance across starts is wider than that. Only two of those twenty starts finished at 621 km or
better.

Annealing at 200,000 proposals, calibrated in the same data-driven way, gives a mean of 625.8 km with
standard deviation 27.2 km over 20 seeds, a 2.24 percent gap to 612.1 km, with individual runs as bad
as 736.3 km. Iterated local search from a random start with 200 double-bridge kicks gives 623.9 km
mean and 9.8 km standard deviation, with 2 of 20 seeds reaching 612.1 km, at a total cost of 1.59
million evaluations — eight times the annealing budget. At a matched budget of 800,000 evaluations,
which is 100 kicks at that per-kick cost, ILS averages 627.3 km with standard deviation 10.6 km, while
annealing averages 628.4 km with standard deviation 40.1 km and no seed at 612.1 km.

At equal budget the two methods are within about 1 km of each other, which is one tenth of the
across-seed spread: on this instance they are indistinguishable, and the eight-times-budget
comparison in the previous paragraph was measuring budget rather than method. The conclusion a
reviewer should draw is not "ILS beats annealing" but "the method whose inner loop is a full descent
spends its evaluations far better per unit of remaining gap on this instance, and the 2-opt descent
is doing the work in both cases".

### 5e Independent arithmetic re-check, and a sensitivity note

Recheck 1, additivity of the tour length. The optimum's twelve edges are 2.2561, 5.3151, 3.1401,
8.3096, 3.8601, 4.1869, 6.3127, 1.9026, 0.3162, 5.1971, 9.1263 and 3.4828 km. Summing in two
different orders gives 53.4056 km; the printed 53.4 is that value rounded to 0.1 km, and the rounding
contributes at most 0.05 km, so no conclusion in this sheet turns on it.

Recheck 2, delta versus full re-evaluation. The 2-opt move applied in §5a was scored by a delta
computed from four edge lookups, giving -5.059 km. Recomputing both tours in full gives 67.2263 →
62.1685 km, a difference of -5.0578 km. The two methods agree to 0.001 km because the delta formula
is exact for the move; a disagreement of more than a rounding unit would mean the move implementation
touches an edge it does not own, which is the classic 2-opt bug at the depot boundary.

Recheck 3, the bound. The minimum spanning tree over all twelve points is 43.2117 km, and the
one-tree bound — a spanning tree on points 1 to 11 plus the two cheapest edges from the depot, which
are 0-3 at 2.2561 and 0-10 at 3.4828 — is 43.0674 + 2.2561 + 3.4828 = 48.8063 km. Since the optimum
is at least as large as any lower bound, the found 53.4056 km tour is at most
(53.4056 - 48.8063)/48.8063 = 9.42 percent above the optimum; stated against the optimum it is 8.61
percent. Any claimed solution below 48.81 km is provably wrong. The direction of the denominator
matters: quoting 8.61 percent because it is the smaller number is a presentation choice that hides
the conservative figure an auditor should be given.

Sensitivity. All search figures above used distances rounded to 0.1 km for speed. Recomputing the
three tours with exact distances gives 67.2263, 62.1685 and 53.4056 km, and the MST changes from
43.3000 (sum of rounded edges) to 43.2117 km. The rounding therefore shifts the bound by 0.088 km and
the tours by at most 0.069 km, so no acceptance probability or gap in this sheet changes by more than
1 percent, and the largest single relative change is the 2-opt delta of -5.059 versus -5.0578 km, a
0.02 percent shift. If the distances had been rounded to whole kilometres the smallest uphill delta
of 0.1 km would have vanished and the neighbourhood's connectivity would have changed, which is the
failure mode this check exists to rule out.

## 6. Visual / Geometric Intuition

**The landscape is not a picture of the plane.** Practitioners routinely conflate the geometric
layout of the stops with the search landscape. They are different spaces: the plane has twelve
points, while the search space has 19,958,400 distinct cycles, and distance in the plane tells you
almost nothing about the height difference between two cycles that share eleven edges.

**The 2-opt move is a four-endpoint exchange.** Deleting two edges from a cycle leaves four odd-degree
endpoints, and there are exactly two ways to reconnect them into a cycle; one is the original, so the
move set has one alternative per pair of edges. Geometrically the move reverses an interval of the
tour, which is why it can remove a crossing: a crossing is precisely a pair of edges whose endpoints
alternate around the polygon, and one of the two reconnections removes the alternation.

**Crossings as a diagnostic, not a goal.** The nearest-neighbour tour on the running instance has one
crossing, the 2-opt optimum has none, and the true optimum has none either. That correlation is
tempting but not a theorem: in Euclidean instances an optimal tour is non-crossing, but a
non-crossing tour is not necessarily good, and in instances with obstacles, time windows, or
asymmetric costs, crossings can be optimal. Use crossing count as a cheap red flag on a *Euclidean*
instance and never as an objective term in a general setting.

**Temperature as a blur radius.** A useful intuition for choosing T0: at temperature T, two
candidates whose costs differ by about T are treated as roughly interchangeable, since their
acceptance ratio is within a factor of e. Calibrating T0 = 25.5 km on deltas of 5 to 26 km means the
first phase treats a 62 km tour and an 87 km tour as within a factor of 2.7 of each other, which is
exactly the blindness you want while the search has no information.

**The kick is a topological operation.** The double-bridge kick cuts the cycle at three places and
reassembles the three resulting paths in a different order. Its value is that no single 2-opt move
can undo it: reversing a segment of the kicked tour does not restore the original adjacency pattern,
so the subsequent descent is forced into a different basin rather than back into the one just left.
This is why iterated local search needs a *structured* kick rather than a random swap of two cities,
which a single 2-opt move frequently reverses.

**Basin geometry explains restart economics.** If a descent from a random start lands in the good
basin with probability p, then R independent restarts succeed with probability 1 - (1-p)^R. On the
sixty-point instance, 2-opt from a random start averaged 656.4 km with standard deviation 18.9 km
against a best of 612.1 km, and only two of twenty starts reached 621 km, so p is evidently small.
Restarts are the cheapest way to convert that small p into a reliable outcome, and they parallelise
without any communication, which serial annealing cannot do.

**Population methods move a cloud, not a point.** The intuitive difference between a single
trajectory and a population is that a population's diversity is a *measurable state variable*: count
distinct candidates per generation. When that count falls to one while the incumbent is still 15
percent above the best known value, the population has converged prematurely and no further
generations will help. That diagnostic has no analogue in a single-trajectory method, whose diversity
must be inferred from the accepted-move rate instead.

## 7. Comparison / Selection Table

### 7.1 Families and what each guarantees

| Family | Mechanism | Guarantee offered | Typical parameters | Cost per iteration |
|---|---|---|---|---|
| Steepest 2-opt descent | best improving move, repeat | local optimum of the move set | none, fully deterministic | n(n-3)/2 move evaluations |
| Simulated annealing | Metropolis acceptance, cooling | stationary distribution at fixed T | T0, α, proposals per round | one move evaluation |
| Threshold / late acceptance | deterministic uphill tolerance | none | τ0, buffer length L | one move evaluation |
| Tabu search | best move with short-term memory | none, but avoids cycling | tenure ⌈0.1n⌉ to ⌈0.3n⌉, aspiration | full neighbourhood scan |
| Iterated local search | kick plus descent | none | kick strength, acceptance, start | one descent per kick |
| Genetic algorithm | selection plus recombination | none under finite budget | population, generations, mutation rate | one evaluation per child |
| Ant colony | pheromone-biased construction | converges to a tour under stated conditions | ants, α, β, ρ, deposit rule | one construction per ant |
| Particle swarm | velocity update toward bests | stability only for coefficient ranges | w, c1, c2, velocity cap | one evaluation per particle |

### 7.2 Parameter defaults with the arithmetic behind them

| Parameter | Starting value | The arithmetic that produces it | Failure when wrong |
|---|---|---|---|
| T0 | 25.5 km for this instance | T0 = -Δ_ref/ln(p0) with Δ_ref the lower quartile of uphill deltas | far below the barrier scale: the chain quenches in its first basin |
| α | 0.995 | K = ln(Tend/T0)/ln(α), then K × proposals per round equals the budget | α = 0.99 with a 62k budget ends at 0.05 in 621 rounds and then idles |
| Tend | 0.05 km | small enough that exp(-5.7/0.05) underflows; no useful uphill survives | above 5 km the run never settles and the incumbent keeps moving |
| Tabu tenure | 2 to 4 for n = 12 | ⌈0.1n⌉ = 2 to ⌈0.3n⌉ = 4, widened on stagnation | tenure 1 cycles; tenure above n/2 freezes the search |
| Population | 40 for n = 12 | at least 2n, so recombination has distinct material to work with | population 20 was stuck at 62.1 km after 60 generations |
| Mutation rate | 0.3 per child | high enough that diversity recovers within a generation | 0.05 needed 46 generations versus 22 for 0.3 |
| Tournament size | 3 | wins with probability 1-((r-1)/N)^k; k=3 gives rank 5 a 0.936 chance | k = N collapses diversity in one generation |
| ACO β | 2 | eta^β gives 0.663/0.286/0.051 across three candidates; β=5 gives 0.890/0.109/0.001 | β too high makes the colony deterministic and it never revisits |
| PSO inertia | 0.729 | with c1 = c2 = 1.49445 the root modulus is 0.8538, stable | w = 1.2 gives modulus 1.0954 and divergence |

### 7.3 Selection rule by situation

| Situation | Choose | Because |
|---|---|---|
| Model is linear with a solvable relaxation | not this sheet | a bound and a proof are available; a heuristic trades them away for nothing |
| Neighbourhood is small and descent saturates | iterated local search from a good start | the kick escapes basins at a cost of one descent each |
| Objective is a black-box simulator with second-scale runs | annealing or late acceptance | every method needs only objective values; annealing's budget is easiest to state |
| Constraint satisfaction dominates, feasibility is rare | repair operators or a feasibility-preserving encoding, not a penalty | λ that dominates the violation price can still land infeasible from some starts |
| Need to report uncertainty | replicated runs, never one seed | the standard error comes from spread across seeds, and 30 runs is the working minimum |
| Solution must be defensible to an auditor | run the heuristic, then compute a bound and report the gap | only the gap is a certified statement; the heuristic supplies a point, not a proof |
| Problem size grows by an order of magnitude | re-derive the neighbourhood, do not retune | a full-neighbourhood scan at n = 5,000 costs 12.5 million evaluations per sweep |

### 7.4 What to put in the report

| Field | Example from this sheet | Why it is not optional |
|---|---|---|
| Instance identity and hash | 12-point grid instance, coordinates printed in 5a | the number means nothing without the instance |
| Method and move set | 2-opt, steepest descent, 54 moves | different move sets are different algorithms |
| Budget in evaluations | 62,250 proposals | comparisons at unequal budgets are not comparisons |
| Seeds and replicates | 30 seeds, 30 of 30 at the optimum | one run cannot distinguish a method from its luck |
| Comparator | nearest neighbour 67.2, 2-opt 62.1, optimum 53.4 | improvement over what, exactly |
| Bound and gap | one-tree 48.81, gap 9.42 percent against the bound | the only certified quality statement available |
| Wall clock and hardware | stated per run, not aggregated | a budget in seconds is not portable across machines |

## 8. Failure Modes & Edge Cases

Each entry names the symptom as a user would see it, the causes that produce that same symptom, and
the check that separates them. The check column is what makes the entry usable: without it, every
diagnosis degenerates into "retune and hope".

### 8.1 The run reports a worse answer than a simple construction

**Symptom.** A 40-hour annealing experiment returns 71 km where nearest neighbour returned 67 km, and
the team concludes the method does not work on their data.

**Competing causes.** (a) The incumbent is being overwritten by the current state, so the reported
number is a random-walk sample rather than the best seen; (b) the starting temperature is so high
that the whole schedule is spent at acceptance rates near one and the final phase is too short to
descend; (c) the move implementation occasionally produces an invalid tour and the objective is being
evaluated on garbage; (d) the construction's tour is genuinely in a better basin and the search never
visits it.

**Discriminating check.** Log the incumbent and the current value separately from the first iteration.
If the logged incumbent never beats the construction while the current value wanders, cause (a) is
proven. If the incumbent improves only in the last 5 percent of rounds, cause (b) applies. If any
reported tour has a repeated or missing stop, cause (c) is proven by a set comparison against the
identity permutation.

### 8.2 Identical results from every seed

**Symptom.** Thirty seeds produce the same objective to four decimal places, and the reviewer
suspects the seeds are not being used.

**Competing causes.** (a) The instance is small enough to saturate, as the twelve-point instance does
above 2,000 proposals; (b) the seed is read once outside the run loop, so every replicate replays the
same stream; (c) the search is deterministic because the move set or the acceptance rule has no
random component — steepest descent with a fixed tie-break is exactly this; (d) the seed is used but
the incumbent is inherited across replicates through a shared archive.

**Discriminating check.** Print the first five random numbers drawn in each replicate; if they match
across replicates, cause (b) or (d). Then run the same seed twice at two different budgets: saturation
shows as convergence at both, whereas a shared stream shows identical trajectories. If the method is
steepest descent, cause (c) is the explanation and zero variance is correct rather than suspicious.

### 8.3 The search stalls at a value it can describe but not improve

**Symptom.** The incumbent stops improving for tens of thousands of iterations while the acceptance
rate remains plausible, and the final answer is 16 percent above a known good value.

**Competing causes.** (a) The move set cannot express the needed change in one step, so the barrier is
infinite rather than merely high; (b) the temperature has fallen below the barrier, so the chain is
effectively frozen; (c) the search cycles deterministically between a handful of states because the
move set is invertible and the acceptance of uphill moves has stopped; (d) the incumbent is a local
optimum of a *different* neighbourhood than the one being searched.

**Discriminating check.** Compute the delta distribution at the stalled incumbent and compare its
minimum uphill value against the current temperature. In the running instance the stalled 62.1 km
tour has a cheapest uphill 2-opt move of 0.10 km and a median of 10.6 km; at T = 0.05 km the small
move is accepted with probability 0.135 while the median move is accepted with probability
exp(-212) ≈ 0, so the search can creep along near-neutral moves and cannot cross the median barrier.
If the temperature is high and the stall persists, count the states visited in the last 1,000
iterations: fewer than ten distinct states indicates cause (c). To separate (a) from (d), add a
single-stop relocation move temporarily; the 62.1 km tour improves by 8.5 km under that move, which
proves the barrier was a property of the move set and not of the landscape.

### 8.4 The best solution violates a constraint that was "in the model"

**Symptom.** The reported plan exceeds capacity by a wide margin, and the person who wrote the
objective insists the constraint is present.

**Competing causes.** (a) The constraint was encoded as a penalty and λ is below the marginal value of
violation; (b) the constraint was applied to the decoded solution but the search operates on an
encoding whose intermediate states decode to infeasible tours; (c) the feasibility check uses a
tolerance wider than the operational rule; (d) the reported solution was never re-checked after the
run because the check lived only inside the objective.

**Discriminating check.** Re-evaluate the reported solution with an independent feasibility routine
that does not share code with the objective. Then sweep λ upward by factors of two and watch whether
the violation disappears at a specific λ. In the knapsack example the transition is sharp: λ = 2.0
gives weights 34, 35 and 45 against capacity 26, λ = 2.5 gives weights 25, 26 and 26, and λ = 3.0 is
already feasible from two of three starts. A sharp transition proves (a); a violation that survives
λ = 100 proves (b) or (d).

### 8.5 Results cannot be reproduced by a colleague

**Symptom.** A rerun of the documented procedure produces a different objective, and the difference is
larger than the reported confidence interval.

**Competing causes.** (a) The pseudo-random generator differs between environments, so the same seed
gives a different stream; (b) the neighbourhood is iterated in dictionary or set order, whose
iteration order is not stable across versions; (c) the budget was recorded in wall-clock seconds and
the second machine is faster or slower; (d) the objective uses a distance table rounded at a different
precision.

**Discriminating check.** Record the first five objective evaluations in the run log and compare them
across machines. Identical evaluations followed by divergence indicates (c); different evaluations
from the first step indicates (a), (b) or (d). Then pin the budget in evaluations rather than seconds
and freeze the distance precision; the twelve-point instance changes by less than 0.07 km between
0.1 km and exact distances, so a discrepancy larger than that is not a rounding artefact.

### 8.6 A population method converges to a single candidate early

**Symptom.** By generation 15 the population contains one distinct tour repeated 40 times, and the
incumbent is 15 percent above the best known value.

**Competing causes.** (a) Selection pressure is too high, for example tournament size equal to the
population; (b) mutation probability is so low that diversity cannot be restored after selection; (c)
the population was initialised from one constructive heuristic with small perturbations, so it began
with almost no diversity; (d) elitism copies the best individual many times by accident.

**Discriminating check.** Plot distinct-candidate count per generation against incumbent value. A
collapse at generation 1 indicates (c); a gradual decay that accelerates indicates (a) or (b); a
sudden drop to exactly the elitism count indicates (d). Then recompute the selection pressure: at
N = 10 and k = 3 the rank-5 individual wins 0.936 of its tournaments, so pressure is already high and
raising k further removes the bottom half from the breeding pool within two generations.

### 8.7 The same code performs well on a benchmark and badly in production

**Symptom.** A method tuned on a public instance set returns solutions 20 percent worse on live data
of similar size.

**Competing causes.** (a) The live instances have a different cost structure, for instance asymmetric
travel times where the benchmark was symmetric, so the 2-opt delta formula is simply wrong; (b) the
live instances have side constraints that the benchmark lacks, so many moves produce infeasible
candidates and the effective neighbourhood shrinks; (c) the production objective is noisier, so
apparent improvements are within the noise; (d) the benchmark's budget was larger in evaluations than
the production window allows.

**Discriminating check.** Recompute the delta formula against full re-evaluation on live data. For a
symmetric instance they agree to rounding, as §5e shows to 0.001 km; a systematic mismatch proves (a)
because in an asymmetric instance the reversed segment costs differently in the two directions. Then
audit how many proposed moves are rejected as infeasible: a rejection rate above 50 percent proves
(b), and re-running the same instance twice at the production budget separates (c) from (d) because
noise shows as spread across seeds while a budget shortfall shows as a consistent shortfall.

### 8.8 The stopping rule fires before the method has done anything

**Symptom.** The run terminates in a fraction of the intended time and returns the construction.

**Competing causes.** (a) A stagnation counter compares the incumbent to itself because of a shared
reference; (b) the wall-clock deadline was set from a pilot that ran on a warm cache; (c) the schedule
ends at Tend in 621 rounds instead of 6,233 because α was copied from a different endpoint pair; (d)
an evaluation cap counts moves rather than objective calls, so the run ends after one sweep.

**Discriminating check.** Print rounds completed, evaluations used and final temperature. A final
temperature far above Tend proves (c) by arithmetic: with T0 = 25.544 and α = 0.99 the schedule needs
ln(0.05/25.544)/ln(0.99) = 621 rounds, so a run reporting 621 rounds has not been truncated but has
simply been specified with a fast schedule. A final temperature at Tend with a tiny evaluation count
proves (d): 621 rounds at 50 proposals each is 31,050 proposals, and anything far below that is a
counting bug.

### 8.9 The method's parameters were tuned on the test instance

**Symptom.** Reported performance collapses when the tuned configuration is applied to a fresh
instance from the same family; the team attributes the drop to "overfitting the solver".

**Competing causes.** (a) Genuine parameter overfitting to one instance's delta scale; (b) the fresh
instance has a different scale, so a temperature calibrated in kilometres is wrong by an order of
magnitude; (c) the fresh instance is harder in a structural sense and the neighbourhood, not the
parameters, is the binding constraint; (d) the fresh evaluation uses a different budget.

**Discriminating check.** Recompute the delta distribution on the fresh instance and recalibrate T0
from its own quantiles. If performance recovers, cause (b) is proven and the fix is to calibrate per
instance, which is cheap: the calibration needs one sweep of move evaluation, 54 evaluations here.
If performance does not recover, test a richer neighbourhood before touching any parameter, since a
structural constraint cannot be tuned away.

## 9. Complexity & Cost Analysis

**Neighbourhood size drives everything.** For the 2-opt move on a cycle of n stops the number of
distinct moves is n(n-3)/2, which is 54 at n = 12, 1,710 at n = 60, 124,250 at n = 500,
12,492,500 at n = 5,000 and 1,249,925,000 at n = 50,000. A full steepest-descent sweep therefore costs
quadratic work, and at n = 50,000 one sweep is more evaluations than any practical run can afford,
which is why production routing codes use candidate lists that restrict each city to its nearest 10
to 20 neighbours and accept that the resulting local optimum is relative to a reduced neighbourhood.

**Memory is the second wall.** A dense distance matrix of n points at 8 bytes per entry is n²×8 bytes:
0.0 MB at n = 12, 0.0 MB at n = 60, 1.9 MB at n = 500, 190.7 MB at n = 5,000 and 19,073.5 MB at
n = 50,000. The last figure is about 18.6 GB, which is why large instances store coordinates and
compute distances on demand, trading 4 floating-point operations for 8 bytes of cache pressure.

**Move evaluation versus full evaluation.** In the twelve-point instance a full tour length costs 12
distance lookups, while a delta costs 4. At 54 moves per sweep the two strategies cost 648 and 216
lookups respectively, so the delta is a threefold saving before any algorithmic benefit. On the
sixty-point instance, the 2-opt descent found 612.1 km in 11 improving moves after 20,520 move
evaluations; at 4 lookups each that is 82,080 lookups, against 1.59 million evaluations for the ILS
comparison, which is 19.4 times more work for a mean improvement from 656.4 km to 623.9 km starting
from a random tour.

**Evaluation budget converted to wall clock.** At a nominal 1e6 objective calls per second on one
core, 62,250 proposals is 0.062 seconds, 200,000 is 0.20 seconds and 1.59 million is 1.59 seconds.
Those figures are arithmetic on a stated rate, not a measurement; the point of stating the rate is
that it makes the budget portable. A run described as "40 minutes" cannot be compared with a run
described as "20 minutes" on another machine, while two runs described in evaluations can.

**Cost per unit of quality.** The 2-opt descent gained 5.1 km with 54 move evaluations, or 0.094 km
per evaluation. Annealing on the same instance gained 67.2 → 53.4 km, that is 13.8 km, over about
2,000 proposals, or 0.0069 km per evaluation. The descent is roughly 14 times more efficient per
evaluation on this instance and cannot reach the optimum at all; the annealing is inefficient per
evaluation and does reach it. Both statements are true and the pair is the honest characterisation of
the trade.

**Where the annealing budget actually went.** Of the 1,245 rounds in the reported run, 401 were
needed to reach 53.4 km and 844 produced nothing. At 50 proposals per round the productive phase cost
20,050 evaluations and the idle phase 42,200, so 68 percent of the budget bought no improvement. That
ratio, not the total budget, is the first thing to fix: truncating at 450 rounds would have returned
the same tour at 36 percent of the cost.

**Replication cost.** The standard error of a mean over R independent runs is s/sqrt(R) with s the
across-seed standard deviation. On the sixty-point instance, s = 27.21 km, so a target standard error
of 6 km needs R = (27.21/6)² = 21 runs, 3 km needs 83 runs, and 1.5 km needs 330 runs. At 200,000
evaluations per run those are 4.2, 16.6 and 66 million evaluations respectively. This is the dominant
cost in any serious study and it is routinely omitted from reports that quote a single "best of five".

**Diminishing returns.** On the twelve-point instance, budgets of 500, 1,000, 2,000, 4,000, 8,000 and
16,000 proposals give hit rates of 28/30, 29/30, 30/30, 30/30, 30/30 and 30/30. The marginal value of
the eighth doubling is zero, and a report that raises the budget from 2,000 to 16,000 and shows the
same answer has demonstrated nothing except that the budget was already sufficient.

**Parallelism.** Independent seeded runs parallelise with no communication, so R restarts on P cores
take ceil(R/P) run-times. A single long chain cannot be parallelised without changing the algorithm's
semantics, since the chain's next state depends on the current one; parallel tempering exists precisely
to work around that and pays for it with a swap acceptance rule that must itself be calibrated.

## 10. Key Theorems / Results Reference

**Metropolis et al. (1953), the acceptance rule.** The rule named in §4.1 originated as a sampling
method for equilibrium distributions in statistical physics. Its relevance here is that it converts an
optimisation objective into a probability distribution whose mode is the global minimum, so that
"sampling well" and "optimising" become the same activity at a fixed temperature.

**Detailed balance as a sufficient condition.** If π(x)P(x,y) = π(y)P(y,x) for all x, y, then π is
stationary for P. Combined with irreducibility and aperiodicity on a finite space, stationarity is
unique and the chain converges to it from any start. §5c verifies this numerically with a residual of
1.4e-17, which is the standard of evidence an implementation review should demand.

**Hastings (1970), the correction for asymmetric proposals.** The acceptance ratio is multiplied by
q(y,x)/q(x,y). Its practical content is that you may use any proposal you like, including one biased
toward promising moves, without breaking the target distribution — provided the correction is applied
in both branches of the minimum. §5c measures the cost of omitting it: a 37 percent inflation of the
probability of the middle-cost state.

**Kirkpatrick, Gelatt and Vecchi (1983), annealing as optimisation.** Cooling a Metropolis sampler
turns a sampler into an optimiser. The result is a heuristic with a convergence claim only under an
infinite-time schedule; any finite schedule, including every schedule in this sheet, has no optimality
guarantee.

**Hajek's convergence condition.** For a logarithmically cooled chain, convergence in probability to
the set of global optima requires that the cooling constant exceed the depth of the deepest local
minimum that is not global, measured relative to the optimum. The operational reading is blunt: a
barrier deeper than your cooling constant is never crossed with probability approaching one, and the
log schedule's constant is a property of the landscape you must estimate, not a free parameter.

**Wolpert and Macready (1997), no free lunch.** Averaged over all objective functions, every search
algorithm has the same expected performance. The usable content is negative — no method is
universally best — and the misreading to avoid is that method choice does not matter for a specific
problem family, which it plainly does.

**Clerc and Kennedy (2002), particle swarm stability.** The velocity recursion's behaviour follows from
the roots of λ² - (1 + w - φ)λ + w with φ = c1 + c2; the standard coefficient set gives a root modulus
of 0.8538 at w = 0.729 and 0.9487 at w = 0.9, both stable, while w = 1.2 gives 1.0954 and divergence.
This is one of the few exact results available for a population method and it should be checked before
any swarm run.

**Stützle and Dorigo, ant colony convergence.** Under stated conditions on the pheromone update, an
ant-colony method converges to a tour in the limit of infinite iterations, with probability one under
some variants. As with annealing, the guarantee is asymptotic and says nothing about a 500-ant budget.

**Held and Karp (1970, 1971), the one-tree bound.** A minimum spanning tree plus the two cheapest edges
at a distinguished node gives a lower bound on the optimal tour, and with subgradient ascent on node
penalties the bound is tight enough to prove optimality on instances far larger than enumeration
allows. In this sheet's instance the bound is 48.8063 km against a tour of 53.4056 km, so the tour is
at most 9.42 percent above the optimum — the certified ceiling that makes the heuristic's number
meaningful.

**Glover's tabu memory claim.** Short-term memory of recently reversed attributes prevents cycling in
deterministic local search without requiring randomisation; the aspiration criterion allows a tabu move
when it improves the incumbent, which restores the possibility of returning to a good region. The
claim is a design principle rather than a theorem, and the tenure range is empirical.

**Schema theorem and its limits.** The classical genetic-algorithm analysis shows that short,
low-order, above-average schemata receive exponentially growing representation. It is a statement
about expected copy counts under a particular selection and crossover model, not about convergence to
the optimum, and it does not predict performance on a permutation problem with a combinatorial
neighbourhood. Treat it as motivation for maintaining diversity, not as a guarantee.

**The gap statement that matters operationally.** Without a bound, the only honest quality claim is
comparative: "this method reached 53.4 km where the construction reached 67.2 km and a 2-opt descent
reached 62.1 km, at these evaluation counts, over these seeds". With a bound, the claim becomes
absolute: "the result is at most 9.42 percent above the optimum". The second claim is what an auditor
can accept, and computing it costs one MST, which on twelve points is a by-hand calculation.

## 11. Interview / Exam Questions

**Q1. You have 10,000 objective evaluations and a 60-city instance. Your 2-opt descent improves the
nearest-neighbour tour from 741.5 to 612.1 km in 20,520 move evaluations. Is the descent inside
budget, and what does that tell you?**
Answer: no, it is outside. At 60 cities there are 60×57/2 = 1,710 candidate moves, so one full sweep
is 1,710 evaluations and 20,520 evaluations is twelve sweeps; 10,000 evaluations is under six sweeps,
so the descent would stop partway. The practical consequence is that a budget smaller than the
descent's requirement must be spent differently — on a restricted candidate list of the nearest 15
cities per stop, which cuts each sweep to roughly 900 candidate moves.
Reasoning: the count comes from the neighbourhood size formula, not from timing.
Follow-up: what is the largest instance for which a full 2-opt sweep fits in 10,000 evaluations? Solve
n(n-3)/2 ≤ 10,000: n = 142 gives 9,869 evaluations and n = 143 gives 10,010, so the answer is 142.

**Q2. Compute the starting temperature that accepts a 5.7 km uphill move 80 percent of the time.**
Answer: T0 = -5.7/ln(0.8) = 25.54 km. The unit is kilometres because the objective is a distance.
Reasoning: solve exp(-Δ/T) = 0.8 for T, which gives T = -Δ/ln(0.8) = Δ/0.22314.
Follow-up: what if the instance's distances were in metres? The temperature becomes 25,544 m and every
delta scales by 1,000, so the acceptance probabilities are unchanged; a temperature that is not scaled
with the objective's unit is meaningless.

**Q3. Your annealing run reports 71 km on an instance where the nearest-neighbour tour is 67 km. Give
the three most likely causes in order and one check each.**
Answer: first, the incumbent is not tracked separately from the current state, checked by logging both
from iteration one; second, the schedule spent its low-temperature phase too briefly, checked by
comparing the round at which Tend was reached against the total rounds; third, the move implementation
produced invalid tours, checked by verifying that each reported tour is a permutation of the stops.
Reasoning: these are ordered by frequency in practice, and each has a one-line check that does not
require rerunning the experiment.
Follow-up: which of the three would improve with a longer run? Only the second, because the first and
third produce wrong numbers at every budget.

**Q4. Why does the Metropolis rule need a Hastings correction when the proposal is "pick a random
neighbour of the current state"?**
Answer: because that proposal is not symmetric when the neighbourhood graph has unequal degrees: from
a boundary state with one neighbour you propose that neighbour with probability 1, while the neighbour
proposes you with probability 0.5. The correction multiplies the acceptance by q(y,x)/q(x,y), which is
2 in that direction, restoring detailed balance. Without it the stationary weights are proportional to
exp(-f/T) times the degree, and §5c measures the resulting distortion as a 37 percent inflation of the
middle state.
Reasoning: detailed balance requires that probability flux in both directions match, and the proposal
asymmetry appears on both sides of the equation.
Follow-up: on a path graph the uncorrected chain puts 0.5109 on the middle state instead of 0.3724,
while the corrected chain reproduces 0.3724 to four decimals.

**Q5. A tabu search on 12 stops uses a tenure of 20. What goes wrong?**
Answer: the tenure exceeds the number of moves that can meaningfully be tabu, so the search spends most
iterations forbidden from undoing any recent change and effectively freezes, or it oscillates among the
few non-tabu moves that remain. The rule of thumb is ⌈0.1n⌉ to ⌈0.3n⌉, which is two to four iterations
here, widened only on detected stagnation.
Reasoning: tenure is a memory length measured in iterations; a memory longer than the cycle it is meant
to break removes the method's ability to move.
Follow-up: how would you detect the freeze? Count distinct incumbent values over the last 200
iterations; fewer than three distinct values is the signature.

**Q6. Distinguish a local optimum from a global optimum using only quantities you can compute in one
sweep.**
Answer: a local optimum is a point with no improving neighbour in the declared move set, verifiable by
one sweep costing n(n-3)/2 move evaluations (54 here). A global optimum cannot be verified by any number
of sweeps; it requires either exhaustive search or a bound. In the running instance the 62.1 km tour has
no improving 2-opt move, with a minimum delta of +0.10 km, and the optimum is 53.4 km — the local check
passes on a point that is 16.3 percent worse than optimal.
Reasoning: "no improving neighbour" is a statement about the neighbourhood; "best of all tours" is a
statement about the whole space.
Follow-up: what is the cheapest certificate that 62.1 km is not optimal? A single exhibited tour of
53.4 km, which costs one evaluation once found.

**Q7. Your penalty weight is 2.0 and every returned solution violates capacity by 8 to 19 units. What
is the arithmetic of the fix?**
Answer: λ must exceed the marginal value per unit of violation. In the knapsack example the feasible
optimum has value 56 at weight 26, while the infeasible weight-45 set has value 95: the extra 19 units
of weight bought 39 units of value, about 2.05 value per unit. λ = 2.0 is just below that price, so the
optimum of the penalised objective is infeasible. Raising λ to 2.5 flips the ordering and the search
returns feasible sets of value 54, 56 and 55 depending on the start.
Reasoning: the penalised objective is a Lagrangian-style surrogate, and its minimiser is feasible only
when the multiplier dominates the constraint's shadow price.
Follow-up: why does λ = 3.0 still return a weight-28 set from the empty start? Because its penalised
value is -60 + 3×2 = -54, better than the feasible 56 at -56, and the one-flip neighbourhood from there
contains no improving feasible move.

**Q8. Explain why 30 seeds reaching the optimum tells you less than you think on a 12-stop instance.**
Answer: with 54 moves per sweep and 62,250 proposals, each run performs about 1,153 sweeps over the
whole neighbourhood, so the instance is saturated and the hit rate is 1 by construction rather than by
search quality. The same design at 60 stops averaged 2.24 percent above the best known value, which is
where differences between schedules appear. Reporting "30 of 30" on the easy instance and implying
method quality is an instance-selection artefact.
Reasoning: a ceiling effect hides every difference; a benchmark must be near the method's capability
limit to discriminate.
Follow-up: what change makes the small instance informative? Reduce the budget until the hit rate drops
below 1 — at 500 proposals it is 28 of 30 — so the measurement sits on the steep part of the curve.

**Q9. You must compare annealing and iterated local search. State the two errors that would invalidate
the comparison.**
Answer: unequal budgets, and unequal starting points. Annealing at 200,000 evaluations gave 625.8 km
mean while ILS at 1.59 million averaged 623.9 km; at a matched 800,000 evaluations ILS averages 627.3
km against annealing's 628.4 km, a difference of about 1 km that is one tenth of the across-seed
spread, so at equal budget the two are indistinguishable and the earlier comparison was measuring
budget. The start matters independently: ILS from the nearest-neighbour 2-opt optimum reached 612.1 km
in 20 of 20 seeds, while ILS from a random tour reached it in 2 of 20 at the same kick count.
Reasoning: both methods are anytime, and their quality is a function of budget and start separately.
Follow-up: what budget-matched experiment would you run next? Annealing with a 2-opt descent applied to
its incumbent at intervals, which isolates the acceptance rule from the local improvement.

**Q10. Compute the number of cooling rounds for α = 0.998 from 25.54 km to 0.05 km, and say what it
means for the budget.**
Answer: K = ln(0.05/25.544)/ln(0.998) = (-6.2362)/(-0.0020020) = 3,115 rounds. At 50 proposals per
round that is 155,750 proposals, 2.5 times the α = 0.995 budget of 62,205. The schedule is buying slower
cooling with evaluations, and there is no free lunch inside the method any more than across methods.
Reasoning: the geometric schedule's round count is a ratio of logarithms, so a finer temperature step
increases the work almost proportionally.
Follow-up: at what α does the round count exceed 10,000? ln(0.05/25.544)/ln(α) > 10,000 requires
ln(α) > -0.00062362, that is α > 0.999377.

**Q11. A colleague says "the annealing converged because the objective stopped changing". What is
wrong with the sentence, and what would you report instead?**
Answer: a constant incumbent means no improvement was found, not that the search converged; the
current state may still be moving, the acceptance rate may still be substantial, and the final
temperature may still be above every barrier. Report the acceptance rate over the last 200 iterations,
the number of distinct states visited in that window, and the final temperature. On the simulated
annealing trace in §5b the incumbent has been final since round 401 while the current tour only
collapsed onto it around round 800, so "the objective stopped changing" would have been a false
diagnosis for 400 rounds.
Reasoning: convergence in the technical sense is a statement about a distribution, and no finite run
demonstrates it.
Follow-up: what acceptance rate would you expect at a final temperature of 0.05 km with deltas of 0.1 to
26 km? 0.135 for the smallest move and essentially zero for anything above 0.5 km, since
exp(-0.5/0.05) = 4.5e-5.

**Q12. You are told the true optimum for a new 80-city instance is 612.1 km and your heuristic returned
618.0 km. Is that good, and what would you need to say it without the optimum?**
Answer: it is a 0.96 percent gap, which is good by routing standards. Without the known optimum you
would compute a lower bound — an MST-based one-tree bound with subgradient ascent on node penalties is
the standard cheap choice — and report the gap to that bound, which is always non-negative and
therefore conservative. The MST alone on the 12-point instance gives 43.2117 km against 53.4056 km, a
23.59 percent gap that is too loose to be informative; the one-tree refinement tightens it to 48.8063
km and a 9.42 percent gap, which is usable.
Reasoning: a bound is a certified statement about all tours, so the gap it produces is defensible
without knowing the optimum.
Follow-up: how many points does the subgradient step need? Each step costs one MST, which is O(n² log n)
with a dense implementation, so a few hundred steps is affordable at n = 80 but not at n = 50,000.

**Q13. Two runs of the same configuration differ by 40 km. Is the configuration unstable?**
Answer: not necessarily; it may simply be that the across-seed standard deviation is about 27 km, as
measured over 20 seeds on the 60-point instance, in which case a 40 km difference between two runs is
about 1.5 standard deviations and entirely ordinary. The configuration is unstable only if the
difference survives replication: measure the standard deviation over at least 30 seeds, then check
whether the reported effect exceeds 1.96 s/sqrt(R).
Reasoning: seed-to-seed variation is the dominant noise source, and single-run comparisons are samples
of size one from a wide distribution.
Follow-up: how many runs give a 95 percent half-width of 3 km with s = 27.21? R = (1.96 × 27.21/3)² =
316, which is why serious tuning studies are expensive.

**Q14. Why is a double-bridge kick preferred to swapping two random cities in iterated local search?**
Answer: because a single 2-opt move can largely undo a random swap, so the descent following the
perturbation walks straight back into the basin just left; the double-bridge rearrangement cannot be
undone by one 2-opt move, so the descent lands elsewhere. The mechanism is combinatorial: the kick
changes three adjacencies in a way that no single interval reversal restores.
Reasoning: a perturbation is useful only if it moves the starting point outside the basin of the
current local optimum.
Follow-up: on the 60-point instance ILS with 200 kicks moved the mean from 656.4 to 623.9 km, a 4.95
percent gain over random-restart descent; what does that say about the basins? That the good basin is
reachable by a few kicks from mediocre ones, but rarely by a random start.

**Q15. Your swarm's particles leave the feasible box and come back clamped, and the best objective
oscillates. Diagnose it with arithmetic.**
Answer: check the velocity recursion's roots. With w = 1.2 and c1 = c2 = 1.49445 the characteristic
polynomial λ² - (1 + 1.2 - 2.989)λ + 1.2 has complex roots of modulus sqrt(1.2) = 1.0954, so velocity
grows by 9.5 percent per iteration and the clamp is doing all the stabilisation. With w = 0.729 the
modulus is 0.8538 and 200 iterations shrink a deviation to 1.9e-14. Setting w to 0.729 or capping the
velocity at a stated fraction of the box width both fix it, and the first is preferable because it
removes the cause rather than the symptom.
Reasoning: the swarm is a linear dynamical system in the absence of best-position changes, so stability
is an eigenvalue question.
Follow-up: what does the clamp do to the optima? It projects all divergence onto the boundary, so the
swarm reports boundary solutions that are artefacts of the clamp, not of the objective.

## 12. Quick Reference Card

| Decision | Rule | Number to check |
|---|---|---|
| Is a metaheuristic justified? | only after an exact method's gate is closed with a reason | bound available? if yes, prefer exact and report the gap |
| Which family? | descent saturates → ILS; black-box simulator → annealing; feasibility-dominated → repair, not penalty | neighbourhood size n(n-3)/2 |
| Starting temperature | T0 = -Δ_ref/ln(p0) from measured uphill deltas | 25.54 km accepts a 5.7 km move at 0.80 |
| Cooling rate | K = ln(Tend/T0)/ln(α), then match proposals per round to the budget | α = 0.995 → 1,245 rounds; α = 0.999 → 6,233 |
| Budget | in objective evaluations, identical across compared methods | 62,250 proposals, of which 844 rounds added nothing |
| Replication | R = (s/SE)² runs, at least 30 for a reportable mean | s = 27.21 → 83 runs for SE = 3 |
| Quality claim | gap to a bound, or difference to a stated comparator at equal budget | one-tree bound 48.81, gap 9.42 percent |
| What to log | incumbent and current separately, acceptance rate, distinct states, final T | acceptance 0.247, final T 0.05 km |
| What never to say | "converged" without a distribution, "optimal" without a certificate | a 2-opt certificate passes on a 16.3 percent gap |
