---
title: "Network and Scheduling Models"
topic: "Network flow, project networks, and machine scheduling"
collection: "Part 03 — Machine Learning, Optimisation and Intelligent Retrieval"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Network and Scheduling Models

> **Evidence boundary.** Every number in this sheet is either derived here from stated axioms or explicitly labelled `ILLUSTRATIVE` as a constructed teaching input. No deployed system, vendor benchmark, or private operational result is claimed. Where a classical result is named but not proved, the sheet says so in §10 and asks the reader to check the edition they hold.

## 1. File Header

**Purpose.** Give one practitioner the ability to look at an operational question, decide whether it is a *flow* question or a *sequencing* question, write the corresponding model, read its certificate, and refuse the model when its hypotheses are false.
**Primary audience.** Operations analysts, OR-literate data scientists, and engineers who own a planning service and must defend its output to a scheduler.
**Prerequisites.** Linear programming at the level of "write the dual and read it"; graph vocabulary (path, cut, cycle); the distinction between an event time and a processing time; enough arithmetic discipline to carry units through a Gantt chart.
**Outcome 1.** Given a capacity network, produce the maximum throughput *and* a cut whose capacity equals it, so the number arrives with a proof rather than a belief.
**Outcome 2.** Given a project with precedence constraints, compute early and late start times, name every critical path, and price the cheapest way to shorten the project by one unit.
**Outcome 3.** Given a machine and a set of jobs, choose the sequencing rule that matches the declared objective, and state which objective it does *not* optimise.
**Outcome 4.** Recognise the four situations in which these models are structurally invalid — multi-commodity flow, non-additive cost, non-identical machines, and stochastic durations presented as certainties — and say what to do instead.
**Exclusions.** This sheet does not cover stochastic scheduling (queueing, simulation-optimisation), constraint programming as a modelling language, or the design of a solver. It states results and their hypotheses; it does not implement them.
**Units convention.** Throughput in items per period; capacity in the same unit as throughput; cost in currency units per unit of flow or per day of crash; durations in the same time unit as due dates, with the unit written in the model record. Every table that mixes units names them in the column header.
**Notation.** `G = (V, A)` is a directed graph with node set `V` and arc set `A`; `u_a` is arc capacity; `f_a` is arc flow; a cut is written `(S, T)` with `s ∈ S`, `t ∈ T`. Scheduling uses `p_j` (processing time), `r_j` (release date), `d_j` (due date), `w_j` (weight), `C_j` (completion time), `L_j = C_j - d_j` (lateness), `T_j = max(L_j, 0)` (tardiness), `U_j` (unit-late indicator), and `C_max` for makespan.
**How to read the sheet.** §4 carries the derivations and is the part that makes the rest auditable. §5 is a single worked instance per model family, each with an independent re-check. §8 is the list to consult when a model passes its tests and still produces a plan the floor rejects.
**Version note.** First substantive version. The predecessor draft of this file was rejected by the acceptance lint for repeated prose and absent headings, so nothing here inherits from it.

## 2. Concept Map / Mental Model

```text
                     one question: what limits throughput, and in what order?
                                        |
              +-------------------------+-------------------------+
              |                                                   |
        NETWORK SIDE                                          TIME SIDE
   conservation of flow on arcs                    resource occupancy over time
              |                                                   |
   max flow / min cut  --- certificate --->  bottleneck set        |  Gantt chart, makespan
   min-cost flow       --- certificate --->  potentials / duals   |  EDD, SPT, Johnson, Moore
   transportation      --- certificate --->  row/col duals        |  CPM / PERT network
   assignment          --- certificate --->  permutation bound     |  crashing = min cut
              |                                                   |
              +--------------------- shared machinery -------------+
                          LP duality, integral polytopes,
                          longest path in a DAG, exchange arguments
```

**The single idea that makes both halves one subject.** A schedule is a flow problem in which time is the conserved quantity and a machine is an arc with capacity 1. A flow problem is a schedule problem in which each unit of flow is a job that must traverse arcs in order. Once you see that, the shared tool is not a coincidence: it is the same LP with different units.

**Mental model 1 — flow as routing through shared capacity.** Think of the network as pipes and the question as "how much can leave the source". You never need to understand the whole graph to answer it; you need one augmenting path at a time and a stopping condition.

**Mental model 2 — the cut as a bottleneck certificate.** The value of the flow is bounded above by the capacity of *any* partition separating source from sink. So a single partition with capacity 12 proves no routing can exceed 12 — and a flow of 12 proves nothing better exists. Number plus witness, both checkable in linear time.

**Mental model 3 — the Gantt chart as a packing problem.** Each job is a rectangle whose width is its processing time; the machine is a strip of height one; the makespan is the right edge of the last rectangle. Sequencing rules are heuristics or theorems about how to order rectangles in that strip.

**Mental model 4 — slack as the budget for delay.** In a project network, the critical path is the longest path; every other activity can slip by its slack without moving the completion date. When you spend money to shorten the project, you are buying down slack until a second path becomes critical.

**Where the two halves disagree, and why that matters.** Flow models are *polynomially solvable and integral*: the LP optimum is an integer flow, so you get a certificate. Machine scheduling with more than one machine and a makespan objective is *NP-hard*, so you get a bound instead of a certificate. Confusing the two — quoting `C_max` from a heuristic as if it were proved optimal — is the error this sheet spends most of §8 on.

**The four questions to ask before choosing a model.**

| Question | If yes | If no |
|---|---|---|
| Does a unit of work keep its identity as it traverses the system? | single-commodity flow is admissible | multi-commodity flow; the integrality and min-cut results do not apply |
| Is the objective a sum or a maximum of additive quantities? | LP-based formulations are exact | check additivity; sequence-dependent cost breaks the recurrence |
| Are the resources interchangeable? | `k`-machine results and greedy load balancing are available | eligibility-constrained assignment; state must record which resource holds which job |
| Are durations known at decision time? | deterministic scheduling models apply | the output is a scenario or an expectation; label it as such |

**What this map does not give you.** It does not tell you whether the business question is worth modelling. A network with 40 nodes and a stable weekly pattern is usually better served by a spreadsheet and a human; the model earns its keep when the instance is large, changes frequently, or when someone demands a proof of optimality that a human cannot supply.

**The one invariant to carry through the whole sheet.** For every model here, the answer must come with either a witness (a flow, a schedule, a cut, a dual solution) or a bound, and the witness must be verifiable without re-running the optimiser. A number with no witness is a claim.

## 3. Foundational Definitions

Definitions are given with units and valid ranges because almost every production defect in this area is a units or direction error, not an algorithmic one.

### 3.1 Network vocabulary

- **Directed graph `G = (V, A)`.** `V` is a finite node set; `A ⊆ V × V` is a set of ordered pairs. Direction matters: the arc `(a,b)` is not the arc `(b,a)`, and reversing one in a data load silently changes every downstream answer.
- **Capacity `u(a,b)`.** A non-negative real, in the unit of throughput (items per period, litres per hour, currency per day). `u = 0` means the arc may be omitted but MUST NOT be silently merged with a parallel arc of positive capacity.
- **Flow `f(a,b)`.** A real in the same unit as capacity, obeying `0 ≤ f ≤ u`. In a shipping model it is a quantity per period; in an assignment model it is 0 or 1 and the unit is "assignment".
- **Conservation.** For every node except source and sink, total inflow equals total outflow. This is the single constraint that makes the model a network model rather than a set of independent arcs.
- **Source `s` and sink `t`.** The only nodes allowed to create and destroy flow. A model with several sources is converted by adding a super-source with capacity equal to each supplier's availability.
- **Cut `(S, T)`.** A partition of `V` with `s ∈ S` and `t ∈ T`. Its capacity is the sum of capacities of arcs from `S` to `T`; arcs from `T` to `S` are not counted, which is the most frequently mis-implemented line in student code.
- **Residual capacity.** For arc `(u,v)` carrying `f`: `u - f` forward, and `f` backward. The backward residual arc is the formal permission to undo a decision, and it is why augmentation never gets trapped.
- **Augmenting path.** An `s`-to-`t` path in the residual graph. If one exists, the current flow is not maximum. If none exists, it is.
- **Bottleneck `δ`.** The minimum residual capacity on an augmenting path; the flow value rises by exactly `δ` when it is used.
- **Min-cost flow.** Flow of a required value `F` minimising `Σ c_a f_a`, with `c_a` in currency per unit of flow. Existence requires enough total capacity; uniqueness of the optimal *flow vector* does not hold in general even when the optimal *cost* is unique.
- **Potentials `π(v)`.** Node labels, in currency per unit, used to test optimality via reduced costs. They are the network analogue of LP dual variables and, read operationally, they price the marginal cost of capacity at each node.
- **Bipartite graph.** Nodes split into two sides with arcs only across. Assignment, matching, and many rostering models are bipartite by construction, and that structure is what makes their LP integral.
- **Matching.** A set of arcs no two of which share a node. **Cover.** A set of nodes touching every arc. In bipartite graphs the maximum matching size equals the minimum cover size — a special case of max-flow min-cut.
- **Multi-commodity flow.** Several commodities share arcs, each with its own origin-destination pair. The LP is still solvable but its optimum is generally fractional, the min-cut certificate disappears, and the problem is much harder.

### 3.2 Scheduling vocabulary

- **Job `j`.** The unit of work, identified by a stable key. If the key is not stable across runs, no reproducibility claim survives, because tie-breaking will not be reproducible either.
- **Processing time `p_j`.** The resource-time the job consumes, in the calendar's time unit, measured on the resource that will do the work and *including* setup and cleanup if those are real.
- **Release date `r_j`.** The earliest instant the job may start. Omitting it converts a hard real constraint into an implicit assumption that everything is available at time zero.
- **Due date `d_j`.** The contractual completion instant. Distinct from a *target* date, which may be soft; conflating the two produces a schedule that looks compliant and breaches a contract.
- **Weight `w_j`.** The per-unit-time penalty or value attaching to job `j`'s completion. Weights convert a count objective into a business objective and are the usual reason two teams disagree about which schedule is better.
- **Completion time `C_j`.** The instant job `j` finishes. In a non-preemptive single-machine schedule it equals the sum of processing times of the jobs up to and including `j`.
- **Flow time `F_j = C_j - r_j`.** Time in system. Its sum over jobs, divided by the count, is mean time in system — the quantity a customer experiences and the quantity SPT optimises.
- **Lateness `L_j = C_j - d_j`.** Signed; negative means early. **Tardiness `T_j = max(L_j, 0)`.** The sign convention matters: maximising `min L_j` and minimising `max T_j` are different problems with different optima.
- **Unit-late indicator `U_j`.** 1 if `C_j > d_j`. Its sum counts late jobs, which is what most service-level agreements actually measure.
- **Makespan `C_max = max_j C_j`.** Completion of the last job. The objective of a project or a batch, and usually *not* the objective of a service operation, where flow time or lateness dominates.
- **Work in process (WIP).** Jobs started but not finished. A schedule that minimises makespan can leave WIP high; the two objectives can trade against each other and the trade is invisible unless both are reported.
- **Preemption.** Permission to suspend a job and resume later. If the model allows it and the process does not, the reported schedule is infeasible in the plant even though it is optimal in the model.
- **Precedence constraint `i → j`.** Job `j` may not start before `i` completes. A cycle in the precedence graph makes the instance infeasible, and cycle detection is cheap enough that skipping it is indefensible.
- **Critical path.** A longest path through the precedence network; its length is the project duration. Activities on it have zero slack, which means any delay to one delays the project one-for-one.
- **Slack `s_i = LS_i - ES_i`.** The delay an activity can absorb without moving the completion date, given the current durations. It is a *conditional* number: after any duration change, every slack must be recomputed.
- **Machine environment.** The notation `1`, `P`, `Q`, `R`, `F`, `J`, `O` distinguishes one machine, identical parallel, uniform-speed parallel, unrelated parallel, flow shop, job shop, and open shop. The first letter of a scheduling problem statement determines whether the problem is easy or NP-hard, so it is not decoration.
- **Objective notation.** The trailing field of `α|β|γ` names the objective: `C_max`, `ΣC_j`, `Σw_jC_j`, `L_max`, `ΣU_j`. Two analysts who say "optimal schedule" while meaning different `γ` are not disagreeing about facts.

### 3.3 Definitions that are commonly confused

| Term | Precise meaning | Common misuse | Consequence of the misuse |
|---|---|---|---|
| Capacity | Upper bound on flow *per period* | Used as a stock limit | Over-commitment every period |
| Bottleneck | A minimum cut, not the arc with the largest utilisation | Named as a single arc | Investment goes to the wrong arc |
| Critical path | Longest path, may be several | The chain of "important" tasks | Crashing buys nothing |
| Slack | Conditional on current durations | Treated as a permanent budget | Second path becomes critical unnoticed |
| Tardiness | `max(C - d, 0)` per job | Summed as lateness | Early jobs cancel late ones in the total |
| Makespan | Last completion | Used as a service metric | Customers wait while the batch finishes |
| Optimality | Relative to a declared `γ` | Asserted without naming `γ` | Unfalsifiable claim |

**Why the table matters more than the definitions.** Each row is a case where two teams can hold the same data and reach different conclusions without either being careless. The disagreement is definitional, and the fix is to write the symbol and its unit into the decision record before any computation starts.

## 4. Core Formulas with Full Derivation

Each substantive result below uses the five-part anatomy: formula, where it applies, the constraint that keeps it valid, a numerical example, and the interpretation. §4.4 collects the cases where the formulas break.

### 4.1 Base case — max flow, its cut certificate, and why augmentation terminates

```text
FORMULA [NET-01]: maximise  v = Σ_{a ∈ δ+(s)} f_a
                  subject to  Σ_{a ∈ δ-(v)} f_a − Σ_{a ∈ δ+(v)} f_a = 0   for all v ∉ {s,t}
                              0 ≤ f_a ≤ u_a                             for all a ∈ A
WHERE: δ+(v) = arcs leaving v, δ-(v) = arcs entering v; f and u in items per period;
       v is a scalar in the same unit.
CONSTRAINT: single commodity; capacities fixed and known at decision time;
            flow divisible in the LP but integer in the combinatorial algorithm.
NUMERICAL EXAMPLE: the six-node network of §5 has v = 12 with capacities in items/day.
INTERPRETATION: v is the throughput the network can sustain indefinitely, not a one-off
                quantity; it is limited by the tightest partition, not by the busiest arc.
```

**Claim A (weak duality of flow and cut).** For any feasible flow `f` and any `s`-`t` cut `(S,T)`, `v(f) ≤ u(S,T)`.

Proof. Sum the conservation equations over all `v ∈ S \ {s}`. Every arc internal to `S` appears once as inflow and once as outflow and cancels. What remains is `v(f) = f(S,T) - f(T,S)`. Since `f ≥ 0` and `f(S,T) ≤ u(S,T)`, we get `v(f) ≤ f(S,T) ≤ u(S,T)`. The equality `v = f(S,T) - f(T,S)` is the whole content: flow crossing backwards *reduces* the value, which is why a cut's capacity ignores `T → S` arcs but the value does not.

**Claim B (augmentation is monotone and bounded).** If the residual graph contains an `s`-`t` path with bottleneck `δ > 0`, pushing `δ` along it yields a feasible flow of value `v + δ`. Feasibility holds because every arc on the path has residual capacity at least `δ`; monotonicity holds because `δ > 0`.

Proof of feasibility in one line: for each arc on the path, the new flow is `f + δ ≤ u` on forward arcs and `f - δ ≥ 0` on backward arcs, and conservation is preserved at every internal node because the path enters and leaves it once.

**Claim C (termination and optimality).** With integer capacities, each augmentation raises `v` by at least 1, so at most `V_flow` augmentations occur, where `V_flow` is the final flow value. When no augmenting path exists, let `S` be the set reachable from `s` in the residual graph. Every arc from `S` to `T` is saturated and every arc from `T` to `S` carries zero flow. Then `v(f) = f(S,T) - f(T,S) = u(S,T) - 0 = u(S,T)`, and by Claim A no flow can exceed `u(S,T)`. So `f` is maximum and `(S,T)` is a minimum cut.

**What Claim C buys operationally.** The reachable set at termination *is* the bottleneck certificate. You can print it, hand it to a planner, and say "this partition carries 12 items/day and every arc in it is full". No solver log required, no trust in the objective value.

```text
FORMULA [NET-02]: number of augmentations  N ≤ v_max  (integer capacities, any augmenting path)
WHERE: v_max is the maximum flow value, in items per period.
CONSTRAINT: integer capacities and an integer-valued augmenting rule; with real
            capacities the bound is vacuous and the algorithm may not terminate.
NUMERICAL EXAMPLE: v_max = 2,000,000 items/day with unit-push augmentations gives
                   N ≤ 2,000,000; on a 5-arc residual graph that is 10^7 arc touches.
INTERPRETATION: the bound is pseudo-polynomial in the capacity magnitudes, not in the
                graph size; that is the entire difference between Ford-Fulkerson and Dinic.
```

**Why the bound is not good enough in general.** `N ≤ v_max` says the running time scales with the *numbers* in the capacity field, not with the structure. A network with four nodes and capacities of `10^6` can force two million augmentations if each path happens to push one unit. Edmonds-Karp removes the dependence by always choosing a shortest augmenting path, giving `O(V E^2)` independent of capacities; Dinic's blocking-flow construction gives `O(V^2 E)`. The practical rule: use a library that implements a capacity-independent bound, and treat "I wrote Ford-Fulkerson" as a red flag in review.

```text
FORMULA [NET-03]: min-cost optimality test via reduced costs
                  c^π(u,v) = c(u,v) + π(u) − π(v)  ≥ 0  for every residual arc
WHERE: c in currency per unit of flow; π in currency per unit; residual arcs include
       backward arcs with cost −c(u,v).
CONSTRAINT: the current flow is feasible and of the required value F; π updated by
            π(v) ← π(v) + dist(v) after each shortest-path computation.
NUMERICAL EXAMPLE: arc a→b with c = 4, π(a) = 1, π(b) = 6 gives c^π = 4 + 1 − 6 = −1,
                   which signals a profitable residual move and therefore a non-optimal flow.
INTERPRETATION: a negative reduced cost is an arbitrage; the algorithm is a machine for
                removing arbitrages until none remain, at which point the flow is optimal.
```

**Derivation of the reduced-cost test.** The cost of pushing `δ` along a residual path `P` is `δ · Σ_{a∈P} c^π(a)`, because the potential terms telescope: each internal node contributes `+π(v)` from its incoming arc and `−π(v)` from its outgoing arc, so only `π(s) − π(t)` survives, and that is constant for a fixed `F`. Therefore the cheapest way to change the flow is the shortest residual path under reduced costs, and if every residual reduced cost is non-negative, no change of flow can reduce cost. That is the successive-shortest-path algorithm and its optimality proof in one paragraph.

### 4.2 General case — transportation, assignment, and the dual certificate

```text
FORMULA [NET-04]: transportation problem and its dual
   primal: min Σ_i Σ_j c_ij x_ij
           s.t.  Σ_j x_ij = s_i  (supply, per period)
                 Σ_i x_ij = d_j  (demand, per period)
                 x_ij ≥ 0
   dual:   max Σ_i α_i s_i + Σ_j β_j d_j
           s.t.  α_i + β_j ≤ c_ij  for all (i,j)
WHERE: c in currency per unit shipped; s_i, d_j, x_ij in units per period;
       α_i, β_j in currency per unit, free in sign.
CONSTRAINT: Σ s_i = Σ d_j (balanced); if unbalanced, add a dummy with zero cost and
            the difference as its quantity, or the dual is unbounded.
NUMERICAL EXAMPLE: §5(b) gives primal 9 and dual 2·1+3·1+... = 9 on a 3×3 instance.
INTERPRETATION: strong duality means the shipping plan's cost equals the sum of node
                prices; a node price is the marginal value of one more unit of supply there.
```

**Why the assignment problem is the transportation problem with all supplies and demands equal to 1.** Set `s_i = d_j = 1`. The constraint matrix is the node-arc incidence matrix of a bipartite graph, which is totally unimodular, so every vertex of the feasible polytope is integral. The LP therefore returns a permutation, and the Hungarian algorithm is a specialised primal-dual method for the same LP. The practical consequence: you never need branch-and-bound for an assignment; if your solver is branching on an assignment model, the formulation is wrong.

**Complementary slackness, read operationally.** If `x_ij > 0` then `α_i + β_j = c_ij`: a used route is priced exactly at its cost. If `x_ij = 0` then `α_i + β_j ≤ c_ij`: an unused route is priced at or below its cost, i.e. it is not worth using at current prices. A planner reading the duals is reading marginal values: `α_i` is what one extra unit of supply at `i` is worth, and `β_j` is what one extra unit of demand at `j` costs.

```text
FORMULA [NET-05]: assignment lower bound from row and column reductions
                  LB = Σ_i min_j c_ij  +  Σ_j min_i (c_ij − min_k c_kj)
WHERE: c in currency units per assignment; LB is a lower bound on the optimal
       total assignment cost.
CONSTRAINT: square cost matrix; entries finite (an infeasible pair must be modelled
            with an explicit large cost and a feasibility check, never with 0).
NUMERICAL EXAMPLE: §5(b) row minima 2+3+1 = 6, then column minima 3+0+0 = 3, LB = 9,
                   and the achieved assignment costs exactly 9.
INTERPRETATION: when LB equals the cost of a feasible assignment you have proved
                optimality without a solver; when it does not, the gap tells you
                how much room a better assignment could still have.
```

### 4.3 Variants — project networks, crashing, and the classical sequencing rules

```text
FORMULA [NET-06]: critical path method, forward and backward pass
   forward:  ES_j = max_{i ∈ Pred(j)} EF_i ,  EF_j = ES_j + d_j ,  ES_source = 0
   backward: LF_i = min_{j ∈ Succ(i)} LS_j ,  LS_i = LF_i − d_i ,  LF_sink = project duration
   slack:    s_i = LS_i − ES_i = LF_i − EF_i
   duration: D = max_j EF_j = max over all s-t paths of Σ d_i
WHERE: d in days (same calendar as due dates); ES/EF/LS/LF all in days from time zero.
CONSTRAINT: the precedence graph is acyclic; durations are deterministic and known;
            a single project start; resource availability is not yet constrained.
NUMERICAL EXAMPLE: §5(c) gives D = 14 days, two critical paths, and slack 1 on activity B.
INTERPRETATION: D is a longest-path length, not a sum of all durations; the difference
                between the two is the parallelism the network actually achieves.
```

**Why the objective is a longest path and not a shortest one.** Every activity must complete, and an activity may start only when *all* predecessors finish, so the start of `j` is the maximum over predecessor finish times. The recursion takes maxima, which is a longest-path computation on a DAG. Shortest-path code applied here returns the earliest-finish chain, which is the path that constrains nothing.

**Crashing as a minimum cut.** Suppose the project must be shortened by one unit of time, and each activity `i` can be crashed at marginal cost `κ_i` per unit, up to a stated limit. The only activities worth crashing are those on some critical path — crashing anything else leaves `D` unchanged. To reduce `D` you must cut *every* critical path. Hence the cheapest single-unit crash is a minimum cut in the critical subnetwork, where each critical activity is an arc of capacity `κ_i`.

```text
FORMULA [NET-07]: cheapest one-unit crash = min cut on the critical subnetwork
                  cost* = min_{X hitting every critical path} Σ_{i ∈ X} κ_i
WHERE: κ_i in currency per day crashed; X is a set of critical activities.
CONSTRAINT: linear crash cost up to the stated limit per activity; independent costs;
            re-derive critical paths after every crash, because new ones appear.
NUMERICAL EXAMPLE: §5(c) — paths A→C→E→G and A→D→F→G both of length 14; crashing A
                   (κ = 500) cuts both for 500, and after the crash the next unit costs
                   κ_C + κ_D = 800 + 300 = 1,100 because both paths must be cut again.
INTERPRETATION: the marginal crash cost rises in steps as paths become critical; a
                single average "cost per day saved" figure is wrong for any horizon.
```

```text
FORMULA [NET-08]: earliest-due-date rule for maximum lateness
                  sequence jobs in non-decreasing d_j  ⇒  L_max is minimised
WHERE: L_j = C_j − d_j in days; jobs available at time 0; one machine; no preemption.
CONSTRAINT: all release dates equal; processing times independent of position.
NUMERICAL EXAMPLE: §5(d) EDD gives L_max = 0 with ΣC_j = 48, SPT gives L_max = 1 with
                   ΣC_j = 35; the better rule depends entirely on the declared objective.
INTERPRETATION: EDD is a theorem about the worst job, not about the average job; using
                it to improve mean flow time is a category error with a measurable cost.
```

**Exchange argument for NET-08, stated precisely.** Take an optimal sequence containing adjacent jobs `i` then `j` with `d_i > d_j`. Before the pair, completion time is `t`. With order `i,j` the latenesses are `t + p_i - d_i` and `t + p_i + p_j - d_j`; with order `j,i` they are `t + p_j - d_j` and `t + p_j + p_i - d_i`. The second maximum is at most the first because both completion times are lower or equal under `j,i` and `d_j < d_i`. Swapping never increases `L_max`, so a sequence with an inversion can be improved; EDD has none, hence EDD is optimal. Note what the argument uses: only the *maximum*, which is why it does not transfer to sums.

```text
FORMULA [NET-09]: Moore–Hodgson for minimising the number of late jobs
   order by EDD; scan; if job k completes after d_k, delete the job with the
   largest processing time among those scanned so far; continue.
WHERE: ΣU_j objective, one machine, all jobs available at time 0.
CONSTRAINT: non-preemptive; deleting a job removes it entirely rather than deferring it.
NUMERICAL EXAMPLE: §5(d) with six jobs deletes the 6-day job when the 12-day due date
                   is breached, leaving 5 jobs on time and total processing 15 of 21 days.
INTERPRETATION: the rule keeps the on-time set as large as possible, and it may discard
                a short-due-date long job; the discarded set is the plan's explicit
                "not this week" list, which a scheduler needs to see.
```

```text
FORMULA [NET-10]: Johnson's rule for a two-machine flow shop makespan
   partition: A = { j : p_1j ≤ p_2j },  B = { j : p_1j > p_2j }
   order: A in ascending p_1 , then B in descending p_2
WHERE: p_1j, p_2j in hours on machine 1 and machine 2; every job visits 1 then 2.
CONSTRAINT: exactly two machines; the same machine order for all jobs; no waiting cost.
NUMERICAL EXAMPLE: §5(e) yields makespan 21 h against 24 h for index order, and the
                   lower bound max(Σp_1 + min p_2, Σp_2 + min p_1) = 21 proves optimality.
INTERPRETATION: the rule is exact but narrow; the moment a third machine appears the
                problem becomes NP-hard and the rule is merely a heuristic.
```

### 4.4 Limits and counterexamples

**Limit 1 — capacities as numbers, not structure.** `N ≤ v_max` makes Ford-Fulkerson pseudo-polynomial. A network with `V = 4`, `E = 5`, and capacities of `10^6` can require about `2 × 10^6` augmentations, each touching 5 arcs, for `10^7` arc operations on a graph whose structure is trivial. The bound `O(V E^2)` for Edmonds-Karp on the same graph is `4 × 25 = 100` operations. The failure is not a slow constant; it is a scaling law that depends on the data values.

**Limit 2 — fractional optimum with integral data.** In a max-flow LP, integral capacities guarantee an integral optimum. That guarantee is about *single-commodity* flow. Add a second commodity that must not mix, and the same integral data admits half-integral optima; the min-cut certificate vanishes because there is no single scalar flow value to bound. A model that ignores commodity identity to keep the integrality guarantee is not an approximation of the real problem — it is a different problem whose optimum can be infeasible.

**Limit 3 — EDD on a sum objective.** The exchange argument in NET-08 depends on taking a maximum over jobs. For `ΣC_j` the correct rule is SPT and its exchange argument compares sums, where a shorter job placed earlier reduces *every* later completion time. On the §5(d) instance EDD gives `ΣC_j = 48` and SPT gives 35, a 27.1% difference in mean time in system. Neither rule is wrong; each is optimal for a different `γ`, and the cost of choosing by habit is measured in days of customer waiting.

**Limit 4 — Johnson's rule with a third machine.** With three machines and the same job order, no simple index rule is optimal. The standard counterexample class uses a few jobs where the rule's preferred order leaves machine 3 idle while an alternative order overlaps better; the makespan difference is a few percent, which is why the error survives review. The correct statement is: `F2||C_max` is solved in `O(n log n)` by Johnson, `F3||C_max` is NP-hard, and a rule that works for two machines is evidence about two machines only.

**Limit 5 — the crashing min-cut assumes independent linear costs.** If crashing activity `i` changes the cost of crashing `j` (shared crew, shared equipment), the cut formulation prices the two independently and the result is not the cheapest plan. The discriminating observation is a joint cost that does not decompose; the fix is a MIP over crash variables with the coupling constraints, not a bigger cut.

**Limit 6 — a DAG assumption hiding a cycle.** Both CPM passes assume acyclicity. A data extract containing `A precedes B` and `B precedes A` makes both recursions non-terminating or silently wrong depending on implementation order. Cycle detection is `O(V + E)` and belongs in the loader, not in the optimiser.

**Limit 7 — deterministic durations sold as commitments.** `D = 14` days is a statement about a network with known durations. If the durations are estimates with a spread, `D` is one quantile of a distribution, and quoting it as a promise transfers the entire estimation error to the delivery commitment. The honest output is a range with the assumption set attached.

## 5. Worked Numerical Example

Five instances, one per model family. Each is small enough to check by hand; each number is recomputed in a re-check block. All inputs are `ILLUSTRATIVE` constructed values.

### 5a. Max flow on a six-node, eight-arc network, with its cut certificate

Nodes: `s = 1`, `a = 2`, `b = 3`, `c = 4`, `d = 5`, `t = 6`. Capacities in items/day.

| Arc | From | To | Capacity |
|---|---|---|---|
| e1 | 1 | 2 | 8 |
| e2 | 1 | 3 | 6 |
| e3 | 2 | 3 | 3 |
| e4 | 2 | 4 | 5 |
| e5 | 3 | 5 | 7 |
| e6 | 4 | 5 | 4 |
| e7 | 4 | 6 | 9 |
| e8 | 5 | 6 | 10 |

**Augmentation 1.** Path `1 → 2 → 4 → 6`. Residual bottlenecks: 8, 5, 9; so `δ = 5`. Flow now `e1 = 5`, `e4 = 5`, `e7 = 5`, value 5.
**Augmentation 2.** Path `1 → 2 → 3 → 5 → 6`. Residuals: `e1` has 3 left, `e3` has 3, `e5` has 7, `e8` has 10; `δ = 3`. Flows: `e1 = 8` (saturated), `e3 = 3` (saturated), `e5 = 3`, `e8 = 3`; value 8.
**Augmentation 3.** Path `1 → 3 → 5 → 6`. Residuals: `e2` has 6, `e5` has 4, `e8` has 7; `δ = 4`. Flows: `e2 = 4`, `e5 = 7` (saturated), `e8 = 7`; value 12.
**Termination check.** Residual reachable set from node 1: node 2 unreachable (`e1` saturated, no backward arc into 1 available), node 3 reachable via `e2` with 2 residual, node 4 unreachable (only entry is `e4`, saturated, and `e6` runs `4 → 5` not into 4), node 5 reachable from 3 only via `e5` which is saturated, node 6 unreachable. Hence `S = {1, 3}`, `T = {2, 4, 5, 6}`... the cut must be read from the reachable set only, so `S = {1, 3}` and the cut capacity is `u(1→2) + u(3→5) = 8 + 7 = 15`, which exceeds 12 and therefore is not minimum. Recomputing: the residual reachable set must be recomputed after the *final* flow, and at value 12 node 3 is reachable with residual 2, so `S` includes 3. The correct minimum cut found by inspection is `S = {1, 2, 3}`, `T = {4, 5, 6}`, capacity `u(2→4) + u(3→5) = 5 + 7 = 12`.

Let me state this carefully rather than leave a contradiction in the sheet: at value 12 the residual graph has `e2` residual 2, so node 3 is reachable, and from node 3 no residual arc leaves (`e5` saturated, no `3 → 2` original arc but a backward residual arc `3 → 2` exists with capacity 3, which reaches node 2). So node 2 *is* reachable via the backward arc `3 → 2`, from which `e4` is saturated, `e3` backward to 2 — and node 4 is reached from 2 only through `e4`, which carries 5 of 5, so no. Therefore `S = {1, 2, 3}`, `T = {4, 5, 6}`, and the forward arcs are `2 → 4` (5) and `3 → 5` (7), total 12. The earlier reachability sketch was wrong about node 2 and the corrected set is consistent with Claim C.

```text
FORMULA [NET-11]: flow value = capacity of the cut found by residual reachability
WHERE: S = nodes reachable from s in the residual graph of the final flow.
CONSTRAINT: every arc S→T saturated, every arc T→S empty; otherwise the set is not
            the reachable set and the equality fails by construction.
NUMERICAL EXAMPLE: S = {1,2,3}, T = {4,5,6}; arcs S→T are 2→4 (cap 5, flow 5) and
                   3→5 (cap 7, flow 7); sum 12 = flow value.
INTERPRETATION: the certificate is checkable with a two-line assertion — sum the
                saturated crossing arcs and compare with the reported value.
```

**Conservation re-check, node by node.** Node 1 sends `8 + 4 = 12`; node 6 receives `5 + 7 = 12`. Node 2 receives 8 and sends `5 + 3 = 8`. Node 3 receives `4 + 3 = 7` and sends 7. Node 4 receives 5 and sends 5. Node 5 receives 7 and sends 7. All balanced; total 12.

**Sensitivity.** Raise `u(3→5)` from 7 to 9: path 3 can now carry 6, so `e2` saturates at 6, value rises to `5 + 3 + 6 = 14`, and the cut becomes `2→4` (5) plus `3→5`? No — with `e2` saturated the cut `{1,2,3}` has capacity `5 + 9 = 14`, matching, so the value rises by exactly 2. Lower `u(2→4)` to 3 instead: value falls to `3 + 3 + 4 = 10` and the cut capacity becomes `3 + 7 = 10`. In both cases the value moves one-for-one with the crossing-arc capacity, which is the operational meaning of a bottleneck.

### 5b. A 3×3 assignment with a dual certificate

Cost matrix in currency units per assignment. Workers `W1..W3` to jobs `J1..J3`.

| | J1 | J2 | J3 |
|---|---|---|---|
| W1 | 9 | 2 | 7 |
| W2 | 6 | 4 | 3 |
| W3 | 5 | 8 | 1 |

**Enumeration of all six permutations.** `(J1,J2,J3)` for `(W1,W2,W3)`: `9+4+1 = 14`; `(J1,J3,J2)`: `9+3+8 = 20`; `(J2,J1,J3)`: `2+6+1 = 9`; `(J2,J3,J1)`: `2+3+5 = 10`; `(J3,J1,J2)`: `7+6+8 = 21`; `(J3,J2,J1)`: `7+4+5 = 16`. Minimum 9 at `W1→J2`, `W2→J1`, `W3→J3`.

**Reduction certificate.** Row minima are 2, 3, 1, summing to 6. Subtracting: row 1 becomes `7, 0, 5`; row 2 becomes `3, 1, 0`; row 3 becomes `4, 7, 0`. Column minima are 3, 0, 0, summing to 3. The lower bound is `6 + 3 = 9`, and the assignment `W1→J2` (0), `W2→J1` (0 after both reductions), `W3→J3` (0) achieves 9. Bound equals achieved cost, so optimality is proved by arithmetic, not by solver trust.

```text
FORMULA [NET-12]: optimal assignment cost = sum of row minima + sum of column minima
                  when the reduced matrix admits a zero-cost perfect assignment
WHERE: reduced cost r_ij = c_ij − rowmin_i − colmin_j, all in currency units.
CONSTRAINT: a perfect assignment using only zero entries must exist; otherwise
            the equality is a strict lower bound and further reductions are needed.
NUMERICAL EXAMPLE: 6 + 3 = 9 and the zero-assignment costs 9.
INTERPRETATION: the dual variables are literally the row and column minima here,
                so the certificate is the arithmetic already performed.
```

**Why the greedy "assign each worker their cheapest job" fails on this instance.** W1's cheapest is J2 at 2, W2's cheapest is J3 at 3, W3's cheapest is J3 at 1 — a collision on J3, and resolving it arbitrarily can force W2 onto J1 (6) or J2 (4). Greedy with arbitrary collision resolution can reach 14 while the optimum is 9, a 55.6% overspend. The defect is that greedy reads `c_ij` and ignores the shadow price of consuming a scarce column.

### 5c. Project network, critical paths, and the price of one day

Seven activities, durations in days, predecessors as listed.

| Activity | Duration | Predecessors |
|---|---|---|
| A | 4 | — |
| B | 6 | — |
| C | 3 | A |
| D | 5 | A |
| E | 4 | B, C |
| F | 2 | D |
| G | 3 | E, F |

**Forward pass.** `A: ES 0, EF 4`. `B: ES 0, EF 6`. `C: ES 4, EF 7`. `D: ES 4, EF 9`. `E: ES max(6,7) = 7, EF 11`. `F: ES 9, EF 11`. `G: ES max(11,11) = 11, EF 14`. Duration `D = 14` days.

**Backward pass.** `G: LF 14, LS 11`. `F: LF 11, LS 9`. `E: LF 11, LS 7`. `D: LF 9, LS 4`. `C: LF 7, LS 4`. `B: LF 7, LS 1`. `A: LF min(4,4) = 4, LS 0`.

**Slack.** A 0, B 1, C 0, D 0, E 0, F 0, G 0. Two critical paths exist: `A→C→E→G` with `4+3+4+3 = 14` and `A→D→F→G` with `4+5+2+3 = 14`.

**Crash prices** in currency units per day: A 500, C 800, D 300, E 700, F 400. Crash limits: one day per activity in this example.

**Cheapest first crash.** Any single activity on *all* critical paths is only A, since C and E lie on one path and D and F on the other. Crashing A by one day saves 500 and yields `D = 13`, with both paths at 13. Crashing C alone costs 800 and saves nothing: `A→D→F→G` is still 14, so `D` stays 14. This is the classic wasted spend, and it is invisible in a report that lists "activities crashed" without listing "project duration before and after".

**Cheapest second crash.** With A at its one-day limit, both paths must be cut: `{C, D}` costs `800 + 300 = 1,100`; `{C, F}` costs `800 + 400 = 1,200`; `{E, D}` costs `700 + 300 = 1,000`; `{E, F}` costs `700 + 400 = 1,100`. Cheapest is `{E, D}` at 1,000, giving `D = 12`. Note the cut is over the critical subnetwork, and the cheapest set is not the cheapest *pair of individually cheap* activities.

```text
FORMULA [NET-13]: total crash cost to reach duration 12 = κ_A + min(κ_C+κ_D, κ_C+κ_F, κ_E+κ_D, κ_E+κ_F)
WHERE: κ in currency per day; each activity crashed at most one day in this instance.
CONSTRAINT: after each crash, recompute the critical paths; new ones can appear and
            then the cut must cover them too.
NUMERICAL EXAMPLE: 500 + min(1,100, 1,200, 1,000, 1,100) = 500 + 1,000 = 1,500.
INTERPRETATION: the marginal cost of schedule compression roughly doubles from the
                first day to the second; a linear extrapolation underestimates by 100%.
```

**Independent re-check.** Enumerate all paths: `A→C→E→G = 14`, `A→D→F→G = 14`, `B→E→G = 13`. The longest is 14, matching the forward pass. After crashing A to 3: `13, 13, 12` — longest 13, matching. After additionally crashing E to 3 and D to 4: paths `A→C→E→G = 3+3+3+3 = 12`, `A→D→F→G = 3+4+2+3 = 12`, `B→E→G = 6+3+3 = 12` — three critical paths now, which means the *next* crash must cut all three, and E and G lie on all three. The step from two to three critical paths is the reason crash plans are re-solved rather than extended.

**Sensitivity.** If B's duration rises from 6 to 7 days, its slack becomes 0 and `B→E→G = 13` stays under 14 only while B ≤ 7. At B = 8 the path becomes 14 and there are three critical paths with no crash performed at all — a schedule can acquire a new bottleneck through an unrelated estimate change.

### 5d. Single-machine sequencing: EDD versus SPT on the same five jobs

Jobs with processing times in days and due dates measured from time zero.

| Job | p | d | Ratio p/d |
|---|---|---|---|
| J1 | 3 | 6 | 0.500 |
| J2 | 4 | 8 | 0.500 |
| J3 | 2 | 11 | 0.182 |
| J4 | 5 | 14 | 0.357 |
| J5 | 1 | 20 | 0.050 |

**EDD order** `J1, J2, J3, J4, J5`. Completion times 3, 7, 9, 14, 15. Lateness `-3, -1, -2, 0, -5`; `L_max = 0`. `ΣC_j = 48`, mean flow `9.6` days. No job late, so `ΣU_j = 0` and `ΣT_j = 0`.

**SPT order** `J5, J3, J1, J2, J4`. Completion times 1, 3, 6, 10, 15. Lateness `-19, -8, 0, 2, 1`; `L_max = 2`. `ΣC_j = 35`, mean flow `7.0` days. One job late (`J4`), `ΣT_j = 3`, `ΣU_j = 1`.

```text
FORMULA [NET-14]: ΣC_j under any sequence = Σ_j (n − j + 1) · p_j  (jobs indexed 1..n in sequence order)
WHERE: p in days; the coefficient of a job's processing time is the number of jobs
       whose completion includes it; all coefficients are integers ≥ 1.
CONSTRAINT: single machine, no idle time inserted deliberately, all jobs available at 0.
NUMERICAL EXAMPLE: EDD order gives 5·3 + 4·4 + 3·2 + 2·5 + 1·1 = 15+16+6+10+1 = 48.
                   SPT order gives 5·1 + 4·2 + 3·3 + 2·4 + 1·5 = 5+8+9+8+5 = 35.
INTERPRETATION: coefficients decrease along the sequence, so placing the smallest
                processing time where the coefficient is largest (front) minimises the sum.
```

**The 27.1% figure.** `(48 − 35)/48 = 13/48 = 0.2708`, so SPT reduces mean time in system by 27.1% relative to EDD on this instance, while EDD delivers `L_max = 0` against SPT's 2 days. Both are optimal for their own objective: NET-08 and NET-14 are both theorems, and the instance is a clean demonstration that they disagree.

**Adding a sixth job to exercise Moore–Hodgson.** Job `J6` with `p = 6`, `d = 12`. EDD order becomes `J1(6), J2(8), J3(11), J6(12), J4(14), J5(20)`. Cumulative completion: 3, 7, 9, 15, 20, 21. At `J6`, completion 15 exceeds due date 12, so delete the longest job among `{J1 3, J2 4, J3 2, J6 6}` — delete `J6`, and cumulative drops to 9. Then `J4` completes at 14 (on time), `J5` at 15 (on time). Result: five jobs on time, one dropped, scheduled processing 15 days of 21 available, `ΣU_j = 1`. The dropped set is `{J6}` and its contribution is 6 days of work deferred, which is the number a capacity discussion needs.

**Sensitivity.** If `J4`'s due date moves from 14 to 13, EDD's `L_max` becomes `max(0, ...)` — recomputing: completions unchanged at 3, 7, 9, 14, 15 with dues 6, 8, 11, 13, 20, so latenesses are `-3, -1, -2, 1, -5` and `L_max = 1`. A one-day due-date change turns a zero-lateness plan into a breach, and no sequencing rule repairs it: the total work is 15 days and the binding due date is 13.

### 5e. Two-machine flow shop: Johnson's rule and a lower-bound certificate

Jobs with processing times in hours on machine 1 and machine 2.

| Job | p1 | p2 | Partition |
|---|---|---|---|
| J1 | 3 | 4 | A (p1 ≤ p2) |
| J2 | 5 | 2 | B (p1 > p2) |
| J3 | 2 | 6 | A |
| J4 | 6 | 3 | B |
| J5 | 1 | 5 | A |

**Johnson order.** Set A sorted by `p1` ascending: `J5 (1), J3 (2), J1 (3)`. Set B sorted by `p2` descending: `J4 (3), J2 (2)`. Sequence `J5, J3, J1, J4, J2`.

**Machine 1 completions.** `1, 3, 6, 12, 17`. **Machine 2.** J5 starts 1, ends 6; J3 starts 6, ends 12; J1 starts 12, ends 16; J4 starts 16, ends 19; J2 starts 19, ends 21. Makespan 21.

**Index-order comparison** `J1, J2, J3, J4, J5`. Machine 1 completions `3, 8, 10, 16, 17`. Machine 2: J1 3→7; J2 8→10; J3 10→16; J4 16→19; J5 19→24. Makespan 24.

```text
FORMULA [NET-15]: makespan lower bound  LB = max(Σ_j p_1j + min_j p_2j, Σ_j p_2j + min_j p_1j)
WHERE: p in hours; the bound applies to any sequence, not only the rule's sequence.
CONSTRAINT: two machines, same order for all jobs, no preemption.
NUMERICAL EXAMPLE: Σp1 = 17, Σp2 = 20, min p2 = 2, min p1 = 1;
                   LB = max(17+2, 20+1) = max(19, 21) = 21 = achieved makespan.
INTERPRETATION: machine 2 cannot start before the first job's machine-1 work (≥ min p1)
                and must then process all 20 hours, so 21 is unavoidable; Johnson attains it.
```

**Gain from the rule.** `(24 − 21)/24 = 0.125`, a 12.5% makespan reduction on five jobs, achieved by reordering alone with no capacity added.

**Independent re-check of the makespan.** Simulate the sequence with the recursion `C2_0 = 0`, `C1_k = C1_{k-1} + p1_k`, `C2_k = max(C1_k, C2_{k-1}) + p2_k`. For `k = 1..5`: `C1 = 1, 3, 6, 12, 17`; `C2 = 1+5 = 6`, `max(3,6)+6 = 12`, `max(6,12)+4 = 16`, `max(12,16)+3 = 19`, `max(17,19)+2 = 21`. Consistent with the table.

**Sensitivity.** Add a third machine with `p3 = 2` for every job and the rule's sequence gives machine 3 finishing at `21 + 2 = 23`, but the instance is now `F3`, which is NP-hard; the rule's order is a heuristic whose gap must be measured, not assumed. Add a release date to J4 and Johnson's partition argument no longer applies at all, because the exchange compares adjacent jobs with a common start time.

## 6. Visual / Geometric Intuition

**The cut as a closed curve.** Draw the network and draw any closed curve around the source that crosses arcs but no nodes. The capacity of the cut is the sum of the capacities it crosses outward. Making the curve bulge where capacity is plentiful and pinch where capacity is scarce shows why the minimum cut sits at the pinches: the flow value can never exceed the smallest set of arcs you can force it through.

**The residual graph as permission to change your mind.** Forward residual capacity is unused headroom; backward residual capacity is the right to reduce an earlier decision. Augmentation is therefore not "adding flow" but "improving a plan", and the algorithm's correctness comes from it never committing irreversibly. A planner who understands this stops being surprised that the final routing differs from the first path chosen.

**The Gantt chart as a packing in a strip.** Width is time, height is the resource count. A single machine is a strip of height 1; the makespan is the rightmost edge; WIP at time `t` is the number of rectangles crossing the vertical line at `t`; flow time of a job is the distance from its release line to its right edge. Visualising the objective this way immediately shows why SPT (short rectangles first) minimises total right-edge sum and why EDD (early deadlines first) minimises the worst overshoot past a deadline marker.

**Critical path as the longest route through a DAG.** Lay the activities out with arcs pointing right; assign each arc a length equal to the duration. The project duration is the longest path from start to finish. Crashing is shortening edges until the longest path is short enough; when two routes tie for longest, shortening one changes nothing, which is exactly the picture of wasted expenditure.

**The crash as a cut through a bundle of parallel routes.** Draw the critical subnetwork as a set of parallel routes from start to finish. To shorten the project, every route must be shortened, so the cheapest action is a set of edges meeting every route — a cut. The picture makes the "cheapest pair" intuition wrong: a cheap edge on only one route is worthless until the other routes are also cut.

**The assignment polytope as a permutahedron.** The feasible LP region for an assignment has vertices at permutation matrices only. The optimum is therefore at a corner, which is why no interior search is needed. Adding a side constraint such as "worker 1 and worker 2 cannot share a shift" can cut a corner off and expose a fractional vertex; the picture explains precisely when integrality is lost.

**The scheduling polytope as a disjunctive union.** Sequencing constraints "either `i` before `j` or `j` before `i`" describe a union of regions rather than one convex region. That non-convexity is the geometric reason `J||C_max` needs branching, while the flow polytope's convexity is the reason max flow needs none. The picture is the clearest available explanation of why one family of problems is polynomially solvable and the other is not.

**Reading a utilisation chart against a cut.** Utilisation colours arcs by `f/u`; a minimum cut is not the most-coloured arc. On the §5a network the cut contains `2→4` at 100% and `3→5` at 100%, but the arc `1→2` is also at 100% and is not in the minimum cut, because raising its capacity by one does not raise throughput — the cut would still be 12. Utilisation identifies candidates; the cut decides.

## 7. Comparison / Selection Table

### 7.1 Model families against the question they answer

| Model | Question answered | Input size guide | Algorithm | Certificate produced | Reject when |
|---|---|---|---|---|---|
| Max flow | throughput ceiling | up to 10^5 nodes routinely | Dinic / push-relabel | min cut with equal capacity | commodities must stay separate |
| Min-cost flow | cheapest routing for a fixed volume | 10^4 nodes, 10^5 arcs | successive shortest path with potentials | node potentials, reduced costs ≥ 0 | costs depend on flow magnitude non-linearly |
| Transportation | cheapest sourcing plan | 10^2 × 10^2 dense | network simplex | row/column duals | supply or demand is not fixed |
| Assignment | one-to-one matching at least cost | 10^3 × 10^3 | Hungarian / min-cost flow | row and column reductions | eligibility is many-to-many or capacity-limited |
| Bipartite matching | maximum pairs | 10^5 nodes | Hopcroft–Karp | König cover of equal size | graph is not bipartite |
| CPM / PERT network | project duration, slack | 10^4 activities | longest path on a DAG | critical path list | durations are stochastic and the promise is a date |
| Crashing by cut | cheapest compression | critical subnetwork only | min cut on critical arcs | cut set and its cost | crash costs interact |
| Single-machine rules | order for a declared objective | any `n` | sort | proof by exchange argument | release dates or precedence are present |
| Johnson (F2) | makespan for two machines | any `n` | sort and partition | lower bound equality | three or more machines |
| Job/flow shop (m ≥ 3) | makespan | small | MIP with bound or heuristic | incumbent and gap | someone asks for a proof of optimality |

### 7.2 Objective to rule, with the guarantee and the failure mode

| Declared objective | Exact rule | Guarantee | Fails when | Observable symptom of misuse |
|---|---|---|---|---|
| `L_max` minimise | EDD | exact, `O(n log n)` | release dates present | jobs with early release are scheduled first regardless of deadline |
| `ΣC_j` minimise | SPT | exact, `O(n log n)` | weights differ | mean time improves while a critical customer breaches |
| `Σw_jC_j` minimise | Smith ratio `p_j/w_j` ascending | exact | precedence constraints | weighted jobs idle behind long unweighted work |
| `ΣU_j` minimise | Moore–Hodgson | exact | weights on jobs | a high-value job is dropped as if it were ordinary |
| `C_max` on 2 machines | Johnson | exact | a third machine appears | makespan reported as optimal for `F3` |
| `C_max` on parallel machines | LPT or MIP with bound | heuristic / bound | machines differ in speed | identical-machine assumption in a mixed fleet |
| `ΣT_j` minimise | none simple | NP-hard | any instance | a heuristic labelled optimal |
| Project duration | longest path | exact on a DAG | precedence cycle exists | recursion terminates with a wrong duration |

### 7.3 Choosing between the two certificates

| Situation | Prefer | Why |
|---|---|---|
| Stakeholder asks "how much more can we push through" | max flow plus min cut | the cut names the arcs to invest in |
| Stakeholder asks "where is the cheapest capacity" | min-cost flow potentials | the dual prices each node's marginal unit |
| Stakeholder asks "why is the date slipping" | CPM slack table | slack localises the constraint to activities |
| Stakeholder asks "what if we add a shift" | network model with a changed capacity | one parameter change, re-solve, compare values |
| Stakeholder asks "can you guarantee the plan" | neither, alone | guarantees need bounds or proofs, not plans |

### 7.4 Selection procedure in five steps

1. Write the objective symbol (`L_max`, `ΣC_j`, `C_max`, throughput, cost) and its unit into the decision record before choosing a method.
2. Classify the resource environment: one machine, identical parallel, unrelated parallel, flow shop, job shop, or a capacity network. The class determines tractability.
3. Check the hypotheses the class result needs: release dates, precedence, preemption, commodity identity, additivity, deterministic durations.
4. Pick the strongest method whose hypotheses all hold; if none does, pick a bound-producing method and label the output "feasible, gap X%".
5. Attach the certificate — cut, duals, slack table, or bound — to the plan; a plan without its certificate cannot be reviewed.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom, the two or three mechanisms that produce it, and a check that separates them. Ordered roughly by how often they occur.

### F-01 — Throughput reported higher than any cut can support

**Symptom.** The model reports 14 items/day where a hand audit finds a partition carrying 12.
**Competing causes.** (a) The cut capacity computation included `T → S` arcs; (b) capacities were loaded per week while the reported figure is per day, a factor of 7; (c) parallel arcs between the same pair were merged by taking the maximum instead of the sum.
**Discriminating check.** Recompute the value as `Σ_{a ∈ S→T} u_a` for the cut found by residual reachability and compare with `Σ_{a ∈ S→T} f_a`. If the two agree, the flow is valid and the reported number was mis-scaled (b). If they disagree, recompute without `T → S` arcs (a). If the disagreement is exactly the sum of the dropped parallel arc (c), the loader merged arcs.
**Correction.** Sum parallel arcs; assert unit consistency by recomputing from raw quantities; treat `T → S` arcs as absent from capacity.

### F-02 — A single "bottleneck arc" is named and investment goes there

**Symptom.** A capacity project raises one arc's capacity by 20% and throughput moves by 0%.
**Competing causes.** (a) The named arc was the most utilised but not in any minimum cut; (b) the arc was in one minimum cut but a different minimum cut became binding after the change; (c) the arc's capacity was raised but its downstream arc remained the constraint.
**Discriminating check.** Recompute the min cut after the change and compare with the pre-change cut. Unchanged cut with a raised capacity inside it means the arc was not in the binding cut (a). A cut that moved to a disjoint set means multiple minimum cuts exist (b) — confirm by finding two distinct cuts of equal capacity in the original network.
**Correction.** Report the whole cut, not one arc; state that when several minimum cuts exist, a capacity increase helps only if it raises every cut simultaneously.

### F-03 — Fractional shipments in an integer world, or the reverse

**Symptom.** The plan routes 0.5 of a truck, 2.5 of a pallet.
**Competing causes.** (a) Divisible units genuinely allowed (bulk liquids) and the plan is right; (b) the model has integer data but a commodity mix that destroys integrality; (c) the unit of flow was chosen as a fraction of the real unit.
**Discriminating check.** Ask whether the LP optimum is integral. With a single commodity and integer capacities it must be; if it is fractional, either the network has a multi-commodity extension (b) or a capacity is fractional (c). Enumerate the flow vector and count non-integer entries; a single non-integer entry on an arc with integer capacity is diagnostic of (b) or (c).
**Correction.** Declare the unit of flow in the model record; if units are indivisible, use an integer formulation and expect a bound rather than a certificate.

### F-04 — Lower bounds dropped, plan satisfies a constraint it should not

**Symptom.** A route that contractually must carry at least 3 units carries 0 and the model reports feasible.
**Competing causes.** (a) The lower bound was never entered; (b) it was entered as an upper bound due to a sign or column-swap error; (c) the lower bound was entered but the feasibility check ran against the LP relaxation only.
**Discriminating check.** For each arc with a lower bound, assert `f_a ≥ l_a` on the returned flow vector. A violation with `l_a > 0` in the input record rules out (a); then check whether `f_a ≤ l_a` holds instead, which identifies (b).
**Correction.** Add the check to the plan validator, not to the optimiser, so it fires even when a different method produces the plan.

### F-05 — Multi-commodity reality modelled as single commodity

**Symptom.** Two products share a lane, the plan mixes them, or a truck arrives carrying the wrong product.
**Competing causes.** (a) Product identity was dropped to make the model a plain max flow; (b) identity was kept in the data but not in the variables; (c) the products are genuinely interchangeable and the "error" is a labelling problem downstream.
**Discriminating check.** Read the variable index: if it lacks a commodity index, (a) or (b). If it has one, check whether constraints were written per commodity or shared — a shared capacity constraint with per-commodity conservation is the correct form.
**Correction.** Either add the commodity index and accept a harder problem, or document the interchangeability assumption and verify it with the product owners.

### F-06 — Critical path computed on a precedence graph that contains a cycle

**Symptom.** The forward pass returns a duration shorter than a single activity's, or the recursion loops.
**Competing causes.** (a) Data entry created `A → B` and `B → A`; (b) a milestone was modelled as both predecessor and successor of a summary task; (c) the graph is acyclic but a self-loop was introduced by a duplicate key.
**Discriminating check.** Run a topological sort and inspect the residual set after Kahn's algorithm. Any remaining nodes form cycles (a, c); a self-loop is visible as `u = v` in the arc list (c); summary-task arcs are identifiable by the milestone flag (b).
**Correction.** Reject instances with cycles at load time; require summary tasks to have no out-arcs.

### F-07 — Crashing a non-critical activity and reporting a saving

**Symptom.** Spend is recorded, project duration is unchanged, and the report claims a benefit.
**Competing causes.** (a) The activity had slack at decision time; (b) the activity was critical before an earlier crash and stopped being critical after it; (c) the duration did change but the reported duration came from a stale cached value.
**Discriminating check.** Compare the slack table before and after the crash. Positive slack before means (a). Zero slack before and positive after means the previous crash removed it from the critical set (b). Recomputed duration unchanged with a recomputed slack table means (c) — the cache.
**Correction.** Recompute slack after every crash and gate spend on slack being zero; invalidate caches on any duration edit.

### F-08 — Slack treated as a permanent budget across changes

**Symptom.** An activity uses its slack, and the project overruns anyway.
**Competing causes.** (a) Another activity on the same non-critical path also used slack and the path became critical; (b) a duration estimate rose elsewhere; (c) slack was shared between two paths through a common activity and both consumed it.
**Discriminating check.** Recompute the path lengths through the affected activity. If two activities on one path each consumed slack, the path sum exceeded the old slack (a, c). If no other consumption occurred, compare the duration estimate against the one used when slack was published (b).
**Correction.** Publish slack per path, not per activity, or publish it as "slack available if nothing else on this path moves".

### F-09 — EDD applied to a weighted or average objective

**Symptom.** Service-level breaches concentrate on high-value customers while the "worst lateness" metric looks acceptable.
**Competing causes.** (a) `L_max` was optimised while the contract measures `Σw_jT_j`; (b) due dates in the extract are internal targets, not contract dates; (c) weights exist in the data but not in the objective.
**Discriminating check.** Recompute the plan's `Σw_jT_j` and compare with a weighted-ratio order on the same instance. A materially better weighted objective with the same `L_max` proves (a). Compare the due-date column with the contract table to test (b).
**Correction.** Name the objective symbol in the plan header; where weights exist, optimise the weighted objective and report both metrics.

### F-10 — Johnson's rule used on three or more machines

**Symptom.** Reported makespan is beaten by a schedule a scheduler improvised.
**Competing causes.** (a) Third machine exists and the rule is invalid; (b) two machines but with different job orders on each, i.e. a job shop not a flow shop; (c) two machines but with machine 2 required before machine 1 for some jobs.
**Discriminating check.** Count distinct machines in the routing data. Three or more rules out any exact two-machine claim (a). Distinct per-job machine orders identify (b); a per-job order starting with machine 2 identifies (c).
**Correction.** Use the rule only for `F2`; elsewhere solve a MIP with a time limit and report the gap.

### F-11 — Makespan optimised while WIP explodes

**Symptom.** The batch finishes earliest possible and the floor is congested for the whole batch.
**Competing causes.** (a) The objective was `C_max` on a system whose real cost is WIP; (b) flow time was computed on the schedule's own completion times, hiding queueing; (c) jobs were released all at once, so release policy rather than sequencing created the congestion.
**Discriminating check.** Compute mean WIP as the area under the concurrency curve divided by the makespan, and compare with a release-controlled plan. Lower makespan with much higher WIP area confirms (a); equal WIP with equal makespan but different flow times indicates (b) or (c).
**Correction.** Report makespan, mean flow time, and peak concurrency together; if WIP is the business cost, optimise flow time subject to a makespan ceiling.

### F-12 — Calendar time versus working time

**Symptom.** Completion dates drift by exactly the weekends or holidays between start and finish.
**Competing causes.** (a) Durations are in working days but arithmetic is in calendar days; (b) shifts cross midnight and the date was taken from the start of the shift; (c) a daylight-saving transition shifted a local-time boundary by an hour.
**Discriminating check.** Compute the elapsed days between the model's start and finish and compare with the sum of processing times. A difference equal to the count of non-working days identifies (a); a consistent one-day offset for night shifts identifies (b); a one-hour error twice a year identifies (c).
**Correction.** Keep durations and calendars in the same unit; store instants in UTC with an explicit calendar mapping and test the transition dates.

### F-13 — The plan is feasible in the model and impossible in the plant

**Symptom.** Every constraint checks out and the supervisor says the plan cannot be run.
**Competing causes.** (a) Setup or cleanup time is absent from `p_j`; (b) a machine qualification is not modelled, so a job is assigned to a resource that cannot do it; (c) preemption is allowed in the model but not in the process.
**Discriminating check.** Compare planned and actual durations on the jobs that overran; a systematic shortfall proportional to job type indicates setup (a). Intersect the assignment with the qualification matrix; any non-empty intersection identifies (b). Read the model flag for preemption and compare with the process description (c).
**Correction.** Load setup matrices; model eligibility explicitly; set preemption to false unless the process physically supports it.

### F-14 — Two runs, same input, different schedule, same objective value

**Symptom.** Reproducibility failure on a validated model.
**Competing causes.** (a) The sort key is not total, so equal-cost alternatives are resolved by input order; (b) solver parameters (seed, threads, time limit) differ between runs; (c) the reconstruction pass breaks ties differently from the table pass.
**Discriminating check.** Shuffle the input rows and hash the resulting plan. A hash change identifies (a). Identical hash under shuffling but different hashes across machines identifies (b). A stable plan from the solver with an unstable accepted set identifies (c).
**Correction.** Add the job identifier as the final sort key; pin solver seed and thread count; centralise tie-breaking in one function used by both passes.

### F-15 — Sensitivity analysis run on a stale model

**Symptom.** The "what if capacity rises" table says no change, and a small change in reality does change throughput.
**Competing causes.** (a) The sensitivity was computed on a model with a different cut structure than the current one; (b) the capacity increase was applied to an arc outside every minimum cut; (c) the table was computed once and the input changed afterwards.
**Discriminating check.** Recompute the min cut on the current input and compare with the cut used in the table. A different cut identifies (a); the same cut with the edited arc outside it identifies (b); input hash mismatch identifies (c).
**Correction.** Store the cut and the input hash beside every sensitivity figure; recompute rather than re-label.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity

Sizes are `V` nodes, `E` arcs, `n` jobs, `m` activities, `F` the required flow value, and `U` the largest capacity. The operation model is a unit-cost arithmetic or comparison operation on 64-bit values, ignoring memory hierarchy and I/O.

| Procedure | Input size | Time | Space | Dominant bottleneck |
|---|---|---|---|---|
| Residual augmentation, arbitrary path | `V, E, F` | `O(F · E)` | `O(V + E)` | pseudo-polynomial in capacity magnitude |
| Edmonds–Karp | `V, E` | `O(V · E²)` | `O(V + E)` | repeated BFS over all arcs |
| Dinic blocking flow | `V, E` | `O(V² · E)` | `O(V + E)` | level-graph construction |
| Push–relabel | `V, E` | `O(V³)` generic, better in practice | `O(V + E)` | relabel operations |
| Capacity scaling | `V, E, U` | `O(E log U · (E + V log V))` | `O(V + E)` | shortest-path work per scale |
| Successive shortest path (min cost) | `V, E, F` | `O(F · (E + V log V))` | `O(V + E)` | one shortest path per unit of flow |
| Network simplex (transportation) | `V, E` | fast in practice, exponential worst case | `O(V + E)` | pivot sequence |
| Hungarian (assignment) | `n × n` | `O(n³)` | `O(n²)` | matrix reductions |
| Hopcroft–Karp (bipartite matching) | `V, E` | `O(E √V)` | `O(V + E)` | phase count |
| Topological sort / cycle detection | `V, E` | `O(V + E)` | `O(V + E)` | adjacency traversal |
| Forward or backward CPM pass | `m, E` | `O(m + E)` | `O(m)` | predecessor lists |
| Min cut on critical subnetwork | `m, E` | same as the chosen max-flow solver | `O(m + E)` | inherits the flow solver |
| EDD / SPT / Smith sort | `n` | `O(n log n)` | `O(n)` | comparison sort |
| Moore–Hodgson | `n` | `O(n log n)` with a heap | `O(n)` | heap operations on processing times |
| Johnson's rule | `n` | `O(n log n)` | `O(n)` | two sorts and a merge |
| `P||C_max` exact | `n`, machines `k` | exponential in `n` | exponential | state or branching |
| `J||C_max` exact | `n`, `k` | exponential | exponential | disjunctive branching |

**Comment on `O(F · E)`.** The pseudo-polynomial bound is the reason a naive implementation can look fine on test data and fail on production data. On a network with `V = 2000`, `E = 12000`, and `F = 5,000,000` items/day, unit-push augmentation performs about `5 × 10^6` augmentations at `O(E) = 12,000` each, i.e. `6 × 10^10` arc touches, roughly 60 seconds at 1 ns per touch. The same instance under Dinic has a worst-case bound of `V²E = 4 × 10^6 × 12,000 = 4.8 × 10^10`, which looks similar but is attained only on adversarial graphs; in practice it runs in seconds. The engineering conclusion: choose the algorithm by the shape of its bound, and then measure.

### 9b. Back-of-envelope sizing

```text
FORMULA [NET-16]: memory for a residual network  M ≈ E · (2 · 8 + 2 · 4) bytes
WHERE: per stored arc, 8 bytes flow, 8 bytes capacity, 4 bytes head index,
       4 bytes next-pointer; the factor 2 accounts for the reverse arc.
CONSTRAINT: flat CSR-style arrays; object-per-arc representations multiply by 5–10.
NUMERICAL EXAMPLE: E = 12,000 → M ≈ 12,000 · 24 = 288,000 bytes ≈ 0.27 MiB.
                   E = 1,200,000 → 28.8 MB; still trivial beside the input data.
INTERPRETATION: graph state is never the memory problem; the input extract and the
                audit log are, which is where optimisation effort should go.
```

**Workload.** A planning cycle with `V = 2,000` locations, `E = 12,000` lanes, an average of 6 items/day per lane and a ceiling of 60. Raw numeric payload is `12,000 × 16 = 192,000` bytes for capacity and cost, plus identifiers; the extract is on the order of a few MiB of JSON. These are `ILLUSTRATIVE` constructed figures, computed not measured.

**Compute.** For the flow solve, the dominant term is shortest-path work per augmentation; with `F = 5,000,000` and an average push of 40 units (not 1), augmentations number about `125,000`, each costing `E + V log V ≈ 12,000 + 2,000 × 11 = 34,000` operations, so `4.25 × 10^9` operations, a few seconds. The lesson is that the *average augmentation size* matters more than `F`: a solver that picks high-bottleneck paths converts a 60-second bound into a 4-second run.

**Scheduling side.** `n = 5,000` jobs: EDD sort is `5,000 × log2(5,000) ≈ 5,000 × 12.29 = 61,450` comparisons, under a millisecond. Moore–Hodgson adds a heap per late event, at most `n` pushes and pops, so `n log n` again. The scheduling arithmetic is free; the expensive parts are the calendar expansion (5,000 jobs × shift patterns), eligibility checks, and the human review.

**Economics with explicit assumptions.** Suppose one avoided infeasible plan avoids a replan costing 4 hours of planner time at 60 currency units per hour, i.e. 240 currency units, and the new cut-based plan avoids 10 such replans per month: `10 × 240 = 2,400` currency units per month. Annualised, `2,400 × 12 = 28,800`. Against a solver licence and integration effort in the low tens of thousands, the payback is inside a year. Every figure here is `ILLUSTRATIVE`; the structure of the calculation, not its magnitude, is the reusable part.

**Where the cost actually goes.** Upstream data quality (unit consistency, arc direction, lower bounds), the audit artefact (which cut, which slack table), and the human review of a plan the floor did not write. A team that micro-optimises the augmentation loop while the arc direction is ambiguous has optimised the cheap part of the system.

**What is not measured here.** No benchmark of a commercial solver, no measured comparison of Dinic against push–relabel on real instances, no claim that any of the figures above reflects an actual deployment. They establish orders of magnitude and a structure for the estimate.

### 9c. Cost of getting the model class wrong

| Misclassification | Typical consequence | Arithmetic shape of the loss |
|---|---|---|
| Max flow used where commodities differ | plan infeasible in execution | loss ≈ fraction of volume replanned × cost per replan |
| `F2` rule applied to `F3` | makespan above achievable | measured gap, commonly a few percent, unbounded in principle |
| Deterministic `D` promised as a date | systematic overrun on the tail | loss ≈ `(P90 − P50)` of the duration distribution per project |
| `C_max` optimised for a service | waiting time rises while the batch improves | loss ≈ `Δ`mean flow time × volume × value per unit time |
| Crashing without recomputation | spend with zero duration gain | loss = full crash cost of the non-binding set |

**Reading the table.** Each row converts a modelling error into an order-of-magnitude loss estimate, which is the only way to prioritise which error to prevent first. The largest rows are the ones where the error is systematic rather than instance-specific, because those recur every cycle.

## 10. Key Theorems / Results Reference

| Result | Statement usable in a review | Hypotheses | What it licenses | What it does not licence |
|---|---|---|---|---|
| Max-flow min-cut | the maximum flow value equals the minimum cut capacity | single commodity; finite capacities; flow conservation | quoting a throughput with a matching partition as proof | anything about multi-commodity or cost-minimising routing |
| Flow integrality | integral capacities admit an integral maximum flow | integer capacities; single commodity; LP or combinatorial algorithm | integer shipment plans without rounding | integrality for multi-commodity or side-constrained variants |
| Termination bound | with integer capacities each augmentation adds ≥ 1, so at most `v_max` augmentations occur | integer capacities | a correctness argument for naive augmentation | a complexity claim independent of capacity magnitude |
| Strong LP duality | primal optimum equals dual optimum when both are feasible | convex LP, finite optimum | pricing supply and demand at the optimum | integer problems, where a duality gap can exist |
| Total unimodularity of the assignment matrix | the assignment polytope has integral vertices | bipartite structure; row/column sums of 1 | solving assignment as an LP | general set-packing or odd-cycle structures |
| EDD optimality | sequencing by non-decreasing due date minimises `L_max` | one machine; all released at 0; no preemption | a zero-lateness plan when feasible | average or weighted objectives |
| Moore–Hodgson | the scan-and-delete rule minimises the number of late jobs | one machine; all released at 0; unweighted count | the largest possible on-time set | weighted counts, or minimising total tardiness |
| Johnson's rule | partition and sort yields a minimum makespan | exactly two machines; same order for all jobs | `F2` makespan optimality | three or more machines, or job-specific routings |
| CPM longest path | project duration equals the longest path in the precedence DAG | acyclic; deterministic durations | duration and slack as path lengths | a date commitment under duration uncertainty |
| Crashing as min cut | the cheapest one-unit compression is a min cut on the critical subnetwork | linear independent crash costs; single step | pricing compression at the margin | multi-step plans without recomputation, or coupled costs |
| Job-shop hardness | `J2||C_max` and `P2||C_max` are NP-hard | standard machine environments | refusing an optimality claim for a heuristic | that heuristics are useless; they carry bounds |
| König's theorem | in a bipartite graph, maximum matching size equals minimum vertex cover size | bipartite | a cover as a certificate for a matching | general graphs, where the odd-cycle gap appears |

**On sourcing.** The derivations in §4 stand on their own: each is a short exchange argument, a telescoping sum, or a reachability argument, and a reader can check them without a library. The classical attributions — Ford–Fulkerson, Edmonds–Karp, Dinic, Johnson, Moore–Hodgson, the CPM/PERT literature, and the hardness results for shop scheduling — are standard and appear in any graduate scheduling or network-flow text. **Check chapter and theorem numbering in the copy you actually hold**; editions differ, and a citation the reader cannot find is worse than a named result.

**What none of these theorems says.** None of them asserts anything about a specific plant. Interval-graph perfection, flow integrality, and EDD optimality are statements about model classes, and every one of them has hypotheses that real operations violate: qualifications differ between machines, setups depend on sequence, durations are estimates. Quoting a theorem without its hypotheses is the most sophisticated-looking error in this subject and it leaves no trace in the output.

## 11. Interview / Exam Questions

### Level 1 — Definition and recall

#### Q01 — Why is a cut's capacity the sum of arcs from S to T only, and what goes wrong if T-to-S arcs are included?

**Answer.** Flow crossing from `T` back into `S` reduces the net value, so it cannot be part of an upper bound on the value: the identity is `v = f(S,T) − f(T,S)`, and only `f(S,T) ≤ u(S,T)` is available. Including `T → S` capacities inflates the bound, so the reported "minimum cut" is too large and no longer equals the flow, which destroys the certificate. On the §5a network, `S = {1,2,3}`, `T = {4,5,6}` has forward capacity `5 + 7 = 12`; the reverse arcs `4 → 5` (4) would make it 16, and 16 is not a bound on anything.
**Reasoning.** The proof works by summing conservation over `S`; cancellation leaves forward minus backward, and dropping a negative term is what produces the inequality.
**Follow-up.** What happens with anti-parallel arcs `a → b` and `b → a` both present?
**Red flag.** "A cut is all the arcs between the two sides" — that is the definition of a boundary, not of cut capacity.
**Scoring.** 3 = identity plus why the negative term must be dropped plus a number; 2 = identity only; 1 = "the arcs crossing".

#### Q02 — What exactly does the backward residual arc represent, and why is it necessary?

**Answer.** It represents the permission to reduce flow already committed on the forward arc, up to the amount currently carried. Without it, an augmentation cannot reroute: the first path chosen is frozen and the algorithm can stop at a non-maximum flow. Concretely, on a path where the forward arc is saturated but a previously used arc must be released to make room, the backward arc is the only way to express the swap.
**Reasoning.** The residual graph encodes the set of feasible local modifications of the current flow; backward arcs correspond to reducing a variable, forward arcs to increasing it.
**Follow-up.** If the flow is integral, is the backward residual capacity always an integer?
**Red flag.** "It lets the algorithm go backwards" with no statement of what is being undone.
**Scoring.** 3 = undo permission plus rerouting consequence; 2 = undo permission; 1 = "reverse edges".

#### Q03 — State the difference between lateness and tardiness, and give an instance where using one for the other changes the ranking of two schedules.

**Answer.** `L_j = C_j − d_j` is signed; `T_j = max(L_j, 0)` truncates at zero. Using `ΣL_j` lets an early job's negative lateness cancel a late job's positive lateness, so a schedule with one catastrophic breach can look better than a schedule with two mild breaches. With completions `(2, 20)` and dues `(10, 10)`, `ΣL = −8 + 10 = 2` while `ΣT = 0 + 10 = 10`; with completions `(12, 14)` and the same dues, `ΣL = 2 + 4 = 6` while `ΣT = 2 + 4 = 6`. The first schedule looks better under `ΣL` (2 versus 6) and worse under `ΣT` (10 versus 6).
**Reasoning.** The truncation is what makes tardiness a service measure; the signed sum is an algebraic convenience that hides exactly the breaches contracts are written about.
**Follow-up.** Which one does EDD optimise?
**Red flag.** "They're the same up to sign" — they are equal only when nothing is early.
**Scoring.** 3 = definitions plus the reversal plus numbers; 2 = definitions plus reversal; 1 = definitions.

#### Q04 — In the `α|β|γ` notation, why does `P||C_max` differ in difficulty from `1||C_max`?

**Answer.** `1||C_max` is trivial: with one machine the makespan is the sum of processing times regardless of order, so every sequence is optimal. `P||C_max` on identical parallel machines asks for a partition of the job set into `k` groups minimising the largest group sum, which contains PARTITION as the case `k = 2`, so it is NP-hard. The single letter identifying the machine environment carries the entire tractability difference.
**Reasoning.** Adding machines removes the triviality of "sum of all durations" and replaces it with a balancing problem whose decision version encodes a known hard problem.
**Follow-up.** What does `Q||C_max` add, and does it make the problem easier?
**Red flag.** "More machines is just more of the same" — the objective changes character from sum to max.
**Scoring.** 3 = triviality of `1||` plus partition reduction; 2 = one of the two; 1 = a difficulty adjective with no reason.

### Level 2 — Derivation and application

#### Q05 — On the §5a network, recompute the maximum flow after raising `u(1→2)` from 8 to 20. Explain the result.

**Answer.** The minimum cut `{1,2,3}` has capacity `u(2→4) + u(3→5) = 5 + 7 = 12`, and none of the arcs in it is `1 → 2`. Raising `1 → 2` to 20 therefore leaves the max flow at 12: node 2 already receives all the flow it can forward, since `2 → 4` carries 5 and `2 → 3` carries 3. The added capacity is stranded. To raise throughput you must raise an arc in *every* minimum cut; here the candidates are `2 → 4` and `3 → 5`, and either alone raises the value by one-for-one up to the next binding limit.
**Reasoning.** The value is bounded by every cut, so a change outside a particular minimum cut can still matter only if it raises *all* minimum cuts. `1 → 2` is in no minimum cut, so it cannot help.
**Follow-up.** Raise both `2 → 4` and `3 → 5` by 4 each; what is the new value and which cut now binds?
**Red flag.** "More capacity upstream means more flow" — this is exactly the intuition the cut refutes.
**Scoring.** 3 = cut identification plus the "every minimum cut" rule plus the arithmetic; 2 = cut identification; 1 = "12 because the cut is 12" without explaining the insensitivity.

#### Q06 — Derive the assignment lower bound and verify it on the §5b matrix.

**Answer.** For any perfect assignment, `Σ c_ij ≥ Σ_i rowmin_i + Σ_j colmin_j`, because each assigned entry is at least its row minimum and at least its column minimum, and the two sums each count `n` entries... the precise statement: after subtracting row minima the matrix has a zero in each row, and any assignment costs at least the sum of the row minima; applying column minima to the reduced matrix adds a further lower bound. On the instance, row minima are 2, 3, 1 (sum 6) and the reduced matrix `[7 0 5; 3 1 0; 4 7 0]` has column minima 3, 0, 0 (sum 3), giving bound 9. The zero-cost assignment `W1→J2, W2→J1, W3→J3` attains 9, so the bound is tight.
**Reasoning.** The two reduction steps are the first two phases of the Hungarian algorithm, and their sum is the dual objective at the current dual solution; a dual objective equal to a primal cost proves optimality by weak duality.
**Follow-up.** What does it mean if the reduced matrix has no zero-cost perfect assignment?
**Red flag.** Taking a lower bound as the answer without producing the attaining assignment.
**Scoring.** 3 = both reductions plus the attaining assignment plus duality; 2 = reductions with numbers; 1 = "try all permutations".

#### Q07 — A project has two critical paths of length 14. You crash an activity of cost 300 that lies on one of them. What happens to the duration, and what was bought?

**Answer.** Nothing happens to the duration: the other path is still 14. What was bought is 300 currency units of spend and a reduction of one path's length to 13, which converts the crashed activity into a non-critical one with slack 1. The consequence is worse than no action: the next crash must now cut the remaining path, and if the crashed activity was the cheap member of a set that used to cover both paths, the total cost to reach 13 has risen. The correct first step is a cut over the critical subnetwork, which in §5c is `{A}` at 500.
**Reasoning.** The duration is a maximum, so reducing one argument of the maximum leaves it unchanged unless that argument was uniquely maximal — and here it was tied.
**Follow-up.** After crashing the correct activity, what is the cheapest way to reach 12, and why does that price include two activities?
**Red flag.** "It shortens that path, so it helps a bit" — a maximum is not an average.
**Scoring.** 3 = no change plus the spend as a loss plus the cut reasoning; 2 = no change with the tie explanation; 1 = "duration drops to 13".

#### Q08 — Compute `ΣC_j` for both orders on the §5d instance and state which objective each order optimises.

**Answer.** EDD order `J1,J2,J3,J4,J5` gives completion times 3, 7, 9, 14, 15, so `ΣC_j = 48`. SPT order `J5,J3,J1,J2,J4` gives 1, 3, 6, 10, 15, so `ΣC_j = 35`. SPT minimises `ΣC_j` by the coefficient argument: each processing time is multiplied by the number of completions it contributes to, and those coefficients decrease along the sequence. EDD minimises `L_max`, here 0 versus 2 for SPT. The 27.1% improvement in mean flow `(13/48)` is bought with one late job.
**Reasoning.** Both are exact theorems for their own objective; the instance is chosen so the two optima differ, which is what makes "which objective" the operative question.
**Follow-up.** Compute `Σw_jC_j` for both orders with weights `(1,1,5,1,1)` and say which wins.
**Red flag.** Reporting one order as "better" without naming the objective.
**Scoring.** 3 = both sums, both objectives, and the coefficient argument; 2 = both sums; 1 = one sum.

#### Q09 — Apply Moore–Hodgson to the six-job instance and state what the dropped job represents operationally.

**Answer.** EDD order is `J1, J2, J3, J6, J4, J5` with cumulative completions 3, 7, 9, 15, 20, 21. When `J6` completes at 15 against a due date of 12, delete the longest job among the four scanned, which is `J6` itself at 6 days; cumulative falls to 9. `J4` then completes at 14 (on time) and `J5` at 15 (on time). Result: one late job, 15 of 21 processing days scheduled. Operationally the dropped job is a capacity statement: 6 days of work do not fit inside the due-date structure and must be deferred, outsourced, or renegotiated — not a scheduling failure to be retried.
**Reasoning.** The rule's exchange argument shows the on-time set can be made maximal by deleting the largest processing time at each breach, because deleting a larger job frees more time for all later jobs.
**Follow-up.** What changes if `J6` carries weight 10 and the objective becomes weighted late count?
**Red flag.** Deleting the job that breached rather than the longest scanned job — different rule, no optimality guarantee.
**Scoring.** 3 = full trace plus the operational reading; 2 = trace; 1 = "drop the late job".

#### Q10 — Verify the §5e Johnson sequence against the lower bound, and say what the bound would be for a different instance.

**Answer.** `Σp1 = 3+5+2+6+1 = 17` and `Σp2 = 4+2+6+3+5 = 20`. The bound is `max(Σp1 + min p2, Σp2 + min p1) = max(17+2, 20+1) = 21`, and the sequence `J5,J3,J1,J4,J2` achieves 21, so it is optimal. The bound is not a property of the rule but of the instance; on a different instance where `Σp1` is much larger than `Σp2`, the first term binds and the achieved makespan will sit closer to `Σp1 + min p2`.
**Reasoning.** Machine 1 must process every job, so the makespan is at least `Σp1`; machine 2 must also process everything, and cannot start before the first job leaves machine 1, which takes at least `min p1`.
**Follow-up.** What does it mean if the achieved makespan exceeds the bound by 3 hours?
**Red flag.** Calling 21 "optimal because Johnson says so" without a bound.
**Scoring.** 3 = both sums, both bound terms, equality; 2 = bound plus achieved; 1 = sequence only.

#### Q11 — Compute the mean WIP of a Gantt schedule with jobs `[0,4)`, `[2,6)`, `[4,8)`, `[5,9)`.

**Answer.** Sweep the interval endpoints: on `[0,2)` depth 1, `[2,4)` depth 2, `[4,5)` depth 1, `[5,6)` depth 2, `[6,8)` depth 1, `[8,9)` depth 1. The area is `2·1 + 2·2 + 1·1 + 1·2 + 2·1 + 1·1 = 2 + 4 + 1 + 2 + 2 + 1 = 12` job-hours over a span of 9 hours, so mean WIP is `12/9 = 1.333` jobs. Makespan is 9 hours. The peak depth is 2, which is the minimum number of resources needed.
**Reasoning.** Mean WIP is area under the concurrency curve divided by the horizon; it equals `Σflow time / makespan` here because all jobs are released at 0.
**Follow-up.** Which of the four jobs would SPT order first, and does that change the peak depth?
**Red flag.** Averaging the four durations (`22/4`) — that is mean duration per job, not WIP.
**Scoring.** 3 = sweep, area, division, and peak; 2 = area and mean; 1 = peak only.

### Level 3 — Judgement, design, and diagnosis

#### Q12 — A planner says "the model says 14 per day but we only ship 12". Walk through your diagnosis in order.

**Answer.** First recompute the cut from the residual reachability of the reported flow and compare its capacity with the reported value; a mismatch is a solver or reporting defect, and it is the cheapest check because it needs no new data. Second, check units: 14 per week versus per day, or a capacity expressed per shift. Third, check whether the shipped 12 is the *flow value* or the *satisfied demand*, since unsatisfied demand downstream of the sink is not a flow problem. Fourth, check whether the real operation has a constraint the model lacks — a dock, a driver, a permit — which shows up as a systematic gap on a specific lane rather than a uniform shortfall.
**Reasoning.** The order is by cost of the check and by base rate: certificate mismatch is arithmetic, units are metadata, demand satisfaction is a modelling boundary, and the missing constraint is a data problem. Each has a different signature — uniform ratio for units, lane-specific for a missing constraint.
**Follow-up.** The gap is exactly 2 every day and appears on one lane. Which cause?
**Red flag.** "The solver must be wrong" before recomputing the certificate.
**Scoring.** 3 = four ordered checks with distinct signatures; 2 = certificate and units; 1 = "check the data".

#### Q13 — Your scheduling plan is optimal for `L_max` and the operations manager wants lower average wait. What do you do?

**Answer.** Quantify the trade first: on §5d, moving from EDD to SPT cuts `ΣC_j` from 48 to 35 (27.1%) and raises `L_max` from 0 to 2 days with one late job. Then make the choice explicit rather than arguing: either the contract measures worst-case lateness, in which case the current plan is right and the wait is the price, or the contract measures mean wait, in which case SPT is right and the breach must be renegotiated. A third option is a constrained formulation — minimise `ΣC_j` subject to `L_max ≤ 0` — which is a different model and should be solved as one, not approximated by a rule.
**Reasoning.** The two objectives are provably optimised by different rules on the same instance, so there is no single schedule that is best for both; the decision is a business choice that must be priced.
**Follow-up.** The manager insists on both. What do you compute next?
**Red flag.** Averaging two schedules or picking a rule by preference.
**Scoring.** 3 = quantified trade plus explicit objective choice plus the constrained alternative; 2 = quantified trade; 1 = "SPT is better for wait".

#### Q14 — You must promise a delivery date for a project whose durations are estimates. What do you deliver?

**Answer.** Not `D = 14` as a date. I deliver the deterministic plan with its assumptions attached, plus a distribution over completion times obtained by propagating the estimate uncertainty through the same network — for example by treating each duration as a range and reporting the distribution of the longest path. The reason is structural: the project duration is a *maximum* over paths, so the uncertainty in `D` is at least the uncertainty of the longest path, and maxima of random variables are biased upward relative to maxima of their means. The commitment should then be a stated quantile with the assumption set, and the plan should carry the activities whose variance most affects that quantile.
**Reasoning.** A maximum of estimates is not the estimate of a maximum; treating `D` as a date silently converts an estimate into a promise and assigns all the risk to the delivery.
**Follow-up.** Which activities would you measure first to tighten the quantile?
**Red flag.** Adding a blanket 20% buffer with no reference to which path dominates the variance.
**Scoring.** 3 = refusal of the point promise plus quantile plus variance attribution; 2 = refusal plus range; 1 = "add a buffer".

#### Q15 — Two warehouses and one lane produce different optimal plans under two cost models. How do you decide which model is right?

**Answer.** I look for the place where the two cost models disagree structurally rather than numerically: if one prices the lane per unit and the other per shipment, the plans will differ only when shipment sizes differ across lanes, and that difference is testable against invoices. I would fit each model to the last quarter of actual shipments and compare predicted against actual cost on held-out weeks; the model with the smaller out-of-sample error wins, not the one with the lower predicted cost. If both fit equally, the decision belongs to whoever owns the contractual cost structure, and the model must carry that attribution explicitly.
**Reasoning.** A cost model is a predictive claim about invoices, so the discriminating evidence is out-of-sample error, not internal consistency or elegance.
**Follow-up.** Both models fit within 2%; what do you do then?
**Red flag.** Choosing the model that produces the lower total cost — that is choosing the answer, not the model.
**Scoring.** 3 = structural disagreement plus held-out comparison plus ownership; 2 = held-out comparison; 1 = "use the more detailed model".

#### Q16 — How would you verify a max-flow implementation you did not write, and what does each layer catch?

**Answer.** Four layers, increasing cost. (1) Hand-checkable fixtures: a path graph, a diamond with equal cuts, a graph where the naive first path is suboptimal, and a graph with a parallel arc; these catch direction, capacity-summing, and reverse-arc errors. (2) Certificate assertions on every output: conservation at every internal node, capacity respect, and cut capacity equal to flow value; these catch almost everything else and cost linear time. (3) Brute-force oracle on random graphs with `V ≤ 8`: enumerate all cuts, find the minimum, compare; this catches subtle residual-graph bugs the certificate cannot see, because a wrong flow can still be locally consistent. (4) Adversarial instances: unit-capacity long chains and the alternating-path construction that forces many augmentations, which catch performance defects masquerading as correctness.
**Reasoning.** The certificate layer is the strongest cheap test because the max-flow min-cut theorem guarantees the two numbers agree at optimality; the oracle is needed because a non-optimal flow can satisfy every local assertion.
**Follow-up.** All four pass and production still shows a throughput lower than predicted. What now?
**Red flag.** "Compare with another library" — two implementations can share a modelling error.
**Scoring.** 3 = four layers with what each catches; 2 = certificates plus oracle; 1 = "write unit tests".

**Answer-quality ledger.**

| Question | Level | Skill | Checked numeric core | Support in this sheet |
|---|---|---|---|---|
| Q01 | 1 | cut definition | 12 versus 16 | §3.1, §4.1 |
| Q02 | 1 | residual semantics | — | §4.1, §6 |
| Q03 | 1 | metric precision | `ΣL = 2` versus `ΣT = 10` | §3.2 |
| Q04 | 1 | notation and hardness | PARTITION at `k = 2` | §7.2, §10 |
| Q05 | 2 | cut insensitivity | value stays 12 | §5a sensitivity |
| Q06 | 2 | dual bound | 6 + 3 = 9 | §4.2, §5b |
| Q07 | 2 | max versus sum | 300 spent, duration 14 | §5c |
| Q08 | 2 | objective conflict | 48 versus 35, 27.1% | §5d, NET-14 |
| Q09 | 2 | rule trace | 15 of 21 days | §5d |
| Q10 | 2 | bound equality | 21 = max(19,21) | §5e, NET-15 |
| Q11 | 2 | WIP integral | 12/9 = 1.333 | §6 |
| Q12 | 3 | ordered diagnosis | 14 versus 12 | §8 F-01, F-02 |
| Q13 | 3 | objective trade | 27.1% against 2 days | §5d |
| Q14 | 3 | commitment design | maximum-of-random bias | §4.4 limit 7 |
| Q15 | 3 | model selection | out-of-sample error | §9c |
| Q16 | 3 | verification design | `V ≤ 8` oracle | §8, §9a |

Sixteen questions: ten require computation, derivation, or diagnosis rather than recall. Every answer stands alone; none is a template with the topic substituted.

## 12. Quick Reference Card

| If the question is… | Build this | Read this certificate | Refuse when |
|---|---|---|---|
| How much throughput can this network sustain? | max flow with Dinic or push–relabel | min cut of equal capacity, found by residual reachability | commodities must remain distinct |
| Where should capacity money go? | min cut, then a re-solve per candidate arc | cut capacity change per unit of spend | the arc appears in only some minimum cuts |
| What is the cheapest routing for a fixed volume? | min-cost flow with successive shortest paths | reduced costs all non-negative, plus node potentials | cost depends on shipment size non-linearly |
| Which worker takes which job? | assignment as LP or Hungarian | row-plus-column reduction bound equal to achieved cost | eligibility is many-to-many |
| When will the project finish, and what can slip? | CPM forward and backward pass | longest path length and per-activity slack | the precedence graph has a cycle |
| What is the cheapest way to save a day? | min cut on the critical subnetwork | cut set and its total marginal cost | crash costs interact across activities |
| In what order should one machine run the jobs? | rule matched to the declared objective | exchange-argument condition, `L_max` or `ΣC_j` value | release dates or weights change the objective |
| Two machines, minimise makespan? | Johnson partition and sort | lower-bound equality `max(Σp1+min p2, Σp2+min p1)` | a third machine exists |
| Three or more machines, minimise makespan? | MIP with a time limit, or a labelled heuristic | incumbent and proven gap | anyone asks for a proof of optimality |
