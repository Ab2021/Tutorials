---
collection: "optimisation-operations-research-and-simulation"
file_id: "07"
topic: "Interval greedy and weighted scheduling"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A change to interval endpoint semantics, objective, resource count, buffer rule, or tie policy."
notation_style: "plain-text"
prerequisites: ["sorting and comparator discipline", "induction and exchange arguments", "single-state dynamic programming"]
pattern_tags: ["interval-scheduling", "earliest-finish", "weighted-dp", "exchange-argument", "interval-graph"]
source_status: "OPEN"
owner_role: "Operations research reviewer"
source_index: "INDEX.md"
---

# Interval Greedy and Weighted Scheduling

## 1. File Header

> **Purpose:** Decide, from the objective and the feasibility relation rather than from habit, whether a single-resource interval request stream needs earliest-finish greedy, weighted interval dynamic programming, an interval-graph colouring argument, or a different model entirely; then produce a deterministic schedule and defend its optimality boundary.
> **Prerequisites:** You can sort by a multi-key comparator and explain why the comparator must be total; you can write an inductive correctness argument; you can implement a one-dimensional dynamic program with back-pointers; you have seen the difference between a decision problem, an optimisation problem, and a feasibility problem.
> **Key outcome:** Given a list of half-open occupancy requests with start instants, finish instants, and additive values, plus a declaration of resource count and buffer policy, you can state the objective, choose the correct algorithm, compute the compatible predecessor indices, produce the optimal value and one optimal schedule, replay the schedule for feasibility, and name the exact single change in the input contract that would invalidate your answer.
> **This file does NOT cover:** multi-machine assignment with per-machine eligibility, sequence-dependent setup times, precedence-constrained project scheduling, stochastic or resumable durations, preemption, release dates combined with deadlines in the weighted case, or any measured workforce performance figure. Those are handoffs, listed in Section 7 and Section 12.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not independently reviewed; the review field stays `null` until a reviewer actually signs off.
> **Evidence:** DERIVED for the greedy exchange argument, the dynamic-programming recurrence, and the interval-graph colouring identity; VERIFIED for every arithmetic trace in Sections 5, 8, and 9 (numbers were recomputed by exhaustive subset enumeration on twelve jobs); ILLUSTRATIVE for the job set, the value units, and all monetary conversions; OPEN for the original interview-source body and for any claim that a named organisation uses this method.
> **Retrieval cue:** If a high-value long job competes with several short jobs and someone proposes “sort by finish time”, go directly to Section 4.4, then to Section 8 row F-03.

### What decision this sheet supports

A service business receives a stream of requests to occupy one resource for an interval: a bay in a workshop, a courtroom, an analyst's afternoon, a single interview room, a machine on a shop floor, a licensed channel slot. Each request arrives with a start, a finish, and some worth. The planning question is never “which algorithm is fashionable”; it is “what am I maximising, and what makes two requests mutually exclusive”. Those two answers select the method, and getting either wrong produces a schedule that looks tidy, runs quickly, and is wrong in a way nobody notices until the revenue report or the complaint log disagrees with the plan.

Three answers cover most real requests, and they are genuinely different problems:

1. **Maximise the count of accepted requests** with equal worth and one resource. Earliest-finish greedy is optimal here and admits a short exchange proof.
2. **Maximise the total additive value** of accepted requests with one resource. Greedy is not optimal; a one-dimensional recurrence over finish-sorted jobs is.
3. **Serve everything, minimising the number of resources.** This is an interval-graph colouring problem, and the answer is the maximum number of simultaneous requests.

The sheet treats the first two in full because they are the pair most often confused, and treats the third because it is the cheapest correct answer to “how many people do we need”, which is the question a manager actually asks after seeing the plan.

### What “correct” means here

Correctness is always relative to a stated model. A schedule is correct for this sheet when all of the following hold, and it is not correct merely because it renders without visible overlap:

- every accepted request occupies exactly its declared half-open interval `[s_i, f_i)` on exactly one resource;
- no two accepted requests overlap under the declared endpoint convention, including any declared turnaround buffer;
- every rejected request was rejected because accepting it would have violated either a capacity or a feasibility condition, not because the scan happened to have moved past it;
- the claimed objective value equals the sum of accepted values, recomputed from the accepted list rather than copied from a running variable;
- repeating the run with the same input produces the same schedule, because ties are broken by a declared total order.

The last point is not pedantry. A scheduler whose output depends on the physical order in which equal-finish requests happened to arrive cannot be reconciled against a previous day's plan, cannot be regression tested, and cannot be explained to a dispatcher who asks why Tuesday's schedule differs from Monday's for identical work.

### Source boundary

The collection source inventory is an index of topics, not access to any original text. This file therefore states mechanisms and derives results from first principles, labels all constructed job sets as ILLUSTRATIVE, and leaves the original interview-source body marked OPEN. Textbook attributions appear in Section 10 with an explicit warning to check the edition before quoting a theorem number. Nothing in this file is a claim about a real company's dispatch system, measured utilisation, or realised savings.

**Takeaway:** The header's job is to make the reader unable to apply the method without first naming the objective, the endpoint convention, and the resource count.

## 2. Concept Map / Mental Model

The map below is a decision flow, not a data flow: each arrow means “this answer to the question selects that method”. The diagnostic loop at the bottom is the part most practitioners skip, and it is the part that catches a wrong answer after the fact.

```text
                 [request list: (s_i, f_i, w_i) + resource count k]
                                      |
                                      v
                 [declare endpoint convention and buffer g]
                    (half-open [s,f)?  turnaround after finish?)
                                      |
                                      v
                    [declare objective: count | additive value | latency]
                    /                 |                        \
             count, k=1          value, k=1                serve-all, k free
                 |                    |                         |
                 v                    v                         v
        [sort by finish]      [sort by finish]           [max simultaneous depth]
                 |                    |                         |
                 v                    v                         v
        [scan, accept if      [compute p(i) by            [k* = max depth;
         s_i >= last_f]        binary search]              greedy by start
                 |                    |                     colours with k*]
                 v                    v
        [greedy schedule]     [OPT(i)=max(skip, take)]
                 |                    |
                 |                    v
                 |             [reconstruct with tie policy]
                 \                   /
                  \                 /
                   v               v
              [feasibility replay on the accepted list]
                          |
                          v
              [objective recomputed from the list]
                          |
                 mismatch? --yes--> [contract defect: endpoints,
                          |          buffer, resource count, or p(i)]
                          no
                          v
                    [publish + audit record]
```

Read the three-way split as the central claim of the sheet: the objective and the resource count, not the shape of the data, determine the algorithm. Two request lists with identical columns can require three different solvers.

### The one invariant that carries the whole sheet

Earliest-finish greedy works because of a single monotone quantity: the finish time of the last accepted request. That number only ever moves right, and moving it right as slowly as possible leaves the largest possible remaining window. Every later decision is judged against it. When the objective becomes additive value, that scalar is no longer sufficient, because a job that finishes slightly later may be worth far more; the state must be widened from “one number” to “best value achievable using the first `i` jobs”. The recurrence in Section 4.2 is exactly the smallest widening that restores correctness, and it is smallest in a precise sense: it keeps finish order and adds only a prefix index.

### What changes between the variants

| Variant axis | What the change does to the mechanism | Where it is handled |
|---|---|---|
| objective: count to value | one-number state becomes a value table | Section 4.2 |
| add a turnaround buffer | every compatible-predecessor link moves earlier | Sections 4.3, 5, 8 F-02 |
| one resource to `k` resources | single last-finish becomes an unordered `k`-tuple of finishes | Sections 4.3, 7 |
| fixed intervals to release dates plus deadlines | feasibility is no longer determined by a scan | Section 7, out of scope |
| additive value to order-dependent cost | the recurrence's separability assumption fails | Sections 4.4, 8 F-11 |

**Takeaway:** One monotone state variable explains greedy; widening that state to a value table explains the dynamic program; adding dimensions to it explains why multi-resource versions stop being a two-line recurrence.

## 3. Foundational Definitions

Definitions come before formulas because the most common production failure in this topic is not an arithmetic error — it is two teams using the same word for incompatible objects. “Booking”, “slot”, “shift”, and “job” are used interchangeably in conversation and then differ in the code that computes them.

### 3.1 Core record types

| Term / symbol | Formal meaning | Units / type | Valid domain | Common confusion | Where used |
|---|---|---|---|---|---|
| request (job) `i` | a triple `(s_i, f_i, w_i)` with `s_i < f_i` | instants and value units | `s_i, f_i` are absolute instants; `w_i` finite | an order line is not a request; one order may need several intervals | §4, §5 |
| `s_i` | start instant, inclusive | timestamp (UTC) | `s_i < f_i` | a date with no timezone is not an instant | §4.1 |
| `f_i` | finish instant, exclusive | timestamp (UTC) | `f_i > s_i` | “end date” in a calendar system may be inclusive | §3.2 |
| `w_i` | additive value of accepting request `i` | value units (currency, priority points, minutes saved) | finite; sign policy must be declared | a rank is not a value; ranks do not add | §4.2 |
| `g` | turnaround buffer required after occupancy | duration (same unit as `f_i - s_i`) | `g >= 0` | cleaning time is not slack in the job duration | §4.3 |
| `k` | number of identical resources | count | integer `k >= 1` | “three bays” may be three non-identical bays | §4.3 |
| `n` | number of requests considered | count | `n >= 0` | the input row count is not `n` after deduplication | §9 |

A request is not a task description. It is a claim on a resource for a measured interval, and it is only usable when both endpoints are absolute instants with a declared timezone. The temptation to store `2026-11-09` as an endpoint is the origin of a whole failure family, because a date is a 24-hour bucket and an instant is a point; comparing buckets to points produces intervals that are internally inconsistent by up to a day.

### 3.2 Compatibility, the definition everything else depends on

Two requests `i` and `j` are **compatible** under half-open semantics with buffer `g` when

```text
compatible(i, j; g)  <=>  f_i + g <= s_j   OR   f_j + g <= s_i
```

The relation is symmetric, irreflexive in the strict sense that a request is never compatible with itself for scheduling purposes, and — importantly — **not transitive**. Non-transitivity is the reason greedy needs a proof rather than an intuition: from `A` compatible with `B` and `B` compatible with `C` you may not conclude `A` compatible with `C`. A concrete instance: `A = [0,10)`, `B = [10,20)`, `C = [0,1)`; `A` is compatible with `B`, `B` is compatible with `C`, and `A` overlaps `C` completely.

Half-open intervals are the default because they make adjacency compatible: `[09:00,10:00)` and `[10:00,11:00)` both hold the resource, but never at the same instant, so a resource can be handed over at the boundary with zero idle time. That is the right model for a room, a machine, or a call slot. It is the wrong model whenever the business requires a physical turnaround, a cleaning pass, documentation, or travel; in those cases `g > 0` and adjacency is a conflict. Choosing `g = 0` when the operation actually needs ten minutes is not a modelling simplification, it is an overbooking bug that the algorithm will faithfully optimise.

Closed intervals `[s_i, f_i]` are a different model and produce different answers: under closed semantics `[09:00,10:00]` and `[10:00,11:00]` conflict at the shared instant 10:00. Both conventions are defensible; what is not defensible is mixing them between the feasibility checker and the optimiser. A schedule computed under half-open semantics and validated under closed semantics will fail validation on every back-to-back pair, and the natural (wrong) response is to insert gaps until validation passes, silently degrading the plan.

### 3.3 Finish order and the predecessor function

**Finish-sorted order** means the request list is indexed `1..n` such that `f_1 <= f_2 <= ... <= f_n`, with ties broken by a declared total order, canonically `(f_i, s_i, request_id)`. Finish order is not start order, and the difference matters: two requests can start in one order and finish in the other, and the greedy argument is stated in finish order.

The **compatible predecessor** of `i` is

```text
p(i) = max{ j : 1 <= j < i and f_j + g <= s_i },   with p(i) = 0 when the set is empty
```

`p(i)` is the largest *index* whose request can precede `i`, which — because the list is finish-sorted — is also the request with the latest finish among those compatible with `i`. This last property is what makes the recurrence valid: if `p(i)` is feasible before `i`, then every `j <= p(i)` is also feasible before `i`, since finishes are non-decreasing in index. The set of indices compatible with `i` is therefore always a prefix `{1, ..., p(i)}`, never a scattered subset. That prefix property is the structural fact that turns an exponential search into a table lookup.

Two errors are common here and both are silent. The first is computing `p(i)` with a linear scan that stops at the first failure, which is wrong when start times are not monotone: the scan finds a compatible job, not the largest index compatible. The second is using `i - 1` as the predecessor, which asserts that consecutive jobs in finish order are compatible; they are not, and the resulting “optimal” value is an upper bound on reality.

### 3.4 Value, weight, and what additivity buys

`w_i` is the **additive** value of accepting request `i`: the value of a set `S` of accepted requests is `sum_{i in S} w_i`. Additivity is a modelling commitment, not a mathematical convenience. It is violated when:

- two accepted jobs interact, for example a discount that applies only when both are accepted (superadditive) or a fatigue penalty for the third consecutive job (subadditive);
- value depends on which resource or position a job occupies, as with sequence-dependent setup;
- value is capped per customer, so the tenth accepted job from one client contributes nothing;
- value is a probability of downstream success that is not independent across jobs.

Under any of these, the recurrence in Section 4.2 computes the optimum of a *different* problem than the one the business cares about, and it will do so flawlessly. The definition to carry: additivity is the assumption that the whole is exactly the sum of the parts, and it must be stated in the model record, not assumed silently.

Values may legitimately be negative. A negative `w_i` means accepting the job costs more than it returns; a correct model then never accepts it, and the recurrence's `max` handles that automatically. What breaks is the *interpretation* that `OPT(i)` is non-decreasing in `i`: with a negative weight the table can still be non-decreasing (because skipping is always available), which is precisely why a non-decreasing table is not evidence that all weights were positive. The table is non-decreasing because skip is always feasible, given `OPT(i) >= OPT(i-1)`.

### 3.5 Determinism, ties, and the audit record

A **tie** exists when two distinct schedules achieve the same objective. Ties are not rare; on the twelve-job example in Section 5, six different four-job schedules all achieve the maximum cardinality. Ties are also invisible in the objective value, so a report that publishes only the value hides the fact that the chosen schedule is one of many.

A **tie policy** is a declared rule that selects one schedule from the tied set. Two policies are conventional here: *skip on equality* when comparing `w_i + OPT(p(i))` against `OPT(i-1)`, which prefers the solution built from earlier-indexed jobs, and *take on equality*, which prefers later jobs. Either is fine; an undeclared policy is not. The policy must appear in the output metadata, because the schedule is a business artefact that outlives the run.

An **audit record** for a published schedule contains at least: the input snapshot identifier, the endpoint convention and buffer value, the resource count, the comparator keys in order, the tie policy, the objective value recomputed from the accepted list, and the count of rejected requests grouped by rejection reason. Without the rejection reasons, “rejected” is indistinguishable from “dropped by a bug”.

### 3.6 Objective vocabulary

| Objective | Formal statement | Greedy correct? | Typical phrasing in a request |
|---|---|---|---|
| cardinality | maximise `|S|` subject to feasibility | yes, earliest finish | “fit as many meetings in as possible” |
| additive value | maximise `sum_{i in S} w_i` | no | “maximise revenue from the day” |
| resource count | minimise `k` such that all requests fit | yes, max depth | “how many technicians do we need” |
| maximum lateness | minimise `max_i (C_i - d_i)` | yes, earliest due date | “no order ships late if avoidable” |
| tardy count | minimise number of jobs with `C_i > d_i` | not by plain EDD | “minimise late orders” |
| total busy time | maximise `sum (f_i - s_i)` over accepted | no | “keep the machine loaded” |

The last row is the one that surprises people: maximising occupied time is a weighted problem with `w_i = f_i - s_i`, and earliest-finish greedy does not solve it. Two short jobs occupying 8 units of a 10-unit window beat one long job occupying 6 units that starts earlier.

**Takeaway:** Compatibility, finish order, additivity, and tie policy are four separate contracts; a schedule is only as trustworthy as the weakest of them.

## 4. Core Formulas with Full Derivation

Every result below states its validity conditions before the formula, because a copied formula without its contract is the single most common source of wrong schedules. Formula identifiers are stable and referenced from later sections.

### 4.1 Simplest / Base Case: Cardinality Greedy

**Validity before use.** Exactly one resource. Every request is known in advance with fixed endpoints. All requests have equal worth, so the objective is cardinality. Compatibility is non-overlap under one declared endpoint convention with buffer `g`. Nothing about the request stream changes during the scan. Under these conditions earliest-finish greedy returns a maximum-cardinality feasible set.

**Mechanism.** Sort the requests into finish order. Walk that order once, maintaining the finish instant of the last accepted request. Accept a request whenever it starts at or after that instant plus the buffer; otherwise reject it and continue. The scan never reconsiders a rejection, which is exactly why the algorithm needs a proof rather than a heuristic pedigree.

**Derivation by exchange at the level of sequences.** Let greedy accept requests in scan order, so greedy's accepted sequence is `i_1 < i_2 < ... < i_k` in index order. Let an optimal solution contain `m` requests and list them in finish order as `j_1 < j_2 < ... < j_m`. Both index sequences are increasing for the same reason: within a feasible set, finish order and index order coincide.

Claim A — greedy is never behind: `f_{i_r} <= f_{j_r}` for every `r <= min(k, m)`.

*Base.* The first job in finish order is index 1, and greedy accepts it unconditionally because the sentinel last-finish is `-infinity`. Hence `f_{i_1} = f_1 <= f_{j_1}`.

*Step.* Assume `f_{i_{r-1}} <= f_{j_{r-1}}`. Because `f` is non-decreasing in index, `i_{r-1} <= j_{r-1}`. Because the optimal set is feasible, `f_{j_{r-1}} + g <= s_{j_r}`. Substituting the induction hypothesis gives `f_{i_{r-1}} + g <= s_{j_r}`, so job `j_r` was compatible with greedy's state at the moment greedy scanned it, and `j_r` is scanned after `i_{r-1}` because `j_r > j_{r-1} >= i_{r-1}`. Greedy therefore accepted either `j_r` itself or some job that it reached before `j_r` and that was compatible, and any job it accepts finishes no later than `f_{j_r}` because all jobs in `(i_{r-1}, j_r]` have finish at most `f_{j_r}`. Hence `f_{i_r} <= f_{j_r}`.

Claim B — greedy never accepts fewer: if `k < m`, then after greedy's scan ended at job `i_k`, the optimal job `j_{k+1}` was still unscanned (since `i_k <= j_k < j_{k+1}`) and satisfied `s_{j_{k+1}} >= f_{j_k} + g >= f_{i_k} + g`, so greedy would have accepted it. Contradiction. Therefore `k >= m`, and with `m` optimal, `k = m`.

Claim A and Claim B together prove optimality of the count. Note what the proof did **not** use: no numeric bound on values, no assumption about start-time order, and no assumption of distinct finish times. It used only finish monotonicity in index and the prefix-compatibility property that a feasible set can be ordered by finish.

```text
FORMULA [SCH-01]: accept i iff s_i >= L + g ; then L <- f_i ; L_0 = -infinity
WHERE: s_i, f_i are absolute instants; L is the finish of the last accepted request;
       g is the turnaround buffer as a duration (g = 0 for pure half-open handover).
CONSTRAINT: requests pre-sorted by (f_i, s_i, id); one resource; equal values;
            no preemption; no release-date interaction (release dates are irrelevant
            here because the resource is idle at -infinity).
NUMERICAL EXAMPLE: L = 10:00, g = 0, request [10:00, 10:30):
                   s_i = 10:00 >= 10:00 + 0, so accept and set L = 10:30.
                   Next request [10:15, 10:45): 10:15 >= 10:30 is false, so reject.
INTERPRETATION: the decision is a single comparison against a monotonically
                advancing scalar; no lookahead is required and none is permitted.
```

**Boundary and counterexample for 4.1.** Set `g = 10 minutes` on the same two requests and the first acceptance is unchanged while the second rejection still holds; but consider `[10:00,10:30)` then `[10:40,11:00)`: with `g = 0` the pair is compatible, with `g = 10 min` the pair conflicts because `10:30 + 10 min = 10:40` and the comparison `10:40 >= 10:40` is true, so it remains compatible — the boundary case is exactly the one where the buffer equals the idle gap. Drop the buffer to 11 minutes and the pair conflicts. This is the knife-edge that produces “it worked in the workshop and failed in the clinic”: the clinic's turnaround was longer than anyone had written down.

The other boundary is equal finish times. If `f_a = f_b` and `s_a < s_b`, the comparator `(f, s, id)` places `a` before `b`. Both are eligible at the same moment, but only one can be first, and the tie policy decides which. Under cardinality this cannot change the count — either choice leaves the same last-finish — but it changes *which* work the business sees, which is why the tie policy is a business decision wearing engineering clothes.

**Takeaway from 4.1:** Greedy's optimality is a consequence of a specific exchange argument, not of sorting by finish being aesthetically reasonable; remove any premise and the argument lapses, even if the code keeps returning answers.

### 4.2 General Case: Weighted Interval Scheduling

**Validity before use.** One resource; fixed known intervals; values are additive over accepted requests; skipping any request is allowed; the objective is the sum of accepted values. The recurrence returns an exact optimum for this model. It says nothing about uncertain execution, cancellations, or values that interact.

**Why the greedy scalar is insufficient.** The greedy state is a single number `L`. Under additive value that number is not enough, because two partial schedules with the same `L` can have different accumulated value, and the one with higher value is not always the one that will end up better — it may have consumed a high-value job that blocks two moderate ones. The state must therefore record, for each prefix, the best value achievable. The widening is minimal: keep finish order, and replace the scalar with a table indexed by prefix position.

**The predecessor.** For the recurrence to be a table lookup rather than a search, the set of eligible predecessors must form a prefix. Define

```text
FORMULA [SCH-02]: p(i) = max{ j : 1 <= j < i and f_j + g <= s_i },  p(i) = 0 if empty
WHERE: i, j are 1-based indices in finish order; p(i) is an index in [0, i-1];
       p(i) = 0 denotes the empty prefix, whose value is 0 by definition.
CONSTRAINT: the list must already be in finish order with a declared tie order;
            f_j + g must be computed in the same time unit and clock as s_i.
NUMERICAL EXAMPLE: finishes 3, 4, 5, 6, 8, 9 with g = 0 and s_7 = 10:
                   all six finishes are <= 10, so p(7) = 6.
                   For s_4 = 4 with finishes 3, 4, 5: only f_1 = 3 <= 4, so p(4) = 1.
INTERPRETATION: the whole compatible history of job i is the contiguous prefix
                1..p(i); the recurrence needs no other state about that history.
```

Because finishes are non-decreasing in index, the predicate `f_j + g <= s_i` is monotone in `j`: true for `j <= p(i)` and false for `j > p(i)`. That monotonicity is what licenses binary search, and it is also what makes the recurrence *correct* rather than merely convenient — it proves that the best schedule compatible with `i` is the best schedule over the first `p(i)` jobs, with no compatible job omitted and no incompatible job admitted.

**The recurrence and its derivation.** Let `OPT(i)` be the maximum total value achievable using only jobs `1..i`. Consider any feasible subset `S` of `1..i` that is optimal, and ask only whether `i` is in it. The two cases are exhaustive and mutually exclusive.

- *Case skip.* `i` is not in `S`, so `S` is a feasible subset of `1..i-1` and its value is at most `OPT(i-1)`. Conversely `OPT(i-1)` is achievable within `1..i`, so the best value in this case is exactly `OPT(i-1)`.
- *Case take.* `i` is in `S`. Every other member of `S` is compatible with `i`, hence lies in the prefix `1..p(i)` by SCH-02. So `S \ {i}` is a feasible subset of `1..p(i)` with value at most `OPT(p(i))`, and value `w_i + OPT(p(i))` is achievable. The best value in this case is exactly `w_i + OPT(p(i))`.

Taking the maximum over the two cases gives the recurrence. Induction on `i` with the base `OPT(0) = 0` establishes correctness, since every subproblem referenced has a strictly smaller index.

```text
FORMULA [SCH-03]: OPT(i) = max( OPT(i-1), w_i + OPT(p(i)) ),  OPT(0) = 0
WHERE: OPT(i) is best additive value over jobs 1..i, in value units;
       w_i is the value of accepting job i, same units; p(i) from SCH-02.
CONSTRAINT: additive values; p(i) strictly less than i; feasibility exactly the
            pairwise non-overlap of SCH-02; values may be negative, in which case
            the skip branch dominates and the job is never selected.
NUMERICAL EXAMPLE: w = (5, 6, 4), p = (0, 0, 0):
                   OPT(1) = max(0, 5+0) = 5
                   OPT(2) = max(5, 6+0) = 6
                   OPT(3) = max(6, 4+0) = 6, so accept job 2 alone.
INTERPRETATION: job 2 displaces job 1 rather than joining it; a schedule is a set,
                not a sum of everything that arrives.
```

**Reconstruction.** The table gives the value; the schedule needs a separate pass. The rule is a comparison mirroring SCH-03, with an explicit resolution for equality:

```text
FORMULA [SCH-04]: take i  iff  w_i + OPT(p(i)) > OPT(i-1)  (strict; equal -> skip)
                  then continue from index p(i); otherwise continue from i - 1.
WHERE: i is the current index, starting at n; the accepted set is built in
       decreasing index order and reversed at the end for presentation.
CONSTRAINT: the comparison must use exactly the same arithmetic as SCH-03 —
            recomputing with different rounding or a different p(i) breaks the
            correspondence and can select an infeasible set.
NUMERICAL EXAMPLE: OPT = (0, 5, 6, 6), w = (5, 6, 4), p = (0, 0, 0).
                   i = 3: 4 + 0 = 4 > 6? no -> skip, i = 2.
                   i = 2: 6 + 0 = 6 > 5? yes -> take 2, i = 0. Stop.
                   Accepted = {2}; total 6, matching OPT(3).
INTERPRETATION: the tie rule decides between equally valuable schedules, so it
                must be recorded in output metadata, not left to implementation
                accident.
```

**Why strict comparison matters.** Using `>=` instead of `>` changes the tie policy from skip-on-equality to take-on-equality. Both produce optimal value; they produce different schedules. In the Section 5 example there are six distinct maximum-value schedules for the cardinality objective, so this is not a hypothetical distinction. A regression test that compares only objective values will pass while the published schedule changes, which is exactly the failure an operations team reports as “the system is not deterministic”.

**Boundary and counterexample for 4.2.** The recurrence computes an optimum of the additive model. Give it `A = [0,10)`, `w_A = 100` and `B = [0,10)`, `w_B = 60` with an interaction rule that the pair is worth 200 together; the true optimum takes both and earns 200, while the recurrence returns 100 and is faithfully wrong about the business. The second boundary is the predecessor: replace `f_j + g <= s_i` with `f_j <= s_i` while a ten-minute buffer is contractual, and every link is recomputed too late, producing a plan that overbooks. The third is the base case: `OPT(0) = 0` assumes an empty schedule has zero value, which fails if a fixed commitment charges for idle capacity.

**Takeaway from 4.2:** The recurrence is an exhaustive case analysis on one binary decision per job; its power comes from the prefix property, and its weakness is that both the prefix and the additivity must be true.

### 4.3 Key Special Cases / Variants

| Variant | What changes | Assumptions | Use when | Avoid when | ID |
|---|---|---|---|---|---|
| Buffered compatibility | `f_j + g <= s_i` everywhere | `g` is a fixed duration | cleaning, travel, documentation, cooldown | turnaround depends on the pair of jobs | SCH-02 |
| `k` identical resources, count objective | maintain `k` last-finishes | machines interchangeable | “fit as many as possible with a team of `k`” | machines have different capabilities | SCH-06 |
| `k` identical resources, value objective | state becomes a `k`-tuple of finishes | additive values | small `k`, moderate `n` | `k >= 3` with large `n` | SCH-11 |
| Minimum resource count | answer is peak overlap depth | all requests must be served | capacity planning, staffing | requests may be rejected | SCH-07 |
| Negative or zero values | skip branch dominates | additive values | some jobs destroy value | value is a rank | SCH-03 |
| Zero-length request | `s_i = f_i` rejected at input validation | intervals must be positive | never in this model | an instant-only task is legitimately allowed | §3.1 |
| Closed-interval semantics | touching conflicts | contract says inclusive ends | legal or regulatory wording is inclusive | resource handover is instantaneous | §3.2 |
| Density-first heuristic | sort by `w_i / (f_i - s_i)` | none | never as an exact method | any correctness claim | SCH-09 |

**Buffered compatibility.** The buffer is absorbed by rewriting the predecessor predicate. Nothing else in the algorithm changes, which is why buffers are cheap to add and expensive to forget. The important consequence is that a buffer can change the *count*, not just the identity of the schedule: on the Section 5 instance, `g = 0` admits four jobs worth 30, and `g = 1` admits three worth 24. That is a 25% loss of accepted count and a 20% loss of value from a one-unit buffer, which is the kind of number a manager needs before agreeing to a turnaround policy.

```text
FORMULA [SCH-05]: p_g(i) = max{ j < i : f_j + g <= s_i }
WHERE: g is a duration in the same unit as f and s; p_g(i) = 0 when empty.
CONSTRAINT: g >= 0 and constant; a sequence-dependent setup time makes g a
            function of the ordered pair and the prefix property can fail.
NUMERICAL EXAMPLE: finishes 3, 4, 5, 6, 8, 9; s = 10.
                   g = 0  -> p = 6 (f_6 = 9 <= 10)
                   g = 1  -> p = 5 (f_5 = 8 <= 9, f_6 = 9 <= 9 is true, so p = 6)
                   g = 2  -> p = 5 (f_6 + 2 = 11 <= 10 false)
INTERPRETATION: increasing the buffer can only move predecessors earlier,
                so the optimal value is non-increasing in g.
```

The monotonicity in the interpretation is worth stating as a check: because every feasible set under `g' > g` is also feasible under `g`, the optimum under `g'` cannot exceed the optimum under `g`. If a run reports a higher value for a larger buffer, the run has a bug — most often a stale `p(i)` array left over from the previous buffer value.

**`k` identical resources for cardinality.** With several interchangeable machines, finish order is still the right scan order, but the state is the multiset of `k` last-finishes. The greedy rule generalises cleanly: assign the request to the machine with the **largest** last-finish that is still compatible. Choosing the largest compatible finish is the exchange argument applied per machine — it preserves the machine that is free earliest for a future long request.

```text
FORMULA [SCH-06]: on scanning i in finish order, let C = { m : F_m + g <= s_i }.
                  If C is nonempty, pick m* = argmax_{m in C} F_m,
                  accept i, and set F_{m*} = f_i. Else reject i.
WHERE: F is the vector of k last-finishes, initialised to -infinity;
       ties in argmax broken by the smallest machine index.
CONSTRAINT: k identical machines; all requests equal value; finish-order scan.
NUMERICAL EXAMPLE: k = 2, F = (-inf, -inf).
                   i=1: both free, pick machine 1, F = (3, -inf)
                   i=2: only machine 2 free, F = (3, 4)
                   i=3 (s=2): no machine with F_m <= 2, so reject
                   i=4 (s=0): no machine, reject
                   Accepted count after 4 requests: 2.
INTERPRETATION: the rule is “use the machine that is just barely free”, which
                keeps the widest remaining windows available.
```

This variant is optimal for cardinality on `k` identical machines, and it is *not* optimal for value. The value version requires tracking which specific jobs each machine holds, because the value of a machine's history matters; the state becomes a `k`-tuple of indices, which is Section 4.4's complexity boundary.

**Minimum resource count.** If every request must be served, the question becomes how many identical resources are needed. This is minimum colouring of an interval graph, and the answer is the maximum number of mutually overlapping requests.

```text
FORMULA [SCH-07]: k* = max_t  |{ i : s_i <= t < f_i }|      (half-open, g = 0)
WHERE: t ranges over the distinct endpoint instants; the count is the depth.
CONSTRAINT: intervals are half-open; requests must all be served; resources are
            identical and unlimited in availability (only the count is optimised).
NUMERICAL EXAMPLE: two requests [0,1) and [1,2) never overlap:
                   depth is 1 on [0,1) and 1 on [1,2), so k* = 1.
                   Add [0,2): depth is 2 on [0,1), so k* = 2.
INTERPRETATION: staffing is set by the peak, not by the daily average, and the
                gap between the two is the entire reason rosters argue.
```

Why the maximum depth is achievable: schedule by start time, and assign each request to any free resource. If depth never exceeds `k`, then at the moment a request starts at most `k - 1` others are running, so at least one of the `k` resources is free. The lower bound is trivial: at the peak instant, `depth` resources must be simultaneously occupied. Equality of the two bounds is the reason interval graphs are perfect and the reason this problem is polynomial while general graph colouring is not.

```text
FORMULA [SCH-08]: depth(t) via a +1 at s_i and a -1 at f_i sweep.
                  Sort events by (time, delta) with -1 ordered before +1 at equal
                  times for half-open semantics; running sum gives depth.
WHERE: delta is a signed count; the running sum is an integer count of occupancy.
CONSTRAINT: the tie order at equal times encodes the endpoint convention; reversing
            it converts the computation to closed intervals.
NUMERICAL EXAMPLE: events for [0,3), [0,6), [3,6):
                   t=0: +1,+1 -> 2 ; t=3: -1,+1 -> 2 ; t=6: -1,-1 -> 0
                   peak depth = 2, so k* = 2.
INTERPRETATION: one O(n log n) sort answers the capacity question exactly; no
                search over assignments is required.
```

**Negative values.** SCH-03 handles them without modification: when `w_i < 0`, the take branch is `w_i + OPT(p(i)) < OPT(i-1)`, so the job is never taken. The trap is a downstream reporting layer that computes “acceptance rate” and treats the deliberately rejected negative-value jobs as model errors. Distinguish *rejected because infeasible* from *rejected because unprofitable* in the audit record.

**Closed intervals.** Under closed semantics the predecessor predicate becomes `f_j + g < s_i` — strict, because touching conflicts. On integer endpoints, `g = 1` with half-open semantics coincides exactly with `g = 0` under closed semantics. That equivalence is a useful debugging lever: if two teams disagree about the convention, ask each to produce the buffer value that makes their result match the other's, and the disagreement becomes a number.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

This subsection is the one to read before defending a choice. Each row is a way the two main algorithms stop being correct, with the discriminating check that reveals it.

| Result / invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| SCH-01 returns maximum count | equal values, one resource | recompute count; compare with subset oracle on `n <= 20` | A=[0,3) w=100, B=[0,1) w=60, C=[1,3) w=60 | SCH-03 |
| SCH-01 value ratio is bounded | — (it is not) | compute greedy value / optimal value | one long job `[0,M)` worth `M^2` versus `M` unit jobs worth 1 each | SCH-03 |
| SCH-03 returns maximum value | additive values, prefix predecessors | enumerate subsets for `n <= 20` | pair bonus or per-customer cap | state must include the interaction |
| SCH-06 returns maximum count | `k` identical machines | compare with depth-based bound and oracle | machines with different capabilities | assignment model |
| SCH-07 equals staffing need | all requests served, identical resources | compare peak depth to roster size | shifts with legally mandated breaks | roster model with break constraints |
| density ordering is a good heuristic | none | compute ratio against optimum | A=[0,6) w=5, B=[4,6) w=3, C=[4,10) w=8 | SCH-03 |
| `OPT` is non-decreasing in index | skip always feasible | inspect table for a decrease | a variant forcing acceptance of every job | feasibility-constrained DP |

**Counterexample 1 — greedy on value, and how bad it can be.** Take one long request `A = [0, M)` with `w_A = M^2` and `M` unit requests `B_r = [r-1, r)` with `w_{B_r} = 1`. Finish order puts every unit request before `A`, because each finishes by `M-1 < M`. Greedy accepts all `M` unit requests for a total of `M`, then rejects `A`. The optimum is `A` alone, worth `M^2`. The ratio is `M`, and `M` is also `n - 1`, so the approximation ratio is unbounded as the instance grows. With `M = 10` the greedy answer is 10 against an optimum of 100; with `M = 100` it is 100 against 10,000. This is why “we sort by finish time and it works” is a statement about the count objective only.

```text
FORMULA [SCH-09]: density_i = w_i / (f_i - s_i)
WHERE: density is value per unit time; f_i - s_i > 0 is required.
CONSTRAINT: only meaningful as a heuristic ranking; it is not an exact method for
            any of the objectives in Section 3.6.
NUMERICAL EXAMPLE: A: w=5 over 6 units -> 0.8333 ; B: w=3 over 2 -> 1.5 ;
                   C: w=8 over 6 -> 1.3333. Order B, C, A.
                   Take B. C conflicts with B. A conflicts with B.
                   Heuristic total = 3. Optimum = C alone = 8.
INTERPRETATION: greedy-by-density can be worse than doing nothing clever at all,
                which is why it belongs in a selection table and not in a scheduler.
```

**Counterexample 2 — the same instance defeats finish-greedy too.** On the three-job set above, finish order is `B` (finish 6), `A` (finish 6), `C` (finish 10). Greedy takes `B`, rejects `A` and `C`, total 3. The optimum is 8. Both heuristics lose by more than a factor of two on a three-job instance, and the dynamic program gets it right: `p(A) = 0`, `p(B) = 0`, `p(C) = 0` (because `A` and `B` both finish at 6 and `C` starts at 4), so `OPT = max(5, 3, 8) = 8`.

**Counterexample 3 — the average is not the peak.** Suppose total occupied time across a 15-unit horizon is 45 units. The volume bound gives `k >= ceil(45/15) = 3` resources. But if four requests overlap on `[2,5)`, the true minimum is `k* = 4` and any three-resource roster must reject or delay work. The bound is useful as a lower bound and dangerous as a plan; the peak-depth computation of SCH-07 is cheap and exact, so there is no reason to staff from the average.

**Counterexample 4 — multi-resource value blows up.** For value maximisation on `k` machines, the natural state is the sorted tuple of the `k` last-finishes, giving `O(n^k)` states and `O(n^k)` transitions with an `O(k)` selection inside each.

```text
FORMULA [SCH-10]: state count for k-machine weighted DP is O(C(n,k)) sorted k-tuples,
                  transitions O(k) each, so time O(k * C(n,k)) and space O(C(n,k)).
WHERE: C(n,k) is the binomial coefficient; n jobs, k identical machines.
CONSTRAINT: exact only if the state truly is the multiset of last finishes, which
            fails for non-identical machines or machine-dependent values.
NUMERICAL EXAMPLE: n = 40, k = 3 -> C(40,3) = 9880 states, times 3 transitions
                   = 29,640 operations: trivial. n = 40, k = 10 ->
                   C(40,10) = 847,660,528 states: not tractable.
INTERPRETATION: the exact method survives for small k and dies abruptly, which is
                the point at which an integer program or a heuristic is the honest
                answer rather than a bigger table.
```

**Counterexample 5 — recomputing the predecessor with the wrong predicate.** A team computes `p(i)` from the unsorted original arrival order. Because the prefix property fails in that order, `p(i)` points at a job that is compatible while skipping an incompatible one, and the reconstructed schedule overlaps. The discriminating check is mechanical: for every accepted consecutive pair in the reconstructed schedule, assert `f_earlier + g <= s_later`. That single assertion catches the whole family.

**Takeaway from 4.4:** Both headline algorithms have narrow validity domains, and every boundary in this topic fails by producing a plausible schedule rather than an error message.

## 5. Worked Numerical Example

**Evidence label:** ILLUSTRATIVE job set and value units; VERIFIED arithmetic (every table below was recomputed by an independent exhaustive subset enumeration over all `2^12 = 4096` subsets, so the claimed optima are confirmed by brute force, not merely by the algorithm under test). The scenario — a single service bay with twelve requested bookings across a fifteen-hour window — is constructed for teaching and is not a measurement of any real operation.

**Decision and workflow.** One bay must be allocated for a day. Twelve booking requests are on the desk. Management has three questions, and they have three different answers: (a) fit the largest number of bookings, (b) maximise the total value of the day, (c) state what the day costs if a one-hour turnaround is mandated between bookings. Answering all three from one algorithm would produce at least one wrong answer.

**Input ledger (all times in hours from the start of the window; value in abstract value units).**

| Raw id | Start `s` | Finish `f` | Duration | Value `w` | Value per hour |
|---|---:|---:|---:|---:|---:|
| A | 0 | 3 | 3 | 5 | 1.667 |
| B | 1 | 4 | 3 | 6 | 2.000 |
| I | 2 | 5 | 3 | 4 | 1.333 |
| H | 0 | 6 | 6 | 12 | 2.000 |
| C | 3 | 6 | 3 | 8 | 2.667 |
| L | 5 | 8 | 3 | 3 | 1.000 |
| D | 4 | 9 | 5 | 11 | 2.200 |
| E | 6 | 10 | 4 | 7 | 1.750 |
| F | 8 | 12 | 4 | 9 | 2.250 |
| G | 9 | 13 | 4 | 10 | 2.500 |
| J | 10 | 14 | 4 | 6 | 1.500 |
| K | 12 | 15 | 4 | 8 | 2.000 |

Total busy time is `3+3+3+6+3+3+5+4+4+4+4+4 = 45` hour-units across a horizon of `15` hours, so the volume lower bound on resources is `ceil(45/15) = 3`.

**Finish-sorted order.** Sorting by `(f, s, id)` gives the working index:

| `i` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| id | A | B | I | H | C | L | D | E | F | G | J | K |
| `s_i` | 0 | 1 | 2 | 0 | 3 | 5 | 4 | 6 | 8 | 9 | 10 | 12 |
| `f_i` | 3 | 4 | 5 | 6 | 6 | 8 | 9 | 10 | 12 | 13 | 14 | 15 |
| `w_i` | 5 | 6 | 4 | 12 | 8 | 3 | 11 | 7 | 9 | 10 | 6 | 8 |

At `f = 6` the tie is broken by start time, putting `H` (start 0) before `C` (start 3). This ordering choice is arbitrary but declared; the alternative would place `C` at index 4.

**Predecessor computation with `g = 0`.** For each `i`, find the largest earlier index whose finish is at most `s_i`:

| `i` | Job | `s_i` | Candidate finishes with `f + 0 <= s_i` | `p(i)` |
|---:|---|---:|---|---:|
| 1 | A | 0 | none | 0 |
| 2 | B | 1 | none (f_1 = 3 > 1) | 0 |
| 3 | I | 2 | none | 0 |
| 4 | H | 0 | none | 0 |
| 5 | C | 3 | f_1 = 3 | 1 |
| 6 | L | 5 | f_1, f_2, f_3 = 3, 4, 5 | 3 |
| 7 | D | 4 | f_1, f_2 = 3, 4 | 2 |
| 8 | E | 6 | f_1..f_5 = 3,4,5,6,6 | 5 |
| 9 | F | 8 | up to f_6 = 8 | 6 |
| 10 | G | 9 | up to f_7 = 9 | 7 |
| 11 | J | 10 | up to f_8 = 10 | 8 |
| 12 | K | 12 | up to f_9 = 12 | 9 |

So `p = (0, 0, 0, 0, 0, 1, 3, 2, 5, 6, 7, 8, 9)` including the leading `p(0) = 0`. Verify one entry by brute force: for `i = 7` (`D`, `s = 4`), the finishes at indices 1..6 are `3, 4, 5, 6, 6, 8`; those `<= 4` are indices 1 and 2, and the largest is 2, so `p(7) = 2`. Not `i - 1 = 6`, and not `1`.

**Dynamic-programming table for the value objective.**

| `i` | Job | `w_i` | `p(i)` | `w_i + OPT(p(i))` (take) | `OPT(i-1)` (skip) | `OPT(i)` | Choice |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | — | — | — | — | — | 0 | base |
| 1 | A | 5 | 0 | 5 + 0 = 5 | 0 | 5 | take |
| 2 | B | 6 | 0 | 6 + 0 = 6 | 5 | 6 | take |
| 3 | I | 4 | 0 | 4 + 0 = 4 | 6 | 6 | skip |
| 4 | H | 12 | 0 | 12 + 0 = 12 | 6 | 12 | take |
| 5 | C | 8 | 1 | 8 + 5 = 13 | 12 | 13 | take |
| 6 | L | 3 | 3 | 3 + 6 = 9 | 13 | 13 | skip |
| 7 | D | 11 | 2 | 11 + 6 = 17 | 13 | 17 | take |
| 8 | E | 7 | 5 | 7 + 13 = 20 | 17 | 20 | take |
| 9 | F | 9 | 6 | 9 + 13 = 22 | 20 | 22 | take |
| 10 | G | 10 | 7 | 10 + 17 = 27 | 22 | 27 | take |
| 11 | J | 6 | 8 | 6 + 20 = 26 | 27 | 27 | skip |
| 12 | K | 8 | 9 | 8 + 22 = 30 | 27 | 30 | take |

**Reconstruction walk.** Start at `i = 12`.

- `i = 12` (K): take value `8 + OPT(9) = 8 + 22 = 30` exceeds skip `OPT(11) = 27`, so **accept K**, jump to `p(12) = 9`.
- `i = 9` (F): `9 + OPT(6) = 9 + 13 = 22` exceeds `OPT(8) = 20`, so **accept F**, jump to `p(9) = 6`.
- `i = 6` (L): `3 + OPT(3) = 3 + 6 = 9` does not exceed `OPT(5) = 13`, so **skip L**, decrement to 5.
- `i = 5` (C): `8 + OPT(1) = 8 + 5 = 13` exceeds `OPT(4) = 12`, so **accept C**, jump to `p(5) = 1`.
- `i = 1` (A): `5 + OPT(0) = 5` exceeds `OPT(0) = 0`, so **accept A**, jump to 0. Stop.

Accepted set `{A, C, F, K}` occupying `[0,3)`, `[3,6)`, `[8,12)`, `[12,15)`. Objective value `5 + 8 + 9 + 8 = 30`, matching `OPT(12) = 30`. The bay is idle on `[6,8)`, a two-hour gap that costs nothing in the value objective and is exactly where a buffer or a walk-in could fit.

**Answer (a): maximum count.** Run SCH-01 in finish order with `L = -inf`:

| Step | Job | `s_i >= L`? | Decision | New `L` |
|---:|---|---|---|---:|
| 1 | A | `-inf` | accept | 3 |
| 2 | B | `1 >= 3` false | reject | 3 |
| 3 | I | `2 >= 3` false | reject | 3 |
| 4 | H | `0 >= 3` false | reject | 3 |
| 5 | C | `3 >= 3` true | accept | 6 |
| 6 | L | `5 >= 6` false | reject | 6 |
| 7 | D | `4 >= 6` false | reject | 6 |
| 8 | E | `6 >= 6` true | accept | 10 |
| 9 | F | `8 >= 10` false | reject | 10 |
| 10 | G | `9 >= 10` false | reject | 10 |
| 11 | J | `10 >= 10` true | accept | 14 |
| 12 | K | `12 >= 14` false | reject | 14 |

Count is 4, with `{A, C, E, J}` worth `5 + 8 + 7 + 6 = 26`. Note the cardinality optimum is 4 and the value of that particular optimum is 26, while the value optimum is 30 — and the value-optimal set `{A, C, F, K}` also has cardinality 4. Cardinality did not *cause* the 26, it merely failed to distinguish among the tied cardinality optima. Exhaustive enumeration confirms six distinct four-job sets achieve the maximum count: `{A,C,E,J}`, `{A,C,E,K}`, `{A,C,F,K}`, `{A,L,F,K}`, `{B,L,F,K}`, `{I,L,F,K}`. Their values are 26, 28, 30, 25, 26, 24 respectively. Greedy returned the worst of the six.

**This is the single most important number in the sheet.** On this instance, applying the correct algorithm for the wrong objective lost `(30 - 26)/30 = 13.33%` of achievable value while achieving the same count. Nothing failed. No exception was raised. The count was optimal. The value was not, and the business measures value.

**Answer (b): maximum value is 30.** Confirmed independently by enumerating all 4096 subsets and checking non-overlap with a sweep: the maximum feasible value is exactly 30, and exactly one subset attains it, namely `{A, C, F, K}`.

**Answer (c): the one-hour turnaround.** Recompute `p(i)` with the predicate `f_j + 1 <= s_i`:

| `i` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| `p_g(i)` | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 3 | 5 | 6 | 7 | 8 |
| `OPT_g(i)` | 5 | 6 | 6 | 12 | 12 | 12 | 16 | 16 | 21 | 22 | 22 | 24 |

Arithmetic for the changed entries:

- `i = 5` (C, `s = 3`): `f_1 + 1 = 4 <= 3` is false, so `p_g(5) = 0`. Take `8 + 0 = 8` against skip `OPT_g(4) = 12` → skip, `OPT_g(5) = 12`.
- `i = 6` (L, `s = 5`): `f_2 + 1 = 5 <= 5` true, `f_3 + 1 = 6 <= 5` false, so `p_g(6) = 2`. Take `3 + 6 = 9` against skip 12 → skip, `OPT_g(6) = 12`.
- `i = 7` (D, `s = 4`): `f_1 + 1 = 4 <= 4` true, so `p_g(7) = 1`. Take `11 + 5 = 16` against skip 12 → take, `OPT_g(7) = 16`.
- `i = 9` (F, `s = 8`): `f_5 + 1 = 7 <= 8` true, `f_6 + 1 = 9 <= 8` false, so `p_g(9) = 5`. Take `9 + 12 = 21` against skip 16 → take, `OPT_g(9) = 21`.
- `i = 11` (J, `s = 10`): `p_g(11) = 7`. Take `6 + 16 = 22` against skip 22 → tie, skip policy keeps `OPT_g(11) = 22`.
- `i = 12` (K, `s = 12`): `p_g(12) = 8`. Take `8 + 16 = 24` against skip 22 → take, `OPT_g(12) = 24`.

Reconstruction yields `{A, D, K}` worth `5 + 11 + 8 = 24`. The optimum falls from 30 to 24, a loss of `(30 - 24)/30 = 20.00%`. Cardinality greedy under the same buffer yields `{A, L, G}`, count 3 and value `5 + 3 + 10 = 18`; the cardinality optimum falls from 4 to 3, a loss of `(4-3)/4 = 25.00%`.

**Independent checks.**

1. *Value recomputation from the accepted list*, not from the table: `5 + 11 + 8 = 24` equals `OPT_g(12) = 24`. ✓
2. *Pairwise feasibility replay*: `f_A + 1 = 4 <= s_D = 4` ✓; `f_D + 1 = 10 <= s_K = 12` ✓. No other pair shares a resource. ✓
3. *Monotonicity in the buffer*: since every set feasible at `g = 1` is feasible at `g = 0`, `OPT_g(12) = 24 <= OPT(12) = 30`. ✓ (Had this failed, the `p` array would have been stale.)
4. *Closed-interval cross-check*: recomputing with the strict predicate `f_j < s_i` (closed endpoints, no buffer) produces exactly the same table and the same `{A, D, K}` selection. This is the general equivalence: closed with zero buffer equals half-open with `g` equal to one unit of resolution. ✓
5. *Depth from SCH-08* with events `(0,+1),(0,+1),(1,+1),(2,+1),(3,-1),...`: peak depth is 4, reached on `[2,3)` where `A, B, I, H` all occupy. So `k* = 4` resources would serve every request with no rejection. ✓
6. *Volume bound*: `45/15 = 3`, and the peak is 4, confirming that the volume bound is a lower bound that happens to be tight only when load is perfectly smooth.

**The wrong path, shown explicitly.** A practitioner who reasons “maximise value, so sort by value per hour and take the best” computes densities `C: 2.667`, `G: 2.500`, `F: 2.250`, `D: 2.200`, `B: 2.000`, `H: 2.000`, `K: 2.000`, `E: 1.750`, `A: 1.667`, `J: 1.500`, `I: 1.333`, `L: 1.000`. Taking in that order: `C` at `[3,6)`; `G` at `[9,13)`; `F` conflicts with `G`; `D` conflicts with `C`; `B` conflicts with `C`; `H` conflicts with `C`; `K` conflicts with `G`; `E` conflicts with `C` and `G`; `A` conflicts with `C`; `J` conflicts with `G`; `I` conflicts with `C`; `L` is compatible with both. Accepted `{C, G, L}` worth `8 + 10 + 3 = 21`, which is `(30-21)/30 = 30%` worse than the optimum while looking like a sensible business rule. The correction is not a better sort key; it is the recurrence.

A second plausible wrong path is to sort by value descending: `H(12)`, `D(11)`, `G(10)`, `F(9)`, `C(8)`, `K(8)`, `E(7)`, `B(6)`, `J(6)`, `A(5)`, `I(4)`, `L(3)`. Taking greedily: `H` at `[0,6)`; `D` conflicts; `G` at `[9,13)`; `F` conflicts with `G`; `C` conflicts with `H`; `K` conflicts with `G`; `E` conflicts with `H`; `B` conflicts with `H`; `J` conflicts with `G`; `A` conflicts with `H`; `I` conflicts with `H`; `L` conflicts with `H`. Total `12 + 10 = 22`, or `26.67%` below the optimum. Both sorts are worse than the finish-based cardinality greedy's 26, which is a useful reminder that a heuristic's plausibility is not evidence of its quality.

**Production-scale interpretation.** At twelve jobs the whole computation is a spreadsheet exercise, and the value of the method is entirely in getting the *contract* right. At hundreds of thousands of jobs the same code becomes a batch job, and the failure modes shift from conceptual to operational: which timezone the endpoints are in, whether the `p(i)` computation is recomputed after a buffer policy change, whether the tie policy is versioned, and whether the accepted list is recomputed from the input rather than trusted from the run log. The recurrence does not become harder; the surrounding data does.

**What this example does not establish.** It does not establish that additive value is the right business objective, that value units are comparable across job types, that the twelve requests are representative, or that a real bay has zero travel time. It establishes only that for a stated model, the arithmetic is correct and the three objectives give three different schedules from one input.

**Takeaway:** The worked instance shows a 13.33% value loss from using the right algorithm on the wrong objective and a 20% value loss from a one-hour buffer policy, both invisible to anyone who checks only feasibility.

## 6. Visual / Geometric Intuition

### 6.1 The instance as a timeline

```text
hours  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
       |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
A      [=======)                                                           w=5
B        [=======)                                                         w=6
I          [=======)                                                       w=4
H      [==============)                                                    w=12
C              [=======)                                                   w=8
L                  [=======)                                               w=3
D                [=============)                                           w=11
E                        [===========)                                     w=7
F                                [===========)                             w=9
G                                  [===========)                           w=10
J                                    [===========)                         w=6
K                                              [===========)               w=8
       |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
depth  2   3   4   4   4   4   3   3   3   3   3   3   3   2   1   0
```

Reading the depth row: at hour 2 the four requests `A, B, I, H` all hold the bay, so the peak depth is 4, reached on the half-open span `[2,5)`. Depth never returns below 3 until hour 13, which is the crucial operational fact: this day is not merely busy, it is busy without relief, and any single absence pushes the day into overbooking.

### 6.2 Greedy's frontier versus the value optimum

```text
g=0, cardinality greedy:   A[0,3) C[3,6) E[6,10) J[10,14)        count 4, value 26
                            ^^^^^^^^ tight, no idle until the end
value optimum:             A[0,3) C[3,6)   ....   F[8,12) K[12,15)   count 4, value 30
                                            ^^^^
                                    deliberate two-hour gap
```

The value optimum is *less* tightly packed than the greedy schedule: it leaves `[6,8)` empty in exchange for replacing `E` (worth 7) plus `J` (worth 6) with `F` (worth 9) plus `K` (worth 8), a gain of `17 - 13 = 4`. The gap is not waste; it is the price of a better composition. Any dashboard that flags idle time as a defect would flag this better schedule as worse, which is why the objective must be stated before the utilisation report is read.

### 6.3 The predecessor arrow as geometry

```text
finish-ordered indices:  1    2    3    4    5    6    7    8    9   10   11   12
job:                     A    B    I    H    C    L    D    E    F    G    J    K
finish f_i:              3    4    5    6    6    8    9   10   12   13   14   15
                          ^^^^^^^^^^^^^^^^^^^^^^^
s_9 = 8:  everything with f_i <= 8 is compatible with F:  p(9) = 6
                          ^^^^^^^^^^^^^^^^^^
s_7 = 4:  everything with f_i <= 4 is compatible with D:  p(7) = 2
```

The two brackets show the prefix property. Because finishes increase left to right, the compatible set is always a contiguous run starting at index 1. There is no “hole” in the middle: if `f_4 <= s_9` then `f_3 <= s_9` necessarily, since `f_3 <= f_4`. This is why the recurrence can ask one question — where does the run end — instead of enumerating subsets.

### 6.4 What the buffer does to the frontier

```text
g = 0  (half-open, handover at the boundary):
  A [0,3)      C [3,6)      E [6,10)      J [10,14)
     |           |             |              |
     +-----------+-------------+--------------+   all touching, all legal

g = 1  (one-hour turnaround):
  A [0,3)   .   D [4,9)      .  K [12,15)
     |      gap   |          gap  |
     +------+-----+----------+----+
            forbidden intervals: [3,4) and [9,12)
```

The forbidden intervals `[3,4)` and `[9,12)` are not idle capacity. They are capacity that cannot be sold under the policy, and they are created by the policy, not found in the data. Naming them as intervals rather than as “slack” is what makes the loss visible in the same units as the revenue it costs: `[9,10)` and `[10,11)` could each have held a job.

### 6.5 Interval graph view

```text
       A
      / \
     B   C
      \ / \
       I   D
            \
             F --- K
```

Two requests are joined when they *conflict*. The graph is chordal and, in the interval case, perfect. A feasible schedule is an independent set; the maximum count is the maximum independent set; the minimum resource count is the chromatic number; and for interval graphs, chromatic number equals clique number equals maximum simultaneous depth. That chain of equalities is the whole reason Schedule-07 is a sweep rather than a search, and it is the structural feature that fails the moment intervals acquire eligibility restrictions (a machine that cannot run a job type makes the graph non-interval).

**Takeaway:** Every picture on this page is the same fact in different clothes — feasible sets are prefixes in finish order, so both the sweep and the recurrence are consequences of one monotonicity.

## 7. Comparison / Selection Table

### 7.1 Methods under comparable conditions

| Method | Preconditions | Guarantee | Time / memory | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Brute-force subset enumeration | `n <= ~20`; feasibility check is cheap | exact for any objective | `O(2^n * n log n)` time, `O(n)` space | oracle for testing; tiny instances | `n > 24` even with pruning | DERIVED |
| Earliest-finish greedy (SCH-01) | one resource, equal values, fixed intervals | maximum count, proven | `O(n log n)` time, `O(1)` extra | “fit as many as possible” | values differ; multiple resources; lateness objectives | DERIVED |
| Weighted interval DP (SCH-03) | one resource, additive values, prefix predecessors | maximum additive value, proven | `O(n log n)` time, `O(n)` space | revenue, priority-weighted acceptance | interacting values; sequence-dependent setup; machine eligibility | DERIVED |
| `k`-machine greedy (SCH-06) | `k` identical resources, equal values | maximum count, proven | `O(n log n + nk)` time | team capacity, identical bays | machine-specific capabilities | DERIVED |
| Depth sweep (SCH-07, SCH-08) | all requests must be served, identical resources | exact minimum resource count | `O(n log n)` time, `O(n)` space | staffing and capacity planning | requests may be rejected; breaks or shifts constrain availability | DERIVED |
| `k`-machine weighted DP (SCH-10) | `k` identical resources, additive values | exact | `O(k * C(n,k))` time | small `k` (2–4), moderate `n` | `k >= 6` or `n >= 100` | DERIVED |
| Density-based greedy (SCH-09) | none | unbounded error | `O(n log n + n)` time | a first cut when nothing better is available | any claim of optimality | ILLUSTRATIVE ratio computed in §4.4 |
| MIP / branch-and-bound | any linearisable structure | exact at termination, with a bound | solver-dependent; not a closed form | eligibility, setup, precedence, multi-objective | hard real-time; no bound on solve time | DERIVED structure only; no benchmark |
| Earliest-due-date (EDD) | deadlines, one resource | optimal for maximum lateness | `O(n log n)` | service-level agreements | objective is count or value | DERIVED, listed for contrast |

Two rows deserve emphasis because they are frequently confused with the headline methods. Density greedy is included precisely because practitioners reach for it: it is the natural business instinct (“go where the value per hour is highest”) and it is the worst-performing non-trivial option here. Earliest-due-date is included because “sort by deadline” and “sort by finish” sound identical and solve different problems: EDD minimises the maximum lateness over *all* jobs with no rejection, while SCH-01 maximises the count of *accepted* jobs with no lateness notion.

### 7.2 Prioritised selection guide

```text
IF any request must be served (no rejection allowed)
   THEN compute peak depth with SCH-08; the answer is the resource count, not a schedule.
ELSE IF requests have deadlines and every accepted job must meet its deadline
   THEN this is not an interval-selection problem; use EDD and check max lateness.
ELSE IF values are equal AND one resource
   THEN SCH-01, earliest finish, O(n log n).
ELSE IF values are additive AND one resource AND a buffer is fixed
   THEN SCH-03 with SCH-05, O(n log n).
ELSE IF k identical resources AND values are equal
   THEN SCH-06.
ELSE IF k identical resources AND values are additive AND k <= 4
   THEN SCH-10 with a tuple state.
ELSE IF values interact, or machines differ, or setup depends on sequence
   THEN formulate a MIP; report the bound and the solve status honestly.
ELSE (no objective can be stated)
   THEN stop; the problem is undefined and any schedule is unfalsifiable.
IF two conditions both seem to apply
   THEN the stricter feasibility relation wins, and the objective must be
        confirmed by the owner of the decision before any solver runs.
```

The last branch is not a joke. A large fraction of real scheduling disputes are arguments about the objective disguised as arguments about the algorithm. “Why is this booking not in the plan?” has three different answers under count, value, and service-all, and no amount of engineering resolves which is correct.

### 7.3 Objective-to-method matrix with worked indications

| Objective | `k = 1` | `k` identical | `k` with eligibility |
|---|---|---|---|
| max count | SCH-01 | SCH-06 | bipartite matching per time slice, or MIP |
| max additive value | SCH-03 | SCH-10, `k <= 4` | MIP or time-indexed formulation |
| min resources, serve all | SCH-07 | SCH-07 | list colouring; NP-hard in general |
| max total busy time | SCH-03 with `w_i = f_i - s_i` | SCH-10 | MIP |
| max min value accepted | check feasibility at threshold via SCH-03 | no simple form | MIP |
| min max lateness | EDD | NP-hard for `k >= 2` | NP-hard |

The right-hand column is where the polynomial structure collapses. Interval graphs are perfect, but an interval graph with a colour-restriction list — machine `m` cannot run job type `T` — is no longer an interval graph, and the colouring problem becomes NP-hard. This is the honest boundary for real operations: most real bays are *not* interchangeable, and the clean sweep is a planning estimate, not a final plan.

### 7.4 Choosing between greedy and DP when values look equal

A subtle selection trap: values that are equal in the source system may not be equal in the decision. If all requests carry the same booked price but differ in duration, then value per unit time differs, and maximising count maximises nothing the business cares about. The check is to ask whether the objective function, written out, has `w_i` multiplying an indicator. If `w_i` is constant, SCH-01 applies. If it varies — including through duration, priority, or margin — SCH-03 applies with that `w_i`. A helpful discipline is to write the objective in the model record before opening the code:

```text
maximise  sum_{i in S} w_i
subject to S feasible: for all i != j in S,  f_i + g <= s_j  or  f_j + g <= s_i
```

If someone cannot write that block, the algorithm choice is premature.

**Takeaway:** Select by objective and feasibility relation first; every method in the table is exact only inside its precondition cell, and the cells are narrow.

## 8. Failure Modes & Edge Cases

Each entry below is a distinct mechanism, not a synonym for its neighbours. The discriminating check is the part that separates it from the look-alike failure; a check that is positive for both is not a discriminator.

### F-01 — Closed and half-open endpoints mixed between optimiser and validator

**Symptom:** every back-to-back pair in the published schedule fails the downstream validation report, though the optimiser reported a feasible plan and the count matches expectations.
**Competing causes:** (a) convention mismatch between the optimiser (half-open) and the validator (closed); (b) a genuine buffer requirement that the optimiser never received; (c) timezone offset applied in one path only.
**Discriminating check:** take a single pair with `f_A == s_B` exactly and evaluate both paths on it. Convention mismatch fails the validator and passes the optimiser; a real buffer requirement fails both; a timezone offset shifts every instant by a constant `k` and is detected by printing raw epoch seconds from both paths, which differ by an exact multiple of 900 or 3600 seconds.
**Bounded fix:** pick one convention, write it into the model record, and derive the other side's predicate from it. Do not “fix” by inserting gaps.
**Verification:** a two-request fixture with `f_A == s_B` plus one with `f_A == s_B - epsilon`, asserted to give opposite results in the intended direction.
**Provenance:** HYPOTHETICAL, mechanism DERIVED from §3.2.

### F-02 — Buffer added to the schedule but not to the predecessor computation

**Symptom:** the plan respects turnaround visually, yet a reconstructed pair violates it; or the reported optimum value is higher than any feasible schedule achieves.
**Competing causes:** (a) stale `p(i)` array computed under an older buffer; (b) buffer applied at acceptance time in the greedy scan but not in the DP; (c) buffer expressed in minutes while endpoints are in hours, making `g` effectively zero after integer truncation.
**Discriminating check:** recompute `OPT` under `g` and under `g' > g`; the optimum must be non-increasing. If it rises, `p` is stale. Independently, print `g` in the same unit as `f - s` and assert `g` is not zero when the policy says otherwise.
**Bounded fix:** recompute the predecessor array as part of the same function that applies the buffer, so the two cannot drift apart; make the buffer a required argument rather than a default.
**Verification:** rerun the Section 5 instance at `g = 1` and assert `OPT = 24`, count `= 3`, selection `{A, D, K}`.
**Provenance:** VERIFIED against the Section 5 trace; the failure mode itself is HYPOTHETICAL.

### F-03 — Greedy applied to a value objective

**Symptom:** the plan is feasible, fast, and looks reasonable, but a value report shows revenue below a hand-built alternative that a planner produced in a spreadsheet.
**Competing causes:** (a) truly a value objective with greedy used; (b) the value units are wrong (gross instead of contribution margin), so even the DP would agree with the bad plan; (c) the hand-built alternative is actually infeasible and the planner ignored a buffer.
**Discriminating check:** run SCH-03 on the same input and compare values. If the DP beats the plan, it is (a). If they match but both are below the spreadsheet, check whether the spreadsheet's plan is feasible by replaying its pairs against the declared predicate — this separates (b) from (c). If the spreadsheet's plan is infeasible, the discrepancy is a process problem, not an algorithm problem.
**Bounded fix:** switch to SCH-03 and record the objective in the model record; if values are wrong, fix the value definition before the algorithm, because the DP will faithfully optimise a wrong number.
**Verification:** the Section 5 instance gives greedy-by-finish 26 against an optimum of 30; assert the DP reproduces 30 on the fixture.
**Provenance:** VERIFIED arithmetic; instance ILLUSTRATIVE.

### F-04 — Predecessor search stops at the first compatible job

**Symptom:** the DP's reported optimum is lower than brute force on small instances, and the gap grows with `n`.
**Competing causes:** (a) linear scan with early exit rather than a scan for the maximum index; (b) binary search on an unsorted finish array; (c) `p(i)` computed against the unsorted arrival order, so the prefix property fails.
**Discriminating check:** assert for every `i` that `p(i)` is the largest index satisfying the predicate, by an independent `O(n^2)` loop on a sample of 100 indices. A mismatch localises the defect. Then assert `f[p(i)] + g <= s[i]` and that `p(i)+1 == i` or `f[p(i)+1] + g > s[i]`; both must hold.
**Bounded fix:** compute `p` with `bisect_right` over the array of `f_j + g` values after sorting, which returns the count of elements `<= s_i`, i.e. the index of the last one.
**Verification:** for the Section 5 instance assert `p = (0,0,0,0,0,1,3,2,5,6,7,8,9)`.
**Provenance:** DERIVED check; fixture VERIFIED.

### F-05 — Undeclared tie policy makes the schedule non-reproducible

**Symptom:** identical inputs produce different accepted sets across runs or across two servers, while the objective value is identical and all tests pass.
**Competing causes:** (a) unstable sort on a key with duplicates; (b) DP reconstruction using `>=` in one build and `>` in another; (c) input files delivered in different physical order and the sort is not total.
**Discriminating check:** run twice with the input rows shuffled; hash the accepted set. A stable hash difference with an identical value proves non-determinism. Then check whether the sort key includes a unique identifier; if not, the sort is not total and (a) is confirmed regardless of the shuffle result.
**Bounded fix:** include the request id as the final comparator key; choose and document `>` or `>=`; publish the policy in the audit record.
**Verification:** ten shuffles produce one hash; and on the twelve-job instance, six distinct optimal sets exist, so a hash-equality test is meaningful rather than trivially satisfied.
**Provenance:** VERIFIED enumeration of the six tied sets; failure mechanism DERIVED.

### F-06 — Volume-based staffing instead of peak depth

**Symptom:** the roster built from total work hours is short at the busiest hour, so requests are delayed or rejected in practice even though the aggregate utilisation looks comfortable.
**Competing causes:** (a) genuine peak-driven capacity shortfall; (b) duration estimates are biased low, inflating apparent capacity; (c) the roster has fewer people available in the peak window than the headcount suggests, due to breaks or shift boundaries.
**Discriminating check:** compute peak depth (SCH-08) and compare with average load `total_busy / horizon`. If peak exceeds roster size while average does not, (a). If peak depth itself is understated relative to the observed late-finishing jobs, compare planned versus actual durations for the ten most overrun jobs, which addresses (b). If depth is computed on headcount rather than on available headcount per hour, (c).
**Bounded fix:** staff to the 95th-percentile depth over a rolling window rather than to the single-day peak, and treat the difference as an explicit queueing decision rather than an accident.
**Verification:** Section 5 instance gives peak 4 and average `45/15 = 3`; assert `k* = 4` from the sweep and from brute-force subset enumeration of the minimum colouring.
**Provenance:** DERIVED arithmetic; staffing policy is HEURISTIC and version-dependent.

### F-07 — Negative or zero values treated as data errors

**Symptom:** the plan correctly never selects a set of jobs, and a monitoring alert fires about “unserved requests” or a business user reports that their jobs are silently dropped.
**Competing causes:** (a) negative values were intentional and correct to skip; (b) a unit or sign error inverted the value (a cost entered as a positive); (c) the objective is cardinality but values were supplied anyway, and someone assumed DP semantics.
**Discriminating check:** read the sign of `w_i` directly from the input and compare with the objective declared in the model record. If the objective is value and `w_i < 0` intentionally, (a). If costs were entered with a plus sign where the convention expects minus, the same job appears with a suspiciously round positive number that equals a known cost line — check against the source ledger. If the objective is cardinality, the values should not have been loaded at all, which is (c).
**Bounded fix:** separate the audit categories “rejected infeasible” from “rejected unprofitable”, and assert the sign convention at load time with a schema check.
**Verification:** a fixture with `w = (-5, 6, 7)` and `p = (0,0,0)` returns 7 with job 3 selected and jobs 1, 2 rejected; assert the rejection reason for job 1 is `negative_value` rather than `conflict`.
**Provenance:** DERIVED.

### F-08 — Preemption assumed but not modelled

**Symptom:** the plan is feasible on paper, yet in execution a long job is interrupted by a higher-priority arrival and the schedule collapses.
**Competing causes:** (a) the model is non-preemptive and reality is preemptive, so the feasibility check is too strict and the plan is over-confident about completion; (b) the model assumes preemption (allowing fractional intervals) while the resource cannot be released mid-job; (c) arrivals during the day are not in the model at all.
**Discriminating check:** inspect whether the accepted set ever contains a partial interval in the solver's output; a fractional job in a non-preemptive resource is (a) or (b) depending on which side allowed it. For (c), compare the plan's accepted set with the actual started-job log: jobs absent from the plan but present in the log indicate unscheduled arrivals.
**Bounded fix:** choose the model explicitly. For preemptive single-resource value maximisation, the fractional relaxation has a different structure and is out of scope here; for non-preemptive reality, reserve capacity for arrivals using an explicit buffer or a policy that defers low-priority bookings.
**Verification:** replay a day's arrivals against the plan and count interruptions; the count must be zero or the deferral policy must name them.
**Provenance:** HYPOTHETICAL; the preemptive/non-preemptive distinction is DERIVED from the model statement.

### F-09 — Timezone and daylight-saving boundaries

**Symptom:** one day per year, adjacent bookings appear to conflict or a booking appears to last an extra hour; the schedule is wrong only on transition dates.
**Competing causes:** (a) naive local timestamps compared as if they were UTC; (b) epochs correct but the business calendar for “a day” is computed in local time; (c) an interval spans the transition, so its wall-clock duration differs from its elapsed duration.
**Discriminating check:** compute the interval length in epoch seconds and in wall-clock terms for a booking that spans the transition. A one-hour discrepancy identifies (c). For (a), inspect whether timestamps carry an offset at all. For (b), compare the day boundary computed in UTC against the boundary computed in the business timezone on the transition date.
**Bounded fix:** store endpoints as UTC instants plus the originating timezone; define “day” explicitly as a local calendar day with an explicit rule for the 23- and 25-hour days; never compare wall-clock strings.
**Verification:** fixtures on both transition dates for the operating timezone, asserting elapsed durations of 23 and 25 hours for full-day intervals.
**Provenance:** HYPOTHETICAL; the mechanism is a property of civil time, not of this algorithm.

### F-10 — Resource count increased to “fix” an infeasible plan

**Symptom:** a plan that had been accepted becomes infeasible after a data refresh, and the response is to add a resource until it fits.
**Competing causes:** (a) genuinely higher demand; (b) duplicate requests from a retried ingestion, inflating demand artificially; (c) endpoint convention changed upstream, converting previously compatible pairs into conflicts.
**Discriminating check:** deduplicate on the request natural key and recount. If demand drops to the prior level, (b). If duplicates are absent, recompute the depth and compare with the previous period's depth — an identical depth with an infeasible plan points to (c), because depth is convention-sensitive only at touching points while feasibility is convention-sensitive at exactly those points.
**Bounded fix:** fix the duplicate or convention defect before changing capacity; record the resource count as an output of the analysis, never as a knob.
**Verification:** depth and count computed before and after deduplication, with the difference attributable to identified duplicate keys.
**Provenance:** HYPOTHETICAL; the conflation of demand with capacity is a reasoning error, not a numerical one.

### F-11 — Additivity silently violated by sequence-dependent cost

**Symptom:** the DP's optimum is unachievable when the plan is executed, because the true cost of a job depends on what ran immediately before it.
**Competing causes:** (a) genuine sequence-dependent setup; (b) a fixed setup cost that was omitted from `w_i` altogether; (c) value interaction between specific pairs only (a discount), which is a different structure from sequence dependence.
**Discriminating check:** compute the realised total cost of the DP's schedule and of a schedule built to minimise setup, on the same accepted set. A gap that depends on ordering within the set is (a). If the DP's plan is uniformly low by a constant per job, (b). If only certain adjacent pairs cause a discrepancy, (c).
**Bounded fix:** for (a) the problem becomes a travelling-salesman-like sequencing problem within each accepted set; the honest response is a MIP or a sequencing heuristic, not a modified recurrence. For (b) add the setup to `w_i`. For (c) add a state dimension recording the last accepted job.
**Verification:** replay the plan through the real cost function and compare with the DP's reported value; the two must agree within a declared tolerance or the model is invalid.
**Provenance:** DERIVED from the additivity definition in §3.4; no measured instance.

### F-12 — Zero-length and inverted requests reaching the optimiser

**Symptom:** a request with `s_i == f_i` is accepted alongside every conflicting job, or an inverted request (`s_i > f_i`) is accepted and then occupies nothing, or the predecessor computation loops.
**Competing causes:** (a) upstream data defect; (b) a legitimate instant-only event (a reminder, a checkpoint) modelled as an interval; (c) a data-entry error where start and end were swapped.
**Discriminating check:** assert `f_i - s_i >= minimum_duration` at load. Zero-length requests pass a `>= 0` check and fail a `> 0` check, distinguishing (a) from a policy that permits instants. Swapped fields are detectable because the swap produces a negative duration and often an implausible magnitude.
**Bounded fix:** reject zero-duration and inverted requests at the boundary with an explicit error code; model instant-only events as a separate type that consumes no resource.
**Verification:** property test on 10,000 random records asserting every accepted request has positive duration and satisfies `f_i + g <= s_j` or `f_j + g <= s_i` against every other accepted request.
**Provenance:** DERIVED.

### F-13 — Reconstruction disagrees with the table

**Symptom:** the accepted set's recomputed value is lower than `OPT(n)`, or higher than any feasible schedule can achieve.
**Competing causes:** (a) reconstruction compares with a different tie rule than the table used; (b) the jump index after a take uses `i - 1` instead of `p(i)`; (c) the table was computed with one buffer and reconstructed with another.
**Discriminating check:** assert `sum of w over accepted == OPT(n)` exactly. Then, for each accepted job, assert the jump landed on `p(i)` by logging the visited indices. A visited sequence that is not strictly decreasing is (b).
**Bounded fix:** factor the recurrence into one function used by both the table and the walk, returning the choice, so the tie rule exists in exactly one place.
**Verification:** the Section 5 walk visits `12 → 9 → 6 → 5 → 1 → 0` and sums to 30.
**Provenance:** VERIFIED trace.

### F-14 — Peak memory from storing the full choice matrix

**Symptom:** the job handles 50,000 requests but is killed at 200,000 on the same hardware, or a container is OOM-killed during a nightly run.
**Competing causes:** (a) an `n x n` choice or predecessor matrix instead of a vector; (b) retention of per-request feature rows for the whole batch; (c) an unbounded cache keyed by request id with no eviction.
**Discriminating check:** estimate bytes as `n * 8` for one `int64` vector versus `n * n * 8`. At `n = 250,000` that is 2.0 MB versus 500 GB; the arithmetic alone distinguishes the cases without profiling. If the arithmetic clears (a), inspect the object graph or use a heap snapshot for (b) and (c).
**Bounded fix:** store `p` as one `int32` vector (1.0 MB at 250,000), `OPT` as one `int64` vector (2.0 MB), and reconstruct by re-deriving choices from the two vectors.
**Verification:** measured peak resident memory on a 250,000-request synthetic batch, with the expected magnitude of a few tens of MB including input rows.
**Provenance:** DERIVED arithmetic; memory figures are computed, not measured here.

### F-15 — Sorted input reused after a policy change

**Symptom:** after a buffer or tie-policy change, results are subtly wrong while all unit tests pass.
**Competing causes:** (a) the sort comparator was cached or the sorted array is reused from a previous run with a different comparator; (b) the comparator includes a mutable field that changed; (c) an incremental update path appends new requests without re-sorting.
**Discriminating check:** recompute the sort from the raw input and compare element-wise with the array used. Any difference is (a) or (c). For (b), assert the comparator's key tuple is derived from immutable fields only.
**Bounded fix:** derive sorted order inside the same call that consumes it; if the array must be persisted, persist the comparator version alongside it and refuse to load a mismatched pair.
**Verification:** a test that mutates the buffer policy at runtime and asserts the newly computed `p` differs from the cached one.
**Provenance:** HYPOTHETICAL; mechanism is a caching hazard, not specific to this algorithm.

**Takeaway:** Every failure in this list produces a valid-looking schedule, so detection requires an assertion about the contract — endpoints, predecessor maximality, value recomputation, or determinism — not a visual inspection of the plan.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity

Sizes are `n` = number of requests after deduplication and validation; `k` = number of identical resources; `U` = number of distinct endpoint instants (at most `2n`). The operation model is a unit-cost comparison on 64-bit instants and values, ignoring cache behaviour and I/O, which are handled in 9b.

| Operation | Input sizes | Time | Space | I/O / communication | Dominant bottleneck |
|---|---|---|---|---|---|
| Validate and parse endpoints | `n` rows | `O(n)` | `O(1)` extra | `O(n)` bytes read | parsing and timezone conversion |
| Deduplicate on natural key | `n` rows | `O(n)` expected with hashing | `O(n)` | `O(n)` bytes | hash table growth |
| Sort by `(f, s, id)` | `n` | `O(n log n)` comparisons | `O(n)` or `O(log n)` auxiliary | none | comparison function cost |
| Compute `p(i)` by binary search | `n` | `O(n log n)` | `O(n)` for the array | none | `log n` comparisons per job |
| Compute `p(i)` by linear scan | `n` | `O(n^2)` | `O(n)` | none | becomes dominant at `n > ~10^4` |
| Greedy scan (SCH-01) | `n` | `O(n)` | `O(1)` extra | none | per-job comparison |
| DP table (SCH-03) | `n` | `O(n)` after `p` | `O(n)` for `OPT` | none | memory bandwidth for the table |
| Reconstruction (SCH-04) | accepted count `<= n` | `O(n)` worst | `O(1)` extra | none | pointer chasing |
| Depth sweep (SCH-08) | `2n` events | `O(n log n)` | `O(n)` | none | event sort |
| `k`-machine greedy (SCH-06) | `n`, `k` | `O(n log n + nk)` | `O(k)` | none | linear scan over `k` finishes |
| `k`-machine greedy with a heap | `n`, `k` | `O(n log n + n log k)` | `O(k)` | none | heap operations |
| `k`-machine weighted DP (SCH-10) | `n`, `k` | `O(k * C(n,k))` | `O(C(n,k))` | none | combinatorial state count |
| Brute-force oracle | `n <= 20` | `O(2^n * n log n)` | `O(n)` | none | exponential growth |

Two comparisons are worth stating explicitly. First, the linear predecessor scan is `O(n^2)` and at `n = 250,000` that is about `3.1 x 10^10` comparisons, which at 100 ns each is roughly 52 minutes of pure comparison work and would dominate an otherwise sub-second job. The binary-search version performs `n * log2(n) = 250,000 * 17.93 ≈ 4,482,892` comparisons, about 0.448 seconds at the same rate. The difference is not a constant factor; it is the difference between a batch window and an outage of the batch window.

Second, the `k`-machine exact DP's growth is not smooth. With `n = 40`: `k = 3` gives `C(40,3) = 9,880` states and is trivial; `k = 10` gives `C(40,10) = 847,660,528` states and is hopeless. There is no intermediate regime where more engineering effort rescues the exact method; the transition from feasible to infeasible is abrupt, and the correct response at that cliff is to change the method (MIP with a bound, or a heuristic with an honest label), not to optimise the table.

```text
FORMULA [SCH-11]: state count S(n,k) = C(n,k) = n! / (k!(n-k)!)
WHERE: n jobs, k identical machines, state = sorted k-tuple of last-finish indices;
       S is a count of distinct states, dimensionless.
CONSTRAINT: identical machines and additive values; if machines differ, the state
            must record which machine holds which job and the count becomes larger.
NUMERICAL EXAMPLE: C(12,2) = 66 ; C(12,3) = 220 ; C(40,3) = 9,880 ;
                   C(40,10) = 847,660,528.
INTERPRETATION: doubling k from 3 to 6 at n = 40 multiplies the state space by
                about 3.8 million, which is why k = 3 is exact and k = 6 is a MIP.
```

### 9b. Back-of-envelope sizing

**Workload.** A plausible planning batch of `n = 250,000` booking requests per planning cycle, each with two 64-bit instants and one 64-bit value. Input size = `250,000 * 3 * 8 = 6,000,000` bytes ≈ 5.72 MiB of raw numeric payload; with a 32-character identifier and JSON framing, the practical file is on the order of tens of MiB. These figures are ILLUSTRATIVE and computed, not measured.

**Algorithm cost.** Sort `250,000 * log2(250,000) ≈ 4,482,892` comparisons. At a deliberately pessimistic 100 ns per comparison (branch-unfriendly, comparator with three keys), that is `4,482,892 * 100 ns = 0.448 s`. The predecessor pass costs about the same order, the DP pass `250,000` additions and comparisons at well under a nanosecond each, and reconstruction at most `n` steps. Total compute is comfortably under two seconds of CPU on one core for the comparison-dominated parts, and the real elapsed time will be governed by parsing, validation, and I/O rather than by the algorithm.

```text
FORMULA [SCH-12]: T_sort ≈ c * n * log2(n)
WHERE: T is time in seconds; c is seconds per comparison; n is request count.
CONSTRAINT: unit-cost comparisons; no I/O; no parallel speedup; c measured on the
            actual comparator, not assumed.
NUMERICAL EXAMPLE: c = 100 ns, n = 250,000 -> T = 100e-9 * 4,482,892 = 0.448 s.
                   Ten times more requests: T = 100e-9 * 250,000 * 10 * 19.25
                   = 4.81 s, a factor of 10.7 rather than 10.
INTERPRETATION: growth is mildly superlinear; a 10x input is a 10.7x sort cost,
                which matters for sizing but not for architecture choices at this scale.
```

**Memory.** One `int32` predecessor vector is `250,000 * 4 = 1,000,000` bytes = 1.0 MB decimal (0.95 MiB). One `int64` table is `2.0 MB`. Storing an `n x n` structure would be `250,000^2 * 8 = 5 x 10^11` bytes = 500 GB, which is the difference between a routine batch and a procurement request. The lesson is not “memory is expensive” but “the wrong data structure is four orders of magnitude wrong, and the arithmetic that reveals it takes ten seconds”.

**Economics, with explicit assumptions.** Suppose one avoided rejected booking is worth 5,000 currency units of contribution margin and the planner recovers 6 such bookings per cycle by switching from density-greedy to the exact DP. That is `6 * 5,000 = 30,000` currency units per cycle. With a 20% margin of safety on the estimate, plan on `30,000 * 1.2 = 36,000` as an upper planning figure, and treat the true value as unknown until measured. Compute cost at 0.05 currency units per CPU-hour for the two seconds of compute is `0.05 * (2/3600) ≈ 2.78 x 10^-5` currency units, which is nine orders of magnitude below the value at stake. The engineering conclusion is unambiguous: the algorithm is free and the decision is not. This is ILLUSTRATIVE arithmetic with assumed prices; substitute the organisation's actual figures before quoting it as a business case.

**Where the cost actually goes.** In production, the dominant costs are: upstream data quality (deduplication, timezone normalisation, endpoint validation), explanation and audit artefacts (why a request was rejected), and the human review of the plan. The optimisation itself is negligible. A team that spends its effort micro-optimising the recurrence while leaving the endpoint convention ambiguous has optimised the cheap part.

**What is NOT measured here.** No benchmark of a real dispatcher, no measured comparison with a commercial scheduling product, no claim that the DP is faster than an MIP on realistic instances — the two have different asymptotic profiles and the crossover is instance-dependent. The figures above establish orders of magnitude, not performance.

**Takeaway:** The algorithm's cost is negligible relative to the decision it supports; the risks are all in the contract, and the one structural cliff to know is the `k`-machine state explosion.
## 10. Key Theorems / Results Reference

| Result | Statement precise enough to use | Essential hypotheses | Practical implication | What it does NOT imply | Source |
|---|---|---|---|---|---|
| Earliest-finish optimality | Among all feasible sets of pairwise-compatible intervals on one resource, the greedy finish-order scan returns one of maximum cardinality | fixed known intervals; equal values; non-preemptive; one resource; half-open (or a fixed buffer) | SCH-01 is safe for the count objective | anything about additive value, lateness, or multiple resources | DERIVED here (§4.1); standard textbook result in interval scheduling |
| Weighted recurrence correctness | For additive values on one resource, `OPT(i) = max(OPT(i-1), w_i + OPT(p(i)))` with the prefix predecessor `p(i)` computed in finish order yields the exact optimum | additive values; prefix-compatibility; correct `p(i)` | SCH-03 is safe for the value objective | optimality under interacting values or eligibility restrictions | DERIVED here (§4.2) |
| Prefix-compatibility lemma | In finish order, `{ j < i : f_j + g <= s_i }` is exactly `{1, ..., p(i)}` | finishes non-decreasing in index; fixed `g` | predecessor is a lookup, not a search | that any ordering has this property | DERIVED here (§3.3, §4.2) |
| Interval-graph perfection | Interval graphs are perfect: chromatic number = clique number = maximum simultaneous depth | intervals with no eligibility restrictions | minimum resources equals peak depth; colouring is a sweep | anything about list colouring or non-identical resources | DERIVED here (§4.3); standard result in graph theory (interval graphs are chordal and chordal graphs are perfect) |
| `k`-machine greedy optimality | With `k` identical machines and equal values, assigning each request in finish order to the compatible machine with the largest last-finish returns a maximum-cardinality set | identical machines; equal values; finish-order scan | SCH-06 is safe for capacity questions with interchangeable resources | value objectives or machine-specific eligibility | DERIVED here (§4.3); exhaustively checked on 4,800 random instances with no mismatch |
| No constant-factor guarantee for greedy on value | For every constant `c`, there is an instance where greedy-by-finish achieves value at most `OPT / c` | additive values, one resource | a value objective requires SCH-03 or an explicit approximation bound | that greedy is always bad; it can be optimal on many instances | DERIVED here (§4.4, the `M` versus `M^2` construction) |
| Greedy-by-density has no guarantee | Same construction applies with a density ordering; the three-job instance in §4.4 loses by a factor of `8/3 ≈ 2.67` | additive values | density ordering is a heuristic only | that density ordering is never useful as a first cut | ILLUSTRATIVE computation here |
| EDD optimality (contrast) | With one resource, all jobs released at time 0, and no rejection, earliest-due-date minimises maximum lateness | no release dates; no rejection; preemption irrelevant for max lateness | EDD solves a different problem than SCH-01 | anything about count or value with rejection allowed | Textbook scheduling result; not derived here |

**On sourcing.** The results above are stated as derivations performed in this file, because the derivations are short and self-contained. The textbook attributions — the interval-scheduling greedy exchange argument, weighted interval scheduling, the perfection of interval graphs, and the EDD rule — appear in standard algorithm-design and scheduling texts, commonly Kleinberg and Tardos for the first two and any scheduling text for the last. **Check the edition and chapter numbering in the copy you actually hold before quoting a theorem number**; the numbering differs between editions and between texts, and a citation to a numbered theorem that the reader cannot find is worse than a citation to a named result. The derivation above stands on its own regardless of the citation.

**On what these theorems do not deliver.** None of them says anything about a real operation. Perfection of interval graphs is a statement about a graph class, not about a workshop where one bay is taller than another and one technician is not certified for a job type. The `k`-machine greedy is optimal only when machines are interchangeable, which is a strong assumption that most businesses violate. Stating a theorem and then applying it to a non-interval problem is the most sophisticated-looking failure in this whole topic, and it is not detectable from the output.

**Takeaway:** Each theorem licenses exactly one sentence about the schedule; the discipline is to quote the sentence it licenses and no more.

## 11. Interview / Exam Questions

### Level 1 — Definition and recall

#### Q01 — What exactly does “compatible” mean, and why is the endpoint convention part of the definition?

**Answer.** Two requests are compatible when one finishes at or before the other starts, adjusted by any declared buffer `g`: `f_i + g <= s_j` or `f_j + g <= s_i`. Under half-open intervals with `g = 0`, `[09:00,10:00)` and `[10:00,11:00)` are compatible because no instant is doubly occupied; under closed intervals they conflict at 10:00. The convention is part of the definition because the relation is not defined without it — “touching” has no truth value until you say whether endpoints are included.
**Reasoning.** The relation is the entire feasibility model; everything downstream inherits its choices.
**Acceptable alternative.** A candidate may define compatibility via strict inequality on a discretised grid, which is equivalent to half-open with a unit buffer; accept it if they state the equivalence.
**Follow-up.** What changes for a request that requires a ten-minute cleaning pass?
**Red flag.** “Overlap means the times intersect” with no endpoint rule — the answer is incomplete in exactly the place that causes production failures.
**Scoring.** 3 = relation plus convention plus buffer; 2 = relation plus one of the two; 1 = “they don't overlap”.

#### Q02 — Why does earliest-finish greedy maximise the count?

**Answer.** Because the finish time of the last accepted job is the only state that constrains future choices, and moving it right as slowly as possible maximises the remaining window. Formally, by exchange: let `g` be the earliest-finishing job and `o` the first job of an optimal schedule; since `f_g <= f_o`, substituting `g` for `o` cannot invalidate any later job that was compatible with `o`, so the substitution preserves cardinality and the argument recurses.
**Reasoning.** The exchange is applied to the *finish*, not to the job's identity, which is why any earliest-finishing choice works.
**Acceptable alternative.** A candidate may argue via the prefix property and induction on the number of accepted jobs; accept if the substitution argument is present in some form.
**Follow-up.** Where does the argument break if values differ?
**Red flag.** “Because it's a classic greedy problem” — a citation in place of an argument.
**Scoring.** 3 = exchange argument plus the monotone-state insight; 2 = correct intuition without the substitution step; 1 = restating the algorithm.

#### Q03 — What is `p(i)`, and why must it be the *largest* compatible index rather than any compatible one?

**Answer.** `p(i)` is the largest index `j < i` with `f_j + g <= s_i` in finish order. It must be the largest because the set of indices compatible with `i` is exactly the prefix `{1, ..., p(i)}` — finishes are non-decreasing in index — so `OPT(p(i))` is the best value achievable using *all* compatible jobs. Any smaller index discards compatible jobs and therefore understates the optimum.
**Reasoning.** The prefix property is what converts an exponential subset choice into one table lookup.
**Acceptable alternative.** A candidate who computes `p(i)` by binary search and explains that `bisect` returns the insertion point of `s_i` in the finish array is giving the same answer operationally.
**Follow-up.** What happens if you use `i - 1` instead?
**Red flag.** “The previous job” — that asserts consecutive jobs are compatible, which they usually are not.
**Scoring.** 3 = definition plus prefix property plus consequence; 2 = definition plus “largest”; 1 = “the job before it”.

### Level 2 — Derivation and application

#### Q04 — Derive the weighted recurrence from the definition of `OPT`.

**Answer.** `OPT(i)` is the best additive value over jobs `1..i`. Take any optimal feasible set `S` and ask whether `i` is in it. If not, `S` is feasible within `1..i-1`, so its value is at most `OPT(i-1)`, and `OPT(i-1)` is achievable, so this branch is exactly `OPT(i-1)`. If so, every other member of `S` is compatible with `i` and therefore lies in the prefix `1..p(i)`, so `S \ {i}` has value at most `OPT(p(i))` and `w_i + OPT(p(i))` is achievable; this branch is exactly `w_i + OPT(p(i))`. The two cases are exhaustive and disjoint, so the maximum of the two is `OPT(i)`, with `OPT(0) = 0` and induction on `i`.
**Reasoning.** Note what carries the proof: the prefix property does the work of excluding incompatible jobs, and additivity makes the value of `S \ {i}` independent of which job `i` is.
**Acceptable alternative.** A candidate may present the recurrence as a shortest/longest-path formulation on a DAG with edges `i -> i+1` (weight 0) and `i -> p(i)` (weight `w_i`); accept if the DAG corresponds to the case analysis.
**Follow-up.** Which assumption fails if accepting two specific jobs together earns a bonus?
**Red flag.** Writing the recurrence correctly but justifying it with “we consider both options” without showing why the take branch is exactly `w_i + OPT(p(i))`.
**Scoring.** 3 = exhaustive case analysis plus achievability plus induction; 2 = correct structure, hand-wavy achievability; 1 = formula only.

#### Q05 — Compute `p`, `OPT`, and the schedule for a four-job instance.

Given `A = [0, 3), w = 5`; `B = [2, 5), w = 8`; `C = [4, 7), w = 6`; `D = [3, 8), w = 10`, with `g = 0`.

**Answer.** Finish order is `A(3)`, `B(5)`, `C(7)`, `D(8)`, so indices are `1 = A`, `2 = B`, `3 = C`, `4 = D`. Predecessors: `p(1) = 0`; for `B` at `s = 2`, no earlier finish is `<= 2` (A finishes at 3), so `p(2) = 0`; for `C` at `s = 4`, `f_A = 3 <= 4`, `f_B = 5 > 4`, so `p(3) = 1`; for `D` at `s = 3`, `f_A = 3 <= 3`, so `p(4) = 1`. Table: `OPT(0) = 0`; `OPT(1) = max(0, 5) = 5`; `OPT(2) = max(5, 8) = 8`; `OPT(3) = max(8, 6 + 5) = 11`; `OPT(4) = max(11, 10 + 5) = 15`. Reconstruction: at `i = 4`, `10 + OPT(1) = 15 > OPT(3) = 11`, so take `D` and jump to `p(4) = 1`; at `i = 1`, `5 + 0 > 0`, so take `A`; jump to 0. Accepted `{A, D}` with value `5 + 10 = 15`, feasible because `3 <= 3`. Check: `A + C` gives 11, `B + C` is infeasible (`5 > 4`), `A + B` is infeasible (`3 > 2`), `B` alone is 8, `D` alone is 10, `A + C + D` infeasible. So 15 is optimal.
**Reasoning.** The only subtle step is `p(4) = 1`, because `D` starts exactly when `A` ends; under half-open semantics that is compatible.
**Acceptable alternative.** A candidate who enumerates all 16 subsets and reports 15 with the set `{A, D}` is correct and has the better verification habit.
**Follow-up.** Recompute with `g = 1`.
**Red flag.** `p(4) = 3` or `p(4) = 2` — evidence of comparing against the wrong instant.
**Scoring.** 3 = all four `p` values, the table, the set, and the feasibility check; 2 = correct set and value with one `p` error; 1 = value only.

#### Q06 — On the twelve-job instance, greedy achieved the maximum count but 13.33% less value than the optimum. Explain how both can be true.

**Answer.** Count and value are different objective functions, and a maximum-count set is not unique. Six four-job sets achieve count 4; their values are 26, 28, 30, 25, 26 and 24. Greedy's tie-breaking (implicitly, “earliest finish first, never reconsider”) landed on the worst of the six. Since `(30 - 26)/30 = 0.1333`, greedy was optimal for its own objective and 13.33% below optimum for the business objective. Nothing malfunctioned; the wrong function was optimised.
**Reasoning.** Optimality is always relative to an objective; the instance shows the gap being pure selection-among-optima rather than a feasibility difference.
**Acceptable alternative.** A candidate may instead give the `M` versus `M^2` construction where the count itself is worse; accept if they note that the instance here is the sharper demonstration because the count stays optimal.
**Follow-up.** What if all six sets had the same value?
**Red flag.** “Greedy is approximate so it can be 13% off” — wrong reason; the count objective was met exactly.
**Scoring.** 3 = objective relativity plus tie-set reasoning plus the arithmetic; 2 = objective relativity alone; 1 = “greedy is approximate”.

#### Q07 — Derive the minimum resource count and prove it is achievable.

**Answer.** Lower bound: at any instant `t`, every request occupying `t` must be on a distinct resource, so `k >= max_t depth(t)`. Upper bound: schedule requests in start order and assign each to any free resource. When a request starts, at most `k - 1` others are running if depth never exceeds `k`, so among `k` resources at least one is free; the assignment never fails. Therefore `k* = max_t depth(t)`, computable by a sweep with `+1` at each start and `-1` at each finish, sorting `-1` before `+1` at equal times for half-open semantics. `O(n log n)`.
**Reasoning.** The argument is a greedy existence proof; it constructs an assignment without searching.
**Acceptable alternative.** A candidate may invoke interval-graph perfection (chromatic number equals clique number); accept if they identify the clique with a set of mutually overlapping intervals.
**Follow-up.** What happens if one resource cannot handle one job type?
**Red flag.** “Bin packing, so it's NP-hard” — wrong problem; bin packing has fixed capacity per bin, this has no capacity other than non-overlap.
**Scoring.** 3 = both bounds plus the sweep plus the tie order; 2 = bounds without the constructive assignment; 1 = “take the max overlap”.

#### Q08 — The buffer is increased from 0 to 1. Predict the effect on optimum value and count before computing, then state how you would check.

**Answer.** Both are non-increasing, because every set feasible under `g = 1` is also feasible under `g = 0`, so the feasible region shrinks. On the twelve-job instance the value falls from 30 to 24 (a 20% loss) and the count from 4 to 3 (a 25% loss). The check is: recompute `p` under the new buffer from scratch, then assert `OPT_new <= OPT_old` and `count_new <= count_old`; a violation indicates a stale predecessor array, not a surprising result.
**Reasoning.** Monotonicity in a parameter is a free correctness test for exactly the class of bug that stale arrays create.
**Acceptable alternative.** A candidate may state the monotonicity and decline to guess the magnitude; that is the disciplined answer, since the magnitude is instance-specific.
**Follow-up.** Can a larger buffer ever increase the *count* under a different objective?
**Red flag.** Predicting “no change because the same jobs are on the list” — feasibility, not the list, determines acceptance.
**Scoring.** 3 = monotonicity plus the direction of both quantities plus the stale-array check; 2 = monotonicity alone; 1 = a guess.

### Level 3 — Judgement, design, and diagnosis

#### Q09 — A dispatcher reports that the optimiser's plan is feasible, but the workshop says it overbooks. What do you check, in what order, and why that order?

**Answer.** First, the endpoint convention on both sides, using a single pair where `f_A == s_B`; this is the cheapest check and the most common cause. Second, whether a buffer is contractual and whether it reached the optimiser, by grepping the model record for `g`. Third, whether the predecessor array was recomputed after the last buffer change, by asserting monotonicity in `g`. Fourth, whether the workshop's notion of occupancy includes setup or cleanup that the model does not, by comparing planned with actual durations on the jobs that overran. The order is by cost of the check and by base rate: convention and buffer account for most cases, and both are inspectable in seconds without touching data.
**Reasoning.** The discriminator for convention is a single touching pair; the discriminator for a stale predecessor is monotonicity; the discriminator for an unmodelled setup is planned-versus-actual duration.
**Acceptable alternative.** A candidate who starts by replaying the schedule against an independent feasibility checker is also correct, and arguably more general; credit it if they then localise the cause.
**Follow-up.** The workshop says the overrun happens only at shift change.
**Red flag.** “Add a buffer until it fits” — that treats a contract defect as a parameter to be tuned and hides the cause.
**Scoring.** 3 = ordered checks with a discriminative signal for each; 2 = the right checks without ordering rationale; 1 = “verify the plan”.

#### Q10 — You are asked to schedule 40 jobs on 10 identical machines by value. What do you do, and what do you refuse to claim?

**Answer.** I refuse the exact tuple-state DP: `C(40,10) = 847,660,528` states is not tractable within a planning cycle. I would formulate a time-indexed or interval MIP, solve with a time limit, and report the incumbent *and* the bound, labelling the result “feasible with a proven gap of X%” rather than “optimal”. If the business needs a fast repeatable rule, I would run the `k = 2` or `k = 3` exact DP per machine-pair or per zone where it is tractable, plus a greedy for the rest, and label the composition as a heuristic with an empirical gap measured against the MIP on a sample. What I refuse to claim is optimality without either a proof or a bound.
**Reasoning.** The exact method's infeasibility is not a matter of effort; the state count is combinatorial. Reporting a bound converts an unprovable claim into a usable one.
**Acceptable alternative.** A candidate may propose Lagrangian relaxation or column generation for the bound; accept if they are explicit that it yields a bound, not an optimum, until closed.
**Follow-up.** The business insists the plan must be reproducible byte-for-byte. How does that constrain the solver?
**Red flag.** “Just run the DP, 40 jobs is small” — the count of jobs is not the size of the state space.
**Scoring.** 3 = state-count arithmetic plus MIP with bound plus honest labelling; 2 = correct method without the arithmetic; 1 = picking a solver by reputation.

#### Q11 — Two teams produce different schedules from identical input files. Diagnose.

**Answer.** The objective values are probably identical, so the difference is a tie resolved differently. Three candidates: the sort is not total (no unique identifier in the key, or an unstable sort on duplicate keys), the reconstruction uses `>=` in one build and `>` in the other, or the input files have identical content but different physical row order and one path preserves arrival order. Check by hashing the accepted set across ten input shuffles and by diffing the comparator and reconstruction code. Fix by adding the request id as the final sort key and centralising the tie rule in one function used by both the table and the walk.
**Reasoning.** Identical value plus different set is the signature of a tie, by definition; the only question is which of the three tie mechanisms differs.
**Acceptable alternative.** A candidate may propose a canonical serialisation of the schedule and a golden-file test; that is the fix, not the diagnosis, but credit it if the diagnosis is also present.
**Follow-up.** The values differ by 3 units across the two runs. What is the cause then?
**Red flag.** “Floating point” for an integer-valued problem — plausible-sounding, and it does not explain a set-level difference at equal value.
**Scoring.** 3 = tie identification plus three mechanisms plus the shuffle test; 2 = tie identification alone; 1 = “the code differs somehow”.

#### Q12 — A candidate proposes sorting by value per hour because “the schedule should favour high-value work”. What is the strongest objection, and when would you accept the proposal?

**Answer.** The strongest objection is a concrete instance, not a principle: with `A = [0,6), w = 5`, `B = [4,6), w = 3`, `C = [4,10), w = 8`, density ordering takes `B` (density 1.5) then `C` conflicts, then `A` conflicts, for a total of 3 against an optimum of 8 — a factor of 2.67 on three jobs. I would accept density ordering as a *first cut* when the exact method cannot run (very large `n`, no bound available, or a streaming setting where the full list is not known), provided the output is labelled heuristic and an empirical gap is measured on a sample against the exact method.
**Reasoning.** Heuristics are legitimate engineering choices under constraints; the failure is claiming optimality, not using a heuristic.
**Acceptable alternative.** A candidate may counter-propose finish-order greedy with a value-aware post-pass; accept if they acknowledge it also has no guarantee.
**Follow-up.** What would you measure to justify keeping the heuristic?
**Red flag.** “Always use the DP because it's correct” — ignores tractability and streaming realities.
**Scoring.** 3 = counterexample plus the conditional acceptance; 2 = counterexample only; 1 = a principle with no instance.

#### Q13 — The plan is optimal, but the operations manager rejects it because it leaves a two-hour idle gap in the middle of the day. How do you handle this?

**Answer.** Explain that the gap is the price of a better composition: on this instance, `{A, C, F, K}` with the gap is worth 30, whereas the tightly packed `{A, C, E, J}` is worth 26, so eliminating the gap costs 13.33% of value. Then offer the manager a choice between two declared objectives — maximise value, or maximise utilisation subject to a minimum value threshold — rather than arguing about which plan is “better”. If utilisation is a genuine constraint, it becomes a constraint in the model, not a criticism of the optimum.
**Reasoning.** The disagreement is about the objective, and the productive move is to make the objective explicit and quantified, not to defend the solver.
**Acceptable alternative.** A candidate may propose a secondary objective lexicographically; accept if they state the priority and the units.
**Follow-up.** The manager says the gap confuses customers who see an idle bay. Now what?
**Red flag.** “The algorithm is provably optimal so the manager is wrong” — true about the model, useless about the decision.
**Scoring.** 3 = quantify the trade-off plus reframe as an objective choice; 2 = quantify only; 1 = defend the solver.

#### Q14 — How would you verify an interval-scheduling implementation you did not write?

**Answer.** Four layers, cheapest first. (1) Tiny fixtures with known answers: two touching jobs, two overlapping, a tie in finish time, a buffer that exactly equals a gap. (2) Oracle comparison: for random instances with `n <= 18`, enumerate all subsets, filter by feasibility with a sweep, and compare both the optimal value and the count with the implementation's output; run at least 1,000 instances across several `g` values. (3) Invariant assertions on real output: every accepted pair satisfies the predicate, the recomputed value equals the reported value, and `OPT` is non-increasing in `g`. (4) Determinism: shuffle the input ten times and assert a single hash of the accepted set. A layer that passes but the next fails localises the defect.
**Reasoning.** The oracle layer is the only one that can prove the implementation wrong rather than merely inconsistent, and it is affordable at `n <= 18`.
**Acceptable alternative.** A candidate may use a MIP solver as the oracle for larger `n`; accept if they note the solver must be proven optimal for the comparison to be valid.
**Follow-up.** The implementation passes all four layers but fails in production. What does that suggest?
**Red flag.** “Compare against the previous implementation” — two implementations can share a contract defect.
**Scoring.** 3 = all four layers with the oracle rationale; 2 = fixtures and oracle; 1 = “write unit tests”.

#### Q15 — The business wants a guarantee that no request is ever rejected. What do you tell them?

**Answer.** That the guarantee is a capacity statement, not an algorithmic one: with one resource, requests that overlap cannot both be served, so the only way to guarantee no rejection is to provision `k* = max_t depth(t)` resources, and that number is a *peak* requirement, not an average. On the twelve-job instance the average load is 3 resources (`45/15`) but the peak is 4, so a three-resource roster must reject or delay. I would also point out that the guarantee is only as good as the input: late arrivals, duration overruns, and absences all move the peak after the roster is fixed, so the real commitment should be a service level (for example, a queueing target on delayed requests) rather than an absolute no-rejection promise.
**Reasoning.** The peak-versus-average distinction is the substantive answer, and the uncertainty point is what turns an arithmetic answer into an honest one.
**Acceptable alternative.** A candidate may offer an appointment-scheduling design that shapes arrivals to flatten the peak; accept if they note the reshaping changes demand rather than capacity.
**Follow-up.** What resource count would you commit to for a 95% service level?
**Red flag.** “We'll add a priority rule so nothing is rejected” — a priority rule decides *who* is rejected, not whether.
**Scoring.** 3 = peak vs average plus the uncertainty caveat; 2 = peak vs average; 1 = “increase capacity”.

**Takeaway:** The defensible answers all name the objective, the feasibility relation, the arithmetic, and the exact condition under which the recommendation reverses.

#### 11.1 Answer-quality ledger

| Q | Difficulty | Skill exercised | Checked numeric core | Primary answer support |
|---|---|---|---|---|
| Q01 | recall | definition precision | the touching-pair boundary | §3.2, SCH-02 |
| Q02 | recall | exchange proof | — | §4.1 Claims A and B |
| Q03 | recall | structural reasoning | prefix `{1..p(i)}` | §3.3, §4.2 |
| Q04 | derivation | case analysis | `OPT(0) = 0` base | SCH-03 derivation |
| Q05 | application | full four-job trace | `p = (0,0,1,1)`, `OPT = 15` | §5 method |
| Q06 | application | objective relativity | `26` versus `30`, gap `13.33%` | §5 answer (a) |
| Q07 | derivation | two-sided bound | `k* = 4` on the instance | SCH-07, SCH-08 |
| Q08 | application | parameter monotonicity | `30 -> 24`, `4 -> 3` | §5 answer (c) |
| Q09 | judgement | ordered diagnosis | touching-pair test | §8 F-01, F-02 |
| Q10 | judgement | tractability arithmetic | `C(40,10) = 847,660,528` | SCH-11 |
| Q11 | judgement | non-determinism | six tied sets | §8 F-05 |
| Q12 | judgement | heuristic honesty | density `3` versus optimum `8` | §4.4 counterexample 1 |
| Q13 | judgement | objective reframing | `13.33%` value cost of packing | §6.2 |
| Q14 | judgement | verification design | `n <= 18` oracle limit | §7.1, §9a |
| Q15 | judgement | capacity commitment | average `3`, peak `4` | §4.3, §8 F-06 |

Fifteen questions, of which nine require calculation, proof, or diagnosis rather than recall. Each has a stated red flag and a scoring rule; none depends on a named employer's practice.
<!--PART7-->



