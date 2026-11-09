---
collection: "optimisation-operations-research-and-simulation"
file_id: "07"
topic: "Interval greedy and weighted scheduling"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A change to interval semantics, objective, resources, or tie rule."
notation_style: "plain-text"
prerequisites: ["01_optimisation_formulation.md (planned): objective and feasibility"]
source_status: "OPEN"
owner_role: "Operations research reviewer"
source_index: "INDEX.md"
---

# Interval Greedy and Weighted Scheduling

> **Purpose:** choose earliest-finish greedy only for its exact cardinality objective and derive weighted interval dynamic programming when values differ.
> **Prerequisites:** feasibility constraints, sorting, induction, and dynamic programming recurrence.
> **Key outcome:** Given half-open jobs with finish times and values, you can produce a deterministic compatible schedule, prove its boundary, and reconstruct a maximum-value schedule.
> **This file does NOT cover:** multi-machine scheduling, sequence-dependent setup, stochastic duration, or measured workforce performance.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 1.0 / not independently reviewed.
> **Evidence:** DERIVED proof and recurrence; VERIFIED hand trace; ILLUSTRATIVE jobs; OPEN original interview-source body.
> **Retrieval cue:** If a high-value long job conflicts with many short jobs, start at Section 4.4.

**Source boundary.** `directory.txt` is an inventory, not access to the original source. The focused plan [REF-IR025](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir025-choose-scheduling-algorithms-from-objective-and-constraints) requires half-open intervals, deterministic ties, the exchange argument, and the weighted-DP boundary. Values here are illustrative.

## 2. Concept Map / Mental Model

```text
[job i: start s_i, finish f_i, value w_i]
              |
              v
[write interval convention and resource count]
              |
     objective = maximize number? ---- yes ---> [sort by finish; greedy]
              |
              no
              v
[add values; sort; find compatible predecessor p(i)]
              |
              v
[OPT(i)=max(skip i, take i plus OPT(p(i)))]
              |
              v
[deterministic reconstruction and feasibility replay]
```

A scheduling algorithm is correct only relative to an objective and feasibility relation. Earliest finish leaves the largest remaining time interval and is optimal for maximizing the **number** of compatible single-resource intervals. It is not automatically optimal for revenue, priority, lateness, multiple resources, buffers, or uncertain durations.

**Takeaway:** State the objective before naming greedy or dynamic programming.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Constraint/nonexample |
|---|---|---|---|
| job `i` | requested occupancy | record | one resource in this lesson |
| `[s_i,f_i)` | half-open occupied interval | time | `s_i < f_i` |
| compatible | `f_i <= s_j` or `f_j <= s_i` | Boolean | closed intervals would treat touching jobs differently |
| `w_i` | job value | value units | may be negative only if problem permits skipping |
| `p(i)` | largest earlier index compatible with job i | integer or 0 | depends on finish-sort/tie rule |
| deterministic tie | fixed secondary sort and fixed reconstruction choice | policy | “any answer” prevents reproducibility |

Use UTC instants or an explicit local-time/calendar policy. A displayed `09:00` without a timezone is not a scheduling timestamp. With half-open intervals, `[09:00,10:00)` and `[10:00,11:00)` are compatible because the first no longer occupies the resource at 10:00.

**Takeaway:** Interval endpoint convention is a business rule with algorithmic consequences.

## 4. Core Formulas with Full Derivation

### 4.1 Cardinality greedy invariant

**Validity before use.** One resource; fixed known intervals; all jobs have equal value; feasibility is non-overlap of half-open intervals; objective is maximum cardinality. No buffers, precedence, release penalties, or weights alter the selection criterion.

**Mechanism.** Sort by nondecreasing `f_i`, then by `s_i`, then stable ID. Scan in that order. Accept a job when `s_i >= last_finish`; set `last_finish=f_i`. The invariant after each accepted job is: among schedules with the same number of accepted jobs from scanned candidates, the greedy schedule has the smallest final finish time.

**Exchange derivation.** Let greedy choose earliest-finishing job `g` and let an optimal schedule begin with `o`. Since `f_g <= f_o`, replacing `o` by `g` cannot invalidate any later job compatible with `o`. The replacement preserves cardinality. Apply the argument recursively to jobs starting at or after `f_g`.

```text
FORMULA [SCH-01]: accept(i) iff s_i >= last_finish; last_finish <- f_i
WHERE: s_i,f_i,last_finish are instants; intervals are [s_i,f_i).
CONSTRAINT: finish-order scan; one resource; equal job value.
NUMERICAL EXAMPLE: last_finish=10:00; job=[10:00,10:30); 10:00>=10:00, so accept.
INTERPRETATION: touching jobs are compatible under the declared convention.
```

**Boundary/counterexample.** With jobs A=[0,3), `w=100`; B=[0,1), `w=60`; C=[1,3), `w=60`, greedy accepts A (one job) while a weighted objective prefers B+C (value 120).

### 4.2 Weighted interval recurrence

**Validity before use.** Jobs are finish-sorted; `p(i)` is correctly computed under the same endpoint convention; one resource; additive value; skipping is allowed. This returns an exact optimum for this finite deterministic model, not for uncertain execution.

**Derivation.** An optimal schedule through job `i` either excludes it, worth `OPT(i-1)`, or includes it. If it includes `i`, all included earlier jobs must end no later than `s_i`; the best such prefix is `OPT(p(i))`. These are exhaustive and mutually exclusive choices.

```text
FORMULA [SCH-02]: OPT(i)=max(OPT(i-1), w_i + OPT(p(i))); OPT(0)=0
WHERE: OPT(i) [value units] is best value among first i finish-sorted jobs; p(i) [index].
CONSTRAINT: additive values; p(i)<i; use one consistent tie policy.
NUMERICAL EXAMPLE: OPT(2)=max(OPT(1)=60, w_2=100+OPT(0)=100)=100.
INTERPRETATION: job 2 replaces, rather than appends to, incompatible job 1.
```

Compute `p(i)` by binary search over earlier finish times after sorting, giving `O(n log n)` preprocessing. The recurrence costs `O(n)` time and `O(n)` memory; parent pointers or recomputation recover the chosen jobs.

**Boundary/counterexample.** If a mandatory 10-minute cleaning buffer follows each appointment, compatibility is `f_i + 10 min <= s_j`; using raw `f_i <= s_j` overbooks the resource.

### 4.3 Deterministic reconstruction

At index `i`, take job `i` if `w_i+OPT(p(i)) > OPT(i-1)`. On equality choose the declared policy—here, **skip on equality** to retain the earlier solution—and record the policy in output metadata. Then jump to `p(i)` after a take, otherwise decrement `i`.

**Takeaway:** Recurrence correctness depends on a correctly defined predecessor and a reproducible tie contract.

## 5. Worked Numerical Example

**ILLUSTRATIVE; VERIFIED by hand.** Finish-sort by `(finish,start,ID)`:

| i | Job | Interval | Value `w_i` | `p(i)` |
|---:|---|---|---:|---:|
| 1 | A | [0,2) | 4 | 0 |
| 2 | B | [1,4) | 7 | 0 |
| 3 | C | [3,5) | 5 | 1 |
| 4 | D | [4,7) | 8 | 2 |
| 5 | E | [5,8) | 6 | 3 |

Cardinality greedy accepts A, C, E: three jobs. Weighted recurrence: `OPT(1)=4`; `OPT(2)=7`; `OPT(3)=max(7,5+4)=9`; `OPT(4)=max(9,8+7)=15`; `OPT(5)=max(15,6+9)=15`. Reconstruction takes D then B, yielding value 15 and intervals `[1,4),[4,7)`.

Independent checks: B ends at 4 and D starts at 4, hence compatible; value `7+8=15`; no compatible pair/triple in the listed jobs exceeds 15 by enumeration: A+C+E=15 ties but skip-on-equality retains B+D. **Sensitivity:** add a 1-unit buffer: B+D becomes infeasible (`4+1 > 4`); the raw recurrence is invalid and predecessor links must be recomputed. A+C+E also becomes invalid at touch points. The decision can reverse; do not “adjust” its score afterward.

**Takeaway:** An exact answer for one feasibility definition is not an answer for a buffered operation.

## 6. Visual / Geometric Intuition

```text
time: 0   1   2   3   4   5   6   7   8
A:    [-------)
B:        [-----------)
C:                [-------)
D:                    [-----------)
E:                        [-----------)
```

Greedy pushes the right edge of the selected prefix as far left as possible. Weighted DP asks whether the value gained by extending an interval is larger than the best value relinquished before it. The predecessor arrow is the bridge from geometry to the recurrence.

**Takeaway:** Greedy protects future space; DP explicitly prices the space it gives up.

## 7. Comparison / Selection Table

| Situation | Select | Reason | Disqualifier | Verification |
|---|---|---|---|---|
| equal-value, one resource | earliest-finish greedy | exchange proof | weights/penalties | compatibility replay |
| additive values, one resource | weighted interval DP | exact recurrence | dependencies or multiple machines | enumerate small oracle |
| buffers/setup times | transform compatibility then solve | models occupancy | sequence-specific setup | replay transformed edges |
| uncertain durations | scenario/robust model | duration is random | treating estimates as fixed | holdout scenarios |
| many resources | flow/MIP/heuristic | assignments couple jobs | forcing one-resource recurrence | feasibility certificate |

**Takeaway:** Change the solver when the feasible set or objective changes, not when the answer is inconvenient.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Diagnostic | Correction | Verify |
|---|---|---|---|---|---|
| SCH-F01 | touching jobs conflict | closed endpoints vs buffer | inspect interval convention | use `[s,f)` or explicit buffer | two-job replay |
| SCH-F02 | schedule differs run-to-run | unspecified tie | log sort/reconstruction keys | declare stable ID policy | repeated trace |
| SCH-F03 | weighted greedy loses value | wrong objective | enumerate small instance | weighted DP | oracle comparison |
| SCH-F04 | DP schedule overlaps | incorrect `p(i)` | assert `f_p<=s_i` | recompute with binary search | edge replay |
| SCH-F05 | feasible plan fails live | durations/setup uncertain | compare planned/actual durations | robust buffer or reschedule policy | scenario replay |

**Takeaway:** Most interval failures are contract failures—endpoint, value, or resource assumptions—not coding speed problems.

## 9. Complexity & Cost Analysis

Sorting costs `O(n log n)`. Greedy scan costs `O(n)` time and `O(1)` selection state beyond output. Weighted scheduling with binary-search predecessors costs `O(n log n)` time and `O(n)` memory; naive predecessor search costs `O(n^2)` and becomes a bottleneck for large daily job lists. Costs exclude data validation, timezone conversion, solver integration, and dispatcher overrides.

**Takeaway:** Preserve sorted inputs and predecessor evidence so the schedule can be audited without recomputing opaque state.

## 10. Key Theorems / Results Reference

- Earliest-finish interval scheduling is optimal for maximum cardinality under the Section 4.1 assumptions (exchange argument).
- Weighted interval scheduling recurrence is optimal by exhaustive include/exclude decomposition under additive values and correct predecessors.
- Source: Kleinberg and Tardos, *Algorithm Design*, interval scheduling chapter; consult the edition used before claiming a specific theorem numbering.

**Takeaway:** The theorem states a narrow optimization problem; its assumptions are part of the result.

## 11. Interview / Exam Questions

1. Why is earliest finish optimal for count but not value? The exchange preserves remaining time, not weight; use weighted DP when values differ.
2. Are `[1,2)` and `[2,3)` compatible? Yes under half-open semantics; no if a buffer or closed interval is contractual.
3. Compute `OPT` for values 4,7,5 with predecessors 0,0,1. It is 4,7,9.
4. What does `p(i)` represent? The latest earlier compatible finish-sorted job, not merely the previous row.
5. How do you make output deterministic? Specify stable sorting and a recurrence-equality choice.
6. A scheduler uses UTC but operations use local calendars. What changes? Convert and define calendar constraints before generating intervals; do not compare display times.
7. When is an MIP more appropriate? Multiple resources, assignment constraints, sequence-dependent setup, or coupled decisions invalidate the simple recurrence.
8. What is the best small oracle? Enumerate all subsets, test compatibility, and compare objective value to algorithm output.
9. Why can a buffer reverse the answer? It changes compatibility edges and therefore every predecessor.
10. What should an API return when no feasible job remains? An explicit empty schedule plus violated constraint/retry policy, not an arbitrary overlapping job.

**Takeaway:** Interview defenses should identify the objective, feasibility relation, oracle, and changed-constraint response.

## 12. Quick Reference Card

| Decision | Rule | Assumption | Avoid | Check |
|---|---|---|---|---|
| maximize count | earliest finish | equal values, one resource | weighted jobs | exchange/replay |
| maximize additive value | `max(skip,take)` DP | correct predecessor | uncertain/multi-resource model | subset oracle |
| allow touching | half-open intervals | no buffer | service turnaround | endpoint test |
| reproduce output | fixed sort + tie policy | stable IDs | unspecified equal solutions | repeat trace |
| handle operations | model buffers/timezones | contract is explicit | raw timestamps | feasibility replay |
