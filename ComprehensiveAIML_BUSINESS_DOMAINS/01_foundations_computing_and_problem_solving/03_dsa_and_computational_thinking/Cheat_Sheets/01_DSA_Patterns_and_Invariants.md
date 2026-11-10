---
collection: "dsa-and-computational-thinking"
file_id: "01"
topic: "DSA patterns and invariants"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-08"
reviewed: null
review_trigger: "A change to the cited complexity results (binary-search comparison bound, hash expected-cost theorem, union-find bound), to a worked example's inputs, or to the collection INDEX topic ladder."
notation_style: "plain-text"
prerequisites:
  - "Basic programming: arrays, loops, recursion, integer arithmetic in a fixed-width type"
  - "Module prerequisite: ../INDEX.md (scope and topic ladder for Module 03)"
pattern_tags:
  - "invariant"
  - "two-pointers"
  - "prefix-sums"
  - "binary-search"
  - "graph-traversal"
  - "dynamic-programming"
  - "amortized-analysis"
source_status: "OPEN — no source lesson, measured benchmark, or production trace was available for this draft. Every numeric input below is a constructed teaching input, labeled ILLUSTRATIVE at the point of use."
owner_role: "curriculum maintainer"
source_index: "Cheat_Sheets/INDEX.md"
---

# DSA Patterns and Invariants

> **Purpose:** Choose a data-structure/algorithm pattern from the *contract* the problem actually imposes — input representation, operation mix, correctness obligation, resource ceiling — and prove the chosen pattern correct with a stated invariant rather than a remembered label. The sheet also teaches the reverse skill: reading a suspicious complexity claim or a failing edge case and locating the invariant that was violated.
> **Prerequisites:** Arrays and indexing from zero; loops and recursion; two's-complement integer ranges; the idea of a graph as vertices plus edges; the difference between a function's inputs and its side effects. No prior exposure to a specific algorithm library is assumed.
> **Key outcome:** Given a problem statement plus its size and resource limits, you can (a) name the pattern and the invariant it maintains, (b) derive the time and auxiliary-space bounds under stated operation-model assumptions, (c) compute the expected cost on concrete inputs, and (d) reject a plausible-but-invalid application by exhibiting either a counterexample or a failed validity condition.
> **This file does NOT cover:** proofs of the asymptotic theorems in full textbook form (statements, hypotheses and consequences are given, with the primary source named); implementation-level tuning for a particular language runtime or compiler; distributed or external-memory algorithm design, which the collection's later planned sheets and the routing/streaming briefs in the INDEX cover; and machine-learning-specific data pipelines, which belong to later domains.
> **Level:** 2 — Core Methods. Definitions and one base derivation are Level 1 material; the variants, the operation-count models and the failure diagnostics assume the reader has traced at least one algorithm by hand.
> **Version / reviewed:** 2.0 / not reviewed. `reviewed: null` — the front matter status remains `draft` and the review record stays OPEN until a named reviewer actually executes the Section 11 answer checks and the Section 5 arithmetic.
> **Evidence:** Structural claims about algorithms are `DERIVED` (from definitions and stated premises) or `SOURCED` (standard results, cited by textbook section). All workloads, per-operation costs and byte counts in Sections 5 and 9 are `ILLUSTRATIVE` teaching inputs, arithmetically checked but not measured on any machine. The absence of measurement is recorded as `OPEN`, not as a result.
> **Retrieval cue:** If you see "wrong answer only on the empty input", "works on the sample, times out on the hidden set", "loops forever when the target is absent", "fast in testing, slow under odd traffic", or "the count is off by the number of zero-length matches", start at Section 8 (Failure Modes) and use the discriminator column to pick between competing causes; then read the matching invariant in Section 4.

## 1. File Header

### 1.1 The decision this file supports

The recurring decision is not "which algorithm do I remember?" but "which obligation did I accept when I chose a representation?" Every pattern in this sheet is a *contract* with three parts: the condition under which
it produces a correct answer (validity), the quantity that shrinks or grows monotonically while it runs
(the measure that proves termination), and the resource it consumes as a function of a declared input-size
variable. Selecting a pattern from a label — "use two pointers because the problem says array" — silently
asserts all three parts. When the assertion is false the code usually still compiles, still passes the sample
case, and still returns a number; it is simply the wrong number, or the right number at the wrong cost.

Consider the concrete contrast that organises the whole sheet. Counting contiguous subarrays whose sum
equals a target `k` looks like a sliding-window problem: there is an array, there is a contiguous range, and
there is a sum. Sliding-window counting is valid when all elements are non-negative, because then the window
sum is monotone in the right endpoint and the left pointer can be advanced without loss. With a single
negative element the monotonicity disappears, the left pointer may advance past a still-valid window, and the
count comes out low by an amount that depends on the data rather than on the code. The prefix-sum-and-counter
pattern accepts no monotonicity assumption at all and is correct for arbitrary integers; it pays with an
expected O(n) auxiliary map instead of O(1) extra pointers. Two patterns, one problem statement, opposite
validity conditions and opposite resource profiles. A practitioner who knows only the labels cannot choose;
one who knows the conditions can.

### 1.2 Reader task and assumed background

The reader is expected to be able to write a loop, reason about an index range that may be empty, and read a
recurrence. Familiarity with a standard library's hash map and heap is assumed at the level of "I know what
operations it offers", not at the level of "I know its worst-case guarantees". The sheet therefore restates
the guarantees it depends on, with the assumption attached, in Section 10.

The task the reader must be able to perform after reading is deliberately narrow and testable: produce, for a
given problem, a written five-line correctness argument consisting of (i) precondition, (ii) invariant,
(iii) progress measure, (iv) termination condition, (v) postcondition-to-answer mapping. Sections 4, 5 and 11
are all built to make that five-line artefact easy to write and easy to attack.

### 1.3 Scope, exclusions and handoffs

| In scope here | Boundary | Where the boundary is handled |
|---|---|---|
| Invariant-based correctness for iteration, two pointers, binary search, BFS/DFS, Dijkstra-as-pattern, DP, heap top-k, union-find, prefix sums, amortised growth | Full formal semantics, weakest-precondition calculi, refinement types | Out of scope; the sheet uses the informal-but-checkable loop-invariant principle |
| Operation-count models with declared size variables | Micro-architectural effects (cache line size, branch prediction, SIMD width) as *measured* quantities | Section 9 states them as assumed constants, labeled ILLUSTRATIVE |
| Integer-range and overflow reasoning for indices and prefix sums | Arbitrary-precision arithmetic as a default | Section 8 treats fixed-width arithmetic as the default and demands an explicit width decision |
| Adversarial-input reasoning for hash-based patterns | Full algorithmic-complexity attack catalogue for every runtime | One worked adversarial mechanism (collision flooding) plus a discriminating check |
| Deterministic tie-breaking for outputs that are sets of equally good answers | Preference modelling, ranking fairness | Section 8 treats ties as a contract element, not a taste |

Two exclusions deserve a sentence each, because readers arrive expecting them. First, this sheet does not
teach "the 14 patterns" as a memorised taxonomy with one canonical problem per pattern; the comparison table
in Section 7 is organised by *validity condition*, so two rows can share a shape and differ in precondition.
Second, it does not benchmark. Every per-operation cost used in Sections 5 and 9 is an assumption of the form
"assume a hash probe costs about 100 ns"; the arithmetic that follows is exact, the assumption is not
evidence, and the sheet says so at each use.

### 1.4 Evidence policy for this file

| Claim class | Label used | What a reader may conclude |
|---|---|---|
| A bound follows from the stated premises by the stated derivation | `DERIVED` | The arithmetic/logic is checkable; the premises are visible |
| A standard result is quoted with its hypotheses | `SOURCED` | The textbook statement is being reported, not re-proved |
| A workload, byte count or per-operation cost is constructed for teaching | `ILLUSTRATIVE` | Arithmetic is checked; the numbers are not measurements |
| A rule of thumb is offered without proof | `HEURISTIC` | Usable, but validate on your own workload before relying on it |
| Something the draft could not establish | `OPEN` | Do not treat the gap as resolved |
| A hand-run check of an example appears in the text | `VERIFIED (arithmetic only)` | The stated substitution was recomputed; the model was not validated |

The distinction that matters most in practice is between `DERIVED` arithmetic and `VERIFIED` measurement. In
this file the two are kept apart at every number: a table row can carry an exactly-checked sum while the
per-operation cost that turns operations into seconds remains an assumption. Mixing those two categories is
the single most common way a complexity analysis becomes a false performance claim.

**Takeaway of Section 1:** the sheet's unit of knowledge is the triple (validity condition, progress measure, declared cost model), and the reader's deliverable is a five-line correctness argument plus a counterexample test for the pattern that was rejected.

## 2. Concept Map / Mental Model

### 2.1 The selection pipeline

The map below is the order in which the questions must be asked. It is a dependency arrow from top to
bottom, not a dataflow: each box constrains the next, and a failure at any box sends the reader back to
re-specify the problem rather than forward to code.

```text
[Problem statement: input representation, operation mix, output contract]
        |
        v
[Box A: what is the input size variable?]  (n? n and k? V and E? bits? rows x cols?)
        |
        v
[Box B: which obligation is mandatory?]  (exact answer / approximate / any-of-many / just feasibility)
        |                                              |
        | fails: no pattern meets the obligation        +--> [relax obligation explicitly, or reduce to a
        |                                                    solved problem, and say so in the write-up]
        v
[Box C: which structural property does the input guarantee?]
        |   monotone predicate? non-negative weights? acyclic dependencies? bounded alphabet?
        |   sparse graph? overlapping subproblems with a small state?
        v
[Box D: choose the pattern whose validity condition is implied by Box C]
        |
        v
[Box E: name the invariant and the progress measure] --> [cost model: size variables + per-op assumptions]
        |
        v
[Box F: state the failure boundary] --> [design the oracle test that would catch a wrong answer]
```

Three properties of this map are worth stating explicitly because they are where selection usually goes
wrong. Box B comes before Box C: a mandatory exactness requirement disqualifies every approximate structure
before any structural question is asked, and an obligation of "any feasible assignment" keeps greedy
patterns alive that exactness would kill. Box C is a question about the *input class*, not about the
particular test instance — the sliding-window count is valid for "all inputs in this class are non-negative",
not for "the sample I ran was non-negative". Box F is not optional polish: a pattern selected without a
stated failure boundary cannot be tested, because the test designer does not know what would count as
evidence of correctness.

### 2.2 The pattern-to-invariant map

Each pattern is paired with the single predicate that, if maintained, makes the answer correct. The right-hand
column is the assumption that the predicate usually rests on; that is the column people forget, and it is the
column Section 8 attacks.

| Pattern | Invariant maintained | Assumption the invariant rests on |
|---|---|---|
| Two pointers on sorted data | Everything left of `l` and right of `r` has been excluded from the answer for a proven reason | Ordering is total and stable for the comparison used |
| Sliding window (shrinkable) | `[l, r)` is the maximal/minimal window for the current `r` under the rule | Window aggregate is monotone in growth; elements non-negative for sum rules |
| Prefix sums + counter | `count[x]` equals the number of prefixes `i < j` with `P_i = x` | Integer prefix values compare exactly; counter updates are atomic per step |
| Binary search on index | Answer, if present, lies in `[lo, hi)` and the interval strictly shrinks | Data sorted, or predicate monotone over the index |
| Binary search on answer value | Predicate `feasible(x)` is monotone: true for all `x ≥ x*` (or all `x ≤ x*`) | Feasibility is decidable in bounded time and monotone in `x` |
| BFS | Queue holds vertices in non-decreasing distance; first discovery is minimal | Edges unweighted (or unit cost); discovery marked at enqueue time |
| Dijkstra | Every settled vertex has its final distance | All edge weights non-negative |
| Heap top-k | Heap contains exactly the best `k` seen so far under the comparison | Comparison is consistent; eviction happens against the weakest retained item |
| Dynamic programming | `dp[state]` is the optimum/count for the subproblem whose history is fully captured by `state` | Optimal substructure and acyclic dependency order |
| Greedy with exchange argument | Prefix of the greedy choice sequence is extendable to an optimum | The exchange step is provable for the specific objective |
| Union-find | Each component is represented by exactly one root; `find` returns it | Union by rank/size plus path compression are both applied |
| Amortised growth (doubling) | Capacity is always ≥ size, and total copy work over `n` appends is bounded | A geometric growth factor is used, not an additive increment |

Read the third column as a list of preconditions to check *before* writing code. "Dijkstra because it is the
shortest-path algorithm" fails the third column whenever the graph has a negative edge; the invariant that
makes Dijkstra correct is precisely the one a negative weight breaks, as Section 4.4 shows with a three-node
counterexample.

### 2.3 What changes between variants, and what never changes

Across every variant of a family, exactly one thing stays fixed: the *meaning of the state* and therefore the
invariant that justifies the answer. In the sliding-window family, variants change the shrink rule (while
sum exceeds budget, while the window is invalid, while a duplicate exists) and the recorded statistic
(longest, shortest, count with a boundary condition). They never change the underlying claim that the window
represents precisely the range currently under consideration. In the DP family, variants change the state
dimension (index only, index plus capacity, index plus last choice) and the transition order; they never
change the requirement that the state must contain every piece of history the recurrence reads.

This gives a practical diagnostic. When a variant produces a wrong answer, ask whether the invariant or the
variant-specific rule was broken. A broken shrink rule typically produces answers that are *nearly* right and
wrong only on some inputs of a recognisable shape (zeros, duplicates, ties at the boundary). A broken state
meaning produces answers that are systematically wrong — often by a constant factor or by exactly the number
of paths that share a prefix — which is a much louder signal.

The central boundary of the whole sheet is monotonicity. Every sub-quadratic pattern here is powered by one
of three monotone quantities: a shrinking interval (binary search), a non-decreasing frontier distance (BFS),
or a prefix value that can be looked up instead of scanned (prefix sums and hashing). Remove monotonicity and
the pattern does not degrade gracefully; it becomes invalid, which is why Section 4.4 is a required part of
the derivation rather than an appendix of curiosities.

```text
Monotonicity inventory (what each pattern needs, and the cheapest thing that breaks it)
  binary search ........ answer monotone in index ....... unsorted array
  binary search answer . feasibility monotone in value .. a non-monotone constraint (e.g. "at most one gap")
  sliding window ....... aggregate monotone in growth ... any negative element under a sum rule
  BFS .................. unit edge cost ................. weighted edges
  Dijkstra ............. non-negative weights ........... one negative edge
  greedy exchange ...... local choice provably safe ..... weighted objective (needs DP)
  prefix + counter ..... exact integer keys ............. floating-point prefix sums (rounding)
```

### 2.4 The proof-obligation flow

The second map is the one the reader will use when writing the correctness argument. It is a control loop:
the invariant is not a decoration added after coding, it is the thing the code is checked against, and a
failed check returns to the invariant, not to the code.

```text
[Precondition: what must be true of the input before line 1]
        |
        v
[Invariant I: a predicate over program variables that is true at loop entry]
        |
        +--> [Initialisation: does I hold before the first iteration?] --no--> repair the setup code
        |
        v
[Body: does each iteration re-establish I while making progress toward the goal?] --no--> invariant too weak/strong
        |
        v
[Measure M: a non-negative integer that strictly decreases each iteration] --> proves termination
        |
        v
[Exit: I AND (NOT guard)] --> [postcondition] --> [map postcondition to the requested answer]
        |
        v
[Oracle: an independent brute-force implementation on small inputs] --> [differential test]
```

The loop terminates at the oracle, not at the exit condition, because "I believe the postcondition" is a
hypothesis and the oracle is the cheapest available test of it. Section 5.6 and Section 11 both use this
structure: the invariant argument explains *why* the answer should be right, and a brute-force enumeration
on a small instance demonstrates that it is right for that instance.

**Takeaway of Section 2:** select by obligation and structural guarantee, then write the invariant and its progress measure before writing code — the invariant is the object you test, and monotonicity is the assumption almost every fast pattern silently borrows.

## 3. Foundational Definitions

### 3.1 Core vocabulary

The table gives the formal meaning used throughout, the type/shape the term lives in, its valid domain, the
near-neighbour it is most often confused with, and where it is used again. A reader who can distinguish the
"common confusion" column from the "formal meaning" column can already answer half of Section 11.

| Term | Formal meaning | Type / shape | Valid domain | Common confusion | Used again in |
|---|---|---|---|---|---|
| Pattern | A representation + control structure + invariant triple that is correct for a declared input class | Structural template | Any problem stated with a size variable | "Algorithm" as a named procedure | §2.2, §7 |
| Invariant | A predicate over program state that holds at every loop entry (and at the corresponding recursive entry) | Boolean over state | Must be inductive: initialisation and maintenance both hold | Assertion checked only at exit | §4.1, §10 T-01 |
| Progress measure | A function from state to a well-ordered set that strictly decreases per iteration | Natural number, usually | Must be bounded below | Loop counter that is incremented instead | §4.1, §4.3 |
| Precondition | Everything assumed true of the inputs before the algorithm runs | Predicate over input | Must be checkable or explicitly documented | Validation of hostile user input (different concern) | §5.1, §8 |
| Postcondition | What the algorithm guarantees on return, including tie-breaking | Predicate over output | Must be strong enough to imply the answer | "It returns something" | §5.6 |
| Monotone predicate | `P(x)` true for all `x ≥ x*` or for all `x ≤ x*`, for some threshold `x*` | Boolean function over ordered domain | Threshold must exist | "The function is non-decreasing" (different claim) | §4.3 F-04, §8 F-05 |
| State (DP) | The minimal tuple of values that makes the remaining subproblem independent of how it was reached | Tuple, fixed dimension | Must be finite and acyclic in dependency order | "The current array index" alone | §4.3 F-06, §9 |
| Frontier | The set of discovered-but-not-expanded vertices | Set/queue | All vertices at one or two consecutive distances | All visited vertices | §4.3 F-05, §6 |
| Amortised cost | Total cost of a sequence of `m` operations divided by `m` | Real number per operation | Requires a fixed starting state and a growth policy | Worst-case per-operation cost | §4.4 F-07, §10 T-11 |
| Auxiliary space | Extra memory beyond the input and the required output | Bytes or machine words | Excludes the input array and the returned structure | Total memory including input | §9a |
| Tie-break rule | A total order used to select one answer when several are equally good | Comparator over answers | Must be deterministic and documented | "First found" without a stated order | §8 F-08 |
| Oracle | An independent, obviously-correct implementation used to judge a fast one | Program under test harness | Must be feasible at small size | The fast implementation run twice | §5.6, §11 Q14 |
| Operation model | The unit whose count is being bounded (comparisons, probes, relaxations, byte moves) | Cost function over sizes | Must be stated with the bound | Wall-clock seconds | §9a |

### 3.2 Terms that need more than a row

**Invariant.** Formally, for a loop with guard `G` and body `B`, a predicate `I` is an invariant when
`I ∧ G → wp(B, I)` and `I` holds before the first test of `G`. The consequence is the rule the sheet uses
everywhere: on exit, `I ∧ ¬G` is available, and if that conjunction implies the postcondition, the loop is
correct. The intuition is a railway signal that must be green at every station, not only at the destination.
The most common operational failure is an invariant that is true but too weak — for example "the interval
contains the answer if the answer exists" is true for a binary search that has already discarded the answer
when the array is empty, because the conditional is vacuously satisfied. The fix is to strengthen the
predicate until it is falsifiable by the bug you fear, and then test it with an assertion in a debug build.

**Monotone predicate.** `P: D → {true, false}` is monotone (in the threshold sense) when there exists `x*`
such that `P(x)` is false for `x < x*` and true for `x ≥ x*`, over an ordered domain `D`. This is a
different claim from "`P` is a non-decreasing real function", and the difference matters: a predicate can be
monotone while the underlying quantity oscillates, as long as every oscillation stays on one side of the
threshold. The classic error is to verify monotonicity by sampling a few values and observing a trend. A
single counterexample pair — `P(x)` true and `P(x+1)` false — disproves it, so the correct verification is a
search for such a pair over the reachable domain, or a proof from the problem structure.

**State.** The DP state is the answer to "what is the smallest tuple I would have to know about the prefix I
have already processed in order to finish optimally?" It is a definition, not a data structure. The
operational test is the *two-trace test*: construct two distinct partial solutions that map to the same state
tuple but have different futures; if you can, the state is too small. A state that is too large is not
incorrect, only expensive, which is why state design is a cost decision as much as a correctness decision.

**Amortised cost.** An amortised bound of `O(1)` per append means there exists a constant `c` such that any
sequence of `m` appends starting from an empty structure costs at most `c·m` elementary operations in total.
It does not mean each append is cheap. The distinction is observable: a doubling array can copy `n/2`
elements in one particular append, and any system with a per-operation deadline (a real-time control loop, a
latency SLO with a single-request tail) will see that spike. Section 9 quantifies the spike.

**Tie-break rule.** When the problem admits several optimal answers — two shortest paths of equal length, two
subarrays of equal length, several maximal windows — the output is a set unless a rule selects one element.
The rule must be total and deterministic: "prefer the lexicographically smaller parent sequence" is a rule;
"whichever the hash iteration order produced" is not, because it changes between runs and between machines.
Interviewers use this to distinguish candidates who tested their solution from candidates who tested it once.

**Operation model.** A bound such as `O(V + E)` is meaningless until `V` and `E` are defined for the
representation in use (adjacency list versus matrix changes the edge-scan cost) and until the unit is named
(relaxations, not nanoseconds). The sheet's convention is that every complexity claim carries its unit in
parentheses on first use, for example "O(V + E) relaxations".

### 3.3 Necessary versus sufficient conditions

Two distinctions recur in Section 11 and are worth separating here. Sorted input is *sufficient* for binary
search on index but not *necessary*: a rotated sorted array still supports it with a modified predicate.
Non-negative elements are *necessary* for the shrinking-window sum pattern in its simple form, but a
window pattern can still be valid for arrays with negatives when the rule is "maintain a window containing
at most `k` distinct values", because that predicate is monotone in growth for a different reason. The
practical consequence is that a validity condition should be stated as "what the proof uses", not as "what
the examples had", because the latter is usually stronger than required and hides where the real risk sits.

**Takeaway of Section 3:** the vocabulary exists to make three questions askable — is the invariant inductive, is the state minimal, is the cost model declared — and a definition that cannot falsify a bug is not yet doing work.

## 4. Core Formulas with Full Derivation

This section carries the quantitative core. Every result is derived from a definition stated immediately
above it, and the four subsections follow the template's ladder: one base identity (4.1), its general
counting form with the operation-count consequence (4.2), the variants that practitioners actually select
between (4.3), and the limiting behaviour with counterexamples (4.4). Each block uses the five-part anatomy
and is safe to copy in isolation, because the constraint travels with the formula.

Notational conventions used below, fixed once and not redefined per block:

| Symbol | Meaning | Units / type | Domain |
|---|---|---|---|
| `n` | Number of elements in the input array, or of vertices in a graph | count | `n ≥ 0` |
| `m` | Number of edges (directed) in a graph | count | `m ≥ 0` |
| `a_i` | Element at zero-based index `i` | integer (assume 64-bit where the sheet says so) | problem-defined |
| `P_j` | Prefix sum of the first `j` elements | integer | `P_0 = 0` by definition |
| `k` | Target sum | integer | any |
| `l, r` | Left and right boundaries of a half-open range `[l, r)` | index | `0 ≤ l ≤ r ≤ n` |
| `d(v)` | Distance label of vertex `v` | integer or `∞` | `d ≥ 0` under non-negative weights |
| `α(n)` | Inverse Ackermann function | dimensionless | grows slower than any practical logarithm |
| `τ` | Cost of one elementary operation in the chosen model | operations | declared per table |

### 4.1 Base Case — the prefix-sum telescoping identity

**Validity before use:** The elements are drawn from a commutative group under addition with exact
representation in the chosen integer width (or the arithmetic is performed in a type wide enough that no
intermediate prefix overflows, which Section 8.3 shows must be checked explicitly). No monotonicity,
ordering, or non-negativity is assumed — this identity is the reason the pattern survives negative input.

**Derivation.** Define the prefix sum by the recurrence `P_0 = 0` and `P_j = P_{j-1} + a_{j-1}` for
`1 ≤ j ≤ n`, which unrolls to `P_j = Σ_{t=0}^{j-1} a_t`. For any `0 ≤ i ≤ j ≤ n`, split the defining sum at
`i`: `P_j = Σ_{t=0}^{i-1} a_t + Σ_{t=i}^{j-1} a_t = P_i + Σ_{t=i}^{j-1} a_t`. Subtracting the known
quantity `P_i` (an element of a group, so subtraction is available) gives the range sum as a difference of
two prefix values. The step that is easy to skip: this is exactly the statement that the map
`i ↦ P_i` is a group homomorphism from concatenation-with-addition to addition, and the identity holds
precisely because the operation is associative and invertible. It fails for a non-invertible aggregate such
as `max` (there is no "subtract the maximum"), which is why maximum-range queries need a different structure.

```text
[1] FORMULA: sum(i, j) = P_j - P_i          with P_0 = 0, P_j = P_{j-1} + a_{j-1}
[2] WHERE:  sum(i, j) = Σ_{t=i}^{j-1} a_t   for 0 ≤ i ≤ j ≤ n
            P [integer array of length n+1]; a [integer array of length n];
            indices half-open on the right: sum(i, i) = 0 by convention.
[3] CONSTRAINT: addition must be associative with an inverse in the working type,
            and every P_j must be exactly representable. Not valid for max/min/set-union.
[4] NUMERICAL EXAMPLE: a = [10, 20, 10, 5, 5, 20, 10, 15, 5, 20]
            P_1 = 10, P_3 = 10+20+10 = 40  (so P_3 - P_1 = 30 = a_1 + a_2 = 20+10)
            P_6 = 70, P_9 = 100            (so sum(6,9) = 30 = 10+15+5)
[5] INTERPRETATION: a range-sum query becomes two indexed reads.
            One O(n) preprocessing pass replaces an O(j-i) scan per query;
            with q queries the saving is q·(j-i) vs n + 2q operations.
```

The interpretation is worth making concrete, because it is the first place where a pattern's value is
quantifiable rather than rhetorical. With `n` elements and `q` queries over ranges of average length `L`,
the scanning baseline costs about `q·L/2` additions (average half-length if ranges are symmetric), while the
prefix form costs `n` additions to build plus one subtraction per query. For `n = 10^5`, `q = 10^5` and
`L = 10^3` the comparison is `q·L/2 = 5·10^7` additions against `n + 2q = 3·10^5` — a factor of about 167 at these illustrative sizes,
and the ratio grows with `L`. This is the honest form of "prefix sums are fast": fast relative to repeated
scanning, not fast in absolute terms.

**Takeaway of 4.1:** the telescoping identity is the only algebra the sheet needs for range aggregates, and it is available exactly when the aggregate is invertible — that single word decides between prefix sums and a segment tree.

### 4.2 General Case — counting pairs of prefixes with a fixed difference

**Validity before use:** The array elements are integers (or any type with exact equality and invertible
addition); the counter's key type holds `P_j` without truncation; the map's equality semantics are exact
(no floating-point keys, no locale-dependent string comparison). Negative elements are allowed — this is the
general case precisely because it does not need monotonicity.

**Derivation.** We want the cardinality of `S = {(i, j) : 0 ≤ i < j ≤ n, sum(i, j) = k}`. By 4.1,
`sum(i, j) = k` iff `P_j - P_i = k` iff `P_i = P_j - k`. For each right endpoint `j`, define
`C_j = |{ i : 0 ≤ i < j, P_i = P_j - k }|`. Every pair `(i, j)` with `i < j` belongs to exactly one `C_j`
(namely the one indexed by its right endpoint), so `|S| = Σ_{j=1}^{n} C_j`. The streaming algorithm computes
`C_j` with a hash map `cnt` that must contain the multiplicities of `P_0, …, P_{j-1}` at the moment `j` is
processed; hence the order of the two operations per step is fixed: *query first, insert second*. Inserting
`P_j` before querying would count the zero-length range `(j, j)`, which is not a subarray. That ordering is
the entire correctness content of the implementation, and it is the step that Section 11 Q4 asks a candidate
to defend.

```text
[2] FORMULA: count(k) = Σ_{j=1}^{n} cnt_before_j(P_j - k)
             with cnt_{j+1}[x] = cnt_j[x] + 1 if x = P_j else cnt_j[x]
[3] WHERE:  cnt_j is a multiset (map from value to count) of {P_0, …, P_{j-1}};
             P_j as defined in 4.1; count(k) is the number of non-empty contiguous
             subarrays whose element sum equals k.
[4] CONSTRAINT: exact integer keys and arithmetic; P_j within the key type;
             query-before-insert ordering; expected O(1) map operations
             (Section 10, T-09) — not worst-case O(1).
[5] NUMERICAL EXAMPLE: a = [10,20,10,5,5,20,10,15,5,20], k = 30
             P = [0,10,30,40,45,50,70,80,95,100,120]
             j=2: P_2=30, need 0  -> cnt has {0:1}           -> 1
             j=3: P_3=40, need 10 -> cnt has {0,10}          -> 1
             j=6: P_6=70, need 40 -> cnt has {...,40}        -> 1
             j=7: P_7=80, need 50 -> cnt has {...,50}        -> 1
             j=9: P_9=100,need 70 -> cnt has {...,70}        -> 1
             total = 1+1+1+1+1 = 5
[6] INTERPRETATION: O(n) expected time and O(distinct prefix values) space,
             replacing the O(n²) enumeration of all 55 subarrays for n = 10.
             At n = 10^5 the pair count of 5·10^9 candidate pairs becomes
             10^5 map operations — the pattern's whole value is this ratio.
```

Two properties of this derivation deserve separate statement, because they are where the "general" case
differs from the base case in ways that change engineering decisions.

First, the space bound is not `O(n)` in the worst case by accident — it is `O(u)` where `u` is the number of
*distinct* prefix values, and `u = n + 1` when the prefix sums are all distinct. The map therefore holds up
to `n + 1` entries, each with a key, a count and hash-table overhead. Section 9b turns that into bytes and
shows that the constant per entry is large enough to matter at `n = 5·10^7`. A bounded-value variant can
replace the map with a dense counter array of size equal to the value range, which trades a range assumption
for a large constant-factor saving.

Second, the recurrence is order-dependent in a way that forbids parallelising the step naively. The count at
step `j` depends on the multiset built from all earlier prefixes, so splitting the array into two halves and
counting each independently misses every pair that straddles the split. A correct parallel version either
(a) computes the full prefix array, then counts cross-half pairs by sorting or hashing the left prefix values
and probing with the right ones, or (b) merges per-block counters with an explicit cross term. The
single-pass formulation teaches the invariant; the parallel formulation is a different algorithm with the
same answer, and confusing them produces exactly the undercount that Section 8.4 describes.

**Takeaway of 4.2:** the general counting pattern is a bijection between pairs of prefix values and subarrays, and its two fragile details are exact-key arithmetic and the query-before-insert order.

### 4.3 Key Variants

Each variant below changes the validity condition, the recorded statistic, or the progress measure — never the
meaning of the state. The table is the selection surface; the blocks that follow it derive the two variants
whose bounds are most often misquoted.

| Variant | What changes | Added assumptions | Use when | Avoid when | Equation/result |
|---|---|---|---|---|---|
| Prefix + counter (general) | Stores multiplicities of prefix values | Exact integer keys, expected O(1) probes | Arbitrary integers, count or longest with a difference target | Memory ceiling below `u` entries; streaming with tiny memory | F-02 |
| Prefix + dense array | Counter indexed by value | Value range small and known (e.g. `[0, 10^6]`) | Sums bounded by a known modulus or budget | Sparse, wide-range sums | F-02 with array in place of map |
| Prefix modulo `m` | Key is `P_j mod m`; count pairs | Modulus arithmetic; careful with negatives | "Sum divisible by `m`" condition | The condition is not a modulus relation | F-02′ |
| Shrinking window | Left pointer advances under a rule | Aggregate monotone in growth (sum ⇒ non-negative) | Longest/shortest window under a monotone budget | Signs mixed; rule non-monotone | F-03 |
| Binary search on index | Interval `[lo, hi)` halves | Total order or monotone predicate over index | Sorted data, rotated data, peak finding | Predicate not monotone; unsorted with no structure | F-04 |
| Binary search on answer | Search over value domain | `feasible(x)` monotone, decidable in bounded time | "Minimise the maximum"; capacity/throughput thresholds | Feasibility itself is NP-hard, or ties need enumeration | F-04′ |
| BFS | Distance by first discovery | Unit edge cost; mark on enqueue | Unweighted shortest hops, level-order | Weighted edges; enormous frontier with tight memory | F-05 |
| DP over states | Recurrence over a finite state set | Optimal substructure, acyclic order | Counting/optimising with overlapping subproblems | State explosion; history not captured | F-06 |
| Heap top-k | Bounded heap of size `k` | Consistent comparator; eviction rule | `k ≪ n`, streaming, memory-bounded selection | Need full sorted order; `k` close to `n` | F-06′ |
| Amortised growth | Geometric capacity policy | Fixed growth factor ≥ 2 | Append-heavy structures with no per-op deadline | Hard latency bound on every operation | F-07 |
| Union-find | Root per component | Union by rank/size + compression | Connectivity, incremental merging | Deletions; need to enumerate a component cheaply | T-08 |

#### 4.3.1 Variant F-03: the shrinking window and its progress measure

**Validity before use:** `a_i ≥ 0` for all `i` (for sum-based rules), and the maintained aggregate is
monotone non-decreasing in the right endpoint for fixed left endpoint. Under those conditions the window
`[l, r)` is exactly the smallest window ending at `r` that violates the budget, so both the longest valid and
the shortest violating window can be read off during a single pass. With a single negative element the
monotonicity claim fails and the recorded answer may be wrong; Section 8.5 gives the discrimination test.

**Derivation.** For fixed `l`, `S(l, r) = P_r - P_l` is non-decreasing in `r` because each added element is
non-negative. Hence the set of `r` for which `S(l, r) ≤ K` is a prefix of `[l, n]`, and there is a maximal
such `r`, call it `r*(l)`. Two facts follow: (i) advancing `r` can only increase `S`, so the "shrink" step
must reduce `S` by removing elements from the left; (ii) when `l` increases, `S(l+1, r) = S(l, r) - a_l ≤
S(l, r)`, so a right endpoint that was valid for `l` remains valid for `l+1` — the right pointer never needs
to move backwards. Combining (i) and (ii), both pointers traverse their ranges at most once, giving at most
`2n` pointer advances in total. The progress measure is `M = (n - l) + (n - r)`: every iteration either
increments `r` or increments `l`, both by one, so `M` strictly decreases and the loop terminates. Non-strict
inequality in (ii) is what makes zeros harmless here — a zero element advances `l` without changing `S`,
and `M` still decreases, so termination is unaffected.

```text
[3] FORMULA: for each r: while S > K: S -= a_l; l += 1
             best = max(best, r - l + 1) after the shrink loop
             total pointer advances ≤ 2n
[4] WHERE:  S = Σ_{t=l}^{r} a_t maintained incrementally; K = budget;
             [l, r] inclusive window for this variant (watch the convention shift).
[5] CONSTRAINT: a_i ≥ 0 (monotonicity); K ≥ 0; integer arithmetic wide enough for S.
[6] NUMERICAL EXAMPLE: a = [10,20,10,5,5,20,10,15,5,20], K = 30, n = 10
             r=0: S=10  best=1
             r=1: S=30  best=2
             r=2: S=40 > 30 -> drop a0: S=30, l=1; win len 2, best stays 2
             r=3: S=35 > 30 -> drop a1: S=15, l=2; len 2
             r=4: S=20           len 3  best=3
             r=5: S=40 -> drop a2 (S=30, l=3) len 3
             r=6: S=40 -> drop a3, a4 (S=30, l=5) len 2
             r=7: S=45 -> drop a5 (S=25, l=6) len 2
             r=8: S=30           len 3  best=3
             r=9: S=50 -> drop a6, a7 (S=25, l=8) len 2
             best = 3.  Independent check: the minimum 4-window sum is 40 > 30,
             so no length-4 window is valid.  best = 3 is optimal.
[7] INTERPRETATION: O(n) time, O(1) auxiliary space, one pass, no map allocation.
             The price is the non-negativity precondition, which is a statement
             about the input class, not about this input.
```

The check in the numeric example is the kind of verification that should become automatic: after the pass,
take the reported optimum, then try to construct a better one by a *different* method. Here the different
method is "the minimum sum over all windows of length `best+1`", which is computable by a second linear pass
and gives 40 for length 4, proving that no longer window can qualify because every longer window contains a
4-window with sum at least 40. This is a proof by monotone containment, and it is available precisely because
the array is non-negative — the same check would be invalid on signed data.

#### 4.3.2 Variant F-04: half-open binary search and the exact comparison bound

**Validity before use:** the search space is an ordered index range and the predicate is monotone over it, or
the array is sorted under a total order. The version derived here is the half-open `[lo, hi)` variant with
`mid = lo + (hi - lo) / 2` using integer division, branching `lo = mid + 1` when the probe is too small and
`hi = mid` when it is not.

**Derivation of the bound.** Let `L = hi - lo`. Since `mid ∈ [lo, hi-1]`, we have `mid - lo = ⌊L/2⌋`.
The "go right" branch sets `lo = mid + 1`, giving new length `L - ⌊L/2⌋ - 1`, and the "go left" branch sets
`hi = mid`, giving new length `⌊L/2⌋`. For even `L = 2t`: left branch leaves `t`, right branch leaves `t-1`.
For odd `L = 2t+1`: left leaves `t`, right leaves `t`. In both cases the new length is at most `⌊L/2⌋`, so
after `t` iterations `L_t ≤ ⌊n / 2^t⌋`. The loop continues while `L ≥ 1`, hence it halts after at most
`⌊log2 n⌋` iterations, and the final equality check adds one probe, for `⌊log2 n⌋ + 1 = ⌈log2(n+1)⌉` probes
in the worst case. The equality is an identity on integers: for `n = 2^k` the left side is `k+1` and the
right side is `⌈log2(2^k+1)⌉ = k+1`. This exact statement matters because many sources quote either `⌈log2 n⌉`
or `⌈log2(n+1)⌉` for slightly different loop shapes; the sheet pins the variant and the count together.

```text
[4] FORMULA: probes(n) = ⌊log2 n⌋ + 1 = ⌈log2(n+1)⌉
             mid = lo + ⌊(hi - lo)/2⌋   ;  loop while lo < hi
[5] WHERE:  n = hi0 - lo0 = number of candidate indices; probes counted as
             comparisons against a[m]; the post-loop equality check counts as one.
[6] CONSTRAINT: sorted data or monotone predicate; n ≥ 1; integer arithmetic
             in a width where hi - lo does not overflow (see the mid-overflow note).
[7] NUMERICAL EXAMPLE: n = 1,000,000
             2^19 = 524,288 ≤ 1,000,000 < 2^20 = 1,048,576
             ⌊log2 10^6⌋ = 19  -> 19 iterations, 20 probes.
             At an assumed 2 µs per dependent cache miss, the probe budget is
             20 × 2 µs = 40 µs, which is the number a latency budget must cover.
[8] INTERPRETATION: doubling n adds exactly one probe. Going from 10^6 to 10^8
             candidates changes the worst case from 20 to 27 probes: the pattern
             is chosen for a ~30x larger problem at +35% probe cost.
```

The overflow note is not a digression; it is part of the validity condition on a 32-bit index type.
Take `lo = 1,500,000,000` and `hi = 2,000,000,000`. Then `lo + hi = 3,500,000,000 > 2^31 - 1 =
2,147,483,647`, so the naive `(lo + hi) / 2` overflows a signed 32-bit integer and yields a negative `mid`,
which then indexes out of bounds or loops forever. The safe form `lo + (hi - lo)/2` keeps every intermediate
within `2^31 - 1` because `hi - lo` is smaller than `hi`, and returns `1,750,000,000` as expected. The
corrective engineering rule is: use the safe form unconditionally, and add an assertion `lo <= mid < hi` in
debug builds, because the overflow only appears when the search space starts above about `2^30`.

**Takeaway of 4.3:** variants are distinguished by their validity condition, not their shape; the window needs non-negativity, the half-open search needs monotonicity and a safe midpoint, and each has an exact cost expression that a latency budget can consume.

### 4.4 Asymptotic / Limiting Behaviour and Counterexamples

The limits below are the ones that change a decision. Each is stated as an assumption, a diagnostic that
detects its violation, and a bounded alternative.

| Result / invariant | Required assumption | Diagnostic | Violating example | Bounded alternative |
|---|---|---|---|---|
| Prefix identity `P_j - P_i` | Invertible, exact aggregation | Compare against a scan on random ranges | `max` over a range | Sparse table, segment tree |
| Counter pattern `O(n)` expected | Expected O(1) map operations | Instrument probe counts per lookup; look for a long tail | Adversarial keys hashing to one bucket | Keyed/universal hash, randomise per process, bound input size |
| Window `O(n)` with `O(1)` space | Non-negative elements | Test with one negative element inserted | `a = [5, -3, 8]`, `K = 5` | Prefix + counter (O(n) space) |
| `⌊log2 n⌋ + 1` probes | Monotone search space | Check `pred(mid)` monotonicity on a sample grid | Non-monotone predicate from a heuristic score | Linear scan, or fix the predicate |
| BFS shortest hops | Unit edge cost | Compare against Dijkstra on a weighted instance | Weighted graph, BFS returns a longer path | Dijkstra / 0-1 BFS for weights in {0,1} |
| Dijkstra finality | Non-negative weights | Re-relax a settled vertex and observe an improvement | `s→a=1, s→b=2, b→a=-2` | Bellman–Ford, or Johnson reweighting |
| Greedy earliest-finish optimum | Single resource, intervals, maximisation of count | Try the shortest-duration rule on a 3-interval instance | `S=[2,4], A=[0,3], B=[3,6]` | Weighted-interval DP |
| Amortised `O(1)` append | Geometric growth, no per-op deadline | Measure the maximum single-append latency | Doubling at `n = 10^8` copies `1.3·10^8` items cumulatively | Preallocate, or use a chunked deque |
| `O(m α(n))` union-find | Both union by rank and compression | Disable one optimisation and re-measure depth | Chain unions without rank: depth `n` | Always apply both |

#### 4.4.1 Counterexample 1: Dijkstra with one negative edge

Take vertices `s, a, b` and directed edges `s→a` with weight 1, `s→b` with weight 2, and `b→a` with
weight `-2`. Dijkstra initialises `d = (0, ∞, ∞)`, settles `s`, relaxes to `d(a) = 1` and `d(b) = 2`, then
extracts `a` (smallest unsettled) and marks it settled. Extracting `b` next relaxes `b→a`, giving a
candidate `d(a) = 0 < 1`, which contradicts the settled status of `a`. The algorithm has no legal move: it
cannot un-settle `a` without abandoning its own invariant, and keeping `d(a) = 1` returns a wrong answer.
The correct distance to `a` is `0`, via `s→b→a`. This is the exact content of the non-negativity assumption:
the proof that a settled vertex is final uses the fact that any alternative path must cost at least as much
as the shortest unsettled label, which fails when a later edge can subtract. The bounded response is
Bellman–Ford `O(V·E)` for graphs without negative cycles, or Johnson's reweighting to make all weights
non-negative before running Dijkstra. The diagnostic is cheap: scan the weight array for a negative value
before choosing the algorithm.

#### 4.4.2 Counterexample 2: greedy shortest-first for interval scheduling

Take three half-open intervals on a single resource: `S = [2, 4]` with length 2, `A = [0, 3]` with length 3,
`B = [3, 6]` with length 3. The maximum number of mutually non-overlapping intervals is 2, achieved by
`A` then `B` (they touch at 3, and half-open intervals do not overlap at a shared endpoint). Sorting by
*shortest duration* selects `S` first, after which both `A` and `B` overlap it, and the greedy returns 1.
Sorting by *earliest finish* selects `A` (finish 3), then `B` (start 3 ≥ 3), and returns 2, which is optimal.
The exchange argument that proves earliest-finish optimal — replace the first interval of any optimal
solution by the one with the earliest finish; the replacement cannot conflict with the rest because it
finishes no later, and the count is unchanged — does not apply to duration, because replacing by a shorter
interval can *increase* the finish time. The general lesson: a greedy rule is only as good as its exchange
step, and "prefer the cheap item" is a rule without one.

#### 4.4.3 Limit 3: the amortised append and its single-operation spike

Aggregate analysis: with capacity starting at 1 and doubling, the copies performed by `n` appends total
`1 + 2 + 4 + … + 2^{K-1} = 2^K - 1 ≤ n`, where `2^K` is the final capacity, so the amortised copy cost per
append is below 1 and the amortised total (insert + copy) is below 3. The potential method gives the same
number per operation: with `Φ = 2·size - capacity`, `Φ ≥ 0` at every capacity boundary and `Φ` rises by 2 per
cheap append, so the amortised cost `ĉ = 1 + ΔΦ` is at most 3. Both derivations agree, and both are silent
about the individual operation that copies `2^{K-1}` items. Section 9b computes that spike in bytes for
`n = 10^8`; the engineering consequence is that amortised bounds are appropriate for throughput planning and
inappropriate for a per-request latency guarantee, and a document that uses one for the other is wrong even
though the arithmetic is right.

```text
[5] FORMULA: total copies(n) = 2^⌈log2 n⌉ - 1 ≤ 2n - 1 ; amortised copies < 1 per append
             peak transient bytes = capacity_new·s + capacity_old·s   with s = element size
[6] WHERE:  doubling policy from capacity 1; n appends; s bytes per element;
             peak occurs inside the copy, before the old block is released.
[7] CONSTRAINT: geometric growth (factor 2 here); no per-append deadline; allocator
             able to hold both blocks simultaneously.
[8] NUMERICAL EXAMPLE: n = 10^8 appends of 4-byte ints. Final capacity 2^27 = 134,217,728.
             copies = 2^27 - 1 = 134,217,727 (< 2·10^8).
             peak = (2^27 + 2^26)·4 B = 201,326,592·4 = 805,306,368 B = 768 MiB.
[9] INTERPRETATION: the throughput story is fine (≈1.34 extra copies per append);
             the tail story is not (one append touches 512 MiB). Preallocate when
             the size is known, or use a chunked structure with O(1) worst-case append.
```

#### 4.4.4 Limit 4: expected O(1) hashing is an average over hash choices

The standard expected-cost theorem (Section 10, T-09) assumes the hash function is drawn from a universal
family, or equivalently that keys are independent of the function. An adversary who knows the function can
choose keys that collide into one bucket, at which point a lookup costs `Θ(n)` comparisons. With
`n = 10^5` keys in a single bucket and `q = 10^5` lookups, the work is about `q · n / 2 = 5·10^9`
comparisons instead of `≈2·10^5` under uniform hashing — a factor of roughly 25,000. The mechanism is
identical to a quadratic algorithm and the wall-clock symptom is identical, which is why Section 8.2 makes
collision flooding and an accidental O(n²) loop compete as explanations for the same observation. The bounded
responses are: randomise the hash seed per process (so the adversary cannot precompute collisions), use a
comparison-based ordered map when keys are adversarial and the log factor is affordable, and cap the number
of distinct keys per request with an explicit rejection rather than silently degrading.

**Takeaway of Section 4:** every fast pattern is a monotonicity claim in disguise; the four limits above are the four monotonicity claims that practitioners break most often, and each has a cheap detector and a stated fallback.

## 5. Worked Numerical Example

### 5.0 Example record

```text
Example title:  Twelve-day order window audit (pattern-selection walkthrough)
Evidence label: ILLUSTRATIVE — constructed teaching input. Arithmetic is VERIFIED
                by hand and by an independent brute-force enumeration; the data,
                the costs, and the machine are not measurements of anything.
Workflow:       A planner receives a daily order-count series and must answer four
                questions from the SAME array before choosing an index structure.
Decision:       Which pattern to precompute, and what it costs at 10^7 days.
Valid assumptions: counts are non-negative integers; a "window" is a contiguous run
                of days inclusive of both endpoints; the query set is known in advance.
Intentionally simplified: no seasonality, no missing days, no late-arriving data,
                no concurrency, one machine, uniform per-operation costs.
```

The point of running four questions over one array is to expose the selection step that a
single-question example hides. A candidate who sees only "count windows summing to 15" writes a prefix
counter and moves on. A candidate who sees all four questions must decide whether one structure serves
all of them or whether the questions belong to different patterns with different costs — and the
honest answer is the second: prefix sums and the sliding window agree on nothing except the input
array, and the top-three question needs a third structure entirely.

### 5.1 The input

```text
Index i :   0   1   2   3   4   5   6   7   8   9  10  11
a[i]    :   4   9   2   7   5   3   8   1   6   2   9   4
```

Derived prefix array, computed left to right with `P_0 = 0`:

```text
j     :   0    1    2    3    4    5    6    7    8    9   10   11   12
P_j   :   0    4   13   15   22   27   30   38   39   45   47   56   60
```

Checks on the input before any query: `P_12 = 60` equals the direct sum
`4+9+2+7+5+3+8+1+6+2+9+4 = 60`, and the array length is `n = 12`, so the number of non-empty
contiguous subarrays is `n(n+1)/2 = 12·13/2 = 78`. That 78 is the size of the brute-force space every
question below must be checked against, and it is small enough that exhaustive enumeration is a
practical oracle — which is exactly why the example uses `n = 12` rather than a realistic `n`.

### 5.2 Question A — how many contiguous windows sum to exactly 15?

Apply the general counting result from 4.2 with `k = 15`. Walk `j` from 1 to 12, query the map for
`P_j - 15`, then insert `P_j`.

```text
j   P_j   P_j-15   cnt before query          hits   running total
1     4      -11   {0:1}                        0      0
2    13       -2   {0,4}                        0      0
3    15        0   {0,4,13}                     1      1
4    22        7   {0,4,13,15}                  0      1
5    27       12   {0,4,13,15,22}               0      1
6    30       15   {...,27}                     1      2
7    38       23   {...,30}                     0      2
8    39       24   {...,38}                     0      2
9    45       30   {...,39}                     1      3
10   47       32   {...,45}                     0      3
11   56       41   {...,47}                     0      3
12   60       45   {...,56}                     1      4
```

Result: 4 windows. Brute-force check: `(0,3)` sums to `4+9+2 = 15`; `(3,6)` sums to `7+5+3 = 15`;
`(6,9)` sums to `8+1+6 = 15`; `(9,12)` sums to `2+9+4 = 15`. The enumeration found exactly these four
and no others, so the streaming count and the oracle agree. The structure of the answer is a hint about
the data: four disjoint windows of length 3, each summing to 15, which is why a naive scan for "the
first window" would have reported 1 and been wrong for the question asked.

The cost at this size is trivial (12 queries), but the ratio is the point. The brute-force enumeration
examined 78 intervals; the streaming count examined 12. For `n` days the ratio grows as `n(n+1)/2`
against `n`, so at `n = 10^7` the difference is 5·10^13 interval checks versus 10^7 map operations.
The pattern is not faster by a constant; it changes the exponent.

### 5.3 Question B — what is the longest window whose total is at most 20?

Apply the shrinking window from 4.3.1 with `K = 20`. The trace is the executable form of the invariant
"`[l, r]` is the longest suffix-window ending at `r` with sum ≤ 20".

```text
r   add a[r]   S before shrink   shrink steps            l   window [l,r]   len   best
0      4            4            none                    0   [0,0]=4          1     1
1      9           13            none                    0   [0,1]=13         2     2
2      2           15            none                    0   [0,2]=15         3     3
3      7           22            drop a0 (22-4=18)       1   [1,3]=18         3     3
4      5           23            drop a1 (23-9=14)       2   [2,4]=14         3     3
5      3           17            none                    2   [2,5]=17         4     4
6      8           25            drop a2 (25-2=23)       3   [3,6]=23 >20
                                 drop a3 (23-7=16)       4   [4,6]=16         3     4
7      1           17            none                    4   [4,7]=17         4     4
8      6           23            drop a4 (23-5=18)       5   [5,8]=18         4     4
9      2           20            none                    5   [5,9]=20         5     5
10     9           29            drop a5 (29-3=26)       6   [6,10]=26 >20
                                 drop a6 (26-8=18)       7   [7,10]=18        4     5
11     4           22            drop a7 (22-1=21)       8   [8,11]=21 >20
                                 drop a8 (21-6=15)       9   [9,11]=15        3     5
```

Result: 5, achieved by `[5, 9] = [3, 8, 1, 6, 2]` summing to 20. Independent check: there are
`12 - 6 + 1 = 7` windows of length 6, with sums 30, 34, 26, 30, 25, 29, 30 — the minimum is 25, which
already exceeds 20. Under non-negativity a window of length `L > 6` contains a length-6 sub-window whose
sum is at most its own sum, so every window longer than 6 also exceeds 20; hence 5 is optimal. Note the
check is one-sided: it proves no *longer* window works, which is exactly the claim being made. It is also
weaker in its assumptions than the pattern it checks — computing seven concrete sums needs no
monotonicity argument at all, because each sum is computed directly rather than inferred. That is the
property to look for in any independent check: it must be valid under weaker assumptions than the thing
it checks, or it will happily confirm a bug.

### 5.4 Question C — what is the smallest capacity C such that some 5-day window averages at most C?

This is binary search on the answer (variant F-04′), not on an index. The predicate
`feasible(C) = "there exists a window of length exactly 5 with sum ≤ 5C"` is monotone in `C` because
`5C` increases with `C`, so the set of feasible `C` is an upward-closed interval `[C*, ∞)`.

`feasible(C)` is evaluated by computing the minimum 5-window sum once. Scanning:

```text
window start i   elements            sum
0                4,9,2,7,5            27
1                9,2,7,5,3            26
2                2,7,5,3,8            25
3                7,5,3,8,1            24
4                5,3,8,1,6            23
5                3,8,1,6,2            20   <- minimum
6                8,1,6,2,9            26
7                1,6,2,9,4            22
```

The minimum is 20 at `i = 5`, so `C* = 20/5 = 4` exactly, and `feasible(C)` is true iff `5C ≥ 20`, i.e.
iff `C ≥ 4`. Binary search over `C ∈ [0, 9]` with a monotone predicate:

```text
lo=0 hi=9  mid=4  feasible(4): 20 <= 20  true   -> hi=4
lo=0 hi=4  mid=2  feasible(2): 20 <= 10  false  -> lo=3
lo=3 hi=4  mid=3  feasible(3): 20 <= 15  false  -> lo=4
lo=4 hi=4  stop.  answer 4
```

Three probes for a domain of ten values; `⌊log2 10⌋ + 1 = 4` is the worst case, and this instance
happened to take 3. The example is worth reading slowly because it shows the two binary searches
behaving differently: index search over 12 elements needed up to `⌊log2 12⌋ + 1 = 4` probes, and the
answer search over 10 values needed up to 4 as well — the bound depends on the size of the *search
space*, not on whether the space consists of indices or of candidate values.

The decision consequence is concrete: if the capacity unit is "pickers available", then 3 pickers
cannot serve any 5-day stretch at the stated rate (the best stretch still needs 4.0), while 4 can serve
the best one and nothing worse. The subtlety an interviewer will probe: the answer 4 says *some* window
averages 4, not that *every* window can be served by 4 — the maximum 5-window mean is `27/5 = 5.4`.
Reporting 4 as "the required capacity" without saying "for the best-stretch question" is a
misstatement of the estimate, not an arithmetic error.

### 5.5 Question D — which three days had the highest counts?

A bounded heap of size 3 over the stream. Insert until full, then compare each new element against the
current minimum of the heap and replace only when strictly greater.

```text
day  count  heap (as a set)          action
0      4     {4}                     fill
1      9     {4,9}                   fill
2      2     {2,4,9}                 fill (min 2)
3      7     {4,7,9}                 evict 2, insert 7
4      5     {5,7,9}                 evict 4, insert 5
5      3     {5,7,9}                 reject (3 <= 5)
6      8     {7,8,9}                 evict 5, insert 8
7      1     {7,8,9}                 reject (1 <= 7)
8      6     {7,8,9}                 reject (6 <= 7)
9      2     {7,8,9}                 reject (2 <= 7)
10     9     {8,9,9}                 evict 7, insert 9
11     4     {8,9,9}                 reject (4 <= 8)
```

Result: counts 9 (day 1), 9 (day 10), 8 (day 6). The brute-force oracle is `sorted(a)[-3:] = [8, 9, 9]`,
which matches. Note that the heap holds values, not indices, so the answer as stated — "which three
days" — is not recoverable from the heap alone: there are two days with 9 and the heap cannot say
which. Storing `(count, index)` pairs with a tie-break on index fixes this, and the tie-break rule must
be stated in the postcondition. This is the smallest possible illustration of why tie-breaking is part
of the contract rather than a detail: the pattern is correct for the question "what are the three
largest values" and incomplete for the question "which three days".

Cost: `n` comparisons plus at most `n` heap operations of `O(log k)`, so `O(n log k)` time and `O(k)`
space. For `k = 3` that is a constant factor of about 1.6 comparisons per element; for `k = n/2` it
degrades to `O(n log n)`, at which point a partial selection or a full sort is the better choice.

### 5.6 Independent checks and the wrong path

| Check | Method | Result |
|---|---|---|
| Answer A | Exhaustive enumeration of 78 intervals | 4 windows, same four |
| Answer B | Minimum 6-window sum > 20 for all six windows | no length-6 window; 5 optimal |
| Answer C | Direct computation of all eight 5-window means | minimum is 4.0; predicate threshold confirmed |
| Answer D | `sorted(a)[-3:]` | `[8, 9, 9]`, matching the heap |
| Prefix array | `P_12` versus direct sum | both 60 |
| Pointer count | Total advances in Question B trace | 12 additions + 9 drops = 21 ≤ 2n = 24 |

The pointer count is the check most often skipped, and it is the one that catches the "almost linear"
window: an implementation that advances `r` and then rescans from `l` on every shrink is `O(n²)` while
producing identical answers. Counting advances against the `2n` bound distinguishes an implementation
that respects the invariant's cost from one that only respects its correctness.

The wrong path worth recording is the tempting one: applying the shrinking window to Question A.
Sliding a window with the rule "shrink while the sum exceeds 15" and counting windows whose sum equals
15 finds only windows that are *maximal* under the budget, and misses `(3,6)` and `(6,9)` depending on
the shrink timing. Concretely, with `K = 15` the window ending at `r = 3` has sum 15 and would be
counted, but the window `(3,6)` is discovered only by starting a fresh window at `i = 3`, which the
single-pass window never does. The misleading partial result is "2 or 3 windows", which looks plausible;
the correction is to use the prefix-and-counter pattern, which has no monotonicity precondition and
enumerates all pairs implicitly. Running the wrong pattern produced a number, and only the oracle
exposed it — this is why Section 5.6 exists before Section 11's questions.

### 5.7 Production-scale reinterpretation

Replace 12 days with 10^7 days (about 27,000 years of daily data, or — more usably — 10^7 events in a
stream). Everything above scales as follows, with per-operation costs labeled ILLUSTRATIVE:

```text
n = 10^7, element width 8 bytes (int64)
Input array bytes        = 10^7 * 8 = 8.0*10^7 B = 76.29 MiB       (1 MiB = 1048576 B)
Prefix array bytes       = (10^7 + 1) * 8 = 80,000,008 B = 76.29 MiB
Counter map, worst case  = n + 1 distinct keys, at ~48 B/entry (key+count+overhead)
                         = (10^7 + 1) * 48 = 4.8*10^8 B = 457.76 MiB
Total resident for A     = 76.29 + 76.29 + 457.76 = 610.34 MiB  (plus allocator slack)
Total resident for B     = 76.29 MiB (O(1) auxiliary: 4 integers)
Brute-force interval work = n(n+1)/2 = 5.0*10^13 interval checks
Streaming work           = n = 10^7 map operations
Ratio                    = 5.0*10^6 to 1
```

The map's constant factor is the headline, not the asymptotics. At `n = 10^7` the counter pattern needs
roughly 458 MiB of map storage for 80 MiB of data — a 6:1 overhead ratio driven by per-entry metadata,
not by the counts themselves. Two consequences follow: the same query on a memory-constrained worker
should use the dense-array variant if the value range permits (a `10^7`-entry count array of 4-byte
counts is 40 MB, an order of magnitude smaller), and the streaming implementation must decide *before*
allocation whether `u` distinct keys is acceptable. Both numbers are arithmetic on assumed entry sizes;
the honest statement is "at 48 bytes per entry this is the budget", and a reader with a different map
implementation should substitute its measured entry size.

**Takeaway of Section 5:** one array, four questions, three different patterns and one rejected pattern — the selection step is the deliverable, and only an independent oracle distinguishes a wrong pattern from a right one.

## 6. Visual / Geometric Intuition

### 6.1 The index line and the three interval conventions

The diagram is the single most useful picture in the sheet, because most off-by-one bugs are a mismatch
between two conventions drawn on the same line.

```text
Array of n = 6 elements, indices 0..5

  value:    4      9      2      7      5      3
  index:  [0]    [1]    [2]    [3]    [4]    [5]

Prefix positions (there are n+1 of them, each BETWEEN elements):
  P0   P1     P2     P3     P4     P5     P6
   |    |      |      |      |      |      |
   0    4     13     15     22     27     30
   ^                    ^
   |                    |
  P_i = 4  ---------->  P_j = 22   ==>  sum(i,j) = sum[1..3) = a1+a2+a3 = 9+2+7 = 18
                                        and indeed 22 - 4 = 18

Three conventions on the same picture (do not mix them in one program):
  (a) half-open [i, j)   : covers positions i .. j-1 ; uses P_j - P_i  ; empty when i == j
  (b) closed    [i, j]   : covers positions i .. j   ; uses P_{j+1} - P_i
  (c) both inclusive in prefix terms (i, j) : uses P_j - P_i but means elements i+1 .. j-1
```

The key point is geometric rather than algebraic: prefix sums live on the *gaps between* elements, so
there are `n+1` of them, and an interval is a pair of gaps. Every "off by one" in a prefix-sum
implementation is a program that used element indices where gap indices were required. The visual
diagnostic is to draw the two vertical bars and check that the shaded region between them matches the
elements the code claims to include. If the code says `P[j] - P[i]` then the shaded region must start
at element `i` and end at element `j-1`; if the author intended the closed interval `[i, j]` the
expression must be `P[j+1] - P[i]`.

### 6.2 The window as a stencil sliding over the line

```text
Non-negative array, budget K; window is [l, r] and S = sum(l..r)

 r ->  0    1    2    3    4    5    6    7    8    9   10   11
      [4]   9    2    7    5    3    8    1    6    2    9    4     r=0, l=0, S=4   len1
      [4    9]  2    7    5    3    8    1    6    2    9    4     r=1, l=0, S=13  len2
      [4    9    2]  7    5    3    8    1    6    2    9    4     r=2, l=0, S=15  len3
       4   [9    2    7]  5    3    8    1    6    2    9    4     r=3, S=22>20 -> l=1, S=18
       4    9   [2    7    5]  3    8    1    6    2    9    4     r=4, S=23>20 -> l=2, S=14
       4    9   [2    7    5    3]  8    1    6    2    9    4     r=5, l=2, S=17  len4
       4    9    2    7   [5    3    8]  1    6    2    9    4     r=6, S=25 -> l=3 S=23 -> l=4 S=16
       4    9    2    7   [5    3    8    1]  6    2    9    4     r=7, l=4, S=17  len4
       4    9    2    7    5   [3    8    1    6]  2    9    4     r=8, S=23 -> l=5, S=18 len4
       4    9    2    7    5   [3    8    1    6    2]  9    4     r=9, l=5, S=20  len5  <== BEST
       4    9    2    7    5    3    8   [1    6    2    9]  4     r=10, S=29 -> l=6 S=26 -> l=7 S=18
       4    9    2    7    5    3    8    1    6   [2    9    4]  r=11, S=22 -> l=8 S=21 -> l=9 S=15
```

Read the stencil vertically: the left edge `l` only moves right, and in the whole pass it moves from 0
to 9 across 12 iterations, which is the `2n` bound made visible. A signed array breaks the picture at
the first shrink step, because after dropping `a_1 = 9` from `S = 23` the sum falls to 14 — and an
earlier window that had been rejected at `S = 23` might have become valid had it been kept. The
stencil's left edge can never go back, so the rejected window is lost.

### 6.3 Binary search as a halving interval

```text
Search for value 7 in sorted [1,3,5,7,9,11,13,15], half-open [lo, hi)

lo=0 hi=8  |---------[--------]         mid=4, a[4]=9  > 7  -> hi=4
lo=0 hi=4  [---|----]                   mid=2, a[2]=5  < 7  -> lo=3
lo=3 hi=4        [|]                    mid=3, a[3]=7  == 7 -> found at index 3
                                        probes = 3

Unsuccessful search for value 8:
lo=0 hi=8  mid=4 a[4]=9 >8 -> hi=4
lo=0 hi=4  mid=2 a[2]=5 <8 -> lo=3
lo=3 hi=4  mid=3 a[3]=7 <8 -> lo=4
lo=4 hi=4  loop ends; lo is the insertion point (first index with a[i] >= 8)

Invariant: if 8 is present, its index lies in [lo, hi).
   after 1st step: 8 < 9 = a[4] so index < 4, interval [0,4) correct
   after 2nd step: 8 > 5 = a[2] so index > 2, interval [3,4) correct
   after 3rd step: 8 > 7 = a[3] so index > 3, interval [4,4) correct but EMPTY
Exit with an empty interval is the correct terminal state for "absent", not a bug.
```

The diagram makes the difference between "not found" and "failure" visual: an empty interval with
`lo = hi` is the search's *successful* conclusion that the key is absent, and any implementation that
treats `lo == hi` as an error has confused the two. The second thing the picture shows is why the
invariant must be half-open: the closed convention `[lo, hi]` needs a different update (`hi = mid - 1`)
and a different terminal test, and mixing the two produces the infinite loop that Section 8.1 diagnoses.

### 6.4 The BFS frontier as a distance contour

```text
Unweighted graph:  A-B, A-C, B-D, C-D, D-E, C-E

        (d=2)    D ------- E (d=2 or d=2? see below)
                  |  \      |
        (d=1)     B   \     |
                  |    \    |
        (d=0)     A ------- C (d=1)


BFS queue evolution from A:
  step 0: queue [A]            dist A=0
  step 1: dequeue A, enqueue B,C          dist B=1, C=1    queue [B,C]
  step 2: dequeue B, enqueue D            dist D=2         queue [C,D]
  step 3: dequeue C, D already discovered -> skip; enqueue E  dist E=2  queue [D,E]
  step 4: dequeue D, E already discovered -> skip            queue [E]
  step 5: dequeue E, nothing new                             queue []

Distance contour: {A}=0 | {B,C}=1 | {D,E}=2
Both A-B-D and A-C-D have 2 edges -> tie; the parent recorded first (B) wins.
```

The contour is the invariant made visible: vertices are dequeued in non-decreasing distance order, so
the queue always contains a suffix of one distance level followed by a prefix of the next — never a
scattered mixture. That is why "mark on enqueue" is required: if `C` were allowed to enqueue `D` before
`B` dequeues it, `D` could receive distance 2 twice, and the second write is at best wasted work and at
worst a correctness bug if the second path is longer. The tie between `A-B-D` and `A-C-D` is the same
tie-breaking contract from Section 5.5 appearing in a different pattern.

### 6.5 The DP dependency lattice

```text
0/1 knapsack over items 1..3 with weight capacity axis, dependencies point down-left

        cap:  0    1    2    3    4    5
  i=0        0    0    0    0    0    0
  i=1 (w=2,v=3)  0   0    3    3    3    3
  i=2 (w=3,v=4)  0   0    3    4    4    7
  i=3 (w=1,v=2)  0   2    3    5    6    7

  cell (i,c) reads (i-1, c)      [skip item i]
            and (i-1, c-w_i)     [take item i]

  valid fill order: rows top to bottom (i increasing), any column order within a row
  invalid: column 3 of row 2 before column 0 of row 1 -> reads an uninitialised cell
```

The lattice picture answers the question "why is this DP's order forced?" Because every cell reads only
from the row above, filling row by row is a topological order of the dependency graph. In a
one-dimensional in-place optimisation the read `(i-1, c-w_i)` becomes `dp[c-w_i]` in the *same* array,
which is why that optimisation must iterate capacity *downwards*: iterating upwards would let item `i`
be used twice, silently converting 0/1 knapsack into unbounded knapsack. The diagram is the reason the
one-line change matters, and it is the mechanism behind failure F-06 in Section 8.

### 6.6 The amortised-growth sawtooth

```text
Copy work per append when capacity doubles from 1 (n = 16 appends, drawn to scale)

appends: 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
copies : 0  1  0  2  0  0  0  4  0  0  0  0  0  0  0  8
         |  |     ||          ||||                    ||||||||
         .  .     . .         . . . .                 . . . . . . . .

cumulative copies after n appends:  0 1 1 3 3 3 3 7 7 7 7 7 7 7 7 15
cumulative / n (amortised)      :  0 0.5 0.33 0.75 0.6 0.5 0.43 0.875 0.78 ...  0.94 at n=16
bound: total copies <= 2n - 1, so amortised < 2, and with insert cost 1, total < 3
```

The sawtooth is the visual form of the Section 4.4.3 result, and its practical reading is a warning
about percentiles: at `n = 16` the worst single append performs 8 copies while the average performs
0.94. A system with a 1 ms per-append budget must plan for 8 copies, not 0.94, and the ratio between
the tail and the average widens as `n` grows (the spike scales with `n/2` while the average stays below
2). This is why "amortised O(1)" and "low latency" are different requirements with different evidence.

**Takeaway of Section 6:** each picture fixes one convention that prose leaves ambiguous — prefix gaps versus element indices, the one-way window edge, the half-open empty interval, the level-ordered queue, the row-wise DP order, and the amortised spike.

## 7. Comparison / Selection Table

### 7.1 Pattern comparison under one workload model

The table compares patterns on the same hypothetical workload class: `n` elements or `|V|`/`|E|` graph,
`q` queries or a single pass, and a memory ceiling. Costs are bounds in the stated unit; the "evidence"
column records whether the bound is derived here, quoted, or assumed.

| Pattern | Preconditions | Guarantee | Time (unit) | Space | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|---|
| Naive scan per query | None | Exact | `O(n)` comparisons per query, `O(nq)` total | `O(1)` aux | `n` tiny, or one query only | `q` and `n` both large | DERIVED |
| Sort + binary search | Comparable keys, bulk queries | Exact, `⌊log2 n⌋+1` probes | `O(n log n)` build + `O(q log n)` probes | `O(n)` for the sorted copy | Static data, many lookups | Data changes every query (rebuild dominates) | DERIVED |
| Hash map lookup | Exact equality, hashable keys | Expected exact; worst-case `Θ(n)` per op | Expected `O(1)` per op | `O(n)` entries | Point lookups, counting | Adversarial keys, ordered iteration needed | SOURCED T-09 |
| Prefix sums | Invertible associative aggregate | Exact | `O(n)` build, `O(1)` per range query | `O(n)` (or `O(1)` with a rolling window) | Static array, range sums | Updates; non-invertible aggregate | DERIVED 4.1 |
| Difference array | Invertible aggregate, offline updates | Exact | `O(1)` per range update, `O(n)` materialise | `O(n)` | Many range adds, one final read | Interleaved updates and queries | DERIVED |
| Fenwick tree | Invertible aggregate, point updates | Exact | `O(log n)` update and query | `O(n)` | Interleaved point updates and range queries | Range updates on arbitrary aggregates | SOURCED |
| Segment tree | Associative aggregate (max/min ok) | Exact | `O(log n)` update/query, `O(n)` build | `O(4n)` typical | Range min/max, lazy range updates | Tiny `n`; memory-critical | SOURCED |
| Sparse table | Idempotent aggregate (min/max/gcd) | Exact | `O(n log n)` build, `O(1)` query | `O(n log n)` | Static range min/max, many queries | Updates; memory ceiling below `n log n` | SOURCED |
| Sorted array + two pointers | Total order | Exact | `O(n log n)` sort + `O(n)` scan | `O(1)` aux after sort | Pair/triple-sum on sorted data | Unsorted and sort is disallowed | DERIVED |
| Sliding window | Monotone aggregate in growth | Exact under precondition | `O(n)` | `O(1)` or `O(alphabet)` | Longest/shortest window under budget | Signed data with sum rules | DERIVED 4.3.1 |
| BFS | Unit edge cost | Shortest hop count | `O(V + E)` dequeues/relaxations | `O(V)` queue + visited | Unweighted shortest path, levels | Weighted edges; frontier too large | DERIVED |
| 0-1 BFS | Weights in `{0, 1}` | Exact shortest path | `O(V + E)` with a deque | `O(V)` | Two-tier costs | Weights outside `{0,1}` | SOURCED |
| Dijkstra (binary heap) | Non-negative weights | Exact shortest path | `O((V + E) log V)` | `O(V + E)` | Weighted shortest path, one source | Negative edges (see 4.4.1) | SOURCED |
| Bellman–Ford | No negative cycle | Exact, detects negative cycles | `O(V·E)` | `O(V)` | Negative weights, cycle detection | Large dense graphs (cost blows up) | SOURCED |
| Top-k heap | Consistent comparator | Exact `k` best | `O(n log k)` | `O(k)` | Streaming selection, `k ≪ n` | `k` close to `n` (sort instead) | DERIVED |
| Quickselect | Mutable array, random pivots | Expected exact; `O(n²)` worst | Expected `O(n)`, worst `O(n²)` | `O(1)` aux | Offline k-th order statistic | Adversarial input without pivot randomisation | SOURCED |
| Union-find | Incremental merges only | Exact connectivity | `O(m α(n))` amortised | `O(n)` | Kruskal, dynamic connectivity | Deletions; need component enumeration | SOURCED T-08 |
| DP over states | Optimal substructure + acyclic state graph | Exact optimum/count | `O(#states × #transitions)` | `O(#states)` (or frontier) | Counting, optimisation with overlap | State explosion; history not captured | DERIVED |
| Greedy + exchange proof | Provable exchange step | Exact for the proven objective | Usually `O(n log n)` (sorting) | `O(1)`–`O(n)` | Interval scheduling, Huffman-style merges | Weighted/buffered variants (needs DP) | DERIVED 4.4.2 |

### 7.2 The selection decision guide

```text
IF the output obligation is exact AND a structural precondition above is violated
        -> do not "try anyway"; either relax the obligation explicitly or switch family.
ELSE IF there are no updates between queries and the aggregate is invertible
        -> prefix sums (O(1) per query) rather than a tree (O(log n)).
ELSE IF updates and queries interleave
        -> Fenwick for point-update/range-query on invertible aggregates;
           segment tree when the aggregate is not invertible (min/max) or updates are ranges.
ELSE IF the aggregate is idempotent and the array is static
        -> sparse table only if the memory for n log n is affordable; otherwise segment tree.
ELSE IF the problem is a monotone budget over non-negative data
        -> one-pass sliding window; if data can be signed, prefix + counter with a map.
ELSE IF the graph is unweighted or weights are in {0,1}
        -> BFS / 0-1 BFS; do not pay a heap when a deque suffices.
ELSE IF the graph has non-negative weights
        -> Dijkstra with a binary heap; add early exit at the target if only one pair is needed.
ELSE IF negative weights exist but no negative cycle
        -> Bellman–Ford, or reweight with Johnson and then Dijkstra.
ELSE IF a small k is required from a large stream
        -> bounded heap of size k; if the stream is fully resident and k is large, sort or quickselect.
ELSE IF the state space is small but the transition count is large
        -> DP with prefix-max/prefix-sum transitions to collapse the inner loop.
ELSE
        -> state the obligation, the size bound, and the resource ceiling; escalate to measurement
           rather than guessing, and record the pattern choice as provisional.
If two rules conflict -> resolve by the mandatory obligation (exactness, then hard memory ceiling,
then latency), and write down which was sacrificed.
```

The guide's structure — obligations before structures — is the same as the Section 2 map, restated as a
decision procedure. Two of its lines carry most of the practical value. The first line forbids the most
common failure of pattern-selection reasoning: choosing a fast pattern whose precondition the input
violates, then adjusting the code until the sample case passes. The line about early exit in Dijkstra
is the second: a single-pair shortest path can stop as soon as the target is settled, which on typical
road-like graphs avoids expanding a substantial fraction of the frontier, while a full single-source
run must expand everything reachable. The saving is not asymptotic (`O((V+E) log V)` in the worst case
either way) but it is often large in practice, and it is worth stating because it changes the constant
without changing the bound — a distinction the sheet keeps insisting on.

### 7.3 Selection under conflicting constraints

The conflict that recurs in interviews and in production is memory versus latency. A concrete instance:
`n = 10^7` 64-bit values, a memory ceiling of 512 MiB, `q = 10^6` range-sum queries, and a 50 ms p99
target.

| Option | Memory (assumed constants) | Query cost | Verdict |
|---|---|---|---|
| Prefix array | `8n = 76.3 MiB` | one subtraction | Fits comfortably; best latency |
| Sparse table (sum variant, invalid) | `8n log2 n ≈ 8·10^7·23.25 B = 1.73 GiB` | — | Exceeds ceiling *and* sum is not idempotent: doubly disqualified |
| Segment tree over sums | about `2·2^24·8 B = 268,435,456 B = 256 MiB` | `O(log n)` ≈ 24 node reads | Fits, but unnecessary: no updates |
| Naive scan | `0` | average `n/2 = 5·10^6` additions per query, `5·10^12` total | Hopeless at `q = 10^6` |
| Difference array | `8n` | `O(1)` range add, `O(n)` to materialise | Only correct if queries are batched after updates |

The table shows the resolution rule at work: exactness and the memory ceiling are mandatory, latency is
an objective, and the prefix array is the only option that satisfies all three because the workload has
no updates. The sparse-table row is included precisely because it is what a reader who memorised "sparse
table gives O(1) range queries" would reach for, and it fails twice over — once on memory, once on the
idempotence precondition that a sum does not satisfy. A selection table that only lists winners teaches
labels; one that lists a plausible loser with two independent disqualifiers teaches selection.

**Takeaway of Section 7:** compare patterns under one workload and let mandatory obligations eliminate options before latency preferences rank them; the most instructive rows are the plausible options that fail a precondition.

## 8. Failure Modes & Edge Cases

Each failure below is a mechanism with competing causes, not a single bug. The discriminating check column
is the operational core: it names the observation that separates the hypotheses, and the negative result's
meaning is stated so a reader knows when to stop testing and when to escalate.

### 8.1 F-01: the non-terminating binary search

| Field | Content |
|---|---|
| Trigger / symptom | The search hangs, or the process times out, only for keys that are absent or at the array boundary |
| Competing cause 1 | Mixed interval conventions: a half-open update (`hi = mid`) applied to a closed terminal test (`while lo <= hi`) |
| Competing cause 2 | Integer midpoint truncation combined with `lo = mid` (not `mid + 1`) when the probe is too small |
| Competing cause 3 | A comparator that is not a strict weak ordering (for example `<=` used where `<` is required), so the predicate is inconsistent between iterations |
| Discriminating check | Log `(lo, hi, mid)` each iteration. A non-shrinking interval shows `hi - lo` constant across two iterations with the same `mid`; an inconsistent comparator shows non-monotone decisions on adjacent probes |
| Correct action | Pin one convention and derive the update from the invariant: half-open `[lo, hi)` ⇒ too-small sets `lo = mid + 1`, too-large sets `hi = mid`. Then the length strictly decreases (proof in 4.3.2) |
| Prevention / verification | Assert `lo <= mid < hi` each iteration in debug builds; add a termination test with a step counter bounded by `⌊log2 n⌋ + 2`; run on `n = 0`, `n = 1`, absent key below the minimum, above the maximum |
| Provenance | HYPOTHETICAL teaching scenario; the mechanism is derived from the update rule |

The reason this failure resists casual debugging is that it is *conditional on the answer*: a search for a
present key often terminates even with mixed conventions, because the equality branch exits early. Only
the absent-key path runs the loop to its terminal condition, which is why "it worked in testing" and "it
hangs in production" coexist. The bounded fix is not to add an iteration cap (that converts a hang into a
wrong answer) but to restore the invariant and then rely on the derived bound to detect regressions.

### 8.2 F-02: the quadratic that looks linear — hash collisions and accidental rescans

| Field | Content |
|---|---|
| Trigger / symptom | A pass over `n` items whose measured time grows faster than `n`; CPU profile dominated by one comparison or one loop |
| Competing cause 1 | Adversarial or unlucky key distribution causing bucket collisions, making expected `O(1)` operations `Θ(n)` each |
| Competing cause 2 | An inner loop that rescans the whole structure on each iteration (for example calling a linear `contains` inside a loop over the same list) |
| Competing cause 3 | A comparator or hash that allocates (hashing a string that is rebuilt per lookup), so the cost per operation is not constant but proportional to a field length |
| Discriminating check | Instrument the number of comparisons or probes per operation and plot its distribution. Collision-driven cost shows a long tail on the *lookup* count with a fixed loop structure; an accidental rescan shows a linear relationship between iterations and work with a *uniform* per-iteration count |
| Correct action | For collisions: randomise the seed per process and cap distinct keys per unit of work; for rescans: replace the membership test with a set built once; for allocation: precompute the hash and store the key in normalised form |
| Prevention / verification | A test with keys deliberately sharing a prefix (near-miss strings) and a test with integer keys spaced by the table size; then compare operation counts, not seconds, against the expected `O(n)` total |
| Provenance | HYPOTHETICAL; the collision mechanism is DERIVED from the expected-cost theorem in Section 10 |

The two causes produce the same wall-clock symptom and require opposite fixes, which is why the
discriminator is a *distribution* rather than a mean. A uniform `n`-per-iteration count means the
algorithm is quadratic by construction and no runtime tuning will help; a heavy-tailed per-operation
count means the algorithm is linear under the intended hash and the input is hostile. Averaging the two
together produces the useless conclusion "it is sort of slow sometimes".

### 8.3 F-03: the silent integer overflow in the prefix array

| Field | Content |
|---|---|
| Trigger / symptom | A range sum returns a negative or implausibly small value; the error appears only for long ranges or large values |
| Competing cause 1 | Prefix values exceed the width of the accumulator type (`int32` with `n·max|a| > 2^31 - 1`) |
| Competing cause 2 | The subtraction `P_j - P_i` overflows even though each stored prefix fits, because the true range sum is larger than the type allows |
| Competing cause 3 | Unsigned arithmetic wrapping, so the comparison `S > K` in a window loop silently inverts |
| Discriminating check | Compute the maximum possible prefix `n · max(|a_i|)` and compare it to the type maximum before running; then compare a suspect range sum against an arbitrary-precision computation. If the wide computation agrees and the narrow one does not, the width is the cause. A wrap rather than a truncation is visible as a sign flip |
| Correct action | Widen to `int64` (or the language's 64-bit type) when `n · max|a| ≥ 2^31`; for genuinely unbounded sums use arbitrary precision and accept the cost. Do not "fix" by clamping: clamping changes the answer |
| Prevention / verification | A boundary test with `a = [2^30, 2^30, 2^30]` in `int32` (which must fail loudly or be rejected by a precondition check), and a property test asserting `sum(i,j) = -sum(j,i)` on random ranges under the same type |
| Provenance | HYPOTHETICAL; the bound is DERIVED from two's-complement range arithmetic |

The interesting part is that this failure is *silent* in every mainstream language by default, and that
it can be introduced by a refactor that changes nothing else: switching an accumulator from 64 to 32 bits
"because the values are small" is a correctness change disguised as a memory optimisation. The
discriminator is arithmetic, not empirical — the maximum prefix is computable from the input before the
run, so the check belongs in the precondition rather than in the test suite.

### 8.4 F-04: the undercount from a broken pair-ordering

| Field | Content |
|---|---|
| Trigger / symptom | A count of subarrays with a target sum is consistently low, often by exactly the number of zero-sum subarrays or the number of matches involving the first element |
| Competing cause 1 | The counter is updated before the query (`cnt[P_j] += 1` then `query(P_j - k)`), counting the empty range `(j, j)` when `k = 0` |
| Competing cause 2 | The map is initialised without `P_0 = 0`, losing every match that starts at index 0 |
| Competing cause 3 | The array was split across workers and each worker counted only its own pairs, dropping cross-boundary pairs |
| Discriminating check | Run `k = 0` on an array with no zero-sum subarray: an implementation with cause 1 returns `n` instead of `0`. Then run a target that matches only a prefix-starting window (for example `k = a_0`): cause 2 returns 0. Then halve a small array and compare the split sum against the single pass: cause 3 shows a shortfall proportional to the number of cross-boundary matches |
| Correct action | Fix the operation order and state it in the invariant ("`cnt` holds `P_0..P_{j-1}` at the time of the query"); seed with `{0: 1}`; for parallelism compute the full prefix array and count cross-half pairs explicitly rather than summing independent workers |
| Prevention / verification | A differential test against brute force on 10^4 random arrays of length ≤ 30 including zeros, negatives and duplicates; a zero-target case; a two-worker case with a hand-checkable boundary match |
| Provenance | HYPOTHETICAL; cause 3 is the mechanism described in the INDEX brief PLAN-2026-IR035 for shard-crossing counts |

Cause 1 deserves emphasis because it only manifests for `k = 0`. A test suite built entirely around
non-zero targets will pass while the zero case counts empty windows, and empty-window counting is
exactly the kind of error that survives review because the code "obviously" queries before inserting.
The general lesson is that ordering assumptions should be written where the map is declared, not implied
by the sequence of two statements fifty lines apart.

### 8.5 F-05: the sliding window applied to signed data

| Field | Content |
|---|---|
| Trigger / symptom | Longest-window or count-of-windows answers that are too small, with the error growing as the array accumulates negative values; results change when the same multiset is permuted |
| Competing cause 1 | The non-negativity precondition for the sum-based window is violated, so the left pointer advances past windows that are still valid |
| Competing cause 2 | The shrink rule is right but the answer is recorded before shrinking rather than after, so invalid windows are counted |
| Competing cause 3 | The aggregate is maintained incrementally but a reset path clears the sum without resetting `l` (or vice versa), desynchronising the state |
| Discriminating check | Insert one negative element into an otherwise non-negative test array and compare against brute force. If the answer changes while the shrink-rule code is unchanged, cause 1. Check the trace for `S` recomputed from scratch: a discrepancy between `S` and `sum(a[l..r])` at any step indicates cause 3 |
| Correct action | For cause 1, switch to prefix-and-counter (monotonicity is not required) and pay `O(n)` space, or use a monotonic deque over transformed prefix values where applicable. For cause 2, re-read the invariant's statement of what is recorded. For cause 3, recompute the aggregate in assertions |
| Prevention / verification | Property test: for every `r` after the loop body, assert `S == sum(a[l..r])`; brute-force differential test including one generated case with mixed signs, kept in the suite as a *documented expected failure* of the fast path that must be routed to the general path |
| Provenance | HYPOTHETICAL; the counterexample `a = [5, -3, 8], K = 5` is derived by hand |

The bounded fix has an architectural consequence: if the application cannot guarantee non-negative
input, the fast window should not be the code path at all, because "usually non-negative" is not a
precondition a proof can use. The honest designs are (a) validate and route, (b) always use the general
pattern and accept the memory, or (c) document the assumption and reject violations at the boundary.

### 8.6 F-06: the DP state that forgets history

| Field | Content |
|---|---|
| Trigger / symptom | A DP returns an optimal value that is *achievable in the DP but not in the problem*, or counts configurations that are not valid; often the DP is too optimistic |
| Competing cause 1 | The state omits a dimension the recurrence depends on (for example 1-D state in a problem whose transition needs the last chosen item) |
| Competing cause 2 | The state contains the needed dimensions but the fill order violates the dependency direction, so a transition reads a cell from the current layer instead of the previous one |
| Competing cause 3 | The transition is correct but the base cases are incomplete (missing the "empty prefix" row), producing systematic undercounts |
| Discriminating check | Reconstruct an optimal solution from the table and evaluate it against the original problem's constraints. If the reconstruction violates a constraint, cause 1 or 2. If it is valid but a brute-force optimum is strictly better, the state was too restrictive or a transition missing. Compare the DP's count on a tiny instance against exhaustive enumeration; a constant-difference error suggests base cases (cause 3) |
| Correct action | Apply the two-trace test from Section 3.2 to the state definition and add the missing dimension; fix the fill order to a topological order of the dependency graph; write the base row explicitly and test `n = 0` and `n = 1` |
| Prevention / verification | Mandatory small-instance differential test against exhaustive enumeration; reconstruction check on every sample; assertion that the DP value is never better than a brute-force optimum on random small inputs |
| Provenance | HYPOTHETICAL; the 1-D in-place trap is DERIVED from the knapsack dependency diagram in 6.5 |

The reconstruction check is the single most valuable DP test and the one most often omitted, because it
tests the *meaning* of the state rather than the arithmetic of the table. A DP whose table is
self-consistent but whose states do not correspond to feasible solutions produces a beautifully typed
wrong answer, and only reconstruction reveals it.

### 8.7 F-07: the amortised bound used as a latency guarantee

| Field | Content |
|---|---|
| Trigger / symptom | p99.9 latency collapses at irregular intervals while the mean is comfortable; the stalls correlate with a size crossing a power of two |
| Competing cause 1 | A geometric-growth container copies its whole buffer on one operation, and the copy is inside the request's critical path |
| Competing cause 2 | The allocator cannot serve the doubled block without compaction or a page fault, adding a second spike on top of the copy |
| Competing cause 3 | A batch flush or compaction policy that is amortised in throughput terms but not in tail terms (a log-structured index, a cache eviction sweep) |
| Discriminating check | Histogram per-operation work, not per-request latency. If a small number of operations do `O(capacity)` work, cause 1. Correlate spikes with allocated bytes and RSS: a spike that tracks RSS growth points to cause 2. If the spike period is driven by entry count rather than capacity, cause 3 |
| Correct action | Preallocate when the size is knowable; use a chunked structure with `O(1)` worst-case append; move compaction off the hot path; if none of these is possible, express the guarantee as a throughput SLO and exclude the spike explicitly |
| Prevention / verification | A latency test that runs long enough to cross at least three capacity doublings, with per-operation work plotted against operation index; an assertion that no single operation exceeds a declared work budget |
| Provenance | HYPOTHETICAL; the byte arithmetic is DERIVED in 4.4.3 |

The failure is really a specification failure: an amortised bound answers "how much work per operation
on average over a long sequence", and a latency SLO asks "how much work in this one". Both are
legitimate questions and the same measurement can support the first while refuting the second. The
verification step is therefore not a code change but a test-design change.

### 8.8 F-08: non-deterministic tie-breaking

| Field | Content |
|---|---|
| Trigger / symptom | Two runs over identical input produce different-but-equally-valid outputs; golden tests flake; a downstream diff shows reordered results |
| Competing cause 1 | The output depends on hash iteration order, which varies with the seed or the runtime version |
| Competing cause 2 | The comparison admits ties and the selection uses "first encountered", which depends on input order or on parallel scheduling |
| Competing cause 3 | Floating-point accumulation order differs between runs (for example a parallel reduction), changing which of two near-equal candidates wins |
| Discriminating check | Run the same input twice with different hash seeds or thread counts. If the difference appears only across seeds, cause 1; only across thread schedules, cause 3; only when input order is permuted, cause 2 |
| Correct action | Define a total order that breaks ties on a stable secondary key (index, id, timestamp) and sort by it before emitting; iterate maps in a sorted key order when order is observable; fix the reduction order or use an order-independent comparison |
| Prevention / verification | A determinism test that runs the pipeline twice in one process and once in a subprocess with a different seed, asserting byte-identical output; a golden test built from a *sorted canonical form* of the result |
| Provenance | HYPOTHETICAL; the mechanism follows from the tie-break definition in Section 3.2 |

The subtlety is that non-determinism is not always a bug: "any maximum matching" or "any shortest path"
is a legitimate output, and forcing a canonical order can cost real work. The decision is whether
anything downstream consumes the order. If nothing does, canonicalise once at the boundary; if
something does — a diff view, a ledger, a test suite — the total order is part of the contract and
belongs in the postcondition.

### 8.9 F-09: the unbounded frontier (BFS memory blow-up)

| Field | Content |
| --- | --- |
| Trigger / symptom | The traversal dies with an allocation failure, or the machine swaps, on a graph whose edge count is modest |
| Competing cause 1 | Marking visited on *dequeue* rather than on enqueue, so the same vertex is enqueued once per incoming edge and the queue holds `O(E)` duplicates instead of `O(V)` vertices |
| Competing cause 2 | The frontier genuinely grows to `O(V)` (a high-degree hub graph, a grid diagonal), and the per-vertex record is heavy (objects with maps instead of a compact integer array) |
| Competing cause 3 | The queue is a list with `pop(0)` (an `O(n)` shift per dequeue) and the memory pressure is accompanied by time blow-up |
| Discriminating check | Instrument queue length against step count and the number of enqueues per vertex. Enqueues greater than 1 per edge into the vertex indicates cause 1; enqueues equal to `V` with large frontier width indicates cause 2; dequeue time growing linearly with queue length indicates cause 3 |
| Correct action | Mark at enqueue and assert `enqueued[v]` is false before pushing; replace per-vertex objects with arrays indexed by a compact vertex id and, if the frontier is the constraint, use a bidirectional search or a depth-first variant with an explicit stack when the goal is reachability rather than distance |
| Prevention / verification | A synthetic graph with a high-degree hub and a long path; assert peak queue length ≤ `V`; compare against a reference BFS on a graph with 10^5 vertices |
| Provenance | HYPOTHETICAL; cause 1 is the most commonly reproduced mechanism in traversal code |

### 8.10 F-10: the greedy that is optimal for the wrong objective

| Field | Content |
| --- | --- |
| Trigger / symptom | A schedule or selection is feasible and "sensible" but measurably worse than a known bound; the gap grows with instance size |
| Competing cause 1 | The objective was changed between problem statement and implementation (maximise count versus minimise total duration) |
| Competing cause 2 | The greedy rule is correct for the objective but the tie-break is not (equal finish times handled arbitrarily, breaking the exchange argument's premise of a fixed canonical order) |
| Competing cause 3 | Constraints were added that the exchange proof does not cover (buffers between tasks, two resources, release times) |
| Discriminating check | Compare against a DP or exhaustive optimum on instances of size ≤ 12. If the gap is 0 for small instances and grows with size, suspect cause 1 (asymptotic suboptimality). If the gap appears only with ties, cause 2. If it appears only when the extra constraint binds, cause 3 |
| Correct action | Restate the objective and re-derive the exchange step for it; for weighted objectives switch to DP (weighted-interval scheduling); for added constraints either extend the proof or fall back to a general method with a stated optimality gap |
| Prevention / verification | Exhaustive-optimum comparison on all instances up to a small size; a documented counterexample in the test suite for every claimed limitation |
| Provenance | HYPOTHETICAL; the duration-versus-finish counterexample in 4.4.2 is DERIVED |

The pattern in these ten entries is deliberate: each has more than one plausible cause, and the
discriminator is an observation that a test can make cheaply. A failure catalogue whose rows read
"check your code" is a list of symptoms; one whose rows name the observation that separates the
hypotheses is a diagnostic instrument, which is what Section 8 is for.

**Takeaway of Section 8:** every failure here has a cheaper discriminating observation than a fix, and the observation is almost always a comparison against a brute-force oracle or a distribution rather than a mean.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity with declared operation models

Every entry below states its size variables and its unit of work. The unit is not decoration: `O(n log n)`
comparisons and `O(n log n)` cache misses are different engineering problems, and a table that omits the
unit invites the reader to compare a comparison bound with a byte bound.

| Operation | Size variables | Time (unit) | Auxiliary space | Dominant bottleneck under a declared assumption | Assumption |
|---|---|---|---|---|---|
| Build prefix array | `n` | `O(n)` additions | `O(n)` values (`n+1` slots) | Memory bandwidth: one sequential write per element | Sequential access is cheap |
| Range-sum query (prefix) | `n`, `q` | `O(1)` subtractions per query, `O(q)` total | `O(1)` extra | Constant per query; only the index fetch touches memory | Array is resident |
| Range-sum query (scan) | `n`, `q` | `O(n)` per query, `O(nq)` total | `O(1)` | Additions, then bandwidth | Same arithmetic cost per element |
| Difference array build | `n`, `u` updates | `O(u)` additions, `O(n)` materialise | `O(n)` | Memory bandwidth on materialise | All updates precede all reads |
| Fenwick point update | `n` | `O(log2 n)` additions | `O(n)` tree | Random access across `log2 n` cache lines | Invertible aggregate |
| Fenwick prefix query | `n` | `O(log2 n)` additions | `O(1)` extra | Same random-access pattern | Same |
| Segment tree range query | `n` | `O(log2 n)` node merges | `O(4n)` typical | Pointer chasing; node layout | Associative merge |
| Sparse table build | `n` | `O(n log2 n)` merges | `O(n log2 n)` | Memory capacity, then bandwidth | Idempotent aggregate |
| Sparse table query | `n` | `O(1)` merges (two overlapping blocks) | `O(1)` | Constant | Idempotence (sum disqualified) |
| Sort (comparison based) | `n` | `O(n log2 n)` comparisons | `O(n)` or `O(log2 n)` | Comparison count if keys are expensive; otherwise bandwidth | Comparison model |
| Binary search (index) | `n` | `⌊log2 n⌋ + 1` probes | `O(1)` | Dependent cache misses along the probe chain | Random access, sorted |
| Binary search (answer) | `R` = search-range size | `⌊log2 R⌋ + 1` feasibility evaluations | `O(1)` extra | Cost of one feasibility check | Predicate monotone |
| Sliding window | `n` | `≤ 2n` pointer advances | `O(1)` (or `O(σ)` for a distinct-count rule) | Sequential access | Monotone aggregate in growth |
| Prefix + counter | `n` | `O(n)` expected map operations | `O(u)` where `u` = distinct prefix values | Hash quality; per-entry metadata | Universal hashing |
| BFS | `V`, `E` | `O(V + E)` edge scans, `O(V)` dequeues | `O(V)` queue plus visited | Frontier width and random vertex access | Unit edge cost |
| Dijkstra (binary heap) | `V`, `E` | `O((V + E) log2 V)` heap operations | `O(V + E)` | Heap operations, then memory random access | Non-negative weights |
| Bellman–Ford | `V`, `E` | `O(V·E)` relaxations | `O(V)` | Quadratic-ish relaxation count | No negative cycle |
| Top-k heap | `n`, `k` | `O(n log2 k)` comparisons | `O(k)` | Comparison count; `log2 k` is small for small `k` | Consistent comparator |
| Union-find | `n`, `m` unions/finds | `O(m α(n))` amortised | `O(n)` | Memory random access through parent pointers | Union by rank + path compression |
| DP (tabular) | `S` states, `T` transitions per state | `O(S·T)` | `O(S)` (or one layer) | State count; often memory-bound | Acyclic state dependency |
| Amortised append (doubling) | `n` appends | `O(n)` total copies, `O(1)` amortised | `O(n)` capacity | One spike of `O(n)` copy work | Geometric growth |

Two conventions that prevent recurring confusion. First, **auxiliary space excludes the input and the
required output**: the prefix array is auxiliary, the answer is not. Second, **amortised is not
worst-case**: the union-find row is amortised over a sequence, and any single `find` can traverse a long
parent chain; the table says so rather than implying a per-operation guarantee.

The DP row deserves a note because `S` is where all the difficulty lives. A DP over "index and capacity"
with `n = 10^3` and capacity `10^4` has `S = 10^7` states; a DP over "index, capacity, and last chosen item"
has `S = 10^7 · n = 10^10` and is no longer a candidate at any realistic scale. The complexity formula is
trivial; estimating `S` before writing the code is the skill, and it is a skill about problem structure
rather than about algebra.

### 9b. Worked sizing: four patterns, one machine

The scenario is ILLUSTRATIVE throughout. Per-operation costs are assumed constants chosen to be plausible
for a modern server-class CPU (a dependent cache miss at 50–200 ns, a sequential simple operation at
0.3–1 ns, an allocation at tens of nanoseconds); none of them is measured here, and a reader on different
hardware should substitute measured values. What the arithmetic establishes is the *structure* of the
comparison — which term dominates, and by how much — not the absolute seconds.

```text
Workload (illustrative):
  n = 10^7 int64 events in memory (80 MB of payload)
  q = 10^6 range-sum queries over arbitrary [i, j)
  m = 2*10^8 union/find operations on a connectivity structure
  one pass computing a longest-window statistic
Assumed unit costs (ILLUSTRATIVE):
  t_seq = 0.3 ns   per sequential simple operation (add, compare, pointer bump)
  t_rand = 100 ns  per dependent random access that misses cache
  t_hash = 120 ns  per hash-map probe including hashing a 64-bit key
  t_heap = 60 ns   per heap sift step
```

**Option 1 — prefix array.** Build cost: `n` additions at `t_seq`, plus `8(n+1)` bytes of writes.

```text
build ops      = 10^7
build time     = 10^7 * 0.3 ns = 3.0*10^-3 s = 3.0 ms
query ops      = 2 * 10^6 = 2*10^6 sequential loads/adds
query time     = 2*10^6 * 0.3 ns = 6.0*10^-4 s = 0.6 ms  (fully sequential in aggregate)
memory         = 8 * (10^7 + 1) = 80,000,008 B = 76.29 MiB auxiliary
total          = 3.6 ms plus the input's 76.29 MiB
```

**Option 2 — Fenwick tree.** Each query and update touches `⌊log2 n⌋ + 1 = 24` tree cells, and those
accesses are not sequential, so the cost is dominated by `t_rand`.

```text
query ops      = 10^6 * 24 = 2.4*10^7 random accesses
query time     = 2.4*10^7 * 100 ns = 2.4 s
build          = 10^7 * 24 worst case, though linear build in O(n) is available
memory         = 8 * (n + 1) = 76.29 MiB
comparison     = 2.4 s vs 3.6 ms -> factor of about 667 on the query path alone
```

**Option 3 — hash map counter.** One probe per element, so `10^7` probes, plus `≈48 B` of metadata per
distinct key.

```text
probe time     = 10^7 * 120 ns = 1.2 s
memory         = (10^7 + 1) * 48 B = 480,000,048 B = 457.76 MiB
comparison     = 1.2 s and 458 MiB, versus 3.6 ms and 76 MiB for the prefix array
```

The gap between options 1 and 3 is instructive: both are "linear", and the counter is 333 times slower
in this model because every operation pays a cache miss while the prefix scan streams. This is the
concrete form of "constant factors matter", and it is a reason to prefer the array form whenever the
query set is known in advance.

**Option 4 — union-find over 2·10^8 operations.** With `α(n) ≈ 4` at practical sizes, the operation count
is `8·10^8`, dominated by parent-pointer chases.

```text
ops            = 2*10^8 * 4 = 8.0*10^8
time           = 8.0*10^8 * 50 ns = 40.0 s
memory         = 2 arrays of n entries; at n = 10^8, 2 * 10^8 * 4 B = 800,000,000 B = 762.9 MiB
comparison     = 40 s for 2*10^8 connectivity operations is ~5 million ops/s
```

The 50 ns assumed here is lower than the 100 ns used above because a union-find parent array is small
enough per component to stay partly resident; the point is not the constant but the arithmetic habit:
convert operation counts to time with a stated cost, then check the result against the deadline.

**Option 5 — the same statistic by brute force.** The naive longest-window computation is `O(n²)`
interval checks; at `n = 10^7` that is `5·10^13` checks.

```text
checks         = n(n+1)/2 = 10^7 * (10^7 + 1)/2 = 5.00000005*10^13
time at t_seq  = 5*10^13 * 0.3 ns = 1.5*10^4 s = 15,000 s = 4.17 hours
window pass    = 2*10^7 * 0.3 ns = 6.0*10^-3 s = 6 ms
ratio          = 2.5*10^6
```

This is the row that justifies the sheet's insistence on the invariant argument: the sliding window is
not an optimisation of the brute force, it is a different algorithm whose validity rests on
non-negativity. On the illustrative numbers it turns 4.17 hours into 6 milliseconds, and the only thing
that keeps that answer honest is the precondition.

### 9c. Memory sizing for the traversal patterns

```text
Workload: V = 5*10^7 vertices, E = 2*10^8 directed edges (illustrative social-scale graph)
Representation choices, all ILLUSTRATIVE constants:

  distance array, int32          : V * 4 B = 2.0*10^8 B = 190.73 MiB
  visited bitset                 : V / 8  = 6,250,000 B = 5.96 MiB
  queue as a ring of int32       : V * 4 B worst case = 190.73 MiB
  adjacency list, CSR, 2E ints   : 2E * 4 B = 1.6*10^9 B = 1.49 GiB
  adjacency list as per-node vectors: + per-node overhead, typically large

  BFS total (CSR)   = 190.73 + 5.96 + 190.73 + 1525.88 = 1913.3 MiB ≈ 1.87 GiB
  Dijkstra total    = BFS total + heap of (dist, vertex) pairs, up to E entries worst case
                    = + 2E * 8 B = 3.2*10^9 B = 2980.23 MiB
             total = 1913.3 + 2980.23 = 4893.5 MiB ≈ 4.78 GiB
```

The adjacency list dominates every traversal, which is the fact that changes design decisions: a
BFS whose "queue is too big" may actually be limited by the CSR array it must hold to traverse at all.
The second consequence is that the heap in Dijkstra is not free — at worst `E` entries, each 8 bytes in
this estimate — which is why a decrease-key-free implementation with lazy duplicate entries trades memory
for speed, and why the memory ceiling should be computed before choosing the variant.

### 9d. Cost of an algorithmic-complexity attack

```text
Workload: a request handler hashes k = 10^5 attacker-supplied keys into a table with
          b buckets. Attacker chooses keys colliding into one bucket.
Assumed per-comparison cost 1 ns (string comparison short-circuited, ILLUSTRATIVE).

  honest distribution: average probe length = k / b = 10^5 / 10^5 = 1
     work per request = 10^5 comparisons = 10^5 ns = 0.1 ms
  collision attack  : all keys in one bucket, average scan = k/2 = 5*10^4
     work per request = 10^5 * 5*10^4 = 5*10^9 comparisons = 5.0 s
  amplification     = 5.0 s / 0.1 ms = 50,000x

  at 100 requests/s of this shape, honest CPU = 0.01 s/s = 1% of one core
                                attack CPU = 500 s/s = 500 cores saturated
```

The arithmetic is what makes the failure mode legible: the same code, the same input size, and a factor of
50,000 in CPU, obtained purely by choosing the keys. No amount of horizontal scaling fixes a per-request
cost that grows with a request-controlled parameter, which is why the bounded fix is a structural one
(randomised hashing, a per-request key cap, or a comparison-based map with a guaranteed `O(log n)`).

### 9e. Cost of the amortised spike in a latency budget

```text
Assumed: an append-only buffer of int32 doubling from capacity 1; request budget 10 ms;
         one append per request; n grows to 10^8 over the day.

  peak single-append copy = capacity_old elements = 2^26 * 4 B = 268,435,456 B = 256 MiB
  assumed copy bandwidth  = 8 GiB/s (ILLUSTRATIVE, sequential memcpy)
  spike duration          = 256 MiB / 8192 MiB/s = 0.03125 s = 31.25 ms
  budget                  = 10 ms
  verdict                 = the spike exceeds the budget by 3.1x, so any request that
                            triggers it fails its latency objective
  frequency               = once per capacity doubling; at one append per request and
                            n = 10^8, doublings up to 2^27 occur ~27 times per day
```

Twenty-seven failures a day from a design whose *average* append costs well under a microsecond is the
argument for chunked growth or preallocation, and the numbers show why an amortised argument cannot be
used to dismiss it. Note the honesty requirement: the 8 GiB/s figure is an assumption, and a slower
memory system makes the spike worse, not better — so the conclusion is robust in the direction that
matters.

**Takeaway of Section 9:** the asymptotics choose the family and the constants choose the option; the three numbers that decide most real cases are bytes of auxiliary state, dependent-cache-miss count on the hot path, and the largest single operation.

## 10. Key Theorems / Results Reference

The results below are stated in the form the sheet uses them, with their hypotheses and their most common
over-reading. Where a full proof is outside this sheet's scope, the derivation sketch is labelled as such
and a standard source is named rather than invented.

### T-01 Loop-invariant principle (Hoare-style)

| Field | Content |
|---|---|
| Statement | For a loop with guard `G`, body `B`, and predicate `I`: if `I` holds on entry, and `I ∧ G → wp(B, I)`, then on exit `I ∧ ¬G` holds. If additionally `I ∧ ¬G → Q`, the loop establishes postcondition `Q`. |
| Hypotheses | `B` terminates (a separate progress measure is required); `I` is expressible over program state; the loop has a single entry and exit test (no `break` that bypasses `I`, or those exits must be handled separately). |
| Practical implication | Gives the five-line correctness argument of Section 1.2. Every pattern in Section 2.2 is a named choice of `I`. |
| What it does not imply | Nothing about efficiency, and nothing about termination — invariance and termination are separate obligations. |
| Source | Standard program-logic result; see any graduate algorithms or formal-methods text. Statement here is a précis, not a proof. |

### T-02 Telescoping prefix identity

| Field | Content |
|---|---|
| Statement | For a sequence in an abelian group and `P_j = Σ_{t<j} a_t` with `P_0 = 0`, `Σ_{t=i}^{j-1} a_t = P_j - P_i` for `0 ≤ i ≤ j ≤ n`. |
| Hypotheses | Associativity, a two-sided identity, and inverses in the working type; exact representation of every prefix. |
| Practical implication | `O(1)` range query after `O(n)` build for any invertible aggregate. |
| What it does not imply | Nothing about `max`, `min`, set union, or modular arithmetic with a non-invertible modulus relation beyond the standard subtraction-with-adjustment. |
| Source | Algebraic identity (derived in 4.1); no external claim. |

### T-03 Comparison lower bound for membership search

| Field | Content |
|---|---|
| Statement | Any comparison-based algorithm that decides membership of a key in a sorted array of `n` distinct elements needs at least `⌈log2(n+1)⌉` comparisons in the worst case. |
| Hypotheses | Only comparisons of the form `a_i < x` (or `=`) are available; the array is sorted; worst-case over inputs. |
| Practical implication | The half-open binary search of 4.3.2 attains `⌊log2 n⌋ + 1 = ⌈log2(n+1)⌉` probes, so it is optimal in the comparison model. No comparison-based variant can beat it. |
| What it does not imply | That hash lookup is "faster" in a model-free sense: hashing uses a different operation (computing a function of the whole key), and its expected bound is over hash choices. |
| Source | Decision-tree argument with `n + 1` outcomes (the `n + 1` insertion gaps); standard result. |

### T-04 Comparison lower bound for sorting

| Field | Content |
|---|---|
| Statement | Any comparison-based sorting algorithm performs at least `log2(n!) = n log2 n - O(n)` comparisons in the worst case, hence `Ω(n log2 n)`. |
| Hypotheses | Comparison model only; arbitrary input order. |
| Practical implication | Merge sort and heapsort are asymptotically optimal; a comparison sort cannot be made linear by constant-factor engineering. |
| What it does not imply | That `O(n log n)` is the floor for all sorting: radix and counting sorts are linear under a bounded-key assumption, which is a different model. |
| Source | Decision-tree argument with `n!` leaves; Stirling's bound for the estimate. |

### T-05 Expected cost of hashing with universal hashing

| Field | Content |
|---|---|
| Statement | With a hash function drawn uniformly from a universal family and `n` keys in `m` buckets, the expected length of the chain probed for any key is `≤ 1 + n/m`. With `m = Θ(n)` the expected search cost is `O(1)`. |
| Hypotheses | Universal (or a comparable pairwise-independent) family; the adversary does not know the chosen function; exact key equality; a chaining or open-addressing scheme whose probe count is proportional to the chain length. |
| Practical implication | Justifies the `O(1)` expected column in Section 9a. It also explains precisely what an attacker breaks: choosing keys *after* seeing the function violates the hypothesis, so the bound no longer applies and `Θ(n)` per operation is possible. |
| What it does not imply | A worst-case `O(1)` guarantee, nor `O(1)` amortised across resizes without accounting for rehashing cost (which adds `O(n)` per resize but `O(1)` amortised per insert under doubling). |
| Source | Standard universal-hashing theorem (Cormen et al., *Introduction to Algorithms*, hashing chapter). Expected value is over the random choice of hash function. |

### T-06 Correctness of BFS distance for unit-cost graphs

| Field | Content |
|---|---|
| Statement | In a graph with unit edge costs, BFS that marks a vertex when it is enqueued assigns each reachable vertex its exact shortest-path hop count, and processes vertices in non-decreasing distance order. |
| Hypotheses | Unit (or equal) edge cost; marking at enqueue; a FIFO queue; the graph may be cyclic and disconnected (unreachable vertices keep `∞`). |
| Practical implication | `O(V + E)` shortest hops without a priority queue. The level-order property is what makes the queue invariant usable for "all vertices at distance ≤ d" queries. |
| What it does not imply | Correctness for weighted edges: with edge weights in `{0,1}` the FIFO order breaks and a deque (0-1 BFS) or a heap is required; with general non-negative weights Dijkstra is required. |
| Source | Standard proof by induction on distance layers; textbook result. |

### T-07 Dijkstra correctness under non-negative weights

| Field | Content |
|---|---|
| Statement | If all edge weights are non-negative, then when Dijkstra's algorithm extracts a vertex `u` with the minimum tentative label, that label equals the true shortest-path distance, and `u` never needs to be relaxed again. |
| Hypotheses | Non-negative weights; a correct priority queue; reachable target (otherwise the label stays `∞`). |
| Practical implication | One extraction per vertex plus relaxations, `O((V + E) log V)` with a binary heap. Early exit at the target is safe because the finality property holds at extraction. |
| What it does not imply | Correctness with a single negative edge (counterexample in 4.4.1), and nothing about graphs with negative cycles (no finite shortest path exists). |
| Source | Exchange argument on the extracted minimum; textbook result. The hypothesis is load-bearing, not decorative. |

### T-08 Union-find amortised bound

| Field | Content |
|---|---|
| Statement | A sequence of `m` union and find operations on `n` elements, using union by rank (or size) together with path compression, costs `O(m α(n))` in the amortised sense, where `α` is the inverse Ackermann function and `α(n) ≤ 4` for any `n` representable in practice. Using either optimisation alone gives `O(m log n)`. |
| Hypotheses | Both heuristics applied; a disjoint-set forest; operations are unions and finds only (no deletions). |
| Practical implication | Effectively linear connectivity, which is why Kruskal's algorithm is `O(E log E)` dominated by sorting rather than by the union-find work. |
| What it does not imply | Support for deletions, nor cheap enumeration of a component's members, nor a worst-case per-operation bound of `O(1)`. |
| Source | Tarjan's bound; the inverse-Ackermann result is a classic amortised analysis, quoted here, not re-proved. |

### T-09 Bellman's principle of optimality (DP validity)

| Field | Content |
|---|---|
| Statement | If a problem's optimal solution can be decomposed so that each sub-solution is itself optimal for its subproblem, and the subproblem depends only on a state that captures all relevant history, then the recurrence over states yields a globally optimal value. |
| Hypotheses | Optimal substructure for the specific objective; a state that is a sufficient statistic for the future; an acyclic dependency order over states (or a monotone fixpoint iteration for cyclic dependencies). |
| Practical implication | Justifies the DP family and dictates the two design obligations: state sufficiency (the two-trace test) and fill order (topological order of the state graph). |
| What it does not imply | That a state which makes the recurrence *computable* also makes it *correct*; the classic failure F-06 is exactly this gap. Nor does it bound the state count, which is where the cost lives. |
| Source | Bellman's dynamic-programming principle; the precise hypotheses are stated in operations-research texts. |

### T-10 Exchange argument for earliest-finish interval scheduling

| Field | Content |
|---|---|
| Statement | For maximising the number of mutually non-overlapping intervals on a single resource, sorting by earliest finish time and taking each interval that starts at or after the last chosen finish yields an optimal solution. |
| Hypotheses | One resource; each interval occupies it exclusively; the objective is the *count* of selected intervals; intervals are half-open (touching endpoints are compatible). |
| Practical implication | `O(n log n)` for the sort plus `O(n)` scan, with no DP table. |
| What it does not imply | Optimality for weighted intervals (needs DP), for multiple resources (needs matching/flow), for buffered schedules, or for the shortest-duration rule (counterexample 4.4.2). |
| Source | Classic greedy exchange proof; the exchange step replaces the earliest-finishing interval of an optimal solution without reducing the count. |

### T-11 Amortised analysis: aggregate and potential methods

| Field | Content |
|---|---|
| Statement | If a sequence of `m` operations performs at most `T(m)` elementary operations in total from a fixed initial state, the amortised cost per operation is `T(m)/m` (aggregate method). Equivalently, with a potential function `Φ ≥ 0` and `Φ_0 = 0`, the amortised cost of operation `i` is `c_i + Φ_i - Φ_{i-1}`, and the sum over a sequence bounds the true total (potential method). |
| Hypotheses | A fixed starting state; a fixed growth policy; the sequence length is the unit of accounting. |
| Practical implication | Doubling growth gives `≤ 3` amortised per append including the copy (derived in 4.4.3); it is the correct instrument for throughput budgeting. |
| What it does not imply | Any bound on an individual operation's cost, hence no latency guarantee (failure F-07). |
| Source | Standard amortised-analysis framework; both derivations for the doubling array are shown in 4.4.3. |

### T-12 Master theorem (for divide-and-conquer recurrences)

| Field | Content |
|---|---|
| Statement | For `T(n) = a·T(n/b) + f(n)` with `a ≥ 1`, `b > 1`, and `f` asymptotically positive, compare `f(n)` with `n^{log_b a}`: if `f(n) = O(n^{log_b a - ε})` then `T(n) = Θ(n^{log_b a})`; if `f(n) = Θ(n^{log_b a} log^k n)` then `T(n) = Θ(n^{log_b a} log^{k+1} n)`; if `f(n) = Ω(n^{log_b a + ε})` and the regularity condition `a·f(n/b) ≤ c·f(n)` holds for some `c < 1`, then `T(n) = Θ(f(n))`. |
| Hypotheses | The stated asymptotic comparisons, including the ε-separation between cases and the regularity condition in the third case; the recurrence must have the stated form (equal subproblem sizes). |
| Practical implication | Instant answers for merge sort (`a=2, b=2, f=Θ(n)` gives `Θ(n log n)`) and for binary search (`a=1, b=2, f=Θ(1)` gives `Θ(log n)`), matching T-03. |
| What it does not imply | Anything about unequal splits, non-polynomial `f`, or constant-factor behaviour; the theorem is asymptotic. |
| Source | Standard statement; see any algorithms text for the full proof. |

### T-13 Idempotence requirement for sparse-table range queries

| Field | Content |
|---|---|
| Statement | The two-block sparse-table query is correct for an aggregate `⊕` when `⊕` is associative and idempotent (`x ⊕ x = x`), because the two overlapping blocks of length `2^k` together cover the range and the overlap contributes nothing under idempotence. |
| Hypotheses | Static array; associative and idempotent merge; overlapping blocks allowed. |
| Practical implication | `O(1)` range min/max/gcd; `O(n log n)` build; memory `O(n log n)`. |
| What it does not imply | Correctness for sums (overlap double-counts), which is why the Section 7.3 table disqualifies the sum-sparse-table twice. |
| Source | Standard sparse-table construction; the idempotence requirement is the load-bearing hypothesis. |

### T-14 The two-trace test as a state-sufficiency criterion

| Field | Content |
|---|---|
| Statement | A DP state `s` is sufficient if and only if, for every pair of partial solutions mapping to `s`, the optimal completions from that point are interchangeable; equivalently, no pair of distinct partial solutions with the same state has different feasible completions. |
| Hypotheses | The objective depends only on the completed solution; "completion" means the remaining decisions. |
| Practical implication | Operational test: construct two prefixes with the same state tuple but different futures; if you can, add a dimension. This is the cheapest available check for F-06. |
| What it does not imply | Minimality — a sufficient state can be far larger than necessary, and state size is the cost driver. |
| Source | Sufficient-statistic argument from dynamic programming theory; the operational form is derived here for teaching. |

### 10.1 How to read these results without over-claiming

Three over-readings recur and each has a cheap correction. First, treating expected bounds as guarantees:
T-05 is an average over hash functions and T-11 is an average over a sequence, and neither bounds a single
operation on an adversarially chosen input. The correction is to ask "averaged over what?" every time a
bound is quoted. Second, treating asymptotic results as performance predictions: T-04 says comparison
sorting cannot beat `n log n`, which is silent about whether an `n log n` sort on 10^7 records takes 2 or
20 seconds. The correction is Section 9's arithmetic. Third, treating a theorem's hypothesis as a
formality: T-07 fails on a single negative edge, T-13 fails on a sum, T-10 fails on weights, and in each
case the hypothesis is the entire content of the result rather than a technicality to be waived when
inconvenient.

**Takeaway of Section 10:** each result is a package of statement plus hypotheses plus a specific thing it does not say, and the practical skill is recalling the third part at the moment a fast pattern looks attractive.

## 11. Interview / Exam Questions

Fourteen questions across three levels. Each answer is checked against the body of this sheet; the "trap"
line names the plausible wrong answer and the mechanism that makes it wrong; the scoring line is the
interviewer's rubric. Questions marked *(calc)* require arithmetic or a trace, and questions marked
*(diag)* require discriminating between causes.

### Level 1 — Definition and recall

**Q1 — What is a loop invariant, and what two obligations must it satisfy?**
*Answer.* A predicate over program state that holds on loop entry and is re-established by every
execution of the body under the guard; on exit, invariant plus negated guard must imply the
postcondition. The two obligations are **initialisation** (it holds before the first test) and
**maintenance** (each body execution preserves it). A third obligation, **termination**, is separate and
requires a progress measure — an invariant alone proves only conditional correctness.
*Alternative framing.* Some candidates describe it as "what stays true while the loop runs", which is
correct but insufficient; the scored version names the exit implication.
*Follow-up.* Give the invariant for a partition step and say what it implies at exit.
*Trap.* "An assertion I add to the code" — an assertion is an implementation of the check, not the
predicate; and an invariant that is never falsifiable by the bug you fear is decoration.
*Scoring.* 0 = "a condition"; 1 = predicate plus one obligation; 2 = both obligations plus the exit
implication; 3 = adds the separate termination obligation and an example.
*Source.* Section 3.2, T-01.

**Q2 — Distinguish a monotone predicate from a non-decreasing function, and say why binary-search-on-answer needs the former.**
*Answer.* A monotone predicate `P` over an ordered domain has a threshold `x*` with `P(x)` false below and
true above (or the reverse); a non-decreasing real function satisfies `f(x) ≤ f(y)` for `x < y`. A
predicate can be monotone while the underlying quantity oscillates, provided every oscillation stays on
one side of the threshold. Binary-search-on-answer needs the threshold form, because the halving step
discards one half of the *domain* on the basis of one probe, which is only sound if all values in the
discarded half share the same truth value as the probe.
*Example / nonexample.* `feasible(C) = "min 5-window sum ≤ 5C"` is monotone (Section 5.4). `P(x) = "x is
prime"` is not monotone and no halving argument applies.
*Follow-up.* What breaks if the predicate is monotone except at one point? (Answer: the halving can
discard the unique feasible point; the fix is to prove monotonicity over the reachable domain, not to
sample it.)
*Trap.* "The function is increasing, so binary search works" — increasing functions do not by themselves
give a binary-searchable threshold; the decision structure is what matters.
*Scoring.* 0 = conflates the two; 1 = distinguishes but cannot say why halving needs the threshold; 2 =
correct distinction with the discard argument; 3 = adds a counterexample and the sampling critique.
*Source.* Section 3.2, 4.3.2.

**Q3 — What does "amortised O(1) append" promise, and what does it not promise?**
*Answer.* It promises that any sequence of `m` appends from a fixed empty state performs at most `c·m`
elementary operations in total, so the average is bounded. It does not promise that any single append is
cheap; a doubling container copies `capacity_old` elements on one particular append. It also does not
promise anything across an unfixed starting state, or when the growth policy is additive rather than
geometric.
*Example.* At `n = 10^8` int32 elements the spike copies up to 2^26 elements = 256 MiB (Section 9e),
which is 31.25 ms at an assumed 8 GiB/s — a violation of a 10 ms per-request budget even though the
average is under a microsecond.
*Follow-up.* Which structure would you choose for a hard per-append deadline?
*Trap.* "Each append is O(1) because the total is O(n)" — the total being `O(n)` is precisely what allows
one operation to be `O(n)`.
*Scoring.* 0 = "each append is constant"; 1 = average over the sequence; 2 = plus the spike; 3 = plus the
latency-budget consequence and a structural alternative.
*Source.* 4.4.3, T-11, F-07.

### Level 2 — Derivation, application, trace

**Q4 — Derive the count of subarrays summing to `k`, and explain why the map update must follow the query. *(calc)***
*Answer.* With `P_j = Σ_{t<j} a_t`, a subarray `[i, j)` sums to `k` iff `P_j - P_i = k` iff `P_i = P_j - k`.
For each right endpoint `j` there is exactly one left endpoint for each earlier prefix equal to `P_j - k`,
so the count is `Σ_j cnt_j(P_j - k)` where `cnt_j` holds the multiplicity of `P_0 … P_{j-1}`. The update
must follow the query because inserting `P_j` first would make `i = j` a candidate, counting the empty
subarray when `k = 0`.
*Trace.* On `[4,9,2,7,5,3,8,1,6,2,9,4]` with `k = 15`, the hits occur at `j = 3, 6, 9, 12` giving 4
windows, verified by enumerating all 78 intervals (Section 5.2).
*Follow-up.* What changes for the parallel version? (Answer: independent per-shard counts drop
cross-boundary pairs; compute the full prefix array and count cross-half pairs explicitly.)
*Trap.* Writing the recurrence as "count pairs `i ≤ j`", which includes the empty range and inflates the
answer by `n+1` when `k = 0`.
*Scoring.* 0 = quadratic enumeration described as optimal; 1 = correct recurrence without the ordering
reason; 2 = ordering justified; 3 = plus the parallel-boundary caveat and a verified trace.
*Source.* 4.2, 5.2, F-04.

**Q5 — Compute the worst-case probe count for half-open binary search on `n = 10^6`, and show the interval shrinks. *(calc)***
*Answer.* With `L = hi - lo` and `mid = lo + ⌊L/2⌋`: the "lo = mid + 1" branch leaves
`L - ⌊L/2⌋ - 1` and the "hi = mid" branch leaves `⌊L/2⌋`, both at most `⌊L/2⌋`. So `L_t ≤ ⌊n/2^t⌋`, the
loop ends when `L < 1`, giving at most `⌊log2 n⌋` iterations plus one equality check. For `n = 10^6`,
`⌊log2 10^6⌋ = 19`, so 20 probes.
*Independent check.* `2^19 = 524,288 ≤ 10^6 < 2^20 = 1,048,576`, and `⌈log2(10^6+1)⌉ = 20`; the two
formulas agree.
*Follow-up.* Where can the midpoint overflow, and what is the safe form? (Answer: `lo + hi` overflows for
`lo = 1.5·10^9`, `hi = 2·10^9` in signed 32-bit; use `lo + (hi - lo)/2`.)
*Trap.* Saying "`log n` iterations, so `log n` probes" — the exact count differs by one and the loop shape
determines which; more importantly, quoting a count without pinning the variant is untestable.
*Scoring.* 0 = "about 20"; 1 = correct count; 2 = shrin' proof; 3 = plus the overflow note and the exact
identity check.
*Source.* 4.3.2, T-03.

**Q6 — Trace the shrinking window on `[4,9,2,7,5,3,8,1,6,2,9,4]` with budget 20 and prove the answer optimal. *(calc)***
*Answer.* Trace in Section 5.3: best length 5 at `[3,8,1,6,2]` summing to 20. Optimality: the seven
length-6 windows sum to 30, 34, 26, 30, 25, 29, 30, all above 20; any longer window contains a length-6
sub-window whose sum is at most its own by non-negativity, so none qualifies.
*Cost check.* Pointer advances total 12 additions and 9 drops = 21 ≤ 2n = 24, consistent with the `O(n)`
bound.
*Follow-up.* Which single element would you change to break the algorithm's precondition, and what would you
observe? (Answer: any negative element; the answer can change because the left pointer has already passed a
still-valid window.)
*Trap.* Claiming optimality from "no length-6 window worked" without the containment argument — the claim
needs the step that a longer window contains a shorter one and non-negativity keeps its sum at least as
large.
*Scoring.* 0 = trace only; 1 = correct answer with a plausible but incomplete argument; 2 = containment
argument; 3 = plus the pointer-count check.
*Source.* 4.3.1, 5.3.

**Q7 — Why does Dijkstra need non-negative weights? Give a minimal counterexample. *(calc)***
*Answer.* The extraction step claims finality: the minimum tentative label is the true distance because any
other path must leave the unsettled set through a vertex whose label is at least as large, and a non-negative
continuation cannot reduce it. With a negative edge the continuation can reduce it. Minimal counterexample:
`s→a = 1`, `s→b = 2`, `b→a = -2`, source `s`. Dijkstra settles `a` at 1, then relaxes `b→a` to find 0,
contradicting finality; true distance to `a` is 0.
*Follow-up.* What if exactly one weight is negative but there is no negative cycle? (Answer: Bellman–Ford,
or Johnson reweighting then Dijkstra; correctness is not recoverable by tuning Dijkstra.)
*Trap.* "Dijkstra still finds a path, so it is fine" — it finds a path, not a shortest one; the failure is
in the optimality claim, and it is silent.
*Scoring.* 0 = "Dijkstra is for shortest paths"; 1 = states non-negativity without reason; 2 = finality
argument; 3 = plus the counterexample with numbers.
*Source.* 4.4.1, T-07.

**Q8 — A DP returns an answer better than any feasible solution. What is the most likely cause and how do you confirm it? *(diag)***
*Answer.* The state omits a dimension the objective depends on, so the DP is optimising over a relaxed
problem. Confirmation: reconstruct the solution from the table and check it against the problem's
constraints. If the reconstruction violates a constraint that the original problem imposes, the state is
insufficient; apply the two-trace test — find two prefixes with identical state and different feasible
futures.
*Competing cause.* A fill-order bug that reads the current layer instead of the previous one, which
usually inflates counts rather than improving an optimum; distinguished by checking whether the error is a
count excess rather than an infeasible optimum.
*Follow-up.* What test would have caught it before submission? (Answer: exhaustive enumeration on instances
of size ≤ 12, plus reconstruction assertions on every sample.)
*Trap.* "Add more test cases" — more instances of the same size distribution rarely expose a state-dimension
error, because the error is in the problem model rather than the arithmetic.
*Scoring.* 0 = "the recurrence is wrong"; 1 = names one cause; 2 = reconstruction check; 3 = plus the
two-trace test and the competing-cause discrimination.
*Source.* 3.2, F-06, T-09, T-14.

### Level 3 — Judgement, design, diagnosis

**Q9 — A service is fast in staging and slow in production under the same request count. Walk through your hypotheses and the order you would test them. *(diag)***
*Answer.* Hypothesis order should run from cheapest to most informative. (1) **Input distribution**: staging
traffic has benign keys; production may include near-miss strings or long shared prefixes that stress a
hash. Test: instrument probe counts per lookup and plot the distribution — collision cost shows a long tail
at fixed loop structure. (2) **Access pattern**: production data does not fit cache while staging does. Test:
compare per-operation time against working-set size, not against request count. (3) **Concurrency**: staging
may be single-threaded, so lock contention or false sharing appears only in production. Test: time by thread
count at fixed total work. (4) **A latent quadratic structure**: a membership test inside a loop, visible as
a uniform `n`-per-iteration count.
*Discrimination.* A tail-shaped per-operation count implicates hash or cache; a uniform linear count
implicates the algorithm; scaling with thread count implicates contention.
*Follow-up.* If probe counts are uniform and low, what have you ruled out? (Answer: collision-driven and
rescan-driven cost; the remaining candidates are memory latency, contention, and downstream dependencies —
which means the next step is a profile, not more algorithmic reasoning.)
*Trap.* "Add more servers" — if per-request cost grows with a request-controlled parameter, scaling multiplies
the problem.
*Scoring.* 0 = one guess; 1 = several hypotheses without discrimination; 2 = ordered tests with signals;
3 = plus the meaning of a negative result.
*Source.* 8.2, 9d.

**Q10 — You must answer `10^6` range-sum queries over `10^7` static int64 values within 512 MiB and 50 ms p99. Choose and defend. *(calc)*
*Answer.* Static data plus an invertible aggregate makes the prefix array the choice: build `8(10^7+1) B =
76.29 MiB`, one subtraction per query. A Fenwick tree would cost `24` dependent random accesses per query
(`2.4·10^7` accesses) and is unnecessary because there are no updates. A sparse table is disqualified twice:
`8·10^7·24 B ≈ 1.79 GiB` exceeds the ceiling, and a sum is not idempotent.
*Cost arithmetic.* At an assumed 0.3 ns per sequential operation the build is 3 ms and the whole query set
about 0.6 ms of arithmetic; at 100 ns per random access the Fenwick alternative is 2.4 s of memory stalls,
which does not meet a 50 ms p99 for a batch but might for a single query — the distinction is batch versus
per-query latency, and the question is about the batch.
*Follow-up.* What changes if updates arrive every second? (Answer: point updates plus range queries make
Fenwick or a segment tree appropriate; the prefix array's build would dominate.)
*Trap.* Choosing a sparse table because "it gives O(1) queries" — both disqualifiers (memory, idempotence)
are independent, and either alone rejects it.
*Scoring.* 0 = picks the fastest-sounding structure; 1 = correct choice without sizing; 2 = with memory and
cost arithmetic; 3 = plus the updates follow-up and the idempotence reason.
*Source.* 7.1, 7.3, 9b, T-02, T-13.

**Q11 — Design the correctness evidence for a new shortest-path routine in a routing service. What do you compare against, and what does agreement not prove? *(design)***
*Answer.* Evidence layers: (1) a brute-force oracle — breadth-first hops on the unweighted version, and
Bellman–Ford or exhaustive path enumeration on small weighted graphs with non-negative weights; (2) a
property suite — triangle inequality `d(v) ≤ d(u) + w(u,v)` for every edge, `d(s) = 0`, unreachable vertices
`∞`, symmetry of the underlying undirected distance; (3) adversarial instances — a graph with one
zero-weight edge, a disconnected component, a single vertex, a self-loop, parallel edges with different
weights; (4) a determinism check for tie-broken path output.
What agreement does **not** prove: that the routine is correct on graphs outside the generator's
distribution, that it respects a turn-restriction model, or that a preprocessed structure (contraction
hierarchies, shortcuts) preserves optimality after a live closure — the INDEX briefs PLAN-2026-IR019 and
PLAN-2026-IR023 record exactly this gap, where a legal route after filtering does not establish completeness.
*Follow-up.* How would you test a preprocessed structure against a live-base-graph fallback? (Answer:
differential testing on a snapshot where both are legal, plus explicit unreachable-pair cases.)
*Trap.* "We compared against Dijkstra and it matched" — if the routine *is* a Dijkstra variant, agreement
is not independent evidence; the oracle must not share the implementation's assumptions.
*Scoring.* 0 = "write unit tests"; 1 = one oracle; 2 = oracle plus properties; 3 = plus the limits of
agreement and the preprocessed-structure caveat.
*Source.* 5.6, Section 11 pattern from T-06/T-07, INDEX briefs IR019/IR023 as stated gaps.

**Q12 — A candidate proposes a hash-map solution for counting subarrays with sum `k`, citing expected `O(n)`. An interviewer objects that the worst case is `O(n²)`. Who is right, and what would you ship? *(judgement)***
*Answer.* Both are right about different statements: `O(n)` expected over hash choices under universal
hashing, `O(n²)` worst case for adversarial keys under a fixed hash. Shipping decision: keep the map (the
alternative `O(n log n)` ordered map is usually worse in the average case), randomise the seed per process
so keys cannot be precomputed against it, and cap the number of distinct keys or the request size so a
single request cannot force a large colliding set. Document both bounds in the design note, because the
worst case is the one the capacity plan must survive.
*Follow-up.* When would you choose the ordered map? (Answer: when keys are adversarial by construction —
user-chosen identifiers in a security-sensitive path — and the log factor is affordable; or when ordered
iteration is required anyway.)
*Trap.* "Expected O(n) means O(n)" — dropping the expectation silently converts an average-case engineering
claim into a false guarantee, which is the same error as calling an amortised bound a latency bound.
*Scoring.* 0 = picks a side; 1 = states both bounds; 2 = explains the mechanism; 3 = plus the mitigation
and the documentation requirement.
*Source.* 4.4.4, T-05, 9d.

**Q13 — Your BFS worker dies with an out-of-memory error at 5·10^7 vertices. Diagnose and bound the fix. *(diag calc)***
*Answer.* Competing causes and their signals: enqueueing duplicates because marking happens at dequeue
(enqueues per vertex > 1, queue length exceeds `V`); a genuinely wide frontier with heavy per-vertex records
(peak queue length ≈ `V`, record size dominates); or the CSR adjacency itself (1.49 GiB for 2·10^8 directed
edges at 4 bytes per endpoint) leaving no headroom for the queue's 190.73 MiB.
*Bounded fix.* Mark at enqueue and assert uniqueness; replace object-per-vertex maps with arrays indexed by a
compact id, giving a distance array of 190.73 MiB and a visited bitset of 5.96 MiB; if the frontier is still
the constraint, switch to bidirectional search when a target is known, or to an explicit-stack DFS when only
reachability is required.
*Verification.* Assert peak queue length ≤ `V`; compare distances against a reference BFS on a 10^5-vertex
subgraph; watch resident bytes against the ceiling.
*Trap.* "Increase the heap" — it buys one factor and leaves the duplicate-enqueue bug multiplying the queue
by the average in-degree.
*Scoring.* 0 = "add memory"; 1 = one cause; 2 = discriminating signals; 3 = plus the arithmetic and the
bounded fix with verification.
*Source.* 8.9, 9c.

**Q14 — How do you know your fast implementation is right, and what is the cheapest evidence that would change your mind? *(judgement)***
*Answer.* The cheapest evidence is a differential test against a brute-force oracle on small instances,
because the oracle is obviously correct by construction and the test space is enumerated rather than
sampled. It should cover the structural edge cases the invariant names: empty input, one element, all
equal, all zeros, negative values where allowed, absent keys, and ties. Beyond that, property assertions
embedded in the fast path (the window's incremental sum equals a recomputed sum; the DP's reconstructed
solution satisfies the constraints; the returned path's length equals the reported distance) catch the
class of bug where the answer is self-consistent but wrong.
What would change my mind: a counterexample from the oracle; a property violation; or an observation that
the input class violates a stated precondition. What would *not* change my mind: more passing tests of the
same shape, since they add no information about the failure boundary.
*Follow-up.* Where does the oracle stop being feasible? (Answer: when the brute force is exponential or
quadratic on an input size the production path must handle, at which point the oracle is a small-instance
instrument and the property suite becomes the primary evidence.)
*Trap.* "It passed the sample" — the sample is one point in an input class, and the failure modes in
Section 8 are defined on *classes*, not on points.
*Scoring.* 0 = "run tests"; 1 = mentions an oracle; 2 = oracle plus edge cases; 3 = plus in-path properties
and the explicit statement of what counts as disconfirming evidence.
*Source.* 5.6, 8, T-14.

### 11.1 Answer-key notes and common scoring disagreements

Two questions in this set are commonly scored inconsistently, and the reason is worth recording. For Q9,
candidates who propose profiling first are not wrong, but "profile first" without a hypothesis list answers
a different question: profiling without a hypothesis finds hot functions, not the input property that makes
them hot, and the failure modes here are distributional. A good answer names the distributional hypotheses
and says which profile signal would discriminate them.

For Q12, the disagreement is usually about whether "expected" bounds are acceptable in a design review. The
sheet's position is that they are acceptable *if stated*: an expected bound with its expectation named and
its worst case bounded by an input cap is a complete engineering statement, while an expected bound
presented as a guarantee is not. Candidates should be rewarded for making that distinction explicitly
rather than for choosing a side.

**Takeaway of Section 11:** the interview skill is not recalling the pattern but naming the invariant, computing the exact bound, and stating what evidence would falsify the choice.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Selecting and defending DSA patterns by validity condition, invariant and declared cost model; diagnosing invariant violations |
| Core formula | `sum(i,j) = P_j - P_i` with `P_0 = 0` (valid only for invertible, exactly-represented aggregates) and `count(k) = Σ_j cnt_j(P_j - k)` with query-before-insert |
| Key assumption | Monotonicity — the property every sub-quadratic pattern silently borrows: non-negative elements for shrinking windows, unit edges for BFS, non-negative weights for Dijkstra, a threshold predicate for answer search |
| Complexity | `O(n)` prefix build plus `O(1)` per range query; `⌊log2 n⌋ + 1` binary-search probes; `≤ 2n` window pointer advances; `O(V + E)` BFS; `O((V+E) log V)` Dijkstra; `O(n log k)` top-k; `O(m α(n))` union-find amortised |
| Use when | The input class provably satisfies the pattern's precondition, and an oracle test at small size is available to check the answer |
| Avoid when | The precondition is "usually" true rather than guaranteed; a per-operation latency guarantee is required of an amortised structure; or the aggregate is not invertible/idempotent as the structure requires |
| Evidence status | This draft: `DERIVED` bounds and `ILLUSTRATIVE` workloads with `VERIFIED` arithmetic; no measured performance, no source lesson; review OPEN |
| Go deeper | The planned siblings in this collection (complexity reference; interview problem selection — see `Cheat_Sheets/INDEX.md` rows 2–3, status `planned`) and the routing, rate-limiting and search briefs PLAN-2026-IR018 through IR035, which carry these invariants into specific services |
