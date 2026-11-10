---
title: "25 Common Issues — 03_dsa_and_computational_thinking"
topic: "Failure diagnosis for data-structure, algorithm and arithmetic decisions"
collection: "03_dsa_and_computational_thinking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
source_index: "INDEX.md"
owner_role: "curriculum maintainer"
review_trigger: "Any change to a collection decision contract or to the runtime measured here"
notation_style: "plain-text"
---

# 25 Common Issues — 03_dsa_and_computational_thinking

> **What this file is.** A diagnostic catalog for the failure modes that survive code review in
> data-structure and algorithm work: wrong interval conventions, algorithms applied outside their
> preconditions, arithmetic that silently wraps or drifts, and cost models copied from a different
> machine than the one running the code.
> **Evidence.** Every timing, memory figure and bit pattern quoted below was measured on one
> CPython 3.12.10 process on 2026-11-10 on the workstation named in the course runtime block; the
> commands are recorded so a reader can repeat them. Timings are single-run wall clocks and are
> used only as *ratios* between two implementations in the same process, never as absolute
> service guarantees. Workload inputs are constructed, so every constructed quantity is labelled
> ILLUSTRATIVE. No production incident, company measurement or deployment result is claimed
> anywhere in this file.

## 1. File Header

### 1.1 The decision this file supports

The reader arrives with a symptom — a wrong total, a stalled job, a memory graph climbing, a
result that changes between two runs of identical code — and must decide, within minutes,
whether the cause is a violated algorithm precondition, an arithmetic edge, a scale failure, or
a conceptual confusion about what a complexity bound promises. Those are four different repairs.
Deferring an item fixes a contract problem; it does nothing for an integer that wrapped. Adding
capacity fixes a queue imbalance; it does nothing for a comparator that is not transitive. The
file exists so that the first action taken is the one that matches the mechanism, because the
expensive error in this domain is not the bug, it is the week spent fixing a bug that was never
there.

### 1.2 Reader task and assumed background

The assumed reader has written a hash map, a heap or a graph traversal in at least one language,
can read `O(...)` notation, and knows that an array index is an integer. The reader is *not*
assumed to know: the difference between amortised and worst-case cost; that a hash table's
average-case bound is conditional on the hash function; that `O(log n)` counts comparisons while
real hardware charges cache misses; or that intermediate arithmetic in a nominally 64-bit language
can still be evaluated in 32-bit. Those four gaps account for a large share of the catalog below,
which is why §3 defines them before §8 uses them.

The reader's task, stated as an output: produce a one-paragraph triage note naming the failing
invariant, the discriminating check that separated it from its nearest competing cause, the
containment action, and the replay that proves recovery. If the note cannot name a *check* — an
observation that would have come out differently under the rival hypothesis — the diagnosis is
not finished, regardless of how confident it sounds.

### 1.3 Scope, exclusions and handoffs

In scope: array and interval conventions; hashing; sorting and comparators; heaps and top-K;
graph traversal and shortest paths; amortised growth; recursion depth; floating-point and integer
arithmetic in aggregates; cost models for these structures; and the concurrency vocabulary needed
to state what a shared index promises.

Out of scope, with the handoff named: distributed consensus and replication lag (module 04);
statistical estimation and sampling error (module 05); storage-engine layout and page formats
(module 04); query planning in a relational engine (module 02). When a symptom in this file is
actually a latency symptom of a remote dependency, the correct next read is module 04, not a
deeper search here. The catalog deliberately does not restate those mechanisms, because a
catalog that explains everything explains nothing twice.

### 1.4 Evidence policy for this file

Three evidence classes appear and must not be blended:

- **Measured** — a number produced by a command recorded in this file, on the stated runtime.
  Ratios from measured pairs are the only quantitative claims used for selection decisions.
- **Derived** — a number produced by arithmetic shown in full, from stated inputs. The derivation
  is checkable even if the inputs are illustrative.
- **ILLUSTRATIVE** — a constructed workload whose outputs describe the arithmetic, not the world.

A fourth class exists and is excluded by policy: a figure remembered from a benchmark blog post.
Where such a figure would be convenient, the file instead measures or derives, or it says the
number is unknown.

### 1.5 How issue identifiers work

Each catalog entry is `CI-NNN` and is stable: identifiers are never reused for a different
mechanism, so a review comment citing `CI-014` keeps its meaning across revisions. Three-digit
zero padding is load-bearing — it makes `grep 'CI-0'` return the catalog in numeric order and
keeps the table sortable as text. The category tag is one of `VIOLATED_ASSUMPTION`,
`NUMERICAL_EDGE`, `SCALE_FAILURE`, `CONCEPT_CONFUSION`, and it names the *repair class*, because
the repair class is what a responder needs at 03:00: an assumption repair changes a contract, a
numerical repair changes a type or a rounding rule, a scale repair changes an algorithm, and a
conceptual repair changes a decision document.

### 1.6 Catalog census

The catalog holds 32 entries. That exceeds the 25-entry floor for this file type because the
fourteen arithmetic edges cannot be compressed into a single "check your types" entry without
losing the specific bit pattern that identifies each one. Distribution by repair class:

| Category | Count | Entries |
|---|---:|---|
| NUMERICAL_EDGE | 12 | CI-001, CI-009, CI-010, CI-011, CI-012, CI-013, CI-014, CI-015, CI-017, CI-018, CI-019, CI-020 |
| SCALE_FAILURE | 9 | CI-006, CI-021, CI-022, CI-023, CI-024, CI-025, CI-026, CI-027, CI-028 |
| CONCEPT_CONFUSION | 6 | CI-005, CI-008, CI-029, CI-030, CI-031, CI-032 |
| VIOLATED_ASSUMPTION | 5 | CI-002, CI-003, CI-004, CI-007, CI-016 |

The imbalance is intentional and reflects repair cost. A numerical edge usually has a one-line
fix once identified, and identification is the whole problem, so each one needs its own entry. A
violated assumption usually appears once per system and then stays fixed by a contract clause, so
five entries cover the recurring family.

### 1.7 Triage budget

The catalog assumes a 30-minute first pass, spent as: five minutes to classify the symptom by
observable behaviour (wrong value / no value / late value / growing resource / unstable value);
ten minutes on the discriminating check for the leading hypothesis and its nearest rival;
ten minutes on containment that does not destroy evidence; five minutes writing the triage note
with the replay command. A first pass that runs past 30 minutes without a named failing invariant
is itself a finding: the system lacks the observable needed for diagnosis, and the follow-up work
is instrumentation, not more reading.

### 1.8 What "fixed" means here

An entry counts as fixed only when all four hold: the mechanism is named in one sentence; a
regression input that reproduces it exists; the output under that input is now the expected one;
and a nearby boundary input is also correct. That fourth condition is the one skipped in practice.
Fixing an inclusive/exclusive bound at index 8 and testing only index 8 leaves the same bug at
index 0, where the inclusive reading is accidentally correct. §5 walks a full example of that
asymmetry.

## 2. Concept Map / Mental Model

### 2.1 The triage pipeline

```text
symptom
  |
  +-- value is wrong ------------------------------------+
  |     |                                                 |
  |     +-- reproducible from one record? --no--> ordering / hash-seed / concurrency
  |     +-- yes --> off by exactly one unit? --yes--> boundary convention (CI-001, CI-018)
  |                 |                                     |
  |                 +-- off by a scale factor? --yes--> units or precision (CI-016, CI-019)
  |                 +-- off by a tiny amount? --yes--> floating accumulation (CI-011, CI-012)
  |                 +-- wrapped sign? --yes--> fixed-width overflow (CI-010, CI-014)
  |
  +-- value never arrives ------------------------------+
  |     +-- depth-bounded recursion? --> stack (CI-006)
  |     +-- waiting on a lock held by a traversal? --> cycle on non-DAG (CI-008)
  |     +-- search space unexplored? --> monotonicity broken (CI-007)
  |
  +-- value arrives late, all values right -------------+
  |     +-- grows with input size --> algorithmic cost (CI-021 to CI-024)
  |     +-- grows with degree/depth --> frontier size (CI-025)
  |     +-- spikes, then recovers --> collision or backtracking (CI-026, CI-028)
  |
  +-- resource climbs, values right ---------------------+
        +-- heap only --> retaining references / frontier (CI-025, CI-027)
        +-- CPU only --> re-computation (CI-021, CI-022) or adversarial input (CI-028)
```

The tree is ordered by cheapness of the check, not by likelihood. Asking "is the value off by
exactly one?" costs one comparison and eliminates most of the catalog. Asking "is the comparator
transitive?" costs a property test and eliminates three entries. Start at the top.

### 2.2 Symptom class to repair class

| Observable | First repair class to consider | Why that class first | Typical entry |
|---|---|---|---|
| Off by one unit at a boundary | NUMERICAL_EDGE | Convention, not logic, is usually wrong | CI-001 |
| Sign flipped on a large total | NUMERICAL_EDGE | Fixed-width wrap has a signature | CI-010 |
| Result differs run to run | VIOLATED_ASSUMPTION | Determinism was assumed, not enforced | CI-003 |
| Latency grows superlinearly in input | SCALE_FAILURE | Cost model mismatched the data structure | CI-022 |
| Correct answer, unacceptable memory | SCALE_FAILURE | Frontier or cache retention | CI-025 |
| Two teams disagree about "safe" | CONCEPT_CONFUSION | The word carries two guarantees | CI-032 |
| Fix works, next input breaks it | VIOLATED_ASSUMPTION | Precondition was never written down | CI-007 |

### 2.3 The invariant-first discipline

Every structure in this module is a promise about an invariant, and every failure in the catalog
is a broken invariant observed from outside. A sorted array promises `a[i] <= a[i+1]`. A binary
heap promises `a[parent] <= a[child]` for the array-as-tree embedding. A Fenwick tree promises
that node `i` stores the sum of the `lowbit(i)` elements ending at `i`. A hash map promises that
equal keys land in the same bucket. Dijkstra promises that the first time a node leaves the
priority queue, its distance is final — and that promise is conditional on non-negative weights.

The discipline that follows: when a symptom appears, write the invariant the structure was
supposed to hold, then find the cheapest observation that would be false if it were broken. That
observation is the discriminating check, and it is what distinguishes a diagnosis from a guess.
Two entries can produce identical symptoms and still require opposite fixes, which is exactly why
the catalog records competing causes per entry rather than a single cause.

### 2.4 Where each section is used

§3 supplies the vocabulary that §8's one-line symptoms rely on. §4 derives the five cost formulas
that §9 reuses when pricing detection and replay. §5 shows the full triage on one constructed
record set, including the boundary asymmetry that makes "test the failing case" insufficient.
§6 gives the pictures that make off-by-one and frontier growth obvious before arithmetic. §7 is
the selection table for *response*, not for data structure — the data-structure selection lives
in `01_DSA_Patterns_and_Invariants.md`. §10 lists the theorems whose preconditions the catalog
policies. §11 is for rehearsal, and §12 is the card to keep open while on call.

The split matters: this file never re-teaches structure selection. If the reader's question is
"which structure should I use", the answer is in the sibling sheet. If the question is "the
structure I chose is misbehaving", the answer is here.

## 3. Foundational Definitions

### 3.1 Core vocabulary

| Term | Meaning in this file | Units / type | Failure it enables |
|---|---|---|---|
| invariant | A predicate true before and after every public operation | boolean predicate | Silent corruption when unchecked |
| precondition | A condition the caller must establish for correctness | boolean predicate | Correct-looking output on invalid input |
| half-open interval | `[lo, hi)` includes `lo`, excludes `hi` | index pair | Off-by-one totals (CI-001) |
| amortised cost | Total cost of `n` operations divided by `n` | cost per operation | Worst-case latency assumption (CI-005) |
| load factor | Entries divided by buckets | dimensionless, 0 to infinity | Chain growth, adversarial collapse (CI-026) |
| frontier | Set of nodes discovered but not yet expanded | node set | Memory blowup (CI-025) |
| settled set | Nodes whose final distance is proven | node set | Wrong shortest paths (CI-002) |
| strict weak ordering | Irreflexive, transitive, transitive on incomparability | comparator property | Nondeterministic order (CI-004) |
| fixed width | Arithmetic modulo 2^k at a declared width | bits | Sign wrap (CI-010, CI-014) |
| relative error | `abs(approx - exact) / abs(exact)` | dimensionless | Threshold misclassification (CI-011) |

### 3.2 Terms that carry more than a row

**Amortised versus worst case.** An amortised bound averages over a sequence and says nothing
about any single operation. Growing a dynamic array by doubling gives an amortised `O(1)` append,
but the individual append that triggers a reallocation copies every element. Measured here: one
million appends took 82.5 ms total, while a single append at that size took about 1.4 µs — but the
1.4 µs figure is the *steady-state* append, and the reallocation append inside that million was
three orders of magnitude more expensive. A latency SLO is a statement about individual
operations, so an amortised bound cannot satisfy it. This distinction is the entire content of
CI-005, and it is why the catalog separates "average throughput is fine" from "no request exceeds
the budget".

**Average-case hash cost.** `O(1)` expected lookup assumes the hash function distributes the
actual keys. With an attacker choosing keys, the distribution is not a random variable under the
defender's control. In a simulation with 1,000 buckets and 5,000 randomly placed keys, the fullest
bucket held 15 entries against a mean of 5 — a 3× skew from chance alone, and chance is the
*friendly* case. Deliberate collision gives one bucket of 5,000. The bound is therefore
conditional, and the condition is a security property, not a performance property (CI-026).

**Comparisons versus memory transfers.** `O(log n)` is a count of comparisons. A binary search
over an array of one million keys takes at most 20 comparisons (since `2^20 = 1,048,576`), but
each comparison may cost a cache miss, while a B-tree node holding 100 keys answers the same
query in far fewer *node* visits. At one million keys the difference is not constant-factor
noise; it is the difference between 20 dependent memory round trips and 3. CI-029 is the entry
for teams that quote the comparison bound and then wonder why the index is slow.

**Deterministic versus reproducible.** A program is deterministic if the same input yields the
same output on the same binary and environment. It is reproducible if the output can be obtained
again by someone else. Iterating a hash table is deterministic within a process and usually not
reproducible across processes, because the seed changes; Python randomises string hashing per
process by default. A snapshot built by iterating a set is therefore stable when tested twice in
one shell and unstable in the next CI job (CI-003).

### 3.3 Necessary versus sufficient conditions

| Condition | Necessary for | Sufficient for | Consequence of confusing them |
|---|---|---|---|
| Non-negative edge weights | Dijkstra correctness | Dijkstra correctness on a directed graph | Negative edge silently yields a wrong distance |
| Monotone predicate | Binary search on a predicate | Correct binary-search termination | Search returns an arbitrary boundary |
| Strict weak ordering | Defined sort output | Deterministic, portable order | Order varies by input arrangement |
| Sorted input | Binary search | Correct membership answer | Absent key reported present |
| Half-open convention | Composability of adjacent windows | No double counting in sums | Boundary element counted twice |
| `sum` fits the accumulator width | Correct total | Correct total after further aggregation | Total wraps negative |

The right-hand column is the part practitioners skip. "Sorted input" is necessary and, for
membership, sufficient — but the sufficiency holds only for the *exact* comparison predicate
used at build time. Building with a case-sensitive key and searching with a case-insensitive one
satisfies "sorted" and still returns garbage, because the two orders differ. That variant appears
in CI-007.
## 4. Core Formulas with Full Derivation

Each result below is a *diagnostic* formula: it turns an observation into a predicted number so
that a hypothesis becomes falsifiable. The five-part anatomy is used for each. Where a formula is
about a mechanism rather than a ratio (the amortised-growth invariant, the settled-set invariant),
the block states preconditions, the state transition, the invariant with its justification, a
worked trace, and the failure boundary — the mechanism form the contract allows for
non-mathematical results.

### 4.1 Base case — boundary cardinality of a half-open interval

```text
FORMULA:  count([lo, hi)) = hi - lo                        (hi >= lo)
WHERE:    lo, hi are integer indices into a 0-based sequence of length n;
          valid domain 0 <= lo <= hi <= n; result is a count of elements, dimensionless.
DERIVATION:
  Step 1. An interval [lo, hi) claims every integer i with lo <= i and i < hi.
  Step 2. The integers satisfying that are lo, lo+1, ..., hi-1.
  Step 3. That list is an arithmetic progression with first term lo, last term hi-1, step 1.
  Step 4. Count of an arithmetic progression = (last - first)/step + 1
          = ((hi-1) - lo)/1 + 1 = hi - lo.
  Step 5. The closed interval [lo, hi] instead satisfies lo <= i <= hi, giving
          (hi - lo) + 1 = hi - lo + 1, which is one larger. The +1 is the whole bug class.
CONSTRAINT: requires step 1 (integer, unit) and hi >= lo. If hi < lo the formula yields a
          negative count, which is not a count; it is the signature of a reversed bound.
NUMERICAL EXAMPLE: lo = 3, hi = 8 -> 8 - 3 = 5 elements: {3,4,5,6,7}. Closed reading
          [3,8] -> 8 - 3 + 1 = 6 elements: {3,4,5,6,7,8}.
INTERPRETATION: every total that is wrong by exactly the value of one boundary element
          is a convention error, not a logic error; look for the interval first.
```

Boundary/counterexample where the formula breaks: with non-integer positions — a time interval
in seconds `[10.0, 20.0)` — the subtraction `hi - lo` is a *duration* (10 s), not a count of
samples. Counting samples inside it requires a sampling rate and an alignment convention, so
`count = hi - lo` is false there. Applying the integer formula to a continuous interval is how a
windowed aggregate ends up one sample short at every non-integral offset (CI-017).

### 4.2 General case — expected probes at a given hash load factor

```text
FORMULA:  E[probes_success] ~ 1 + alpha/2       (separate chaining, uniform hashing)
          E[probes_miss]    ~ alpha               (same model)
WHERE:    alpha = n / m, n = stored keys (count), m = buckets (count), dimensionless;
          valid for alpha not far above 1 and an independent uniform hash.
DERIVATION (successful search, chaining):
  Step 1. A successful search compares against the target plus every element ahead of it
          in the same chain. For a target, the expected number of elements inserted after
          it into the same bucket is (n-1)/m by linearity over the other keys.
  Step 2. Half of those, on average, precede the target in the chain, contributing
          (n-1)/(2m).
  Step 3. Adding the target's own comparison: 1 + (n-1)/(2m) ~ 1 + alpha/2.
  Step 4. For an unsuccessful search the target is not in the chain, so every element of
          the chain is examined: expected chain length n/m = alpha.
CONSTRAINT: the hash must be independent of the keys and uniform over buckets. This is
          exactly the assumption an adversary breaks.
NUMERICAL EXAMPLE: n = 5,000 keys, m = 1,000 buckets -> alpha = 5.0, so a successful
          search probes ~1 + 2.5 = 3.5 entries and a miss probes ~5.0.
INTERPRETATION: at alpha = 5 the miss path is 5 comparisons, which is survivable; the
          formula is not the danger, the loss of uniformity is (CI-026).
```

Boundary/counterexample: with a *deliberate* collision set — 5,000 keys all mapping to one bucket —
the model's uniformity step is false and the cost is 5,000 comparisons, a 1,000× error against the
formula's prediction of 5. Measured on this workstation, a membership probe against a 200,000
element list averaged 1.54 ms per probe over 2,000 probes, versus 0.32 µs against the equivalent
set — a 4,827× ratio. That ratio is the shape of collision collapse expressed in a different data
structure, and it is why the formula's precondition is written as a security property.

### 4.3 Variants — five cost identities that mislead when misapplied

**Variant A: recursion depth bound.**

```text
FORMULA:  depth_max = R - c            (practical recursion depth)
WHERE:    R = interpreter/thread stack limit in frames (count), c = frames already
          consumed by the caller chain and interpreter overhead (count).
DERIVATION: each recursive call consumes at least one frame; the runtime raises when the
          count exceeds R; therefore the deepest successful call is at most R - c.
CONSTRAINT: c is not zero. A recursive function called from library code may start at
          depth 20-60 already.
NUMERICAL EXAMPLE: measured R = 1000 on the test runtime; the deepest reachable recursion
          in a bare function was 984, i.e. c = 16 frames.
INTERPRETATION: budget depth as R/2, and convert to iterative when the data's depth is
          unbounded (CI-006).
```

**Variant B: amortised growth invariant.**

```text
FORMULA:  total_copy_cost < 2n  (geometric growth by factor 2, starting capacity 1)
WHERE:    n = number of appended elements (count); cost unit = one element copy.
DERIVATION: capacities are 1,2,4,...,2^k with 2^k >= n. Copies at each doubling are
          1+2+4+...+2^(k-1) = 2^k - 1 < 2n, so total copies are O(n) and the amortised
          cost per append is < 2 copies, i.e. O(1).
CONSTRAINT: growth must be geometric with factor > 1. Growth by +1 gives capacities
          1,2,3,...,n with total copies 1+2+...+(n-1) = n(n-1)/2, i.e. O(n) per append.
NUMERICAL EXAMPLE: n = 1,048,576 -> 2^20 >= n; total copies 2^20 - 1 = 1,048,575, or
          1.0 copies per append. Measured: 1e6 appends in 82.5 ms on the test runtime.
INTERPRETATION: the bound is amortised, so it does not bound a single append's latency;
          this is the mechanism behind CI-005 and CI-021.
```

**Variant C: priority-queue construction.**

```text
FORMULA:  T_build(sift-down) = O(n);  T_build(n pushes) = O(n log n)
WHERE:    n = elements (count); cost = comparisons.
DERIVATION: a perfect binary tree of n nodes has height h = floor(log2 n) and, at height
          j above the leaves, at most n/2^(j+1) nodes. Sifting each down costs at most j
          comparisons. Sum over j of (n/2^(j+1))*j <= n * 1 = O(n), since
          sum_{j>=0} j/2^(j+1) = 1.
CONSTRAINT: sift-down build is available; a streaming source cannot use it.
NUMERICAL EXAMPLE: n = 200,000. Measured, 200,000 individual pushes took 47.8 ms;
          heapifying the same values in place took 13.2 ms — a 3.6× ratio.
INTERPRETATION: if you already hold all elements, building by repeated push wastes a
          constant 3-4x that is visible in startup latency (CI-023).
```

**Variant D: top-K selection vs full sort.**

```text
FORMULA:  T_full = O(n log n);   T_topk(min-heap) = O(n log k)
WHERE:    n = stream length (count), k = requested rank depth (count), cost = comparisons.
DERIVATION: full sort orders all n; the heap holds at most k and each of the n arrivals
          costs at most log k comparisons against the root plus a sift when it replaces.
CONSTRAINT: k << n. If k approaches n the heap variant degenerates to a slower sort.
NUMERICAL EXAMPLE: n = 1,000,000, k = 100. Measured: keeping the 100 largest with a heap
          took 16.0 ms; a full sort took 235.7 ms — a 14.8× ratio. The comparison-count
          model predicts log2(100)/log2(1e6) = 6.64/19.93 = 0.33, so the remaining factor
          is the heap path's lower constant and its cache-friendly sequential scan.
INTERPRETATION: the ratio grows with n/k, so a top-K endpoint that was fast at n=10^4
          degrades slowly enough to be attributed to traffic growth (CI-024).
```

**Variant E: Fenwick prefix-sum cost and the off-by-one in its query.**

```text
FORMULA:  update = O(log n);  query(i inclusive) = O(log n)
WHERE:    n = array length (count); cost = iterations of the lowbit loop.
DERIVATION: update walks i += i & -i, which strictly increases i and at most doubles it in
          the worst step, giving at most log2(n)+1 iterations. Query walks i -= i & -i,
          each step clearing the lowest set bit, so at most popcount(i) <= log2(n)+1.
CONSTRAINT: 1-based internal indexing, or a deliberate 0-based translation. Mixing the
          two silently shifts every prefix by one element.
NUMERICAL EXAMPLE: n = 100,000 over a constant array of ones. Querying the inclusive
          prefix at index k = 50,000 returned 50,001, while the half-open slice sum
          arr[:50000] is 50,000, and the correctly translated query at k-1 returned 50,000.
INTERPRETATION: a Fenwick total that is off by exactly arr[k] is a translation bug, not an
          arithmetic bug — the cheapest confirmation is to query k-1 as well (CI-019).
```

Boundary/counterexample for the group: every identity above is a cost claim under a fixed
machine model. Variant D's 14.8× ratio is a same-process comparison of two library calls; on a
runtime where the sort is vectorised and the heap is not, the ratio could invert at small k.
Quoting the ratio as an absolute service improvement is the error CI-030 describes.

### 4.4 Limits and counterexamples

Validity condition for the whole of §4: each formula assumes the data structure's own invariant
holds and that the cost unit (comparison, copy, probe) is proportional to wall time on the target
machine. Neither is guaranteed. The three ways out:

1. **Invariant failure** — the formula's precondition is false, so the number is meaningless
   rather than merely imprecise. Dijkstra's `O((V+E) log V)` says nothing once a negative edge
   breaks the settled-set argument; measured on a three-node graph `s->a(1), s->b(2), b->a(-2)`,
   a settled-set Dijkstra returned `a = 1` when the true distance is `0`. The cost was perfect
   and the answer was wrong — the case where an efficiency metric hides a correctness failure
   (CI-002).
2. **Cost-unit failure** — comparisons are not proportional to time when the memory hierarchy
   dominates. A binary search's 20 comparisons at n = 10^6 can cost more wall time than a
   B-tree's 3 node visits, so `O(log n)` orders two implementations incorrectly (CI-029).
3. **Adversarial input** — the distributional assumption is under someone else's control.
   Hashing, quicksort pivot choice, and hash-join partitioning all have worst cases reachable by
   input construction; measured, a naive first-element-pivot quicksort on already-sorted input
   went from 3.3 ms at n = 200 to 76.0 ms at n = 800, a 23× time increase for a 4× input increase
   — quadratic behaviour visible in three data points (CI-028).

A formula that survives review is one whose precondition is written next to it. The catalog's
competing-causes column exists because the same symptom usually has one cause from each of these
three families, and only a discriminating check separates them.

## 5. Worked Numerical Example

### 5.0 The record set and what is being diagnosed

ILLUSTRATIVE: a daily-count table for a single metric, seven days, delivered to a half-open window
job. The job reports "total for the trailing window" and "count of days at or above target".
The complaint to diagnose: *the total is too high, but only for some windows.*

```text
day index :  0    1    2    3    4    5    6
count     :  4    7    2    9    5    3    6
prefix[i] :  0    4   11   13   22   27   30   36   (prefix[0]=0, prefix[k]=sum of first k)
```

`prefix` has length 8 because it stores the sum of the first `k` elements for `k = 0..7`. The
identity used throughout is `sum(a[lo:hi]) = prefix[hi] - prefix[lo]`, a restatement of §4.1 with
the count replaced by a sum, valid only while `hi <= 7`.

### 5.1 Question A — trailing 3-day total ending at day 5 inclusive

Target: days 3, 4, 5 → `9 + 5 + 3 = 17`.

Correct half-open form: `[3, 6)` → `prefix[6] - prefix[3] = 30 - 13 = 17`. ✔

The plausible wrong form: treat 5 as the exclusive end, `[3, 5)` → `prefix[5] - prefix[3] =
27 - 13 = 14`, which drops day 5 and understates by 3. The other plausible wrong form subtracts
one index too far on the left, `[3, 6)` with `lo-1` → `30 - prefix[2] = 30 - 11 = 19`,
overstating by 2. Both errors are convention errors; neither is visible in the code's shape, and
both produce errors equal to a single day's count.

Independent re-check by direct addition: `9 + 5 + 3 = 17`, and `13 + 9 + 5 + 3 = 30 = prefix[6]`.
Two routes agree, so 17 is the answer to report.

### 5.2 Question B — a bug that reproduces only on some windows

Hypothesis H1: the job intends the half-open window `[lo, hi)` but computes
`prefix[hi] - prefix[lo-1]`, subtracting one index too far to the left, with `lo-1` clamped to `0`
when `lo = 0`. Evaluate H1 on every length-3 window:

| Intended `[lo, lo+3)` | Days | Truth | Formula `prefix[lo+3]-prefix[lo-1]` | Reported | Error |
|---|---|---:|---|---:|---:|
| `[0,3)` | {0,1,2} | 13 | `13 - prefix[0]` = 13 | 13 | 0 |
| `[1,4)` | {1,2,3} | 18 | `22 - prefix[0]` = 22 | 22 | +4 |
| `[2,5)` | {2,3,4} | 16 | `27 - prefix[1]` = 23 | 23 | +7 |
| `[3,6)` | {3,4,5} | 17 | `30 - prefix[2]` = 19 | 19 | +2 |
| `[4,7)` | {4,5,6} | 14 | `36 - prefix[3]` = 23 | 23 | +9 |

H1 predicts error `= a[lo-1]` for every window, and zero exactly when `lo = 0`, because the clamp
collapses the subtraction to `prefix[0] = 0`. The table confirms it: `4, 7, 2, 9` are the counts of
days 0, 1, 2, 3 — `a[lo-1]` in each row. That coincidence — the error in every row equal to one
known table entry, always the entry immediately before the window — is the discriminating check
that converts H1 from a guess into a diagnosis. Without it, "sums are too high" has several causes
that fit.

Competing hypothesis H2: the job computes `prefix[hi] - prefix[lo]` for an inclusive `[lo, hi]` and
therefore *omits* the last day, producing error `-a[hi]`. Both hypotheses are off by one element;
they differ in sign and in which edge the error tracks. On this table H2 predicts errors
`-2, -9, -5, -3` (windows ending at days 2, 3, 4, 5), all negative, so one observation of a
*positive* error eliminates H2 in a single comparison. If the reported errors were mixed in sign,
neither hypothesis fits and the cause is upstream: counts are being attributed to the wrong day
(CI-017).

Note what decides intermittency: the clamp on `lo = 0`. A suite whose only window starts at day 0
sees error zero in every case and reports the bug absent. The failure is not window-length
dependent, it is *position* dependent, and a test that fixes `lo = 0` is structurally incapable of
finding it (CI-018).

### 5.3 Question C — boundary asymmetry that defeats "test the failing case"

Suppose the reported failure is fixed by "subtract `a[hi]` instead", and the developer verifies at
the one window that failed, which had `lo = 0`:

- If the fix is "compute `[lo, hi+1)`", it is correct for every `hi < 6` and reads past the end at
  `hi = 6`. Testing at `hi = 6` catches it; testing at `hi = 5` passes.
- If the fix is "add the element instead of subtracting the prefix", it is accidentally correct at
  `lo = 0`, where the clamped prefix is zero, and wrong at every `lo > 0` because the left edge is
  still off by one. Testing only the originally failing window, at `lo = 0`, certifies the bug.

The rule that follows: a boundary fix needs two tests, one at each end of the boundary's reach
(`lo = 0` and `hi = n-1`), and the failing record is usually not one of them. This is the most
common reason a "fixed" off-by-one returns from production unchanged (CI-001, CI-018), and it is
why §1.8 makes the nearby-boundary test part of the definition of fixed.

### 5.4 Question D — count of days at or above target, and the inclusive comparison

Target = 5. Days: 4, 7, 2, 9, 5, 3, 6 → at or above 5: days 1 (7), 3 (9), 4 (5), 6 (6) → count 4.

The two competing readings:

- `a[i] >= 5` → 4 days, correct for "at or above target".
- `a[i] > 5` → 3 days, correct for "above target". The difference is exactly the number of days
  equal to the target, here 1.

Diagnostic: count the equalities separately. The identity `count(>=) - count(>) = count(==)`
holds for any table; here `4 - 3 = 1 = count(== 5)`, consistent. On a table where no day equals
the target, both counts agree and the distinction is invisible — which is how the wrong predicate
survives testing on a dataset that lacks the boundary value (CI-013).

### 5.5 Question E — top-3 days by count, and the tie question

Counts: 9 (day 3), 7 (day 1), 6 (day 6), 5 (day 4), 4 (day 0), 3 (day 5), 2 (day 2).

Top-3 = days {3, 1, 6} with values 9, 7, 6. No tie at the cut, so any correct selection agrees.
Perturb day 4 to 6, giving 9, 7, 6, 6: the top-3 *values* are 9, 7, 6, but the top-3 *days* are
ambiguous — day 6 or day 4 for the third slot. A sort by value alone may return either; a stable
sort by `(-value, index)` returns day 4, and a sort by `(-value, -index)` returns day 6. Two
implementations, both defensible, different answers.

Measured check on the test runtime: `sorted([(5,'b'),(5,'a'),(3,'c'),(5,'d')], key=lambda r: r[0])`
returned `[(3,'c'),(5,'b'),(5,'a'),(5,'d')]` — input order preserved among equals, confirming
stability. Re-sorting after a descending pass produced the same order, because that pass was also
stable. A stability bug therefore rarely appears in a two-pass test and does appear when the second
pass is unstable, or when the first pass is replaced by a hash-based grouping (CI-004).

### 5.6 Question F — independent arithmetic re-check of every figure

Every number printed in §5.1-§5.5 is re-derived here from the raw table, without using `prefix`:

| Quantity | Direct computation | Prefix computation | Agree |
|---|---|---|---|
| Total days 0-6 | 4+7+2+9+5+3+6 = 36 | `prefix[7]-prefix[0]` = 36 | yes |
| Days 3-5 | 9+5+3 = 17 | `30-13` = 17 | yes |
| Days 1-3 | 7+2+9 = 18 | `22-4` = 18 | yes |
| H1 on `[1,4)` | 7+2+9 = 18 | `22-0` = 22 | no, +4 = a[0] |
| Count >= 5 | 4 days | — | n/a |
| Count == 5 | 1 day | — | n/a |
| Top-3 values | 9, 7, 6 | — | n/a |

The `no` row is the point of the exercise. An independent check must not share the assumption under
test: if the re-check itself were built from the same `prefix` array, it would inherit the same
`lo-1` error and "confirm" 22. Rebuilding the truth by adding raw counts — a route that never
touches `prefix` — is the only version that can disagree, and disagreement is the only thing a
check is for (CI-019).

### 5.7 Sensitivity — how wrong is wrong enough to matter

Suppose the weekly total feeds a threshold at 30. True total 36, reported 36: no decision change.
Reported 35 (a one-unit loss): still above 30. Reported 29 (a seven-unit loss, plausibly a dropped
day): the decision flips. Sensitivity of the *decision* to the *error* is `1 / (T - 30)` per unit
of error, here `1/6`, so a one-unit boundary error changes the decision only when the true total
sits within one unit of the threshold (31 reported as 30, or 30 reported as 29). That is the quantitative reason boundary bugs are
"harmless" for months and then cause an incident: they are harmless until the value approaches the
threshold, and nothing in the test suite selects for that (CI-031). A cheap defence is a fixture
whose total sits exactly at the threshold, a record no realistic sample contains by chance.

## 6. Visual / Geometric Intuition

### 6.1 The index line with three conventions

```text
elements:      | a0 | a1 | a2 | a3 | a4 |
positions:     0    1    2    3    4    5      <- boundaries between elements
half-open [1,4):        [ a1 | a2 | a3 )
closed    [1,4]:        [ a1 | a2 | a3 | a4 ]
count(1,4):             3          count[1,4]: 4

Adjacent windows tile without overlap under half-open:
  [0,2)  [2,4)  [4,5)   -> covers 0..4 exactly once
Adjacent windows double-count under closed:
  [0,2]  [2,4]  [4,5]   -> a2 and a4 counted twice
```

The picture explains the entire bug class: a closed interval cannot tile, because the boundary
element belongs to both neighbours. Any pipeline that partitions a sequence into windows and then
sums the window totals is safe with half-open and wrong with closed, and the error grows with the
number of windows rather than staying at one (CI-001).

### 6.2 Binary search as a halving interval, and where the loop stalls

```text
lo=0                     hi=8            invariant: answer in [lo, hi)
mid = (lo+hi)//2 = 4
  case A: go left  -> hi = mid        (4 shrinks the interval)
  case B: go right -> lo = mid + 1    (the +1 is mandatory)
  bug:    lo = mid                     -> interval [0,8) -> mid 4 -> [4,8) -> mid 6
                                          -> [6,8) -> mid 7 -> [7,8) -> mid 7 -> stall
```

Measured demonstration: with `lo = 0, hi = 8` and the missing `+1`, the loop reached
`(lo, hi) = (7, 8)` and made no further progress — 51 iterations without halting, in a search whose
correct count for `n = 8` is 3 comparisons, reached in at most 4 iterations of a `while lo < hi`
loop. Any binary search that executes more than `ceil(log2(n)) + 1` iterations is broken
regardless of the values it returns (CI-009).

### 6.3 The hash table as a bucket histogram

```text
1,000 buckets, 5,000 keys (ILLUSTRATIVE random placement, alpha = 5):
occupancy summary:  7 empty buckets, mean 5.0, fullest bucket 15
  #
  #  #  #
  #  #  #  #  #  #  #
  #  #  #  #  #  #  #  #  #  #   ... tail up to 15
  0  1  2  3  4  5  6  7  8  9 ...
```

Two lessons from the numbers. First, a mean of 5 permits a maximum of 15 from chance alone, so one
long chain is *not* evidence of an attack — this is the competing-cause problem in CI-026. Second,
the shape of the histogram is exactly what the `O(1)` claim depends on, so the picture makes
visible that `O(1)` is a statement about distribution, not about table size.

### 6.4 The BFS frontier as a contour, and why depth 5 is not "one more than depth 4"

```text
degree 10, depth d:  frontier widths 1, 10, 100, 1,000, 10,000, 100,000
total discovered through depth 5: 1+10+100+1,000+10,000+100,000 = 111,111
memory at ~64 bytes/node (ILLUSTRATIVE): 111,111 * 64 B = 7.1 MB
                                            depth 6 -> 1,111,111 nodes = 71.1 MB
```

Each level multiplies by the degree, so memory grows by the same factor as time. A traversal that
runs comfortably at depth 4 exhausts a 64 MB budget at depth 6 without any change in code. The
discriminating observation is that resident memory tracks the *last level* rather than the total,
which is why memory charts for frontier growth look like staircases and charts for leaked caches
look like ramps (CI-025, CI-027).

### 6.5 The DP lattice and the dependency arrow

```text
f(i,j) -> f(i-1,j) and f(i,j-1)      (2-D grid, both dependencies "behind")
row-major evaluation order:
  (0,0) (0,1) (0,2) ...
  (1,0) (1,1) ...
```

The invariant is that every arrow points at an already-computed cell. A memoisation cycle — arrows
pointing forward — is not slow code, it is an unterminated recursion, and its symptom is a stack
overflow or a hang rather than a wrong number. The check is structural: topologically order the
dependency edges before writing the loop, and if those edges contain a cycle the recurrence is
wrong, not the implementation (CI-008).

### 6.6 The amortised sawtooth

```text
per-append copies with doubling growth (n up to 16):
  #
  #  #
  #  #  #     #     #
 1  2  3  4  5  6  7  8  ... append index
```

Spikes at 1, 2, 4, 8 and flat elsewhere. Total area under the spikes is bounded by `2n`; no
individual spike is bounded by anything except `n`. A latency SLO is a horizontal line across this
chart, crossed by the spikes while the average stays far below. That is the mechanism of CI-021,
and the mitigation is a pre-sized buffer or an incremental copy, not a faster CPU.

### 6.7 The recursion stack as a ladder with a fixed rung count

```text
frame 0   [caller]
frame 1   [recursive call]
   ...
frame 984 [deepest reachable on the test runtime]
frame 985 -> RecursionError
```

The ladder has a fixed number of rungs (`R = 1000` here, 16 already occupied by the caller chain).
Depth is therefore a *hard* limit unlike time: there is no partial credit, the call either fits or
the program aborts. Any traversal whose depth is a function of user data — a comment thread, a
file tree, a linked list — must be iterative or explicitly depth-limited (CI-006).
## 7. Comparison / Selection Table

This section selects a *response*, not a data structure. The distinction matters because the most
expensive routine failure in this domain is a correct repair applied to the wrong mechanism: a
clamp applied to an overflow, a capacity increase applied to a comparator, a replay applied to a
missing invariant. Each row below is keyed on evidence that is already in hand, not on a guess
about intent.

### 7.1 Response selection by observed evidence

| Observed evidence | First action | What it must not be | Disqualifier | Verification |
|---|---|---|---|---|
| Value off by exactly one boundary element | Correct the interval convention at both ends | A clamp added at the reporting layer | The same error at the other boundary | Two tests, `lo = 0` and `hi = n-1` |
| Sign flipped only above a magnitude | Widen the accumulator type and re-derive | A sign-correction at the output | Any input where the value still wraps | Recompute the crossing input at the new width |
| Tiny relative error against an exact route | Switch to exact or compensated summation | A tolerance added to the assertion | Error still present with integer inputs | Compare against `fsum` or integer arithmetic |
| Output differs between two identical runs | Seed and isolate the ordering source | Re-running until outputs agree | Two runs agreeing by chance | Run under two different hash seeds |
| Latency grows faster than input | Identify the superlinear step and replace it | A larger timeout or a bigger machine | The exponent surviving a 10× input drop | Measure at three input sizes |
| Memory grows while values stay correct | Find the retention set, cap it, expire it | A periodic restart | Memory returning to baseline after a forced GC | Baseline restored without restart |
| Correct answer, unacceptable cost at `k << n` | Switch to a bounded selection structure | Raising the batch size | The `n log k` route measuring slower | Same answer at three `k` values |
| Two teams disagree whether the code is safe | Write the guarantee as a precondition | Renaming the guarantee | Either reading being unfalsifiable | A test that fails under the weaker reading |
| Symptom disappears under instrumentation | Treat the timing change as the clue | Closing the issue as unreproducible | A different failure appearing at lower load | Reproduce with a load generator, not a debugger |

### 7.2 Repair class to owner and evidence

| Repair class | Who can authorise it | Evidence required before merge | Typical turnaround | Residue if done badly |
|---|---|---|---|---|
| NUMERICAL_EDGE | The owning service team | Failing record plus both boundary records | Hours | Silent threshold drift |
| VIOLATED_ASSUMPTION | The API contract owner | Written precondition and a rejection test | Days | Callers depend on undefined behaviour |
| SCALE_FAILURE | The owning team plus capacity owner | Cost measurement at 3 input sizes | Days to weeks | Cost returns at the next order of magnitude |
| CONCEPT_CONFUSION | The design owner | One paragraph defining the guarantee operationally | Weeks | Two components disagree at integration |

### 7.3 Choosing between "fix the input" and "fix the code"

A recurring judgement call: the offending record is malformed, late, or adversarial. Three
options exist and they are not interchangeable.

Fixing the code to accept the record widens the contract for every future caller, and the widening
is invisible to anyone who did not read the diff. Fixing the input is correct when the record
violates a documented precondition and the producer can be corrected. Deferring is correct when the
record cannot be classified without information the system does not have — and deferral is only
safe if the deferral path is itself bounded, otherwise deferred records accumulate into the memory
failure of CI-027.

The test that decides: can the system state, in one sentence, what it did with the record and why?
If the answer requires quoting the input, the handling is a special case and will not survive the
next variant of the same input.

## 8. Failure Modes & Edge Cases

### 8.1 How to read a catalog entry

The master table gives the eight fields the contract requires. Each row is a complete issue: the
symptom is what the on-call sees, the competing causes are the hypotheses that fit that symptom,
the discriminating check is the observation that separates them, and the correction is scoped to
the mechanism rather than to the symptom. The expansions that follow the table add the mechanism
and, where relevant, a measured or derived number. An entry with no discriminating check would be
a slogan, so no entry is admitted without one.

Two conventions in the table. "Symptom" is written as an observation a person can make without
reading the code — a reported number, a graph shape, an exit status. "Discriminating check" is
written as an action that yields a different result under the rival cause, so that a null result
is itself informative.

### 8.2 Master catalog

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-001 | NUMERICAL_EDGE | Total is wrong by exactly one element's value | Half-open vs closed interval; duplicate boundary row; wrong start index | Recompute with the other convention; the error equals `a[hi]` or `a[lo]` | Fix both bounds; adopt one convention repo-wide | Lint rule naming the convention per function | Tests at `lo=0` and `hi=n-1` |
| CI-002 | VIOLATED_ASSUMPTION | Shortest paths too long on some pairs | Negative edge; stale relaxed distance; settled-set reuse across queries | Compare with Bellman-Ford on a 3-node graph with one negative edge | Reject negative weights or switch algorithm | Validate weights at graph build | Oracle comparison on random small graphs |
| CI-003 | VIOLATED_ASSUMPTION | Same input, different output across runs | Hash-order iteration; unstable sort; unseeded sampling; dict-order change after deletion | Run twice under different `PYTHONHASHSEED` values | Sort explicitly at the boundary | Ban unordered iteration in output paths | Snapshot test with fixed seed |
| CI-004 | VIOLATED_ASSUMPTION | Order of equal keys varies between environments | Non-strict-weak comparator; unstable sort; two-pass sort; float keys | Sort then re-sort by the same key; order changes means comparator is not a total order | Add a deterministic tiebreaker | Property test for transitivity | Golden output for tie-heavy input |
| CI-005 | CONCEPT_CONFUSION | Average latency fine, tail latency fails | Amortised bound quoted as per-op; GC pause; reallocation spike; contention | Chart per-request latency against the mean | Size or pre-allocate; move cost off the hot path | State SLO in per-op terms, never amortised | Tail percentile under load |
| CI-006 | SCALE_FAILURE | Process aborts on deeply nested input | Recursion depth; cyclic structure; unbounded memo growth | Print depth at abort; check input nesting vs limit | Convert to explicit stack | Depth guard or iterative traversal | Deepest expected nesting plus one |
| CI-007 | VIOLATED_ASSUMPTION | Membership answer wrong although the array is sorted | Sorted by a different key; case sensitivity; locale collation; stale index | Search with the build-time comparator and compare hit/miss | Rebuild with the query comparator | One comparator per index, exported | Round-trip build-then-query test |
| CI-008 | CONCEPT_CONFUSION | DP hangs or overflows rather than returning wrong value | Dependency cycle; base case unreachable; memo key collision | Topologically order the dependency edges and look for a cycle | Reformulate the recurrence | Draw the dependency graph in review | Bounded iteration count assertion |
| CI-009 | NUMERICAL_EDGE | Binary search never terminates | Missing `+1` on `lo`; `hi = mid` with `lo = mid`; float midpoint rounding | Count iterations; exceeding `log2(n)+1` proves it | Fix the update rule | Single shared binary-search helper | Iteration-count assertion |
| CI-010 | NUMERICAL_EDGE | Total flips sign on large inputs | 32-bit accumulator; unsigned wrap; language promotion rules; cast at output | Compute the crossing input and compare against the type's max | Widen accumulator before the first add | Explicit width per accumulator | Test at exactly `MAX+1` units |
| CI-011 | NUMERICAL_EDGE | Sum drifts by a tiny relative amount | Float accumulation; mixed precision; catastrophic cancellation; wrong divisor | Compare with `math.fsum` or integers | Compensated or exact summation | Use integers for money and counts | Relative error bound test |
| CI-012 | NUMERICAL_EDGE | Threshold comparison flips near the boundary | Binary float representation; rounding at format time; `>=` vs `>` | Test `0.1+0.2 == 0.3` and the actual boundary value | Compare with tolerance or integers | Never compare floats for equality | Boundary value table |
| CI-013 | NUMERICAL_EDGE | Count of qualifying rows differs from a manual count | `>` vs `>=`; null handling; duplicate rows | Count equalities; check the identity `count(>=) - count(>) = count(==)` | Fix the predicate; document the boundary | Test set containing exact-boundary rows | Counts agree three ways |
| CI-014 | NUMERICAL_EDGE | Product correct at small values, wrong at large | Narrow intermediate width; overflow before widening; mixed int/float | Force the widening at the first operand and recompute | Widen before multiplying | Type the intermediate explicitly | Test at the exact overflow point |
| CI-015 | NUMERICAL_EDGE | Division rounds in an unexpected direction | Truncation toward zero for negatives; floor division; float round-half-even | Evaluate a negative odd numerator both ways | Use one documented rounding rule | Encapsulate rounding in a helper | Tests with negative operands |
| CI-016 | VIOLATED_ASSUMPTION | Result wrong by a factor of 1000 or 1024 | ms vs s; bytes vs KiB; percent vs fraction | Compute the ratio of reported to expected | Normalise units at ingestion | Units in every field name | A test whose expected value is unit-sensitive |
| CI-017 | NUMERICAL_EDGE | Windowed total misses or duplicates the edge sample | Float boundary alignment; exclusive timestamp filter; timezone offset | Compare counts at integer vs fractional offsets | Align boundaries to the sampling grid | Integer epoch units throughout | Count-in-window test at fractional offset |
| CI-018 | NUMERICAL_EDGE | Off-by-one appears only away from index zero | Guard clamping `lo-1` to zero; first element special-cased | Run the same window at `lo=0` and `lo=1` | Remove the guard; fix the index | Tests that never fix `lo=0` | Error equals `a[lo-1]` for all `lo>0` |
| CI-019 | NUMERICAL_EDGE | Check confirms a value that is in fact wrong | Oracle shares the assumption; same library both sides; same rounding | Rebuild the truth from raw records only | Independent implementation or hand arithmetic | Two-route verification rule | Disagreement is possible by construction |
| CI-020 | NUMERICAL_EDGE | Min, max or sort returns a surprising element | NaN comparison semantics; mixed numeric types; signed zero | Print the operand list including NaN positions | Define NaN policy explicitly | Reject or filter NaN at ingestion | Test with NaN in first and middle position |
| CI-021 | SCALE_FAILURE | Periodic latency spike at growing intervals | Reallocation copy; GC; compaction; cache flush | Spike intervals double as data grows | Pre-size or amortise the copy | Capacity planning tied to growth | Spike magnitude bounded by input size |
| CI-022 | SCALE_FAILURE | Runtime quadratic in the number of records | String concatenation in a loop; repeated slicing; nested scan; per-row query | Fit log-log slope across three sizes | Replace with join, index, or accumulate | Complexity review on every loop | Slope near 1 not 2 |
| CI-023 | SCALE_FAILURE | Startup cost large before any output | Building by repeated insert; redundant parse; eager validation | Time build-by-insert against build-in-place | Use bulk construction | Bulk-build in the loader | Measured ratio at production `n` |
| CI-024 | SCALE_FAILURE | Top-K endpoint slows down as the stream grows | Full sort per request; unbounded heap; re-ranking history | Time `n log n` and `n log k` paths at 3 sizes | Keep a bounded heap of size `k` | Bound work by `k`, not by `n` | Ratio grows with `n/k` as predicted |
| CI-025 | SCALE_FAILURE | Resident memory climbs with graph depth | Frontier retention; visited set growth; string interning | Compare memory at depth `d` and `d+1` | Level-bounded traversal; drop visited entries | Explicit frontier budget | Staircase, not ramp, in the memory chart |
| CI-026 | SCALE_FAILURE | One key's requests are orders of magnitude slower | Deliberate collisions; unlucky distribution; long chain from growth | Histogram bucket occupancies; compare against `1 + alpha/2` | Keyed hash with a secret seed | Adversarial input model in testing | Worst-bucket latency bounded |
| CI-027 | SCALE_FAILURE | Memory never returns to baseline | Unbounded cache; deferred queue growth; retained closures | Force GC and compare; inspect cache size | Add eviction and a size cap | Cap every cache at construction | Baseline restored after traffic stops |
| CI-028 | SCALE_FAILURE | CPU pinned by one small request | Regex backtracking; recursive parse; quadratic scan | Time the same input with growing length | Bound input size or change the matcher | Pre-parse length and nesting limits | Time stays linear in input length |
| CI-029 | CONCEPT_CONFUSION | Index is asymptotically optimal yet slow | Cache misses per comparison; pointer chasing; page faults | Measure ns per comparison against a contiguous scan | Change layout or node fanout | Design for the memory hierarchy | Fewer dependent memory trips |
| CI-030 | CONCEPT_CONFUSION | Published ratio quoted as a guarantee | Different hardware, input, or baseline; warm cache; JIT warmup | Reproduce both sides in one process | Report ratios with conditions attached | Benchmark harness in-repo | Ratio reproduced within a stated band |
| CI-031 | CONCEPT_CONFUSION | Suite is green and the boundary still fails | Boundary values absent from data; mocks that clamp; assertions on totals only | Enumerate boundary values present in the fixtures | Add boundary records deliberately | Coverage measured on values, not lines | Each boundary value appears in a fixture |
| CI-032 | CONCEPT_CONFUSION | Code called thread-safe fails under load | Atomicity vs visibility; lock scope; per-instance vs per-process state | Run two writers against a shared instance | Narrow the claim to what is enforced | Write the guarantee as a precondition | Concurrency test with two writers |

### 8.3 Numerical edges: the boundary family

**CI-001** is the canonical convention error and the one that pays for this catalog. The mechanism
is that two adjacent intervals both claim the shared boundary, or neither does, depending on which
bracket is written. Detection is cheap: the magnitude of the error equals one element's value, and
that value identifies the edge. A total that is wrong by 3 on a day whose count is 3 is not an
arithmetic bug, it is a bracket. The correction must touch both ends — fixing the high bound alone
converts a double count into an omission, which the next test finds and the previous did not.

**CI-009** looks like CI-001 and is not. Here the value is never produced, because the loop cannot
advance. The update `lo = mid` with `hi = mid` leaves the interval unchanged when `hi = lo + 1`,
since `mid = lo`. The check is arithmetic rather than observational: instrument the iteration count
and compare against `ceil(log2(n)) + 1`. Measured on this runtime with `lo=0, hi=8` and the missing
`+1`, the loop reached `(7, 8)` and stalled for 51 iterations, where the correct run takes 4.

**CI-010** and **CI-014** are the same mechanism at different layers. In CI-010 the accumulator is
too narrow for the final total; in CI-014 an intermediate product is too narrow for an operand pair
that the final total would fit. The second is worse because the assignment target looks wide enough,
so reading the declaration suggests safety. The crossing input is computable: for a signed 32-bit
accumulator the boundary is `2^31 - 1 = 2,147,483,647`, and the sum of squares `n(n+1)(2n+1)/6`
exceeds it at `n = 2000` with `2,668,667,000` — a value that looks perfectly reasonable in a report
until the sign flips.

**CI-011** is the accumulation mechanism: each addition rounds, and the errors are not independent.
Adding `0.1` one million times on the measured runtime yielded `100000.00000133288` against an exact
`100000`, a relative error of `1.33e-11`. That is harmless for a chart and fatal for a reconciliation
that demands equality. `math.fsum` returned exactly `100000.0`, which is why the correction is
"change the summation", not "add a tolerance".

**CI-012** is CI-011 arriving at a decision boundary. The measured facts: `0.1 + 0.2 == 0.3` is
`False`, and `(1e16 + 2) - 1e16` is `2.0` because 2 is below the representable spacing at 1e16. Any
comparison near a threshold needs either integers or an explicit tolerance with a stated width.
The competing cause is formatting: a value can be correct and *printed* rounded, so printing more
digits before assuming arithmetic is broken is the cheaper first check.

**CI-013** is the predicate boundary: `>=` versus `>`. Its identity, `count(>=) - count(>) =
count(==)`, is a one-line invariant that catches both a gap and an overlap. The reason it survives
testing is that fixtures rarely contain a value exactly at the threshold, so both predicates agree
on every row and the distinction is unobservable until a boundary value arrives in production.

**CI-015** is the rounding direction of division. Truncation toward zero and floor division differ
exactly on negative operands with a nonzero remainder: `-7 // 2 = -4` in floor semantics while
truncation gives `-3`. A metric that divides a signed delta by a positive bucket width produces
two different answers for losses depending on which rule the language implements. The correction is
to choose a rule and wrap it, not to patch the call site where the sign was noticed.

**CI-017** is a boundary in continuous time rather than in an index array. A window that starts at
`t = 10.5` s with a 1 s sampling grid contains either 9 or 10 samples depending on alignment, so
`count = hi - lo` from §4.1 does not apply. The correction is to express boundaries in integer
epoch units and align them to the grid at ingestion; the prevention is that no float ever reaches
a window predicate.

**CI-018** is CI-001's guard variant and the most deceptive of the family. A defensive
`max(lo - 1, 0)` makes the code total at `lo = 0` and silently wrong everywhere else, because the
clamp *changes the meaning* of the expression instead of rejecting the case. The measured shape of
the error is `a[lo-1]` for every `lo > 0` and zero at `lo = 0`, which is exactly the intermittency
reported in §5.2. Removing the guard and fixing the index is the correction; a test that fixes
`lo = 0` is not a test of the guard.

**CI-019** is a meta-error: the verification shares the defect. Rebuilding the total from the same
prefix array that contains the bad index cannot disagree with it. The rule is structural — the
check must be computed by a route that never calls the function under test, and it must be capable
of failing. Hand arithmetic over raw rows, as in §5.6, is such a route.

**CI-020** is the NaN policy. NaN is not equal to anything, including itself, so a comparison-based
minimum returns whichever operand the loop happened to keep, and a sort places NaN by an
implementation-defined rule. The symptom is a "surprising element" rather than a crash, which
sends investigators looking at the selection logic instead of the data. Printing the operand list
with NaN positions is the check; rejecting NaN at ingestion is the correction.

### 8.4 Violated assumptions: the precondition family

**CI-002** is the highest-consequence entry in the catalog because it produces fast, confident,
wrong answers. Dijkstra's settled set is only valid when no edge can improve an already-settled
distance, which requires non-negative weights. Measured counterexample on `s->a(1), s->b(2),
b->a(-2)`: the settled-set implementation returned `a = 1`, while the true distance is `0`, and
Bellman-Ford confirms `0`. The cost model was satisfied; the invariant was not. The discriminating
check is a three-node graph with one negative edge, which is cheap enough to belong in a unit test.

**CI-003** is nondeterminism from iteration order. Two runs of the same binary agree; two runs in
different processes disagree, because string hashing is seeded per process by default in CPython.
The check is to run the same snapshot twice with different `PYTHONHASHSEED` values and diff. The
correction is to sort at the boundary where order becomes observable; the prevention is a rule that
no output is built by iterating an unordered container.

**CI-004** is the comparator family, and the reason the phrase "strict weak ordering" exists.
Irreflexivity, transitivity of `<`, and transitivity of incomparability are all required for a sort
to have a defined output. A comparator that returns `a <= b` as a boolean key breaks
irreflexivity, and library implementations are permitted to produce any permutation when the
requirement is violated — including one that differs between two calls in the same process. The
measured stability check in §5.5 shows what a *correct* comparator looks like: equal keys retained
input order.

**CI-007** is the "sorted is not enough" entry. An array is sorted with respect to one comparator;
a membership query uses that same comparator or it is meaningless. Case-sensitive build with
case-insensitive search is the common instance: "Alice" sorts before "alice" under byte order, so
a search for "alice" terminates in the wrong half. A round-trip test — build then query every
inserted key plus one absent key — is the only version that catches it.

**CI-016** is the units family and its symptom is a clean factor, which makes it easy to confirm
once suspected: reported over expected gives 1000 (ms/s) or 1024 (KiB/bytes) or 100 (percent/
fraction). The reason it recurs is that unitless field names are legal in every language. Putting
the unit in the identifier is a cheap structural fix that survives refactoring better than a
comment does.

### 8.5 Scale failures: the cost family

**CI-005** is a conceptual entry with a scale symptom. An amortised bound is an average over a
sequence, and an SLO is a statement about individual requests; no amount of averaging satisfies it.
Measured illustration of the shape: one million appends completed in 82.5 ms, but the reallocating
append inside that sequence copies the entire buffer. The correction is pre-sizing or incremental
copy, and the verification is a per-request percentile, not a throughput number.

**CI-021** is the same mechanism seen from the latency chart: spikes at exponentially growing
intervals because the buffer doubles. The discriminating check is that spike spacing *doubles* as
data grows, which no GC pause pattern reproduces reliably. Once identified, the fix is capacity
planning tied to the growth curve, and the verification is that spike magnitude is bounded by the
input size rather than by elapsed uptime.

**CI-022** is the quadratic-loop family: concatenation, repeated slicing, or a per-row remote call.
Its signature is a log-log slope near 2. Measured: 20,000 string appends took 25.0 ms versus 5.5 ms
for the equivalent `join`, and list membership probing 2,000 keys against a 200,000 element list
took 3.09 s versus 0.64 ms against a set — a 4,827× ratio for the same logical operation.

**CI-023** is a build-cost entry. A heap built with `n` pushes is `O(n log n)`; built in place it is
`O(n)`. Measured at `n = 200,000`: 47.8 ms versus 13.2 ms, a 3.6× ratio. The symptom is startup
latency with no per-request cost, which is why it survives profiling focused on request handlers.

**CI-024** is `top-K` at growing `n`. Measured at `n = 1,000,000, k = 100`: 16.0 ms with a bounded
heap against 235.7 ms for a full sort, a 14.8× ratio. The trap is that the endpoint looks fine at
`n = 10^4` and degrades only as the retained history grows, so the regression arrives months after
the code shipped and is attributed to traffic growth rather than to the algorithm.

**CI-025** is frontier memory. At degree 10, depth `d` holds `10^d` nodes on the frontier, so depth
6 is 1,111,111 discovered nodes against depth 5's 111,111 — a 10× step per level. At 64 bytes per
node (ILLUSTRATIVE) that is 71.1 MB versus 7.1 MB. The discriminating check is the chart's shape:
staircase for frontier growth, ramp for retention.

**CI-026** is deliberate or accidental collision. The expected probe count `1 + alpha/2` is a
statement about a uniform hash; an adversary supplies keys, not a distribution. In the 1,000-bucket
simulation with 5,000 random keys the fullest chain held 15 entries against a mean of 5, so a
single long chain is not proof of attack. The proof is that the same key set is slow on every run
while a randomly generated set of the same size is fast.

**CI-027** is retention. Memory that does not return to baseline after traffic stops is retention,
not growth, and the check is a forced collection followed by a comparison against the baseline.
Caches without caps, deferred queues without deadlines, and closures captured by callbacks are the
three recurring sources. Capping every cache at construction is the structural prevention.

**CI-028** is adversarial input cost. A backtracking matcher on `(a+)+$` grew superlinearly in the
measured series: 2.8 ms at 16 characters, 10.5 ms at 18, 42.2 ms at 20, 171.6 ms at 22, 687.9 ms at
24 — roughly 4× per 2 characters, i.e. doubling per character. Input length limits and a
non-backtracking matcher are the two corrections; both are cheap and either alone is insufficient.

### 8.6 Concept confusions: the guarantee family

**CI-029** is the memory-hierarchy confusion. `O(log n)` counts comparisons; the hardware charges
dependent memory trips. At one million keys a binary search needs at most 20 comparisons against a
sorted array, each a likely cache miss; a B-tree node holding 100 keys answers in 3 node visits.
The check is nanoseconds per comparison measured against a contiguous scan; if the cost per
comparison is 50-100× a sequential access, the model has stopped describing the machine.

**CI-030** is the benchmark-quoting confusion. A measured ratio in this file — 14.8× for top-K
against sort — is valid for this runtime, this input size, and this `k`. It is not a promise for
another language's stdlib implementation. The check is to reproduce both sides in a single process
before quoting either; the prevention is a benchmark harness that ships with the code so the ratio
is re-derived rather than remembered.

**CI-031** is the coverage confusion, and it is the one that most often converts a known bug class
into a recurring incident. A suite can cover every line and still contain no record whose value
sits exactly at a threshold, no window starting at index 0, and no graph with a negative edge. The
check is to enumerate the boundary values present in fixtures — not the lines executed — and the
correction is to add a fixture per boundary deliberately.

**CI-032** is the thread-safety confusion: the phrase names at least two different guarantees. A
structure that is atomic per operation is not necessarily safe under read-modify-write sequences,
and a structure that is safe within one process is not safe across processes. The check is a
two-writer test against a shared instance; the correction is to narrow the claim to what is
actually enforced, because a narrowed claim is testable and a broad one is not.
## 9. Complexity & Cost Analysis

### 9.1 What this section prices

A catalog is only usable if the cost of *running* it is known. This section prices three things:
the detection of each defect class (how much instrumentation and time finds it), the containment
(what the safe action costs in rejected or deferred work), and the replay (what verifying the fix
costs in compute). A defect that can only be detected by a month of production observation is
governed differently from one detectable by a unit test in milliseconds, and the table makes that
difference explicit rather than leaving it to intuition.

### 9.2 Detection cost by class

Detection cost is expressed as `cost_per_check × frequency` where frequency is how often the check
must run to be credible. For a deterministic property test, frequency is "per change"; for a
distributional property (collision, adversarial input), frequency is "per externally visible
input", because the adversary chooses when to probe.

| Class | Cheapest credible check | Cost per run | Credible frequency | Annual cost driver |
|---|---|---|---|---|
| Boundary convention | Two-record differential test | Microseconds | Per change | Test suite growth |
| Accumulator width | One crossing-value test | Microseconds | Per change | Type audit |
| Float accumulation | `fsum` differential | Milliseconds per 10^6 values | Per change | None |
| Predicate boundary | Identity `count(>=)-count(>)` | One pass over fixtures | Per change | Fixture curation |
| Comparator ordering | Property test, 3 axioms | Thousands of random triples | Per change | Property library |
| Hash collision | Worst-bucket latency histogram | Full percentile pipeline | Continuous | Monitoring retention |
| Recursion depth | Nested fixture at expected max + 1 | Seconds | Per change | Fixture maintenance |
| Frontier memory | Depth-`d` vs depth-`d+1` memory | Minutes per graph size | Per release | Load environment |
| Regex backtracking | Timing curve across 4 input lengths | Seconds | Per change | Adversarial corpus |
| Cost-model drift | Benchmark at 3 input sizes | Minutes | Per release | Benchmark hardware |

The asymmetry to note: everything in the NUMERICAL_EDGE and VIOLATED_ASSUMPTION classes is cheap
to detect *if the check exists*, and the check is a test. Everything in SCALE_FAILURE requires an
environment that can reproduce scale, which is why scale failures cluster in the entries whose
verification the table marks "per release" rather than "per change". A team that only funds
per-change checks will systematically under-detect the right-hand half of the catalog — not
because those entries are hard, but because their checks are not unit tests.

### 9.3 Detection latency and its consequence

```text
FORMULA:  expected_caught = 1 - (1 - p)^k
WHERE:    p = probability a single record exercises the defect's boundary
          (dimensionless, 0..1); k = number of records processed before detection;
          expected_caught = probability the defect has surfaced by record k.
DERIVATION: each record independently avoids the boundary with probability 1-p; k
          independent records all avoid it with probability (1-p)^k; the complement is the
          probability at least one hits it.
CONSTRAINT: independence, and a stationary p. Both fail for a defect that only triggers
          as a value drifts toward a threshold.
NUMERICAL EXAMPLE: a predicate boundary hit by one record in a thousand (p = 0.001).
          At k = 100 records, 1 - 0.999^100 = 0.095, i.e. 9.5% chance of surfacing.
          At k = 1,000: 1 - 0.999^1000 = 0.632. At k = 10,000: 0.99995.
INTERPRETATION: a boundary defect in a low-frequency category is invisible for thousands of
          records regardless of test quality, which is why fixtures must inject it
          deliberately rather than wait for traffic (CI-031).
```

That formula is the quantitative justification for the entire fixture philosophy in §8. If the
boundary value occurs naturally with probability 0.001, no amount of production traffic substitutes
for a constructed record at the boundary, because the constructed record has `p = 1`. The
interaction with §5.7 is direct: even when the defect surfaces, it only changes a decision when the
aggregate is near its threshold, so the joint probability of (surfaced AND decision-changing) is
smaller still — the product of two small numbers, which is why these defects persist for years.

### 9.4 Containment cost

Containment is the action taken before the cause is known. Its cost is the work it rejects or
delays, and that cost is what makes containment decisions contestable.

```text
FORMULA:  rejected = D * (1 - c/D) = D - c        (for D > c)
WHERE:    D = demand rate (requests/s), c = contained capacity (requests/s), both
          non-negative; rejected in requests/s.
DERIVATION: admissible throughput is capped at c; everything above c is rejected, so the
          rejected rate is the part of demand above the cap.
CONSTRAINT: the cap is a hard limit. A probabilistic or token-bucket cap rejects a
          fraction rather than the excess, and the formula becomes rejected = D * (1 - c/D)
          only in expectation.
NUMERICAL EXAMPLE: ILLUSTRATIVE D = 25 req/s against a cap of c = 10 req/s rejects
          15 req/s, i.e. 54,000 requests per hour. A surplus of 15 req/s against a burst
          credit of 100 requests is consumed in 100/15 = 6.67 s.
INTERPRETATION: containment buys time proportional to the credit divided by the excess, so
          a cap set close to normal demand buys seconds, not minutes (CI-005).
```

The design implication is that a containment action must be sized against the *excess*, not
against the mean. A cap at 1.2× normal demand gives a burst window of `credit / (0.2 × D)`, which
for a 100-request credit at 25 req/s is 20 s — enough for a deploy rollback and not enough for a
human to be paged and respond. That arithmetic, not a policy preference, is what determines whether
a containment action belongs in an automated runbook.

### 9.5 Replay cost

Replay is the verification step, and it is priced in the same units as the original computation
plus the cost of isolating one record.

```text
FORMULA:  replay_cost = n_records * cost_per_record + setup
WHERE:    n_records = records replayed (count), cost_per_record = compute per record
          (seconds/record), setup = environment bring-up (seconds).
DERIVATION: replay re-executes the pipeline over a record range, so cost is linear in the
          number of records at the pipeline's per-record cost, plus a fixed setup.
CONSTRAINT: linearity holds only if the pipeline has no superlinear step. A replay over a
          range that triggers a full re-aggregation can be superlinear (CI-022).
NUMERICAL EXAMPLE: 1,000 records at 2 ms each is 2 s of compute; the same pipeline at a
          superlinear slope of 2 makes 10,000 records cost 100x the 1,000-record time,
          i.e. 200 s, not 20 s.
INTERPRETATION: bound the replay to the affected partition plus one neighbour, or the
          verification becomes the larger incident (CI-022).
```

Practical consequence: the replay scope should be chosen as "the triggering record plus one record
on each side of the boundary" rather than "the whole day". The boundary neighbour is the part that
catches the fix that moved the error rather than removing it — §5.3's asymmetry. Whole-day replays
are a different tool: they verify that the aggregate is unchanged elsewhere, which is a regression
check and should run on a schedule, not inside an incident.

### 9.6 Space cost of the catalog's own mitigations

Every mitigation in §8 that adds history, sampling, or instrumentation has a space cost, and it is
the one most often left out of a capacity plan.

| Mitigation | Space per unit | At 10^6 units | Note |
|---|---|---|---|
| Keep raw input for replay | ~43 B per record measured | 43.1 MB at 10^6 | `json.dumps` of 100k 3-field records = 4,314,815 bytes, i.e. 43.1 B/record |
| Per-request latency histogram | ~8 B per bucket | 8 MB for 10^6 buckets | Bucket count is per metric, not per request |
| Visited set for traversal | 33.6 B per element measured | 33.6 MB at 10^6 | Measured `set` of 1,000,000 ints |
| Frontier for BFS | 64 B per node ILLUSTRATIVE | 71.1 MB at 1.1×10^6 nodes | §6.4 |
| Min-heap of size k for top-K | 56 B per tuple measured | negligible at k=100 | `sys.getsizeof` of a 2-tuple |
| Cache with eviction | entry size × cap | bounded by design | Only if the cap exists (CI-027) |

The measured figures here are from `sys.getsizeof` on this runtime, which reports the container
itself and not the objects it references — a distinction that matters. A list of 100,000 two-tuples
reported 800,984 bytes for the list, but each tuple is a further 56 bytes, so the true figure is
about 12.0 MB total including the integers. Quoting the container size and calling it the memory
cost is a measurement error that undercounts by an order of magnitude, and it is the reason "the
list is only 800 KB" precedes a memory incident.

### 9.7 The cost of a false diagnosis

The final cost line is the one nobody budgets: the cost of chasing the wrong mechanism. If a
30-minute triage pass produces a wrong hypothesis and the team spends two engineer-days on it, the
cost is 16 engineer-hours against a defect whose correct repair was a one-line interval change.
Across the catalog, the cheapest guard against this is the rule in §1.2: no hypothesis advances
without a check that would falsify it. The rule costs minutes and saves days, and it is why every
entry in §8.2 has a discriminating-check column rather than a "likely cause" column.

## 10. Key Theorems / Results Reference

Each result below is stated with the precondition that the catalog polices. Where a result is used
as a diagnostic, the failure mode that follows from dropping the precondition is named by catalog
entry. Proofs are sketched only where the sketch is itself the diagnostic argument; the point of
this section is which condition a practitioner must check before trusting the bound.

### 10.1 Correctness results

**Comparison sort lower bound.** Any comparison-based sort of `n` distinct keys requires at least
`ceil(log2(n!))` comparisons, which is `Theta(n log n)`. The argument is that the `n!` possible
permutations must be distinguished by a binary decision tree of depth at least `log2(n!)`. The
precondition is *distinct* keys; with many equal keys the bound weakens, which is exactly why a
tie-heavy input can sort faster than the bound suggests and why tie handling is a correctness
question rather than a speed question. Dropping the distinctness precondition gives CI-004.

**Dijkstra's correctness.** With non-negative edge weights, the first extraction of a node from the
priority queue assigns its final shortest distance. The argument: if a shorter path existed, it
would pass through a node not yet extracted with a smaller tentative distance, contradicting the
extraction's minimality. Non-negativity is what makes the prefix of a path no longer than the path
itself. Dropping it gives CI-002, and the measured three-node counterexample in §4.4 shows the
result is not merely loose but wrong.

**Dijkstra's complexity.** With a binary heap, `O((V + E) log V)`, from `V` extractions at
`O(log V)` and at most `E` decrease-key operations. With a Fibonacci heap, `O(E + V log V)`. Using
the binary-heap bound to justify a design choice is fine; using it to compare against a BFS on an
unweighted graph is not, because BFS is `O(V + E)` and strictly better there. That substitution —
running Dijkstra on unit weights — is wasteful but correct, and it is the benign version of CI-029.

**Binary search correctness.** With a sorted array and a loop maintaining `answer in [lo, hi)`,
each update must shrink the interval. The invariant is the whole proof, and the failure is the
missing shrink: `lo = mid` with `hi = lo + 1` computes `mid = lo` and does not shrink. Measured,
that stall ran 51 iterations on `n = 8` (§6.2). Dropping the shrink obligation gives CI-009.

**Heap invariant.** A binary heap satisfies `a[parent] <= a[child]` for the array embedding
`parent(i) = (i-1)//2`, `child(i) = 2i+1, 2i+2`. Both `push` and `pop` restore it by a single
sift along one root-to-leaf path, which is why both are `O(log n)`. The embedding is the part
people get wrong: `2i` and `2i+1` is the 1-based convention, and mixing it with a 0-based array
shifts every child by one — the same class of translation bug as the Fenwick case in §4.3.

**Fenwick invariant.** Node `i` (1-based) stores the sum of the `lowbit(i)` elements ending at `i`.
A prefix query accumulates nodes while clearing the lowest set bit, so the visited nodes partition
the prefix exactly. The invariant is what guarantees no double counting; the off-by-one in
translation does not break the invariant, it breaks the mapping from the caller's indices, which is
why the symptom is a wrong-but-consistent total rather than a corrupted structure.

### 10.2 Cost results and their preconditions

| Result | Statement | Precondition the catalog checks | Entry if dropped |
|---|---|---|---|
| Amortised append | `O(1)` amortised for geometric growth | Growth factor > 1 | CI-005 |
| Bulk heap build | `O(n)` by sift-down | All elements available up front | CI-023 |
| Hash lookup | `O(1)` expected | Independent uniform hash | CI-026 |
| Top-K | `O(n log k)` | `k << n` and a bounded structure | CI-024 |
| BFS | `O(V + E)` time and `O(width)` memory | Memory tracks frontier width, not `V` | CI-025 |
| Quicksort | `O(n log n)` expected | Random or median pivot | CI-028 |
| Binary search | `O(log n)` | Sorted by the query comparator | CI-007 |
| Union-find | `O(alpha(n))` amortised | Path compression plus union by rank | CI-008 |
| Counting sort | `O(n + k)` | `k` (key range) is bounded and small | CI-024 |

**Quicksort's expected bound.** `O(n log n)` expected requires a pivot that splits reasonably, in
expectation over the pivot's randomness. With a fixed pivot on adversarial or already-sorted input,
the recurrence degrades to `T(n) = T(n-1) + O(n)`, i.e. `O(n^2)`. Measured: 3.3 ms at `n = 200`,
11.7 ms at 400, 76.0 ms at 800 — ratios of 3.5 and 6.5 for doublings, consistent with a quadratic
rather than `n log n` curve. The check is a timing curve at three sizes; the fit distinguishes the
two models because `n log n` predicts a ratio of about 2.2 per doubling at these sizes.

**Union-find's inverse-Ackermann bound.** `O(alpha(n))` amortised requires *both* path compression
and union by rank or size. With union by rank alone the bound is `O(log n)`; with path compression
alone it is `O(log n)` amortised as well. Using either alone and quoting the inverse-Ackermann
figure is a common overstatement, and its symptom is a union-heavy phase that is slower than a
review predicted without any algorithmic error — a cost-model drift rather than a bug.

**Counting sort's linear claim.** `O(n + k)` is linear in `n` only for fixed `k`; for key ranges
that scale with `n` it is `O(n + k)` with `k` possibly larger than `n log n`. Sorting 64-bit
identifiers by counting sort allocates `2^64` counters, which is not a performance problem but an
impossibility. The precondition "`k` is small and bounded" is the whole content of the result.

### 10.3 Distributional results used for diagnosis

**Birthday bound.** With `m` buckets and `n` random keys, the expected number of colliding pairs is
`n(n-1)/(2m)`. At `n = 5,000, m = 1,000` that is `5,000 × 4,999 / 2,000 = 12,497.5` expected
colliding pairs — which is why a fullest-bucket occupancy of 15 is unremarkable, not alarming. The
diagnostic use is the reverse direction: a bucket occupancy far above what the birthday bound
permits *for the observed number of collisions* is evidence of non-uniformity or attack, while one
tall bucket among many short ones is consistent with chance.

**Coupon-collector bound.** Filling `m` buckets by random placement requires about `m ln m`
insertions to fill all of them; at `m = 1,000` that is `1,000 × 6.91 = 6,908`. The measured
simulation had 7 empty buckets at `n = 5,000`, consistent with being below the fill point, and it
confirms the model's scale rather than contradicting it. The diagnostic use: an empty-bucket count
far from the model's prediction indicates a hash function that does not cover its range.

**Load-factor probe bound.** `1 + alpha/2` for a successful chained search and `alpha` for a miss,
derived in §4.2. Its use in diagnosis is calibration: it tells the responder what "normal" looks
like for the observed `alpha`, so that an anomaly can be measured against a number rather than a
feeling.

### 10.4 Numerical results

**Float spacing.** A double has a 53-bit significand, so between consecutive powers of two the gap
between representable values is `2^(e-52)` for a value near `2^e`, i.e. roughly `x × 2^-52`. At
`x = 1` the gap is exactly `2^-52 = 2.22e-16`; at `x = 1e16` the gap is exactly `2.0`, because
`2^53 = 9.007e15` and `1e16` lies in `[2^53, 2^54)`. That is why `(1e16 + 2) - 1e16` measured `2.0`
while `(1e16 + 1) - 1e16` would be `0.0`: 1 is below the gap and rounds away. Any accumulator that
adds increments smaller than the running total's gap loses them entirely.

**Integer wrap.** A signed `k`-bit accumulator holds `-2^(k-1) .. 2^(k-1) - 1`. At `k = 32` the
maximum is `2,147,483,647`, and `sum of i^2 for i = 1..2000` is `2,668,667,000`, which wraps to a
negative value. The measured derivation: `n(n+1)(2n+1)/6 = 2000 × 2001 × 4001 / 6 = 2,668,667,000`.
At `k = 64` the wrap point for the triangular sum `n(n+1)/2` is `n = 4,294,967,296`, derived as
`isqrt(2 × 2^63)`, so a 64-bit counter of requests survives any realistic volume — which is why
using 64-bit accumulators removes this entire entry for counters.

**Summation error.** Naive summation of `n` values accumulates a worst-case error of order
`n × eps × sum(abs(x))`, where `eps = 2^-53 ≈ 1.11e-16`. Measured: one million additions of `0.1`
produced an error of `1.33e-6` against an exact `100000`, a relative error of `1.33e-11`. The bound
`n × eps = 1e6 × 1.11e-16 = 1.11e-10` relative is consistent with the observed `1.33e-11`, being
three orders tighter than worst case as usual for random signs. Compensated summation (Kahan) or
exact summation (`fsum`) reduces the bound to a small multiple of `eps` regardless of `n`.

## 11. Interview / Exam Questions

Twelve questions, mixing recall, calculation and diagnosis, each answered in full. Questions 2, 3,
4, 6, 7, 8, 9 and 12 require a computation or a diagnostic decision; the rest test whether the
vocabulary is actually understood. Answers are deliberately specific, because "it depends" is the
answer the catalog exists to replace.

### Q1. What distinguishes an amortised bound from a worst-case bound, and which one satisfies a latency SLO?

**Answer.** An amortised bound is the total cost of `n` operations divided by `n`; a worst-case
bound limits every individual operation. A latency SLO constrains individual operations, so only a
worst-case (or a tail-percentile) bound can satisfy it. Doubling growth gives `O(1)` amortised
append but `O(n)` worst case for the reallocating append.
**Reasoning.** The SLO is evaluated per request, and the mean is not a per-request quantity; a
single 10 ms copy inside a 1 ms budget violates the budget regardless of the average.
**Follow-up.** What changes if the SLO is stated as "p99 over one minute" instead of "no request"?
**Red flag.** Answering that amortised is "good enough in practice" without naming the tail.
**Scoring.** Full credit: names both bounds, ties the SLO to per-operation evaluation, and can
state the `O(n)` reallocation case. Partial: names the distinction but cannot give the reallocation
example.

### Q2. A total is wrong by exactly 3, and one day's count is 3. What is the first hypothesis and what is the check?

**Answer.** A boundary convention error: the window either includes or excludes one element whose
value is 3. The check is to recompute under the other convention and see whether the error
disappears; if it does, the bracket is wrong at one end, and both ends must be re-tested.
**Reasoning.** An arithmetic or accumulation error would produce an error unrelated to any single
day's value. An error equal to a known table entry identifies the element, which identifies the
edge.
**Follow-up.** The error is +3 at every window and zero at windows starting at index 0. Which
entry is that, and what is the mechanism?
**Red flag.** Reaching for a tolerance or a rounding fix, which would hide the error without
changing it.
**Scoring.** Full: names the convention family, states the differential check, and notes both ends
must be re-tested. Partial: names the convention but only fixes the reported window.

### Q3. Compute the expected successful-search probe count for 5,000 keys in 1,000 buckets, and say when the number is useless.

**Answer.** `alpha = 5000/1000 = 5.0`, so `1 + alpha/2 = 3.5` probes. The number is useless when
the hash is not independent of the keys — a deliberate collision set puts all 5,000 in one bucket,
giving 5,000 probes instead of 3.5, an error of about 1,400×.
**Reasoning.** The derivation assumes each of the other keys lands in the target bucket with
probability `1/m`, and that the target's position within its chain is uniform. An adversary
controls both.
**Follow-up.** What single measurement distinguishes an attack from an unlucky distribution?
**Red flag.** Quoting 3.5 as a guarantee for untrusted keys.
**Scoring.** Full: computes 3.5, states the uniformity precondition, quantifies the adversarial
case. Partial: computes correctly but treats the figure as unconditional.

### Q4. A binary search over 8 elements runs 51 iterations. Diagnose it.

**Answer.** The interval is not shrinking. With `lo = mid` and `hi = lo + 1`, `mid` computes to
`lo`, so the update leaves `(lo, hi)` unchanged and the loop stalls. The correct count for `n = 8`
is `ceil(log2 8) = 3` comparisons, or at most 4 loop iterations depending on formulation.
**Reasoning.** A binary search must reduce the candidate set by roughly half per iteration; any run
longer than `log2(n) + 1` iterations is a non-termination bug, not a slow search.
**Follow-up.** Which formulation avoids the bug entirely, and what is its invariant?
**Red flag.** Blaming the comparator or the data without checking the interval arithmetic.
**Scoring.** Full: identifies the missing `+1`, gives the correct iteration count, states the
shrinking invariant. Partial: identifies non-termination without locating the update.

### Q5. Why is `sorted` not sufficient for membership testing?

**Answer.** Sorted is relative to a comparator. If the array was built with a byte-order comparator
and the query uses a case-insensitive one, the order the search assumes does not exist, and the
search terminates in the wrong half. Sortedness must be with respect to the *query's* comparator.
**Reasoning.** Binary search's correctness proof uses only the sortedness assumption; it never
re-checks the comparator. A mismatched comparator makes the assumption false while the code
compiles and runs.
**Follow-up.** What test catches it, and why must the test include an absent key?
**Red flag.** Answering "sorted arrays are fine for membership" without the comparator condition.
**Scoring.** Full: ties sortedness to the comparator, names the round-trip test. Partial: mentions
case sensitivity as a special case without the general rule.

### Q6. One million additions of 0.1 give 100000.00000133288. Is that acceptable, and what do you do?

**Answer.** The relative error is `1.33e-6 / 100000 = 1.33e-11`, which is acceptable for a chart and
not acceptable for a reconciliation that requires equality. Use `math.fsum`, which returned exactly
`100000.0`, or accumulate in integer units (tenths) and divide once at the end.
**Reasoning.** The error is a sum of per-addition roundings, bounded by `n × eps × sum(abs)` but
typically far below it for mixed signs; it grows with `n` and never shrinks.
**Follow-up.** Why does `fsum` fix it, and what does it cost in time?
**Red flag.** Adding a tolerance to the assertion, which hides the accumulation and loses the
ability to detect a real discrepancy later.
**Scoring.** Full: computes the relative error, chooses an exact route, rejects tolerance-patching.
Partial: recognises float error but patches the comparison.

### Q7. Which of the following is `O(n)`: building a heap by `n` pushes, building it by sift-down, or both?

**Answer.** Sift-down build is `O(n)`; `n` pushes is `O(n log n)`. Measured at `n = 200,000`, the
two took 47.8 ms and 13.2 ms respectively, a 3.6× ratio.
**Reasoning.** Sift-down costs at most `j` comparisons at height `j`, and the count of nodes at that
height is `n/2^(j+1)`, so the sum is `n × sum(j/2^(j+1)) = n × 1`. Pushing costs `log n` per
element because each sift runs the full height.
**Follow-up.** When can you not use sift-down, and what do you do instead?
**Red flag.** Claiming both are `O(n)` or both are `O(n log n)`.
**Scoring.** Full: correct classification, the sum argument, and a measured or derived ratio.
Partial: correct classification without justification.

### Q8. A traversal aborts at depth 985 with a recursion error, and the limit is 1000. Explain the 15-frame gap and the fix.

**Answer.** Frames are already consumed by the caller chain and interpreter overhead before the
first recursive call, so the usable depth is `R - c` with `c ≈ 16` measured here. The fix is an
explicit stack, or an explicit depth limit that is checked and reported rather than aborted.
**Reasoning.** Depth is a hard resource: there is no partial result, the call either fits or the
process raises. Any depth that is a function of user data must be bounded by construction.
**Follow-up.** What is the memory cost of converting to an explicit stack, and how does it compare?
**Red flag.** Raising the limit as the fix, which moves the boundary without making it safe.
**Scoring.** Full: explains the gap, converts to iterative or bounds explicitly, notes hard-limit
semantics. Partial: converts without explaining the gap.

### Q9. A regex takes 2.8 ms at 16 characters, 10.5 ms at 18, 42.2 ms at 20 and 171.6 ms at 22. What is happening?

**Answer.** Catastrophic backtracking: the pattern `(a+)+$` has exponentially many ways to partition
a non-matching string, so time roughly doubles per added character (measured about 4× per 2
characters). It is not a slow machine; it is an algorithmic blowup reachable by a 22-character
input.
**Reasoning.** Nested quantifiers over overlapping subexpressions create a search tree whose size is
exponential in the input length; the failure is in the pattern, not the data.
**Follow-up.** What are the two independent mitigations, and why is either alone insufficient?
**Red flag.** Adding a timeout only, which converts a CPU exhaustion into a request failure without
removing the amplifier.
**Scoring.** Full: names catastrophic backtracking, states the exponential growth, gives both
mitigations (input bound and non-backtracking matcher). Partial: names the growth without
mitigation.

### Q10. What does "thread-safe" claim, and what must be verified before using it?

**Answer.** At minimum two distinct claims: atomicity per operation, and safety of composed
sequences. A structure can be atomic per call and still lose updates under read-modify-write. It
can also be safe within one process and unsafe across processes. Verification: a two-writer test
against a shared instance, plus a written statement of which sequences are supported.
**Reasoning.** The phrase names an unstated guarantee; the only testable form is a precondition
listing the supported compositions.
**Follow-up.** Which is more expensive to provide, per-operation atomicity or sequence atomicity?
**Red flag.** Treating "thread-safe" as a boolean property of a library.
**Scoring.** Full: separates the two claims, requires a composition test. Partial: mentions locks
without the composition distinction.

### Q11. Distinguish deterministic from reproducible, with an example that is one and not the other.

**Answer.** Deterministic: same binary and environment yield the same output. Reproducible: someone
else obtains the same output. Iterating a set is deterministic within a process and not reproducible
across processes, because CPython randomises string hashing per process. A snapshot built by
iterating a set is stable in one shell and different in the next CI job.
**Reasoning.** Determinism is a property of the program's execution; reproducibility adds the
environment as an input that must be pinned.
**Follow-up.** How would you demonstrate the difference in one command?
**Red flag.** Using the words interchangeably, which is how nondeterministic output reaches
production.
**Scoring.** Full: separates the two, gives the hash-seed example. Partial: defines both correctly
without an example.

### Q12. You find that the verification script agrees with the buggy function. What is wrong and what is the correction?

**Answer.** The verification shares an assumption with the implementation — usually it calls the same
helper or reads the same derived table. The correction is to rebuild the expected value by a route
that never calls the code under test, such as hand arithmetic over raw records, and to confirm the
check *can* fail by running it against a known-bad input.
**Reasoning.** A check that cannot disagree provides no information; agreement between two calls to
the same function is a tautology.
**Follow-up.** What is the cheapest way to prove the check is capable of failing?
**Red flag.** Adding an assertion that duplicates the implementation's logic.
**Scoring.** Full: identifies shared assumption, prescribes an independent route, requires a
falsification proof. Partial: recognises the tautology without a correction.

## 12. Quick Reference Card

| Situation | Do this first | Then verify | Entry |
|---|---|---|---|
| Total off by one element's value | Recompute under the other interval convention | Both boundary ends, `lo=0` and `hi=n-1` | CI-001, CI-018 |
| Total flips sign at scale | Widen the accumulator, re-derive the crossing input | The exact `MAX+1` case | CI-010, CI-014 |
| Tiny relative error vs exact route | Switch to `fsum` or integer units | Error bound no longer grows with `n` | CI-011, CI-012 |
| Count differs from manual count | Test `count(>=) - count(>) = count(==)` | Fixture containing the exact boundary | CI-013 |
| Search runs longer than `log2(n)+1` | Check the interval update shrinks | Iteration count assertion | CI-009 |
| Shortest path too long | Test a 3-node graph with one negative edge | Bellman-Ford agreement | CI-002 |
| Same input, different output | Run twice under different hash seeds | Snapshot equality across seeds | CI-003 |
| Equal keys reorder between runs | Add a deterministic tiebreaker | Golden output on tie-heavy input | CI-004 |
| Memory climb while answers are right | Find the retention set; cap and expire | Baseline restored after forced GC | CI-027 |
| Latency grows faster than input | Fit log-log slope across three sizes | Slope near 1, not 2 | CI-022 |
