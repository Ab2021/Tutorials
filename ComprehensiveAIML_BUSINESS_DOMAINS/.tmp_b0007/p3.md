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
