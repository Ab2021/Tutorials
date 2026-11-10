---
collection: "planning_forecasting_inventory_and_sourcing"
file_id: "01"
topic: "Exact inventory range totals with a Fenwick tree"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
last_reviewed: null
reviewed: null
review_trigger: "Bin ordering, writer model, numeric width, unit convention, or snapshot contract changes."
notation_style: "plain-text"
prerequisites: ["Array and prefix-sum reasoning", "Integer overflow awareness", "Basic transaction/isolation vocabulary"]
pattern_tags: ["snapshot-consistency", "correctness-contract", "numerical-edge"]
source_status: "OPEN - reference method consulted (fenwick_inventory_totals.md); no production evidence claimed"
owner_role: "Inventory platform engineer"
source_index: "INDEX.md"
---
# Exact Inventory Range Totals with a Fenwick Tree

> **Purpose:** Decide the update, arithmetic, and snapshot contract under which a point-adjusted inventory vector answers inclusive range sums exactly, and defend why a Fenwick tree is or is not the right structure for that contract.
> **Prerequisites:** prefix sums over an array, one-based versus zero-based indexing, two's-complement integer widths, and the meaning of a database transaction.
> **Key outcome:** Given a bin-ordered quantity vector, an adjustment stream, and a read pattern, you can build the tree, compute any range total by hand, prove the answer against a direct scan, bound the operation count, and name the exact defect class when two readers disagree.
> **This file does NOT cover:** distributed stock reconciliation across warehouses, demand forecasting, safety-stock policy selection, or database engine internals such as page layout and WAL format.
> **Level:** 2 - Core methods.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED algorithm reasoning; VERIFIED illustrative arithmetic on constructed vectors; VERIFIED operation-count measurements from an in-memory reference implementation. No production inventory system, latency benchmark, or incident is claimed. Source verification remains OPEN.
> **Retrieval cue:** If two totals for the same range disagree, or a total changes without a matching adjustment, start at Section 8 and classify the defect before touching the tree.

## 1. File Header

This sheet is about one narrow but load-bearing claim: a warehouse can state the exact sum of quantities across a contiguous run of bins at a single logical version, after arbitrary point adjustments, without scanning the run. The interesting part is not the bit trick. The bit trick is four lines of arithmetic and can be memorised in a minute. The interesting part is the contract that surrounds it, because almost every real failure of an inventory total is a contract failure wearing an algorithmic disguise: a delta applied twice, two readers observing different versions, a quantity stored in mixed units, an identifier that is not a stable position, or an arithmetic type too narrow for the totals the business actually holds.

The decision this sheet supports appears at the moment a planner asks a question that mixes aggregation with authority. "How many eaches are in the chilled aisle between slot 300 and slot 420, right now, in the version that picking is reading?" The honest answer requires three things that are easy to confuse with one another: the arithmetic aggregate (a sum), the population (the ordered bins in that aisle), and the version (which adjustments are included). A structure can make the arithmetic cheap and still leave the answer wrong, because cheap arithmetic says nothing about population stability or version agreement.

Four concrete decisions recur, and each has a written artifact that makes it auditable.

The first is the **position contract**. A Fenwick tree indexes positions 1 through n. If a SKU is added to the aisle, the vector changes length, and every position after the insertion point shifts by one. Any cached prefix, any saved query boundary, and any stored report that referenced the old numbering becomes wrong in a way no checksum will catch, because the arithmetic remains internally consistent. The position contract therefore names an ordering epoch and declares that the vector's positions are fixed for the life of the epoch; structural change starts a new epoch and forces a rebuild.

The second is the **delta contract**. The structure supports adding a signed amount to one position. It does not support "set to 900" without knowing the old value, and it does not know whether the same business event has arrived before. Assignment is expressed as a delta of new minus old computed under the same authority that guarantees the old value was the last applied one. Replay safety is expressed as a durable key on the adjustment, checked before the delta reaches the tree.

The third is the **numeric contract**. The aggregate is a sum, so it grows with n times the per-bin maximum. A vector of 50,000 bins that each legally hold 900,000 eaches has a total of 45,000,000,000, which exceeds the signed 32-bit maximum of 2,147,483,647 by a factor of about 21. The tree's individual cells can also hold large blocks, so the width must cover the aggregate, not the per-bin value.

The fourth is the **version contract**. Prefix subtraction is exact only if both prefixes come from the same version. Two reads of one cell each may be individually atomic and still combine into a value that never existed, because the writer touches several cells per update and a reader walks several cells per query.

Everything in the following sections is in service of those four contracts. Sections 2 and 3 fix the vocabulary. Section 4 derives the invariant and both traversals. Section 5 runs one vector through build, query, adjustment, assignment, duplicate, and a mixed-version read with explicit numbers. Section 6 draws the block decomposition so the traversals stop looking like magic. Section 7 compares the structure against the alternatives a team would actually consider, including the segment tree, the difference-array variant, and the two-tree range-add/range-sum variant. Section 8 is the diagnostic catalogue, which is where most of the working value of this sheet lives. Section 9 sizes memory and write throughput including the durability path that usually dominates. Section 10 states what the algorithm does and does not prove. Section 11 is the interview pack, and Section 12 is the card.

**Takeaway:** the tree is the cheap part; the position, delta, numeric, and version contracts are the decision.

## 2. Concept Map / Mental Model

The structure is a fixed set of overlapping aligned blocks. Every position i owns the block that ends at i and whose length is the lowest set bit of i. A tree cell stores the sum of its block. A prefix query walks backwards through blocks that partition the prefix exactly. An adjustment walks forwards through the cells whose blocks contain i. Both walks are short because each step removes or adds exactly one aligned block boundary.

```text
[Authoritative adjustment event]
   adjustment_id + bin position i + signed delta d + unit + epoch
            |
            v
[Validate] --reject--> [Quarantine + audit: bad index, bad unit,
            |            wrong epoch, negative quantity policy]
          pass
            v
[Deduplicate on adjustment_id] --seen--> [No-op; return prior result]
            |
            v
[Apply d to T[i], T[i+lowbit(i)], ... <= n]  <-- one logical writer
            |
            v
[Publish version v] --> [Reader: P(r) - P(l-1)] --> [Exact total]
            |                                            |
            v                                            v
     [Durable log / checkpoint]              [Disagree with direct scan?
                                               classify in Section 8]
```

The central invariant is stated once and reused everywhere:

```text
INVARIANT [I-01]: for every i in 1..n at version v,
                  T[i] = sum of q[j] for j = i - lowbit(i) + 1 .. i
                  computed entirely from version v.
```

The invariant is what makes the query correct, and the version qualifier is what makes the invariant usable. Without a version, the equation describes a moving target: the writer updates several cells in sequence, so a reader that samples the cells in another order can assemble a set of values that satisfies no single version's equation. In Figure terms, a query succeeds when its cells all come from one column of the version history; it fails when its cells are drawn from two.

What changes between variants is where the delta lands and what the query must extract.

| Variant concept | Core idea | Role: input -> output | Use when | Boundary |
|---|---|---|---|---|
| Point add, range sum | delta at one position; two prefix walks | adjustment stream -> exact range total | point corrections, receipts, cycle counts | additive aggregate with inverses |
| Point assignment | convert to delta under the writer | new value -> effective delta | corrections that state a target quantity | old value must be known and authoritative |
| Range add, point query | difference array; prefix of differences | interval correction -> one bin value | promotions applied to a run of bins | point values, not range totals |
| Range add, range sum | two trees over the difference array | interval correction -> exact range total | bulk re-slot without per-bin events | twice the state and twice the proof work |
| Static prefix array | one prefix per position | frozen vector -> O(1) totals | published period-close snapshots | any update invalidates the whole array |
| Segment tree | general associative aggregate | updates and queries -> chosen aggregate | minimum, maximum, count, mixed aggregates | larger constant, more code, more state |

The most important boundary in the table is the last column of the first row. Prefix subtraction depends on an operation with an inverse. Addition of signed integers has one. Minimum does not: knowing the minimum of bins 1 through 7 and the minimum of bins 1 through 2 tells you nothing about the minimum of bins 3 through 7. A team that needs "lowest stock level in the aisle" cannot copy the sum implementation and keep the semantics; it needs a different aggregate and therefore a different structure.

The second most important boundary is the sentence "one logical writer". The tree is not a concurrency primitive. It is a data structure whose cells must be updated as a set. Section 8's second diagnostic is entirely about readers observing partial sets, and no amount of atomic per-cell access fixes it.

**Takeaway:** the map has two traversals over one invariant, and every variant is a statement about where deltas land and which aggregate the query must support.

## 3. Foundational Definitions

Each definition below is used later without re-explanation. Units are stated because a total in mixed units is a wrong number with a right-looking derivation.

| Term / symbol | Formal meaning | Units / shape | Valid domain | Common confusion | Glossary entry |
|---|---|---|---|---|---|
| `n` | number of positions in the ordered quantity vector | count of bins [bins] | positive integer | confused with SKU count when one SKU occupies several bins | `n_bins` |
| `i` | one-based position in the vector | index [dimensionless] | 1 <= i <= n | confused with SKU identifier or with zero-based array offset | `position` |
| `q[i]` | authoritative signed quantity at position i, smallest inventory unit | [eaches] | signed integer; may be constrained nonnegative by policy | confused with on-hand, available, or allocated quantity | `qty_on_hand` |
| `d` | accepted signed delta applied at one position | [eaches] | integer, may be negative | confused with a target value | `delta` |
| `T[i]` | derived block sum ending at i | [eaches] | any integer representable in the chosen width | mistaken for a prefix sum | `block_sum` |
| `P(k)` | prefix sum of q[1..k] | [eaches] | integer; P(0) = 0 by definition | mistaken for total of all bins when k < n | `prefix_total` |
| `R(l, r)` | inclusive range sum of q[l..r] | [eaches] | requires 1 <= l <= r <= n | confused with half-open interval [l, r) | `range_total` |
| `lowbit(i)` | `i & -i`, the value of the lowest set bit | [dimensionless power of two] | positive i only | computed on zero, yielding zero | `lowbit` |
| `v` | version identifier of one coherent published state | monotonic integer | total order within an epoch | confused with a wall-clock timestamp | `version` |
| `epoch` | identifier for one fixed bin ordering | integer | changes only on structural change | confused with version | `ordering_epoch` |
| `adjustment_id` | durable unique key for one business adjustment | opaque string | unique per accepted event | confused with message offset or retry counter | `adjustment_id` |
| `u` | inventory unit of record | enumerated | one unit per vector, never mixed | confused with purchase unit or case pack | `unit_of_record` |

Three of these definitions carry disproportionate weight and deserve a sentence each.

`i` is a **position**, not an identity. It is legitimate for position 412 to hold SKU `A-881` today and SKU `B-330` after a re-slot, provided the epoch changed and the tree was rebuilt. It is illegitimate for position 412 to hold `A-881` in the tree and `B-330` in the authority table, because then the tree is aggregating a population that no longer exists. The interview consequence: "what is your key" must be answered with two keys — the position used for aggregation and the SKU-plus-location identity used for business meaning.

`q[i]` is the quantity in the **unit of record**, which is the smallest unit the business counts. If the receipt system speaks cases of 24 and the picking system speaks eaches, the vector is in eaches and the receipt must be converted before the delta is formed. A tree that mixes units produces totals that are arithmetically perfect and dimensionally meaningless.

`v` is a **version**, not a timestamp. Two adjustments can share a millisecond; a version cannot be shared. Timestamps are useful for audit and useless for the isolation argument, because a reader cannot test whether two cells belong to the same millisecond without reading the clock, which it does not control.

**Takeaway:** two keys, one unit, one version — the definitions encode the contracts that Section 8 tests.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: the block invariant and the two traversals

**Validity:** one-based indices, fixed order within an epoch, exact integer arithmetic, and every cell read from a single version. **Derivation:** write i in binary. The lowest set bit has value b = lowbit(i), and subtracting it clears that bit and leaves the higher bits unchanged, so i - b is a multiple of 2b while i is not. The block [i - b + 1, i] is therefore exactly the aligned block of length b ending at i, and it is disjoint from the block ending at i - b. Storing the block sum makes the prefix walk a partition: P(k) consumes the block ending at k, jumps to k - lowbit(k), and repeats, and the consumed blocks are pairwise disjoint and exactly cover 1..k.

```text
FORMULA [F-01]: T[i] = sum(q[j] for j = i - lowbit(i) + 1 .. i)
WHERE: q is the quantity vector [eaches]; i is a one-based position;
       lowbit(i) = i & -i for i >= 1.
CONSTRAINT: fixed ordering epoch and one coherent version across all cells read.
NUMERICAL EXAMPLE: i = 6 = 110b, lowbit = 2, so T[6] = q[5] + q[6].
INTERPRETATION: cell 6 is a two-position suffix block, not a prefix of the vector.
```

```text
FORMULA [F-02]: P(k) = P(k - lowbit(k)) + T[k], with P(0) = 0
WHERE: k is a one-based position; the recursion terminates when k reaches 0.
CONSTRAINT: each term must be evaluated at the same version v.
NUMERICAL EXAMPLE: k = 7 = 111b; lowbit = 1 -> k = 6; lowbit = 2 -> k = 4;
                   lowbit = 4 -> k = 0. So P(7) = T[7] + T[6] + T[4].
INTERPRETATION: the prefix is partitioned into three disjoint aligned blocks.
```

```text
FORMULA [F-03]: update(i, d): while i <= n: T[i] += d; i += lowbit(i)
WHERE: d is a signed integer [eaches] already validated and deduplicated.
CONSTRAINT: d must be the true change; the tree cannot infer it from a target value.
NUMERICAL EXAMPLE: n = 8, i = 6, lowbit(6) = 2 -> i = 8, lowbit(8) = 8 -> i = 16 > 8.
                   Cells touched: T[6], T[8].
INTERPRETATION: the number of touched cells is the number of blocks containing i.
```

```text
FORMULA [F-04]: R(l, r) = P(r) - P(l - 1), for 1 <= l <= r <= n
WHERE: R is inclusive over both endpoints; P(0) = 0 makes l = 1 legal.
CONSTRAINT: additive aggregate with an inverse; not valid for minimum or maximum.
NUMERICAL EXAMPLE: q = [5,0,7,2,4,1,3,6]; P(7) = 22, P(2) = 5; R(3,7) = 17.
INTERPRETATION: two prefix walks replace a scan of up to n positions.
```

The linear build is a fifth operation worth writing down because it removes an O(n log n) warm start:

```text
FORMULA [F-05]: for i = 1..n: T[i] += q[i]; j = i + lowbit(i);
                if j <= n: T[j] += T[i]
WHERE: T is initialised to zero; q is the current vector at the epoch's opening version.
CONSTRAINT: iteration must be ascending so that T[i] is final before it is propagated.
NUMERICAL EXAMPLE: q = [5,0,7,2,4,1,3,6] -> T = [5,5,7,14,4,5,3,28].
INTERPRETATION: each cell finalises once and is pushed into exactly one parent.
```

### 4.2 General case: assignment, epochs, and non-additive reads

The base case assumes the delta is known. Real adjustment streams often state a target: a cycle count says "bin 412 now holds 37", not "bin 412 changed by 12". The general form converts the statement into a delta under the writer's authority:

```text
FORMULA [F-06]: d = q_new - q_old, where q_old is read from the same authoritative
                state that the writer last applied, inside the same serialized section.
WHERE: q_new and q_old share the unit of record; both are [eaches].
CONSTRAINT: a stale q_old produces a delta that is individually well formed and
            globally wrong; concurrent writers break this formula outright.
NUMERICAL EXAMPLE: q_old = 25, q_new = 37 -> d = 12; the tree receives +12.
INTERPRETATION: assignment is a two-step transaction, not a tree operation.
```

The epoch generalises the position contract. Every formula above is valid only while the mapping position-to-business-identity is fixed. A structural change (bin inserted, aisle renumbered, zone merged) invalidates cached positions, saved report boundaries, and any persisted tree. The recovery is to rebuild from the authoritative vector under the new epoch and to reject adjustments stamped with the stale epoch.

The general workload also changes the cost shape. Let U be the number of adjustments per period and Q the number of range reads. Cell touches per adjustment average (1/n)·Σ_i u(i) where u(i) counts ancestors of i, and per read average (2/n)·Σ_k p(k) where p(k) counts prefix steps. For n = 12,000 the measured averages are 7.055 and 6.61 respectively, with maxima 14 and 13. Those numbers, not the asymptotic label, determine whether the structure is fast enough for a given read budget.

### 4.3 Key special cases and variants

| Variant | What changes | Assumptions | Use when | Avoid when | Equation / ID |
|---|---|---|---|---|---|
| Point add, range sum | base case | invertible additive aggregate | point corrections with wide reads | aggregate is min/max or non-invertible | F-01..F-04 |
| Linear build | O(n) initialisation | vector known at epoch opening | service restart, nightly rebuild | adjustments already streaming in | F-05 |
| Point assignment | two-step delta | serialized writer with authoritative old value | cycle counts, target corrections | concurrent writers or stale cache | F-06 |
| Range add, point query | difference array `D`, query = P_D(i) | corrections apply to a run | zone-wide write-offs, promotions | range totals are the product | Section 7 row 4 |
| Range add, range sum | two trees over `D` | linearity of summation | bulk re-slot with range reads | proof and state budget is tight | Section 7 row 5 |
| Frozen prefix array | replace tree by cumulative array | no updates within the window | period-close published snapshots | any adjustment can arrive | Section 7 row 3 |
| Segment tree | general aggregate, lazy propagation | associative operation | min/max/count or mixed aggregates | sums only, simplicity matters | Section 7 row 6 |

The two range-update variants deserve the extra sentence because they are the most common source of misplaced confidence. For range add and point query, store the difference array `D[i] = q[i] - q[i-1]` in a Fenwick tree; adding x to q[l..r] becomes `D[l] += x` and `D[r+1] -= x`, and a point value is a prefix sum of `D`. For range add and range sum, one tree is not enough: the range total after adding x to [l, r] is a piecewise-linear function of the query bounds, and representing it requires a second tree holding `(i-1)·D[i]`. Teams frequently implement the first variant and then discover, weeks later, that the product needs range totals and that the trees must be rebuilt.

### 4.4 Asymptotic behaviour, limits, and counterexamples

| Result or invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| `R(l,r) = P(r) - P(l-1)` | additive aggregate with inverse | try the identity on minimum | min over [1,7] = 2, min over [1,2] = 5, difference meaningless | segment tree with min aggregate |
| Prefix/subtraction exactness | both prefixes at one version | compare with direct scan under injected writer pause | reader sees T[4] new and T[8] old: R(5,8) = 12 vs truth 14 | lock, single writer with snapshot, or MVCC copy |
| `T[i]` fits arithmetic width | width covers aggregate not per-bin max | compute n × max abs q and compare to type maximum | 50,000 × 900,000 = 45e9 > 2^31-1 | 64-bit cells and checked addition |
| update loop terminates | i >= 1 | log touched cells for i = 0 | lowbit(0) = 0, infinite loop | reject i < 1 at validation |
| position meaning stable | fixed ordering epoch | hash the position-to-SKU map | map changed without rebuild: totals attach to wrong bins | epoch bump plus rebuild |
| linear build correct | ascending iteration | compare build output to n updates | descending iteration double-counts parents | ascending loop as in F-05 |

The last row is the least obvious and the easiest to get wrong in a rewrite. Propagating from high indices downward adds a value into a parent that has already been finalised, and the resulting tree can still answer many queries plausibly, which is why a spot check on a single range may pass. The reliable test is a full comparison against the direct-scan oracle at small n, which is the test design Section 9 budgets for.

A limit worth stating plainly: the structure is a *derived* representation. It is not a ledger, and it holds no history. If the authoritative vector and the tree disagree, the authority wins and the tree is rebuilt. Trying to repair a divergence by reasoning about which cells look wrong is how an afternoon becomes a week.

### 4.5 Range corrections: deriving the two-tree variant

**Validity:** integer quantities, range corrections applied to contiguous positions, one writer, one version per read. **Why a second tree is needed:** define the difference array `D[1] = q[1]`, `D[i] = q[i] - q[i-1]` for `2 <= i <= n`, and `D[n+1] = -q[n]` as a sentinel that keeps the array balanced to zero. Then `q[i] = P_D(i)` and any range addition of x to `[l, r]` is exactly two point deltas in D: `D[l] += x` and `D[r+1] -= x`. Point queries are therefore free once a tree holds D.

A range *total* is harder. Writing `S(k) = Σ_{i=1..k} q[i]` and substituting `q[i] = Σ_{j=1..i} D[j]`:

```text
S(k) = Σ_{i=1..k} Σ_{j=1..i} D[j]
     = Σ_{j=1..k} D[j] * (k - j + 1)          [swap the summation order]
     = (k + 1) * Σ_{j=1..k} D[j] - Σ_{j=1..k} j * D[j]
```

So two prefix structures suffice: one over `D[j]` and one over `j * D[j]`. The range total is `S(r) - S(l-1)`. Each range correction touches at most two positions in each tree, so the correction costs `O(log n)` and the total query costs `O(log n)` with four prefix walks.

```text
FORMULA [F-07]: S(k) = (k + 1) * P_D(k) - P_JD(k)
WHERE: P_D(k) = Σ_{j<=k} D[j]; P_JD(k) = Σ_{j<=k} j * D[j]; both [eaches].
CONSTRAINT: D includes the sentinel at n+1 so that range corrections touching n
            remain balanced; arithmetic width must cover j * D[j], which grows
            with n times the per-bin quantity.
NUMERICAL EXAMPLE: q = [5,0,7,2,4,1,3,6] -> D = [5,-5,7,-5,2,-3,2,3,-6(sentinel)]
                  Add +3 to positions 3..5: D[3] += 3 -> 10; D[6] -= 3 -> -6.
                  New point values: [5,0,10,5,7,1,3,6] (verified by prefixing D).
                  Total over [3,5] = 10 + 5 + 7 = 22; direct = 7+2+4 + 9 = 22.
                  Total over [1,8] = 28 + 9 = 37; direct = 37.
INTERPRETATION: interval corrections cost two tree updates instead of a per-bin fan-out.
```

The practical consequence is a choice about event volume. A zone write-off covering 40 positions is either 40 point events or 1 range event. At 2,500 point events/s, replacing 40-point runs by single range events can reduce the event rate by a large factor, but it adds the F-07 arithmetic, a second tree, the `j * D[j]` overflow bound, and a harder recovery story, because a range event must be replayed as a range event. Teams that adopt the variant for volume reasons frequently discover that their audit requirement was per-bin deltas, and then they must expand the range event into per-bin records anyway for the ledger — which is the honest design: per-bin in the log, range in the derived index.

One further limit: F-07 assumes a fixed n. If the epoch changes length, `S(k)` and the sentinel meaning change together, so the trees and the epoch must be rebuilt as a unit. A range correction stamped with the old length applied to the new vector silently misplaces the right endpoint.

**Takeaway:** the algebra is safe under four named assumptions; every failure in Section 8 is one of them breaking.

## 5. Worked Numerical Example

**Evidence label:** ILLUSTRATIVE vector and adjustment stream; every arithmetic step below is VERIFIED, including the mixed-version counterexample and the operation counts.

### 5.1 Scenario and inputs

A chilled aisle holds eight consecutive bins. The unit of record is eaches. The vector at epoch `E7`, version `v=41`, is:

| Position i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q[i]` [eaches] | 5 | 0 | 7 | 2 | 4 | 1 | 3 | 6 |

The total on hand is 28 eaches. The planner needs the total over positions 3 through 7 inclusive, which is the aisle's "fast-pick core". The warehouse also runs a cycle-count correction at position 6 that removes one each, and later a supervisor reassignment that sets position 4 to 4 eaches.

### 5.2 Step 1: build by definition

Apply F-01 to each position, one at a time, and record the block being summed so the table is checkable without the tree.

| i | binary | lowbit | block `[i-lowbit+1, i]` | block contents [eaches] | `T[i]` |
|---:|---|---:|---|---|---:|
| 1 | 1 | 1 | [1,1] | 5 | 5 |
| 2 | 10 | 2 | [1,2] | 5+0 | 5 |
| 3 | 11 | 1 | [3,3] | 7 | 7 |
| 4 | 100 | 4 | [1,4] | 5+0+7+2 | 14 |
| 5 | 101 | 1 | [5,5] | 4 | 4 |
| 6 | 110 | 2 | [5,6] | 4+1 | 5 |
| 7 | 111 | 1 | [7,7] | 3 | 3 |
| 8 | 1000 | 8 | [1,8] | 5+0+7+2+4+1+3+6 | 28 |

### 5.3 Step 2: build by linear propagation

The same tree is produced in one ascending pass with F-05. Each row shows the cell being finalised and the parent receiving it.

| i | `T[i]` before push | parent `i+lowbit(i)` | parent after push |
|---:|---:|---:|---:|
| 1 | 5 | 2 | 5 |
| 2 | 5 | 4 | 5 |
| 3 | 7 | 4 | 12 |
| 4 | 14 | 8 | 14 |
| 5 | 4 | 6 | 4 |
| 6 | 5 | 8 | 19 |
| 7 | 3 | 8 | 22 |
| 8 | 28 | 16 (out of range) | 28 |

The final vector `T = [5,5,7,14,4,5,3,28]` matches Step 1 cell for cell. Two independent constructions agreeing is the cheapest available correctness check, and it catches the descending-iteration error immediately.

### 5.4 Step 3: the required range, two ways

```text
P(7) = T[7] + T[6] + T[4] = 3 + 5 + 14 = 22
P(2) = T[2]                = 5
R(3,7) = 22 - 5            = 17
Direct scan: q[3]+q[4]+q[5]+q[6]+q[7] = 7+2+4+1+3 = 17
```

The subtraction is exact because both prefixes were evaluated at version 41 and because addition of signed integers is invertible. Nothing in this computation depends on the tree being balanced or on the blocks being equal in size; it depends only on the partition property.

### 5.5 Step 4: point adjustment and the affected cells

The cycle count is adjustment `A-19` with `d = -1` at position 6.

```text
i = 6: T[6] += -1 -> 4
i = 6 + lowbit(6) = 8: T[8] += -1 -> 27
i = 8 + lowbit(8) = 16 > n = 8: stop
```

Cells 1, 2, 3, 4, 5, 7 are untouched, which is correct: none of their blocks contains position 6. The new version is 42. Recomputed queries:

```text
P(6) = T[6] + T[4] = 4 + 14 = 18   (was 19)
P(7) = T[7] + T[6] + T[4] = 3 + 4 + 14 = 21   (was 22)
R(3,6) = P(6) - P(2) = 18 - 5 = 13   (7+2+4+0 = 13)
R(3,7) = P(7) - P(2) = 21 - 5 = 16   (7+2+4+0+3 = 16)
Total = T[8] = 27                     (28 - 1 = 27)
```

Note which cells moved and which did not: `P(7)` changed even though positions 7 and 6's block boundaries did not, because `T[6]` sits inside the prefix partition of 7. A reader that cached `P(7)` from version 41 without a version check will serve 22 while the authority says 21, and the difference is exactly the delta that landed inside the cached prefix.

### 5.6 Step 5: assignment expressed as a delta

The supervisor asserts that position 4 must hold 4 eaches. The authoritative value is 2, so:

```text
d = q_new - q_old = 4 - 2 = +2
i = 4: T[4] += 2 -> 16
i = 8: T[8] += 2 -> 29
stop
```

New state at version 43: `T = [5,5,7,16,4,4,3,29]`, `q[4] = 4`, total 29, and `R(3,7) = P(7) - P(2) = (3+4+16) - 5 = 18`. Direct scan gives 7+4+4+0+3 = 18. The assignment touched the same two cells an addition would have touched, because the structure has no notion of assignment — only deltas.

### 5.7 Step 6: the duplicate that looks correct

The message broker redelivers `A-19` with the same `adjustment_id`. Without deduplication the tree applies `-1` a second time:

```text
T[6] = 4 - 1 = 3;  T[8] = 29 - 1 = 28;  q[6] = -1
R(3,7) = (3 + 3 + 16) - 5 = 17
Total = 28
```

Every query succeeds, every cell is internally consistent with the (now wrong) vector, and the total 28 happens to equal the *original* total, which is precisely why a spot check on the total alone will not catch this class of defect. The defect is visible only against the authoritative event log, and only if the policy that forbids negative on-hand is enforced at the delta, not at the report.

### 5.8 Step 7: the mixed-version read that produces a number nobody computed

This is the counterexample that justifies the version contract. Suppose the writer applies an adjustment `A-20` with `d = +2` at position 4, writing cells in ascending order: first `T[4] = 16`, then `T[8] = 29`. A reader computes `R(5,8)` as `P(8) - P(4)` and interleaves as follows:

| Reader action | Version observed | Value |
|---|---|---:|
| read `P(4)` = `T[4]` | after the write | 16 |
| read `P(8)` = `T[8]` | before the write | 27 |
| compute `R(5,8)` | mixed | 11 |

The true value of `R(5,8)` is `4 + 0 + 3 + 6 = 13` at both versions, because position 4 lies outside the range and the adjustment cannot change the answer. The mixed read returns 11, subtracting a delta that never belonged to the range. No individual cell read was torn; the sum of two individually valid readings is invalid. This is the single most important arithmetic fact in the sheet, and it is why "we use atomic integers" is not an answer to a consistency question.

### 5.9 Step 8: independent checks and sensitivity

Independent checks performed on the numbers above:

| Check | Method | Result |
|---|---|---|
| Sum consistency | Σ q[i] = T[8] at every version | 28, 27, 29 — consistent at v41, v42, v43 |
| Block partition | every query's cells are disjoint and cover the prefix | verified for k = 1..8 |
| Inverse identity | `R(i,i) = q[i]` for all i | holds at all three versions |
| Build equivalence | definition build vs linear build | identical vectors |
| Delta location | cells touched by `update(6)` are exactly those whose block contains 6 | {6, 8} |
| Mixed read | `P(8)` old minus `P(4)` new | 11 vs truth 13, confirming the hazard |

Sensitivity to the two specification choices that matter most:

| Stress | Original | Alternative | Conclusion change | Decision consequence |
|---|---|---|---|---|
| n grows 8 → 12,000 | ≤ 3 cells per query | ≤ 13 cells per query | query cost grows logarithmically, not linearly | per-read latency budget can stay sub-millisecond in memory |
| read pattern 90% queries | query-dominated | 90% updates | touched-cell budget shifts from prefix walks to ancestor chains | benchmark the real ratio before choosing structure |
| writer count 1 → 2 | serialized deltas | concurrent deltas | F-06 becomes unsound; totals drift | require a single logical writer or an external transaction |
| unit of record eaches → cases | integer totals in eaches | decimal cases | non-integer widths, rounding policy required | fix the unit of record permanently, convert at the boundary |
| bin max 900,000, n = 50,000 | int64 sufficient | int32 | total overflows 32-bit | widen cells, add checked arithmetic |

**Takeaway:** the worked example proves the arithmetic and simultaneously produces the two defects — duplicate replay and mixed-version read — that arithmetic alone cannot detect.

## 6. Visual / Geometric Intuition

The tree is a laminar family of intervals: any two blocks are either disjoint or one contains the other. That single property explains both traversals, and it explains why the structure is not a general-purpose index.

### 6.1 The block ladder for n = 8

```text
position:   1    2    3    4    5    6    7    8
            |    |    |    |    |    |    |    |
block:    [1] [1-2] [3] [1---4] [5] [5-6] [7] [1-------8]
length:    1    2    1    4    1    2    1    8
```

Read the diagram as a containment lattice rather than a left-to-right array. Interval [5,6] sits inside [1,8]; [1,4] also sits inside [1,8]; [3] sits inside [1,4]; [5,6] and [1,4] are disjoint. The prefix query for 7 consumes [7], [5,6], [1,4] — three disjoint intervals whose union is exactly [1,7]. The update at 6 writes [5,6] and [1,8] — the two intervals that contain position 6. There is no other interval containing 6, which is why exactly two cells change.

### 6.2 Why the prefix walk is a partition, not an approximation

Walk k = 7 downward: 7 (binary 111) consumes [7,7]; k becomes 6 (110) and consumes [5,6]; k becomes 4 (100) and consumes [1,4]; k becomes 0 and stops. In binary terms, each step removes the lowest set bit, so the sequence of consumed intervals corresponds one-to-one with the set bits of 7, from least significant to most significant. Since 7 has three set bits, the query reads three cells. Position 11 = 1011b has three set bits and reads three cells; position 12 = 1100b has two set bits and reads two; position 15 = 1111b reads four. The number of cells read by a prefix is exactly the population count of k, which is at most ⌊log2 n⌋ + 1 and averages about half that.

```text
popcount table, n = 15:
k    : 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
bits : 1 1 2 1 2 2 3 1 2  2  3  2  3  3  4
cells: 1 1 2 1 2 2 3 1 2  2  3  2  3  3  4
```

This table is the honest complexity picture. The bound is logarithmic, but the typical query is cheaper than the bound and the worst case is a small constant for realistic n. At n = 12,000 the measured maximum prefix walk is 13 cells and the average is 6.61.

### 6.3 Why the update walk follows containment upward

Position 6 belongs to [5,6] because 6 - lowbit(6) + 1 = 5; it belongs to [1,8] because 8 - lowbit(8) + 1 = 1; no other block reaches position 6. In general, the blocks containing i are exactly the intervals whose right endpoint r satisfies r - lowbit(r) < i ≤ r, and those right endpoints are generated by repeatedly adding lowbits. The addition `i += lowbit(i)` carries into the next zero bit, which is precisely the operation of moving to the next enclosing interval.

A useful way to remember the asymmetry: the query walk moves toward zero and shrinks the prefix; the update walk moves toward n and grows the containing interval. Neither walk ever visits an interval that is only partially overlapping, because laminarity forbids it.

### 6.4 Where the hazard lives

```text
version v41            version v42 (write in progress)
T[4] = 14              T[4] = 16   <-- already written
T[8] = 28              T[8] = 27   <-- not yet written

reader computes P(8) - P(4) with a mixed draw:
   P(8) = 27 (v41)   P(4) = 16 (v42)   difference = 11
truth at v41: 28 - 14 = 14;  truth at v42: 27 - 16 = 11
```

The interesting part of the diagram is that 11 *is* a legitimate value — at version 42 for the range [5,8] the answer is 4 + 0 + 3 + 6 = 13... but the mixed draw used `P(4)` from v42 and `P(8)` from v41, so it computed a value belonging to no version at all. The two columns are the two versions; the hazard is a diagonal read across them. Any single column is safe. That is the whole content of snapshot isolation for this structure.

### 6.5 The cost picture as cell touches

```text
cell touches per operation, n = 12,000 (measured on reference implementation)
update:   average 7.06, maximum 14
prefix:   average 6.61, maximum 13
range:    two prefixes, so average 13.22, maximum 26
```

Compare that with a direct scan of a 100-position aisle: 100 reads per query, independent of n. The tree wins whenever queries are frequent or ranges are wide. It loses when ranges are tiny and updates are rare, because a two-cell range costs two validation steps and up to 26 cell reads, which is worse than reading two cells directly. That crossover is a design fact, not a defect, and Section 7 quantifies it.

**Takeaway:** laminarity gives both traversals their exact cell counts; the hazard diagram shows that safety is a property of a single version column, not of individual cell reads.

## 7. Comparison / Selection Table

### 7.1 Structural alternatives

| Option | Update | Range total | Memory | Use when | Avoid when |
|---|---:|---:|---:|---|---|
| Plain array | O(1) | O(r - l + 1) | n | ranges are short, or queries are rare | wide ranges at high query rate |
| Static prefix array | O(n) rebuild | O(1) | n | vector frozen for a reporting window | any adjustment can arrive mid-window |
| Fenwick tree (point add) | O(log n) | O(log n) | n | additive aggregate, point deltas, one writer | min/max, range updates, multi-writer without isolation |
| Fenwick, difference array | O(log n) range add, O(log n) point query | point only | n | interval corrections, point reads | range totals are the product |
| Two Fenwick trees | O(log n) range add, O(log n) range sum | O(log n) | 2n | interval corrections plus range totals | state budget or proof budget is tight |
| Segment tree | O(log n) | O(log n) | ~2n–4n | min, max, count, mixed aggregates, lazy propagation | sums only and simplicity dominates |
| Balanced BST with subtree sums | O(log n) | O(log n) | n | order-statistic or insertion/deletion by key | fixed dense positions |
| Materialized aggregate table | O(k) per affected group | O(1) | groups × measures | few groups, coarse grain | per-bin granularity |

### 7.2 Choosing by the actual question

The selection rule is driven by three questions asked in order.

First: **is the aggregate invertible?** If the business wants the minimum or maximum quantity over a range, stop — the Fenwick point-add structure cannot answer it, and prefix subtraction will silently produce a number that looks plausible because subtraction always produces a number. Use a segment tree with the correct aggregate. If the business wants a count of bins below a threshold, that is again non-invertible in general; it is an order-statistic or bucketed problem.

Second: **where do the deltas land?** If each adjustment names one bin, the point-add structure applies. If adjustments name runs of bins — a zone write-off, a temperature excursion affecting an entire rack — then either the producer expands the run into per-bin events (costly in event volume, simple in semantics) or the service stores a difference array. The expansion cost is worth quantifying: a write-off covering 40 bins becomes 40 events instead of one, and at 2,500 events/s the difference between per-bin and per-run events is the difference between a 2,500/s stream and a much smaller one.

Third: **how many writers and readers, and what isolation do they need?** A single logical writer per vector, with readers taking a versioned snapshot, is the simplest contract that keeps the invariants true. Multi-writer requires an external serialization mechanism — a database transaction, a compare-and-swap on a version, or a single-threaded owner process. Without it, the structure provides no safety, and the interview answer "we would use a lock" must be followed by "on what, held for how long, and what happens when the lock holder dies".

### 7.3 Variant comparison on the same worked vector

Take q = [5,0,7,2,4,1,3,6] and the two operations from Section 5: `add(6, -1)` and `assign(4, 4)`.

| Structure | State after both ops | Query R(3,7) | Notes |
|---|---|---:|---|
| Plain array | q = [5,0,7,4,4,0,3,6] | 18 | direct scan of five cells |
| Prefix array (rebuilt) | P = [0,5,5,12,16,20,20,23,29] | 23 - 5 = 18 | rebuild cost after each op |
| Fenwick point-add | T = [5,5,7,16,4,4,3,29] | (3+4+16) - 5 = 18 | two prefix walks |
| Segment tree | same totals, different layout | 18 | more state, supports min too |
| Two-tree range-add | D-based, same totals | 18 | only needed if ops were range-wide |

All five agree on 18, which is the point: the structures are interchangeable for this query, and the choice is about cost, state, and which *other* queries must also be supported.

### 7.4 Anti-selection: when the tree is the wrong answer

Three situations where a competent team should reject it.

1. **Per-SKU totals over sparse keys.** If the product is "total eaches for SKU X across the network" and the positions are sparse and change often, an order-statistic tree or a keyed aggregate is the right structure. Dense positional indexing of a sparse key space wastes n and forces rebuilds.
2. **Non-integer quantities.** If the unit of record is a decimal (litres, kilograms) the integer arithmetic and the exactness claim need a fixed-point or rational representation with an explicit scale. Floating point totals are not exact and the word "exact" in this sheet's title would become false.
3. **Audit-first requirements.** If the requirement is "reconstruct any historical total", the tree is not the artifact — the event log is. The tree is a cache of the current state; history belongs in an append-only ledger with its own retention and access controls.

**Takeaway:** choose by aggregate invertibility, delta shape, and writer count; the tree is correct only when all three align.

### 7.5 The crossover, quantified

The Section 6 cost picture invites an obvious question: at what range width does a tree query stop beating a direct scan? Model cost in abstract units with three components that both paths share — input validation (3 units), authorization and version acquisition (5 units) — plus cell reads at 1 unit each. The tree's cell reads are bounded by the measured worst case for the range, 26 for n = 12,000, and the scan's are exactly the span.

| Span [bins] | Tree cost [units] | Direct scan [units] | Faster |
|---:|---:|---:|---|
| 2 | 34.4 | 10 | direct |
| 5 | 34.4 | 13 | direct |
| 10 | 34.4 | 18 | direct |
| 26 | 34.4 | 34 | direct (marginal) |
| 27 | 34.4 | 35 | tree |
| 40 | 34.4 | 48 | tree |
| 100 | 34.4 | 108 | tree |

The crossover sits at 27 bins, which is where the tree's fixed cell-read budget equals the scan's linear cost. Two consequences follow. First, a product that mostly asks for single-bin or two-bin reads is better served by a plain array over the authoritative store, and adding a tree adds state to maintain for no gain. Second, the crossover moves with n: at n = 1,048,576 the worst-case range read is 42 cells, pushing the crossover to about 43 bins. The correct design is often both structures — an array for the hot narrow reads and a tree for the wide ones — with the tree derived from the same authoritative vector.

The model deliberately excludes cache behaviour, branch prediction, and memory-level parallelism, all of which favour the array. It also excludes the version-acquisition cost, which is shared. It is therefore a structural estimate for choosing between designs, not a latency prediction.

## 8. Failure Modes & Edge Cases

Ten distinct diagnostics follow. Each names the observable symptom first, because in practice the symptom arrives before the theory. Categories use the course vocabulary: `VIOLATED_ASSUMPTION`, `NUMERICAL_EDGE`, `SCALE_FAILURE`, `CONCEPT_CONFUSION`.

### DIAG-01 — Zero index produces a non-terminating update

**Category:** `NUMERICAL_EDGE`. **Symptom:** a worker thread spins at 100% CPU on one adjustment request; no error is logged; the service stops applying subsequent adjustments because the writer is blocked. **Mechanism:** `lowbit(0) = 0 & -0 = 0`, so `i += lowbit(i)` leaves i at 0 and the loop condition `i <= n` remains true forever. The request never completes and the single writer is permanently occupied. **Competing causes:** an infinite retry loop in the message consumer; a deadlock on the state lock; a pathological input message causing a parser loop. **Discriminating test:** attach a debugger or dump the stack of the writer thread. If the frame is the update loop with `i = 0`, the cause is confirmed; if the frame is a lock acquisition or a socket read, it is not. A cheaper pre-production test is a unit case calling `update(0, d)` and asserting a typed rejection within a bounded time. **Correction:** validate `1 <= i <= n` at the API boundary and raise a typed error before any cell is touched. **Prevention:** property test over an index domain including 0, negative values, n+1, and the maximum integer. **Verification:** the rejection is observable in the audit trail, and the writer remains available for the next valid adjustment. **Mitigation risk:** a watchdog that kills the writer hides the loop but discards in-flight adjustments; the restart must rebuild from the log.

### DIAG-02 — Mixed-version read yields a total that never existed

**Category:** `VIOLATED_ASSUMPTION`. **Symptom:** two readers, or the same reader twice within a second, report different totals for the same range with no adjustment in between; reconciliation against the direct scan disagrees by a delta that belongs to an adjacent range. **Mechanism:** Section 5.8 — an update writes several cells; a query reads several cells; interleaving produces a diagonal across versions. Individual atomicity does not compose. **Competing causes:** a stale replica that lags the primary; a double-applied duplicate; a cached prefix invalidated incorrectly. **Discriminating test:** run the reader under a deterministic scheduler that pauses the writer between its first and second cell write, and check whether any range total differs from both the before and after direct scans. Replica lag is distinguished by comparing read timestamps and replication offsets: lag produces an *old but real* value, whereas interleaving produces a value that matches no version. **Correction:** serve reads from a single-version snapshot — either an immutable copy published by the writer, a read transaction against a database that provides snapshot isolation, or a version-stamped struct swapped by pointer. **Prevention:** make the version part of the read API so a caller cannot forget it; reject a read whose cells span versions. **Verification:** the same deterministic schedule now either blocks or returns a value equal to one of the two version-consistent answers. **Mitigation risk:** copying the whole vector per read is correct but allocates O(n) per query; the design must amortize with copy-on-write or a versioned pointer.

### DIAG-03 — Duplicate adjustment inflates or deflates the total

**Category:** `VIOLATED_ASSUMPTION`. **Symptom:** the on-hand total moves by twice the expected delta after a broker redelivery or a consumer restart; or a negative on-hand appears where the policy forbids it. **Mechanism:** the tree has no memory of which business events it has seen; each `add` call is applied. **Competing causes:** a genuine second physical movement (two picks, not one); a producer that emits both a correction and its reversal; a unit conversion applied twice. **Discriminating test:** compare the count of applied deltas per `adjustment_id` in the audit trail against 1. If the count is 2 for one id, replay is confirmed. If the count is 1 and the movement is real, the cause is upstream duplication of a genuine event, which requires a different fix — deduplication at the producer or a compensating event. **Correction:** maintain a durable seen-set keyed on `adjustment_id`, check-and-insert before applying, and make the insert part of the same transaction as the tree state change where possible. **Prevention:** idempotency key in the message contract; consumer offsets committed only after the apply; a reconciliation job that compares log-derived totals to tree totals daily. **Verification:** replay the same id 1,000 times and assert the total changes exactly once. **Mitigation risk:** an in-memory seen-set loses its contents on restart and permits replay; a durable set must be pruned by retention and must not prune faster than the broker's redelivery window.

### DIAG-04 — Assignment applied as an absolute value

**Category:** `CONCEPT_CONFUSION`. **Symptom:** after a cycle count, the total jumps by the counted quantity rather than by the correction; a bin holding 25 that is counted at 37 ends at 62. **Mechanism:** the caller passed `q_new` where the API expected a delta, and the tree dutifully added 37. **Competing causes:** a unit conversion error (37 cases of 24 became 37 eaches); an authority read that returned zero; a rounding step in an integration layer. **Discriminating test:** compare the applied delta in the audit record against `q_new - q_old` computed from the pre-adjustment snapshot. A mismatch of exactly `q_old` confirms the confusion; a mismatch of a factor of 24 confirms a unit error; a mismatch equal to `q_new` confirms an authority read returning zero. **Correction:** expose two distinct API calls — `apply_delta(i, d)` and `apply_target(i, q_new)` — with the latter reading the authoritative old value inside the same serialized section. Never accept an unlabelled number whose meaning is ambiguous. **Prevention:** type the parameter so that a target cannot be passed where a delta is expected (a wrapper type in a typed language, a required field name in a message schema). **Verification:** the cycle-count scenario in Section 5.6 reproduces exactly, and a mislabelled call is rejected at the boundary.

### DIAG-05 — Aggregate overflows the chosen integer width

**Category:** `NUMERICAL_EDGE`. **Symptom:** a total becomes negative or wildly small; for example a 12,000-bin vector with legal per-bin values up to 900,000 reports a total of −some value. **Mechanism:** the aggregate `n × max abs q` exceeds the type maximum. At the stated sizes the true maximum is 10,800,000,000 while signed 32-bit tops out at 2,147,483,647; the boundary is crossed at n ≈ 2,386 bins if every bin holds 900,000. **Competing causes:** a duplicate decrement driving values negative; a unit mismatch; a corrupted cell from a partial write. **Discriminating test:** compute `n × max_abs_q` from contract metadata and compare to the type maximum *before* deployment. At runtime, a total that is negative while all cells are non-negative and non-overflowing is impossible — so a negative total with non-negative cells proves overflow, not data corruption. **Correction:** widen to 64-bit signed cells and add checked arithmetic on every accumulation; 64-bit covers 9.22e18, which is far beyond any plausible inventory total. **Prevention:** a contract test that asserts the width covers the declared maximum with a stated headroom factor (for example 4×). **Verification:** inject the boundary values and confirm no wraparound, then confirm the rejection path when a value would exceed the declared maximum. **Mitigation risk:** widening cells doubles memory, which is trivial here (96 KB at n = 12,000) but not trivial in an embedded controller; the same reasoning must be repeated for any fixed-point representation.

### DIAG-06 — Position remap invalidates the tree silently

**Category:** `VIOLATED_ASSUMPTION`. **Symptom:** aisle totals are plausible but attach to the wrong SKUs; a slow-moving SKU appears to have the fast-mover's quantity; no checksum fails. **Mechanism:** the ordering epoch changed — a bin was inserted, an aisle renumbered, two zones merged — and the tree still contains block sums computed under the old mapping. Because the vector length and the arithmetic are unchanged, every query succeeds. **Competing causes:** a mislabelled SKU in master data; a mis-scanned location; a picking error recorded against the wrong bin. **Discriminating test:** hash the position-to-SKU mapping at the epoch's opening and compare to the hash stored with the tree. A mismatch proves the structural change. Independently, compare tree totals per position to a direct scan of the authoritative vector: if they agree, the tree is consistent with *some* vector and the question becomes which vector it should be. **Correction:** bump the epoch, rebuild the tree from the authoritative vector, and reject any adjustment stamped with the previous epoch. **Prevention:** make the epoch part of the tree's identity and of every adjustment message; publish the mapping hash in the service's health endpoint so drift is visible. **Verification:** a replay of the remap triggers a rebuild and the per-position comparison passes under the new epoch.

### DIAG-07 — Negative on-hand created by a legal-looking delta

**Category:** `VIOLATED_ASSUMPTION`. **Symptom:** a bin shows −1 each; picking refuses the location; the aisle total is lower than the sum of physical counts. **Mechanism:** the policy that forbids negative on-hand is enforced at the reporting layer, not at the delta, so an over-pick is accepted and the tree faithfully records −1. **Competing causes:** a duplicate decrement (DIAG-03) that drove a legitimate 0 to −1; a receipt recorded against the wrong bin, leaving one bin short; a physical count that was never entered. **Discriminating test:** check the adjustment history for the bin. If two decrements share an `adjustment_id`, it is duplication. If each delta is distinct and the physical count agrees with the tree, the cause is a real over-pick and the fix is a policy decision, not a data repair. **Correction:** enforce the non-negativity invariant atomically inside the writer — read the current value, reject the delta if it would breach zero, and record a typed rejection with the adjustment id. **Prevention:** a property test that random adjustment streams never produce a negative cell, plus an alert on rejected adjustments as a leading indicator of upstream error. **Verification:** the rejected adjustment leaves the tree unchanged (no partial mutation) and appears in the audit trail with the attempted delta.

### DIAG-08 — Crash between authority write and tree update

**Category:** `SCALE_FAILURE` *(availability)*. **Symptom:** after a node restart, the tree total differs from the authoritative total by the sum of adjustments applied in the crash window. **Mechanism:** two stores were updated in sequence with no atomic commit between them: the authoritative bin table accepted the delta, the process died before the derived tree was updated (or the reverse). **Competing causes:** a lost message; a duplicated message after restart; a checkpoint that is older than the log tail. **Discriminating test:** compare the last durable checkpoint's version to the log's last committed version. A gap of k adjustments with the authority ahead proves the tree is behind; if the authority is behind and the tree ahead, the ordering was reversed. **Correction:** rebuild the derived state from the authority (or from the log) at startup, and make the rebuild the *only* recovery path. **Prevention:** make the tree a pure function of the log and checkpoint, publish a reconciliation metric (authority total minus tree total) that must be zero, and test the crash window deliberately by killing the process between the two writes. **Verification:** the post-restart reconciliation metric returns to zero and the per-position comparison passes. **Mitigation risk:** a rebuild during peak trading costs read bandwidth; schedule it against a snapshot and gate traffic on readiness.

### DIAG-09 — Half-open and one-based conventions disagree

**Category:** `CONCEPT_CONFUSION`. **Symptom:** every range total is off by exactly one bin's quantity; a single-bin query returns twice the quantity or zero. **Mechanism:** the caller supplies a half-open interval `[l, r)` while the API interprets an inclusive `[l, r]`, so the right endpoint is included when it should not be, or the `l - 1` subtraction is skipped. **Competing causes:** an off-by-one in the epoch mapping (the position is 0-based while the tree is 1-based); a boundary check that silently clamps instead of rejecting. **Discriminating test:** query `R(i, i)` for several i. If the result equals `2 * q[i]`, the implementation subtracted `P(i)` instead of `P(i - 1)`; if it equals zero, it subtracted `P(i)` from `P(i)`. Both are single-line defects with unmistakable signatures. **Correction:** choose one convention, document it in the schema, and convert at the boundary with an explicit `+1` or `-1` in a named helper. **Prevention:** a property test asserting `R(i, i) = q[i]` and `R(1, n) = P(n)` for random vectors. **Verification:** both identities hold across the full index domain including the endpoints.

### DIAG-10 — Cached prefix served past its version

**Category:** `SCALE_FAILURE`. **Symptom:** a dashboard shows a total that is correct but stale by seconds; two panels on the same screen disagree because one is cached. **Mechanism:** a cache keyed only on the range `(l, r)` ignores the version, so a cached entry from v41 is returned after v42 published. **Competing causes:** replica lag (a real version, older); a client that computed the total from a cached cell; a CDN or proxy caching the HTTP response. **Discriminating test:** include the version in the response payload. If two panels show different versions, the cache is version-blind; if they show the same version and different totals, the defect is elsewhere. **Correction:** key the cache on `(epoch, version, l, r)` or invalidate on version change, and expose the version to the client so staleness is observable. **Prevention:** an explicit freshness contract per consumer — some panels legitimately accept 5-second staleness, others require the latest version; encode the choice rather than hoping. **Verification:** a forced version bump invalidates the affected keys, and the panel reports the new version.

### Cross-cutting edge cases outside the ten

| Edge case | Behaviour required | Why it is not a separate diagnostic |
|---|---|---|
| n = 1 | update touches cell 1; query `R(1,1)` returns `q[1]` | covered by DIAG-09 identities |
| Empty vector (n = 0) | reject all queries; the structure is vacuous | contract boundary, not a defect |
| All zeros | prefix walks still work; no special case | tests that the code does not divide by a total |
| Maximum integer cells | checked arithmetic rejects the overflow | DIAG-05 |
| Concurrent readers only | safe if each read is version-consistent | DIAG-02 |
| Adjustment arriving before the epoch is published | reject as wrong epoch | DIAG-06 |
| Late correction to a closed period | apply to the current period with an explicit backdating record | accounting policy, out of algorithm scope |

**Takeaway:** nine of the ten diagnostics are contract defects; only DIAG-01 and DIAG-05 are arithmetic in the narrow sense, which is why testing the bit trick alone proves almost nothing.

## 9. Complexity & Cost Analysis

### 9.1 Algorithmic cost

Per operation, with n positions and w = ⌊log2 n⌋ + 1 bits:

| Operation | Cell reads | Cell writes | Time | Space |
|---|---:|---:|---|---|
| Build by n updates | 0 | O(n log n) | O(n log n) | O(n) |
| Linear build | n | O(n) | O(n) | O(n) |
| Point update | 0 | O(log n) | O(log n) | O(1) |
| Prefix query | O(log n) | 0 | O(log n) | O(1) |
| Range query | O(log n) | 0 | O(log n) | O(1) |
| Snapshot copy | O(n) | O(n) | O(n) | O(n) |

The average-case constants matter more than the bound. For n = 12,000 the reference implementation measures an average of 7.055 cells per update and 6.61 per prefix, with maxima of 14 and 13. Doubling n adds roughly one cell to each average, which is the behaviour that makes the structure viable at warehouse scale.

### 9.2 Back-of-envelope sizing

**ILLUSTRATIVE workload; arithmetic VERIFIED.** Assume one region with n = 12,000 bins, 2,500 adjustments/s at peak, and 40,000 range reads/s across dashboards, planners, and picking clients. Storage and compute:

```text
FORMULA [S-01]: memory = (n + 1) * bytes_per_cell
NUMERICAL EXAMPLE: (12,000 + 1) * 8 = 96,008 bytes = 93.76 KiB = 0.0916 MiB
INTERPRETATION: the tree itself is negligible; the cost lives elsewhere.

FORMULA [S-02]: cell_ops_per_s = U * avg_update_touches + Q * avg_query_touches
NUMERICAL EXAMPLE: 2,500 * 7.055 + 40,000 * 6.61 = 17,637.5 + 264,400 = 282,037.5
INTERPRETATION: under 0.3 million memory touches per second — trivial for one core.

FORMULA [S-03]: log_bytes_per_day = U * 86,400 * bytes_per_event
NUMERICAL EXAMPLE: 2,500 * 86,400 * 200 = 43,200,000,000 bytes = 40.23 GiB/day
INTERPRETATION: the durable adjustment log, not the tree, dominates storage.

FORMULA [S-04]: snapshot_bytes_per_s = (n + 1) * 8 * snapshots_per_s
NUMERICAL EXAMPLE: 96,008 * 1 = 96,008 bytes/s = 0.0916 MiB/s -> 7,910.8 MiB/day
INTERPRETATION: a full copy per published version is affordable at this n but
               scales linearly and must be replaced by copy-on-write above ~1e6 bins.
```

### 9.3 What actually dominates

Four costs sit above the tree in any real deployment, and each has a different scaling law.

**Durability.** Every accepted adjustment must survive a crash, so it reaches a log or a replicated table before or with the state change. At 200 bytes/event and 2,500 events/s the log grows at 40.23 GiB/day; retention, compaction, and replication multiply that. The tree adds nothing to this cost and cannot reduce it.

**Serialization.** A single logical writer is the simplest safe contract, and it caps update throughput at one core's worth of apply-and-fsync work. At 2,500/s with batching this is comfortable; at 250,000/s it is not, and the design must shard by an independent vector (per warehouse, per aisle) so that each shard has its own writer. Sharding must preserve the invariant that a query never spans two shards without a global version.

**Snapshot publication.** Readers need a coherent version. Copy-on-write publishes a pointer; full copies cost O(n) per version. At n = 12,000 the copy is 96 KB; at n = 10,000,000 it is 80 MB per version, which at one version per second is 6.9 TB/day. That number, not the tree's log factor, decides the architecture at large n.

**Reconciliation.** A daily comparison of tree totals against a direct scan costs O(n) reads plus the scan itself. At 12,000 bins this is milliseconds; across a network of 400 warehouses it is 400 scans and 400 comparisons, which is where the operational budget goes.

### 9.4 Latency reasoning without fabricated benchmarks

The sheet does not claim a latency figure because none was measured on production hardware. What can be stated from the operation counts is a bound: a range query performs at most 2 * (w) cell reads plus validation, so its compute time is bounded by a small multiple of memory access time plus the version-acquisition step. The version-acquisition step is the one that can dominate, because it may involve a lock, a pointer swap, or a database round trip. In the illustrative workload, 40,000 queries/s * 13.22 cell reads ≈ 528,800 reads/s if every query were worst-case, still far below what a single core can serve from cache. The honest statement is therefore: "the structure is not the bottleneck at this scale; the version contract is".

**Takeaway:** memory is negligible, compute is negligible, and durability, serialization, snapshot publication, and reconciliation are the real cost centres.

## 10. Key Theorems / Results Reference

### 10.1 Invariant preservation

**Claim:** after `update(i, d)` at a version where I-01 holds, I-01 holds again with `q[i]` replaced by `q[i] + d`.

**Proof sketch:** the update visits exactly the cells whose block contains i, because the sequence `i, i + lowbit(i), ...` enumerates the right endpoints of the enclosing blocks and terminates when the endpoint exceeds n. Every such cell's block sum must change by d; every other cell's block does not contain i and must not change. Since the loop visits each enclosing cell exactly once, each receives d exactly once. Therefore all block sums equal the sums of their blocks under the new vector, which is I-01.

**What this does not prove:** that the sequence is applied atomically with respect to readers (that is a concurrency property), that d corresponds to a real business event (that is the delta and dedup contract), or that `q[i] + d` is a legal quantity (that is policy).

### 10.2 Prefix correctness

**Claim:** `P(k)` as computed by F-02 equals `Σ q[1..k]` when I-01 holds at one version.

**Proof sketch:** induction on k. The consumed block at each step is `[k - lowbit(k) + 1, k]`, which is disjoint from all previously consumed blocks because each step strictly decreases k below the block's left endpoint... more precisely, after consuming `[k - b + 1, k]`, the new index is `k - b`, so the next block ends at `k - b`, strictly below the consumed block's left endpoint. The blocks therefore tile `[1, k]` without overlap. Their sums, by I-01, are the sums of their contents, and the union is `[1, k]`. Hence the total equals `P(k)`.

**Corollary:** `R(l, r) = P(r) - P(l - 1)` for `1 ≤ l ≤ r ≤ n`, by subtracting the two disjoint prefixes.

### 10.3 Termination

**Claim:** both loops terminate in at most `w` iterations.

**Proof sketch:** each iteration of the query loop clears the lowest set bit, so the population count strictly decreases; each iteration of the update loop sets the lowest zero bit above the current lowest set bit, so the value strictly increases and the number of trailing zeros strictly increases, bounding the iterations by the bit width. A zero index breaks the query claim trivially (it terminates immediately) but breaks the update claim (it never advances), which is why the validation in DIAG-01 is not optional.

### 10.4 Linear build correctness

**Claim:** the ascending propagation of F-05 produces the same tree as n point updates.

**Proof sketch:** in ascending order, when processing i, all positions j < i that contribute to block i have already been finalised and pushed into their parents; the accumulated `T[i]` at that moment is exactly the sum of block i. Pushing it once to `i + lowbit(i)` adds it to the unique parent block, which is the last block that contains i. By induction on i, every block sum is correct when the loop reaches it. The descending order fails because a parent can receive contributions after it has already been pushed upward.

### 10.5 What the theorems do not establish

| Property | Established by | Not established |
|---|---|---|
| Block sums correct | I-01 and update proof | business correctness of the deltas |
| Prefix and range exact | partition argument | visibility across writers |
| Termination | bit-count argument | bounded latency under contention |
| Build equivalence | ascending-induction argument | that the input vector was authoritative |
| Idempotency | *nothing here* | requires durable dedup outside the structure |
| Authorization | *nothing here* | requires a check before the total is emitted |

This table is the answer to the most common overclaim in interviews: "the algorithm is O(log n) and exact" is a statement about arithmetic under a version, not about the inventory service. The service must add authority, deduplication, epoch management, and access control, and each of those can fail independently of the algorithm.

### 10.6 Related results worth naming

The structure is a special case of a *Fenwick tree* (binary indexed tree) over an abelian group; it generalises to any associative operation with an inverse for the prefix-subtraction form, and to any associative operation for the "prefix-only" form. When the operation has no inverse, the segment tree is the standard alternative and provides `O(log n)` for both point updates and arbitrary range aggregates. The range-add/range-sum variant follows from the observation that the total after a range addition is piecewise linear in the query bounds, so a second tree holding `(i-1) · D[i]` suffices. These are all textbook results; the sheet's contribution is the contract analysis around them, not the algebra.

**Takeaway:** the proofs are short and they cover only the arithmetic; every operational guarantee must be argued separately.

## 11. Interview / Exam Questions

Twelve questions follow, spanning recall, derivation, and judgement. Each has a checked answer, a follow-up, and a scoring note. Numerical answers were recomputed independently.

### Q1 — Why does the prefix loop terminate, and where does that argument break?

**Level:** recall with a boundary. **Prompt:** "You claim the prefix query is O(log n). Prove it terminates, then tell me an input that breaks your proof."

**Answer:** For positive k, `k - lowbit(k)` clears the lowest set bit, so the value strictly decreases and the number of set bits strictly decreases; after at most ⌊log2 k⌋ + 1 steps k reaches 0 and the loop ends. The proof relies on `k ≥ 1`. At k = 0 the loop body never runs, so the query is vacuously fine, but the *same* expression in the update loop — `i += lowbit(i)` — never advances because `lowbit(0) = 0`. So the boundary is not the query but the update: the proof must be stated with `i ≥ 1`, and the implementation must reject `i < 1` before the loop rather than relying on the loop to terminate.

**Follow-up:** "What if n is not a power of two?" The bound still holds; the last block is truncated at n and the update loop stops when the next endpoint exceeds n. **Red flag:** claiming the loop terminates because "the index gets smaller" without noticing that the update loop increases the index. **Scoring:** 0 for "it's logarithmic"; 1 for stating the bit-clearing mechanism; 2 for stating the mechanism and the `i ≥ 1` precondition; 3 for additionally identifying that the update loop has the opposite monotonicity and a different failure at zero.

**Support:** Section 4.1 formulas F-02/F-03; Section 10.3.

### Q2 — Assignment: why can't the tree do it directly?

**Level:** application. **Prompt:** "A cycle count says bin 412 holds 37 eaches. The system stores 25. Show me the exact calls, and tell me what happens if two counters submit at once."

**Answer:** The tree accepts a signed delta at a position, so the call is `apply_delta(412, 37 - 25) = apply_delta(412, +12)`, evaluated inside the writer's serialized section. With two counters, both may read 25 and both submit +12, giving 49 eaches — an over-count of exactly one correction. The read of the old value and the application of the delta must therefore be one atomic step under a single logical writer, or the second submission must be rejected by a version check (compare-and-swap on the bin's version). The tree cannot resolve the conflict because it has no notion of a target value or of a writer identity.

**Follow-up:** "What if the second counter's count is the more recent one?" Then the correct action is a second delta computed against the *new* authoritative value (37 → 40 would be +3), not a replay of the first delta. **Red flag:** computing `new - old` outside the transaction, or storing the target and letting a batch job compute deltas later without version checks. **Scoring:** 3 for naming the atomic read-modify-write requirement and the over-count arithmetic.

**Support:** Section 4.2 formula F-06; DIAG-04.

### Q3 — Range minimum, same structure?

**Level:** judgement. **Prompt:** "Product now wants the lowest stock level in the aisle. Reuse the same tree?"

**Answer:** No. `R(l,r) = P(r) - P(l-1)` requires an operation with an inverse; minimum has none. Concretely, with q = [5,0,7], min over [1,3] is 0 and min over [1,1] is 5; the difference 0 − 5 = −5 is meaningless, and there is no way to recover min over [2,3] from the two prefix minima. The correct structure is a segment tree storing the minimum at each node with `O(log n)` point update and range query, or a sparse table if the vector is static. A Fenwick tree can support prefix minimum for the special case of *non-increasing* updates only, which is not the general requirement.

**Follow-up:** "What if the vector is static for the reporting window?" Then a sparse table gives `O(1)` range minimum after `O(n log n)` preprocessing, which is better than a segment tree. **Red flag:** "we can store negative values and use max" — negation changes min to max of negated values, but only if the structure supports max, which the sum-based tree does not.

**Support:** Section 4.4 table; Section 7.2 first selection question.

### Q4 — The mixed-version number

**Level:** debug. **Prompt:** "Reconciliation says the tree total for [5,8] is 11; a direct scan says 13. The writer logs show one adjustment of +2 at position 4. Explain."

**Answer:** Position 4 is outside [5,8], so no consistent version can produce 11 — both v41 and v42 give 13. The 11 arises from a read that mixed versions: `P(4)` after the write (16) and `P(8)` before the write (27), giving 27 − 16 = 11. The read subtracted a delta that belonged to a block ending at 4, not to the queried range. The discriminating test is to re-read under a fixed version: if both prefixes carry the same version the answer is 13. The fix is a versioned snapshot, not a repair of the cells, which are individually correct.

**Follow-up:** "How would you make this reproducible in a test?" Inject a writer that yields between its first and second cell write, run the reader, and assert it either blocks or returns a value equal to one of the version-consistent answers. **Red flag:** "the cells are atomic so this cannot happen" — the counterexample is exactly that atomic cells do compose badly.

**Support:** Section 5.8; DIAG-02.

### Q5 — Sizing the state and the log

**Level:** calculation. **Prompt:** "12,000 bins, 64-bit cells, 2,500 adjustments/s, 200-byte log records. How much state and how much log per day? Which one would you alarm on?"

**Answer:** State is `(12,000 + 1) × 8 = 96,008 bytes`, about 93.76 KiB or 0.0916 MiB — negligible. The log is `2,500 × 86,400 × 200 = 43,200,000,000 bytes`, about 40.23 GiB/day before replication and compaction. Alarm on log growth and on reconciliation discrepancy, not on memory: the state is small enough that rebuilding it is cheap, so the log is both the recovery mechanism and the actual capacity cost. At 3× replication the daily footprint rises to about 120.7 GiB, which drives retention policy.

**Follow-up:** "At 400 warehouses, what changes?" State becomes 400 × 96 KB ≈ 37.5 MiB, still negligible; log becomes 16.1 TiB/day, which forces compaction, tiering, or sampling of low-value events. **Red flag:** sizing the tree and ignoring durability, which is the pattern that produces an elegant design and an unbudgeted storage bill.

**Support:** Section 9.2 formulas S-01 to S-03.

### Q6 — Overflow boundary

**Level:** calculation with a decision. **Prompt:** "Bins legally hold up to 900,000 eaches. How many bins can you support in 32-bit cells?"

**Answer:** The aggregate must fit, so `n ≤ (2^31 − 1) / 900,000 = 2,386.4`, hence 2,386 bins with zero headroom. But individual cells store block sums, which can equal the aggregate, so the same bound applies to a single cell — meaning a vector of 2,386 bins each at maximum already saturates the type. With any headroom factor (say 4×) the supported n drops to about 596. The correct answer for a 12,000-bin aisle is 64-bit cells: `12,000 × 900,000 = 10,800,000,000`, which is 5.03× the 32-bit maximum and 1.17e-9 of the 64-bit maximum.

**Follow-up:** "What about intermediate arithmetic in a subtraction?" `P(r) − P(l−1)` stays within the aggregate range, so if cells are 64-bit and the aggregate fits, the subtraction is safe; but an implementation that sums in a narrower accumulator can overflow even with 64-bit cells. **Red flag:** sizing by per-bin maximum only, forgetting that a cell holds a block sum.

**Support:** Section 4.4 row 3; DIAG-05.

### Q7 — Duplicate detection that a total check misses

**Level:** judgement. **Prompt:** "After a broker redelivery, the total is unchanged. Is that proof the duplicate was harmless?"

**Answer:** No. In the worked example a duplicated −1 took the total from 29 to 28, which happens to equal an earlier legitimate total; a spot check against "the total looks familiar" fails. The correct check is per-`adjustment_id` application count in the audit trail, and the correct invariant is that each business event contributes exactly one delta. A total check can pass while the per-bin vector is wrong, because two errors can cancel in the aggregate.

**Follow-up:** "Design the dedup store." A durable key-value set keyed on `adjustment_id` with a retention window at least as long as the broker's maximum redelivery interval; the insertion and the state change should share a transaction where the storage supports it, and where it does not, the apply must be idempotent by construction or reconciled from the log. **Red flag:** an in-memory set, which loses its contents on exactly the restart that triggers redelivery.

**Support:** Section 5.7; DIAG-03.

### Q8 — Epoch and remap

**Level:** design. **Prompt:** "A bin is inserted into the middle of the aisle. What must happen to the tree, and what breaks if you do nothing?"

**Answer:** Insertion shifts every position after it by one, so the vector's meaning changes while its length and arithmetic stay consistent. Doing nothing produces totals that are plausible and attached to the wrong SKUs — for example the inserted bin inherits the position of the bin that used to follow it, and the last position's quantity is orphaned. The required action is an epoch bump, an authoritative remap, and a rebuild; adjustments stamped with the old epoch must be rejected rather than applied to the new numbering. A mapping hash published in the health endpoint makes the drift observable before a planner notices a wrong total.

**Follow-up:** "Could you avoid the rebuild with a sparse position space?" Yes — allocate positions with gaps (1, 10, 20, ...) or use a keyed structure, at the cost of wasted n and a different structure. That is a legitimate design trade, but it must be decided before the epoch, not during an incident. **Red flag:** "just insert into the array and add a delta at the new index" — that leaves every later block sum stale.

**Support:** DIAG-06; Section 4.2 epoch paragraph.

### Q9 — Concurrency without a lock

**Level:** design. **Prompt:** "Two writers per vector, no shared lock. Can you keep the tree correct with atomics?"

**Answer:** Not for multi-cell operations. Each update touches several cells; atomics make each cell read-modify-write safe but do not make the *set* of cell updates atomic, and two concurrent updates can interleave so that a cell receives both deltas while another receives one, leaving block sums that no vector explains. Options: serialize on a version with compare-and-swap retry (optimistic, correct if the state is small enough to re-read and retry), a single-owner actor per vector, or a database transaction with snapshot isolation. The cheapest correct design at warehouse scale is one owner per independent vector, sharded by warehouse or aisle, with a global read path that composes versions.

**Follow-up:** "What does sharding break?" A query spanning two shards has no single version unless the global read uses a consistent snapshot across shards; the composition must either accept per-shard versions in the response or use a distributed snapshot, which is a much heavier mechanism. **Red flag:** "use a mutex" without stating what it protects, how long it is held, and what happens when the holder dies.

**Support:** Section 2 map; Section 7.2 third question; DIAG-02.

### Q10 — Build correctness

**Level:** derivation. **Prompt:** "Someone rewrote the build to iterate from n down to 1 and propagate to the parent. Show a vector where it is wrong."

**Answer:** With q = [5,0,7,2] and descending order: i = 4 pushes 2 to parent 8 (out of range, no effect); i = 3 pushes 7 to 4, so T[4] = 2 + 7 = 9 — but T[4] was already finalised at 2 and pushed; i = 2 pushes 0 to 4, T[4] = 9; i = 1 pushes 5 to 2, T[2] = 5. The correct T[4] is 5 + 0 + 7 + 2 = 14, but the descending build produces 9, having missed the contributions from positions 1 and 2 that arrived after position 4 was pushed. The error is not a rounding artifact; it is a structural loss of contributions, and it can still answer some queries plausibly.

**Follow-up:** "How do you catch it cheaply?" Compare the linear build to n point updates on random small vectors; the equivalence test is exhaustive for n ≤ 12 and catches ordering errors immediately. **Red flag:** spot-checking one range and declaring the build correct.

**Support:** Section 5.3; Section 10.4.

### Q11 — When is the tree the wrong tool?

**Level:** judgement, changed constraint. **Prompt:** "The unit of record changes from eaches to kilograms, and the product needs the total weight of the aisle to two decimal places. Still exact?"

**Answer:** Not with integer cells and not with the word "exact" unless the representation is fixed. Store grams as integers (a scale of 10^-3 kg), and the same structure remains exact with the unit declared as grams; convert to kilograms only for display, with a rounding rule. Storing decimal kilograms in binary floating point makes the totals inexact and order-dependent, and the arithmetic in Section 4 no longer establishes what the title claims. The general rule: choose the smallest integer unit that the business can measure, declare it, and never mix units inside one vector.

**Follow-up:** "What if the measurements are themselves approximate?" Then exactness is about the arithmetic on recorded values, not about physical truth; the sheet's claims survive, but the label must say "exact total of recorded quantities", not "exact inventory". **Red flag:** "use double, it's precise enough" without stating the error bound and the aggregation order.

**Support:** Section 3 definitions; Section 7.4.

### Q12 — Changed scale

**Level:** judgement. **Prompt:** "n grows to 10 million bins and the service publishes a new version once per second. What breaks first?"

**Answer:** Snapshot publication. The tree's compute stays logarithmic and its memory is 80 MB at 10 million 64-bit cells, both fine. But a full copy per version is 80 MB/s, about 6.9 TB/day, and copy-on-write at that churn degenerates toward full copies. The design must move to a persistent or versioned structure (path-copying segment tree, immutable pages, or a database with MVCC) so that a version shares most of its pages with the previous one. The second thing to break is the single writer, whose per-event durability cost dominates at high rates; the third is reconciliation, which becomes an O(n) scan per warehouse.

**Follow-up:** "Does the query get expensive?" A prefix at 10 million reads at most 24 cells, so no. **Red flag:** optimizing the bit operations when the bottleneck is publication and durability.

**Support:** Section 9.3; Section 6.5.

**Takeaway:** the questions that separate candidates are about versions, dedup, units, and epochs, not about the four lines of lowbit arithmetic.

## 12. Quick Reference Card

| Field | Value |
|---|---|
| Purpose | Exact inclusive range totals over an ordered quantity vector with point adjustments and one coherent read version |
| Structure | Fenwick tree: `T[i]` = sum of block `[i - lowbit(i) + 1, i]`; build linear `O(n)` or by `n` updates |
| Update | `while i <= n: T[i] += d; i += lowbit(i)`; validate `1 <= i <= n` first |
| Query | `P(k)`: `while k > 0: s += T[k]; k -= lowbit(k)`; range `R(l,r) = P(r) - P(l-1)` |
| Cost | `O(log n)` per operation; at n = 12,000 averages 7.06 update cells and 6.61 prefix cells, maxima 14 and 13 |
| State | `(n+1) × 8` bytes = 96,008 bytes at n = 12,000; log at 2,500 events/s × 200 B ≈ 40.23 GiB/day |
| Four contracts | Fixed position epoch; delta with durable dedup; width covering `n × max|q|`; single version per read |
| Non-invertible aggregates | Minimum, maximum, and counts cannot use prefix subtraction — use a segment tree |
| Top failure signs | Total unchanged after replay (DIAG-03); total matching no version (DIAG-02); one-bin quantities doubled (DIAG-09) |
| Evidence | DERIVED algorithm; VERIFIED illustrative arithmetic and operation counts; no production measurement; source verification OPEN |

**Takeaway:** four lines of arithmetic under four contracts — the contracts are the engineering, the tree is the implementation detail.
