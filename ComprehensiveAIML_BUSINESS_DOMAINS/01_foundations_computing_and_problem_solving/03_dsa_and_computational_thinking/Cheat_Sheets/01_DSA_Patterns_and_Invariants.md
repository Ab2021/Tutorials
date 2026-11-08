---
collection: "dsa-and-computational-thinking"
file_id: "01"
topic: "DSA patterns and invariants"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# DSA Patterns and Invariants

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is source inventory only. This is a new course-topic draft, not a reconstruction of unavailable source material.

## 1. Pattern selection

Select an algorithm from input representation, operation mix, correctness contract and resource bound—not from a memorized label.

| Pattern | Signal | Invariant | Boundary |
|---|---|---|---|
| Hash map/set | Membership/count by key | Each stored key maps to intended latest/aggregate state | Hash collision/resource limits |
| Two pointers/window | Contiguous range with monotonic expand/shrink rule | Window represents exactly eligible range | Non-monotone predicates need other method |
| BFS | Unweighted shortest hop / frontier | First discovery distance is minimal | Memory can grow with frontier |
| Heap | Repeated min/max/top-k | Heap order, not sorted order | Arbitrary lookup is not O(1) |
| Binary search | Monotone predicate/order | Answer remains in declared interval | Off-by-one and nonmonotone predicate |
| Dynamic programming | Overlapping subproblems/optimal substructure | State meaning and recurrence cover all cases | State explosion |

## 2. Complexity is a contract

State time, auxiliary memory, input size unit and worst/average/amortized condition. `O(n)` with a hash table assumes bounded expected hash operations, not adversarial collision immunity. `O(log n)` binary search assumes an ordered domain and monotone predicate. Big-O omits constants but does not excuse unbounded allocations or database/network calls hidden inside one operation.

## 3. Example: BFS shortest hop

For an unweighted graph, initialize source distance 0; enqueue it; when dequeuing `u`, assign unvisited neighbor `v` distance `dist[u]+1` and enqueue once. The invariant: queue discovery occurs in nondecreasing distance, so first assignment is shortest hop count. On edges `A-B`, `A-C`, `B-D`, `C-D`, BFS from A gives `dist(D)=2`; both paths are valid and a deterministic parent/tie policy decides which path is returned.

## 4. Correctness and testing

Write preconditions, state invariant, termination argument and output contract. Test an executable small oracle: compare BFS to exhaustive shortest paths on small graphs; compare range query structure to an array; compare DP to exhaustive enumeration at small size. Include empty, one-item, duplicate, tie, overflow, disconnected/cyclic and adversarial inputs.

## 5. Failure modes

| Symptom | Check | Response |
|---|---|---|
| BFS revisits/explodes | Mark-on-discovery and queue trace | Visit once; bound input/admission |
| Binary search loops | Interval/variant trace | Use half-open invariant and shrinking proof |
| Heap output assumed sorted | Pop sequence test | Pop all or use sort; document order |
| Hash DoS | Collision/adversarial timing | Bound input/use resistant map/runtime controls |
| Recursion stack overflow | Depth trace | Iterative traversal or declared depth limit |

## 6. Interview checks

1. **What proves BFS shortest hops?** Discovery-order invariant on unweighted edges.
2. **What must binary search have?** Monotone predicate and shrinking interval.
3. **Why not say O(1) hash lookup absolutely?** It is usually expected/amortized with assumptions.
4. **What makes a DP state valid?** It captures all history needed for recurrence.
5. **How test algorithm correctness?** Small exhaustive/reference oracle plus boundary/adversarial cases.

## 7. Quick reference

| Before implementation | State |
|---|---|
| Input and operation mix | Representation/size/update/query constraints |
| Correctness | Preconditions, invariant, termination, tie rule |
| Resource | Time/memory unit and worst/expected assumptions |
| Validation | Small oracle and boundary/adversarial trace |

**Limitations:** No source lesson or measured production trace was available. Examples are illustrative.
