# Day 43 (Part 1): Advanced Algorithms

> **Phase**: 6 - Deep Dive
> **Topic**: Hard Coding Problems
> **Focus**: DP, Sampling, and Bit Manipulation
> **Reading Time**: 60 mins

---

## 1. Dynamic Programming on Trees

### 1.1 Diameter of Tree
*   Longest path between any two nodes.
*   **DFS**: Return (max_depth, max_diameter_in_subtree).
*   `max_depth = 1 + max(left, right)`
*   `diameter = max(left_depth + right_depth, left_diam, right_diam)`.

### 1.2 House Robber III (Tree)
*   Can't rob adjacent nodes (Parent/Child).
*   **State**: `rob(node)` returns `[money_with_node, money_without_node]`.

---

## 2. Sampling Algorithms

### 2.1 Reservoir Sampling
*   **Problem**: Select K items from a stream of unknown length N.
*   **Algo**:
    1.  Keep first K items.
    2.  For $i > K$, keep item with probability $K/i$. Replace random existing item.
*   **Proof**: Probability is always $K/N$.

### 2.2 Alias Method
*   **Problem**: Sample from categorical distribution in $O(1)$.
*   **Setup**: $O(N)$ to build table.
*   **Idea**: Pair "Overfull" buckets with "Underfull" buckets to create uniform blocks.

---

## 3. Tricky Interview Questions

### Q1: Detect Cycle in Directed Graph?
> **Answer**:
> *   **DFS**: Keep `visited` set and `recursion_stack` set.
> *   If node in `recursion_stack`, cycle found.
> *   **Kahn's Algo**: Topological Sort. If nodes left with degree > 0, cycle exists.

### Q2: Median of Data Stream?
> **Answer**:
> *   Two Heaps: Max-Heap (Lower half), Min-Heap (Upper half).
> *   Balance sizes. Median is top of heaps.

### Q3: Bitmask DP?
> **Answer**:
> *   **TSP (Traveling Salesman)**.
> *   State: `dp[mask][last_city]`.
> *   `mask` represents set of visited cities (10110).

---

## 4. Practical Edge Case: Floating Point Equality
*   **Problem**: `0.1 + 0.2 == 0.3` is False.
*   **Fix**: `abs(a - b) < epsilon`.

