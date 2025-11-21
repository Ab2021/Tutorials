# Day 43: Algorithms - Interview Questions

> **Topic**: Advanced Algorithms
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. QuickSort vs MergeSort.
**Answer:**
*   **QuickSort**: $O(N \log N)$ avg, $O(N^2)$ worst. In-place ($O(\log N)$ stack). Unstable. Good for arrays.
*   **MergeSort**: $O(N \log N)$ always. $O(N)$ space. Stable. Good for Linked Lists.

### 2. Binary Search implementation.
**Answer:**
*   `low = 0, high = n-1`.
*   `mid = low + (high - low) // 2`.
*   Avoid overflow.

### 3. Dijkstra's Algorithm.
**Answer:**
*   Shortest path in weighted graph (non-negative).
*   **Priority Queue**.
*   Greedily pick closest node. Relax neighbors.
*   Time: $O(E \log V)$.

### 4. Bellman-Ford Algorithm.
**Answer:**
*   Shortest path with **negative weights**.
*   Relax all edges $V-1$ times.
*   Detect negative cycles.
*   Time: $O(V \cdot E)$.

### 5. A* Search.
**Answer:**
*   Pathfinding with Heuristic ($f = g + h$).
*   $g$: cost so far. $h$: estimated cost to goal.
*   Faster than Dijkstra if heuristic is good.

### 6. Dynamic Programming: Climbing Stairs.
**Answer:**
*   $dp[i] = dp[i-1] + dp[i-2]$.
*   Fibonacci sequence.

### 7. DP: Longest Common Subsequence (LCS).
**Answer:**
*   2D Grid.
*   If `s1[i] == s2[j]`: `1 + dp[i-1][j-1]`.
*   Else: `max(dp[i-1][j], dp[i][j-1])`.

### 8. DP: 0/1 Knapsack Problem.
**Answer:**
*   Include item or exclude item.
*   `dp[i][w] = max(val[i] + dp[i-1][w-wt[i]], dp[i-1][w])`.

### 9. Topological Sort.
**Answer:**
*   Ordering of tasks with dependencies (DAG).
*   **Kahn's Algorithm** (Indegree).
*   Or DFS (Reverse post-order).

### 10. Union-Find (Disjoint Set).
**Answer:**
*   Operations: `find(x)`, `union(x, y)`.
*   Optimizations: **Path Compression**, **Union by Rank**.
*   Time: $\alpha(N)$ (Inverse Ackermann - nearly constant).

### 11. KMP Algorithm (String Matching).
**Answer:**
*   Find pattern in text.
*   Precompute **LPS** (Longest Prefix Suffix) array.
*   Avoids backtracking in text.
*   Time: $O(N + M)$.

### 12. Reservoir Sampling.
**Answer:**
*   Select K items from a stream of unknown length N.
*   Keep first K.
*   For $i > K$, replace element in reservoir with prob $K/i$.

### 13. Segment Tree.
**Answer:**
*   Range Queries (Sum, Min, Max) in $O(\log N)$.
*   Updates in $O(\log N)$.
*   Tree structure where node covers a range.

### 14. Trie (Prefix Tree).
**Answer:**
*   Store strings.
*   Fast prefix lookup.
*   Used for Autocomplete.

### 15. Heap Sort.
**Answer:**
*   Build Max-Heap.
*   Swap root with last. Heapify down.
*   Time: $O(N \log N)$. Space: $O(1)$.

### 16. Floyd-Warshall Algorithm.
**Answer:**
*   All-pairs shortest paths.
*   3 nested loops.
*   Time: $O(V^3)$.

### 17. Prim's vs Kruskal's (MST).
**Answer:**
*   **Prim**: Grow tree from a node (Priority Queue). Good for dense graphs.
*   **Kruskal**: Sort edges. Add if no cycle (Union-Find). Good for sparse graphs.

### 18. Boyer-Moore Voting Algorithm.
**Answer:**
*   Find majority element (> N/2).
*   Maintain `candidate` and `count`.
*   Time: $O(N)$. Space: $O(1)$.

### 19. Rabin-Karp Algorithm.
**Answer:**
*   String matching using **Rolling Hash**.
*   Good for multiple pattern search.

### 20. Convex Hull (Graham Scan).
**Answer:**
*   Find boundary polygon of points.
*   Sort by polar angle. Stack.
*   Time: $O(N \log N)$.
