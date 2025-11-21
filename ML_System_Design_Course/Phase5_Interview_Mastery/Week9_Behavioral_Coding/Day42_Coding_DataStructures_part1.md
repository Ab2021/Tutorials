# Day 42 (Part 1): Advanced Data Structures for ML

> **Phase**: 6 - Deep Dive
> **Topic**: Coding Round Survival
> **Focus**: Heaps, Graphs, and Segment Trees
> **Reading Time**: 60 mins

---

## 1. Heaps (Priority Queues)

Crucial for "Top K" problems.

### 1.1 Patterns
*   **Top K Elements**: Keep a Min-Heap of size K. If new element > root, pop and push. Complexity $O(N \log K)$.
*   **Merge K Sorted Lists**: Push first element of each list to Min-Heap. Pop min, push next from same list. $O(N \log K)$.

---

## 2. Advanced Graphs

### 2.1 Union Find (Disjoint Set)
*   **Use Case**: Connected components, Cycle detection.
*   **Ops**: `find(x)`, `union(x, y)`.
*   **Optimization**: Path Compression + Rank. $O(\alpha(N))$ (Inverse Ackermann - nearly constant).

### 2.2 Segment Tree
*   **Use Case**: Range Queries (Sum, Max) with Updates.
*   **Complexity**: $O(\log N)$ for query and update.
*   **ML Context**: Efficiently sampling from a distribution that changes (e.g., Prioritized Experience Replay in RL).

---

## 3. Tricky Interview Questions

### Q1: Monotonic Stack?
> **Answer**:
> *   Find "Next Greater Element".
> *   Keep stack sorted. Pop elements smaller than current.
> *   $O(N)$.

### Q2: Sliding Window Maximum?
> **Answer**:
> *   Use **Monotonic Deque**.
> *   Keep indices of decreasing elements.
> *   Remove indices out of window.
> *   Front is always max. $O(N)$.

### Q3: Trie (Prefix Tree)?
> **Answer**:
> *   **Use Case**: Autocomplete.
> *   **Space**: $O(N \times L)$.
> *   **Search**: $O(L)$.

---

## 4. Practical Edge Case: Sparse Vector Dot Product
*   **Problem**: Vectors are 1M dim, but only 100 non-zeros.
*   **Format**: List of `(index, value)`.
*   **Algo**: Two pointers. Iterate through sorted indices. Multiply if indices match. $O(N_1 + N_2)$.

