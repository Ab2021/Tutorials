# Day 42: ML Coding Round - Data Structures

> **Phase**: 5 - Interview Mastery
> **Week**: 9 - Behavioral & Coding
> **Focus**: DSA for ML Engineers
> **Reading Time**: 60 mins

---

## 1. Why DSA matters for ML?

You won't invert binary trees. But you will need to:
*   Implement Autocomplete (Trie).
*   Find Top-K items (Heap).
*   Traverse a computation graph (DFS/BFS).

---

## 2. Essential Structures

### 2.1 Tries (Prefix Trees)
*   **Use Case**: Autocomplete, Spell Checker, Tokenizer implementation.
*   **Complexity**: $O(L)$ to insert/search, where $L$ is word length. Faster than Hash Map for prefix search.
*   **Code Snippet**:
    ```python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
    ```

### 2.2 Heaps (Priority Queues)
*   **Use Case**: "Find the Top 10 most similar vectors."
*   **Algorithm**: Keep a Min-Heap of size K. Iterate through N items. If item > heap.min, pop min and push item.
*   **Complexity**: $O(N \log K)$. Much better than sorting $O(N \log N)$.

### 2.3 Graphs (DAGs)
*   **Use Case**: TensorFlow/PyTorch computation graphs. Workflow orchestration (Airflow).
*   **Algorithm**: Topological Sort (Kahn's Algorithm) to determine execution order.

---

## 3. Practice Problems

### Problem 1: Implement a Tokenizer
**Task**: Given a dictionary `["the", "their", "there"]`, implement `tokenize("thethere")` -> `["the", "there"]`.
**Solution**: Use a Trie to match longest prefixes.

### Problem 2: K-Nearest Neighbors
**Task**: Given a list of points and a query point, find K nearest.
**Solution**: Calculate distances. Use `heapq.nsmallest`.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: When would you use a KD-Tree vs. a Ball Tree?**
> **Answer**:
> *   **KD-Tree**: Good for low dimensions (< 20). Splits space along axes.
> *   **Ball Tree**: Better for high dimensions. Splits data into hyperspheres.
> *   **HNSW**: Best for very high dimensions (approximate).

**Q2: How do you detect a cycle in a computation graph?**
> **Answer**: DFS. Keep track of `visited` nodes and `recursion_stack`. If you see a node currently in `recursion_stack`, there is a cycle.

---

## 5. Further Reading
- [LeetCode: Trie Problems](https://leetcode.com/tag/trie/)
- [Topological Sort Explained](https://www.geeksforgeeks.org/topological-sorting/)
