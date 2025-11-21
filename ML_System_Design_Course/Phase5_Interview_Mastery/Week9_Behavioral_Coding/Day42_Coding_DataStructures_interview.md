# Day 42: Coding (Data Structures) - Interview Questions

> **Topic**: LeetCode Style
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Reverse a Linked List.
**Answer:**
*   Iterative: 3 pointers (prev, curr, next).
*   `curr.next = prev`. Move all forward.
*   Time: $O(N)$. Space: $O(1)$.

### 2. Detect Cycle in a Linked List.
**Answer:**
*   **Floyd's Cycle-Finding Algorithm** (Tortoise and Hare).
*   Slow moves 1 step. Fast moves 2 steps.
*   If they meet, cycle exists.

### 3. Valid Parentheses.
**Answer:**
*   Use a **Stack**.
*   Push open brackets `(`, `{`, `[`.
*   If close bracket, pop and check match.
*   Return `stack.isEmpty()`.

### 4. Implement a Queue using Stacks.
**Answer:**
*   Two Stacks: `Input`, `Output`.
*   **Push**: Push to `Input`.
*   **Pop**: If `Output` is empty, pop all from `Input` and push to `Output`. Pop from `Output`.
*   Amortized $O(1)$.

### 5. Maximum Subarray Sum (Kadane's Algorithm).
**Answer:**
*   `current_sum = max(num, current_sum + num)`.
*   `max_sum = max(max_sum, current_sum)`.
*   Time: $O(N)$.

### 6. Two Sum.
**Answer:**
*   Use a **Hash Map** (Value -> Index).
*   Iterate `x`. Check if `target - x` exists in map.
*   Time: $O(N)$. Space: $O(N)$.

### 7. Group Anagrams.
**Answer:**
*   Map: `SortedString -> List[String]`.
*   Sort each string ("eat" -> "aet"). Use as key.
*   Time: $O(N \cdot K \log K)$.

### 8. Longest Substring Without Repeating Characters.
**Answer:**
*   **Sliding Window**.
*   Use Set/Map to track chars in window.
*   If duplicate, shrink window from left.
*   Time: $O(N)$.

### 9. Top K Frequent Elements.
**Answer:**
*   Count frequencies (Map).
*   Use **Min-Heap** of size K.
*   Or **Bucket Sort** (Frequency as index).
*   Time: $O(N \log K)$ or $O(N)$.

### 10. Merge Intervals.
**Answer:**
*   Sort by start time.
*   Iterate. If `current.start <= previous.end`, merge.
*   Else, add new interval.
*   Time: $O(N \log N)$.

### 11. Search in Rotated Sorted Array.
**Answer:**
*   **Binary Search**.
*   Determine which half is sorted.
*   Check if target is in that half.

### 12. Number of Islands (Graph DFS/BFS).
**Answer:**
*   Iterate grid. If '1', increment count and trigger DFS/BFS to sink island (mark '0').
*   Time: $O(M \times N)$.

### 13. Clone Graph.
**Answer:**
*   DFS/BFS with a Hash Map (`OriginalNode -> ClonedNode`) to avoid cycles.

### 14. Lowest Common Ancestor of BST.
**Answer:**
*   If `p` and `q` both < root, go left.
*   If `p` and `q` both > root, go right.
*   Else, root is LCA.

### 15. Validate Binary Search Tree.
**Answer:**
*   DFS with range `(min, max)`.
*   Left child must be in `(min, root.val)`.
*   Right child must be in `(root.val, max)`.

### 16. Serialize and Deserialize Binary Tree.
**Answer:**
*   **Preorder Traversal** (Root, Left, Right).
*   Use `None` (or `#`) for nulls.

### 17. Find Median from Data Stream.
**Answer:**
*   Two Heaps: **Max-Heap** (Left half), **Min-Heap** (Right half).
*   Balance sizes. Median is top of heap(s).
*   Time: $O(\log N)$ per add.

### 18. LRU Cache.
**Answer:**
*   **Hash Map** + **Doubly Linked List**.
*   Map: `Key -> Node`.
*   List: Most Recently Used at head.
*   $O(1)$ get and put.

### 19. Trapping Rain Water.
**Answer:**
*   Two Pointers (Left, Right).
*   Track `left_max` and `right_max`.
*   Water = `min(left_max, right_max) - height`.
*   Time: $O(N)$.

### 20. Word Search (Grid).
**Answer:**
*   DFS (Backtracking).
*   Mark visited. Revert after recursion.
