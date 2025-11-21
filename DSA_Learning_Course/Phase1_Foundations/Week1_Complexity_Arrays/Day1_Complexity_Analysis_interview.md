# Day 1: Complexity Analysis - Interview Preparation

> **Focus**: Common interview questions and problem-solving patterns
> **Difficulty**: Easy to Medium
> **Time**: 60-90 mins

---

## Interview Question Patterns

### Pattern 1: Analyzing Given Code

**Q1: What is the time complexity of this function?**
```python
def mystery_function(n):
    result = 0
    i = 1
    while i < n:
        j = 0
        while j < i:
            result += 1
            j += 1
        i *= 2
    return result
```

**Solution**:
- Outer loop: `i` takes values 1, 2, 4, 8, ..., n/2 → **log n iterations**
- Inner loop at iteration k: runs `2^k` times
- Total: 1 + 2 + 4 + ... + n/2 = n - 1
- **Answer: O(n)**

---

**Q2: Time and space complexity?**
```python
def generate_subsets(arr):
    if not arr:
        return [[]]
    first = arr[0]
    rest_subsets = generate_subsets(arr[1:])
    new_subsets = [[first] + subset for subset in rest_subsets]
    return rest_subsets + new_subsets
```

**Solution**:
- Generates all 2^n subsets
- Each subset copied: O(n) per subset
- **Time: O(n · 2^n)**
- Recursion depth: n, each level stores subsets
- **Space: O(n · 2^n)**

---

### Pattern 2: Optimizing Algorithms

**Q3: Optimize this duplicate detection**
```python
# Given: O(n²) solution
def has_duplicate_slow(arr):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False
```

**Solution 1: Sorting - O(n log n) time, O(1) space**
```python
def has_duplicate_sort(arr):
    arr.sort()
    for i in range(len(arr) - 1):
        if arr[i] == arr[i+1]:
            return True
    return False
```

**Solution 2: Hash Set - O(n) time, O(n) space**
```python
def has_duplicate_hash(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add(num)
    return False
```

**Follow-up**: What if we can't use extra space?
**Answer**: Sorting modifies input but uses O(1) extra space. If input must be preserved, O(n²) is optimal for O(1) space.

---

**Q4: Find two numbers that sum to target**
```python
# Brute force: O(n²)
def two_sum_brute(arr, target):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] + arr[j] == target:
                return [i, j]
    return None
```

**Optimized: O(n) time, O(n) space**
```python
def two_sum_hash(arr, target):
    seen = {}
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None
```

**Follow-up**: What if the array is sorted?
**Answer**: Use two pointers - O(n) time, O(1) space
```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None
```

---

### Pattern 3: Recurrence Analysis

**Q5: Analyze this recursive function**
```python
def mystery(n):
    if n <= 1:
        return 1
    return mystery(n-1) + mystery(n-1)
```

**Solution**:
- Recurrence: `T(n) = 2T(n-1) + O(1)`
- Expansion: `T(n) = 2·2·T(n-2) = 4T(n-2) = 2^k·T(n-k)`
- Base case when `n-k = 1`: `k = n-1`
- **Time: O(2^n)**
- **Space: O(n)** (recursion depth)

**Optimization**: Memoization
```python
def mystery_memo(n, memo={}):
    if n <= 1:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = mystery_memo(n-1, memo) + mystery_memo(n-1, memo)
    return memo[n]
# Time: O(n), Space: O(n)
```

---

**Q6: Merge K sorted arrays**
```python
# Given: K arrays, each of size N
# Total elements: K*N

# Approach 1: Merge pairs repeatedly
def merge_k_arrays_pairwise(arrays):
    while len(arrays) > 1:
        merged = []
        for i in range(0, len(arrays), 2):
            if i+1 < len(arrays):
                merged.append(merge_two(arrays[i], arrays[i+1]))
            else:
                merged.append(arrays[i])
        arrays = merged
    return arrays[0]

# Time: O(NK log K) - log K levels, each merges NK elements
# Space: O(NK)
```

**Approach 2: Min-Heap**
```python
import heapq

def merge_k_arrays_heap(arrays):
    heap = []
    result = []
    
    # Initialize heap with first element from each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))  # (value, array_idx, elem_idx)
    
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    
    return result

# Time: O(NK log K) - NK elements, each heap op is O(log K)
# Space: O(K) - heap size
```

---

### Pattern 4: Space-Time Tradeoffs

**Q7: Find the first non-repeating character**
```python
# Approach 1: O(n²) time, O(1) space
def first_unique_char_brute(s):
    for i, char in enumerate(s):
        if s.count(char) == 1:  # O(n) for each char
            return i
    return -1

# Approach 2: O(n) time, O(1) space (26 letters)
def first_unique_char_hash(s):
    from collections import Counter
    counts = Counter(s)  # O(n)
    for i, char in enumerate(s):  # O(n)
        if counts[char] == 1:
            return i
    return -1
```

---

**Q8: Check if array contains duplicates within k distance**
```python
# Given: Return True if arr[i] == arr[j] and |i-j| <= k

# Approach 1: Brute force - O(nk) time, O(1) space
def contains_nearby_duplicate_brute(arr, k):
    for i in range(len(arr)):
        for j in range(i+1, min(i+k+1, len(arr))):
            if arr[i] == arr[j]:
                return True
    return False

# Approach 2: Sliding window with set - O(n) time, O(k) space
def contains_nearby_duplicate_set(arr, k):
    window = set()
    for i, num in enumerate(arr):
        if num in window:
            return True
        window.add(num)
        if len(window) > k:
            window.remove(arr[i-k])
    return False
```

---

## Behavioral Interview Questions

### Q9: "Describe a time you optimized an algorithm"

**Framework (STAR)**:
- **Situation**: Working on a recommendation system processing 1M users
- **Task**: Initial O(n²) similarity computation took 12 hours
- **Action**: 
  1. Profiled code, identified bottleneck
  2. Switched to locality-sensitive hashing (LSH)
  3. Reduced to O(n) expected time
- **Result**: Processing time reduced to 20 minutes, enabling real-time updates

---

### Q10: "How do you approach algorithm optimization?"

**Answer Structure**:
1. **Measure first**: Profile to find actual bottlenecks
2. **Analyze complexity**: Identify theoretical limits
3. **Consider tradeoffs**: Time vs space, accuracy vs speed
4. **Apply patterns**: Hashing, sorting, two pointers, etc.
5. **Validate**: Benchmark and test edge cases

---

## System Design Integration

### Q11: "Design a rate limiter - what data structure and why?"

**Answer**:
- **Requirement**: Allow N requests per minute per user
- **Data Structure**: Hash map + Sliding window (queue/deque)

```python
from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = {}  # user_id -> deque of timestamps
    
    def allow_request(self, user_id):
        now = time.time()
        if user_id not in self.requests:
            self.requests[user_id] = deque()
        
        # Remove old requests outside window
        while self.requests[user_id] and \
              self.requests[user_id][0] <= now - self.window:
            self.requests[user_id].popleft()
        
        if len(self.requests[user_id]) < self.max_requests:
            self.requests[user_id].append(now)
            return True
        return False

# Time per request: O(max_requests) worst case, O(1) amortized
# Space: O(users * max_requests)
```

**Complexity Justification**:
- Hash map: O(1) user lookup
- Deque: O(1) append/popleft
- Cleanup: Amortized O(1) (each request removed once)

---

## Rapid-Fire Complexity Questions

1. **Accessing the middle element of a linked list?** O(n)
2. **Inserting at the beginning of an array?** O(n)
3. **Inserting at the beginning of a linked list?** O(1)
4. **Finding min in a min-heap?** O(1)
5. **Deleting min from a min-heap?** O(log n)
6. **Searching in a balanced BST?** O(log n)
7. **Searching in a hash table (average case)?** O(1)
8. **Searching in a hash table (worst case)?** O(n)
9. **DFS on a graph with V vertices, E edges?** O(V + E)
10. **Dijkstra's algorithm with binary heap?** O((V + E) log V)

---

## Take-Home Challenges

1. **Analyze**: Find time complexity of finding the k-th largest element using QuickSelect
2. **Optimize**: Given a function that computes Fibonacci, reduce space complexity to O(1)
3. **Design**: Implement a data structure supporting insert, delete, and getRandom in O(1) average time
4. **Prove**: Show that the amortized cost of table doubling is O(1)

---

## Key Interview Tips

1. **Always state assumptions**: "Assuming the array is sorted..." or "If we can use extra space..."
2. **Start with brute force**: Show you can solve it, then optimize
3. **Explain tradeoffs**: "This uses more space but runs faster because..."
4. **Consider edge cases**: Empty input, single element, duplicates
5. **Analyze before coding**: Discuss complexity before implementing
6. **Test your solution**: Walk through with an example

---

## Common Pitfalls

❌ **Forgetting hidden complexity**: `list.remove()`, `substring()`, `list.sort()`
❌ **Confusing best/average/worst case**: QuickSort is O(n²) worst, O(n log n) average
❌ **Ignoring space complexity**: Recursion uses O(depth) stack space
❌ **Premature optimization**: Solve correctly first, optimize second
❌ **Not considering constraints**: n ≤ 100 vs n ≤ 10^9 requires different approaches

---

**Next**: [Day 2: Arrays & Memory Management](Day2_Arrays_Memory.md)
