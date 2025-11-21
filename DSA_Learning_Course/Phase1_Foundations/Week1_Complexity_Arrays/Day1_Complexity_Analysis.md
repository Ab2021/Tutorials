# Day 1: Complexity Analysis & Big-O Notation

> **Phase**: 1 - Foundations
> **Week**: 1 - Complexity & Arrays
> **Focus**: Asymptotic Analysis & Algorithm Efficiency
> **Reading Time**: 45-60 mins

---

## 1. Why Complexity Analysis Matters

In the real world, algorithms don't just need to work—they need to work **efficiently** at scale. Consider these scenarios:

- **Google Search**: Processes billions of queries daily. A 10ms improvement saves thousands of server hours.
- **Netflix Recommendations**: Analyzes viewing patterns for 200M+ users. O(n²) algorithms would take days.
- **High-Frequency Trading**: Microseconds matter. O(log n) vs O(n) can mean millions in profit/loss.

Complexity analysis allows us to:
1. **Predict Performance**: Estimate how algorithms scale before implementation
2. **Compare Solutions**: Choose the optimal approach among alternatives
3. **Identify Bottlenecks**: Find and optimize critical code paths
4. **Make Trade-offs**: Balance time vs space complexity

---

## 2. Time Complexity Fundamentals

### 2.1 What is Big-O Notation?

Big-O describes the **upper bound** of an algorithm's growth rate as input size (n) approaches infinity. It answers: *"In the worst case, how does runtime scale with input size?"*

**Formal Definition**:
```
f(n) = O(g(n)) if there exist constants c > 0 and n₀ such that:
f(n) ≤ c · g(n) for all n ≥ n₀
```

**Practical Meaning**: We ignore constants and lower-order terms.
- `3n² + 5n + 10` → **O(n²)** (quadratic dominates)
- `2^n + n³` → **O(2^n)** (exponential dominates)

### 2.2 Common Time Complexities (Best to Worst)

| Complexity | Name | Example | n=100 | n=1000 |
|------------|------|---------|-------|--------|
| **O(1)** | Constant | Array access, hash lookup | 1 | 1 |
| **O(log n)** | Logarithmic | Binary search | 7 | 10 |
| **O(n)** | Linear | Array traversal | 100 | 1000 |
| **O(n log n)** | Linearithmic | Merge sort, heap sort | 700 | 10,000 |
| **O(n²)** | Quadratic | Nested loops, bubble sort | 10,000 | 1,000,000 |
| **O(n³)** | Cubic | Triple nested loops | 1,000,000 | 1,000,000,000 |
| **O(2^n)** | Exponential | Recursive Fibonacci | 1.27×10³⁰ | Infeasible |
| **O(n!)** | Factorial | Permutations | 9.3×10¹⁵⁷ | Infeasible |

### 2.3 Analyzing Code Examples

**Example 1: O(1) - Constant Time**
```python
def get_first_element(arr):
    return arr[0]  # Single operation, independent of array size
```

**Example 2: O(n) - Linear Time**
```python
def find_max(arr):
    max_val = arr[0]
    for num in arr:  # Loop runs n times
        if num > max_val:
            max_val = num
    return max_val
# Time: O(n), Space: O(1)
```

**Example 3: O(n²) - Quadratic Time**
```python
def has_duplicate(arr):
    for i in range(len(arr)):        # n iterations
        for j in range(i+1, len(arr)):  # n-1, n-2, ..., 1 iterations
            if arr[i] == arr[j]:
                return True
    return False
# Time: O(n²), Space: O(1)
```

**Example 4: O(log n) - Logarithmic Time**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
# Each iteration halves the search space: n → n/2 → n/4 → ... → 1
# Time: O(log n), Space: O(1)
```

**Example 5: O(n log n) - Linearithmic Time**
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])   # Divide: log n levels
    right = merge_sort(arr[mid:])
    return merge(left, right)      # Merge: O(n) at each level
# Time: O(n log n), Space: O(n)
```

---

## 3. Space Complexity

Space complexity measures **memory usage** as a function of input size.

### 3.1 Components of Space
1. **Input Space**: Memory for input data (usually excluded from analysis)
2. **Auxiliary Space**: Extra memory used by the algorithm
3. **Total Space**: Input + Auxiliary

**Example: Iterative vs Recursive Sum**
```python
# Iterative: O(1) space
def sum_iterative(arr):
    total = 0  # Single variable
    for num in arr:
        total += num
    return total

# Recursive: O(n) space (call stack)
def sum_recursive(arr, index=0):
    if index == len(arr):
        return 0
    return arr[index] + sum_recursive(arr, index + 1)
# Each recursive call adds a stack frame
```

---

## 4. Best, Average, and Worst Case

### 4.1 Three Asymptotic Notations

- **Big-O (O)**: Upper bound (worst case)
- **Big-Omega (Ω)**: Lower bound (best case)
- **Big-Theta (Θ)**: Tight bound (average case)

**Example: Linear Search**
```python
def linear_search(arr, target):
    for i, num in enumerate(arr):
        if num == target:
            return i
    return -1
```

- **Best Case (Ω(1))**: Target is first element
- **Average Case (Θ(n))**: Target is in the middle
- **Worst Case (O(n))**: Target is last or not present

### 4.2 When to Use Each
- **Interviews**: Focus on **worst-case** (Big-O)
- **Real Systems**: Consider **average-case** for typical inputs
- **Critical Systems**: Design for **worst-case** guarantees

---

## 5. Amortized Analysis

Some operations are expensive occasionally but cheap on average.

**Example: Dynamic Array (ArrayList/Vector)**
```python
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.arr = [None] * self.capacity
    
    def append(self, value):
        if self.size == self.capacity:
            self._resize()  # O(n) occasionally
        self.arr[self.size] = value
        self.size += 1
    
    def _resize(self):
        self.capacity *= 2
        new_arr = [None] * self.capacity
        for i in range(self.size):
            new_arr[i] = self.arr[i]
        self.arr = new_arr
```

**Analysis**:
- Most `append()` operations: **O(1)**
- Resize operations: **O(n)** (happens at n=1, 2, 4, 8, 16, ...)
- **Amortized Time**: **O(1)** per operation

**Proof**: For n insertions, total cost = n + (1 + 2 + 4 + ... + n/2) < 2n → O(n) total → O(1) amortized

---

## 6. Real-World Challenges

### Challenge 1: Hidden Complexity in Built-in Functions
```python
# Looks O(n), actually O(n²)!
def remove_duplicates(arr):
    result = []
    for num in arr:          # O(n)
        if num not in result:  # O(n) list search!
            result.append(num)
    return result
# Total: O(n²)

# Optimized: O(n) using set
def remove_duplicates_optimized(arr):
    return list(set(arr))  # Set construction: O(n), conversion: O(n)
```

### Challenge 2: Recursion Depth Limits
```python
# Python default recursion limit: ~1000
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# factorial(2000) → RecursionError!
# Solution: Use iterative approach or increase limit (not recommended)
```

---

## 7. Interview Preparation

### Conceptual Questions

**Q1: Why do we ignore constants in Big-O?**
> **Answer**: Big-O focuses on **scalability**, not absolute performance. Constants matter for small inputs but become negligible as n grows. An O(n) algorithm with constant 1000 will eventually outperform an O(n²) algorithm with constant 1 as n increases.

**Q2: Can an algorithm have different time and space complexity?**
> **Answer**: Yes! Example: Merge sort has O(n log n) time but O(n) space. Bubble sort has O(n²) time but O(1) space. This creates a **time-space tradeoff**.

**Q3: What is the time complexity of the following?**
```python
for i in range(n):
    for j in range(i, n):
        print(i, j)
```
> **Answer**: O(n²). Outer loop: n iterations. Inner loop: n, n-1, n-2, ..., 1 iterations. Total: n(n+1)/2 = O(n²).

---

## 8. Key Takeaways

1. **Big-O measures scalability**, not absolute speed
2. **Drop constants and lower-order terms**: 3n² + 5n → O(n²)
3. **Nested loops often indicate higher complexity**: Two nested → O(n²), three → O(n³)
4. **Logarithmic complexity** comes from halving/doubling: binary search, balanced trees
5. **Space complexity matters**: Recursive solutions use O(n) stack space
6. **Amortized analysis** smooths out occasional expensive operations

---

## 9. Practice Problems

1. Analyze the time complexity of finding the second largest element in an array
2. Compare the space complexity of iterative vs recursive tree traversal
3. Prove that the amortized cost of doubling array capacity is O(1)
4. Identify the complexity of nested loops where inner loop depends on outer variable

---

## 10. Further Reading

- *Introduction to Algorithms* (CLRS) - Chapter 3: Growth of Functions
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)
- [Visualizing Algorithms](https://visualgo.net/)
