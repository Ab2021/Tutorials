# Day 1: Complexity Analysis - Deep Dive

> **Advanced Topics**: Master Theorem, Recurrence Relations, Amortized Analysis
> **Reading Time**: 30-40 mins

---

## 1. Recurrence Relations

Many algorithms are recursive, requiring recurrence analysis to determine complexity.

### 1.1 Common Recurrence Patterns

**Pattern 1: Divide and Conquer**
```
T(n) = aT(n/b) + f(n)
```
- `a`: Number of subproblems
- `n/b`: Size of each subproblem
- `f(n)`: Cost of divide/combine

**Examples**:
- Binary Search: `T(n) = T(n/2) + O(1)` → **O(log n)**
- Merge Sort: `T(n) = 2T(n/2) + O(n)` → **O(n log n)**
- Karatsuba Multiplication: `T(n) = 3T(n/2) + O(n)` → **O(n^1.58)**

### 1.2 The Master Theorem

For recurrences of form `T(n) = aT(n/b) + f(n)` where `a ≥ 1, b > 1`:

Let `c = log_b(a)`

**Case 1**: If `f(n) = O(n^d)` where `d < c` → **T(n) = Θ(n^c)**
**Case 2**: If `f(n) = Θ(n^c)` → **T(n) = Θ(n^c log n)**
**Case 3**: If `f(n) = Ω(n^d)` where `d > c` → **T(n) = Θ(f(n))**

**Example Applications**:
```python
# Binary Search: T(n) = T(n/2) + O(1)
# a=1, b=2, c=log₂(1)=0, f(n)=O(1)=O(n⁰)
# Case 2: T(n) = Θ(log n)

# Merge Sort: T(n) = 2T(n/2) + O(n)
# a=2, b=2, c=log₂(2)=1, f(n)=O(n)=O(n¹)
# Case 2: T(n) = Θ(n log n)

# Strassen Matrix Multiplication: T(n) = 7T(n/2) + O(n²)
# a=7, b=2, c=log₂(7)≈2.81, f(n)=O(n²)
# Case 1 (2 < 2.81): T(n) = Θ(n^2.81)
```

---

## 2. Advanced Amortized Analysis

### 2.1 Aggregate Method

Calculate total cost of n operations, divide by n.

**Example: Stack with MultiPop**
```python
class Stack:
    def push(self, x):
        self.items.append(x)  # O(1)
    
    def pop(self):
        return self.items.pop()  # O(1)
    
    def multipop(self, k):
        for _ in range(min(k, len(self.items))):
            self.pop()  # O(min(k, n))
```

**Analysis**: In n operations, each element pushed at most once and popped at most once.
- Total pushes: ≤ n
- Total pops: ≤ n
- Total cost: ≤ 2n → **O(1) amortized per operation**

### 2.2 Accounting Method

Assign different charges to operations, store credit for future expensive operations.

**Example: Binary Counter Increment**
```python
def increment(counter):
    i = 0
    while i < len(counter) and counter[i] == 1:
        counter[i] = 0  # Flip 1→0
        i += 1
    if i < len(counter):
        counter[i] = 1  # Flip 0→1
```

**Accounting**:
- Charge $2 per increment
- $1 pays for flipping 0→1
- $1 stored as credit on the bit
- When flipping 1→0, use stored credit
- **Amortized cost: O(1)**

### 2.3 Potential Method

Define potential function Φ(D) representing stored work.

**Amortized cost** = Actual cost + Δ Φ

**Example: Dynamic Array Doubling**
```python
Φ(D) = 2·size - capacity

# When not resizing:
# Actual cost = 1, Δ Φ = 2
# Amortized = 1 + 2 = 3

# When resizing (size = capacity):
# Actual cost = size, Δ Φ = 2·size - 2·size = 0
# Amortized = size + 0 = size (but happens rarely)
```

---

## 3. Space-Time Tradeoffs

### 3.1 Memoization vs Recomputation

**Fibonacci Without Memoization: O(2^n) time, O(n) space**
```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

**Fibonacci With Memoization: O(n) time, O(n) space**
```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

**Fibonacci Iterative: O(n) time, O(1) space**
```python
def fib_iter(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

### 3.2 Lookup Tables

**Example: Prime Checking**
```python
# Naive: O(√n) per query
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Sieve of Eratosthenes: O(n log log n) preprocessing, O(1) per query
def sieve(max_n):
    is_prime = [True] * (max_n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_n + 1, i):
                is_prime[j] = False
    return is_prime

# Precompute once, query many times
primes = sieve(1000000)
print(primes[999983])  # O(1)
```

---

## 4. Cache-Aware Algorithms

Modern CPUs have cache hierarchies (L1, L2, L3). **Cache-friendly code** accesses memory sequentially.

### 4.1 Row-Major vs Column-Major Access

**Slow (Column-Major in Row-Major Language)**:
```python
# Python/C/Java use row-major order
matrix = [[0]*1000 for _ in range(1000)]
for col in range(1000):
    for row in range(1000):
        matrix[row][col] += 1  # Cache miss every access!
```

**Fast (Row-Major)**:
```python
for row in range(1000):
    for col in range(1000):
        matrix[row][col] += 1  # Sequential access, cache hits
```

**Performance**: Row-major can be **10-100x faster** for large matrices.

### 4.2 Loop Tiling (Blocking)

**Matrix Multiplication Optimization**:
```python
# Standard: O(n³) but poor cache usage
def matmul_standard(A, B):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Tiled: Same O(n³) but better cache locality
def matmul_tiled(A, B, block_size=64):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i0 in range(0, n, block_size):
        for j0 in range(0, n, block_size):
            for k0 in range(0, n, block_size):
                for i in range(i0, min(i0+block_size, n)):
                    for j in range(j0, min(j0+block_size, n)):
                        for k in range(k0, min(k0+block_size, n)):
                            C[i][j] += A[i][k] * B[k][j]
    return C
```

---

## 5. Probabilistic Analysis

### 5.1 Expected vs Worst-Case Complexity

**QuickSort**:
- **Worst Case**: O(n²) (already sorted array with bad pivot)
- **Average Case**: O(n log n) (random pivots)
- **Randomized QuickSort**: O(n log n) expected (random pivot selection)

```python
import random

def quicksort_randomized(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)  # Randomization guarantees expected O(n log n)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_randomized(left) + mid + quicksort_randomized(right)
```

### 5.2 Las Vegas vs Monte Carlo Algorithms

- **Las Vegas**: Always correct, random runtime (e.g., Randomized QuickSort)
- **Monte Carlo**: Fixed runtime, probabilistically correct (e.g., Miller-Rabin primality test)

---

## 6. Complexity Classes (P, NP, NP-Complete)

### 6.1 Decision Problems

- **P**: Problems solvable in polynomial time (O(n^k))
- **NP**: Problems verifiable in polynomial time
- **NP-Complete**: Hardest problems in NP (if any is in P, all NP problems are)
- **NP-Hard**: At least as hard as NP-Complete (may not be in NP)

**Examples**:
- **P**: Sorting, shortest path, binary search
- **NP-Complete**: Traveling Salesman, Knapsack (decision version), SAT
- **NP-Hard**: Halting Problem, Optimization TSP

### 6.2 Practical Implications

For NP-Complete problems:
1. **Approximation Algorithms**: Get close to optimal (e.g., 2-approximation for TSP)
2. **Heuristics**: Greedy, local search (no guarantees)
3. **Exact for Small Inputs**: Backtracking, branch-and-bound
4. **Parameterized Complexity**: Efficient for specific parameter ranges

---

## 7. Modern Complexity Considerations (2025)

### 7.1 Parallel Complexity

- **Work**: Total operations (sequential complexity)
- **Span**: Longest dependency chain (parallel time)
- **Parallelism**: Work / Span

**Example: Parallel Sum**
```python
# Sequential: O(n) work, O(n) span
def sum_sequential(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Parallel (tree reduction): O(n) work, O(log n) span
def sum_parallel(arr):
    if len(arr) == 1:
        return arr[0]
    mid = len(arr) // 2
    left = sum_parallel(arr[:mid])   # Parallel
    right = sum_parallel(arr[mid:])  # Parallel
    return left + right
# Parallelism: n / log n
```

### 7.2 I/O Complexity

For data larger than RAM, disk I/O dominates.

**External Merge Sort**:
- **RAM**: M elements
- **Disk**: N elements (N >> M)
- **I/O Complexity**: O((N/M) log(N/M)) disk accesses
- **Time Complexity**: Still O(N log N) comparisons

---

## 8. Key Takeaways

1. **Master Theorem** solves most divide-and-conquer recurrences
2. **Amortized analysis** reveals true cost of operations with occasional spikes
3. **Cache locality** can provide 10-100x speedups without changing Big-O
4. **Space-time tradeoffs** are fundamental: memoization trades space for time
5. **Probabilistic analysis** explains why randomized algorithms work in practice
6. **NP-Completeness** identifies inherently hard problems requiring approximations

---

## 9. Further Reading

- *Algorithms* (Dasgupta, Papadimitriou, Vazirani) - Chapter 2: Divide and Conquer
- *The Algorithm Design Manual* (Skiena) - Chapter 4: Sorting and Searching
- [MIT 6.046J Lecture Notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/)
