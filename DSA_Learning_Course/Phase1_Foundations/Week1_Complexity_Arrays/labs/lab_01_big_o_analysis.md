# Lab 01: Big-O Analysis Practice

## Difficulty
🟢 **Easy**

## Estimated Time
30 minutes

## Learning Objectives
- Identify time complexity of code snippets
- Understand asymptotic notation (Big-O, Big-Θ, Big-Ω)
- Recognize common complexity patterns
- Simplify complexity expressions

## Prerequisites
- Day 1: Complexity Analysis & Big-O Notation
- Understanding of loops and recursion

## Problem Statement

Analyze the time complexity of various code snippets and express them in Big-O notation. For each function, determine:
1. The exact operation count as a function of `n`
2. The Big-O complexity (worst case)
3. The Big-Θ complexity (average case)
4. The Big-Ω complexity (best case)

## Code Snippets to Analyze

### Snippet 1: Simple Loop
```python
def snippet_1(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total
```

### Snippet 2: Nested Loops
```python
def snippet_2(n):
    count = 0
    for i in range(n):
        for j in range(n):
            count += 1
    return count
```

### Snippet 3: Logarithmic Pattern
```python
def snippet_3(n):
    i = 1
    count = 0
    while i < n:
        count += 1
        i *= 2
    return count
```

### Snippet 4: Mixed Complexity
```python
def snippet_4(arr):
    n = len(arr)
    # Part 1
    for i in range(n):
        print(arr[i])
    
    # Part 2
    for i in range(n):
        for j in range(n):
            print(arr[i] + arr[j])
    
    # Part 3
    for i in range(100):
        print("constant")
```

### Snippet 5: Dependent Nested Loops
```python
def snippet_5(n):
    count = 0
    for i in range(n):
        for j in range(i):
            count += 1
    return count
```

### Snippet 6: Recursive Function
```python
def snippet_6(n):
    if n <= 1:
        return 1
    return snippet_6(n - 1) + snippet_6(n - 1)
```

### Snippet 7: Optimized Search
```python
def snippet_7(arr, target):
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
```

### Snippet 8: String Operations
```python
def snippet_8(s):
    result = ""
    for char in s:
        result += char  # String concatenation
    return result
```

## Requirements

For each snippet, provide:
1. **Operation Count**: Express the number of operations as a function of n
2. **Big-O (Worst Case)**: Upper bound
3. **Big-Θ (Average Case)**: Tight bound
4. **Big-Ω (Best Case)**: Lower bound
5. **Justification**: Explain your reasoning

## Hints

<details>
<summary>Hint 1: Counting Operations</summary>

Count the number of times the innermost operation executes. For loops, multiply the iterations of each nested level.
</details>

<details>
<summary>Hint 2: Logarithmic Patterns</summary>

When a variable is multiplied/divided by a constant in each iteration, the complexity is logarithmic. If `i *= 2`, the loop runs log₂(n) times.
</details>

<details>
<summary>Hint 3: Dependent Loops</summary>

For `for i in range(n): for j in range(i):`, the inner loop runs 0, 1, 2, ..., n-1 times. Sum this arithmetic series: n(n-1)/2.
</details>

<details>
<summary>Hint 4: Recursive Complexity</summary>

Use recurrence relations. For snippet_6, T(n) = 2T(n-1) + O(1). Solve using the recursion tree method.
</details>

<details>
<summary>Hint 5: String Concatenation</summary>

In Python, string concatenation creates a new string object. Concatenating to a string of length k takes O(k) time.
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Snippet 1: Simple Loop

```python
def snippet_1(arr):
    total = 0              # O(1)
    for i in range(len(arr)):  # n iterations
        total += arr[i]    # O(1) per iteration
    return total           # O(1)
```

**Analysis**:
- **Operation Count**: n additions + constant operations = n + c
- **Big-O**: O(n) - linear in array size
- **Big-Θ**: Θ(n) - always performs n operations
- **Big-Ω**: Ω(n) - minimum n operations required

**Justification**: Single loop iterates n times, each iteration does constant work.

---

### Snippet 2: Nested Loops

```python
def snippet_2(n):
    count = 0              # O(1)
    for i in range(n):     # n iterations
        for j in range(n): # n iterations per outer iteration
            count += 1     # O(1)
    return count           # O(1)
```

**Analysis**:
- **Operation Count**: n × n = n²
- **Big-O**: O(n²) - quadratic
- **Big-Θ**: Θ(n²) - always n² operations
- **Big-Ω**: Ω(n²) - minimum n² operations

**Justification**: Nested loops, each running n times → n × n = n² operations.

---

### Snippet 3: Logarithmic Pattern

```python
def snippet_3(n):
    i = 1
    count = 0
    while i < n:
        count += 1
        i *= 2  # i doubles each iteration: 1, 2, 4, 8, ..., 2^k
    return count
```

**Analysis**:
- **Operation Count**: Loop runs until i ≥ n, where i = 2^k → k = log₂(n)
- **Big-O**: O(log n) - logarithmic
- **Big-Θ**: Θ(log n)
- **Big-Ω**: Ω(log n)

**Justification**: i doubles each iteration, so loop runs log₂(n) times.

---

### Snippet 4: Mixed Complexity

```python
def snippet_4(arr):
    n = len(arr)
    # Part 1: O(n)
    for i in range(n):
        print(arr[i])
    
    # Part 2: O(n²)
    for i in range(n):
        for j in range(n):
            print(arr[i] + arr[j])
    
    # Part 3: O(1)
    for i in range(100):
        print("constant")
```

**Analysis**:
- **Operation Count**: n + n² + 100
- **Big-O**: O(n²) - dominated by the quadratic term
- **Big-Θ**: Θ(n²)
- **Big-Ω**: Ω(n²)

**Justification**: When summing complexities, the highest order term dominates. O(n) + O(n²) + O(1) = O(n²).

---

### Snippet 5: Dependent Nested Loops

```python
def snippet_5(n):
    count = 0
    for i in range(n):     # i = 0, 1, 2, ..., n-1
        for j in range(i): # j runs 0, 1, 2, ..., i-1 times
            count += 1
    return count
```

**Analysis**:
- **Operation Count**: 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = (n² - n)/2
- **Big-O**: O(n²)
- **Big-Θ**: Θ(n²)
- **Big-Ω**: Ω(n²)

**Justification**: Sum of arithmetic series. Although it's n²/2, we drop constants in Big-O.

---

### Snippet 6: Recursive Function

```python
def snippet_6(n):
    if n <= 1:
        return 1
    return snippet_6(n - 1) + snippet_6(n - 1)
```

**Analysis**:
- **Recurrence Relation**: T(n) = 2T(n-1) + O(1)
- **Recursion Tree**: 
  - Level 0: 1 call
  - Level 1: 2 calls
  - Level 2: 4 calls
  - ...
  - Level n-1: 2^(n-1) calls
- **Total Calls**: 1 + 2 + 4 + ... + 2^(n-1) = 2^n - 1
- **Big-O**: O(2^n) - exponential
- **Big-Θ**: Θ(2^n)
- **Big-Ω**: Ω(2^n)

**Justification**: Each call makes 2 recursive calls, creating a binary tree of height n with 2^n leaves.

---

### Snippet 7: Binary Search

```python
def snippet_7(arr, target):
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
```

**Analysis**:
- **Operation Count**: Search space halves each iteration → log₂(n) iterations
- **Big-O**: O(log n)
- **Big-Θ**: Θ(log n) - average case
- **Big-Ω**: Ω(1) - best case (target at middle)

**Justification**: Classic binary search. Search space: n → n/2 → n/4 → ... → 1 (log n steps).

---

### Snippet 8: String Concatenation

```python
def snippet_8(s):
    result = ""
    for char in s:
        result += char  # Creates new string each time
    return result
```

**Analysis**:
- **Operation Count**: 
  - Iteration 1: Copy 0 chars + add 1 = 1
  - Iteration 2: Copy 1 char + add 1 = 2
  - Iteration 3: Copy 2 chars + add 1 = 3
  - ...
  - Iteration n: Copy (n-1) chars + add 1 = n
  - Total: 1 + 2 + 3 + ... + n = n(n+1)/2
- **Big-O**: O(n²) - quadratic!
- **Big-Θ**: Θ(n²)
- **Big-Ω**: Ω(n²)

**Justification**: String concatenation in Python creates a new string object, copying all existing characters. This makes it O(n²) for n concatenations.

**Better Approach**:
```python
def snippet_8_optimized(s):
    result = []
    for char in s:
        result.append(char)  # O(1) amortized
    return ''.join(result)   # O(n)
# Total: O(n)
```

</details>

## Extensions

1. **Challenge 1**: Analyze the complexity of the following:
   ```python
   def mystery(n):
       i = n
       while i > 0:
           j = i
           while j > 0:
               j //= 2
           i -= 1
   ```

2. **Challenge 2**: What is the complexity of this recursive function?
   ```python
   def fib(n):
       if n <= 1:
           return n
       return fib(n-1) + fib(n-2)
   ```

3. **Challenge 3**: Prove that O(n log n) is faster than O(n²) for large n. At what value of n does n log n become less than n²?

## Related Concepts
- [Day 1: Complexity Analysis](../Day1_Complexity_Analysis.md)
- [Day 1 Deep Dive: Amortized Analysis](../Day1_Complexity_Analysis_part1.md)
- [Lab 02: Time Complexity Calculation](lab_02_time_complexity.md)

## Key Takeaways

1. **Drop Constants**: O(2n) = O(n), O(n²/2) = O(n²)
2. **Drop Lower Terms**: O(n² + n) = O(n²)
3. **Different Inputs**: O(a + b) ≠ O(n) if a and b are independent
4. **Logarithms**: Base doesn't matter in Big-O (log₂ n = O(log n))
5. **Recursion**: Use recurrence relations or recursion tree method

## Common Complexities (Ranked)

From fastest to slowest:
1. O(1) - Constant
2. O(log n) - Logarithmic
3. O(n) - Linear
4. O(n log n) - Linearithmic
5. O(n²) - Quadratic
6. O(n³) - Cubic
7. O(2^n) - Exponential
8. O(n!) - Factorial

---

**Next**: [Lab 02: Time Complexity Calculation](lab_02_time_complexity.md)
