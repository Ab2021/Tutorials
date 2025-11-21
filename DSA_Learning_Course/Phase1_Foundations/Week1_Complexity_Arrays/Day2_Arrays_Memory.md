# Day 2: Arrays & Memory Management

> **Phase**: 1 - Foundations
> **Week**: 1 - Complexity & Arrays  
> **Focus**: Array internals, memory layout, cache optimization
> **Reading Time**: 40-50 mins

---

## 1. Array Fundamentals

### 1.1 What is an Array?

An array is a **contiguous block of memory** storing elements of the same type. This simple definition has profound implications:

- **Random Access**: O(1) access to any element via index
- **Cache Friendly**: Sequential elements are adjacent in memory
- **Fixed Size** (static arrays): Size determined at creation
- **Type Homogeneous**: All elements have the same size

**Memory Layout**:
```
Array: [10, 20, 30, 40, 50]
Memory: [10][20][30][40][50]
Address: 1000 1004 1008 1012 1016 (assuming 4-byte integers)

Element at index i: base_address + (i * element_size)
arr[3] = *(1000 + 3*4) = *(1012) = 40
```

### 1.2 Static vs Dynamic Arrays

**Static Array** (C/C++):
```cpp
int arr[5] = {1, 2, 3, 4, 5};  // Stack-allocated, fixed size
```

**Dynamic Array** (Python list, Java ArrayList, C++ vector):
```python
arr = []  # Starts with capacity 0
arr.append(1)  # Grows automatically
```

**Key Difference**: Dynamic arrays resize when full (typically doubling capacity).

---

## 2. Dynamic Array Implementation

### 2.1 How Python Lists Work

```python
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.arr = self._make_array(self.capacity)
    
    def _make_array(self, capacity):
        return [None] * capacity
    
    def append(self, value):
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        self.arr[self.size] = value
        self.size += 1
    
    def _resize(self, new_capacity):
        new_arr = self._make_array(new_capacity)
        for i in range(self.size):
            new_arr[i] = self.arr[i]
        self.arr = new_arr
        self.capacity = new_capacity
    
    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError('Index out of bounds')
        return self.arr[index]
```

**Complexity Analysis**:
- `append()`: **O(1) amortized** (occasional O(n) resize)
- `__getitem__()`: **O(1)**
- Space: **O(n)** (may have unused capacity)

### 2.2 Resize Strategies

| Strategy | Growth | Wasted Space | Amortized Time |
|----------|--------|--------------|----------------|
| **Add 1** | n+1 | 0% | O(n) |
| **Add k** | n+k | ~0% | O(n/k) |
| **Double** | 2n | ~50% | O(1) |
| **1.5x** | 1.5n | ~33% | O(1) |

**Why Doubling?**
- Amortized O(1) append
- Reasonable space overhead
- Industry standard (Python, Java, C++)

---

## 3. Array Operations Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| Access `arr[i]` | O(1) | Direct memory address calculation |
| Search (unsorted) | O(n) | Must check each element |
| Search (sorted) | O(log n) | Binary search |
| Insert at end | O(1) amortized | May trigger resize |
| Insert at index i | O(n) | Must shift elements |
| Delete at index i | O(n) | Must shift elements |
| Delete at end | O(1) | Just decrement size |

### 3.1 Insertion Example

```python
def insert_at_index(arr, index, value):
    # Shift elements right
    arr.append(None)  # Make space
    for i in range(len(arr)-1, index, -1):
        arr[i] = arr[i-1]
    arr[index] = value

# Example: Insert 99 at index 2
# Before: [10, 20, 30, 40]
# After:  [10, 20, 99, 30, 40]
# Operations: Shift 30→, 40→, then insert 99
# Time: O(n)
```

### 3.2 Deletion Example

```python
def delete_at_index(arr, index):
    # Shift elements left
    for i in range(index, len(arr)-1):
        arr[i] = arr[i+1]
    arr.pop()  # Remove last element

# Example: Delete index 1
# Before: [10, 20, 30, 40]
# After:  [10, 30, 40]
# Operations: Shift 30←, 40←, then remove last
# Time: O(n)
```

---

## 4. Memory Layout & Cache

### 4.1 CPU Cache Hierarchy

Modern CPUs have multiple cache levels:

| Cache | Size | Latency | Speed |
|-------|------|---------|-------|
| **L1** | 32-64 KB | ~1 ns | Fastest |
| **L2** | 256 KB - 1 MB | ~3-10 ns | Fast |
| **L3** | 8-32 MB | ~10-20 ns | Moderate |
| **RAM** | 8-64 GB | ~100 ns | Slow |

**Cache Line**: CPUs fetch memory in chunks (typically 64 bytes).
- Accessing `arr[0]` loads `arr[0]` through `arr[15]` (for 4-byte ints)
- Subsequent accesses to `arr[1]`, `arr[2]`, ... are **cache hits** (fast)

### 4.2 Cache-Friendly Code

**Good: Sequential Access**
```python
total = 0
for i in range(len(arr)):
    total += arr[i]  # Cache-friendly
# CPU prefetches next cache line
```

**Bad: Random Access**
```python
import random
indices = list(range(len(arr)))
random.shuffle(indices)
total = 0
for i in indices:
    total += arr[i]  # Cache misses!
```

**Performance**: Sequential can be **10-100x faster** for large arrays.

---

## 5. Multi-Dimensional Arrays

### 5.1 Row-Major vs Column-Major

**Row-Major** (C, C++, Python, Java):
```
Matrix:     Memory Layout:
[1 2 3]     [1][2][3][4][5][6][7][8][9]
[4 5 6]
[7 8 9]

Address of matrix[i][j] = base + (i * num_cols + j) * element_size
```

**Column-Major** (Fortran, MATLAB):
```
Matrix:     Memory Layout:
[1 2 3]     [1][4][7][2][5][8][3][6][9]
[4 5 6]
[7 8 9]
```

### 5.2 Traversal Performance

**Python (Row-Major)**:
```python
# FAST: Row-major traversal
matrix = [[0]*1000 for _ in range(1000)]
for row in range(1000):
    for col in range(1000):
        matrix[row][col] += 1  # Sequential memory access

# SLOW: Column-major traversal
for col in range(1000):
    for row in range(1000):
        matrix[row][col] += 1  # Jumps between rows (cache misses)
```

**Benchmark** (1000x1000 matrix):
- Row-major: ~50 ms
- Column-major: ~500 ms (10x slower!)

---

## 6. Array Tricks & Optimizations

### 6.1 In-Place Algorithms

Modify array without extra space.

**Example: Reverse Array**
```python
def reverse_in_place(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
# Time: O(n), Space: O(1)
```

### 6.2 Sentinel Values

Avoid boundary checks in loops.

**Without Sentinel**:
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

**With Sentinel**:
```python
def linear_search_sentinel(arr, target):
    arr.append(target)  # Sentinel
    i = 0
    while arr[i] != target:
        i += 1
    arr.pop()  # Remove sentinel
    return i if i < len(arr) else -1
# Eliminates one comparison per iteration
```

### 6.3 Prefix Sum Array

Precompute cumulative sums for range queries.

**Problem**: Find sum of elements from index `l` to `r` (multiple queries).

**Naive: O(n) per query**
```python
def range_sum_naive(arr, l, r):
    return sum(arr[l:r+1])
```

**Optimized: O(1) per query, O(n) preprocessing**
```python
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i+1] = prefix[i] + arr[i]
    return prefix

def range_sum_fast(prefix, l, r):
    return prefix[r+1] - prefix[l]

# Example:
# arr = [1, 2, 3, 4, 5]
# prefix = [0, 1, 3, 6, 10, 15]
# sum(arr[1:4]) = sum([2,3,4]) = 9
# prefix[4] - prefix[1] = 10 - 1 = 9 ✓
```

---

## 7. Real-World Applications

### 7.1 Image Processing

Images are 2D arrays of pixels.

```python
# Grayscale image: 2D array of intensities (0-255)
image = [[0]*width for _ in range(height)]

# Apply blur (average of neighbors)
def blur(image):
    h, w = len(image), len(image[0])
    result = [[0]*w for _ in range(h)]
    for i in range(1, h-1):
        for j in range(1, w-1):
            result[i][j] = (
                image[i-1][j] + image[i+1][j] +
                image[i][j-1] + image[i][j+1] +
                image[i][j]
            ) // 5
    return result
```

### 7.2 Time Series Data

Stock prices, sensor readings, logs.

```python
# Moving average
def moving_average(prices, window):
    result = []
    window_sum = sum(prices[:window])
    result.append(window_sum / window)
    
    for i in range(window, len(prices)):
        window_sum += prices[i] - prices[i-window]
        result.append(window_sum / window)
    return result
# Time: O(n), not O(n*window)
```

---

## 8. Common Pitfalls

### 8.1 Shallow vs Deep Copy

```python
# Shallow copy (1D array)
arr1 = [1, 2, 3]
arr2 = arr1  # Reference copy!
arr2[0] = 99
print(arr1)  # [99, 2, 3] - MODIFIED!

# Correct: Deep copy
arr2 = arr1.copy()  # or arr1[:]
arr2[0] = 99
print(arr1)  # [1, 2, 3] - UNCHANGED

# 2D array trap
matrix1 = [[1, 2], [3, 4]]
matrix2 = matrix1.copy()  # Shallow copy of rows!
matrix2[0][0] = 99
print(matrix1)  # [[99, 2], [3, 4]] - MODIFIED!

# Correct: Deep copy
import copy
matrix2 = copy.deepcopy(matrix1)
```

### 8.2 Off-by-One Errors

```python
# Wrong: Misses last element
for i in range(len(arr) - 1):
    print(arr[i])

# Correct
for i in range(len(arr)):
    print(arr[i])

# Or use iteration
for element in arr:
    print(element)
```

---

## 9. Key Takeaways

1. Arrays provide **O(1) random access** via contiguous memory
2. Dynamic arrays use **amortized O(1) append** via doubling strategy
3. **Insertion/deletion** in middle is O(n) due to shifting
4. **Cache locality** makes sequential access 10-100x faster than random
5. **Row-major traversal** is critical for performance in Python/C/Java
6. **Prefix sums** enable O(1) range queries after O(n) preprocessing

---

## 10. Practice Problems

1. Implement a dynamic array from scratch with `append`, `insert`, `delete`
2. Rotate an array to the right by k positions in O(n) time, O(1) space
3. Find the maximum sum of a contiguous subarray (Kadane's algorithm)
4. Merge two sorted arrays in O(n+m) time

---

**Next**: [Day 3: Two Pointers & Sliding Window](Day3_Two_Pointers.md)
