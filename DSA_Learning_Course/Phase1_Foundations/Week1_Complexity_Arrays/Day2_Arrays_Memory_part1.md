# Day 2: Arrays & Memory - Deep Dive

> **Advanced Topics**: SIMD, Memory Alignment, Bit Arrays
> **Reading Time**: 30-40 mins

---

## 1. SIMD (Single Instruction, Multiple Data)

Modern CPUs can process multiple array elements simultaneously.

### 1.1 Vectorization

**Scalar Code** (processes one element at a time):
```python
def add_arrays_scalar(a, b):
    result = [0] * len(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result
```

**Vectorized Code** (NumPy uses SIMD):
```python
import numpy as np

def add_arrays_vectorized(a, b):
    return np.array(a) + np.array(b)
# Can be 10-100x faster for large arrays
```

**Why It's Faster**:
- CPU processes 4-8 elements per instruction (AVX/SSE)
- Reduces loop overhead
- Better instruction pipelining

---

## 2. Memory Alignment

CPUs access memory most efficiently at aligned addresses.

### 2.1 Alignment Rules

- **1-byte** (char): Any address
- **2-byte** (short): Even addresses (0, 2, 4, ...)
- **4-byte** (int): Addresses divisible by 4
- **8-byte** (long/double): Addresses divisible by 8

**Unaligned Access Penalty**: 2-10x slower (or crashes on some architectures)

### 2.2 Struct Padding Example

```c
// Without padding awareness
struct BadLayout {
    char a;    // 1 byte
    int b;     // 4 bytes (needs 4-byte alignment)
    char c;    // 1 byte
};
// Actual size: 12 bytes (padding: a[1] + pad[3] + b[4] + c[1] + pad[3])

// Optimized layout
struct GoodLayout {
    int b;     // 4 bytes
    char a;    // 1 byte
    char c;    // 1 byte
};
// Actual size: 8 bytes (b[4] + a[1] + c[1] + pad[2])
```

---

## 3. Bit Arrays

Store boolean values efficiently using bits instead of bytes.

### 3.1 Implementation

```python
class BitArray:
    def __init__(self, size):
        self.size = size
        # Use integers to store bits (each int holds 32/64 bits)
        self.arr = [0] * ((size + 31) // 32)
    
    def set(self, index):
        arr_index = index // 32
        bit_index = index % 32
        self.arr[arr_index] |= (1 << bit_index)
    
    def clear(self, index):
        arr_index = index // 32
        bit_index = index % 32
        self.arr[arr_index] &= ~(1 << bit_index)
    
    def get(self, index):
        arr_index = index // 32
        bit_index = index % 32
        return (self.arr[arr_index] >> bit_index) & 1
```

**Space Savings**: 8x less memory than boolean array

---

## 4. Sparse Arrays

When most elements are zero/default, use sparse representation.

### 4.1 Dictionary-Based Sparse Array

```python
class SparseArray:
    def __init__(self, size, default=0):
        self.size = size
        self.default = default
        self.data = {}  # Only store non-default values
    
    def __getitem__(self, index):
        return self.data.get(index, self.default)
    
    def __setitem__(self, index, value):
        if value == self.default:
            self.data.pop(index, None)
        else:
            self.data[index] = value

# Example: 1 million element array with 100 non-zero values
# Dense: 1M * 8 bytes = 8 MB
# Sparse: 100 * (8 + 8) bytes ≈ 1.6 KB (5000x savings!)
```

---

## 5. Circular Buffers

Fixed-size buffer that wraps around.

### 5.1 Implementation

```python
class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
    
    def enqueue(self, value):
        if self.size == self.capacity:
            raise Exception("Buffer full")
        self.buffer[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
    
    def dequeue(self):
        if self.size == 0:
            raise Exception("Buffer empty")
        value = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return value
```

**Use Cases**: Audio/video streaming, network buffers, logging

---

## 6. Cache-Oblivious Algorithms

Algorithms that perform well regardless of cache size.

### 6.1 Matrix Transpose

**Cache-Aware (Blocked)**:
```python
def transpose_blocked(matrix, block_size=64):
    n = len(matrix)
    result = [[0]*n for _ in range(n)]
    
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            # Transpose block
            for ii in range(i, min(i+block_size, n)):
                for jj in range(j, min(j+block_size, n)):
                    result[jj][ii] = matrix[ii][jj]
    return result
```

**Cache-Oblivious (Recursive)**:
```python
def transpose_recursive(matrix, r1, r2, c1, c2, result):
    if r2 - r1 <= 16:  # Base case
        for i in range(r1, r2):
            for j in range(c1, c2):
                result[j][i] = matrix[i][j]
    else:
        rm = (r1 + r2) // 2
        cm = (c1 + c2) // 2
        # Divide into 4 quadrants
        transpose_recursive(matrix, r1, rm, c1, cm, result)
        transpose_recursive(matrix, r1, rm, cm, c2, result)
        transpose_recursive(matrix, rm, r2, c1, cm, result)
        transpose_recursive(matrix, rm, r2, cm, c2, result)
```

---

## 7. Key Takeaways

1. **SIMD** vectorization can provide 10-100x speedups
2. **Memory alignment** affects performance and correctness
3. **Bit arrays** save 8x space for boolean data
4. **Sparse arrays** efficient for mostly-empty data
5. **Circular buffers** enable efficient fixed-size queues
6. **Cache-oblivious algorithms** adapt to any cache hierarchy

---

**Further Reading**:
- [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
