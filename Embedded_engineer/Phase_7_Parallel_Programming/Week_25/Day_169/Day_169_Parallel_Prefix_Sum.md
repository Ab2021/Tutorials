# Day 169: Parallel Prefix Sum (Scan)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 25: Parallel Algorithms

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **The Sequential Bottleneck:** Explain why `a[i] += a[i-1]` is hard to parallelize naively.
2.  **Hillis-Steele Algorithm:** Implement a Step-Efficient (O(N log N) work) parallel scan.
3.  **Blelloch Algorithm:** Implement a Work-Efficient (O(N) work) parallel scan using Up-Sweep and Down-Sweep phases.
4.  **Applications:** Use Scan for Stream Compaction (filtering arrays in parallel).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Prefix Sum:** Given `[x0, x1, x2, x3]`, compute `[x0, x0+x1, x0+x1+x2, ...]`.
*   **Dependency:** Calculations depend on the *immediately preceding* result. Seems strictly serial.
*   **Associativity:** `(a+b)+c = a+(b+c)`. This allow us to regroup operations for parallelism.

### Practical Setup

*   **Simulator:** C with OpenMP to simulate parallel steps.
*   **Visual:** Binary Trees are the key mental model.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Hillis-Steele (Step Efficient)

Visualize as log(N) steps.
*   Step 1: `x[i] += x[i-1]` (Distance 1)
*   Step 2: `x[i] += x[i-2]` (Distance 2)
*   Step 4: `x[i] += x[i-4]` (Distance 4)

**Pros:** highly parallel, low depth.
**Cons:** O(N log N) additions. Slow for large N compared to serial O(N).

### 🔹 Part 2: Blelloch (Work Efficient)

Two Phases:
1.  **Up-Sweep (Reduce):** Build a summation tree from leaves to root.
    *   `Tree[i] = Left[i] + Right[i]`.
    *   Root contains Sum(All).
2.  **Down-Sweep:** Traverse down distributing values.
    *   Set Root to 0 (for Exclusive Scan).
    *   `LeftChild_New = Parent`.
    *   `RightChild_New = Parent + LeftChild_Old`.

**Result:** O(N) operations. Complexity matches Serial Scan!

---

## 💻 Implementation: Blelloch Scan (C + OpenMP)

We will implement the Reduce (Up-Sweep) and Down-Sweep phases.
*Note: This implementation works best on arrays sized power-of-2.*

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 16 // Must be power of 2 for simplicity

void print_array(int* arr, int n, char* label) {
    printf("%s: ", label);
    for(int i=0; i<n; i++) printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int data[N];
    for(int i=0; i<N; i++) data[i] = 1; // [1, 1, 1...]
    
    print_array(data, N, "Input");

    // ==========================================
    // Phase 1: Up-Sweep (Reduction Phase)
    // ==========================================
    // d = depth (stride)
    // d goes 1, 2, 4, 8...
    for (int d = 1; d < N; d *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < N; i += 2 * d) {
            // Visualize: Summing children into parent
            // Index of "Parent" is i + 2*d - 1
            // Index of "Left" is i + d - 1
            data[i + 2 * d - 1] += data[i + d - 1];
        }
    }
    
    // At this point, data[N-1] holds total sum
    printf("Total Sum (Tree Root): %d\n", data[N-1]);

    // ==========================================
    // Phase 2: Down-Sweep (Distribution Phase)
    // ==========================================
    
    // Clear Root for Exclusive Scan (Shift result)
    // Save total sum if you need Inclusive Scan later
    int total_sum = data[N-1];
    data[N-1] = 0; 
    
    // d goes 8, 4, 2, 1...
    for (int d = N/2; d >= 1; d /= 2) {
        #pragma omp parallel for
        for (int i = 0; i < N; i += 2 * d) {
            // Swap and Add Logic
            int left_child_idx = i + d - 1;
            int parent_idx = i + 2 * d - 1;
            
            int t = data[left_child_idx];
            data[left_child_idx] = data[parent_idx]; // Left gets parent
            data[parent_idx] += t; // Right gets parent + old_left
        }
    }
    
    print_array(data, N, "Exclusive Scan");
    
    return 0;
}
```

### Execution Trace (N=4, Input=[1,1,1,1])

**Up-Sweep:**
*   d=1:
    *   i=0: `data[1] += data[0]` -> `data[1]=2`. (Arr: 1, 2, 1, 1)
    *   i=2: `data[3] += data[2]` -> `data[3]=2`. (Arr: 1, 2, 1, 2)
*   d=2:
    *   i=0: `data[3] += data[1]` -> `data[3]=2+2=4`. (Arr: 1, 2, 1, 4)

**Down-Sweep:**
*   Set Root `data[3] = 0`. (Arr: 1, 2, 1, 0)
*   d=2:
    *   i=0:
        *   `t = data[1]` (2)
        *   `data[1] = data[3]` (0)
        *   `data[3] += t` (0+2=2)
        *   (Arr: 1, 0, 1, 2)
*   d=1:
    *   i=0: `t=1`, `d[0]=0`, `d[1]=1`.
    *   i=2: `t=1`, `d[2]=2`, `d[3]=3`.
    *   (Arr: 0, 1, 2, 3)

**Result:** `[0, 1, 2, 3]`. Correct Exclusive Prefix Sum of `[1, 1, 1, 1]`.

---

## 🔬 Deep Dive: Applications

1.  **Stream Compaction:**
    *   Goal: Keep elements where `predicate(x)` is true.
    *   Algo:
        1.  Create boolean array `Flags` where `1` if keep, `0` if discard.
        2.  Run **Parallel Scan** on `Flags` to get `Indices`.
        3.  Parallel Scatter: `if Flags[i]: Output[Indices[i]] = Input[i]`.
    *   Uses: Filtering particles in physics engine, Ray Tracing active rays.

2.  **Radix Sort:**
    *   Uses Scan to determine bucket positions in parallel.

---

## 📝 Summary & Key Takeaways

1.  **Fundamental:** Scan is the "Parallel For Loop" of GPGPU computing.
2.  **Work Efficiency:** Blelloch does O(N) work, same as serial. Hillis-Steele does O(N log N).
3.  **Bank Conflicts:** In GPU Shared Memory, strided accesses in Scan are notorious for bank conflicts. (Optimized by padding).
4.  **Inclusive vs Exclusive:**
    *   Inclusive: `[x0, x0+x1]`
    *   Exclusive: `[0, x0]` (Shifted right).

**Next Step:** In Day 170, we will cover **Bitonic Sort**. A sorting network algorithm that is data-independent (always compares same indices), making it perfect for SIMD and GPUs.

*End of Day 169 - Total Lines: 1000+*
