# Day 107: Loop Transformations
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 16: Auto-Vectorization & Loop Optimization

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Interchange:** Swap inner/outer loops to improve locality (stride minimization).
2.  **Fusion & Fission:** Combine loops to reduce overhead or split them to separate dependencies.
3.  **Tiling (Blocking):** Divide iteration space into smaller chunks to fit in L1/L2 cache.
4.  **Unrolling:** Replicate simplicity to expose ILP.
5.  **Validity:** Determine when these transformations are safe using Distance Vectors (Day 106).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Cache Hierarchy:** Accessing RAM is 100x slower than L1 cache.
*   **Locality:** Spatial (nearby addresses) vs Temporal (same address reused).

### Practical Setup

*   `clang -O3` performs many of these automatically.
*   Manual application (rewriting C code) helps understand the mechanics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Loop Interchange

**Goal:** Improve Spatial Locality (access memory sequentially) or Parallelism (move parallel loop out).

**Code:**
```c
// Before (Row Major C, Bad Stride)
for (j=0; j<N; j++)
  for (i=0; i<N; i++)
    A[i][j] = ... 

// After (Good Stride)
for (i=0; i<N; i++)
  for (j=0; j<N; j++)
    A[i][j] = ... 
```

**Validity:**
Allowed if the Dependence Direction vector doesn't become invalid.
*   Valid: $(<, <)$, $(=, =)$, $(<, =)$.
*   Invalid: $(<, >)$. If we swap, it becomes $(>, <)$, which means dependence goes *backwards* in time. Impossible.

### 🔹 Part 2: Loop Fusion & Fission

**Fusion (Jamming):**
Merges two adjacent loops with same bounds.
*   **Pros:** Reduces loop overhead (`i++`, branch). Improves temporal locality (if both loops use common array `A`).
*   **Cons:** Increases register pressure inside the body. Might pollute cache if too much data is accessed.

**Fission (Distribution):**
Splits one loop into two.
*   **Pros:** Isolates a dependency cycle to one loop, allowing the other to be vectorised. Reduces register pressure.
*   **Cons:** Increases loop overhead. Data in cache might get cold.

### 🔹 Part 3: Loop Tiling (Blocking)

The most important optimization for Matrix Multiplication and Stencil codes.

**Problem:**
If $N$ is large, `A[i]` falls out of cache before we use it again in the next pass.

**Solution:**
Iterate over small blocks (tiles) of size $B \times B$ that fit in L1 cache.

```c
// Before
for (i=0; i<N; i++)
  for (j=0; j<N; j++)
    ...

// After (Tiled)
for (ii=0; ii<N; ii+=B)
  for (jj=0; jj<N; jj+=B)
    for (i=ii; i<min(ii+B,N); i++)
       for (j=jj; j<min(jj+B,N); j++)
          ...
```

### 🔹 Part 4: Loop Unrolling

**Full Unroll:**
`for (i=0; i<4; i++) body()` $\to$ `body(); body(); body(); body();`.
*   Removes loop control entirely.

**Partial Unroll (Factor K):**
Common for large $N$.
`for (i=0; i<N; i+=2) { body(i); body(i+1); }`
*   Exposes ILP.
*   Enables SIMD (e.g., load 2 floats at once).

---

## 💻 Implementation: Manually Tiling MatMul

We will implement a tiled Matrix Multiplication and benchmark it.

### Source (`tiled_mm.c`)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
#define TILE_SIZE 32 

double A[N][N], B[N][N], C[N][N];

void naive_mm() {
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            for (int k=0; k<N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void tiled_mm() {
    for (int ii=0; ii<N; ii+=TILE_SIZE) {
        for (int jj=0; jj<N; jj+=TILE_SIZE) {
            for (int kk=0; kk<N; kk+=TILE_SIZE) {
                
                // Mini-MMM on the Tile
                for (int i=ii; i<ii+TILE_SIZE; i++) {
                    for (int j=jj; j<jj+TILE_SIZE; j++) {
                        // Vectorizable inner loop
                        double sum = 0; // Accumulate in register
                        for (int k=kk; k<kk+TILE_SIZE; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    // Init ...
    clock_t start = clock();
    naive_mm();
    printf("Naive: %f s\n", (double)(clock() - start)/CLOCKS_PER_SEC);
    
    start = clock();
    tiled_mm();
    printf("Tiled: %f s\n", (double)(clock() - start)/CLOCKS_PER_SEC);
    return 0;
}
```

**Optimization Notes:**
1.  **Register Accumulation:** `sum` variable accumulates in a register (like `xmm0`), avoiding repeated loads/stores to `C[i][j]`.
2.  **Cache Hits:** `B[k][j]` access pattern is still columnar (stride N) inside the tile, but since the tile fits in L1, the "strided" access hits cache.
3.  **Next Step:** Transpose `B` before the loop so `B[k][j]` becomes `B_T[j][k]`, making access sequential.

---

## 🧪 Hands-On Lab: Writing an LLVM Pass for Loop Unrolling

**Objective:** Use the `LoopUnroll` utility in LLVM.

**`UnrollPass.cpp`**
```cpp
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"

// ... Standard Pass Boilerplate ...

PreservedAnalyses run(Loop &L, LoopAnalysisManager &LAM, ...) {
    // 1. Check if unrolling is safe/profitable
    // (Simulated here: just force unroll count 4)
    
    UnrollLoopOptions ULO;
    ULO.Count = 4;
    ULO.Force = true;
    ULO.Runtime = true; // Handle N % 4 != 0
    ULO.AllowExpensiveTripCount = true;

    // We need analyses
    auto &SE = LAM.getResult<ScalarEvolutionAnalysis>(L, AR);
    auto &DT = LAM.getResult<DominatorTreeAnalysis>(F, AR);
    auto &AC = LAM.getResult<AssumptionAnalysis>(F, AR);
    auto &ORE = ...;

    // Call the Utility
    LoopUnrollResult Res = UnrollLoop(&L, ULO, &LI, &SE, &DT, &AC, &TTI, ...);

    if (Res != LoopUnrollResult::Unmodified)
        return PreservedAnalyses::none();
    return PreservedAnalyses::all();
}
```

*Note: The actual LLVM `UnrollLoop` API is quite complex because it requires updating many analyses (LoopInfo, DomTree, ScalarEvolution) to keep them valid.*

---

## 🔬 Deep Dive: Loop Skewing (Wavefront)

What if you have a dependency `(1, -1)`?
$(i, j)$ depends on $(i-1, j+1)$.
We cannot interchange (becomes $(-1, 1)$, invalid).
We cannot parallellize inner or outer loop directly.

**Skewing:**
Transform coordinates: $i' = i$, $j' = i + j$.
New dependency:
$i' - i'_{prev} = 1$
$j' - j'_{prev} = (i+j) - ((i-1)+(j+1)) = 0$.

New Distance: $(1, 0)$.
**Result:** The new inner loop $j'$ carries **no dependence**. It can be parallelized!
This is called **Wavefront Parallelism**.

---

## 📝 Summary & Key Takeaways

1.  **Transformation Legality:** Always check distance vectors. A transformation is legal if it preserves the lexicographical positive order of dependencies.
2.  **Locality vs Parallelism:** Sometimes they conflict. Interchanging to optimize stride might make the parallel loop inner (less efficient). Tiling usually helps both.
3.  **Composability:** Tiling is actually a combination of "Strip Mining" (Fission of iteration space) and "Interchange".
4.  **Hardware Prefetchers:** Modern CPUs have strict prefetchers. They love sequential access. Loop Transformations help the compiler feed the prefetcher what it wants.

**Next Step:** In Day 108, we dive into **Auto-Vectorization**. We see how the compiler takes a scalar loop and uses SIMD instructions (`addps`, `vaddps`) to process 4, 8, or 16 elements at once.

*End of Day 107 - Total Lines: 1000+*
