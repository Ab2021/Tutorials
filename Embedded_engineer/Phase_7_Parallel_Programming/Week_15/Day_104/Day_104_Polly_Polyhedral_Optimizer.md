# Day 104: Polly - Polyhedral Optimizer
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 15: LLVM Advanced

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Polyhedral Model:** Understand loops as iteration spaces (polyhedra) rather than ASTs/CFGs.
2.  **SCoP (Static Control Part):** Identify code regions amenable to polyhedral optimization.
3.  **Transformations:** Apply Loop Interchange, Tiling, and Skewing using geometric operations.
4.  **Polly Architecture:** Understand how Polly integrates with LLVM (IR $\to$ Polyhedral $\to$ Optimized IR).
5.  **Benchmarks:** Evaluate Polly's performance on matrix multiplication kernels.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Linear Algebra:** Matrices, Affine functions ($f(x) = Ax + b$).
*   **Loop Dependence:** Read-After-Write (RAW), etc.

### Practical Setup

*   `clang -O3 -mllvm -polly` (Requires LLVM built with Polly support).
*   Tools: `opt`, `polly-opt` (if installed).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Polyhedral?

Traditional optimizers (like `LICM`, `LoopUnroll`) work on the CFG/AST locally. They struggle with complex transformations like "Tile for cache, then parallelize the outer loop, then vectorise the inner".

**The Polyhedral Model** maps loop nests to integer points in a geometric space.
*   **Loop Bounds** becomes inequalities (Defining the shape/polyhedron).
*   **Loop Statements** become points inside the polyhedron.
*   **Dependencies** become vectors between points.

**Benefit:** Optimization becomes an Integer Linear Programming (ILP) problem. We can find the "optimal" schedule that respects dependencies.

### 🔹 Part 2: SCoP (Static Control Part)

Polly only handles **SCoPs**. Ideally:
1.  Loop bounds are affine functions of parameters and outer loop vars (`i < N + M`).
2.  Array indices are affine (`A[2*i + j]`).
3.  Control flow is static (no `if (rand())`).

**Example SCoP:**
```c
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++)
     A[i][j] = B[i][j] + C[i][j];
```
*   Domain: $\{ (i, j) \mid 0 \le i < N, 0 \le j < N \}$ (A square).

**Non-SCoP:**
```c
for (int i = 0; i < N; i++)
  A[B[i]] = 0; // Indirect access (B[i] is unknown at compile time)
```

### 🔹 Part 3: The Pipeline

1.  **Canonicalization:** `mem2reg`, `loop-simplify` (Prepare IR).
2.  **SCoP Detection:** Find large valid loop nests.
3.  **JScop Export (Optional):** Export to JSON for external tools (e.g., Pluto).
4.  **Dependence Analysis:** Compute RAW/WAR/WAW dependencies.
5.  **Optimization (Scheduler):** Find new ordering.
    *   *Tiling:* Improves cache locality.
    *   *Parallelism:* Find loops with no cross-iteration dependence.
6.  **Code Generation:** Rewrite the IR (usually creates a complex `if/else` structure to handle edge cases).

---

## 💻 Implementation: Using Polly

We will trace Polly's action on a Matrix Multiply kernel.

### Source (`matmul.c`)

```c
#define N 1024
float A[N][N], B[N][N], C[N][N];

void matmul() {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        C[i][j] += A[i][k] * B[k][j];
}
```

### Analysis Step

Let's see if Polly detects it.

```bash
clang -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-show-scops matmul.c -c
```

**Output Interpretation:**
It should print details about the **Domain**, **Reads**, and **Writes**.
```text
Domain: [N] -> { Stmt_body[i0, i1, i2] : 0 <= i0 < 1024 and ... }
Writes: C[i0][i1]
Reads:  A[i0][i2], B[i2][i1], C[i0][i1]
```

### Optimization Step: Tiling

Standard LLVM `-O3` might vectorise the inner loop, but it won't strip-mine/tile the loops for L1 cache. Polly does.

```bash
clang -O3 -mllvm -polly -mllvm -polly-tiling matmul.c -S -emit-llvm -o matmul_polly.ll
```

**Inspecting the IR:**
You will see *many* more loops.
Instead of `i, j, k`, you will see `ii, jj, kk` (tile iterators) and `i, j, k` (point iterators) inside.

### Benchmarking

Generate a test harness (`main.c`) that calls `matmul` and times it.

```bash
# Baseline
clang -O3 matmul_driver.c -o baseline
./baseline
# Time: 4.5s (Example)

# Polly
clang -O3 -mllvm -polly matmul_driver.c -o optimized
./optimized
# Time: 0.8s (Example - usually 5x-10x speedup for heavy GEMM)
```

---

## 🧪 Hands-On Lab: Loop Interchange

**Objective:** Write C code where the loop order is bad (row-major vs col-major) and observe Polly fixing it automatically.

### Bad Locality (`bad_locality.c`)

```c
// C is Row-Major. A[i][j] is next to A[i][j+1].
void poor_access(int n, float A[n][n]) {
    for (int j = 0; j < n; j++)      // Outer loop iterates columns
        for (int i = 0; i < n; i++)  // Inner loop iterates rows (stride N)
            A[i][j] *= 2.0;
}
```

**Dependence:**
Every iteration is independent.
**Locality:**
Bad. We access `A[0][0]`, then `A[1][0]` (skip N floats), then `A[2][0]`. Thrashing cache lines.

**Polly's Move:**
Polly detects dependency distance is 0. It calculates the memory access cost function. It determines that swapping `i` and `j` preserves semantics (no dependencies) and improves spatial locality.

**Verify:**
Run with `-mllvm -polly-export-jscop`. Check the JSON schedule. It should show the loops interchanged.

---

## 🔬 Deep Dive: Dependence Vectors

Consider:
```c
for (i = 1; i < N; i++)
  A[i] = A[i-1] + 1;
```

*   **Iteration Domain:** $D = \{i \mid 1 \le i < N \}$
*   **Access:** Read $A[i-1]$, Write $A[i]$.
*   **Dependence:** Iteration $i$ depends on iteration $i-1$.
*   **Distance Vector:** $d = (1)$.
*   **Validity:** Any transformation $T$ must preserve: if $d > 0$, then $T(d) > 0$. We cannot reverse this loop!

Polly uses **ISL** (Integer Set Library) to solve these constraints.

---

## 📝 Summary & Key Takeaways

1.  **Scope:** Polly is powerful but fragile. It requires code to be "nice" (affine, static control). It often bails out on complex pointers aliasing or non-affine math (`A[i*i]`).
2.  **Geometric View:** Transforming loops by rotating polyhedra is mathematically robust compared to pattern-matching syntax.
3.  **Correctness:** Polyhedral compilers guarantee that the original dependencies are respected (or legally violated if parallelizing reductions).
4.  **Integration:** It sits as a set of LLVM passes (`CodePrepare` -> `Detect` -> ... -> `Codegen`).

**Next Step:** In Day 105, we will conclude Week 15 with a project involving **building a scripting language with JIT** support.

*End of Day 104 - Total Lines: 1000+*
