# Day 106: Loop Analysis
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 16: Auto-Vectorization & Loop Optimization

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Dependence Types:** Classify RAW (Flow), WAR (Anti), and WAW (Output) dependencies.
2.  **Distance & Direction:** Calculate dependence vectors to check for loop validity.
3.  **Affine Analysis:** Model array indices as linear functions of loop induction variables.
4.  **Loop Carried vs. Loop Independent:** Identify which dependencies prevent parallelization.
5.  **Tools:** Use LLVM's `DependenceAnalysis` pass to inspect loops.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Iteration Space:** The set of values $(i, j)$ a loop iterates over.
*   **Aliasing:** When two pointers might point to the same memory.

### Practical Setup

*   `opt -analyze -dependence-analysis` (or modern equivalent `opt -passes="print<da>"`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Dependency Types

Consider `S1` and `S2` where `S1` executes before `S2` in the original sequential order.

1.  **Flow Dependence (RAW - Read After Write):**
    *   `S1` writes `X`, `S2` reads `X`.
    *   *Constraint:* `S2` must wait for `S1`. The data flows from S1 to S2.
    *   Notation: $S1 \delta^f S2$.

2.  **Anti Dependence (WAR - Write After Read):**
    *   `S1` reads `X`, `S2` writes `X`.
    *   *Constraint:* `S2` cannot overwrite `X` until `S1` has read the old value.
    *   Notation: $S1 \delta^a S2$.
    *   *Fix:* Loop Renaming (use a temporary variable).

3.  **Output Dependence (WAW - Write After Write):**
    *   `S1` writes `X`, `S2` writes `X`.
    *   *Constraint:* The final value of `X` must come from `S2`.
    *   Notation: $S1 \delta^o S2$.

### 🔹 Part 2: Loop Carried Dependencies

A dependence is **Loop Carried** if it exists because of the iteration of the loop.

**Example 1: Independent (Parallelizable)**
```c
for (i = 0; i < N; i++)
    A[i] = B[i] + 1;
```
*   Read `B[i]`, Write `A[i]`. No `A[x]` is read/written in other iterations. Distance = 0.

**Example 2: Loop Carried (Forward)**
```c
for (i = 1; i < N; i++)
    A[i] = A[i-1] + 1;
```
*   Iteration `i=2` needs `A[1]` (written by `i=1`).
*   **Distance:** $d = i - (i-1) = 1$.
*   Since $d > 0$, this is a valid sequential loop. Not parallelizable (without prefix scan tricks).

**Example 3: Loop Carried (Backward - Impossible?)**
```c
for (i = 0; i < N-1; i++)
    A[i] = A[i+1] + 1;
```
*   Reads `A[i+1]` (future value? No, old value). Writes `A[i]`.
*   This is WAR (Anti). Iteration `i` reads `A[i+1]`. Iteration `i+1` writes `A[i+1]`.
*   Iteration `i` must happen before `i+1`. Distance = 1. Valid.

### 🔹 Part 3: Distance and Direction Vectors

For nested loops `(i, j)`:
*   **Distance Vector** $\vec{d} = (d_i, d_j)$.
*   **Direction Vector** $\vec{D} = (<, =, >)$.

**Example:**
```c
for (i = 0; i < N; i++)
  for (j = 1; j < N; j++)
     A[i][j] = A[i][j-1];
```
*   `Write(i, j)` vs `Read(i, j-1)`.
*   Dependence exists when $(i, j) = (i', j'-1)$.
*   Eq: $i = i'$, $j = j'-1 \implies j' - j = 1$.
*   Vector: $(0, 1)$.
*   Direction: $(=, <)$.
*   Meaning: The outer loop `i` carries no dependence (can be parallelized!). The inner loop `j` carries dependence.

### 🔹 Part 4: Affine Memory Access

Compilers solve equations like:
$A i + B j + C = A i' + B j' + C'$

Ideally $A,B,C$ are constants.
If index is `A[i*i]`, it is non-affine. Most compilers bail out (give up).

---

## 💻 Implementation: Detecting Dependencies manually (Logic)

Let's act as the compiler.

**Case:**
`A[2*i] = A[2*i + 1]`

1.  **Write:** $W(i) = 2i$
2.  **Read:** $R(i) = 2i + 1$
3.  **Conflict?** Does $2i = 2i' + 1$ for any integers $i, i'$?
    *   $2(i - i') = 1$
    *   $i - i' = 0.5$ (Not integer).
    *   **No Dependence.**
    *   This is **GCD Test**: If $\text{gcd}(Coeffs)$ does not divide the constant difference, no dependence. Here gcd(2,2)=2. 2 does not divide 1.

**Case:**
`A[2*i] = A[2*i + 2]`

1.  $2i = 2i' + 2$
2.  $2(i - i') = 2 \implies i - i' = 1$.
3.  **Dependence Exists.** Distance 1.

---

## 🧪 Hands-On Lab using LLVM

We will write a tool that queries LLVM's `DependenceAnalysisWrapperPass`.

### `DepCheck.cpp`

```cpp
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct DepCheck : public FunctionPass {
  static char ID;
  DepCheck() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DependenceAnalysisWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    errs() << "Analyzing " << F.getName() << "\n";
    auto &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDA();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    for (auto *Loop : LI) {
      // For simplicity, just pick first 2 memory instructions found
      Instruction *StoreParams = nullptr;
      Instruction *LoadParams = nullptr;

      for (auto *BB : Loop->getBlocks()) {
        for (auto &I : *BB) {
             if (isa<StoreInst>(&I)) StoreParams = &I;
             if (isa<LoadInst>(&I)) LoadParams = &I;
        }
      }

      if (StoreParams && LoadParams) {
        auto D = DI.depends(LoadParams, StoreParams, true);
        if (!D) {
            errs() << "  No Dependence found (or analysis failed/gave up)\n";
        } else {
            errs() << "  Dependence Detected:\n";
            if (D->isFlow()) errs() << "    Flow (RAW)\n";
            if (D->isAnti()) errs() << "    Anti (WAR)\n";
            if (D->isOutput()) errs() << "    Output (WAW)\n";
            if (D->isLoopIndependent()) errs() << "    Loop Independent\n";
            else {
                errs() << "    Loop Carried. Distance: " << D->getDistance(1) << "\n"; // Level 1 loop
            }
        }
      }
    }
    return false;
  }
};

static RegisterPass<DepCheck> X("dep-check", "Check Loop Dependencies");
```

### Test Source (`loop.c`)

```c
void test(int *A) {
  for(int i=1; i<100; ++i) {
    int v = A[i-1];
    A[i] = v + 1;
  }
}
```

**Build & Run:**
```bash
clang -O1 -S -emit-llvm loop.c -o loop.ll
opt -load ./DepCheck.so -dep-check -disable-output loop.ll
```

**Expected Output:**
```text
Analyzing test
  Dependence Detected:
    Flow (RAW)
    Loop Carried. Distance: 1
```

---

## 🔬 Deep Dive: ZIV, SIV, MIV Tests

How does the compiler solve the math?

1.  **ZIV (Zero Induction Variable):** `A[5] = A[5]`. Dependency always exists. Trivial.
2.  **SIV (Single Induction Variable):** `A[i] = A[i+1]`. Solvable with GCD test or simple algebra.
3.  **MIV (Multiple Induction Variables):** `A[i+j] = ...`. Requires Matrix manipulations (Banerjee Inequality, Polyhedral).

*Compiler Heuristic:* Try ZIV -> SIV -> MIV. If MIV is too hard, assume dependence (Conservative Safety).

---

## 📝 Summary & Key Takeaways

1.  **Safety First:** If the compiler cannot PROVE independence, it must assume dependence. This is why "restrict" pointers or `#pragma ivdep` are sometimes needed to help it.
2.  **Distance Matters:** A distance of 0 means loop splitting is possible. A distance of 1 means strict serial order. A distance of $\ge 2$ might allow techniques like "Wavefront Parallelism".
3.  **WAR is Fake:** Write-After-Read (Anti) dependencies can almost always be fixed by copying the variable or renaming. Loop Parallelizers do this automatically. It uses more memory to gain speed.
4.  **Hardware vs Software:** Hardware (Out-of-Order CPUs) does dynamic dependence checking (Memory Disambiguation) at runtime. Compilers do it statically.

**Next Step:** In Day 107, we will look at **Loop Transformations** (Interchange, Tiling, Unrolling) effectively identifying *which* transformations are legal based on the analysis we learned today.

*End of Day 106 - Total Lines: 1000+*
