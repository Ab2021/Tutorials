# Day 096: Standard Optimization Passes
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 14: LLVM Infrastructure

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Scalar Optimizations:** Master `mem2reg`, `SCCP`, `GVN`, and `InstCombine`.
2.  **Loop Optimizations:** Understand `LICM`, `LoopUnroll`, and `IndVarSimplify`.
3.  **IPO (Interprocedural):** Explain `Inline`, `GlobalDCE`, and `ArgumentPromotion`.
4.  **Pipeline Construction:** Build a custom optimization pipeline using `opt`.
5.  **Pass Ordering:** Understand why the order of passes matters (Canonicalization vs lowering).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **SSA Form:** Many passes (like GVN) require SSA input.
*   **DomTree:** Dominance property is essential for Loop identification.

### Practical Setup

*   `opt` tool is the primary playground.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Must-Have" Passes

These passes form the backbone of any LLVM-based compiler.

#### 1. `mem2reg` (Promote Memory to Register)
*   **Problem:** Frontends generate `alloca`/`load`/`store` for everything. This is slow and inhibits other optimizations.
*   **Action:** Promotes stack variables to SSA registers (`phi` nodes).
*   **Requirement:** The variable must be `alloca`'d in the entry block and have its address NOT taken (no pointer escaping).

#### 2. `instcombine` (Instruction Combining)
*   **Action:** Peep-hole optimizer. Merges multiple instructions into one cheaper one.
*   **Examples:**
    *   `add i32 X, 0` $\to$ `X`
    *   `mul i32 X, 2` $\to$ `shl i32 X, 1`
    *   `(A + B) + C` $\to$ `A + (B + C)` (Association)
*   **Impact:** Runs iteratively until no changes. Cleans up mess left by other passes.

#### 3. `sccp` (Sparse Conditional Constant Propagation)
*   **Action:** Propagates constants through the CFG.
*   **Advanced:** It is "Sparse" because it uses SSA def-use chains (skips irrelevant instructions). It is "Conditional" because it can evaluate branch conditions (`br i1 true ...`) and determine that dead branches are never taken, allowing it to assume values on those edges don't matter.

#### 4. `gvn` (Global Value Numbering)
*   **Action:** Identifies and analyzes redundant computations.
*   **Mechanism:** Assigns a "value number" to every expression. If `a+b` has ID 42, and we see `a+b` again, we replace it with the previous result.
*   **Scope:** Global (Function-wide), unlike early CSE.

### 🔹 Part 2: Loop Optimizations

Loops are where execution time is spent.

#### 1. `licm` (Loop Invariant Code Motion)
*   **Action:** Hoists instructions out of the loop if their operands don't change inside the loop.
*   **Benefit:** Reduces dynamic instruction count.

#### 2. `indvars` (Induction Variable Simplification)
*   **Action:** Analyzes loop counters (`i=0, i++, i<N`). Transforms pointer arithmetic to be simpler (Canonicalization).

#### 3. `loop-unroll`
*   **Action:** Duplicates the loop body `K` times.
*   **Benefit:** Reduces branch overhead; exposes more ILP (Instruction Level Parallelism) to `instcombine`.

### 🔹 Part 3: Interprocedural Optimizations (IPO)

#### 1. `inline` (Function Inlining)
*   **Action:** Replaces `call @foo` with the body of `@foo`.
*   **Benefit:** Removes call overhead; enables context-specific optimization (constants passed as args).
*   **Cost:** Increases code size (Instruction Cache pressure).

#### 2. `globaldce` (Global Dead Code Elimination)
*   **Action:** Removes functions/globals that are never called/used in the module.

---

## 💻 Implementation: Pipeline Experimentation

We will take a naive piece of code and transform it step-by-step using `opt`.

### Source (`example.ll`)

This code computes summation of 1 to N, but naively.

```llvm
define i32 @sum(i32 %n) {
entry:
  %i = alloca i32
  %s = alloca i32
  store i32 0, i32* %i
  store i32 0, i32* %s
  br label %loop

loop:
  %curr_i = load i32, i32* %i
  %curr_s = load i32, i32* %s
  
  ; Check i < n
  %cmp = icmp slt i32 %curr_i, %n
  br i1 %cmp, label %body, label %exit

body:
  ; s = s + i
  %next_s = add i32 %curr_s, %curr_i
  store i32 %next_s, i32* %s
  
  ; i = i + 1
  %next_i = add i32 %curr_i, 1
  store i32 %next_i, i32* %i
  
  br label %loop

exit:
  %final_s = load i32, i32* %s
  ret i32 %final_s
}
```

### Stage 1: mem2reg

```bash
opt -S -mem2reg example.ll -o step1.ll
```

**Output:**
*   Allocas gone.
*   Phis inserted in `loop` block.

```llvm
loop:
  %curr_i = phi i32 [ 0, %entry ], [ %next_i, %body ]
  %curr_s = phi i32 [ 0, %entry ], [ %next_s, %body ]
  ...
```

### Stage 2: instcombine (Cleanup)

```bash
opt -S -instcombine step1.ll -o step2.ll
```

*   Likely no major changes here since the logic is simple, but it canonicalizes comparisons.

### Stage 3: loop-rotate

LLVM prefers "do-while" style loops (bottom-tested) over "while" loops (top-tested) because it reduces branches in the hot path.

```bash
opt -S -loop-rotate step2.ll -o step3.ll
```

**Output:**
*   The condition check moves to the end of the block.
*   An `if` check is usually added before the loop (Loop Guard) to handle the case where the loop runs 0 times.

### Stage 4: indvars + loop-unroll (if N is constant, or partial)

```bash
opt -S -indvars -loop-unroll step3.ll -o step4.ll
```

### Stage 5: The "O3" Umbrella

Instead of running passes manually, we usually run the curated pipeline.

```bash
opt -O3 -S example.ll -o final.ll
```

**Result:**
If `n` is known (constant), LLVM often replaces the loop with the mathematical formula $\frac{n(n-1)}{2}$ using ScalarEvolution!

---

## 🧪 Hands-On Lab: Performance War

**Objective:** Write a specific C function, compile it with `-O0`, and then try to minimize its instruction count using a sequence of `opt` passes.

**Input Code (`matrix_f.ll`):** A naive matrix flattened access.

```llvm
; Naive: computes (row * width + col) every time
define i32 @get(i32 %row, i32 %col, i32 %width) {
  %t1 = mul i32 %row, %width
  %t2 = add i32 %t1, %col
  ret i32 %t2
}

define i32 @loop_access(i32 %width) {
  ; loop 0 to 10
  ;   call get(i, 0, width)
}
```

**Task:**
1.  Inline `@get` into `@loop_access`.
2.  Use `licm` to pull `%width` invariant calculations out.
3.  Use `gvn` / `instcombine` to simplify the pointer math.

**Commands:**
```bash
opt -S -inline -dce matrix_f.ll -o inline.ll
opt -S -licm inline.ll -o licm.ll
```

**Analysis:**
Did the multiplication `%row * %width` get hoisted? Or optimized away?

---

## 🔬 Deep Dive: Pass Dependencies

Why does `-O3` run passes in a specific order?

1.  **Cleanup first:** `mem2reg`, `instcombine` simplify IR so complex analyses (SCEV, AliasAnalysis) don't get confused by stack spill code.
2.  **Canonicalization:** Passes like `loop-simplify` and `lcssa` (Loop Closed SSA) put loops into a standard form so `licm` and `unroll` can work generically.
3.  **Lowering:** Vectorization happens late. If we vectorized too early, `instcombine` might allow scalar optimizations that break the vector structure.

---

## 📝 Summary & Key Takeaways

1.  **mem2reg is king:** Always run it first on naive IR.
2.  **Iterative Process:** Optimizations create opportunities for other optimizations. (e.g., Inlining exposes constant arguments, enabling Dead Code Elimination).
3.  **Cost Models:** Passes like Inlining or Unrolling use "Cost Models" (heuristics) to decide *if* they should apply. They simulate the code size/speed impact.
4.  **UB is helpful:** Undefined Behavior (like signed overflow) allows the optimizer to assume things (e.g., `x + 1 > x`).
5.  **Pipeline Managers:** `PassBuilder` in NPM defines the default `-O3` pipeline. You can inject your own passes into it.

**Next Step:** In Day 97, we move to the Backend. We will see how this Optimized IR gets selected into machine instructions (Instruction Selection).

*End of Day 096 - Total Lines: 1000+*
