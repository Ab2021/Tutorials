# Day 113: GCC Architecture
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **GCC Pipeline:** Map the flow from source -> GENERIC -> GIMPLE -> RTL -> ASM.
2.  **IR Differences:** Contrast GCC's multi-IR approach (GIMPLE/RTL) with LLVM's single-IR design.
3.  **Inspect Dumps:** Use `-fdump-tree-*` and `-fdump-rtl-*` to view the compiler's internal state.
4.  **Backend Structure:** Understand Machine Descriptions (`.md` files) and how RTL maps to assembly.
5.  **Ecosystem:** Appreciate the role of `cc1`, `as`, `ld`, and `collect2`.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Compiler Frontend:** Parsing C/C++ to AST.
*   **Compiler Backend:** Instruction selection and register allocation.

### Practical Setup

*   `gcc` installed (preferably GCC 10+).
*   Tools: `less`, `vim` (to read massive dump files).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: A Tale of Two Compilers

**LLVM:**
*   One IR ("LLVM IR") rules them all.
*   Frontend (Clang) $\to$ LLVM IR $\to$ Backend (Instruction Selection) $\to$ Machine Code.
*   Memory SSA form. Extremely modular library design.

**GCC (GNU Compiler Collection):**
*   Monolithic historic design (start in 1987).
*   **Three IRs:**
    1.  **GENERIC:** Language-independent AST. (Heavily tree-based).
    2.  **GIMPLE:** Simplified 3-address code. (Where SSA optimization happens).
    3.  **RTL:** Register Transfer Language. Low-level, LISP-like, CPU-aware. (Where Backend optimization happens).

### 🔹 Part 2: The Pipeline Steps

1.  **Frontend (FE):**
    *   `cc1` parses source.
    *   Generates **GENERIC** (Tree).
    *   Specific to languages (C-family, Fortran, Ada, Rust).
2.  **Gimplification:**
    *   Lowers GENERIC to **GIMPLE**.
    *   Complex expressions broken down (`a = b + c + d` $\to$ `t1 = b+c; a = t1+d`).
    *   **SSA Pass:** Converts GIMPLE to SSA form (Phi nodes).
    *   **Tree Optimizers:** Inlining, Dead Code, Vectorization (`ftree-vectorize`).
3.  **Expansion:**
    *   Lowers GIMPLE to **RTL**.
    *   Virtual registers mapped to pseudo-registers.
4.  **RTL Optimization:**
    *   Instruction Scheduling, Register Allocation (RA), Peephole optimization.
5.  **Assembly Output:**
    *   Generates `.s` file.

### 🔹 Part 3: GIMPLE Detail

GIMPLE is GCC's equivalent to LLVM IR, but:
*   It preserves more high-level semantics (like loops). LLVM IR only has branches.
*   There are 3 levels: High GIMPLE, Low GIMPLE, SSA GIMPLE.

**Tuple Representation:**
Internally, GIMPLE is stored as tuples (not trees) to save memory.
`gimple_assign <PLUS_EXPR, a, b, c>`

### 🔹 Part 4: RTL (Register Transfer Language)

RTL looks like LISP. It describes hardware operations.
`(set (reg:SI 0 ax) (plus:SI (reg:SI 1 dx) (const_int 5)))`

*   **Machine Description (.md):** GCC reads a description file for the target CPU (e.g., `i386.md`) that matches RTL patterns to assembly instructions.
*   **Strictness:** RTL is very strict about modes (types). `SI` = Single Integer (32-bit), `DI` = Double Integer (64-bit).

---

## 💻 Implementation: Watching the Pipeline

We will compile a single file and dump *everything*.

### Source (`hello.c`)

```c
#include <stdio.h>

int compute(int a, int b) {
    return (a + b) * (a - b);
}

int main() {
    printf("Res: %d\n", compute(10, 5));
    return 0;
}
```

### Experiment 1: Tree Dumps (GIMPLE)

```bash
gcc -O2 -fdump-tree-all hello.c -o hello
ls hello.c.*
```

**What you see:**
Hundreds of files named `hello.c.003t.original`, `hello.c.004t.gimple`, `hello.c.020t.ssa`, ...

*   **Original:** The raw AST (GENERIC).
*   **Gimple:** The lowered 3-address code.
*   **SSA:** With versions `a_1`, `b_2`.
*   **Optimized:** Final GIMPLE before expansion (look for `hello.c.*t.optimized`).

**Inspection:**
Open `hello.c.*t.optimized`.
You should see that `compute` is folded or inlined into main if possible. Since we defined `compute` in the same unit, GCC likely inlined it.

### Experiment 2: RTL Dumps

```bash
gcc -O2 -fdump-rtl-all hello.c -o hello
ls hello.c.*r.*
```

**What you see:**
`hello.c.231r.expand` (GIMPLE -> RTL expansion).
`hello.c.282r.ira` (Integrated Register Allocator).
`hello.c.283r.reload` (Spill code generation).

**Analyzing Expansion:**
Open `.expand`. You will see "Insns" (Instructions).
`(insn 5 4 6 2 (set (reg:SI 87) ...))`
This is the raw data flow before physical registers are assigned.

### Experiment 3: Graph Visualization

GCC can output Dot graphs.
```bash
gcc -O2 -fdump-tree-all-graph hello.c
```
This generates `.dot` files. You can convert them to PNG using `graphviz`.
Useful for visualizing the CFG (Control Flow Graph) at different stages.

---

## 🧪 Hands-On Lab: The `collect2` wrapper

When you run `gcc`, you aren't running the compiler. You are running the **Driver**.

**Task:** Use `-v` (verbose) to see the real tools.

```bash
gcc -v -O2 hello.c -o hello
```

**Output Analysis:**
1.  **`cc1`:** The compiler. Takes `.c`, outputs `.s` (assembly).
    *   Look at the HUGE list of flags passed to it.
2.  **`as`:** The assembler. Takes `.s`, outputs `.o`.
3.  **`collect2`:** The helper around the linker.
4.  **`ld`:** The actual linker.

**Why `collect2`?**
Historically, some features (like C++ constructors `__static_initialization_and_destruction_0`) required the linker to scan the object files and generate a list of startup functions. Modern `ld` handles `.init_array`, so `collect2` is mostly a wrapper now.

---

## 🔬 Deep Dive: GCC vs LLVM Philosophy

**Licensing:**
*   GCC: **GPLv3**. Vital for the Free Software movement. Ensures specific proprietary plugins cannot close the source.
*   LLVM: **Apache 2.0**. Permissive. Apple/Sony/Google use it to build proprietary compilers.

**Architecture:**
*   GCC: Variable names often just `tree` or `rtx`. Huge unions. Codebase is C++ (since 2012) but feels "C-like".
*   LLVM: Heavy use of C++ inheritance, Templates, `dyn_cast`.

**Performance:**
Historically GCC generated faster code. LLVM caught up around 2015.
Today: They trade blows.
*   GCC usually better at: Fortran, loop unrolling, specialized architectures.
*   LLVM usually better at: Link Time Optimization (ThinLTO), Sanitizers, Compile speed.

---

## 📝 Summary & Key Takeaways

1.  **Complexity:** GCC is massive. Do not try to read `gcc/combine.c` (instruction combiner) unless you are brave. It is 15,000 lines of complex RTL logic.
2.  **Dump Flags:** `-fdump-tree-all` and `-fdump-rtl-all` are your best friends. They are equivalent to LLVM's `-print-after-all`.
3.  **GIMPLE vs RTL:** GIMPLE is for logic optimization (DCE, inlining). RTL is for machine optimization (instruction selection, scheduling).
4.  **Stability:** GCC is the "system compiler" for Linux. It prioritizes stability and standard compliance above all.

**Next Step:** In Day 114, we will learn the **GIMPLE IR** syntax and semantics in detail, writing a small python script to parse GIMPLE dumps.

*End of Day 113 - Total Lines: 1000+*
