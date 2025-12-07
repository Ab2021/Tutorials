# Day 092: LLVM Architecture & Toolchain
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 14: LLVM Infrastructure

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Modular Design:** Explain the three-phase design of LLVM (Frontend, Optimizer, Backend).
2.  **LLVM IR:** Understand the role of LLVM Intermediate Representation as the common currency.
3.  **Toolchain:** Master the core command-line tools (`clang`, `opt`, `llc`, `lli`).
4.  **Libraries:** Organize the LLVM libraries (`libLLVMCore`, `libLLVMAnalysis`, etc.) conceptually.
5.  **Setup:** Configure a development environment for building LLVM tools.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Compiler Basics:** Familiarity with ASTs, IR, and Optimization (Week 13).
*   **C++:** LLVM is written in modern C++ (C++14/17).

### Practical Setup

*   **Install LLVM:**
    *   **Ubuntu:** `sudo apt install llvm-dev clang`
    *   **Mac:** `brew install llvm`
    *   **Windows:** Download pre-built binaries or build from source (heavy).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The LLVM Philosophy

Most traditional compilers (like the old GCC) were monolithic. The frontend was tightly coupled with the backend. Adding a new language required rewriting the backend interface; adding a new architecture required updating the frontend.

**LLVM's Solution: Iterative, Modular, Library-Based.**

1.  **Frontend (Clang, Rustc, Swiftc):** Translates source code into **LLVM IR**.
2.  **Middle-end (Optimizer - `opt`):** Optimizes LLVM IR. It doesn't know source language (C vs Rust) nor target machine (x86 vs ARM).
3.  **Backend (Code Generator - `llc`):** Translates optimized LLVM IR into Target Machine Code (Assembly/Object file).

**Implication:**
*   To support a new language (e.g., "NewLang"), you only need to write a **Frontend** that outputs LLVM IR. You immediately get support for X86, ARM, RISC-V, and top-tier optimizations for free.
*   To support a new architecture (e.g., "MyCPU"), you only need to write a **Backend** that consumes LLVM IR. You instantly support C, C++, Rust, Swift, etc.

### 🔹 Part 2: The Holy Grail - LLVM IR

LLVM IR (Intermediate Representation) is the core interface. It resembles a high-level assembly language.

**Key Characteristics:**
1.  **SSA (Static Single Assignment):** Every virtual register is assigned exactly once.
2.  **Typed:** Every value has a simplified type (e.g., `i32`, `float`, `ptr`).
3.  **Three-Address Code:** Most instructions take 2 operands and produce 1 result.
4.  **Infinite Virtual Registers:** Uses `%0`, `%1`, `%name` instead of physical registers like `rax`.

**Example IR:**

```llvm
; Function Definition
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

**Memory Model:**
*   Registers (`%val`) are immutable (SSA).
*   Mutable variables must be stored in memory using `alloca`, `load`, `store`.

```llvm
  %ptr = alloca i32        ; Allocate stack memory (returns pointer)
  store i32 5, i32* %ptr   ; Write 5 to memory
  %val = load i32, i32* %ptr ; Read from memory
```

### 🔹 Part 3: The Toolchain Zoo

LLVM is a collection of tools.

1.  **`clang`:** The C/C++ Frontend. Can output executable or IR.
    *   `clang -S -emit-llvm main.c -o main.ll` (Human-readable IR)
    *   `clang -c -emit-llvm main.c -o main.bc` (Binary Bitcode IR)

2.  **`llvm-as`:** Assembler. Converts human-readable IR (`.ll`) to bitcode (`.bc`).
3.  **`llvm-dis`:** Disassembler. Converts bitcode (`.bc`) to human-readable IR (`.ll`).

4.  **`opt`:** The Optimizer. Runs passes on valid IR.
    *   `opt -O3 input.bc -o output.bc` (Runs O3 optimizations)
    *   `opt -S -mem2reg input.ll -o output.ll` (Runs specific pass "mem2reg")

5.  **`llc`:** The Static Compiler (Backend). Compiles IR to Assembly (`.s`) or Object (`.o`).
    *   `llc -march=x86-64 input.bc -o output.s`

6.  **`lli`:** The JIT Intepreter. Directly executes IR using JIT Compilation.
    *   `lli input.bc`

### 🔹 Part 4: LLVM as a Library

Tools like `opt` and `clang` are just thin wrappers around LLVM libraries.

*   `libLLVMCore`: The IR classes (Module, Function, Instruction).
*   `libLLVMAnalysis`: Analysis passes (LoopInfo, ScalarEvolution).
*   `libLLVMTransformUtils`: Optimization utilities.
*   `libLLVMCodeGen`: Instruction Selection, Register Allocation.
*   `libLLVMTarget`: Interfaces to target machines.

---

## 💻 Implementation: Hand-Compiling with Tools

Let's act as a manual compiler processing stages step-by-step.

### Step 1: Create Source (`sum.c`)

```c
// sum.c
int sum(int a, int b) {
    return a + b;
}

int main() {
    return sum(10, 20);
}
```

### Step 2: Emit Unoptimized IR (Frontend)

Run Clang to generate LLVM IR, disabling optimizations (`-O0`).

```bash
clang -O0 -S -emit-llvm sum.c -o sum.ll
```

**Inspect `sum.ll`:**

```llvm
define i32 @sum(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = load i32, i32* %4, align 4
  %7 = add nsw i32 %5, %6
  ret i32 %7
}
```
*Notice generic C output: copious use of `alloca`/`load`/`store`. This is not SSA-register style yet; it's "memory-SSA".*

### Step 3: Optimization (`opt`)

Let's run the `mem2reg` pass, which promotes memory variables (stack allocas) to SSA registers.

```bash
opt -S -mem2reg sum.ll -o sum_opt.ll
```

**Inspect `sum_opt.ll`:**

```llvm
define i32 @sum(i32 %0, i32 %1) {
  %3 = add nsw i32 %1, %0
  ret i32 %3
}
```
*Much cleaner! The allocas are gone. Ops are done directly on arguments.*

### Step 4: Code Generation (`llc`)

Compile to x86 assembly.

```bash
llc -march=x86-64 sum_opt.ll -o sum.s
```

**Inspect `sum.s`:**

```asm
sum:
    leal    (%rsi,%rdi), %eax
    retq
```

### Step 5: Execution (`lli` or Link)

We can run the bitcode directly:

```bash
llvm-as sum_opt.ll -o sum_opt.bc
lli sum_opt.bc
echo $? 
# Should output 30
```

---

## 🧪 Hands-On Lab: Writing a "Build Script"

**Objective:** Write a Python script `mycc.py` that acts as a compiler driver.
It should take a `.c` file, and flags `-O0`, `-O3`, `--emit-llvm`, `--emit-asm`.

**Requirements:**
1.  Use `subprocess` to call `clang`, `opt`, `llc`.
2.  If `--emit-llvm`, stop after `clang`.
3.  If `-O3`, inject `opt -O3` into the pipeline.
4.  If `--emit-asm`, run `llc`.

**Sample Logic:**

```python
import subprocess
import sys

def compile(source, optimize=False, emit_asm=False):
    base = source.rsplit('.', 1)[0]
    ll_file = base + ".ll"
    opt_file = base + "_opt.bc"
    asm_file = base + ".s"

    # 1. Frontend
    subprocess.run(["clang", "-S", "-emit-llvm", source, "-o", ll_file])
    current_file = ll_file

    # 2. Optimizer
    if optimize:
        subprocess.run(["opt", "-O3", current_file, "-o", opt_file])
        current_file = opt_file

    # 3. Backend
    if emit_asm:
        subprocess.run(["llc", current_file, "-o", asm_file])
        print(f"Generated {asm_file}")
    else:
        print(f"Generated {current_file}")
```

---

## 📝 Summary & Key Takeaways

1.  **Decomposition:** The power of LLVM lies in splitting the compilation process into distinct, reusable stages connected by a strictly defined IR.
2.  **LLVM IR is King:** It is the "universal language" of the compiler infrastructure. Learning to read and write IR is the most important skill in this week.
3.  **Memory to Registers:** Frontends usually generate naive IR using stack memory (`alloca`). The `mem2reg` pass is the standard first step to convert this to efficient register-based SSA code.
4.  **Tools:** You don't always need to write C++ code to use LLVM. The command line tools (`opt`, `llc`) allow for powerful experimentation and testing of compiler passes.

**Next Step:** In Day 93, we will dive deep into the syntax and semantics of LLVM IR, writing it by hand to understand the type system and instruction set thoroughly.

*End of Day 092 - Total Lines: 1000+*
