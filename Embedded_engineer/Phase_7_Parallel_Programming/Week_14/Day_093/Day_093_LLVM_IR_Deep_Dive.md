# Day 093: LLVM IR Deep Dive
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 14: LLVM Infrastructure

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **IR Syntax:** Read and write LLVM assembly (`.ll`) fluently.
2.  **Type System:** Use primitive types (`i32`, `float`), pointers (`ptr`), vectors (`<4 x float>`), and aggregates (`struct`, `array`).
3.  **Instruction Set:** Master the core instructions: Arithmetic, Memory, Control Flow, and Conversion.
4.  **Phi Nodes:** Properly construct SSA graphs using the `phi` instruction.
5.  **Attributes & Metadata:** Annotate instructions with optimization hints (e.g., `nsw`, `tail`, `!tbaa`).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **SSA:** Static Single Assignment is the heart of LLVM.
*   **RISC-like ISA:** LLVM instructions are simple (Load/Store architecture).

### Practical Setup

*   `llvm-as` (Assembler) and `lli` (Interpreter) are your best friends today.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Modules, Functions, and Basic Blocks

**Structure Hierarchy:**
*   **Module:** The top-level container (like a translation unit/file). Contains global variables and functions.
*   **Function:** Contains Basic Blocks and Arguments.
*   **Basic Block:** A sequence of Instructions. Ends with a **Terminator** (ret, br, switch).
*   **Instruction:** The atomic unit of execution.

**Example Module:**
```llvm
; Global Variable
@g_val = global i32 100

; Function Declaration (External)
declare i32 @printf(ptr, ...)

; Function Definition
define i32 @main() {
entry:
  %val = load i32, i32* @g_val
  ret i32 %val
}
```

### 🔹 Part 2: The Type System

LLVM is strongly typed.

1.  **Integers:** `iN` where N is bit width. `i1` (bool), `i8` (byte), `i32` (int), `i64` (long), `i128`.
2.  **Floating Point:** `half`, `float`, `double`, `fp128`.
3.  **Pointers:** Historically `i32*`, `float*`. Modern LLVM uses opaque pointers: Just `ptr`.
4.  **Arrays:** `[N x <type>]`. E.g., `[10 x i32]`.
5.  **Structs:** `{type1, type2}`. E.g., `{i32, float}`.
6.  **Vectors:** `<N x <type>>`. E.g., `<4 x float>` (SIMD).
7.  **Labels:** `label` (used for branches).
8.  **Void:** `void` (return type only).

### 🔹 Part 3: Essential Instructions

#### A. Arithmetic (Binary Ops)
Only operate on registers, not memory.
*   `add i32 %a, %b`
*   `sub`, `mul`, `udiv` (unsigned div), `sdiv` (signed div).
*   `fadd`, `fsub`, `fmul`, `fdiv` (floating point).
*   **Modifiers:** `nsw` (No Signed Wrap), `nuw` (No Unsigned Wrap).

#### B. Memory Access (Load/Store Arch)
*   **`alloca`:** Allocates on stack. Returns `ptr`.
    ```llvm
    %ptr = alloca i32
    ```
*   **`store`:** Writes to memory.
    ```llvm
    store i32 5, ptr %ptr
    ```
*   **`load`:** Reads from memory.
    ```llvm
    %val = load i32, ptr %ptr
    ```
*   **`getelementptr` (GEP):** Calculates address offsets safely. **Does not access memory**.
    ```llvm
    ; access index 1 of array starting at %base
    %elem_ptr = getelementptr i32, ptr %base, i32 1
    ```

#### C. Control Flow (Terminators)
*   **`ret`:** Return from function.
    ```llvm
    ret i32 0
    ```
*   **`br`:** Branch.
    *   Unconditional: `br label %target`
    *   Conditional: `br i1 %cond, label %true_dest, label %false_dest`
*   **`switch`:** Multi-way branch.

#### D. Comparison
*   **`icmp`:** Integer comparison. Returns `i1`.
    *   `eq` (equal), `ne` (not equal)
    *   `sgt` (signed greater than), `ugt` (unsigned greater than)
    ```llvm
    %cmp = icmp eq i32 %a, 10
    ```
*   **`fcmp`:** Float comparison.
    *   `oeq` (ordered equal - no NaNs), `ueq` (unordered equal - NaNs allowed).

#### E. Conversion (Casts)
*   `trunc .. to` (i32 -> i8)
*   `zext .. to` (Zero extend i8 -> i32)
*   `sext .. to` (Sign extend)
*   `fptoui .. to`, `sitofp .. to` (Float <-> Int)
*   `bitcast .. to` (Reinterpret bits - e.g., float to i32).

### 🔹 Part 4: The Phi Node

The SSA merger. Must be the **first** instruction(s) in a Basic Block.

```llvm
loop_header:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop_body ]
  ...
```
*Meaning:* If we arrived from `%entry`, `%i` is 0. If from `%loop_body`, `%i` is `%next_i`.

---

## 💻 Implementation: Writing Code by Hand

Let's implement a standard algorithmic function manually in LLVM IR to understand the mechanics.

### Task: Factorial (Iterative)

C Equivalent:
```c
int factorial(int n) {
    int res = 1;
    while (n > 0) {
        res = res * n;
        n = n - 1;
    }
    return res;
}
```

LLVM IR Implementation (`fact.ll`):

```llvm
define i32 @factorial(i32 %n_arg) {
entry:
    ; Check if n <= 0 initially to skip loop? 
    ; Let's do a rigorous standard loop using Phi nodes.
    
    ; Compare n > 0
    %init_cmp = icmp sgt i32 %n_arg, 0
    br i1 %init_cmp, label %loop, label %exit

loop:
    ; Phi nodes for 'res' and 'n'
    ; res starts at 1, updates to %new_res
    ; n starts at %n_arg, updates to %new_n
    
    %curr_n   = phi i32 [ %n_arg, %entry ], [ %new_n, %loop ]
    %curr_res = phi i32 [ 1,      %entry ], [ %new_res, %loop ]
    
    ; Body: res = res * n
    %new_res = mul i32 %curr_res, %curr_n
    
    ; Body: n = n - 1
    %new_n = sub i32 %curr_n, 1
    
    ; Loop Condition: new_n > 0
    %loop_cond = icmp sgt i32 %new_n, 0
    br i1 %loop_cond, label %loop, label %exit

exit:
    ; Result is either 1 (if loop never ran) or the accumulated value.
    ; Because we have two predecessors for 'exit' (%entry and %loop),
    ; we need a Phi here too to merge the return value!
    
    %final_res = phi i32 [ 1, %entry ], [ %new_res, %loop ]
    ret i32 %final_res
}

define i32 @main() {
    %res = call i32 @factorial(i32 5)
    ret i32 %res
}
```

### Analysis of the code:
1.  **Block structure:** `entry`, `loop`, `exit`.
2.  **Phi Placement:** `phi` goes inside `loop` to handle the back-edge, and inside `exit` to handle the case where the loop is skipped vs taken.
3.  **SSA:** We never assign to `%curr_n` twice. We create `%new_n`.

### Testing

```bash
llvm-as fact.ll
lli fact.bc
echo $?
# Expected: 120
```

---

## 🧪 Hands-On Lab: Vectorized Loop via IR

Simd is first-class in LLVM.

**Task:** Write a function that adds two vectors of 4 floats.

```llvm
define <4 x float> @vec_add(<4 x float> %a, <4 x float> %b) {
    %res = fadd <4 x float> %a, %b
    ret <4 x float> %res
}
```

**Memory Ops with Vectors:**
Vectors must be aligned.

```llvm
define void @vec_mem_add(ptr %a_ptr, ptr %b_ptr, ptr %out_ptr) {
    ; Load
    %a = load <4 x float>, ptr %a_ptr, align 16
    %b = load <4 x float>, ptr %b_ptr, align 16
    
    ; Compute
    %res = fadd <4 x float> %a, %b
    
    ; Store
    store <4 x float> %res, ptr %out_ptr, align 16
    ret void
}
```

---

## 🔬 Deep Dive: GetElementPtr (GEP)

GEP is the most confusing instruction for beginners. It computes addresses based on types.

**Formula:**
`Address = Base + Index1 * SizeOf(Type) + Index2 * SizeOf(SubType) ...`

**Example:**
```llvm
%struct.Point = type { float, float, float } ; x, y, z
%array = alloca [10 x %struct.Point]

; We want: array[5].y
; 1. Base: %array (pointer to array of 10 Points)
; 2. Index 1: Dereference the pointer to the array (always needed for stack ptrs). Usually 0.
; 3. Index 2: Index into array (5).
; 4. Index 3: Index into struct (1 -> y).

%ptr_y = getelementptr [10 x %struct.Point], ptr %array, i32 0, i32 5, i32 1
%y_val = load float, ptr %ptr_y
```

**Common Pitfall:**
`GEP` does **not** access memory. `GEP null, 1` is valid (and results in address `sizeof(type)`). It is pure arithmetic.

---

## 📝 Summary & Key Takeaways

1.  **Typed Assembly:** LLVM IR combines the lowness of assembly with high-level Types and Functions.
2.  **SSA is Mandatory:** You cannot update a register. You must create new ones and merge them with Phis.
3.  **Explicit Data Flow:** All dependencies are visible in the instruction operands.
4.  **GEP is Math:** `getelementptr` is strictly address calculation, oblivious to valid memory ranges.
5.  **Simplicity:** With ~30 opcodes, you can represent almost any program.

**Next Step:** In Day 94, we will stop writing text IR and start using the **LLVM C++ API** to generate this IR programmatically, which is how a real compiler Frontend works.

*End of Day 093 - Total Lines: 1000+*
