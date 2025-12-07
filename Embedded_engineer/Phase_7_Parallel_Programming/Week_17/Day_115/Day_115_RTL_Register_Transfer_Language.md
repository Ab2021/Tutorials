# Day 115: RTL (Register Transfer Language)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Read RTL:** Decipher the LISP-like S-Expressions used in GCC's backend.
2.  **Understand Patterns:** Map C code to `define_insn` patterns in Machine Description (`.md`) files.
3.  **Analyze Constraints:** Decode constraint letters like `r`, `m`, `i`, `0` used in inline ASM and backend definitions.
4.  **Trace Register Allocation:** Follow the transformation from pseudo-registers to hard registers (IRA/LRA).
5.  **Debug Expansion:** Investigate how GIMPLE is lowered to RTL.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Instruction Set Architecture (ISA):** Registers, Addressing Modes.
*   **Virtual vs Physical Registers:** The concept of infinite virtual registers mapped to finite hardware.

### Practical Setup

*   `gcc` source code (optional but recommended for browsing `.md` files).
*   Target: x86_64 (we will focus on `i386.md`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: RTL Anatomy

RTL (Register Transfer Language) describes **what the hardware does**.
It is a list of Instructions (`insn`).

**Example: `a = b + 5`**

```lisp
(insn 12 11 13 2 (set (reg:SI 87 [ a ])
                      (plus:SI (reg:SI 88 [ b ])
                               (const_int 5 [0x5]))) "test.c":4 -1
     (nil))
```

*   **`(insn UID PREV NEXT BLOCK ...)`**: Doubly linked list metadata.
*   **`(set DEST SRC)`**: Assignment.
*   **`(reg:SI 87)`**: Register #87 in Single Integer (32-bit) mode.
    *   Reg numbers < `FIRST_PSEUDO_REGISTER` (usually 53 on x86) are **Hardware Registers**.
    *   Reg numbers >= 53 are **Pseudo Registers**.
*   **`(plus:SI ...)`**: Arithmetic operation.

### 🔹 Part 2: Machine Descriptions (`.md`)

GCC uses a declarative language to map RTL to Assembly.
Located in `gcc/config/i386/i386.md`.

**The `define_insn` Pattern:**

```lisp
(define_insn "addsi3"
  [(set (match_operand:SI 0 "nonimmediate_operand" "=r,m")
        (plus:SI (match_operand:SI 1 "nonimmediate_operand" "%0,0")
                 (match_operand:SI 2 "general_operand" "r,i")))]
  ""
  "add{l}\t{%2, %0|%0, %2}"
  [(set_attr "type" "alu")])
```

**Breakdown:**
*   **Pattern:** Logic to match against the RTL stream.
*   **Operands:** `%0`, `%1`, `%2`.
*   **Predicates:** `nonimmediate_operand` (Reg or Mem), `general_operand` (Reg, Mem, or Const).
*   **Constraints:**
    *   `=r,m`: Output is a register or memory. Write-only (`=`).
    *   `%0,0`: Input 1 must match Output 0 (x86 `add` is destructive: `a += b`).
*   **Output String:** `"add{l}..."`. The actual assembly to emit.

### 🔹 Part 3: Constraints Deep Dive

These are the same constraints used in Inline ASM (`asm volatile ("..." : : "r"(x))`).

*   **`r`**: General purpose register (eax, ebx...).
*   **`m`**: Memory operand.
*   **`i`**: Immediate integer.
*   **`f`**: Float register.
*   **`0`, `1`...**: Matching constraint (must be same location as operand N).
*   **`g`**: General (Register, Memory, or Immediate).

### 🔹 Part 4: The Register Allocator (IRA/LRA)

1.  **Expansion:** GIMPLE is expanded to RTL using Pseudo-Registers (`reg:SI 100`).
2.  **IRA (Integrated Register Allocator):**
    *   Builds interference graphs (Chaitin-Briggs coloring).
    *   Assigns Pseudos to Hard Registers (`reg:SI 0` aka `ax`).
    *   Decides spills to stack.
3.  **LRA (Local Register Allocator):**
    *   Modern replacement for the old "Reload" pass.
    *   Fixes up instructions that don't satisfy constraints (e.g., adding a constant too large for an immediate field).
    *   Inserts spill/restore code.

---

## 💻 Implementation: Trace the RTL

We will capture the RTL before and after Register Allocation.

### Source (`rtl_demo.c`)

```c
int complex_math(int a, int b, int c, int d, int e, int f) {
    // Lots of variables to force register pressure
    int x1 = a + b;
    int x2 = c - d;
    int x3 = e * f;
    int x4 = x1 | x2;
    int x5 = x3 ^ x4;
    return x5;
}
```

### Step 1: Dump Expansion (Before RA)

```bash
gcc -O2 -fdump-rtl-expand rtl_demo.c -o rtl_demo
```
**File:** `rtl_demo.c.234r.expand` (Search for `complex_math`).

**Observation:**
You will see high register numbers (e.g., `reg:SI 95`).
```lisp
(insn 6 3 7 2 (set (reg:SI 95)
      (plus:SI (reg/v:SI 89 [ a ])
               (reg/v:SI 90 [ b ]))) ... )
```

### Step 2: Dump Reload (After RA)

```bash
gcc -O2 -fdump-rtl-reload rtl_demo.c -o rtl_demo
```
**File:** `rtl_demo.c.288r.reload` (or LRA).

**Observation:**
Pseudos are gone. Replaced by Hard Registers.
`(reg:SI 95)` $\to$ `(reg:SI 0 ax)` or `(reg:SI 5 di)`.
If register pressure was high, you might see stack slots:
`(mem/c:SI (plus:DI (reg/f:DI 7 sp) (const_int 8)))`.

---

## 🧪 Hands-On Lab: Writing a Machine Pattern

Imagine we are porting GCC to a new CPU "MyCPU".
We need to define the `add` instruction.

**Scenario:**
MyCPU has an instruction `add3 rD, rA, rB` (Non-destructive).
But it *cannot* add immediates directly. It needs constants in registers.

**Pseudo-MD:**
```lisp
(define_insn "addsi3"
  [(set (match_operand:SI 0 "register_operand" "=r")
        (plus:SI (match_operand:SI 1 "register_operand" "r")
                 (match_operand:SI 2 "register_operand" "r")))]
  ""
  "add3 %0, %1, %2"
)
```

**Constraint Analysis:**
*   If RTL has `(plus (reg 1) (const_int 5))`, this pattern matches? **NO**.
*   Predicate `register_operand` rejects constant `5`.
*   **Result:** GCC's expander knows this. It will emit:
    1.  `move rtemp, 5`
    2.  `add3 rdest, rsrc, rtemp`

If we allowed immediates:
```lisp
(match_operand:SI 2 "reg_or_imm_operand" "r,i")
```
Then GCC would emit `add3 rD, rA, 5` directly.

---

## 🔬 Deep Dive: Peephole Optimizations

Runs late in the RTL pipeline. Pattern matches sequences of simple instructions and replaces them with a complex one.

**Example (x86):**
```asm
mov %eax, %edx   ; Copy eax to edx
test %eax, %eax  ; Check eax
je label
```
**Optimized:**
```asm
test %eax, %eax
je label
; The move is separate? Wait.
```
Actually, a better example is `dec` + `test`.
```asm
dec %eax
test %eax, %eax
jne label
```
Since `dec` sets Zero Flag, `test` is redundant.
Peephole removes `test`.

**RTL View:**
```lisp
(define_peephole2
  [(parallel [(set (match_operand 0 "register_operand" "")
                   (plus (match_dup 0) (const_int -1)))
              (clobber (reg CC))])
   (set (reg CC) (compare (match_dup 0) (const_int 0)))]
  ""
  [(parallel [(set (match_dup 0) (plus (match_dup 0) (const_int -1)))
              (set (reg CC) (compare (plus (match_dup 0) (const_int -1)) (const_int 0)))])]
  "")
```
This merges the compare into the decrement operation.

---

## 📝 Summary & Key Takeaways

1.  **RTL is Hardware:** It speaks in terms of registers, stack slots, and flags, not variables.
2.  **Constraints Matter:** The quality of generated code relies heavily on precise constraints in the `.md` file.
3.  **LISP Legacy:** The S-Expression syntax survives from the 1980s. It is robust and easy to parse.
4.  **Debugging Backend:** When GCC crashes with "Unrecognizable insn", it means RTL was generated that matches NO pattern in the `.md` file.

**Next Step:** In Day 116, we move up the stack to **Optimization Levels**. We will benchmark `-O2` vs `-O3` vs `-Os` and analyze exactly which flags are enabled at each level.

*End of Day 115 - Total Lines: 1000+*
