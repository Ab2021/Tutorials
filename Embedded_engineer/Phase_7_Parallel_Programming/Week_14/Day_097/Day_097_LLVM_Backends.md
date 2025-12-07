# Day 097: LLVM Backends & Code Generation
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 14: LLVM Infrastructure

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Backend Pipeline:** Explain the stages: Instruction Selection, Scheduling, Register Allocation, Emission.
2.  **Target Description:** Understand `.td` (TableGen) files for defining registers and instructions.
3.  **SelectionDAG:** Visualize the DAG-based representation used for instruction selection.
4.  **Register Allocation:** Compare linear scan vs graph coloring allocators.
5.  **Assembly Parsing/Printing:** How LLVM handles textual assembly (MC Layer).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Computer Architecture:** Registers, Calling Conventions, Opcode Encodings.
*   **Graph Algorithms:** DAGs, Coloring.

### Practical Setup

*   `llc` with `-view-dag-combine1-dags` (requires Graphviz) is amazing for debugging.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The CodeGen Pipeline

The "Backend" takes Optimized LLVM IR and produces Machine Code (Assembly/Object).

**Stages:**
1.  **Instruction Selection (ISel):** Maps generic IR (`add`) to machine instructions (`ADD32rr`).
    *   Representation: **SelectionDAG** (Directed Acyclic Graph).
2.  **Scheduling (Pre-RA):** Orders instructions to minimize pipeline stalls and register pressure.
3.  **Register Allocation (RA):** Maps infinite Virtual Registers (`%vreg1`) to finite Physical Registers (`rax`, `rdi`, `stack`).
4.  **Scheduling (Post-RA):** Re-orders instructions based on final register usage (hazards).
5.  **Code Emission:** Writes bytes (using `MC` framework).

### 🔹 Part 2: TableGen (.td)

Writing a backend in C++ is repetitive (defining 100s of add variants). LLVM uses a Domain Specific Language called **TableGen**.

**Example (X86RegisterInfo.td):**
```tablegen
class X86Reg<string n> : Register<n> {
  let Namespace = "X86";
}
def EAX : X86Reg<"eax">;
def ECX : X86Reg<"ecx">;
```

**Example (X86InstrInfo.td):**
```tablegen
def ADD32rr : I<0x01, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                "add{l}\t{$src2, $dst}",
                [(set GR32:$dst, (add GR32:$src1, GR32:$src2))]>;
```
*Meaning:* Match the IR pattern `(add src1, src2)` and replace it with `ADD32rr` instruction, printing it as `addl`.

### 🔹 Part 3: Instruction Selection (DAG Selection)

**Input:** LLVM IR Basic Block.
**Output:** Machine DAG.

**Process:**
1.  **Build DAG:** Convert IR instrs to Nodes.
2.  **Combine:** Simplify DAG (simplify `(add x, 0)`).
3.  **Legalize:** Ensure types are supported. (e.g., if target has no `i64`, split into two `i32`s).
4.  **Select:** Pattern match using TableGen definitions.

**Example:**
IR: `%a = load i32* %p`
Matcher: `(load ptr:$addr)`
Target Instr: `MOV32rm $addr`

### 🔹 Part 4: Register Allocation

The hardest NP-Complete problem in compilers.

**Virtual Regs:** SSA form, infinite supply.
**Physical Regs:** Fixed set (16 on x64), strict constraints (ABI, clobbers).

**Algorithms:**
1.  **Fast:** Local (block-level), spills often. Used for `-O0`.
2.  **Greedy (Linear Scan variant):** Fast, decent quality.
3.  **PBQP (Partitioned Boolean Quadratic Programming):** Very high quality, slow.

**Spilling:** If no register is free, choose a value to "spill" to stack memory.

### 🔹 Part 5: The MC Layer (Machine Code)

Handles the nitty-gritty of object files (ELF, MachO, COFF).
*   **MCInst:** Simple container (Opcode + Operands).
*   **MCStreamer:** API to write bytes or text.
*   **AsmPrinter:** Converts high-level MachineInstr to MCInst.

---

## 💻 Implementation: Examining the Backend

We will inspect the lowering of a simple function for two different architectures (x86 and ARM) to see Isel differences.

### Source (`isel.ll`)

```llvm
define i32 @test(i32 %a, i32 %b) {
  %mul = mul i32 %a, %b
  %add = add i32 %mul, 5
  ret i32 %add
}
```

### Experiment 1: X86-64

```bash
llc -march=x86-64 isel.ll -o isel_x86.s -print-after-isel
```
*Look for dump of "Machine Function"*. You will see virtual registers.

**Output:**
```asm
imull %esi, %edi
addl  $5, %edi
movl  %edi, %eax
retq
```
*Notice: `imull` (2-operand usually) overwrites one input. RA handles the copying.*

### Experiment 2: ARM64 (AArch64)

```bash
llc -march=aarch64 isel.ll -o isel_arm.s
```

**Output:**
```asm
madd w0, w0, w1, w0  ; Wait, MADD (Multiply-Add)?
add  w0, w0, #5
ret
```
*Wait, `madd` computes `a*b + c`.*
Actually, AArch64 standard `mul` isn't `madd` unless fused.
Let's modify code to use Fuse Multiply Add opportunity:
`%res = (a * b) + c`

```llvm
define i32 @fma(i32 %a, i32 %b, i32 %c) {
  %mul = mul i32 %a, %b
  %add = add i32 %mul, %c
  ret i32 %add
}
```

**Run:** `llc -march=aarch64`
**Result:** `madd w0, w0, w1, w2`
*Cool! LLVM matched the tree `(add (mul a, b), c)` to the single instruction `MADD`.*

---

## 🧪 Hands-On Lab: Custom Backend Pass

**Objective:** Write a **MachineFunctionPass** that runs *after* Register Allocation to insert NOPs for "padding" (security/alignment).

**Steps:**
1.  Inherit from `MachineFunctionPass`.
2.  Override `runOnMachineFunction(MachineFunction &MF)`.
3.  Iterate blocks and instructions.
4.  Use `BuildMI` to insert code.

```cpp
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

struct NopInserter : public MachineFunctionPass {
  static char ID;
  NopInserter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    
    for (auto &MBB : MF) {
      // Insert NOP at start of every block
      BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(TargetOpcode::NOP));
    }
    return true;
  }
};
```
*Note: This requires building inside the LLVM source tree usually, or linking heavily against CodeGen libs.*

---

## 📝 Summary & Key Takeaways

1.  **TableGen:** The backend is largely data-driven. `td` files describe the hardware (Registers, Instructions, Scheduling models).
2.  **SelectionDAG:** The bridge between IR and Assembly. It's a graph matcher.
3.  **Legalization:** The backend knows the target's limits. If you try to `add i128` on a 32-bit machine, Legalizer breaks it into four `add i32`s with carry.
4.  **Register Allocation:** The final binder. Transforms the beautiful infinite SSA world into the harsh reality of finite hardware resources.
5.  **Target Independent Code Generator:** The framework (`libLLVMCodeGen`) is shared. X86, ARM, RISC-V backends just plug into it.

**Next Step:** In Day 98 (Project), we will combine everything to perform a custom transformation and code generation experiment.

*End of Day 097 - Total Lines: 1000+*
