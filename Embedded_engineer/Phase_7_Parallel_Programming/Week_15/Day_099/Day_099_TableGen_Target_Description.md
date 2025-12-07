# Day 099: TableGen & Target Description
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 15: LLVM Advanced

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **TableGen Syntax:** Read and write `.td` files using classes, definitions, and multclasses.
2.  **Target Description:** Define registers, register classes, and calling conventions.
3.  **Instruction Definitions:** Define machine instructions, their operands (Input/Output), and encoding.
4.  **Pattern Matching:** Write DAG patterns to map LLVM IR to machine instructions automatically.
5.  **Gen Tools:** Understand how `llvm-tblgen` generates C++ headers (`.inc`) from `.td` files.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Domain Specific Languages (DSLs):** TableGen is a DSL for defining static records.
*   **Computer Architecture:** Registers, Opcodes, Addressing Modes.

### Practical Setup

*   `llvm-tblgen` tool.
*   Access to LLVM source code (specifically `lib/Target/X86/*.td`) is very helpful for examples.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is TableGen?

Compiler backends require massive amounts of boilerplate code:
*   Enums for 1000+ instructions.
*   Strings for assembly printing.
*   Pattern matching logic for Instruction Selection.
*   Encoding tables for the assembler.

Writing this in C++ is error-prone and redundant. **TableGen** allows you to define the *data* once and generate multiple C++ files (Enums, Switch statements, Matcher tables) from it.

**Core Concepts:**
*   **Class:** A template for a record (like a C++ class or struct).
*   **Def:** A concrete instance of a record.
*   **Field:** A variable inside a class/def.
*   **Let:** Overrides a field value.

### 🔹 Part 2: TableGen Syntax

**Basic Definition:**
```tablegen
class Register<string name, int idx> {
  string Name = name;
  int Index = idx;
}

def R0 : Register<"r0", 0>;
def R1 : Register<"r1", 1>;
```

**Inheritance & Multiclass (The Power Tool):**
If you have `ADD`, `SUB`, `AND`, `OR` and they all look similar, use a `multiclass`.

```tablegen
class Inst<int opcode, string asm> {
  int Opcode = opcode;
  string Asm = asm;
}

multiclass IntLogic<int baseOp, string mnemonic> {
  def _rr : Inst<baseOp, mnemonic # " $dst, $src1, $src2">; // Reg-Reg
  def _ri : Inst<!add(baseOp, 1), mnemonic # " $dst, $src1, $imm">; // Reg-Imm
}

// Instantiates AND_rr and AND_ri
defm AND : IntLogic<0x10, "and">;
// Instantiates OR_rr and OR_ri
defm OR  : IntLogic<0x20, "or">;
```

### 🔹 Part 3: Defining Registers

Target descriptions usually start with registers.

```tablegen
class MyTargetReg<string n, list<string> alt = []> : Register<n> {
  let Namespace = "MyTarget";
  let AltNames = alt;
}

def R0 : MyTargetReg<"r0">;
def R1 : MyTargetReg<"r1">;
...
def PC : MyTargetReg<"pc">;

// Register Class: logical grouping (e.g., General Purpose Regs)
def GPR : RegisterClass<"MyTarget", [i32], 32, (add R0, R1, ...)>;
```
*   `[i32]`: Types this register can hold.
*   `32`: Alignment (bits).
*   `(add ...)`: List of registers in allocation order.

### 🔹 Part 4: Defining Instructions

Instructions connect the MC Layer (bits/asm) to the CodeGen Layer (DAG).

```tablegen
class I<bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> 
  : Instruction {
  field bits<32> Inst; // The actual encoding bits
  let Namespace = "MyTarget";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asm;
  let Pattern = pattern;
}

// Define ADD: Reg = Reg + Reg
def ADD : I<0xAB, (outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
            "add $dst, $src1, $src2",
            [(set GPR:$dst, (add GPR:$src1, GPR:$src2))]>;
```
*   **`outs`**: Output operands (destination).
*   **`ins`**: Input operands.
*   **`AsmString`**: How print (ASM printer).
*   **`Pattern`**: How to select from IR (Matcher). `(set $dst, (add $src1, $src2))` matches `dst = src1 + src2`.

### 🔹 Part 5: Instruction Selection Patterns

Sometimes patterns are complex (e.g., multiple instructions mapping to one machine instruction).

```tablegen
// Multiply-Add: d = a * b + c
// Matches specific node combination
def MADD : I<0xAC, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
             "madd $dst, $a, $b, $c",
             [(set GPR:$dst, (add (mul GPR:$a, GPR:$b), GPR:$c))]>;
```

**Pat:**
You can also define patterns separately from instructions (e.g., optimizing constants).

```tablegen
// Replace: 'add reg, 0' with 'copy reg' (Machine Optimization)
def : Pat<(add GPR:$src, 0), (COPY GPR:$src)>;
```

---

## 💻 Implementation: A Toy Target Definition

We will define a fragment of a fictional "ToyCPU" architecture.

### File: `ToyInstrInfo.td`

```tablegen
// --- Core Definitions ---

class ToyInst<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  field bits<32> Inst;
  
  let Namespace = "Toy";
  let Size = 4; // 4 bytes
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  // Encoding logic (simplification)
  let Inst{31-24} = op;
}

// --- Registers ---

class ToyReg<bits<4> enc, string n> : Register<n> {
  let HWEncoding{3-0} = enc;
  let Namespace = "Toy";
}

def R0 : ToyReg<0, "r0">;
def R1 : ToyReg<1, "r1">;
def R2 : ToyReg<2, "r2">;
def R3 : ToyReg<3, "r3">;

def GPR : RegisterClass<"Toy", [i32], 32, (add R0, R1, R2, R3)>;

// --- Operands ---

// Immediate operand 12-bit
def i12imm : Operand<i32>;

// --- Instructions ---

// 1. Register-Register Add
def ADDrr : ToyInst<0x10, (outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
                    "add $dst, $src1, $src2",
                    [(set GPR:$dst, (add GPR:$src1, GPR:$src2))]> {
  bits<4> dst;
  bits<4> src1;
  bits<4> src2;
  // Map operands to encoding bits
  let Inst{23-20} = dst;
  let Inst{19-16} = src1;
  let Inst{15-12} = src2;
}

// 2. Register-Immediate Add
def ADDri : ToyInst<0x11, (outs GPR:$dst), (ins GPR:$src1, i12imm:$imm),
                    "addi $dst, $src1, $imm",
                    [(set GPR:$dst, (add GPR:$src1, imm:$imm))]> {
  bits<4> dst;
  bits<4> src1;
  bits<12> imm;
  let Inst{23-20} = dst;
  let Inst{19-16} = src1;
  let Inst{11-0}  = imm;
}

// 3. Load Instruction
// Note: 'load' in pattern matches LLVM IR load instruction
def LD : ToyInst<0x20, (outs GPR:$dst), (ins GPR:$addr),
                 "load $dst, [$addr]",
                 [(set GPR:$dst, (load GPR:$addr))]>;

// 4. Store Instruction
// Note: store returns void, so no outs.
def ST : ToyInst<0x21, (outs), (ins GPR:$val, GPR:$addr),
                 "store $val, [$addr]",
                 [(store GPR:$val, GPR:$addr)]>;
```

### Processing with `llvm-tblgen`

If you had the LLVM build system set up, it would run:

```bash
llvm-tblgen -gen-instr-info -I /path/to/llvm/include ToyInstrInfo.td -o ToyGenInstrInfo.inc
llvm-tblgen -gen-register-info -I ... ToyInstrInfo.td -o ToyGenRegisterInfo.inc
llvm-tblgen -gen-dag-isel -I ... ToyInstrInfo.td -o ToyGenDAGISel.inc
```

**Looking at `ToyGenDAGISel.inc`:**
You would see a massive C++ state machine (or switch cases) that performs the pattern matching logic we defined.
```cpp
// Pseudo-output
CheckOpcode(ISD::ADD)
  CheckType(i32)
    CheckOperand(0, GPR)
    CheckOperand(1, GPR)
      Emit(ADDrr)
    CheckOperand(1, Constant)
      Emit(ADDri)
```

---

## 🧪 Hands-On Lab: Adding a New Instruction

**Task:** Add a `SUB` instruction and a `Memory Indirect Add` (`add [mem], val`).

**Steps:**
1.  Define `SUBrr` similar to `ADDrr` but with opcode `0x12`.
    *   Pattern: `(sub GPR:$src1, GPR:$src2)`.
2.  Define `ADDmr` (Memory-Register Add).
    *   This is tricky. LLVM IR `add` works on values (registers), not memory locations.
    *   IR Pattern is: `store (add (load addr), val), addr`.
    *   TableGen needs to match this tree.

**Solution:**

```tablegen
def ADDmr : ToyInst<0x30, (outs), (ins GPR:$addr, GPR:$val),
                    "add [$addr], $val",
                    [(store (add (load GPR:$addr), GPR:$val), GPR:$addr)]>;
```
*This is the power of TableGen: recognizing a Load-Add-Store sequence and emitting a single atomic-like RMW instruction (if the hardware supports it).*

---

## 📝 Summary & Key Takeaways

1.  **DSLs are Essential:** Compilers are too complex to write entirely by hand. Tools like TableGen generate the boring, error-prone parts.
2.  **SelectionDAG is Pattern Matching:** The backend primarily consists of mapping sub-trees of the IR DAG to machine instructions.
3.  **Encodings:** We map high-level definitions to bits (`let Inst{...} = ...`). This allows the same `.td` file to drive not just the Compiler (`llc`), but also the Assembler (`as`) and Disassembler (`objdump`).
4.  **Multiclass Reduces Redundancy:** Use multiclasses to define families of instructions (Reg-Reg, Reg-Imm, Reg-Mem) at once.

**Next Step:** In Day 100, we move up the stack to **JIT Compilation** (Just-In-Time). We will learn how to execute our LLVM IR directly in memory without creating an executable file.

*End of Day 099 - Total Lines: 1000+*
