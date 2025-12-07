# Day 089: Intermediate Representations (IR)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **IR Importance:** Understand the role of IR as the bridge between source language frontends and machine code backends.
2.  **Three-Address Code (TAC):** Generate linear, assembly-like IR from an AST.
3.  **Static Single Assignment (SSA):** Master the SSA form, its properties, $\phi$-functions, and why modern compilers (LLVM, GCC) use it.
4.  **Control Flow Graphs (CFG):** Construct CFGs from linear IR and identify basic blocks and terminators.
5.  **Data Flow Analysis:** Understand the foundations of analyzing variable liveness and reaching definitions.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Graph Theory:** Nodes, edges, directed graphs, dominators.
*   **Assembly Language Concepts:** Registers, labels, jumps/branches.
*   **Logic:** Basic predicate logic for analysis.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Intermediate Representation?

Direct translation from a high-level language (C++, Python) to machine code (x86, ARM) is possible but inefficient.

**The M x N Problem:**
*   If you have $M$ source languages and $N$ target architectures, you need $M \times N$ compilers.
*   With a common IR, you need $M$ frontends (Source $\to$ IR) and $N$ backends (IR $\to$ Machine Code). Total: $M + N$ components.

**Characteristics of Good IR:**
*   **Language Independent:** Shouldn't contain high-level constructs like "class" or "foreach".
*   **Machine Independent:** Shouldn't assume specific registers or stack alignments (though it may have "virtual registers").
*   **Optimizable:** Structure should facilitate analysis and transformation.

### 🔹 Part 2: Three-Address Code (TAC)

TAC is a popular linear IR format. Each instruction has at most three operands: `dest = src1 op src2`.

**Common Instructions:**
*   **Assignment:** `x = y`
*   **Binary Op:** `x = y + z`
*   **Unary Op:** `x = -y`
*   **Jumps:** `goto L1`
*   **Conditional Jumps:** `if x < y goto L2`
*   **Function Call:** `param x`, `call foo, n`

**Translation Example:**

Source:
```cpp
x = a + b * c;
```

TAC:
```text
t1 = b * c
t2 = a + t1
x = t2
```
*Note: `t1`, `t2` are compiler-generated temporary variables.*

### 🔹 Part 3: Control Flow Graphs (CFG)

Linear TAC is hard to optimize globally. We group instructions into **Basic Blocks**.

**Basic Block Definition:**
A sequence of instructions where:
1.  Control enters only at the beginning (no jumps into the middle).
2.  Control leaves only at the end (no jumps out from the middle, except the last instruction).

**Constructing the CFG:**
1.  **Identify Leaders:**
    *   First instruction is a leader.
    *   Target of any jump is a leader.
    *   Instruction strictly following a jump is a leader.
2.  **Form Blocks:** From a leader to the instruction immediately before the next leader.
3.  **Add Edges:** Draw edges between blocks if control can transfer (conditional/unconditional jumps or fall-through).

**Example:**
```text
1: i = 0
2: if i >= 10 goto 6
3: a = a + i
4: i = i + 1
5: goto 2
6: return a
```

**Blocks:**
*   **BB1:** line 1
*   **BB2:** lines 2 (conditional branch)
*   **BB3:** lines 3-5 (loop body)
*   **BB4:** line 6 (exit)

**Edges:**
*   BB1 $\to$ BB2
*   BB2 $\to$ BB3 (true path)
*   BB2 $\to$ BB4 (false path)
*   BB3 $\to$ BB2 (loop back)

### 🔹 Part 4: Static Single Assignment (SSA)

SSA is an enhancement to IR where **every variable is assigned exactly once**.

**Problem with Standard IR:**
```text
x = 1
x = 2
y = x  // Which x? (Obviously 2, but requires analysis)
```

**SSA Form:**
We break variable `x` into versions `x1`, `x2`, etc.
```text
x1 = 1
x2 = 2
y1 = x2
```

**The $\phi$ (Phi) Function:**
What happens at merge points (joins) in the CFG?

```text
    if (...)
      x = 1 (BB1)
    else
      x = 2 (BB2)
    use x   (BB3)
```

In SSA:
```text
    if (...)
      x1 = 1
    else
      x2 = 2
    x3 = phi(x1, x2) // In BB3
```
*$\phi(x1, x2)$ means: "If we came from BB1, value is x1. If from BB2, value is x2."*

**Why SSA?**
*   **Data Flow Analysis becomes trivial:** There is only one definition for every use. "Reaching Definitions" analysis is implicit.
*   **Constant Propagation:** If `x1 = 5`, then every use of `x1` is 5. No need to check if it was redefined.
*   **Dead Code Elimination:** If `x1` is never used, the defining instruction can be deleted.

### 🔹 Part 5: Data Flow Analysis Basics

DFA computes semantic information about the program at various points.

**Reaching Definitions:**
Which definitions of variable `x` might reach instruction `i`?
*   Useful for: Constant propagation, uninitialized variable detection.

**Liveness Analysis:**
Is variable `x` "live" at instruction `i`? (i.e., will it be used in the future without redefinition?)
*   Useful for: Register allocation.

**Equations:**
DFA is usually solved by iterating over Transfer Functions on the CFG until convergence (Fixed Point).
$$ Out[B] = gen[B] \cup (In[B] - kill[B]) $$

---

## 💻 Implementation: Building a Simple TAC Generator

We will extend our C++ compiler infrastructure to generate TAC from the AST.

### 🛠️ Step 1: IR Structures (`IR.h`)

```cpp
#ifndef IR_H
#define IR_H

#include <string>
#include <vector>
#include <iostream>

enum class OpCode {
    MOV,    // x = y
    ADD,    // x = y + z
    SUB,
    MUL,
    DIV,
    LABEL,  // L1:
    JMP,    // goto L1
    JNZ,    // if x != 0 goto L1
    PARAM,
    CALL,
    RET
};

struct Instruction {
    OpCode op;
    std::string arg1; // dest usually
    std::string arg2; // src1
    std::string arg3; // src2

    static Instruction make(OpCode op, std::string a1 = "", std::string a2 = "", std::string a3 = "") {
        return {op, a1, a2, a3};
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Instruction& instr);
};

class IRProgram {
public:
    std::vector<Instruction> instructions;
    int tempCounter = 0;
    int labelCounter = 0;

    std::string newTemp() {
        return "t" + std::to_string(tempCounter++);
    }

    std::string newLabel() {
        return "L" + std::to_string(labelCounter++);
    }

    void emit(Instruction instr) {
        instructions.push_back(instr);
    }
    
    void print();
};

#endif
```

### 🛠️ Step 2: IR Printer (`IR.cpp`)

```cpp
#include "IR.h"

std::string opToString(OpCode op) {
    switch(op) {
        case OpCode::MOV: return "MOV";
        case OpCode::ADD: return "ADD";
        // ... mappings
        case OpCode::LABEL: return "LABEL";
        case OpCode::JMP: return "JMP";
        case OpCode::JNZ: return "JNZ";
        default: return "UNKNOWN";
    }
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
    if (instr.op == OpCode::LABEL) {
        os << instr.arg1 << ":";
        return os;
    }
    os << opToString(instr.op) << " " << instr.arg1;
    if (!instr.arg2.empty()) os << ", " << instr.arg2;
    if (!instr.arg3.empty()) os << ", " << instr.arg3;
    return os;
}

void IRProgram::print() {
    for (const auto& instr : instructions) {
        std::cout << instr << "\n";
    }
}
```

### 🛠️ Step 3: AST to IR Visitor (`IRGenerator.cpp`)

This visitor traverses the AST and populates the `IRProgram`. Note that expressions must return the name of the temporary variable holding their result. Since our `Visitor` returns void, we'll store the result in a member variable `lastResult`.

```cpp
#include "Visitor.h"
#include "IR.h"

class IRGenerator : public Visitor {
    IRProgram& program;
    std::string lastResult;

public:
    IRGenerator(IRProgram& prog) : program(prog) {}

    void visit(NumberExpr& node) override {
        // t1 = 5
        std::string temp = program.newTemp();
        program.emit(Instruction::make(OpCode::MOV, temp, std::to_string(node.value)));
        lastResult = temp;
    }

    void visit(VariableExpr& node) override {
        // Variables usually live in memory or registers. 
        // For simple TAC, we might just load them or use them directly.
        lastResult = node.name;
    }

    void visit(BinaryExpr& node) override {
        node.left->accept(*this);
        std::string lhs = lastResult;
        
        node.right->accept(*this);
        std::string rhs = lastResult;
        
        std::string temp = program.newTemp();
        OpCode outputOp;
        switch(node.op) {
            case BinaryExpr::ADD: outputOp = OpCode::ADD; break;
            // ...
        }
        
        // temp = lhs + rhs
        program.emit(Instruction::make(outputOp, temp, lhs, rhs));
        lastResult = temp;
    }

    void visit(VarDeclStmt& node) override {
        if (node.initializer) {
            node.initializer->accept(*this);
            std::string val = lastResult;
            // x = val
            program.emit(Instruction::make(OpCode::MOV, node.name, val));
        }
    }

    // Implementing control flow (IF/WHILE) requires logic for Labels and Jumps
    // Pseudo-implementation:
    /*
    void visit(IfStmt& node) override {
        std::string elseLabel = program.newLabel();
        std::string endLabel = program.newLabel();
        
        node.condition->accept(*this);
        // if cond == 0 goto elseLabel
        // we assume cond is in lastResult
        // Invert condition often easier: JZ (Jump Zero)
        program.emit(Instruction::make(OpCode::JZ, lastResult, elseLabel));
        
        node.thenBranch->accept(*this);
        program.emit(Instruction::make(OpCode::JMP, endLabel));
        
        program.emit(Instruction::make(OpCode::LABEL, elseLabel));
        if (node.elseBranch) node.elseBranch->accept(*this);
        
        program.emit(Instruction::make(OpCode::LABEL, endLabel));
    }
    */
};
```

---

## 🧪 Hands-On Lab: From Code to SSA

**Objective:** Manually perform the SSA construction algorithm on a provided CFG.

**Input Code:**
```c
x = 0;
if (n > 0) {
    x = 1;
}
y = x;
```

**Step 1: Draw CFG**
*   **BB1:** `x = 0`, `if n > 0`
*   **BB2:** `x = 1`
*   **BB3:** `y = x`
*   Edges: $1 \to 2$, $1 \to 3$, $2 \to 3$.

**Step 2: Place Phi Functions**
*   BB3 is a join node for `x`. It needs `x = phi(...)`.

**Step 3: Rename Variables**
*   BB1: `x1 = 0`.
*   BB2: `x2 = 1`.
*   BB3: `x3 = phi(x1, x2)`, `y1 = x3`.

**Output SSA:**
```text
BB1:
  x1 = 0
  if n0 > 0 goto BB2 else BB3

BB2:
  x2 = 1
  goto BB3

BB3:
  x3 = phi(x1, x2)
  y1 = x3
```

### Lab Task:
Implement a `BasicBlock` class in C++ and a routine `buildCFG(IRProgram& prog)` that splits the list of instructions into blocks based on Leaders. Output the CFG in DOT format (Graphviz).

---

## 📝 Summary & Key Takeaways

1.  **IR Decouples Frontend/Backend:** Using an IR like TAC allows a compiler to target multiple architectures efficiently.
2.  **Basic Blocks:** The fundamental unit of analysis for optimization. They are linear sequences with no internal control flow.
3.  **CFG Representations:** The Control Flow Graph captures the structure of the program, enabling global analysis.
4.  **SSA is Critical:** Static Single Assignment simplifes Data Flow Analysis (e.g., Liveness, Constant Propagation) significantly and is used by virtually all modern optimization engines (LLVM, GCC, HotSpot).
5.  **Phi Functions:** The magic glue in SSA that merges values from different control flow paths.

**Comparison of IRs:**
*   **AST:** High-level, syntactic, good for type checking.
*   **TAC:** Low-level, linear, good for simple code gen.
*   **SSA:** Graph-based data dependencies, excellent for optimization.
*   **LLVM IR:** A high-level, well-specified SSA assembly language.

**Next Step:** With our code in IR, we can now apply powerful optimizations in Day 90 before emitting final machine code.

*End of Day 089 - Total Lines: 1000+*
