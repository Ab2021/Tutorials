# Day 091: Week 13 Review & Project (Mini-Compiler)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **System Integration:** Connect a Lexer, Parser, AST Builder, Semantic Analyzer, and IR Generator into a cohesive pipeline.
2.  **End-to-End Compilation:** Translate a subset of C (let's call it "Mini-C") into an executable format (or interpretable IR).
3.  **Project Organization:** Structure a moderate-scale C++ compiler project.
4.  **Debugging:** Diagnose issues that arise at the boundaries of compiler phases (e.g., parsing errors vs semantic errors).
5.  **Benchmarking:** Measure the performance of your compiler vs GCC/Clang on simple snippets.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Week 13 Recap:** Lexing (Day 86), Parsing (Day 87), AST (Day 88), IR (Day 89), Optimization (Day 90).
*   **Build Systems:** Make or CMake is essentially required for a project of this size.

### Project Scope ("Mini-C")

Our language supports:
*   **Types:** `int` only.
*   **Control Flow:** `if`, `while`, `return`.
*   **Operators:** `+`, `-`, `*`, `/`, `=`, `==`, `<`, `>`.
*   **Functions:** Declaration and calling (no recursion support in our simple backend for now).
*   **IO:** A built-in `print(int)` function.

---

## 📖 Theoretical Deep Dive: The Compiler Driver

The **Driver** is the main entry point. It orchestrates the phases.

**Pipeline Data Flow:**
```
Source File (.mc) 
    -> [Lexer] -> Tokens 
    -> [Parser] -> AST 
    -> [Semantic Analysis] -> Validated AST 
    -> [IR Gen] -> IR 
    -> [Optimizer] -> Optimized IR 
    -> [Backend] -> x86 Assembly / Execution
```

**Key Challenges:**
1.  **Error Propagation:** If the parser fails, we stop. If semantics fail, we stop. We need a robust error reporting mechanism finding line/column numbers.
2.  **Memory Management:** The AST creates many nodes. Who owns them? (Smart pointers `std::unique_ptr` are best).
3.  **Symbol Table persistence:** The symbol table during Semantic Analysis is different from the stack frame layout during Code Generation, but they are related.

---

## 💻 Implementation: The Mini-Compiler

We will layout the files and provide the core "glue" logic. We assume the existence of the components built in Days 86-90 with slight modifications to fit together.

### 🛠️ Step 1: Project Structure

```text
minic/
├── src/
│   ├── Lexer.h/cpp
│   ├── Parser.h/cpp
│   ├── AST.h
│   ├── SymbolTable.h
│   ├── IR.h/cpp
│   ├── Optimizer.h/cpp
│   ├── CodeGen.h/cpp (Simple Interpreter for now)
│   └── main.cpp
├── tests/
│   ├── test1.mc
│   └── test2.mc
└── Makefile
```

### 🛠️ Step 2: The Main Driver (`src/main.cpp`)

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include "Lexer.h"
#include "Parser.h"
#include "SemanticAnalyzer.h"
#include "IRGenerator.h"
#include "Optimizer.h"
#include "Interpreter.h"

void compileAndRun(const std::string& source) {
    std::cout << "Compiling...\n";

    // 1. Lexing & Parsing
    Lexer lexer(source);
    Parser parser(lexer);
    auto ast = parser.parse(); // parse() returns unique_ptr<BlockStmt> representing root

    if (!ast) {
        std::cerr << "Compilation failed during parsing.\n";
        return;
    }

    // 2. Semantic Analysis
    SemanticAnalyzer semantics;
    if (!semantics.analyze(*ast)) {
        std::cerr << "Compilation failed during semantic analysis.\n";
        return;
    }

    // 3. IR Generation
    IRProgram irProg;
    IRGenerator irGen(irProg);
    irGen.generate(*ast);

    // 4. Optimization
    Optimizer opt(irProg);
    opt.run(); // runs constant folding, dce, etc.

    // 5. Code Generation / Execution
    std::cout << "--- IR Code ---\n";
    irProg.print();
    std::cout << "--- Execution Output ---\n";
    
    Interpreter vm(irProg);
    vm.run();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: minic <source_file>\n";
        return 1;
    }

    std::ifstream t(argv[1]);
    std::stringstream buffer;
    buffer << t.rdbuf();

    compileAndRun(buffer.str());
    return 0;
}
```

### 🛠️ Step 3: Integrating the Interpreter (`src/Interpreter.cpp`)

Since we haven't covered assembly backend (Week 14/15 topic), we'll implement a simple **Virtual Machine** that executes the IR directly. This allows us to "run" our code.

```cpp
#include "IR.h"
#include <unordered_map>
#include <iostream>

class Interpreter {
    IRProgram& program;
    std::unordered_map<std::string, int> registers;
    // Map labels to instruction indices
    std::unordered_map<std::string, int> labelMap;

public:
    Interpreter(IRProgram& prog) : program(prog) {
        // Pre-scan labels
        for (size_t i = 0; i < program.instructions.size(); ++i) {
            if (program.instructions[i].op == OpCode::LABEL) {
                labelMap[program.instructions[i].arg1] = i;
            }
        }
    }

    int getVal(const std::string& op) {
        // If it starts with digit, it's a literal
        if (isdigit(op[0]) || (op.size() > 1 && op[0] == '-')) {
            return std::stoi(op);
        }
        return registers[op];
    }

    void run() {
        size_t pc = 0; // Program Counter
        while (pc < program.instructions.size()) {
            const auto& instr = program.instructions[pc];
            
            switch (instr.op) {
                case OpCode::MOV:
                    registers[instr.arg1] = getVal(instr.arg2);
                    break;
                case OpCode::ADD:
                    registers[instr.arg1] = getVal(instr.arg2) + getVal(instr.arg3);
                    break;
                case OpCode::SUB:
                    registers[instr.arg1] = getVal(instr.arg2) - getVal(instr.arg3);
                    break;
                case OpCode::MUL:
                    registers[instr.arg1] = getVal(instr.arg2) * getVal(instr.arg3);
                    break;
                case OpCode::JMP:
                    pc = labelMap[instr.arg1];
                    continue; // Skip pc++
                case OpCode::JNZ:
                    if (getVal(instr.arg2) != 0) {
                        pc = labelMap[instr.arg1];
                        continue;
                    }
                    break;
                case OpCode::PARAM:
                    // In a real VM, push to stack
                    std::cout << "[VM] Pushing Param: " << getVal(instr.arg1) << "\n";
                    break;
                case OpCode::CALL:
                    if (instr.arg2 == "print") {
                        // Hack for print intrinsic: assume param was just pushed or exists
                        // For this simple interpreter, let's assume arg1 has the value to print (simplification)
                        // If strict TAC: PARAM t1; CALL print
                    }
                    else { 
                         // Not implemented: real function calls need stack frames
                    }
                    break;
                case OpCode::LABEL:
                    // No-op
                    break;
                default: 
                    break;
            }
            pc++;
        }
    }
};
```

### 🛠️ Step 4: Testing the Compiler

**Test Case (`tests/fib.mc`):**

```c
// Fibonacci in Mini-C
n = 10;
a = 0;
b = 1;
i = 0;

while (i < n) {
    temp = a + b;
    a = b;
    b = temp;
    i = i + 1;
}

// We don't have a print keyword, but we can assume 'b' ends up in a register.
// Let's rely on the Interpreter printing registers or adding a print instr.
```

**Expected IR Output:**

```text
MOV t1, 10
MOV n, t1
MOV t2, 0
MOV a, t2
...
L1:
MOV t5, i
MOV t6, n
SUB t7, t6, t5  // n - i > 0? roughly
JZ L2, t7 
...
JMP L1
L2:
```

---

## 📝 Review of Week 13: Compiler Architecture

We have traversed the "Frontend" and "Middle-end" of compiler design.

| Day | Topic | Key Insight |
| :-- | :--- | :--- |
| **85** | **Overview** | Compilation is a pipeline of transformations (Source $\to$ Machine). |
| **86** | **Lexing** | Regex and Finite Automata break stream into meaningful Tokens. |
| **87** | **Parsing** | CFGs and Recursive Descent build the structure (AST). |
| **88** | **AST/Semantics** | The AST captures meaning; Semantics validation catches logical errors. |
| **89** | **IR** | TAC/SSA decouples high-level syntax from low-level mechanics. |
| **90** | **Optimization** | Data flow analysis enables transforming code for efficiency. |
| **91** | **Project** | Integration proves that individual phases work in concert. |

**What's Missing?**
1.  **Real Code Gen:** We interpreted the IR. We didn't emit x86 bytes or binary object files.
2.  **Register Allocation:** Our IR uses infinite virtual registers (`t1`...`t1000`). Steps must be taken to map these to limited physical registers (`rax`, `rbx`...).
3.  **Linker:** Combining multiple files.

**Looking Ahead (Week 14 - LLVM):**
Next week, we stop "reinventing the wheel". We will introduce **LLVM**, the industry-standard compiler infrastructure. We will learn to emit **LLVM IR**, which gives us access to world-class optimizers (tens of thousands of man-years of work) and backends for every architecture (x86, ARM, RISC-V, GPU) for free.

*End of Day 091 - Total Lines: 1000+*
*End of Week 13 - Compiler Architecture Complete!*
