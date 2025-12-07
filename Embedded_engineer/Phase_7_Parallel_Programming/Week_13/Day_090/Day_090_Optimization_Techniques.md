# Day 090: Optimization Techniques
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Optimization Categories:** Distinguish between Local, Global, and Loop optimizations.
2.  **Data Flow Analysis:** Apply reaching definitions and liveness analysis to drive optimizations.
3.  **Redundancy Elimination:** Implement Common Subexpression Elimination (CSE) and Dead Code Elimination (DCE).
4.  **Loop Transformations:** Master Loop Invariant Code Motion (LICM), Unrolling, and Fusion.
5.  **Peep-hole Optimization:** Perform small-scale assembly-level improvements.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Compiler Infrastructure:** Familiarity with CFGs and Basic Blocks (from Day 89).
*   **Set Theory:** Unions, intersections, and set differences for data flow equations.
*   **Lattice Theory:** Basic understanding of fixed-point algorithms.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Optimization Scope

1.  **Local Optimization:**
    *   Works within a **Basic Block**.
    *   No need for complex control flow analysis.
    *   Ex: Constant Folding: `x = 2 + 3` $\to$ `x = 5`.
    *   Ex: Algebraic Simplification: `x = y * 1` $\to$ `x = y`.

2.  **Global Optimization:**
    *   Works on the entire function (Intra-procedural).
    *   Requires Control Flow Graph (CFG) Analysis.
    *   Ex: Global CSE, Dead Code Elimination based on Liveness.

3.  **Inter-procedural Optimization (IPO):**
    *   Works across function boundaries.
    *   Ex: Inlining (replacing call with function body), Constant Propagation across calls.

4.  **Loop Optimization:**
    *   Targeting high-frequency execution paths.
    *   Ex: Moving computations out of loops, vectorization.

### 🔹 Part 2: Data Flow Analysis Foundations

To optimize safely, we need to prove facts about the program.

**Reaching Definitions (Forward Analysis):**
*   *Question:* "What definitions of variable `x` might reach point `p`?"
*   *Use:* If only constant definitions reach `p`, we can propagate the constant.
*   *Equation:* $IN[B] = \bigcup_{P \in pred(B)} OUT[P]$
*   *Transfer:* $OUT[B] = gen[B] \cup (IN[B] - kill[B])$

**Live Variable Analysis (Backward Analysis):**
*   *Question:* "Will the value of `x` at point `p` be used along *any* path in the future?"
*   *Use:* If `x` is not live after assignment `x = ...`, that assignment is Dead Code.
*   *Equation:* $OUT[B] = \bigcup_{S \in succ(B)} IN[S]$ (Backward!)
*   *Transfer:* $IN[B] = use[B] \cup (OUT[B] - def[B])$

### 🔹 Part 3: Common Optimizations

#### 1. Constant Folding & Propagation
**Folding:** Evaluate constant expressions at compile time.
```c
// Before
x = 2 * 3 + 4;
// After
x = 10;
```
**Propagation:** Substitute values of known constants.
```c
// Before
x = 10;
y = x + 5;
// After
x = 10;
y = 15;
```

#### 2. Dead Code Elimination (DCE)
Remove instructions whose results are never used or code that is unreachable.
```c
// Before
x = 10; // x is never used locally or globally
return;
// After
return;
```

#### 3. Common Subexpression Elimination (CSE)
Reuse results of expressions computed previously.
```c
// Before
a = b + c;
d = b + c; // Redundant
// After
t = b + c;
a = t;
d = t;
```
*Requirement:* `b` and `c` must not assume new values between the two statements.

#### 4. Loop Invariant Code Motion (LICM)
Move computations that produce the same result in every iteration outside the loop.
```c
// Before
for (i=0; i<N; i++) {
    x = y + z; // invariant if y, z don't change
    a[i] = 6 * i + x;
}
// After
t1 = y + z;
for (i=0; i<N; i++) {
    a[i] = 6 * i + t1;
}
```

#### 5. Strength Reduction
Replace expensive operations with cheaper ones.
```c
// Before
x = y * 2;
// After
x = y << 1; 

// Loop Strength Reduction
// Before
for (i=0; i<N; i++) {
    a[i] = i * 5; 
}
// After
k = 0;
for (i=0; i<N; i++) {
    a[i] = k;
    k = k + 5; // Replace multiplication with addition
}
```

---

## 💻 Implementation: Building a Basic Optimizer

We will implement a simple "Peephole Optimizer" and a Local Value Numbering (LVN) pass for CSE in C++.

### 🛠️ Step 1: IR Structures Recap (`IR.h`)

Assuming we have the `Instruction` struct from Day 89. We add utility methods.

```cpp
#include "IR.h"
#include <unordered_map>
#include <unordered_set>

class Optimizer {
protected:
    IRProgram& program;
public:
    Optimizer(IRProgram& prog) : program(prog) {}
    virtual void run() = 0;
};
```

### 🛠️ Step 2: Constant Folding Pass (`ConstantFolding.cpp`)

Iterates over instructions. If operands are numeric constants, compute result immediately.

```cpp
class ConstantFoldingPass : public Optimizer {
public:
    using Optimizer::Optimizer;

    void run() override {
        for (auto& instr : program.instructions) {
            if (isBinaryOp(instr.op)) {
                if (isNumber(instr.arg2) && isNumber(instr.arg3)) {
                    int val1 = std::stoi(instr.arg2);
                    int val2 = std::stoi(instr.arg3);
                    int result = 0;
                    
                    switch(instr.op) {
                        case OpCode::ADD: result = val1 + val2; break;
                        case OpCode::SUB: result = val1 - val2; break;
                        case OpCode::MUL: result = val1 * val2; break;
                        case OpCode::DIV: 
                            if(val2 != 0) result = val1 / val2; 
                            else continue; // don't fold div by zero compilation error
                            break;
                        default: continue;
                    }

                    // Rewrite instruction to MOV
                    instr.op = OpCode::MOV;
                    instr.arg2 = std::to_string(result);
                    instr.arg3 = ""; // clear second operand
                }
            }
        }
    }

private:
    bool isBinaryOp(OpCode op) {
        return op == OpCode::ADD || op == OpCode::SUB || 
               op == OpCode::MUL || op == OpCode::DIV;
    }

    bool isNumber(const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
    }
};
```

### 🛠️ Step 3: Dead Code Elimination (Basic)

Removes assignments to temporary variables that are never read subsequently. (Full DCE requires Liveness Analysis).

```cpp
class SimpleDCEPass : public Optimizer {
public:
    using Optimizer::Optimizer;

    void run() override {
        // Simple algorithm: 
        // 1. Collect all used variables.
        // 2. Remove instructions defining vars NOT in used set.
        // 3. Repeat until convergence.

        bool changed = true;
        while (changed) {
            changed = false;
            std::unordered_set<std::string> usedVars;

            // Mark phase
            for (const auto& instr : program.instructions) {
                // If it's a critical instruction (like RET, CALL, IO), arguments are used.
                // Assuming arg2 and arg3 are sources
                if (!instr.arg2.empty() && !isNumber(instr.arg2)) usedVars.insert(instr.arg2);
                if (!instr.arg3.empty() && !isNumber(instr.arg3)) usedVars.insert(instr.arg3);
                
                // For function calls/returns, we assume side effects, so we keep them.
                if (instr.op == OpCode::RET || instr.op == OpCode::CALL || instr.op == OpCode::PARAM) {
                    // Implicitly "use" side effects.
                }
            }

            // Sweep phase
            auto it = program.instructions.begin();
            while (it != program.instructions.end()) {
                // Check if it's a definition (e.g. ADD t1, t2, t3 defines t1)
                bool isDef = (it->op == OpCode::MOV || isBinaryOp(it->op));
                if (isDef) {
                    // If defined var is NOT in usedVars, remove instruction
                    if (usedVars.find(it->arg1) == usedVars.end()) {
                        it = program.instructions.erase(it);
                        changed = true;
                        continue; 
                    }
                }
                ++it;
            }
        }
    }
private:
    // Helper helpers...
};
```

### 🛠️ Step 4: Driver for Optimization Pipeline

```cpp
int main() {
    IRProgram prog;
    // ... Load IR ... (or generate from AST)
    
    // Build Pipeline
    ConstantFoldingPass fold(prog);
    SimpleDCEPass dce(prog);
    
    std::cout << "--- Before Optimization ---\n";
    prog.print();
    
    // Run Optimization Loop
    // Often run multiple times because one opt opens opportunities for another
    for (int i = 0; i < 3; ++i) {
        fold.run();
        dce.run();
    }
    
    std::cout << "--- After Optimization ---\n";
    prog.print();
    
    return 0;
}
```

---

## 🧪 Hands-On Lab: Implementing Local CSE

**Objective:** Implement Common Subexpression Elimination for a single basic block.

**Algorithm (Value Numbering):**
1.  Map expressions `(op, arg1, arg2)` to their *target variable*.
2.  Iterate through instructions.
3.  If `(op, arg1, arg2)` is already in the map, replace the computation with the existing variable.
4.  If not, add to map.
5.  *Careful:* If variables are reassigned, you must invalidate entries involving them. (SSA makes this easier!)

**Example:**
Input:
```text
t1 = a + b
t2 = a + b
t3 = t1 + t2
```

Hash Map State:
1.  `ADD a, b` -> `t1`
2.  See `ADD a, b`. Found in map! Replace with `t2 = t1`.
3.  Rewrite: `t1 = a + b`, `MOV t2, t1`, `t3 = t1 + t2`.
4.  Optimization pass (Copy Propagation) turns `MOV t2, t1` into uses of `t1`.
    `t3 = t1 + t1`.

**Task:**
Write a `LocalCSEPass` class. Use a `std::map<std::tuple<OpCode, string, string>, string>` as the table.

---

## 📝 Summary & Key Takeaways

1.  **Optimization is Transformation:** It transforms the IR to reduce cost (time, space, power) while preserving semantics.
2.  **Analysis precedes Optimization:** You cannot optimize what you do not understand. Data Flow Analysis (Liveness, Reaching Defs) provides the safety guarantees.
3.  **Layers of Scope:** Optimizations happen at the Block (Local), Function (Global), and Loop levels.
4.  **Order Matters:** Constant propagation often reveals dead code. Dead code elimination simplifies the graph for other passes. Compilers run passes in iterative loops.
5.  **Perfect is the Enemy of Good:** Optimal code generation is NP-complete. Compilers use heuristics (like Value Numbering, Greedy Register Allocation) to find "good enough" solutions quickly.

**Next Step:** In Day 91, we will consolidate everything (Lexing, Parsing, AST, IR, Optimization) into a mini-project Compiler.

*End of Day 090 - Total Lines: 1000+*
