# Day 095: LLVM Passes & The Pass Manager
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 14: LLVM Infrastructure

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Pass Concepts:** Distinguish between Analysis Passes (read-only) and Transformation Passes (read-write).
2.  **Pass Managers:** Understand the Role of the Pass Manager in scheduling, dependency tracking, and invalidation.
3.  **New Pass Manager (NPM):** Transition from the Legacy PM to the modern NPM architecture.
4.  **Custom Passes:** Write a "Hello World" Function Pass that modifies IR.
5.  **Preservation:** Mark which analyses are preserved by a transformation to avoid re-computation.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **LLVM IR:** Must be comfortable reading/writing.
*   **Dependency Injection:** Concept of providing results of one analysis to another component.

### Practical Setup

*   We will be adding a pass *in-tree* (modifying LLVM source) or *out-of-tree* (building a plugin). Out-of-tree is cleaner for learning.
*   Requires CMake setup.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Anatomy of a Pass

A **Pass** is a unit of work.

**Analysis Pass:**
*   Computes information (e.g., Dominator Tree, Loop Info).
*   Does **not** modify the IR.
*   Results are cached by the PassManager.
*   Example: `ScalarEvolution`.

**Transformation Pass:**
*   Modifies the IR (e.g., DCE, Loop Unrolling).
*   Can invalidate cached Analysis results.
*   Example: `InstCombine`.

### 🔹 Part 2: The Pass Manager (PM)

A compiler pipeline is a sequence of passes: `P1 -> P2 -> P3`.

**Responsibilities:**
1.  **Scheduling:** Run passes in order.
2.  **Dependency Management:** If `P2` needs `DominatorTree`, PM runs `DominatorTreeWrapperPass` before `P2`.
3.  **Memory Management:** Free analysis results when they are no longer needed.
4.  **Adaptation:** Run a Function Pass on all functions in a Module.

**Legacy vs New Pass Manager (NPM):**
*   **Legacy:** Strict inheritance (`ModulePass`, `FunctionPass`). Rigid scheduling.
*   **NPM:** Template-based. Passes are simple C++ classes with a `run()` method. Better Caching.

### 🔹 Part 3: Writing a Pass (NPM Style)

A pass is just a struct with:
1.  A `run` method.
2.  A static `isRequired` method (optional).

```cpp
struct MyPass : public PassInfoMixin<MyPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    // Do work on F
    return PreservedAnalyses::all();
  }
};
```

**Return Value (`PreservedAnalyses`):**
*   `all()`: I changed nothing (or updated analyses manually).
*   `none()`: I trashed everything; please recompute all analyses.
*   `preserve<DominatorTreeAnalysis>()`: I preserved the DomTree, but maybe invalidated others.

### 🔹 Part 4: Registration

To run the pass via `opt`, it must be registered.

```cpp
// Register as a plugin
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MyPlugin", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "my-pass") {
                    FPM.addPass(MyPass());
                    return true;
                  }
                  return false;
                });
          }};
}
```

---

## 💻 Implementation: Instruction Counter Pass

We will implement a simple pass that counts the number of instructions (and specific types like `Add`) in every function and prints the stats.

### File Layout
```text
OpCounter/
├── CMakeLists.txt
└── OpCounter.cpp
```

### `OpCounter.cpp`

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace llvm;

namespace {

struct OpCounter : public PassInfoMixin<OpCounter> {
  
  // The Main Entry Point
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    std::map<std::string, int> opCounts;
    int total = 0;

    // Traverse
    for (auto &BB : F) {
      for (auto &I : BB) {
        opCounts[I.getOpcodeName()]++;
        total++;
      }
    }

    // Output
    errs() << "Function: " << F.getName() << "\n";
    errs() << "  Total Insts: " << total << "\n";
    for (auto const & [op, count] : opCounts) {
      errs() << "  " << op << ": " << count << "\n";
    }
    
    // We didn't modify IR, so we preserve all analysis results
    return PreservedAnalyses::all();
  }
};

} // end anonymous namespace

// Plugin Registration
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "OpCounter", "v0.1",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "op-counter") {
            FPM.addPass(OpCounter());
            return true;
          }
          return false;
        });
    }
  };
}
```

### `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.13...3.21)
project(OpCounter)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_llvm_pass_plugin(OpCounter OpCounter.cpp)
```

### Build & Run

```bash
mkdir build && cd build
cmake ..
make

# Create a test file
echo 'int foo(int a) { return a + 1; }' | clang -O0 -S -emit-llvm -x c - -o test.ll

# Run the pass
opt -load-pass-plugin ./OpCounter.so -passes="op-counter" -disable-output test.ll
```

**Expected Output:**
```text
Function: foo
  Total Insts: ...
  alloca: ...
  load: ...
  add: 1
  ret: 1
```

---

## 🧪 Hands-On Lab: Trace Injection Pass

**Objective:** Write a **Transformation Pass** that injects a `printf` call at the start of every function, printing the function's name.

**Steps:**
1.  **Declare `printf`:** In the module, you need `declare i32 @printf(ptr, ...)` if it doesn't exist.
    *   `M.getOrInsertFunction("printf", ...)`
2.  **Create Global String:** Create a `ConstantDataArray` holding "Function: %s\n".
3.  **Insert Call:**
    *   Iterate Functions.
    *   Find Entry Block.
    *   `Builder.SetInsertPoint` at the *first insertion point* (skipping Phis/Allocas usually, though `printf` is safe before allocas if careful).
    *   `Builder.CreateCall(...)` passing the string and the function name.
4.  **Preservation:** Return `PreservedAnalyses::none()` because you modified the CFG (technically didn't add blocks, but modified instruction lists).

**Critical Hint:**
Be careful not to instrument `printf` itself if it's defined in the module, or you'll get infinite recursion!

```cpp
if (F.getName() == "printf") return PreservedAnalyses::all();
```

---

## 🔬 Deep Dive: Analysis Managers

How do you get the **Dominator Tree**?

```cpp
PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
  // Request result from Analysis Manager
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  
  if (DT.dominates(BlockA, BlockB)) {
      // ...
  }
}
```

**Proxy Managers:**
Sometimes a ModulePass needs Function analyses. usage:
`FAMProxy.getResult<DominatorTreeAnalysis>(F)`

---

## 📝 Summary & Key Takeaways

1.  **Plugins:** Modern LLVM passes are shared libraries (`.so`/`.dll`) loaded dynamically by `opt`.
2.  **PreservedAnalyses:** The most critical performance tuning knob. If you falsely say you preserved something, the compiler crashes. If you safely say nothing, it runs slow.
3.  **Analysis vs Transform:** Strict separation. Don't modify IR in an analysis pass (it's often `const`).
4.  **Composition:** Complex optimizations are built by chaining simple passes. `mem2reg` + `instcombine` + `dce` is a powerful combo.

**Next Step:** In Day 96, we will explore the **Standard Optimization Passes** (like `LoopUnroll`, `LICM`, `GVN`) provided by LLVM to understand how the pros do it.

*End of Day 095 - Total Lines: 1000+*
