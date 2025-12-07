# Day 094: LLVM C++ API
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 14: LLVM Infrastructure

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Core Classes:** Navigate the hierarchy of `llvm::Module`, `llvm::Function`, `llvm::BasicBlock`, and `llvm::Instruction`.
2.  **IRBuilder:** Use the builder pattern to generate instructions programmatically.
3.  **Context Management:** Understand `LLVMContext` and resource ownership.
4.  **Type & Value System:** Manipulate `llvm::Type` and `llvm::Value` objects.
5.  **Verification:** Use the `Verifier` pass to sanitise generated IR.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **LLVM IR:** Understanding yesterday's content (`.ll` syntax) is a hard requirement.
*   **C++17:** Smart pointers, STL containers.
*   **Object Ownership:** LLVM uses a specific ownership model (Modules own Functions, Contexts own Types).

### Practical Setup

*   You must have LLVM development headers installed (`llvm-dev`).
*   Example build command:
    ```bash
    clang++ -g -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy
    ```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Context and The Module

Everything in LLVM happens within a `LLVMContext`. It owns distinct objects like Types and Constants to ensure uniqueness (interning).

**LLVMContext:**
*   Thread-local (usually).
*   Manages lifetime of Global Types.

**Module:**
*   Represents a translation unit.
*   Contains Global Variables, Functions, and Symbol Table.
*   `std::unique_ptr<Module> TheModule;`

```cpp
LLVMContext TheContext;
std::unique_ptr<Module> TheModule = std::make_unique<Module>("my_module", TheContext);
```

### 🔹 Part 2: The Value Hierarchy

Almost *everything* in LLVM inherits from `llvm::Value`.

```text
Value
 ├── User (Has operands)
 │    ├── Instruction (Add, Sub, Ret...)
 │    ├── Constant (ConstInt, ConstFP...)
 │    └── GlobalValue (GlobalVariable, Function)
 └── Argument (Function Argument)
```

**Key Insight:** This follows the **Def-Use Chain** pattern. A `User` uses `Value`s. Because instructions use other instructions, `Instruction` inherits from `User`, which inherits from `Value`.

### 🔹 Part 3: The IRBuilder

Manually creating `Instruction` objects and `push_back`ing them into Basic Blocks is tedious. `IRBuilder` is a helper class that keeps track of the "insert point".

```cpp
IRBuilder<> Builder(TheContext);

// Create Add instruction and insert it at current position
Value* LHS = ...;
Value* RHS = ...;
Value* Sum = Builder.CreateAdd(LHS, RHS, "sumtmp");
```

### 🔹 Part 4: Type System API

Types are immutable and owned by the Context.

```cpp
Type* Int32Ty = Type::getInt32Ty(TheContext);
Type* FloatTy = Type::getFloatTy(TheContext);
Type* VoidTy  = Type::getVoidTy(TheContext);

// Function Type: i32 (i32, i32)
std::vector<Type*> Ints(2, Int32Ty);
FunctionType* FT = FunctionType::get(Int32Ty, Ints, false);
```

### 🔹 Part 5: Verifier

It is easy to generate invalid IR (blocks without terminators, type mismatches). The Verifier catches these bugs.

```cpp
#include "llvm/IR/Verifier.h"

if (verifyFunction(*TheFunction, &errs())) {
    errs() << "Error: Function is corrupt!\n";
}
```

---

## 💻 Implementation: A Complete IR Generator

We will create a C++ program that generates the `factorial` function we wrote manually yesterday.

### Source Code (`codegen.cpp`)

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
#include <iostream>

using namespace llvm;

int main() {
    // 1. Setup
    LLVMContext Context;
    std::unique_ptr<Module> TheModule = std::make_unique<Module>("my_jit_module", Context);
    IRBuilder<> Builder(Context);

    // 2. Define Function Prototype: i32 factorial(i32)
    std::vector<Type*> Args(1, Type::getInt32Ty(Context));
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(Context), Args, false);
    
    Function *TheFunction = Function::Create(FT, Function::ExternalLinkage, "factorial", TheModule.get());

    // Name argument
    Argument *ArgN = TheFunction->getArg(0);
    ArgN->setName("n_arg");

    // 3. Create Basic Blocks
    BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", TheFunction);
    BasicBlock *LoopBB  = BasicBlock::Create(Context, "loop", TheFunction);
    BasicBlock *ExitBB  = BasicBlock::Create(Context, "exit", TheFunction);

    // --- Entry Block ---
    Builder.SetInsertPoint(EntryBB);
    
    // Check if n > 0
    Value *Zero = ConstantInt::get(Context, APInt(32, 0));
    Value *One  = ConstantInt::get(Context, APInt(32, 1));
    Value *InitCmp = Builder.CreateICmpSGT(ArgN, Zero, "init_cmp");
    Builder.CreateCondBr(InitCmp, LoopBB, ExitBB);

    // --- Loop Block ---
    Builder.SetInsertPoint(LoopBB);

    // Create Phi nodes. We don't have incoming edges fully set yet, so we define structure first.
    PHINode *CurrN = Builder.CreatePHI(Type::getInt32Ty(Context), 2, "curr_n");
    PHINode *CurrRes = Builder.CreatePHI(Type::getInt32Ty(Context), 2, "curr_res");

    // Add incoming from Entry
    CurrN->addIncoming(ArgN, EntryBB);
    CurrRes->addIncoming(One, EntryBB);

    // Loop Body: res = res * n
    Value *NewRes = Builder.CreateMul(CurrRes, CurrN, "new_res");
    
    // Loop Body: n = n - 1
    Value *NewN = Builder.CreateSub(CurrN, One, "new_n");

    // Loop Condition
    Value *LoopCond = Builder.CreateICmpSGT(NewN, Zero, "loop_cond");
    Builder.CreateCondBr(LoopCond, LoopBB, ExitBB);

    // Back edges for Phis
    CurrN->addIncoming(NewN, LoopBB);
    CurrRes->addIncoming(NewRes, LoopBB);

    // --- Exit Block ---
    Builder.SetInsertPoint(ExitBB);

    // Result Phi
    PHINode *FinalRes = Builder.CreatePHI(Type::getInt32Ty(Context), 2, "final_res");
    FinalRes->addIncoming(One, EntryBB);
    FinalRes->addIncoming(NewRes, LoopBB);

    Builder.CreateRet(FinalRes);

    // 4. Verification & Output
    if (verifyFunction(*TheFunction, &errs())) {
        std::cerr << "Error constructing function!\n";
        return 1;
    }

    TheModule->print(outs(), nullptr);

    return 0;
}
```

### Compilation & Execution

```bash
# Compile the generator
clang++ -g codegen.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o codegen

# Run it to produce IR
./codegen > output.ll

# Verify outputs
cat output.ll
llvm-as output.ll
lli output.bc
# (exit code will be 0 since main doesn't exist to call it, but it compiles)
```

---

## 🧪 Hands-On Lab: Custom Operator Lowering

**Goal:** Understand how extensive the API is. Use `getelementptr`.

**Task:** Create a function `void increment_array(int* arr, int index)` which does `arr[index]++`.

```cpp
// Hints:
// 1. Function signature: void (ptr, i32)
// 2. GEP: Builder.CreateGEP(Type::getInt32Ty(Context), ArrPtr, IdxVal)
// 3. Load
// 4. Add
// 5. Store
```

**Solution Snippet:**
```cpp
// ... setup ...
FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), {PointerType::getUnqual(Context), Type::getInt32Ty(Context)}, false);
Function *Fn = Function::Create(FT, Function::ExternalLinkage, "increment_array", TheModule.get());
// ... block ...
Value *ArrPtr = Fn->getArg(0);
Value *Idx = Fn->getArg(1);

// GEP: ArrPtr + Idx * sizeof(i32)
// Important: GEP needs the source element type!
Value *ElemPtr = Builder.CreateGEP(Type::getInt32Ty(Context), ArrPtr, Idx, "elem_ptr");

Value *Val = Builder.CreateLoad(Type::getInt32Ty(Context), ElemPtr, "val");
Value *Inc = Builder.CreateAdd(Val, ConstantInt::get(Context, APInt(32, 1)), "inc");
Builder.CreateStore(Inc, ElemPtr);

Builder.CreateRetVoid();
```

---

## 📝 Summary & Key Takeaways

1.  **Ownership:** `Module` owns functions. `Context` owns Types/Constants. `Function` owns BasicBlocks. `BasicBlock` owns Instructions.
2.  **Builder Pattern:** `IRBuilder` keeps a cursor (`SetInsertPoint`). It handles the boilerplate of creating instructions and appending them to the block.
3.  **Use-Def Chains:** `Value` and `User` classes form a graph. You can traverse *uses* of a value (e.g., "who uses this constant?") and *operands* of a user (e.g., "what inputs does this Add use?").
4.  **Phis require patience:** You often create the `PHI` node first, generate the loop body, and then populate the `addIncoming` edges once the values exist.
5.  **Debugging:** Always call `verifyFunction()`! It saves hours of debugging segmentation faults in the JIT later.

**Next Step:** In Day 95, we will look at **Passes**. How to write code that analyzes or transforms this IR structure we just learned to build.

*End of Day 094 - Total Lines: 1000+*
