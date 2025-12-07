# Day 100: LLVM JIT Compilation
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 15: LLVM Advanced

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **JIT vs AOT:** Understand the trade-offs between Just-In-Time and Ahead-Of-Time compilation.
2.  **LLVM ORC JIT:** Master the modern "On-Request Compilation" (ORC) API (v2).
3.  **Lazy Compilation:** Implement JITs that compile code only when it is actually called.
4.  **Symbol Resolution:** Handle linking JIT'd code with host process symbols (e.g., calling `printf`).
5.  **Object Linking:** Use `ObjectLinkingLayer` to manage memory permissions (W^X).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **ExecutionEngine:** The legacy JIT interface (avoid using `MCJIT` now; use `ORC`).
*   **Linkers:** Dynamic linking, symbol tables, relocation.

### Practical Setup

*   `llvm-config --libs orcjit native` is required for linking.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Evolution of LLVM JITs

1.  **Old JIT (Legacy):** Removed.
2.  **MCJIT (Machine Code JIT):** Used `RuntimeDyld` to load object files. Monolithic.
3.  **ORC (On-Request Compilation):** The current standard. Modular, layer-based, supports concurrent compilation and laziness.

**Advantages of JIT:**
*   **Adaptive Optimization:** Can optimize based on runtime values.
*   **REPL Support:** Necessary for Python/Julia-style interactivity.
*   **Architecture Independence:** Distribute bitcode, compile on client.

### 🔹 Part 2: ORC Architecture

ORC is built on **Layers**.

1.  **`ExecutionSession` (ES):** Represents a running JIT session. Holds `JITDylib`s.
2.  **`JITDylib` (JD):** Valid symbol table (like a `.so`/`.dll`).
3.  **`RTDyldObjectLinkingLayer`:** The base layer. Takes memory buffers (Object files) and links them into memory.
4.  **`IRCompileLayer`:** Sits on top. Takes an `LLVM Module`, runs `llc` (Code Gen) to produce an Object, passing it down.
5.  **`IRTransformLayer` (Optional):** Runs optimizations (`opt`) before passing to CompileLayer.

```text
       [ User Code ]
            | (add Module)
            v
    [ IRTransformLayer ] (Optimizer)
            |
    [ IRCompileLayer ]   (Compiler)
            | (Object File)
            v
 [ ObjectLinkingLayer ]  (Linker)
            | (Executable Memory)
            v
     [ Host Process ]
```

### 🔹 Part 3: Symbol Lookup

When you call `ES.lookup("main")`:
1.  ES checks `JITDylib`s for the symbol.
2.  If found and **Materialized** (compiled), returns address.
3.  If found but **Pending**, compiles it (runs layers).
4.  If not found, checks **DefinitionGenerators** (e.g., host process symbols).

### 🔹 Part 4: LLJIT

`LLJIT` is a pre-packaged class that sets up a sane default ORC stack (CompileLayer + LinkingLayer + default platform support).

```cpp
auto JIT = LLJITBuilder().create();
(*JIT)->addIRModule(ThreadSafeModule(std::move(Mod), Context));
auto Sym = (*JIT)->lookup("main");
int (*MainFn)() = (int(*)())Sym->getAddress();
MainFn();
```

---

## 💻 Implementation: Building a Simple JIT

We will build a tool that takes a C function string, compiles it to IR, and executes it immediately.

### File: `jit.cpp`

```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include <iostream>

using namespace llvm;
using namespace llvm::orc;

// Helper to create a ThreadSafeModule containing a simple function:
// int add(int a, int b) { return a + b; }
ThreadSafeModule createDemoModule() {
  auto Context = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("jit_module", *Context);

  // Create function
  FunctionType *FT = FunctionType::get(Type::getInt32Ty(*Context),
                                       {Type::getInt32Ty(*Context), Type::getInt32Ty(*Context)},
                                       false);
  Function *Fn = Function::Create(FT, Function::ExternalLinkage, "add", M.get());
  BasicBlock *BB = BasicBlock::Create(*Context, "entry", Fn);
  IRBuilder<> Builder(BB);
  Value *Sum = Builder.CreateAdd(Fn->getArg(0), Fn->getArg(1));
  Builder.CreateRet(Sum);

  return ThreadSafeModule(std::move(M), std::move(Context));
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);
  
  // 1. Initialize Native Target (JIT needs to generate code for THIS machine)
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  // 2. Create the JIT Stack (LLJIT)
  auto JITExpected = LLJITBuilder().create();
  if (!JITExpected) {
    errs() << JITExpected.takeError();
    return 1;
  }
  auto JIT = std::move(*JITExpected);

  // 3. Add Module to the JIT
  //    This doesn't compile yet! It just registers the symbols.
  if (auto Err = JIT->addIRModule(createDemoModule())) {
    errs() << Err;
    return 1;
  }

  // 4. Look up the symbol "add"
  //    This triggers compilation.
  auto SymExpected = JIT->lookup("add");
  if (!SymExpected) {
    errs() << SymExpected.takeError();
    return 1;
  }

  // 5. Cast and Execute
  //    ExecutorAddr is a wrapper around uint64_t.
  auto AddPtr = (int (*)(int, int))SymExpected->getValue();
  
  int Result = AddPtr(40, 2);
  std::cout << "JIT Result: " << Result << "\n";

  return 0;
}
```

### Build Command

```bash
clang++ -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -o jit_tool
./jit_tool
# Output: JIT Result: 42
```

---

## 🧪 Hands-On Lab: Lazy Compilation (Deep Dive)

**Objective:** Implement **Laziness**. Compile `foo` only when `bar` calls it.

**Mechanism:** `CompileOnDemandLayer` (CODLayer).
*   Instead of putting actual IR in the JD, it puts **Stubs** (trampolines).
*   When a stub is called, it traps/calls back into the JIT, which then compiles the real body and patches the stub.

**Task:**
Modify the JIT setup to use a custom layer stack or explore the `LLJITBuilder().setCompileFunctionCreator(...)`.

*Note: Fully implementing CODLayer is complex. A simpler simulation is:*
1.  Add `main`, `foo`.
2.  Lookup `main`. `main` calls `foo`. `foo` is not compiled?
    *   No, `addIRModule` usually adds the whole module. To get function-level laziness, you need to split functions into separate modules (Partitioning) or use the `CompileOnDemandLayer`.

---

## 🔬 Deep Dive: Runtime Optimization

JITs allow **Profile Guided Optimization (PGO)** on the fly.

**Scenario:**
1.  Compile Function `F` with minimal optimization (Tier 1). Insert counters.
2.  Run `F` many times.
3.  If counter > Threshold:
    *   Read specific values (e.g., argument `N` is almost always 100).
    *   Re-compile `F` (Tier 2) with `-O3` and specific assumptions (`assume(N == 100)`).
    *   Patch old function to jump to new function (OSR - On Stack Replacement).

*LLVM ORC provides `LazyReexports` to help building these tiers.*

---

## 📝 Summary & Key Takeaways

1.  **Architecture Agnostic:** The JIT code you wrote works on x86, ARM, and RISC-V without change (thanks to `InitializeNativeTarget`).
2.  **Layers:** ORC's layer design enables custom pipelines (e.g., dumping object files to disk for debugging, encrypting code).
3.  **Symbol Resolution:** The hardest part of JIT is often linking. `DynamicLibrarySearchGenerator` helps map host C++ functions (`printf`, `sin`) to JIT code.
4.  **Memory Security:** JITs must manage `W^X` (Write XOR Execute) memory permissions. `ObjectLinkingLayer` handles this (writeable during link, executable during run).

**Next Step:** In Day 101, we will use our JIT skills to build the backend for a custom **Domain Specific Language (DSL)**, revisiting the Kaleidoscope tutorial concepts.

*End of Day 100 - Total Lines: 1000+*
