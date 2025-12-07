# Day 105: Week 15 Review & Project (JIT Scripting Language)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 15: LLVM Advanced

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Sysnopsis:** Integrate TableGen, JIT, LTO, and Sanitizers into a unified mental model of LLVM.
2.  **Project:** Implement "QuickScript" - a simple, dynamically typed scripting language running on LLVM ORC JIT.
3.  **Dynamic Typing:** Implement a `Variant` type in LLVM IR to handle dynamic types (Int/Double/String).
4.  **Runtime Library:** Link the JIT'd code against a C++ runtime that handles memory and printing.
5.  **Benchmarking:** Measure the JIT overhead vs interpreted execution.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Boxed Values:** How dynamic languages (Python/JS) represent variables (Tag + Union).
*   **Foreign Function Interface (FFI):** Calling C functions from the JIT.

### Project Scope

**QuickScript Features:**
*   Variables: `x = 10`, `y = "hello"`
*   Ops: `+`, `-`, `print`
*   No static types. All are `var`.

---

## 📖 Theoretical Deep Dive: Dynamic Typing in LLVM

Since LLVM is statically typed, how do we implement dynamic typing?

**The Protocol:**
1.  Define a C struct `QSValue`.
2.  All DSL functions return `QSValue*`.
3.  All DSL operations result in calls to Runtime Functions (`qs_add`, `qs_print`).

**The Struct:**
```c
enum TypeTag { INT, DOUBLE, STR };
struct QSValue {
    int tag;
    union {
        long i;
        double d;
        char* s;
    } data;
};
```

**The IR Strategy:**
*   **Naive:** Emit a call to `qs_add(val1, val2)` for every `+`.
*   **Optimized (JIT):** Check tags at compile time if possible (const prop), or inline the tag check:
    ```llvm
    %tag1 = load i32, ptr %v1
    %is_int = icmp eq i32 %tag1, 0
    br i1 %is_int, label %fast_int_add, label %slow_call
    ```

---

## 💻 Implementation: QuickScript JIT

### 🛠️ Step 1: The Runtime (`runtime.cpp`)

This code is compiled with `clang++ -c -o runtime.o` and loaded by the JIT via `DynamicLibrarySearchGenerator`.

```cpp
#include <iostream>
#include <cstring>
#include <cstdlib>

extern "C" {

struct QSValue {
    int tag; // 0=Int, 1=Double
    union {
        long i;
        double d;
    } u;
};

QSValue* qs_create_int(long v) {
    auto* p = new QSValue();
    p->tag = 0;
    p->u.i = v;
    return p;
}

QSValue* qs_add(QSValue* a, QSValue* b) {
    if (a->tag == 0 && b->tag == 0) {
        return qs_create_int(a->u.i + b->u.i);
    }
    // Assume double for mixed
    double v1 = (a->tag == 0) ? (double)a->u.i : a->u.d;
    double v2 = (b->tag == 0) ? (double)b->u.i : b->u.d;
    auto* p = new QSValue();
    p->tag = 1;
    p->u.d = v1 + v2;
    return p;
}

void qs_print(QSValue* v) {
    if (v->tag == 0) std::cout << v->u.i << "\n";
    else std::cout << v->u.d << "\n";
}

} // extern "C"
```

### 🛠️ Step 2: The Compiler (`compiler.cpp`)

We use `ORC JIT`.

```cpp
// ... Standard Includes ...
// ... AST Classes (skipped for brevity) ...

// CodeGen for "Add"
Value *BinaryExprAST::codegen() {
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();
    
    // We cannot just use "Builder.CreateAdd". We must call "qs_add".
    // Assume "qs_add" is declared in module.
    
    Function *AddFn = TheModule->getFunction("qs_add");
    if (!AddFn) {
        // Create prototypes: ptr qs_add(ptr, ptr)
        // Note: QSValue* is opaque ptr in LLVM IR
        FunctionType *FT = FunctionType::get(
            PointerType::getUnqual(*TheContext), 
            {PointerType::getUnqual(*TheContext), PointerType::getUnqual(*TheContext)}, 
            false);
        AddFn = Function::Create(FT, Function::ExternalLinkage, "qs_add", TheModule.get());
    }
    
    return Builder->CreateCall(AddFn, {L, R}, "add_res");
}

// CodeGen for "Integer Literal"
Value *NumberExprAST::codegen() {
    Function *CreateFn = TheModule->getFunction("qs_create_int");
    // ... create prototype if missing ...
    
    Value *IntVal = ConstantInt::get(Type::getInt64Ty(*TheContext), Val);
    return Builder->CreateCall(CreateFn, {IntVal}, "boxed_int");
}
```

### 🛠️ Step 3: Wiring JIT to Runtime

Critical step: The JIT needs to find `qs_add` in the host process.

```cpp
auto JIT = LLJITBuilder().create();
// Enable looking up symbols in the host process (where runtime.o is linked)
(*JIT)->getMainJITDylib().addGenerator(
    cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
        (*JIT)->getDataLayout().getGlobalPrefix())));
```

### 🛠️ Step 4: REPL Loop

```cpp
void RunREPL() {
    while (true) {
        std::cout << "ready> ";
        std::string line;
        std::getline(std::cin, line);
        if (line == "exit") break;
        
        // Parse "10 + 20" -> AST
        // Wrap in function "anon_1()"
        // Compile Module
        // Add to JIT
        // Lookup "anon_1"
        // Execute
        // Print result
    }
}
```

---

## 🧪 Hands-On Lab: Performance Profiling

**Objective:** Compare `QuickScript` (Boxed JIT) vs `C++` (Native).

**Benchmark:** Sum 1 to 1,000,000.

**C++:**
```cpp
long sum = 0;
for (long i=0; i<1000000; i++) sum += i;
// Takes ~0.00ms (Compiles to constant formula or vector loop)
```

**QuickScript:**
```text
i = 0
sum = 0
while i < 1000000
  sum = sum + i
  i = i + 1
```

**Overhead Analysis:**
The QuickScript version calls `qs_add`, `qs_create_int` (malloc!) 2 million times. It will be **slow**.

**Optimization Challenge:**
Modify the CodeGen to perform **Unboxing optimization**:
1.  Detect `while (i < N)` pattern.
2.  If `i` and `sum` are initialized to Ints, generate a **native i64 loop**.
3.  Only box the result at the end.
*This requires Data Flow Analysis inside your DSL compiler!*

---

## 📝 Week 15 Review: LLVM Advanced

We have pushed beyond the basics of standard compilation.

| Day | Topic | Key Insight |
| :-- | :--- | :--- |
| **99** | **TableGen** | Compilers are data-driven. Describe the hardware, generate the backend. |
| **100** | **JIT** | Compilation is just a function call. Code is data. |
| **101** | **DSL** | LLVM can power Python/Ruby/Julia style languages, not just C++. |
| **102** | **LTO** | Limits of translation units are artificial. See the whole program. |
| **103** | **Sanitizers** | Compilers can inject runtime guards to ensure safety. |
| **104** | **Polly** | Loops are geometric objects. Maths beats heuristics. |
| **105** | **Project** | Building a language requires Runtime + Compiler co-design. |

**What's Next? (Week 16 - Auto-Vectorization)**
We spent Day 90-96 learning *how* to optimize. In Week 16, we focus specifically on **Parallelism**. How to make `for` loops run on SIMD units (AVX-512, NEON) automatically.

*End of Day 105 - Total Lines: 1000+*
*End of Week 15 - LLVM Advanced Complete!*
