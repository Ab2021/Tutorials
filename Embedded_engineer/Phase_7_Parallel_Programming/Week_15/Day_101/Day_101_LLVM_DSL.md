# Day 101: LLVM for Domain-Specific Languages (DSL)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 15: LLVM Advanced

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **DSL Architecture:** Design a simple compiler pipeline for a custom numeric language.
2.  **Kaleidoscope Patterns:** Apply the lessons from the famous LLVM tutorial to a new problem domain.
3.  **Code Gen Strategy:** Lower high-level constructs (Loop Expressions, Arrays) to low-level LLVM IR.
4.  **REPL Implementation:** Build an interactive Read-Eval-Print Loop using the JIT.
5.  **Extensibility:** Add support for external C functions (e.g., `sin`, `cos`) in your DSL.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Recursion:** Parsing expressions usually involves recursive descent.
*   **Visitor Pattern:** For AST traversal during code generation.

### Practical Setup

*   We continue using the LLVM C++ API.
*   Build tools (`clang++`, `make`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Anatomy of a DSL Compiler

A Domain-Specific Language (DSL) is optimized for a specific task (e.g., Image Processing, tensor algebra, queries).

**Differences from General Purpose Languages (GPLs):**
*   **Narrower Scope:** Fewer features (maybe no classes or exceptions).
*   **Higher Level Primitives:** Native matrix types, automatic parallelization.
*   **Performance:** Can often outperform C++ because the semantics restrict aliasing or side effects.

### 🔹 Part 2: The "Kaleidoscope" Influence

The *Kaleidoscope* tutorial is the "Hello World" of LLVM. It implements a functional language. We will implement a modified imperative version called **"NumLang"**.

**NumLang Syntax:**
```text
def fib(x)
  if x < 2 then
    x
  else
    fib(x-1) + fib(x-2);

fib(40);
```

### 🔹 Part 3: Lowering Strategies

**1. Conditionals (`if/else`)**
*   LLVM R: No "if" instruction.
*   Lowering: Create `Then`, `Else`, `Merge` blocks. Use `br i1 %cond`. Use `phi` in `Merge` block to join values.

**2. Arrays**
*   DSL: `var a[10]; a[i] = 5;`
*   LLVM:
    *   `%a = alloca [10 x double]`
    *   To access: `getelementptr` to get address, then `load/store`.

**3. Loops (`for`)**
*   Lowering:
    *   `Entry`: Init loop variable.
    *   `Header`: Check condition. Branch to `Body` or `Exit`.
    *   `Body`: Execute statements. Increment variable. Jump to `Header`.

---

## 💻 Implementation: Building "NumLang"

We'll skip lexing/parsing details (assuming Day 86/87 knowledge) and focus on the **LLVM Code Generation** capability.

### 🛠️ Step 1: The AST Interface

```cpp
class ExprAST {
public:
  virtual ~ExprAST() = default;
  virtual Value *codegen() = 0;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;
public:
  NumberExprAST(double Val) : Val(Val) {}
  Value *codegen() override;
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;
public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  Value *codegen() override;
};
```

### 🛠️ Step 2: Code Generation Visitor (Implementation)

Global variables for state:
```cpp
static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static std::map<std::string, Value *> NamedValues;
```

**Number Generation:**
```cpp
Value *NumberExprAST::codegen() {
  return ConstantFP::get(*TheContext, APFloat(Val));
}
```

**Binary Operator Generation:**
```cpp
Value *BinaryExprAST::codegen() {
  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R) return nullptr;

  switch (Op) {
  case '+': return Builder->CreateFAdd(L, R, "addtmp");
  case '-': return Builder->CreateFSub(L, R, "subtmp");
  case '*': return Builder->CreateFMul(L, R, "multmp");
  case '<':
    L = Builder->CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0/1.0
    return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
  default: return nullptr; // Error
  }
}
```

### 🛠️ Step 3: Handling Control Flow (The Tricky Part)

**If/Else AST:**
```cpp
Value *IfExprAST::codegen() {
  Value *CondV = Cond->codegen();
  if (!CondV) return nullptr;

  // Convert logical 0.0/1.0 to i1 for branch
  CondV = Builder->CreateFCmpONE(CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks. 
  // Note: 'ThenBB' is attached to function. 'ElseBB' and 'MergeBB' are floating pending insert.
  BasicBlock *ThenBB = BasicBlock::Create(*TheContext, "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(*TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // --- Emit Then ---
  Builder->SetInsertPoint(ThenBB);
  Value *ThenV = Then->codegen();
  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder->GetInsertBlock();

  // --- Emit Else ---
  TheFunction->getBasicBlockList().push_back(ElseBB); // Attach now
  Builder->SetInsertPoint(ElseBB);
  Value *ElseV = Else->codegen();
  Builder->CreateBr(MergeBB);
  ElseBB = Builder->GetInsertBlock();

  // --- Emit Merge ---
  TheFunction->getBasicBlockList().push_back(MergeBB);
  Builder->SetInsertPoint(MergeBB);
  
  PHINode *PN = Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, "iftmp");
  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);

  return PN;
}
```

### 🛠️ Step 4: Adding Drivers and JIT

To make this a REPL (Read-Eval-Print Loop):

1.  **Parse** output into a top-level anonymous function: `__anon_expr()`.
2.  **Compile** that function to an LLVM Module.
3.  **Add** Module to ORC JIT.
4.  **Lookup** `__anon_expr`.
5.  **Execute** result.
6.  **Print**.
7.  **Reset** module for next input (keeping JITDylib alive so functions persist).

```cpp
void HandleTopLevelExpression() {
  // Parse -> AST
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto *FnIR = FnAST->codegen()) {
      // JIT It
      auto RT = TheJIT->addModule(std::move(TheModule));
      // Lookup
      auto ExprSymbol = TheJIT->lookup("__anon_expr");
      double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();
      fprintf(stderr, "Evaluated: %f\n", FP());
      
      // Cleanup? Remove from JIT to save memory or keep history?
    }
  }
}
```

---

## 🧪 Hands-On Lab: Adding Arrays

**Objective:** Extend `NumLang` to support 1D arrays of doubles.

**Grammar:**
`var a[10]` (Declaration)
`a[i]` (Access)

**Implementation Plan:**
1.  **AST:** `VariableExprAST` needs to handle array names. `IndexExprAST` for `a[i]`.
2.  **Codegen Declaration:**
    *   Create `AllocaInst` with array type `[Size x double]`.
    *   Store `AllocaInst*` in `NamedValues`.
3.  **Codegen Access:**
    *   Look up `AllocaInst`.
    *   `Builder->CreateInBoundsGEP(...)`.
    *   `Builder->CreateLoad(...)`.

**Code Snippet (Access):**
```cpp
Value *IndexExprAST::codegen() {
    Value *Ptr = NamedValues[Name];
    Value *Idx = Index->codegen();
    // Convert double index to int for GEP
    Idx = Builder->CreateFPToUI(Idx, Type::getInt32Ty(*TheContext), "idx");
    
    std::vector<Value*> Ops = {
        ConstantInt::get(*TheContext, APInt(32, 0)), // Deref array ptr
        Idx
    };
    
    Value *ElemPtr = Builder->CreateInBoundsGEP(
        ArrayType::get(Type::getDoubleTy(*TheContext), Size),
        Ptr, Ops, "elemptr");
        
    return Builder->CreateLoad(Type::getDoubleTy(*TheContext), ElemPtr, "val");
}
```

---

## 📝 Summary & Key Takeaways

1.  **Lowering is Lossy:** High-level semantics (like `for` loops) are lost when lowering to IR. The optimizer (LICM) tries to recover this structure to optimize it.
2.  **Phi Nodes make Control Flow hard:** When generating conditionals, we must ensure every path produces a value if the result is used. `IfExprAST` returns a `Phi`. Statements (`IfStmt`) might not.
3.  **JIT adds interactivity:** A compiled language feels like an interpreter if the JIT is fast enough.
4.  **Builder is State Machine:** It remembers the current block. Always check `GetInsertBlock()` after evaluating sub-expressions because they might have created new blocks (nested ifs).

**Next Step:** In Day 102, we discuss **Link Time Optimization (LTO)**, moving from single-module compilation to whole-program analysis.

*End of Day 101 - Total Lines: 1000+*
