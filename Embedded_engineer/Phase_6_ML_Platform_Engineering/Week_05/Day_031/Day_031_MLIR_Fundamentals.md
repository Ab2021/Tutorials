# Day 31: MLIR - Multi-Level Intermediate Representation
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Understand **MLIR**, the LLVM project that powers TensorFlow, JAX, and PyTorch 2.0. Learn how "Dialects" allow compilers to retain high-level structure (Loops, Linear Algebra) before lowering to machine code.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why efficient compilers need multiple levels of abstraction (Graph -> Loops -> Assembly).
2.  **Read** MLIR textual format (`.mlir`) and identify SSA values, Types, and Attributes.
3.  **Differentiate** between key Dialects: `linalg`, `scf` (Structured Control Flow), `affine`, and `func`.
4.  **Trace** a Lowering Pipeline: `linalg.matmul` -> `scf.for` -> `llvm.intr`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None specific (CPU).

### Software Environment
- MLIR is typically built from source (`llvm-project`). For this module, we used pre-built binaries if available, or theoretical analysis.
- *Optional:* `pip install iree-compiler-snapshot` (contains `iree-opt` which works like `mlir-opt`).

### Prior Knowledge
- Assembly code concepts (Registers/SSA).
- Matrix Multiplication (Logic).

---

## 📖 Theoretical Foundation

### 1. The "Fragile Common" Problem
In the past, TensorFlow had `XLA`, PyTorch had `Glow`, ONNX had `ONNX Runtime`.
*   If you wanted to add support for a new RISC-V Accelerator, you had to write 3 backends.
*   **MLIR** unifies this. It is an "infrastructure to build compilers".

### 2. Attributes of MLIR
*   **SSA (Static Single Assignment):** `%1 = add %0, %a`
*   **Dialects:** Namespaces of operations.
    *   `math.sqrt` belongs to `math`.
    *   `gpu.launch` belongs to `gpu`.
*   **Progressive Lowering:** Instead of jumping from "Matrix Mul" to "Assembly", MLIR goes:
    1.  `linalg.matmul` (High Level: abstract math).
    2.  `scf.for` (Mid Level: nested loops).
    3.  `llvm.fadd` (Low Level: instructions).

### 3. Key Dialects
*   **tensor:** Immutable values (good for Graphs).
*   **memref:** Mutable buffers (good for C++ generation).
*   **linalg:** High-level ops (`matmul`, `conv`).
*   **scf:** `scf.if`, `scf.for` (Structured Control Flow).
*   **affine:** Loops with strict constraints (easy to vectorize).

---

## 💻 Implementation

### 👨‍💻 Core: Anatomy of an MLIR File

We will write a textual MLIR file representing a function that calls a Linear Algebra operation and examine how it describes computation.

#### 📁 `src/example.mlir`
```mlir
// Day 31: MLIR Definition
// Phase 6: DL Compiler Stack

module {
  // Define a function 'matmul_demo'
  // Inputs: 2 tensors of 10x10 f32, 1 tensor of 10x10 f32 (accumulator)
  // Output: 1 tensor
  func.func @matmul_demo(%A: tensor<10x10xf32>, %B: tensor<10x10xf32>, %C: tensor<10x10xf32>) -> tensor<10x10xf32> {
    
    // linalg.matmul is a declarative Op.
    // It doesn't say "HOW" to loop, it says "WHAT" to do.
    %result = linalg.matmul
      ins(%A, %B : tensor<10x10xf32>, tensor<10x10xf32>)
      outs(%C : tensor<10x10xf32>) -> tensor<10x10xf32>
      
    // Return the result
    return %result : tensor<10x10xf32>
  }
}
```

### 👨‍💻 Analysis: Lowering to Loops (Theoretical)

If we ran this through a compiler pass `--convert-linalg-to-loops`, the `linalg.matmul` would be replaced by `scf.for` loops.

**Before:**
```mlir
linalg.matmul ...
```

**After (Conceptual):**
```mlir
// Outer Loop (i)
%result = scf.for %i = %c0 to %c10 step %c1 iter_args(%out_i = %C) -> (tensor<10x10xf32>) {
  // Inner Loop (j)
  %res_j = scf.for %j = %c0 to %c10 step %c1 iter_args(%out_j = %out_i) ... {
    // Reduction Loop (k)
    %res_k = scf.for %k = %c0 to %c10 ... {
       %a_val = tensor.extract %A[%i, %k]
       %b_val = tensor.extract %B[%k, %j]
       %prod = arith.mulf %a_val, %b_val
       ...
    }
  }
  scf.yield %res_j ...
}
```
*Note the verbosity. This is why we like Linalg! It keeps the intent clear until the last moment.*

### 👨‍💻 Python Binding Generation (IREE/Python)

Modern frameworks generate this MLIR via Python.

#### 📁 `src/generate_mlir.py`
```python
#!/usr/bin/env python3
"""
Day 31: Generating MLIR with Python
(Requires: pip install iree-compiler iree-runtime or mlir bindings)
Note: This script simulates the printing if libs unavailable.
"""

def generate_naive_matmul_ir():
    print('module {')
    print('  func.func @main(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {')
    print('    // Affine Loop Nest for Matrix Mul')
    print('    affine.for %i = 0 to 4 {')
    print('      affine.for %j = 0 to 4 {')
    print('        affine.for %k = 0 to 4 {')
    print('          %a = affine.load %A[%i, %k] : memref<4x4xf32>')
    print('          %b = affine.load %B[%k, %j] : memref<4x4xf32>')
    print('          %c = affine.load %C[%i, %j] : memref<4x4xf32>')
    print('          %p = arith.mulf %a, %b : f32')
    print('          %res = arith.addf %c, %p : f32')
    print('          affine.store %res, %C[%i, %j] : memref<4x4xf32>')
    print('        }')
    print('      }')
    print('    }')
    print('    return')
    print('  }')
    print('}')

if __name__ == "__main__":
    generate_naive_matmul_ir()
    print("\n[Analysis]")
    print("This 'affine' dialect code is perfect for the compiler.")
    print("It knows exact loop bounds (0 to 4).")
    print("It checks memory dependencies.")
    print("Optimization Pass 'affine-loop-tile' can easily tile this.")
```

---

## 🔬 Lab Exercise: "Dialect Detective"

### Scenario
You are debugging a compiler crash. You have an `.mlir` dump.

### Task
Identify the dialects used in these lines:
1.  `%0 = arith.constant 42 : i32` -> **arith** dialect.
2.  `gpu.launch blocks(%b0, %b1, %b2) ...` -> **gpu** dialect.
3.  `%1 = tensor.empty() : tensor<10xf32>` -> **tensor** dialect.
4.  `tt.dot %a, %b, %c ...` -> **Triton (tt)** dialect (used by OpenAI Triton).

### Why it matters?
If you see `scf.for`, you are on CPU/Host logic usually. If you see `gpu.thread_id`, you are inside a GPU kernel. Understanding where you are in the lowering stack helps pinpoint performance issues.

---

## 📝 Daily Summary

### Key Takeaways
1.  **MLIR is a Framework:** It's not a single "language" but a toolkit to build IRs.
2.  **Dialects are Modular:** You can mix `gpu` ops and `arith` ops in the same function.
3.  **Progressive Lowering:** By keeping ops "High Level" (linalg.matmul) as long as possible, we can do easy mathematical fusions (Matmul + Bias + Relu) before getting lost in the details of loops and pointers.
4.  **The Future:** PyTorch 2.0 `torch.compile` generates Triton and MLIR/Inductor code. This is the underlying tech of next-gen Frameworks.

### Syntax Cheatsheet
```mlir
// Operation structure
%results = dialect.op_name %operands { attributes } : (types) -> (result_types)

// Example
%1 = arith.addf %0, %matrix : f32
```

---

**Day 31 Complete** ✅

*Next: Day 32 - XLA (Accelerated Linear Algebra) & JAX - Seeing MLIR in action within Google's JAX library.*
