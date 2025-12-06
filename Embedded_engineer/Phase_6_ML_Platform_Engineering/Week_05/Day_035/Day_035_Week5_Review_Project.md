# Day 35: Week 5 Review & The Mini-DL Compiler
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Consolidate mastery of the Compiler Stack by building a **Mini-Compiler** from scratch. It will take a computation graph, perform operator fusion, and generate **OpenAI Triton** kernels for execution.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** the full compilation pipeline: Frontend (Graph) -> Optimizer (Passes) -> Backend (Codegen).
2.  **Implement** a Graph Fusion pass that merges element-wise operations.
3.  **Generate** valid Triton Python code dynamically from graph nodes.
4.  **Execute** the compiled graph on a GPU.
5.  **Review** key concepts from Week 5 (TVM, MLIR, JAX, Triton, Optimization).

---

## 📚 Week 5 Recap

### Topics Covered

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 29 | Intro to Compilers | Library approaches vs Compilers. TVM Workflow. |
| 30 | AutoTuning | AutoTVM (Templates) & Ansor (Search). Tuning loop split factors. |
| 31 | MLIR | Dialects (`linalg`, `scf`), Lowering Pipeline, SSA form. |
| 32 | JAX & XLA | Pure functions, JIT compilation, HLO Fusion. |
| 33 | OpenAI Triton | Block-based programming, `@triton.jit`, `tl.load/store`. |
| 34 | Graph Opt | Algebraic Simplification, Operator Fusion (Vertical), Layout Transform. |

### The "Stack"
1.  **Frontend:** PyTorch/JAX (User Code).
2.  **Graph IR:** MLIR/Relay (Optimization passes: Fusion, Layout).
3.  **Kernel IR:** Triton/TVM TIR (Loop schedules, Tiling).
4.  **Hardware:** PTX/SASS (GPU Assembly).

---

## 🏗️ Week 5 Project: "GraphToTriton" Compiler

### Project Overview
We will build a simple compiler `GraphToTriton` that:
1.  Defines a Graph: `Input -> Add -> Relu -> Output`.
2.  Fuses it into: `Input -> Fused_Add_Relu -> Output`.
3.  Generates a Triton Kernel: `def fused_kernel(...)`.
4.  Runs it.

### Architecture
```
[User Graph] 
    │
    ▼
[Optimizer Pass] ──▶ (Identifies 'Add+Relu')
    │
    ▼
[Codegen] ─────────▶ (Writes 'triton_kernels.py')
    │
    ▼
[Runtime] ─────────▶ (Loads & Executes Kernel)
```

---

## 💻 Complete Project Implementation

### Step 1: The Graph IR

#### 📁 `project/ir.py`
```python
class Node:
    def __init__(self, name, op_type, inputs=[]):
        self.name = name
        self.op_type = op_type # "Input", "Add", "Relu"
        self.inputs = inputs
        
    def __repr__(self):
        return f"{self.name}({self.op_type})"

class Graph:
    def __init__(self):
        self.nodes = [] # Topological order list
        
    def add_node(self, node):
        self.nodes.append(node)
```

### Step 2: The Fusion Optimizer

#### 📁 `project/optimizer.py`
```python
from ir import Node, Graph

def fuse_add_relu(graph):
    print("Running Fusion Pass: Add + Relu...")
    new_nodes = []
    skip = set()
    
    for i in range(len(graph.nodes)):
        if i in skip: continue
        
        curr = graph.nodes[i]
        
        # Check standard fusion pattern: Add -> Relu
        # Simplification: Only look at index i and i+1
        if i + 1 < len(graph.nodes):
            next_node = graph.nodes[i+1]
            
            # Pattern Match
            if (curr.op_type == "Add" and 
                next_node.op_type == "Relu" and 
                next_node.inputs[0] == curr):
                
                print(f"  Fusing {curr.name} + {next_node.name} -> FusedAddRelu")
                
                # Create Fused Node
                fused = Node(
                    name=f"fused_{curr.name}_{next_node.name}",
                    op_type="FusedAddRelu",
                    inputs=curr.inputs
                )
                new_nodes.append(fused)
                skip.add(i+1) # Skip the Relu
                continue
        
        new_nodes.append(curr)
        
    new_graph = Graph()
    new_graph.nodes = new_nodes
    return new_graph
```

### Step 3: Triton Codegen

#### 📁 `project/codegen.py`
```python
def generate_kernel_code(node):
    if node.op_type == "FusedAddRelu":
        # Generate Python source for Triton
        code = f"""
@triton.jit
def {node.name}_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute (Fused)
    val = x + y          # Add
    res = tl.maximum(val, 0.0) # Relu
    
    # Store
    tl.store(out_ptr + offsets, res, mask=mask)
"""
        return code
    return ""

def generate_full_source(graph):
    src = "import torch\nimport triton\nimport triton.language as tl\n\n"
    for node in graph.nodes:
        if node.op_type.startswith("Fused"):
            src += generate_kernel_code(node) + "\n"
    return src
```

### Step 4: End-to-End Pipeline

#### 📁 `project/main.py`
```python
import torch
from ir import Graph, Node
from optimizer import fuse_add_relu
from codegen import generate_full_source
import os

def main():
    # 1. Build Graph
    # A + B -> C -> Relu -> Out
    g = Graph()
    inp_a = Node("A", "Input")
    inp_b = Node("B", "Input")
    add_node = Node("add1", "Add", inputs=[inp_a, inp_b])
    relu_node = Node("relu1", "Relu", inputs=[add_node])
    
    g.add_node(inp_a)
    g.add_node(inp_b)
    g.add_node(add_node)
    g.add_node(relu_node)
    
    print("--- Original Graph ---")
    print(g.nodes)
    
    # 2. Optimize
    opt_g = fuse_add_relu(g)
    print("\n--- Optimized Graph ---")
    print(opt_g.nodes)
    
    # 3. Codegen
    print("\n--- Generating Kernel ---")
    kernel_src = generate_full_source(opt_g)
    print(kernel_src)
    
    # Write to file
    with open("generated_kernels.py", "w") as f:
        f.write(kernel_src)
        
    # 4. Runtime / Execution (Dynamic Import)
    print("\n--- Executing ---")
    import generated_kernels
    
    # Setup Data
    size = 1024
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    out = torch.empty_like(a)
    
    # Find the fused node
    fused_node_name = opt_g.nodes[2].name # fused_add1_relu1
    kernel_func = getattr(generated_kernels, f"{fused_node_name}_kernel")
    
    # Launch Config
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    print(f"Launching {fused_node_name}_kernel...")
    kernel_func[grid](a, b, out, size, BLOCK_SIZE=1024)
    
    # Verify
    expected = torch.relu(a + b)
    assert torch.allclose(out, expected)
    print("Verification Passed! Graph -> Optimized -> Compiled -> Executed.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Double Fusion"

### Task
Extend the pipeline to support the `Mul -> Add -> Relu` pattern (Gemm-like activation).
1.  Add "Mul" to the IR support.
2.  Update Fusion Pass to look for triples: `Mul, Add, Relu`.
3.  Update Codegen to emit `val = x * y + z`.

This is exactly what **XLA** does. It pattern matches subgraph sequences and emits a single HLO instruction `fusion(...)` which lowers to a single loop.

---

## 📝 Week 5 Summary

You have demystified the "Black Box" of Deep Learning.
*   **PyTorch** is just a UI.
*   **The Compiler** (TVM/Triton/XLA) is the engine.
*   **Performance** comes from Fusion (saving Memory) and Tiling (saving Cache).

You are now capable of not just *using* AI frameworks, but *building* them or optimizing them when they fail.

---

**Week 5 Complete** ✅

*Next Week: Week 6 - Distributed Training on Multi-Node Clusters - Scaling from 1 GPU to 1000.*
