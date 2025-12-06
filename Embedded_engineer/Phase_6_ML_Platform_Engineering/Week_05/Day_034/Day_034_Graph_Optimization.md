# Day 34: Graph Optimization & Operator Fusion
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Explore the "Frontend" of an ML Compiler. Learn how compilers optimize the Computation Graph structure itself using techniques like **Operator Fusion**, **Layout Transformation**, and **Dead Code Elimination**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** the three types of Operator Fusion: Element-wise, Reduction, and Injective.
2.  **Explain** why Fusion reduces Memory Bandwidth pressure and Kernel Launch overhead.
3.  **Implement** a basic "Pattern Matcher" to detect fuse-able sequences (e.g., Conv+ReLU).
4.  **Understand** Layout Transform (NCHW vs NHWC) and why compilers rewrite graphs to match hardware preferences.
5.  **Visualize** Graph Rewrites using naive Python graph structures.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Graph theory runs on CPU).

### Software Environment
- Python 3.x.
- `networkx` (for graph visualization/manipulation).

### Prior Knowledge
- Neural Network Layers (Conv, ReLU, Add).
- Memory Hierarchy (Registers are fast, VRAM is slow).

---

## 📖 Theoretical Foundation

### 1. The Cost of modularity
Deep Learning frameworks design layers as separate blocks:
```python
x = conv(input) # Kernel 1: Read Input, Write X to Global Mem
y = relu(x)     # Kernel 2: Read X from Global Mem, Write Y
```
This is **Memory Bound**. The data `x` traveled VRAM -> Core -> VRAM -> Core -> VRAM.

**Fusion** merges them:
```python
y = conv_relu(input) # Kernel Fused: Read Input, Compute Conv+ReLU in Registers, Write Y
```
Data `x` never hits VRAM. **2x Speedup on bandwidth-bound ops.**

### 2. Types of Pass

*   **Algebraic Simplification:** `x + 0 -> x`, `x * 1 -> x`.
*   **Constant Folding:** `Conv(Constant_Input)` -> Precompute result at compile time.
*   **Dead Code Elimination:** Remove branches that are never taken.
*   **Layout Transform:**
    *   Tensor Cores prefer **NHWC** (Channel-Last) so vector load `load.v4` gets contiguous channels (RGBA).
    *   PyTorch uses **NCHW** (Channel-First).
    *   Compiler inserts `Transpose(NCHW->NHWC)` before heavy ops and `Transpose(NHWC->NCHW)` after.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Fusion Pass

We will build a toy Compiler Frontend using `networkx` to demonstrate how a generic Fusion Pass works.

#### 📁 `src/toy_compiler_pass.py`
```python
#!/usr/bin/env python3
"""
Day 34: Toy Graph Optimization Pass
Phase 6: DL Compiler Stack
"""

import networkx as nx
import matplotlib.pyplot as plt

# 1. Define Ops
class Op:
    def __init__(self, name, type):
        self.name = name
        self.type = type
    
    def __repr__(self):
        return f"{self.type}:{self.name}"

def build_graph():
    G = nx.DiGraph()
    # Create the graph: Input -> Conv -> Add -> Relu -> Output
    ops = [
        Op("input", "Input"),
        Op("conv1", "Conv2D"),
        Op("bias1", "Add"),
        Op("act1", "ReLU"),
        Op("pool1", "MaxPool"),
        Op("output", "Output")
    ]
    
    # Add Nodes
    for op in ops:
        G.add_node(op.name, data=op)
        
    # Add Edges (Data Flow)
    G.add_edge("input", "conv1")
    G.add_edge("conv1", "bias1")
    G.add_edge("bias1", "act1")
    G.add_edge("act1", "pool1")
    G.add_edge("pool1", "output")
    return G

# 2. Pattern Matching Fusion Pass
def fuse_conv_relu(G):
    """
    Looks for pattern: Conv2D -> Add -> ReLU
    Fuses into: CBR_Fused
    """
    print("Running Fusion Pass...")
    nodes = list(nx.topological_sort(G))
    optimized = False
    
    # Naive Pattern Matcher
    for i in range(len(nodes) - 2):
        n1 = nodes[i]
        n2 = nodes[i+1] # Assuming linear chain for demo
        n3 = nodes[i+2]
        
        op1 = G.nodes[n1]['data']
        op2 = G.nodes[n2]['data']
        op3 = G.nodes[n3]['data']
        
        # Check Pattern: Conv -> Add -> ReLU
        if (op1.type == "Conv2D" and 
            op2.type == "Add" and 
            op3.type == "ReLU"):
            
            # Check Connectivity (n1->n2, n2->n3)
            if G.has_edge(n1, n2) and G.has_edge(n2, n3):
                print(f"  Found Pattern: {n1} -> {n2} -> {n3}")
                
                # Perform Rewrite
                new_name = f"{n1}_fused"
                new_op = Op(new_name, "Fused_ConvAddRelu")
                
                # Add new node
                G.add_node(new_name, data=new_op)
                
                # Wire inputs: predecessors of n1 point to new node
                for pred in list(G.predecessors(n1)):
                    G.add_edge(pred, new_name)
                    
                # Wire outputs: successors of n3 point from new node
                for succ in list(G.successors(n3)):
                    G.add_edge(new_name, succ)
                    
                # Remove old nodes
                G.remove_node(n1)
                G.remove_node(n2)
                G.remove_node(n3)
                optimized = True
                break # Graph changed, restart or handle carefully traversal
                
    return optimized

def print_graph(G):
    chain = []
    for node in nx.topological_sort(G):
        chain.append(G.nodes[node]['data'].type)
    print(" -> ".join(chain))

def main():
    G = build_graph()
    print("--- Before Optimization ---")
    print_graph(G)
    
    # Run Pattern Matcher
    fuse_conv_relu(G)
    
    print("\n--- After Optimization ---")
    print_graph(G)

if __name__ == "__main__":
    main()
```

### 👨‍💻 Advanced: Layout Transformation Cost

In real compilers, layout usage is a cost model decision.
Cost = `Execute_Time_in_Layout` + `Transform_Overhead`.

```python
# Conceptual Decision logic
def decide_layout(op, machine):
    cost_nchw = machine.perf(op, "NCHW")
    cost_nhwc = machine.perf(op, "NHWC")
    
    transpose_cost = machine.bandwidth * tensor_size
    
    if cost_nhwc + transpose_cost < cost_nchw:
        return "NHWC" + "Insert_Transposes"
    return "NCHW"
```

---

## 🔬 Lab Exercise: "Dead Code Buster"

### Task
Implement a Dead Code Elimination pass for the Toy Compiler.
1.  Create a graph with a branch that has no path to "Output".
    *   `Input -> A -> B -> Output`
    *   `Input -> C -> D` (D has no successors, it is dead).
2.  Algorithm:
    *   Loop Backwards from leaves.
    *   If a node is NOT "Output" and has `out_degree == 0`, remove it.
    *   Repeat until stable.

### Importance
Model exporters often leave "Debug Nodes" or "Loss Outputs" (used during training) in the ONNX file. Compilers must strip these to save 10-20% compute.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Fusion is King:** The single most important speedup in DL Compilers is fusing element-wise ops into their producers. It hides memory latency completely.
2.  **Pattern Matching:** Compilers function by recognizing sub-graphs (`Conv -> Add`) and replacing them with optimized implementations (`cudnnConvBiasRelu`).
3.  **Graph Rewrites:** The graph structure is mutable. Optimization is just a sequence of safe mutations.
4.  **Limits:** You cannot fuse everything. Fusing a `MatMul` into a `Conv` is usually impossible because they require different data tiling/blocking strategies.

### API Summary
```python
# NetworkX standard ops
G.add_node(name, attr=val)
G.add_edge(u, v)
nx.topological_sort(G) # Order of execution
G.predecessors(n)
G.successors(n)
```

---

**Day 34 Complete** ✅

*Next: Day 35 - Week 5 Project - Building a Mini-Compiler that takes a graph, fuses ops, and generates Triton/TVM code.*
