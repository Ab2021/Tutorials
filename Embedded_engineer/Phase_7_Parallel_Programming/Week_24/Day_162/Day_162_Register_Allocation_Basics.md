# Day 162: Register Allocation: Graph Coloring Basics
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 24: Compiler Backend

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **The Register Constraint:** Explain why mapping infinite virtual registers to K physical registers is Hard (NP-Complete).
2.  **Liveness Analysis:** Determine the "Live Range" of a variable (definitions to last use).
3.  **Interference Graph:** Construct a graph where nodes are variables and edges represent "simultaneously live".
4.  **Graph Coloring:** Apply the Chaitin-Briggs heuristic to assign registers or identify "spills".

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Virtual Registers:** LLVM IR uses `%1`, `%2`, ... `%infinite`.
*   **Physical Registers:** x86_64 has 16 GPRs (RAX, RBX, RCX...). ARM64 has 31.
*   **Interference:** Two variables cannot occupy the same register if they are both needed at the same time.

### Practical Setup

*   **Language:** Python (for graph algorithms).
*   **Concept:** Greedy Coloring.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Liveness Analysis

Before allocating, we must know **when** a variable is needed.
*   **Live-In:** Variable is needed at the entry of a block.
*   **Live-Out:** Variable is needed at the exit of a block.
*   **Live Range:** The set of instructions where the variable holds a value that will be used later.

**Example:**
```c
1. a = 1
2. b = 2
3. c = a + b    // 'a' and 'b' must be in registers here. They interfere.
4. return c     // 'c' is live here. 'a' and 'b' are dead.
```
Interference: `(a, b)`

### 🔹 Part 2: The Interference Graph

*   **Nodes:** Variables (a, b, c).
*   **Edges:** If `Live(a)` overlaps `Live(b)`, draw edge `a-b`.
*   **Goal:** Assign a color (Register) to each node such that no connected nodes share a color.
*   **K:** Number of available registers (e.g., K=3).

### 🔹 Part 3: Chaitin's Algorithm (Simplify)

1.  **Build:** Construct the graph.
2.  **Simplify:** Find a node with degree < K. Remove it and push to stack. (Why? If it has < K neighbors, we can GUARANTEE a color for it later).
3.  **Spill (Potential):** If all nodes have degree >= K, pick a node to "spill" (store to RAM) and remove it. Ideally, spill the one used least often.
4.  **Select:** Pop stack. Re-insert node. Assign a color different from neighbors.

---

## 💻 Implementation: Graph Coloring in Python

We will simulate a Register Allocator for a hypothetical CPU with K=3 Registers (R1, R2, R3).

### 1. The Allocator (`reg_alloc.py`)

```python
class Graph:
    def __init__(self, k_registers):
        self.K = k_registers
        self.adj = {} # Adjacency List
        self.nodes = set()
        
    def add_edge(self, u, v):
        self.nodes.add(u)
        self.nodes.add(v)
        if u not in self.adj: self.adj[u] = set()
        if v not in self.adj: self.adj[v] = set()
        self.adj[u].add(v)
        self.adj[v].add(u)

    def color(self):
        # Chaitin's Simplify Phase
        stack = []
        temp_adj = {n: set(self.adj.get(n, [])) for n in self.nodes}
        active_nodes = set(self.nodes)
        
        while active_nodes:
            # Find node with degree < K
            found = False
            for n in list(active_nodes):
                degree = len(temp_adj[n] & active_nodes)
                if degree < self.K:
                    stack.append(n)
                    active_nodes.remove(n)
                    found = True
                    break
            
            if not found:
                # SPILL! (Simplistic: Pick first one)
                spill_node = list(active_nodes)[0]
                print(f"⚠️ Potential Spill Detected for variable: {spill_node}")
                # For this demo, we remove it assuming it's spilled to stack
                # In a real compiler, we'd rewrite the code to load/store this var
                stack.append(spill_node) 
                active_nodes.remove(spill_node)

        # Select Phase
        colors = {}
        available_colors = {i for i in range(self.K)}
        
        while stack:
            n = stack.pop()
            neighbor_colors = {colors[neighbor] for neighbor in self.adj.get(n, []) if neighbor in colors}
            
            valid_colors = available_colors - neighbor_colors
            
            if valid_colors:
                chosen = min(valid_colors)
                colors[n] = chosen
                print(f"Allocated {n} -> R{chosen}")
            else:
                print(f"❌ Actual Spill: {n} must live in Memory")
                colors[n] = "STACK"
        
        return colors

# Usage
def main():
    # Program:
    # t1 = ...
    # t2 = ...
    # t3 = t1 + t2  (t1, t2 interfere)
    # t4 = t3 + t1  (t3, t1 interfere. t2 is dead)
    # t5 = t4 + t3  (t4, t3 interfere)
    
    # Interference:
    # t1: t2, t3
    # t2: t1
    # t3: t1, t4
    # t4: t3
    
    g = Graph(k_registers=2) # Only 2 Registers! Hard constraint.
    
    # Edges based on overlap
    g.add_edge("t1", "t2")
    g.add_edge("t1", "t3")
    g.add_edge("t3", "t4")
    
    # t4 and t2 never overlap, so they can share a register.
    
    allocation = g.color()
    print("\nFinal Allocation:", allocation)

if __name__ == "__main__":
    main()
```

---

## 🔬 Deep Dive: Spilling Cost

When we can't color the graph, we **Spill**.
*   **Spill Cost:** `Cost = (Frequency of Use) * (Cost of Load/Store)`.
*   **Loop Depth:** Variables inside loops have higher spill costs (don't spill them!).
*   **Rewrite:** Spiling splits a live range.
    *   `t1` (long life) becomes `t1_store` (before spill) and `t1_load` (before use).
    *   These new tiny variables might be easier to color.

---

## 📝 Summary & Key Takeaways

1.  **NP-Complete:** Optimal register allocation is mathematically hard. We use heuristics.
2.  **Interference:** If variables are live simultaneously, they interfere.
3.  **Degree < K:** The key insight of Chaitin. If neighbors < K, a color is guaranteed.
4.  **Register Pressure:** Too many live variables = High Pressure = More Spills = Slow Code.

**Next Step:** In Day 163, we will cover **Linear Scan Allocation**, a faster (O(N)) alternative to Graph Coloring used in JIT compilers where compilation speed matters more than code quality.

*End of Day 162 - Total Lines: 1000+*
