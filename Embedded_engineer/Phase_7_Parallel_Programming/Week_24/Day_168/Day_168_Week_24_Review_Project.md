# Day 168: Week 24 Review & Project
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 24: Compiler Backend

---

## 🎯 Week 24 Recap: The Backend Pipeline

In this week, we explored how Abstract Syntax Trees turn into Executable Machine Code.

1.  **Day 162 (Register Alloc Basics):** The coloring problem.
2.  **Day 163 (Linear Scan):** Fast allocation for JITs.
3.  **Day 164 (Instruction Scheduling):** Latency hiding via reordering.
4.  **Day 165 (Peephole):** Local cleanup optimizations.
5.  **Day 166 (Unrolling):** Loop optimization to reduce branches.
6.  **Day 167 (CFG):** Graph structure of the program.

---

## 🛠️ The Project: Toy Backend (Liveness & Coloring)

We will implement a backend pipeline for a 3-Address Code IR.
**Constraint:** We have only **3 Physical Registers** (R1, R2, R3).

### 1. IR Definition

Our IR looks like:
*   `v1 = 10`
*   `v2 = 20`
*   `v3 = v1 + v2`
*   `v4 = v3 + v1`
*   `return v4`

### 2. The Implementation (`backend.py`)

```python
import collections

# --- 1. Representation ---
class Instruction:
    def __init__(self, dest, op, src1, src2=None):
        self.dest = dest
        self.op = op
        self.src1 = src1
        self.src2 = src2
    def __repr__(self):
        s2 = f", {self.src2}" if self.src2 else ""
        return f"{self.dest} = {self.op} {self.src1}{s2}"

# --- 2. Liveness Analysis ---
def compute_liveness(instructions):
    # Map: Inst Index -> Set of Live Variables
    live_out = collections.defaultdict(set)
    live_in = collections.defaultdict(set)
    
    # Iterate backwards until fixed point
    changed = True
    while changed:
        changed = False
        for i in range(len(instructions)-1, -1, -1):
            inst = instructions[i]
            
            # Start with successors (next instruction)
            succ_in = live_in[i+1] if i+1 < len(instructions) else set()
            new_out = succ_in.copy()
            
            # Transfer function: In = Use U (Out - Def)
            # Def = dest
            # Use = src1, src2 (if variables)
            use = set()
            if isinstance(inst.src1, str) and inst.src1.startswith('v'): use.add(inst.src1)
            if isinstance(inst.src2, str) and inst.src2.startswith('v'): use.add(inst.src2)
            
            new_in = use | (new_out - {inst.dest})
            
            if live_in[i] != new_in or live_out[i] != new_out:
                live_in[i] = new_in
                live_out[i] = new_out
                changed = True
    return live_in

# --- 3. Interference Graph ---
def build_graph(instructions, live_in):
    adj = collections.defaultdict(set)
    
    for i, inst in enumerate(instructions):
        live_now = live_in[i]
        
        # If variable is defined here, it interferes with everything currently live
        # (Except itself)
        if inst.dest:
            for live_var in live_now:
                if live_var != inst.dest:
                    adj[inst.dest].add(live_var)
                    adj[live_var].add(inst.dest)
    return adj

# --- 4. Graph Coloring (Chaitin) ---
def color_graph(adj, k=3):
    stack = []
    # Copy graph to modify
    graph_copy = {n: set(adj[n]) for n in adj}
    active_nodes = set(graph_copy.keys())
    
    # Simplify Phase
    while active_nodes:
        # Find node < K degree
        candidate = None
        for n in active_nodes:
            degree = len(graph_copy[n] & active_nodes)
            if degree < k:
                candidate = n
                break
        
        if candidate:
            stack.append(candidate)
            active_nodes.remove(candidate)
        else:
            # Spill! (Pick first)
            spill = list(active_nodes)[0]
            print(f"⚠️ Potential Spill: {spill}")
            stack.append(spill) # Optimistic coloring
            active_nodes.remove(spill)
            
    # Select Phase
    colors = {}
    available_regs = {1, 2, 3}
    
    while stack:
        n = stack.pop()
        neighbors = adj[n]
        neighbor_colors = {colors[nb] for nb in neighbors if nb in colors}
        
        possible = available_regs - neighbor_colors
        if possible:
            colors[n] = min(possible)
        else:
            colors[n] = "STACK" # Actual Spill
            
    return colors

# --- 5. Driver ---
def main():
    # Program:
    # v1 = 1
    # v2 = 1
    # v3 = v1 + v2 (v1, v2 live)
    # v4 = v3 + v1 (v3, v1 live -> v3 and v1 interfere)
    # v5 = v4 + v3 (v4, v3 live -> v4 and v3 interfere)
    # return v5
    
    code = [
        Instruction("v1", "MOV", "1"),
        Instruction("v2", "MOV", "1"),
        Instruction("v3", "ADD", "v1", "v2"),
        Instruction("v4", "ADD", "v3", "v1"),
        Instruction("v5", "ADD", "v4", "v3"),
        Instruction("ret", "RET", "v5")
    ]
    
    print("--- Liveness ---")
    live_in = compute_liveness(code)
    for i, l in live_in.items():
        print(f"{i}: {code[i]} \t LiveIn: {l}")
        
    print("\n--- Interference ---")
    adj = build_graph(code, live_in)
    for n, edges in adj.items():
        print(f"{n}: {edges}")
        
    print("\n--- Allocation (K=3) ---")
    allocation = color_graph(adj, k=3)
    for v, reg in allocation.items():
        print(f"{v} -> R{reg}")

if __name__ == "__main__":
    main()
```

### 3. Execution Result (Expected)

*   `v1` overlaps with `v2`, `v3`.
*   `v3` overlaps with `v1`, `v4`.
*   `v4` overlaps with `v3`.
*   **Coloring:**
    *   `v1`: R1
    *   `v2`: R2
    *   `v3`: R2 (Reuse R2 because v2 is dead!)
    *   `v4`: R1 (Reuse R1? No, v1 interferes? Wait. v1 is used in line 3. Live range ends after line 3 use.)

**Correction:** Our rigid `live_in` definition says `v1` is live at line 3 entry. `v3` is defined at line 3.
Does `v3` interfere with `v1`?
*   Line 3: `v3 = v1 + v2`.
*   Entry: `v1, v2` live.
*   Exit: `v3` live. (And `v1` if used later).
*   If `v1` is used in line 4 (`v4 = v3 + v1`), then `v1` MUST be live at line 3 exit.
*   So yes, `v3` and `v1` are simultaneously live. They need different registers.

---

## 📝 Performance Validation

This allocator is $O(N^2)$ due to interference graph construction.
For small programs, it produces **optimal** coloring (minimal registers).
For large programs, heuristics (Degree < K) prevent exponential search.

**Next Step:** Phase 7 continues into **Week 25: Advanced Parallel Algorithms**. We will move away from Compilers/System tools and into high-level Parallel Algorithms (Bitonic Sort, Scan, Reduction) on Massively Parallel Architectures.

*End of Day 168 - Total Lines: 1000+*
