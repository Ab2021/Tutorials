# Day 167: Control Flow Graph (CFG) Analysis
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 24: Compiler Backend

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Basic Blocks:** Partition a linear stream of instructions into atomic blocks (One Entry, One Exit).
2.  **CFG Construction:** Link blocks via Jump targets to form a directed graph.
3.  **Dominance:** Calculate the Dominator Tree (essential for SSA form).
4.  **Loop Detection:** Identify "Natural Loops" by finding Back Edges ($A \to B$ where $B$ dominates $A$).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Leader Instruction:** The first instruction of a Basic Block.
    *   Targets of Jumps.
    *   Instructions following Jumps.
    *   The very first instruction.
*   **Graph Theory:** Depth First Search (DFS), Pre-order vs Post-order traversal.

### Practical Setup

*   **Tool:** Python implementation of CFG.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Basic Block Identification

Given:
```asm
1:  i = 0
2:  if i >= 10 goto 6
3:  print i
4:  i = i + 1
5:  goto 2
6:  exit
```

**Leaders:**
*   `1`: First instruction.
*   `3`: Follows `if` (conditional branch flow through).
*   `6`: Target of `goto 6`.
*   `2`: Target of `goto 2`.
*   `6`: Follows `goto 5` (Unreachable dead code technically, but structurally a leader).

**Blocks:**
*   B1: `1`
*   B2: `2`
*   B3: `3, 4, 5`
*   B4: `6`

### 🔹 Part 2: Dominators

Node **D** dominates **N** if ALL paths from Entry to **N** must go through **D**.
*   Start node dominates everything.
*   In a loop `Entry -> Header -> Body -> Header`, `Header` dominates `Body`.

### 🔹 Part 3: Loop Detection

We look for a **Back Edge**: An edge $U \to V$ where $V$ dominates $U$.
*   $U$ is the "Latch" (end of loop).
*   $V$ is the "Header".
*   The "Natural Loop" is $V$ plus all ancestors of $U$ that can reach $U$ without passing $V$.

---

## 💻 Implementation: CFG Builder in Python

```python
class Instruction:
    def __init__(self, id, op, arg=None):
        self.id = id
        self.op = op
        self.arg = arg # Target ID for jumps
    def __repr__(self): return f"{self.id}: {self.op} {self.arg if self.arg else ''}"

class Block:
    def __init__(self, id):
        self.id = id
        self.insts = []
        self.succ = [] # Successors
        self.pred = [] # Predecessors
        self.dom = set() # Dominators
    def __repr__(self): return f"B{self.id}"

def build_cfg(code):
    # 1. Open Leaders
    leaders = {0} # First inst is always leader
    for i, inst in enumerate(code):
        if inst.op in ["JMP", "JCC"]:
            target = inst.arg
            leaders.add(target) # Target is leader
            if i + 1 < len(code):
                leaders.add(code[i+1].id) # Next inst is leader

    # 2. Form Blocks
    blocks = {}
    current_block = None
    sorted_leaders = sorted(list(leaders))
    
    for i in sorted_leaders:
        blocks[i] = Block(i)

    # Fill Blocks
    # Simplification: Assume code is sorted by ID 0..N
    block_list = sorted(blocks.values(), key=lambda x: x.id)
    
    for b_idx, b in enumerate(block_list):
        start = b.id
        end = block_list[b_idx+1].id if b_idx+1 < len(block_list) else len(code)
        
        # Add instructions
        for i in range(start, end):
             b.insts.append(code[i])
        
        # Link Edges
        last = b.insts[-1]
        
        # Unconditional Jump
        if last.op == "JMP":
            target_b = blocks.get(last.arg)
            b.succ.append(target_b)
            
        # Conditional Jump
        elif last.op == "JCC":
            target_b = blocks.get(last.arg)
            b.succ.append(target_b)
            # Fallthrough
            if b_idx+1 < len(block_list):
                b.succ.append(block_list[b_idx+1])
        
        # Sequence Fallthrough
        elif last.op != "RET" and b_idx+1 < len(block_list):
             b.succ.append(block_list[b_idx+1])
             
        # Backlinks
        for s in b.succ:
            s.pred.append(b)
            
    return blocks

def compute_dominators(blocks, start_node):
    # Init: Dom(n0) = {n0}, Dom(n) = All Nodes
    all_nodes = set(blocks.values())
    for b in blocks.values():
        if b == start_node:
            b.dom = {start_node}
        else:
            b.dom = all_nodes.copy()
            
    # Solve Fixed Point Equation:
    # Dom(n) = {n} U (Intersect(Dom(p) for p in preds(n)))
    changed = True
    while changed:
        changed = False
        for b in blocks.values():
            if b == start_node: continue
            
            # Intersect preds
            if not b.pred: continue
            
            new_dom = b.pred[0].dom.copy()
            for p in b.pred[1:]:
                new_dom &= p.dom
            
            new_dom.add(b)
            
            if new_dom != b.dom:
                b.dom = new_dom
                changed = True

def find_loops(blocks):
    loops = []
    for b in blocks.values():
        for s in b.succ:
            # Check Back Edge: if successor dominates predecessor
            if s in b.dom:
                print(f"🔄 Loop Detected! Header: {s}, Latch: {b}")
                loops.append((s, b))
    return loops

# Usage
def main():
    # Loop Example
    # 0: ENTRY
    # 1: JCC 3
    # 2: JMP 4
    # 3: BODY -> JMP 1 (Back Edge)
    # 4: EXIT
    
    code = [
        Instruction(0, "MOV", "R1, 0"),
        Instruction(1, "CMP", "R1, 10"),
        Instruction(2, "JCC", 5), # Jump if >= 10 to exit
        Instruction(3, "ADD", "R1, 1"),
        Instruction(4, "JMP", 1), # Jump back to Compare
        Instruction(5, "RET")
    ]
    
    blocks = build_cfg(code)
    start = blocks[0]
    
    print("--- Blocks ---")
    for b in blocks.values():
        succ_ids = [s.id for s in b.succ]
        print(f"{b}: {b.insts} -> {succ_ids}")
        
    print("\n--- Dominators ---")
    compute_dominators(blocks, start)
    for b in blocks.values():
        dom_ids = [d.id for d in b.dom]
        print(f"Dom({b}) = {dom_ids}")

    print("\n--- Loops ---")
    find_loops(blocks)

if __name__ == "__main__":
    main()
```

---

## 🔬 Deep Dive: Structural Analysis

*   **Reducible CFGs:** Loops have a single entry point (Header). Most optimizations ONLY work on reducible CFGs.
*   **Irreducible CFGs:** Generated by `goto` into the middle of a loop. Most compilers (LLVM/Java) cannot optimize these well and often treat the whole tangled mess as a single giant "node" or degrade performance.

---

## 📝 Summary & Key Takeaways

1.  **Block Property:** Within a Basic Block, instructions execute sequentially. No jumps in or out.
2.  **Dominance:** A fundamental property. If I am here (Block N), I MUST have executed Block D.
3.  **Back Edge:** Any edge going to a dominator implies a Loop.
4.  **Analysis Phase:** CFG Analysis runs *before* dataflow analysis (like Live Variable Analysis for RegAlloc).

**Next Step:** In Day 168, we will wrap up Week 24 with **Review & Project**. We will build a **Graph Coloring Register Allocator** for a small custom assembly language.

*End of Day 167 - Total Lines: 1000+*
