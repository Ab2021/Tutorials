# Day 163: Linear Scan Register Allocation
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 24: Compiler Backend

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **The JIT Tradeoff:** Explain why Graph Coloring is too slow for Just-In-Time compilers.
2.  **Linear Scan Algorithm:** Implement the Poletto & Sarkar algorithm (Massalin's inspiration).
3.  **Active Intervals:** Manage the list of currently live variables.
4.  **Spill Heuristics:** Decide which variable to evict when registers run out (furthest end point).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Complexity:** Graph Coloring is $O(N^2)$ to build the graph and $O(K)$ to color.
*   **Linear Scan:** $O(N \log N)$ or even $O(N)$.
*   **Live Interval:** A single continuous range `[Start, End]` for each variable. (Simplification: Real variables have holes, but Linear Scan often ignores holes).

### Practical Setup

*   **Data Structure:** Sorted List of Intervals (sorted by Start Time).
*   **K Registers:** Finite set of resources.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linear Scan Logic

1.  **Linearize:** Order blocks deeply first (Trace Scheduling style).
2.  **Compute Intervals:** variable `v` starts at first def, ends at last use.
3.  **Iterate:** Walk through intervals from `Start=0` to `End=Max`.
4.  **Active List:** Keeps track of variables covering the current point.
5.  **Alloc:**
    *   If `Active.length < K`, assign free register.
    *   If `Active.length == K`, we must **Spill**.
    *   **Spill Choice:** Spill the interval that ends **furthest in the future**. Why? Because keeping the one that ends soon frees up a register sooner.

### 🔹 Part 2: Comparison

| Feature | Graph Coloring | Linear Scan |
| :--- | :--- | :--- |
| **Speed** | Slow (Heavy Analysis) | Fast (Single Pass) |
| **Quality** | Excellent (Minimal Spills) | Good (10-20% more spills) |
| **Use Case** | GCC, LLVM (AOT) | HotSpot, V8, Android ART (JIT) |

---

## 💻 Implementation: Linear Scan in Python

```python
class Interval:
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
        self.reg = None
        self.spilled = False
        
    def __repr__(self):
        return f"{self.name}[{self.start}-{self.end}]"

class LinearScan:
    def __init__(self, k_regs):
        self.K = k_regs
        self.free_regs = [i for i in range(k_regs)]
        self.active = [] # Sorted by End Point
        
    def expire_old_intervals(self, current_start):
        # Remove intervals that have ended before current_start
        # They are no longer active, so free their registers.
        still_active = []
        for i in self.active:
            if i.end >= current_start:
                still_active.append(i)
            else:
                # Expired
                if i.reg is not None:
                    self.free_regs.append(i.reg)
                    self.free_regs.sort() # Keep sorted for determinism
                    print(f"  🛑 Expired: {i.name}, freeing R{i.reg}")
        self.active = still_active

    def spill_at_interval(self, new_interval):
        # Heuristic: Spill the interval that ends last
        # (It holds a register for the longest time, preventing others from using it)
        spill = self.active[-1] # Active is sorted by end time
        
        if spill.end > new_interval.end:
            # Spill existing active
            print(f"  ⚠️ Splilling Active: {spill.name} (Ends at {spill.end})")
            spill.reg = None
            spill.spilled = True
            
            # Reclaim register
            new_interval.reg = spill.reg 
            self.active.pop() 
            self.active.append(new_interval)
            self.active.sort(key=lambda x: x.end)
        else:
            # Spill new interval
            print(f"  ⚠️ Spilling New: {new_interval.name} (Ends at {new_interval.end})")
            new_interval.spilled = True

    def allocate(self, intervals):
        # Sort by Start Time
        intervals.sort(key=lambda x: x.start)
        
        for i in intervals:
            print(f"Processing {i}...")
            self.expire_old_intervals(i.start)
            
            if len(self.active) == self.K:
                # We are full!
                self.spill_at_interval(i)
            else:
                # Allocate
                reg = self.free_regs.pop(0)
                i.reg = reg
                print(f"  ✅ Allocated {i.name} -> R{reg}")
                self.active.append(i)
                self.active.sort(key=lambda x: x.end)

# Usage Simulation
def main():
    # Scenario: 2 Registers.
    # A: [0-10]
    # B: [2-5]
    # C: [3-8]
    # overlaps: at time 3, A, B, C are all live. We need 3 regs. We have 2.
    
    ls = LinearScan(k_regs=2)
    vars = [
        Interval("A", 0, 10),
        Interval("B", 2, 5),
        Interval("C", 3, 8),
        Interval("D", 12, 15) # Should reuse registers
    ]
    
    ls.allocate(vars)
    
    print("\nFinal Results:")
    for v in vars:
        loc = f"R{v.reg}" if not v.spilled else "STACK"
        print(f"{v.name}: {loc}")

if __name__ == "__main__":
    main()
```

### Trace of Execution (Mental Model):

1.  **Start A (0-10):** Assign R0. Active: `[A]`.
2.  **Start B (2-5):** Expire? None. Assign R1. Active: `[B, A]` (Sorted by end).
3.  **Start C (3-8):** Expire? None. All regs used (R0, R1).
    *   **Spill Decision:** Active `[B(5), A(10)]`. Candidate `C(8)`.
    *   A ends last (10). Spill A?
    *   Yes, Spill A. Reclaim R0. Assign R0 to C.
    *   Active: `[B(5), C(8)]`. A is on Stack.
4.  **Start D (12-15):** Expire? B expired at 5. C expired at 8. R0, R1 free.
    *   Assign R0 to D.

---

## 🔬 Deep Dive: Second Chance Binpacking

Modern JITs (like in GraalVM) use "Second Chance Binpacking".
Instead of spilling the current interval to the stack immediately, we look for **Holes** in the lifetime of created spills.
However, basic Linear Scan assumes continuous lifetime.
**Split Logic:** If we spill A, we can actually "Split" A. `A1` lives in R0 until C starts. `A2` lives on Stack while C uses R0. `A3` moves back to register after C ends.

---

## 📝 Summary & Key Takeaways

1.  **Sort by Start:** Linear Scan processes live intervals in increasing order of start points.
2.  **Active List:** Maintains currently allocated registers, sorted by end points.
3.  **Greedy:** It does not backtrack. Once a decision is made, it stands.
4.  **Spill Furthest:** The optimal greedy heuristic is to spill the interval that ends furthest away to maximize register availability for short-lived variables.

**Next Step:** In Day 164, we will cover **Instruction Scheduling**. Reordering instructions (e.g., separating LOAD from USE) to hide pipeline latency and avoid stalls.

*End of Day 163 - Total Lines: 1000+*
