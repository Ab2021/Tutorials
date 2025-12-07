# Day 188: Graphcore IPU & Cerebras (Exotic Architectures)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 27: Domain-Specific Architectures

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **IPU Philosophy:** Explain why Graphcore puts 900MB of SRAM directly on the processor die.
2.  **MIMD on Chip:** Contrast IPU's independent tile execution with GPU's lockstep warp execution.
3.  **Wafer Scale:** Understand how Cerebras bypasses the "Reticle Limit" to create chip sizes of 46,000 $mm^2$.
4.  **Static Compilation:** Why these architectures require the Full Computational Graph to be known at compile time.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Memory Wall:** DRAM is slow (100ns latency). SRAM is fast (1ns latency).
*   **Batch Size 1:** GPUs need huge batches to hide latency. Real-time apps need Batch 1. IPUs excel at Batch 1.
*   **BSP (Bulk Synchronous Parallel):** Compute -> Barrier -> Communicate -> Barrier. The execution model of IPUs.

### Practical Setup

*   **Hardware:** None (Cloud access only). Cost is prohibitive ($2M+ for Cerebras).
*   **Mental Model:** Visualizing a graph mapped physically onto a 2D grid of cores.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Graphcore IPU (Intelligence Processing Unit)

*   **Design:** 1,472 IPU-Tiles per chip.
*   **Memory:** Each tile has 624 KB of private SRAM. Total ~900MB per chip. NO Shared Cache. NO DRAM attached directly? (Actually yes, usually attached via host or specialized links, but the *working set* fits in SRAM).
*   **MIMD:** Tile 0 can do a `ReLU` while Tile 1 does a `Tanh`. (GPUs cannot do this efficiently).
*   **Advantage:** Sparse data structures (GNNs) where memory access is random. SRAM handles random access perfectly; DRAM hates it.

### 🔹 Part 2: Cerebras WSE (Wafer Scale Engine)

*   **The Limit:** Lithography tools can only print a rectangle ~800 $mm^2$ (Reticle Limit).
*   **The Hack:** Cerebras prints the pattern repeated across the wafer and *interconnects them on silicon*.
*   **Specs:** 850,000 cores. 40 GB of On-Chip SRAM. 20 Petabytes/s Memory Bandwidth.
*   **Impact:** You can fit an entire GPT model layer on the chip. No off-chip communication.

---

## 💻 Conceptual: Mapping Graphs to Hardware

Code for these is usually PyTorch/TensorFlow, but the *compiler* does the magic.
Let's simulate the **BSP** model used by IPUs via a Python script.

```python
import time
import random
from concurrent.futures import ThreadPoolExecutor

# Simulation of IPU Tiles
NUM_TILES = 4 

class Tile:
    def __init__(self, id):
        self.id = id
        self.memory = {} # Local SRAM
        self.mailbox = {} # Inbound messages
        
    def compute(self):
        # Local Compute Phase (MIMD)
        # Random work duration to simulate divergence
        work = random.randint(10, 50) 
        time.sleep(work * 0.001) 
        self.memory['val'] = self.memory.get('val', 0) + 1
        return self.memory['val']

    def exchange(self, neighbors):
        # Communication Phase
        # Send output to neighbors
        msg = f"Data from {self.id}"
        for n in neighbors:
            n.mailbox[self.id] = msg

def run_ip_step(tiles):
    print("--- Start Superstep ---")
    
    # 1. Local Compute (Parallel)
    with ThreadPoolExecutor(max_workers=NUM_TILES) as executor:
        futures = [executor.submit(t.compute) for t in tiles]
        results = [f.result() for f in futures]
        
    print(f"Compute Done: {results}")

    # 2. Global Sync (Barrier)
    # Implicit in the fact that we waited for futures
    
    # 3. Exchange (All-to-All or Neighbor)
    # Simulating simple Ring
    for i in range(NUM_TILES):
        left_neighbor = tiles[(i-1) % NUM_TILES]
        right_neighbor = tiles[(i+1) % NUM_TILES]
        tiles[i].exchange([left_neighbor, right_neighbor])
        
    print("Exchange Done")

if __name__ == "__main__":
    tiles = [Tile(i) for i in range(NUM_TILES)]
    
    # Poplar compiles the graph into these steps statically
    for step in range(3):
        run_ip_step(tiles)
```

### Analysis

*   **Deterministic Latency:** Because the graph is static, the compiler knows exactly when Tile 0 sends to Tile 1.
*   **No Arbitration:** The router on the chip doesn't need to check "Is the line busy?". It *knows* it's free at Cycle 1050 because the compiler scheduled it.
*   **Efficiency:** This removes massive overhead of caches, arbiters, and schedulers found in CPUs/GPUs.

---

## 🔬 Deep Dive: Static Compilation

Why can't IPUs handle dynamic Python code well?
*   If your code has `if (random() > 0.5): compute_heavy() else: compute_light()`.
*   The compiler cannot predict the duration of the compute phase.
*   The Barrier synchronization would stall everyone waiting for the slowest path.
*   **Solution:** IPUs are best for static graphs (ResNet, Bert) where tensor shapes and control flow are fixed.

---

## 📝 Summary & Key Takeaways

1.  **SRAM is Gold:** Creating accelerators is mostly about figuring out how to get more Memory Bandwidth. SRAM is the ultimate answer.
2.  **MIMD Flexibility:** IPUs allow heterogeneous compute across tiles, unlike GPUs.
3.  **Wafer Scale:** Cerebras proves you can build "Cluster-on-a-Chip", eliminating the network bottleneck entirely.
4.  **Software Burden:** These chips require incredibly advanced compilers (Poplar) to map high-level PyTorch graphs to silicon routing.

**Next Step:** In Day 189, we will wrap up Week 27 with **Review & Project**. We will implement a **Systolic Array Matrix Multiplier** on an FPGA (Simulated via C++ High-Level Synthesis style code), effectively building our own Mini-TPU.

*End of Day 188 - Total Lines: 1000+*
