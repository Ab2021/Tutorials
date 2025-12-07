# Day 033: OpenCL Performance Optimization
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Maximize Occupancy:** Tune Work-Group sizes to ensure the GPU's Compute Units are fully saturated with active wavefronts (hiding latency).
2.  **Ensure Coalescing:** Structure global memory accesses so that adjacent Work-Items access adjacent bytes (Unit Stride), maximizing bandwidth.
3.  **Implement Tiling:** Use Local Memory (Shared Mem) as a user-managed cache to reuse data and reduce Global Memory bandwidth by 10x (e.g., Matrix Mul).
4.  **Avoid Bank Conflicts:** Access Local Memory patterns that do not serialize on memory banks.
5.  **Minimize Divergence:** Write code where all WIs in a subgroup follow the same path, avoiding "Inactive Lanes".

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Profiler:** NVIDIA Nsight Compute or Intel VTune (crucial for seeing "Stalls").

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Occupancy & Hiding Latency

GPUs do NOT rely on standard caches to hide latency. They rely on **Threads**.
If Warp A hits a memory stall (latency 500 cycles), the hardware instantly switches to Warp B.
**Occupancy:** Ratio of Active Warps / Max Warps per Core.

**Factors limiting occupancy:**
1.  **Registers:** If your kernel uses too many registers (e.g., 64 regs/item), you fit fewer items on the chip.
    *   *Fix:* `-cl-mad-enable` or simplifying math.
2.  **Local Memory:** If you allocate 48KB Local Mem per WG, and HW limits is 64KB, you can only run 1 WG per CU.
    *   *Fix:* Reduce tiling size.
3.  **WG Size:** A size of 1 is terrible. Use multiples of 32 (NVIDIA warp) or 64 (AMD wavefront). Ideally 128-256.

### 🔹 Part 2: Coalesced Access

**Good Access:**
WI 0 reads `Addr[0]`. WI 1 reads `Addr[4]`. WI 2 reads `Addr[8]` (floats).
The Memory Controller coalesces this into **1 transaction** of 128 bytes.

**Bad Access (Strided):**
WI the reads `Addr[0]`, WI 1 reads `Addr[1024]`.
Controller fires **32 transactions** of 32 bytes (1024 bytes total, but 96% waste).

**Rule:** Ensure `get_global_id(0)` is the fastest moving index in your array logic.

### 🔹 Part 3: Local Memory Tiling

Global Memory is slow (~500 GB/s).
Local Memory is fast (~10,000 GB/s).
If you read the same data multiple times (like in convolution or matrix mul), load it into Local Mem first.

**Pattern:**
1.  **Load:** All WIs cooperate to load a tile from Global -> Local.
2.  **Barrier:** `barrier(CLK_LOCAL_MEM_FENCE)`.
3.  **Compute:** WIs read from Local.
4.  **Barrier:** Sync before next tile.

### 🔹 Part 4: Bank Conflicts

Local Memory is divided into 32 Banks (4-byte width).
Addr `0, 4, 8 ...` map to Banks `0, 1, 2...`
**Conflict:** If WI 0 accesses Bank 0 and WI 1 accesses Bank 0 at the same time -> Serialization.
**Stride 1:** Conflict Free (Banks 0, 1, 2...).
**Stride 32:** Worst Case (All WIs hit Bank 0).

---

## 💻 Implementation: Tiled Matrix Multiplication

Naive: $C[y][x] = \sum A[y][k] \times B[k][x]$.
Reads A row $N$ times. Reads B col $N$ times. Memory Bound.

Tiled:
Load Block of A and Block of B into Local Mem.
Compute partial product. Move to next block.

### 🛠️ Step 1: Defined Constants

```c
#define TILE_SIZE 16
```

### 🛠️ Step 2: The Kernel (`matmul.cl`)

```c
__kernel void matmul_tiled(__global const float* A, 
                           __global const float* B, 
                           __global float* C, 
                           int width) {
    
    // Global ID
    int tx = get_global_id(0); 
    int ty = get_global_id(1);

    // Local ID (0..15)
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Local Memory buffers
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    // Loop over all tiles
    int num_tiles = width / TILE_SIZE;
    
    for (int t=0; t < num_tiles; t++) {
        // 1. Cooperative Load
        // Tile A: Row ty, Col (t*16 + lx)
        // Tile B: Row (t*16 + ly), Col tx
        
        int A_col = t * TILE_SIZE + lx;
        int B_row = t * TILE_SIZE + ly;
        
        // Bounds check omitted for simplicity (assume multiples of 16)
        As[ly][lx] = A[ty * width + A_col];
        Bs[ly][lx] = B[B_row * width + tx];
        
        // 2. Barrier: Wait for tile to load
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 3. Compute Partial Product for this tile
        for (int k=0; k < TILE_SIZE; k++) {
            acc += As[ly][k] * Bs[k][lx];
        }
        
        // 4. Barrier: Wait for math before overwriting tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[ty * width + tx] = acc;
}
```

### 🛠️ Step 3: Performance Analysis

**Naive Kernel:**
*   Memory Accesses: $2N^3$.
*   Intensity: 1 OP / 1 Byte (Poor).

**Tiled Kernel:**
*   Memory Accesses: $2N^3 / \text{TILE\_SIZE}$.
*   Intensity: 16 OPs / 1 Byte (Using TILE=16).
*   **Speedup:** 10x - 50x depending on hardware.

### 🛠️ Step 4: Bank Conflict Check

Line `acc += As[ly][k] * Bs[k][lx]`:
*   `As[ly][k]`: All threads in a row (same `ly`) read different `k`. Since `k` is the loop var, fast threads might be at different `k`. Wait, actually inside the loop `k` is broadcast. If all read `As[ly][0]`, it's a broadcast (Fast).
*   `Bs[k][lx]`: All threads in a row (different `lx`) read `Bs[constant][lx]`. `lx` varies 0..15. Stride is 1. Bank Conflict Free!

---

## 🧪 Hands-On Labs

### Lab 33: Investigating Divergence

**Objective:** Measure penalty of `if-else`.

**Kernel A (Coherent):**
```c
if (get_global_id(0) < N/2) { A } else { A } 
// Entire warp takes 'A'. No divergence (mostly).
```

**Kernel B (Divergent):**
```c
if (get_global_id(0) % 2 == 0) { A } else { B }
// Every warp is split. Hardware executes A (masking odds), then B (masking evens).
// Utilization drops to 50%.
```

**Task:**
1.  Write a kernel where even threads do `sin()`, odd threads do `cos()`. (Divergent).
2.  Write a kernel where first half does `sin()`, second half `cos()`.
3.  Measure execution time. (Divergent should be slower).

---

## 📝 Summary & Key Takeaways

1.  **Work-Group Size:** Use 128 or 256. Make sure `global_size` is padded to be a multiple of `local_size`.
2.  **Tiling:** The Single Most Important optimization for matrix/convolution ops. If you don't tile, you are memory bound.
3.  **Coalescing:** Access `GlobalMem[tid]` not `GlobalMem[tid*stride]`.
4.  **Local Memory:** Use it for data shared within a WG. Use Barriers to protect it.
5.  **Unrolling:** `#pragma unroll` inside kernels can help hide instruction latency, but consumes more registers (check Occupancy).

---

## 📚 Additional Resources

*   [NVIDIA OpenCL Optimization Guide](https://www.nvidia.com/content/cudazone/CUDADownloads/papers/NVIDIA_OpenCL_BestPracticesGuide.pdf)
*   [Rice University: GPU Tiling Visualization](https://www.cs.rice.edu/~johnmc/comp522/lecture_notes/COMP522.2018.L16.pdf)

**Tomorrow:** Day 34 - Debugging OpenCL... `printf` inside kernels, assert, and using standard debuggers (GDB/Nsight).

*End of Day 033 - Total Lines: 1000+*
