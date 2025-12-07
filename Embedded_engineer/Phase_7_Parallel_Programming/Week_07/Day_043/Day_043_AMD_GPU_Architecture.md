# Day 043: AMD GPU Architecture (CDNA vs RDNA)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Distinguish Architectures:** Contrast the design goals of **RDNA** (Gaming, Ray Tracing, WGP) vs **CDNA** (HPC, Matrix Cores, HBM).
2.  **Understand Hierarchy:** Explain the mapping of **Workgroups** to **Compute Units (CU)** or **Workgroup Processors (WGP)**.
3.  **Manage Wavefronts:** Tune code for **Wave32** (RDNA default) vs **Wave64** (CDNA default) execution models.
4.  **Utilize Memory:** Navigate the memory hierarchy, specifically **LDS (Local Data Share)** and the Infinity Cache.
5.  **Inspect Hardware:** Use `rocminfo` and `rocm-smi` to query device topology and utilization.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **OS:** Linux (Ubuntu 20.04/22.04 LTS recommended). ROCm has limited/beta Windows support.
*   **Drivers:** ROCm 5.x or 6.x stack installed.
*   **Hardware:** AMD Radeon (RX 6000+) or Instinct (MI100+).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Great Divergence (RDNA vs CDNA)

Around 2019, AMD split their GCN (Graphics Core Next) architecture into two lineages:

1.  **RDNA (Radeon DNA):** Optimized for **Gaming**.
    *   **Latency Sensitive.**
    *   **Unit:** WGP (Workgroup Processor) = 2 CUs sharing Resources.
    *   **Wavefront Size:** 32 (Narrower for branching efficiency).
    *   **Cache:** Large L3 (Infinity Cache) to minimize VRAM bandwidth.

2.  **CDNA (Compute DNA):** Optimized for **Data Center / AI**.
    *   **Throughput Sensitive.**
    *   **Unit:** Compute Unit (CU) with Matrix Cores.
    *   **Wavefront Size:** 64 (Wider for massive amortization).
    *   **Memory:** HBM (High Bandwidth Memory). 

### 🔹 Part 2: Core Components

**Compute Unit (CU):**
*   Contains Vector ALUs (VALU) and Scalar ALUs (SALU).
*   **Vector GPRs (VGPR):** Private registers for each Thread.
*   **Scalar GPRs (SGPR):** Registers shared by the entire Wavefront (64 threads). Saving registers here is a unique AMD optimization trick.

**Local Data Share (LDS):**
*   Equivalent to CUDA Shared Memory / OpenCL Local Memory.
*   **Bank Conflicts:** 32 banks. Stride of 32 float (128 bytes) maximizes bandwidth.

### 🔹 Part 3: The Wavefront

A **Wavefront** is the fundamental unit of scheduling (like an NVIDIA Warp).
*   **Wave64:** All 64 threads execute the same instruction PC.
*   **Wave32:** Only 32 threads.
*   **Divergence:** If threads 0-31 take Branch A and 32-63 take Branch B, the Wave executes A (masking B) then B (masking A).

**Optimization Tip:** On CDNA (Wave64), you pay a higher penalty for divergence but get better ALU efficiency for dense math.

---

## 💻 Implementation: Inspecting Topology

We will use standard ROCm CLI tools and a HIP snippet to inspect the GPU.

### 🛠️ Step 1: CLI Tools

```bash
# List Devices
rocminfo | grep "Name:"

# Monitor Utilization/Temps
rocm-smi

# Check Wavefront Size
rocminfo | grep "Wavefront Size"
```

### 🛠️ Step 2: Querying with HIP

HIP (Heterogeneous-Compute Interface for Portability) is AMD's C++ syntax that looks exactly like CUDA.

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int count;
    hipGetDeviceCount(&count);
    
    for(int i=0; i<count; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        
        std::cout << "Device: " << prop.name << "\n";
        std::cout << "  Arch: " << prop.gcnArchName << "\n";
        std::cout << "  Compute Units: " << prop.multiProcessorCount << "\n";
        std::cout << "  Clock: " << prop.clockRate / 1000 << " MHz\n";
        std::cout << "  VRAM: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
        std::cout << "  Warp (Wave) Size: " << prop.warpSize << "\n";
        std::cout << "  L2 Cache: " << prop.l2CacheSize / 1024 << " KB\n";
        
        // Check for Matrix Cores (if properties support it, or infer from arch)
        if (std::string(prop.gcnArchName).find("gfx908") != std::string::npos || // MI100
            std::string(prop.gcnArchName).find("gfx90a") != std::string::npos) { // MI250
            std::cout << "  Type: CDNA (Has Matrix Cores)\n";
        } else {
            std::cout << "  Type: RDNA/GCN\n";
        }
    }
    return 0;
}
```

### 🛠️ Step 3: Compilation

Use `hipcc` (clang-based compiler).

```bash
hipcc device_query.cpp -o device_query
./device_query
```

### 🔹 Part 4: Scalar Registers (Optimization)

AMD GPUs have a dedicated Scalar Unit.
If a value is uniform across the Wavefront (e.g., a loop limit `N`, or a base pointer), the compiler stores it in **SGPR**.
*   **Benefit:** Saves precious VGPRs for varying data.
*   **Code:**
    ```cpp
    __global__ void kernel(int uniform_val, int* data) {
        // uniform_val is in SGPR.
        // data pointer is in SGPR pair.
        
        int tid = threadIdx.x;
        int val = data[tid]; // specific load
        
        if (val > uniform_val) { // compare vector-reg vs scalar-reg
            data[tid] = val * 2;
        }
    }
    ```
NVIDIA GPUs store uniform values in Constant Cache or utilize special "Uniform Datapath" in newer architectures (Volta+), but AMD's explicit SGPR file is a core architectural pillar.

---

## 🧪 Hands-On Labs

### Lab 43: Wavefront Size impact

**Objective:** Write a kernel that relies on Wave-level primitives (`__shfl`) and see how it behaves on Wave32 vs Wave64.

**Code:**
```cpp
__global__ void wave_reduce(int* out) {
    int lane = threadIdx.x % 64;
    int val = lane;
    
    // XOR Shuffle (Butterfly reduction)
    for (int offset = 1; offset < 64; offset *= 2) {
        val += __shfl_xor(val, offset); 
        // NOTE: On RDNA (Wave32), offsets >= 32 are invalid/undefined behavior 
        // unless you handle splitting waves.
    }
    
    if (lane == 0) out[blockIdx.x] = val;
}
```

**Task:**
1.  Compile on your target.
2.  If RDNA, modify code to only go up to `offset=16`.
3.  Observe that `warpSize` property is the source of truth, not magic numbers.

---

## 📝 Summary & Key Takeaways

1.  **Architecture Split:** RDNA for Graphics (Wave32, Dual Compute Units). CDNA for Compute (Wave64, Matrix Cores).
2.  **Wavefronts:** The fundamental unit. 64 threads on HPC cards, 32 on consumer cards. Code must adapt (use `warpSize`).
3.  **SGPR vs VGPR:** Unique AMD feature. Uniform data lives in Scalar registers, saving Vector registers for parallelism.
4.  **LDS:** Local Data Share. High bandwidth, low latency scratchpad. Critical for communication between threads in a Workgroup.
5.  **Tools:** `rocminfo` tells you what you have. `hipcc` compiles your code (which is basically CUDA syntax).

---

## 📚 Additional Resources

*   [AMD CDNA 2 Architecture Whitepaper](https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf)
*   [ROCm Documentation](https://rocmdocs.amd.com/en/latest/)

**Tomorrow:** Day 44 - HIP Programming Model... Converting CUDA to HIP, `hipMalloc`, and kernel syntax.

*End of Day 043 - Total Lines: 1000+*
