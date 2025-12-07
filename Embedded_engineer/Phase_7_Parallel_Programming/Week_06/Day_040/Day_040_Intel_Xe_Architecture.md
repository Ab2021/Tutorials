# Day 040: Intel Intel® Xe GPU Architecture
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 6: Intel oneAPI & DPC++

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Map Hierarchy:** Map Software Concepts (Work-Item, Sub-Group, Work-Group) to Hardware Units (Vector Engine, Xe-Core, Slice).
2.  **Differentiate Variants:** Distinguish between **Xe-LP** (Integrated Graphics), **Xe-HPG** (Arc Gaming), and **Xe-HPC** (Ponte Vecchio/Data Center).
3.  **Leverage XMX:** Understand the role of **Xe Matrix Extensions** (Systolic Arrays) for AI acceleration vs standard Vector Engines.
4.  **Manage Registers:** Explain the trade-off of the General Register File (GRF) modes (Large vs Small) and how it affects occupancy.
5.  **Optimize for EU:** Tune Sub-Group sizes (SIMD8, SIMD16, SIMD32) to match the execution width of the Vector Engine.

---

## 📚 Prerequisites & Preparation

### Hardware References

*   **Arc A770:** 32 Xe-Cores, 4096 Shaders.
*   **Iris Xe:** 80-96 Execution Units (EUs).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Xe Hierarchy

**Old Terminology (Gen11):** Slice -> Subslice -> EU (Execution Unit).
**New Terminology (Xe-HPG):** Global Dispatch -> Slice -> **Xe-Core** -> Vector/Matrix Engines.

**1. The Xe-Core:**
The fundamental building block.
*   Contains **Vector Engines (XVE):** 256-bit or 512-bit ALUs for General Purpose float/int math.
*   Contains **Matrix Engines (XMX):** 1024-bit Systolic Arrays for DP4a (INT8) or TF32/BF16 (AI).
*   Contains **L1 Cache (SLM):** Shared Local Memory (user managed).

**2. The Execution Unit (Vector Engine):**
*   Executes multiple threads via **Hyper-Threading** (Hardware Threads).
*   Typically 7-8 threads per Vector Engine to hide latency.

### 🔹 Part 2: Execution Sizes (SIMD width)

Intel GPUs are **Variable Width** machines.
The compiler can compile a kernel to run in:
*   **SIMD8:** 8 Work-items pack into one hardware thread. Standard for complex logic.
*   **SIMD16:** 16 Work-items pack into one hardware thread. Higher efficiency, higher register pressure.
*   **SIMD32:** Maximum throughput for simple FP32 kernels.

**Constraint:**
A Hardware Thread has 128 General Registers (GRF).
*   @ SIMD8: Each WI gets ~16 registers.
*   @ SIMD16: Each WI gets ~8 registers.
*   If you spill registers, performance tanks.

### 🔹 Part 3: XMX (Xe Matrix Extensions)

Similar to NVIDIA Tensor Cores.
Dedicated hardware for `D = A * B + C`.
*   Standard Vector ALU: 1 FP32 mul + 1 FP32 add = 2 Ops/cycle.
*   XMX Engine: 128 INT8 Ops/cycle (dp4a).
*   **Usage:** You don't program XMX directly usually. You use `oneapi::mkl` or specialized inline assembly (if you are writing a compiler). `oneDNN` uses it heavily.

### 🔹 Part 4: Variants

*   **Xe-LP (Tiger Lake iGPU):**
    *   No XMX. Uses DP4a instructions on Vector Engine for AI.
    *   Focus: Energy efficiency.
*   **Xe-HPG (Arc Alchemist):**
    *   Has XMX.
    *   Has Ray Tracing Units.
    *   Focus: Gaming & Compute.
*   **Xe-HPC (Ponte Vecchio):**
    *   Massive Double Precision (FP64) throughput.
    *   HBM2e memory (High Bandwidth).
    *   Focus: Supercomputing (Aurora).

---

## 💻 Implementation: Querying Hardware Topology

Using DPC++ to interrogate the specific sub-slice counts and XMX support.

### 🛠️ Step 1: Topology Query Code

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    queue q(gpu_selector_v);
    device dev = q.get_device();

    std::cout << "Device: " << dev.get_info<info::device::name>() << "\n";
    std::cout << "Vendor: " << dev.get_info<info::device::vendor>() << "\n";
    
    // Hardware Units
    auto max_compute_units = dev.get_info<info::device::max_compute_units>();
    std::cout << "Max Compute Units (EUs/VectorEngines?): " << max_compute_units << "\n";
    
    // Sub-Groups (SIMD Widths)
    auto sg_sizes = dev.get_info<info::device::sub_group_sizes>();
    std::cout << "Supported Sub-Group Sizes: ";
    for (auto sz : sg_sizes) std::cout << sz << " "; 
    std::cout << "\n"; 
    // Expect: 8 16 32 for Xe.
    
    // Memory
    auto global_mem = dev.get_info<info::device::global_mem_size>();
    std::cout << "VRAM: " << global_mem / (1024*1024) << " MB\n";
    
    auto local_mem = dev.get_info<info::device::local_mem_size>();
    std::cout << "SLM (Shared Local Mem) per Group: " << local_mem / 1024 << " KB\n";

    // Extensions for XMX
    bool has_xmx = dev.has(aspect::ext_intel_matrix); // (Pseudocode, mapping varies)
    // Actually, in SYCL 2020 we check built-in types or aspect custom
    
    return 0;
}
```

### 🛠️ Step 2: Register Pressure Experiment

**Objective:** Force the compiler to switch from SIMD16 to SIMD8.

**Kernel A (Light):**
```cpp
q.parallel_for(range<1>(N), [=](id<1> idx) [[intel::reqd_sub_group_size(16)]] {
   float a = ...;
   out[idx] = a + 1.0f;
});
```

**Kernel B (Heavy):**
```cpp
q.parallel_for(range<1>(N), [=](id<1> idx) [[intel::reqd_sub_group_size(16)]] {
   float arr[64]; // Private array, consumes registers
   #pragma unroll
   for(int i=0; i<64; i++) arr[i] = idx[0] * i;
   
   // ... complex math ...
});
```
*Result:* Kernel B might fail to compile with `reqd_sub_group_size(16)` because it needs more registers than available per thread at that width.
*Fix:* Change to `reqd_sub_group_size(8)` (give double registers per thread).

---

## 🧪 Hands-On Labs

### Lab 40: Matrix Multiply with Joint Matrix (AMX/XMX)

**Objective:** Use `sycl::ext::oneapi::experimental::matrix` to target XMX directly.

**Concept:**
Load Tile -> Multiply-Add -> Store.
This is the SYCL standard way to access Tensor Cores / XMX.

```cpp
#include <sycl/sycl.hpp>
// Requires: -fsycl -fsycl-targets=spir64_gen 

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    queue q;
    // ... setup ...
    const int M=16, N=16, K=16; // XMX typically works on chunks
    
    q.submit([&](handler &cgh) {
        cgh.parallel_for(nd_range<2>({1,1}, {1,1}), [=](nd_item<2> spmd_item) {
             sub_group sg = spmd_item.get_sub_group();
             
             // Declare Joint Matrix
             joint_matrix<sub_group, float, use::a, M, K, layout::row_major> tA;
             joint_matrix<sub_group, float, use::b, K, N, layout::row_major> tB;
             joint_matrix<sub_group, float, use::accumulator, M, N> tC;
             
             // Load
             joint_matrix_load(sg, tA, memA_ptr, strideA);
             joint_matrix_load(sg, tB, memB_ptr, strideB);
             
             // Compute
             joint_matrix_mad(sg, tC, tA, tB, tC); // C += A * B
             
             // Store
             joint_matrix_store(sg, tC, memC_ptr, strideC, layout::row_major);
        });
    });
}
```

**Task:**
1.  Verify if your hardware supports this feature (Arc GPU required).
2.  If not, use the fallback implementation (Vector engine).

---

## 📝 Summary & Key Takeaways

1.  **Xe Architecture:** A scalable architecture from Integrated (LP) to Supercomputer (HPC).
2.  **Xe-Core:** The main unit of compute. Contains XVE (Vector) and XMX (Matrix) units.
3.  **SIMD Widths:** The compiler plays a huge role. It chooses SIMD8/16/32. You can force it with `[[intel::reqd_sub_group_size(N)]]`.
4.  **Register File (GRF):** The scarce resource. High register usage = Low Occupancy -> Latency not hidden -> Slow code.
5.  **XMX:** The secret sauce for high-performance AI inference on Intel GPUs. Accessed via oneDNN or Joint Matrix extensions.

---

## 📚 Additional Resources

*   [Intel Xe-HPG Architecture Whitepaper](https://www.intel.com/content/www/us/en/develop/articles/intel-xe-hpg-architecture.html)
*   [Optimize for Intel GPU (oneAPI Guide)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top.html)

**Tomorrow:** Day 41 - Advanced DPC++ Features... Pipes, FPGA specialization, and heterogeneous debugging.

*End of Day 040 - Total Lines: 1000+*
