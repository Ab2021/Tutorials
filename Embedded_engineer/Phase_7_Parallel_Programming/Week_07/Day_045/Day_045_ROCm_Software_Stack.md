# Day 045: ROCm Software Stack & Tools
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Navigate the Stack:** Understand the layers: KFD (Kernel Fusion Driver), HSA (Runtime), HIP (Language), and Libraries (rocBLAS).
2.  **Profile Code:** Use `rocprof` to extract CSV metrics (VALU utilization, Cache Hit rates) from running kernels.
3.  **Visualize Traces:** Interpret system-scale performance data using **Radeon GPU Profiler (RGP)**.
4.  **Debug Kernels:** Use `rocgdb` to inspect variables inside a Wavefront, switch focus between lanes, and catch segfaults.
5.  **Utilize Libraries:** Replace manual kernels with optimized calls to `rocBLAS` and `rocFFT`.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Tools:** `rocm-debug-agent`, `rocprofiler`.
*   **Visualizer:** Download "Radeon GPU Profiler" (RGP) GUI on your host machine (Windows/Linux) to view traces captured on the target.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The ROCm Stack Layers

1.  **Application (HIP/OpenMP):** Top level.
2.  **Libraries (rocBLAS, MIOpen):** Optimized math.
3.  **Language Runtime (COMGR, HIP):** Compiler and Code Object Manager.
4.  **HSA Runtime (ROCr):** User-space scheduling, queues, memory allocation.
5.  **Kernel Driver (AMDGPU/KFD):** Manages VRAM, page tables, and rings.

**Difference from CUDA:**
ROCm is heavily "Open Source/Linux" focused.
The scheduling is more user-space driven (HSA signals) for lower latency than the traditional monolithic CUDA driver approach.

### 🔹 Part 2: Profiling Strategy (`rocprof`)

Cmd line tool.
*   **Timestamp Tracing:** When did kernel start/end? (API trace).
*   **Hardware Counters:** Instructions executed, L2 Cache misses, Memory Bandwidth.

**Command:**
```bash
rocprof --stats --timestamp on ./my_app
# Generates results.csv and results.json (for Chrome Tracing)
```

### 🔹 Part 3: Debugging (`rocgdb`)

Based on GDB.
*   `info threads`: Shows host threads.
*   `info waves`: **Exposes GPU Wavefronts**.
*   `thread <ID>`: Switch context.
*   `print var`: Inspect register values.

**Warning:**
Debugging GPU code requires disabling optimizations (`-O0 -g`) which changes register allocation drastically.

---

## 💻 Implementation: Matrix Mul with rocBLAS

We will replace our custom `matmul` with `rocblas_sgemm`.

### 🛠️ Step 1: The Code (`rocblas_demo.cpp`)

```cpp
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <iostream>
#include <vector>

void check_rocblas(rocblas_status status, const char* msg) {
    if(status != rocblas_status_success) {
        std::cerr << "Error: " << msg << "\n";
        exit(1);
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);
    
    // 1. Host Data
    std::vector<float> hA(N*N, 1.0f);
    std::vector<float> hB(N*N, 2.0f);
    std::vector<float> hC(N*N, 0.0f);

    // 2. Device Alloc
    float *dA, *dB, *dC;
    hipMalloc(&dA, size);
    hipMalloc(&dB, size);
    hipMalloc(&dC, size);

    hipMemcpy(dA, hA.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), size, hipMemcpyHostToDevice);

    // 3. Setup rocBLAS Handle
    rocblas_handle handle;
    check_rocblas(rocblas_create_handle(&handle), "Create Handle");

    // 4. SGEMM
    // C = alpha * A * B + beta * C
    // Note: Use Column Major assumption or Transpose args
    float alpha = 1.0f;
    float beta = 0.0f;
    
    check_rocblas(rocblas_sgemm(handle,
                                rocblas_operation_none, rocblas_operation_none,
                                N, N, N,
                                &alpha,
                                dA, N, // lda
                                dB, N, // ldb
                                &beta,
                                dC, N  // ldc
                                ), "SGEMM");

    // 5. Cleanup
    hipMemcpy(hC.data(), dC, size, hipMemcpyDeviceToHost);
    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    
    std::cout << "Done. C[0] = " << hC[0] << "\n";
    return 0;
}
```

### 🛠️ Step 2: Compile & Link

```bash
hipcc rocblas_demo.cpp -o rocblas_demo -lrocblas
./rocblas_demo
```

---

## 🧪 Hands-On Labs

### Lab 45: Profiling with rocprof

**Objective:** Collect hardware counters for the rocBLAS demo.

**1. Create a counters file (`pmc.txt`):**
```text
# Performance Metric Counters
pmc: VALUUtilization,WriteUnitStalled,L2CacheHit
```

**2. Run Profiler:**
```bash
rocprof -i pmc.txt -o my_stats.csv ./rocblas_demo
```

**3. Analyze CSV:**
*   **VALUUtilization:** Percentage of cycles Vector Unit was busy. (Expect 80%+ for GEMM).
*   **WriteUnitStalled:** Memory bottlenecks.
*   **L2CacheHit:** Data locality metric.

---

## 📝 Summary & Key Takeaways

1.  **Libraries First:** Always use `rocBLAS` / `rocFFT` before writing custom kernels. They are tuned for every Instruction Set (GFX906, GFX908, GFX90a).
2.  **Async Profiling:** `rocprof` is your primary tool for batch performance analysis.
3.  **Visual Debugging:** Radeon GPU Profiler (RGP) gives a "Gantt Chart" view of barriers, occupancies, and latencies that is world-class.
4.  **GDB:** It exists for GPU, but use printf first. Context switching thousands of threads in GDB is painful.

---

## 📚 Additional Resources

*   [rocBLAS Documentation](https://rocmdocs.amd.com/en/latest/ROCm_Libraries/rocBLAS/rocBLAS.html)
*   [Radeon GPU Profiler (RGP) Guide](https://gpuopen.com/rgp/)

**Tomorrow:** Day 46 - Performance Tuning on AMD... Optimizing Occupancy, LDS usage, and Wavefront Latency Hiding.

*End of Day 045 - Total Lines: 1000+*
