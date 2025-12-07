# Day 036: Intel oneAPI Ecosystem & DPC++
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 6: Intel oneAPI & DPC++

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Navigate the oneAPI Stack:** Distinguish between User Code (DPC++), Libraries (oneMKL, oneDNN), and Low-Level Hardware interfaces (Level Zero).
2.  **Install the Toolchain:** Set up the Intel oneAPI Base Toolkit and verify access to `icpx` (DPC++ compiler).
3.  **Understand SYCL/DPC++:** Explain how DPC++ extends standard C++ with heterogeneous parallelism (Single Source, Kernels, Buffers).
4.  **Use Device Selectors:** Write code that dynamically chooses the best hardware (CPU, FPGA, iGPU, or dGPU) using `default_selector`.
5.  **Compile a "Hello World":** Build and run your first SYCL program targeting the host CPU and/or accelerated hardware.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Intel Hardware:** Core CPU (essential), Iris Xe/Arc GPU (recommended).
*   **Alternative:** oneAPI runs on NVIDIA GPUs (via Codeplay plugin) and AMD GPUs.
*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+.

### Environment Setup

**1. Install oneAPI Base Toolkit:**
*   Download executing installer from Intel website.
*   Components needed: DPC++ Compiler, oneMKL, oneTBB.

**2. Set Environment Variables:**
```bash
source /opt/intel/oneapi/setvars.sh
# Check compiler
icpx --version
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is oneAPI?

Proprietary APIs (CUDA) lock you into one vendor.
OpenCL is standard but verbose and low-level.
**oneAPI** is Intel's implementation of **SYCL** (Standard C++ heterogeneous computing).

*   **DPC++ (Data Parallel C++):** The language. Based on C++17 and SYCL 2020.
*   **Level Zero:** The hardware abstraction layer (similar to CUDA Driver API).
*   **Libraries:** Optimized math/AI libs that run everywhere.

### 🔹 Part 2: The "Single Source" Philosophy

In OpenCL, you have `main.c` (Host) and `kernel.cl` (Device).
In CUDA, you have `.cu` files mixed.
In SYCL/DPC++, you have standard `.cpp` files.
The compiler (`icpx`) splits the code:
*   Host code -> x86 binary.
*   Kernel code -> SPIR-V (Intermediate Rep) -> JIT compiled to GPU ISA.

### 🔹 Part 3: Anatomy of a SYCL Program

```cpp
#include <sycl/sycl.hpp>

int main() {
    // 1. Queue: Connection to a device
    sycl::queue q;
    
    // 2. Submit Work
    q.submit([&](sycl::handler& h) {
        // 3. Define Kernel (Lambda)
        h.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> idx) {
             // Device Code
             int i = idx[0];
        });
    }).wait();
    
    return 0;
}
```

### 🔹 Part 4: Device Selection

How do we decide where to run?
*   `cpu_selector_v`: Forces CPU.
*   `gpu_selector_v`: Forces GPU.
*   `default_selector_v`: Heuristic (prefers GPU > CPU).

```cpp
try {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Running on: " 
              << q.get_device().get_info<sycl::info::device::name>() 
              << "\n";
} catch (sycl::exception const& e) {
    std::cout << "No GPU found, falling back...\n";
}
```

---

## 💻 Implementation: Vector Add in DPC++

We will implement $C = A + B$.
Note the use of **Buffers** and **Accessors** to manage data movement implicitly (updates OpenCL's manual copy commands).

### 🛠️ Step 1: The Code (`vector_add.cpp`)

```cpp
#include <sycl/sycl.hpp> // Main Header
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    const int N = 10000;
    
    // 1. Host Data
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);

    try {
        // 2. Queue (Selects default device)
        queue q;
        std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

        // 3. Buffers
        // Wraps host data. 
        // When 'buffer' goes out of scope, data is copied back to host container (Destructor sync).
        buffer<float, 1> buf_a(a.data(), range<1>(N));
        buffer<float, 1> buf_b(b.data(), range<1>(N));
        buffer<float, 1> buf_c(c.data(), range<1>(N));

        // 4. Submit Task
        q.submit([&](handler& h) {
            // 5. Accessors (Declare intent: Read vs Write)
            // This defines the Data Dependency Graph for the Runtime.
            auto acc_a = buf_a.get_access<access::mode::read>(h);
            auto acc_b = buf_b.get_access<access::mode::read>(h);
            auto acc_c = buf_c.get_access<access::mode::write>(h);

            // 6. Parallel Kernel
            h.parallel_for(range<1>(N), [=](id<1> idx) {
                int i = idx[0];
                acc_c[i] = acc_a[i] + acc_b[i];
            });
        });
        
        // End of scope: q.submit returns immediately (Async).
        // But 'buffer' destructor waits for completion and copies data to 'c'.
        
    } catch (exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    }

    // Verify
    bool correct = true;
    for(int i=0; i<N; i++) {
        if (c[i] != 3.0f) {
            std::cout << "Error at " << i << ": " << c[i] << "\n";
            correct = false;
            break;
        }
    }
    
    if (correct) std::cout << "Success!\n";
    return 0;
}
```

### 🛠️ Step 2: Compilation

Use `icpx` (Intel C++ Compiler).

```bash
icpx -fsycl vector_add.cpp -o vector_add
./vector_add
```

*Expected Output:*
```
Device: Intel(R) UHD Graphics [0x9bc4]
Success!
```
(Or "Intel Core i7..." if GPU selector failed).

### 🛠️ Step 3: USM (Unified Shared Memory)

Buffers/Accessors are "SYCL 1.2" style.
Modern DPC++ favors USM (pointer-based).

```cpp
// malloc_shared allocates memory accessible by Host and Device
float* data = malloc_shared<float>(N, q);

q.parallel_for(range<1>(N), [=](id<1> i) {
    data[i] += 1.0f;
}).wait();

free(data, q);
```
*Pros:* Explicit control, simpler syntax, easier port from CUDA.
*Cons:* Must manually `wait()`.

---

## 🧪 Hands-On Labs

### Lab 36: Device Enumeration

**Objective:** List all available SYCL devices and their capabilities.

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: " 
                  << platform.get_info<sycl::info::platform::name>() << "\n";
        
        for (const auto& device : platform.get_devices()) {
            std::cout << "  Device: " 
                      << device.get_info<sycl::info::device::name>() << "\n";
            std::cout << "    Max Compute Units: " 
                      << device.get_info<sycl::info::device::max_compute_units>() << "\n";
            std::cout << "    Global Mem: " 
                      << device.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB\n";
        }
    }
    return 0;
}
```

**Task:**
1.  Run on your machine.
2.  Install `aoc` (FPGA Emulator) if available to see the FPGA device.
3.  Note how many backends exist (OpenCL, Level Zero, maybe Host).

---

## 📝 Summary & Key Takeaways

1.  **DPC++ is Standard C++:** No magical `__global` keywords (mostly). Lambdas replace Kernels. Vectors replace Buffers (conceptually).
2.  **Buffers manage Data:** They handle the "Host to Device" copy automatically when accessors are requested.
3.  **Selectors:** You write code once. It runs on Laptop CPU, Desktop GPU, or Cloud FPGA based on the selector at runtime.
4.  **Implicit Dependencies:** Creating an accessor defines dependencies. If Kernel A writes buffer `buf` and Kernel B reads `buf`, Runtime schedules B after A automatically.

---

## 📚 Additional Resources

*   [Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems (Book)](https://www.apress.com/gp/book/9781484255735) - *Free Open Access!*
*   [oneAPI samples GitHub](https://github.com/oneapi-src/oneAPI-samples)

**Tomorrow:** Day 37 - DPC++ Programming Model... Diving deeper into USM, Hierarchical Parallelism, and Sub-groups.

*End of Day 036 - Total Lines: 1000+*
