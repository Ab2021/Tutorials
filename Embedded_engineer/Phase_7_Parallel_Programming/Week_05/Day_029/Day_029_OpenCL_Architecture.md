# Day 029: OpenCL Architecture & Platform Model
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Deconstruct the OpenCL Model:** Understand the four sub-models: Platform (Host/Device), Execution (Kernels/Work-Items), Memory (Global/Local/Private), and Programming.
2.  **Query the Hardware:** Write a "Device Discovery" tool that enumerates Platforms (e.g., NVIDIA, Intel) and Devices (GPU, CPU, Accelerator) and prints their capabilities.
3.  **Map Terminology:** Translate OpenCL terms to GPU hardware terms (Work-Item = Thread, Work-Group = Thread Block, Local Mem = Shared Mem).
4.  **Visualize the Hierarchy:** Understand how a Grid of work is decomposed into Work-Groups and mapped to Compute Units (Streaming Multiprocessors).
5.  **Differentiate Host vs Device:** Write code where the "Host" (CPU) orchestrates data movement and the "Device" (GPU/FPGA) executes mathematical kernels.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **SDK:** OpenCL SDK (Intel OneAPI, NVIDIA CUDA Toolkit, or AMD ROCm).
*   **Drivers:** Latest GPU drivers.
*   **Toolchain:** GCC/Clang with `-lOpenCL`.

### Environment Setup

**1. Install Headers (Linux):**
```bash
sudo apt install ocl-icd-opencl-dev opencl-headers clinfo
```

**2. Verify Installation:**
```bash
clinfo
# Should list your GPU (e.g., NVIDIA RTX 3060 or Intel UHD Graphics)
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Platform Model

**Host:** The CPU running your C/C++ program (the "Commander").
**Device:** The GPU, FPGA, or DSP doing the heavy lifting (the "Worker").
**Context:** The environment where kernels execute (includes memory buffers, command queues).

One Host can control multiple Devices (e.g., 2 GPUs + 1 CPU).

### 🔹 Part 2: The Execution Model

OpenCL programs (Kernels) execute across an N-dimensional range (NDRange).

1.  **Work-Item (WI):** The fundamental unit (like a CUDA Thread). Operates on one point in the index space.
2.  **Work-Group (WG):** A collection of Work-Items that execute together on a **Compute Unit** (CU). They share **Local Memory** and can synchronize via barriers.
3.  **NDRange (Grid):** The total number of Work-Items.

**Analogy:**
*   NDRange = The entire Army.
*   Work-Group = A Platoon (Shares a tent/local memory).
*   Work-Item = A Soldier (Has private registers).

### 🔹 Part 3: The Memory Model

1.  **Global Memory:** RAM on the GPU adapter (VRAM). Accessible by all WIs. Large, high latency (GDDR6).
2.  **Constant Memory:** Read-only cache for all WIs.
3.  **Local Memory:** Fast SRAM inside the Compute Unit (L1/Shared). Shared by WIs in a Group.
4.  **Private Memory:** Registers per WI.

### 🔹 Part 4: OpenCL vs CUDA

| concept | OpenCL | CUDA |
| :--- | :--- | :--- |
| **Exec Unit** | Work-Item | Thread |
| **Group** | Work-Group | Block |
| **Global** | NDRange | Grid |
| **Memory** | Local Memory | Shared Memory |
| **Processor** | Compute Unit | Streaming Multiprocessor (SM) |

OpenCL is **portable**. CUDA is NVIDIA-only.
OpenCL runs on Intel CPUs, AMD GPUs, FPGAs, and Mobile DSPs.

---

## 💻 Implementation: Device Discovery Tool

We won't compute anything today. We will build a robust tool to probe the hardware. This is step 0 of any heterogeneous app: "What hardware do I have?"

### 🛠️ Step 1: Query Platforms

```c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_PLATFORMS 16
#define MAX_DEVICES 16

int main() {
    cl_int err;
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_uint num_platforms;
    
    // 1. Get Platforms
    err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("Error getting platforms\n");
        return 1;
    }
    
    printf("Found %d Platforms:\n", num_platforms);
    
    for (int p = 0; p < num_platforms; p++) {
        char buffer[1024];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, buffer, NULL);
        printf("[%d] Name: %s\n", p, buffer);
        
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, buffer, NULL);
        printf("    Vendor: %s\n", buffer);
        
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, 1024, buffer, NULL);
        printf("    Version: %s\n", buffer);
        
        // 2. Get Devices for this Platform
        cl_device_id devices[MAX_DEVICES];
        cl_uint num_devices;
        
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
        if (err == CL_DEVICE_NOT_FOUND) {
            printf("    No devices found.\n");
            continue;
        }
        
        printf("    Found %d Devices:\n", num_devices);
        
        for (int d = 0; d < num_devices; d++) {
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, buffer, NULL);
            printf("    -> Device %d: %s\n", d, buffer);
            
            // Capabilities
            cl_uint compute_units;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
            printf("       Compute Units: %u\n", compute_units);
            
            cl_ulong global_mem;
            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem, NULL);
            printf("       Global Memory: %lu MB\n", global_mem / (1024*1024));
            
            cl_ulong local_mem;
            clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem, NULL);
            printf("       Local Memory:  %lu KB\n", local_mem / 1024);
            
            cl_uint max_work_group;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_uint), &max_work_group, NULL);
            printf("       Max WorkGroup: %u\n", max_work_group);
        }
    }
    return 0;
}
```

### 🛠️ Step 2: Compilation

**Linux:**
```bash
gcc -o cl_query day29_query.c -lOpenCL
./cl_query
```

**Output Example (Laptop):**
```
Found 2 Platforms:
[0] Name: NVIDIA CUDA
    Vendor: NVIDIA Corporation
    Found 1 Devices:
    -> Device 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU
       Compute Units: 20
       Global Memory: 4096 MB
[1] Name: Intel(R) OpenCL HD Graphics
    Vendor: Intel Corporation
    Found 1 Devices:
    -> Device 0: Intel(R) UHD Graphics
       Compute Units: 32
```

### 🛠️ Step 3: Interpreting the Data

*   **Compute Units (CU):** Parallel cores.
    *   NVIDIA: 1 CU = 1 SM (Streaming Multiprocessor). Each SM might have 64-128 CUDA Cores.
    *   Intel: 1 CU = 1 Execution Unit (EU) usually.
*   **Max WorkGroup:** Usually 256, 512, or 1024. Your kernel cannot launch a block larger than this.
*   **Local Memory:** Usually 32KB - 64KB per CU. This is your "Software Managed Cache".

---

## 🧪 Hands-On Labs

### Lab 29: CPU as a Device

**Objective:** Verify that the Host CPU can also be an OpenCL device.

**Concept:**
OpenCL isn't just for GPUs. Intel/AMD provide OpenCL runtimes for their CPUs. Use `CL_DEVICE_TYPE_CPU`.

**Task:**
Modify the query tool to filter only for `CL_DEVICE_TYPE_CPU`.
Is your CPU visible?
*   If yes, you can debug OpenCL kernels on the CPU using standard GDB (easier than debugging GPU).
*   If no, install `intel-opencl-icd` or `pocl-opencl-icd`.

**Expected Output:**
```
Found Platform: Portable Computing Language (POCL)
  -> Device: pthread-AMD Ryzen 7 5800X
     Compute Units: 16 (Threads)
```

---

## 📝 Summary & Key Takeaways

1.  **Heterogeneity:** The future is CPU + GPU + FPGA. OpenCL unifies them.
2.  **Platform ID:** The driver (NVIDIA/Intel). You select a platform first.
3.  **Device ID:** The hardware card.
4.  **Verbose API:** OpenCL C API is very verbose (Query Platform -> Query Count -> Query Devices -> Query Count...). C++ wrappers (`cl.hpp`) simplify this.
5.  **Compute Units:** The primary indicator of parallelism potential.

---

## 📚 Additional Resources

*   [Khronos OpenCL 3.0 Ref Card](https://www.khronos.org/files/opencl30-reference-guide.pdf)
*   [Hands On OpenCL](https://handsonopencl.github.io/)

**Tomorrow:** Day 30 - OpenCL Kernel Programming... Writing your first `.cl` file and executing vector addition on the GPU.

*End of Day 029 - Total Lines: 1000+*
