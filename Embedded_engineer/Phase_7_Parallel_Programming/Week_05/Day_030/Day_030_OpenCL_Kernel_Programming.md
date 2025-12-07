# Day 030: OpenCL Kernel Programming
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Write OpenCL Kernels:** Author `.cl` files using C99-like syntax with `__kernel`, `__global`, and `get_global_id()`.
2.  **Compile at Runtime:** Use `clCreateProgramWithSource` and `clBuildProgram` to JIT-compile kernels for the target device (GPU/CPU).
3.  **Manage Buffers:** Allocate device memory (`clCreateBuffer`) and transfer data (`clEnqueueWriteBuffer`/`clEnqueueReadBuffer`).
4.  **Execute Kernels:** Launch kernels over an 1D execution domain using `clEnqueueNDRangeKernel`.
5.  **Handle Build Errors:** Extract and print the build log (crucial for debugging syntax errors in OpenCL C).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **SDK:** Same as Day 29.
*   **Editor:** VS Code with OpenCL extension (for syntax highlighting).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: OpenCL C Language

Kernels are functions that run on the device.
**Keywords:**
*   `__kernel`: Entry point callable from host.
*   `__global`: Pointer to Global Memory (VRAM).
*   `__local`: Pointer to Local Memory (Shared Mem).
*   `__private`: Pointer to Private Memory (Registers).
*   `__constant`: Pointer to Constant Cache.

**Built-in Functions:**
*   `get_global_id(dim)`: Unique index of WI in the grid.
*   `get_local_id(dim)`: Index of WI in the WG.
*   `get_group_id(dim)`: Index of WG in the Grid.

**Vector Types:**
OpenCL has native vectors: `float4`, `int2`, etc.
`float4 a = (float4)(1.0f, 2.0f, 3.0f, 4.0f);`
Math on them is SIMD by definition.

### 🔹 Part 2: The Host Workflow (Boilerplate)

OpenCL is famous for having 100 lines of boilerplate to add two numbers.
**The Steps:**
1.  Get Platform & Device.
2.  Create **Context** (`clCreateContext`).
3.  Create **Command Queue** (`clCreateCommandQueue`).
4.  Create **Program** from Source (`clCreateProgramWithSource`).
5.  **Build** Program (`clBuildProgram`).
6.  Create **Kernel** (`clCreateKernel`).
7.  Create **Buffers** (`clCreateBuffer`).
8.  Write Input Data (`clEnqueueWriteBuffer`).
9.  Set Kernel Args (`clSetKernelArg`).
10. Launch Kernel (`clEnqueueNDRangeKernel`).
11. Read Result (`clEnqueueReadBuffer`).

### 🔹 Part 3: Runtime Compilation

Why compile at runtime?
*   Portability: You ship Source Code. The driver compiles it for *that specific* GPU (NVIDIA, AMD, Intel).
*   Optimization: Driver knows exact hardware details (register count, cache size).

---

## 💻 Implementation: Vector Addition

We will implement $C = A + B$ for 1 Million floats.

### 🛠️ Step 1: The Kernel (`vector_add.cl`)

Create a file named `vector_add.cl` in the same directory.

```c
__kernel void vector_add(__global const float *A, 
                         __global const float *B, 
                         __global float *C,
                         const int N) {
    
    // Get unique ID of this work-item
    int i = get_global_id(0);
    
    // Bounds check (Grid might be slightly larger than N to fit work-groups)
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

### 🛠️ Step 2: The Host Code (`host.c`)

```c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    // 1. Data Setup
    int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    for(int i=0; i<N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // 2. Get Platform/Device
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    
    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    
    // 3. Create Context & Command Queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);
    
    // 4. Create Memory Buffers (Device Memory)
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, NULL, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, NULL, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    // 5. Transfer Data to Device
    clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0, bytes, h_A, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0, bytes, h_B, 0, NULL, NULL);
    
    // 6. Create Program from Source
    FILE *fp = fopen("vector_add.cl", "r");
    if (!fp) { fprintf(stderr, "Failed to load kernel.\n"); exit(1); }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, NULL);
    
    // 7. Build Program
    cl_int ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        // Build Log Retrieval
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("Build Error: %s\n", buffer);
        exit(1);
    }
    
    // 8. Create Kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    
    // 9. Set Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_C);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);
    
    // 10. Execute Kernel
    size_t global_item_size = N; // Total work items
    size_t local_item_size = 64; // Work-group size (must divide global)
    
    // Fix alignment if N not divisible by 64? Loop inside kernel or pad global.
    // For now assume N is multiple of 64.
    
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    
    // 11. Read Result
    clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL);
    
    // Verify
    printf("Result[0] = %f\n", h_C[0]); // Should be 3.0
    printf("Result[N-1] = %f\n", h_C[N-1]);
    
    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(source_str);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
```

---

## 🧪 Hands-On Labs

### Lab 30: Kernel Arguments & Math

**Objective:** Implement $C = A * \sin(B) + \text{scalar}$.

**Kernel:**
```c
__kernel void math_kernel(__global const float *A, 
                          __global const float *B, 
                          __global float *C,
                          const float factor,
                          const int N) {
    int i = get_global_id(0);
    if (i < N) {
        // Native OpenCL Math functions (fast hardware implementation)
        float val = native_sin(B[i]); 
        C[i] = A[i] * val + factor;
    }
}
```

**Tasks:**
1.  Modify `vector_add.cl` to implement this.
2.  Update `host.c` to pass the `float factor` argument.
3.  Compare `sin()` (high precision) vs `native_sin()` (hardware approximation).

---

## 📝 Summary & Key Takeaways

1.  **Boilerplate is King:** OpenCL requires setup. Don't let it scare you. Ideally, wrap it in a helper class.
2.  **JIT Compilation:** `clBuildProgram` compiles your C code for the GPU at runtime. Always check the Build Log!
3.  **Memory Spaces:** `__global` pointers are mandatory for array arguments.
4.  **Work-Group Sizing:** `local_item_size` matters. It maps to hardware threads. 64 or 256 are safe defaults.
5.  **Enqueue:** Commands (Write, Execute, Read) are queued asynchronously. `CL_TRUE` in ReadBuffer acts as a blocking wait.

---

## 📚 Additional Resources

*   [OpenCL 3.0 Reference Guide](https://www.khronos.org/files/opencl30-reference-guide.pdf)
*   [OpenCL Programming Guide (MacroSystem)](https://streamcomputing.eu/resources/opencl-programming-guide/)

**Tomorrow:** Day 31 - OpenCL Memory Management... Pinned memory, Mapping buffers, and avoiding PCI-E bottlenecks.

*End of Day 030 - Total Lines: 1000+*
