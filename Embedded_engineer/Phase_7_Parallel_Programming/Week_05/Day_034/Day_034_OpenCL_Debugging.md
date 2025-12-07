# Day 034: Debugging OpenCL & Error Handling
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Print from GPU:** Use the `cl_intel_printf` or standard OpenCL 1.2 `printf` extension to output values directly from inside a kernel.
2.  **Catch Async Errors:** Register a **Context Callback** to catch asynchronous errors (like `CL_OUT_OF_RESOURCES`) that occur after enqueue returns.
3.  **Decode Error Codes:** Map elusive integers like `-5` (`CL_OUT_OF_RESOURCES`) or `-54` (`CL_INVALID_WORK_GROUP_SIZE`) to root causes.
4.  **Debug on CPU:** Force execution on the CPU device (`CL_DEVICE_TYPE_CPU`) to use standard GDB/LLDB for stepping through kernels.
5.  **Validate Inputs:** Write defensive host code that checks buffer sizes and `nullptr` handles before launching.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Debugger:** GDB (Linux) or Visual Studio (Windows).
*   **Driver Support:** Most modern drivers support `printf`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Asynchronous Error Problem

`clEnqueueNDRangeKernel` returns `CL_SUCCESS`.
But the GPU crashes 50ms later.
Where is the error code?
*   It might appear in the *next* API call (e.g., `clFinish`).
*   It might trigger the **Context Notification Callback** (if registered).
*   It might just freeze the screen (TDR - Timeout Detection and Recovery).

**Lesson:** Always check return codes of `clFinish` and register a callback!

### 🔹 Part 2: `printf` inside Kernels

Standard since OpenCL 1.2.
Output buffer is circular and limited (e.g., 1MB). If you print from 1 million threads, you flush the buffer and lose data.

**Usage:**
```c
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
__kernel void debug_k(__global int* data) {
    if (get_global_id(0) == 0) {
        printf("Kernel start: Data[0] = %d\n", data[0]);
    }
}
```
**Constraint:** Format specifiers are limited (%d, %f, %v4f). No pointers (%p is risky).

### 🔹 Part 3: CPU Debugging

The "Secret Weapon" of OpenCL devs.
1.  Initialize context with `CL_DEVICE_TYPE_CPU`.
2.  Compile with `-g` (host) and `-g -s` (OpenCL build options).
3.  Run `gdb ./host_app`.
4.  Break inside the kernel?
    *   Intel OpenCL SDK: Yes.
    *   POCL (Portable OpenCL): Yes.
    *   NVIDIA: No (Kernels are compiled to PTX).

---

## 💻 Implementation: Robust Host Boilerplate

We will creating a "Safe OpenCL" wrapper that catches errors.

### 🛠️ Step 1: Error Lookup Table

```c
const char* get_error_string(cl_int err) {
    switch (err) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        // ... (Add all 60 codes) ...
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        default: return "Unknown OpenCL Error";
    }
}

void check_err(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during %s: %s (%d)\n", 
                operation, get_error_string(err), err);
        exit(1);
    }
}
```

### 🛠️ Step 2: Context Callback

```c
void CL_CALLBACK pfn_notify(const char *errinfo, 
                            const void *private_info, 
                            size_t cb, 
                            void *user_data) {
    fprintf(stderr, "[OpenCL Error Notification]: %s\n", errinfo);
}

// In main:
cl_context context = clCreateContext(NULL, 1, &dev, pfn_notify, NULL, &err);
```

### 🛠️ Step 3: Debugging a Kernel Crash

**The Bad Kernel:** Access out of bounds.

```c
__kernel void crash_me(__global int* arr, int N) {
    int i = get_global_id(0);
    // Bug: Accessing i+1 when i == N-1
    arr[i] = arr[i+1]; 
}
```

**Host Code:**
```c
int main() {
    // ... Setup ...
    
    // 1. Launch Bad Kernel
    err = clEnqueueNDRangeKernel(...); // Returns SUCCESS usually (Async)
    
    // 2. Wait
    err = clFinish(queue); 
    
    if (err == CL_OUT_OF_RESOURCES) {
        printf("GPU crashed or ran out of memory!\n");
    }
    
    // Add printf to kernel to trace
}
```

**Fixing with `printf`:**
```c
__kernel void crash_me(__global int* arr, int N) {
    int i = get_global_id(0);
    if (i >= N-1) {
        printf("Error: Thread %d accessing %d which is >= %d\n", i, i+1, N);
        // return; // Fix logic
    }
    arr[i] = arr[i+1]; 
}
```

---

## 🧪 Hands-On Labs

### Lab 34: The CPU Fallback

**Objective:** Write a host program that tries GPU first, fails (simulated), and falls back to CPU.

**Logic:**
```c
// Try GPU
err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ...);
if (err != CL_SUCCESS) {
    printf("No GPU found. Trying CPU...\n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, ...);
    if (err != CL_SUCCESS) {
        printf("No OpenCL devices found.\n");
        exit(1);
    }
}
```

**Task:**
1.  Implement the fallback logic.
2.  Force CPU usage.
3.  Add `int *ptr = NULL; *ptr = 0;` inside your kernel (Segfault).
4.  Run with GDB. Does it catch the segfault line inside the kernel? (Requires OpenCL CPU driver).

---

## 📝 Summary & Key Takeaways

1.  **Return Codes:** Check **every** OpenCL call. A missed `-38` (Invalid Mem) early on can cause a crash 100 lines later.
2.  **Callbacks:** `pfn_notify` is your hotline to the driver. It usually prints descriptive strings ("NVIDIA: Mem limit exceeded").
3.  **Printf:** Use `printf` sparingly. It serializes threads and fills buffers. Remove in production.
4.  **Boundary Checks:** GPU MMUs are permissive or brutal. Sometimes they ignore OOB, sometimes they TDR. Always bounds check: `if (gid >= N) return;`.
5.  **Build Logs:** Compiler syntax errors only appear if you query `CL_PROGRAM_BUILD_LOG`.

---

## 📚 Additional Resources

*   [Intel OpenCL Debugging Guide](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/debug-kernel.html)
*   [Common OpenCL Error Codes](https://streamhpc.com/blog/2013-04-28/opencl-error-codes/)

**Tomorrow:** Day 35 - Week 5 Review & Project... Building a Heterogeneous Image Filter Pipeline (CPU loads, GPU filters).

*End of Day 034 - Total Lines: 1000+*
