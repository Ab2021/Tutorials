# Day 032: OpenCL Synchronization & Events
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Master Command Queues:** Differentiate between In-Order (default) and Out-of-Order queues, and knowing when to use each.
2.  **Control Dependencies:** Use **Events** (`cl_event`) to build dependency graphs between commands (e.g., "Wait for Transfer A before starting Kernel B").
3.  **Overlap Compute & Transfer:** Hide PCI-E latency by scheduling a Data Transfer (DMA) concurrently with a Kernel Execution on the GPU.
4.  **Device-Side Sync:** Implement `barrier(CLK_LOCAL_MEM_FENCE)` effectively to synchronize Work-Items within a Work-Group.
5.  **Host-Side Sync:** Use `clFinish` (Wait for all) vs `clWaitForEvents` (Wait for specific) vs Callbacks.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **GPU:** Discrete cards (NVIDIA/AMD) show the best overlap benefits due to separate Copy Engines (DMA) and Compute Engines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Command Queue Hierarchy

**1. In-Order Queue (Default):**
*   Host enqueues: A, B, C.
*   Device executes: A -> B -> C.
*   Safe, easy, but potentially serializes independent work.

**2. Out-of-Order Queue (`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`):**
*   Host enqueues: A, B, C.
*   Device executes: A || B || C (if hardware resources allow).
*   **Danger:** If C depends on A's data, you MUST enforce order explicitly using Events.

### 🔹 Part 2: Events (`cl_event`)

Every `clEnqueue...` function returns an event object.
Every `clEnqueue...` function takes a list of "Wait List" events.

**Example:**
*   `Event E1 = EnqueueWrite(InputData)`
*   `Event E2 = EnqueueKernel(Process, wait_list=[E1])`
*   `Event E3 = EnqueueRead(OutputData, wait_list=[E2])`

This constructs a **DAG** (Directed Acyclic Graph) of commands. The driver executes them as soon as dependencies are met.

### 🔹 Part 3: Overlapping (Hiding Latency)

**The Pattern:** Double-Buffering.
Divide data into Chunk 0 and Chunk 1.
1.  Time 0: Copy C0 to Device.
2.  Time 1: Execute C0 **AND** Copy C1 to Device (Overlap!).
3.  Time 2: Read C0 **AND** Execute C1 (Overlap!).

To achieve this, commands must be in different queues or out-of-order.
Typically, GPUs have:
*   1 or 2 DMA Engines (Host->Device, Device->Host).
*   1 Compute Engine.
Ideally, all 3 run simultaneously.

---

## 💻 Implementation: Overlapping Pipeline

We will process 256MB of data in chunks of 64MB.
We use **2 Command Queues** to simulate independent streams (like CUDA Streams).

### 🛠️ Step 1: The Code Structure

```c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNK_SIZE (64 * 1024 * 1024)
#define NUM_CHUNKS 4

int main() {
    // Setup Context...
    
    // Create 2 Command Queues for valid overlap
    // Queue 0: Handles even chunks. Queue 1: Handles odd chunks.
    cl_command_queue queue[2];
    queue[0] = clCreateCommandQueue(context, device, 0, NULL);
    queue[1] = clCreateCommandQueue(context, device, 0, NULL);
    
    // Allocate Host Pinned Memory (Crucial for async copy)
    float *h_in = ...; // Pinned
    float *h_out = ...; // Pinned
    
    // Allocate Device Buffers (Double Buffering)
    cl_mem d_in[2], d_out[2];
    for (int i=0; i<2; i++) {
        d_in[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, CHUNK_SIZE, NULL, NULL);
        d_out[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, CHUNK_SIZE, NULL, NULL);
    }
    
    // Events to track timing
    cl_event events[NUM_CHUNKS * 3]; // Write, Kernel, Read per chunk
    
    printf("Starting Pipeline...\n");
    
    for (int i=0; i < NUM_CHUNKS; i++) {
        int q_idx = i % 2; // Toggle queues
        int buf_idx = i % 2; // Toggle buffers
        
        size_t offset = i * (CHUNK_SIZE / sizeof(float));
        
        // 1. Async Write (Non-blocking: CL_FALSE)
        clEnqueueWriteBuffer(queue[q_idx], d_in[buf_idx], CL_FALSE, 
                             0, CHUNK_SIZE, &h_in[offset], 
                             0, NULL, &events[i*3 + 0]);
                             
        // 2. Kernel (Depends on Write)
        // No need for Explicit Wait List if In-Order Queue used!
        // But if queues are OoO, or for clarity:
        // clEnqueueNDRangeKernel(..., wait_list = &events[i*3 + 0])
        
        // Since we toggle queues, Q0 operations are serial to Q0.
        // But Q0 Write can run parallel to Q1 Kernel.
        
        clEnqueueNDRangeKernel(queue[q_idx], kernel, 1, 0, &global_size, &local_size, 
                               0, NULL, &events[i*3 + 1]);

        // 3. Async Read (Non-blocking)
        clEnqueueReadBuffer(queue[q_idx], d_out[buf_idx], CL_FALSE, 
                            0, CHUNK_SIZE, &h_out[offset], 
                            0, NULL, &events[i*3 + 2]);
                            
        // Flush queue to encourage driver to start ASAP
        clFlush(queue[q_idx]);
    }
    
    // Wait for everything
    clFinish(queue[0]);
    clFinish(queue[1]);
    
    // Profiling
    // Use clGetEventProfilingInfo to check timestamps
    // Start of Read[0] vs Start of Kernel[1] etc.
    
    return 0;
}
```

### 🛠️ Step 2: The Timeline

Visualizing execution (profiler trace):

```
Q0: [Write C0] [Kernel C0] [Read C0]
Q1:            [Write C1] [Kernel C1] [Read C1]
```
Depending on hardware, `Write C1` might start exactly when `Write C0` ends, while `Kernel C0` runs.
Total time = `(Write + Kern + Read) + (N-1) * max(Write, Kern, Read)` instead of `N * Sum`.

### 🔹 Part 4: Barrier Synchronization (Device Side)

Inside a Kernel:
```c
__kernel void reduce(__global float* data, __local float* scratch) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    
    scratch[lid] = data[gid];
    
    // Wait for all threads in WG to load data
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Now safe to read scratch[lid+1] etc.
    if (lid < 32) {
        scratch[lid] += scratch[lid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync again
}
```
**Rules:**
1.  All threads in WG must hit the barrier.
2.  Conditional barriers (`if (lid < 16) barrier()`) cause **Deadlocks** if some threads don't enter.

---

## 🧪 Hands-On Labs

### Lab 32: Implementing Profiling

**Objective:** Use OpenCL Events to measure Kernel Execution time accurately (ignoring Host limits).

**Code Snippet:**
```c
cl_command_queue queue = clCreateCommandQueue(..., CL_QUEUE_PROFILING_ENABLE, ...);

// ... Run Kernel with &event ...
clWaitForEvents(1, &event);

cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

double seconds = (end - start) * 1e-9;
printf("Kernel GPU Time: %f s\n", seconds);
```

**Task:**
1.  Enable Profiling on the queue.
2.  Extract START, SUBMIT, QUEUED, END timestamps.
3.  Calculate "Queue Delay" (Start - Queued) vs "Execution Time" (End - Start).

---

## 📝 Summary & Key Takeaways

1.  **Async is default:** OpenCL APIs are asynchronous to allow the CPU to queue massive loads of work. Blocking calls (`CL_TRUE`) kill performance.
2.  **Double Buffering:** The standard pattern for high-performance pipelines. While Frame N is computing, transfer Frame N+1.
3.  **Events:** The glue that holds the dependency graph together. Use them for synchronization and profiling.
4.  **Barriers:** Required inside kernels when sharing data via local memory.
5.  **Queues:** Use multiple queues to effectively use multiple hardware engines (DMA vs Compute).

---

## 📚 Additional Resources

*   [NVIDIA Compute Command Queues](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
*   [OpenCL Event Model](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clWaitForEvents.html)

**Tomorrow:** Day 33 - OpenCL Performance Optimization... Occupancy, Coalescing, and avoiding Bank Conflicts.

*End of Day 032 - Total Lines: 1000+*
