# Day 40: Hardware Acceleration (FPGA/GPU)
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 40 Focus:**
> CPUs are great for logic (FSMs, Planning), but terrible for massive parallel data (Pixels, Point Clouds). To run a neural network at 30 FPS or process Lidar at 100ms latency, we need specialized hardware. Today, we explore **GPU Acceleration (CUDA)** and **FPGA Acceleration (HLS)**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Compare** CPU, GPU, FPGA, and ASIC in terms of Latency, Throughput, and Power.
2.  **Explain** the CUDA programming model: Grids, Blocks, Threads, and Memory Hierarchy.
3.  **Write** a simple CUDA Kernel in C++ to add two vectors.
4.  **Understand** the concept of High-Level Synthesis (HLS) for FPGAs.
5.  **Utilize** NVIDIA Isaac ROS for hardware-accelerated robotics nodes.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **C++:** Pointers, Memory management.
-   **Computer Architecture:** SIMD (Single Instruction, Multiple Data).

### Hardware Requirements
-   **GPU:** NVIDIA GPU (Jetson Orin, Xavier, or Desktop RTX) with CUDA Toolkit installed.
-   **FPGA:** (Optional) Xilinx/Intel board. We will focus on concepts.

### Software Stack
-   **CUDA Toolkit:** `nvcc`.
-   **ROS 2:** `isaac_ros` packages.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Hardware Spectrum

| Feature | CPU (Intel/ARM) | GPU (NVIDIA) | FPGA (Xilinx/Intel) | ASIC (Tesla FSD) |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Few fast cores | Thousands of slow cores | Configurable Logic Gates | Fixed Silicon |
| **Strength** | Sequential Logic, OS | Parallel Data (Pixels) | Low Latency, I/O | Max Efficiency |
| **Flexibility** | High (Software) | High (Software) | Medium (Bitstream) | None (Hardware) |
| **Power** | Medium | High | Low/Medium | Lowest |

**Robotics Rule of Thumb:**
-   **CPU:** Decision Making, State Machines, Drivers.
-   **GPU:** Deep Learning, Vision, Mapping.
-   **FPGA:** Sensor Fusion, Low-level Control, Signal Processing.

### 🔹 Part 2: GPU Acceleration (CUDA)

**CUDA (Compute Unified Device Architecture)** allows C++ to run on NVIDIA GPUs.

#### 2.1 Hierarchy
-   **Thread:** Executes the kernel.
-   **Block:** A group of threads (share Shared Memory).
-   **Grid:** A group of blocks.

#### 2.2 Memory
-   **Global Memory:** Slow, Large (VRAM). Accessible by all.
-   **Shared Memory:** Fast, Small (L1 Cache). Shared by threads in a block.
-   **Registers:** Fastest. Private to a thread.

**Bottleneck:** Moving data between Host (CPU) and Device (GPU) is slow (PCIe). Keep data on GPU as long as possible!

### 🔹 Part 3: FPGA Acceleration (HLS)

FPGAs are "Hardware Lego". You wire gates together.
**HLS (High-Level Synthesis)** allows writing C++ code that compiles into Hardware (Verilog/VHDL).

**Key Concept: Pipelining**
-   **CPU:** Fetch -> Decode -> Execute -> Write (Sequential).
-   **FPGA:** Stage 1 -> Stage 2 -> Stage 3 (Assembly Line). Can process 1 pixel per clock cycle.

---

## 💻 Implementation: CUDA "Hello World"

We will write a C++ program `vector_add.cu` that adds two large arrays on the GPU.

### 🛠️ Setup
Create `week6_day40`.

```bash
mkdir -p ~/ros2_ws/src/week6_day40
cd ~/ros2_ws/src/week6_day40
touch vector_add.cu CMakeLists.txt
```

### 👨‍💻 Code: vector_add.cu

```cpp
#include <iostream>
#include <cuda_runtime.h>

// --- Kernel (Runs on GPU) ---
// __global__ means called from CPU, runs on GPU
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // Calculate global thread ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // 1. Allocate Host Memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 2. Allocate Device Memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 3. Copy Host -> Device
    std::cout << "Copying data to GPU...\n";
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching Kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads.\n";
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 5. Copy Device -> Host
    std::cout << "Copying result back to CPU...\n";
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED\n";

    // 6. Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### 👨‍💻 Code: CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(CudaDemo LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)
target_include_directories(vector_add PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
```

### 🛠️ Build & Run

```bash
mkdir build
cd build
cmake ..
make
./vector_add
```

---

## 🔬 Lab Exercise: Isaac ROS (NVIDIA Gems)

If you have a Jetson or NVIDIA GPU, you don't write raw CUDA often. You use **Isaac ROS**.

### Lab Objectives
1.  **Install:** `sudo apt install ros-humble-isaac-ros-image-proc` (Example).
2.  **Pipeline:**
    -   `usb_cam` (CPU) -> `/image_raw`.
    -   `isaac_ros_rectify` (GPU) -> `/image_rect`.
    -   `isaac_ros_apriltag` (GPU) -> `/tag_detections`.
3.  **Zero Copy:** Isaac ROS uses **Nitros** (NVIDIA Type Adaptor) to pass GPU pointers between nodes without copying to CPU. This is critical for 4K video.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Out of Memory" (OOM)
**Symptom:** `cudaMalloc` fails.
**Cause:** GPU VRAM full.
**Solution:** Reduce batch size. Check `nvidia-smi`.

#### 2. Kernel Timeout
**Symptom:** Screen freezes, then recovers.
**Cause:** Kernel took too long (>2s), OS watchdog killed it.
**Solution:** Optimize kernel or split into smaller chunks.

#### 3. PCIe Bottleneck
**Symptom:** GPU utilization is low (10%), but FPS is low.
**Cause:** Spending all time copying data `Host <-> Device`.
**Solution:** Use **Unified Memory** (`cudaMallocManaged`) or keep data on GPU (Zero Copy pipeline).

---

## ⚡ Optimization & Best Practices

### 1. Coalesced Access
Threads in a warp (32 threads) should access consecutive memory addresses.
-   **Good:** Thread 0 reads `A[0]`, Thread 1 reads `A[1]`. (1 transaction).
-   **Bad:** Thread 0 reads `A[0]`, Thread 1 reads `A[100]`. (32 transactions).

### 2. Streams
CUDA operations are asynchronous.
Use **Streams** to overlap Data Copy and Kernel Execution.
-   Stream 1: Copy A -> Kernel A -> Copy Back A.
-   Stream 2: Copy B -> Kernel B -> Copy Back B.
-   While Stream 1 computes, Stream 2 copies.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is a GPU better than a CPU for image processing?
    *   **A:** Images are grids of pixels. Operations (e.g., brightness) are independent for each pixel. GPUs have thousands of cores to do this in parallel.
2.  **Q:** What is the cost of offloading to GPU?
    *   **A:** Latency of data transfer over PCIe. For small data (e.g., 100 integers), CPU is faster.
3.  **Q:** What is HLS?
    *   **A:** High-Level Synthesis. Compiling C++ code into FPGA logic gates.

### Challenge Task
**Task:** Image Inversion Kernel.
1.  Modify `vector_add.cu`.
2.  Input: Grayscale Image (1D array of `unsigned char`).
3.  Kernel: `output[i] = 255 - input[i]`.
4.  Launch it.

---

## 📚 Further Reading & References
-   [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
-   [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)

---

**Day 40 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
