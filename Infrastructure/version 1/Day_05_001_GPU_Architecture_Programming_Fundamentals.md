# Day 5.1: GPU Architecture & Programming Fundamentals

## 🚀 Compute & Accelerator Optimization - Part 1

**Focus**: GPU/TPU Architecture Theory, CUDA Programming Models, Memory Hierarchy Optimization  
**Duration**: 2-3 hours  
**Level**: Intermediate to Advanced  

---

## 🎯 Learning Objectives

- Master GPU architecture fundamentals and understand the theoretical foundations of parallel computing
- Understand memory hierarchy optimization principles for ML workloads
- Learn parallel programming patterns and thread organization strategies
- Analyze performance bottlenecks and optimization opportunities in GPU-accelerated systems

---

## 🔧 GPU Architecture Theoretical Foundations

### **GPU vs CPU: The Fundamental Paradigms**

The fundamental difference between GPU and CPU architectures stems from their design philosophies:

**CPU Design Philosophy: Latency Optimization**
- Few cores (4-32) optimized for single-thread performance
- Large caches (L1: 32KB, L2: 256KB-1MB, L3: 8-64MB) to minimize memory latency
- Complex branch prediction and out-of-order execution
- Optimized for control-intensive tasks with unpredictable data access patterns

**GPU Design Philosophy: Throughput Optimization**
- Many cores (1024-10240) optimized for parallel execution
- Small caches per core, emphasis on memory bandwidth over latency
- Simple in-order execution with massive parallelism
- Optimized for data-parallel tasks with predictable access patterns

### **Mathematical Performance Model**

```
GPU Performance Model:
P_gpu = N_cores × f_clock × IPC × η_utilization × η_memory

Where:
- N_cores = number of CUDA cores (2048-10240 for modern GPUs)
- f_clock = base clock frequency (1.2-2.5 GHz)
- IPC = instructions per clock cycle (0.5-2.0)
- η_utilization = compute utilization efficiency (0.6-0.95)
- η_memory = memory subsystem efficiency (0.3-0.8)

Memory Bandwidth Utilization:
BW_effective = BW_theoretical × η_coalescing × η_occupancy
Target: η_memory > 0.8 for memory-bound kernels

Arithmetic Intensity Analysis:
AI = FLOPs / Bytes_transferred
- AI < 1: Memory-bound (optimize memory access)
- 1 < AI < 10: Balanced (optimize both compute and memory)
- AI > 10: Compute-bound (optimize arithmetic operations)
```

### **GPU Memory Hierarchy Deep Dive**

**1. Global Memory (DRAM)**
- **Capacity**: 16-80GB on modern GPUs
- **Bandwidth**: 900-3350 GB/s (theoretical)
- **Latency**: 200-800 cycles
- **Access Pattern Sensitivity**: High - coalesced access critical
- **Use Case**: Large datasets, model parameters, primary data storage

**2. Shared Memory (On-chip SRAM)**  
- **Capacity**: 64-228KB per streaming multiprocessor
- **Bandwidth**: ~10TB/s (shared among threads in block)
- **Latency**: 1-2 cycles (when no bank conflicts)
- **Access Pattern**: Bank conflicts can reduce performance
- **Use Case**: Inter-thread communication, data reuse within thread blocks

**3. Constant Memory**
- **Capacity**: 64KB cached portion of global memory
- **Bandwidth**: High when all threads access same location
- **Latency**: Similar to L1 cache when cached
- **Access Pattern**: Broadcast reads optimal, scattered reads poor
- **Use Case**: Read-only data accessed uniformly by all threads

**4. Texture Memory**
- **Capacity**: Cached portion of global memory
- **Bandwidth**: Optimized for 2D spatial locality
- **Latency**: Lower than global memory for spatially local access
- **Access Pattern**: 2D locality and interpolation hardware
- **Use Case**: Image processing, spatially coherent data access

**5. Register Memory**
- **Capacity**: 32-64K registers per streaming multiprocessor
- **Bandwidth**: Highest (local to each thread)
- **Latency**: Zero cycles (when available)
- **Access Pattern**: Thread-private, no conflicts
- **Use Case**: Thread-local variables, intermediate computations

### **Thread Hierarchy and Execution Model**

**CUDA Thread Organization:**
```
Grid → Blocks → Warps → Threads

Grid: Collection of thread blocks executing the same kernel
- Dimensions: up to 3D (gridDim.x, gridDim.y, gridDim.z)
- Scale: Limited by GPU memory and compute capability

Block: Collection of threads that can cooperate
- Dimensions: up to 3D (blockDim.x, blockDim.y, blockDim.z)
- Size limit: 1024 threads per block (modern GPUs)
- Synchronization: __syncthreads() within block only

Warp: Collection of 32 threads executed in lockstep
- SIMT execution: Single Instruction, Multiple Thread
- Branch divergence penalty when threads take different paths
- Memory coalescing unit: 32 threads access memory together
```

**Occupancy Theory:**
```
Theoretical Occupancy = Active_Warps / Max_Warps_per_SM

Limiting Factors:
1. Registers per thread × Threads per block ≤ Registers per SM
2. Shared memory per block ≤ Shared memory per SM  
3. Thread blocks per SM ≤ Max blocks per SM
4. Threads per block ≤ Max threads per SM

Optimal occupancy ≠ Optimal performance
- High occupancy improves latency hiding
- May reduce cache efficiency due to resource competition
- Sweet spot often 50-75% occupancy for many workloads
```

---

## 🧮 Parallel Algorithm Design Principles

### **Data Parallelism Patterns**

**1. Map Pattern**
- **Definition**: Apply same operation to each element independently
- **Examples**: Element-wise operations, activation functions
- **Characteristics**: Perfect parallelization, no synchronization needed
- **Memory Access**: Can be optimized for coalescing

**2. Reduction Pattern**
- **Definition**: Combine all elements using associative operation
- **Examples**: Sum, max, matrix norms
- **Challenges**: Requires synchronization, potential for divergence
- **Optimization**: Tree-based reduction, warp shuffle operations

**3. Scan Pattern**
- **Definition**: Compute prefix operations (cumulative sum, etc.)
- **Examples**: Prefix sum, histogram construction
- **Complexity**: O(log n) steps with O(n log n) work
- **Applications**: Memory allocation, stream compaction

**4. Stencil Pattern**
- **Definition**: Update elements based on local neighborhood
- **Examples**: Convolutions, finite difference methods
- **Memory Requirements**: Significant data reuse opportunities
- **Optimization**: Tiling, shared memory utilization

### **Memory Access Optimization Theory**

**Coalescing Requirements:**
```
Perfect Coalescing Conditions:
1. Consecutive threads access consecutive memory addresses
2. Access starts at aligned boundary (128-byte for modern GPUs)
3. Data type size is 4, 8, or 16 bytes
4. No gaps or overlaps in access pattern

Performance Impact:
- Coalesced access: 1 memory transaction per warp
- Uncoalesced access: up to 32 memory transactions per warp
- Performance difference: 10-30x in memory-bound kernels
```

**Bank Conflict Analysis:**
```
Shared Memory Banks:
- Modern GPUs: 32 banks, 4-byte wide
- Bank index = (address / 4) % 32
- Conflict occurs when multiple threads access same bank
- Broadcast: All threads access same address (no conflict)

Conflict Resolution:
- 2-way conflict: 2x slower access
- n-way conflict: n×x slower access
- Padding strategies to avoid conflicts
```

---

## 🎯 Performance Analysis Framework

### **Roofline Model for GPU Performance**

The Roofline Model provides theoretical performance bounds based on arithmetic intensity:

```
Performance Bounds:
P_compute = Peak_FLOPS (compute-bound ceiling)
P_memory = Arithmetic_Intensity × Memory_Bandwidth (memory-bound ceiling)

Actual_Performance = min(P_compute, P_memory)

Key Insights:
- Low AI kernels are memory-bound: optimize memory access
- High AI kernels are compute-bound: optimize arithmetic operations
- Knee point = Peak_FLOPS / Memory_Bandwidth
```

**Modern GPU Roofline Characteristics:**
- **A100**: Knee at AI ≈ 9.6 (19.5 TFLOPS / 2.0 TB/s)
- **H100**: Knee at AI ≈ 20 (67 TFLOPS / 3.35 TB/s)
- **V100**: Knee at AI ≈ 17.4 (15.7 TFLOPS / 0.9 TB/s)

### **Performance Bottleneck Identification**

**Memory-Bound Indicators:**
- Low arithmetic intensity (< 10 FLOPs/byte)
- Poor memory coalescing efficiency
- High memory latency sensitivity
- Performance scales with memory bandwidth

**Compute-Bound Indicators:**
- High arithmetic intensity (> 50 FLOPs/byte)
- High register usage per thread
- Performance scales with core count/frequency
- Little sensitivity to memory optimizations

**Occupancy-Bound Indicators:**
- Resource limitations prevent high occupancy
- Performance improves with reduced resource usage
- Synchronization bottlenecks between warps
- Load imbalance across threads

---

## 🔬 Advanced GPU Architecture Features

### **Tensor Cores and Mixed Precision**

**Tensor Core Architecture:**
- **Purpose**: Accelerate mixed-precision matrix operations
- **Supported Operations**: GEMM (General Matrix Multiply)
- **Data Types**: FP16, BF16, INT8, INT4 (architecture dependent)
- **Throughput**: 10-20x higher than CUDA cores for supported operations

**Mixed Precision Benefits:**
```
Memory Reduction:
- FP16: 50% memory usage vs FP32
- Bandwidth: 2x effective bandwidth
- Capacity: 2x model size capacity

Performance Improvement:
- Tensor Cores: 10-20x speedup for GEMM operations
- Memory Bandwidth: 2x effective utilization
- Cache Efficiency: More data fits in cache hierarchy

Precision Considerations:
- Dynamic range: FP16 (5-bit exponent vs 8-bit for FP32)
- Gradient scaling required for training stability
- Automatic mixed precision (AMP) frameworks handle complexity
```

### **Multi-Instance GPU (MIG) Technology**

**MIG Partitioning Theory:**
- **Purpose**: Partition single GPU into multiple isolated instances  
- **Resource Isolation**: Memory, compute, cache partitioning
- **Use Cases**: Multi-tenant environments, inference serving
- **Granularity**: 1/7, 2/7, 3/7, 4/7 GPU partitions (A100)

**Performance Implications:**
- **Isolation**: Guaranteed resources, no interference
- **Overhead**: Some performance loss due to partitioning
- **Scheduling**: Enables better resource utilization
- **QoS**: Predictable performance characteristics

---

## 📊 GPU Selection and Optimization Strategy

### **Workload Characterization Framework**

**Training Workloads:**
- **Memory Requirements**: Model parameters + activations + gradients
- **Compute Pattern**: Forward pass + backward pass
- **Communication**: Parameter updates, gradient synchronization
- **Optimization Focus**: Training throughput, memory efficiency

**Inference Workloads:**
- **Memory Requirements**: Model parameters + single batch activations
- **Compute Pattern**: Forward pass only
- **Latency Sensitivity**: Real-time constraints common
- **Optimization Focus**: Latency, throughput per dollar

**Data Processing Workloads:**
- **Memory Requirements**: Large datasets, streaming processing
- **Compute Pattern**: Embarrassingly parallel operations
- **I/O Requirements**: High bandwidth data movement
- **Optimization Focus**: Throughput, memory bandwidth utilization

### **Architecture Selection Criteria**

**Memory-First Analysis:**
```
Memory Requirements Estimation:
Training: Memory = Parameters × 4 + Activations + Gradients + Optimizer_State
Inference: Memory = Parameters × 4 + Batch_Size × Activation_Memory

Modern GPU Memory Capacities:
- Consumer GPUs: 12-24GB (RTX 3090, RTX 4090)
- Professional GPUs: 32-80GB (V100, A100, H100)
- Memory scaling: Often the primary constraint for large models
```

**Compute Requirements Analysis:**
```
FLOPs Estimation:
Training: FLOPs = 2 × Parameters + Activation_FLOPs (per sample)
Inference: FLOPs = Parameters + Activation_FLOPs (per sample)

Modern GPU Compute Capabilities:
- CUDA Cores: 15-67 TFLOPS (FP32)
- Tensor Cores: 125-1979 TFLOPS (FP16/BF16)
- Specialization: Tensor cores critical for large model training
```

---

## 🎯 Key Takeaways and Optimization Principles

### **Fundamental Optimization Principles**

**1. Memory Access Optimization**
- Coalesce global memory accesses for maximum bandwidth
- Utilize shared memory for data reuse within thread blocks
- Minimize bank conflicts in shared memory access patterns
- Consider memory hierarchy: registers > shared > constant > global

**2. Compute Optimization**
- Maximize occupancy while avoiding resource oversubscription
- Minimize branch divergence within warps
- Utilize specialized units (tensor cores, special function units)
- Balance arithmetic intensity with memory bandwidth

**3. Algorithm Design**
- Design for data parallelism and minimal synchronization
- Consider GPU-friendly algorithms over CPU-optimized approaches
- Implement efficient reduction and scan patterns
- Optimize for the target architecture's strengths

**4. Performance Analysis**
- Use roofline model to identify performance bottlenecks
- Profile memory access patterns and compute utilization
- Analyze occupancy and resource usage
- Consider end-to-end pipeline optimization, not just kernel optimization

### **Architecture Evolution Trends**

**Specialization Trend:**
- Increasing focus on AI/ML specific operations
- Tensor cores, sparsity support, new data types
- Trade-off: Higher peak performance, less general-purpose flexibility

**Memory Wall Solutions:**
- Increasing memory bandwidth (HBM2e, HBM3)
- Larger on-chip memories and caches
- Memory compression and bandwidth optimization techniques

**Multi-GPU and Distributed Computing:**
- High-speed interconnects (NVLink, InfiniBand)
- Hardware support for collective operations
- Overlapping computation and communication

This theoretical foundation provides the basis for understanding modern GPU architectures and optimization strategies. The key is matching algorithm characteristics to architectural strengths while avoiding known performance pitfalls.