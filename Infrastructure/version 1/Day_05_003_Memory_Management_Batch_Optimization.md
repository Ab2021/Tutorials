# Day 5.3: Memory Management & Batch Size Optimization

## 🧠 Compute & Accelerator Optimization - Part 3

**Focus**: Memory Hierarchy Optimization, Dynamic Batch Sizing, Memory-Efficient Training Techniques  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master memory hierarchy optimization for deep learning workloads
- Understand dynamic batch sizing strategies and their theoretical foundations
- Learn memory-efficient training techniques and their trade-offs
- Analyze memory bottlenecks and develop optimization strategies

---

## 🧮 Memory Hierarchy Theory for Deep Learning

### **Memory Requirements Analysis Framework**

Deep learning models have complex memory requirements that vary significantly across different phases of training and inference. Understanding these patterns is crucial for optimization.

**Training Memory Decomposition:**
```
Total Memory = M_parameters + M_gradients + M_optimizer + M_activations + M_temp

Detailed Breakdown:
M_parameters = P × sizeof(dtype)  // Model weights
M_gradients = P × sizeof(dtype)   // Same size as parameters
M_optimizer = P × k × sizeof(dtype)  // k depends on optimizer (2 for Adam momentum + variance)
M_activations = Σᵢ B × Hᵢ × Wᵢ × Cᵢ × sizeof(dtype)  // Layer-wise activations
M_temp = Variable based on operations  // Temporary computation buffers

Where:
- P = total number of parameters
- B = batch size
- Hᵢ, Wᵢ, Cᵢ = height, width, channels for layer i
```

**Memory Scaling Laws:**
```
Parameter Memory: O(Model_size) - Independent of batch size
Activation Memory: O(Model_depth × Batch_size) - Linear in both
Optimizer Memory: O(Model_size × Optimizer_factor)

Critical Insight: Activation memory often dominates for large batch training
```

### **Activation Memory Optimization Theory**

**Gradient Checkpointing Mathematical Framework:**

Gradient checkpointing trades computation for memory by selectively storing intermediate activations and recomputing others during backpropagation.

**Optimal Checkpointing Strategy:**
```
Memory-Time Trade-off:
Let L = number of layers, C = computational cost, M = memory cost

Without checkpointing:
- Memory: O(L)
- Computation: O(L)  (forward + backward)

With optimal checkpointing:
- Memory: O(√L)  
- Computation: O(L√L)

Optimal Checkpoint Placement:
For L layers, place checkpoints every √L layers
Results in √L checkpoints, each requiring recomputation of √L layers
```

**Dynamic Programming Formulation:**
```
Define: dp[i][j] = minimum memory to compute gradients from layer i to j

Recurrence:
dp[i][j] = min over k ∈ [i,j-1] of:
    max(memory[i][k], dp[k+1][j]) + recomputation_cost[i][k]

Optimal Solution: dp[0][L-1] gives minimum memory for full model
```

**Activation Compression Techniques:**

**Lossy Compression Theory:**
```
Quantization-based Compression:
- Store activations in lower precision (e.g., INT8)
- Memory reduction: 4× for FP32 → INT8
- Precision loss: Controlled through quantization scales

Sparsification-based Compression:
- Store only top-k% of activation values
- Memory reduction: k-dependent (e.g., 10× for k=10%)
- Information loss: Potentially higher impact on convergence
```

---

## 📊 Batch Size Optimization Theory

### **Batch Size Impact Analysis**

**Convergence Theory:**
```
SGD Convergence Rate:
E[||∇f(θₜ)||²] ≤ O(1/t) + O(σ²/B)

Where:
- t = iteration number
- σ² = gradient variance
- B = batch size

Key Insights:
1. Larger batch sizes reduce gradient noise (σ²/B term)
2. May require more iterations due to reduced stochasticity
3. Optimal batch size balances noise reduction with exploration
```

**Critical Batch Size Theory:**
```
Critical Batch Size: B_crit = σ²/ε²

Where:
- σ² = gradient noise scale
- ε = learning rate

Beyond B_crit:
- Diminishing returns in convergence speed
- Wasted computational resources
- Potential generalization degradation
```

### **Adaptive Batch Sizing Strategies**

**Gradient Noise Scale Estimation:**
```
Noise Scale Estimation:
σ² = E[||∇f(θ,ξᵢ) - ∇f(θ)||²]

Where ξᵢ represents individual sample gradients

Practical Estimation:
1. Compute gradients for subset of samples
2. Estimate variance across sample gradients
3. Adjust batch size based on noise scale
```

**Dynamic Batch Size Scheduling:**

**Linear Scaling Strategy:**
```
Batch Size Schedule: B(t) = B₀ × (1 + αt)

Rationale:
- Start with smaller batch for better exploration
- Increase batch size as optimization progresses
- Reduce gradient noise near convergence

Theoretical Justification:
- Early training: High curvature, needs exploration
- Late training: Near optimum, benefits from noise reduction
```

**AdaBatch Algorithm:**
```
Adaptive Batch Size Selection:
1. Monitor gradient variance σ²(t)
2. Estimate optimal batch size: B_opt(t) = σ²(t)/ε²
3. Adjust current batch size towards B_opt(t)
4. Constraints: Memory limits, hardware efficiency

Update Rule:
B(t+1) = clip(αB_opt(t) + (1-α)B(t), B_min, B_max)
```

---

## 🔧 Memory-Efficient Training Techniques

### **Zero Redundancy Optimizer (ZeRO) Deep Dive**

**Theoretical Foundation:**
ZeRO eliminates memory redundancy in data parallel training by partitioning optimizer states, gradients, and parameters across data parallel processes.

**ZeRO-1: Optimizer State Partitioning**
```
Memory Analysis:
Standard Data Parallel: Each worker stores complete optimizer state
- Adam: 8 bytes per parameter (momentum + variance in FP32)
- Total: 8P bytes per worker

ZeRO-1: Partition optimizer state across workers
- Each worker: 8P/N bytes
- Memory reduction: N× (number of workers)
- Communication: No additional overhead during forward/backward
```

**ZeRO-2: Gradient Partitioning**
```
Gradient Communication Optimization:
Standard: All-reduce gradients (each worker sends/receives full gradients)
- Communication: 2P bytes per worker

ZeRO-2: Reduce-scatter gradients
- Each worker owns subset of gradients
- Communication: 2P/N bytes useful data per worker
- Additional scatter: Each worker broadcasts parameters after update
```

**ZeRO-3: Parameter Partitioning**
```
Just-in-Time Parameter Communication:
- Each worker stores only P/N parameters
- Forward pass: All-gather parameters for each layer
- Backward pass: Parameters discarded after use
- Memory: P/N + small buffer for communication

Communication Analysis:
- Forward: P bytes communicated per layer
- Backward: P bytes communicated per layer  
- Total: 2P bytes per iteration (same as standard data parallel)
```

### **Gradient Accumulation Theory**

**Memory-Computation Trade-off:**
```
Effective Batch Size: B_eff = B_micro × accumulation_steps

Memory Usage:
- With accumulation: M = M_model + M_activations(B_micro)
- Without accumulation: M = M_model + M_activations(B_eff)

For B_eff >> B_micro:
Memory reduction ≈ B_eff / B_micro

Computational Overhead:
- Additional gradient accumulation operations
- Multiple forward passes per backward pass
- Typically < 5% overhead
```

**Gradient Accumulation with Mixed Precision:**
```
Precision Strategy:
1. Forward pass: FP16 computation
2. Gradient accumulation: FP32 for numerical stability
3. Parameter updates: FP32

Memory Impact:
- Gradient buffer: FP32 (4 bytes per parameter)
- Temporary gradients: FP16 (2 bytes per parameter)
- Net memory increase: 2 bytes per parameter
```

---

## 🎯 Advanced Memory Optimization Techniques

### **Memory Pool Management**

**Memory Allocation Theory:**

Deep learning frameworks benefit from sophisticated memory management due to the predictable and repetitive nature of training workloads.

**Memory Pool Benefits:**
```
Fragmentation Reduction:
- Traditional malloc: O(log n) allocation time, fragmentation issues
- Memory pool: O(1) allocation, predictable layout

Peak Memory Reduction:
- Reuse memory across non-overlapping tensors
- Optimal scheduling: Graph coloring problem
- Heuristics: Largest-first allocation, lifetime analysis
```

**Memory Layout Optimization:**
```
Data Layout Strategies:
1. Array of Structures (AoS): [x₁,y₁,z₁][x₂,y₂,z₂]...
2. Structure of Arrays (SoA): [x₁,x₂,...][y₁,y₂,...][z₁,z₂,...]

Performance Implications:
- AoS: Better for accessing all features of one sample
- SoA: Better for SIMD operations on single feature
- Deep Learning: Generally prefers SoA (batch operations)
```

### **Memory-Aware Model Architecture Design**

**Efficient Architecture Patterns:**

**Depthwise Separable Convolutions:**
```
Standard Convolution Memory:
M_weights = K × K × C_in × C_out
M_activations = H × W × C_out × B

Depthwise Separable Memory:
M_weights = K × K × C_in + C_in × C_out  
Parameter reduction: (K × K × C_in × C_out) / (K × K × C_in + C_in × C_out)
                   ≈ C_out / (K² + C_out/C_in) for large C_out
```

**MobileNet Efficiency Analysis:**
```
Memory Efficiency:
- 28× fewer parameters than VGG-16
- Similar accuracy on ImageNet
- Activation memory: Proportional to network depth and width

Width Multiplier α:
- Parameters scale as α²
- Memory scales as α
- Accuracy degrades gradually with α
```

**Attention Mechanism Memory Analysis:**
```
Self-Attention Memory Complexity:
M_attention = O(L² × d + L × d²)

Where:
- L = sequence length
- d = hidden dimension

Memory Optimization Strategies:
1. Sparse Attention: Reduce L² term to O(L√L) or O(L log L)
2. Low-rank Approximation: Reduce d² term
3. Gradient Checkpointing: Trade computation for memory
```

---

## 📈 Performance Analysis and Optimization

### **Memory Bandwidth Utilization Analysis**

**Roofline Model for Memory-Bound Operations:**
```
Operational Intensity Analysis:
OI = FLOPs / Bytes_accessed

Memory-bound threshold:
OI_threshold = Peak_FLOPS / Memory_Bandwidth

For modern GPUs:
- A100: ~10 FLOPs/byte
- V100: ~17 FLOPs/byte
- H100: ~20 FLOPs/byte

Operations below threshold are memory-bound
```

**Memory Access Pattern Optimization:**
```
Coalescing Efficiency:
η_coalescing = Useful_bytes / Total_bytes_transferred

Factors Affecting Coalescing:
1. Access stride: Consecutive threads → consecutive addresses
2. Alignment: Starting address multiple of cache line size
3. Data type: 4, 8, 16 byte types coalesce well

Target: η_coalescing > 0.8 for memory-intensive kernels
```

### **Dynamic Memory Management Strategies**

**Adaptive Memory Allocation:**
```
Memory Pressure Detection:
- Monitor available GPU memory
- Track allocation/deallocation patterns
- Predict future memory requirements

Adaptive Strategies:
1. Reduce batch size when memory pressure high
2. Enable gradient checkpointing dynamically
3. Adjust precision (FP32 → FP16) for non-critical operations
```

**Memory Defragmentation:**
```
Fragmentation Metrics:
Fragment_ratio = (Allocated_memory - Used_memory) / Total_memory

Defragmentation Strategies:
1. Periodic garbage collection
2. Memory compaction during idle periods
3. Smart allocation ordering (largest allocations first)
```

---

## 🔍 Profiling and Debugging Memory Issues

### **Memory Profiling Methodology**

**GPU Memory Analysis:**
```
Key Metrics:
1. Peak memory usage vs available memory
2. Memory utilization over time
3. Allocation/deallocation patterns
4. Memory fragmentation levels

Tools and Techniques:
- NVIDIA NSight Systems: Timeline analysis
- PyTorch Profiler: Python-level memory tracking
- Custom memory hooks: Application-specific profiling
```

**Memory Leak Detection:**
```
Leak Detection Strategy:
1. Baseline memory measurement
2. Run training loop iterations
3. Monitor memory growth trend
4. Identify accumulating allocations

Common Leak Sources:
- Retained computation graphs
- Cached intermediate results
- Growing data structures (lists, dictionaries)
- Circular references in Python
```

### **Optimization Decision Framework**

**Memory vs Computation Trade-offs:**
```
Decision Matrix:
                    Memory Abundant    Memory Constrained
Computation Cheap  Store all          Gradient checkpointing
Computation Costly Recompute minimal  Selective checkpointing

Quantitative Analysis:
Cost = α × Memory_usage + β × Computation_time
Optimize: min(Cost) subject to memory constraints
```

**Batch Size Selection Strategy:**
```
Multi-objective Optimization:
Objectives:
1. Maximize throughput (samples/second)
2. Minimize memory usage
3. Maintain convergence quality

Pareto Frontier Analysis:
- Identify non-dominated solutions
- Select based on resource constraints
- Adapt dynamically during training
```

This comprehensive memory management framework provides the theoretical foundation for optimizing deep learning workloads across different memory hierarchies and constraints. The key insight is that effective memory management requires understanding the interplay between model architecture, algorithm design, and hardware capabilities.