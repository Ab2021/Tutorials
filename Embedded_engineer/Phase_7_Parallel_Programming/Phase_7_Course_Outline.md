# Phase 7: Advanced Parallel Programming & Compiler Engineering
## Course Outline - 30 Weeks (210 Days)

---

## Course Overview

**Duration:** 30 weeks / 210 days  
**Focus:** Deep-Dive into Parallel Programming Across Architectures & Compiler Infrastructure  
**Prerequisites:** Phase 6 completion (AI/ML Platform Engineering) or equivalent systems programming background

**Learning Path:**
- CPU SIMD (x86 SSE/AVX) → ARM NEON → RISC-V Vector Extensions
- OpenMP → OpenCL → SYCL → oneAPI
- GPU Advanced (AMD ROCm, Intel oneAPI, Apple Metal)
- Compiler Foundations → LLVM IR → Backend Development
- Optimization Passes → Auto-vectorization → Polyhedral Compilation
- Domain-Specific Languages → Custom Backends → Hardware ISA Design

---

## Phase 7A: CPU SIMD & Vector Programming (Weeks 1-6)

---

## Week 1: x86 SIMD Foundations (SSE/AVX) (Days 1-7)

### Day 1: x86 Architecture & SIMD Evolution
- x86-64 microarchitecture deep dive (Sandy Bridge → Zen 4)
- Pipeline stages, Out-of-Order execution, and register renaming
- SIMD instruction set timeline: MMX (64-bit) → SSE (128-bit) → AVX (256-bit) → AVX-512 (512-bit)
- Vector register files: MM (MMX), XMM (SSE), YMM (AVX), ZMM (AVX-512)
- Data-level parallelism (DLP) vs instruction-level parallelism (ILP) vs thread-level parallelism (TLP)
- SIMD execution units (ports and throughput)
- **Lab:** Use `cpuid` instruction and `lscpu` to enumerate SIMD features; analyze `/proc/cpuinfo`

### Day 2: SSE Programming Model
- Intel intrinsics header (`<immintrin.h>`, `<xmmintrin.h>`) vs inline assembly
- SSE generations: SSE (float), SSE2 (double, integer), SSE3 (horizontal ops), SSSE3 (shuffle), SSE4.1/4.2
- Data types: `__m128`, `__m128d`, `__m128i` for packed operations
- Intrinsic naming convention: `_mm_<op>_<type>` (e.g., `_mm_add_ps`, `_mm_mul_pd`)
- Alignment requirements (16-byte boundaries) and `_mm_malloc` / `_mm_free`
- Masked loads/stores and conditional execution
- **Lab:** Vectorize array addition (4x float32) using `_mm_add_ps`; compare assembly output vs scalar

### Day 3: AVX & Wider Vectors
- AVX introduction: 256-bit YMM registers (8x float, 4x double)
- Three-operand format (non-destructive) vs two-operand SSE
- `_mm256_` intrinsic family and VEX encoding
- Broadcast operations (`_mm256_broadcast_ss`) for efficient constant distribution
- Permute and shuffle: `_mm256_permute_ps`, `_mm256_shuffle_ps`
- Blend and select: `_mm256_blendv_ps` for conditional operations
- FMA (Fused Multiply-Add): `_mm256_fmadd_ps` for reduced rounding error
- **Lab:** Implement matrix-vector multiplication y = Ax using AVX2; measure GFLOPS

### Day 4: AVX-512 & Advanced Features
- AVX-512 Foundation (F): 512-bit ZMM0-ZMM31 registers (16x float, 8x double)
- Opmask registers (k0-k7) for fine-grained predication
- Embedded rounding control (nearest, down, up, zero)
- Conflict detection and exponential/reciprocal instructions
- VNNI (Vector Neural Network Instructions) for INT8 dot products
- Tile registers (AMX) on Sapphire Rapids
- **Lab:** Implement optimized GEMM kernel with AVX-512; use perf counters to measure vectorization efficiency

### Day 5: Memory Access Patterns
- Aligned vs unaligned loads: `_mm256_load_ps` vs `_mm256_loadu_ps`
- Cache line effects (64 bytes on x86) and spatial locality
- Non-temporal stores (`_mm256_stream_ps`) to bypass cache for write-only data
- Prefetch instructions (`_mm_prefetch`) for manual cache control
- Gather (`_mm256_i32gather_ps`) and scatter (`_mm256_i32scatter_ps`) for indirect access
- Software prefetching strategies for streaming workloads
- **Lab:** Optimize memory-bound stencil computation; measure L1/L2/L3 cache hit rates with `perf stat`

### Day 6: SIMD Code Generation
- Compiler auto-vectorization: `-O3 -march=native -ftree-vectorize`
- OpenMP SIMD directives: `#pragma omp simd` and attributes (`aligned`, `safelen`)
- GCC vector_size attribute: `typedef float vec8 __attribute__((vector_size(32)))`
- Clang loop hints: `#pragma clang loop vectorize(enable)`
- Analyzing failed vectorization: `-fopt-info-vec-missed` (GCC), `-Rpass-missed=loop-vectorize` (Clang)
- Disassembly inspection: `objdump -d` and `-S -fverbose-asm` for annotated assembly
- **Lab:** Compare hand-written AVX2 vs compiler-vectorized code for SAXPY; analyze vectorization reports

### Day 7: Week 1 Review & Project
- **Project:** SIMD-optimized 2D image convolution (3x3, 5x5 kernels)
- Implement scalar baseline, SSE (4-wide), AVX2 (8-wide), AVX-512 (16-wide) versions
- Handle boundary conditions (zero-padding, clamp, wrap)
- Prefetch input tiles and use non-temporal stores for output
- Benchmark on large images (4K resolution); plot speedup vs vector width
- Profile with `perf` (cycles, instructions, cache misses) and Intel VTune
- Document assembly code quality and identify bottlenecks


---

## Week 2: ARM NEON & Mobile SIMD (Days 8-14)

### Day 8: ARM Architecture Overview
- ARMv8-A architecture: AArch64 execution state and register file (X0-X30, V0-V31)
- NEON/Advanced SIMD unit: 128-bit vector operations (4x float32, 2x float64)
- Load-store architecture vs register-memory (x86)
- Big.LITTLE heterogeneous multiprocessing (Cortex-A76 big + A55 little)
- ARM server CPUs: Graviton3, Ampere Altra, Fujitsu A64FX
- Memory ordering models (Weakly-Ordered vs TSO)
- **Lab:** Cross-compile C code for ARM64 with `aarch64-linux-gnu-gcc`; run in QEMU user-mode emulation

### Day 9: NEON Intrinsics
- ARM NEON intrinsics header (`<arm_neon.h>`) and naming: `v<op><shape>_<type>` (e.g., `vaddq_f32`)
- Vector load/store: `vld1q_f32`, `vst1q_f32` for 128-bit transfers
- Arithmetic operations: `vadd`, `vsub`, `vmul`, `vfma` (fused multiply-add)
- Lane-wise operations: `vget_lane_f32`, `vset_lane_f32` for element access
- Q-registers (128-bit: `float32x4_t`) vs D-registers (64-bit: `float32x2_t`)
- Pairwise operations: `vpaddq_f32` for horizontal reductions
- Interleaving: `vzip`, `vuzp`, `vtrn` for data structure transformations
- **Lab:** Implement SAXPY (y = a*x + y) using NEON; validate with scalar version

### Day 10: ARM SVE (Scalable Vector Extension)
- Vector-length agnostic (VLA) programming: code adapts to hardware vector length (128-2048 bits)
- SVE predicate registers (P0-P15) for per-element masking
- `svfloat32_t` scalable types and `svadd_f32_z` zero-predicated operations
- Loop vectorization with `svwhilelt_b32` for remaining elements
- Comparison with NEON (fixed 128-bit) and x86 AVX-512 (fixed 512-bit)
- SVE2: additional integer/crypto/BFloat16 operations
- **Lab:** Write VLA matrix transpose that runs on any SVE vector length; test with sve-length simulator

### Day 11: Mobile GPU Programming (Mali, Adreno)
- OpenGL ES 3.1+ compute shaders: layout qualifiers, shared memory, barriers
- Vulkan compute pipeline: descriptor sets, command buffers, synchronization
- RenderScript (deprecated): parallel forEach kernels on Android
- Metal Performance Shaders (MPS): convolution, matrix multiply, image filters
- Heterogeneous computing: offload to GPU, DSP (Hexagon), or NPU
- **Lab:** Implement parallel reduction in OpenGL ES compute shader; run on Android device or emulator

### Day 12: Apple Silicon & AMX
- Apple M1/M2/M3 architecture: Performance + Efficiency cores, unified memory, media engines
- AMX (Apple Matrix Extension): undocumented coprocessor for matrix operations
- Accessing AMX via Accelerate framework (`vDSP`, `BNNS`)
- Metal Performance Shaders Graph API for neural networks
- Rosetta 2 translation for x86 binaries with runtime optimization
- Neural Engine (ANE) for CoreML inference
- **Lab:** Use Accelerate `vDSP_mmul` for matrix multiply; compare vs NEON manual implementation

### Day 13: Cross-Platform SIMD Abstraction
- Google Highway: portable SIMD with runtime dispatch (x86/ARM/PPC/WASM)
- Highway ops: `Add`, `Mul`, `LoadU` with automatic width selection
- Vc library: expression templates for vector types
- simde (SIMD Everywhere): SSE/AVX intrinsics emulated on ARM/POWER
- Build system integration: CMake `target_compile_options` for architecture flags
- **Lab:** Implement image blur filter with Highway; verify identical output on x86 and ARM

### Day 14: Week 2 Review & Project
- **Project:** Cross-architecture image processing library (RGB<->YUV conversion, resize, blur)
- Implement backends: x86 (AVX2), ARM (NEON), Apple (Accelerate), fallback (scalar)
- Runtime CPU feature detection with `__builtin_cpu_supports` (x86) or `/proc/cpuinfo` parsing (ARM)
- Function pointer dispatch table for dynamic backend selection
- Build with CMake for multi-target support
- Benchmarking: measure throughput (MP/sec) on representative test suite
- Validate pixel-perfect output across all implementations

---

## Week 3: RISC-V Vector Extensions (Days 15-21)

### Day 15: RISC-V ISA Fundamentals
- RISC-V base instruction sets (RV32I, RV64I)
- Modular ISA design philosophy
- Standard extensions (M, A, F, D, C)
- Privilege levels and CSRs
- **Lab:** Emulate RISC-V with QEMU

### Day 16: RISC-V Vector Extension (RVV) v1.0
- Vector register groups and LMUL
- Vector length (VLEN) configurability
- vsetvl (vector length configuration)
- Element width polymorphism
- **Lab:** Write first RVV assembly program

### Day 17: RVV Intrinsics & C Programming
- RISC-V Vector C Intrinsics (rvv-intrinsic)
- Masked operations
- Reduction operations
- Segment load/store
- **Lab:** Implement parallel reduction with RVV

### Day 18: Performance Optimization on RISC-V
- Register pressure and spilling
- Chaining and instruction fusion
- Memory bandwidth considerations
- Comparing with ARM SVE
- **Lab:** Optimize matrix transpose

### Day 19: Custom RISC-V Extensions
- Adding custom instructions
- Rocket Chip and BOOM cores
- Chisel HDL for ISA extension
- Software toolchain integration
- **Lab:** Design a simple custom instruction

### Day 20: RISC-V

 Ecosystem
- RISC-V compilers (GCC, LLVM)
- SiFive boards and simulators
- OpenSBI and bootloader
- Linux on RISC-V
- **Lab:** Boot Linux on RISC-V simulator

### Day 21: Week 3 Review & Project
- Project: Vectorized FFT implementation on RISC-V
- Use RVV intrinsics
- Compare with x86 AVX implementation
- Cross-architecture performance analysis

---

## Week 4: OpenMP & Shared-Memory Parallelism (Days 22-28)

### Day 22: OpenMP Basics
- Fork-join parallelism model
- `#pragma omp parallel`, `for`, `sections`
- Thread creation and management
- Environment variables (`OMP_NUM_THREADS`)
- **Lab:** Parallelize loop with OpenMP

### Day 23: OpenMP Work-Sharing Constructs
- `schedule` clause (static, dynamic, guided)
- `nowait` and explicit barriers
- `single` and `master` directives
- Nested parallelism
- **Lab:** Dynamic scheduling for load imbalance

### Day 24: OpenMP Data Environment
- Private vs shared variables
- `firstprivate`, `lastprivate`, `reduction`
- Data races and synchronization
- Atomic operations and critical sections
- **Lab:** Debug race condition with Thread Sanitizer

### Day 25: OpenMP SIMD Directives
- `#pragma omp simd`
- `simdlen` and alignment clauses
- Loop transformations (fusion, distribution)
- Interplay with compiler auto-vectorization
- **Lab:** Combine OpenMP threading + SIMD

### Day 26: OpenMP Tasking
- `task` and `taskwait` directives
- Task dependencies (`depend` clause)
- Taskloop for recursive algorithms
- Load balancing with tasks
- **Lab:** Parallel quicksort with tasks

### Day 27: OpenMP Offloading (Target)
- `target` directive for accelerators
- Data mapping (`map` clause)
- Device management
- Unified shared memory
- **Lab:** Offload computation to GPU with OpenMP

### Day 28: Week 4 Review & Project
- Project: Parallel N-body simulation
- Use OpenMP for threading and SIMD
- Implement Barnes-Hut tree algorithm with tasks
- Performance profiling and scaling analysis

---

## Week 5: OpenCL & Heterogeneous Computing (Days 29-35)

### Day 29: OpenCL Architecture
- Platform model (Host, Devices, Compute Units)
- Execution model (Kernels, Work-Items, Work-Groups)
- Memory model (Global, Local, Private, Constant)
- OpenCL vs CUDA comparison
- **Lab:** Query OpenCL devices

### Day 30: OpenCL Kernel Programming
- Kernel language (C99-based)
- `__global`, `__local`, `__private` qualifiers
- Built-in functions (math, vector, atomic)
- Kernel compilation at runtime
- **Lab:** Write first OpenCL kernel (vector add)

### Day 31: OpenCL Memory Management
- Buffer objects and images
- Memory transfer patterns
- Pinned memory (CL_MEM_ALLOC_HOST_PTR)
- Sub-buffers and memory regions
- **Lab:** Optimize host-device transfers

### Day 32: OpenCL Synchronization
- Command queues and events
- In-order vs out-of-order queues
- Event dependencies
- Explicit barriers and fences
- **Lab:** Pipeline overlapping compute and transfer

### Day 33: OpenCL Performance Optimization
- Occupancy and work-group sizing
- Local memory (shared memory) usage
- Coalesced memory access
- Vectorization (`float4`, `int8`)
- **Lab:** Optimize matrix multiplication kernel

### Day 34: SYCL - C++ Abstraction for OpenCL
- SYCL programming model
- Single-source compilation
- Buffers, accessors, and queues
- Parallel_for and reductions
- **Lab:** Port OpenCL code to SYCL

### Day 35: Week 5 Review & Project
- Project: Multi-device image processing pipeline
- Use OpenCL to target CPU + multiple GPUs
- Implement sobel edge detection
- Load balancing across heterogeneous devices

---

## Week 6: Intel oneAPI & DPC++ (Days 36-42)

### Day 36: oneAPI Ecosystem
- Base Toolkit and specialized toolkits
- DPC++ (Data Parallel C++)
- oneDNN, oneMKL, oneTBB, oneDAL
- Cross-architecture portability
- **Lab:** Install oneAPI toolkit

### Day 37: DPC++ Programming Model
- SYCL 2020 compliance
- Device selectors and platforms
- Unified shared memory (USM)
- Parallel kernels and hierarchical parallelism
- **Lab:** Hello world DPC++ program

### Day 38: oneDNN (Deep Neural Networks)
- Primitives for convolution, pooling, ReLU
- Memory formats and reorders
- Integration with frameworks
- Performance on Intel GPUs
- **Lab:** Run CNN inference with oneDNN

### Day 39: oneMKL (Math Kernel Library)
- BLAS/LAPACK/FFT on accelerators
- Batched operations
- Sparse linear algebra
- Random number generation
- **Lab:** GPU-accelerated GEMM with oneMKL

### Day 40: Intel GPUs & Xe Architecture
- Xe-LP, Xe-HPG, Xe-HPC
- EU (Execution Units) architecture
- Intel Arc and Data Center Max GPUs
- Comparison with NVIDIA/AMD
- **Lab:** Profile workload on Intel GPU

### Day 41: Advanced DPC++ Features
- Subgroups and group algorithms
- Specialization constants
- Device code split
- Explicit SIMD (`sycl::ext::intel::esimd`)
- **Lab:** Write eSIMD kernel for fine-grained control

### Day 42: Week 6 Review & Project
- Project: Portable HPC application with oneAPI
- Implement conjugate gradient solver
- Run on CPU (AVX-512), NVIDIA GPU (via OpenCL backend), Intel GPU
- Performance comparison report

---

## Phase 7B: Multi-GPU & Advanced Acceleration (Weeks 7-12)

---

## Week 7: AMD ROCm & HIP (Days 43-49)

### Day 43: AMD GPU Architecture
- RDNA vs CDNA architectures
- Compute Units (CU) and Wavefronts
- GCN instruction set
- MI-series GPUs for HPC
- **Lab:** Inspect AMD GPU with `rocm-smi`

### Day 44: HIP Programming Model
- HIP API (similar to CUDA)
- hipMalloc, hipMemcpy, hipLaunchKernel
- CUDA-to-HIP porting (hipify tools)
- Kernel syntax differences
- **Lab:** Port CUDA vector add to HIP

### Day 45: ROCm Software Stack
- ROCm runtime and drivers
- ROCm libraries (rocBLAS, rocFFT, MIOpen)
- ROCgdb debugger
- Radeon GPU Profiler
- **Lab:** Profile HIP kernel with rocprof

### Day 46: Performance Tuning on AMD GPUs
- Wavefront execution and occupancy
- LDS (Local Data Share) optimization
- Global memory coalescing
- Utilizing high-bandwidth HBM
- **Lab:** Optimize matmul for MI100/MI250

### Day 47: Multi-GPU with ROCm
- Peer-to-peer transfers
- RCCL (ROCm Collective Communication Library)
- Multi-GPU workload distribution
- Infinity Fabric interconnect
- **Lab:** Multi-GPU reduction with RCCL

### Day 48: Machine Learning on AMD
- MIOpen vs cuDNN
- ROCm for PyTorch
- TensorFlow-ROCm
- Triton compiler for AMD
- **Lab:** Train model on AMD GPU

### Day 49: Week 7 Review & Project
- Project: Cross-vendor GPU framework
- Same codebase for NVIDIA (CUDA) and AMD (HIP)
- Implement 2D FFT
- Performance parity analysis

---

## Week 8: Apple Metal & GPU Compute (Days 50-56)

### Day 50: Metal Architecture
- Metal API overview
- GPU families (Apple7, Apple8, Apple9)
- Metal Shading Language (MSL)
- Unified memory on Apple Silicon
- **Lab:** Create first Metal compute pipeline

### Day 51: Metal Compute Kernels
- Kernel functions and attributes
- Threadgroups and threadgroup memory
- Texture sampling in compute
- Atomic operations
- **Lab:** Parallel prefix sum in Metal

### Day 52: Metal Performance Shaders (MPS)
- Pre-built primitives (convolution, pooling)
- MPS Graph API
- Neural network operations
- Image processing filters
- **Lab:** Build CNN with MPS Graph

### Day 53: Metal for Machine Learning
- Core ML integration
- ANE (Apple Neural Engine) deployment
- MPS backend in PyTorch
- TensorFlow Metal plugin
- **Lab:** Run inference on ANE

### Day 54: GPU-Driven Rendering Pipelines
- Compute-based culling
- GPU particle systems
- Ray tracing with Metal
- Indirect command buffers
- **Lab:** Build compute-driven renderer

### Day 55: Debugging & Profiling Metal
- Xcode GPU debugger
- Metal System Trace
- Shader validation
- GPU frame capture
- **Lab:** Optimize Metal kernel with Instruments

### Day 56: Week 8 Review & Project
- Project: Unified compute library for macOS/iOS
- Implement parallel sorting network
- Use Metal for GPU, Accelerate for CPU
- Benchmark on M1/M2/M3

---

## Week 9: cuML & RAPIDS Ecosystem (Days 57-63)

### Day 57: RAPIDS Overview
- cuDF (GPU DataFrames)
- cuML (GPU Machine Learning)
- cuGraph (GPU Graph Analytics)
- cuSpatial (Geospatial)
- **Lab:** Install RAPIDS conda environment

### Day 58: cuML Algorithms
- K-Means, DBSCAN, UMAP
- Random Forest, XGBoost
- Linear regression, LogisticRegression
- GPU-accelerated scikit-learn API
- **Lab:** Train classifier with cuML

### Day 59: cuDF for Data Processing
- GPU-accelerated Pandas operations
- Reading Parquet/CSV on GPU
- Joins and aggregations
- Dask-cuDF for distributed processing
- **Lab:** ETL pipeline with cuDF

### Day 60: cuGraph for Network Analysis
- PageRank, Louvain, BFS/DFS
- Triangle counting
- Jaccard similarity
- Multi-GPU graph analytics
- **Lab:** Analyze social network with cuGraph

### Day 61: Integration with Deep Learning
- PyTorch + cuDF dataloaders
- Feature engineering on GPU
- End-to-end ML pipeline
- Hyperparameter tuning with Optuna-GPU
- **Lab:** Complete ML workflow

### Day 62: Custom CUDA Kernels for cuML
- Extending cuML with custom ops
- CUDA kernel integration
- JIT compilation with Numba
- CuPy interoperability
- **Lab:** Add custom distance metric

### Day 63: Week 9 Review & Project
- Project: GPU-accelerated recommender system
- Use cuML for collaborative filtering
- cuDF for data preprocessing
- cuGraph for similarity computation
- Benchmark vs CPU Spark

---

## Week 10: Parallel Algorithms & Patterns (Days 64-70)

### Day 64: Map-Reduce Patterns
- Parallel map implementation
- Tree-based reduction
- Scan (prefix sum) algorithms
- Applications in data analytics
- **Lab:** Implement parallel histogram

### Day 65: Sorting Algorithms
- Bitonic sort on GPU
- Radix sort implementation
- Thrust library (sort, stable_sort)
- Sorting networks
- **Lab:** GPU vs CPU sort benchmarks

### Day 66: Graph Algorithms
- BFS/DFS parallelization
- Parallel shortest path (Bellman-Ford)
- Connected components
- Graph coloring
- **Lab:** Parallel graph traversal

### Day 67: Stencil Computations
- Finite difference methods
- Halo exchange in MPI
- GPU implementation strategies
- 2D/3D convolution optimization
- **Lab:** Heat equation solver

### Day 68: Dynamic Programming
- Parallel recurrence relations
- Smith-Waterman sequence alignment
- Matrix chain multiplication
- Needleman-Wunsch algorithm
- **Lab:** GPU-accelerated sequence alignment

### Day 69: Load Balancing Techniques
- Work stealing
- Dynamic scheduling
- Atomic counters for work queues
- Irregular workload distribution
- **Lab:** Parallel tree traversal with load balancing

### Day 70: Week 10 Review & Project
- Project: Parallel computational geometry
- Convex hull (parallel Graham scan)
- Delaunay triangulation
- Voronoi diagrams
- GPU vs multi-core CPU comparison

---

## Week 11: Distributed Memory & MPI (Days 71-77)

### Day 71: MPI Fundamentals
- Message-passing model
- MPI_Init, MPI_Finalize
- Communicators and ranks
- Point-to-point communication
- **Lab:** Hello MPI world

### Day 72: Collective Operations
- MPI_Bcast, MPI_Scatter, MPI_Gather
- MPI_Reduce and MPI_Allreduce
- MPI_Barrier synchronization
- Tree-based algorithms
- **Lab:** Parallel summation with MPI

### Day 73: Non-Blocking Communication
- MPI_Isend and MPI_Irecv
- MPI_Wait and MPI_Test
- Overlapping compute and communication
- Persistent communication
- **Lab:** Pipelined data transfer

### Day 74: Derived Datatypes
- MPI_Type_contiguous, vector, indexed
- Struct packing
- Subarray types
- Performance benefits
- **Lab:** Transfer 2D matrix slices

### Day 75: MPI + CUDA/HIP
- GPU-aware MPI
- CUDA-aware MPI (MVAPICH2-GDR)
- Direct GPU-to-GPU transfers
- UCX (Unified Communication X)
- **Lab:** Multi-node multi-GPU GEMM

### Day 76: One-Sided Communication
- MPI-3 RMA (Remote Memory Access)
- MPI_Put, MPI_Get, MPI_Accumulate
- Active vs passive target synchronization
- PGAS (Partitioned Global Address Space)
- **Lab:** Distributed hash table with RMA

### Day 77: Week 11 Review & Project
- Project: Distributed parallel sorting
- Implement sample sort
- Use MPI for inter-node communication
- CUDA for intra-node sorting
- Scalability study up to 64 nodes

---

## Week 12: Advanced GPU Topics (Days 78-84)

### Day 78: GPU Multi-Process Service (MPS)
- MPS architecture
- Concurrent kernel execution from multiple processes
- Use cases (containers, multi-tenant)
- Limitations and best practices
- **Lab:** Run multiple workloads with MPS

### Day 79: CUDA Graphs
- Stream capture API
- Explicit graph construction
- Conditional nodes
- Reducing launch overhead
- **Lab:** Build and execute CUDA graph

### Day 80: Cooperative Groups
- Block-level, grid-level cooperation
- Multi-device grids
- Dynamic parallelism replacement
- Warp-level primitives
- **Lab:** Grid-wide synchronization

### Day 81: CUDA Unified Memory Advanced
- Prefetching and mem_advise
- Oversubscription handling
- Access counters
- Page migration
- **Lab:** Optimize UM with hints

### Day 82: Tensor Cores & WMMA API
- Warp matrix multiply-accumulate
- WMMA fragments (A, B, C)
- Mixed-precision computation (FP16, BF16, TF32)
- Integration with cuBLAS
- **Lab:** Implement matmul with Tensor Cores

### Day 83: Quantum Computing Simulation on GPUs
- Statevector simulation
- GPU-accelerated Qiskit
- cuQuantum SDK
- Tensor network contractions
- **Lab:** Simulate quantum circuit

### Day 84: Week 12 Review & Project
- Project: High-performance linear algebra library
- Support CPU (OpenBLAS), NVIDIA (CUBLAS), AMD (rocBLAS)
- GEMM batching and fusion
- Benchmark against vendor libraries

---

## Phase 7C: Compiler Foundations (Weeks 13-18)

---

## Week 13: Compiler Architecture (Days 85-91)

### Day 85: Compiler Phases Overview
- Lexical analysis (Scanner)
- Syntax analysis (Parser)
- Semantic analysis
- Intermediate representation (IR)
- Optimization passes
- Code generation
- **Lab:** Trace compilation with GCC `-v` flag

### Day 86: Lexical Analysis
- Regular expressions and finite automata
- Token generation
- Flex/Lex scanner generator
- Handling keywords and identifiers
- **Lab:** Build lexer for simple language

### Day 87: Parsing Techniques
- Context-free grammars (CFG)
- LL vs LR parsing
- Recursive descent parser
- Yacc/Bison parser generator
- **Lab:** Write parser for arithmetic expressions

### Day 88: Abstract Syntax Trees (AST)
- AST construction
- Visitor pattern for tree traversal
- Symbol table management
- Type checking
- **Lab:** Build AST for expression language

### Day 89: Intermediate Representations
- Three-address code (TAC)
- Static Single Assignment (SSA)
- Control Flow Graph (CFG)
- Data Flow Analysis
- **Lab:** Convert AST to TAC

### Day 90: Optimization Techniques
- Constant folding/propagation
- Dead code elimination
- Common subexpression elimination (CSE)
- Loop invariant code motion
- **Lab:** Implement simple optimizer

### Day 91: Week 13 Review & Project
- Project: Mini-compiler for subset of C
- Lexer, parser, AST, TAC generation
- Basic optimizations
- Interpret TAC or emit x86 assembly

---

## Week 14: LLVM Infrastructure (Days 92-98)

### Day 92: LLVM Architecture
- LLVM toolchain components
- Frontend (Clang), Middle-end, Backend
- LLVM IR (bitcode)
- Pass infrastructure
- **Lab:** Inspect LLVM IR with `clang -S -emit-llvm`

### Day 93: LLVM IR Deep Dive
- SSA form in LLVM
- Basic blocks and terminators
- Instruction set (arithmetic, memory, control flow)
- Metadata and attributes
- **Lab:** Write LLVM IR by hand

### Day 94: LLVM C++ API
- Module, Function, BasicBlock classes
- IRBuilder for instruction generation
- Value and Type hierarchy
- Creating functions programmatically
- **Lab:** Generate LLVM IR from C++ program

### Day 95: LLVM Passes
- Analysis passes vs transformation passes
- Pass manager and legacy PM
- New Pass Manager (NPM)
- Writing custom passes
- **Lab:** Write pass to count instructions

### Day 96: Optimization Passes
- mem2reg (promote allocas to registers)
- SCCP (Sparse Conditional Constant Propagation)
- LICM (Loop Invariant Code Motion)
- Inlining, vectorization, unrolling
- **Lab:** Enable and analyze opt passes

### Day 97: LLVM Backends
- Target architecture description
- Instruction selection (DAG-to-DAG)
- Register allocation
- Instruction scheduling
- **Lab:** Examine backend for x86-64

### Day 98: Week 14 Review & Project
- Project: Custom LLVM pass for optimization
- Implement strength reduction pass
- Test with Clang on benchmarks
- Measure performance impact

---

## Week 15: LLVM Advanced (Days 99-105)

### Day 99: TableGen & Target Description
- TableGen language syntax
- Defining instructions and registers
- Instruction patterns and selection
- Schedule models
- **Lab:** Define custom instruction in TableGen

### Day 100: LLVM JIT Compilation
- ORC JIT APIs
- Lazy compilation
- LLJIT and ExecutionSession
- Runtime code generation
- **Lab:** Build simple JIT compiler

### Day 101: LLVM for Domain-Specific Languages
- Kaleidoscope tutorial revisited
- Lexer/Parser for custom language
- Code generation to LLVM IR
- Adding JIT support
- **Lab:** Extend Kaleidoscope with arrays

### Day 102: Link-Time Optimization (LTO)
- Whole-program optimization
- ThinLTO for scalability
- Interprocedural analysis
- Devirtualization
- **Lab:** Measure LTO impact on real codebase

### Day 103: LLVM Sanitizers
- AddressSanitizer (ASan)
- MemorySanitizer (MSan)
- ThreadSanitizer (TSan)
- UndefinedBehaviorSanitizer (UBSan)
- **Lab:** Debug memory bugs with ASan

### Day 104: Polly - Polyhedral Optimizer
- Polyhedral compilation model
- Loop nest optimization
- Tiling and parallelization
- Integration with LLVM pipeline
- **Lab:** Optimize stencil code with Polly

### Day 105: Week 15 Review & Project
- Project: Implement simple scripting language
- Lexer, parser, AST, LLVM IR emission
- JIT execution
- Interactive REPL
- Performance benchmarks

---

## Week 16: Auto-Vectorization & Loop Optimization (Days 106-112)

### Day 106: Loop Analysis
- Dependence analysis
- Loop-carried dependencies
- Distance and direction vectors
- Affine transformations
- **Lab:** Analyze loop with polyhedral tools

### Day 107: Loop Transformations
- Loop interchange
- Loop fusion and fission
- Loop tiling (blocking)
- Loop unrolling
- **Lab:** Manually apply transformations

### Day 108: Auto-Vectorization in Compilers
- SLP (Superword-Level Parallelism)
- Loop vectorization
- Cost models
- Aliasing and alignment
- **Lab:** Enable auto-vectorization with pragmas

### Day 109: LLVM Loop Vectorizer
- LoopVectorize pass
- Reduction detection
- Interleaving
- Vector width selection
- **Lab:** Debug vectorization failures with `-Rpass`

### Day 110: Custom Vectorization Hints
- `#pragma clang loop vectorize`
- `__builtin_assume_aligned`
- Restrict pointers
- Loop metadata
- **Lab:** Guide compiler with hints

### Day 111: Multi-Version Function Dispatching
- Function multi-versioning (FMV)
- Target clones attribute
- IFUNC (indirect functions)
- Runtime CPU feature detection
- **Lab:** Build library with multi-versioned GEMM

### Day 112: Week 16 Review & Project
- Project: Matrix library with optimized kernels
- Implement transpose, GEMM, GEMV
- Compare scalar, auto-vectorized, intrinsics
- Profile and document optimization journey

---

## Week 17: GCC Internals (Days 113-119)

### Day 113: GCC Architecture
- Frontend, GIMPLE, RTL, Backend
- Compilation pipeline
- Plugin system
- GCC vs LLVM comparison
- **Lab:** Dump GCC intermediate representations

### Day 114: GIMPLE Intermediate Representation
- GIMPLE vs LLVM IR
- GIMPLE passes
- SSA in GCC
- Tree-level optimizations
- **Lab:** Write GCC GIMPLE pass

### Day 115: RTL (Register Transfer Language)
- RTL structure and semantics
- Machine description files (.md)
- Constraints and patterns
- Register allocation in GCC
- **Lab:** Examine RTL for target architecture

### Day 116: GCC Optimization Levels
- -O0, -O1, -O2, -O3, -Ofast, -Os
- Per-function optimization attributes
- Profile-guided optimization (PGO)
- Link-time optimization (-flto)
- **Lab:** Benchmark optimization levels

### Day 117: GCC Auto-Vectorization
- Tree vectorizer
- SLP vectorizer
- Vectorization cost model
- Alignment and aliasing hints
- **Lab:** Compare GCC vs Clang vectorization

### Day 118: GCC for Embedded Systems
- Cross-compilation
- Newlib and minimal C library
- Size optimizations (-Os, -Oz)
- Code generation for microcontrollers
- **Lab:** Build bare-metal ARM program with GCC

### Day 119: Week 17 Review & Project
- Project: Contribute to GCC
- Write test case for bug
- Develop optimization pass
- Submit patch to mailing list

---

## Week 18: Polyhedral Compilation (Days 120-126)

### Day 120: Polyhedral Model Foundations
- Integer polyhedra
- Affine loop nests
- Iteration domains
- Dependence polyhedra
- **Lab:** Represent loop nest as polyhedron

### Day 121: Polyhedral Tools
- ISL (Integer Set Library)
- Pluto automatic  parallelizer
- PoCC (Polyhedral Compiler Collection)
- CLooG code generator
- **Lab:** Use ISL API for set operations

### Day 122: Scheduling Algorithms
- Feautrier's algorithm
- Pluto algorithm (tiling for locality)
- Affine schedules
- Legality constraints
- **Lab:** Compute optimal schedule

### Day 123: Tiling & Locality Optimization
- Cache-oblivious algorithms
- Multi-level tiling
- Parametric tiling
- Code generation for tiled loops
- **Lab:** Tile matrix multiplication

### Day 124: Parallelization with Polyhedral Model
- Extracting parallel loops
- DOALL detection
- Wavefront parallelism
- GPU code generation from polyhedral
- **Lab:** Generate OpenMP from Pluto

### Day 125: Tensor Compiler Integration
- TVM's TE (Tensor Expression)
- Halide scheduling
- MLIR affine dialect
- Polyhedral loop fusion
- **Lab:** Optimize convolution with Halide

### Day 126: Week 18 Review & Project
- Project: Stencil compiler
- Parse stencil specification
- Apply polyhedral optimizations
- Generate tiled + vectorized code
- Benchmark against manual code

---

## Phase 7D: Hardware & ISA Design (Weeks 19-24)

---

## Week 19: Digital Logic & RTL Design (Days 127-133)

### Day 127: Digital Logic Fundamentals
- Boolean algebra and logic gates
- Combinational vs sequential circuits
- Flip-flops and registers
- Finite state machines
- **Lab:** Simulate circuits with Logisim

### Day 128: Hardware Description Languages
- Verilog vs VHDL vs SystemVerilog
- Synthesizable vs non-synthesizable code
- Always blocks and sensitivity lists
- Testbenches
- **Lab:** Write first Verilog module

### Day 129: Register-Transfer Level (RTL)
- RTL design methodology
- Datapath and control unit
- Pipeline registers
- State machine encoding
- **Lab:** Design simple ALU in Verilog

### Day 130: Verification & Simulation
- Testbench development
- Waveform analysis
- Code coverage
- Formal verification basics
- **Lab:** Verify ALU with testbench

### Day 131: Synthesis & Timing
- Logic synthesis tools (Yosys, Synplify)
- Gate-level netlist
- Timing constraints
- Setup and hold time
- **Lab:** Synthesize design for FPGA

### Day 132: FPGA Architecture
- LUTs, flip-flops, BRAMs, DSP blocks
- Xilinx vs Intel (Altera) FPGAs
- Reconfigurability
- Partial reconfiguration
- **Lab:** Deploy design on FPGA board

### Day 133: Week 19 Review & Project
- Project: RISC-V core subset in Verilog
- Implement RV32I base ISA (minimal)
- Testbench with instruction tests
- Synthesize and run on FPGA

---

## Week 20: Processor Microarchitecture (Days 134-140)

### Day 134: Pipelining Fundamentals
- 5-stage RISC pipeline (IF, ID, EX, MEM, WB)
- Data hazards (RAW, WAR, WAW)
- Forwarding and stalling
- Branch prediction and flushing
- **Lab:** Simulate pipelined processor

### Day 135: Out-of-Order Execution
- Tomasulo's algorithm
- Reservation stations
- Reorder buffer (ROB)
- Register renaming
- **Lab:** Study OoO with gem5 simulator

### Day 136: Branch Prediction
- Static vs dynamic prediction
- Saturating counters
- BTB (Branch Target Buffer)
- Two-level adaptive predictors
- **Lab:** Implement branch predictor

### Day 137: Cache Design
- Direct-mapped, set-associative, fully-associative
- Replacement policies (LRU, FIFO, Random)
- Write policies (write-back, write-through)
- Cache coherence (MSI, MESI, MOESI)
- **Lab:** Simulate cache with CacheSim

### Day 138: Superscalar & VLIW
- Instruction-level parallelism (ILP)
- Multiple issue
- VLIW (Very Long Instruction Word)
- EPIC (Itanium architecture)
- **Lab:** Analyze superscalar scheduling

### Day 139: Memory Systems
- Virtual memory and TLBs
- DRAM organization (banks, rows, columns)
- Memory controllers
- Prefetching techniques
- **Lab:** Profile memory latency with STREAM

### Day 140: Week 20 Review & Project
- Project: Performance simulator
- Build cycle-accurate simulator for RISC-V subset
- Model pipeline hazards and branch prediction
- Measure IPC on benchmarks

---

## Week 21: RISC-V Processor Design (Days 141-147)

### Day 141: RISC-V Open ISA
- RISC-V Foundation and governance
- ISA modularity and extensions
- Privileged architecture
- Ecosystem (tools, cores, boards)
- **Lab:** Study RISC-V ISA manual

### Day 142: Single-Cycle RISC-V Core
- Datapath for RV32I
- Control unit design
- Instruction fetch and decode
- ALU operations
- **Lab:** Implement single-cycle core in Verilog

### Day 143: Pipelined RISC-V Core
- 5-stage pipeline for RV32I
- Hazard detection and forwarding
- Branch handling
- Performance analysis
- **Lab:** Extend to pipelined core

### Day 144: Adding M Extension (Multiply/Divide)
- Hardware multiplier designs
- Division algorithms
- Integration into pipeline
- Multicycle operations
- **Lab:** Add M extension instructions

### Day 145: Privilege Modes & CSRs
- Machine, Supervisor, User modes
- Control and Status Registers
- Exception and interrupt handling
- Trap vectors
- **Lab:** Implement basic CSR operations

### Day 146: RISC-V Compliance Tests
- riscv-tests suite
- Compliance test framework
- Debugging failures
- Coverage analysis
- **Lab:** Run compliance tests on core

### Day 147: Week 21 Review & Project
- Project: Complete RV32IM core
- Support base + multiply/divide
- Pass compliance tests
- Synthesize for FPGA
- Boot simple bare-metal program

---

## Week 22: Chisel & Hardware Construction (Days 148-154)

### Day 148: Chisel Basics
- Scala-embedded HDL
- Hardware types (UInt, SInt, Bool)
- Modules and IOs
- Combinational vs sequential logic
- **Lab:** Hello world in Chisel

### Day 149: Chisel Combinators
- Mux, Cat, Fill
- Vec (vectors) and Bundle
- Switch/when statements
- Functional construction
- **Lab:** Parameterized adder tree

### Day 150: Sequential Circuits in Chisel
- RegNext and registers
- Counters and shift registers
- State machines with Enum
- Memory (Mem, SyncReadMem)
- **Lab:** FIFO queue in Chisel

### Day 151: Chisel Testing
- ChiselTest framework
- peek, poke, step
- expect for assertions
- Waveform generation
- **Lab:** Write tests for FIFO

### Day 152: Rocket Chip Generator
- TileLink interconnect
- Rocket core architecture
- Generator parameters
- SoC integration
- **Lab:** Generate Rocket core

### Day 153: BOOM (Berkeley Out-of-Order Machine)
- Out-of-order RISC-V core
- Superscalar capabilities
- Configuration options
- Performance analysis
- **Lab:** Generate and simulate BOOM

### Day 154: Week 22 Review & Project
- Project: Custom accelerator in Chisel
- Design fixed-point matrix multiply unit
- Integrate with Rocket core as RoCC
- Test with bare-metal C program

---

## Week 23: Custom ISA Extensions (Days 155-161)

### Day 155: RISC-V Custom Instructions
- Custom opcode space
- Encoding formats
- RoCC (Rocket Custom Coprocessor) interface
- Software integration
- **Lab:** Define custom instruction encoding

### Day 156: Implementing Custom ALU Operations
- Extend ALU with new operations
- Decoding custom opcodes
- Register file access
- Writeback integration
- **Lab:** Add population count (popcount) instruction

### Day 157: Vector Coprocessors
- Decoupled vector unit
- VLEN configuration
- Element-wise operations
- Reduction units
- **Lab:** Simple vector add unit

### Day 158: Domain-Specific Accelerators
- Cryptographic accelerators (AES, SHA)
- FFT accelerator
- Matrix multiply unit
- Tight coupling vs loosely coupled
- **Lab:** Implement AES S-box lookup accelerator

### Day 159: Software Toolchain Integration
- Binutils support (assembler, linker)
- GCC intrinsics
- LLVM backend modifications
- Compiler test suite
- **Lab:** Add intrinsic for custom instruction

### Day 160: Performance Evaluation
- Benchmarking custom instructions
- Area/power/performance tradeoffs
- Synthesis results
- Comparing with software implementation
- **Lab:** Measure speedup of accelerated code

### Day 161: Week 23 Review & Project
- Project: Cryptography accelerator
- Implement AES encryption engine
- Custom RISC-V instructions for AES rounds
- Integrate with core and test
- Performance vs software AES

---

## Week 24: SoC Design & Integration (Days 162-168)

### Day 162: System-on-Chip Architecture
- CPU, memory, peripherals, interconnect
- Bus standards (AXI, AHB, APB)
- DMA controllers
- Interrupt controllers
- **Lab:** Design simple SoC block diagram

### Day 163: Memory Subsystem
- On-chip SRAM
- External DRAM interface
- Cache hierarchies
- Memory-mapped I/O
- **Lab:** Implement simple memory controller

### Day 164: Peripheral Integration
- UART, SPI, I2C controllers
- GPIO (General Purpose I/O)
- Timers and watchdogs
- Device tree for configuration
- **Lab:** Add UART to RISC-V SoC

### Day 165: Interconnect Fabrics
- Crossbar switches
- Network-on-Chip (NoC)
- TileLink protocol
- AXI4 protocol
- **Lab:** Implement simple crossbar

### Day 166: Bootloader & Software Stack
- Boot ROM
- First-stage bootloader (FSBL)
- U-Boot integration
- Loading Linux kernel
- **Lab:** Boot Linux on FPGA SoC

### Day 167: Power Management
- Clock gating
- Power gating (DVFS)
- Sleep modes
- Power domains
- **Lab:** Add clock gating to modules

### Day 168: Week 24 Review & Project
- Project: Complete RISC-V SoC
- Rocket/BOOM core + peripherals
- Bootable Linux SoC
- Deploy on FPGA
- Run benchmarks and applications

---

## Phase 7E: Advanced Topics & Specialization (Weeks 25-30)

---

## Week 25: Tensor Compilers & ML Infrastructure (Days 169-175)

### Day 169: TVM Deep Dive
- Relay (high-level IR)
- Tensor Expression (TE)
- Scheduling primitives
- AutoTVM/AutoScheduler
- **Lab:** Optimize conv2d with TVM

### Day 170: XLA (Accelerated Linear Algebra)
- HLO (High-Level Optimizer)
- Fusion and layout optimization
- Target-specific code generation
- JAX integration
- **Lab:** Compile JAX program with XLA

### Day 171: MLIR (Multi-Level IR)
- Dialects (Affine, Linalg, Vector, LLVM)
- Progressive lowering
- Pass infrastructure
- Python bindings
- **Lab:** Write custom MLIR dialect

### Day 172: Glow Compiler
- Graph-level optimization
- Quantization
- Backend for ARM/Intel
- Integration with PyTorch
- **Lab:** Compile model with Glow

### Day 173: OpenAI Triton
- Python DSL for GPU kernels
- Automatic tiling and caching
- Kernel fusion
- Benchmarking vs cuDNN/CUTLASS
- **Lab:** Write fused attention kernel

### Day 174: CUTLASS (CUDA Templates for Linear Algebra)
- Tile iterators and layouts
- GEMM kernel templates
- Epilogue fusion
- Ampere/Hopper optimization
- **Lab:** Customize CUTLASS GEMM

### Day 175: Week 25 Review & Project
- Project: End-to-end model compiler
- Support PyTorch/TensorFlow input
- Generate code for CPU (x86), NVIDIA, AMD
- Benchmark latency and accuracy

---

## Week 26: Specialized Accelerators (Days 176-182)

### Day 176: Google TPU Architecture
- Systolic array design
- Matrix multiply unit (MXU)
- TPUv1 vs v2 vs v3 vs v4
- XLA for TPU
- **Lab:** Simulate systolic array in Python

### Day 177: AWS Inferentia & Trainium
- NeuronCore architecture
- Neuron SDK and compiler
- Mixed precision and quantization
- Distributed training on Trainium
- **Lab:** Deploy model on Inferentia

### Day 178: Cerebras Wafer-Scale Engine
- Largest chip (WSE-2: 850,000 cores)
- 2D mesh interconnect
- Memory-near-compute
- Sparsity support
- **Lab:** Research Cerebras architecture

### Day 179: Graphcore IPU
- MIMD architecture (1,472 cores)
- Bulk Synchronous Parallel (BSP)
- PopART framework
- Graph partitioning
- **Lab:** Run model on IPU simulator

### Day 180: SambaNova Reconfigurable Dataflow
- Dataflow architecture
- SambaFlow SDK
- Pipelined parallelism
- Variable precision
- **Lab:** Study dataflow execution model

### Day 181: Intel Habana Gaudi
- Gaudi-2 for training
- GEMM and TPC (Tensor Processing Core)
- Synapse AI framework
- RDMA over converged Ethernet (RoCE)
- **Lab:** Profile workload on Gaudi

### Day 182: Week 26 Review & Project
- Project: Accelerator comparison study
- Same model on: CPU, NVIDIA, AMD, TPU, Inferentia
- Latency, throughput, cost, power analysis
- Document architecture tradeoffs

---

## Week 27: Neuromorphic & Quantum Computing (Days 183-189)

### Day 183: Neuromorphic Computing
- Spiking neural networks (SNNs)
- Event-based computation
- Intel Loihi chip
- IBM TrueNorth
- **Lab:** Simulate SNN with Brian2

### Day 184: Quantum Computing Basics
- Qubits and superposition
- Quantum gates (Hadamard, CNOT, Toffoli)
- Quantum algorithms (Grover, Shor)
- Error correction
- **Lab:** Run circuit on Qiskit simulator

### Day 185: Quantum Hardware Platforms
- Superconducting qubits (IBM, Google)
- Ion trap (IonQ, Honeywell)
- Photonic (Xanadu)
- Neutral atom (QuEra)
- **Lab:** Access IBM Quantum cloud

### Day 186: Quantum Simulation on GPUs
- Statevector representation
- Tensor network methods
- cuQuantum library
- Distributed quantum simulation
- **Lab:** Accelerate Qiskit with cuQuantum

### Day 187: Analog Computing
- Memristors and analog matrix multiply
- Mythic AI analog inference
- Optical neural networks
- Hybrid analog-digital systems
- **Lab:** Research analog computing papers

### Day 188: Processing-In-Memory (PIM)
- Samsung HBM-PIM
- UPMEM DPU
- Near-data processing
- Bandwidth-bound applications
- **Lab:** Study PIM programming model

### Day 189: Week 27 Review & Project
- Project: Future architecture survey
- Compare neuromorphic, quantum, analog, PIM
- Assess suitability for ML workloads
- Roadmap predictions (5-10 years)

---

## Week 28: Performance Engineering (Days 190-196)

### Day 190: Performance Measurement
- Hardware performance counters
- perf, VTune, TAU
- Roofline model
- Profiling overhead
- **Lab:** Build roofline plot for kernel

### Day 191: Cache Optimization
- Cache-aware vs cache-oblivious algorithms
- Loop tiling and blocking
- Prefetch instructions
- False sharing mitigation
- **Lab:** Optimize stencil for cache

### Day 192: NUMA Optimization
- Non-uniform memory access
- numactl and memory affinity
- First-touch policy
- Interleaving vs local allocation
- **Lab:** NUMA-aware thread pinning

### Day 193: Lock-Free & Wait-Free Algorithms
- Atomic operations (CAS, fetch-and-add)
- Lock-free queues and stacks
- Memory ordering (acquire/release)
- Hazard pointers
- **Lab:** Implement lock-free stack

### Day 194: SIMD Optimization Patterns
- Struct-of-arrays vs array-of-structs
- Alignment and padding
- Masked operations
- Horizontal vs vertical operations
- **Lab:** Convert AoS to SoA

### Day 195: Benchmarking Methodology
- Microbenchmarking pitfalls
- Statistical significance
- Controlling for noise
- Google Benchmark framework
- **Lab:** Write robust benchmark suite

### Day 196: Week 28 Review & Project
- Project: Optimization case study
- Take unoptimized scientific code
- Apply: loop tiling, SIMD, threading, cache optimizations
- Document 10x+ speedup journey

---

## Week 29: Research Topics & Emerging Trends (Days 197-203)

### Day 197: Chiplets & Disaggregation
- AMD chiplet architecture (EPYC, MI300)
- UCIe (Universal Chiplet Interconnect)
- 2.5D/3D stacking
- Heterogeneous integration
- **Lab:** Study chiplet packaging tech

### Day 198: Advanced Packaging
- HBM (High-Bandwidth Memory)
- CoWoS (Chip-on-Wafer-on-Substrate)
- Interposer technologies
- Silicon photonics
- **Lab:** Analyze HBM bandwidth benefits

### Day 199: Carbon Nanotube & Photonic Processors
- Beyond-CMOS technologies
- CNT transistors
- Photonic interconnects
- Energy efficiency gains
- **Lab:** Research papers on CNT computing

### Day 200: DNA & Molecular Computing
- DNA storage
- DNA logic gates
- Molecular state machines
- Biocomputing applications
- **Lab:** Simulate DNA computation

### Day 201: Brain-Computer Interfaces
- Neural decoding
- Spike sorting
- Real-time signal processing
- Neuralink architecture
- **Lab:** Process EEG data on GPU

### Day 202: Edge AI & TinyML
- Model quantization (INT8, INT4)
- Pruning and knowledge distillation
- On-device training
- MCU inference (Cortex-M)
- **Lab:** Deploy TFLite model on microcontroller

### Day 203: Week 29 Review & Project
- Project: Emerging tech white paper
- Choose one topic (chiplets, photonics, neuromorphic, quantum)
- Literature review
- Technical deep dive
- Future outlook

---

## Week 30: Capstone Project & Course Wrap-Up (Days 204-210)

### Day 204: Capstone Project Kickoff
- Choose problem domain:
  - Option A: Full compiler for custom DSL
  - Option B: Novel ISA extension + hardware implementation
  - Option C: Heterogeneous accelerator framework
  - Option D: Open-ended research project
- **Lab:** Project proposal and plan

### Day 205: Capstone - Design Phase
- Architecture design
- API specification
- Test plan
- Milestone breakdown
- **Lab:** Design document and diagrams

### Day 206: Capstone - Implementation (Part 1)
- Core functionality
- Unit tests
- Code reviews
- **Lab:** Implement first milestone

### Day 207: Capstone - Implementation (Part 2)
- Integration testing
- Performance optimization
- Documentation
- **Lab:** Complete implementation

### Day 208: Capstone - Benchmarking & Analysis
- Performance evaluation
- Comparison with baselines
- Profiling and bottleneck analysis
- **Lab:** Generate benchmark results

### Day 209: Capstone - Final Presentation
- Presentation slides
- Demo preparation
- Q&A practice
- **Lab:** Present to peers/instructors

### Day 210: Course Retrospective & Next Steps
- Review learning journey (Phases 1-7)
- Industry career paths
- Contributing to open source (LLVM, RISC-V, etc.)
- Continued learning resources
- **Celebration:** Phase 7 Complete! 🎓

---

## Appendix: Tools & Resources

### Development Tools
- **Compilers:** GCC, Clang/LLVM, Intel ICC, NVCC
- **Profilers:** perf, VTune, Nsight, rocprof, Instruments
- **Simulators:** QEMU, gem5, Spike (RISC-V), Verilator
- **HDL Tools:** Vivado, Quartus, Yosys, Chisel
- **Libraries:** cuBLAS, cuDNN, rocBLAS, MIOpen, oneMKL

### Recommended Hardware
- x86 workstation with AVX-512 support
- NVIDIA GPU (Ampere/Hopper for best features)
- AMD GPU (optional, for ROCm learning)
- FPGA development board (Xilinx or Intel)
- RISC-V board or access to cloud instances

### Reference Books
- "Computer Architecture: A Quantitative Approach" - Hennessy & Patterson
- "Engineering a Compiler" - Cooper & Torczon
- "LLVM Cookbook" - M. Pandey & S. Sarda
- "Getting Started with LLVM Core Libraries" - B. Cardoso Lopes
- "Digital Design and Computer Architecture" - Harris & Harris

### Online Courses
- MIT 6.823 (Computer System Architecture)
- Stanford CS143 (Compilers)
- Berkeley CS152 (Computer Architecture)
- UIUC ECE565 (Compiler Optimization)

---

**Total Content:** 210 days of intensive parallel programming, compiler engineering, and hardware design

**Outcome:** World-class expertise in performance optimization across the full computing stack from silicon to software.
