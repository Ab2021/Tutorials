# Day 001: x86 Architecture & SIMD Evolution
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 1: x86 SIMD Foundations

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Understand x86-64 Microarchitecture:** Explain the evolution from Sandy Bridge to modern Zen 4 processors, including pipeline design, execution units, and micro-ops
2. **Analyze SIMD Instruction Sets:** Trace the complete timeline from MMX through SSE generations to AVX-512, understanding width progression and capability expansion
3. **Master Register Architecture:** Identify and utilize vector register files (MM, XMM, YMM, ZMM) and understand register aliasing relationships
4. **Distinguish Parallelism Types:** Differentiate between Data-Level Parallelism (DLP), Instruction-Level Parallelism (ILP), and Thread-Level Parallelism (TLP)
5. **Enumerate CPU Capabilities:** Use `cpuid`, `lscpu`, and `/proc/cpuinfo` to programmatically detect SIMD features and plan architecture-specific optimizations

---

## 📚 Prerequisites & Preparation

### Hardware Requirements

| Component | Specification | Purpose | Notes |
|-----------|--------------|---------|-------|
| CPU | x86-64 with SSE4.2+ | Learning platform | AVX2+ highly recommended |
| RAM | 16GB+ | Compiling examples | 8GB minimum acceptable |
| Storage | 20GB free | Tools and documentation | SSD preferred for compilation |

### Software Environment

```bash
# Ubuntu 22.04 LTS / Windows WSL2 Setup
# Ensure you have administrative privileges

# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install build-essential -y
sudo apt install gcc-13 g++-13 -y
sudo apt install clang-17 lldb-17 -y

# Install CPU analysis tools
sudo apt install cpuid hwloc util-linux -y

# Install performance monitoring
sudo apt install linux-tools-generic linux-tools-$(uname -r) -y

# Verification
gcc-13 --version     # Should show 13.x
clang-17 --version   # Should show 17.x
cpuid --version      # For CPU feature detection
lscpu               # System CPU information
```

### Prior Knowledge Checklist

- [ ] **Computer Architecture Basics:** Understanding of CPU components (ALU, registers, cache)
- [ ] **Binary & Hexadecimal:** Comfortable with bit-level operations and representations
- [ ] **C/C++ Fundamentals:** Pointers, arrays, basic compilation workflow
- [ ] **Assembly Reading:** Ability to understand basic x86-64 assembly (helpful but not required)

### Key Resources

**Essential Reading:**
- Intel® 64 and IA-32 Architectures Software Developer's Manual Volume 1 (Chapter 11: SIMD)
- AMD64 Architecture Programmer's Manual Volume 1: Application Programming
- Agner Fog's "Optimizing Software in C++" (Chapter 9: Vectorization)

**Online References:**
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [uops.info - Instruction Tables](https://uops.info/)
- [WikiChip Microarchitecture Database](https://en.wikichip.org/wiki/WikiChip)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The x86-64 Architecture Foundation

#### 1.1 Historical Context: The Journey to Modern x86

The x86 instruction set architecture represents one of the longest-running and most successful ISAs in computing history, originating from Intel's 8086 processor in 1978. Understanding its evolution is crucial to appreciating modern SIMD capabilities.

**Timeline of Major Milestones:**

- **1978 - Intel 8086:** 16-bit processor, foundation of x86
- **1985 - Intel 80386:** First 32-bit x86 processor (IA-32)
- **1993 - Intel Pentium:** Superscalar architecture (dual pipeline)
- **1997 - Intel Pentium MMX:** First SIMD instructions (64-bit MMX)
- **1999 - Intel Pentium III:** SSE (Streaming SIMD Extensions) introduced
- **2003 - AMD Athlon 64:** First x86-64 (AMD64/x64) processor
- **2008 - Intel Core i7 (Nehalem):** Modern microarchitecture template
- **2011 - Intel Sandy Bridge:** AVX (256-bit) and massive redesign
- **2017 - Intel Skylake-X:** AVX-512 in consumer processors
- **2022 - AMD Zen 4:** AVX-512 support in AMD mainstream
- **2024 - Current State:** hybrid microarchitectures (P-cores/E-cores)

**Why This Evolution Matters:**

The progression from simple scalar execution to wide SIMD represents a fundamental shift in how CPUs achieve performance. When single-core clock speeds plateaued around 2005 (the "power wall"), architects turned to parallelism at multiple levels:

1. **Instruction-Level Parallelism (ILP):** Execute multiple independent instructions simultaneously within a single thread
2. **Data-Level Parallelism (DLP):** Process multiple data elements with a single instruction (SIMD)
3. **Thread-Level Parallelism (TLP):** Run multiple threads/processes concurrently

Modern x86 processors exploit ALL three forms simultaneously, achieving 100+ GFLOPS from a single core.

#### 1.2 x86-64 Microarchitecture Deep Dive

**What is a Microarchitecture?**

While the *Instruction Set Architecture* (ISA) defines the programmer-visible interface (instructions, registers, memory model), the *microarchitecture* is the actual hardware implementation. Different microarchitectures can implement the same ISA with vastly different performance characteristics.

**Core Components of Modern x86-64 Microarchitectures:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Instruction Fetch)                │
├─────────────────────────────────────────────────────────────────┤
│  Branch Predictor → Instruction Cache (L1-I) → Fetch Queue     │
│  Decode → Micro-op Cache (DSB) → Micro-op Queue                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND (Execution & Memory)                       │
├─────────────────────────────────────────────────────────────────┤
│  Scheduler/Reservation Station → Execution Units               │
│  ├─ Port 0: ALU, FP_ADD, FP_MUL, Vector (Integer/FP)          │
│  ├─ Port 1: ALU, FP_ADD, FP_MUL, Vector (Integer/FP)          │
│  ├─ Port 2: Load AGU (Address Generation Unit)                │
│  ├─ Port 3: Load AGU                                           │
│  ├─ Port 4: Store Data                                         │
│  ├─ Port 5: ALU, Vector Shuffle, Branch                       │
│  ├─ Port 6: ALU, Branch                                        │
│  └─ Port 7: Store AGU                                          │
│                                                                 │
│  Load/Store Buffers ↔ L1 Data Cache (32-64KB)                 │
│                       L2 Cache (256KB-1MB per core)            │
│                       L3 Cache (8-64MB shared)                 │
└─────────────────────────────────────────────────────────────────┘
```

**Pipeline Stages in Modern x86:**

Unlike RISC processors with simple 5-stage pipelines, modern x86 processors have 14-20+ pipeline stages. Here's a conceptual breakdown (using Intel's terminology):

1. **Fetch (F1-F2):** Retrieve instructions from L1 instruction cache
2. **Decode (D1-D4):** Convert complex x86 instructions into micro-ops (µops)
3. **Allocate/Rename (A1-A2):** Assign physical registers, handle register renaming
4. **Schedule/Dispatch (S1-S2):** Wait for operands, dispatch to execution ports
5. **Execute (E1-E4):** Actual computation on functional units
6. **Retire (R1-R2):** Commit results in program order, update architectural state

**Total:** ~14-19 stages depending on microarchitecture

**Key Microarchitecture Comparison:**

| Microarch | Year | IPC | Pipeline Depth | SIMD Width | Ports | ROB Size |
|-----------|------|-----|----------------|------------|-------|----------|
| Sandy Bridge (Intel) | 2011 | ~1.5 | 14 stages | 256-bit (AVX) | 6 | 168 |
| Haswell (Intel) | 2013 | ~1.7 | 14 stages | 256-bit (AVX2) | 8 | 192 |
| Skylake (Intel) | 2015 | ~1.8 | 14 stages | 512-bit (AVX-512) | 8 | 224 |
| Zen 2 (AMD) | 2019 | ~1.7 | 19 stages | 256-bit (AVX2) | 10 | 224 |
| Zen 3 (AMD) | 2020 | ~1.9 | 19 stages | 256-bit (AVX2) | 10 | 256 |
| Zen 4 (AMD) | 2022 | ~2.0 | 19 stages | 512-bit (AVX-512) | 10 | 256 |
| Golden Cove (Intel) | 2021 | ~2.1 | 17 stages | 512-bit (AVX-512) | 12 | 512 |

*IPC = Instructions Per Cycle (higher is better), ROB = Re-Order Buffer*

**Out-of-Order Execution Explained:**

One of the most critical features of modern CPUs is **Out-of-Order (OoO) execution**. This allows the processor to execute instructions in a different order than they appear in the program, maximizing utilization of execution units.

**Example: Register Renaming**

Consider this code:
```asm
ADD EAX, EBX    ; EAX = EAX + EBX
MOV ECX, EAX    ; ECX = EAX
ADD EAX, EDX    ; EAX = EAX + EDX  (WAW hazard!)
```

Without renaming, the third instruction must wait for the first to complete (Write-After-Write dependency). With register renaming:

```asm
; Physical registers: P0, P1, P2, P3...
ADD P0, EBX     ; P0 = EAX + EBX
MOV ECX, P0     ; ECX = P0
ADD P1, EDX     ; P1 = P0 + EDX (can execute in parallel!)
```

The processor maintains a mapping from architectural registers (EAX, EBX) to a larger set of physical registers (P0-P191 on Skylake). This eliminates false dependencies and increases parallelism.

**Reorder Buffer (ROB):**

The ROB is a circular buffer that holds µops until they can be safely retired (committed). A larger ROB allows the processor to "look ahead" further in the instruction stream, finding more opportunities for parallel execution.

| Processor | ROB Size | Benefit |
|-----------|----------|---------|
| Skylake | 224 entries | Can track ~75 x86 instructions |
| Zen 3 | 256 entries | Can track ~85 x86 instructions |
| Golden Cove | 512 entries | Can track ~170 x86 instructions |

A 512-entry ROB means the CPU can essentially "see" 170+ instructions ahead, dramatically improving ILP extraction.

#### 1.3 SIMD Instruction Set Evolution

**What is SIMD?**

**Single Instruction, Multiple Data** (SIMD) is a parallel computing model where one instruction operates on multiple data elements simultaneously. Instead of:

```c
// Scalar (1 operation per instruction)
c[0] = a[0] + b[0];
c[1] = a[1] + b[1];
c[2] = a[2] + b[2];
c[3] = a[3] + b[3];
```

SIMD allows:
```c
// SIMD (4 operations per instruction)
vec_c = vec_add(vec_a, vec_b);  // Adds 4 floats in one instruction
```

This provides a theoretical 4x speedup with perfect data parallelism.

**The MMX Era (1997-2000):**

Intel's **MMX** (MultiMedia eXtension) was the first SIMD instruction set for x86, introduced with the Pentium MMX.

**Features:**
- **Register Width:** 64-bit (MM0-MM7)
- **Data Types:** 8x8-bit, 4x16-bit, 2x32-bit integers only
- **Critical Flaw:** MMX registers aliased to x87 FPU registers (MMi ≡ ST(i)), requiring mode switching
- **Performance:** 2x speedup for multimedia workloads

**Instruction Categories:**
- Packed arithmetic: `PADDB`, `PADDW`, `PADDD` (add bytes/words/dwords)
- Packed logic: `PAND`, `POR`, `PXOR`
- Pack/unpack: `PUNPCKLBW`, `PACKUSWB`

**Why MMX Failed:**

The fatal flaw was register aliasing with x87. Switching between MMX and floating-point required expensive state saves via `EMMS` (Empty MMX State), creating huge performance penalties in mixed-mode code. This made MMX unsuitable for general-purpose SIMD.

**The SSE Revolution (1999-2006):**

**SSE (Streaming SIMD Extensions)** fixed MMX's mistakes by introducing dedicated 128-bit registers.

**SSE1 (Pentium III, 1999):**
- **Registers:** 8 new 128-bit registers (XMM0-XMM7)
- **Data Types:** 4x 32-bit float (ONLY single precision)
- **Instructions:** 70 new instructions
- **Key Innovation:** Independent from x87, no mode switching

**Sample SSE1 Instructions:**
```asm
MOVAPS   xmm0, [mem]     ; Move 4 aligned floats
ADDPS    xmm0, xmm1      ; Add 4 floats: xmm0 += xmm1
MULPS    xmm0, xmm1      ; Multiply 4 floats
MAXPS    xmm0, xmm1      ; Element-wise maximum
CMPPS    xmm0, xmm1, 0   ; Compare (0=EQ, 1=LT, 2=LE, etc.)
```

**SSE2 (Pentium 4, 2001):**
- **Expansion:** Added double-precision (2x64-bit) and integer operations
- **Registers:** Expanded to XMM0-XMM15 in x86-64 mode (2003)
- **Impact:** Made SSE viable for scientific computing (double precision)

**SSE3 (Pentium 4 Prescott, 2004):**
- **Horizontal operations:** `HADDPS` (horizontal add) for reductions
- **Thread synchronization:** `MONITOR`/`MWAIT` instructions

**SSSE3 (Core 2 Duo, 2006):**
- **Shuffle improvements:** `PSHUFB` (byte shuffle) for complex permutations
- **Note:** "Supplemental SSE3"

**SSE4.1 & SSE4.2 (Core 2 Penryn, 2007-2008):**
- **SSE4.1:** Blend, min/max for integers, dot products (`DPPS`)
- **SSE4.2:** String processing (`PCMPISTRM`), CRC32

**Summary of SSE Generations:**

| Version | Year | Key Addition | Example Instruction |
|---------|------|--------------|---------------------|
| SSE1 | 1999 | 4x float32 | `ADDPS` |
| SSE2 | 2001 | 2x float64, integers | `ADDPD`, `PADDQ` |
| SSE3 | 2004 | Horizontal ops | `HADDPS` |
| SSSE3 | 2006 | Byte shuffle | `PSHUFB` |
| SSE4.1 | 2007 | Blending, DP | `BLENDVPS`, `DPPS` |
| SSE4.2 | 2008 | Strings, CRC | `PCMPISTRM`, `CRC32C` |

**The AVX Expansion (2011-2017):**

**AVX (Advanced Vector Extensions)** represented a massive leap by doubling vector width to 256 bits.

**AVX1 (Sandy Bridge, 2011):**
- **Register Width:** 256-bit (YMM0-YMM15)
- **Data Types:** 8x32-bit float or 4x64-bit double
- **Key Change:** 3-operand format (non-destructive)

**Comparison:**
```asm
; SSE2 (2-operand, destructive)
ADDPS xmm0, xmm1    ; xmm0 = xmm0 + xmm1 (xmm0 overwritten)

; AVX (3-operand, non-destructive)
VADDPS ymm0, ymm1, ymm2  ; ymm0 = ymm1 + ymm2 (ymm1 preserved)
```

This 3-operand form reduces register pressure and intermediate moves.

**AVX2 (Haswell, 2013):**
- **Integer Operations:** Extended AVX to integers (finally!)
- **Gather:** `VGATHERDPS` for non-contiguous memory access
- **FMA:** Fused Multiply-Add (`VFMADD231PS`)

**FMA Importance:**

FMA computes `a*b + c` with a single rounding error (vs. two for separate multiply+add). For neural networks and matrix operations, this provides:
- **Performance:** 2x theoretical peak (both multiply and add in one cycle)
- **Accuracy:** Single rounding reduces numerical error

```
Peak GFLOPS = Cores × Frequency × (SIMD_width / 32) × FMA_units × 2
            = 8 cores × 3.5GHz × (256/32) × 2 ports × 2
            = 8 × 3.5 × 8 × 2 × 2 = 896 GFLOPS (single precision)
```

**AVX-512 (Skylake-X, 2017):**

AVX-512 is not a single extension but a family of extensions, each adding specific capabilities.

**Core Extensions:**
- **AVX-512F (Foundation):** 512-bit vectors (16x float32, 8x float64)
- **AVX-512CD (Conflict Detection):** Detect conflicts in gather/scatter
- **AVX-512BW (Byte/Word):** 64x 8-bit or 32x 16-bit operations
- **AVX-512DQ (DWord/QWord):** Additional 32/64-bit operations
- **AVX-512VL (Vector Length):** Apply 512-bit ops to 128/256-bit registers

**Optional Extensions:**
- **AVX-512VNNI:** Vector Neural Network Instructions (INT8 dot product)
- **AVX-512VBMI:** Vector Bit Manipulation (advanced shuffle)
- **AVX-512IFMA:** Integer FMA for cryptography
- **AVX-512BF16:** BFloat16 support (AI/ML)

**Key AVX-512 Features:**

1. **Opmask Registers (k0-k7):** 64-bit predicate masks for conditional execution
```asm
VADDPS zmm0{k1}, zmm1, zmm2  ; Only update elements where k1 bit is set
```

2. **Embedded Rounding Control:** Specify rounding mode per-instruction
```asm
VADDPS zmm0, zmm1, zmm2, {rn-sae}  ; Round to nearest, suppress exceptions
```

3. **Broadcast:** Efficiently replicate scalar to all vector lanes
```asm
VADDPS zmm0, zmm1, [mem]{1to16}  ; Broadcast mem to all 16 floats, then add
```

**AVX-512 Controversy:**

AVX-512 has been controversial due to:
- **Power/Thermal:** Wide vectors consume significant power, reducing clock speeds
- **Die Area:** Large execution units increase chip cost
- **Portability:** Not all Intel CPUs have AVX-512 (removed from Alder Lake)

AMD only added AVX-512 in Zen 4 (2022), five years after Intel.

#### 1.4 Register Architecture Deep Dive

**Understanding Register Aliasing:**

One of the most confusing aspects of x86 SIMD is register aliasing. Registers of different widths are actually views into the same underlying storage:

```
ZMM0 [512 bits]  ┌──────────────────────────────────────────────────────┐
                 │                                                      │
YMM0 [256 bits]  │                    ┌─────────────────────────────────┤
                 │                    │                                 │
XMM0 [128 bits]  │                    │              ┌──────────────────┤
                 │                    │              │                  │
                 └────────────────────┴──────────────┴──────────────────┘
                 Bits: 511----------256-----------128------------------0
```

**Register Relationships:**
- `XMM0` = Lower 128 bits of `YMM0`
- `YMM0` = Lower 256 bits of `ZMM0`
- Writing to `XMM0` zero-extends to `YMM0` and `ZMM0` (bits 128-511 cleared!)
- Writing to `YMM0` zero-extends to `ZMM0` (bits 256-511 cleared!)

**Why This Matters:**

Mixing different vector widths can cause unexpected behavior:
```asm
VMOVAPS ymm0, ymm1     ; ymm0 = ymm1 (256 bits)
VADDPS  xmm0, xmm0, xmm2  ; Upper 128 bits of ymm0 NOW ZERO!
```

The `xmm0` write zeroed the upper half of `ymm0`, potentially destroying data.

**Register Count by Mode:**

| Mode | Available Registers | Total Width |
|------|---------------------|-------------|
| MMX (obsolete) | MM0-MM7 (8 registers) | 64-bit each |
| SSE (32-bit mode) | XMM0-XMM7 (8 registers) | 128-bit each |
| SSE (64-bit mode) | XMM0-XMM15 (16 registers) | 128-bit each |
| AVX/AVX2 | YMM0-YMM15 (16 registers) | 256-bit each |
| AVX-512 | ZMM0-ZMM31 (32 registers) | 512-bit each |
| AVX-512 Opmask | K0-K7 (8 registers) | 64-bit masks |

**Physical Register Implementation:**

While the ISA exposes 16 (or 32) vector registers, modern CPUs have many more physical registers for renaming:

| Microarchitecture | Architectural XMM/YMM | Physical Vector Registers |
|-------------------|------------------------|---------------------------|
| Sandy Bridge | 16 | 144 |
| Haswell | 16 | 168 |
| Skylake | 16 (32 for ZMM) | 168 |
| Zen 3 | 16 | 160 |

This large physical register file enables aggressive out-of-order execution.

#### 1.5 Parallelism Taxonomy: DLP vs. ILP vs. TLP

Modern processors exploit three distinct forms of parallelism simultaneously. Understanding their differences is crucial for effective optimization.

**1. Data-Level Parallelism (DLP) - SIMD**

**Definition:** Applying the same operation to multiple data elements in parallel.

**Example:**
```c
// 4 independent additions in one instruction
__m128 a = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
__m128 b = _mm_set_ps(5.0, 6.0, 7.0, 8.0);
__m128 c = _mm_add_ps(a, b);  // [6.0, 8.0, 10.0, 12.0] in one cycle
```

**Characteristics:**
- **Requires:** Regular data structures (arrays) and uniform operations
- **Compiler Support:** Good (auto-vectorization in modern compilers)
- **Efficiency:** Near-linear scaling (8x wider → ~8x faster)
- **Limitations:** Control flow (if/else) breaks vectorization

**Best Use Cases:**
- Image processing (apply filter to all pixels)
- Linear algebra (matrix operations)
- Signal processing (FFT, convolution)

**2. Instruction-Level Parallelism (ILP) - Superscalar**

**Definition:** Executing multiple independent instructions simultaneously within a single thread.

**Example:**
```c
int a = x + y;      // Independent operations
int b = z * w;      // Can execute in parallel
int c = p - q;      //
int d = a + b + c;  // Depends on previous results
```

A CPU with 4 ALUs can execute the first three lines in parallel (IPC = 3), then execute the fourth (IPC drops to 1 due to dependency).

**Characteristics:**
- **Requires:** Independent instructions with no data dependencies
- **Compiler Support:** Excellent (instruction scheduling)
- **Efficiency:** Limited by dependency chains (~2-4 IPC typical)
- **Transparency:** Completely automatic (hardware handles it)

**Measurement:**
```bash
perf stat -e instructions,cycles ./program
# IPC = instructions / cycles
# Modern CPUs: 1.5-2.5 IPC on average workloads
```

**3. Thread-Level Parallelism (TLP) - Multi-core**

**Definition:** Running multiple independent threads/processes simultaneously on different cores.

**Example:**
```c
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    result[i] = compute(input[i]);  // Each iteration on different core
}
```

**Characteristics:**
- **Requires:** Decomposable workload, minimal shared state
- **Compiler Support:** Limited (requires explicit threading)
- **Efficiency:** Near-linear scaling up to core count (8 cores → ~7.5x)
- **Overhead:** Thread creation, synchronization, cache coherency

**Comparison Table:**

| Aspect | DLP (SIMD) | ILP (OoO) | TLP (Threading) |
|--------|------------|-----------|-----------------|
| **Granularity** | Vector lanes (4-16) | Instructions (4-6) | Cores (4-64) |
| **Scaling** | Up to 16x (AVX-512) | Up to 4x (typical) | Up to 100x+ |
| **Overhead** | Minimal | Zero (automatic) | High (synchronization) |
| **Explicit Code** | Yes (intrinsics) | No (automatic) | Yes (threads) |
| **Power Efficiency** | High | Moderate | Lower |

**Combining All Three:**

Modern optimized code uses all three:
```c
// TLP: Distribute work across 8 threads
#pragma omp parallel for num_threads(8)
for (int i = 0; i < N; i += 8) {
    // DLP: Process 8 elements at once with AVX2
    __m256 data = _mm256_loadu_ps(&input[i]);
    __m256 result = _mm256_mul_ps(data, coefficient);
    _mm256_storeu_ps(&output[i], result);
    
    // ILP: CPU automatically overlaps multiple iterations
}
```

This achieves: 8 (threads) × 8 (SIMD width) × 2 (ILP) = **128x theoretical** speedup!

#### 1.6 SIMD Execution Units and Ports

**Port Architecture:**

Modern CPUs use **ports** to route µops to functional units. Each port can handle specific instruction types.

**Intel Skylake Port Mapping:**

| Port | Execution Units | Latency | Throughput |
|------|----------------|---------|------------|
| **Port 0** | Integer ALU, FP Add, FP Mul, Vector Integer/FP, Divide | 3-4 cycles | 1 µop/cycle |
| **Port 1** | Integer ALU, FP Add, FP Mul, Vector Integer/FP | 3-4 cycles | 1 µop/cycle |
| **Port 2** | Load AGU | - | 1 load/cycle |
| **Port 3** | Load AGU | - | 1 load/cycle |
| **Port 4** | Store Data | - | 1 store/cycle |
| **Port 5** | Integer ALU, Vector Shuffle, Branch | 1-3 cycles | 1 µop/cycle |
| **Port 6** | Integer ALU, Branch | 1 cycle | 1 µop/cycle |
| **Port 7** | Store AGU | - | 1 store/cycle |

**SIMD Instruction Port Usage:**

```asm
VADDPS ymm0, ymm1, ymm2   ; Port 0 or Port 1 (FP Add unit)
VMULPS ymm0, ymm1, ymm2   ; Port 0 or Port 1 (FP Mul unit)
VFMADD231PS ymm0, ymm1, ymm2  ; Port 0 or Port 1 (FMA unit)
VPSHUFB ymm0, ymm1, ymm2  ; Port 5 (Vector Shuffle)
VMOVAPS ymm0, [rax]       ; Port 2 or Port 3 (Load)
VMOVAPS [rax], ymm0       ; Port 4 + Port 7 (Store)
```

**Throughput Analysis:**

If both Port 0 and Port 1 can execute FMA instructions, theoretical peak is:
```
Peak = Frequency × Ports × SIMD_width / element_size
     = 3.5 GHz × 2 × 256 bits / 32 bits × 2 (FMA counts as 2 ops)
     = 3.5 × 2 × 8 × 2 = 112 GFLOPS per core
```

**Port Contention:**

If code uses only one type of instruction (e.g., only FP adds), it bottlenecks on ports 0 and 1:
```c
// Poor: All adds contend for ports 0/1
for (...) {
    c[i] = a[i] + b[i];           // Port 0/1
    d[i] = e[i] + f[i];           // Port 0/1 (contention!)
}

// Better: Mix instruction types
for (...) {
    c[i] = a[i] + b[i];           // Port 0/1
    d[i] = e[i] + f[i];           // Port 0/1
    g[i] = h[i] & mask;           // Port 0/1/5 (integer)
    x[i] = shuffle(y[i]);         // Port 5 (no contention)
}
```

Balanced port utilization maximizes throughput.

#### 1.7 Micro-op Fusion and Optimization

**Macro-Fusion:**

The CPU can fuse certain instruction pairs into a single µop:
```asm
CMP rax, rbx
JE  target        ; These fuse into one µop (CMP+JE)
```

**Benefits:**
- Reduced ROB pressure (1 slot instead of 2)
- Higher effective throughput
- Common in loop conditions

**Micro-Fusion:**

Memory operations can fuse address calculation with load/store:
```asm
ADD rax, [rbx + rcx*8 + 16]   ; Fuses into: Load + Add (2 µops, not 3)
```

This is why x86's complex addressing modes are efficient despite appearing expensive.

**Instruction Latency vs. Throughput:**

| Instruction | Latency | Reciprocal Throughput | Notes |
|-------------|---------|----------------------|-------|
| `VADDPS ymm` | 4 cycles | 0.5 cycles | Can issue 2 per cycle |
| `VMULPS ymm` | 4 cycles | 0.5 cycles | Can issue 2 per cycle |
| `VFMADD ymm` | 4 cycles | 0.5 cycles | Replaces mul+add |
| `VDIVPS ymm` | 13-14 cycles | 5 cycles | Very slow! |
| `VSQRTPS ymm` | 12 cycles | 6 cycles | Slow |
| `VPSHUFB ymm` | 1 cycle | 1 cycle | Fast |

**Latency** = How long for one instruction to complete
**Throughput** = How many can be issued per cycle

With sufficient ILP, throughput matters more than latency.

---

### 🔹 Part 2: CPU Feature Detection and Enumeration

#### 2.1 The CPUID Instruction

**What is CPUID?**

`CPUID` is a special x86 instruction that returns processor identification and feature information. It's the standard way to detect CPU capabilities at runtime.

**Basic Usage:**

```c
#include <cpuid.h>

void detect_features() {
    unsigned int eax, ebx, ecx, edx;
    
    // CPUID with EAX=1: Processor Info and Feature Bits
    __cpuid(1, eax, ebx, ecx, edx);
    
    bool hasSSE    = (edx >> 25) & 1;  // Bit 25 of EDX
    bool hasSSE2   = (edx >> 26) & 1;  // Bit 26 of EDX
    bool hasSSE3   = (ecx >> 0)  & 1;  // Bit 0 of ECX
    bool hasSSSE3  = (ecx >> 9)  & 1;  // Bit 9 of ECX
    bool hasSSE4_1 = (ecx >> 19) & 1;  // Bit 19 of ECX
    bool hasSSE4_2 = (ecx >> 20) & 1;  // Bit 20 of ECX
    bool hasAVX    = (ecx >> 28) & 1;  // Bit 28 of ECX
    
    printf("SSE: %d, SSE2: %d, AVX: %d\n", hasSSE, hasSSE2, hasAVX);
}
```

**CPUID Leaf Functions:**

| EAX Input | Information Returned |
|-----------|---------------------|
| 0x00000000 | Maximum supported basic leaf, vendor ID string |
| 0x00000001 | Processor info, feature bits (SSE, SSE2, etc.) |
| 0x00000007 | Extended features (AVX2, AVX-512, etc.) |
| 0x80000000 | Maximum supported extended leaf |
| 0x80000001 | Extended processor info |
| 0x80000002-4 | Processor brand string (e.g., "Intel Core i9") |

**Extended Features (Leaf 7):**

```c
// CPUID with EAX=7, ECX=0: Extended Features
__cpuid_count(7, 0, eax, ebx, ecx, edx);

bool hasAVX2     = (ebx >> 5)  & 1;  // Bit 5 of EBX
bool hasAVX512F  = (ebx >> 16) & 1;  // Bit 16 of EBX  (Foundation)
bool hasAVX512DQ = (ebx >> 17) & 1;  // Bit 17 of EBX
bool hasAVX512BW = (ebx >> 30) & 1;  // Bit 30 of EBX
bool hasAVX512VL = (ebx >> 31) & 1;  // Bit 31 of EBX
```

**Complete Detection Example:**

```c
#include <stdio.h>
#include <cpuid.h>
#include <stdbool.h>

typedef struct {
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;
    bool avx;
    bool avx2;
    bool fma;
    bool avx512f;
    bool avx512dq;
    bool avx512bw;
    bool avx512vl;
    bool avx512vnni;
} CPUFeatures;

CPUFeatures detect_cpu_features() {
    CPUFeatures features = {0};
    unsigned int eax, ebx, ecx, edx;
    
    // Basic features (EAX=1)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        features.sse    = (edx >> 25) & 1;
        features.sse2   = (edx >> 26) & 1;
        features.sse3   = (ecx >> 0)  & 1;
        features.ssse3  = (ecx >> 9)  & 1;
        features.sse4_1 = (ecx >> 19) & 1;
        features.sse4_2 = (ecx >> 20) & 1;
        features.avx    = (ecx >> 28) & 1;
        features.fma    = (ecx >> 12) & 1;
    }
    
    // Extended features (EAX=7, ECX=0)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.avx2      = (ebx >> 5)  & 1;
        features.avx512f   = (ebx >> 16) & 1;
        features.avx512dq  = (ebx >> 17) & 1;
        features.avx512bw  = (ebx >> 30) & 1;
        features.avx512vl  = (ebx >> 31) & 1;
        features.avx512vnni = (ecx >> 11) & 1;
    }
    
    return features;
}

void print_features(CPUFeatures f) {
    printf("CPU SIMD Features:\n");
    printf("  SSE:         %s\n", f.sse ? "YES" : "NO");
    printf("  SSE2:        %s\n", f.sse2 ? "YES" : "NO");
    printf("  SSE3:        %s\n", f.sse3 ? "YES" : "NO");
    printf("  SSSE3:       %s\n", f.ssse3 ? "YES" : "NO");
    printf("  SSE4.1:      %s\n", f.sse4_1 ? "YES" : "NO");
    printf("  SSE4.2:      %s\n", f.sse4_2 ? "YES" : "NO");
    printf("  AVX:         %s\n", f.avx ? "YES" : "NO");
    printf("  AVX2:        %s\n", f.avx2 ? "YES" : "NO");
    printf("  FMA:         %s\n", f.fma ? "YES" : "NO");
    printf("  AVX-512F:    %s\n", f.avx512f ? "YES" : "NO");
    printf("  AVX-512DQ:   %s\n", f.avx512dq ? "YES" : "NO");
    printf("  AVX-512BW:   %s\n", f.avx512bw ? "YES" : "NO");
    printf("  AVX-512VL:   %s\n", f.avx512vl ? "YES" : "NO");
    printf("  AVX-512VNNI: %s\n", f.avx512vnni ? "YES" : "NO");
}
```

#### 2.2 Using lscpu for System Analysis

**lscpu Overview:**

`lscpu` is a Linux utility that gathers CPU architecture information from sysfs and /proc/cpuinfo.

**Basic Usage:**

```bash
$ lscpu
Architecture:            x86_64
CPU op-mode(s):          32-bit, 64-bit
Byte Order:              Little Endian
CPU(s):                  16
On-line CPU(s) list:     0-15
Thread(s) per core:      2
Core(s) per socket:      8
Socket(s):               1
NUMA node(s):            1
Vendor ID:               GenuineIntel
CPU family:              6
Model:                   154
Model name:              12th Gen Intel(R) Core(TM) i9-12900K
Stepping:                3
CPU MHz:                 3187.456
CPU max MHz:             5200.0000
CPU min MHz:             800.0000
BogoMIPS:                6374.91
Virtualization:          VT-x
L1d cache:               384 KiB
L1i cache:               256 KiB
L2 cache:                12 MiB
L3 cache:                30 MiB
NUMA node0 CPU(s):       0-15
Flags:                   fpu vme de sse sse2 sse3 ssse3 sse4_1 sse4_2
                         avx avx2 fma ...
```

**Key Information:**
- **Architecture:** x86_64 confirms 64-bit support
- **Threads per core:** 2 = Hyper-Threading enabled
- **Flags:** Lists all supported instruction sets

**Parsing Flags Programmatically:**

```bash
# Check for specific features
lscpu | grep -o 'avx2'    # Returns "avx2" if supported
lscpu | grep -o 'avx512f' # Returns "avx512f" if supported

# Count SIMD features
lscpu | grep Flags | grep -o 'sse[^ ]*' | wc -l  # Count SSE variants
```

**Advanced lscpu Usage:**

```bash
# JSON output (easier to parse)
lscpu -J

# Extended output with cache details
lscpu --extended

# Show CPU topology (which cores are on same socket)
lscpu --parse=CPU,Core,Socket,Node
```

#### 2.3 Analyzing /proc/cpuinfo

**Structure of /proc/cpuinfo:**

```bash
$ cat /proc/cpuinfo
processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model           : 154
model name      : 12th Gen Intel(R) Core(TM) i9-12900K @ 3.20GHz
stepping        : 3
microcode       : 0x32
cpu MHz         : 800.000
cache size      : 30720 KB
physical id     : 0
siblings        : 16
core id         : 0
cpu cores       : 8
apicid          : 0
initial apicid  : 0
fpu             : yes
fpu_exception   : yes
cpuid level     : 32
wp              : yes
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca
                  cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm
                  pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon
                  pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf
                  ...
                  avx avx2 fma ... [continues]
```

**Important Fields:**

- **processor:** Logical CPU number (0-15 for 8-core HT system)
- **physical id:** Physical socket ID (for multi-socket systems)
- **core id:** Core ID within socket
- **cpu cores:** Physical cores
- **siblings:** Logical processors (cores × threads_per_core)
- **flags:** All CPU feature flags

**Grep Tricks:**

```bash
# Show only flags
cat /proc/cpuinfo | grep ^flags | head -1 | sed 's/^flags\s*:\s*//' | tr ' ' '\n' | sort

# Check specific feature
grep -o 'avx512' /proc/cpuinfo && echo "AVX-512 supported" || echo "No AVX-512"

# Count logical vs physical cores
echo "Logical CPUs: $(grep -c ^processor /proc/cpuinfo)"
echo "Physical cores: $(grep ^cpu\ cores /proc/cpuinfo | head -1 | awk '{print $4}')"
echo "Sockets: $(grep ^physical\ id /proc/cpuinfo | sort -u | wc -l)"
```

---

### 🔹 Part 3: Practical Implications

#### 3.1 Choosing the Right SIMD Level

**Decision Matrix:**

| Use Case | Recommended SIMD | Reasoning |
|----------|------------------|-----------|
| Maximum compatibility | SSE2 | Universal on x86-64 (mandatory since 2003) |
| Good performance/portability | SSE4.2 or AVX | Widely available (2008+/2011+) |
| High performance | AVX2 + FMA | Sweet spot (2013+, most systems) |
| Cutting edge | AVX-512 | Best perf, but limited availability |
| Cross-platform library | Runtime dispatch | Detect and choose at startup |

**Runtime Dispatch Pattern:**

```c
// Function pointers for different implementations
typedef void (*vector_add_func)(float*, float*, float*, size_t);

vector_add_func choose_implementation() {
    CPUFeatures f = detect_cpu_features();
    
    if (f.avx512f) return vector_add_avx512;
    if (f.avx2)    return vector_add_avx2;
    if (f.sse4_2)  return vector_add_sse42;
    return vector_add_scalar;
}

int main() {
    vector_add_func vec_add = choose_implementation();
    vec_add(a, b, c, n);  // Calls best available version
}
```

This pattern is used by libraries like Eigen, OpenBLAS, and Intel MKL.

#### 3.2 Performance Expectations

**Theoretical Speedups:**

| SIMD Level | Float32 Width | Int32 Width | vs. Scalar |
|------------|---------------|-------------|------------|
| MMX | N/A | 2x | 2x |
| SSE | 4x | 4x | 4x |
| AVX/AVX2 | 8x | 8x | 8x |
| AVX-512 | 16x | 16x | 16x |

**Real-World Speedups:**

In practice, speedups are lower due to:
- Memory bandwidth limits (often the real bottleneck)
- Overhead (loop setup, alignment, tail handling)
- Amdahl's Law (non-vectorizable code)

**Typical Observed:**
- SSE: 2-3x vs scalar
- AVX2: 4-6x vs scalar  
- AVX-512: 6-10x vs scalar (when not bandwidth-limited)

#### 3.3 Caveats and Gotchas

**AVX-512 Downclocking:**

Intel CPUs reduce clock frequency when executing AVX-512 instructions (to stay within TDP):

| Instruction Type | Typical Frequency | Downclock |
|------------------|-------------------|-----------|
| Scalar/SSE | 4.0 GHz (boost) | 0% |
| AVX2 | 3.8 GHz | -5% |
| AVX-512 (Light) | 3.4 GHz | -15% |
| AVX-512 (Heavy) | 2.8 GHz | -30% |

This means AVX-512 must provide >30% speedup over AVX2 just to break even!

**Transition Penalties:**

Mixing different SIMD widths incurs penalties:
```c
// BAD: Mixes SSE and AVX
__m128 a =_mm_add_ps(...);     // SSE
__m256 b = _mm256_add_ps(...);  // AVX (incurs transition penalty!)
```

Penalty: ~60-70 cycles on some microarchitectures. Stick to one width per hot loop.

**Denormal Numbers:**

Denormal (subnormal) floating-point numbers can cause 100x slowdowns:
```c
// Enable Flush-To-Zero and Denormals-Are-Zero
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
```

This trades numerical accuracy for performance (acceptable in many applications).

---

## 🧪 Hands-On Labs

### Lab 1: CPU Feature Detection

**Objective:** Write a program to enumerate all SIMD features of your CPU.

**Code:**

```c
// File: cpu_detect.c
// Compile: gcc -O2 -o cpu_detect cpu_detect.c

#include <stdio.h>
#include <cpuid.h>
#include <stdbool.h>

int main() {
    unsigned int eax, ebx, ecx, edx;
    
    // Get vendor string
    char vendor[13] = {0};
    __cpuid(0, eax, ebx, ecx, edx);
    *(unsigned int*)(vendor + 0) = ebx;
    *(unsigned int*)(vendor + 4) = edx;
    *(unsigned int*)(vendor + 8) = ecx;
    
    printf("CPU Vendor: %s\n", vendor);
    printf("Max CPUID level: %u\n\n", eax);
    
    // Get brand string
    char brand[49] = {0};
    for (unsigned int i = 0; i < 3; i++) {
        __cpuid(0x80000002 + i, eax, ebx, ecx, edx);
        *(unsigned int*)(brand + i*16 + 0) = eax;
        *(unsigned int*)(brand + i*16 + 4) = ebx;
        *(unsigned int*)(brand + i*16 + 8) = ecx;
        *(unsigned int*)(brand + i*16 + 12) = edx;
    }
    printf("CPU Brand: %s\n\n", brand);
    
    // Feature detection
    printf("SIMD Features:\n");
    __cpuid(1, eax, ebx, ecx, edx);
    printf("  MMX:     %s\n", (edx & (1<<23)) ? "YES" : "NO");
    printf("  SSE:     %s\n", (edx & (1<<25)) ? "YES" : "NO");
    printf("  SSE2:    %s\n", (edx & (1<<26)) ? "YES" : "NO");
    printf("  SSE3:    %s\n", (ecx & (1<<0))  ? "YES" : "NO");
    printf("  SSSE3:   %s\n", (ecx & (1<<9))  ? "YES" :"NO");
    printf("  SSE4.1:  %s\n", (ecx & (1<<19)) ? "YES" : "NO");
    printf("  SSE4.2:  %s\n", (ecx & (1<<20)) ? "YES" : "NO");
    printf("  AVX:     %s\n", (ecx & (1<<28)) ? "YES" : "NO");
    printf("  FMA:     %s\n", (ecx & (1<<12)) ? "YES" : "NO");
    
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    printf("  AVX2:    %s\n", (ebx & (1<<5))  ? "YES" : "NO");
    printf("  AVX512F: %s\n", (ebx & (1<<16)) ? "YES" : "NO");
    
    return 0;
}
```

**Expected Output:**
```
CPU Vendor: GenuineIntel
Max CPUID level: 32

CPU Brand:       Intel(R) Core(TM) i9-12900K CPU @ 3.20GHz

SIMD Features:
  MMX:     YES
  SSE:     YES
  SSE2:    YES
  SSE3:    YES
  SSSE3:   YES
  SSE4.1:  YES
  SSE4.2:  YES
  AVX:     YES
  FMA:     YES
  AVX2:    YES
  AVX512F: NO
```

### Lab 2: Analyzing System Topology

```bash
# Show cache hierarchy
lscpu --caches

# Expected output:
# NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL SETS PHY-LINE COHERENCY-SIZE
# L1d       48K     384K    12 Data            1   64        1             64
# L1i       32K     256K     8 Instruction     1   64        1             64
# L2      1.25M      10M    10 Unified         2 1024        1             64
# L3        30M      30M    12 Unified         3   ..        1             64
```

---

## 📝 Summary & Key Takeaways

### Core Concepts

1. **x86-64 is a Complex, High-Performance ISA:** Out-of-order execution, register renaming, and multi-port design enable IPC > 2.0
2. **SIMD Evolution:** MMX (64-bit) → SSE (128-bit) → AVX (256-bit) → AVX-512 (512-bit) represents 40+ years of parallel computing innovation
3. **Register Aliasing:** XMM ⊂ YMM ⊂ ZMM; writing narrower register zeros upper bits
4. **Three Parallelism Types:** DLP (SIMD), ILP (OoO), TLP (multi-core) work together
5. **Feature Detection:** Use `cpuid` instruction for runtime capability detection

### Performance Guidelines

✅ **DO:**
- Use SSE4.2 or AVX2 for broad compatibility
- Detect CPU features at runtime for portable code
- Align data to vector width (16/32/64 bytes)
- Minimize mixing of different SIMD widths in hot loops

❌ **DON'T:**
- Assume AVX-512 is always faster (downclocking!)
- Mix SSE and AVX in tight loops (transition penalty)
- Forget to handle denormals (flush-to-zero mode)
- Use AVX-512 if code will run on AMD Zen 3 or earlier

---

## 📚 Additional Resources

### Documentation
- [Intel® 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [AMD Software Optimization Guide for Zen](https://www.amd.com/en/support/tech-docs?keyword=software+optimization)
- [Agner Fog's Instruction Tables](https://www.agner.org/optimize/instruction_tables.pdf)

### Tools
- [Intel SDE (Software Development Emulator)](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) - Test AVX-512 without hardware
- [uops.info](https://uops.info/) - Detailed instruction performance data
- [LLVM-MCA](https://llvm.org/docs/CommandGuide/llvm-mca.html) - Machine Code Analyzer

### Further Reading
- "Computer Architecture: A Quantitative Approach" by Hennessy & Patterson (Chapter 4: SIMD)
- "Modern Microprocessors: A 90-Minute Guide" by Jason Robert Carey Patterson

---

**Tomorrow:** Day 2 - SSE Programming Model with hands-on intrinsics programming

*End of Day 001 - Total Lines: 1100+*
