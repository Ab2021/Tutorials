# Day 002: SSE Programming Model
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 1: x86 SIMD Foundations

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Master SSE Intrinsics:** Write production-quality code using Intel SSE intrinsics from `<xmmintrin.h>` through `<smmintrin.h>`
2. **Understand Data Types:** Work fluently with `__m128`, `__m128d`, and `__m128i` types and their semantic meanings
3. **Navigate Naming Conventions:** Decode intrinsic naming patterns (`_mm_<op>_<type>`) and choose appropriate instructions
4. **Handle Memory Alignment:** Implement correct aligned and unaligned memory operations, understanding performance implications
5. **Implement Conditional Execution:** Use masked operations and blend instructions for SIMD control flow

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86-64 with SSE2 | x86-64 with SSE4.2 |
| Compiler | GCC 7+ or Clang 8+ | GCC 13 or Clang 17 |
| Headers | `<immintrin.h>` | Full intrinsics support |

### Environment Setup

```bash
# Verify SSE support (from Day 001)
cat /proc/cpuinfo | grep sse

# Compile test
cat > test_sse.c << 'EOF'
#include <emmintrin.h>  // SSE2
int main() {
    __m128 a = _mm_setzero_ps();
    return 0;
}
EOF

gcc -msse2 -o test_sse test_sse.c && echo "SSE2 OK" || echo "SSE2 FAIL"
```

### Prior Knowledge Checklist

- [ ] Completed Day 001 (x86 Architecture & SIMD Evolution)
- [ ] Understand SIMD concept (Single Instruction, Multiple Data)
- [ ] Familiar with C pointers and arrays
- [ ] Can read basic assembly (helpful for verification)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SSE Architecture and Design Philosophy

#### 1.1 The SSE Programming Model

**Fundamental Premise:**

SSE (Streaming SIMD Extensions) treats 128-bit XMM registers as vectors of smaller elements that are processed in parallel. Unlike scalar programming where one instruction processes one data element, SSE instructions process 4 elements (for 32-bit data) or 2 elements (for 64-bit data) simultaneously.

**Conceptual View:**

```
Scalar Addition:
    a = 1.0     b = 5.0     =>  c = 6.0
    (1 operation per instruction)

SSE Vector Addition:
    a = [1.0, 2.0, 3.0, 4.0]   (128 bits = 4×32-bit floats)
    b = [5.0, 6.0, 7.0, 8.0]
    ==================================
    c = [6.0, 8.0, 10.0, 12.0]
    (4 operations in one instruction!)
```

This parallelism is **explicit** - the programmer (or compiler) must organize data into SIMD-friendly layouts.

#### 1.2 SSE Generations and Capabilities

SSE evolved through multiple generations, each adding new capabilities:

**SSE1 (1999 - Pentium III):**
- **Target:** Gaming and multimedia (MP3 decoding, 3D graphics)
- **Data Types:** ONLY 32-bit floating-point (4 per register)
- **Instructions:** 70 instructions (load, store, arithmetic, compare)
- **Limitation:** No integer operations, no double precision

**Critical Design Decision:**

Intel chose to add ONLY single-precision float support in SSE1 because:
1. Games and graphics primarily use float32
2. Simpler hardware (only FP units, not integer)
3. Faster time-to-market

This was later recognized as limiting, leading to SSE2.

**SSE2 (2001 - Pentium 4):**
- **Expansion:** Added double-precision (2×64-bit) and full integer support
- **Impact:** Made SSE viable for scientific computing
- **Mandate:** Became mandatory in x86-64 specification (AMD64)

**Data Type Support:**

```
SSE2 Register (XMM): 128 bits

┌─────────────────────────────────────────┐
│  Interpretation depends on instruction  │
├───────┬───────┬───────┬───────────────┤
│ 4× float32                             │  _mm_add_ps
├───────┴───────┴───────────────────────┤
│ 2× float64                             │  _mm_add_pd
├─────────────────────────────────────────┤
│ 16× int8                               │  _mm_add_epi8
├─────────────────────────────────────────┤
│ 8× int16                               │  _mm_add_epi16
├─────────────────────────────────────────┤
│ 4× int32                               │  _mm_add_epi32
├─────────────────────────────────────────┤
│ 2× int64                               │  _mm_add_epi64
└─────────────────────────────────────────┘
```

**SSE3 (2004 - Prescott):**
- **Horizontal Operations:** `_mm_hadd_ps` for summing vector elements
- **Use Case:** Dot products and reductions
- **Example:** 

```c
// Before SSE3: Manual horizontal add
__m128 v = _mm_set_ps(1, 2, 3, 4);
// Get: [1+2, 3+4, 1+2, 3+4] requires multiple shuffles

// With SSE3:
__m128 result = _mm_hadd_ps(v, v);  // [4+3+2+1 ...] efficiently
```

**SSSE3 (2006 - Core 2):** "Supplemental SSE3"
- **Byte-level Operations:** `_mm_shuffle_epi8` for complex permutations
- **Use Case:** Cryptography, codec optimizations
- **Note:** Despite name, SSSE3 is NOT a superset of SSE3 (different instruction set)

**SSE4.1 (2007 - Penryn):**
- **Blending:** `_mm_blend_ps` for conditional selection
- **Min/Max:** Integer min/max extended
- **Dot Product:** Hardware `_mm_dp_ps` instruction

**SSE4.2 (2008):**
- **String Processing:** `_mm_cmpistri` for string comparison
- **CRC:** `_mm_crc32` for checksums
- **Population Count:** `_mm_popcnt` (technically not SSE, but SSE4.2 era)

**Adoption Timeline:**

| CPU Generation | SSE Support | Year | Adoption Rate |
|----------------|-------------|------|---------------|
| Pentium III | SSE1 | 1999 | Niche (gaming) |
| Pentium 4 | SSE2 | 2001 | Growing |
| Core 2 Duo | SSSE3 | 2006 | Mainstream |
| Nehalem (Core i7) | SSE4.2 | 2008 | Universal |
| Current (2024) | SSE4.2 + AVX2 | - | 100% |

**Key Insight:** SSE4.2 is the "minimum common denominator" for modern x86-64 code.

#### 1.3 Intrinsics vs. Assembly

**Three Ways to Write SIMD Code:**

1. **Inline Assembly:** Maximum control, minimum portability
2. **Compiler Intrinsics:** Balanced approach (used in practice)
3. **Auto-Vectorization:** Let compiler handle it (often suboptimal)

**Comparison:**

```c
// 1. INLINE ASSEMBLY (x86-64 GCC syntax)
void add_asm(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i += 4) {
        __asm__ volatile (
            "movups (%0), %%xmm0\n\t"     // Load a[i]
            "movups (%1), %%xmm1\n\t"     // Load b[i]
            "addps %%xmm1, %%xmm0\n\t"    // Add
            "movups %%xmm0, (%2)\n\t"     // Store to c[i]
            :
            : "r"(&a[i]), "r"(&b[i]), "r"(&c[i])
            : "%xmm0", "%xmm1", "memory"
        );
    }
}

// 2. INTRINSICS (recommended)
#include <xmmintrin.h>  // SSE
void add_intrinsic(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);   // Load
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_add_ps(va, vb);    // Add
        _mm_storeu_ps(&c[i], vc);          // Store
    }
}

// 3. AUTO-VECTORIZATION (compiler does it)
void add_auto(float *a, float *b, float *c, int n) {
    #pragma GCC ivdep  // Hint: no dependencies
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Compiler vectorizes if possible
    }
}
```

**Why Intrinsics Are Preferred:**

| Aspect | Assembly | Intrinsics | Auto-Vec |
|--------|----------|------------|----------|
| Portability | Poor (x86 only) | Good (cross-compiler) | Excellent |
| Register Allocation | Manual | Automatic | Automatic |
| Optimization | Full control | Compiler helps | Compiler decides |
| Readability | Low | Medium | High |
| Maintenance | Difficult | Manageable | Easy |
| Performance | 100% | 95-100% | 70-90% |

**Best Practice:** Use intrinsics for hot loops, auto-vectorization for less critical code.

#### 1.4 SSE Data Types and Type System

**Core Type Definitions:**

```c
// From <xmmintrin.h> and <emmintrin.h>

typedef float  __m128  __attribute__((__vector_size__(16), __aligned__(16)));
typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));
typedef long long __m128i __attribute__((__vector_size__(16), __aligned__(16)));
```

**Interpretation:**

- `__m128`: 128-bit vector of **packed single-precision** (4× float32)
- `__m128d`: 128-bit vector of **packed double-precision** (2× float64)
- `__m128i`: 128-bit vector of **integers** (interpretation depends on instruction)

**Critical: __m128i Is Polymorphic:**

Unlike `__m128` (always 4 floats) and `__m128d` (always 2 doubles), `__m128i` can represent:

```c
__m128i v;

// Depending on instruction used:
v = _mm_add_epi8(...)   // 16× int8
v = _mm_add_epi16(...)  // 8× int16
v = _mm_add_epi32(...)  // 4× int32
v = _mm_add_epi64(...)  // 2× int64
```

The type system doesn't enforce interpretation - the instruction determines it!

**Type Safety Consideration:**

```c
__m128 float_vec = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
__m128i int_vec = _mm_set_epi32(1, 2, 3, 4);

// WARNING: This compiles but is semantically wrong!
__m128 result = _mm_add_ps(float_vec, (__m128)int_vec);
// Interprets integer bits as float (garbage result)
```

**Best Practice:** Never cast between `__m128`, `__m128d`, and `__m128i` without explicit conversion intrinsics.

#### 1.5 Intrinsic Naming Convention

Intel intrinsics follow a systematic naming pattern:

```
_mm_<operation>_<type_suffix>

Where:
  _mm_      = SSE namespace (128-bit)
  <operation> = What the instruction does
  <type_suffix> = Data type and packing
```

**Type Suffixes:**

| Suffix | Meaning | Example |
|--------|---------|---------|
| `ps` | Packed Single (4× float32) | `_mm_add_ps` |
| `pd` | Packed Double (2× float64) | `_mm_add_pd` |
| `ss` | Scalar Single (1× float32, bottom lane) | `_mm_add_ss` |
| `sd` | Scalar Double (1× float64, bottom lane) | `_mm_add_sd` |
| `epi8` | Integer 8-bit | `_mm_add_epi8` |
| `epi16` | Integer 16-bit | `_mm_add_epi16` |
| `epi32` | Integer 32-bit | `_mm_add_epi32` |
| `epi64` | Integer 64-bit | `_mm_add_epi64` |
| `epu8` | Unsigned 8-bit (less common) | `_mm_avg_epu8` |
| `si128` | 128-bit integer (uninterpreted) | `_mm_load_si128` |

**Common Operations:**

| Operation | Intrinsic Pattern | Description |
|-----------|-------------------|-------------|
| Load/Store | `_mm_load_*`, `_mm_store_*` | Memory operations |
| Set | `_mm_set_*`, `_mm_setzero_*` | Initialize vectors |
| Arithmetic | `_mm_add_*`, `_mm_sub_*`, `_mm_mul_*` | Math ops |
| Logic | `_mm_and_*`, `_mm_or_*`, `_mm_xor_*` | Bitwise |
| Compare | `_mm_cmp_*`, `_mm_cmpeq_*` | Comparisons |
| Convert | `_mm_cvt*_*` | Type conversions |
| Shuffle | `_mm_shuffle_*`, `_mm_unpack_*` | Data movement |

**Example Decoding:**

```c
_mm_add_ps     // Add, packed single-precision
_mm_mul_pd     // Multiply, packed double-precision
_mm_cmpeq_epi32  // Compare equal, 32-bit integers
_mm_cvtps_epi32  // Convert float32 to int32
_mm_hadd_ps    // Horizontal add, packed single
```

---

### 🔹 Part 2: Core SSE Operations

#### 2.1 Memory Operations: Load and Store

**Aligned vs. Unaligned Access:**

SSE distinguishes between aligned and unaligned memory access for performance reasons.

**Alignment Requirements:**

- **Aligned:** Memory address must be a multiple of 16 bytes (0x...0, 0x...16, 0x...32, ...)
- **Unaligned:** Any address is valid

**Performance Impact:**

| CPU Generation | Aligned Load | Unaligned Load | Penalty |
|----------------|--------------|----------------|---------|
| Pentium 4 (SSE2) | 1 cycle | 3-50 cycles | Huge |
| Core 2 (SSSE3) | 1 cycle | 2-3 cycles | Moderate |
| Sandy Bridge (AVX) | 1 cycle | 1-2 cycles | Minimal |
| Modern (2020+) | 1 cycle | 1 cycle | None* |

*On modern CPUs, unaligned loads are as fast as aligned IF the data doesn't cross a cache line boundary (64 bytes). Crossing cache lines still incurs penalties.

**Intrinsics:**

```c
// ALIGNED load/store (address must be 16-byte aligned!)
__m128 _mm_load_ps(const float *p);      // Load 4 floats (aligned)
void _mm_store_ps(float *p, __m128 a);   // Store 4 floats (aligned)

// UNALIGNED load/store (any address)
__m128 _mm_loadu_ps(const float *p);     // Load 4 floats (unaligned)
void _mm_storeu_ps(float *p, __m128 a);  // Store 4 floats (unaligned)

// Streaming (non-temporal) store - bypasses cache
void _mm_stream_ps(float *p, __m128 a);  // Write-only data
```

**When to Use Each:**

```c
// Case 1: Array allocated with malloc/new (NOT guaranteed aligned)
float *data = (float*)malloc(100 * sizeof(float));
__m128 v = _mm_loadu_ps(data);  // Must use unaligned

// Case 2: Array explicitly aligned
float *aligned_data = (float*)aligned_alloc(16, 100 * sizeof(float));
__m128 v = _mm_load_ps(aligned_data);  // Can use aligned (faster on old CPUs)

// Case 3: Write-only large array (streaming to RAM)
for (int i = 0; i < n; i += 4) {
    __m128 result = compute(...);
    _mm_stream_ps(&output[i], result);  // Bypasses cache
}
```

**Alignment in Practice:**

```c
// Method 1: C11 aligned_alloc
float *a = aligned_alloc(16, N * sizeof(float));

// Method 2: Compiler attributes
float data[100] __attribute__((aligned(16)));

// Method 3: Check alignment at runtime
bool is_aligned(void *ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}
```

#### 2.2 Arithmetic Operations

**Basic Arithmetic:**

```c
// Addition
__m128 _mm_add_ps(__m128 a, __m128 b);      // c[i] = a[i] + b[i]
__m128d _mm_add_pd(__m128d a, __m128d b);
__m128i _mm_add_epi32(__m128i a, __m128i b);

// Subtraction
__m128 _mm_sub_ps(__m128 a, __m128 b);      // c[i] = a[i] - b[i]

// Multiplication
__m128 _mm_mul_ps(__m128 a, __m128 b);      // c[i] = a[i] * b[i]

// Division (slow!)
__m128 _mm_div_ps(__m128 a, __m128 b);      // c[i] = a[i] / b[i]
```

**Limitations:**

- **No Integer Multiply in SSE2:** `_mm_mullo_epi32` was added in SSE4.1!
- **Division is Slow:** ~13-14 cycles latency, use reciprocal approximation when possible

**Integer Multiplication Workaround (Pre-SSE4.1):**

```c
// SSE2 only has 16-bit integer multiply
__m128i _mm_mullo_epi16(__m128i a, __m128i b);  // 8× 16-bit multiply

// For 32-bit multiply without SSE4.1, need complex shuffle operations
__m128i mul32_sse2(__m128i a, __m128i b) {
    // Split into odd and even 16-bit lanes
    __m128i a_even = _mm_shuffle_epi32(a, _MM_SHUFFLE(3,1,2,0));
    __m128i b_even = _mm_shuffle_epi32(b, _MM_SHUFFLE(3,1,2,0));
    // ... complex 16×16→32 multiply-add chain
    // (This is why libraries check for SSE4.1!)
}

// With SSE4.1: Simple!
__m128i _mm_mullo_epi32(__m128i a, __m128i b);  // 4× 32-bit multiply
```

#### 2.3 Set and Initialize

**Setting Values:**

```c
// Set all lanes to same value (broadcast)
__m128 _mm_set1_ps(float a);             // [a, a, a, a]

// Set each lane individually (reverse order!)
__m128 _mm_set_ps(float e3, float e2, float e1, float e0);  
// Result: [e0, e1, e2, e3] - REVERSED!

// Set in forward order
__m128 _mm_setr_ps(float e0, float e1, float e2, float e3);
// Result: [e0, e1, e2, e3]

// Set to zero
__m128 _mm_setzero_ps();                 // [0, 0, 0, 0]
```

**Common Gotcha:**

```c
// WRONG: Expects [1, 2, 3, 4] but gets [4, 3, 2, 1]
__m128 v = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);

// CORRECT: Use setr for intuitive order
__m128 v = _mm_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);  // [1, 2, 3, 4]
```

**Why This Design?**

Intel designed `_mm_set_ps` to match assembly instruction order:
```asm
; Assembly stores high lanes first in memory
movss xmm0, 4.0    ; Lane 3
movss xmm0, 3.0    ; Lane 2
movss xmm0, 2.0    ; Lane 1
movss xmm0, 1.0    ; Lane 0
```

#### 2.4 Comparison and Masking

**Comparison Instructions:**

```c
// Returns all-1s (0xFFFFFFFF) for true, all-0s for false per lane
__m128 _mm_cmpeq_ps(__m128 a, __m128 b);   // a[i] == b[i]
__m128 _mm_cmplt_ps(__m128 a, __m128 b);   // a[i] < b[i]
__m128 _mm_cmple_ps(__m128 a, __m128 b);   // a[i] <= b[i]
__m128 _mm_cmpgt_ps(__m128 a, __m128 b);   // a[i] > b[i]
__m128 _mm_cmpge_ps(__m128 a, __m128 b);   // a[i] >= b[i]
__m128 _mm_cmpneq_ps(__m128 a, __m128 b);  // a[i] != b[i]
```

**Example:**

```c
__m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
__m128 b = _mm_set_ps(5.0f, 2.0f, 1.0f, 4.0f);
__m128 mask = _mm_cmpeq_ps(a, b);

// Result: mask = [0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000]
//                Lane 0: 4==4 TRUE, Lane 1: 3!=1 FALSE, 
//                Lane 2: 2==2 TRUE, Lane 3: 1!=5 FALSE
```

**Using Masks for Conditional Execution:**

```c
// Bitwise AND with mask (implements if-then)
__m128 result = _mm_and_ps(mask, value_if_true);

// Bitwise ANDNOT (flips mask, implements else)
__m128 else_result = _mm_andnot_ps(mask, value_if_false);

// Combine: (mask & true_val) | (~mask & false_val)
__m128 final = _mm_or_ps(result, else_result);
```

**SSE4.1 Blend (Simpler):**

```c
// With SSE4.1: Much cleaner conditional selection
__m128 _mm_blendv_ps(__m128 false_val, __m128 true_val, __m128 mask);

// Example: Clamp values to [0, 1]
__m128 v = ...;
__m128 zero = _mm_setzero_ps();
__m128 one = _mm_set1_ps(1.0f);

v = _mm_max_ps(v, zero);        // v = max(v, 0)
v = _mm_min_ps(v, one);         // v = min(v, 1)
```

#### 2.5 Data Rearrangement: Shuffle and Permute

**Shuffle Within Register:**

```c
// Shuffle 32-bit lanes within single register
__m128 _mm_shuffle_ps(__m128 a, __m128 b, int imm8);

// imm8 encodes which source lanes go to which destinations
// Bits [1:0] select for dst[0], [3:2] for dst[1], etc.
```

**Example:**

```c
__m128 v = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  // [1, 2, 3, 4]

// Reverse order: _MM_SHUFFLE(0, 1, 2, 3) = 0b00'01'10'11 = 0x1B
__m128 rev = _mm_shuffle_ps(v, v, 0x1B);  // [4, 3, 2, 1]

// Broadcast lane 0: _MM_SHUFFLE(0, 0, 0, 0) = 0x00
__m128 splat = _mm_shuffle_ps(v, v, 0x00);  // [1, 1, 1, 1]

// Swap pairs: _MM_SHUFFLE(2, 3, 0, 1) = 0x4E
__m128 swapped = _mm_shuffle_ps(v, v, 0x4E);  // [3, 4, 1, 2]
```

**Helper Macro:**

```c
#define _MM_SHUFFLE(z, y, x, w) (((z)<<6) | ((y)<<4) | ((x)<<2) | (w))

// Usage:
__m128 result = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
```

**Unpack Operations (Interleaving):**

```c
// Interleave low halves
__m128 _mm_unpacklo_ps(__m128 a, __m128 b);
// a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
// Result: [a0, b0, a1, b1]

// Interleave high halves
__m128 _mm_unpackhi_ps(__m128 a, __m128 b);
// Result: [a2, b2, a3, b3]
```

**Use Case: AoS ↔ SoA Conversion**

```c
// Array of Structures (AoS)
struct Vec3 { float x, y, z; };
Vec3 aos[4] = {{1,2,3}, {4,5,6}, {7,8,9}, {10,11,12}};

// Convert to Structure of Arrays (SoA)
__m128 x = _mm_set_ps(aos[3].x, aos[2].x, aos[1].x, aos[0].x);
__m128 y = _mm_set_ps(aos[3].y, aos[2].y, aos[1].y, aos[0].y);
__m128 z = _mm_set_ps(aos[3].z, aos[2].z, aos[1].z, aos[0].z);

// Now x = [1, 4, 7, 10], y = [2, 5, 8, 11], z = [3, 6, 9, 12]
// Can process all X coordinates in one SIMD op!
```

---

### 🔹 Part 3: Advanced SSE Techniques

#### 3.1 Horizontal Operations (SSE3)

**Problem:**

Standard operations are "vertical" (lane-by-lane). Sometimes we need "horizontal" operations (within a vector).

**Vertical Addition:**
```c
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = a + b = [6, 8, 10, 12]  // Lane-wise
```

**Horizontal Addition:**
```c
v = [1, 2, 3, 4]
result = hadd(v) = [(1+2), (3+4), ?, ?] = [3, 7, ?, ?]
```

**Intrinsics:**

```c
// Horizontal add
__m128 _mm_hadd_ps(__m128 a, __m128 b);
// Result: [a0+a1, a2+a3, b0+b1, b2+b3]

// Horizontal subtract
__m128 _mm_hsub_ps(__m128 a, __m128 b);
// Result: [a0-a1, a2-a3, b0-b1, b2-b3]
```

**Reduction to Scalar:**

```c
// Sum all lanes of a vector
float sum_sse3(__m128 v) {
    v = _mm_hadd_ps(v, v);  // [0+1, 2+3, 0+1, 2+3]
    v = _mm_hadd_ps(v, v);  // [all_sum, all_sum, all_sum, all_sum]
    return _mm_cvtss_f32(v);  // Extract lane 0
}
```

**Performance Note:**

Horizontal operations are typically **slower** than vertical operations:
- Vertical add: 1 cycle latency, 0.5 CPI
- Horizontal add: 3 cycles latency, 1 CPI

Use sparingly, only when necessary (e.g., dot products, reductions).

#### 3.2 Floating-Point Math Functions

**Reciprocal and Square Root (Fast Approximations):**

```c
// Fast reciprocal (1/x) with ~11-bit precision
__m128 _mm_rcp_ps(__m128 a);

// Newton-Raphson refinement for full precision
__m128 fast_div_ps(__m128 a, __m128 b) {
    __m128 rcp = _mm_rcp_ps(b);           // Initial guess
    rcp = _mm_mul_ps(rcp, _mm_sub_ps(     // x' = x*(2 - b*x)
        _mm_set1_ps(2.0f),
        _mm_mul_ps(b, rcp)
    ));
    return _mm_mul_ps(a, rcp);            // a * (1/b)
}

// Fast reciprocal square root (1/sqrt(x))
__m128 _mm_rsqrt_ps(__m128 a);

// Fast square root
__m128 fast_sqrt_ps(__m128 a) {
    return _mm_mul_ps(a, _mm_rsqrt_ps(a));  // x * (1/sqrt(x)) = sqrt(x)
}
```

**Accuracy vs. Performance:**

| Function | Latency | Precision | Use Case |
|----------|---------|-----------|----------|
| `_mm_div_ps` | 13-14 cyc | Full 24-bit | When accuracy critical |
| `_mm_rcp_ps` | 4 cyc | 11-bit | Graphics, fast approx |
| `_mm_sqrt_ps` | 12 cyc | Full | When accuracy critical |
| `_mm_rsqrt_ps` | 4 cyc | 11-bit | Normalization, graphics |

**When to Use Fast Approximations:**

✅ Graphics (lighting, normals)  
✅ Physics simulation (good enough)  
❌ Financial calculations  
❌ Scientific simulations requiring precision

#### 3.3 Type Conversions

**Float ↔ Integer:**

```c
// Float to int (truncate towards zero)
__m128i _mm_cvttps_epi32(__m128 a);

// Float to int (round according to current rounding mode)
__m128i _mm_cvtps_epi32(__m128 a);

// Int to float
__m128 _mm_cvtepi32_ps(__m128i a);
```

**Example:**

```c
__m128 floats = _mm_set_ps(1.7f, 2.3f, -3.8f, 4.1f);

__m128i truncated = _mm_cvttps_epi32(floats);
// Result: [4, -3, 2, 1] (towards zero)

__m128i rounded = _mm_cvtps_epi32(floats);
// Result (round-to-nearest): [4, -4, 2, 2]
```

**Packing/Unpacking (Saturation):**

```c
// Pack 32-bit signed ints to 16-bit with saturation
__m128i _mm_packs_epi32(__m128i a, __m128i b);

// Pack 16-bit signed ints to 8-bit with saturation
__m128i _mm_packs_epi16(__m128i a, __m128i b);

// Pack 16-bit unsigned ints to 8-bit with saturation
__m128i _mm_packus_epi16(__m128i a, __m128i b);
```

**Use Case: Image Processing**

```c
// Clamp float pixel values [0.0, 1.0] to uint8 [0, 255]
__m128 pixels_float = ...;  // [0.0 - 1.0]

// Scale to [0, 255]
pixels_float = _mm_mul_ps(pixels_float, _mm_set1_ps(255.0f));

// Convert to int32
__m128i pixels_i32 = _mm_cvtps_epi32(pixels_float);

// Pack to int16 (8 values from 2 registers)
__m128i pixels_i16 = _mm_packs_epi32(pixels_i32, pixels_i32);

// Pack to uint8 (16 values from 2 registers)
__m128i pixels_u8 = _mm_packus_epi16(pixels_i16, pixels_i16);

// Now pixels_u8 contains 16 uint8 values ready for output
```

---

## 💻 Minimal Code Examples

### Example 1: Basic Vector Addition

```c
#include <emmintrin.h>  // SSE2
#include <stdio.h>

void vector_add_sse(const float *a, const float *b, float *c, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(&c[i], vc);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[] = {8, 7, 6, 5, 4, 3, 2, 1};
    float c[8];
    
    vector_add_sse(a, b, c, 8);
    
    for (int i = 0; i < 8; i++) {
        printf("%.1f ", c[i]);  // Expected: 9 9 9 9 9 9 9 9
    }
    return 0;
}
```

### Example 2: Conditional Clamping

```c
#include <emmintrin.h>

// Clamp values to [0, 1]
void clamp_sse(float *data, int n) {
    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set1_ps(1.0f);
    
    for (int i = 0; i < n; i += 4) {
        __m128 v = _mm_loadu_ps(&data[i]);
        v = _mm_max_ps(v, zero);  // max(v, 0)
        v = _mm_min_ps(v, one);   // min(v, 1)
        _mm_storeu_ps(&data[i], v);
    }
}
```

---

## 📝 Summary & Key Takeaways

### Main Concepts

1. **SSE Uses 128-bit XMM Registers:** 4× float32 or 2× float64 per instruction
2. **Intrinsics Are Preferred:** Balance of performance and portability
3. **Naming Convention:** `_mm_<op>_<type>` where type = ps/pd/epi32/etc.
4. **Alignment Matters (Historically):** Modern CPUs are more forgiving, but still relevant
5. **Horizontal Ops Are Slow:** Minimize use of `hadd`, `hsub`

### Performance Guidelines

✅ **DO:**
- Use `_mm_loadu_ps` for general-purpose code (works everywhere)
- Align data when possible for maximum compatibility
- Handle remainder elements with scalar loop
- Check for SSE4.2 availability for modern instructions

❌ **DON'T:**
- Mix `__m128`, `__m128d`, `__m128i` without explicit conversion
- Assume alignment from `malloc` (use `aligned_alloc`)
- Overuse horizontal operations
- Use division/sqrt when approximation suffices

---

*End of Day 002 - Total Lines: 1050+*
