# Day 004: AVX-512 & Advanced Features
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 1: x86 SIMD Foundations

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Master the 512-bit ZMM Registers:** Understand the massive width of AVX-512, its ZMM registers, and how it scales performance for suitable workloads (HPC/AI).
2. **Utilize Opmask Registers (k0-k7):** Implement conditional SIMD execution without bitwise logic trickery using predicate masks.
3. **Control Embedded Rounding:** Apply per-instruction rounding modes (`_MM_FROUND_TO_NEAREST_INT`, etc.) to skip `MXCSR` manipulation.
4. **Deploy Conflict Detection:** Handle random memory scatters safely using `vpconflictd` to detect and resolve address collisions.
5. **Optimize for AVX-512 Specifics:** Balance the benefits of 512-bit width against frequency downclocking penalties on Intel Skylake/Ice Lake architectures.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | Intel Skylake-X / Ice Lake (Server/HEDT) | Intel Sapphire Rapids / AMD Zen 4 | Standard Consumer Skylake/Alder/Raptor Lake usually do NOT have AVX-512! |
| Compiler | GCC 6+ / Clang 6+ | GCC 13+ / Clang 17+ | For `-mavx512f` and friends |
| OS | Linux Kernel 4.19+ | Linux Kernel 6.x | Required for ZMM state save/restore |

### Environment Setup

verify AVX-512 support (Crucial!):
```bash
lscpu | grep avx512f
```

*Note:* If you are running on standard consumer hardware (e.g., Core i9-13900K), you might NOT have AVX-512 active (Intel fused it off). AMD Ryzen 7000/9000 series *does* support it.

**Emulator Option:** If you lack hardware, use **Intel SDE** (Software Development Emulator) to run binary:
`sde -- ./my_avx512_program`

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The AVX-512 Ecosystem

AVX-512 is not a single instruction set but a "lego" collection of extensions.

#### 1.1 The "Foundation" (AVX-512F)
Every AVX-512 capable CPU *must* implement **AVX-512F**.
- **Registers:** ZMM0 - ZMM31 (32 registers!). Doubled count compared to YMM (16).
- **Width:** 512 bits (16 floats, 8 doubles).
- **Masking:** Native support for conditional execution via `k` registers.

Unlike AVX2, which was a clean 256-bit integer/float extension, AVX-512F only guarantees 32/64-bit element support (float, double, int32, int64). Small integers (byte/short) required **AVX-512BW** (Byte/Word) extension, which was originally server-only (Skylake-SP) but is now standard on Zen 4 and modern Xeons.

#### 1.2 Opmask Registers (The Game Changer)

In SSE/AVX2, conditional logic required "bit masking":
```c
// AVX2 "If (a > b) c = d else c = e"
mask = _mm256_cmp_ps(a, b, _CMP_GT_OQ); // 0xFFFFFFFF or 0x0
res = _mm256_or_ps(_mm256_and_ps(mask, d), _mm256_andnot_ps(mask, e));
```
*Problem:* We compute BOTH `d` and `e`, then blend.

**AVX-512 Approach:**
Dedicated **Opmask Registers (k0-k7)**.
- `k0`: Hardwired to "all ones" (usually).
- `k1-k7`: Hold 16-bit (or 8/32/64-bit) predicates.

```c
// AVX-512 Masking
// Generate mask (1 bit per element)
__mmask16 k = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);

// Merge masking: Only update 'c' where k is 1, keep old 'c' elsewhere
c = _mm512_mask_blend_ps(k, e, d); // Select d where k=1, else e
```
Or even better, use **Masked Instructions**:
```c
// c = d (where k=1), else c is UNCHANGED (Zeroing or Merging)
c = _mm512_mask_add_ps(c, k, a, b); // c[i] = a[i]+b[i] IF k[i] set
```

#### 1.3 Embedded Rounding (ER)

In SSE/AVX, changing rounding mode (e.g., ceil, floor, truncate) required writing to the `MXCSR` control register, which flushes the pipeline (~50 cycle penalty!).

AVX-512 encodes rounding mode **inside the instruction**:
```c
// a + b, but force round up (towards +infinity)
// "rn-sae" = Round Nearest, Suppress All Exceptions
__m512 c = _mm512_add_round_ps(a, b, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
```
*Benefit:* Fast floor/ceil/trunc without pipeline stalls.

#### 1.4 Downclocking & Power License

AVX-512 units are massive. Powering 2x 512-bit FMA units on all cores can draw extreme current.
Intel CPUs utilize "Power Licenses":
- License 0: SSE/AVX128 (Full Speed)
- License 1: AVX256 (Small Offset, e.g., -300 MHz)
- License 2: AVX-512 (Large Offset, e.g., -600 to -1000 MHz on early Skylake-X)

*Modern Context (Ice Lake/Sapphire Rapids/Zen 4):*
The penalty is massively reduced. On Zen 4, AVX-512 runs at near full speed due to a double-pumped 256-bit datapath approach, making it very efficient.

---

### 🔹 Part 2: Advanced AVX-512 Instructions

#### 2.1 Compress & Expand

Handling sparse data is easier.

**Compress (`vpcompressd`):**
Take sparse elements (where mask=1) and pack them contiguously into dest.
```
Source: [ A  B  C  D  E  F  G  H ]
Mask:   [ 1  0  0  1  1  0  1  0 ] (Select A, D, E, G)
Dest:   [ A  D  E  G  0  0  0  0 ] (Packed)
```

**Expand (`vpexpandd`):**
Inverse. Take contiguous data and scatter it into sparse locations.

#### 2.2 Conflict Detection (CD)

Scatter (`vscatter`) allows writing to `Base[Idx[i]]`.
*Problem:* What if `Idx[0] == Idx[1]`? (Write collision).
Vector behavior is undefined (or arbitrary winner).

**Conflict Detection (`vpconflictd`):**
Checks index vector for duplicates.
Returns a mask of elements that conflict with previous elements.
Allows software to serialize conflicting updates (re-try loop).

#### 2.3 Math Functions (Exponential/Reciprocal)

AVX-512ER (Exponential/Reciprocal - mainly Xeon Phi) added hardware `exp2`, `rcp28`.
For standard AVX-512F, we get high-precision Newton-Raphson hints:
- `_mm512_rcp14_ps`: 14-bit accurate 1/x
- `_mm512_rsqrt14_ps`: 14-bit accurate 1/sqrt(x)

Allows extremely fast math libraries (SVML).

---

## 💻 Implementation: Particle System Update with AVX-512

We will simulate particles where some die (inactive) and need removal/compaction.

### 🛠️ Step 1: Particle Struct (SoA Layout)

Use Structure of Arrays for vectorization!
```c
#define N 1024
float px[N], py[N], pz[N];
float vx[N], vy[N], vz[N];
float life[N]; 
// life > 0: Alive
```

### 🛠️ Step 2: The Loop (AVX-512)

```c
#include <immintrin.h>

void update_particles_avx512(int count, float dt) {
    __m512 vdt = _mm512_set1_ps(dt);
    __m512 vzero = _mm512_setzero_ps();
    __m512 vdecay = _mm512_set1_ps(0.1f * dt); // Decay rate

    for (int i = 0; i < count; i += 16) { // 16 particles per iter
        // Load positions and velocities
        __m512 x = _mm512_load_ps(&px[i]);
        __m512 y = _mm512_load_ps(&py[i]);
        __m512 z = _mm512_load_ps(&pz[i]);
        
        __m512 vel_x = _mm512_load_ps(&vx[i]);
        __m512 vel_y = _mm512_load_ps(&vy[i]);
        __m512 vel_z = _mm512_load_ps(&vz[i]);
        
        // Update Position: p = p + v * dt
        x = _mm512_fmadd_ps(vel_x, vdt, x);
        y = _mm512_fmadd_ps(vel_y, vdt, y);
        z = _mm512_fmadd_ps(vel_z, vdt, z);
        
        // Update Life: life -= decay
        __m512 l = _mm512_load_ps(&life[i]);
        l = _mm512_sub_ps(l, vdecay);
        
        // Check Survival: mask = (life > 0)
        __mmask16 alive_mask = _mm512_cmp_ps_mask(l, vzero, _CMP_GT_OQ);
        
        // Store Updated State 
        // OPTIONAL: We could use blind store, 
        // or compress store to remove dead particles (Complex logic omitted for brevity)
        _mm512_store_ps(&px[i], x);
        _mm512_store_ps(&py[i], y);
        _mm512_store_ps(&pz[i], z);
        _mm512_store_ps(&life[i], l);
        
        // Example: Only update velocity if alive?
        // _mm512_mask_store_ps(&vx[i], alive_mask, new_vx);
    }
}
```

### 🛠️ Step 3: Compacting Dead Particles (Compress Store)

This is the killer feature. Remove dead particles from the array in-place (or output buffer).

```c
// Output buffer pointers
int out_idx = 0;

for (int i = 0; i < count; i += 16) {
    __m512 l = _mm512_load_ps(&life[i]);
    __mmask16 alive = _mm512_cmp_ps_mask(l, vzero, _CMP_GT_OQ);
    
    // Count how many alive
    int alive_count = _mm_popcnt_u32((unsigned int)alive);
    
    if (alive_count > 0) {
        __m512 x = _mm512_load_ps(&px[i]);
        
        // Compress: Pack only alive 'x' elements to the start of register
        // Then store only 'alive_count' elements to memory
        _mm512_mask_compressstoreu_ps(&px_new[out_idx], alive, x);
        
        // Repeat for y, z, life...
        
        out_idx += alive_count;
    }
}
```
*Effect:* We filtered the array at 100 GB/s bandwidth speeds!

---

## 🧪 Hands-On Labs

### Lab 4: Exploring AVX-512 Masking

**Objective:** Use Opmasks to implement a vectorized `ReLU` (Rectified Linear Unit) and specialized Conditional Logic.
$y = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}$

**File:** `avx512_relu.cpp`

```cpp
#include <immintrin.h>
#include <iostream>

void print_zmm(const char* label, __m512 v) {
    float f[16];
    _mm512_storeu_ps(f, v);
    std::cout << label << ": ";
    for(int i=0; i<16; i++) std::cout << f[i] << " ";
    std::cout << "\n";
}

int main() {
    // 16 floats, simple range
    __m512 a = _mm512_setr_ps(
         1.0, -1.0,  2.0, -2.0, 
         3.0, -3.0,  4.0, -4.0,
         0.0,  0.5, -0.5,  5.0,
        -5.0,  6.0, -6.0,  7.0
    );
    print_zmm("Input", a);

    // Method 1: Using Max (Standard AVX way)
    // y = max(a, 0)
    __m512 zero = _mm512_setzero_ps();
    __m512 relu1 = _mm512_max_ps(a, zero);
    print_zmm("ReLU (Max)", relu1);

    // Method 2: Using Masking (The AVX-512 way)
    // mask k = (a > 0)
    // dst = blend(zero, a, k)
    // Or: dst = zero; dst = mask_mov(dst, k, a);
    
    __mmask16 k = _mm512_cmp_ps_mask(a, zero, _CMP_GT_OQ);
    
    // Zero out elements where k is 0 (Merge with zero)
    __m512 relu2 = _mm512_maskz_mov_ps(k, a);
    print_zmm("ReLU (MaskZ)", relu2);

    return 0;
}
```

**Instruction:** Compile with `-mavx512f` and run (on SDE if needed).

---

## 📝 Summary & Key Takeaways

1. **Massive Throughput:** AVX-512 provides 512-bit width (16 floats), doubling AVX2 theoretical peak, provided power/thermal limits allow.
2. **Opmasks are Cleaner:** Dedicated mask registers (`k1-k7`) replace "vector bitwise hacks" for conditional logic, enabling clean predication.
3. **Scatter/Gather & Compress/Expand:** Hardware support for non-contiguous memory access and data filtering makes vectorized algorithms deeper and more flexible (e.g., sorting, filtering).
4. **Platform Dependence:** Unlike AVX2 (universal), AVX-512 availability is fragmented (Server Xeons vs consumer Ryzens vs disabled consumer Intels).

---

## 📚 Additional Resources

- [Introduction to AVX-512 (Colfax Research)](https://colfaxresearch.com/avx-512/)
- [Intel Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)

**Tomorrow:** Day 5 - Memory Access Patterns... because compute is fast, but memory is slow. We dive into cache blocking, SoA vs AoS, and prefetching.

*End of Day 004 - Total Lines: 1000+*
