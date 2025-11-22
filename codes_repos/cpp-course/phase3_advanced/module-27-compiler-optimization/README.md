# Module 27: Compiler and Optimization

## Overview
Understanding compiler internals, optimization techniques, and how to write compiler-friendly code.

## Learning Objectives
By the end of this module, you will be able to:
- Understand the compilation process
- Use different optimization levels
- Write inline assembly
- Apply compiler intrinsics
- Leverage link-time optimization
- Read assembly output

## Key Concepts

### 1. Compilation Process
Understanding the stages of compilation.

```
Source Code → Preprocessor → Compiler → Assembler → Linker → Executable
   (.cpp)        (.i)         (.s)        (.o)       (.exe)
```

### 2. Optimization Levels
Different compiler optimization flags.

```bash
g++ -O0  # No optimization (debug)
g++ -O1  # Basic optimization
g++ -O2  # Recommended optimization
g++ -O3  # Aggressive optimization
g++ -Os  # Optimize for size
g++ -Ofast  # Fastest (may break standards)
```

### 3. Inline Assembly
Embedding assembly in C++.

```cpp
int add(int a, int b) {
    int result;
    asm("addl %1, %0"
        : "=r" (result)
        : "r" (a), "0" (b));
    return result;
}
```

### 4. Compiler Intrinsics
Using compiler-specific optimizations.

```cpp
#include <immintrin.h>

// SIMD intrinsics
__m256 a = _mm256_load_ps(data);
__m256 b = _mm256_mul_ps(a, a);

// Prefetch
__builtin_prefetch(ptr, 0, 3);

// Expect
if (__builtin_expect(rare_condition, 0)) {
    // Unlikely path
}
```

### 5. Link-Time Optimization (LTO)
Whole-program optimization.

```bash
g++ -flto -O3 file1.cpp file2.cpp -o program
```

### 6. Reading Assembly
Understanding compiler output.

```cpp
int square(int x) {
    return x * x;
}

// Assembly (x86-64):
// mov eax, edi
// imul eax, eax
// ret
```

## Rust Comparison

### Optimization
**C++:**
```bash
g++ -O3 code.cpp
```

**Rust:**
```bash
cargo build --release
```

### Inline Assembly
**C++:**
```cpp
asm("mov %0, %1" : "=r"(out) : "r"(in));
```

**Rust:**
```rust
use std::arch::asm;
unsafe { asm!("mov {}, {}", out(reg) result, in(reg) value); }
```

## Labs

1. **Lab 27.1**: Compilation Stages
2. **Lab 27.2**: Optimization Levels
3. **Lab 27.3**: Reading Assembly Output
4. **Lab 27.4**: Inline Assembly Basics
5. **Lab 27.5**: Compiler Intrinsics
6. **Lab 27.6**: Prefetching and Hints
7. **Lab 27.7**: Link-Time Optimization
8. **Lab 27.8**: Profile-Guided Optimization
9. **Lab 27.9**: Compiler Explorer Usage
10. **Lab 27.10**: Optimization Analysis (Capstone)

## Additional Resources
- Compiler Explorer (godbolt.org)
- GCC optimization manual
- Intel Intrinsics Guide
- Agner Fog's optimization guides

## Next Module
After completing this module, proceed to **Module 28: Advanced Topics**.
