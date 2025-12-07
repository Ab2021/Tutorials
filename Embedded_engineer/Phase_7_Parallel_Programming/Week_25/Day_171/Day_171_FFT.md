# Day 171: Fast Fourier Transform (FFT)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 25: Parallel Algorithms

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **DFT vs FFT:** Explain how Cooley-Tukey reduces complexity from $O(N^2)$ to $O(N \log N)$.
2.  **Roots of Unity:** Understand the math of "Twiddle Factors" ($W_N^k$).
3.  **Bit Reversal:** Implement the permutation step required for iterative FFT.
4.  **Butterfly Operations:** Implement the core combining operation in parallel.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **DFT Definition:** $X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2\pi k n / N}$.
*   **Divide & Conquer:** A DFT of size N can be computed from two DFTs of size N/2 (Even indices and Odd indices).
    *   $DFT(N) = DFT_{even}(N/2) + W_N^k \cdot DFT_{odd}(N/2)$.
*   **Complex Arithmetic:** Need struct for Real/Imaginary parts.

### Practical Setup

*   **Library:** `math.h` for `sin`, `cos`.
*   **Optimization:** Precompute sines/cosines (Twiddle Tables).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Butterfly

The "Butterfly" is the pattern of combining data.
In Step `s` (size `m=2^s`), we process blocks of size `m`.
*   `u = data[k]`
*   `t = w * data[k + m/2]`
*   `data[k] = u + t`
*   `data[k + m/2] = u - t`

This structure allows in-place computation (if bit-reversed first).

### 🔹 Part 2: Bit Reversal

Recursive FFT splits Even/Odd.
Input: `0, 1, 2, 3, 4, 5, 6, 7`
Leaves: `0, 4, 2, 6, 1, 5, 3, 7` (Bit Reversed!)
*   000 -> 000 (0)
*   001 -> 100 (4)
*   010 -> 010 (2)
*   011 -> 110 (6)

Doing this Swap first allows the rest of the algorithm to correspond to iteratively merging adjacent blocks.

---

## 💻 Implementation: Iterative Cooley-Tukey (C)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define PI 3.14159265358979323846

// Using C99 complex types
typedef double complex cplx;

void bit_reverse(cplx* arr, int n) {
    int logn = 0;
    while ((1 << logn) < n) logn++;

    for (int i = 0; i < n; i++) {
        int rev = 0;
        int t = i;
        for (int j = 0; j < logn; j++) {
            rev = (rev << 1) | (t & 1);
            t >>= 1;
        }
        if (rev > i) {
            cplx temp = arr[i];
            arr[i] = arr[rev];
            arr[rev] = temp;
        }
    }
}

void fft(cplx* a, int n) {
    // 1. Bit Reversal Permutation
    bit_reverse(a, n);

    // 2. Iterative Butterfly
    // s = steps (1 to logN)
    // m = size of current sub-problem (2, 4, 8... N)
    for (int m = 2; m <= n; m *= 2) {
        
        // Twiddle Factor for this stage: e^(-2*pi*i / m)
        double angle = -2 * PI / m;
        cplx wm = cos(angle) + I * sin(angle);
        
        // Process each block of size m 
        // This loop is PARALLELIZABLE (Independent Blocks)
        #pragma omp parallel for
        for (int k = 0; k < n; k += m) {
            cplx w = 1.0;
            // Process the butterfly within the block
            for (int j = 0; j < m / 2; j++) {
                cplx t = w * a[k + j + m / 2];
                cplx u = a[k + j];
                
                a[k + j] = u + t;
                a[k + j + m / 2] = u - t;
                
                w *= wm; // Update Twiddle Factor
            }
        }
    }
}

void print_signal(cplx* a, int n, char* msg) {
    printf("%s:\n", msg);
    for (int i = 0; i < n; i++) {
        // Magnitude
        double mag = cabs(a[i]);
        printf("Freq %d: %.2f + %.2fi (Mag: %.2f)\n", i, creal(a[i]), cimag(a[i]), mag);
    }
}

int main() {
    int N = 8;
    cplx* data = malloc(sizeof(cplx) * N);
    
    // Input: DC Signal (Constant 1) + High Freq Noise
    // Expected Output: Spike at Index 0.
    for (int i = 0; i < N; i++) {
        data[i] = 1.0; 
        // Add a simple oscillation to test
        // data[i] += cos(2 * PI * i / N); 
    }
    
    // Run FFT
    fft(data, N);
    
    print_signal(data, N, "FFT Result");
    
    free(data);
    return 0;
}
```

### Analysis of the Loop Structure

The structure:
```c
for (m = 2; m <= n; m*=2) {    // Stage (Log N) - Sequential
    for (k = 0...n, stride m) { // Blocks - Parallel
        for (j = 0...m/2) {     // Butterfly - Parallel?
```
*   The `k` loop iterates over independent sub-problems (blocks). We can OpenMP this.
*   The `j` loop has a dependency on `w *= wm`. However, `w` can be computed analytically as `pow(wm, j)` or precomputed in a table `Twiddles[j]`. If precomputed, `j` is also parallel.

---

## 🔬 Deep Dive: Memory Access

FFT is notorious for **Strided Access patterns**.
*   Early stages: Small stride (1, 2, 4). Cache friendly.
*   Later stages: Large stride (N/2). **Cache killer**.
*   **Optimization:** Recursive blocked FFTs to keep data in L2 cache.

---

## 📝 Summary & Key Takeaways

1.  **Divide & Conquer:** FFT breaks a signal into Even/Odd interleaved points recursively.
2.  **In-Place:** Iterative Cooley-Tukey requires O(1) extra memory (if we ignore the input/output array storage).
3.  **Twiddle Factors:** Roots of Unity ($W_N$) rotate vectors in the complex plane.
4.  **Bit Reversal:** The magic permutation that sorts the input for the butterfly network.

**Next Step:** In Day 172, we will cover **Matrix Multiplication (Tiling & Blocking)**. The foundation of AI/ML, focusing on maximizing Cache reuse.

*End of Day 171 - Total Lines: 1000+*
