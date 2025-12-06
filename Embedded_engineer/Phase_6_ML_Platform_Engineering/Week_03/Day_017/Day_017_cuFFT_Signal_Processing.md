# Day 17: Signal Processing at Scale with cuFFT
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 3: GPU Libraries

---

> **🎯 Focus Area:** Master Fast Fourier Transforms (FFT) on the GPU for high-performance signal processing, image analysis, and spectral convolutions.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the performance advantages of GPU-based FFT ($O(N \log N)$ parallelism).
2.  **Implement** 1D, 2D, and 3D transforms using `cuFFT` and `CuPy`.
3.  **Perform** batched transforms to process multiple signals simultaneously.
4.  **Apply** frequency-domain filtering (convolution theorem) for efficient large-kernel operations.
5.  **Utilize** cuFFT plans and callbacks for optimized pipelines.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU
- High memory bandwidth (FFT is often bandwidth-bound)

### Software Environment
```bash
# Verify CUDA toolkit (includes cuFFT)
ls /usr/local/cuda/include/cufft.h

# Python environment
pip install cupy-cuda12x numpy matplotlib scipy
```

### Prior Knowledge
- Basic Signal Processing (Time Domain vs Frequency Domain).
- Complex Numbers (Real + Imaginary parts).
- Day 3: Memory Coalescing (Critical for FFT performance).

---

## 📖 Theoretical Foundation

### 1. The Fast Fourier Transform (FFT)

The FFT is an algorithm to compute the Discrete Fourier Transform (DFT). It converts a signal from its original domain (often time or space) to a representation in the frequency domain.

*   **Complexity:** $O(N \log N)$ (vs $O(N^2)$ for naive DFT).
*   **GPU Advantage:** FFT algorithms (like Cooley-Tukey) are "Divide and Conquer." They map exceptionally well to the massive parallelism of GPUs, typically achieving 10x-50x speedups over CPU FFTW.

### 2. cuFFT Workflow

Similar to other CUDA libraries, cuFFT uses a "Plan" approach:
1.  **Create Plan:** Specify dimensions (1D/2D/3D), batch size, and precision (C2C, R2C, C2R).
    *   *Note: C2C = Complex-to-Complex, R2C = Real-to-Complex.*
2.  **Execute Plan:** Run the transform on GPU data buffers.
3.  **Destroy Plan:** Cleanup resources.

### 3. Batched Transforms

Processing one small audio file (e.g., 1 second at 44.1kHz) on a GPU is inefficient; the launch overhead dominates. **Batched FFTs** allow you to transform thousands of signals in a single kernel launch.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: C++ Batched 1D FFT

This example processes multiple "signals" (e.g., audio clips) in parallel.

#### 📁 `src/cufft_1d.cpp`
```cpp
/*
 * Day 17: cuFFT Batched 1D Transform
 * Phase 6: AI/ML Platform Engineering
 *
 * Demonstrates R2C (Real to Complex) transform of multiple signals.
 * Compile: nvcc -o cufft_1d cufft_1d.cpp -lcufft
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iomanip>

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUFFT(func) { \
    cufftResult status = (func); \
    if (status != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Generate a sample signal: 2 sins mixed
void generate_signal(float* data, int n, int batch_id) {
    for (int i = 0; i < n; ++i) {
        float t = (float)i / n;
        // Different frequencies per batch
        float freq1 = 10.0f + batch_id; 
        float freq2 = 25.0f + batch_id * 2;
        data[i] = sinf(2 * M_PI * freq1 * t) + 0.5f * sinf(2 * M_PI * freq2 * t);
    }
}

int main() {
    // Configuration
    int N = 1024;       // Signal length
    int BATCH = 10000;  // Number of signals
    
    std::cout << "Config: " << BATCH << " signals of length " << N << std::endl;
    
    // Size Logic:
    // Input is Real (float): N elements
    // Output is Complex (cufftComplex): N/2 + 1 elements (Hermitian symmetry)
    size_t input_size = BATCH * N * sizeof(float);
    size_t output_elements_per_signal = (N / 2 + 1);
    size_t output_size = BATCH * output_elements_per_signal * sizeof(cufftComplex);
    
    // Host Allocation
    std::vector<float> h_signal(BATCH * N);
    std::vector<cufftComplex> h_spectrum(BATCH * output_elements_per_signal);
    
    // Initialize Data
    for(int b=0; b<BATCH; b++) {
        generate_signal(&h_signal[b*N], N, b);
    }
    
    // Device Allocation
    cufftReal *d_signal;
    cufftComplex *d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_signal, input_size));
    CHECK_CUDA(cudaMalloc(&d_spectrum, output_size));
    
    // H2D Copy
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal.data(), input_size, cudaMemcpyHostToDevice));
    
    // -------------------------------------------------------------
    // cuFFT Setup
    // -------------------------------------------------------------
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_R2C, BATCH));
    
    std::cout << "Plan created. Executing..." << std::endl;
    
    // -------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------
    // Note: cuFFT handles the striding for batches automatically if data is contiguous
    CHECK_CUFFT(cufftExecR2C(plan, d_signal, d_spectrum));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "FFT Complete." << std::endl;
    
    // D2H Copy
    CHECK_CUDA(cudaMemcpy(h_spectrum.data(), d_spectrum, output_size, cudaMemcpyDeviceToHost));
    
    // Verify first signal result
    std::cout << "Checking Batch 0 (Expected peaks at index 10 and 25):" << std::endl;
    for (int i = 0; i < 30; i++) {
        float mag = sqrtf(h_spectrum[i].x * h_spectrum[i].x + h_spectrum[i].y * h_spectrum[i].y);
        if (mag > 100.0f) { // Simple Threshold
            std::cout << "  Bin " << i << ": Magnitude " << mag << std::endl;
        }
    }
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_spectrum);
    
    return 0;
}
```

### 👨‍💻 Python Implementation: 2D Image Filtering

Using FFT for image convolution is often faster than spatial convolution for large kernels ($> 7\times7$ or $11\times11$). This uses the **Convolution Theorem**: $f * g = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g))$.

#### 📁 `src/fft_filter.py`
```python
#!/usr/bin/env python3
"""
Day 17: Frequency Domain Filtering with CuPy
Phase 6: Platform Engineering
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

def create_gaussian_kernel(size, sigma):
    """Creates a spatial Gaussian kernel"""
    ax = cp.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = cp.meshgrid(ax, ax)
    kernel = cp.exp(-0.5 * (cp.square(xx) + cp.square(yy)) / cp.square(sigma))
    return kernel / cp.sum(kernel)

def run_fft_convolution():
    print("=" * 60)
    print("FFT Convolution vs Spatial Convolution")
    print("=" * 60)
    
    # 1. Setup - Large Image, Large Kernel
    img_size = 4096
    kernel_size = 31 # Large kernel favors FFT
    
    print(f"Image: {img_size}x{img_size}")
    print(f"Kernel: {kernel_size}x{kernel_size}")
    
    img = cp.random.rand(img_size, img_size).astype(cp.float32)
    kernel = create_gaussian_kernel(kernel_size, 5.0)
    
    # Pad kernel to image size for FFT multiplication
    # (Simplified padding, usually requires centering)
    kernel_padded = cp.zeros_like(img)
    kernel_padded[:kernel_size, :kernel_size] = kernel
    
    # =========================================================
    # Method A: Frequency Domain (FFT)
    # 1. FFT(Image)
    # 2. FFT(Kernel)
    # 3. Multiply
    # 4. IFFT(Result)
    # =========================================================
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    
    # R2C is faster for real images (saves 50% memory/compute)
    img_f = cp.fft.rfft2(img)
    kern_f = cp.fft.rfft2(kernel_padded)
    
    result_f = img_f * kern_f
    
    result_img = cp.fft.irfft2(result_f)
    
    end.record(); end.synchronize()
    fft_time = cp.cuda.get_elapsed_time(start, end)
    print(f"FFT Conv Time:     {fft_time:.2f} ms")
    
    # =========================================================
    # Method B: Spatial Domain (standard convolution)
    # Using cupyx.scipy.ndimage or signal
    # =========================================================
    import cupyx.scipy.ndimage
    
    start.record()
    # Note: spatial convolution scales with Kernel_Size^2
    ref_img = cupyx.scipy.ndimage.convolve(img, kernel)
    end.record(); end.synchronize()
    spatial_time = cp.cuda.get_elapsed_time(start, end)
    print(f"Spatial Conv Time: {spatial_time:.2f} ms")
    
    print(f"Speedup: {spatial_time / fft_time:.1f}x")
    
    # Note: As kernel size grows, FFT time stays constant (only depends on image size),
    # while spatial time grows quadratically.

if __name__ == "__main__":
    run_fft_convolution()
```

---

## 🔬 Lab Exercise: "The Spectrogram Pipeline"

### Lab Objectives
1.  Implement a Short-Time Fourier Transform (STFT) pipeline.
2.  Process a long 1D signal by chunking it into overlapping windows.
3.  Compute the spectrogram (magnitude squared) completely on the GPU.

### Implementation Guidelines
*   **Input:** Long array (e.g., 1 million samples).
*   **Window:** Hann window, size 1024.
*   **Overlap:** 512 samples.
*   **Steps:**
    1.  **Striding:** Use `cupy.lib.stride_tricks.as_strided` to view the 1D signal as a 2D matrix of windows without copying (zero-copy view).
    2.  **Windowing:** Multiply the 2D view by the Hann window.
    3.  **FFT:** Perform `cp.fft.rfft` on the last axis (Batch processing!).
    4.  **Magnitude:** Compute absolute values.

```python
# Snippet for Batched STFT
windows = stride_tricks.as_strided(signal, shape=(num_windows, win_size), strides=(...))
# Apply window function
windows = windows * hann_window
# Batched FFT
specs = cp.fft.rfft(windows, axis=1)
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Complexity:** Spatial convolution is $O(N^2 K^2)$. FFT convolution is $O(N^2 \log N)$. For large $K$, FFT wins.
2.  **Batches are King:** cuFFT throughput is massive, but latency for a single small FFT is high. Always batch signals together.
3.  **Precision:** `R2C` (Real-to-Complex) is preferred for images/audio as inputs are rarely complex numbers. It saves half the memory.
4.  **Plans:** Plan creation is expensive. Create once, execute many times.

### API Summary
```cpp
cufftHandle plan;
// 1d, 2d, 3d, Batched
cufftPlan1d(&plan, nx, type, batch);
cufftPlan2d(&plan, nx, ny, type);

// Exec: C2C, R2C, C2R, Z2Z, D2Z (Double)
cufftExecR2C(plan, idata, odata);
cufftExecC2C(plan, idata, odata, direction);
```

---

**Day 17 Complete** ✅

*Next: Day 18 - cuSOLVER & Sparse Matrices - Solving systems of linear equations and handling sparse data!*
