# Day 21: Week 3 Review & The Accelerated Compute Engine
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 3: GPU Libraries

---

> **🎯 Focus Area:** Consolidate mastery of cuBLAS, cuDNN, cuFFT, and Thrust by building a unified **Multi-Library Compute Engine** for a scientific simulation pipeline.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** multiple NVIDIA libraries (cuBLAS, cuFFT, Thrust) into a single C++ application.
2.  **Manage** interoperability between libraries (sharing pointers, handling diverse data layouts).
3.  **Design** a pipeline that minimizes Device-to-Host transfers between library calls.
4.  **Profile** a multi-stage application to identify bottlenecks.
5.  **Review** key concepts from Weeks 1-3 (Architecture, CUDA C++, Libraries).

---

## 📚 Week 3 Recap

### Topics Covered

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 15 | cuBLAS | Dense Linear Algebra, GEMM, Memory Layouts, Tensor Cores. |
| 16 | cuDNN | Deep Learning Primitives (Conv, Pool), Descriptors, Algorithms. |
| 17 | cuFFT | Signal Processing, Plans, Batched R2C Transforms. |
| 18 | cuSOLVER/Sparse | Linear Solvers ($Ax=b$), Sparse Matrices (CSR). |
| 19 | CUTLASS | Templates for Custom High-Performance Kernels. |
| 20 | Thrust & CUB | High-Productivity Algorithms (Sort, Reduce, Scan). |

### The "Platform Engineering" Stack
You now possess the full stack of skills:
1.  **Low Level:** Writing raw kernels (Week 1 & 2).
2.  **Mid Level:** Using CUB/CUTLASS templates (Week 3).
3.  **High Level:** Using cuBLAS/cuDNN/Thrust libraries (Week 3).

---

## 🏗️ Week 3 Project: The Accelerated Compute Engine

### Project Overview
We will build a high-performance **Signal correlation & Analysis Pipeline**. This mimics workloads in Radar processing, Seismic imaging, or Quantitative Finance.

**Use Case:**
We have a massive stream of noisy sensor data.
1.  **Ingest:** Generate synthetic data on GPU (Thrust).
2.  **Filter:** Denoise using Frequency Domain analysis (cuFFT).
3.  **Correlate:** Compute correlation matrix of signals (cuBLAS).
4.  **Analyze:** Find statistical outliers (Thrust/CUB).

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ACCELERATED COMPUTE ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [1. Ingest]       [2. Filter]          [3. Correlate]      [4. Analyze]     │
│  (Thrust)          (cuFFT)              (cuBLAS)            (Thrust)         │
│                                                                              │
│  Raw Data ──────▶  FFT R2C  ──────▶     GEMM (A * A^T) ──▶  Reduce/Sort      │
│  (Random)          Mask High Freq                                            │
│                    IFFT C2R                                                  │
│                                                                              │
│  Memory:           Memory:               Memory:             Memory:         │
│  Device Vector     Complex Buffers       Dense Matrix        Scalars         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Complete Project Implementation

This is a monolithic C++ file integrating 3 different libraries.

#### 📁 `src/compute_engine.cu`
```cpp
/*
 * Day 21 Project: Accelerated Compute Engine
 * Phase 6: AI/ML Platform Engineering
 *
 * Integrates Thrust, cuFFT, and cuBLAS.
 * 
 * Compile: 
 *   nvcc -o compute_engine compute_engine.cu -lcublas -lcufft -lcurand
 */

#include <iostream>
#include <vector>
#include <iomanip>

// Libraries
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>

// Error Handling Wrappers
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// -------------------------------------------------------------
// Functors
// -------------------------------------------------------------

// RNG Functor for Thrust
struct Prg {
    float a, b;
    __host__ __device__
    Prg(float _a, float _b) : a(_a), b(_b) {};

    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};

// Complex Frequency Masking Kernel (Simple Low-Pass)
// Zero out frequencies above a threshold
__global__ void frequency_mask_kernel(cufftComplex* data, int n_complex, int batch, int threshold_bin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_complex * batch;
    
    if (idx < total_elements) {
        int bin = idx % n_complex;
        // Keep DC and low frequencies (bin < threshold)
        if (bin > threshold_bin) {
            data[idx].x = 0.0f;
            data[idx].y = 0.0f;
        }
    }
}

// Rescale after IFFT (cuFFT does not normalize)
// IFFT(FFT(x)) = N * x. We must divide by N.
struct ScaleOp {
    float scale;
    ScaleOp(float _s) : scale(_s) {}
    __host__ __device__
    float operator()(float x) const { return x * scale; }
};

// -------------------------------------------------------------
// Main Engine
// -------------------------------------------------------------
int main() {
    std::cout << "=== Accelerated Compute Engine Startup ===" << std::endl;

    // Config
    int BATCH = 1000;      // Number of signals
    int N = 1024;          // Samples per signal (Time domain limits)
    int THRESHOLD = 100;   // Low pass filter bin
    
    size_t signal_size = BATCH * N;
    size_t complex_size = BATCH * (N / 2 + 1);

    // ========================================================
    // Step 1: Ingest (Thrust)
    // ========================================================
    std::cout << "[Step 1] Ingesting " << BATCH << " signals..." << std::endl;
    
    thrust::device_vector<float> d_signals(signal_size);
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    
    // Fill with random noise (-1.0 to 1.0)
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + signal_size,
                      d_signals.begin(),
                      Prg(-1.0f, 1.0f));

    // Verify raw data stats
    float max_val = *thrust::max_element(d_signals.begin(), d_signals.end());
    std::cout << "  Raw Max Amplitude: " << max_val << std::endl;

    // ========================================================
    // Step 2: Denoise (cuFFT)
    // ========================================================
    std::cout << "[Step 2] Filtering (R2C -> Mask -> C2R)..." << std::endl;

    // Allocate Complex Buffer
    cufftComplex* d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_spectrum, complex_size * sizeof(cufftComplex)));

    // Create Plans
    cufftHandle plan_fwd, plan_inv;
    CHECK_CUFFT(cufftPlan1d(&plan_fwd, N, CUFFT_R2C, BATCH));
    CHECK_CUFFT(cufftPlan1d(&plan_inv, N, CUFFT_C2R, BATCH));

    // Forward FFT
    // Thrust vector .data().get() gives raw pointer for C Libraries
    float* raw_signal_ptr = thrust::raw_pointer_cast(d_signals.data());
    
    CHECK_CUFFT(cufftExecR2C(plan_fwd, (cufftReal*)raw_signal_ptr, d_spectrum));

    // Apply Mask (Custom Kernel)
    int threads = 256;
    int blocks = (complex_size + threads - 1) / threads;
    frequency_mask_kernel<<<blocks, threads>>>(d_spectrum, (N/2+1), BATCH, THRESHOLD);
    CHECK_CUDA(cudaGetLastError());

    // Inverse FFT (In-place back to d_signals? No, typically cuFFT R2C/C2R out-of-place is safer if sizes differ padding-wise)
    // But here d_signals is large enough.
    CHECK_CUFFT(cufftExecC2R(plan_inv, d_spectrum, (cufftReal*)raw_signal_ptr));

    // Normalize (Thrust)
    thrust::transform(d_signals.begin(), d_signals.end(), d_signals.begin(), ScaleOp(1.0f / N));

    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Cleanup FFT
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_spectrum);

    std::cout << "  Filtering Complete." << std::endl;

    // ========================================================
    // Step 3: Correlation (cuBLAS)
    // ========================================================
    // Compute Covariance/Correlation Matrix of the signals: C = S * S^T
    // S is (BATCH x N). Result C is (BATCH x BATCH).
    // BATCH=1000, N=1024. C will be 1000x1000.
    
    std::cout << "[Step 3] Computing Correlation Matrix (" << BATCH << "x" << BATCH << ")..." << std::endl;

    cublasHandle_t blas_handle;
    CHECK_CUBLAS(cublasCreate(&blas_handle));

    float* d_correlation;
    CHECK_CUDA(cudaMalloc(&d_correlation, BATCH * BATCH * sizeof(float)));

    float alpha = 1.0f;
    float beta = 0.0f;

    // S * S^T
    // S is in d_signals. 
    // Is it Row-major or Column-major? 
    // Thrust vector is 1D contiguous. Logically we treat it as BATCH rows of N columns (Row Major).
    // cuBLAS is Col-Major.
    // If we pass S as is, cuBLAS sees it as N rows of BATCH cols (S^T logically).
    // So we want C = S * S^T.
    // In cuBLAS terms (ColMajor): C = A * B.
    // Use the boolean trick from Day 15 or cublasSgemm directly.
    // Let's rely on Sgemm with Transpose ops.
    
    // Op: RowMajor Matrix Multiply Logic using ColMajor cuBLAS:
    // C(RowMaj) = A(RowMaj) * B(RowMaj)
    // => C(ColMaj)^T = A(ColMaj)^T * B(ColMaj)^T
    // => C_out = B_in * A_in (swapped) logic? No, simpler to just map dimensions.
    
    // Let's assume we treat d_signals as ColMajor matrix: N rows, BATCH cols.
    // Wait, BATCH (1000) signals of length N (1024). 
    // Usually signals are rows.
    // If S is (BATCH x N) Row Major. 
    // We want G = S x S^T (BATCH x BATCH).
    // cuBLAS call:
    // C = Sgemm(N (trans), T (not trans), ...)
    // Because of the layout complexity, we will simply compute:
    // C = S * S^T. 
    // Dimensions: (Batch x N) * (N x Batch) -> (Batch x Batch).
    // M=Batch, N=Batch, K=N.
    
    // We pass d_signals as 'A' and 'B'.
    // Since input is Row Major, we tell cuBLAS Sgemm to read A as Transposed and B as Not Transposed?
    // Actually, for C = A * B (all row major):
    // Call cublasSgemm(handle, N, N, M, K, alpha, B, K, A, K, beta, C, N).
    // Swapping A and B handles the conversion.
    
    // So: C = S * S^T.
    // M=Batch, N=Batch, K=N.
    // We want C.
    // Call Sgemm(..., B=S^T, A=S, ...).
    // It's tricky. Let's simplify:
    // We will just perform a matrix multiplication C = S * S' using standard call.
    // Arguments:
    // TransA = CUBLAS_OP_T (Treat S as S^T effectively? No)
    // A standard result for correlation of Row Major data S (MxK):
    // Call cublasTsyrk (Symmetric Rank K update)?
    // Or just Sgemm.
    
    // To match Row Major C = A * B:
    // cublasSgemm(handle, OP_N, OP_N, N, M, K, alpha, B, n, A, k, beta, C, n)
    // Here A = S, B = S^T.
    // M=Batch, N=Batch, K=N.
    // C (BxB) = S (BxN) * S^T (NxB).
    // Passed to BLAS (Swapped):
    // B_blas = S^T (row maj rep) -> Transpose Layout?
    
    // Simplified: We use cublasSsyrk (Symmetric Rank-k) which is specialized for A*A^T.
    // But it expects Col Major.
    // If S is RowMajor(BATCH, N), it looks like ColMajor(N, BATCH) to BLAS.
    // Let A_col = S (NxBATCH).
    // We want Result = S * S^T = A_col^T * A_col.
    // This is C = A^T * A.
    // Using cublasSsyrk with CUBLAS_OP_T will computer A^T * A.
    
    // Params:
    // uplo: Upper/Lower
    // trans: OP_T (Use A^T) - We want A^T * A.
    // n: BATCH (Rows of result)
    // k: N (Cols of result calculation shared dim)
    // lda: Leading dim of A_col = N.
    
    CHECK_CUBLAS(cublasSsyrk(
        blas_handle,
        CUBLAS_FILL_MODE_LOWER, // Store in lower triangle
        CUBLAS_OP_T,            // Transpose A (N x Batch) -> (Batch x N)
        BATCH,                  // N result dim
        N,                      // K shared dim
        &alpha,
        raw_signal_ptr,         // A array
        N,                      // lda
        &beta,
        d_correlation,          // C array
        BATCH                   // ldc
    ));

    // Note: Result is Lower Triangular in ColMajor (which is Upper Triangular in RowMajor).
    // Valid correlation matrix.
    
    std::cout << "  Correlation Computed." << std::endl;

    // ========================================================
    // Step 4: Analysis (Thrust)
    // ========================================================
    // Find maximum correlation value (excluding diagonal self-correlation ~1.0)
    // For simplicity, just find global max
    
    thrust::device_ptr<float> dev_ptr_corr(d_correlation);
    float max_corr = *thrust::max_element(dev_ptr_corr, dev_ptr_corr + (BATCH*BATCH));
    
    std::cout << "[Step 4] Analysis Results:" << std::endl;
    std::cout << "  Max Correlation Score: " << max_corr << std::endl;

    // Cleanup
    cublasDestroy(blas_handle);
    cudaFree(d_correlation);
    
    std::cout << "=== Pipeline Complete ===" << std::endl;

    return 0;
}
```

---

## 🔬 Lab Exercise: "Profile and Optimize"

### Lab Objectives
1.  Compile the project: `nvcc -o engine compute_engine.cu -lcublas -lcufft`.
2.  Run with `nsys profile ./engine`.
3.  Analyze the timeline.

### Questions
1.  **Memory Transfers:** Are we copying data between Host and Device unnecessarily? (Hint: The code above keeps everything on Device! Only initial generation touches logic).
2.  **Overlap:** Could we run the cuFFT of Batch 2 while computing cuBLAS of Batch 1? (Yes, if we split batches and use Streams).
3.  **Kernel utilization:** Which kernel takes longest? The FFT or the Correlation `Ssyrk`?

---

## 📝 Week 3 Summary

### Mastered Libraries
1.  **cuBLAS:** The workhorse of AI.
2.  **cuDNN:** The brain of AI.
3.  **cuFFT:** The frequency analyzer.
4.  **cuSOLVER:** The engineer's tool.
5.  **Thrust:** The developer's best friend.

### Skills Acquired
- Building heterogeneous pipelines.
- Linking multiple libraries.
- Managing device memory across library boundaries.

### Looking Ahead (Week 4)
We move from **Libraries** (Building blocks) to **Inference Engines** (Production systems). Week 4 covers **TensorRT**, NVIDIA's high-performance deep learning inference optimizer. We will take trained PyTorch models and compile them for speed.

---

**Week 3 Complete! 🎉**

*You are now a proficient GPU ecosystem developer. Ready for TensorRT?*
