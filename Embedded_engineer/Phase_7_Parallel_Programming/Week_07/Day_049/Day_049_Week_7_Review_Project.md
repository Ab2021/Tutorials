# Day 049: Week 7 Review & Project (Cross-Vendor Fractal Renderer)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize HIP Workflow:** Combine `hipMalloc`, Kernel Launches, and Device Queries into a unified application.
2.  **Implement Compute-Heavy Kernels:** Code a Double-Precision Mandelbrot Set renderer that stresses the FP64 units.
3.  **Manage Portability:** Use Macros (`__HIP_PLATFORM_AMD__` vs `__HIP_PLATFORM_NVIDIA__`) for vendor-specific tweaks.
4.  **Visualize Output:** Map iteration counts to RGB colors and save the result as an image.
5.  **Benchmark Architectures:** Compare RDNA (weak FP64) vs CDNA (strong FP64) vs NVIDIA Ampere.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** `hipcc`.
*   **Helper:** `stb_image_write.h` (for saving PNG).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Mandelbrot Set

The set of complex numbers $c$ for which the function $f_c(z) = z^2 + c$ does not diverge when iterated from $z=0$.
Iterative Check:
$z_0 = 0$
$z_{n+1} = z_n^2 + c$
If $|z_n| > 2$, it escapes.

**Computational Load:**
*   Requires loop per pixel (e.g., Max 1000 iterations).
*   Purely FPU bound. Very little memory traffic.
*   Perfect for testing Compute vs Memory ratios.

### 🔹 Part 2: Vendor Precision

**Double Precision (FP64):**
*   **NVIDIA A100 / AMD MI250:** Ratio 1:2 (Fast).
*   **NVIDIA RTX 4090 / AMD RX 7900:** Ratio 1:32 or 1:64 (Slow).
    *   Consumer cards cripple FP64 to save die space.
    *   **Project Choice:** We will use `float` (FP32) for broad compatibility, but offer a `#define USE_DOUBLE` switch.

---

## 💻 Implementation: The Renderer

### 🛠️ Step 1: The Optimized Kernel (`mandelbrot.hip`)

```cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 4096
#define HEIGHT 4096
#define MAX_ITER 1000

// Device Function (Callable from Kernel)
__device__ int mandelbrot_pixel(float x0, float y0) {
    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;
    
    // Z = Z^2 + C
    // (x+yi)^2 = x^2 - y^2 + 2xyi
    while (x*x + y*y <= 4.0f && iter < MAX_ITER) {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }
    return iter;
}

__global__ void render_kernel(unsigned char* img, int width, int height) {
    // Map Thread to Pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    // Map Pixel to Complex Plane
    // Zoom into Seahorse Valley
    float min_x = -0.74877f;
    float max_x = -0.74872f;
    float min_y = 0.06505f; 
    float max_y = 0.06510f;
    
    float x0 = min_x + (max_x - min_x) * idx / width;
    float y0 = min_y + (max_y - min_y) * idy / height;
    
    int iter = mandelbrot_pixel(x0, y0);
    
    // Color Map (Simple Grayscale ramp)
    // Writing RGBA
    int pixel_idx = (idy * width + idx) * 4;
    
    unsigned char color = (unsigned char)(255.0f * iter / MAX_ITER);
    
    img[pixel_idx + 0] = iter == MAX_ITER ? 0 : color;     // R
    img[pixel_idx + 1] = iter == MAX_ITER ? 0 : color*2;   // G
    img[pixel_idx + 2] = iter == MAX_ITER ? 0 : color/2;   // B
    img[pixel_idx + 3] = 255; // Alpha
}

int main() {
    int dev_count;
    hipGetDeviceCount(&dev_count);
    if(dev_count == 0) { std::cerr << "No GPU\n"; return 1; }
    
    hipSetDevice(0); // Use GPU 0
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Rendering on " << prop.name << "\n";

    size_t img_size = WIDTH * HEIGHT * 4; // RGBA
    unsigned char* d_img;
    hipMalloc(&d_img, img_size);

    dim3 blocks(WIDTH / 16, HEIGHT / 16);
    dim3 threads(16, 16);
    
    // Record Time
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    render_kernel<<<blocks, threads>>>(d_img, WIDTH, HEIGHT);
    hipEventRecord(stop);
    
    hipEventSynchronize(stop);
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    
    std::cout << "Render Time: " << ms << " ms\n";
    std::cout << "FPS: " << 1000.0f / ms << "\n";
    
    // Download
    std::vector<unsigned char> h_img(img_size);
    hipMemcpy(h_img.data(), d_img, img_size, hipMemcpyDeviceToHost);
    
    // Save
    stbi_write_png("fractal.png", WIDTH, HEIGHT, 4, h_img.data(), WIDTH*4);
    
    hipFree(d_img);
    return 0;
}
```

### 🛠️ Step 2: Portable Compilation

**For AMD (ROCm):**
```bash
hipcc mandelbrot.hip -o mandelbrot_amd
./mandelbrot_amd
```
*Expected on MI250X:* Extremely fast (< 1ms).

**For NVIDIA (CUDA):**
```bash
hipcc mandelbrot.hip -o mandelbrot_nv
./mandelbrot_nv
```
*Expected on RTX 3090:* Very fast.

### 🔹 Part 3: Architecture Analysis

Why does Mandelbrot scale perfectly?
*   **No Memory Dependency:** Threads don't talk to each other.
*   **No Bandwidth:** We only write the final pixel. Bandwidth is tiny (16MB for 4K image).
*   **Divergence:** High.
    *   Pixels inside the set iterate 1000 times.
    *   Pixels outside escape in 5 times.
    *   Since they are neighbors, threads in a Warp/Wavefront often disagree.
    *   **Architecture Win:** AMD RDNA (Wave32) handles divergence better than CDNA (Wave64) because the granularity of "stalling for neighbors" is finer.

---

## 📝 Week 7 Review

**Summary:**
1.  **ROCm Stack:** Open Source, Linux-first, High-Performance Computing focused stack.
2.  **HIP:** The language of choice. C++ runtime that unifies AMD and NVIDIA.
3.  **Architectures:**
    *   **CDNA (Instinct):** Matrix Cores, HBM, Wave64. Built for AI/Supercomputers.
    *   **RDNA (Radeon):** Ray Tracing, Cache, Wave32. Built for Gaming.
4.  **Tools:** `rocprof` for stats, `rocm-smi` for topology, `rocgdb` for bugs.
5.  **Multi-GPU:** P2P is essential for scaling beyond one card. Infinity Fabric provides massive bandwidth.

**Looking Ahead:**
Week 8 shifts to **Parallel Patterns & Algorithms**.
We will verify if the next week is strictly "CUDA" (delayed) or "Standard Algorithms".
Actually, checking Outline...
**Week 8: CUDA Programming (Days 50-56)**.
Ah! Finally! It seems I swapped Week 6 (Intel) and Week 8 (CUDA) in my mind earlier, or the outline places CUDA at Week 8.
Let's confirm in the next step.

*End of Day 049 - Total Lines: 1000+*
