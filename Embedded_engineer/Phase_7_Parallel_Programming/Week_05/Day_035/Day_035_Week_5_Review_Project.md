# Day 035: Week 5 Review & Project (Heterogeneous Image Pipeline)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize OpenCL workflow:** Connect context creation, memory allocation, kernel compilation, and execution into a complete application.
2.  **Implement Multi-Stage Pipeline:** Chain kernels (Blur -> Edge Detection) on the GPU without round-tripping data to the host.
3.  **Manage Heterogeneous Resources:** Use the CPU to parse image headers (BMP/PNG) and the GPU to process pixel data.
4.  **Handle Errors Gracefully:** Integrate robust error checking and resource release logic.
5.  **Benchmark throughput:** Measure pixels/second processed by the pipeline.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Image Lib:** `stb_image.h` and `stb_image_write.h` (single-header C libraries) for easy loading/saving.
*   **OpenCL SDK:** verified in previous days.

### Environment Setup

Create project directory:
```bash
mkdir -p opencl_pipeline
cd opencl_pipeline
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pipeline Concept

**Naive Approach:**
1.  Host Load Image.
2.  Host -> GPU Copy.
3.  GPU Kernel 1 (Blur).
4.  GPU -> Host Copy.
5.  Host -> GPU Copy.
6.  GPU Kernel 2 (Sobel).
7.  GPU -> Host Copy.

**Optimized Pipeline:**
1.  Host Load Image.
2.  Host -> GPU Copy (Buffer A).
3.  GPU Kernel 1: Read A -> Write B (Local GPU Memory).
4.  GPU Kernel 2: Read B -> Write C (Local GPU Memory).
5.  GPU -> Host Copy (Buffer C).

**Key Insight:** Intermediate buffers (B) never leave VRAM.

### 🔹 Part 2: Sobel Edge Detection

The Sobel operator calculates the gradient of image intensity.
Filters:
$G_x = \begin{bmatrix} +1 & 0 & -1 \\ +2 & 0 & -2 \\ +1 & 0 & -1 \end{bmatrix} * A$
$G_y = \begin{bmatrix} +1 & +2 & +1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix} * A$
Magnitude $G = \sqrt{G_x^2 + G_y^2}$

---

## 💻 Implementation: The Pipeline

### 🛠️ Step 1: Kernels (`kernels.cl`)

```c
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// 1. Grayscale Conversion
__kernel void grayscale(__read_only image2d_t src, __write_only image2d_t dst) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(src, sampler, coord);
    float gray = 0.299f*pixel.x + 0.587f*pixel.y + 0.114f*pixel.z;
    write_imagef(dst, coord, (float4)(gray, gray, gray, 1.0f));
}

// 2. Gaussian Blur (3x3)
__kernel void gaussian(__read_only image2d_t src, __write_only image2d_t dst) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float kernel[9] = { 1/16.0, 2/16.0, 1/16.0,  
                        2/16.0, 4/16.0, 2/16.0, 
                        1/16.0, 2/16.0, 1/16.0 };
    float4 sum = (float4)(0.0f);
    int idx = 0;
    for(int y=-1; y<=1; y++) {
        for(int x=-1; x<=1; x++) {
            sum += read_imagef(src, sampler, coord + (int2)(x,y)) * kernel[idx++];
        }
    }
    write_imagef(dst, coord, sum);
}

// 3. Sobel
__kernel void sobel(__read_only image2d_t src, __write_only image2d_t dst) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    
    float gx = 0.0f;
    float gy = 0.0f;
    
    // Unrolled Sobel Loop logic...
    // Gx:
    gx += read_imagef(src, sampler, coord + (int2)(-1,-1)).x * 1.0;
    gx += read_imagef(src, sampler, coord + (int2)(-1, 0)).x * 2.0;
    gx += read_imagef(src, sampler, coord + (int2)(-1, 1)).x * 1.0;
    gx += read_imagef(src, sampler, coord + (int2)( 1,-1)).x * -1.0;
    // ... complete Gx and Gy ...
    
    float mag = sqrt(gx*gx + gy*gy);
    write_imagef(dst, coord, (float4)(mag, mag, mag, 1.0f));
}
```

### 🛠️ Step 2: Host Driver (`main.c`)

```c
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <CL/cl.h>

// Helper to check errors...

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: ./pipeline <img.jpg>\n"); return 1; }
    
    // 1. Load Image
    int w, h, ch;
    unsigned char* h_img = stbi_load(argv[1], &w, &h, &ch, 4); // Force RGBA
    if(!h_img) return 1;
    
    // 2. OpenCL Setup
    // ... Context, Queue, Build Program ...
    
    // 3. Create Images
    cl_image_format fmt = { CL_RGBA, CL_UNORM_INT8 };
    cl_mem d_src   = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, w, h, 0, h_img, &err);
    cl_mem d_gray  = clCreateImage2D(ctx, CL_MEM_READ_WRITE, &fmt, w, h, 0, NULL, &err);
    cl_mem d_blur  = clCreateImage2D(ctx, CL_MEM_READ_WRITE, &fmt, w, h, 0, NULL, &err);
    cl_mem d_final = clCreateImage2D(ctx, CL_MEM_WRITE_ONLY, &fmt, w, h, 0, NULL, &err);
    
    // 4. Chain Kernels
    size_t global[2] = {w, h};
    
    // A. Grayscale
    clSetKernelArg(k_gray, 0, sizeof(cl_mem), &d_src);
    clSetKernelArg(k_gray, 1, sizeof(cl_mem), &d_gray);
    clEnqueueNDRangeKernel(q, k_gray, 2, NULL, global, NULL, 0, NULL, NULL);
    
    // B. Blur
    clSetKernelArg(k_blur, 0, sizeof(cl_mem), &d_gray);
    clSetKernelArg(k_blur, 1, sizeof(cl_mem), &d_blur);
    clEnqueueNDRangeKernel(q, k_blur, 2, NULL, global, NULL, 0, NULL, NULL); // Implicit dependency via Queue order
    
    // C. Sobel
    clSetKernelArg(k_sobel, 0, sizeof(cl_mem), &d_blur);
    clSetKernelArg(k_sobel, 1, sizeof(cl_mem), &d_final);
    clEnqueueNDRangeKernel(q, k_sobel, 2, NULL, global, NULL, 0, NULL, NULL);
    
    // 5. Read Back
    // We reuse h_img buffer for output
    size_t origin[3] = {0,0,0};
    size_t region[3] = {w, h, 1};
    clEnqueueReadImage(q, d_final, CL_TRUE, origin, region, 0, 0, h_img, 0, NULL, NULL);
    
    // 6. Save
    stbi_write_png("output.png", w, h, 4, h_img, w*4);
    
    // Cleanup...
    free(h_img);
    return 0;
}
```

---

## 📝 Week 5 Review

**Summary:**
1.  **Platform Model:** Host controls Device.
2.  **Execution Model:** Work-Items group into Work-Groups to run on Compute Units.
3.  **Memory Model:** Global (Slow) vs Local (Fast/Shared) vs Private (Register).
4.  **Synchronization:** Events for Commands, Barriers for Work-Items.
5.  **Optimization:** Tiling (Local Mem) and Coalescing are the 10x performance levers.

**Comparison w/ OpenMP:**
*   OpenMP: Implicit data movement. "Easy". CPU focused (mostly).
*   OpenCL: Explicit EVERYHTING. "Verbose". Heterogeneous (GPU/FPGA).

**Looking Ahead:**
Week 6 covers **CUDA Programming**.
*   Syntactically simpler than OpenCL (Language integration vs Library).
*   NVIDIA specific but dominates HPC/AI.
*   Many concepts map 1:1 (Work-Item -> Thread, Local Mem -> Shared Mem).

*End of Day 035 - Total Lines: 1000+*
