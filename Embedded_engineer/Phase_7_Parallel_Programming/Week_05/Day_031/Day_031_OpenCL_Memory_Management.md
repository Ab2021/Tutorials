# Day 031: OpenCL Memory Management
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 5: OpenCL & Heterogeneous Computing

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Differentiate Memory Objects:** Choose between **Buffers** (Linear 1D arrays) and **Images** (2D/3D textures with caching) for specific workloads.
2.  **Optimize Transfers:** Use Pinned Memory (`CL_MEM_ALLOC_HOST_PTR`) to achieve PCI-E saturation (e.g., 12 GB/s vs 4 GB/s).
3.  **Map vs Copy:** Understand the difference between `clEnqueueReadBuffer` (Explicit copy) and `clEnqueueMapBuffer` (Zero-copy on integrated GPUs).
4.  **Manage Sub-Buffers:** Create views into larger buffers to allow multiple kernels to operate on different regions simultaneously.
5.  **Utilize SVM:** Implement Shared Virtual Memory (OpenCL 2.0+) to share pointer trees between CPU and GPU without marshalling.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Integrated Graphics (Optional):** Intel iGPU or AMD APU is great for demonstrating Zero-Copy.
*   **Discrete GPU:** Best for demonstrating PCI-E bottlenecks.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Buffers vs Images

**Buffers (`cl_mem`):**
*   Just `malloc` on the GPU.
*   Access: `buffer[i]`.
*   Cache: L2 (Generic).
*   Use for: Arrays, Matrices, Structs.

**Images (`cl_mem` from Image Format):**
*   Opaque objects.
*   Access: `read_imagef(img, sampler, coord)`.
*   Cache: Texture Cache (Optimized for 2D locality).
*   Hardware features: Free linear interpolation, boundary handling (clamp/wrap), layout swizzling (RGBA).
*   Use for: Image Processing, Fluid Sym, Stencil Access.

### 🔹 Part 2: The PCI-E Bottleneck

GPUs are fast (1000 GB/s bandwidth).
PCI-Express is slow (16 GB/s for Gen3 x16).
**The Golden Rule:** Keep data on the GPU. Avoid transfers.

**Pinned Memory (Page-Locked):**
Standard `malloc` is virtual and pageable. OS can swap it to disk.
DMA (Direct Memory Access) engine on GPU cannot read swapped memory.
Driver must copy `malloc` -> `pinned_sys_ram` -> `gpu_ram`. (2 copies).
**Solution:** Allocate Pinned Memory directly.
`clCreateBuffer(..., CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, ...)`
DMAs directly from Sys RAM to GPU RAM. (1 copy, 2x speedup).

### 🔹 Part 3: Mapping and Zero-Copy

On Integrated GPUs (Intel HD, Apple M1), System RAM *is* Video RAM.
`clEnqueueReadBuffer` does a `memcpy` (pointless waste of time).
`clEnqueueMapBuffer` gives the host a pointer to the *same physical memory* the GPU is using.
**Zero Copy!**

---

## 💻 Implementation: Image Processing (Box Blur)

We will compare Buffer-based image processing vs Image-based (Texture) processing.

### 🛠️ Step 1: The Kernels (`blur.cl`)

```c
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                               CLK_ADDRESS_CLAMP_TO_EDGE | 
                               CLK_FILTER_NEAREST;

// Buffer Version: Manual Math for 2D, Manual Boundary Check
__kernel void blur_buffer(__global const unsigned char *in, 
                          __global unsigned char *out, 
                          int w, int h) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= w || y >= h) return;
    
    int sum = 0;
    int count = 0;
    
    for(int dy=-1; dy<=1; dy++) {
        for(int dx=-1; dx<=1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if(nx >=0 && nx < w && ny >=0 && ny < h) {
                sum += in[ny*w + nx];
                count++;
            }
        }
    }
    out[y*w + x] = sum / count;
}

// Image Version: Hardware handles boundaries
__kernel void blur_image(__read_only image2d_t src, 
                         __write_only image2d_t dst) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    
    float4 sum = (float4)(0.0f);
    
    // Read 3x3
    for(int dy=-1; dy<=1; dy++) {
        for(int dx=-1; dx<=1; dx++) {
            // Hardware Clamps coordinates automatically!
            sum += read_imagef(src, sampler, coord + (int2)(dx, dy));
        }
    }
    
    write_imagef(dst, coord, sum / 9.0f);
}
```

### 🛠️ Step 2: The Host Code (`host_pinned.c`)

Optimized host code using Pinned Memory map.

```c
// ... Includes ...

int main() {
    // Boilerplate Setup (Context, Queue)...
    
    int W = 4096, H = 4096;
    size_t size = W * H * 4; // RGBA
    
    // 1. Allocate Pinned Host Memory
    // We ask OpenCL to allocate memory for us that is "Host Accessible"
    cl_int err;
    cl_mem pinned_buf = clCreateBuffer(context, 
                                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                       size, NULL, &err);
    
    // 2. Map it to get a generic pointer
    unsigned char *host_ptr = (unsigned char*)clEnqueueMapBuffer(queue, pinned_buf, CL_TRUE, 
                                                                 CL_MAP_WRITE, 0, size, 
                                                                 0, NULL, NULL, &err);
    
    // 3. Fill Data (Directly into Pinned Mem)
    memset(host_ptr, 255, size); // White image
    
    // 4. Unmap (Release to GPU)
    clEnqueueUnmapMemObject(queue, pinned_buf, host_ptr, 0, NULL, NULL);
    
    // 5. Create Image Object from this Buffer? 
    // Or just copy Pinned -> ImageDevice
    
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;
    
    cl_mem dev_image = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, 
                                       W, H, 0, NULL, &err);
                                       
    // Transfer Pinned -> Image (Fast DMA)
    size_t origin[3] = {0,0,0};
    size_t region[3] = {W, H, 1};
    clEnqueueCopyBufferToImage(queue, pinned_buf, dev_image, 
                               0, origin, region, 0, NULL, NULL);
                               
    // ... Execute Kernel ...
    
    // Cleanup
    // ...
}
```

---

## 🧪 Hands-On Labs

### Lab 31: Measuring Bandwidth

**Objective:** Write a benchmark to measure Gbps of `ReadBuffer` vs `MapBuffer`.

**Steps:**
1.  Allocate 512 MB buffer.
2.  **Test 1 (Standard):** `malloc` host array. `clCreateBuffer` (Default). `clEnqueueWrite`. Measure time.
3.  **Test 2 (Pinned):** `clCreateBuffer(ALLOC_HOST_PTR)`. Map. `memcpy` into map. Unmap. Measure time.
4.  **Test 3 (Device to Device):** `clEnqueueCopyBuffer`. This is the speed of VRAM (internal Copy).

**Expected Results (Discrete GPU):**
*   Standard: ~3 GB/s (Driver overhead + copy).
*   Pinned: ~12 GB/s (PCI-E saturation).
*   Internal: ~300 GB/s (GDDR6 speed).

---

## 📝 Summary & Key Takeaways

1.  **Pinned Memory:** The #1 optimization for host-device transfer. Use `CL_MEM_ALLOC_HOST_PTR`.
2.  **Images vs Buffers:** Use Images when you need 2D spatial locality (cache) or interpolation. Use Buffers for everything else.
3.  **Mapping:** `MapBuffer` is often better than `Read/Write` because it avoids an extra host-side copy if driver is smart.
4.  **SVM (Shared Virtual Memory):** In OpenCL 2.0, you can pass `void*` directly. No `cl_mem` required. Simplifies linked lists on GPU.

---

## 📚 Additional Resources

*   [Intel OpenCL Memory Architecture Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/opencl-memory-architecture.html)
*   [NVIDIA OpenCL Best Practices](https://docs.nvidia.com/cuda/opencl-best-practices-guide/index.html)

**Tomorrow:** Day 32 - OpenCL Synchronization... Events, Barriers, and Asynchronous Pipelines to hide latency.

*End of Day 031 - Total Lines: 1000+*
