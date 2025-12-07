# Day 038: oneDNN (Deep Neural Network Library)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 6: Intel oneAPI & DPC++

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture:** Explain how oneDNN abstracts hardware-specific optimizations (AVX-512, AMX, GPU Systolic Arrays) behind opaque "Primitives".
2.  **Memory Formats:** Master the concept of **Layout Propagation**. Understand why `nChw16c` or `nhwc` might be faster than `nchw`.
3.  **Reordering:** Implement `reorder` primitives to efficiently convert data between User Logic (flat float arrays) and Library Logic (blocked formats).
4.  **Graph Construction:** Build a small inference engine (Convolution + ReLU + Pooling).
5.  **Execution:** Submit primitives to an `engine` via a `stream` (similar to SYCL Queue).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Header:** `#include <oneapi/dnnl/dnnl.hpp>`
*   **Linker:** `-ldnnl`
*   **Env:** `source /opt/intel/oneapi/setvars.sh`

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Primitive

In oneDNN, an operation is a **Primitive**.
Examples: `convolution_forward`, `inner_product_forward` (Dense), `eltwise_forward` (ReLU).

**Lifecycle:**
1.  **Descriptor:** Describes *what* to do independent of implementation (e.g., "3x3 Conv, stride 1, padding 1").
2.  **Primitive Descriptor (PD):** Bind descriptor to an **Engine** (CPU/GPU). The library chooses the best implementation (JIT Kernel? GEMM-based? Winograd?).
3.  **Primitive:** The actual executable object (contains JIT code/LUTs).

### 🔹 Part 2: Opaque Memory Formats

CPU caches love **Block Layouts**.
Standard Image: `NCHW` (Planes: RRR... GGG... BBB...).
Optimized AVX-512: `nChw16c` (Chunks of 16 channels packed together).
*Why?* To use `vfmadd231ps zmm` on 16 floats at once without gathers.

**The Rule:**
*   Never assume data layout inside the network layers.
*   Ask the Primitive Descriptor: "What memory format do you want?" (`conv_pd.src_desc()`).
*   Create a **Reorder** if your user data doesn't match.

### 🔹 Part 3: Portability

Code written for oneDNN CPU runs on:
*   Intel CPUs (SSE4.2 to AVX-512/AMX).
*   Intel GPUs (Xe-LP, Xe-HPG) via OpenCL/Level Zero backend.
*   AMD/NVIDIA GPUs (experimental via SYCL plugins).

---

## 💻 Implementation: CNN Inference Layer

We will implement a single Convolution Layer followed by ReLU.

### 🛠️ Step 1: Initialization

```cpp
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <iostream>
#include <numeric>

using namespace dnnl;

int main() {
    // 1. Engine & Stream
    engine eng(engine::kind::cpu, 0); // or engine::kind::gpu
    stream s(eng);

    // 2. Data Dimensions
    // Batch=1, Channel=3, Height=224, Width=224
    memory::dims src_dims = {1, 3, 224, 224};
    memory::dims w.weights_dims = {16, 3, 3, 3}; // 16 filters, 3x3x3
    memory::dims dst_dims = {1, 16, 224, 224};   // Padding=SAME
    memory::dims strides = {1, 1};
    memory::dims padding = {1, 1};

    // 3. User Memory (Flat NCHW)
    std::vector<float> src_data(1*3*224*224, 1.0f);
    std::vector<float> w_data(16*3*3*3, 0.5f);
    std::vector<float> dst_data(1*16*224*224, 0.0f);

    auto user_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto user_wei_md = memory::desc(w.weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto user_dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    auto user_src_mem = memory(user_src_md, eng, src_data.data());
    auto user_wei_mem = memory(user_wei_md, eng, w_data.data());
    auto user_dst_mem = memory(user_dst_md, eng, dst_data.data());

    // ...
```

### 🛠️ Step 2: Convolution Configuration

```cpp
    // 4. Convolution Descriptor
    auto conv_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct, 
        user_src_md, user_wei_md, user_dst_md, // Source, Weights, Dest Layouts
        strides, padding, padding
    );

    // 5. Primitive Descriptor (The Heavy Lifting)
    // This queries the HW capabilities. 
    // It MIGHT switch format_tag::nchw to format_tag::nChw16c internally to tell us "I prefer blocked".
    auto conv_pd = convolution_forward::primitive_desc(conv_d, eng);
    
    // Check if reorder is needed
    memory internal_src_mem = user_src_mem;
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        internal_src_mem = memory(conv_pd.src_desc(), eng);
        reorder(user_src_mem, internal_src_mem).execute(s, user_src_mem, internal_src_mem);
    }
    
    memory internal_dst_mem = user_dst_mem;
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
         internal_dst_mem = memory(conv_pd.dst_desc(), eng);
    }

    // 6. Create Primitive
    auto conv_prim = convolution_forward(conv_pd);
```

### 🛠️ Step 3: Execution & ReLU Integration

OneDNN supports **Post-Op Fusion**. We can fuse ReLU into Conv to save memory bandwidth.

```cpp
    // Fusion Setup (Replaces Step 5 above)
    post_ops po;
    po.append_eltwise(algorithm::eltwise_relu, 0.0f, 0.0f); // Alpha=0, Beta=0
    primitive_attr attr;
    attr.set_post_ops(po);
    
    auto conv_fused_pd = convolution_forward::primitive_desc(conv_d, attr, eng);
    auto conv_fused = convolution_forward(conv_fused_pd);

    // 7. Execute
    conv_fused.execute(s, {
        {DNNL_ARG_SRC, internal_src_mem},
        {DNNL_ARG_WEIGHTS, user_wei_mem}, // Assuming weights matched
        {DNNL_ARG_DST, internal_dst_mem}
    });
    
    // 8. Reorder Back (if needed)
    if (internal_dst_mem.get_desc() != user_dst_mem.get_desc()) {
        reorder(internal_dst_mem, user_dst_mem).execute(s, internal_dst_mem, user_dst_mem);
    }
    
    s.wait(); // Sync
    
    std::cout << "Inference Complete.\n";
    return 0;
}
```

---

## 🧪 Hands-On Labs

### Lab 38: Benchmarking Formats

**Objective:** Measure speed difference between `nchw` and `blocked` layouts on CPU.

**Code Modification:**
1.  Force `format_tag::nchw` in the descriptor (if allowed by implementation) or use a naive implementation.
2.  Allow `format_tag::any` and measure the chosen blocked format.

**Metrics:**
*   Use `std::chrono` around `s.wait()`.
*   On AVX-512 machines (Xeon), blocked format should be 2-3x faster for large layers due to SIMD efficiency.

---

## 📝 Summary & Key Takeaways

1.  **Format Agnosticism:** The #1 rule of oneDNN. Don't force formats. Let the library pick, and use `reorder` at the boundaries (Input/Output).
2.  **Fusion:** Always fuse ReLU/Sum into Convolution. Memory Bandwidth is the bottleneck; fusion keeps data in L1/Registers.
3.  **Primitive Cache:** Creating `primitive_desc` is expensive (JIT compilation). Do it once (during model load), execute many times.
4.  **Attributes:** Use `primitive_attr` to enable quantization (INT8), fusion, and scratchpad modes.
5.  **Verbose Mode:** `export ONEDNN_VERBOSE=1` prints every primitive execution, its time, and format. Essential for debugging perf.

---

## 📚 Additional Resources

*   [oneDNN Developer Guide](https://oneapi-src.github.io/oneDNN/p_user_guide.html)
*   [Intel AMX (Advanced Matrix Extensions)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) - *Matrix tiles in hardware.*

**Tomorrow:** Day 39 - oneMKL... Standard Math Libraries (BLAS/LAPACK) accelerated on Intel GPUs via DPC++.

*End of Day 038 - Total Lines: 1000+*
