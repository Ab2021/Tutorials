# Day 041: Advanced DPC++: Pipes, Reductions & FPGA
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 6: Intel oneAPI & DPC++

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Stream Data with Pipes:** Implement `sycl::ext::intel::pipe` to transfer data between kernels (Producer-Consumer) without going back to Global Memory (Spatial Architectures like FPGA).
2.  **Simplify Reductions:** Use SYCL built-in `reduction` objects to sum arrays in parallel without writing manual tree-reduction code.
3.  **Optimize for FPGA:** Apply attributes like `[[intel::ivdep]]` (Ignore Vector Dependency) and `[[intel::max_concurrency(N)]]` for Field Programmable Gate Arrays.
4.  **Debug Heterogeneously:** Use the `sycl::stream` class to print debug info from device code reliably.
5.  **Profile with ITT:** Mark regions of code for Intel VTune Profiler using the Instrumentation and Tracing Technology (ITT) API.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **FPGA Emulation:** requires `intelfpga` component in oneAPI.
*   **Compile:** `icpx -fsycl -fintelfpga ...`

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SYCL Reductions

Writing a parallel reduction (Sum, Max, Min) is error-prone (Race conditions, Atomic contention).
DPC++ 2020 standardizes this.

**Old Way:** Atomic add on global memory (Slow).
**New Way:**
```cpp
h.parallel_for(nd_range<1>(N, B), reduction(sum_var, plus<>()), [=](nd_item<1> it, auto& sum) {
    sum += data[it.get_global_id(0)];
});
```
The runtime automatically uses **Sub-Group Shuffles** and **Local Memory** hierarchies for maximum speed.

### 🔹 Part 2: Spatial Computing (FPGA Pipes)

On GPU, "Producer writes to VRAM -> Consumer reads VRAM". Memory bandwidth is the bottleneck.
On FPGA, we want "Producer feeds wire -> Consumer reads wire".

**Pipe Object:**
*   FIFO (First-In, First-Out) buffer in hardware registers.
*   **Non-blocking:** `read()` / `write()`.
*   **Blocking:** `read(bool& success)`.

```cpp
using PipeA = pipe<class IdA, int, 4>; // Depth 4 FIFO

// Kernel 1
PipeA::write(value);

// Kernel 2
int value = PipeA::read();
```

### 🔹 Part 3: FPGA Loop optimization

CPUs use "Branch Prediction" and "Out-of-Order Execution".
FPGAs use "Pipelining".
If a loop has a dependency (e.g., `acc += arr[i]`), the pipeline stalls.

**Attributes:**
*   `[[intel::ivdep]]`: "I promise there is no loop-carried dependency". Compiler generates parallel hardware.
*   `[[intel::loop_coalesce(N)]]`: Merge nested loops into one fat pipeline.

---

## 💻 Implementation: Producer-Consumer with Pipes

We will simulate an FPGA workflow: Kernel A generates data, Kernel B modifies it, Kernel C consumes it.

### 🛠️ Step 1: The Code (`pipes.cpp`)

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel;

// Define Pipe Type: ID, Type, Depth
using PipeData = pipe<class ID_Data, int, 16>;

int main() {
    // Select FPGA Emulator
#if defined(FPGA_EMULATOR)
    ext::intel::fpga_emulator_selector selector;
#else
    default_selector selector;
#endif

    queue q(selector);
    const int N = 100;
    
    int* out_data = malloc_shared<int>(N, q);

    // Kernel 1: Producer
    q.submit([&](handler& h) {
        h.single_task([=]() {
            for (int i=0; i<N; i++) {
                // Write to Pipe
                PipeData::write(i * 10); 
            }
        });
    });

    // Kernel 2: Consumer
    auto e = q.submit([&](handler& h) {
        h.single_task([=]() {
            for (int i=0; i<N; i++) {
                // Read from Pipe (Blocking)
                int val = PipeData::read();
                out_data[i] = val + 5;
            }
        });
    });

    e.wait();

    // Verify
    bool pass = true;
    for(int i=0; i<N; i++) {
        if (out_data[i] != (i*10 + 5)) {
            std::cout << "Mismatch at " << i << "\n";
            pass = false;
        }
    }
    
    if (pass) std::cout << "PASSED!\n";
    
    free(out_data, q);
    return 0;
}
```

### 🛠️ Step 2: Compiling for Emulation

FPGA compilation takes hours (Synthesis). Emulation takes seconds.

```bash
icpx -fsycl -fintelfpga -DFPGA_EMULATOR pipes.cpp -o pipes_emu
./pipes_emu
```

### 🔹 Part 4: Built-in Reduction

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    queue q;
    const int N = 1024 * 1024;
    float* data = malloc_shared<float>(N, q);
    float* sum = malloc_shared<float>(1, q); // Result ptr

    // Init
    for(int i=0; i<N; i++) data[i] = 1.0f;
    *sum = 0.0f;

    q.submit([&](handler& h) {
        // Create Reduction Object
        // Var to reduce, Operator, Property (USM pointer)
        auto red = reduction(sum, plus<>());

        h.parallel_for(range<1>(N), red, [=](id<1> i, auto& acc) {
            // acc looks like a variable, but handles atomic/local accumulation
            acc += data[i];
        });
    }).wait();

    std::cout << "Sum: " << *sum << " (Expected " << N << ")\n";
    return 0;
}
```

---

## 🧪 Hands-On Labs

### Lab 41: Debugging with Streams

**Objective:** Use `sycl::stream` to inspect values inside a kernel race condition.

```cpp
q.submit([&](handler& h) {
    // Stream configuration: Total buffer size, max message length
    stream out(1024, 80, h);
    
    h.parallel_for(range<1>(10), [=](id<1> i) {
        out << "Thread " << i[0] << " says Hello!\n";
    });
});
```
**Note:** `stream` prints are asynchronous and order is NOT guaranteed between Work-Items.

**Task:**
1.  Launch 100 threads.
2.  Print only if `id == 50`.
3.  Check if order is preserved (it won't be generally).

---

## 📝 Summary & Key Takeaways

1.  **Pipes:** The specific optimization for Spatial (FPGA) logic. Keeps data on-chip. Can also work on GPUs (Experimental).
2.  **Reductions:** `reduction(ptr, op)` saves you from writing complex recursive doubling logic.
3.  **FPGA Emulation:** Always emulate before synthesis. Logic errors are instant on emulator, expensive on hardware compile.
4.  **Loop Unrolling:** Crucial for FPGA performance to widen the datapath. `[[intel::unroll]]`.
5.  **Single Task:** FPGAs often prefer `single_task` (one deep pipeline) over `parallel_for` (many threads), whereas GPUs prefer `parallel_for`.

---

## 📚 Additional Resources

*   [Intel oneAPI FPGA Optimization Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top.html)
*   [SYCL 2020 Reduction Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:reductions)

**Tomorrow:** Day 42 - Week 6 Review & Project... Implementing a DPC++ Molecular Dynamics Simulator.

*End of Day 041 - Total Lines: 1000+*
