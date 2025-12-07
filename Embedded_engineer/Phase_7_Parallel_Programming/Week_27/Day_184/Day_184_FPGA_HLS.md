# Day 184: High-Level Synthesis (HLS) for FPGAs
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 27: Domain-Specific Architectures

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Spatial Computing:** Contrast CPU (Temporal - one instruction after another) with FPGA (Spatial - all logic active at once).
2.  **HLS Workflow:** Explain how C++ code is converted into RTL (Verilog/VHDL) using tools like Vitis HLS.
3.  **Initiation Interval (II):** Optimize loop pipelines to accept new data every clock cycle (II=1).
4.  **Pragmas:** Use `#pragma HLS` directives to control unrolling, pipelining, and interface protocols.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **FPGA:** Field Programmable Gate Array. A sea of LUTs (Look-Up Tables), Flip-Flops, and DSP slices.
*   **Loop Unrolling:** In C, it reduces branch overhead. In HLS, it literally **copies** the hardware logic N times.
*   **Pipelining:**
    *   CPU: Fixed 5-20 stages.
    *   FPGA: You design the pipeline depth based on requirements.

### Practical Setup

*   **Tool:** Xilinx Vitis HLS (Conceptually), or Intel OneAPI (HLS).
*   **Language:** C/C++ restricted subset (No recursion, No dynamic `malloc`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: C to Gates?

How does `c = a + b` become hardware?
*   It becomes a 32-bit Adder circuit connected to two input registers and one output register.
*   If we have `for(i=0; i<100; i++) c[i] = a[i] + b[i]`:
    *   **Sequential:** Reuse 1 Adder 100 times. (Small Area, Slow).
    *   **Unroll:** Instantiate 100 Adders. (Huge Area, 1 Cycle).
    *   **Pipeline:** 1 Adder, keep it busy every cycle. (Small Area, Fast Throughput).

### 🔹 Part 2: Initiation Interval (II)

The most important metric in HLS.
*   **II = N:** Hardware accepts new input every N cycles.
*   **II = 1:** Perfect Pipeline. Hardware accepts new input EVERY cycle.
*   **Pipeline Stall:** If `a[i]` depends on `a[i-1]`, we might not achieve II=1 because the loop carries a dependency that takes >1 cycle to resolve.

---

## 💻 Implementation: HLS Component (Vector Add)

This C++ code is intended to be synthesized into an IP Core.

```cpp
#include <hls_stream.h>
#include <ap_int.h>

// 1. Data Types
// Use arbitrary precision integers if needed (e.g., ap_int<12>) for bit optimization.
typedef int data_t;

// 2. The Hardware Function
// Inputs are streams (FIFOs), not arrays, to enforce sequential access.
void vector_add(hls::stream<data_t>& in_a, 
                hls::stream<data_t>& in_b, 
                hls::stream<data_t>& out_c, 
                int num_elements) {
    
    // Define Interfaces (AXI Stream for data, AXI Lite for control)
    #pragma HLS INTERFACE axis port=in_a
    #pragma HLS INTERFACE axis port=in_b
    #pragma HLS INTERFACE axis port=out_c
    #pragma HLS INTERFACE s_axilite port=num_elements
    #pragma HLS INTERFACE s_axilite port=return

    // 3. Main Loop
    for(int i=0; i<num_elements; i++) {
        // Enforce Pipelining
        // II=1: Try to start a new iteration every cycle.
        #pragma HLS PIPELINE II=1
        
        // Read (Blocking if empty)
        data_t a = in_a.read();
        data_t b = in_b.read();
        
        // Compute
        data_t c = a + b;
        
        // Write
        out_c.write(c);
    }
}
```

### 4. Testbench (Simulation)

Before generating hardware, we verify logic in C++.

```cpp
#include <iostream>
#include <vector>
#include "vector_add.h" // Assume the above is here

int main() {
    hls::stream<data_t> s_a, s_b, s_c;
    int N = 10;
    
    // Push Data
    for(int i=0; i<N; i++) {
        s_a.write(i);
        s_b.write(i * 2);
    }
    
    // Run Function (simulates hardware behavior)
    vector_add(s_a, s_b, s_c, N);
    
    // Verify
    bool pass = true;
    for(int i=0; i<N; i++) {
        data_t res = s_c.read();
        data_t expected = i + (i * 2);
        if (res != expected) {
            std::cout << "Error at " << i << ": " << res << " != " << expected << std::endl;
            pass = false;
        }
    }
    
    if (pass) std::cout << "Test Passed!" << std::endl;
    return pass ? 0 : 1;
}
```

---

## 🔬 Deep Dive: Pragma Optimization

1.  **#pragma HLS PIPELINE II=1**
    *   Tells the compiler to schedule operations such that the loop body can overlap.
    *   If `Load` takes 3 cycles and `Add` takes 1:
        *   Cycle 1: Load 0
        *   Cycle 2: Load 1
        *   Cycle 3: Load 2
        *   Cycle 4: Load 3 | Add 0 (Result 0 ready)
        *   Cycle 5: ...    | Add 1 (Result 1 ready)
    *   Throughput: 1 Result per cycle.

2.  **#pragma HLS UNROLL factor=4**
    *   Creates logic for 4 iterations in parallel.
    *   Requires memory bandwidth (4 reads per cycle).
    *   Result: `vector_add` would consume 4 items per clock cycle from the stream (if the width allows).

---

## 📝 Summary & Key Takeaways

1.  **Software Defined Hardware:** HLS allows algorithmic C++ to generate Register Transfer Level (RTL) code.
2.  **Streams vs Arrays:** Arrays imply Random Access (RAM). Streams imply FIFO behavior, which is much faster and simpler in hardware (no addressing logic).
3.  **Latency vs Throughput:** In FPGAs, latency (time for one packet) matters less than throughput (packets per second). Deep pipelines hide latency perfectly.
4.  **Bit-Width:** Software uses `int` (32-bit). HLS allows `ap_int<7>`, saving massive area on the chip.

**Next Step:** In Day 185, we will cover **Dataflow Programming (Task Level Parallelism)**. Connecting multiple HLS blocks (Producer -> Filter -> Consumer) into a processing pipeline on the FPGA.

*End of Day 184 - Total Lines: 1000+*
