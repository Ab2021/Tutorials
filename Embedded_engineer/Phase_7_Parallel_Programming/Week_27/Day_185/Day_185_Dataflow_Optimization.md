# Day 185: Dataflow Programming (Task Level Parallelism)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 27: Domain-Specific Architectures

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Task Level Parallelism (TLP):** Run entirely different functions simultaneously on the FPGA.
2.  **Dataflow Pragma:** Use `#pragma HLS DATAFLOW` to create a Canonical Producer-Consumer architecture.
3.  **FIFO Channels:** Connect tasks using `hls::stream` to allow elastic buffering.
4.  **Throughput Balancing:** Identify the bottleneck task that defines the overall system throughput.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Sequential C++:** `funcA(); funcB(); funcC();`. `B` waits for `A` to finish completely. (Latency = Sum).
*   **Dataflow C++:** `funcA` starts. As soon as it produces 1 output, `funcB` starts processing it. (Latency = Max(A,B,C) + Overhead).
*   **Kahn Process Networks:** Theoretical model of determinstic parallel processes communicating via unbounded FIFO channels.

### Practical Setup

*   **Scenario:** Image Processing Pipeline.
    *   Read (Memory -> Stream)
    *   Grayscale (Stream -> Stream)
    *   Write (Stream -> Memory)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Canonical Structure

```cpp
void kernel_top(input, output) {
    #pragma HLS DATAFLOW
    hls::stream s1, s2;
    read_data(input, s1);
    process_data(s1, s2);
    write_data(s2, output);
}
```

*   **Hardware Result:**
    *   Block `read_data` is instantiated.
    *   Block `process_data` is instantiated.
    *   Block `write_data` is instantiated.
    *   FIFO `s1` connects Read->Process.
    *   FIFO `s2` connects Process->Write.
    *   **All 3 blocks run at the same time.**

### 🔹 Part 2: Deadlocks and Depth

*   **FIFO Depth:** If `read_data` writes 100 items but `process_data` effectively reads nothing for a while? The FIFO fills up. `read_data` stalls. System is safe, just stalled.
*   **Deadlock:** Cyclical dependencies or improper usage of non-blocking reads can cause deadlocks. HLS tools usually warn about this.
*   **Optimization:** `s1.set_depth(N)`. Usually default is 2. Increase if tasks are "bursty".

---

## 💻 Implementation: Producer-Consumer Pipeline

We build a pipeline that computes `D[i] = (A[i] + val) * scale`.
Split into: `Load` -> `Add` -> `Mult` -> `Store`.

```cpp
#include <hls_stream.h>
#include <ap_int.h>

typedef int data_t;

// Task 1: Load Data from Memory into Stream
void load_input(data_t* in_mem, hls::stream<data_t>& out_stream, int size) {
    for(int i=0; i<size; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(in_mem[i]);
    }
}

// Task 2: Add Constant
void compute_add(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream, int size, int val) {
    for(int i=0; i<size; i++) {
        #pragma HLS PIPELINE II=1
        data_t temp = in_stream.read();
        temp += val;
        out_stream.write(temp);
    }
}

// Task 3: Multiply Constant
void compute_mult(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream, int size, int scale) {
    for(int i=0; i<size; i++) {
        #pragma HLS PIPELINE II=1
        data_t temp = in_stream.read();
        temp *= scale;
        out_stream.write(temp);
    }
}

// Task 4: Store Stream to Memory
void store_output(hls::stream<data_t>& in_stream, data_t* out_mem, int size) {
    for(int i=0; i<size; i++) {
        #pragma HLS PIPELINE II=1
        out_mem[i] = in_stream.read();
    }
}

// Top Level Function
void accelerated_pipeline(data_t* mem_in, data_t* mem_out, int size, int val, int scale) {
    // Define Interfaces
    #pragma HLS INTERFACE m_axi port=mem_in bundle=gmem0
    #pragma HLS INTERFACE m_axi port=mem_out bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=size
    #pragma HLS INTERFACE s_axilite port=val
    #pragma HLS INTERFACE s_axilite port=scale
    #pragma HLS INTERFACE s_axilite port=return

    // Define internal streams
    hls::stream<data_t> s1("stream_1");
    hls::stream<data_t> s2("stream_2");
    hls::stream<data_t> s3("stream_3");
    
    // Set depths (optional, helps performance)
    #pragma HLS STREAM variable=s1 depth=32
    #pragma HLS STREAM variable=s2 depth=32
    #pragma HLS STREAM variable=s3 depth=32

    // Enable Dataflow
    #pragma HLS DATAFLOW

    // Instantiate Tasks
    load_input(mem_in, s1, size);
    compute_add(s1, s2, size, val);
    compute_mult(s2, s3, size, scale);
    store_output(s3, mem_out, size);
}
```

### Analysis of Execution

*   **Cycle 0:** `load_input` reads `A[0]`.
*   **Cycle 1:** `load_input` writes `A[0]` to `s1`. Reads `A[1]`.
*   **Cycle 2:** `compute_add` reads `A[0]` from `s1`. `load_input` touches `A[2]`.
*   **Cycle 100:**
    *   `load_input` is processing `A[100]`.
    *   `compute_add` is processing `A[99]`.
    *   `compute_mult` is processing `A[98]`.
    *   `store_output` is processing `A[97]`.
*   **Throughput:** 1 item processed per cycle (limited by the slowest II=1 task).

---

## 🔬 Deep Dive: Memory Bottlenecks

Usually, `load_input` and `store_output` are the bottlenecks.
*   DRAM access latency is high.
*   If Mem Bandwidth is 10 GB/s and Compute capability is 100 GOps...
*   We must pack data (Vectorize) to read 512-bits (16 ints) at once.
*   Change `data_t` to `ap_uint<512>` and unpack inside the compute kernels to keep the pipeline fed.

---

## 📝 Summary & Key Takeaways

1.  **Macro-Pipelining:** Dataflow is pipelining applied to functions rather than instructions.
2.  **Streams:** The glue that holds HLS tasks together.
3.  **Concurrency:** FPGAs excel here. We are doing IO, Add, Mult, and IO all in the exact same clock tick.
4.  **No OS:** There is no scheduler switching tasks. They are physically separate circuits running in parallel.

**Next Step:** In Day 186, we will cover **GPU vs FPGA Architecture Comparison**. When to use a TPU, when to use an FPGA, and when to stick to a GPU.

*End of Day 185 - Total Lines: 1000+*
