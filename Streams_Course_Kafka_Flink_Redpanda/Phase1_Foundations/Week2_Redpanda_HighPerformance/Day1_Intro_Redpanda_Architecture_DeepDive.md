# Day 1: Redpanda Architecture - Deep Dive

## Deep Dive & Internals

### Seastar Framework
Redpanda is built on **Seastar**, a C++ framework for high-performance server applications.
-   **Futures/Promises**: Uses a future-based programming model for non-blocking I/O.
-   **Shared-Nothing**: Memory is pre-allocated per core. No locks are needed for memory access, avoiding contention.
-   **User-Space Networking**: Can use DPDK to bypass the kernel network stack (though often runs on standard sockets for compatibility).

### The Log Structure (Segments)
Redpanda uses a similar segment structure to Kafka but optimized.
-   **Open-Addressing Hash Table**: For the index (vs Kafka's sparse index).
-   **DMA (Direct Memory Access)**: Writes go from memory to disk controller without CPU copying.

### Advanced Reasoning
**Why Thread-Per-Core?**
In traditional multi-threaded apps (like JVM Kafka), threads fight for CPU time and locks. Context switches are expensive (microseconds). By pinning one thread to one core, Redpanda treats the CPU as a distributed system of independent nodes, communicating via message passing. This eliminates lock contention and maximizes instruction-per-cycle (IPC).

### Performance Implications
-   **Tail Latency**: Because there is no "Stop-the-World" GC, p99 latency is stable even at high throughput.
-   **Hardware Utilization**: Redpanda can saturate NVMe SSDs and 100GbE NICs with fewer CPUs than Kafka.
