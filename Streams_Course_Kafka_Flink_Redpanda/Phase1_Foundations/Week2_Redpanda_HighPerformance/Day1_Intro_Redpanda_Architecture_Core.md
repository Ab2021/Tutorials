# Day 1: Redpanda Architecture

## Core Concepts & Theory

### What is Redpanda?
Redpanda is a modern, Kafka-compatible streaming platform written in C++. It is designed to be a drop-in replacement for Kafka but with significantly higher performance and operational simplicity.

### Thread-Per-Core Architecture
Unlike Kafka (JVM-based), which relies on the OS kernel for thread scheduling and page cache, Redpanda uses a **Thread-Per-Core** (TPC) architecture (Seastar framework).
-   **Shared-Nothing**: Each core has its own memory and task queue. There is no locking or contention between cores.
-   **Direct I/O**: Redpanda bypasses the OS page cache and manages disk I/O directly (DMA).

### Architectural Reasoning
**Why C++ and TPC?**
Hardware has changed. Modern NVMe SSDs and 100GbE networks are incredibly fast. The JVM and the Linux kernel context switching overhead become bottlenecks.
-   **Zero-Copy**: Redpanda moves data from disk to network with minimal CPU involvement.
-   **Tail Latency**: By pinning threads to cores and avoiding GC pauses (no JVM), Redpanda offers predictable, low tail latency (p99).

### Key Components
-   **Seastar**: The C++ framework for high-performance async I/O.
-   **Single Binary**: No Zookeeper. Redpanda includes a built-in Raft consensus engine.
