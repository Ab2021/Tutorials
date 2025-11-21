# Day 10 Interview Prep: Phase 1 Mock

## Q1: Design a unique ID generator.
**Answer:**
*   **UUID:** Simple, huge (128-bit), unordered. Bad for DB indexing.
*   **Auto-Increment:** Simple, but SPOF (Single DB). Hard to shard.
*   **Snowflake (Twitter):** 64-bit integer. Timestamp + MachineID + Sequence. Ordered (k-sortable). Best for distributed systems.

## Q2: How to handle 1 Million concurrent connections?
**Answer:**
*   **C10k Problem:** Solved by non-blocking I/O (Epoll/Kqueue).
*   **Architecture:** Use Nginx/HAProxy as entry point. Use Async servers (Node.js, Go, Netty).
*   **Kernel Tuning:** Increase file descriptors (`ulimit -n`), ephemeral ports.

## Q3: What is Backpressure?
**Answer:**
*   When a consumer cannot keep up with the producer.
*   **Mechanism:** Consumer tells Producer to slow down (TCP Window, Reactive Streams).
*   **If ignored:** OutOfMemoryError (Queue fills up).

## Q4: Explain "Data Locality".
**Answer:**
*   Moving computation to the data, rather than data to computation.
*   Crucial in Big Data (Hadoop/Spark). Sending code (KB) is cheaper than sending data (TB).
