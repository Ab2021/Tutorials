# Day 2: Redpanda vs. Kafka

## Core Concepts & Theory

### Performance Comparison
-   **Throughput**: Redpanda can often achieve 10x the throughput of Kafka on the same hardware due to its efficient I/O and lack of JVM overhead.
-   **Latency**: Redpanda maintains single-digit millisecond latency even at high loads.

### Operational Simplicity
-   **Kafka**: Requires Zookeeper (historically), JVM tuning (Heap size, GC algorithms), and OS tuning (Page cache, file descriptors).
-   **Redpanda**: A single binary. Autotunes itself to the hardware (`rpk redpanda tune`). No Zookeeper.

### WASM Transforms
Redpanda allows you to run **WebAssembly (WASM)** code directly inside the broker.
-   **Data Sovereignty**: Filter or mask PII data *before* it leaves the broker.
-   **Push-down Processing**: Move computation to the data, rather than moving data to the computation.

### Architectural Reasoning
**The Cost of Complexity**
Kafka's complexity leads to "Kafka teams" just to manage the cluster. Redpanda aims to be "developer-first" by removing the operational burden, allowing teams to focus on the application logic.

### Key Components
-   **rpk**: The Redpanda CLI tool (all-in-one).
-   **WASM Engine**: Embedded V8 engine for transforms.
