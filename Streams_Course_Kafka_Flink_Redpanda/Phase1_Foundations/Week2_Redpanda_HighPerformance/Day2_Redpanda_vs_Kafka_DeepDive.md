# Day 2: Redpanda vs Kafka - Deep Dive

## Deep Dive & Internals

### Autotuning (`rpk redpanda tune`)
Redpanda includes a tuner that optimizes the OS for streaming.
-   **NIC Interrupts**: Distributes IRQs across cores to prevent one core from handling all network traffic.
-   **Disk I/O Scheduler**: Sets the scheduler to `noop` or `deadline` (preferred for SSDs).
-   **AIO**: Enables Asynchronous I/O.

### Memory Management
-   **Kafka**: Heap (Garbage Collected) + Off-heap (Page Cache). Tuning heap size is an art.
-   **Redpanda**: Allocates *all* available RAM at startup. It manages its own cache. No GC pauses.

### Advanced Reasoning
**Why is "Single Binary" a feature?**
-   **Deployment**: `apt-get install redpanda`. No JVM, no Zookeeper, no separate Jars.
-   **Upgrades**: Rolling upgrades are simpler.
-   **Edge Computing**: The small footprint allows Redpanda to run on IoT devices or edge servers where a full JVM+ZK stack is too heavy.

### Performance Implications
-   **Cold Reads**: Redpanda's Shadow Indexing (Tiered Storage) is often faster than Kafka's because it can prefetch data from S3 more aggressively using its async engine.
