# Day 1: State Backends

## Core Concepts & Theory

### What is State?
State is any information that the application remembers across events.
-   **Keyed State**: Scoped to a key (e.g., "current balance for user X").
-   **Operator State**: Scoped to a parallel task (e.g., "Kafka offsets for this partition").

### State Backends
The **State Backend** determines *where* state is stored and *how* it is checkpointed.
1.  **HashMapStateBackend** (Memory/FS):
    -   Stores state in the Java Heap (On-Heap).
    -   Fastest (no serialization/disk I/O).
    -   Limited by GC and Heap size.
    -   Checkpoints stored in FileSystem (S3/HDFS).
2.  **EmbeddedRocksDBStateBackend**:
    -   Stores state in local RocksDB instance (Off-Heap/Disk).
    -   Slower (serialization overhead).
    -   Scales to TBs of state per node.
    -   Supports incremental checkpoints.

### Architectural Reasoning
**When to use RocksDB?**
Use RocksDB when your state is larger than memory (e.g., 7-day window, large joins). Use HashMap for low-latency, small-state jobs.

### Key Components
-   **Checkpoint Storage**: Where the snapshot goes (S3, HDFS).
-   **Async Snapshot**: State backends snapshot asynchronously to avoid blocking processing.
