# Day 2: DataStream API Basics

## Core Concepts & Theory

### Sources & Sinks
-   **Source**: Where data comes from (Kafka, File, Socket). `env.add_source(...)`.
-   **Sink**: Where data goes (Kafka, File, DB). `stream.add_sink(...)`.

### Transformations
-   **Map**: 1-to-1 transformation.
-   **FlatMap**: 1-to-N transformation (or 0). Good for splitting strings or filtering.
-   **Filter**: Keep only elements satisfying a condition.
-   **KeyBy**: Logically partitions the stream. All records with same key go to same parallel instance.

### Physical Partitioning
-   **Rescale**: Round-robin to a subset of downstream tasks.
-   **Rebalance**: Round-robin to ALL downstream tasks (handles skew).
-   **Broadcast**: Send element to ALL downstream tasks.

### Architectural Reasoning
**Why KeyBy is expensive?**
`keyBy` causes a **Network Shuffle**. Data must be serialized and sent over the network to the correct TaskManager. This is the most expensive operation in a distributed system. Minimize shuffles.

### Key Components
-   **StreamExecutionEnvironment**: The context for creating the job.
-   **DataStream**: The immutable collection of data.
