# Day 3: Topics, Partitions, and Segments

## Core Concepts & Theory

### Topics
A **Topic** is a logical category or feed name to which records are published. It is analogous to a table in a database or a folder in a filesystem.

### Partitions
A Topic is broken down into **Partitions**.
-   **Scalability**: Partitions allow a topic to be spread across multiple brokers. This allows parallel processing.
-   **Ordering**: Ordering is guaranteed *only within a partition*, not across the entire topic.
-   **Offset**: Each message in a partition has a unique sequential ID called an offset.

### Segments
Partitions are physically stored as a set of **Segment** files on the disk.
-   **Active Segment**: The file currently being written to.
-   **Closed Segments**: Older files that are read-only.
-   **Indexes**: Kafka maintains an index file to map offsets to physical file positions for fast lookups.

### Architectural Reasoning
**Why Partitioning?**
If a topic were a single log file, it could only grow as large as one machine's disk and handle the I/O of one machine. Partitioning allows the log to be distributed (sharded) across the entire cluster, enabling horizontal scaling.

### Key Components
-   **Topic**: Logical stream.
-   **Partition**: Physical shard of the stream.
-   **Segment**: Physical file on disk (`.log`, `.index`, `.timeindex`).
