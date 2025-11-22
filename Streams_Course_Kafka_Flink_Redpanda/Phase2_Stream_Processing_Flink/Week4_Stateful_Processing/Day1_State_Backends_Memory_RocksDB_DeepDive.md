# Day 1: State Backends - Deep Dive

## Deep Dive & Internals

### RocksDB Tuning
RocksDB is a Log-Structured Merge-Tree (LSM) KV store.
-   **Block Cache**: In-memory cache for uncompressed blocks.
-   **Write Buffer (MemTable)**: In-memory buffer for writes.
-   **Compaction**: Merging SSTables. High CPU usage.
-   **Serialization**: Flink must serialize objects to bytes to store in RocksDB. This is the main CPU cost.

### Incremental Checkpoints
-   **Full Checkpoint**: Uploads the entire state to S3.
-   **Incremental**: Uploads only the *new* SSTables created since the last checkpoint.
    -   RocksDB supports this natively.
    -   Drastically reduces checkpoint duration and network usage.

### Advanced Reasoning
**The "Managed Memory" Fraction**
Flink reserves a portion of memory (default 40%) for "Managed Memory". RocksDB uses this for its Block Cache and Write Buffers. If you see OOMs, you might need to tune `state.backend.rocksdb.memory.managed`.

### Performance Implications
-   **Disk I/O**: RocksDB is disk-bound. Use NVMe SSDs.
-   **Serialization**: Use POJOs or Avro to keep serialization cheap.
