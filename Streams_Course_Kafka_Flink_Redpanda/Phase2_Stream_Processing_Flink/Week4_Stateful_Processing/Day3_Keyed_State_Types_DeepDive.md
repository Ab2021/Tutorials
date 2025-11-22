# Day 3: Keyed State - Deep Dive

## Deep Dive & Internals

### State Serialization
Flink needs a `TypeSerializer` for the state.
-   If you change the class definition of the state object, the serializer might become incompatible.
-   **Schema Evolution**: Flink supports evolving POJO and Avro schemas (adding fields) for state.

### Key Groups
State is not sharded by key hash directly (too many shards).
-   **Key Groups**: The key space is divided into `maxParallelism` (default 128) groups.
-   **Rescaling**: When parallelism changes, whole Key Groups are moved between TaskManagers. This is faster than re-hashing every key.

### Advanced Reasoning
**ReducingState Efficiency**
`ReducingState` merges values *before* serialization (if possible) or during compaction. It is much more efficient than `ListState` if you only care about the aggregate.

### Performance Implications
-   **State Leaks**: Forgetting to set TTL or manually `clear()` state is the #1 cause of Flink job failure over time.
-   **RocksDB Iterators**: Iterating over `MapState` in RocksDB is slow. Avoid large iterations in the hot path.
