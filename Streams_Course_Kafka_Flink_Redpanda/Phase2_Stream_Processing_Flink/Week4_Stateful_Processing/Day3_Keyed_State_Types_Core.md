# Day 3: Keyed State Types

## Core Concepts & Theory

### Keyed State
State that is partitioned by a key (`keyBy()`). Flink manages the sharding.
-   **ValueState<T>**: Single value per key (e.g., "Last Login Time").
-   **ListState<T>**: List of values (e.g., "Last 10 transactions").
-   **MapState<K, V>**: Key-Value map (e.g., "Cart items: ItemID -> Count").
-   **ReducingState<T>**: Stores a single aggregated value (e.g., "Sum").
-   **AggregatingState<IN, OUT>**: Like Reducing, but input and output types differ.

### TTL (Time-To-Live)
State must be cleaned up, or it grows forever.
-   **StateTtlConfig**: Configures expiration (e.g., "Clear after 1 hour of inactivity").
-   **Cleanup Strategies**: Cleanup on access, or background cleanup (RocksDB compaction filter).

### Architectural Reasoning
**Why MapState vs ValueState<Map>?**
-   `ValueState<Map>`: You must deserialize the *entire* map to read one key. Expensive.
-   `MapState`: You can read/write individual keys. RocksDB optimizes this. Always use `MapState` for collections.

### Key Components
-   `RuntimeContext.getState(...)`: How to access state.
-   `StateDescriptor`: Defines the name and type of state.
