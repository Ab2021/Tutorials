import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink\Week4_Stateful_Processing"

content_map = {
    # --- Day 1: State Backends ---
    "Day1_State_Backends_Memory_RocksDB_Core.md": """# Day 1: State Backends

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
""",
    "Day1_State_Backends_Memory_RocksDB_DeepDive.md": """# Day 1: State Backends - Deep Dive

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
""",
    "Day1_State_Backends_Memory_RocksDB_Interview.md": """# Day 1: State Backends - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between HashMapStateBackend and RocksDBStateBackend?**
    -   *A*: HashMap stores objects on Heap (fast, GC pressure, size limit). RocksDB stores serialized bytes on Disk/Off-Heap (slower, scalable, no GC).

2.  **Q: How does Flink handle state larger than memory?**
    -   *A*: By using the RocksDB state backend, which spills to disk.

3.  **Q: What is an Incremental Checkpoint?**
    -   *A*: A checkpoint that only persists the changes (diff) since the last checkpoint, rather than the full state.

### Production Challenges
-   **Challenge**: **Long GC Pauses**.
    -   *Scenario*: Using HashMapStateBackend with large windows.
    -   *Fix*: Switch to RocksDB.

-   **Challenge**: **RocksDB High CPU**.
    -   *Cause*: Heavy serialization or aggressive compaction.
    -   *Fix*: Optimize data types (avoid generic Objects), tune compaction threads.

### Troubleshooting Scenarios
**Scenario**: Checkpoint fails with "Size exceeded".
-   *Cause*: State is too large for the target storage or timeout.
-   *Fix*: Enable incremental checkpoints, increase timeout, or check for state leaks (keys never deleted).
""",

    # --- Day 2: Checkpointing ---
    "Day2_Checkpointing_Savepoints_Core.md": """# Day 2: Checkpointing & Savepoints

## Core Concepts & Theory

### Checkpointing (Fault Tolerance)
Automatic, periodic snapshots of the application state.
-   **Purpose**: Recovery from failure.
-   **Mechanism**: Chandy-Lamport algorithm (Barrier alignment).
-   **Consistency**: Guarantees **Exactly-Once** state consistency.

### Savepoints (Operations)
Manual, user-triggered snapshots.
-   **Purpose**: Updates, A/B testing, Rescaling, Migration.
-   **Format**: Canonical format, portable across versions.
-   **Self-Contained**: Includes all necessary metadata.

### Architectural Reasoning
**Barrier Alignment**
Barriers flow with the stream. When an operator receives barriers from all inputs, it snapshots its state.
-   **Exactly-Once**: Wait for all barriers (alignment). No data processed from fast streams while waiting.
-   **At-Least-Once**: Don't wait. Process data as it comes. (Faster, but duplicates possible on recovery).

### Key Components
-   `checkpoint.interval`: How often? (e.g., 1 min).
-   `min.pause.between.checkpoints`: Prevent "Checkpoint Storm".
-   `state.checkpoints.dir`: S3 path.
""",
    "Day2_Checkpointing_Savepoints_DeepDive.md": """# Day 2: Checkpointing - Deep Dive

## Deep Dive & Internals

### Unaligned Checkpoints
In high-backpressure scenarios, barrier alignment takes too long (barriers are stuck in queues).
-   **Unaligned**: Snapshot the *inflight data* (buffers) along with the state.
-   **Pros**: Checkpoints succeed even under heavy load.
-   **Cons**: Larger checkpoint size (storing buffers).

### State Recovery Process
1.  **Restart**: Job fails, all tasks restart.
2.  **Load Metadata**: JobManager reads the latest valid checkpoint metadata.
3.  **Restore State**: TaskManagers download their assigned state chunks from S3.
4.  **Resume**: Processing starts from the checkpoint barrier.

### Advanced Reasoning
**Savepoint vs Checkpoint**
-   **Checkpoint**: "Implementation detail". Optimized for speed. Might depend on specific file paths. Not guaranteed to be portable.
-   **Savepoint**: "Logical backup". Optimized for portability. Can be used to upgrade Flink version or change parallelism.

### Performance Implications
-   **Copy-On-Write**: HashMapBackend uses COW to snapshot without blocking.
-   **Async Upload**: The heavy lifting (uploading to S3) happens in the background.
""",
    "Day2_Checkpointing_Savepoints_Interview.md": """# Day 2: Checkpointing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: Explain the Chandy-Lamport algorithm in Flink.**
    -   *A*: It uses "Barriers" injected into the stream. Operators snapshot state when they receive barriers. It allows consistent snapshots without stopping the world.

2.  **Q: What is the difference between a Checkpoint and a Savepoint?**
    -   *A*: Checkpoints are automatic for recovery. Savepoints are manual for operations (upgrades/rescaling).

3.  **Q: Why would you use Unaligned Checkpoints?**
    -   *A*: To allow checkpoints to succeed even when the network is saturated (backpressure), at the cost of larger storage.

### Production Challenges
-   **Challenge**: **Checkpoints timing out**.
    -   *Cause*: State is too big, network to S3 is slow, or backpressure is delaying barriers.
    -   *Fix*: Incremental checkpoints, Unaligned checkpoints, or optimize state.

-   **Challenge**: **State Processor API**.
    -   *Scenario*: You need to fix a bug in the state (e.g., remove bad keys) inside a Savepoint.
    -   *Fix*: Use the State Processor API to read/write Savepoints offline.

### Troubleshooting Scenarios
**Scenario**: Job stuck in "In Progress" checkpoint loop.
-   *Cause*: One operator is stuck (infinite loop or deadlock) and not processing the barrier.
""",

    # --- Day 3: Keyed State ---
    "Day3_Keyed_State_Types_Core.md": """# Day 3: Keyed State Types

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
""",
    "Day3_Keyed_State_Types_DeepDive.md": """# Day 3: Keyed State - Deep Dive

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
""",
    "Day3_Keyed_State_Types_Interview.md": """# Day 3: Keyed State - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens to Keyed State when you change parallelism?**
    -   *A*: Flink redistributes the **Key Groups** among the new number of tasks. State is preserved.

2.  **Q: Why is `MapState` better than `ValueState<HashMap>`?**
    -   *A*: `MapState` allows partial updates/reads. `ValueState` requires full serialization/deserialization of the whole map.

3.  **Q: How do you handle State Schema Evolution?**
    -   *A*: Use Avro/Protobuf or Flink's POJO serializer which supports adding fields. Or use the State Processor API to migrate state offline.

### Production Challenges
-   **Challenge**: **State Explosion**.
    -   *Scenario*: Job runs fine for a week, then OOMs.
    -   *Cause*: Unique keys are infinite, and no TTL is set.
    -   *Fix*: Enable TTL.

### Troubleshooting Scenarios
**Scenario**: `InvalidProgramException: Serializer mismatch`.
-   *Cause*: You changed the class structure of the state object but tried to restore from an old checkpoint.
-   *Fix*: Revert code or use State Processor API to migrate.
""",

    # --- Day 4: Operator State ---
    "Day4_Operator_State_Broadcast_Core.md": """# Day 4: Operator State & Broadcast

## Core Concepts & Theory

### Operator State
State bound to a parallel task instance, not a key.
-   **ListState**: A list of items. On rescale, can be:
    -   **Redistributed**: Round-robin.
    -   **Union**: Broadcast to all (everyone gets everything).

### Broadcast State
A special type of Operator State.
-   **Pattern**: One low-throughput "Control Stream" (Rules) is broadcast to all instances of a high-throughput "Data Stream".
-   **Storage**: Replicated on every node.
-   **Usage**: Dynamic Rules, Feature Flags, Lookup Tables.

### Architectural Reasoning
**Why Broadcast?**
Imagine a "Fraud Detection" job. You have 1000 rules. You want to update rules dynamically without restarting.
-   Stream 1: Transactions (Keyed by User).
-   Stream 2: Rules (Broadcast).
-   `connect(rules).process()`: The process function has access to the "current rules" in Broadcast State.

### Key Components
-   `BroadcastProcessFunction`: The function to handle connected streams.
-   `MapStateDescriptor`: Broadcast state is always a Map.
""",
    "Day4_Operator_State_Broadcast_DeepDive.md": """# Day 4: Operator State - Deep Dive

## Deep Dive & Internals

### Checkpointing Operator State
-   **Snapshot**: Each task snapshots its local list/map.
-   **Restore**:
    -   **Even-Split**: (Default for ListState). The global list is shuffled and distributed evenly.
    -   **Union**: (For Kafka Offsets). Every task gets the *full* list of all offsets, then decides which ones it owns.

### Broadcast State Consistency
-   Flink guarantees that the broadcast element is processed by *all* parallel instances.
-   However, there is no "Global Consensus". Instance A might see Rule V2 at time T, while Instance B sees it at T+1. Order is preserved per-stream, but cross-stream synchronization depends on watermarks (if used).

### Advanced Reasoning
**Kafka Source Offsets**
The Kafka Source uses **Operator State (ListState)** to store the offsets `(Topic, Partition, Offset)`.
-   When parallelism increases, the list of partitions is redistributed among the new tasks.

### Performance Implications
-   **Broadcast Size**: Don't broadcast gigabytes of data. It is replicated in memory on *every* TaskManager. It will blow up the heap. Keep broadcast state small (MBs).
""",
    "Day4_Operator_State_Broadcast_Interview.md": """# Day 4: Operator State - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: When would you use Operator State instead of Keyed State?**
    -   *A*: For sources (offsets), sinks (transaction handles), or when you need to broadcast configuration/rules to all nodes.

2.  **Q: How does Broadcast State behave during rescaling?**
    -   *A*: It is replicated. If you scale from 2 to 4 nodes, the new 2 nodes get a copy of the broadcast state.

3.  **Q: What is the difference between `Union` and `Even-Split` redistribution?**
    -   *A*: `Union` sends the full state to everyone. `Even-Split` partitions it.

### Production Challenges
-   **Challenge**: **Broadcast State OOM**.
    -   *Scenario*: Broadcasting a large lookup table (10GB).
    -   *Fix*: Use an external KV store (Redis/Cassandra) with async I/O instead of Broadcast State.

### Troubleshooting Scenarios
**Scenario**: Rules are not applying to some keys.
-   *Cause*: The control stream might be partitioned (Keyed) instead of Broadcast. Ensure you call `.broadcast()`.
""",

    # --- Day 5: State Evolution ---
    "Day5_State_Evolution_Schema_Migration_Core.md": """# Day 5: State Evolution & Schema Migration

## Core Concepts & Theory

### The Problem
You have a running job with State `User(name, age)`. You want to upgrade code to `User(name, age, email)`.
-   If you just deploy, deserialization fails.

### Evolution Strategies
1.  **POJO / Avro Evolution**: Flink supports adding/removing fields if using supported serializers.
2.  **State Processor API**: Offline tool to read a Savepoint, transform it (ETL for State), and write a new Savepoint.
3.  **Drop State**: Start from scratch (reprocess from Kafka).

### Architectural Reasoning
**State Processor API**
Think of it as "MapReduce for Savepoints". It allows you to read the binary snapshot as a DataSet/DataStream, modify it using Flink code, and write it back.
-   Use cases: Schema migration, changing window definitions, bootstrapping state from a DB.

### Key Components
-   `SavepointReader`: Reads savepoint.
-   `SavepointWriter`: Writes savepoint.
-   `BootstrapTransformation`: Defines how to write new state.
""",
    "Day5_State_Evolution_Schema_Migration_DeepDive.md": """# Day 5: State Evolution - Deep Dive

## Deep Dive & Internals

### Serializer Snapshots
When a checkpoint is taken, Flink saves the **Serializer Configuration** (schema).
-   On restore, Flink checks: "Is the registered serializer compatible with the saved one?"
-   **Compatible**: Proceed.
-   **Compatible after Reconfiguration**: Proceed (maybe slower).
-   **Incompatible**: Fail.

### Avro Schema Evolution
If you use Avro:
-   Store the writer schema in the checkpoint.
-   If the new reader schema is compatible (according to Avro rules), Flink handles the migration on-the-fly during restore.

### Advanced Reasoning
**Blue/Green Deployment with State**
1.  Take Savepoint of Job A (Blue).
2.  Use State Processor API to convert Savepoint A -> Savepoint B.
3.  Start Job B (Green) from Savepoint B.
4.  Switch traffic.

### Performance Implications
-   **Migration Cost**: On-the-fly migration (Avro) adds CPU overhead during the first read of each key.
-   **State Processor API**: Is a batch job. Can take time for TB-sized states.
""",
    "Day5_State_Evolution_Schema_Migration_Interview.md": """# Day 5: State Evolution - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you add a field to a POJO in State without losing data?**
    -   *A*: Ensure the POJO follows Flink rules (public fields). Flink's POJO serializer supports adding fields (they will be null for old records).

2.  **Q: What is the State Processor API?**
    -   *A*: A library to read, modify, and write Savepoints offline.

3.  **Q: Can you change the `keyBy` field and restore state?**
    -   *A*: No. Changing the key changes the partitioning. State is bound to the key. You must use State Processor API to re-key the state.

### Production Challenges
-   **Challenge**: **Incompatible Serializer**.
    -   *Scenario*: `java.io.InvalidClassException`.
    -   *Fix*: Provide a custom `TypeSerializerSnapshot` or use State Processor API.

### Troubleshooting Scenarios
**Scenario**: Job fails to restore after adding a field.
-   *Cause*: You were using `Kryo` (generic) serializer instead of POJO. Kryo does not support evolution.
-   *Fix*: Always ensure your classes are recognized as POJOs or Avro from Day 1.
"""
}

print("🚀 Populating Week 4 Stateful Processing Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 4 Content Population Complete!")
