import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase4_Production_CaseStudies\Week10_Challenges_Trends"

content_map = {
    # --- Day 1: Handling Skewed Data ---
    "Day1_Handling_Skewed_Data_Core.md": """# Day 1: Handling Skewed Data

## Core Concepts & Theory

### The Problem of Skew
In distributed systems, we assume data is evenly distributed. In reality, it follows a **Zipfian distribution** (Power Law).
-   **Example**: 80% of traffic comes from 20% of keys (e.g., "Justin Bieber" on Twitter, "iPhone" on Amazon).
-   **Impact**: One partition/task is overloaded (100% CPU), while others are idle. The whole job slows down to the speed of the slowest task.

### Types of Skew
1.  **Data Skew**: Some keys have more data than others.
2.  **Processing Skew**: Some records take longer to process (e.g., complex regex on large payload).

### Mitigation Strategies
1.  **Salting (Random Prefix)**: Add a random number (0-N) to the key. `Key` -> `Key_1`, `Key_2`. Distributes the hot key to N partitions.
2.  **Local Aggregation**: Pre-aggregate on the random key, then global aggregate.
3.  **Broadcast Join**: If one side is small, broadcast it to avoid shuffling the large skewed side.

### Architectural Reasoning
**Why not just "Auto-Scale"?**
Auto-scaling adds more workers. But if *one single key* has more data than *one single CPU* can handle, adding 1000 CPUs won't help. That key must go to one CPU (for correctness/ordering). You MUST break the key (Salting).
""",

    "Day1_Handling_Skewed_Data_DeepDive.md": """# Day 1: Skew - Deep Dive

## Deep Dive & Internals

### Two-Phase Aggregation (Salting)
**Scenario**: Count views for "Justin Bieber" (1M/sec).
**Phase 1 (Local)**:
-   Add Salt: `JustinBieber` -> `JustinBieber_0` ... `JustinBieber_9`.
-   KeyBy: `SaltedKey`.
-   Count: `JustinBieber_0: 100k`, `JustinBieber_1: 100k`...
**Phase 2 (Global)**:
-   KeyBy: Original Key (`JustinBieber`).
-   Sum: `100k + 100k ... = 1M`.
**Result**: The heavy lifting (counting 1M events) is spread across 10 tasks. The final aggregation only sums 10 numbers.

### Skew in Joins
**Scenario**: Join `Orders` (Skewed) with `Users`.
-   **Regular Join**: Both sides shuffled by UserID. Skewed task dies.
-   **Broadcast Join**: Broadcast `Users` to all tasks. `Orders` stays local (no shuffle).
-   **Salted Join**: Salt `Orders` (0-9). Replicate `Users` 10 times (`User_0`...`User_9`). Join `Order_N` with `User_N`.

### Performance Implications
-   **Network**: Salting increases network shuffle (if not careful).
-   **Memory**: Replicating the small side (in Salted Join) increases memory usage.
""",

    "Day1_Handling_Skewed_Data_Interview.md": """# Day 1: Skew - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you detect skew in a running Flink job?**
    -   *A*: Check the Flink Web UI. Look for **Backpressure** on specific subtasks. Check `numRecordsIn` per subtask. If Subtask 0 has 100x more records than Subtask 1, it's skew.

2.  **Q: Can you handle skew without changing the code?**
    -   *A*: Sometimes. Flink SQL has `table.exec.skew-join.enabled` (adaptive skew handling). But usually, you need explicit salting.

3.  **Q: What is the downside of Salting?**
    -   *A*: Correctness complexity. You can't rely on global ordering anymore. And you need a second aggregation step.

### Production Challenges
-   **Challenge**: **Dynamic Skew**.
    -   *Scenario*: A key becomes hot suddenly (Breaking News).
    -   *Fix*: Adaptive Salting. Detect hot keys at runtime and only salt those. (Complex to implement).

-   **Challenge**: **Checkpoint Timeout**.
    -   *Scenario*: Skewed task takes too long to process barriers. Checkpoint fails.
    -   *Fix*: Unaligned Checkpoints (Flink 1.11+). Allows barriers to jump over data.
""",

    # --- Day 2: Schema Evolution ---
    "Day2_Schema_Evolution_Challenges_Core.md": """# Day 2: Schema Evolution Challenges

## Core Concepts & Theory

### The Problem
Streaming jobs run for years. Data formats change.
-   **Producer**: Adds a field `email`.
-   **Consumer**: Expects `email` to be missing.
-   **State**: Flink state contains old objects.

### Compatibility Modes
1.  **Backward**: New schema can read Old data. (Consumer upgrade first).
2.  **Forward**: Old schema can read New data. (Producer upgrade first).
3.  **Full**: Both ways.

### Strategies
1.  **Schema Registry**: Central authority. Rejects incompatible schemas.
2.  **Protobuf/Avro**: Binary formats with built-in evolution rules (tags/defaults).
3.  **JSON**: Flexible but dangerous (no type safety).

### Architectural Reasoning
**Why is State Evolution Hard?**
In a stateless app, you just restart with new code. In Flink, the **State** (Checkpoints) is serialized binary data.
-   If you change the class definition (`User`), Flink cannot deserialize the old checkpoint.
-   **Solution**: Use POJO serialization or Avro for State. Avoid Java Serialization.
""",

    "Day2_Schema_Evolution_Challenges_DeepDive.md": """# Day 2: Schema Evolution - Deep Dive

## Deep Dive & Internals

### Flink State Evolution
How to add a field to a `ValueState<User>`?
1.  **Avro**: If `User` is an Avro generated class, Flink supports schema evolution out of the box (using Avro Serializer).
2.  **POJO**: Flink supports adding fields *if* they are new POJO fields.
3.  **Kryo**: **NO**. Kryo is not compatible. If you change the class, state is lost.

### Schema Registry in Production
-   **Subject Naming**: `topic-value`.
-   **Validation**: Producer checks registry before sending.
-   **Caching**: Producer/Consumer cache the schema ID (4 bytes). The payload only contains the ID + Data.

### Handling "Breaking" Changes
Sometimes you MUST break compatibility (e.g., rename field).
-   **Dual Topics**:
    1.  Create `topic_v2`.
    2.  Producer writes to both (or switches).
    3.  New Consumer reads `topic_v2`.
    4.  Old Consumer reads `topic_v1`.
-   **Translation Layer**: A Flink job that reads `v1`, maps to `v2`, writes to `topic_v2`.

### Performance Implications
-   **Deserialization Overhead**: JSON is slow. Avro is fast. Protobuf is fastest.
-   **Payload Size**: JSON is verbose. Avro/Proto are compact (no field names in payload).
""",

    "Day2_Schema_Evolution_Challenges_Interview.md": """# Day 2: Schema Evolution - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if a producer sends a message with a schema not in the registry?**
    -   *A*: The send fails (if validation is on). Or it registers a new version (if auto-register is on). In Prod, **Auto-Register should be OFF**.

2.  **Q: How do you migrate Flink state to a new schema?**
    -   *A*: **State Processor API**. Read the old Savepoint (offline), map the data to the new class, write a new Savepoint. Start job from new Savepoint.

3.  **Q: JSON vs Avro for Streaming?**
    -   *A*: **Avro**. Strong typing, schema evolution, smaller payload. JSON is only for debugging or external APIs.

### Production Challenges
-   **Challenge**: **The "Unknown Field" Crash**.
    -   *Scenario*: Upstream adds a field. Downstream JSON parser crashes on unknown field.
    -   *Fix*: Configure parser to `IGNORE_UNKNOWN_PROPERTIES`.

-   **Challenge**: **Registry Downtime**.
    -   *Scenario*: Schema Registry is down.
    -   *Fix*: Clients cache schemas. They can survive if they've seen the schema before. New schemas will fail. High Availability (HA) for Registry is critical.
""",

    # --- Day 3: Late Data ---
    "Day3_Late_Data_Correctness_Core.md": """# Day 3: Late Data & Correctness

## Core Concepts & Theory

### The Reality of Time
-   **Event Time**: When it happened (Phone clock).
-   **Processing Time**: When the server saw it (Server clock).
-   **Skew**: The difference. Caused by network partitions, airplane mode, crashes.

### Watermarks
A Watermark `W(T)` means: "I assert that no event with timestamp < T will arrive anymore."
-   It flows with the stream.
-   It triggers windows.
-   It trades **Latency** vs **Completeness**.

### Handling Lateness
1.  **Allowed Lateness**: Keep window state open for X minutes. If late data arrives, update the result.
2.  **Side Output**: If data is *too* late (after allowed lateness), send to a "Late" stream (DLQ) for manual fix.

### Architectural Reasoning
**Correctness vs Latency**
-   **Strict Correctness**: Wait for 100% of data. Latency = Infinity.
-   **Low Latency**: Emit result immediately. Accuracy = Low.
-   **Streaming Solution**: Emit early result (speculative), then emit updates (retractions) as data arrives.
""",

    "Day3_Late_Data_Correctness_DeepDive.md": """# Day 3: Late Data - Deep Dive

## Deep Dive & Internals

### Watermark Strategies
1.  **Monotonous**: Timestamps always increase. (Rare).
2.  **Bounded Out of Orderness**: `Watermark = MaxTimestamp - Delay`.
    -   *Delay*: How much lateness we tolerate before triggering.
    -   *Example*: Delay = 5s. If we see T=100, W=95. We are ready to close window [0-90].

### Idleness
If a partition has no data, it sends no watermarks. The global watermark (Min of all partitions) stalls.
-   **Impact**: Windows never close. Downstream waits forever.
-   **Fix**: `withIdleness(Duration.ofMinutes(1))`. If no data for 1m, mark partition as idle (ignore it for watermark calculation).

### Retractions
If we emit a result "Count=5", and a late event arrives, the count becomes 6.
-   **Append Stream**: Emits `5`, then `6`. (Downstream must know `6` replaces `5`).
-   **Retract Stream**: Emits `5`, then `-5` (Undo), then `6`. (Standard for SQL).
-   **Upsert Stream**: Emits `Key=X, Val=6`. (Overwrites).

### Performance Implications
-   **State Size**: `allowedLateness` keeps windows in memory. Large lateness = Huge state.
""",

    "Day3_Late_Data_Correctness_Interview.md": """# Day 3: Late Data - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Watermark and Event Time?**
    -   *A*: Event Time is an attribute of the *record*. Watermark is a *control signal* in the stream that measures the progress of Event Time.

2.  **Q: Can Watermarks go backwards?**
    -   *A*: No. Watermarks are monotonically increasing. If a source sends a lower watermark, it is ignored.

3.  **Q: How do you handle data from a device that was offline for 3 days?**
    -   *A*: If your window is 1 hour, this data is "Late".
    -   Option A: Drop it.
    -   Option B: Side Output -> Batch Process -> Merge with real-time results.

### Production Challenges
-   **Challenge**: **Stuck Watermark**.
    -   *Scenario*: One Kafka partition is empty. Job stops producing output.
    -   *Fix*: Enable Idleness detection.

-   **Challenge**: **Future Timestamps**.
    -   *Scenario*: Buggy device sends Year 3000. Watermark jumps to 3000. All windows close instantly. Real data is now "Late".
    -   *Fix*: Filter "Future" timestamps at ingestion.

### Troubleshooting Scenarios
**Scenario**: Windows are closing too early (missing data).
-   *Cause*: Watermark delay is too short (e.g., 1s) for the network jitter.
-   *Fix*: Increase bounded out-of-orderness delay.
""",

    # --- Day 4: Streaming Databases ---
    "Day4_Streaming_Databases_Core.md": """# Day 4: Streaming Databases

## Core Concepts & Theory

### The Convergence
-   **Database**: Stores state. Queries are transient.
-   **Stream Processor**: Stores queries (Topology). Data is transient.
-   **Streaming Database**: Stores State AND Continuous Queries. (Materialize, RisingWave, ksqlDB).

### Materialized Views
A standard DB view is calculated on read. A **Materialized View** is pre-calculated.
-   **Streaming DB**: Updates the Materialized View incrementally as new data arrives.
-   **Benefit**: Sub-millisecond query latency for complex joins/aggregates.

### Key Players
1.  **ksqlDB**: Kafka-native. Good for simple transformations.
2.  **Materialize**: Postgres-compatible. Uses Differential Dataflow. Strong consistency.
3.  **RisingWave**: Cloud-native. S3-based state.

### Architectural Reasoning
**Flink vs Streaming DB**
-   **Flink**: Imperative (Java/Python) + SQL. Good for complex logic, external calls, pipelines.
-   **Streaming DB**: Pure SQL. Good for "Serving Layer" (powering dashboards/APIs).
-   **Pattern**: Kafka -> Flink (Complex ETL) -> Kafka -> RisingWave (Serving).
""",

    "Day4_Streaming_Databases_DeepDive.md": """# Day 4: Streaming Databases - Deep Dive

## Deep Dive & Internals

### Incremental View Maintenance (IVM)
How to update `SELECT sum(sales) FROM orders` without rescanning the table?
-   **New Event**: `+10`.
-   **Old Sum**: `500`.
-   **New Sum**: `510`.
-   **Join**: `A JOIN B`. If `A` changes, look up matching `B` in index, emit change.

### Consistency Models
-   **Eventual Consistency**: You might see old data.
-   **Strong Consistency**: You see the correct answer as of a specific timestamp.
-   **Materialize**: Offers "Consistency" (all views update atomically for a timestamp).

### Pull vs Push Queries
-   **Push Query**: "Tell me whenever the result changes". (Subscription / WebSocket).
-   **Pull Query**: "Tell me the current result". (Request / Response).

### Performance Implications
-   **Join State**: Maintaining a join requires storing both tables in state. Expensive.
-   **Churn**: High update rate = High CPU to maintain the view.
""",

    "Day4_Streaming_Databases_Interview.md": """# Day 4: Streaming Databases - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: When would you use ksqlDB over Flink?**
    -   *A*: When the team knows SQL but not Java. When the use case is simple filtering/joining/aggregation on Kafka. For complex state machines or async I/O, use Flink.

2.  **Q: What is the "Read-After-Write" consistency problem in streaming?**
    -   *A*: You write to Kafka, then immediately query the View. The View hasn't updated yet.
    -   *Fix*: Wait for the watermark/offset to catch up, or accept eventual consistency.

3.  **Q: How does a Streaming DB handle backpressure?**
    -   *A*: Same as Flink. Stop reading from Kafka.

### Production Challenges
-   **Challenge**: **State Explosion**.
    -   *Scenario*: `SELECT * FROM events`. Materializing the raw stream.
    -   *Fix*: Only materialize Aggregates or Filtered datasets. Set TTL (Retention) on the view.

-   **Challenge**: **Migration**.
    -   *Scenario*: Changing the SQL query.
    -   *Fix*: Usually requires rebuilding the view from scratch (Replay history).
""",

    # --- Day 5: Unified Batch & Stream ---
    "Day5_Unified_Batch_Stream_Core.md": """# Day 5: Unified Batch & Stream Processing

## Core Concepts & Theory

### The Lambda Architecture (Old)
-   **Speed Layer**: Streaming (Approximate, Fast). Storm/Flink.
-   **Batch Layer**: Hadoop/Spark (Correct, Slow).
-   **Serving Layer**: Merge results.
-   **Problem**: Maintain two codebases (Java vs SQL). "Logic Drift".

### The Kappa Architecture (New)
-   **Everything is a Stream**.
-   **Real-time**: Process latest data.
-   **Batch**: Replay old data (it's just a bounded stream).
-   **Engine**: Flink (or Spark Structured Streaming).
-   **Benefit**: One codebase.

### Flink's Unification
-   **DataStream API**: Works for both Bounded (Batch) and Unbounded (Stream) sources.
-   **Batch Mode**: Flink optimizes for batch (sort-merge shuffle instead of pipelined shuffle) when input is bounded.

### Architectural Reasoning
**Is Batch Dead?**
No.
-   **Ad-hoc Queries**: "Find all users who did X last year". This is Batch.
-   **Model Training**: Training ML models is usually Batch.
-   **But**: The *Engine* can be the same.

""",

    "Day5_Unified_Batch_Stream_DeepDive.md": """# Day 5: Unified Batch/Stream - Deep Dive

## Deep Dive & Internals

### Batch Execution Mode in Flink
When `RuntimeMode.BATCH` is enabled:
1.  **Blocking Shuffle**: Tasks write all output to disk, then next stage reads. (Recoverable).
2.  **Sort-Merge Join**: Sort both inputs, then merge. Faster than Hash Join for huge data.
3.  **No Watermarks**: Not needed. We have all data.

### The "Streaming First" Mindset
Design for Streaming. Run as Batch for backfill.
-   **Windowing**: Works in both.
-   **State**: In Batch, state is just "Spill to Disk". In Stream, it's RocksDB.

### Iceberg / Hudi / Delta
The "Lakehouse" enables Kappa Architecture on S3.
-   **Stream**: Flink writes to Iceberg.
-   **Batch**: Spark/Trino reads from Iceberg.
-   **Upserts**: These formats support `UPDATE/DELETE` on S3.

### Performance Implications
-   **Latency**: Streaming engine overhead is higher than pure Batch engine (Spark) for massive historical loads.
-   **Optimization**: Flink's Batch scheduler is getting better, but Spark is still king of pure Batch ETL.
""",

    "Day5_Unified_Batch_Stream_Interview.md": """# Day 5: Unified Batch/Stream - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the Lambda Architecture?**
    -   *A*: Parallel Batch and Speed layers. Complex to maintain.

2.  **Q: How do you backfill 1 year of data in a Streaming system?**
    -   *A*:
        1.  Start Flink job from "Earliest".
        2.  It processes historical data (high throughput).
        3.  It catches up to "Now".
        4.  It continues as a streaming job.

3.  **Q: Why use Flink for Batch?**
    -   *A*: Code reuse. Write logic once, run on historical data (backtesting) and real-time data (production).

### Production Challenges
-   **Challenge**: **Backfill Speed**.
    -   *Scenario*: Replaying 1 year takes 1 month.
    -   *Fix*: Increase parallelism for the backfill job. Use Batch Mode (Blocking Shuffle) for efficiency.

-   **Challenge**: **Side Effects**.
    -   *Scenario*: Job sends emails. Replaying history sends 1M emails.
    -   *Fix*: Disable "Sinks" (or switch to "Dry Run" Sink) during backfill.
"""
}

print("🚀 Populating Week 10 Challenges Content (Detailed)...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 10 Content Population Complete!")
