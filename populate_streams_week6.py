import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase3_Advanced_Architecture\Week6_Streaming_Patterns"

content_map = {
    # --- Day 1: Event Sourcing & CQRS ---
    "Day1_Event_Sourcing_CQRS_Core.md": """# Day 1: Event Sourcing & CQRS

## Core Concepts & Theory

### Event Sourcing
Instead of storing the *current state* of an entity, store the *sequence of events* that led to that state.
-   **State**: Derived by replaying events.
-   **Immutability**: Events are facts. They cannot be changed, only compensated (e.g., "OrderCreated", "OrderCancelled").

### CQRS (Command Query Responsibility Segregation)
Splitting the model into:
-   **Command Side (Write)**: Validates commands and emits events. High consistency.
-   **Query Side (Read)**: Consumes events and builds materialized views (e.g., SQL, ElasticSearch). High availability/scalability.

### Architectural Reasoning
**Why Event Sourcing?**
-   **Audit Trail**: You have a perfect history of "who did what and when".
-   **Time Travel**: You can reconstruct the state of the system at any point in time.
-   **Debuggability**: Copy the production event log to dev and replay it to reproduce a bug.

### Key Components
-   **Event Store**: Kafka/Redpanda is the perfect Event Store (durable, ordered).
-   **Projection**: A Flink job that consumes events and updates a Read Model (DB).
""",
    "Day1_Event_Sourcing_CQRS_DeepDive.md": """# Day 1: Event Sourcing - Deep Dive

## Deep Dive & Internals

### Snapshotting
Replaying 1 million events to get the current balance is slow.
-   **Snapshot**: Periodically save the current state (e.g., every 1000 events).
-   **Recovery**: Load latest snapshot + replay subsequent events.
-   **Flink State**: Flink's checkpoints act as automatic snapshots for the stream processing part.

### Event Schema Evolution
-   **Upcasting**: Converting old event formats to new ones on-the-fly during replay.
-   **Weak Schema**: Storing events as JSON blobs (flexible but risky).
-   **Strong Schema**: Avro/Protobuf (safe but requires migration logic).

### Advanced Reasoning
**Consistency in CQRS**
CQRS implies **Eventual Consistency**. The Read Model lags behind the Write Model.
-   **Read-Your-Own-Writes**: A UI pattern where the client waits for the event to be indexed before reloading the page, or optimistically updates the UI.

### Performance Implications
-   **Write Throughput**: Extremely high (append-only).
-   **Read Latency**: Depends on the speed of the Projection engine (Flink).
""",
    "Day1_Event_Sourcing_CQRS_Interview.md": """# Day 1: Event Sourcing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the main disadvantage of Event Sourcing?**
    -   *A*: Complexity. Handling eventual consistency, schema evolution, and GDPR (deleting data from an immutable log) is hard.

2.  **Q: How do you handle GDPR "Right to be Forgotten" in Kafka?**
    -   *A*: Crypto-shredding (encrypting user data with a key, and deleting the key) or using short retention with a compacted topic for the "current state" only.

3.  **Q: What is the difference between a Command and an Event?**
    -   *A*: Command = Intent ("CreateOrder"). Can be rejected. Event = Fact ("OrderCreated"). Cannot be rejected, has already happened.

### Production Challenges
-   **Challenge**: **Replay takes too long**.
    -   *Scenario*: Rebuilding a view takes 2 days.
    -   *Fix*: Parallelize the replay (partitioning) or use snapshots.

### Troubleshooting Scenarios
**Scenario**: Read Model is out of sync.
-   *Cause*: The projection job failed or is lagging.
-   *Fix*: Monitor consumer lag. Implement an "anti-entropy" mechanism to compare Write/Read models periodically.
""",

    # --- Day 2: Kappa Architecture ---
    "Day2_Kappa_vs_Lambda_Architecture_Core.md": """# Day 2: Kappa vs Lambda Architecture

## Core Concepts & Theory

### Lambda Architecture
Hybrid approach (Big Data 1.0).
-   **Speed Layer**: Stream processing (Approximate, Low Latency).
-   **Batch Layer**: Hadoop/Spark (Accurate, High Latency).
-   **Serving Layer**: Merges results.
-   **Problem**: Maintaining two codebases (Batch + Stream) is painful.

### Kappa Architecture
Stream-only approach.
-   **Idea**: "Batch is just a stream with a bounded start and end."
-   **Single Codebase**: Use Flink for both real-time and historical reprocessing.
-   **Long Retention**: Kafka stores data for weeks/months/forever.

### Architectural Reasoning
**Why Kappa?**
Simplicity. You write the logic once (Flink SQL/DataStream). To recompute history (e.g., bug fix), you just start a new instance of the job reading from offset 0.

### Key Components
-   **Unified Engine**: Flink or Spark Structured Streaming.
-   **Tiered Storage**: Makes storing PBs of data in Kafka affordable, enabling Kappa.
""",
    "Day2_Kappa_vs_Lambda_Architecture_DeepDive.md": """# Day 2: Kappa Architecture - Deep Dive

## Deep Dive & Internals

### Reprocessing in Kappa
1.  **Parallel Run**: Start `Job_V2` reading from the beginning. `Job_V1` keeps running.
2.  **Catch Up**: `Job_V2` processes history at high throughput.
3.  **Switch**: When `Job_V2` catches up to real-time, switch the downstream application to read from `Job_V2`'s output. Kill `Job_V1`.

### The "Out-of-Order" Problem
When reprocessing history, data arrives at maximum speed.
-   **Watermarks**: Crucial for handling event time correctly during replay.
-   **Throttling**: You might need to throttle the replay to avoid overwhelming the downstream DB.

### Advanced Reasoning
**Is Lambda dead?**
Not entirely. Some complex ML training or graph algorithms are still better suited for batch (finite datasets). But for ETL and analytics, Kappa is the standard.

### Performance Implications
-   **Backfill Speed**: Depends on parallelism. Flink can process historical data much faster than real-time (limited only by CPU/Network).
""",
    "Day2_Kappa_vs_Lambda_Architecture_Interview.md": """# Day 2: Kappa Architecture - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the main benefit of Kappa over Lambda?**
    -   *A*: Code reuse. You don't need to maintain separate Batch and Streaming paths.

2.  **Q: How do you handle code bugs in Kappa?**
    -   *A*: Fix the code, deploy a new job reading from the beginning (or a savepoint), and reprocess the data.

3.  **Q: Does Kappa require infinite retention in Kafka?**
    -   *A*: Ideally yes (Tiered Storage). Or you can archive old data to S3 (Parquet) and have Flink read from S3 for history and Kafka for real-time (Hybrid Source).

### Production Challenges
-   **Challenge**: **Resource Contention**.
    -   *Scenario*: Replaying history saturates the cluster, affecting real-time jobs.
    -   *Fix*: Use a separate "Batch" cluster for backfills, or use priority/quotas.

### Troubleshooting Scenarios
**Scenario**: Backfill job is slow.
-   *Cause*: Sink bottleneck (e.g., writing to RDS).
-   *Fix*: Optimize the sink (batch writes) or scale the DB.
""",

    # --- Day 3: CDC ---
    "Day3_CDC_Debezium_Core.md": """# Day 3: Change Data Capture (CDC)

## Core Concepts & Theory

### What is CDC?
Capturing changes (INSERT, UPDATE, DELETE) from a database transaction log and streaming them as events.
-   **Pattern**: DB -> CDC Connector (Debezium) -> Kafka -> Flink.

### Debezium
The standard open-source CDC platform.
-   Reads binary logs (MySQL binlog, Postgres WAL).
-   Guarantees **ordering** and **completeness**.

### Architectural Reasoning
**Why CDC?**
-   **No Dual Writes**: Don't write to DB and Kafka manually (race conditions). Write to DB, let CDC propagate to Kafka.
-   **Legacy Integration**: Turn a monolithic SQL DB into an event stream without changing the app code.

### Key Components
-   **Snapshot**: Initial load of the table.
-   **Streaming**: Tailing the log.
-   **Tombstone**: A null value in Kafka indicating a DELETE.
""",
    "Day3_CDC_Debezium_DeepDive.md": """# Day 3: CDC - Deep Dive

## Deep Dive & Internals

### The "Outbox Pattern"
Solving the "Dual Write" problem elegantly.
1.  **Transaction**: App writes to `Orders` table AND `Outbox` table in the *same* DB transaction.
2.  **CDC**: Debezium reads the `Outbox` table and pushes events to Kafka.
3.  **Consumer**: Flink reads Kafka.
-   **Benefit**: Events are guaranteed to be published if and only if the transaction committed.

### Schema Evolution in CDC
-   **DDL**: `ALTER TABLE` in DB.
-   **Debezium**: Detects the change and updates the Schema Registry.
-   **Downstream**: Must handle the new schema (e.g., new column).

### Advanced Reasoning
**Postgres WAL vs MySQL Binlog**
-   **MySQL**: Statement-based (unsafe) or Row-based (safe). Debezium needs Row-based.
-   **Postgres**: Logical Decoding plugins (`pgoutput`).
-   **Impact**: WAL slots in Postgres can fill up disk if the consumer (Debezium) is down.

### Performance Implications
-   **Log Volume**: High-churn tables generate massive CDC logs. Filter unnecessary columns or tables.
""",
    "Day3_CDC_Debezium_Interview.md": """# Day 3: CDC - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the Outbox Pattern?**
    -   *A*: A pattern to ensure transactional consistency between a DB write and a message publication.

2.  **Q: How does Debezium handle initial snapshots?**
    -   *A*: It scans the table (SELECT *) while holding a lock (or using MVCC) to get a consistent snapshot, then switches to log tailing.

3.  **Q: What happens if you delete a row in the DB?**
    -   *A*: Debezium emits a record with `op='d'` (delete) and then a Tombstone (Key, Null) to allow Kafka Log Compaction to remove it.

### Production Challenges
-   **Challenge**: **WAL Disk Usage**.
    -   *Scenario*: Debezium is down, Postgres keeps WAL segments forever. Disk fills up. DB crashes.
    -   *Fix*: Monitor replication slot lag. Set `max_slot_wal_keep_size`.

### Troubleshooting Scenarios
**Scenario**: CDC stream is lagging.
-   *Cause*: DB is under heavy write load. Debezium is single-threaded per connector.
-   *Fix*: Shard the connector (one per table) or optimize DB log I/O.
""",

    # --- Day 4: Stream Joins ---
    "Day4_Stream_Joins_Enrichment_Core.md": """# Day 4: Stream Joins & Enrichment

## Core Concepts & Theory

### Join Types
1.  **Stream-Stream Join**: Windowed Join or Interval Join. (e.g., AdClick JOIN AdImpression).
2.  **Stream-Table Join (Enrichment)**:
    -   **Lookup Join**: Query external DB for every record. (Slow).
    -   **Temporal Join**: Join with a versioned table in Flink state. (Fast).

### Enrichment Patterns
-   **Async I/O**: Use `AsyncDataStream` to query a remote service (User Profile Service) without blocking.
-   **Broadcast State**: Broadcast the "Dimension Table" (e.g., Currency Rates) to all nodes. Local lookup.

### Architectural Reasoning
**Latency vs Freshness**
-   **Broadcast**: Zero latency, but data might be stale (eventual consistency).
-   **Async I/O**: High latency (network RTT), but data is fresh.
-   **Temporal Join**: Best of both. High throughput, consistent point-in-time semantics.

### Key Components
-   `AsyncFunction`
-   `BroadcastProcessFunction`
-   `TemporalTableFunction`
""",
    "Day4_Stream_Joins_Enrichment_DeepDive.md": """# Day 4: Stream Joins - Deep Dive

## Deep Dive & Internals

### Temporal Table Join
`SELECT * FROM Orders o JOIN Rates FOR SYSTEM_TIME AS OF o.ts r ON o.currency = r.currency`
-   Flink keeps the history of `Rates` in state.
-   When an Order arrives with `ts=10:00`, Flink looks up the rate *at 10:00*, even if the current time is 10:05.
-   **Prerequisite**: The Rates stream must be a changelog.

### Async I/O Caching
To reduce load on the external DB:
-   **Cache**: Use Guava/Caffeine cache inside the AsyncFunction.
-   **TTL**: Expire cache entries to balance freshness.

### Advanced Reasoning
**Handling Late Data in Joins**
If an Order arrives late (`ts=09:55`), and we only keep Rates for 1 hour, and current time is 11:00, the join might fail or produce incorrect results if state was cleaned up.
-   **Watermarks** drive state cleanup.

### Performance Implications
-   **Lookup Join**: The bottleneck is usually the external DB. Use a high-performance KV store (Redis/Cassandra) and batch requests.
""",
    "Day4_Stream_Joins_Enrichment_Interview.md": """# Day 4: Stream Joins - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between a Window Join and an Interval Join?**
    -   *A*: Window Join joins elements in the same fixed window. Interval Join joins elements within a relative time range (e.g., A.ts between B.ts - 5min and B.ts + 5min).

2.  **Q: How do you optimize a stream enrichment with a large static dataset?**
    -   *A*: If it fits in memory, use Broadcast State. If not, use Async I/O with caching or a Temporal Join if the dataset changes over time.

3.  **Q: What is "Skew" in joins?**
    -   *A*: One key (e.g., "Null" or a popular Item) has massive data. All goes to one node.
    -   *Fix*: Salt the key (add random suffix) to distribute load.

### Production Challenges
-   **Challenge**: **External Service Failure**.
    -   *Scenario*: Async I/O calls to User Service start timing out.
    -   *Fix*: Circuit Breaker pattern, exponential backoff, and fallback (default values).

### Troubleshooting Scenarios
**Scenario**: Join producing no results.
-   *Cause*: Timezones! One stream is UTC, other is EST. Timestamps don't align.
""",

    # --- Day 5: Exactly-Once E2E ---
    "Day5_Exactly_Once_E2E_Core.md": """# Day 5: Exactly-Once End-to-End

## Core Concepts & Theory

### The Guarantee
"Exactly-Once" usually means **Exactly-Once State Processing**.
"End-to-End Exactly-Once" means **Source + Processing + Sink** guarantees no duplicates in the external system.

### Requirements
1.  **Source**: Must be replayable (Kafka).
2.  **Processing**: Deterministic state (Flink Checkpointing).
3.  **Sink**: Must be Transactional (Two-Phase Commit) or Idempotent.

### Two-Phase Commit (2PC)
-   **Phase 1 (Pre-Commit)**: Flink writes data to the sink (e.g., Kafka "pending" transaction). Happens continuously.
-   **Phase 2 (Commit)**: When Flink completes a checkpoint, it tells the sink to "Commit" the transaction.

### Architectural Reasoning
**Idempotency vs Transactions**
-   **Idempotent Sink**: `PUT(key, val)`. If you retry, it just overwrites. Simple, fast. (Redis, Cassandra, Elastic).
-   **Transactional Sink**: `BEGIN... INSERT... COMMIT`. Needed for append-only systems (Kafka, Files) or multi-row updates (RDBMS).

### Key Components
-   `TwoPhaseCommitSinkFunction`
-   `KafkaSink` (EOS mode)
""",
    "Day5_Exactly_Once_E2E_DeepDive.md": """# Day 5: Exactly-Once - Deep Dive

## Deep Dive & Internals

### Kafka Transactional Protocol
Flink acts as a Kafka Producer.
-   **Transaction ID**: Derived from `JobName + OperatorID`. Must be consistent across restarts.
-   **Zombies**: If a Flink task crashes, the new task must "fence" the old zombie transaction to prevent data corruption. Kafka handles this via Epochs.

### The "Read-Committed" Isolation
Downstream consumers MUST be configured with `isolation.level=read_committed`.
-   Otherwise, they will see "open" (uncommitted) transactions, breaking exactly-once.

### Advanced Reasoning
**Latency Trade-off**
E2E Exactly-Once adds latency.
-   Data is only visible downstream after the checkpoint completes (Commit).
-   Latency = `checkpoint.interval` + processing time.
-   If you need sub-second latency, you might have to accept At-Least-Once.

### Performance Implications
-   **Transaction Overhead**: Kafka transactions have overhead. Don't set checkpoint interval too low (e.g., < 1s).
""",
    "Day5_Exactly_Once_E2E_Interview.md": """# Day 5: Exactly-Once - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: Can you achieve Exactly-Once with a non-transactional sink?**
    -   *A*: Only if the sink is Idempotent (e.g., KV store). If it's append-only (like a log) and not transactional, you will get duplicates (At-Least-Once).

2.  **Q: What happens to open transactions if the Flink job fails?**
    -   *A*: Flink recovers, aborts the old transactions (or lets them time out), and replays from the last checkpoint.

3.  **Q: Why do consumers need `read_committed`?**
    -   *A*: To ignore messages that are part of an aborted transaction or a transaction that is still in progress.

### Production Challenges
-   **Challenge**: **Data not visible**.
    -   *Scenario*: Pipeline running, but downstream sees nothing.
    -   *Cause*: `isolation.level=read_committed` and Checkpoints are failing (so nothing is ever committed).
    -   *Fix*: Fix checkpoints.

### Troubleshooting Scenarios
**Scenario**: `ProducerFencedException`.
-   *Cause*: Multiple producers with the same Transactional ID. Usually caused by a zombie task or misconfiguration.
"""
}

print("🚀 Populating Week 6 Streaming Patterns Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 6 Content Population Complete!")
