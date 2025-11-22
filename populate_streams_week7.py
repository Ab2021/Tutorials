import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase3_Advanced_Architecture\Week7_Reliability_Scalability"

content_map = {
    # --- Day 1: Backpressure ---
    "Day1_Backpressure_Handling_Core.md": """# Day 1: Backpressure & Lag

## Core Concepts & Theory

### What is Backpressure?
When a downstream consumer is slower than the upstream producer.
-   **Symptom**: Buffers fill up.
-   **Mechanism**: The slow consumer stops asking for data. The producer's send buffer fills up. The producer blocks (or slows down).
-   **Propagation**: Backpressure propagates upstream all the way to the source (Kafka).

### Consumer Lag
The difference between the "Latest Offset" in Kafka and the "Current Offset" of the consumer group.
-   **Lag > 0**: Normal.
-   **Lag Increasing**: The consumer cannot keep up.

### Architectural Reasoning
**Handling Backpressure**
1.  **Scale Up**: Increase parallelism (more consumers).
2.  **Optimize**: Fix the slow sink or slow processing logic (e.g., async I/O).
3.  **Drop Data**: If real-time is more important than completeness (e.g., metrics), sample/drop data.

### Key Components
-   **Credit-Based Flow Control**: Flink's internal mechanism to handle backpressure between tasks.
-   **Burrow / Lag Exporter**: Tools to monitor Kafka lag.
""",
    "Day1_Backpressure_Handling_DeepDive.md": """# Day 1: Backpressure - Deep Dive

## Deep Dive & Internals

### Flink Credit-Based Flow Control
-   Each TaskManager has a **Network Buffer Pool**.
-   The receiver sends "credits" to the sender indicating how many buffers it has available.
-   The sender only sends data if it has credits.
-   This prevents a single slow task from overwhelming the network and causing OOMs.

### Identifying the Bottleneck
1.  **Source**: If source is idle but lag is high -> Downstream is slow.
2.  **Sink**: If sink is busy (100% CPU or high I/O wait) -> Sink is the bottleneck.
3.  **Skew**: If only one subtask is backpressured -> Data Skew.

### Advanced Reasoning
**The "Death Spiral"**
If a consumer is slow, it might trigger rebalances (timeouts). Rebalancing stops the world. Lag increases. When it resumes, it has *more* data to catch up, causing more load, leading to another timeout.
-   **Fix**: Increase `session.timeout.ms` and `max.poll.interval.ms`.

### Performance Implications
-   **Buffer Bloat**: Too many buffers increase latency. Flink automatically adjusts buffer size to balance throughput and latency.
""",
    "Day1_Backpressure_Handling_Interview.md": """# Day 1: Backpressure - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does Flink handle backpressure?**
    -   *A*: Using Credit-Based Flow Control. It propagates backpressure naturally without needing a special "Rate Limiter" component.

2.  **Q: What is the difference between Backpressure and Lag?**
    -   *A*: Backpressure is the *state* of the system (buffers full). Lag is the *metric* (offset difference) resulting from backpressure.

3.  **Q: How do you debug a "High Backpressure" alert?**
    -   *A*: Check the Flink Web UI. Find the task with "High" backpressure. Look at its *downstream* task. The bottleneck is usually the first task *without* backpressure (it's the one causing it).

### Production Challenges
-   **Challenge**: **GC Pauses causing Lag**.
    -   *Scenario*: Consumer pauses for 10s due to GC. Kafka rebalances.
    -   *Fix*: Tune JVM (G1GC), increase heap, or reduce state object creation.

### Troubleshooting Scenarios
**Scenario**: Lag is increasing, but CPU is low.
-   *Cause*: I/O Bottleneck (Sink is slow, or Network is saturated).
-   *Fix*: Use Async I/O or batch writes to sink.
""",

    # --- Day 2: Schema Registry ---
    "Day2_Schema_Registry_Evolution_Core.md": """# Day 2: Schema Registry & Evolution

## Core Concepts & Theory

### Why Schema Registry?
In a decoupled system, producers and consumers need a contract.
-   **Producer**: Serializes data using Schema ID 1.
-   **Registry**: Stores Schema 1 = `User(name, age)`.
-   **Consumer**: Downloads Schema 1 to deserialize.

### Evolution Rules
1.  **Backward Compatibility**: New schema can read old data. (Add optional field).
2.  **Forward Compatibility**: Old schema can read new data. (Add optional field, ignore unknown).
3.  **Full Compatibility**: Both ways.

### Architectural Reasoning
**The "Central Nervous System"**
The Schema Registry is the single source of truth for data governance. It prevents "Garbage In, Garbage Out". If a producer tries to send bad data, the serializer fails *before* sending to Kafka.

### Key Components
-   **Subject**: The scope of the schema (usually `TopicName-value`).
-   **ID**: Global unique ID for a schema version.
-   **Avro/Protobuf**: Preferred serialization formats.
""",
    "Day2_Schema_Registry_Evolution_DeepDive.md": """# Day 2: Schema Registry - Deep Dive

## Deep Dive & Internals

### Serialization Process
1.  **Producer**: `record = {name: "A"}`.
2.  **Look up**: Hash the schema. Check local cache. If missing, POST to Registry. Get ID=5.
3.  **Serialize**: `[MagicByte][ID=5][BinaryData]`.
4.  **Send**: To Kafka.

### Deserialization Process
1.  **Consumer**: Read bytes.
2.  **Extract ID**: Read first 5 bytes. ID=5.
3.  **Look up**: Check local cache. If missing, GET /schemas/ids/5.
4.  **Deserialize**: Use Schema 5 to read the binary data.

### Advanced Reasoning
**Transitive Compatibility**
-   **Check**: Is V3 compatible with V2?
-   **Transitive Check**: Is V3 compatible with V2 AND V1?
-   **Why?**: If you have data from 1 year ago (V1) in the topic, the consumer must be able to read it even if the current version is V3.

### Performance Implications
-   **Caching**: The Registry is not in the hot path. Producers/Consumers cache IDs. If Registry goes down, the app continues working (as long as it doesn't encounter a *new* schema).
""",
    "Day2_Schema_Registry_Evolution_Interview.md": """# Day 2: Schema Registry - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if the Schema Registry is down?**
    -   *A*: Producers/Consumers use their local cache. They fail only if they encounter a schema ID they haven't seen before.

2.  **Q: How do you delete a field safely?**
    -   *A*: In Forward Compatibility: Consumers must stop using the field first. Then Producer stops sending it. Then remove from schema.

3.  **Q: Why Avro over JSON?**
    -   *A*: Compact (no field names in payload), fast, and has strict schema evolution rules enforced by the Registry.

### Production Challenges
-   **Challenge**: **Incompatible Schema Change**.
    -   *Scenario*: Dev changes `age` from `int` to `string`. Registry rejects registration. Producer fails.
    -   *Fix*: Follow evolution rules. Add `age_str` as a new field.

### Troubleshooting Scenarios
**Scenario**: `SerializationException: Error retrieving Avro schema`.
-   *Cause*: Network issue to Registry, or the ID in the message does not exist (Registry was wiped?).
""",

    # --- Day 3: Multi-DC ---
    "Day3_Multi_DC_Replication_Core.md": """# Day 3: Multi-DC Replication

## Core Concepts & Theory

### Disaster Recovery (DR)
What if the entire Data Center (AWS Region) goes down?
-   **Active-Passive**: Write to DC1. Replicate to DC2. If DC1 dies, switch to DC2.
-   **Active-Active**: Write to DC1 and DC2. Replicate bi-directionally. (Complex conflict resolution).

### Replication Tools
-   **MirrorMaker 2 (MM2)**: Connect-based. Replicates topics, ACLs, configs.
-   **Confluent Replicator**: Commercial tool.
-   **Redpanda Remote Read Replica**: Native replication.

### Architectural Reasoning
**Offset Translation**
Offsets are not identical across clusters (DC1 offset 100 != DC2 offset 100).
-   MM2 emits checkpoints for the *downstream* cluster so consumers can resume seamlessly.
-   **Timestamp preservation**: Crucial for time-based lookups.

### Key Components
-   **Heartbeats**: To detect cluster health.
-   **Cluster Linking**: Protocol for replication.
""",
    "Day3_Multi_DC_Replication_DeepDive.md": """# Day 3: Multi-DC - Deep Dive

## Deep Dive & Internals

### Active-Active Conflicts
User updates Profile in DC1 (Name=A) and DC2 (Name=B) at the same time.
-   **Last Write Wins (LWW)**: Based on timestamp.
-   **CRDTs**: Conflict-free Replicated Data Types (merge logic).
-   **Sticky Routing**: Route User X always to DC1 to avoid conflicts.

### MirrorMaker 2 Architecture
-   **Source Connector**: Reads from DC1.
-   **Sink Connector**: Writes to DC2.
-   **Checkpoint Connector**: Translates consumer group offsets.
-   **Heartbeat Connector**: Monitoring.

### Advanced Reasoning
**Stretch Cluster vs Replication**
-   **Stretch Cluster**: One Kafka cluster spanning 3 AZs (Availability Zones). Synchronous replication. Zero RPO. High latency.
-   **Async Replication**: Two separate clusters. Asynchronous. Non-zero RPO. Low latency.

### Performance Implications
-   **Bandwidth**: Cross-region traffic is expensive ($$$). Compress data (Zstd) before replicating.
""",
    "Day3_Multi_DC_Replication_Interview.md": """# Day 3: Multi-DC - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between RPO and RTO?**
    -   *A*: RPO (Recovery Point Objective) = How much data can we lose? (e.g., 5 mins). RTO (Recovery Time Objective) = How fast can we be back up? (e.g., 1 hour).

2.  **Q: How does MirrorMaker 2 handle infinite loops in Active-Active?**
    -   *A*: It adds a header to the message indicating the source cluster. It filters out messages that originated from the target cluster.

3.  **Q: Why is Offset Translation necessary?**
    -   *A*: Because messages might be dropped or compacted differently in the two clusters, so Offset 500 in Source might be Offset 450 in Target.

### Production Challenges
-   **Challenge**: **Split Brain**.
    -   *Scenario*: Network partition. Both DCs think they are the "Active" one.
    -   *Fix*: Use a tie-breaker (Witness) or manual failover.

### Troubleshooting Scenarios
**Scenario**: Replication Lag is high.
-   *Cause*: WAN link saturation.
-   *Fix*: Increase batch size, compression, or bandwidth.
""",

    # --- Day 4: Tiered Storage ---
    "Day4_Tiered_Storage_Infinite_Retention_Core.md": """# Day 4: Tiered Storage

## Core Concepts & Theory

### The Problem
Storing PBs of data on local NVMe SSDs is expensive and limits scalability (adding brokers just for disk space).

### The Solution: Tiered Storage
Offload old data to Object Storage (S3/GCS).
-   **Hot Data**: Local Disk (Fast).
-   **Cold Data**: S3 (Cheap, Infinite).
-   **Transparent**: Consumers don't know the difference. The broker fetches from S3 automatically if needed.

### Architectural Reasoning
**Decoupling Compute and Storage**
-   **Stateless Brokers**: If a broker fails, it doesn't need to recover TBs of data. It just connects to S3.
-   **Elasticity**: Scale brokers up/down for *throughput* (CPU/Network), not for *storage*.

### Key Components
-   **Local Cache**: To speed up reads of recently accessed cold data.
-   **Upload Policy**: When to move data to S3 (e.g., after 1 hour).
""",
    "Day4_Tiered_Storage_Infinite_Retention_DeepDive.md": """# Day 4: Tiered Storage - Deep Dive

## Deep Dive & Internals

### Redpanda / Kafka Implementation
-   **Segments**: Kafka logs are split into segments (e.g., 1GB).
-   **Archiver**: A background thread uploads closed segments to S3.
-   **Remote Index**: The broker keeps the index of S3 segments in memory (or local disk).
-   **Fetch**:
    1.  Consumer asks for Offset 0.
    2.  Broker checks local disk. Miss.
    3.  Broker checks Remote Index. Found in S3 Object X.
    4.  Broker downloads Object X (or range read) to local cache.
    5.  Broker serves data to consumer.

### Advanced Reasoning
**Impact on Kappa Architecture**
Tiered Storage is the enabler for Kappa. You can keep years of history.
-   **Backfill**: High-throughput sequential read from S3 is very fast (often saturates network).

### Performance Implications
-   **First Byte Latency**: Reading cold data has higher latency (S3 GET).
-   **Cost**: S3 API costs (GET/PUT) can be high if not optimized (batching).
""",
    "Day4_Tiered_Storage_Infinite_Retention_Interview.md": """# Day 4: Tiered Storage - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the main benefit of Tiered Storage?**
    -   *A*: Cost reduction (S3 is cheaper than SSD) and infinite retention without operational headache.

2.  **Q: Does Tiered Storage affect write latency?**
    -   *A*: No. Writes always go to local disk first. Upload happens asynchronously.

3.  **Q: How does it handle compaction?**
    -   *A*: Compaction usually happens locally before upload, or tiered storage engines support compacted topics in S3 (complex).

### Production Challenges
-   **Challenge**: **S3 Throttling**.
    -   *Scenario*: 503 Slow Down.
    -   *Fix*: Randomize object prefixes (S3 partitioning) or reduce parallelism of archiver.

### Troubleshooting Scenarios
**Scenario**: High egress cost.
-   *Cause*: Consumers reading cold data frequently (e.g., a buggy job restarting from 0 every hour).
-   *Fix*: Fix the consumer or increase local retention.
""",

    # --- Day 5: SRE & Observability ---
    "Day5_SRE_Observability_Core.md": """# Day 5: SRE & Observability

## Core Concepts & Theory

### The 4 Golden Signals
1.  **Latency**: End-to-End latency, Request latency.
2.  **Traffic**: Throughput (Messages/sec, Bytes/sec).
3.  **Errors**: Failed requests, Exceptions, Dead Letter Queue rate.
4.  **Saturation**: CPU, Disk, Network utilization.

### Service Level Objectives (SLO)
-   **SLH (Indicator)**: "99th percentile latency".
-   **SLO (Objective)**: "99% of requests < 100ms".
-   **SLA (Agreement)**: Contract with penalty.

### Architectural Reasoning
**Whitebox vs Blackbox Monitoring**
-   **Whitebox**: Metrics emitted by the app (JMX, Prometheus). "I am processing 100 msg/sec".
-   **Blackbox**: Synthetic checks. "Can I produce to this topic?". "Is the consumer lagging?".

### Key Components
-   **Prometheus/Grafana**: Standard stack.
-   **OpenTelemetry**: Tracing.
""",
    "Day5_SRE_Observability_DeepDive.md": """# Day 5: SRE - Deep Dive

## Deep Dive & Internals

### Distributed Tracing
Tracing a message from Producer -> Kafka -> Flink -> Sink.
-   **Context Propagation**: Injecting `traceparent` header into the Kafka message.
-   **Span**: Each hop creates a span.
-   **Sampling**: Tracing every message is expensive. Sample 0.1%.

### Kafka Lag Monitoring
-   **Burrow**: LinkedIn's tool. Checks if consumer is committing offsets but falling behind.
-   **Flink Metrics**: `records-lag-max`.

### Advanced Reasoning
**Alert Fatigue**
Don't alert on "CPU > 80%". Alert on "SLO Breach" (User is impacted).
-   **Burn Rate**: How fast are we consuming the Error Budget? Alert if we will run out of budget in 4 hours.

### Performance Implications
-   **Metric Cardinality**: Avoid metrics with high cardinality tags (e.g., `user_id`, `transaction_id`). It will kill Prometheus.
""",
    "Day5_SRE_Observability_Interview.md": """# Day 5: SRE - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you calculate End-to-End Latency?**
    -   *A*: Inject a timestamp at the source. Compare with current time at the sink. (Clock synchronization required).

2.  **Q: What is the difference between Liveness and Readiness probes?**
    -   *A*: Liveness = "Am I alive?" (Restart if no). Readiness = "Can I take traffic?" (Remove from load balancer if no).

3.  **Q: Why is high CPU not always bad?**
    -   *A*: If throughput is high and latency is low, high CPU means we are utilizing resources efficiently.

### Production Challenges
-   **Challenge**: **Missing Metrics**.
    -   *Scenario*: Job fails silently. No alert.
    -   *Fix*: Alert on "Absence of Data" (Heartbeat).

### Troubleshooting Scenarios
**Scenario**: Prometheus OOM.
-   *Cause*: Cardinality explosion (someone added `pod_ip` or `session_id` as a label).
-   *Fix*: Drop the label in relabeling rules.
"""
}

print("🚀 Populating Week 7 Reliability Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 7 Content Population Complete!")
