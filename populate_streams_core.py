import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda"

core_content = {
    "Phase1_Foundations/Week1_Kafka_Architecture/Day1_Intro_Log_Abstraction_Core.md": """# Day 1: The Log Abstraction & Event Streaming

## Core Concepts & Theory

### The "Log" Abstraction
In the context of streaming systems, a **Log** is not just a text file for error messages. It is an **append-only, totally ordered sequence of records ordered by time**.
- **Append-Only**: You can only add to the end. Old data is immutable.
- **Ordered**: Events have a strict order ($t_1 < t_2 < t_3$).
- **Durable**: The log is persisted to disk.

This simple abstraction is the heart of Kafka and Redpanda. It unifies database commit logs and distributed messaging.

### Batch vs. Stream Processing
- **Batch**: Processing a bounded dataset at rest. High latency, high throughput. "Tell me what happened yesterday."
- **Stream**: Processing an unbounded dataset in motion. Low latency. "Tell me what is happening right now."
- **The Dual Nature**: A table is a snapshot of a stream at a point in time. A stream is the history of changes to a table.

### Architectural Reasoning
**Why the Log?**
1.  **Simplicity**: Append-only operations are O(1) and extremely fast on spinning disks and SSDs (sequential I/O).
2.  **Buffering**: Decouples producers from consumers. Producers don't block if consumers are slow.
3.  **Replayability**: Because the log is durable, consumers can rewind and re-read data. This enables fault tolerance and new use cases (like training a new ML model on old data).

### Key Components
-   **Event**: A key-value pair with a timestamp (e.g., `UserLogin {id: 123, time: 10:00}`).
-   **Stream**: An unbounded sequence of events.
-   **Producer**: The application creating events.
-   **Consumer**: The application reading events.
""",
    "Phase1_Foundations/Week1_Kafka_Architecture/Day2_Kafka_Architecture_Deep_Dive_Core.md": """# Day 2: Kafka Architecture Deep Dive

## Core Concepts & Theory

### The Broker
A **Broker** is a single Kafka server. It receives messages from producers, assigns them offsets, and commits them to storage on disk. It also services fetch requests from consumers.

### The Cluster
A **Cluster** is a group of brokers working together.
-   **Controller**: One broker is elected as the Controller. It manages the states of partitions and replicas and performs administrative tasks (like reassigning partitions).
-   **Metadata**: Information about where partitions exist. Clients (producers/consumers) cache this metadata to know which broker to talk to.

### Zookeeper vs. KRaft
-   **Zookeeper (Legacy)**: External service used for cluster coordination, leader election, and storing metadata. It was a bottleneck and operational burden.
-   **KRaft (Kafka Raft)**: The modern architecture (KIP-500). Metadata is stored in an internal Kafka topic (`@metadata`). The Controller is now a quorum of brokers using the Raft consensus algorithm. This removes the Zookeeper dependency.

### Architectural Reasoning
**Why Dumb Broker / Smart Client?**
Kafka pushes complexity to the client. The broker does not track which messages a consumer has read.
-   **Broker**: "Here is message 100 to 200." (Stateless-ish)
-   **Consumer**: "I have read up to 200." (Tracks its own state)
This design allows Kafka to scale massively because the broker does minimal work per consumer.

### Key Components
-   **Broker**: Storage and network layer.
-   **Controller**: Brain of the cluster.
-   **Zookeeper/KRaft**: Consensus and metadata store.
""",
    "Phase1_Foundations/Week1_Kafka_Architecture/Day3_Topics_Partitions_Segments_Core.md": """# Day 3: Topics, Partitions, and Segments

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
""",
    "Phase1_Foundations/Week1_Kafka_Architecture/Day4_Producers_Consumers_Core.md": """# Day 4: Producers & Consumers

## Core Concepts & Theory

### The Producer
Producers write data to topics.
-   **Partitioning Strategy**: How does the producer decide which partition to send to?
    -   *Round-Robin*: If no key is provided.
    -   *Key-Hash*: If a key is provided (`hash(key) % num_partitions`). This ensures all events for the same key (e.g., `user_id`) go to the same partition (and thus are ordered).
-   **Batching**: Producers buffer messages to send them in batches for higher throughput.

### The Consumer
Consumers read data from topics.
-   **Pull Model**: Consumers pull data from brokers. This allows the consumer to control the rate (backpressure).
-   **Consumer Groups**: A set of consumers working together to consume a topic.
    -   Each partition is consumed by *only one* consumer in the group.
    -   This is the mechanism for **parallel consumption**.

### Architectural Reasoning
**Why Consumer Groups?**
If you have a topic with 1TB of data, a single consumer is too slow. You want to parallelize. Consumer Groups allow you to spin up N consumers, and Kafka automatically distributes the M partitions among them. If a consumer fails, Kafka performs a **Rebalance** to reassign its partitions to the survivors.

### Key Components
-   **ProducerRecord**: The object sent (Key, Value, Timestamp).
-   **ConsumerGroup**: Logical grouping for parallel processing.
-   **Rebalancing**: The process of redistributing partitions.
""",
    "Phase1_Foundations/Week1_Kafka_Architecture/Day5_Reliability_Durability_Core.md": """# Day 5: Reliability & Durability

## Core Concepts & Theory

### Replication
Kafka replicates partitions across multiple brokers for fault tolerance.
-   **Leader**: The replica that handles all reads and writes.
-   **Follower**: Passive replicas that fetch data from the leader to stay in sync.
-   **ISR (In-Sync Replicas)**: The set of replicas that are currently caught up with the leader.

### Acknowledgements (acks)
Producers can choose their durability level:
-   `acks=0`: Fire and forget. Fastest, least safe.
-   `acks=1`: Leader acknowledges. Safe from follower failure, not leader failure.
-   `acks=all`: All ISRs acknowledge. Strongest durability.

### Min.Insync.Replicas
This config defines the minimum number of replicas that must acknowledge a write for it to be considered successful when `acks=all`.
-   If `min.insync.replicas=2` and only 1 replica is alive, the broker rejects the write.

### Architectural Reasoning
**Consistency vs. Availability (CAP Theorem)**
Kafka defaults to Availability (AP) but can be tuned for Consistency (CP).
-   `acks=all` + `min.insync.replicas=2` favors Consistency.
-   `acks=1` favors Availability and Latency.

### Key Components
-   **Replication Factor**: Total copies of data (usually 3).
-   **ISR**: The "healthy" replicas.
-   **High Watermark**: The offset up to which all ISRs have replicated. Consumers can only read up to here.
""",
    "Phase1_Foundations/Week2_Redpanda_HighPerformance/Day1_Intro_Redpanda_Architecture_Core.md": """# Day 1: Redpanda Architecture

## Core Concepts & Theory

### What is Redpanda?
Redpanda is a modern, Kafka-compatible streaming platform written in C++. It is designed to be a drop-in replacement for Kafka but with significantly higher performance and operational simplicity.

### Thread-Per-Core Architecture
Unlike Kafka (JVM-based), which relies on the OS kernel for thread scheduling and page cache, Redpanda uses a **Thread-Per-Core** (TPC) architecture (Seastar framework).
-   **Shared-Nothing**: Each core has its own memory and task queue. There is no locking or contention between cores.
-   **Direct I/O**: Redpanda bypasses the OS page cache and manages disk I/O directly (DMA).

### Architectural Reasoning
**Why C++ and TPC?**
Hardware has changed. Modern NVMe SSDs and 100GbE networks are incredibly fast. The JVM and the Linux kernel context switching overhead become bottlenecks.
-   **Zero-Copy**: Redpanda moves data from disk to network with minimal CPU involvement.
-   **Tail Latency**: By pinning threads to cores and avoiding GC pauses (no JVM), Redpanda offers predictable, low tail latency (p99).

### Key Components
-   **Seastar**: The C++ framework for high-performance async I/O.
-   **Single Binary**: No Zookeeper. Redpanda includes a built-in Raft consensus engine.
""",
    "Phase1_Foundations/Week2_Redpanda_HighPerformance/Day2_Redpanda_vs_Kafka_Core.md": """# Day 2: Redpanda vs. Kafka

## Core Concepts & Theory

### Performance Comparison
-   **Throughput**: Redpanda can often achieve 10x the throughput of Kafka on the same hardware due to its efficient I/O and lack of JVM overhead.
-   **Latency**: Redpanda maintains single-digit millisecond latency even at high loads.

### Operational Simplicity
-   **Kafka**: Requires Zookeeper (historically), JVM tuning (Heap size, GC algorithms), and OS tuning (Page cache, file descriptors).
-   **Redpanda**: A single binary. Autotunes itself to the hardware (`rpk redpanda tune`). No Zookeeper.

### WASM Transforms
Redpanda allows you to run **WebAssembly (WASM)** code directly inside the broker.
-   **Data Sovereignty**: Filter or mask PII data *before* it leaves the broker.
-   **Push-down Processing**: Move computation to the data, rather than moving data to the computation.

### Architectural Reasoning
**The Cost of Complexity**
Kafka's complexity leads to "Kafka teams" just to manage the cluster. Redpanda aims to be "developer-first" by removing the operational burden, allowing teams to focus on the application logic.

### Key Components
-   **rpk**: The Redpanda CLI tool (all-in-one).
-   **WASM Engine**: Embedded V8 engine for transforms.
""",
    "Phase2_Stream_Processing_Flink/Week3_Flink_Fundamentals/Day1_Intro_Stream_Processing_Core.md": """# Day 1: Introduction to Stream Processing

## Core Concepts & Theory

### The Dataflow Model
Stream processing is based on the **Dataflow Model** (pioneered by Google).
-   **DAG (Directed Acyclic Graph)**: A job is represented as a graph where nodes are operators (Map, Filter, KeyBy) and edges are data streams.
-   **Parallelism**: Each operator can have multiple parallel instances running on different machines.

### JobManager & TaskManager
-   **JobManager (Master)**: Coordinates the execution, checkpoints, and recovery. It turns the JobGraph into an ExecutionGraph.
-   **TaskManager (Worker)**: Executes the actual tasks (sub-tasks of operators) in slots.

### Architectural Reasoning
**Why Flink?**
Flink is a **True Streaming** engine.
-   **Spark Streaming**: Micro-batch (simulates streaming by chopping data into small batches). High latency.
-   **Flink**: Row-at-a-time processing. Ultra-low latency.
Flink treats batch processing as a special case of streaming (bounded stream).

### Key Components
-   **DataStream**: The core abstraction for unbounded data.
-   **Operator**: A transformation function.
-   **Slot**: The unit of resource allocation in a TaskManager.
""",
    "Phase2_Stream_Processing_Flink/Week3_Flink_Fundamentals/Day3_Time_Semantics_Watermarks_Core.md": """# Day 3: Time Semantics & Watermarks

## Core Concepts & Theory

### The Three Times
1.  **Event Time**: The time the event actually occurred (timestamp in the device). This is what matters for correctness.
2.  **Processing Time**: The time the event arrived at the Flink machine.
3.  **Ingestion Time**: The time the event entered the Flink source.

### The Problem of Out-of-Order Data
In distributed systems, events often arrive out of order. An event from 10:00 might arrive after an event from 10:05 due to network lag.
If you calculate "How many clicks at 10:00?", you can't just wait for the clock to hit 10:01. You might miss the late data.

### Watermarks
A **Watermark(t)** is a control message that flows through the stream and asserts: "No more events with timestamp < t will arrive."
-   It allows the system to measure progress in Event Time.
-   When a window operator receives Watermark(t), it knows it can safely close any window ending before t.

### Architectural Reasoning
**Correctness vs. Latency**
Watermarks allow you to trade off latency for correctness.
-   **Aggressive Watermarks**: Assume little lag. Low latency, but data might be dropped as "late".
-   **Conservative Watermarks**: Assume huge lag. High correctness, but high latency (waiting for late data).

### Key Components
-   **TimestampAssigner**: Extracts the timestamp from the event.
-   **WatermarkGenerator**: Emits watermarks (Periodic or Punctuated).
"""
}

print("🚀 Populating Core Content...")

for path, content in core_content.items():
    full_path = os.path.join(base_path, path)
    if os.path.exists(full_path):
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {path}")
    else:
        print(f"⚠️ File not found: {path}")

print("✅ Core Content Population Complete!")
