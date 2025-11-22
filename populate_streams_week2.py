import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase1_Foundations\Week2_Redpanda_HighPerformance"

content_map = {
    # --- Day 1: Intro to Redpanda ---
    "Day1_Intro_Redpanda_Architecture_DeepDive.md": """# Day 1: Redpanda Architecture - Deep Dive

## Deep Dive & Internals

### Seastar Framework
Redpanda is built on **Seastar**, a C++ framework for high-performance server applications.
-   **Futures/Promises**: Uses a future-based programming model for non-blocking I/O.
-   **Shared-Nothing**: Memory is pre-allocated per core. No locks are needed for memory access, avoiding contention.
-   **User-Space Networking**: Can use DPDK to bypass the kernel network stack (though often runs on standard sockets for compatibility).

### The Log Structure (Segments)
Redpanda uses a similar segment structure to Kafka but optimized.
-   **Open-Addressing Hash Table**: For the index (vs Kafka's sparse index).
-   **DMA (Direct Memory Access)**: Writes go from memory to disk controller without CPU copying.

### Advanced Reasoning
**Why Thread-Per-Core?**
In traditional multi-threaded apps (like JVM Kafka), threads fight for CPU time and locks. Context switches are expensive (microseconds). By pinning one thread to one core, Redpanda treats the CPU as a distributed system of independent nodes, communicating via message passing. This eliminates lock contention and maximizes instruction-per-cycle (IPC).

### Performance Implications
-   **Tail Latency**: Because there is no "Stop-the-World" GC, p99 latency is stable even at high throughput.
-   **Hardware Utilization**: Redpanda can saturate NVMe SSDs and 100GbE NICs with fewer CPUs than Kafka.
""",
    "Day1_Intro_Redpanda_Architecture_Interview.md": """# Day 1: Redpanda Architecture - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does Redpanda differ from Kafka architecturally?**
    -   *A*: Redpanda is C++ (vs Java), uses Thread-Per-Core architecture (Seastar), has no Zookeeper dependency (built-in Raft), and is a single binary.

2.  **Q: What is the advantage of bypassing the Page Cache?**
    -   *A*: It gives the application full control over caching and eviction policies, preventing "neighbor noise" from other processes and avoiding double-buffering (storing data in both JVM heap and OS cache).

3.  **Q: Is Redpanda 100% compatible with Kafka?**
    -   *A*: It is compatible with the Kafka *API* (protocol). Existing Kafka clients work without modification. However, it does not support Kafka *plugins* (JARs) like custom Authorizers or Tiered Storage implementations (it has its own).

### Production Challenges
-   **Challenge**: **CPU Pinning in Containers**.
    -   *Scenario*: Running Redpanda in Docker/K8s without dedicated CPU limits.
    -   *Issue*: Seastar expects to own the core. If shared, performance degrades.
    -   *Fix*: Use `--cpuset-cpus` or K8s `cpu-manager-policy: static`.

### Troubleshooting Scenarios
**Scenario**: High latency on one specific node.
-   **Check**: Is the node overloaded? (Use `rpk cluster status`).
-   **Check**: Is there a "hot partition" on that node?
""",

    # --- Day 2: Redpanda vs Kafka ---
    "Day2_Redpanda_vs_Kafka_DeepDive.md": """# Day 2: Redpanda vs Kafka - Deep Dive

## Deep Dive & Internals

### Autotuning (`rpk redpanda tune`)
Redpanda includes a tuner that optimizes the OS for streaming.
-   **NIC Interrupts**: Distributes IRQs across cores to prevent one core from handling all network traffic.
-   **Disk I/O Scheduler**: Sets the scheduler to `noop` or `deadline` (preferred for SSDs).
-   **AIO**: Enables Asynchronous I/O.

### Memory Management
-   **Kafka**: Heap (Garbage Collected) + Off-heap (Page Cache). Tuning heap size is an art.
-   **Redpanda**: Allocates *all* available RAM at startup. It manages its own cache. No GC pauses.

### Advanced Reasoning
**Why is "Single Binary" a feature?**
-   **Deployment**: `apt-get install redpanda`. No JVM, no Zookeeper, no separate Jars.
-   **Upgrades**: Rolling upgrades are simpler.
-   **Edge Computing**: The small footprint allows Redpanda to run on IoT devices or edge servers where a full JVM+ZK stack is too heavy.

### Performance Implications
-   **Cold Reads**: Redpanda's Shadow Indexing (Tiered Storage) is often faster than Kafka's because it can prefetch data from S3 more aggressively using its async engine.
""",
    "Day2_Redpanda_vs_Kafka_Interview.md": """# Day 2: Redpanda vs Kafka - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: Why might you choose Kafka over Redpanda?**
    -   *A*: Ecosystem maturity (thousands of plugins), enterprise support (Confluent), existing deep expertise in the team, or specific legacy requirements.

2.  **Q: How does Redpanda handle Zookeeper removal?**
    -   *A*: It implements the Raft consensus algorithm directly within the broker code to manage cluster metadata.

3.  **Q: What is "WASM" in Redpanda?**
    -   *A*: WebAssembly. It allows users to run custom data transformation code (filters, masking) *inside* the broker, close to the data.

### Production Challenges
-   **Challenge**: **Migration**.
    -   *Scenario*: Moving from Kafka to Redpanda.
    -   *Strategy*: Use MirrorMaker 2 or Redpanda's built-in replication to sync data, then switch consumers.

### Troubleshooting Scenarios
**Scenario**: "Connection Refused" on Redpanda.
-   **Check**: `rpk` configuration. Redpanda binds to specific IPs. Ensure `advertised_kafka_api` is reachable by the client.
""",

    # --- Day 3: Schema Registry ---
    "Day3_Schema_Registry_Serialization_Core.md": """# Day 3: Schema Registry & Serialization

## Core Concepts & Theory

### The Contract
In a decoupled system, Producers and Consumers need a **Contract** to understand the data format.
-   **JSON**: Flexible but verbose. No schema enforcement. "Schemaless".
-   **Avro/Protobuf**: Binary, compact, strongly typed. Requires a schema.

### The Schema Registry
A central repository that stores schemas.
1.  **Producer**: Checks Registry. "Is this schema ID 1?" -> Sends `[ID=1][Payload]`.
2.  **Consumer**: Reads `[ID=1]`. Asks Registry "What is schema 1?" -> Deserializes payload.

### Compatibility Modes
-   **Backward**: New schema can read old data. (Delete field).
-   **Forward**: Old schema can read new data. (Add optional field).
-   **Full**: Both ways.

### Architectural Reasoning
**Why Schema Registry?**
It prevents "Poison Pills". If a producer changes the data format (e.g., renames "id" to "user_id") without a registry, downstream consumers will crash. The Registry rejects incompatible changes at the *producer* level.

### Key Components
-   **Subject**: Scope for a schema (usually `topic-value`).
-   **Schema ID**: Global unique ID.
-   **Avro/Protobuf/JSON Schema**: Supported formats.
""",
    "Day3_Schema_Registry_Serialization_DeepDive.md": """# Day 3: Schema Registry - Deep Dive

## Deep Dive & Internals

### Caching
-   **Client-Side**: Producers and Consumers cache Schema IDs. They don't hit the Registry for every message.
-   **Server-Side**: The Registry is backed by a Kafka topic (`_schemas`). It is a stateful application built on top of Kafka.

### Redpanda's Built-in Registry
Redpanda embeds the Schema Registry *inside* the broker binary.
-   **Port 8081**: Exposes the standard Confluent Schema Registry API.
-   **No Sidecar**: You don't need a separate `schema-registry` container.

### Advanced Reasoning
**Why Binary Formats (Avro/Proto)?**
-   **Size**: JSON `{"id": 123456789}` is 16 bytes. Avro might be 4 bytes (varint). At 1TB/day, this saves 50% storage and bandwidth.
-   **Parsing Speed**: Binary parsing is orders of magnitude faster than JSON parsing.

### Performance Implications
-   **First Request Latency**: The first message is slow (fetching schema). Subsequent messages are fast (cached).
""",
    "Day3_Schema_Registry_Serialization_Interview.md": """# Day 3: Schema Registry - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if the Schema Registry goes down?**
    -   *A*: Existing producers/consumers continue working (cached schemas). New producers/consumers (or new schemas) will fail.

2.  **Q: Explain Backward Compatibility.**
    -   *A*: Consumers with the *new* schema can read data written with the *old* schema. This allows you to upgrade consumers *first*.

3.  **Q: How does the consumer know which schema to use?**
    -   *A*: The first 5 bytes of the message payload contain the "Magic Byte" and the 4-byte Schema ID.

### Production Challenges
-   **Challenge**: **Incompatible Schema Change**.
    -   *Scenario*: Developer renames a required field. Registry rejects the registration.
    -   *Fix*: Add an alias (if supported) or create a new version with a default value.

### Troubleshooting Scenarios
**Scenario**: `SerializationException: Unknown magic byte`.
-   *Cause*: You are trying to consume data using an Avro deserializer, but the data was produced as raw JSON or String.
""",

    # --- Day 4: Admin Operations ---
    "Day4_Admin_Operations_Tiered_Storage_Core.md": """# Day 4: Admin Operations & Tiered Storage

## Core Concepts & Theory

### Tiered Storage (Shadow Indexing)
Tiered Storage allows you to offload older log segments to object storage (S3, GCS).
-   **Hot Set**: Recent data stays on local NVMe SSD (fast).
-   **Cold Set**: Old data moves to S3 (cheap, infinite).
-   **Transparent**: Consumers don't know the difference. They just request offset 0, and the broker fetches it from S3.

### Partition Rebalancing
Moving partitions between brokers to balance load.
-   **Data Movement**: Heavy operation. Uses network bandwidth.
-   **Throttling**: Crucial to avoid starving production traffic during rebalance.

### Architectural Reasoning
**Why Tiered Storage?**
It decouples **Compute** (Brokers/CPU) from **Storage** (Disk).
-   Without Tiered Storage: To store more data, you need more brokers (expensive).
-   With Tiered Storage: You just pay for S3. You can have a small cluster storing PBs of data.

### Key Components
-   **Remote Write**: Uploading segments to S3.
-   **Remote Read**: Fetching segments from S3.
-   **Cache**: Local disk cache for recently accessed remote segments.
""",
    "Day4_Admin_Operations_Tiered_Storage_DeepDive.md": """# Day 4: Admin Operations - Deep Dive

## Deep Dive & Internals

### Redpanda Tiered Storage Internals
-   **Manifest**: Redpanda maintains a manifest file in S3 that lists all segments.
-   **Upload**: As soon as a segment is closed (or reaches a size limit), it is queued for upload.
-   **Prefetching**: When a consumer reads from S3, Redpanda speculatively fetches the *next* segment to hide latency.

### ACLs (Access Control Lists)
-   **Principal**: User (User:Alice).
-   **Resource**: Topic, Group, Cluster.
-   **Operation**: Read, Write, Describe, Create.
-   **Pattern**: Literal (exact name) or Prefixed.

### Advanced Reasoning
**Why is Rebalancing hard?**
It involves copying gigabytes of data. If you do it too fast, you saturate the NIC. If you do it too slow, the cluster remains unbalanced. Redpanda uses a specialized **Partition Balancer** that continuously optimizes placement based on disk/CPU usage.

### Performance Implications
-   **Tiered Storage Latency**: Reading from S3 has high First-Byte Latency (50-100ms). Throughput is high. Not suitable for real-time latency-sensitive apps, but great for catch-up or replay.
""",
    "Day4_Admin_Operations_Tiered_Storage_Interview.md": """# Day 4: Admin Operations - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the benefit of Tiered Storage?**
    -   *A*: Infinite retention, lower cost, and decoupling of compute and storage.

2.  **Q: How do you secure a Kafka/Redpanda cluster?**
    -   *A*: Encryption in transit (TLS), Authentication (SASL/SCRAM/mTLS), and Authorization (ACLs).

3.  **Q: What is a "Preferred Leader"?**
    -   *A*: The replica that was originally assigned as leader. If a broker restarts, leadership moves. "Preferred Leader Election" moves it back to restore balance.

### Production Challenges
-   **Challenge**: **S3 Cost Spike**.
    -   *Scenario*: High API request costs (PUT/GET) from Tiered Storage.
    -   *Fix*: Increase segment size (fewer files = fewer API calls).

-   **Challenge**: **Rebalance stuck**.
    -   *Scenario*: A partition move never completes.
    -   *Cause*: Network issues or disk full on destination.

### Troubleshooting Scenarios
**Scenario**: Consumers are timing out when reading old data.
-   *Cause*: Data is in S3, and the fetch is taking longer than `request.timeout.ms`.
-   *Fix*: Increase consumer timeout or check S3 connectivity.
""",

    # --- Day 5: Advanced Config ---
    "Day5_Advanced_Config_Tuning_Core.md": """# Day 5: Advanced Configuration & Tuning

## Core Concepts & Theory

### Latency vs Throughput
-   **Latency**: Time to deliver one message. (Optimize: `linger.ms=0`, `compression=none`).
-   **Throughput**: Messages per second. (Optimize: `linger.ms=5`, `batch.size=64KB`, `compression=lz4`).
You cannot maximize both simultaneously.

### Network Threads
-   **Kafka**: `num.network.threads`. Handles TCP connections.
-   **Redpanda**: Seastar handles this automatically per core.

### Disk I/O Tuning
-   **Commit Latency**: `log.flush.interval.messages` (fsync).
    -   Kafka defaults to letting OS handle fsync (fast, but risk of data loss on OS crash).
    -   Redpanda defaults to `fsync` on every batch (safe).

### Architectural Reasoning
**The "Zero-Copy" Myth**
Zero-copy is great for plaintext. But if you use TLS (Encryption), the CPU *must* read the data to encrypt it. So Zero-copy is disabled for TLS. Redpanda uses hardware acceleration (AES-NI) to minimize this cost.

### Key Components
-   **linger.ms**: Artificial delay to build batches.
-   **socket.send.buffer.bytes**: TCP buffer size.
-   **compression.type**: Gzip (high CPU), Snappy/LZ4 (fast), Zstd (balanced).
""",
    "Day5_Advanced_Config_Tuning_DeepDive.md": """# Day 5: Tuning - Deep Dive

## Deep Dive & Internals

### Redpanda Tuning (`rpk redpanda tune`)
This command applies OS-level tuning:
-   **Transparent Hugepages (THP)**: Disables it (causes latency spikes).
-   **Swappiness**: Sets to 0.
-   **Clocksource**: Sets to `tsc` (fastest).

### Producer Tuning Checklist
1.  `batch.size`: Increase to 16KB-64KB.
2.  `linger.ms`: Set to 5-10ms.
3.  `compression.type`: `lz4` or `zstd`.
4.  `acks`: `1` for throughput, `all` for safety.

### Consumer Tuning Checklist
1.  `fetch.min.bytes`: Wait for data before returning.
2.  `max.poll.records`: Process more data per poll.
3.  `socket.receive.buffer.bytes`: Increase for high-latency WAN links.

### Performance Implications
-   **Compression**: High compression (Gzip) saves disk/network but burns CPU. If your broker is CPU-bound, switch to LZ4.
""",
    "Day5_Advanced_Config_Tuning_Interview.md": """# Day 5: Tuning - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you optimize for low latency?**
    -   *A*: `linger.ms=0`, `acks=1`, `compression=none`, ensure producers/consumers are close to brokers.

2.  **Q: How do you optimize for high throughput?**
    -   *A*: Batching (`linger.ms > 0`), Compression (`lz4`), Parallelism (more partitions/consumers).

3.  **Q: What is the impact of `fsync`?**
    -   *A*: `fsync` forces data to physical disk. It is slow. Doing it on every message destroys throughput. Kafka relies on replication for safety, not fsync.

### Production Challenges
-   **Challenge**: **High Tail Latency (p99)**.
    -   *Cause*: GC pauses (Kafka), noisy neighbors, or slow disk.
    -   *Fix*: Tune GC, isolate resources, upgrade disks.

-   **Challenge**: **Network Saturation**.
    -   *Scenario*: 10Gbps link is full.
    -   *Fix*: Enable compression (`zstd`), scale out brokers.

### Troubleshooting Scenarios
**Scenario**: Producer throughput is low, but CPU/Disk are idle.
-   *Cause*: `linger.ms` might be too high, or `max.in.flight.requests` is 1 (stop-and-wait).
-   *Fix*: Check client config.
"""
}

print("🚀 Populating Week 2 Redpanda Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 2 Content Population Complete!")
