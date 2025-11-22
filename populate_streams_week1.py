import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase1_Foundations\Week1_Kafka_Architecture"

content_map = {
    "Day1_Intro_Log_Abstraction_DeepDive.md": """# Day 1: The Log Abstraction - Deep Dive

## Deep Dive & Internals

### The Log as the Source of Truth
In a log-centric architecture, the **Log** is the single source of truth.
-   **Database as a Derived View**: A database table is just a cached view of the log. If you lose the database, you can rebuild it by replaying the log.
-   **Dual Writes Problem**: Writing to a DB and publishing an event is hard (two-phase commit?). With the Log, you write to the Log first, and the DB updates by consuming the Log (Change Data Capture).

### Sequential I/O vs Random I/O
The "Log" abstraction exploits the physics of disk drives.
-   **Random I/O**: Slow (seeking). 100-200 IOPS on HDD.
-   **Sequential I/O**: Fast. 100MB/s+ on HDD.
Kafka is designed to be "disk-bound" but acts like "network-bound" because disk sequential writes are so fast they often saturate the network before the disk.

### Zero-Copy Optimization
Kafka uses the Linux `sendfile` system call (Java `FileChannel.transferTo`).
-   **Traditional**: Disk -> Kernel Buffer -> User Space -> Kernel Socket Buffer -> NIC. (4 copies, 4 context switches).
-   **Zero-Copy**: Disk -> Kernel Buffer -> NIC. (2 copies, 2 context switches).
This is why Kafka can handle millions of messages per second with very low CPU usage.

### Advanced Reasoning
**Why not a Message Queue (RabbitMQ)?**
-   **RabbitMQ**: Smart broker, dumb consumer. Broker tracks state. Messages are removed after consumption. Good for complex routing.
-   **Kafka**: Dumb broker, smart consumer. Log is persisted. Good for high throughput, replayability, and massive scale.
""",
    "Day1_Intro_Log_Abstraction_Interview.md": """# Day 1: The Log Abstraction - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between a Queue and a Log?**
    -   *A*: A Queue (like RabbitMQ) is transient; messages are deleted once consumed. A Log (like Kafka) is durable; messages are persisted and can be replayed.

2.  **Q: Why is sequential I/O faster than random I/O?**
    -   *A*: Sequential I/O avoids disk seek time (moving the head) and rotational latency. It allows the OS to perform aggressive read-ahead (prefetching).

3.  **Q: Explain the concept of "Log Compaction".**
    -   *A*: Instead of deleting old logs by time, Kafka keeps the *latest* value for each key. This effectively turns the log into a database table snapshot.

### Production Challenges
-   **Challenge**: **Disk Full**.
    -   *Scenario*: Producers write faster than retention policy deletes old segments.
    -   *Fix*: Monitor disk usage. Use tiered storage (S3) to offload old data.

-   **Challenge**: **Slow Consumers**.
    -   *Scenario*: A consumer falls behind and data is deleted before it can be read.
    -   *Fix*: Increase retention time or fix the consumer performance.

### Troubleshooting Scenarios
**Scenario**: You see high I/O wait times on your broker.
-   **Check**: Are you doing random reads? (Consumers reading very old data that is not in page cache).
-   **Check**: Is the disk saturated? (Use `iostat`).
""",
    "Day2_Kafka_Architecture_Deep_Dive_DeepDive.md": """# Day 2: Kafka Architecture - Deep Dive

## Deep Dive & Internals

### The Controller's Role
The **Controller** is the "Brain".
-   It monitors broker liveness (via Zookeeper/KRaft heartbeats).
-   It elects partition leaders.
-   It tells other brokers: "You are now the leader for Partition 0."
-   **Split Brain**: If two brokers think they are the controller, chaos ensues. Generation IDs (Epochs) prevent this.

### KRaft (Kafka Raft) Internals
KRaft removes Zookeeper.
-   **Metadata Topic**: `@metadata` is an internal topic that stores the cluster state.
-   **Quorum**: A subset of brokers (Voters) participate in Raft consensus to elect the active Controller.
-   **Snapshotting**: To prevent the metadata log from growing forever, snapshots of the state are taken.

### Request Purgatory
When a request cannot be satisfied immediately (e.g., `acks=all` but replicas haven't acked yet), it is placed in the **Purgatory**.
-   It is a hierarchical timing wheel.
-   Requests sit there until the condition is met or they timeout.

### Advanced Reasoning
**Why is Zookeeper removal a big deal?**
-   **Scalability**: Zookeeper was a bottleneck for clusters with millions of partitions. KRaft scales much better.
-   **Simplicity**: One less distributed system to manage.
-   **Atomic Metadata**: Metadata updates are now atomic log appends.

### Performance Implications
-   **Controller Failover**: In Zookeeper, failover could take seconds/minutes (loading metadata). In KRaft, the standby controller already has the metadata in memory, so failover is near-instant.
""",
    "Day2_Kafka_Architecture_Deep_Dive_Interview.md": """# Day 2: Kafka Architecture - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the role of the Controller in Kafka?**
    -   *A*: It manages partition states, elects leaders, and handles broker failures.

2.  **Q: How does Kafka handle Split Brain?**
    -   *A*: It uses **Controller Epochs** (Generation IDs). If a broker receives a command from a controller with an older epoch, it ignores it.

3.  **Q: What is KRaft and why is it better than Zookeeper?**
    -   *A*: KRaft is Kafka's internal consensus mechanism. It removes the external Zookeeper dependency, improves scalability (millions of partitions), and simplifies operations.

### Production Challenges
-   **Challenge**: **Controller Failover Slowness** (Legacy Zookeeper).
    -   *Scenario*: Controller dies, new one takes 30s to load metadata. Cluster is unavailable for writes during this time.
    -   *Fix*: Upgrade to KRaft.

-   **Challenge**: **Metadata inconsistent**.
    -   *Scenario*: A broker thinks it's the leader, but the controller thinks otherwise.
    -   *Fix*: Usually caused by Zookeeper/Broker desync. Restarting the broker often fixes it.

### Troubleshooting Scenarios
**Scenario**: "Not Leader for Partition" errors.
-   **Cause**: The client has stale metadata. It's trying to write to a broker that is no longer the leader.
-   **Fix**: The client usually refreshes metadata automatically. If persistent, check network connectivity to the Controller.
""",
    "Day3_Topics_Partitions_Segments_DeepDive.md": """# Day 3: Topics, Partitions, Segments - Deep Dive

## Deep Dive & Internals

### Segment File Internals
A segment consists of:
1.  `.log`: The actual messages.
2.  `.index`: Maps **Offset -> Physical Position** (Byte offset in .log file).
3.  `.timeindex`: Maps **Timestamp -> Offset**.

### Indexing Strategy
Kafka indexes are **sparse**. It doesn't index every message.
-   It might index every 4KB of data.
-   To find offset 100:
    1.  Look in `.index` for the largest offset <= 100 (say, 90).
    2.  Jump to the physical position of 90 in `.log`.
    3.  Scan forward linearly until you find 100.
-   **Trade-off**: Saves RAM (index fits in memory) at the cost of a tiny linear scan.

### Log Compaction Details
-   **Cleaner Thread**: Background thread that compacts logs.
-   **Head vs Tail**:
    -   **Tail**: Cleaned portion. Contains only unique keys.
    -   **Head**: Active portion. Contains duplicates.
-   **Tombstones**: A message with `Value=null` is a delete marker. It removes the key from the log eventually.

### Advanced Reasoning
**Why not one file per partition?**
-   **File Size**: A single file is hard to manage (delete old data). Segments allow deleting old data by simply deleting the oldest file (`rm segment-000.log`).
-   **OS Limits**: Filesystems handle smaller files better than one multi-TB file.

### Performance Implications
-   **Open File Descriptors**: Each segment uses FDs. Too many partitions * too many segments = `Too many open files` error.
""",
    "Day3_Topics_Partitions_Segments_Interview.md": """# Day 3: Topics, Partitions, Segments - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does Kafka find a message by offset?**
    -   *A*: It uses the sparse `.index` file to find the nearest physical position, then scans forward.

2.  **Q: What is Log Compaction used for?**
    -   *A*: It's used for restoring state (e.g., KTable, database CDC). We only care about the *latest* state of a key, not the history.

3.  **Q: What happens if you have too many partitions?**
    -   *A*: High unavailability during failover (leader election takes time), high memory usage (metadata), and high open file handles.

### Production Challenges
-   **Challenge**: **"Too many open files"**.
    -   *Scenario*: Broker crashes because it hit the OS limit on file descriptors.
    -   *Fix*: Increase `ulimit -n`. Reduce retention period or increase segment size.

-   **Challenge**: **Unbalanced Partitions**.
    -   *Scenario*: One partition is 1TB, others are 1GB.
    -   *Cause*: Poor partition key choice (Data Skew).
    -   *Fix*: Fix the producer's partitioning logic.

### Troubleshooting Scenarios
**Scenario**: Disk usage is not going down despite low retention.
-   **Check**: Is `log.cleanup.policy=compact`? Compaction doesn't delete old data by time immediately; it waits for the cleaner thread.
-   **Check**: Are there "stray" partitions that are not being deleted?
""",
    "Day4_Producers_Consumers_DeepDive.md": """# Day 4: Producers & Consumers - Deep Dive

## Deep Dive & Internals

### Producer Internals
1.  **`send()`**: Adds record to a buffer (accumulator).
2.  **Sender Thread**: Background thread that drains the buffer.
3.  **Batching**: Group records by partition.
    -   `batch.size`: Max bytes per batch (e.g., 16KB).
    -   `linger.ms`: Max time to wait to fill a batch (e.g., 5ms).
    -   **Trade-off**: High `linger.ms` = High Throughput, Higher Latency.

### Consumer Group Protocol
1.  **JoinGroup**: Consumers send "I want to join" to the **Group Coordinator** (a specific broker).
2.  **SyncGroup**: The Coordinator elects a **Leader Consumer**. The Leader decides the partition assignment and sends it back to the Coordinator.
3.  **Heartbeats**: Consumers send heartbeats. If missed, they are kicked out -> Rebalance.

### Sticky Assignor
-   **Range/RoundRobin**: Reassigns everything from scratch. High churn.
-   **Sticky**: Tries to keep existing assignments. Only moves what's necessary. Reduces rebalance "stop-the-world" time.

### Advanced Reasoning
**Why Client-Side Partitioning?**
The producer decides the partition. This avoids double-hop (Producer -> Broker A -> Broker B). The producer sends directly to the leader of the correct partition.

### Performance Implications
-   **Compression**: Batches are compressed (Snappy, LZ4). Better compression with larger batches.
-   **Fetch Size**: Consumers fetch bytes, not messages. `fetch.min.bytes` allows the broker to wait until it has enough data (efficiency).
""",
    "Day4_Producers_Consumers_Interview.md": """# Day 4: Producers & Consumers - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you ensure strict ordering of messages?**
    -   *A*: Ensure all messages go to the **same partition** (use the same Key). Set `max.in.flight.requests.per.connection=1` (or 5 if idempotence is on).

2.  **Q: What triggers a Consumer Rebalance?**
    -   *A*: A consumer joining/leaving, a consumer crashing (missed heartbeat), or a topic partition count changing.

3.  **Q: What is the difference between `at-least-once` and `exactly-once`?**
    -   *A*: `at-least-once`: Retries on failure, duplicates possible. `exactly-once`: Transactional guarantees, no duplicates.

### Production Challenges
-   **Challenge**: **Rebalance Storm**.
    -   *Scenario*: Consumers keep joining and leaving in a loop.
    -   *Cause*: Processing takes too long (`max.poll.interval.ms` exceeded). The broker thinks the consumer is dead.
    -   *Fix*: Increase `max.poll.interval.ms` or optimize processing logic.

-   **Challenge**: **Poison Pill**.
    -   *Scenario*: A message crashes the consumer. Consumer restarts, reads same message, crashes again.
    -   *Fix*: Dead Letter Queue (DLQ) + Error handling.

### Troubleshooting Scenarios
**Scenario**: Consumer Lag is increasing.
-   **Check**: Is the processing logic too slow?
-   **Check**: Do you need more consumers (and partitions)?
-   **Check**: Is there a rebalance loop?
""",
    "Day5_Reliability_Durability_DeepDive.md": """# Day 5: Reliability & Durability - Deep Dive

## Deep Dive & Internals

### The High Watermark (HW)
The HW is the offset of the *last message that was successfully replicated to all ISRs*.
-   **Consumers only see up to HW**. They cannot read uncommitted data.
-   This prevents "phantom reads" (reading data that is later lost due to leader failure).

### Leader Epochs
Used to prevent data loss during tricky failure scenarios (like a follower becoming leader, then the old leader coming back with divergent logs).
-   **Epoch**: A monotonic counter increased every time a new leader is elected.
-   Brokers truncate their logs to the point where they diverged from the *current* leader's epoch.

### Idempotent Producer
-   **Sequence Numbers**: Each message has a (ProducerID, SequenceNumber).
-   **De-duplication**: The broker keeps track of the last SequenceNumber. If it receives a duplicate (due to network retry), it silently drops it and acks.
-   **Overhead**: Negligible. Always enable it (`enable.idempotence=true`).

### Advanced Reasoning
**Why not synchronous replication to ALL replicas?**
-   If you have 3 replicas and wait for ALL 3, then 1 slow/dead node stops the whole cluster.
-   **ISR** is the compromise: Wait only for the *healthy* ones. If a node is slow, kick it out of ISR so it doesn't block writes.

### Performance Implications
-   `acks=all`: Increases latency (must wait for network round-trip to followers).
-   `compression`: Reduces network bandwidth and disk usage, improving effective durability (faster replication).
""",
    "Day5_Reliability_Durability_Interview.md": """# Day 5: Reliability & Durability - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if `min.insync.replicas=2` and you only have 1 ISR?**
    -   *A*: The producer receives a `NOT_ENOUGH_REPLICAS` exception. The partition becomes read-only (effectively).

2.  **Q: How does Kafka prevent data loss?**
    -   *A*: Replication, `acks=all`, `min.insync.replicas > 1`, and `unclean.leader.election.enable=false`.

3.  **Q: What is an "Unclean Leader Election"?**
    -   *A*: Electing a replica that was NOT in the ISR (i.e., it is missing data). It restores availability but causes data loss. Default is `false`.

### Production Challenges
-   **Challenge**: **Data Loss**.
    -   *Scenario*: `acks=1`, Leader crashes before replicating to follower.
    -   *Fix*: Use `acks=all`.

-   **Challenge**: **Cluster unavailable for writes**.
    -   *Scenario*: 2 out of 3 brokers crash. `min.insync.replicas=2`.
    -   *Fix*: Bring brokers back up. Or dynamically lower `min.insync.replicas` (risky).

### Troubleshooting Scenarios
**Scenario**: Producer latency spikes.
-   **Check**: Is one of the followers slow? (Slow disk/network). It might be dragging down the `acks=all` latency.
-   **Check**: Check `UnderReplicatedPartitions` metric.
"""
}

print("🚀 Populating Week 1 DeepDive & Interview Content...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 1 Content Population Complete!")
