import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase4_Production_CaseStudies\Week8_Observability_Operations"

content_map = {
    # --- Day 1: Monitoring Kafka & Redpanda ---
    "Day1_Monitoring_Kafka_Redpanda_Core.md": """# Day 1: Monitoring Kafka & Redpanda

## Core Concepts & Theory

### The Observability Pillars
Observability is more than just "monitoring". It's about understanding the internal state of the system based on its external outputs.
1.  **Metrics**: Aggregatable data (counters, gauges, histograms). "What is happening?"
2.  **Logs**: Discrete events. "Why did it happen?"
3.  **Traces**: Request lifecycle. "Where did it happen?"

### Key Kafka Metrics
To operate Kafka/Redpanda at scale, you must monitor these 4 Golden Signals:

#### 1. Latency
-   `RequestQueueTimeMs`: Time waiting in the request queue. High values = Broker overloaded.
-   `LocalTimeMs`: Time processing the request (writing to disk). High values = Slow disk.
-   `RemoteTimeMs`: Time waiting for follower replication. High values = Slow network/follower.
-   **Total Request Latency** = Queue + Local + Remote.

#### 2. Traffic (Throughput)
-   `BytesInPerSec` / `BytesOutPerSec`: Network saturation.
-   `MessagesInPerSec`: CPU load indicator.

#### 3. Errors
-   `OfflinePartitionsCount`: **CRITICAL**. Partitions with no leader. Data unavailable.
-   `UnderReplicatedPartitions`: **CRITICAL**. Replicas falling behind. Risk of data loss.
-   `ActiveControllerCount`: Should be exactly 1. If 0, cluster is brainless. If >1, split brain.

#### 4. Saturation
-   **Disk Usage**: Alert at 80%. Kafka stops accepting writes at 95% (usually).
-   **CPU Usage**: High CPU is fine, but look for Thread Pool usage (`NetworkProcessorAvgIdlePercent`). If < 0.3 (30%), you need more network threads.

### Redpanda Specifics
Redpanda uses a thread-per-core architecture.
-   **Reactor Utilization**: The most important metric. If > 90%, the core is saturated.
-   **IO Queue**: If high, disk cannot keep up.

### Architectural Reasoning
**Why "Under Replicated" is the Holy Grail Metric**
It captures almost all failures:
-   Broker down? -> Under Replicated.
-   Network slow? -> Under Replicated.
-   Disk slow? -> Under Replicated.
-   GC pause? -> Under Replicated.
If you only alert on one thing, make it this.
""",

    "Day1_Monitoring_Kafka_Redpanda_DeepDive.md": """# Day 1: Monitoring Kafka - Deep Dive

## Deep Dive & Internals

### JMX vs Native Metrics
-   **Kafka (JVM)**: Uses JMX (Java Management Extensions). Heavy. Requires `jmx_exporter` sidecar/agent to convert to Prometheus format.
    -   *Overhead*: JMX scraping can be CPU intensive. Don't scrape too often (< 15s).
-   **Redpanda (C++)**: Native Prometheus endpoint (`:9644/metrics`). Zero overhead.

### Histogram Pitfalls
Kafka's JMX histograms (e.g., `99thPercentile`) are calculated *inside* the broker using a decaying reservoir.
-   **Pros**: Pre-calculated.
-   **Cons**: Can be misleading if traffic is bursty.
-   **Best Practice**: Export raw histograms (buckets) to Prometheus and calculate percentiles there (`histogram_quantile`).

### Consumer Lag Monitoring
Lag is the difference between `LogEndOffset` (Broker) and `CurrentOffset` (Consumer).
-   **Internal**: Consumers send offset commits to `__consumer_offsets`.
-   **External Monitoring (Burrow)**: Look at the *rate* of commits vs *rate* of production.
    -   If `ProductionRate > ConsumptionRate`, Lag grows.
    -   If `Lag` is stable but high, it's just latency.
    -   If `Lag` is growing, it's an incident.

### Advanced Reasoning
**The "Stale Metric" Trap**
If a broker crashes, the Prometheus exporter might stop sending data.
-   If you alert on `UnderReplicatedPartitions > 0`, and the metric *disappears*, the alert resolves!
-   **Fix**: Always alert on `up == 0` (Target down) or `absent(metric)`.

### Performance Implications
-   **Cardinality**: Kafka creates metrics *per partition*.
    -   100 topics * 50 partitions = 5,000 metrics.
    -   Multiply by 10 metric types = 50,000 time series.
    -   **Fix**: Use JMX Exporter regex to aggregate metrics (remove `partition` label) unless debugging.
""",

    "Day1_Monitoring_Kafka_Redpanda_Interview.md": """# Day 1: Monitoring - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you distinguish between a slow consumer and a stuck consumer?**
    -   *A*: **Slow Consumer**: Offsets are advancing, but Lag is increasing (Production > Consumption). **Stuck Consumer**: Offsets are NOT advancing at all.

2.  **Q: Why is `NetworkProcessorAvgIdlePercent` important?**
    -   *A*: It measures how much time network threads are idle. If it's 0%, the broker is network-bound (cannot accept new connections/requests), even if CPU is low.

3.  **Q: What is the impact of scraping JMX metrics too frequently?**
    -   *A*: JMX is a blocking operation in some JVM versions. High scrape frequency can cause GC pauses or slow down the broker.

### Production Challenges
-   **Challenge**: **Metric Explosion in Prometheus**.
    -   *Scenario*: Dev team creates 10,000 dynamic topics. Prometheus OOMs.
    -   *Fix*: Whitelist specific metrics in JMX Exporter. Drop partition-level metrics for non-critical topics.

-   **Challenge**: **False Positives on Lag**.
    -   *Scenario*: Batch job runs every hour. Lag spikes to 1M, then drops. Alert wakes you up.
    -   *Fix*: Alert on *Lag Duration* (Lag > 1M for > 10 mins) or *Derivative* (Lag is increasing fast).

### Troubleshooting Scenarios
**Scenario**: Broker is up, but `OfflinePartitions` > 0.
-   *Cause*: The disk containing those partitions might be corrupted or read-only (I/O error).
-   *Fix*: Check `server.log` for I/O errors. Replace disk.
""",

    # --- Day 2: Monitoring Flink ---
    "Day2_Monitoring_Flink_Core.md": """# Day 2: Monitoring Flink

## Core Concepts & Theory

### The Flink Metric System
Flink has a pluggable metric system.
-   **System Metrics**: CPU, Memory, GC, Threads (from JVM).
-   **Flink Metrics**: Checkpointing, Restart/Failover, Network Buffers.
-   **User Metrics**: Counters, Gauges, Histograms defined in your code.

### Critical Flink Metrics
1.  **Availability**
    -   `uptime`: Time since last restart.
    -   `numRestarts`: If increasing, the job is unstable.
    -   `fullRestarts`: JobManager failure.

2.  **Throughput & Latency**
    -   `numRecordsInPerSecond` / `numRecordsOutPerSecond`.
    -   `latency`: End-to-end latency (requires Latency Markers, expensive!).

3.  **Backpressure**
    -   `outPoolUsage`: If 100%, the downstream task is slow.
    -   `isBackPressured`: Boolean indicator (newer Flink versions).

4.  **Checkpointing**
    -   `lastCheckpointDuration`: If increasing, state is growing or storage is slow.
    -   `lastCheckpointSize`: Size of the state.

### Architectural Reasoning
**Why not use Latency Markers?**
Flink can inject "Latency Markers" that travel with records.
-   **Pros**: Exact per-operator latency.
-   **Cons**: modifying the stream, adds overhead, skews throughput.
-   **Alternative**: Measure "Business Latency" (Event Time - Processing Time) using a Histogram in the Sink.

### Key Components
-   **MetricReporter**: Interface to send metrics (Prometheus, Datadog, StatsD).
-   **Flink Web UI**: Real-time dashboard (good for debugging, not for alerting).
""",

    "Day2_Monitoring_Flink_DeepDive.md": """# Day 2: Monitoring Flink - Deep Dive

## Deep Dive & Internals

### Backpressure Monitoring
How does Flink know it's backpressured?
1.  **Old Way (Stack Trace Sampling)**: JobManager periodically triggers thread dumps on TaskManagers to see if they are stuck in `requestBuffer`. High overhead.
2.  **New Way (Credit-Based)**: TaskManagers report `outPoolUsage` (how many output buffers are full). If buffers are full, it cannot send data -> Backpressure. Zero overhead.

### Checkpoint Monitoring
Checkpoint metrics reveal the health of your state backend.
-   **Sync Duration**: Time to snapshot state in memory. If high -> CPU bottleneck or huge state object.
-   **Async Duration**: Time to upload to S3/DFS. If high -> Network/Storage bottleneck.
-   **Alignment Time**: Time waiting for barriers. If high -> Skew or Backpressure.

### Memory Monitoring
Flink manages its own off-heap memory (Managed Memory).
-   **Heap**: User code objects.
-   **Off-Heap**: Network buffers, RocksDB native memory.
-   **Metaspace**: Class metadata.
**OOM Debugging**:
-   `Heap Space OOM`: User code memory leak.
-   `Direct Buffer OOM`: Network buffer leak or RocksDB growing too large.

### Performance Implications
-   **User Metrics**: Don't create a new `Counter` for every user ID (Cardinality explosion!).
-   **Tagging**: Flink adds tags (`job_name`, `task_name`, `subtask_index`). Ensure your TSDB supports high cardinality if you have many jobs.
""",

    "Day2_Monitoring_Flink_Interview.md": """# Day 2: Monitoring Flink - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you detect data skew using metrics?**
    -   *A*: Compare `numRecordsIn` across subtasks. If Subtask 0 has 1M records and Subtask 1 has 100 records, you have skew.

2.  **Q: What does it mean if `lastCheckpointDuration` is close to `checkpointInterval`?**
    -   *A*: Dangerous. The job spends all its time checkpointing. If it exceeds the interval, checkpoints will start failing or skipping.

3.  **Q: How do you monitor RocksDB specifically?**
    -   *A*: Enable `state.backend.rocksdb.metrics.enable`. Monitor `block-cache-usage`, `memtable-size`, and `estimate-num-keys`.

### Production Challenges
-   **Challenge**: **"Silent" Failure**.
    -   *Scenario*: Job is running (RUNNING status), but processing 0 records.
    -   *Fix*: Alert on `numRecordsOutPerSecond == 0` for > 5 mins.

-   **Challenge**: **GC Pauses**.
    -   *Scenario*: Job pauses for 10s every minute. Throughput drops.
    -   *Fix*: Monitor `GarbageCollector.CollectionTime`. Tune JVM (G1GC), increase Heap, or reduce object creation rate.

### Troubleshooting Scenarios
**Scenario**: `CheckpointExpiredException`.
-   *Cause*: Checkpoint took longer than timeout (10 mins).
-   *Fix*: Check `AsyncDuration` (Storage slow?) and `AlignmentTime` (Backpressure?). Increase timeout or optimize state.
""",

    # --- Day 3: Alerting & SLOs ---
    "Day3_Alerting_SLOs_Core.md": """# Day 3: Alerting & SLOs

## Core Concepts & Theory

### The Hierarchy of Reliability
1.  **SLI (Service Level Indicator)**: The metric. "Latency is 50ms".
2.  **SLO (Service Level Objective)**: The goal. "Latency should be < 100ms for 99% of requests".
3.  **SLA (Service Level Agreement)**: The contract. "If we miss the SLO, we pay you back".

### Designing Good Alerts
-   **Page**: Immediate action required. User is impacted. (e.g., "Site Down", "Data Loss").
-   **Ticket**: Action required eventually. (e.g., "Disk 80% full").
-   **Log**: No action required. (e.g., "Job restarted successfully").

### Error Budgets
The allowed amount of unreliability.
-   SLO: 99.9% Availability.
-   Error Budget: 0.1% (43 minutes / month).
-   **Philosophy**: If you have budget left, you can ship risky features. If budget is exhausted, freeze changes and focus on stability.

### Architectural Reasoning
**Symptom-Based vs Cause-Based Alerting**
-   **Cause-Based**: "CPU is high". (Bad. High CPU might be fine).
-   **Symptom-Based**: "Latency is high". (Good. User is suffering).
-   **Rule**: Page on Symptoms. Use Causes for debugging.

### Key Components
-   **Prometheus Alertmanager**: Deduplicates, groups, and routes alerts (Slack, PagerDuty).
-   **Grafana Alerting**: Visual alerting.
""",

    "Day3_Alerting_SLOs_DeepDive.md": """# Day 3: Alerting - Deep Dive

## Deep Dive & Internals

### Multi-Window Multi-Burn-Rate Alerts
How to alert fast enough to save the SLO, but not so fast it's noisy?
-   **Burn Rate**: How fast are we consuming the error budget?
    -   Burn Rate 1 = Consuming budget exactly at the limit (will exhaust in 30 days).
    -   Burn Rate 14.4 = Will exhaust in 2 days.
-   **Strategy**:
    -   **Page**: High Burn Rate (14.4) over Short Window (1h) AND Long Window (6h).
    -   **Ticket**: Low Burn Rate (6) over Long Window (3 days).

### Alert Grouping & Inhibition
-   **Grouping**: If 100 brokers go down, don't send 100 pages. Send 1 page: "Cluster Down (100 brokers)".
-   **Inhibition**: If "Data Center Power Outage" is firing, suppress "Server Down" alerts.

### Advanced Reasoning
**The "Null" Alert**
What if the monitoring system itself is down?
-   **Dead Man's Switch**: Have an external system (e.g., PagerDuty or a Lambda) expect a "heartbeat" from Alertmanager every minute. If missing, Page.

### Performance Implications
-   **Evaluation Interval**: How often to check rules. 1m is standard. 10s is aggressive.
-   **State**: Alertmanager stores active alerts in memory/disk. High churn alerts can overload it.
""",

    "Day3_Alerting_SLOs_Interview.md": """# Day 3: Alerting - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Availability and Reliability?**
    -   *A*: **Availability**: Is the system up? (Uptime). **Reliability**: Is the system doing the right thing? (Correctness, no data loss).

2.  **Q: How do you calculate availability for a streaming pipeline?**
    -   *A*: Harder than request/response. Usually defined as "Freshness" (Data is available within X seconds) or "Completeness" (No data loss).

3.  **Q: Describe an alerting strategy for a Kafka cluster.**
    -   *A*: Page on: UnderReplicatedPartitions > 0, OfflinePartitions > 0, ConsumerLag > Threshold (for critical apps). Ticket on: Disk > 80%, Broker skewed.

### Production Challenges
-   **Challenge**: **Alert Flapping**.
    -   *Scenario*: CPU goes 91% -> 89% -> 91%. Alert fires, resolves, fires.
    -   *Fix*: Use **Hysteresis** (Fire at 90%, Resolve at 80%) or `for: 5m` duration.

-   **Challenge**: **On-Call Burnout**.
    -   *Scenario*: Too many non-actionable alerts.
    -   *Fix*: Delete any alert that doesn't require immediate action. Move to tickets/dashboards.

### Troubleshooting Scenarios
**Scenario**: Alertmanager is sending emails but not Slack messages.
-   *Cause*: Slack Webhook URL expired or network firewall blocking egress.
-   *Fix*: Check Alertmanager logs.
""",

    # --- Day 4: Troubleshooting ---
    "Day4_Troubleshooting_Issues_Core.md": """# Day 4: Troubleshooting Streaming Systems

## Core Concepts & Theory

### The Troubleshooting Loop
1.  **Detect**: Alert fires or user complains.
2.  **Isolate**: Is it the Source (Kafka), the Processor (Flink), or the Sink (DB)?
3.  **Mitigate**: Stop the bleeding (Scale up, restart, rollback).
4.  **Diagnose**: Find root cause (Logs, Metrics, Traces).
5.  **Fix**: Apply permanent fix.

### Common Failure Patterns
1.  **The "Poison Pill"**: A message that crashes the consumer.
    -   *Symptom*: Consumer restarts loop.
    -   *Fix*: Dead Letter Queue (DLQ).
2.  **The "Data Skew"**: One partition has 90% of data.
    -   *Symptom*: One task is 100% CPU, others idle. Backpressure.
    -   *Fix*: Re-key (Salt) or local aggregation.
3.  **The "Death Spiral"**: System slows down -> Retries increase -> Load increases -> System slows down more.
    -   *Fix*: Circuit Breakers, Exponential Backoff.

### Architectural Reasoning
**Fail Fast vs Fail Safe**
-   **Fail Fast**: Crash immediately on error. Good for data integrity.
-   **Fail Safe**: Drop the bad record and continue. Good for availability.
-   **Stream Processing**: Usually prefers Fail Fast (to prevent data loss), but with DLQ for bad data.

### Key Components
-   **Logs**: `server.log`, `jobmanager.log`.
-   **Thread Dumps**: `jstack`. Crucial for "stuck" processes.
-   **Heap Dumps**: `jmap`. Crucial for Memory Leaks.
""",

    "Day4_Troubleshooting_Issues_DeepDive.md": """# Day 4: Troubleshooting - Deep Dive

## Deep Dive & Internals

### Debugging Network Issues
Kafka/Flink are distributed. Network is often the culprit.
-   **TCP Retransmits**: High retransmits = Bad cable/switch.
-   **DNS Latency**: Java caches DNS. If IP changes, app might fail.
-   **Bandwidth Saturation**: Check `sar -n DEV`.

### Analyzing Thread Dumps
When a process is "stuck" (not processing, but CPU low):
1.  Take thread dump (`jstack <pid>`).
2.  Look for `BLOCKED` or `WAITING` threads.
3.  **Common Culprits**:
    -   Waiting on Lock (Deadlock).
    -   Waiting on I/O (Socket read).
    -   Waiting on external service (HTTP call without timeout).

### Analyzing Heap Dumps
When `OutOfMemoryError` occurs:
1.  Take heap dump (`jmap -dump:format=b,file=heap.bin <pid>`).
2.  Open in **Eclipse MAT** or **VisualVM**.
3.  Look at "Dominator Tree".
4.  **Common Culprits**:
    -   Huge `HashMap` (Caching without TTL).
    -   Large `byte[]` (Accumulating payloads).
    -   Flink State objects (RocksDB JNI objects).

### Advanced Reasoning
**Heisenbugs**
Bugs that disappear when you try to study them (e.g., enabling debug logging slows down the system enough to hide a race condition).
-   **Fix**: Distributed Tracing and high-resolution metrics are better than logs for race conditions.

### Performance Implications
-   **Logging Overhead**: `DEBUG` logging in a hot loop (processElement) can reduce throughput by 100x. Use `isDebugEnabled()` check.
""",

    "Day4_Troubleshooting_Issues_Interview.md": """# Day 4: Troubleshooting - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: A Kafka consumer is lagging. How do you debug it?**
    -   *A*: Check: Is the consumer CPU bound? (Scale up). Is it I/O bound? (Check DB/Sink). Is it rebalancing? (Check logs). Is there skew? (Check partition lag).

2.  **Q: What is a Zombie Process in Flink?**
    -   *A*: A TaskManager that lost connection to JobManager but is still running. It might write duplicate data to the sink. Fencing (Epochs) prevents this.

3.  **Q: How do you handle a Schema Mismatch in production?**
    -   *A*: The consumer fails to deserialize. Stop the job. Update the schema (if backward compatible) or route bad messages to DLQ.

### Production Challenges
-   **Challenge**: **Cascading Failure**.
    -   *Scenario*: DB slows down -> Flink backpressures -> Kafka fills up -> Disk full -> Broker crash.
    -   *Fix*: Backpressure is good! It protects the DB. But monitor disk space and set retention policies.

-   **Challenge**: **Slow Memory Leak**.
    -   *Scenario*: App crashes every 3 days.
    -   *Fix*: Monitor Heap usage trend. It will look like a "sawtooth" that gradually rises. Take heap dump before crash.

### Troubleshooting Scenarios
**Scenario**: Flink job stuck in `CREATED` state.
-   *Cause*: Not enough slots (Resources) in the cluster.
-   *Fix*: Scale up cluster or cancel other jobs.
""",

    # --- Day 5: Capacity Planning ---
    "Day5_Capacity_Planning_Core.md": """# Day 5: Capacity Planning & Sizing

## Core Concepts & Theory

### The Sizing Equation
Capacity Planning is math, not magic.
**Inputs**:
1.  **Throughput**: 100 MB/sec (Peak).
2.  **Retention**: 7 Days.
3.  **Replication**: 3x.

**Outputs**:
1.  **Storage**: `100MB * 86400s * 7days * 3replicas = 181 TB`.
2.  **Bandwidth**: `100MB (In) + 300MB (Replication) + 100MB (Consumer) = 500 MB/sec`.
3.  **CPU**: Compression cost + Serialization cost.

### Headroom
Always provision for **Peak** traffic + **Headroom** (e.g., 30%).
-   **Why?**: To handle catch-up bursts. If you size exactly for peak, you can never catch up after an outage.

### Architectural Reasoning
**Disk I/O: The Bottleneck**
Kafka is usually Disk I/O bound or Network bound.
-   **IOPS**: Not important for Kafka (Sequential I/O).
-   **Throughput**: Very important. NVMe SSDs are preferred.
-   **HDD**: Can be used for "Cold" storage (Tiered Storage), but risky for active segments.

### Key Components
-   **Broker Count**: `Max(StorageReq / DiskPerBroker, BandwidthReq / NetworkPerBroker)`.
-   **Partition Count**: `Throughput / MaxThroughputPerPartition`.
""",

    "Day5_Capacity_Planning_DeepDive.md": """# Day 5: Capacity Planning - Deep Dive

## Deep Dive & Internals

### Network Bandwidth Planning
Network is often the hidden bottleneck.
-   **Ingress**: Producer traffic.
-   **Replication**: Ingress * (ReplicationFactor - 1).
-   **Egress**: Consumer traffic. (Ingress * NumberOfConsumers).
-   **Total**: Ingress + Replication + Egress.
**Example**: 100MB In, 3x Rep, 2 Consumers.
-   Total = 100 + 200 + 200 = 500 MB/sec.
-   Ensure NIC (Network Interface Card) supports this (e.g., 10Gbps = 1.25 GB/sec).

### CPU Sizing: Compression
Compression (Zstd/LZ4) saves disk/network but burns CPU.
-   **Producer**: Compresses batch.
-   **Broker**: No CPU cost (Zero Copy) *unless* message format conversion is needed.
-   **Consumer**: Decompresses.
**Warning**: If Broker and Client versions mismatch, Broker must down-convert, burning massive CPU.

### Partition Sizing
-   **Too Few**: Limited parallelism.
-   **Too Many**: High unavailability during leader election (Controller overload). High memory overhead.
-   **Rule of Thumb**: < 4000 partitions per broker. < 200,000 per cluster.

### Advanced Reasoning
**OS Page Cache**
Kafka relies heavily on Linux Page Cache.
-   **RAM Sizing**: You don't need massive Heap (4-6GB is enough). You need massive **Free RAM** for Page Cache.
-   If Consumers are fast, they read from Page Cache (RAM) -> Zero Disk Read.
-   If Consumers lag, they read from Disk -> Slow.

### Performance Implications
-   **RAID**: RAID 10 is good for performance/redundancy. JBOD (Just a Bunch of Disks) is supported by Kafka (software redundancy) and preferred for cost.
""",

    "Day5_Capacity_Planning_Interview.md": """# Day 5: Capacity Planning - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How many partitions should I create for a topic?**
    -   *A*: `Max(TargetThroughput / ProducerThroughput, TargetThroughput / ConsumerThroughput)`. Also consider future growth.

2.  **Q: Why is "Zero Copy" important for capacity?**
    -   *A*: It allows the broker to send data from Disk to Network without copying it into User Space (RAM). Reduces CPU usage and Context Switches.

3.  **Q: How do you handle a sudden 10x traffic spike?**
    -   *A*: Kafka buffers it on disk. Latency might increase, but data is safe. Consumers will lag. To fix: Scale consumers or throttle producers.

### Production Challenges
-   **Challenge**: **Disk Full Outage**.
    -   *Scenario*: Disk fills up. Broker crashes.
    -   *Fix*: Set `log.retention.bytes`. Use Tiered Storage. Monitor disk usage.

-   **Challenge**: **Noisy Neighbor**.
    -   *Scenario*: One topic consumes all network bandwidth.
    -   *Fix*: Use **Quotas** (Network bandwidth quotas) to limit throughput per client/user.

### Troubleshooting Scenarios
**Scenario**: High I/O Wait time on Linux.
-   *Cause*: Disk saturation.
-   *Fix*: Check if consumers are reading old data (Page Cache miss). Add more disks or brokers.
"""
}

print("🚀 Populating Week 8 Observability Content (Detailed)...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 8 Content Population Complete!")
