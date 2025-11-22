# Day 1: Monitoring Kafka & Redpanda

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
