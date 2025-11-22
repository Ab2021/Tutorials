# Streams Course: Kafka, Flink, Redpanda
## Mastering Event Streaming Architectures

**Course Objective:**
To provide a deep, theoretical, and practical understanding of modern event streaming architectures using Apache Kafka, Apache Flink, and Redpanda. This course focuses on reasoning, architectural patterns, latest advancements, and real-world production challenges.

**Structure:**
- **Phases**: 4 Major Phases
- **Weeks**: 10 Weeks total
- **Daily Structure**: 
    - **Core Concepts**: Deep dive into theory and architecture.
    - **Deep Dive**: Advanced reasoning, internal mechanics, and "why" things work the way they do.
    - **Interview/Challenges**: Common interview questions and production issues.
- **Labs**: Weekly hands-on labs (15 per week) focusing on implementation and experimentation.

---

## Phase 1: Event Streaming Foundations (Weeks 1-2)
*Focus: Understanding the log abstraction, distributed commit logs, and the architecture of modern streaming platforms.*

### Week 1: The Log & Kafka Architecture
- **Day 1**: Introduction to Event Streaming vs. Batch Processing. The "Log" abstraction.
- **Day 2**: Kafka Architecture Deep Dive (Brokers, Zookeeper/KRaft, Controller).
- **Day 3**: Topics, Partitions, and Segments (Storage internals, Indexing).
- **Day 4**: Producers & Consumers (Delivery semantics, Consumer Groups, Rebalancing).
- **Day 5**: Reliability & Durability (Replication, ISR, Acks, Min.insync.replicas).

### Week 2: Redpanda & High-Performance Streaming
- **Day 1**: Introduction to Redpanda (Thread-per-core architecture, C++ vs JVM).
- **Day 2**: Redpanda vs. Kafka (Performance, Operational simplicity, WASM transforms).
- **Day 3**: Schema Registry & Serialization (Avro, Protobuf, Compatibility modes).
- **Day 4**: Admin Operations (Topic management, Tiered Storage, ACLs).
- **Day 5**: Advanced Configuration & Tuning (Network threads, Disk I/O, Latency optimization).

---

## Phase 2: Stream Processing with Apache Flink (Weeks 3-5)
*Focus: Stateful computations over data streams, time semantics, and fault tolerance.*

### Week 3: Flink Fundamentals
- **Day 1**: Introduction to Stream Processing (Dataflow model, DAGs, JobManager/TaskManager).
- **Day 2**: DataStream API Basics (Sources, Sinks, Transformations).
- **Day 3**: Time Semantics (Event Time vs. Processing Time, Watermarks).
- **Day 4**: Windowing Strategies (Tumbling, Sliding, Session, Global).
- **Day 5**: Triggers & Evictors (Customizing window behavior).

### Week 4: Stateful Stream Processing
- **Day 1**: State Management (Keyed State, Operator State, Broadcast State).
- **Day 2**: State Backends (HashMap, EmbeddedRocksDB).
- **Day 3**: Checkpointing & Fault Tolerance (Chandy-Lamport algorithm, Barriers).
- **Day 4**: Savepoints vs. Checkpoints (Operational usage, Upgrading jobs).
- **Day 5**: State Evolution & Schema Migration.

### Week 5: Advanced Flink & Table API
- **Day 1**: Flink SQL & Table API (Unified batch/stream processing).
- **Day 2**: Complex Event Processing (CEP) (Pattern detection).
- **Day 3**: Joins in Streaming (Interval joins, Temporal table joins).
- **Day 4**: Async I/O & Side Outputs (Handling external systems, Late data).
- **Day 5**: Flink Deployment Modes (Session, Per-Job, Application, Kubernetes).

---

## Phase 3: Advanced Architecture & Patterns (Weeks 6-7)
*Focus: Architectural patterns for building scalable and resilient streaming systems.*

### Week 6: Streaming Patterns
- **Day 1**: Event Sourcing & CQRS (Concepts, Implementation with Kafka).
- **Day 2**: The Kappa Architecture (vs Lambda).
- **Day 3**: Stream Enrichment Patterns (Join vs Lookup).
- **Day 4**: Dead Letter Queues & Error Handling Strategies.
- **Day 5**: Idempotency & Transactional Messaging (Exactly-Once Semantics).

### Week 7: Reliability, Scalability & Challenges
- **Day 1**: Backpressure Handling (Credit-based flow control in Flink).
- **Day 2**: Scaling Streaming Systems (Partitioning strategies, Rescaling Flink jobs).
- **Day 3**: Multi-Region/Geo-Replication (MirrorMaker 2, Cluster Linking).
- **Day 4**: Stream Governance (Data contracts, Lineage, Quality).
- **Day 5**: Security in Streaming (mTLS, SASL, RBAC, Encryption at rest).

---

## Phase 4: Production & Real-World Case Studies (Weeks 8-10)
*Focus: Operational excellence, observability, and applying concepts to real-world scenarios.*

### Week 8: Observability & Operations
- **Day 1**: Monitoring Kafka/Redpanda (Lag, Throughput, Disk usage, Key metrics).
- **Day 2**: Monitoring Flink (Checkpoint duration, Backpressure, GC pauses).
- **Day 3**: Alerting Strategies & SLOs/SLAs.
- **Day 4**: Troubleshooting Common Issues (Rebalance storms, Stuck consumers, OOMs).
- **Day 5**: Capacity Planning & Sizing.

### Week 9: Real-World Case Studies
- **Day 1**: Case Study: Real-time Fraud Detection (Financial services).
- **Day 2**: Case Study: IoT Telemetry Ingestion & Processing (Automotive).
- **Day 3**: Case Study: Real-time Clickstream Analytics (E-commerce).
- **Day 4**: Case Study: Change Data Capture (CDC) (Debezium, DB migration).
- **Day 5**: Case Study: Log Aggregation & SIEM (Cybersecurity).

### Week 10: Challenges, Issues & Future Trends
- **Day 1**: Challenge: Handling Skewed Data (Hot keys).
- **Day 2**: Challenge: Schema Evolution in Long-Running Streams.
- **Day 3**: Challenge: Late Data Handling & Correctness.
- **Day 4**: Future Trend: Streaming Databases (Materialize, RisingWave).
- **Day 5**: Future Trend: Unified Batch & Stream (Apache Paimon, Hudi, Iceberg).
