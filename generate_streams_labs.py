import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda"

def create_lab_content(title, difficulty, time, objectives, problem, starter_code):
    return f"""# {title}

## Difficulty
{difficulty}

## Estimated Time
{time}

## Learning Objectives
{objectives}

## Problem Statement
{problem}

## Starter Code
```python
{starter_code}
```

## Hints
<details>
<summary>Hint 1</summary>
Focus on the core logic first.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>
Solution will be provided after you attempt the problem.
</details>
"""

def update_lab(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {path}")

lab_content = {
    "Phase1_Foundations/Week1_Kafka_Architecture": [
        ("Lab 01: Start a Kafka Cluster", "🟢 Easy", "30 mins", "- Docker Compose\n- Kafka CLI", "Set up a 3-node Kafka cluster using Docker Compose and verify it's running.", "version: '2'\nservices:\n  zookeeper:\n    image: confluentinc/cp-zookeeper:latest"),
        ("Lab 02: Topic Management", "🟢 Easy", "30 mins", "- AdminClient", "Create a topic with 3 partitions and replication factor of 2 using Python AdminClient.", "from confluent_kafka.admin import AdminClient, NewTopic\n\ndef create_topic(conf, topic_name):\n    pass"),
        ("Lab 03: Basic Producer", "🟢 Easy", "30 mins", "- Producer API", "Implement a producer that sends JSON events to a topic.", "from confluent_kafka import Producer\nimport json\n\ndef produce_events(topic):\n    pass"),
        ("Lab 04: Consumer Groups", "🟡 Medium", "45 mins", "- Consumer Groups", "Start multiple consumers in the same group and observe partition assignment.", "from confluent_kafka import Consumer\n\ndef consume_loop(group_id):\n    pass"),
        ("Lab 05: ISR & Min.Insync.Replicas", "🟡 Medium", "60 mins", "- Reliability", "Simulate a broker failure and observe producer behavior with acks=all.", "conf = {'acks': 'all', 'min.insync.replicas': 2}"),
        ("Lab 06: Log Compaction", "🟡 Medium", "45 mins", "- Storage", "Configure a topic for log compaction and produce duplicate keys to verify compaction.", "config={'cleanup.policy': 'compact'}"),
        ("Lab 07: Custom Partitioner", "🟡 Medium", "60 mins", "- Partitioning", "Implement a custom partitioner to send VIP users to a specific partition.", "def custom_partitioner(key, all_partitions, available):\n    pass"),
        ("Lab 08: Idempotent Producer", "🟡 Medium", "45 mins", "- Exactly-Once", "Enable idempotence on the producer and simulate network retries.", "conf = {'enable.idempotence': True}"),
        ("Lab 09: Consumer Offsets", "🟡 Medium", "45 mins", "- Offsets", "Manually commit offsets and seek to a specific offset.", "consumer.commit(asynchronous=False)\nconsumer.seek(partition)"),
        ("Lab 10: Kafka Connect Basics", "🟢 Easy", "45 mins", "- Connect", "Set up a FileStreamSource connector to ingest data from a file.", "name=file-source\nconnector.class=FileStreamSource"),
        ("Lab 11: Kafka Streams DSL", "🟡 Medium", "60 mins", "- Streams API", "Implement a Word Count application using Kafka Streams (Java/Scala concept in Python via Faust).", "app = faust.App('word-count', broker='kafka://localhost')"),
        ("Lab 12: Schema Registry Setup", "🟡 Medium", "60 mins", "- Schemas", "Set up Schema Registry and produce Avro messages.", "from confluent_kafka.schema_registry import SchemaRegistryClient"),
        ("Lab 13: ACLs & Security", "🔴 Hard", "60 mins", "- Security", "Configure ACLs to restrict read/write access to a topic.", "admin.create_acls([acl_binding])"),
        ("Lab 14: Monitoring with JMX", "🟡 Medium", "60 mins", "- Observability", "Enable JMX ports and connect via JConsole to view metrics.", "KAFKA_JMX_OPTS='-Dcom.sun.management.jmxremote'"),
        ("Lab 15: Multi-Broker Setup", "🔴 Hard", "90 mins", "- Operations", "Manually set up a multi-broker cluster without Docker (local processes).", "server.properties files setup")
    ],
    "Phase1_Foundations/Week2_Redpanda_HighPerformance": [
        ("Lab 01: Redpanda Docker Setup", "🟢 Easy", "30 mins", "- Redpanda", "Start a Redpanda cluster using Docker.", "docker run -d --name redpanda -p 9092:9092..."),
        ("Lab 02: rpk CLI Basics", "🟢 Easy", "30 mins", "- CLI", "Use `rpk` to create topics, produce, and consume messages.", "rpk topic create my-topic -p 3"),
        ("Lab 03: Redpanda vs Kafka Benchmarking", "🟡 Medium", "60 mins", "- Performance", "Run a producer benchmark against both Kafka and Redpanda.", "rpk topic produce --help"),
        ("Lab 04: WASM Data Transforms", "🔴 Hard", "90 mins", "- Data Transforms", "Deploy a WASM transform to Redpanda to mask PII data in real-time.", "transform.yaml configuration"),
        ("Lab 05: Tiered Storage Config", "🟡 Medium", "60 mins", "- Archival", "Configure S3 tiered storage for a topic.", "rpk topic create --topic-config redpanda.remote.write=true"),
        ("Lab 06: Redpanda Console", "🟢 Easy", "30 mins", "- UI", "Explore the Redpanda Console (formerly Kowl) to view messages.", "docker-compose up console"),
        ("Lab 07: Schema Registry in Redpanda", "🟡 Medium", "45 mins", "- Schemas", "Use Redpanda's built-in Schema Registry with a Python producer.", "SchemaRegistryClient(url='http://localhost:8081')"),
        ("Lab 08: Admin API", "🟡 Medium", "45 mins", "- Administration", "Use the Admin API to manage Redpanda users.", "requests.post('http://admin-api/v1/users')"),
        ("Lab 09: Tuning Redpanda", "🔴 Hard", "60 mins", "- Tuning", "Tune Redpanda for high throughput (batch size, linger.ms).", "rpk redpanda tune all"),
        ("Lab 10: Redpanda Connect", "🟢 Easy", "45 mins", "- Integration", "Use Redpanda Connect (Benthos) to ingest data.", "input:\n  generate:\n    mapping: root = 'hello world'"),
        ("Lab 11: Shadow Indexing", "🟡 Medium", "60 mins", "- Storage", "Demonstrate fetching data from S3 via Shadow Indexing.", "Consume from offset 0 (old data)"),
        ("Lab 12: Maintenance Mode", "🟡 Medium", "45 mins", "- Operations", "Put a node into maintenance mode for upgrades.", "rpk cluster maintenance enable <node-id>"),
        ("Lab 13: Partition Rebalancing", "🟡 Medium", "60 mins", "- Operations", "Observe and trigger partition rebalancing.", "rpk cluster partitions balancer-status"),
        ("Lab 14: Redpanda Security", "🔴 Hard", "60 mins", "- Security", "Enable SASL/SCRAM authentication.", "rpk acl user create myuser"),
        ("Lab 15: HTTP Proxy", "🟢 Easy", "30 mins", "- Access", "Produce and consume messages via the PandaProxy HTTP API.", "curl -X POST http://localhost:8082/topics/test")
    ],
    "Phase2_Stream_Processing_Flink/Week3_Flink_Fundamentals": [
        ("Lab 01: Local Flink Cluster", "🟢 Easy", "30 mins", "- Setup", "Start a local Flink cluster and access the Web UI.", "./bin/start-cluster.sh"),
        ("Lab 02: Word Count (DataStream)", "🟢 Easy", "45 mins", "- Basics", "Implement the classic Word Count using DataStream API.", "env.from_elements(...).flat_map(...).key_by(...).sum(...)"),
        ("Lab 03: Kafka Source & Sink", "🟡 Medium", "60 mins", "- Connectors", "Read from Kafka, process, and write back to Kafka.", "KafkaSource.builder()..."),
        ("Lab 04: Event Time & Watermarks", "🟡 Medium", "60 mins", "- Time", "Implement a custom WatermarkStrategy for out-of-order data.", "WatermarkStrategy.for_bounded_out_of_orderness(...)"),
        ("Lab 05: Tumbling Windows", "🟢 Easy", "45 mins", "- Windows", "Aggregate data in fixed-size non-overlapping windows.", "window(TumblingEventTimeWindows.of(Time.seconds(10)))"),
        ("Lab 06: Sliding Windows", "🟢 Easy", "45 mins", "- Windows", "Implement sliding windows for moving averages.", "window(SlidingEventTimeWindows.of(...))"),
        ("Lab 07: Session Windows", "🟡 Medium", "45 mins", "- Windows", "Group events by user activity sessions.", "window(EventTimeSessionWindows.withGap(...))"),
        ("Lab 08: ProcessWindowFunction", "🟡 Medium", "60 mins", "- Low-level", "Use ProcessWindowFunction to access window metadata.", "class MyProcessWindow(ProcessWindowFunction):"),
        ("Lab 09: Rich Functions", "🟡 Medium", "45 mins", "- Lifecycle", "Use RichMapFunction to initialize resources (open/close).", "class MyRichMap(RichMapFunction):"),
        ("Lab 10: Side Outputs", "🟡 Medium", "45 mins", "- Late Data", "Route late data to a side output tag.", "ctx.output(late_tag, element)"),
        ("Lab 11: CoProcessFunction", "🔴 Hard", "60 mins", "- Joins", "Connect two streams and process them together.", "stream1.connect(stream2).process(...)"),
        ("Lab 12: Broadcast State", "🔴 Hard", "90 mins", "- Patterns", "Broadcast a control stream (rules) to all parallel instances.", "ctx.getBroadcastState(rule_state_descriptor)"),
        ("Lab 13: Accumulators", "🟢 Easy", "30 mins", "- Metrics", "Use accumulators to count events globally.", "getRuntimeContext().addAccumulator(...)"),
        ("Lab 14: ParameterTool", "🟢 Easy", "30 mins", "- Config", "Pass configuration parameters to the Flink job.", "ParameterTool.fromArgs(args)"),
        ("Lab 15: Job Submission", "🟢 Easy", "30 mins", "- Deployment", "Submit a job via CLI and REST API.", "flink run -c com.example.Job my-jar.jar")
    ],
    "Phase2_Stream_Processing_Flink/Week4_Stateful_Processing": [
        ("Lab 01: ValueState", "🟢 Easy", "45 mins", "- State", "Implement a stateful mapper using ValueState.", "state.update(current + 1)"),
        ("Lab 02: ListState", "🟢 Easy", "45 mins", "- State", "Buffer elements in ListState.", "list_state.add(value)"),
        ("Lab 03: MapState", "🟡 Medium", "45 mins", "- State", "Use MapState to store per-key dictionaries.", "map_state.put(key, value)"),
        ("Lab 04: ReducingState", "🟡 Medium", "45 mins", "- State", "Use ReducingState for continuous aggregation.", "reducing_state.add(value)"),
        ("Lab 05: AggregatingState", "🟡 Medium", "45 mins", "- State", "Use AggregatingState for complex aggregations (Avg).", "agg_state.add(value)"),
        ("Lab 06: State TTL", "🟡 Medium", "45 mins", "- Cleanup", "Configure Time-To-Live for state to prevent unlimited growth.", "StateTtlConfig.newBuilder(...)"),
        ("Lab 07: Checkpointing Config", "🟢 Easy", "30 mins", "- Fault Tolerance", "Enable and configure checkpointing in Flink.", "env.enableCheckpointing(1000)"),
        ("Lab 08: RocksDB Backend", "🟡 Medium", "45 mins", "- Backends", "Switch state backend to RocksDB for large state.", "env.setStateBackend(EmbeddedRocksDBStateBackend())"),
        ("Lab 09: Savepoints", "🟡 Medium", "60 mins", "- Operations", "Trigger a savepoint, stop the job, and resume from savepoint.", "flink savepoint <job-id>"),
        ("Lab 10: Schema Evolution", "🔴 Hard", "90 mins", "- Migration", "Modify state schema and restore from an old savepoint using State Processor API.", "State Processor API usage"),
        ("Lab 11: Keyed Process Function", "🟡 Medium", "60 mins", "- Timers", "Register processing time and event time timers.", "ctx.timerService().registerEventTimeTimer(...)"),
        ("Lab 12: Async I/O", "🔴 Hard", "60 mins", "- Performance", "Implement Async I/O for external database lookups.", "AsyncDataStream.orderedWait(...)"),
        ("Lab 13: Operator State", "🔴 Hard", "60 mins", "- State", "Implement CheckpointedFunction for non-keyed state.", "snapshotState / initializeState"),
        ("Lab 14: Queryable State", "🔴 Hard", "60 mins", "- Access", "Expose state for external querying (if supported/simulated).", "client.getKvState(...)"),
        ("Lab 15: State Size Monitoring", "🟡 Medium", "45 mins", "- Metrics", "Monitor checkpoint size and duration.", "Web UI Checkpoints tab")
    ],
    "Phase2_Stream_Processing_Flink/Week5_Advanced_Flink": [
        ("Lab 01: Flink SQL Basics", "🟢 Easy", "45 mins", "- SQL", "Run simple SQL queries on DataStreams.", "table_env.sql_query('SELECT * FROM source')"),
        ("Lab 02: Table API", "🟢 Easy", "45 mins", "- Table API", "Use Table API for relational operations.", "table.select(...).filter(...)"),
        ("Lab 03: Kafka Connector in SQL", "🟡 Medium", "60 mins", "- Connectors", "Define a Kafka source/sink table using DDL.", "CREATE TABLE KafkaTable (...) WITH ('connector'='kafka'...)"),
        ("Lab 04: Windowing in SQL", "🟡 Medium", "60 mins", "- SQL", "Perform Tumble and Hop window aggregations in SQL.", "GROUP BY TUMBLE(rowtime, INTERVAL '1' MINUTE)"),
        ("Lab 05: Pattern Recognition (CEP)", "🔴 Hard", "90 mins", "- CEP", "Detect a specific sequence of events (Start -> Middle -> End).", "Pattern.begin('start').next('middle')..."),
        ("Lab 06: CEP with Time Constraints", "🔴 Hard", "60 mins", "- CEP", "Detect patterns that happen within a specific timeframe.", "within(Time.seconds(10))"),
        ("Lab 07: Interval Joins", "🟡 Medium", "60 mins", "- Joins", "Join two streams based on a time interval.", "WHERE a.ts BETWEEN b.ts - INTERVAL '5' MINUTE AND ..."),
        ("Lab 08: Temporal Table Joins", "🔴 Hard", "90 mins", "- Joins", "Join a stream with a versioned table (CDC).", "FOR SYSTEM_TIME AS OF stream.proctime"),
        ("Lab 09: User Defined Functions (UDF)", "🟡 Medium", "60 mins", "- SQL", "Implement a ScalarFunction for custom logic.", "class MyUDF(ScalarFunction):"),
        ("Lab 10: User Defined Aggregate Functions", "🔴 Hard", "90 mins", "- SQL", "Implement an AggregateFunction.", "class MyAgg(AggregateFunction):"),
        ("Lab 11: SQL Client", "🟢 Easy", "30 mins", "- CLI", "Use the Flink SQL Client to submit queries.", "./bin/sql-client.sh"),
        ("Lab 12: Catalogs", "🟡 Medium", "45 mins", "- Metadata", "Register a Hive or Generic InMemoryCatalog.", "table_env.register_catalog(...)"),
        ("Lab 13: Deduplication", "🟡 Medium", "45 mins", "- SQL", "Remove duplicates using ROW_NUMBER() over partition.", "ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC)"),
        ("Lab 14: Top-N Query", "🟡 Medium", "60 mins", "- SQL", "Find top N items per category.", "WHERE row_num <= N"),
        ("Lab 15: Kubernetes Deployment", "🔴 Hard", "90 mins", "- Ops", "Deploy a Flink job to a local Minikube/Kind cluster.", "kubectl apply -f flink-configuration.yaml")
    ],
    "Phase3_Advanced_Architecture/Week6_Streaming_Patterns": [
        ("Lab 01: Event Sourcing Producer", "🟡 Medium", "60 mins", "- Patterns", "Implement a producer that emits state change events.", "Event(type='AccountCreated', data={...})"),
        ("Lab 02: Event Sourcing Consumer (Materialized View)", "🟡 Medium", "60 mins", "- Patterns", "Reconstruct current state from event log.", "apply_event(state, event)"),
        ("Lab 03: CQRS Implementation", "🔴 Hard", "90 mins", "- Patterns", "Separate Command and Query sides using Kafka.", "Command Service -> Kafka -> Query Service"),
        ("Lab 04: Kappa Architecture", "🟡 Medium", "60 mins", "- Architecture", "Process same data for real-time and reprocessing (replay).", "Reset offsets to 0 to reprocess"),
        ("Lab 05: Stream Enrichment (Join)", "🟡 Medium", "60 mins", "- Enrichment", "Enrich a click stream with user data using a join.", "clicks.join(users)"),
        ("Lab 06: Stream Enrichment (Async Lookup)", "🔴 Hard", "90 mins", "- Enrichment", "Enrich stream using Async I/O to external DB.", "AsyncFunction to call REST API/DB"),
        ("Lab 07: Dead Letter Queue (DLQ)", "🟡 Medium", "45 mins", "- Error Handling", "Route failed messages to a DLQ topic.", "catch exception -> produce to 'dlq-topic'"),
        ("Lab 08: Retry Strategy", "🟡 Medium", "45 mins", "- Error Handling", "Implement exponential backoff for retries.", "sleep(2 ** attempt)"),
        ("Lab 09: Idempotent Consumer", "🟡 Medium", "60 mins", "- Reliability", "Use a unique ID to prevent processing duplicates.", "if id in processed_cache: skip"),
        ("Lab 10: Transactional Producer", "🔴 Hard", "60 mins", "- Transactions", "Write to multiple topics atomically.", "producer.init_transactions()... commit_transaction()"),
        ("Lab 11: Read-Process-Write Transaction", "🔴 Hard", "90 mins", "- Transactions", "Consume, process, and produce exactly-once.", "consume -> process -> produce (in transaction)"),
        ("Lab 12: Outbox Pattern", "🔴 Hard", "90 mins", "- Patterns", "Implement Outbox pattern to sync DB and Kafka.", "DB Insert -> CDC -> Kafka"),
        ("Lab 13: Saga Pattern Orchestration", "🔴 Hard", "90 mins", "- Microservices", "Implement a Saga orchestrator using Kafka.", "Order -> Payment -> Shipping"),
        ("Lab 14: Strangler Fig Pattern", "🟡 Medium", "60 mins", "- Migration", "Migrate legacy system to streaming by intercepting events.", "Intercept calls -> produce event"),
        ("Lab 15: Throttling Pattern", "🟡 Medium", "45 mins", "- Stability", "Implement a token bucket throttler in a Flink map.", "if bucket.try_consume(): process else: wait")
    ],
    "Phase3_Advanced_Architecture/Week7_Reliability_Scalability": [
        ("Lab 01: Backpressure Simulation", "🟡 Medium", "45 mins", "- Performance", "Create a slow consumer and observe producer slowdown.", "Thread.sleep(100) in consumer"),
        ("Lab 02: Flink Backpressure Monitoring", "🟢 Easy", "30 mins", "- Ops", "Identify backpressure in Flink Web UI.", "Check 'Backpressure' tab"),
        ("Lab 03: Partitioning Strategies", "🟡 Medium", "45 mins", "- Scaling", "Compare Round-Robin vs Key-Hash partitioning.", "producer.produce(..., key=...)"),
        ("Lab 04: Handling Skew", "🔴 Hard", "90 mins", "- Scaling", "Implement 'Salted Keys' to handle hot partitions.", "key = original_key + random_salt"),
        ("Lab 05: Rescaling Flink Job", "🟡 Medium", "45 mins", "- Scaling", "Change parallelism of a running job (with savepoint).", "flink modify -p <new-p>"),
        ("Lab 06: MirrorMaker 2 Setup", "🔴 Hard", "90 mins", "- Geo-Replication", "Set up MM2 to replicate between two local clusters.", "mm2.properties configuration"),
        ("Lab 07: Cluster Linking (Confluent/Redpanda)", "🟡 Medium", "60 mins", "- Geo-Replication", "Simulate cluster linking concepts.", "link configuration"),
        ("Lab 08: Data Contracts", "🟡 Medium", "60 mins", "- Governance", "Validate data against a strict contract/schema.", "validate(json, schema)"),
        ("Lab 09: Lineage Tracking", "🟡 Medium", "60 mins", "- Governance", "Add headers to trace message lineage.", "headers={'trace_id': ...}"),
        ("Lab 10: Encryption at Rest", "🟢 Easy", "30 mins", "- Security", "Configure volume encryption for Kafka/Redpanda (conceptual).", "Disk encryption setup"),
        ("Lab 11: TLS Encryption", "🔴 Hard", "90 mins", "- Security", "Generate certs and configure TLS for Kafka.", "keystore/truststore setup"),
        ("Lab 12: RBAC Configuration", "🟡 Medium", "60 mins", "- Security", "Configure Role-Based Access Control.", "Grant 'Reader' role to user"),
        ("Lab 13: Quotas", "🟡 Medium", "45 mins", "- Multi-tenancy", "Set produce/consume quotas for a client.", "admin.alter_client_quotas(...)"),
        ("Lab 14: Rack Awareness", "🔴 Hard", "60 mins", "- Availability", "Configure rack awareness to survive rack failures.", "broker.rack configuration"),
        ("Lab 15: Chaos Engineering", "🔴 Hard", "90 mins", "- Reliability", "Randomly kill brokers/taskmanagers and verify recovery.", "kill -9 <pid>")
    ],
    "Phase4_Production_CaseStudies/Week8_Observability_Operations": [
        ("Lab 01: Prometheus & Grafana Setup", "🟡 Medium", "60 mins", "- Monitoring", "Set up Prometheus to scrape Kafka/Flink metrics.", "prometheus.yml config"),
        ("Lab 02: Consumer Lag Monitoring", "🟢 Easy", "45 mins", "- Metrics", "Create a Grafana dashboard for Consumer Lag.", "kafka_consumergroup_lag metric"),
        ("Lab 03: Flink Checkpoint Monitoring", "🟢 Easy", "30 mins", "- Metrics", "Alert on failed checkpoints.", "flink_jobmanager_job_numberOfFailedCheckpoints"),
        ("Lab 04: Log Analysis with ELK", "🟡 Medium", "60 mins", "- Logging", "Ingest Kafka logs into Elasticsearch/Kibana.", "Filebeat -> Logstash -> ES"),
        ("Lab 05: Distributed Tracing (Jaeger)", "🔴 Hard", "90 mins", "- Tracing", "Implement OpenTelemetry tracing in producer/consumer.", "tracer.start_span(...)"),
        ("Lab 06: AlertManager Config", "🟡 Medium", "45 mins", "- Alerting", "Configure alerts for high lag and broker down.", "alertmanager.yml"),
        ("Lab 07: Capacity Planning Calculation", "🟢 Easy", "30 mins", "- Planning", "Calculate disk/network requirements based on throughput.", "Excel/Python calculator"),
        ("Lab 08: Topic Inspection", "🟢 Easy", "30 mins", "- Ops", "Use `kcat` (kafkacat) to inspect headers and payloads.", "kcat -C -b ..."),
        ("Lab 09: Reset Consumer Offsets", "🟡 Medium", "30 mins", "- Ops", "Reset a consumer group to a specific datetime.", "kafka-consumer-groups --reset-offsets"),
        ("Lab 10: Handling Poison Pills", "🟡 Medium", "60 mins", "- Ops", "Implement a deserialization error handler.", "try: deserialize except: send_to_dlq"),
        ("Lab 11: Rolling Restart", "🔴 Hard", "60 mins", "- Ops", "Perform a rolling restart of the cluster.", "Stop broker 1 -> Start -> Wait -> Stop broker 2..."),
        ("Lab 12: Reassign Partitions", "🔴 Hard", "60 mins", "- Ops", "Move partitions to new brokers using reassignment tool.", "kafka-reassign-partitions"),
        ("Lab 13: Flink Heap Tuning", "🔴 Hard", "60 mins", "- Tuning", "Analyze GC logs and tune heap size.", "-Xmx configuration"),
        ("Lab 14: Network Tuning", "🟡 Medium", "45 mins", "- Tuning", "Tune socket buffer sizes for high throughput.", "socket.send.buffer.bytes"),
        ("Lab 15: SLO Definition", "🟢 Easy", "30 mins", "- SRE", "Define SLOs for availability and latency.", "Document SLOs")
    ],
    "Phase4_Production_CaseStudies/Week9_RealWorld_CaseStudies": [
        ("Lab 01: Fraud Detection Pipeline", "🔴 Hard", "90 mins", "- Case Study", "Build a Flink pipeline to detect high-value transactions in short windows.", "Window join transaction stream with rules"),
        ("Lab 02: IoT Sensor Aggregation", "🟡 Medium", "60 mins", "- Case Study", "Aggregate temperature sensors by location.", "KeyBy(location).Window(1 min).Avg()"),
        ("Lab 03: Clickstream Sessionization", "🟡 Medium", "60 mins", "- Case Study", "Sessionize user clicks with a 30-min gap.", "SessionWindows.withGap(Time.minutes(30))"),
        ("Lab 04: CDC with Debezium", "🔴 Hard", "90 mins", "- Case Study", "Set up Debezium to capture MySQL changes to Kafka.", "Debezium connector setup"),
        ("Lab 05: Log Aggregation Pipeline", "🟡 Medium", "60 mins", "- Case Study", "Parse and filter system logs for errors.", "Filter(level='ERROR')"),
        ("Lab 06: Real-time Leaderboard", "🟡 Medium", "60 mins", "- Case Study", "Maintain a top-10 leaderboard for a game.", "ProcessFunction with PriorityQueue"),
        ("Lab 07: Inventory Management", "🔴 Hard", "90 mins", "- Case Study", "Handle inventory reservations with timeouts.", "KeyedProcessFunction with timers"),
        ("Lab 08: Dynamic Pricing", "🔴 Hard", "90 mins", "- Case Study", "Adjust prices based on demand window.", "Demand > Threshold -> Increase Price"),
        ("Lab 09: Social Media Feed", "🔴 Hard", "90 mins", "- Case Study", "Fan-out architecture for user feeds.", "Produce to user-specific timelines"),
        ("Lab 10: Geo-Fencing", "🔴 Hard", "90 mins", "- Case Study", "Detect when a vehicle enters a polygon.", "Point in Polygon check in Flink"),
        ("Lab 11: Stock Market Analysis", "🟡 Medium", "60 mins", "- Case Study", "Calculate Moving Average Convergence Divergence (MACD).", "Complex window aggregation"),
        ("Lab 12: Recommendation Engine", "🔴 Hard", "90 mins", "- Case Study", "Update user profile in real-time based on clicks.", "Stateful profile update"),
        ("Lab 13: Ad Tech Bidding", "🔴 Hard", "90 mins", "- Case Study", "Real-time bidding within 100ms latency budget.", "Low latency pipeline optimization"),
        ("Lab 14: Cybersecurity Threat Detection", "🟡 Medium", "60 mins", "- Case Study", "Detect multiple failed logins from same IP.", "CEP Pattern: Fail -> Fail -> Fail"),
        ("Lab 15: ETL Offloading", "🟡 Medium", "60 mins", "- Case Study", "Move heavy ETL from Batch to Streaming.", "Continuous ETL pipeline")
    ],
    "Phase4_Production_CaseStudies/Week10_Challenges_Trends": [
        ("Lab 01: Hot Key Handling", "🔴 Hard", "90 mins", "- Challenge", "Implement 'Local Key Aggregation' to mitigate hot keys.", "Pre-aggregate before keyBy"),
        ("Lab 02: Schema Evolution Test", "🟡 Medium", "60 mins", "- Challenge", "Produce data with new schema and verify consumer compatibility.", "Add field with default value"),
        ("Lab 03: Late Data Handling", "🟡 Medium", "60 mins", "- Challenge", "Implement strategy to update results when late data arrives.", "allowedLateness(Time.hours(1))"),
        ("Lab 04: Materialize View", "🟡 Medium", "60 mins", "- Trend", "Simulate a Materialized View using Flink Table.", "CREATE MATERIALIZED VIEW"),
        ("Lab 05: Iceberg Sink", "🟡 Medium", "60 mins", "- Trend", "Write streaming data to Apache Iceberg table.", "Flink Iceberg Sink"),
        ("Lab 06: Paimon Sink", "🟡 Medium", "60 mins", "- Trend", "Write to Apache Paimon for unified batch/stream.", "Flink Paimon Sink"),
        ("Lab 07: Tiered Storage Access", "🟡 Medium", "45 mins", "- Trend", "Read old data transparently from S3 via Redpanda.", "Consume from offset 0"),
        ("Lab 08: WASM Filter", "🔴 Hard", "60 mins", "- Trend", "Write a custom WASM filter for Redpanda.", "Rust/Go WASM code"),
        ("Lab 09: Data Mesh Product", "🟡 Medium", "60 mins", "- Trend", "Define a 'Data Product' on a stream.", "Metadata + Schema + SLO"),
        ("Lab 10: Stream Governance API", "🟡 Medium", "45 mins", "- Trend", "Interact with a governance API to register a stream.", "POST /streams"),
        ("Lab 11: Vector Database Sink", "🟡 Medium", "60 mins", "- AI/ML", "Stream embeddings to a Vector DB (Pinecone/Milvus).", "Sink to VectorDB"),
        ("Lab 12: Real-time RAG", "🔴 Hard", "90 mins", "- AI/ML", "Build a pipeline for Real-time Retrieval Augmented Generation.", "Kafka -> Embedding -> VectorDB"),
        ("Lab 13: Feature Engineering Pipeline", "🟡 Medium", "60 mins", "- AI/ML", "Compute real-time features for ML inference.", "Sliding window features"),
        ("Lab 14: Online Prediction", "🟡 Medium", "60 mins", "- AI/ML", "Serve ML model within a Flink function.", "Load model -> Predict"),
        ("Lab 15: Final Capstone Project", "🔴 Hard", "120 mins", "- Capstone", "Design and implement an end-to-end streaming architecture.", "Architecture Diagram + Implementation")
    ]
}

print("🚀 Starting Streams Lab Generation...")

for folder, labs in lab_content.items():
    base_lab_path = os.path.join(base_path, folder, "labs")
    for i, (title, diff, time, obj, prob, code) in enumerate(labs):
        filename = f"lab_{i+1:02d}.md"
        path = os.path.join(base_lab_path, filename)
        content = create_lab_content(title, diff, time, obj, prob, code)
        update_lab(path, content)

print("✅ Streams Lab Generation Complete!")
