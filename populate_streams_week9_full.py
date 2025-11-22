import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase4_Production_CaseStudies\Week9_RealWorld_CaseStudies"

content_map = {
    # --- Day 1: Fraud Detection ---
    "Day1_Fraud_Detection_Core.md": """# Day 1: Real-Time Fraud Detection

## Core Concepts & Theory

### The Use Case
Detect fraudulent credit card transactions in real-time (< 200ms).
-   **Input**: Stream of transactions (CardID, Amount, Merchant, Location).
-   **Logic**: Complex patterns (e.g., "Card used in London and NYC within 5 mins").
-   **Output**: Block transaction or alert analyst.

### Architecture
1.  **Ingestion**: Payment Gateway -> Kafka (`transactions` topic).
2.  **Enrichment**: Flink joins with `CustomerProfile` (Redis/State).
3.  **Pattern Matching**: Flink CEP (Complex Event Processing).
4.  **Action**: Flink writes to `alerts` topic -> Consumer blocks card.

### Key Patterns
-   **Dynamic Rules**: Rules are stored in a database and broadcasted to Flink. No code redeployment needed for new rules.
-   **Feature Engineering**: Calculating rolling aggregates (e.g., "Avg spend in last 24h") in real-time.

### Architectural Reasoning
**Why Flink CEP?**
Standard SQL is good for aggregation ("Sum of sales"). CEP is good for **sequences** ("Event A followed by Event B within 10 mins"). Fraud is almost always a sequence pattern.
""",

    "Day1_Fraud_Detection_DeepDive.md": """# Day 1: Fraud Detection - Deep Dive

## Deep Dive & Internals

### The "Impossible Travel" Pattern
**Rule**: Two transactions from the same card in different locations with speed > 500 mph.
**Implementation**:
-   **KeyBy**: `card_id`.
-   **State**: Store `last_location` and `last_timestamp`.
-   **Process**:
    1.  On new event, calculate distance and time diff.
    2.  Calculate speed.
    3.  If speed > threshold, emit Alert.
    4.  Update state.

### Handling Late Data
Fraud detection is time-sensitive.
-   **Watermarks**: Crucial. If data arrives 1 hour late, it might trigger a false positive (or negative).
-   **Strategy**: Drop extremely late data? Or process it and send a "Correction" event? Usually, for blocking, we only care about low latency.

### Feature Store Integration
Models need historical features (e.g., "Is this amount > 5x the user's average?").
-   **Online Feature Store (Redis/Cassandra)**: Flink queries this via Async I/O.
-   **Latency**: Async I/O adds latency.
-   **Optimization**: Pre-load hot profiles into Flink Managed Memory (State).

### Performance Implications
-   **State Size**: Storing profile for 100M users is huge. Use **RocksDB** backend.
-   **Throughput**: CEP is CPU intensive. Scale out.
""",

    "Day1_Fraud_Detection_Interview.md": """# Day 1: Fraud Detection - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you update fraud rules without stopping the job?**
    -   *A*: Use **Broadcast State**. Read rules from a side stream (Kafka topic `rules`). Broadcast them to all parallel tasks. Store in `MapState`. Apply active rules to every transaction.

2.  **Q: How do you handle false positives?**
    -   *A*: The system emits a "Probability Score" rather than a binary Block/Allow. If Score > 90, block. If 50-90, SMS verification.

3.  **Q: Why not use a database for this?**
    -   *A*: Latency. Polling a DB for "last 5 transactions" is too slow for 10k TPS. Flink keeps state locally.

### Production Challenges
-   **Challenge**: **Cold Start**.
    -   *Scenario*: New Flink job starts with empty state. It doesn't know the "Average Spend".
    -   *Fix*: **State Bootstrap**. Read historical data from S3/DB and load it into Flink state using the State Processor API before starting the stream.

-   **Challenge**: **Hot Keys**.
    -   *Scenario*: One merchant (e.g., Amazon) has huge volume.
    -   *Fix*: This logic is usually keyed by `CardID`, which is well distributed. If keyed by Merchant, use Salting.

### Troubleshooting Scenarios
**Scenario**: Latency spikes to 2 seconds.
-   *Cause*: Async I/O to Feature Store is timing out.
-   *Fix*: Add local caching (Guava) or scale the Feature Store.
""",

    # --- Day 2: IoT Telemetry ---
    "Day2_IoT_Telemetry_Core.md": """# Day 2: IoT Telemetry Processing

## Core Concepts & Theory

### The Use Case
Process sensor data from 1 Million connected cars.
-   **Input**: GPS, Speed, EngineTemp, FuelLevel (1 msg/sec per car).
-   **Volume**: 1M msg/sec.
-   **Goal**: Real-time dashboard, Geofencing, Predictive Maintenance.

### Architecture
1.  **Edge**: MQTT Broker (e.g., HiveMQ/VerneMQ).
2.  **Bridge**: MQTT -> Kafka Connect -> Kafka.
3.  **Processing**: Flink (Windowed Aggregation, Geofencing).
4.  **Storage**:
    -   **Hot**: TimescaleDB / Druid (Dashboards).
    -   **Cold**: S3 (Data Lake).

### Key Patterns
-   **Downsampling**: Convert 1Hz data to 1-minute averages for long-term storage.
-   **Sessionization**: Detect "Trips" (Ignition ON to Ignition OFF).

### Architectural Reasoning
**MQTT vs Kafka**
-   **MQTT**: Lightweight, good for unreliable networks (IoT devices). Push model.
-   **Kafka**: Heavy, high throughput. Pull model.
-   **Pattern**: Use MQTT for Device-to-Cloud. Bridge to Kafka for Cloud-Internal processing.
""",

    "Day2_IoT_Telemetry_DeepDive.md": """# Day 2: IoT Telemetry - Deep Dive

## Deep Dive & Internals

### Handling Out-of-Order Data
IoT devices often lose connectivity and upload buffered data later.
-   **Event Time**: MUST use device timestamp, not ingestion timestamp.
-   **Watermarks**: Allow for significant lateness (e.g., 1 hour).
-   **Allowed Lateness**: Update previous windows if data arrives late.

### Geofencing Implementation
**Task**: Alert if a car enters a restricted zone.
-   **Naive**: Check every point against every polygon. (O(N*M) - Slow).
-   **Optimized**:
    -   **Spatial Indexing**: Use S2 Geometry or H3 (Uber) to map Lat/Lon to a Cell ID.
    -   **KeyBy**: Cell ID.
    -   **Broadcast**: Broadcast active Geofences to all tasks.

### Compression
IoT data is repetitive.
-   **Delta Encoding**: Store difference from previous value.
-   **Gorilla Compression**: Specialized float compression (used in Prometheus/InfluxDB).
-   **Kafka**: Use Zstd. It works great on JSON/Avro IoT data.

### Performance Implications
-   **Write Load**: 1M writes/sec is heavy for any DB.
-   **Solution**: Flink aggregates (1-min avg) reduce volume by 60x before writing to DB.
""",

    "Day2_IoT_Telemetry_Interview.md": """# Day 2: IoT Telemetry - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you handle 1 million unique keys (Cars) in Flink?**
    -   *A*: Flink scales horizontally. 1M keys is fine. State is sharded. Ensure `maxParallelism` is high enough (e.g., 4096) to allow future scaling.

2.  **Q: How do you detect if a device has stopped sending data?**
    -   *A*: **ProcessFunction** with a Timer. Reset timer on every event. If timer fires, emit "Device Offline" alert.

3.  **Q: Why use a Time Series Database (TSDB) instead of Postgres?**
    -   *A*: TSDBs (Influx, Timescale) are optimized for high write throughput and time-range queries. They compress data much better.

### Production Challenges
-   **Challenge**: **Thundering Herd**.
    -   *Scenario*: Network outage recovers. 1M devices reconnect and upload buffered data simultaneously.
    -   *Fix*: Kafka handles the spike. Flink might lag. Enable Backpressure. Ensure Autoscaling is not too aggressive (don't scale up for a temporary spike).

-   **Challenge**: **Clock Drift**.
    -   *Scenario*: Device clock is wrong (Year 1970).
    -   *Fix*: Filter invalid timestamps at ingestion. Use NTP on devices.

### Troubleshooting Scenarios
**Scenario**: Kafka topic retention is too short for the outage duration.
-   *Cause*: Devices were offline for 7 days. Kafka retention is 3 days. Data lost.
-   *Fix*: Tiered Storage (Infinite Retention).
""",

    # --- Day 3: Clickstream Analytics ---
    "Day3_Clickstream_Analytics_Core.md": """# Day 3: Clickstream Analytics

## Core Concepts & Theory

### The Use Case
Track user behavior on a website (Clicks, Views, AddToCart).
-   **Goal**: Real-time personalization, A/B test monitoring, Funnel analysis.
-   **Volume**: Extremely high (billions of events/day).

### Architecture
1.  **Collection**: JavaScript SDK -> Beacon API -> Nginx -> Kafka.
2.  **Processing**: Flink (Sessionization, Enrichment).
3.  **Serving**:
    -   **Real-time**: Redis (User Profile).
    -   **Analytics**: ClickHouse / Druid / Pinot.

### Key Patterns
-   **Sessionization**: Group events by UserID with a "Session Timeout" (e.g., 30 mins inactivity).
-   **Enrichment**: Join IP address with GeoIP database. Join UserAgent with Device database.

### Architectural Reasoning
**Why ClickHouse/Druid?**
Traditional Data Warehouses (Snowflake/BigQuery) are great for batch but slow/expensive for real-time ingestion. Real-time OLAP engines (ClickHouse, Druid, Pinot) can ingest from Kafka instantly and serve sub-second aggregations.
""",

    "Day3_Clickstream_Analytics_DeepDive.md": """# Day 3: Clickstream - Deep Dive

## Deep Dive & Internals

### Session Windows
**Definition**: A period of activity separated by a gap of inactivity.
**Flink Implementation**: `EventTimeSessionWindows.withGap(Time.minutes(30))`.
-   **Merging**: Session windows merge. If Event A is at 10:00 and Event B is at 10:20, they merge into one session (10:00-10:20). If Event C comes at 10:40, it extends the session.
-   **Trigger**: Fires when the gap passes.

### Bot Detection
Bots skew analytics.
-   **Heuristics**: High request rate, missing UserAgent, known Data Center IPs.
-   **Implementation**: Flink Filter or Side Output.
-   **Bloom Filter**: Check IP against a massive blacklist efficiently.

### User Unification (Identity Resolution)
User starts anonymous (CookieID), then logs in (UserID).
-   **Problem**: Associate previous anonymous events with the UserID.
-   **Solution**:
    1.  **Late Binding**: Do it in the Data Warehouse (Join Cookie table with User table).
    2.  **Real-time**: Harder. Flink needs to update the "Anonymous Profile" with the UserID.

### Performance Implications
-   **Skew**: "Null" UserID (anonymous users) might all go to one partition.
    -   *Fix*: Generate a random UUID for anonymous users instead of Null.
""",

    "Day3_Clickstream_Analytics_Interview.md": """# Day 3: Clickstream - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you calculate "Active Users" (DAU) in real-time?**
    -   *A*: Use a **HyperLogLog** (HLL) sketch. It estimates cardinality with low memory (KB) and high accuracy. Exact count requires storing every UserID (GBs of state).

2.  **Q: How do you handle late-arriving events in Session Windows?**
    -   *A*: Flink can merge the late event into an existing session (and extend it). If the session was already emitted, it can emit an "Update" (Retraction).

3.  **Q: What is the "Funnel Analysis" problem?**
    -   *A*: "View -> Click -> Buy". Order matters. Use Flink CEP or SQL `MATCH_RECOGNIZE`.

### Production Challenges
-   **Challenge**: **High Cardinality Dimensions**.
    -   *Scenario*: Grouping by URL. If you have 1M unique URLs (query params), the result set explodes.
    -   *Fix*: Normalize URLs (strip query params) or use Top-N patterns.

-   **Challenge**: **Ad Blockers**.
    -   *Scenario*: 20% of events are missing.
    -   *Fix*: Server-side tracking (Proxy) instead of Client-side.

### Troubleshooting Scenarios
**Scenario**: Session Window never closes.
-   *Cause*: Watermark is stuck. No new data arriving to push the watermark forward.
-   *Fix*: Use `withIdleness` in watermark strategy.
""",

    # --- Day 4: CDC ---
    "Day4_CDC_Debezium_Core.md": """# Day 4: CDC with Debezium

## Core Concepts & Theory

### The Use Case
Sync a legacy Monolithic Database (MySQL/Postgres) to a Microservice ecosystem or Data Lake.
-   **Goal**: Zero code change in the monolith.
-   **Latency**: Sub-second.

### Architecture
1.  **Source**: MySQL (Binlog) / Postgres (WAL).
2.  **Connector**: Debezium (running in Kafka Connect).
3.  **Transport**: Kafka.
4.  **Sink**: Flink (Transformation) -> ElasticSearch / Snowflake.

### Key Patterns
-   **The Outbox Pattern**: Reliable messaging. Write to `Outbox` table in DB transaction. Debezium captures it.
-   **Strangler Fig**: Migrate functionality from Monolith to Microservices piece by piece using CDC data.

### Architectural Reasoning
**Log-Based CDC vs Query-Based (JDBC)**
-   **JDBC (Polling)**: `SELECT * FROM table WHERE updated_at > last_check`.
    -   *Cons*: Misses hard deletes. Polls add load. Latency.
-   **Log-Based (Debezium)**: Reads the transaction log.
    -   *Pros*: Captures Deletes. Real-time. No impact on DB query engine.
""",

    "Day4_CDC_Debezium_DeepDive.md": """# Day 4: CDC - Deep Dive

## Deep Dive & Internals

### Handling Schema Evolution
What happens when `ALTER TABLE` runs on the source?
-   **Debezium**: Detects the change. Updates Schema Registry. Emits a schema change event.
-   **Downstream**:
    -   **Avro**: If compatible (e.g., add optional field), consumer continues.
    -   **Incompatible**: Consumer fails. Requires manual intervention or a "Schema Router" to send bad data to DLQ.

### Snapshotting
Initial load of a 1TB table.
-   **Locking**: Debezium used to lock the table (Global Read Lock). Bad for prod.
-   **Incremental Snapshot**: New algorithm. Interleaves chunks of snapshot reads with log streaming. No locks.

### Ordering Guarantees
-   **Kafka Partitioning**: Must partition by Primary Key. Ensures all updates to `ID=1` go to the same partition (ordered).
-   **Compaction**: Kafka Log Compaction keeps only the latest state for a key. Crucial for CDC to save space.

### Performance Implications
-   **Toast Columns (Postgres)**: Large text/binary fields are stored separately. Debezium might not see them unless `REPLICA IDENTITY FULL` is set (High DB overhead).
""",

    "Day4_CDC_Debezium_Interview.md": """# Day 4: CDC - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you handle "Hard Deletes" in the source DB?**
    -   *A*: Debezium emits a `DELETE` event (op='d') and a **Tombstone** (Key, Null). The Tombstone tells Kafka to delete the key during compaction.

2.  **Q: What is the "Dual Write" problem and how does CDC fix it?**
    -   *A*: App writes to DB and then tries to publish to Kafka. If app crashes in between, data is inconsistent. CDC ensures consistency by reading the DB log (Source of Truth).

3.  **Q: How do you re-process data if you find a bug in the consumer?**
    -   *A*: Reset consumer offset to 0. But if data was compacted, you lost history.
    -   *Better*: Trigger a new **Ad-hoc Snapshot** in Debezium (using Signals).

### Production Challenges
-   **Challenge**: **WAL Growth**.
    -   *Scenario*: Kafka is down. Debezium stops reading WAL. Postgres keeps WAL segments until confirmed. Disk fills up.
    -   *Fix*: Monitoring! Alert on Replication Slot Lag.

-   **Challenge**: **Sensitive Data**.
    -   *Scenario*: PII in DB log.
    -   *Fix*: Debezium SMT (Single Message Transform) to blacklist columns or hash PII before it hits Kafka.

### Troubleshooting Scenarios
**Scenario**: Debezium connector fails with `invalid authorization specification`.
-   *Cause*: DB password changed.
-   *Fix*: Update connector config (REST API).
""",

    # --- Day 5: Log Aggregation ---
    "Day5_Log_Aggregation_SIEM_Core.md": """# Day 5: Log Aggregation & SIEM

## Core Concepts & Theory

### The Use Case
Centralized Logging and Security Information and Event Management (SIEM).
-   **Input**: Server logs, Firewall logs, App logs.
-   **Goal**: Searchable logs (ELK), Threat Detection (SIEM).

### Architecture
1.  **Agent**: Filebeat / Fluentd / Vector (running on nodes).
2.  **Buffer**: Kafka (The shock absorber).
3.  **Indexer**: Logstash / Vector -> ElasticSearch / Splunk / ClickHouse.
4.  **Detection**: Flink (Real-time rules).

### Key Patterns
-   **Unified Schema**: Convert all logs (Nginx, Syslog, Java) to a common JSON schema (e.g., ECS - Elastic Common Schema).
-   **Hot/Warm/Cold Architecture**:
    -   Hot: SSD (7 days).
    -   Warm: HDD (30 days).
    -   Cold: S3 (Years).

### Architectural Reasoning
**Why Kafka in the middle?**
-   **Backpressure**: If ElasticSearch is slow/down, logs buffer in Kafka. Agents don't crash or drop logs.
-   **Fan-out**: Send logs to Elastic (Search) AND S3 (Archive) AND Flink (Alerting) simultaneously.
""",

    "Day5_Log_Aggregation_SIEM_DeepDive.md": """# Day 5: Log Aggregation - Deep Dive

## Deep Dive & Internals

### Parsing & Normalization
Logs are messy text.
-   **Grok**: Regex-based parsing (slow).
-   **Structured Logging**: App logs in JSON (fast).
-   **Vector (Rust)**: High-performance replacement for Logstash. Can parse/transform in transit.

### Sigma Rules (SIEM)
Standard format for security rules.
-   *Example*: "Detect 5 failed logins from same IP in 1 minute".
-   **Implementation**: Flink CEP or Kafka Streams.
-   **State**: Requires keeping counters per IP.

### Cost Optimization
Logging is expensive (Volume).
-   **Sampling**: Log 100% of Errors, but only 1% of Info/Debug.
-   **Dynamic Level**: Change log level at runtime without restart.
-   **Tiered Storage**: Move data to S3 ASAP. Query directly from S3 (e.g., using ChaosSearch or Athena).

### Performance Implications
-   **Index Rate**: ElasticSearch bottleneck.
    -   *Fix*: Bulk API, remove unnecessary fields, increase refresh interval.
-   **Network**: Compressing logs (Zstd) at the Agent level saves massive bandwidth.
""",

    "Day5_Log_Aggregation_SIEM_Interview.md": """# Day 5: Log Aggregation - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you handle multiline logs (e.g., Java Stack Traces)?**
    -   *A*: Handle at the Agent level (Filebeat). Configure it to aggregate lines starting with whitespace into the previous event.

2.  **Q: Why is "At-Least-Once" acceptable for logs?**
    -   *A*: Duplicates are annoying but usually fine for search/debugging. Data loss is worse.

3.  **Q: How do you secure the log pipeline?**
    -   *A*: mTLS between Agents and Kafka. Encryption at rest. ACLs. Mask PII (Credit Cards) at the source.

### Production Challenges
-   **Challenge**: **The "Debug" Flood**.
    -   *Scenario*: Dev leaves DEBUG logging on. 1TB/hour. Cluster crashes.
    -   *Fix*: Quotas per topic/user. Rate limiting at the Agent.

-   **Challenge**: **Field Mapping Explosion**.
    -   *Scenario*: Every app sends different field names (`user_id`, `userId`, `uid`). ElasticSearch mapping explodes.
    -   *Fix*: Enforce schema at the Ingest Pipeline. Drop unknown fields.

### Troubleshooting Scenarios
**Scenario**: Logs are delayed by 15 minutes.
-   *Cause*: Logstash cannot keep up with Kafka.
-   *Fix*: Scale Logstash horizontally (Consumer Group). Check ES indexing latency.
"""
}

print("🚀 Populating Week 9 Case Studies Content (Detailed)...")

for filename, content in content_map.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")

print("✅ Week 9 Content Population Complete!")
