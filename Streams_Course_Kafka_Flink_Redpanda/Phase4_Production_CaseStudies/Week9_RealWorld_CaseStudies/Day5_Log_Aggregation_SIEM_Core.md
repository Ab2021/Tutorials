# Day 5: Log Aggregation & SIEM

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
