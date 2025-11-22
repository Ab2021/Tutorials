# Day 5: Log Aggregation - Interview Prep

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
