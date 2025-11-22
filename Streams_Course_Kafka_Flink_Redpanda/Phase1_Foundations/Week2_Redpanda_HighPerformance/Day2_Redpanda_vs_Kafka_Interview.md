# Day 2: Redpanda vs Kafka - Interview Prep

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
