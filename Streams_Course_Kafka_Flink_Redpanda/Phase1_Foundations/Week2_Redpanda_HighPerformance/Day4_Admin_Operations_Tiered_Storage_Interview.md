# Day 4: Admin Operations - Interview Prep

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
