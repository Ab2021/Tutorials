# Day 2: Kafka Architecture - Interview Prep

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
