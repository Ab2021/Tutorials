# Day 5: Reliability & Durability - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if `min.insync.replicas=2` and you only have 1 ISR?**
    -   *A*: The producer receives a `NOT_ENOUGH_REPLICAS` exception. The partition becomes read-only (effectively).

2.  **Q: How does Kafka prevent data loss?**
    -   *A*: Replication, `acks=all`, `min.insync.replicas > 1`, and `unclean.leader.election.enable=false`.

3.  **Q: What is an "Unclean Leader Election"?**
    -   *A*: Electing a replica that was NOT in the ISR (i.e., it is missing data). It restores availability but causes data loss. Default is `false`.

### Production Challenges
-   **Challenge**: **Data Loss**.
    -   *Scenario*: `acks=1`, Leader crashes before replicating to follower.
    -   *Fix*: Use `acks=all`.

-   **Challenge**: **Cluster unavailable for writes**.
    -   *Scenario*: 2 out of 3 brokers crash. `min.insync.replicas=2`.
    -   *Fix*: Bring brokers back up. Or dynamically lower `min.insync.replicas` (risky).

### Troubleshooting Scenarios
**Scenario**: Producer latency spikes.
-   **Check**: Is one of the followers slow? (Slow disk/network). It might be dragging down the `acks=all` latency.
-   **Check**: Check `UnderReplicatedPartitions` metric.
