# Day 3: Multi-DC - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between RPO and RTO?**
    -   *A*: RPO (Recovery Point Objective) = How much data can we lose? (e.g., 5 mins). RTO (Recovery Time Objective) = How fast can we be back up? (e.g., 1 hour).

2.  **Q: How does MirrorMaker 2 handle infinite loops in Active-Active?**
    -   *A*: It adds a header to the message indicating the source cluster. It filters out messages that originated from the target cluster.

3.  **Q: Why is Offset Translation necessary?**
    -   *A*: Because messages might be dropped or compacted differently in the two clusters, so Offset 500 in Source might be Offset 450 in Target.

### Production Challenges
-   **Challenge**: **Split Brain**.
    -   *Scenario*: Network partition. Both DCs think they are the "Active" one.
    -   *Fix*: Use a tie-breaker (Witness) or manual failover.

### Troubleshooting Scenarios
**Scenario**: Replication Lag is high.
-   *Cause*: WAN link saturation.
-   *Fix*: Increase batch size, compression, or bandwidth.
