# Day 4: Operator State - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: When would you use Operator State instead of Keyed State?**
    -   *A*: For sources (offsets), sinks (transaction handles), or when you need to broadcast configuration/rules to all nodes.

2.  **Q: How does Broadcast State behave during rescaling?**
    -   *A*: It is replicated. If you scale from 2 to 4 nodes, the new 2 nodes get a copy of the broadcast state.

3.  **Q: What is the difference between `Union` and `Even-Split` redistribution?**
    -   *A*: `Union` sends the full state to everyone. `Even-Split` partitions it.

### Production Challenges
-   **Challenge**: **Broadcast State OOM**.
    -   *Scenario*: Broadcasting a large lookup table (10GB).
    -   *Fix*: Use an external KV store (Redis/Cassandra) with async I/O instead of Broadcast State.

### Troubleshooting Scenarios
**Scenario**: Rules are not applying to some keys.
-   *Cause*: The control stream might be partitioned (Keyed) instead of Broadcast. Ensure you call `.broadcast()`.
