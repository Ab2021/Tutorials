# Day 5: SRE - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you calculate End-to-End Latency?**
    -   *A*: Inject a timestamp at the source. Compare with current time at the sink. (Clock synchronization required).

2.  **Q: What is the difference between Liveness and Readiness probes?**
    -   *A*: Liveness = "Am I alive?" (Restart if no). Readiness = "Can I take traffic?" (Remove from load balancer if no).

3.  **Q: Why is high CPU not always bad?**
    -   *A*: If throughput is high and latency is low, high CPU means we are utilizing resources efficiently.

### Production Challenges
-   **Challenge**: **Missing Metrics**.
    -   *Scenario*: Job fails silently. No alert.
    -   *Fix*: Alert on "Absence of Data" (Heartbeat).

### Troubleshooting Scenarios
**Scenario**: Prometheus OOM.
-   *Cause*: Cardinality explosion (someone added `pod_ip` or `session_id` as a label).
-   *Fix*: Drop the label in relabeling rules.
