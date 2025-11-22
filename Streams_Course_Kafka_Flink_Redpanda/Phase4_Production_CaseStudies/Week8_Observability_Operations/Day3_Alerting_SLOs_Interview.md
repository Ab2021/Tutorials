# Day 3: Alerting - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Availability and Reliability?**
    -   *A*: **Availability**: Is the system up? (Uptime). **Reliability**: Is the system doing the right thing? (Correctness, no data loss).

2.  **Q: How do you calculate availability for a streaming pipeline?**
    -   *A*: Harder than request/response. Usually defined as "Freshness" (Data is available within X seconds) or "Completeness" (No data loss).

3.  **Q: Describe an alerting strategy for a Kafka cluster.**
    -   *A*: Page on: UnderReplicatedPartitions > 0, OfflinePartitions > 0, ConsumerLag > Threshold (for critical apps). Ticket on: Disk > 80%, Broker skewed.

### Production Challenges
-   **Challenge**: **Alert Flapping**.
    -   *Scenario*: CPU goes 91% -> 89% -> 91%. Alert fires, resolves, fires.
    -   *Fix*: Use **Hysteresis** (Fire at 90%, Resolve at 80%) or `for: 5m` duration.

-   **Challenge**: **On-Call Burnout**.
    -   *Scenario*: Too many non-actionable alerts.
    -   *Fix*: Delete any alert that doesn't require immediate action. Move to tickets/dashboards.

### Troubleshooting Scenarios
**Scenario**: Alertmanager is sending emails but not Slack messages.
-   *Cause*: Slack Webhook URL expired or network firewall blocking egress.
-   *Fix*: Check Alertmanager logs.
