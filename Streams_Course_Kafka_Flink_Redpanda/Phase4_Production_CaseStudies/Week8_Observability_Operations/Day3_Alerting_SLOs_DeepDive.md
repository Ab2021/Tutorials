# Day 3: Alerting - Deep Dive

## Deep Dive & Internals

### Multi-Window Multi-Burn-Rate Alerts
How to alert fast enough to save the SLO, but not so fast it's noisy?
-   **Burn Rate**: How fast are we consuming the error budget?
    -   Burn Rate 1 = Consuming budget exactly at the limit (will exhaust in 30 days).
    -   Burn Rate 14.4 = Will exhaust in 2 days.
-   **Strategy**:
    -   **Page**: High Burn Rate (14.4) over Short Window (1h) AND Long Window (6h).
    -   **Ticket**: Low Burn Rate (6) over Long Window (3 days).

### Alert Grouping & Inhibition
-   **Grouping**: If 100 brokers go down, don't send 100 pages. Send 1 page: "Cluster Down (100 brokers)".
-   **Inhibition**: If "Data Center Power Outage" is firing, suppress "Server Down" alerts.

### Advanced Reasoning
**The "Null" Alert**
What if the monitoring system itself is down?
-   **Dead Man's Switch**: Have an external system (e.g., PagerDuty or a Lambda) expect a "heartbeat" from Alertmanager every minute. If missing, Page.

### Performance Implications
-   **Evaluation Interval**: How often to check rules. 1m is standard. 10s is aggressive.
-   **State**: Alertmanager stores active alerts in memory/disk. High churn alerts can overload it.
