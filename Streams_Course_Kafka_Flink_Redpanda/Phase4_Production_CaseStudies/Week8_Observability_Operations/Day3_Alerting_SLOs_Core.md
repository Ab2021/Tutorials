# Day 3: Alerting & SLOs

## Core Concepts & Theory

### The Hierarchy of Reliability
1.  **SLI (Service Level Indicator)**: The metric. "Latency is 50ms".
2.  **SLO (Service Level Objective)**: The goal. "Latency should be < 100ms for 99% of requests".
3.  **SLA (Service Level Agreement)**: The contract. "If we miss the SLO, we pay you back".

### Designing Good Alerts
-   **Page**: Immediate action required. User is impacted. (e.g., "Site Down", "Data Loss").
-   **Ticket**: Action required eventually. (e.g., "Disk 80% full").
-   **Log**: No action required. (e.g., "Job restarted successfully").

### Error Budgets
The allowed amount of unreliability.
-   SLO: 99.9% Availability.
-   Error Budget: 0.1% (43 minutes / month).
-   **Philosophy**: If you have budget left, you can ship risky features. If budget is exhausted, freeze changes and focus on stability.

### Architectural Reasoning
**Symptom-Based vs Cause-Based Alerting**
-   **Cause-Based**: "CPU is high". (Bad. High CPU might be fine).
-   **Symptom-Based**: "Latency is high". (Good. User is suffering).
-   **Rule**: Page on Symptoms. Use Causes for debugging.

### Key Components
-   **Prometheus Alertmanager**: Deduplicates, groups, and routes alerts (Slack, PagerDuty).
-   **Grafana Alerting**: Visual alerting.
