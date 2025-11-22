# Day 5: SRE & Observability

## Core Concepts & Theory

### The 4 Golden Signals
1.  **Latency**: End-to-End latency, Request latency.
2.  **Traffic**: Throughput (Messages/sec, Bytes/sec).
3.  **Errors**: Failed requests, Exceptions, Dead Letter Queue rate.
4.  **Saturation**: CPU, Disk, Network utilization.

### Service Level Objectives (SLO)
-   **SLH (Indicator)**: "99th percentile latency".
-   **SLO (Objective)**: "99% of requests < 100ms".
-   **SLA (Agreement)**: Contract with penalty.

### Architectural Reasoning
**Whitebox vs Blackbox Monitoring**
-   **Whitebox**: Metrics emitted by the app (JMX, Prometheus). "I am processing 100 msg/sec".
-   **Blackbox**: Synthetic checks. "Can I produce to this topic?". "Is the consumer lagging?".

### Key Components
-   **Prometheus/Grafana**: Standard stack.
-   **OpenTelemetry**: Tracing.
