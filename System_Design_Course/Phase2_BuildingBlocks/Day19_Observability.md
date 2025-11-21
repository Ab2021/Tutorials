# Day 19: Observability

## 1. Monitoring vs Observability
*   **Monitoring:** "Is the system healthy?" (Dashboard: CPU 80%).
*   **Observability:** "Why is the system broken?" (Debugging: Which request caused the spike?).

## 2. The Three Pillars
### Logs (Events)
*   **What:** "Something happened."
*   **Format:** JSON/Text. `{"level": "error", "msg": "DB timeout"}`.
*   **Tools:** ELK Stack (Elasticsearch, Logstash, Kibana), Splunk.
*   **Cost:** High (Storage).

### Metrics (Aggregates)
*   **What:** "Counts and Gauges over time."
*   **Format:** `http_requests_total{status="500"} = 42`.
*   **Tools:** Prometheus, Grafana, Datadog.
*   **Cost:** Low (Fixed size).

### Traces (Context)
*   **What:** "The path of a request across services."
*   **Format:** Spans. `Service A (10ms) -> Service B (50ms)`.
*   **Tools:** Jaeger, Zipkin, OpenTelemetry.
*   **Cost:** Medium (Sampling needed).

## 3. Golden Signals (Google SRE)
1.  **Latency:** Time to serve request.
2.  **Traffic:** Demand (QPS).
3.  **Errors:** Rate of failure (5xx).
4.  **Saturation:** How "full" is the service (CPU/Memory/Disk).
