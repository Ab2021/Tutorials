# Monitoring and Logging Basics

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Observability, including:
- **Concepts**: The Three Pillars (Metrics, Logs, Traces).
- **Metrics**: Collecting time-series data with **Prometheus**.
- **Visualization**: Creating dashboards with **Grafana**.
- **Logging**: Aggregating logs with the **ELK Stack** or **Loki**.
- **Alerting**: Designing effective alerts that don't cause fatigue.

---

## 📖 Theoretical Concepts

### 1. What is Observability?

Monitoring tells you *if* the system is down. Observability tells you *why*.

**The Three Pillars:**
1.  **Metrics**: "What is happening?" (e.g., CPU is at 90%). Aggregatable, cheap to store.
2.  **Logs**: "Why is it happening?" (e.g., "NullPointerException at line 42"). Detailed, expensive to store.
3.  **Traces**: "Where is it happening?" (e.g., Service A called Service B, which took 5s).

### 2. Prometheus (Metrics)

Prometheus is the industry standard for cloud-native monitoring.
- **Pull Model**: Prometheus scrapes metrics from your app's `/metrics` endpoint.
- **TSDB**: Time-Series Database optimized for storing numbers over time.
- **PromQL**: Powerful query language.
  - `rate(http_requests_total[5m])`: Rate of requests over the last 5 minutes.

**Metric Types:**
- **Counter**: Only goes up (e.g., Total Requests).
- **Gauge**: Goes up and down (e.g., Memory Usage).
- **Histogram**: Distribution of values (e.g., Request Duration).

### 3. Grafana (Visualization)

Grafana connects to data sources (Prometheus, MySQL, Loki) and visualizes them.
- **Dashboards**: Collections of panels.
- **Panels**: Graphs, Gauges, Tables.
- **Variables**: Dropdowns to filter data (e.g., Select Server).

### 4. Logging Stacks

- **ELK Stack (Elasticsearch, Logstash, Kibana)**: Powerful, full-text search. Heavy resource usage.
- **PLG Stack (Promtail, Loki, Grafana)**: "Like Prometheus, but for logs". Indexes labels, not content. Lightweight.

---

## 🔧 Practical Examples

### Prometheus Configuration (`prometheus.yml`)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']
```

### PromQL Queries

```promql
# CPU Usage Percentage
100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory Usage
node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
```

### Logging Best Practices (JSON)

Don't log unstructured text. Log JSON.

**Bad:**
`User 123 logged in at 10:00`

**Good:**
```json
{
  "event": "user_login",
  "user_id": 123,
  "timestamp": "2023-10-27T10:00:00Z",
  "status": "success"
}
```

---

## 🎯 Hands-on Labs

- [Lab 9.1: Introduction to Monitoring Concepts](./labs/lab-09.1-intro-monitoring.md)
- [Lab 9.2: Prometheus Setup](./labs/lab-09.2-prometheus-setup.md)
- [Lab 9.3: Grafana Dashboards](./labs/lab-09.3-grafana-dashboards.md)
- [Lab 9.4: ELK Stack Basics (Logging)](./labs/lab-09.4-elk-stack.md)
- [Lab 9.5: Monitoring Capstone Project](./labs/lab-09.5-monitoring-project.md)

---

## 📚 Additional Resources

### Official Documentation
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)

### Books
- "Prometheus: Up & Running" by Brian Brazil.
- "Site Reliability Engineering" (Google SRE Book).

---

## 🔑 Key Takeaways

1.  **USE Method**: For every resource, check **U**tilization, **S**aturation, and **E**rrors.
2.  **RED Method**: For every service, check **R**ate, **E**rrors, and **D**uration.
3.  **Alerts**: Alert on symptoms (High Latency), not causes (High CPU).
4.  **Correlation**: Use Trace IDs in your logs to link them to traces.

---

## ⏭️ Next Steps

1.  Complete the labs to set up your own monitoring stack.
2.  Proceed to **[Module 10: Cloud Fundamentals](../module-10-cloud-fundamentals-aws/README.md)** to apply these skills in the cloud.
