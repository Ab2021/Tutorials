# Advanced Monitoring & Observability

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of advanced observability, including:
- **Tracing**: Implementing Distributed Tracing with **Jaeger** and **OpenTelemetry**.
- **SRE Principles**: Defining and measuring **SLIs**, **SLOs**, and **SLAs**.
- **Advanced Metrics**: Using Prometheus **Recording Rules** and **Service Discovery**.
- **Alerting**: Managing alert fatigue with **Alertmanager** grouping and inhibition.
- **Instrumentation**: Exposing custom business metrics from your application code.

---

## 📖 Theoretical Concepts

### 1. Distributed Tracing

In microservices, a single user request might hit 10 different services. Logs are isolated. Tracing connects them.
- **Trace**: The journey of a request.
- **Span**: A single unit of work (e.g., "Database Query").
- **Context Propagation**: Passing the Trace ID via HTTP Headers (`x-b3-traceid`).

### 2. SRE Concepts (SLI/SLO/SLA)

- **SLI (Service Level Indicator)**: The metric (e.g., "Latency"). "What is the latency right now?"
- **SLO (Service Level Objective)**: The goal (e.g., "99% of requests < 200ms"). "What should it be?"
- **SLA (Service Level Agreement)**: The contract (e.g., "If we miss the SLO, we pay you"). "What happens if we fail?"
- **Error Budget**: 100% - SLO. The amount of unreliability you are allowed.

### 3. Advanced Prometheus

- **Service Discovery**: Prometheus asks Kubernetes "Where are the pods?" instead of hardcoding IPs.
- **Recording Rules**: Pre-calculate expensive queries (e.g., `rate(http_requests_total[1h])`) and save the result as a new time series. Speeds up dashboards.

### 4. Alertmanager

Handles alerts sent by Prometheus.
- **Grouping**: "The database is down" causes 100 apps to fail. Send 1 alert, not 101.
- **Inhibition**: If "Data Center is Down" is firing, suppress "Server A is Down".
- **Routing**: Send "Critical" to PagerDuty, "Warning" to Slack.

---

## 🔧 Practical Examples

### Custom Metrics (Python)

```python
from prometheus_client import start_http_server, Counter

# Define a metric
REQUESTS = Counter('app_requests_total', 'Total app requests')

def process_request():
    REQUESTS.inc()  # Increment
    # ... logic ...

if __name__ == '__main__':
    start_http_server(8000)  # Expose on /metrics
```

### Prometheus Recording Rule

```yaml
groups:
  - name: example
    rules:
    - record: job:http_inprogress_requests:sum
      expr: sum by (job) (http_inprogress_requests)
```

### Alertmanager Config

```yaml
route:
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'slack-notifications'

receivers:
- name: 'slack-notifications'
  slack_configs:
  - channel: '#alerts'
```

---

## 🎯 Hands-on Labs

- [Lab 16.1: Distributed Tracing with Jaeger](./labs/lab-16.1-distributed-tracing.md)
- [Lab 16.2: Custom Metrics & SLOs](./labs/lab-16.2-custom-metrics-slos.md)
- [Lab 16.3: Service Discovery](./labs/lab-16.3-service-discovery.md)
- [Lab 16.4: Alertmanager](./labs/lab-16.4-alertmanager.md)
- [Lab 16.5: Grafana Advanced](./labs/lab-16.5-grafana-advanced.md)
- [Lab 16.6: Distributed Tracing](./labs/lab-16.6-distributed-tracing.md)
- [Lab 16.7: Jaeger Setup](./labs/lab-16.7-jaeger-setup.md)
- [Lab 16.8: Metrics Best Practices](./labs/lab-16.8-metrics-best-practices.md)
- [Lab 16.9: Slo Sli Setup](./labs/lab-16.9-slo-sli-setup.md)
- [Lab 16.10: Observability Patterns](./labs/lab-16.10-observability-patterns.md)

---

## 📚 Additional Resources

### Official Documentation
- [OpenTelemetry](https://opentelemetry.io/)
- [Google SRE Book - SLOs](https://sre.google/workbook/implementing-slos/)

### Tools
- [Jaeger](https://www.jaegertracing.io/)
- [Prometheus Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/)

---

## 🔑 Key Takeaways

1.  **Don't Alert on Everything**: Alert on symptoms that affect users (Latency, Errors).
2.  **Context is King**: A log message without a Trace ID is a needle in a haystack.
3.  **Define SLOs**: You can't improve reliability if you don't measure it against a target.
4.  **Code Instrumentation**: The best metrics come from inside the application, not the OS.

---

## ⏭️ Next Steps

1.  Complete the labs to instrument your application.
2.  Proceed to **[Module 17: Logging & Log Management](../module-17-logging-log-management/README.md)** to master log aggregation.
