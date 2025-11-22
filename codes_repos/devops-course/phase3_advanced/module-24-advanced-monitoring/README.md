# Advanced Monitoring & APM

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of enterprise-grade monitoring, including:
- **APM (Application Performance Monitoring)**: Deep code-level insights with tools like New Relic/Datadog.
- **Prometheus at Scale**: Using the Prometheus Operator and Thanos for long-term storage.
- **Error Budgets**: Implementing SRE practices to balance velocity and reliability.
- **Synthetic Monitoring**: Proactively testing your app from the user's perspective.
- **Capacity Planning**: Predicting future resource needs based on trends.

---

## 📖 Theoretical Concepts

### 1. APM (Application Performance Monitoring)

Goes beyond infrastructure metrics to show *what your code is doing*.
- **Distributed Tracing**: See the full request path across microservices.
- **Code Profiling**: Identify slow functions (e.g., "This SQL query takes 2s").
- **Error Tracking**: Capture stack traces and context for exceptions.

### 2. Prometheus Operator

Managing Prometheus manually in K8s is tedious. The Operator automates it.
- **ServiceMonitor**: A CRD that tells Prometheus "Scrape this service".
- **PrometheusRule**: A CRD for Recording/Alerting rules.
- **Automatic Discovery**: Prometheus finds new pods automatically.

### 3. Thanos (Long-Term Storage)

Prometheus stores data for 15 days by default. Thanos extends this to years.
- **Sidecar**: Uploads Prometheus data to S3.
- **Query**: Unified query interface across multiple Prometheus instances.
- **Compactor**: Downsamples old data (5m resolution -> 1h resolution).

### 4. Error Budgets

- **SLO**: "99.9% of requests should succeed" (allows 0.1% to fail).
- **Error Budget**: The 0.1%. If you burn through it, freeze feature releases and focus on reliability.
- **Burn Rate**: How fast you're consuming the budget. "At this rate, we'll run out in 3 days."

### 5. Synthetic Monitoring

Don't wait for users to find bugs. Simulate user behavior 24/7.
- **Uptime Checks**: Ping `/health` every minute.
- **Transaction Monitoring**: Selenium script that logs in, adds to cart, checks out.

---

## 🔧 Practical Examples

### ServiceMonitor (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: my-app
spec:
  selector:
    matchLabels:
      app: my-app
  endpoints:
  - port: metrics
    interval: 30s
```

### Thanos Sidecar

```yaml
- name: thanos-sidecar
  image: quay.io/thanos/thanos:v0.31.0
  args:
  - sidecar
  - --prometheus.url=http://localhost:9090
  - --objstore.config-file=/etc/thanos/objstore.yml
```

### Error Budget Calculation

```python
# SLO: 99.9% success rate over 30 days
total_requests = 10_000_000
allowed_failures = total_requests * 0.001  # 10,000
actual_failures = 5_000

remaining_budget = allowed_failures - actual_failures  # 5,000
budget_percentage = (remaining_budget / allowed_failures) * 100  # 50%
```

---

## 🎯 Hands-on Labs

- [Lab 24.1: Prometheus Operator](./labs/lab-24.1-prometheus-operator.md)
- [Lab 24.2: Thanos (Long-Term Storage)](./labs/lab-24.2-thanos.md)
- [Lab 24.3: Custom Instrumentation](./labs/lab-24.3-custom-instrumentation.md)
- [Lab 24.4: Slo Sli Sla](./labs/lab-24.4-slo-sli-sla.md)
- [Lab 24.5: Error Budgets](./labs/lab-24.5-error-budgets.md)
- [Lab 24.6: Synthetic Monitoring](./labs/lab-24.6-synthetic-monitoring.md)
- [Lab 24.7: Real User Monitoring](./labs/lab-24.7-real-user-monitoring.md)
- [Lab 24.8: Performance Analysis](./labs/lab-24.8-performance-analysis.md)
- [Lab 24.9: Capacity Planning](./labs/lab-24.9-capacity-planning.md)
- [Lab 24.10: Monitoring At Scale](./labs/lab-24.10-monitoring-at-scale.md)

---

## 📚 Additional Resources

### Official Documentation
- [Prometheus Operator](https://prometheus-operator.dev/)
- [Thanos Documentation](https://thanos.io/)

### Books
- "Site Reliability Engineering" (Google SRE Book) - Chapter on Monitoring.

---

## 🔑 Key Takeaways

1.  **Observe the User Experience**: Metrics about CPU are useless if users are happy. Metrics about latency matter.
2.  **Cardinality is the Enemy**: Don't create metrics with high-cardinality labels (e.g., User ID). It will kill Prometheus.
3.  **Error Budgets Drive Behavior**: If you have budget left, ship features. If not, fix bugs.
4.  **Synthetic != Real**: Synthetic monitoring catches obvious failures. Real User Monitoring (RUM) shows actual user pain.

---

## ⏭️ Next Steps

1.  Complete the labs to build a production-grade monitoring stack.
2.  Proceed to **[Module 25: Chaos Engineering](../module-25-chaos-engineering/README.md)** to test your system's resilience.
