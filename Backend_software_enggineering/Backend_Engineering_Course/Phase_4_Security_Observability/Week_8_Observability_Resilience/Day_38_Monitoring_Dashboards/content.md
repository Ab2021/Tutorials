# Day 38: Monitoring & Dashboards

## 1. The Dashboard

A picture is worth 1000 logs.
*   **Prometheus**: The database (Time Series).
*   **Grafana**: The UI (Charts).

### 1.1 Prometheus Architecture (Pull Model)
*   **Exporter**: Your app exposes `/metrics` (text format).
*   **Scraper**: Prometheus visits `/metrics` every 15s and stores the data.
*   **Benefit**: If your app is overloaded, it doesn't crash trying to push metrics. Prometheus just fails to scrape (which is a signal itself).

---

## 2. The 4 Golden Signals (Google SRE)

What should you monitor?
1.  **Latency**: Time taken to serve a request. (p50, p95, p99).
2.  **Traffic**: Demand on the system (Req/sec).
3.  **Errors**: Rate of requests failing (5xx).
4.  **Saturation**: How "full" is the service? (CPU, Memory, Queue Depth).

---

## 3. PromQL (Prometheus Query Language)

*   `http_requests_total`: The raw counter.
*   `rate(http_requests_total[5m])`: Requests per second (averaged over 5m).
*   `sum(rate(http_requests_total[5m])) by (service)`: Aggregated by service.

---

## 4. Alerting (SLI, SLO, SLA)

Don't alert on "CPU > 80%". Alert on "User Experience".

*   **SLI (Indicator)**: The metric. (Latency).
*   **SLO (Objective)**: The goal. (99% of requests < 200ms).
*   **SLA (Agreement)**: The contract. (If we miss SLO, we pay you money).
*   **Alert**: "Burn Rate". If we are burning through our Error Budget too fast, wake someone up.

---

## 5. Summary

Today we built the cockpit.
*   **Prometheus**: Collects the dots.
*   **Grafana**: Connects the dots.
*   **Golden Signals**: The vital signs.

**Tomorrow (Day 39)**: We will use these metrics to make things fast. **Performance Tuning & Caching**.
