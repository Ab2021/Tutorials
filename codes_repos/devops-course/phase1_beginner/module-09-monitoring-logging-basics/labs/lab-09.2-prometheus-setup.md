# Lab 9.2: Prometheus Setup

## 🎯 Objective

Install and configure Prometheus. You will run Prometheus in Docker, configure it to scrape itself, and query the metrics using PromQL.

## 📋 Prerequisites

-   Docker installed.

## 📚 Background

### Prometheus Architecture
-   **Server**: Scrapes (pulls) metrics from targets. Stores them in a Time Series Database (TSDB).
-   **Targets**: Services exposing a `/metrics` endpoint.
-   **Exporters**: Helpers that translate metrics (e.g., Node Exporter for Linux metrics).

### PromQL
The query language.
`up` -> Is the target up?
`rate(http_requests_total[5m])` -> Rate of requests.

---

## 🔨 Hands-On Implementation

### Part 1: Configuration ⚙️

1.  **Create `prometheus.yml`:**
    ```yaml
    global:
      scrape_interval: 15s

    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
    ```

### Part 2: Run Prometheus 🐳

1.  **Start Container:**
    ```bash
    docker run -d \
      -p 9090:9090 \
      -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
      --name prometheus \
      prom/prometheus
    ```

2.  **Verify:**
    Open `http://localhost:9090`.

### Part 3: Querying 🔍

1.  **Check Status:**
    Go to **Status** -> **Targets**.
    You should see `prometheus` (UP).

2.  **Graphing:**
    Go to **Graph**.
    Type `prometheus_http_requests_total`.
    Click **Execute**.
    Switch to **Graph** tab.
    *Result:* You see a line chart of requests.

### Part 4: Node Exporter (Monitoring the Host) 🖥️

Prometheus can't see your Host CPU by default. We need **Node Exporter**.

1.  **Run Node Exporter:**
    ```bash
    docker run -d \
      -p 9100:9100 \
      --name node-exporter \
      prom/node-exporter
    ```

2.  **Update `prometheus.yml`:**
    Add a new job.
    ```yaml
      - job_name: 'node'
        static_configs:
          - targets: ['host.docker.internal:9100']
    ```
    *Note:* Use your local IP if `host.docker.internal` fails.

3.  **Restart Prometheus:**
    ```bash
    docker restart prometheus
    ```

4.  **Query Host Metrics:**
    Query: `node_memory_MemFree_bytes`.
    *Result:* Shows your free RAM.

---

## 🎯 Challenges

### Challenge 1: Rate Query (Difficulty: ⭐⭐)

**Task:**
The raw counter `prometheus_http_requests_total` just goes up forever.
Write a PromQL query to show the **Rate** of requests per second over the last 1 minute.
*Hint: `rate(...[1m])`*

### Challenge 2: Alerting Rule (Difficulty: ⭐⭐⭐)

**Task:**
Research how to add an `alerting_rules.yml` file.
Create a rule that fires if `up == 0` (Instance down).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```promql
rate(prometheus_http_requests_total[1m])
```

**Challenge 2:**
`rules.yml`:
```yaml
groups:
- name: example
  rules:
  - alert: InstanceDown
    expr: up == 0
    for: 1m
```
Add `rule_files: ["rules.yml"]` to `prometheus.yml`.
</details>

---

## 🔑 Key Takeaways

1.  **Pull Model**: Prometheus pulls metrics. This is different from Push (Datadog/NewRelic).
2.  **Exporters**: There is an exporter for everything (MySQL, Nginx, AWS).
3.  **Ephemeral Data**: Prometheus is not for long-term storage (years). Use Thanos or Cortex for that.

---

## ⏭️ Next Steps

Graphs in Prometheus are ugly. Let's make them pretty.

Proceed to **Lab 9.3: Grafana Dashboards**.
