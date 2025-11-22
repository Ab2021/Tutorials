# Lab 17.2: Loki & Grafana (PLG Stack)

## 🎯 Objective

"Like Prometheus, but for Logs." **Loki** is a log aggregation system designed to be cost-effective. It doesn't index the text of the logs (like Elasticsearch does), it only indexes the **labels**.

## 📋 Prerequisites

-   Docker & Docker Compose.
-   Grafana (from Module 9/16).

## 📚 Background

### The PLG Stack
1.  **Promtail**: The Agent. Runs on every node. Tails log files and sends them to Loki.
2.  **Loki**: The Server. Stores logs.
3.  **Grafana**: The UI. Queries Loki.

### LogQL
The query language.
`{app="nginx"} |= "error"` -> Find logs from nginx containing "error".

---

## 🔨 Hands-On Implementation

### Part 1: The Infrastructure 🏗️

1.  **Create `docker-compose.yml`:**
    ```yaml
    version: "3"
    services:
      loki:
        image: grafana/loki:2.6.1
        ports:
          - "3100:3100"
        command: -config.file=/etc/loki/local-config.yaml

      promtail:
        image: grafana/promtail:2.6.1
        volumes:
          - /var/log:/var/log
          - ./promtail-config.yaml:/etc/promtail/config.yaml
        command: -config.file=/etc/promtail/config.yaml

      grafana:
        image: grafana/grafana:latest
        ports:
          - "3000:3000"
    ```

### Part 2: Promtail Config ⚙️

1.  **Create `promtail-config.yaml`:**
    ```yaml
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    positions:
      filename: /tmp/positions.yaml

    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
      - job_name: system
        static_configs:
        - targets:
            - localhost
          labels:
            job: varlogs
            __path__: /var/log/*.log
    ```
    *Note:* This tells Promtail to read any `.log` file in `/var/log` (on your host, mapped via Docker).

### Part 3: Run & Query 🔍

1.  **Start:**
    ```bash
    docker-compose up -d
    ```

2.  **Setup Grafana:**
    -   Login (`admin`/`admin`).
    -   Add Data Source -> **Loki**.
    -   URL: `http://loki:3100`.
    -   Save & Test.

3.  **Explore:**
    -   Go to **Explore** (Compass icon).
    -   Select **Loki** data source.
    -   Click **Log browser**.
    -   Select label `job` -> `varlogs`.
    -   Click **Show logs**.

### Part 4: LogQL ⚡

1.  **Filter:**
    In the query bar:
    ```logql
    {job="varlogs"} |= "error"
    ```
    *Result:* Only shows lines containing "error".

2.  **Count Errors (Metric from Logs):**
    Loki can generate metrics!
    ```logql
    count_over_time({job="varlogs"} |= "error" [1m])
    ```
    *Result:* A graph showing the rate of errors over time.

---

## 🎯 Challenges

### Challenge 1: JSON Parsing (Difficulty: ⭐⭐⭐)

**Task:**
If your logs are JSON, use the `| json` parser in LogQL.
Query: `{job="varlogs"} | json | level="ERROR"`.
*Goal:* Filter based on a specific JSON field, not just text search.

### Challenge 2: Alerting (Difficulty: ⭐⭐⭐)

**Task:**
Create a Grafana Alert based on a Loki query.
"Alert if the number of 'Connection Refused' logs > 10 in 1 minute".

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Requires the log line to be valid JSON.
`{...} | json` extracts keys as labels.
Then you can filter: `... | level="ERROR"`.
</details>

---

## 🔑 Key Takeaways

1.  **Cost**: Loki is much cheaper than Elasticsearch because it doesn't index the content.
2.  **Context**: In Grafana, you can split screen: Metrics on left, Logs on right.
3.  **Live Tailing**: You can "tail" logs in real-time in the Grafana UI.

---

## ⏭️ Next Steps

We have Logs and Metrics. Now let's secure the pipeline.

Proceed to **Module 18: Security & Compliance (DevSecOps)**.
