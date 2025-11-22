# Lab 9.3: Grafana Dashboards

## 🎯 Objective

Visualize your data. Prometheus collects the numbers, but Grafana makes them beautiful and understandable. You will connect Grafana to Prometheus and import a pre-made dashboard.

## 📋 Prerequisites

-   Completed Lab 9.2 (Prometheus running).
-   Docker installed.

## 📚 Background

### Grafana
-   **Data Source**: Where data comes from (Prometheus, MySQL, AWS CloudWatch).
-   **Dashboard**: A collection of Panels.
-   **Panel**: A single graph/chart.

---

## 🔨 Hands-On Implementation

### Part 1: Run Grafana 📊

1.  **Start Container:**
    ```bash
    docker run -d \
      -p 3000:3000 \
      --name grafana \
      grafana/grafana
    ```

2.  **Login:**
    Open `http://localhost:3000`.
    User: `admin`
    Pass: `admin` (Skip password change).

### Part 2: Add Data Source 🔌

1.  **Configuration:**
    -   Click **Gear Icon** (Configuration) -> **Data Sources**.
    -   Click **Add data source**.
    -   Select **Prometheus**.

2.  **Settings:**
    -   URL: `http://host.docker.internal:9090` (or your local IP).
    -   Click **Save & Test**.
    -   *Result:* "Data source is working".

### Part 3: Create a Dashboard 📈

1.  **New Dashboard:**
    -   Click **+** -> **Dashboard**.
    -   Click **Add a new panel**.

2.  **Query:**
    -   Data source: `Prometheus`.
    -   Metrics: `rate(prometheus_http_requests_total[1m])`.
    -   Click **Run queries**.
    -   *Result:* A line graph appears.

3.  **Save:**
    -   Click **Save** (Disk icon).
    -   Name: "My First Dashboard".

### Part 4: Import Node Exporter Dashboard 📥

Building dashboards from scratch is hard. Let's use the community.

1.  **Find ID:**
    Go to `grafana.com/dashboards`. Search for "Node Exporter Full".
    ID is usually `1860`.

2.  **Import:**
    -   In Grafana, Click **+** -> **Import**.
    -   Enter ID: `1860`.
    -   Click **Load**.
    -   Select Prometheus Data Source.
    -   Click **Import**.

3.  **Result:**
    A professional dashboard showing CPU, RAM, Disk, and Network traffic of your host!

---

## 🎯 Challenges

### Challenge 1: The Gauge (Difficulty: ⭐⭐)

**Task:**
Add a new Panel to your dashboard.
Visualization: **Gauge**.
Metric: `node_memory_MemFree_bytes`.
*Goal:* A speedometer-style gauge showing free RAM.

### Challenge 2: Alerting in Grafana (Difficulty: ⭐⭐⭐)

**Task:**
Create a Grafana Alert on the CPU usage graph.
Condition: If CPU > 80% for 5 minutes.
Notification: Send to "Grafana" (default).
*Note:* Grafana Alerting is an alternative to Prometheus Alertmanager.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
1. Add Panel.
2. Query: `node_memory_MemFree_bytes`.
3. Right side "Visualization" -> Select "Gauge".
4. Unit: "Data" -> "bytes".

**Challenge 2:**
1. Edit Panel.
2. Click "Alert" tab.
3. Create Alert Rule.
</details>

---

## 🔑 Key Takeaways

1.  **Don't Build from Scratch**: Always check Grafana Labs for existing dashboards (Nginx, MySQL, Docker, etc.).
2.  **Variables**: Dashboards can have dropdowns (e.g., select "Server A" or "Server B") using Variables.
3.  **Single Pane of Glass**: Grafana can show metrics (Prometheus) and logs (Loki) side-by-side.

---

## ⏭️ Next Steps

We have metrics. Now let's handle logs.

Proceed to **Lab 9.4: ELK Stack Basics**.
