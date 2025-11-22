# Lab 24.1: Prometheus Operator

## 🎯 Objective

Stop writing raw Prometheus config. The **Prometheus Operator** allows you to define monitoring targets as Kubernetes Objects (`ServiceMonitor`, `PodMonitor`). It automatically reloads configuration without restarts.

## 📋 Prerequisites

-   Minikube running.
-   Helm installed.

## 📚 Background

### CRDs (Custom Resource Definitions)
-   **Prometheus**: Defines the server instance.
-   **ServiceMonitor**: "Scrape any Service with label `app: nginx`".
-   **AlertManager**: Defines alerting configuration.

---

## 🔨 Hands-On Implementation

### Part 1: Install the Stack 📦

We use the community `kube-prometheus-stack` which includes Operator, Prometheus, Grafana, and Alertmanager.

1.  **Add Repo:**
    ```bash
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    ```

2.  **Install:**
    ```bash
    helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
    ```

3.  **Verify:**
    `kubectl get pods -n monitoring`.
    `kubectl get servicemonitors -A`.

### Part 2: Monitor an App (The Old Way vs New Way) 🆚

**Old Way**: Edit `prometheus.yml`, add `static_configs`, restart.
**New Way**: Create a `ServiceMonitor`.

1.  **Deploy App:**
    ```bash
    kubectl create deployment my-app --image=nginx
    kubectl expose deployment my-app --port=80
    ```

2.  **Create `servicemonitor.yaml`:**
    ```yaml
    apiVersion: monitoring.coreos.com/v1
    kind: ServiceMonitor
    metadata:
      name: my-app-monitor
      namespace: monitoring
      labels:
        release: monitoring # Must match the Operator's selector
    spec:
      selector:
        matchLabels:
          app: my-app
      endpoints:
      - port: port-80 # Name of the port in the Service
    ```
    *Note:* The Service must have a named port.
    Update service: `kubectl edit svc my-app` -> name the port `port-80`.

3.  **Apply:**
    ```bash
    kubectl apply -f servicemonitor.yaml
    ```

4.  **Verify:**
    Open Prometheus UI (`kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090`).
    Status -> Targets.
    You should see `my-app` automatically discovered.

### Part 3: PrometheusRule (Alerts as Code) 🚨

1.  **Create `alert.yaml`:**
    ```yaml
    apiVersion: monitoring.coreos.com/v1
    kind: PrometheusRule
    metadata:
      name: my-alert
      namespace: monitoring
      labels:
        release: monitoring
    spec:
      groups:
      - name: example
        rules:
        - alert: ExampleAlert
          expr: vector(1)
          for: 1m
          labels:
            severity: warning
    ```

2.  **Apply:**
    `kubectl apply -f alert.yaml`.
    Check Alertmanager.

---

## 🎯 Challenges

### Challenge 1: Blackbox Exporter (Difficulty: ⭐⭐⭐)

**Task:**
Install `prometheus-blackbox-exporter`.
Create a `Probe` CRD to ping `google.com`.
*Goal:* Monitor external uptime.

### Challenge 2: Grafana Dashboards as Code (Difficulty: ⭐⭐)

**Task:**
Create a ConfigMap with label `grafana_dashboard: "1"`.
Put the JSON of a dashboard inside.
The Operator will automatically import it into Grafana.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  labels:
    grafana_dashboard: "1"
data:
  my-dash.json: |
    { ... }
```
</details>

---

## 🔑 Key Takeaways

1.  **Dynamic Discovery**: No more manual config updates.
2.  **GitOps Ready**: Everything (Monitors, Alerts, Dashboards) is YAML.
3.  **Standard**: This is the standard way to run Prometheus on K8s.

---

## ⏭️ Next Steps

Prometheus is great, but it runs out of disk space.

Proceed to **Lab 24.2: Thanos**.
