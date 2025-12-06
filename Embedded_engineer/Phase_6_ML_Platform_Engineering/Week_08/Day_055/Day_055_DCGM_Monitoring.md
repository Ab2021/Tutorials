# Day 55: Pulse Check: DCGM & GPU Monitoring
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** You cannot optimize what you cannot measure. Deploy **DCGM (Data Center GPU Manager)** to visualize GPU Utilization, Memory, and Power across your entire cluster in Real-Time.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the `dcgm-exporter` DaemonSet to Kubernetes.
2.  **Explain** the metrics flow: GPU -> DCGM -> Exporter -> Prometheus -> Grafana.
3.  **Identify** critical metrics (`GPU_UTIL`, `FB_USED`, `XID_ERRORS`).
4.  **Write** PromQL queries to find "Idle GPUs" (Cost Optimization).
5.  **Detect** hardware failures (ECC double-bit errors) utilizing XID codes.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU Cluster.

### Software Environment
- Helm.
- A running Prometheus/Grafana stack (e.g., `kube-prometheus-stack`).

---

## 📖 Theoretical Foundation

### 1. `nvidia-smi` vs DCGM
*   `nvidia-smi` (CLI): Good for ad-hoc checking. "Is the GPU on?"
*   `DCGM` (Library): Designed for sampling at millisecond resolution (profiling) or low overhead background monitoring. Can aggregate metrics across NVLinks and NVSwitches.

### 2. The Exporter Pattern
Kubernetes monitoring uses "Pull Model".
1.  **Exporter Pod** runs on Node. It translates DCGM binary data into Text (`GPU_UTIL 85`).
2.  **Prometheus** scrapes `http://node-ip:9400/metrics`.
3.  **Grafana** queries Prometheus.

### 3. Critical Metrics
*   `DCGM_FI_DEV_GPU_UTIL`: Compute Core usage. (Start here).
*   `DCGM_FI_DEV_FB_USED`: Framebuffer (VRAM) used.
*   `DCGM_FI_DEV_POWER_USAGE`: Watts drawn. (Throttling check).
*   `DCGM_FI_DEV_XID_ERRORS`: The "Check Engine Light". XID 79 = GPU fell off bus. XID 48 = Double Bit ECC Error.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Deploying DCGM Exporter

Using the official NVIDIA Helm chart.

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/gpu-monitoring-tools/helm-charts
helm repo update

helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring \
  --create-namespace \
  --set serviceMonitor.enabled=true # Tell Prometheus Operator to scrape it
```

### 👨‍💻 Verifying Metrics

Check the text output manually before setting up Grafana.

```bash
# 1. Forward Port
kubectl port-forward svc/dcgm-exporter 9400:9400 -n monitoring

# 2. Curl
curl localhost:9400/metrics
```

**Output:**
```text
# HELP DCGM_FI_DEV_GPU_UTIL GPU Utilization
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0", UUID="GPU-ba..."} 98
DCGM_FI_DEV_FB_USED{gpu="0", UUID="GPU-ba..."} 10240
```

### 👨‍💻 PromQL Cheatsheet

If you have Grafana, use these queries:

**1. Cluster-wide GPU Utilization:**
```promql
avg(DCGM_FI_DEV_GPU_UTIL) by (kubernetes_node)
```

**2. Find Idle GPUs (Zombies):**
```promql
count(DCGM_FI_DEV_GPU_UTIL == 0)
```

**3. Power Consumption (Total Watts):**
```promql
sum(DCGM_FI_DEV_POWER_USAGE)
```

---

## 🔬 Lab Exercise: "The Stress Test"

### Task
Generate load and watch Grafana.
1.  Launch the `gpu-burn` pod (common stress test tool).
    ```bash
    kubectl run stress-test --image=wooksong/gpu-burn --restart=Never -- 300 # Run 300s
    ```
2.  Open Grafana.
3.  **Observation:**
    *   `GPU_UTIL` should spike to 100%.
    *   `POWER_USAGE` should hit TDP Limit (e.g., 300W for V100).
    *   `GPU_TEMP` should rise.
4.  Kill the pod. Watch metrics return to 0.

### Insight
This loop is essential for MLOps. If `GPU_UTIL` is 0% but `POD_STATUS` is Running, you are burning money for nothing. Set up alerts on this condition!

---

## 📝 Daily Summary

### Key Takeaways
1.  **XID Errors:** Configure an AlertManager rule for `DCGM_FI_DEV_XID_ERRORS > 0`. This alerts you *before* users complain that "Training is failing randomly" (usually due to bad VRAM).
2.  **Pod Mapping:** Modern DCGM Exporter automatically adds `pod` and `namespace` labels to the GPU metrics by talking to Kubelet. This allows you to bill usage back to specific teams: "Team A used 500 GPU-hours".
3.  **Overhead:** DCGM is lightweight (~1% CPU). Always run it in production.

### API Summary
```bash
helm install dcgm-exporter
curl http://localhost:9400/metrics
```

---

**Day 55 Complete** ✅

*Next: Day 56 - Week 8 Review & Project - Building a Shared, Monitored GPU Cluster with Time-Slicing.*
