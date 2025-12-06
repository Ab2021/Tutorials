# Day 189: Week 27 Review & Project - The War Room
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** We have learned the tools of the trade: eBPF, Tcpdump, Nsys, Memray, Loki, Tempo. Now we build **The War Room**: An integrated dashboard that correlates Logs, Metrics, and Traces to solve outages in minutes, not hours.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the full "LGTM" Stack (Loki, Grafana, Tempo, Mimir/Prometheus).
2.  **Simulate** a complex outage (Memory Leak + Network Latency) using Chaos Mesh.
3.  **Triangulate** the root cause using the "Three Pillars of Observability".
4.  **Create** an Automated Alert that triggers a PagerDuty incident when Golden Signals degrade.

---

## 📚 Week 27 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 183 | eBPF Info | "I can trace C++ functions without recompiling the code." |
| 184 | Network Debug | "It wasn't the code; it was a firewall drop (RST)." |
| 185 | GPU Debug | "Xid 79 means hardware failure. Stop restarting the pod." |
| 186 | Memory Debug | "Python Reference Cycles keep 10GB of RAM alive." |
| 187 | Log Analytics | "LogQL lets me calculate Error Rate from text logs." |
| 188 | Distributed Tracing | "Redis Scan caused the 300ms latency, not the LLM." |

---

## 🏗️ Final Project: "Project Sherlock"

### Architecture
The "LGTM" Stack.
*   **Logs:** Loki.
*   **Graphs:** Prometheus (Mimir).
*   **Traces:** Tempo.
*   **Visualization:** Grafana.

### Step 1: Deploying the Stack (Helm)

#### 📁 `project/deploy_stack.sh`
```bash
# Add Grafana Repo
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Deploy Everything (Loki, Tempo, Promtail, Grafana)
helm install lgtm grafana/loki-stack \
    --set grafana.enabled=true \
    --set prometheus.enabled=true \
    --set promtail.enabled=true \
    --set tempo.enabled=true
```

### Step 2: Chaos Injection (The Failure)

We will break the system intentionally.

#### 📁 `project/chaos/network-delay.yaml`
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: redis-delay
spec:
  action: delay
  mode: all
  selector:
    namespaces:
      - default
    labelSelectors:
      app: redis
  delay:
    latency: "200ms"
    jitter: "50ms"
  duration: "5m"
```

#### 📁 `project/chaos/memory-leak.yaml`
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: kill-pod-randomly
spec:
  action: pod-kill
  mode: one
  selector:
    labelSelectors:
      app: inference-worker
  scheduler:
    cron: "@every 1m"
```

### Step 3: The Dashboard (Grafana JSON)

*I cannot embed JSON here, but the concept is:*
1.  **Row 1 (Golden Signals):** Latency (Line Chart), Error Rate (Gauge), Saturation (CPU/Mem).
2.  **Row 2 (Logs):** `failed` or `error` or `exception` logs.
3.  **Row 3 (Traces):** List of Slowest Traces (> 2s).

### Step 4: The Debugging Workflow (Script)

#### 📁 `project/debug_guide.md`
**Scenario:** "Users reporting 500 Errors and slowness."

1.  **Check Metrics:**
    *   Dashboard shows Latency spiked to 2s. Error rate 5%.
    *   *Clue:* Redis Latency metric is high.
2.  **Check Traces:**
    *   Click on a slow Trace.
    *   Visualize Waterfall.
    *   *Observation:* 80% of time spent in `redis.get`.
    *   *Hypothesis:* Network Delay or Slow Query.
3.  **Check Logs:**
    *   Filter `{app="redis"}`.
    *   No errors. Just slow.
4.  **Check Network (Tcpdump):**
    *   Capture traffic to Redis.
    *   Wireshark shows `SYN` -> `SYN-ACK` takes 200ms.
    *   *Root Cause:* Network Latency (Chaos Mesh).

---

## 🔬 Lab Exercise: "The Post-Mortem"

### Task
Write a Root Cause Analysis (RCA) Report.
1.  **Incident:** 2023-10-27 14:00 UTC. Outage for 15 mins.
2.  **Impact:** 500 users affected. 10% request failure.
3.  **Root Cause:** NetworkChaos injected 200ms delay to Redis.
4.  **Detection:** Alert `HighLatency` fired at 14:02.
5.  **Resolution:** Deleted NetworkChaos CRD.
6.  **Action Items:**
    *   Add Timeout to Redis Client (currently infinite).
    *   Implement Circuit Breaker (Fail fast if Redis is slow).

---

## 📝 Success Criteria
1.  **Correlation:** Clicking a Log line in Grafana opens the corresponding Trace (via TraceID).
2.  **Alerting:** Receiving an email/Slack message when Error Rate > 1%.
3.  **Visibility:** Identifying the difference between "App Slow" (Code) and "Network Slow" (Infrastructure) using Tempo.

---

**Week 27 Complete** ✅
**Phase 6E In Progress**

*Next Phase: Platform Engineering Practices - Building the IDP.*
