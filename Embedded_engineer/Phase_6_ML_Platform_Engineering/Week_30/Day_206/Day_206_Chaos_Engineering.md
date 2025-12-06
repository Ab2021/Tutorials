# Day 206: Controlled Explosions: Chaos Engineering with Chaos Mesh
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 30: Capstone Project Part 2

---

> **🎯 Focus Area:** "Everything fails all the time." (Werner Vogels). You can't prevent failure, but you can inoculate your system against it. Today we run a **Game Day**: We intentionally inject faults into Titan to verify our resilience.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** a "Game Day" with Hypothesis, Blast Radius, and Rollback Plan.
2.  **Execute** DNS Chaos (Simulate CoreDNS failure) and verifying Client-side Caching.
3.  **Execute** Network Partition (Split Brain) between Trainer and Parameter Server.
4.  **Automate** Chaos Workflows to run weekly via Cron.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-control`.

### Software Environment
- `helm install chaos-mesh`.

---

## 📖 Theoretical Foundation

### 1. The Principles of Chaos
*   **Hypothesis:** "If I kill the Redis Leader, the Followers will elect a new Leader in < 5s."
*   **Blast Radius:** "Only affect the `dev` namespace." (Never run global chaos on Prod first).
*   **Abort:** "If Error Rate > 5%, STOP immediately."

### 2. Chaos Mesh Architecture
*   **Controller Manager:** Manages CRDs (`PodChaos`, `NetworkChaos`).
*   **Chaos Daemon:** DaemonSet (runs on every node). Uses `tc` (Traffic Control) and `iptables` to mangle packets at the Kernel level.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Install Chaos Mesh

```bash
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh \
    --namespace chaos-mesh --create-namespace \
    --set chaosDaemon.runtime=containerd \
    --set chaosDaemon.socketPath=/run/containerd/containerd.sock
```

### 👨‍💻 Core Implementation: Scenario 1 - DNS Failure

Hypothesis: "Our Python code caches DNS lookups, so a 10s DNS outage should be invisible."

#### 📁 `manifests/chaos/dns-fail.yaml`
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: DNSChaos
metadata:
  name: dns-random-error
  namespace: chaos-mesh
spec:
  action: error # Return NXDOMAIN
  mode: all
  selector:
    namespaces:
      - default
    labelSelectors:
      app: inference-worker
  duration: "30s"
  patterns:
    - "google.com" # Only fail external lookups
```

### 👨‍💻 Core Implementation: Scenario 2 - Blackhole (Packet Loss)

Hypothesis: "If the Training Worker loses connection to the Head, the Head should mark it dead after 30s."

#### 📁 `manifests/chaos/network-loss.yaml`
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: worker-isolation
spec:
  action: loss
  mode: one
  selector:
    labelSelectors:
      ray.io/node-type: worker
  loss:
    loss: "100" # 100% Packet Loss
    correlation: "100"
  duration: "60s"
  direction: both
```

### 👨‍💻 Core Implementation: Game Day Dashboard

A Grafana Dashboard is critical during Chaos.

1.  **Panel 1:** `sum(rate(http_requests_total{status="500"}[1m]))` (Abort signal).
2.  **Panel 2:** `up{job="chaos-mesh"}` (Ensure chaos is running).
3.  **Panel 3:** `ray_cluster_active_workers` (Watch it drop and recover).

---

## 🔬 Lab Exercise: "The Unplanned Outage"

### Task
Simulate Etcd Latency.
1.  **Objective:** Test if the Kubernetes Control Plane can handle slow Etcd.
2.  **Hypothesis:** If Etcd latency > 100ms, API Server will time out requests.
3.  **Injection:** `IOChaos` on Etcd datadir (requires Sidecar Injection or Host Access). *Simpler alternative: NetworkChaos on Etcd Pod.*
    *   `delay: 200ms`.
4.  **Observation:**
    *   `kubectl get pods` becomes slow (3s).
    *   Leader Election (Lease) might time out if `renewDeadline` is tight.
    *   Controller Manager might restart.
5.  **Conclusion:** Titan is sensitive to Etcd latency.
6.  **Action Item:** Move Etcd to dedicated nodes with NVMe SSDs.

---

## 📖 Advanced Theory: Kernel Chaos
Chaos Mesh can do **Kernel Chaos** (using eBPF).
*   **File I/O:** Inject errors into `open()`, `read()`. Make the disk "full" virtually.
*   **Time:** Make the clock jump forward. (Testing Lease expiration).
*   **Memory:** Poison pages.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Start Small:** Don't start with "Delete the Cluster". Start with "Delete one Pod".
2.  **Verify Observability:** Chaos Engineering is actually a test of your *Monitoring*. If you inject a fault and your dashboard stays Green, your dashboard is broken.
3.  **Culture:** Don't blame people for failures during Chaos. Celebrate the finding. "We found a bug in the Retry Logic before customers did."

### API Summary
```bash
kubectl apply -f chaos.yaml
kubectl delete -f chaos.yaml # Stop chaos immediately
```

---

**Day 206 Complete** ✅

*Next: Day 207 - Technical Documentation - TechDocs.*
