# Day 82: It's Always DNS: CoreDNS & Service Discovery
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** In Distributed Training, ranks must find each other immediately. A slow DNS lookup causes timeout crashes. Master **CoreDNS**, **Headless Services**, and the **Ndots** optimization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Trace** the resolution path of a K8s DNS query in `/etc/resolv.conf`.
2.  **Explain** the "Ndots:5" performance stampede and how to fix it.
3.  **Deploy** a **Headless Service** for peer-to-peer discovery (PyTorch DDP).
4.  **Install** NodeLocal DNSCache to offload CoreDNS.
5.  **Debug** `NXDOMAIN` errors using `dnstools`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Search Path
Inside a Pod, `cat /etc/resolv.conf` shows:
```text
search default.svc.cluster.local svc.cluster.local cluster.local
options ndots:5
```
**Scenario:** You request `postgres`.
1.  Try `postgres.default.svc.cluster.local` -> Found! (1 query).
**Scenario:** You request `google.com`.
1.  Try `google.com.default.svc.cluster.local` -> NXDOMAIN.
2.  Try `google.com.svc.cluster.local` -> NXDOMAIN.
3.  ... (3 more) ...
4.  Try `google.com.` -> Found!
**Result:** 5x Amplification on all external traffic.

### 2. Headless Services
Standard Service returns a Virtual IP (VIP).
**Headless Service** (`clusterIP: None`) returns the IPs of *all* backing pods.
*   **Why for ML?** In Ring All-Reduce, GPU 0 needs to open a TCP socket to GPU 1 directly. It needs the real IP, not a load balancer VIP.

### 3. NodeLocal DNSCache
A daemonset that runs a mini-dns cache on every node.
*   Pod queries local cache (Listen IP 169.254.20.10).
*   Avoids conntrack entries and network hops to specific CoreDNS pods.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Ndots Fix

If you use Fully Qualified Domain Names (FQDN), you can lower ndots.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: optimized-dns-pod
spec:
  dnsConfig:
    options:
    - name: ndots
      value: "2"
  containers:
  - name: app
    image: nginx
```
*Effect:* `google.com` (1 dot) < 2. It is resolved absolutely immediately. `mysvc` (0 dots) < 2. It applies search domains.

### 👨‍💻 Core Implementation: Headless Service for DDP

```yaml
apiVersion: v1
kind: Service
metadata:
  name: training-headless
  labels:
    app: pytorch-dist
spec:
  ports:
  - port: 29500 # PyTorch Default
    name: comms
  clusterIP: None # <--- HEADLESS
  selector:
    app: pytorch-dist
```

### 👨‍💻 Verification

1.  Deploy 3 replicas behind `training-headless`.
2.  Run `nslookup training-headless`.
3.  **Output:**
    ```text
    Name: training-headless.default.svc.cluster.local
    Address: 10.244.1.5
    Address: 10.244.2.6
    Address: 10.244.3.7
    ```
    (Returns 3 A Records).

---

## 🔬 Lab Exercise: "DNS Bench"

### Task
Measure the impact.
1.  Run a pod with `ndots:5` looping `curl -I google.com`.
2.  Check CoreDNS logs/metrics. See the flood of `NXDOMAIN`.
3.  Run a pod with `ndots:1`.
4.  **Observation:** Traffic drops by 80%.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Always use FQDN in Prod:** Don't call `postgres`. Call `postgres.default.svc.cluster.local`. It skips the search path traversal.
2.  **UDP Issues:** DNS uses UDP. Packets can be dropped during network congestion (common in ML bursts). TCP DNS (`force_tcp`) is more reliable but slower.
3.  **StatefulSets:** Use Headless Services coupled with StatefulSets to get stable hostnames: `web-0.nginx`, `web-1.nginx`.

### API Summary
```bash
nslookup <host>
dig <host>
```

---

**Day 82 Complete** ✅

*Next: Day 83 - Network Policies - Ensuring Tenant Isolation.*
