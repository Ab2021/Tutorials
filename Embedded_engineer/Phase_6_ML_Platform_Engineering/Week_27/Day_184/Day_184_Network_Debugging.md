# Day 184: The Packet Never Lies: Network Debugging
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** "Connection Refused". "Timeout". "DNS Lookup Failure". 90% of Distributed ML failures are networking. **Tcpdump** and **Wireshark** let you see the raw bytes on the wire.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Capture** traffic inside a Kubernetes Pod using `ksniff` or sidecar `tcpdump`.
2.  **Analyze** a TCP Handshake (SYN, SYN-ACK, ACK) to diagnose firewall drops.
3.  **Debug** CoreDNS latency using `dnstools` and `dig`.
4.  **Trace** dropped packets due to NetworkPolicies using `calicoctl` or `cilium monitor`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `tcpdump`, `wireshark` (optional UI), `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The TCP State Machine
*   **Establishment:** Client sends SYN. Server sends SYN-ACK. Client sends ACK. (3-Way Handshake).
*   **Termination:** FIN -> ACK -> FIN -> ACK.
*   **Reset (RST):** "I don't know you. Go away." (Port closed or Firewall reject).

### 2. DNS in Kubernetes
`svc-name.ns.svc.cluster.local` -> 10.96.0.10 (CoreDNS Service) -> 10.244.1.5 (CoreDNS Pod) -> Upstream DNS (8.8.8.8).
Common failure: `ndots:5` config causes 5 DNS lookups for every request.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Debug Pod (Netshoot)

Always have this Swiss Army Knife ready.

#### 📁 `manifests/debug-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: netshoot
spec:
  containers:
  - name: netshoot
    image: nicolaka/netshoot
    command: ["sleep", "infinity"]
```

### 👨‍💻 Core Implementation: Capturing Traffic (Tcpdump)

Capture traffic on port 80.

```bash
# Exec into pod
kubectl exec -it netshoot -- bash

# Capture GET requests
# -i any: All interfaces
# -nn: Don't resolve hostnames/ports (faster)
# -w capture.pcap: Write to file
tcpdump -i any -nn port 80 -w /tmp/capture.pcap

# Copy out to analyze in Wireshark
kubectl cp netshoot:/tmp/capture.pcap ./capture.pcap
```

### 👨‍💻 Core Implementation: Packet Analysis (TShark)

Text-mode Wireshark.

```bash
# Read file
tshark -r capture.pcap

# Filter: Only Retransmissions (Packet Loss indicator)
tshark -r capture.pcap -Y "tcp.analysis.retransmission"

# Filter: Only HTTP Errors
tshark -r capture.pcap -Y "http.response.code >= 400"
```

### 👨‍💻 Infrastructure: Ksniff (Kubectl Plugin)

Simplifies capturing from a Pod without installing tcpdump *inside* the pod (uses static binary injection).

```bash
kubectl krew install sniff
kubectl sniff my-pod -n my-ns -o output.pcap
```

---

## 🔬 Lab Exercise: " The Missing SYN-ACK"

### Task
Diagnose a Network Policy Drop.
1.  **Scenario:** Pod A tries to curl Pod B. Hangs.
2.  **Capture:** Tcpdump on Pod A.
    *   Output: `SYN ... SYN ... SYN ...` (Retries).
    *   Diagnosis: Pod A is sending, but getting no response.
3.  **Capture:** Tcpdump on Pod B.
    *   Output: Silent. No packets arriving.
4.  **Conclusion:** Dropped *between* A and B. Likely NetworkPolicy or Cloud Security Group.
5.  **Fix:** `kubectl describe networkpolicy`. Allow Ingress from Pod A.

---

## 📖 Advanced Theory: DNS ndots
`/etc/resolv.conf` typically has `search default.svc.cluster.local svc.cluster.local cluster.local`.
Options: `ndots:5`.
*   User queries `google.com` (1 dot).
*   Resolver tries: `google.com.default.svc.cluster.local` (NXDOMAIN).
*   Resolver tries: `google.com.svc.cluster.local` (NXDOMAIN).
*   ... 3 more times ...
*   Finally tries: `google.com.` (Success).
*   **Fix:** Use FQDN `google.com.` (trailing dot) in application code to bypass search path.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Latency vs Bandwidth:** Tcpdump shows latency (timestamps). Iperf shows bandwidth.
2.  **RST Packet:** If you see an immediate `RST` after `SYN`, the port is closed (Application not running) or a Firewall is actively rejecting it. If you see *nothing*, it's a silent Drop.
3.  **Keepalive:** Long-running ML jobs (e.g., Parameter Server) need TCP Keepalives enabled, otherwise the firewall/NAT Gateway might silently drop the connection table entry after 300s of silence.

### API Summary
```bash
tcpdump -A -s 0 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'
# (Prints HTTP Body ascii)
```

---

**Day 184 Complete** ✅

*Next: Day 185 - The Black Screen: GPU Debugging.*
