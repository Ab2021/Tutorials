# Day 78: The Plumbing: CNI Deep Dive (Container Network Interface)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** Before we run 100Gbps training jobs, we must understand the "Pipe". Learn how **CNI Plugins** (Calico, Cilium, AWS VPC CNI) wire up your Pods, and why "Overlay Networks" kill performance.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Packet Flow of Pod-to-Pod communication (Intra-Node vs Inter-Node).
2.  **Differentiate** Overlay (VXLAN/IPIP) vs Underlay (Direct Routing) modes.
3.  **Analyze** the performance overhead of Encapsulation (CPU usage).
4.  **Compare** Calico (BGP) vs Cilium (eBPF).
5.  **Debug** networking issues using `tcpdump` inside netshoot pods.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Multi-Node Cluster (Minikube multi-node or EKS).

### Software Environment
- `kubectl`.
- `tcpdump` / `wireshark`.

---

## 📖 Theoretical Foundation

### 1. The Pod Network Requirement
Kubernetes guarantees:
*   Every Pod gets a unique IP.
*   Pod A can talk to Pod B without NAT.
*   Agents on a node (Kubelet) can talk to all Pods on that node.

### 2. Implementation: The CNI Plugin
When a Pod starts, Kubelet calls the CNI binary (`/opt/cni/bin/calico`).
The CNI:
1.  Creates a vETH pair (one end in Pod, one end in Host).
2.  Assigns an IP from the IPAM (IP Address Management) pool.
3.  Configures Routing Table (`ip route`).

### 3. Overlay vs Direct
*   **Overlay (VXLAN):**
    *   Pod sends packet.
    *   Host wraps packet in distinct UDP packet.
    *   Host sends UDP to Dest Host.
    *   Dest Host unwraps packet.
    *   **Overhead:** 30-40 bytes per packet. High CPU usage at >10Gbps.
*   **Direct Routing (AWS VPC CNI):**
    *   Pod acts like a VM. No wrapping.
    *   **Performance:** Near line-rate. EFA compatible. **Required for Distributed Training.**

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Investigating Interfaces

Launch a Netshoot pod (Swiss army knife for network debugging).

```bash
kubectl run netshoot --image=nicolaka/netshoot --restart=Never -it -- /bin/bash
```

Inside the pod:
```bash
# 1. Check IP
ip addr
# You see 'eth0@if5'. This is the vETH pair.

# 2. Check Routing
ip route
# default via 169.254.1.1 ...
```

### 👨‍💻 Lab: Tracing Encapsulation (If using Calico VXLAN)

1.  Find the Host Node of the pod.
2.  SSH into the Host Node.
3.  Run `tcpdump` on the physical interface (`eth0` or `ens5`).
    ```bash
    tcpdump -i eth0 port 4789 -n
    ```
    *Note: 4789 is VXLAN port.*
4.  Ping from Pod A to Pod B (on different node).
5.  **Observation:** You will see UDP packets on port 4789. The ICMP ping is hidden inside.

---

## 🔬 Lab Exercise: "The MTU Trap"

### Task
Jumbo Frames.
1.  Default MTU (Maximum Transmission Unit) is 1500 bytes.
2.  VXLAN adds 50 bytes header.
3.  If Physical MTU is 1500, Pod Interface MTU must be 1450.
4.  **Experiment:** Try to send a 1480 byte ping from Pod A.
    ```bash
    ping -s 1480 <pod-b-ip> -M do
    ```
5.  **Result:** Fragmentation Needed (Packet Dropped).
6.  **Fix:** Configure CNI to auto-detect MTU, or enable Jumbo Frames (9000 bytes) on physical switch for high-performance ML.

---

## 📝 Daily Summary

### Key Takeaways
1.  **For ML:** Avoid Overlays. Use **Route Routing** (Calico BGP) or **Cloud CNI** (AWS/Azure/GKE). Encapsulation CPU overhead steals cycles from the DataLoader.
2.  **Cilium:** The modern choice. Uses eBPF to bypass iptables (which is slow at scale). Provides excellent visibility (`hubble observe`).
3.  **Debug:** `kubectl run -it --rm netshoot` is your best friend.

### API Summary
```bash
ip link show
ip route get <ip>
```

---

**Day 78 Complete** ✅

*Next: Day 79 - Service Mesh for ML - Why Istio matters for Canary Deployments.*
