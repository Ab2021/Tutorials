# Day 53: The Path of Least Resistance: Topology Awareness
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** Architecture is physical. A GPU is soldered to a specific PCIe lane, connected to a specific CPU Socket. Learn how **Topology Awareness** prevents 50% performance drops caused by cross-socket traffic.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** NUMA (Non-Uniform Memory Access) and its impact on GPU data transfer.
2.  **Interpret** the output of `nvidia-smi topo -m`.
3.  **Configure** the Kubelet **Topology Manager** to enforce alignment.
4.  **Demonstrate** the performance penalty of QPI/UPI traversal (Cross-Socket).
5.  **Select** the appropriate Topology Policy (`single-numa-node` vs `restricted`).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- A Multi-Socket Server (e.g., Dual Xeon/Epyc) with GPUs.
- *Simulation:* Can be studied conceptually on single-socket machines.

### Software Environment
- Kubelet configuration access.

---

## 📖 Theoretical Foundation

### 1. The Physical Map
Imagine a server with:
*   **Socket 0:** Cores 0-15. Connected to PCIe Root A (GPU 0, GPU 1).
*   **Socket 1:** Cores 16-31. Connected to PCIe Root B (GPU 2, GPU 3).
*   **Interconnect:** QPI/UPI / Infinity Fabric connects Socket 0 <-> Socket 1.

**Scenario A (Good):** Pod gets Core 0 and GPU 0. Data flows: RAM -> Core 0 -> PCIe -> GPU 0. (Fast).
**Scenario B (Bad):** Pod gets Core 16 and GPU 0. Data flows: RAM -> Core 16 -> QPI -> Core 0 -> PCIe -> GPU 0. (Slow, High Latency).

### 2. Kubelet Topology Manager
The Kubelet is responsible for allocating CPUs (CPU Manager) and Devices (Device Manager).
The **Topology Manager** sits in between and says: "Hey, CPU Manager wants Socket 1, but Device Manager picked GPU 0 (Socket 0). This is a mismatch."

### 3. Policies
*   **none:** Default. Chaos.
*   **best-effort:** Try to align, but allow mismatch if resources are fragmented.
*   **restricted:** Reject pods if alignment fails.
*   **single-numa-node:** Strict. Everything (CPU, RAM, GPU) must come from ONE NUMA node.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Checking Topology

On the Host (Node):

```bash
nvidia-smi topo -m
```

**Output Interpretation:**
```text
        GPU0  GPU1  GPU2  GPU3
GPU0     X    PIX   SYS   SYS
GPU1    PIX    X    SYS   SYS
GPU2    SYS   SYS    X    PIX
GPU3    SYS   SYS   PIX    X
```
*   **PIX:** PCIe Switch (Close neighbor). Good for NVLink/P2P.
*   **SYS:** System (Cross-NUMA). Bad.

### 👨‍💻 Core Implementation: Kubelet config

To enable Topology Awareness, edit `/var/lib/kubelet/config.yaml` on the node.

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
# ...
topologyManagerPolicy: single-numa-node
topologyManagerScope: container # or 'pod'
```

Restart Kubelet:
```bash
systemctl restart kubelet
```

### 👨‍💻 Manifest Impact

You don't change your YAML. K8s handles it automatically.
However, if you request `cpu: 10` and `nvidia.com/gpu: 1`, and your Node has 40 cores (20 per socket):
*   If Kubelet picks GPU 0 (Socket 0), it **MUST** give you 10 cores from Socket 0.
*   If Socket 0 has 15 cores free: Success.
*   If Socket 0 has 5 cores free (even if Socket 1 has 20 free): **Pod Pending (Topology Affinity Error).**

---

## 🔬 Lab Exercise: "Stress the Interconnect"

### Task
(Requires Bare Metal or large VM).
1.  Run `p2pBandwidthLatencyTest` (CUDA Sample).
2.  Observed transfer between GPU 0 and GPU 1 (Same Socket) should be PCIe Gen4 speed (~50GB/s) or NVLink speed (~600GB/s).
3.  Observe transfer between GPU 0 and GPU 2 (Cross Socket). It will drop to UPI speed (typically ~20-30GB/s) and increase latency.

This confirms why Topology Awareness is critical for **Distributed Training** (where latency kills convergence).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Invisible Performance Killer:** You can define everything correctly in Docker/PyTorch, but if Kubelet misaligns NUMA, your training is 20-30% slower silently.
2.  **Scope Matters:** Use `topologyManagerScope: pod` for TensorFlow/PyTorch where multiple containers in a pod might share memory (shm).
3.  **Bin Packing:** Strict topology policies reduce fragmentation efficiency (you might leave cores stranded on Socket 1 because GPU 0 is taken), but guarantee performance.

### API Summary
```bash
# Debugging
kubectl get pod -o wide # See which node it landed on
lscpu # See NUMA layout
```

---

**Day 53 Complete** ✅

*Next: Day 54 - GPU Sharing with Time-Slicing & MPS - What to do if you don't have an A100 for MIG.*
