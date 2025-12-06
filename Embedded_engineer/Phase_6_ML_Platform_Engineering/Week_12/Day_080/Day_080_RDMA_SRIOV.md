# Day 80: Speed of Light: SR-IOV & RDMA
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** TCP/IP is too slow for 1000-GPU clusters. Learn how **RDMA (Remote Direct Memory Access)** and **SR-IOV** bypass the Linux Kernel to achieve microsecond latency.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** TCP/IP overhead vs RDMA "Zero Copy" architecture.
2.  **Explain** SR-IOV: Splitting one Physical NIC (PF) into many Virtual Functions (VF).
3.  **Identify** RoCE (RDMA over Converged Ethernet) requirements in Cloud.
4.  **Deploy** the SR-IOV Network Device Plugin to expose VFs to K8s.
5.  **Debug** InfiniBand connectivity using `ib_write_bw`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Mellanox ConnectX NICs (or AWS EFA instances).
- *Simulation:* Conceptual only if hardware unavailable.

### Software Environment
- `ibverbs-utils`.

---

## 📖 Theoretical Foundation

### 1. The Kernel Bottleneck
Sending 1GB via TCP:
1.  Application `buffer` -> System Call.
2.  Kernel copies `buffer` to `socket buffer`.
3.  TCP Logic (Acks, Windows, Checksums).
4.  IP Logic.
5.  Driver -> NIC.
**Result:** High CPU usage. High Latency.

### 2. RDMA (Remote Direct Memory Access)
1.  Application registers memory region.
2.  Application tells NIC: "Send this region".
3.  NIC DMA reads memory and sends.
4.  Remote NIC writes directly to Remote App memory.
**Result:** CPU is free. Latency ~1us.

### 3. SR-IOV (Single Root I/O Virtualization)
To use RDMA in containers, the container needs direct access to hardware.
SR-IOV allows one Physical Function (PF) to spawn 64 Virtual Functions (VFs).
Each VF looks like a real PCIe device to the container.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: SR-IOV Device Plugin

Like the NVIDIA Device Plugin, but for NICs.

```yaml
# 1. Config Map (Define Resource)
apiVersion: v1
kind: ConfigMap
metadata:
  name: sriovdp-config
data:
  config.json: |
    {
        "resourceList": [{
            "resourceName": "mellanox_nic",
            "selectors": {
                "vendors": ["15b3"], # Mellanox
                "devices": ["1017"], # ConnectX-5
                "drivers": ["mlx5_core"]
            }
        }]
    }
```

Deploy the DaemonSet.

### 👨‍💻 Core Implementation: Requesting the VF

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: rdma-pod
  annotations:
    k8s.v1.cni.cncf.io/networks: sriov-net # Attaches the interface
spec:
  containers:
  - name: app
    image: mellanox/ofed-driver
    resources:
      limits:
        mellanox.com/mellanox_nic: 1 # Request 1 VF
    securityContext:
      capabilities:
        add: ["IPC_LOCK"] # Required for RDMA memory registration
```

---

## 🔬 Lab Exercise: "Bandwidth Test"

### Task
Measure the difference.
1.  Run `iperf3` (TCP) between two pods. Max ~25Gbps (limited by CPU).
2.  Run `ib_write_bw` (RDMA Write Bandwidth).
    ```bash
    # Server
    ib_write_bw -d mlx5_0
    
    # Client
    ib_write_bw -d mlx5_0 <server-ip>
    ```
3.  **Observation:**
    *   Throughput: Line rate (100Gbps).
    *   CPU Usage: Near 0%.
    *   Latency: Sub-microsecond.

---

## 📝 Daily Summary

### Key Takeaways
1.  **NCCL & RDMA:** PyTorch's `torch.distributed` uses NCCL. NCCL automatically detects RDMA interfaces (`ib0`) if present. If setup correctly, `DDP` automatically goes fast.
2.  **RoCE vs InfiniBand:** RoCE runs RDMA over standard Ethernet (requires lossless network/PFC). InfiniBand uses specialized cables. Cloud (AWS/Azure) usually provides specialized adapters (EFA/Elastic RDMA) that act like RoCE.
3.  **Memlock:** You MUST set `ulimit -l unlimited` (IPC_LOCK) for RDMA containers, or memory registration fails.

### API Summary
```bash
ibstat
ibdev2netdev
rdma link show
```

---

**Day 80 Complete** ✅

*Next: Day 81 - Multus CNI - Giving a Pod two interfaces (Management + High Speed).*
