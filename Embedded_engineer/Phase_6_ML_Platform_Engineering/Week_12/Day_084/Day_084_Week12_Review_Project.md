# Day 84: Week 12 Review & Project - The HPC Cluster Network
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 12: Networking for Distributed ML

---

> **🎯 Focus Area:** Connectivity is the backbone of Distributed Computing. We will synthesize CNI, RDMA, Service Mesh, and Security to build a **High-Performance Training Network**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architecture** a dual-plane network (Control vs Data).
2.  **Configure** Multus to expose secondary high-speed interfaces.
3.  **Secure** the control plane with Istio mTLS and Network Policies.
4.  **Optimize** DNS for bursty peer discovery.

---

## 📚 Week 12 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 78 | CNI Deep Dive | "Overlay networks eat CPU. Avoid them for training." |
| 79 | Istio | "I can route 1% of traffic to the new model without redeploying." |
| 80 | RDMA/RoCE | "Bypass the Kernel. Speed of light latency." |
| 81 | Multus | "Why choose? Have Calico for K8s and Macvlan for Data." |
| 82 | DNS | "`ndots:5` was killing my startup time." |
| 83 | NetPol | "Default Deny is the only safe default." |

---

## 🏗️ Final Project: "PhotonNet"

### Use Case
A cluster of 16 nodes (128 GPUs). We need to run a Link Prediction model on a massive graph. Communication overhead is the bottleneck.

### Architecture
1.  **Network A (Control):** standard K8s CNI (Cilium). Handles Logging, Metrics, Kube API. Secured by NetworkPolicy.
2.  **Network B (Data):** RoCEv2 (RDMA) via Multus + SR-IOV. Zero-trust not applied (Performance priority). Handles NCCL `AllReduce`.

### Step 1: Physical Setup Specification (Mock)
*   **NIC 0:** Management (1Gbps). Connected to Switch A.
*   **NIC 1:** High Speed (100Gbps). Connected to Switch B (PFC Enabled).

### Step 2: Multus Configuration

#### 📁 `project/cni-conf.yaml`
```yaml
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
  name: rdma-net
  namespace: hpc-training
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "sriov",
      "ipam": {
        "type": "whereabouts", # Cluster-wide IPAM
        "range": "192.168.100.0/24"
      }
    }'
```

### Step 3: Pod Specification (MPIJob)

We use the Kubeflow training operator (Week 9 concept) combined with Multus.

#### 📁 `project/mpi-job.yaml`
```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: graph-training
  namespace: hpc-training
spec:
  slotsPerWorker: 8
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - image: mpi-operator:latest
            name: mpi-launcher
    Worker:
      replicas: 2
      template:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: rdma-net # <--- ATTACH RDMA
            sidecar.istio.io/inject: "false"      # <--- SKIP ISTIO (Performance)
        spec:
          containers:
          - image: pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime
            name: pytorch
            resources:
              limits:
                nvidia.com/gpu: 8
                mellanox.com/rdma_vf: 1
            env:
              - name: NCCL_SOCKET_IFNAME
                value: "net1" # Use the fast network!
              - name: GLOO_SOCKET_IFNAME
                value: "eth0" # Fallback
```

### Step 4: Security Layer (Control Plane Only)

Protect the Launcher from external tampering.

#### 📁 `project/security.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: secure-launcher
  namespace: hpc-training
spec:
  podSelector:
    matchLabels:
      training.kubeflow.org/replica-type: Launcher
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          training.kubeflow.org/replica-type: Worker
    # Only Workers can talk to Launcher
```

---

## 🔬 Lab Exercise: "The Partition"

### Task
Simulate `net1` failure.
1.  Deploy the job.
2.  NCCL uses `net1`. Training speed: 100 it/s.
3.  "Break" the cable (Change `NCCL_SOCKET_IFNAME=eth0`).
4.  NCCL falls back to `eth0`.
5.  **Observation:** Speed drops to 5 it/s. CPU usage on Host spikes (Kernel TCP processing).

---

## 📝 Success Criteria
1.  **Dual Homing:** Pods have 2 IPs. `ip addr` shows `eth0` and `net1`.
2.  **Performance:** `ib_write_bw` on `net1` achieves line rate.
3.  **Isolation:** Applying a NetworkPolicy to `eth0` does not block RDMA traffic on `net1` (since RDMA bypasses kernel/iptables usually, but check CNI specifics!).

---

**Week 12 Complete** ✅
**Phase 6B Complete** ✅

*Next Phase: Phase 6C - We move from Infrastructure to **MLOps Pipelines & Automation**.*
