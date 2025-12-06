# Day 52: Slicing the Beast: Multi-Instance GPU (MIG)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** If you have an A100 or H100, assigning it to a single Jupyter Notebook is a waste. Learn to use **MIG (Multi-Instance GPU)** to partition one GPU into up to 7 fully isolated hardware instances.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between MIG (Hardware Partition) and Time-Slicing (Software Scheduling).
2.  **List** valid MIG Profiles (e.g., `1g.10gb`, `3g.40gb`) for A100.
3.  **Configure** the NVIDIA Device Plugin to advertise MIG devices to Kubernetes.
4.  **Schedule** a Pod specifically to a "Slice" of a GPU.
5.  **Explain** the benefits of Fault Isolation in MIG.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA Ampere (A100, A30) or Hopper (H100) GPU. *Earlier GPUs (V100/T4) do NOT support MIG.*

### Software Environment
- NVIDIA Driver > 450.
- Kubernetes.

---

## 📖 Theoretical Foundation

### 1. The Utilization Problem
An A100 has 80GB VRAM and massive FP32 compute. A typical BERT-Base Inference job uses 2GB VRAM and 10% Compute.
*   **Result:** 90% wasted capital.

### 2. What is MIG?
MIG physically partitions the GPU interconnects.
*   **Memory:** Dedicated VRAM regions (no sharing).
*   **Compute:** Dedicated SMs (Streaming Multiprocessors).
*   **Cache:** Dedicated L2 Cache slices.
*   **Fault Isolation:** If Slice A has a memory error or infinite loop, Slices B-G continue running unaffected. This is critical for Multi-Tenant clusters.

### 3. Profiles
Named by `{Compute Slices}g.{Memory}gb`.
*   **1g.10gb:** 1 SM Block, 10GB Mem. (Max 7 per A100).
*   **3g.40gb:** 3 SM Blocks, 40GB Mem. (Max 2 per A100).
*   **7g.80gb:** Full A100.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Configuring K8s for MIG

By default, K8s sees "1 GPU". We must tell the Device Plugin to see "MIG Instances".

#### 1. Enable MIG Mode (Host Level)
This requires root on the node and a GPU reset.
```bash
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi -i 0 --gpu-reset
```

#### 2. Create Instances (Host Level)
Create 7 small slices.
```bash
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C
# 19 is the ID for 1g.10gb on A100
```
Verify:
```bash
nvidia-smi
# You will see 7 "MIG Xg.Ygb" devices listed.
```

#### 3. Device Plugin Config
We need to update the Helm Chart values for the Device Plugin (Day 51).

```yaml
# values.yaml
migStrategy: "mixed"
config:
  name: "default-mig"
  map:
    default:
      namingStrategy: type-and-size # Produces nvidia.com/mig-1g.10gb
```

#### 4. Requesting a Slice in Pod

#### 📁 `manifests/mig-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mig-inference
spec:
  containers:
  - name: server
    image: nvcr.io/nvidia/tritonserver:23.01-py3
    resources:
      limits:
        # Request exactly one 1g.10gb slice
        nvidia.com/mig-1g.10gb: 1
```

---

## 🔬 Lab Exercise: "Instance Packing"

### Task
(Conceptual if no A100 available).
1.  Assume Node A has 1 Physical A100.
2.  Enable MIG with 7x `1g.10gb`.
3.  Check Node Capacity (`kubectl describe node`). It should show `nvidia.com/mig-1g.10gb: 7`.
4.  Launch 7 Pods requesting this resource.
5.  **Observation:** All 7 run simultaneously.
6.  Launch an 8th Pod.
7.  **Observation:** Pending (Insufficient resources).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Cost Efficiency:** MIG allows one expensive A100 to serve 7 different dev teams simultaneously with guaranteed performance.
2.  **Immutability:** Reconfiguring MIG (e.g., changing from 7x1g to 2x3g) often requires draining the node and resetting the GPU. It is not "Dynamic autoscaling" in real-time.
3.  **Use Case:** MIG is King for **Inference** and **Dev Environments**. For Large Model Training (LLM), you want the full GPU (Disable MIG).

### API Summary
```bash
nvidia-smi mig -lgi # List GPU Instances
```

---

**Day 52 Complete** ✅

*Next: Day 53 - Topology Awareness - NUMA Nodes and NVLink routing.*
