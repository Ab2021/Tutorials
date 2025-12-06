# Day 165: Roommates: Multi-Tenancy & Resource Quotas
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** Team A wants to run 100 experiments. Team B wants to serve steady traffic. If Team A hogs all the GPUs, Team B goes down. **Multi-Tenancy** ensures fair sharing of the dedicated hardware.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** ResourceQuotas to limit Team A to 4 GPUs.
2.  **Enable** GPU Time-Slicing to run 5 Pods on a single GPU (great for dev/notebooks).
3.  **Implement** LimitRanges so pods default to small resource requests.
4.  **Partition** an A100 GPU using MIG (Multi-Instance GPU).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- NVIDIA GPU Cluster.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Isolation Levels
*   **Cluster Isolation:** Separate implementation plan for every team. (Expensive management).
*   **Namespace Isolation:** Shared cluster, separate namespaces. Quotas enforce limits. (Standard).
*   **Node Isolation:** `nodeSelector` keeps "Team A" pods on "Team A" nodes. (Inefficient).

### 2. GPU Sharing
*   **Exclusive Mode:** Pod gets full GPU. (Standard for Training).
*   **Time-Slicing:** Use MPS (Multi-Process Service) or Time-Slicing plugin. Multiple processes context-switch on GPU. Good for Inference/Dev.
*   **MIG:** Hardware partitioning. Guaranteed isolation of Memory and Compute.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Resource Quotas

Stop Team A from bankrupting the company.

#### 📁 `manifests/quota-team-a.yaml`
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-quota
  namespace: team-a-research
spec:
  hard:
    requests.cpu: "40"
    requests.memory: 100Gi
    requests.nvidia.com/gpu: "4" # Max 4 physical GPUs
    pods: "20"
```

### 👨‍💻 Infrastructure: Limit Ranges

Prevent "No Request" pods. If a user doesn't specify CPU, give them 0.5, not Unlimited.

#### 📁 `manifests/limit-range.yaml`
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: defaults
  namespace: team-a-research
spec:
  limits:
  - default: # The Limit
      memory: 8Gi
      cpu: 2
    defaultRequest: # The Request
      memory: 4Gi
      cpu: 1
    type: Container
```

### 👨‍💻 Infrastructure: GPU Time-Slicing

How to make 1 GPU look like 10 GPUs.

#### 📁 `manifests/cluster-policy-time-slicing.yaml`
```yaml
# Config for NVIDIA Device Plugin
version: v1
sharing:
  timeSlicing:
    resources:
    - name: nvidia.com/gpu
      replicas: 10 # 1 Physical GPU = 10 Virtual GPUs
```
Now, a pod requests `nvidia.com/gpu: 1`, it gets 1/10th of a GPU (conceptually). It shares memory with 9 others. OOM is possible.

### 👨‍💻 Core Implementation: Priority Classes

Team B (Prod) > Team A (Dev).

#### 📁 `manifests/priority-classes.yaml`
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: production-critical
value: 1000000
globalDefault: false
description: "Mission Critical Inference"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: research-preemptible
value: 50
globalDefault: false
description: "Experiments that can die"
```

If the cluster is full, and a `production-critical` pod arrives, K8s will **Evict** `research-preemptible` pods to make space.

---

## 🔬 Lab Exercise: "The Bad Neighbor"

### Task
Simulate Noisy Neighbors with Time-Slicing.
1.  Configure Time-Slicing (Replicas: 2).
2.  Launch **Pod A**: Runs `while True: compute_heavy_matrix_mul()`.
3.  Launch **Pod B**: Runs Inference.
4.  **Observation:** Pod B latency spikes by 500%.
5.  **Reason:** Time-slicing means context switching. Memory bandwidth is shared.
6.  **Fix:** Use MIG (Hardware Partitioning) if Latency guarantees are required. Time-slicing is only for Notebooks where humans tolerate lag.

---

## 📖 Advanced Theory: Multi-Cluster Federation (Karmada)
When you outgrow 1 cluster (limit is ~5000 nodes), you need Multi-Cluster.
*   **Karmada / KubeFed:** Defines a `PropagationPolicy`: "Spread this Deployment across AWS and GCP clusters to reduce cost."

---

## 📝 Daily Summary

### Key Takeaways
1.  **Quotas:** Set quotas on `requests`, not just `limits`. Requests determine scheduling. Limits determine OOM kill.
2.  **Oversubscription:** You can oversubscribe CPU (Request total > Capacity), but you cannot oversubscribe Memory or GPU (unless Time-Slicing).
3.  **Chargeback:** Use `kubecost` to look at Namespace spend. "Team A spent $500 this week." Send them the bill.

### API Summary
```yaml
requests.nvidia.com/gpu: "1"
```

---

**Day 165 Complete** ✅

*Next: Day 166 - Serverless Inference - Scale to Zero with Lambda/Fargate.*
