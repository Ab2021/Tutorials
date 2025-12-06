# Day 56: Week 8 Review & Project - The Multi-Tenant GPU Cluster
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** Connectivity and Scheduling Capstone. We will configure a Kubernetes cluster to handle two distinct classes of citizens: **Data Scientists** (who need massive dedicated power) and **Inference Bots** (who need cheap, shared resources).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Node/Taint strategy to isolate training workloads.
2.  **Configure** the NVIDIA Device Plugin to expose both whole GPUs and Time-Slicing replicas.
3.  **Create** PriorityClasses to ensure Training jobs preempt Inference jobs (or vice-versa).
4.  **Visualize** the partitioned cluster state using DCGM metrics.

---

## 📚 Week 8 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 50 | Scheduling | "Taints keep the riff-raff (web apps) off my H100s." |
| 51 | Device Plugin | "K8s is blind until this plugin opens its eyes to GPUs." |
| 52 | MIG | "One A100 can become 7 Data Science workstations." |
| 53 | Topology | "Cross-socket talk is slow. Keep data local." |
| 54 | Time-Slicing | "I can run 10 tiny bots on one GPU without MIG." |
| 55 | Monitoring | "XID 48 means my card is frying. Alert me!" |

---

## 🏗️ Final Project: "ComputeHub"

### Architecture
*   **Node Pool A (Training):** 2x V100. Tainted `workload=training`. No Time-Slicing.
*   **Node Pool B (Inference):** 1x T4. Untainted. Time-Slicing (4 replicas).

### Step 1: Cluster Config (Simulation)

We will use Labels and Taints to simulate two pools on Minikube (or imaginary nodes).

#### 📁 `project/setup-nodes.sh`
```bash
#!/bin/bash
# Mocking a Multi-Node setup on a single node for logic testing
# In real life, run these on distinct nodes.

# 1. Label nodes
kubectl label node minikube accelerator=nvidia-t4
kubectl label node minikube-m02 accelerator=nvidia-v100

# 2. Taint the V100 node (Dedicated for Training)
kubectl taint node minikube-m02 workload=training:NoSchedule

# 3. Create Priority Classes
cat <<EOF | kubectl apply -f -
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: research-critical
value: 1000000
globalDefault: false
description: "Do not interrupt Training"
EOF
```

### Step 2: Device Plugin Config (Time Slicing)

We want Time-Slicing ONLY on the T4 node. The V100 should remain whole.
*Note: The Plugin Config allows `resourceManager` to target specific devices by UUID or Name, but simpler is to use a specific ConfigMap.*

#### 📁 `project/nvdp-config.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: complex-sharing
  namespace: nvidia-device-plugin
data:
  any: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
          devices:
            # Only time-slice the T4 (Mock UUID or Name)
            # In a homogeneous node-pool, we just deploy different DaemonSets to different pools.
            names: ["Tesla T4"] 
```

### Step 3: Workload Manifests

#### 📁 `project/workloads.yaml`
```yaml
# 1. The Training Job (Needs V100, Dedicated)
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training
spec:
  template:
    spec:
      priorityClassName: research-critical
      containers:
      - name: trainer
        image: nvidia/cuda:11.7.0-base-ubuntu20.04
        command: ["/bin/bash", "-c"]
        args: ["nvidia-smi; echo Training...; sleep 100"]
        resources:
          limits:
            nvidia.com/gpu: 1
      tolerations:
      - key: "workload"
        operator: "Equal"
        value: "training"
        effect: "NoSchedule"
      nodeSelector:
        accelerator: nvidia-v100
      restartPolicy: Never

---
# 2. The Inference Bot (Needs T4, Shared)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chat-bot
spec:
  replicas: 3 # Fits on 1 T4 (capacity 4)
  selector:
    matchLabels:
      app: bot
  template:
    metadata:
      labels:
        app: bot
    spec:
      containers:
      - name: bot
        image: nvidia/cuda:11.7.0-base-ubuntu20.04
        command: ["/bin/bash", "-c"]
        args: ["nvidia-smi; echo Inferencing...; sleep 3600"]
        resources:
          limits:
            nvidia.com/gpu: 1 # Will get a slice
      nodeSelector:
        accelerator: nvidia-t4
```

### Step 4: Verification

```bash
kubectl apply -f project/workloads.yaml
kubectl get pods -o wide
```
**Expected Outcome:**
*   `llm-training` lands on `minikube-m02` (V100). It tolerates the taint.
*   `chat-bot` pods land on `minikube` (T4). They share the GPU.

### Step 5: Monitoring Dashboard (Conceptual)

Create a Grafana Text Panel describing the state:
*   **Query A:** `count(kube_pod_status_phase{phase="Running", namespace="default"} * on(pod) group_left(resource) kube_pod_container_resource_requests{resource="nvidia.com/gpu"})`
*   **Goal:** Visualize "We have 3 Bots and 1 Trainer running".

---

## 📝 Success Criteria
1.  **Isolation:** Inference pods NEVER land on the Training node (due to Taint).
2.  **Utilization:** The T4 node runs multiple pods (Time-Slicing active).
3.  **Preemption:** (If config'd) A High Priority Training job kicks off a Low Priority job if VRAM is needed (requires sophisticated scheduler plugins like Volcano, but PriorityClass is the first step).

---

**Week 8 Complete** ✅

*Next Week: Week 9 - Helm, Operators & GitOps - Moving from `kubectl apply` to automated Platform Engineering.*
