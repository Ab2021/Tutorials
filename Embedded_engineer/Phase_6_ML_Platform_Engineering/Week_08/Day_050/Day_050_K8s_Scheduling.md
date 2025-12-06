# Day 50: The Brain of the Cluster: Kubernetes Scheduler
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** Before we run GPUs, we must master **Scheduling**. Learn how to force your training jobs onto specific high-performance nodes using **Affinity** and **Taints**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Filter -> Score -> Bind lifecycle of the kube-scheduler.
2.  **Use** `nodeSelector` for simple placement constraints.
3.  **Implement** `nodeAffinity` for complex logic (e.g., "Run on Zone A OR Zone B").
4.  **Apply** `Taints` to dedicated nodes (e.g., "GPU Nodes") and `Tolerations` to allow Pods to run there.
5.  **Debug** `Pending` pods caused by scheduling constraints.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Scheduling Loop
When you run `kubectl apply -f pod.yaml`, the Pod starts with `nodeName: ""` (Empty).
The Scheduler watches for empty nodeNames.
1.  **Filtering (Predicates):** "Does Node A have enough RAM? Does it match the Selector?"
2.  **Scoring (Priorities):** "Node A has the Docker Image cached. Node B is empty. Node A wins."
3.  **Binding:** Writes `nodeName: NodeA` to the Pod spec.

### 2. Taints & Tolerations (The Bouncer)
*   **Taint:** Applied to a **Node**. "No entry unless you are on the VIP list."
*   **Toleration:** Applied to a **Pod**. "I'm on the VIP list."
*   **Use Case:** Preventing a web server from landing on an expensive H100 GPU node.

### 3. Affinity (The Magnet)
*   **NodeAffinity:** "I want to be on a node with label `gpu=h100`."
*   **PodAffinity:** "I want to be on the *same* node as the `redis` pod." (Low latency).
*   **PodAntiAffinity:** "I do *NOT* want to be on the same node as another `training-job`." (Spread/High Availability).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Taints & Tolerations

We will simulate a "GPU Node" by tainting Minikube.

```bash
# 1. Taint the node (Simulate "Dedicated Hardware")
# Key=dedicated, Value=gpu, Effect=NoSchedule
kubectl taint nodes minikube dedicated=gpu:NoSchedule

# 2. Try to run a normal pod
kubectl run nginx --image=nginx
# Check status
kubectl get pod nginx
# Output: Pending (Reason: 1 node(s) had taint {dedicated: gpu}, that the pod didn't tolerate)
```

Now, let's write a Pod that tolerates it.

#### 📁 `manifests/gpu-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  containers:
  - name: sleeper
    image: python:3.9-slim
    command: ["sleep", "3600"]
  # The VIP Pass
  tolerations:
  - key: "dedicated"
    operator: "Equal"
    value: "gpu"
    effect: "NoSchedule"
  # Optional: Explicitly ask for it via Affinity too
  # (Toleration allows access, Affinity requests preference)
  nodeSelector:
    kubernetes.io/hostname: minikube
```

### 👨‍💻 Core Implementation: Affinity

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: zone-aware-app
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: topology.kubernetes.io/zone
            operator: In
            values:
            - us-east-1a
            - us-east-1b
  containers:
  - name: app
    image: nginx
```

---

## 🔬 Lab Exercise: "The Priority Class"

### Task
What if the cluster is full? We want Training Jobs to yield to Inference Jobs.
1.  Create PriorityClasses.
    ```yaml
    apiVersion: scheduling.k8s.io/v1
    kind: PriorityClass
    metadata:
      name: high-priority
    value: 1000000
    globalDefault: false
    description: "Inference Production"
    ```
2.  Deploy a Pod with `priorityClassName: high-priority`.
3.  Fill the cluster with low priority pods.
4.  **Observation:** The Scheduler will **Evict** (kill) a low priority pod to make room for the high priority one. This is Preemption.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Dedication:** Use Taints to reserve expensive hardware (GPUs) for specific teams/workloads.
2.  **Spread:** Use PodAntiAffinity to ensure 3 replicas of your API land on 3 *different* nodes (HA).
3.  **Debugging:** ALWAYS check `kubectl describe pod <pending-pod>`. The Events log tells you exactly why scheduling failed ("0/5 nodes available: 3 Insufficient cpu, 2 Taints").

### API Summary
```bash
kubectl taint nodes <node> key=val:NoSchedule
kubectl label nodes <node> key=val
kubectl get priorityclass
```

---

**Day 50 Complete** ✅

*Next: Day 51 - NVIDIA Device Plugin - Teaching Kubernetes to see GPUs as a resource.*
