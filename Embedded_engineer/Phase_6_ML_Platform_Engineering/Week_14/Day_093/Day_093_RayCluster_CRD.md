# Day 93: The Blueprint: RayCluster CRD
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** You don't want a "One Size Fits All" cluster. You want a cheap CPU head node and massive GPU worker nodes. The **RayCluster** Custom Resource Definition (CRD) lets you define this topology.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** a `RayCluster` manifest with distinct Head and Worker specs.
2.  **Configure** `rayStartParams` to advertise resources (e.g., `--num-gpus=1`) to the Ray Scheduler.
3.  **Deploy** a heterogeneous cluster (CPU Group + GPU Group).
4.  **Access** the Ray Dashboard via `kubectl port-forward`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster.
- KubeRay Operator installed (Day 92).

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Head vs Worker
*   **Head Node:** Runs GCS (State), Autoscaler, and Dashboard. Can also run tasks (but best practice is `num-cpus: 0` to keep it stable).
*   **Worker Group:** A template for a set of Scalable Pods. You can have multiple groups (e.g., "spot-group", "on-demand-group").

### 2. The Resource Mirroring Problem
K8s allocates resources (`requests.nvidia.com/gpu: 1`).
Ray needs to know it has resources.
*   **Auto-detect:** Ray usually auto-detects CPU/GPU inside the container.
*   **Manual:** You can force it via `rayStartParams: {"num-gpus": "1"}`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Heterogeneous Cluster

#### 📁 `manifests/gpu-raycluster.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ml-cluster
  namespace: ray-system
spec:
  rayVersion: '2.9.0'
  # HEAD NODE
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
      num-cpus: '0' # Don't schedule tasks on head
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
          ports:
          - containerPort: 6379
            name: gcs
          - containerPort: 8265
            name: dashboard

  # WORKER GROUP 1 (CPU)
  workerGroupSpecs:
  - replicas: 1
    minReplicas: 1
    maxReplicas: 5
    groupName: cpu-group
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
  
  # WORKER GROUP 2 (GPU)
  - replicas: 1 # Start with 1 GPU node
    minReplicas: 0
    maxReplicas: 2
    groupName: gpu-group
    rayStartParams: {}
    template:
      spec:
        tolerations: # Allow scheduling on GPU nodes
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0-gpu
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              cpu: "4"
              memory: "16Gi"
```

### 👨‍💻 Lab Operations

1.  **Apply:** `kubectl apply -f manifests/gpu-raycluster.yaml`.
2.  **Wait:** `kubectl wait --for=condition=Ready raycluster/ml-cluster -n ray-system`.
3.  **Port Forward Dashboard:**
    ```bash
    kubectl port-forward svc/ml-cluster-head-svc 8265:8265 -n ray-system
    ```
4.  **Open:** `http://localhost:8265`. You should see 3 nodes (Head + 2 Workers).

---

## 🔬 Lab Exercise: "Resource Verification"

### Task
Verify Ray sees the K8s resources.
1.  Run a local python script `test_connect.py`.
2.  Connect to cluster? (Hard from laptop without VPN).
3.  **Easier:** Exec into Head Node.
    ```bash
    kubectl exec -it deployment/ml-cluster-head -n ray-system -- python
    ```
4.  Run:
    ```python
    import ray
    ray.init()
    print(ray.cluster_resources())
    ```
5.  **Observation:** Should list `{"CPU": X, "GPU": 1.0}`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Headless Head:** Setting `num-cpus: 0` on the Head Node is critical for stability. If the Head runs user code and OOMs (Runs out of memory), the whole cluster crashes.
2.  **Services:** KubeRay creates a `Service` automatically named `<cluster-name>-head-svc`. Workers use this DNS name to join.
3.  **Images:** Ensure Head and Worker run the **Exact Same Ray Version**. Version mismatch leads to weird protocol errors.

### API Summary
```bash
kubectl get rayclusters
```

---

**Day 93 Complete** ✅

*Next: Day 94 - RayJob CRD - The correct way to submit work.*
