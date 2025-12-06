# Day 96: Elastic Compute: Autoscaling Ray on K8s
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** Static clusters are wasteful. Learn how to configure the **Ray Autoscaler** to coordinate with the **Kubernetes Cluster Autoscaler** for truly elastic AI infrastructure.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Diagram** the interaction between Ray Autoscaler (Pod requester) and K8s Autoscaler (Node requester).
2.  **Configure** `minReplicas` and `maxReplicas` in a RayCluster.
3.  **Trigger** scale-up by submitting resource-hungry tasks.
4.  **Verify** scale-down (idle timeout) behaviors.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster (Mocking is possible, but real scaling requires Node capacity).

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Trigger Chain
1.  **User:** Submits 100 tasks. Each needs 1 CPU.
2.  **Head Node Scheduler:** Sees only 2 CPUs available. 98 Tasks are "Pending".
3.  **Ray Autoscaler (in Head):** Calculates: "I need 98 more CPUs". Looks at WorkerGroups. Finds "cpu-group".
4.  **Ray Autoscaler:** API Call to K8s -> `ReplicaSet.scale(50)`.
5.  **K8s Scheduler:** Sees 50 new Pods. Puts them on Nodes. 
6.  *If Nodes are full:* **Cluster Autoscaler (AWS/GCP)** sees Pending Pods. Provisions new EC2 instances.
7.  **Ray:** Workers join cluster. Tasks run.

### 2. Scale Down
If a worker is idle for `idle_timeout_minutes` (default 1m), Ray kills the worker pod.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Autoscaling Cluster

#### 📁 `manifests/auto-raycluster.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: elastic-cluster
spec:
  rayVersion: '2.9.0'
  enableInTreeAutoscaling: true # <--- CRITICAL
  
  headGroupSpec:
    # ... standard head ...
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0
          resources:
            requests:
              cpu: "1"

  workerGroupSpecs:
  - replicas: 0 # Start with empty cluster
    minReplicas: 0
    maxReplicas: 10
    groupName: worker-group
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0
          resources:
            requests:
              cpu: "1" # Each worker provides 1 CPU
              memory: "1Gi"
```

### 👨‍💻 Lab Operations: Triggering Scale

1.  **Apply:** `kubectl apply -f manifests/auto-raycluster.yaml`.
2.  **Check:** `kubectl get pods`. Only Head exists.
3.  **Run Script:**
    ```python
    import ray
    import time
    
    # Connect via local port-forward
    ray.init("ray://localhost:10001") 
    
    @ray.remote(num_cpus=1)
    def heavy_task(i):
        time.sleep(30)
        return i

    # Launch 10 tasks
    # Needs 10 CPUs. Cluster sends request for 10 Pods.
    print(ray.get([heavy_task.remote(i) for i in range(10)]))
    ```
4.  **Watch:**
    *   Ray Dashboard: "Demanding 10 CPUs".
    *   Kubectl: `ray-worker-xxxxx` pods appearing (Pending -> Running).
5.  **Wait:** After script finishes (plus 1 minute), pods terminate.

---

## 🔬 Lab Exercise: "Deadlock"

### Task
Resource Mismatch.
1.  Worker Pod defines `resources: requests: cpu: "2"`.
2.  Ray Start Params defines nothing (Auto-detects 2 CPUs).
3.  Task requests `@ray.remote(num_cpus=3)`.
4.  **Observation:** Ray Autoscaler sees demand for "3 CPU chunks".
5.  It looks at WorkerGroup. WorkerGroup provides "2 CPU chunks".
6.  **Result:** It CANNOT satisfy the request. It might scale up infinite workers hoping one is big enough (it won't be), or log an "Infeasible Requirement" warning.
7.  **Insight:** Always ensure your Tasks fit within your largest Worker type.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Latency:** Autoscaling is slow. K8s Pod (10s) + EC2 Node (2m). Not suitable for bursty, millisecond-SLA traffic. Use `minReplicas` to keep a warm pool.
2.  **Fragmentation:** If you have 10 workers with 0.5 CPU free each, and you ask for 1 CPU... Ray cannot combine them. You need a new empty worker.
3.  **Knative?** KubeRay is usually better than Knative for Ray because Ray is stateful/cluster-aware, whereas Knative is purely request-driven stateless.

### API Summary
```bash
kubectl get rayclusters --show-labels
```

---

**Day 96 Complete** ✅

*Next: Day 97 - GCS High Availability - Making the Head Node resilient.*
