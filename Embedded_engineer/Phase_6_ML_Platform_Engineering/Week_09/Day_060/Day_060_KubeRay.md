# Day 60: Scaling Python: The KubeRay Operator
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** Combine the flexibility of **Ray** with the robustness of **Kubernetes**. Learn to use the **KubeRay Operator** to manage ephemeral Ray clusters for distributed training and serving.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the KubeRay Operator using Helm.
2.  **Define** a `RayCluster` manifest with specified CPU/GPU resources.
3.  **Submit** a `RayJob` that automatically creates a cluster, runs code, and cleans up.
4.  **Connect** to the Ray Dashboard through Kubernetes Port-Forwarding.
5.  **Explain** how KubeRay maps Ray concepts (Head/Worker) to K8s Pods.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster (4GB+ RAM).

### Software Environment
- `helm`.
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Ray on Bare Metal vs K8s
*   **Bare Metal:** You manually start `ray start --head` on one node and `ray start --address head:6379` on workers. Fragile.
*   **K8s (KubeRay):** You apply YAML. The Operator creates the Head Pod. It creates Worker Pods. If a Worker crashes, K8s restarts it, and it rejoins the Ray cluster automatically.

### 2. CRDs Overview
1.  **RayCluster:** A long-running cluster (like a Development environment).
2.  **RayJob:** An ephemeral workflow. "Spin up cluster -> Run Script -> Delete Cluster". (Cost efficient).
3.  **RayService:** For deploying Ray Serve applications with high availability.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Installing KubeRay

Use the official Helm chart.

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install Operator
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0
```

### 👨‍💻 Core Implementation: Defining a Cluster

#### 📁 `manifests/ray-cluster.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: my-first-ray-cluster
spec:
  # Ray Version
  rayVersion: '2.4.0'
  
  # Head Node Configuration
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.4.0
          resources:
            limits:
              cpu: 1
              memory: 2Gi
          ports:
          - containerPort: 8265 # Dashboard
          - containerPort: 6379 # GCS

  # Worker Node Configuration
  workerGroupSpecs:
  - replicas: 2
    minReplicas: 1
    maxReplicas: 5
    groupName: small-group
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.4.0
          resources:
            limits: # Important to set limits for Autoscaling behavior
              cpu: 1
              memory: 1Gi
```

### 👨‍💻 Lab Operations

1.  **Apply Cluster:**
    ```bash
    kubectl apply -f manifests/ray-cluster.yaml
    ```
2.  **Verify Pods:**
    ```bash
    kubectl get pods
    # Should see:
    # my-first-ray-cluster-head-xxxxx
    # my-first-ray-cluster-worker-small-group-yyyyy
    ```
3.  **Access Dashboard:**
    ```bash
    kubectl port-forward svc/my-first-ray-cluster-head-svc 8265:8265
    # Open http://localhost:8265
    ```
4.  **Run Code (Interactive):**
    ```bash
    # Exec into Head node
    kubectl exec -it my-first-ray-cluster-head-xxxxx -- python
    ```
    ```python
    import ray
    ray.init()
    print(ray.cluster_resources())
    # Should show total CPUs = Head + 2*Worker
    ```

---

## 🔬 Lab Exercise: "The Ephemeral Job"

### Task
Use `RayJob` to save money.
1.  Create `ray-job.yaml` (See KubeRay docs or standard example).
2.  Set `shutdownAfterJobFinishes: true`.
3.  Apply it.
4.  **Observation:**
    *   Pods appear.
    *   Dashboard shows job running.
    *   Job completes.
    *   Pods disappear.
5.  This is ideal for nightly training runs.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Abstraction:** KubeRay abstracts the complexity of `ray start`. You focus on "How many CPUs do I need?".
2.  **Head Node:** The interaction point. It hosts the GCS (Global Control Store) and Dashboard. If Head dies, Cluster dies (unless GCS Fault Tolerance is enabled).
3.  **Scheduling:** KubeRay ensures Workers are scheduled correctly on K8s nodes, respecting Taints/Tolerations if specified.

### API Summary
```bash
kubectl get rayclusters
kubectl get rayjobs
```

---

**Day 60 Complete** ✅

*Next: Day 61 - GitOps Principles - Stop running `kubectl apply` manually. Let Git do it.*
