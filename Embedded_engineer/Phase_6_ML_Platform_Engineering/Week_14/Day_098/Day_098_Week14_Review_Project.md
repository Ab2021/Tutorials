# Day 98: Week 14 Review & Project - The Ray Platform
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** We have learned the components. Now we construct the platform. You will build a consolidated environment where Data Scientists can submit Training Jobs and deploy Inference Services without touching `kubectl`.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architecture** a Multi-Tenant Ray Platform (Namespaces + Quotas).
2.  **Deploy** a `RayJob` for batch training.
3.  **Deploy** a `RayService` for model serving.
4.  **Scrape** Ray metrics using Prometheus.

---

## 📚 Week 14 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 92 | KubeRay Operator | "It's the brain that manages the Ray pods." |
| 93 | RayCluster | "I can mix CPU Head nodes with GPU Worker nodes." |
| 94 | RayJob | "Ephemeral clusters save money." |
| 95 | RayService | "Zero-downtime upgrades for ML models." |
| 96 | Autoscaling | "The Ray Autoscaler talks to the K8s Autoscaler." |
| 97 | GCS HA | "Redis keeps the cluster alive even if the Head dies." |

---

## 🏗️ Final Project: "RayOps Platform"

### Architecture
*   **Namespace:** `ray-platform`.
*   **Shared Services:** Prometheus, Grafana, KubeRay Operator.
*   **Workloads:**
    1.  **Training:** A Matrix Multiplication Job (Simulating Model Training).
    2.  **Serving:** A Text Generator Service.

### Step 1: Platform Setup

```bash
kubectl create ns ray-platform
helm install kuberay-operator kuberay/kuberay-operator -n ray-platform
```

### Step 2: The Training Job (Batch)

#### 📁 `project/job.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: matrix-training
  namespace: ray-platform
spec:
  entrypoint: "python -c 'import ray; ray.init(); print(ray.available_resources())'"
  shutdownAfterJobFinishes: true
  rayClusterSpec:
    rayVersion: '2.9.0'
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.9.0
            resources: { requests: { cpu: 1, memory: 2Gi } }
    workerGroupSpecs:
    - replicas: 2
      groupName: workers
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.9.0
            resources: { requests: { cpu: 1, memory: 1Gi } }
```

### Step 3: The Inference Service (Long Running)

#### 📁 `project/service.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: text-gen
  namespace: ray-platform
spec:
  serviceUnhealthySecondThreshold: 60
  serveConfigV2: |
    applications:
      - name: text_app
        import_path: text_generator.app
        runtime_env:
          working_dir: "https://github.com/ray-project/serve_config_examples/archive/master.zip"
        deployments:
          - name: Generator
            num_replicas: 1
            ray_actor_options:
              num_cpus: 1
  rayClusterConfig:
    rayVersion: '2.9.0'
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.9.0
            ports:
            - containerPort: 8000
              name: serve
            - containerPort: 8265
              name: dashboard
            - containerPort: 8080
              name: metrics # <--- For Prometheus
    workerGroupSpecs:
    - replicas: 1
      groupName: serving-workers
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.9.0
```

### Step 4: Observability (ServiceMonitor)

Tell Prometheus to scrape Ray.

#### 📁 `project/monitor.yaml`
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ray-monitor
  namespace: ray-platform
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      ray.io/node-type: head
  endpoints:
  - port: metrics
    path: /metrics
```

---

## 🔬 Lab Exercise: "The Full Cycle"

### Task
1.  **Submit Job:** Apply `job.yaml`. Watch it spin up, print resources (CPU: 3.0), and shut down.
2.  **Deploy Service:** Apply `service.yaml`. Port forward 8265. Check Dashboard.
3.  **Scale Service:** Edit `service.yaml` -> `num_replicas: 5`. Apply. Watch K8s scale up.
4.  **Visualize:** Use Grafana to see "Active Actors".

---

## 📝 Success Criteria
1.  **Separation:** Job and Service run in separate Clusters (managed by Operator).
2.  **Automation:** No manual `ray start` commands.
3.  **Visibility:** Metrics flow to Prometheus.

---

**Week 14 Complete** ✅

*Next Week: Week 15 - Ray Train - Distributed Deep Learning with PyTorch and Ray.*
