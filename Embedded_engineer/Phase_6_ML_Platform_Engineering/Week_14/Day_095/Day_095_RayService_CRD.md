# Day 95: Zero Downtime: RayService CRD
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** Serving ML models requires high availability. **RayServe** allows you to host Python-based model pipelines, and the **RayService** CRD ensures seamless, zero-downtime updates.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between RayService (Long-running) and RayJob (Batch).
2.  **Deploy** a `RayService` that exposes an HTTP endpoint.
3.  **Diagram** the Blue/Green deployment strategy used by KubeRay for updates.
4.  **Send** inference requests to the exposed Service.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster.

### Software Environment
- `kubectl`.
- `curl`.

---

## 📖 Theoretical Foundation

### 1. Ray Serve
Ray Serve is an Actor-based framework for Model Serving.
*   **Deployment:** A class decorated with `@serve.deployment`.
*   **Ingress:** A specialized actor that listens on HTTP/gRPC.
*   **Controller:** Manages scaling of deployments.

### 2. RayService Controller Logic
When you update a `RayService` (e.g., change `replicas: 2` to `replicas: 5`):
*   **In-Place Update:** If possible, it just updates the running cluster.
When you update the Cluster Spec (e.g., Image `v1` to `v2`):
*   **Blue/Green:**
    1.  Create `New-Cluster` (v2).
    2.  Wait for Serve apps to be healthy on `New-Cluster`.
    3.  Switch K8s Service to point to `New-Cluster`.
    4.  Delete `Old-Cluster`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Serve Code

We need an image that contains the Serve app. For this lab, we use inline code via `runtime_env` (Day 90 concept) inside the yaml.

#### 📁 `manifests/ray-service.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: model-serving
  namespace: ray-system
spec:
  serviceUnhealthySecondThreshold: 300
  deploymentUnhealthySecondThreshold: 300
  
  serveConfigV2: |
    applications:
      - name: my_app
        import_path: app.model
        runtime_env:
          working_dir: "https://github.com/ray-project/test_dag/archive/41d09119cb7151a51543aa71a5f57fc625292415.zip"
        deployments:
          - name: Model
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
            - containerPort: 8000 # SERVE PORT
              name: serve
    workerGroupSpecs:
    - replicas: 1
      groupName: small-group
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.9.0
```

### 👨‍💻 Lab Operations

1.  **Apply:** `kubectl apply -f manifests/ray-service.yaml`.
2.  **Wait:** `kubectl get rayservice -n ray-system -w`.
    *   `Pending` -> `Running`.
3.  **Access:**
    *   KubeRay creates a service: `model-serving-head-svc`.
    *   It also creates specific serve services often.
    *   Forward Port 8000:
        ```bash
        kubectl port-forward svc/model-serving-head-svc 8000:8000 -n ray-system
        ```
4.  **Test:**
    ```bash
    curl -X POST http://localhost:8000/
    ```

---

## 🔬 Lab Exercise: "The Crash Upgrade"

### Task
Simulate an upgrade.
1.  Change `rayVersion` to `2.9.1` (or any different version/image).
2.  Apply.
3.  **Observation:** `kubectl get pods` shows a NEW RayCluster starting up alongside the OLD one.
4.  Once the new one is Ready, the old one Terminates.
5.  **Downtime:** 0 seconds (requests are routed by K8s Service).

---

## 📝 Daily Summary

### Key Takeaways
1.  **HA:** RayService is the only CRD that provides high availability for the service endpoint.
2.  **Ingress:** In production, you typically put an Nginx Ingress or Istio Gateway *in front* of the `RayService` to handle SSL and Authn.
3.  **State:** Since clusters are ephemeral during upgrades, your Serve actors must be **Stateless** (fetching weights from S3) or support quick hydration.

### API Summary
```bash
serve status
```

---

**Day 95 Complete** ✅

*Next: Day 96 - Autoscaling on K8s - Dynamic Resource Provisioning.*
