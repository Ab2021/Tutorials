# Day 59: The Robot Sysadmin: Kubernetes Operators
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** Helm handles Day 1 (Installation). **Operators** handle Day 2 (Updates, Backups, Failover). Learn how to extend Kubernetes with custom logic to manage complex stateful applications.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** the "Fire and Forget" model of Helm with the "Reconciliation Loop" of Operators.
2.  **Define** a CRD (Custom Resource Definition) to create a new K8s Object type.
3.  **Explain** the role of the Controller in the Operator pattern.
4.  **Implement** a simple Operator logic using `kopf` (Python).
5.  **Evaluate** existing Operators (Prometheus, Elastic, PyTorch) for production use.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `kubectl`.
- Python with `kopf` and `kubernetes` libraries (for the lab).

---

## 📖 Theoretical Foundation

### 1. The Limits of Helm
Helm packages YAML. It installs a Database, but if the Database runs out of disk, Helm doesn't know. If the Primary DB fails, Helm doesn't know how to promote a Replica.
**An Operator is a Code (not just YAML) running in a Pod that manages other Pods.**

### 2. The Pattern
1.  **CRD:** You tell K8s "I want to create `kind: ModelService`".
2.  **CR:** User applies `kind: ModelService`.
3.  **Controller:** A Python/Go program watches for `ModelService` events. It sees the new CR and creates the necessary Deployments, Services, and Ingresses.

### 3. Capability Levels
*   **Level 1:** Basic Install (Helm equivalent).
*   **Level 2:** Seamless Upgrades.
*   **Level 3:** Full Lifecycle (Backup, Restore).
*   **Level 4:** Deep Insights (Metrics, Alerts).
*   **Level 5:** Auto Pilot (Auto Scaling, Auto Tuning).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Custom Resource

Let's teach K8s what a `TrainingJob` is.

#### 📁 `manifests/crd.yaml`
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trainingjobs.ai.example.com
spec:
  group: ai.example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                model:
                  type: string
                epochs:
                  type: integer
  scope: Namespaced
  names:
    plural: trainingjobs
    singular: trainingjob
    kind: TrainingJob
    shortNames:
    - tj
```

Apply it: `kubectl apply -f manifests/crd.yaml`.
Now `kubectl get tj` works (return empty).

### 👨‍💻 Core Implementation: The Operator (Python)

We use `kopf` (Kubernetes Operator Pythonic Framework) to write the controller logic.

#### 📁 `src/operator.py`
```python
import kopf
import kubernetes.client as k8s

@kopf.on.create('ai.example.com', 'v1', 'trainingjobs')
def create_fn(spec, name, namespace, logger, **kwargs):
    model = spec.get('model')
    epochs = spec.get('epochs', 10)
    
    logger.info(f"Creating TrainingJob {name} for model {model} with {epochs} epochs")
    
    # 1. create a K8s Job programmatically
    api = k8s.BatchV1Api()
    
    job_body = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"{name}-job",
            "namespace": namespace
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "train",
                        "image": "python:3.9",
                        "command": ["echo", f"Training {model} for {epochs} epochs"]
                    }],
                    "restartPolicy": "Never"
                }
            }
        }
    }
    
    # Make the Job a child of the CustomResource (Cascading Deletion)
    kopf.adopt(job_body)
    
    obj = api.create_namespaced_job(namespace, job_body)
    return {'job-name': obj.metadata.name}

# To run: kopf run src/operator.py --verbose
```

### 👨‍💻 Lab Operations

1.  **Run Operator:**
    ```bash
    kopf run src/operator.py
    ```
    *(Keep running in terminal)*
2.  **Create Custom Resource:**
    ```yaml
    # job.yaml
    apiVersion: ai.example.com/v1
    kind: TrainingJob
    metadata:
      name: my-resnet
    spec:
      model: resnet50
      epochs: 100
    ```
    `kubectl apply -f job.yaml`
3.  **Observation:**
    *   The Operator CLI prints logs.
    *   `kubectl get job` shows `my-resnet-job`.
    *   It worked! K8s created a Job *because* we created a TrainingJob.

---

## 🔬 Lab Exercise: "Automated Cleanup"

### Task
Test "OwnerReferences".
1.  Run `kubectl delete tj my-resnet`.
2.  Check pods/jobs.
3.  **Observation:** The underlying `Job` (my-resnet-job) is automatically deleted.
4.  **Why?** `kopf.adopt(job_body)` set the `TrainingJob` as the parent. K8s Garbage Collector cleans up orphans.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Complexity:** Operators are harder to write/debug than Helm Charts. Use them only when you need active state management (e.g., Database failover).
2.  **Ecosystem:** Don't write your own if possible. Use **Prometheus Operator**, **Kubeflow Operator**, **Postgres Operator**.
3.  **Logic:** The Controller must be idempotent. It should calculate `Current - Desired = Diff` and apply `Diff`.

### API Summary
```bash
kubectl get crd
kubectl api-resources | grep ai
kubectl get trainingjobs
```

---

**Day 59 Complete** ✅

*Next: Day 60 - KubeRay Operator - The standard for distributed computing on K8s.*
