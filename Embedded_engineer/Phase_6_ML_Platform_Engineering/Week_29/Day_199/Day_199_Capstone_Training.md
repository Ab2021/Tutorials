# Day 199: Titan Phase 2: The Training Pipeline with KubeRay
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 29: Capstone Project Part 1

---

> **🎯 Focus Area:** We have the clusters. Now we need the Engine. **KubeRay** allows us to submit massive distributed training jobs to the `titan-gpu` cluster, manage Python dependencies dynamically, and autoscale workers based on demand.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the KubeRay Operator and a default `RayCluster` via Helm.
2.  **Submit** a Distributed PyTorch Training Job (`RayJob`) that uses 4 GPUs.
3.  **Implement** Auto-Scaling (Scale from 0 to 10 workers) using Ray Autoscaler.
4.  **Debug** a failed training actor using Ray Dashboard and Kubernetes Logs.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-gpu` Cluster.

### Software Environment
- `helm install kuberay-operator`.

---

## 📖 Theoretical Foundation

### 1. The Ray Pattern
*   **RayCluster:** A set of Pods (Head + Workers).
*   **RayJob:** A custom resource that creates a RayCluster, submits a python script, waits for completion, and deletes the cluster. (Ephemeral).
*   **KubeRay Operator:** The controller that watches `RayJob` CRs and talks to K8s API.

### 2. Dependency Management
*   **Docker Image:** Bake deps into `my-image:v1`. (Best for large deps like PyTorch).
*   **RuntimeEnv:** Install deps at runtime via `pip`. `runtime_env={"pip": ["transformers"]}`. (Best for small, fast-moving deps).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: KubeRay Installation

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0
```

### 👨‍💻 Core Implementation: The RayJob Manifest

This defines the compute infrastructure AND the code to run.

#### 📁 `manifests/jobs/train-resnet.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: train-resnet
spec:
  # 1. Submission Config
  entrypoint: python train.py
  runtimeEnvYAML: |
    pip:
      - torchvision
      - tqdm
    working_dir: "https://github.com/myorg/models/archive/main.zip"

  # 2. Cluster Config
  rayClusterSpec:
    rayVersion: '2.9.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-gpu
            resources:
              requests:
                cpu: 2
                memory: 4Gi
    workerGroupSpecs:
    - replicas: 2
      minReplicas: 0
      maxReplicas: 10
      groupName: gpu-group
      rayStartParams: {}
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.9.0-gpu
            resources:
              limits:
                nvidia.com/gpu: 1
              requests:
                cpu: 4
                memory: 16Gi
```

### 👨‍💻 Core Implementation: The Training Script (Ray Train)

#### 📁 `src/train.py`
```python
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torchvision

def train_func(config):
    # Standard PyTorch Training Loop
    model = torchvision.models.resnet18()
    # Ray handles DistributedDataParallel (DDP) setup
    model = train.torch.prepare_model(model)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        # ... training logic ...
        train.report({"loss": 0.5, "epoch": epoch})

if __name__ == "__main__":
    ray.init()
    
    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True)
    )
    
    result = trainer.fit()
    print(f"Training Result: {result.metrics}")
```

---

## 🔬 Lab Exercise: "The Scale Up"

### Task
Observe Autoscaling.
1.  **Submit Job:** `replicas: 0` (Start with Head only).
2.  **Script:** `ScalingConfig(num_workers=4)`.
3.  **Observation:**
    *   Ray Head starts.
    *   Job script runs. Request 4 GPU workers.
    *   Ray Autoscaler (in Head) sees pending demand.
    *   Ray Autoscaler requests 4 Pods from K8s.
    *   K8s (Cluster Autoscaler) sees pending Pods. Requests 4 Nodes from AWS.
    *   Nodes join. Pods start. Training begins.
4.  **Duration:** ~3-5 minutes for cold start.

---

## 📖 Advanced Theory: Fault Tolerance
What if a worker dies mid-training?
*   **Ray Train:** Automatically detects worker failure.
*   **Action:** If `checkpoint` exists, it restarts training from the last checkpoint on the remaining workers (or waits for replacement).
*   **Config:** `run_config=RunConfig(failure_config=FailureConfig(max_failures=3))`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Ephemeral Clusters:** Don't keep a giant Ray Cluster running 24/7. Use `RayJob` to create clusters on-demand. This saves massive amounts of money (Spot Instances).
2.  **Dashboard:** Access the Ray Dashboard via `kubectl port-forward svc/train-resnet-raycluster-head-svc 8265:8265`. It shows logs, GPU utilization, and actor status.
3.  **Working Dir:** `runtime_env.working_dir` allows you to inject code from a ZIP or Git URL. No need to build a new Docker image for every script change.

### API Summary
```bash
kubectl get rayjobs
kubectl logs -l ray.io/node-type=head
```

---

**Day 199 Complete** ✅

*Next: Day 200 - Capstone Part 4 - The Serving Pipeline.*
