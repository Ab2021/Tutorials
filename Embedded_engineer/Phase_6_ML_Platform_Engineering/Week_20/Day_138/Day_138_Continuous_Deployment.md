# Day 138: Go For Launch: Continuous Deployment (CD)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** "It works on my machine" is not an excuse. **Continuous Deployment** automates the promotion of your trained model into a production Kubernetes cluster, utilizing Safe Deployment strategies.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between CD for Code (Rebuilding Image) and CD for Models (Updating URI).
2.  **Implement** a Canary Deployment (10% Traffic) using Istio or Ray Serve.
3.  **Execute** a Shadow Deployment to validate a new model on real traffic without risk.
4.  **Automate** the rollback procedure if error rates spike.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- `pip install ray[serve] kubernetes`.

---

## 📖 Theoretical Foundation

### 1. The Deployment Artifact
What are we deploying?
*   **Approach A (Baked Image):** Build a new Docker Image with `model.pkl` inside.
    *   *Pros:* Atomic. Immutable.
    *   *Cons:* Slow build (GBs).
*   **Approach B (Runtime Load):** Container runs generically. It downloads `model_v2.pkl` at startup based on Environment Variable.
    *   *Pros:* Fast.
    *   *Cons:* Verification happens at runtime.

### 2. Deployment Strategies (Recap)
*   **Recreate:** Kill V1. Start V2. (Downtime).
*   **Rolling:** Update 1 replica at a time. (Zero Downtime).
*   **Blue/Green:** Spin up full V2 cluster. Switch Load Balancer. (Safe, Expensive).
*   **Canary:** Route 1% traffic to V2. Monitor. Increase to 100%.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: GitOps Deployment (ArgoCD Style)

We don't run `kubectl apply` in CI. We commit to a `manifests/` repo.

#### 📁 `manifests/values.yaml`
```yaml
# The CI pipeline updates this file
model:
  name: fraud-detector
  version: "production-v1.2.3" # CI updates this
  uri: "s3://models/fraud/v1.2.3"
  trafficSplit:
    canary: 0 # Initially 0
```

### 👨‍💻 Core Implementation: The CD Pipeline (GitHub Actions)

Scenario: Training finished. Model Registered. We want to deploy to Staging.

#### 📁 `.github/workflows/deploy.yaml`
```yaml
name: Deploy Model
on:
  workflow_dispatch:
    inputs:
      model_version:
        description: "Version to deploy"
        required: true

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          repository: myorg/infra-manifests # Separate Repo
          
      - name: Update Staging Manifest
        run: |
          # Use yq to update yaml
          yq e '.model.version = "${{ inputs.model_version }}"' -i staging/values.yaml
          
          git config user.name "CI Bot"
          git commit -am "Deploy Model ${{ inputs.model_version }} to Staging"
          git push
          
      # ArgoCD picks this up and syncs the cluster.
```

### 👨‍💻 Infrastructure: Canary Logic (Istio VirtualService)

If using K8s directly:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: fraud-service
spec:
  hosts:
  - fraud.example.com
  http:
  - route:
    - destination:
        host: fraud-v1
        subset: v1
      weight: 90
    - destination:
        host: fraud-v2
        subset: v2
      weight: 10
```

### 👨‍💻 Core Implementation: Automated Rollout Script (Python)

If not using Argo Rollouts, manage the rollout via script.

#### 📁 `scripts/rollout.py`
```python
import time
import requests

def health_check(url):
    try:
        r = requests.get(url + "/health")
        return r.status_code == 200
    except:
        return False

def rollout(canary_weight):
    print(f"Setting Canary to {canary_weight}%")
    # Call K8s/Ray API to update traffic split
    # update_traffic_split(weight=canary_weight)
    
    # Wait for stabilization
    time.sleep(60)
    
    # Monitoring
    error_rate = get_metric("error_rate")
    latency = get_metric("p99_latency")
    
    if error_rate > 0.01:
        raise RuntimeError(f"Error rate spike: {error_rate}. Rollling back!")
        
    print("Metrics healthy.")

# Main Loop
try:
    steps = [10, 25, 50, 100]
    for step in steps:
        rollout(step)
    print("Deployment Successful!")
except RuntimeError as e:
    print(e)
    print("Executing Rollback...")
    rollout(0)
```

---

## 🔬 Lab Exercise: "The Shadow"

### Task
Implement Shadow Mode.
1.  Deploy V2 alongside V1.
2.  Configure Ingress (e.g., Nginx Mirroring) to send traffic to *both*.
3.  V1 returns response to user. V2 processes it but response is discarded.
4.  **Logging:** Both V1 and V2 log predictions to ElasticSearch/W&B.
5.  **Comparison:** Run analysis. "Did V2 crash? Did V2 latency spike? Did V2 predict differently?"
6.  **Benefit:** Zero risk to user.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Metric-Driven:** CD is not just "pushing code". It is "pushing code and watching metrics". If `latency` > 200ms, the CD system must auto-rollback.
2.  **State:** ML Models are stateless (usually). This makes Blue/Green easier than upgrading a Database.
3.  **Config:** Keep config (Traffic Split %s) in Git. If you change it manually via CLI, Git is out of sync (Drift).

### API Summary
```bash
kubectl rollout status deployment/v2
kubectl rollout undo deployment/v2
```

---

**Day 138 Complete** ✅

*Next: Day 139 - Infrastructure as Code (IaC) - Terraform for ML.*
