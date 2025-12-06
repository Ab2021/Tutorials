# Day 63: Week 9 Review & Project - GitOps ML Platform
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** The "ClickOps" era is over. Today, we build a **GitOps Deployment Pipeline**. You commit code; the machine does the rest.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Structure** a Git Repository for Multi-Environment GitOps.
2.  **Package** an ML application into a versioned Helm Chart.
3.  **Bootstrap** ArgoCD using the "App of Apps" pattern.
4.  **Execute** a Promotion workflow (Update Dev -> Verify -> Update Prod).

---

## 📚 Week 9 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 57 | Helm Basics | "YAML is data; Templates are logic." |
| 58 | Helm Advanced | "Hooks let me migrate the DB *before* the code lands." |
| 59 | Operators | "I can extend the K8s API with Python." |
| 60 | KubeRay | "Ray Clusters should be disposable." |
| 61 | GitOps Principles | "If it's not in Git, it's a hallucination." |
| 62 | ArgoCD | "The cluster fixes itself." |

---

## 🏗️ Final Project: "Ops-less Inference"

### Architecture
1.  **Repo A (Source):** Contains the Helm Chart source code.
2.  **Repo B (Config):** Contains the `values.yaml` for Dev and Prod.
3.  **ArgoCD:** Watches Repo B and syncs to Cluster.

*(For simplicity in this lab, we will use Monorepo: `chart/` and `envs/` in same folder).*

### Step 1: The Helm Chart

Create a generic chart `charts/tf-serving`.

#### 📁 `project/charts/tf-serving/values.yaml` (Defaults)
```yaml
modelName: "resnet"
replicas: 1
image: "tensorflow/serving:latest"
gpu: false
```

#### 📁 `project/charts/tf-serving/templates/deploy.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Values.modelName }}
spec:
  replicas: {{ .Values.replicas }}
  selector:
    matchLabels:
      app: {{ .Values.modelName }}
  template:
    metadata:
      labels:
        app: {{ .Values.modelName }}
    spec:
      containers:
      - name: tf-serving
        image: {{ .Values.image }}
        ports:
        - containerPort: 8501
        env:
        - name: MODEL_NAME
          value: {{ .Values.modelName }}
        {{- if .Values.gpu }}
        resources:
          limits:
            nvidia.com/gpu: 1
        {{- end }}
```

### Step 2: Environment Configuration

We define two environments with different configs.

#### 📁 `project/envs/dev/values.yaml`
```yaml
modelName: "resnet-dev"
replicas: 1
gpu: false # Save money
```

#### 📁 `project/envs/prod/values.yaml`
```yaml
modelName: "resnet-prod"
replicas: 3
gpu: true # High Performance
```

### Step 3: ArgoCD Applications

We tell ArgoCD about these environments.

#### 📁 `project/argocd/dev-app.yaml`
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-dev
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/YOUR_USER/gitops-lab.git
    targetRevision: main
    path: project/charts/tf-serving
    helm:
      valueFiles:
      - ../../envs/dev/values.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: dev
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

#### 📁 `project/argocd/prod-app.yaml`
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-prod
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/YOUR_USER/gitops-lab.git
    targetRevision: main
    path: project/charts/tf-serving
    helm:
      valueFiles:
      - ../../envs/prod/values.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: prod
  syncPolicy: # Manual Sync for Prod safety
    automated: null 
```

### Step 4: The Workflow Simulation

1.  **Bootstrap:**
    ```bash
    kubectl create ns dev
    kubectl create ns prod
    # In real life, commit these to Git. Locally, we apply directly to simulate the Repo existance.
    kubectl apply -f project/argocd/
    ```
    *Result:* ArgoCD creates `ml-dev` (Healthy) and `ml-prod` (OutOfSync, waiting for manual trigger).

2.  **Dev Update:**
    *   Edit `project/envs/dev/values.yaml`: Change `replicas: 2`.
    *   Commit & Push.
    *   ArgoCD (Auto) sees change -> Scales Dev deployment to 2.

3.  **Promotion:**
    *   Edit `project/envs/prod/values.yaml`: Change `image: tensorflow/serving:2.8.0` (Simulating upgrade).
    *   Commit & Push.
    *   ArgoCD (Manual) shows "OutOfSync" (Diff: Tag latest -> 2.8.0).
    *   Engineer clicks "Sync".
    *   Prod updates.

---

## 📝 Success Criteria
1.  **Visualization:** ArgoCD UI shows two distinct Applications pointing to the same Cluster but controlled by different Values files.
2.  **Consistency:** `kubectl get deploy -n dev` matches exactly what is in `envs/dev/values.yaml`.
3.  **Idempotency:** A `kubectl apply -f` of the Helm Chart locally is NOT used. Only ArgoCD applies changes.

---

**Week 9 Complete** ✅

*Next Week: Week 10 - Cloud Platforms for ML - Taking this stack to AWS EKS and GCP GKE.*
