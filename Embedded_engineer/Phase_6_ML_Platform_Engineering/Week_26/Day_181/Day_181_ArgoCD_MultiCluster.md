# Day 181: Deployment at Scale: Multi-Cluster CI/CD with ArgoCD
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 26: Multi-Cluster & Federation

---

> **🎯 Focus Area:** You have 50 clusters. Manually applying `kubectl apply` to each is impossible. **ArgoCD ApplicationSets** allow you to define a single "Template" that generates Applications for every cluster automatically.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** ArgoCD to manage remote clusters.
2.  **Define** an ApplicationSet with the **Cluster Generator** to target all production clusters.
3.  **Implement** Progressive Rollouts (Deploy to `dev` first, then `staging`, then `prod` waves).
4.  **Templatize** specific overrides (e.g., `replicas: 5` in US, `replicas: 2` in EU).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `helm install argo-cd`.

---

## 📖 Theoretical Foundation

### 1. The ApplicationSet Controller
Standard ArgoCD `Application` maps 1 Git Repo -> 1 Cluster.
`ApplicationSet` maps 1 Git Repo -> N Clusters.
It uses **Generators** to produce the list of clusters:
*   **List Generator:** Hardcoded list.
*   **Cluster Generator:** "All clusters labeled `env=prod`".
*   **Git Generator:** "Every folder in `clusters/` repo".

### 2. Progressive Sync
If you push a bug, you don't want to break all 50 clusters simultaneously.
**Sync Waves** in ApplicationSet allow you to say:
*   Wave 1: `cluster-canary`.
*   Wave 2: `cluster-us-east`.
*   Wave 3: `cluster-global`.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Registering Clusters

ArgoCD needs credentials to talk to remote clusters.

```bash
# Register a cluster context
argocd cluster add kind-member1 --name member1 --label env=prod --label region=us
argocd cluster add kind-member2 --name member2 --label env=prod --label region=eu
```

### 👨‍💻 Core Implementation: The Cluster Generator

Deploy "Inference Service" to ALL Prod clusters.

#### 📁 `manifests/appset-prod.yaml`
```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: inference-global
  namespace: argocd
spec:
  goTemplate: true
  generators:
  - clusters:
      selector:
        matchLabels:
          env: prod
  template:
    metadata:
      name: 'inference-{{.name}}' # e.g., inference-member1
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/models
        targetRevision: HEAD
        path: charts/inference-service
        helm:
          values: |
            region: {{.metadata.labels.region}}
            replicaCount: {{ if eq .metadata.labels.region "us" }}10{{ else }}5{{ end }}
      destination:
        server: '{{.server}}'
        namespace: production
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

### 👨‍💻 Core Implementation: Matrix Generator (Combinatorial)

Deploy 3 Models (ResNet, Bert, GPT) to 2 Clusters (US, EU). Total 6 Apps.

#### 📁 `manifests/appset-matrix.yaml`
```yaml
spec:
  generators:
  - matrix:
      generators:
      - git: # Discover Models from Git directories
          repoURL: https://github.com/myorg/models
          revision: HEAD
          directories:
          - path: models/*
      - clusters: # Discover Clusters
          selector:
            matchLabels:
              env: prod
  template:
    metadata:
      name: '{{.path.basename}}-{{.name}}' # e.g., bert-member1
    spec:
      source:
        path: '{{.path}}'
        # ...
```

---

## 🔬 Lab Exercise: "The Rollout"

### Task
Simulate a Bad Config Push.
1.  Define ApplicationSet targeting `dev` and `prod` clusters.
2.  **Config:** Use `RollingSync` (progressive).
3.  **Action:** Push a breaking change (Invalid Image Tag) to Git.
4.  **Observation:**
    *   ArgoCD syncs `dev` cluster first.
    *   `dev` Deployment fails (ImagePullBackOff).
    *   ArgoCD **Pre-Sync Analysis** (if configured) or Health Check detects failure.
    *   **Result:** `prod` sync is BLOCKED. Production is saved.

---

## 📖 Advanced Theory: ApplicationSet vs Karmada
*   **Karmada:** 1 Deployment object in Host -> Split into members. (Best for "One logical app spanning clusters").
*   **ArgoCD AppSet:** N Application objects -> N Deployments. (Best for "N independent copies of the app").
*   **Verdict:** Use AppSet for "Fleet Management" (e.g., installing Prometheus/Drivers on all clusters). Use Karmada for "Workload Distribution" (e.g., Bursting 50% traffic to Cloud).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Templating:** The `goTemplate: true` feature in ApplicationSet is powerful. You can inject Cluster Labels (`region`, `gpu_type`) directly into Helm Values.
2.  **Drift Detection:** ArgoCD warns you if someone manually ran `kubectl edit` in a remote cluster.
3.  **Generators:** The `PullRequest` generator is useful for "Ephemeral Environments". It creates a temporary cluster/namespace for every PR.

### API Summary
```yaml
kind: ApplicationSet
generators:
  - clusters: {}
```

---

**Day 181 Complete** ✅

*Next: Day 182 - Week 26 Review & Project - The Federation.*
