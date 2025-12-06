# Day 61: The End of `kubectl apply`: GitOps Principles
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** Manual operations are the root of all outages. Adopt **GitOps**: a paradigm where Git is the *only* interface to your infrastructure, and software agents ensure reality matches the repo.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Imperative (`kubectl scale`) vs Declarative (`git commit`) workflows.
2.  **Explain** the security benefits of the "Pull Model" (ArgoCD) vs "Push Model" (Jenkins/GitHub Actions).
3.  **Structure** a GitOps repository for Multi-Environment (Dev/Stage/Prod) apps.
4.  **Define** Configuration Drift and how GitOps solves it.
5.  **Roleplay** a production rollback using `git revert`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Conceptual Design Day).

### Software Environment
- Git.

---

## 📖 Theoretical Foundation

### 1. The Drift
*   **Repo:** `replicas: 1`.
*   **Cluster:** `replicas: 10` (Because an on-call engineer fixed an outage last night and forgot to commit).
*   **Event:** Deployment runs.
*   **Result:** Cluster reverts to 1 replica. Outage returns.
*   **Solution:** **GitOps**. The on-call engineer *must* configure the cluster via Git (even at 3 AM).

### 2. Push vs Pull
*   **Push (CI Pipelins):** Jenkins/GitHub builds Docker image, then runs `kubectl apply`.
    *   *Risk:* Jenkins needs `KUBECONFIG` with Admin access to Prod. If Jenkins is hacked, Prod is gone.
*   **Pull (GitOps Operator):** ArgoCD lives *inside* the cluster. It watches Git. When Git changes, It applies the change to itself.
    *   *Security:* No external credentials. ArgoCD only needs Read-Only access to Git.

### 3. The Sync Loop
1.  **Desired State:** Manifests in Git.
2.  **Live State:** Objects in Etcd.
3.  **Reconciliation:** Agent calculates `Diff`. If `Diff > 0`, Sync.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Repo Structure

A well-structured GitOps repo avoids spaghetti configs.

#### Option A: Monorepo (Code + Config)
Good for small teams.
```text
/
├── src/                # Python Code
├── Dockerfile
└── k8s/
    ├── dev/
    │   └── values.yaml # Helm override for Dev
    └── prod/
        └── values.yaml # Helm override for Prod
```

#### Option B: Config Repo (Separation of Concerns)
Enterprise Standard. Application code in Repo A. Infrastructure config in Repo B.
**Repo: infrastructure-live**
```text
/
├── apps/
│   ├── model-training/
│   │   ├── base/           # Common YAMLs
│   │   ├── overlays/
│   │   │   ├── dev/        # Dev specifics (kustomize)
│   │   │   └── prod/       # Prod specifics
│   ├── inference-api/
│   │   └── ...
└── cluster-admin/          # Namespaces, RBAC, Quotas
```

### 👨‍💻 Workflow Simulation

**Task:** Increase Memory Limit for `inference-api`.

**Old Way (Bad):**
1.  `kubectl edit deploy inference-api`.
2.  Change memory.
3.  Save.
4.  Forget to tell anyone.

**GitOps Way (Good):**
1.  `git checkout -b chore/increase-mem`.
2.  Edit `k8s/prod/values.yaml`.
3.  `git commit -m "Bump memory to 4Gi"`.
4.  `git push`.
5.  Open Pull Request.
6.  Team Lead approves & merges.
7.  ArgoCD detects change and syncs.

---

## 🔬 Lab Exercise: "The Human Controller"

### Task
Simulate being ArgoCD manually.
1.  Create a folder `gitops-repo/` locally.
2.  Add `pod.yaml`.
3.  Apply it: `kubectl apply -f pod.yaml`.
4.  **Edit `pod.yaml`** (Change image tag).
5.  **Check Drift:**
    ```bash
    kubectl diff -f pod.yaml
    ```
    *Result:* It shows the difference between file and cluster.
6.  **Sync:**
    ```bash
    kubectl apply -f pod.yaml
    ```
    *Result:* Drift eliminated.

### Insight
ArgoCD is just a robot running `kubectl diff` and `kubectl apply` in a loop every 3 minutes.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Code Review for Infra:** Infrastructure changes go through Pull Requests. You get audit logs ("Alice approved Bob's change") for free.
2.  **Easy Rollbacks:** `git revert <commit-hash>`. Push. Done. The cluster goes back in time.
3.  **Disaster Recovery:** If the cluster blows up, point a new cluster at the Git Repo. It rebuilds itself perfectly.

### API Summary
```bash
kubectl diff -f file.yaml
```

---

**Day 61 Complete** ✅

*Next: Day 62 - ArgoCD Implementation - Deploying the GitOps engine.*
