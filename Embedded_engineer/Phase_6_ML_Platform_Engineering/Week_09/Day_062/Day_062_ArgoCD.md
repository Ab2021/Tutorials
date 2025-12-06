# Day 62: The GitOps Engine: ArgoCD
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** Turn Principles into Practice. Deploy **ArgoCD**, the industry-standard Continuous Delivery tool for Kubernetes, and use it to manage an application.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Install** ArgoCD into a Kubernetes Cluster.
2.  **Access** the ArgoCD UI and understand the concept of "Apps of Apps".
3.  **Define** an `Application` manifest that maps a Git Repo to a Namespace.
4.  **Demonstrate** Self-Healing: Delete a Deployment and watch ArgoCD recreate it.
5.  **Configure** Auto-Sync and Pruning policies.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `kubectl`.
- `argocd` CLI (optional, but helpful).

---

## 📖 Theoretical Foundation

### 1. Architecture Components
*   **Repo Server:** A cache of your Git Repositories. It clones them and runs `helm template` or `kustomize build`.
*   **Application Controller:** Compares the Live State (K8s) vs Target State (Repo Server). Status: `Synced` or `OutOfSync`.
*   **Redis:** Caching.

### 2. The Application Object
ArgoCD uses a CRD `kind: Application` to define a deployment.
*   **Source:** `https://github.com/my/repo`, `branch: main`, `path: /overlays/prod`.
*   **destination:** `server: https://kubernetes.default.svc`, `namespace: production`.

### 3. Sync Options
*   **Prune:** If I delete `job.yaml` from Git, should ArgoCD delete it from K8s? (Default: No. Safety feature).
*   **Self-Heal:** If someone manually runs `kubectl delete`, should ArgoCD revert it? (Default: No. It just shows 'OutOfSync').

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Installing ArgoCD

We create a dedicated namespace for the comprehensive install.

```bash
# 1. Create Namespace
kubectl create namespace argocd

# 2. Apply Manifest (Official Stable)
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 3. Wait for Pods
kubectl wait --for=condition=Ready pods --all -n argocd

# 4. Access UI
# Default user: admin
# Password: is in a secret
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo

# Port Forward
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Open https://localhost:8080 (Accept invalid cert)
```

### 👨‍💻 Core Implementation: First App

We will deploy the official "Guestbook" example.

#### 📁 `manifests/guestbook-app.yaml`
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: guestbook
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/argoproj-labs/argocd-example-apps.git
    targetRevision: HEAD
    path: guestbook
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

Apply it: `kubectl apply -f manifests/guestbook-app.yaml`.

### 👨‍💻 Lab Operations

1.  **Check UI:** You should see "guestbook" appearing. It will read "Synced" and "Healthy".
2.  **Check Cluster:** `kubectl get deploy guestbook-ui`.
3.  **Sabotage:**
    ```bash
    kubectl delete deploy guestbook-ui
    ```
4.  **Watch Self-Healing:** Within seconds (or immediately if using Refresh), ArgoCD detects the deletion and re-applies the manifest.

---

## 🔬 Lab Exercise: "The Bad Commit"

### Task
Simulate a broken config.
1.  Fork the example repo (or use your own).
2.  Point the Application to your fork.
3.  Commit a change with valid YAML but invalid Logic (e.g., `image: nginx:non-existent-tag`).
4.  **Observation:**
    *   ArgoCD Syncs.
    *   Deployment updates.
    *   Pods fail (`ImagePullBackOff`).
    *   ArgoCD Status: `Synced` (Config matches) but `Degraded` (Health check failed).

### Insight
GitOps ensures Configuration Consistency. It does not guarantee Application Logic Correctness. You still need **Helm Tests** or **Rollout Strategies** (Argo Rollouts) to catch bad code.

---

## 📝 Daily Summary

### Key Takeaways
1.  **UI is Read-Only (Mostly):** While you *can* create apps in the UI, true GitOps means defining even the Argo Apps themselves in Git ("App of Apps" pattern).
2.  **Secret Management:** ArgoCD cannot read encrypted secrets in Git by default. You need plugins like **ArgoCD-Vault-Plugin** or **SealedSecrets**.
3.  **Single Pane of Glass:** ArgoCD can manage *multiple* clusters. You can see Prod, Staging, and Dev status all in one dashboard.

### API Summary
```bash
argocd app list
argocd app sync <appname>
argocd app history <appname>
```

---

**Day 62 Complete** ✅

*Next: Day 63 - Week 9 Review & Project - Building a GitOps Platform for ML Model Deployment.*
