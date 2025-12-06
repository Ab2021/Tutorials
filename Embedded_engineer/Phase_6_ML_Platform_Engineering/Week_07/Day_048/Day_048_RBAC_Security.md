# Day 48: Who Goes There? RBAC & Multi-Tenancy
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** Security. Master **Role-Based Access Control (RBAC)** to ensure that your Training Job can read S3 buckets but cannot delete the Production Database.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** the Core RBAC Components: ServiceAccount, Role, and RoleBinding.
2.  **Contrast** Namespace-scoped Roles vs Cluster-scoped ClusterRoles.
3.  **Create** a restricted `view-only` user for a specific namespace.
4.  **Assign** a ServiceAccount to a Pod to give it specific API permissions.
5.  **Audit** permissions using `kubectl auth can-i`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Identity in K8s
*   **Users:** K8s does NOT store users. It relies on External IDPs (Google, AWS IAM, OIDC).
*   **ServiceAccounts (SA):** K8s DOES manage these. They are identities for Processes (Pods). Every Pod runs as `default` SA unless specified.

### 2. The RBAC Triad
1.  **Subject:** Who? (User "Dave", ServiceAccount "monitoring-bot").
2.  **Role:** What? (Can "get", "list", "watch" on "pods").
3.  **Binding:** Connection. (Attach Role to Subject).

### 3. Namespace Isolation
Namespaces (`dev`, `prod`, `monitoring`) are the primary boundary.
*   A **Role** exists only inside a Namespace.
*   A **ClusterRole** exists globally.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The "Viewer" Role

We will create a ServiceAccount that can *only* list pods, nothing else.

#### 📁 `manifests/rbac-setup.yaml`
```yaml
# 1. The Identity
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pod-viewer
  namespace: default
---
# 2. The Permission Rule
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: default
rules:
- apiGroups: [""] # Core API Group
  resources: ["pods", "pods/log"]
  verbs: ["get", "watch", "list"]
---
# 3. The Binding (Connection)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: view-binding
  namespace: default
subjects:
- kind: ServiceAccount
  name: pod-viewer
  namespace: default
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### 👨‍💻 Lab Operations: Testing Permissions

1.  **Apply Manifests:**
    ```bash
    kubectl apply -f manifests/rbac-setup.yaml
    ```
2.  **Impersonation Test (Admin only):**
    As Admin, you can check what `pod-viewer` can do.
    ```bash
    # Can it list pods?
    kubectl auth can-i list pods --as=system:serviceaccount:default:pod-viewer
    # Output: yes

    # Can it delete pods?
    kubectl auth can-i delete pods --as=system:serviceaccount:default:pod-viewer
    # Output: no
    
    # Can it list secrets? (Crucial check)
    kubectl auth can-i get secrets --as=system:serviceaccount:default:pod-viewer
    # Output: no
    ```

### 👨‍💻 Using SA in a Pod

If you have a Python script that uses `kubernetes` python client to watch pods, you must attach this SA.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: watcher-py
spec:
  serviceAccountName: pod-viewer # <--- The Magic Line
  containers:
  - name: app
    image: my-k8s-client-app
```

---

## 🔬 Lab Exercise: "Privilege Escalation Prevention"

### Task
Try to grant `cluster-admin` (God mode) to `pod-viewer` via a RoleBinding.
1.  Create a RoleBinding tying `pod-viewer` to `cluster-admin`.
2.  Apply it.
3.  **Observation:** If you are currently Admin, it works. But if you were a User with limited rights, K8s prevents you from granting permissions *you don't have*.

### Insight
RBAC is the firewall of your cluster. If an attacker compromises a Pod, they become the ServiceAccount of that Pod. If that SA has `ClusterAdmin` (often seen in poorly configured Helm charts), they own your cluster. **Principle of Least Privilege.**

---

## 📝 Daily Summary

### Key Takeaways
1.  **Default is Deny:** If no RoleBinding exists, you can do nothing (403 Forbidden).
2.  **Namespace vs Cluster:** Use **Role** for application permissions. Use **ClusterRole** only for system-wide agents (e.g., Node Monitoring, Ingress Controllers).
3.  **Automate:** Don't create bindings manually. Use Terraform or GitOps (Day 61) to manage permissions auditably.

### API Summary
```bash
kubectl create serviceaccount <name>
kubectl create rolebinding <name> --role=<role> --serviceaccount=<ns:sa>
kubectl auth can-i <verb> <resource> --as=<user>
```

---

**Day 48 Complete** ✅

*Next: Day 49 - Week 7 Review & Project - Deploying a complete Microservices Application with Ingress, Storage, and RBAC.*
