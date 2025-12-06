# Day 92: The Bridge: KubeRay Operator
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** Managing Ray processes manually on EC2 is "Day 1". "Day 2" is using the **KubeRay Operator** to automate the lifecycle of Ray Clusters on Kubernetes.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the role of the KubeRay Operator (Controller Pattern).
2.  **Install** the KubeRay Operator using Helm.
3.  **Analyze** the logs of the Operator to understand its reconciliation loop.
4.  **Differentiate** between the Operator Pod and the Ray Cluster Pods.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster (Minikube/Kind).

### Software Environment
- `helm`.
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Operator Pattern (Recap)
An Operator watches a Custom Resource (CR) and makes the K8s state match the spec.
*   **CR:** `RayCluster` (I want 1 Head, 3 Workers).
*   **Controller:** Sees CR. Creates 1 Head Pod (Service, ConfigMap). Creates 3 Worker Pods. Connects Workers to Head.

### 2. KubeRay Components
*   **KubeRay Operator:** One per cluster (or namespace). The "Brain".
*   **CRDs:** `RayCluster`, `RayJob`, `RayService`.
*   **Ray Head Pod:** Runs Ray GCS (Global Control Store), Dashboard, and Scheduler.
*   **Ray Worker Pod:** Runs `raylet` and user code.

### 3. Networking
The Operator ensures:
*   Head Node gets a `Service` (for Workers to connect to GCS).
*   Head Node exposes Dashboard ports (8265).
*   Workers discover Head via DNS (`mycluster-head-svc`).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Install Operator

We use the official Helm chart.

```bash
# 1. Add Repo
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# 2. Install Operator in specific namespace
helm install kuberay-operator kuberay/kuberay-operator \
    --namespace ray-system \
    --create-namespace \
    --version 1.0.0
```

### 👨‍💻 Verification

1.  **Check Pods:**
    ```bash
    kubectl get pods -n ray-system
    # kuberay-operator-xxxx  1/1  Running
    ```
2.  **Check CRDs:**
    ```bash
    kubectl get crds | grep ray
    # rayclusters.ray.io
    # rayjobs.ray.io
    # rayservices.ray.io
    ```

### 👨‍💻 Lab: Inspecting the Reconciliation Loop

1.  Tail the logs:
    ```bash
    kubectl logs -f deploy/kuberay-operator -n ray-system
    ```
2.  It will be quiet. It waits for a `RayCluster` event.
3.  (Preview): When we apply a cluster, we will see "Reconciling RayCluster... Creating Head Pod... Creating WorkerGroup...".

---

## 🔬 Lab Exercise: "Operator Versioning"

### Task
Why does version matter?
1.  Ray releases fast (monthly). KubeRay releases slower.
2.  **Compatibility Matrix:** Ensure your KubeRay Operator version supports the Ray Image version you intend to use.
3.  **Check:** `kuberay 1.0.0` generally supports `ray 2.4` to `2.9`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Batteries Included:** KubeRay handles the complexity of "How does Worker find Head?". It injects the `ray start --address=head-svc:6379` command automatically.
2.  **Stateless Operator:** The Operator itself is stateless. If it crashes, the running Ray Clusters *keep running*. The Operator just loses the ability to heal them until it restarts.
3.  **Helm:** Always use Helm. The YAML manifests are too complex to manage manually.

### API Summary
```bash
helm install kuberay-operator
```

---

**Day 92 Complete** ✅

*Next: Day 93 - RayCluster CRD - Defining your compute topology.*
