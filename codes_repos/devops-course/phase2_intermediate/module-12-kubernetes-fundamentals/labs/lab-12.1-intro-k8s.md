# Lab 12.1: Introduction to Kubernetes (K8s)

## 🎯 Objective

Welcome to the beast. Kubernetes is the de-facto standard for container orchestration. In this lab, you will install a local cluster and explore the architecture.

## 📋 Prerequisites

-   Docker installed.
-   A lot of RAM (at least 4GB free).

## 📚 Background

### The Architecture
1.  **Control Plane (Master)**: The Brain.
    -   **API Server**: The front door (kubectl talks to this).
    -   **Etcd**: The database (stores cluster state).
    -   **Scheduler**: Decides where to put pods.
    -   **Controller Manager**: Ensures desired state (e.g., "Keep 3 replicas").
2.  **Worker Nodes**: The Muscle.
    -   **Kubelet**: Agent that runs on the node.
    -   **Kube-proxy**: Handles networking.
    -   **Container Runtime**: Docker/Containerd.

---

## 🔨 Hands-On Implementation

### Part 1: Install Tools 🛠️

1.  **Install kubectl** (The CLI):
    -   [Official Docs](https://kubernetes.io/docs/tasks/tools/).
    -   Verify: `kubectl version --client`.

2.  **Install Minikube** (Local Cluster):
    -   [Official Docs](https://minikube.sigs.k8s.io/docs/start/).
    -   *Alternative:* **Kind** (Kubernetes in Docker) or **Docker Desktop** (Enable Kubernetes).

### Part 2: Start the Cluster 🚀

1.  **Start Minikube:**
    ```bash
    minikube start
    ```
    *Note:* This downloads a VM/Container image (~1GB).

2.  **Check Status:**
    ```bash
    kubectl cluster-info
    ```
    *Result:* "Kubernetes control plane is running at..."

3.  **View Nodes:**
    ```bash
    kubectl get nodes
    ```
    *Result:* `minikube   Ready   control-plane`.

### Part 3: The First Command ☝️

1.  **Run Nginx:**
    ```bash
    kubectl run my-nginx --image=nginx
    ```

2.  **Check Pods:**
    ```bash
    kubectl get pods
    ```
    *Result:* `my-nginx   1/1   Running`.

---

## 🎯 Challenges

### Challenge 1: Namespaces (Difficulty: ⭐⭐)

**Task:**
Kubernetes supports multi-tenancy via Namespaces.
1.  Create a namespace `dev`.
2.  Run a pod inside `dev`.
3.  List pods in `default` (should be empty/different) vs `dev`.
    *Hint: `kubectl get pods -n dev`.*

### Challenge 2: K9s (Difficulty: ⭐)

**Task:**
Install **k9s**. It is a terminal-based UI for Kubernetes.
Run `k9s`. Navigate your cluster using arrow keys. It changes your life.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
kubectl create namespace dev
kubectl run dev-pod --image=alpine --namespace=dev -- sleep 1000
kubectl get pods -n dev
```
</details>

---

## 🔑 Key Takeaways

1.  **Declarative**: Like Terraform, K8s is declarative. You tell it "I want 3 pods", and it makes it happen.
2.  **API First**: Everything is an API call. `kubectl` is just a wrapper around curl.
3.  **Local vs Prod**: Minikube is single-node. Production clusters (EKS/GKE) have multiple nodes for High Availability.

---

## ⏭️ Next Steps

We ran a single Pod. But Pods die. We need something to manage them.

Proceed to **Lab 12.2: Pods & Deployments**.
