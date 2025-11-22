# Lab 13.1: GitOps with ArgoCD

## 🎯 Objective

Stop running `kubectl apply`. In GitOps, the **Git Repository** is the source of truth. An agent (ArgoCD) inside the cluster pulls changes from Git and applies them. You will install ArgoCD and sync a sample app.

## 📋 Prerequisites

-   Minikube running.
-   GitHub Account.

## 📚 Background

### Push vs Pull
-   **Push (Jenkins/GitHub Actions)**: CI server runs `kubectl apply`. Requires giving CI server admin access to K8s. (Security Risk).
-   **Pull (ArgoCD/Flux)**: Agent inside K8s watches Git. Pulls changes. No external access needed.

### Drift Detection
If someone manually deletes a deployment, ArgoCD sees the "Drift" (Live state != Git state) and heals it automatically.

---

## 🔨 Hands-On Implementation

### Part 1: Install ArgoCD 🐙

1.  **Create Namespace:**
    ```bash
    kubectl create namespace argocd
    ```

2.  **Install Manifests:**
    ```bash
    kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
    ```

3.  **Access UI:**
    -   Wait for pods to run: `kubectl get pods -n argocd`.
    -   Port Forward:
        ```bash
        kubectl port-forward svc/argocd-server -n argocd 8080:443
        ```
    -   Open `https://localhost:8080` (Accept security warning).

4.  **Login:**
    -   User: `admin`.
    -   Password (Get from Secret):
        ```bash
        kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
        ```

### Part 2: The Git Repo 📂

1.  **Fork a Repo:**
    Use `https://github.com/argoproj/argocd-example-apps`.
    Or create your own repo with a `guestbook` folder containing YAMLs.

### Part 3: Create Application 🚀

1.  **In ArgoCD UI:**
    -   Click **New App**.
    -   **Application Name**: `guestbook`.
    -   **Project**: `default`.
    -   **Sync Policy**: `Manual` (for now).
    -   **Repository URL**: `https://github.com/argoproj/argocd-example-apps.git` (or your fork).
    -   **Path**: `guestbook`.
    -   **Cluster URL**: `https://kubernetes.default.svc`.
    -   **Namespace**: `default`.
    -   Click **Create**.

2.  **Sync:**
    -   Status is `OutOfSync` (Yellow).
    -   Click **Sync** -> **Synchronize**.
    -   Status becomes `Synced` (Green).
    -   Check `kubectl get pods`. The app is running!

### Part 4: Self-Healing 🩹

1.  **Break it:**
    Manually delete the deployment.
    ```bash
    kubectl delete deployment guestbook-ui
    ```

2.  **Observe ArgoCD:**
    -   Status goes `OutOfSync` (or `Missing`).
    -   If you enabled "Self Heal", it fixes it instantly.
    -   If not, click **Sync** again.

---

## 🎯 Challenges

### Challenge 1: Auto-Sync (Difficulty: ⭐⭐)

**Task:**
Enable **Automated Sync** and **Self-Heal** in the App Details.
Make a change in the Git Repo (e.g., change replica count).
Commit and Push.
Watch ArgoCD update the cluster automatically (might take 3 mins).

### Challenge 2: App of Apps (Difficulty: ⭐⭐⭐⭐)

**Task:**
Research the "App of Apps" pattern.
Create one ArgoCD Application that points to a folder containing *other* Application manifests.
*Goal:* Bootstrap an entire cluster with 50 microservices using one click.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
App Details -> Sync Policy -> Enable Automated -> Check Prune and Self Heal.

**Challenge 2:**
Create a YAML file `apps.yaml` that defines `kind: Application`. Point ArgoCD to that file.
</details>

---

## 🔑 Key Takeaways

1.  **Git is Truth**: If it's not in Git, it doesn't exist.
2.  **No more kubectl**: Developers don't need kubectl access. They just make Pull Requests.
3.  **Audit Trail**: Git history shows exactly who changed what and when.

---

## ⏭️ Next Steps

Managing raw YAMLs is tedious. Let's use a package manager.

Proceed to **Lab 13.2: Helm Charts**.
