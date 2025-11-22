# Lab 12.2: Pods & Deployments

## 🎯 Objective

Understand the atomic unit of Kubernetes (The Pod) and how to manage them at scale using **Deployments**. You will learn why you almost *never* create a Pod directly in production.

## 📋 Prerequisites

-   Completed Lab 12.1.
-   Minikube running.

## 📚 Background

### Pod
-   The smallest deployable unit.
-   Usually 1 container per Pod (sometimes sidecars).
-   **Ephemeral**: If it dies, it stays dead.

### Deployment
-   Manages a **ReplicaSet**.
-   Ensures **X** number of pods are always running.
-   Handles **Rolling Updates** (Zero Downtime).
-   **Self-Healing**: If a Pod dies, Deployment creates a new one.

---

## 🔨 Hands-On Implementation

### Part 1: The Manifest (YAML) 📄

We stop using CLI commands and start using YAML files.

1.  **Create `nginx-pod.yaml`:**
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: nginx-manual
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f nginx-pod.yaml
    ```

3.  **Delete:**
    ```bash
    kubectl delete -f nginx-pod.yaml
    ```

### Part 2: The Deployment 🚀

1.  **Create `nginx-deployment.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: nginx-deployment
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: nginx
      template:
        metadata:
          labels:
            app: nginx
        spec:
          containers:
          - name: nginx
            image: nginx:1.14.2
            ports:
            - containerPort: 80
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f nginx-deployment.yaml
    ```

3.  **Verify:**
    ```bash
    kubectl get deployments
    kubectl get pods
    ```
    *Result:* You see 3 pods named `nginx-deployment-xxxxx`.

### Part 3: Self-Healing 🩹

1.  **Kill a Pod:**
    Copy a pod name.
    ```bash
    kubectl delete pod nginx-deployment-xxxxx
    ```

2.  **Watch Magic:**
    ```bash
    kubectl get pods -w
    ```
    *Result:* The old pod Terminates. A **NEW** pod starts immediately. The Deployment ensures 3 replicas exist.

### Part 4: Scaling 📈

1.  **Scale Up:**
    Change `replicas: 3` to `replicas: 10` in the YAML.
    ```bash
    kubectl apply -f nginx-deployment.yaml
    ```
    *Result:* 7 new pods start.

---

## 🎯 Challenges

### Challenge 1: Rolling Update (Difficulty: ⭐⭐⭐)

**Task:**
1.  Change the image version in YAML from `nginx:1.14.2` to `nginx:latest`.
2.  Apply.
3.  Watch `kubectl get pods -w`.
    *Observation:* It updates pods one by one (or in batches). The service never goes down.

### Challenge 2: Rollback (Difficulty: ⭐⭐)

**Task:**
Oops, `nginx:latest` is broken (hypothetically).
Undo the deployment.
*Hint: `kubectl rollout undo ...`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```bash
kubectl rollout undo deployment/nginx-deployment
```
</details>

---

## 🔑 Key Takeaways

1.  **Pods are Mortal**: Don't get attached to them.
2.  **Deployments are Forever**: Use Deployments for stateless apps.
3.  **Labels & Selectors**: The Deployment finds its pods using `matchLabels`. This is the glue of Kubernetes.

---

## ⏭️ Next Steps

We have apps running, but we can't access them.

Proceed to **Lab 12.3: Services & Networking**.
