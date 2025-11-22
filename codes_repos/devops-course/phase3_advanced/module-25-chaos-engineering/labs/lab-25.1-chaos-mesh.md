# Lab 25.1: Chaos Mesh (Pod Kill)

## 🎯 Objective

Break things on purpose. **Chaos Engineering** is the discipline of experimenting on a system to build confidence in its capability to withstand turbulent conditions. You will use **Chaos Mesh** to kill pods randomly and see if your app survives.

## 📋 Prerequisites

-   Minikube running.
-   Helm installed.

## 📚 Background

### Experiments
-   **PodChaos**: Kill Pod, Pod Failure (Unready).
-   **NetworkChaos**: Delay, Loss, Corrupt.
-   **StressChaos**: CPU/Memory burn.

---

## 🔨 Hands-On Implementation

### Part 1: Install Chaos Mesh 🕸️

1.  **Install:**
    ```bash
    helm repo add chaos-mesh https://charts.chaos-mesh.org
    helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh --create-namespace --set dashboard.securityMode=false
    ```

2.  **Access Dashboard:**
    `kubectl port-forward svc/chaos-dashboard -n chaos-mesh 2333:2333`.
    Open `http://localhost:2333`.

### Part 2: Deploy Victim App 🎯

1.  **Deploy Nginx:**
    ```bash
    kubectl create deployment nginx --image=nginx --replicas=3
    ```

### Part 3: The Experiment (Pod Kill) 💀

1.  **Create `pod-kill.yaml`:**
    ```yaml
    apiVersion: chaos-mesh.org/v1alpha1
    kind: PodChaos
    metadata:
      name: pod-kill-example
      namespace: default
    spec:
      action: pod-kill
      mode: one
      selector:
        labelSelectors:
          app: nginx
      scheduler:
        cron: "@every 1m"
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f pod-kill.yaml
    ```

3.  **Observe:**
    Watch pods: `kubectl get pods -w`.
    Every minute, one pod will Terminate and restart.
    *Result:* Since we have 3 replicas, the service stays up. **Success!**

### Part 4: Network Latency 🐢

1.  **Create `network-delay.yaml`:**
    ```yaml
    apiVersion: chaos-mesh.org/v1alpha1
    kind: NetworkChaos
    metadata:
      name: network-delay
    spec:
      action: delay
      mode: all
      selector:
        labelSelectors:
          app: nginx
      delay:
        latency: "200ms"
      duration: "30s"
    ```

2.  **Apply:**
    `kubectl apply -f network-delay.yaml`.

3.  **Test:**
    Exec into a pod and ping another.
    `ping <pod-ip>`.
    *Result:* Latency jumps to 200ms+.

---

## 🎯 Challenges

### Challenge 1: Blast Radius (Difficulty: ⭐⭐)

**Task:**
Modify the experiment to only target pods in a specific namespace `dev`.
*Goal:* Ensure you don't accidentally kill Production.

### Challenge 2: Kernel Chaos (Difficulty: ⭐⭐⭐⭐)

**Task:**
(Requires Linux Node). Use **IOChaos** to simulate disk failure.
Make `read` operations fail.
See how your Database handles it.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Set `namespace: dev` in the metadata and selector.
</details>

---

## 🔑 Key Takeaways

1.  **Hypothesis**: "If I kill a pod, the user sees no error."
2.  **Experiment**: Run PodChaos.
3.  **Verification**: Check 500 error rate.

---

## ⏭️ Next Steps

Chaos Mesh is great. Let's try another tool.

Proceed to **Lab 25.2: LitmusChaos**.
