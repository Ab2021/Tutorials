# Lab 22.2: Argo Rollouts (Canary & Blue-Green)

## 🎯 Objective

Native Kubernetes Deployments are boring (Rolling Update). **Argo Rollouts** is a CRD that adds advanced deployment strategies like **Blue-Green** and **Canary** with automated analysis.

## 📋 Prerequisites

-   Minikube running.
-   Argo Rollouts Controller installed.
-   `kubectl-argo-rollouts` plugin installed.

## 📚 Background

### Strategies
-   **Blue-Green**: Spin up new version (Green) alongside old (Blue). Switch traffic instantly. Easy rollback. Expensive (double resources).
-   **Canary**: Slowly shift traffic (5% -> 20% -> 50%). Analyze metrics. If error rate spikes, rollback automatically.

---

## 🔨 Hands-On Implementation

### Part 1: Install Argo Rollouts 🥐

1.  **Install Controller:**
    ```bash
    kubectl create namespace argo-rollouts
    kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
    ```

2.  **Install Plugin:**
    (Follow official docs for your OS, or use `brew install argoproj/tap/kubectl-argo-rollouts`).

### Part 2: Blue-Green Deployment 🔵🟢

1.  **Create `rollout-bg.yaml`:**
    ```yaml
    apiVersion: argoproj.io/v1alpha1
    kind: Rollout
    metadata:
      name: rollout-bluegreen
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: rollout-bluegreen
      strategy:
        blueGreen: 
          activeService: rollout-bluegreen-active
          previewService: rollout-bluegreen-preview
          autoPromotionEnabled: false
      template:
        metadata:
          labels:
            app: rollout-bluegreen
        spec:
          containers:
          - name: rollouts-demo
            image: argoproj/rollouts-demo:blue
            ports:
            - containerPort: 8080
    ```

2.  **Create Services:**
    You need two services: `rollout-bluegreen-active` and `rollout-bluegreen-preview`. (Standard Service YAMLs).

3.  **Apply & Update:**
    `kubectl apply -f ...`
    Update image to `yellow`.
    `kubectl argo rollouts get rollout rollout-bluegreen --watch`.
    *Result:* The `preview` service points to Yellow. `active` still points to Blue.
    
4.  **Promote:**
    ```bash
    kubectl argo rollouts promote rollout-bluegreen
    ```
    *Result:* Traffic switches to Yellow. Blue scales down.

### Part 3: Canary Deployment 🐤

1.  **Create `rollout-canary.yaml`:**
    ```yaml
    apiVersion: argoproj.io/v1alpha1
    kind: Rollout
    metadata:
      name: rollout-canary
    spec:
      replicas: 5
      strategy:
        canary:
          steps:
          - setWeight: 20
          - pause: {} # Wait for manual approval
          - setWeight: 40
          - pause: {duration: 10s}
          - setWeight: 60
          - pause: {duration: 10s}
      template: ... (same as above)
    ```

2.  **Apply & Update:**
    Update image to `yellow`.
    Watch it pause at 20%.
    Promote manually.
    Watch it proceed to 40%, wait 10s, then 60%, then 100%.

---

## 🎯 Challenges

### Challenge 1: Analysis Run (Difficulty: ⭐⭐⭐⭐)

**Task:**
Integrate with Prometheus.
Add an `analysis` step.
"If error rate > 1%, abort the rollout."
*Hint:* You need an `AnalysisTemplate` and a Prometheus server.

### Challenge 2: UI Dashboard (Difficulty: ⭐)

**Task:**
Run `kubectl argo rollouts dashboard`.
Open `http://localhost:3100`.
Visualize the steps and promote via the UI.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
strategy:
  canary:
    analysis:
      templates:
      - templateName: success-rate
      startingStep: 2
```
</details>

---

## 🔑 Key Takeaways

1.  **Safety**: Rollouts reduce the blast radius of bad code.
2.  **Integration**: Works perfectly with ArgoCD. ArgoCD syncs the Rollout object, and the Rollout Controller handles the pod creation.
3.  **No Service Mesh Needed**: Argo Rollouts can do basic Canary using standard K8s Services (by manipulating replica counts), though it's better with a Mesh/Ingress.

---

## ⏭️ Next Steps

We have deployed apps. Now let's deploy Functions.

Proceed to **Module 23: Serverless & Functions**.
