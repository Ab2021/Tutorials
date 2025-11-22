# Lab 28.1: Kubecost (Cost Visibility)

## 🎯 Objective

Show me the money. Kubernetes is a black box for finance. "Why is the bill $5,000?" **Kubecost** breaks down costs by Namespace, Deployment, and Label.

## 📋 Prerequisites

-   Minikube running.
-   Helm installed.

## 📚 Background

### Metrics
-   **Allocation**: How much CPU/RAM you *requested*.
-   **Usage**: How much you actually *used*.
-   **Efficiency**: Usage / Allocation. (Low efficiency = Wasted money).

---

## 🔨 Hands-On Implementation

### Part 1: Install Kubecost 💸

1.  **Install:**
    ```bash
    helm repo add kubecost https://kubecost.github.io/cost-analyzer/
    helm upgrade --install kubecost kubecost/cost-analyzer \
        --namespace kubecost --create-namespace \
        --set kubecostToken="ZXhhbXBsZUBrdWJlY29zdC5jb20=xm343yadf98"
    ```
    *(Note: The token is a public demo token).*

2.  **Access UI:**
    ```bash
    kubectl port-forward --namespace kubecost deployment/kubecost-cost-analyzer 9090
    ```
    Open `http://localhost:9090`.

### Part 2: Analyze Costs 📊

1.  **Overview:**
    See the "Monthly Run Rate". On Minikube, it estimates based on standard cloud prices (e.g., $0.04/core-hour).

2.  **Breakdown:**
    Go to **Monitor** -> **Allocation**.
    Group by: **Namespace**.
    *Result:* See how much `kube-system` costs vs `default`.

3.  **Efficiency:**
    Go to **Savings** -> **Request Sizing**.
    It tells you: "Your Nginx deployment requests 1 CPU but uses 0.001 CPU. Resize it to save $20/month."

### Part 3: Showback 🧾

1.  **Labeling:**
    Deploy a pod with label `owner=team-a`.
    Wait 15 mins.
    Group by `Label: owner`.
    *Result:* You can bill Team A exactly for what they use.

---

## 🎯 Challenges

### Challenge 1: Alerting (Difficulty: ⭐⭐⭐)

**Task:**
Set up a budget alert.
"If daily spend > $10, send Slack message."
(Requires configuring Helm values with Slack webhook).

### Challenge 2: Spot Readiness (Difficulty: ⭐⭐)

**Task:**
Use the "Spot Readiness" checklist in Kubecost.
It checks if your pods are "Spot Ready" (have PDBs, multiple replicas, controller managed).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
In `values.yaml`:
```yaml
notifications:
  slack:
    enabled: true
    webhook: https://hooks.slack.com/...
```
</details>

---

## 🔑 Key Takeaways

1.  **Visibility**: You can't optimize what you can't measure.
2.  **Requests vs Limits**: You pay for Requests (what you reserve), not Limits.
3.  **Idle Cost**: Kubecost shows "Idle" cost (resources on nodes that no pod is using). This means your nodes are too big or you need Autoscaling.

---

## ⏭️ Next Steps

We see the waste. Let's fix it with Autoscaling.

Proceed to **Lab 28.2: Karpenter**.
