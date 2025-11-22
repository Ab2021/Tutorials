# Lab 28.2: Karpenter (Just-in-Time Scaling)

## 🎯 Objective

The right node, at the right time. The standard **Cluster Autoscaler** is slow and rigid (Auto Scaling Groups). **Karpenter** (by AWS) bypasses ASGs and launches EC2 instances directly. It picks the *cheapest* instance that fits your pods.

## 📋 Prerequisites

-   AWS Account (This lab requires real cloud permissions).
-   EKS Cluster (or ability to create one).
-   *Simulation Mode:* If you don't have AWS, read through to understand the concepts.

## 📚 Background

### Cluster Autoscaler vs Karpenter
-   **CA**: "I need a node. Scale up ASG size +1." (Wait for ASG, wait for boot).
-   **Karpenter**: "I need 5 CPUs. Launch a `c5.large` immediately." (Faster, Cheaper).

---

## 🔨 Hands-On Implementation

### Part 1: Install Karpenter 🪚

1.  **Helm Install:**
    (Requires IAM Roles for Service Accounts - IRSA).
    ```bash
    helm install karpenter oci://public.ecr.aws/karpenter/karpenter ...
    ```

### Part 2: Provisioner (The Policy) 📜

1.  **Create `provisioner.yaml`:**
    ```yaml
    apiVersion: karpenter.sh/v1alpha5
    kind: Provisioner
    metadata:
      name: default
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
      limits:
        resources:
          cpu: 1000
      providerRef:
        name: default
      ttlSecondsAfterEmpty: 30
    ```
    *Translation:* "Launch Spot or On-Demand instances. Kill them 30s after they become empty."

### Part 3: Scale Up 📈

1.  **Deploy Inflate App:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: inflate
    spec:
      replicas: 0
      template:
        spec:
          containers:
          - name: inflate
            image: public.ecr.aws/eks-distro/kubernetes/pause:3.2
            resources:
              requests:
                cpu: 1
    ```

2.  **Scale:**
    ```bash
    kubectl scale deployment inflate --replicas=5
    ```

3.  **Observe:**
    `kubectl get pods -w`.
    Karpenter sees 5 pending pods (5 CPUs needed).
    It calls EC2 API: "Give me the cheapest instance with >5 CPUs".
    It launches an `m5.large` (2 vCPU) and a `c5.xlarge` (4 vCPU) or whatever combination is cheapest.
    Pods bind in seconds.

### Part 4: Consolidation (Scale Down) 📉

1.  **Scale Down:**
    ```bash
    kubectl scale deployment inflate --replicas=1
    ```

2.  **Observe:**
    Karpenter sees nodes are underutilized.
    It moves the remaining pod to a smaller node (or existing node) and terminates the expensive nodes.

---

## 🎯 Challenges

### Challenge 1: Spot Only (Difficulty: ⭐⭐)

**Task:**
Modify Provisioner to ONLY use Spot instances.
`values: ["spot"]`.
*Risk:* Spot interruptions. Ensure your app handles graceful termination.

### Challenge 2: ARM64 (Difficulty: ⭐⭐⭐)

**Task:**
Allow ARM64 instances (Graviton).
`values: ["amd64", "arm64"]`.
Deploy a multi-arch image.
*Benefit:* Graviton is 20% cheaper.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
requirements:
  - key: karpenter.sh/capacity-type
    operator: In
    values: ["spot"]
```
</details>

---

## 🔑 Key Takeaways

1.  **Bin Packing**: Karpenter packs pods tightly to minimize wasted space.
2.  **Spot Instances**: Can save up to 90%. Karpenter handles the complexity of picking the pool with the lowest price/interruption.
3.  **Speed**: Karpenter binds pods to nodes *before* the node is ready (Node Binding), speeding up scheduling.

---

## ⏭️ Next Steps

We saved money. Now let's handle the fire.

Proceed to **Module 29: Incident Management**.
