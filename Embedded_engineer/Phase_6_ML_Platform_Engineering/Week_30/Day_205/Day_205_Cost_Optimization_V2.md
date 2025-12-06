# Day 205: Pennies on the Dollar: Deep Cost Optimization
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 30: Capstone Project Part 2

---

> **🎯 Focus Area:** Your Platform works at scale (Day 204), but it costs $100k/month. The CFO is angry. Today we replace expensive On-Demand G5s with transient Spot Instances, saving 70-90% of compute costs, while handling interruptions gracefully.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** Mixed Instance Policies (e.g., "Use g4dn.2xlarge OR g5.xlarge") to increase Spot liquidity.
2.  **Deploy** the AWS Node Termination Handler (NTH) to drain nodes *before* they are reclaimed.
3.  **Implement** Karpenter to replace the slow Cluster Autoscaler (CAS) for sub-minute node provisioning.
4.  **Migrate** CPU workloads (Control Plane) to Fargate to eliminate idle node overhead.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-gpu`.

### Software Environment
- `helm install karpenter`.

---

## 📖 Theoretical Foundation

### 1. Spot Liquidity
*   **Problem:** If you ask for 1000 `p3.2xlarge` Spot instance in `us-east-1a`, AWS might say "No". Capacity is finite.
*   **Solution: Diversification.** Ask for:
    *   `us-east-1a`, `1b`, `1c` (Availability Zones).
    *   `p3.2xlarge`, `p3.8xlarge`, `g5.8xlarge` (Instance Families).
    *   **Result:** Probability of getting *some* GPU approaches 100%.

### 2. Karpenter vs CAS
*   **Cluster Autoscaler (CAS):** Watches Pending Pods -> Updates ASG Desired Count -> ASG launches node. Slow (3-5 mins). Ties to specific ASG types.
*   **Karpenter:** Watches Pending Pods -> Calls EC2 `RunInstances` API directly. Fast (30s). Can pick the *cheapest instance that fits the pod*.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Karpenter Provisioner

Define "GPU Spot Nodes".

#### 📁 `manifests/cost/karpenter-gpu.yaml`
```yaml
apiVersion: karpenter.sh/v1beta1
kind: Provisioner
metadata:
  name: gpu-spot
spec:
  # 1. Requirements
  requirements:
    - key: "karpenter.k8s.aws/instance-category"
      operator: In
      values: ["g", "p"] # GPU instances
    - key: "karpenter.k8s.aws/instance-generation"
      operator: Gt
      values: ["3"] # Gen 3 or newer
    - key: "karpenter.sh/capacity-type"
      operator: In
      values: ["spot"]
    - key: "kubernetes.io/arch"
      operator: In
      values: ["amd64"]
      
  # 2. Limits (Budget Cap)
  limits:
    resources:
      cpu: 1000
      nvidia.com/gpu: 100
      
  # 3. Consolidation (Defrag)
  consolidation:
    enabled: true # Kill empty nodes aggressively
  ttlSecondsUntilExpired: 2592000 # 30 Days (Force recycle)
```

### 👨‍💻 Core Implementation: Node Termination Handler

Graceful Shutdown.

```bash
helm repo add eks https://aws.github.io/eks-charts
helm install aws-node-termination-handler eks/aws-node-termination-handler \
    --namespace kube-system \
    --set enableSpotInterruptionDraining=true \
    --set enableRebalanceMonitoring=true
```
**Mechanism:**
1.  DaemonSet watches IMDS (Metadata Service) for "Spot Warning" (2 min notice).
2.  If warning received -> Taint Node `NoSchedule`.
3.  Drain Node (Evict Pods).
4.  K8s reschedules Pods to a stable node.

### 👨‍💻 Infrastructure: Fargate Profile (Serverless CPU)

Move the "boring" stuff off EC2.

#### 📁 `iac/live/fargate.tf`
```hcl
module "eks" {
  # ...
  fargate_profiles = {
    karpenter = {
      name = "karpenter"
      selectors = [{ namespace = "karpenter" }]
    }
    coredns = {
      name = "coredns"
      selectors = [{ namespace = "kube-system", labels = { "k8s-app" = "kube-dns" } }]
    }
  }
}
```

---

## 🔬 Lab Exercise: "The Reaper"

### Task
Survive a Spot Interruption.
1.  **Deploy:** Long-running Training Job (ResNet).
2.  **Simulate:** Use `fis` (AWS Fault Injection Simulator) or a simple script to Trigger Spot Interruption.
3.  **Observation:**
    *   Node Termination Handler logs: `Received Spot Interruption`.
    *   Node Status changes to `SchedulingDisabled`.
    *   Pod Status: `Terminating`.
    *   Ray Train Actor: Detects failure.
    *   Ray Head: Reschedules Actor to a new Node.
    *   **Result:** Training pauses for 2 minutes, then resumes. Checkpoint loaded.

---

## 📖 Advanced Theory: Price Capacity Optimized Strategy
When requesting Spot, you can set allocation strategy:
*   `lowest-price`: Give me the cheapest. (High interruption risk).
*   `capacity-optimized`: Give me the one with most spare capacity. (Low interruption risk, slightly higher price).
*   **Recommendation:** Use `price-capacity-optimized` (Balanced). Karpenter does this by default.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Bin Packing:** Karpenter excels at tetris. If you have a pod needing 0.5 GPU (Time Slicing), it puts 2 pods on 1 GPU. If you have 3 pods needing 1 GPU, it launches a `g4dn.12xlarge` (4 GPUs) because it's cheaper than 4 `g4dn.xlarge`.
2.  **Overprovisioning:** To beat the "Cold Start", deploy a "Pause Pod" (Priority -1) that reserves space. When a Real Pod (Priority 0) comes, the Pause Pod is evicted, freeing space instantly. Karpenter then provisions space for the Pause Pod in the background.
3.  **Tags:** Karpenter propagates tags from `Provisioner` to EC2. Ensure `CostCenter` tag is set, or Kubecost will mark it "Unallocated".

### API Summary
```bash
kubectl get provisioners
kubectl get machines # Karpenter's view of EC2
```

---

**Day 205 Complete** ✅

*Next: Day 206 - Chaos Engineering - Game Day.*
