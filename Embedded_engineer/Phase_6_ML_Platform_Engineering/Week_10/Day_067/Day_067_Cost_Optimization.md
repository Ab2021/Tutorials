# Day 67: FinOps: Cost Optimization Strategies
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 10: Cloud Platforms for ML

---

> **🎯 Focus Area:** Cloud GPUs are expensive. An idle 8x A100 node costs $30/hr ($20k/month). Learn to use **Autoscaling**, **Spot Instances**, and **Karpenter** to pay only for what you use.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** the Kubernetes Cluster Autoscaler (CA) to scale GPU nodes from 0 to N.
2.  **Implement** graceful termination for Spot Instance interruptions.
3.  **Contrast** Karpenter vs Cluster Autoscaler.
4.  **Right-size** requests to avoid "Bin Packing" waste (e.g., requesting 3 GPUs on a 4-GPU node leaves 1 stranded).
5.  **Set up** "Scale-to-Zero" for inference endpoints.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Cloud Provider (AWS/GCP/Azure).

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. Cluster Autoscaler (CA)
Traditional approach.
*   **Trigger:** Pod pending (Insufficient CPU/GPU).
*   **Action:** Increment `AutoScalingGroup.DesiredCapacity` +1.
*   **Latency:** Slow (3-5 minutes to boot config).
*   **Limitation:** Bound to specific Instance Types defined in the Node Group (e.g., "This group is all `p3.2xlarge`").

### 2. Karpenter (Just-in-Time)
AWS-native (but expandable).
*   **Trigger:** Pod pending.
*   **Action:** Analyzes Pod requests. API calls EC2 `RunInstances`.
*   **Flexibility:** Can choose `g4dn.xlarge` OR `g5.xlarge` based on price and availability zones.
*   **Latency:** Fast (1-2 minutes).

### 3. Spot Instances
Unused capacity sold at 60-90% discount.
*   **Catch:** Provider can reclaim it with 30s - 2min warning.
*   **Solution:** Checkpointing (save weights every epoch). Auto-resume logic.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Cluster Autoscaler

(Conceptual manifest for CA deployment).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
spec:
  # ...
  containers:
    - command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste # Choose node group that wastes least RAM/CPU
        - --nodes=0:10:gpu-node-group-1 # Min:Max:Name
```

### 👨‍💻 Core Implementation: Spot Graceful Termination

If a node is preempted, K8s sends `SIGTERM` to your pod. You have ~30s to save state.

#### 📁 `src/train_with_grace.py`
```python
import signal
import sys
import time

def save_checkpoint():
    print("Saving Checkpoint to S3...")
    time.sleep(2) # Simulate upload
    print("Saved.")

def handle_sigterm(signum, frame):
    print("Received SIGTERM! Preemption imminent.")
    save_checkpoint()
    sys.exit(0) # Exit cleanly

signal.signal(signal.SIGTERM, handle_sigterm)

print("Training started...")
while True:
    print("Epoch running...")
    time.sleep(10)
```

In Deployment YAML:
```yaml
spec:
  terminationGracePeriodSeconds: 120 # Give Python script time to handle SIGTERM
  containers:
  # ...
  lifecycle:
    preStop:
      exec:
        command: ["/bin/sh", "-c", "echo Pre-Stop Hook"]
```

---

## 🔬 Lab Exercise: "Bin Packing Tetris"

### Task
Visualize fragmentation.
*   **Scenario:** Node has 4 GPUs.
*   **Pod A:** Requests 1 GPU.
*   **Pod B:** Requests 1 GPUs.
*   **Pod C:** Requests 1 GPUs.
*   **Waste:** 1 GPU is idle.
*   **Pod D (Requests 2 GPUs):** Cannot fit. CA spawns New Node (4 GPUs).
*   **Total:** 8 GPUs paid for, 5 used. 37% Waste.
*   **Solution:** Use `p3.2xlarge` (1 GPU) node groups for small jobs, or force users to request full nodes.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Scale to Zero:** Ensure your Inference Node Groups have `minSize: 0`. During the night (low traffic), you should pay $0 for GPU compute.
2.  **Priority Expanders:** Configure CA to prefer Spot instances first, then fallback to On-Demand if Spot is unavailable.
3.  **Monitoring:** Monitor `kube_node_status_condition{condition="MemoryPressure"}` to detect if you are under-provisioning.

### API Summary
```bash
kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml
```

---

**Day 67 Complete** ✅

*Next: Day 68 - High-Availability - Zones, Regions, and Disaster Recovery.*
