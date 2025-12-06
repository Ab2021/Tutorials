# Day 176: One Ring to Rule Them All: Federation Architecture
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 26: Multi-Cluster & Federation

---

> **🎯 Focus Area:** Your company has grown. You now have a cluster in `us-east-1` (Virginia) and `eu-central-1` (Frankfurt). Managing them separately is painful. **Kubernetes Federation** unifies them into a single "Super Cluster".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Hybrid Cloud architecture (On-Prem GPU Cluster + AWS Cloud).
2.  **Explain** the architectural differences between **KubeFed** (Legacy) and **Karmada** (Modern).
3.  **Deploy** a Control Plane Cluster and join Member Clusters.
4.  **Visualize** the flow of a Federated Deployment (Propagation Policy).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (Simulating 3 clusters using `kind`).

### Software Environment
- `kind`, `kubectl`, `helm`.

---

## 📖 Theoretical Foundation

### 1. Why Federation?
*   **Latency:** Serve EU users from EU cluster. serve US users from US cluster.
*   **High Availability:** If `us-east-1` burns down, `us-west-2` survives.
*   **Bursting:** Train on-prem (Cheap). Burst to Cloud (Expensive) when deadlines approach.
*   **Compliance:** "German Data must never leave Germany" (GDPR).

### 2. Architecture Patterns
*   **Centralized (Hub-and-Spoke):** One "Host" cluster manages N "Member" clusters.
    *   *Pro:* Simple. Single pane of glass.
    *   *Con:* Host is Single Point of Failure (SPOF).
*   **Peer-to-Peer:** Clusters verify each other. (Complex, rare).

### 3. Karmada
"Kubernetes Armada".
*   **API Server:** Compatible with K8s API. You `kubectl apply` to Karmada, not individual clusters.
*   **Scheduler:** Decides *which* cluster gets the deployment based on Policy.
*   **Controller Manager:** Syncs status back from members.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Setting up the Federation (Kind)

We need 3 clusters: `host`, `member1`, `member2`.

#### 📁 `scripts/setup_clusters.sh`
```bash
#!/bin/bash
# 1. Create Host
kind create cluster --name host --image kindest/node:v1.27.3

# 2. Create Members
kind create cluster --name member1 --image kindest/node:v1.27.3
kind create cluster --name member2 --image kindest/node:v1.27.3

# 3. Install Karmada CLI
curl -s https://raw.githubusercontent.com/karmada-io/karmada/master/hack/install-cli.sh | bash

# 4. Init Karmada on Host
# This installs the Control Plane components (etcd, apiserver, scheduler) on 'host'
kubectl config use-context kind-host
karmadactl init
```

### 👨‍💻 Infrastructure: Joining Members

Teaching the Host about the Members.

#### 📁 `scripts/join_members.sh`
```bash
# Join Member 1
karmadactl join member1 \
    --cluster-kubeconfig=$HOME/.kube/config \
    --cluster-context=kind-member1

# Join Member 2
karmadactl join member2 \
    --cluster-kubeconfig=$HOME/.kube/config \
    --cluster-context=kind-member2

# Verify
kubectl get clusters
# NAME      VERSION   MODE   READY   AGE
# member1   v1.27.3   Push   True    2m
# member2   v1.27.3   Push   True    1m
```

### 👨‍💻 Core Implementation: The Propagation Policy

The magic that splits the workload.

#### 📁 `manifests/deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-global
  labels:
    app: nginx
spec:
  replicas: 10 # Global count? Or per cluster? Depends on Policy.
  selector:
    matchLabels:
      app: nginx
  template:
    spec:
      containers:
      - image: nginx
        name: nginx
```

#### 📁 `manifests/policy-split.yaml`
```yaml
apiVersion: policy.karmada.io/v1alpha1
kind: PropagationPolicy
metadata:
  name: nginx-propagation
spec:
  resourceSelectors:
    - apiVersion: apps/v1
      kind: Deployment
      name: nginx-global
  placement:
    clusterAffinity:
      clusterNames:
        - member1
        - member2
    replicaScheduling:
      replicaDivisionPreference: Weighted
      replicaSchedulingType: Divided
      weightPreference:
        staticWeightList:
          - targetCluster:
              clusterNames: ["member1"]
            weight: 1
          - targetCluster:
              clusterNames: ["member2"]
            weight: 1
```
**Result:** 5 replicas on `member1`, 5 replicas on `member2`.

---

## 🔬 Lab Exercise: "The Failover"

### Task
Simulate Region Failure.
1.  Deploy Nginx (10 replicas split 5/5).
2.  **Action:** "Kill" Member 1. (Disable network or `kind delete cluster member1`).
3.  **Observation:** Karmada detects `member1` NotReady.
4.  **Reaction:** Karmada Scheduler reschedules the 5 missing replicas to `member2`.
5.  **Result:** `member2` now runs 10 replicas. Service Continuity maintained.

---

## 📖 Advanced Theory: Cluster Auto-Detection
How does Karmada know `member1` supports GPUs but `member2` doesn't?
Karmada syncs `Node` information (aggregated) to the Host.
You can write policies like:
```yaml
clusterAffinity:
  clusterSelectorTerms:
    - matchExpressions:
      - key: "gpu-available"
        operator: In
        values: ["true"]
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Complexity:** Federation adds a massively complex layer. Debugging "Why is my pod pending?" now involves checking Host Scheduler AND Member Scheduler. Use only if absolutely necessary.
2.  **Push vs Pull:** Karmada supports **Push** (Host edits Member API) and **Pull** (Member Agent watches Host). Push is simpler but requires Host -> Member network connectivity.
3.  **State:** Databases cannot be federated easily. You can propagate `StatefulSet` definitions, but syncing the *storage* (EBS Volumes) between `us-east-1` and `eu-central-1` is physics-limited (Speed of Light).

### API Summary
```bash
karmadactl init
karmadactl join <cluster>
```

---

**Day 176 Complete** ✅

*Next: Day 177 - Karmada in Depth - Advanced Scheduling.*
