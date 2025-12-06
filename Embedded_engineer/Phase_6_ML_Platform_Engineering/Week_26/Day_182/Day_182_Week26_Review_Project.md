# Day 182: Week 26 Review & Project - The Hybrid Cloud Platform
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 26: Multi-Cluster & Federation

---

> **🎯 Focus Area:** You've built islands of compute. Now build bridges. We will integrate **Karmada** (Control), **Istio** (Network), **Fluid** (Data), and **Thanos** (Vision) into a unified Hybrid Cloud ML Platform.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a "Burst to Cloud" strategy where sensitive data stays on-prem (`member-prem`) but public data/training bursts to AWS (`member-cloud`).
2.  **Deploy** a unified observability stack (Thanos) to monitor both clusters from a single Grafana.
3.  **Implement** Policy Enforcement (OPA/Kyverno) that prevents "Confidential" labeled datasets from landing on the Cloud cluster.

---

## 📚 Week 26 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 176 | Federation Arch | "Karmada lets me treat 10 clusters like 1 big computer." |
| 177 | Advanced Sched | "I can override the container image based on the region." |
| 178 | Istio Mesh Fed | "Services in US can call services in EU securely via mTLS." |
| 179 | Data Federation | "Fluid caches S3 buckets on local SSDs. Training is fast." |
| 180 | Thanos Obs | "Prometheus is ephemeral. Thanos is forever (S3)." |
| 181 | Multi-Cluster CD | "ArgoCD ApplicationSets generate apps for 50 clusters automatically." |

---

## 🏗️ Final Project: "CloudBurst"

### Scenario
**Hybrid Corp** has:
1.  **On-Prem:** 10x A100s (Fixed capacity, Low Latency, Secrets).
2.  **Cloud (AWS):** Infinite capacity (Spot Instances, High Latency).
**Goal:** Run experiments on-prem. If capacity full, spill over to Cloud automatically.

### Step 1: Karmada Control Plane

#### 📁 `project/karmada/propagation.yaml`
```yaml
apiVersion: policy.karmada.io/v1alpha1
kind: PropagationPolicy
metadata:
  name: ml-job-policy
spec:
  resourceSelectors:
    - apiVersion: batch/v1
      kind: Job
      labelSelector:
        matchLabels:
          type: training
  placement:
    clusterAffinity:
      clusterNames:
        - member-prem
        - member-cloud
    replicaScheduling:
      replicaDivisionPreference: Weighted
      replicaSchedulingType: Divided
      weightPreference:
        staticWeightList:
          - targetCluster:
              clusterNames: ["member-prem"]
            weight: 10 # Prefer Prem (Cheaper)
          - targetCluster:
              clusterNames: ["member-cloud"]
            weight: 1  # Spillover
```

### Step 2: Data Policy (Security)

Prevent Sensitive Data leakage.

#### 📁 `project/policy/block-cloud-data.yaml`
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: block-sensitive-on-cloud
spec:
  validationFailureAction: enforce
  rules:
    - name: check-cluster
      match:
        resources:
          kinds: [Pod]
      validate:
        message: "Sensitive data cannot verify on Cloud!"
        pattern:
          metadata:
            labels:
              data-classification: "sensitive"
        deny:
          conditions:
            all:
              - key: "{{ request.clusterName }}" # Injected by Karmada? 
                operator: Equals
                value: "member-cloud"
```
*Note: Validating cluster name inside a member cluster is tricky. Better approach: Use Karmada `ClusterOverridePolicy` to Inject a strictly `deny` nodeSelector if the payload is sensitive.*

### Step 3: Fluid Data Caching

#### 📁 `project/data/dataset.yaml`
```yaml
apiVersion: data.fluid.io/v1alpha1
kind: Dataset
metadata:
  name: public-imagenet
  namespace: ml-jobs
spec:
  mounts:
    - mountPoint: s3://public/imagenet
      name: imagenet
---
# Federated Runtime (Karmada Propagates this to BOTH clusters)
apiVersion: data.fluid.io/v1alpha1
kind: AlluxioRuntime
metadata:
  name: public-imagenet
  namespace: ml-jobs
spec:
  replicas: 2
```

### Step 4: The Training Job (Submit to Karmada)

#### 📁 `project/jobs/train.yaml`
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: resnet-train
  namespace: ml-jobs
  labels:
    type: training
spec:
  parallelism: 5 # 5 Workers
  template:
    spec:
      containers:
      - name: train
        image: my-registry/resnet:v1
        volumeMounts:
        - mountPoint: /data
          name: data-vol
      volumes:
      - name: data-vol
        persistentVolumeClaim:
          claimName: public-imagenet
```

---

## 🔬 Lab Exercise: "The Spillover"

### Task
Simulate Saturation.
1.  **Initial:** On-Prem has 5 GPU slots free.
2.  **Submit Job:** `parallelism: 5`.
    *   Karmada schedules all 5 to `member-prem`.
3.  **Submit Job 2:** `parallelism: 10`.
    *   Karmada sees On-Prem full (via Aggregated Metrics or Scheduling Failure).
    *   Karmada schedules spillover to `member-cloud`.
4.  **Observe:**
    *   Cloud Pod starts up (Wait for Spot Instance).
    *   Fluid Runtime on Cloud warms up cache from S3.
    *   Training starts.
5.  **Cost:** 5 Cheap Jobs + 10 Expensive Jobs.
6.  **Cleanup:** Job finishes. Cloud instances terminate (Cluster Autoscaler).

---

## 📝 Success Criteria
1.  **Unified Control:** `kubectl get jobs` on Karmada Host shows jobs running in both NY and London.
2.  **Data Locality:** Cloud pods read from Cloud S3 (Fast). Prem pods read from Prem Cache (Fast). No Cross-Atlantic heavy downloads.
3.  **Observability:** Grafana shows "Global GPU Utilization: 85%".

---

**Week 26 Complete** ✅
**Phase 6E In Progress**

*Next Phase: Advanced Troubleshooting - Sherlock Holmes Mode.*
