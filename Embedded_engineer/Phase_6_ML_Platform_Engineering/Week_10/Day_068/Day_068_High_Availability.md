# Day 68: Keeping the Lights On: High Availability
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 10: Cloud Platforms for ML

---

> **🎯 Focus Area:** Cloud Regions fail. Availability Zones (AZs) go offline. Learn to design **Resilient** Architectures using **Topology Spread Constraints** and **Zone-Aware Storage**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Availability Zone (AZ) and Region boundaries.
2.  **Implement** `topologySpreadConstraints` to force Pods into different AZs.
3.  **Analyze** the impact of Zonal failures on EBS/Persistent Volumes.
4.  **Design** a StatefulSet architecture that survives a single AZ loss.
5.  **Evaluate** Trade-offs: Cross-Zone Latency ($$) vs High Availability.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Multi-Zone Cluster (e.g., EKS/GKE in `us-east-1`).

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Zoning Laws
*   **Region:** Geography (Virginia). 10ms-100ms latency to other regions.
*   **AZ:** Data Center (Building A vs Building B). <1ms latency to other AZs in same region.
*   **Best Practice:** Spread stateless apps across 3 AZs. If one AZ burns, you lose 33% capacity, but app stays up.

### 2. The Storage Trap
Block Storage (AWS EBS, Azure Disk, GCE PD) acts like a physical hard drive. It exists in **ONE AZ**.
*   If Pod A writes to `vol-1` in `us-east-1a`...
*   And `us-east-1a` dies...
*   K8s moves Pod A to `us-east-1b`.
*   **FAILURE:** Pod cannot attach `vol-1`. Data is trapped in the dead AZ.

### 3. Solutions
*   **Stateless:** Easy. Just spread them.
*   **Stateful (DB):** Use Application Replication (Postgres Master in 1a, Replica in 1b).
*   **Stateful (Files):** Use EFS/Filestore (Regional Storage).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Topology Spread

Force 3 replicas to sit in 3 different zones (if available).

#### 📁 `manifests/ha-inference.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ha-model
spec:
  replicas: 6
  selector:
    matchLabels:
      app: ha-model
  template:
    metadata:
      labels:
        app: ha-model
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: ha-model
      containers:
      - name: nginx
        image: nginx
```

**Scenario:**
*   Nodes in Zone A, B, C.
*   Replicas: 6.
*   Distribution: A=2, B=2, C=2.
*   If Zone C dies, K8s detects 4 healthy pods. It tries to reschedule the 2 dead ones to A and B.

### 👨‍💻 Core Implementation: Zone-Aware Storage (GCP Example)

Use Regional Persistent Disk (Replicated synchronously to 2 zones).

```yaml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: regional-pd
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-balanced
  replication-type: regional-pd # <--- Magic Key
  availability-class: regional-hard-failover
volumeBindingMode: WaitForFirstConsumer
allowedTopologies:
- matchLabelExpressions:
  - key: topology.gke.io/zone
    values:
    - us-central1-a
    - us-central1-b
```

---

## 🔬 Lab Exercise: "Kill the Zone"

### Task
Simulate AZ failure.
1.  Deploy `ha-model` (6 replicas).
2.  Cordon ALL nodes in `us-east-1a`:
    ```bash
    kubectl cordon -l topology.kubernetes.io/zone=us-east-1a
    ```
3.  Delete the pods in that zone.
4.  **Observation:**
    *   Pods go Pending? (If you used `DoNotSchedule` and other zones are full).
    *   Pods move to `1b`? (If capacity exists).
    *   **Traffic:** Service Load Balancer automatically stops sending traffic to the 1a nodes.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Cross-Zone Costs:** In AWS/Azure, traffic *between* AZs costs money ($0.01/GB). Massive synchronization (All-Reduce) for Distributed Training should usually stay in **ONE** AZ (Placement Group) to avoid cost and latency.
    *   **Inference:** Multi-AZ (Reliability).
    *   **Training:** Single-AZ (Performance).
2.  **PDB (Pod Disruption Budget):** Use `minAvailable: 2` to ensure that during voluntary disruptions (Cluster Upgrade), K8s never kills too many replicas at once.

### API Summary
```bash
kubectl get nodes -L topology.kubernetes.io/zone
```

---

**Day 68 Complete** ✅

*Next: Day 69 - Hybrid & Multi-Cloud - Managing clusters on AWS and On-Prem simultaneously.*
