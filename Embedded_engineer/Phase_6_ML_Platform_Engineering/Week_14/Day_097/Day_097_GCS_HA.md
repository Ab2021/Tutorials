# Day 97: The Brain: GCS High Availability
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 14: KubeRay & Production Ray

---

> **🎯 Focus Area:** If the Head Node dies, your 1000-node training job usually crashes. Learn to externalize the **Global Control Store (GCS)** to make the cluster resilient to Head Node failures.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** the Single Point of Failure (SPOF) in standard Ray Clusters.
2.  **Deploy** a High-Availability Redis cluster (using Helm).
3.  **Configure** a RayCluster to point to external Redis for GCS storage.
4.  **Simulate** a Head Node crash and verify Worker survival.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- K8s Cluster.

### Software Environment
- `helm`.

---

## 📖 Theoretical Foundation

### 1. What is GCS?
The **Global Control Store** is the database of Ray.
*   **Actor Directory:** Where is `Actor#123` living?
*   **Object Table:** Where is `Object#abc` stored?
*   **Node Table:** Who is alive?

### 2. Failure Modes
*   **Standard:** GCS is a process on Head Node. Head Node Pod dies -> GCS dies -> Workers panic and suicide.
*   **HA Mode:** GCS writes to persistent Redis. Head Node restarts -> Reads Redis -> Reconnects to existing Workers.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: External Redis

First, we need a robust Redis.

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis-ha bitnami/redis \
  --set architecture=replication \
  --namespace ray-system
```
Ref: `redis-ha-master.ray-system.svc.cluster.local:6379`.

### 👨‍💻 Core Implementation: HA Ray Cluster

#### 📁 `manifests/ha-raycluster.yaml`
```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ha-cluster
  annotations:
    ray.io/ft-enabled: "true" # Logic to enable GCS Fault Tolerance
spec:
  rayVersion: '2.9.0'
  headGroupSpec:
    rayStartParams:
      # Point to external Redis
      redis-address: "redis-ha-master.ray-system.svc.cluster.local:6379"
      # Secure password (if configured)
      # redis-password: ...
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0
          env:
            - name: RAY_REDIS_ADDRESS
              value: "redis-ha-master.ray-system.svc.cluster.local:6379"

  workerGroupSpecs:
  - replicas: 2
    groupName: worker-group
    # ... standard worker ...
```

### 👨‍💻 Lab Operations: The Kill Switch

1.  Deploy Cluster.
2.  Start a long running actor (e.g., a Loop printing "Alive").
3.  **Kill Head:**
    ```bash
    kubectl delete pod -l ray.io/node-type=head
    ```
4.  **Observation:**
    *   K8s creates new Head Pod.
    *   New Head connects to Redis.
    *   Finds existing Workers.
    *   The Actor on the worker *keeps running* (maybe a small hiccup).

---

## 🔬 Lab Exercise: "Resource Leaks"

### Task
If GCS is persistent, what about dead clusters?
1.  If you delete the `RayCluster` (K8s pods), but Redis keeps the data...
2.  Start a new cluster pointing to same Redis.
3.  It might try to connect to dead workers from the previous run.
4.  **Solution:** Ray supports Namespacing in Redis, or you must flush Redis/use ephemeral Redis for each cluster.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Complexity:** HA adds complexity (managing Redis). Only use it if your Head Node stability is a proven bottleneck.
2.  **Dashboard:** The Dashboard also runs on the Head. During a Head restart, the UI is inaccessible, but compute continues.
3.  **GCS Server:** In Ray 2.0+, GCS is its own server process. Redis is mostly used for sharding. The architecture is moving towards "No Redis Required" for HA in future versions (using Etcd or SQL). Check release notes!

### API Summary
```bash
ray start --redis-address=...
```

---

**Day 97 Complete** ✅

*Next: Day 98 - Week 14 Review & Project - Building a Production Ray Platform.*
