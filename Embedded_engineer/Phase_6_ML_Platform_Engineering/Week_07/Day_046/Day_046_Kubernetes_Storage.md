# Day 46: State in a Stateless World: K8s Storage
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** Your Model Weights and Datasets need a home that survives Pod restarts. Master **Persistent Volumes (PV)** and **Persistent Volume Claims (PVC)** to manage stateful data in Kubernetes.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the relationship between StorageClass, PV, and PVC.
2.  **Differentiate** between Ephemeral (`emptyDir`) and Persistent (`hostPath`, `ebs`) storage.
3.  **Deploy** a Postgres database with a Persistent Volume.
4.  **Choose** the right Access Mode: ReadWriteOnce (RWO) vs ReadWriteMany (RWX).
5.  **Simulate** a node failure and verify data retention.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.
- *Note:* Minikube usually provides a "standard" StorageClass that maps to the host disk.

### Software Environment
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Persistence Ladder

1.  **Container FS:** Dies when container restarts.
2.  **emptyDir Volume:** Dies when Pod is deleted from Node. (Good for cache).
3.  **Persistent Volume (PV):** Independent lifecycle. Lives on AWS EBS / NFS / Host Disk.

### 2. The Claim System (PVC)

Developers don't want to know about AWS EBS IDs.
*   **Admin** sets up a **Storage Class** (e.g., "fast-ssd").
*   **Developer** creates a **PVC**: "Give me 10Gi from fast-ssd".
*   **Controller** automatically provisions a PV (Dynamic Provisioning) and binds it to the PVC.

### 3. Access Modes
*   **RWO (ReadWriteOnce):** Only ONE Node can mount this. (Typical for Databases, Block Storage).
*   **RWX (ReadWriteMany):** Many Nodes can mount read/write. (Typical for Shared Datasets, NFS/EFS). **Crucial for Distributed Training**.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Postgres with PVC

We need a database (e.g., for MLflow).

#### 📁 `manifests/postgres-storage.yaml`
```yaml
# 1. The Claim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  # storageClassName: standard # Minikube's default
---
# 2. The Pod (Using the Claim)
apiVersion: v1
kind: Pod
metadata:
  name: ml-db
  labels:
    app: postgres
spec:
  containers:
  - name: postgres
    image: postgres:15
    env:
    - name: POSTGRES_PASSWORD
      value: "mysecretpassword"
    ports:
    - containerPort: 5432
    volumeMounts:
    - mountPath: /var/lib/postgresql/data
      name: pg-data
  volumes:
  - name: pg-data
    persistentVolumeClaim:
      claimName: postgres-pvc
```

### 👨‍💻 Lab: Data Persistence Test

1.  **Deploy:**
    ```bash
    kubectl apply -f manifests/postgres-storage.yaml
    ```
2.  **Write Data:**
    ```bash
    # Wait for running
    kubectl exec -it ml-db -- psql -U postgres -c "CREATE TABLE experiments (id int, name text);"
    kubectl exec -it ml-db -- psql -U postgres -c "INSERT INTO experiments VALUES (1, 'ResNet Training');"
    ```
3.  **Delete Pod (Simulate Crash):**
    ```bash
    kubectl delete pod ml-db
    ```
    *Note: The PVC is NOT deleted. It persists.*
4.  **Recreate Pod:**
    ```bash
    kubectl apply -f manifests/postgres-storage.yaml
    ```
5.  **Verify Data:**
    ```bash
    kubectl exec -it ml-db -- psql -U postgres -c "SELECT * FROM experiments;"
    # Output should show 'ResNet Training'.
    ```

### 👨‍💻 Concept: Shared Datasets (RWX)

For Multi-Node Training, all pods need access to ImageNet. You cannot use RWO (EBS) because EBS cannot attach to 2 nodes. You need **NFS**.

```yaml
kind: PersistentVolumeClaim
metadata:
  name: dataset-pvc
spec:
  accessModes:
    - ReadWriteMany # RWX
  storageClassName: efs-sc # AWS EFS
  resources:
    requests:
      storage: 1Ti
```

---

## 🔬 Lab Exercise: "The HostPath Trap"

### Task
In Minikube, PVCs use `hostPath` (directory on your laptop).
1.  Find where the data lives.
    ```bash
    kubectl get pv
    # Look for the PV bound to postgres-pvc
    kubectl describe pv <pv-name>
    ```
2.  You will see `Path: /tmp/hostpath-provisioner/...`.
3.  **Warning:** `hostPath` is dangerous in multi-node production. If the Pod reschedules to Node B, but data is on Node A's disk, the DB starts empty! **Always use Network Storage (EBS/NFS) in Prod.**

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decoupling:** PVCs decouple "I need storage" from "How storage is provided".
2.  **Dynamic Provisioning:** Magic. You ask for coverage, K8s calls the Cloud API and creates a disk in seconds.
3.  **ML Context:**
    *   **Datasets:** Use RWX (NFS/EFS) PVCs mounted at `/data` for training jobs.
    *   **Checkpoints:** Use RWO or Object Storage (S3).
    *   **Databases:** Use RWO.

### API Summary
```bash
kubectl get pvc
kubectl get pv
kubectl get sc # storage classes
```

---

**Day 46 Complete** ✅

*Next: Day 47 - Configuration & Secrets - Managing passwords and hyperparameters without hardcoding them.*
