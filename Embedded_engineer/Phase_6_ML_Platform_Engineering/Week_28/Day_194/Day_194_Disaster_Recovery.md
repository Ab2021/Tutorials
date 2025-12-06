# Day 194: The Lifeboat: Disaster Recovery with Velero
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 28: Platform Engineering Practices

---

> **🎯 Focus Area:** "The datacenter just flooded." Or worse, "Someone ran `terraform destroy` by accident." **Disaster Recovery (DR)** ensures you can restore the entire Platform state (Kubernetes Objects + Persistent Volumes) to a new region in minutes.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Recovery Point Objective (RPO) and Recovery Time Objective (RTO).
2.  **Deploy** Velero with the AWS Plugin to backup cluster resources to S3.
3.  **Execute** a Full Cluster Restore from an existing backup.
4.  **Schedule** periodic backups for Critical Namespaces (e.g., `ml-registry`).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (Kind Cluster).

### Software Environment
- `velero` CLI.
- AWS S3 Bucket.

---

## 📖 Theoretical Foundation

### 1. RPO vs RTO
*   **RPO (Point):** How much data can you lose? (e.g., "Last 24 hours").
*   **RTO (Time):** How fast must you come back online? (e.g., "Within 4 hours").
*   **Velero:** Allows Low RPO (Hourly backups) and Low RTO (Fast restore).

### 2. Velero Architecture
*   **BackupController:** Watches for `Backup` CRDs. Uploads K8s YAMLs to S3.
*   **VolumeSnapshotter:** Takes EBS Snapshots (for PVCs).
*   **Restic/Kopia:** Takes Filesystem-level backups (for non-EBS volumes like HostPath or local disk).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Install Velero

Connect to S3.

#### 📁 `manifests/velero-install.sh`
```bash
# 1. Create S3 Bucket & IAM User (Terraform usually handles this)
BUCKET="my-cluster-backups"
REGION="us-east-1"

# 2. Install Velero
velero install \
    --provider aws \
    --plugins velero/velero-plugin-for-aws:v1.6.0 \
    --bucket $BUCKET \
    --backup-location-config region=$REGION \
    --snapshot-location-config region=$REGION \
    --secret-file ./credentials-velero
```

### 👨‍💻 Core Implementation: On-Demand Backup

Save the namespace before a risky upgrade.

```bash
# Backup 'ml-jobs' namespace including PVCs
velero backup create pre-upgrade-backup \
    --include-namespaces ml-jobs \
    --wait

# Verify
velero backup get
# NAME                 STATUS      COMPLETED
# pre-upgrade-backup   Completed   2023-10-27 10:00:00
```

### 👨‍💻 Core Implementation: Scheduled Backup

Daily insurance policy.

#### 📁 `manifests/schedule.yaml`
```yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-full-cluster
  namespace: velero
spec:
  schedule: "0 1 * * *" # 1 AM Daily
  template:
    includedNamespaces:
      - "*" # All namespaces
    ttl: 720h # Keep for 30 days
```

### 👨‍💻 Core Implementation: The Restore

The moment of truth.

```bash
# 1. Simulate Disaster
kubectl delete ns ml-jobs

# 2. Restore
velero restore create --from-backup pre-upgrade-backup

# 3. Verify
kubectl get pods -n ml-jobs
# They should reappear. PVCs will rebind to EBS snapshots.
```

---

## 🔬 Lab Exercise: "The Migration"

### Task
Move Cluster A (US-East) to Cluster B (US-West).
1.  **Backup:** Run `velero backup create migration-backup` on Cluster A.
2.  **Config:** Point Cluster B's Velero to the *same* S3 bucket.
3.  **Restore:** Run `velero restore create --from-backup migration-backup` on Cluster B.
4.  **Result:** All deployments, services, and configs appear in Cluster B.
5.  **Note:** EBS Volumes (PVCs) cannot migrate regions easily. You must use Restic/Kopia (File-level copy) for cross-region data migration, or use replication at the storage layer.

---

## 📖 Advanced Theory: Etcd Backup
Velero backs up *Kubernetes Objects*. It does not backup the Etcd database file directly.
If the Control Plane completely fails (split brain), Velero cannot run.
**Etcd Snapshot:**
*   `etcdctl snapshot save snapshot.db`
*   This is the "Nuclear Option" for restoring a corrupted Control Plane.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Test Restores:** A backup is broken until you successfully restore it. Schedule "Game Days" where you attempt to restore Prod into a Staging environment.
2.  **Stateless is Easy:** If your apps are GitOps managed (ArgoCD), you don't really need Velero for Deployments (Just sync Git).
3.  **State is Hard:** Velero is crucial for PVCs (Database data, Model Checkpoints). Focus your backup strategy on State.

### API Summary
```bash
velero backup logs <backup-name>
velero restore describe <restore-name>
```

---

**Day 194 Complete** ✅

*Next: Day 195 - Secret Management at Scale - Vault.*
