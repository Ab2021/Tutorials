# Day 66: The Enterprise Cloud: ML on Azure
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 10: Cloud Platforms for ML

---

> **🎯 Focus Area:** Microsoft Azure. Learn how **AKS (Azure Kubernetes Service)** and **Azure ML** power some of the world's largest AI workloads (including OpenAI's infrastructure).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** Azure VM sizes (`NCsv3` vs `NDmv4`) for specific ML tasks.
2.  **Deploy** an AKS cluster with GPU-enabled node pools.
3.  **Configure** Blob Storage connectivity using the Blob CSI Driver.
4.  **Integrate** Azure Active Directory (Entra ID) for K8s RBAC.
5.  **Explain** the relationship between Azure ML (YAML pipelines) and AKS.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Azure Subscription.

### Software Environment
- `az` CLI.
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Instance Families
*   **NC-Series (Compute):** Older generations (K80, V100, T4). Good for Inference & small training.
*   **ND-Series (Deep Learning):** The beast. `Standard_ND96amsr_A100_v4` contains 8x A100 GPUs with InfiniBand. This is what trained GPT-4.

### 2. Networking (InfiniBand)
Azure is unique in exposing raw InfiniBand to VMs (using PKEYs). You need specific NCCL plugins to use it effectively.
Unlike AWS EFA (which is custom Ethernet), Azure uses standard IB.

### 3. Identity (Entra ID / AAD)
Identity is Azure's superpower.
*   **Pod Identity:** Assign a Managed Identity to a Pod.
*   **Workload Identity:** The modern replacement (OIDC based).
*   **RBAC:** Map `Azure AD Group: DataScientists` to `K8s Role: View` seamlessly.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: AKS Cluster

```bash
# 1. Resource Group
az group create --name ml-rg --location eastus

# 2. Create Cluster (System Node Pool)
az aks create \
    --resource-group ml-rg \
    --name ml-cluster \
    --node-count 1 \
    --enable-managed-identity \
    --generate-ssh-keys

# 3. Add GPU Node Pool
az aks nodepool add \
    --resource-group ml-rg \
    --cluster-name ml-cluster \
    --name gpulinux \
    --node-count 1 \
    --node-vm-size Standard_NC6s_v3 \
    --labels accelerator=nvidia-v100
```

### 👨‍💻 Core Implementation: Blob Storage CSI

Mounting a container (bucket) as a file system.

#### 📁 `manifests/blob-storage.yaml`
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-weights-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: azureblob-fuse-premium
---
apiVersion: v1
kind: Pod
metadata:
  name: blob-check
spec:
  containers:
  - name: inspector
    image: busybox
    command: ["ls", "-l", "/mnt/blob"]
    volumeMounts:
    - name: blob
      mountPath: /mnt/blob
  volumes:
  - name: blob
    persistentVolumeClaim:
      claimName: model-weights-pvc
```

---

## 🔬 Lab Exercise: "Spot Eviction Simulation"

### Task
Azure Spot VMs offer massive discounts but can be evicted.
1.  Add a node pool with `--priority Spot --eviction-policy Delete`.
2.  Deploy a Training Job.
3.  Simulate eviction:
    ```bash
    az aks nodepool scale --node-count 0 ...
    ```
4.  **Observation:** Pods die.
5.  **Mitigation:** Use `pytorch-checkpoint` or `KubeRay` restart capabilities. Always configure `tolerations` for `kubernetes.azure.com/scalesetpriority: spot`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **InfiniBand:** If you are doing Multi-Node, Azure is often preferred due to native InfiniBand support in the ND series.
2.  **BlobFuse:** Accessing Blob storage as files is convenient but check `blobfuse2` caching settings. Random seeks (like reading HDF5) can be slow without local SSD caching.
3.  **Hybrid:** Azure Arc allows you to manage On-Prem GPU clusters from the Azure Portal.

### API Summary
```bash
az aks get-credentials --resource-group rg --name cluster
```

---

**Day 66 Complete** ✅

*Next: Day 67 - Cost Optimization Strategies - Saving millions with Spot, Autoscaling, and Right-Sizing.*
