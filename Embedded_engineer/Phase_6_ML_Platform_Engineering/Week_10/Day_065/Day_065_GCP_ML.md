# Day 65: The Home of Kubernetes & TPUs: GCP
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 10: Cloud Platforms for ML

---

> **🎯 Focus Area:** Google Cloud Platform (GCP). Learn to leverage **GKE**, the gold standard of Managed Kubernetes, and **TPUs (Tensor Processing Units)**, Google's custom ASIC for Deep Learning.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** NVIDIA GPU architecture vs Google TPU architecture.
2.  **Deploy** a GKE Cluster with GPU and TPU Node Pools.
3.  **Request** TPU resources in Kubernetes manifests (`google.com/tpu`).
4.  **Explain** the TPU VM architecture (v3 vs v4) and "Pod Slices".
5.  **Use** GCS (Google Cloud Storage) with FUSE for data loading.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GCP Account.
- *Note:* TPUs are quota-restricted. We will analyze the manifests conceptually.

### Software Environment
- `gcloud` CLI.
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The TPU Advantage
A GPU is a general purpose graphics chip used for AI. A TPU is an **ASIC** (Application Specific Integrated Circuit) built *only* for Matrix Multiplication (MXU).
*   **Architecture:** Systolic Array. Data flows through the chip like a pulse, reducing memory access.
*   **Networking:** TPU Chips have dedicated high-speed Interconnects (ICI) to form massive "Pods" (e.g., v4 Pod = 4096 chips) without using Ethernet switch.

### 2. GKE Standard vs Autopilot
*   **Standard:** You define Node Pools (`n1-standard-4` with `T4`). You install drivers (mostly automated).
*   **Autopilot:** You create a Pod asking for `cpu: 4, gpu: 1`. GCP spawns a node invisibly. Good for Ops-lite teams.

### 3. TPU in Kubernetes
Unlike GPUs (Time Slicing), TPUs are usually assigned as **Slices**.
*   `v3-8`: A single board with 8 Cores (4 Chips).
*   `v4-32`: A slice of a Pod.
*   Your Pod requests `google.com/tpu: 8`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: GKE Cluster Creation

```bash
# 1. Create Cluster
gcloud container clusters create ml-cluster \
    --region us-central1 \
    --num-nodes 1

# 2. Add GPU Node Pool
gcloud container node-pools create gpu-pool \
    --cluster ml-cluster \
    --region us-central1 \
    --machine-type n1-standard-8 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes 1

# 3. Add TPU Node Pool (Available in specific zones)
gcloud container node-pools create tpu-pool \
    --cluster ml-cluster \
    --region us-central1 \
    --machine-type ct4p-hightpu-4t \
    --num-nodes 1
```

### 👨‍💻 Core Implementation: The TPU Pod

Running a JAX/TensorFlow workload on TPU.

#### 📁 `manifests/tpu-job.yaml`
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: tpu-training
spec:
  template:
    spec:
      containers:
      - name: jax-code
        image: python:3.9
        command: ["python", "train.py"]
        resources:
          limits:
            google.com/tpu: 8 # Request 8 TPU v3 Cores
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v3
        cloud.google.com/gke-tpu-topology: 2x2
      restartPolicy: Never
```

### 👨‍💻 Storage: GCS Fuse

Access buckets as file systems.

```yaml
spec:
  containers:
  - name: app
    volumeMounts:
    - name: data-bucket
      mountPath: /data
  volumes:
  - name: data-bucket
    csi:
      driver: gcsfuse.csi.storage.gke.io
      volumeAttributes:
        bucketName: my-dataset-bucket
        mountOptions: "implicit-dirs"
```

---

## 🔬 Lab Exercise: "XLA Compatibility"

### Task
Why can't I just run my PyTorch CUDA code on TPU?
1.  **CUDA Kernels:** Written in C++ for NVIDIA GPUs. Do NOT run on TPUs.
2.  **XLA (Accelerated Linear Algebra):** The Compiler (Day 30).
3.  **Migration:**
    *   **PyTorch:** Use `torch_xla` library.
    *   **JAX:** Native support (Preferred).
    *   **TensorFlow:** Native support.

### Insight
GCP is excellent if you use high-level frameworks (JAX, TF). If you write custom CUDA kernels (Triton, Cutlass), you are stuck with NVIDIA GPUs (which GCP also offers, but AWS is often preferred for H100s).

---

## 📝 Daily Summary

### Key Takeaways
1.  **GKE is Smooth:** GKE is widely considered the most stable and feature-rich Managed K8s. Upgrades and Scaling are smoother than EKS.
2.  **TPU Pricing:** Usually cheaper performanced-per-dollar than GPUs for large batch training, but requires code changes (`torch_xla`).
3.  **Spot TPUs:** Preemptible TPUs offer massive savings, but losing a TPU Pod slice can kill a whole distributed run unless checkopinting is excellent.

### API Summary
```bash
gcloud container clusters get-credentials name
```

---

**Day 65 Complete** ✅

*Next: Day 66 - Azure for ML - AKS and how Microsoft handles AI.*
