# Day 51: The Bridge: NVIDIA Device Plugin
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** Kubernetes doesn't know what a GPU is. Learn how the **NVIDIA Device Plugin** bridges the gap, allowing you to schedule GPU workloads securely.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Kubernetes Device Plugin Framework.
2.  **Deploy** the NVIDIA Device Plugin DaemonSet locally or on a cluster.
3.  **Request** GPU resources in a Pod manifest (`nvidia.com/gpu`).
4.  **Understand** how the plugin handles GPU isolation (Environment Variables & Volume Mounts).
5.  **Debug** "0/N nodes available: Insufficient nvidia.com/gpu" errors.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- A machine with an NVIDIA GPU and Linux Drivers installed.
- Docker + NVIDIA Container Toolkit configured.
- *Simulation:* Minikube can pass-through GPUs with `--gpus=all`, or we can study the manifests conceptually.

### Software Environment
- `helm`.

---

## 📖 Theoretical Foundation

### 1. The Default Blindness
Kubelet tracks CPU (millis) and RAM (bytes). It has no concept of "PCIe Device".
If you mount `/dev/nvidia0` manually into containers, you risk two containers fighting over the same GPU (Memory corruption).

### 2. How the Plugin Works
The **NVIDIA Device Plugin**:
1.  Runs as a **DaemonSet** on every GPU Node.
2.  Queries NVML (NVIDIA Management Library) to count devices.
3.  Registers with Kubelet via gRPC.
4.  Updates Node Capacity: `nvidia.com/gpu: 8`.

### 3. Allocation and Scheduler
When you request `nvidia.com/gpu: 1`:
1.  Scheduler finds a node with free GPUs.
2.  Kubelet calls the Plugin: "Allocate 1 GPU for this container."
3.  Plugin returns the list of devices (e.g., `UUID-ABC...`).
4.  Kubelet applies `NVIDIA_VISIBLE_DEVICES=UUID-ABC` env var to the container.
5.  NVIDIA Container Runtime isolates the container to *only* see that GPU.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Installing the Plugin

Use Helm (Industry Standard).

```bash
# 1. Add Repo
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

# 2. Install
# Fails if no GPUs detected or no 'nvidia' runtime class.
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --set gfd.enabled=true # Enable GPU Feature Discovery (Labeling)
```

**What does GFD (GPU Feature Discovery) do?**
It automatically labels nodes with hardware details:
*   `nvidia.com/gpu.product=Tesla-T4`
*   `nvidia.com/gpu.memory=15360`
This allows you to write: `nodeSelector: nvidia.com/gpu.memory: "15360"`.

### 👨‍💻 Core Implementation: Requesting a GPU

#### 📁 `manifests/gpu-vector-add.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:11.7.1-base-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        # The Magic Key
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
```

*Note: You CANNOT request 0.5 GPUs with the default plugin. It is an integer resource.* (See Day 54 for sharing).

---

## 🔬 Lab Exercise: "Verification"

### Task
Verify the plugin is working (assuming GPU node availability).
1.  Check Node Capacity:
    ```bash
    kubectl describe node <node-name>
    ```
    Look for:
    ```text
    Capacity:
      cpu: 4
      memory: 16Gi
      nvidia.com/gpu: 1  <-- Success
    ```
2.  Deploy `gpu-vector-add.yaml`.
3.  Check logs:
    ```bash
    kubectl logs gpu-pod
    ```
    You should see the standard `nvidia-smi` table.

### Debugging Common Issues
*   **FailedScheduling:** usually means no node has free GPUs. Check `kubectl describe node`.
*   **ImagePullBackOff:** `nvidia/cuda` images are large.
*   **RuntimeError:** If `nvidia-smi` inside the pod says "Failed to initialize NVML", the Host Driver might be mismatched with the Container CUDA version, or the NVIDIA Runtime Hook is missing from Docker config.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Privileged Operation:** The Device Plugin needs privileged access to `/var/lib/kubelet/device-plugins` to register itself.
2.  **Integer Only:** By default, 1 GPU is the smallest unit. If you have a V100 and a tiny inference job, you waste the GPU.
3.  **Labels:** GFD is essential for heterogeneous clusters (mixing T4s and A100s). Without it, your A100 training job might land on a T4.

### API Summary
```bash
helm install nvidia-device-plugin
kubectl get nodes -L nvidia.com/gpu.product
```

---

**Day 51 Complete** ✅

*Next: Day 52 - Multi-Instance GPU (MIG) - Partitioning one big A100 into 7 smaller instances.*
