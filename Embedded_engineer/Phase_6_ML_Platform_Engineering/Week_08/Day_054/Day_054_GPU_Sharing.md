# Day 54: Sharing is Caring: Time-Slicing & MPS
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 8: Kubernetes Scheduling & GPUs

---

> **🎯 Focus Area:** Not everyone has an A100 for MIG. For older GPUs (T4, V100, RTX 3090), we use **Time-Slicing** to share one physical GPU among multiple Pods.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Time-Slicing (Software) vs MIG (Hardware).
2.  **Configure** the Time-Slicing strategy in the NVIDIA Device Plugin.
3.  **Deploy** multiple pods sharing a single GPU.
4.  **Explain** the risks of shared VRAM (OOM Propagation).
5.  **Describe** MPS (Multi-Process Service) and when to use it over standard Time-Slicing.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Any NVIDIA GPU (Pascal or newer).

### Software Environment
- Helm (for Device Plugin updates).

---

## 📖 Theoretical Foundation

### 1. The Context Switch
By default, CUDA is exclusive. If Process A launched a Context on GPU 0, Process B blocks.
**Time-Slicing** (enabled in driver/plugin) allows the GPU Scheduler to switch contexts rapidly.
*   **Analogy:** CPU Multitasking. It looks parallel, but it's interleaved execution.
*   **Trade-off:** Minimal memory protection. If Pod A allocates 100% VRAM, Pod B OOMs immediately.

### 2. Time-Slicing vs MIG
| Feature | Time-Slicing | MIG (Day 52) |
| :--- | :--- | :--- |
| **Hardware Support** | All (Pascal+) | Ampere/Hopper+ |
| **Isolation** | None (Shared VRAM) | Strong (Dedicated VRAM) |
| **Performance** | Context Switch Overhead | Hardware Parallelism |
| **Max Replicas** | Arbitrary (User defined) | Fixed (7 per A100) |

### 3. MPS (Multi-Process Service)
An advanced mode for Volta/Turing/Ampere.
*   **Concept:** A "Server" process holds the GPU context. Client processes send work to the Server.
*   **Benefit:** Allows *concurrent* kernel execution (Spatial Sharing) rather than just interleaved (Temporal Sharing).
*   **K8s Limitation:** Harder to configure securely per-pod. Often used in HPC, less in Cloud Native K8s.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Configuring Time-Slicing

We update the `nvidia-device-plugin` configuration using a ConfigMap.

#### 📁 `manifests/time-slicing-config.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: nvidia-device-plugin
data:
  any: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # Advertise 1 Physical GPU as 4 Virtual GPUs
```

#### Apply Update via Helm

```bash
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin \
  --set config.name=time-slicing-config \
  --set config.default=any
```

#### Verification
```bash
kubectl describe node <node-name>
# Capacity:
#   nvidia.com/gpu: 4  (If you had 1 physical card)
#   nvidia.com/gpu: 8  (If you had 2 physical cards)
```

Now, scheduler sees 4 slots.

### 👨‍💻 Lab: Oversubscription

1.  Create `consumer.yaml` that requests `nvidia.com/gpu: 1`.
2.  Deploy 4 replicas.
    ```bash
    kubectl create deployment gpu-test --image=cuda-vector-add --replicas=4
    ```
3.  **Observation:** All 4 run.
4.  Run `nvidia-smi` inside one pod.
    *   It sees the *Full* GPU Name (e.g., "Tesla T4").
    *   It sees the *Full* VRAM (16GB).
    *   **Danger:** If it tries to `cudaMalloc(16GB)`, it might succeed if others are idle, or crash if others are using memory.

---

## 🔬 Lab Exercise: "The Noisy Neighbor"

### Task
Demonstrate lack of isolation.
1.  Launch Pod A running a heavy matrix multiplication loop.
2.  Launch Pod B running the same.
3.  Check execution time.
4.  **Result:** Both run 2x slower (sharing compute).
5.  Now make Pod A allocate 90% VRAM.
6.  Start Pod B.
7.  **Result:** Pod B crashes (CUDA OOM), or Pod A crashes (if B steals memory).

### Insight
Time-Slicing is excellent for **Dev/Test** clusters or **Low-Duty Inference** (Chatbots that reply once per minute). It is dangerous for Production SLAs unless you control the workloads strictly.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Virtualization:** Time-Slicing is "Poor Man's Virtualization". It tricks K8s into scheduling more pods than GPUs.
2.  **Configuration:** Done entirely in various ConfigMaps passed to the Device Plugin. No host reboot required usually (unlike MIG).
3.  **Recommendation:** Use MIG for A100s. Use Time-Slicing for T4s/L4s/A10s when running many small inference servers.

### API Summary
```yaml
sharing:
  timeSlicing:
    resources:
    - name: nvidia.com/gpu
      replicas: 10
```

---

**Day 54 Complete** ✅

*Next: Day 55 - DCGM & GPU Monitoring - Prometheus, Grafana, and seeing how hot your GPU is.*
