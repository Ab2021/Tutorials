# Days 50-56: Week 8 - K8s Scheduling & GPUs Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 50: Scheduler Deep Dive

```yaml
# Node affinity example
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu-type
            operator: In
            values: ["a100", "h100"]
```

---

## Day 51: NVIDIA Device Plugin

```bash
# Install device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPUs
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu'
```

```yaml
# GPU pod
spec:
  containers:
  - name: cuda
    image: nvidia/cuda:12.0-base
    resources:
      limits:
        nvidia.com/gpu: 1
```

---

## Day 52: Multi-Instance GPU (MIG)

```bash
# Configure MIG (on node)
nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C

# View MIG devices
nvidia-smi mig -lgi
```

---

## Day 53: GPU Topology

```bash
# Check topology
nvidia-smi topo -m

# NVLink verification
nvidia-smi nvlink -s
```

---

## Day 54: GPU Sharing

```yaml
# Time-slicing config
kind: ConfigMap
metadata:
  name: time-slicing-config
data:
  default: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
```

---

## Day 55: DCGM Monitoring

```bash
# Deploy DCGM exporter
helm install dcgm-exporter nvidia/dcgm-exporter

# Verify metrics
kubectl port-forward svc/dcgm-exporter 9400:9400
curl localhost:9400/metrics | grep DCGM
```

---

## Day 56: Week 8 Project

```yaml
# GPU-enabled training job
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
spec:
  template:
    spec:
      containers:
      - name: train
        image: pytorch/pytorch:2.0-cuda12.0
        command: ["python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 4
      restartPolicy: Never
```

---

## 📝 Week 8 Summary
| Day | Topic | Key Concept |
|-----|-------|-------------|
| 50 | Scheduler | Affinity, Taints |
| 51 | Device Plugin | nvidia.com/gpu |
| 52 | MIG | GPU partitioning |
| 53 | Topology | NVLink, NVSwitch |
| 54 | Sharing | Time-slicing |
| 55 | DCGM | GPU metrics |
| 56 | Project | GPU cluster |
