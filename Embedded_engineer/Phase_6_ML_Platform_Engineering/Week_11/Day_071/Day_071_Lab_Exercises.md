# Days 71-77: Week 11 - Container Optimization Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 71: Docker for ML

```dockerfile
# Multi-stage build
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM nvidia/cuda:12.0-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "serve.py"]
```

---

## Day 72: Registry Best Practices

```bash
# Scan image
trivy image myorg/ml-model:v1.0

# Sign image
cosign sign --key cosign.key myorg/ml-model:v1.0
```

---

## Day 73: NGC Catalog

```bash
# Pull NGC container
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Run with GPU
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
```

---

## Day 74: Security

```dockerfile
# Distroless example
FROM gcr.io/distroless/python3-debian11
COPY --from=builder /app /app
COPY --from=builder /root/.local /root/.local
USER nonroot
CMD ["python", "/app/serve.py"]
```

---

## Day 75: Container Networking

```yaml
# Host networking for RDMA
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
```

---

## Day 76: Resource Limits

```yaml
# Proper resource settings
resources:
  requests:
    cpu: 4
    memory: 16Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 8
    memory: 32Gi
    nvidia.com/gpu: 1
```

---

## Day 77: Week 11 Project

```dockerfile
# Optimized ML container
# Start: 5GB -> Target: 1GB
# Techniques:
# - Multi-stage build
# - Slim base image
# - .dockerignore
# - Layer caching
```

---

## 📝 Week 11 Summary
| Day | Topic | Focus |
|-----|-------|-------|
| 71 | Docker | Multi-stage |
| 72 | Registry | Security |
| 73 | NGC | NVIDIA containers |
| 74 | Hardening | Rootless |
| 75 | Networking | RDMA |
| 76 | Resources | Limits |
| 77 | Project | Optimization |
