# Day 77: Week 11 Review & Project - The Golden Image Factory
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** Combine Multi-Stage builds, Security Scanning, and Registry Management to build a **Golden Image Pipeline** that produces secure, minimal containers for your experimental teams.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Dockerfile that balances size, build speed, and security.
2.  **Automate** image scanning and signing in a CI pipeline (mocked).
3.  **Deploy** a locked-down container that passes a "Trivy Gate".

---

## 📚 Week 11 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 71 | Internals | "A deleted file in Layer 3 is still present in Layer 2." |
| 72 | Minimization | "Distroless has no shell. Hackers can't `ls`." |
| 73 | GPU Opt | "The `-devel` image is for compiling, `-runtime` is for running." |
| 74 | Security | "My image had 500 vulnerabilities. Upgrading the base fixed 450." |
| 75 | BuildKit | "Cache Mounts made my pip install instant." |
| 76 | Harbor | "I can proxy Docker Hub and never hit rate limits again." |

---

## 🏗️ Final Project: "SecureTorch"

### Goal
Create a reusable base image `securetorch:2.0` that:
1.  Has PyTorch 2.0 + CUDA 11.8.
2.  Is <3GB (uncompressed).
3.  Has 0 Critical CVEs.
4.  Has no `nvcc` compiler.

### Step 1: The Dockerfile

#### 📁 `project/Dockerfile`
```dockerfile
# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Dependency Resolver (Heavy)
# ==========================================
FROM python:3.9-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use Cache Mount for Pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir /app/wheels \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Remove bloat from wheels (Strip debugging symbols if possible)
# (Advanced: Unzip wheel, strip .so, re-zip. Skipped for brevity).

# ==========================================
# Stage 2: Runtime (Light)
# ==========================================
FROM python:3.9-slim

# Security: Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Runtime basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Wheels
COPY --from=builder /app/wheels /wheels

# Install
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache /wheels/* \
    && rm -rf /wheels

# Final Cleanup
RUN find /usr/local/lib/python3.9 -name "__pycache__" -delete

USER appuser

# Self-Check
RUN python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"

CMD ["python3"]
```

### Step 2: The Build Script (CI Mock)

#### 📁 `project/build_pipeline.sh`
```bash
#!/bin/bash
set -e

IMAGE="localhost:5000/securetorch:2.0"

echo "1. Building..."
docker build -t $IMAGE .

echo "2. Scanning..."
# Fail if Critical found
trivy image --exit-code 1 --severity CRITICAL $IMAGE

echo "3. Pushing..."
# (Assuming local registry or Harbor)
docker push $IMAGE

echo "Success! Image is safe and pushed."
```

### Step 3: Validation

1.  Run the script.
2.  **Size Check:** `docker images | grep securetorch`.
    *   Compare with `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` + pip install.
    *   Should be smaller due to lack of Apt cache and stripped OS layers.
3.  **Security Check:** Trivy report should be clean (or actionable).
4.  **Runtime Check:** `docker run --gpus all securetorch:2.0 python -c "import torch; print(torch.cuda.is_available())"`

---

## 📝 Success Criteria
1.  **Size:** Image is smaller than the naive approach.
2.  **Security:** Runs as non-root (`uid=1000`).
3.  **Speed:** Subsequent builds use the Pip Cache.

---

**Week 11 Complete** ✅

*Next Week: Week 12 - Networking for Distributed ML - RDMA, InfiniBand, and making nodes talk fast.*
