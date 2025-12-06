# Day 73: The 5GB Bloat: Optimizing GPU Containers
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** ML Containers often exceed 10GB. Understand the NVIDIA Container anatomy (**Base, Runtime, Devel**) and how to strip unused CUDA libraries.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** the correct NVIDIA Base Image (`runtime` vs `devel`) for the task.
2.  **Explain** why `pip install torch` results in massive images (Bundled CUDA).
3.  **Implement** a Multi-Stage Build that compiles CUDA Extensions (`nvcc`) in stage 1 and runs in stage 2.
4.  **Remove** static libraries (`.a`) and unused architectures from the image.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Docker.

### Software Environment
- `docker`.

---

## 📖 Theoretical Foundation

### 1. NVIDIA Docker Tags
*   **base (100MB):** Includes `libcudart` (CUDA Runtime API). Useful if you have a statically linked app.
*   **runtime (1GB):** Includes `libcublas`, `libcufft`, `libcurand`. **Standard for Deep Learning** (PyTorch needs these shared objects).
*   **devel (4GB):** Includes `nvcc` compiler, headers, static libs. **Use only for building**.

### 2. The PyTorch Problem
When you do `pip install torch`, the Wheel (Binary) bundles its *own copy* of CUDA and cuDNN in `site-packages/torch/lib`.
*   **Repo Size:** The `nvidia/cuda` image has CUDA. The `torch` wheel has CUDA. You have **Double CUDA**.
*   **Solution:** No easy fix for `pip`. But you can strip unused parts.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Dynamic Linking Strategy

If we build custom CUDA Extensions (e.g., FlashAttention), we need `nvcc`.

#### 📁 `gpu-optimized.Dockerfile`
```dockerfile
# Stage 1: Build (Devel Image - 4GB)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install PyTorch
RUN pip3 install --no-cache-dir torch

# Install Custom Extension (Needs nvcc)
RUN pip3 install --no-cache-dir git+https://github.com/HazyResearch/flash-attention.git

# Stage 2: Runtime (Runtime Image - 1GB)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
# Note: This copies the huge torch folder too.
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

CMD ["python3", "-c", "import flash_attn; print('Success')"]
```

### 👨‍💻 Size Reduction Tricks

To remove bloat from `site-packages/torch`:

```dockerfile
# In the builder stage, after pip install:
RUN find /usr/local/lib/python3.10/dist-packages/torch -name "*.a" -delete
RUN find /usr/local/lib/python3.10/dist-packages/torch -name "*.so" | xargs strip --strip-unneeded
# Remove tests
RUN rm -rf /usr/local/lib/python3.10/dist-packages/torch/test
```

---

## 🔬 Lab Exercise: "The Double CUDA Audit"

### Task
1.  Build an image `FROM python:3.9` and `RUN pip install torch`. Size: ~2GB.
2.  Build an image `FROM nvidia/cuda:11.8-runtime` and `RUN pip install torch`. Size: ~3.5GB.
3.  **Observation:** You have the OS CUDA (in `/usr/local/cuda`) and the Pip CUDA (in `/usr/local/lib/python/site-packages/torch/lib`).
4.  **Mitigation:** Only deploy the `python:3.9` image if you rely on pip wheels. Only deploy `nvidia/cuda` if you compile things from source against system CUDA.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Devel is for Building:** Never ship a `-devel` image to production. It contains compilers that attackers love.
2.  **Layer Caching:** PyTorch changes infrequently. Your code changes often. Keep `pip install torch` high up in the Dockerfile.
3.  **Lazy Loading:** Container Registries and Runtimes (stargz-snapshotter) are exploring "Lazy Pulling" where you can run the container before downloading the full 10GB.

### API Summary
```bash
docker history <image>
```

---

**Day 73 Complete** ✅

*Next: Day 74 - Security Scanning - Trivy and Grype.*
