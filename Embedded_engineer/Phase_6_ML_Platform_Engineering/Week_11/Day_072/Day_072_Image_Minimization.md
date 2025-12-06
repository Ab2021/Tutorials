# Day 72: The Diet: Distroless vs Alpine vs Slim
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** "Security by Minimalim". A container with no `bash` shell is hard to hack. Compare Base Images and learn why **Distroless** is the ultimate production target.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Evaluate** the trade-offs between Alpine (Size) and Debian Slim (Compatibility).
2.  **Explain** why Python + Alpine is often an anti-pattern (Missing Wheels).
3.  **Implement** a Distroless container for a Python application.
4.  **Remove** unneeded files (tests, pycache) manually to shave megabytes.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Docker.

### Software Environment
- `docker`.

---

## 📖 Theoretical Foundation

### 1. The Base Image Trinity
*   **Debian Slim (`python:3.9-slim`):**
    *   **Pros:** Compatible with `manylinux` wheels (NumPy/PyTorch install instantly).
    *   **Cons:** Has Shell/Apt (Attack surface).
    *   **Verdict:** Best for Data Science/ML.
*   **Alpine (`python:3.9-alpine`):**
    *   **Pros:** Tiny (5MB base).
    *   **Cons:** Uses `musl libc`. Standard wheels use `glibc`. Pip will try to *compile* NumPy from source (takes 20 mins, needs gcc). Final image often *larger* than Slim due to build deps.
    *   **Verdict:** Avoid for ML. Good for Go/Rust.
*   **Distroless (`gcr.io/distroless/python3-debian11`):**
    *   **Pros:** No Shell (`/bin/sh`). No Package Manager (`apt`). Minimal attack surface.
    *   **Cons:** Hard to debug (cannot `kubectl exec` into it).
    *   **Verdict:** Gold Standard for Production.

### 2. The "Strip" Command
Compiled libraries (`.so` files in PyTorch) differ:
*   **Debug Symbols:** Info for gdb. Huge.
*   **Stripped:** Code only. Small.
*   `strip --strip-unneeded libtorch.so` can save 100s of MBs.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Using Distroless

Distroless images don't have `pip`. You MUST use Multi-Stage.

#### 📁 `distroless.Dockerfile`
```dockerfile
# Stage 1: Build Class
FROM python:3.9-slim AS builder

WORKDIR /app
COPY requirements.txt .
# Install into a specific directory --target
RUN pip install --no-cache-dir -r requirements.txt --target /app/site-packages

# Stage 2: Distroless Runtime
# Supports Python 3.9
FROM gcr.io/distroless/python3-debian11

WORKDIR /app
# Copy libraries
COPY --from=builder /app/site-packages /app/site-packages
# Copy app
COPY src/ .

# Set PYTHONPATH
ENV PYTHONPATH=/app/site-packages

CMD ["app.py"]
```

### 👨‍💻 Lab Operations: Debugging the Undebuggable

1.  **Build & Run:**
    ```bash
    docker build -t app:distroless -f distroless.Dockerfile .
    docker run app:distroless
    ```
2.  **Try to Exec:**
    ```bash
    docker exec -it <container-id> /bin/sh
    ```
    *Result:* `OCI runtime exec failed: exec: "/bin/sh": stat /bin/sh: no such file or directory`.
    **Success!** An attacker with RCE (Remote Code Execution) vulnerability cannot open a shell.

3.  **How to Debug then?**
    *   **Logs:** `docker logs`.
    *   **Ephemeral Debug Containers (K8s v1.23+):**
        ```bash
        kubectl debug -it <pod> --image=busybox --target=<container>
        ```
        This attaches a *sidecar* with a shell to the process namespace.

---

## 🔬 Lab Exercise: "The Alpine Trap"

### Task
Measure build time.
1.  Create `Dockerfile.alpine` importing `pandas`.
2.  Build it. Watch it hang at `Building wheel for pandas`.
3.  Check size.
4.  Compare with `Dockerfile.slim`.
5.  **Observation:** Slim builds in 10s. Alpine builds in 5m (or fails).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Default to Slim:** For Python ML apps, `python:X.Y-slim` is the sweet spot of size and compatibility.
2.  **Use Distroless for High Security:** If you process sensitive data (FinTech/Health), the lack of shell is a major compliance win.
3.  **Clean Up:** `find / -name "__pycache__" -delete` in your Dockerfile before the final COPY.

### API Summary
```bash
docker system df # Check space usage
```

---

**Day 72 Complete** ✅

*Next: Day 73 - GPU Container Optimization - The massive CUDA blobs.*
