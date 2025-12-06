# Day 71: The 10GB Snake: Taming Large Containers
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** ML Containers are notoriously huge (PyTorch + CUDA = 5GB+). Learn **Multi-Stage Builds** and **Layer Caching** to shrink your images and speed up deployments.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Analyze** docker history to pinpoint bloated layers.
2.  **Explain** why `RUN apt update && apt install` must be in the same line.
3.  **Implement** a Multi-Stage Build to separate Build Tools (`gcc`) from Runtime Artifacts.
4.  **Use** `.dockerignore` to prevent leaking secrets and git history.
5.  **Reduce** a naive PyTorch container from ~6GB to ~3GB.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- A machine with Docker installed.

### Software Environment
- `docker` CLI.
- `dive` (Optional but recommended tool for inspecting layers).

---

## 📖 Theoretical Foundation

### 1. The Overlay Filesystem (UnionFS)
Docker images are immutable read-only layers.
*   **Layer 1:** ❌ 100MB File created.
*   **Layer 2:** 🗑️ File deleted.
*   **Result:** The Image Size is **100MB**. The file is hidden, not gone.
*   **Rule:** You must create and clean up in the *same* `RUN` layer to save space.

### 2. The Build Context
When you run `docker build .`, the CLI sends the *entire current directory* to the Docker Daemon.
If you have a `.git` folder (200MB) or `data.csv` (1GB), it gets uploaded *before build starts*.
**Solution:** `.dockerignore`.

### 3. Multi-Stage Builds
*   **Stage 0 (Builder):** Has Compilers (gcc, g++), Headers, Git. Builds the code/wheels.
*   **Stage 1 (Runner):** Has only Python Runtime and Libraries. `COPY --from=0 /build /app`.
*   **Result:** Final image has no `gcc`. Smaller and More Secure.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Naive (Bad) Approach

#### 📁 `bad/Dockerfile`
```dockerfile
FROM python:3.9
WORKDIR /app

# Bad: Invalidates cache every time code changes
COPY . . 

# Bad: Creates large layer cache
RUN pip install torch numpy pandas

# Bad: Compilers left in image
RUN apt-get update
RUN apt-get install -y build-essential

CMD ["python", "app.py"]
```
**Size:** ~1.2GB (Generic Python) + Code.

### 👨‍💻 Core Implementation: The Optimized Approach

#### 📁 `good/Dockerfile`
```dockerfile
# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Build wheels instead of installing directly
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install pre-built wheels
RUN pip install --no-cache /wheels/*

COPY src/ .

CMD ["python", "app.py"]
```

### 👨‍💻 Layer Tricks

**1. Combine Apt Commands:**
```dockerfile
# GOOD
RUN apt-get update && apt-get install -y vim && rm -rf /var/lib/apt/lists/*
```

**2. Python Cache:**
```dockerfile
# Don't keep pip cache (~/.cache/pip) in the image
RUN pip install --no-cache-dir -r requirements.txt
```

---

## 🔬 Lab Exercise: "The Dive Audit"

### Task
Use `dive` or `docker history` to inspect.
1.  Build the Bad Dockerfile. `docker build -t app:bad -f bad/Dockerfile .`
2.  Run `docker history app:bad`.
    *   Notice the Size of `COPY . .`. If you changed one line of code, did the `pip install` layer rebuild? (Yes, because COPY came before RUN).
3.  Build the Good Dockerfile.
4.  **Observation:**
    *   Smaller size.
    *   If you change `src/file.py`, the `pip install` step is **Cached** (Skipped). Build takes 1s instead of 60s.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Order Matters:** Always put **Least Frequently Changing** instructions (System Deps) first, and **Most Frequently Changing** (Source Code) last. This maximizes Layer Caching.
2.  **Distroless?** "Slim" images are usually good enough for Python. "Alpine" is tricky for Python because of `musl` vs `glibc` compatibility issues (wheels often break). Stick to `python:3.9-slim-bullseye`.
3.  **Ignore:** Always add `.git`, `__pycache__`, `venv`, and `*.csv` to `.dockerignore`.

### API Summary
```bash
docker build --target builder . # Build only the first stage (debug)
docker system prune # Clean build cache
```

---

**Day 71 Complete** ✅

*Next: Day 72 - Minimizing Image Size - Distroless, Alpine, and stripping unnecessary files.*
