# Day 75: Turbocharged Builds: BuildKit & Caching
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** Waiting for `pip install` to download 2GB of wheels every time you add a library is painful. Use **BuildKit Cache Mounts** to persist dependency caches across builds.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Enable** Docker BuildKit backend (`DOCKER_BUILDKIT=1`).
2.  **Use** `RUN --mount=type=cache` to speed up pip and apt installs.
3.  **Use** `RUN --mount=type=secret` to inject credentials without leaving traces in the image.
4.  **Parallelize** multi-stage builds using the dependency graph.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Docker.

### Software Environment
- Docker version > 18.09.

---

## 📖 Theoretical Foundation

### 1. Traditional vs Cache Mount
*   **Traditional:** If `requirements.txt` changes, the `RUN pip install` layer is blown away. Docker starts a fresh container. Pip cache is empty. Download everything again.
*   **BuildKit Mount:** We mount a host directory (volume) into the build container at `/root/.cache/pip`. Even if the layer is rebuilt, the *volume* persists. Pip sees cached wheels and installs instantly.

### 2. Secrets
Passing `API_KEY` as `ARG` logs it in `docker history`.
BuildKit allows mounting a secret file *temporarily* into the build step, never committing it to the layer.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Syntax

You must add the syntax directive at the top.

#### 📁 `fast.Dockerfile`
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

# 1. Apt Cache
rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential

# 2. Pip Cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .
```

### 👨‍💻 Core Implementation: Remote Cache (CI/CD)

In GitHub Actions, your local cache is lost. You need to push cache to the Registry.

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag my-app:latest \
  --cache-to type=inline \
  --cache-from type=registry,ref=my-app:latest \
  --push .
```
This embeds cache metadata into the image layers pushed to the registry.

---

## 🔬 Lab Exercise: "The Speed Test"

### Task
1.  Create `requirements.txt` with `numpy`.
2.  Build with `--mount=type=cache`. Time: 10s.
3.  Add `pandas` to requirements.
4.  Build again.
    *   **Old Docker:** Would download numpy AND pandas.
    *   **BuildKit:** Finds numpy in cache, downloads only pandas.
    *   **Time:** 2s.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Defaults:** BuildKit is default in Docker Desktop/Boot2Docker now. Ensure CI runners have it enabled.
2.  **Concurrency:** If you have 3 stages that don't depend on each other, BuildKit runs them in parallel.
3.  **Secrets:** Use `--mount=type=secret,id=mykey` to access files passed via `docker build --secret id=mykey,src=key.txt`.

### API Summary
```bash
docker buildx create --use # Enable advanced features
```

---

**Day 75 Complete** ✅

*Next: Day 76 - Registry Management - Hosting your own Docker Hub (Harbor).*
