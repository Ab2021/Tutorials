# Day 16: Docker & Containerization

## 1. It Works on My Machine?

The age-old problem: "It works on my laptop but fails in production."
*   **Cause**: Different OS, different library versions, different file paths.
*   **Solution**: **Containers**. Package the code *and* the environment (OS libs, dependencies) into a single artifact.

### 1.1 VM vs Container
*   **Virtual Machine (VM)**: Virtualizes the Hardware. Each VM has a full OS kernel. Heavy (GBs). Slow boot.
*   **Container**: Virtualizes the OS. Shares the Host Kernel. Lightweight (MBs). Instant boot.

---

## 2. Docker Core Concepts

1.  **Dockerfile**: The blueprint (Recipe).
2.  **Image**: The built artifact (The Cake). Immutable.
3.  **Container**: The running instance (A Slice of Cake). Mutable (ephemeral).
4.  **Registry**: The library (Docker Hub, ECR). Stores images.

### 2.1 The Dockerfile
```dockerfile
# 1. Base Image (Start with OS + Runtime)
FROM python:3.11-slim

# 2. Work Directory
WORKDIR /app

# 3. Copy Dependencies first (Layer Caching!)
COPY requirements.txt .

# 4. Install Deps
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code
COPY . .

# 6. Command to run
CMD ["python", "app.py"]
```

### 2.2 Layer Caching
Docker builds in layers. If you change line 5, Docker reuses layers 1-4 from cache.
*   *Tip*: Always copy `requirements.txt` and install deps *before* copying source code. This way, changing code doesn't trigger a re-install of libraries.

---

## 3. Advanced Docker Patterns

### 3.1 Multi-Stage Builds
Keep your production image tiny.
*   **Stage 1 (Builder)**: Has compilers (GCC), SDKs, Maven. Builds the binary.
*   **Stage 2 (Runner)**: Has only the runtime (Alpine/Distroless). Copies binary from Stage 1.
*   *Result*: A 10MB image instead of 500MB.

```dockerfile
# Stage 1: Build
FROM golang:1.21 as builder
WORKDIR /app
COPY . .
RUN go build -o myapp main.go

# Stage 2: Run
FROM alpine:latest
WORKDIR /root/
COPY --from=builder /app/myapp .
CMD ["./myapp"]
```

### 3.2 Docker Compose
Orchestrate multiple containers locally.
```yaml
version: '3.8'
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
```

---

## 4. Security Best Practices

1.  **Don't run as Root**: By default, containers run as root. If escaped, attacker has root on host.
    *   `USER appuser`
2.  **Use Trusted Base Images**: Don't use `frank/python-hacked`. Use `python:3.11-slim`.
3.  **Scan Images**: Use tools like `Trivy` or `Snyk` to find CVEs in your dependencies.
4.  **Minimal Images**: Use `distroless` or `alpine` to reduce attack surface.

---

## 5. Summary

Today we containerized our applications.
*   **Immutable Infrastructure**: Build once, run anywhere.
*   **Isolation**: Dependencies don't clash.
*   **Efficiency**: Multi-stage builds save space.

**Tomorrow (Day 17)**: One container is easy. Managing 1000 containers is hard. We will introduce **Kubernetes**, the operating system of the cloud.
