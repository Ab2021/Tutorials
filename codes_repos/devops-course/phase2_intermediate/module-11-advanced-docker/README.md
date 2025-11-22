# Advanced Docker Techniques

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of advanced Docker concepts, including:
- **Optimization**: Reducing image size by 90% using Multi-Stage Builds and Distroless images.
- **Security**: Hardening containers, scanning for vulnerabilities (CVEs), and running as non-root.
- **Resilience**: Implementing Health Checks and Resource Limits (CPU/RAM).
- **Networking**: Deep dive into custom networks and DNS service discovery.
- **Orchestration**: Introduction to Docker Swarm for multi-node clusters.

---

## 📖 Theoretical Concepts

### 1. Multi-Stage Builds

The single most effective way to optimize Docker images.
- **Problem**: You need Go compilers to build your app, but you don't need them to run it. Shipping the compiler wastes space and adds security risks.
- **Solution**: Use one stage to build (with tools) and copy *only* the binary to a second, minimal stage (runtime).

### 2. Container Security

- **Least Privilege**: Never run as `root`. If an attacker breaks out of the app, they have root on the container (and potentially the host).
- **Distroless Images**: Images that contain *only* your application and its runtime dependencies. No shell, no package manager, no text editors.
- **Scanning**: Tools like **Trivy** or **Clair** scan your images for known vulnerabilities (CVEs) in OS packages and libraries.

### 3. Resource Management

- **Memory Limits**: If a container leaks memory, it can crash the host. Set limits (`--memory="512m"`).
- **CPU Limits**: Prevent one container from starving others (`--cpus="1.5"`).
- **Health Checks**: Docker can poll your app (`curl localhost/health`). If it fails, Docker can restart it automatically.

### 4. Advanced Networking

- **User-Defined Bridges**: Containers on the same custom bridge can resolve each other by name (DNS).
- **Overlay Networks**: Connect containers across multiple physical hosts (Swarm/Kubernetes).

---

## 🔧 Practical Examples

### Multi-Stage Dockerfile (Go)

```dockerfile
# Stage 1: Build
FROM golang:1.19 AS builder
WORKDIR /app
COPY . .
RUN go build -o myapp main.go

# Stage 2: Runtime
FROM gcr.io/distroless/static-debian11
COPY --from=builder /app/myapp /
CMD ["/myapp"]
```

### Security Hardening

```dockerfile
# Create a non-root user
RUN useradd -m appuser
USER appuser
```

### Running with Limits

```bash
docker run -d \
  --name limited-app \
  --memory="256m" \
  --cpus="0.5" \
  --health-cmd="curl -f http://localhost/ || exit 1" \
  nginx
```

### Scanning with Trivy

```bash
trivy image python:3.4-alpine
```

---

## 🎯 Hands-on Labs

- [Lab 11.1: Multi-Stage Builds & Optimization](./labs/lab-11.1-multi-stage-builds.md)
- [Lab 11.2: Docker Security & Distroless](./labs/lab-11.2-docker-security.md)
- [Lab 11.3: Docker Security Scanning](./labs/lab-11.3-docker-security-scanning.md)
- [Lab 11.4: Health Checks](./labs/lab-11.4-health-checks.md)
- [Lab 11.5: Resource Limits](./labs/lab-11.5-resource-limits.md)
- [Lab 11.6: Docker Networking Advanced](./labs/lab-11.6-docker-networking-advanced.md)
- [Lab 11.7: Custom Networks](./labs/lab-11.7-custom-networks.md)
- [Lab 11.8: Docker Swarm Intro](./labs/lab-11.8-docker-swarm-intro.md)
- [Lab 11.9: Container Orchestration](./labs/lab-11.9-container-orchestration.md)
- [Lab 11.10: Production Best Practices](./labs/lab-11.10-production-best-practices.md)

---

## 📚 Additional Resources

### Official Documentation
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)

### Tools
- [Trivy](https://github.com/aquasecurity/trivy)
- [Hadolint](https://github.com/hadolint/hadolint) - Dockerfile Linter.

---

## 🔑 Key Takeaways

1.  **Small is Secure**: Fewer files = fewer vulnerabilities. Use Multi-stage builds.
2.  **Don't Trust Defaults**: Default Docker settings are for usability, not security. Harden them.
3.  **Automate Scanning**: Fail the CI pipeline if high-severity CVEs are found.
4.  **Limits are Mandatory**: Never run a container in production without CPU/RAM limits.

---

## ⏭️ Next Steps

1.  Complete the labs to master image optimization.
2.  Proceed to **[Module 12: Kubernetes Fundamentals](../module-12-kubernetes-fundamentals/README.md)** to learn how to manage thousands of containers.
