# Lab 11.3: Docker Security Scanning

## Objective
Scan Docker images for vulnerabilities and implement security best practices.

## Prerequisites
- Docker installed
- Basic Dockerfile knowledge

## Learning Objectives
- Scan images for CVEs using Trivy
- Implement security best practices
- Use distroless base images
- Run containers as non-root

---

## Part 1: Install Trivy

```bash
# macOS
brew install trivy

# Linux
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

# Verify
trivy --version
```

---

## Part 2: Scan Images

### Scan Public Image

```bash
# Scan nginx
trivy image nginx:latest

# Scan with severity filter
trivy image --severity HIGH,CRITICAL nginx:latest

# Output to JSON
trivy image -f json -o results.json nginx:latest
```

### Scan Custom Image

```dockerfile
# Dockerfile
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y curl
COPY app.sh /app.sh
CMD ["/app.sh"]
```

```bash
docker build -t myapp:v1 .
trivy image myapp:v1
```

**Typical findings:**
- CVE-2021-XXXX: OpenSSL vulnerability (HIGH)
- CVE-2022-XXXX: curl vulnerability (MEDIUM)

---

## Part 3: Fix Vulnerabilities

### Use Updated Base Image

```dockerfile
# Before: ubuntu:20.04 (many CVEs)
FROM ubuntu:22.04  # Newer = fewer CVEs

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

### Use Distroless

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o main .

FROM gcr.io/distroless/static-debian11
COPY --from=builder /app/main /
USER nonroot:nonroot
CMD ["/main"]
```

**Result:** Zero CVEs (no OS packages!)

---

## Part 4: Security Best Practices

### 1. Run as Non-Root

```dockerfile
FROM python:3.11-slim

# Create user
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser . .

# Switch to non-root
USER appuser

CMD ["python", "app.py"]
```

### 2. Use .dockerignore

```
.git
.env
secrets/
*.key
*.pem
```

### 3. Scan in CI/CD

```yaml
# GitHub Actions
- name: Run Trivy scan
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'myapp:${{ github.sha }}'
    severity: 'CRITICAL,HIGH'
    exit-code: '1'  # Fail build on vulnerabilities
```

---

## Part 5: Docker Bench Security

```bash
# Run Docker Bench
docker run -it --net host --pid host --userns host --cap-add audit_control \
  -v /var/lib:/var/lib \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /etc:/etc --label docker_bench_security \
  docker/docker-bench-security

# Check results
# [PASS] 1.1.1 Ensure a separate partition for containers
# [WARN] 1.2.1 Ensure the container host has been Hardened
# [INFO] 2.1 Run the Docker daemon as a non-root user
```

---

## Challenges

### Challenge 1: Fix All HIGH Vulnerabilities

Scan an image and fix all HIGH severity issues.

### Challenge 2: Implement Content Trust

Enable Docker Content Trust to verify image signatures.

```bash
export DOCKER_CONTENT_TRUST=1
docker pull nginx:latest  # Will verify signature
```

---

## Success Criteria

✅ Scanned images with Trivy  
✅ Identified and fixed vulnerabilities  
✅ Implemented non-root user  
✅ Used distroless base image  
✅ Integrated scanning in CI/CD  

---

## Key Learnings

- **Scan regularly** - New CVEs discovered daily
- **Update base images** - Newer versions have fewer vulnerabilities
- **Minimize attack surface** - Fewer packages = fewer CVEs
- **Never run as root** - Principle of least privilege

**Estimated Time:** 40 minutes  
**Difficulty:** Intermediate
