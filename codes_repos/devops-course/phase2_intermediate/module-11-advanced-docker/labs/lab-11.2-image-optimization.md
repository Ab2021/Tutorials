# Lab 11.2: Docker Image Optimization

## Objective
Optimize Docker images for size, build speed, and security using multi-stage builds and best practices.

## Prerequisites
- Docker installed
- Basic Dockerfile knowledge
- Completed Module 5 (Docker Fundamentals)

## Learning Objectives
- Reduce image size by 70%+
- Implement multi-stage builds
- Use .dockerignore effectively
- Apply layer caching strategies

---

## Part 1: The Problem - Bloated Images

### Unoptimized Dockerfile

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    build-essential

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt
RUN pip3 install pytest flake8  # Dev dependencies

CMD ["python3", "app.py"]
```

**Result:** 1.2 GB image!

---

## Part 2: Optimization Techniques

### 1. Use Smaller Base Images

```dockerfile
# Before: ubuntu:22.04 (77MB)
FROM python:3.11-slim  # Only 45MB

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "app.py"]
```

**Savings:** 32MB base image reduction

### 2. Multi-Stage Builds

```dockerfile
# Build stage
FROM python:3.11 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

**Savings:** Build tools not in final image

### 3. Layer Caching

```dockerfile
# ❌ Bad - Invalidates cache on any code change
COPY . .
RUN pip install -r requirements.txt

# ✅ Good - Cache dependencies separately
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### 4. Use .dockerignore

```
# .dockerignore
.git
.gitignore
README.md
tests/
*.pyc
__pycache__
.pytest_cache
.venv
node_modules
.DS_Store
```

---

## Part 3: Real-World Example

### Application Structure

```bash
mkdir docker-optimization
cd docker-optimization

# Create app
cat > app.py << 'EOF'
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Optimized Docker Image!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

# Requirements
echo "flask==2.3.0" > requirements.txt
```

### Optimized Dockerfile

```dockerfile
# Multi-stage build
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Final stage
FROM python:3.11-alpine

# Create non-root user
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install dependencies
RUN pip install --no-cache /wheels/*

# Copy application
COPY --chown=appuser:appuser app.py .

# Switch to non-root user
USER appuser

EXPOSE 5000
CMD ["python", "app.py"]
```

### Build and Compare

```bash
# Build optimized image
docker build -t flask-app:optimized .

# Check size
docker images flask-app:optimized

# Compare with unoptimized
docker build -f Dockerfile.unoptimized -t flask-app:unoptimized .
docker images | grep flask-app
```

**Results:**
- Unoptimized: 1.2 GB
- Optimized: 65 MB
- **Reduction: 94%!**

---

## Part 4: Advanced Optimizations

### Use Distroless Images

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o main .

FROM gcr.io/distroless/static-debian11
COPY --from=builder /app/main /
CMD ["/main"]
```

**Benefits:**
- No shell, package manager, or unnecessary binaries
- Minimal attack surface
- ~2MB final image

### Combine RUN Commands

```dockerfile
# ❌ Bad - Creates multiple layers
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get clean

# ✅ Good - Single layer
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

---

## Challenges

### Challenge 1: Optimize Node.js App

Given this Dockerfile, reduce it from 900MB to <200MB:

```dockerfile
FROM node:18
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "server.js"]
```

<details>
<summary>Solution</summary>

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
USER node
CMD ["node", "server.js"]
```
</details>

---

## Success Criteria

✅ Reduced image size by 70%+  
✅ Implemented multi-stage build  
✅ Used .dockerignore  
✅ Applied layer caching  
✅ Running as non-root user  

---

## Key Learnings

- **Alpine images are smallest** - But may have compatibility issues
- **Multi-stage builds are essential** - Separate build and runtime
- **Order matters** - Put frequently changing layers last
- **Security through minimalism** - Fewer packages = fewer vulnerabilities

---

## Next Steps

- **Lab 11.3:** Docker security scanning
- **Lab 11.4:** Health checks and monitoring

**Estimated Time:** 45 minutes  
**Difficulty:** Intermediate
