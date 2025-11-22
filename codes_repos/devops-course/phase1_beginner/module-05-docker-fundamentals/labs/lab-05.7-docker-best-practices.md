# Lab 5.7: Docker Best Practices (Security & Size)

## 🎯 Objective

Optimize your Docker images. You will take a "bad" Dockerfile (large, insecure) and refactor it into a "good" one (small, secure, fast).

## 📋 Prerequisites

-   Completed Lab 5.3.

## 📚 Background

### The "Bad" Image
-   Runs as **Root**. (Security risk).
-   Uses **Latest** tag. (Unpredictable).
-   Includes **Build Tools** in final image. (Bloat).
-   Has **Secrets** baked in. (Disaster).

### The "Good" Image
-   Runs as **Non-Root User**.
-   Uses **Specific Version** tags (`python:3.9.1-slim`).
-   Uses **Multi-Stage Builds**.
-   Uses **Environment Variables** for config.

---

## 🔨 Hands-On Implementation

### Part 1: The Bad Dockerfile 😈

1.  **Create `bad/Dockerfile`:**
    ```dockerfile
    FROM ubuntu:latest
    RUN apt-get update && apt-get install -y python3 pip
    COPY . /app
    WORKDIR /app
    RUN pip install -r requirements.txt
    CMD ["python3", "app.py"]
    ```

2.  **Build & Measure:**
    ```bash
    docker build -t bad-image ./bad
    docker images bad-image
    ```
    *Result:* ~800MB. Runs as root.

### Part 2: Optimization 1 (Base Image) 📉

1.  **Change Base:**
    Use `python:3.9-slim` instead of `ubuntu`.
    *Result:* ~150MB.

### Part 3: Optimization 2 (Security) 🔒

1.  **Create User:**
    ```dockerfile
    RUN useradd -m myuser
    USER myuser
    ```
    *Benefit:* If hacker breaks into container, they are not Root on your host.

### Part 4: Optimization 3 (Caching) ⚡

1.  **Order Instructions:**
    Copy `requirements.txt` BEFORE the rest of the code.
    ```dockerfile
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    ```

### Part 5: The Final "Good" Dockerfile 😇

```dockerfile
# Pin version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m appuser

# Install dependencies (Cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Switch to user
USER appuser

# Run
CMD ["python", "app.py"]
```

**Build & Measure:**
`docker build -t good-image .`
*Result:* ~120MB. Secure. Fast rebuilds.

---

## 🎯 Challenges

### Challenge 1: Linting (Difficulty: ⭐⭐)

**Task:**
Install **Hadolint** (Haskell Dockerfile Linter) or use the VS Code extension.
Run it against the "Bad" Dockerfile.
Fix the warnings (e.g., "Pin versions in apt get install").

### Challenge 2: Scan for Vulnerabilities (Difficulty: ⭐⭐⭐)

**Task:**
Run `docker scan good-image` (uses Snyk).
See if your base image has known CVEs.
*Note:* You might need to log in to Docker Hub.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Hadolint will complain about `ubuntu:latest` and missing version pins for `apt-get install python3`.

**Challenge 2:**
`docker scan` will output a list of CVEs. To fix, usually you update the base image tag to a newer patch version.
</details>

---

## 🔑 Key Takeaways

1.  **Never Run as Root**: This is the #1 security rule for containers.
2.  **Slim is In**: Use `alpine` or `slim` images. Distroless is even better (Google it).
3.  **Don't Leak Secrets**: Never `COPY .env .`. Use `.dockerignore`.

---

## ⏭️ Next Steps

We have a perfect image. Let's share it with the team.

Proceed to **Lab 5.8: Docker Registry & Hub**.
