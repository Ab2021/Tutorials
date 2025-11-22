# Lab 5.3: Dockerfiles & Building Images

## 🎯 Objective

Learn how to package your own application into a Docker Image. You will write a `Dockerfile`, understand layers, and optimize the build process.

## 📋 Prerequisites

-   Completed Lab 5.2.
-   VS Code.

## 📚 Background

### The Recipe (Dockerfile)
A text file containing instructions to build the image.
-   `FROM`: Base image (e.g., `python:3.9`).
-   `WORKDIR`: Set working directory (like `cd`).
-   `COPY`: Copy files from host to container.
-   `RUN`: Execute commands (during build).
-   `CMD`: Command to run (when container starts).

---

## 🔨 Hands-On Implementation

### Part 1: The Python App 🐍

1.  **Create a folder:**
    ```bash
    mkdir docker-lab-3
    cd docker-lab-3
    ```

2.  **Create `app.py`:**
    ```python
    import os
    print(f"Hello from {os.environ.get('NAME', 'World')}!")
    ```

### Part 2: The Dockerfile 📄

1.  **Create `Dockerfile`:**
    ```dockerfile
    # 1. Base Image
    FROM python:3.9-slim

    # 2. Set Directory
    WORKDIR /app

    # 3. Copy Files
    COPY app.py .

    # 4. Default Command
    CMD ["python", "app.py"]
    ```

### Part 3: Build & Run 🏗️

1.  **Build:**
    ```bash
    docker build -t my-python-app .
    ```
    *Note:* The `.` means "current directory".

2.  **Run:**
    ```bash
    docker run my-python-app
    ```
    *Output:* `Hello from World!`

3.  **Run with Env Var:**
    ```bash
    docker run -e NAME=DevOps my-python-app
    ```
    *Output:* `Hello from DevOps!`

### Part 4: Layer Caching (Optimization) ⚡

1.  **Modify Dockerfile:**
    Add a `RUN` step before `COPY`.
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    RUN pip install flask  # <--- New Step
    COPY app.py .
    CMD ["python", "app.py"]
    ```

2.  **Build:**
    Notice it downloads Flask.

3.  **Build Again:**
    Notice "Using cache". Instant!

4.  **Modify `app.py` (Add a comment).**

5.  **Build Again:**
    Notice `RUN pip install flask` is **Cached**. Only `COPY` runs again.
    *Lesson:* Put stable things (dependencies) early. Put changing things (code) late.

---

## 🎯 Challenges

### Challenge 1: The .dockerignore (Difficulty: ⭐⭐)

**Task:**
1.  Create a huge file `big_data.tmp` (100MB).
2.  Build the image. Notice the "Sending build context" takes time.
3.  Create `.dockerignore` and add `*.tmp`.
4.  Build again. It should be fast.

### Challenge 2: Multi-Stage Build (Difficulty: ⭐⭐⭐⭐)

**Scenario:** You are building a Go or C++ app. You need the compiler (`gcc`) to build, but you don't need it to run.
**Task:**
Write a Dockerfile that:
1.  Stage 1: Compiles a C program.
2.  Stage 2: Copies *only* the binary to a fresh `alpine` image.
    *Result:* A tiny image (5MB) instead of a huge one (500MB).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Create `.dockerignore`:
```text
*.tmp
```

**Challenge 2 (C Example):**
```dockerfile
# Stage 1: Build
FROM gcc:latest AS builder
COPY main.c .
RUN gcc -o myapp main.c

# Stage 2: Run
FROM alpine:latest
COPY --from=builder /myapp /myapp
CMD ["/myapp"]
```
</details>

---

## 🔑 Key Takeaways

1.  **Order Matters**: Optimize caching by ordering instructions from least changed to most changed.
2.  **Small Base Images**: Use `alpine` or `slim` variants to reduce attack surface and download time.
3.  **One Process**: A container should run **one** process (PID 1). Don't try to run SSH + Nginx + MySQL in one container.

---

## ⏭️ Next Steps

We have an image. Now let's connect it to the network.

Proceed to **Lab 5.4: Docker Networking**.
