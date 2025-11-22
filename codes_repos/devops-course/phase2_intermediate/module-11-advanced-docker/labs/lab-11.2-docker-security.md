# Lab 11.2: Docker Security & Distroless

## 🎯 Objective

Harden your containers. You will learn about **Distroless** images (Google's secure base images), running as non-root, and scanning for vulnerabilities with Trivy.

## 📋 Prerequisites

-   Completed Lab 11.1.
-   Trivy installed (`apt install trivy` or `brew install trivy`).

## 📚 Background

### Distroless
"Distroless" images contain *only* your application and its runtime dependencies. They do not contain package managers (`apt`), shells (`/bin/sh`), or any other programs you would expect to find in a standard Linux distribution.
-   **Why?** If a hacker gets in, they can't run `ls`, `curl`, or `cat`. They are trapped.

---

## 🔨 Hands-On Implementation

### Part 1: Building with Distroless 🛡️

1.  **Modify `Dockerfile` (from Lab 11.1):**
    ```dockerfile
    FROM golang:1.19 AS builder
    WORKDIR /app
    COPY main.go .
    RUN CGO_ENABLED=0 GOOS=linux go build -o myapp main.go

    # Use Google's Distroless Base
    FROM gcr.io/distroless/static-debian11
    COPY --from=builder /app/myapp /
    CMD ["/myapp"]
    ```

2.  **Build:**
    ```bash
    docker build -t distroless-app .
    ```

3.  **Try to Shell in:**
    ```bash
    docker run -it distroless-app sh
    ```
    *Result:* `docker: Error response from daemon: OCI runtime create failed: ... exec: "sh": executable file not found`.
    **Success!** There is no shell.

### Part 2: Vulnerability Scanning (Trivy) 🕵️‍♂️

1.  **Scan the "Fat" image (from Lab 11.1):**
    ```bash
    trivy image fat-app
    ```
    *Result:* Likely hundreds of vulnerabilities (CVEs) in the base OS libraries.

2.  **Scan the "Distroless" image:**
    ```bash
    trivy image distroless-app
    ```
    *Result:* Zero (or very few) vulnerabilities.

### Part 3: Running as Non-Root 👤

By default, Docker runs as Root. This is bad.

1.  **Modify Dockerfile:**
    Distroless has a built-in `nonroot` user.
    ```dockerfile
    FROM gcr.io/distroless/static-debian11
    COPY --from=builder /app/myapp /
    USER nonroot:nonroot
    CMD ["/myapp"]
    ```

2.  **Build & Run:**
    ```bash
    docker build -t secure-app .
    docker run secure-app
    ```

---

## 🎯 Challenges

### Challenge 1: Debugging Distroless (Difficulty: ⭐⭐⭐)

**Task:**
Since there is no shell, how do you debug a Distroless container?
Research **Ephemeral Containers** (Kubernetes) or `docker run --entrypoint`.
Try to run the container with `--entrypoint /myapp` (it works) vs `--entrypoint ls` (fails).

### Challenge 2: Read-Only Filesystem (Difficulty: ⭐⭐)

**Task:**
Run the container with `--read-only`.
```bash
docker run --read-only secure-app
```
*Goal:* Prevent any file modification at runtime. If your app needs to write logs, mount a volume for that specific path.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
In Docker, you can't easily debug Distroless without a shell. In Kubernetes 1.23+, you can use `kubectl debug` to attach a sidecar container that *does* have a shell.

**Challenge 2:**
If the app tries to write to disk, it will crash. This is good for immutable infrastructure.
</details>

---

## 🔑 Key Takeaways

1.  **Less is More**: The fewer binaries in your image, the safer it is.
2.  **Supply Chain Security**: Scanning images (Trivy) ensures you aren't inheriting known bugs.
3.  **Defense in Depth**: Distroless + Non-Root + Read-Only = Very hard target.

---

## ⏭️ Next Steps

We have mastered the single container. Now it's time to orchestrate thousands of them.

Proceed to **Module 12: Kubernetes Fundamentals**.
