# Lab 11.1: Multi-Stage Builds & Optimization

## 🎯 Objective

Shrink your images. In Phase 1, we built simple images. Now, we will use **Multi-Stage Builds** to separate the build environment (compilers, SDKs) from the runtime environment (just the binary), reducing image size by 90%+.

## 📋 Prerequisites

-   Docker installed.
-   Basic Go or C knowledge (optional, code provided).

## 📚 Background

### The Problem
To build a Go app, you need the Go compiler (500MB). To run it, you need... nothing (5MB).
If you ship the compiler to production:
1.  **Size**: Wasted bandwidth/storage.
2.  **Security**: Hackers can use the compiler to build malware on your server.

### The Solution
**Multi-Stage Build**:
-   Stage 1 (`builder`): Has compiler. Builds the app.
-   Stage 2 (`runner`): Empty. Copies only the binary from Stage 1.

---

## 🔨 Hands-On Implementation

### Part 1: The Application (Go) 🐹

1.  **Create `main.go`:**
    ```go
    package main
    import "fmt"
    func main() {
        fmt.Println("Hello from a Tiny Container!")
    }
    ```

### Part 2: The "Fat" Image (Single Stage) 🐘

1.  **Create `Dockerfile.fat`:**
    ```dockerfile
    FROM golang:1.19
    WORKDIR /app
    COPY main.go .
    RUN go build -o myapp main.go
    CMD ["./myapp"]
    ```

2.  **Build & Measure:**
    ```bash
    docker build -f Dockerfile.fat -t fat-app .
    docker images fat-app
    ```
    *Result:* ~800MB.

### Part 3: The "Tiny" Image (Multi-Stage) 🐜

1.  **Create `Dockerfile`:**
    ```dockerfile
    # Stage 1: Builder
    FROM golang:1.19 AS builder
    WORKDIR /app
    COPY main.go .
    # CGO_ENABLED=0 creates a static binary (no dependencies)
    RUN CGO_ENABLED=0 GOOS=linux go build -o myapp main.go

    # Stage 2: Runner
    FROM alpine:latest
    WORKDIR /root/
    # Copy ONLY the binary from the builder stage
    COPY --from=builder /app/myapp .
    CMD ["./myapp"]
    ```

2.  **Build & Measure:**
    ```bash
    docker build -t tiny-app .
    docker images tiny-app
    ```
    *Result:* ~10MB. (Alpine is 5MB + Binary is 2MB).
    *Reduction:* **98%**.

### Part 4: Scratch (The Empty Image) 👻

Can we go smaller? Yes. `scratch` is a special Docker image that is completely empty (0 bytes).

1.  **Modify `Dockerfile`:**
    Change `FROM alpine:latest` to `FROM scratch`.
    
2.  **Build:**
    ```bash
    docker build -t scratch-app .
    docker images scratch-app
    ```
    *Result:* ~2MB. (Pure binary size).

---

## 🎯 Challenges

### Challenge 1: Node.js Multi-Stage (Difficulty: ⭐⭐⭐)

**Task:**
Node.js isn't a compiled language, but you still need build tools for `npm install`.
Write a multi-stage Dockerfile for a Node app:
1.  Stage 1: Install dependencies (`npm install`).
2.  Stage 2: Copy `node_modules` and source code to a `node:alpine` image.

### Challenge 2: CA Certificates (Difficulty: ⭐⭐)

**Task:**
If your Go app makes HTTPS requests (e.g., to Google), it will fail in `scratch` because `scratch` has no Root CA certificates.
Fix this by copying `/etc/ssl/certs/ca-certificates.crt` from the builder stage to the scratch stage.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```dockerfile
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
```
</details>

---

## 🔑 Key Takeaways

1.  **Static Binaries**: Languages like Go and Rust are perfect for containers because they compile to a single file.
2.  **Attack Surface**: A `scratch` container has no shell (`/bin/sh`). Hackers can't run commands even if they exploit your app.
3.  **Efficiency**: Smaller images pull faster and scale faster.

---

## ⏭️ Next Steps

We made it small. Now let's make it secure.

Proceed to **Lab 11.2: Docker Security & Distroless**.
