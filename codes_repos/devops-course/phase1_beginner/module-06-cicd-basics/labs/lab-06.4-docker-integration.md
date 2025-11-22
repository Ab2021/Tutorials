# Lab 6.4: Docker Integration in CI

## 🎯 Objective

Automate the Build and Push of Docker images. You will configure GitHub Actions to build your Dockerfile and push the image to Docker Hub securely using Secrets.

## 📋 Prerequisites

-   Completed Lab 6.3.
-   Docker Hub Account.

## 📚 Background

### Secrets Management
**NEVER** put passwords in `main.yml`.
GitHub provides **Secrets** (Settings -> Secrets and variables -> Actions).
We will store `DOCKER_USERNAME` and `DOCKER_PASSWORD` there.

### The Workflow
`Push` -> `Checkout` -> `Login to Docker Hub` -> `Build Image` -> `Push Image`.

---

## 🔨 Hands-On Implementation

### Part 1: Configure Secrets 🔐

1.  **Go to GitHub Repo Settings.**
2.  **Secrets and variables** -> **Actions**.
3.  **New Repository Secret**:
    -   Name: `DOCKER_USERNAME`
    -   Value: `your_dockerhub_username`
4.  **New Repository Secret**:
    -   Name: `DOCKER_PASSWORD`
    -   Value: `your_dockerhub_password` (Or Access Token - recommended).

### Part 2: The Docker Workflow 🐳

1.  **Create `.github/workflows/docker.yml`:**
    ```yaml
    name: Build and Push Docker Image

    on:
      push:
        branches: [ "main" ]

    jobs:
      docker:
        runs-on: ubuntu-latest
        steps:
          - name: Checkout
            uses: actions/checkout@v2

          - name: Login to Docker Hub
            uses: docker/login-action@v1
            with:
              username: ${{ secrets.DOCKER_USERNAME }}
              password: ${{ secrets.DOCKER_PASSWORD }}

          - name: Build and Push
            uses: docker/build-push-action@v2
            with:
              context: .
              push: true
              tags: ${{ secrets.DOCKER_USERNAME }}/ci-demo:latest
    ```

2.  **Add a Dockerfile (if missing):**
    Use the one from Lab 5.3 or create a simple one.
    ```dockerfile
    FROM alpine
    CMD ["echo", "Hello from CI"]
    ```

3.  **Push:**
    ```bash
    git add .
    git commit -m "Add Docker pipeline"
    git push
    ```

### Part 3: Verify 🕵️‍♂️

1.  **Check GitHub Actions:**
    Wait for green checkmark.
2.  **Check Docker Hub:**
    Go to your profile. You should see `ci-demo` updated "a few seconds ago".
3.  **Pull locally:**
    ```bash
    docker pull <YOUR_USERNAME>/ci-demo:latest
    docker run <YOUR_USERNAME>/ci-demo:latest
    ```

---

## 🎯 Challenges

### Challenge 1: Dynamic Tagging (Difficulty: ⭐⭐⭐)

**Task:**
Instead of just `latest`, tag the image with the Git Commit SHA (`sha-12345`).
*Why?* So you can roll back to a specific commit.
*Hint: Use `${{ github.sha }}`.*

### Challenge 2: Scan Image in CI (Difficulty: ⭐⭐⭐)

**Task:**
Add a step to scan the image for vulnerabilities using Trivy or Docker Scan *before* pushing.
*Hint: `uses: aquasecurity/trivy-action@master`.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
tags: |
  ${{ secrets.DOCKER_USERNAME }}/ci-demo:latest
  ${{ secrets.DOCKER_USERNAME }}/ci-demo:${{ github.sha }}
```

**Challenge 2:**
```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: '${{ secrets.DOCKER_USERNAME }}/ci-demo:latest'
    format: 'table'
    exit-code: '1' # Fail pipeline if critical bugs found
    ignore-unfixed: true
    severity: 'CRITICAL,HIGH'
```
</details>

---

## 🔑 Key Takeaways

1.  **Secrets**: This is the standard way to handle credentials in CI.
2.  **Automation**: No more "I forgot to push the image".
3.  **Traceability**: By tagging with Commit SHA, you know exactly which code produced which image.

---

## ⏭️ Next Steps

We have built and pushed. Now let's deploy.

Proceed to **Lab 6.5: Continuous Deployment (CD)**.
