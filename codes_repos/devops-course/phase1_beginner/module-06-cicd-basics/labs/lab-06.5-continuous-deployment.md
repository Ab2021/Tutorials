# Lab 6.5: Continuous Deployment (CD)

## 🎯 Objective

The final mile. Automate the deployment of your application. You will learn how to use GitHub Actions to SSH into a remote server, pull the latest Docker image, and restart the container.

## 📋 Prerequisites

-   Completed Lab 6.4.
-   A Linux Server (VM, VPS, or even your laptop if reachable via SSH).
-   *Alternative:* If you don't have a server, we will use a "Simulation" mode.

## 📚 Background

### CD Patterns
1.  **Push-Based (Traditional)**: CI Server (GitHub) SSHs into Prod Server and runs commands.
2.  **Pull-Based (GitOps)**: Prod Server (ArgoCD) watches Git and pulls changes. (Advanced - Module 12).

We will focus on **Push-Based** as it's the easiest to start with.

---

## 🔨 Hands-On Implementation

### Part 1: Prepare the Server (Real Mode) 🖥️

*Skip to Part 2 if you are doing Simulation Mode.*

1.  **Generate SSH Key Pair:**
    On your local machine:
    ```bash
    ssh-keygen -t ed25519 -f deploy_key -C "github-actions"
    ```
    *Do not add a passphrase.*

2.  **Add Public Key to Server:**
    Copy content of `deploy_key.pub` to `~/.ssh/authorized_keys` on the remote server.

3.  **Add Private Key to GitHub Secrets:**
    -   Name: `SSH_PRIVATE_KEY`
    -   Value: Content of `deploy_key`.
    -   Also add `SSH_HOST` (IP) and `SSH_USER` (username).

### Part 2: The Deployment Workflow 🚀

1.  **Create `.github/workflows/deploy.yml`:**

    ```yaml
    name: CD Pipeline

    on:
      push:
        branches: [ "main" ]

    jobs:
      build:
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
              tags: ${{ secrets.DOCKER_USERNAME }}/app:latest

      deploy:
        needs: build
        runs-on: ubuntu-latest
        steps:
          - name: Deploy via SSH
            uses: appleboy/ssh-action@master
            with:
              host: ${{ secrets.SSH_HOST }}
              username: ${{ secrets.SSH_USER }}
              key: ${{ secrets.SSH_PRIVATE_KEY }}
              script: |
                docker pull ${{ secrets.DOCKER_USERNAME }}/app:latest
                docker stop myapp || true
                docker rm myapp || true
                docker run -d --name myapp -p 80:80 ${{ secrets.DOCKER_USERNAME }}/app:latest
    ```

### Part 3: Simulation Mode (No Server Required) 🎭

If you don't have a server, we will simulate the deployment by printing to the console.

1.  **Modify the `deploy` job:**
    ```yaml
      deploy:
        needs: build
        runs-on: ubuntu-latest
        steps:
          - name: Simulate Deployment
            run: |
              echo "Connecting to Production Server..."
              sleep 2
              echo "Pulling image ${{ secrets.DOCKER_USERNAME }}/app:latest..."
              sleep 2
              echo "Restarting Container..."
              sleep 1
              echo "✅ Deployment Successful!"
    ```

2.  **Push and Watch:**
    Go to GitHub Actions. Watch the logs. It feels real!

---

## 🎯 Challenges

### Challenge 1: Staging vs Production (Difficulty: ⭐⭐⭐)

**Task:**
Create two environments.
1.  Push to `develop` branch -> Deploys to Staging Server (Simulation).
2.  Push to `main` branch -> Deploys to Production Server.
*Hint: Use `if: github.ref == 'refs/heads/main'`.*

### Challenge 2: Rollback Strategy (Difficulty: ⭐⭐⭐⭐)

**Scenario:** The deployment succeeded, but the app is crashing.
**Task:**
Design a workflow (manual trigger) that takes a `tag` input and redeploys that specific version.
*Hint: `on: workflow_dispatch: inputs: tag: ...`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    ...
  deploy-prod:
    if: github.ref == 'refs/heads/main'
    ...
```

**Challenge 2:**
```yaml
on:
  workflow_dispatch:
    inputs:
      tag:
        description: 'Tag to rollback to'
        required: true
        default: 'latest'

jobs:
  rollback:
    steps:
      - uses: appleboy/ssh-action@master
        with:
          script: |
            docker run -d ... ${{ secrets.DOCKER_USERNAME }}/app:${{ github.event.inputs.tag }}
```
</details>

---

## 🔑 Key Takeaways

1.  **SSH Keys**: Manage them carefully. If `SSH_PRIVATE_KEY` leaks, your server is compromised.
2.  **Zero Downtime**: The simple `stop` -> `run` method causes downtime (seconds). Advanced patterns (Blue/Green) avoid this.
3.  **Idempotency**: Your deployment script should work whether the container exists or not (`|| true`).

---

## ⏭️ Next Steps

We have the pieces. Let's build the full highway.

Proceed to **Lab 6.6: CI/CD Capstone Project**.
