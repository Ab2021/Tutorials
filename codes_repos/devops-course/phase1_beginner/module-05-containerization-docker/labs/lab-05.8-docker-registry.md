# Lab 5.8: Docker Registry & Hub

## 🎯 Objective

Learn how to publish your images. You will push your image to Docker Hub (Public) and run a local Private Registry.

## 📋 Prerequisites

-   Docker Hub Account (Free).
-   `docker login` completed.

## 📚 Background

### The Registry
A server that stores Docker Images.
-   **Docker Hub**: The default public registry.
-   **ECR/GCR/ACR**: Cloud provider registries (AWS/Google/Azure).
-   **Private Registry**: Self-hosted.

### Naming Convention
`registry.example.com/username/image:tag`
-   If registry is missing -> `docker.io` (Hub).
-   If tag is missing -> `latest`.

---

## 🔨 Hands-On Implementation

### Part 1: Pushing to Docker Hub ☁️

1.  **Tag the Image:**
    You must prefix the image with your username.
    ```bash
    docker tag good-image <YOUR_USERNAME>/my-app:v1.0
    ```

2.  **Push:**
    ```bash
    docker push <YOUR_USERNAME>/my-app:v1.0
    ```

3.  **Verify:**
    Go to `hub.docker.com`. You should see your repo.

4.  **Pull from anywhere:**
    Delete local image and pull it back.
    ```bash
    docker rmi <YOUR_USERNAME>/my-app:v1.0
    docker run <YOUR_USERNAME>/my-app:v1.0
    ```

### Part 2: Running a Private Registry 🏠

**Scenario:** You have proprietary code you don't want on public Docker Hub.

1.  **Start Registry:**
    Docker has an official image for this!
    ```bash
    docker run -d -p 5000:5000 --name registry registry:2
    ```

2.  **Tag for Localhost:**
    ```bash
    docker tag good-image localhost:5000/my-secret-app:v1
    ```

3.  **Push:**
    ```bash
    docker push localhost:5000/my-secret-app:v1
    ```

4.  **Verify:**
    Check the catalog API.
    ```bash
    curl http://localhost:5000/v2/_catalog
    ```
    *Output:* `{"repositories":["my-secret-app"]}`

---

## 🎯 Challenges

### Challenge 1: Retagging (Difficulty: ⭐)

**Task:**
Retag your `v1.0` image as `latest` and push it.
*Why?* Users expect `latest` to work.

### Challenge 2: Insecure Registries (Difficulty: ⭐⭐⭐)

**Scenario:** You try to push to a remote private registry (not localhost) without HTTPS. Docker blocks it.
**Task:**
Find out how to configure `daemon.json` to allow "insecure-registries".

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
docker tag <USER>/my-app:v1.0 <USER>/my-app:latest
docker push <USER>/my-app:latest
```

**Challenge 2:**
Edit `/etc/docker/daemon.json`:
```json
{
  "insecure-registries" : ["myregistrydomain.com:5000"]
}
```
Restart Docker.
</details>

---

## 🔑 Key Takeaways

1.  **Tags are Mutable**: You can overwrite `v1.0`. This is bad practice. Use SHA digest if you need 100% certainty.
2.  **Authentication**: `docker login` saves credentials in `~/.docker/config.json`. Keep this file safe.
3.  **Cleanup**: Registries get full. You need a policy to delete old images.

---

## ⏭️ Next Steps

We can store images. Now let's debug them when they crash.

Proceed to **Lab 5.9: Debugging Containers**.
