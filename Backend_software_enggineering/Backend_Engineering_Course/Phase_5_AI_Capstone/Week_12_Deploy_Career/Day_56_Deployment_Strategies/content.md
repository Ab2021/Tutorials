# Day 56: Deployment Strategies

## 1. It Works on My Machine... Now What?

Deployment is moving code from your laptop to a server where users can reach it.

### 1.1 The Options
1.  **Virtual Machine (VM)**: AWS EC2, DigitalOcean Droplet.
    *   *Pros*: Full control.
    *   *Cons*: You manage OS updates, security patching.
2.  **Containers (Docker)**: AWS ECS, Kubernetes.
    *   *Pros*: Consistent environment. Scalable.
    *   *Cons*: Complexity (K8s is hard).
3.  **Serverless**: AWS Lambda, Google Cloud Run.
    *   *Pros*: No servers to manage. Pay per request.
    *   *Cons*: Cold starts. Vendor lock-in.

---

## 2. The Reverse Proxy (Nginx)

Never expose your Python/Node app directly to the internet (Port 8000).
Always put a **Reverse Proxy** in front.
*   **SSL Termination**: Handles HTTPS.
*   **Static Files**: Serves images/CSS faster than Python.
*   **Load Balancing**: Distributes traffic to multiple app instances.

---

## 3. Docker for Production

*   **Multi-Stage Builds**: Keep image size small.
    *   *Build Stage*: Install compilers, build deps.
    *   *Run Stage*: Copy only the binary/artifacts.
*   **Environment Variables**: Never hardcode secrets. Use `.env` or Secret Manager.

---

## 4. Summary

Today we went live.
*   **Cloud**: Renting someone else's computer.
*   **Nginx**: The bouncer at the door.
*   **Docker**: The shipping container.

**Tomorrow (Day 57)**: We automate this process with **CI/CD**.
