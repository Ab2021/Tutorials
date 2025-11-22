# Lab 4.6: Load Balancing Basics (Nginx)

## 🎯 Objective

Understand how to distribute traffic across multiple servers. You will configure Nginx as a Layer 7 Load Balancer to split traffic between two "backend" applications.

## 📋 Prerequisites

-   Docker installed (Lab 1.4).
-   Basic Nginx knowledge (we will learn as we go).

## 📚 Background

### Why Load Balance?
1.  **Scale**: One server can't handle 1 million users.
2.  **Availability**: If one server crashes, the other takes over.

**Algorithms:**
-   **Round Robin**: A -> B -> A -> B.
-   **Least Connections**: Send to the server with fewest active users.
-   **IP Hash**: User A always goes to Server A (Sticky Sessions).

---

## 🔨 Hands-On Implementation

### Part 1: Setup Backends 🐳

We need two "servers". We will use Docker containers.

1.  **Start Backend 1:**
    ```bash
    docker run -d --rm --name app1 -p 8001:80 kennethreitz/httpbin
    ```

2.  **Start Backend 2:**
    ```bash
    docker run -d --rm --name app2 -p 8002:80 kennethreitz/httpbin
    ```

3.  **Verify:**
    Visit `http://localhost:8001` and `http://localhost:8002`.

### Part 2: Configure Nginx LB ⚙️

1.  **Create `nginx.conf`:**
    ```nginx
    events {}
    
    http {
        upstream myapp {
            server host.docker.internal:8001;
            server host.docker.internal:8002;
        }
    
        server {
            listen 80;
    
            location / {
                proxy_pass http://myapp;
            }
        }
    }
    ```
    *Note:* `host.docker.internal` allows the container to talk to your laptop's ports. On Linux, you might need to use your actual LAN IP (e.g., 192.168.x.x).

### Part 3: Run Load Balancer 🚀

1.  **Start Nginx:**
    ```bash
    docker run -d --rm --name lb -p 8080:80 -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro nginx
    ```

2.  **Test:**
    Visit `http://localhost:8080/get`.
    Refresh multiple times.
    *Observation:* Nginx forwards requests to 8001 and 8002 in a Round Robin fashion.

### Part 4: Simulate Failure 💥

1.  **Kill App 1:**
    ```bash
    docker stop app1
    ```

2.  **Test:**
    Refresh `http://localhost:8080/get`.
    *Result:* It still works! Nginx detects App 1 is down and sends everything to App 2.

---

## 🎯 Challenges

### Challenge 1: Weighted Balancing (Difficulty: ⭐⭐)

**Scenario:** App 2 is a powerful server (64GB RAM). App 1 is weak (4GB RAM).
**Task:**
Configure Nginx so App 2 gets 3x more traffic than App 1.
*Hint: `server ... weight=3;`*

### Challenge 2: Health Checks (Difficulty: ⭐⭐⭐)

**Task:**
Research Nginx Plus (or Open Source workarounds) for "Active Health Checks".
How does Nginx know App 1 is down? (Passive vs Active).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```nginx
upstream myapp {
    server host.docker.internal:8001 weight=1;
    server host.docker.internal:8002 weight=3;
}
```

**Challenge 2:**
Open Source Nginx uses **Passive Health Checks**. If a request fails, it marks the server as down for a short time.
Nginx Plus (Paid) supports **Active Health Checks** (periodically pinging `/health` to verify status).
</details>

---

## 🔑 Key Takeaways

1.  **Reverse Proxy**: Nginx acts as a Reverse Proxy. The client talks to Nginx; Nginx talks to the App.
2.  **Single Point of Failure**: If Nginx dies, everything dies. (Solution: HAProxy with Keepalived).
3.  **SSL Termination**: Usually, you put your SSL certificates on the Load Balancer to offload work from the apps.

---

## ⏭️ Next Steps

We have networking basics down. Let's wrap up with some troubleshooting tools.

Proceed to **Lab 4.7: Network Troubleshooting Tools**.
