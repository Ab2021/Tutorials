# Lab 5.9: Debugging Containers

## 🎯 Objective

Learn how to fix broken containers. You will troubleshoot common errors like "CrashLoopBackOff", "Connection Refused", and "File Not Found".

## 📋 Prerequisites

-   Completed Lab 5.8.

## 📚 Background

### The Toolkit
1.  `docker logs`: See stdout/stderr.
2.  `docker inspect`: See configuration (IP, Env Vars, Mounts).
3.  `docker exec`: Go inside.
4.  `docker events`: Real-time stream of what Docker daemon is doing.

---

## 🔨 Hands-On Implementation

### Part 1: The Crashing Container 💥

1.  **Run a broken container:**
    ```bash
    docker run -d --name crasher alpine sh -c "sleep 2; exit 1"
    ```

2.  **Check Status:**
    ```bash
    docker ps -a
    ```
    *Status:* `Exited (1)`.

3.  **Check Logs:**
    ```bash
    docker logs crasher
    ```
    *Result:* Empty (because `sleep` doesn't print anything).

4.  **Debug Strategy:**
    If logs are empty, check the command.
    ```bash
    docker inspect crasher | grep Cmd
    ```

### Part 2: The "Connection Refused" 🚫

1.  **Run Nginx on wrong port:**
    ```bash
    docker run -d --name web -p 8080:8080 nginx
    ```
    *Note:* Nginx listens on 80 by default, but we mapped host 8080 to container 8080.

2.  **Test:**
    `curl localhost:8080` -> `Connection reset by peer` or empty reply.

3.  **Debug:**
    Go inside and check listening ports.
    ```bash
    docker exec -it web bash
    # Inside container
    apt-get update && apt-get install -y net-tools
    netstat -tuln
    ```
    *Result:* Nginx is listening on `:80`.
    *Fix:* Re-run with `-p 8080:80`.

### Part 3: The Missing File 📁

1.  **Run with Volume:**
    ```bash
    docker run -d --name vol-test -v $(pwd)/missing.conf:/app/config.conf nginx
    ```
    *Result:* Docker creates a **Directory** named `missing.conf` on host if it doesn't exist!
    *Error:* Nginx crashes because it expects a file, not a directory.

2.  **Inspect:**
    ```bash
    ls -l missing.conf
    ```
    *Result:* `drwxr-xr-x` (Directory).

---

## 🎯 Challenges

### Challenge 1: Inspecting IP (Difficulty: ⭐)

**Task:**
Find the internal IP address of a running container using `docker inspect` and `grep`.

### Challenge 2: Viewing Resource Usage (Difficulty: ⭐⭐)

**Task:**
Use `docker stats` to see how much CPU/RAM your containers are using in real-time.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_name>
```

**Challenge 2:**
```bash
docker stats
```
</details>

---

## 🔑 Key Takeaways

1.  **Logs First**: Always check `docker logs` first.
2.  **Inspect Second**: Check if Env Vars or Mounts are correct.
3.  **Exec Last**: If you have to SSH into a container to fix it, you are doing it wrong (Anti-pattern). Fix the Dockerfile instead.

---

## ⏭️ Next Steps

We can fix anything. Now let's build the final project.

Proceed to **Lab 5.10: Docker Capstone Project**.
