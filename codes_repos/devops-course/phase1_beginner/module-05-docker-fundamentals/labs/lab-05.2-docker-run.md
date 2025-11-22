# Lab 5.2: Docker Run & Basic Commands

## 🎯 Objective

Master the `docker run` command. Learn how to manage container lifecycle (start, stop, remove) and interact with running containers.

## 📋 Prerequisites

-   Docker installed.

## 📚 Background

### The Lifecycle
`Image` -> `run` -> `Container` (Running) -> `stop` -> `Container` (Stopped) -> `rm` -> `Gone`.

### Key Flags
-   `-d`: Detached (Background).
-   `-it`: Interactive TTY (Shell).
-   `--name`: Give it a name.
-   `--rm`: Remove automatically when stopped.
-   `-p`: Port mapping.

---

## 🔨 Hands-On Implementation

### Part 1: Foreground vs Background 🏃‍♂️

1.  **Interactive (Foreground):**
    ```bash
    docker run ubuntu echo "Hello"
    ```
    *Result:* Prints "Hello" and exits.

2.  **Interactive Shell (`-it`):**
    ```bash
    docker run -it ubuntu bash
    ```
    *Result:* You are inside the container. Type `ls`. Type `exit` to leave.

3.  **Detached (`-d`):**
    ```bash
    docker run -d --name sleeper ubuntu sleep 1000
    ```
    *Result:* Prints a long ID. Container runs in background.

### Part 2: Managing Containers 🎮

1.  **List Running:**
    ```bash
    docker ps
    ```

2.  **List All (Including Stopped):**
    ```bash
    docker ps -a
    ```

3.  **Stop:**
    ```bash
    docker stop sleeper
    ```

4.  **Start:**
    ```bash
    docker start sleeper
    ```

5.  **Remove:**
    ```bash
    docker stop sleeper
    docker rm sleeper
    ```
    *Note:* You cannot remove a running container without `-f` (force).

### Part 3: Executing Commands Inside (`exec`) 💉

**Scenario:** You have a web server running in background. You want to debug it.

1.  **Start Nginx:**
    ```bash
    docker run -d --name myweb nginx
    ```

2.  **Enter the container:**
    ```bash
    docker exec -it myweb bash
    ```
    *Note:* `run` creates a *new* container. `exec` enters an *existing* one.

3.  **Modify file:**
    ```bash
    echo "Hacked" > /usr/share/nginx/html/index.html
    exit
    ```

---

## 🎯 Challenges

### Challenge 1: The Cleanup (Difficulty: ⭐⭐)

**Task:**
You have 10 stopped containers cluttering your system.
Find the single command to remove ALL stopped containers.
*Hint: `docker system ...` or `docker container ...`*

### Challenge 2: Port Conflict (Difficulty: ⭐⭐)

**Task:**
1.  Run Nginx on port 8080 (`-p 8080:80`).
2.  Try to run another Nginx on port 8080.
3.  Observe the error.
4.  Fix it by running the second one on port 8081.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
docker container prune
# OR
docker rm $(docker ps -aq)
```

**Challenge 2:**
Error: `Bind for 0.0.0.0:8080 failed: port is already allocated`.
Fix:
```bash
docker run -d -p 8081:80 nginx
```
</details>

---

## 🔑 Key Takeaways

1.  **Cattle, not Pets**: Don't be afraid to `rm` containers. You can always `run` a fresh one.
2.  **Exec vs Run**: Use `run` to start. Use `exec` to debug.
3.  **Naming**: Always use `--name`. Random names like `romantic_curie` are hard to manage in scripts.

---

## ⏭️ Next Steps

We can run images. Now let's build our own.

Proceed to **Lab 5.3: Dockerfiles & Building Images**.
