# Lab 5.5: Docker Volumes & Storage

## 🎯 Objective

Learn how to persist data. By default, when a container dies, its data dies with it. You will use **Volumes** and **Bind Mounts** to keep data safe.

## 📋 Prerequisites

-   Completed Lab 5.4.

## 📚 Background

### The Problem
Containers are **Ephemeral** (temporary).
If you run a Database in a container and delete the container, the database files are gone.

### The Solution
1.  **Volumes**: Managed by Docker. Stored in `/var/lib/docker/volumes`. Best for persistence.
2.  **Bind Mounts**: Maps a folder on your Host to the Container. Best for development (live code editing).
3.  **tmpfs**: Stored in RAM. Fast, but lost on reboot.

---

## 🔨 Hands-On Implementation

### Part 1: The Data Loss Simulation 💥

1.  **Run Redis:**
    ```bash
    docker run -d --name redis1 redis
    ```

2.  **Write Data:**
    ```bash
    docker exec redis1 redis-cli set mykey "DevOps is Cool"
    ```

3.  **Destroy Container:**
    ```bash
    docker rm -f redis1
    ```

4.  **Run New Redis:**
    ```bash
    docker run -d --name redis2 redis
    ```

5.  **Check Data:**
    ```bash
    docker exec redis2 redis-cli get mykey
    ```
    *Result:* `(nil)`. Data is lost.

### Part 2: Using Volumes (Persistence) 💾

1.  **Create Volume:**
    ```bash
    docker volume create my-data
    ```

2.  **Run Redis with Volume:**
    ```bash
    docker run -d --name redis3 -v my-data:/data redis
    ```
    *Note:* Redis stores data in `/data`. We map `my-data` volume to `/data`.

3.  **Write Data:**
    ```bash
    docker exec redis3 redis-cli set mykey "Persistent Data"
    ```

4.  **Destroy & Recreate:**
    ```bash
    docker rm -f redis3
    docker run -d --name redis4 -v my-data:/data redis
    ```

5.  **Check Data:**
    ```bash
    docker exec redis4 redis-cli get mykey
    ```
    *Result:* `"Persistent Data"`. It survived!

### Part 3: Bind Mounts (Development) 💻

**Scenario:** You want to edit `index.html` on your laptop and see it update in Nginx instantly.

1.  **Create file:**
    ```bash
    echo "<h1>Version 1</h1>" > index.html
    ```

2.  **Run Nginx with Mount:**
    ```bash
    docker run -d -p 8080:80 -v $(pwd)/index.html:/usr/share/nginx/html/index.html nginx
    ```

3.  **Check Browser:**
    `http://localhost:8080` -> "Version 1".

4.  **Edit File (Host):**
    ```bash
    echo "<h1>Version 2</h1>" > index.html
    ```

5.  **Check Browser:**
    Refresh. -> "Version 2".
    *Note:* No rebuild or restart needed!

---

## 🎯 Challenges

### Challenge 1: Inspecting Volumes (Difficulty: ⭐⭐)

**Task:**
Find out *where* on your hard drive the `my-data` volume is actually stored.
*Hint: `docker volume inspect ...`*

### Challenge 2: Read-Only Mounts (Difficulty: ⭐⭐⭐)

**Task:**
Run a container that mounts `index.html` but prevents the container from modifying it.
*Hint: Add `:ro` to the volume flag.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
docker volume inspect my-data
```
*Linux:* `/var/lib/docker/volumes/my-data/_data`
*Mac/Windows:* It's inside the Docker VM, so you can't see it directly on your OS without tricks.

**Challenge 2:**
```bash
docker run -d -v $(pwd)/index.html:/target/index.html:ro nginx
```
If you try to `echo "test" > /target/index.html` inside the container, it will fail ("Read-only file system").
</details>

---

## 🔑 Key Takeaways

1.  **Databases need Volumes**: Never run a DB container without a volume.
2.  **Bind Mounts for Code**: Use them during development to avoid rebuilding images for every comma change.
3.  **Permissions**: Bind mounts can have permission issues (User ID on host vs container).

---

## ⏭️ Next Steps

Running one container is easy. Running 5 connected containers? We need a tool.

Proceed to **Lab 5.6: Docker Compose**.
