# Lab 5.4: Docker Networking

## 🎯 Objective

Understand how containers talk to each other and the outside world. You will explore the default Bridge network, Host networking, and create custom networks for DNS resolution.

## 📋 Prerequisites

-   Completed Lab 5.3.

## 📚 Background

### Network Drivers
1.  **Bridge (Default)**: Containers get a private IP (e.g., `172.17.0.2`). They can talk to each other if they know the IP.
2.  **Host**: Container shares the host's network stack. No isolation.
3.  **None**: No network.
4.  **Overlay**: For multi-host (Swarm/Kubernetes).

### Service Discovery (DNS)
On the default bridge, you must use IPs. On **Custom Bridges**, you can use **Container Names** as hostnames.

---

## 🔨 Hands-On Implementation

### Part 1: The Default Bridge 🌉

1.  **Start two containers:**
    ```bash
    docker run -d --name c1 alpine sleep 1000
    docker run -d --name c2 alpine sleep 1000
    ```

2.  **Get IP of c1:**
    ```bash
    docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' c1
    ```
    *Example:* `172.17.0.2`

3.  **Ping from c2:**
    ```bash
    docker exec -it c2 ping 172.17.0.2
    ```
    *Result:* Works.

4.  **Ping by Name:**
    ```bash
    docker exec -it c2 ping c1
    ```
    *Result:* **Fails!** "bad address 'c1'".

### Part 2: Custom Bridge (DNS Magic) 🪄

1.  **Create Network:**
    ```bash
    docker network create mynet
    ```

2.  **Start containers in network:**
    ```bash
    docker run -d --name c3 --net mynet alpine sleep 1000
    docker run -d --name c4 --net mynet alpine sleep 1000
    ```

3.  **Ping by Name:**
    ```bash
    docker exec -it c4 ping c3
    ```
    *Result:* **Works!** Docker's embedded DNS resolves `c3` to its IP.

### Part 3: Host Networking 🏠

1.  **Run Nginx on Host Network:**
    ```bash
    docker run -d --name host-nginx --net host nginx
    ```

2.  **Test:**
    Visit `http://localhost`.
    *Note:* You didn't use `-p 80:80`. It just grabbed port 80 on your machine directly.
    *Limitation:* Only works on Linux (not Docker Desktop for Mac/Windows).

---

## 🎯 Challenges

### Challenge 1: Connecting Networks (Difficulty: ⭐⭐⭐)

**Scenario:** `c1` is on Default Bridge. `c3` is on `mynet`. They cannot talk.
**Task:**
Connect `c1` to `mynet` *without* restarting it.
*Hint: `docker network connect ...`*

### Challenge 2: The Legacy Link (Difficulty: ⭐⭐)

**Task:**
Research the `--link` flag.
Why is it deprecated? What replaced it? (Hint: We just did it in Part 2).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
docker network connect mynet c1
docker exec -it c1 ping c3
```
*Result:* Works. `c1` now has two IPs (one in each network).

**Challenge 2:**
`--link` was the old way to allow containers to talk. It was fragile (if container restarted, link broke). It is replaced by **User-defined Networks** (Custom Bridges) which provide automatic DNS resolution.
</details>

---

## 🔑 Key Takeaways

1.  **Always use Custom Networks**: The default bridge is limited (no DNS).
2.  **Isolation**: Networks provide security. Database network should not be accessible from the public.
3.  **Host Mode**: Good for performance (no NAT), but creates port conflicts.

---

## ⏭️ Next Steps

We have networking. Now let's handle data persistence.

Proceed to **Lab 5.5: Docker Volumes & Storage**.
