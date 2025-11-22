# Lab 5.1: Introduction to Containers

## 🎯 Objective

Understand what a container actually is. We will demystify the "magic" by manually creating a container-like environment using Linux namespaces (simulated) and then comparing it to a VM.

## 📋 Prerequisites

-   Linux Terminal (or WSL).
-   Docker installed.

## 📚 Background

### VM vs Container
-   **Virtual Machine (VM)**: Hardware virtualization. Has a full OS kernel. Heavy (GBs). Slow boot (minutes).
-   **Container**: OS virtualization. Shares the host kernel. Lightweight (MBs). Fast boot (milliseconds).

### How it works (The Magic)
1.  **Namespaces**: Isolation. "I can only see my own processes/files."
2.  **Cgroups**: Resource Control. "I can only use 512MB RAM."
3.  **Union Filesystem**: Layering. "I am built on top of Ubuntu."

---

## 🔨 Hands-On Implementation

### Part 1: The "Works on My Machine" Problem 😫

**Scenario:**
1.  Dev: "It works on my laptop (Python 3.9)."
2.  Ops: "It crashed on prod (Python 3.6)."
3.  **Solution**: Ship the Python version *with* the app. That's a container.

### Part 2: Exploring Namespaces (Simulation) 📦

We will use `unshare` to create a new namespace (if on Linux).

1.  **Check current PID:**
    ```bash
    echo $$
    ```

2.  **Start a new Process Namespace:**
    ```bash
    # Requires sudo
    sudo unshare --fork --pid --mount-proc /bin/bash
    ```

3.  **Check PIDs inside:**
    ```bash
    ps aux
    ```
    *Result:* You only see `bash` and `ps`. PID 1 is your bash. You are isolated!
    *Exit:* Type `exit`.

### Part 3: Docker Hello World 🐳

1.  **Run it:**
    ```bash
    docker run hello-world
    ```

2.  **Analyze Output:**
    -   "Unable to find image... locally" -> **Pulling** from Registry.
    -   "Hello from Docker!" -> **Running** the container.
    -   Exits immediately.

3.  **Check Image:**
    ```bash
    docker images
    ```
    *Size:* ~13kB. Tiny!

---

## 🎯 Challenges

### Challenge 1: Kernel Sharing (Difficulty: ⭐⭐)

**Task:**
1.  Run `uname -r` on your host.
2.  Run `docker run alpine uname -r`.
3.  Compare the outputs.
    *Question:* Why are they identical?

### Challenge 2: Isolation Check (Difficulty: ⭐⭐)

**Task:**
1.  Create a file `host.txt` on your desktop.
2.  Run `docker run -it alpine sh`.
3.  Try to find `host.txt` inside the container.
    *Result:* You can't. It's a different filesystem.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
They are identical because **Containers share the Host Kernel**. Unlike a VM, the container does not have its own kernel.

**Challenge 2:**
The container has its own isolated filesystem (RootFS). It cannot see the host's files unless you explicitly Mount them (Volume).
</details>

---

## 🔑 Key Takeaways

1.  **Containers are Processes**: They are just normal Linux processes with a mask on (Namespaces).
2.  **Immutability**: Once built, a container image doesn't change. This guarantees consistency.
3.  **Efficiency**: You can run 100 containers on a laptop. You can't run 100 VMs.

---

## ⏭️ Next Steps

We ran a pre-made container. Now let's control it.

Proceed to **Lab 5.2: Docker Run & Basic Commands**.
