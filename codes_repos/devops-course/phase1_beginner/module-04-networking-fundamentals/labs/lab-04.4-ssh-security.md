# Lab 4.4: SSH & Secure Access

## 🎯 Objective

Master Secure Shell (SSH). Move beyond passwords to Key-Based Authentication, understand `ssh_config`, and learn how to tunnel traffic securely (Port Forwarding).

## 📋 Prerequisites

-   Two machines (or a VM/Container) to connect between.
-   OpenSSH Client & Server installed.

## 📚 Background

### How SSH Works
1.  **Encryption**: Traffic is encrypted (no one can snoop).
2.  **Authentication**: Passwords or **Public Key Cryptography**.
    -   **Private Key**: Stored on your laptop. NEVER SHARE.
    -   **Public Key**: Stored on the server (`authorized_keys`).
    -   *Analogy*: The Public Key is the lock. The Private Key is the key. You put the lock on the server door.

---

## 🔨 Hands-On Implementation

### Part 1: Key-Based Authentication 🔑

1.  **Generate Keys (If not done in Lab 2.9):**
    ```bash
    ssh-keygen -t ed25519 -C "lab-key"
    ```
    *Note:* `ed25519` is newer and more secure than `rsa`.

2.  **Copy ID to Server:**
    ```bash
    ssh-copy-id user@remote-server
    ```
    *If you don't have `ssh-copy-id` (Windows):*
    ```powershell
    cat ~/.ssh/id_ed25519.pub | ssh user@remote-server "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
    ```

3.  **Login:**
    ```bash
    ssh user@remote-server
    ```
    *Result:* No password asked!

### Part 2: The Config File (`~/.ssh/config`) 📝

Typing `ssh user@192.168.1.50 -i ~/.ssh/my_key` is annoying.

1.  **Create Config:**
    ```bash
    nano ~/.ssh/config
    ```

2.  **Add Entry:**
    ```text
    Host myserver
        HostName 192.168.1.50
        User student
        IdentityFile ~/.ssh/id_ed25519
    ```

3.  **Connect:**
    ```bash
    ssh myserver
    ```

### Part 3: SSH Tunnels (Port Forwarding) 🚇

**Scenario:** A database is running on the server on port 3306 (MySQL). It is blocked by a firewall (only localhost access). You want to access it from your laptop.

1.  **Local Port Forwarding (`-L`):**
    ```bash
    ssh -L 9000:localhost:3306 myserver
    ```
    *Syntax:* `-L local_port:destination_host:destination_port`.

2.  **Access:**
    Connect your local database client to `localhost:9000`. SSH forwards the traffic to the server's `localhost:3306`.

---

## 🎯 Challenges

### Challenge 1: Disable Password Auth (Difficulty: ⭐⭐⭐)

**Task:**
Secure your server by disabling password login entirely.
1.  Edit `/etc/ssh/sshd_config` on the server.
2.  Set `PasswordAuthentication no`.
3.  Restart SSH (`sudo systemctl restart sshd`).
4.  Try to log in from a machine *without* your key. It should fail immediately.

### Challenge 2: Run a Command Remotely (Difficulty: ⭐⭐)

**Task:**
Check disk space on the remote server without logging in interactively.
*Hint: `ssh myserver <command>`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
(See steps in task). **Warning:** Make sure your key works before doing this, or you lock yourself out!

**Challenge 2:**
```bash
ssh myserver "df -h"
```
</details>

---

## 🔑 Key Takeaways

1.  **Keys > Passwords**: Keys are harder to brute force.
2.  **Config File**: Saves typing and organizes your servers.
3.  **Tunnels**: A DevOps superpower. Access private resources securely without VPNs.

---

## ⏭️ Next Steps

We have secured the connection. Now let's protect the server itself.

Proceed to **Lab 4.5: Firewalls & Security Groups**.
