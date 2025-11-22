# Lab 2.2: Linux File System Hierarchy

## 🎯 Objective

Understand the standard Linux directory structure (FHS - Filesystem Hierarchy Standard). You will explore the critical directories used for configuration, logs, binaries, and user data.

## 📋 Prerequisites

-   Completed Lab 2.1.
-   Access to a Linux terminal.

## 📚 Background

### The Root (`/`)

Unlike Windows, which has `C:\`, `D:\`, etc., Linux has a single tree starting at `/`. Even other hard drives are "mounted" onto branches of this tree.

**Key Directories:**
-   `/bin` & `/usr/bin`: Essential user binaries (ls, cp).
-   `/sbin` & `/usr/sbin`: System binaries (reboot, fdisk).
-   `/etc`: Configuration files (Host specific). **Crucial for DevOps.**
-   `/var`: Variable data (Logs, Spool). **Crucial for Logs.**
-   `/home`: User home directories.
-   `/root`: Home directory for the root user.
-   `/tmp`: Temporary files (deleted on reboot).
-   `/proc`: Virtual filesystem (System info).

---

## 🔨 Hands-On Implementation

### Part 1: Exploring `/etc` (Configuration) ⚙️

This is where the system lives.

1.  **List contents:**
    ```bash
    cd /etc
    ls | less
    ```

2.  **View OS Release:**
    ```bash
    cat /etc/os-release
    ```
    *Output:* Shows your Linux distribution version (Ubuntu, CentOS, etc.).

3.  **View Hostname:**
    ```bash
    cat /etc/hostname
    ```

4.  **View DNS Configuration:**
    ```bash
    cat /etc/resolv.conf
    ```
    *Note:* This tells Linux which DNS server to use (e.g., 8.8.8.8).

### Part 2: Exploring `/var` (Logs & Data) 📝

This is where the system writes.

1.  **Check Logs:**
    ```bash
    cd /var/log
    ls
    ```

2.  **View System Log:**
    ```bash
    # Ubuntu/Debian
    tail -n 20 /var/log/syslog
    # RHEL/CentOS
    # tail -n 20 /var/log/messages
    ```
    *Note:* You might need `sudo` to read some logs.

### Part 3: Exploring `/proc` (System Info) 🧠

This directory doesn't exist on the disk. It's a window into the Kernel's mind.

1.  **Check CPU Info:**
    ```bash
    cat /proc/cpuinfo
    ```

2.  **Check Memory Info:**
    ```bash
    cat /proc/meminfo
    ```

3.  **Check Uptime:**
    ```bash
    cat /proc/uptime
    ```

### Part 4: Exploring `/bin` vs `/usr/bin` 📦

1.  **Where is `ls`?**
    ```bash
    which ls
    # Output: /usr/bin/ls (or /bin/ls)
    ```

2.  **Where is `python`?**
    ```bash
    which python3
    ```

---

## 🎯 Challenges

### Challenge 1: Disk Usage (Difficulty: ⭐⭐)

**Task:**
Find out which directory in `/var` is taking up the most space.
*Hint: Use `du` (Disk Usage) command with options `-h` (human readable) and `-s` (summary).*

### Challenge 2: Finding Configs (Difficulty: ⭐⭐)

**Task:**
Find the configuration file for the SSH service.
*Hint: It's in `/etc` and related to `ssh`.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
cd /var
sudo du -h --max-depth=1 | sort -hr
```
*Explanation:*
-   `du`: Disk Usage
-   `-h`: Human readable (K, M, G)
-   `--max-depth=1`: Only show immediate children
-   `sort -hr`: Sort by human readable numbers, reverse (biggest first)

**Challenge 2:**
```bash
ls /etc/ssh/sshd_config
```
</details>

---

## 🔑 Key Takeaways

1.  **Config is in `/etc`**: If you install Nginx, look for `/etc/nginx`. If you install MySQL, look for `/etc/mysql`.
2.  **Logs are in `/var/log`**: Something crashed? Look here first.
3.  **Everything is a File**: In Linux, even hardware info (`/proc`) looks like a file.

---

## ⏭️ Next Steps

Now that we know where things are, let's learn how to control who can touch them.

Proceed to **Lab 2.3: File Permissions & Ownership**.
