# Lab 2.4: Process Management

## 🎯 Objective

Learn how to monitor, manage, and terminate processes in Linux. You will understand Process IDs (PIDs), background jobs, and system services (systemd).

## 📋 Prerequisites

-   Completed Lab 2.3.
-   Access to a Linux terminal.

## 📚 Background

### What is a Process?

A process is a running instance of a program.
-   **PID**: Unique Process ID.
-   **Parent**: The process that started this one (e.g., the Shell started Python).
-   **Daemon**: A background process (Service).

---

## 🔨 Hands-On Implementation

### Part 1: Viewing Processes (`ps`, `top`) 🕵️‍♂️

1.  **Snapshot of Current Shell (`ps`):**
    ```bash
    ps
    ```
    *Output:* Shows `bash` and `ps`.

2.  **All Processes (`ps aux`):**
    The standard sysadmin command.
    ```bash
    ps aux | less
    ```
    -   `a`: All users.
    -   `u`: User format (shows CPU/Mem).
    -   `x`: Processes without a terminal (daemons).

3.  **Real-time Monitor (`top` or `htop`):**
    ```bash
    top
    ```
    -   Press `M` to sort by Memory.
    -   Press `P` to sort by CPU.
    -   Press `q` to quit.

### Part 2: Managing Jobs (Foreground/Background) 🏃‍♂️

1.  **Start a long-running process:**
    ```bash
    sleep 1000
    ```
    *Result:* Terminal is stuck. You can't type.

2.  **Suspend it:**
    Press `Ctrl+Z`.
    *Output:* `[1]+  Stopped  sleep 1000`

3.  **Background it (`bg`):**
    ```bash
    bg
    ```
    *Result:* It's running in the background now.

4.  **List Jobs:**
    ```bash
    jobs
    ```

5.  **Foreground it (`fg`):**
    ```bash
    fg %1
    ```
    *Result:* Back to being stuck. Press `Ctrl+C` to kill it.

6.  **Start directly in background:**
    Add `&` at the end.
    ```bash
    sleep 1000 &
    ```

### Part 3: Killing Processes (`kill`) 💀

1.  **Start a dummy process:**
    ```bash
    sleep 5000 &
    ```
    *Output:* `[1] 12345` (12345 is the PID).

2.  **Terminate Gracefully (SIGTERM - 15):**
    ```bash
    kill 12345  # Replace with your PID
    ```

3.  **Force Kill (SIGKILL - 9):**
    Use this only if the process is stuck/zombie.
    ```bash
    sleep 5000 &
    kill -9 <PID>
    ```

4.  **Kill by Name:**
    ```bash
    sleep 5000 &
    pkill sleep
    # OR
    killall sleep
    ```

### Part 4: System Services (`systemd`) ⚙️

Most Linux distros use `systemd` to manage services (web servers, docker, etc.).

1.  **Check Status:**
    ```bash
    systemctl status sshd
    # OR
    systemctl status cron
    ```

2.  **Start/Stop/Restart:**
    *(Requires sudo)*
    ```bash
    sudo systemctl restart cron
    ```

3.  **Enable on Boot:**
    ```bash
    sudo systemctl enable cron
    ```

---

## 🎯 Challenges

### Challenge 1: The CPU Hog (Difficulty: ⭐⭐)

**Task:**
1.  Create a script `hog.sh` that runs an infinite loop:
    ```bash
    #!/bin/bash
    while true; do :; done
    ```
2.  Run it in the background: `./hog.sh &`
3.  Use `top` to find it using 100% CPU.
4.  Kill it from within `top` (Press `k`, then enter PID).

### Challenge 2: Finding the Parent (Difficulty: ⭐⭐)

**Task:**
1.  Run `ps -ef`.
2.  Look at the columns. `PID` is Process ID. `PPID` is Parent Process ID.
3.  Find the PPID of your current `ps` command. What process is it? (Hint: It's your shell).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
-   Run `./hog.sh &`
-   Run `top`
-   See `bash` or `hog.sh` at top of list.
-   Press `k`. Enter PID. Press Enter (default signal 15).
-   Process disappears.

**Challenge 2:**
```bash
ps -ef | grep ps
```
The PPID of `ps` will match the PID of your `bash` session (check `echo $$` to see your shell's PID).
</details>

---

## 🔑 Key Takeaways

1.  **Don't panic**: If a terminal hangs, try `Ctrl+C`. If that fails, `Ctrl+Z` then `kill`.
2.  **`kill -9` is the last resort**: It doesn't give the process time to save data or clean up.
3.  **Services**: In production, apps run as systemd services, not as background jobs (`&`).

---

## ⏭️ Next Steps

We can manage processes. Now let's learn how to install software properly.

Proceed to **Lab 2.5: Package Management**.
