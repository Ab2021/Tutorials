# Lab 2.8: System Monitoring

## 🎯 Objective

Learn how to check the health of your Linux system. You will monitor Disk usage, Memory consumption, CPU load, and Network activity using standard CLI tools.

## 📋 Prerequisites

-   Completed Lab 2.7.
-   Access to a Linux terminal.

## 📚 Background

### The Four Resources

1.  **CPU**: Processing power. Measured in % usage and "Load Average".
2.  **Memory (RAM)**: Temporary storage. Measured in GB/MB.
3.  **Disk**: Permanent storage. Measured in Space (GB) and I/O (Speed).
4.  **Network**: Connectivity. Measured in Bandwidth and Latency.

---

## 🔨 Hands-On Implementation

### Part 1: Disk Monitoring 💾

1.  **Disk Space (`df`):**
    ```bash
    df -h
    ```
    *Look for:* The `/` (root) filesystem. Is it full? (Use % column).

2.  **Directory Size (`du`):**
    ```bash
    du -sh /var/log
    ```
    *Note:* `-s` = summary, `-h` = human readable.

### Part 2: Memory Monitoring 🧠

1.  **Free Memory (`free`):**
    ```bash
    free -h
    ```
    *Columns:*
    -   **Total**: Installed RAM.
    -   **Used**: Currently used.
    -   **Available**: RAM available for new apps (includes Cache). **This is the most important number.**

2.  **Virtual Memory Stats (`vmstat`):**
    ```bash
    vmstat 1
    ```
    *Action:* Prints stats every 1 second. Watch the `si` (swap in) and `so` (swap out) columns. If non-zero, you are out of RAM!

### Part 3: CPU Monitoring ⚡

1.  **Uptime & Load (`uptime`):**
    ```bash
    uptime
    ```
    *Output:* `load average: 0.05, 0.10, 0.05`
    *Meaning:* Average number of processes waiting for CPU over 1, 5, and 15 minutes.
    *Rule of Thumb:* If Load > Number of Cores, you are overloaded.

2.  **CPU Info (`lscpu`):**
    ```bash
    lscpu
    ```
    *Check:* `CPU(s)` line to see how many cores you have.

### Part 4: Network Monitoring 🌐

1.  **IP Address (`ip`):**
    ```bash
    ip addr show
    ```
    *Legacy:* `ifconfig` (might not be installed).

2.  **Connectivity (`ping`):**
    ```bash
    ping -c 4 google.com
    ```

3.  **Listening Ports (`ss`):**
    See what services are accepting connections.
    ```bash
    ss -tuln
    ```
    -   `t`: TCP
    -   `u`: UDP
    -   `l`: Listening
    -   `n`: Numeric (show ports numbers, not names)

---

## 🎯 Challenges

### Challenge 1: The Stress Test (Difficulty: ⭐⭐)

**Task:**
1.  Open two terminals.
2.  In Terminal 1, run `top`.
3.  In Terminal 2, run a command to generate load:
    ```bash
    yes > /dev/null &
    ```
4.  Watch Terminal 1. What happens to CPU usage? What happens to Load Average?
5.  **Cleanup:** Don't forget to `kill` the `yes` process!

### Challenge 2: Port Finder (Difficulty: ⭐⭐)

**Task:**
1.  Start a Python web server in the background:
    ```bash
    python3 -m http.server 8080 &
    ```
2.  Use `ss` or `netstat` to confirm port 8080 is listening.
3.  Kill the python process.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
-   CPU usage for the `yes` process will jump to 100% (of one core).
-   Load average will slowly rise towards 1.0.

**Challenge 2:**
```bash
ss -tuln | grep 8080
# Output should show a line with :8080
fg %1  # Bring python to foreground
Ctrl+C # Kill it
```
</details>

---

## 🔑 Key Takeaways

1.  **Load Average vs CPU %**: CPU % is instantaneous. Load Average is a trend. High load with low CPU % usually means Disk I/O bottleneck (waiting for disk).
2.  **Available RAM**: Don't panic if "Free" RAM is low. Linux uses free RAM to cache files. Look at "Available".
3.  **Swap is Slow**: If you are swapping (`vmstat` si/so), your system will crawl. Add RAM.

---

## ⏭️ Next Steps

We can monitor the local machine. Now let's talk to other machines.

Proceed to **Lab 2.9: Networking Commands**.
