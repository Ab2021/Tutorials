# Lab 9.1: Introduction to Monitoring Concepts

## 🎯 Objective

Understand why we monitor. You will manually check system health using Linux tools, then understand how modern tools automate this.

## 📋 Prerequisites

-   Linux Terminal.

## 📚 Background

### The 4 Golden Signals (Google SRE)
1.  **Latency**: Time it takes to serve a request.
2.  **Traffic**: How much demand (req/sec).
3.  **Errors**: Rate of requests failing (5xx).
4.  **Saturation**: How "full" is the service (CPU/RAM usage).

### Monitoring vs Logging
-   **Monitoring**: "Is it healthy?" (Metrics, Graphs).
-   **Logging**: "What happened?" (Text, Debugging).

---

## 🔨 Hands-On Implementation

### Part 1: Manual Monitoring (The Old Way) 👴

1.  **Check CPU/RAM:**
    ```bash
    top
    # OR
    htop
    ```
    *Look for:* Load Average, %CPU, Mem.

2.  **Check Disk:**
    ```bash
    df -h
    ```
    *Look for:* Use%. If 100%, server crashes.

3.  **Check Network:**
    ```bash
    iftop
    # OR
    nload
    ```

### Part 2: Simulate High Load 🏋️‍♂️

1.  **Run a stress test:**
    ```bash
    # Install stress
    sudo apt install -y stress
    # Run 2 CPU hogs for 60 seconds
    stress --cpu 2 --timeout 60 &
    ```

2.  **Monitor:**
    Run `top`. Observe CPU usage spiking to 100%.

### Part 3: The Problem ⚠️

-   You can't watch `top` 24/7.
-   You have 100 servers.
-   **Solution**: Prometheus (collects metrics) + Grafana (visualizes them).

---

## 🎯 Challenges

### Challenge 1: Load Average (Difficulty: ⭐⭐)

**Task:**
Run `uptime`. You see `load average: 0.50, 0.30, 0.10`.
What do these 3 numbers mean?
*Hint: 1 min, 5 min, 15 min averages.*

### Challenge 2: Disk Inodes (Difficulty: ⭐⭐⭐)

**Task:**
You have 50GB free space, but you can't create a file. Why?
Run `df -i`.
*Explanation:* You ran out of Inodes (file slots), usually caused by millions of tiny files.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
The numbers represent the average number of processes waiting for CPU over the last 1, 5, and 15 minutes. If Load > Number of Cores, you are saturated.

**Challenge 2:**
`df -i` shows Inode usage. If 100%, delete files.
</details>

---

## 🔑 Key Takeaways

1.  **Metrics are Cheap**: Storing numbers (CPU=50%) is cheaper than storing text logs.
2.  **Alerting**: Monitoring is useless without alerts ("Page me if CPU > 90%").
3.  **Baseline**: You need to know what "Normal" looks like to detect "Abnormal".

---

## ⏭️ Next Steps

Let's install a real monitoring stack.

Proceed to **Lab 9.2: Prometheus Setup**.
