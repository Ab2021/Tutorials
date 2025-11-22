# Lab 4.7: Network Troubleshooting Tools

## 🎯 Objective

Become a network detective. Learn to diagnose connectivity issues, identify bottlenecks, and map out networks using `netstat`, `nmap`, and `mtr`.

## 📋 Prerequisites

-   Linux Terminal.
-   `nmap` and `mtr` installed.

## 📚 Background

### The Detective's Toolkit
1.  **netstat / ss**: Who is talking to whom on *my* machine?
2.  **nmap**: What ports are open on *that* machine?
3.  **mtr**: Where is the packet getting lost? (Ping + Traceroute).

---

## 🔨 Hands-On Implementation

### Part 1: Listening Ports (`netstat` / `ss`) 👂

**Scenario:** You started a web server, but you can't reach it. Is it actually running?

1.  **Check Listening Ports:**
    ```bash
    sudo ss -tulnp
    ```
    *Flags:*
    -   `t`: TCP
    -   `u`: UDP
    -   `l`: Listening
    -   `n`: Numeric
    -   `p`: Process (Show PID/Name) **<-- Critical**

2.  **Find who is using Port 22:**
    ```bash
    sudo ss -tulnp | grep :22
    ```
    *Output:* Should show `sshd`.

### Part 2: Network Mapping (`nmap`) 🗺️

**Scenario:** You forgot the IP of your Raspberry Pi, or you want to see what services a server exposes.

1.  **Scan a single host:**
    ```bash
    nmap google.com
    ```
    *Output:* Shows open ports (80, 443).

2.  **Scan a network (Ping Scan):**
    Find all live hosts on your home network.
    ```bash
    nmap -sn 192.168.1.0/24
    ```
    *Note:* Replace with your actual subnet.

3.  **Service Version Detection:**
    ```bash
    nmap -sV google.com
    ```
    *Output:* Tries to guess the software version (e.g., `nginx 1.18`).

### Part 3: Path Analysis (`mtr`) 🛣️

**Scenario:** Your connection to a server is laggy. Is it your WiFi? The ISP? Or the server?

1.  **Run MTR:**
    ```bash
    mtr google.com
    ```
    *Interface:* Shows a live updating table.
    *Columns:*
    -   **Loss%**: Packet loss at this hop.
    -   **Avg**: Average latency.

    *Analysis:*
    -   Loss at Hop 1? Your router/WiFi is bad.
    -   Loss at Hop 10? The destination is bad.
    -   Loss in middle? ISP issue.

---

## 🎯 Challenges

### Challenge 1: The Phantom Port (Difficulty: ⭐⭐)

**Task:**
1.  Use `nc -l 9999` to open a listener on port 9999 in one terminal.
2.  Use `nmap localhost` in another terminal to verify it's open.
3.  Use `ss` to find the Process ID (PID) of `nc`.

### Challenge 2: Aggressive Scan (Difficulty: ⭐⭐⭐)

**Task:**
Run `nmap -A scanme.nmap.org`.
Read the output. It tries to detect OS, Traceroute, and Scripts.
*Warning: Do not run this on servers you don't own (except scanme.nmap.org).*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Term 1: `nc -l 9999`
Term 2:
```bash
nmap -p 9999 localhost
sudo ss -tulnp | grep 9999
# Output shows "users:(("nc",pid=1234,fd=3))"
```
</details>

---

## 🔑 Key Takeaways

1.  **Permission Denied**: You need `sudo` for `ss -p` (to see process names) and `nmap` (for OS detection).
2.  **Nmap is Noisy**: Security teams monitor for Nmap scans. Don't scan your work network without permission.
3.  **MTR is King**: It's the best tool for proving to an ISP that *their* router is dropping packets.

---

## ⏭️ Next Steps

We can find open ports. Now let's secure the traffic flowing through them.

Proceed to **Lab 4.8: SSL/TLS & Certificates**.
