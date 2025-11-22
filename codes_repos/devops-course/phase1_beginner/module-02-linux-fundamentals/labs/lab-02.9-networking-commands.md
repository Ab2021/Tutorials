# Lab 2.9: Networking Commands

## 🎯 Objective

Master the essential Linux networking commands. You will learn how to check connectivity (`ping`), transfer data (`curl`, `wget`), resolve names (`dig`, `nslookup`), and log in remotely (`ssh`).

## 📋 Prerequisites

-   Completed Lab 2.8.
-   Internet connection.

## 📚 Background

### TCP/IP Basics

-   **IP Address**: The phone number of the computer (e.g., `192.168.1.5`).
-   **Port**: The specific extension (e.g., `:80` for Web, `:22` for SSH).
-   **DNS**: The phonebook (converts `google.com` to IP).

---

## 🔨 Hands-On Implementation

### Part 1: Connectivity (`ping`) 🏓

1.  **Check if a host is alive:**
    ```bash
    ping -c 4 google.com
    ```
    *Output:* Look for `time=... ms`. High time = lag. Packet loss = bad connection.

### Part 2: Data Transfer (`curl`, `wget`) 📥

1.  **Download a file (`wget`):**
    ```bash
    wget https://example.com/index.html
    ```

2.  **Fetch a webpage (`curl`):**
    ```bash
    curl https://example.com
    ```

3.  **Check Headers (`curl -I`):**
    See what the server says about itself.
    ```bash
    curl -I https://google.com
    ```
    *Look for:* `HTTP/1.1 200 OK` or `301 Moved`.

### Part 3: DNS Lookup (`dig`, `nslookup`) 📒

1.  **Find IP of a domain:**
    ```bash
    nslookup google.com
    ```

2.  **Detailed Lookup (`dig`):**
    The pro tool.
    ```bash
    dig google.com
    ```
    *Look for:* The `ANSWER SECTION`.

### Part 4: Remote Access (`ssh`) 🔑

*Note: You need an SSH server to connect to. If you don't have one, just learn the syntax.*

1.  **Connect to a server:**
    ```bash
    # ssh user@host
    ssh student@192.168.1.50
    ```

2.  **Generate Keys:**
    To log in without a password.
    ```bash
    ssh-keygen -t rsa -b 4096
    # Press Enter for defaults
    ```
    *Result:* Creates `~/.ssh/id_rsa` (Private) and `~/.ssh/id_rsa.pub` (Public).

---

## 🎯 Challenges

### Challenge 1: The API Call (Difficulty: ⭐⭐)

**Task:**
Use `curl` to fetch data from a public API.
URL: `https://api.github.com/users/defunkt`
1.  Fetch the JSON.
2.  Pipe it to `grep` to find the "name" field.

### Challenge 2: Port Scanning (Difficulty: ⭐⭐⭐)

**Task:**
Use `nc` (Netcat) or `telnet` to check if port 80 is open on `google.com`.
*Hint:* `nc -zv google.com 80`

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
curl -s https://api.github.com/users/defunkt | grep "name"
```

**Challenge 2:**
```bash
nc -zv google.com 80
# Output: Connection to google.com 80 port [tcp/http] succeeded!
```
</details>

---

## 🔑 Key Takeaways

1.  **Curl is powerful**: It can do POST requests, send cookies, and more. It's the main tool for testing APIs.
2.  **DNS is usually the problem**: If `ping 8.8.8.8` works but `ping google.com` fails, your DNS is broken.
3.  **SSH Keys**: Never share your private key (`id_rsa`). Only share the public key (`id_rsa.pub`).

---

## ⏭️ Next Steps

We have learned the tools. Now, let's build something real.

Proceed to **Lab 2.10: Bash Automation Project**.
