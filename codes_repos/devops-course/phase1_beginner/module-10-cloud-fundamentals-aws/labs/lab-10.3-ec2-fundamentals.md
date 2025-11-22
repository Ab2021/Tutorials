# Lab 10.3: EC2 Fundamentals

## 🎯 Objective

Launch a virtual server in the cloud. You will manually provision an EC2 instance, connect to it via SSH, host a simple website, and understand the pricing model.

## 📋 Prerequisites

-   AWS Account.
-   SSH Client (Terminal or Putty).

## 📚 Background

### EC2 (Elastic Compute Cloud)
-   **Instance Type**: Hardware specs. `t2.micro` (1 vCPU, 1GB RAM) is Free Tier.
-   **AMI (Amazon Machine Image)**: The OS (Ubuntu, Amazon Linux, Windows).
-   **Security Group**: The Firewall.
-   **Key Pair**: The password (SSH Key).

---

## 🔨 Hands-On Implementation

### Part 1: Launch Instance 🚀

1.  **Go to EC2 Console** -> **Launch Instance**.
2.  **Name**: `My-First-Server`.
3.  **OS**: Ubuntu 22.04 LTS (Free Tier eligible).
4.  **Instance Type**: `t2.micro`.
5.  **Key Pair**: Create new key pair -> `my-key` -> Download `.pem`.
6.  **Network Settings**:
    -   Allow SSH traffic from **My IP** (Security Best Practice).
    -   Allow HTTP traffic from the internet.
7.  **Advanced Details (User Data)**:
    Paste this script to auto-install a web server on boot:
    ```bash
    #!/bin/bash
    apt update
    apt install -y apache2
    echo "<h1>Hello from AWS EC2</h1>" > /var/www/html/index.html
    systemctl start apache2
    systemctl enable apache2
    ```
8.  **Launch Instance**.

### Part 2: Connect 🔌

1.  **Get Public IP**:
    Select instance. Copy "Public IPv4 address".

2.  **SSH (Linux/Mac):**
    ```bash
    chmod 400 my-key.pem
    ssh -i my-key.pem ubuntu@<PUBLIC-IP>
    ```

3.  **Verify Web Server:**
    Open browser: `http://<PUBLIC-IP>`.
    *Result:* "Hello from AWS EC2".

### Part 3: Stop vs Terminate 🛑

1.  **Stop Instance**:
    -   The computer shuts down.
    -   You stop paying for Compute (CPU/RAM).
    -   You **still pay** for Storage (EBS Volume).
    -   Public IP changes (unless you use Elastic IP).

2.  **Terminate Instance**:
    -   The computer is destroyed.
    -   Storage is deleted (usually).
    -   Billing stops completely.

---

## 🎯 Challenges

### Challenge 1: Spot Instances (Difficulty: ⭐⭐)

**Task:**
Launch a "Spot Instance".
*Concept:* AWS sells unused capacity at 90% discount.
*Risk:* AWS can terminate your instance with 2 minutes notice if they need the capacity back.
*Goal:* Verify the price difference in the console.

### Challenge 2: Elastic IP (Difficulty: ⭐⭐)

**Task:**
Allocate an Elastic IP.
Associate it with your running instance.
Stop and Start the instance.
*Observation:* The IP stays the same.
**Important:** Release the Elastic IP when done (AWS charges for *unused* Elastic IPs).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
In Launch Wizard -> Advanced Options -> Request Spot Instances.

**Challenge 2:**
EC2 Console -> Network & Security -> Elastic IPs.
</details>

---

## 🔑 Key Takeaways

1.  **Security Groups**: If you can't connect, check the Security Group (Port 22/80).
2.  **User Data**: Powerful way to bootstrap servers.
3.  **Cattle, not Pets**: In the cloud, servers are disposable. Don't give them cute names; give them tags.

---

## ⏭️ Next Steps

We have compute. Now we need storage.

Proceed to **Lab 10.4: S3 Storage**.
