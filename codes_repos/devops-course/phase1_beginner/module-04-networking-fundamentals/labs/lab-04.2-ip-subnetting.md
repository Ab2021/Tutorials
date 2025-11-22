# Lab 4.2: IP Addressing & Subnetting (CIDR)

## 🎯 Objective

Master the math of networking. Understand IPv4, CIDR notation (e.g., `/24`), and how to calculate subnets. This is **critical** for AWS/Cloud networking.

## 📋 Prerequisites

-   Basic math skills.
-   `ipcalc` tool (Optional: `sudo apt install ipcalc`).

## 📚 Background

### IPv4 Anatomy
`192.168.1.5`
-   4 octets (bytes).
-   Each octet is 0-255.
-   Total 32 bits.

### CIDR (Classless Inter-Domain Routing)
`192.168.1.0/24`
-   The `/24` is the **Mask**.
-   It means the first 24 bits are the **Network** (Fixed).
-   The remaining 8 bits (32-24) are for **Hosts** (Changeable).

**Calculation:**
-   2^8 = 256 IPs.
-   Minus 2 (Network Address + Broadcast Address).
-   Usable Hosts: 254.

---

## 🔨 Hands-On Implementation

### Part 1: Analyzing your Network 🏠

1.  **Check IP:**
    ```bash
    ip addr
    ```
    *Example:* `inet 192.168.1.15/24`

2.  **Calculate Range:**
    If mask is `/24`:
    -   Network: `192.168.1.0`
    -   First IP: `192.168.1.1`
    -   Last IP: `192.168.1.254`
    -   Broadcast: `192.168.1.255`

### Part 2: Using `ipcalc` 🧮

1.  **Analyze a /24:**
    ```bash
    ipcalc 192.168.1.0/24
    ```

2.  **Analyze a /16 (Larger Network):**
    ```bash
    ipcalc 10.0.0.0/16
    ```
    *Hosts:* 65,534.

3.  **Analyze a /28 (Small Subnet):**
    ```bash
    ipcalc 192.168.1.0/28
    ```
    *Hosts:* 14.

### Part 3: Subnetting Scenario 🍰

**Scenario:** You have `10.0.0.0/24` (256 IPs). You need to split it into 2 equal subnets for "Dev" and "Prod".

**Math:**
-   Current Mask: `/24`
-   To split in 2, add 1 bit to mask -> `/25`.

**Result:**
1.  **Subnet A (Dev)**: `10.0.0.0/25`
    -   Range: `10.0.0.0` - `10.0.0.127`
    -   Hosts: 126.
2.  **Subnet B (Prod)**: `10.0.0.128/25`
    -   Range: `10.0.0.128` - `10.0.0.255`
    -   Hosts: 126.

---

## 🎯 Challenges

### Challenge 1: The AWS VPC (Difficulty: ⭐⭐⭐)

**Task:**
You are designing an AWS VPC.
CIDR: `10.0.0.0/16`.
You need 4 subnets (Public-A, Public-B, Private-A, Private-B).
What CIDR should each subnet have to be equal size?
*Hint: /16 split into 4.*

### Challenge 2: Overlap Check (Difficulty: ⭐⭐)

**Task:**
Can `192.168.1.50/24` talk to `192.168.2.50/24` directly without a router?
*Hint: Look at the network portion.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
To split into 4 (2^2), add 2 bits to mask.
`/16` + 2 = `/18`.
-   Subnet 1: `10.0.0.0/18`
-   Subnet 2: `10.0.64.0/18`
-   Subnet 3: `10.0.128.0/18`
-   Subnet 4: `10.0.192.0/18`

**Challenge 2:**
No.
Network 1: `192.168.1.x`
Network 2: `192.168.2.x`
They are different networks. They need a Router (Gateway) to talk.
</details>

---

## 🔑 Key Takeaways

1.  **Smaller Mask = More Hosts**: `/8` is huge (Internet). `/32` is one IP.
2.  **Private Ranges (RFC 1918)**:
    -   `10.0.0.0/8` (Big corps/Cloud)
    -   `172.16.0.0/12` (Docker/AWS)
    -   `192.168.0.0/16` (Home routers)
3.  **Don't overlap**: You can't have two subnets with the same IPs.

---

## ⏭️ Next Steps

We have addresses. Now let's look at names.

Proceed to **Lab 4.3: DNS & HTTP**.
