# Lab 4.5: Firewalls & Security Groups

## 🎯 Objective

Learn how to protect your servers using host-based firewalls. You will use `ufw` (Uncomplicated Firewall) to block and allow traffic, simulating a production security setup.

## 📋 Prerequisites

-   Linux VM (Ubuntu preferred for `ufw`).
-   Sudo access.

## 📚 Background

### The First Line of Defense
A firewall controls incoming and outgoing network traffic based on rules.
-   **Allow**: Traffic can pass.
-   **Deny**: Traffic is blocked silently.
-   **Reject**: Traffic is blocked with an error message.

**Cloud Security Groups**: In AWS/Azure, "Security Groups" act as a firewall *outside* your VM.
**Host Firewalls**: `ufw`, `iptables`, `firewalld` run *inside* your VM. **Defense in Depth** means using both.

---

## 🔨 Hands-On Implementation

### Part 1: Checking Status 🛡️

1.  **Check UFW:**
    ```bash
    sudo ufw status
    ```
    *Default:* `Status: inactive`.

### Part 2: Setting Defaults (Safety First) 🔒

1.  **Default Policy:**
    Block everything coming in. Allow everything going out.
    ```bash
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    ```

### Part 3: Allowing Access (Don't lock yourself out!) 🔑

1.  **Allow SSH:**
    **CRITICAL:** If you are on SSH, run this before enabling!
    ```bash
    sudo ufw allow ssh
    # OR specific port
    sudo ufw allow 22/tcp
    ```

2.  **Allow Web:**
    ```bash
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    ```

### Part 4: Enabling & Testing 🟢

1.  **Enable:**
    ```bash
    sudo ufw enable
    ```
    *Warning:* It will ask for confirmation. Say `y`.

2.  **Verify:**
    ```bash
    sudo ufw status verbose
    ```

3.  **Test:**
    Try to connect to a blocked port (e.g., 8080) using `nc` or `telnet` from another machine. It should time out.

### Part 5: Deleting Rules 🗑️

1.  **List with Numbers:**
    ```bash
    sudo ufw status numbered
    ```

2.  **Delete Rule #:**
    ```bash
    sudo ufw delete 2
    ```

---

## 🎯 Challenges

### Challenge 1: Specific IP Access (Difficulty: ⭐⭐⭐)

**Scenario:** You have a database on port 5432. You only want the "App Server" (IP: 192.168.1.50) to access it. Everyone else should be blocked.
**Task:**
Write the `ufw` command to allow port 5432 ONLY from 192.168.1.50.

### Challenge 2: Rate Limiting (Difficulty: ⭐⭐)

**Scenario:** Bots are brute-forcing your SSH.
**Task:**
Use `ufw limit` to block IPs that try to log in too many times in 30 seconds.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
sudo ufw allow from 192.168.1.50 to any port 5432
```

**Challenge 2:**
```bash
sudo ufw limit ssh
```
*Explanation:* This limits connections to 6 attempts per 30 seconds.
</details>

---

## 🔑 Key Takeaways

1.  **Order Matters**: Firewalls process rules top-to-bottom.
2.  **Default Deny**: The safest policy is "Block everything, then allow what is needed."
3.  **Cloud vs Host**: Even if AWS Security Group allows port 80, `ufw` can still block it. Check both.

---

## ⏭️ Next Steps

We secured a single server. Now let's handle traffic for multiple servers.

Proceed to **Lab 4.6: Load Balancing Basics**.
