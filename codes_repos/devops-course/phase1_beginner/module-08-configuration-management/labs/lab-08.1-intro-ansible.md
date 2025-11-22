# Lab 8.1: Introduction to Ansible

## 🎯 Objective

Understand Configuration Management. Terraform built the server (Infrastructure), now Ansible configures it (Software). You will install Ansible and run your first "Ad-Hoc" commands.

## 📋 Prerequisites

-   Linux/Mac or WSL (Windows Subsystem for Linux).
-   *Note:* Ansible Control Node does not run natively on Windows. Use WSL.

## 📚 Background

### Agentless Architecture
-   **Puppet/Chef**: Require an agent installed on every server.
-   **Ansible**: Uses SSH. No agent required.

### Inventory
A text file listing the servers you want to manage.
```ini
[webservers]
192.168.1.50
192.168.1.51
```

---

## 🔨 Hands-On Implementation

### Part 1: Install Ansible 🛠️

1.  **Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install -y ansible
    ```

2.  **Verify:**
    ```bash
    ansible --version
    ```

### Part 2: The Inventory File 📝

1.  **Create `inventory.ini`:**
    Since we might not have remote servers, we will manage `localhost`.
    ```ini
    [local]
    localhost ansible_connection=local
    ```

### Part 3: Ad-Hoc Commands ⚡

Run commands without writing a playbook.

1.  **Ping:**
    ```bash
    ansible -i inventory.ini all -m ping
    ```
    *Output:* `localhost | SUCCESS => { "ping": "pong" }`

2.  **Shell Command:**
    Check disk space.
    ```bash
    ansible -i inventory.ini all -m shell -a "df -h"
    ```

3.  **Install Package (Requires Sudo):**
    ```bash
    # -b means "become" (sudo)
    # -K asks for sudo password
    ansible -i inventory.ini all -m apt -a "name=git state=present" -b -K
    ```

---

## 🎯 Challenges

### Challenge 1: Managing Remote Servers (Difficulty: ⭐⭐⭐)

**Task:**
If you have the EC2 instance from Lab 7.4 running:
1.  Add its Public IP to `inventory.ini`.
2.  Configure SSH key: `[web] 1.2.3.4 ansible_user=ubuntu ansible_ssh_private_key_file=./mykey.pem`.
3.  Ping it.

### Challenge 2: The Setup Module (Difficulty: ⭐⭐)

**Task:**
Run the `setup` module against localhost.
`ansible -i inventory.ini all -m setup`
*Observe:* It gathers "Facts" (OS version, IP, CPU info). This is how Ansible knows what to do.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
`inventory.ini`:
```ini
[web]
54.1.2.3 ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/id_rsa
```
Command: `ansible -i inventory.ini web -m ping`

**Challenge 2:**
`ansible -i inventory.ini all -m setup`
Output is a huge JSON object.
</details>

---

## 🔑 Key Takeaways

1.  **Agentless**: If you can SSH, you can Ansible.
2.  **Idempotency**: Ansible modules (like `apt`, `copy`) check if the change is needed before doing it. `shell` module does not.
3.  **Inventory**: Can be static (text file) or dynamic (query AWS for all running instances).

---

## ⏭️ Next Steps

Ad-Hoc commands are good for one-offs. For repeatable configs, we need Playbooks.

Proceed to **Lab 8.2: Ansible Playbooks**.
