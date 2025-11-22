# Lab 15.1: Ansible Dynamic Inventory (AWS)

## 🎯 Objective

Stop hardcoding IP addresses. In the cloud, IPs change. You will configure Ansible to automatically query AWS for running instances and group them by tags.

## 📋 Prerequisites

-   Ansible installed.
-   AWS Account & CLI configured.
-   `boto3` python library installed (`pip install boto3`).

## 📚 Background

### Static vs Dynamic
-   **Static (`inventory.ini`)**: `[web] 1.2.3.4`. Good for pets.
-   **Dynamic (`aws_ec2.yml`)**: "Find all instances with tag `Role=Web`". Good for cattle.

---

## 🔨 Hands-On Implementation

### Part 1: Setup AWS Infrastructure ☁️

1.  **Launch 2 EC2 Instances:**
    -   Instance 1 Name: `Web-Server-1`. Tag: `Role=Web`.
    -   Instance 2 Name: `DB-Server-1`. Tag: `Role=DB`.
    -   Ensure your SSH Key is added to both.

### Part 2: Configure Dynamic Inventory 📋

1.  **Create `aws_ec2.yml`:**
    *Note: The filename MUST end in `aws_ec2.yml`.*
    ```yaml
    plugin: aws_ec2
    regions:
      - us-east-1
    filters:
      instance-state-name: running
    keyed_groups:
      - key: tags.Role
        prefix: role
      - key: tags.Name
        prefix: name
    hostnames:
      - dns-name
      - ip-address
    compose:
      ansible_host: public_ip_address
    ```

2.  **Test:**
    ```bash
    ansible-inventory -i aws_ec2.yml --graph
    ```
    *Result:*
    ```text
    @all:
      |--@role_Web:
      |  |--1.2.3.4
      |--@role_DB:
      |  |--5.6.7.8
    ```

### Part 3: Run Playbook 🏃‍♂️

1.  **Create `ping.yml`:**
    ```yaml
    ---
    - hosts: role_Web
      remote_user: ubuntu
      gather_facts: no
      tasks:
        - name: Ping Web Servers
          ping:
    ```

2.  **Execute:**
    ```bash
    ansible-playbook -i aws_ec2.yml ping.yml --private-key=my-key.pem
    ```
    *Result:* Only pings the Web Server.

---

## 🎯 Challenges

### Challenge 1: Private IP (Difficulty: ⭐⭐)

**Task:**
If you are running Ansible from *inside* the VPC (e.g., a Bastion Host), you should use Private IPs, not Public IPs.
Change `compose: ansible_host: public_ip_address` to `private_ip_address`.
Verify with `ansible-inventory`.

### Challenge 2: Group by Region (Difficulty: ⭐)

**Task:**
Add a keyed group to group instances by Availability Zone.
`key: placement.availability_zone`.
*Result:* `@us_east_1a` group.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
`aws_ec2.yml`:
```yaml
compose:
  ansible_host: private_ip_address
```
</details>

---

## 🔑 Key Takeaways

1.  **Plugins**: Ansible has plugins for Azure, GCP, Docker, and VMWare.
2.  **Tags are King**: Your automation relies entirely on correct tagging. Enforce tagging policies!
3.  **Speed**: Dynamic inventory queries the API every time. For large fleets, use caching (`cache: yes`).

---

## ⏭️ Next Steps

We can find servers dynamically. Now let's secure our secrets.

Proceed to **Lab 15.2: Ansible Vault**.
