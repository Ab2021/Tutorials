# Lab 8.4: Configuration Management Capstone Project

## 🎯 Objective

Configure a **LAMP Stack** (Linux, Apache, MySQL, PHP) - or a modern equivalent - using Ansible. You will write a playbook that installs the software, starts the services, and deploys a simple app.

## 📋 Prerequisites

-   Completed Module 8.
-   A target machine (VM or Localhost).
-   *Note:* We will use **Nginx** instead of Apache for consistency with previous modules.

## 📚 Background

### The Goal
Turn a "Blank" Ubuntu server into a "Web Server" with one command.
1.  Update Apt Cache.
2.  Install Nginx.
3.  Install Python (for the app).
4.  Copy App Code.
5.  Start Services.

---

## 🔨 Hands-On Implementation

### Step 1: Structure 🏗️

```text
capstone/
├── inventory.ini
├── site.yml
└── roles/
    └── webapp/
        ├── tasks/main.yml
        ├── templates/index.html.j2
        └── handlers/main.yml
```

### Step 2: The Inventory 📝

`inventory.ini`:
```ini
[web]
localhost ansible_connection=local
```

### Step 3: The Role (Tasks) 📋

`roles/webapp/tasks/main.yml`:
```yaml
---
- name: Install Nginx
  apt:
    name: nginx
    state: present
    update_cache: yes
  notify: Restart Nginx

- name: Start Nginx
  service:
    name: nginx
    state: started
    enabled: yes

- name: Deploy App
  template:
    src: index.html.j2
    dest: /var/www/html/index.html
    mode: '0644'
```

### Step 4: The Handler 🔔

Handlers only run if a task reports "Changed".
`roles/webapp/handlers/main.yml`:
```yaml
---
- name: Restart Nginx
  service:
    name: nginx
    state: restarted
```

### Step 5: The Template 📄

`roles/webapp/templates/index.html.j2`:
```html
<h1>Deployed by Ansible</h1>
<p>Server: {{ ansible_hostname }}</p>
<p>OS: {{ ansible_distribution }}</p>
```

### Step 6: The Playbook 🎭

`site.yml`:
```yaml
---
- hosts: web
  become: yes
  roles:
    - webapp
```

### Step 7: Execution 🚀

```bash
ansible-playbook -i inventory.ini site.yml -K
```

---

## 🎯 Challenges

### Challenge 1: Multi-OS Support (Difficulty: ⭐⭐⭐)

**Task:**
Modify the role to support both **Ubuntu** (`apt`) and **CentOS** (`yum`).
*Hint: Use `when: ansible_os_family == "Debian"`.*

### Challenge 2: Vault (Difficulty: ⭐⭐⭐)

**Task:**
Create a file `secrets.yml` with `db_password: supersecret`.
Encrypt it with `ansible-vault encrypt secrets.yml`.
Include it in the playbook and use the variable.
Run with `--ask-vault-pass`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
- name: Install Nginx (Debian)
  apt: name=nginx state=present
  when: ansible_os_family == "Debian"

- name: Install Nginx (RedHat)
  yum: name=nginx state=present
  when: ansible_os_family == "RedHat"
```
</details>

---

## 🔑 Key Takeaways

1.  **Handlers**: Essential for restarting services only when config changes.
2.  **Facts**: `ansible_hostname` comes from the `setup` module (gathered automatically).
3.  **Infrastructure as Code**: You now have a repo that defines your server configuration.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 8.
You can provision servers (Terraform) and configure them (Ansible).

Proceed to **Module 9: Monitoring & Logging**.
