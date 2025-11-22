# Configuration Management with Ansible

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Configuration Management, including:
- **Concepts**: Why "Snowflake Servers" are bad and how to avoid them.
- **Ansible Architecture**: Agentless, Push-based, and SSH-driven.
- **Playbooks**: Writing YAML to define the state of your infrastructure.
- **Modules**: Using built-in modules (`apt`, `copy`, `service`) to automate tasks.
- **Roles**: Structuring your code for reusability and scale.

---

## 📖 Theoretical Concepts

### 1. What is Configuration Management?

Configuration Management (CM) is the practice of handling changes systematically so that a system maintains its integrity over time.
- **Problem**: You manually install Nginx on Server A. A month later, you install it on Server B but forget a config flag. They are now different.
- **Solution**: Define the configuration in code. Apply it to both servers. They are identical.

### 2. Ansible Architecture

Ansible is a radical departure from older tools (Puppet, Chef) because it is **Agentless**.
- **Control Node**: Your laptop or CI server. Runs Ansible.
- **Managed Nodes**: The servers you want to configure.
- **Protocol**: SSH. No agents to install.
- **Inventory**: A text file listing IP addresses of Managed Nodes.

### 3. Core Concepts

- **Ad-Hoc Commands**: One-liners for quick tasks.
  - `ansible all -m ping`
- **Playbooks**: The core of Ansible. YAML files that describe a list of tasks.
- **Modules**: The "tools" in the toolkit.
  - `apt`/`yum`: Manage packages.
  - `copy`/`template`: Manage files.
  - `service`: Manage daemons.
- **Idempotency**: An operation is idempotent if the result of performing it once is exactly the same as the result of performing it multiple times without any intervening actions.

---

## 🔧 Practical Examples

### Inventory File (`hosts.ini`)

```ini
[webservers]
192.168.1.10
192.168.1.11

[dbservers]
192.168.1.20
```

### Basic Playbook (`site.yml`)

```yaml
---
- name: Configure Web Servers
  hosts: webservers
  become: yes  # Run as sudo

  tasks:
    - name: Install Nginx
      apt:
        name: nginx
        state: present

    - name: Start Nginx
      service:
        name: nginx
        state: started
        enabled: yes

    - name: Deploy Index Page
      copy:
        src: index.html
        dest: /var/www/html/index.html
```

### Using Variables

```yaml
vars:
  http_port: 80

tasks:
  - name: Ensure port is open
    ufw:
      rule: allow
      port: "{{ http_port }}"
```

---

## 🎯 Hands-on Labs

- [Lab 8.1: Introduction to Ansible](./labs/lab-08.1-intro-ansible.md)
- [Lab 8.2: Ansible Playbooks](./labs/lab-08.2-ansible-playbooks.md)
- [Lab 8.3: Ansible Roles & Templates](./labs/lab-08.3-ansible-roles.md)
- [Lab 8.4: Configuration Management Capstone Project](./labs/lab-08.4-ansible-project.md)

---

## 📚 Additional Resources

### Official Documentation
- [Ansible Documentation](https://docs.ansible.com/)
- [Ansible Galaxy](https://galaxy.ansible.com/) - Repository of community roles.

### Tutorials
- [Jeff Geerling's Ansible 101](https://www.youtube.com/playlist?list=PL2_OBreMn7FqZkvMYt6ATmgC0KAGGJNAN)

---

## 🔑 Key Takeaways

1.  **Automate Everything**: If you do it twice, automate it.
2.  **State, Not Scripts**: Don't write "run this command". Write "ensure this file exists".
3.  **Roles are Libraries**: Don't write one giant playbook. Break it down into Roles (e.g., `nginx`, `mysql`, `security`).
4.  **Test It**: Use tools like Molecule to test your roles.

---

## ⏭️ Next Steps

1.  Complete the labs to configure your first server fleet.
2.  Proceed to **[Module 9: Monitoring & Logging Basics](../module-09-monitoring-logging-basics/README.md)** to see what your servers are doing.
