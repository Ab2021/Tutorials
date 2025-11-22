# Lab 8.2: Ansible Playbooks

## 🎯 Objective

Write your first Playbook. A Playbook is a YAML file that describes the desired state of your system (e.g., "Nginx should be installed and started").

## 📋 Prerequisites

-   Completed Lab 8.1.

## 📚 Background

### YAML Syntax
-   Start with `---`.
-   List of **Plays**.
-   Each Play has **Tasks**.
-   Each Task uses a **Module**.

### Structure
```yaml
---
- name: Configure Webserver
  hosts: local
  tasks:
    - name: Install Nginx
      apt:
        name: nginx
        state: present
```

---

## 🔨 Hands-On Implementation

### Part 1: The Playbook 📖

1.  **Create `site.yml`:**
    ```yaml
    ---
    - name: Setup Local Dev Environment
      hosts: local
      become: yes  # Run as sudo

      tasks:
        - name: Install Git
          apt:
            name: git
            state: present
            update_cache: yes

        - name: Create a directory
          file:
            path: /tmp/ansible_lab
            state: directory
            mode: '0755'

        - name: Create a file
          copy:
            dest: /tmp/ansible_lab/hello.txt
            content: "Hello from Ansible!"
    ```

### Part 2: Run It 🏃‍♂️

1.  **Execute:**
    ```bash
    ansible-playbook -i inventory.ini site.yml -K
    ```
    *Note:* `-K` asks for your sudo password.

2.  **Output:**
    -   `TASK [Gathering Facts]`: OK.
    -   `TASK [Install Git]`: Changed (or OK if already installed).
    -   `TASK [Create directory]`: Changed.
    -   `PLAY RECAP`: ok=4 changed=3 ...

### Part 3: Idempotency Check 🔄

1.  **Run it again:**
    ```bash
    ansible-playbook -i inventory.ini site.yml -K
    ```

2.  **Output:**
    -   `changed=0`.
    -   Everything should be green (OK).
    -   *Why?* Ansible checked the state, saw Git was installed and the file existed, so it did nothing.

---

## 🎯 Challenges

### Challenge 1: Uninstall (Difficulty: ⭐⭐)

**Task:**
Create a playbook `cleanup.yml` that:
1.  Removes the `/tmp/ansible_lab` directory (`state: absent`).
2.  Uninstalls Git (`state: absent`).
    *Warning:* Only uninstall Git if you are in a disposable VM! Otherwise just skip that step.

### Challenge 2: Variables (Difficulty: ⭐⭐⭐)

**Task:**
1.  Define a variable `dir_name: my_folder` in the `vars:` section.
2.  Use it in the task: `path: /tmp/{{ dir_name }}`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
tasks:
  - name: Remove directory
    file:
      path: /tmp/ansible_lab
      state: absent
```

**Challenge 2:**
```yaml
- hosts: local
  vars:
    dir_name: my_folder
  tasks:
    - file:
        path: "/tmp/{{ dir_name }}"
        state: directory
```
</details>

---

## 🔑 Key Takeaways

1.  **Declarative**: You say "State: Present". You don't say "apt-get install".
2.  **Modules**: There is a module for everything (AWS, Docker, Files, Users).
3.  **Become**: Use `become: yes` for tasks requiring root privileges.

---

## ⏭️ Next Steps

We can install software. Now let's manage complex configurations.

Proceed to **Lab 8.3: Roles & Templates**.
