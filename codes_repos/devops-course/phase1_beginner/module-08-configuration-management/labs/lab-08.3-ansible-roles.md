# Lab 8.3: Ansible Roles & Templates

## 🎯 Objective

Organize your code. A single `site.yml` with 100 tasks is unreadable. You will use **Roles** to split tasks into reusable components and **Jinja2 Templates** to generate dynamic configuration files.

## 📋 Prerequisites

-   Completed Lab 8.2.

## 📚 Background

### Roles
Standard directory structure:
```text
roles/
  webserver/
    tasks/main.yml
    templates/nginx.conf.j2
    vars/main.yml
```

### Jinja2 Templates (`.j2`)
Allow you to inject variables into text files.
`Welcome {{ user_name }}!` -> `Welcome Alice!`

---

## 🔨 Hands-On Implementation

### Part 1: Create Role Structure 📂

1.  **Command:**
    ```bash
    ansible-galaxy init roles/webserver
    ```
    *Result:* Creates the folder structure automatically.

### Part 2: The Template 📝

We want to configure a welcome message file dynamically.

1.  **Create `roles/webserver/templates/index.html.j2`:**
    ```html
    <html>
    <body>
        <h1>Welcome to {{ server_name }}</h1>
        <p>Managed by Ansible.</p>
    </body>
    </html>
    ```

### Part 3: The Tasks 📋

1.  **Edit `roles/webserver/tasks/main.yml`:**
    ```yaml
    ---
    - name: Create web root
      file:
        path: /tmp/www
        state: directory

    - name: Deploy Index Page
      template:
        src: index.html.j2
        dest: /tmp/www/index.html
    ```

### Part 4: The Playbook 🎭

1.  **Create `site.yml`:**
    ```yaml
    ---
    - hosts: local
      vars:
        server_name: "My DevOps Lab"
      roles:
        - webserver
    ```

2.  **Run:**
    ```bash
    ansible-playbook -i inventory.ini site.yml
    ```

3.  **Verify:**
    ```bash
    cat /tmp/www/index.html
    ```
    *Result:* `<h1>Welcome to My DevOps Lab</h1>`.

---

## 🎯 Challenges

### Challenge 1: Loops (Difficulty: ⭐⭐⭐)

**Task:**
In the role, create a list of users to create.
`vars/main.yml`: `users: [alice, bob]`
`tasks/main.yml`: Use `user` module with `loop: "{{ users }}"`.
*Note:* Requires sudo (`become: yes`).

### Challenge 2: Ansible Galaxy (Difficulty: ⭐⭐)

**Task:**
Search Ansible Galaxy for a pre-made role (e.g., `geerlingguy.nginx`).
Install it: `ansible-galaxy install geerlingguy.nginx`.
Use it in your playbook.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
`tasks/main.yml`:
```yaml
- name: Create users
  user:
    name: "{{ item }}"
    state: present
  loop: "{{ users }}"
```
</details>

---

## 🔑 Key Takeaways

1.  **Don't Reinvent the Wheel**: Use Roles.
2.  **Separation of Concerns**: Variables go in `vars`, Tasks in `tasks`, Files in `templates`.
3.  **Dynamic Config**: Jinja2 is powerful. You can use `{% if %}` loops inside config files.

---

## ⏭️ Next Steps

We have the skills. Let's configure a full stack.

Proceed to **Lab 8.4: Configuration Management Capstone**.
