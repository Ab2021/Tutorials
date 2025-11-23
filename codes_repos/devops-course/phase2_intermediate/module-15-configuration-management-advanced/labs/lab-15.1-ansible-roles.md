# Lab 15.1: Ansible Roles

## Objective
Create reusable Ansible roles for configuration management.

## Learning Objectives
- Create Ansible role structure
- Use role variables and defaults
- Implement role dependencies
- Share roles via Ansible Galaxy

---

## Role Structure

```bash
ansible-galaxy init webserver

# Creates:
webserver/
├── defaults/
│   └── main.yml
├── files/
├── handlers/
│   └── main.yml
├── tasks/
│   └── main.yml
├── templates/
├── vars/
│   └── main.yml
└── meta/
    └── main.yml
```

## Role Tasks

```yaml
# roles/webserver/tasks/main.yml
- name: Install nginx
  apt:
    name: nginx
    state: present

- name: Copy config
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: restart nginx

- name: Start nginx
  service:
    name: nginx
    state: started
    enabled: yes
```

## Using Roles

```yaml
# playbook.yml
- hosts: webservers
  roles:
    - webserver
    - { role: database, db_port: 5432 }
```

## Success Criteria
✅ Created role structure  
✅ Implemented tasks and handlers  
✅ Used role in playbook  
✅ Role is reusable  

**Time:** 40 min
