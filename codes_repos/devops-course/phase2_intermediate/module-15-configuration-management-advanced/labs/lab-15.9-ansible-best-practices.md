# Lab 15.9: Ansible Best Practices

## Objective
Implement Ansible best practices for production use.

## Learning Objectives
- Structure playbooks properly
- Use naming conventions
- Implement security
- Optimize performance

---

## Project Structure

```
ansible/
├── inventory/
│   ├── production/
│   └── staging/
├── group_vars/
│   ├── all.yml
│   └── webservers.yml
├── host_vars/
├── roles/
│   ├── common/
│   └── webserver/
├── playbooks/
│   ├── site.yml
│   └── webservers.yml
└── ansible.cfg
```

## Best Practices

```yaml
# Use descriptive names
- name: Install and configure nginx web server
  apt:
    name: nginx
    state: present

# Use variables
- name: Configure application
  template:
    src: app.conf.j2
    dest: "{{ app_config_path }}"

# Use handlers
- name: Update config
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: restart nginx

# Use tags
- name: Install packages
  apt:
    name: "{{ item }}"
  loop: "{{ packages }}"
  tags: [packages, install]
```

## Security

```yaml
# Use vault for secrets
db_password: !vault |
  $ANSIBLE_VAULT;1.1;AES256
  ...

# Use become sparingly
- name: Install package
  apt:
    name: nginx
  become: yes
  become_user: root
```

## Success Criteria
✅ Project well-structured  
✅ Best practices followed  
✅ Security implemented  
✅ Performance optimized  

**Time:** 40 min
