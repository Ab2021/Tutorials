# Lab 15.3: Ansible Galaxy

## Objective
Use Ansible Galaxy to share and consume community roles.

## Learning Objectives
- Search and install Galaxy roles
- Create requirements file
- Publish custom roles
- Manage role versions

---

## Install Roles

```bash
# Install single role
ansible-galaxy install geerlingguy.nginx

# Install from requirements
cat > requirements.yml << 'EOF'
roles:
  - name: geerlingguy.nginx
    version: 3.1.4
  - name: geerlingguy.mysql
EOF

ansible-galaxy install -r requirements.yml
```

## Use Galaxy Role

```yaml
# playbook.yml
- hosts: webservers
  roles:
    - geerlingguy.nginx
  vars:
    nginx_vhosts:
      - listen: "80"
        server_name: "example.com"
        root: "/var/www/html"
```

## Publish Role

```bash
# Login to Galaxy
ansible-galaxy login

# Import role from GitHub
ansible-galaxy import username repo-name
```

## Success Criteria
✅ Installed Galaxy roles  
✅ Used community roles  
✅ Created requirements file  
✅ Published custom role  

**Time:** 35 min
