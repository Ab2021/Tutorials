# Lab 8.4: Ansible Automation at Scale

## Objective
Automate infrastructure management at scale with Ansible.

## Learning Objectives
- Manage large inventories
- Use dynamic inventory
- Optimize playbook performance
- Implement rolling updates

---

## Dynamic Inventory

```python
#!/usr/bin/env python3
# inventory.py
import json

inventory = {
    'webservers': {
        'hosts': ['web1.example.com', 'web2.example.com'],
        'vars': {
            'ansible_user': 'ubuntu',
            'http_port': 80
        }
    },
    'databases': {
        'hosts': ['db1.example.com', 'db2.example.com'],
        'vars': {
            'ansible_user': 'ubuntu',
            'db_port': 5432
        }
    },
    '_meta': {
        'hostvars': {
            'web1.example.com': {'ansible_host': '10.0.1.10'},
            'web2.example.com': {'ansible_host': '10.0.1.11'},
            'db1.example.com': {'ansible_host': '10.0.2.10'},
            'db2.example.com': {'ansible_host': '10.0.2.11'}
        }
    }
}

print(json.dumps(inventory))
```

## Rolling Updates

```yaml
- name: Rolling update
  hosts: webservers
  serial: "25%"  # Update 25% at a time
  max_fail_percentage: 10
  
  pre_tasks:
    - name: Remove from load balancer
      uri:
        url: "http://lb.example.com/remove/{{ inventory_hostname }}"
      delegate_to: localhost
  
  tasks:
    - name: Update application
      apt:
        name: myapp
        state: latest
      notify: restart app
    
    - name: Wait for service
      wait_for:
        port: 8080
        delay: 5
  
  post_tasks:
    - name: Add back to load balancer
      uri:
        url: "http://lb.example.com/add/{{ inventory_hostname }}"
      delegate_to: localhost
```

## Performance Optimization

```ini
# ansible.cfg
[defaults]
forks = 50
gathering = smart
fact_caching = redis
fact_caching_timeout = 3600
host_key_checking = False

[ssh_connection]
pipelining = True
ssh_args = -o ControlMaster=auto -o ControlPersist=60s
```

## Success Criteria
✅ Dynamic inventory working  
✅ Rolling updates functional  
✅ Performance optimized  
✅ Large scale managed  

**Time:** 45 min
