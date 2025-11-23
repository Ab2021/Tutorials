# Lab 15.10: Ansible at Scale

## Objective
Manage large-scale infrastructure with Ansible.

## Learning Objectives
- Optimize for large inventories
- Use delegation and local actions
- Implement rolling updates
- Monitor playbook execution

---

## Large Inventory Optimization

```ini
# ansible.cfg
[defaults]
forks = 100
host_key_checking = False
gathering = smart
fact_caching = redis
fact_caching_timeout = 3600
```

## Delegation

```yaml
- name: Update load balancer
  hosts: webservers
  tasks:
    - name: Remove from load balancer
      uri:
        url: "http://lb.example.com/remove/{{ inventory_hostname }}"
      delegate_to: localhost
    
    - name: Update application
      apt:
        name: myapp
        state: latest
    
    - name: Add back to load balancer
      uri:
        url: "http://lb.example.com/add/{{ inventory_hostname }}"
      delegate_to: localhost
```

## Rolling Updates

```yaml
- name: Rolling update
  hosts: webservers
  serial: "25%"  # Update 25% at a time
  max_fail_percentage: 10
  
  tasks:
    - name: Update application
      apt:
        name: myapp
        state: latest
      notify: restart app
```

## Monitoring

```yaml
- name: Track execution time
  hosts: all
  tasks:
    - name: Start timer
      set_fact:
        start_time: "{{ ansible_date_time.epoch }}"
    
    - name: Long running task
      command: /usr/bin/long_task
    
    - name: Calculate duration
      debug:
        msg: "Duration: {{ ansible_date_time.epoch | int - start_time | int }}s"
```

## Success Criteria
✅ Large inventory managed  
✅ Delegation working  
✅ Rolling updates functional  
✅ Execution monitored  

**Time:** 45 min
