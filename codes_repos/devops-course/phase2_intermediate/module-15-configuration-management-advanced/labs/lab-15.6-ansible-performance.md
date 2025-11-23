# Lab 15.6: Ansible Performance

## Objective
Optimize Ansible playbook performance.

## Learning Objectives
- Use async and poll
- Implement forks
- Enable pipelining
- Optimize fact gathering

---

## Async Tasks

```yaml
- name: Long running task
  command: /usr/bin/long_task
  async: 3600  # Run for up to 1 hour
  poll: 0      # Fire and forget
  register: long_task

- name: Check status later
  async_status:
    jid: "{{ long_task.ansible_job_id }}"
  register: job_result
  until: job_result.finished
  retries: 30
  delay: 60
```

## Forks

```ini
# ansible.cfg
[defaults]
forks = 50  # Run on 50 hosts simultaneously
```

## Pipelining

```ini
[ssh_connection]
pipelining = True
```

## Fact Caching

```ini
[defaults]
gathering = smart
fact_caching = jsonfile
fact_caching_connection = /tmp/ansible_facts
fact_caching_timeout = 86400
```

## Optimize Playbook

```yaml
- hosts: all
  gather_facts: no  # Skip if not needed
  
  tasks:
    - name: Gather minimal facts
      setup:
        gather_subset:
          - '!all'
          - '!min'
          - network
```

## Success Criteria
✅ Async tasks working  
✅ Forks optimized  
✅ Pipelining enabled  
✅ Playbook runs faster  

**Time:** 35 min
