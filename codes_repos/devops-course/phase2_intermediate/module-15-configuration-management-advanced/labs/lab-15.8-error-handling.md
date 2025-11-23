# Lab 15.8: Ansible Error Handling

## Objective
Implement robust error handling in Ansible playbooks.

## Learning Objectives
- Use block/rescue/always
- Handle failures gracefully
- Implement retries
- Debug playbook issues

---

## Block/Rescue/Always

```yaml
- name: Error handling example
  block:
    - name: Risky task
      command: /usr/bin/risky_command
      register: result
    
    - name: Process result
      debug:
        msg: "Success: {{ result.stdout }}"
  
  rescue:
    - name: Handle error
      debug:
        msg: "Task failed, running recovery"
    
    - name: Recovery action
      command: /usr/bin/recovery_script
  
  always:
    - name: Cleanup
      file:
        path: /tmp/tempfile
        state: absent
```

## Ignore Errors

```yaml
- name: Task that might fail
  command: /usr/bin/might_fail
  ignore_errors: yes
  register: result

- name: Continue only if succeeded
  debug:
    msg: "Previous task succeeded"
  when: result is succeeded
```

## Retries

```yaml
- name: Retry on failure
  uri:
    url: http://api.example.com
    status_code: 200
  register: result
  until: result.status == 200
  retries: 5
  delay: 10
```

## Failed When

```yaml
- name: Custom failure condition
  command: /usr/bin/check_status
  register: result
  failed_when: "'ERROR' in result.stdout"
```

## Success Criteria
✅ Error handling working  
✅ Retries functional  
✅ Custom failure conditions  
✅ Cleanup always runs  

**Time:** 35 min
