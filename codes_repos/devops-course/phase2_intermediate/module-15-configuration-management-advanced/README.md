# Advanced Configuration Management with Ansible

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of advanced Ansible patterns, including:
- **Dynamic Inventory**: Automatically discovering hosts from Cloud Providers (AWS, Azure).
- **Security**: Encrypting sensitive data with **Ansible Vault**.
- **Scalability**: Optimizing playbook execution with Pipelining, Forks, and Async.
- **Extensibility**: Writing custom modules in Python.
- **Enterprise**: Managing complex workflows with AWX/Tower.

---

## 📖 Theoretical Concepts

### 1. Dynamic Inventory

In the cloud, servers come and go. A static `hosts.ini` file is obsolete the moment you write it.
- **Plugins**: Ansible uses plugins (e.g., `aws_ec2`) to query the Cloud API.
- **Grouping**: Automatically group hosts by tags (e.g., `tag_Role_Webserver`).

### 2. Ansible Vault

Never commit plain-text passwords to Git.
- **Encryption**: Vault encrypts variables or entire files using AES-256.
- **Usage**: You need a password (or key file) to run the playbook.
- **Best Practice**: Encrypt *only* the specific variables that are sensitive, not the whole playbook.

### 3. Optimization Strategies

- **Forks**: By default, Ansible talks to 5 hosts at a time. Increase this (`-f 50`) to speed up large fleets.
- **Pipelining**: Reduces the number of SSH connections required.
- **Async**: Fire-and-forget long-running tasks (e.g., Database migration) to avoid blocking the playbook.
- **Fact Caching**: Cache gathered facts (OS, IP) to Redis so you don't have to gather them every run.

### 4. Custom Modules

If a task is too complex for shell commands or existing modules, write your own in Python.
- **Input**: JSON arguments from Ansible.
- **Output**: JSON return values (changed, failed, msg).

---

## 🔧 Practical Examples

### Dynamic Inventory (`aws_ec2.yml`)

```yaml
plugin: aws_ec2
regions:
  - us-east-1
filters:
  tag:Environment: Production
keyed_groups:
  - key: tags.Role
    prefix: role
```

### Using Ansible Vault

```bash
# Encrypt a string
ansible-vault encrypt_string 's3cr3tp@ssword' --name 'db_password'

# Run playbook
ansible-playbook site.yml --ask-vault-pass
```

### Async Task

```yaml
- name: Long running upgrade
  apt:
    name: "*"
    state: latest
  async: 3600  # Wait up to 1 hour
  poll: 0      # Don't wait, move to next task
  register: upgrade_job
```

### Custom Module Structure (Python)

```python
from ansible.module_utils.basic import AnsibleModule

def run_module():
    module = AnsibleModule(
        argument_spec=dict(
            name=dict(type='str', required=True),
        )
    )
    
    # Logic here...
    
    module.exit_json(changed=True, message="Success")

if __name__ == '__main__':
    run_module()
```

---

## 🎯 Hands-on Labs

- [Lab 15.1: Ansible Dynamic Inventory (AWS)](./labs/lab-15.1-dynamic-inventory.md)
- [Lab 15.2: Ansible Vault](./labs/lab-15.2-ansible-vault.md)
- [Lab 15.3: Ansible Galaxy](./labs/lab-15.3-ansible-galaxy.md)
- [Lab 15.4: Dynamic Inventory](./labs/lab-15.4-dynamic-inventory.md)
- [Lab 15.5: Custom Modules](./labs/lab-15.5-custom-modules.md)
- [Lab 15.6: Ansible Tower Intro](./labs/lab-15.6-ansible-tower-intro.md)
- [Lab 15.7: Callback Plugins](./labs/lab-15.7-callback-plugins.md)
- [Lab 15.8: Error Handling](./labs/lab-15.8-error-handling.md)
- [Lab 15.9: Ansible Testing](./labs/lab-15.9-ansible-testing.md)
- [Lab 15.10: Enterprise Automation](./labs/lab-15.10-enterprise-automation.md)

---

## 📚 Additional Resources

### Official Documentation
- [Ansible Vault](https://docs.ansible.com/ansible/latest/user_guide/vault.html)
- [Developing Modules](https://docs.ansible.com/ansible/latest/dev_guide/developing_modules_general.html)

### Tools
- [Molecule](https://molecule.readthedocs.io/) - Testing framework for Ansible roles.
- [Ansible Lint](https://ansible-lint.readthedocs.io/) - Best practices linter.

---

## 🔑 Key Takeaways

1.  **Don't Hardcode IPs**: Use Dynamic Inventory.
2.  **Secrets Management**: Use Vault, or better yet, integrate with external secret managers (HashiCorp Vault).
3.  **Speed**: Tuning `forks` and `pipelining` can make your playbooks 10x faster.
4.  **Community**: Check Ansible Galaxy before writing a role from scratch.

---

## ⏭️ Next Steps

1.  Complete the labs to master advanced automation.
2.  Proceed to **[Module 16: Monitoring & Observability](../module-16-monitoring-observability/README.md)** to gain deep visibility into your automated infrastructure.
