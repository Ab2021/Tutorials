# Lab 15.7: Ansible Testing

## Objective
Test Ansible playbooks and roles.

## Learning Objectives
- Use Molecule for testing
- Write test scenarios
- Implement CI/CD testing
- Validate idempotence

---

## Molecule Setup

```bash
# Install
pip install molecule molecule-docker

# Initialize
molecule init role my-role

# Directory structure
my-role/
├── molecule/
│   └── default/
│       ├── converge.yml
│       ├── molecule.yml
│       └── verify.yml
```

## Molecule Config

```yaml
# molecule/default/molecule.yml
driver:
  name: docker
platforms:
  - name: instance
    image: ubuntu:22.04
provisioner:
  name: ansible
verifier:
  name: ansible
```

## Test Playbook

```yaml
# molecule/default/verify.yml
- name: Verify
  hosts: all
  tasks:
    - name: Check nginx is running
      service:
        name: nginx
        state: started
      check_mode: yes
      register: result
      failed_when: result.changed
```

## Run Tests

```bash
# Full test sequence
molecule test

# Individual steps
molecule create
molecule converge
molecule verify
molecule destroy
```

## Success Criteria
✅ Molecule configured  
✅ Tests passing  
✅ Idempotence verified  

**Time:** 45 min
