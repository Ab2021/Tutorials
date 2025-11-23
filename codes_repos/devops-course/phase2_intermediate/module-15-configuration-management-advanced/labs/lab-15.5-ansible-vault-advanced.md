# Lab 15.5: Ansible Vault Advanced

## Objective
Secure sensitive data with Ansible Vault.

## Learning Objectives
- Encrypt variables and files
- Use vault IDs
- Integrate with CI/CD
- Rotate vault passwords

---

## Encrypt Variables

```bash
# Encrypt string
ansible-vault encrypt_string 'secret123' --name 'db_password'

# Use in playbook
vars:
  db_password: !vault |
    $ANSIBLE_VAULT;1.1;AES256
    ...
```

## Encrypt Files

```bash
# Encrypt file
ansible-vault encrypt secrets.yml

# Edit encrypted file
ansible-vault edit secrets.yml

# Decrypt file
ansible-vault decrypt secrets.yml

# View encrypted file
ansible-vault view secrets.yml
```

## Vault IDs

```bash
# Create with vault ID
ansible-vault encrypt --vault-id prod@prompt secrets.yml

# Use multiple vaults
ansible-playbook site.yml \
  --vault-id dev@dev-password \
  --vault-id prod@prod-password
```

## CI/CD Integration

```yaml
# .github/workflows/deploy.yaml
- name: Run Ansible
  env:
    ANSIBLE_VAULT_PASSWORD: ${{ secrets.VAULT_PASSWORD }}
  run: |
    echo "$ANSIBLE_VAULT_PASSWORD" > .vault_pass
    ansible-playbook site.yml --vault-password-file .vault_pass
    rm .vault_pass
```

## Success Criteria
✅ Secrets encrypted  
✅ Vault IDs working  
✅ CI/CD integration functional  

**Time:** 40 min
