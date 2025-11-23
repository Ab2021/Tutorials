# Lab 18.4: Vault Setup

## Objective
Deploy and configure HashiCorp Vault in production mode.

## Learning Objectives
- Deploy Vault with high availability
- Configure auto-unseal
- Set up authentication methods
- Implement policies

---

## Production Deployment

```hcl
# vault.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  tls_cert_file = "/etc/vault/tls/vault.crt"
  tls_key_file  = "/etc/vault/tls/vault.key"
}

seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "alias/vault-unseal"
}

api_addr = "https://vault.example.com:8200"
cluster_addr = "https://vault.example.com:8201"
ui = true
```

## Initialize Vault

```bash
vault operator init -key-shares=5 -key-threshold=3

# Unseal (3 times with different keys)
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# Login
vault login <root-token>
```

## Authentication

```bash
# Enable GitHub auth
vault auth enable github

# Configure
vault write auth/github/config organization=myorg

# Map team to policy
vault write auth/github/map/teams/devops value=devops-policy

# Login
vault login -method=github token=<github-token>
```

## Policies

```hcl
# devops-policy.hcl
path "secret/data/devops/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "database/creds/readonly" {
  capabilities = ["read"]
}
```

```bash
vault policy write devops-policy devops-policy.hcl
```

## Success Criteria
✅ Vault deployed in HA mode  
✅ Auto-unseal configured  
✅ Authentication methods enabled  
✅ Policies enforced  

**Time:** 50 min
