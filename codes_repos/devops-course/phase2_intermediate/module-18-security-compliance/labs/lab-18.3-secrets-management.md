# Lab 18.3: Secrets Management

## Objective
Securely manage secrets using HashiCorp Vault.

## Learning Objectives
- Set up Vault server
- Store and retrieve secrets
- Use dynamic secrets
- Implement secret rotation

---

## Vault Setup

```bash
# Start Vault dev server
vault server -dev

# Set environment
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='root'

# Enable KV secrets engine
vault secrets enable -path=secret kv-v2
```

## Store Secrets

```bash
# Write secret
vault kv put secret/database username=admin password=secret123

# Read secret
vault kv get secret/database

# Read JSON
vault kv get -format=json secret/database | jq -r .data.data.password
```

## Dynamic Secrets

```bash
# Enable database secrets
vault secrets enable database

# Configure PostgreSQL
vault write database/config/postgresql \
  plugin_name=postgresql-database-plugin \
  allowed_roles="readonly" \
  connection_url="postgresql://{{username}}:{{password}}@localhost:5432/mydb" \
  username="vault" \
  password="vaultpass"

# Create role
vault write database/roles/readonly \
  db_name=postgresql \
  creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
  default_ttl="1h" \
  max_ttl="24h"

# Get dynamic credentials
vault read database/creds/readonly
```

## Application Integration

```python
import hvac

client = hvac.Client(url='http://127.0.0.1:8200', token='root')

# Read secret
secret = client.secrets.kv.v2.read_secret_version(path='database')
password = secret['data']['data']['password']

# Use dynamic secret
db_creds = client.read('database/creds/readonly')
username = db_creds['data']['username']
```

## Success Criteria
✅ Vault server running  
✅ Static secrets stored  
✅ Dynamic secrets generated  
✅ Application integrated  

**Time:** 50 min
