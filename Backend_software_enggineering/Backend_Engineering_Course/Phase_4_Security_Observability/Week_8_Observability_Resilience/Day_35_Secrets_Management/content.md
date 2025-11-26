# Day 35: Secrets Management - Securing Credentials

## Table of Contents
1. [Secrets Management Fundamentals](#1-secrets-management-fundamentals)
2. [Environment Variables](#2-environment-variables)
3. [HashiCorp Vault](#3-hashicorp-vault)
4. [AWS Secrets Manager](#4-aws-secrets-manager)
5. [Secret Rotation](#5-secret-rotation)
6. [Secret Scanning](#6-secret-scanning)
7. [Production Patterns](#7-production-patterns)
8. [Summary](#8-summary)

---

## 1. Secrets Management Fundamentals

### 1.1 What are Secrets?

- API keys
- Database passwords
- Encryption keys
- OAuth client secrets
- TLS certificates

### 1.2 Common Mistakes

❌ **Hardcoded secrets**:
```python
DB_PASSWORD = "super_secret_password"  # NEVER!
```

❌ **Committed to Git**:
```python
# config.py (in Git repo)
API_KEY = "sk_live_abc123..."  # NEVER!
```

❌ **Plaintext in config files**:
```yaml
# config.yaml
database:
  password: "plaintext_password"  # NEVER!
```

---

## 2. Environment Variables

### 2.1 Basic Usage

```python
import os

DB_PASSWORD = os.getenv('DB_PASSWORD')
API_KEY = os.getenv('API_KEY')

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set")
```

### 2.2 .env Files (Development)

**.env** (NOT committed to Git):
```
DB_PASSWORD=dev_password
API_KEY=dev_api_key
```

**.gitignore**:
```
.env
```

**Load with python-dotenv**:
```python
from dotenv import load_dotenv

load_dotenv()  # Loads .env file

DB_PASSWORD = os.getenv('DB_PASSWORD')
```

### 2.3 Production Deployment

**Docker**:
```bash
docker run -e DB_PASSWORD=secret_password myapp
```

**Kubernetes**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  DB_PASSWORD: c2VjcmV0X3Bhc3N3b3Jk  # base64 encoded

---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: DB_PASSWORD
```

---

## 3. HashiCorp Vault

### 3.1 What is Vault?

**Vault**: Centralized secrets management platform.

**Features**:
- Dynamic secrets
- Encryption as a service
- Fine-grained access control
- Audit logging

### 3.2 Installation

```bash
# Download and install
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Start dev server (NOT for production!)
vault server -dev
```

### 3.3 Storing Secrets

```bash
# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'

# Login (dev mode uses root token)
vault login <root_token>

# Write secret
vault kv put secret/myapp/db password=super_secret_password

# Read secret
vault kv get secret/myapp/db

# Output:
# ====== Data ======
# Key       Value
# ---       -----
# password  super_secret_password
```

### 3.4 Python Integration

```python
import hvac

# Initialize client
client = hvac.Client(url='http://127.0.0.1:8200', token='root_token')

# Read secret
secret = client.secrets.kv.v2.read_secret_version(path='myapp/db')
db_password = secret['data']['data']['password']

# Write secret
client.secrets.kv.v2.create_or_update_secret(
    path='myapp/api',
    secret={'api_key': 'sk_live_abc123'}
)
```

### 3.5 Dynamic Database Credentials

```bash
# Enable database secrets engine
vault secrets enable database

# Configure PostgreSQL connection
vault write database/config/postgresql \
    plugin_name=postgresql-database-plugin \
    allowed_roles="readonly" \
    connection_url="postgresql://{{username}}:{{password}}@localhost:5432/mydb" \
    username="vault" \
    password="vault_password"

# Create role with TTL
vault write database/roles/readonly \
    db_name=postgresql \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    default_ttl="1h" \
    max_ttl="24h"

# Generate credentials (automatically deleted after 1 hour!)
vault read database/creds/readonly

# Output:
# Key            Value
# ---            -----
# lease_id       database/creds/readonly/abc123...
# username       v-root-readonly-xyz789
# password       A1B2C3D4E5...
```

**Python usage**:
```python
# Get dynamic credentials
creds = client.read('database/creds/readonly')
db_username = creds['data']['username']
db_password = creds['data']['password']

# Credentials expire automatically after 1 hour
# Vault rotates them automatically!
```

---

## 4. AWS Secrets Manager

### 4.1 Creating Secrets

```bash
aws secretsmanager create-secret \
    --name myapp/db/password \
    --secret-string "super_secret_password"
```

### 4.2 Retrieving Secrets

```python
import boto3
import json

secrets_client = boto3.client('secretsmanager')

# Get secret
response = secrets_client.get_secret_value(SecretId='myapp/db/password')
db_password = response['SecretString']

# For JSON secrets
response = secrets_client.get_secret_value(SecretId='myapp/credentials')
secrets = json.loads(response['SecretString'])
api_key = secrets['api_key']
```

### 4.3 Automatic Rotation

```python
# Lambda function for rotation
import boto3

def lambda_handler(event, context):
    service_client = boto3.client('secretsmanager')
    
    # Get current secret
    current_secret = service_client.get_secret_value(
        SecretId=event['SecretId'],
        VersionStage='AWSCURRENT'
    )
    
    # Generate new password
    new_password = generate_random_password()
    
    # Update database with new password
    update_database_password(new_password)
    
    # Store new secret
    service_client.put_secret_value(
        SecretId=event['SecretId'],
        SecretString=new_password,
        VersionStages=['AWSPENDING']
    )
    
    # Mark as current
    service_client.update_secret_version_stage(
        SecretId=event['SecretId'],
        VersionStage='AWSCURRENT',
        MoveToVersionId=event['VersionId']
    )
```

---

## 5. Secret Rotation

### 5.1 Why Rotate?

- Limit damage if secret is leaked
- Compliance requirements
- Defense in depth

### 5.2 Manual Rotation

```python
import secrets

def rotate_api_key(user_id):
    # Generate new API key
    new_key = f"sk_live_{secrets.token_urlsafe(32)}"
    
    # Store in database
    db.execute(
        "UPDATE users SET api_key = ?, api_key_created_at = ? WHERE id = ?",
        (new_key, datetime.utcnow(), user_id)
    )
    
    # Invalidate old key after grace period (7 days)
    schedule_invalidation(user_id, days=7)
    
    return new_key
```

### 5.3 Automatic Rotation

```python
from apscheduler.schedulers.background import BackgroundScheduler

def rotate_all_secrets():
    # Rotate database password
    new_db_password = rotate_database_password()
    vault_client.write('secret/db/password', password=new_db_password)
    
    # Rotate API keys
    for user in User.query.filter(User.api_key_age > 90).all():
        rotate_api_key(user.id)

# Schedule rotation every 30 days
scheduler = BackgroundScheduler()
scheduler.add_job(rotate_all_secrets, 'interval', days=30)
scheduler.start()
```

---

## 6. Secret Scanning

### 6.1 git-secrets

```bash
# Install
git clone https://github.com/awslabs/git-secrets.git
cd git-secrets
sudo make install

# Setup
cd /path/to/your/repo
git secrets --install
git secrets --register-aws

# Scan
git secrets --scan
```

### 6.2 TruffleHog

```bash
# Install
pip install truffleHog

# Scan repo
trufflehog https://github.com/your-org/your-repo
```

### 6.3 Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/sh

# Scan for secrets
git secrets --pre_commit_hook -- "$@"

if [ $? -ne 0 ]; then
    echo "Commit rejected: secrets detected!"
    exit 1
fi
```

---

## 7. Production Patterns

### 7.1 Secret Injection at Runtime

```python
# Don't store secrets in config files
# Inject at runtime from Vault/AWS Secrets Manager

def get_db_connection():
    # Fetch secret at runtime
    db_password = vault_client.read('secret/db/password')['data']['password']
    
    return psycopg2.connect(
        host='db.example.com',
        user='app_user',
        password=db_password,
        database='mydb'
    )
```

### 7.2 Secret Caching

```python
import time

class SecretCache:
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
        
        # Fetch from Vault
        value = vault_client.read(key)['data']['password']
        self.cache[key] = (value, time.time())
        return value

secret_cache = SecretCache(ttl=300)  # Cache for 5 minutes

# Usage
db_password = secret_cache.get('secret/db/password')
```

### 7.3 Least Privilege Access

```bash
# Vault policy (restrict access)
vault policy write app-policy - <<EOF
path "secret/data/myapp/*" {
  capabilities = ["read"]
}
EOF

# Create app token with limited policy
vault token create -policy=app-policy
```

---

## 8. Summary

### 8.1 Key Takeaways

1. ✅ **Never hardcode** secrets in code
2. ✅ **Environment variables** for simple cases
3. ✅ **HashiCorp Vault** for dynamic secrets
4. ✅ **AWS Secrets Manager** for automatic rotation
5. ✅ **Secret scanning** to prevent leaks
6. ✅ **Rotate regularly** (30-90 days)

### 8.2 Secrets Management Comparison

| Solution | Complexity | Features | Cost |
|:---------|:-----------|:---------|:-----|
| **Env Vars** | Low | Basic | Free |
| **Kubernetes Secrets** | Medium | Integration | Free |
| **Vault** | High | Dynamic, encryption-as-a-service | Self-hosted |
| **AWS Secrets Manager** | Medium | Automatic rotation | $0.40/secret/month |

### 8.3 Tomorrow (Day 36): Testing Strategies

- **Unit testing**: pytest, mocking
- **Integration testing**: Database, API tests
- **E2E testing**: Playwright, Selenium
- **Test coverage**: pytest-cov
- **TDD**: Test-driven development
- **Production patterns**: CI/CD integration

See you tomorrow! 🚀

---

**File Statistics**: ~900 lines | Secrets Management mastered ✅
