# Day 2.4: Secret Management - Vault, AWS KMS, and Azure Key Vault Integration

## 🎯 Learning Objectives
By the end of this section, you will understand:
- Secret management fundamentals and security principles
- HashiCorp Vault architecture and implementation patterns
- AWS Key Management Service (KMS) and Secrets Manager integration
- Azure Key Vault and managed identity workflows
- Automated secret rotation and lifecycle management
- Integration with AI/ML pipelines and infrastructure

---

## 📚 Theoretical Foundation

### 1. Introduction to Secret Management

#### 1.1 Secret Management Fundamentals

**Definition and Scope**:
Secret management encompasses the secure storage, distribution, rotation, and lifecycle management of sensitive information such as passwords, API keys, certificates, and encryption keys. In AI/ML environments, this extends to model weights, training data access credentials, and service-to-service authentication tokens.

**Critical Security Principles**:
```
Principle of Least Privilege:
- Grant minimal access required for specific operations
- Time-limited access with automatic expiration
- Role-based access control with fine-grained permissions
- Regular access reviews and automated cleanup

Defense in Depth:
- Multiple layers of protection for sensitive data
- Encryption at rest, in transit, and in use
- Network isolation and access controls
- Comprehensive audit logging and monitoring

Zero Trust Architecture:
- Never trust, always verify principle
- Continuous authentication and authorization
- Dynamic policy enforcement based on context
- Assume breach and limit blast radius
```

**AI/ML Specific Secret Management Challenges**:
```
Scale and Complexity:
- Thousands of training jobs with unique credentials
- Distributed inference endpoints requiring authentication
- Cross-cloud deployments with diverse secret stores
- Automated pipelines needing programmatic access

Performance Requirements:
- Low-latency secret retrieval for real-time inference
- High-throughput access for distributed training
- Minimal overhead in critical training loops
- Efficient caching and local secret storage

Compliance and Governance:
- Regulatory requirements for data protection
- Audit trails for all secret access operations
- Geographic restrictions and data sovereignty
- Integration with enterprise identity systems
```

#### 1.2 Secret Lifecycle Management

**Secret Lifecycle Phases**:
```
1. Generation/Creation:
   - Cryptographically secure random generation
   - Compliance with complexity requirements
   - Initial classification and metadata assignment
   - Secure initial distribution to authorized systems

2. Storage and Protection:
   - Encrypted storage with appropriate key management
   - Access control and authorization mechanisms
   - Backup and disaster recovery procedures
   - Geographic distribution and replication

3. Distribution and Access:
   - Secure delivery to consuming applications
   - Just-in-time access provisioning
   - Context-aware access controls
   - Monitoring and logging of access patterns

4. Rotation and Updates:
   - Automated rotation based on age or events
   - Coordinated updates across dependent systems
   - Graceful handling of rotation failures
   - Validation of successful rotation

5. Revocation and Cleanup:
   - Immediate revocation upon compromise
   - Cleanup of cached or distributed copies
   - Audit of systems that accessed revoked secrets
   - Documentation of revocation events
```

**AI/ML Pipeline Integration Points**:
```
Data Pipeline Secrets:
- Database connection strings for training data
- API keys for external data sources
- Cloud storage access credentials
- Data transformation service authentication

Training Infrastructure Secrets:
- GPU cluster authentication tokens
- Container registry credentials
- Model repository access keys
- Distributed training coordination secrets

Inference Serving Secrets:
- Model serving endpoint certificates
- API gateway authentication tokens
- Database connections for feature stores
- External service integration credentials

Monitoring and Operations:
- Observability platform API keys
- Log aggregation service credentials
- Alerting system authentication tokens
- Backup and disaster recovery credentials
```

### 2. HashiCorp Vault Architecture

#### 2.1 Vault Core Architecture

**Vault Components and Design**:
```
Storage Backend:
- Persistent storage for encrypted data
- Support for various backends (Consul, etcd, S3, etc.)
- High availability and replication capabilities
- Backup and disaster recovery integration

Security Barrier:
- Encryption layer protecting all data at rest
- AES-256-GCM encryption with authenticated encryption
- Automatic key rotation and cryptographic agility
- Protection against storage backend compromise

Authentication Methods:
- Multiple authentication backends (AWS IAM, Kubernetes, LDAP)
- Short-lived tokens with automatic expiration
- Multi-factor authentication support
- Dynamic secret generation capabilities

Secret Engines:
- Pluggable backends for different secret types
- Dynamic secret generation for databases, cloud providers
- Key-value storage for static secrets
- Certificate authority for PKI management
```

**High Availability Architecture**:
```
Multi-Node Deployment:
- Active/standby cluster configuration
- Automatic leader election and failover
- Consistent replication across nodes
- Load balancing for read operations

Disaster Recovery:
- Cross-region replication capabilities
- Automated backup and restore procedures
- Point-in-time recovery options
- Disaster recovery testing automation
```

#### 2.2 Vault Authentication and Authorization

**AWS IAM Authentication**:
```
IAM Authentication Flow:
1. Application assumes IAM role
2. Signs STS GetCallerIdentity request
3. Submits signed request to Vault
4. Vault validates signature with AWS STS
5. Returns Vault token with appropriate policies

Configuration Example:
vault auth enable aws

vault write auth/aws/config/client \
    access_key=AKIAI... \
    secret_key=... \
    region=us-west-2

vault write auth/aws/role/ml-training-role \
    auth_type=iam \
    policies=ml-training-policy \
    max_ttl=1h \
    bound_iam_principal_arn=arn:aws:iam::123456789012:role/ML-Training-Role
```

**Kubernetes Authentication**:
```
Kubernetes Service Account Integration:
1. Pod presents service account JWT token
2. Vault validates token with Kubernetes API
3. Maps service account to Vault policies
4. Returns Vault token for secret access

Configuration:
vault auth enable kubernetes

vault write auth/kubernetes/config \
    token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
    kubernetes_host="https://kubernetes.default.svc.cluster.local:443" \
    kubernetes_ca_cert=@ca.crt

vault write auth/kubernetes/role/ml-training \
    bound_service_account_names=ml-training-sa \
    bound_service_account_namespaces=ml-training \
    policies=ml-training-policy \
    ttl=24h
```

#### 2.3 Dynamic Secret Generation

**Database Secret Engine**:
```
MySQL Dynamic Credentials:
vault secrets enable database

vault write database/config/ml-mysql \
    plugin_name=mysql-database-plugin \
    connection_url="{{username}}:{{password}}@tcp(mysql.example.com:3306)/" \
    allowed_roles="ml-training-role" \
    username="vault-admin" \
    password="admin-password"

vault write database/roles/ml-training-role \
    db_name=ml-mysql \
    creation_statements="CREATE USER '{{name}}'@'%' IDENTIFIED BY '{{password}}';GRANT SELECT,INSERT,UPDATE ON ml_training.* TO '{{name}}'@'%';" \
    default_ttl="1h" \
    max_ttl="24h"

# Generate dynamic credentials
vault read database/creds/ml-training-role
```

**Cloud Provider Integration**:
```
AWS Dynamic Credentials:
vault secrets enable aws

vault write aws/config/root \
    access_key=AKIAI... \
    secret_key=... \
    region=us-west-2

vault write aws/roles/ml-s3-access \
    credential_type=iam_user \
    policy_document=-<<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::ml-training-data/*"]
    }
  ]
}
EOF

# Generate AWS credentials
vault read aws/creds/ml-s3-access
```

#### 2.4 PKI and Certificate Management

**Certificate Authority Setup**:
```
Root CA Configuration:
vault secrets enable pki
vault secrets tune -max-lease-ttl=87600h pki

vault write -field=certificate pki/root/generate/internal \
    common_name="ML Infrastructure Root CA" \
    ttl=87600h > ca_cert.crt

vault write pki/config/urls \
    issuing_certificates="$VAULT_ADDR/v1/pki/ca" \
    crl_distribution_points="$VAULT_ADDR/v1/pki/crl"

# Intermediate CA
vault secrets enable -path=pki_int pki
vault secrets tune -max-lease-ttl=43800h pki_int

vault write -format=json pki_int/intermediate/generate/internal \
    common_name="ML Infrastructure Intermediate Authority" \
    | jq -r '.data.csr' > pki_intermediate.csr

vault write -format=json pki/root/sign-intermediate \
    csr=@pki_intermediate.csr \
    format=pem_bundle ttl="43800h" \
    | jq -r '.data.certificate' > intermediate.cert.pem

# Role for ML services
vault write pki_int/roles/ml-services \
    allowed_domains="ml.example.com" \
    allow_subdomains=true \
    max_ttl="720h"
```

### 3. AWS Key Management Service Integration

#### 3.1 AWS KMS Architecture

**KMS Key Types and Usage**:
```
Customer Managed Keys (CMK):
- Full control over key policies and usage
- Custom rotation schedules (annual automatic)
- Cross-account access capabilities
- Detailed CloudTrail logging

AWS Managed Keys:
- Service-specific keys managed by AWS
- Automatic rotation every year
- No key policy modification allowed
- Integrated with AWS services

Data Keys:
- Symmetric encryption keys for local data encryption
- Generated by KMS but used outside of KMS
- Envelope encryption pattern
- Automatic key hierarchy management
```

**KMS Key Policies**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:root"
      },
      "Action": "kms:*",
      "Resource": "*"
    },
    {
      "Sid": "ML Training Access",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::123456789012:role/ML-Training-Role"
        ]
      },
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey",
        "kms:CreateGrant"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": [
            "s3.us-west-2.amazonaws.com",
            "secretsmanager.us-west-2.amazonaws.com"
          ]
        },
        "StringLike": {
          "kms:EncryptionContext:SecretARN": "arn:aws:secretsmanager:us-west-2:123456789012:secret:ml-*"
        }
      }
    }
  ]
}
```

#### 3.2 AWS Secrets Manager Integration

**Secret Configuration and Rotation**:
```terraform
# KMS key for Secrets Manager
resource "aws_kms_key" "secrets_manager" {
  description             = "KMS key for ML Secrets Manager"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Purpose = "secrets-encryption"
  })
}

resource "aws_kms_alias" "secrets_manager" {
  name          = "alias/ml-secrets-manager"
  target_key_id = aws_kms_key.secrets_manager.key_id
}

# Database credentials with automatic rotation
resource "aws_secretsmanager_secret" "ml_database" {
  name                    = "ml-database-credentials"
  description             = "Database credentials for ML training"
  kms_key_id             = aws_kms_key.secrets_manager.arn
  recovery_window_in_days = 30
  
  replica {
    region = "us-east-1"
    kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/backup-key-id"
  }
  
  tags = merge(local.common_tags, {
    SecretType = "database-credentials"
  })
}

resource "aws_secretsmanager_secret_version" "ml_database" {
  secret_id = aws_secretsmanager_secret.ml_database.id
  secret_string = jsonencode({
    username = "ml_admin"
    password = random_password.db_password.result
    host     = aws_rds_instance.ml_database.endpoint
    port     = aws_rds_instance.ml_database.port
    dbname   = aws_rds_instance.ml_database.db_name
  })
}

# Automatic rotation configuration
resource "aws_secretsmanager_secret_rotation" "ml_database" {
  secret_id           = aws_secretsmanager_secret.ml_database.id
  rotation_lambda_arn = aws_lambda_function.rotate_secret.arn
  
  rotation_rules {
    automatically_after_days = 30
  }
}

# Lambda function for secret rotation
resource "aws_lambda_function" "rotate_secret" {
  filename         = "rotate_secret.zip"
  function_name    = "ml-secret-rotation"
  role            = aws_iam_role.rotation_lambda.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 300
  
  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.us-west-2.amazonaws.com"
    }
  }
  
  tags = local.common_tags
}
```

#### 3.3 Parameter Store Integration

**Hierarchical Parameter Organization**:
```terraform
# Application configuration parameters
resource "aws_ssm_parameter" "ml_config" {
  for_each = {
    "/ml/training/batch_size"           = "32"
    "/ml/training/learning_rate"        = "0.001"
    "/ml/training/epochs"               = "100"
    "/ml/model/checkpoint_interval"     = "1000"
    "/ml/storage/model_bucket"          = aws_s3_bucket.models.bucket
    "/ml/storage/data_bucket"           = aws_s3_bucket.training_data.bucket
  }
  
  name  = each.key
  type  = "String"
  value = each.value
  
  tags = merge(local.common_tags, {
    ParameterType = "configuration"
  })
}

# Secure parameters with encryption
resource "aws_ssm_parameter" "ml_secrets" {
  for_each = {
    "/ml/secrets/api_key"        = var.external_api_key
    "/ml/secrets/webhook_token"  = random_password.webhook_token.result
    "/ml/secrets/jwt_secret"     = random_password.jwt_secret.result
  }
  
  name   = each.key
  type   = "SecureString"
  value  = each.value
  key_id = aws_kms_key.secrets_manager.arn
  
  tags = merge(local.common_tags, {
    ParameterType = "secret"
  })
}

# Parameter with advanced tier for large values
resource "aws_ssm_parameter" "ml_model_config" {
  name  = "/ml/model/large_config"
  type  = "String"
  value = file("${path.module}/configs/large_model_config.json")
  tier  = "Advanced"
  
  tags = merge(local.common_tags, {
    ParameterType = "model-configuration"
  })
}
```

### 4. Azure Key Vault Integration

#### 4.1 Azure Key Vault Architecture

**Key Vault Types and Features**:
```
Standard Tier:
- Software-protected keys and secrets
- Cost-effective for most use cases
- Shared infrastructure with security isolation
- Suitable for development and testing

Premium Tier:
- Hardware Security Module (HSM) backed keys
- FIPS 140-2 Level 2 validated HSMs
- Enhanced security for production workloads
- Compliance with strict regulatory requirements

Managed HSM:
- Dedicated HSM pools for customers
- Full administrative control
- Role-based access control (RBAC)
- Air-gapped key management
```

**Access Control Models**:
```
Vault Access Policies (Classic):
- Object-level permissions
- Principal-based access control
- Compatible with existing Azure AD users/groups
- Granular permissions per secret/key/certificate

Azure RBAC (Recommended):
- Unified access management across Azure
- Integration with Azure AD Privileged Identity Management
- Conditional access policies
- Centralized audit and compliance
```

#### 4.2 Managed Identity Integration

**System-Assigned Managed Identity**:
```terraform
# Virtual Machine with managed identity
resource "azurerm_linux_virtual_machine" "ml_training" {
  name                = "ml-training-vm"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = azurerm_resource_group.ml_rg.location
  size                = "Standard_NC6s_v3"
  
  # Enable system-assigned managed identity
  identity {
    type = "SystemAssigned"
  }
  
  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }
  
  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-focal"
    sku       = "20_04-lts-gen2"
    version   = "latest"
  }
  
  tags = var.common_tags
}

# Key Vault access policy for managed identity
resource "azurerm_key_vault_access_policy" "ml_training_vm" {
  key_vault_id = azurerm_key_vault.ml_vault.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_linux_virtual_machine.ml_training.identity[0].principal_id
  
  secret_permissions = [
    "Get",
    "List"
  ]
  
  key_permissions = [
    "Get",
    "Decrypt",
    "Unwrap"
  ]
}
```

**User-Assigned Managed Identity**:
```terraform
# User-assigned managed identity for ML workloads
resource "azurerm_user_assigned_identity" "ml_workload" {
  name                = "ml-workload-identity"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = azurerm_resource_group.ml_rg.location
  
  tags = var.common_tags
}

# Azure ML workspace with user-assigned identity
resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = "ml-workspace"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = azurerm_resource_group.ml_rg.location
  
  application_insights_id = azurerm_application_insights.ml_insights.id
  key_vault_id           = azurerm_key_vault.ml_vault.id
  storage_account_id     = azurerm_storage_account.ml_storage.id
  
  identity {
    type = "UserAssigned"
    identity_ids = [
      azurerm_user_assigned_identity.ml_workload.id
    ]
  }
  
  encryption {
    user_assigned_identity_id = azurerm_user_assigned_identity.ml_workload.id
    key_vault_key_id         = azurerm_key_vault_key.ml_encryption.id
  }
  
  tags = var.common_tags
}
```

#### 4.3 Key Vault Secret Management

**Secret Configuration and Access**:
```terraform
# Key Vault configuration
resource "azurerm_key_vault" "ml_vault" {
  name                = "ml-keyvault-${random_string.suffix.result}"
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  
  sku_name = "premium"
  
  # Security configurations
  enabled_for_deployment          = false
  enabled_for_disk_encryption     = true
  enabled_for_template_deployment = false
  enable_rbac_authorization       = true
  
  purge_protection_enabled   = true
  soft_delete_retention_days = 90
  
  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    
    ip_rules = var.allowed_ip_ranges
    
    virtual_network_subnet_ids = [
      azurerm_subnet.ml_training.id
    ]
  }
  
  tags = var.common_tags
}

# Customer-managed encryption key
resource "azurerm_key_vault_key" "ml_encryption" {
  name         = "ml-encryption-key"
  key_vault_id = azurerm_key_vault.ml_vault.id
  key_type     = "RSA"
  key_size     = 2048
  
  key_opts = [
    "decrypt",
    "encrypt",
    "sign",
    "unwrapKey",
    "verify",
    "wrapKey",
  ]
  
  rotation_policy {
    automatic {
      time_before_expiry = "P30D"
    }
    
    expire_after         = "P90D"
    notify_before_expiry = "P29D"
  }
  
  tags = var.common_tags
}

# ML training secrets
resource "azurerm_key_vault_secret" "ml_database_connection" {
  name         = "ml-database-connection"
  value        = "Server=${azurerm_postgresql_server.ml_db.fqdn};Database=${azurerm_postgresql_database.ml_db.name};User Id=${azurerm_postgresql_server.ml_db.administrator_login};Password=${random_password.db_password.result};"
  key_vault_id = azurerm_key_vault.ml_vault.id
  
  expiration_date = timeadd(timestamp(), "8760h") # 1 year
  content_type    = "database-connection-string"
  
  tags = {
    SecretType  = "database-credentials"
    Environment = var.environment
  }
}

resource "azurerm_key_vault_secret" "ml_api_keys" {
  for_each = {
    "external-api-key"     = var.external_api_key
    "model-registry-token" = random_password.model_registry_token.result
    "monitoring-api-key"   = random_password.monitoring_api_key.result
  }
  
  name         = each.key
  value        = each.value
  key_vault_id = azurerm_key_vault.ml_vault.id
  
  expiration_date = timeadd(timestamp(), "2160h") # 90 days
  content_type    = "api-key"
  
  tags = {
    SecretType  = "api-credentials"
    Environment = var.environment
  }
}
```

### 5. Advanced Secret Management Patterns

#### 5.1 Secret Injection and Distribution

**Kubernetes Secret Injection**:
```yaml
# Vault Secrets Operator configuration
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultStaticSecret
metadata:
  name: ml-database-credentials
  namespace: ml-training
spec:
  type: kv-v2
  mount: secret
  path: ml-training/database
  destination:
    name: ml-db-secret
    create: true
  refreshAfter: 3600s
  vaultAuthRef: ml-training-auth

---
# Kubernetes Service Account with Vault authentication
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-training-sa
  namespace: ml-training
  annotations:
    vault.hashicorp.com/auth-path: "auth/kubernetes"
    vault.hashicorp.com/role: "ml-training"

---
# Pod using Vault secrets
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
  namespace: ml-training
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "ml-training"
    vault.hashicorp.com/agent-inject-secret-database: "secret/ml-training/database"
    vault.hashicorp.com/agent-inject-template-database: |
      {{- with secret "secret/ml-training/database" -}}
      export DATABASE_URL="postgresql://{{ .Data.data.username }}:{{ .Data.data.password }}@{{ .Data.data.host }}:{{ .Data.data.port }}/{{ .Data.data.database }}"
      {{- end -}}
spec:
  serviceAccountName: ml-training-sa
  containers:
  - name: ml-training
    image: ml-training:latest
    command: ["/bin/sh"]
    args: ["-c", "source /vault/secrets/database && ./train_model.sh"]
    env:
    - name: VAULT_ADDR
      value: "http://vault.vault.svc.cluster.local:8200"
    volumeMounts:
    - name: ml-db-secret
      mountPath: "/etc/secrets"
      readOnly: true
  volumes:
  - name: ml-db-secret
    secret:
      secretName: ml-db-secret
```

**AWS Secrets Manager Integration with EKS**:
```yaml
# AWS Load Balancer Controller with Secrets Manager
apiVersion: v1
kind: Secret
metadata:
  name: aws-load-balancer-webhook-tls
  namespace: kube-system
type: Opaque
data:
  ca.crt: # Retrieved from AWS Secrets Manager
  tls.crt: # Retrieved from AWS Secrets Manager  
  tls.key: # Retrieved from AWS Secrets Manager

---
# Secrets Store CSI Driver configuration
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: ml-secrets-provider
  namespace: ml-training
spec:
  provider: aws
  parameters:
    objects: |
      - objectName: "ml-database-credentials"
        objectType: "secretsmanager"
        jmesPath:
          - path: "username"
            objectAlias: "db_username"
          - path: "password" 
            objectAlias: "db_password"
          - path: "host"
            objectAlias: "db_host"
      - objectName: "ml-api-keys"
        objectType: "secretsmanager"
        jmesPath:
          - path: "external_api_key"
            objectAlias: "api_key"
  secretObjects:
  - secretName: ml-database-secret
    type: Opaque
    data:
    - objectName: "db_username"
      key: "username"
    - objectName: "db_password"
      key: "password"
    - objectName: "db_host"
      key: "host"

---
# Pod using Secrets Store CSI Driver
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
  namespace: ml-training
spec:
  serviceAccountName: ml-training-sa
  containers:
  - name: ml-training
    image: ml-training:latest
    volumeMounts:
    - name: secrets-store
      mountPath: "/mnt/secrets"
      readOnly: true
    env:
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: ml-database-secret
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: ml-database-secret
          key: password
  volumes:
  - name: secrets-store
    csi:
      driver: secrets-store.csi.k8s.io
      readOnly: true
      volumeAttributes:
        secretProviderClass: "ml-secrets-provider"
```

#### 5.2 Automated Secret Rotation

**Vault Database Secret Rotation**:
```python
#!/usr/bin/env python3
# secret_rotation_manager.py

import hvac
import boto3
import psycopg2
import time
import logging
from datetime import datetime, timedelta

class SecretRotationManager:
    def __init__(self, vault_url: str, vault_token: str):
        self.vault_client = hvac.Client(url=vault_url, token=vault_token)
        self.logger = logging.getLogger(__name__)
        
    def rotate_database_credentials(self, db_role: str, connection_info: dict):
        """Rotate database credentials using Vault dynamic secrets"""
        try:
            # Generate new credentials
            new_creds = self.vault_client.read(f'database/creds/{db_role}')
            
            if not new_creds:
                raise Exception(f"Failed to generate credentials for role {db_role}")
            
            username = new_creds['data']['username']
            password = new_creds['data']['password']
            
            # Test new credentials
            if self._test_database_connection(connection_info, username, password):
                self.logger.info(f"Successfully rotated credentials for {db_role}")
                return {
                    'username': username,
                    'password': password,
                    'lease_id': new_creds['lease_id'],
                    'lease_duration': new_creds['lease_duration']
                }
            else:
                raise Exception("New credentials failed connection test")
                
        except Exception as e:
            self.logger.error(f"Failed to rotate credentials for {db_role}: {str(e)}")
            raise
    
    def _test_database_connection(self, connection_info: dict, username: str, password: str) -> bool:
        """Test database connection with new credentials"""
        try:
            conn = psycopg2.connect(
                host=connection_info['host'],
                port=connection_info['port'],
                database=connection_info['database'],
                user=username,
                password=password,
                connect_timeout=10
            )
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def update_application_secrets(self, app_secret_path: str, new_credentials: dict):
        """Update application secrets in Vault KV store"""
        try:
            # Read current secret
            current_secret = self.vault_client.secrets.kv.v2.read_secret_version(
                path=app_secret_path
            )
            
            # Update with new credentials
            updated_secret = current_secret['data']['data'].copy()
            updated_secret.update({
                'username': new_credentials['username'],
                'password': new_credentials['password'],
                'rotated_at': datetime.utcnow().isoformat(),
                'lease_id': new_credentials['lease_id']
            })
            
            # Write updated secret
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=app_secret_path,
                secret=updated_secret
            )
            
            self.logger.info(f"Updated application secret at {app_secret_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to update application secret: {str(e)}")
            raise
    
    def notify_applications(self, app_endpoints: list, secret_path: str):
        """Notify applications of secret rotation"""
        for endpoint in app_endpoints:
            try:
                # Implementation depends on application notification mechanism
                # Could be webhook, message queue, file system signal, etc.
                self._send_rotation_notification(endpoint, secret_path)
            except Exception as e:
                self.logger.warning(f"Failed to notify {endpoint}: {str(e)}")
    
    def _send_rotation_notification(self, endpoint: str, secret_path: str):
        """Send rotation notification to application endpoint"""
        import requests
        
        payload = {
            'event': 'secret_rotated',
            'secret_path': secret_path,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        response = requests.post(
            f"{endpoint}/api/secrets/rotation",
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            raise Exception(f"Notification failed with status {response.status_code}")

def main():
    """Main rotation workflow"""
    rotation_manager = SecretRotationManager(
        vault_url="https://vault.example.com",
        vault_token="s.AbCdEfGhIjKlMnOpQrStUvWx"
    )
    
    # Database connection info
    db_connection = {
        'host': 'ml-database.example.com',
        'port': 5432,
        'database': 'ml_training'
    }
    
    # Rotate credentials
    new_creds = rotation_manager.rotate_database_credentials(
        'ml-training-role', 
        db_connection
    )
    
    # Update application secrets
    rotation_manager.update_application_secrets(
        'ml-training/database',
        new_creds
    )
    
    # Notify applications
    app_endpoints = [
        'http://ml-training-service.ml.svc.cluster.local:8080',
        'http://ml-inference-service.ml.svc.cluster.local:8080'
    ]
    
    rotation_manager.notify_applications(app_endpoints, 'ml-training/database')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

#### 5.3 Cross-Platform Secret Synchronization

**Multi-Cloud Secret Synchronization**:
```python
#!/usr/bin/env python3
# multi_cloud_secret_sync.py

import hvac
import boto3
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from google.cloud import secretmanager
import json
import logging
from typing import Dict, Any

class MultiCloudSecretSync:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.vault_client = hvac.Client(
            url=config['vault']['url'],
            token=config['vault']['token']
        )
        
        self.aws_secrets = boto3.client('secretsmanager', 
                                       region_name=config['aws']['region'])
        
        self.azure_credential = DefaultAzureCredential()
        self.azure_client = SecretClient(
            vault_url=config['azure']['vault_url'],
            credential=self.azure_credential
        )
        
        self.gcp_client = secretmanager.SecretManagerServiceClient()
        self.gcp_project = config['gcp']['project_id']
    
    def sync_secret_to_aws(self, vault_path: str, aws_secret_name: str):
        """Sync secret from Vault to AWS Secrets Manager"""
        try:
            # Read from Vault
            vault_secret = self.vault_client.secrets.kv.v2.read_secret_version(
                path=vault_path
            )
            
            secret_data = vault_secret['data']['data']
            
            # Update in AWS Secrets Manager
            try:
                self.aws_secrets.update_secret(
                    SecretId=aws_secret_name,
                    SecretString=json.dumps(secret_data)
                )
                self.logger.info(f"Updated existing AWS secret: {aws_secret_name}")
            except self.aws_secrets.exceptions.ResourceNotFoundException:
                self.aws_secrets.create_secret(
                    Name=aws_secret_name,
                    SecretString=json.dumps(secret_data),
                    Description=f"Synced from Vault path: {vault_path}"
                )
                self.logger.info(f"Created new AWS secret: {aws_secret_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to sync to AWS: {str(e)}")
            raise
    
    def sync_secret_to_azure(self, vault_path: str, azure_secret_name: str):
        """Sync secret from Vault to Azure Key Vault"""
        try:
            # Read from Vault
            vault_secret = self.vault_client.secrets.kv.v2.read_secret_version(
                path=vault_path
            )
            
            secret_data = vault_secret['data']['data']
            
            # Update in Azure Key Vault
            self.azure_client.set_secret(
                azure_secret_name, 
                json.dumps(secret_data)
            )
            
            self.logger.info(f"Synced to Azure Key Vault: {azure_secret_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync to Azure: {str(e)}")
            raise
    
    def sync_secret_to_gcp(self, vault_path: str, gcp_secret_id: str):
        """Sync secret from Vault to GCP Secret Manager"""
        try:
            # Read from Vault
            vault_secret = self.vault_client.secrets.kv.v2.read_secret_version(
                path=vault_path
            )
            
            secret_data = vault_secret['data']['data']
            
            # Create or update in GCP Secret Manager
            parent = f"projects/{self.gcp_project}"
            secret_name = f"{parent}/secrets/{gcp_secret_id}"
            
            try:
                # Try to create secret
                secret = {'replication': {'automatic': {}}}
                self.gcp_client.create_secret(
                    request={
                        'parent': parent,
                        'secret_id': gcp_secret_id,
                        'secret': secret
                    }
                )
                self.logger.info(f"Created new GCP secret: {gcp_secret_id}")
            except Exception:
                # Secret already exists
                pass
            
            # Add new version
            self.gcp_client.add_secret_version(
                request={
                    'parent': secret_name,
                    'payload': {'data': json.dumps(secret_data).encode('utf-8')}
                }
            )
            
            self.logger.info(f"Added version to GCP secret: {gcp_secret_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync to GCP: {str(e)}")
            raise
    
    def sync_all_secrets(self):
        """Sync all configured secrets across clouds"""
        sync_mappings = self.config.get('sync_mappings', [])
        
        for mapping in sync_mappings:
            vault_path = mapping['vault_path']
            
            try:
                if 'aws' in mapping:
                    self.sync_secret_to_aws(vault_path, mapping['aws'])
                
                if 'azure' in mapping:
                    self.sync_secret_to_azure(vault_path, mapping['azure'])
                
                if 'gcp' in mapping:
                    self.sync_secret_to_gcp(vault_path, mapping['gcp'])
                    
            except Exception as e:
                self.logger.error(f"Failed to sync {vault_path}: {str(e)}")

# Configuration example
sync_config = {
    'vault': {
        'url': 'https://vault.example.com',
        'token': 's.AbCdEfGhIjKlMnOpQrStUvWx'
    },
    'aws': {
        'region': 'us-west-2'
    },
    'azure': {
        'vault_url': 'https://ml-keyvault.vault.azure.net'
    },
    'gcp': {
        'project_id': 'ml-project-123456'
    },
    'sync_mappings': [
        {
            'vault_path': 'ml-training/database',
            'aws': 'ml-database-credentials',
            'azure': 'ml-database-credentials',
            'gcp': 'ml-database-credentials'
        },
        {
            'vault_path': 'ml-training/api-keys',
            'aws': 'ml-api-keys',
            'azure': 'ml-api-keys',
            'gcp': 'ml-api-keys'
        }
    ]
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    syncer = MultiCloudSecretSync(sync_config)
    syncer.sync_all_secrets()
```

### 6. Performance and Security Optimization

#### 6.1 Caching and Performance

**Local Secret Caching Strategy**:
```python
#!/usr/bin/env python3
# secret_cache_manager.py

import time
import threading
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class SecretCache:
    def __init__(self, cache_ttl: int = 3600, max_cache_size: int = 1000):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get secret from cache if valid"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.cache_ttl:
                    self.access_times[key] = time.time()
                    return entry['data']
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            return None
    
    def put(self, key: str, data: Dict[str, Any]):
        """Store secret in cache"""
        with self.lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_cache_size:
                self._evict_lru()
            
            self.cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()
    
    def invalidate(self, key: str):
        """Remove secret from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

class CachedSecretManager:
    def __init__(self, secret_client, cache_ttl: int = 3600):
        self.secret_client = secret_client
        self.cache = SecretCache(cache_ttl=cache_ttl)
        self.logger = logging.getLogger(__name__)
    
    def get_secret(self, secret_path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get secret with caching"""
        cache_key = self._generate_cache_key(secret_path)
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_secret = self.cache.get(cache_key)
            if cached_secret:
                self.logger.debug(f"Retrieved {secret_path} from cache")
                return cached_secret
        
        # Fetch from secret store
        try:
            secret_data = self._fetch_secret(secret_path)
            self.cache.put(cache_key, secret_data)
            self.logger.debug(f"Fetched and cached {secret_path}")
            return secret_data
        except Exception as e:
            self.logger.error(f"Failed to fetch secret {secret_path}: {str(e)}")
            raise
    
    def _generate_cache_key(self, secret_path: str) -> str:
        """Generate consistent cache key"""
        return hashlib.sha256(secret_path.encode()).hexdigest()
    
    def _fetch_secret(self, secret_path: str) -> Dict[str, Any]:
        """Fetch secret from underlying secret store"""
        # Implementation depends on secret store type
        # This is a placeholder for the actual implementation
        pass
    
    def refresh_secret(self, secret_path: str):
        """Force refresh of cached secret"""
        cache_key = self._generate_cache_key(secret_path)
        self.cache.invalidate(cache_key)
        return self.get_secret(secret_path, force_refresh=True)
```

#### 6.2 Security Monitoring and Auditing

**Secret Access Monitoring**:
```python
#!/usr/bin/env python3
# secret_audit_monitor.py

import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

class SecretAccessMonitor:
    def __init__(self, alert_thresholds: Dict[str, Any]):
        self.alert_thresholds = alert_thresholds
        self.access_patterns = defaultdict(list)
        self.anomaly_scores = defaultdict(float)
        self.logger = logging.getLogger(__name__)
    
    def log_access(self, event: Dict[str, Any]):
        """Log secret access event"""
        try:
            # Normalize event data
            normalized_event = self._normalize_event(event)
            
            # Store access pattern
            user_id = normalized_event['user_id']
            self.access_patterns[user_id].append(normalized_event)
            
            # Analyze for anomalies
            self._analyze_access_pattern(user_id, normalized_event)
            
            # Check for policy violations
            self._check_policy_violations(normalized_event)
            
        except Exception as e:
            self.logger.error(f"Failed to process access event: {str(e)}")
    
    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize access event data"""
        return {
            'timestamp': event.get('timestamp', datetime.utcnow().isoformat()),
            'user_id': event.get('user_id', 'unknown'),
            'secret_path': event.get('secret_path', ''),
            'action': event.get('action', 'read'),
            'source_ip': event.get('source_ip', ''),
            'user_agent': event.get('user_agent', ''),
            'success': event.get('success', True)
        }
    
    def _analyze_access_pattern(self, user_id: str, event: Dict[str, Any]):
        """Analyze access patterns for anomalies"""
        # Get recent access history
        recent_access = [
            e for e in self.access_patterns[user_id]
            if datetime.fromisoformat(e['timestamp']) > 
               datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Calculate anomaly score based on various factors
        anomaly_score = 0.0
        
        # Unusual access frequency
        if len(recent_access) > self.alert_thresholds.get('max_daily_access', 100):
            anomaly_score += 0.3
        
        # Unusual access time
        access_hour = datetime.fromisoformat(event['timestamp']).hour
        if access_hour < 6 or access_hour > 22:  # Outside business hours
            anomaly_score += 0.2
        
        # New IP address
        recent_ips = {e['source_ip'] for e in recent_access[:-1]}
        if event['source_ip'] not in recent_ips and len(recent_ips) > 0:
            anomaly_score += 0.4
        
        # Unusual secret path access
        recent_paths = {e['secret_path'] for e in recent_access[:-1]}
        if event['secret_path'] not in recent_paths and len(recent_paths) > 0:
            anomaly_score += 0.3
        
        # Update anomaly score
        self.anomaly_scores[user_id] = anomaly_score
        
        # Generate alert if threshold exceeded
        if anomaly_score > self.alert_thresholds.get('anomaly_threshold', 0.7):
            self._generate_anomaly_alert(user_id, event, anomaly_score)
    
    def _check_policy_violations(self, event: Dict[str, Any]):
        """Check for policy violations"""
        violations = []
        
        # Check access time restrictions
        access_time = datetime.fromisoformat(event['timestamp'])
        if access_time.weekday() >= 5:  # Weekend access
            violations.append("weekend_access")
        
        # Check IP restrictions
        allowed_ip_ranges = self.alert_thresholds.get('allowed_ip_ranges', [])
        if allowed_ip_ranges and not self._ip_in_ranges(event['source_ip'], allowed_ip_ranges):
            violations.append("unauthorized_ip")
        
        # Check secret path restrictions
        restricted_paths = self.alert_thresholds.get('restricted_paths', [])
        for restricted_path in restricted_paths:
            if event['secret_path'].startswith(restricted_path):
                violations.append(f"restricted_path_access:{restricted_path}")
        
        # Generate violation alerts
        for violation in violations:
            self._generate_violation_alert(event, violation)
    
    def _generate_anomaly_alert(self, user_id: str, event: Dict[str, Any], score: float):
        """Generate anomaly alert"""
        alert = {
            'alert_type': 'anomaly_detection',
            'user_id': user_id,
            'anomaly_score': score,
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high' if score > 0.9 else 'medium'
        }
        
        self.logger.warning(f"Anomaly detected for user {user_id}: {json.dumps(alert)}")
        self._send_alert(alert)
    
    def _generate_violation_alert(self, event: Dict[str, Any], violation: str):
        """Generate policy violation alert"""
        alert = {
            'alert_type': 'policy_violation',
            'violation': violation,
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high'
        }
        
        self.logger.error(f"Policy violation detected: {json.dumps(alert)}")
        self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to monitoring system"""
        # Implementation depends on alerting system
        # Could be email, Slack, PagerDuty, etc.
        pass
    
    def _ip_in_ranges(self, ip: str, ranges: List[str]) -> bool:
        """Check if IP is in allowed ranges"""
        import ipaddress
        
        try:
            ip_addr = ipaddress.ip_address(ip)
            for cidr_range in ranges:
                if ip_addr in ipaddress.ip_network(cidr_range):
                    return True
            return False
        except Exception:
            return False

# Usage example
monitor_config = {
    'max_daily_access': 50,
    'anomaly_threshold': 0.6,
    'allowed_ip_ranges': ['10.0.0.0/8', '192.168.0.0/16'],
    'restricted_paths': ['ml-prod/', 'customer-data/']
}

monitor = SecretAccessMonitor(monitor_config)

# Example access event
access_event = {
    'timestamp': '2024-01-15T14:30:00Z',
    'user_id': 'ml-engineer-123',
    'secret_path': 'ml-training/database-creds',
    'action': 'read',
    'source_ip': '10.1.1.100',
    'user_agent': 'vault-cli/1.12.0',
    'success': True
}

monitor.log_access(access_event)
```

---

## 🔍 Key Questions

### Beginner Level

1. **Q**: What is the difference between static secrets and dynamic secrets, and when would you use each in an ML environment?
   **A**: Static secrets are fixed values stored securely (API keys, certificates), while dynamic secrets are generated on-demand with limited lifetimes (database credentials, cloud access keys). Use static secrets for long-lived integrations and dynamic secrets for short-term access to reduce exposure risk.

2. **Q**: Why is secret rotation important, and what are the key challenges in implementing it for AI/ML workloads?
   **A**: Secret rotation reduces exposure time and limits damage from compromised credentials. Challenges include coordinating updates across distributed training jobs, handling long-running processes, and maintaining service availability during rotation.

3. **Q**: What are the main components of HashiCorp Vault, and how do they work together?
   **A**: Storage backend (encrypted data persistence), security barrier (encryption layer), authentication methods (identity verification), secret engines (secret generation/storage), and policies (access control). They work together to provide secure, auditable secret management.

### Intermediate Level

4. **Q**: Design a secret management strategy for a multi-cloud ML pipeline that uses AWS for training, Azure for inference, and GCP for data storage.
   **A**: 
   ```
   Strategy:
   - Central Vault cluster for secret orchestration
   - Cloud-specific secret stores (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)
   - Automated synchronization between Vault and cloud stores
   - Cloud-native authentication (IAM roles, managed identities)
   - Environment-specific secret namespaces
   - Automated rotation with cross-cloud coordination
   ```

5. **Q**: How would you implement automated secret rotation for a database used by 100+ ML training jobs without causing service disruption?
   **A**: Implement blue-green credential rotation with grace periods, use connection pooling with credential refresh capabilities, coordinate rotation during low-traffic windows, implement gradual rollout with canary testing, and provide fallback mechanisms for failed rotations.

6. **Q**: Explain how to securely inject secrets into Kubernetes pods running ML workloads while maintaining least privilege access.
   **A**: Use Vault Agent or Secrets Store CSI Driver for secret injection, implement pod-specific service accounts with minimal Vault policies, use init containers for one-time secret retrieval, implement secret caching with appropriate TTLs, and monitor access patterns for anomalies.

### Advanced Level

7. **Q**: Design a comprehensive secret lifecycle management system for a federated learning platform where multiple organizations need access to shared secrets while maintaining organizational boundaries.
   **A**:
   ```
   Architecture:
   - Hierarchical Vault deployment with organization-specific namespaces
   - Cross-organization secret sharing with explicit approval workflows
   - Organization-specific encryption keys and access policies
   - Federated authentication with external identity providers
   - Audit logging with organization-specific views
   - Automated secret cleanup based on project lifecycles
   - Emergency revocation capabilities with cross-org notification
   ```

8. **Q**: How would you implement zero-trust secret management for a high-security AI research environment with strict compliance requirements?
   **A**: Implement identity-based access control with continuous verification, use hardware security modules (HSMs) for key protection, implement comprehensive audit logging with immutable trails, use just-in-time access with approval workflows, implement context-aware access policies, and deploy advanced threat detection for secret access patterns.

### Tricky Questions

9. **Q**: You discover that a critical API key for your ML training pipeline has been compromised and is being used by an unauthorized party. Design a comprehensive incident response plan that minimizes disruption to ongoing training jobs while ensuring security.
   **A**:
   ```
   Immediate Response:
   - Revoke compromised API key immediately
   - Generate new API key with different scope if possible
   - Identify all systems using the compromised key
   - Implement temporary access controls or alternative credentials
   
   Investigation:
   - Analyze access logs to determine scope of compromise
   - Identify unauthorized usage patterns and accessed data
   - Review related secrets that may be compromised
   - Document timeline and attack vectors
   
   Recovery:
   - Deploy new credentials to all legitimate systems
   - Verify successful credential updates across all services
   - Implement additional monitoring for related secrets
   - Update security procedures based on lessons learned
   
   Prevention:
   - Implement shorter key rotation cycles
   - Deploy secret usage monitoring and anomaly detection
   - Review and update access policies
   - Enhance security training for development teams
   ```

10. **Q**: Design a secret management architecture that can handle the scale of a major AI research lab with 10,000+ researchers, 100,000+ experiments, and strict regulatory compliance requirements while maintaining high performance and availability.
    **A**:
    ```
    Architecture:
    - Multi-region Vault clusters with automatic failover
    - Hierarchical secret organization with project/team namespaces
    - Distributed caching layer with regional replicas
    - Integration with enterprise identity systems (LDAP/AD)
    - Automated secret lifecycle management with ML-driven optimization
    - Compliance monitoring with automated reporting
    - Performance optimization with secret usage analytics
    
    Implementation:
    - Database clustering for high availability and performance
    - Load balancing with health checks and automatic scaling
    - Monitoring and alerting for all system components
    - Disaster recovery with cross-region replication
    - Security controls including HSM integration and network isolation
    - API rate limiting and abuse detection
    - Regular security audits and penetration testing
    ```

---

## 🛡️ Security Deep Dive

### Advanced Threat Scenarios

#### Secret Exfiltration Attacks

**Attack Vectors**:
```
Common Exfiltration Methods:
- Memory dumps from compromised applications
- Log file analysis and credential extraction
- Side-channel attacks on secret retrieval
- Insider threats with authorized access
- Supply chain attacks on secret management tools

Detection Strategies:
- Anomaly detection on secret access patterns
- Memory protection and anti-debugging measures
- Log sanitization and secure logging practices
- Privileged access monitoring and behavioral analysis
- Integrity monitoring for secret management components
```

#### Zero-Day Vulnerabilities

**Mitigation Strategies**:
```
Defense Measures:
- Defense in depth with multiple secret storage layers
- Network segmentation and access controls
- Regular security updates and patch management
- Incident response plans for secret compromise
- Backup secret stores and recovery procedures
```

### Compliance and Governance

#### Regulatory Compliance

**Framework Implementation**:
```
GDPR Article 32 - Security of Processing:
- Encryption of personal data at rest and in transit
- Measures to ensure ongoing confidentiality and integrity
- Regular testing and evaluation of security measures
- Procedures for restoring availability after incidents

SOC 2 Type II - CC6.1 Logical Access:
- Logical access security controls
- User access provisioning and deprovisioning
- Privileged access management
- Monitoring of user access activities

HIPAA 164.312(a)(1) - Access Control:
- Unique user identification
- Emergency access procedures
- Automatic logoff
- Encryption and decryption controls
```

---

## 🚀 Performance Optimization

### Secret Access Optimization

#### Caching Strategies

**Multi-Level Caching**:
```
Application Layer:
- In-memory caching with TTL
- Local file system caching for offline access
- Memory-mapped file caching for large secrets

Network Layer:
- Regional secret store replicas
- CDN-style distribution for static secrets
- Load balancing across secret store instances

Infrastructure Layer:
- SSD storage for secret databases
- Memory-optimized database configurations
- Connection pooling for secret store access
```

#### Batch Operations

**Bulk Secret Operations**:
```python
# Efficient bulk secret retrieval
def get_secrets_batch(secret_paths: List[str]) -> Dict[str, Any]:
    """Retrieve multiple secrets in single operation"""
    secrets = {}
    
    # Group by secret engine for efficient batching
    grouped_paths = defaultdict(list)
    for path in secret_paths:
        engine = path.split('/')[0]
        grouped_paths[engine].append(path)
    
    # Batch retrieve per engine
    for engine, paths in grouped_paths.items():
        engine_secrets = vault_client.secrets.kv.v2.read_secret_version_bulk(
            mount_point=engine,
            paths=[p.split('/', 1)[1] for p in paths]
        )
        secrets.update(engine_secrets)
    
    return secrets
```

---

## 📝 Practical Exercises

### Exercise 1: Multi-Cloud Secret Management
Design and implement a comprehensive secret management solution that:
- Supports AWS, Azure, and GCP simultaneously
- Provides automated secret rotation across all platforms
- Implements cross-cloud secret synchronization
- Includes monitoring and alerting for secret access
- Handles failover and disaster recovery scenarios

### Exercise 2: High-Performance Secret Caching
Develop a high-performance secret caching system that:
- Minimizes latency for ML training workloads
- Implements intelligent cache warming and prefetching
- Provides cache coherency across distributed systems
- Includes cache security and encryption
- Handles cache invalidation for rotated secrets

### Exercise 3: Compliance-Ready Secret Audit System
Create a comprehensive secret audit and compliance system that:
- Tracks all secret access with immutable audit trails
- Implements real-time anomaly detection
- Provides compliance reporting for multiple frameworks
- Includes automated policy violation detection
- Supports forensic investigation capabilities

### Exercise 4: Zero-Trust Secret Architecture
Design a zero-trust secret management architecture that:
- Implements continuous authentication and authorization
- Provides context-aware access controls
- Includes behavioral analysis and threat detection
- Supports just-in-time access provisioning
- Handles emergency access and revocation scenarios

---

## 🔗 Next Steps
In the next section (day02_005), we'll explore drift detection and automated compliance scanning, focusing on maintaining security posture over time, detecting configuration drift, and implementing continuous compliance validation for AI/ML infrastructure.