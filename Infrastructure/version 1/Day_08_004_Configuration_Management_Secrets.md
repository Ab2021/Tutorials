# Day 8.4: Configuration Management & Secrets Handling

## 🔐 Infrastructure as Code & Automation - Part 4

**Focus**: Environment Configuration, Secrets Management, Security Automation  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master advanced configuration management strategies for ML environments
- Learn comprehensive secrets management and encryption patterns
- Understand automated security controls and certificate management
- Analyze configuration drift detection and remediation strategies

---

## ⚙️ Configuration Management Theory

### **Configuration Lifecycle Management**

Configuration management in ML systems requires handling complex, multi-dimensional configurations that span infrastructure, applications, models, and data pipelines.

**Configuration Taxonomy:**
```
Configuration Classification:
1. Infrastructure Configuration:
   - Compute resources (CPU, GPU, memory allocations)
   - Network settings (VPC, subnets, security groups)
   - Storage configurations (volumes, backup policies)
   - Scaling parameters (min/max replicas, thresholds)

2. Application Configuration:
   - Framework settings (TensorFlow, PyTorch configurations)
   - Runtime parameters (batch sizes, worker counts)
   - Feature flags and experimental toggles
   - Integration endpoints and service discovery

3. Model Configuration:
   - Hyperparameters and training settings
   - Model serving parameters (batch size, timeout)
   - A/B testing configurations and traffic splits
   - Model versioning and artifact locations

4. Data Configuration:
   - Data source connections and credentials
   - Pipeline processing parameters
   - Quality thresholds and validation rules
   - Retention policies and archival settings

Configuration Hierarchy Model:
Global_Config ⊃ Environment_Config ⊃ Service_Config ⊃ Instance_Config

Where:
Config_Effective = merge(Global_Config, Environment_Config, Service_Config, Instance_Config)
Priority: Instance_Config > Service_Config > Environment_Config > Global_Config
```

**Configuration Validation Framework:**
```
Validation Rule Engine:
validation_rules = {
    "resource_limits": {
        "cpu": {"min": "100m", "max": "32", "pattern": r"^\d+(\.\d+)?[m]?$"},
        "memory": {"min": "128Mi", "max": "256Gi", "pattern": r"^\d+[KMGT]i$"},
        "gpu": {"min": 0, "max": 8, "type": "integer"}
    },
    "environment_constraints": {
        "production": {
            "allowed_registries": ["gcr.io/company", "company.azurecr.io"],
            "required_labels": ["team", "cost-center", "environment"],
            "security_level": "high"
        },
        "development": {
            "allowed_registries": ["*"],
            "required_labels": ["team", "environment"],
            "security_level": "medium"
        }
    },
    "data_governance": {
        "sensitive_data": {
            "encryption": "required",
            "access_logging": "enabled",
            "retention_days": {"min": 90, "max": 2555}  # 7 years max
        }
    }
}

Configuration Validation Algorithm:
def validate_configuration(config, rules, context):
    violations = []
    
    for rule_category, category_rules in rules.items():
        if rule_category in config:
            category_violations = validate_category(
                config[rule_category], 
                category_rules, 
                context
            )
            violations.extend(category_violations)
    
    # Cross-category validation
    cross_violations = validate_cross_category_constraints(config, rules, context)
    violations.extend(cross_violations)
    
    return ValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=extract_warnings(config, rules)
    )

def validate_category(config_section, rules, context):
    violations = []
    
    for field, field_rules in rules.items():
        if field in config_section:
            value = config_section[field]
            
            # Type validation
            if "type" in field_rules:
                if not validate_type(value, field_rules["type"]):
                    violations.append(f"Field {field} has invalid type")
            
            # Range validation
            if "min" in field_rules or "max" in field_rules:
                if not validate_range(value, field_rules.get("min"), field_rules.get("max")):
                    violations.append(f"Field {field} is outside allowed range")
            
            # Pattern validation
            if "pattern" in field_rules:
                if not re.match(field_rules["pattern"], str(value)):
                    violations.append(f"Field {field} doesn't match required pattern")
            
            # Context-dependent validation
            if "context_rules" in field_rules:
                context_violations = validate_context_rules(
                    value, field_rules["context_rules"], context
                )
                violations.extend(context_violations)
    
    return violations
```

### **Environment-Specific Configuration**

**Multi-Environment Strategy:**
```
Environment Configuration Matrix:
environments = {
    "development": {
        "resource_profile": "minimal",
        "auto_scaling": False,
        "monitoring_level": "basic",
        "data_sources": ["synthetic", "sample_datasets"],
        "security_level": "relaxed",
        "cost_optimization": "aggressive"
    },
    
    "staging": {
        "resource_profile": "production_like",
        "auto_scaling": True,
        "monitoring_level": "comprehensive",
        "data_sources": ["production_subset", "anonymized_data"],
        "security_level": "strict",
        "cost_optimization": "moderate"
    },
    
    "production": {
        "resource_profile": "high_availability",
        "auto_scaling": True,
        "monitoring_level": "full_observability",
        "data_sources": ["live_production_data"],
        "security_level": "maximum",
        "cost_optimization": "balanced"
    }
}

Resource Profile Configurations:
resource_profiles = {
    "minimal": {
        "compute": {
            "cpu_request": "100m",
            "cpu_limit": "500m",
            "memory_request": "128Mi",
            "memory_limit": "512Mi",
            "gpu_count": 0
        },
        "scaling": {
            "min_replicas": 1,
            "max_replicas": 2,
            "target_cpu_utilization": 80
        },
        "storage": {
            "volume_size": "10Gi",
            "storage_class": "standard",
            "backup_enabled": False
        }
    },
    
    "production_like": {
        "compute": {
            "cpu_request": "1",
            "cpu_limit": "2",
            "memory_request": "2Gi",
            "memory_limit": "4Gi",
            "gpu_count": 1
        },
        "scaling": {
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70
        },
        "storage": {
            "volume_size": "100Gi",
            "storage_class": "ssd",
            "backup_enabled": True
        }
    },
    
    "high_availability": {
        "compute": {
            "cpu_request": "2",
            "cpu_limit": "4",
            "memory_request": "4Gi",
            "memory_limit": "8Gi",
            "gpu_count": 2
        },
        "scaling": {
            "min_replicas": 3,
            "max_replicas": 50,
            "target_cpu_utilization": 60
        },
        "storage": {
            "volume_size": "1Ti",
            "storage_class": "premium-ssd",
            "backup_enabled": True,
            "replication": "multi_zone"
        }
    }
}

Configuration Template Engine:
class ConfigurationTemplateEngine:
    def __init__(self, templates_dir):
        self.templates = self.load_templates(templates_dir)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            undefined=jinja2.StrictUndefined
        )
    
    def render_configuration(self, template_name, context):
        template = self.jinja_env.get_template(template_name)
        
        # Add built-in functions
        context.update({
            'resource_calculation': self.calculate_resources,
            'security_level': self.get_security_level,
            'compliance_requirements': self.get_compliance_requirements
        })
        
        try:
            rendered = template.render(**context)
            return yaml.safe_load(rendered)
        except Exception as e:
            raise ConfigurationRenderError(f"Failed to render {template_name}: {e}")
    
    def calculate_resources(self, workload_type, expected_load, environment):
        base_resources = {
            "training": {"cpu": 4, "memory": "16Gi", "gpu": 1},
            "inference": {"cpu": 1, "memory": "2Gi", "gpu": 0},
            "data_processing": {"cpu": 8, "memory": "32Gi", "gpu": 0}
        }
        
        base = base_resources.get(workload_type, base_resources["inference"])
        
        # Scale based on expected load
        scaling_factor = min(max(expected_load / 1000, 0.1), 10.0)
        
        # Environment-specific adjustments
        env_multipliers = {
            "development": 0.25,
            "staging": 0.75,
            "production": 1.0
        }
        
        multiplier = env_multipliers.get(environment, 1.0) * scaling_factor
        
        return {
            "cpu": f"{int(base['cpu'] * multiplier)}",
            "memory": f"{int(base['memory'].rstrip('Gi')) * multiplier}Gi",
            "gpu": int(base['gpu'] * multiplier) if base['gpu'] > 0 else 0
        }
```

**Configuration as Code Implementation:**
```
# config/base/ml-training-service.yaml.j2
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training-service
  labels:
    app: ml-training-service
    environment: {{ environment }}
    team: {{ team }}
spec:
  replicas: {{ resource_profile.scaling.min_replicas }}
  selector:
    matchLabels:
      app: ml-training-service
  template:
    metadata:
      labels:
        app: ml-training-service
        environment: {{ environment }}
    spec:
      containers:
      - name: training-container
        image: {{ image_registry }}/ml-training:{{ image_tag }}
        resources:
          requests:
            cpu: {{ resource_profile.compute.cpu_request }}
            memory: {{ resource_profile.compute.memory_request }}
            {% if resource_profile.compute.gpu_count > 0 %}
            nvidia.com/gpu: {{ resource_profile.compute.gpu_count }}
            {% endif %}
          limits:
            cpu: {{ resource_profile.compute.cpu_limit }}
            memory: {{ resource_profile.compute.memory_limit }}
            {% if resource_profile.compute.gpu_count > 0 %}
            nvidia.com/gpu: {{ resource_profile.compute.gpu_count }}
            {% endif %}
        env:
        - name: ENVIRONMENT
          value: {{ environment }}
        - name: LOG_LEVEL
          value: {{ log_levels[environment] }}
        - name: MODEL_REGISTRY_URL
          value: {{ model_registry.url }}
        - name: DATA_SOURCE_URL
          valueFrom:
            secretKeyRef:
              name: data-source-credentials
              key: url
        {% for key, value in environment_variables.items() %}
        - name: {{ key }}
          value: "{{ value }}"
        {% endfor %}
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: data-cache
          mountPath: /data/cache
        {% if security_level == "maximum" %}
        - name: security-policies
          mountPath: /etc/security
          readOnly: true
        {% endif %}
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: data-cache
        emptyDir:
          sizeLimit: {{ resource_profile.storage.cache_size | default("10Gi") }}
      {% if security_level == "maximum" %}
      - name: security-policies
        configMap:
          name: security-policies
      {% endif %}

# config/environments/production/values.yaml
environment: production
team: ml-platform
image_registry: gcr.io/company-ml
image_tag: v1.2.3

resource_profile: !include ../profiles/high_availability.yaml

log_levels:
  development: DEBUG
  staging: INFO
  production: WARN

model_registry:
  url: https://model-registry.company.com
  
environment_variables:
  BATCH_SIZE: "128"
  MAX_EPOCHS: "100"
  LEARNING_RATE: "0.001"
  
security_level: maximum

# config/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: ml-training-prod

resources:
- ../../base/ml-training-service.yaml
- ../../base/ml-training-pvc.yaml
- ../../base/ml-training-hpa.yaml

configMapGenerator:
- name: ml-training-config
  files:
  - config.yaml=../environments/production/config.yaml
  
secretGenerator:
- name: data-source-credentials
  literals:
  - url=postgresql://prod-db:5432/mldata
  
patchesStrategicMerge:
- production-patches.yaml

images:
- name: ml-training
  newTag: v1.2.3
```

---

## 🔐 Secrets Management Architecture

### **Multi-Layered Secrets Strategy**

**Secrets Classification and Handling:**
```
Secrets Taxonomy:
1. Infrastructure Secrets:
   - Cloud provider API keys and service accounts
   - Database connection strings and passwords
   - Container registry credentials
   - TLS certificates and private keys

2. Application Secrets:
   - API keys for external services
   - JWT signing keys and tokens
   - Feature store access credentials
   - Model registry authentication tokens

3. Data Access Secrets:
   - Data warehouse connection credentials
   - S3/GCS bucket access keys
   - Kafka cluster authentication
   - Data encryption keys

4. Operational Secrets:
   - Monitoring system credentials
   - Alerting webhook URLs
   - Backup system access keys
   - CI/CD pipeline tokens

Secrets Lifecycle Management:
Secret_Lifecycle = Creation → Storage → Distribution → Usage → Rotation → Revocation

Rotation_Frequency = f(Secret_Criticality, Compliance_Requirements, Usage_Patterns)

Critical_Secrets_Rotation = every 30 days
Standard_Secrets_Rotation = every 90 days
Low_Risk_Secrets_Rotation = every 365 days
```

**Kubernetes Secrets Integration:**
```
External Secrets Operator Configuration:
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: gcpsm-secret-store
spec:
  provider:
    gcpsm:
      projectId: "ml-platform-prod"
      auth:
        workloadIdentity:
          clusterLocation: us-central1
          clusterName: ml-platform-cluster
          serviceAccountRef:
            name: external-secrets-sa

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
spec:
  refreshInterval: 30m
  secretStoreRef:
    name: gcpsm-secret-store
    kind: SecretStore
  target:
    name: database-credentials
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: ml-database-username
  - secretKey: password
    remoteRef:
      key: ml-database-password
  - secretKey: connection_string
    remoteRef:
      key: ml-database-connection-string

Sealed Secrets for GitOps:
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: ml-api-keys
  namespace: ml-platform
spec:
  encryptedData:
    openai-api-key: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAx...
    huggingface-token: AgAKAoiQm+/LrE2po1pkitjMaDnIDXq...
    wandb-api-key: AgAi+wcptLz5OpEWq6Dx9V4jLg5uqwPwx...
  template:
    metadata:
      name: ml-api-keys
      namespace: ml-platform
    type: Opaque

Bank-Vaults Integration:
apiVersion: "vault.security.banzaicloud.io/v1alpha1"
kind: "Vault"
metadata:
  name: "ml-platform-vault"
spec:
  size: 3
  image: vault:1.14.0
  
  # High availability configuration
  ha:
    enabled: true
    raft:
      enabled: true
  
  # Vault configuration
  config:
    storage:
      raft:
        path: "/vault/data"
    listener:
      tcp:
        address: "0.0.0.0:8200"
        tls_cert_file: "/vault/tls/server.crt"
        tls_key_file: "/vault/tls/server.key"
    api_addr: "https://vault.ml-platform.svc.cluster.local:8200"
    cluster_addr: "https://vault.ml-platform.svc.cluster.local:8201"
    ui: true
  
  # Unsealing configuration
  unsealConfig:
    kubernetes:
      secretNamespace: "ml-platform"
      secretName: "vault-unseal-keys"
  
  # Authentication methods
  externalConfig:
    auth:
      - type: kubernetes
        roles:
          - name: ml-training
            bound_service_account_names: ["ml-training-sa"]
            bound_service_account_namespaces: ["ml-training"]
            policies: ["ml-training-policy"]
            ttl: "1h"
    
    policies:
      - name: ml-training-policy
        rules: |
          path "ml/data/training/*" {
            capabilities = ["read"]
          }
          path "ml/data/models/*" {
            capabilities = ["read", "write"]
          }
    
    secrets:
      - path: ml
        type: kv-v2
        description: ML platform secrets

Vault Agent Sidecar Pattern:
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training-with-vault
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "ml-training"
        vault.hashicorp.com/agent-inject-secret-database: "ml/data/database"
        vault.hashicorp.com/agent-inject-template-database: |
          {{- with secret "ml/data/database" -}}
          export DB_USERNAME="{{ .Data.data.username }}"
          export DB_PASSWORD="{{ .Data.data.password }}"
          export DB_CONNECTION="{{ .Data.data.connection_string }}"
          {{- end }}
        vault.hashicorp.com/agent-inject-secret-api-keys: "ml/data/api-keys"
        vault.hashicorp.com/agent-inject-template-api-keys: |
          {{- with secret "ml/data/api-keys" -}}
          export OPENAI_API_KEY="{{ .Data.data.openai }}"
          export HUGGINGFACE_TOKEN="{{ .Data.data.huggingface }}"
          {{- end }}
    spec:
      serviceAccountName: ml-training-sa
      containers:
      - name: ml-training
        image: ml-training:latest
        command: ["/bin/sh"]
        args:
        - -c
        - |
          source /vault/secrets/database
          source /vault/secrets/api-keys
          exec python train_model.py
```

### **Encryption and Key Management**

**Envelope Encryption Pattern:**
```
Envelope Encryption Architecture:
Master_Key (KMS) → Data_Encryption_Key (DEK) → Encrypted_Data

Encryption Process:
1. Generate Data Encryption Key (DEK) locally
2. Encrypt data with DEK using AES-256-GCM
3. Encrypt DEK with Master Key using KMS
4. Store encrypted DEK alongside encrypted data
5. Delete plaintext DEK from memory

Decryption Process:
1. Retrieve encrypted DEK from storage
2. Decrypt DEK using Master Key via KMS
3. Decrypt data using plaintext DEK
4. Delete plaintext DEK from memory

Key Rotation Strategy:
class EnvelopeEncryption:
    def __init__(self, kms_client, master_key_id):
        self.kms_client = kms_client
        self.master_key_id = master_key_id
    
    def encrypt_data(self, plaintext_data, additional_authenticated_data=None):
        # Generate data encryption key
        dek_response = self.kms_client.generate_data_key(
            KeyId=self.master_key_id,
            KeySpec='AES_256'
        )
        
        plaintext_dek = dek_response['Plaintext']
        encrypted_dek = dek_response['CiphertextBlob']
        
        # Encrypt data with DEK
        cipher = AES.new(plaintext_dek, AES.MODE_GCM)
        if additional_authenticated_data:
            cipher.update(additional_authenticated_data)
        
        ciphertext, auth_tag = cipher.encrypt_and_digest(plaintext_data)
        
        # Clear plaintext DEK from memory
        self._secure_delete(plaintext_dek)
        
        return {
            'encrypted_data': ciphertext,
            'encrypted_dek': encrypted_dek,
            'nonce': cipher.nonce,
            'auth_tag': auth_tag,
            'aad': additional_authenticated_data
        }
    
    def decrypt_data(self, encrypted_package):
        # Decrypt DEK
        dek_response = self.kms_client.decrypt(
            CiphertextBlob=encrypted_package['encrypted_dek']
        )
        plaintext_dek = dek_response['Plaintext']
        
        # Decrypt data
        cipher = AES.new(
            plaintext_dek, 
            AES.MODE_GCM, 
            nonce=encrypted_package['nonce']
        )
        
        if encrypted_package['aad']:
            cipher.update(encrypted_package['aad'])
        
        plaintext_data = cipher.decrypt_and_verify(
            encrypted_package['encrypted_data'],
            encrypted_package['auth_tag']
        )
        
        # Clear plaintext DEK from memory
        self._secure_delete(plaintext_dek)
        
        return plaintext_data
    
    def _secure_delete(self, sensitive_data):
        # Overwrite memory with random data
        if isinstance(sensitive_data, bytes):
            ctypes.memset(id(sensitive_data) + 32, 0, len(sensitive_data))

Kubernetes Encryption at Rest:
apiVersion: apiserver.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets
  - configmaps
  providers:
  - kms:
      name: vault-kms
      endpoint: unix:///var/run/kmsplugin/socket.sock
      cachesize: 1000
      timeout: 3s
  - aescbc:
      keys:
      - name: key1
        secret: <base64-encoded-secret>
  - identity: {}

Kubernetes KMS Plugin:
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: vault-kms-plugin
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: vault-kms-plugin
  template:
    metadata:
      labels:
        app: vault-kms-plugin
    spec:
      hostNetwork: true
      containers:
      - name: vault-kms-plugin
        image: vault-kms-plugin:latest
        env:
        - name: VAULT_ADDR
          value: "https://vault.ml-platform.svc.cluster.local:8200"
        - name: VAULT_ROLE
          value: "k8s-encryption"
        volumeMounts:
        - name: kmsplugin
          mountPath: /var/run/kmsplugin
        - name: vault-token
          mountPath: /var/run/secrets/vault
      volumes:
      - name: kmsplugin
        hostPath:
          path: /var/run/kmsplugin
          type: DirectoryOrCreate
      - name: vault-token
        secret:
          secretName: vault-kms-token
```

---

## 🔄 Configuration Drift Detection

### **Automated Drift Monitoring**

**Configuration Drift Detection System:**
```
Drift Detection Algorithm:
def detect_configuration_drift(baseline_config, current_config, tolerance_thresholds):
    drift_report = DriftReport()
    
    # Structural drift detection
    structural_drift = detect_structural_changes(baseline_config, current_config)
    drift_report.add_structural_drift(structural_drift)
    
    # Value drift detection
    value_drift = detect_value_changes(baseline_config, current_config, tolerance_thresholds)
    drift_report.add_value_drift(value_drift)
    
    # Semantic drift detection
    semantic_drift = detect_semantic_changes(baseline_config, current_config)
    drift_report.add_semantic_drift(semantic_drift)
    
    # Calculate drift severity
    drift_score = calculate_drift_score(drift_report)
    drift_report.set_severity(categorize_severity(drift_score))
    
    return drift_report

def detect_structural_changes(baseline, current):
    changes = []
    
    # Added configurations
    added_keys = set(flatten_dict(current).keys()) - set(flatten_dict(baseline).keys())
    for key in added_keys:
        changes.append(StructuralChange(
            type="addition",
            key=key,
            value=get_nested_value(current, key)
        ))
    
    # Removed configurations
    removed_keys = set(flatten_dict(baseline).keys()) - set(flatten_dict(current).keys())
    for key in removed_keys:
        changes.append(StructuralChange(
            type="removal",
            key=key,
            value=get_nested_value(baseline, key)
        ))
    
    return changes

def detect_value_changes(baseline, current, thresholds):
    changes = []
    common_keys = set(flatten_dict(baseline).keys()) & set(flatten_dict(current).keys())
    
    for key in common_keys:
        baseline_value = get_nested_value(baseline, key)
        current_value = get_nested_value(current, key)
        
        if baseline_value != current_value:
            change_magnitude = calculate_change_magnitude(baseline_value, current_value)
            threshold = thresholds.get(key, thresholds.get("default", 0.1))
            
            if change_magnitude > threshold:
                changes.append(ValueChange(
                    key=key,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    magnitude=change_magnitude,
                    threshold=threshold
                ))
    
    return changes

Continuous Drift Monitoring:
apiVersion: batch/v1
kind: CronJob
metadata:
  name: configuration-drift-detector
spec:
  schedule: "*/15 * * * *"  # Every 15 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: drift-detector
            image: ml-platform/config-drift-detector:latest
            env:
            - name: BASELINE_CONFIG_PATH
              value: "/config/baseline"
            - name: CURRENT_CONFIG_SOURCE
              value: "kubernetes_api"
            - name: DRIFT_THRESHOLDS_PATH
              value: "/config/thresholds.yaml"
            - name: ALERT_WEBHOOK_URL
              valueFrom:
                secretKeyRef:
                  name: monitoring-secrets
                  key: webhook_url
            command:
            - python
            - -c
            - |
              import yaml
              import requests
              import json
              from kubernetes import client, config
              
              # Load configuration
              config.load_incluster_config()
              v1 = client.CoreV1Api()
              apps_v1 = client.AppsV1Api()
              
              # Load baseline configuration
              with open('/config/baseline/deployment.yaml') as f:
                  baseline_config = yaml.safe_load(f)
              
              # Get current configuration from cluster
              current_deployments = apps_v1.list_deployment_for_all_namespaces()
              
              # Detect drift for each deployment
              for deployment in current_deployments.items:
                  if deployment.metadata.labels.get('drift-monitoring') == 'enabled':
                      drift_report = detect_drift(baseline_config, deployment)
                      
                      if drift_report.severity in ['high', 'critical']:
                          send_alert(drift_report)
            
            volumeMounts:
            - name: baseline-config
              mountPath: /config/baseline
            - name: drift-thresholds
              mountPath: /config/thresholds.yaml
              subPath: thresholds.yaml
          volumes:
          - name: baseline-config
            configMap:
              name: baseline-configurations
          - name: drift-thresholds
            configMap:
              name: drift-thresholds
          restartPolicy: OnFailure

Configuration Compliance Scanner:
class ConfigurationComplianceScanner:
    def __init__(self, compliance_rules, k8s_client):
        self.rules = compliance_rules
        self.k8s_client = k8s_client
    
    def scan_cluster_compliance(self):
        compliance_report = ComplianceReport()
        
        # Scan deployments
        deployments = self.k8s_client.list_deployment_for_all_namespaces()
        for deployment in deployments.items:
            deployment_compliance = self.check_deployment_compliance(deployment)
            compliance_report.add_deployment_result(deployment_compliance)
        
        # Scan services
        services = self.k8s_client.list_service_for_all_namespaces()
        for service in services.items:
            service_compliance = self.check_service_compliance(service)
            compliance_report.add_service_result(service_compliance)
        
        # Scan security policies
        security_compliance = self.check_security_policies()
        compliance_report.add_security_result(security_compliance)
        
        return compliance_report
    
    def check_deployment_compliance(self, deployment):
        violations = []
        
        # Check resource limits
        for container in deployment.spec.template.spec.containers:
            if not container.resources or not container.resources.limits:
                violations.append(ComplianceViolation(
                    rule="resource_limits_required",
                    severity="high",
                    description=f"Container {container.name} missing resource limits"
                ))
        
        # Check security context
        security_context = deployment.spec.template.spec.security_context
        if not security_context or security_context.run_as_root:
            violations.append(ComplianceViolation(
                rule="non_root_required",
                severity="medium",
                description="Container running as root user"
            ))
        
        # Check image policy
        for container in deployment.spec.template.spec.containers:
            if not self.is_approved_image(container.image):
                violations.append(ComplianceViolation(
                    rule="approved_images_only",
                    severity="high",
                    description=f"Unapproved image: {container.image}"
                ))
        
        return DeploymentComplianceResult(
            name=deployment.metadata.name,
            namespace=deployment.metadata.namespace,
            violations=violations,
            compliant=len(violations) == 0
        )
```

### **Configuration Remediation**

**Automated Remediation Strategies:**
```
Configuration Remediation Engine:
class ConfigurationRemediationEngine:
    def __init__(self, remediation_policies, k8s_client):
        self.policies = remediation_policies
        self.k8s_client = k8s_client
        self.dry_run = False
    
    def remediate_drift(self, drift_report):
        remediation_actions = []
        
        for drift_item in drift_report.drift_items:
            policy = self.find_applicable_policy(drift_item)
            if policy:
                action = self.create_remediation_action(drift_item, policy)
                remediation_actions.append(action)
        
        # Execute remediation actions
        results = []
        for action in remediation_actions:
            if self.dry_run:
                results.append(self.simulate_action(action))
            else:
                results.append(self.execute_action(action))
        
        return RemediationResult(
            actions_attempted=len(remediation_actions),
            actions_successful=len([r for r in results if r.success]),
            actions_failed=len([r for r in results if not r.success]),
            results=results
        )
    
    def find_applicable_policy(self, drift_item):
        for policy in self.policies:
            if policy.matches(drift_item):
                return policy
        return None
    
    def create_remediation_action(self, drift_item, policy):
        if policy.action_type == "restore_from_baseline":
            return RestoreAction(
                resource=drift_item.resource,
                baseline_value=drift_item.baseline_value,
                current_value=drift_item.current_value
            )
        elif policy.action_type == "update_to_standard":
            return UpdateAction(
                resource=drift_item.resource,
                target_value=policy.standard_value
            )
        elif policy.action_type == "alert_and_track":
            return AlertAction(
                drift_item=drift_item,
                alert_channels=policy.alert_channels
            )
        else:
            return NoOpAction(reason="No applicable action defined")

Remediation Policies Configuration:
remediation_policies:
  - name: "resource_limits_drift"
    match_criteria:
      resource_type: "deployment"
      drift_type: "resource_limits"
      severity: ["medium", "high"]
    action_type: "restore_from_baseline"
    approval_required: false
    max_attempts: 3
    
  - name: "security_context_violation"
    match_criteria:
      resource_type: "deployment"
      drift_type: "security_context"
      severity: ["high", "critical"]
    action_type: "update_to_standard"
    standard_value:
      run_as_non_root: true
      run_as_user: 1000
      fs_group: 2000
    approval_required: true
    
  - name: "image_policy_violation"
    match_criteria:
      resource_type: "deployment"
      drift_type: "image_policy"
      severity: ["critical"]
    action_type: "alert_and_track"
    alert_channels: ["security_team", "platform_team"]
    approval_required: false

GitOps-Based Remediation:
class GitOpsRemediationEngine:
    def __init__(self, git_repo, argocd_client):
        self.git_repo = git_repo
        self.argocd_client = argocd_client
    
    def remediate_via_gitops(self, drift_report):
        # Create remediation branch
        branch_name = f"auto-remediation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.git_repo.create_branch(branch_name)
        
        # Apply remediation changes to configuration files
        changes_made = []
        for drift_item in drift_report.high_priority_items:
            config_file = self.find_config_file(drift_item.resource)
            if config_file:
                change = self.apply_remediation_to_file(config_file, drift_item)
                changes_made.append(change)
        
        if changes_made:
            # Commit changes
            commit_message = f"Auto-remediation: Fix configuration drift\n\nDrift items fixed:\n"
            for change in changes_made:
                commit_message += f"- {change.description}\n"
            
            self.git_repo.commit_changes(commit_message)
            
            # Create pull request
            pr = self.git_repo.create_pull_request(
                title="Auto-remediation: Configuration drift fixes",
                body=self.generate_pr_description(drift_report, changes_made),
                base_branch="main",
                head_branch=branch_name,
                reviewers=["platform-team"]
            )
            
            return RemediationResult(
                method="gitops",
                pull_request_url=pr.url,
                changes_made=changes_made,
                requires_approval=True
            )
        
        return RemediationResult(
            method="gitops",
            changes_made=[],
            requires_approval=False
        )

Remediation Monitoring:
apiVersion: v1
kind: ConfigMap
metadata:
  name: remediation-dashboard
data:
  dashboard.json: |
    {
      "dashboard": {
        "title": "Configuration Drift Remediation",
        "panels": [
          {
            "title": "Drift Detection Rate",
            "targets": [
              {
                "expr": "rate(config_drift_detected_total[5m])",
                "legendFormat": "{{severity}}"
              }
            ]
          },
          {
            "title": "Remediation Success Rate",
            "targets": [
              {
                "expr": "rate(config_remediation_success_total[5m]) / rate(config_remediation_attempted_total[5m])",
                "legendFormat": "Success Rate"
              }
            ]
          },
          {
            "title": "Mean Time to Remediation",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(config_remediation_duration_seconds_bucket[5m]))",
                "legendFormat": "P95 Remediation Time"
              }
            ]
          }
        ]
      }
    }
```

This comprehensive framework for configuration management and secrets handling provides the theoretical foundations and practical strategies for building secure, compliant, and maintainable ML infrastructure. The key insight is that effective configuration management requires automated validation, drift detection, and remediation capabilities while maintaining strong security controls and audit trails throughout the configuration lifecycle.