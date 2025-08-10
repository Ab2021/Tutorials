# Day 2.2: Policy-as-Code - OPA, Sentinel, and Gatekeeper Implementation

## 🎯 Learning Objectives
By the end of this section, you will understand:
- Policy-as-Code fundamentals and implementation strategies
- Open Policy Agent (OPA) architecture and Rego language
- HashiCorp Sentinel policy engine for Terraform
- Kubernetes Gatekeeper for admission control
- Integration patterns for AI/ML infrastructure security
- Advanced policy scenarios and compliance frameworks

---

## 📚 Theoretical Foundation

### 1. Introduction to Policy-as-Code

#### 1.1 Policy-as-Code Fundamentals

**Definition and Core Concepts**:
Policy-as-Code (PaC) represents the practice of defining organizational policies through machine-readable code rather than human-readable documents. This approach enables automated policy enforcement, consistent application across environments, and version-controlled policy management.

**Why Policy-as-Code Matters for AI/ML Security**:
```
Traditional Policy Challenges:
- Manual policy interpretation and implementation
- Inconsistent enforcement across environments
- Reactive security posture with manual reviews
- Difficulty scaling security governance
- Limited audit trails for policy decisions

Policy-as-Code Benefits:
- Automated policy enforcement at deployment time
- Consistent security controls across infrastructure
- Proactive security with preventive controls
- Scalable governance for large AI/ML infrastructures
- Comprehensive audit trails and compliance reporting
- Version-controlled policy evolution
```

**AI/ML Specific Policy Requirements**:
```
Data Protection Policies:
- Encryption requirements for training datasets
- Access controls for sensitive model data
- Geographic restrictions for data processing
- Retention policies for experimental data

Infrastructure Security Policies:
- GPU resource allocation and isolation
- Network segmentation for training clusters
- Compliance requirements for regulatory frameworks
- Resource cost optimization and quotas

Operational Policies:
- Model deployment approval workflows
- Training job resource limits and priorities
- Audit logging for model access and changes
- Incident response and remediation procedures
```

#### 1.2 Policy Enforcement Patterns

**Admission Control Pattern**:
```
Pre-deployment Validation:
- Policy evaluation before resource creation
- Rejection of non-compliant configurations
- Early feedback in development cycle
- Prevention rather than detection

Application in AI/ML:
- Validate GPU instance configurations
- Ensure encryption for training data storage
- Check network isolation requirements
- Verify compliance with data residency rules
```

**Runtime Monitoring Pattern**:
```
Continuous Compliance Validation:
- Ongoing policy evaluation of running systems
- Detection of configuration drift
- Automated remediation of violations
- Real-time security posture assessment

Application in AI/ML:
- Monitor training job resource usage
- Detect unauthorized data access patterns
- Validate model serving configurations
- Ensure ongoing compliance with regulations
```

**Advisory Pattern**:
```
Policy Guidance and Recommendations:
- Non-blocking policy evaluation
- Warning and advisory messages
- Best practice recommendations
- Gradual policy adoption

Application in AI/ML:
- Recommend optimization opportunities
- Suggest security improvements
- Provide cost optimization guidance
- Offer compliance enhancement suggestions
```

### 2. Open Policy Agent (OPA) Architecture

#### 2.1 OPA Core Architecture

**OPA Components and Design**:
```
Core Components:
- Policy Engine: Evaluates policies written in Rego
- Data Store: Holds policy data and external context
- REST API: Provides policy decision endpoints
- Bundle System: Manages policy distribution and updates

Decision Flow:
1. Input data received (resource configuration, context)
2. Policy evaluation against Rego rules
3. Decision output (allow/deny, violations, warnings)
4. Optional data transformation or enrichment
```

**Rego Language Fundamentals**:
Rego is a declarative query language designed for policy definition:

```rego
# Basic Rego syntax
package example

# Simple rule
allow {
    input.action == "read"
    input.user.role == "admin"
}

# Rule with conditions
deny[msg] {
    input.resource.type == "gpu_instance"
    not input.resource.encrypted
    msg := "GPU instances must have encrypted storage"
}

# Complex data manipulation
violations[violation] {
    instance := input.instances[_]
    instance.type == "p3.2xlarge"
    not instance.security_groups[_] == "ml-training-sg"
    violation := {
        "resource": instance.id,
        "message": "GPU instances must use ml-training security group"
    }
}
```

#### 2.2 OPA Integration Patterns

**Infrastructure-as-Code Integration**:
```
Terraform Integration:
- OPA policy evaluation during terraform plan
- Custom policy validation for resource configurations
- Integration with terraform-compliance tool
- Policy decisions influencing resource creation

Example Terraform-OPA Integration:
# terraform/policies/security.rego
package terraform.security

deny[msg] {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_instance"
    resource.values.instance_type == "p3.2xlarge"
    not resource.values.vpc_security_group_ids[_] == "sg-ml-training"
    msg := sprintf("GPU instance %s must use ml-training security group", [resource.address])
}

# Usage in CI/CD pipeline
terraform plan -out=tfplan
terraform show -json tfplan | opa eval -d policies/ -I "data.terraform.security.deny[x]"
```

**Kubernetes Integration with OPA**:
```
Admission Controller Integration:
- ValidatingAdmissionWebhook with OPA
- Policy evaluation for Kubernetes resources
- Dynamic policy updates without cluster restart
- Integration with GitOps workflows

Example Kubernetes Policy:
package kubernetes.admission

deny[msg] {
    input.request.kind.kind == "Pod"
    input.request.object.spec.containers[_].image
    not starts_with(input.request.object.spec.containers[_].image, "registry.company.com/")
    msg := "Containers must use approved registry"
}

deny[msg] {
    input.request.kind.kind == "Pod"
    input.request.object.metadata.labels["ml-workload"] == "training"
    not input.request.object.spec.securityContext.runAsNonRoot
    msg := "ML training pods must run as non-root user"
}
```

#### 2.3 Advanced OPA Scenarios

**Multi-Tenant Policy Management**:
```
Tenant-Specific Policies:
package multitenancy

# Base policy for all tenants
default allow = false

# Tenant-specific resource limits
allow {
    input.tenant.name == "research-team-a"
    input.request.resource == "gpu_hours"
    input.request.amount <= data.tenants["research-team-a"].gpu_quota
}

allow {
    input.tenant.name == "research-team-b"
    input.request.resource == "gpu_hours"
    input.request.amount <= data.tenants["research-team-b"].gpu_quota
}

# Cross-tenant data access control
deny[msg] {
    input.action == "read"
    input.resource.data_classification == "confidential"
    input.tenant.name != input.resource.owner_tenant
    msg := "Cross-tenant access to confidential data not allowed"
}
```

**Data-Driven Policy Decisions**:
```
External Data Integration:
package compliance

import future.keywords.if
import future.keywords.in

# Policy using external compliance database
deny[msg] if {
    resource := input.resource
    compliance_requirement := data.compliance_db[resource.region][resource.type]
    not compliance_requirement.encryption_required
    resource.encrypted == false
    msg := sprintf("Resource %s in region %s requires encryption", [resource.id, resource.region])
}

# Dynamic policy based on threat intelligence
deny[msg] if {
    ip := input.request.source_ip
    ip in data.threat_intel.malicious_ips
    msg := sprintf("Request from malicious IP %s blocked", [ip])
}

# Time-based policy enforcement
allow if {
    time.now_ns() > data.maintenance_windows[input.service].start
    time.now_ns() < data.maintenance_windows[input.service].end
    input.action == "maintenance"
}
```

### 3. HashiCorp Sentinel Integration

#### 3.1 Sentinel Architecture and Language

**Sentinel Policy Language**:
Sentinel is HashiCorp's policy-as-code framework with a focus on infrastructure automation:

```sentinel
# Basic Sentinel policy structure
import "tfplan"
import "strings"

# Policy rule
main = rule {
    all tfplan.resource_changes as _, rc {
        rc.type is "aws_instance" and
        rc.change.after.instance_type contains "p3" implies
        rc.change.after.vpc_security_group_ids contains "sg-ml-training"
    }
}

# Helper function
get_gpu_instances = func() {
    instances = []
    for tfplan.resource_changes as _, rc {
        if rc.type is "aws_instance" and strings.has_prefix(rc.change.after.instance_type, "p3") {
            append(instances, rc)
        }
    }
    return instances
}

# Advanced rule with custom logic
enforce_ml_security = rule {
    gpu_instances = get_gpu_instances()
    all gpu_instances as instance {
        instance.change.after.ebs_block_device else [] as ebs_devices {
            all ebs_devices as device {
                device.encrypted is true
            }
        }
    }
}
```

#### 3.2 Terraform Enterprise Integration

**Policy Set Configuration**:
```
Terraform Enterprise Policy Integration:
- Policy sets attached to workspaces
- Three enforcement levels: advisory, soft-mandatory, hard-mandatory
- Policy evaluation during plan and apply phases
- Integration with VCS for policy management

Example Policy Set Structure:
sentinel-policies/
├── policy-sets/
│   ├── ml-security/
│   │   ├── sentinel.hcl
│   │   ├── encryption-required.sentinel
│   │   ├── network-security.sentinel
│   │   └── resource-limits.sentinel
│   └── compliance/
│       ├── sentinel.hcl
│       ├── data-residency.sentinel
│       └── audit-logging.sentinel
└── test/
    ├── encryption-required/
    └── network-security/

# sentinel.hcl configuration
policy "encryption-required" {
    source = "./encryption-required.sentinel"
    enforcement_level = "hard-mandatory"
}

policy "network-security" {
    source = "./network-security.sentinel"
    enforcement_level = "soft-mandatory"
}
```

**AI/ML Specific Sentinel Policies**:
```sentinel
# GPU resource allocation policy
import "tfplan"
import "decimal"

# Calculate total GPU hours requested
total_gpu_hours = func() {
    total = 0
    for tfplan.resource_changes as _, rc {
        if rc.type is "aws_instance" and strings.has_prefix(rc.change.after.instance_type, "p3") {
            # Extract GPU count from instance type
            if rc.change.after.instance_type is "p3.2xlarge" {
                gpu_count = 1
            } else if rc.change.after.instance_type is "p3.8xlarge" {
                gpu_count = 4
            } else if rc.change.after.instance_type is "p3.16xlarge" {
                gpu_count = 8
            } else {
                gpu_count = 0
            }
            
            # Assume 24/7 usage for simplicity
            total += gpu_count * 24 * 30  # 30 days
        }
    }
    return total
}

# Main policy rule
main = rule {
    decimal.new(total_gpu_hours()) <= decimal.new(1000)  # 1000 GPU hours limit
}

# Print statement for debugging
print("Total GPU hours requested:", total_gpu_hours())
```

#### 3.3 Advanced Sentinel Patterns

**Cost Control Policies**:
```sentinel
# Cost estimation and control
import "tfplan"
import "tfrun"

# Calculate estimated monthly cost
estimated_cost = func() {
    cost = 0
    
    # Instance costs (simplified)
    instance_costs = {
        "p3.2xlarge":  3.06 * 24 * 30,  # Per month
        "p3.8xlarge":  12.24 * 24 * 30,
        "p3.16xlarge": 24.48 * 24 * 30,
    }
    
    for tfplan.resource_changes as _, rc {
        if rc.type is "aws_instance" and rc.change.actions contains "create" {
            instance_type = rc.change.after.instance_type
            if instance_type in keys(instance_costs) {
                cost += instance_costs[instance_type]
            }
        }
    }
    
    return cost
}

# Budget enforcement
budget_limit = 10000  # $10,000 per month

main = rule {
    estimated_cost() <= budget_limit
}

# Warning for high costs
cost_warning = rule when estimated_cost() > budget_limit * 0.8 {
    print("Warning: Estimated cost", estimated_cost(), "approaching budget limit", budget_limit)
    true
}
```

### 4. Kubernetes Gatekeeper Implementation

#### 4.1 Gatekeeper Architecture

**OPA Gatekeeper Components**:
```
Gatekeeper Architecture:
- Admission Controller: Validates Kubernetes resources
- Constraint Templates: Define policy structure and logic
- Constraints: Instantiate policies with specific parameters
- Config: Manages data replication and sync
- Audit: Evaluates existing resources for compliance

Component Interaction:
1. ConstraintTemplate defines policy schema and Rego rules
2. Constraint instantiates template with specific parameters
3. Admission controller evaluates resources against constraints
4. Audit controller periodically checks existing resources
```

**ConstraintTemplate Structure**:
```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mlsecurityrequirements
spec:
  crd:
    spec:
      names:
        kind: MLSecurityRequirements
      validation:
        openAPIV3Schema:
          type: object
          properties:
            requiredLabels:
              type: array
              items:
                type: string
            allowedRegistries:
              type: array
              items:
                type: string
            requireEncryption:
              type: boolean
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mlsecurityrequirements
        
        violation[{"msg": msg}] {
          required := input.parameters.requiredLabels
          provided := input.review.object.metadata.labels
          missing := required[_]
          not provided[missing]
          msg := sprintf("Missing required label: %v", [missing])
        }
        
        violation[{"msg": msg}] {
          image := input.review.object.spec.containers[_].image
          registry := input.parameters.allowedRegistries
          not startswith(image, registry[_])
          msg := sprintf("Container image %v not from approved registry", [image])
        }
        
        violation[{"msg": msg}] {
          input.parameters.requireEncryption
          input.review.object.kind == "PersistentVolumeClaim"
          not input.review.object.metadata.annotations["encrypted"] == "true"
          msg := "PersistentVolumeClaims must be encrypted for ML workloads"
        }
```

#### 4.2 AI/ML Security Constraints

**GPU Resource Management**:
```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: gpuresourcecontrol
spec:
  crd:
    spec:
      names:
        kind: GPUResourceControl
      validation:
        openAPIV3Schema:
          type: object
          properties:
            maxGPUPerPod:
              type: integer
            allowedNodeSelectors:
              type: array
              items:
                type: string
            requiredTolerations:
              type: array
              items:
                type: object
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package gpuresourcecontrol
        
        violation[{"msg": msg}] {
          gpu_limit := input.review.object.spec.containers[_].resources.limits["nvidia.com/gpu"]
          to_number(gpu_limit) > input.parameters.maxGPUPerPod
          msg := sprintf("Pod requests %v GPUs, maximum allowed is %v", [gpu_limit, input.parameters.maxGPUPerPod])
        }
        
        violation[{"msg": msg}] {
          input.review.object.spec.containers[_].resources.limits["nvidia.com/gpu"]
          not input.review.object.spec.nodeSelector
          msg := "GPU pods must specify node selector"
        }
        
        violation[{"msg": msg}] {
          input.review.object.spec.containers[_].resources.limits["nvidia.com/gpu"]
          node_selector := input.review.object.spec.nodeSelector
          allowed := input.parameters.allowedNodeSelectors
          not node_selector[allowed[_]]
          msg := sprintf("GPU pods must use allowed node selectors: %v", [allowed])
        }

---
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- constraint-template.yaml

patchesStrategicMerge:
- constraint.yaml

# constraint.yaml
apiVersion: gatekeeper.sh/v1beta1
kind: GPUResourceControl
metadata:
  name: gpu-limits
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["ml-training", "ml-inference"]
  parameters:
    maxGPUPerPod: 8
    allowedNodeSelectors: ["gpu-type", "instance-family"]
    requiredTolerations:
      - key: "nvidia.com/gpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"
```

**Data Security and Compliance**:
```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mldatasecurity
spec:
  crd:
    spec:
      names:
        kind: MLDataSecurity
      validation:
        openAPIV3Schema:
          type: object
          properties:
            dataClassifications:
              type: array
              items:
                type: string
            encryptionRequired:
              type: boolean
            auditLogRequired:
              type: boolean
            regionRestrictions:
              type: array
              items:
                type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mldatasecurity
        
        violation[{"msg": msg}] {
          input.parameters.encryptionRequired
          volume := input.review.object.spec.volumes[_]
          volume.persistentVolumeClaim
          not input.review.object.metadata.annotations["data.encryption"] == "enabled"
          msg := "ML workloads with persistent volumes must enable encryption"
        }
        
        violation[{"msg": msg}] {
          data_class := input.review.object.metadata.labels["data-classification"]
          data_class == "sensitive"
          not input.review.object.spec.securityContext.runAsNonRoot
          msg := "Pods processing sensitive data must run as non-root"
        }
        
        violation[{"msg": msg}] {
          input.parameters.auditLogRequired
          input.review.object.metadata.labels["ml-workload"]
          not input.review.object.metadata.annotations["audit.logging"] == "enabled"
          msg := "ML workloads must enable audit logging"
        }
        
        violation[{"msg": msg}] {
          node_selector := input.review.object.spec.nodeSelector
          node_region := node_selector["topology.kubernetes.io/region"]
          allowed_regions := input.parameters.regionRestrictions
          count(allowed_regions) > 0
          not node_region in allowed_regions
          msg := sprintf("Pod must run in allowed regions: %v", [allowed_regions])
        }
```

#### 4.3 Advanced Gatekeeper Patterns

**Multi-Tenant Policy Enforcement**:
```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mlmultitenancy
spec:
  crd:
    spec:
      names:
        kind: MLMultiTenancy
      validation:
        openAPIV3Schema:
          type: object
          properties:
            tenantNamespaces:
              type: object
              additionalProperties:
                type: object
                properties:
                  resourceQuota:
                    type: object
                  allowedImages:
                    type: array
                    items:
                      type: string
                  networkPolicies:
                    type: array
                    items:
                      type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mlmultitenancy
        
        violation[{"msg": msg}] {
          namespace := input.review.object.metadata.namespace
          tenant_config := input.parameters.tenantNamespaces[namespace]
          
          # Check resource limits
          container := input.review.object.spec.containers[_]
          gpu_request := to_number(container.resources.requests["nvidia.com/gpu"])
          max_gpu := tenant_config.resourceQuota.maxGPU
          gpu_request > max_gpu
          msg := sprintf("GPU request %v exceeds tenant limit %v", [gpu_request, max_gpu])
        }
        
        violation[{"msg": msg}] {
          namespace := input.review.object.metadata.namespace
          tenant_config := input.parameters.tenantNamespaces[namespace]
          
          # Check allowed images
          image := input.review.object.spec.containers[_].image
          allowed_images := tenant_config.allowedImages
          not startswith(image, allowed_images[_])
          msg := sprintf("Image %v not allowed for tenant in namespace %v", [image, namespace])
        }
        
        violation[{"msg": msg}] {
          namespace := input.review.object.metadata.namespace
          tenant_config := input.parameters.tenantNamespaces[namespace]
          
          # Enforce network policies
          input.review.object.kind == "Pod"
          required_policies := tenant_config.networkPolicies
          count(required_policies) > 0
          not input.review.object.metadata.labels["network-policy"]
          msg := sprintf("Pods in namespace %v must specify network policy", [namespace])
        }
```

### 5. Integration and Orchestration Patterns

#### 5.1 CI/CD Pipeline Integration

**GitOps Policy Management**:
```
Policy Lifecycle Management:
1. Policy Development: Version-controlled policy definitions
2. Testing: Automated policy testing and validation
3. Review: Code review process for policy changes
4. Deployment: Automated policy distribution and activation
5. Monitoring: Continuous monitoring of policy effectiveness

Example GitOps Workflow:
.github/workflows/policy-deployment.yml:
name: Policy Deployment

on:
  push:
    branches: [main]
    paths: ['policies/**']

jobs:
  test-policies:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install OPA
      run: |
        curl -L -o opa https://github.com/open-policy-agent/opa/releases/download/v0.50.0/opa_linux_amd64
        chmod +x opa
        sudo mv opa /usr/local/bin/
    
    - name: Test OPA Policies
      run: |
        opa test policies/
    
    - name: Validate Gatekeeper Templates
      run: |
        kubectl apply --dry-run=client -f gatekeeper/templates/
    
    - name: Test Sentinel Policies
      run: |
        sentinel test policies/sentinel/

  deploy-policies:
    needs: test-policies
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to OPA
      run: |
        # Upload policies to OPA bundle server
        tar -czf policies.tar.gz policies/
        curl -X PUT --data-binary @policies.tar.gz \
          "${OPA_BUNDLE_SERVER}/bundles/policies"
    
    - name: Deploy Gatekeeper Constraints
      run: |
        kubectl apply -f gatekeeper/templates/
        kubectl apply -f gatekeeper/constraints/
    
    - name: Update Terraform Enterprise
      run: |
        # Upload Sentinel policies to TFE
        curl -X POST \
          -H "Authorization: Bearer ${TFE_TOKEN}" \
          -F "data=@sentinel-policies.tar.gz" \
          "${TFE_API}/policy-sets/${POLICY_SET_ID}/versions"
```

#### 5.2 Multi-Platform Policy Orchestration

**Unified Policy Management**:
```
Cross-Platform Policy Strategy:
- Abstract policy requirements into platform-agnostic definitions
- Implement platform-specific policy translations
- Maintain consistency across different enforcement points
- Centralized policy management and monitoring

Example Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Policy Store  │    │  Policy Engine  │    │ Enforcement     │
│   (Git/DB)      │───▶│  (OPA/Sentinel) │───▶│ Points          │
│                 │    │                 │    │ (K8s/TF/Cloud)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌────▼────┐             ┌────▼────┐             ┌────▼────┐
    │ Policy  │             │ Policy  │             │ Violation│
    │ Authoring│             │ Testing │             │ Handling│
    │ Tools   │             │ & Valid.│             │ & Audit │
    └─────────┘             └─────────┘             └─────────┘
```

**Example Unified Policy Definition**:
```yaml
# policy-definition.yaml
apiVersion: policy.company.com/v1
kind: SecurityPolicy
metadata:
  name: ml-encryption-requirement
  version: "1.0"
spec:
  title: "ML Infrastructure Encryption Requirement"
  description: "All ML infrastructure must use encryption at rest and in transit"
  
  scope:
    - kubernetes
    - terraform
    - cloud-resources
  
  requirements:
    - name: "storage-encryption"
      description: "All storage resources must be encrypted"
      severity: "high"
      platforms:
        kubernetes:
          resources: ["PersistentVolumeClaim", "Secret"]
          enforcement: "deny"
        terraform:
          resources: ["aws_ebs_volume", "aws_s3_bucket", "azurerm_storage_account"]
          enforcement: "hard-mandatory"
        aws:
          services: ["s3", "ebs", "rds"]
          enforcement: "preventive"
    
    - name: "network-encryption"
      description: "All network communication must be encrypted"
      severity: "medium"
      platforms:
        kubernetes:
          resources: ["Service", "Ingress"]
          enforcement: "warn"
        terraform:
          resources: ["aws_lb_listener", "azurerm_application_gateway"]
          enforcement: "soft-mandatory"

  compliance:
    frameworks: ["SOC2", "GDPR", "HIPAA"]
    controls: ["CC6.1", "Article 32", "164.312(a)(1)"]
```

### 6. Advanced Policy Scenarios

#### 6.1 Dynamic Policy Adaptation

**Context-Aware Policy Enforcement**:
```rego
package dynamic_policies

import future.keywords.if
import future.keywords.in

# Time-based policy enforcement
allow_gpu_access if {
    time.now_ns() >= data.business_hours.start
    time.now_ns() <= data.business_hours.end
    input.user.role == "researcher"
}

# Cost-based policy adaptation
allow_expensive_instance if {
    input.instance.cost_per_hour <= data.budget.hourly_limit
    input.project.remaining_budget > input.instance.cost_per_hour * 24
}

# Threat-level-based policy adjustment
require_additional_approval if {
    data.threat_level.current == "high"
    input.resource.type == "gpu_cluster"
    input.resource.public_access == true
}

# Load-based resource allocation
max_gpu_allocation := allocation if {
    current_load := data.cluster.current_gpu_utilization
    current_load < 0.7
    allocation := input.user.max_gpu_quota
} else := allocation if {
    allocation := input.user.max_gpu_quota * 0.5
}
```

#### 6.2 Compliance Framework Integration

**Automated Compliance Validation**:
```rego
package compliance

import future.keywords.if
import future.keywords.in

# SOC 2 Type II compliance checks
soc2_violations[violation] if {
    # CC6.1 - Logical and physical access controls
    input.resource.type == "database"
    not input.resource.access_controls.mfa_required
    violation := {
        "control": "CC6.1",
        "description": "Multi-factor authentication required for database access",
        "severity": "high",
        "resource": input.resource.id
    }
}

# GDPR Article 32 compliance
gdpr_violations[violation] if {
    # Technical and organizational measures
    input.data.classification == "personal"
    not input.resource.encryption.at_rest
    violation := {
        "article": "32",
        "description": "Personal data must be encrypted at rest",
        "severity": "critical",
        "resource": input.resource.id
    }
}

# HIPAA 164.312(a)(1) compliance
hipaa_violations[violation] if {
    input.data.type == "phi"
    not input.resource.access_controls.unique_user_identification
    violation := {
        "section": "164.312(a)(1)",
        "description": "Unique user identification required for PHI access",
        "severity": "high",
        "resource": input.resource.id
    }
}

# Consolidated compliance report
compliance_status := {
    "compliant": count(all_violations) == 0,
    "violations": all_violations,
    "frameworks": ["SOC2", "GDPR", "HIPAA"]
} if {
    all_violations := array.concat(array.concat(soc2_violations, gdpr_violations), hipaa_violations)
}
```

---

## 🔍 Key Questions

### Beginner Level

1. **Q**: What is Policy-as-Code and how does it improve security governance compared to traditional policy management?
   **A**: Policy-as-Code defines organizational policies through machine-readable code, enabling automated enforcement, consistent application, version control, and scalable governance compared to manual, document-based policies.

2. **Q**: What are the three main enforcement levels in HashiCorp Sentinel, and when would you use each?
   **A**: Advisory (warnings/recommendations), Soft-mandatory (blocking with override capability), Hard-mandatory (absolute blocking). Use advisory for guidance, soft-mandatory for gradual adoption, hard-mandatory for critical security requirements.

3. **Q**: How does OPA Gatekeeper integrate with Kubernetes to enforce policies?
   **A**: Gatekeeper acts as a validating admission controller, evaluating Kubernetes resources against ConstraintTemplate policies written in Rego before allowing resource creation or modification.

### Intermediate Level

4. **Q**: Design a comprehensive policy set for an AI/ML training cluster that enforces GPU resource limits, network isolation, and data encryption requirements.
   **A**: 
   ```
   Policies needed:
   1. GPU Resource Policy: Limit GPU allocation per user/project, require node selectors
   2. Network Isolation Policy: Mandate security groups, deny public IPs, require VPC
   3. Data Encryption Policy: Enforce encryption at rest/transit, require KMS keys
   4. Access Control Policy: Implement RBAC, require MFA, audit access
   5. Compliance Policy: Tag resources, enable logging, data residency rules
   
   Implementation across platforms (OPA for K8s, Sentinel for Terraform)
   ```

5. **Q**: How would you implement a cost control policy that dynamically adjusts based on budget constraints and project priorities?
   **A**: Create dynamic policies that evaluate current budget utilization, project priority levels, and resource costs. Implement tiered approval workflows and automatic resource scaling based on budget thresholds.

6. **Q**: Explain how to implement multi-tenant policy isolation in a shared AI/ML platform using OPA and Gatekeeper.
   **A**: Use namespace-based isolation with tenant-specific constraints, implement RBAC for policy management, create tenant-specific resource quotas, and enforce cross-tenant data access restrictions.

### Advanced Level

7. **Q**: Design a policy framework that ensures compliance with GDPR, SOC2, and HIPAA simultaneously for a healthcare AI platform.
   **A**: 
   ```
   Framework components:
   - Data classification and tagging policies
   - Encryption requirements (GDPR Article 32, HIPAA 164.312)
   - Access control and audit logging (SOC2 CC6.1, HIPAA 164.308)
   - Data residency and cross-border transfer restrictions
   - Breach detection and notification policies
   - Regular compliance validation and reporting
   - Automated remediation for violations
   ```

8. **Q**: How would you implement a policy system that adapts enforcement based on real-time threat intelligence and security posture?
   **A**: Integrate external threat feeds with OPA data, implement dynamic policy evaluation based on threat levels, create context-aware rules that adjust based on security events, and implement automated response escalation.

### Tricky Questions

9. **Q**: You need to migrate from a legacy manual approval process to automated policy enforcement for a large-scale AI research environment. The researchers are concerned about productivity impact. Design a migration strategy that balances security and usability.
   **A**: 
   ```
   Phased Migration Strategy:
   1. Assessment Phase: Analyze current approval patterns and pain points
   2. Policy Definition: Codify existing policies with researcher input
   3. Advisory Phase: Deploy policies in warning-only mode for feedback
   4. Gradual Enforcement: Start with soft-mandatory for non-critical policies
   5. Exception Handling: Implement emergency override procedures
   6. Monitoring and Adjustment: Track policy violations and adjust thresholds
   7. Full Enforcement: Move to hard-mandatory for critical security policies
   
   Change Management:
   - Training programs for researchers
   - Clear communication about benefits
   - Feedback collection and rapid iteration
   - Success metrics and regular reviews
   ```

10. **Q**: Design a policy architecture for a federated learning system where multiple organizations contribute data but policies must be enforced consistently across all participants while respecting organizational autonomy.
    **A**: 
    ```
    Federated Policy Architecture:
    - Core Security Baseline: Minimum shared policies all participants must adopt
    - Organization-Specific Extensions: Additional policies per organization
    - Policy Negotiation Protocol: Automated policy compatibility checking
    - Distributed Enforcement: Local policy engines with global coordination
    - Compliance Verification: Cross-organizational audit capabilities
    - Policy Synchronization: Updates propagated across federation
    - Conflict Resolution: Automated resolution or escalation procedures
    
    Implementation:
    - OPA for local enforcement with shared policy bundles
    - Blockchain or consensus mechanism for policy agreement
    - Zero-knowledge proofs for compliance verification
    - Graduated response for policy violations
    ```

---

## 🛡️ Security Deep Dive

### Policy Security and Integrity

#### Policy Tampering Prevention

**Policy Integrity Verification**:
```
Security Measures:
- Cryptographic signing of policy bundles
- Hash verification for policy distribution
- Immutable policy storage systems
- Version control with digital signatures
- Audit trails for all policy changes

Implementation Example:
# Policy bundle signing
gpg --detach-sign --armor policies.tar.gz
opa build --signing-key private.pem policies/

# Verification during deployment
opa run --verification-key public.pem --bundle signed-policies.tar.gz
```

#### Privilege Escalation Prevention

**Policy Administration Security**:
```
Security Controls:
- Separation of policy authoring and enforcement
- Multi-person approval for critical policies
- Least privilege for policy administrators
- Regular access reviews and rotation
- Monitoring of policy modification attempts
```

### Advanced Threat Scenarios

#### Policy Bypass Attacks

**Attack Vectors and Mitigations**:
```
Common Attack Patterns:
- Resource label manipulation to bypass policies
- Timing attacks during policy updates
- Context manipulation in policy evaluation
- Privilege escalation through policy loopholes

Mitigation Strategies:
- Comprehensive input validation in policies
- Atomic policy updates with rollback capabilities
- Context validation and sanitization
- Regular policy review and penetration testing
```

---

## 🚀 Performance Optimization

### Policy Evaluation Performance

#### OPA Performance Optimization

**Query Optimization Techniques**:
```rego
# Efficient policy structure
package optimized

# Use indexed data structures
users_by_role := {role: users |
    users := [user | data.users[user].role == role]
    role := data.users[_].role
}

# Minimize expensive operations
allow {
    user_role := data.users[input.user].role
    user_role == "admin"  # Direct comparison vs iteration
}

# Use partial evaluation
policy[decision] {
    decision := data.precomputed_decisions[input.resource.type][input.action]
}
```

#### Gatekeeper Performance Tuning

**Resource Management**:
```yaml
# Gatekeeper resource limits
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gatekeeper-system
  namespace: gatekeeper-system
spec:
  hard:
    requests.cpu: "2"
    requests.memory: "4Gi"
    limits.cpu: "4"
    limits.memory: "8Gi"

# Controller configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: gatekeeper-manager-config
  namespace: gatekeeper-system
data:
  config.yaml: |
    webhook:
      port: 8443
      cert-dir: /certs
    controller:
      webhook-timeout: 30s
      admission-evaluation-enabled: true
      audit-evaluation-enabled: true
      constraint-violations-limit: 1000
```

---

## 📝 Practical Exercises

### Exercise 1: Comprehensive Policy Framework
Design and implement a complete policy-as-code framework for an AI research institution that includes:
- Multi-tenant resource allocation policies
- Data classification and protection policies
- Cost optimization and budget control policies
- Compliance framework integration (choose 2-3 frameworks)
- Emergency access and override procedures

### Exercise 2: Cross-Platform Policy Implementation
Create a unified policy that enforces the same security requirements across:
- Kubernetes clusters (using Gatekeeper)
- Terraform infrastructure (using Sentinel)
- AWS resources (using OPA)
- Include testing strategies for each platform

### Exercise 3: Dynamic Policy System
Implement a dynamic policy system that:
- Adjusts enforcement based on threat levels
- Implements time-based access controls
- Provides context-aware decision making
- Includes performance monitoring and optimization

### Exercise 4: Policy Migration Strategy
Design a complete migration plan for transitioning from manual security reviews to automated policy enforcement for a large AI/ML platform, including:
- Current state assessment
- Stakeholder engagement strategy
- Phased implementation timeline
- Risk mitigation and rollback procedures
- Success metrics and monitoring

---

## 🔗 Next Steps
In the next section (day02_003), we'll dive deep into IaC module hardening and least privilege IAM implementation, focusing on securing infrastructure components and implementing robust access control mechanisms for AI/ML environments.