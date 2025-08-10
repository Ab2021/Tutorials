# Day 2.1: Infrastructure-as-Code Security with Terraform, Pulumi, and Crossplane

## 🎯 Learning Objectives
By the end of this section, you will understand:
- Infrastructure-as-Code (IaC) fundamentals and security principles
- Secure implementation practices for Terraform, Pulumi, and Crossplane
- Security scanning and validation techniques for IaC
- Best practices for managing AI/ML infrastructure through code
- Integration with CI/CD pipelines for automated security

---

## 📚 Theoretical Foundation

### 1. Introduction to Infrastructure-as-Code Security

#### 1.1 IaC Fundamentals in AI/ML Context

**Definition and Core Principles**:
Infrastructure-as-Code represents the practice of managing and provisioning computing infrastructure through machine-readable definition files, rather than physical hardware configuration or interactive configuration tools. In AI/ML environments, this becomes critical due to the complexity and scale of required infrastructure.

**Why IaC Security Matters for AI/ML**:
```
Traditional Infrastructure Challenges:
- Manual configuration errors leading to security gaps
- Inconsistent security policies across environments
- Difficulty in auditing infrastructure changes
- Slow response to security vulnerabilities
- Limited scalability for dynamic AI workloads

IaC Security Benefits:
- Version-controlled security configurations
- Automated security policy enforcement
- Consistent security baselines across environments
- Rapid deployment of security patches
- Immutable infrastructure principles
```

**AI/ML Infrastructure Complexity**:
Modern AI/ML infrastructure involves multiple interconnected components:
```
Compute Layer:
- GPU clusters for training workloads
- CPU nodes for inference serving
- Edge devices for distributed inference
- Auto-scaling groups for dynamic workloads

Storage Layer:
- Object storage for datasets and models
- High-performance file systems for training data
- Database systems for metadata and logs
- Backup and disaster recovery systems

Network Layer:
- High-bandwidth networks for distributed training
- Load balancers for inference endpoints
- VPCs and security groups for isolation
- Service mesh for microservices communication

Security Layer:
- Identity and access management systems
- Encryption key management services
- Monitoring and logging infrastructure
- Compliance and audit systems
```

#### 1.2 Security Challenges in IaC

**Common Security Anti-patterns**:
```
Hardcoded Secrets:
Problem: Passwords, API keys, certificates embedded in code
Impact: Credential exposure in version control systems
Example: Database passwords in Terraform variables

Overprivileged Access:
Problem: Broad permissions granted for convenience
Impact: Increased blast radius during security incidents
Example: Admin access for all service accounts

Insecure Defaults:
Problem: Using default configurations without hardening
Impact: Vulnerable services exposed to attacks
Example: Default security groups allowing all traffic

Lack of Validation:
Problem: No security scanning or policy enforcement
Impact: Deployment of non-compliant infrastructure
Example: Missing encryption for sensitive data stores
```

**Security Shift-Left Principles**:
```
Development Phase:
- Security policy definition in code
- Local security testing and validation
- Secure coding practices and training
- Threat modeling for infrastructure design

CI/CD Pipeline:
- Automated security scanning of IaC code
- Policy validation before deployment
- Credential management and rotation
- Deployment approval workflows

Runtime Phase:
- Continuous compliance monitoring
- Drift detection and remediation
- Security event monitoring and alerting
- Regular security assessments and updates
```

### 2. Terraform Security Implementation

#### 2.1 Terraform Security Architecture

**State Management Security**:
Terraform state files contain sensitive information about infrastructure and must be secured:

**Remote State Storage**:
```
Security Requirements for State Storage:
- Encryption at rest using customer-managed keys
- Encryption in transit for all state operations
- Access control based on least privilege principles
- Versioning and backup for state recovery
- Audit logging for all state modifications

AWS S3 Backend Security:
- Server-side encryption with KMS keys
- Bucket versioning and MFA delete protection
- VPC endpoints for private access
- CloudTrail logging for API calls
- IAM policies restricting access

Azure Storage Backend Security:
- Storage service encryption with customer keys
- Virtual network service endpoints
- Shared access signatures with time limits
- Activity logging and monitoring
- Role-based access control (RBAC)
```

**Provider Security Configuration**:
```
AWS Provider Security:
- AssumeRole for cross-account access
- Session tokens with limited lifetime
- Regional restrictions for resource deployment
- Resource tagging for governance and compliance

provider "aws" {
  region = var.aws_region
  
  assume_role {
    role_arn     = "arn:aws:iam::ACCOUNT:role/TerraformRole"
    session_name = "terraform-deployment"
    external_id  = var.external_id
  }
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
      SecurityLevel = var.security_classification
    }
  }
}

Azure Provider Security:
- Service principal authentication
- Certificate-based authentication
- Managed identity integration
- Subscription and resource group restrictions
```

#### 2.2 Secure Terraform Module Design

**Module Security Patterns**:
```
Input Validation:
- Type constraints for all variables
- Validation rules for security parameters
- Default values following security best practices
- Required variables for critical security settings

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  
  validation {
    condition = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
  
  validation {
    condition = split("/", var.vpc_cidr)[1] >= 16
    error_message = "VPC CIDR must have a subnet mask of /16 or larger."
  }
}

variable "enable_encryption" {
  description = "Enable encryption for all resources"
  type        = bool
  default     = true
  
  validation {
    condition = var.enable_encryption == true
    error_message = "Encryption must be enabled for security compliance."
  }
}
```

**Resource Security Hardening**:
```
Compute Security:
- Encrypted EBS volumes with customer-managed keys
- Security groups with minimal required access
- IAM roles with least privilege permissions
- Regular patching and update schedules

resource "aws_instance" "ml_training_node" {
  ami           = data.aws_ami.hardened_ml_ami.id
  instance_type = var.instance_type
  
  vpc_security_group_ids = [aws_security_group.ml_training.id]
  iam_instance_profile   = aws_iam_instance_profile.ml_training.name
  
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    encrypted             = true
    kms_key_id           = aws_kms_key.ml_infrastructure.arn
    delete_on_termination = true
  }
  
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"  # Require IMDSv2
    http_put_response_hop_limit = 1
  }
  
  tags = merge(local.common_tags, {
    Name = "ml-training-${var.environment}-${count.index + 1}"
    Role = "training"
  })
}

Network Security:
- VPC with private subnets for sensitive workloads
- Network ACLs for additional layer of security
- VPC endpoints for private service access
- Flow logs for network monitoring

resource "aws_vpc" "ml_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name = "ml-vpc-${var.environment}"
  })
}

resource "aws_flow_log" "ml_vpc_flow_log" {
  iam_role_arn    = aws_iam_role.flow_log.arn
  log_destination = aws_cloudwatch_log_group.ml_vpc_flow_log.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.ml_vpc.id
}
```

#### 2.3 Terraform Security Scanning

**Static Analysis Tools**:
```
Checkov Integration:
- Security policy scanning for Terraform code
- Custom rules for organization-specific policies
- Integration with CI/CD pipelines
- Reporting and remediation guidance

TFSec Implementation:
- Fast security scanner for Terraform
- Pre-commit hooks for early detection
- Custom check development
- Integration with existing security tools

Terraform Compliance:
- BDD-style security testing
- Human-readable security requirements
- Compliance framework mapping
- Automated reporting and documentation
```

**Policy Validation Framework**:
```
Security Policy Examples:
1. All S3 buckets must have encryption enabled
2. EC2 instances must not have public IP addresses
3. RDS instances must have backup retention >= 7 days
4. Security groups must not allow 0.0.0.0/0 access
5. IAM policies must not contain wildcard permissions

Implementation Strategy:
- Pre-deployment validation in CI/CD
- Policy-as-code using Open Policy Agent
- Automated remediation for common violations
- Regular policy updates and maintenance
```

### 3. Pulumi Security Implementation

#### 3.1 Pulumi Security Architecture

**Language-Native Security**:
Pulumi's use of general-purpose programming languages enables sophisticated security implementations:

**Type Safety and Validation**:
```
Python Example - Secure VPC Creation:
import pulumi
import pulumi_aws as aws
from typing import List, Dict, Any

class SecureVPCConfig:
    def __init__(self, cidr: str, enable_flow_logs: bool = True):
        if not self._is_valid_cidr(cidr):
            raise ValueError(f"Invalid CIDR block: {cidr}")
        
        self.cidr = cidr
        self.enable_flow_logs = enable_flow_logs
        
    @staticmethod
    def _is_valid_cidr(cidr: str) -> bool:
        try:
            import ipaddress
            network = ipaddress.IPv4Network(cidr, strict=False)
            return network.prefixlen >= 16  # Minimum /16 for security
        except ValueError:
            return False

class MLInfrastructure:
    def __init__(self, config: SecureVPCConfig):
        self.config = config
        self.vpc = self._create_vpc()
        self.subnets = self._create_subnets()
        self.security_groups = self._create_security_groups()
        
    def _create_vpc(self) -> aws.ec2.Vpc:
        return aws.ec2.Vpc(
            "ml-vpc",
            cidr_block=self.config.cidr,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            tags={
                "Name": "ml-vpc",
                "Environment": pulumi.get_stack(),
                "ManagedBy": "pulumi"
            }
        )
```

**Secret Management Integration**:
```
Secure Configuration Management:
- Integration with cloud secret managers
- Encryption of sensitive stack outputs
- Dynamic secret retrieval during deployment
- Audit logging for secret access

TypeScript Example - Secret Management:
import * as aws from "@pulumi/aws";
import * as pulumi from "@pulumi/pulumi";

// Retrieve database password from AWS Secrets Manager
const dbPasswordSecret = aws.secretsmanager.getSecretVersion({
    secretId: "ml-database-password",
});

// Create RDS instance with encrypted password
const mlDatabase = new aws.rds.Instance("ml-database", {
    engine: "postgres",
    engineVersion: "13.7",
    instanceClass: "db.r5.large",
    
    // Use secret for password
    password: dbPasswordSecret.then(secret => secret.secretString),
    
    // Security configurations
    storageEncrypted: true,
    kmsKeyId: mlKmsKey.arn,
    backupRetentionPeriod: 30,
    backupWindow: "03:00-04:00",
    
    // Network security
    dbSubnetGroupName: dbSubnetGroup.name,
    vpcSecurityGroupIds: [dbSecurityGroup.id],
    
    // Monitoring and logging
    monitoringInterval: 60,
    monitoringRoleArn: rdsMonitoringRole.arn,
    enabledCloudwatchLogsExports: ["postgresql"],
    
    tags: {
        Environment: pulumi.getStack(),
        Purpose: "ml-metadata",
        SecurityLevel: "high"
    }
});
```

#### 3.2 Pulumi Policy-as-Code

**CrossGuard Policy Framework**:
```
Policy Definition Architecture:
- TypeScript/Python policy definitions
- Resource-level policy enforcement
- Custom validation logic
- Integration with compliance frameworks

Example Security Policy - Encryption Enforcement:
import { PolicyPack, validateResourceOfType } from "@pulumi/policy";
import { Instance } from "@pulumi/aws/ec2";
import { Bucket } from "@pulumi/aws/s3";

new PolicyPack("ml-security-policies", {
    policies: [
        {
            name: "ec2-encryption-required",
            description: "EC2 instances must have encrypted EBS volumes",
            enforcementLevel: "mandatory",
            validateResource: validateResourceOfType(Instance, (instance, args, reportViolation) => {
                const rootBlockDevice = instance.rootBlockDevice;
                if (!rootBlockDevice || !rootBlockDevice.encrypted) {
                    reportViolation("EC2 instance must have encrypted root volume");
                }
                
                const ebsBlockDevices = instance.ebsBlockDevices || [];
                for (const device of ebsBlockDevices) {
                    if (!device.encrypted) {
                        reportViolation(`EBS block device ${device.deviceName} must be encrypted`);
                    }
                }
            }),
        },
        
        {
            name: "s3-encryption-required",
            description: "S3 buckets must have encryption enabled",
            enforcementLevel: "mandatory",
            validateResource: validateResourceOfType(Bucket, (bucket, args, reportViolation) => {
                if (!bucket.serverSideEncryptionConfiguration) {
                    reportViolation("S3 bucket must have server-side encryption enabled");
                }
            }),
        }
    ],
});
```

**Advanced Policy Scenarios**:
```
ML-Specific Security Policies:
1. GPU instances must be in private subnets
2. Model storage buckets must have versioning enabled
3. Training data must be encrypted with customer-managed keys
4. Inference endpoints must have authentication enabled
5. Distributed training clusters must have network isolation

Implementation Example - GPU Security Policy:
{
    name: "gpu-instances-private-subnet",
    description: "GPU instances must be deployed in private subnets",
    enforcementLevel: "mandatory",
    validateResource: validateResourceOfType(Instance, (instance, args, reportViolation) => {
        // Check if instance type contains GPU
        const instanceType = instance.instanceType;
        const gpuInstanceTypes = ["p2", "p3", "p4", "g3", "g4"];
        
        const isGpuInstance = gpuInstanceTypes.some(type => 
            instanceType.startsWith(type)
        );
        
        if (isGpuInstance) {
            // Validate subnet is private
            const subnetId = instance.subnetId;
            // Implementation would check subnet route table
            // for public internet gateway routes
            validatePrivateSubnet(subnetId, reportViolation);
        }
    })
}
```

### 4. Crossplane Security Implementation

#### 4.1 Crossplane Architecture Security

**Control Plane Security**:
Crossplane operates as a Kubernetes operator, extending the API server with custom resources for infrastructure management.

**RBAC and Access Control**:
```
Kubernetes RBAC Integration:
- Role-based access control for infrastructure resources
- Namespace isolation for multi-tenant deployments
- Service account authentication and authorization
- Integration with external identity providers

Example RBAC Configuration:
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-infrastructure-manager
rules:
- apiGroups: ["aws.crossplane.io"]
  resources: ["*"]
  verbs: ["get", "list", "create", "update", "delete"]
- apiGroups: ["gcp.crossplane.io"]
  resources: ["*"]
  verbs: ["get", "list", "create", "update", "delete"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ml-infrastructure-manager-binding
subjects:
- kind: ServiceAccount
  name: ml-infrastructure-sa
  namespace: ml-team
roleRef:
  kind: ClusterRole
  name: ml-infrastructure-manager
  apiGroup: rbac.authorization.k8s.io
```

**Provider Configuration Security**:
```
Secure Cloud Provider Integration:
- Credential management through Kubernetes secrets
- Cross-account role assumption for AWS
- Service account authentication for GCP
- Managed identity integration for Azure

AWS Provider Security Configuration:
apiVersion: aws.crossplane.io/v1beta1
kind: ProviderConfig
metadata:
  name: aws-ml-provider
spec:
  credentials:
    source: Secret
    secretRef:
      namespace: crossplane-system
      name: aws-ml-credentials
      key: creds
  assumeRoleARN: "arn:aws:iam::123456789012:role/CrossplaneMLRole"
  externalID: "unique-external-id"
```

#### 4.2 Secure Infrastructure Composition

**Composite Resource Definitions (XRDs)**:
```
Secure ML Infrastructure Template:
apiVersion: apiextensions.crossplane.io/v1
kind: CompositeResourceDefinition
metadata:
  name: xmlclusters.ml.example.com
spec:
  group: ml.example.com
  names:
    kind: XMLCluster
    plural: xmlclusters
  versions:
  - name: v1alpha1
    served: true
    referenceable: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              parameters:
                type: object
                properties:
                  region:
                    type: string
                    enum: ["us-west-2", "us-east-1", "eu-west-1"]
                  nodeCount:
                    type: integer
                    minimum: 1
                    maximum: 100
                  instanceType:
                    type: string
                    enum: ["p3.2xlarge", "p3.8xlarge", "p4d.24xlarge"]
                  encryptionRequired:
                    type: boolean
                    default: true
                  networkIsolation:
                    type: boolean
                    default: true
                required:
                - region
                - nodeCount
                - instanceType
            required:
            - parameters
```

**Security-Focused Compositions**:
```
Composition with Security Controls:
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: secure-ml-cluster
  labels:
    provider: aws
    service: eks
    security-level: high
spec:
  compositeTypeRef:
    apiVersion: ml.example.com/v1alpha1
    kind: XMLCluster
    
  resources:
  - name: vpc
    base:
      apiVersion: ec2.aws.crossplane.io/v1alpha1
      kind: VPC
      spec:
        forProvider:
          region: us-west-2
          cidrBlock: "10.0.0.0/16"
          enableDnsHostNames: true
          enableDnsSupport: true
          tags:
            Environment: production
            ManagedBy: crossplane
            SecurityLevel: high
    patches:
    - fromFieldPath: spec.parameters.region
      toFieldPath: spec.forProvider.region
      
  - name: private-subnet-1
    base:
      apiVersion: ec2.aws.crossplane.io/v1alpha1
      kind: Subnet
      spec:
        forProvider:
          region: us-west-2
          availabilityZone: us-west-2a
          cidrBlock: "10.0.1.0/24"
          mapPublicIpOnLaunch: false  # Private subnet
          tags:
            Name: ml-private-subnet-1
            kubernetes.io/role/internal-elb: "1"
    patches:
    - fromFieldPath: spec.parameters.region
      toFieldPath: spec.forProvider.region
    - fromFieldPath: spec.parameters.region
      toFieldPath: spec.forProvider.availabilityZone
      transforms:
      - type: string
        string:
          fmt: "%sa"
          
  - name: eks-cluster
    base:
      apiVersion: eks.aws.crossplane.io/v1alpha1
      kind: Cluster
      spec:
        forProvider:
          region: us-west-2
          version: "1.24"
          encryptionConfig:
          - resources: ["secrets"]
            provider:
              keyArn: ""  # Will be patched with KMS key
          logging:
            enable:
            - api
            - audit
            - authenticator
            - controllerManager
            - scheduler
          endpointConfig:
            privateAccess: true
            publicAccess: false  # Private cluster
    patches:
    - fromFieldPath: spec.parameters.region
      toFieldPath: spec.forProvider.region
```

#### 4.3 Crossplane Security Monitoring

**GitOps Integration Security**:
```
Secure GitOps Workflow:
1. Infrastructure changes committed to Git repository
2. Code review and approval process for all changes
3. Automated security scanning of Crossplane manifests
4. Controlled deployment through ArgoCD or Flux
5. Continuous monitoring and drift detection

ArgoCD Application Security:
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-infrastructure
  namespace: argocd
spec:
  project: ml-project
  source:
    repoURL: https://github.com/company/ml-infrastructure
    targetRevision: main
    path: manifests/production
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-infrastructure
  syncPolicy:
    automated:
      prune: false  # Manual approval for deletions
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        maxDuration: 3m0s
        factor: 2
```

**Compliance and Audit Integration**:
```
Policy Engine Integration:
- OPA Gatekeeper for admission control
- Polaris for security best practices
- Falco for runtime security monitoring
- Custom controllers for compliance validation

Example Gatekeeper Policy for Crossplane:
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: crossplaneencryptionrequired
spec:
  crd:
    spec:
      names:
        kind: CrossplaneEncryptionRequired
      validation:
        openAPIV3Schema:
          type: object
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package crossplaneencryptionrequired
        
        violation[{"msg": msg}] {
          input.review.object.kind == "XMLCluster"
          not input.review.object.spec.parameters.encryptionRequired
          msg := "ML clusters must have encryption enabled"
        }
```

### 5. Security Integration Patterns

#### 5.1 CI/CD Pipeline Security

**Secure Pipeline Architecture**:
```
Pipeline Security Stages:
1. Source Code Security
   - Secret scanning in repositories
   - Dependency vulnerability analysis
   - Code quality and security analysis

2. Infrastructure Code Validation
   - Static analysis of IaC templates
   - Policy compliance checking
   - Security best practice validation

3. Deployment Security
   - Secure credential management
   - Approval workflows for production
   - Deployment environment isolation

4. Runtime Security
   - Continuous compliance monitoring
   - Security event detection and response
   - Regular security assessments
```

**GitHub Actions Security Example**:
```yaml
name: Secure Infrastructure Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run TruffleHog Secret Scan
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
    
    - name: Run Checkov IaC Security Scan
      uses: bridgecrewio/checkov-action@master
      with:
        directory: ./terraform
        framework: terraform
        output_format: sarif
        output_file_path: reports/checkov.sarif
    
    - name: Run TFSec Security Scan
      uses: aquasecurity/tfsec-action@v1.0.0
      with:
        working_directory: ./terraform
    
    - name: Upload Security Scan Results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: reports/checkov.sarif

  deploy:
    needs: security-scan
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
        role-session-name: GitHubActions
        aws-region: us-west-2
    
    - name: Terraform Plan
      run: |
        terraform init
        terraform plan -out=tfplan
        
    - name: Security Review of Plan
      run: |
        terraform show -json tfplan | checkov -f -
    
    - name: Terraform Apply
      if: github.event_name == 'push'
      run: terraform apply tfplan
```

#### 5.2 Multi-Cloud Security Patterns

**Cross-Cloud Security Governance**:
```
Unified Security Framework:
- Consistent security policies across cloud providers
- Centralized identity and access management
- Cross-cloud network security and encryption
- Unified monitoring and compliance reporting

Implementation Strategy:
1. Abstract security requirements into cloud-agnostic policies
2. Implement provider-specific security controls
3. Maintain security baseline across all environments
4. Continuous monitoring and compliance validation

Example Multi-Cloud Security Module:
# security-baseline/main.tf
locals {
  security_tags = {
    SecurityLevel    = var.security_level
    DataClass       = var.data_classification
    Environment     = var.environment
    ComplianceReq   = var.compliance_requirements
    ManagedBy       = "terraform"
  }
}

module "aws_security" {
  source = "./aws-security"
  count  = var.cloud_provider == "aws" ? 1 : 0
  
  security_level = var.security_level
  vpc_cidr      = var.vpc_cidr
  common_tags   = local.security_tags
}

module "azure_security" {
  source = "./azure-security"
  count  = var.cloud_provider == "azure" ? 1 : 0
  
  security_level = var.security_level
  vnet_cidr     = var.vpc_cidr
  common_tags   = local.security_tags
}

module "gcp_security" {
  source = "./gcp-security"
  count  = var.cloud_provider == "gcp" ? 1 : 0
  
  security_level = var.security_level
  vpc_cidr      = var.vpc_cidr
  labels        = local.security_tags
}
```

### 6. Advanced Security Patterns

#### 6.1 Immutable Infrastructure Security

**Immutable Infrastructure Principles**:
```
Core Concepts:
- Infrastructure components are never modified after deployment
- Changes require complete replacement of components
- Version-controlled infrastructure definitions
- Automated deployment and rollback processes

Security Benefits:
- Eliminates configuration drift
- Reduces attack surface through minimal runtime changes
- Improves audit trail and compliance
- Enables rapid response to security incidents

Implementation with IaC:
- Blue-green deployment patterns
- Canary releases for infrastructure changes
- Automated rollback on security policy violations
- Immutable AMIs/container images with security hardening
```

**Container Security for ML Workloads**:
```
Secure Container Build Process:
1. Start with minimal base images (distroless, alpine)
2. Apply security patches during build
3. Remove unnecessary packages and tools
4. Run security scans on final images
5. Sign images with trusted keys

Dockerfile Security Best Practices:
FROM gcr.io/distroless/python3-debian11

# Create non-root user
USER 10001:10001

# Copy application code
COPY --chown=10001:10001 app/ /app/
COPY --chown=10001:10001 requirements.txt /app/

# Set working directory
WORKDIR /app

# Install dependencies (in build stage)
RUN pip install --no-cache-dir -r requirements.txt

# Security metadata
LABEL security.scan.date="2024-01-15"
LABEL security.level="high"
LABEL compliance.framework="SOC2"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python health_check.py

# Expose only necessary port
EXPOSE 8080

# Run application
ENTRYPOINT ["python", "main.py"]
```

#### 6.2 Zero Trust Infrastructure

**Zero Trust Implementation with IaC**:
```
Zero Trust Principles in Infrastructure:
- Never trust, always verify
- Least privilege access
- Assume breach and limit blast radius
- Encrypt everything

IaC Implementation:
1. Network micro-segmentation
2. Identity-based access control
3. Continuous monitoring and validation
4. Encrypted communication channels

Network Micro-segmentation Example:
resource "aws_security_group" "ml_training_internal" {
  name_prefix = "ml-training-internal-"
  vpc_id      = aws_vpc.ml_vpc.id

  # Allow only specific ML training protocols
  ingress {
    description = "NCCL communication"
    from_port   = 8000
    to_port     = 8100
    protocol    = "tcp"
    self        = true
  }
  
  ingress {
    description = "SSH from bastion"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }
  
  # Explicit egress rules (default deny)
  egress {
    description = "HTTPS to model repository"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.model_repo_cidr]
  }
  
  egress {
    description = "NFS to shared storage"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    security_groups = [aws_security_group.shared_storage.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "ml-training-internal"
    Purpose = "zero-trust-segmentation"
  })
}
```

---

## 🔍 Key Questions

### Beginner Level

1. **Q**: What are the main security advantages of using Infrastructure-as-Code compared to manual infrastructure management?
   **A**: IaC provides version control for security configurations, consistent security baselines, automated compliance checking, rapid deployment of security patches, and immutable infrastructure principles that reduce configuration drift.

2. **Q**: Why is it important to encrypt Terraform state files, and what sensitive information might they contain?
   **A**: Terraform state files contain resource metadata, configuration details, and potentially sensitive values like database passwords, private keys, and internal network configurations. Encryption protects this information from unauthorized access.

3. **Q**: What is the principle of "shift-left" security in IaC, and how does it improve security posture?
   **A**: Shift-left security moves security testing and validation earlier in the development process, catching issues during development rather than production, reducing costs and risk of security incidents.

### Intermediate Level

4. **Q**: Design a secure Terraform module for deploying a GPU cluster for AI training that follows security best practices.
   **A**: The module should include:
   ```
   - Input validation for all variables
   - Encrypted EBS volumes with customer-managed KMS keys
   - Private subnets with no public IP addresses
   - Security groups with minimal required access
   - IAM roles with least privilege permissions
   - VPC Flow Logs for network monitoring
   - CloudTrail logging for API auditing
   - Resource tagging for governance
   - Output sanitization to avoid exposing sensitive data
   ```

5. **Q**: How would you implement secret management in a Pulumi project for a multi-cloud AI/ML infrastructure?
   **A**: Use cloud-native secret services (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager), implement encryption at rest and in transit, use IAM/RBAC for access control, rotate secrets regularly, and integrate with CI/CD pipelines securely.

6. **Q**: Explain how Crossplane's RBAC model can be used to implement multi-tenant security for different AI research teams.
   **A**: Create namespace isolation per team, implement role-based access with team-specific permissions, use CompositeResourceDefinitions with security constraints, and integrate with external identity providers for authentication.

### Advanced Level

7. **Q**: Design a comprehensive security scanning and policy enforcement pipeline for IaC that covers development, CI/CD, and runtime phases.
   **A**: 
   ```
   Development Phase:
   - Pre-commit hooks with TFSec/Checkov
   - IDE integration with security linting
   - Local policy validation with OPA
   
   CI/CD Phase:
   - Secret scanning with TruffleHog
   - Static analysis with multiple tools
   - Policy validation with Sentinel/OPA
   - Approval workflows for high-risk changes
   
   Runtime Phase:
   - Continuous compliance monitoring
   - Drift detection and remediation
   - Security event alerting
   - Regular penetration testing
   ```

8. **Q**: How would you implement a zero-trust network architecture for an AI training cluster using IaC, ensuring both security and performance?
   **A**: Implement network micro-segmentation with security groups, use encrypted communication channels, implement identity-based access control, deploy network monitoring and anomaly detection, and optimize for AI workload communication patterns while maintaining security boundaries.

### Tricky Questions

9. **Q**: You discover that a Terraform state file in S3 was accidentally exposed publicly for 24 hours before being secured. The state contained configuration for a production ML training cluster. What security steps should you take, and how would you prevent this in the future?
   **A**: 
   ```
   Immediate Response:
   - Secure the state file and audit access logs
   - Rotate all secrets and credentials referenced in state
   - Review CloudTrail logs for unauthorized access
   - Assess potential data exposure and notify stakeholders
   - Update security groups and access controls
   
   Prevention Measures:
   - Implement bucket policies preventing public access
   - Enable S3 bucket notifications for configuration changes
   - Use least privilege IAM policies for state access
   - Implement state encryption with customer-managed keys
   - Add monitoring and alerting for state file access
   - Regular security audits of state storage configuration
   ```

10. **Q**: Design an IaC architecture that supports both on-premises and multi-cloud deployments for a federated learning system, ensuring consistent security policies across all environments while maintaining performance requirements for distributed AI training.
    **A**: 
    ```
    Architecture Components:
    - Abstract security baseline module with provider-specific implementations
    - Centralized policy management using OPA/CrossGuard
    - Encrypted overlay networks for secure communication
    - Identity federation across environments
    - Consistent monitoring and logging framework
    
    Implementation Strategy:
    - Use composition pattern for reusable security modules
    - Implement GitOps for consistent deployment processes
    - Create environment-specific variable files
    - Use remote state management with cross-environment access
    - Implement automated compliance validation
    - Deploy centralized security monitoring and SIEM
    ```

---

## 🛡️ Security Deep Dive

### IaC-Specific Attack Vectors

#### State File Compromise

**Attack Scenarios**:
```
State File Attacks:
- Unauthorized access to remote state storage
- State file corruption or manipulation
- Secret extraction from state metadata
- Infrastructure reconnaissance through state analysis

Mitigation Strategies:
- Strong encryption for state files
- Access logging and monitoring
- Regular state file audits
- Backup and recovery procedures
```

#### Supply Chain Attacks

**Module and Provider Security**:
```
Risks:
- Malicious Terraform providers
- Compromised community modules
- Dependency confusion attacks
- Backdoors in infrastructure code

Prevention:
- Provider signature verification
- Module source code review
- Dependency pinning and scanning
- Private module registries
```

### Compliance and Governance

#### Regulatory Compliance

**Framework Implementation**:
```
SOC 2 Compliance:
- Automated security controls
- Audit trail for all changes
- Data protection measures
- Access control documentation

GDPR Compliance:
- Data residency controls
- Encryption requirements
- Access logging and monitoring
- Right to erasure implementation

HIPAA Compliance:
- PHI protection measures
- Audit logging requirements
- Encryption at rest and in transit
- Access control and authentication
```

---

## 🚀 Performance Optimization

### IaC Performance Best Practices

#### Terraform Optimization

**State Management Performance**:
```bash
# Terraform performance tuning
export TF_PLUGIN_CACHE_DIR="$HOME/.terraform.d/plugin-cache"
export TF_CLI_ARGS_plan="-parallelism=10"
export TF_CLI_ARGS_apply="-parallelism=10"

# Use targeted operations for large infrastructures
terraform plan -target=module.network
terraform apply -target=module.network
```

#### Pulumi Optimization

**Resource Optimization**:
```typescript
// Pulumi performance optimization
import * as pulumi from "@pulumi/pulumi";

// Use explicit resource dependencies
const vpc = new aws.ec2.Vpc("ml-vpc", {
    cidrBlock: "10.0.0.0/16",
});

const subnet = new aws.ec2.Subnet("ml-subnet", {
    vpcId: vpc.id,  // Explicit dependency
    cidrBlock: "10.0.1.0/24",
}, { dependsOn: [vpc] });

// Parallel resource creation where possible
const securityGroups = ["web", "app", "db"].map(tier => 
    new aws.ec2.SecurityGroup(`${tier}-sg`, {
        vpcId: vpc.id,
        // ... configuration
    })
);
```

---

## 📝 Practical Exercises

### Exercise 1: Secure Terraform Module Development
Create a comprehensive Terraform module for deploying a secure AI training cluster that includes:
- Multi-AZ deployment with private subnets
- Encrypted storage with customer-managed keys
- Least privilege IAM roles and policies
- Network security groups and NACLs
- Monitoring and logging integration
- Input validation and output sanitization

### Exercise 2: Pulumi Policy Implementation
Develop a Pulumi CrossGuard policy pack that enforces:
- Encryption requirements for all storage resources
- Network isolation for AI workloads
- Compliance with industry security standards
- Cost optimization policies
- Resource tagging and governance rules

### Exercise 3: Crossplane Security Architecture
Design a complete Crossplane-based infrastructure platform that provides:
- Multi-tenant isolation for different AI teams
- Self-service infrastructure provisioning
- Automated security policy enforcement
- GitOps-based change management
- Compliance monitoring and reporting

### Exercise 4: IaC Security Pipeline
Build an end-to-end security pipeline that includes:
- Pre-commit security validation
- CI/CD integration with multiple scanning tools
- Policy-as-code enforcement
- Automated remediation for common issues
- Security metrics and reporting dashboard

---

## 🔗 Next Steps
In the next section (day02_002), we'll explore Policy-as-Code implementation with OPA (Open Policy Agent), Sentinel, and Gatekeeper, focusing on how to define, implement, and enforce security policies across AI/ML infrastructure using code-based approaches.