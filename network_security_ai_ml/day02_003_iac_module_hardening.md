# Day 2.3: IaC Module Hardening and Least Privilege IAM

## 🎯 Learning Objectives
By the end of this section, you will understand:
- Infrastructure module hardening principles and techniques
- Least privilege IAM implementation strategies
- Security best practices for IaC module development
- Access control patterns for AI/ML infrastructure
- Advanced IAM scenarios and policy optimization
- Automated security testing and validation

---

## 📚 Theoretical Foundation

### 1. Introduction to IaC Module Hardening

#### 1.1 Module Security Fundamentals

**Security-First Module Design**:
Infrastructure-as-Code modules serve as the building blocks for secure, scalable infrastructure. In AI/ML environments, these modules must address unique security challenges while maintaining performance and usability requirements.

**Core Hardening Principles**:
```
Defense in Depth:
- Multiple security layers at different levels
- Overlapping security controls for redundancy
- Fail-safe defaults with explicit security configurations
- Comprehensive logging and monitoring integration

Least Privilege Access:
- Minimal necessary permissions for all components
- Time-limited access tokens and credentials
- Role-based access control with fine-grained permissions
- Regular access reviews and automated cleanup

Secure by Default:
- Security configurations enabled by default
- Encryption at rest and in transit
- Network isolation and micro-segmentation
- Hardened base images and configurations
```

**AI/ML Specific Security Considerations**:
```
Data Protection Requirements:
- Sensitive training data protection
- Model intellectual property security
- Compliance with data protection regulations
- Cross-border data transfer restrictions

Compute Security:
- GPU resource isolation and access control
- Container and virtualization security
- Distributed training network security
- Edge device management and security

Operational Security:
- Model versioning and lifecycle management
- Experiment tracking and audit trails
- Automated security scanning and validation
- Incident response and recovery procedures
```

#### 1.2 Module Architecture Patterns

**Layered Security Architecture**:
```
Infrastructure Layer:
- Network security groups and NACLs
- VPC isolation and private subnets
- Encrypted storage and databases
- Load balancers with SSL termination

Platform Layer:
- Container runtime security
- Kubernetes RBAC and network policies
- Service mesh security controls
- Certificate management and rotation

Application Layer:
- Authentication and authorization
- API security and rate limiting
- Input validation and sanitization
- Output filtering and data loss prevention
```

**Modular Security Components**:
```
Security Module Structure:
modules/
├── security-baseline/
│   ├── network-security/
│   ├── identity-access/
│   ├── encryption-keys/
│   └── monitoring-logging/
├── ml-workloads/
│   ├── training-cluster/
│   ├── inference-service/
│   ├── data-pipeline/
│   └── model-registry/
└── compliance/
    ├── audit-logging/
    ├── data-governance/
    ├── access-reviews/
    └── vulnerability-scanning/
```

### 2. Least Privilege IAM Implementation

#### 2.1 IAM Fundamentals for AI/ML

**Identity and Access Management Architecture**:
```
IAM Components:
- Principals: Users, roles, service accounts
- Resources: Infrastructure components and data
- Actions: Operations that can be performed
- Conditions: Context-based access controls
- Policies: Rules governing access decisions

AI/ML IAM Considerations:
- Multi-tenant access with strict isolation
- Dynamic resource scaling requiring flexible permissions
- Cross-service communication in ML pipelines
- Long-running training jobs with persistent access
- Edge device authentication and authorization
```

**Principle of Least Privilege Application**:
```
Access Control Strategy:
1. Identify minimum required permissions
2. Grant only necessary access
3. Implement time-limited access where possible
4. Regular review and cleanup of unused permissions
5. Automated detection of privilege escalation

Example Permission Minimization:
Traditional Approach: S3:* (full S3 access)
Least Privilege Approach:
- s3:GetObject for specific training data buckets
- s3:PutObject for specific model output buckets
- s3:ListBucket with prefix restrictions
- Time-limited access tokens
```

#### 2.2 AWS IAM Security Patterns

**Role-Based Access Control for ML Workloads**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MLTrainingDataAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ml-training-data-${aws:userid}/*",
        "arn:aws:s3:::ml-training-data-${aws:userid}"
      ],
      "Condition": {
        "StringEquals": {
          "s3:ExistingObjectTag/DataClassification": ["public", "internal"]
        },
        "IpAddress": {
          "aws:SourceIp": ["10.0.0.0/8"]
        },
        "DateGreaterThan": {
          "aws:CurrentTime": "2024-01-01T00:00:00Z"
        },
        "DateLessThan": {
          "aws:CurrentTime": "2024-12-31T23:59:59Z"
        }
      }
    },
    {
      "Sid": "MLModelOutputAccess",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl"
      ],
      "Resource": [
        "arn:aws:s3:::ml-model-outputs-${aws:userid}/*"
      ],
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms",
          "s3:x-amz-server-side-encryption-aws-kms-key-id": "arn:aws:kms:us-west-2:123456789012:key/ml-model-key"
        }
      }
    }
  ]
}
```

**Cross-Account Access Patterns**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssumeMLTrainingRole",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::TRAINING-ACCOUNT:role/MLTrainingRole"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "${random_external_id}",
          "aws:RequestedRegion": ["us-west-2", "us-east-1"]
        },
        "StringLike": {
          "aws:userid": "*:ml-user-*"
        },
        "IpAddress": {
          "aws:SourceIp": ["203.0.113.0/24"]
        }
      }
    }
  ]
}
```

**Service-Linked Roles for ML Services**:
```terraform
# SageMaker execution role with least privilege
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "ml-sagemaker-execution-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
  
  tags = merge(local.common_tags, {
    Purpose = "ml-training"
    AccessLevel = "service-linked"
  })
}

# Minimal SageMaker permissions
resource "aws_iam_role_policy" "sagemaker_minimal_policy" {
  name = "ml-sagemaker-minimal-${var.environment}"
  role = aws_iam_role.sagemaker_execution_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "CloudWatchLogsAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
        ]
      },
      {
        Sid = "ECRImageAccess"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = [
          "arn:aws:ecr:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:repository/ml-*"
        ]
      }
    ]
  })
}
```

#### 2.3 Azure RBAC and Identity Management

**Azure Role-Based Access Control**:
```terraform
# Custom role for ML engineers
resource "azurerm_role_definition" "ml_engineer" {
  name        = "ML Engineer - ${var.environment}"
  scope       = data.azurerm_subscription.current.id
  description = "Custom role for ML engineers with minimal required permissions"
  
  permissions {
    actions = [
      # Machine Learning workspace access
      "Microsoft.MachineLearningServices/workspaces/read",
      "Microsoft.MachineLearningServices/workspaces/experiments/*",
      "Microsoft.MachineLearningServices/workspaces/models/read",
      "Microsoft.MachineLearningServices/workspaces/datastores/read",
      
      # Compute access for training
      "Microsoft.Compute/virtualMachines/read",
      "Microsoft.Compute/virtualMachines/start/action",
      "Microsoft.Compute/virtualMachines/restart/action",
      
      # Storage access for datasets
      "Microsoft.Storage/storageAccounts/blobServices/containers/read",
      "Microsoft.Storage/storageAccounts/blobServices/generateUserDelegationKey/action"
    ]
    
    not_actions = [
      # Prevent deletion of critical resources
      "Microsoft.MachineLearningServices/workspaces/delete",
      "Microsoft.Storage/storageAccounts/delete",
      "Microsoft.Compute/virtualMachines/delete"
    ]
    
    data_actions = [
      "Microsoft.Storage/storageAccounts/blobServices/containers/blobs/read",
      "Microsoft.Storage/storageAccounts/blobServices/containers/blobs/write"
    ]
    
    not_data_actions = [
      "Microsoft.Storage/storageAccounts/blobServices/containers/blobs/delete"
    ]
  }
  
  assignable_scopes = [
    data.azurerm_subscription.current.id
  ]
}

# Managed identity for ML workloads
resource "azurerm_user_assigned_identity" "ml_workload" {
  name                = "ml-workload-identity-${var.environment}"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = azurerm_resource_group.ml_rg.location
  
  tags = merge(var.common_tags, {
    Purpose = "ml-workload-authentication"
  })
}

# Role assignment with conditions
resource "azurerm_role_assignment" "ml_engineer_assignment" {
  scope              = azurerm_storage_account.ml_storage.id
  role_definition_id = azurerm_role_definition.ml_engineer.role_definition_resource_id
  principal_id       = azurerm_user_assigned_identity.ml_workload.principal_id
  
  condition = "((!(ActionMatches{'Microsoft.Storage/storageAccounts/blobServices/containers/blobs/delete'})) || (@Resource[Microsoft.Storage/storageAccounts/blobServices/containers:name] StringEquals 'temp-data'))"
  condition_version = "2.0"
}
```

#### 2.4 Google Cloud IAM Patterns

**Service Account Management**:
```terraform
# Workload Identity for GKE
resource "google_service_account" "ml_training" {
  account_id   = "ml-training-${var.environment}"
  display_name = "ML Training Service Account"
  description  = "Service account for ML training workloads with minimal permissions"
  project      = var.project_id
}

# Custom IAM role for ML operations
resource "google_project_iam_custom_role" "ml_operator" {
  role_id     = "mlOperator"
  title       = "ML Operator"
  description = "Custom role for ML operations with least privilege"
  project     = var.project_id
  
  permissions = [
    # AI Platform permissions
    "ml.models.predict",
    "ml.models.get",
    "ml.versions.get",
    "ml.versions.predict",
    
    # Storage permissions
    "storage.objects.get",
    "storage.objects.create",
    "storage.buckets.get",
    
    # Logging permissions
    "logging.logEntries.create",
    
    # Monitoring permissions
    "monitoring.timeSeries.create"
  ]
}

# IAM binding with conditions
resource "google_project_iam_member" "ml_training_binding" {
  project = var.project_id
  role    = google_project_iam_custom_role.ml_operator.name
  member  = "serviceAccount:${google_service_account.ml_training.email}"
  
  condition {
    title       = "ML Training Time Restriction"
    description = "Access only during business hours"
    expression  = "request.time.getHours() >= 8 && request.time.getHours() <= 18"
  }
}

# Workload Identity binding
resource "google_service_account_iam_binding" "ml_workload_identity" {
  service_account_id = google_service_account.ml_training.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[ml-training/ml-training-ksa]"
  ]
}
```

### 3. Advanced IAM Security Patterns

#### 3.1 Dynamic and Conditional Access

**Time-Based Access Controls**:
```terraform
# AWS IAM policy with time conditions
data "aws_iam_policy_document" "time_restricted_access" {
  statement {
    sid    = "BusinessHoursAccess"
    effect = "Allow"
    
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:CreateHyperParameterTuningJob"
    ]
    
    resources = ["*"]
    
    condition {
      test     = "DateGreaterThan"
      variable = "aws:RequestedRegion"
      values = [
        "us-west-2",
        "us-east-1"
      ]
    }
    
    condition {
      test     = "IpAddress"
      variable = "aws:SourceIp"
      values   = var.allowed_ip_ranges
    }
    
    # Time-based restrictions
    condition {
      test     = "DateGreaterThanEquals"
      variable = "aws:RequestedTime"
      values   = ["08:00Z"]
    }
    
    condition {
      test     = "DateLessThanEquals"
      variable = "aws:RequestedTime"  
      values   = ["18:00Z"]
    }
  }
  
  statement {
    sid    = "EmergencyAccess"
    effect = "Allow"
    
    actions = [
      "sagemaker:StopTrainingJob",
      "sagemaker:DeleteEndpoint"
    ]
    
    resources = ["*"]
    
    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/EmergencyAccess"
      values   = ["true"]
    }
  }
}
```

**Resource-Based Access Patterns**:
```terraform
# S3 bucket policy with data classification
resource "aws_s3_bucket_policy" "ml_data_policy" {
  bucket = aws_s3_bucket.ml_data.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DataClassificationAccess"
        Effect    = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = [
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.ml_data.arn}/*"
        Condition = {
          StringEquals = {
            "s3:ExistingObjectTag/DataClassification" = [
              "public",
              "internal"
            ]
            "aws:PrincipalTag/SecurityClearance" = [
              "internal",
              "confidential"
            ]
          }
          StringLike = {
            "aws:userid" = "*:${var.allowed_user_pattern}"
          }
        }
      },
      {
        Sid       = "ConfidentialDataAccess"
        Effect    = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = [
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.ml_data.arn}/confidential/*"
        Condition = {
          StringEquals = {
            "aws:PrincipalTag/SecurityClearance" = "confidential"
            "aws:MultiFactorAuthPresent" = "true"
          }
          NumericLessThan = {
            "aws:MultiFactorAuthAge" = "3600"
          }
        }
      }
    ]
  })
}
```

#### 3.2 Cross-Service Access Patterns

**ML Pipeline Service Communication**:
```terraform
# Data processing Lambda function role
resource "aws_iam_role" "data_processor" {
  name = "ml-data-processor-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Condition = {
          StringEquals = {
            "lambda:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

# Minimal permissions for data processing
resource "aws_iam_role_policy" "data_processor_policy" {
  name = "data-processor-policy"
  role = aws_iam_role.data_processor.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3DataAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.raw_data.arn}/incoming/*",
          "${aws_s3_bucket.processed_data.arn}/processed/*"
        ]
        Condition = {
          StringEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      },
      {
        Sid    = "SQSMessageProcessing"
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.data_processing.arn
      },
      {
        Sid    = "StepFunctionExecution"
        Effect = "Allow"
        Action = [
          "states:SendTaskSuccess",
          "states:SendTaskFailure",
          "states:SendTaskHeartbeat"
        ]
        Resource = aws_sfn_state_machine.ml_pipeline.arn
      }
    ]
  })
}

# Training job execution role
resource "aws_iam_role" "training_execution" {
  name = "ml-training-execution-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          StringLike = {
            "aws:RequestTag/Purpose" = "ml-training"
          }
        }
      }
    ]
  })
}
```

#### 3.3 Identity Federation Patterns

**OIDC Integration for Kubernetes**:
```terraform
# OIDC provider for EKS
data "tls_certificate" "eks_oidc" {
  url = aws_eks_cluster.ml_cluster.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks_oidc" {
  url = aws_eks_cluster.ml_cluster.identity[0].oidc[0].issuer
  
  client_id_list = [
    "sts.amazonaws.com"
  ]
  
  thumbprint_list = [
    data.tls_certificate.eks_oidc.certificates[0].sha1_fingerprint
  ]
  
  tags = merge(local.common_tags, {
    Purpose = "eks-workload-identity"
  })
}

# Role for Kubernetes service accounts
resource "aws_iam_role" "k8s_service_account" {
  name = "ml-k8s-service-account-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.eks_oidc.arn
        }
        Condition = {
          StringEquals = {
            "${replace(aws_iam_openid_connect_provider.eks_oidc.url, "https://", "")}:sub" = "system:serviceaccount:ml-training:ml-training-sa"
            "${replace(aws_iam_openid_connect_provider.eks_oidc.url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}
```

### 4. Module Hardening Implementation

#### 4.1 Secure Module Design Patterns

**Input Validation and Sanitization**:
```terraform
# Variable validation with security constraints
variable "instance_types" {
  description = "Allowed EC2 instance types for ML workloads"
  type        = list(string)
  
  validation {
    condition = alltrue([
      for instance_type in var.instance_types :
      can(regex("^(t3|m5|c5|r5|p3|p4|g4)\\.", instance_type))
    ])
    error_message = "Only approved instance families are allowed for ML workloads."
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  
  validation {
    condition = can(cidrhost(var.vpc_cidr, 0)) && (
      can(regex("^10\\.", var.vpc_cidr)) ||
      can(regex("^172\\.(1[6-9]|2[0-9]|3[0-1])\\.", var.vpc_cidr)) ||
      can(regex("^192\\.168\\.", var.vpc_cidr))
    )
    error_message = "VPC CIDR must be a valid private IPv4 CIDR block."
  }
  
  validation {
    condition     = parseint(split("/", var.vpc_cidr)[1], 10) >= 16
    error_message = "VPC CIDR must have a netmask of /16 or smaller for security."
  }
}

variable "environment" {
  description = "Environment name"
  type        = string
  
  validation {
    condition = contains([
      "development",
      "staging", 
      "production"
    ], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}
```

**Secure Resource Configuration**:
```terraform
# Security group with minimal access
resource "aws_security_group" "ml_training" {
  name_prefix = "ml-training-${var.environment}-"
  vpc_id      = aws_vpc.ml_vpc.id
  description = "Security group for ML training instances"
  
  # Minimal ingress rules
  dynamic "ingress" {
    for_each = var.allowed_ssh_cidrs
    content {
      description = "SSH access from ${ingress.value}"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }
  
  # ML-specific communication
  ingress {
    description     = "NCCL communication between training nodes"
    from_port       = 8000
    to_port         = 8100
    protocol        = "tcp"
    self            = true
  }
  
  # Explicit egress rules (deny all by default)
  egress {
    description = "HTTPS for package downloads"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    description = "S3 access via VPC endpoint"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    prefix_list_ids = [aws_vpc_endpoint.s3.prefix_list_id]
  }
  
  tags = merge(local.security_tags, {
    Name = "ml-training-sg-${var.environment}"
  })
  
  lifecycle {
    create_before_destroy = true
  }
}

# Encrypted EBS volumes
resource "aws_launch_template" "ml_training" {
  name_prefix   = "ml-training-${var.environment}-"
  image_id      = data.aws_ami.ml_optimized.id
  instance_type = var.instance_type
  
  vpc_security_group_ids = [aws_security_group.ml_training.id]
  
  # User data with security hardening
  user_data = base64encode(templatefile("${path.module}/scripts/hardening.sh", {
    environment = var.environment
    log_group   = aws_cloudwatch_log_group.ml_training.name
  }))
  
  # Encrypted root volume
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_type           = "gp3"
      volume_size           = var.root_volume_size
      encrypted             = true
      kms_key_id           = aws_kms_key.ml_infrastructure.arn
      delete_on_termination = true
    }
  }
  
  # Additional encrypted data volume
  block_device_mappings {
    device_name = "/dev/xvdf"
    ebs {
      volume_type           = "gp3"
      volume_size           = var.data_volume_size
      encrypted             = true
      kms_key_id           = aws_kms_key.ml_infrastructure.arn
      delete_on_termination = true
    }
  }
  
  # Instance metadata configuration
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"  # IMDSv2 required
    http_put_response_hop_limit = 1
    http_protocol_ipv6          = "disabled"
    instance_metadata_tags      = "enabled"
  }
  
  monitoring {
    enabled = true
  }
  
  tag_specifications {
    resource_type = "instance"
    tags = merge(local.security_tags, {
      Name = "ml-training-${var.environment}"
      Role = "training"
    })
  }
  
  tags = local.security_tags
}
```

#### 4.2 Network Hardening

**VPC Security Configuration**:
```terraform
# VPC with security-focused configuration
resource "aws_vpc" "ml_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Disable default security group and NACL
  tags = merge(local.common_tags, {
    Name = "ml-vpc-${var.environment}"
  })
}

# Custom default security group (deny all)
resource "aws_default_security_group" "default" {
  vpc_id = aws_vpc.ml_vpc.id
  
  # No ingress or egress rules (deny all by default)
  tags = merge(local.common_tags, {
    Name = "default-deny-all"
  })
}

# Private subnets for ML workloads
resource "aws_subnet" "ml_private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = var.availability_zones[count.index]
  
  # Disable public IP assignment
  map_public_ip_on_launch = false
  
  tags = merge(local.common_tags, {
    Name = "ml-private-subnet-${var.availability_zones[count.index]}"
    Type = "private"
    Tier = "ml-workloads"
  })
}

# Network ACLs for additional security
resource "aws_network_acl" "ml_private" {
  vpc_id     = aws_vpc.ml_vpc.id
  subnet_ids = aws_subnet.ml_private[*].id
  
  # Allow intra-VPC communication
  ingress {
    rule_no    = 100
    protocol   = "-1"
    action     = "allow"
    cidr_block = var.vpc_cidr
  }
  
  # Allow HTTPS egress
  egress {
    rule_no    = 100
    protocol   = "tcp"
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 443
    to_port    = 443
  }
  
  # Allow NTP
  egress {
    rule_no    = 110
    protocol   = "udp"
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 123
    to_port    = 123
  }
  
  # Allow return traffic
  egress {
    rule_no    = 32766
    protocol   = "tcp"
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 1024
    to_port    = 65535
  }
  
  tags = merge(local.common_tags, {
    Name = "ml-private-nacl"
  })
}

# VPC endpoints for secure AWS service access
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.ml_vpc.id
  service_name = "com.amazonaws.${data.aws_region.current.name}.s3"
  
  tags = merge(local.common_tags, {
    Name = "s3-vpc-endpoint"
  })
}

resource "aws_vpc_endpoint" "ec2" {
  vpc_id              = aws_vpc.ml_vpc.id
  service_name        = "com.amazonaws.${data.aws_region.current.name}.ec2"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.ml_private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
  
  tags = merge(local.common_tags, {
    Name = "ec2-vpc-endpoint"
  })
}
```

#### 4.3 Data Protection Hardening

**Encryption Key Management**:
```terraform
# Customer-managed KMS key for ML infrastructure
resource "aws_kms_key" "ml_infrastructure" {
  description         = "KMS key for ML infrastructure encryption"
  key_usage          = "ENCRYPT_DECRYPT"
  key_spec           = "SYMMETRIC_DEFAULT"
  enable_key_rotation = true
  
  # Deletion protection
  deletion_window_in_days = 30
  
  tags = merge(local.common_tags, {
    Name = "ml-infrastructure-key"
    Purpose = "encryption"
  })
}

resource "aws_kms_alias" "ml_infrastructure" {
  name          = "alias/ml-infrastructure-${var.environment}"
  target_key_id = aws_kms_key.ml_infrastructure.key_id
}

# Key policy with least privilege access
resource "aws_kms_key_policy" "ml_infrastructure" {
  key_id = aws_kms_key.ml_infrastructure.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "ML Service Access"
        Effect = "Allow"
        Principal = {
          AWS = [
            aws_iam_role.ml_training.arn,
            aws_iam_role.ml_inference.arn
          ]
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = [
              "s3.${data.aws_region.current.name}.amazonaws.com",
              "ebs.${data.aws_region.current.name}.amazonaws.com"
            ]
          }
        }
      }
    ]
  })
}

# Encrypted S3 bucket for ML data
resource "aws_s3_bucket" "ml_data" {
  bucket = "ml-data-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "ml-data-bucket"
    DataClassification = "internal"
  })
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.ml_infrastructure.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}
```

### 5. Automated Security Testing

#### 5.1 Infrastructure Security Scanning

**Terraform Security Testing Framework**:
```terraform
# Test configuration for security validation
# tests/security_test.tf

terraform {
  required_providers {
    test = {
      source = "terraform.io/builtin/test"
    }
  }
}

module "ml_infrastructure" {
  source = "../"
  
  environment          = "test"
  vpc_cidr            = "10.100.0.0/16"
  availability_zones  = ["us-west-2a", "us-west-2b"]
  instance_type       = "p3.2xlarge"
  allowed_ssh_cidrs   = ["10.100.0.0/16"]
}

# Test security group configuration
resource "test_assertions" "security_group_hardening" {
  component = module.ml_infrastructure.training_security_group
  
  equal "vpc_id" {
    description = "Security group must be in the correct VPC"
    got         = module.ml_infrastructure.training_security_group.vpc_id
    want        = module.ml_infrastructure.vpc.id
  }
  
  check "ingress_restricted" {
    description = "Security group should not allow unrestricted ingress"
    condition = length([
      for rule in module.ml_infrastructure.training_security_group.ingress :
      rule if contains(rule.cidr_blocks, "0.0.0.0/0")
    ]) == 0
  }
  
  check "ssh_restricted" {
    description = "SSH access should be restricted to allowed CIDRs"
    condition = alltrue([
      for rule in module.ml_infrastructure.training_security_group.ingress :
      rule.from_port != 22 || !contains(rule.cidr_blocks, "0.0.0.0/0")
    ])
  }
}

# Test encryption configuration
resource "test_assertions" "encryption_validation" {
  component = module.ml_infrastructure.ml_data_bucket
  
  check "encryption_enabled" {
    description = "S3 bucket must have encryption enabled"
    condition   = module.ml_infrastructure.bucket_encryption != null
  }
  
  check "kms_encryption" {
    description = "S3 bucket must use KMS encryption"
    condition = can(
      module.ml_infrastructure.bucket_encryption.rule[0].apply_server_side_encryption_by_default[0].sse_algorithm == "aws:kms"
    )
  }
}

# Test IAM role permissions
resource "test_assertions" "iam_least_privilege" {
  component = module.ml_infrastructure.training_role
  
  check "no_wildcard_actions" {
    description = "IAM policies should not contain wildcard actions"
    condition = !can(regex("\\*", jsonencode(
      module.ml_infrastructure.training_role_policy.policy
    )))
  }
  
  check "no_admin_access" {
    description = "IAM roles should not have admin access"
    condition = !can(regex("AdministratorAccess", jsonencode(
      module.ml_infrastructure.training_role.managed_policy_arns
    )))
  }
}
```

#### 5.2 Continuous Security Validation

**Security Baseline Testing**:
```bash
#!/bin/bash
# scripts/security-validation.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔒 Starting security validation for ML infrastructure..."

# Function to run security checks
run_security_check() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Running $test_name... "
    
    if eval "$command" > /tmp/security_check.log 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "Error details:"
        cat /tmp/security_check.log
        return 1
    fi
}

# Security validation checks
CHECKS=(
    "TFSec static analysis|tfsec --format=json --out=tfsec-results.json ."
    "Checkov policy validation|checkov -d . --framework terraform --output json --output-file checkov-results.json"
    "Terraform plan validation|terraform plan -detailed-exitcode -out=tfplan"
    "OPA policy validation|opa test policies/ --format=json --output=opa-results.json"
    "Security group analysis|python scripts/analyze_security_groups.py"
    "IAM policy analysis|python scripts/analyze_iam_policies.py"
    "Encryption validation|python scripts/validate_encryption.py"
)

failed_checks=0
total_checks=${#CHECKS[@]}

for check in "${CHECKS[@]}"; do
    test_name="${check%%|*}"
    command="${check##*|}"
    
    if ! run_security_check "$test_name" "$command"; then
        ((failed_checks++))
    fi
done

# Generate security report
echo ""
echo "📊 Security Validation Summary"
echo "=============================="
echo "Total checks: $total_checks"
echo -e "Passed: ${GREEN}$((total_checks - failed_checks))${NC}"
echo -e "Failed: ${RED}$failed_checks${NC}"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}🎉 All security checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed_checks security check(s) failed${NC}"
    echo "Please review the errors above and fix the issues before proceeding."
    exit 1
fi
```

**Automated Compliance Checking**:
```python
#!/usr/bin/env python3
# scripts/compliance_checker.py

import json
import boto3
import sys
from typing import Dict, List, Any

class ComplianceChecker:
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.s3 = boto3.client('s3')
        self.iam = boto3.client('iam')
        self.kms = boto3.client('kms')
        
    def check_encryption_compliance(self) -> Dict[str, Any]:
        """Check encryption compliance across services"""
        results = {
            'compliant': True,
            'violations': [],
            'checks_performed': []
        }
        
        # Check EBS volumes
        volumes = self.ec2.describe_volumes()['Volumes']
        for volume in volumes:
            if not volume.get('Encrypted', False):
                results['violations'].append({
                    'resource': f"EBS Volume {volume['VolumeId']}",
                    'issue': 'Volume is not encrypted',
                    'severity': 'HIGH'
                })
                results['compliant'] = False
        
        results['checks_performed'].append('EBS encryption validation')
        
        # Check S3 buckets
        buckets = self.s3.list_buckets()['Buckets']
        for bucket in buckets:
            bucket_name = bucket['Name']
            try:
                encryption = self.s3.get_bucket_encryption(Bucket=bucket_name)
                # Check if using KMS encryption
                rules = encryption['ServerSideEncryptionConfiguration']['Rules']
                if not any(rule['ApplyServerSideEncryptionByDefault']['SSEAlgorithm'] == 'aws:kms' 
                          for rule in rules):
                    results['violations'].append({
                        'resource': f"S3 Bucket {bucket_name}",
                        'issue': 'Bucket not using KMS encryption',
                        'severity': 'MEDIUM'
                    })
                    results['compliant'] = False
            except self.s3.exceptions.NoSuchBucket:
                pass
            except Exception:
                results['violations'].append({
                    'resource': f"S3 Bucket {bucket_name}",
                    'issue': 'No encryption configuration found',
                    'severity': 'HIGH'
                })
                results['compliant'] = False
        
        results['checks_performed'].append('S3 encryption validation')
        return results
    
    def check_iam_compliance(self) -> Dict[str, Any]:
        """Check IAM compliance for least privilege"""
        results = {
            'compliant': True,
            'violations': [],
            'checks_performed': []
        }
        
        # Check for overly permissive policies
        roles = self.iam.list_roles()['Roles']
        for role in roles:
            if 'ml-' in role['RoleName'].lower():
                # Check attached policies
                attached_policies = self.iam.list_attached_role_policies(
                    RoleName=role['RoleName']
                )['AttachedPolicies']
                
                for policy in attached_policies:
                    if 'Administrator' in policy['PolicyName']:
                        results['violations'].append({
                            'resource': f"IAM Role {role['RoleName']}",
                            'issue': f"Role has administrator policy: {policy['PolicyName']}",
                            'severity': 'CRITICAL'
                        })
                        results['compliant'] = False
        
        results['checks_performed'].append('IAM policy validation')
        return results
    
    def check_network_compliance(self) -> Dict[str, Any]:
        """Check network security compliance"""
        results = {
            'compliant': True,
            'violations': [],
            'checks_performed': []
        }
        
        # Check security groups
        security_groups = self.ec2.describe_security_groups()['SecurityGroups']
        for sg in security_groups:
            # Check for overly permissive rules
            for rule in sg.get('IpPermissions', []):
                for ip_range in rule.get('IpRanges', []):
                    if ip_range.get('CidrIp') == '0.0.0.0/0':
                        if rule.get('FromPort', 0) <= 22 <= rule.get('ToPort', 65535):
                            results['violations'].append({
                                'resource': f"Security Group {sg['GroupId']}",
                                'issue': 'SSH access open to 0.0.0.0/0',
                                'severity': 'CRITICAL'
                            })
                            results['compliant'] = False
        
        results['checks_performed'].append('Security group validation')
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        print("🔍 Running compliance checks...")
        
        encryption_results = self.check_encryption_compliance()
        iam_results = self.check_iam_compliance()
        network_results = self.check_network_compliance()
        
        all_violations = (
            encryption_results['violations'] +
            iam_results['violations'] +
            network_results['violations']
        )
        
        overall_compliant = (
            encryption_results['compliant'] and
            iam_results['compliant'] and
            network_results['compliant']
        )
        
        report = {
            'timestamp': '2024-01-15T10:30:00Z',
            'overall_compliance': overall_compliant,
            'total_violations': len(all_violations),
            'violations_by_severity': {
                'CRITICAL': len([v for v in all_violations if v['severity'] == 'CRITICAL']),
                'HIGH': len([v for v in all_violations if v['severity'] == 'HIGH']),
                'MEDIUM': len([v for v in all_violations if v['severity'] == 'MEDIUM'])
            },
            'detailed_results': {
                'encryption': encryption_results,
                'iam': iam_results,
                'network': network_results
            }
        }
        
        return report

if __name__ == "__main__":
    checker = ComplianceChecker()
    report = checker.generate_report()
    
    print(json.dumps(report, indent=2))
    
    if not report['overall_compliance']:
        print(f"\n❌ Compliance check failed with {report['total_violations']} violations")
        sys.exit(1)
    else:
        print("\n✅ All compliance checks passed!")
        sys.exit(0)
```

---

## 🔍 Key Questions

### Beginner Level

1. **Q**: What is the principle of least privilege and how does it apply to IAM roles for ML workloads?
   **A**: Least privilege means granting only the minimum permissions necessary to perform required tasks. For ML workloads, this means specific S3 bucket access, limited compute permissions, and time-bound access rather than broad administrative rights.

2. **Q**: Why is input validation important in Terraform modules, and what are common security risks?
   **A**: Input validation prevents injection attacks, ensures compliance with security policies, and prevents misconfigurations. Common risks include CIDR blocks allowing public access, weak instance types, and missing encryption settings.

3. **Q**: What are the key components of a hardened security group for ML training instances?
   **A**: Minimal ingress rules (SSH only from trusted networks), explicit egress rules (HTTPS and required services only), self-referencing for cluster communication, and no 0.0.0.0/0 access except for specific approved services.

### Intermediate Level

4. **Q**: Design an IAM policy for a SageMaker training job that follows least privilege principles while allowing access to specific S3 buckets and CloudWatch logging.
   **A**:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:ListBucket"],
         "Resource": ["arn:aws:s3:::training-data-bucket/*", "arn:aws:s3:::training-data-bucket"],
         "Condition": {"StringEquals": {"s3:ExistingObjectTag/DataClass": "approved"}}
       },
       {
         "Effect": "Allow", 
         "Action": ["s3:PutObject"],
         "Resource": ["arn:aws:s3:::model-output-bucket/*"],
         "Condition": {"StringEquals": {"s3:x-amz-server-side-encryption": "aws:kms"}}
       },
       {
         "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": ["arn:aws:logs:*:*:log-group:/aws/sagemaker/*"]
       }
     ]
   }
   ```

5. **Q**: How would you implement cross-account access for a multi-environment ML pipeline while maintaining security?
   **A**: Use cross-account IAM roles with external IDs, implement condition-based access controls, use time-limited sessions, enable CloudTrail logging across accounts, and implement least privilege permissions per environment.

6. **Q**: Explain how to implement network hardening for a VPC containing ML workloads.
   **A**: Use private subnets for compute, implement VPC endpoints for AWS services, configure restrictive NACLs, use security groups with minimal access, enable VPC Flow Logs, and implement NAT gateways for controlled outbound access.

### Advanced Level

7. **Q**: Design a comprehensive security hardening strategy for a multi-tenant ML platform that supports different security classifications (public, internal, confidential).
   **A**:
   ```
   Strategy Components:
   - Data classification tagging and automated enforcement
   - Tiered IAM policies based on classification levels
   - Network micro-segmentation per classification
   - Encryption key separation per tenant and classification
   - Separate compute environments with different hardening levels
   - Compliance monitoring and automated remediation
   - Multi-factor authentication for confidential data access
   - Regular access reviews and automated deprovisioning
   ```

8. **Q**: How would you implement automated security testing and compliance validation for IaC modules in a CI/CD pipeline?
   **A**: Integrate multiple scanning tools (TFSec, Checkov, OPA), implement policy-as-code validation, create security-focused test suites, use infrastructure testing frameworks, implement compliance checking scripts, and create security gates with approval workflows.

### Tricky Questions

9. **Q**: You discover that a production ML training environment has been compromised through an overprivileged IAM role. Design a comprehensive response plan that includes immediate containment, investigation, and long-term prevention measures.
   **A**:
   ```
   Immediate Response:
   - Disable compromised IAM roles and rotate all credentials
   - Isolate affected resources using security groups
   - Enable detailed CloudTrail logging and monitoring
   - Snapshot affected instances and data for forensics
   
   Investigation:
   - Analyze CloudTrail logs for unauthorized activities
   - Review all resource modifications and data access
   - Identify scope of compromise and data exposure
   - Document timeline and attack vectors
   
   Long-term Prevention:
   - Implement least privilege IAM policies
   - Deploy comprehensive monitoring and alerting
   - Regular security audits and penetration testing
   - Implement just-in-time access for privileged operations
   - Enhanced security training for development teams
   ```

10. **Q**: Design an IaC architecture that supports zero-trust principles for a federated learning system where multiple organizations contribute data but maintain strict isolation requirements.
    **A**:
    ```
    Zero-Trust Architecture:
    - Identity-based access control for all resources
    - Encrypted communication channels between all components
    - Micro-segmentation with explicit allow rules
    - Continuous monitoring and behavioral analysis
    - Regular access certification and automated deprovisioning
    
    Implementation:
    - Separate AWS accounts per organization with cross-account roles
    - Customer-managed KMS keys with organization-specific access
    - VPC peering with restrictive security groups
    - Service mesh with mutual TLS authentication
    - Policy-as-code for consistent security enforcement
    - Centralized logging and SIEM integration
    - Regular compliance validation and reporting
    ```

---

## 🛡️ Security Deep Dive

### Advanced Threat Scenarios

#### Privilege Escalation Attacks

**Attack Vectors in ML Environments**:
```
Common Escalation Paths:
- Service account token theft from containers
- IAM role assumption through vulnerable applications
- Cross-account access through misconfigured trust policies
- Container escape leading to host compromise
- Kubernetes RBAC bypass through pod security contexts

Prevention Strategies:
- Short-lived tokens with regular rotation
- Pod security policies and admission controllers
- Network segmentation and zero-trust networking
- Comprehensive monitoring and anomaly detection
- Regular access reviews and automated cleanup
```

#### Supply Chain Security

**IaC Module Security**:
```
Supply Chain Risks:
- Malicious Terraform providers
- Compromised module registries
- Dependency confusion attacks
- Backdoors in infrastructure code

Mitigation Measures:
- Provider signature verification
- Private module registries
- Dependency scanning and validation
- Code signing and verification
- Regular security audits of dependencies
```

### Compliance and Governance

#### Automated Compliance Validation

**Regulatory Framework Implementation**:
```
SOC 2 Type II Controls:
- CC6.1: Logical and physical access controls
- CC6.2: Prior authorization of system changes
- CC6.3: User access provisioning and deprovisioning
- CC6.6: Management of system vulnerabilities

Implementation in IaC:
- Automated access control validation
- Change management through GitOps
- User lifecycle management automation
- Vulnerability scanning and remediation
```

---

## 🚀 Performance Optimization

### IaC Performance Best Practices

#### Module Optimization

**Resource Organization**:
```terraform
# Efficient resource organization
locals {
  # Pre-compute values to avoid repeated evaluation
  common_tags = merge(var.base_tags, {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  })
  
  # Use for_each for dynamic resource creation
  availability_zones = toset(var.availability_zones)
  
  # Optimize data source usage
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# Use dynamic blocks efficiently
resource "aws_security_group" "ml_training" {
  name_prefix = "ml-training-"
  vpc_id      = var.vpc_id
  
  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}
```

#### State Management Optimization

**Remote State Configuration**:
```terraform
# Optimized backend configuration
terraform {
  backend "s3" {
    bucket         = "terraform-state-ml-infrastructure"
    key            = "environments/production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-west-2:123456789012:key/terraform-state-key"
    dynamodb_table = "terraform-state-lock"
    
    # Performance optimizations
    skip_region_validation      = true
    skip_credentials_validation = true
    skip_metadata_api_check     = true
  }
}
```

---

## 📝 Practical Exercises

### Exercise 1: Secure Module Development
Create a comprehensive Terraform module for an ML training cluster that includes:
- Least privilege IAM roles and policies
- Network hardening with private subnets and security groups
- Encryption at rest and in transit
- Comprehensive input validation
- Security testing and compliance validation

### Exercise 2: IAM Policy Optimization
Design and implement a complete IAM strategy for a multi-tenant ML platform including:
- Role hierarchy and permission boundaries
- Cross-account access patterns
- Service-to-service authentication
- Automated access reviews and cleanup
- Emergency access procedures

### Exercise 3: Security Hardening Assessment
Conduct a comprehensive security assessment of an existing ML infrastructure including:
- Automated security scanning and validation
- Compliance checking against industry frameworks
- Penetration testing simulation
- Risk assessment and mitigation planning
- Continuous monitoring implementation

### Exercise 4: Zero-Trust Implementation
Design and implement a zero-trust architecture for an AI research platform including:
- Identity-based access control
- Network micro-segmentation
- Continuous monitoring and validation
- Policy-as-code enforcement
- Incident response automation

---

## 🔗 Next Steps
In the next section (day02_004), we'll explore comprehensive secret management strategies using HashiCorp Vault, AWS KMS, and Azure Key Vault, focusing on secure credential lifecycle management, automated rotation, and integration with AI/ML pipelines.