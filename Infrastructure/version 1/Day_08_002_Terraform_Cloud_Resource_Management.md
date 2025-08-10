# Day 8.2: Terraform for Cloud Resource Management

## 🏗️ Infrastructure as Code & Automation - Part 2

**Focus**: Infrastructure as Code, Multi-Cloud Deployment, State Management  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master Terraform infrastructure as code patterns for ML platforms
- Learn advanced state management and collaboration strategies
- Understand multi-cloud deployment architectures and resource optimization
- Analyze module design patterns and reusable infrastructure components

---

## 🏗️ Terraform Theoretical Framework

### **Infrastructure as Code Principles**

Terraform enables declarative infrastructure management through code, providing version control, repeatability, and collaboration capabilities essential for ML platform operations.

**IaC Mathematical Model:**
```
Infrastructure State Theory:
Desired_State = f(Configuration_Code, Input_Variables, Provider_APIs)
Actual_State = Current_Infrastructure_Resources
State_Drift = |Desired_State - Actual_State|

Terraform Operations:
Plan: Δ = Desired_State - Actual_State
Apply: Actual_State ← Actual_State + Δ
Destroy: Actual_State ← ∅

Resource Dependency Graph:
G = (V, E) where:
V = {terraform_resources}
E = {(u,v) | resource u depends on resource v}

Topological ordering ensures correct resource creation/destruction sequence
Time Complexity: O(V + E) for dependency resolution
```

**Terraform State Management Theory:**
```
State File Structure:
state = {
    "version": terraform_version,
    "terraform_version": "1.5.0",
    "resources": [
        {
            "type": "resource_type",
            "name": "resource_name", 
            "provider": "provider_name",
            "instances": [
                {
                    "attributes": {resource_attributes},
                    "dependencies": [dependency_list]
                }
            ]
        }
    ]
}

State Locking Mechanism:
Lock = {
    "ID": unique_lock_id,
    "Operation": "plan|apply|refresh",
    "Info": operation_metadata,
    "Who": user_identity,
    "Version": terraform_version,
    "Created": timestamp,
    "Path": state_file_path
}

Concurrent Access Control:
if acquire_lock(state_file):
    perform_operation()
    release_lock(state_file)
else:
    wait_or_fail("State locked by another operation")

State Consistency Guarantees:
- Atomicity: Operations complete fully or not at all
- Consistency: State transitions maintain resource relationships
- Isolation: Concurrent operations are serialized through locking
- Durability: State persists across failures
```

**Resource Lifecycle Management:**
```
Resource Lifecycle States:
Create → Update → Destroy

Lifecycle Rules:
1. create_before_destroy: Create replacement before destroying original
2. prevent_destroy: Prevent accidental resource destruction
3. ignore_changes: Ignore specific attribute changes
4. replace_triggered_by: Force replacement when specified resources change

Example Lifecycle Configuration:
resource "aws_instance" "ml_training_instance" {
  ami           = var.training_ami
  instance_type = var.instance_type
  
  lifecycle {
    create_before_destroy = true
    prevent_destroy      = true
    ignore_changes = [
      ami,  # Ignore AMI changes to prevent unnecessary replacements
      user_data,
    ]
  }
  
  tags = {
    Name        = "ml-training-${random_id.instance_id.hex}"
    Environment = var.environment
    Purpose     = "ml-training"
  }
}

Replacement Triggers:
resource "null_resource" "model_deployment_trigger" {
  triggers = {
    model_version = var.model_version
    config_hash   = md5(file("${path.module}/config.yaml"))
  }
  
  provisioner "local-exec" {
    command = "kubectl rollout restart deployment/model-service"
  }
}
```

---

## ☁️ Multi-Cloud Architecture Patterns

### **Cloud Provider Abstraction**

**Provider Configuration Strategy:**
```
Multi-Provider Configuration:
# AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  alias  = "primary"
  region = var.aws_primary_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
    }
  }
}

provider "aws" {
  alias  = "disaster_recovery"
  region = var.aws_dr_region
}

provider "google" {
  alias   = "ml_workloads"
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "azurerm" {
  alias = "data_processing"
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

Cross-Cloud Resource Dependencies:
# Data stored in AWS S3
resource "aws_s3_bucket" "ml_datasets" {
  provider = aws.primary
  bucket   = "${var.project_name}-ml-datasets-${random_id.bucket_suffix.hex}"
}

# Processing cluster in GCP
resource "google_container_cluster" "ml_cluster" {
  provider = google.ml_workloads
  name     = "${var.project_name}-ml-cluster"
  location = var.gcp_region
  
  # Configure cross-cloud data access
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }
}

# Cross-cloud IAM setup for data access
resource "google_service_account" "cross_cloud_access" {
  provider     = google.ml_workloads
  account_id   = "cross-cloud-access"
  display_name = "Cross-cloud data access service account"
}

resource "aws_iam_role" "gcp_access_role" {
  provider = aws.primary
  name     = "gcp-cross-cloud-access"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/sts.googleapis.com"
        }
        Condition = {
          StringEquals = {
            "sts.googleapis.com:sub" = "system:serviceaccount:default:${google_service_account.cross_cloud_access.email}"
          }
        }
      }
    ]
  })
}
```

**Multi-Cloud Data Strategy:**
```
Data Residency and Compliance:
locals {
  # Data classification and residency requirements
  data_classification = {
    public = {
      storage_class = "standard"
      replication   = "multi_region"
      encryption    = "google_managed"
    }
    internal = {
      storage_class = "standard"
      replication   = "regional"
      encryption    = "customer_managed"
    }
    restricted = {
      storage_class = "coldline"
      replication   = "single_region"
      encryption    = "customer_managed"
      location_constraint = "EU"
    }
  }
  
  # Regional compliance mapping
  compliance_regions = {
    gdpr     = ["eu-west1", "eu-central1", "eu-west2"]
    ccpa     = ["us-west1", "us-west2", "us-central1"]
    apac     = ["asia-east1", "asia-southeast1", "australia-southeast1"]
  }
}

# Compliance-aware storage bucket creation
resource "google_storage_bucket" "ml_data_bucket" {
  for_each = var.data_buckets
  
  name     = "${var.project_name}-${each.key}-${random_id.bucket_suffix.hex}"
  location = local.compliance_regions[each.value.compliance_region][0]
  
  storage_class = local.data_classification[each.value.classification].storage_class
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = each.value.retention_days
    }
    action {
      type = "Delete"
    }
  }
  
  encryption {
    default_kms_key_name = each.value.classification == "restricted" ? 
      google_kms_crypto_key.data_encryption_key.id : null
  }
  
  uniform_bucket_level_access = true
  
  dynamic "cors" {
    for_each = each.value.enable_cors ? [1] : []
    content {
      origin          = ["https://*.${var.domain_name}"]
      method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
      response_header = ["*"]
      max_age_seconds = 3600
    }
  }
}

Cross-Cloud Networking:
# VPC Peering between AWS and GCP
resource "aws_vpc" "ml_platform_vpc" {
  provider             = aws.primary
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-ml-platform-vpc"
  }
}

resource "google_compute_network" "ml_platform_network" {
  provider = google.ml_workloads
  name     = "${var.project_name}-ml-network"
  
  auto_create_subnetworks = false
  routing_mode           = "REGIONAL"
}

# VPN Gateway for secure cross-cloud communication
resource "aws_vpn_gateway" "ml_platform_vpn_gw" {
  provider = aws.primary
  vpc_id   = aws_vpc.ml_platform_vpc.id
  
  tags = {
    Name = "${var.project_name}-vpn-gateway"
  }
}

resource "google_compute_vpn_gateway" "ml_platform_vpn_gw" {
  provider = google.ml_workloads
  name     = "${var.project_name}-vpn-gateway"
  network  = google_compute_network.ml_platform_network.id
  region   = var.gcp_region
}

# Cross-cloud VPN tunnel
resource "aws_vpn_connection" "cross_cloud_tunnel" {
  provider            = aws.primary
  vpn_gateway_id      = aws_vpn_gateway.ml_platform_vpn_gw.id
  customer_gateway_id = aws_customer_gateway.gcp_gateway.id
  type                = "ipsec.1"
  static_routes_only  = true
  
  tags = {
    Name = "${var.project_name}-cross-cloud-tunnel"
  }
}
```

---

## 📦 Advanced Module Design

### **Reusable ML Infrastructure Modules**

**Module Architecture Patterns:**
```
Module Hierarchy:
root/
├── modules/
│   ├── compute/
│   │   ├── ml-cluster/
│   │   ├── training-infrastructure/
│   │   └── inference-infrastructure/
│   ├── storage/
│   │   ├── data-lake/
│   │   ├── feature-store/
│   │   └── model-registry/
│   ├── networking/
│   │   ├── vpc/
│   │   ├── security-groups/
│   │   └── load-balancers/
│   └── security/
│       ├── iam/
│       ├── encryption/
│       └── secrets-management/
├── environments/
│   ├── development/
│   ├── staging/
│   └── production/
└── shared/
    ├── variables.tf
    ├── outputs.tf
    └── locals.tf

ML Cluster Module:
# modules/compute/ml-cluster/main.tf
terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

locals {
  node_pools = {
    for pool_name, pool_config in var.node_pools : pool_name => merge(
      {
        node_count           = 1
        min_node_count      = 0
        max_node_count      = 10
        machine_type        = "n1-standard-4"
        disk_size_gb        = 100
        disk_type           = "pd-standard"
        preemptible         = false
        auto_upgrade        = true
        auto_repair         = true
        initial_node_count  = 1
      },
      pool_config
    )
  }
  
  # GPU node pool configurations
  gpu_node_pools = {
    for pool_name, pool_config in local.node_pools : pool_name => pool_config
    if lookup(pool_config, "gpu_type", null) != null
  }
  
  # Standard node pool configurations
  standard_node_pools = {
    for pool_name, pool_config in local.node_pools : pool_name => pool_config
    if lookup(pool_config, "gpu_type", null) == null
  }
}

resource "google_container_cluster" "ml_cluster" {
  name     = var.cluster_name
  location = var.region
  
  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Cluster-level configurations
  cluster_autoscaling {
    enabled = var.enable_cluster_autoscaling
    
    dynamic "resource_limits" {
      for_each = var.cluster_autoscaling_limits
      content {
        resource_type = resource_limits.value.resource_type
        minimum       = resource_limits.value.minimum
        maximum       = resource_limits.value.maximum
      }
    }
  }
  
  # Workload Identity for secure GCP service access
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Network configuration
  network    = var.network
  subnetwork = var.subnetwork
  
  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }
  
  # Master authorized networks
  dynamic "master_authorized_networks_config" {
    for_each = var.master_authorized_networks != null ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }
  
  # Add-on configurations
  addons_config {
    http_load_balancing {
      disabled = !var.enable_http_load_balancing
    }
    
    horizontal_pod_autoscaling {
      disabled = !var.enable_horizontal_pod_autoscaling
    }
    
    network_policy_config {
      disabled = !var.enable_network_policy
    }
    
    istio_config {
      disabled = !var.enable_istio
      auth     = var.istio_auth
    }
  }
  
  # Logging and monitoring
  logging_service    = var.logging_service
  monitoring_service = var.monitoring_service
  
  # Maintenance policy
  maintenance_policy {
    daily_maintenance_window {
      start_time = var.maintenance_start_time
    }
  }
}

# Standard node pools
resource "google_container_node_pool" "standard_pools" {
  for_each = local.standard_node_pools
  
  name       = each.key
  location   = var.region
  cluster    = google_container_cluster.ml_cluster.name
  node_count = each.value.initial_node_count
  
  autoscaling {
    min_node_count = each.value.min_node_count
    max_node_count = each.value.max_node_count
  }
  
  management {
    auto_repair  = each.value.auto_repair
    auto_upgrade = each.value.auto_upgrade
  }
  
  node_config {
    preemptible  = each.value.preemptible
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size_gb
    disk_type    = each.value.disk_type
    
    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Labels and taints
    labels = merge(
      var.default_node_labels,
      each.value.labels,
      {
        workload-type = "standard"
        pool-name     = each.key
      }
    )
    
    dynamic "taint" {
      for_each = lookup(each.value, "taints", [])
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }
    
    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

# GPU node pools
resource "google_container_node_pool" "gpu_pools" {
  for_each = local.gpu_node_pools
  
  name       = each.key
  location   = var.region
  cluster    = google_container_cluster.ml_cluster.name
  node_count = each.value.initial_node_count
  
  autoscaling {
    min_node_count = each.value.min_node_count
    max_node_count = each.value.max_node_count
  }
  
  management {
    auto_repair  = each.value.auto_repair
    auto_upgrade = each.value.auto_upgrade
  }
  
  node_config {
    preemptible  = each.value.preemptible
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size_gb
    disk_type    = each.value.disk_type
    
    # GPU configuration
    guest_accelerator {
      type  = each.value.gpu_type
      count = each.value.gpu_count
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = merge(
      var.default_node_labels,
      each.value.labels,
      {
        workload-type = "gpu"
        gpu-type      = each.value.gpu_type
        pool-name     = each.key
      }
    )
    
    # GPU-specific taints
    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
    
    dynamic "taint" {
      for_each = lookup(each.value, "taints", [])
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }
    
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}
```

**Feature Store Module:**
```
# modules/storage/feature-store/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

locals {
  feature_tables = {
    for table_name, table_config in var.feature_tables : table_name => merge(
      {
        deletion_protection = true
        partition_type      = "DAY"
        partition_field     = "timestamp"
        clustering_fields   = []
        labels             = {}
        expiration_ms      = null
      },
      table_config
    )
  }
}

# BigQuery dataset for feature storage
resource "google_bigquery_dataset" "feature_store" {
  dataset_id    = var.dataset_id
  friendly_name = var.dataset_friendly_name
  description   = var.dataset_description
  location      = var.location
  
  # Access control
  dynamic "access" {
    for_each = var.dataset_access
    content {
      role          = access.value.role
      user_by_email = lookup(access.value, "user_by_email", null)
      group_by_email = lookup(access.value, "group_by_email", null)
      special_group = lookup(access.value, "special_group", null)
    }
  }
  
  # Default table expiration
  default_table_expiration_ms = var.default_table_expiration_ms
  
  labels = var.labels
}

# Feature tables
resource "google_bigquery_table" "feature_tables" {
  for_each = local.feature_tables
  
  dataset_id          = google_bigquery_dataset.feature_store.dataset_id
  table_id            = each.key
  deletion_protection = each.value.deletion_protection
  
  # Time partitioning
  time_partitioning {
    type  = each.value.partition_type
    field = each.value.partition_field
  }
  
  # Clustering
  dynamic "clustering" {
    for_each = length(each.value.clustering_fields) > 0 ? [1] : []
    content {
      fields = each.value.clustering_fields
    }
  }
  
  # Schema definition
  schema = jsonencode(each.value.schema)
  
  # Expiration
  expiration_time = each.value.expiration_ms
  
  labels = merge(var.labels, each.value.labels)
}

# Redis cluster for online feature serving
resource "google_redis_instance" "feature_cache" {
  count = var.enable_online_serving ? 1 : 0
  
  name           = "${var.feature_store_name}-cache"
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region
  
  # Network configuration
  authorized_network = var.network
  connect_mode       = "DIRECT_PEERING"
  
  # Redis configuration
  redis_version     = var.redis_version
  display_name      = "${var.feature_store_name} Feature Cache"
  reserved_ip_range = var.redis_reserved_ip_range
  
  # Maintenance
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  labels = var.labels
}

# Cloud Function for feature serving
resource "google_cloudfunctions2_function" "feature_serving" {
  count = var.enable_online_serving ? 1 : 0
  
  name     = "${var.feature_store_name}-serving"
  location = var.region
  
  build_config {
    runtime     = "python39"
    entry_point = "serve_features"
    
    source {
      storage_source {
        bucket = var.source_bucket
        object = var.source_object
      }
    }
  }
  
  service_config {
    max_instance_count               = var.max_instances
    min_instance_count               = var.min_instances
    available_memory                 = var.memory_limit
    timeout_seconds                  = var.timeout_seconds
    ingress_settings                 = "ALLOW_INTERNAL_ONLY"
    all_traffic_on_latest_revision   = true
    
    environment_variables = {
      REDIS_HOST         = var.enable_online_serving ? google_redis_instance.feature_cache[0].host : ""
      REDIS_PORT         = var.enable_online_serving ? google_redis_instance.feature_cache[0].port : ""
      BIGQUERY_DATASET   = google_bigquery_dataset.feature_store.dataset_id
      FEATURE_STORE_NAME = var.feature_store_name
    }
    
    service_account_email = var.service_account_email
  }
}

# IAM bindings for feature store access
resource "google_bigquery_dataset_iam_binding" "feature_store_viewers" {
  dataset_id = google_bigquery_dataset.feature_store.dataset_id
  role       = "roles/bigquery.dataViewer"
  
  members = var.feature_store_viewers
}

resource "google_bigquery_dataset_iam_binding" "feature_store_editors" {
  dataset_id = google_bigquery_dataset.feature_store.dataset_id
  role       = "roles/bigquery.dataEditor"
  
  members = var.feature_store_editors
}
```

---

## 🔒 State Management and Collaboration

### **Remote State Configuration**

**Backend Configuration Strategies:**
```
Remote State Backends:

1. Google Cloud Storage Backend:
terraform {
  backend "gcs" {
    bucket  = "terraform-state-ml-platform"
    prefix  = "environments/production"
    
    # State locking with Cloud Storage
    # Requires google_storage_bucket_object_acl with proper permissions
  }
}

2. AWS S3 Backend with DynamoDB Locking:
terraform {
  backend "s3" {
    bucket         = "terraform-state-ml-platform"
    key            = "environments/production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
    dynamodb_table = "terraform-state-lock"
  }
}

3. Azure Storage Backend:
terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "terraformstatemlplatform"
    container_name       = "tfstate"
    key                  = "environments/production/terraform.tfstate"
  }
}

4. Terraform Cloud Backend:
terraform {
  cloud {
    organization = "ml-platform-org"
    
    workspaces {
      name = "ml-platform-production"
    }
  }
}

State File Security:
# Encryption at rest
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.bucket
  
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.terraform_state_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

# Versioning for state recovery
resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle policy for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  
  rule {
    id     = "terraform_state_lifecycle"
    status = "Enabled"
    
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
    
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 60
      storage_class   = "GLACIER"
    }
  }
}
```

**State Import and Migration:**
```
State Import Procedures:
# Import existing infrastructure
terraform import aws_instance.ml_training_instance i-1234567890abcdef0
terraform import google_compute_instance.ml_training_vm projects/my-project/zones/us-central1-a/instances/ml-training-vm

# Bulk import script
import_resources() {
  local resource_list=$1
  
  while IFS= read -r line; do
    if [[ $line =~ ^#.*$ ]] || [[ -z $line ]]; then
      continue
    fi
    
    resource_type=$(echo $line | cut -d',' -f1)
    resource_name=$(echo $line | cut -d',' -f2)
    resource_id=$(echo $line | cut -d',' -f3)
    
    echo "Importing $resource_type.$resource_name ($resource_id)"
    terraform import "$resource_type.$resource_name" "$resource_id"
    
    if [ $? -eq 0 ]; then
      echo "✓ Successfully imported $resource_type.$resource_name"
    else
      echo "✗ Failed to import $resource_type.$resource_name"
    fi
  done < "$resource_list"
}

State Migration Between Backends:
# Step 1: Configure new backend
terraform {
  backend "gcs" {
    bucket = "new-terraform-state-bucket"
    prefix = "ml-platform"
  }
}

# Step 2: Initialize with backend migration
terraform init -migrate-state

# Step 3: Verify state integrity
terraform plan

State Splitting and Merging:
# Split state into smaller components
terraform state mv 'module.networking' 'module.core_networking'
terraform state mv 'module.compute' 'module.training_compute'

# Move resources between states
terraform state mv 'aws_instance.web[0]' 'aws_instance.web_primary'
terraform state mv 'aws_instance.web[1]' 'aws_instance.web_secondary'

# Remove resources from state without destroying
terraform state rm 'aws_instance.temporary_instance'

State Backup and Recovery:
#!/bin/bash
backup_terraform_state() {
  local environment=$1
  local backup_location=$2
  local timestamp=$(date +%Y%m%d_%H%M%S)
  
  # Pull current state
  terraform state pull > "terraform.tfstate.${timestamp}"
  
  # Upload to backup location
  aws s3 cp "terraform.tfstate.${timestamp}" \
    "${backup_location}/${environment}/terraform.tfstate.${timestamp}"
  
  # Keep local backups for 7 days
  find . -name "terraform.tfstate.*" -mtime +7 -delete
  
  echo "State backed up to ${backup_location}/${environment}/terraform.tfstate.${timestamp}"
}

recover_terraform_state() {
  local backup_file=$1
  
  # Download backup
  aws s3 cp "$backup_file" ./terraform.tfstate.backup
  
  # Push state
  terraform state push ./terraform.tfstate.backup
  
  # Verify recovery
  terraform plan
  
  echo "State recovered from $backup_file"
}
```

### **Workspace Management**

**Multi-Environment Strategies:**
```
Workspace-Based Environment Management:
# Create workspaces for different environments
terraform workspace new development
terraform workspace new staging  
terraform workspace new production

# Switch between workspaces
terraform workspace select production

# Environment-specific variables
locals {
  environment_configs = {
    development = {
      instance_type     = "t3.medium"
      min_size         = 1
      max_size         = 3
      desired_capacity = 1
      spot_instances   = true
    }
    staging = {
      instance_type     = "t3.large"
      min_size         = 2
      max_size         = 6
      desired_capacity = 2
      spot_instances   = true
    }
    production = {
      instance_type     = "t3.xlarge"
      min_size         = 3
      max_size         = 20
      desired_capacity = 5
      spot_instances   = false
    }
  }
  
  current_env = local.environment_configs[terraform.workspace]
}

# Use environment-specific configurations
resource "aws_launch_template" "ml_workers" {
  name_prefix   = "${terraform.workspace}-ml-workers-"
  instance_type = local.current_env.instance_type
  
  # Environment-specific tags
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${terraform.workspace}-ml-worker"
      Environment = terraform.workspace
      Project     = var.project_name
    }
  }
}

Directory-Based Environment Management:
environments/
├── development/
│   ├── main.tf
│   ├── variables.tf
│   ├── terraform.tfvars
│   └── backend.tf
├── staging/
│   ├── main.tf
│   ├── variables.tf
│   ├── terraform.tfvars
│   └── backend.tf
└── production/
    ├── main.tf
    ├── variables.tf
    ├── terraform.tfvars
    └── backend.tf

# environments/production/backend.tf
terraform {
  backend "gcs" {
    bucket = "ml-platform-terraform-state"
    prefix = "environments/production"
  }
}

# environments/production/terraform.tfvars
environment      = "production"
cluster_name     = "ml-platform-prod"
node_pool_config = {
  training = {
    machine_type   = "n1-highmem-8"
    gpu_type      = "nvidia-tesla-v100"
    gpu_count     = 2
    min_nodes     = 0
    max_nodes     = 20
    preemptible   = false
  }
  inference = {
    machine_type   = "n1-standard-4"
    min_nodes     = 3
    max_nodes     = 50
    preemptible   = false
  }
}

Environment Promotion Pipeline:
#!/bin/bash
promote_environment() {
  local source_env=$1
  local target_env=$2
  
  echo "Promoting configuration from $source_env to $target_env"
  
  # Extract configuration from source environment
  cd "environments/$source_env"
  terraform output -json > "../${target_env}/promoted_config.json"
  
  cd "../$target_env"
  
  # Apply promoted configuration
  terraform plan -var-file="promoted_config.json"
  
  read -p "Apply changes to $target_env? (y/N): " confirm
  if [[ $confirm == [yY] ]]; then
    terraform apply -var-file="promoted_config.json"
    echo "✓ Successfully promoted to $target_env"
  else
    echo "Promotion cancelled"
  fi
}

Cross-Environment Data Sharing:
# Remote state data sources
data "terraform_remote_state" "shared_services" {
  backend = "gcs"
  config = {
    bucket = "ml-platform-terraform-state"
    prefix = "shared/services"
  }
}

data "terraform_remote_state" "networking" {
  backend = "gcs"
  config = {
    bucket = "ml-platform-terraform-state"
    prefix = "shared/networking"
  }
}

# Use shared resources
resource "google_container_cluster" "ml_cluster" {
  name     = "${var.environment}-ml-cluster"
  network  = data.terraform_remote_state.networking.outputs.network_name
  subnetwork = data.terraform_remote_state.networking.outputs.subnet_name
  
  # Use shared security group
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = data.terraform_remote_state.networking.outputs.management_cidr
      display_name = "Management Network"
    }
  }
}
```

This comprehensive framework for Terraform cloud resource management provides the theoretical foundations and practical strategies for building scalable, maintainable, and collaborative infrastructure as code for ML platforms. The key insight is that effective IaC requires careful consideration of state management, module design, multi-cloud strategies, and collaboration patterns to achieve reliable and efficient infrastructure operations.