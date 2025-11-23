# Lab 26.1: Multi-Cloud Strategy

## Objective
Design and implement multi-cloud architecture.

## Learning Objectives
- Understand multi-cloud benefits
- Design cloud-agnostic architecture
- Implement abstraction layers
- Manage multiple providers

---

## Multi-Cloud Benefits

- **Avoid vendor lock-in**
- **Leverage best-of-breed services**
- **Geographic distribution**
- **Cost optimization**
- **Disaster recovery**

## Terraform Multi-Cloud

```hcl
# AWS resources
provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
}

# Azure resources
provider "azurerm" {
  features {}
}

resource "azurerm_storage_account" "backup" {
  name                = "backupstorage"
  resource_group_name = azurerm_resource_group.main.name
  location            = "eastus"
}

# GCP resources
provider "google" {
  project = "my-project"
  region  = "us-central1"
}

resource "google_storage_bucket" "archive" {
  name     = "archive-bucket"
  location = "US"
}
```

## Success Criteria
✅ Multi-cloud strategy defined  
✅ Resources deployed across clouds  
✅ Abstraction layer implemented  
✅ Costs compared  

**Time:** 45 min
