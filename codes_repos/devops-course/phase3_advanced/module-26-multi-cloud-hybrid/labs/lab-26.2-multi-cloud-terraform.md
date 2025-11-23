# Lab 26.2: Multi-Cloud Terraform

## Objective
Manage multi-cloud infrastructure with Terraform.

## Learning Objectives
- Configure multiple cloud providers
- Create cloud-agnostic modules
- Manage cross-cloud resources
- Implement disaster recovery

---

## Multi-Provider Configuration

```hcl
# providers.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

provider "azurerm" {
  features {}
}

provider "google" {
  project = "my-project"
  region  = "us-central1"
}
```

## Cloud-Agnostic Module

```hcl
# modules/compute/main.tf
variable "cloud_provider" {
  type = string
}

resource "aws_instance" "vm" {
  count         = var.cloud_provider == "aws" ? 1 : 0
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}

resource "azurerm_linux_virtual_machine" "vm" {
  count               = var.cloud_provider == "azure" ? 1 : 0
  name                = "myvm"
  resource_group_name = azurerm_resource_group.rg.name
  location            = "East US"
  size                = "Standard_B1s"
}

resource "google_compute_instance" "vm" {
  count        = var.cloud_provider == "gcp" ? 1 : 0
  name         = "myvm"
  machine_type = "e2-micro"
  zone         = "us-central1-a"
}
```

## Cross-Cloud Networking

```hcl
# VPN between AWS and Azure
resource "aws_vpn_gateway" "vpn" {
  vpc_id = aws_vpc.main.id
}

resource "azurerm_virtual_network_gateway" "vpn" {
  name                = "azure-vpn"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  type                = "Vpn"
  vpn_type            = "RouteBased"
}
```

## Success Criteria
✅ Multi-cloud providers configured  
✅ Resources deployed across clouds  
✅ Cross-cloud networking working  
✅ Disaster recovery tested  

**Time:** 50 min
