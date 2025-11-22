# Lab 26.1: Multi-Cloud Terraform

## 🎯 Objective

Don't put all your eggs in one basket. You will write a single Terraform configuration that deploys a Web Server to **AWS** and a Database to **Azure**.

## 📋 Prerequisites

-   Terraform installed.
-   AWS Account & Azure Account (Free Tier).
-   Azure CLI installed (`az login`).

## 📚 Background

### Why Multi-Cloud?
-   **Resilience**: If AWS US-East-1 goes down, Azure US-East might be up.
-   **Best of Breed**: Use Google for AI, AWS for Compute, Azure for Active Directory.
-   **Leverage**: Negotiate better prices.

---

## 🔨 Hands-On Implementation

### Part 1: The Providers 🌍

1.  **`main.tf`:**
    ```hcl
    terraform {
      required_providers {
        aws = { source = "hashicorp/aws" }
        azurerm = { source = "hashicorp/azurerm" }
      }
    }

    provider "aws" { region = "us-east-1" }
    provider "azurerm" { features {} }
    ```

### Part 2: AWS Resource (The App) 🅰️

1.  **Add to `main.tf`:**
    ```hcl
    resource "aws_instance" "app" {
      ami           = "ami-0aa2b7722dc1b5612"
      instance_type = "t2.micro"
      tags = { Name = "MultiCloud-App" }
    }
    ```

### Part 3: Azure Resource (The DB) 🔷

1.  **Add to `main.tf`:**
    ```hcl
    resource "azurerm_resource_group" "rg" {
      name     = "multicloud-rg"
      location = "East US"
    }

    resource "azurerm_storage_account" "db_backup" {
      name                     = "multicloudstorage${random_id.suffix.hex}"
      resource_group_name      = azurerm_resource_group.rg.name
      location                 = azurerm_resource_group.rg.location
      account_tier             = "Standard"
      account_replication_type = "LRS"
    }

    resource "random_id" "suffix" {
      byte_length = 4
    }
    ```

### Part 4: Deploy 🚀

1.  **Init & Apply:**
    ```bash
    terraform init
    terraform apply
    ```

2.  **Verify:**
    -   Check AWS Console: EC2 instance running.
    -   Check Azure Portal: Storage Account created.

---

## 🎯 Challenges

### Challenge 1: VPN Peering (Difficulty: ⭐⭐⭐⭐⭐)

**Task:**
Conceptual.
How do you connect the AWS EC2 to the Azure Storage securely?
*Answer:* Site-to-Site VPN (AWS VPN Gateway <-> Azure VPN Gateway).
Write the Terraform to create the VPN Gateways.

### Challenge 2: Unified Output (Difficulty: ⭐)

**Task:**
Output the AWS Public IP and the Azure Storage Connection String in the same terminal.
*Goal:* One command to get all info.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```hcl
output "aws_ip" { value = aws_instance.app.public_ip }
output "azure_conn" { value = azurerm_storage_account.db_backup.primary_connection_string }
```
</details>

---

## 🔑 Key Takeaways

1.  **Complexity**: Multi-cloud doubles the complexity (IAM, Networking, Billing).
2.  **Latency**: Traffic between clouds goes over the public internet (slow/insecure) unless you use VPN/DirectConnect.
3.  **Abstraction**: Terraform abstracts the *syntax* (HCL), but not the *concepts* (AMI vs Image, VPC vs VNet).

---

## ⏭️ Next Steps

Terraform is great, but it's external to K8s. What if K8s could create the cloud resources?

Proceed to **Lab 26.2: Crossplane**.
