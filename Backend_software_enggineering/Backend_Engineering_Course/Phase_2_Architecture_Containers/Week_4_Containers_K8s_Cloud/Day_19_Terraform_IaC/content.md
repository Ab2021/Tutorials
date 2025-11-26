# Day 19: Infrastructure as Code (IaC) with Terraform

## 1. ClickOps vs Code

*   **ClickOps**: Logging into AWS Console and clicking "Create EC2".
    *   *Problem*: Unrepeatable, prone to human error, no history.
*   **IaC**: Defining infrastructure in text files (HCL, YAML).
    *   *Benefit*: Version controlled, repeatable, automated.

### 1.1 Terraform
The industry standard for IaC. Cloud Agnostic (AWS, Azure, GCP, K8s).

---

## 2. Core Concepts

### 2.1 Provider
The plugin that talks to the API (e.g., AWS Provider).
```hcl
provider "aws" {
  region = "us-east-1"
}
```

### 2.2 Resource
A piece of infrastructure (EC2, S3 Bucket).
```hcl
resource "aws_s3_bucket" "my_bucket" {
  bucket = "my-unique-bucket-name-2025"
  
  tags = {
    Environment = "Dev"
  }
}
```

### 2.3 State (`terraform.tfstate`)
Terraform needs to know what it created. It stores the mapping between your code and real-world IDs in a JSON file.
*   **Local State**: Stored on your laptop (Bad for teams).
*   **Remote State**: Stored in S3 + DynamoDB (Locking). (Best Practice).

---

## 3. The Workflow

1.  **Init**: `terraform init` (Downloads providers).
2.  **Plan**: `terraform plan` (Dry run. Shows what *will* happen).
    *   *Output*: `+ create aws_s3_bucket.my_bucket`
3.  **Apply**: `terraform apply` (Executes the API calls).
4.  **Destroy**: `terraform destroy` (Deletes everything).

---

## 4. Modules

Don't copy-paste code. Wrap common patterns into reusable modules.
*   *Structure*:
    ```
    modules/
      vpc/
        main.tf
        variables.tf
        outputs.tf
    ```
*   *Usage*:
    ```hcl
    module "my_vpc" {
      source = "./modules/vpc"
      cidr   = "10.0.0.0/16"
    }
    ```

---

## 5. Summary

Today we automated the cloud.
*   **Declarative**: You say "I want 3 servers", Terraform figures out how to get there.
*   **State**: The source of truth.
*   **Plan**: Always check before you apply.

**Tomorrow (Day 20)**: We combine K8s and IaC into the ultimate workflow: **GitOps**. We will also explore Cloud Platform Services (Managed DBs, Serverless).
