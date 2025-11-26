# Lab: Day 19 - Terraform Basics

## Goal
Learn the Terraform workflow (`init`, `plan`, `apply`) without needing an AWS account. We will use the `local` provider to manage files.

## Directory Structure
```
day19/
├── main.tf
└── README.md
```

## Step 1: Install Terraform
Download from [terraform.io](https://www.terraform.io/downloads).

## Step 2: The Code (`main.tf`)

```hcl
terraform {
  required_providers {
    local = {
      source = "hashicorp/local"
      version = "2.4.0"
    }
  }
}

provider "local" {}

# Resource 1: A simple file
resource "local_file" "pet" {
  filename = "${path.module}/pet.txt"
  content  = "We love pets!"
}

# Resource 2: A file with sensitive content
resource "local_sensitive_file" "secret" {
  filename = "${path.module}/secret.txt"
  content  = "super-secret-password"
}
```

## Step 3: The Workflow

1.  **Init**:
    ```bash
    terraform init
    ```
    *Action*: Downloads the `local` provider plugin.

2.  **Plan**:
    ```bash
    terraform plan
    ```
    *Output*: Shows it will create 2 files.

3.  **Apply**:
    ```bash
    terraform apply -auto-approve
    ```
    *Action*: Creates `pet.txt` and `secret.txt`. Check the folder!

4.  **Modify**:
    Change `content = "We love pets!"` to `content = "We love dogs!"` in `main.tf`.

5.  **Plan Again**:
    ```bash
    terraform plan
    ```
    *Output*: `~ update in-place`. It detects the drift.

6.  **Apply Again**:
    ```bash
    terraform apply
    ```
    *Action*: Updates the file.

7.  **Destroy**:
    ```bash
    terraform destroy
    ```
    *Action*: Deletes the files.

## Challenge
Look at the `terraform.tfstate` file (it's JSON). Find the "secret" content.
*   **Lesson**: Terraform state files store secrets in plain text! Never commit `terraform.tfstate` to Git.
