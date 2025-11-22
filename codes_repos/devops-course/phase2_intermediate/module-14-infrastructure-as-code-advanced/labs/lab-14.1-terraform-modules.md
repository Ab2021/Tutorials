# Lab 14.1: Terraform Modules

## 🎯 Objective

Stop Copy-Pasting Resource Blocks. **Modules** allow you to package a group of resources (e.g., "A Web Server with Security Group and IAM Role") and reuse it multiple times.

## 📋 Prerequisites

-   Terraform installed.
-   AWS CLI configured.

## 📚 Background

### Module Structure
```text
modules/
  webserver/
    main.tf       # The resources
    variables.tf  # Inputs (instance_type)
    outputs.tf    # Outputs (public_ip)
main.tf           # Root module calling the child module
```

---

## 🔨 Hands-On Implementation

### Part 1: Create the Module 📦

1.  **Create Directory:**
    ```bash
    mkdir -p modules/webserver
    ```

2.  **`modules/webserver/variables.tf`:**
    ```hcl
    variable "ami" { type = string }
    variable "size" { default = "t2.micro" }
    variable "name" { type = string }
    ```

3.  **`modules/webserver/main.tf`:**
    ```hcl
    resource "aws_instance" "web" {
      ami           = var.ami
      instance_type = var.size
      tags = {
        Name = var.name
      }
    }
    ```

4.  **`modules/webserver/outputs.tf`:**
    ```hcl
    output "instance_id" {
      value = aws_instance.web.id
    }
    ```

### Part 2: Consume the Module 🍽️

1.  **Create Root `main.tf`:**
    ```hcl
    provider "aws" { region = "us-east-1" }

    module "my_web_1" {
      source = "./modules/webserver"
      ami    = "ami-0aa2b7722dc1b5612" # Ubuntu 20.04
      name   = "Server-A"
    }

    module "my_web_2" {
      source = "./modules/webserver"
      ami    = "ami-0aa2b7722dc1b5612"
      name   = "Server-B"
      size   = "t2.small"
    }
    ```

2.  **Init & Apply:**
    ```bash
    terraform init
    terraform apply
    ```
    *Result:* Creates 2 instances with different configs using the same code.

### Part 3: Public Registry Modules 🌐

Don't reinvent the VPC. Use the official AWS module.

1.  **Update `main.tf`:**
    ```hcl
    module "vpc" {
      source = "terraform-aws-modules/vpc/aws"
      version = "3.14.0"

      name = "my-vpc"
      cidr = "10.0.0.0/16"

      azs             = ["us-east-1a", "us-east-1b"]
      private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
      public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

      enable_nat_gateway = false # Save money
    }
    ```

2.  **Init:**
    `terraform init` (Downloads the module).

---

## 🎯 Challenges

### Challenge 1: Output Passthrough (Difficulty: ⭐⭐)

**Task:**
In the root `main.tf`, output the ID of `Server-A`.
*Hint:* You need to access the module output: `value = module.my_web_1.instance_id`.

### Challenge 2: Conditional Creation (Difficulty: ⭐⭐⭐)

**Task:**
Add `count` to the module call.
Create 3 instances of `Server-A` using `count = 3`.
*Note:* Modules support `count` and `for_each` in newer Terraform versions.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Root `outputs.tf`:
```hcl
output "server_a_id" {
  value = module.my_web_1.instance_id
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Abstraction**: Modules hide complexity. The user just sets `size = "large"`, and the module handles the details.
2.  **Registry**: Always check the Terraform Registry before writing your own module.
3.  **Versioning**: Pin module versions (`version = "1.0.0"`) to avoid breaking changes.

---

## ⏭️ Next Steps

We are coding efficiently. Now let's collaborate safely.

Proceed to **Lab 14.2: Remote State & Locking**.
