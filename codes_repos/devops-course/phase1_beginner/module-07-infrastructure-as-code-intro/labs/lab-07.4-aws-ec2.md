# Lab 7.4: Provisioning AWS EC2 with Terraform

## 🎯 Objective

The "Hello World" of Cloud Infrastructure. You will use Terraform to provision a virtual machine (EC2 Instance) on AWS.

## 📋 Prerequisites

-   AWS Account.
-   AWS CLI configured (Lab 7.1).
-   **Cost Warning**: EC2 `t2.micro` is Free Tier eligible. If you are outside Free Tier, terminate immediately after the lab.

## 📚 Background

### The AWS Provider
Terraform needs the `aws` provider to talk to Amazon.
Resources we need:
1.  `aws_instance`: The VM.
2.  `aws_security_group`: The Firewall (to allow SSH).
3.  `aws_key_pair`: The SSH Key.

---

## 🔨 Hands-On Implementation

### Part 1: Provider Setup ☁️

1.  **Create `main.tf`:**
    ```hcl
    terraform {
      required_providers {
        aws = {
          source  = "hashicorp/aws"
          version = "~> 4.0"
        }
      }
    }

    provider "aws" {
      region = "us-east-1"
    }
    ```

2.  **Init:**
    ```bash
    terraform init
    ```

### Part 2: The Resources 🖥️

Add to `main.tf`:

1.  **Find an AMI (Amazon Machine Image):**
    We'll use Ubuntu 20.04. (ID varies by region, check AWS Console or use Data Source).
    *Hardcoded for us-east-1:* `ami-0aa2b7722dc1b5612` (Ubuntu 20.04).

2.  **Define Instance:**
    ```hcl
    resource "aws_instance" "web" {
      ami           = "ami-0aa2b7722dc1b5612" # Ubuntu 20.04 us-east-1
      instance_type = "t2.micro"

      tags = {
        Name = "Terraform-Lab-Instance"
      }
    }
    ```

### Part 3: Apply 🚀

1.  **Plan:**
    ```bash
    terraform plan
    ```
    *Check:* It should plan to add 1 resource.

2.  **Apply:**
    ```bash
    terraform apply -auto-approve
    ```

3.  **Verify:**
    Go to AWS Console -> EC2. You should see "Terraform-Lab-Instance" running.

### Part 4: Cleanup (Crucial!) 🧹

1.  **Destroy:**
    ```bash
    terraform destroy -auto-approve
    ```
    *Result:* Instance is Terminated. No costs incurred.

---

## 🎯 Challenges

### Challenge 1: Data Sources (Difficulty: ⭐⭐⭐)

**Task:**
Hardcoding AMIs is bad (they change).
Use a `data` block to dynamically fetch the latest Ubuntu AMI ID.
*Hint:*
```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}
```
Update resource to use `ami = data.aws_ami.ubuntu.id`.

### Challenge 2: Security Group (Difficulty: ⭐⭐)

**Task:**
The instance currently has no firewall rules (default SG).
Create an `aws_security_group` that allows port 22 (SSH) and attach it to the instance (`vpc_security_group_ids`).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
(See Hint above).

**Challenge 2:**
```hcl
resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "web" {
  # ...
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Infrastructure as Code**: You just launched a server with code. You can launch 100 just by changing `count = 100`.
2.  **Cleanup**: Always `destroy` lab resources. AWS bills by the second.
3.  **Region**: AMIs are region-specific. An ID in `us-east-1` does not exist in `us-west-2`.

---

## ⏭️ Next Steps

We launched a server. Now let's configure it automatically.

Proceed to **Module 8: Configuration Management (Ansible)**.
