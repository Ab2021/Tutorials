# Lab 07.3: Terraform Basics - AWS Resources

## Objective
Create real AWS infrastructure using Terraform, including EC2 instances, security groups, and S3 buckets.

## Prerequisites
- Terraform installed (Lab 07.2)
- AWS account (free tier)
- AWS CLI configured with credentials
- Basic understanding of AWS services

## Learning Objectives
- Write Terraform configuration for AWS resources
- Understand Terraform resource syntax
- Manage AWS credentials securely
- Apply and destroy AWS infrastructure

---

## Part 1: AWS Credentials Setup

### Configure AWS CLI

```bash
# Install AWS CLI (if not installed)
# Windows: choco install awscli
# macOS: brew install awscli
# Linux: sudo apt install awscli

# Configure credentials
aws configure
```

Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (`json`)

### Verify Configuration

```bash
aws sts get-caller-identity
```

---

## Part 2: Create EC2 Instance with Terraform

### Project Structure

```bash
mkdir terraform-aws-demo
cd terraform-aws-demo
```

### main.tf

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Security Group
resource "aws_security_group" "web_sg" {
  name        = "terraform-web-sg"
  description = "Allow HTTP and SSH"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "terraform-web-sg"
  }
}

# EC2 Instance
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2
  instance_type = "t2.micro"
  
  vpc_security_group_ids = [aws_security_group.web_sg.id]

  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y httpd
              systemctl start httpd
              systemctl enable httpd
              echo "<h1>Hello from Terraform!</h1>" > /var/www/html/index.html
              EOF

  tags = {
    Name = "terraform-web-server"
  }
}

# S3 Bucket
resource "aws_s3_bucket" "demo_bucket" {
  bucket = "my-terraform-demo-bucket-${random_id.bucket_id.hex}"

  tags = {
    Name = "terraform-demo-bucket"
  }
}

resource "random_id" "bucket_id" {
  byte_length = 4
}
```

### outputs.tf

```hcl
output "instance_public_ip" {
  description = "Public IP of EC2 instance"
  value       = aws_instance.web_server.public_ip
}

output "instance_id" {
  description = "ID of EC2 instance"
  value       = aws_instance.web_server.id
}

output "bucket_name" {
  description = "Name of S3 bucket"
  value       = aws_s3_bucket.demo_bucket.id
}
```

---

## Part 3: Deploy Infrastructure

### Initialize

```bash
terraform init
```

### Plan

```bash
terraform plan
```

Review the output - should show 4 resources to create.

### Apply

```bash
terraform apply
```

Type `yes` when prompted.

**Wait 2-3 minutes** for EC2 instance to boot.

### Verify

```bash
# Get public IP
terraform output instance_public_ip

# Test web server
curl http://$(terraform output -raw instance_public_ip)
```

Expected: `<h1>Hello from Terraform!</h1>`

---

## Part 4: Modify Infrastructure

### Update Instance Type

Edit `main.tf`:

```hcl
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.small"  # Changed from t2.micro
  # ... rest unchanged
}
```

### Apply Changes

```bash
terraform plan  # Shows instance will be replaced
terraform apply
```

**Note:** Changing instance type requires replacement (downtime).

---

## Part 5: Cleanup

### Destroy All Resources

```bash
terraform destroy
```

Type `yes` when prompted.

**Verify in AWS Console:**
- EC2 instance terminated
- Security group deleted
- S3 bucket deleted

---

## Challenges

### Challenge 1: Add Multiple Instances

Create 3 EC2 instances using `count`:

<details>
<summary>Solution</summary>

```hcl
resource "aws_instance" "web_servers" {
  count         = 3
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {
    Name = "web-server-${count.index + 1}"
  }
}

output "instance_ips" {
  value = aws_instance.web_servers[*].public_ip
}
```
</details>

### Challenge 2: Use Variables

Create `variables.tf` for region and instance type.

---

## Success Criteria

✅ Successfully created AWS resources with Terraform  
✅ Accessed web server via public IP  
✅ Modified infrastructure and re-applied  
✅ Destroyed all resources cleanly  

---

## Key Learnings

- **Terraform manages real infrastructure** - Be careful with apply/destroy
- **State tracks resources** - Terraform knows what exists
- **Changes may require replacement** - Some updates cause downtime
- **Always destroy test resources** - Avoid unexpected AWS charges

---

## Troubleshooting

**Issue:** "Error: UnauthorizedOperation"  
**Solution:** Check AWS credentials and IAM permissions.

**Issue:** AMI not found  
**Solution:** Use AMI ID for your region. Check AWS Console.

---

## Next Steps

- **Lab 07.4:** Terraform variables and outputs
- **Module 8:** Configuration Management with Ansible

**Estimated Time:** 45 minutes  
**Difficulty:** Intermediate  
**Cost:** Free tier eligible (remember to destroy!)
