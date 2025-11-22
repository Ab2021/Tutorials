# Infrastructure as Code (IaC) with Terraform

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of IaC, including:
- **Principles**: Declarative vs. Imperative, Idempotency, and Immutability.
- **Terraform Core**: Providers, Resources, Variables, and Outputs.
- **Workflow**: Mastering the `init` -> `plan` -> `apply` cycle.
- **State Management**: Understanding how Terraform tracks infrastructure.
- **Best Practices**: Organizing code and managing secrets.

---

## 📖 Theoretical Concepts

### 1. What is Infrastructure as Code?

IaC is the process of managing and provisioning computer data centers through machine-readable definition files, rather than physical hardware configuration or interactive configuration tools.

**Key Principles:**
- **Declarative**: You define *what* you want (e.g., "I want 3 servers"), not *how* to get it. Terraform figures out the steps.
- **Idempotency**: Running the same code multiple times produces the same result.
- **Immutable Infrastructure**: Instead of patching servers, you replace them with new ones.

### 2. Terraform Concepts

- **HCL (HashiCorp Configuration Language)**: The syntax used by Terraform.
- **Provider**: A plugin that interacts with an API (e.g., AWS, Azure, Docker).
- **Resource**: A specific piece of infrastructure (e.g., `aws_instance`).
- **Data Source**: Read-only information fetched from the provider (e.g., "Get the ID of the latest Ubuntu AMI").
- **State File (`terraform.tfstate`)**: A JSON file that maps your code to real-world resources. **Never edit this manually.**

### 3. The Terraform Workflow

1.  **Write**: Create `.tf` files.
2.  **Init** (`terraform init`): Download providers and initialize the backend.
3.  **Plan** (`terraform plan`): Preview changes. Terraform compares Code vs State vs Real World.
4.  **Apply** (`terraform apply`): Execute changes.
5.  **Destroy** (`terraform destroy`): Remove all resources.

### 4. State Management

The State file is the "Brain" of Terraform.
- **Local State**: Stored on your laptop. Good for learning. Bad for teams.
- **Remote State**: Stored in a shared backend (e.g., AWS S3). Supports locking (DynamoDB) to prevent concurrent edits.

---

## 🔧 Practical Examples

### Basic EC2 Instance

```hcl
# Configure Provider
provider "aws" {
  region = "us-east-1"
}

# Define Resource
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "HelloWorld"
  }
}
```

### Using Variables

```hcl
variable "instance_type" {
  description = "Size of the VM"
  default     = "t2.micro"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = var.instance_type
}
```

### Outputs

```hcl
output "public_ip" {
  value = aws_instance.web.public_ip
  description = "The public IP of the web server"
}
```

---

## 🎯 Hands-on Labs

- [Lab 07.1: Iac Concepts](./labs/lab-07.1-iac-concepts.md)
- [Lab 7.1: Introduction to IaC & Terraform Setup](./labs/lab-07.1-intro-iac.md)
- [Lab 07.10: Infrastructure Versioning](./labs/lab-07.10-infrastructure-versioning.md)
- [Lab 7.2: Terraform Basics (Providers & Resources)](./labs/lab-07.2-terraform-basics.md)
- [Lab 07.2: Terraform Installation](./labs/lab-07.2-terraform-installation.md)
- [Lab 07.3: Terraform Basics](./labs/lab-07.3-terraform-basics.md)
- [Lab 7.3: Terraform State & Variables](./labs/lab-07.3-terraform-state.md)
- [Lab 7.4: Provisioning AWS EC2 with Terraform](./labs/lab-07.4-aws-ec2.md)
- [Lab 07.4: Cloudformation Intro](./labs/lab-07.4-cloudformation-intro.md)
- [Lab 07.5: Resource Creation](./labs/lab-07.5-resource-creation.md)
- [Lab 07.6: State Management](./labs/lab-07.6-state-management.md)
- [Lab 07.7: Terraform Variables](./labs/lab-07.7-terraform-variables.md)
- [Lab 07.8: Iac Comparison](./labs/lab-07.8-iac-comparison.md)
- [Lab 07.9: Terraform Plan Apply](./labs/lab-07.9-terraform-plan-apply.md)

---

## 📚 Additional Resources

### Official Documentation
- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [AWS Provider Registry](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

### Tutorials
- [HashiCorp Learn](https://developer.hashicorp.com/terraform/tutorials)

---

## 🔑 Key Takeaways

1.  **Treat Infrastructure as Code**: Version control it, review it, test it.
2.  **Plan First**: Always run `terraform plan` before applying.
3.  **Don't Touch the Console**: If you make changes manually in AWS Console, Terraform will detect "Drift" and might undo them.
4.  **Secure State**: Your state file contains sensitive data (passwords). Encrypt it.

---

## ⏭️ Next Steps

1.  Complete the labs to provision your first server.
2.  Proceed to **[Module 8: Configuration Management](../module-08-configuration-management/README.md)** to configure the server after it's provisioned.
