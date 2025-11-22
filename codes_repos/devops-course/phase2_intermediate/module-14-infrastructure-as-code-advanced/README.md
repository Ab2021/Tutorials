# Advanced Infrastructure as Code

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of advanced IaC patterns, including:
- **Modularity**: Writing reusable Terraform Modules to DRY (Don't Repeat Yourself) up your code.
- **Environments**: Managing Dev, Staging, and Prod using Workspaces and Directory layouts.
- **State Operations**: Importing existing infrastructure and fixing drift.
- **Testing**: Validating IaC with `tflint`, `checkov`, and `terratest`.
- **Polyglot IaC**: Introduction to **Pulumi** for defining infrastructure in Python/TypeScript.

---

## 📖 Theoretical Concepts

### 1. Terraform Modules

A module is a container for multiple resources that are used together.
- **Root Module**: The directory where you run `terraform apply`.
- **Child Module**: A module called by another module.
- **Structure**:
  - `main.tf`: Resources.
  - `variables.tf`: Inputs.
  - `outputs.tf`: Return values.

### 2. Managing Environments

Two main strategies:
1.  **Workspaces**: Same state file, different "workspace" prefix. Good for similar environments.
2.  **Directory Layout**: Separate folders (`env/dev`, `env/prod`) that call the same modules. Safer and more explicit.

### 3. Advanced State Operations

- **Import**: Bring existing AWS resources under Terraform control (`terraform import aws_s3_bucket.b bucket-name`).
- **Taint**: Mark a resource for recreation (`terraform taint aws_instance.web`).
- **State Move**: Refactor code without destroying resources (`terraform state mv`).

### 4. IaC Testing & Security

- **Static Analysis**: `tflint` (syntax), `checkov` (security/compliance).
- **Unit Testing**: `terratest` (Go library) deploys real infrastructure, validates it, and destroys it.
- **Policy as Code**: **Sentinel** or **OPA** (Open Policy Agent) to enforce rules (e.g., "No S3 buckets without encryption").

---

## 🔧 Practical Examples

### Creating a Module (`modules/webserver/main.tf`)

```hcl
resource "aws_instance" "this" {
  ami           = var.ami
  instance_type = var.size
  tags          = var.tags
}
```

### Consuming a Module (`main.tf`)

```hcl
module "web_server" {
  source = "./modules/webserver"

  ami  = "ami-123456"
  size = "t2.micro"
  tags = { Env = "Dev" }
}
```

### Terraform Workspaces

```bash
# Create new workspace
terraform workspace new dev

# Switch
terraform workspace select prod
```

### Pulumi Example (Python)

```python
import pulumi
import pulumi_aws as aws

bucket = aws.s3.Bucket("my-bucket")

pulumi.export("bucket_name", bucket.id)
```

---

## 🎯 Hands-on Labs

- [Lab 14.1: Terraform Modules](./labs/lab-14.1-terraform-modules.md)
- [Lab 14.2: Remote State & Locking](./labs/lab-14.2-remote-state.md)
- [Lab 14.3: Cloudformation Stacks](./labs/lab-14.3-cloudformation-stacks.md)
- [Lab 14.4: Pulumi Intro](./labs/lab-14.4-pulumi-intro.md)
- [Lab 14.5: State Locking](./labs/lab-14.5-state-locking.md)
- [Lab 14.6: Terraform Import](./labs/lab-14.6-terraform-import.md)
- [Lab 14.7: Cross Tool Comparison](./labs/lab-14.7-cross-tool-comparison.md)
- [Lab 14.8: Iac Testing](./labs/lab-14.8-iac-testing.md)
- [Lab 14.9: Terraform Cloud](./labs/lab-14.9-terraform-cloud.md)
- [Lab 14.10: Enterprise Patterns](./labs/lab-14.10-enterprise-patterns.md)

---

## 📚 Additional Resources

### Official Documentation
- [Terraform Modules](https://developer.hashicorp.com/terraform/language/modules)
- [Pulumi Documentation](https://www.pulumi.com/docs/)

### Tools
- [Checkov](https://www.checkov.io/) - Security scanning for IaC.
- [Terratest](https://terratest.gruntwork.io/) - Testing library.

---

## 🔑 Key Takeaways

1.  **Module Granularity**: Don't make modules too small (1 resource) or too big (entire VPC + App). Group logically related resources.
2.  **State Hygiene**: Use remote state with locking. Never commit `.tfstate` to Git.
3.  **Refactoring**: Use `terraform state mv` to move resources between modules without downtime.
4.  **Policy**: Catch security issues at build time, not deploy time.

---

## ⏭️ Next Steps

1.  Complete the labs to refactor your monolithic Terraform into modules.
2.  Proceed to **[Module 15: Advanced Configuration Management](../module-15-configuration-management-advanced/README.md)** to manage complex server configurations.
