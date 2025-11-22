# Lab 07.1: Infrastructure as Code Concepts

## Objective
Understand the fundamental principles of Infrastructure as Code (IaC) and why it's essential for modern DevOps practices.

## Prerequisites
- Basic understanding of cloud computing
- Familiarity with command-line interfaces
- Completed Module 6 (CI/CD Basics)

## Learning Objectives
- Understand what Infrastructure as Code means
- Differentiate between declarative and imperative approaches
- Recognize the benefits of IaC over manual infrastructure management
- Identify common IaC tools and their use cases

---

## Part 1: What is Infrastructure as Code?

### Definition

**Infrastructure as Code (IaC)** is the practice of managing and provisioning infrastructure through machine-readable definition files, rather than physical hardware configuration or interactive configuration tools.

### The Problem: Manual Infrastructure Management

**Traditional Approach:**
```
1. Log into AWS Console
2. Click "Launch Instance"
3. Select AMI, instance type, network settings
4. Configure security groups manually
5. Add tags
6. Launch
7. Repeat for each environment (Dev, Staging, Prod)
```

**Problems:**
- ❌ Time-consuming and error-prone
- ❌ No version control
- ❌ Difficult to replicate
- ❌ No audit trail
- ❌ "Snowflake servers" - each one is unique

### The Solution: Infrastructure as Code

**IaC Approach:**
```hcl
# server.tf
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {
    Name = "WebServer"
    Environment = "Production"
  }
}
```

**Benefits:**
- ✅ Version controlled (Git)
- ✅ Repeatable and consistent
- ✅ Self-documenting
- ✅ Automated
- ✅ Testable

---

## Part 2: Declarative vs. Imperative

### Imperative Approach

**You tell the system HOW to do something:**

```bash
# Bash script (Imperative)
aws ec2 run-instances --image-id ami-123 --instance-type t2.micro
aws ec2 create-tags --resources i-123 --tags Key=Name,Value=WebServer
aws ec2 create-security-group --group-name web-sg
aws ec2 authorize-security-group-ingress --group-id sg-123 --protocol tcp --port 80
```

**Characteristics:**
- Step-by-step instructions
- Order matters
- Hard to maintain state
- Difficult to handle changes

### Declarative Approach

**You tell the system WHAT you want:**

```hcl
# Terraform (Declarative)
resource "aws_instance" "web" {
  ami           = "ami-123"
  instance_type = "t2.micro"
  
  tags = {
    Name = "WebServer"
  }
}

resource "aws_security_group" "web_sg" {
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

**Characteristics:**
- Describes desired end state
- Tool figures out how to get there
- Idempotent (safe to run multiple times)
- Easier to understand and maintain

---

## Part 3: IaC Tools Comparison

### Popular IaC Tools

| Tool | Type | Best For | Language |
|------|------|----------|----------|
| **Terraform** | Declarative | Multi-cloud | HCL |
| **CloudFormation** | Declarative | AWS only | JSON/YAML |
| **Ansible** | Both | Configuration + Provisioning | YAML |
| **Pulumi** | Declarative | Developers who prefer code | Python/TypeScript/Go |
| **Chef/Puppet** | Declarative | Configuration management | Ruby/DSL |

### When to Use Each

**Terraform:**
- Multi-cloud deployments
- Complex infrastructure
- Team collaboration

**CloudFormation:**
- AWS-only infrastructure
- Deep AWS integration needed
- Free (no additional cost)

**Ansible:**
- Configuration management
- Simple provisioning
- Agentless approach preferred

**Pulumi:**
- Developers prefer real programming languages
- Need to leverage existing libraries
- Complex logic in infrastructure code

---

## Part 4: Hands-On Exercise - Design IaC for a Web Application

### Scenario

You need to deploy a 3-tier web application:
- **Web Tier:** 2 web servers (load balanced)
- **App Tier:** 3 application servers
- **Data Tier:** 1 database server (with backup)

### Exercise: Plan Your Infrastructure

On paper or in a text file, answer these questions:

1. **What resources do you need?**
   - List all infrastructure components

2. **How would you organize your IaC files?**
   - One big file or multiple files?
   - How to handle different environments?

3. **What should be parameterized?**
   - What values change between Dev/Staging/Prod?

4. **How would you handle secrets?**
   - Database passwords, API keys, etc.

**Example Answer:**

```
Resources Needed:
- VPC with public and private subnets
- Internet Gateway
- Load Balancer
- 2 EC2 instances (web servers)
- 3 EC2 instances (app servers)
- 1 RDS instance (database)
- Security Groups (web, app, db)
- S3 bucket (for backups)

File Organization:
├── main.tf           # Main configuration
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── network.tf        # VPC, subnets, etc.
├── compute.tf        # EC2 instances
├── database.tf       # RDS instance
└── environments/
    ├── dev.tfvars
    ├── staging.tfvars
    └── prod.tfvars

Parameters:
- instance_count (dev: 1, prod: 2)
- instance_type (dev: t2.micro, prod: t2.large)
- db_size (dev: db.t3.micro, prod: db.m5.large)
- environment_name

Secrets Management:
- Use AWS Secrets Manager
- Reference secrets in code, don't hardcode
- Rotate regularly
```

---

## Part 5: IaC Best Practices

### 1. Version Control Everything

```bash
git init
git add *.tf
git commit -m "Initial infrastructure"
git push origin main
```

### 2. Use Modules for Reusability

```hcl
# Don't repeat yourself
module "web_server" {
  source = "./modules/ec2"
  
  instance_type = "t2.micro"
  name          = "web-1"
}

module "app_server" {
  source = "./modules/ec2"
  
  instance_type = "t2.small"
  name          = "app-1"
}
```

### 3. Separate Environments

```
environments/
├── dev/
│   └── main.tf
├── staging/
│   └── main.tf
└── prod/
    └── main.tf
```

### 4. Use Remote State

```hcl
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}
```

### 5. Plan Before Apply

```bash
# Always review changes first
terraform plan

# Then apply
terraform apply
```

---

## Part 6: Benefits of IaC

### 1. Speed and Efficiency
- Provision infrastructure in minutes, not days
- Automate repetitive tasks

### 2. Consistency
- Same code = same infrastructure
- Eliminate configuration drift

### 3. Reduced Risk
- Test infrastructure changes before applying
- Easy rollback with version control

### 4. Documentation
- Code IS the documentation
- Always up-to-date

### 5. Cost Optimization
- Easily spin down dev/test environments
- Track resource usage in code

---

## Reflection Questions

1. **Why is idempotency important in IaC?**
   <details>
   <summary>Answer</summary>
   Idempotency means running the same code multiple times produces the same result. This is crucial because it makes infrastructure predictable and safe to re-apply without fear of creating duplicate resources or breaking existing ones.
   </details>

2. **When would you choose imperative over declarative IaC?**
   <details>
   <summary>Answer</summary>
   Imperative approaches can be useful for one-time migrations, complex conditional logic, or when you need fine-grained control over the exact sequence of operations. However, declarative is generally preferred for ongoing infrastructure management.
   </details>

3. **How does IaC support disaster recovery?**
   <details>
   <summary>Answer</summary>
   With IaC, your entire infrastructure is defined in code. If a data center fails, you can quickly recreate the entire infrastructure in a different region by running the same code. This is much faster than manual recreation.
   </details>

---

## Success Criteria

✅ You can explain IaC to a non-technical person  
✅ You understand the difference between declarative and imperative  
✅ You can list 3 benefits of IaC over manual management  
✅ You can identify which IaC tool to use for different scenarios  

---

## Key Learnings

- **IaC treats infrastructure like software** - Version controlled, tested, automated
- **Declarative is generally better than imperative** - Easier to maintain and understand
- **Choose tools based on your needs** - Multi-cloud vs. single cloud, team skills, etc.
- **IaC enables DevOps practices** - Fast, reliable, repeatable deployments

---

## Additional Resources

- [Infrastructure as Code (Book) by Kief Morris](https://www.oreilly.com/library/view/infrastructure-as-code/9781098114664/)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [AWS CloudFormation User Guide](https://docs.aws.amazon.com/cloudformation/)

---

## Next Steps

- **Lab 07.2:** Install and configure Terraform
- **Lab 07.3:** Write your first Terraform configuration

**Estimated Time:** 30-40 minutes  
**Difficulty:** Beginner  
**Type:** Conceptual + Exercise
