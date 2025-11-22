# Lab 07.2: Terraform Installation and Setup

## Objective
Install Terraform, configure your development environment, and verify the installation.

## Prerequisites
- Command-line access (Terminal/PowerShell)
- Administrator/sudo privileges
- Completed Lab 07.1 (IaC Concepts)

## Learning Objectives
- Install Terraform on your operating system
- Verify Terraform installation
- Understand Terraform CLI basics
- Configure Terraform for AWS (optional)

---

## Part 1: Install Terraform

### Option A: Windows (Using Chocolatey)

```powershell
# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Terraform
choco install terraform -y

# Verify installation
terraform version
```

### Option B: macOS (Using Homebrew)

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Terraform
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Verify installation
terraform version
```

### Option C: Linux (Ubuntu/Debian)

```bash
# Add HashiCorp GPG key
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

# Add HashiCorp repository
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list

# Update and install
sudo apt update
sudo apt install terraform -y

# Verify installation
terraform version
```

**Expected Output:**
```
Terraform v1.6.0
on linux_amd64
```

---

## Part 2: Terraform CLI Basics

### Essential Commands

```bash
# Initialize a Terraform working directory
terraform init

# Validate configuration files
terraform validate

# Format configuration files
terraform fmt

# Show execution plan
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy

# Show current state
terraform show

# List resources in state
terraform state list
```

### Get Help

```bash
# General help
terraform -help

# Command-specific help
terraform plan -help
terraform apply -help
```

---

## Part 3: Create Your First Terraform File

### Step 1: Create Project Directory

```bash
mkdir terraform-hello-world
cd terraform-hello-world
```

### Step 2: Create main.tf

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

# Create a local file
resource "local_file" "hello" {
  content  = "Hello, Terraform!"
  filename = "${path.module}/hello.txt"
}

# Output the file content
output "file_content" {
  value = local_file.hello.content
}
```

### Step 3: Initialize Terraform

```bash
terraform init
```

**Expected Output:**
```
Initializing the backend...
Initializing provider plugins...
- Finding hashicorp/local versions matching "~> 2.0"...
- Installing hashicorp/local v2.4.0...

Terraform has been successfully initialized!
```

### Step 4: Validate Configuration

```bash
terraform validate
```

**Expected Output:**
```
Success! The configuration is valid.
```

### Step 5: Preview Changes

```bash
terraform plan
```

**Expected Output:**
```
Terraform will perform the following actions:

  # local_file.hello will be created
  + resource "local_file" "hello" {
      + content              = "Hello, Terraform!"
      + filename             = "./hello.txt"
      + id                   = (known after apply)
    }

Plan: 1 to add, 0 to change, 0 to destroy.
```

### Step 6: Apply Configuration

```bash
terraform apply
```

Type `yes` when prompted.

**Expected Output:**
```
local_file.hello: Creating...
local_file.hello: Creation complete after 0s [id=...]

Apply complete! Resources: 1 added, 0 changed, 0 destroyed.

Outputs:

file_content = "Hello, Terraform!"
```

### Step 7: Verify File Created

```bash
cat hello.txt
```

**Output:**
```
Hello, Terraform!
```

---

## Part 4: Understanding Terraform State

### View State

```bash
terraform show
```

### List Resources

```bash
terraform state list
```

**Output:**
```
local_file.hello
```

### Inspect Specific Resource

```bash
terraform state show local_file.hello
```

---

## Part 5: Cleanup

### Destroy Resources

```bash
terraform destroy
```

Type `yes` when prompted.

**Expected Output:**
```
local_file.hello: Destroying... [id=...]
local_file.hello: Destruction complete after 0s

Destroy complete! Resources: 1 destroyed.
```

Verify file is deleted:
```bash
ls hello.txt  # Should show "file not found"
```

---

## Challenges

### Challenge 1: Create Multiple Files

Modify `main.tf` to create 3 files with different content.

<details>
<summary>Solution</summary>

```hcl
resource "local_file" "file1" {
  content  = "File 1"
  filename = "${path.module}/file1.txt"
}

resource "local_file" "file2" {
  content  = "File 2"
  filename = "${path.module}/file2.txt"
}

resource "local_file" "file3" {
  content  = "File 3"
  filename = "${path.module}/file3.txt"
}
```
</details>

### Challenge 2: Use Variables

Create a `variables.tf` file and parameterize the file content.

<details>
<summary>Solution</summary>

```hcl
# variables.tf
variable "message" {
  description = "Content for the file"
  type        = string
  default     = "Hello, Terraform!"
}

# main.tf
resource "local_file" "hello" {
  content  = var.message
  filename = "${path.module}/hello.txt"
}
```

Apply with custom value:
```bash
terraform apply -var="message=Custom message!"
```
</details>

---

## Success Criteria

✅ Terraform installed and version verified  
✅ Successfully ran `terraform init`  
✅ Created and applied first Terraform configuration  
✅ Understand basic Terraform workflow (init → plan → apply → destroy)  
✅ Can view and inspect Terraform state  

---

## Key Learnings

- **terraform init** downloads providers and prepares the working directory
- **terraform plan** shows what will change (dry run)
- **terraform apply** makes the changes
- **State file tracks infrastructure** - Never edit manually
- **Terraform is idempotent** - Safe to run apply multiple times

---

## Troubleshooting

### Issue: "terraform: command not found"

**Solution:** Add Terraform to PATH or restart terminal after installation.

### Issue: Permission denied

**Solution:** Run with sudo (Linux/macOS) or as Administrator (Windows).

### Issue: Provider download fails

**Solution:** Check internet connection. Try `terraform init -upgrade`.

---

## Next Steps

- **Lab 07.3:** Write Terraform configuration for AWS resources

**Estimated Time:** 30 minutes  
**Difficulty:** Beginner
