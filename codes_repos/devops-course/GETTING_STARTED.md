# Getting Started with DevOps Course

## 🎯 Welcome!

This guide will help you set up your development environment and prepare for the DevOps learning journey.

---

## 📋 Prerequisites

### Hardware Requirements
- **Minimum:** 8GB RAM, 50GB free disk space
- **Recommended:** 16GB RAM, 100GB free disk space, SSD
- **Processor:** 64-bit processor with virtualization support

### Software Requirements
- Modern operating system (Windows 10/11, macOS 10.15+, or Linux)
- Administrator/sudo access
- Stable internet connection

### Knowledge Prerequisites
- Basic command-line familiarity
- Understanding of basic programming concepts
- Familiarity with text editors

---

## 🛠️ Essential Tools Installation

### 1. Docker

Docker is essential for containerization throughout this course.

**Windows:**
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Run the installer
3. Enable WSL 2 backend when prompted
4. Restart your computer

**macOS:**
1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Drag Docker.app to Applications
3. Launch Docker from Applications

**Linux (Ubuntu/Debian):**
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**Verify Installation:**
```bash
docker --version
docker run hello-world
```

---

### 2. Git

Version control is fundamental to DevOps.

**Windows:**
1. Download [Git for Windows](https://git-scm.com/download/win)
2. Run installer with default options
3. Choose your preferred text editor

**macOS:**
```bash
# Using Homebrew
brew install git

# Or using Xcode Command Line Tools
xcode-select --install
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install git

# RHEL/CentOS/Fedora
sudo yum install git
```

**Configure Git:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

**Verify:**
```bash
git --version
```

---

### 3. AWS CLI

Essential for cloud operations.

**Windows:**
```powershell
# Download and run installer
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

**macOS:**
```bash
# Using Homebrew
brew install awscli

# Or using installer
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Configure AWS CLI:**
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-east-1 (or your preferred region)
# Default output format: json
```

**Verify:**
```bash
aws --version
```

---

### 4. Terraform

Infrastructure as Code tool.

**Windows (using Chocolatey):**
```powershell
choco install terraform
```

**macOS:**
```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

**Linux:**
```bash
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

**Verify:**
```bash
terraform --version
```

---

### 5. kubectl (Kubernetes CLI)

**Windows:**
```powershell
curl.exe -LO "https://dl.k8s.io/release/v1.28.0/bin/windows/amd64/kubectl.exe"
# Add to PATH
```

**macOS:**
```bash
brew install kubectl
```

**Linux:**
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

**Verify:**
```bash
kubectl version --client
```

---

### 6. Ansible

**Windows:**
Ansible doesn't run natively on Windows. Use WSL2:
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install ansible
```

**macOS:**
```bash
brew install ansible
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ansible

# RHEL/CentOS
sudo yum install ansible
```

**Verify:**
```bash
ansible --version
```

---

## 🎓 Optional but Recommended Tools

### Visual Studio Code
Excellent editor with DevOps extensions.

**Install:**
- Download from [code.visualstudio.com](https://code.visualstudio.com/)

**Recommended Extensions:**
- Docker
- Kubernetes
- Terraform
- YAML
- GitLens
- Remote - SSH

### Terminal Emulators

**Windows:**
- Windows Terminal (recommended)
- PowerShell 7+

**macOS:**
- iTerm2
- Default Terminal

**Linux:**
- Terminator
- Tilix

---

## 🔧 Environment Verification

Create a verification script to check all installations:

**verify-setup.sh (Linux/macOS):**
```bash
#!/bin/bash

echo "=== DevOps Environment Verification ==="
echo ""

echo "Docker:"
docker --version

echo ""
echo "Git:"
git --version

echo ""
echo "AWS CLI:"
aws --version

echo ""
echo "Terraform:"
terraform --version

echo ""
echo "kubectl:"
kubectl version --client

echo ""
echo "Ansible:"
ansible --version

echo ""
echo "=== Verification Complete ==="
```

**verify-setup.ps1 (Windows PowerShell):**
```powershell
Write-Host "=== DevOps Environment Verification ===" -ForegroundColor Green
Write-Host ""

Write-Host "Docker:" -ForegroundColor Yellow
docker --version

Write-Host ""
Write-Host "Git:" -ForegroundColor Yellow
git --version

Write-Host ""
Write-Host "AWS CLI:" -ForegroundColor Yellow
aws --version

Write-Host ""
Write-Host "Terraform:" -ForegroundColor Yellow
terraform --version

Write-Host ""
Write-Host "kubectl:" -ForegroundColor Yellow
kubectl version --client

Write-Host ""
Write-Host "=== Verification Complete ===" -ForegroundColor Green
```

---

## 🌐 Create Free Accounts

### AWS Free Tier
1. Visit [aws.amazon.com/free](https://aws.amazon.com/free/)
2. Create an account
3. **Important:** Set up billing alerts to avoid unexpected charges
4. Enable MFA for security

### GitHub
1. Visit [github.com](https://github.com/)
2. Create a free account
3. Set up SSH keys for authentication

### Docker Hub
1. Visit [hub.docker.com](https://hub.docker.com/)
2. Create a free account
3. Used for storing container images

---

## 📚 Learning Resources Setup

### Create a Learning Directory
```bash
mkdir -p ~/devops-learning
cd ~/devops-learning
```

### Clone Course Repository (if applicable)
```bash
git clone <course-repo-url>
cd devops-course
```

---

## 🚀 Next Steps

1. ✅ Verify all tools are installed correctly
2. ✅ Create necessary accounts
3. ✅ Set up your learning directory
4. ✅ Proceed to [Phase 1, Module 01: Introduction to DevOps](./phase1_beginner/module-01-introduction-devops/)

---

## 🆘 Troubleshooting

### Docker Issues
- **Windows:** Ensure WSL2 is enabled and updated
- **Linux:** Check if Docker daemon is running: `sudo systemctl status docker`
- **Permissions:** Ensure your user is in the docker group

### AWS CLI Issues
- Verify credentials: `aws sts get-caller-identity`
- Check configuration: `cat ~/.aws/config`

### General Issues
- Ensure PATH is set correctly for all tools
- Restart terminal after installations
- Check firewall/antivirus settings

---

## 💡 Tips for Success

1. **Practice Daily:** Dedicate at least 1-2 hours daily
2. **Take Notes:** Document your learning and commands
3. **Join Communities:** Engage with DevOps communities
4. **Build Projects:** Apply concepts to real scenarios
5. **Stay Updated:** DevOps tools evolve rapidly

---

**You're Ready to Begin!** 🎉

Start your journey with [Module 01: Introduction to DevOps](./phase1_beginner/module-01-introduction-devops/)
