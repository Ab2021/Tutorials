# Lab 1.2: DevOps Principles in Practice

## 🎯 Objective

Apply core DevOps principles—Continuous Integration, Continuous Delivery, Infrastructure as Code, and Monitoring—to real-world scenarios. Understand how these principles work together to enable rapid, reliable software delivery.

## 📋 Prerequisites

- Completed Lab 1.1 (DevOps Culture)
- Basic understanding of software development lifecycle
- Familiarity with version control concepts
- Access to a computer with internet connection

## 🧰 Required Tools

- **Git**: Version control
- **Text Editor**: VS Code, Sublime, or similar
- **Terminal/Command Line**: Bash, PowerShell, or similar
- **Web Browser**: For research and documentation
- **Optional**: GitHub/GitLab account for hands-on practice

## 📚 Background

### The Four Pillars of DevOps

DevOps principles provide the technical foundation for the cultural practices we learned in Lab 1.1. While culture is the "why," principles are the "how."

**The Four Core Principles:**

1. **Continuous Integration (CI)**
   - Integrate code frequently (multiple times per day)
   - Automated builds and tests
   - Fast feedback on code quality

2. **Continuous Delivery/Deployment (CD)**
   - Software is always in a releasable state
   - Automated deployment pipelines
   - Rapid, reliable releases

3. **Infrastructure as Code (IaC)**
   - Manage infrastructure through code
   - Version-controlled infrastructure
   - Reproducible environments

4. **Monitoring and Logging**
   - Observe system behavior in real-time
   - Data-driven decisions
   - Proactive issue detection

### Real-World Impact

Organizations implementing these principles see:
- **46x more frequent deployments**
- **440x faster lead time from commit to deploy**
- **170x faster mean time to recover from downtime**
- **5x lower change failure rate**

*(Source: Accelerate: State of DevOps Report)*

---

## 📖 Theory Review

### Principle 1: Continuous Integration (CI)

**Definition:**
Continuous Integration is the practice of merging all developer working copies to a shared mainline several times a day.

**Core Practices:**
```
Developer → Commit Code → Automated Build → Automated Tests → Feedback
    ↑                                                              ↓
    └──────────────────── Fix Issues ←─────────────────────────────┘
```

**Key Components:**

1. **Single Source Repository**
   - All code in version control
   - One source of truth
   - Example: Git repository

2. **Automated Build**
   - Every commit triggers a build
   - Build must be fast (<10 minutes)
   - Fail fast on errors

3. **Automated Testing**
   - Unit tests run on every build
   - Integration tests validate components
   - Fast feedback to developers

4. **Frequent Commits**
   - Developers commit daily (or more)
   - Small, incremental changes
   - Reduces integration conflicts

**Benefits:**
- ✅ Early detection of integration issues
- ✅ Reduced integration problems
- ✅ Faster feedback cycles
- ✅ Higher code quality
- ✅ Reduced risk

**Anti-Patterns:**
- ❌ Infrequent commits (weekly/monthly)
- ❌ Manual build processes
- ❌ No automated testing
- ❌ Long-running builds (hours)
- ❌ Ignoring build failures

---

### Principle 2: Continuous Delivery (CD)

**Definition:**
Continuous Delivery is the ability to get changes of all types—features, configuration, bug fixes, experiments—into production safely and quickly in a sustainable way.

**Deployment Pipeline:**
```
Code → Build → Unit Test → Integration Test → UAT → Staging → Production
 ↓       ↓        ↓             ↓              ↓       ↓          ↓
Auto   Auto     Auto          Auto          Manual  Auto      Manual
                                           (optional)         (optional)
```

**Key Practices:**

1. **Deployment Automation**
   - One-click deployments
   - Consistent across environments
   - Repeatable and reliable

2. **Configuration Management**
   - Environment-specific configs
   - Externalized configuration
   - Version-controlled

3. **Comprehensive Testing**
   - Unit tests (fast, isolated)
   - Integration tests (component interaction)
   - End-to-end tests (full workflow)
   - Performance tests (load, stress)

4. **Version Control Everything**
   - Application code
   - Configuration files
   - Infrastructure code
   - Database scripts
   - Documentation

**Continuous Delivery vs Continuous Deployment:**

| Aspect | Continuous Delivery | Continuous Deployment |
|--------|-------------------|---------------------|
| **Definition** | CAN deploy anytime | DOES deploy automatically |
| **Manual Step** | Manual approval to production | Fully automated |
| **Risk** | Lower (human gate) | Higher (full automation) |
| **Speed** | Fast | Fastest |
| **Best For** | Regulated industries | Fast-moving products |

**Benefits:**
- ✅ Faster time to market
- ✅ Lower deployment risk
- ✅ Higher quality releases
- ✅ Better customer feedback
- ✅ Improved team morale

---

### Principle 3: Infrastructure as Code (IaC)

**Definition:**
Infrastructure as Code is the practice of managing and provisioning infrastructure through machine-readable definition files, rather than physical hardware configuration or interactive configuration tools.

**Traditional vs IaC:**

**Traditional Approach:**
```
1. Log into server
2. Manually install packages
3. Manually configure settings
4. Document steps (maybe)
5. Repeat for each server
```

**Problems:**
- ❌ Manual errors
- ❌ Configuration drift
- ❌ Not reproducible
- ❌ Slow provisioning
- ❌ Poor documentation

**IaC Approach:**
```
1. Write infrastructure code
2. Version control the code
3. Review changes (pull request)
4. Apply code (automated)
5. Infrastructure created consistently
```

**Benefits:**
- ✅ Version controlled
- ✅ Reproducible
- ✅ Self-documenting
- ✅ Fast provisioning
- ✅ Consistent environments

**IaC Tools:**

| Tool | Type | Best For |
|------|------|----------|
| **Terraform** | Declarative | Multi-cloud infrastructure |
| **CloudFormation** | Declarative | AWS-specific infrastructure |
| **Ansible** | Imperative | Configuration management |
| **Pulumi** | Declarative | Programming language IaC |

**Example IaC (Terraform):**
```hcl
# Define infrastructure as code
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {
    Name = "WebServer"
    Environment = "Production"
  }
}
```

---

### Principle 4: Monitoring and Logging

**Definition:**
Continuous observation of systems and applications to understand their behavior, detect issues, and make data-driven decisions.

**The Three Pillars of Observability:**

1. **Metrics** (What is happening?)
   - Quantitative measurements
   - Time-series data
   - Examples: CPU usage, request rate, error rate

2. **Logs** (Why is it happening?)
   - Event records
   - Detailed context
   - Examples: Application logs, system logs, audit logs

3. **Traces** (Where is it happening?)
   - Request flow through system
   - Distributed tracing
   - Examples: Request path, latency breakdown

**The Four Golden Signals:**

1. **Latency**
   - Time to serve a request
   - Distinguish between successful and failed requests
   - Track percentiles (p50, p95, p99)

2. **Traffic**
   - Demand on the system
   - Requests per second
   - User activity

3. **Errors**
   - Rate of failed requests
   - HTTP 5xx errors
   - Application exceptions

4. **Saturation**
   - How "full" the system is
   - CPU, memory, disk, network utilization
   - Queue depths

**Monitoring Stack:**
```
Application → Metrics Collection → Storage → Visualization → Alerting
                (Prometheus)       (TSDB)    (Grafana)     (AlertManager)
```

---

## 🔨 Hands-On Implementation

### Part 1: Continuous Integration Simulation

#### Step 1.1: Create a Simple CI Workflow

**Objective:** Understand CI principles through a simulated workflow

**Create Project Structure:**
```bash
# Create project directory
mkdir devops-ci-demo
cd devops-ci-demo

# Initialize Git repository
git init

# Create application file
cat > app.py << 'EOF'
def add(a, b):
    """Add two numbers"""
    return a + b

def subtract(a, b):
    """Subtract two numbers"""
    return a - b

def multiply(a, b):
    """Multiply two numbers"""
    return a * b

def divide(a, b):
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    print("Calculator App")
    print(f"2 + 3 = {add(2, 3)}")
    print(f"5 - 2 = {subtract(5, 2)}")
    print(f"4 * 3 = {multiply(4, 3)}")
    print(f"10 / 2 = {divide(10, 2)}")
EOF

# Create test file
cat > test_app.py << 'EOF'
import unittest
from app import add, subtract, multiply, divide

class TestCalculator(unittest.TestCase):
    
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
    
    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(0, 5), -5)
    
    def test_multiply(self):
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(0, 5), 0)
    
    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(9, 3), 3)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide(10, 0)

if __name__ == '__main__':
    unittest.main()
EOF

# Create build script
cat > build.sh << 'EOF'
#!/bin/bash
# Continuous Integration Build Script

echo "========================================="
echo "Starting CI Build Process"
echo "========================================="

# Step 1: Code Quality Check
echo ""
echo "Step 1: Running Code Quality Checks..."
python3 -m py_compile app.py
if [ $? -eq 0 ]; then
    echo "✅ Code quality check passed"
else
    echo "❌ Code quality check failed"
    exit 1
fi

# Step 2: Run Unit Tests
echo ""
echo "Step 2: Running Unit Tests..."
python3 -m unittest test_app.py
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi

# Step 3: Build Application
echo ""
echo "Step 3: Building Application..."
python3 app.py
if [ $? -eq 0 ]; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ CI Build Completed Successfully!"
echo "========================================="
EOF

chmod +x build.sh
```

**Run the CI Build:**
```bash
# Execute build script
./build.sh
```

**Expected Output:**
```
=========================================
Starting CI Build Process
=========================================

Step 1: Running Code Quality Checks...
✅ Code quality check passed

Step 2: Running Unit Tests...
.....
----------------------------------------------------------------------
Ran 5 tests in 0.001s

OK
✅ All tests passed

Step 3: Building Application...
Calculator App
2 + 3 = 5
5 - 2 = 3
4 * 3 = 12
10 / 2 = 5.0
✅ Build successful

=========================================
✅ CI Build Completed Successfully!
=========================================
```

**Commit to Git:**
```bash
# Add files
git add .

# Commit with meaningful message
git commit -m "Initial commit: Calculator app with tests and CI build"

# View commit history
git log --oneline
```

---

#### Step 1.2: Simulate CI Workflow with Branches

**Objective:** Practice feature branch workflow with CI

**Create Feature Branch:**
```bash
# Create and switch to feature branch
git checkout -b feature/add-power-function

# Add new function to app.py
cat >> app.py << 'EOF'

def power(base, exponent):
    """Raise base to the power of exponent"""
    return base ** exponent
EOF

# Add test for new function
cat >> test_app.py << 'EOF'

    def test_power(self):
        self.assertEqual(power(2, 3), 8)
        self.assertEqual(power(5, 2), 25)
        self.assertEqual(power(10, 0), 1)
EOF

# Run CI build to verify changes
./build.sh
```

**If Build Passes:**
```bash
# Commit changes
git add .
git commit -m "Add power function with tests"

# Merge to main branch
git checkout main
git merge feature/add-power-function

# Run CI build on main
./build.sh

# Tag release
git tag -a v1.1.0 -m "Release version 1.1.0 - Added power function"
```

**Document CI Workflow:**
```bash
# Create CI workflow documentation
cat > CI_WORKFLOW.md << 'EOF'
# Continuous Integration Workflow

## Workflow Steps

1. **Developer commits code to feature branch**
   ```bash
   git checkout -b feature/new-feature
   # Make changes
   git add .
   git commit -m "Add new feature"
   ```

2. **CI build runs automatically**
   ```bash
   ./build.sh
   ```
   - Code quality checks
   - Unit tests
   - Build verification

3. **If build passes:**
   - Merge to main branch
   - Tag release
   - Deploy to staging

4. **If build fails:**
   - Fix issues
   - Commit fix
   - CI runs again

## CI Principles Demonstrated

✅ **Frequent Integration**: Commit to main multiple times per day
✅ **Automated Build**: build.sh runs automatically
✅ **Automated Testing**: Unit tests run on every build
✅ **Fast Feedback**: Build completes in seconds
✅ **Fix Broken Builds Immediately**: Don't commit on top of failures

## Metrics

- **Build Time**: < 1 minute
- **Test Coverage**: 100%
- **Build Success Rate**: Target 95%+
- **Time to Fix Broken Build**: < 10 minutes
EOF
```

---

### Part 2: Continuous Delivery Pipeline Design

#### Step 2.1: Design a CD Pipeline

**Objective:** Create a comprehensive CD pipeline diagram

**Create Pipeline Documentation:**
```bash
cat > CD_PIPELINE.md << 'EOF'
# Continuous Delivery Pipeline

## Pipeline Stages

```
┌─────────────┐
│   Commit    │  Developer commits code
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Build     │  Compile code, create artifacts
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ Unit Tests  │  Fast, isolated tests
└──────┬──────┘
       │
       ↓
┌─────────────┐
│Integration  │  Component interaction tests
│   Tests     │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Security   │  Vulnerability scanning
│   Scan      │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Deploy    │  Deploy to staging
│  Staging    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Smoke     │  Basic functionality tests
│   Tests     │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    UAT      │  User acceptance testing
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Manual    │  Approval gate
│  Approval   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Deploy    │  Deploy to production
│ Production  │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Monitor    │  Observe production behavior
└─────────────┘
```

## Stage Details

### 1. Commit Stage (< 5 minutes)
**Purpose:** Fast feedback on code quality
**Activities:**
- Checkout code
- Compile/build
- Run unit tests
- Code quality checks (linting)
**Exit Criteria:** All tests pass, code compiles

### 2. Acceptance Stage (< 30 minutes)
**Purpose:** Validate functionality
**Activities:**
- Integration tests
- End-to-end tests
- Performance tests
**Exit Criteria:** All tests pass, performance acceptable

### 3. Security Stage (< 15 minutes)
**Purpose:** Identify vulnerabilities
**Activities:**
- Dependency scanning
- Static code analysis
- Container scanning
**Exit Criteria:** No critical vulnerabilities

### 4. Staging Deployment (< 10 minutes)
**Purpose:** Production-like environment testing
**Activities:**
- Deploy to staging
- Run smoke tests
- Validate deployment
**Exit Criteria:** Deployment successful, smoke tests pass

### 5. Production Deployment (< 10 minutes)
**Purpose:** Release to users
**Activities:**
- Deploy to production
- Run smoke tests
- Monitor metrics
**Exit Criteria:** Deployment successful, metrics healthy

## Pipeline Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Total Pipeline Time | < 60 minutes | - |
| Commit Stage Time | < 5 minutes | - |
| Test Coverage | > 80% | - |
| Deployment Success Rate | > 95% | - |
| Mean Time to Recovery | < 1 hour | - |

## Deployment Strategies

### Blue-Green Deployment
```
Blue (Current)     Green (New)
     ↓                  ↓
   Users  →  Switch  →  Users
```
- Zero downtime
- Easy rollback
- Requires double resources

### Canary Deployment
```
Production
    ├─ 95% → Old Version
    └─  5% → New Version (Canary)
```
- Gradual rollout
- Monitor canary metrics
- Rollback if issues detected

### Rolling Deployment
```
Server 1: Old → New
Server 2: Old → New
Server 3: Old → New
```
- One server at a time
- No downtime
- Gradual migration
EOF
```

---

#### Step 2.2: Create Deployment Checklist

**Create Deployment Runbook:**
```bash
cat > DEPLOYMENT_CHECKLIST.md << 'EOF'
# Deployment Checklist

## Pre-Deployment

### Code Quality
- [ ] All tests passing in CI
- [ ] Code review approved
- [ ] No critical security vulnerabilities
- [ ] Documentation updated
- [ ] Changelog updated

### Environment Preparation
- [ ] Staging environment matches production
- [ ] Database migrations tested
- [ ] Configuration files updated
- [ ] Secrets/credentials rotated if needed
- [ ] Backup of current production state

### Communication
- [ ] Deployment window scheduled
- [ ] Stakeholders notified
- [ ] On-call engineer assigned
- [ ] Rollback plan documented

## During Deployment

### Deployment Steps
- [ ] Put application in maintenance mode (if needed)
- [ ] Run database migrations
- [ ] Deploy new application version
- [ ] Run smoke tests
- [ ] Verify health checks
- [ ] Monitor error rates
- [ ] Check application logs

### Validation
- [ ] Key user flows working
- [ ] API endpoints responding
- [ ] Database connections healthy
- [ ] External integrations working
- [ ] Performance metrics acceptable

## Post-Deployment

### Monitoring
- [ ] Error rates normal
- [ ] Response times acceptable
- [ ] Resource utilization healthy
- [ ] No alerts triggered
- [ ] User feedback positive

### Documentation
- [ ] Deployment notes recorded
- [ ] Issues documented
- [ ] Metrics captured
- [ ] Lessons learned noted

### Communication
- [ ] Stakeholders notified of completion
- [ ] Team updated on any issues
- [ ] Documentation updated

## Rollback Criteria

Rollback immediately if:
- ❌ Error rate > 5%
- ❌ Response time > 2x baseline
- ❌ Critical functionality broken
- ❌ Data corruption detected
- ❌ Security vulnerability introduced

## Rollback Procedure

1. **Stop new deployments**
2. **Revert to previous version**
   ```bash
   # Example rollback command
   git revert HEAD
   ./deploy.sh --version previous
   ```
3. **Verify rollback successful**
4. **Monitor metrics**
5. **Notify stakeholders**
6. **Conduct post-mortem**
EOF
```

---

### Part 3: Infrastructure as Code Practice

#### Step 3.1: Write Infrastructure Code

**Objective:** Define infrastructure as code

**Create Terraform Configuration:**
```bash
# Create IaC directory
mkdir infrastructure
cd infrastructure

# Create main Terraform file
cat > main.tf << 'EOF'
# Infrastructure as Code Example
# This defines a simple web server infrastructure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region
}

# VPC for network isolation
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Public subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "${var.project_name}-public-subnet"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name        = "${var.project_name}-igw"
    Environment = var.environment
  }
}

# Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name        = "${var.project_name}-public-rt"
    Environment = var.environment
  }
}

# Route Table Association
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Security Group
resource "aws_security_group" "web" {
  name        = "${var.project_name}-web-sg"
  description = "Security group for web servers"
  vpc_id      = aws_vpc.main.id
  
  # Allow HTTP
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow HTTP traffic"
  }
  
  # Allow HTTPS
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow HTTPS traffic"
  }
  
  # Allow SSH (restrict in production!)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_ips
    description = "Allow SSH access"
  }
  
  # Allow all outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
  
  tags = {
    Name        = "${var.project_name}-web-sg"
    Environment = var.environment
  }
}

# EC2 Instance
resource "aws_instance" "web" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.web.id]
  
  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y httpd
              systemctl start httpd
              systemctl enable httpd
              echo "<h1>Hello from Terraform!</h1>" > /var/www/html/index.html
              EOF
  
  tags = {
    Name        = "${var.project_name}-web-server"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
EOF

# Create variables file
cat > variables.tf << 'EOF'
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "devops-demo"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
  default     = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2
}

variable "allowed_ssh_ips" {
  description = "IP addresses allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production!
}
EOF

# Create outputs file
cat > outputs.tf << 'EOF'
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "web_server_public_ip" {
  description = "Public IP of web server"
  value       = aws_instance.web.public_ip
}

output "web_server_url" {
  description = "URL to access web server"
  value       = "http://${aws_instance.web.public_ip}"
}
EOF

# Create README for IaC
cat > README.md << 'EOF'
# Infrastructure as Code - Web Server

This Terraform configuration creates a simple web server infrastructure on AWS.

## Resources Created

- VPC with CIDR 10.0.0.0/16
- Public subnet
- Internet Gateway
- Route Table
- Security Group (HTTP, HTTPS, SSH)
- EC2 instance running Apache web server

## Usage

### Initialize Terraform
```bash
terraform init
```

### Plan Infrastructure Changes
```bash
terraform plan
```

### Apply Infrastructure
```bash
terraform apply
```

### Destroy Infrastructure
```bash
terraform destroy
```

## Benefits of IaC

✅ **Version Controlled**: All changes tracked in Git
✅ **Reproducible**: Create identical environments
✅ **Self-Documenting**: Code describes infrastructure
✅ **Testable**: Validate before applying
✅ **Collaborative**: Team can review changes

## Best Practices

1. **Use variables** for environment-specific values
2. **Output important values** for reference
3. **Use remote state** for team collaboration
4. **Tag all resources** for organization
5. **Follow naming conventions** for clarity
EOF

cd ..
```

---

### Part 4: Monitoring and Logging Setup

#### Step 4.1: Create Monitoring Dashboard

**Objective:** Design a comprehensive monitoring strategy

**Create Monitoring Plan:**
```bash
cat > MONITORING_PLAN.md << 'EOF'
# Monitoring and Logging Strategy

## Monitoring Objectives

1. **Detect issues before users do**
2. **Understand system behavior**
3. **Make data-driven decisions**
4. **Optimize performance**
5. **Plan capacity**

## The Four Golden Signals

### 1. Latency
**What:** Time to serve a request
**Metrics:**
- Average response time
- p50, p95, p99 latency
- Slow query count

**Thresholds:**
- ✅ Good: < 100ms (p95)
- ⚠️  Warning: 100-500ms (p95)
- ❌ Critical: > 500ms (p95)

**Alerts:**
```yaml
alert: HighLatency
expr: http_request_duration_seconds{quantile="0.95"} > 0.5
for: 5m
annotations:
  summary: "High latency detected"
  description: "95th percentile latency is {{ $value }}s"
```

### 2. Traffic
**What:** Demand on the system
**Metrics:**
- Requests per second
- Active users
- Bandwidth usage

**Thresholds:**
- ✅ Normal: < 1000 req/s
- ⚠️  High: 1000-5000 req/s
- ❌ Overload: > 5000 req/s

**Alerts:**
```yaml
alert: HighTraffic
expr: rate(http_requests_total[5m]) > 5000
for: 2m
annotations:
  summary: "High traffic detected"
  description: "Request rate is {{ $value }} req/s"
```

### 3. Errors
**What:** Rate of failed requests
**Metrics:**
- HTTP 5xx error rate
- HTTP 4xx error rate
- Application exceptions

**Thresholds:**
- ✅ Good: < 0.1% error rate
- ⚠️  Warning: 0.1-1% error rate
- ❌ Critical: > 1% error rate

**Alerts:**
```yaml
alert: HighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
for: 2m
annotations:
  summary: "High error rate detected"
  description: "Error rate is {{ $value | humanizePercentage }}"
```

### 4. Saturation
**What:** How "full" the system is
**Metrics:**
- CPU utilization
- Memory utilization
- Disk usage
- Network bandwidth

**Thresholds:**
- ✅ Healthy: < 70%
- ⚠️  Warning: 70-85%
- ❌ Critical: > 85%

**Alerts:**
```yaml
alert: HighCPUUsage
expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
for: 5m
annotations:
  summary: "High CPU usage"
  description: "CPU usage is {{ $value }}%"
```

## Logging Strategy

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed diagnostic info | Variable values, function calls |
| **INFO** | General informational | Request received, task completed |
| **WARNING** | Warning messages | Deprecated API used, retry attempt |
| **ERROR** | Error events | Failed to connect, invalid input |
| **CRITICAL** | Critical failures | System crash, data corruption |

### Structured Logging

**Bad (Unstructured):**
```
User john logged in from 192.168.1.1 at 2025-11-22 10:30:45
```

**Good (Structured):**
```json
{
  "timestamp": "2025-11-22T10:30:45Z",
  "level": "INFO",
  "event": "user_login",
  "user": "john",
  "ip": "192.168.1.1",
  "user_agent": "Mozilla/5.0..."
}
```

### Log Aggregation

```
Application Logs → Log Shipper → Central Storage → Visualization
    (JSON)         (Fluentd)      (Elasticsearch)    (Kibana)
```

## Dashboard Design

### System Health Dashboard

**Metrics to Display:**
1. **Service Status**
   - Uptime percentage
   - Current status (up/down)
   - Last incident

2. **Performance**
   - Request rate (req/s)
   - Average latency (ms)
   - Error rate (%)

3. **Resources**
   - CPU usage (%)
   - Memory usage (%)
   - Disk usage (%)

4. **Business Metrics**
   - Active users
   - Transactions/hour
   - Revenue/hour

### Alert Dashboard

**Information to Show:**
1. Active alerts (critical, warning)
2. Alert history (last 24 hours)
3. Mean time to acknowledge (MTTA)
4. Mean time to resolve (MTTR)

## Monitoring Tools

| Tool | Purpose | Best For |
|------|---------|----------|
| **Prometheus** | Metrics collection | Time-series data |
| **Grafana** | Visualization | Dashboards |
| **ELK Stack** | Log management | Centralized logging |
| **Datadog** | Full-stack monitoring | All-in-one solution |
| **New Relic** | APM | Application performance |

## SLOs and SLIs

### Service Level Indicators (SLIs)
- Availability: 99.9% uptime
- Latency: 95% of requests < 200ms
- Error rate: < 0.1%

### Service Level Objectives (SLOs)
- **Availability SLO**: 99.9% (43.2 minutes downtime/month)
- **Latency SLO**: p95 < 200ms
- **Error Budget**: 0.1% (allows for innovation)

### Monitoring SLOs

```prometheus
# Availability
up{job="web-server"} == 1

# Latency
histogram_quantile(0.95, http_request_duration_seconds_bucket) < 0.2

# Error Rate
rate(http_requests_total{status=~"5.."}[5m]) < 0.001
```
EOF
```

---

## 🎯 Challenges

### Challenge 1: Build a Complete CI/CD Pipeline (Difficulty: ⭐⭐⭐)

**Scenario:**
Create a complete CI/CD pipeline for a simple web application that includes build, test, security scan, and deployment stages.

**Requirements:**
1. Create a sample web application
2. Write automated tests
3. Create a build script
4. Add security scanning
5. Create deployment automation
6. Document the entire pipeline

**Deliverables:**
- Application code
- Test suite
- CI/CD scripts
- Pipeline documentation

---

### Challenge 2: Infrastructure as Code Project (Difficulty: ⭐⭐⭐⭐)

**Scenario:**
Design and implement a complete infrastructure for a three-tier web application (web, application, database) using Infrastructure as Code.

**Requirements:**
1. Use Terraform or CloudFormation
2. Include networking (VPC, subnets, security groups)
3. Include compute (EC2 or containers)
4. Include database (RDS or similar)
5. Include load balancer
6. All resources properly tagged
7. Outputs for important values

**Deliverables:**
- Complete IaC code
- Variables for different environments
- README with usage instructions
- Architecture diagram

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 1 solution</summary>

### Challenge 1 Solution: Complete CI/CD Pipeline

[Solution provided in separate file due to length - includes complete working pipeline with all stages]

**Key Components:**
1. **Application**: Simple Node.js web app
2. **Tests**: Unit tests with Jest
3. **Build**: Docker containerization
4. **Security**: Dependency scanning with npm audit
5. **Deployment**: Automated deployment script
6. **Monitoring**: Health check endpoints

**Pipeline Stages:**
```
Commit → Build → Test → Security → Deploy Staging → Deploy Production
```

</details>

---

## ✅ Success Criteria

### Understanding
- [ ] Can explain CI/CD principles
- [ ] Understand IaC benefits
- [ ] Know the Four Golden Signals
- [ ] Understand deployment strategies

### Practical Skills
- [ ] Created CI build script
- [ ] Designed CD pipeline
- [ ] Wrote infrastructure code
- [ ] Created monitoring plan

### Application
- [ ] Can implement CI in a project
- [ ] Can design deployment pipelines
- [ ] Can write IaC for infrastructure
- [ ] Can set up monitoring

---

## 🚀 Next Steps

1. **Implement CI in your project**
2. **Design a deployment pipeline**
3. **Write infrastructure as code**
4. **Set up basic monitoring**
5. **Proceed to Lab 1.3: Understanding the DevOps Lifecycle**

---

**Apply DevOps Principles!** 🔄

*Remember: Principles without practice are just theory. Start implementing today!*
