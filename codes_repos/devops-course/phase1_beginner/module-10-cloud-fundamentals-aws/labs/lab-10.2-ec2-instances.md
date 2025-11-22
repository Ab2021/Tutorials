# Lab 10.2: Launching EC2 Instances

## Objective
Launch, configure, and manage EC2 instances using both AWS Console and CLI.

## Prerequisites
- AWS account with IAM user (Lab 10.1)
- AWS CLI configured
- SSH key pair knowledge

## Learning Objectives
- Launch EC2 instances via Console and CLI
- Connect to instances via SSH
- Understand instance types and pricing
- Manage instance lifecycle (start/stop/terminate)

---

## Part 1: Launch EC2 via Console

### Step 1: Navigate to EC2

1. Sign in to AWS Console
2. Search for "EC2" in services
3. Click "Launch Instance"

### Step 2: Configure Instance

**Name:** `my-first-ec2`

**AMI:** Amazon Linux 2023 (Free tier eligible)

**Instance Type:** `t2.micro` (1 vCPU, 1GB RAM)

**Key Pair:**
- Click "Create new key pair"
- Name: `my-ec2-key`
- Type: RSA
- Format: `.pem` (Linux/Mac) or `.ppk` (Windows/PuTTY)
- **Download and save securely!**

**Network Settings:**
- Create security group
- Allow SSH (port 22) from "My IP"
- Allow HTTP (port 80) from "Anywhere"

**Storage:** 8 GB gp3 (default)

**Advanced Details:**
- User data (paste this):

```bash
#!/bin/bash
yum update -y
yum install -y httpd
systemctl start httpd
systemctl enable httpd
echo "<h1>Hello from $(hostname -f)</h1>" > /var/www/html/index.html
```

### Step 3: Launch

Click "Launch instance"

Wait 2-3 minutes for instance to reach "Running" state.

---

## Part 2: Connect to Instance

### Get Public IP

1. Select your instance
2. Copy "Public IPv4 address"

### SSH Connection

**Linux/Mac:**
```bash
chmod 400 my-ec2-key.pem
ssh -i my-ec2-key.pem ec2-user@<PUBLIC_IP>
```

**Windows (PowerShell):**
```powershell
ssh -i my-ec2-key.pem ec2-user@<PUBLIC_IP>
```

### Verify Web Server

```bash
# Inside EC2 instance
curl localhost

# From your laptop
curl http://<PUBLIC_IP>
```

Expected: `<h1>Hello from ...</h1>`

---

## Part 3: Launch EC2 via CLI

### Create Key Pair

```bash
aws ec2 create-key-pair \
  --key-name cli-key \
  --query 'KeyMaterial' \
  --output text > cli-key.pem

chmod 400 cli-key.pem
```

### Create Security Group

```bash
# Create group
aws ec2 create-security-group \
  --group-name cli-sg \
  --description "Security group for CLI instance"

# Get group ID
SG_ID=$(aws ec2 describe-security-groups \
  --group-names cli-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow SSH
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Allow HTTP
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0
```

### Launch Instance

```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t2.micro \
  --key-name cli-key \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cli-instance}]'
```

### Get Instance Details

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cli-instance" \
  --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress,State.Name]' \
  --output table
```

---

## Part 4: Instance Management

### Stop Instance

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cli-instance" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Stop
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

**Note:** Stopped instances don't incur compute charges, only storage.

### Start Instance

```bash
aws ec2 start-instances --instance-ids $INSTANCE_ID
```

### Terminate Instance

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

**Warning:** Termination is permanent!

---

## Part 5: Instance Types

### Common Types

| Type | vCPUs | RAM | Use Case | Cost/hour |
|------|-------|-----|----------|-----------|
| t2.micro | 1 | 1GB | Testing, low traffic | $0.0116 |
| t2.small | 1 | 2GB | Small apps | $0.023 |
| t2.medium | 2 | 4GB | Medium apps | $0.0464 |
| c5.large | 2 | 4GB | Compute-intensive | $0.085 |
| r5.large | 2 | 16GB | Memory-intensive | $0.126 |

### Check Pricing

```bash
aws pricing get-products \
  --service-code AmazonEC2 \
  --filters "Type=TERM_MATCH,Field=instanceType,Value=t2.micro" \
  --region us-east-1
```

---

## Challenges

### Challenge 1: Launch Instance in Specific Subnet

Research how to launch an instance in a specific VPC subnet.

### Challenge 2: Attach Additional EBS Volume

Create and attach a 10GB EBS volume to your instance.

<details>
<summary>Solution</summary>

```bash
# Create volume
VOLUME_ID=$(aws ec2 create-volume \
  --availability-zone us-east-1a \
  --size 10 \
  --query 'VolumeId' \
  --output text)

# Attach to instance
aws ec2 attach-volume \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf
```
</details>

---

## Success Criteria

✅ Launched EC2 instance via Console  
✅ Connected via SSH  
✅ Web server accessible via public IP  
✅ Launched instance via CLI  
✅ Managed instance lifecycle (stop/start/terminate)  

---

## Key Learnings

- **t2.micro is free tier eligible** - Use for learning
- **Always terminate test instances** - Avoid unnecessary charges
- **Security groups are stateful** - Return traffic automatically allowed
- **User data runs once** - On first boot only

---

## Cleanup

```bash
# Terminate all instances
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances --query 'Reservations[].Instances[].InstanceId' --output text)

# Delete security group (after instances terminated)
aws ec2 delete-security-group --group-id $SG_ID

# Delete key pair
aws ec2 delete-key-pair --key-name cli-key
```

---

## Next Steps

- **Lab 10.3:** S3 storage and bucket management

**Estimated Time:** 45 minutes  
**Difficulty:** Beginner
