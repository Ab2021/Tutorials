# Cloud Fundamentals with AWS

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of AWS, including:
- **Core Concepts**: Regions, Availability Zones, and the Shared Responsibility Model.
- **Security**: Managing identity and access with **IAM**.
- **Compute**: Launching and managing virtual servers with **EC2**.
- **Storage**: Storing data reliably with **S3**.
- **Networking**: Building isolated networks with **VPC**.
- **Databases**: Using managed databases with **RDS**.

---

## 📖 Theoretical Concepts

### 1. Cloud Computing Basics

- **IaaS (Infrastructure as a Service)**: You rent the hardware (EC2). You manage OS and App.
- **PaaS (Platform as a Service)**: You rent the platform (Elastic Beanstalk). You manage App.
- **SaaS (Software as a Service)**: You use the software (Gmail). You manage nothing.

**Global Infrastructure:**
- **Region**: A physical location (e.g., `us-east-1` N. Virginia).
- **Availability Zone (AZ)**: One or more discrete data centers with redundant power/networking (e.g., `us-east-1a`).

### 2. IAM (Identity and Access Management)

The gatekeeper of AWS.
- **User**: A person or service (has credentials).
- **Group**: A collection of users.
- **Role**: A set of permissions that can be assumed by a User or Service (e.g., EC2 needs to access S3).
- **Policy**: A JSON document defining permissions (Allow/Deny).

**Best Practice**: Always follow the **Principle of Least Privilege**.

### 3. Core Services

#### EC2 (Elastic Compute Cloud)
Virtual servers.
- **Instance Type**: CPU/RAM combo (e.g., `t2.micro`).
- **AMI (Amazon Machine Image)**: The OS template (Ubuntu, Amazon Linux).
- **Security Group**: The firewall for the instance.

#### S3 (Simple Storage Service)
Object storage. Unlimited scale.
- **Bucket**: A container for objects (Must be globally unique).
- **Object**: File + Metadata.
- **Storage Classes**: Standard (Hot), Glacier (Cold/Archive).

#### VPC (Virtual Private Cloud)
Your private network in the cloud.
- **Subnet**: A segment of the VPC IP range (Public vs Private).
- **Internet Gateway**: Connects VPC to the internet.
- **Route Table**: Rules for traffic flow.

---

## 🔧 Practical Examples

### AWS CLI Commands

**1. S3 Operations**
```bash
# List buckets
aws s3 ls

# Upload file
aws s3 cp hello.txt s3://my-bucket/
```

**2. EC2 Operations**
```bash
# List instances
aws ec2 describe-instances

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

### IAM Policy Example (JSON)

Allow read-only access to S3.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:Get*",
                "s3:List*"
            ],
            "Resource": "*"
        }
    ]
}
```

---

## 🎯 Hands-on Labs

- [Lab 10.1: Aws Account Setup](./labs/lab-10.1-aws-account-setup.md)
- [Lab 10.1: Introduction to Cloud Computing & AWS Free Tier](./labs/lab-10.1-intro-cloud.md)
- [Lab 10.10: Aws Best Practices](./labs/lab-10.10-aws-best-practices.md)
- [Lab 10.2: Ec2 Instances](./labs/lab-10.2-ec2-instances.md)
- [Lab 10.2: IAM (Identity and Access Management)](./labs/lab-10.2-iam-basics.md)
- [Lab 10.3: EC2 Fundamentals](./labs/lab-10.3-ec2-fundamentals.md)
- [Lab 10.3: S3 Storage](./labs/lab-10.3-s3-storage.md)
- [Lab 10.4: S3 Storage (Simple Storage Service)](./labs/lab-10.4-s3-storage.md)
- [Lab 10.4: Vpc Networking](./labs/lab-10.4-vpc-networking.md)
- [Lab 10.5: Iam Basics](./labs/lab-10.5-iam-basics.md)
- [Lab 10.5: RDS (Relational Database Service)](./labs/lab-10.5-rds.md)
- [Lab 10.6: Cloud Capstone Project](./labs/lab-10.6-cloud-project.md)
- [Lab 10.6: Security Groups](./labs/lab-10.6-security-groups.md)
- [Lab 10.7: Aws Cli](./labs/lab-10.7-aws-cli.md)
- [Lab 10.8: Cloudwatch Basics](./labs/lab-10.8-cloudwatch-basics.md)
- [Lab 10.9: Load Balancers](./labs/lab-10.9-load-balancers.md)

---

## 📚 Additional Resources

### Official Documentation
- [AWS Documentation](https://docs.aws.amazon.com/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

### Certifications
- AWS Certified Cloud Practitioner (CLF-C01)
- AWS Certified Solutions Architect - Associate (SAA-C03)

---

## 🔑 Key Takeaways

1.  **Shared Responsibility**: AWS secures the cloud; you secure what's *in* the cloud.
2.  **Cost Awareness**: Turn off resources when not in use (especially in labs!).
3.  **Automation**: Use CLI or IaC (Terraform) instead of the Console for production.
4.  **Resilience**: Design for failure. Use multiple Availability Zones.

---

## ⏭️ Next Steps

1.  Complete the labs to build your first cloud infrastructure.
2.  **Congratulations!** You have completed Phase 1 (Beginner). Proceed to **Phase 2 (Intermediate)** to dive deeper into Advanced Docker and Kubernetes.
