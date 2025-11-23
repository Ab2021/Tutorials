# Lab 10.4: VPC and Networking

## Objective
Create and configure AWS VPC with subnets, route tables, and gateways.

## Learning Objectives
- Create VPC and subnets
- Configure route tables
- Set up Internet Gateway
- Implement NAT Gateway

---

## Create VPC

```bash
# Create VPC
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=my-vpc}]'

VPC_ID=<vpc-id>
```

## Create Subnets

```bash
# Public subnet
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=public-subnet}]'

# Private subnet
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=private-subnet}]'
```

## Internet Gateway

```bash
# Create and attach IGW
aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=my-igw}]'

aws ec2 attach-internet-gateway \
  --vpc-id $VPC_ID \
  --internet-gateway-id $IGW_ID
```

## Route Tables

```bash
# Create route table
aws ec2 create-route-table \
  --vpc-id $VPC_ID

# Add route to IGW
aws ec2 create-route \
  --route-table-id $RTB_ID \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id $IGW_ID

# Associate with subnet
aws ec2 associate-route-table \
  --subnet-id $SUBNET_ID \
  --route-table-id $RTB_ID
```

## Success Criteria
✅ VPC created  
✅ Public/private subnets configured  
✅ Internet connectivity working  

**Time:** 45 min
