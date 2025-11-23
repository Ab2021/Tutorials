# Lab 10.6: Security Groups and NACLs

## Objective
Configure network security using Security Groups and Network ACLs.

## Learning Objectives
- Create security groups
- Configure inbound/outbound rules
- Use Network ACLs
- Understand stateful vs stateless

---

## Security Group

```bash
# Create security group
aws ec2 create-security-group \
  --group-name web-sg \
  --description "Web server security group" \
  --vpc-id $VPC_ID

# Allow HTTP
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

# Allow SSH from specific IP
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr 203.0.113.0/24
```

## Network ACL

```bash
# Create NACL
aws ec2 create-network-acl \
  --vpc-id $VPC_ID

# Allow inbound HTTP
aws ec2 create-network-acl-entry \
  --network-acl-id $NACL_ID \
  --ingress \
  --rule-number 100 \
  --protocol tcp \
  --port-range From=80,To=80 \
  --cidr-block 0.0.0.0/0 \
  --rule-action allow

# Allow outbound
aws ec2 create-network-acl-entry \
  --network-acl-id $NACL_ID \
  --egress \
  --rule-number 100 \
  --protocol -1 \
  --cidr-block 0.0.0.0/0 \
  --rule-action allow
```

## Success Criteria
✅ Security groups configured  
✅ NACLs working  
✅ Traffic filtered correctly  

**Time:** 40 min
