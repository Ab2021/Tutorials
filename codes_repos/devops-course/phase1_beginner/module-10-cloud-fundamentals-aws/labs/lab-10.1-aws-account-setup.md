# Lab 10.1: AWS Account Setup and IAM Basics

## Objective
Set up an AWS account, configure IAM users, and understand AWS security best practices.

## Prerequisites
- Valid email address
- Credit/debit card (for verification)
- Phone number

## Learning Objectives
- Create and secure AWS root account
- Set up IAM users with appropriate permissions
- Enable MFA (Multi-Factor Authentication)
- Understand AWS Free Tier limits

---

## Part 1: Create AWS Account

### Step 1: Sign Up

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Enter email and account name
4. Choose "Personal" account type
5. Enter payment information (won't be charged if staying in Free Tier)
6. Verify phone number
7. Select "Basic Support - Free"

### Step 2: Secure Root Account

**CRITICAL:** Never use root account for daily tasks!

1. Sign in to AWS Console
2. Click account name → Security Credentials
3. Enable MFA:
   - Click "Activate MFA"
   - Choose "Virtual MFA device"
   - Use Google Authenticator or Authy app
   - Scan QR code
   - Enter two consecutive MFA codes

---

## Part 2: Create IAM Admin User

### Why IAM Users?

- Root account has unlimited access
- IAM users have limited, specific permissions
- Follows principle of least privilege

### Create Admin User

1. Go to IAM Console
2. Click "Users" → "Add users"
3. Username: `admin`
4. Select "Provide user access to AWS Management Console"
5. Choose "I want to create an IAM user"
6. Set custom password
7. Uncheck "Users must create new password at next sign-in"
8. Click "Next"

### Attach Permissions

1. Select "Attach policies directly"
2. Search and select: `AdministratorAccess`
3. Click "Next" → "Create user"

### Save Credentials

**Download CSV** with:
- Console sign-in URL
- Username
- Password

**Sign out of root account** and sign in as IAM admin user.

---

## Part 3: Create Developer User

### Create User with Limited Permissions

```bash
# Using AWS CLI (install first if needed)
aws iam create-user --user-name developer

# Attach EC2 read-only policy
aws iam attach-user-policy \
  --user-name developer \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess

# Create access keys
aws iam create-access-key --user-name developer
```

**Save the Access Key ID and Secret Access Key** - shown only once!

---

## Part 4: IAM Best Practices

### 1. Enable MFA for All Users

```bash
# Check MFA status
aws iam list-mfa-devices --user-name admin
```

### 2. Use Groups for Permissions

```bash
# Create group
aws iam create-group --group-name Developers

# Attach policy to group
aws iam attach-group-policy \
  --group-name Developers \
  --policy-arn arn:aws:iam::aws:policy/PowerUserAccess

# Add user to group
aws iam add-user-to-group \
  --user-name developer \
  --group-name Developers
```

### 3. Rotate Access Keys Regularly

```bash
# List access keys
aws iam list-access-keys --user-name developer

# Delete old key
aws iam delete-access-key \
  --user-name developer \
  --access-key-id AKIAIOSFODNN7EXAMPLE
```

---

## Part 5: Understanding AWS Free Tier

### Always Free Services

- **Lambda:** 1M requests/month
- **DynamoDB:** 25GB storage
- **CloudWatch:** 10 custom metrics

### 12-Month Free Tier

- **EC2:** 750 hours/month of t2.micro
- **S3:** 5GB storage
- **RDS:** 750 hours/month of db.t2.micro

### Monitor Usage

1. Go to Billing Dashboard
2. Enable "Receive Free Tier Usage Alerts"
3. Set up billing alarm:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name billing-alarm \
  --alarm-description "Alert when charges exceed $10" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold
```

---

## Challenges

### Challenge 1: Create Custom IAM Policy

Create a policy that allows S3 read-only access to a specific bucket:

<details>
<summary>Solution</summary>

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    }
  ]
}
```
</details>

---

## Success Criteria

✅ AWS account created and root account secured with MFA  
✅ IAM admin user created and can sign in  
✅ Developer user created with limited permissions  
✅ Billing alarm configured  

---

## Key Learnings

- **Never use root account** - Create IAM users instead
- **Enable MFA everywhere** - Adds critical security layer
- **Use groups for permissions** - Easier to manage than individual user policies
- **Monitor costs** - Set up billing alarms immediately

---

## Next Steps

- **Lab 10.2:** Launch your first EC2 instance
- **Lab 10.3:** Create and configure S3 buckets

**Estimated Time:** 30 minutes  
**Difficulty:** Beginner
