# Lab 10.5: IAM Advanced

## Objective
Implement advanced IAM policies and best practices.

## Learning Objectives
- Create custom IAM policies
- Use policy conditions
- Implement least privilege
- Set up MFA and password policies

---

## Custom IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-bucket/*",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": "203.0.113.0/24"
        }
      }
    }
  ]
}
```

## Policy Conditions

```json
{
  "Condition": {
    "StringEquals": {
      "aws:RequestedRegion": "us-east-1"
    },
    "DateGreaterThan": {
      "aws:CurrentTime": "2024-01-01T00:00:00Z"
    },
    "Bool": {
      "aws:SecureTransport": "true"
    }
  }
}
```

## Assume Role Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## MFA Enforcement

```json
{
  "Condition": {
    "BoolIfExists": {
      "aws:MultiFactorAuthPresent": "true"
    }
  }
}
```

## Success Criteria
✅ Custom policies created  
✅ Conditions working  
✅ MFA enforced  
✅ Least privilege implemented  

**Time:** 40 min
